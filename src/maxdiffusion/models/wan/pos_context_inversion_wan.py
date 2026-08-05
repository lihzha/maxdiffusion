"""Positive-context inversion helpers for the frozen WAN TI2V backbone (exp_05).

Round S1 is only the context-construction layer -- the optimizer (S2) and the replay operator (S3)
land later, and will import exp_04's ``null_inversion_wan`` for the shared primitives (noise keys,
sigma validation, fingerprints). Per plan §5/F6 exp_05 lives entirely in this module and **never
edits exp_04's settled ones**; S1 therefore imports nothing from ``null_inversion_wan`` yet.

**The 8-token convention (plan §3), which is the load-bearing design.** exp_04 optimizes the leading
rows of a 512-row T5 context and splices them back in, because the null branch must keep the padding
rows the frozen backbone was trained with. The positive slot is different: the deployed adapter
(``side_adapter_wan.wan_pre_context_adapter_forward``) passes its head's ``[B, 8, 4096]`` output as
the *entire* ``encoder_hidden_states``, with no 512-row restoration. So exp_05's conditional context
is a bare ``[8, 4096]`` -- and the warm start is the reference's L_pos forcing applied to T5(""),
not a 512-row context. Anything else would optimize a representation the trained adapter can never
emit, and the K4 closed-loop comparison would be measuring that mismatch instead of the adapter.

Ported from ``run_positive_inversion``'s L_pos block
(``third_party/Wan2.2/scripts/embedding_search.py:1181-1195``, submodule pin f370228).

**Deployment-parity note for S3.** The deployed forward's final re-run is the same call as the
velocity seam exp_04 uses (``transformer(hidden_states=, timestep=, encoder_hidden_states=)``; the
deployed site merely spells out ``deterministic``/``rngs`` at their defaults) -- with one difference:
it casts the head output to the activation dtype first (``side_adapter_wan.py:767``). At the deployed
bf16 activation dtype that cast is *not* a no-op, so a replay operator that hands the frozen
transformer an fp32 ``C`` conditions it differently from the adapter. Pinned in
``tests/worklogs_yixun/test_pos_context_truncate_pad.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# The deployed ``pre_context_tokens``: the adapter head emits exactly this many context tokens, so
# the inversion target must have exactly this many rows. Plan §3/§11 decision 1.
POS_L = 8


def truncate_or_pad_context(context, l_pos: int = POS_L) -> jax.Array:
    """Force a context to exactly ``l_pos`` rows, as the reference's L_pos block does (:1181-1195).

    ``L > l_pos`` keeps the **leading** ``l_pos`` rows contiguously; ``L < l_pos`` **appends** zero
    rows in the context's own dtype; ``L == l_pos`` returns it unchanged, bit for bit. No arithmetic
    touches the surviving rows, so the values the model is conditioned on are the input's own.

    Args:
      context: ``[L, D]``, ``[1, L, D]`` (the T5 convention -- *one* context, so the leading axis is
        squeezed and the result is rank-2), or ``[B, L, D]`` with ``B > 1``, forced per example.
        ``B = 0`` is rejected rather than passed through.
      l_pos: target row count; a positive Python/NumPy integer (``bool`` and ``float`` are rejected
        rather than coerced, so a wrong type cannot silently run a different recipe).

    Returns:
      ``[l_pos, D]``, or ``[B, l_pos, D]`` for a genuinely batched input. Dtype is the input's.
    """
    if isinstance(l_pos, bool) or not isinstance(l_pos, (int, np.integer)):
        raise ValueError(f"l_pos must be an integer, got {l_pos!r}")
    l_pos = int(l_pos)
    if l_pos < 1:
        raise ValueError(f"l_pos must be >= 1, got {l_pos}")

    context = jnp.asarray(context)
    if context.ndim == 3 and context.shape[0] == 1:
        context = context[0]
    if context.ndim not in (2, 3):
        raise ValueError(f"context must be [L, D] or [B, L, D], got shape {context.shape}")
    if context.ndim == 3 and context.shape[0] < 1:
        # An empty batch is not "a batch of contexts": every branch below would happily return a
        # zero-example array that only fails much later, at the velocity seam's shape check.
        raise ValueError(f"context must carry at least one example, got shape {context.shape}")
    length, dim = context.shape[-2:]
    if length < 1 or dim < 1:
        raise ValueError(f"context must carry at least one row and one feature, got shape {context.shape}")

    if length > l_pos:
        return context[..., :l_pos, :]
    if length < l_pos:
        pad = jnp.zeros((*context.shape[:-2], l_pos - length, dim), dtype=context.dtype)
        return jnp.concatenate([context, pad], axis=-2)
    return context


def pos_context_from_t5(base_context, l_pos: int = POS_L) -> jax.Array:
    """The warm start ``C_init``: T5("") forced to ``l_pos`` rows (plan §3).

    T5("") is the ``[512, 4096]`` (or ``[1, 512, 4096]``) context the frozen backbone's unconditional
    branch keeps using untouched; the conditional branch starts from its leading ``POS_L`` rows,
    which is what the reference does with T5(heuristic). ``l_pos`` is a parameter only so plan §4's
    diagnostic ``L_pos ∈ {1, 8}`` ablation can call this same constructor; K2/K3 use ``POS_L``.
    """
    base_context = jnp.asarray(base_context)
    if base_context.ndim == 3 and base_context.shape[0] != 1:
        raise ValueError(f"base_context must have a unit leading axis when rank-3, got {base_context.shape}")
    if base_context.ndim not in (2, 3):
        raise ValueError(f"base_context must be [S, D] or [1, S, D], got shape {base_context.shape}")
    return truncate_or_pad_context(base_context, l_pos)
