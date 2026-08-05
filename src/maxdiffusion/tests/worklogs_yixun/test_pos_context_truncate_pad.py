"""exp_05 S1 — the 8-token positive-context construction layer (plan §3).

exp_05 conditions on ``C ∈ R^{8×4096}`` passed *directly* as the whole ``encoder_hidden_states`` --
not spliced into a 512-row T5 context the way exp_04's nulls are. Two parity properties make that
convention safe, and each has a test here:

1. **Construction parity.** ``truncate_or_pad_context`` must reproduce the reference's L_pos forcing
   (``third_party/Wan2.2/scripts/embedding_search.py:1181-1195``, submodule pin f370228) exactly:
   ``L > l_pos`` keeps the *leading* ``l_pos`` rows contiguously, ``L < l_pos`` appends zero rows in
   the input's own dtype, ``L == l_pos`` is a no-op. Pinned against hand-written expectations, not
   against a re-derivation of the implementation.
2. **Conditional-velocity parity (the deployment guarantee).** Feeding the frozen transformer an
   8-token ``C`` through the replay operator's ``v_cond`` seam must be the same computation the
   deployed ``wan_pre_context_adapter_forward`` performs when its head emits that same ``C``
   (``side_adapter_wan.py:737-774``). This is what makes a serialized target and the trained adapter
   realize the *same* conditioning, so the closed-loop gap measured at K4 is an adapter gap and not a
   representation gap.

Property 2 is characterized **at the seam**: these tests compare the deployed forward against a
direct transformer call, because S3's replay operator does not exist yet. S3 must re-run this
fixture against the real ``replay_with_positive`` (Codex S1 review, verification 1).

The velocity-parity tests below record what that investigation actually found: the two call idioms
are the *same call* (the deployed one only spells out ``deterministic``/``rngs`` at their defaults),
but the deployed path additionally casts the head output to the activation dtype
(``side_adapter_wan.py:767``). At the deployed bf16 activation dtype that cast is **not** a no-op --
so parity holds against ``C.astype(activation_dtype)`` and fails against fp32 ``C``. S3's replay
operator has to perform that cast.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bit_test_helpers import bf16_bits as _bf16_bits, bits as _bits, f32_bits as _f32_bits
from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L, pos_context_from_t5, truncate_or_pad_context


_DIM = 6
_T5_SEQ, _T5_DIM = 512, 4096
# Values a numeric ``==`` cannot separate (+0/-0), calls unequal to themselves (NaN), or that a lossy
# round-trip would flush (subnormal). Planted in the kept rows and past them.
_EDGE_VALUES = (0.0, -0.0, np.float32(np.inf), np.float32(-np.inf), np.float32(np.nan), np.float32(1e-45), -1.0, 1.0)


def _raw_bits(x):
    """Bit pattern in the array's own storage dtype, so bf16/fp16 are compared without an upcast."""
    x = jnp.asarray(x)
    return _bits(x, x.dtype, jnp.uint16 if x.dtype.itemsize == 2 else jnp.uint32)


def _context(seq, dim=_DIM, batch=None, seed=0, dtype=jnp.float32):
    shape = (seq, dim) if batch is None else (batch, seq, dim)
    return jax.random.normal(jax.random.PRNGKey(seed), shape, jnp.float32).astype(dtype)


def _edge_case_context(seq=16, dim=_DIM):
    base = np.asarray(_context(seq, dim)).copy()
    edges = np.asarray(_EDGE_VALUES, dtype=np.float32)[:dim]
    base[0, : len(edges)] = edges  # inside the kept range [0:POS_L]
    base[min(POS_L, seq) - 1, : len(edges)] = edges  # last kept row
    if seq > POS_L:
        base[POS_L, : len(edges)] = edges  # first dropped row
    return jnp.asarray(base)


# --------------------------------------------------------------------------------------------------
# 1. Construction parity — hand-written expectations for the reference's L_pos forcing.
# --------------------------------------------------------------------------------------------------

_ROWS_5 = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]


@pytest.mark.parametrize(
    "rows, l_pos, expected",
    [
        # L > l_pos: the LEADING rows, contiguously (``context_pos[:L_target]`` at :1185).
        (_ROWS_5, 3, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        (_ROWS_5, 1, [[1.0, 2.0]]),  # the L_pos = 1 end of the plan §4 ablation
        # L < l_pos: zero rows APPENDED (``torch.cat([context_pos, pad])`` at :1187-1191).
        (_ROWS_5[:2], 4, [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]),
        # L == l_pos: neither branch runs, the context is returned unchanged.
        (_ROWS_5[:3], 3, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    ],
)
def test_construction_matches_the_hand_written_reference_semantics(rows, l_pos, expected):
    """Ported semantics of ``embedding_search.py:1181-1195`` (pin f370228), pinned by hand."""
    out = truncate_or_pad_context(jnp.asarray(rows, jnp.float32), l_pos)

    assert out.shape == (l_pos, 2)
    # Bitwise, so the appended rows are pinned as +0.0 rather than merely "numerically zero".
    np.testing.assert_array_equal(_f32_bits(out), _f32_bits(jnp.asarray(expected, jnp.float32)))


def test_truncation_keeps_the_leading_rows_and_is_not_the_trailing_window():
    context = _edge_case_context(seq=23)  # a natural T5 length well past POS_L

    out = truncate_or_pad_context(context)

    assert out.shape == (POS_L, _DIM)
    np.testing.assert_array_equal(_f32_bits(out), _f32_bits(context[:POS_L]))
    assert not np.array_equal(_f32_bits(out), _f32_bits(context[-POS_L:]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_padding_appends_zero_rows_in_the_input_dtype(dtype):
    """``pad`` is built with ``dtype=context_pos.dtype`` (:1189) -- a promotion would change the
    representation the model is conditioned on, and zeros-vs-base-rows changes it outright."""
    context = _context(seq=5, dtype=dtype)

    out = truncate_or_pad_context(context)

    assert out.dtype == dtype and out.shape == (POS_L, _DIM)
    np.testing.assert_array_equal(_raw_bits(out[:5]), _raw_bits(context))
    np.testing.assert_array_equal(_raw_bits(out[5:]), _raw_bits(jnp.zeros((3, _DIM), dtype)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_equal_length_is_returned_bit_for_bit(dtype):
    """Neither reference branch runs, so nothing may touch the values -- including an fp32 upcast,
    which at bf16 would hand the model a context it can no longer round-trip."""
    context = _edge_case_context(seq=POS_L).astype(dtype)

    out = truncate_or_pad_context(context)

    assert out.dtype == dtype
    np.testing.assert_array_equal(_raw_bits(out), _raw_bits(context))


@pytest.mark.parametrize("seq", [23, 5, POS_L])
def test_the_t5_convention_leading_axis_is_squeezed(seq):
    """``[1, L, D]`` is *one* context (the T5 convention), not a batch of one."""
    context = _context(seq=seq)

    out = truncate_or_pad_context(context[None])

    assert out.shape == (POS_L, _DIM)
    np.testing.assert_array_equal(_f32_bits(out), _f32_bits(truncate_or_pad_context(context)))


@pytest.mark.parametrize("seq, kept", [(11, POS_L), (5, 5)])
def test_batched_contexts_are_forced_per_example(seq, kept):
    context = _context(seq=seq, batch=3, seed=4)

    out = truncate_or_pad_context(context)

    assert out.shape == (3, POS_L, _DIM)
    np.testing.assert_array_equal(_f32_bits(out[:, :kept]), _f32_bits(context[:, :kept]))
    np.testing.assert_array_equal(_f32_bits(out[:, kept:]), _f32_bits(jnp.zeros((3, POS_L - kept, _DIM))))


@pytest.mark.parametrize(
    "shape, l_pos, message",
    [
        ((8,), POS_L, "context must be"),  # rank-1
        ((2, 3, 8, 4), POS_L, "context must be"),  # rank-4
        ((0, 4), POS_L, "at least one row"),  # an empty context would pad to all zeros
        ((4, 0), POS_L, "at least one row"),  # a zero feature dim
        ((0, 12, 4), POS_L, "at least one example"),  # an empty batch axis, on the truncate path
        ((0, 5, 4), POS_L, "at least one example"),  # ... and on the pad path
        ((0, POS_L, 4), POS_L, "at least one example"),  # ... and on the equal-length path
        ((4, 4), 0, "l_pos must be >= 1"),
        ((4, 4), -1, "l_pos must be >= 1"),
        ((4, 4), 1.5, "l_pos must be an integer"),  # no silent truncation to 1
        ((4, 4), 8.0, "l_pos must be an integer"),
        ((4, 4), True, "l_pos must be an integer"),  # bool is an int subclass
    ],
)
def test_truncate_or_pad_rejects_malformed_arguments(shape, l_pos, message):
    with pytest.raises(ValueError, match=message):
        truncate_or_pad_context(jnp.zeros(shape, jnp.float32), l_pos)


# --------------------------------------------------------------------------------------------------
# 2. ``pos_context_from_t5`` — the warm start C_init (plan §3).
# --------------------------------------------------------------------------------------------------


def test_pos_context_from_t5_returns_the_deployed_eight_token_context():
    base = _context(seq=_T5_SEQ, dim=_T5_DIM, seed=7)

    out = pos_context_from_t5(base)

    assert out.shape == (POS_L, _T5_DIM)
    np.testing.assert_array_equal(_f32_bits(out), _f32_bits(base[:POS_L]))
    np.testing.assert_array_equal(_f32_bits(pos_context_from_t5(base[None])), _f32_bits(out))
    assert pos_context_from_t5(base, l_pos=1).shape == (1, _T5_DIM)  # the plan §4 ablation end


def test_pos_context_from_t5_is_bitwise_stable_at_the_bfloat16_boundary():
    """The model consumes bf16, so "truncation" must be a row copy -- not a value laundered through
    fp32 arithmetic that would land on a different bf16 code point (exp_04 R1's boundary rule)."""
    base = _edge_case_context(seq=_T5_SEQ)

    out = pos_context_from_t5(base)
    from_bf16 = pos_context_from_t5(base.astype(jnp.bfloat16))

    np.testing.assert_array_equal(_bf16_bits(out), _bf16_bits(base[:POS_L]))
    np.testing.assert_array_equal(_f32_bits(out), _f32_bits(base[:POS_L]))
    assert from_bf16.dtype == jnp.bfloat16
    np.testing.assert_array_equal(_raw_bits(from_bf16), _bf16_bits(base[:POS_L]))


@pytest.mark.parametrize(
    "shape, message",
    [
        ((3, 12, 4), "unit leading axis"),  # a batch of T5 contexts is not the convention
        ((12,), "base_context must be"),
        ((2, 2, 12, 4), "base_context must be"),
    ],
)
def test_pos_context_from_t5_rejects_non_t5_shapes(shape, message):
    with pytest.raises(ValueError, match=message):
        pos_context_from_t5(jnp.zeros(shape, jnp.float32))


# --------------------------------------------------------------------------------------------------
# 3. Conditional-velocity parity — the deployment guarantee (plan §3).
# --------------------------------------------------------------------------------------------------

# B > 1 with a DISTINCT context per example. At B = 1 -- or with one context broadcast over the batch
# -- collapsing, broadcasting or permuting the per-example contexts is invisible: the Codex S1
# review's ``broadcast(C[:1])`` mutant moved example 1 by 1.87 and still passed the old fixture.
_B, _C, _F, _H, _W = 2, 4, 2, 4, 6
_NUM_TRAIN_TIMESTEPS = 1000  # hardcoded, not imported (exp_04 R2's rule)
_LATENTS = jax.random.normal(jax.random.PRNGKey(0), (_B, _C, _F, _H, _W), jnp.float32)


class _ConstantContextHead:
    """The deployed adapter stack, reduced to the one thing parity is about: it emits ``C``.

    ``mutate`` inserts a synthetic defect into the deployed path only, so the parity assertions can
    be shown to have teeth. ``collapse`` and ``permute`` are the batch-identity defects a B = 1
    fixture cannot see; ``reproject`` is a value-changing linear map.
    """

    action_adapter_type = "pre_context"

    def __init__(self, context, mutate=None):
        self.context = context
        self.mutate = mutate

    def predict_pre_context(self, features, actions):
        del features, actions
        if self.mutate == "reproject":
            dim = self.context.shape[-1]
            matrix = jax.random.normal(jax.random.PRNGKey(21), (dim, dim), jnp.float32) / dim**0.5
            return self.context @ matrix
        if self.mutate == "collapse":  # every example conditioned on example 0's context
            return jnp.broadcast_to(self.context[:1], self.context.shape)
        if self.mutate == "permute":  # the right contexts, attached to the wrong examples
            return self.context[::-1]
        return self.context


def _timestep_2d(sigma, batch=_B, f_lat=_F, h_lat=_H, w_lat=_W):
    tokens_per_frame = (h_lat // 2) * (w_lat // 2)
    full = jnp.full((batch, f_lat * tokens_per_frame), sigma * _NUM_TRAIN_TIMESTEPS, jnp.float32)
    return full.at[:, :tokens_per_frame].set(0.0)  # frame-0 tokens are clean


def _paths(dtype, head_context, seam_context, mutate=None, examples=slice(None)):
    """Run the deployed pre_context forward and the replay operator's ``v_cond`` seam on one model.

    ``seam_context`` is the exp_04 velocity-seam idiom -- ``transformer(hidden_states=, timestep=,
    encoder_hidden_states=)``. ``head_context`` is what ``wan_pre_context_adapter_forward``'s head
    emits; the deployed path casts it to the activation dtype before its own final re-run
    (``side_adapter_wan.py:768-774``), which is the only argument-level difference between the two.

    ``examples`` slices the latents *and* both contexts together, so a sub-batch run conditions the
    same examples on the same contexts as the corresponding rows of a full-batch run.
    """
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxdiffusion.models.wan.side_adapter_wan import wan_action_adapter_forward
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    set_mesh = getattr(jax, "set_mesh", None)
    mesh_ctx = set_mesh(mesh) if set_mesh is not None else mesh
    z = _LATENTS[examples]
    head_context, seam_context = head_context[examples], seam_context[examples]
    timestep = _timestep_2d(0.7, batch=z.shape[0])

    with nn_partitioning.axis_rules(()), mesh_ctx:
        model = WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=_C,
            out_channels=_C,
            text_dim=head_context.shape[-1],
            freq_dim=16,
            ffn_dim=32,
            num_layers=2,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
            dtype=dtype,
            weights_dtype=dtype,
        )
        deployed = wan_action_adapter_forward(
            model,
            _ConstantContextHead(head_context, mutate=mutate),
            hidden_states=z,
            timestep=timestep,
            encoder_hidden_states=None,  # the pre_context path ignores it (`del encoder_hidden_states`)
            actions=None,
            deterministic=True,
            rngs=None,
        )
        seam = model(hidden_states=z, timestep=timestep, encoder_hidden_states=seam_context)
    return np.asarray(deployed), np.asarray(seam)


def _eight_token_contexts(dim=32):
    """One distinct C per example, built by the real constructor -- ``[_B, POS_L, dim]``."""
    contexts = truncate_or_pad_context(_context(seq=_T5_SEQ, dim=dim, batch=_B, seed=9))
    assert not np.array_equal(np.asarray(contexts[0]), np.asarray(contexts[1]))  # the fixture's point
    return contexts


def test_the_deployed_pre_context_forward_is_the_replay_operator_s_v_cond_call():
    """plan §3: the deployed forward with a head emitting C == the replay seam's ``v_cond`` at C.

    The idioms turn out to be the *same call*: ``wan_pre_context_adapter_forward``'s final re-run
    (``side_adapter_wan.py:768-774``) only spells out ``deterministic``/``rngs`` at the values the
    seam leaves defaulted, so equality is asserted bitwise rather than with a tolerance.

    **Scope, for S3's Coder (Codex S1 review, verification 1).** What this pins is the *seam*: the
    deployed forward against a direct transformer call. It does not and cannot cover the replay
    operator, which does not exist yet. S3 must re-run this fixture against the real
    ``replay_with_positive`` -- substituting its ``v_cond`` for the direct call here -- or the
    guarantee stops at the boundary of code S1 could see.
    """
    contexts = _eight_token_contexts()

    deployed, seam = _paths(jnp.float32, contexts, contexts)

    assert deployed.shape == (_B, _C, _F, _H, _W)
    assert float(np.std(seam)) > 1e-3, seam  # a degenerate output would make the equality vacuous
    np.testing.assert_array_equal(_f32_bits(deployed), _f32_bits(seam))


def test_each_example_is_conditioned_on_its_own_context():
    """Batch-collapse guard: example i of a batched run must equal a singleton run on context i.

    Bitwise across batch sizes, so the deployed path is pinned to route contexts per example rather
    than to any batch-shaped accident -- the property a B = 1 fixture structurally cannot state.
    """
    contexts = _eight_token_contexts()

    batched, _ = _paths(jnp.float32, contexts, contexts)

    for i in range(_B):
        single, _ = _paths(jnp.float32, contexts, contexts, examples=slice(i, i + 1))
        np.testing.assert_array_equal(_f32_bits(batched[i]), _f32_bits(single[0]), err_msg=f"example {i}")


def test_parity_at_the_deployed_activation_dtype_requires_the_head_s_cast():
    """The one argument-level difference: the deployed path casts C to the activation dtype
    (``side_adapter_wan.py:767``). At bf16 that cast is NOT a no-op, so S3's replay operator must
    perform it -- an fp32 C reaches the frozen transformer as a different conditioning."""
    contexts = _eight_token_contexts()

    deployed, seam_cast = _paths(jnp.bfloat16, contexts, contexts.astype(jnp.bfloat16))
    _, seam_fp32 = _paths(jnp.bfloat16, contexts, contexts)

    np.testing.assert_array_equal(_raw_bits(deployed), _raw_bits(seam_cast))
    assert not np.array_equal(_raw_bits(deployed), _raw_bits(seam_fp32))


@pytest.mark.parametrize("mutate", ["reproject", "collapse", "permute"])
def test_velocity_parity_detects_a_corrupted_deployed_context(mutate):
    """Guard the guard: each defect inserted into the deployed path only must break the assertions
    above. ``collapse`` and ``permute`` are the two the review's B = 1 fixture let through. The gap
    must be far larger than rounding, so no guard can pass on reassociation noise alone."""
    contexts = _eight_token_contexts()

    deployed, seam = _paths(jnp.float32, contexts, contexts, mutate=mutate)

    assert not np.array_equal(_f32_bits(deployed), _f32_bits(seam))
    assert float(np.max(np.abs(deployed - seam))) > 1e-3, (mutate, deployed, seam)
