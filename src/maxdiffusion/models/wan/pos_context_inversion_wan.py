"""Positive-context inversion helpers for the frozen WAN TI2V backbone (exp_05).

S1 built the context-construction layer, S2 added ``optimize_positive_embeddings`` and S3
``replay_with_positive``; the inversion side of exp_05 is complete here. Shared primitives come from
exp_04's ``null_inversion_wan`` **by import** -- per plan §5/F6 exp_05 lives entirely in this module
and never edits exp_04's settled ones.

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

**THE CAST-SEAM RULE -- ONE OWNER, stated once for S2, S3 and the runner (S2 review, finding 2).**
The deployed forward's final re-run is the same call as the velocity seam exp_04 uses
(``transformer(hidden_states=, timestep=, encoder_hidden_states=)``; the deployed site merely spells
out ``deterministic``/``rngs`` at their defaults) -- with one difference: it casts the head output to
the activation dtype first (``side_adapter_wan.py:767``), and at the deployed bf16 activation dtype
that cast is *not* a no-op. So the cast has to happen, and exactly one component owns it::

    the optimizer (S2) and the replay operator (S3) pass the fp32 context through UNCHANGED;
    the runner-built real-backbone ``velocity_fn`` casts to the transformer's activation dtype
    immediately before the transformer call -- for BOTH the conditional and the unconditional branch.

That is the same placement deployment uses (inside the wrapper that turns latents + context into a
velocity) and the same one upstream uses (inside ``_dit_velocity``'s autocast,
``third_party/Wan2.2/scripts/embedding_search.py:503-513``). It supersedes S1's earlier phrasing that
"the replay operator must cast": the obligation did not disappear, it moved to ``velocity_fn``, where
one implementation serves optimize, replay and deployment alike.

**Carry-forward contracts this rule creates** (S2 review, finding 2):

* **S3 MUST** re-run S1's conditional-velocity parity fixture through the ACTUAL
  ``replay_with_positive`` operator, not against a direct transformer call as S1 could only do.
  **DISCHARGED in S3** by ``test_pos_context_replay.py`` --
  ``test_the_deployed_forward_is_the_replay_operator_s_own_v_cond`` (fp32, bitwise) and
  ``test_parity_at_bf16_holds_for_the_operator_plus_casting_velocity_fn`` (bf16, bitwise for the
  operator + casting-closure composition, and *unequal* without the cast). Both evaluate the deployed
  forward on exactly the ``(z, timestep)`` the operator's own conditional call received.
* **S4 MUST** add a closure test over the runner-built ``velocity_fn`` showing it casts BOTH branches
  at bf16. **STILL OPEN** -- and the only part of the rule still unpinned: the S2/S3 tests prove the
  operators do *not* cast, which is exactly why nothing yet proves the wiring *does*.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from maxdiffusion.models.wan.null_inversion_wan import (
    ADAM_B1,
    ADAM_B2,
    ADAM_EPS,
    NUM_TRAIN_TIMESTEPS,
    N_HIST_FRAMES,
    _checked_velocity,
    _validate_sigmas,
)
from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    apply_first_frame_pin,
    rollout_timesteps_from_sigmas,
)


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


def _normalized_base_context(base_context) -> jax.Array:
    """The T5("") convention: ``[S, D]``, or ``[1, S, D]`` squeezed to it. One context, never a batch."""
    base_context = jnp.asarray(base_context).astype(jnp.float32)
    if base_context.ndim == 3:
        if base_context.shape[0] != 1:
            raise ValueError(f"base_context must have a unit leading axis when rank-3, got {base_context.shape}")
        base_context = base_context[0]
    if base_context.ndim != 2:
        raise ValueError(f"base_context must be [S, D] or [1, S, D], got shape {base_context.shape}")
    return base_context


def optimize_positive_embeddings(
    velocity_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    traj: jax.Array,
    z_i0: jax.Array,
    sigmas: jax.Array,
    base_context: jax.Array,
    *,
    inner_iters: int,
    lr: float,
    guide_scale: float,
    pos_init: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-step POSITIVE-context optimization -- the branch swap of exp_04's null-text loop (plan §3).

    For each sampler step i the state ``z_bar_i`` is held fixed while ``inner_iters`` Adam steps run on
    the conditional context ``C_i`` so that one CFG Euler step from ``z_bar_i`` lands on the inversion
    pivot ``traj[i+1]``. Then ``C_i`` is locked, one more forward with the locked value advances
    ``z_bar_{i+1}``, and step i+1 warm-starts from it. Ported from ``optimize_positive_embeddings``
    (``third_party/Wan2.2/scripts/embedding_search.py:681-788``, submodule pin f370228).

    **The asymmetry with the null slot** (the reference's own docstring says it): gradients flow
    through ``v_cond``, so it is ``v_uncond`` that is C-independent and gets computed **once per outer
    step** (:736-741) and held fixed across the inner loop. Two consequences the tests pin: at
    ``guide_scale = 1`` the null branch would cancel but the positive branch *is* the whole velocity,
    so C-gradients are non-zero there (exp_04's are exactly zero); and the objective is the **sum** of
    per-example losses, not their mean, so each example's gradient is exactly the gradient of its own
    loss and batch composition cannot change any example's result (plan §3 batching contract).

    **The context convention (S1).** ``C`` is the deployed ``[B, POS_L, D]`` context passed *directly*
    as the whole ``encoder_hidden_states`` -- no 512-row splice, because
    ``side_adapter_wan.wan_pre_context_adapter_forward`` conditions the frozen backbone on exactly
    that. ``base_context`` plays two roles: its ``pos_context_from_t5`` truncation is the warm start,
    and the full ``[S, D]`` tensor (broadcast over the batch) is the frozen unconditional context.

    **THE CAST SEAM.** Per the module docstring's one rule, this optimizer hands ``velocity_fn`` the
    fp32 context **unmodified -- bit for bit the tensor it returns** -- and the runner-built
    real-backbone ``velocity_fn`` is what casts to the activation dtype. Casting here instead would
    need an activation dtype this operator cannot see behind an opaque ``velocity_fn``, would have to
    be re-implemented and kept in sync in replay, and would leave deployment's own cast
    (``side_adapter_wan.py:767``) as a second owner. ``test_pos_context_optimize.py`` pins the half of
    that rule this function can prove; the module docstring records the S3/S4 obligations it cannot.

    Seam: ``velocity_fn(latents, timestep_2d, encoder_hidden_states) -> v``, as in exp_04's R3/R4a.
    ``v`` must come back with the latent shape; it is used in float32.

    Args:
      velocity_fn: the seam above.
      traj: ``[N+1, B, C, F, H, W]`` inversion pivots; ``traj[i+1]`` is step i's target and ``traj[0]``
        is the starting state (the caller passes the arm's choice -- traj[0] or eps_0 -- per plan §4).
      z_i0: ``[B, C, 1, H, W]`` (or ``[B, C, F, H, W]``) first-frame condition.
      sigmas: the ``N+1`` descending grid ending at 0.0; validated eagerly, so it must be concrete.
      base_context: ``[S, D]`` or ``[1, S, D]`` T5("") context -- the unconditional branch verbatim,
        and, truncated to ``POS_L`` rows, the warm start. In production ``S = 512``, ``D = 4096``.
      inner_iters: Adam steps per sampler step (``J``); at least 1.
      lr: Adam learning rate; ``0`` freezes C, which is the B0/B2-0 frozen-context control.
      guide_scale: deployment CFG weight ``w``.
      pos_init: optional warm-start override, ``[L, D]`` (broadcast over the batch) or ``[B, L, D]``.
        Defaults to ``pos_context_from_t5(base_context)``, the plan §3 warm start; supplying
        ``pos_context_from_t5(base_context, l_pos=1)`` is how plan §4's L_pos ablation runs, and a
        per-example tensor is how a caller resumes a partially optimized trajectory.

    Returns:
      ``(pos_embeds, z_bar_states, z_final, losses, grad_norms)``, all float32:

      * ``pos_embeds`` ``[N, B, L, D]`` -- the locked contexts ``C_0 .. C_{N-1}``.
      * ``z_bar_states`` ``[N, B, C, F, H, W]`` -- the **pre-step** states ``z_bar_0 .. z_bar_{N-1}``,
        post-pin, aligned one-to-one with ``pos_embeds``: state i is the latent the forwards that
        produced ``C_i`` were evaluated at. This is the K2 cache schema (plan §3/P2', ``[25, ...]``)
        and exactly what the teacher-forced regression trainer consumes -- a first-class output, not a
        diagnostic. The terminal ``z_bar_N`` is deliberately **not** in here: it is the endpoint of the
        run, has no context to pair with, and returning it inline would leave S4/K2 to remember an
        implicit ``[:-1]`` slice (S2 review, finding 1).
      * ``z_final`` ``[B, C, F, H, W]`` -- that terminal state ``z_bar_N``. The full trajectory, when
        something genuinely needs it, is ``jnp.concatenate([z_bar_states, z_final[None]])``.
      * ``losses`` / ``grad_norms`` ``[N, J, B]`` -- per-example traces. ``losses[i, j]`` is the loss
        *before* the j-th update (as the reference logs it), so ``losses[i, -1]`` is the value after
        ``J-1`` updates, not the post-optimization loss.
    """
    _validate_sigmas(sigmas)
    if isinstance(inner_iters, bool) or not isinstance(inner_iters, (int, np.integer)):
        # Truncating 1.5 to 1 would silently run a different recipe than the caller asked for.
        raise ValueError(f"inner_iters must be an integer, got {inner_iters!r}")
    if int(inner_iters) < 1:
        raise ValueError(f"inner_iters must be >= 1, got {inner_iters}")
    if not np.isfinite(lr) or float(lr) < 0.0:
        # A negative lr ascends the objective and a nonfinite one poisons every target.
        raise ValueError(f"lr must be finite and non-negative, got {lr}")
    if not np.isfinite(guide_scale):
        raise ValueError(f"guide_scale must be finite, got {guide_scale}")

    sigmas = jnp.asarray(sigmas, dtype=jnp.float32)
    traj = jnp.asarray(traj).astype(jnp.float32)
    z_i0 = jnp.asarray(z_i0).astype(jnp.float32)
    base_context = _normalized_base_context(base_context)

    if traj.ndim != 6:
        raise ValueError(f"traj must be [N+1, B, C, F, H, W], got shape {traj.shape}")
    if traj.shape[0] != sigmas.shape[0]:
        raise ValueError(f"traj length {traj.shape[0]} must match the sigma grid length {sigmas.shape[0]}")
    batch, channels, f_lat, h_lat, w_lat = traj.shape[1:]
    if z_i0.ndim != 5 or (z_i0.shape[0], z_i0.shape[1], z_i0.shape[3:]) != (batch, channels, (h_lat, w_lat)):
        raise ValueError(f"z_i0 shape {z_i0.shape} is inconsistent with traj shape {traj.shape}")
    if z_i0.shape[2] not in (1, f_lat):
        raise ValueError(f"z_i0 must carry 1 or {f_lat} latent frames, got {z_i0.shape[2]}")

    warm = pos_context_from_t5(base_context) if pos_init is None else jnp.asarray(pos_init).astype(jnp.float32)
    if warm.ndim == 2:
        warm = jnp.broadcast_to(warm, (batch, *warm.shape))
    if warm.ndim != 3 or warm.shape[0] != batch or warm.shape[2] != base_context.shape[1]:
        # Checked against the context width rather than against 4096 so the same operator runs at the
        # toy geometry the oracle tests use; production 512x4096 is fail-closed at the runner boundary.
        raise ValueError(f"pos_init shape {warm.shape} is inconsistent with base_context {base_context.shape}")
    if warm.shape[1] < 1:
        raise ValueError(f"pos_init must carry at least one row, got shape {warm.shape}")

    timesteps = rollout_timesteps_from_sigmas(sigmas, NUM_TRAIN_TIMESTEPS)
    # eps_root=0.0 spelled out: the recipe is exactly torch's -lr * m_hat / (sqrt(v_hat) + eps).
    optimizer = optax.adam(lr, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS, eps_root=0.0)
    uncond_context = jnp.broadcast_to(base_context, (batch, *base_context.shape))
    weight = jnp.asarray(guide_scale, dtype=jnp.float32)

    def outer(carry, i):
        z_bar, pos = carry
        step_t = jnp.broadcast_to(timesteps[i], (batch,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=N_HIST_FRAMES)
        dsigma = sigmas[i + 1] - sigmas[i]
        target = traj[i + 1]
        # C-independent, so it is evaluated once here and reused by every inner iteration.
        v_unc = lax.stop_gradient(_checked_velocity(velocity_fn, z_bar, timestep_2d, uncond_context))

        def euler_step(value):
            v_cond = _checked_velocity(velocity_fn, z_bar, timestep_2d, value)
            return apply_first_frame_pin(z_bar + dsigma * (v_unc + weight * (v_cond - v_unc)), z_i0)

        def objective(value):
            per_example = jnp.mean((euler_step(value) - target) ** 2, axis=(1, 2, 3, 4))
            return jnp.sum(per_example), per_example  # sum, so per-example gradients stay independent

        def inner(state, _):
            value, opt_state = state
            (_, per_example), grads = jax.value_and_grad(objective, has_aux=True)(value)
            updates, opt_state = optimizer.update(grads, opt_state, value)
            grad_norm = jnp.sqrt(jnp.sum(grads**2, axis=(1, 2)))
            return (optax.apply_updates(value, updates), opt_state), (per_example, grad_norm)

        # Fresh Adam moments for every sampler step, as the reference constructs a new optimizer.
        state = (pos, optimizer.init(pos))
        (locked, _), (losses, grad_norms) = lax.scan(inner, state, length=int(inner_iters))
        z_next = euler_step(locked)  # the advance uses the LOCKED context, not the pre-loop one
        return (z_next, locked), (locked, z_next, losses, grad_norms)

    z_bar_0 = apply_first_frame_pin(traj[0], z_i0)
    _, (pos_embeds, z_next, losses, grad_norms) = lax.scan(outer, (z_bar_0, warm), jnp.arange(sigmas.shape[0] - 1))
    # ``z_next`` holds z_bar_1 .. z_bar_N. The cache schema wants the N states the optimizer actually
    # conditioned on -- z_bar_0 .. z_bar_{N-1} -- so the terminal state is split off by name here
    # rather than sliced off by a downstream reader who might forget (S2 review, finding 1).
    z_bar_states = jnp.concatenate([z_bar_0[None], z_next[:-1]], axis=0)
    return pos_embeds, z_bar_states, z_next[-1], losses, grad_norms


def replay_with_positive(
    velocity_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    z_start: jax.Array,
    z_i0: jax.Array,
    sigmas: jax.Array,
    pos_embeds: jax.Array,
    base_context: jax.Array,
    *,
    guide_scale: float,
    return_trajectory: bool = False,
):
    """Deployment CFG replay from ``z_start`` under per-step positive contexts (plan §3 step 3).

    ``z_0 = pin(z_start)`` and, for i = 0 .. N-1::

        z_{i+1} = pin(z_i + (sigma_{i+1} - sigma_i) * [v_unc + w * (v_cond(C_i) - v_unc)])

    with ``C_i`` passed **directly** as the whole ``encoder_hidden_states`` (8 rows, the deployed
    convention) and the unconditional branch on the frozen full-length T5("") context. Ported from
    ``regenerate_with_positive_embeds`` (``third_party/Wan2.2/scripts/embedding_search.py:822-853``,
    submodule pin f370228).

    **Both velocities are fresh at every step -- the same symmetry exp_04's ``replay_with_nulls`` has,
    and for the same reason.** Neither branch can be hoisted out of the loop: replay *moves* ``z``
    after every step and both branches are evaluated at the current ``z_i``, so a cached velocity
    would be a stale-latent correctness bug, not a saving. (``optimize_positive_embeddings`` caches
    ``v_unc`` only because it holds ``z_bar_i`` fixed across its inner loop -- the one thing replay
    never does. Which branch carries the optimized tensor differs between the two operators; the
    freshness rule does not.)

    **The B0 control is an ACTIVE-CFG control, unlike exp_04's A0.** Passing
    ``pos_context_from_t5(base_context)`` at every step gives the frozen-context arm -- but it does
    *not* collapse CFG. exp_04's A0 works because base-row nulls make the two branches the identical
    forward; here they are different forwards at different sequence lengths (``POS_L`` vs ``S``), so
    ``v_cond != v_unc`` and B0's output genuinely depends on ``w``. Plan §4 calls B0 "the matched
    control under the active-CFG convention" for exactly this reason; it is not an unguided sampler.
    Pinned in ``test_pos_context_replay.py``.

    **The cast seam:** per the module docstring's one rule this operator passes the fp32 contexts to
    ``velocity_fn`` **unchanged** -- both branches -- and the runner-built real-backbone
    ``velocity_fn`` performs the activation-dtype cast. **The S3 MUST contract is DISCHARGED** by
    ``test_pos_context_replay.py``, which re-runs S1's conditional-velocity parity fixture through
    this operator: fp32 bitwise, plus bf16 bitwise for the operator + casting-closure composition.

    Args:
      velocity_fn: ``(latents, timestep_2d, encoder_hidden_states) -> v``, the R3/R4a seam.
      z_start: ``[B, C, F, H, W]`` starting latents (an inversion endpoint, or fresh noise).
      z_i0: ``[B, C, 1, H, W]`` (or ``[B, C, F, H, W]``) first-frame condition.
      sigmas: the ``N+1`` descending grid ending at 0.0; validated eagerly, so it must be concrete.
      pos_embeds: ``[N, B, L, D]`` per-step per-example contexts, or ``[N, L, D]`` to share one context
        per step across the batch.
      base_context: ``[S, D]`` or ``[1, S, D]`` frozen T5("") context for the unconditional branch.
      guide_scale: deployment CFG weight ``w``.
      return_trajectory: also return the ``[N+1, B, C, F, H, W]`` trajectory ``z_0 .. z_N``.

    Returns:
      ``z_final`` ``[B, C, F, H, W]`` float32, or ``(z_final, trajectory)``.
    """
    _validate_sigmas(sigmas)
    if not np.isfinite(guide_scale):
        raise ValueError(f"guide_scale must be finite, got {guide_scale}")

    sigmas = jnp.asarray(sigmas, dtype=jnp.float32)
    z_start = jnp.asarray(z_start).astype(jnp.float32)
    z_i0 = jnp.asarray(z_i0).astype(jnp.float32)
    pos_embeds = jnp.asarray(pos_embeds).astype(jnp.float32)
    base_context = _normalized_base_context(base_context)
    steps = sigmas.shape[0] - 1

    if z_start.ndim != 5:
        raise ValueError(f"z_start must be [B, C, F, H, W], got shape {z_start.shape}")
    batch, channels, f_lat, h_lat, w_lat = z_start.shape
    if z_i0.ndim != 5 or (z_i0.shape[0], z_i0.shape[1], z_i0.shape[3:]) != (batch, channels, (h_lat, w_lat)):
        raise ValueError(f"z_i0 shape {z_i0.shape} is inconsistent with z_start shape {z_start.shape}")
    if z_i0.shape[2] not in (1, f_lat):
        raise ValueError(f"z_i0 must carry 1 or {f_lat} latent frames, got {z_i0.shape[2]}")
    if pos_embeds.ndim == 3:
        pos_embeds = jnp.broadcast_to(pos_embeds[:, None], (pos_embeds.shape[0], batch, *pos_embeds.shape[1:]))
    if pos_embeds.ndim != 4:
        raise ValueError(f"pos_embeds must be [N, L, D] or [N, B, L, D], got shape {pos_embeds.shape}")
    if pos_embeds.shape[0] != steps:
        raise ValueError(
            f"pos_embeds must carry one entry per sampler step: got {pos_embeds.shape[0]}, expected {steps}"
        )
    if pos_embeds.shape[1] != batch:
        # Checked separately from the width clause below so neither can mask the other: a rank-4
        # [N, 1, L, D] tensor at B > 1 would otherwise drive the whole batch with example 0's context
        # while the velocity seam's own guard still saw v.shape == z.shape (exp_04 R4a's lesson).
        raise ValueError(f"pos_embeds batch {pos_embeds.shape[1]} does not match z_start batch {batch}")
    if pos_embeds.shape[3] != base_context.shape[1]:
        raise ValueError(f"pos_embeds shape {pos_embeds.shape} is inconsistent with base_context {base_context.shape}")
    if pos_embeds.shape[2] < 1:
        raise ValueError(f"pos_embeds must carry at least one row, got shape {pos_embeds.shape}")

    timesteps = rollout_timesteps_from_sigmas(sigmas, NUM_TRAIN_TIMESTEPS)
    uncond_context = jnp.broadcast_to(base_context, (batch, *base_context.shape))
    weight = jnp.asarray(guide_scale, dtype=jnp.float32)

    def body(current, i):
        step_t = jnp.broadcast_to(timesteps[i], (batch,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=N_HIST_FRAMES)
        # Both fresh, in the reference's order: the conditional branch on this step's C_i, the
        # unconditional on the frozen context -- both at the CURRENT latent.
        v_cond = _checked_velocity(velocity_fn, current, timestep_2d, pos_embeds[i])
        v_unc = _checked_velocity(velocity_fn, current, timestep_2d, uncond_context)
        v_cfg = v_unc + weight * (v_cond - v_unc)
        filled = apply_first_frame_pin(current + (sigmas[i + 1] - sigmas[i]) * v_cfg, z_i0)
        return filled, filled

    start = apply_first_frame_pin(z_start, z_i0)
    z_final, stepped = lax.scan(body, start, jnp.arange(steps))
    if return_trajectory:
        return z_final, jnp.concatenate([start[None], stepped], axis=0)
    return z_final
