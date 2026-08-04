"""exp_03 `rollout_objective` — the trainer that can carry the new objectives.

Everything about a run except the objective is inherited from the exp_02 overfit100 trainer: the
same ``Overfit100TrainState``, the same Orbax item shapes, the same preflights (config gates, pinned
snapshot, dataset byte-verify, manifest-bound context table), the same optimizer, dataset, sharding,
checkpoint schedule and loop. Only :meth:`Exp03Trainer._loss_and_step_fns` is overridden, because
that is the one thing exp_03 changes.

``exp03_objective: control`` returns the parent's own functions *by identity*, not a copy that would
merely look equal. That is what makes ctrl0 a replication guard rather than a second implementation:
with the same seed, data order and checkpoint bytes, ctrl0 through this trainer executes exactly the
code exp_02 executed. The three trials (``corrective_ss``, ``rollout_loss``, ``combined``) are
implemented below; anything declared without an implementation raises :class:`NotImplementedError`
at startup rather than quietly training the control under another arm's run name.

**RNG discipline (plan v3.1 §1b, delta-review 2).** The *shared stream* is exp_02's: a single
``jax.random.key(seed + 1)`` advanced by exactly one ``split`` per step inside the train step, whose
per-step key then splits into (noise, timestep, dropout). ctrl0 and the shared draws of every arm
must reproduce it exactly, so the new objectives' own randomness — the ``p_ss`` coins and the sigma
index supports — comes from :func:`exp03_aux_key`, an *auxiliary* key derived from
``(seed, global_step, purpose)``. The epsilon is NOT among them: every arm takes it from the shared
stream exactly where the control does, which is what makes an arm's noise at a given step the
control's noise, and what lets A's off-path state be compared with its teacher-forced twin. It is derived, never split off the stream, so
consuming it cannot advance the stream; and because it is keyed on the global step rather than on
call order, it survives a resume unchanged.
"""

from __future__ import annotations

import hashlib
import math
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from maxdiffusion import max_logging
from maxdiffusion.models.wan.overfit100_sampling import (
    overfit100_sampler_grid,
    overfit100_sampler_step,
    overfit100_step_timestep,
)
from maxdiffusion.models.wan.side_adapter_wan import (
    apply_first_frame_pin,
    build_noisy_pinned_latents,
    masked_velocity_mse,
    _dtype,
)
from maxdiffusion.trainers.wan_ti2v_overfit100_trainer import (
    WanTI2VOverfit100Trainer,
    _denoising_loss,
    _make_train_step,
    _train_step,
)
from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import _build_noise

EXP03_MODEL_TYPE = "EXP03_TI2V"

# The objective surface, fixed by plan v3.1 §1. "control" is the plain one-step objective, i.e.
# exp_02's; the other three are the trials.
EXP03_CONTROL_OBJECTIVE = "control"
EXP03_OBJECTIVES = (EXP03_CONTROL_OBJECTIVE, "corrective_ss", "rollout_loss", "combined")

# Auxiliary-key purposes. Named rather than numbered so adding one later cannot renumber the
# others: the purpose's id is a hash of its NAME (see _purpose_id).
EXP03_AUX_PURPOSES = (
    "p_ss_coin",  # trial A: the scheduled-sampling coin
    "k_a_draw",  # trial A: k_A ~ U{1..exp03_k_a}, drawn FIRST (plan v2.2)
    "index_support",  # trial A: the start index s ~ U{0 .. 24 - k_A}
    "index_support_rollout",  # trial B: the start index s ~ U{0 .. 22}
)
# There is deliberately NO self-generation noise purpose: A's off-path state is produced by the
# sampler from the SAME epsilon as its teacher-forced branch, which is exactly what makes the two
# branches comparable. A separate noise draw would break that.

# Offset for the auxiliary root key. The shared stream is key(seed + 1); the auxiliary root is a
# DIFFERENT key, not a descendant of it, so the two can never collide or interleave.
EXP03_AUX_SEED_OFFSET = 1_000_003


def _purpose_id(purpose: str) -> int:
    """A stable 32-bit id for a purpose NAME (sha256, so it never depends on declaration order)."""
    if purpose not in EXP03_AUX_PURPOSES:
        raise ValueError(f"unknown exp_03 rng purpose {purpose!r}; declared purposes are {list(EXP03_AUX_PURPOSES)}")
    return int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:4], "big")


def exp03_aux_key(*, seed: int, global_step, purpose: str, salt: int = 0) -> jax.Array:
    """An auxiliary key for a NEW-objective draw, derived without touching the shared stream.

    ``fold_in(fold_in(key(seed + offset), global_step), purpose_id)``. Four properties matter and
    each is tested:

    * **Non-advancing.** It is derived from the config seed, not split off the training rng, so a
      step that draws from it leaves the shared stream in exactly the state exp_02 would have left
      it in. This is what lets ctrl0 be a bitwise replication guard while A/B/C add randomness.
    * **Resume-stable.** Keyed on the global step, not on how many times it has been called, so a
      preempted-and-restarted segment draws the same values at the same steps. The caller must pass
      the LOOP's step: ``state.step`` is not resume-safe, because a restore brings back params and
      opt_state and leaves the freshly-built step behind.
    * **Cross-arm aligned.** The same ``(seed, step, purpose)`` gives the same key in every arm, so
      where two arms make the same kind of draw they make the *same* draw.
    * **Tracer-safe.** ``global_step`` arrives as a traced scalar from inside the compiled train
      step, so it is folded in as an array and never coerced with ``int()``. ``seed`` and
      ``purpose`` are static (config values), so they may stay Python-level. The non-negative check
      therefore applies only when the step is a concrete Python integer -- under tracing there is
      no value to check, and a fabricated one would be worse than none.
    """
    if isinstance(global_step, (int, np.integer)) and not isinstance(global_step, bool) and int(global_step) < 0:
        raise ValueError(f"global_step must be non-negative; got {global_step}")
    key = jax.random.key(int(seed) + EXP03_AUX_SEED_OFFSET)
    key = jax.random.fold_in(key, jnp.asarray(global_step, dtype=jnp.uint32))
    key = jax.random.fold_in(key, _purpose_id(purpose))
    # ``salt`` re-draws THIS purpose while everything else about the step stays put. It is folded
    # only when non-zero, so every existing draw is bit-identical to what it was before the salt
    # existed. S1.5 uses it to vary the sigma SUPPORT alone: moving the global step instead would
    # also move A's p_ss coin and the ramp, and the resulting variance would mix three things.
    if int(salt) != 0:
        key = jax.random.fold_in(key, int(salt))
    return key


def validate_exp03_config(config) -> str:
    """Gate the exp_03 knobs before anything expensive happens; return the objective."""
    objective = str(getattr(config, "exp03_objective", EXP03_CONTROL_OBJECTIVE))
    if objective not in EXP03_OBJECTIVES:
        raise ValueError(f"exp03_objective must be one of {list(EXP03_OBJECTIVES)}; got {objective!r}")

    k_a = int(getattr(config, "exp03_k_a", 2))
    if k_a not in (1, 2):
        raise ValueError(f"exp03_k_a is the MAX scheduled-sampling advance and the plan fixes it at 2; got {k_a}")
    k_b = int(getattr(config, "exp03_k_b", 2))
    if k_b != 2:
        raise ValueError(f"exp03_k_b is fixed at 2 by the plan (a two-step rollout loss); got {k_b}")
    lam = float(getattr(config, "exp03_lambda", 0.5))
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"exp03_lambda must be in [0, 1]; got {lam}")
    p_ss_max = float(getattr(config, "exp03_p_ss_max", 0.5))
    if not 0.0 <= p_ss_max <= 1.0:
        raise ValueError(f"exp03_p_ss_max must be in [0, 1]; got {p_ss_max}")
    ramp = int(getattr(config, "exp03_p_ss_ramp_steps", 500))
    if ramp < 0:
        raise ValueError(f"exp03_p_ss_ramp_steps must be non-negative; got {ramp}")
    origin = int(getattr(config, "exp03_ramp_origin", 0))
    if origin < 0:
        raise ValueError(f"exp03_ramp_origin must be non-negative; got {origin}")
    return objective


# =================================================================================================
# The three objectives (plan v3.1 §1, index supports per the v2.2 direction correction).
#
# The sigma grid DESCENDS: index 0 is the highest sigma and the eval advances ``i -> i+1`` toward
# lower sigma, so a "start" ``s`` and an "end" ``e = s + k`` satisfy ``sigma[s] > sigma[e]``, and
# the terminal index 25 (sigma = 0) is never reached because ``e <= 24``. The smallest positive grid
# sigma is ~0.1724, so dividing by ``sigma_lo`` needs no clamp (reviewer-verified, plan review P1).
#
# ONE SUPPORT PER BATCH (a specification reading, flagged for review): the plan describes the draw
# per batch ("each batch draws BOTH supports independently"), and per-EXAMPLE indices would break
# the extracted sampler step's Euler broadcast -- its ``(sigma[i+1] - sigma[i])`` scale is a scalar
# against ``[B, C, F, H, W]``. Keeping the one-sampler rule intact therefore also fixes this choice.
#
# FORWARD CONVENTIONS (round-3 review D3). A's final SUPERVISED forward uses exp_02's training
# convention (``deterministic=False`` with the dropout rng from the shared stream) -- it is the
# one-step objective's forward, just at a different state. Every forward that is part of a
# TRAJECTORY uses the EVAL convention (``deterministic=True``): A's detached advance, and B's two
# DIFFERENTIATED rollout steps (and C's B-term). Differentiability does not require training mode,
# and B is defined as a rollout loss through the operator the evaluation runs, so training mode
# there would differentiate a different trajectory the moment dropout is nonzero.
# =================================================================================================


def exp03_p_ss(config, global_step) -> jax.Array:
    """The scheduled-sampling probability: linear ``0 -> p_ss_max`` over the ramp, then constant.

    Keyed to ``global_step - exp03_ramp_origin``. The plan says "global step - 10,000" because
    Tier 1 continues from the exp_02 step-10,000 checkpoint; Tier 2 starts from the pretrained init
    at step 0. One expression covers both if the origin is configuration rather than a constant, so
    ``exp03_ramp_origin`` defaults to 0 (Tier 2) and is set to 10000 for Tier-1 arms. Keyed to the
    GLOBAL step, so a preempted-and-restarted segment resumes the ramp where it was.
    """
    ramp = int(getattr(config, "exp03_p_ss_ramp_steps", 500))
    p_max = jnp.asarray(float(getattr(config, "exp03_p_ss_max", 0.5)), dtype=jnp.float32)
    # NOT ``int()``-coerced: the origin is used only in arithmetic, so it may arrive as a TRACER.
    # That is what lets S1.5 thread it as an explicit argument and have one compilation serve both
    # states -- rather than capturing it in a closure, where the checkpoint state's compiled function
    # would silently be reused for the init state and run it with the wrong ramp.
    origin = getattr(config, "exp03_ramp_origin", 0)
    elapsed = jnp.asarray(global_step, dtype=jnp.float32) - jnp.asarray(origin, dtype=jnp.float32)
    if ramp <= 0:
        return p_max * (elapsed >= 0.0).astype(jnp.float32)
    return p_max * jnp.clip(elapsed / jnp.asarray(float(ramp), dtype=jnp.float32), 0.0, 1.0)


def _exp03_prologue(params, state, data, rng, config, scheduler):
    """Everything the three objectives share with exp_02's loss, drawn in exp_02's ORDER.

    The shared stream is split exactly as ``_denoising_loss`` splits it -- ``(noise, step, dropout)``
    -- so the epsilon and the dropout key an arm uses at a given step are the ones the control uses
    at that step. ``step_rng`` (the control's per-example timestep draw) is deliberately unused: an
    arm's loss point comes from its index support, which is drawn from an AUXILIARY key so that not
    drawing it cannot shift the shared stream.
    """
    noise_rng, step_rng, dropout_rng = jax.random.split(rng, 3)
    del step_rng
    weights_dtype = _dtype(config.weights_dtype)
    activations_dtype = _dtype(config.activations_dtype)
    bsz = config.global_batch_size_to_train_on
    transformer = nnx.merge(state.graphdef, params, state.rest_of_state)

    z_i0_f32 = data["z_i0"][:bsz].astype(jnp.float32)
    z_video_f32 = data["z_video"][:bsz].astype(jnp.float32)
    b, _, f_lat, h_lat, w_lat = z_video_f32.shape
    episode_index = data["episode_index"][:bsz].astype(jnp.int32)
    context = state.context_table[episode_index].astype(activations_dtype)
    eps = _build_noise(noise_rng, z_video_f32.shape, jnp.float32, config)

    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=config.side_adapter_sampling_steps,
        flow_shift=config.flow_shift,
        sigma_min=scheduler.config.sigma_min,
        sigma_max=scheduler.config.sigma_max,
        num_train_timesteps=scheduler.config.num_train_timesteps,
    )
    return SimpleNamespace(
        transformer=transformer,
        weights_dtype=weights_dtype,
        dropout_rng=dropout_rng,
        z_i0=z_i0_f32,
        z_video=z_video_f32,
        eps=eps,
        context=context,
        sigmas=sigmas,
        timesteps=timesteps,
        b=b,
        shape=(f_lat, h_lat, w_lat),
        num_steps=int(config.side_adapter_sampling_steps),
    )


def _interpolant_at(ctx, index) -> jax.Array:
    """``pin((1 - sigma[index]) * z_gt + sigma[index] * eps)`` — the teacher-forced state."""
    sigma = ctx.sigmas[index].astype(jnp.float32)
    return build_noisy_pinned_latents(ctx.z_video, ctx.z_i0, ctx.eps, jnp.full((ctx.b,), sigma))


def _training_velocity_fn(ctx):
    """The DIFFERENTIATED forward: exp_02's training convention."""

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        return ctx.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=False,
            rngs=nnx.Rngs(dropout=ctx.dropout_rng),
        )

    return velocity_fn


def _sampling_velocity_fn(ctx):
    """The state-producing forward: the EVAL convention, because it imitates the eval sampler."""

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        return ctx.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=True,
        )

    return velocity_fn


def _forward_velocity(ctx, z, index) -> jax.Array:
    """One differentiated forward at grid ``index`` (no Euler step, just the prediction)."""
    timestep_2d = overfit100_step_timestep(ctx.timesteps, index, ctx.b, *ctx.shape)
    return _training_velocity_fn(ctx)(
        hidden_states=z.astype(ctx.weights_dtype),
        timestep=timestep_2d,
        encoder_hidden_states=ctx.context,
    )


def corrective_support(*, seed: int, global_step, num_steps: int, k_a_max: int, support_salt: int = 0):
    """Trial A's support: ``k_A ~ U{1..k_a_max}`` FIRST, then ``s ~ U{0 .. num_steps-1-k_A}``.

    Drawn in that order because the start's range depends on the length (plan v2.2). ``e = s + k_A``
    is at most ``num_steps - 1 = 24``, so the terminal index (sigma = 0) is unreachable by
    construction rather than by a clamp.
    """
    k_a = jax.random.randint(
        exp03_aux_key(seed=seed, global_step=global_step, purpose="k_a_draw", salt=support_salt),
        (),
        1,
        int(k_a_max) + 1,
    )
    start = jax.random.randint(
        exp03_aux_key(seed=seed, global_step=global_step, purpose="index_support", salt=support_salt),
        (),
        0,
        num_steps - k_a,
    )
    return start, start + k_a, k_a


def rollout_support(*, seed: int, global_step, num_steps: int, k_b: int, support_salt: int = 0):
    """Trial B's support: ``s ~ U{0 .. num_steps-1-k_B}``, path ``s -> s+1 -> ... -> s+k_B``."""
    start = jax.random.randint(
        exp03_aux_key(seed=seed, global_step=global_step, purpose="index_support_rollout", salt=support_salt),
        (),
        0,
        num_steps - int(k_b),
    )
    return start, start + int(k_b)


def _advance_with_sampler(ctx, z, start, k, *, velocity_fn, k_max):
    """Advance the EXTRACTED sampler step ``k`` times from grid index ``start`` (one-sampler rule).

    FIXED LENGTH, not ``fori_loop(start, end)``: ``k`` is a traced draw, so the loop had a dynamic
    trip COUNT (not a per-draw compiled graph shape -- that earlier wording was wrong). ``k_max`` is 2, so the advance is unrolled to ``k_max`` steps and the state after
    ``k`` of them is SELECTED -- selection-equivalent for every ``k in 1..k_max``, pinned exactly by
    a test against an explicit unroll.

    **Cost, stated plainly:** the unroll ALWAYS runs ``k_max`` forwards, so A's runtime advance
    count rises from a mean of 1.5 to 2. A's S1 measurement of 1.47x was taken under the old
    variable-trip loop; the new code will land higher and may reach the 1.6x STOP budget. That is a
    real trade and the re-smoke must re-measure it.

    **Status: a compiler-shape HYPOTHESIS, not a proven root cause.** The construct was shared by
    standalone A, which was finite, so it is not C-unique; a dynamic trip count is not a different
    compiled graph shape per draw. What decides is the frozen-state discriminator
    (:func:`exp03_frozen_replay`) run from the snapshot, comparing old-loop and new-unroll
    executables under identical diagnostics.
    """
    state = z
    result = z
    for offset in range(int(k_max)):
        state = overfit100_sampler_step(
            state,
            start + offset,
            velocity_fn=velocity_fn,
            sigmas=ctx.sigmas,
            timesteps=ctx.timesteps,
            context=ctx.context,
            z_i0=ctx.z_i0.astype(ctx.weights_dtype),
        )
        result = jnp.where(k == offset + 1, state, result)
    return result


def _corrective_ss_loss(params, state, data, rng, config, scheduler, *, global_step=None):
    """Trial A — corrective scheduled sampling.

    Teacher-forced state at ``sigma[s]``; with probability ``p_ss`` advance ``k_A`` steps of the
    EXTRACTED sampler under ``stop_gradient`` (off-path exposure without gradient), else take the
    interpolant at ``sigma[e]`` -- the SAME ``(s, e)`` draw either way, so the loss point's
    distribution over ``sigma_lo`` is identical between branches by construction. Then ONE
    differentiated forward at ``sigma[e]`` against the corrective target
    ``v* = (z_lo - z_gt) / sigma_lo``, which is exact under the Euler rule
    (``z_next - z_gt = (sigma_next / sigma) (z - z_gt)``) and reduces to ``eps - z_gt`` on-path.

    No fresh noise is drawn for the self-generated state: it is produced by the sampler from the
    SAME epsilon as the teacher-forced branch, which is what keeps the two branches comparable.
    """
    if global_step is None:
        raise ValueError("the exp_03 trial objectives need the threaded global_step (round-2 plumbing)")
    ctx = _exp03_prologue(params, state, data, rng, config, scheduler)
    seed = int(getattr(config, "seed", 0))
    start, end, k_a = corrective_support(
        seed=seed,
        global_step=global_step,
        num_steps=ctx.num_steps,
        k_a_max=int(getattr(config, "exp03_k_a", 2)),
        support_salt=int(getattr(config, "exp03_support_salt", 0)),
    )
    sigma_lo = ctx.sigmas[end].astype(jnp.float32)
    sigma_hi = ctx.sigmas[start].astype(jnp.float32)

    z_hi = _interpolant_at(ctx, start)
    teacher_forced = _interpolant_at(ctx, end)
    advanced = jax.lax.stop_gradient(
        _advance_with_sampler(
            ctx,
            jax.lax.stop_gradient(z_hi).astype(ctx.weights_dtype),
            start,
            k_a,
            velocity_fn=_sampling_velocity_fn(ctx),
            k_max=int(getattr(config, "exp03_k_a", 2)),
        )
    ).astype(jnp.float32)

    coin = jax.random.uniform(exp03_aux_key(seed=seed, global_step=global_step, purpose="p_ss_coin"), ())
    take_self_generated = coin < exp03_p_ss(config, global_step)
    z_lo = jnp.where(take_self_generated, advanced, teacher_forced)
    z_lo = apply_first_frame_pin(z_lo, ctx.z_i0)

    v_pred = _forward_velocity(ctx, z_lo, end)
    v_target = (z_lo - ctx.z_video) / sigma_lo
    loss = masked_velocity_mse(v_pred, v_target, ctx.b)
    aux = _exp03_aux(
        loss,
        v_pred=v_pred,
        v_target=v_target,
        z_state=z_lo,
        ctx=ctx,
        sigma=sigma_lo,
        timestep=sigma_lo * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32),
    )
    aux.update(
        {
            "k_a": k_a.astype(jnp.float32),
            "s_a": start.astype(jnp.float32),
            "e_a": end.astype(jnp.float32),
            "sigma_hi_a": sigma_hi,
            "sigma_lo_a": sigma_lo,
            "coin": coin.astype(jnp.float32),
            "p_ss": exp03_p_ss(config, global_step),
            "take_self_generated": take_self_generated.astype(jnp.float32),
            "advance_finite": jnp.all(jnp.isfinite(advanced)).astype(jnp.float32),
            "z_lo_finite": jnp.all(jnp.isfinite(z_lo)).astype(jnp.float32),
            "v_target_finite": jnp.all(jnp.isfinite(v_target)).astype(jnp.float32),
            "loss_a_finite": jnp.isfinite(loss).astype(jnp.float32),
        }
    )
    return loss, aux


def _rollout_loss(params, state, data, rng, config, scheduler, *, global_step=None):
    """Trial B — short-horizon rollout loss, horizon-normalized.

    Teacher-forced start at ``sigma[s]``, then ``k_B`` steps of the EXTRACTED sampler in the EVAL
    convention (``deterministic=True``) with gradients flowing through every forward (``lax.scan``
    with ``jax.remat`` per step, so the unroll is rematerialized rather than held), scored against
    the ideal trajectory point with the SAME epsilon and divided by ``(sigma_hi - sigma_lo)**2``. Without that normalizer the nonuniform grid
    would reweight the loss by the square of the step size (plan review P1); with it, the optimum
    ``v = eps - z_gt`` gives exactly zero at every support.
    """
    if global_step is None:
        raise ValueError("the exp_03 trial objectives need the threaded global_step (round-2 plumbing)")
    ctx = _exp03_prologue(params, state, data, rng, config, scheduler)
    seed = int(getattr(config, "seed", 0))
    k_b = int(getattr(config, "exp03_k_b", 2))
    start, end = rollout_support(
        seed=seed,
        global_step=global_step,
        num_steps=ctx.num_steps,
        k_b=k_b,
        support_salt=int(getattr(config, "exp03_support_salt", 0)),
    )
    sigma_hi = ctx.sigmas[start].astype(jnp.float32)
    sigma_lo = ctx.sigmas[end].astype(jnp.float32)

    # THE approved operator: B is a short-horizon rollout loss through the sampler the EVALUATION
    # runs, so both differentiated forwards use the eval convention. Differentiability does not
    # require training mode -- gradients flow through a deterministic forward exactly as well --
    # and with a future nonzero dropout the training convention would differentiate a different
    # trajectory than the one the experiment is about (round-3 review D3).
    velocity_fn = _sampling_velocity_fn(ctx)

    @jax.remat
    def _step(carry, offset):
        z, index = carry
        z_next = overfit100_sampler_step(
            z,
            index,
            velocity_fn=velocity_fn,
            sigmas=ctx.sigmas,
            timesteps=ctx.timesteps,
            context=ctx.context,
            z_i0=ctx.z_i0.astype(ctx.weights_dtype),
        )
        del offset
        return (z_next, index + 1), None

    z_start = _interpolant_at(ctx, start).astype(ctx.weights_dtype)
    (z_end, _), _ = jax.lax.scan(_step, (z_start, start), xs=jnp.arange(k_b))

    z_ideal = _interpolant_at(ctx, end)
    horizon = jnp.maximum((sigma_hi - sigma_lo) ** 2, jnp.finfo(jnp.float32).tiny)
    raw = masked_velocity_mse(z_end.astype(jnp.float32), z_ideal, ctx.b)
    loss = raw / horizon
    aux = _exp03_aux(
        loss,
        v_pred=z_end.astype(jnp.float32),
        v_target=z_ideal,
        z_state=z_end.astype(jnp.float32),
        ctx=ctx,
        sigma=sigma_lo,
        timestep=sigma_lo * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32),
    )
    aux.update(
        {
            "raw_endpoint_mse": raw,
            "horizon_sq": horizon,
            "s_b": start.astype(jnp.float32),
            "e_b": end.astype(jnp.float32),
            "sigma_hi_b": sigma_hi,
            "sigma_lo_b": sigma_lo,
            "z_end_finite": jnp.all(jnp.isfinite(z_end)).astype(jnp.float32),
            "loss_b_finite": jnp.isfinite(loss).astype(jnp.float32),
        }
    )
    return loss, aux


def _combined_loss(params, state, data, rng, config, scheduler, *, global_step=None):
    """Trial C — the LITERAL ``lambda * L_A + (1 - lambda) * L_B`` on the same batch, one update.

    Both terms are computed from the same examples with independently drawn supports (they come
    from different auxiliary purposes, so independence is structural), and a single Adam update sees
    their weighted sum. No alternation, so "C wins => the terms are complementary" needs no
    expectation argument.
    """
    lam = jnp.asarray(float(getattr(config, "exp03_lambda", 0.5)), dtype=jnp.float32)
    loss_a, aux_a = _corrective_ss_loss(params, state, data, rng, config, scheduler, global_step=global_step)
    loss_b, aux_b = _rollout_loss(params, state, data, rng, config, scheduler, global_step=global_step)
    loss = lam * loss_a + (1.0 - lam) * loss_b
    aux = dict(aux_a)
    # B's diagnostics travel alongside A's; the shared metric names (sigma_mean, v_pred_l2, ...)
    # stay A's, and every B-specific key is already suffixed at its source.
    aux.update({key: value for key, value in aux_b.items() if key not in aux})
    aux["velocity_mse"] = loss
    aux["loss_a"] = loss_a
    aux["loss_b"] = loss_b
    aux["lambda"] = lam
    aux["loss_combined_finite"] = jnp.isfinite(loss).astype(jnp.float32)
    return loss, aux


def _exp03_aux(loss, *, v_pred, v_target, z_state, ctx, sigma, timestep) -> dict:
    """The parent's metric keys, so the training loop logs an arm exactly as it logs the control."""
    return {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(sigma.astype(jnp.float32)),
        "timestep_mean": jnp.mean(timestep.astype(jnp.float32)),
        "v_pred_l2": jnp.linalg.norm(v_pred.astype(jnp.float32)),
        "v_target_l2": jnp.linalg.norm(v_target.astype(jnp.float32)),
        "z_noisy_std": jnp.std(z_state.astype(jnp.float32)),
        "z_target_std": jnp.std(ctx.z_video),
        "z_init_anchor_mse": jnp.mean((z_state[:, :, :1].astype(jnp.float32) - ctx.z_i0[:, :, :1]) ** 2),
    }


EXP03_LOSSES = {
    "corrective_ss": _corrective_ss_loss,
    "rollout_loss": _rollout_loss,
    "combined": _combined_loss,
}

# Derived from the dispatch table, never hand-maintained: what is implemented is what can be run.
EXP03_IMPLEMENTED_OBJECTIVES = (EXP03_CONTROL_OBJECTIVE, *EXP03_LOSSES)


@jax.jit
def _jit_tree_vdot(left, right):
    """Leafwise float32 dot, as ONE compiled graph.

    Jitted, not eager: at 5B an eager leafwise reduction materializes a full product tree before
    anything is freed. Under jit, XLA schedules the whole reduction and keeps only the accumulator.
    """
    return sum(
        jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32))
        for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


@jax.jit
def _jit_grad_stats(grad):
    """Every gradient statistic in ONE compiled pass: sq norm, l2, max-abs, finite-leaf count.

    Replaces four separate eager whole-tree traversals (jaxopt's ``tree_l2_norm`` squared tree, a
    ``tree_reduce`` with a full ``jnp.abs`` copy per leaf, and a per-leaf finiteness sweep). Those
    stacked their temporaries on top of four resident 5B gradients, which is what exhausted HBM in
    Job 8c -- the allocator was asking for 18 MB with 12.64 MB free.
    """
    leaves = jax.tree_util.tree_leaves(grad)
    if not leaves:
        # An empty gradient reports max_abs = 0, DELIBERATELY: the old tree_reduce seeded its fold
        # with -1.0, which is not a magnitude and would read as a real measurement in the artifact.
        # Zero is the honest answer for "no elements" and is what the empty-tree test pins.
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {"sq_norm": zero, "l2_norm": zero, "max_abs": zero, "finite_leaves": jnp.asarray(0, jnp.int32)}
    sq_norm = sum(jnp.sum(leaf.astype(jnp.float32) ** 2) for leaf in leaves)
    max_abs = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf.astype(jnp.float32))) for leaf in leaves]))
    finite_leaves = sum(jnp.all(jnp.isfinite(leaf)).astype(jnp.int32) for leaf in leaves)
    return {
        "sq_norm": sq_norm,
        "l2_norm": jnp.sqrt(sq_norm),
        "max_abs": max_abs,
        "finite_leaves": finite_leaves,
    }


def grad_stats(grad) -> dict:
    """:func:`_jit_grad_stats` with the host-side scalars pulled out (and the static leaf count)."""
    stats = _jit_grad_stats(grad)
    return {
        "sq_norm": float(stats["sq_norm"]),
        "l2_norm": float(stats["l2_norm"]),
        "max_abs": float(stats["max_abs"]),
        "finite_leaves": float(stats["finite_leaves"]),
        "total_leaves": float(len(jax.tree_util.tree_leaves(grad))),
    }


def tree_dot(left, right) -> float:
    """Leafwise float32 dot product — compiled, so nothing whole-tree is materialized on the way."""
    return float(_jit_tree_vdot(left, right))


def tree_sq_norm(tree) -> float:
    """Leafwise squared L2 norm, in one compiled pass."""
    return float(_jit_grad_stats(tree)["sq_norm"])


def grad_cosine(left, right, *, left_sq_norm: float | None = None, right_sq_norm: float | None = None) -> float:
    """Cosine between two gradient pytrees; ``nan`` if either is degenerate.

    The squared norms can be supplied by a caller that already computed them, so a cosine costs one
    compiled dot rather than a dot plus two more whole-tree reductions.
    """
    left_sq = tree_sq_norm(left) if left_sq_norm is None else float(left_sq_norm)
    right_sq = tree_sq_norm(right) if right_sq_norm is None else float(right_sq_norm)
    denominator = math.sqrt(left_sq) * math.sqrt(right_sq)
    return tree_dot(left, right) / denominator if denominator > 0 else float("nan")


class ResidentTrees:
    """Counts how many gradient trees are alive at once, so the cap is a measurement.

    The replay's memory contract is "at most three 5B gradients resident". A comment cannot enforce
    that; this counter records the high-water mark into the report, where the run's own artifact
    shows whether the contract held.
    """

    def __init__(self):
        self.live = 0
        self.peak = 0

    def acquire(self) -> None:
        self.live += 1
        self.peak = max(self.peak, self.live)

    def release(self, count: int = 1) -> None:
        self.live -= count


_LOSS_AND_GRAD_CACHE: dict = {}
COMPILE_TIMINGS: dict = {}


def _timed_first_call(tag: str, compiled):
    """Log the FIRST call's wall time for a jitted tag — its compile cost, on the real machine.

    Compilation dominates a no-update probe's runtime, and the only machine where that number means
    anything is the one the job runs on. One line per tag at first call costs nothing and makes the
    run its own compile-cost measurement, with no separate probe.
    """

    def _wrapper(*args, **kwargs):
        if tag in COMPILE_TIMINGS:
            return compiled(*args, **kwargs)
        started = time.perf_counter()
        result = compiled(*args, **kwargs)
        jax.block_until_ready(result)
        COMPILE_TIMINGS[tag] = time.perf_counter() - started
        max_logging.log(f"[exp03] first call (compile+run) {tag}: {COMPILE_TIMINGS[tag]:.1f}s")
        return result

    _wrapper.jitted = compiled  # so a test can read the real ._cache_size()
    return _wrapper


def _loss_and_grad_fn(name: str, loss_fn, config, scheduler):
    """A jitted ``value_and_grad`` per objective, cached.

    THE fix for Job 8c: the backward pass runs as one compiled graph instead of op by op, so XLA
    schedules and frees it. The value comes from the same call as the gradient -- the previous shape
    ran the whole 5B forward twice per objective, once for the loss and once inside ``jax.grad``.
    """
    key = (name, id(config), id(scheduler))
    if key not in _LOSS_AND_GRAD_CACHE:

        def _call(params, state, data, rng, global_step, fn=loss_fn, objective=name):
            kwargs = {} if objective == "control" else {"global_step": global_step}
            return fn(params, state, data, rng, config, scheduler, **kwargs)

        _LOSS_AND_GRAD_CACHE[key] = (
            _timed_first_call(f"replay_{name}", jax.jit(jax.value_and_grad(_call, has_aux=True))),
            jax.jit(_call),
        )
    return _LOSS_AND_GRAD_CACHE[key]


def exp03_frozen_replay(
    state,
    data,
    rng,
    config,
    scheduler,
    *,
    global_step,
    with_gradients=True,
    include_control=False,
    gauge=None,
) -> dict:
    """NO-UPDATE replay of A, B and C on a frozen state — the discriminator, off the timing path.

    Deliberately NOT part of the training step: separate per-term gradients need extra reverse
    passes, which would change both the cost being measured and the compilation being blamed. This
    runs on a snapshot instead, and reports for each term the loss, whether it is finite, its
    gradient norm, its max-abs gradient, its finite-leaf count, and (for A vs B) the gradient cosine
    -- everything needed to say WHICH term and WHICH pass produced a non-finite value.
    """
    out: dict[str, float] = {"global_step": float(global_step)}
    resident = ResidentTrees()
    objectives = [("a", _corrective_ss_loss), ("b", _rollout_loss), ("c", _combined_loss)]
    if include_control:
        # The plain objective is the reference every cosine is taken against, so it is replayed here
        # rather than in a second pass that could drift from this one.
        objectives.insert(0, ("control", _denoising_loss))

    control_grad = None
    control_sq = None
    a_grad = None
    a_sq = None
    for name, loss_fn in objectives:
        compiled_grad, compiled_value = _loss_and_grad_fn(name, loss_fn, config, scheduler)
        if with_gradients:
            (value, aux), grad = compiled_grad(state.params, state, data, rng, global_step)
            resident.acquire()
        else:
            # Also compiled: a 5B forward run op-by-op is the same memory failure, and running both
            # modes through XLA keeps their values identical rather than differing in the last ULPs.
            value, aux = compiled_value(state.params, state, data, rng, global_step)
            grad = None
        loss = float(value)
        out[f"loss_{name}"] = loss
        out[f"loss_{name}_finite"] = float(math.isfinite(loss))
        # B's RAW endpoint MSE and its fp32 horizon travel with the normalized loss, so the
        # raw/normalized pair the plan promises is actually in the artifact.
        for extra in (
            "raw_endpoint_mse",
            "horizon_sq",
            "sigma_hi_b",
            "sigma_lo_b",
            "s_b",
            "e_b",
            "sigma_hi_a",
            "sigma_lo_a",
            "s_a",
            "e_a",
            "coin",
            "take_self_generated",
            "p_ss",
            "k_a",
        ):
            if extra in aux:
                out[f"{extra}_{name}"] = float(aux[extra])
        if not with_gradients:
            continue

        if gauge is not None:
            # INSIDE the peak window: this objective's gradient has just materialized and every
            # tree the incremental scheme still holds is live. Sampling after the release would
            # measure the trough and prove nothing.
            gauge.sample(f"replay_{name}")
        stats = grad_stats(grad)
        out[f"grad_norm_{name}"] = stats["l2_norm"]
        out[f"grad_sq_norm_{name}"] = stats["sq_norm"]
        out[f"grad_max_abs_{name}"] = stats["max_abs"]
        out[f"grad_finite_leaves_{name}"] = stats["finite_leaves"]
        out[f"grad_total_leaves_{name}"] = stats["total_leaves"]

        # INCREMENTAL cosines, so no more than three gradients are ever resident: the control is
        # kept as the reference, A is kept only until B can be compared against it, and every other
        # tree is consumed and dropped in the same iteration that produced it.
        if name == "control":
            control_grad, control_sq = grad, stats["sq_norm"]
            continue
        if control_grad is not None:
            out[f"grad_cosine_{name}_vs_control"] = grad_cosine(
                grad, control_grad, left_sq_norm=stats["sq_norm"], right_sq_norm=control_sq
            )
        if name == "a":
            a_grad, a_sq = grad, stats["sq_norm"]
            continue
        if name == "b" and a_grad is not None:
            out["grad_cosine_ab"] = grad_cosine(grad, a_grad, left_sq_norm=stats["sq_norm"], right_sq_norm=a_sq)
            del a_grad
            a_grad = None
            resident.release()
        del grad
        resident.release()

    if a_grad is not None:  # no B in this replay: A was never consumed
        del a_grad
        resident.release()
    if control_grad is not None:
        del control_grad
        resident.release()
    out["grad_trees_peak_resident"] = float(resident.peak)
    return out


class Exp03Trainer(WanTI2VOverfit100Trainer):
    """The exp_02 overfit100 trainer with a switchable objective (``model_type: EXP03_TI2V``).

    The subclass exists for provenance as much as for behaviour: a run recorded as ``EXP03_TI2V``
    can be told apart from an exp_02 run in every artifact, even when — as with ctrl0 — it is
    deliberately executing the identical code.
    """

    def _loss_and_step_fns(self):
        objective = validate_exp03_config(self.config)
        if objective == EXP03_CONTROL_OBJECTIVE:
            # BY IDENTITY, not by copy: ctrl0's job is to reproduce exp_02, and the only way to be
            # sure it does is to run the same functions.
            return _denoising_loss, _train_step
        loss_fn = EXP03_LOSSES.get(objective)
        if loss_fn is None:
            raise NotImplementedError(
                f"exp03_objective={objective!r} is declared by plan v3.1 but has no implementation; "
                f"implemented objectives are {sorted(('control', *EXP03_LOSSES))}. Refusing to start rather "
                f"than silently training the control under another arm's run name."
            )
        return loss_fn, _make_train_step(loss_fn)

    def start_training(self):
        # Fail on a bad objective/knob before the ~5B load, and say which one is running.
        objective = validate_exp03_config(self.config)
        if jax.process_index() == 0:
            max_logging.log(f"[wan_exp03] objective={objective} (model_type={self.config.model_type})")
            if objective == EXP03_CONTROL_OBJECTIVE:
                max_logging.log(
                    "[wan_exp03] control arm: the parent's plain one-step objective, by identity -- "
                    "this run is a replication of exp_02's recipe through the exp_03 trainer"
                )
        self._loss_and_step_fns()  # raises here, not 20 minutes in, for an unimplemented trial
        return super().start_training()
