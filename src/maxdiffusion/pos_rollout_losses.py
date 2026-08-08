"""exp_06 `rollout_adapter` — the R-B endpoint loss kernel, exp_06-owned and pinned to exp_03.

Plan §3 makes **R-B, short-horizon differentiable rollout at k=2**, the primary arm, and names
exp_03's construction as the validated one: a fixed-k unroll of THE sampler the evaluation runs
(`lax.scan` + `jax.remat`), scored at the endpoint against the ideal trajectory point built from the
SAME epsilon, divided by ``(sigma_hi - sigma_lo)**2``. That normalizer is not cosmetic: without it
the non-uniform sigma grid would reweight the loss by the square of the step size, and the optimum
``v = eps - z_gt`` would not give zero at every support.

**What this round did** (plan §5-2, T2):

* **Extracted** exp_03's ``_rollout_loss`` (pin lines 435-510) into :func:`rollout_endpoint_loss`,
  together with the two private helpers it stands on, ``_interpolant_at`` (245-248) and
  ``_exp03_aux`` (537-548).
* **Re-homed** ``masked_velocity_mse`` and ``build_noisy_pinned_latents``. Both exist in
  ``side_adapter_wan.py`` AT THE PIN but NOT in exp_06's inherited tree, where the side-adapter
  trainer computes the same math inline. The Planner's ruling is that they live HERE — no
  dual-touch edit to the inherited ``side_adapter_wan.py`` — and that they carry a **double
  equivalence obligation**: bitwise-equal to exp_03's pinned construction AND to exp_06's own inline
  trainer math. Both are proven on shared fixtures in
  ``tests/worklogs_yixun/test_pos_rollout_losses.py``; they agree, so the two branches' losses were
  and are the same function. ``masked_velocity_mse_per_example`` was NOT re-homed: the kernel does
  not need it (take the minimum).
* **Removed the config coupling.** exp_03's ``_rollout_loss`` read ``seed``/``k_b``/``salt`` through
  three-argument ``getattr(config, ...)`` (pin lines 448-455). That is forbidden here (issue #11:
  a pyconfig ``HyperParameters.__getattr__`` raises ``ValueError``, so the default never applies —
  it killed two TPU jobs last campaign). Every one of those is an EXPLICIT ARGUMENT below, and this
  module performs no config access of any kind; a test pins that.

**What this round deliberately did NOT do.** The kernel takes ``velocity_fn`` as an argument and
applies **no** ``stop_gradient`` anywhere. The §3a CFG gradient contract — which branches see the
rollout state's gradient, and the frozen/adapter split — belongs to T3a's step; a kernel-level
stop-grad would silently pre-empt that contract before it is written. A test pins the absence.

**CARRY-FORWARD FOR T3b — the latents, the noise and the sigmas must arrive in float32.** The double
equivalence above is proven on the domain the two existing callers actually use: both prologues cast
``z_i0``/``z_video``/``eps`` to float32 before any loss math. Outside it the two sources genuinely
DISAGREE, and the measured boundary is sharper than "bf16 is bad": exp_03's re-homed construction
casts to float32 internally and exp_06's inline math does not, but JAX *promotes* a bf16 latent
before multiplying, so the two stay bitwise equal as long as the SIGMA is float32. They part company
once a bf16 sigma meets a bf16 latent — ``1.0 - sigma_b`` then rounds in bf16 on one side and
float32 on the other (measured ~5e-4 to 1e-2, and at all-bf16 the two outputs differ in dtype as
well). T3b is a NEW caller and nothing structural would have stopped it passing bf16, so the
precondition is **enforced, not documented**: :func:`build_noisy_pinned_latents` and
:func:`rollout_endpoint_loss` raise on any non-float32 latent/noise/sigma input. The guard covers
all four rather than only the sigma, because "inside the proven domain" is the contract worth
stating; note that within :func:`rollout_endpoint_loss` the divergence is in fact structurally
unreachable (``interpolant_at`` rebuilds the sigma in float32), so the entry-point check is a
contract guard while the one in :func:`build_noisy_pinned_latents` closes the reachable hazard for a
caller that uses the helper directly.

These preconditions, and the config-coupling removal above, are the ONLY deliberate divergences from
the pinned bodies — checks added ahead of arithmetic that is left verbatim, so outputs on the proven
domain are unchanged (the equivalence tests still assert that bitwise) and both are recorded in
:data:`EXP03_T2_EXTRACTED_SYMBOLS`. ``weights_dtype`` is a DIFFERENT thing and stays free: it is the
dtype the rollout STATE is carried in (bf16 in production, matching the deployed evaluator), not the
dtype of the loss inputs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_step
from maxdiffusion.models.wan.side_adapter_wan import apply_first_frame_pin

# Re-exported so a reader of THIS module sees the pin without chasing it; the constant itself lives
# in pos_rollout_support (single source of truth) and is pinned by T1's tests.
from maxdiffusion.pos_rollout_support import EXP03_SOURCE_COMMIT

__all__ = [
    "EXP03_SOURCE_COMMIT",
    "EXP03_T2_EXTRACTED_SYMBOLS",
    "build_noisy_pinned_latents",
    "interpolant_at",
    "masked_velocity_mse",
    "rollout_endpoint_loss",
]

# Provenance of what T2 lifted, in the same form T1 used. ``sha256`` is of the exact source segment
# at exp_03 @ 2ef9b8a (``ast.get_source_segment``), recorded so a future re-pin is a checkable diff.
# It is a RECORD, not a gate: no hermetic test in exp_06's tree can reach exp_03's git objects, so
# the behavioural pin is the equivalence + analytic oracles in the T2 tests (see T1's M15).
_EXP03_TRAINER = "src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py"
_EXP03_SIDE_ADAPTER = "src/maxdiffusion/models/wan/side_adapter_wan.py"
EXP03_T2_EXTRACTED_SYMBOLS = {
    "rollout_endpoint_loss": {
        "source_symbol": "_rollout_loss",
        "source_path": _EXP03_TRAINER,
        "source_lines": (435, 510),
        "sha256": "a4e4c8d8e7424b255ee37ca195beabf0e844c22d7d1540ef776f47d3b89fa23a",
        # Deliberate, reviewed divergences from the pinned body. The arithmetic is verbatim; these
        # are the config-coupling removal and the added preconditions (module docstring).
        "divergences": (
            "config reads replaced by explicit arguments",
            "float32 precondition enforced",
            "support window supplied by the caller rather than derived (T3b-2 review, BLOCKER 1)",
        ),
    },
    "interpolant_at": {
        "source_symbol": "_interpolant_at",
        "source_path": _EXP03_TRAINER,
        "source_lines": (245, 248),
        "sha256": "dc90e351d1a0a2f2002c526f426cfdd4381ca5d2bd98e0c4ed6cfd1463cc8281",
    },
    "_endpoint_aux": {
        "source_symbol": "_exp03_aux",
        "source_path": _EXP03_TRAINER,
        "source_lines": (537, 548),
        "sha256": "796ed07cf0ce0a3c0818f98113486fc7fcacc8dd823bb92114c71d86a861b1fb",
    },
    "build_noisy_pinned_latents": {
        "source_symbol": "build_noisy_pinned_latents",
        "source_path": _EXP03_SIDE_ADAPTER,
        "source_lines": (537, 558),
        "sha256": "2be54924c28388567d151201f288b691ec25043dfdc947c08d869d8b2d37c397",
        # THE dtype-sensitive one: this is the construction that diverges from exp_06's inline math
        # outside float32, so the precondition is enforced here rather than assumed.
        "divergences": ("float32 precondition enforced",),
    },
    "masked_velocity_mse": {
        "source_symbol": "masked_velocity_mse",
        "source_path": _EXP03_SIDE_ADAPTER,
        "source_lines": (561, 580),
        "sha256": "ba758b4d95dca179a9bf6017a753cd8c21d3059f2cc68b2bf9c0e6715c0a6d4a",
    },
}


# =============================================================================================
# Re-homed from side_adapter_wan.py AT THE PIN. Bodies verbatim; both carry the double
# equivalence obligation (vs the pin, and vs exp_06's inline trainer math).
# =============================================================================================


def _require_float32(**arrays) -> None:
    """The proven-equivalence precondition, enforced loudly (see the module docstring).

    exp_03's construction casts to float32 internally and exp_06's inline trainer math does not, so
    the two are the same function ONLY where the caller has already cast. Rather than leave that an
    unwritten property of today's two callers, any other dtype is refused here — a future caller
    that would have landed silently outside the proven regime fails at its first trace instead.
    """
    for name, array in arrays.items():
        dtype = jnp.asarray(array).dtype
        if dtype != jnp.float32:
            raise ValueError(
                f"{name} must be float32, got {dtype}: the re-homed construction is proven equivalent to "
                "exp_03's pin AND to exp_06's inline trainer math only on float32 inputs (they diverge in "
                "bf16/fp16). Cast in the caller's prologue, as both existing callers do."
            )


def build_noisy_pinned_latents(
    z_video_f32: jax.Array,
    z_i0_f32: jax.Array,
    eps: jax.Array,
    sigma_t: jax.Array,
) -> jax.Array:
    """Flow-matching noisy latents ``z_t`` with latent frame 0 pinned.

    THE shared objective construction. Computes ``z_t = (1 - sigma_t) * z_video + sigma_t * eps`` in
    float32, then pins frame 0 to the image-conditioning latent via :func:`apply_first_frame_pin`.
    ``eps`` is an explicit argument so the caller owns the noise policy (``fresh``/``fixed``).

    Shapes: ``z_video_f32``/``eps`` ``[B, C, F, H, W]``, ``z_i0_f32`` ``[B, C, 1, H, W]``,
    ``sigma_t`` ``[B]``; all four float32 (enforced), returns float32.
    """
    _require_float32(z_video_f32=z_video_f32, z_i0_f32=z_i0_f32, eps=eps, sigma_t=sigma_t)
    b = z_video_f32.shape[0]
    sigma_b = sigma_t.astype(jnp.float32).reshape((b, 1, 1, 1, 1))
    z_t_f32 = (1.0 - sigma_b) * z_video_f32.astype(jnp.float32) + sigma_b * eps.astype(jnp.float32)
    return apply_first_frame_pin(z_t_f32, z_i0_f32.astype(jnp.float32))


def masked_velocity_mse(v_pred: jax.Array, v_target: jax.Array, batch_size: int) -> jax.Array:
    """Frame-0-masked flow-matching MSE (THE shared objective's reduction).

    Latent frame 0 is pinned to the image condition and carries no learning signal, so it is masked
    out. ``n_valid`` counts the non-frame-0 elements of ONE example times ``batch_size`` (floored at
    1), so the result is the mean squared error over unmasked positions — equivalently the mean over
    examples of each example's masked MSE. The reduction is float32. ``v_pred``/``v_target`` are
    ``[B, C, F, H, W]`` and must have identical shapes (a broadcastable-but-malformed prediction is
    rejected rather than silently mis-normalized); the mask is built from ``v_target``, matching the
    pre-refactor reference. Returns a float32 scalar.
    """
    if v_pred.shape != v_target.shape:
        raise ValueError(f"masked_velocity_mse: v_pred shape {v_pred.shape} != v_target shape {v_target.shape}")
    mask = jnp.ones((1, *v_target.shape[1:]), dtype=jnp.float32)
    mask = mask.at[:, :, :1, :, :].set(0.0)
    diff = (v_pred.astype(jnp.float32) - v_target.astype(jnp.float32)) * mask
    n_valid = jnp.maximum(jnp.sum(mask) * batch_size, 1.0)
    return jnp.sum(diff**2) / n_valid


# =============================================================================================
# The kernel, extracted from exp_03's ``_rollout_loss`` with ctx unpacked into explicit arguments.
# =============================================================================================


def interpolant_at(z_video_f32, z_i0_f32, eps_f32, sigmas, index) -> jax.Array:
    """``pin((1 - sigma[index]) * z_gt + sigma[index] * eps)`` — the teacher-forced state.

    ``index`` may be traced. exp_03 read the batch size off its ctx; here it comes from the target's
    own leading dimension, which is the same number.
    """
    sigma = sigmas[index].astype(jnp.float32)
    b = z_video_f32.shape[0]
    return build_noisy_pinned_latents(z_video_f32, z_i0_f32, eps_f32, jnp.full((b,), sigma))


def _endpoint_aux(loss, *, v_pred, v_target, z_state, z_video_f32, z_i0_f32, sigma, timestep) -> dict:
    """exp_02's metric key set, so the loop logs this arm exactly as it logs the control."""
    return {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(sigma.astype(jnp.float32)),
        "timestep_mean": jnp.mean(timestep.astype(jnp.float32)),
        "v_pred_l2": jnp.linalg.norm(v_pred.astype(jnp.float32)),
        "v_target_l2": jnp.linalg.norm(v_target.astype(jnp.float32)),
        "z_noisy_std": jnp.std(z_state.astype(jnp.float32)),
        "z_target_std": jnp.std(z_video_f32),
        "z_init_anchor_mse": jnp.mean((z_state[:, :, :1].astype(jnp.float32) - z_i0_f32[:, :, :1]) ** 2),
    }


def rollout_endpoint_loss(
    *,
    z_video_f32: jax.Array,
    z_i0_f32: jax.Array,
    eps_f32: jax.Array,
    sigmas: jax.Array,
    timesteps: jax.Array,
    context: jax.Array,
    velocity_fn,
    weights_dtype,
    num_train_timesteps,
    support_start,
    support_end,
    k_b: int,
) -> tuple[jax.Array, dict]:
    """R-B — the short-horizon rollout loss, horizon-normalized.

    Teacher-forced start at ``sigma[s]``, then ``k_B`` steps of the shared sampler with gradients
    flowing through every forward (``lax.scan`` with ``jax.remat`` per step, so the unroll is
    rematerialized rather than held), scored against the ideal trajectory point built from the SAME
    epsilon and divided by ``(sigma_hi - sigma_lo)**2``. With that normalizer the optimum
    ``v = eps - z_gt`` gives exactly zero at every support, and the loss of a constant velocity
    offset ``c`` is exactly ``mean(c**2)`` regardless of which support was drawn — the two analytic
    identities the T2 oracles check.

    ``velocity_fn(hidden_states, timestep, encoder_hidden_states)`` is the caller's: the eval closes
    over the frozen transformer, T3a closes over the CFG branch under the §3a gradient contract.
    Nothing here differentiates or stop-gradients on the caller's behalf. ``k_b`` is static (it is
    the scan length). The latents, the noise and the sigma grid must be float32 (module docstring);
    ``weights_dtype`` — the dtype the rollout state is carried in — is unconstrained.

    **The support is SUPPLIED, not derived here (T3b-2 review, BLOCKER 1).** This kernel used to take
    ``seed``/``global_step``/``support_salt`` and call ``rollout_support`` itself. That let a caller
    hand it an epsilon from one step's draw while the kernel silently redrew the support from
    another — the two halves of one ``StepDraws`` coming apart without any test being able to see it.
    Taking ``support_start``/``support_end`` as explicit arguments is the same move the plan already
    made for ``seed``/``k``/``salt`` (issue #11): **the arithmetic is unchanged**, and the round's
    equivalence tests pass in exactly the values the old derivation produced.
    """
    if support_start is None or support_end is None:
        raise ValueError(
            "rollout_endpoint_loss needs its support window from the caller's StepDraws: the kernel no "
            "longer derives one, so that the epsilon and the support of a step cannot come apart"
        )
    _require_float32(z_video_f32=z_video_f32, z_i0_f32=z_i0_f32, eps_f32=eps_f32, sigmas=sigmas)
    start, end = support_start, support_end
    b = z_video_f32.shape[0]
    sigma_hi = sigmas[start].astype(jnp.float32)
    sigma_lo = sigmas[end].astype(jnp.float32)

    @jax.remat
    def _step(carry, offset):
        z, index = carry
        z_next = overfit100_sampler_step(
            z,
            index,
            velocity_fn=velocity_fn,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0_f32.astype(weights_dtype),
        )
        del offset
        return (z_next, index + 1), None

    z_start = interpolant_at(z_video_f32, z_i0_f32, eps_f32, sigmas, start).astype(weights_dtype)
    (z_end, _), _ = jax.lax.scan(_step, (z_start, start), xs=jnp.arange(int(k_b)))

    z_ideal = interpolant_at(z_video_f32, z_i0_f32, eps_f32, sigmas, end)
    horizon = jnp.maximum((sigma_hi - sigma_lo) ** 2, jnp.finfo(jnp.float32).tiny)
    raw = masked_velocity_mse(z_end.astype(jnp.float32), z_ideal, b)
    loss = raw / horizon
    aux = _endpoint_aux(
        loss,
        v_pred=z_end.astype(jnp.float32),
        v_target=z_ideal,
        z_state=z_end.astype(jnp.float32),
        z_video_f32=z_video_f32,
        z_i0_f32=z_i0_f32,
        sigma=sigma_lo,
        timestep=sigma_lo * jnp.asarray(num_train_timesteps, dtype=jnp.float32),
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
