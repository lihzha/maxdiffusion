"""THE single-step sampler of the overfit100 rollout — one definition, shared by eval and training.

exp_03's whole premise is that the training objective and the evaluation trajectory must run the
*same* operator, so this module exists to make "the same" a property of the code rather than a claim
about two copies. The eval rollouts in ``generate_wan_side_adapter.py`` call
:func:`overfit100_sampler_step`; the exp_03 trainers unroll the identical function under
``jax.grad``/``jax.remat``; the sigma-trajectory diagnostic traces it. There is no second
implementation to drift.

**Provenance.** Every line here was moved verbatim out of ``_rollout_overfit100_sample`` (and its
exp_01 sibling ``_rollout_sample``) — the same grid construction, the same per-token timestep with
``n_hist=1``, the same ``(sigma[i+1] - sigma[i]).astype(z.dtype)`` scaling in the *latent's* dtype,
the same frame-0 pin after the update. The eval path's bitwise reproducibility is a measured asset
(the exp_02 sampling probe reproduced the landed 30-window scalars to 0.0), so the motion is pinned
by an exact-equality parity test against a verbatim copy of the pre-extraction step, and by an
AST test that no duplicate survives in the evaluator
(``tests/worklogs_yixun/test_exp03_sampler_extraction.py``).

**Trainer contract.** The step takes explicit arguments and returns state — no module-level state,
no config object, no Python side effects (no logging, no host callbacks, no RNG drawn inside), so it
is safe to ``scan`` and ``remat`` over. The velocity model enters as ``velocity_fn``, a callable of
``(hidden_states, timestep, encoder_hidden_states)``: the eval closes over the frozen transformer
with ``deterministic=True``, the adapter path closes over its CFG branch, and a trainer closes over
the differentiated parameters. What the caller does inside that callable is the caller's business;
what happens *around* it is defined exactly once, here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    apply_first_frame_pin,
    build_rollout_sigmas,
    rollout_timesteps_from_sigmas,
)

# Latent frame 0 is the image condition: it is pinned, and its tokens carry timestep 0.
OVERFIT100_N_HIST = 1


def overfit100_sampler_grid(
    *,
    num_inference_steps,
    flow_shift,
    sigma_min,
    sigma_max,
    num_train_timesteps,
) -> tuple[jax.Array, jax.Array]:
    """``(sigmas, timesteps)`` for the rollout: ``N+1`` descending sigmas, ``N`` timesteps.

    The bookkeeping that wraps the step, moved with it so the trials draw their sigma supports from
    the very grid the eval advances along. ``sigmas`` ends at exactly 0 (the terminal boundary every
    exp_03 draw excludes); ``timesteps[i] = sigmas[i] * num_train_timesteps``.
    """
    sigmas = build_rollout_sigmas(num_inference_steps, flow_shift, sigma_min, sigma_max)
    timesteps = rollout_timesteps_from_sigmas(sigmas, num_train_timesteps)
    return sigmas, timesteps


def overfit100_step_timestep(
    timesteps: jax.Array,
    index,
    batch_size: int,
    f_lat: int,
    h_lat: int,
    w_lat: int,
) -> jax.Array:
    """The ``[b, seq]`` per-token timestep the step at grid ``index`` feeds the transformer."""
    step_t = jnp.broadcast_to(timesteps[index], (batch_size,))
    return _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=OVERFIT100_N_HIST)


def overfit100_euler_update(
    z: jax.Array,
    v: jax.Array,
    sigmas: jax.Array,
    index,
    z_i0: jax.Array,
) -> jax.Array:
    """``pin(z + (sigma[index+1] - sigma[index]) * v)`` — the update, without a network.

    Exposed on its own because trial A's corrective target is derived from this rule: it implies the
    contraction ``z_next - z_gt = (sigma_next / sigma) * (z - z_gt)``, which is what makes
    ``v* = (z - z_gt)/sigma`` the exact one-step correction. The scale is cast to the *latent's*
    dtype before multiplying, exactly as the landed eval does.
    """
    return apply_first_frame_pin(z + (sigmas[index + 1] - sigmas[index]).astype(z.dtype) * v, z_i0)


def overfit100_sampler_step(
    z: jax.Array,
    index,
    *,
    velocity_fn,
    sigmas: jax.Array,
    timesteps: jax.Array,
    context: jax.Array,
    z_i0: jax.Array,
) -> jax.Array:
    """One sampler step: build the timestep, predict the velocity, Euler-update, pin frame 0.

    ``index`` may be a traced integer (the eval calls this from ``jax.lax.fori_loop``); the shapes
    come from ``z``, so nothing about the batch is baked in elsewhere.
    """
    batch_size, _, f_lat, h_lat, w_lat = z.shape
    timestep_2d = overfit100_step_timestep(timesteps, index, batch_size, f_lat, h_lat, w_lat)
    v = velocity_fn(
        hidden_states=z,
        timestep=timestep_2d,
        encoder_hidden_states=context,
    )
    return overfit100_euler_update(z, v, sigmas, index, z_i0)
