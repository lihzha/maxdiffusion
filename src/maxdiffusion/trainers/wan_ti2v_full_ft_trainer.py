"""WAN TI2V full-finetune overfit diagnostic -- objective, state, and steps.

This is the side-adapter trainer *minus the adapter* (plan_full_ft_overfit §2):
the Wan2.2 TI2V 5B transformer is fully trainable (all its params go in the
optimizer, no adapter exists) while the data path, noise schedule, one-step
flow-matching objective, and conditioning stay identical to the adapter runs.
The purpose is to measure how fast a fully-trainable backbone can memorize the
cached-DROID train split, isolating whether the pipeline is learnable at all
versus whether frozen-backbone + adapter optimization is the bottleneck.

Parity with the side-adapter run is enforced *by shared code*, not by
re-implementation: the noisy-latent construction and masked velocity MSE come
from ``side_adapter_wan`` (``build_noisy_pinned_latents`` / ``masked_velocity_mse``),
and the noise / step-index samplers are imported from the side-adapter trainer.
The only differences here are the experiment variable itself: the whole
transformer is the trainable ``params``, there is exactly one plain
``transformer(...)`` forward (no adapter, no actions), and the CFG branch is
bypassed (``guide_scale`` is asserted to 1.0 in the round-3 trainer class).

This module (round 2) provides only ``FullFTTrainState``, ``_denoising_loss``,
and ``_train_step`` / ``_eval_step``. The ``WanTI2VFullFTTrainer`` class, its
startup asserts, dtype/byte logging, sharding override, and dispatch land in
round 3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxopt
from flax import nnx
from flax.training import train_state

from maxdiffusion.models.wan.side_adapter_wan import (
    build_noisy_pinned_latents,
    build_rollout_sigmas,
    masked_velocity_mse,
    _build_per_token_timestep,
    _dtype,
)
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import _build_noise, _sample_step_indices


class FullFTTrainState(train_state.TrainState):
    """Train state whose trainable ``params`` are the full WAN transformer.

    Unlike the side-adapter ``TrainState`` (which carries a frozen transformer
    alongside trainable adapter params), here the transformer *is* the trainable
    module: ``params`` holds its ``nnx.Param`` leaves and gets optimizer state,
    ``graphdef`` / ``rest_of_state`` are its static graph and non-Param state, and
    ``null_context`` is the reused null text embedding.
    """

    graphdef: nnx.GraphDef
    rest_of_state: nnx.State
    null_context: jax.Array


def _denoising_loss(
    params,
    state: FullFTTrainState,
    data: dict,
    rng: jax.Array,
    config,
    scheduler: FlaxFlowMatchScheduler,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One-step flow-matching velocity MSE for the fully-trainable transformer.

    Mirrors ``wan_ti2v_side_adapter_trainer._denoising_loss`` but with no adapter,
    no actions, and no CFG branch: a single plain ``transformer(...)`` forward on
    the frame-0-pinned noisy latents against the null text context. The
    interpolation/pin math stays float32; only the transformer input is cast to
    ``weights_dtype``.
    """
    noise_rng, step_rng, dropout_rng = jax.random.split(rng, 3)
    weights_dtype = _dtype(config.weights_dtype)
    activations_dtype = _dtype(config.activations_dtype)
    bsz = config.global_batch_size_to_train_on

    transformer = nnx.merge(state.graphdef, params, state.rest_of_state)

    z_i0_f32 = data["z_i0"][:bsz].astype(jnp.float32)
    z_video_f32 = data["z_video"][:bsz].astype(jnp.float32)

    b, _, f_lat, h_lat, w_lat = z_video_f32.shape
    num_steps = int(config.side_adapter_sampling_steps)
    sigmas = build_rollout_sigmas(
        num_steps,
        config.flow_shift,
        scheduler.config.sigma_min,
        scheduler.config.sigma_max,
    )
    t_idx = _sample_step_indices(step_rng, b, num_steps, sigmas, config)
    sigma_t = sigmas[t_idx]
    step_t = sigma_t * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    null_context = jnp.broadcast_to(
        state.null_context.astype(activations_dtype),
        (b, state.null_context.shape[1], state.null_context.shape[2]),
    )

    eps = _build_noise(noise_rng, z_video_f32.shape, jnp.float32, config)
    z_t_f32 = build_noisy_pinned_latents(z_video_f32, z_i0_f32, eps, sigma_t)

    v_pred = transformer(
        hidden_states=z_t_f32.astype(weights_dtype),
        timestep=timestep_2d,
        encoder_hidden_states=null_context,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )

    v_target = eps - z_video_f32
    loss = masked_velocity_mse(v_pred, v_target, b)
    aux = {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(sigma_t.astype(jnp.float32)),
        "timestep_mean": jnp.mean(step_t.astype(jnp.float32)),
        "v_pred_l2": jnp.linalg.norm(v_pred.astype(jnp.float32)),
        "v_target_l2": jnp.linalg.norm(v_target.astype(jnp.float32)),
        "z_noisy_std": jnp.std(z_t_f32),
        "z_target_std": jnp.std(z_video_f32),
        "z_init_anchor_mse": jnp.mean((z_t_f32[:, :, :1] - z_i0_f32[:, :, :1]) ** 2),
    }
    return loss, aux


def _train_step(state: FullFTTrainState, data: dict, rng: jax.Array, scheduler, config):
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        return _denoising_loss(params, state, data, loss_rng, config, scheduler)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    max_abs_grad = jax.tree_util.tree_reduce(
        lambda m, arr: jnp.maximum(m, jnp.max(jnp.abs(arr))), grads, initializer=-1.0
    )
    state = state.apply_gradients(grads=grads)
    metrics = {
        "scalar": {
            "learning/loss": loss,
            "learning/velocity_mse": aux["velocity_mse"],
            "learning/grad_norm": grad_norm,
            "learning/max_abs_grad": max_abs_grad,
            "learning/sigma_mean": aux["sigma_mean"],
            "learning/timestep_mean": aux["timestep_mean"],
            "learning/v_pred_l2": aux["v_pred_l2"],
            "learning/v_target_l2": aux["v_target_l2"],
            "learning/z_noisy_std": aux["z_noisy_std"],
            "learning/z_target_std": aux["z_target_std"],
            "learning/z_init_anchor_mse": aux["z_init_anchor_mse"],
        },
        "scalars": {},
    }
    return state, metrics, rng


def _eval_step(state: FullFTTrainState, data: dict, rng: jax.Array, scheduler, config):
    losses = jnp.zeros((config.global_batch_size_to_train_on,), dtype=jnp.float32)
    loss, aux = _denoising_loss(state.params, state, data, rng, config, scheduler)
    losses = losses.at[:].set(loss)
    metrics = {
        "scalar": {
            "learning/eval_loss": losses,
            "learning/eval_sigma_mean": aux["sigma_mean"],
            "learning/eval_v_pred_l2": aux["v_pred_l2"],
            "learning/eval_v_target_l2": aux["v_target_l2"],
            "learning/eval_z_noisy_std": aux["z_noisy_std"],
            "learning/eval_z_target_std": aux["z_target_std"],
        },
        "scalars": {},
    }
    return metrics, rng
