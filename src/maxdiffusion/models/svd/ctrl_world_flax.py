# Copyright 2026 Princeton. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Training-time forward pass for the action-conditioned SVD world model.

JAX / Flax port of ``Ctrl-World/models/ctrl_world.py::CrtlWorld.forward``.

The UNet + VAE + image encoder weights are loaded via the existing
:class:`SVDCheckpointer`; only the action encoder is introduced as a new
trainable module. This file contains the pure-functional EDM training step
so it can be wrapped in ``jax.jit`` / ``jax.pmap`` / ``jax.value_and_grad``
without forcing a specific training harness.

High-level recipe (matches Ctrl-World exactly):

    1. Noise the first future latent with a small (0..0.2) sigma and tile
       it across the frame axis as the channel-concat conditioning stream
       (optionally zero-ing the ``num_history`` leading slots).
    2. Encode the per-frame action sequence with
       :class:`FlaxActionEncoder`; apply 5% classifier-free-guidance
       dropout on the whole action hidden state.
    3. Draw an EDM sigma per sample (``log sigma = P_mean + P_std * N(0,1)``,
       P_mean=0.7, P_std=1.6). Produce the ``c_skip / c_out / c_in /
       c_noise`` preconditioning weights.
    4. Noise history latents with an independent, weaker sigma (N(0, 0.3))
       and stitch ``[noisy_history, c_in * noisy_future]`` on the frame axis.
    5. Channel-concat with ``condition_latent / vae_scaling_factor``,
       forward the UNet, EDM-undo to ``predict_x0``, compute the
       ``(sigma**2 + 1)/sigma**2``-weighted MSE loss on the future slots
       only.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from ...models.embeddings_flax import svd_micro_cond_embed
from .action_encoder_flax import FlaxActionEncoder


def probe_add(probe, name: str, x) -> None:
    """Record a forward intermediate for NaN localisation; no-op when probe is None.

    Defined here rather than imported from max_utils: max_utils imports
    models.attention_flax, so importing it from a models module risks a cycle.
    """
    if probe is not None and hasattr(x, "shape"):
        probe.append((name, x))


@dataclass
class CtrlWorldTrainConfig:
    """Knobs for the EDM training objective. Defaults match Ctrl-World."""

    num_history: int = 6
    num_frames: int = 5
    action_dim: int = 7
    hidden_size: int = 1024
    text_embed_dim: Optional[int] = 512

    # EDM log-sigma prior for the future slots.
    p_mean: float = 0.7
    p_std: float = 1.6

    # Max noise added to the conditioning image latent (uniform [0, cond_aug_max]).
    cond_aug_max: float = 0.2
    # Std of the Gaussian noise-aug on history latents; note Ctrl-World uses
    # an *unbounded* Gaussian (``randn * 0.3``) here, not a clipped uniform.
    history_sigma_std: float = 0.3

    # Classifier-free-guidance dropout probability on the action hidden state.
    cfg_drop_prob: float = 0.05

    # Micro-cond scalars for the 768-dim ADM vector (fps / motion / cond_aug).
    fps_id: float = 7.0
    motion_bucket_id: float = 127.0
    noise_aug_strength: float = 0.0

    # Zero out the history slots of the channel-concat stream (Ctrl-World
    # ``his_cond_zero``). Default False matches the released training config.
    his_cond_zero: bool = False


def _build_concat_stream(
    rng: jax.Array,
    future_latent_first: jnp.ndarray,
    num_total_frames: int,
    num_history: int,
    cond_aug_max: float,
    his_cond_zero: bool,
) -> jnp.ndarray:
    """Build the channel-concat conditioning stream for one training step.

    Tiles the noise-augmented first-future-frame latent across all
    ``num_total_frames`` slots. Shape out: ``(B, num_total_frames, 4, H/8, W/8)``.
    """
    b, c, h, w = future_latent_first.shape
    sigma_aug = jax.random.uniform(rng, (b, 1, 1, 1)) * cond_aug_max
    c_in = 1.0 / jnp.sqrt(sigma_aug ** 2 + 1.0)
    noise_rng, _ = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, future_latent_first.shape)
    cond = c_in * (future_latent_first + noise * sigma_aug)
    cond = jnp.broadcast_to(cond[:, None], (b, num_total_frames, c, h, w))
    if his_cond_zero and num_history > 0:
        zero_slice = jnp.zeros_like(cond[:, :num_history])
        cond = jnp.concatenate([zero_slice, cond[:, num_history:]], axis=1)
    return cond


def _edm_preconditioning(sigma: jnp.ndarray):
    """Return ``(c_skip, c_out, c_in, c_noise, loss_weight)`` for a given sigma.

    Sigma is expected to be broadcastable to ``(B, 1, 1, 1, 1)``; scalar arr
    is accepted. Matches sgm's ``EDMDenoiser``.
    """
    c_skip = 1.0 / (sigma ** 2 + 1.0)
    c_out = -sigma / jnp.sqrt(sigma ** 2 + 1.0)
    c_in = 1.0 / jnp.sqrt(sigma ** 2 + 1.0)
    c_noise = jnp.log(sigma) / 4.0
    loss_weight = (sigma ** 2 + 1.0) / (sigma ** 2)
    return c_skip, c_out, c_in, c_noise, loss_weight


def _apply_cfg_dropout(
    rng: jax.Array,
    action_hidden: jnp.ndarray,
    drop_prob: float,
) -> jnp.ndarray:
    """Zero the whole action hidden state for a Bernoulli-``drop_prob`` fraction of samples.

    Mirrors Ctrl-World's
    ``text_mask = (rand(B) > 0.05).unsqueeze(1).unsqueeze(2)`` used to enable
    classifier-free guidance at inference.
    """
    b = action_hidden.shape[0]
    keep = jax.random.uniform(rng, (b,)) >= drop_prob
    keep = keep.astype(action_hidden.dtype).reshape((b,) + (1,) * (action_hidden.ndim - 1))
    return action_hidden * keep


def _build_adm_vector(
    batch: int,
    fps_id: float,
    motion_bucket_id: float,
    noise_aug_strength: float,
) -> jnp.ndarray:
    """SVD's 768-dim ADM micro-cond vector, replicated over a batch."""
    fps = jnp.full((batch,), float(fps_id))
    motion = jnp.full((batch,), float(motion_bucket_id))
    aug = jnp.full((batch,), float(noise_aug_strength))
    return svd_micro_cond_embed(fps, motion, aug, per_dim=256)


def action_world_train_step(
    rng: jax.Array,
    params: Dict[str, Any],
    apply_fns: Dict[str, Any],
    batch: Dict[str, jnp.ndarray],
    cfg: CtrlWorldTrainConfig,
    vae_scaling_factor: float,
    train: bool = True,
    probe: list | None = None,
):
    """Single training-step forward + loss for Ctrl-World (JAX).

    ``probe``: optional list for NaN localisation. When supplied, each forward
    stage appends ``(name, tensor)`` to it during tracing so the caller can
    report the first stage that goes non-finite (see ``max_utils.probe_add``).
    Leave it None — the default — and the computation is completely unchanged.

    ``batch`` layout mirrors the Ctrl-World torch dataset:

        ``latent``: ``(B, num_history+num_frames, 4, H/8, W/8)``
        ``action``: ``(B, num_history+num_frames, action_dim)``
        ``text_embeds``: ``(B, text_embed_dim)`` or None (pre-computed CLIP
            text embedding; leave out of ``apply_fns`` if not using text cond)

    ``apply_fns`` maps ``"unet"`` → ``unet.apply``,
    ``"action_encoder"`` → ``action_encoder.apply``. ``params`` is the
    matching pytree (``params["unet"]``, ``params["action_encoder"]``).

    Returns the scalar loss.
    """
    latents = batch["latent"]
    actions = batch["action"]
    text_embeds = batch.get("text_embeds", None)
    _p = probe_add
    _p(probe, "00_input_latents", latents)
    _p(probe, "01_input_actions", actions)
    if text_embeds is not None:
        _p(probe, "02_input_text_embeds", text_embeds)

    b, t_total = latents.shape[:2]
    t_future = cfg.num_frames
    t_history = cfg.num_history
    assert t_total == t_history + t_future, (
        f"batch['latent'] has {t_total} frames but cfg expects {t_history}+{t_future}"
    )

    (rng_cond, rng_action_drop, rng_sigma, rng_noise,
     rng_hist_sigma, rng_hist_noise) = jax.random.split(rng, 6)

    # 1. Channel-concat conditioning stream from the first FUTURE frame.
    future_first = latents[:, t_history]  # (B, 4, H/8, W/8)
    condition_latent = _build_concat_stream(
        rng_cond,
        future_first,
        num_total_frames=t_total,
        num_history=t_history,
        cond_aug_max=cfg.cond_aug_max,
        his_cond_zero=cfg.his_cond_zero,
    )
    _p(probe, "03_concat_stream", condition_latent)
    condition_latent = condition_latent / vae_scaling_factor
    _p(probe, "04_concat_stream_scaled", condition_latent)

    # 2. Per-frame action embedding (+ text), with CFG dropout.
    action_hidden = apply_fns["action_encoder"](
        {"params": params["action_encoder"]},
        actions,
        text_embeds,
        True,  # frame_level_cond
    )  # (B, T, hidden_size)
    _p(probe, "05_action_hidden", action_hidden)
    action_hidden = _apply_cfg_dropout(rng_action_drop, action_hidden, cfg.cfg_drop_prob)
    _p(probe, "06_action_hidden_cfg", action_hidden)

    # 3. EDM sigma per sample for the future slots.
    log_sigma = cfg.p_mean + cfg.p_std * jax.random.normal(rng_sigma, (b, 1, 1, 1, 1))
    sigma = jnp.exp(log_sigma)
    c_skip, c_out, c_in, c_noise, loss_weight = _edm_preconditioning(sigma)
    # c_noise is (B,1,1,1,1); UNet wants one scalar per sample in the flat batch.
    c_noise_btile = jnp.broadcast_to(c_noise.reshape(b, 1), (b, t_total)).reshape(-1)
    _p(probe, "07_sigma", sigma)
    _p(probe, "08_c_skip", c_skip)
    _p(probe, "09_c_out", c_out)
    _p(probe, "10_c_in", c_in)
    _p(probe, "11_c_noise", c_noise_btile)
    _p(probe, "12_loss_weight", loss_weight)

    # 4. Noise the latents (future: EDM; history: independent weak sigma).
    noise = jax.random.normal(rng_noise, latents.shape)
    noisy_latents = latents + noise * sigma  # uses sigma on all slots; we'll override history

    sigma_h = jax.random.normal(rng_hist_sigma, (b, t_history, 1, 1, 1)) * cfg.history_sigma_std
    hist_noise = jax.random.normal(rng_hist_noise, latents[:, :t_history].shape)
    hist_scale = 1.0 / jnp.sqrt(sigma_h ** 2 + 1.0)
    noisy_history = hist_scale * (latents[:, :t_history] + sigma_h * hist_noise)

    _p(probe, "13_noisy_latents", noisy_latents)
    _p(probe, "14_hist_scale", hist_scale)
    _p(probe, "15_noisy_history", noisy_history)
    noisy_future = c_in * noisy_latents[:, t_history:]
    _p(probe, "16_noisy_future", noisy_future)
    input_latents = jnp.concatenate([noisy_history, noisy_future], axis=1)
    # Channel-concat conditioning stream.
    input_latents = jnp.concatenate([input_latents, condition_latent], axis=2)
    _p(probe, "17_unet_input", input_latents)

    # Flatten batch+frame for the UNet.
    # (B, F, 8, H, W) -> (B*F, 8, H, W)
    input_flat = input_latents.reshape((b * t_total,) + input_latents.shape[2:])

    adm_vec = _build_adm_vector(
        b, cfg.fps_id, cfg.motion_bucket_id, cfg.noise_aug_strength
    ).astype(input_flat.dtype)
    image_only_indicator = jnp.zeros((b, t_total), dtype=input_flat.dtype)

    v_pred = apply_fns["unet"](
        {"params": params["unet"]},
        input_flat,
        c_noise_btile,
        encoder_hidden_states=action_hidden,
        added_cond_kwargs={"adm_vector": adm_vec},
        image_only_indicator=image_only_indicator,
        num_frames=t_total,
        frame_level_cond=True,
    ).sample  # (B*F, 4, H, W)
    _p(probe, "18_adm_vector", adm_vec)
    _p(probe, "19_unet_v_pred", v_pred)
    v_pred = v_pred.reshape((b, t_total) + v_pred.shape[1:])

    predict_x0 = c_out * v_pred + c_skip * noisy_latents
    _p(probe, "20_predict_x0", predict_x0)

    # Loss on future slots only. `loss_weight` is (B,1,1,1,1) so broadcast.
    diff = predict_x0[:, t_history:] - latents[:, t_history:]
    _p(probe, "21_diff", diff)
    loss = jnp.mean((diff ** 2) * loss_weight)
    _p(probe, "22_loss", loss)
    return loss


# -------------------------------------------------------------------------
# Convenience top-level module that binds the action encoder with the UNet
# apply_fn so callers only juggle one params pytree.
# -------------------------------------------------------------------------


def build_action_encoder(
    cfg: CtrlWorldTrainConfig,
    dtype: jnp.dtype = jnp.bfloat16,
    weights_dtype: jnp.dtype = jnp.float32,
) -> FlaxActionEncoder:
    """Construct the action encoder using the global ``CtrlWorldTrainConfig``."""
    return FlaxActionEncoder(
        action_dim=cfg.action_dim,
        hidden_size=cfg.hidden_size,
        text_embed_dim=cfg.text_embed_dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )
