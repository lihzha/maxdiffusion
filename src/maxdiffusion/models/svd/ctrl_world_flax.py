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
from .action_encoder_flax import FlaxActionEncoder, tile_text_to_hidden


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

    # How the action signal reaches the UNet:
    #   'cross_attn' (default) — one action token per frame is the UNet's
    #                            cross-attention context (frame_level_cond).
    #   'adaln'                — action tokens are projected per frame and summed
    #                            into the timestep embedding instead; the
    #                            per-frame cross-attention path is dropped and
    #                            cross-attention carries the (per-sample) text
    #                            embedding on its own.
    # Not checkpoint-compatible across modes: 'adaln' adds an action_adaln_proj
    # parameter subtree and changes what cross-attention receives.
    action_cond_mode: str = "cross_attn"
    # UNet timestep-embedding width; only read in 'adaln' mode (block_out_channels[0]*4).
    time_embed_dim: int = 1280

    # Whether the DROID task instruction (the pre-computed CLIP text embedding)
    # is fed to the model at all. False makes the run action-only, which is what
    # the WAN arm has always done — see ``use_task_instructions`` in
    # base_ctrl_world.yml for why the two models' defaults differ.
    #
    # Where the text goes when enabled depends on action_cond_mode, and the WAN
    # trainer mirrors this exactly so the two arms stay comparable:
    #   'cross_attn' — tiled to hidden_size and broadcast-added to the action
    #                  tokens inside the action encoder, i.e. it shares the
    #                  action's cross-attention route (and its CFG dropout).
    #   'adaln'      — carried by cross-attention on its own, since the action
    #                  has moved to the timestep embedding. Not CFG-dropped.
    # Costs no parameters either way, so it is safe to flip between runs.
    use_task_instructions: bool = True


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
):
    """Single training-step forward + loss for Ctrl-World (JAX).

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
    if not cfg.use_task_instructions:
        # Single gate for BOTH conditioning modes: downstream, cross_attn skips
        # the tiled-text add inside the action encoder and adaln falls back to a
        # zero cross-attention context. Dropping text costs no parameters —
        # ``tile_text_to_hidden`` is a pure reshape — so the checkpoint tree is
        # identical either way and the flag can be flipped between runs.
        text_embeds = None

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
    condition_latent = condition_latent / vae_scaling_factor

    # 2. Per-frame action embedding, routed by action_cond_mode.
    adaln = cfg.action_cond_mode == "adaln"
    if adaln:
        # Action-only encoder output: text is NOT folded in here, it goes to
        # cross-attention on its own below. Feeding text into both routes would
        # double-count it.
        action_hidden = apply_fns["action_encoder"](
            {"params": params["action_encoder"]},
            actions,
            None,   # no text on the action route
            True,   # frame_level_cond
        )  # (B, T, hidden_size)
        # Drop BEFORE the projection, matching the WAN trainer. The dropped
        # sample's contribution is then proj(0) = the projector bias, not zero —
        # which is the point: at inference the uncond branch is built the same
        # way, so the bias appears in both cond and uncond and cancels out of the
        # CFG delta (cond - uncond = W·a). Dropping after the projection would
        # leave the bias in the delta, so raising guidance_scale would amplify a
        # learned constant that carries no action information.
        action_hidden = _apply_cfg_dropout(rng_action_drop, action_hidden, cfg.cfg_drop_prob)
        action_temb = apply_fns["action_adaln_proj"](
            {"params": params["action_adaln_proj"]}, action_hidden
        )  # (B, T, time_embed_dim)
        action_hidden_states = action_temb.reshape(b * t_total, -1)
        # Cross-attention carries the per-sample text embedding, tiled to the
        # cross-attn width and repeated over frames. The UNet is called with
        # frame_level_cond=False, so it expects (B*T, S, C) already flattened.
        if text_embeds is not None and cfg.text_embed_dim is not None:
            text_ctx = tile_text_to_hidden(
                text_embeds, cfg.hidden_size, cfg.text_embed_dim
            )  # (B, 1, hidden_size)
        else:
            text_ctx = jnp.zeros((b, 1, cfg.hidden_size), dtype=action_hidden.dtype)
        encoder_hidden_states = jnp.repeat(
            text_ctx.astype(action_hidden.dtype), t_total, axis=0
        )  # (B*T, 1, hidden_size)
    else:
        # Text is added AFTER the dropout, not folded in by the encoder, so the
        # instruction is never dropped — it is present in every sample and at
        # every inference call, and keeping it out of the mask means CFG scales
        # the action alone (the text term is identical in both branches and
        # cancels out of the delta).
        #
        # NOTE: this deviates from upstream Ctrl-World, whose mask is literally
        # named ``text_mask`` and blanks the combined action+text hidden state.
        # Runs trained before this change learned a no-text branch; adaln mode
        # was always on the current behaviour (its text never entered the mask).
        action_hidden = apply_fns["action_encoder"](
            {"params": params["action_encoder"]},
            actions,
            None,   # text added below, outside the dropout
            True,   # frame_level_cond
        )  # (B, T, hidden_size)
        action_hidden = _apply_cfg_dropout(rng_action_drop, action_hidden, cfg.cfg_drop_prob)
        if text_embeds is not None and cfg.text_embed_dim is not None:
            tiled = tile_text_to_hidden(
                text_embeds, cfg.hidden_size, cfg.text_embed_dim
            )  # (B, 1, hidden_size)
            action_hidden = action_hidden + tiled.astype(action_hidden.dtype)
        encoder_hidden_states = action_hidden
        action_hidden_states = None

    # 3. EDM sigma per sample for the future slots.
    log_sigma = cfg.p_mean + cfg.p_std * jax.random.normal(rng_sigma, (b, 1, 1, 1, 1))
    sigma = jnp.exp(log_sigma)
    c_skip, c_out, c_in, c_noise, loss_weight = _edm_preconditioning(sigma)
    # c_noise is (B,1,1,1,1); UNet wants one scalar per sample in the flat batch.
    c_noise_btile = jnp.broadcast_to(c_noise.reshape(b, 1), (b, t_total)).reshape(-1)

    # 4. Noise the latents (future: EDM; history: independent weak sigma).
    noise = jax.random.normal(rng_noise, latents.shape)
    noisy_latents = latents + noise * sigma  # uses sigma on all slots; we'll override history

    sigma_h = jax.random.normal(rng_hist_sigma, (b, t_history, 1, 1, 1)) * cfg.history_sigma_std
    hist_noise = jax.random.normal(rng_hist_noise, latents[:, :t_history].shape)
    hist_scale = 1.0 / jnp.sqrt(sigma_h ** 2 + 1.0)
    noisy_history = hist_scale * (latents[:, :t_history] + sigma_h * hist_noise)

    noisy_future = c_in * noisy_latents[:, t_history:]
    input_latents = jnp.concatenate([noisy_history, noisy_future], axis=1)
    # Channel-concat conditioning stream.
    input_latents = jnp.concatenate([input_latents, condition_latent], axis=2)

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
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"adm_vector": adm_vec},
        image_only_indicator=image_only_indicator,
        num_frames=t_total,
        # adaln already flattened its (per-sample text) context to (B*T, S, C),
        # so the per-frame reshape must be off; cross_attn mode still needs it.
        frame_level_cond=not adaln,
        action_hidden_states=action_hidden_states,
    ).sample  # (B*F, 4, H, W)
    v_pred = v_pred.reshape((b, t_total) + v_pred.shape[1:])

    predict_x0 = c_out * v_pred + c_skip * noisy_latents

    # Loss on future slots only. `loss_weight` is (B,1,1,1,1) so broadcast.
    diff = predict_x0[:, t_history:] - latents[:, t_history:]
    loss = jnp.mean((diff ** 2) * loss_weight)
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
