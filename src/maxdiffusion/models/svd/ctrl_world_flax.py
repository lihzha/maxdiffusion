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
    #   'skeleton'             — the 7-dim vector actions are NOT used. A rendered
    #                            2D-kinematic-skeleton video, VAE-encoded by the
    #                            SAME VAE as the RGB video, is embedded by a
    #                            separate conv and ADDED onto conv_in's output.
    #   'skeleton_adaln'       — the same skeleton latents pooled to one vector per
    #                            frame and summed into t_emb, the site the vector
    #                            'adaln' route uses. NOTE this site is spatially
    #                            blind in SVD (t_emb is (B*T, time_embed_dim),
    #                            with no spatial axis), so the pooling discards
    #                            exactly what makes a skeleton richer than a
    #                            7-dim action. Unavoidable, and unlike WAN, whose
    #                            per-token AdaLN preserved the grid.
    #   'skeleton_cross_attn'  — the same skeleton latents as a per-frame spatial
    #                            K/V grid for the SPATIAL cross-attention, with a
    #                            learned positional embedding standing in for the
    #                            rotary embeddings SVD does not have.
    # Not checkpoint-compatible across modes: each adds a different parameter
    # subtree ('adaln' -> action_adaln_proj, the three skeleton modes -> one
    # skeleton_* module each and NO action_encoder at all).
    action_cond_mode: str = "cross_attn"
    # UNet timestep-embedding width; read in 'adaln' and 'skeleton_adaln' modes
    # (block_out_channels[0]*4).
    time_embed_dim: int = 1280
    # conv_in's output width (block_out_channels[0]); read in 'skeleton' mode,
    # which must match it to be added onto conv_in's output.
    model_channels: int = 320
    # Fixed scale on the additive skeleton bias; 'skeleton' only, as in the WAN
    # arm and OSCAR. The other two skeleton routes deliberately have none.
    skeleton_embed_alpha: float = 0.1
    # Spatial downsample of the skeleton K/V grid; 'skeleton_cross_attn' only.
    # 4 -> 180 keys per frame at 72x40, matching the WAN route's per-frame count.
    skeleton_cross_attn_stride: int = 4

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


def _is_skeleton_mode(action_cond_mode: str) -> bool:
    """Whether the conditioning is the rendered-skeleton video rather than the
    vector actions. True for all three skeleton routes, which agree on what the
    dataset must carry and on there being no action encoder, and differ only in
    where the encoded skeleton is injected."""
    return action_cond_mode in ("skeleton", "skeleton_adaln", "skeleton_cross_attn")


def _skeleton_apply_key(action_cond_mode: str) -> str | None:
    """The ``apply_fns``/``params`` key of the live skeleton module for this mode.

    Each mode builds a DIFFERENT module (different output shape, different
    injection site) and only ever one at a time, so a checkpoint from one cannot
    be restored into another.
    """
    return {
        "skeleton":            "skeleton_embed",
        "skeleton_adaln":      "skeleton_adaln_proj",
        "skeleton_cross_attn": "skeleton_cross_attn_embed",
    }.get(action_cond_mode)


def _encode_skeleton(
    apply_fns: Dict[str, Any],
    params: Dict[str, Any],
    action_cond_mode: str,
    skeleton: jnp.ndarray | None,
    cfg_rng: jax.Array | None,
    drop_prob: float,
) -> jnp.ndarray | None:
    """Embed skeleton latents into this mode's conditioning tensor.

    ``(B, T, 4, H, W)`` -> whatever the mode's site wants:
    ``(B*T, H, W, model_channels)`` for ``skeleton``, ``(B*T, time_embed_dim)``
    for ``skeleton_adaln``, ``(B*T, S, hidden_size)`` for
    ``skeleton_cross_attn``. Returns ``None`` outside the skeleton modes, which
    jit traces as an absent pytree so those modes are untouched.

    CFG dropout zeroes the *token contribution* — i.e. skips the injection — for
    a ``drop_prob`` fraction of samples, which is the true no-conditioning state.
    Zeroing the skeleton *latents* instead would not be: an empty (all-black)
    skeleton frame encodes to a perfectly ordinary nonzero latent, so a zero
    latent is off-manifold input rather than an absent condition. Pass
    ``cfg_rng=None`` (eval, and the cond branch of a guided rollout) to skip it.
    """
    key = _skeleton_apply_key(action_cond_mode)
    if key is None or skeleton is None:
        return None
    tokens = apply_fns[key]({"params": params[key]}, skeleton)
    if cfg_rng is not None and drop_prob > 0.0:
        keep = jax.random.uniform(cfg_rng, (tokens.shape[0],) + (1,) * (tokens.ndim - 1)) >= drop_prob
        tokens = tokens * keep.astype(tokens.dtype)
    return tokens


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
    skeleton_mode = _is_skeleton_mode(cfg.action_cond_mode)
    adaln = cfg.action_cond_mode == "adaln"
    skeleton_hidden_states = None
    if skeleton_mode:
        # No action encoder in these modes — the conditioning is the rendered
        # skeleton video, and the vector actions are unused. CFG drops the
        # SKELETON here, which is what a guided rollout's uncond branch drops too.
        skel_tokens = _encode_skeleton(
            apply_fns, params, cfg.action_cond_mode,
            batch.get("skeleton", None), rng_action_drop, cfg.cfg_drop_prob,
        )
        if skel_tokens is None:
            raise ValueError(
                f"action_cond_mode={cfg.action_cond_mode!r} needs batch['skeleton']; "
                "the dataset must be built with load_skeleton=True on a dataset "
                "carrying skeleton_cam0/1/2."
            )
        # Text, when enabled, is tiled to the cross-attention width exactly as
        # the vector routes tile it.
        if text_embeds is not None and cfg.text_embed_dim is not None:
            text_ctx = tile_text_to_hidden(
                text_embeds, cfg.hidden_size, cfg.text_embed_dim
            )                                                    # (B, 1, hidden)
        else:
            text_ctx = jnp.zeros((b, 1, cfg.hidden_size), dtype=skel_tokens.dtype)
        text_ctx = jnp.repeat(text_ctx.astype(skel_tokens.dtype), t_total, axis=0)

        if cfg.action_cond_mode == "skeleton_cross_attn":
            # The skeleton IS the cross-attention K/V. Unlike WAN — whose
            # frame-locked (B*F, K, D) reshape leaves no room for a second
            # sequence, forcing the instruction to be POOLED onto the action
            # tokens — SVD's context is inherently per-(sample, frame) already,
            # so the full text token simply rides alongside the grid as one extra
            # key. Prepended, and never CFG-dropped, so it is identical in both
            # guidance branches and cancels out of the delta.
            encoder_hidden_states = jnp.concatenate([text_ctx, skel_tokens], axis=1)
            action_hidden_states = None
        else:
            # 'skeleton' and 'skeleton_adaln' leave cross-attention free, so it
            # carries the instruction on its own (or zeros), exactly as 'adaln'
            # does.
            encoder_hidden_states = text_ctx
            if cfg.action_cond_mode == "skeleton_adaln":
                action_hidden_states = skel_tokens        # (B*T, time_embed_dim)
            else:
                action_hidden_states = None
                skeleton_hidden_states = skel_tokens      # (B*T, H, W, model_ch)
    elif adaln:
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
        # adaln and all three skeleton modes hand cross-attention a context that
        # is ALREADY flattened to (B*T, S, C), so the per-frame reshape must be
        # off; only the vector cross_attn route still needs it.
        frame_level_cond=not (adaln or skeleton_mode),
        action_hidden_states=action_hidden_states,
        skeleton_hidden_states=skeleton_hidden_states,
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
