# Copyright 2026 Princeton. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Flax inference pipeline for the action-conditioned SVD world model.

JAX / Flax port of
``Ctrl-World/models/pipeline_ctrl_world.py::CtrlWorldDiffusionPipeline``.

Differences from the base :class:`FlaxStableVideoDiffusionPipeline`:

  1. **No CLIP image encoder** on the conditioning path: the cross-attention
     context is the per-frame action-encoder output ``action_hidden``
     (shape ``(B, T, 1024)``). For CFG, the uncond branch zeros the whole
     action hidden state.
  2. **Pre-encoded or pixel image conditioning.** Accepts either a latent
     ``(B, 4, H/8, W/8)`` (the default in Ctrl-World training/eval) or a
     PIL image; the latent branch skips the VAE encode entirely and only
     unscales by ``vae.scaling_factor``.
  3. **History conditioning.** Accepts a ``history`` tensor shaped
     ``(B, num_history, 4, H/8, W/8)`` (pre-encoded) that is prepended to
     the noisy latent on the frame axis at every denoising step. The
     scheduler step only updates the ``num_frames`` future slots.
  4. **``his_cond_zero``** forwarded to the channel-concat stream: if True,
     the history slots of the tiled conditioning latent are zeroed.

Design note: this pipeline is structured as an inference helper that the
caller drives; it does not re-use the micro-cond / VAE load path, which is
already correctly implemented in the base pipeline. The base pipeline's
fori_loop is replaced with a tight Python loop (``B=1``) that concatenates
history + future on the frame axis before the UNet.
"""

import functools
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from PIL import Image

from maxdiffusion.models.svd.action_encoder_flax import tile_text_to_hidden

from ...models.embeddings_flax import svd_micro_cond_embed
from ...schedulers.scheduling_edm_euler_flax import (
    FlaxEDMEulerScheduler,
    LinearPredictionGuider,
    v_scaling_edm,
)
from .pipeline_flax_svd import FlaxStableVideoDiffusionPipeline, _preprocess_image_for_vae


# Mirror of ``pipeline_flax_svd.DEBUG``: set True to run the denoising sampler
# as a Python ``for`` (easier to step through with jax.debug / pdb); leave False
# to wrap the body in ``jax.lax.fori_loop`` so XLA compiles the body once and
# the 50-step loop becomes a single cheap scan. Matching the base SVD pattern
# here takes first-call compile from ~15 min (unrolled) to ~1-3 min.
DEBUG = False


class FlaxCtrlWorldPipeline(FlaxStableVideoDiffusionPipeline):
    """Action-conditioned SVD pipeline with history conditioning."""

    def __init__(
        self,
        vae: Any,
        unet: Any,
        action_encoder: Any,
        scheduler: FlaxEDMEulerScheduler,
        action_adaln_proj: Any = None,
        image_encoder: Any = None,
        feature_extractor: Any = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        # Base SVD pipeline registers CLIP + feature extractor, which we only
        # use if text-cond text embeddings are NOT pre-computed. Accept None
        # to allow dropping the CLIP image tower entirely for robot-world-model
        # use cases where actions replace image cross-attn.
        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            dtype=dtype,
        )
        # register_modules on the parent only knows about the SVD set; stash
        # the action encoder directly.
        self.action_encoder = action_encoder
        # Non-None only for action_cond_mode='adaln' checkpoints, which carry an
        # extra "action_adaln_proj" params subtree.
        self.action_adaln_proj = action_adaln_proj
        # jitted samplers keyed by their static signature; see _get_sampler.
        self._sampler_cache: dict = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_cond_latent_from_tensor(
        self,
        image_latent: jnp.ndarray,
        num_total_frames: int,
        num_history: int,
        his_cond_zero: bool,
    ) -> jnp.ndarray:
        """Build the channel-concat stream from an already-encoded latent.

        ``image_latent`` is ``(B, 4, H/8, W/8)`` in scaled space (VAE output
        × scaling_factor); Ctrl-World unscales by dividing here to match
        diffusers' ``_encode_vae_image`` contract on the concat path.

        Returns ``(B, num_total_frames, 4, H/8, W/8)``.
        """
        b, c, h, w = image_latent.shape
        unscaled = image_latent / self.vae.config.scaling_factor
        tiled = jnp.broadcast_to(unscaled[:, None], (b, num_total_frames, c, h, w))
        if his_cond_zero and num_history > 0:
            zero_slice = jnp.zeros_like(tiled[:, :num_history])
            tiled = jnp.concatenate([zero_slice, tiled[:, num_history:]], axis=1)
        return tiled

    def _encode_cond_latent_from_pixels(
        self,
        pil_image: Image.Image,
        vae_params: Any,
        prng_seed: jax.Array,
        num_total_frames: int,
        num_history: int,
        height: int,
        width: int,
        cond_aug: float,
        his_cond_zero: bool,
    ) -> jnp.ndarray:
        """Same as above but run the VAE encoder first. Output scaled by
        ``scaling_factor`` then divided out (i.e. VAE raw mode)."""
        pixels = _preprocess_image_for_vae(pil_image, height, width)
        noise = jax.random.normal(prng_seed, pixels.shape, dtype=pixels.dtype) * cond_aug
        pixels = pixels + noise
        posterior = self.vae.apply(
            {"params": vae_params}, pixels, deterministic=True, method=self.vae.encode
        )
        latent = posterior.latent_dist.mode()  # (1, H/8, W/8, 4) NHWC
        latent = jnp.transpose(latent, (0, 3, 1, 2))  # (1, 4, H/8, W/8)
        b, c, h, w = latent.shape
        tiled = jnp.broadcast_to(latent[:, None], (b, num_total_frames, c, h, w))
        if his_cond_zero and num_history > 0:
            zero_slice = jnp.zeros_like(tiled[:, :num_history])
            tiled = jnp.concatenate([zero_slice, tiled[:, num_history:]], axis=1)
        return tiled

    # ------------------------------------------------------------------
    # Sampler (jitted once, reused for every call)
    # ------------------------------------------------------------------

    def _sample(
        self,
        unet_params,
        latents,
        history,
        concat_stream_all,
        action_hidden_all,
        action_hidden_states_all,
        adm_vec_all,
        sigmas,
        *,
        t_total: int,
        t_history: int,
        t_future: int,
        do_cfg: bool,
        adaln: bool,
        frame_level_cond: bool,
        num_inference_steps: int,
        min_guidance_scale: float,
        max_guidance_scale: float,
    ):
        """EDM Euler denoising loop. Everything varying is an argument.

        This exists as a separate function purely so it can be ``jax.jit``-ed
        once and reused. When the loop lived inside ``__call__`` it closed over
        ``history`` / ``concat_stream_all`` / ``action_hidden_all``, which are
        fresh arrays on every call: ``lax.fori_loop`` then re-traced the body per
        call and ``while_p``'s executable cache — keyed on the traced jaxpr —
        missed every time, so the whole UNet recompiled once per AR chunk.
        Profiling a single 24s chunk showed 15s inside ``PJRT_Client_Compile``
        and 11% GPU utilisation. Passing these in as arguments makes the cache
        key (shape, dtype), so the sweep compiles exactly once.
        """
        guider = LinearPredictionGuider(
            num_frames=t_future,
            min_scale=min_guidance_scale,
            max_scale=max_guidance_scale,
        )
        b_all = concat_stream_all.shape[0]
        # dtype follows the UNet input, matching action_world_train_step (which
        # builds it as ``input_flat.dtype``). Forcing f32 here would promote the
        # AlphaBlender mix differently than training did.
        image_only_indicator = jnp.zeros((b_all, t_total), dtype=concat_stream_all.dtype)

        def loop_body(i, latents_future):
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]

            c_skip, c_out, c_in, _ = v_scaling_edm(sigma)
            scaled_future = latents_future * c_in  # (B, F, 4, H, W)

            # Prepend history slots (noisy history IS used as-is in Ctrl-World
            # inference; the torch pipeline cats `history` at full scale).
            if t_history > 0:
                # (B, H, 4, h, w) + (B, F, 4, h, w) -> (B, T, 4, h, w)
                full_noisy = jnp.concatenate([history, scaled_future], axis=1)
            else:
                full_noisy = scaled_future
            if do_cfg:
                full_noisy_all = jnp.concatenate([full_noisy] * 2, axis=0)
            else:
                full_noisy_all = full_noisy

            # Channel-concat with conditioning stream -> 8-channel input.
            unet_in = jnp.concatenate([full_noisy_all, concat_stream_all], axis=2)
            # Flatten batch+frame for the UNet.
            unet_flat = unet_in.reshape((b_all * t_total,) + unet_in.shape[2:])

            c_noise = 0.25 * jnp.log(sigma)
            timesteps = jnp.broadcast_to(c_noise, (b_all * t_total,))

            v_pred = self.unet.apply(
                {"params": unet_params},
                unet_flat,
                timesteps,
                encoder_hidden_states=action_hidden_all,
                added_cond_kwargs={"adm_vector": adm_vec_all},
                image_only_indicator=image_only_indicator,
                num_frames=t_total,
                frame_level_cond=(False if adaln else frame_level_cond),
                action_hidden_states=action_hidden_states_all,
            ).sample  # (b_all*T, 4, H, W)
            v_pred = v_pred.reshape((b_all, t_total) + v_pred.shape[1:])

            # Drop history slots from the prediction — only future gets updated.
            if t_history > 0:
                v_pred = v_pred[:, t_history:]  # (b_all, F, 4, H, W)

            # EDM -> x0. Both cond and uncond branches start from the same
            # `latents_future`, so replicate it for the EDM x_in term.
            if do_cfg:
                latents_expanded = jnp.concatenate([latents_future] * 2, axis=0)
            else:
                latents_expanded = latents_future
            pred_x0_all = c_skip * latents_expanded + c_out * v_pred

            if do_cfg:
                pred_x0_uncond, pred_x0_cond = jnp.split(pred_x0_all, 2, axis=0)
                # Guider expects leading dim B*F; flatten then reshape back.
                bb = pred_x0_cond.shape[0]
                flat_shape = (bb * t_future,) + pred_x0_cond.shape[2:]
                pred_x0 = guider(
                    pred_x0_cond.reshape(flat_shape),
                    pred_x0_uncond.reshape(flat_shape),
                ).reshape(pred_x0_cond.shape)
            else:
                pred_x0 = pred_x0_all

            derivative = (latents_future - pred_x0) / sigma
            dt = next_sigma - sigma
            return latents_future + derivative * dt

        if DEBUG:
            for i in range(num_inference_steps):
                latents = loop_body(i, latents)
            return latents
        return jax.lax.fori_loop(0, num_inference_steps, loop_body, latents)

    def _get_sampler(self, **statics):
        """``jax.jit(_sample)`` memoised on the static signature.

        Cached on the instance rather than rebuilt per call: a fresh ``jax.jit``
        object each time would carry a fresh compilation cache and defeat the
        whole point.
        """
        key = tuple(sorted((k, v) for k, v in statics.items()))
        fn = self._sampler_cache.get(key)
        if fn is None:
            fn = jax.jit(functools.partial(self._sample, **statics))
            self._sampler_cache[key] = fn
        return fn

    # ------------------------------------------------------------------
    # Main inference
    # ------------------------------------------------------------------

    def __call__(
        self,
        params: Union[dict, FrozenDict],
        prng_seed: jax.Array,
        action: jnp.ndarray,
        image: Optional[Union[Image.Image, jnp.ndarray]] = None,
        image_latent: Optional[jnp.ndarray] = None,
        history: Optional[jnp.ndarray] = None,
        text_embeds: Optional[jnp.ndarray] = None,
        num_frames: int = 5,
        num_history: int = 6,
        height: int = 192 * 3,
        width: int = 320,
        num_inference_steps: int = 50,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 1.0,
        fps_id: int = 7,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02,
        frame_level_cond: bool = True,
        his_cond_zero: bool = False,
        action_cond_mode: str = "cross_attn",
        output_type: str = "latent",
        decode_chunk_size: Optional[int] = None,
    ):
        """Run the full action-conditioned denoising loop for one video.

        One of ``image`` (PIL / pixel tensor) or ``image_latent``
        (pre-encoded VAE latent of shape ``(B, 4, H/8, W/8)``) must be
        provided. The latent path mirrors Ctrl-World's typical usage where
        frames are pre-extracted offline.

        ``action`` has shape ``(B, num_history + num_frames, action_dim)``;
        it is fed through ``self.action_encoder`` to produce the per-frame
        cross-attention context.

        ``history``, when provided, has shape
        ``(B, num_history, 4, H/8, W/8)`` and is prepended on the frame
        axis at every denoising step; its slots are ignored by the
        scheduler update.

        ``output_type`` of ``"latent"`` returns the ``num_frames`` future
        latents directly (shape ``(B, num_frames, 4, H/8, W/8)``); ``"np"``
        additionally decodes to pixels.
        """
        do_cfg = max_guidance_scale > 1.0
        t_future = num_frames
        t_history = num_history if history is not None else 0
        t_total = t_future + t_history

        adaln = action_cond_mode == "adaln"
        if adaln and self.action_adaln_proj is None:
            raise ValueError(
                "action_cond_mode='adaln' needs the pipeline to be constructed with "
                "action_adaln_proj=<FlaxActionAdaLNProjector>."
            )
        # 1. Action conditioning. Mirrors action_world_train_step exactly — any
        #    divergence here silently degrades generation rather than erroring.
        if adaln:
            # Action route carries no text (text goes to cross-attention below),
            # matching training.
            action_hidden = self.action_encoder.apply(
                {"params": params["action_encoder"]},
                action,
                None,
                True,   # frame_level_cond: one action token per frame
            )  # (B, T, 1024)
            b = action_hidden.shape[0]
            if do_cfg:
                # Uncond = zero action tokens, projected — i.e. proj(0) = bias,
                # NOT a zero contribution. Training drops the tokens before the
                # projection, so the uncond branch must be built the same way.
                action_hidden = jnp.concatenate(
                    [jnp.zeros_like(action_hidden), action_hidden], axis=0
                )
            action_temb = self.action_adaln_proj.apply(
                {"params": params["action_adaln_proj"]}, action_hidden
            )  # (b_all, T, time_embed_dim)
            action_hidden_states_all = action_temb.reshape(
                action_temb.shape[0] * t_total, -1
            )
            # Cross-attention carries the per-sample text embedding, identical on
            # both CFG branches, pre-flattened to (b_all*T, 1, C) because the UNet
            # is called with frame_level_cond=False.
            if text_embeds is not None:
                text_ctx = tile_text_to_hidden(
                    text_embeds, self.action_encoder.hidden_size,
                    self.action_encoder.text_embed_dim,
                )  # (B, 1, 1024)
            else:
                text_ctx = jnp.zeros((b, 1, self.action_encoder.hidden_size),
                                     dtype=action_temb.dtype)
            if do_cfg:
                text_ctx = jnp.concatenate([text_ctx, text_ctx], axis=0)
            action_hidden_all = jnp.repeat(
                text_ctx.astype(action_temb.dtype), t_total, axis=0
            )
        else:
            action_hidden = self.action_encoder.apply(
                {"params": params["action_encoder"]},
                action,
                text_embeds,
                frame_level_cond,
            )  # (B, T, 1024) where T = num_history + num_frames
            b = action_hidden.shape[0]
            if do_cfg:
                uncond_hidden = jnp.zeros_like(action_hidden)
                action_hidden_all = jnp.concatenate([uncond_hidden, action_hidden], axis=0)
            else:
                action_hidden_all = action_hidden
            action_hidden_states_all = None

        # 2. Channel-concat conditioning stream.
        rng_cond, rng_noise = jax.random.split(prng_seed)
        if image_latent is not None:
            concat_stream = self._encode_cond_latent_from_tensor(
                image_latent,
                num_total_frames=t_total,
                num_history=t_history,
                his_cond_zero=his_cond_zero,
            )
        else:
            if image is None:
                raise ValueError(
                    "FlaxCtrlWorldPipeline requires either image_latent= or image=."
                )
            if not isinstance(image, Image.Image):
                raise ValueError(
                    "FlaxCtrlWorldPipeline: raw tensor images are not supported in "
                    "the inference path; pass a PIL.Image or image_latent=."
                )
            concat_stream = self._encode_cond_latent_from_pixels(
                image,
                params["vae"],
                rng_cond,
                num_total_frames=t_total,
                num_history=t_history,
                height=height,
                width=width,
                cond_aug=cond_aug,
                his_cond_zero=his_cond_zero,
            )
        if do_cfg:
            # Ctrl-World duplicates the concat stream for CFG (does NOT zero
            # the uncond concat stream — only the CLIP / action stream is
            # zeroed on the uncond branch).
            concat_stream_all = jnp.concatenate([concat_stream] * 2, axis=0)
        else:
            concat_stream_all = concat_stream

        # 3. ADM micro-cond vector (shared between cond and uncond).
        adm_cond = svd_micro_cond_embed(
            jnp.full((b,), float(fps_id)),
            jnp.full((b,), float(motion_bucket_id)),
            jnp.full((b,), float(cond_aug)),
            per_dim=256,
        )
        if do_cfg:
            adm_vec_all = jnp.concatenate([adm_cond, adm_cond], axis=0)
        else:
            adm_vec_all = adm_cond

        # 5. Initial noise for the FUTURE slots only.
        h_lat = concat_stream.shape[-2]
        w_lat = concat_stream.shape[-1]
        future_shape = (b, t_future, self.unet.config.out_channels, h_lat, w_lat)
        latents = jax.random.normal(rng_noise, future_shape, dtype=jnp.float32)

        # 6. Scheduler setup.
        scheduler_state = self.scheduler.set_timesteps(
            self.scheduler.create_state(num_inference_steps=num_inference_steps),
            num_inference_steps=num_inference_steps,
        )
        latents = latents * scheduler_state.init_noise_sigma

        # 7. Denoising loop — EDM Euler sampler, jitted once and memoised on the
        # instance. Everything the loop needs is passed in as an argument rather
        # than closed over; see _sample's docstring for why that matters (it is
        # the difference between compiling once and compiling per call).
        sampler = self._get_sampler(
            t_total=t_total,
            t_history=t_history,
            t_future=t_future,
            do_cfg=do_cfg,
            adaln=adaln,
            frame_level_cond=frame_level_cond,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=float(min_guidance_scale),
            max_guidance_scale=float(max_guidance_scale),
        )
        latents = sampler(
            params["unet"],
            latents,
            history,
            concat_stream_all,
            action_hidden_all,
            action_hidden_states_all,
            adm_vec_all,
            scheduler_state.sigmas,
        )

        if output_type == "latent":
            return latents

        # 8. Decode future slots to pixels.
        chunk_size = decode_chunk_size if decode_chunk_size is not None else t_future
        chunk_size = max(1, min(chunk_size, t_future))
        latents_flat = latents.reshape((b * t_future,) + latents.shape[2:])
        latents_flat = latents_flat / self.vae.config.scaling_factor
        frame_chunks = []
        for start in range(0, latents_flat.shape[0], chunk_size):
            chunk = latents_flat[start : start + chunk_size]
            frame_chunks.append(
                self.vae.apply(
                    {"params": params["vae"]},
                    chunk,
                    num_frames=chunk.shape[0],
                    deterministic=True,
                    method=self.vae.decode,
                ).sample
            )
        frames = jnp.concatenate(frame_chunks, axis=0)
        frames = (frames / 2.0 + 0.5).clip(0.0, 1.0)
        if output_type == "np":
            frames = jnp.transpose(frames, (0, 2, 3, 1))
            frames = np.asarray(frames)
        return frames
