# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0

"""Flax pipeline for Stable Video Diffusion (base).

Stages (mirrors ``sgm.inference.sampling`` for img2vid):

  1. Preprocess the conditioning image (CLIP processor + VAE-input resize).
  2. CLIP ViT-H/14 image tower → (B, 1, 1024) conditioning.
  3. VAE-encode the noise-augmented conditioning image → 4-channel concat
     latent; tile along T.
  4. Build the ADM vector from (fps_id, motion_bucket_id, cond_aug).
  5. Sample initial noise with std sigma_max; EDM Euler loop over sigmas;
     each step: UNet forward (cond + uncond), per-frame linear CFG, Euler step.
  6. VAE video-decode to pixel frames.

The uncond branch follows sgm's simple_video_sample.py — zero CLIP features
and zero concat stream.
"""

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from PIL import Image

from ...models.embeddings_flax import svd_micro_cond_embed
from ...schedulers.scheduling_edm_euler_flax import (
    FlaxEDMEulerScheduler,
    LinearPredictionGuider,
    v_scaling_edm,
)
from ..pipeline_flax_utils import FlaxDiffusionPipeline


# Set True to use a python for loop instead of fori_loop (easier to debug).
DEBUG = False


def _preprocess_image_for_vae(image: Image.Image, height: int, width: int) -> jnp.ndarray:
    """Resize and normalize the conditioning image for VAE encode.

    Output: ``(1, 3, H, W)`` float32 array in [-1, 1].
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # (1, C, H, W)
    return jnp.asarray(arr)


class FlaxStableVideoDiffusionPipeline(FlaxDiffusionPipeline):
    """Stable Video Diffusion (base) — 14-frame img2vid at 576x1024."""

    def __init__(
        self,
        vae: Any,
        image_encoder: Any,
        feature_extractor: Any,
        unet: Any,
        scheduler: FlaxEDMEulerScheduler,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # 8

    # ---------------------------------------------------------------------------
    # Conditioning
    # ---------------------------------------------------------------------------

    def encode_image_clip(self, pil_image: Image.Image, params: Any) -> jnp.ndarray:
        """Run the CLIP vision tower on the input image. Returns (1, 1, 1024).

        Matches Diffusers' ``StableVideoDiffusionPipeline._encode_image``:
        stretch-resize with antialiasing to 224x224 (NO aspect preservation, NO
        center crop), then CLIP-normalize only. HF's default preprocessing would
        aspect-preserve resize + center crop, which drops content on non-square
        inputs (e.g. 576x1024) and produces wrong CLIP features.
        """
        import torch as _torch
        from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
            _resize_with_antialiasing,
        )

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]
        arr = np.transpose(arr, (2, 0, 1))[None, ...]  # (1, C, H, W)
        image_t = _torch.from_numpy(arr)
        image_t = image_t * 2.0 - 1.0
        image_t = _resize_with_antialiasing(image_t, (224, 224))
        image_t = (image_t + 1.0) / 2.0

        processed = self.feature_extractor(
            images=image_t,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="np",
        )
        pixel_values = jnp.asarray(processed["pixel_values"])
        out = self.image_encoder(pixel_values=pixel_values, params=params)
        # FlaxCLIPVisionModelWithProjection returns .image_embeds (B, 1024).
        image_embeds = out.image_embeds
        return image_embeds[:, None, :]  # (B, 1, 1024)

    def encode_cond_latent(
        self,
        pil_image: Image.Image,
        vae_params: Any,
        prng_seed: jax.Array,
        num_frames: int,
        height: int,
        width: int,
        cond_aug: float,
    ) -> jnp.ndarray:
        """VAE-encode the (noise-augmented) conditioning image and tile over T.

        Output: ``(B*T, 4, H/8, W/8)`` ready to be channel-concatenated with the
        noisy latent.
        """
        pixels = _preprocess_image_for_vae(pil_image, height, width)  # (1, 3, H, W)
        noise = jax.random.normal(prng_seed, pixels.shape, dtype=pixels.dtype) * cond_aug
        pixels = pixels + noise

        posterior = self.vae.apply({"params": vae_params}, pixels, deterministic=True, method=self.vae.encode)
        # Mode of the posterior gives a deterministic 4-channel latent.
        latent = posterior.latent_dist.mode()  # (1, H/8, W/8, 4) channels-last (post-encode NHWC)
        # Convert to PyTorch-like (B, C, H/8, W/8) for the concat stream path.
        latent = jnp.transpose(latent, (0, 3, 1, 2))  # (1, 4, H/8, W/8)
        # Tile over T: (1, 4, H/8, W/8) -> (T, 4, H/8, W/8)
        return jnp.tile(latent, (num_frames, 1, 1, 1))

    def _debug_save_vae_roundtrip(
        self,
        *,
        image,
        params,
        cond_latent,
        num_frames,
        height,
        width,
        debug_dir,
    ):
        """Save the preprocessed conditioning image and a VAE-roundtrip video.

        ``cond_latent`` already has shape ``(T, 4, H/8, W/8)`` in NCHW (the concat
        stream before channel-concat into the UNet). We decode it directly to
        pixel frames — this is what the UNet sees as the target to reproduce.
        If frame 0 of this round-trip does NOT match ``image_0.png``, the VAE
        encoder/decoder load is broken. If it DOES match, the bug is somewhere
        in the UNet forward or the EDM sampling loop.
        """
        import os

        os.makedirs(debug_dir, exist_ok=True)

        # What the VAE encoder actually sees (after resize + normalize).
        prepped = _preprocess_image_for_vae(image, height, width)  # (1, 3, H, W) in [-1, 1]
        prepped_img = ((np.asarray(prepped)[0] + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        prepped_img = np.transpose(prepped_img, (1, 2, 0))  # (H, W, 3)
        Image.fromarray(prepped_img).save(os.path.join(debug_dir, "debug_vae_input.png"))

        # Decode the (unscaled) concat latent. The concat stream is NOT scaled by
        # vae.config.scaling_factor, so don't divide by it here.
        rt = self.vae.apply(
            {"params": params["vae"]},
            cond_latent,
            num_frames=num_frames,
            deterministic=True,
            method=self.vae.decode,
        ).sample  # (T, 3, H, W) in [-1, 1]
        rt = (np.asarray(rt) / 2.0 + 0.5).clip(0.0, 1.0)
        rt = np.transpose(rt, (0, 2, 3, 1))  # (T, H, W, 3)
        # Save frame 0 of the round-trip — this is what the UNet is trying to hit.
        frame0 = (rt[0] * 255).astype(np.uint8)
        Image.fromarray(frame0).save(os.path.join(debug_dir, "debug_vae_roundtrip_frame0.png"))
        # Save the full 14-frame round-trip too so temporal structure is visible.
        for t in range(rt.shape[0]):
            Image.fromarray((rt[t] * 255).astype(np.uint8)).save(
                os.path.join(debug_dir, f"debug_vae_roundtrip_frame{t:02d}.png")
            )

    def _debug_save_pred_x0(self, *, x, params, num_frames, debug_dir):
        """Decode the final sampled latent and save frame 0.

        ``x`` is the last-step latent in scaling_factor-scaled space (the output
        of the Euler loop). Divide by scaling_factor and run the VAE decoder
        exactly like the main output path. This isolates the UNet+sampler from
        any post-processing in the main return path.
        """
        import os

        os.makedirs(debug_dir, exist_ok=True)
        latents = x / self.vae.config.scaling_factor
        decoded = self.vae.apply(
            {"params": params["vae"]},
            latents,
            num_frames=num_frames,
            deterministic=True,
            method=self.vae.decode,
        ).sample  # (B*T, 3, H, W) in [-1, 1]
        frames = (np.asarray(decoded) / 2.0 + 0.5).clip(0.0, 1.0)
        frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, W, 3)
        for t in range(frames.shape[0]):
            Image.fromarray((frames[t] * 255).astype(np.uint8)).save(
                os.path.join(debug_dir, f"debug_final_latent_frame{t:02d}.png")
            )

    def build_adm_vector(self, batch: int, fps_id: int, motion_bucket_id: int, cond_aug: float) -> jnp.ndarray:
        """SVD 768-dim micro-cond vector. Shape (B, 768)."""
        fps = jnp.full((batch,), float(fps_id))
        motion = jnp.full((batch,), float(motion_bucket_id))
        aug = jnp.full((batch,), float(cond_aug))
        return svd_micro_cond_embed(fps, motion, aug, per_dim=256)

    # ---------------------------------------------------------------------------
    # Denoising
    # ---------------------------------------------------------------------------

    def _run_unet(
        self,
        unet_params: Any,
        x_concat: jnp.ndarray,
        sigma: jnp.ndarray,
        encoder_hidden_states: jnp.ndarray,
        adm_vector: jnp.ndarray,
        num_frames: int,
        image_only_indicator: jnp.ndarray,
    ) -> jnp.ndarray:
        """One UNet forward pass. Returns raw v-prediction.

        ``x_concat`` is already the c_in-scaled noisy latent channel-concatenated
        with the cond latent: shape ``(B*T, 8, H/8, W/8)``.
        ``sigma`` is a scalar.
        """
        c_noise = 0.25 * jnp.log(sigma)
        timesteps = jnp.broadcast_to(c_noise, (x_concat.shape[0],))
        out = self.unet.apply(
            {"params": unet_params},
            x_concat,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"adm_vector": adm_vector},
            image_only_indicator=image_only_indicator,
            num_frames=num_frames,
        ).sample
        return out

    # ---------------------------------------------------------------------------
    # Public call
    # ---------------------------------------------------------------------------

    def __call__(
        self,
        params: Union[dict, FrozenDict],
        image: Image.Image,
        prng_seed: jax.Array,
        num_frames: int = 14,
        height: int = 576,
        width: int = 1024,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 2.5,
        fps_id: int = 6,
        motion_bucket_id: int = 127,
        cond_aug: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        debug_dir: Optional[str] = None,
        return_dict: bool = True,
        output_type: str = "np",
    ):
        """Run end-to-end inference.

        ``params`` is a nested pytree with keys ``"vae"``, ``"image_encoder"``,
        ``"unet"`` (scheduler has no trainable params).
        """
        del return_dict
        batch = 1  # SVD base inference is always batch=1.

        # 1. CLIP image embed (cond + uncond).
        image_embeds = self.encode_image_clip(image, params["image_encoder"])  # (1, 1, 1024)
        cond_clip = jnp.tile(image_embeds, (num_frames, 1, 1))  # (T, 1, 1024)
        uncond_clip = jnp.zeros_like(cond_clip)

        # 2. Concat stream (cond + uncond).
        rng_cond, rng_noise = jax.random.split(prng_seed)
        cond_latent = self.encode_cond_latent(
            image,
            params["vae"],
            rng_cond,
            num_frames=num_frames,
            height=height,
            width=width,
            cond_aug=cond_aug,
        )  # (T, 4, H/8, W/8)
        uncond_latent = jnp.zeros_like(cond_latent)

        # Debug: save a VAE round-trip of the conditioning image so the user can
        # verify VAE load + tiling is correct. If this doesn't match image_0.png,
        # the VAE path is broken. If it does, the UNet / sampler is at fault.
        if debug_dir is not None:
            self._debug_save_vae_roundtrip(
                image=image,
                params=params,
                cond_latent=cond_latent,
                num_frames=num_frames,
                height=height,
                width=width,
                debug_dir=debug_dir,
            )

        # 3. ADM vector (cond + uncond).
        # SGM's `GeneralConditioner.get_unconditional_conditioning` zeros the
        # embeddings listed in `force_uc_zero_embeddings=["cond_frames",
        # "cond_frames_without_noise"]` (i.e. only the CLIP and concat streams).
        # The micro-cond scalars (fps_id, motion_bucket_id, cond_aug) are NOT in
        # that list, so their 768-dim embedding is identical between cond and
        # uncond. Diffusers does the same via
        # `_get_add_time_ids` → `torch.cat([add_time_ids, add_time_ids])`.
        # Previously we zeroed adm_uncond, which sent a different time embedding
        # into the uncond UNet call and silently corrupted the CFG extrapolation.
        adm_cond = self.build_adm_vector(batch, fps_id, motion_bucket_id, cond_aug)  # (B, 768)
        adm_uncond = adm_cond

        # 4. Initial noise.
        latents_shape = (
            batch * num_frames,
            self.unet.config.out_channels,  # 4
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        x = jax.random.normal(rng_noise, latents_shape, dtype=jnp.float32)

        # 5. Scheduler and guider.
        scheduler_state = self.scheduler.set_timesteps(
            params.get("scheduler", None) or self.scheduler.create_state(num_inference_steps=num_inference_steps),
            num_inference_steps=num_inference_steps,
        )
        x = x * scheduler_state.init_noise_sigma

        guider = LinearPredictionGuider(
            num_frames=num_frames,
            min_scale=min_guidance_scale,
            max_scale=max_guidance_scale,
        )

        image_only_indicator = jnp.zeros((batch, num_frames), dtype=jnp.float32)

        # 6. Denoising loop — explicit Euler on the EDM sigma grid.
        #
        # Python-driven loop over a jit-compiled single-step function. One
        # compiled program per step means XLA owns per-step intermediates for
        # only that step's call; when the call returns, the UNet activations,
        # v_cond/v_uncond, pred_x0, derivative are all released and only `x`
        # persists. This matches PyTorch's allocator behavior and is what lets
        # 512×512 fit on a 24 GB GPU (the prior `jax.lax.fori_loop` traced all
        # 25 iterations as one program, and XLA's buffer allocator tended to
        # overlap intermediates across iterations → worst-case live set → OOM).
        # `donate_argnums=(0,)` lets XLA reuse the `x` buffer in place.
        # @functools.partial(jax.jit, donate_argnums=(0,))
        # def _sampler_step(x, sigma, next_sigma):
        #     c_skip, c_out, c_in, _ = v_scaling_edm(sigma)

        #     x_in_scaled = x * c_in  # (B*T, 4, H/8, W/8)

        #     # Channel-concat with concat stream (cond and uncond separately).
        #     x_cond_in = jnp.concatenate([x_in_scaled, cond_latent], axis=1)  # (B*T, 8, ...)
        #     x_uncond_in = jnp.concatenate([x_in_scaled, uncond_latent], axis=1)

        #     v_cond = self._run_unet(
        #         params["unet"],
        #         x_cond_in,
        #         sigma,
        #         cond_clip,
        #         adm_cond,
        #         num_frames,
        #         image_only_indicator,
        #     )
        #     v_uncond = self._run_unet(
        #         params["unet"],
        #         x_uncond_in,
        #         sigma,
        #         uncond_clip,
        #         adm_uncond,
        #         num_frames,
        #         image_only_indicator,
        #     )

        #     # Combine v-prediction into denoised sample x0_hat, per-sigma.
        #     pred_x0_cond = c_skip * x + c_out * v_cond
        #     pred_x0_uncond = c_skip * x + c_out * v_uncond

        #     # Per-frame linear CFG.
        #     pred_x0 = guider(pred_x0_cond, pred_x0_uncond)

        #     # Euler step.
        #     derivative = (x - pred_x0) / sigma
        #     dt = next_sigma - sigma
        #     return x + derivative * dt

        # sigmas = scheduler_state.sigmas
        # for i in range(num_inference_steps):
        #     x = _sampler_step(x, sigmas[i], sigmas[i + 1])

        # --- Previous implementation (kept for reference) --------------------
        # Traced as a single fori_loop. Numerically identical to the Python
        # loop above, but at 512×512 on a 24 GB GPU this OOMs because XLA
        # allocates for the worst-case live set across all 25 iterations.
        # See docs/svd_debug_notes.md §6.3.
        #
        def loop_body(i, carry):
            x = carry
            sigma = scheduler_state.sigmas[i]
            c_skip, c_out, c_in, _ = v_scaling_edm(sigma)
            x_in_scaled = x * c_in
            x_cond_in = jnp.concatenate([x_in_scaled, cond_latent], axis=1)
            x_uncond_in = jnp.concatenate([x_in_scaled, uncond_latent], axis=1)
            v_cond = self._run_unet(
                params["unet"],
                x_cond_in,
                sigma,
                cond_clip,
                adm_cond,
                num_frames,
                image_only_indicator,
            )
            v_uncond = self._run_unet(
                params["unet"],
                x_uncond_in,
                sigma,
                uncond_clip,
                adm_uncond,
                num_frames,
                image_only_indicator,
            )
            pred_x0_cond = c_skip * x + c_out * v_cond
            pred_x0_uncond = c_skip * x + c_out * v_uncond
            pred_x0 = guider(pred_x0_cond, pred_x0_uncond)
            derivative = (x - pred_x0) / sigma
            dt = scheduler_state.sigmas[i + 1] - sigma
            return x + derivative * dt

        if DEBUG:
            for i in range(num_inference_steps):
                x = loop_body(i, x)
        else:
            x = jax.lax.fori_loop(0, num_inference_steps, loop_body, x)

        # Debug: decode `x` (the final denoised latent in scaled space) directly.
        # If frame 0 of this save matches the conditioning image, the UNet +
        # sampler converged correctly and any remaining discrepancy is in the
        # VAE decode path (which we already know is correct from the roundtrip).
        # If frame 0 is distorted, the UNet or sampler is still broken.
        if debug_dir is not None:
            self._debug_save_pred_x0(
                x=x,
                params=params,
                num_frames=num_frames,
                debug_dir=debug_dir,
            )

        # 7. Unscale latents and decode frames.
        # Decode in chunks along T to bound peak VAE activation memory. The
        # temporal conv has kernel 3, so chunk boundaries introduce a small
        # temporal discontinuity — Diffusers accepts this same trade-off.
        latents = x / self.vae.config.scaling_factor
        chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        chunk_size = max(1, min(chunk_size, num_frames))
        frame_chunks = []
        for start in range(0, latents.shape[0], chunk_size):
            chunk = latents[start : start + chunk_size]
            frame_chunks.append(
                self.vae.apply(
                    {"params": params["vae"]},
                    chunk,
                    num_frames=chunk.shape[0],
                    deterministic=True,
                    method=self.vae.decode,
                ).sample
            )
        frames = jnp.concatenate(frame_chunks, axis=0)  # (B*T, 3, H, W) in [-1, 1]

        frames = (frames / 2.0 + 0.5).clip(0.0, 1.0)
        if output_type == "np":
            # (T, H, W, 3) numpy in [0, 1]
            frames = jnp.transpose(frames, (0, 2, 3, 1))
            frames = np.asarray(frames)
        return frames
