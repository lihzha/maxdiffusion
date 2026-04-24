# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Inference entry point for Stable Video Diffusion (base).

Example::

    python -m maxdiffusion.generate_svd \
        src/maxdiffusion/configs/base_svd.yml \
        image_url=https://.../astronaut.jpg \
        output_dir=./svd_out
"""

import os
import time
from typing import Sequence

import jax
from absl import app

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.checkpointing.svd_checkpointer import SVDCheckpointer
from maxdiffusion.common_types import SVD
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image


def run(config):
  if config.model_name != SVD:
    raise ValueError(f"generate_svd expected model_name={SVD}, got {config.model_name}")

  max_logging.log(
      f"[svd] {config.num_frames} frames @ {config.height}x{config.width}, "
      f"{config.num_inference_steps} EDM steps, "
      f"fps_id={config.fps_id}, motion_bucket_id={config.motion_bucket_id}, "
      f"cond_aug={config.cond_aug}, "
      f"cfg_min={config.min_guidance_scale}, cfg_max={config.max_guidance_scale}"
  )

  t0 = time.perf_counter()
  loader = SVDCheckpointer(config)
  pipeline, params = loader.load_diffusers_checkpoint()
  # Checkpoint staged on CPU to avoid thrashing the GPU during load. Migrate
  # the leaves onto the accelerator before inference so sharding_constraint
  # inside the fori_loop doesn't see a CPU/GPU mismatch.
  accel_device = jax.devices()[0]

  def _to_accel(leaf):
    if isinstance(leaf, jax.Array):
      return jax.device_put(leaf, accel_device)
    return leaf

  params = jax.tree_util.tree_map(_to_accel, params)
  max_logging.log(f"[svd] checkpoint loaded in {time.perf_counter() - t0:.1f}s")

  pil_image = load_image(config.image_url)
  prng_seed = jax.random.PRNGKey(config.seed)

  t1 = time.perf_counter()
  # FlaxAttention's with_sharding_constraint inside the denoising loop needs
  # an active mesh context; the pipeline doesn't open its own.
  with loader.mesh:
    frames = pipeline(
        params=params,
        image=pil_image,
        prng_seed=prng_seed,
        num_frames=config.num_frames,
        height=config.height,
        width=config.width,
        num_inference_steps=config.num_inference_steps,
        min_guidance_scale=config.min_guidance_scale,
        max_guidance_scale=config.max_guidance_scale,
        fps_id=config.fps_id,
        motion_bucket_id=config.motion_bucket_id,
        cond_aug=config.cond_aug,
        decode_chunk_size=getattr(config, "decode_chunk_size", None),
        debug_dir=getattr(config, "debug_dir", None) or None,
    )
  max_logging.log(f"[svd] generation finished in {time.perf_counter() - t1:.1f}s")

  os.makedirs(config.output_dir, exist_ok=True)
  video_path = os.path.join(config.output_dir, f"svd_output_{config.seed}.mp4")
  # frames: (T, H, W, 3) in [0, 1].
  export_to_video([f for f in frames], video_path, fps=int(config.fps))
  max_logging.log(f"[svd] wrote {video_path}")


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
