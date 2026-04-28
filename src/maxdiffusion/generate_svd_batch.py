# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Batched Stable Video Diffusion inference.

Loads the checkpoint and compiles the pipeline ONCE, then iterates over a list
of input images in the same Python process so JAX's JIT cache is reused.

Usage::

    python -m maxdiffusion.generate_svd_batch \\
        src/maxdiffusion/configs/base_svd_gpu.yml \\
        image_url=/path/a.png,/path/b.png \\
        output_dir=/path/out \\
        seed=23

Each image is sampled at its own (H, W) read from the PIL image. A new unique
(H, W) triggers one XLA compile on first encounter; subsequent calls with the
same shape hit the cache. Output files are ``{output_dir}/{stem}_svd_{seed}.mp4``.
"""

import os
import time
from pathlib import Path
from typing import Sequence

import jax
from absl import app

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.checkpointing.svd_checkpointer import SVDCheckpointer
from maxdiffusion.common_types import SVD
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image


def run(config):
  if config.model_name != SVD:
    raise ValueError(
        f"generate_svd_batch expected model_name={SVD}, got {config.model_name}"
    )

  image_urls = [p.strip() for p in str(config.image_url).split(",") if p.strip()]
  if not image_urls:
    raise ValueError("image_url must be a non-empty comma-separated list of paths.")

  max_logging.log(
      f"[svd-batch] {len(image_urls)} image(s), {config.num_frames} frames, "
      f"{config.num_inference_steps} EDM steps, "
      f"fps_id={config.fps_id}, motion_bucket_id={config.motion_bucket_id}, "
      f"cond_aug={config.cond_aug}, "
      f"cfg=[{config.min_guidance_scale},{config.max_guidance_scale}]"
  )

  t0 = time.perf_counter()
  loader = SVDCheckpointer(config)
  pipeline, params = loader.load_diffusers_checkpoint()
  accel_device = jax.devices()[0]

  def _to_accel(leaf):
    if isinstance(leaf, jax.Array):
      return jax.device_put(leaf, accel_device)
    return leaf

  params = jax.tree_util.tree_map(_to_accel, params)
  max_logging.log(f"[svd-batch] checkpoint loaded in {time.perf_counter() - t0:.1f}s")

  os.makedirs(config.output_dir, exist_ok=True)
  # Derive a distinct sub-key per image so same-shape images don't share noise.
  keys = jax.random.split(jax.random.PRNGKey(config.seed), len(image_urls))

  for i, image_url in enumerate(image_urls):
    pil_image = load_image(image_url)
    w, h = pil_image.size
    if h % 8 or w % 8:
      raise ValueError(
          f"{image_url} dims {w}x{h}: both sides must be divisible by 8 (VAE scale)."
      )
    max_logging.log(f"[svd-batch] ({i + 1}/{len(image_urls)}) {image_url} @ {h}x{w}")

    t1 = time.perf_counter()
    with loader.mesh:
      frames = pipeline(
          params=params,
          image=pil_image,
          prng_seed=keys[i],
          num_frames=config.num_frames,
          height=h,
          width=w,
          num_inference_steps=config.num_inference_steps,
          min_guidance_scale=config.min_guidance_scale,
          max_guidance_scale=config.max_guidance_scale,
          fps_id=config.fps_id,
          motion_bucket_id=config.motion_bucket_id,
          cond_aug=config.cond_aug,
          decode_chunk_size=getattr(config, "decode_chunk_size", None),
          debug_dir=getattr(config, "debug_dir", None) or None,
      )
    max_logging.log(
        f"[svd-batch] ({i + 1}/{len(image_urls)}) generation finished in "
        f"{time.perf_counter() - t1:.1f}s"
    )

    stem = Path(image_url).stem
    video_path = os.path.join(config.output_dir, f"{stem}_svd_{config.seed}.mp4")
    export_to_video([f for f in frames], video_path, fps=int(config.fps))
    max_logging.log(f"[svd-batch] wrote {video_path}")


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
