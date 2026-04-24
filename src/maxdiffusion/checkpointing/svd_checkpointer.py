# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0

"""Checkpoint loader for Stable Video Diffusion (base).

Loads each module of ``stabilityai/stable-video-diffusion-img2vid`` from its
Diffusers repo layout:

  pretrained_model_name_or_path/
    unet/
    vae/
    image_encoder/
    feature_extractor/
    scheduler/

Relies on the shared ``FlaxModelMixin.from_pretrained`` → ``from_pt=True``
PyTorch → Flax converter in ``models/modeling_flax_pytorch_utils.py``. Any
missing or shape-mismatched keys are logged by ``validate_flax_state_dict``.
"""

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict
from jax.sharding import Mesh

from .. import max_logging, max_utils
from ..common_types import SVD
from ..models.modeling_flax_pytorch_utils import validate_flax_state_dict
from ..models.svd.video_autoencoder_flax import FlaxSVDAutoencoderKL
from ..models.svd.video_unet_flax import FlaxVideoUNet
from ..pipelines.svd.pipeline_flax_svd import FlaxStableVideoDiffusionPipeline
from ..schedulers.scheduling_edm_euler_flax import FlaxEDMEulerScheduler


class _TorchCLIPVisionWithProjection:
  """PyTorch-backed shim for ``CLIPVisionModelWithProjection``.

  ``transformers==4.57+`` dropped ``FlaxCLIPVisionModelWithProjection`` but kept
  the PyTorch variant. The SVD image encoder runs once per generation (outside
  the jitted denoising loop), so doing the forward in torch is cheap and avoids
  porting OpenCLIP-style projection weights into Flax.

  Mimics HF's Flax API surface so ``pipeline.encode_image_clip`` doesn't need
  to know which backend is live: ``image_encoder(pixel_values, params=...)``
  returns an object with a jnp ``image_embeds`` attribute; ``.params`` is an
  empty dict (weights live inside the torch module).
  """

  def __init__(self, torch_model):
    import torch
    self._torch = torch  # cache handle
    self.model = torch_model.eval()
    # Keep on CPU — the encoder is small and runs once per generation. Putting
    # it on the GPU would fight jax for memory.
    self.model = self.model.to("cpu")
    self.params = {}

  @classmethod
  def from_pretrained(cls, repo_id, subfolder=None, **kwargs):
    from transformers import CLIPVisionModelWithProjection
    m = CLIPVisionModelWithProjection.from_pretrained(
        repo_id, subfolder=subfolder or "", torch_dtype="float32"
    )
    return cls(m)

  def __call__(self, pixel_values, params=None):
    torch = self._torch
    pv = np.asarray(pixel_values)
    with torch.no_grad():
      out = self.model(torch.from_numpy(pv))
    image_embeds = jnp.asarray(np.asarray(out.image_embeds))

    class _Out:
      pass
    r = _Out()
    r.image_embeds = image_embeds
    return r


def _strict_validate_state_dict(
    eval_shapes, loaded_params, component: str
) -> None:
  """Compare ``loaded_params`` against ``eval_shapes`` and raise on any mismatch.

  ``FlaxModelMixin.from_pretrained(from_pt=True)`` silently keeps init-random
  weights for any PT key the converter fails to map. This wrapper catches that
  class of silent failure by diffing key sets and shapes against the expected
  init shape tree, then raising a ``RuntimeError`` if anything is off.

  Also calls :func:`validate_flax_state_dict` for its per-key log output.
  """
  expected_flat = flatten_dict(eval_shapes)
  loaded_flat = flatten_dict(loaded_params)

  expected_keys = set(expected_flat.keys())
  loaded_keys = set(loaded_flat.keys())
  missing = expected_keys - loaded_keys
  extra = loaded_keys - expected_keys

  shape_mismatches = []
  for k in expected_keys & loaded_keys:
    exp = expected_flat[k]
    exp_shape = getattr(exp, "shape", None)
    if exp_shape is None:
      exp_shape = exp.value.shape  # Partitioned
    got_shape = loaded_flat[k].shape
    if exp_shape != got_shape:
      shape_mismatches.append((k, exp_shape, got_shape))

  # Let the shared helper emit its per-key log.
  validate_flax_state_dict(eval_shapes, loaded_flat)

  if missing or extra or shape_mismatches:
    details = []
    if missing:
      details.append(f"{len(missing)} missing keys (e.g. {sorted(missing)[:3]})")
    if extra:
      details.append(f"{len(extra)} extra keys (e.g. {sorted(extra)[:3]})")
    if shape_mismatches:
      details.append(
          f"{len(shape_mismatches)} shape mismatches "
          f"(e.g. {shape_mismatches[0][0]}: expected {shape_mismatches[0][1]}, "
          f"got {shape_mismatches[0][2]})"
      )
    raise RuntimeError(
        f"[svd] {component} weight validation failed: " + "; ".join(details)
    )
  max_logging.log(
      f"[svd] {component} weights validated: {len(expected_keys)} keys OK."
  )


class SVDCheckpointer:
  """Minimal Diffusers-format loader for SVD.

  Unlike ``BaseStableDiffusionCheckpointer`` this is inference-focused —
  no orbax checkpointing, no optimizer state, no training hooks.
  """

  def __init__(self, config):
    if config.model_name != SVD:
      raise ValueError(
          f"SVDCheckpointer expected model_name={SVD}, got {config.model_name!r}"
      )
    self.config = config
    self.rng = jax.random.PRNGKey(config.seed)
    devices_array = max_utils.create_device_mesh(config)
    self.mesh = Mesh(devices_array, config.mesh_axes)

  def load_diffusers_checkpoint(self):
    """Load UNet + VAE + image encoder + feature extractor from HF repo.

    Returns ``(pipeline, params)`` where ``params`` is a dict keyed by
    ``"unet"``, ``"vae"``, ``"image_encoder"``.
    """
    from transformers import CLIPImageProcessor

    precision = max_utils.get_precision(self.config)
    flash_block_sizes = max_utils.get_flash_block_sizes(self.config)

    context = (
        jax.default_device(jax.devices("cpu")[0])
        if jax.device_count() == jax.local_device_count()
        else nullcontext()
    )

    repo_id = self.config.pretrained_model_name_or_path
    # HF Hub rejects empty-string revisions (tries to resolve "" as a git ref).
    revision = self.config.revision or None

    # init_weights for the UNet triggers ``jax.lax.with_sharding_constraint``
    # inside ``FlaxAttention``, which requires an active mesh. The same
    # ``from_pretrained`` path calls ``init_weights`` internally to build the
    # shape template, so everything weight-related must run under ``self.mesh``.
    with context, self.mesh:
      max_logging.log(f"[svd] loading UNet from {repo_id}/unet ...")
      unet, unet_params = FlaxVideoUNet.from_pretrained(
          repo_id,
          subfolder="unet",
          revision=revision,
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          from_pt=self.config.from_pt,
          use_safetensors=True,
          attention_kernel=self.config.attention,
          temporal_attention_kernel=getattr(self.config, "temporal_attention", "dot_product"),
          use_memory_efficient_attention=getattr(
              self.config, "use_memory_efficient_attention", False
          ),
          flash_block_sizes=flash_block_sizes,
          flash_min_seq_length=self.config.flash_min_seq_length,
          mesh=self.mesh,
          precision=precision,
          norm_num_groups=self.config.norm_num_groups,
      )
      if self.config.from_pt:
        unet_eval_shapes = unet.init_weights(self.rng, eval_only=True)
        _strict_validate_state_dict(unet_eval_shapes, unet_params, component="UNet")

      max_logging.log(f"[svd] loading VAE from {repo_id}/vae ...")
      vae, vae_params = FlaxSVDAutoencoderKL.from_pretrained(
          repo_id,
          subfolder="vae",
          revision=revision,
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          from_pt=self.config.from_pt,
          use_safetensors=True,
      )
      if self.config.from_pt:
        vae_eval_shapes = vae.init_weights(self.rng, eval_only=True)
        _strict_validate_state_dict(vae_eval_shapes, vae_params, component="VAE")

      # HF image encoder: CLIP ViT-H/14 with projection head (1024-dim output).
      # Diffusers ships the safetensors under repo_id/image_encoder.
      image_encoder_repo = getattr(
          self.config, "image_encoder_pretrained_model_name_or_path", None
      ) or repo_id
      image_encoder_subfolder = (
          None if image_encoder_repo != repo_id else "image_encoder"
      )
      max_logging.log(
          f"[svd] loading CLIP vision encoder from {image_encoder_repo}"
          f"{'/' + image_encoder_subfolder if image_encoder_subfolder else ''} ..."
      )
      image_encoder = _TorchCLIPVisionWithProjection.from_pretrained(
          image_encoder_repo,
          subfolder=image_encoder_subfolder,
      )
      image_encoder_params = image_encoder.params  # {} — weights live in torch

      # Feature extractor (CLIPImageProcessor) — no learnable parameters.
      try:
        feature_extractor = CLIPImageProcessor.from_pretrained(
            repo_id, subfolder="feature_extractor"
        )
      except Exception:
        feature_extractor = CLIPImageProcessor.from_pretrained(image_encoder_repo)

      # Build a fresh EDM scheduler — we ignore Diffusers' EulerDiscrete config
      # because SVD uses EDM preconditioning (c_skip/c_out/c_in/c_noise) with a
      # Karras rho=7 sigma schedule that Diffusers' scheduler doesn't encode.
      scheduler = FlaxEDMEulerScheduler(
          sigma_min=float(self.config.diffusion_scheduler_config.get("sigma_min", 0.002)),
          sigma_max=float(self.config.diffusion_scheduler_config.get("sigma_max", 700.0)),
          rho=float(self.config.diffusion_scheduler_config.get("rho", 7.0)),
          prediction_type="v_prediction",
          dtype=self.config.weights_dtype,
      )

    pipeline = FlaxStableVideoDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=scheduler,
        dtype=self.config.activations_dtype,
    )
    params = {
        "unet": unet_params,
        "vae": vae_params,
        "image_encoder": image_encoder_params,
    }
    params = jax.tree_util.tree_map(
        lambda x: x.astype(self.config.weights_dtype), params
    )
    return pipeline, params
