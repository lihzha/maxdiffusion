# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Outer VAE module for Stable Video Diffusion (base).

Encoder is the 2D SD VAE encoder (``FlaxEncoder`` from ``models/vae_flax.py``)
operating per-frame. Decoder is :class:`FlaxVideoDecoder` with interleaved
temporal convolutions.

Checkpoint layout mirrors SD VAE: ``encoder.*``, ``decoder.*``,
``quant_conv.*``, ``post_quant_conv.*``.
"""

from typing import Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ...configuration_utils import ConfigMixin, flax_register_to_config
from ...utils import BaseOutput
from ..modeling_flax_utils import FlaxModelMixin
from ..vae_flax import FlaxDiagonalGaussianDistribution, FlaxEncoder
from .video_decoder_flax import FlaxVideoDecoder


@flax.struct.dataclass
class FlaxSVDDecoderOutput(BaseOutput):
  """Decoder output: ``(B*T, C, H, W)`` pixel sample."""

  sample: jnp.ndarray


@flax.struct.dataclass
class FlaxSVDAutoencoderKLOutput(BaseOutput):
  """Encoder output: posterior distribution."""

  latent_dist: FlaxDiagonalGaussianDistribution


@flax_register_to_config
class FlaxSVDAutoencoderKL(nn.Module, FlaxModelMixin, ConfigMixin):
  """Autoencoder for Stable Video Diffusion.

  Encode runs the spatial SD encoder frame-by-frame. Decode runs the temporal
  :class:`FlaxVideoDecoder` which threads ``num_frames`` through every video
  block. Scaling factor is SD-1.x's 0.18215 (NOT SDXL's 0.13025).
  """

  in_channels: int = 3
  out_channels: int = 3
  down_block_types: Tuple[str, ...] = (
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
  )
  up_block_types: Tuple[str, ...] = (
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
  )
  block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
  layers_per_block: int = 2
  act_fn: str = "silu"
  latent_channels: int = 4
  norm_num_groups: int = 32
  sample_size: int = 72
  scaling_factor: float = 0.18215
  video_kernel_size: Tuple[int, int, int] = (3, 1, 1)
  merge_strategy: str = "learned"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  # Diffusers' AutoencoderKLTemporalDecoder has a quant_conv (1x1 on the 2*z
  # encoder moments) but NO post_quant_conv — the decoder consumes the 4-channel
  # latent directly. Keep a single toggle for encode-side; no post-quant path.
  use_quant_conv: bool = True

  def setup(self):
    self.encoder = FlaxEncoder(
        in_channels=self.config.in_channels,
        out_channels=self.config.latent_channels,
        down_block_types=self.config.down_block_types,
        block_out_channels=self.config.block_out_channels,
        layers_per_block=self.config.layers_per_block,
        act_fn=self.config.act_fn,
        norm_num_groups=self.config.norm_num_groups,
        double_z=True,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )
    self.decoder = FlaxVideoDecoder(
        in_channels=self.config.latent_channels,
        out_channels=self.config.out_channels,
        up_block_types=self.config.up_block_types,
        block_out_channels=self.config.block_out_channels,
        layers_per_block=self.config.layers_per_block,
        norm_num_groups=self.config.norm_num_groups,
        act_fn=self.config.act_fn,
        video_kernel_size=self.config.video_kernel_size,
        merge_strategy=self.config.merge_strategy,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )
    if self.use_quant_conv:
      self.quant_conv = nn.Conv(
          2 * self.config.latent_channels,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
      )

  def init_weights(self, rng: jax.Array, eval_only: bool = False) -> FrozenDict:
    # Use num_frames=2 at init so every temporal conv (kernel=3 with pad=1) has
    # a non-trivial time axis. Minimal shape keeps init cheap.
    num_frames = 2
    sample_shape = (num_frames, self.in_channels, self.sample_size, self.sample_size)
    if eval_only:
      sample = jax.ShapeDtypeStruct(sample_shape, dtype=jnp.float32)
    else:
      sample = jnp.zeros(sample_shape, dtype=jnp.float32)

    params_rng, dropout_rng, gaussian_rng = jax.random.split(rng, 3)
    rngs = {"params": params_rng, "dropout": dropout_rng, "gaussian": gaussian_rng}

    def _init_fn(rngs, sample):
      # Drive both encode and decode to exercise every sub-module.
      posterior = self.encode(sample, deterministic=True).latent_dist
      z = posterior.mode()
      return self.decode(z, num_frames=num_frames, deterministic=True).sample

    if eval_only:
      return jax.eval_shape(lambda r, s: self.init(r, s, num_frames=num_frames), rngs, sample)["params"]
    return self.init(rngs, sample, num_frames=num_frames)["params"]

  def encode(
      self,
      sample: jnp.ndarray,
      deterministic: bool = True,
      return_dict: bool = True,
  ):
    """Encode pixel frames.

    ``sample`` is ``(B*T, C, H, W)`` (PyTorch-style). Frames are encoded
    independently — no temporal mixing in the encoder.
    """
    sample = jnp.transpose(sample, (0, 2, 3, 1))  # NCHW -> NHWC

    hidden_states = self.encoder(sample, deterministic=deterministic)
    moments = hidden_states
    if self.use_quant_conv:
      moments = self.quant_conv(hidden_states)
    posterior = FlaxDiagonalGaussianDistribution(moments)

    if not return_dict:
      return (posterior,)
    return FlaxSVDAutoencoderKLOutput(latent_dist=posterior)

  def decode(
      self,
      latents: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      return_dict: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ):
    """Decode video latents.

    ``latents`` is ``(B*T, C, H/8, W/8)`` or ``(B*T, H/8, W/8, C)`` (detected
    by the channel dim). ``num_frames`` is ``T``, a static Python int.
    """
    if latents.shape[-1] != self.config.latent_channels:
      latents = jnp.transpose(latents, (0, 2, 3, 1))  # NCHW -> NHWC

    # AutoencoderKLTemporalDecoder has no post_quant_conv; feed the latent
    # directly into the temporal decoder.
    hidden_states = self.decoder(
        latents,
        num_frames=num_frames,
        deterministic=deterministic,
        image_only_indicator=image_only_indicator,
    )

    hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))  # NHWC -> NCHW

    if not return_dict:
      return (hidden_states,)
    return FlaxSVDDecoderOutput(sample=hidden_states)

  def __call__(
      self,
      sample: jnp.ndarray,
      num_frames: int,
      sample_posterior: bool = False,
      deterministic: bool = True,
      return_dict: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ):
    posterior = self.encode(sample, deterministic=deterministic, return_dict=return_dict)
    latent_dist = posterior.latent_dist if return_dict else posterior[0]
    if sample_posterior:
      rng = self.make_rng("gaussian")
      z = latent_dist.sample(rng)
    else:
      z = latent_dist.mode()
    out = self.decode(
        z,
        num_frames=num_frames,
        deterministic=deterministic,
        return_dict=return_dict,
        image_only_indicator=image_only_indicator,
    )
    if not return_dict:
      return (out[0],)
    return out
