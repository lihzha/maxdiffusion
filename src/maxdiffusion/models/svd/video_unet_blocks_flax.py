# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VideoUNet down/up/mid blocks.

Forks of the spatial-only UNet blocks in
``models/unet_2d_blocks_flax.py``. Changes:

  * each ``FlaxResnetBlock2D`` is wrapped by :class:`FlaxVideoResBlockUNet`
    (spatial + temporal + AlphaBlender);
  * each ``FlaxTransformer2DModel`` is replaced by
    :class:`FlaxSpatialVideoTransformer`;
  * ``num_frames`` and ``image_only_indicator`` are threaded through every
    ``__call__``.

The tuples of residuals returned by the down blocks (and popped by the up
blocks) match the spatial UNet's contract, so the outer UNet's skip-connection
plumbing is unchanged.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..resnet_flax import FlaxDownsample2D, FlaxUpsample2D
from ...common_types import BlockSizes
from .. import quantizations
from .video_attention_flax import FlaxSpatialVideoTransformer
from .video_blocks_flax import FlaxVideoResBlockUNet

Quant = quantizations.AqtQuantization


class FlaxCrossAttnDownVideoBlock(nn.Module):
  """Cross-attention video down-block.

  Each layer: ``FlaxVideoResBlockUNet`` → ``FlaxSpatialVideoTransformer``.
  Optionally followed by a spatial ``FlaxDownsample2D``.
  """

  in_channels: int
  out_channels: int
  dropout: float = 0.0
  num_layers: int = 1
  num_attention_heads: int = 1
  add_downsample: bool = True
  use_linear_projection: bool = True
  context_dim: int = 1024
  transformer_layers_per_block: int = 1
  norm_num_groups: int = 32
  # Diffusers' CrossAttn{Down,Up}BlockSpatioTemporal passes eps=1e-6 to
  # SpatioTemporalResBlock. The Mid block and plain Down/UpBlockSpatioTemporal
  # use eps=1e-5. Each block hard-codes the right default for its Diffusers
  # counterpart below.
  resnet_eps: float = 1e-6
  video_kernel_size: tuple = (3, 1, 1)
  merge_strategy: str = "learned_with_images"
  attention_kernel: str = "dot_product"
  temporal_attention_kernel: str = "dot_product"
  # Chunked-query attention (avoids N×N scores). Plumbed through to
  # FlaxSpatialVideoTransformer → FlaxBasicTransformerBlock → FlaxAttention.
  use_memory_efficient_attention: bool = False
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  # "default" | "scale_shift" (AdaGN); set by FlaxVideoUNet only when
  # action_cond_mode == "adaln". See FlaxResnetBlock2D.time_embedding_norm.
  time_embedding_norm: str = "default"
  quant: Quant = None

  def setup(self):
    resnets = []
    attentions = []
    for i in range(self.num_layers):
      in_c = self.in_channels if i == 0 else self.out_channels
      resnets.append(
          FlaxVideoResBlockUNet(
              in_channels=in_c,
              out_channels=self.out_channels,
              dropout=self.dropout,
              norm_num_groups=self.norm_num_groups,
              norm_eps=self.resnet_eps,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              time_embedding_norm=self.time_embedding_norm,
          )
      )
      attentions.append(
          FlaxSpatialVideoTransformer(
              in_channels=self.out_channels,
              n_heads=self.num_attention_heads,
              d_head=self.out_channels // self.num_attention_heads,
              depth=self.transformer_layers_per_block,
              use_linear_projection=self.use_linear_projection,
              context_dim=self.context_dim,
              norm_num_groups=self.norm_num_groups,
              merge_strategy=self.merge_strategy,
              attention_kernel=self.attention_kernel,
              temporal_attention_kernel=self.temporal_attention_kernel,
              use_memory_efficient_attention=self.use_memory_efficient_attention,
              flash_min_seq_length=self.flash_min_seq_length,
              flash_block_sizes=self.flash_block_sizes,
              mesh=self.mesh,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              quant=self.quant,
          )
      )
    self.resnets = resnets
    self.attentions = attentions
    if self.add_downsample:
      self.downsamplers_0 = FlaxDownsample2D(
          self.out_channels, dtype=self.dtype, weights_dtype=self.weights_dtype
      )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      temb: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
      cross_attention_kwargs=None,
  ):
    output_states = ()
    for resnet, attn in zip(self.resnets, self.attentions):
      hidden_states = resnet(
          hidden_states, temb, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
      )
      hidden_states = attn(
          hidden_states, encoder_hidden_states, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
          cross_attention_kwargs=cross_attention_kwargs,
      )
      output_states += (hidden_states,)
    if self.add_downsample:
      hidden_states = self.downsamplers_0(hidden_states)
      output_states += (hidden_states,)
    return hidden_states, output_states


class FlaxDownVideoBlock(nn.Module):
  """Video down-block without attention (bottom stage of UNet)."""

  in_channels: int
  out_channels: int
  dropout: float = 0.0
  num_layers: int = 1
  add_downsample: bool = True
  norm_num_groups: int = 32
  # Diffusers' DownBlockSpatioTemporal uses eps=1e-5.
  resnet_eps: float = 1e-5
  video_kernel_size: tuple = (3, 1, 1)
  merge_strategy: str = "learned_with_images"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  # "default" | "scale_shift" (AdaGN); set by FlaxVideoUNet only when
  # action_cond_mode == "adaln". See FlaxResnetBlock2D.time_embedding_norm.
  time_embedding_norm: str = "default"

  def setup(self):
    resnets = []
    for i in range(self.num_layers):
      in_c = self.in_channels if i == 0 else self.out_channels
      resnets.append(
          FlaxVideoResBlockUNet(
              in_channels=in_c,
              out_channels=self.out_channels,
              dropout=self.dropout,
              norm_num_groups=self.norm_num_groups,
              norm_eps=self.resnet_eps,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              time_embedding_norm=self.time_embedding_norm,
          )
      )
    self.resnets = resnets
    if self.add_downsample:
      self.downsamplers_0 = FlaxDownsample2D(
          self.out_channels, dtype=self.dtype, weights_dtype=self.weights_dtype
      )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      temb: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ):
    output_states = ()
    for resnet in self.resnets:
      hidden_states = resnet(
          hidden_states, temb, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
      )
      output_states += (hidden_states,)
    if self.add_downsample:
      hidden_states = self.downsamplers_0(hidden_states)
      output_states += (hidden_states,)
    return hidden_states, output_states


class FlaxCrossAttnUpVideoBlock(nn.Module):
  """Cross-attention video up-block."""

  in_channels: int
  out_channels: int
  prev_output_channel: int
  dropout: float = 0.0
  num_layers: int = 1
  num_attention_heads: int = 1
  add_upsample: bool = True
  use_linear_projection: bool = True
  context_dim: int = 1024
  transformer_layers_per_block: int = 1
  norm_num_groups: int = 32
  # Diffusers' CrossAttn{Down,Up}BlockSpatioTemporal passes eps=1e-6 to
  # SpatioTemporalResBlock. The Mid block and plain Down/UpBlockSpatioTemporal
  # use eps=1e-5. Each block hard-codes the right default for its Diffusers
  # counterpart below.
  resnet_eps: float = 1e-6
  video_kernel_size: tuple = (3, 1, 1)
  merge_strategy: str = "learned_with_images"
  attention_kernel: str = "dot_product"
  temporal_attention_kernel: str = "dot_product"
  # Chunked-query attention (avoids N×N scores). Plumbed through to
  # FlaxSpatialVideoTransformer → FlaxBasicTransformerBlock → FlaxAttention.
  use_memory_efficient_attention: bool = False
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  # "default" | "scale_shift" (AdaGN); set by FlaxVideoUNet only when
  # action_cond_mode == "adaln". See FlaxResnetBlock2D.time_embedding_norm.
  time_embedding_norm: str = "default"
  quant: Quant = None

  def setup(self):
    resnets = []
    attentions = []
    for i in range(self.num_layers):
      res_skip_channels = self.in_channels if i == self.num_layers - 1 else self.out_channels
      resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels
      resnets.append(
          FlaxVideoResBlockUNet(
              in_channels=resnet_in_channels + res_skip_channels,
              out_channels=self.out_channels,
              dropout=self.dropout,
              norm_num_groups=self.norm_num_groups,
              norm_eps=self.resnet_eps,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              time_embedding_norm=self.time_embedding_norm,
          )
      )
      attentions.append(
          FlaxSpatialVideoTransformer(
              in_channels=self.out_channels,
              n_heads=self.num_attention_heads,
              d_head=self.out_channels // self.num_attention_heads,
              depth=self.transformer_layers_per_block,
              use_linear_projection=self.use_linear_projection,
              context_dim=self.context_dim,
              norm_num_groups=self.norm_num_groups,
              merge_strategy=self.merge_strategy,
              attention_kernel=self.attention_kernel,
              temporal_attention_kernel=self.temporal_attention_kernel,
              use_memory_efficient_attention=self.use_memory_efficient_attention,
              flash_min_seq_length=self.flash_min_seq_length,
              flash_block_sizes=self.flash_block_sizes,
              mesh=self.mesh,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              quant=self.quant,
          )
      )
    self.resnets = resnets
    self.attentions = attentions
    if self.add_upsample:
      self.upsamplers_0 = FlaxUpsample2D(
          self.out_channels, dtype=self.dtype, weights_dtype=self.weights_dtype
      )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      res_hidden_states_tuple,
      temb: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
      cross_attention_kwargs=None,
  ):
    for resnet, attn in zip(self.resnets, self.attentions):
      res_hidden_states = res_hidden_states_tuple[-1]
      res_hidden_states_tuple = res_hidden_states_tuple[:-1]
      hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)
      hidden_states = resnet(
          hidden_states, temb, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
      )
      hidden_states = attn(
          hidden_states, encoder_hidden_states, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
          cross_attention_kwargs=cross_attention_kwargs,
      )
    if self.add_upsample:
      hidden_states = self.upsamplers_0(hidden_states)
    return hidden_states


class FlaxUpVideoBlock(nn.Module):
  """Video up-block without attention (top stage of UNet)."""

  in_channels: int
  out_channels: int
  prev_output_channel: int
  dropout: float = 0.0
  num_layers: int = 1
  add_upsample: bool = True
  norm_num_groups: int = 32
  # Diffusers' UpBlockSpatioTemporal uses eps=1e-5.
  resnet_eps: float = 1e-5
  video_kernel_size: tuple = (3, 1, 1)
  merge_strategy: str = "learned_with_images"
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  # "default" | "scale_shift" (AdaGN); set by FlaxVideoUNet only when
  # action_cond_mode == "adaln". See FlaxResnetBlock2D.time_embedding_norm.
  time_embedding_norm: str = "default"

  def setup(self):
    resnets = []
    for i in range(self.num_layers):
      res_skip_channels = self.in_channels if i == self.num_layers - 1 else self.out_channels
      resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels
      resnets.append(
          FlaxVideoResBlockUNet(
              in_channels=resnet_in_channels + res_skip_channels,
              out_channels=self.out_channels,
              dropout=self.dropout,
              norm_num_groups=self.norm_num_groups,
              norm_eps=self.resnet_eps,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              time_embedding_norm=self.time_embedding_norm,
          )
      )
    self.resnets = resnets
    if self.add_upsample:
      self.upsamplers_0 = FlaxUpsample2D(self.out_channels, dtype=self.dtype)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      res_hidden_states_tuple,
      temb: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ):
    for resnet in self.resnets:
      res_hidden_states = res_hidden_states_tuple[-1]
      res_hidden_states_tuple = res_hidden_states_tuple[:-1]
      hidden_states = jnp.concatenate((hidden_states, res_hidden_states), axis=-1)
      hidden_states = resnet(
          hidden_states, temb, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
      )
    if self.add_upsample:
      hidden_states = self.upsamplers_0(hidden_states)
    return hidden_states


class FlaxVideoMidBlock2DCrossAttn(nn.Module):
  """Mid block: video_resblock → (spatial_video_transformer → video_resblock)^num_layers."""

  in_channels: int
  dropout: float = 0.0
  num_layers: int = 1
  num_attention_heads: int = 1
  use_linear_projection: bool = True
  context_dim: int = 1024
  transformer_layers_per_block: int = 1
  norm_num_groups: int = 32
  # Diffusers' UNetMidBlockSpatioTemporal passes eps=1e-5.
  resnet_eps: float = 1e-5
  video_kernel_size: tuple = (3, 1, 1)
  merge_strategy: str = "learned_with_images"
  attention_kernel: str = "dot_product"
  temporal_attention_kernel: str = "dot_product"
  # Chunked-query attention (avoids N×N scores). Plumbed through to
  # FlaxSpatialVideoTransformer → FlaxBasicTransformerBlock → FlaxAttention.
  use_memory_efficient_attention: bool = False
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  # "default" | "scale_shift" (AdaGN); set by FlaxVideoUNet only when
  # action_cond_mode == "adaln". See FlaxResnetBlock2D.time_embedding_norm.
  time_embedding_norm: str = "default"
  quant: Quant = None

  def setup(self):
    resnets = [
        FlaxVideoResBlockUNet(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            dropout=self.dropout,
            norm_num_groups=self.norm_num_groups,
            norm_eps=self.resnet_eps,
            video_kernel_size=self.video_kernel_size,
            merge_strategy=self.merge_strategy,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
            time_embedding_norm=self.time_embedding_norm,
        )
    ]
    attentions = []
    for _ in range(self.num_layers):
      attentions.append(
          FlaxSpatialVideoTransformer(
              in_channels=self.in_channels,
              n_heads=self.num_attention_heads,
              d_head=self.in_channels // self.num_attention_heads,
              depth=self.transformer_layers_per_block,
              use_linear_projection=self.use_linear_projection,
              context_dim=self.context_dim,
              norm_num_groups=self.norm_num_groups,
              merge_strategy=self.merge_strategy,
              attention_kernel=self.attention_kernel,
              temporal_attention_kernel=self.temporal_attention_kernel,
              use_memory_efficient_attention=self.use_memory_efficient_attention,
              flash_min_seq_length=self.flash_min_seq_length,
              flash_block_sizes=self.flash_block_sizes,
              mesh=self.mesh,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              quant=self.quant,
          )
      )
      resnets.append(
          FlaxVideoResBlockUNet(
              in_channels=self.in_channels,
              out_channels=self.in_channels,
              dropout=self.dropout,
              norm_num_groups=self.norm_num_groups,
              norm_eps=self.resnet_eps,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
              precision=self.precision,
              time_embedding_norm=self.time_embedding_norm,
          )
      )
    self.resnets = resnets
    self.attentions = attentions

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      temb: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
      cross_attention_kwargs=None,
  ):
    hidden_states = self.resnets[0](
        hidden_states, temb, num_frames=num_frames,
        deterministic=deterministic, image_only_indicator=image_only_indicator,
    )
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      hidden_states = attn(
          hidden_states, encoder_hidden_states, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
          cross_attention_kwargs=cross_attention_kwargs,
      )
      hidden_states = resnet(
          hidden_states, temb, num_frames=num_frames,
          deterministic=deterministic, image_only_indicator=image_only_indicator,
      )
    return hidden_states
