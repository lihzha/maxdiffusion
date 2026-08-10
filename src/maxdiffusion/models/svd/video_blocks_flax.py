# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Primitive video blocks for the SVD VideoUNet.

Three new pieces layered on top of the existing SD UNet primitives:

  * :class:`FlaxTemporalTransformerBlock` — transformer block that operates
    along the time axis; matches sgm's ``VideoTransformerBlock`` with
    ``ff_in=True``, ``disable_self_attn=False``,
    ``disable_temporal_crossattention=False``.
  * :class:`FlaxVideoResBlockUNet` — wraps a spatial ``FlaxResnetBlock2D``
    (time-embedding consumer) with a temporal ``FlaxTemporalResBlock3D`` and
    a ``learned_with_images`` :class:`FlaxAlphaBlender`.
  * :class:`FlaxTimePosEmbed` — small MLP (``in -> 4*in -> in``) applied to
    sinusoidal per-frame indices. Matches sgm's
    ``SpatialVideoTransformer.time_pos_embed``.

All shapes are channels-last. Temporal primitives expect ``(B, T, H, W, C)``;
block-boundary tensors remain ``(B*T, H, W, C)``.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from ..attention_flax import FlaxAttention, FlaxFeedForward
from ..resnet_flax import FlaxResnetBlock2D
from ...common_types import BlockSizes
from ..embeddings_flax import get_sinusoidal_embeddings
from .. import quantizations
from .video_decoder_flax import FlaxAlphaBlender, FlaxTemporalResBlock3D

Quant = quantizations.AqtQuantization


class FlaxTimePosEmbed(nn.Module):
  """Per-frame positional embedding applied inside SpatialVideoTransformer.

  Computes ``mlp(sinusoidal(frame_index))`` where the MLP is
  ``Linear(in -> 4*in) -> SiLU -> Linear(4*in -> in)`` — matches sgm's
  ``SpatialVideoTransformer.time_pos_embed``.
  """

  channels: int
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.linear_1 = nn.Dense(
        self.channels * 4, dtype=self.dtype, param_dtype=self.weights_dtype
    )
    self.linear_2 = nn.Dense(
        self.channels, dtype=self.dtype, param_dtype=self.weights_dtype
    )

  def __call__(self, frame_index: jnp.ndarray) -> jnp.ndarray:
    # Matches Diffusers' TransformerSpatioTemporalModel.time_proj =
    # Timesteps(in_channels, flip_sin_to_cos=True, downscale_freq_shift=0).
    t_emb = get_sinusoidal_embeddings(
        frame_index, embedding_dim=self.channels, flip_sin_to_cos=True, freq_shift=0
    )
    t_emb = self.linear_1(t_emb)
    t_emb = nn.silu(t_emb)
    t_emb = self.linear_2(t_emb)
    return t_emb


class FlaxTemporalTransformerBlock(nn.Module):
  """Transformer block that operates along the time axis.

  Matches sgm's ``VideoTransformerBlock(ff_in=True, disable_self_attn=False,
  switch_temporal_ca_to_sa=False, disable_temporal_crossattention=False)``:
  optional ``ff_in`` (``extra_ff_mix_layer=True`` in base SVD), then
  self-attn → cross-attn → FF, each with its own LayerNorm and residual.

  Input is ``(B', T, C)`` where ``B' = B * H * W``. Context ``(B', S, C_ctx)``
  is the per-spatial-position CLIP feature (already tiled outside).
  """

  dim: int
  n_heads: int
  d_head: int
  dropout: float = 0.0
  context_dim: Optional[int] = None
  use_ff_in: bool = True
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  attention_kernel: str = "dot_product"
  # When True, routes FlaxAttention through the chunked-query path
  # (jax_memory_efficient_attention) — avoids materializing the full N×N
  # scores. Irrelevant for the temporal axis (T=14) but plumbed for
  # symmetry with the spatial branch.
  use_memory_efficient_attention: bool = False
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  quant: Quant = None

  def setup(self):
    if self.use_ff_in:
      self.norm_in = nn.LayerNorm(
          epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype
      )
      # Named `ff_in` to match Diffusers' TemporalBasicTransformerBlock.
      self.ff_in = FlaxFeedForward(
          dim=self.dim,
          dropout=self.dropout,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
      )

    # self-attention
    self.attn1 = FlaxAttention(
        query_dim=self.dim,
        heads=self.n_heads,
        dim_head=self.d_head,
        dropout=self.dropout,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        quant=self.quant,
    )
    # cross-attention with the per-spatial-position CLIP context
    self.attn2 = FlaxAttention(
        query_dim=self.dim,
        heads=self.n_heads,
        dim_head=self.d_head,
        dropout=self.dropout,
        use_memory_efficient_attention=self.use_memory_efficient_attention,
        attention_kernel=self.attention_kernel,
        flash_min_seq_length=self.flash_min_seq_length,
        flash_block_sizes=self.flash_block_sizes,
        mesh=self.mesh,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        quant=self.quant,
    )
    self.ff = FlaxFeedForward(
        dim=self.dim,
        dropout=self.dropout,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )
    self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)
    self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      context: Optional[jnp.ndarray] = None,
      deterministic: bool = True,
  ) -> jnp.ndarray:
    if self.use_ff_in:
      residual = hidden_states
      hidden_states = self.norm_in(hidden_states)
      hidden_states = self.ff_in(hidden_states, deterministic=deterministic)
      hidden_states = hidden_states + residual

    # self-attn
    residual = hidden_states
    hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
    hidden_states = hidden_states + residual

    # cross-attn (spatial context shared across frames)
    residual = hidden_states
    hidden_states = self.attn2(self.norm2(hidden_states), context=context, deterministic=deterministic)
    hidden_states = hidden_states + residual

    # ff
    residual = hidden_states
    hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
    hidden_states = hidden_states + residual

    return hidden_states


class FlaxVideoResBlockUNet(nn.Module):
  """UNet-side ResBlock with spatial + temporal + learned alpha blend.

  * Spatial branch: :class:`FlaxResnetBlock2D` (consumes ``temb``).
  * Temporal branch: :class:`FlaxTemporalResBlock3D` (no temb, ``kernel=(3,1,1)``).
  * Merge: :class:`FlaxAlphaBlender` with ``merge_strategy='learned_with_images'``
    (temporal branch is disabled when ``image_only_indicator`` is set for that
    sample/frame — matches base SVD's pure-video default).

  Input/output shape: ``(B*T, H, W, C)``.
  """

  in_channels: int
  out_channels: Optional[int] = None
  temb_channels: int = 1280  # time_embed_dim for the UNet (block_out_channels[0] * 4)
  dropout: float = 0.0
  norm_num_groups: int = 32
  # Diffusers' SpatioTemporalResBlock passes a single ``eps`` to both the
  # spatial and temporal sub-blocks (via ``temporal_eps=None`` default). In
  # the SVD UNet, cross-attn down/up blocks use eps=1e-6 and no-cross-attn
  # down/up + mid block use eps=1e-5. The outer block passes the right value.
  norm_eps: float = 1e-5
  video_kernel_size: tuple = (3, 1, 1)
  merge_strategy: str = "learned_with_images"
  alpha_init: float = 0.0
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  # "default" | "scale_shift" (AdaGN), applied to BOTH branches so the action
  # signal modulates the temporal path too. Set only by FlaxVideoUNet when
  # ``action_cond_mode == 'adaln'``.
  time_embedding_norm: str = "default"

  def setup(self):
    out_c = self.in_channels if self.out_channels is None else self.out_channels

    self.spatial_res_block = FlaxResnetBlock2D(
        in_channels=self.in_channels,
        out_channels=out_c,
        dropout_prob=self.dropout,
        norm_num_groups=self.norm_num_groups,
        norm_eps=self.norm_eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
        time_embedding_norm=self.time_embedding_norm,
    )
    # Named `temporal_res_block` to match Diffusers' SpatioTemporalResBlock.
    # temb_channels is set so the temporal block has a `time_emb_proj` Dense
    # (matches Diffusers' UNet-side ``TemporalResnetBlock(temb_channels=...)``).
    self.temporal_res_block = FlaxTemporalResBlock3D(
        in_channels=out_c,
        out_channels=out_c,
        groups=self.norm_num_groups,
        kernel_size=self.video_kernel_size,
        dropout=self.dropout,
        temb_channels=self.temb_channels,
        eps=self.norm_eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        time_embedding_norm=self.time_embedding_norm,
    )
    self.time_mixer = FlaxAlphaBlender(
        merge_strategy=self.merge_strategy,
        alpha_init=self.alpha_init,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      temb: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    # Spatial ResBlock consumes the time embedding.
    x = self.spatial_res_block(hidden_states, temb, deterministic=deterministic)
    x_spatial = rearrange(x, "(b t) h w c -> b t h w c", t=num_frames)
    # Reshape temb (B*T, C_temb) -> (B, T, C_temb) for the temporal branch.
    batch = x_spatial.shape[0]
    temb_bt = temb.reshape(batch, num_frames, -1)
    x_temporal = self.temporal_res_block(x_spatial, temb=temb_bt, deterministic=deterministic)
    merged = self.time_mixer(
        x_spatial=x_spatial,
        x_temporal=x_temporal,
        image_only_indicator=image_only_indicator,
    )
    return rearrange(merged, "b t h w c -> (b t) h w c")
