# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SpatialVideoTransformer — SVD's stacked spatial + temporal transformer.

Matches sgm's ``SpatialVideoTransformer``. Each depth level runs:

  1. spatial transformer block (self-attn over H*W + cross-attn with CLIP),
  2. add per-frame learned positional embedding ``time_pos_embed``,
  3. temporal transformer block (self-attn over T + cross-attn with the
     CLIP features spread across spatial positions),
  4. learned sigmoid blend between spatial and temporal tensors (AlphaBlender
     with ``merge_strategy='learned_with_images'``).

Base SVD config (from ``svd.yaml``): ``depth=1``, ``use_linear_projection=True``,
``use_spatial_context=True``, ``extra_ff_mix_layer=True``, ``merge_strategy=
learned_with_images``, ``context_dim=1024``.

All tensors are channels-last. Input ``(B*T, H, W, C)``; output same shape.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..attention_flax import FlaxBasicTransformerBlock
from ...common_types import BlockSizes
from .. import quantizations
from .video_blocks_flax import FlaxTemporalTransformerBlock, FlaxTimePosEmbed
from .video_decoder_flax import FlaxAlphaBlender

Quant = quantizations.AqtQuantization


class FlaxSpatialVideoTransformer(nn.Module):
  """Stacked spatial + temporal transformer with AlphaBlender merge.

  Drop-in replacement for ``FlaxTransformer2DModel`` inside the UNet. The
  preamble (GroupNorm + proj_in) and postamble (proj_out + residual) are
  identical to the spatial-only version; only the inner block loop changes.
  """

  in_channels: int
  n_heads: int
  d_head: int
  depth: int = 1
  dropout: float = 0.0
  use_linear_projection: bool = True
  context_dim: int = 1024
  use_ff_in: bool = True
  norm_num_groups: int = 32
  merge_strategy: str = "learned_with_images"
  alpha_init: float = 0.0
  attention_kernel: str = "dot_product"
  temporal_attention_kernel: str = "dot_product"
  # Routes FlaxAttention through the chunked-query path that avoids
  # materializing the full N×N scores matrix. Main win is at the shallowest
  # spatial stage (H*W≥4096 at 512×512, 14 frames, 5 heads). Enables 512×512
  # on a 24 GB GPU. See docs/svd_debug_notes.md §6.3.
  use_memory_efficient_attention: bool = False
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  quant: Quant = None

  def setup(self):
    inner_dim = self.n_heads * self.d_head

    # Diffusers' TransformerSpatioTemporalModel uses eps=1e-6 for the preamble
    # GroupNorm (see diffusers/models/transformers/transformer_temporal.py).
    self.norm = nn.GroupNorm(
        num_groups=self.norm_num_groups, epsilon=1e-6, dtype=self.dtype, param_dtype=self.weights_dtype
    )

    if self.use_linear_projection:
      self.proj_in = nn.Dense(inner_dim, dtype=self.dtype, param_dtype=self.weights_dtype)
      self.proj_out = nn.Dense(inner_dim, dtype=self.dtype, param_dtype=self.weights_dtype)
    else:
      self.proj_in = nn.Conv(
          inner_dim,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
      )
      self.proj_out = nn.Conv(
          inner_dim,
          kernel_size=(1, 1),
          strides=(1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
      )

    # Per-frame positional embedding (shared across depth levels in sgm).
    self.time_pos_embed = FlaxTimePosEmbed(
        channels=self.in_channels, dtype=self.dtype, weights_dtype=self.weights_dtype
    )

    # Paired spatial + temporal blocks, AlphaBlenders one per depth.
    self.transformer_blocks = [
        FlaxBasicTransformerBlock(
            dim=inner_dim,
            n_heads=self.n_heads,
            d_head=self.d_head,
            dropout=self.dropout,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            attention_kernel=self.attention_kernel,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mesh=self.mesh,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,
            quant=self.quant,
        )
        for _ in range(self.depth)
    ]

    # Named `temporal_transformer_blocks` to match Diffusers'
    # TransformerSpatioTemporalModel.
    self.temporal_transformer_blocks = [
        FlaxTemporalTransformerBlock(
            dim=inner_dim,
            n_heads=self.n_heads,
            d_head=self.d_head,
            dropout=self.dropout,
            context_dim=self.context_dim,
            use_ff_in=self.use_ff_in,
            attention_kernel=self.temporal_attention_kernel,
            use_memory_efficient_attention=self.use_memory_efficient_attention,
            flash_min_seq_length=self.flash_min_seq_length,
            flash_block_sizes=self.flash_block_sizes,
            mesh=self.mesh,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            quant=self.quant,
        )
        for _ in range(self.depth)
    ]

    self.time_mixer = FlaxAlphaBlender(
        merge_strategy=self.merge_strategy,
        alpha_init=self.alpha_init,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )

    self.dropout_layer = nn.Dropout(rate=self.dropout)

  def _build_time_context(
      self, context: jnp.ndarray, num_frames: int, h: int, w: int
  ) -> jnp.ndarray:
    """From (B*T, S, C_ctx) pick frame-0 per batch then repeat over H*W.

    Result shape ``(B*H*W, 1 or S, C_ctx)``. Matches sgm's
    ``time_context_first_timestep = context[::timesteps]; repeat(..., n=h*w)``.

    The frame-0 slice is inherited from sgm/diffusers, where the context is a
    single pooled CLIP image embedding identical across frames, so slicing was a
    no-op. It stays faithful here, but note what it means once the context is
    per-frame (Ctrl-World's action tokens, or a skeleton grid): the temporal
    blocks see only latent frame 0's conditioning, whatever the other frames
    carry.

    Multi-token contexts are mean-pooled to a single key first. That matters for
    ``action_cond_mode='skeleton_cross_attn'``, whose context is a 180-token
    spatial grid: without pooling, every temporal block would attend
    ``(B*H*W, T)`` queries over 180 keys drawn from ONE frame's grid — at the
    shallowest stage that is millions of extra pairs per sample to read a
    spatially-scrambled snapshot of frame 0, which is cost without information.
    Pooling keeps the temporal branch at exactly the single key it has always
    had, so the skeleton grid is spent where it is aligned with the queries: the
    spatial blocks.
    """
    bt, s, c_ctx = context.shape
    batch = bt // num_frames
    context = context.reshape(batch, num_frames, s, c_ctx)
    context_first = context[:, 0]  # (B, S, C_ctx)
    if s > 1:
      context_first = jnp.mean(context_first, axis=1, keepdims=True)  # (B, 1, C_ctx)
    # Repeat each row H*W times: [b0,b0,..,b0, b1,b1,..,b1, ...]
    return jnp.repeat(context_first, h * w, axis=0)

  def _frame_indices(self, batch: int, num_frames: int) -> jnp.ndarray:
    """(B*T,) tensor of frame indices tiled across batch.

    For B=2, T=3 → ``[0, 1, 2, 0, 1, 2]``.
    """
    arange = jnp.arange(num_frames, dtype=jnp.float32)
    return jnp.tile(arange, batch)

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      context: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
      cross_attention_kwargs=None,
  ) -> jnp.ndarray:
    bt, h, w, channels = hidden_states.shape
    batch = bt // num_frames
    residual = hidden_states

    hidden_states = self.norm(hidden_states)
    if self.use_linear_projection:
      hidden_states = hidden_states.reshape(bt, h * w, channels)
      hidden_states = self.proj_in(hidden_states)
    else:
      hidden_states = self.proj_in(hidden_states)
      hidden_states = hidden_states.reshape(bt, h * w, channels)

    # spatial_context: (B*T, S, C_ctx) — one CLIP token per frame.
    spatial_context = context
    time_context = self._build_time_context(context, num_frames, h, w)

    # (B*T,) frame indices → (B*T, C) learned pos embed
    frame_idx = self._frame_indices(batch, num_frames)
    frame_emb = self.time_pos_embed(frame_idx)[:, None, :]  # (B*T, 1, C)

    for spatial_block, temporal_block in zip(
        self.transformer_blocks, self.temporal_transformer_blocks
    ):
      # spatial: (B*T, H*W, C)
      hidden_states = spatial_block(
          hidden_states, spatial_context, deterministic=deterministic,
          cross_attention_kwargs=cross_attention_kwargs,
      )

      # temporal branch: add per-frame pos embed, reshape, mix, reshape back.
      x_mix = hidden_states + frame_emb
      x_mix = x_mix.reshape(batch, num_frames, h * w, channels)
      # (B, T, H*W, C) -> (B, H*W, T, C) -> (B*H*W, T, C)
      x_mix = jnp.transpose(x_mix, (0, 2, 1, 3)).reshape(batch * h * w, num_frames, channels)
      x_mix = temporal_block(x_mix, context=time_context, deterministic=deterministic)
      # back to (B*T, H*W, C)
      x_mix = x_mix.reshape(batch, h * w, num_frames, channels)
      x_mix = jnp.transpose(x_mix, (0, 2, 1, 3)).reshape(bt, h * w, channels)

      hidden_states = self.time_mixer(
          x_spatial=hidden_states,
          x_temporal=x_mix,
          image_only_indicator=image_only_indicator,
      )

    if self.use_linear_projection:
      hidden_states = self.proj_out(hidden_states)
      hidden_states = hidden_states.reshape(bt, h, w, channels)
    else:
      hidden_states = hidden_states.reshape(bt, h, w, channels)
      hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states + residual
    return self.dropout_layer(hidden_states, deterministic=deterministic)
