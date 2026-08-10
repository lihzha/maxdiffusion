# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Temporal VAE decoder for Stable Video Diffusion (base).

All tensors are channels-last: ``(B*T, H, W, C)`` at block boundaries and
``(B, T, H, W, C)`` inside temporal layers. The spatial primitives from
``models/vae_flax.py`` (``FlaxResnetBlock2D``, ``FlaxAttentionBlock``) are
reused as-is; only temporal 3D convs and learned spatial/temporal blending
are added here.

Base SVD config (svd.yaml)::

    video_kernel_size: [3, 1, 1]   # temporal receptive field = 3 frames
    merge_strategy: learned        # sigmoid(mix_factor), no image gating
    time_mode: conv-only           # mid-block attention stays purely spatial

References (sgm, Stability-AI/generative-models):
  sgm/modules/autoencoding/temporal_ae.py  AE3DConv, VideoResBlock, VideoBlock
  sgm/modules/diffusionmodules/util.py     AlphaBlender
"""

from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange

from ..vae_flax import (
    FlaxAttentionBlock,
    FlaxResnetBlock2D,
    FlaxUpsample2D,
)


# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------


class FlaxAlphaBlender(nn.Module):
  """Learned sigmoid blend between a spatial and a temporal tensor.

  Matches Diffusers' ``AlphaBlender`` (``diffusers/models/attention.py``). The
  stored parameter ``mix_factor`` is a size-1 tensor; ``alpha =
  sigmoid(mix_factor)``.

  ``merge_strategy``:
    - ``"learned"``: no image gating. Used by the VAE decoder.
    - ``"learned_with_images"``: alpha is forced to 1 per-frame where
      ``image_only_indicator`` is set (disables the temporal branch). Used by
      the UNet.

  ``switch_spatial_to_temporal_mix``: flips ``alpha → 1 - alpha`` before the
  blend. Enabled in the VAE; disabled in the UNet.

  Output is ``alpha * x_spatial + (1 - alpha) * x_temporal`` (after the
  optional flip).
  """

  merge_strategy: str = "learned"
  alpha_init: float = 0.0
  switch_spatial_to_temporal_mix: bool = False
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    if self.merge_strategy not in ("learned", "learned_with_images", "fixed"):
      raise ValueError(f"AlphaBlender: unknown merge_strategy={self.merge_strategy!r}")
    self.mix_factor = self.param(
        "mix_factor", nn.initializers.constant(self.alpha_init), (1,), self.weights_dtype
    )

  def get_alpha(
      self, image_only_indicator: Optional[jnp.ndarray], ndims: int
  ) -> jnp.ndarray:
    if self.merge_strategy == "fixed":
      return self.mix_factor.astype(self.dtype)

    alpha = nn.sigmoid(self.mix_factor).astype(self.dtype)  # (1,)
    if self.merge_strategy == "learned":
      return alpha

    # learned_with_images: per-(B, T) gate.
    if image_only_indicator is None:
      return alpha
    gate = (image_only_indicator > 0.5).astype(self.dtype)  # (B, T)
    alpha_bt = jnp.where(gate > 0, jnp.ones_like(gate), alpha)  # (B, T)
    # Reshape for downstream broadcasting. maxdiffusion is channels-last, so:
    #   5D (B, T, H, W, C) → (B, T, 1, 1, 1)
    #   3D (B*T, H*W, C)   → (B*T, 1, 1)
    if ndims == 5:
      return alpha_bt[:, :, None, None, None]
    if ndims == 3:
      return alpha_bt.reshape(-1)[:, None, None]
    raise ValueError(f"AlphaBlender: unsupported ndims={ndims} for learned_with_images")

  def __call__(
      self,
      x_spatial: jnp.ndarray,
      x_temporal: jnp.ndarray,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
    if self.switch_spatial_to_temporal_mix:
      alpha = 1.0 - alpha
    return alpha * x_spatial + (1.0 - alpha) * x_temporal


class FlaxConv3DTemporal(nn.Module):
  """3D conv with kernel ``(3, 1, 1)`` along (T, H, W); purely temporal.

  Input ``(B, T, H, W, C)`` → output ``(B, T, H, W, out_channels)``.
  """

  out_channels: int
  kernel_size: Tuple[int, int, int] = (3, 1, 1)
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    pads = tuple((k // 2, k // 2) for k in self.kernel_size)
    self.conv = nn.Conv(
        self.out_channels,
        kernel_size=self.kernel_size,
        strides=(1, 1, 1),
        padding=pads,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )

  def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
    return self.conv(hidden_states)


class FlaxAE3DConv(nn.Module):
  """Spatial 2D conv followed by a temporal 3D conv with kernel ``(3,1,1)``.

  Replaces the SD-VAE ``conv_in`` / ``conv_out`` in the decoder. Shape flow:
  ``(B*T, H, W, in_c) -> (B*T, H, W, out_c) -> (B, T, H, W, out_c) ->
  (B, T, H, W, out_c) -> (B*T, H, W, out_c)``.
  """

  out_channels: int
  kernel_size: Tuple[int, int] = (3, 3)
  padding: Tuple[Tuple[int, int], Tuple[int, int]] = ((1, 1), (1, 1))
  video_kernel_size: Tuple[int, int, int] = (3, 1, 1)
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.conv = nn.Conv(
        self.out_channels,
        kernel_size=self.kernel_size,
        strides=(1, 1),
        padding=self.padding,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    self.time_mix_conv = FlaxConv3DTemporal(
        self.out_channels,
        kernel_size=self.video_kernel_size,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )

  def __call__(self, hidden_states: jnp.ndarray, num_frames: int) -> jnp.ndarray:
    x = self.conv(hidden_states)
    x = rearrange(x, "(b t) h w c -> b t h w c", t=num_frames)
    x = self.time_mix_conv(x)
    return rearrange(x, "b t h w c -> (b t) h w c")


class FlaxTemporalResBlock3D(nn.Module):
  """Temporal counterpart of a spatial ResBlock, using 3D convs with kernel
  ``(3, 1, 1)``.

  Matches Diffusers' ``TemporalResnetBlock`` (``diffusers.models.resnet``):
  ``norm1 → silu → conv1 → [+ time_emb_proj(temb)] → norm2 → silu → dropout →
  conv2`` with a 1x1x1 ``conv_shortcut`` if channels change.

  If ``temb_channels`` is set (UNet case), ``time_emb_proj`` projects the time
  embedding and broadcast-adds it across spatial/temporal dims. When
  ``temb_channels is None`` (VAE case), no ``time_emb_proj`` is created.

  Input/output shape: ``(B, T, H, W, C)``. ``temb`` is reshaped externally to
  ``(B, T, C_temb)`` before being passed in.
  """

  in_channels: int
  out_channels: Optional[int] = None
  groups: int = 32
  kernel_size: Tuple[int, int, int] = (3, 1, 1)
  dropout: float = 0.0
  temb_channels: Optional[int] = None
  # Diffusers' ``TemporalResnetBlock`` takes eps from the outer SpatioTemporal
  # block. VAE uses 1e-5 (passed via ``temporal_eps=1e-5``); the UNet side's
  # cross-attn blocks use 1e-6 and the no-cross-attn/mid blocks use 1e-5.
  eps: float = 1e-6
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  # "default" | "scale_shift" (AdaGN). See FlaxResnetBlock2D.time_embedding_norm
  # — same semantics, and likewise only ever set by FlaxVideoUNet when
  # ``action_cond_mode == 'adaln'``. Ignored when ``temb_channels is None``
  # (the VAE case, which has no time embedding at all).
  time_embedding_norm: str = "default"

  def setup(self):
    if self.time_embedding_norm not in ("default", "scale_shift"):
      raise ValueError(
          "FlaxTemporalResBlock3D.time_embedding_norm must be 'default' or "
          f"'scale_shift', got {self.time_embedding_norm!r}"
      )
    out_c = self.in_channels if self.out_channels is None else self.out_channels
    pads = tuple((k // 2, k // 2) for k in self.kernel_size)

    self.norm1 = nn.GroupNorm(
        num_groups=self.groups, epsilon=self.eps, dtype=self.dtype, param_dtype=self.weights_dtype
    )
    self.conv1 = nn.Conv(
        out_c,
        kernel_size=self.kernel_size,
        strides=(1, 1, 1),
        padding=pads,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    if self.temb_channels is not None:
      self.time_emb_proj = nn.Dense(
          out_c, dtype=self.dtype, param_dtype=self.weights_dtype
      )
      # AdaGN scale; separate zero-init Dense so the pretrained time_emb_proj
      # still loads and serves as the shift. See FlaxResnetBlock2D.setup.
      if self.time_embedding_norm == "scale_shift":
        self.adagn_scale_proj = nn.Dense(
            out_c,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )
      else:
        self.adagn_scale_proj = None
    else:
      self.time_emb_proj = None
      self.adagn_scale_proj = None
    self.norm2 = nn.GroupNorm(
        num_groups=self.groups, epsilon=self.eps, dtype=self.dtype, param_dtype=self.weights_dtype
    )
    self.dropout_layer = nn.Dropout(self.dropout)
    self.conv2 = nn.Conv(
        out_c,
        kernel_size=self.kernel_size,
        strides=(1, 1, 1),
        padding=pads,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    if self.in_channels != out_c:
      self.conv_shortcut = nn.Conv(
          out_c,
          kernel_size=(1, 1, 1),
          strides=(1, 1, 1),
          padding="VALID",
          dtype=self.dtype,
          param_dtype=self.weights_dtype,
      )
    else:
      self.conv_shortcut = None

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      temb: Optional[jnp.ndarray] = None,
      deterministic: bool = True,
  ) -> jnp.ndarray:
    residual = hidden_states
    x = self.norm1(hidden_states)
    x = nn.swish(x)
    x = self.conv1(x)

    if self.time_emb_proj is not None and temb is not None:
      # temb: (B, T, C_temb) → project → (B, T, C_out) → (B, T, 1, 1, C_out)
      t = nn.swish(temb)
      shift = self.time_emb_proj(t)[:, :, None, None, :]  # broadcast over H, W
      if self.adagn_scale_proj is not None:
        # AdaGN: modulate after norm2 instead of adding before it.
        scale = self.adagn_scale_proj(t)[:, :, None, None, :]
        x = self.norm2(x) * (1 + scale) + shift
      else:
        x = self.norm2(x + shift)
    else:
      x = self.norm2(x)

    x = nn.swish(x)
    x = self.dropout_layer(x, deterministic)
    x = self.conv2(x)
    if self.conv_shortcut is not None:
      residual = self.conv_shortcut(residual)
    return x + residual


class FlaxVideoResnetBlock(nn.Module):
  """Spatial ``FlaxResnetBlock2D`` composed with a temporal ``FlaxTemporalResBlock3D``.

  The two outputs are merged by a learned sigmoid ``AlphaBlender``. Used inside
  the mid block and every up-decoder block of the VideoDecoder.

  Input shape: ``(B*T, H, W, in_channels)``.
  Output shape: ``(B*T, H, W, out_channels)``.
  """

  in_channels: int
  out_channels: Optional[int] = None
  groups: int = 32
  video_kernel_size: Tuple[int, int, int] = (3, 1, 1)
  merge_strategy: str = "learned"
  alpha_init: float = 0.0
  switch_spatial_to_temporal_mix: bool = False
  dropout: float = 0.0
  # Diffusers' VAE-side ``SpatioTemporalResBlock`` is constructed with
  # ``eps=1e-6, temporal_eps=1e-5`` — spatial uses 1e-6, temporal uses 1e-5.
  temporal_eps: float = 1e-5
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    out_c = self.in_channels if self.out_channels is None else self.out_channels
    # Spatial branch — identical to SD VAE (hardcodes eps=1e-6, matching
    # Diffusers' SpatioTemporalResBlock(eps=1e-6) on the VAE side).
    self.spatial_res_block = FlaxResnetBlock2D(
        in_channels=self.in_channels,
        out_channels=out_c,
        dropout=self.dropout,
        groups=self.groups,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )
    # Temporal branch — 3D convs with kernel (3,1,1).
    # Named `temporal_res_block` to match Diffusers' SpatioTemporalResBlock.
    # VAE side: temb_channels=None → no time_emb_proj.
    self.temporal_res_block = FlaxTemporalResBlock3D(
        in_channels=out_c,
        out_channels=out_c,
        groups=self.groups,
        kernel_size=self.video_kernel_size,
        dropout=self.dropout,
        temb_channels=None,
        eps=self.temporal_eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )
    self.time_mixer = FlaxAlphaBlender(
        merge_strategy=self.merge_strategy,
        alpha_init=self.alpha_init,
        switch_spatial_to_temporal_mix=self.switch_spatial_to_temporal_mix,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    x = self.spatial_res_block(hidden_states, deterministic=deterministic)
    x_spatial = rearrange(x, "(b t) h w c -> b t h w c", t=num_frames)
    x_temporal = self.temporal_res_block(x_spatial, deterministic=deterministic)
    merged = self.time_mixer(
        x_spatial=x_spatial,
        x_temporal=x_temporal,
        image_only_indicator=image_only_indicator,
    )
    return rearrange(merged, "b t h w c -> (b t) h w c")


# -----------------------------------------------------------------------------
# Blocks
# -----------------------------------------------------------------------------


class FlaxUpBlockTemporalDecoder(nn.Module):
  """Matches Diffusers' ``UpBlockTemporalDecoder``: video resblocks + optional
  spatial upsample. Attribute names are ``resnets`` (list) and ``upsamplers``
  (list, only present when ``add_upsample=True``).
  """

  in_channels: int
  out_channels: int
  dropout: float = 0.0
  num_layers: int = 1
  resnet_groups: int = 32
  add_upsample: bool = True
  video_kernel_size: Tuple[int, int, int] = (3, 1, 1)
  merge_strategy: str = "learned"
  switch_spatial_to_temporal_mix: bool = True
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    resnets = []
    for i in range(self.num_layers):
      in_c = self.in_channels if i == 0 else self.out_channels
      resnets.append(
          FlaxVideoResnetBlock(
              in_channels=in_c,
              out_channels=self.out_channels,
              groups=self.resnet_groups,
              dropout=self.dropout,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              switch_spatial_to_temporal_mix=self.switch_spatial_to_temporal_mix,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
          )
      )
    self.resnets = resnets
    if self.add_upsample:
      # Diffusers uses `upsamplers = ModuleList([Upsample2D(...)])`, which maps
      # to the Flax attribute name `upsamplers_0` after the PT→Flax renamer.
      self.upsamplers_0 = FlaxUpsample2D(
          self.out_channels, dtype=self.dtype, weights_dtype=self.weights_dtype
      )

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    for resnet in self.resnets:
      hidden_states = resnet(
          hidden_states,
          num_frames=num_frames,
          deterministic=deterministic,
          image_only_indicator=image_only_indicator,
      )
    if self.add_upsample:
      hidden_states = self.upsamplers_0(hidden_states)
    return hidden_states


class FlaxMidBlockTemporalDecoder(nn.Module):
  """Matches Diffusers' ``MidBlockTemporalDecoder``.

  Structure: ``resnets[0]`` → (``attentions[0]`` → ``resnets[1:]``). There is
  exactly one 2D attention (no temporal attention, per
  ``time_mode='conv-only'`` in base SVD).
  """

  in_channels: int
  out_channels: Optional[int] = None
  dropout: float = 0.0
  num_layers: int = 1
  resnet_groups: int = 32
  num_attention_heads: int = 1
  video_kernel_size: Tuple[int, int, int] = (3, 1, 1)
  merge_strategy: str = "learned"
  switch_spatial_to_temporal_mix: bool = True
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    out_c = self.in_channels if self.out_channels is None else self.out_channels

    resnets = []
    for i in range(self.num_layers):
      in_c = self.in_channels if i == 0 else out_c
      resnets.append(
          FlaxVideoResnetBlock(
              in_channels=in_c,
              out_channels=out_c,
              groups=self.resnet_groups,
              dropout=self.dropout,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              switch_spatial_to_temporal_mix=self.switch_spatial_to_temporal_mix,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
          )
      )
    attentions = [
        FlaxAttentionBlock(
            channels=self.in_channels,
            num_head_channels=self.num_attention_heads,
            num_groups=self.resnet_groups,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
        )
    ]
    self.resnets = resnets
    self.attentions = attentions

  def __call__(
      self,
      hidden_states: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    hidden_states = self.resnets[0](
        hidden_states,
        num_frames=num_frames,
        deterministic=deterministic,
        image_only_indicator=image_only_indicator,
    )
    for resnet, attn in zip(self.resnets[1:], self.attentions):
      hidden_states = attn(hidden_states)
      hidden_states = resnet(
          hidden_states,
          num_frames=num_frames,
          deterministic=deterministic,
          image_only_indicator=image_only_indicator,
      )
    return hidden_states


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------


class FlaxVideoDecoder(nn.Module):
  """VideoDecoder for SVD.

  Matches Diffusers' ``TemporalDecoder``:

    - ``conv_in``: plain 2D conv (no temporal mix).
    - ``mid_block``: :class:`FlaxMidBlockTemporalDecoder` (resnets + 1 spatial
      attention).
    - ``up_blocks``: list of :class:`FlaxUpBlockTemporalDecoder`.
    - ``conv_norm_out``, ``conv_out``: plain 2D conv.
    - ``time_conv_out``: a final temporal 3D conv applied to the decoded frames.

  VAE resblocks use ``merge_strategy='learned'`` and
  ``switch_spatial_to_temporal_mix=True`` (matches Diffusers VAE). Temporal
  attention is disabled (base SVD ``time_mode='conv-only'``).

  Input latent shape: ``(B*T, H/8, W/8, latent_channels)``.
  Output pixels shape: ``(B*T, H, W, out_channels)``.
  """

  in_channels: int = 4
  out_channels: int = 3
  up_block_types: Tuple[str, ...] = (
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
  )
  block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
  layers_per_block: int = 2
  norm_num_groups: int = 32
  act_fn: str = "silu"
  video_kernel_size: Tuple[int, int, int] = (3, 1, 1)
  merge_strategy: str = "learned"
  switch_spatial_to_temporal_mix: bool = True
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32

  def setup(self):
    block_out_channels = self.block_out_channels

    # conv_in: plain 2D Conv (no temporal mix — matches Diffusers TemporalDecoder).
    self.conv_in = nn.Conv(
        block_out_channels[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )

    self.mid_block = FlaxMidBlockTemporalDecoder(
        in_channels=block_out_channels[-1],
        out_channels=block_out_channels[-1],
        num_layers=self.layers_per_block,
        resnet_groups=self.norm_num_groups,
        # Diffusers uses attention_head_dim = block_out_channels[-1] (full-dim
        # single head). FlaxAttentionBlock sets num_heads = channels //
        # num_head_channels.
        num_attention_heads=block_out_channels[-1],
        video_kernel_size=self.video_kernel_size,
        merge_strategy=self.merge_strategy,
        switch_spatial_to_temporal_mix=self.switch_spatial_to_temporal_mix,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
    )

    reversed_block_out_channels = list(reversed(block_out_channels))
    output_channel = reversed_block_out_channels[0]
    up_blocks = []
    for i, _ in enumerate(self.up_block_types):
      prev_output_channel = output_channel
      output_channel = reversed_block_out_channels[i]
      is_final_block = i == len(block_out_channels) - 1

      up_blocks.append(
          FlaxUpBlockTemporalDecoder(
              in_channels=prev_output_channel,
              out_channels=output_channel,
              num_layers=self.layers_per_block + 1,
              resnet_groups=self.norm_num_groups,
              add_upsample=not is_final_block,
              video_kernel_size=self.video_kernel_size,
              merge_strategy=self.merge_strategy,
              switch_spatial_to_temporal_mix=self.switch_spatial_to_temporal_mix,
              dtype=self.dtype,
              weights_dtype=self.weights_dtype,
          )
      )
    self.up_blocks = up_blocks

    self.conv_norm_out = nn.GroupNorm(
        num_groups=self.norm_num_groups,
        epsilon=1e-6,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    # conv_out: plain 2D conv.
    self.conv_out = nn.Conv(
        self.out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    # time_conv_out: final temporal 3D conv applied to the decoded frames.
    # Diffusers' TemporalDecoder stores this as a plain nn.Conv3d at
    # `time_conv_out.{weight,bias}`, not wrapped under a `.conv` sub-attribute.
    # Use nn.Conv directly so the PT→Flax key path matches without a custom
    # key translator.
    _pads = tuple((k // 2, k // 2) for k in self.video_kernel_size)
    self.time_conv_out = nn.Conv(
        self.out_channels,
        kernel_size=self.video_kernel_size,
        strides=(1, 1, 1),
        padding=_pads,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )

  def __call__(
      self,
      sample: jnp.ndarray,
      num_frames: int,
      deterministic: bool = True,
      image_only_indicator: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    sample = self.conv_in(sample)

    sample = self.mid_block(
        sample,
        num_frames=num_frames,
        deterministic=deterministic,
        image_only_indicator=image_only_indicator,
    )

    for block in self.up_blocks:
      sample = block(
          sample,
          num_frames=num_frames,
          deterministic=deterministic,
          image_only_indicator=image_only_indicator,
      )

    sample = self.conv_norm_out(sample)
    sample = nn.swish(sample)
    sample = self.conv_out(sample)  # (B*T, H, W, C_out)

    # Final temporal mix over frames.
    sample_t = rearrange(sample, "(b t) h w c -> b t h w c", t=num_frames)
    sample_t = self.time_conv_out(sample_t)
    return rearrange(sample_t, "b t h w c -> (b t) h w c")
