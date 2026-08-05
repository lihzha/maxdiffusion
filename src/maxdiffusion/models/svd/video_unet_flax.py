# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""VideoUNet — the denoiser backbone of Stable Video Diffusion (base).

Fork of ``FlaxUNet2DConditionModel`` that threads ``num_frames`` and
``image_only_indicator`` through every block, replaces the spatial UNet blocks
with their video counterparts, and adds a 768-dim ADM vector path
(``addition_embed_type='adm_vector'``) for the micro-conditioning
``(fps_id, motion_bucket_id, cond_aug)``.

Base SVD config (from ``svd.yaml``)::

    in_channels:   8      # 4 latent + 4 VAE-encoded concat image
    out_channels:  4
    model_channels: 320   # block_out_channels = (320, 640, 1280, 1280)
    layers_per_block: 2
    num_head_channels: 64 # -> num_attention_heads = (5, 10, 20, 20)
    context_dim: 1024     # CLIP ViT-H/14 image tower
    adm_in_channels: 768  # 3 x 256 sinusoidal micro-conditions
    use_linear_projection: true
    transformer_depth: 1
    merge_strategy: learned_with_images
    video_kernel_size: [3, 1, 1]
"""

from typing import Dict, Optional, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from ...configuration_utils import ConfigMixin, flax_register_to_config
from ...utils import BaseOutput
from ..embeddings_flax import FlaxTimestepEmbedding, FlaxTimesteps
from ..modeling_flax_utils import FlaxModelMixin
from .. import quantizations
from ...common_types import BlockSizes
from .video_unet_blocks_flax import (
    FlaxCrossAttnDownVideoBlock,
    FlaxCrossAttnUpVideoBlock,
    FlaxDownVideoBlock,
    FlaxUpVideoBlock,
    FlaxVideoMidBlock2DCrossAttn,
)

Quant = quantizations.AqtQuantization


@flax.struct.dataclass
class FlaxVideoUNetOutput(BaseOutput):
  sample: jnp.ndarray


@flax_register_to_config
class FlaxVideoUNet(nn.Module, FlaxModelMixin, ConfigMixin):
  """VideoUNet for Stable Video Diffusion.

  Inputs to ``__call__``:

      sample:               (B*T, in_channels, H, W) noisy latent concatenated
                            with the tiled VAE-encoded conditioning image.
      timesteps:            (B,) or scalar; EDM-scaled ``c_noise(sigma)``.
      encoder_hidden_states:(B*T, 1, 1024) CLIP ViT-H/14 image features.
      added_cond_kwargs:    {"adm_vector": (B, adm_in_channels)} — the
                            sinusoidal micro-cond vector from
                            ``embeddings_flax.svd_micro_cond_embed``.
      num_frames:           static int T (14 for base SVD).
      image_only_indicator: (B, T) float; 0 = pure video (default), 1 = treat
                            this frame as image-only (disables temporal branch).
  """

  sample_size: int = 72
  in_channels: int = 8
  out_channels: int = 4
  down_block_types: Tuple[str, ...] = (
      "CrossAttnDownBlock2D",
      "CrossAttnDownBlock2D",
      "CrossAttnDownBlock2D",
      "DownBlock2D",
  )
  up_block_types: Tuple[str, ...] = (
      "UpBlock2D",
      "CrossAttnUpBlock2D",
      "CrossAttnUpBlock2D",
      "CrossAttnUpBlock2D",
  )
  block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
  layers_per_block: int = 2
  num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = (5, 10, 20, 20)
  cross_attention_dim: int = 1024
  transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1
  use_linear_projection: bool = True
  dropout: float = 0.0
  norm_num_groups: int = 32
  flip_sin_to_cos: bool = True
  freq_shift: int = 0

  # SVD-specific
  addition_embed_type: Optional[str] = "adm_vector"
  adm_in_channels: int = 768
  video_kernel_size: Tuple[int, int, int] = (3, 1, 1)
  merge_strategy: str = "learned_with_images"

  attention_kernel: str = "dot_product"
  temporal_attention_kernel: str = "dot_product"
  # Chunked-query attention path (jax_memory_efficient_attention). Enables
  # 512×512 on a 24 GB GPU by avoiding the full N×N scores matrix at the
  # shallowest spatial stage. See docs/svd_debug_notes.md §6.3.
  use_memory_efficient_attention: bool = False
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  split_head_dim: bool = False
  precision: jax.lax.Precision = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  quant: Quant = None

  def _resolve_num_heads(self) -> Tuple[int, ...]:
    if isinstance(self.num_attention_heads, int):
      return (self.num_attention_heads,) * len(self.down_block_types)
    return tuple(self.num_attention_heads)

  def _resolve_transformer_layers(self) -> Tuple[int, ...]:
    if isinstance(self.transformer_layers_per_block, int):
      return (self.transformer_layers_per_block,) * len(self.down_block_types)
    return tuple(self.transformer_layers_per_block)

  def setup(self):
    block_out_channels = self.block_out_channels
    time_embed_dim = block_out_channels[0] * 4

    num_attention_heads = self._resolve_num_heads()
    transformer_layers = self._resolve_transformer_layers()

    # Input conv (8 -> 320).
    self.conv_in = nn.Conv(
        block_out_channels[0],
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

    # Time embedding.
    self.time_proj = FlaxTimesteps(
        block_out_channels[0], flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
    )
    self.time_embedding = FlaxTimestepEmbedding(
        time_embed_dim, dtype=self.dtype, weights_dtype=self.weights_dtype
    )

    # ADM vector path (SVD-specific).
    if self.addition_embed_type == "adm_vector":
      self.add_embedding = FlaxTimestepEmbedding(
          time_embed_dim, dtype=self.dtype, weights_dtype=self.weights_dtype
      )
    elif self.addition_embed_type is None:
      self.add_embedding = None
    else:
      raise ValueError(
          f"FlaxVideoUNet: addition_embed_type must be 'adm_vector' or None, "
          f"got {self.addition_embed_type!r}"
      )

    # Down blocks.
    down_blocks = []
    output_channel = block_out_channels[0]
    for i, btype in enumerate(self.down_block_types):
      input_channel = output_channel
      output_channel = block_out_channels[i]
      is_final_block = i == len(block_out_channels) - 1
      if btype in ("CrossAttnDownBlock2D", "CrossAttnDownBlockSpatioTemporal"):
        down_blocks.append(
            FlaxCrossAttnDownVideoBlock(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=self.dropout,
                num_layers=self.layers_per_block,
                num_attention_heads=num_attention_heads[i],
                add_downsample=not is_final_block,
                use_linear_projection=self.use_linear_projection,
                context_dim=self.cross_attention_dim,
                transformer_layers_per_block=transformer_layers[i],
                norm_num_groups=self.norm_num_groups,
                video_kernel_size=self.video_kernel_size,
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
      elif btype in ("DownBlock2D", "DownBlockSpatioTemporal"):
        down_blocks.append(
            FlaxDownVideoBlock(
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=self.dropout,
                num_layers=self.layers_per_block,
                add_downsample=not is_final_block,
                norm_num_groups=self.norm_num_groups,
                video_kernel_size=self.video_kernel_size,
                merge_strategy=self.merge_strategy,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                precision=self.precision,
            )
        )
      else:
        raise ValueError(f"FlaxVideoUNet: unsupported down_block_type={btype!r}")
    self.down_blocks = down_blocks

    # Mid block.
    self.mid_block = FlaxVideoMidBlock2DCrossAttn(
        in_channels=block_out_channels[-1],
        dropout=self.dropout,
        num_attention_heads=num_attention_heads[-1],
        use_linear_projection=self.use_linear_projection,
        context_dim=self.cross_attention_dim,
        transformer_layers_per_block=transformer_layers[-1],
        norm_num_groups=self.norm_num_groups,
        video_kernel_size=self.video_kernel_size,
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

    # Up blocks.
    reversed_block_out_channels = list(reversed(block_out_channels))
    reversed_num_attention_heads = list(reversed(num_attention_heads))
    reversed_transformer_layers = list(reversed(transformer_layers))

    up_blocks = []
    output_channel = reversed_block_out_channels[0]
    for i, btype in enumerate(self.up_block_types):
      prev_output_channel = output_channel
      output_channel = reversed_block_out_channels[i]
      input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
      is_final_block = i == len(block_out_channels) - 1
      if btype in ("CrossAttnUpBlock2D", "CrossAttnUpBlockSpatioTemporal"):
        up_blocks.append(
            FlaxCrossAttnUpVideoBlock(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                dropout=self.dropout,
                num_layers=self.layers_per_block + 1,
                num_attention_heads=reversed_num_attention_heads[i],
                add_upsample=not is_final_block,
                use_linear_projection=self.use_linear_projection,
                context_dim=self.cross_attention_dim,
                transformer_layers_per_block=reversed_transformer_layers[i],
                norm_num_groups=self.norm_num_groups,
                video_kernel_size=self.video_kernel_size,
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
      elif btype in ("UpBlock2D", "UpBlockSpatioTemporal"):
        up_blocks.append(
            FlaxUpVideoBlock(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                dropout=self.dropout,
                num_layers=self.layers_per_block + 1,
                add_upsample=not is_final_block,
                norm_num_groups=self.norm_num_groups,
                video_kernel_size=self.video_kernel_size,
                merge_strategy=self.merge_strategy,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                precision=self.precision,
            )
        )
      else:
        raise ValueError(f"FlaxVideoUNet: unsupported up_block_type={btype!r}")
    self.up_blocks = up_blocks

    # Output norm + conv.
    self.conv_norm_out = nn.GroupNorm(
        num_groups=self.norm_num_groups, epsilon=1e-5, dtype=self.dtype, param_dtype=self.weights_dtype
    )
    self.conv_out = nn.Conv(
        self.out_channels,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding=((1, 1), (1, 1)),
        dtype=jnp.float32,
        param_dtype=self.weights_dtype,
        precision=self.precision,
    )

  def init_weights(
      self, rng: jax.Array, eval_only: bool = False, quantization_enabled: bool = False
  ) -> FrozenDict:
    batch = 1
    num_frames = 2  # minimal T that still exercises every temporal conv (kernel=3)
    bt = batch * num_frames

    sample_shape = (bt, self.in_channels, self.sample_size, self.sample_size)
    if eval_only:
      sample = jax.ShapeDtypeStruct(sample_shape, dtype=jnp.bfloat16)
      timesteps = jax.ShapeDtypeStruct((bt,), dtype=jnp.bfloat16)
      encoder_hidden_states = jax.ShapeDtypeStruct((bt, 1, self.cross_attention_dim), dtype=jnp.bfloat16)
    else:
      sample = jnp.zeros(sample_shape, dtype=jnp.bfloat16)
      timesteps = jnp.ones((bt,), dtype=jnp.int32)
      encoder_hidden_states = jnp.zeros((bt, 1, self.cross_attention_dim), dtype=jnp.bfloat16)

    added_cond_kwargs = None
    if self.addition_embed_type == "adm_vector":
      if eval_only:
        adm_vec = jax.ShapeDtypeStruct((batch, self.adm_in_channels), dtype=jnp.bfloat16)
      else:
        adm_vec = jnp.zeros((batch, self.adm_in_channels), dtype=jnp.bfloat16)
      added_cond_kwargs = {"adm_vector": adm_vec}

    params_rng, dropout_rng = jax.random.split(rng)
    rngs = {"params": params_rng, "dropout": dropout_rng}
    if quantization_enabled:
      rngs["aqt"] = params_rng

    # num_frames must stay a concrete Python int through init — the UNet uses
    # it for jnp.repeat / jnp.arange / einops rearranges that require static
    # values. Bind it via closure so Flax's internal jit doesn't trace it.
    def _init_fn(r, s, t, e, a, i):
      return self.init(r, s, t, e, a, i, num_frames=num_frames)

    if eval_only:
      return jax.eval_shape(
          _init_fn, rngs, sample, timesteps, encoder_hidden_states,
          added_cond_kwargs, None,
      )["params"]
    return _init_fn(
        rngs, sample, timesteps, encoder_hidden_states,
        added_cond_kwargs, None,
    )["params"]

  def __call__(
      self,
      sample: jnp.ndarray,
      timesteps: jnp.ndarray,
      encoder_hidden_states: jnp.ndarray,
      added_cond_kwargs: Optional[Union[Dict, FrozenDict]] = None,
      image_only_indicator: Optional[jnp.ndarray] = None,
      num_frames: int = 14,
      return_dict: bool = True,
      train: bool = False,
      cross_attention_kwargs: Optional[Union[Dict, FrozenDict]] = None,
      frame_level_cond: bool = False,
      action_hidden_states: Optional[jnp.ndarray] = None,
  ) -> Union[FlaxVideoUNetOutput, Tuple]:
    # 1. time
    if not isinstance(timesteps, jnp.ndarray):
      timesteps = jnp.array([timesteps], dtype=jnp.int32)
    elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
      timesteps = timesteps.astype(dtype=self.dtype)
      timesteps = jnp.expand_dims(timesteps, 0)

    t_emb = self.time_proj(timesteps)
    t_emb = self.time_embedding(t_emb)

    # 2. additional embeddings (SVD ADM vector)
    if self.addition_embed_type == "adm_vector":
      if added_cond_kwargs is None or "adm_vector" not in added_cond_kwargs:
        raise ValueError(
            "FlaxVideoUNet needs added_cond_kwargs={'adm_vector': (B, adm_in_channels)}"
        )
      adm_vec = added_cond_kwargs["adm_vector"]
      add_emb = self.add_embedding(adm_vec)  # (B, time_embed_dim)
      # Tile ADM embedding over T so it matches t_emb's (B*T, time_embed_dim) shape.
      add_emb = jnp.repeat(add_emb, num_frames, axis=0)
      # t_emb may be (1, time_embed_dim) (scalar timesteps); broadcast.
      t_emb = t_emb + add_emb

    # 2c. AdaLN action conditioning (action_cond_mode='adaln'). t_emb is already
    # per-(sample, frame) at (B*T, time_embed_dim) and is what every spatial and
    # temporal resnet consumes via its time_emb_proj, so summing here routes the
    # action signal through the same modulation path the timestep uses — the
    # SVD analogue of WAN's per-token temb injection. In this mode the caller
    # sends no per-frame action tokens to cross-attention.
    if action_hidden_states is not None:
      if action_hidden_states.shape != t_emb.shape:
        raise ValueError(
            "FlaxVideoUNet: action_hidden_states must match t_emb's shape "
            f"{t_emb.shape} (B*T, time_embed_dim), got {action_hidden_states.shape}"
        )
      t_emb = t_emb + action_hidden_states.astype(t_emb.dtype)

    # 2b. frame-level cross-attn context (Ctrl-World / action-conditioned SVD).
    # Base SVD passes encoder_hidden_states pre-tiled to (B*T, 1, C). Ctrl-World
    # instead passes a per-frame context (B, T, C) and reshapes it here so each
    # frame attends to its own token. We detect the case by rank=3 with a
    # leading dim equal to batch rather than batch*num_frames.
    if frame_level_cond and encoder_hidden_states.ndim == 3:
      b_lead = encoder_hidden_states.shape[0]
      # If the caller already flattened to (B*T, S, C), leave it alone.
      if b_lead * num_frames == sample.shape[0]:
        encoder_hidden_states = encoder_hidden_states.reshape(
            b_lead * num_frames, -1, encoder_hidden_states.shape[-1]
        )

    # 3. conv_in (NCHW -> NHWC)
    sample = jnp.transpose(sample, (0, 2, 3, 1))
    sample = self.conv_in(sample)

    # 4. down
    down_block_res_samples = (sample,)
    for block in self.down_blocks:
      if isinstance(block, FlaxCrossAttnDownVideoBlock):
        sample, res_samples = block(
            sample, t_emb, encoder_hidden_states,
            num_frames=num_frames, deterministic=not train,
            image_only_indicator=image_only_indicator,
            cross_attention_kwargs=cross_attention_kwargs,
        )
      else:
        sample, res_samples = block(
            sample, t_emb, num_frames=num_frames, deterministic=not train,
            image_only_indicator=image_only_indicator,
        )
      down_block_res_samples += res_samples

    # 5. mid
    sample = self.mid_block(
        sample, t_emb, encoder_hidden_states,
        num_frames=num_frames, deterministic=not train,
        image_only_indicator=image_only_indicator,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    # 6. up
    for block in self.up_blocks:
      res_samples = down_block_res_samples[-(self.layers_per_block + 1):]
      down_block_res_samples = down_block_res_samples[:-(self.layers_per_block + 1)]
      if isinstance(block, FlaxCrossAttnUpVideoBlock):
        sample = block(
            sample, res_samples, t_emb, encoder_hidden_states,
            num_frames=num_frames, deterministic=not train,
            image_only_indicator=image_only_indicator,
            cross_attention_kwargs=cross_attention_kwargs,
        )
      else:
        sample = block(
            sample, res_samples, t_emb,
            num_frames=num_frames, deterministic=not train,
            image_only_indicator=image_only_indicator,
        )

    # 7. post
    sample = self.conv_norm_out(sample)
    sample = nn.silu(sample)
    sample = self.conv_out(sample)
    sample = jnp.transpose(sample, (0, 3, 1, 2))  # NHWC -> NCHW

    if not return_dict:
      return (sample,)
    return FlaxVideoUNetOutput(sample=sample)
