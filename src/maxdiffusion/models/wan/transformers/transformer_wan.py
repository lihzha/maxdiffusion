"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Tuple, Optional, Dict, Union, Any
import contextlib
import math
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P
from flax import nnx
import flax.linen as nn
import numpy as np
from .... import common_types
from ...modeling_flax_utils import FlaxModelMixin, get_activation
from ....configuration_utils import ConfigMixin, register_to_config
from ...embeddings_flax import (
    NNXWanImageEmbedding,
    get_1d_rotary_pos_embed,
    NNXFlaxTimesteps,
    NNXTimestepEmbedding,
    NNXPixArtAlphaTextProjection,
)
from ...normalization_flax import FP32LayerNorm
from ...attention_flax import FlaxWanAttention
from ...gradient_checkpoint import GradientCheckpointType

BlockSizes = common_types.BlockSizes


def get_frequencies(max_seq_len: int, theta: int, attention_head_dim: int):
  h_dim = w_dim = 2 * (attention_head_dim // 6)
  t_dim = attention_head_dim - h_dim - w_dim
  freqs = []
  for dim in [t_dim, h_dim, w_dim]:
    freq = get_1d_rotary_pos_embed(dim, max_seq_len, theta, freqs_dtype=jnp.float32, use_real=False)
    freqs.append(freq)
  freqs = jnp.concatenate(freqs, axis=1)
  t_size = attention_head_dim // 2 - 2 * (attention_head_dim // 6)
  hw_size = attention_head_dim // 6

  dims = [t_size, hw_size, hw_size]

  # Calculate split indices as a static list of integers
  cumulative_sizes = np.cumsum(dims)
  split_indices = cumulative_sizes[:-1].tolist()
  freqs_split = jnp.split(freqs, split_indices, axis=1)
  return freqs_split


class WanRotaryPosEmbed(nnx.Module):

  def __init__(self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0):
    self.attention_head_dim = attention_head_dim
    self.patch_size = patch_size
    self.max_seq_len = max_seq_len
    self.theta = theta

  def __call__(self, hidden_states: jax.Array, frame_positions=None) -> jax.Array:
    batch, num_frames, height, width, _ = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    freqs_split = get_frequencies(self.max_seq_len, self.theta, self.attention_head_dim)

    if frame_positions is not None:
      # frame_positions: (B, W) int array — per-batch-item latent frame indices.
      # Produces (B, 1, ppf*pph*ppw, freq_dim) so each sample gets its own RoPE.
      temporal_freqs = freqs_split[0][frame_positions]               # (B, ppf, freq_dim)
      freqs_f = temporal_freqs[:, :, None, None, :]                  # (B, ppf, 1, 1, freq_dim)
      freqs_f = jnp.broadcast_to(freqs_f, (batch, ppf, pph, ppw, freqs_split[0].shape[-1]))

      freqs_h = freqs_split[1][:pph][None, None, :, None, :]         # (1, 1, pph, 1, freq_dim)
      freqs_h = jnp.broadcast_to(freqs_h, (batch, ppf, pph, ppw, freqs_split[1].shape[-1]))

      freqs_w = freqs_split[2][:ppw][None, None, None, :, :]         # (1, 1, 1, ppw, freq_dim)
      freqs_w = jnp.broadcast_to(freqs_w, (batch, ppf, pph, ppw, freqs_split[2].shape[-1]))

      freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
      return jnp.reshape(freqs_concat, (batch, 1, ppf * pph * ppw, -1))
    else:
      temporal_freqs = freqs_split[0][:ppf]
      freqs_f = jnp.expand_dims(jnp.expand_dims(temporal_freqs, axis=1), axis=1)
      freqs_f = jnp.broadcast_to(freqs_f, (ppf, pph, ppw, freqs_split[0].shape[-1]))

      freqs_h = jnp.expand_dims(jnp.expand_dims(freqs_split[1][:pph], axis=0), axis=2)
      freqs_h = jnp.broadcast_to(freqs_h, (ppf, pph, ppw, freqs_split[1].shape[-1]))

      freqs_w = jnp.expand_dims(jnp.expand_dims(freqs_split[2][:ppw], axis=0), axis=1)
      freqs_w = jnp.broadcast_to(freqs_w, (ppf, pph, ppw, freqs_split[2].shape[-1]))

      freqs_concat = jnp.concatenate([freqs_f, freqs_h, freqs_w], axis=-1)
      return jnp.reshape(freqs_concat, (1, 1, ppf * pph * ppw, -1))


class WanTimeTextImageEmbedding(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      time_freq_dim: int,
      time_proj_dim: int,
      text_embed_dim: int,
      image_embed_dim: Optional[int] = None,
      pos_embed_seq_len: Optional[int] = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      flash_min_seq_length: int = 4096,
  ):
    self.timesteps_proj = NNXFlaxTimesteps(dim=time_freq_dim, flip_sin_to_cos=True, freq_shift=0)
    self.time_embedder = NNXTimestepEmbedding(
        rngs=rngs,
        in_channels=time_freq_dim,
        time_embed_dim=dim,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
    )
    self.act_fn = get_activation("silu")
    self.time_proj = nnx.Linear(
        rngs=rngs,
        in_features=dim,
        out_features=time_proj_dim,
        dtype=jnp.float32,
        param_dtype=weights_dtype,
        precision=precision,
        # Sharding this kernel ("embed" -> fsdp) requires the explicit batch
        # sharding constraint on timestep_proj in the per-token path (see
        # WanModel.__call__): without it GSPMD resolves the batch/contracting
        # fsdp conflict by dropping fsdp from the batch sharding, blowing up
        # the per-device [B, seq, 6*dim] f32 activation.
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )
    self.text_embedder = NNXPixArtAlphaTextProjection(
        rngs=rngs,
        in_features=text_embed_dim,
        hidden_size=dim,
        act_fn="gelu_tanh",
    )

    self.image_embedder = nnx.data(None)
    if image_embed_dim is not None:
      self.image_embedder = NNXWanImageEmbedding(
          rngs=rngs,
          in_features=image_embed_dim,
          out_features=dim,
          pos_embed_seq_len=pos_embed_seq_len,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          flash_min_seq_length=flash_min_seq_length,
      )

  def __call__(
      self, timestep: jax.Array, encoder_hidden_states: jax.Array, encoder_hidden_states_image: Optional[jax.Array] = None
  ):
    timestep = self.timesteps_proj(timestep)
    temb = self.time_embedder(timestep)
    with jax.named_scope("time_proj"):
      timestep_proj = self.time_proj(self.act_fn(temb))

    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
    encoder_attention_mask = None
    if encoder_hidden_states_image is not None:
      encoder_hidden_states_image, encoder_attention_mask = self.image_embedder(encoder_hidden_states_image)
    return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image, encoder_attention_mask


class ApproximateGELU(nnx.Module):
  r"""
  The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
  [paper](https://arxiv.org/abs/1606.08415).
  """

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim_in: int,
      dim_out: int,
      bias: bool,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
  ):
    self.proj = nnx.Linear(
        rngs=rngs,
        in_features=dim_in,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "embed",
                "mlp",
            ),
        ),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    with jax.named_scope("gelu"):
      x = self.proj(x)
    return nnx.gelu(x)


class WanFeedForward(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      dim_out: Optional[int] = None,
      mult: int = 4,
      dropout: float = 0.0,
      activation_fn: str = "geglu",
      final_dropout: bool = False,
      inner_dim: int = None,
      bias: bool = True,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      enable_jax_named_scopes: bool = False,
  ):
    if inner_dim is None:
      inner_dim = int(dim * mult)
    dim_out = dim_out if dim_out is not None else dim

    self.enable_jax_named_scopes = enable_jax_named_scopes
    self.act_fn = nnx.data(None)
    if activation_fn == "gelu-approximate":
      self.act_fn = ApproximateGELU(
          rngs=rngs, dim_in=dim, dim_out=inner_dim, bias=bias, dtype=dtype, weights_dtype=weights_dtype, precision=precision
      )
    else:
      raise NotImplementedError(f"{activation_fn} is not implemented.")

    self.drop_out = nnx.Dropout(dropout, deterministic=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=dim_out,
        use_bias=bias,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (
                "mlp",
                "embed",
            ),
        ),
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(self, hidden_states: jax.Array, deterministic: bool = True, rngs: nnx.Rngs = None) -> jax.Array:
    hidden_states = self.act_fn(hidden_states)  # Output is (4, 75600, 13824)
    hidden_states = checkpoint_name(hidden_states, "ffn_activation")
    if self.drop_out.rate > 0:
      hidden_states = self.drop_out(hidden_states, deterministic=deterministic, rngs=rngs)
    with jax.named_scope("proj_out"):
      return self.proj_out(hidden_states)  # output is (4, 75600, 5120)


class WanTransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      dim: int,
      ffn_dim: int,
      num_heads: int,
      qk_norm: str = "rms_norm_across_heads",
      cross_attn_norm: bool = False,
      eps: float = 1e-6,
      added_kv_proj_dim: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      dropout: float = 0.0,
      mask_padding_tokens: bool = True,
      enable_jax_named_scopes: bool = False,
  ):
    self.enable_jax_named_scopes = enable_jax_named_scopes

    # 1. Self-attention
    self.norm1 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
    self.attn1 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=True,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="self_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
    )

    # 1. Cross-attention
    self.attn2 = FlaxWanAttention(
        rngs=rngs,
        query_dim=dim,
        heads=num_heads,
        dim_head=dim // num_heads,
        qk_norm=qk_norm,
        eps=eps,
        added_kv_proj_dim=added_kv_proj_dim,
        image_seq_len=image_seq_len,
        flash_min_seq_length=flash_min_seq_length,
        flash_block_sizes=flash_block_sizes,
        mesh=mesh,
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        attention_kernel=attention,
        dropout=dropout,
        is_self_attention=False,
        mask_padding_tokens=mask_padding_tokens,
        residual_checkpoint_name="cross_attn",
        enable_jax_named_scopes=enable_jax_named_scopes,
    )
    assert cross_attn_norm is True
    self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)

    # 3. Feed-forward
    self.ffn = WanFeedForward(
        rngs=rngs,
        dim=dim,
        inner_dim=ffn_dim,
        activation_fn="gelu-approximate",
        dtype=dtype,
        weights_dtype=weights_dtype,
        precision=precision,
        dropout=dropout,
        enable_jax_named_scopes=enable_jax_named_scopes,
    )
    self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)

    key = rngs.params()
    self.adaln_scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 6, dim)) / dim**0.5,
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  def __call__(
      self,
      hidden_states: jax.Array,
      encoder_hidden_states: jax.Array,
      temb: jax.Array,
      rotary_emb: jax.Array,
      deterministic: bool = True,
      rngs: nnx.Rngs = None,
      encoder_attention_mask: Optional[jax.Array] = None,
      frame_level_cond: bool = False,
      cond_tokens_per_frame: int = 1,
  ):
    with self.conditional_named_scope("transformer_block"):
      # Support both global [B, 6, dim] and per-token [B, seq_len, 6, dim] temb.
      # Per-token temb is used by TI2V where first-frame tokens have timestep=0.
      if temb.ndim == 4:  # Per-token: [B, seq_len, 6, dim]
        adaln = jnp.expand_dims(self.adaln_scale_shift_table, 0)  # [1, 1, 6, dim]
        combined = adaln + temb.astype(jnp.float32)  # [B, seq_len, 6, dim]
        parts = jnp.split(combined, 6, axis=2)
        shift_msa = parts[0].squeeze(2)
        scale_msa = parts[1].squeeze(2)
        gate_msa = parts[2].squeeze(2)
        c_shift_msa = parts[3].squeeze(2)
        c_scale_msa = parts[4].squeeze(2)
        c_gate_msa = parts[5].squeeze(2)
      else:  # Global: [B, 6, dim]
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = jnp.split(
            (self.adaln_scale_shift_table + temb.astype(jnp.float32)),
            6,
            axis=1,
        )
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_heads"))
      hidden_states = jax.lax.with_sharding_constraint(hidden_states, axis_names)
      hidden_states = checkpoint_name(hidden_states, "hidden_states")
      axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_kv"))
      encoder_hidden_states = jax.lax.with_sharding_constraint(encoder_hidden_states, axis_names)

      # 1. Self-attention
      with self.conditional_named_scope("self_attn"):
        with self.conditional_named_scope("self_attn_norm"):
          norm_hidden_states = (self.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("self_attn_attn"):
          attn_output = self.attn1(
              hidden_states=norm_hidden_states,
              encoder_hidden_states=norm_hidden_states,
              rotary_emb=rotary_emb,
              deterministic=deterministic,
              rngs=rngs,
          )
        with self.conditional_named_scope("self_attn_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(hidden_states.dtype)

      # 2. Cross-attention
      with self.conditional_named_scope("cross_attn"):
        with self.conditional_named_scope("cross_attn_norm"):
          norm_hidden_states = self.norm2(hidden_states.astype(jnp.float32)).astype(hidden_states.dtype)
        with self.conditional_named_scope("cross_attn_attn"):
          if frame_level_cond:
            # SVD-style per-frame locking: each latent frame's patches attend only
            # to that frame's own cond_tokens_per_frame action tokens.
            # hidden_states: (B, F_lat*Sp, D), encoder: (B, F_lat*K, D) frame-major.
            # With K=1 softmax over a single key is constant (additive injection);
            # K>1 makes the attention weights query-dependent.
            B, L, D = norm_hidden_states.shape
            K = cond_tokens_per_frame
            F = encoder_hidden_states.shape[1] // K
            hs_r = norm_hidden_states.reshape(B * F, L // F, D)
            enc_r = encoder_hidden_states.reshape(B * F, K, D)
            attn_output = self.attn2(
                hidden_states=hs_r,
                encoder_hidden_states=enc_r,
                deterministic=deterministic,
                rngs=rngs,
            )
            attn_output = attn_output.reshape(B, L, D)
          else:
            attn_output = self.attn2(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=deterministic,
                rngs=rngs,
                encoder_attention_mask=encoder_attention_mask,
            )
        with self.conditional_named_scope("cross_attn_residual"):
          hidden_states = hidden_states + attn_output

      # 3. Feed-forward
      with self.conditional_named_scope("mlp"):
        with self.conditional_named_scope("mlp_norm"):
          norm_hidden_states = (self.norm3(hidden_states.astype(jnp.float32)) * (1 + c_scale_msa) + c_shift_msa).astype(
              hidden_states.dtype
          )
        with self.conditional_named_scope("mlp_ffn"):
          ff_output = self.ffn(norm_hidden_states, deterministic=deterministic, rngs=rngs)
        with self.conditional_named_scope("mlp_residual"):
          hidden_states = (hidden_states.astype(jnp.float32) + ff_output.astype(jnp.float32) * c_gate_msa).astype(
              hidden_states.dtype
          )
      return hidden_states


class WanModel(nnx.Module, FlaxModelMixin, ConfigMixin):

  @register_to_config
  def __init__(
      self,
      rngs: nnx.Rngs,
      model_type="t2v",
      patch_size: Tuple[int] = (1, 2, 2),
      num_attention_heads: int = 40,
      attention_head_dim: int = 128,
      in_channels: int = 16,
      out_channels: int = 16,
      text_dim: int = 4096,
      freq_dim: int = 256,
      ffn_dim: int = 13824,
      num_layers: int = 40,
      dropout: float = 0.0,
      cross_attn_norm: bool = True,
      qk_norm: Optional[str] = "rms_norm_across_heads",
      eps: float = 1e-6,
      image_dim: Optional[int] = None,
      added_kv_proj_dim: Optional[int] = None,
      rope_max_seq_len: int = 1024,
      pos_embed_seq_len: Optional[int] = None,
      image_seq_len: Optional[int] = None,
      flash_min_seq_length: int = 4096,
      flash_block_sizes: BlockSizes = None,
      mesh: jax.sharding.Mesh = None,
      dtype: jnp.dtype = jnp.float32,
      weights_dtype: jnp.dtype = jnp.float32,
      precision: jax.lax.Precision = None,
      attention: str = "dot_product",
      remat_policy: str = "None",
      names_which_can_be_saved: list = [],
      names_which_can_be_offloaded: list = [],
      mask_padding_tokens: bool = True,
      scan_layers: bool = True,
      enable_jax_named_scopes: bool = False,
  ):
    inner_dim = num_attention_heads * attention_head_dim
    out_channels = out_channels or in_channels
    self.num_layers = num_layers
    self.scan_layers = scan_layers
    self.enable_jax_named_scopes = enable_jax_named_scopes

    # 1. Patch & position embedding
    self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
    self.patch_embedding = nnx.Conv(
        in_channels,
        inner_dim,
        rngs=rngs,
        kernel_size=patch_size,
        strides=patch_size,
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            (None, None, None, None, "conv_out"),
        ),
    )

    # 2. Condition embeddings
    # image_embedding_dim=1280 for I2V model
    self.condition_embedder = WanTimeTextImageEmbedding(
        rngs=rngs,
        dim=inner_dim,
        time_freq_dim=freq_dim,
        time_proj_dim=inner_dim * 6,
        text_embed_dim=text_dim,
        image_embed_dim=image_dim,
        pos_embed_seq_len=pos_embed_seq_len,
        flash_min_seq_length=flash_min_seq_length,
    )

    # 3. Transformer blocks
    @nnx.split_rngs(splits=num_layers)
    @nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"})
    def init_block(rngs):
      return WanTransformerBlock(
          rngs=rngs,
          dim=inner_dim,
          ffn_dim=ffn_dim,
          num_heads=num_attention_heads,
          qk_norm=qk_norm,
          cross_attn_norm=cross_attn_norm,
          eps=eps,
          flash_min_seq_length=flash_min_seq_length,
          flash_block_sizes=flash_block_sizes,
          mesh=mesh,
          dtype=dtype,
          weights_dtype=weights_dtype,
          precision=precision,
          attention=attention,
          dropout=dropout,
          mask_padding_tokens=mask_padding_tokens,
          enable_jax_named_scopes=enable_jax_named_scopes,
          added_kv_proj_dim=added_kv_proj_dim,
          image_seq_len=image_seq_len,
      )

    self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
    self.names_which_can_be_offloaded = names_which_can_be_offloaded
    self.names_which_can_be_saved = names_which_can_be_saved
    if scan_layers:
      self.blocks = init_block(rngs)
    else:
      blocks = []
      for _ in range(num_layers):
        block = WanTransformerBlock(
            rngs=rngs,
            dim=inner_dim,
            ffn_dim=ffn_dim,
            num_heads=num_attention_heads,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            image_seq_len=image_seq_len,
            flash_min_seq_length=flash_min_seq_length,
            flash_block_sizes=flash_block_sizes,
            mesh=mesh,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            attention=attention,
            enable_jax_named_scopes=enable_jax_named_scopes,
        )
        blocks.append(block)
      self.blocks = nnx.data(blocks)

    self.norm_out = FP32LayerNorm(rngs=rngs, dim=inner_dim, eps=eps, elementwise_affine=False)
    self.proj_out = nnx.Linear(
        rngs=rngs,
        in_features=inner_dim,
        out_features=out_channels * math.prod(patch_size),
        dtype=dtype,
        param_dtype=weights_dtype,
        precision=precision,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("embed", None)),
    )
    key = rngs.params()
    self.scale_shift_table = nnx.Param(
        jax.random.normal(key, (1, 2, inner_dim)) / inner_dim**0.5,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, None, "embed")),
    )

  def conditional_named_scope(self, name: str):
    """Return a JAX named scope if enabled, otherwise a null context."""
    return jax.named_scope(name) if self.enable_jax_named_scopes else contextlib.nullcontext()

  @jax.named_scope("WanModel")
  def __call__(
      self,
      hidden_states: jax.Array,
      timestep: jax.Array,
      encoder_hidden_states: jax.Array,
      encoder_hidden_states_image: Optional[jax.Array] = None,
      return_dict: bool = True,
      attention_kwargs: Optional[Dict[str, Any]] = None,
      deterministic: bool = True,
      rngs: Optional[nnx.Rngs] = None,
      skip_blocks: Optional[jax.Array] = None,
      cached_residual: Optional[jax.Array] = None,
      return_residual: bool = False,
      frame_level_cond: bool = False,
      cond_tokens_per_frame: int = 1,
      frame_positions: Optional[tuple] = None,
  ) -> Union[jax.Array, Tuple[jax.Array, jax.Array], Dict[str, jax.Array]]:
    hidden_states = nn.with_logical_constraint(hidden_states, ("batch", None, None, None, None))
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    with self.conditional_named_scope("rotary_embedding"):
      rotary_emb = self.rope(hidden_states, frame_positions=frame_positions)
    with self.conditional_named_scope("patch_embedding"):
      hidden_states = self.patch_embedding(hidden_states)
      hidden_states = jax.lax.collapse(hidden_states, 1, -1)
    per_token_t = timestep.ndim == 2  # [B, seq_len] for TI2V
    with self.conditional_named_scope("condition_embedder"):
      if per_token_t:
        # Per-token timestep: process time and text embeddings separately.
        # This matches the official WAN 2.2 TI2V pipeline where first-frame
        # tokens receive timestep=0 (clean) and other tokens receive timestep=t.
        bt, sl = timestep.shape
        t_sinusoidal = self.condition_embedder.timesteps_proj(timestep)  # [B, sl, freq_dim]
        temb = self.condition_embedder.time_embedder(t_sinusoidal)  # [B, sl, dim]
        # Force full (data×fsdp×context)-way batch sharding after time_embedder,
        # but only when context > 1 (e.g. TI2V with context=4).
        # time_embedder weights have ("embed","mlp") partitioning where "embed"
        # maps to [context,fsdp], so GSPMD may drop those axes from the batch
        # sharding to avoid doubly-sharded axes.  This constraint restores the
        # full batch sharding before the time_proj matmul.
        # When context=1 the natural GSPMD sharding is already correct and
        # adding any constraint here breaks op fusion (~20% overhead).
        _mesh = getattr(self.config, "mesh", None)
        _ctx = _mesh.shape.get("context", 1) if _mesh is not None else 1
        if _ctx > 1:
          batch_axes = ("data", "fsdp", "context")
        else:
          batch_axes = ("data", "fsdp")
        # temb.ndim is static at trace time; rank varies with parallelism config.
        temb = jax.lax.with_sharding_constraint(
            temb, P(batch_axes, *([None] * (temb.ndim - 1)))
        )
        with jax.named_scope("time_proj"):
          timestep_proj = self.condition_embedder.time_proj(self.condition_embedder.act_fn(temb))  # [B, sl, dim*6]
        # time_proj's kernel is fsdp-sharded on its contracting ("embed") dim,
        # which conflicts with the fsdp-sharded batch here.  Pin the output's
        # batch sharding so GSPMD must all-gather the weight (standard FSDP)
        # instead of dropping fsdp from the batch, which would materialize the
        # full [B, sl, 6*dim] f32 activation on every device.
        timestep_proj = jax.lax.with_sharding_constraint(timestep_proj, P(batch_axes, None, None))
        timestep_proj = timestep_proj.reshape(bt, sl, 6, -1)  # [B, sl, 6, dim]
        # Text processing
        encoder_hidden_states = self.condition_embedder.text_embedder(encoder_hidden_states)
        encoder_hidden_states_image = None
        encoder_attention_mask = None
      else:
        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
            encoder_attention_mask,
        ) = self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    if encoder_hidden_states_image is not None:
      encoder_hidden_states = jnp.concatenate([encoder_hidden_states_image, encoder_hidden_states], axis=1)
      if encoder_attention_mask is not None:
        text_mask = jnp.ones(
            (encoder_hidden_states.shape[0], encoder_hidden_states.shape[1] - encoder_hidden_states_image.shape[1]),
            dtype=jnp.int32,
        )
        encoder_attention_mask = jnp.concatenate([encoder_attention_mask, text_mask], axis=1)
      encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)

    def _run_all_blocks(h):
      if self.scan_layers:
        # Use jax.lax.scan directly over split block state instead of nnx.scan.
        # nnx.scan rejects parameters that come from an outer JAX trace level,
        # which breaks when the model is called from inside jax.lax.fori_loop
        # (e.g. the on-policy distillation rollout).  Slicing a JitTracer array
        # inside jax.lax.scan produces scan-level tracers, so nnx.merge inside
        # the body sees Variables at the correct level.  The generated XLA is
        # identical to the nnx.scan path.
        graphdef_blocks, blocks_state = nnx.split(self.blocks)

        def scan_fn(carry, block_state):
          hidden_states_carry, rngs_carry = carry
          block = nnx.merge(graphdef_blocks, block_state)
          hidden_states = block(
              hidden_states_carry,
              encoder_hidden_states,
              timestep_proj,
              rotary_emb,
              deterministic,
              rngs_carry,
              encoder_attention_mask,
              frame_level_cond=frame_level_cond,
              cond_tokens_per_frame=cond_tokens_per_frame,
          )
          new_carry = (hidden_states, rngs_carry)
          return new_carry, None

        rematted_block_forward = self.gradient_checkpoint.apply(
            scan_fn, self.names_which_can_be_saved, self.names_which_can_be_offloaded, prevent_cse=not self.scan_layers
        )
        initial_carry = (h, rngs)
        final_carry, _ = jax.lax.scan(rematted_block_forward, initial_carry, blocks_state)

        h_out, _ = final_carry
      else:
        h_out = h
        for block in self.blocks:

          def layer_forward(hidden_states):
            return block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                deterministic,
                rngs,
                encoder_attention_mask=encoder_attention_mask,
                frame_level_cond=frame_level_cond,
                cond_tokens_per_frame=cond_tokens_per_frame,
            )

          rematted_layer_forward = self.gradient_checkpoint.apply(
              layer_forward,
              self.names_which_can_be_saved,
              self.names_which_can_be_offloaded,
              prevent_cse=not self.scan_layers,
          )
          h_out = rematted_layer_forward(h_out)
      return h_out

    hidden_states_before_blocks = hidden_states

    if skip_blocks:
      if cached_residual is None:
        raise ValueError("cached_residual must be provided when skip_blocks is True")
      hidden_states = hidden_states + cached_residual
    else:
      hidden_states = _run_all_blocks(hidden_states)

    residual_x = hidden_states - hidden_states_before_blocks

    if per_token_t:
      # temb: [B, seq_len, dim] — per-token modulation for final head
      combined_head = jnp.expand_dims(self.scale_shift_table, 0) + jnp.expand_dims(temb, 2)  # [B, sl, 2, dim]
      shift, scale = jnp.split(combined_head, 2, axis=2)
      shift = shift.squeeze(2)  # [B, sl, dim]
      scale = scale.squeeze(2)  # [B, sl, dim]
    else:
      shift, scale = jnp.split(self.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)
    hidden_states = (self.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift).astype(hidden_states.dtype)
    with jax.named_scope("proj_out"):
      hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    hidden_states = hidden_states.reshape(batch_size, -1, num_frames, height, width)

    if return_residual:
      return hidden_states, residual_x
    return hidden_states
