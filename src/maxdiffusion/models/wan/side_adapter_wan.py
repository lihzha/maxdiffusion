"""Action-adapter conditioning for frozen WAN TI2V transformers.

This module ports the side-adapter architecture from ``../Wan2.2`` to NNX.
The frozen WAN transformer is kept outside the trainable adapter module so the
trainer can put only adapter parameters in the optimizer state.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import nnx
from jax.ad_checkpoint import checkpoint_name

from maxdiffusion.models.normalization_flax import FP32LayerNorm


def parse_side_adapter_layers(text: str | Sequence[int] | None, num_layers: int) -> tuple[int, ...]:
    """Parse layer selection strings such as ``"0-29"`` or ``"0,2,4"``."""
    if text is None:
        layers = tuple(range(num_layers))
    elif isinstance(text, str):
        value = text.strip().lower()
        if value in ("", "all"):
            layers = tuple(range(num_layers))
        elif value == "none":
            layers = tuple()
        else:
            out: list[int] = []
            for part in value.split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start, end = [int(x) for x in part.split("-", 1)]
                    if end < start:
                        raise ValueError(f"bad side adapter layer range {part!r}")
                    out.extend(range(start, end + 1))
                else:
                    out.append(int(part))
            layers = tuple(dict.fromkeys(out))
    else:
        layers = tuple(int(x) for x in text)

    bad = [x for x in layers if x < 0 or x >= num_layers]
    if bad:
        raise ValueError(f"side_adapter_layers contains invalid indices {bad}; num_layers={num_layers}")
    return layers


def _dtype(name: str | jnp.dtype) -> jnp.dtype:
    if isinstance(name, jnp.dtype):
        return name
    return {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }[name]


class NNXSimpleCrossAttention(nnx.Module):
    """Small multi-head cross attention used by action pooling/adapters."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        query_dim: int,
        kv_dim: int,
        hidden_dim: int,
        heads: int,
        out_dim: int,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
        zero_init_out: bool = False,
    ):
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if heads <= 0 or hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        qkv_init = nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "heads"))
        self.to_q = nnx.Linear(
            in_features=query_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=qkv_init,
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
        )
        self.to_k = nnx.Linear(
            in_features=kv_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=qkv_init,
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
        )
        self.to_v = nnx.Linear(
            in_features=kv_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=qkv_init,
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("heads",)),
        )
        out_kernel_init = nnx.initializers.zeros if zero_init_out else nnx.initializers.lecun_normal()
        self.to_out = nnx.Linear(
            in_features=hidden_dim,
            out_features=out_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(out_kernel_init, ("heads", "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
        )

    def __call__(self, query: jax.Array, key_value: jax.Array) -> jax.Array:
        q = self.to_q(query.astype(self.dtype))
        k = self.to_k(key_value.astype(self.dtype))
        v = self.to_v(key_value.astype(self.dtype))

        b, q_len, _ = q.shape
        kv_len = k.shape[1]
        q = jnp.reshape(q, (b, q_len, self.heads, self.head_dim))
        k = jnp.reshape(k, (b, kv_len, self.heads, self.head_dim))
        v = jnp.reshape(v, (b, kv_len, self.heads, self.head_dim))
        attended = jax.nn.dot_product_attention(q, k, v)
        attended = jnp.reshape(attended, (b, q_len, self.hidden_dim))
        return self.to_out(attended)


class NNXActionTokenEncoder(nnx.Module):
    """Encode a fixed DROID action trajectory into a small token set."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        action_dim: int,
        action_len: int,
        token_dim: int,
        num_tokens: int,
        hidden_dim: int = 512,
        heads: int = 4,
        action_repr: str = "delta",
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
    ):
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        self.action_dim = int(action_dim)
        self.action_len = int(action_len)
        self.num_tokens = int(num_tokens)
        self.action_repr = action_repr
        self.dtype = dtype

        self.action_proj = nnx.Linear(
            in_features=action_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )
        pos_key = rngs.params()
        query_key = rngs.params()
        self.action_pos = nnx.Param(
            jax.random.normal(pos_key, (1, action_len, hidden_dim), dtype=weights_dtype) * jnp.asarray(0.02, weights_dtype)
        )
        self.query_tokens = nnx.Param(
            jax.random.normal(query_key, (1, num_tokens, hidden_dim), dtype=weights_dtype) * jnp.asarray(0.02, weights_dtype)
        )
        self.pool = NNXSimpleCrossAttention(
            rngs=rngs,
            query_dim=hidden_dim,
            kv_dim=hidden_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            out_dim=hidden_dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
        self.out_norm = FP32LayerNorm(rngs=rngs, dim=hidden_dim, eps=1e-5, elementwise_affine=True)
        self.out_1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )
        self.out_2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=token_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
        )

    def _prepare_actions(self, actions: jax.Array) -> jax.Array:
        if actions.shape[1] != self.action_len or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"actions shape mismatch: got {actions.shape}; expected [B, {self.action_len}, {self.action_dim}]"
            )
        actions = actions.astype(jnp.float32)
        if self.action_repr == "delta":
            return actions - actions[:, 0:1]
        if self.action_repr == "raw":
            return actions
        raise ValueError(f"unknown action_repr {self.action_repr!r}")

    def __call__(self, actions: jax.Array) -> jax.Array:
        actions = self._prepare_actions(actions)
        memory = self.action_proj(actions.astype(self.dtype)) + self.action_pos.astype(self.dtype)
        query = jnp.broadcast_to(self.query_tokens.astype(self.dtype), (actions.shape[0], self.num_tokens, memory.shape[-1]))
        tokens = self.pool(query, memory)
        tokens = self.out_norm(tokens)
        tokens = jax.nn.silu(self.out_1(tokens))
        return self.out_2(tokens)


class NNXActionSideAdapter(nnx.Module):
    """Zero-initialized residual adapter from video tokens to action tokens."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        action_token_dim: int,
        hidden_dim: int,
        heads: int,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
    ):
        self.dtype = dtype
        self.norm_x = FP32LayerNorm(rngs=rngs, dim=dim, eps=1e-5, elementwise_affine=True)
        self.norm_action = FP32LayerNorm(rngs=rngs, dim=action_token_dim, eps=1e-5, elementwise_affine=True)
        self.x_proj = nnx.Linear(
            in_features=dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )
        self.action_proj = nnx.Linear(
            in_features=action_token_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )
        self.attn = NNXSimpleCrossAttention(
            rngs=rngs,
            query_dim=hidden_dim,
            kv_dim=hidden_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            out_dim=hidden_dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
        self.ffn_norm = FP32LayerNorm(rngs=rngs, dim=hidden_dim, eps=1e-5, elementwise_affine=True)
        self.ffn_1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=4 * hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )
        self.ffn_2 = nnx.Linear(
            in_features=4 * hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
        )
        self.out = nnx.Linear(
            in_features=hidden_dim,
            out_features=dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp", "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
        )
        self.gate = nnx.Param(jnp.ones((), dtype=weights_dtype))

    def __call__(self, hidden_states: jax.Array, action_tokens: jax.Array) -> jax.Array:
        q = self.x_proj(self.norm_x(hidden_states).astype(self.dtype))
        kv = self.action_proj(self.norm_action(action_tokens).astype(self.dtype))
        delta = self.attn(q, kv)
        ffn = self.ffn_norm(delta)
        ffn = jax.nn.silu(self.ffn_1(ffn))
        delta = delta + self.ffn_2(ffn)
        return self.gate.astype(delta.dtype) * self.out(delta)


class NNXPreContextFeatureContextHead(nnx.Module):
    """Predict T5-space context tokens from WAN pre-cross-attention features."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        feature_dim: int,
        action_token_dim: int,
        text_dim: int,
        context_tokens: int,
        heads: int,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
    ):
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if feature_dim % heads != 0:
            heads = math.gcd(feature_dim, heads)
        self.context_tokens = int(context_tokens)
        self.heads = int(heads)
        self.dtype = dtype
        self.norm_features = FP32LayerNorm(rngs=rngs, dim=feature_dim, eps=1e-5, elementwise_affine=True)
        self.norm_actions = FP32LayerNorm(rngs=rngs, dim=action_token_dim, eps=1e-5, elementwise_affine=True)
        self.action_proj = nnx.data(None)
        if action_token_dim != feature_dim:
            self.action_proj = nnx.Linear(
                in_features=action_token_dim,
                out_features=feature_dim,
                rngs=rngs,
                dtype=dtype,
                param_dtype=weights_dtype,
                precision=precision,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
                bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
            )
        query_key = rngs.params()
        self.query_tokens = nnx.Param(
            jax.random.normal(query_key, (1, context_tokens, feature_dim), dtype=weights_dtype)
            * jnp.asarray(0.02, weights_dtype)
        )
        self.attn = NNXSimpleCrossAttention(
            rngs=rngs,
            query_dim=feature_dim,
            kv_dim=feature_dim,
            hidden_dim=feature_dim,
            heads=self.heads,
            out_dim=feature_dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
        self.ffn_norm = FP32LayerNorm(rngs=rngs, dim=feature_dim, eps=1e-5, elementwise_affine=True)
        self.ffn_1 = nnx.Linear(
            in_features=feature_dim,
            out_features=4 * feature_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )
        self.ffn_2 = nnx.Linear(
            in_features=4 * feature_dim,
            out_features=feature_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("mlp", "embed")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("embed",)),
        )
        self.to_context = nnx.Linear(
            in_features=feature_dim,
            out_features=text_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )

    def __call__(self, features: jax.Array, action_tokens: jax.Array) -> jax.Array:
        features = self.norm_features(features).astype(self.dtype)
        action_tokens = self.norm_actions(action_tokens).astype(self.dtype)
        if self.action_proj is not None:
            action_tokens = self.action_proj(action_tokens)
        memory = jnp.concatenate([features, action_tokens], axis=1)
        query = jnp.broadcast_to(self.query_tokens.astype(self.dtype), (features.shape[0], self.context_tokens, features.shape[-1]))
        hidden = self.attn(query, memory)
        ffn = self.ffn_norm(hidden)
        ffn = jax.nn.silu(self.ffn_1(ffn))
        hidden = hidden + self.ffn_2(ffn)
        return self.to_context(hidden)


class NNXWanSideAdapterStack(nnx.Module):
    """Trainable action token encoder plus a selected WAN action adapter."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        num_layers: int,
        model_dim: int,
        text_dim: int,
        action_adapter_type: str = "side_adapter",
        action_dim: int = 7,
        action_len: int = 32,
        action_repr: str = "delta",
        action_tokens: int = 8,
        action_hidden: int = 512,
        action_heads: int = 4,
        side_adapter_layers: str | Sequence[int] | None = None,
        side_adapter_hidden: int = 512,
        side_adapter_heads: int = 8,
        pre_context_tokens: int = 8,
        pre_context_heads: int = 8,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
    ):
        action_adapter_type = action_adapter_type.strip().lower()
        if action_adapter_type not in ("side_adapter", "pre_context"):
            raise ValueError(f"unknown action_adapter_type {action_adapter_type!r}")
        self.action_adapter_type = action_adapter_type
        self.dtype = dtype
        self.action_encoder = NNXActionTokenEncoder(
            rngs=rngs,
            action_dim=action_dim,
            action_len=action_len,
            token_dim=model_dim,
            num_tokens=action_tokens,
            hidden_dim=action_hidden,
            heads=action_heads,
            action_repr=action_repr,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
        )
        self.side_adapter_layers = nnx.data(())
        self.side_adapters = nnx.data([])
        self.pre_context_head = nnx.data(None)
        if action_adapter_type == "side_adapter":
            self.side_adapter_layers = parse_side_adapter_layers(side_adapter_layers, num_layers)
            self.side_adapters = nnx.data([
                NNXActionSideAdapter(
                    rngs=rngs,
                    dim=model_dim,
                    action_token_dim=model_dim,
                    hidden_dim=side_adapter_hidden,
                    heads=side_adapter_heads,
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                    precision=precision,
                )
                for _ in self.side_adapter_layers
            ])
        else:
            self.pre_context_head = NNXPreContextFeatureContextHead(
                rngs=rngs,
                feature_dim=model_dim,
                action_token_dim=model_dim,
                text_dim=text_dim,
                context_tokens=pre_context_tokens,
                heads=pre_context_heads,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
            )

    def predict_pre_context(self, features: jax.Array, actions: jax.Array) -> jax.Array:
        if self.pre_context_head is None:
            raise RuntimeError("pre_context_head is not initialized")
        action_tokens = self.action_encoder(actions).astype(features.dtype)
        return self.pre_context_head(features, action_tokens)


def _build_per_token_timestep(timesteps: jax.Array, f_lat: int, h_lat: int, w_lat: int, n_hist: int) -> jax.Array:
    b = timesteps.shape[0]
    tokens_per_frame = (h_lat // 2) * (w_lat // 2)
    seq_len = f_lat * tokens_per_frame
    n_hist_tokens = n_hist * tokens_per_frame
    full = jnp.broadcast_to(timesteps[:, None], (b, seq_len))
    is_future = jnp.arange(seq_len)[None, :] >= n_hist_tokens
    return jnp.where(is_future, full, jnp.zeros_like(full))


def apply_first_frame_pin(latents: jax.Array, z_i0: jax.Array) -> jax.Array:
    """Keep latent frame 0 equal to the cached image-conditioning latent."""
    if z_i0.shape[2] == 1:
        anchor = z_i0
    else:
        anchor = z_i0[:, :, :1]
    return latents.at[:, :, :1].set(anchor)


def _patchify_and_time_embed(
    transformer,
    hidden_states: jax.Array,
    timestep: jax.Array,
    frame_positions: Optional[tuple[int, ...]] = None,
):
    """Run WAN patch embedding and timestep embedding without text context."""
    hidden_states = nn.with_logical_constraint(hidden_states, ("batch", None, None, None, None))
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = transformer.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = jnp.transpose(hidden_states, (0, 2, 3, 4, 1))
    with transformer.conditional_named_scope("rotary_embedding"):
        rotary_emb = transformer.rope(hidden_states, frame_positions=frame_positions)
    with transformer.conditional_named_scope("patch_embedding"):
        hidden_states = transformer.patch_embedding(hidden_states)
        hidden_states = jax.lax.collapse(hidden_states, 1, -1)

    per_token_t = timestep.ndim == 2
    with transformer.conditional_named_scope("condition_embedder"):
        if per_token_t:
            bt, sl = timestep.shape
            t_flat = timestep.reshape(-1)
            t_sinusoidal = transformer.condition_embedder.timesteps_proj(t_flat)
            t_sinusoidal = t_sinusoidal.reshape(bt, sl, -1)
            temb = transformer.condition_embedder.time_embedder(t_sinusoidal)
            with jax.named_scope("time_proj"):
                timestep_proj = transformer.condition_embedder.time_proj(transformer.condition_embedder.act_fn(temb))
            timestep_proj = timestep_proj.reshape(bt, sl, 6, -1)
        else:
            t_sinusoidal = transformer.condition_embedder.timesteps_proj(timestep)
            temb = transformer.condition_embedder.time_embedder(t_sinusoidal)
            with jax.named_scope("time_proj"):
                timestep_proj = transformer.condition_embedder.time_proj(transformer.condition_embedder.act_fn(temb))
            timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)

    shape_info = (batch_size, num_frames, height, width, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w)
    return hidden_states, rotary_emb, temb, timestep_proj, per_token_t, shape_info


def _embed_text_condition(
    transformer,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
    temb: Optional[jax.Array] = None,
    timestep_proj: Optional[jax.Array] = None,
    per_token_t: Optional[bool] = None,
):
    """Embed timestep and text context for the normal WAN block stack."""
    per_token_t = timestep.ndim == 2 if per_token_t is None else per_token_t
    with transformer.conditional_named_scope("condition_embedder"):
        if per_token_t:
            if temb is None or timestep_proj is None:
                bt, sl = timestep.shape
                t_flat = timestep.reshape(-1)
                t_sinusoidal = transformer.condition_embedder.timesteps_proj(t_flat)
                t_sinusoidal = t_sinusoidal.reshape(bt, sl, -1)
                temb = transformer.condition_embedder.time_embedder(t_sinusoidal)
                with jax.named_scope("time_proj"):
                    timestep_proj = transformer.condition_embedder.time_proj(transformer.condition_embedder.act_fn(temb))
                timestep_proj = timestep_proj.reshape(bt, sl, 6, -1)
            encoder_hidden_states = transformer.condition_embedder.text_embedder(encoder_hidden_states)
            encoder_attention_mask = None
        else:
            temb, timestep_proj, encoder_hidden_states, _, encoder_attention_mask = transformer.condition_embedder(
                timestep, encoder_hidden_states, None
            )
            timestep_proj = timestep_proj.reshape(timestep_proj.shape[0], 6, -1)
    return temb, timestep_proj, encoder_hidden_states, encoder_attention_mask, per_token_t


def _first_block_self_attention_features(
    transformer,
    hidden_states: jax.Array,
    timestep_proj: jax.Array,
    rotary_emb: jax.Array,
    deterministic: bool = True,
    rngs: Optional[nnx.Rngs] = None,
) -> jax.Array:
    """Return first-block features before any cross-attention context enters."""
    block = transformer.blocks[0]
    if timestep_proj.ndim == 4:
        adaln = jnp.expand_dims(block.adaln_scale_shift_table, 0)
        combined = adaln + timestep_proj.astype(jnp.float32)
        parts = jnp.split(combined, 6, axis=2)
        shift_msa = parts[0].squeeze(2)
        scale_msa = parts[1].squeeze(2)
        gate_msa = parts[2].squeeze(2)
    else:
        shift_msa, scale_msa, gate_msa, _, _, _ = jnp.split(
            block.adaln_scale_shift_table + timestep_proj.astype(jnp.float32),
            6,
            axis=1,
        )
    axis_names = nn.logical_to_mesh_axes(("activation_batch", "activation_length", "activation_heads"))
    hidden_states = jax.lax.with_sharding_constraint(hidden_states, axis_names)
    hidden_states = checkpoint_name(hidden_states, "pre_context_hidden_states")
    with block.conditional_named_scope("pre_context_self_attn"):
        norm_hidden_states = (block.norm1(hidden_states.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(
            hidden_states.dtype
        )
        attn_output = block.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            deterministic=deterministic,
            rngs=rngs,
        )
        features = (hidden_states.astype(jnp.float32) + attn_output * gate_msa).astype(hidden_states.dtype)
    return jax.lax.stop_gradient(features)


def _project_wan_output(transformer, hidden_states: jax.Array, temb: jax.Array, per_token_t: bool, shape_info) -> jax.Array:
    batch_size, num_frames, height, width, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w = shape_info
    hidden_states = checkpoint_name(hidden_states, "side_adapter_hidden_states")
    if per_token_t:
        combined_head = jnp.expand_dims(transformer.scale_shift_table, 0) + jnp.expand_dims(temb, 2)
        shift, scale = jnp.split(combined_head, 2, axis=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = jnp.split(transformer.scale_shift_table + jnp.expand_dims(temb, axis=1), 2, axis=1)
    hidden_states = (
        transformer.norm_out(hidden_states.astype(jnp.float32)) * (1 + scale) + shift
    ).astype(hidden_states.dtype)
    with jax.named_scope("proj_out"):
        hidden_states = transformer.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = jnp.transpose(hidden_states, (0, 7, 1, 4, 2, 5, 3, 6))
    return hidden_states.reshape(batch_size, -1, num_frames, height, width)


def wan_side_adapter_forward(
    transformer,
    adapters: NNXWanSideAdapterStack,
    hidden_states: jax.Array,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
    actions: jax.Array,
    deterministic: bool = True,
    rngs: Optional[nnx.Rngs] = None,
    frame_positions: Optional[tuple[int, ...]] = None,
) -> jax.Array:
    """WAN transformer forward with trainable side-adapter residuals."""
    if transformer.scan_layers:
        raise ValueError("WAN side-adapter training currently requires scan_layers=False")

    hidden_states, rotary_emb, temb, timestep_proj, per_token_t, shape_info = _patchify_and_time_embed(
        transformer,
        hidden_states,
        timestep,
        frame_positions=frame_positions,
    )
    temb, timestep_proj, encoder_hidden_states, encoder_attention_mask, per_token_t = _embed_text_condition(
        transformer,
        timestep,
        encoder_hidden_states,
        temb=temb,
        timestep_proj=timestep_proj,
        per_token_t=per_token_t,
    )

    encoder_hidden_states = encoder_hidden_states.astype(hidden_states.dtype)
    action_tokens = adapters.action_encoder(actions).astype(hidden_states.dtype)

    adapter_cursor = 0
    for layer_idx, block in enumerate(transformer.blocks):

        def layer_forward(h):
            return block(
                h,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                deterministic,
                rngs,
                encoder_attention_mask=encoder_attention_mask,
                frame_level_cond=False,
            )

        rematted_layer_forward = transformer.gradient_checkpoint.apply(
            layer_forward,
            transformer.names_which_can_be_saved,
            transformer.names_which_can_be_offloaded,
            prevent_cse=True,
        )
        hidden_states = rematted_layer_forward(hidden_states)
        if adapter_cursor < len(adapters.side_adapter_layers) and adapters.side_adapter_layers[adapter_cursor] == layer_idx:
            hidden_states = hidden_states + adapters.side_adapters[adapter_cursor](hidden_states, action_tokens)
            adapter_cursor += 1

    return _project_wan_output(transformer, hidden_states, temb, per_token_t, shape_info)


def wan_pre_context_adapter_forward(
    transformer,
    adapters: NNXWanSideAdapterStack,
    hidden_states: jax.Array,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
    actions: jax.Array,
    deterministic: bool = True,
    rngs: Optional[nnx.Rngs] = None,
    frame_positions: Optional[tuple[int, ...]] = None,
) -> jax.Array:
    """Predict context from context-free first-block features, then run WAN."""
    del encoder_hidden_states
    if transformer.scan_layers:
        raise ValueError("WAN pre-context adapter training currently requires scan_layers=False")

    tokens, rotary_emb, _, timestep_proj, _, _ = _patchify_and_time_embed(
        transformer,
        hidden_states,
        timestep,
        frame_positions=frame_positions,
    )
    features = _first_block_self_attention_features(
        transformer,
        tokens,
        timestep_proj,
        rotary_emb,
        deterministic=True,
        rngs=None,
    )
    predicted_context = adapters.predict_pre_context(features, actions).astype(features.dtype)
    return transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=predicted_context,
        deterministic=deterministic,
        rngs=rngs,
    )


def wan_action_adapter_forward(
    transformer,
    adapters: NNXWanSideAdapterStack,
    hidden_states: jax.Array,
    timestep: jax.Array,
    encoder_hidden_states: jax.Array,
    actions: jax.Array,
    deterministic: bool = True,
    rngs: Optional[nnx.Rngs] = None,
    frame_positions: Optional[tuple[int, ...]] = None,
) -> jax.Array:
    """Route to the configured action-adapter implementation."""
    if getattr(adapters, "action_adapter_type", "side_adapter") == "pre_context":
        return wan_pre_context_adapter_forward(
            transformer,
            adapters,
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            actions=actions,
            deterministic=deterministic,
            rngs=rngs,
            frame_positions=frame_positions,
        )
    return wan_side_adapter_forward(
        transformer,
        adapters,
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        actions=actions,
        deterministic=deterministic,
        rngs=rngs,
        frame_positions=frame_positions,
    )


def build_rollout_sigmas(num_inference_steps: int, shift: float, sigma_min: float, sigma_max: float) -> jax.Array:
    """Build the N+1 descending sigma grid used by side-adapter train/inference."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    sigmas = jnp.linspace(sigma_max, sigma_min, num_inference_steps + 1, dtype=jnp.float32)[:-1]
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    return jnp.concatenate([sigmas, jnp.zeros((1,), dtype=jnp.float32)], axis=0)


def rollout_timesteps_from_sigmas(sigmas: jax.Array, num_train_timesteps: int) -> jax.Array:
    return sigmas[:-1] * jnp.asarray(num_train_timesteps, dtype=jnp.float32)


def adapter_param_count(params) -> int:
    return sum(int(math.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(params) if hasattr(leaf, "shape"))
