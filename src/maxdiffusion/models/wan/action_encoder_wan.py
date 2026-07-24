"""Action encoder for action-conditioned WAN video generation.

Per-latent-frame 3-layer SiLU MLP encoding the 4 raw-frame actions that
correspond to one latent frame into the WAN cross-attention space.

``tokens_per_frame`` controls the granularity:

* ``tokens_per_frame=1`` (legacy): the 4 raw actions are concatenated into a
  single 28-dim vector and encoded as ONE token per latent frame. With a
  single K/V token, per-frame cross-attention softmax is identically 1 — the
  conditioning is a query-independent additive injection.
* ``tokens_per_frame=4``: each raw action is encoded as its OWN token
  (4 tokens per latent frame). A learned slot embedding distinguishes the 4
  temporal positions. Cross-attention now runs over 4 keys, so attention
  weights become query/content-dependent and the transformer's pretrained
  Q/K cross-attention weights participate in training.

    action (B, F_lat, 4, action_dim) → (B, F_lat * tokens_per_frame, out_dim)

The output tokens are used as K/V in the WAN transformer's per-frame
cross-attention (frame_level_cond=True, cond_tokens_per_frame=tokens_per_frame),
where the image latent's spatial patches are the queries.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class NNXWanActionEncoder(nnx.Module):
    """3-layer SiLU MLP encoding raw-frame actions into per-latent-frame tokens.

    Args:
        rngs:             NNX Rngs for parameter initialisation.
        action_dim:       Width of a single raw-frame action vector (7 for DROID EEF).
        num_actions:      Number of raw frames per latent frame (4 for WAN 4× temporal compression).
        hidden_dim:       Width of hidden layers.
        out_dim:          Output width — must equal wan_text_dim (4096).
        tokens_per_frame: Output tokens per latent frame. Must divide num_actions.
                          1 = concat all actions into one token (legacy);
                          num_actions = one token per raw action.
        dtype:            Activation dtype.
        weights_dtype:    Parameter storage dtype.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        action_dim: int = 7,
        num_actions: int = 4,
        hidden_dim: int = 1024,
        out_dim: int = 4096,
        tokens_per_frame: int = 1,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
    ):
        if num_actions % tokens_per_frame != 0:
            raise ValueError(
                f"tokens_per_frame ({tokens_per_frame}) must divide num_actions ({num_actions})"
            )
        self.dtype = dtype
        self.tokens_per_frame = tokens_per_frame
        self.actions_per_token = num_actions // tokens_per_frame
        kaiming = nnx.initializers.he_normal()

        self.linear_1 = nnx.Linear(
            in_features=action_dim * self.actions_per_token,  # 28 for 1 token, 7 for 4 tokens
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            kernel_init=kaiming,
        )
        self.linear_2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            kernel_init=kaiming,
        )
        self.linear_3 = nnx.Linear(
            in_features=hidden_dim,
            out_features=out_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            # Zero-init the output projection so the encoder starts as a no-op:
            # action tokens are all-zero at step 0, so the pretrained cross-attention
            # contributes ~nothing and training begins at the pretrained operating
            # point (a true finetune) instead of being knocked off it by random
            # conditioning — this removes the large initial loss + exponential drop.
            # Bias defaults to zeros in nnx.Linear, so the output is exactly 0.
            kernel_init=nnx.initializers.zeros,
        )
        # Learned slot embedding: attention over keys is permutation-invariant,
        # so without this the model couldn't tell which of the tokens_per_frame
        # sub-steps a token encodes. Omitted for the single-token (legacy)
        # layout to keep old checkpoints loadable.
        if tokens_per_frame > 1:
            self.slot_embed = nnx.Param(
                jax.random.normal(rngs.params(), (tokens_per_frame, hidden_dim), dtype=weights_dtype)
                * (hidden_dim**-0.5)
            )
        else:
            self.slot_embed = None

    def __call__(
        self,
        action: jax.Array,
        text_embed: jax.Array | None = None,
    ) -> jax.Array:
        """Encode raw-frame actions into per-latent-frame tokens.

        Args:
            action:     ``(B, F_lat, num_actions, action_dim)`` normalised actions.
            text_embed: ``(B, out_dim)`` pooled T5 embedding, broadcast-added
                        as a per-sample bias.

        Returns:
            ``(B, F_lat * tokens_per_frame, out_dim)`` — tokens_per_frame
            action tokens per latent frame, frame-major.
        """
        B, F, T_a, A = action.shape
        K = self.tokens_per_frame
        # Group actions_per_token consecutive raw actions into each token.
        x = action.reshape(B * F * K, self.actions_per_token * A).astype(self.dtype)
        x = jax.nn.silu(self.linear_1(x))  # (B*F*K, hidden_dim)
        if self.slot_embed is not None:
            slot = jnp.tile(self.slot_embed.value.astype(x.dtype), (B * F, 1))
            x = x + slot
        x = jax.nn.silu(self.linear_2(x))  # (B*F*K, hidden_dim)
        x = self.linear_3(x)               # (B*F*K, out_dim)
        if text_embed is not None:
            # Tile (B, out_dim) → (B*F*K, out_dim): repeat each sample F*K times.
            x = x + jnp.repeat(text_embed, F * K, axis=0).astype(x.dtype)
        return x.reshape(B, F * K, -1)     # (B, F_lat * tokens_per_frame, out_dim)


class NNXWanActionAdaLNProjector(nnx.Module):
    """Projects per-latent-frame action tokens into the WAN transformer's AdaLN
    conditioning space, as an alternative to feeding them through cross-attention.

    The ``tokens_per_frame`` action tokens for a given latent frame (each
    ``wan_text_dim`` wide, as produced by ``NNXWanActionEncoder``) are
    concatenated and projected down to ``inner_dim`` with a single linear
    layer, giving one AdaLN vector per latent frame. The caller repeats that
    vector across the frame's spatial patch tokens and adds it to the WAN
    transformer's per-token timestep embedding (see
    ``WanModel.__call__``'s ``action_hidden_states`` argument).

    The init no-op (a freshly-created model identical to the no-action baseline)
    comes from ``NNXWanActionEncoder.linear_3`` being zero-init: the action tokens
    are exactly 0 at init, so this projection outputs 0 regardless of its own
    weights. This kernel is therefore NORMAL-initialised, NOT zero. Zero-init here
    too would be a bug: in AdaLN mode the cross-attention path is zeroed, so
    ``encoder → adaln_proj`` is the only action route, and two zero-inits in series
    deadlock — ``adaln_proj``'s kernel gets no gradient (zero input) and
    ``linear_3`` gets no gradient (through ``adaln_proj``'s zero kernel), so only
    this layer's bias ever moves and the action encoder stays frozen at 0 forever.
    A non-zero kernel here keeps the init no-op (via ``linear_3=0``) while giving
    ``linear_3`` a gradient path so the encoder can actually learn.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        tokens_per_frame: int,
        wan_text_dim: int,
        inner_dim: int,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.dtype = dtype
        self.tokens_per_frame = tokens_per_frame
        self.proj = nnx.Linear(
            in_features=tokens_per_frame * wan_text_dim,
            out_features=inner_dim,
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
            # Kernel is NORMAL-init, not zero (see class docstring: zero-init here
            # deadlocks the action encoder in AdaLN mode). The init no-op comes
            # from linear_3=0 upstream, not from this kernel. Bias stays zero.
            # with_partitioning only attaches FSDP sharding metadata, it doesn't
            # change the init values; it keeps this large kernel
            # (tokens_per_frame*wan_text_dim x inner_dim, e.g. 16384x5120 ≈ 320MB
            # at float32) FSDP-sharded rather than replicated on every device,
            # like every other large linear in the WAN stack (see
            # NNXPixArtAlphaTextProjection).
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
        )

    def __call__(self, action_tokens_grouped: jax.Array) -> jax.Array:
        """``(B, F_lat, tokens_per_frame, wan_text_dim)`` → ``(B, F_lat, inner_dim)``."""
        B, F, K, D = action_tokens_grouped.shape
        x = action_tokens_grouped.reshape(B, F, K * D).astype(self.dtype)
        return self.proj(x)
