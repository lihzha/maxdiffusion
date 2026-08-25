"""Action conditioning modules for action-conditioned WAN video generation.

One module per ``action_cond_mode`` route:

* ``NNXWanActionEncoder`` (``cross_attn``, ``adaln``) — per-latent-frame 3-layer
  SiLU MLP encoding the 4 raw-frame 7-dim actions into the WAN
  cross-attention space.
* ``NNXWanActionAdaLNProjector`` (``adaln``) — projects those action tokens into
  the transformer's per-token AdaLN conditioning space.
* ``NNXWanSkeletonPatchEmbed`` (``skeleton``) — patch-embeds VAE latents of a
  rendered 2D-kinematic-skeleton video and adds them to the video tokens. This
  route does not use the vector actions at all, so ``NNXWanActionEncoder`` is
  not built in that mode.

The action-encoder docs below apply to the first two modes.

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


class NNXWanSkeletonPatchEmbed(nnx.Module):
    """Patch-embeds 2D-kinematic-skeleton latents into the WAN transformer's
    token space, so they can be *added* to the video tokens (OSCAR-style).

    The skeleton conditioning signal is a rendered 2D skeleton video pushed
    through the *same* WAN VAE as the RGB video, so its latents are
    token-for-token aligned with the video latents — identical
    ``(C, F_lat, H_lat, W_lat)`` shape, identical 3-camera H-concat, identical
    normalisation. That alignment is what makes an additive injection
    well-defined at all.

    This mirrors OSCAR's ``addition_patch_embedding`` (see
    ``worldsim/_src/networks/wan2pt1_i2v_concat.py``): a second patch-embedding
    convolution, *separate* from the transformer's pretrained
    ``patch_embedding``, whose output is added to the patchified video tokens
    scaled by ``alpha``::

        h = patch_embedding(video_latents) + alpha * skel_patch_embed(skel_latents)

    Why a separate conv rather than adding the raw latents: the video latents
    are the thing the model has to denoise, so contaminating those channels
    would make the regression target unrecoverable from the input. Injecting in
    token space leaves the noisy latent intact and gives the skeleton its own
    learned read-out.

    Why this lives outside ``WanModel``: ``create_sharded_logical_transformer``
    materialises the transformer's params by *replacing* the ``nnx.eval_shape``
    pytree with whatever ``load_wan_transformer`` found in the pretrained
    safetensors. A submodule added inside ``WanModel`` has no counterpart there,
    so its params would survive as unmaterialised ``ShapeDtypeStruct`` leaves.
    Keeping it in the ``WanCtrlWorldModel`` wrapper — next to
    ``action_encoder`` / ``action_adaln_proj``, which exist for the same reason
    — means it is built with real weights and checkpointed with the rest of the
    combined params. ``WanModel.__call__`` only takes the finished tokens, via
    its ``skeleton_hidden_states`` argument, exactly as it already takes
    ``action_hidden_states``.

    Args:
        rngs:          NNX Rngs for parameter initialisation.
        in_channels:   Latent channels of the skeleton video (48 for WAN 2.2's VAE).
        inner_dim:     Transformer width — ``num_attention_heads * attention_head_dim``.
                       Must come from the *loaded* transformer's registered config,
                       not the top-level yaml (those fields are stale for this pipeline).
        patch_size:    The transformer's ``(p_t, p_h, p_w)`` patch size, so the token
                       grid matches ``patch_embedding``'s exactly.
        alpha:         Fixed scale on the injected tokens. Also scales the gradient
                       into this kernel, so it caps how fast the skeleton path ramps
                       up out of the zero init.
        dtype:         Activation dtype.
        weights_dtype: Parameter storage dtype.
        precision:     Matmul precision, threaded through to the conv.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        in_channels: int,
        inner_dim: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        alpha: float = 0.1,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
    ):
        self.dtype = dtype
        self.alpha = float(alpha)
        self.patch_size = tuple(patch_size)
        self.proj = nnx.Conv(
            in_channels,
            inner_dim,
            rngs=rngs,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            # Zero-init, so a freshly built model is *exactly* the no-skeleton
            # baseline at step 0 and training starts from the pretrained
            # operating point — the same reason NNXWanActionEncoder.linear_3 is
            # zero-init. Unlike NNXWanActionAdaLNProjector (see its docstring)
            # there is no deadlock risk here: the skeleton latents are nonzero
            # *data*, not another zero-init module's output, so
            # d(loss)/d(kernel) = d(loss)/d(token) * skel_latent is nonzero on
            # the very first step. OSCAR xavier-inits this conv and relies on
            # alpha=0.1 alone to stay near the pretrained point; zero-init makes
            # that exact instead of approximate.
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                (None, None, None, None, "conv_out"),
            ),
        )

    def __call__(self, skeleton_latents: jax.Array) -> jax.Array:
        """``(B, C, F_lat, H_lat, W_lat)`` skeleton latents → ``(B, seq_len, inner_dim)``.

        Layout matches ``WanModel.__call__``'s own patch-embedding path
        (channels-last transpose → conv → collapse), so the result can be added
        straight onto the video tokens.
        """
        x = jnp.transpose(skeleton_latents, (0, 2, 3, 4, 1)).astype(self.dtype)
        x = self.proj(x)                          # (B, F, H/p_h, W/p_w, inner_dim)
        x = jax.lax.collapse(x, 1, -1)            # (B, seq_len, inner_dim)
        return self.alpha * x
