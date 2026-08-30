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
* ``NNXWanSkeletonAdaLNEmbed`` (``skeleton_adaln``) — same skeleton latents,
  patch-embedded into per-token AdaLN conditioning instead of a video-token bias,
  so the signal re-modulates every block rather than being injected once. Shares
  ``WanModel``'s ``action_hidden_states`` argument with the vector-action
  ``adaln`` route, so "adaln" means one site regardless of representation. Also
  does not use the vector actions.

The two axes these modules span are independent: an action *representation*
(vector actions vs. rendered-skeleton latents) and a *conditioning site*
(cross-attention K/V, AdaLN modulation, or additive in video-token space).
``action_cond_mode`` currently names four of the six combinations.

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


class NNXWanSkeletonAdaLNEmbed(nnx.Module):
    """Patch-embeds 2D-kinematic-skeleton latents into AdaLN conditioning.

    Same conditioning signal as ``NNXWanSkeletonPatchEmbed``, injected at a
    different site. That module adds the skeleton to the video tokens once,
    right after the patch embedding, and relies on the residual stream to carry
    it through the remaining blocks. This one hands it to ``WanModel``'s
    ``action_hidden_states``, which sums it into the per-token time embedding
    before the shared ``time_proj`` MLP expands that into the 6-way modulation
    vector — so it re-modulates the activations in *every* block.

    That is deliberately the **same site the vector-action ``adaln`` route uses**
    (``NNXWanActionAdaLNProjector`` → ``action_hidden_states``), and the reason
    is comparability rather than mechanism. The two axes here are meant to be
    independent — an action *representation* (vector actions vs. rendered-skeleton
    latents) crossed with a conditioning *site* — so "adaln" has to mean one
    thing in both cells or a cross-representation comparison confounds the
    representation with the wiring. Sharing ``action_hidden_states`` makes that
    structural: there is no second AdaLN path to drift out of sync.

    What the skeleton does *not* need, which the action route does: a projection
    down to one vector per latent frame, and the ``jnp.repeat`` across that
    frame's spatial patches. Those exist because a 7-dim EEF vector has no
    spatial extent. A skeleton's token grid is already per-token and already
    ``inner_dim`` wide, and it lines up with the modulation grid token for token
    exactly as it lines up with the video token grid in the additive route — the
    transformer's per-token AdaLN (built for WAN 2.2 TI2V's per-token timesteps)
    is spatially varying to begin with. So spatial and camera alignment stay
    structural, and no positional or camera embedding is involved.

    Cross-attention also stays free, so the full T5 instruction sequence can be
    the cross-attention context (see ``_route_action_conditioning``) rather than
    a pooled bias.

    The cost of this site is real, but it is *shared* with the vector-action
    ``adaln`` route rather than specific to the skeleton: summing into ``temb``
    puts the control signal in the same representation the model reads as "how
    noisy is this token", and reaches all six modulation components including
    scale and gate, which act multiplicatively in every block. That applies
    identically to both representations, and ``train_ac_wan_adaln.sh`` already
    trains at this site — the best evidence available that it is tolerable.

    Only the *shape* of the perturbation differs, and not obviously for the
    worse. The per-token timestep varies across frames but is constant across the
    patches within one, so the action route's per-frame-repeated vector lands in
    the same subspace a timestep change occupies — maximally confusable with
    "this frame is at a different noise level". A skeleton's spatially varying
    offset is a signature the timestep can never produce, so it is linearly
    separable from the time signal, at the cost of being out-of-distribution for
    ``time_proj``'s pretrained kernel. Two different risks, not a ranking.

    Watch the per-token modulation statistics either way. If this destabilises,
    the escape hatch is a separate projection into modulation space
    (post-``time_proj``, shift slots only) — at the price of the two routes no
    longer sharing a site, and so of "adaln" naming two things again.

    Note this route has no scale factor, matching the vector ``adaln`` route,
    which has none either (``NNXWanActionAdaLNProjector`` applies no alpha and
    neither does ``_route_action_conditioning``'s adaln branch).

    There is deliberately no ``alpha``. ``NNXWanSkeletonPatchEmbed`` carries one
    inherited from OSCAR, which xavier-inits its conv and needs a small fixed
    scale to stay near the pretrained operating point; with a zero-init kernel
    that scale is redundant for its stated purpose (the injection is *exactly*
    zero at step 0 either way) and all it still does is throttle this path's
    effective learning rate by ``alpha**2`` — a hyperparameter in disguise. The
    zero init below gives the exact no-op on its own, and the learning rate is
    left to be the learning rate.

    Args:
        rngs:          NNX Rngs for parameter initialisation.
        in_channels:   Latent channels of the skeleton video (48 for WAN 2.2's VAE).
        inner_dim:     Transformer width — ``num_attention_heads * attention_head_dim``.
                       Must come from the *loaded* transformer's registered config,
                       not the top-level yaml (those fields are stale for this pipeline).
        patch_size:    The transformer's ``(p_t, p_h, p_w)`` patch size, so the token
                       grid matches ``patch_embedding``'s exactly.
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
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
    ):
        self.dtype = dtype
        self.inner_dim = int(inner_dim)
        self.patch_size = tuple(patch_size)
        # One conv straight from the patch to inner_dim, deliberately matching
        # NNXWanSkeletonPatchEmbed's single-linear-map structure rather than the
        # 3-layer MLP the vector-action encoder needs. A patch is only
        # in_channels * prod(patch_size) = 192 dims here, so a wider or deeper
        # head adds capacity but no information — and keeping the two skeleton
        # routes structurally identical means an additive-vs-adaln comparison
        # varies the injection site alone, not the adapter.
        self.proj = nnx.Conv(
            in_channels,
            self.inner_dim,
            rngs=rngs,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            # Zero-init: a freshly built model is *exactly* the no-skeleton
            # baseline at step 0, so training starts from the pretrained
            # operating point. No deadlock risk (cf. NNXWanActionAdaLNProjector,
            # where a zero-init encoder feeding a zero-init projector starves
            # both of gradient): this is the only module on the route and its
            # input is nonzero *data*, so d(loss)/d(kernel) is nonzero on the
            # very first step.
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                (None, None, None, None, "conv_out"),
            ),
        )

    def __call__(self, skeleton_latents: jax.Array) -> jax.Array:
        """``(B, C, F_lat, H_lat, W_lat)`` skeleton latents → ``(B, seq_len, inner_dim)``.

        Layout matches ``WanModel.__call__``'s own patch-embedding path
        (channels-last transpose → conv → collapse), so ``seq_len`` and its token
        ordering are identical to the video tokens' — which is what lets the
        result be summed into the per-token time embedding with no regridding,
        and why this route needs no spatial repeat.
        """
        x = jnp.transpose(skeleton_latents, (0, 2, 3, 4, 1)).astype(self.dtype)
        x = self.proj(x)                          # (B, F, H/p_h, W/p_w, inner_dim)
        return jax.lax.collapse(x, 1, -1)         # (B, seq_len, inner_dim)


class NNXWanSkeletonCrossAttnEmbed(nnx.Module):
    """Patch-embeds 2D-kinematic-skeleton latents into the cross-attention K/V.

    Third skeleton route, completing the representation x site grid: the same
    rendered-skeleton latents ``NNXWanSkeletonPatchEmbed`` (additive, video-token
    space) and ``NNXWanSkeletonAdaLNEmbed`` (AdaLN, ``temb``) consume, injected
    instead at the site the vector-action ``cross_attn`` route uses.

    Output width is ``wan_text_dim`` (4096), NOT ``inner_dim``, because
    ``WanModel.__call__`` pushes ``encoder_hidden_states`` through the pretrained
    ``condition_embedder.text_embedder`` before the blocks see it. Emitting 4096
    means the skeleton reaches cross-attention through the exact path the action
    tokens do, so "cross_attn" names one site across both representations and a
    cross-representation comparison is not confounded by the wiring — the same
    argument ``NNXWanSkeletonAdaLNEmbed`` makes for sharing ``action_hidden_states``
    with the vector ``adaln`` route.

    Alignment
    ---------
    Unlike the other two routes, this one does NOT get token-for-token alignment
    for free: softmax over keys is permutation-invariant, so nothing in a bare
    key says which grid cell it came from. Two mechanisms restore it, and both
    live outside this module:

    * Temporal, structural: the caller sets ``frame_level_cond=True`` with
      ``cond_tokens_per_frame = (H_lat//p_h) * (W_lat//p_w)``, so
      ``WanTransformerBlock`` folds F into the batch and latent frame k's video
      tokens attend only to latent frame k's skeleton tokens. Same machinery the
      vector ``cross_attn`` route uses, with the per-frame key count raised from
      a handful of action tokens to a full spatial grid.
    * Spatial, structural: ``cross_attn_rope=True`` makes the block hand
      ``attn2`` the same 3D rotary embedding ``attn1`` already uses, sliced to
      this frame. Q and K then carry identical phases at identical grid
      positions, so the logit is maximised on the diagonal with no parameters
      and no learned lookup. The shared temporal phase cancels in the relative
      rotation, leaving a purely spatial offset.

    So this module is deliberately just the patch embedding — structurally the
    same single conv as its two siblings, differing only in output width. Keeping
    the adapter identical across all three means a site comparison varies the
    site alone.

    Step 0
    ------
    Zero-init kernel, as in both siblings. The tokens are then exactly zero,
    which is precisely the all-zero cross-attention context ``adaln``,
    ``skeleton`` and ``skeleton_adaln`` already feed (and the CFG-uncond state),
    so a freshly built model reproduces that baseline exactly rather than being
    knocked off it by random conditioning.

    Note the gradient sequencing this creates, which is benign but not instant.
    With every key identical (all zero) the softmax is uniform AND every value is
    the same vector, so ``d(loss)/d(query)`` and ``d(loss)/d(key)`` are exactly
    zero on step 0 — the K=1 degeneracy's milder cousin, arising here from
    identical keys rather than a single key. This conv still gets gradient
    immediately, through ``to_v``'s normally-initialised kernel, so the tokens
    become distinct after one update and ``to_q``/``to_k`` receive gradient from
    step 2 onward. That is a one-step delay, not the permanent starvation
    ``NNXWanActionAdaLNProjector`` guards against, because nothing here is
    zero-init downstream of another zero-init module.

    Args:
        rngs:          NNX Rngs for parameter initialisation.
        in_channels:   Latent channels of the skeleton video (48 for WAN 2.2's VAE).
        wan_text_dim:  Cross-attention context width the transformer expects
                       *before* ``text_embedder`` (4096), matching what
                       ``NNXWanActionEncoder`` emits.
        patch_size:    The transformer's ``(p_t, p_h, p_w)`` patch size, so the
                       skeleton token grid matches ``patch_embedding``'s exactly.
                       Required for the frame-locked reshape and for RoPE to line
                       Q and K up cell for cell.
        dtype:         Activation dtype.
        weights_dtype: Parameter storage dtype.
        precision:     Matmul precision, threaded through to the conv.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        in_channels: int,
        wan_text_dim: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.Precision = None,
    ):
        self.dtype = dtype
        self.wan_text_dim = int(wan_text_dim)
        self.patch_size = tuple(patch_size)
        self.proj = nnx.Conv(
            in_channels,
            self.wan_text_dim,
            rngs=rngs,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            # Zero-init — see the class docstring. No deadlock: the input is
            # nonzero *data*, and the only zero-init module on this route is
            # this one.
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                (None, None, None, None, "conv_out"),
            ),
        )

    def __call__(self, skeleton_latents: jax.Array) -> jax.Array:
        """``(B, C, F_lat, H_lat, W_lat)`` skeleton latents → ``(B, seq_len, wan_text_dim)``.

        Same channels-last transpose → conv → collapse as both siblings, so the
        token ordering is frame-major and identical to the video tokens'. That
        ordering is what makes the caller's ``(B*F_lat, tokens_per_frame, D)``
        reshape line the two grids up frame for frame and cell for cell.
        """
        x = jnp.transpose(skeleton_latents, (0, 2, 3, 4, 1)).astype(self.dtype)
        x = self.proj(x)                          # (B, F, H/p_h, W/p_w, wan_text_dim)
        return jax.lax.collapse(x, 1, -1)         # (B, seq_len, wan_text_dim)
