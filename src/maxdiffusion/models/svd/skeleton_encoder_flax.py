# Copyright 2026 Princeton. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Skeleton conditioning encoders for the action-conditioned SVD world model.

SVD counterpart of ``models/wan/action_encoder_wan.py``'s three skeleton
modules. The conditioning signal is identical in kind — a rendered
2D-kinematic-skeleton video pushed through the *same* VAE as the RGB video, so
its latents are element-for-element aligned with the video latents — and the
three modules differ only in where that signal is injected:

    skeleton              -> added onto conv_in's output      (FlaxSkeletonPatchEmbed)
    skeleton_adaln        -> summed into t_emb                (FlaxSkeletonAdaLNProjector)
    skeleton_cross_attn   -> the spatial cross-attention K/V  (FlaxSkeletonCrossAttnEmbed)

At 192x320 per camera with 3 cameras stacked along H and SVD's 8x VAE, a
skeleton latent is ``(T, 4, 72, 40)`` — matching ``latent_stacked``.

Three places where the SVD port is NOT a transliteration of the WAN one, all
forced by architecture rather than choice:

1. There is no patch embedding. WAN is a DiT and adds the skeleton onto
   patchified tokens; SVD is a conv UNet, so the structural analogue of "inject
   once, before the blocks, and let the residual stream carry it" is a second
   conv added onto ``conv_in``'s output at full latent resolution.

2. SVD's AdaLN site is spatially blind. ``t_emb`` is ``(B*T, time_embed_dim)``
   — one vector per (sample, frame), with no spatial axis — where WAN 2.2 TI2V's
   per-token timestep gave a per-token modulation grid. So the skeleton MUST be
   pooled to one vector per frame here, discarding exactly the spatial structure
   that makes a skeleton more informative than a 7-dim action. This route is
   therefore strictly weaker in SVD than in WAN, and that is a property of the
   site, not of this code: see ``FlaxSkeletonAdaLNProjector``.

3. There are no rotary embeddings anywhere in SVD, so the WAN cross-attention
   route's parameter-free RoPE alignment has nothing to reuse. Alignment is
   instead a learned positional embedding on the keys — see
   ``FlaxSkeletonCrossAttnEmbed``.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


def _kaiming_normal_relu_fan_in():
    """Matches ``nn.init.kaiming_normal_(..., nonlinearity='relu')``; see
    ``action_encoder_flax._kaiming_normal_relu_fan_in``."""
    return nn.initializers.he_normal(in_axis=-2, out_axis=-1, dtype=jnp.float32)


def _to_nhwc(skeleton: jnp.ndarray) -> jnp.ndarray:
    """``(B, T, C, H, W)`` or ``(B*T, C, H, W)`` -> channels-last, batch-flattened.

    Returns ``(B*T, H, W, C)``. The UNet flattens (sample, frame) into the batch
    axis before ``conv_in``, so every skeleton route works in that same flattened
    space and the caller never has to track which convention it is in.
    """
    if skeleton.ndim == 5:
        b, t, c, h, w = skeleton.shape
        skeleton = skeleton.reshape(b * t, c, h, w)
    elif skeleton.ndim != 4:
        raise ValueError(
            f"skeleton latents must be (B, T, C, H, W) or (B*T, C, H, W), got {skeleton.shape}"
        )
    return jnp.transpose(skeleton, (0, 2, 3, 1))


class FlaxSkeletonPatchEmbed(nn.Module):
    """Additive route: skeleton latents -> a bias on ``conv_in``'s output.

    ``(B*T, 4, 72, 40)`` -> ``(B*T, 72, 40, model_channels)``, added inside
    ``FlaxVideoUNet.__call__`` immediately after ``conv_in`` and scaled by
    ``alpha``::

        sample = conv_in(latents) + alpha * skeleton_embed(skeleton_latents)

    This is the SVD analogue of OSCAR's ``addition_patch_embedding`` (and of
    ``NNXWanSkeletonPatchEmbed``): a *separate* convolution, never the
    transformer's own input projection, whose output is added in feature space.
    Injecting here rather than channel-concatenating into ``conv_in`` matters for
    two reasons — concatenation would change ``conv_in``'s ``in_channels`` and so
    invalidate the pretrained kernel, and contaminating the noisy latent's own
    channels would make the regression target unrecoverable from the input.

    Kernel is zero-init, so a freshly built model is *exactly* the no-skeleton
    baseline at step 0 and training starts from the pretrained operating point.
    No deadlock risk (cf. ``FlaxActionAdaLNProjector``, where a zero-init encoder
    feeding a zero-init projector starves both): the input here is nonzero
    *data*, so d(loss)/d(kernel) is nonzero on the very first step.

    Args:
        model_channels: ``block_out_channels[0]`` of the UNet (320 for SVD) —
                        conv_in's output width, which this must match to be added.
        alpha:          Fixed scale on the injected bias, as in the WAN route and
                        OSCAR. Also scales the gradient into this kernel, so it
                        caps how fast the skeleton path ramps up out of zero.
    """

    model_channels: int = 320
    alpha: float = 0.1
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, skeleton: jnp.ndarray) -> jnp.ndarray:
        x = _to_nhwc(skeleton).astype(self.dtype)
        x = nn.Conv(
            self.model_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            # Zero-init: exact no-op at step 0. See the class docstring.
            kernel_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            name="proj",
        )(x)
        return self.alpha * x

    def init_weights(self, rng, batch=1, num_frames=1, channels=4, height=72, width=40):
        x = jnp.zeros((batch * num_frames, channels, height, width), dtype=self.dtype)
        return self.init({"params": rng}, x)["params"]


class FlaxSkeletonAdaLNProjector(nn.Module):
    """AdaLN route: skeleton latents -> one ``time_embed_dim`` vector per frame.

    ``(B*T, 4, 72, 40)`` -> ``(B*T, time_embed_dim)``, summed into the UNet's
    ``t_emb`` by the caller, exactly where the vector-action ``adaln`` route puts
    ``FlaxActionAdaLNProjector``'s output. Sharing that site is deliberate: the
    two axes are meant to be an action *representation* crossed with a
    conditioning *site*, so "adaln" has to mean one thing in both cells or a
    skeleton-vs-vector comparison confounds representation with wiring.

    SPATIALLY BLIND, and unavoidably so. SVD's ``t_emb`` is
    ``(B*T, time_embed_dim)`` — one vector per (sample, frame) — and
    ``FlaxVideoUNet`` asserts that shape. WAN 2.2 TI2V's per-token timestep gave
    a modulation grid that a skeleton could line up with token for token, so the
    WAN route needed neither projection nor pooling. Here there is no spatial
    axis to land in, so the grid must be collapsed. That throws away precisely
    what distinguishes a rendered skeleton from a 7-dim action vector, which
    makes this the weakest of the three SVD routes a priori — worth stating
    plainly rather than discovering from the loss curves. It is still worth
    training: it is the one cell that isolates "does the *site* matter" against
    the vector adaln run, holding the site fixed.

    Pooling is a strided conv stack (learned downsample, so the collapse is
    weighted rather than a flat average) followed by a global mean and a dense
    projection. The final dense is zero-init, giving the exact step-0 no-op; the
    convolutions before it are Kaiming, so there is no zero-init-in-series
    deadlock — gradient reaches them through the dense kernel once it moves, and
    the dense kernel itself sees nonzero input from step 0.
    """

    time_embed_dim: int = 1280
    hidden_channels: int = 128
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, skeleton: jnp.ndarray) -> jnp.ndarray:
        x = _to_nhwc(skeleton).astype(self.dtype)
        for i, ch in enumerate((self.hidden_channels, self.hidden_channels * 2)):
            x = nn.Conv(
                ch,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                kernel_init=_kaiming_normal_relu_fan_in(),
                dtype=self.dtype,
                param_dtype=self.weights_dtype,
                name=f"down_{i}",
            )(x)
            x = nn.silu(x)
        x = jnp.mean(x, axis=(1, 2))                      # (B*T, hidden*2)
        return nn.Dense(
            self.time_embed_dim,
            # Zero-init the output so step 0 leaves t_emb untouched — the exact
            # no-skeleton baseline. The convs above are NOT zero, so this layer
            # has nonzero input and the convs get gradient through its kernel as
            # soon as it moves; no deadlock.
            kernel_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            name="proj",
        )(x)

    def init_weights(self, rng, batch=1, num_frames=1, channels=4, height=72, width=40):
        x = jnp.zeros((batch * num_frames, channels, height, width), dtype=self.dtype)
        return self.init({"params": rng}, x)["params"]


class FlaxSkeletonCrossAttnEmbed(nn.Module):
    """Cross-attention route: skeleton latents -> a per-frame K/V token grid.

    ``(B*T, 4, 72, 40)`` -> ``(B*T, S, hidden_size)`` with ``S =
    (72//stride) * (40//stride)`` — 180 tokens per frame at the default stride 4,
    which is exactly the per-frame key count the WAN route uses at this
    resolution, so the two arms are matched on keys as well as on site.

    Frame locking is free here. SVD's cross-attention context is already
    ``(B*T, S, C)`` — (sample, frame) is folded into the batch axis before the
    blocks — so frame k's patches can only ever see frame k's keys. WAN needed an
    explicit ``frame_level_cond`` reshape to get the same property.

    ALIGNMENT. Softmax over keys is permutation-invariant, so a bare key says
    nothing about which grid cell it came from — the additive and adaln routes
    get that correspondence for free by being elementwise, attention does not.
    WAN restores it with the transformer's existing 3D RoPE, at zero parameter
    cost. SVD has no rotary embeddings anywhere, so there is nothing to reuse and
    the alignment is instead a learned absolute position embedding added to the
    keys. It is zero-init, which keeps the exact step-0 no-op (all keys are then
    identically zero) while still receiving gradient through ``to_k``/``to_v``
    from the first step.

    Note the queries are a *different* grid size at every UNet stage (72x40,
    36x20, 18x10 for the three cross-attention resolutions), while these keys are
    one fixed grid shared by all of them. That is fine — cross-attention does not
    require q_len == kv_len — but it does mean the correspondence the model has
    to learn is a resampling, not an identity, at two of the three stages. Which
    is the honest cost of not having RoPE.

    Args:
        hidden_size: The UNet's ``context_dim`` (1024 for SVD). Emitting this
                     width directly is right for SVD: unlike WAN, there is no
                     pretrained text projection between the caller and the
                     blocks' ``to_k``/``to_v``, so no 4096-dim detour.
        stride:      Spatial downsample of the key grid. 4 -> 180 keys/frame.
                     Raising it cuts cross-attention cost quadratically in the
                     key count; lowering it to 2 gives 720 keys/frame, which at
                     the shallowest stage is ~25% of a self-attention.
    """

    hidden_size: int = 1024
    stride: int = 4
    latent_height: int = 72
    latent_width: int = 40
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32

    @property
    def num_tokens(self) -> int:
        return (self.latent_height // self.stride) * (self.latent_width // self.stride)

    @nn.compact
    def __call__(self, skeleton: jnp.ndarray) -> jnp.ndarray:
        x = _to_nhwc(skeleton).astype(self.dtype)
        x = nn.Conv(
            self.hidden_size,
            kernel_size=(self.stride, self.stride),
            strides=(self.stride, self.stride),
            padding="VALID",
            # Zero-init -> keys are exactly zero at step 0, which is the same
            # all-zero cross-attention context the other modes feed (and the
            # CFG-uncond state), so the model starts at the pretrained point.
            kernel_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            name="proj",
        )(x)                                              # (B*T, H', W', hidden)
        bt, hp, wp, c = x.shape
        x = x.reshape(bt, hp * wp, c)                     # row-major: h-major, w-minor
        # Learned absolute position on the keys — the stand-in for WAN's RoPE.
        # Zero-init so it does not break the step-0 no-op; it still gets gradient
        # via to_k/to_v immediately.
        pos = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (hp * wp, c),
            self.weights_dtype,
        )
        return x + pos.astype(x.dtype)[None]

    def init_weights(self, rng, batch=1, num_frames=1, channels=4, height=None, width=None):
        h = self.latent_height if height is None else height
        w = self.latent_width if width is None else width
        x = jnp.zeros((batch * num_frames, channels, h, w), dtype=self.dtype)
        return self.init({"params": rng}, x)["params"]
