# Copyright 2026 Princeton. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Action encoder for the action-conditioned SVD world model.

JAX / Flax port of ``Ctrl-World/models/ctrl_world.py::Action_encoder2``.

Maps a per-frame action sequence ``(B, T, action_dim)`` to the UNet's
cross-attention dimension ``(B, T, hidden_size)`` with a 3-layer SiLU MLP.
Optionally adds a CLIP text-embedding (repeated to match ``hidden_size``) as
a task/language condition — this is the same additive text-conditioning used
by Ctrl-World at training time.

The cross-attention hidden states fed into the UNet are expected to have
shape ``(B*T, 1, hidden_size)`` (one token per frame); the reshape from
``(B, T, hidden_size)`` happens inside :class:`FlaxVideoUNet` when called
with ``frame_level_cond=True``.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


def _kaiming_normal_relu_fan_in():
    """Kaiming-normal initializer matching ``nn.init.kaiming_normal_(..., nonlinearity='relu')``.

    PyTorch's formula is ``std = sqrt(2 / fan_in)``. Flax's
    ``he_normal(in_axis=-2, out_axis=-1)`` is the same distribution for a
    ``(fan_in, fan_out)`` Dense kernel; spelled out here so the intent stays
    visible next to the port.
    """
    return nn.initializers.he_normal(in_axis=-2, out_axis=-1, dtype=jnp.float32)


class FlaxActionEncoder(nn.Module):
    """MLP that lifts a per-frame action sequence to the cross-attn dim.

    Matches Ctrl-World's ``Action_encoder2``:

        Linear(action_dim, hidden_size) -> SiLU
        Linear(hidden_size, hidden_size) -> SiLU
        Linear(hidden_size, hidden_size)

    The first two Dense kernels use Kaiming-normal init
    (``fan_in``, ``relu`` nonlinearity); the output projection is zero-init so
    a fresh encoder is a no-op at step 0 (see ``setup``). Note this diverges
    from upstream Ctrl-World, which leaves the third layer at PyTorch's default
    ``kaiming_uniform_(a=sqrt(5))`` — warm-starting via
    ``action_encoder_init_path`` restores the upstream weights and bypasses
    this init entirely.

    If ``text_embed_dim`` is not None and a ``text_embeds`` argument is
    passed to ``__call__``, the text embedding is repeated to
    ``hidden_size`` (via ``tile`` over the feature axis) and added to the
    action hidden state — this mirrors sgm's trick of concatenating the
    512-dim CLIP text projection with itself to reach 1024 dims.

    Args:
        action_dim: dimension of each per-frame action (7 for DROID EEF).
        hidden_size: output feature dim (1024 for base SVD).
        text_embed_dim: expected feature size of ``text_embeds`` inputs. If
            ``hidden_size`` is not an integer multiple of this, an error is
            raised. Set to ``None`` to disable text conditioning.
    """

    action_dim: int = 7
    hidden_size: int = 1024
    text_embed_dim: Optional[int] = 512
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32

    def setup(self):
        kaiming = _kaiming_normal_relu_fan_in()
        self.linear_1 = nn.Dense(
            self.hidden_size,
            kernel_init=kaiming,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
        )
        self.linear_2 = nn.Dense(
            self.hidden_size,
            kernel_init=kaiming,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
        )
        self.linear_3 = nn.Dense(
            self.hidden_size,
            # Zero-init the output projection so a cold-started encoder emits
            # nothing at step 0: the pretrained UNet's cross-attention sees a
            # zero (or, with text conditioning on, constant) context instead of
            # random conditioning, so training begins at the pretrained
            # operating point rather than being knocked off it. Same rationale
            # as NNXWanActionEncoder.linear_3. Bias already defaults to zeros.
            kernel_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
        )
        if self.text_embed_dim is not None:
            if self.hidden_size % self.text_embed_dim != 0:
                raise ValueError(
                    f"FlaxActionEncoder: hidden_size ({self.hidden_size}) must be a "
                    f"multiple of text_embed_dim ({self.text_embed_dim}) for the "
                    "tile-to-hidden-size trick used by Ctrl-World."
                )

    def __call__(
        self,
        action: jnp.ndarray,
        text_embeds: Optional[jnp.ndarray] = None,
        frame_level_cond: bool = True,
    ) -> jnp.ndarray:
        """Encode a per-frame action sequence.

        Args:
            action: ``(B, T, action_dim)`` per-frame action. When
                ``frame_level_cond=False`` it is flattened to
                ``(B, 1, T*action_dim)`` first — matches the
                non-frame-level branch of Ctrl-World.
            text_embeds: optional ``(B, text_embed_dim)`` pooled CLIP text
                embedding. Ignored if ``text_embed_dim`` is None.
            frame_level_cond: if True, preserve one token per frame
                (output shape ``(B, T, hidden_size)``); if False, pool
                actions into a single token (output ``(B, 1, hidden_size)``).

        Returns:
            ``(B, T_out, hidden_size)`` action hidden states; ``T_out == T``
            if ``frame_level_cond`` else 1.
        """
        if action.ndim != 3:
            raise ValueError(
                f"FlaxActionEncoder: expected action shape (B, T, D), got {action.shape}"
            )
        if not frame_level_cond:
            # (B, T, D) -> (B, 1, T*D)
            b, t, d = action.shape
            action = action.reshape(b, 1, t * d)

        x = self.linear_1(action)
        x = nn.silu(x)
        x = self.linear_2(x)
        x = nn.silu(x)
        x = self.linear_3(x)

        if text_embeds is not None and self.text_embed_dim is not None:
            # Tile the (B, C_text) projection to (B, 1, hidden_size) by
            # repeating along the feature axis; broadcast-add across T.
            reps = self.hidden_size // self.text_embed_dim
            tiled = jnp.tile(text_embeds[:, None, :], (1, 1, reps))
            x = x + tiled.astype(x.dtype)
        return x

    def init_weights(
        self,
        rng: jax.Array,
        batch: int = 1,
        num_frames: int = 1,
    ):
        """Build an init-shaped param tree for this module."""
        action = jnp.zeros((batch, num_frames, self.action_dim), dtype=self.dtype)
        if self.text_embed_dim is not None:
            text = jnp.zeros((batch, self.text_embed_dim), dtype=self.dtype)
        else:
            text = None
        params_rng, dropout_rng = jax.random.split(rng)
        return self.init(
            {"params": params_rng, "dropout": dropout_rng},
            action,
            text,
            True,
        )["params"]
