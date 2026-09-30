# Copyright 2026 Princeton. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Camera-frame action encoder for the action-conditioned SVD world model.

SVD counterpart of ``NNXWanCameraActionEncoder``. Encodes the anchored
camera-frame action (see ``input_pipeline/robot/camera_frame_actions.py``) once
per camera, so each frame carries one token per view instead of one::

    (B, T, num_views, action_dim) -> (B, T, num_views, hidden_size)

The three ``cam_action*`` routes then send view ``v``'s token only to view
``v``'s rows of the H-stacked latent grid (the adaln route is the exception:
SVD's ``t_emb`` has no spatial axis, so it concatenates the views, exactly as
``skeleton_adaln`` has to pool its grid).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from .action_encoder_flax import FlaxActionAdaLNProjector, _kaiming_normal_relu_fan_in


class FlaxCameraActionEncoder(nn.Module):
    """``FlaxActionEncoder``'s 3-layer MLP, applied per camera, plus a view embedding.

    Same layer names, Kaiming init and zero-init ``linear_3`` as
    ``FlaxActionEncoder`` (so a fresh encoder emits exactly zero and step 0 is the
    pretrained model). The learned view embedding is added after the first
    activation; the sites already keep each camera's token inside its own
    region, so it is not needed for routing — it lets the shared MLP treat the
    anchored wrist ego-motion and the static exterior-camera poses differently.
    """

    action_dim: int = 10
    hidden_size: int = 1024
    num_views: int = 3
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32

    def setup(self):
        kaiming = _kaiming_normal_relu_fan_in()
        self.linear_1 = nn.Dense(
            self.hidden_size, kernel_init=kaiming, dtype=self.dtype, param_dtype=self.weights_dtype
        )
        self.linear_2 = nn.Dense(
            self.hidden_size, kernel_init=kaiming, dtype=self.dtype, param_dtype=self.weights_dtype
        )
        self.linear_3 = nn.Dense(
            self.hidden_size,
            # Zero-init: see FlaxActionEncoder.linear_3.
            kernel_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
        )
        self.view_embed = self.param(
            "view_embed",
            nn.initializers.normal(stddev=self.hidden_size**-0.5),
            (self.num_views, self.hidden_size),
            self.weights_dtype,
        )

    def __call__(self, action: jnp.ndarray) -> jnp.ndarray:
        """``(B, T, V, action_dim)`` -> ``(B, T, V, hidden_size)``."""
        if action.ndim != 4 or action.shape[2] != self.num_views:
            raise ValueError(
                f"FlaxCameraActionEncoder: expected (B, T, {self.num_views}, D), got {action.shape}"
            )
        x = nn.silu(self.linear_1(action))
        x = x + self.view_embed.astype(x.dtype)
        x = nn.silu(self.linear_2(x))
        return self.linear_3(x)

    def init_weights(self, rng: jax.Array, batch: int = 1, num_frames: int = 1):
        action = jnp.zeros((batch, num_frames, self.num_views, self.action_dim), dtype=self.dtype)
        return self.init({"params": rng}, action)["params"]


def build_cam_action_modules(
    action_cond_mode: str,
    *,
    action_dim: int,
    hidden_size: int,
    num_views: int,
    time_embed_dim: int,
    model_channels: int,
    seed: int,
    dtype: jnp.dtype,
    weights_dtype: jnp.dtype,
) -> dict:
    """``{params_key: (module, init_params)}`` for a cam_action mode, else ``{}``.

    Shared by the trainer and ``generate_ctrl_world.py`` so the rollout's restore
    template comes from the same code (and seeds) as the checkpoint. The adaln
    and additive projections reuse ``FlaxActionAdaLNProjector`` — a single
    Dense, the vector adaln route's own adapter — so the sites differ only in
    where the output goes: ``time_embed_dim`` (all views concatenated, since
    t_emb has no spatial axis) or conv_in's ``model_channels`` (one per view).
    """
    if action_cond_mode not in ("cam_action", "cam_action_adaln", "cam_action_cross_attn"):
        return {}
    encoder = FlaxCameraActionEncoder(
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_views=num_views,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )
    extras = {"cam_action_encoder": (encoder, encoder.init_weights(jax.random.PRNGKey(seed + 5)))}
    if action_cond_mode == "cam_action_adaln":
        proj = FlaxActionAdaLNProjector(
            time_embed_dim=time_embed_dim, dtype=dtype, weights_dtype=weights_dtype
        )
        extras["cam_action_adaln_proj"] = (
            proj,
            proj.init_weights(jax.random.PRNGKey(seed + 6), hidden_size=num_views * hidden_size),
        )
    elif action_cond_mode == "cam_action":
        proj = FlaxActionAdaLNProjector(
            time_embed_dim=model_channels, dtype=dtype, weights_dtype=weights_dtype
        )
        extras["cam_action_add_proj"] = (
            proj,
            proj.init_weights(jax.random.PRNGKey(seed + 7), hidden_size=hidden_size),
        )
    return extras
