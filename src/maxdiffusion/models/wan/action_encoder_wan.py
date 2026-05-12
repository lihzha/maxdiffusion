"""Action encoder for action-conditioned WAN video generation.

Per-latent-frame 3-layer SiLU MLP: the 4 raw-frame actions that correspond
to one latent frame are concatenated into a single 28-dim vector and encoded
into the WAN cross-attention space.

    action (B, F_lat, 4, action_dim) → (B, F_lat, out_dim=4096)

Concatenating preserves temporal ordering across the 4 frames so the network
can learn velocity and trajectory-shape features. The output token is used as
K/V in the WAN transformer's per-frame cross-attention (frame_level_cond=True),
where the image latent's spatial patches are the queries.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class NNXWanActionEncoder(nnx.Module):
    """3-layer SiLU MLP encoding 4 concatenated raw-frame actions per latent frame.

    Args:
        rngs:         NNX Rngs for parameter initialisation.
        action_dim:   Width of a single raw-frame action vector (7 for DROID EEF).
        num_actions:  Number of raw frames concatenated per latent frame (4 for WAN 4× temporal compression).
        hidden_dim:   Width of hidden layers.
        out_dim:      Output width — must equal wan_text_dim (4096).
        dtype:        Activation dtype.
        weights_dtype: Parameter storage dtype.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        action_dim: int = 7,
        num_actions: int = 4,
        hidden_dim: int = 1024,
        out_dim: int = 4096,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.dtype = dtype
        kaiming = nnx.initializers.he_normal()

        self.linear_1 = nnx.Linear(
            in_features=action_dim * num_actions,   # 28 for 4×7
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
        )

    def __call__(
        self,
        action: jax.Array,
        text_embed: jax.Array | None = None,
    ) -> jax.Array:
        """Encode 4 concatenated raw-frame actions per latent frame.

        Args:
            action:     ``(B, F_lat, 4, action_dim)`` normalised action sequence.
            text_embed: ``(B, out_dim)`` pooled T5 embedding, broadcast-added
                        as a per-sample bias.

        Returns:
            ``(B, F_lat, out_dim)`` — one action token per latent frame.
        """
        B, F, T_a, A = action.shape
        # Flatten frames into batch and concatenate the 4 actions into one vector.
        x = action.reshape(B * F, T_a * A).astype(self.dtype)  # (B*F, 4*action_dim)
        x = jax.nn.silu(self.linear_1(x))  # (B*F, hidden_dim)
        x = jax.nn.silu(self.linear_2(x))  # (B*F, hidden_dim)
        x = self.linear_3(x)               # (B*F, out_dim)
        if text_embed is not None:
            # Tile (B, out_dim) → (B*F, out_dim): repeat each sample F times.
            x = x + jnp.repeat(text_embed, F, axis=0).astype(x.dtype)
        return x.reshape(B, F, -1)         # (B, F_lat, out_dim)
