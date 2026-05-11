"""Action encoder for action-conditioned WAN video generation.

NNX 3-layer SiLU MLP that lifts per-frame robot actions into the WAN
cross-attention space so they can be prepended to T5 text token sequences.

    action (B, T, action_dim)  →  (B, T, out_dim=4096)

The output dim matches the T5 text embedding dim (text_dim in the WAN config,
default 4096). Action tokens are concatenated with T5 tokens before the WAN
transformer's text_embedder projection, which applies a shared Linear to the
combined sequence — action tokens pass through the same (4096 → 5120) linear
as text tokens. No changes to the WAN model are needed.

Mirrors the FlaxActionEncoder used by the SVD Ctrl-World trainer but written
in NNX so it integrates cleanly with nnx.split / nnx.merge.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class NNXWanActionEncoder(nnx.Module):
    """3-layer SiLU MLP mapping robot actions to the WAN cross-attention dim.

    Args:
        rngs:         NNX Rngs for parameter initialisation.
        action_dim:   Width of per-frame action vectors (7 for DROID EEF).
        hidden_dim:   Width of hidden layers.
        out_dim:      Output feature dim — must equal the WAN text_dim (4096)
                      so action tokens can be concatenated with T5 tokens.
        dtype:        Activation dtype (bfloat16 in mixed-precision training).
        weights_dtype: Parameter storage dtype.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        action_dim: int = 7,
        hidden_dim: int = 1024,
        out_dim: int = 4096,
        dtype: jnp.dtype = jnp.bfloat16,
        weights_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.dtype = dtype
        kaiming = nnx.initializers.he_normal()

        self.linear_1 = nnx.Linear(
            in_features=action_dim,
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
        """Encode per-frame robot actions, optionally fusing a pooled text embed.

        Args:
            action:     ``(B, T, action_dim)`` normalised action sequence.
            text_embed: ``(B, out_dim)`` pooled text embedding (e.g. mean of T5
                        tokens). Broadcast-added to every action token, matching
                        the additive text-fusion used by SVD Ctrl-World.

        Returns:
            ``(B, T, out_dim)`` action token embeddings.
        """
        x = action.astype(self.dtype)
        x = jax.nn.silu(self.linear_1(x))
        x = jax.nn.silu(self.linear_2(x))
        x = self.linear_3(x)
        if text_embed is not None:
            # (B, out_dim) → (B, 1, out_dim), broadcast across T
            x = x + text_embed[:, None, :].astype(x.dtype)
        return x
