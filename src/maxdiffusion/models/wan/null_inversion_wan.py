"""Null-text inversion helpers for the frozen WAN TI2V backbone (exp_04).

Round R1 covers the pieces every later round and every cached artifact depends on: the noise
conventions (plan §3, N3/M1) and the null-token embedding that turns optimized tokens into the
``[B, 512, 4096]`` context the CFG null branch consumes. The sigma grid itself is reused from
``side_adapter_wan.build_rollout_sigmas`` -- there is one sampler in this repo, and inversion must
run on it.

Noise conventions. Comparisons in this experiment are paired at matched noise (A1 vs A0 per example,
A2 vs A2-0 at the single canonical ``eps_0 = global_noise(0)``, adapter vs pre_context at identical
``z_start``), and cached targets are only valid for the exact tensor the optimizer saw. So the draw
is a pure function of ``(name, k)`` -- one float32 ``LATENT_SHAPE`` tensor, with no shape argument to
vary -- independent of batch composition, iteration order, host count, and of every other RNG stream
in the process. Batches are assembled by the caller (stack keyed draws, broadcast the global draw),
never by drawing a batch-shaped tensor, whose rows past 0 would not be the canonical noise.
``NOISE_DOMAIN`` separates this ``fold_in`` chain from any other use of the same seed. The mapping is
pinned by golden fingerprints (including a non-ASCII manifest name, which pins the UTF-8 encoding) in
``tests/worklogs_yixun/test_null_adapter_noise.py``; changing seed, domain, fold order or hash
slicing invalidates every cached target.
"""

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np


NOISE_DOMAIN = 0x4E4F4953  # "NOIS" in ASCII.
NOISE_SEED = 2026
LATENT_SHAPE = (48, 9, 12, 20)
GLOBAL_NOISE_NAME = "GLOBAL"


def noise_key(name: str, k: int) -> jax.Array:
    """Derive the PRNG key for ``(name, k)`` exactly as specified in plan §3."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    w0 = int.from_bytes(digest[0:4], "big")
    w1 = int.from_bytes(digest[4:8], "big")
    key = jax.random.PRNGKey(NOISE_SEED)
    for word in (w0, w1, NOISE_DOMAIN, int(k)):
        key = jax.random.fold_in(key, word)
    return key


def keyed_noise(name: str, k: int) -> jax.Array:
    """Per-example standard normal noise, keyed by the manifest ``name`` and seed index ``k``.

    Always exactly one float32 ``LATENT_SHAPE`` draw: the shape is part of the convention, not a
    caller's choice. Assemble a batch by stacking one draw per name --
    ``jnp.stack([keyed_noise(n, k) for n in names])``.
    """
    return jax.random.normal(noise_key(name, k), LATENT_SHAPE, dtype=jnp.float32)


def global_noise(k: int) -> jax.Array:
    """The one draw shared by all examples under the ``global`` convention; ``eps_0 = global_noise(0)``.

    Assemble a batch by broadcasting it -- ``jnp.broadcast_to(global_noise(k), (b, *LATENT_SHAPE))``.
    Never by requesting a batch-shaped draw: that is a different tensor whose rows past 0 are not
    ``eps_k``, which would silently turn "one canonical noise" into ``b`` different ones.
    """
    return keyed_noise(GLOBAL_NOISE_NAME, k)


def embed_null_tokens(null_tokens: jax.Array, base_context: jax.Array) -> jax.Array:
    """Splice ``null_tokens`` into the leading rows of the T5("") context.

    ``null_tokens`` is ``[B, L, D]`` (or ``[L, D]``, broadcast to ``B = 1``) and ``base_context`` is
    ``[S, D]`` (or ``[1, S, D]``). The result is ``[B, S, D]``: rows ``[0:L]`` are the null tokens,
    rows ``[L:S]`` are ``base_context``'s own rows, bit for bit. The output dtype is the promotion of
    the two inputs, so passing ``base_context[0:L]`` as the tokens reproduces ``base_context``
    exactly -- the branch-equality contract of plan §3.
    """
    null_tokens = jnp.asarray(null_tokens)
    base_context = jnp.asarray(base_context)

    if base_context.ndim == 3:
        if base_context.shape[0] != 1:
            raise ValueError(f"base_context must have a unit leading axis when rank-3, got {base_context.shape}")
        base_context = base_context[0]
    if base_context.ndim != 2:
        raise ValueError(f"base_context must be [S, D] or [1, S, D], got shape {base_context.shape}")
    if null_tokens.ndim == 2:
        null_tokens = null_tokens[None]
    if null_tokens.ndim != 3:
        raise ValueError(f"null_tokens must be [L, D] or [B, L, D], got shape {null_tokens.shape}")

    seq_len, dim = base_context.shape
    batch, length, token_dim = null_tokens.shape
    if token_dim != dim:
        raise ValueError(f"null_tokens dim {token_dim} does not match base_context dim {dim}")
    if length > seq_len:
        raise ValueError(f"null_tokens length {length} exceeds base_context length {seq_len}")

    dtype = jnp.promote_types(null_tokens.dtype, base_context.dtype)
    context = jnp.broadcast_to(base_context.astype(dtype)[None], (batch, seq_len, dim))
    return context.at[:, :length].set(null_tokens.astype(dtype))


def base_context_fingerprint(base_context) -> str:
    """sha256 of the context, so a reader can reject nulls optimized against a different T5("").

    Byte convention (fixed; a cached artifact stores this digest): after the host transfer the array
    is canonicalized to explicitly little-endian float32 in C order and hashed as
    ``np.ascontiguousarray(np.asarray(base_context).astype(np.dtype("<f4"))).tobytes()``. The
    endianness is spelled out rather than left to ``np.float32`` -- which is native-endian -- so a
    digest written on one host stays valid on another. Neither shape nor dtype enters the preimage:
    ``[1, S, D]`` and ``[S, D]`` fingerprint identically, and a bf16 or float64 context fingerprints
    as its float32 conversion.
    """
    canonical = np.ascontiguousarray(np.asarray(base_context).astype(np.dtype("<f4")))
    return hashlib.sha256(canonical.tobytes()).hexdigest()
