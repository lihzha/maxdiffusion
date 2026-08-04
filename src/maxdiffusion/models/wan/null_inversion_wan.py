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
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    apply_first_frame_pin,
    rollout_timesteps_from_sigmas,
)


NOISE_DOMAIN = 0x4E4F4953  # "NOIS" in ASCII.
NOISE_SEED = 2026
LATENT_SHAPE = (48, 9, 12, 20)
GLOBAL_NOISE_NAME = "GLOBAL"
# The WAN scheduler's training horizon (``scheduler.config.num_train_timesteps``). The deployed
# sampler scales sigma by it, so inversion must use the same value or it queries the velocity at a
# different point of the schedule than the replay it is supposed to invert.
NUM_TRAIN_TIMESTEPS = 1000
N_HIST_FRAMES = 1  # frame 0 is the pinned image condition, so its tokens carry timestep 0.


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


def _validate_sigmas(sigmas) -> np.ndarray:
    """The grid is a host-side constant, so it is checked eagerly (never under ``jit``)."""
    values = np.asarray(sigmas, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"sigmas must be 1-D, got shape {values.shape}")
    if values.size < 2:
        raise ValueError(f"sigmas must define at least one step (>= 2 entries), got {values.size}")
    # Before monotonicity: [inf, 1.0, 0.0] is "strictly descending" and would otherwise be accepted,
    # after which every step's arithmetic is invalid.
    if not np.all(np.isfinite(values)):
        raise ValueError("sigmas must be finite")
    if not np.all(np.diff(values) < 0.0):
        raise ValueError("sigmas must be strictly descending")
    if values[-1] != 0.0:
        raise ValueError(f"sigmas must end at 0.0, got {values[-1]}")
    return values


def invert_trajectory(
    velocity_fn: Callable[[jax.Array, jax.Array], jax.Array],
    z_video: jax.Array,
    z_i0: jax.Array,
    sigmas: jax.Array,
) -> jax.Array:
    """Reverse-Euler inversion of the clean latent back up the sigma grid (plan §3 step 1).

    ``traj[i]`` is the pivot at ``sigmas[i]``, filled from the clean end downwards::

        traj[N] = pin(z_video)
        traj[i] = pin(traj[i+1] + (sigmas[i] - sigmas[i+1]) * velocity_fn(traj[i+1], t(sigmas[i])))

    i.e. the velocity is evaluated at the *incoming* point ``traj[i+1]`` but with ``sigma_i`` -- the
    DDIM-inversion small-step approximation, and the one asymmetry that makes this an inversion
    rather than a replay. Ported from ``compute_inversion_trajectory``
    (``third_party/Wan2.2/scripts/embedding_search.py:522-572``, submodule pin f370228).

    Seam: ``velocity_fn(latents, timestep_2d) -> v``. The context (and any CFG mixing) is closed over
    by the caller, because inversion runs at w=1 with a single fixed context -- keeping the seam at
    two arguments means this function never has to know about T5, adapters or guidance. ``v`` may
    come back in any float dtype; latent arithmetic stays float32 (as in the reference, which calls
    ``.float()`` on the model output). ``v`` must have exactly the latent shape: a scalar or
    unbatched velocity would broadcast across the batch, so it is rejected rather than accepted.

    ``timestep_2d`` is the per-token tensor the WAN DiT expects, built by the sampler's own
    ``_build_per_token_timestep`` with ``n_hist=1``: frame-0 tokens carry 0, all later tokens carry
    ``sigmas[i] * NUM_TRAIN_TIMESTEPS`` (the reference's ``temp_ts``).

    Args:
      velocity_fn: the seam above.
      z_video: ``[B, C, F, H, W]`` clean target latents.
      z_i0: ``[B, C, 1, H, W]`` (or ``[B, C, F, H, W]``) first-frame condition; frame 0 of every
        trajectory entry is pinned to it.
      sigmas: the ``N+1`` descending grid ending at 0.0, from ``build_rollout_sigmas``. Validated
        eagerly, so it must be concrete.

    Returns:
      ``[N+1, B, C, F, H, W]`` float32 trajectory, indexed by sigma step.
    """
    _validate_sigmas(sigmas)
    z_video = jnp.asarray(z_video).astype(jnp.float32)
    z_i0 = jnp.asarray(z_i0).astype(jnp.float32)
    sigmas = jnp.asarray(sigmas, dtype=jnp.float32)

    if z_video.ndim != 5:
        raise ValueError(f"z_video must be [B, C, F, H, W], got shape {z_video.shape}")
    if z_i0.ndim != 5:
        raise ValueError(f"z_i0 must be [B, C, 1, H, W] or [B, C, F, H, W], got shape {z_i0.shape}")
    batch, channels, f_lat, h_lat, w_lat = z_video.shape
    if (z_i0.shape[0], z_i0.shape[1], z_i0.shape[3:]) != (batch, channels, (h_lat, w_lat)):
        raise ValueError(f"z_i0 shape {z_i0.shape} is inconsistent with z_video shape {z_video.shape}")
    if z_i0.shape[2] not in (1, f_lat):
        raise ValueError(f"z_i0 must carry 1 or {f_lat} latent frames, got {z_i0.shape[2]}")

    timesteps = rollout_timesteps_from_sigmas(sigmas, NUM_TRAIN_TIMESTEPS)

    def body(current, i):
        step_t = jnp.broadcast_to(timesteps[i], (batch,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=N_HIST_FRAMES)
        velocity = jnp.asarray(velocity_fn(current, timestep_2d)).astype(jnp.float32)
        # Shapes are static under tracing, so this fires at trace time, once, with no host callback.
        # Without it a scalar or unbatched velocity broadcasts across the batch and corrupts every
        # pivot plausibly instead of failing.
        if velocity.shape != current.shape:
            raise ValueError(f"velocity_fn returned shape {velocity.shape}, expected the latent shape {current.shape}")
        filled = apply_first_frame_pin(current + (sigmas[i] - sigmas[i + 1]) * velocity, z_i0)
        return filled, filled

    start = apply_first_frame_pin(z_video, z_i0)
    _, descending = lax.scan(body, start, jnp.arange(sigmas.shape[0] - 2, -1, -1))
    # ``descending`` holds i = N-1 .. 0; flip it back into sigma order and append the clean pivot.
    return jnp.concatenate([descending[::-1], start[None]], axis=0)
