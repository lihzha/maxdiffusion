"""Independent replay verification for exp_04 cached artifacts (plan §5 item 5, F14).

``verify_replay`` answers one question: **does this record, under its own shard provenance header,
regenerate the output it claims?** It rebuilds the CFG replay from the record's ``z_start`` and
per-step nulls and compares the landing point against the record's ``expected_final_latent``.

It reads five fields of the record and no others -- ``ALLOWED_RECORD_FIELDS``. In particular it never
touches ``z_video``, ``actions``, ``per_step_final_losses`` or ``final_future_mse``: ``z_video`` is
the trainer's copy of the answer, and a verifier that consulted it would be checking the pipeline
against itself. That mirrors the reference verifier
(``third_party/Wan2.2/scripts/verify_reconstruction_from_null.py``, submodule pin f370228), whose
whole argument is what it declines to load -- no ground-truth frames, no inversion trajectory, no
scheduler RNG.

Everything provenance-related is checked **before** any model forward, each with its own message, so
a mismatch costs nothing and says exactly what disagreed. The sigma grid is not taken on trust: it is
recomputed from ``build_rollout_sigmas`` and required to match the header elementwise.

This lives outside ``null_adapter_records`` on purpose. Record IO is numpy-only and usable without
jax; verification is host orchestration of a jax replay, so keeping it here preserves that property.

``atol`` is a required argument with no default: what counts as "reproduced" depends on the storage
dtype and on the caller's fidelity gate, and that policy belongs to the runner, not here.

**Contract for the writer (R6/R8):** compute ``expected_final_latent`` from the values the record
will *store*, not from the pre-cast inputs. ``z_i0`` is always kept at the source fp16 (plan §4-P2),
so a replay seeded from an fp32 ``z_i0`` lands ~5e-4 away from one seeded from the stored fp16 copy
and could never verify at a tight ``atol``. Cast first, then replay, then record.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.wan.null_inversion_wan import base_context_fingerprint, replay_with_nulls
from maxdiffusion.models.wan.side_adapter_wan import build_rollout_sigmas
from maxdiffusion.null_adapter_records import LATENT_DTYPES, _validate_header


# The canonical deployment grid (plan §3); the shift is 5.0 and is unrelated to the CFG weight.
SIGMA_STEPS, SIGMA_SHIFT, SIGMA_MIN, SIGMA_MAX = 25, 5.0, 0.0, 1.0
# Every record attribute the verifier is permitted to read: the replay inputs, plus the two
# provenance labels it cross-checks. None of them carries the answer. The hostile-proxy test pins
# this set exactly -- reaching for anything else (z_video above all) fails the suite.
ALLOWED_RECORD_FIELDS = (
    "latent_dtype",
    "noise_convention",
    "arm",
    "nulls",
    "z_start",
    "z_i0",
    "expected_final_latent",
)
_LATENT_SCOPED = ("nulls", "z_start", "expected_final_latent")


def canonical_sigmas() -> np.ndarray:
    """The one grid every exp_04 arm runs on, recomputed rather than trusted."""
    return np.asarray(build_rollout_sigmas(SIGMA_STEPS, SIGMA_SHIFT, SIGMA_MIN, SIGMA_MAX), dtype=np.float32)


def verify_replay(
    record: Any,
    header: Any,
    velocity_fn: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
    base_context,
    *,
    expected_model_revision: str,
    expected_guide_scale: float,
    expected_noise_convention: str,
    expected_arm: str,
    atol: float,
) -> None:
    """Raise unless ``record`` + ``header`` regenerate ``record.expected_final_latent``.

    Args:
      record: a ``NullAdapterRecord`` (or anything exposing ``ALLOWED_RECORD_FIELDS``).
      header: the ``ProvenanceHeader`` of the shard the record came from.
      velocity_fn: ``(latents, timestep_2d, encoder_hidden_states) -> v``, wrapping the same weights
        ``expected_model_revision`` names.
      base_context: the ``[S, D]`` T5("") context; its fingerprint must match the header's.
      expected_model_revision: the revision the caller actually loaded. The header must agree.
      expected_guide_scale: the CFG weight the caller is verifying under. The header must agree;
        declaring it turns a wrong w into a provenance error instead of a numeric mystery.
      expected_noise_convention: 'keyed' or 'global' (plan §3). The record must agree.
      expected_arm: the arm the caller believes it is verifying (A0/A1/A2/...). The record must agree.
      atol: maximum tolerated ``max|replay - expected|``. Required: the fidelity policy is the
        runner's, not this function's.

    Raises:
      ValueError: on any provenance mismatch, or if the replay misses by more than ``atol``.
    """
    _validate_header(header)  # the writer's own contract, reused rather than re-stated

    header_sigmas = np.asarray(header.sigma_vector, dtype=np.float32)
    if not np.array_equal(header_sigmas, canonical_sigmas()):
        raise ValueError("header sigma vector does not match the canonical grid from build_rollout_sigmas")
    if header.model_revision != expected_model_revision:
        raise ValueError(
            f"header model_revision {header.model_revision!r} does not match the loaded "
            f"weights {expected_model_revision!r}"
        )
    if not np.isfinite(header.guide_scale):
        raise ValueError(f"header guide_scale must be finite, got {header.guide_scale}")
    if float(header.guide_scale) != float(expected_guide_scale):
        # Declared by the caller like the revision is: a wrong w would otherwise surface only as a
        # numeric divergence, indistinguishable from a corrupted artifact.
        raise ValueError(f"header guide_scale {header.guide_scale} does not match the expected {expected_guide_scale}")
    if base_context_fingerprint(base_context) != header.base_context_fingerprint:
        raise ValueError('base_context fingerprint does not match the header\'s: this is a different T5("") context')

    if header.dtype_policy != record.latent_dtype:
        # The pair has to agree with itself: an fp32 record under an fp16-declaring header means one
        # of the two is from a different run, and the numbers would not say which.
        raise ValueError(
            f"header dtype_policy {header.dtype_policy!r} does not match the record's "
            f"latent_dtype {record.latent_dtype!r}"
        )
    if record.noise_convention != expected_noise_convention:
        raise ValueError(
            f"record noise_convention {record.noise_convention!r} does not match the "
            f"expected {expected_noise_convention!r}"
        )
    if record.arm != expected_arm:
        raise ValueError(f"record arm {record.arm!r} does not match the expected {expected_arm!r}")

    nulls = np.asarray(record.nulls)
    if nulls.ndim != 3 or nulls.shape[1] != header.l_null:
        raise ValueError(f"header l_null {header.l_null} does not match the record's nulls of shape {nulls.shape}")
    stored_dtype = LATENT_DTYPES.get(record.latent_dtype)
    if stored_dtype is None or any(np.asarray(getattr(record, f)).dtype != stored_dtype for f in _LATENT_SCOPED):
        raise ValueError(f"record latent_dtype {record.latent_dtype!r} does not describe its stored arrays")
    if not np.isfinite(atol) or float(atol) < 0.0:
        raise ValueError(f"atol must be finite and non-negative, got {atol}")

    z_final = replay_with_nulls(
        velocity_fn,
        jnp.asarray(record.z_start, dtype=jnp.float32)[None],
        jnp.asarray(record.z_i0, dtype=jnp.float32)[None],
        jnp.asarray(header_sigmas, dtype=jnp.float32),
        jnp.asarray(nulls, dtype=jnp.float32),
        jnp.asarray(base_context, dtype=jnp.float32),
        guide_scale=float(header.guide_scale),
    )
    expected = jnp.asarray(record.expected_final_latent, dtype=jnp.float32)
    deviation = float(jnp.max(jnp.abs(z_final[0] - expected)))  # max, not mean: one bad voxel is a failure
    if not deviation <= float(atol):
        raise ValueError(
            f"replay does not reproduce expected_final_latent: max|delta| = {deviation:g} exceeds atol {float(atol):g}"
        )
