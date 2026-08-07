"""exp_04 J1c — do J1b's jointly-optimized nulls transfer across noise basins?

J1b showed a jointly-optimized null tensor can reach own-basin quality from fresh noise on 3/8 DEV
examples. J1c asks what that tensor *is*: a property of the example, or of the draw it was optimized
against -- the question A1/A2 died on. So this mode never optimizes anything. It loads J1b's published
tensors, replays them from four noise settings, and scores each.

Two invariants make the answer meaningful:

- **The rows stay with their example.** ``a3_nulls.npz`` is step-major -- ``direct_optimize_nulls``
  returns ``[N, B, L, D]`` and ``write_arrays`` stores it unchanged -- so example i is ``nulls[:, i]``.
  Read as ``[B, N, ...]``, or reordered against the manifest, the probe would produce a complete table
  showing no transfer, which is indistinguishable from the finding it is looking for.
- **The noise is the conventions' own.** ``global(0)`` and ``keyed(name, k)`` come from R1's
  golden-pinned helpers by import, so J1c's numbers sit in the same basins as J1's.

The artifact carries the npz URI **and its sha256** beside the run's provenance: a transfer table is
only interpretable next to the J1b tensors it was computed from.
"""

from __future__ import annotations

import hashlib
import io
import posixpath
import time
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.wan.null_inversion_wan import global_noise, keyed_noise, replay_with_nulls
from maxdiffusion.null_adapter_modes import decode_cohort
from maxdiffusion.null_adapter_pixels import fill_pixel_metrics
from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY
from maxdiffusion.null_adapter_runner_core import _metric_tables
from maxdiffusion.null_adapter_verify import canonical_sigmas

TRANSFER_NAME = "transfer_probe.json"
NULLS_FIELD = "nulls"
# global(0) is A2's canonical eps_0 -- the "own" basin J1b optimized from; the keyed draws are the
# deployment k-set the capacity study reports on.
TRANSFER_SETTINGS = (("global", 0), ("keyed", 0), ("keyed", 1), ("keyed", 2))


def setting_label(convention: str, k: int) -> str:
    return f"{convention}_{int(k)}"


def _read_bytes(uri: str) -> bytes:  # pragma: no cover -- object-store glue
    from tensorflow.io import gfile

    with gfile.GFile(uri, "rb") as handle:
        return handle.read()


def load_transfer_nulls(
    uri: str,
    *,
    read_bytes: Callable[[str], bytes],
    steps: int,
    l_null: int,
    width: int,
    examples: int,
) -> tuple[np.ndarray, str]:
    """J1b's ``a3_nulls.npz`` as ``[N, B, L, D]`` float32, with the sha256 of the bytes it came from.

    Every geometry claim is checked against what this run is about to replay: a tensor from another
    step count, row count, context width or cohort size is not this experiment's.
    """
    if not uri:
        raise ValueError(
            "null_transfer_nulls_uri is required for the transfer_probe mode: the probe replays J1b's "
            "published tensors and has nothing to measure without them"
        )
    payload = read_bytes(uri)
    digest = hashlib.sha256(payload).hexdigest()
    with np.load(io.BytesIO(payload)) as archive:
        if NULLS_FIELD not in archive:
            raise ValueError(f"{uri} carries {sorted(archive)!r}, not the {NULLS_FIELD!r} array J1b writes")
        nulls = np.asarray(archive[NULLS_FIELD], dtype=np.float32)
    if nulls.ndim != 4:
        raise ValueError(f"{uri}: nulls must have rank 4 [N, B, L, D] as J1b stores them, got shape {nulls.shape}")
    if int(nulls.shape[0]) != int(steps):
        raise ValueError(f"{uri}: nulls carry {nulls.shape[0]} sampler-step rows, this run replays {steps}")
    if int(nulls.shape[1]) != int(examples):
        raise ValueError(
            f"{uri}: nulls cover {nulls.shape[1]} examples but the cohort names {examples}; J1b published "
            f"the first eight DEV examples and the probe replays exactly those"
        )
    if int(nulls.shape[2]) != int(l_null):
        raise ValueError(f"{uri}: nulls carry l_null={nulls.shape[2]}, this run runs at {l_null}")
    if int(nulls.shape[3]) != int(width):
        raise ValueError(f"{uri}: nulls carry context width {nulls.shape[3]}, this run's base context is {width}")
    if not np.all(np.isfinite(nulls)):
        raise ValueError(f"{uri}: nulls must be finite -- a tensor carrying NaN or inf replays to nothing")
    return nulls, digest


def transfer_start_latents(names: Sequence[str], convention: str, k: int, *, geometry=None) -> jnp.ndarray:
    """``[B, C, F, H, W]`` starting latents for one setting, from R1's golden-pinned draws."""
    geometry = PRODUCTION_GEOMETRY.z_video if geometry is None else tuple(geometry)
    if convention == "global":
        return jnp.broadcast_to(global_noise(k), (len(names), *geometry))
    if convention == "keyed":
        return jnp.stack([keyed_noise(name, k) for name in names])
    raise ValueError(f"unknown noise convention {convention!r}: the probe runs global | keyed")


def run_transfer_probe(
    plan: Mapping[str, Any],
    backend: Any,
    sinks: Any,
    *,
    artifact_dir: str,
    nulls_uri: str,
    manifest_hash: str,
    code_sha: str,
    decode_batch_size: int = 8,
    read_bytes: Callable[[str], bytes] | None = None,
    replay: Callable[..., Any] = replay_with_nulls,
    sigmas: Any = None,
    geometry: Any = None,
) -> dict:
    """Replay J1b's nulls from every setting and publish the scored table."""
    started = time.time()
    names = tuple(plan["names"])
    params = plan["params"]
    guide_scale, l_null = float(params["guide_scale"]), int(params["l_null"])
    if not str(code_sha):
        raise ValueError("a transfer_probe artifact carries the run's code_sha or it is not published")
    if not str(manifest_hash):
        raise ValueError("a transfer_probe artifact carries the manifest_hash of the cohort it scored")

    batch, _ = backend.read_batch(names)
    z_i0 = jnp.asarray(batch.z_i0, jnp.float32)
    z_video = jnp.asarray(batch.z_video, jnp.float32)
    base_context = jnp.asarray(backend.base_context, jnp.float32)
    grid = canonical_sigmas() if sigmas is None else np.asarray(sigmas, np.float32)
    latent_geometry = tuple(z_video.shape[1:]) if geometry is None else tuple(geometry)

    nulls, digest = load_transfer_nulls(
        nulls_uri,
        read_bytes=read_bytes or _read_bytes,
        steps=int(len(grid) - 1),
        l_null=l_null,
        width=int(base_context.shape[-1]),
        examples=len(names),
    )
    shared = jnp.asarray(nulls, jnp.float32)

    latents = {}
    for convention, k in TRANSFER_SETTINGS:
        z_start = transfer_start_latents(names, convention, k, geometry=latent_geometry)
        # The SAME tensor every time: the probe varies the basin, never the nulls.
        latents[setting_label(convention, k)] = np.asarray(
            replay(
                backend.velocity_fn, z_start, z_i0, jnp.asarray(grid), shared, base_context, guide_scale=guide_scale
            )
        )

    tables = _metric_tables(names, {label: jnp.asarray(value) for label, value in latents.items()}, z_video)
    pixels = decode_cohort(backend.decode_fn, backend.read_batch, latents, names, batch_size=int(decode_batch_size))
    report = {
        "mode": "transfer_probe",
        "settings": [setting_label(*setting) for setting in TRANSFER_SETTINGS],
        "table": fill_pixel_metrics(tables, pixels),
        "provenance": {
            "nulls_uri": str(nulls_uri),
            "nulls_sha256": digest,
            "code_sha": str(code_sha),
            "manifest_hash": str(manifest_hash),
            "model_revision": str(getattr(backend, "model_revision", "")),
            "cohort": str(plan["cohort"]),
            "names": list(names),
            "l_null": l_null,
            "guide_scale": guide_scale,
            "sigma_steps": int(len(grid) - 1),
        },
        "seconds": round(time.time() - started, 3),
    }
    sinks.write_json(posixpath.join(artifact_dir, TRANSFER_NAME), report)
    return report
