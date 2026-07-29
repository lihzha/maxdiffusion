"""Build the exp_02 `train100` / `train10` latent datasets (plan v4 §2 D4/D6, cycle B).

Consumes ONLY the committed `overfit100_manifest.json` (cycle A) and turns its 100 DROID
episodes into two fingerprinted schema-v2 TFRecord sets:

    gs://v6_east1d/datasets/exp02_overfit100/train100/   all 100 episodes (1,629 windows)
    gs://v6_east1d/datasets/exp02_overfit100/train10/    manifest episode_index 0-9 (167)

Stages, in order, every one fail-loud:

1. **Preflight** -- structural gate on the manifest (offline), a live re-stat of every
   fingerprinted object (zero drift required), the V1 fixture (pinned download, md5/size
   binding, names/shapes/dtypes, bitwise `z_i0 == z_video[:, :1]`), and the **mandatory VAE
   pin**: the manifest's `{hf_repo, revision, vae_config_sha256}` is resolved to ONE local
   snapshot directory, that directory is fingerprinted, and the SAME directory is handed to
   the loader -- so the weights that encode the dataset are the fingerprinted ones by
   construction. Absence or mismatch aborts before any model load (B1).
2. **Per episode** (manifest order) -- pinned MP4 download (md5/size verified BEFORE use),
   ffmpeg -> RGB frames (count must equal the manifest's `nb_frames`, geometry 320x192),
   window starts `0, 4, ...` while `s + 33 <= nb_frames` (count must equal `n_windows`).
3. **Per window** -- 33 frames -> `preprocess_frames` (pipeline-parity `[B, C, T, H, W]`,
   [-1, 1]) -> `vae.encode(x, cache)[0].mode()` -> pipeline `latents_mean/std` normalization
   -> channels-first -> float16 `z_video [48, 9, 12, 20]`; `z_i0 = z_video[:, 0:1]`.
4. **Gates V1-V4** (thresholds FINAL, plan D4). V3 decodes with the pipeline's bfloat16
   postprocess so its SSIM really is the ceiling the rollout evaluator is measured against
   (B7). V4 fails on non-finite output and, on failure, persists both frame-0 tensors plus
   difference quantiles; a V4 trip aborts and is NEVER re-thresholded in place (B2, see
   `V4_FAILURE_POLICY`). Any failure writes `failed_gates.json` and aborts.
5. **Staged writes -> readback -> promotion** (B3/B4) -- every shard and sidecar is written to
   `<set>/_staging_<build_id>/` and fingerprinted (sha256 + size locally, generation/md5/size
   remotely); after the gates, counts and audit pass, EVERY staged shard is physically read
   back (pinned), fully parsed, and checked for ordered names, schema, byte lengths, the
   `z_i0` slice contract and train10/train100 byte identity. Only then is the set promoted
   into a canonical prefix that must be EMPTY, and `_SUCCESS` written LAST.
   **Every reader (cycle-C trainer, cycle-D eval) must require `_SUCCESS`.**
6. **`--probe`** -- first 2 manifest episodes into `probe2/` only. It benchmarks the
   full-scale audit BEFORE loading the VAE, prechecks all ten fixed V3 windows, forces shard
   rollover (16 records/shard), requires peak memory from every local device, and reports
   preflight / vae_load / first_window_compile / steady_state_encode / upload / audit timings
   separately, extrapolating from the steady-state rate plus measured fixed costs (B5).
7. **Provenance** -- production AND probe refuse to start unless the cycle-B implementation
   (including the shared cycle-A guard and the scripts this job invokes) is committed and
   clean at HEAD, and unless the consumed manifest is byte-identical to the committed one (B6).

The encode path is a verbatim replica of `wan_pipeline.py`'s `_encode_video_to_t2v_latents`
(and its `_denormalize_latents` / `_decode_latents_to_video` inverse); D4 forbids any
deviation, including sampling the posterior (`.mode()` only -- no RNG enters the dataset).

CLI:
    python -m maxdiffusion.data_preprocessing.build_overfit100_dataset \
        --manifest docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json \
        --out-root gs://v6_east1d/datasets/exp02_overfit100 \
        --config src/maxdiffusion/configs/base_wan_5b_full_ft.yml
    # --probe   -> 2 episodes into <out-root>/probe2/, cost extrapolation, no train100/train10
    # --dry-run -> preflight only (manifest + fixture + VAE pin), no encoding
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.data_preprocessing.build_overfit100_manifest import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    MIN_FRAMES,
    WINDOW_STRIDE,
    DirtyImplementationError,
    assert_implementation_committed,
    collect_tool_versions,
    implementation_provenance_errors,  # noqa: F401 -- re-exported: cycle B reuses the cycle-A guard
    is_git_worktree,
    n_windows,
    resolve_vae_snapshot,
    vae_pin_errors,
    validate_manifest_structure,
    verify_manifest,
)
from maxdiffusion.data_preprocessing.extract_v1_fixture import (
    TARGET_NAMES,
    Z_I0_SHAPE,
    Z_VIDEO_SHAPE,
    load_fixture_npz,
    md5_b64,
    pinned_uri,
    run_gsutil,
    verify_fixture,
    verify_payload_binding,
)

# ---------------------------------------------------------------------------------- constants

OUT_ROOT = "gs://v6_east1d/datasets/exp02_overfit100"
DEFAULT_MANIFEST = "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json"
DEFAULT_CONFIG = "src/maxdiffusion/configs/base_wan_5b_full_ft.yml"

WINDOW_FRAMES = MIN_FRAMES  # 33 consecutive pixel frames -> 9 latent frames
FRAME_HEIGHT, FRAME_WIDTH = EXPECTED_HEIGHT, EXPECTED_WIDTH
VIEW_INDEX = 0

SCHEMA_V2_FIELDS = ("name", "episode_id", "episode_index", "window_start", "z_i0", "z_video", "instruction")
SHARD_SIZE = 256
# B5: the probe must exercise shard rollover, and 41 windows would be ONE 256-record shard.
PROBE_SHARD_SIZE = 16
TRAIN10_MAX_EPISODE_INDEX = 10
PROBE_EPISODES = 2
EXPECTED_EPISODES = 100
SUCCESS_MARKER = "_SUCCESS"
STAGING_PREFIX = "_staging_"
AUDIT_BENCHMARK_WINDOWS = 1629  # the full-build audit shape, benchmarked before any encode
AUDIT_BENCHMARK_DIM = int(np.prod(Z_I0_SHAPE))

# Gate thresholds -- plan v4 D4, FINAL.
V1_REL_L2_MAX = 0.25
V1_PEARSON_MIN = 0.97
V2_STD_MIN, V2_STD_MAX = 0.35, 0.95
V2_ABS_MEAN_MAX = 0.15
V3_SSIM_MIN = 0.80
V3_EPISODE_INDICES = tuple(range(0, 100, 10))
V4_RTOL = 1e-3
V4_ATOL = 0.0  # the plan states rtol only; kept explicit so a change is a one-line decision
V4_SHORT_FRAMES = 17  # -> 5 latent frames; latent frame 0 must be unchanged

AUDIT_HISTOGRAM_N = 100
AUDIT_CHUNK = 128

# B2 (cycle-B review): a V4 trip is NOT an in-job tolerance knob. It aborts; the recovery is a
# separately reviewed rerun that first establishes -- with the saved frame-0 arrays and
# same-shape controls (repeated encode of the identical window; future-replacement) -- that the
# failure is numerical rather than causal. Renewed launch sign-off is required either way.
V4_FAILURE_POLICY = (
    "V4 aborts the build. Do NOT re-threshold in place: the only recovery is a rerun from a "
    "separately reviewed commit whose same-shape controls (repeated-encode and "
    "future-replacement at identical shapes) show the difference is numerical, not causal "
    "leakage -- with renewed launch approval. Evidence saved: v4_frame0.npz + quantiles."
)

CYCLE_B_IMPLEMENTATION_PATHS = (
    "src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py",
    # B6: the shared cycle-A guard and every script this job invokes are code inputs too.
    "src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py",
    "bash_scripts/build_overfit100_dataset.sh",
    "bash_scripts/prefetch_hf_snapshot.sh",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_windows.py",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_encode.py",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_gates.py",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_tfrecord.py",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_audit.py",
)


class BuildError(RuntimeError):
    """A contract of the build was violated -- abort, never write a partial dataset."""


class GateFailure(BuildError):
    """An encoder-validation gate (V1-V4) failed. Plan R1: stop, no training."""


def _tf():
    """Import TensorFlow lazily: only the TFRecord path needs it."""
    import tensorflow as tf

    return tf


# ---------------------------------------------------------------------------------- D4: windows


def window_starts(nb_frames: int) -> list[int]:
    """Window starts `s = 0, 4, 8, ...` with `s + 33 <= nb_frames` (plan D4)."""
    count = n_windows(nb_frames)  # raises for clips shorter than one window
    return [start * WINDOW_STRIDE for start in range(count)]


def window_frame_range(start: int) -> tuple[int, int]:
    """The half-open pixel-frame span `[start, start + 33)` a window denotes."""
    return int(start), int(start) + WINDOW_FRAMES


def slice_window(frames: np.ndarray, start: int) -> np.ndarray:
    """The 33 frames of one window; refuses a start that runs past the decoded clip."""
    begin, end = window_frame_range(start)
    if begin < 0 or end > len(frames):
        raise BuildError(f"window start {start} needs frames [{begin}, {end}) but only {len(frames)} were decoded")
    return frames[begin:end]


def window_name(episode_id: int, start: int, view: int = VIEW_INDEX) -> str:
    """The exp_01 cache record name: `ep<ID>_v<VIEW>_s<START zero-padded to 5>`."""
    return f"ep{int(episode_id)}_v{int(view)}_s{int(start):05d}"


def check_frame_count(episode_id: int, decoded: int, expected: int) -> None:
    if int(decoded) != int(expected):
        raise BuildError(f"episode {episode_id}: ffmpeg decoded {decoded} frames, manifest says {expected}")


def check_window_count(episode_id: int, actual: int, expected: int) -> None:
    if int(actual) != int(expected):
        raise BuildError(f"episode {episode_id}: enumerated {actual} windows, manifest says {expected}")


# ---------------------------------------------------------------------------------- D4: encode


def preprocess_frames(frames: np.ndarray) -> np.ndarray:
    """RGB uint8 `[T, H, W, 3]` -> `[1, 3, T, H, W]` float32 in [-1, 1].

    Byte-for-byte the pipeline's preprocessing: `VideoProcessor.preprocess_video` rescales to
    [0, 1], applies `2x - 1`, then moves channels before frames (characterized in
    `test_overfit100_encode.py` against the real processor at 192x320).
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise BuildError(f"expected RGB frames [T, H, W, 3], got shape {tuple(frames.shape)}")
    scaled = frames.astype(np.float32) / 255.0
    normalized = 2.0 * scaled - 1.0
    return np.ascontiguousarray(np.transpose(normalized, (3, 0, 1, 2))[None])  # [B, C, T, H, W]


def _vae_context(mesh=None, logical_axis_rules=None) -> ExitStack:
    """Enter the VAE mesh + logical axis rules exactly as the pipeline does (no-op in tests)."""
    stack = ExitStack()
    if mesh is not None:
        stack.enter_context(mesh)
    if logical_axis_rules is not None:
        from flax.linen import partitioning as nn_partitioning

        stack.enter_context(nn_partitioning.axis_rules(logical_axis_rules))
    return stack


def encode_pixels_to_latents(video, vae, feat_cache, mesh=None, logical_axis_rules=None):
    """`wan_pipeline._encode_video_to_t2v_latents`, verbatim.

    video: `[B, 3, T, H, W]` in [-1, 1]. Returns normalized latents `[B, C_z, T', H', W']`
    (float32, channels-first) via the deterministic posterior MODE -- never `.sample()`.
    """
    vae_dtype = getattr(vae, "dtype", jnp.float32)
    video = jnp.asarray(video).astype(vae_dtype)
    with _vae_context(mesh, logical_axis_rules):
        encoded = vae.encode(video, feat_cache)[0].mode()
    latents_mean = jnp.array(vae.latents_mean).reshape(1, 1, 1, 1, vae.z_dim)
    latents_std = jnp.array(vae.latents_std).reshape(1, 1, 1, 1, vae.z_dim)
    encoded = (encoded.astype(jnp.float32) - latents_mean) / latents_std
    return jnp.transpose(encoded, (0, 4, 1, 2, 3))


def encode_window_latents(
    frames: np.ndarray,
    vae,
    feat_cache,
    *,
    expected_shape: Sequence[int] = Z_VIDEO_SHAPE,
    mesh=None,
    logical_axis_rules=None,
) -> tuple[np.ndarray, np.ndarray]:
    """One window's pixels -> `(z_video, z_i0)` float16; the f16 cast happens LAST."""
    latents = encode_pixels_to_latents(preprocess_frames(frames), vae, feat_cache, mesh, logical_axis_rules)
    z_video = np.asarray(jax.device_get(latents), dtype=np.float32)[0]
    if tuple(z_video.shape) != tuple(expected_shape):
        raise BuildError(f"encoded latents have shape {tuple(z_video.shape)}, expected {tuple(expected_shape)}")
    z_video = z_video.astype(np.float16)
    return z_video, np.ascontiguousarray(z_video[:, :1])


def decode_latents_to_frames(
    z_video, vae, feat_cache, mesh=None, logical_axis_rules=None, postprocess: str = "float32"
) -> np.ndarray:
    """Inverse path (`_denormalize_latents` + `_decode_latents_to_video`) -> `[T, H, W, 3]` in [0, 1].

    `postprocess` (B7, cycle-B review):

    * ``"bfloat16"`` -- pipeline parity: `_decode_latents_to_video` casts to bfloat16 before
      `postprocess_video`, so this is the decode the ROLLOUT evaluator will use. The V3 gate
      value (which doubles as the per-window VAE ceiling at eval) must come from this branch,
      or the "ceiling" would not be the ceiling of the thing being measured.
    * ``"float32"`` -- the same `x / 2 + 0.5` clipped to [0, 1] without the bf16 rounding.
      Used for V1's encode-path isolation and kept as a named V3 diagnostic.
    """
    if postprocess not in ("float32", "bfloat16"):
        raise BuildError(f"unknown postprocess mode {postprocess!r} (expected 'float32' or 'bfloat16')")
    latents = jnp.asarray(np.asarray(z_video, dtype=np.float32))[None]  # [1, C_z, T', H', W']
    latents_mean = jnp.array(vae.latents_mean).reshape(1, vae.z_dim, 1, 1, 1)
    latents_std = 1.0 / jnp.array(vae.latents_std).reshape(1, vae.z_dim, 1, 1, 1)
    latents = (latents / latents_std + latents_mean).astype(jnp.float32)
    with _vae_context(mesh, logical_axis_rules):
        video = vae.decode(latents, feat_cache)[0]
    frames = jnp.asarray(jax.device_get(video), dtype=jnp.float32)[0]  # [T, H, W, 3] in [-1, 1]
    if postprocess == "bfloat16":
        # `_decode_latents_to_video`: float32 -> bfloat16 tensor -> denormalize -> float32 numpy.
        frames = frames.astype(jnp.bfloat16)
        frames = jnp.clip(frames / 2 + 0.5, 0, 1)
        return np.asarray(frames.astype(jnp.float32), dtype=np.float32)
    return np.clip(np.asarray(frames, dtype=np.float32) / 2.0 + 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------------- gates V1-V4


def relative_l2(actual, reference) -> float:
    """`||actual - reference||_2 / ||reference||_2`."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise BuildError(f"relative_l2 shape mismatch: {actual.shape} vs {reference.shape}")
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        raise BuildError("relative_l2 reference has zero norm")
    return float(np.linalg.norm((actual - reference).ravel()) / denominator)


def pearson_r(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise BuildError(f"pearson_r shape mismatch: {a.shape} vs {b.shape}")
    a_centered, b_centered = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denominator == 0.0:
        raise BuildError("pearson_r is undefined for a constant input")
    return float(np.dot(a_centered, b_centered) / denominator)


def check_v1(name: str, reencoded, cached) -> dict:
    """V1: re-encode of a decoded CACHED window vs the cached latents (both halves required)."""
    rel = relative_l2(reencoded, cached)
    corr = pearson_r(reencoded, cached)
    return {
        "name": name,
        "rel_l2": rel,
        "pearson": corr,
        "passed": bool(rel <= V1_REL_L2_MAX and corr >= V1_PEARSON_MIN),
    }


def check_v2(name: str, z_video) -> dict:
    """V2: per-window statistics envelope, evaluated on EVERY built window."""
    z = np.asarray(z_video, dtype=np.float32)
    finite = bool(np.all(np.isfinite(z)))
    mean = float(z.mean()) if finite else float("nan")
    std = float(z.std()) if finite else float("nan")
    passed = bool(finite and V2_STD_MIN <= std <= V2_STD_MAX and abs(mean) <= V2_ABS_MEAN_MAX)
    return {"name": name, "mean": mean, "std": std, "finite": finite, "passed": passed}


def summarize_v2(stats: Sequence[dict]) -> dict:
    """Aggregate the per-window V2 stats; every failure is carried through verbatim."""
    stats = list(stats)
    finite = [s for s in stats if s["finite"]]
    return {
        "n_windows": len(stats),
        "failures": [s for s in stats if not s["passed"]],
        "std_min": min((s["std"] for s in finite), default=float("nan")),
        "std_max": max((s["std"] for s in finite), default=float("nan")),
        "abs_mean_max": max((abs(s["mean"]) for s in finite), default=float("nan")),
        "n_non_finite": len(stats) - len(finite),
    }


def check_v3(name: str, ssim: float) -> dict:
    """V3: SSIM(decode(z_video), source frames). The value doubles as the per-window VAE ceiling."""
    value = float(ssim)
    return {"name": name, "ssim": value, "passed": bool(np.isfinite(value) and value >= V3_SSIM_MIN)}


def check_v4(frame0_full, frame0_short) -> dict:
    """V4: latent frame 0 must not change when the future frames are truncated.

    B2 (cycle-B review): `difference > tolerance` is FALSE for NaN, so finiteness is checked
    EXPLICITLY -- a non-finite short-graph encode must never report "frame 0 is invariant".
    """
    full = np.asarray(frame0_full, dtype=np.float64)
    short = np.asarray(frame0_short, dtype=np.float64)
    if full.shape != short.shape:
        raise BuildError(f"V4 shape mismatch: {full.shape} vs {short.shape}")
    finite = bool(np.all(np.isfinite(full)) and np.all(np.isfinite(short)))
    difference = np.abs(full - short)
    tolerance = V4_ATOL + V4_RTOL * np.abs(short)
    violations = difference > tolerance
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(np.abs(short) > 0, difference / np.abs(short), np.where(difference > 0, np.inf, 0.0))
    max_abs = float(difference.max()) if difference.size else float("nan")
    max_rel = float(np.nanmax(relative)) if relative.size else float("nan")
    norm = float(np.linalg.norm(short.ravel())) if finite else 0.0
    rel_l2 = relative_l2(full, short) if norm else float("nan")
    metrics_finite = bool(np.isfinite(max_abs) and np.isfinite(max_rel))
    return {
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "n_violations": int(violations.sum()),
        "n_elements": int(full.size),
        "rel_l2": rel_l2,
        "finite": finite and metrics_finite,
        "passed": bool(finite and metrics_finite and not violations.any()),
    }


def v4_diagnostics(frame0_full, frame0_short) -> dict:
    """Difference quantiles for a V4 trip -- the evidence a rerun decision is made from (B2)."""
    full = np.asarray(frame0_full, dtype=np.float64)
    short = np.asarray(frame0_short, dtype=np.float64)
    difference = np.abs(full - short).ravel()
    finite = difference[np.isfinite(difference)]
    quantiles = np.quantile(finite, [0.5, 0.9, 0.99, 0.999, 1.0]) if finite.size else [float("nan")] * 5
    result = check_v4(full, short)
    return {
        **result,
        "abs_diff_quantiles": dict(zip(("p50", "p90", "p99", "p999", "p100"), (float(q) for q in quantiles))),
        "n_non_finite": int(difference.size - finite.size),
        "policy": V4_FAILURE_POLICY,
    }


def persist_v4_diagnostics(destination_root: str, frame0_full, frame0_short, tmp_dir: str | Path) -> str:
    """Save both frame-0 tensors + the quantiles next to the build output (B2)."""
    tmp_dir = Path(tmp_dir) / "v4"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local = tmp_dir / "v4_frame0.npz"
    np.savez(
        local,
        frame0_full=np.asarray(frame0_full, dtype=np.float32),
        frame0_short=np.asarray(frame0_short, dtype=np.float32),
    )
    destination = _join(destination_root, "v4_frame0.npz")
    copy_object(str(local), destination)
    write_json(_join(destination_root, "v4_diagnostics.json"), v4_diagnostics(frame0_full, frame0_short))
    return destination


def structural_similarity_fn() -> Callable:
    """scikit-image's SSIM, or a loud failure -- a NaN here would look like a passing gate."""
    try:
        from skimage.metrics import structural_similarity
    except Exception as exc:  # noqa: BLE001 -- any import failure is fatal for a gate
        raise BuildError(f"gate V3 needs scikit-image (`structural_similarity`) but it is unusable: {exc!r}")
    return structural_similarity


def frames_ssim(pred, target, ssim_fn: Callable | None = None) -> float:
    """Mean per-frame SSIM, exp_01 eval parity (`generate_wan_side_adapter._frame_ssim`)."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise BuildError(f"frames_ssim shape mismatch: {pred.shape} vs {target.shape}")
    ssim_fn = ssim_fn or structural_similarity_fn()
    height, width = pred.shape[1:3]
    win = min(7, height, width)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        raise BuildError(f"frames too small for SSIM: {height}x{width}")
    values = [float(ssim_fn(t, p, channel_axis=-1, data_range=1.0, win_size=win)) for p, t in zip(pred, target)]
    return float(np.mean(values))


def gate_failures(report: dict) -> list[str]:
    """Every gate violation, as human-readable lines. A gate that did not run is a failure."""
    messages: list[str] = []

    v1 = report.get("v1")
    if not v1:
        messages.append(f"V1 did not run (expected the {len(TARGET_NAMES)} cached reference windows)")
    else:
        messages += [
            f"V1 {r['name']}: rel_l2={r['rel_l2']:.4f} (max {V1_REL_L2_MAX}), "
            f"pearson={r['pearson']:.4f} (min {V1_PEARSON_MIN})"
            for r in v1
            if not r["passed"]
        ]

    v2 = report.get("v2")
    if not v2:
        messages.append("V2 did not run (no window statistics were collected)")
    else:
        messages += [
            f"V2 {r['name']}: mean={r['mean']:.4f} std={r['std']:.4f} finite={r['finite']} "
            f"(std in [{V2_STD_MIN}, {V2_STD_MAX}], |mean| <= {V2_ABS_MEAN_MAX})"
            for r in v2.get("failures", [])
        ]

    v3 = report.get("v3")
    if not v3:
        messages.append("V3 did not run (no decode-vs-RGB SSIM was computed)")
    else:
        messages += [f"V3 {r['name']}: ssim={r['ssim']:.4f} (min {V3_SSIM_MIN})" for r in v3 if not r["passed"]]

    v4 = report.get("v4")
    if not v4:
        messages.append("V4 did not run (frame-0 future-invariance was never checked)")
    elif not v4["passed"]:
        detail = (
            "non-finite latents" if not v4.get("finite", True) else f"{v4['n_violations']}/{v4['n_elements']} elements"
        )
        messages.append(
            f"V4 frame-0 invariance failed ({detail} exceed rtol {V4_RTOL}; "
            f"max_abs_diff={v4['max_abs_diff']:.3e}, max_rel_diff={v4['max_rel_diff']:.3e}). {V4_FAILURE_POLICY}"
        )
    return messages


def raise_on_gate_failures(report: dict) -> None:
    failures = gate_failures(report)
    if failures:
        raise GateFailure("encoder-validation gates failed (plan R1: stop, no training):\n  " + "\n  ".join(failures))


# ---------------------------------------------------------------------------------- D6: records


def _bytes_feature(value: bytes):
    tf = _tf()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int):
    tf = _tf()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _checked_array(name: str, array, shape: Sequence[int]) -> np.ndarray:
    array = np.asarray(array)
    if tuple(array.shape) != tuple(shape):
        raise BuildError(f"{name}: shape {tuple(array.shape)} != {tuple(shape)}")
    if array.dtype != np.float16:
        raise BuildError(f"{name}: dtype {array.dtype} != float16")
    return np.ascontiguousarray(array)


def serialize_window_record(
    *,
    name: str,
    episode_id: int,
    episode_index: int,
    window_start: int,
    z_i0,
    z_video,
    instruction: str,
) -> bytes:
    """One schema-v2 `tf.train.Example` (plan D6). No `actions` field."""
    tf = _tf()
    features = {
        "name": _bytes_feature(str(name).encode("utf-8")),
        "episode_id": _int64_feature(episode_id),
        "episode_index": _int64_feature(episode_index),
        "window_start": _int64_feature(window_start),
        "z_i0": _bytes_feature(_checked_array("z_i0", z_i0, Z_I0_SHAPE).tobytes()),
        "z_video": _bytes_feature(_checked_array("z_video", z_video, Z_VIDEO_SHAPE).tobytes()),
        "instruction": _bytes_feature(str(instruction).encode("utf-8")),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def parse_window_record(raw: bytes) -> dict:
    """Inverse of `serialize_window_record` -- the readback used by tests and rung-3 checks."""
    tf = _tf()
    feature = tf.train.Example.FromString(raw).features.feature
    missing = [key for key in SCHEMA_V2_FIELDS if key not in feature]
    if missing:
        raise ValueError(f"record is missing schema-v2 fields {missing}")

    def _array(key: str, shape: Sequence[int]) -> np.ndarray:
        payload = feature[key].bytes_list.value[0]
        expected = int(np.prod(shape)) * 2
        if len(payload) != expected:
            raise ValueError(f"{key}: expected {expected} bytes for {tuple(shape)} float16, got {len(payload)}")
        return np.frombuffer(payload, dtype=np.float16).reshape(shape).copy()

    return {
        "name": feature["name"].bytes_list.value[0].decode("utf-8"),
        "episode_id": int(feature["episode_id"].int64_list.value[0]),
        "episode_index": int(feature["episode_index"].int64_list.value[0]),
        "window_start": int(feature["window_start"].int64_list.value[0]),
        "z_i0": _array("z_i0", Z_I0_SHAPE),
        "z_video": _array("z_video", Z_VIDEO_SHAPE),
        "instruction": feature["instruction"].bytes_list.value[0].decode("utf-8"),
    }


def in_train10(episode_index: int) -> bool:
    return int(episode_index) < TRAIN10_MAX_EPISODE_INDEX


def select_train10(records: Iterable[dict]) -> list[dict]:
    return [record for record in records if in_train10(record["episode_index"])]


def expected_window_counts(manifest: dict) -> dict:
    """The window counts both built sets must contain, straight from the manifest (D6/G3)."""
    episodes = manifest["episodes"]
    return {
        "train100": sum(int(e["n_windows"]) for e in episodes),
        "train10": sum(int(e["n_windows"]) for e in episodes if in_train10(e["episode_index"])),
    }


def shard_ranges(n: int, shard_size: int = SHARD_SIZE) -> list[tuple[int, int]]:
    if shard_size <= 0:
        raise BuildError(f"shard_size must be positive, got {shard_size}")
    return [(start, min(start + shard_size, n)) for start in range(0, int(n), shard_size)]


def shard_filename(prefix: str, index: int, total: int) -> str:
    return f"{prefix}-{int(index):05d}-of-{int(total):05d}.tfrecord"


# ---------------------------------------------------------------------------------- audit


def duplicate_instruction_groups(episodes: Sequence[dict]) -> list[dict]:
    """Exact-duplicate `used_text` groups, largest first (plan §1 duplicate-condition audit)."""
    counts = Counter(episode["used_text"] for episode in episodes)
    groups = []
    for text, count in counts.items():
        if count < 2:
            continue
        members = [e for e in episodes if e["used_text"] == text]
        groups.append(
            {
                "used_text": text,
                "count": int(count),
                "episode_indices": [int(e["episode_index"]) for e in members],
                "episode_ids": [int(e["episode_id"]) for e in members],
            }
        )
    return sorted(groups, key=lambda g: (-g["count"], g["used_text"]))


def target_key(z_video) -> str:
    """Content address of a target: two windows share a target iff these keys are equal."""
    return hashlib.sha1(np.ascontiguousarray(z_video).tobytes()).hexdigest()


def min_pairwise_z_i0(
    vectors,
    target_keys: Sequence[str],
    chunk_size: int = AUDIT_CHUNK,
    histogram_n: int = AUDIT_HISTOGRAM_N,
) -> dict:
    """Minimum pairwise `z_i0` L2 distance over window pairs with DIFFERENT targets.

    Distances come from a float64 Gram expansion: the interesting values are near-duplicate
    inputs, where a float32 expansion loses most of the significant digits to cancellation.
    """
    matrix = np.asarray(vectors, dtype=np.float64).reshape(len(vectors), -1)
    if len(target_keys) != len(matrix):
        raise BuildError(f"{len(target_keys)} target keys for {len(matrix)} windows")
    identifiers: dict[str, int] = {}
    keys = np.asarray([identifiers.setdefault(k, len(identifiers)) for k in target_keys], dtype=np.int64)
    squared = np.einsum("ij,ij->i", matrix, matrix)

    smallest: list[tuple[float, int, int]] = []
    pairs_compared = 0
    for start in range(0, len(matrix), max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), len(matrix))
        block = matrix[start:end]
        distances2 = squared[start:end, None] + squared[None, :] - 2.0 * (block @ matrix.T)
        rows = np.arange(start, end)[:, None]
        columns = np.arange(len(matrix))[None, :]
        valid = (columns > rows) & (keys[start:end, None] != keys[None, :])
        pairs_compared += int(valid.sum())
        if not valid.any():
            continue
        distances = np.sqrt(np.maximum(distances2, 0.0))
        row_index, column_index = np.nonzero(valid)
        candidates = distances[row_index, column_index]
        order = np.argsort(candidates, kind="stable")[:histogram_n]
        smallest += [(float(candidates[k]), int(start + row_index[k]), int(column_index[k])) for k in order]
    smallest.sort(key=lambda item: (item[0], item[1], item[2]))
    smallest = smallest[:histogram_n]
    return {
        "n_windows": int(len(matrix)),
        "n_pairs_compared": pairs_compared,
        "min_distance": smallest[0][0] if smallest else None,
        "argmin_pair": [smallest[0][1], smallest[0][2]] if smallest else None,
        "smallest_pairs": [{"i": i, "j": j, "distance": d} for d, i, j in smallest],
    }


# ---------------------------------------------------------------------------------- IO helpers


def _is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")


def _join(root: str, *parts: str) -> str:
    return "/".join([str(root).rstrip("/"), *[str(p).strip("/") for p in parts]])


def write_json(destination: str, payload: dict, log: Callable = print) -> None:
    """Write a sidecar to a local path or GCS (staged + one sequential `gsutil cp`)."""
    text = json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n"
    if _is_gcs(destination):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write(text)
            staged = handle.name
        try:
            run_gsutil(["cp", staged, destination])
        finally:
            os.unlink(staged)
    else:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(text)
    log(f"[build] wrote {destination}")


def list_objects(prefix: str) -> list[str]:
    """Every object under `prefix` (recursive), local path or GCS. Missing prefix -> []."""
    if _is_gcs(prefix):
        proc = run_gsutil(["ls", "-r", _join(prefix, "**")], check=False)
        stderr = proc.stderr.decode("utf-8", "replace")
        if proc.returncode != 0:
            if "matched no objects" in stderr or "No URLs matched" in stderr:
                return []
            raise BuildError(f"cannot list {prefix}: {stderr.strip()[:400]}")
        lines = proc.stdout.decode("utf-8", "replace").splitlines()
        return sorted(line.strip() for line in lines if line.strip().startswith("gs://") and not line.endswith(":"))
    root = Path(prefix)
    return sorted(str(path) for path in root.rglob("*") if path.is_file()) if root.is_dir() else []


def canonical_objects(prefix: str) -> list[str]:
    """Objects a reader would see under the canonical prefix -- staging is invisible to them."""
    return [uri for uri in list_objects(prefix) if f"/{STAGING_PREFIX}" not in uri]


def copy_object(source: str, destination: str) -> None:
    if _is_gcs(source) or _is_gcs(destination):
        run_gsutil(["cp", source, destination])
        return
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def remove_object(uri: str) -> None:
    if _is_gcs(uri):
        run_gsutil(["rm", "-f", uri], check=False)
        return
    Path(uri).unlink(missing_ok=True)


def stat_object(uri: str) -> dict:
    """Remote generation/md5/size for a written object (empty dict for local paths)."""
    if not _is_gcs(uri):
        return {}
    from maxdiffusion.data_preprocessing.extract_v1_fixture import gsutil_stat_many

    resolved = gsutil_stat_many([uri])[uri]
    if not resolved.ok:
        raise BuildError(f"{uri}: written object could not be statted ({resolved.error or resolved.status})")
    return resolved.fingerprint


def fetch_pinned(uri: str, fingerprint: dict, destination: Path) -> Path:
    """Download the EXACT statted generation and bind the bytes to the recorded md5/size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    proc = run_gsutil(["cp", pinned_uri(uri, fingerprint), str(destination)], check=False)
    if proc.returncode != 0 or not destination.exists():
        detail = proc.stderr.decode("utf-8", "replace").strip()[:400]
        raise BuildError(f"{uri}: pinned download failed (gsutil exit {proc.returncode}): {detail}")
    errors = verify_payload_binding(uri, destination.read_bytes(), fingerprint)
    if errors:
        raise BuildError("; ".join(errors))
    return destination


def decode_mp4_frames(path: str | Path, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> np.ndarray:
    """ffmpeg -> RGB24 `[T, H, W, 3]` uint8, one frame per stored frame (no fps resampling)."""
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-vsync",
        "0",  # passthrough: never duplicate/drop frames to satisfy a frame rate
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.run(command, capture_output=True, check=False)
    if proc.returncode != 0:
        raise BuildError(f"{path}: ffmpeg exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:400]}")
    frame_bytes = int(width) * int(height) * 3
    if not proc.stdout or len(proc.stdout) % frame_bytes:
        raise BuildError(
            f"{path}: ffmpeg produced {len(proc.stdout)} bytes, not a whole number of "
            f"{width}x{height} RGB frames ({frame_bytes} bytes each)"
        )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, int(height), int(width), 3)


def peak_rss_bytes() -> int:
    """`ru_maxrss` is bytes on macOS and kilobytes on Linux."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def device_memory_stats() -> list[dict]:
    """Peak-memory stats for EVERY local device (B5). Devices without stats are reported as-is."""
    stats: list[dict] = []
    try:
        devices = jax.local_devices()
    except Exception:  # noqa: BLE001 -- no backend at all
        return stats
    for device in devices:
        entry = {"device": str(device)}
        try:
            entry.update(dict(device.memory_stats() or {}))
        except Exception as exc:  # noqa: BLE001 -- recorded, then judged by the caller
            entry["error"] = repr(exc)
        stats.append(entry)
    return stats


def require_device_memory_stats(stats: Sequence[dict]) -> list[dict]:
    """A probe that cannot show per-device peak memory has not substantiated its claim (B5)."""
    stats = list(stats)
    if not stats:
        raise BuildError("probe requires device memory stats but no local device reported any")
    missing = [entry.get("device", "?") for entry in stats if "peak_bytes_in_use" not in entry]
    if missing:
        raise BuildError(f"probe requires peak memory from every local device; missing for {missing}")
    return stats


def audit_benchmark(
    n: int = AUDIT_BENCHMARK_WINDOWS,
    dim: int = AUDIT_BENCHMARK_DIM,
    chunk_size: int = AUDIT_CHUNK,
    log: Callable = print,
) -> dict:
    """Time+size the FULL-SCALE duplicate audit on synthetic data, BEFORE any expensive encode.

    B5: the real audit runs after 1,629 windows have been encoded; discovering there that the
    float64 Gram expansion is too slow or too large would waste the whole build.
    """
    started = time.time()
    vectors = np.random.default_rng(0).normal(size=(int(n), int(dim))).astype(np.float16)
    result = min_pairwise_z_i0(vectors, [f"t{i}" for i in range(int(n))], chunk_size=chunk_size)
    elapsed = time.time() - started
    del vectors
    benchmark = {
        "n": int(n),
        "dim": int(dim),
        "seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "n_pairs_compared": result["n_pairs_compared"],
    }
    log(f"[build] audit benchmark: {n}x{dim} float64 in {elapsed:.1f}s, peak RSS {peak_rss_bytes() / 2**30:.2f} GiB")
    return benchmark


class ShardWriter:
    """Buffers serialized records into shards under a STAGING prefix; one shard resident at a time.

    Every shard is fingerprinted at write time (sha256 + size of the exact local bytes) and
    statted after upload, so `readback_set` can prove the published bytes are the written bytes
    (B4). Nothing is written to the canonical prefix here -- promotion happens after readback.
    """

    def __init__(
        self,
        prefix: str,
        staging_dir: str,
        total_shards: int,
        local_dir: Path,
        shard_size: int = SHARD_SIZE,
        log: Callable = print,
    ):
        self.prefix, self.staging_dir, self.total_shards = prefix, staging_dir, int(total_shards)
        self.local_dir, self.shard_size, self.log = Path(local_dir), int(shard_size), log
        self.buffer: list[bytes] = []
        self.written = 0
        self.shards: list[dict] = []
        self.upload_seconds = 0.0

    def add(self, record: bytes) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        index = len(self.shards)
        if index >= self.total_shards:
            raise BuildError(f"{self.prefix}: more shards than the {self.total_shards} planned from the manifest")
        name = shard_filename(self.prefix, index, self.total_shards)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        local = self.local_dir / name
        tf = _tf()
        with tf.io.TFRecordWriter(str(local)) as writer:
            for record in self.buffer:
                writer.write(record)
        payload = local.read_bytes()
        entry = {
            "name": name,
            "records": len(self.buffer),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
        destination = _join(self.staging_dir, name)
        started = time.time()
        copy_object(str(local), destination)
        entry.update(stat_object(destination))
        self.upload_seconds += time.time() - started
        local.unlink(missing_ok=True)
        entry["staging_uri"] = destination
        self.shards.append(entry)
        self.written += len(self.buffer)
        self.log(f"[build] staged {destination} ({len(self.buffer)} records, sha256 {entry['sha256'][:12]})")
        self.buffer = []

    def close(self, expected: int) -> None:
        self.flush()
        if self.written != int(expected):
            raise BuildError(f"{self.prefix}: wrote {self.written} records, manifest expects {expected}")
        if len(self.shards) != self.total_shards:
            raise BuildError(f"{self.prefix}: wrote {len(self.shards)} shards, planned {self.total_shards}")


# ---------------------------------------------------------------------------------- B4: readback


def fetch_shard_bytes(entry: dict, tmp_dir: str | Path) -> bytes:
    """Read one staged shard back off the target filesystem, PINNED to the written generation."""
    uri = entry["staging_uri"]
    if not _is_gcs(uri):
        return Path(uri).read_bytes()
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local = tmp_dir / f"readback_{entry['name']}"
    source = pinned_uri(uri, entry) if entry.get("generation") is not None else uri
    proc = run_gsutil(["cp", source, str(local)], check=False)
    if proc.returncode != 0 or not local.exists():
        raise BuildError(f"{uri}: readback download failed: {proc.stderr.decode('utf-8', 'replace')[:400]}")
    try:
        return local.read_bytes()
    finally:
        local.unlink(missing_ok=True)


def assert_readback_names(entries: Sequence[dict], actual: Sequence[str], expected: Sequence[str]) -> None:
    """The published records must be exactly the intended windows, in order (B4)."""
    if list(actual) != list(expected):
        first = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), min(len(actual), len(expected)))
        raise BuildError(
            f"readback names differ from the build order at index {first} "
            f"({len(actual)} read vs {len(expected)} expected) over {len(entries)} shard(s)"
        )


def readback_set(
    set_name: str,
    entries: Sequence[dict],
    expected_names: Sequence[str],
    tmp_dir: str | Path,
    log: Callable = print,
) -> dict[str, str]:
    """Physically re-read every staged shard and re-assert the whole schema-v2 contract (B4).

    Returns `name -> payload sha256` so the caller can prove train10 is byte-identical to its
    train100 counterparts rather than an independent (possibly divergent) encode.
    """
    tf = _tf()
    names: list[str] = []
    payloads: dict[str, str] = {}
    for entry in entries:
        data = fetch_shard_bytes(entry, tmp_dir)
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            raise BuildError(
                f"{set_name}/{entry['name']}: readback sha256 {digest[:12]} != written {entry['sha256'][:12]} "
                "-- the published bytes are not the bytes this build wrote"
            )
        local = Path(tmp_dir) / f"parse_{entry['name']}"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        try:
            raw_records = [bytes(record.numpy()) for record in tf.data.TFRecordDataset([str(local)])]
        finally:
            local.unlink(missing_ok=True)
        if len(raw_records) != entry["records"]:
            raise BuildError(f"{set_name}/{entry['name']}: read {len(raw_records)} records, wrote {entry['records']}")
        for raw in raw_records:
            try:
                record = parse_window_record(raw)
            except ValueError as exc:
                raise BuildError(f"{set_name}/{entry['name']}: unparsable record ({exc})") from exc
            if record["z_i0"].tobytes() != np.ascontiguousarray(record["z_video"][:, :1]).tobytes():
                raise BuildError(f"{set_name}/{record['name']}: z_i0 is not bitwise z_video[:, :1]")
            names.append(record["name"])
            payloads[record["name"]] = hashlib.sha256(raw).hexdigest()
    assert_readback_names(entries, names, expected_names)
    log(f"[build] readback OK: {set_name} -- {len(names)} records over {len(entries)} shard(s)")
    return payloads


def assert_subset_is_byte_identical(subset: dict[str, str], superset: dict[str, str], subset_name: str) -> None:
    """train10 must be the SAME records as train100, not a second encode of the same windows."""
    missing = [name for name in subset if name not in superset]
    if missing:
        raise BuildError(f"{subset_name}: {len(missing)} records are absent from train100 (e.g. {missing[:3]})")
    differing = [name for name, digest in subset.items() if superset[name] != digest]
    if differing:
        raise BuildError(
            f"{subset_name}: {len(differing)} records differ byte-wise from train100 (e.g. {differing[:3]})"
        )


# ---------------------------------------------------------------------------------- B3: promotion


def build_identifier(commit: str, now: datetime | None = None) -> str:
    """`<12-char commit>-<UTC stamp>` -- unique per attempt, and it names the code that ran."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{str(commit)[:12]}-{stamp}"


def require_empty_canonical(prefix: str) -> None:
    """A canonical set prefix must be empty before promotion -- never silently overwrite (B3)."""
    existing = canonical_objects(prefix)
    if existing:
        raise BuildError(
            f"{prefix} is not empty ({len(existing)} object(s), e.g. {existing[:3]}). "
            "Refusing to overwrite a published dataset: delete it deliberately or build elsewhere."
        )


def promote(staging_dir: str, canonical_dir: str, names: Sequence[str], log: Callable = print) -> list[str]:
    """Copy the enumerated objects staging -> canonical; on failure remove only what we created."""
    require_empty_canonical(canonical_dir)
    created: list[str] = []
    try:
        for name in names:
            destination = _join(canonical_dir, name)
            copy_object(_join(staging_dir, name), destination)
            created.append(destination)
        log(f"[build] promoted {len(created)} object(s) -> {canonical_dir}")
        return created
    except BaseException:
        for uri in created:
            remove_object(uri)
        log(f"[build] promotion failed; removed {len(created)} partially promoted object(s)")
        raise


# ---------------------------------------------------------------------------------- stages


def load_manifest(path: str, expected_episodes: int | None = EXPECTED_EPISODES) -> dict:
    manifest = json.loads(Path(path).read_text())
    errors = validate_manifest_structure(manifest, expected_episodes=expected_episodes)
    if errors:
        raise BuildError(f"{path} failed its structural gate:\n  " + "\n  ".join(errors))
    return manifest


def preflight(
    manifest: dict,
    tmp_dir: Path,
    snapshot_dir: str | None = None,
    local_files_only: bool = False,
    log: Callable = print,
) -> dict:
    """Manifest drift, V1 fixture binding + structure, and the MANDATORY VAE pin (B1).

    Nothing is encoded and no weights are loaded until every one of these passes.
    """
    drift = verify_manifest(manifest)
    if drift:
        raise BuildError("manifest fingerprints drifted -- the source data changed:\n  " + "\n  ".join(drift))
    log(
        f"[build] manifest verified: {manifest['totals']['episodes']} episodes / {manifest['totals']['windows']} windows"
    )

    fixture = manifest["fixture"]
    local = fetch_pinned(fixture["uri"], {**fixture, "size": fixture["size_bytes"]}, tmp_dir / "v1_fixture.npz")
    errors = verify_fixture(local, fixture)
    if errors:
        raise BuildError("V1 fixture preflight failed:\n  " + "\n  ".join(errors))
    log(f"[build] V1 fixture verified: {list(fixture['names'])}")

    # B1: the pin is MANDATORY and it binds the weights, not just the report. The revision is
    # resolved to one local snapshot directory, that directory is fingerprinted, and the SAME
    # directory is handed to the loader -- so "the VAE we checked" and "the VAE we loaded"
    # cannot diverge. There is no warn-and-continue path.
    pin = manifest.get("vae_fingerprint")
    errors = vae_pin_errors(pin)
    if errors:
        raise BuildError("the manifest carries no usable vae_fingerprint pin:\n  " + "\n  ".join(errors))
    resolved = resolve_vae_snapshot(
        snapshot_dir or pin["hf_repo"], pin["revision"], local_files_only=bool(local_files_only)
    )
    observed = resolved["pin"]
    if snapshot_dir:  # a pre-staged directory cannot prove a repo/revision; its BYTES must match
        observed = {**observed, "hf_repo": pin["hf_repo"], "revision": pin["revision"]}
    mismatch = {key: (pin[key], observed.get(key)) for key in pin if pin[key] != observed.get(key)}
    if mismatch:
        raise BuildError(f"the resolved VAE does not match the manifest pin: {mismatch}")
    log(f"[build] VAE pin verified: {pin['hf_repo']}@{pin['revision']} -> {resolved['snapshot_path']}")
    return {
        "fixture_path": local,
        "vae_fingerprint": dict(pin),
        "vae_snapshot_path": resolved["snapshot_path"],
    }


def assert_manifest_matches_committed(
    path: str | Path, repo_root: str | Path | None = None, log: Callable = print
) -> str:
    """The consumed manifest must be byte-identical to the committed artifact (B6).

    On a worker the code arrives as an uploaded TARBALL rather than a checkout, so "the
    committed artifact" is the copy shipped inside that tarball -- and the tarball IS the
    launch-time tree whose cleanliness the launcher verified before submitting (see
    `deployed_code_commit`). The comparison therefore still runs whenever the shipped copy
    exists, so a hand-edited manifest passed by path is still caught; only when no reference
    was shipped does deployed-code mode fall back to hashing and RECORDING the shipped
    manifest, with the reasoning logged. In a real worktree a missing reference stays fatal.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    reference = root / DEFAULT_MANIFEST
    consumed = Path(path).read_bytes()
    actual = hashlib.sha256(consumed).hexdigest()
    deployed = not is_git_worktree(root)
    if not reference.is_file():
        if not deployed:
            raise BuildError(f"{reference}: the committed manifest is missing; cannot verify {path}")
        # T2: the trust boundary is the UPLOADED TREE. A manifest from anywhere else carries
        # none of the launcher's clean-tree verification, so it may not be recorded as "the
        # shipped manifest". `resolve()` also collapses symlinks that would escape the root.
        try:
            Path(path).resolve().relative_to(root.resolve())
        except ValueError:
            raise BuildError(
                f"{path}: resolves outside the deployed-code root {root}. In deployed-code mode only a "
                "manifest shipped inside the uploaded tarball may be recorded as provenance."
            ) from None
        log(
            "[provenance] deployed-code mode: no committed manifest shipped alongside the code; the "
            "uploaded tarball IS the launch-time tree the launcher verified clean, so recording the "
            f"shipped manifest sha256={actual}"
        )
        return actual
    expected = hashlib.sha256(reference.read_bytes()).hexdigest()
    if expected != actual:
        raise BuildError(
            f"{path}: sha256 {actual} does not match the committed manifest ({expected}). "
            "The build may only consume the manifest that is committed at HEAD."
        )
    if deployed:
        log(f"[provenance] deployed-code mode: consumed manifest matches the shipped copy (sha256={actual})")
    return actual


def load_vae_pipeline(config_path: str, snapshot_path: str, overrides: Sequence[str] = (), log: Callable = print):
    """VAE-only Wan2.2 TI2V pipeline (no transformer, no text encoder) from a PINNED snapshot."""
    from maxdiffusion import pyconfig
    from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2

    argv = [__file__, str(config_path), f"pretrained_model_name_or_path={snapshot_path}", *overrides]
    log(f"[build] pyconfig.initialize({argv[1:]})")
    pyconfig.initialize(argv)
    pipeline = WanPipelineTI2V_2_2.from_checkpoint(pyconfig.config, restored_checkpoint=None, vae_only=True)
    log(f"[build] VAE loaded from the pinned snapshot (z_dim={pipeline.vae.z_dim})")
    return pipeline


def run_v1_gate(pipeline, fixture_path: Path, log: Callable = print) -> list[dict]:
    """Decode each cached reference window, re-encode it through OUR path, compare (plan D4)."""
    names, windows = load_fixture_npz(fixture_path)
    results = []
    for name in names:
        cached = windows[name]["z_video"]
        frames = decode_latents_to_frames(
            cached, pipeline.vae, pipeline.vae_cache, pipeline.vae_mesh, pipeline.vae_logical_axis_rules
        )
        # Back through the same door the build uses: uint8 RGB -> preprocess -> encode.
        pixels = np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)
        reencoded, _ = encode_window_latents(
            pixels,
            pipeline.vae,
            pipeline.vae_cache,
            mesh=pipeline.vae_mesh,
            logical_axis_rules=pipeline.vae_logical_axis_rules,
        )
        result = check_v1(name, reencoded.astype(np.float32), np.asarray(cached, dtype=np.float32))
        log(
            f"[build] V1 {name}: rel_l2={result['rel_l2']:.4f} pearson={result['pearson']:.4f} passed={result['passed']}"
        )
        results.append(result)
    return results


def run_v4_gate(
    pipeline, frames: np.ndarray, diagnostics_root: str | None = None, tmp_dir: str | Path = ".", log: Callable = print
) -> dict:
    """Encode `[0, 33)` and `[0, 17)`; latent frame 0 must be identical (no future leakage).

    On failure BOTH frame-0 tensors and the difference quantiles are persisted next to the
    build output: a V4 trip is judged from saved arrays, never re-thresholded in place (B2).
    """
    context = {"mesh": pipeline.vae_mesh, "logical_axis_rules": pipeline.vae_logical_axis_rules}
    full = encode_pixels_to_latents(
        preprocess_frames(frames[:WINDOW_FRAMES]), pipeline.vae, pipeline.vae_cache, **context
    )
    short = encode_pixels_to_latents(
        preprocess_frames(frames[:V4_SHORT_FRAMES]), pipeline.vae, pipeline.vae_cache, **context
    )
    full = np.asarray(jax.device_get(full), dtype=np.float32)[0][:, :1]
    short = np.asarray(jax.device_get(short), dtype=np.float32)[0][:, :1]
    result = check_v4(full, short)
    log(
        f"[build] V4 frame-0 invariance: max_abs_diff={result['max_abs_diff']:.3e} "
        f"max_rel_diff={result['max_rel_diff']:.3e} violations={result['n_violations']}/{result['n_elements']} "
        f"finite={result['finite']} passed={result['passed']}"
    )
    if not result["passed"] and diagnostics_root:
        saved = persist_v4_diagnostics(diagnostics_root, full, short, tmp_dir)
        result["diagnostics_uri"] = saved
        log(f"[build] V4 diagnostics saved to {saved}; {V4_FAILURE_POLICY}")
    return result


def run_v3_window(pipeline, name: str, pixels: np.ndarray, z_video: np.ndarray, log: Callable = print) -> dict:
    """V3 for one window: SSIM(decode(z_video), source frames), bf16-parity with the evaluator.

    B7: the GATED number uses the pipeline's bfloat16 postprocess so it really is the ceiling
    the rollout metrics are measured against; the float32 number rides along as a diagnostic.
    """
    context = {"mesh": pipeline.vae_mesh, "logical_axis_rules": pipeline.vae_logical_axis_rules}
    source = pixels.astype(np.float32) / 255.0
    decoded_bf16 = decode_latents_to_frames(
        z_video, pipeline.vae, pipeline.vae_cache, postprocess="bfloat16", **context
    )
    decoded_f32 = decode_latents_to_frames(z_video, pipeline.vae, pipeline.vae_cache, postprocess="float32", **context)
    result = check_v3(name, frames_ssim(decoded_bf16, source))
    result["ssim_float32_diagnostic"] = frames_ssim(decoded_f32, source)
    result["postprocess"] = "bfloat16"
    log(
        f"[build] V3 {name}: ssim={result['ssim']:.4f} (bf16 parity; f32 diagnostic "
        f"{result['ssim_float32_diagnostic']:.4f}) passed={result['passed']}"
    )
    return result


def _fail_fast(report: dict) -> None:
    """Raise on any gate that has ALREADY run and failed; gates still pending are not failures."""
    raise_on_gate_failures(
        {
            "v1": report["v1"] or [{"name": "pending", "rel_l2": 0.0, "pearson": 1.0, "passed": True}],
            "v2": report["v2"] or {"n_windows": 0, "failures": []},
            "v3": report["v3"] or [{"name": "pending", "ssim": 1.0, "passed": True}],
            "v4": report["v4"] or {"passed": True},
        }
    )


def _plan_sets(manifest: dict, episodes: list[dict], probe: bool) -> dict[str, dict]:
    """Which built set gets which episodes, and how many windows each must end up with (D6/G3)."""
    counts = expected_window_counts(manifest)
    subset_windows = sum(int(e["n_windows"]) for e in episodes)
    if probe:
        return {"probe2": {"indices": {int(e["episode_index"]) for e in episodes}, "expected": subset_windows}}
    if subset_windows != counts["train100"]:
        raise BuildError(f"episode subset carries {subset_windows} windows, manifest totals {counts['train100']}")
    return {
        "train100": {"indices": {int(e["episode_index"]) for e in episodes}, "expected": counts["train100"]},
        "train10": {
            "indices": {int(e["episode_index"]) for e in episodes if in_train10(e["episode_index"])},
            "expected": counts["train10"],
        },
    }


class PhaseTimer:
    """Wall-clock per named build phase (B5: fixed cost and steady-state must be separable)."""

    def __init__(self):
        self.phases: dict[str, float] = {}
        self._started = time.time()

    def record(self, name: str, seconds: float) -> None:
        self.phases[name] = self.phases.get(name, 0.0) + float(seconds)

    class _Scope:
        def __init__(self, timer, name):
            self.timer, self.name = timer, name

        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *exc):
            self.timer.record(self.name, time.time() - self.start)
            return False

    def phase(self, name: str) -> "PhaseTimer._Scope":
        return PhaseTimer._Scope(self, name)

    def total(self) -> float:
        return time.time() - self._started


def _probe_v3_precheck(pipeline, manifest: dict, tmp_dir: Path, log: Callable = print) -> list[dict]:
    """Run the FIXED V3 sample set (manifest index 0, 10, ..., 90) before the full build (B5).

    The probe otherwise builds only episodes 0-1, so a V3 failure at index 90 would surface
    after the entire 1,629-window production encode. Ten extra MP4s is a cheap insurance.
    """
    results = []
    for episode in manifest["episodes"]:
        index = int(episode["episode_index"])
        if index not in V3_EPISODE_INDICES:
            continue
        episode_id = int(episode["episode_id"])
        fingerprint = episode["video_fingerprint"]
        local_mp4 = tmp_dir / f"v3_ep{episode_id}.mp4"
        try:
            fetch_pinned(fingerprint["uri"], fingerprint, local_mp4)
            frames = decode_mp4_frames(local_mp4)
        finally:
            local_mp4.unlink(missing_ok=True)
        check_frame_count(episode_id, len(frames), int(episode["ffprobe"]["nb_frames"]))
        pixels = slice_window(frames, 0)
        z_video, _ = encode_window_latents(
            pixels,
            pipeline.vae,
            pipeline.vae_cache,
            mesh=pipeline.vae_mesh,
            logical_axis_rules=pipeline.vae_logical_axis_rules,
        )
        results.append(run_v3_window(pipeline, window_name(episode_id, 0, VIEW_INDEX), pixels, z_video, log=log))
    log(f"[build] probe V3 precheck covered {len(results)} fixed windows")
    return results


def run(args) -> int:
    """Preflight -> encode -> gates -> staged writes -> readback -> promote -> _SUCCESS."""

    def log(message: str) -> None:
        print(message, flush=True)

    timer = PhaseTimer()
    probe = bool(args.probe)
    shard_size = int(args.shard_size) or (PROBE_SHARD_SIZE if probe else SHARD_SIZE)

    manifest = load_manifest(args.manifest, expected_episodes=args.expected_episodes)
    episodes = list(manifest["episodes"])[: args.probe_episodes] if probe else list(manifest["episodes"])

    # B6: probes are guarded exactly like production -- same clean-commit check, same manifest
    # content hash, and a real recorded SHA (a probe's numbers are evidence for a launch).
    manifest_sha256 = (
        "unverified (--dry-run)" if args.dry_run else assert_manifest_matches_committed(args.manifest, log=log)
    )
    build_commit = None
    if not args.dry_run:
        # In a worktree this proves the tree is clean at HEAD; from a deployed tarball it
        # relays the sha the launcher verified (there is no repository on the worker).
        build_commit = assert_implementation_committed(paths=CYCLE_B_IMPLEMENTATION_PATHS, log=log)
        log(f"[build] build_commit={build_commit}")
    build_id = build_identifier(build_commit or "dryrun")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="exp02_build_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with timer.phase("preflight"):
        preflight_result = preflight(
            manifest, tmp_dir, args.vae_snapshot_dir, local_files_only=not args.allow_hub_download, log=log
        )
    if args.dry_run:
        log("[build] --dry-run: preflight only, nothing encoded or written")
        return 0

    sets = _plan_sets(manifest, episodes, probe)
    report_root = _join(args.out_root, "probe2") if probe else args.out_root
    staging = {name: _join(args.out_root, name, f"{STAGING_PREFIX}{build_id}") for name in sets}
    for name in sets:  # B3: refuse before the expensive work, not after
        require_empty_canonical(_join(args.out_root, name))
    writers = {
        name: ShardWriter(
            name,
            staging[name],
            len(shard_ranges(plan["expected"], shard_size)),
            tmp_dir / "staging",
            shard_size=shard_size,
            log=log,
        )
        for name, plan in sets.items()
    }
    log("[build] planned sets: " + ", ".join(f"{name}={plan['expected']} windows" for name, plan in sets.items()))

    # B5: the full-scale audit is benchmarked BEFORE the VAE is loaded, so a memory/time
    # surprise costs a few seconds instead of the entire production encode.
    benchmark = None
    if probe:
        with timer.phase("audit_benchmark"):
            benchmark = audit_benchmark(log=log)

    with timer.phase("vae_load"):
        pipeline = load_vae_pipeline(args.config, preflight_result["vae_snapshot_path"], args.config_override, log=log)
    context = {"mesh": pipeline.vae_mesh, "logical_axis_rules": pipeline.vae_logical_axis_rules}

    report: dict = {"v1": [], "v2": {}, "v3": [], "v4": None}
    window_stats: list[dict] = []
    episode_records: list[dict] = []
    z_i0_rows: list[np.ndarray] = []
    target_keys: list[str] = []
    ordered_names: dict[str, list[str]] = {name: [] for name in sets}
    v3_seen: set[int] = set()
    encoded_windows = 0

    try:
        with timer.phase("v1_gate"):
            report["v1"] = run_v1_gate(pipeline, preflight_result["fixture_path"], log=log)
        _fail_fast(report)

        if probe:
            with timer.phase("v3_precheck"):
                report["v3"] = _probe_v3_precheck(pipeline, manifest, tmp_dir, log=log)
                v3_seen = {
                    int(e["episode_index"])
                    for e in manifest["episodes"]
                    if int(e["episode_index"]) in V3_EPISODE_INDICES
                }
            _fail_fast(report)

        for position, episode in enumerate(episodes, start=1):
            episode_id, episode_index = int(episode["episode_id"]), int(episode["episode_index"])
            fingerprint = episode["video_fingerprint"]
            local_mp4 = tmp_dir / f"ep{episode_id}_v{VIEW_INDEX}.mp4"
            with timer.phase("download_decode"):
                try:
                    fetch_pinned(fingerprint["uri"], fingerprint, local_mp4)
                    frames = decode_mp4_frames(local_mp4)
                finally:
                    local_mp4.unlink(missing_ok=True)  # storage guardrail: one MP4 at a time
            check_frame_count(episode_id, len(frames), int(episode["ffprobe"]["nb_frames"]))
            if tuple(frames.shape[1:3]) != (FRAME_HEIGHT, FRAME_WIDTH):
                raise BuildError(
                    f"episode {episode_id}: decoded geometry {tuple(frames.shape[1:3])} != {(FRAME_HEIGHT, FRAME_WIDTH)}"
                )
            starts = window_starts(len(frames))
            check_window_count(episode_id, len(starts), int(episode["n_windows"]))

            if episode_index == 0:
                with timer.phase("v4_gate"):
                    report["v4"] = run_v4_gate(pipeline, frames, report_root, tmp_dir, log=log)

            for start in starts:
                name = window_name(episode_id, start, VIEW_INDEX)
                pixels = slice_window(frames, start)
                started = time.time()
                z_video, z_i0 = encode_window_latents(pixels, pipeline.vae, pipeline.vae_cache, **context)
                # The first window pays XLA compilation; the rest is the steady-state rate the
                # full-build extrapolation is allowed to use (B5).
                timer.record(
                    "first_window_compile" if encoded_windows == 0 else "steady_state_encode", time.time() - started
                )
                encoded_windows += 1
                window_stats.append({**check_v2(name, z_video), "episode_index": episode_index, "window_start": start})
                if episode_index in V3_EPISODE_INDICES and start == 0 and episode_index not in v3_seen:
                    with timer.phase("v3_gate"):
                        report["v3"].append(run_v3_window(pipeline, name, pixels, z_video, log=log))
                    v3_seen.add(episode_index)

                record = serialize_window_record(
                    name=name,
                    episode_id=episode_id,
                    episode_index=episode_index,
                    window_start=start,
                    z_i0=z_i0,
                    z_video=z_video,
                    instruction=episode["used_text"],
                )
                for set_name, plan in sets.items():
                    if episode_index in plan["indices"]:
                        writers[set_name].add(record)
                        ordered_names[set_name].append(name)
                z_i0_rows.append(np.asarray(z_i0, dtype=np.float32).ravel())
                target_keys.append(target_key(z_video))

            episode_records.append(
                {
                    "episode_index": episode_index,
                    "episode_id": episode_id,
                    "used_text": episode["used_text"],
                    "n_windows": len(starts),
                    "nb_frames": int(episode["ffprobe"]["nb_frames"]),
                    "video_uri": fingerprint["uri"],
                }
            )
            report["v2"] = summarize_v2(window_stats)
            log(
                f"[build] episode {position}/{len(episodes)} (index {episode_index}, id {episode_id}): {len(starts)} windows"
            )
            _fail_fast(report)

        report["v2"] = summarize_v2(window_stats)
        expected_v3 = {
            i for i in V3_EPISODE_INDICES if any(int(e["episode_index"]) == i for e in manifest["episodes"])
        }
        covered = {int(entry["name"].split("_")[0][2:]) for entry in report["v3"]}
        by_id = {int(e["episode_id"]): int(e["episode_index"]) for e in manifest["episodes"]}
        if {by_id[episode_id] for episode_id in covered} != expected_v3:
            raise GateFailure(
                f"V3 covered episode indices {sorted(by_id[e] for e in covered)}, expected {sorted(expected_v3)}"
            )
        raise_on_gate_failures(report)

        for name, plan in sets.items():
            writers[name].close(plan["expected"])
        timer.record("upload", sum(writer.upload_seconds for writer in writers.values()))

        # B4: prove the PUBLISHED bytes are the bytes this build wrote, before promoting.
        with timer.phase("readback"):
            payloads = {
                name: readback_set(name, writers[name].shards, ordered_names[name], tmp_dir / "readback", log=log)
                for name in sets
            }
        if "train10" in payloads and "train100" in payloads:
            assert_subset_is_byte_identical(payloads["train10"], payloads["train100"], "train10")

        with timer.phase("audit"):
            audit = {
                "duplicate_instruction_groups": duplicate_instruction_groups(episodes),
                "min_pairwise_z_i0": min_pairwise_z_i0(np.stack(z_i0_rows), target_keys),
                "n_windows": len(z_i0_rows),
                "note": "L2 over z_i0 (the conditioning input); pairs that share a target are excluded",
            }
    except BaseException as exc:  # noqa: BLE001 -- every abort must leave a diagnosable trail
        report["v2"] = report["v2"] or summarize_v2(window_stats)
        try:
            write_json(
                _join(report_root, "failed_gates.json"),
                {"error": str(exc), "build_id": build_id, "gates": report, "staging": staging},
                log=log,
            )
        except Exception as write_error:  # noqa: BLE001 -- never mask the original failure
            log(f"[build] could not write failed_gates.json: {write_error!r}")
        raise

    steady_windows = max(encoded_windows - 1, 0)
    steady_rate = steady_windows / timer.phases.get("steady_state_encode", 0.0) if steady_windows else float("nan")
    timing = {
        "phases": {**timer.phases, "total": timer.total()},
        "windows": encoded_windows,
        "steady_state_windows_per_second": steady_rate,
        "peak_rss_bytes": peak_rss_bytes(),
        "device_memory_stats": require_device_memory_stats(device_memory_stats()) if probe else device_memory_stats(),
        "audit_benchmark": benchmark,
    }
    summary = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": build_id,
        "build_commit": build_commit,
        "probe": probe,
        "manifest": {
            "path": str(args.manifest),
            "sha256": manifest_sha256,
            "md5": md5_b64(Path(args.manifest).read_bytes()),
            "selection_seed": manifest["selection_seed"],
            "builder_commit": manifest["builder_commit"],
        },
        "encode": {
            "rng": "none -- deterministic posterior mode(); no sampling",
            "window_frames": WINDOW_FRAMES,
            "window_stride": WINDOW_STRIDE,
            "geometry": {"height": FRAME_HEIGHT, "width": FRAME_WIDTH},
            "z_video_shape": list(Z_VIDEO_SHAPE),
            "z_i0_shape": list(Z_I0_SHAPE),
            "config": str(args.config),
            "config_overrides": list(args.config_override),
            "shard_size": shard_size,
        },
        "vae_fingerprint": preflight_result["vae_fingerprint"],
        "vae_snapshot_path": preflight_result["vae_snapshot_path"],
        "sets": {
            name: {
                "uri": _join(args.out_root, name),
                "expected_windows": plan["expected"],
                "written": writers[name].written,
                "shards": writers[name].shards,
            }
            for name, plan in sets.items()
        },
        "episodes": len(episode_records),
        "gates": report,
        "gate_policy": {"v4": V4_FAILURE_POLICY},
        "readers_must_require": SUCCESS_MARKER,
        "duplicate_audit": {
            "groups": len(audit["duplicate_instruction_groups"]),
            "episodes_in_groups": sum(g["count"] for g in audit["duplicate_instruction_groups"]),
            "min_pairwise_z_i0": audit["min_pairwise_z_i0"]["min_distance"],
            "argmin_pair": audit["min_pairwise_z_i0"]["argmin_pair"],
        },
        "tool_versions": collect_tool_versions(),
        "timing": timing,
    }
    if probe:
        full_windows = expected_window_counts(manifest)["train100"]
        fixed = sum(timer.phases.get(key, 0.0) for key in ("preflight", "vae_load", "first_window_compile", "v1_gate"))
        per_window_upload = timer.phases.get("upload", 0.0) / max(encoded_windows, 1)
        per_window_download = timer.phases.get("download_decode", 0.0) / max(encoded_windows, 1)
        variable = (
            full_windows * (1.0 / steady_rate + per_window_upload + per_window_download) if steady_rate else None
        )
        summary["v2_coverage"] = f"sampled ({encoded_windows}/{full_windows} windows)"
        summary["extrapolation"] = {
            "full_build_windows": full_windows,
            "steady_state_windows_per_second": steady_rate,
            "fixed_seconds": fixed,
            "audit_seconds": (benchmark or {}).get("seconds"),
            "variable_seconds": variable,
            "estimated_seconds": (
                None if variable is None else fixed + variable + (benchmark or {}).get("seconds", 0.0)
            ),
            "estimated_hours": (
                None if variable is None else (fixed + variable + (benchmark or {}).get("seconds", 0.0)) / 3600.0
            ),
        }
        log(
            f"[build] PROBE: {encoded_windows} windows, steady {steady_rate:.2f} w/s -> full build "
            f"~{summary['extrapolation']['estimated_hours']:.2f} h (fixed {fixed:.0f}s + audit "
            f"{(benchmark or {}).get('seconds', 0):.0f}s); peak RSS {timing['peak_rss_bytes'] / 2**30:.2f} GiB; "
            f"devices {timing['device_memory_stats']}"
        )

    # B3: sidecars land in staging, the whole set is promoted, then _SUCCESS is written LAST.
    with timer.phase("promote"):
        for set_name, plan in sets.items():
            members = [e for e in episode_records if e["episode_index"] in plan["indices"]]
            stats = [s for s in window_stats if s["episode_index"] in plan["indices"]]
            summary_bytes = json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n"
            write_json(_join(staging[set_name], "summary.json"), summary, log=log)
            write_json(_join(staging[set_name], "episodes.json"), {"episodes": members}, log=log)
            write_json(_join(staging[set_name], "window_stats.json"), {"windows": stats}, log=log)
            names = [shard["name"] for shard in writers[set_name].shards] + [
                "summary.json",
                "episodes.json",
                "window_stats.json",
            ]
            promote(staging[set_name], _join(args.out_root, set_name), names, log=log)
            write_json(
                _join(args.out_root, set_name, SUCCESS_MARKER),
                {
                    "build_id": build_id,
                    "build_commit": build_commit,
                    "records": writers[set_name].written,
                    "shards": len(writers[set_name].shards),
                    "summary_sha256": hashlib.sha256(summary_bytes.encode("utf-8")).hexdigest(),
                    "manifest_sha256": manifest_sha256,
                },
                log=log,
            )
            for name in names:  # staging is scratch; the canonical copy is now authoritative
                remove_object(_join(staging[set_name], name))
            if not _is_gcs(staging[set_name]):
                shutil.rmtree(staging[set_name], ignore_errors=True)
        write_json(_join(report_root, "duplicate_audit.json"), audit, log=log)

    shutil.rmtree(tmp_dir / "staging", ignore_errors=True)
    shutil.rmtree(tmp_dir / "readback", ignore_errors=True)
    log(f"[build] done in {timer.total():.1f}s: " + ", ".join(f"{name}={writers[name].written}" for name in sets))
    return 0


# ---------------------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the exp_02 overfit100 latent datasets.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="pyconfig base yml for the VAE-only pipeline")
    parser.add_argument("--config-override", action="append", default=[], help="extra pyconfig key=value")
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--expected-episodes", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--probe", action="store_true", help="first 2 episodes -> <out-root>/probe2/ only")
    parser.add_argument("--probe-episodes", type=int, default=PROBE_EPISODES)
    parser.add_argument("--shard-size", type=int, default=0, help="records per shard (0 = production/probe default)")
    parser.add_argument(
        "--vae-snapshot-dir",
        default=None,
        help="pre-staged snapshot directory; its vae/config.json sha256 must still match the manifest pin",
    )
    parser.add_argument(
        "--allow-hub-download",
        action="store_true",
        help="permit resolving the pinned revision from the hub (default: local cache only, prefetched)",
    )
    parser.add_argument("--dry-run", action="store_true", help="preflight only: no encode, no writes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run(args)
    except (BuildError, DirtyImplementationError) as exc:
        print(f"[build] ABORT: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
