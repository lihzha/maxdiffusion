"""Decoded-pixel metrics for exp_04 null-text inversion (plan §4-P1 metrics, round R7).

R6's ``emit_metric_tables`` emits ``future_mse`` and leaves ``future_ssim`` out, which the gates read
as *invalid* rather than as merely absent. This module is what fills it in: decode, score, fill --
after which the tables are gate-ready. Nothing here loads a VAE; ``decode_fn`` is injected.

**The decode seam.** ``decode_fn(latents [B, 48, 9, 12, 20] float32) -> [B, 33, H, W, 3]`` floats in
``[0, 1]``. The caller owns denormalization and the VAE: the runner passes a wrapper around
``pipeline._decode_latents_to_video(pipeline._denormalize_latents(z))``, exactly as
``generate_wan_side_adapter`` does at its decode site (lines 348-349). That module is deliberately
**not** imported -- importing it sets ``jax_use_shardy_partitioner`` globally as an import side
effect -- so the two small pieces borrowed from it are reimplemented here and cited instead.

**The estimand is reconstruction of the decoded ground truth**, not of the original pixels: the
reference video is ``decode_fn(z_video)``, so a perfect method scores 1.0 by construction and the VAE
round-trip is not silently charged to the method.

**The frame mapping.** The Wan VAE upsamples at temporal stride 4: 9 latent frames become
``1 + 4 * 8 = 33`` pixel frames, and latent frame 0 -- the pinned image condition -- maps to pixel
frame 0 **only**. The pinned content is therefore exactly one pixel frame, and the future-frame
metrics average pixel frames 1..32. Future-frame SSIM is the primary pixel metric (plan §4-P1);
full-frame SSIM and the two pixel MSEs ride along as secondaries.

**SSIM semantics are the deployed ones.** ``skimage.metrics.structural_similarity`` with
``channel_axis=-1``, ``data_range=1.0`` and ``win_size = min(7, H, W)`` forced odd, arguments passed
target-first -- a faithful restatement of ``generate_wan_side_adapter._frame_ssim`` (lines 255-279),
so numbers from this round are comparable with the pre_context validation numbers already recorded.
**The metric layer does not clip**, because the reference does not: pixels arrive already in range
(the pipeline's postprocessor clamps, and ``decode_and_score`` enforces ``[0, 1]`` exactly at the
seam), so clipping here could only ever hide a wiring bug while quietly shifting SSIM -- a 5e-4
excursion measurably moves it. One deviation stands: where the reference returns NaN (skimage
missing, or an image too small for a 3-pixel window) this module raises. A NaN would reach the gates
as an ordinary invalid observation, and an environment or geometry fault should not look like a
badly reconstructed example.

**Non-finite latents are refused before any decode.** A decoder is free to turn NaN into finite
pixels -- an input-independent one would score such an arm as a *perfect* reconstruction -- so the
check belongs on the way in, not on the way out, and it covers the ground truth and every arm and
probe seed before the first VAE pass is spent.

**Seed keys are restated, not imported.** ``PROBE_K_SET`` / ``SINGLE_SEED_KEY`` mirror
``null_adapter_runner_core``'s so this module stays numpy-only and never pulls in jax (the R4b
ruling for the record codec); a test pins the two definitions equal.
"""

from __future__ import annotations

import os
import shutil
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from maxdiffusion.null_adapter_records import PRODUCTION_GEOMETRY


LATENT_FRAMES = PRODUCTION_GEOMETRY.z_video[1]  # 9
TEMPORAL_STRIDE = 4  # the Wan VAE's temporal upsampling factor
PIXEL_FRAMES = 1 + TEMPORAL_STRIDE * (LATENT_FRAMES - 1)  # 33: frame 0, then four per later latent frame
PIXEL_CHANNELS = 3
PIXEL_RANGE = (0.0, 1.0)  # enforced exactly at the decode seam: no excursion is tolerated
SSIM_MAX_WIN, SSIM_MIN_WIN = 7, 3
PROBE_K_SET = (0, 1, 2)
SINGLE_SEED_KEY = "0"
PROBE_SUFFIX = "_probe"
PIXEL_METRICS = ("future_ssim", "full_ssim", "future_pixel_mse", "full_pixel_mse")


def _checked_video(video: Any, label: str) -> np.ndarray:
    value = np.asarray(video, dtype=np.float32)
    if value.ndim != 4 or value.shape[-1] != PIXEL_CHANNELS:
        raise ValueError(f"{label} must be [F, H, W, 3] pixel frames, got shape {value.shape}")
    return value


def _ssim_window(height: int, width: int) -> int:
    """``min(7, H, W)`` forced odd -- ``generate_wan_side_adapter._frame_ssim``'s rule, restated."""
    window = min(SSIM_MAX_WIN, height, width)
    window -= window % 2 == 0
    if window < SSIM_MIN_WIN:
        raise ValueError(f"frames of {height}x{width} cannot carry a window of at least {SSIM_MIN_WIN} pixels")
    return window


def frame_ssim_series(pred: Any, gt: Any) -> np.ndarray:
    """Per-frame SSIM of ``pred`` against ``gt``, both ``[F, H, W, 3]`` in ``[0, 1]``.

    Returns a ``[F]`` float64 array. The values are passed to skimage exactly as they arrive --
    **no clipping**, as in the reference -- and target-first, also as in the reference (SSIM is
    symmetric in the two, so the order is mirrored only to keep the two implementations diffable).
    """
    pred, gt = _checked_video(pred, "pred"), _checked_video(gt, "gt")
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:  # louder than the reference's NaN: this is an environment fault
        raise ImportError("scikit-image is required for the exp_04 pixel metrics (SSIM)") from error

    window = _ssim_window(pred.shape[1], pred.shape[2])
    low, high = PIXEL_RANGE
    return np.array(
        [
            structural_similarity(g, p, channel_axis=-1, data_range=high - low, win_size=window)
            for p, g in zip(pred, gt)
        ],
        dtype=np.float64,
    )


def future_frame_ssim(series: Any) -> float:
    """Mean SSIM over pixel frames 1..F-1: everything except the pinned first frame."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"the SSIM series must be 1-D with at least two frames, got shape {values.shape}")
    return float(values[1:].mean())


def full_ssim(series: Any) -> float:
    """Mean SSIM over every pixel frame, the pinned one included (secondary; see the module docstring)."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"the SSIM series must be 1-D and non-empty, got shape {values.shape}")
    return float(values.mean())


def _pixel_mse(pred: Any, gt: Any, *, drop_frame_zero: bool) -> float:
    """Unclipped, like the reference and like the SSIM above; float64 so 23k squares do not drift."""
    pred, gt = _checked_video(pred, "pred"), _checked_video(gt, "gt")
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")
    if drop_frame_zero:
        pred, gt = pred[1:], gt[1:]
    return float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))


def future_frame_pixel_mse(pred: Any, gt: Any) -> float:
    """Pixel MSE over frames 1.., i.e. the same future-frame window the SSIM primary uses."""
    return _pixel_mse(pred, gt, drop_frame_zero=True)


def full_pixel_mse(pred: Any, gt: Any) -> float:
    return _pixel_mse(pred, gt, drop_frame_zero=False)


def _score_pair(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    series = frame_ssim_series(pred, gt)
    return {
        "future_ssim": future_frame_ssim(series),
        "full_ssim": full_ssim(series),
        "future_pixel_mse": future_frame_pixel_mse(pred, gt),
        "full_pixel_mse": full_pixel_mse(pred, gt),
    }


def _checked_latents(latents: Any, label: str, batch: int) -> np.ndarray:
    value = np.asarray(latents, dtype=np.float32)
    if value.shape != (batch, *PRODUCTION_GEOMETRY.z_video):
        raise ValueError(
            f"{label} must have the production shape {(batch, *PRODUCTION_GEOMETRY.z_video)}, got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        # A NaN latent is not a badly reconstructed example, and it cannot be left for the output
        # check to catch: a decoder may map it to perfectly finite pixels, and an input-independent
        # one would then hand the gates a *perfect* SSIM for a corrupt arm.
        raise ValueError(f"{label} must be finite: a non-finite latent cannot be scored")
    return value


def _decoded(decode_fn: Callable[[np.ndarray], Any], latents: np.ndarray) -> np.ndarray:
    """One decode, with the seam's output contract enforced before a single metric is computed."""
    pixels = np.asarray(decode_fn(latents), dtype=np.float32)
    if pixels.ndim != 5 or (pixels.shape[0], pixels.shape[1], pixels.shape[4]) != (
        latents.shape[0],
        PIXEL_FRAMES,
        PIXEL_CHANNELS,
    ):
        raise ValueError(f"decode_fn must return [B, {PIXEL_FRAMES}, H, W, 3] pixel videos, got shape {pixels.shape}")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("decode_fn returned non-finite pixels")
    low, high = PIXEL_RANGE
    if pixels.min() < low or pixels.max() > high:
        # Exactly, with no excursion allowance: the pipeline's postprocessor already clamps, so any
        # value outside the range is a wiring bug (the [-1, 1] convention, a missing postprocess),
        # and forgiving small excursions would only move SSIM silently -- 5e-4 is already measurable.
        raise ValueError(
            f"decode_fn returned pixels outside [0, 1] (min {pixels.min():g}, max {pixels.max():g}): the "
            f"decoder must already map to [0, 1], not [-1, 1]"
        )
    return pixels


def decode_and_score(
    decode_fn: Callable[[np.ndarray], Any],
    final_latents_by_arm: Mapping[str, Any],
    z_video: Any,
    names: Sequence[str],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Decode the ground truth once, decode every arm, and score them frame by frame.

    Args:
      decode_fn: the seam documented at the top of this module.
      final_latents_by_arm: ``ArmResults.final_latents`` -- ``[B, ...]`` per single-noise arm and
        ``[len(PROBE_K_SET), B, ...]`` per probe arm, whose leading axis indexes ``PROBE_K_SET``.
      z_video: the ``[B, 48, 9, 12, 20]`` target latents; ``decode_fn(z_video)`` is the reference.
      names: the batch's manifest names, one per row.

    Returns:
      ``metrics[method][name][seed key] -> {future_ssim, full_ssim, future_pixel_mse, full_pixel_mse}``
      -- the shape ``fill_pixel_metrics`` merges into R6's tables.
    """
    names = tuple(names)
    if not names:
        raise ValueError("batch must carry at least one example")
    if len(set(names)) != len(names):
        raise ValueError(f"batch names must be unique, got {list(names)}")
    z_video = _checked_latents(z_video, "z_video", len(names))

    # Every array is checked before the first decode: a malformed arm should not cost a VAE pass.
    planned = []
    for method, latents in final_latents_by_arm.items():
        stacked = np.asarray(latents, dtype=np.float32)
        stacked = stacked if stacked.ndim == 6 else stacked[None]
        keys = [str(k) for k in PROBE_K_SET] if method.endswith(PROBE_SUFFIX) else [SINGLE_SEED_KEY]
        if stacked.shape[0] != len(keys):
            raise ValueError(f"{method} must carry {len(keys)} seed(s), got a leading axis of {stacked.shape[0]}")
        checked = [_checked_latents(stacked[seed], method, len(names)) for seed in range(len(keys))]
        planned.append((method, keys, checked))

    ground_truth = _decoded(decode_fn, z_video)  # once, and once only: every arm scores against this
    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for method, keys, per_seed in planned:
        decoded = [_decoded(decode_fn, latents) for latents in per_seed]
        metrics[method] = {
            name: {key: _score_pair(decoded[seed][index], ground_truth[index]) for seed, key in enumerate(keys)}
            for index, name in enumerate(names)
        }
    return metrics


def fill_pixel_metrics(tables: Mapping[str, Mapping], pixel_metrics: Mapping[str, Mapping]) -> dict[str, dict]:
    """Merge the pixel metrics into R6's latent tables, completing them for the gates.

    Fail-closed on the namespace: the methods, the names and the seed keys must match exactly. A
    partial fill would leave some observations without ``future_ssim``, and the gates cannot tell that
    apart from a run that genuinely produced none -- it would simply impute against the claim.

    Only finiteness is required of the values. A *negative* SSIM is a real measurement of a bad
    reconstruction, and the [0, 1] semantics belong to the gates (which clip for G3 and treat
    out-of-range probe values as invalid), not to the writer.
    """
    if set(tables) != set(pixel_metrics):
        raise ValueError(f"pixel metrics cover methods {sorted(pixel_metrics)}, tables have {sorted(tables)}")
    filled: dict[str, dict] = {}
    for method, table in tables.items():
        pixels = pixel_metrics[method]
        if set(table) != set(pixels):
            raise ValueError(f"{method}: pixel metrics cover names {sorted(pixels)}, table has {sorted(table)}")
        filled[method] = {}
        for name, seeds in table.items():
            if set(seeds) != set(pixels[name]):
                raise ValueError(
                    f"{method}/{name}: pixel metrics carry seed keys {sorted(pixels[name])}, "
                    f"table has {sorted(seeds)}"
                )
            filled[method][name] = {}
            for key, entry in seeds.items():
                measured = {metric: float(pixels[name][key][metric]) for metric in PIXEL_METRICS}
                for metric, value in measured.items():
                    if not np.isfinite(value):
                        raise ValueError(f"{method}/{name}/{key}: {metric} must be finite, got {value}")
                filled[method][name][key] = {**dict(entry), **measured}
    return filled


def comparison_video_frames(gt: Any, pred: Any) -> np.ndarray:
    """Ground truth on top, prediction underneath -- the repo's comparison layout.

    Mirrors ``generate_wan_side_adapter``'s ``np.concatenate([gt0, pred0], axis=1)`` at its video-saving
    site, so a comparison clip from this experiment reads the same way as the side-adapter ones.
    """
    gt, pred = _checked_video(gt, "gt"), _checked_video(pred, "pred")
    if gt.shape != pred.shape:
        raise ValueError(f"gt and pred must have the same shape, got {gt.shape} and {pred.shape}")
    return np.concatenate([gt, pred], axis=1)


def _imageio():
    """Indirection so the fallback path is reachable in a test without breaking the import."""
    try:
        import imageio

        return imageio
    except ImportError:
        return None


def _as_uint8(frames: Any) -> np.ndarray:
    low, high = PIXEL_RANGE
    return (np.clip(_checked_video(frames, "frames"), low, high) * 255.0 + 0.5).astype(np.uint8)


def save_video_mp4(frames: Any, path: str, fps: int = 16) -> str:
    """Write ``[F, H, W, 3]`` frames in ``[0, 1]`` to ``path``; the only function here that touches disk.

    Uses imageio's ffmpeg writer, as ``maxdiffusion.utils.export_to_video`` does. When imageio or its
    ffmpeg plugin is missing -- common on a bare host -- it falls back to writing the frames as a PNG
    sequence in ``<path without suffix>_frames/`` rather than failing the run, because these clips are
    diagnostics and losing the pixels costs more than losing the container.

    Returns the path actually written (the mp4, or the frame directory), so a caller never has to
    assume which happened.

    **Publication is transactional and never silent.** The mp4 is encoded to a staged name and only
    renamed into place once the last frame is in, so the published path never holds a half-encoded
    clip -- not even briefly, which cleanup-after-the-fact could not promise. A failure removes the
    staged file *and* any previously published clip at that path (afterwards the path holds this
    call's output or nothing, never the last run's), warns with the original exception rather than
    swallowing it, and falls through to a freshly recreated frame directory -- recreated, because a
    shorter rewrite into a stale one would otherwise publish the tail of the previous clip.

    Note for the runner: the ffmpeg writer spawns a subprocess, and forking out of a live JAX process
    raises the "os.fork() was called ... JAX is multithreaded" RuntimeWarning (seen in this round's
    tests). ``maxdiffusion.utils.export_to_video`` has the same property, so this is inherited rather
    than introduced -- but video writing belongs after the compute, on one host, never inside a loop.
    """
    frames = _as_uint8(frames)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    module = _imageio()
    if module is not None:
        staged = f"{path}.partial.mp4"
        try:
            with module.get_writer(staged, format="FFMPEG", fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
            os.replace(staged, path)
            return path
        except Exception as error:  # noqa: BLE001 -- any backend failure degrades to the frame sequence
            for leftover in (staged, path):
                if os.path.exists(leftover):
                    os.remove(leftover)
            warnings.warn(
                f"mp4 backend failed ({error!r}); writing a PNG frame sequence instead",
                RuntimeWarning,
                stacklevel=2,
            )

    directory = os.path.splitext(path)[0] + "_frames"
    if os.path.exists(directory):
        if not os.path.isdir(directory):
            raise RuntimeError(f"cannot publish frames: {directory} exists and is not a directory")
        shutil.rmtree(directory)  # fresh, so no frame of a previous clip survives into this one
    os.makedirs(directory)
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("no video backend: install imageio[ffmpeg] (preferred) or pillow") from error
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(directory, f"frame_{index:04d}.png"))
    return directory
