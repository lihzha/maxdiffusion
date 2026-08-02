"""exp_03 Mechanism A — per-frame SSIM DECAY SLOPES, trial vs control (plan v3.1).

The exp_02 D1 reading established the phenomenon: per-frame SSIM starts around 0.97 for every window
and decays along the rollout. exp_03 asks whether an objective that trains on the eval's own
trajectory flattens that decay, so the statistic here is a **slope**, not a mean:

* **per window** — ordinary least squares of SSIM on the frame index over frames
  ``1 .. T-1`` (frame 0 is the pinned image condition: it is free, and including it would
  contaminate every fit with the same constant);
* **reduction** — ``1 - mean_slope_trial / mean_slope_control``. Both slopes are negative, so a
  trial that decays half as fast scores ``0.5`` and a trial that decays faster scores negative. The
  predeclared threshold is ``>= 0.25``;
* **uncertainty** — a PAIRED per-episode bootstrap (10,000 resamples, 95% CI, seed 0). Paired,
  because the two arms are read on the same 100 canonical windows and the per-episode difference is
  where the noise cancels; per-episode, because the episode is the unit that was sampled.

The exp_02 self-validation check is **retained**: frames are decoded from lossy MP4s, so
mean-over-frames must keep tracking the SSIM the eval recorded for the same window. The frame-index
trend, not the absolute level, is the finding.

This module is deliberately dependency-light (numpy + ffmpeg) and computes no verdict: it reads the
rendered videos and the published aggregation of a pass and prints/returns a diagnostic reading.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# Predeclared, before any trial ran.
D1_FIRST_FRAME = 1  # frame 0 is the pinned condition; the fit starts at frame 1
D1_REDUCTION_THRESHOLD = 0.25
D1_BOOTSTRAP_RESAMPLES = 10000
D1_BOOTSTRAP_SEED = 0
D1_CI = 0.95

W, H = 320, 192


# ------------------------------------------------------------------------------- video + SSIM
# Ported verbatim from the exp_02 D1 script so the absolute SSIM levels stay comparable with the
# landed exp_02 reading (uniform 7x7 filter, K1=0.01/K2=0.03, sample covariance, crop pad=3).


def decode(path: Path) -> np.ndarray:
    """mp4 -> float32 ``[T, H, W, 3]`` in [0, 1]."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    n = len(out) // (W * H * 3)
    return np.frombuffer(out, np.uint8)[: n * W * H * 3].reshape(n, H, W, 3).astype(np.float32) / 255.0


def _box(a: np.ndarray, k: int = 7) -> np.ndarray:
    """Valid-mode uniform filter over the two spatial axes (``a``: ``[H, W]``)."""
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def ssim_frame(x: np.ndarray, y: np.ndarray, k: int = 7, data_range: float = 1.0) -> float:
    """``skimage.metrics.structural_similarity(channel_axis=-1, data_range=1.0, win_size=7)``."""
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    NP = k * k
    cov_norm = NP / (NP - 1.0)
    vals = []
    for ch in range(x.shape[-1]):
        a, b = x[..., ch], y[..., ch]
        ux, uy = _box(a, k), _box(b, k)
        uxx, uyy, uxy = _box(a * a, k), _box(b * b, k), _box(a * b, k)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
        vals.append(S.mean())
    return float(np.mean(vals))


def per_frame_ssim(pred: np.ndarray, target: np.ndarray) -> list[float]:
    """SSIM per frame over the frames both videos have."""
    t = min(len(pred), len(target))
    return [ssim_frame(pred[i], target[i]) for i in range(t)]


# ------------------------------------------------------------------------------------- slopes


def window_slope(values: Sequence[float], first_frame: int = D1_FIRST_FRAME) -> float:
    """OLS slope of SSIM on the FRAME INDEX over ``values[first_frame:]``.

    The regressor is the frame index itself (``first_frame, first_frame+1, ...``), so the slope is
    "SSIM lost per frame" and is comparable across windows of equal length regardless of where the
    fit starts.
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f"per-frame SSIM must be one-dimensional; got shape {series.shape}")
    fitted = series[first_frame:]
    if fitted.size < 2:
        raise ValueError(
            f"a slope needs at least 2 frames after frame {first_frame}; got {fitted.size} "
            f"(window length {series.size})"
        )
    x = np.arange(first_frame, series.size, dtype=np.float64)
    x_centered = x - x.mean()
    return float(np.dot(x_centered, fitted - fitted.mean()) / np.dot(x_centered, x_centered))


def slope_reduction(trial_slopes: Sequence[float], control_slopes: Sequence[float]) -> float:
    """``1 - mean_slope_trial / mean_slope_control`` (the plan's reduction formula)."""
    trial_mean = float(np.mean(np.asarray(trial_slopes, dtype=np.float64)))
    control_mean = float(np.mean(np.asarray(control_slopes, dtype=np.float64)))
    if control_mean == 0.0:
        raise ValueError(
            "the control's mean slope is exactly 0, so the reduction ratio is undefined; report the "
            "two slopes instead of a ratio"
        )
    return 1.0 - trial_mean / control_mean


def paired_bootstrap_reduction(
    pairs: Sequence[Mapping[str, float]],
    *,
    resamples: int = D1_BOOTSTRAP_RESAMPLES,
    seed: int = D1_BOOTSTRAP_SEED,
    ci: float = D1_CI,
) -> dict:
    """Paired per-episode bootstrap of the reduction statistic.

    Each entry is one episode's ``{"episode_id", "trial_slope", "control_slope"}``. A resample draws
    EPISODES with replacement and recomputes the reduction from the episodes drawn, keeping each
    episode's trial and control slopes together -- the pairing is the variance reduction, and
    breaking it (resampling the two arms independently) would widen the interval for no reason.
    """
    if not pairs:
        raise ValueError("the paired bootstrap needs at least one episode")
    trial = np.asarray([float(pair["trial_slope"]) for pair in pairs], dtype=np.float64)
    control = np.asarray([float(pair["control_slope"]) for pair in pairs], dtype=np.float64)
    point = slope_reduction(trial, control)

    rng = np.random.default_rng(int(seed))
    n = trial.size
    draws = rng.integers(0, n, size=(int(resamples), n))
    trial_means = trial[draws].mean(axis=1)
    control_means = control[draws].mean(axis=1)
    usable = control_means != 0.0
    values = 1.0 - trial_means[usable] / control_means[usable]
    if values.size == 0:
        raise ValueError("every bootstrap resample had a zero mean control slope; the ratio is undefined")
    alpha = (1.0 - float(ci)) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        "reduction": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci": float(ci),
        "resamples": int(resamples),
        "seed": int(seed),
        "n_episodes": int(n),
        "n_usable_resamples": int(values.size),
        "mean_slope_trial": float(trial.mean()),
        "mean_slope_control": float(control.mean()),
        "threshold": D1_REDUCTION_THRESHOLD,
        "meets_threshold": bool(point >= D1_REDUCTION_THRESHOLD),
    }


def self_validation(per_window: Mapping[str, Sequence[float]], recorded_ssim: Mapping[str, float]) -> dict:
    """Retained from exp_02: mean-over-frames (from the lossy MP4s) vs the eval's recorded SSIM."""
    checks = [
        (name, float(np.mean(values)), float(recorded_ssim[name]))
        for name, values in sorted(per_window.items())
        if name in recorded_ssim
    ]
    if not checks:
        return {"n": 0, "mean_abs_diff": float("nan"), "max_abs_diff": float("nan")}
    diffs = [abs(from_video - recorded) for _, from_video, recorded in checks]
    return {
        "n": len(checks),
        "mean_abs_diff": float(np.mean(diffs)),
        "max_abs_diff": float(np.max(diffs)),
        "mean_from_video": float(np.mean([value for _, value, _ in checks])),
        "mean_recorded": float(np.mean([value for _, _, value in checks])),
    }


# ---------------------------------------------------------------------------------------- I/O


def per_window_slopes(video_root: Path, first_frame: int = D1_FIRST_FRAME) -> dict[str, dict]:
    """``{window_name: {"slope", "per_frame", "episode_id"}}`` from a pass's rendered videos."""
    out: dict[str, dict] = {}
    for directory in sorted(p for p in Path(video_root).iterdir() if p.is_dir()):
        gt_path, pred_path = directory / "ground_truth.mp4", directory / "sample.mp4"
        if not (gt_path.exists() and pred_path.exists()):
            continue
        values = per_frame_ssim(decode(pred_path), decode(gt_path))
        out[directory.name] = {
            "per_frame": values,
            "slope": window_slope(values, first_frame),
            "episode_id": _episode_id_from_name(directory.name),
        }
    return out


def _episode_id_from_name(name: str) -> int:
    """``ep<episode_id>_v<n>_s<start>`` -> ``episode_id`` (the bootstrap's resampling unit)."""
    head = name.split("_", 1)[0]
    if not head.startswith("ep"):
        raise ValueError(f"cannot read an episode id out of the window directory name {name!r}")
    return int(head[2:])


def recorded_ssim_from_aggregation(path: Path, *, seed: int = 0, context_mode: str = "correct") -> dict[str, float]:
    payload = json.loads(Path(path).read_text())
    return {
        str(row["name"]): float(row["ssim"])
        for row in payload["rows"]
        if int(row["seed"]) == seed and str(row["context_mode"]) == context_mode
    }


def compare(trial_root: Path, control_root: Path) -> dict:
    """The full Mechanism-A reading for one trial against the control."""
    trial = per_window_slopes(Path(trial_root))
    control = per_window_slopes(Path(control_root))
    shared = sorted(set(trial) & set(control))
    if not shared:
        raise ValueError("the two passes share no window directories; the slopes cannot be paired")
    pairs = [
        {
            "episode_id": trial[name]["episode_id"],
            "trial_slope": trial[name]["slope"],
            "control_slope": control[name]["slope"],
        }
        for name in shared
    ]
    return {"n_windows": len(shared), "windows": shared, "bootstrap": paired_bootstrap_reduction(pairs)}


def main() -> None:  # pragma: no cover - the on-disk driver
    trial_root, control_root = Path(sys.argv[1]), Path(sys.argv[2])
    result = compare(trial_root, control_root)
    stats = result["bootstrap"]
    print(f"windows paired: {result['n_windows']}  episodes: {stats['n_episodes']}")
    print(f"mean slope  trial {stats['mean_slope_trial']:+.6f}   control {stats['mean_slope_control']:+.6f}")
    print(
        f"reduction {stats['reduction']:+.4f}  "
        f"{int(stats['ci'] * 100)}% CI [{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]  "
        f"(seed {stats['seed']}, {stats['resamples']} resamples)"
    )
    print(f"predeclared threshold >= {stats['threshold']:.2f}: {'MET' if stats['meets_threshold'] else 'not met'}")

    if len(sys.argv) > 4:
        for label, root, aggregation in (("trial", trial_root, sys.argv[3]), ("control", control_root, sys.argv[4])):
            per_window = {name: data["per_frame"] for name, data in per_window_slopes(Path(root)).items()}
            report = self_validation(per_window, recorded_ssim_from_aggregation(Path(aggregation)))
            print(
                f"self-validation ({label}): n={report['n']} mean|diff| {report['mean_abs_diff']:.4f} "
                f"max|diff| {report['max_abs_diff']:.4f} "
                f"(from-video {report['mean_from_video']:.4f} vs recorded {report['mean_recorded']:.4f})"
            )


if __name__ == "__main__":  # pragma: no cover
    main()
