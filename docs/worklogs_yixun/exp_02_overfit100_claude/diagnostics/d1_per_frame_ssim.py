"""D1 — per-frame SSIM from the rendered videos (H7: how much of the score is the free pinned frame 0?).

Replicates skimage.metrics.structural_similarity(channel_axis=-1, data_range=1.0, win_size=7) exactly:
uniform 7x7 filter, K1=0.01/K2=0.03, sample covariance (N/(N-1)), then crop pad=3 and mean. Because the
result is cropped by pad, the filter's boundary mode is irrelevant, so a 'valid' box filter is identical
to scipy's reflect-mode filter on the retained region.

Source frames come from the rendered mp4s (lossy), so absolute values carry codec noise. The script
therefore VALIDATES itself: mean-over-frames must track the SSIM recorded by the eval for the same window.
The frame-index trend, not the absolute level, is the finding.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

W, H = 320, 192


def decode(path: Path) -> np.ndarray:
    """mp4 -> float32 [T, H, W, 3] in [0,1]."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    n = len(out) // (W * H * 3)
    return np.frombuffer(out, np.uint8)[: n * W * H * 3].reshape(n, H, W, 3).astype(np.float32) / 255.0


def _box(a: np.ndarray, k: int = 7) -> np.ndarray:
    """Valid-mode uniform filter over the two spatial axes (a: [H,W])."""
    c = np.cumsum(np.cumsum(a, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def ssim_frame(x: np.ndarray, y: np.ndarray, k: int = 7, data_range: float = 1.0) -> float:
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


def main() -> None:
    root = Path(sys.argv[1])          # dir holding <window>/{ground_truth,sample}.mp4
    recorded = json.loads(Path(sys.argv[2]).read_text())  # aggregation.json for the same pass
    ref = {r["name"]: r["ssim"] for r in recorded["rows"] if r["seed"] == 0 and r["context_mode"] == "correct"}

    per_frame_all, checks = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        gt_p, pr_p = d / "ground_truth.mp4", d / "sample.mp4"
        if not (gt_p.exists() and pr_p.exists()):
            continue
        gt, pr = decode(gt_p), decode(pr_p)
        t = min(len(gt), len(pr))
        vals = [ssim_frame(pr[i], gt[i]) for i in range(t)]
        per_frame_all.append(vals)
        if d.name in ref:
            checks.append((d.name, float(np.mean(vals)), ref[d.name]))

    arr = np.array([v for v in per_frame_all if len(v) == len(per_frame_all[0])], dtype=np.float64)
    print(f"windows analysed: {len(arr)}  frames/window: {arr.shape[1]}")

    if checks:
        d = [abs(a - b) for _, a, b in checks]
        print("\n=== self-validation (mp4-decoded mean-over-frames vs eval-recorded SSIM) ===")
        print(f"  n={len(checks)}  mean |diff| {np.mean(d):.4f}  max |diff| {np.max(d):.4f}")
        print(f"  recorded mean {np.mean([b for _,_,b in checks]):.4f} | from-video mean {np.mean([a for _,a,b in checks]):.4f}")

    print("\n=== per-frame SSIM (mean across windows) ===")
    m = arr.mean(axis=0)
    for i, v in enumerate(m):
        bar = "#" * int(round((v - 0.5) / 0.5 * 40)) if v > 0.5 else ""
        print(f"  frame {i:2}: {v:.4f} {bar}")
    print(f"\n  frame 0 (pinned): {m[0]:.4f}")
    print(f"  frames 1+ (predicted): {m[1:].mean():.4f}")
    print(f"  all frames: {m.mean():.4f}")
    print(f"  last frame: {m[-1]:.4f}   |  drop frame1 -> last: {m[1]-m[-1]:+.4f}")
    lift = m.mean() - m[1:].mean()
    print(f"  pinned frame lifts the reported mean by only {lift:+.4f} (1 of {len(m)} frames)")

    # Does the decay depend on how good the window is? If the best windows are flat, the
    # ceiling is a per-window property; if they decay too, compounding is universal.
    names = [d.name for d in sorted(p for p in root.iterdir() if p.is_dir())
             if (d / "sample.mp4").exists() and (d / "ground_truth.mp4").exists()]
    scored = [(ref[n], i) for i, n in enumerate(names) if n in ref and i < len(arr)]
    scored.sort()
    if len(scored) >= 6:
        k = len(scored) // 3
        print("\n=== per-frame decay by window quality (does the top decay too?) ===")
        for lbl, chunk in (("bottom third", scored[:k]), ("middle third", scored[k:2 * k]), ("top third", scored[2 * k:])):
            sub = arr[[i for _, i in chunk]]
            sm = sub.mean(axis=0)
            print(f"  {lbl:13} n={len(chunk):2}  frame0 {sm[0]:.4f}  frame1 {sm[1]:.4f}  "
                  f"last {sm[-1]:.4f}  decay(1->last) {sm[1]-sm[-1]:+.4f}")


if __name__ == "__main__":
    main()
