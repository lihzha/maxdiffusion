#!/usr/bin/env python3
"""exp_02 overfit-100 — rollout SSIM vs training steps (+ deterministic one-step val loss).

Renders ``figures/exp02_ssim_vs_steps.{png,svg}``.

WHAT IS PLOTTED (the estimand)
------------------------------
Panel A: the **canonical-cohort, seed-0, correct-context mean rollout SSIM** — the
estimand used across the exp_02 campaign. For every committed aggregation artifact we
filter ``rows`` down to

    seed == 0  AND  context_mode == "correct"  AND  (episode_id, window_start) in canonical_cohort

and take the arithmetic mean of ``ssim``. The ``canonical_cohort`` list is read from the
artifact itself (100 windows), so the filter is derived from the data, never hardcoded.
Every artifact yields exactly n = 100 rows. All evals are 25-step rollouts at
guide_scale 1.0.

Panel B: the deterministic one-step validation loss from
``d2_val_loss_by_checkpoint.json`` (8 checkpoints, n = 1,629 windows each, fixed
validation_seed 0 so (t, eps) are deterministic). This instrument covers the 1e-5 run
only (250 -> 10,000); the escalated segment is deliberately left blank rather than
filled from a different instrument's readings.

DATA PROVENANCE — one row per plotted point
-------------------------------------------
Source directory: ``overfit100_s3_artifacts/``. Where two artifacts cover the same
checkpoint (a full-set pass and a segment-final pass), both restrict to the identical
canonical/seed-0/correct subset and agree exactly; the script asserts that agreement and
plots the value once.

| step   | LR   | run lineage                        | artifact file(s)                                  |
|--------|------|------------------------------------|---------------------------------------------------|
| 250    | 1e-5 | wan-overfit100-s3-20260730         | step_000250_s3_intermediate_aggregation.json      |
| 500    | 1e-5 | wan-overfit100-s3-20260730         | step_000500_s3_intermediate_aggregation.json      |
| 1000   | 1e-5 | wan-overfit100-s3-20260730         | step_001000_s3_intermediate_aggregation.json      |
| 1750   | 1e-5 | wan-overfit100-s3-20260730         | step_001750_s3_intermediate_aggregation.json      |
| 2500   | 1e-5 | wan-overfit100-s3-20260730         | step_002500_s3_full_set_aggregation.json          |
|        |      |                                    | + step_002500_s3_segment_final_aggregation.json   |
| 5000   | 1e-5 | wan-overfit100-s3-20260730         | step_005000_s3_intermediate_aggregation.json      |
| 7500   | 1e-5 | wan-overfit100-s3-20260730         | step_007500_s3_intermediate_aggregation.json      |
| 10000  | 1e-5 | wan-overfit100-s3-20260730         | step_010000_s3_full_set_aggregation.json          |
|        |      |                                    | + step_010000_s3_segment_final_aggregation.json   |
| 15000  | 5e-5 | wan-overfit100-s3ext-lr5e5-20260802| step_015000_lr5e5_s3_intermediate_aggregation.json|
| 17500  | 5e-5 | wan-overfit100-s3ext-lr5e5-20260802| step_017500_lr5e5_s3_intermediate_aggregation.json|
| 20000  | 1e-4 | wan-overfit100-s3ext-lr1e4-20260804| step_020000_lr1e4_s3_full_set_aggregation.json    |
|        |      |                                    | + step_020000_lr1e4_s3_segment_final_aggregation.json |

THE 12,500 GAP (plotted as a hollow marker, excluded from the series)
---------------------------------------------------------------------
``overfit100_results.md`` records a canonical seed-0 mean SSIM of **0.9159** at step
12,500 (the 5e-5 arm of the LR sweep, Jobs 38-39). **No aggregation artifact for step
12,500 is committed** under ``overfit100_s3_artifacts/`` — the only 12,500 file in the
tree is the LR-sweep *loss* figure. That reading therefore cannot be recomputed from
artifacts here. It is drawn as a hollow grey marker, kept OUT of the connected series
and out of the artifact-derived table, and labelled as prose-recorded. Nothing about it
is invented: the value is quoted verbatim from the results doc.

LR SEGMENTS (verified against overfit100_command.md, Jobs 40/44/47)
-------------------------------------------------------------------
1e-5 : 250 -> 10,000     (base run)
5e-5 : 10,000 -> 17,500  (escalated continuation; Job 40 10k->12.5k->15k, Job 44 ->17.5k)
1e-4 : 17,500 -> 20,000  (Job 47; seeded from a byte-identical copy of the 17,500 ckpt)

Marker colour encodes the LR that *trained* that checkpoint; line colour encodes the LR
of the interval. Checkpoint 10,000 is therefore blue (produced at 1e-5) with the orange
5e-5 segment departing from it, and 17,500 is orange with the aqua 1e-4 segment
departing from it.

FINAL-POINT VERDICT (from verdict_lr1e4_step20000_complete.json)
-----------------------------------------------------------------
headline.fraction = 0.69 (69/100 canonical windows >= 0.95 on the 3-seed median),
headline.established = False against threshold 0.95; partial claim established at the
0.90 bar with fraction 0.98; ``verdict = "partial"``.

CROSS-CHECK
-----------
Each computed mean is asserted against the value recorded in ``overfit100_results.md``
with a tolerance of 1e-3. The script raises rather than plotting on any mismatch.

PALETTE
-------
Categorical slots 1-3 of the validated default data-viz palette (blue / orange / aqua),
checked with the skill's ``validate_palette.py`` against a white publication surface
(``--pairs all``): all checks PASS; aqua carries a sub-3:1 contrast WARN whose required
relief (visible direct labels) is satisfied — every data point is directly labelled.

RUN
---
    <venv>/bin/python plot_ssim_vs_steps.py     # needs matplotlib; writes into figures/
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent  # .../exp_02_overfit100_claude/diagnostics
EXP = HERE.parent
ART = EXP / "overfit100_s3_artifacts"
FIGDIR = HERE / "figures"
D2 = HERE / "d2_val_loss_by_checkpoint.json"

# ── palette (validated default data-viz palette, categorical slots 1-3) ───────
C_1E5 = "#2a78d6"  # slot 1 blue
C_5E5 = "#eb6834"  # slot 2 orange
C_1E4 = "#1baf7a"  # slot 3 aqua
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# ── the artifacts backing each plotted checkpoint ─────────────────────────────
# step -> (learning rate, [artifact filenames that must agree])
SOURCES: dict[int, tuple[str, list[str]]] = {
    250: ("1e-5", ["step_000250_s3_intermediate_aggregation.json"]),
    500: ("1e-5", ["step_000500_s3_intermediate_aggregation.json"]),
    1000: ("1e-5", ["step_001000_s3_intermediate_aggregation.json"]),
    1750: ("1e-5", ["step_001750_s3_intermediate_aggregation.json"]),
    2500: ("1e-5", ["step_002500_s3_full_set_aggregation.json",
                    "step_002500_s3_segment_final_aggregation.json"]),
    5000: ("1e-5", ["step_005000_s3_intermediate_aggregation.json"]),
    7500: ("1e-5", ["step_007500_s3_intermediate_aggregation.json"]),
    10000: ("1e-5", ["step_010000_s3_full_set_aggregation.json",
                     "step_010000_s3_segment_final_aggregation.json"]),
    15000: ("5e-5", ["step_015000_lr5e5_s3_intermediate_aggregation.json"]),
    17500: ("5e-5", ["step_017500_lr5e5_s3_intermediate_aggregation.json"]),
    20000: ("1e-4", ["step_020000_lr1e4_s3_full_set_aggregation.json",
                     "step_020000_lr1e4_s3_segment_final_aggregation.json"]),
}

# values recorded in overfit100_results.md — the cross-check, not the data source
RECORDED = {
    250: 0.7580, 500: 0.7707, 1000: 0.7892, 1750: 0.8020, 2500: 0.8139,
    5000: 0.8320, 7500: 0.8377, 10000: 0.8416,
    15000: 0.9451, 17500: 0.9508, 20000: 0.9536,
}
TOL = 1e-3

# step 12,500: recorded in overfit100_results.md, NO committed aggregation artifact
GAP_STEP, GAP_SSIM = 12500, 0.9159

MEMO_BAR = 0.95


def canonical_seed0_correct_mean(path: Path) -> tuple[float, int, str, int]:
    """Mean SSIM over the canonical cohort, seed 0, correct context.

    Returns (mean, n, run_name, checkpoint_step). The cohort is read from the
    artifact's own ``canonical_cohort`` list, so the filter is data-derived.
    """
    d = json.loads(path.read_text())
    cohort = {tuple(w) for w in d["canonical_cohort"]}
    sel = [
        r for r in d["rows"]
        if r["seed"] == 0
        and r["context_mode"] == "correct"
        and (r["episode_id"], r["window_start"]) in cohort
        and r["ssim"] is not None
    ]
    if not sel:
        raise ValueError(f"{path.name}: no rows survived the canonical/seed-0/correct filter")
    return statistics.fmean(r["ssim"] for r in sel), len(sel), d["run_name"], d["checkpoint_step"]


def load_series() -> list[dict]:
    """Compute every plotted point from artifacts, asserting internal + recorded agreement."""
    out = []
    for step, (lr, files) in SOURCES.items():
        means, ns, runs = [], [], []
        for fn in files:
            m, n, run, ck = canonical_seed0_correct_mean(ART / fn)
            if ck != step:
                raise ValueError(f"{fn}: checkpoint_step {ck} != expected {step}")
            means.append(m)
            ns.append(n)
            runs.append(run)
        # artifacts covering the same checkpoint must agree exactly on this estimand
        if max(means) - min(means) > 1e-12:
            raise ValueError(f"step {step}: artifacts disagree on the estimand: "
                             + ", ".join(f"{f}={m!r}" for f, m in zip(files, means)))
        if len(set(ns)) != 1 or len(set(runs)) != 1:
            raise ValueError(f"step {step}: inconsistent n or run_name across {files}")
        mean = means[0]
        rec = RECORDED[step]
        if abs(mean - rec) > TOL:
            raise ValueError(
                f"CROSS-CHECK FAILED at step {step}: computed {mean:.6f} vs "
                f"recorded {rec:.4f} (|delta| {abs(mean - rec):.6f} > {TOL}). Refusing to plot."
            )
        out.append({"step": step, "lr": lr, "mean": mean, "n": ns[0],
                    "run": runs[0], "files": files, "recorded": rec})
    return sorted(out, key=lambda r: r["step"])


def main() -> None:
    pts = load_series()
    d2 = json.loads(D2.read_text())
    d2 = sorted(d2, key=lambda r: r["checkpoint_step"])

    by_step = {p["step"]: p for p in pts}

    # segments: line colour = LR of the interval
    seg_1e5 = [s for s in (250, 500, 1000, 1750, 2500, 5000, 7500, 10000)]
    seg_5e5 = [10000, 15000, 17500]
    seg_1e4 = [17500, 20000]

    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(9.6, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.45, 1.0], "hspace": 0.10},
    )

    # ── panel A: rollout SSIM ────────────────────────────────────────────────
    ax.set_ylim(0.7395, 0.980)
    ax.set_xlim(-400, 20900)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # LR-change event rules (span both panels visually via one rule per axes)
    for xb, lbl in ((10000, "LR 1e-5 → 5e-5"), (17500, "LR 5e-5 → 1e-4")):
        for a in (ax, bx):
            a.axvline(xb, color=AXIS, lw=0.9, zorder=1)
        ax.text(xb - 230, 0.9225, lbl, ha="right", va="center",
                fontsize=7.4, color=MUTED, zorder=5)

    # memorization bar
    ax.axhline(MEMO_BAR, color=INK_2, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax.text(10450, MEMO_BAR + 0.0030, "memorization bar = 0.95",
            fontsize=7.6, color=INK_2, ha="left", va="bottom", zorder=6)

    def draw(steps, color, z):
        xs = [s for s in steps]
        ys = [by_step[s]["mean"] for s in steps]
        ax.plot(xs, ys, color=color, lw=2.0, solid_capstyle="round", zorder=z)

    draw(seg_1e5, C_1E5, 3)
    draw(seg_5e5, C_5E5, 3)
    draw(seg_1e4, C_1E4, 3)

    # markers: colour = LR that trained that checkpoint
    mcol = {s: (C_1E5 if s <= 10000 else C_5E5 if s <= 17500 else C_1E4) for s in by_step}
    for s, p in by_step.items():
        ax.plot(s, p["mean"], "o", ms=6.4, color=mcol[s], mec=SURFACE, mew=1.4, zorder=6)

    # the 12,500 gap — hollow, NOT connected into the series
    ax.plot(GAP_STEP, GAP_SSIM, "o", ms=7.0, mfc="none", mec=MUTED, mew=1.5,
            ls="none", zorder=5)

    # direct value labels on every point (also the contrast relief for aqua)
    lab = {
        250: (7, -9, "left", "top"), 500: (7, -9, "left", "top"),
        1000: (7, -9, "left", "top"), 1750: (7, -9, "left", "top"),
        2500: (7, -9, "left", "top"), 5000: (0, -11, "center", "top"),
        7500: (0, -11, "center", "top"), 10000: (4, -11, "left", "top"),
        15000: (0, 10, "center", "bottom"), 17500: (-4, 10, "right", "bottom"),
        20000: (2, 10, "center", "bottom"),
    }
    for s, p in by_step.items():
        dx, dy, ha, va = lab[s]
        ax.annotate(f"{p['mean']:.4f}", (s, p["mean"]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=7.6, color=INK_2, zorder=7)

    # the honest gap annotation
    ax.annotate(
        "step 12,500 = 0.9159 (5e-5 arm) is recorded in overfit100_results.md\n"
        "but has NO committed aggregation artifact — shown hollow, excluded\n"
        "from the artifact-derived series and from the connecting line.",
        xy=(GAP_STEP, GAP_SSIM), xytext=(13150, 0.8985),
        ha="left", va="top", fontsize=7.4, color=INK_2, zorder=8,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                        shrinkA=2, shrinkB=5, connectionstyle="arc3,rad=-0.18"),
    )

    # final-point annotation
    ax.annotate(
        "final checkpoint — step 20,000, LR 1e-4\n"
        "mean SSIM 0.9536 · 69/100 canonical windows ≥ 0.95\n"
        "on the 3-seed median → formal verdict: partial",
        xy=(20000, 0.9536), xytext=(13350, 0.7855),
        ha="left", va="center", fontsize=7.9, color=INK,
        bbox=dict(boxstyle="round,pad=0.45", fc="#fbfbf9", ec=AXIS, lw=0.8),
        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9,
                        shrinkA=6, shrinkB=6, connectionstyle="arc3,rad=0.22"),
        zorder=8,
    )

    ax.set_ylabel("rollout SSIM (canonical-100, seed 0, correct context)", fontsize=9.2)
    ax.set_title("Rollout SSIM vs training steps", fontsize=14.5, color=INK,
                 loc="left", pad=26, fontweight="bold")
    ax.text(0, 1.045,
            "exp_02 overfit-100 · 100 canonical windows · seed 0 · correct text · 25-step rollout, guide scale 1.0",
            transform=ax.transAxes, fontsize=8.8, color=INK_2, va="bottom", ha="left")

    handles = [
        Line2D([], [], color=C_1E5, lw=2.0, marker="o", ms=6.0, mec=SURFACE, mew=1.2,
               label="LR 1e-5 — base run (250 → 10,000)"),
        Line2D([], [], color=C_5E5, lw=2.0, marker="o", ms=6.0, mec=SURFACE, mew=1.2,
               label="LR 5e-5 — escalated continuation (10,000 → 17,500)"),
        Line2D([], [], color=C_1E4, lw=2.0, marker="o", ms=6.0, mec=SURFACE, mew=1.2,
               label="LR 1e-4 — final segment (17,500 → 20,000)"),
        Line2D([], [], color=INK_2, lw=1.1, ls=(0, (5, 3)), label="memorization bar (0.95)"),
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=6.6, mfc="none", mew=1.5,
               label="recorded in prose, no committed artifact"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.995),
                    frameon=True, fontsize=8.0, handlelength=2.4, borderpad=0.7,
                    labelspacing=0.62, framealpha=1.0)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor("#fdfdfc")
    leg.get_frame().set_linewidth(0.8)
    leg.set_zorder(9)

    # ── panel B: deterministic one-step val loss ─────────────────────────────
    xs = [r["checkpoint_step"] for r in d2]
    ys = [r["mean_loss"] for r in d2]
    es = [r["stderr"] for r in d2]
    bx.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    bx.set_axisbelow(True)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)

    bx.errorbar(xs, ys, yerr=es, color=C_1E5, lw=1.9, marker="o", ms=6.0,
                mec=SURFACE, mew=1.4, ecolor=C_1E5, elinewidth=1.1, capsize=3,
                capthick=1.1, zorder=4)

    bx.set_ylim(0.104, 0.208)
    bx.axvspan(10000, 20900, color="#f4f3ef", zorder=0)
    bx.text(15450, 0.1735, "no d2 coverage beyond 10,000\n(this instrument runs on the 1e-5 run only)",
            ha="center", va="center", fontsize=7.8, color=MUTED, zorder=3)

    for s, (x, y) in enumerate(zip(xs, ys)):
        if x in (250, 10000):
            bx.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(8, 7) if x == 250 else (0, 12),
                        ha="left" if x == 250 else "center", va="bottom",
                        fontsize=7.6, color=INK_2, zorder=6)

    bx.set_ylabel("one-step val loss\n(deterministic t, ε)", fontsize=9.2)
    bx.set_xlabel("training step", fontsize=9.6, labelpad=7)
    bx.text(0.012, 0.055,
            "deterministic one-step validation loss — n = 1,629 windows, validation_seed 0, ±1 s.e.",
            transform=bx.transAxes, fontsize=8.2, color=INK_2, va="bottom", ha="left")

    bx.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
    bx.set_xticklabels([f"{t:,}" for t in [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000]])

    # ── caption ──────────────────────────────────────────────────────────────
    fig.text(
        0.048, 0.022,
        "The 250–10,000 series is the base 1e-5 run (wan-overfit100-s3-20260730). Later points are the LR-escalated continuation from the 10,000 checkpoint:\n"
        "5e-5 to 17,500 (wan-overfit100-s3ext-lr5e5-20260802), then 1e-4 to 20,000 (wan-overfit100-s3ext-lr1e4-20260804, seeded from a byte-identical\n"
        "copy of the 17,500 checkpoint). Every SSIM point is recomputed from the committed aggregation artifacts, n = 100 canonical windows each.\n"
        "Marker colour = the LR that trained that checkpoint; line colour = the LR of the interval.",
        fontsize=7.4, color=MUTED, va="bottom", ha="left", linespacing=1.5,
    )

    fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.165)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / "exp02_ssim_vs_steps.png"
    svg = FIGDIR / "exp02_ssim_vs_steps.svg"
    fig.savefig(png, dpi=200)
    fig.savefig(svg)
    plt.close(fig)

    # ── report ───────────────────────────────────────────────────────────────
    print(f"{'step':>6}  {'LR':>5}  {'mean SSIM':>9}  {'recorded':>8}  {'n':>4}  source artifact(s)")
    for p in pts:
        print(f"{p['step']:>6}  {p['lr']:>5}  {p['mean']:>9.6f}  {p['recorded']:>8.4f}  "
              f"{p['n']:>4}  {'; '.join(p['files'])}")
    print(f"{GAP_STEP:>6}  {'5e-5':>5}  {GAP_SSIM:>9.4f}  {'(prose)':>8}  {'--':>4}  "
          "NO COMMITTED ARTIFACT — value quoted from overfit100_results.md, not plotted in series")
    print(f"\ncross-check: all {len(pts)} artifact-derived means within {TOL} of overfit100_results.md")
    print(f"wrote {png}\nwrote {svg}")


if __name__ == "__main__":
    main()
