"""Python port of the dataviz skill's validate_palette.js (node unavailable here).

Constants, matrices and formulas transcribed 1:1 from scripts/validate_palette.js:
BAND, CHROMA_FLOOR, CVD_TARGET/FLOOR, NORMAL_FLOOR, CONTRAST_MIN, Machado 2009
severity-1.0 matrices, sRGB->linear, OKLab, and Euclidean OKLab dE x100.
"""

import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]


def oklch(h):
    L, a, b = oklab_from_lin(lin(h))
    return L, math.hypot(a, b)


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b)) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True
    rows = []

    off = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not off
    rows.append(("Lightness band", not off, f"outside: {off}" if off else f"all {len(palette)} inside L {lo}-{hi}"))

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    rows.append(("Chroma floor", not lowc, f"below floor: {lowc}" if lowc else f"all {len(palette)} >= {CHROMA_FLOOR}"))

    n = len(palette)
    if pairs == "all":
        pairlist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairlist = [(i, i + 1) for i in range(n - 1)]
    label = "all-pairs" if pairs == "all" else "adjacent"

    worst_d, worst = 99.0, None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if d < worst_d:
                worst_d, worst = d, (kind, palette[i], palette[j])
    tri = min([delta_e(palette[i], palette[j], "tritan") for i, j in pairlist], default=99)
    cvd_ok = worst_d >= CVD_TARGET
    cvd_band = CVD_FLOOR <= worst_d < CVD_TARGET
    ok &= worst_d >= CVD_FLOOR
    rows.append(
        (
            f"CVD separation ({label})",
            cvd_ok,
            (f"worst {label} {worst[2]}<->{worst[1] if worst else ''} dE {worst_d:.1f} ({worst[0]}) - tritan {tri:.1f}"
             + ("  [6-8 BAND: secondary encoding required]" if cvd_band else "")) if worst else "n/a",
        )
    )

    nworst_d, nworst = 99.0, None
    for i, j in pairlist:
        d = delta_e(palette[i], palette[j])
        if d < nworst_d:
            nworst_d, nworst = d, (palette[i], palette[j])
    nok = nworst_d >= NORMAL_FLOOR
    ok &= nok
    rows.append((f"Normal-vision floor ({label})", nok,
                 f"worst {nworst[0]}<->{nworst[1]} dE {nworst_d:.1f} (floor {NORMAL_FLOOR})" if nworst else "n/a"))

    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    rows.append(("Contrast vs surface", not low,
                 f"WARN sub-3:1 (relief rule: visible labels / table view): {low}" if low
                 else f"all >= {CONTRAST_MIN}:1 vs {surface}"))

    return ok, rows


if __name__ == "__main__":
    pal = [c.strip() for c in sys.argv[1].split(",") if c.strip()]
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    pairs = sys.argv[3] if len(sys.argv) > 3 else "adjacent"
    ok, rows = validate(pal, mode=mode, pairs=pairs)
    print(f"=== {mode.upper()} / {pairs} / surface {DEFAULT_SURFACE[mode]} / {len(pal)} colors ===")
    for name, passed, detail in rows:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"  => {'OK' if ok else 'FAILURES PRESENT'}")
