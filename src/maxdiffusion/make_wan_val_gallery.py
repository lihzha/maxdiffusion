"""Build a self-contained HTML gallery from a pulled full-FT val-set step directory.

Part II (Query 8) T2: after ``generate_wan_side_adapter.py`` writes a
``validation_valset/step_020000`` directory to GCS and it is pulled locally, this turns
it into a single ``gallery.html`` that opens straight from disk -- one card per sample
with the three rollout videos and the per-sample metrics.

The load-bearing join contract (Codex F7): the **dataset position** shown for a sample
is ``config.json["validation_ordinals"]`` indexed by that sample's
``metrics.json["sample_index"]`` -- NOT the stored ``metrics.json["ordinal"]`` field
(that stored ordinal is shown separately). Exact artifact keys consumed: ``latent_mse``,
``pixel_mse``, ``ssim_avg`` per sample and ``num_samples`` from ``summary.json`` -- all
verified against ``generate_wan_side_adapter.py``, not a summary.

CPU-only, stdlib-only (no jax/tf/numpy import) so it runs on a laptop. Usage::

    python -m maxdiffusion.make_wan_val_gallery <step_dir> [--out gallery.html] [--title ...]
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import math
import os
import sys
from typing import Sequence

# Verbatim provenance statement required in every gallery (single source of truth).
PROVENANCE = (
    "Ground truth shown here is the VAE decode of the cached z_video latents, " "not the original DROID RGB video."
)

# logical name -> the on-disk filename generate_wan_side_adapter.py writes per sample.
VIDEO_FILES = {
    "ground_truth": "ground_truth.mp4",
    "sample": "sample.mp4",
    "comparison": "comparison_gt_top_pred_bottom.mp4",
}
# figure captions, in render order.
VIDEO_CAPTIONS = (
    ("ground_truth", "ground truth"),
    ("sample", "prediction"),
    ("comparison", "comparison (GT top / prediction bottom)"),
)

REQUIRED_MODEL_TYPE = "FULL_FT_TI2V"

_esc = html.escape


# --------------------------------------------------------------------------------------
# I/O + validation.
# --------------------------------------------------------------------------------------


def _read_json(path: str, what: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {what}: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: could not parse {what} as JSON: {exc}") from exc


def _require_metric(metrics: dict, key: str, path: str):
    if key not in metrics:
        raise ValueError(
            f"{path}: metrics.json missing required key {key!r} (keys present: {sorted(metrics)}). "
            f"The gallery consumes sample_index, ordinal, name, latent_mse, pixel_mse, ssim_avg."
        )
    return metrics[key]


def load_step_meta(step_dir: str) -> tuple[dict, dict]:
    """Read + validate ``config.json`` and ``summary.json``. Returns ``(config, summary)``."""
    config = _read_json(os.path.join(step_dir, "config.json"), "config.json")
    model_type = config.get("model_type")
    if model_type != REQUIRED_MODEL_TYPE:
        raise ValueError(
            f"{step_dir}/config.json: gallery requires model_type == {REQUIRED_MODEL_TYPE!r} "
            f"(a full-FT val run); got {model_type!r}. This is not a full-FT validation step dir."
        )
    ordinals = config.get("validation_ordinals")
    if not isinstance(ordinals, list) or len(ordinals) == 0:
        raise ValueError(
            f"{step_dir}/config.json: 'validation_ordinals' must be a non-empty list of dataset "
            f"positions (the ordered cohort the gallery joins by sample_index); got {ordinals!r}."
        )
    summary = _read_json(os.path.join(step_dir, "summary.json"), "summary.json")
    if "num_samples" not in summary:
        raise ValueError(f"{step_dir}/summary.json: missing required key 'num_samples'.")
    return config, summary


def discover_samples(step_dir: str, config: dict, summary: dict) -> list[dict]:
    """Discover ``sample_*`` dirs, validate, order by ``sample_index``, and join positions.

    Raises actionable errors for: missing per-sample metrics.json or any required video
    (naming the exact path); sample_index gaps/dupes; and count disagreement between the
    sample dirs, ``validation_ordinals``, and ``summary.json`` num_samples.
    """
    ordinals = config["validation_ordinals"]
    num_samples = summary["num_samples"]

    sample_dirs = sorted(p for p in glob.glob(os.path.join(step_dir, "sample_*")) if os.path.isdir(p))

    records: list[dict] = []
    for d in sample_dirs:
        metrics_path = os.path.join(d, "metrics.json")
        metrics = _read_json(metrics_path, "metrics.json")
        records.append(
            {
                "dir": d,
                "dir_name": os.path.basename(d),
                "sample_index": _require_metric(metrics, "sample_index", metrics_path),
                "stored_ordinal": _require_metric(metrics, "ordinal", metrics_path),
                "name": _require_metric(metrics, "name", metrics_path),
                "latent_mse": _require_metric(metrics, "latent_mse", metrics_path),
                "pixel_mse": _require_metric(metrics, "pixel_mse", metrics_path),
                "ssim_avg": _require_metric(metrics, "ssim_avg", metrics_path),
            }
        )

    records.sort(key=lambda r: r["sample_index"])
    indices = [r["sample_index"] for r in records]
    if indices != list(range(len(records))):
        raise ValueError(
            f"{step_dir}: sample_index values must form a contiguous 0..N-1 range with no gaps or "
            f"duplicates; got {indices} across {len(records)} sample_* dirs."
        )
    if len(records) != len(ordinals):
        raise ValueError(
            f"{step_dir}: found {len(records)} sample_* directories but config.json "
            f"validation_ordinals lists {len(ordinals)} dataset positions -- these must match."
        )
    if len(records) != num_samples:
        raise ValueError(
            f"{step_dir}: found {len(records)} sample_* directories but summary.json reports "
            f"num_samples={num_samples} -- these must match."
        )

    for r in records:
        r["dataset_position"] = ordinals[r["sample_index"]]
        r["video_paths"] = {}
        for logical, filename in VIDEO_FILES.items():
            path = os.path.join(r["dir"], filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"missing required video {filename}: {path} (each sample needs "
                    f"{', '.join(VIDEO_FILES.values())})."
                )
            r["video_paths"][logical] = path
    return records


def build_step_meta(config: dict, summary: dict, title: str | None) -> dict:
    """Assemble the header/summary payload the pure renderer consumes."""
    commits = {k: v for k, v in config.items() if "commit" in k.lower()}
    default_title = f"full-FT val gallery — {config.get('run_name', '')} @ step {config.get('checkpoint_step', '?')}"
    return {
        "title": title or default_title,
        "run_name": config.get("run_name", ""),
        "checkpoint_step": config.get("checkpoint_step", ""),
        "seed": config.get("seed", ""),
        "dataset_path": config.get("eval_data_dir"),
        "commits": commits,
        "mean_latent_mse": summary.get("mean_latent_mse"),
        "mean_pixel_mse": summary.get("mean_pixel_mse"),
        "mean_ssim": summary.get("mean_ssim"),
        "num_samples": summary.get("num_samples", ""),
    }


# --------------------------------------------------------------------------------------
# Pure HTML rendering.
# --------------------------------------------------------------------------------------


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return _esc(str(v))
    if math.isnan(fv):
        return "n/a"
    return f"{fv:.{nd}f}"


def _meta_span(label: str, data_k: str, value) -> str:
    return (
        f'<span class="m"><span class="k">{_esc(label)}</span> '
        f'<span class="v" data-k="{data_k}">{_esc(str(value))}</span></span>'
    )


def _stat(label: str, value, data_k: str | None = None) -> str:
    dk = f' data-k="{data_k}"' if data_k else ""
    return f'<div class="stat"><span class="stat-k">{_esc(label)}</span><span class="stat-v"{dk}>{value}</span></div>'


def _card(s: dict) -> str:
    videos = s["videos"]
    figs = "".join(
        f'<figure class="vid"><figcaption>{cap}</figcaption>'
        f'<video controls preload="metadata" src="{_esc(videos[key])}"></video></figure>'
        for key, cap in VIDEO_CAPTIONS
    )
    return (
        '<section class="card">'
        '<div class="ids">'
        f'<div class="dpos"><span class="dpos-label">dataset position</span> '
        f'<span class="dpos-value">{s["dataset_position"]}</span></div>'
        f'<div class="sub"><span class="ord-label">stored ordinal</span> '
        f'<span class="ord-value">{s["stored_ordinal"]}</span> · '
        f'<span class="sname">{_esc(str(s["name"]))}</span> · '
        f'<span class="sidx">sample_index {s["sample_index"]}</span></div>'
        "</div>"
        f'<div class="videos">{figs}</div>'
        '<div class="metrics">'
        f'<span class="metric">latent MSE <b>{_fmt(s["latent_mse"])}</b></span>'
        f'<span class="metric">pixel MSE <b>{_fmt(s["pixel_mse"])}</b></span>'
        f'<span class="metric">SSIM <b>{_fmt(s["ssim_avg"])}</b></span>'
        "</div>"
        "</section>"
    )


_CSS = """
:root{--bg:#fcfcfb;--ink:#1c1c1b;--ink2:#5f5f5c;--muted:#8a8a86;--line:#e4e4e0;--card:#ffffff;--accent:#2a78d6}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#ececea;--ink2:#b0b0ac;--muted:#7c7c78;--line:#33332f;--card:#222220;--accent:#5e9fe8}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:28px 18px 64px}
main{max-width:1040px;margin:0 auto}
h1{font-size:21px;margin:0 0 10px}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);margin-bottom:14px}
.meta .k{color:var(--muted)}
.meta .v{color:var(--ink);font-variant-numeric:tabular-nums;word-break:break-all}
.provenance{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:8px;padding:10px 14px;font-size:13px;color:var(--ink2);margin:0 0 18px}
.summary{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:26px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:130px}
.stat-k{display:block;font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stat-v{font-size:18px;font-variant-numeric:tabular-nums}
.cards{display:flex;flex-direction:column;gap:22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.ids{margin-bottom:12px}
.dpos-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.dpos-value{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums;color:var(--accent)}
.sub{font-size:12.5px;color:var(--ink2);margin-top:2px}
.ord-value{font-variant-numeric:tabular-nums;color:var(--ink)}
.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media (max-width:720px){.videos{grid-template-columns:1fr}}
.vid{margin:0}
.vid figcaption{font-size:12px;color:var(--ink2);margin-bottom:4px}
.vid video{width:100%;max-width:100%;border:1px solid var(--line);border-radius:6px;background:#000}
.metrics{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:12px;font-size:13px;color:var(--ink2);font-variant-numeric:tabular-nums}
.metrics b{color:var(--ink)}
"""


def build_gallery_html(step_meta: dict, samples: list[dict]) -> str:
    """Render the full standalone HTML document (pure: no filesystem access)."""
    title = _esc(str(step_meta.get("title") or "full-FT val gallery"))

    meta_bits = [
        _meta_span("run", "run_name", step_meta.get("run_name", "")),
        _meta_span("checkpoint step", "checkpoint_step", step_meta.get("checkpoint_step", "")),
        _meta_span("validation seed", "seed", step_meta.get("seed", "")),
    ]
    if step_meta.get("dataset_path"):
        meta_bits.append(_meta_span("dataset", "dataset_path", step_meta["dataset_path"]))
    for k, v in (step_meta.get("commits") or {}).items():
        meta_bits.append(_meta_span(k.replace("_", " "), k, v))

    summary_bits = [
        _stat("mean latent MSE", _fmt(step_meta.get("mean_latent_mse"))),
        _stat("mean pixel MSE", _fmt(step_meta.get("mean_pixel_mse"))),
        _stat("mean SSIM", _fmt(step_meta.get("mean_ssim"))),
        _stat("samples", step_meta.get("num_samples", ""), "num_samples"),
    ]

    cards = "\n".join(_card(s) for s in samples)

    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n<main>\n"
        f"<h1>{title}</h1>\n"
        f'<div class="meta">{"".join(meta_bits)}</div>\n'
        f'<p class="provenance">{PROVENANCE}</p>\n'
        f'<div class="summary">{"".join(summary_bits)}</div>\n'
        f'<div class="cards">\n{cards}\n</div>\n'
        "</main>\n</body>\n</html>\n"
    )


# --------------------------------------------------------------------------------------
# Orchestration + CLI.
# --------------------------------------------------------------------------------------


def write_gallery(step_dir: str, out: str | None = None, title: str | None = None) -> str:
    """Load + validate ``step_dir``, render, and write ``gallery.html``. Returns its path.

    Video refs are made relative to the OUT file's directory (default: inside ``step_dir``),
    so the page opens straight from disk with no absolute or scheme-qualified paths.
    """
    config, summary = load_step_meta(step_dir)
    records = discover_samples(step_dir, config, summary)

    out = out or os.path.join(step_dir, "gallery.html")
    out_dir = os.path.dirname(os.path.abspath(out))

    samples = []
    for r in records:
        videos = {logical: os.path.relpath(os.path.abspath(p), out_dir) for logical, p in r["video_paths"].items()}
        samples.append(
            {
                "sample_index": r["sample_index"],
                "dataset_position": r["dataset_position"],
                "stored_ordinal": r["stored_ordinal"],
                "name": r["name"],
                "latent_mse": r["latent_mse"],
                "pixel_mse": r["pixel_mse"],
                "ssim_avg": r["ssim_avg"],
                "videos": videos,
            }
        )

    step_meta = build_step_meta(config, summary, title)
    doc = build_gallery_html(step_meta, samples)
    with open(out, "w") as f:
        f.write(doc)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maxdiffusion.make_wan_val_gallery",
        description="Build a self-contained HTML gallery from a pulled full-FT val step dir.",
    )
    parser.add_argument("step_dir", help="local path to a pulled step_XXXXXX validation directory")
    parser.add_argument("--out", default=None, help="output HTML path (default: <step_dir>/gallery.html)")
    parser.add_argument("--title", default=None, help="page title (default: derived from config.json)")
    args = parser.parse_args(argv)
    out = write_gallery(args.step_dir, out=args.out, title=args.title)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
