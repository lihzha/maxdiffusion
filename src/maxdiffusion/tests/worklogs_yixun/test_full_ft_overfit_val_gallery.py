"""CPU-only tests for the full-FT val-set HTML gallery (Part II, cycle C "val-gallery").

The gallery (``src/maxdiffusion/make_wan_val_gallery.py``) turns a locally pulled
``step_XXXXXX`` validation directory -- the exact artifact layout that
``generate_wan_side_adapter.py`` writes (verified against that module, not a summary) --
into a single self-contained ``gallery.html``. The load-bearing contract (Codex F7):

  * The **dataset position** shown per sample is ``config.json["validation_ordinals"]``
    indexed by that sample's ``metrics.json["sample_index"]`` -- NOT the stored
    ``metrics.json["ordinal"]`` field. The stored ordinal is shown separately. The
    fixtures make the two DELIBERATELY different (positions 0/2927/5854/... vs stored
    9000+i) so a join that reads the wrong field is caught.
  * Exact artifact keys consumed: ``latent_mse``/``pixel_mse``/``ssim_avg`` per sample,
    ``num_samples`` from ``summary.json`` -- wrong key names raise an actionable error.
  * Every referenced video is a RELATIVE path (``sample_dir/file.mp4``), so the page
    opens straight from disk; no absolute / ``file://`` / scheme-qualified src.
  * A verbatim provenance sentence must appear (ground truth is the VAE decode of the
    cached ``z_video`` latents, not the original DROID RGB video).
  * Per-missing-file, count-mismatch, non-FULL_FT, missing-ordinal, and sample_index
    gap/dupe conditions each raise a distinct, actionable, path-naming error.

Pure ``build_gallery_html(step_meta, samples)`` is exercised without a filesystem; the
real ``write_gallery`` path is exercised end-to-end against tmp fixture dirs.

Stdlib only -- the module imports no jax/tf, so this file does too (the darwin grain
stub in ``conftest.py`` is harmless but unused here).
"""

from __future__ import annotations

import json
import os
import re

import pytest

import maxdiffusion.make_wan_val_gallery as gallery

# The six requested dataset positions (Part II D6: "0,2927,5854,8781,11708,14635").
POSITIONS = [0, 2927, 5854, 8781, 11708, 14635]
# Stored TFRecord ordinals -- DELIBERATELY unrelated to positions (join-correctness probe).
STORED = [9000 + i for i in range(6)]
NAMES = [f"droidclip{i}_episode" for i in range(6)]

VIDEO_FILENAMES = ("ground_truth.mp4", "sample.mp4", "comparison_gt_top_pred_bottom.mp4")


# ------------------------------------------------------------------------------------------
# Fixture builders: write a fake step_XXXXXX dir matching the generate-script layout.
# ------------------------------------------------------------------------------------------


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _metrics(idx, ordinal, name):
    # Metric values chosen to format cleanly at 4 decimals so tests can assert substrings.
    return {
        "checkpoint_step": 20000,
        "sample_index": idx,
        "name": name,
        "ordinal": ordinal,
        "latent_mse": round(0.2000 + idx * 0.01, 4),
        "pixel_mse": round(0.0100 + idx * 0.001, 4),
        "ssim_avg": round(0.7000 + idx * 0.01, 4),
        "latent_mae": 0.1,
        "pixel_mae": 0.01,
        "z_pred_std": 1.0,
        "z_target_std": 1.0,
        "z_init_anchor_mse": 0.0,
        "height": 192,
        "width": 320,
        "frames": 25,
    }


def _make_step_dir(
    tmp_path,
    *,
    config_ordinals=POSITIONS,
    sample_specs=None,
    num_samples=None,
    model_type="FULL_FT_TI2V",
    config_extra=None,
    dirname="step_020000",
):
    """Build a fake step dir. ``sample_specs`` is a list of dicts with keys idx/ordinal/name
    and optional flags: ``omit_metrics``, ``drop_keys`` (metric keys to remove),
    ``extra`` (metric key overrides), ``drop_videos`` (filenames to skip)."""
    if sample_specs is None:
        sample_specs = [{"idx": i, "ordinal": STORED[i], "name": NAMES[i]} for i in range(6)]

    step = tmp_path / dirname
    step.mkdir()

    config = {
        "run_name": "wan-full-ft-test",
        "checkpoint_dir": "gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft/run/checkpoints",
        "checkpoint_step": 20000,
        "eval_data_dir": "gs://v6_east1d/datasets/droid_wan_side_adapter/val",
        "num_eval_videos": len(sample_specs),
        "seed": 0,
        "side_adapter_sampling_steps": 25,
        "side_adapter_guide_scale": 1.0,
        "model_type": model_type,
    }
    if config_ordinals is not None:  # None => omit the key entirely (missing-key case)
        config["validation_ordinals"] = list(config_ordinals)
    if config_extra:
        config.update(config_extra)
    _write_json(step / "config.json", config)

    n_summary = num_samples if num_samples is not None else len(sample_specs)
    summary = {
        "checkpoint_step": 20000,
        "num_samples": n_summary,
        "mean_latent_mse": 0.2495,
        "mean_pixel_mse": 0.0190,
        "mean_ssim": 0.7875,
        "summary_csv": str(step / "summary.csv"),
    }
    _write_json(step / "summary.json", summary)

    for spec in sample_specs:
        d = step / f"sample_{spec['idx']:04d}_{spec['name']}"
        d.mkdir()
        if not spec.get("omit_metrics"):
            m = _metrics(spec["idx"], spec["ordinal"], spec["name"])
            for k in spec.get("drop_keys", ()):
                m.pop(k, None)
            m.update(spec.get("extra", {}))
            _write_json(d / "metrics.json", m)
        for fn in VIDEO_FILENAMES:
            if fn in spec.get("drop_videos", ()):
                continue
            (d / fn).write_bytes(b"\x00\x00")
    return step


def _default_specs():
    return [{"idx": i, "ordinal": STORED[i], "name": NAMES[i]} for i in range(6)]


# ------------------------------------------------------------------------------------------
# (1) Dataset positions joined via validation_ordinals[sample_index], in order + labeled;
#     stored ordinals shown separately and distinct.
# ------------------------------------------------------------------------------------------


def test_dataset_positions_joined_in_order_and_labeled(tmp_path):
    step = _make_step_dir(tmp_path)
    out = gallery.write_gallery(str(step))
    html = open(out).read()

    # The position values, in document order, are exactly validation_ordinals -- NOT the
    # stored ordinals (a join off the wrong field would yield [9000..9005] here).
    found_positions = re.findall(r'class="dpos-value">(\d+)<', html)
    assert found_positions == [str(p) for p in POSITIONS]
    # Each card labels the value "dataset position".
    assert html.count("dataset position") == 6


def test_stored_ordinals_shown_separately_and_distinct(tmp_path):
    step = _make_step_dir(tmp_path)
    html = open(gallery.write_gallery(str(step))).read()

    found_stored = re.findall(r'class="ord-value">(\d+)<', html)
    assert found_stored == [str(o) for o in STORED]
    assert html.count("stored ordinal") == 6
    # Positions and stored ordinals are disjoint -> the labels can't be confusing the two.
    assert set(found_stored).isdisjoint({str(p) for p in POSITIONS})
    # The distinct per-sample names ride through too.
    for name in NAMES:
        assert name in html


# ------------------------------------------------------------------------------------------
# (2) Exact metric keys consumed; wrong key names -> actionable error.
# ------------------------------------------------------------------------------------------


def test_metric_values_rendered(tmp_path):
    step = _make_step_dir(tmp_path)
    html = open(gallery.write_gallery(str(step))).read()
    # sample index 2: latent 0.2200, pixel 0.0120, ssim 0.7200 (4-decimal contract).
    assert "0.2200" in html
    assert "0.0120" in html
    assert "0.7200" in html
    assert "latent MSE" in html and "pixel MSE" in html and "SSIM" in html


def test_wrong_metric_key_name_is_actionable(tmp_path):
    specs = _default_specs()
    specs[1]["drop_keys"] = ["latent_mse"]
    specs[1]["extra"] = {"lat_mse": 0.21}  # wrong name
    step = _make_step_dir(tmp_path, sample_specs=specs)
    with pytest.raises((KeyError, ValueError)) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "latent_mse" in msg  # names the exact missing key
    assert "metrics.json" in msg  # and the file it was missing from


# ------------------------------------------------------------------------------------------
# (3) Provenance sentence present verbatim.
# ------------------------------------------------------------------------------------------

PROVENANCE = (
    "Ground truth shown here is the VAE decode of the cached z_video latents, not the original DROID RGB video."
)


def test_provenance_sentence_verbatim(tmp_path):
    step = _make_step_dir(tmp_path)
    html = open(gallery.write_gallery(str(step))).read()
    assert PROVENANCE in html
    # And it is the exact module constant (single source of truth).
    assert gallery.PROVENANCE == PROVENANCE


# ------------------------------------------------------------------------------------------
# (4) All video refs relative, no leading '/' / file:// / absolute tmp path; files exist.
# ------------------------------------------------------------------------------------------


def test_video_refs_are_relative_and_resolve(tmp_path):
    step = _make_step_dir(tmp_path)
    out = gallery.write_gallery(str(step))
    html = open(out).read()
    out_dir = os.path.dirname(os.path.abspath(out))

    srcs = re.findall(r'src="([^"]+)"', html)
    assert len(srcs) == 6 * 3  # three videos per sample
    for src in srcs:
        assert not src.startswith("/"), f"absolute path leaked: {src}"
        assert "://" not in src, f"scheme-qualified (file://?) src leaked: {src}"
        assert str(tmp_path) not in src, f"absolute tmp path leaked: {src}"
        assert os.path.exists(os.path.join(out_dir, src)), f"broken video ref: {src}"
    # All three expected filenames appear as relative refs.
    for fn in VIDEO_FILENAMES:
        assert any(src.endswith(fn) for src in srcs)


# ------------------------------------------------------------------------------------------
# (5) Missing each artifact class -> error naming the exact path.
# ------------------------------------------------------------------------------------------


def test_missing_metrics_json_names_exact_path(tmp_path):
    specs = _default_specs()
    specs[2]["omit_metrics"] = True
    step = _make_step_dir(tmp_path, sample_specs=specs)
    with pytest.raises((FileNotFoundError, ValueError)) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "metrics.json" in msg
    assert "sample_0002" in msg  # the exact offending sample dir


def test_missing_mp4_names_exact_path(tmp_path):
    specs = _default_specs()
    specs[3]["drop_videos"] = ["ground_truth.mp4"]
    step = _make_step_dir(tmp_path, sample_specs=specs)
    with pytest.raises((FileNotFoundError, ValueError)) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "ground_truth.mp4" in msg
    assert "sample_0003" in msg


# ------------------------------------------------------------------------------------------
# (6) Count mismatches -> distinct actionable errors.
# ------------------------------------------------------------------------------------------


def test_count_mismatch_extra_sample_dir(tmp_path):
    # 7 sample dirs (idx 0..6) but only 6 ordinals + summary num_samples 6.
    specs = [{"idx": i, "ordinal": 9000 + i, "name": f"c{i}"} for i in range(7)]
    step = _make_step_dir(tmp_path, config_ordinals=POSITIONS, sample_specs=specs, num_samples=6)
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "validation_ordinals" in msg
    assert "7" in msg and "6" in msg


def test_count_mismatch_ordinals_list_shorter(tmp_path):
    # 6 sample dirs but validation_ordinals lists only 5.
    specs = _default_specs()
    step = _make_step_dir(tmp_path, config_ordinals=POSITIONS[:5], sample_specs=specs, num_samples=6)
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "validation_ordinals" in msg
    assert "5" in msg and "6" in msg


def test_count_mismatch_summary_num_samples(tmp_path):
    # 6 sample dirs, 6 ordinals, but summary.json says num_samples=5.
    specs = _default_specs()
    step = _make_step_dir(tmp_path, config_ordinals=POSITIONS, sample_specs=specs, num_samples=5)
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "summary.json" in msg and "num_samples" in msg
    assert "5" in msg


# ------------------------------------------------------------------------------------------
# (7) Non-FULL_FT config / missing ordinals -> actionable error.
# ------------------------------------------------------------------------------------------


def test_non_full_ft_model_type_is_actionable(tmp_path):
    step = _make_step_dir(tmp_path, model_type="SIDE_ADAPTER_TI2V")
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "FULL_FT_TI2V" in msg
    assert "SIDE_ADAPTER_TI2V" in msg


def test_empty_validation_ordinals_is_actionable(tmp_path):
    step = _make_step_dir(tmp_path, config_ordinals=[], sample_specs=_default_specs())
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    assert "validation_ordinals" in str(ei.value)


def test_missing_validation_ordinals_key_is_actionable(tmp_path):
    step = _make_step_dir(tmp_path, config_ordinals=None, sample_specs=_default_specs())
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    assert "validation_ordinals" in str(ei.value)


# ------------------------------------------------------------------------------------------
# (8) sample_index gap / dupe -> error.
# ------------------------------------------------------------------------------------------


def test_sample_index_gap_is_actionable(tmp_path):
    # Indices 0,1,2,3,4,6 -> position 5 missing (a gap). Counts otherwise consistent.
    idxs = [0, 1, 2, 3, 4, 6]
    specs = [{"idx": ix, "ordinal": 9000 + ix, "name": f"c{ix}"} for ix in idxs]
    step = _make_step_dir(tmp_path, config_ordinals=POSITIONS, sample_specs=specs, num_samples=6)
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    assert "sample_index" in str(ei.value)


def test_sample_index_dupe_is_actionable(tmp_path):
    # Two dirs both claim sample_index 4 -> position 5 never covered (a dupe).
    specs = [{"idx": i, "ordinal": 9000 + i, "name": f"c{i}"} for i in range(5)]
    specs.append({"idx": 4, "ordinal": 9099, "name": "cdupe"})
    step = _make_step_dir(tmp_path, config_ordinals=POSITIONS, sample_specs=specs, num_samples=6)
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    assert "sample_index" in str(ei.value)


# ------------------------------------------------------------------------------------------
# End-to-end + header/summary + pure-function + CLI.
# ------------------------------------------------------------------------------------------


def test_write_gallery_writes_html_inside_step_dir(tmp_path):
    step = _make_step_dir(tmp_path)
    out = gallery.write_gallery(str(step))
    assert out == os.path.join(str(step), "gallery.html")
    assert os.path.exists(out)
    html = open(out).read()
    assert "<video" in html and "controls" in html and 'preload="metadata"' in html


def test_header_and_summary_strip_from_artifacts(tmp_path):
    step = _make_step_dir(tmp_path, config_extra={"generation_commit": "abc1234"})
    html = open(gallery.write_gallery(str(step))).read()
    # Header: run name, checkpoint step, seed, dataset path, generation commit (when present).
    assert 'data-k="run_name">wan-full-ft-test<' in html
    assert 'data-k="checkpoint_step">20000<' in html
    assert 'data-k="seed">0<' in html
    assert "gs://v6_east1d/datasets/droid_wan_side_adapter/val" in html
    assert "abc1234" in html
    # Summary strip: the summary.json means + num_samples.
    assert "0.2495" in html and "0.0190" in html and "0.7875" in html
    assert 'data-k="num_samples">6<' in html


def test_build_gallery_html_pure_no_filesystem():
    step_meta = {
        "title": "gallery",
        "run_name": "r",
        "checkpoint_step": 20000,
        "seed": 0,
        "dataset_path": "gs://x/val",
        "commits": {"commit": "deadbee"},
        "mean_latent_mse": 0.25,
        "mean_pixel_mse": 0.019,
        "mean_ssim": 0.78,
        "num_samples": 2,
    }
    samples = [
        {
            "sample_index": 0,
            "dataset_position": 0,
            "stored_ordinal": 9000,
            "name": "a",
            "latent_mse": 0.20,
            "pixel_mse": 0.01,
            "ssim_avg": 0.70,
            "videos": {
                "ground_truth": "sample_0000_a/ground_truth.mp4",
                "sample": "sample_0000_a/sample.mp4",
                "comparison": "sample_0000_a/comparison_gt_top_pred_bottom.mp4",
            },
        },
        {
            "sample_index": 1,
            "dataset_position": 2927,
            "stored_ordinal": 9001,
            "name": "b",
            "latent_mse": 0.21,
            "pixel_mse": 0.011,
            "ssim_avg": 0.71,
            "videos": {
                "ground_truth": "sample_0001_b/ground_truth.mp4",
                "sample": "sample_0001_b/sample.mp4",
                "comparison": "sample_0001_b/comparison_gt_top_pred_bottom.mp4",
            },
        },
    ]
    html = gallery.build_gallery_html(step_meta, samples)
    assert gallery.PROVENANCE in html
    assert re.findall(r'class="dpos-value">(\d+)<', html) == ["0", "2927"]
    assert re.findall(r'class="ord-value">(\d+)<', html) == ["9000", "9001"]
    assert "deadbee" in html  # commit rides through when present
    for src in re.findall(r'src="([^"]+)"', html):
        assert not src.startswith("/") and "://" not in src


def test_cli_main_writes_gallery(tmp_path):
    step = _make_step_dir(tmp_path)
    rc = gallery.main([str(step), "--title", "My Gallery"])
    assert rc in (0, None)
    out = os.path.join(str(step), "gallery.html")
    assert os.path.exists(out)
    assert "My Gallery" in open(out).read()


def test_cli_custom_out_keeps_refs_relative_to_out(tmp_path):
    # --out elsewhere: video refs must be relative to the OUT file's directory so the
    # page still opens from disk.
    step = _make_step_dir(tmp_path)
    out_path = tmp_path / "elsewhere" / "g.html"
    (tmp_path / "elsewhere").mkdir()
    gallery.write_gallery(str(step), out=str(out_path))
    html = open(out_path).read()
    out_dir = os.path.dirname(os.path.abspath(str(out_path)))
    for src in re.findall(r'src="([^"]+)"', html):
        assert "://" not in src
        assert os.path.exists(os.path.join(out_dir, src)), f"broken ref from custom out: {src}"
