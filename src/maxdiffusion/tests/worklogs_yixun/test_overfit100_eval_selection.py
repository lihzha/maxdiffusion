"""CPU-only tests for the exp_02 OVERFIT100 eval-window selection + reader (plan D11/§4).

``generate_wan_side_adapter.py``'s OVERFIT100_TI2V branch chooses WHICH windows to roll out
and reads them out of the schema-v2 TFRecords. What is pinned here:

  (A) CANONICAL-WINDOW MATH -- ``start = 4 * floor((n_w - 1) / 2)`` per episode, with
      ``n_w`` taken from the COMMITTED MANIFEST (the authenticated source; the dataset's
      ``episodes.json`` only says WHICH episodes the set contains). Edges: a 1-window
      episode -> 0, a 99-window episode -> 196, and the even/odd rounding is the plan's
      ``floor``, not a round-half-up.
  (B) SELECTION SPEC -- the ``--eval-windows`` spec (config key ``eval_windows``):
      ``canonical`` = one canonical window per episode IN THE SET, ordered by
      ``episode_index``; otherwise an explicit comma-separated list of window NAMES, kept
      in the listed order. Unknown episodes, off-grid / out-of-range starts, duplicates
      and an empty spec all raise actionable errors.
  (C) SPARSE SETS -- cycle C's ratified semantics: an eval set may be a SPARSE per-episode
      subset of the manifest, so selection is over the set's own episode indices and never
      assumes contiguity; an index the manifest does not define is refused.
  (D) READER -- the schema-v2 feature description has NO ``actions`` and DOES carry
      ``name``/``episode_id``/``window_start`` (cycle-C review judgment 7), records are
      matched to the requested windows BY NAME and returned in the REQUESTED order, a
      missing name is an actionable error naming it, and a duplicated name in the dataset
      is refused.
  (E) AGGREGATION ARTIFACT (G4) -- the machine-written JSON: schema tag, required
      metadata, per-row keys, and a json round-trip.
  (F) The validate/loss-eval bash arms: ``bash -n``, manifest-driven pinned prefetch,
      ``COMMIT`` export, and the env->override wiring.

Synthetic manifests in ``tmp_path`` plus the committed manifest for the real-cohort checks;
no weights, no GCS, no mesh.
"""

from __future__ import annotations

import inspect
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import maxdiffusion.generate_wan_side_adapter as gen

_REPO = Path(gen.__file__).parents[2]
_MANIFEST = _REPO / "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json"
_VALIDATE = _REPO / "bash_scripts/validate_wan_overfit100.sh"
_LOSS_EVAL = _REPO / "bash_scripts/eval_wan_overfit100_val_loss.sh"
_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_overfit100.yml"

C, F, H, W = 2, 3, 4, 4


# --------------------------------------------------------------------------------------
# Fixtures.
# --------------------------------------------------------------------------------------


def _manifest(tmp_path, episodes, *, name="manifest.json"):
    """A minimal manifest with the fields selection reads: index/id/used_text/n_windows."""
    payload = {
        "selection_seed": 0,
        "vae_fingerprint": {"hf_repo": "r", "revision": "a" * 40},
        "episodes": [
            {
                "episode_index": int(spec["episode_index"]),
                "episode_id": int(spec["episode_id"]),
                "used_text": spec.get("used_text", f"text {spec['episode_index']}"),
                "n_windows": int(spec["n_windows"]),
            }
            for spec in episodes
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path)


def _episodes_sidecar(directory: Path, indices, *, ids=None, texts=None):
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, index in enumerate(indices):
        entries.append(
            {
                "episode_index": int(index),
                "episode_id": int(ids[i]) if ids else 25000 + int(index),
                "used_text": (texts[i] if texts else f"text {index}"),
            }
        )
    (directory / "episodes.json").write_text(json.dumps({"episodes": entries}, indent=2) + "\n")
    return str(directory)


def _fake_record(*, episode_index, episode_id, window_start, fill=None):
    fill = float(window_start if fill is None else fill)
    return {
        "name": gen.overfit100_window_name(episode_id, window_start).encode(),
        "episode_id": int(episode_id),
        "episode_index": int(episode_index),
        "window_start": int(window_start),
        "z_i0": np.full((C, 1, H, W), fill, dtype=np.float16).tobytes(),
        "z_video": np.full((C, F, H, W), fill, dtype=np.float16).tobytes(),
        "instruction": f"text {episode_index}".encode(),
    }


def _reader_config(**overrides):
    base = {
        "latent_channels": C,
        "latent_frames": F,
        "latent_height": H,
        "latent_width": W,
        "eval_data_dir": "gs://fake/train100",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------------------
# (A) Canonical-window math.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_windows,expected",
    [
        (1, 0),  # single-window episode -> the only window
        (2, 0),  # floor((2-1)/2) = 0
        (3, 4),
        (4, 4),
        (5, 8),
        (15, 28),
        (26, 48),
        (99, 196),  # the longest manifest episode
    ],
)
def test_canonical_window_start(n_windows, expected):
    assert gen.canonical_window_start(n_windows) == expected


def test_canonical_window_start_is_the_plan_formula_over_the_whole_range():
    for n in range(1, 200):
        assert gen.canonical_window_start(n) == 4 * ((n - 1) // 2)


@pytest.mark.parametrize("bad", [0, -1, -4])
def test_canonical_window_start_rejects_non_positive_window_counts(bad):
    with pytest.raises(ValueError):
        gen.canonical_window_start(bad)


def test_canonical_start_is_always_a_real_window_of_the_episode():
    for n in range(1, 200):
        start = gen.canonical_window_start(n)
        assert start % 4 == 0
        assert 0 <= start <= 4 * (n - 1)


def test_manifest_episode_windows_reads_ids_texts_and_counts(tmp_path):
    path = _manifest(
        tmp_path,
        [
            {"episode_index": 0, "episode_id": 25189, "n_windows": 26, "used_text": "fold cloth"},
            {"episode_index": 1, "episode_id": 30000, "n_windows": 1, "used_text": "press button"},
            {"episode_index": 2, "episode_id": 40000, "n_windows": 99, "used_text": "hang towel"},
        ],
    )
    out = gen.manifest_episode_windows(path)
    assert sorted(out) == [0, 1, 2]
    assert out[0] == {
        "episode_index": 0,
        "episode_id": 25189,
        "used_text": "fold cloth",
        "n_windows": 26,
        "canonical_start": 48,
        "canonical_name": "ep25189_v0_s00048",
    }
    assert out[1]["canonical_start"] == 0 and out[1]["canonical_name"] == "ep30000_v0_s00000"
    assert out[2]["canonical_start"] == 196 and out[2]["canonical_name"] == "ep40000_v0_s00196"


def test_manifest_episode_windows_matches_the_committed_manifest():
    out = gen.manifest_episode_windows(str(_MANIFEST))
    manifest = json.loads(_MANIFEST.read_text())
    assert len(out) == 100 == len(manifest["episodes"])
    for entry in manifest["episodes"]:
        index = entry["episode_index"]
        assert out[index]["episode_id"] == entry["episode_id"]
        assert out[index]["n_windows"] == entry["n_windows"]
        assert out[index]["canonical_start"] == 4 * ((entry["n_windows"] - 1) // 2)
    # The cohort really does contain both edge shapes the math must handle.
    counts = [e["n_windows"] for e in manifest["episodes"]]
    assert min(counts) == 1 and max(counts) == 99


def test_manifest_episode_windows_rejects_a_missing_n_windows(tmp_path):
    payload = {"episodes": [{"episode_index": 0, "episode_id": 1, "used_text": "t"}]}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((KeyError, ValueError)) as ei:
        gen.manifest_episode_windows(str(path))
    assert "n_windows" in str(ei.value)


# --------------------------------------------------------------------------------------
# (B) The --eval-windows spec.
# --------------------------------------------------------------------------------------


def test_parse_eval_windows_spec_canonical():
    for text in ("canonical", " canonical ", "CANONICAL"):
        assert gen.parse_eval_windows_spec(text) == ("canonical", ())


def test_parse_eval_windows_spec_explicit_names_keep_listed_order():
    kind, names = gen.parse_eval_windows_spec(" ep2_v0_s00008 , ep1_v0_s00000 ")
    assert kind == "names"
    assert names == ("ep2_v0_s00008", "ep1_v0_s00000")


def test_parse_eval_windows_spec_rejects_empty_and_duplicates():
    for bad in ("", "   ", ",,"):
        with pytest.raises(ValueError) as ei:
            gen.parse_eval_windows_spec(bad)
        assert "eval_windows" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        gen.parse_eval_windows_spec("ep1_v0_s00000,ep1_v0_s00000")
    assert "duplicate" in str(ei.value).lower()


def test_parse_eval_windows_spec_rejects_a_malformed_name():
    with pytest.raises(ValueError) as ei:
        gen.parse_eval_windows_spec("ep1_s0")
    assert "ep1_s0" in str(ei.value)


def _three_episode_manifest(tmp_path):
    return _manifest(
        tmp_path,
        [
            {"episode_index": 0, "episode_id": 100, "n_windows": 5},  # canonical s=8
            {"episode_index": 1, "episode_id": 200, "n_windows": 1},  # canonical s=0
            {"episode_index": 2, "episode_id": 300, "n_windows": 4},  # canonical s=4
        ],
    )


def test_select_canonical_windows_one_per_episode_ordered_by_index(tmp_path):
    episodes = gen.manifest_episode_windows(_three_episode_manifest(tmp_path))
    out = gen.select_eval_windows(("canonical", ()), episodes)
    assert [w["name"] for w in out] == ["ep100_v0_s00008", "ep200_v0_s00000", "ep300_v0_s00004"]
    assert [w["episode_index"] for w in out] == [0, 1, 2]
    assert [w["window_start"] for w in out] == [8, 0, 4]
    assert [w["episode_id"] for w in out] == [100, 200, 300]
    assert all(w["canonical"] is True for w in out)


def test_select_explicit_names_in_listed_order_with_canonical_flags(tmp_path):
    episodes = gen.manifest_episode_windows(_three_episode_manifest(tmp_path))
    out = gen.select_eval_windows(("names", ("ep300_v0_s00004", "ep100_v0_s00000")), episodes)
    assert [w["name"] for w in out] == ["ep300_v0_s00004", "ep100_v0_s00000"]
    assert [w["canonical"] for w in out] == [True, False]  # s=0 is not episode 0's canonical (8)
    assert [w["episode_index"] for w in out] == [2, 0]


def test_select_explicit_name_for_an_unknown_episode_is_refused(tmp_path):
    episodes = gen.manifest_episode_windows(_three_episode_manifest(tmp_path))
    with pytest.raises(ValueError) as ei:
        gen.select_eval_windows(("names", ("ep999_v0_s00000",)), episodes)
    assert "999" in str(ei.value)


def test_select_explicit_name_past_the_last_window_is_refused(tmp_path):
    episodes = gen.manifest_episode_windows(_three_episode_manifest(tmp_path))
    # episode 200 has ONE window (start 0); s=4 does not exist.
    with pytest.raises(ValueError) as ei:
        gen.select_eval_windows(("names", ("ep200_v0_s00004",)), episodes)
    msg = str(ei.value)
    assert "ep200_v0_s00004" in msg and "1" in msg


def test_select_explicit_off_grid_start_is_refused(tmp_path):
    episodes = gen.manifest_episode_windows(_three_episode_manifest(tmp_path))
    with pytest.raises(ValueError) as ei:
        gen.select_eval_windows(("names", ("ep100_v0_s00003",)), episodes)
    assert "ep100_v0_s00003" in str(ei.value)


# --------------------------------------------------------------------------------------
# (C) Sparse eval sets (cycle-C ratified semantics).
# --------------------------------------------------------------------------------------


def test_resolve_eval_windows_uses_the_sets_own_sparse_episode_list(tmp_path):
    manifest = _manifest(
        tmp_path,
        [{"episode_index": i, "episode_id": 100 + i, "n_windows": 1 + i} for i in range(10)],
    )
    data_dir = _episodes_sidecar(tmp_path / "sparse", [0, 5, 7], ids=[100, 105, 107])
    config = _reader_config(eval_data_dir=data_dir, eval_windows="canonical", model_manifest_path=manifest)
    out = gen.resolve_eval_windows(config)
    # ONLY the three episodes the set contains, ordered by episode_index, canonical starts
    # from the MANIFEST's n_windows (1, 6, 8) -> 0, 4*2=8, 4*3=12.
    assert [w["episode_index"] for w in out] == [0, 5, 7]
    assert [w["window_start"] for w in out] == [0, 8, 12]
    assert [w["episode_id"] for w in out] == [100, 105, 107]


def test_resolve_eval_windows_refuses_an_index_the_manifest_does_not_define(tmp_path):
    manifest = _manifest(tmp_path, [{"episode_index": 0, "episode_id": 100, "n_windows": 3}])
    data_dir = _episodes_sidecar(tmp_path / "bad", [0, 4], ids=[100, 104])
    config = _reader_config(eval_data_dir=data_dir, eval_windows="canonical", model_manifest_path=manifest)
    with pytest.raises(ValueError) as ei:
        gen.resolve_eval_windows(config)
    assert "4" in str(ei.value)


def test_resolve_eval_windows_refuses_an_episode_id_that_disagrees_with_the_manifest(tmp_path):
    manifest = _manifest(tmp_path, [{"episode_index": 0, "episode_id": 100, "n_windows": 3}])
    data_dir = _episodes_sidecar(tmp_path / "drift", [0], ids=[999])
    config = _reader_config(eval_data_dir=data_dir, eval_windows="canonical", model_manifest_path=manifest)
    with pytest.raises(ValueError) as ei:
        gen.resolve_eval_windows(config)
    msg = str(ei.value)
    assert "999" in msg and "100" in msg


def test_resolve_eval_windows_requires_a_manifest_path(tmp_path):
    data_dir = _episodes_sidecar(tmp_path / "nomanifest", [0])
    config = _reader_config(eval_data_dir=data_dir, eval_windows="canonical", model_manifest_path="")
    with pytest.raises(ValueError) as ei:
        gen.resolve_eval_windows(config)
    assert "model_manifest_path" in str(ei.value)


# --------------------------------------------------------------------------------------
# (D) The schema-v2 reader.
# --------------------------------------------------------------------------------------


def test_overfit100_feature_description_is_schema_v2_with_the_aggregation_fields():
    keys = set(gen._overfit100_feature_description())
    assert keys == {"name", "episode_id", "episode_index", "window_start", "z_i0", "z_video", "instruction"}
    assert "actions" not in keys  # schema v2 has none
    assert "ordinal" not in keys  # that is the exp_01 cache field


def test_read_overfit100_samples_returns_the_requested_windows_in_order(monkeypatch):
    records = [
        _fake_record(episode_index=0, episode_id=100, window_start=0),
        _fake_record(episode_index=0, episode_id=100, window_start=4),
        _fake_record(episode_index=1, episode_id=200, window_start=0),
    ]
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(records))
    wanted = [
        {"name": "ep200_v0_s00000", "episode_id": 200, "episode_index": 1, "window_start": 0, "canonical": True},
        {"name": "ep100_v0_s00004", "episode_id": 100, "episode_index": 0, "window_start": 4, "canonical": True},
    ]
    samples = gen.read_overfit100_samples(_reader_config(), wanted)
    assert [s.name for s in samples] == ["ep200_v0_s00000", "ep100_v0_s00004"]
    assert [s.episode_index for s in samples] == [1, 0]
    assert [s.window_start for s in samples] == [0, 4]
    assert [s.position for s in samples] == [2, 1]  # dataset positions ride along
    assert samples[0].z_video.shape == (C, F, H, W) and samples[0].z_i0.shape == (C, 1, H, W)
    assert samples[0].z_video.dtype == np.float32  # decoded from f16 to f32 like exp_01
    assert samples[1].z_video[0, 0, 0, 0] == pytest.approx(4.0)  # the buffer value IS the start
    assert samples[0].instruction == "text 1"
    assert samples[0].canonical is True


def test_read_overfit100_samples_is_a_sparse_subset_read(monkeypatch):
    # 50 records on disk, 2 requested: the other 48 are simply not returned (no error).
    records = [
        _fake_record(episode_index=i // 5, episode_id=100 + i // 5, window_start=4 * (i % 5)) for i in range(50)
    ]
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(records))
    wanted = [
        {"name": "ep100_v0_s00008", "episode_id": 100, "episode_index": 0, "window_start": 8, "canonical": True},
        {"name": "ep109_v0_s00004", "episode_id": 109, "episode_index": 9, "window_start": 4, "canonical": True},
    ]
    samples = gen.read_overfit100_samples(_reader_config(), wanted)
    assert [s.name for s in samples] == [w["name"] for w in wanted]


def test_read_overfit100_samples_missing_name_is_actionable(monkeypatch):
    records = [_fake_record(episode_index=0, episode_id=100, window_start=0)]
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(records))
    wanted = [
        {"name": "ep100_v0_s00048", "episode_id": 100, "episode_index": 0, "window_start": 48, "canonical": True}
    ]
    with pytest.raises(ValueError) as ei:
        gen.read_overfit100_samples(_reader_config(), wanted)
    msg = str(ei.value)
    assert "ep100_v0_s00048" in msg and "1" in msg  # the name and the record count seen


def test_read_overfit100_samples_refuses_a_duplicated_name_in_the_dataset(monkeypatch):
    dup = _fake_record(episode_index=0, episode_id=100, window_start=0)
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter([dup, dict(dup)]))
    wanted = [{"name": "ep100_v0_s00000", "episode_id": 100, "episode_index": 0, "window_start": 0, "canonical": True}]
    with pytest.raises(ValueError) as ei:
        gen.read_overfit100_samples(_reader_config(), wanted)
    assert "duplicate" in str(ei.value).lower()


def test_read_overfit100_samples_refuses_a_record_whose_fields_contradict_the_selection(monkeypatch):
    # Same name, different stored episode_index -> the context gather would use the wrong row.
    record = _fake_record(episode_index=7, episode_id=100, window_start=0)
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter([record]))
    wanted = [{"name": "ep100_v0_s00000", "episode_id": 100, "episode_index": 0, "window_start": 0, "canonical": True}]
    with pytest.raises(ValueError) as ei:
        gen.read_overfit100_samples(_reader_config(), wanted)
    msg = str(ei.value)
    assert "episode_index" in msg and "7" in msg


# --------------------------------------------------------------------------------------
# (E) The aggregation artifact (G4).
# --------------------------------------------------------------------------------------


def _artifact_config(**overrides):
    base = {
        "run_name": "ovf-s3",
        "checkpoint_dir": "gs://b/ck",
        "eval_data_dir": "gs://v6_east1d/datasets/exp02_overfit100/train100",
        "train_data_dir": "gs://v6_east1d/datasets/exp02_overfit100/train100",
        "model_manifest_path": "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json",
        "model_type": "OVERFIT100_TI2V",
        "eval_pass_role": "s3_segment_final",
        "eval_windows": "canonical",
        "rollout_seeds": "0,1,2",
        "context_modes": "correct,null,shuffled",
        "context_shuffle_seed": 0,
        "side_adapter_sampling_steps": 25,
        "side_adapter_guide_scale": 1.0,
        "num_text_slots": 100,
        "flagged_windows": "",
        "write_videos": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _row(name="ep100_v0_s00008", **kw):
    base = {
        "name": name,
        "episode_id": 100,
        "episode_index": 0,
        "window_start": 8,
        "canonical": True,
        "checkpoint_step": 2500,
        "seed": 0,
        "context_mode": "correct",
        "context_source_episode_index": 0,
        "ssim": 0.96,
        "latent_mse": 0.01,
        "pixel_mse": 0.002,
        "ssim_vs_rgb": None,
        "pixel_mse_vs_rgb": None,
        "vae_ceiling_ssim": None,
        "aux_status": "unavailable",
    }
    base.update(kw)
    return base


def _artifact_kwargs(windows, *, role="s3_segment_final", cohort=None, all_keys=None):
    """The cycle-D artifact contract: the manifest-derived cohort, the role, and the hash (D1/D2)."""
    keys = tuple((int(w["episode_id"]), int(w["window_start"])) for w in windows)
    cohort = keys if cohort is None else tuple(cohort)
    all_keys = cohort if all_keys is None else tuple(all_keys)
    return {
        "pass_role": role,
        "canonical_cohort": cohort,
        "all_window_keys": all_keys,
        "manifest_sha256": "a" * 64,
        "role_validation": {"role": role, "ok": True},
    }


def test_aggregation_artifact_schema_and_metadata():
    windows = [
        {"name": "ep100_v0_s00008", "episode_id": 100, "episode_index": 0, "window_start": 8, "canonical": True},
        {"name": "ep200_v0_s00000", "episode_id": 200, "episode_index": 1, "window_start": 0, "canonical": True},
    ]
    rows = [_row(), _row(name="ep200_v0_s00000", episode_id=200, episode_index=1, window_start=0, ssim=0.20)]
    art = gen.overfit100_aggregation_artifact(
        _artifact_config(),
        checkpoint_step=2500,
        windows=windows,
        rows=rows,
        seeds=[0, 1, 2],
        modes=["correct", "null", "shuffled"],
        derangement=(1, 0),
        flagged_windows=[],
        **_artifact_kwargs(windows),
    )
    assert art["schema"] == gen.OVERFIT100_AGGREGATION_SCHEMA
    for key in (
        "checkpoint_step",
        "run_name",
        "eval_data_dir",
        "train_data_dir",
        "model_manifest_path",
        "eval_windows_spec",
        "rollout_seeds",
        "context_modes",
        "context_shuffle_seed",
        "windows",
        "rows",
        "flagged_windows",
        "sampling_steps",
        "guide_scale",
        "context_derangement",
        # cycle-D strengthening (D1/D2): role, hash, and the FIXED cohort with coverage sets.
        "eval_pass_role",
        "role_validation",
        "manifest_sha256",
        "canonical_cohort",
        "cohort_size",
        "covered_windows",
        "covered_canonical_windows",
        "missing_canonical_windows",
    ):
        assert key in art, key
    assert art["eval_pass_role"] == "s3_segment_final"
    assert art["manifest_sha256"] == "a" * 64
    assert art["checkpoint_step"] == 2500
    assert art["rollout_seeds"] == [0, 1, 2]
    assert art["context_modes"] == ["correct", "null", "shuffled"]
    assert art["canonical_cohort"] == [[100, 8], [200, 0]]  # the FIXED (episode_id, window_start) keys
    assert art["covered_canonical_windows"] == [[100, 8], [200, 0]]
    assert art["missing_canonical_windows"] == []
    assert art["num_windows"] == 2
    assert art["rows"] == rows


def test_aggregation_artifact_json_round_trip_and_statistic_consumption():
    windows = [
        {"name": "ep100_v0_s00008", "episode_id": 100, "episode_index": 0, "window_start": 8, "canonical": True}
    ]
    rows = [_row(seed=s, ssim=ssim) for s, ssim in zip((0, 1, 2), (0.96, 0.97, 0.10))]
    art = gen.overfit100_aggregation_artifact(
        _artifact_config(),
        checkpoint_step=2500,
        windows=windows,
        rows=rows,
        seeds=[0, 1, 2],
        modes=["correct"],
        derangement=(0,),
        flagged_windows=[],
        **_artifact_kwargs(windows),
    )
    text = json.dumps(art, indent=2, sort_keys=True)
    assert json.loads(text) == art

    import maxdiffusion.overfit100_success_statistic as stat

    out = stat.evaluate_success(
        stat.rows_from_artifacts([json.loads(text)]),
        canonical_windows=[(100, 8)],
        segment_final_checkpoints=[2500],
    )
    assert out["per_checkpoint"][0]["m_corr"]["[100, 8]"] == pytest.approx(0.96)  # median, not mean


def test_aggregation_artifact_rejects_a_row_missing_a_required_column():
    windows = [
        {"name": "ep100_v0_s00008", "episode_id": 100, "episode_index": 0, "window_start": 8, "canonical": True}
    ]
    bad = _row()
    del bad["episode_id"]
    with pytest.raises((KeyError, ValueError)) as ei:
        gen.overfit100_aggregation_artifact(
            _artifact_config(),
            checkpoint_step=2500,
            windows=windows,
            rows=[bad],
            seeds=[0],
            modes=["correct"],
            derangement=(0,),
            flagged_windows=[],
            **_artifact_kwargs(windows),
        )
    assert "episode_id" in str(ei.value)


def test_assert_ssim_available_is_the_statistics_input_gate(monkeypatch):
    # SSIM is D11's only statistic input, so a worker without scikit-image must fail in seconds
    # rather than write a full artifact of NaNs. Both branches are exercised through sys.modules,
    # so the test is independent of whether the local env happens to have scikit-image.
    fake_metrics = types.ModuleType("skimage.metrics")
    fake_metrics.structural_similarity = lambda *a, **k: 1.0
    monkeypatch.setitem(sys.modules, "skimage", types.ModuleType("skimage"))
    monkeypatch.setitem(sys.modules, "skimage.metrics", fake_metrics)
    gen.assert_ssim_available()  # available -> silent

    monkeypatch.setitem(sys.modules, "skimage.metrics", None)
    with pytest.raises(ValueError) as ei:
        gen.assert_ssim_available()
    msg = str(ei.value)
    assert "scikit-image" in msg and "SSIM" in msg
    # And the driver gates on it BEFORE the checkpoint restore / 5B load.
    src = inspect.getsource(gen.run_overfit100)
    assert src.index("assert_ssim_available") < src.index("_restore_overfit100_validation_state")


def test_driver_gates_role_flags_and_ssim_before_touching_the_checkpoint():
    # All three seconds-cheap refusals (D1/D2 + the SSIM gate) must precede the 5B load.
    src = inspect.getsource(gen.run_overfit100)
    restore = src.index("_restore_overfit100_validation_state")
    for gate in (
        "assert_ssim_available",
        "parse_eval_pass_role",
        "assert_flagged_windows_in_cohort",
        "assert_pass_role_plan",
    ):
        assert src.index(gate) < restore, gate
    # Flags are checked against the manifest-derived COHORT (ruling 8's SPLIT), not the selection.
    assert "assert_flagged_windows_in_cohort(flagged, canonical_cohort)" in src


def test_flagged_windows_parse_and_ride_into_the_artifact():
    assert gen.parse_flagged_windows("") == ()
    assert gen.parse_flagged_windows(" ep100_v0_s00008 , ep200_v0_s00000 ") == (
        "ep100_v0_s00008",
        "ep200_v0_s00000",
    )
    windows = [
        {"name": "ep100_v0_s00008", "episode_id": 100, "episode_index": 0, "window_start": 8, "canonical": True},
    ]
    art = gen.overfit100_aggregation_artifact(
        _artifact_config(flagged_windows="ep100_v0_s00008"),
        checkpoint_step=2500,
        windows=windows,
        rows=[_row()],
        seeds=[0],
        modes=["correct"],
        derangement=(0,),
        flagged_windows=["ep100_v0_s00008"],
        **_artifact_kwargs(windows),
    )
    # Flagged windows are RECORDED and stay in the denominator (never dropped, G4).
    assert art["flagged_windows"] == [[100, 8]]
    assert art["canonical_cohort"] == [[100, 8]]
    assert art["num_windows"] == 1


# --------------------------------------------------------------------------------------
# (F) The bash arms.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("script", [_VALIDATE, _LOSS_EVAL])
def test_bash_arms_exist_and_pass_bash_n(script):
    assert script.exists(), f"missing {script}"
    bash_exe = shutil.which("bash") or "/bin/bash"
    proc = subprocess.run([bash_exe, "-n", str(script)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr


def test_validate_arm_is_manifest_pinned_and_forwards_the_eval_knobs():
    text = _VALIDATE.read_text()
    assert "src/maxdiffusion/configs/base_wan_5b_overfit100.yml" in text
    assert "src/maxdiffusion/generate_wan_side_adapter.py" in text
    # Manifest-driven pinned prefetch, exactly like train_wan_overfit100.sh.
    assert "MANIFEST_PATH" in text
    assert "vae_fingerprint" in text and "revision" in text
    assert "[0-9a-f]{40}" in text
    assert "bash bash_scripts/prefetch_hf_snapshot.sh" in text
    assert "local_files_only=True" in text
    assert "export COMMIT" in text
    for override in (
        'eval_pass_role="${EVAL_PASS_ROLE}"',
        'eval_windows="${EVAL_WINDOWS}"',
        'rollout_seeds="${ROLLOUT_SEEDS}"',
        'context_modes="${CONTEXT_MODES}"',
        'context_shuffle_seed="${CONTEXT_SHUFFLE_SEED}"',
        'write_videos="${WRITE_VIDEOS}"',
        'checkpoint_step="${CHECKPOINT_STEP}"',
        'model_manifest_path="${MANIFEST_PATH}"',
        'expected_model_revision="${MODEL_REVISION}"',
    ):
        assert override in text, override


def test_loss_eval_arm_forwards_the_overfit100_knobs():
    text = _LOSS_EVAL.read_text()
    assert "src/maxdiffusion/configs/base_wan_5b_overfit100.yml" in text
    assert "src/maxdiffusion/eval_wan_full_ft_val_loss.py" in text
    assert "MANIFEST_PATH" in text and "export COMMIT" in text
    assert "TRAIN_COMMIT" in text
    for override in (
        'validation_checkpoint_steps="${CHECKPOINT_STEPS}"',
        'validation_expected_count="${EXPECTED_COUNT}"',
        'num_text_slots="${NUM_TEXT_SLOTS}"',
        'expected_windows="${EXPECTED_WINDOWS}"',
    ):
        assert override in text, override


def test_config_carries_the_new_eval_keys_for_pyconfig_overridability():
    import yaml

    cfg = yaml.safe_load(_CONFIG.read_text())
    for key, kind in (
        ("eval_pass_role", str),
        ("eval_windows", str),
        ("rollout_seeds", str),
        ("context_modes", str),
        ("context_shuffle_seed", int),
        ("write_videos", bool),
        ("eval_aux_rgb", bool),
        ("flagged_windows", str),
    ):
        assert key in cfg, key
        assert isinstance(cfg[key], kind), (key, type(cfg[key]))
    # The shipped defaults are the D11 S3 segment-final coverage.
    assert cfg["eval_windows"] == "canonical"
    assert cfg["rollout_seeds"] == "0,1,2"
    assert cfg["context_modes"] == "correct"
