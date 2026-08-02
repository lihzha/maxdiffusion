"""CPU-only tests for the exp_02 schema-v2 TFRecord writer (cycle B, deliverable B1/D6).

Plan v4 D6 fixes the record schema for BOTH built sets (`train100`, `train10`):

    name (bytes) | episode_id (int64) | episode_index (int64) | window_start (int64)
    z_i0 (bytes, f16 [48,1,12,20]) | z_video (bytes, f16 [48,9,12,20]) | instruction (bytes, UTF-8)

-- no `actions`. The cycle-C trainer parses exactly these fields and asserts its
`expected_windows` against the records it sees, so a writer that drifts (wrong dtype, wrong
element order, a truncated instruction) breaks training in a way that looks like a model
problem. Everything here is a local round-trip: no GCS, no accelerator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from maxdiffusion.data_preprocessing import build_overfit100_dataset as builder
from maxdiffusion.data_preprocessing.build_overfit100_dataset import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SCHEMA_V2_FIELDS,
    SHARD_SIZE,
    TRAIN10_MAX_EPISODE_INDEX,
    Z_I0_SHAPE,
    Z_VIDEO_SHAPE,
    BuildError,
    expected_window_counts,
    in_train10,
    parse_window_record,
    select_train10,
    serialize_window_record,
    shard_filename,
    shard_ranges,
)

MANIFEST_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "worklogs_yixun"
    / "exp_02_overfit100_claude"
    / "overfit100_manifest.json"
)


def _window(seed=0, shape=Z_VIDEO_SHAPE):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) * 0.65).astype(np.float16)


def _record(seed=0, **overrides):
    z_video = overrides.pop("z_video", None)
    if z_video is None:
        z_video = _window(seed)
    payload = {
        "name": "ep25189_v0_s00004",
        "episode_id": 25189,
        "episode_index": 0,
        "window_start": 4,
        "z_i0": np.ascontiguousarray(z_video[:, :1]),
        "z_video": z_video,
        "instruction": "Move object into or out of container",
    }
    payload.update(overrides)
    return payload


# ----------------------------------------------------------------------------------
# 1. Schema v2 -- exact field set, and a full write -> parse round trip.
# ----------------------------------------------------------------------------------


def test_schema_v2_field_set_is_exactly_the_plan_list():
    assert tuple(SCHEMA_V2_FIELDS) == (
        "name",
        "episode_id",
        "episode_index",
        "window_start",
        "z_i0",
        "z_video",
        "instruction",
    )
    assert "actions" not in SCHEMA_V2_FIELDS  # D6: dropped from the exp_01 schema


def test_record_round_trips_through_a_real_tfrecord_file(tmp_path):
    record = _record()
    path = tmp_path / "train100-00000-of-00001.tfrecord"
    with tf.io.TFRecordWriter(str(path)) as writer:
        writer.write(serialize_window_record(**record))

    raw = list(tf.data.TFRecordDataset([str(path)]))
    assert len(raw) == 1
    parsed = parse_window_record(bytes(raw[0].numpy()))

    assert parsed["name"] == record["name"]
    assert parsed["episode_id"] == 25189 and isinstance(parsed["episode_id"], int)
    assert parsed["episode_index"] == 0 and parsed["window_start"] == 4
    assert parsed["instruction"] == record["instruction"]
    np.testing.assert_array_equal(parsed["z_video"], record["z_video"])
    np.testing.assert_array_equal(parsed["z_i0"], record["z_i0"])
    assert parsed["z_video"].dtype == np.float16 and parsed["z_i0"].dtype == np.float16
    assert parsed["z_video"].shape == Z_VIDEO_SHAPE and parsed["z_i0"].shape == Z_I0_SHAPE


def test_latent_payload_byte_lengths_are_the_declared_geometry():
    raw = serialize_window_record(**_record())
    feature = tf.train.Example.FromString(raw).features.feature
    assert set(feature) == set(SCHEMA_V2_FIELDS)
    assert len(feature["z_video"].bytes_list.value[0]) == int(np.prod(Z_VIDEO_SHAPE)) * 2
    assert len(feature["z_i0"].bytes_list.value[0]) == int(np.prod(Z_I0_SHAPE)) * 2


def test_instruction_survives_non_ascii_as_utf8():
    text = "Coloca la piñata en la caja — 把杯子放好"
    raw = serialize_window_record(**_record(instruction=text))
    feature = tf.train.Example.FromString(raw).features.feature
    assert feature["instruction"].bytes_list.value[0] == text.encode("utf-8")
    assert parse_window_record(raw)["instruction"] == text


def test_z_i0_is_stored_bitwise_equal_to_the_first_latent_frame():
    record = _record(seed=3)
    parsed = parse_window_record(serialize_window_record(**record))
    assert parsed["z_i0"].tobytes() == np.ascontiguousarray(parsed["z_video"][:, :1]).tobytes()


def test_writer_rejects_wrong_shapes_or_dtypes():
    with pytest.raises(BuildError):
        serialize_window_record(**_record(z_video=_window(shape=(48, 5, 12, 20))))
    with pytest.raises(BuildError):
        serialize_window_record(**_record(z_i0=np.zeros((48, 1, 12, 20), dtype=np.float32)))


def test_parse_rejects_a_truncated_payload():
    record = _record()
    raw = serialize_window_record(**record)
    feature = tf.train.Example.FromString(raw).features.feature
    broken = tf.train.Example(
        features=tf.train.Features(
            feature={
                **{k: feature[k] for k in SCHEMA_V2_FIELDS if k != "z_video"},
                "z_video": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"\x00\x01"])),
            }
        )
    ).SerializeToString()
    with pytest.raises(ValueError):
        parse_window_record(broken)


# ----------------------------------------------------------------------------------
# 2. train10 -- the D6 subset filter and its count assert.
# ----------------------------------------------------------------------------------


def test_in_train10_is_the_first_ten_manifest_episode_indices():
    assert TRAIN10_MAX_EPISODE_INDEX == 10
    assert [i for i in range(13) if in_train10(i)] == list(range(10))


def test_select_train10_keeps_order_and_drops_later_episodes():
    records = [{"episode_index": idx, "window_start": start} for idx in range(12) for start in (0, 4)]
    kept = select_train10(records)
    assert len(kept) == 20
    assert {r["episode_index"] for r in kept} == set(range(10))
    assert kept == [r for r in records if r["episode_index"] < 10]  # order preserved


def test_expected_window_counts_on_a_synthetic_manifest():
    manifest = {"episodes": [{"episode_index": i, "n_windows": i + 1} for i in range(12)]}
    counts = expected_window_counts(manifest)
    assert counts["train100"] == sum(range(1, 13))
    assert counts["train10"] == sum(range(1, 11))


def test_expected_window_counts_on_the_committed_manifest():
    # The numbers the builder asserts against, straight from the artifact cycle A committed.
    manifest = json.loads(MANIFEST_PATH.read_text())
    counts = expected_window_counts(manifest)
    assert counts["train100"] == 1629 == manifest["totals"]["windows"]
    assert counts["train10"] == 167


# ----------------------------------------------------------------------------------
# 3. Shard partitioning.
# ----------------------------------------------------------------------------------


def test_shard_size_is_the_plan_value():
    assert SHARD_SIZE == 256


def test_shard_ranges_partition_exactly_and_contiguously():
    for n in (0, 1, 255, 256, 257, 167, 1629):
        ranges = shard_ranges(n, SHARD_SIZE)
        assert [i for start, end in ranges for i in range(start, end)] == list(range(n))
        assert all(end - start <= SHARD_SIZE for start, end in ranges)


def test_shard_counts_for_the_two_built_sets():
    assert len(shard_ranges(1629, SHARD_SIZE)) == 7  # train100
    assert [end - start for start, end in shard_ranges(1629, SHARD_SIZE)] == [256] * 6 + [93]
    assert len(shard_ranges(167, SHARD_SIZE)) == 1  # train10


def test_shard_filename_is_zero_padded_and_globbable():
    assert shard_filename("train100", 0, 7) == "train100-00000-of-00007.tfrecord"
    assert shard_filename("train10", 0, 1) == "train10-00000-of-00001.tfrecord"
    names = [shard_filename("train100", i, 7) for i in range(7)]
    assert names == sorted(names)  # the loader globs *.tfrecord and sorts


def test_shard_ranges_rejects_a_nonpositive_shard_size():
    with pytest.raises(BuildError):
        shard_ranges(10, 0)


# ----------------------------------------------------------------------------------
# 4. The driver end to end, with a stubbed VAE (the only local exercise of `run()`).
# ----------------------------------------------------------------------------------

_STUB_EPISODES = 12
_STUB_FRAMES = 37  # -> exactly 2 windows per episode
_VAE_PIN = {
    "hf_repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "revision": "b8fff7315c768468a5333511427288870b2e9635",
    "vae_config_sha256": "d996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360",
}


class _StubDistribution:
    def __init__(self, latents):
        self._latents = latents

    def mode(self):
        return self._latents

    def sample(self, *args, **kwargs):
        raise AssertionError("the build must never sample the posterior")


class _StubVae:
    """Returns latents of the real geometry so the schema-v2 writer runs unmodified."""

    z_dim = 48
    latents_mean = [0.0] * 48
    latents_std = [1.0] * 48
    dtype = np.float32

    def __init__(self, std=0.65):
        self.latents = (np.random.default_rng(0).normal(size=(1, 9, 12, 20, 48)) * std).astype(np.float32)

    def encode(self, x, feat_cache, return_dict=True):
        frames = int(x.shape[2]) if x.shape[1] == 3 else int(x.shape[1])
        latent_frames = 1 + (frames - 1) // 4
        return (_StubDistribution(self.latents[:, :latent_frames]),)

    def decode(self, z, feat_cache, return_dict=True):
        frames = 1 + (int(z.shape[2]) - 1) * 4
        return (np.zeros((1, frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32),)


class _StubPipeline:
    def __init__(self, std=0.65):
        self.vae = _StubVae(std)
        self.vae_cache = "CACHE"
        self.vae_mesh = None
        self.vae_logical_axis_rules = None


def _stub_manifest():
    return {
        "selection_seed": 0,
        "builder_commit": "a" * 40,
        "totals": {"episodes": _STUB_EPISODES, "windows": _STUB_EPISODES * 2},
        "fixture": {"uri": "gs://bucket/fixture.npz", "names": ["a"], "md5": "x", "size_bytes": 1},
        "vae_fingerprint": dict(_VAE_PIN),
        "episodes": [
            {
                "episode_index": index,
                "episode_id": 1000 + index,
                "used_text": "fold the cloth" if index % 5 else "press button",
                "n_windows": 2,
                "ffprobe": {"nb_frames": _STUB_FRAMES, "width": FRAME_WIDTH, "height": FRAME_HEIGHT},
                "video_fingerprint": {"uri": f"gs://bucket/videos/{1000 + index}/0.mp4", "md5": "m", "size": 1},
            }
            for index in range(_STUB_EPISODES)
        ],
    }


def _stub_driver(monkeypatch, tmp_path, std=0.65, memory_stats=None):
    pipeline = _StubPipeline(std)
    frames = np.random.default_rng(1).integers(
        0, 256, size=(_STUB_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )
    reference = np.arange(4.0)
    monkeypatch.setattr(builder, "load_manifest", lambda *a, **k: _stub_manifest())
    monkeypatch.setattr(builder, "assert_manifest_matches_committed", lambda *a, **k: "f" * 64)
    monkeypatch.setattr(
        builder,
        "preflight",
        lambda *a, **k: {
            "fixture_path": tmp_path / "f.npz",
            "vae_fingerprint": dict(_VAE_PIN),
            "vae_snapshot_path": "/snap/b8fff73",
        },
    )
    monkeypatch.setattr(builder, "load_vae_pipeline", lambda *a, **k: pipeline)
    monkeypatch.setattr(
        builder, "run_v1_gate", lambda *a, **k: [builder.check_v1(n, reference, reference) for n in ("a", "b", "c")]
    )
    monkeypatch.setattr(builder, "fetch_pinned", lambda uri, fingerprint, destination: destination)
    monkeypatch.setattr(builder, "decode_mp4_frames", lambda *a, **k: frames)
    monkeypatch.setattr(builder, "frames_ssim", lambda *a, **k: 0.9)
    monkeypatch.setattr(builder, "collect_tool_versions", lambda: {"python": "3.12"})
    monkeypatch.setattr(builder, "assert_implementation_committed", lambda **k: "b" * 40)
    monkeypatch.setattr(
        builder,
        "device_memory_stats",
        lambda: [{"device": "TPU_0", "peak_bytes_in_use": 1 << 30}] if memory_stats is None else memory_stats,
    )
    monkeypatch.setattr(
        builder,
        "audit_benchmark",
        lambda *a, **k: {"seconds": 1.5, "peak_rss_bytes": 1 << 28, "n": 1629, "dim": 11520},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    return manifest_path


def _run(monkeypatch, tmp_path, extra=(), std=0.65, memory_stats=None):
    manifest_path = _stub_driver(monkeypatch, tmp_path, std, memory_stats)
    out_root = tmp_path / "out"
    args = builder._parse_args(
        ["--manifest", str(manifest_path), "--out-root", str(out_root), "--tmp-dir", str(tmp_path / "tmp"), *extra]
    )
    return builder.run(args), out_root


def _records(path: Path) -> list[dict]:
    return [parse_window_record(bytes(raw.numpy())) for raw in tf.data.TFRecordDataset([str(path)])]


def _shards(root: Path, set_name: str) -> list[Path]:
    return sorted((root / set_name).glob("*.tfrecord"))


def test_run_builds_both_sets_and_their_sidecars(monkeypatch, tmp_path):
    code, out_root = _run(monkeypatch, tmp_path)
    assert code == 0

    train100, train10 = _shards(out_root, "train100"), _shards(out_root, "train10")
    assert [p.name for p in train100] == ["train100-00000-of-00001.tfrecord"]
    assert [p.name for p in train10] == ["train10-00000-of-00001.tfrecord"]

    records100, records10 = _records(train100[0]), _records(train10[0])
    assert len(records100) == _STUB_EPISODES * 2
    assert len(records10) == 20  # episode_index 0-9, two windows each
    assert {r["episode_index"] for r in records10} == set(range(10))
    assert [r["window_start"] for r in records100[:2]] == [0, 4]
    assert records100[0]["name"] == "ep1000_v0_s00000"
    assert records100[0]["instruction"] == "press button"
    # train10 is a prefix-consistent subset of train100 -- identical records, not a re-encode.
    for subset, full in zip(records10, records100[:20]):
        assert subset["name"] == full["name"] and subset["instruction"] == full["instruction"]
        assert subset["z_video"].tobytes() == full["z_video"].tobytes()
        assert subset["z_i0"].tobytes() == full["z_i0"].tobytes()

    for set_name in ("train100", "train10"):
        for sidecar in ("summary.json", "episodes.json", "window_stats.json", "_SUCCESS"):
            assert (out_root / set_name / sidecar).exists()
    assert (out_root / "duplicate_audit.json").exists()

    summary = json.loads((out_root / "train100" / "summary.json").read_text())
    assert summary["sets"]["train100"]["written"] == _STUB_EPISODES * 2
    assert summary["sets"]["train10"]["written"] == 20
    assert summary["build_commit"] == "b" * 40
    assert summary["vae_fingerprint"] == _VAE_PIN
    assert summary["vae_snapshot_path"] == "/snap/b8fff73"
    assert summary["manifest"]["sha256"] == "f" * 64
    assert summary["gates"]["v2"]["n_windows"] == _STUB_EPISODES * 2
    assert [g["name"] for g in summary["gates"]["v3"]] == ["ep1000_v0_s00000", "ep1010_v0_s00000"]
    assert summary["gates"]["v4"]["passed"] is True
    assert summary["encode"]["rng"].startswith("none")

    episodes10 = json.loads((out_root / "train10" / "episodes.json").read_text())["episodes"]
    assert [e["episode_index"] for e in episodes10] == list(range(10))
    stats10 = json.loads((out_root / "train10" / "window_stats.json").read_text())["windows"]
    assert len(stats10) == 20 and all(s["episode_index"] < 10 for s in stats10)

    audit = json.loads((out_root / "duplicate_audit.json").read_text())
    assert audit["n_windows"] == _STUB_EPISODES * 2
    assert audit["min_pairwise_z_i0"]["n_pairs_compared"] == 0  # every window shares one target here
    assert sum(g["count"] for g in audit["duplicate_instruction_groups"]) == _STUB_EPISODES


# --- B3: staging -> readback -> promote -> _SUCCESS ---------------------------------


def test_success_marker_is_written_last_and_describes_the_build(monkeypatch, tmp_path):
    _, out_root = _run(monkeypatch, tmp_path)
    marker = json.loads((out_root / "train100" / "_SUCCESS").read_text())
    summary_bytes = (out_root / "train100" / "summary.json").read_bytes()
    assert marker["build_commit"] == "b" * 40
    assert marker["records"] == _STUB_EPISODES * 2
    assert marker["shards"] == 1
    assert marker["summary_sha256"] == hashlib.sha256(summary_bytes).hexdigest()
    assert marker["build_id"].startswith("b" * 12)


def test_staging_is_cleaned_up_after_promotion(monkeypatch, tmp_path):
    _, out_root = _run(monkeypatch, tmp_path)
    assert list(out_root.rglob("_staging_*")) == []


def test_run_refuses_a_non_empty_canonical_prefix(monkeypatch, tmp_path):
    out_root = tmp_path / "out"
    (out_root / "train10").mkdir(parents=True)
    stray = out_root / "train10" / "train10-00000-of-00001.tfrecord"
    stray.write_bytes(b"an older build")
    with pytest.raises(BuildError, match="not empty"):
        _run(monkeypatch, tmp_path)
    assert stray.read_bytes() == b"an older build"  # never silently overwritten
    assert not (out_root / "train10" / "_SUCCESS").exists()


def test_late_abort_after_a_flushed_shard_leaves_no_canonical_objects(monkeypatch, tmp_path):
    # shard-size 3 with 2 windows/episode: shards flush from episode 2 onward, so the
    # failure at episode 6 happens AFTER real uploads -- the exact B3 scenario.
    manifest_path = _stub_driver(monkeypatch, tmp_path)
    original = builder.check_v2

    def failing_v2(name, z_video):
        result = original(name, z_video)
        if name.startswith("ep1006"):
            result["std"], result["passed"] = 0.2, False
        return result

    monkeypatch.setattr(builder, "check_v2", failing_v2)
    out_root = tmp_path / "out"
    args = builder._parse_args(
        [
            "--manifest",
            str(manifest_path),
            "--out-root",
            str(out_root),
            "--tmp-dir",
            str(tmp_path / "tmp"),
            "--shard-size",
            "3",
        ]
    )
    with pytest.raises(builder.GateFailure):
        builder.run(args)

    assert _shards(out_root, "train100") == []  # nothing a trainer could glob
    assert not (out_root / "train100" / "_SUCCESS").exists()
    assert list((out_root / "train100").rglob("_staging_*/*.tfrecord"))  # the evidence stays in staging
    report = json.loads((out_root / "failed_gates.json").read_text())
    assert "V2" in report["error"]


# --- B4: shard fingerprints + physical readback --------------------------------------


def test_summary_fingerprints_every_shard(monkeypatch, tmp_path):
    _, out_root = _run(monkeypatch, tmp_path)
    summary = json.loads((out_root / "train100" / "summary.json").read_text())
    shard = summary["sets"]["train100"]["shards"][0]
    assert shard["records"] == _STUB_EPISODES * 2
    assert shard["sha256"] == hashlib.sha256(_shards(out_root, "train100")[0].read_bytes()).hexdigest()
    assert shard["size"] == _shards(out_root, "train100")[0].stat().st_size
    assert shard["name"] == "train100-00000-of-00001.tfrecord"


def test_readback_catches_a_shard_whose_bytes_changed_after_the_write(monkeypatch, tmp_path):
    manifest_path = _stub_driver(monkeypatch, tmp_path)
    original = builder.fetch_shard_bytes

    def corrupting_fetch(entry, tmp_dir):
        data = bytearray(original(entry, tmp_dir))
        data[-1] ^= 0xFF  # one flipped bit somewhere in the last record
        return bytes(data)

    monkeypatch.setattr(builder, "fetch_shard_bytes", corrupting_fetch)
    args = builder._parse_args(
        ["--manifest", str(manifest_path), "--out-root", str(tmp_path / "out"), "--tmp-dir", str(tmp_path / "tmp")]
    )
    with pytest.raises(BuildError, match="sha256|readback"):
        builder.run(args)
    assert _shards(tmp_path / "out", "train100") == []  # corruption is caught BEFORE promotion


def test_readback_catches_a_truncated_record(monkeypatch, tmp_path):
    manifest_path = _stub_driver(monkeypatch, tmp_path)
    monkeypatch.setattr(builder, "fetch_shard_bytes", lambda entry, tmp_dir: b"")
    args = builder._parse_args(
        ["--manifest", str(manifest_path), "--out-root", str(tmp_path / "out"), "--tmp-dir", str(tmp_path / "tmp")]
    )
    with pytest.raises(BuildError):
        builder.run(args)


def test_readback_asserts_the_exact_ordered_names(monkeypatch, tmp_path):
    entries = [{"name": "s0", "records": 2, "sha256": "x", "size": 1, "uri": "u"}]
    parsed = [{"name": "ep1_v0_s00000"}, {"name": "ep1_v0_s00004"}]
    with pytest.raises(BuildError, match="order|names"):
        builder.assert_readback_names(entries, [r["name"] for r in parsed], ["ep1_v0_s00004", "ep1_v0_s00000"])


# --- B5 / multi-shard: rollover both as arithmetic and end to end ---------------------


def test_probe_shard_size_forces_rollover_on_the_real_first_two_episodes():
    # manifest episode_index 0 and 1 carry 26 + 15 = 41 windows; 256 would be one shard.
    assert builder.PROBE_SHARD_SIZE == 16
    assert len(shard_ranges(41, builder.PROBE_SHARD_SIZE)) == 3


def test_run_rolls_over_shards_and_reads_every_one_back(monkeypatch, tmp_path):
    manifest_path = _stub_driver(monkeypatch, tmp_path)
    out_root = tmp_path / "out"
    args = builder._parse_args(
        [
            "--manifest",
            str(manifest_path),
            "--out-root",
            str(out_root),
            "--tmp-dir",
            str(tmp_path / "tmp"),
            "--shard-size",
            "5",
        ]
    )
    assert builder.run(args) == 0

    shards = _shards(out_root, "train100")
    assert [p.name for p in shards] == [f"train100-{i:05d}-of-00005.tfrecord" for i in range(5)]
    assert [len(_records(p)) for p in shards] == [5, 5, 5, 5, 4]
    names = [r["name"] for p in shards for r in _records(p)]
    assert names == sorted(names, key=lambda n: (int(n.split("_")[0][2:]), int(n.split("_s")[1])))
    summary = json.loads((out_root / "train100" / "summary.json").read_text())
    assert len(summary["sets"]["train100"]["shards"]) == 5
    assert json.loads((out_root / "train100" / "_SUCCESS").read_text())["shards"] == 5


# --- B5: the probe substantiates its scale/resource claims ---------------------------


def test_run_probe_writes_only_probe2_and_reports_phase_timings(monkeypatch, tmp_path):
    code, out_root = _run(monkeypatch, tmp_path, extra=["--probe", "--shard-size", "2"])
    assert code == 0
    assert not (out_root / "train100").exists() and not (out_root / "train10").exists()

    shards = _shards(out_root, "probe2")
    assert [p.name for p in shards] == ["probe2-00000-of-00002.tfrecord", "probe2-00001-of-00002.tfrecord"]
    assert sum(len(_records(p)) for p in shards) == 4  # 2 episodes x 2 windows

    summary = json.loads((out_root / "probe2" / "summary.json").read_text())
    assert summary["probe"] is True
    assert summary["build_commit"] == "b" * 40  # B6: probes are guarded like production
    assert summary["v2_coverage"] == "sampled (4/24 windows)"
    phases = summary["timing"]["phases"]
    assert {"preflight", "vae_load", "first_window_compile", "steady_state_encode", "upload", "audit", "total"} <= set(
        phases
    )
    assert summary["timing"]["audit_benchmark"]["n"] == 1629
    assert summary["timing"]["device_memory_stats"] == [{"device": "TPU_0", "peak_bytes_in_use": 1 << 30}]
    extrapolation = summary["extrapolation"]
    assert extrapolation["full_build_windows"] == _STUB_EPISODES * 2
    assert extrapolation["steady_state_windows_per_second"] > 0
    assert extrapolation["fixed_seconds"] >= 0
    assert extrapolation["estimated_seconds"] >= extrapolation["fixed_seconds"]
    assert (out_root / "probe2" / "_SUCCESS").exists()


def test_probe_fails_when_no_device_reports_peak_memory(monkeypatch, tmp_path):
    with pytest.raises(BuildError, match="memory"):
        _run(monkeypatch, tmp_path, extra=["--probe"], memory_stats=[])


def test_probe_fails_when_one_device_reports_no_peak(monkeypatch, tmp_path):
    stats = [{"device": "TPU_0", "peak_bytes_in_use": 1 << 30}, {"device": "TPU_1"}]
    with pytest.raises(BuildError, match="memory"):
        _run(monkeypatch, tmp_path, extra=["--probe"], memory_stats=stats)


def test_probe_prechecks_every_fixed_v3_window_including_unbuilt_episodes(monkeypatch, tmp_path):
    _, out_root = _run(monkeypatch, tmp_path, extra=["--probe", "--shard-size", "2"])
    summary = json.loads((out_root / "probe2" / "summary.json").read_text())
    # Episode index 10 is NOT part of the 2-episode probe build, but its V3 window is still
    # checked -- a V3 failure late in the manifest must surface before the full build.
    assert [g["name"] for g in summary["gates"]["v3"]] == ["ep1000_v0_s00000", "ep1010_v0_s00000"]
    assert all(g["passed"] for g in summary["gates"]["v3"])


def test_main_reports_a_gate_failure_as_a_nonzero_exit(monkeypatch, tmp_path):
    manifest_path = _stub_driver(monkeypatch, tmp_path, std=0.2)
    code = builder.main(
        ["--manifest", str(manifest_path), "--out-root", str(tmp_path / "out"), "--tmp-dir", str(tmp_path / "tmp")]
    )
    assert code == 1
