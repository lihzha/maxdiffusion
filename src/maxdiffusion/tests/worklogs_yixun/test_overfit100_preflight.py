"""CPU-only tests for the exp_02 overfit100 fail-closed startup preflight (cycle-C strengthen).

Codex cycle-C review findings C1-C3. Before ANY 5B weight is touched, the trainer must
bind three things it previously took on trust:

  (C1) THE MODEL REVISION -- the manifest pins ``b8fff731…``, but the config/launcher used to
       load the MUTABLE hub default, so a future repo update could silently change the
       transformer or the T5 that builds the context table while the run still claimed the
       reviewed recipe. ``_validate_pinned_snapshot`` now REQUIRES a resolved local snapshot
       directory whose path carries the pinned revision (HF layout: ``…/snapshots/<sha>/``),
       cross-checked against the committed manifest, for BOTH the pipeline path and the
       transformer path -- and it runs before the pipeline loader.

  (C2) THE PUBLISHED DATASET BYTES -- ``_SUCCESS`` existence + a metadata count was weaker
       than what cycle B publishes and ran only after the pipeline/optimizer/state were
       already built. ``verify_dataset_integrity`` now requires a structurally valid marker,
       binds ``summary.json``'s RAW BYTES to ``_SUCCESS.summary_sha256``, requires the exact
       canonical shard set, and verifies every shard's size + sha256 + md5 against the
       summary's fingerprints -- fail-closed, with no broad-except fallback (judgment 8).

  (C3) THE CONTEXT MAPPING -- the table is always built from ``train_data_dir``, so a
       DIFFERENT ``eval_data_dir`` with a valid-but-different ``episode_index`` mapping would
       evaluate against the wrong instruction with no error. Every shared index must agree on
       ``(episode_id, used_text)``.

CPU-only: whole cycle-B-shaped dataset directories are synthesized in ``tmp_path`` (real
TFRecord shards written by the REAL cycle-B writer, real summary/episodes/_SUCCESS sidecars
with real hashes), and every failure mode is a one-knob mutation of that fixture. No GCS, no
weights, no mesh. The darwin grain import stub lives in ``conftest.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100
from maxdiffusion.data_preprocessing.build_overfit100_dataset import (
    SUCCESS_MARKER,
    Z_VIDEO_SHAPE,
    serialize_window_record,
)

_PINNED = "b8fff7315c768468a5333511427288870b2e9635"
_OTHER_REV = "0123456789abcdef0123456789abcdef01234567"
_SNAPSHOT = f"/root/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots/{_PINNED}"


# =======================================================================================
# fixture: a complete, VALID cycle-B publication -- every test below mutates one knob
# =======================================================================================


def _md5_b64(payload: bytes) -> str:
    return base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")


def _publish_dataset(
    directory: Path,
    episode_indices,
    *,
    set_name=None,
    success=True,
    marker_records="auto",
    drop_marker_keys=(),
    marker_extra=None,
    summary_mutator=None,
    shard_mutator=None,
    extra_shard=False,
    drop_shard=False,
    episodes=None,
    instruction_of=None,
):
    """Write a cycle-B-shaped set: shard(s) + summary.json + episodes.json + _SUCCESS."""
    directory.mkdir(parents=True, exist_ok=True)
    set_name = set_name or directory.name
    rng = np.random.default_rng(0)
    shard_name = f"{set_name}-00000-of-00001.tfrecord"
    shard_path = directory / shard_name
    with tf.io.TFRecordWriter(str(shard_path)) as writer:
        for position, episode_index in enumerate(episode_indices):
            z_video = rng.standard_normal(Z_VIDEO_SHAPE).astype(np.float16)
            text = (instruction_of or (lambda i: f"instruction {i}"))(episode_index)
            writer.write(
                serialize_window_record(
                    name=f"ep{25189 + episode_index:05d}_v0_s{position * 4:05d}",
                    episode_id=25189 + episode_index,
                    episode_index=episode_index,
                    window_start=position * 4,
                    z_i0=z_video[:, 0:1].copy(),
                    z_video=z_video,
                    instruction=text,
                )
            )
    if shard_mutator is not None:
        shard_path.write_bytes(shard_mutator(shard_path.read_bytes()))
    payload = shard_path.read_bytes()
    shard_entry = {
        "name": shard_name,
        "records": len(episode_indices),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        # Recorded by cycle B from a stat of the STAGING object: md5 survives the
        # staging -> canonical copy, generation does NOT (see the trainer's docstring).
        "md5": _md5_b64(payload),
        "generation": "1700000000000001",
        "staging_uri": f"{directory}/_staging_/{shard_name}",
    }
    shards = [shard_entry]
    if extra_shard:
        stray = directory / f"{set_name}-00001-of-00002.tfrecord"
        stray.write_bytes(payload)  # a shard on disk that the summary does not list
    if drop_shard:
        shards = []  # the summary lists nothing while a shard exists on disk

    indices = sorted({int(i) for i in episode_indices})
    if episodes is None:
        episodes = [
            {
                "episode_index": index,
                "episode_id": 25189 + index,
                "used_text": (instruction_of or (lambda i: f"instruction {i}"))(index),
                "n_windows": sum(1 for j in episode_indices if j == index),
            }
            for index in indices
        ]
    (directory / "episodes.json").write_text(json.dumps({"episodes": episodes}, indent=2) + "\n")

    summary = {
        "built_utc": "2026-07-29T00:00:00+00:00",
        "build_id": "20260729-000000",
        "build_commit": "0" * 40,
        "episodes": len(episodes),
        "readers_must_require": SUCCESS_MARKER,
        "sets": {
            set_name: {
                "uri": str(directory),
                "expected_windows": len(episode_indices),
                "written": len(episode_indices),
                "shards": shards,
            }
        },
    }
    if summary_mutator is not None:
        summary = summary_mutator(summary)
    summary_bytes = json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n"
    (directory / "summary.json").write_text(summary_bytes)

    if success:
        marker = {
            "build_id": summary["build_id"],
            "build_commit": summary["build_commit"],
            "records": len(episode_indices) if marker_records == "auto" else marker_records,
            "shards": len(shards),
            "summary_sha256": hashlib.sha256(summary_bytes.encode("utf-8")).hexdigest(),
            "manifest_sha256": "f" * 64,
        }
        if marker_records is None:
            marker.pop("records")
        for key in drop_marker_keys:
            marker.pop(key, None)
        marker.update(marker_extra or {})
        (directory / SUCCESS_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
    return directory


def _dataset_config(train_dir, *, eval_dir=None, expected_windows=None, num_text_slots=None, **kw):
    train_dir = Path(train_dir)
    cfg = {
        "train_data_dir": str(train_dir),
        "eval_data_dir": str(eval_dir if eval_dir is not None else train_dir),
        "expected_windows": expected_windows if expected_windows is not None else 3,
        "num_text_slots": num_text_slots if num_text_slots is not None else 3,
        "text_encode_batch": 8,
        "checkpoint_steps": [250],
        "dataset_verify_bytes": True,
    }
    cfg.update(kw)
    return SimpleNamespace(**cfg)


# =======================================================================================
# (C1) the pinned model revision
# =======================================================================================


def _snapshot_config(**kw):
    cfg = {
        "pretrained_model_name_or_path": _SNAPSHOT,
        "wan_transformer_pretrained_model_name_or_path": _SNAPSHOT,
        "expected_model_revision": _PINNED,
        "model_manifest_path": "",
    }
    cfg.update(kw)
    return SimpleNamespace(**cfg)


def test_pinned_snapshot_accepted_and_revision_returned():
    assert overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(_snapshot_config()) == _PINNED


def test_bare_repo_id_is_rejected_as_unpinned():
    # THE C1 regression: the mutable hub default must not be loadable.
    with pytest.raises(ValueError) as ei:
        overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(
            _snapshot_config(
                pretrained_model_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                wan_transformer_pretrained_model_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            )
        )
    msg = str(ei.value)
    assert _PINNED in msg and "Wan-AI/Wan2.2-TI2V-5B-Diffusers" in msg


def test_snapshot_of_a_different_revision_is_rejected():
    wrong = _SNAPSHOT.replace(_PINNED, _OTHER_REV)
    with pytest.raises(ValueError) as ei:
        overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(
            _snapshot_config(pretrained_model_name_or_path=wrong, wan_transformer_pretrained_model_name_or_path=wrong)
        )
    assert _PINNED in str(ei.value)


def test_transformer_path_is_checked_too():
    # pyconfig copies pretrained_model_name_or_path into the transformer key when empty, but an
    # explicit unpinned transformer path must still be refused (it is what wan_pipeline loads).
    with pytest.raises(ValueError, match="wan_transformer_pretrained_model_name_or_path"):
        overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(
            _snapshot_config(wan_transformer_pretrained_model_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
        )


def test_missing_expected_revision_is_rejected():
    with pytest.raises(ValueError, match="expected_model_revision"):
        overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(_snapshot_config(expected_model_revision=""))


def test_non_sha_expected_revision_is_rejected():
    with pytest.raises(ValueError, match="40-hex|revision"):
        overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(_snapshot_config(expected_model_revision="main"))


def _write_manifest(path: Path, revision=_PINNED):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"vae_fingerprint": {"hf_repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "revision": revision}}, indent=2)
        + "\n"
    )
    return path


def test_manifest_supplies_the_revision_when_config_is_silent(tmp_path):
    manifest = _write_manifest(tmp_path / "overfit100_manifest.json")
    cfg = _snapshot_config(expected_model_revision="", model_manifest_path=str(manifest))
    assert overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(cfg) == _PINNED


def test_launcher_and_manifest_revision_disagreement_is_rejected(tmp_path):
    # A launcher that claims a revision the committed manifest does not pin is a
    # provenance break, even if the snapshot path matches the launcher's claim.
    manifest = _write_manifest(tmp_path / "overfit100_manifest.json", revision=_OTHER_REV)
    cfg = _snapshot_config(model_manifest_path=str(manifest))
    with pytest.raises(ValueError) as ei:
        overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(cfg)
    msg = str(ei.value)
    assert _PINNED in msg and _OTHER_REV in msg


def test_unreadable_manifest_path_is_rejected_not_ignored(tmp_path):
    cfg = _snapshot_config(model_manifest_path=str(tmp_path / "nope.json"))
    with pytest.raises(ValueError, match="nope.json"):
        overfit100.WanTI2VOverfit100Trainer._validate_pinned_snapshot(cfg)


def test_real_committed_manifest_pins_the_expected_revision():
    # Guards against the fixture drifting away from the committed manifest.
    repo = Path(overfit100.__file__).parents[3]
    manifest = repo / "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json"
    assert json.loads(manifest.read_text())["vae_fingerprint"]["revision"] == _PINNED


# =======================================================================================
# (C2) the published dataset bytes
# =======================================================================================


def test_valid_publication_verifies_and_reports(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    report = overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)
    assert report["records"] == 3
    assert report["shards"] == 1
    assert report["bytes_verified"] is True
    assert report["set_name"] == "train10"


def test_missing_marker_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], success=False)
    with pytest.raises(ValueError, match=f"has no {SUCCESS_MARKER} marker"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_unparseable_marker_fails_loudly_without_fallback(tmp_path):
    # Judgment 8: a marker that cannot be read/parsed must NOT silently fall back to
    # summary.json -- the old broad-except made an unreadable marker look like a count problem.
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    (data_dir / SUCCESS_MARKER).write_text("{not json\n")
    with pytest.raises(ValueError) as ei:
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)
    msg = str(ei.value)
    assert SUCCESS_MARKER in msg and str(data_dir) in msg
    assert "records" not in msg.split("\n")[0].lower() or "parse" in msg.lower() or "JSON" in msg


@pytest.mark.parametrize("key", ["summary_sha256", "build_id", "build_commit", "shards", "manifest_sha256"])
def test_structurally_invalid_marker_is_refused(tmp_path, key):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], drop_marker_keys=(key,))
    with pytest.raises(ValueError, match=key):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_absent_optional_records_field_falls_back_to_summary(tmp_path):
    # The ONLY sanctioned fallback (judgment 8): a VALID marker that simply lacks the
    # optional count field. Everything else still verifies.
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], marker_records=None)
    report = overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)
    assert report["records"] == 3
    assert report["records_source"] == "summary.json"


def test_summary_byte_mutation_breaks_the_marker_binding(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    original = (data_dir / "summary.json").read_text()
    (data_dir / "summary.json").write_text(original.replace('"built_utc"', '"BUILT_utc"'))
    with pytest.raises(ValueError) as ei:
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)
    assert "summary_sha256" in str(ei.value)


def test_missing_summary_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    (data_dir / "summary.json").unlink()
    with pytest.raises(ValueError, match="summary.json"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_unknown_set_name_in_summary_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], set_name="train100")
    with pytest.raises(ValueError, match="train10"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_extra_shard_on_disk_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], extra_shard=True)
    with pytest.raises(ValueError, match="shard"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_shard_listed_but_absent_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    next(data_dir.glob("*.tfrecord")).unlink()
    with pytest.raises(ValueError, match="shard"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_summary_listing_no_shards_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], drop_shard=True)
    with pytest.raises(ValueError, match="shard"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_shard_size_mismatch_is_refused(tmp_path):
    def bump_size(summary):
        summary["sets"]["train10"]["shards"][0]["size"] += 7
        return summary

    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], summary_mutator=bump_size)
    with pytest.raises(ValueError, match="size"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def _rewrite_shard_with_different_latents(shard: Path, episode_indices, seed=999):
    """A WELL-FORMED replacement shard: same records, same byte length, different latents.

    Fixed-length latent payloads and identical names/instructions/ints mean the serialized
    length is byte-for-byte the same, and re-serializing recomputes TFRecord's per-record
    CRC32C -- so the file stays perfectly readable. This is the mutation that metadata,
    record counts, and TFRecord's own checksums ALL miss.
    """
    rng = np.random.default_rng(seed)
    with tf.io.TFRecordWriter(str(shard)) as writer:
        for position, episode_index in enumerate(episode_indices):
            z_video = rng.standard_normal(Z_VIDEO_SHAPE).astype(np.float16)
            writer.write(
                serialize_window_record(
                    name=f"ep{25189 + episode_index:05d}_v0_s{position * 4:05d}",
                    episode_id=25189 + episode_index,
                    episode_index=episode_index,
                    window_start=position * 4,
                    z_i0=z_video[:, 0:1].copy(),
                    z_video=z_video,
                    instruction=f"instruction {episode_index}",
                )
            )


def test_wellformed_same_count_same_size_mutation_is_caught(tmp_path):
    # THE C2 regression, in its strongest form: the shard is REPLACED by a valid TFRecord with
    # the same record count, the same byte length, and correct per-record CRCs -- only different
    # latent VALUES. Size checks, count checks, and TFRecord's own checksums all pass; only the
    # published content hash catches it.
    indices = [0, 1, 2]
    data_dir = _publish_dataset(tmp_path / "train10", indices)
    entry = json.loads((data_dir / "summary.json").read_text())["sets"]["train10"]["shards"][0]
    shard = data_dir / entry["name"]
    _rewrite_shard_with_different_latents(shard, indices)

    assert shard.stat().st_size == entry["size"]  # SAME byte length
    assert sum(1 for _ in tf.data.TFRecordDataset([str(shard)])) == 3  # SAME count, still READABLE
    assert hashlib.sha256(shard.read_bytes()).hexdigest() != entry["sha256"]  # different content
    with pytest.raises(ValueError) as ei:
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)
    assert "sha256" in str(ei.value)


def test_raw_bit_flip_is_caught_at_preflight_rather_than_mid_training(tmp_path):
    # A raw bit flip IS eventually caught by TFRecord's per-record CRC32C -- but only when the
    # reader reaches that record, i.e. mid-run, after the 5B load and hours of TPU time. The
    # preflight catches it before any of that.
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    entry = json.loads((data_dir / "summary.json").read_text())["sets"]["train10"]["shards"][0]
    shard = data_dir / entry["name"]
    payload = bytearray(shard.read_bytes())
    payload[-64] ^= 0x01
    shard.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="sha256"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)
    # Documenting the fallback line of defence we are NOT relying on:
    with pytest.raises(tf.errors.DataLossError):
        list(tf.data.TFRecordDataset([str(shard)]))


def test_md5_mismatch_is_caught(tmp_path):
    def break_md5(summary):
        summary["sets"]["train10"]["shards"][0]["md5"] = _md5_b64(b"something else")
        return summary

    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], summary_mutator=break_md5)
    with pytest.raises(ValueError, match="md5"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)


def test_generation_is_deliberately_not_verified(tmp_path):
    # Cycle B stats the STAGING object, then promote() COPIES it to the canonical prefix --
    # minting a new generation. Verifying the recorded generation would always fail; content
    # is bound by sha256 + md5 + size instead.
    def stale_generation(summary):
        summary["sets"]["train10"]["shards"][0]["generation"] = "9999999999999999"
        return summary

    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], summary_mutator=stale_generation)
    assert overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3)["records"] == 3


def test_expected_windows_mismatch_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    with pytest.raises(ValueError) as ei:
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=167)
    msg = str(ei.value)
    assert "167" in msg and "3" in msg


def test_marker_count_disagreeing_with_shard_records_is_refused(tmp_path):
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2], marker_records=99)
    with pytest.raises(ValueError, match="99|records"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=99)


def test_bytes_verification_can_be_disabled_but_metadata_still_binds(tmp_path):
    # dataset_verify_bytes=False is an escape hatch for a huge set; the size/name/count and
    # summary-hash bindings remain. The report says so, so a log reader can tell.
    data_dir = _publish_dataset(tmp_path / "train10", [0, 1, 2])
    shard = next(data_dir.glob("*.tfrecord"))
    payload = bytearray(shard.read_bytes())
    payload[-64] ^= 0x01
    shard.write_bytes(bytes(payload))
    report = overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3, verify_bytes=False)
    assert report["bytes_verified"] is False
    with pytest.raises(ValueError, match="sha256"):
        overfit100.verify_dataset_integrity(str(data_dir), expected_windows=3, verify_bytes=True)


# =======================================================================================
# (C3) the context mapping of a distinct eval dir
# =======================================================================================


def test_identical_mapping_is_compatible(tmp_path):
    train = _publish_dataset(tmp_path / "train100", [0, 1, 2])
    ev = _publish_dataset(tmp_path / "evalset", [0, 2])
    overfit100.assert_context_map_compatible(str(train), str(ev))


def test_same_sized_but_different_instruction_is_refused(tmp_path):
    # THE C3 regression: same index count, same indices -- different TEXT. The table is built
    # from the training set, so this would evaluate against the wrong instruction silently.
    train = _publish_dataset(tmp_path / "train100", [0, 1, 2])
    ev = _publish_dataset(tmp_path / "evalset", [0, 1, 2], instruction_of=lambda i: f"a DIFFERENT task {i}")
    with pytest.raises(ValueError) as ei:
        overfit100.assert_context_map_compatible(str(train), str(ev))
    msg = str(ei.value)
    assert "used_text" in msg and "episode_index" in msg


def test_same_sized_but_different_episode_id_is_refused(tmp_path):
    train = _publish_dataset(tmp_path / "train100", [0, 1, 2])
    ev = _publish_dataset(tmp_path / "evalset", [0, 1, 2])
    payload = json.loads((ev / "episodes.json").read_text())
    payload["episodes"][1]["episode_id"] += 1000
    (ev / "episodes.json").write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(ValueError, match="episode_id"):
        overfit100.assert_context_map_compatible(str(train), str(ev))


def test_eval_index_outside_the_training_mapping_is_refused(tmp_path):
    train = _publish_dataset(tmp_path / "train100", [0, 1, 2])
    ev = _publish_dataset(tmp_path / "evalset", [7])
    with pytest.raises(ValueError, match="7"):
        overfit100.assert_context_map_compatible(str(train), str(ev))


# =======================================================================================
# ORDER: the preflight runs before any model work
# =======================================================================================


def _full_config(tmp_path, **kw):
    train = _publish_dataset(Path(tmp_path) / "train10", [0, 1, 2])
    cfg = {
        # probe-config gates (inherited from full-FT)
        "side_adapter_guide_scale": 1.0,
        "side_adapter_noise_mode": "fresh",
        # overfit100 gates
        "num_text_slots": 3,
        "text_encode_batch": 8,
        "expected_windows": 3,
        "checkpoint_steps": [250],
        # C1
        "pretrained_model_name_or_path": _SNAPSHOT,
        "wan_transformer_pretrained_model_name_or_path": _SNAPSHOT,
        "expected_model_revision": _PINNED,
        "model_manifest_path": "",
        # C2/C3
        "train_data_dir": str(train),
        "eval_data_dir": str(train),
        "dataset_verify_bytes": True,
    }
    cfg.update(kw)
    return SimpleNamespace(**cfg), train


def test_start_training_preflights_the_dataset_before_loading_the_pipeline(tmp_path, monkeypatch):
    # THE C2 ordering regression: an invalid dataset must fail BEFORE the 5B pipeline,
    # optimizer, and state are built. The loader is booby-trapped, so if it runs we see
    # its AssertionError instead of the dataset ValueError.
    config, train = _full_config(tmp_path)
    (train / SUCCESS_MARKER).unlink()

    def _boom(self):
        raise AssertionError("pipeline loader ran despite an invalid dataset")

    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_load_wan_pipeline", _boom)
    with pytest.raises(ValueError, match=f"has no {SUCCESS_MARKER} marker"):
        overfit100.WanTI2VOverfit100Trainer(config).start_training()


def test_start_training_rejects_an_unpinned_snapshot_before_loading_the_pipeline(tmp_path, monkeypatch):
    config, _ = _full_config(
        tmp_path,
        pretrained_model_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        wan_transformer_pretrained_model_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    )

    def _boom(self):
        raise AssertionError("pipeline loader ran despite an unpinned snapshot")

    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_load_wan_pipeline", _boom)
    with pytest.raises(ValueError, match=_PINNED):
        overfit100.WanTI2VOverfit100Trainer(config).start_training()


def test_start_training_rejects_an_incompatible_eval_mapping_before_loading_the_pipeline(tmp_path, monkeypatch):
    ev = _publish_dataset(Path(tmp_path) / "evalset", [0, 1, 2], instruction_of=lambda i: f"different {i}")
    config, _ = _full_config(tmp_path, eval_data_dir=str(ev))

    def _boom(self):
        raise AssertionError("pipeline loader ran despite an incompatible eval mapping")

    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_load_wan_pipeline", _boom)
    with pytest.raises(ValueError, match="used_text"):
        overfit100.WanTI2VOverfit100Trainer(config).start_training()


def test_preflight_passes_on_a_valid_publication_and_logs_the_bindings(tmp_path, monkeypatch):
    # Positive control: with a valid pinned snapshot + valid dataset the preflight returns and
    # only then reaches the pipeline load (proven by the sentinel the stub loader raises).
    config, _ = _full_config(tmp_path)
    logged: list[str] = []
    monkeypatch.setattr(overfit100.max_logging, "log", lambda msg, *a, **k: logged.append(str(msg)))

    class _Sentinel(RuntimeError):
        pass

    def _reached(self):
        raise _Sentinel("preflight complete")

    monkeypatch.setattr(overfit100.WanTI2VOverfit100Trainer, "_load_wan_pipeline", _reached)
    with pytest.raises(_Sentinel):
        overfit100.WanTI2VOverfit100Trainer(config).start_training()
    assert any(_PINNED in line for line in logged), logged
    assert any("_SUCCESS verified" in line or "integrity" in line for line in logged), logged
