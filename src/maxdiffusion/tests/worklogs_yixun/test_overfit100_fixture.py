"""CPU-only tests for the exp_02 V1-fixture extractor (cycle A, deliverable A1).

Covers `maxdiffusion.data_preprocessing.extract_v1_fixture` -- the tool that pulls the
three named exp_01 cache windows (`ep0_v0_s00000/s00004/s00008`) out of
`droid_wan_side_adapter/train/train-00000-of-00704.tfrecord`, checks the cache contract
`z_i0 == z_video[:, :1]` bitwise, and publishes a fingerprinted `.npz` fixture to GCS
(plan v4 H1). The build job's preflight verifies that fingerprint before any encoding,
so the parse/extract/verify path is the gate that must be trustworthy.

No network and no accelerator: the TFRecord is synthesized locally with tiny shapes
(dtype float16 and the `z_i0 == z_video[:, :1]` property are preserved -- those are the
properties under test), and the gsutil layer is exercised only through its pure parser.
"""

from __future__ import annotations

import base64
import hashlib
import json

import numpy as np
import pytest
import tensorflow as tf

from maxdiffusion.data_preprocessing.extract_v1_fixture import (
    STATUS_ABSENT,
    STATUS_ERROR,
    STATUS_FOUND,
    TARGET_NAMES,
    Resolved,
    build_fingerprint,
    classify_stat_batch,
    extract_windows,
    iter_tfrecord,
    load_fixture_npz,
    md5_b64,
    parse_absent_uris,
    parse_gsutil_stat,
    parse_record,
    pinned_uri,
    save_fixture_npz,
    validate_fixture_structure,
    verify_fixture,
    verify_payload_binding,
)

# Tiny stand-ins for the real (48, 1, 12, 20) / (48, 9, 12, 20) geometry. Channels/frames
# kept > 1 so the `z_video[:, :1]` slice is a genuine slice, not the whole array.
_Z_I0_SHAPE = (2, 1, 3, 4)
_Z_VIDEO_SHAPE = (2, 5, 3, 4)


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _make_window(seed: int, *, break_contract: bool = False):
    """A (z_i0, z_video) float16 pair honouring (or deliberately breaking) the contract."""
    rng = np.random.default_rng(seed)
    z_video = rng.standard_normal(_Z_VIDEO_SHAPE).astype(np.float16)
    z_i0 = z_video[:, :1].copy()
    if break_contract:
        z_i0 = z_i0 + np.float16(1.0)
    return z_i0, z_video


def _serialize_record(name: str, z_i0: np.ndarray, z_video: np.ndarray, ordinal: int) -> bytes:
    """Same schema as the exp_01 cache: name/z_i0/z_video/actions/meta_json bytes + ordinal."""
    feature = {
        "name": _bytes_feature(name.encode()),
        "z_i0": _bytes_feature(z_i0.tobytes()),
        "z_video": _bytes_feature(z_video.tobytes()),
        "actions": _bytes_feature(np.zeros((4, 7), dtype=np.float32).tobytes()),
        "meta_json": _bytes_feature(json.dumps({"episode_id": 0}).encode()),
        "ordinal": _int64_feature(ordinal),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def _write_tfrecord(path, records: list[bytes]) -> str:
    with tf.io.TFRecordWriter(str(path)) as writer:
        for raw in records:
            writer.write(raw)
    return str(path)


def _default_records(break_contract_on: str | None = None) -> list[bytes]:
    records = []
    for ordinal, name in enumerate(TARGET_NAMES):
        z_i0, z_video = _make_window(ordinal, break_contract=(name == break_contract_on))
        records.append(_serialize_record(name, z_i0, z_video, ordinal))
    return records


_EXTRACT_KWARGS = {"z_i0_shape": _Z_I0_SHAPE, "z_video_shape": _Z_VIDEO_SHAPE}


# ----------------------------------------------------------------------------------
# 1. parse_record -- schema + shape/dtype + the z_i0 == z_video[:, :1] cache contract.
# ----------------------------------------------------------------------------------


def test_parse_record_returns_name_and_float16_arrays():
    z_i0, z_video = _make_window(7)
    name, got_i0, got_video = parse_record(_serialize_record("ep0_v0_s00000", z_i0, z_video, 0), **_EXTRACT_KWARGS)
    assert name == "ep0_v0_s00000"
    assert got_i0.shape == _Z_I0_SHAPE and got_video.shape == _Z_VIDEO_SHAPE
    assert got_i0.dtype == np.float16 and got_video.dtype == np.float16
    np.testing.assert_array_equal(got_i0, z_i0)
    np.testing.assert_array_equal(got_video, z_video)


def test_parse_record_rejects_wrong_byte_length():
    z_i0, z_video = _make_window(8)
    raw = _serialize_record("ep0_v0_s00000", z_i0, z_video, 0)
    with pytest.raises(ValueError, match="z_video"):
        # Declaring one extra latent frame makes the stored payload too short.
        parse_record(raw, z_i0_shape=_Z_I0_SHAPE, z_video_shape=(2, 6, 3, 4))


def test_parse_record_rejects_broken_z_i0_contract():
    z_i0, z_video = _make_window(9, break_contract=True)
    raw = _serialize_record("ep0_v0_s00000", z_i0, z_video, 0)
    with pytest.raises(ValueError, match="z_i0"):
        parse_record(raw, **_EXTRACT_KWARGS)


# ----------------------------------------------------------------------------------
# 2. extract_windows -- name-addressed extraction, bounded scan, loud failure.
# ----------------------------------------------------------------------------------


def test_extract_windows_finds_all_named_records(tmp_path):
    path = _write_tfrecord(tmp_path / "shard.tfrecord", _default_records())
    windows = extract_windows(iter_tfrecord(path), **_EXTRACT_KWARGS)
    assert list(windows) == list(TARGET_NAMES)
    for name in TARGET_NAMES:
        assert windows[name]["z_i0"].shape == _Z_I0_SHAPE
        assert windows[name]["z_video"].shape == _Z_VIDEO_SHAPE
        assert windows[name]["z_i0"].dtype == np.float16
        # The contract the fixture exists to preserve.
        assert windows[name]["z_i0"].tobytes() == windows[name]["z_video"][:, :1].copy().tobytes()


def test_extract_windows_fails_loudly_when_a_name_is_missing(tmp_path):
    records = _default_records()[:2]
    path = _write_tfrecord(tmp_path / "shard.tfrecord", records)
    with pytest.raises(ValueError, match=TARGET_NAMES[2]):
        extract_windows(iter_tfrecord(path), **_EXTRACT_KWARGS)


def test_extract_windows_stops_after_max_records(tmp_path):
    path = _write_tfrecord(tmp_path / "shard.tfrecord", _default_records())
    with pytest.raises(ValueError, match=TARGET_NAMES[2]):
        extract_windows(iter_tfrecord(path), max_records=2, **_EXTRACT_KWARGS)


def test_extract_windows_propagates_broken_contract(tmp_path):
    records = _default_records(break_contract_on=TARGET_NAMES[1])
    path = _write_tfrecord(tmp_path / "shard.tfrecord", records)
    with pytest.raises(ValueError, match="z_i0"):
        extract_windows(iter_tfrecord(path), **_EXTRACT_KWARGS)


# ----------------------------------------------------------------------------------
# 3. npz round-trip -- names / shapes / dtypes survive the save.
# ----------------------------------------------------------------------------------


def test_npz_roundtrip_preserves_names_shapes_dtypes(tmp_path):
    path = _write_tfrecord(tmp_path / "shard.tfrecord", _default_records())
    windows = extract_windows(iter_tfrecord(path), **_EXTRACT_KWARGS)
    npz_path = tmp_path / "v1_cache_windows.npz"
    save_fixture_npz(npz_path, windows)

    names, loaded = load_fixture_npz(npz_path)
    assert names == list(TARGET_NAMES)
    for name in TARGET_NAMES:
        for key in ("z_i0", "z_video"):
            np.testing.assert_array_equal(loaded[name][key], windows[name][key])
            assert loaded[name][key].dtype == np.float16


# ----------------------------------------------------------------------------------
# 4. Fingerprint + preflight verification (plan v4 H1) -- md5 match passes, drift fails.
# ----------------------------------------------------------------------------------


def _fixture_on_disk(tmp_path):
    path = _write_tfrecord(tmp_path / "shard.tfrecord", _default_records())
    windows = extract_windows(iter_tfrecord(path), **_EXTRACT_KWARGS)
    npz_path = tmp_path / "v1_cache_windows.npz"
    save_fixture_npz(npz_path, windows)
    stat = {"generation": 1785251634871545, "md5": md5_b64(npz_path.read_bytes()), "size": npz_path.stat().st_size}
    return npz_path, build_fingerprint(npz_path, "gs://bucket/fixtures/v1_cache_windows.npz", windows, stat)


def test_md5_b64_matches_gcs_convention():
    payload = b"exp_02 fixture bytes"
    assert md5_b64(payload) == base64.b64encode(hashlib.md5(payload).digest()).decode()


def test_build_fingerprint_records_uri_names_shapes_dtypes(tmp_path):
    npz_path, fp = _fixture_on_disk(tmp_path)
    assert fp["uri"] == "gs://bucket/fixtures/v1_cache_windows.npz"
    assert fp["names"] == list(TARGET_NAMES)
    assert fp["generation"] == 1785251634871545
    assert fp["size_bytes"] == npz_path.stat().st_size
    assert fp["md5"] == md5_b64(npz_path.read_bytes())
    assert fp["shapes"] == {"z_i0": list(_Z_I0_SHAPE), "z_video": list(_Z_VIDEO_SHAPE)}
    assert fp["dtypes"] == {"z_i0": "float16", "z_video": "float16"}


def test_verify_fixture_passes_on_untouched_file(tmp_path):
    npz_path, fp = _fixture_on_disk(tmp_path)
    assert verify_fixture(npz_path, fp) == []


def test_verify_fixture_fails_on_md5_drift(tmp_path):
    npz_path, fp = _fixture_on_disk(tmp_path)
    drifted = dict(fp, md5="AAAAAAAAAAAAAAAAAAAAAA==")
    errors = verify_fixture(npz_path, drifted)
    assert errors and any("md5" in e for e in errors)


def test_verify_fixture_fails_on_size_drift(tmp_path):
    npz_path, fp = _fixture_on_disk(tmp_path)
    errors = verify_fixture(npz_path, dict(fp, size_bytes=fp["size_bytes"] + 1))
    assert errors and any("size" in e for e in errors)


def test_verify_fixture_fails_on_missing_name(tmp_path):
    npz_path, fp = _fixture_on_disk(tmp_path)
    errors = verify_fixture(npz_path, dict(fp, names=list(TARGET_NAMES) + ["ep0_v0_s00012"]))
    assert errors and any("ep0_v0_s00012" in e for e in errors)


# ----------------------------------------------------------------------------------
# 4b. A4 -- the preflight must FAIL CLOSED on fixture structure, not just on hashes.
# ----------------------------------------------------------------------------------


def _loaded_fixture(tmp_path):
    npz_path, fp = _fixture_on_disk(tmp_path)
    names, windows = load_fixture_npz(npz_path)
    return names, windows, fp


def test_validate_fixture_structure_accepts_the_real_shape(tmp_path):
    names, windows, fp = _loaded_fixture(tmp_path)
    assert validate_fixture_structure(names, windows, fp) == []


def test_validate_fixture_structure_requires_the_exact_ordered_name_set(tmp_path):
    names, windows, fp = _loaded_fixture(tmp_path)
    reordered = [names[1], names[0], names[2]]
    errors = validate_fixture_structure(reordered, windows, dict(fp, names=reordered))
    assert errors and any("order" in e or "names" in e for e in errors)


def test_validate_fixture_structure_rejects_a_short_name_set(tmp_path):
    names, windows, fp = _loaded_fixture(tmp_path)
    short = names[:2]
    errors = validate_fixture_structure(short, {n: windows[n] for n in short}, dict(fp, names=short))
    assert errors and any(TARGET_NAMES[2] in e for e in errors)


def test_validate_fixture_structure_rejects_a_dtype_mutation(tmp_path):
    names, windows, fp = _loaded_fixture(tmp_path)
    windows[names[0]]["z_video"] = windows[names[0]]["z_video"].astype(np.float32)
    errors = validate_fixture_structure(names, windows, fp)
    assert errors and any("dtype" in e for e in errors)


def test_validate_fixture_structure_rejects_a_shape_mutation(tmp_path):
    names, windows, fp = _loaded_fixture(tmp_path)
    windows[names[1]]["z_i0"] = windows[names[1]]["z_i0"][:, :, :2]
    errors = validate_fixture_structure(names, windows, fp)
    assert errors and any("shape" in e for e in errors)


def test_validate_fixture_structure_rejects_a_broken_first_frame_contract(tmp_path):
    # The whole point of the fixture: z_i0 must stay the bitwise first latent frame.
    names, windows, fp = _loaded_fixture(tmp_path)
    broken = windows[names[2]]["z_i0"].copy()
    broken[0, 0, 0, 0] = np.float16(broken[0, 0, 0, 0] + np.float16(1.0))
    windows[names[2]]["z_i0"] = broken
    errors = validate_fixture_structure(names, windows, fp)
    assert errors and any("z_i0" in e for e in errors)


def test_validate_fixture_structure_rejects_a_missing_array(tmp_path):
    names, windows, fp = _loaded_fixture(tmp_path)
    del windows[names[0]]["z_video"]
    errors = validate_fixture_structure(names, windows, fp)
    assert errors and any("z_video" in e for e in errors)


def test_verify_fixture_runs_the_structural_validator(tmp_path):
    # End-to-end: a structurally broken .npz must be rejected even though md5/size still match.
    npz_path, fp = _fixture_on_disk(tmp_path)
    names, windows = load_fixture_npz(npz_path)
    windows[names[0]]["z_i0"] = windows[names[0]]["z_i0"] + np.float16(1.0)
    save_fixture_npz(npz_path, windows)
    data = npz_path.read_bytes()
    rehashed = dict(fp, md5=md5_b64(data), size_bytes=len(data))
    errors = verify_fixture(npz_path, rehashed)
    assert errors and any("z_i0" in e for e in errors)


# ----------------------------------------------------------------------------------
# 4c. A2 -- binding downloaded bytes to the recorded fingerprint.
# ----------------------------------------------------------------------------------


def test_verify_payload_binding_accepts_matching_bytes():
    data = b"the exact bytes that were statted"
    fingerprint = {"uri": "gs://b/o", "md5": md5_b64(data), "size": len(data), "generation": 7}
    assert verify_payload_binding("gs://b/o", data, fingerprint) == []


def test_verify_payload_binding_rejects_changed_bytes():
    data = b"the exact bytes that were statted"
    fingerprint = {"uri": "gs://b/o", "md5": md5_b64(data), "size": len(data), "generation": 7}
    errors = verify_payload_binding("gs://b/o", b"different bytes entirely!!!!!!!!!", fingerprint)
    assert errors and any("md5" in e for e in errors)


def test_verify_payload_binding_rejects_changed_size():
    data = b"abc"
    fingerprint = {"uri": "gs://b/o", "md5": md5_b64(data), "size": 99, "generation": 7}
    errors = verify_payload_binding("gs://b/o", data, fingerprint)
    assert errors and any("size" in e for e in errors)


def test_pinned_uri_appends_the_recorded_generation():
    assert pinned_uri("gs://b/o.mp4", {"generation": 1785251634871545}) == "gs://b/o.mp4#1785251634871545"


def test_pinned_uri_refuses_a_fingerprint_without_a_generation():
    with pytest.raises(ValueError):
        pinned_uri("gs://b/o.mp4", {"generation": None})


# ----------------------------------------------------------------------------------
# 4d. A3 -- per-object stat classification: found / absent / error, never conflated.
# ----------------------------------------------------------------------------------

_ABSENT_STDERR = "No URLs matched: gs://v6_east1d/datasets/droid_ctrl_world_aligned/annotation/train/999999.json\n"


def test_parse_absent_uris_extracts_exact_uris():
    assert parse_absent_uris(_ABSENT_STDERR) == {
        "gs://v6_east1d/datasets/droid_ctrl_world_aligned/annotation/train/999999.json"
    }


def test_parse_absent_uris_ignores_other_noise():
    assert parse_absent_uris("ServiceException: 503 Backend Error\n") == set()


def test_classify_stat_batch_marks_parsed_objects_found():
    uri = "gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/0/0.mp4"
    resolved = classify_stat_batch([uri], _REAL_STAT, "", 0)
    assert resolved[uri].status == STATUS_FOUND
    assert resolved[uri].fingerprint["generation"] == 1785251634871545


def test_classify_stat_batch_marks_named_missing_objects_absent():
    missing = "gs://v6_east1d/datasets/droid_ctrl_world_aligned/annotation/train/999999.json"
    resolved = classify_stat_batch([missing], "", _ABSENT_STDERR, 1)
    assert resolved[missing].status == STATUS_ABSENT


def test_classify_stat_batch_leaves_unnamed_failures_as_errors():
    # A transient 503 for one member of a batch must NOT be read as "object does not exist".
    uri = "gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/0/0.mp4"
    other = "gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/1/0.mp4"
    resolved = classify_stat_batch([uri, other], _REAL_STAT, "ServiceException: 503 Backend Error\n", 1)
    assert resolved[uri].status == STATUS_FOUND
    assert resolved[other].status == STATUS_ERROR
    assert "503" in resolved[other].error


def test_classify_stat_batch_flags_a_404_substring_inside_a_transient_error_as_error():
    # The old substring heuristic would have called this "absent"; it is an error.
    uri = "gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/7/0.mp4"
    stderr = "ServiceException: 503 upstream connect error (request id 404abc)\n"
    resolved = classify_stat_batch([uri], "", stderr, 1)
    assert resolved[uri].status == STATUS_ERROR


def test_resolved_helpers_are_readable():
    assert Resolved.found({"uri": "gs://b/o"}).status == STATUS_FOUND
    assert Resolved.absent().status == STATUS_ABSENT
    assert Resolved.failed("boom").status == STATUS_ERROR
    assert Resolved.failed("boom").error == "boom"
    assert Resolved.found({"uri": "gs://b/o"}).ok is True
    assert Resolved.absent().ok is False


def test_non_error_outcomes_carry_no_error_text():
    # Regression: a `Resolved.error` CLASSMETHOD would shadow the `error` field default, so every
    # absent/found outcome would report a truthy bound method as its error.
    assert Resolved.absent().error is None
    assert Resolved.found({"uri": "gs://b/o"}).error is None


# ----------------------------------------------------------------------------------
# 5. gsutil stat parser -- captured real output (probe 2026-07-28).
# ----------------------------------------------------------------------------------

_REAL_STAT = """gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/0/0.mp4:
    Creation time:          Tue, 28 Jul 2026 15:13:54 GMT
    Update time:            Tue, 28 Jul 2026 15:13:54 GMT
    Storage class:          STANDARD
    Content-Length:         320331
    Content-Type:           video/mp4
    Hash (crc32c):          ZcQ/Tw==
    Hash (md5):             //EmSFBtOwwCStqdiQ2vDg==
    ETag:                   CPn5n8zU9ZUDEAE=
    Generation:             1785251634871545
    Metageneration:         1
"""


def test_parse_gsutil_stat_single_object():
    parsed = parse_gsutil_stat(_REAL_STAT)
    uri = "gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/0/0.mp4"
    assert set(parsed) == {uri}
    assert parsed[uri] == {
        "uri": uri,
        "generation": 1785251634871545,
        "md5": "//EmSFBtOwwCStqdiQ2vDg==",
        "size": 320331,
    }
