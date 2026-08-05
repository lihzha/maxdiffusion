"""exp_04 R8 — ``null_adapter_shards``: publishing the P2 cache so a resumed job cannot lie.

A cache job runs for hours across preemptible hosts, so every question this module answers is asked
after a crash: *which examples are done?* The answer has to be derived from the artifacts themselves,
never from a log line or a directory listing, because a half-written shard looks exactly like a
finished one until something checks.

What carries the round, after the R8 strengthening:

- **The marker is the publish signal, and it goes last.** Records are staged, published, and only
  then is the marker moved into place. Pinned by a publish-order hook and by a killed write whose
  names all come back in the resume plan.
- **The marker cannot name its own files.** The name-to-file mapping is the canonical bijection and
  is recomputed on read; the reviewer's probe -- a marker pointing at a *different shard's* staging
  leftover, with a genuine record and a correct digest waiting there -- is a test here now.
- **Validation is provenance-closed.** Fingerprint, arm and convention are required arguments, every
  decoded record is checked against them and against its own header, and a shard mixing arms or
  dtypes never gets written in the first place.
- **Reading is total.** Hostile markers -- duplicate keys, ``schema_version: true``, negative counts,
  a string where a quarantine map belongs -- produce invalid reports, never exceptions.
- **Shards are immutable.** A completed shard is never rewritten, publication never overwrites, and
  discarding an incomplete attempt is an explicit call rather than a side effect of retrying.

The reader tests build hostile shards by hand rather than through ``write_shard``: a validator's job
is to survive artifacts its own writer would never produce.

Paths here are local ``tmp_path``s, but the IO goes through ``tf.io.gfile`` -- the same call the
production ``gs://`` prefixes take -- so the code under test is the code that will run.
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

from maxdiffusion.null_adapter_gates import NoiseConvention, gate_g1
from maxdiffusion.null_adapter_records import (
    PRODUCTION_GEOMETRY,
    ProvenanceHeader,
    header_to_json,
    make_record,
    record_from_bytes,
    record_to_bytes,
)
from maxdiffusion.null_adapter_shards import (
    HEADER_NAME,
    MARKER_NAME,
    SHARD_SCHEMA_VERSION,
    ShardMarker,
    canonical_files,
    discard_incomplete_shard,
    header_fingerprint,
    marker_from_json,
    next_shard_index,
    resume_plan,
    validate_shard,
    write_shard,
)
from maxdiffusion.null_adapter_verify import canonical_sigmas


_REVISION = "Wan2.2-TI2V-5B@f370228"
_NAMES = ("droid_ep_000001/w0", "droid_ep_000002/w0", "droid_ep_000003/w0")
_SHAPES = PRODUCTION_GEOMETRY.shapes()
_ARM, _CONVENTION = "A1", "keyed"


def _header(**overrides):
    fields = {
        "manifest_hash": "a" * 64,
        "code_sha": "b" * 40,
        "model_revision": _REVISION,
        "sigma_vector": canonical_sigmas(),
        "guide_scale": 5.0,
        "base_context_fingerprint": "c" * 64,
        "optimization_config": {"inner_iters": 10, "lr": 0.01},
        "dtype_policy": "fp16",
        "l_null": 16,
    }
    return ProvenanceHeader(**{**fields, **overrides})


def _fingerprint(header=None):
    import hashlib

    return hashlib.sha256(header_to_json(header or _header()).encode("utf-8")).hexdigest()


def _record(name, ordinal=0, latent_dtype="fp16", arm=_ARM, convention=_CONVENTION):
    arrays = {field: np.full(shape, np.float32(0.01 * (ordinal + 1)), np.float32) for field, shape in _SHAPES.items()}
    return make_record(
        name=name,
        ordinal=ordinal,
        split="dev",
        episode=name.split("/")[0],
        latent_dtype=latent_dtype,
        noise_convention=convention,
        arm=arm,
        final_future_mse=0.25 + ordinal,
        **arrays,
    )


def _records(names=_NAMES, **kwargs):
    return [_record(name, index, **kwargs) for index, name in enumerate(names)]


def _write(tmp_path, names=_NAMES, shard="shard_00000", header=None, **kwargs):
    header = header or _header()
    return write_shard(_records(names), header, str(tmp_path / "cache" / shard), str(tmp_path / "staging"), **kwargs)


def _validate(tmp_path, shard="shard_00000", **kwargs):
    expectations = {
        "expected_header_fingerprint": _fingerprint(),
        "expected_arm": _ARM,
        "expected_noise_convention": _CONVENTION,
        **kwargs,
    }
    return validate_shard(str(tmp_path / "cache" / shard), **expectations)


def _resume(tmp_path, manifest, shards, **kwargs):
    expectations = {
        "expected_header_fingerprint": _fingerprint(),
        "expected_arm": _ARM,
        "expected_noise_convention": _CONVENTION,
        **kwargs,
    }
    return resume_plan(manifest, [str(tmp_path / "cache" / shard) for shard in shards], **expectations)


def _marker_path(tmp_path, shard="shard_00000"):
    return str(tmp_path / "cache" / shard / MARKER_NAME)


def _read_marker_payload(tmp_path, shard="shard_00000"):
    with open(_marker_path(tmp_path, shard), encoding="utf-8") as handle:
        return json.loads(handle.read())


def _rewrite_marker(tmp_path, payload, shard="shard_00000"):
    with open(_marker_path(tmp_path, shard), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) if isinstance(payload, dict) else payload)


def _handcraft(tmp_path, records, header=None, *, files=None, marker_overrides=None, shard="hand"):
    """Build a shard directly, bypassing the writer: a reader must survive hostile artifacts."""
    import hashlib

    header = header or _header()
    root = tmp_path / "cache" / shard
    root.mkdir(parents=True, exist_ok=True)
    (root / HEADER_NAME).write_text(header_to_json(header), encoding="utf-8")
    names = sorted(record.name for record in records)
    files = files or canonical_files(names)
    digests = {}
    for record in records:
        blob = record_to_bytes(record)
        (root / files[record.name]).write_bytes(blob)
        digests[record.name] = hashlib.sha256(blob).hexdigest()
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "count": len(names),
        "names": names,
        "files": files,
        "sha256": digests,
        "header_fingerprint": _fingerprint(header),
        "quarantined": {},
        **(marker_overrides or {}),
    }
    (root / MARKER_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return str(root)


def test_a_written_shard_validates_and_reads_back_bitwise(tmp_path):
    originals = _records()

    marker = _write(tmp_path)
    report = _validate(tmp_path)

    assert report.valid, report.reasons
    assert report.names == tuple(sorted(_NAMES))
    for record, name in zip(originals, _NAMES):
        with open(str(tmp_path / "cache" / "shard_00000" / marker.files[name]), "rb") as handle:
            restored = record_from_bytes(handle.read())
        assert restored.name == record.name and restored.ordinal == record.ordinal
        for field in ("nulls", "z_start", "expected_final_latent", "z_video", "z_i0", "actions"):
            np.testing.assert_array_equal(getattr(restored, field), getattr(record, field))


def test_the_marker_content_is_exact(tmp_path):
    import hashlib

    marker = _write(tmp_path)

    payload = _read_marker_payload(tmp_path)
    assert marker_from_json(json.dumps(payload)) == marker
    assert marker.schema_version == SHARD_SCHEMA_VERSION
    assert marker.count == len(_NAMES) == len(marker.names)
    assert marker.names == tuple(sorted(_NAMES)) and marker.quarantined == {}
    assert marker.files == canonical_files(_NAMES)
    assert marker.header_fingerprint == _fingerprint()
    for name in _NAMES:
        blob = open(str(tmp_path / "cache" / "shard_00000" / marker.files[name]), "rb").read()
        assert marker.sha256[name] == hashlib.sha256(blob).hexdigest()


def test_the_header_is_published_beside_the_records(tmp_path):
    _write(tmp_path)

    with open(str(tmp_path / "cache" / "shard_00000" / HEADER_NAME), encoding="utf-8") as handle:
        assert json.loads(handle.read())["model_revision"] == _REVISION


def _publish_hook(monkeypatch, tmp_path):
    from maxdiffusion import null_adapter_shards as shards

    seen = []
    original = shards._publish

    def hooked(source, destination):
        marker_present = os.path.exists(_marker_path(tmp_path))
        original(source, destination)
        seen.append((os.path.basename(destination), marker_present))

    monkeypatch.setattr(shards, "_publish", hooked)
    return seen


def test_the_marker_is_the_last_thing_published(tmp_path, monkeypatch):
    seen = _publish_hook(monkeypatch, tmp_path)

    _write(tmp_path)

    published = [name for name, _ in seen]
    assert published[-1] == MARKER_NAME
    assert MARKER_NAME not in published[:-1]
    assert published[0] == HEADER_NAME and len(published) == len(_NAMES) + 2
    assert not any(marker_present for _, marker_present in seen)


def test_records_are_streamed_one_at_a_time_rather_than_all_materialized(tmp_path, monkeypatch):
    """Peak memory is one record, so a shard's worth of blobs never coexists (R8 review, finding 7)."""
    from maxdiffusion import null_adapter_shards as shards

    order = []
    serialize, write = shards.record_to_bytes, shards._write_bytes
    monkeypatch.setattr(shards, "record_to_bytes", lambda record: (order.append("serialize"), serialize(record))[1])
    monkeypatch.setattr(shards, "_write_bytes", lambda path, payload: (order.append("write"), write(path, payload))[1])

    _write(tmp_path)

    assert order[: 2 * len(_NAMES)] == ["serialize", "write"] * len(_NAMES)


def test_a_shard_bigger_than_the_declared_cap_is_refused(tmp_path, monkeypatch):
    from maxdiffusion import null_adapter_shards as shards

    monkeypatch.setattr(shards, "MAX_SHARD_BYTES", 1024)

    with pytest.raises(ValueError, match="MAX_SHARD_BYTES"):
        _write(tmp_path)
    assert not os.path.exists(_marker_path(tmp_path))
    assert not any((tmp_path / "staging").iterdir()) if (tmp_path / "staging").exists() else True


def test_a_write_killed_midway_publishes_no_marker(tmp_path, monkeypatch):
    from maxdiffusion import null_adapter_shards as shards

    original, calls = shards._publish, []

    def failing(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("host preempted")
        original(source, destination)

    monkeypatch.setattr(shards, "_publish", failing)

    with pytest.raises(OSError, match="host preempted"):
        _write(tmp_path)

    assert not os.path.exists(_marker_path(tmp_path))
    report = _validate(tmp_path)
    assert not report.valid and "marker" in " ".join(report.reasons)
    assert _resume(tmp_path, _NAMES, ["shard_00000"]).todo == tuple(_NAMES)


def test_staging_is_unique_per_attempt_and_cleaned_up(tmp_path, monkeypatch):
    """Two writers must not share a staging directory, or each deletes the other's work."""
    from maxdiffusion import null_adapter_shards as shards

    seen = []
    original = shards._write_bytes
    monkeypatch.setattr(shards, "_write_bytes", lambda path, payload: (seen.append(path), original(path, payload))[1])

    _write(tmp_path, shard="shard_00000")
    first = {p for p in seen if ".staging" in p}
    seen.clear()
    _write(tmp_path, shard="shard_00001")
    second = {p for p in seen if ".staging" in p}

    assert first and second
    assert {os.path.dirname(p) for p in first}.isdisjoint({os.path.dirname(p) for p in second})
    staging = tmp_path / "staging"
    assert not staging.exists() or not any(staging.iterdir())


def test_a_completed_shard_is_never_rewritten(tmp_path):
    _write(tmp_path)

    with pytest.raises(FileExistsError, match="already published"):
        _write(tmp_path)
    assert _validate(tmp_path).valid  # ... and the original is untouched


def test_an_incomplete_shard_must_be_discarded_explicitly(tmp_path, monkeypatch):
    from maxdiffusion import null_adapter_shards as shards

    original, calls = shards._publish, []

    def failing(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("host preempted")
        original(source, destination)

    monkeypatch.setattr(shards, "_publish", failing)
    with pytest.raises(OSError):
        _write(tmp_path)
    monkeypatch.setattr(shards, "_publish", original)

    with pytest.raises(FileExistsError, match="discard_incomplete_shard"):
        _write(tmp_path)
    assert discard_incomplete_shard(str(tmp_path / "cache" / "shard_00000")) is True
    assert _write(tmp_path).count == len(_NAMES)  # ... and the retry now works


def test_discarding_refuses_a_completed_shard(tmp_path):
    _write(tmp_path)

    with pytest.raises(FileExistsError, match="completion marker"):
        discard_incomplete_shard(str(tmp_path / "cache" / "shard_00000"))
    assert discard_incomplete_shard(str(tmp_path / "cache" / "never_existed")) is False


def test_a_second_writer_racing_on_the_same_shard_fails_rather_than_corrupts(tmp_path, monkeypatch):
    """Interleaved: the loser's publish lands on an artifact the winner already put there."""
    from maxdiffusion import null_adapter_shards as shards

    original = shards._publish
    state = {"raced": False}

    def racing(source, destination):
        if not state["raced"] and destination.endswith("record_00000.npz"):
            state["raced"] = True
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as handle:  # the other writer got there first
                handle.write(b"someone else's bytes")
        original(source, destination)

    monkeypatch.setattr(shards, "_publish", racing)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write(tmp_path)
    assert state["raced"] and not os.path.exists(_marker_path(tmp_path))


def test_a_writer_arriving_after_another_has_started_publishing_is_refused(tmp_path, monkeypatch):
    from maxdiffusion import null_adapter_shards as shards

    original = shards._publish
    second = {}

    def hooked(source, destination):
        original(source, destination)
        if destination.endswith(HEADER_NAME) and "attempted" not in second:
            second["attempted"] = True
            with pytest.raises(FileExistsError, match="incomplete shard"):
                _write(tmp_path)  # a second writer arrives mid-publication

    monkeypatch.setattr(shards, "_publish", hooked)

    _write(tmp_path)

    assert second.get("attempted") and _validate(tmp_path).valid


def test_the_reviewers_path_escape_probe_is_refused(tmp_path):
    """A marker pointing at another shard's staging leftover -- with a genuine record and a correct
    digest waiting there -- used to validate, defeating marker-last entirely."""
    import hashlib

    elsewhere = tmp_path / "staging" / "other_shard.staging"
    elsewhere.mkdir(parents=True)
    smuggled = record_to_bytes(_record(_NAMES[0], 0))
    (elsewhere / "record_00000.npz").write_bytes(smuggled)
    shard = _handcraft(tmp_path, _records(_NAMES[:1]))
    payload = json.loads((tmp_path / "cache" / "hand" / MARKER_NAME).read_text())
    payload["files"][_NAMES[0]] = "../../staging/other_shard.staging/record_00000.npz"
    payload["sha256"][_NAMES[0]] = hashlib.sha256(smuggled).hexdigest()
    (tmp_path / "cache" / "hand" / MARKER_NAME).write_text(json.dumps(payload))

    report = validate_shard(
        shard,
        expected_header_fingerprint=_fingerprint(),
        expected_arm=_ARM,
        expected_noise_convention=_CONVENTION,
    )

    assert not report.valid and report.names == ()


@pytest.mark.parametrize(
    "override",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"count": -1},
        {"count": 99},
        {"names": ["b/w0", "a/w0"]},
        {"names": [_NAMES[0], _NAMES[0]]},
        {"names": ["", ""]},
        {"quarantined": "x"},
        {"quarantined": ["a"]},
        {"quarantined": {"a": 5}},
        {"header_fingerprint": "nothex"},
        {"header_fingerprint": 7},
        {"sha256": {"a": "z" * 64}},
        {"files": {_NAMES[0]: "record_00001.npz"}},
    ],
)
def test_a_type_confused_marker_yields_an_invalid_report_not_an_exception(tmp_path, override):
    shard = _handcraft(tmp_path, _records(_NAMES[:1]), marker_overrides=override)

    report = validate_shard(
        shard, expected_header_fingerprint=_fingerprint(), expected_arm=_ARM, expected_noise_convention=_CONVENTION
    )

    assert not report.valid and report.names == () and report.quarantined == {}


def test_duplicate_json_keys_in_a_marker_are_refused(tmp_path):
    shard = _handcraft(tmp_path, _records(_NAMES[:1]))
    payload = (tmp_path / "cache" / "hand" / MARKER_NAME).read_text()
    doubled = payload.replace('"count":', '"count": 99, "count":', 1)
    (tmp_path / "cache" / "hand" / MARKER_NAME).write_text(doubled)

    report = validate_shard(
        shard, expected_header_fingerprint=_fingerprint(), expected_arm=_ARM, expected_noise_convention=_CONVENTION
    )

    assert not report.valid


@pytest.mark.parametrize(
    "tamper, reason",
    [
        (lambda p, m: open(os.path.join(p, m.files[_NAMES[0]]), "r+b").write(b"\x00\x01\x02\x03"), "sha256"),
        (lambda p, m: os.remove(os.path.join(p, m.files[_NAMES[1]])), "missing"),
        (lambda p, m: open(os.path.join(p, MARKER_NAME), "w").write("{not json"), "could not be validated"),
        (lambda p, m: os.remove(os.path.join(p, HEADER_NAME)), "header"),
        (
            lambda p, m: open(os.path.join(p, HEADER_NAME), "w").write(header_to_json(_header(guide_scale=7.0))),
            "header",
        ),
    ],
)
def test_a_tampered_shard_is_invalid(tmp_path, tamper, reason):
    marker = _write(tmp_path)
    shard = str(tmp_path / "cache" / "shard_00000")

    tamper(shard, marker)

    report = _validate(tmp_path)
    assert not report.valid
    assert any(reason in line for line in report.reasons), report.reasons


def test_a_record_file_holding_a_different_example_is_invalid(tmp_path):
    """Swap two records *and* their recorded hashes and every byte checks out; only asking the record
    what it calls itself catches a consistent mislabelling."""
    import hashlib

    records = _records(_NAMES[:2])
    first, second = records[0].name, records[1].name
    files = canonical_files([first, second])
    swapped = {first: files[second], second: files[first]}
    digests = {record.name: hashlib.sha256(record_to_bytes(record)).hexdigest() for record in records}
    # Each canonical file now holds the *other* record, and the marker's digests were swapped to match
    # what is actually on disk -- so every hash checks out and only the identity question is left.
    shard = _handcraft(
        tmp_path,
        records,
        files=swapped,
        marker_overrides={"files": files, "sha256": {first: digests[second], second: digests[first]}},
    )

    report = validate_shard(
        shard, expected_header_fingerprint=_fingerprint(), expected_arm=_ARM, expected_noise_convention=_CONVENTION
    )

    assert not report.valid
    assert sum("holds a different example" in line for line in report.reasons) == 2, report.reasons


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"expected_header_fingerprint": "d" * 64}, "fingerprint"),
        ({"expected_arm": "A2"}, "expected A2"),
        ({"expected_noise_convention": "global"}, "expected A1/global"),
    ],
)
def test_a_shard_from_a_different_run_is_invalid(tmp_path, kwargs, reason):
    _write(tmp_path)

    report = _validate(tmp_path, **kwargs)

    assert not report.valid and any(reason in line for line in report.reasons), report.reasons


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"expected_header_fingerprint": "short"}, "64-hex"),
        ({"expected_header_fingerprint": None}, "64-hex"),
        ({"expected_arm": ""}, "expected_arm"),
        ({"expected_noise_convention": "sometimes"}, "expected_noise_convention"),
    ],
)
def test_malformed_expectations_are_a_caller_error_not_an_invalid_report(tmp_path, kwargs, message):
    _write(tmp_path)

    with pytest.raises(ValueError, match=message):
        _validate(tmp_path, **kwargs)


def test_a_record_whose_dtype_contradicts_its_header_is_invalid(tmp_path):
    """The writer refuses this, so the reader is shown one built by hand -- the R4c pair contract."""
    header = _header(dtype_policy="fp32")
    shard = _handcraft(tmp_path, _records(_NAMES[:1], latent_dtype="fp16"), header)

    report = validate_shard(
        shard,
        expected_header_fingerprint=_fingerprint(header),
        expected_arm=_ARM,
        expected_noise_convention=_CONVENTION,
    )

    assert not report.valid and any("latent_dtype" in line for line in report.reasons)


@pytest.mark.parametrize(
    "records, message",
    [
        (lambda: [_record(_NAMES[0], 0, latent_dtype="fp32")], "disagrees with the header"),
        (lambda: [_record(_NAMES[0], 0), _record(_NAMES[1], 1, arm="A2")], "one arm and one convention"),
        (lambda: [_record(_NAMES[0], 0), _record(_NAMES[1], 1, convention="global")], "one arm and one convention"),
    ],
)
def test_a_heterogeneous_shard_is_never_written(tmp_path, records, message):
    with pytest.raises(ValueError, match=message):
        write_shard(records(), _header(), str(tmp_path / "cache" / "mixed"), str(tmp_path / "staging"))
    assert not os.path.exists(str(tmp_path / "cache" / "mixed" / MARKER_NAME))


def test_the_header_contract_from_r6_is_enforced_on_write(tmp_path):
    for config in ({"inner_iters": 10}, {"inner_iters": 10, "lr": 0.01, "extra": 1}, {}):
        with pytest.raises(ValueError, match="optimization_config"):
            write_shard(
                _records(),
                _header(optimization_config=config),
                str(tmp_path / "cache" / "bad"),
                str(tmp_path / "staging"),
            )
    assert not os.path.exists(str(tmp_path / "cache" / "bad" / MARKER_NAME))


def test_duplicate_names_within_one_shard_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        write_shard(
            _records((_NAMES[0], _NAMES[0])), _header(), str(tmp_path / "cache" / "dupe"), str(tmp_path / "staging")
        )


def test_an_empty_shard_with_nothing_quarantined_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one record"):
        write_shard([], _header(), str(tmp_path / "cache" / "empty"), str(tmp_path / "staging"))


def test_resume_returns_exactly_the_uncovered_manifest_names(tmp_path):
    manifest = (*_NAMES, "droid_ep_000004/w0", "droid_ep_000005/w0")
    _write(tmp_path, names=_NAMES[:2], shard="shard_00000")
    _write(tmp_path, names=_NAMES[2:], shard="shard_00001")

    plan = _resume(tmp_path, manifest, ["shard_00000", "shard_00001"])

    assert plan.covered == tuple(sorted(_NAMES))
    assert plan.todo == ("droid_ep_000004/w0", "droid_ep_000005/w0")
    assert plan.quarantined == {} and all(report.valid for report in plan.shards)


def test_resume_ignores_an_invalid_shard_and_redoes_its_names(tmp_path):
    _write(tmp_path, names=_NAMES[:2], shard="shard_00000")
    _write(tmp_path, names=_NAMES[2:], shard="shard_00001")
    os.remove(str(tmp_path / "cache" / "shard_00001" / MARKER_NAME))

    plan = _resume(tmp_path, _NAMES, ["shard_00000", "shard_00001"])

    assert plan.covered == tuple(sorted(_NAMES[:2])) and plan.todo == (_NAMES[2],)


def test_resume_redoes_the_names_of_a_shard_that_fails_its_integrity_check(tmp_path):
    marker = _write(tmp_path, names=_NAMES[:2], shard="shard_00000")
    _write(tmp_path, names=_NAMES[2:], shard="shard_00001")
    with open(str(tmp_path / "cache" / "shard_00000" / marker.files[_NAMES[0]]), "r+b") as handle:
        handle.write(b"\x00\x01\x02\x03")

    plan = _resume(tmp_path, _NAMES, ["shard_00000", "shard_00001"])

    assert plan.covered == (_NAMES[2],)
    assert plan.todo == tuple(_NAMES[:2])
    assert plan.shards[0].names == () and not plan.shards[0].valid


def test_resume_refuses_shards_from_two_different_runs(tmp_path):
    """Both shards are internally perfect; they simply were not written by the same run."""
    _write(tmp_path, names=_NAMES[:2], shard="shard_00000")
    _write(tmp_path, names=_NAMES[2:], shard="shard_00001", header=_header(code_sha="f" * 40))

    plan = _resume(tmp_path, _NAMES, ["shard_00000", "shard_00001"])

    assert plan.covered == tuple(sorted(_NAMES[:2]))
    assert plan.todo == (_NAMES[2],)  # the foreign shard covers nothing


def test_the_same_name_published_by_two_valid_shards_is_a_hard_error(tmp_path):
    """Shards are immutable, so two copies of one example leave no way to say which the cohort means."""
    _write(tmp_path, names=_NAMES[:2], shard="shard_00000")
    _write(tmp_path, names=_NAMES[1:], shard="shard_00001")

    with pytest.raises(ValueError, match="is published by more than one validated shard"):
        _resume(tmp_path, _NAMES, ["shard_00000", "shard_00001"])


def test_a_shard_covering_a_name_outside_the_manifest_is_a_hard_error(tmp_path):
    _write(tmp_path)

    with pytest.raises(ValueError, match="outside the manifest"):
        _resume(tmp_path, _NAMES[:2], ["shard_00000"])


def test_a_quarantined_name_outside_the_manifest_is_a_hard_error(tmp_path):
    _write(tmp_path, names=_NAMES[:2], quarantined={"droid_ep_000999/w0": "boom"})

    with pytest.raises(ValueError, match="outside the manifest"):
        _resume(tmp_path, _NAMES, ["shard_00000"])


def test_a_later_success_supersedes_an_earlier_quarantine(tmp_path):
    """The normal shape of a retry: lost in one attempt, published by the next.

    R8's first cut raised here, which meant a cache job could be retried exactly once -- the retry's
    own shard turned the *next* resume into a duplicate-name error -- while the report meanwhile
    counted the name as both covered and lost (R10 review, finding 5).
    """
    _write(tmp_path, names=_NAMES[:2], shard="shard_00000", quarantined={_NAMES[2]: "boom"})
    _write(tmp_path, names=_NAMES[2:], shard="shard_00001")

    plan = _resume(tmp_path, _NAMES, ["shard_00000", "shard_00001"])

    assert plan.covered == tuple(sorted(_NAMES))
    assert plan.todo == ()
    assert plan.quarantined == {}  # the gap was closed; it is not still open
    assert plan.superseded == (_NAMES[2],)  # and the history is reported, not erased


def test_a_quarantine_no_later_shard_closed_stays_in_the_current_gap(tmp_path):
    _write(tmp_path, names=_NAMES[:2], shard="shard_00000", quarantined={_NAMES[2]: "boom"})

    plan = _resume(tmp_path, _NAMES, ["shard_00000"])

    assert plan.quarantined == {_NAMES[2]: "boom"} and plan.superseded == ()
    assert plan.todo == (_NAMES[2],)  # a quarantined name is never covered


def test_a_manifest_with_duplicates_is_refused(tmp_path):
    with pytest.raises(ValueError, match="manifest names must be unique"):
        _resume(tmp_path, (_NAMES[0], _NAMES[0]), [])


def test_a_quarantined_example_is_named_in_the_marker_and_is_not_covered(tmp_path):
    quarantined = {_NAMES[1]: "ExampleDivergenceError: tracking_losses must be finite"}
    survivors = tuple(name for name in _NAMES if name not in quarantined)

    marker = _write(tmp_path, names=survivors, quarantined=quarantined)

    assert marker.quarantined == quarantined
    report = _validate(tmp_path)
    assert report.valid and report.quarantined == quarantined

    plan = _resume(tmp_path, _NAMES, ["shard_00000"])
    assert plan.covered == tuple(sorted(survivors))
    assert plan.todo == (_NAMES[1],)  # handed back, not written off
    assert plan.quarantined == quarantined


def test_a_diagnostic_shard_with_no_records_still_carries_the_real_header(tmp_path):
    """Ruling 3: a quarantine-only attempt may exist, but only as zero coverage with real provenance."""
    marker = write_shard(
        [], _header(), str(tmp_path / "cache" / "diag"), str(tmp_path / "staging"), quarantined={_NAMES[0]: "boom"}
    )

    assert marker.count == 0 and marker.names == () and marker.header_fingerprint == _fingerprint()
    report = _validate(tmp_path, shard="diag")
    assert report.valid and report.quarantined == {_NAMES[0]: "boom"}
    assert _resume(tmp_path, _NAMES, ["diag"]).todo == tuple(_NAMES)
    assert json.loads((tmp_path / "cache" / "diag" / HEADER_NAME).read_text())["model_revision"] == _REVISION


def test_a_name_cannot_be_both_written_and_quarantined(tmp_path):
    with pytest.raises(ValueError, match="both written and quarantined"):
        _write(tmp_path, names=_NAMES[:2], quarantined={_NAMES[0]: "boom"})


def test_a_gate_cohort_containing_a_quarantined_name_fails_coverage(tmp_path):
    """The other half of the policy: a lost example costs TRAIN-2000 one row, but it can never
    quietly shrink a DEV/TEST cohort into a verdict."""
    survivors = [name for name in _NAMES if name != _NAMES[1]]
    table = {name: {"0": {"future_mse": 1.0, "future_ssim": 0.9}} for name in survivors}
    control = {name: {"0": {"future_mse": 9.0, "future_ssim": 0.4}} for name in survivors}

    verdict = gate_g1(table, control, list(_NAMES), NoiseConvention.GLOBAL)

    assert not verdict.passed and verdict.reasons == ("coverage",)
    assert verdict.numbers["missing_names"]["method"] == [_NAMES[1]]


def _marker_payload(**overrides):
    """A well-formed single-record marker payload; ``overrides`` replaces fields verbatim.

    No parameter shadows a marker field, so ``_marker_payload(names=...)`` really does override the
    marker's ``names`` rather than quietly reconfiguring the helper.
    """
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "count": 1,
        "names": [_NAMES[0]],
        "files": canonical_files([_NAMES[0]]),
        "sha256": {_NAMES[0]: "a" * 64},
        "header_fingerprint": "f" * 64,
        "quarantined": {},
    }
    return json.dumps({**payload, **overrides})


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema_version": True}, "must be an integer"),
        ({"schema_version": 2}, "unsupported marker schema_version"),
        ({"count": True}, "must be an integer"),
        ({"count": -1}, "does not match"),
        (
            {
                "names": [_NAMES[1], _NAMES[0]],  # the same two names, listed out of order
                "count": 2,
                "files": canonical_files(_NAMES[:2]),
                "sha256": dict.fromkeys(_NAMES[:2], "a" * 64),
            },
            "sorted and unique",
        ),
        ({"names": ["", ""], "count": 2}, "non-empty strings"),
        ({"names": "notalist"}, "list of non-empty strings"),
        ({"files": {_NAMES[0]: "record_00001.npz"}}, "canonical"),
        ({"files": {_NAMES[0]: "../elsewhere/record_00000.npz"}}, "canonical"),
        ({"files": "notadict"}, "mapping of non-empty strings"),
        ({"sha256": {_NAMES[0]: "A" * 64}}, "64-hex digest per name"),
        ({"sha256": {_NAMES[0]: "abc"}}, "64-hex digest per name"),
        ({"sha256": {"stranger": "a" * 64}}, "64-hex digest per name"),
        ({"header_fingerprint": "nothex"}, "64-hex digest"),
        ({"header_fingerprint": 7}, "64-hex digest"),
        ({"quarantined": "x"}, "mapping of non-empty strings"),
        ({"quarantined": ["a"]}, "mapping of non-empty strings"),
        ({"quarantined": {"a": 5}}, "mapping of non-empty strings"),
        ({"quarantined": {_NAMES[0]: "boom"}}, "both published and quarantined"),
        ({"names": [], "count": 0, "files": {}, "sha256": {}}, "publish records or record a quarantine"),
    ],
)
def test_every_marker_clause_refuses_its_own_violation(overrides, message):
    """The strict validator is tested at its own boundary: inside ``validate_shard`` most of these
    would also be caught later by a byte comparison, which would leave the parser's clauses unpinned."""
    with pytest.raises(ValueError, match=message):
        marker_from_json(_marker_payload(**overrides))


def test_the_marker_parser_refuses_duplicate_json_keys():
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        marker_from_json(_marker_payload().replace('"count":', '"count": 99, "count":', 1))


@pytest.mark.parametrize(
    "filename",
    [
        "../../staging/other_shard.staging/record_00000.npz",
        "/etc/passwd",
        "sub/record_00000.npz",
        "record_0.npz",
        "record_00000.npz.bak",
        "./record_00000.npz",
        "",
        None,
    ],
)
def test_a_shard_member_can_never_leave_its_shard(filename):
    from maxdiffusion.null_adapter_shards import _shard_member

    with pytest.raises(ValueError):
        _shard_member("gs://bucket/cache/shard_00000", filename)

    assert _shard_member("gs://bucket/cache/shard_00000", "record_00007.npz").endswith(
        "cache/shard_00000/record_00007.npz"
    )


def test_a_shuffled_but_internally_consistent_bijection_is_refused(tmp_path):
    """Every digest matches and every record names itself -- the mapping is simply not the canonical
    one, which is the only thing standing between a marker and a file of its own choosing."""
    import hashlib

    records = _records(_NAMES[:2])
    first, second = records[0].name, records[1].name
    canonical = canonical_files([first, second])
    shuffled = {first: canonical[second], second: canonical[first]}
    digests = {record.name: hashlib.sha256(record_to_bytes(record)).hexdigest() for record in records}
    shard = _handcraft(tmp_path, records, files=shuffled, marker_overrides={"files": shuffled, "sha256": digests})

    report = validate_shard(
        shard, expected_header_fingerprint=_fingerprint(), expected_arm=_ARM, expected_noise_convention=_CONVENTION
    )

    assert not report.valid and report.names == ()


def test_the_provenance_expectations_are_required_keyword_arguments():
    """A default would let a resume skip the provenance question entirely (R8 review, finding 2)."""
    import inspect

    for function in (validate_shard, resume_plan):
        parameters = inspect.signature(function).parameters
        for key in ("expected_header_fingerprint", "expected_arm", "expected_noise_convention"):
            assert parameters[key].default is inspect.Parameter.empty, (function.__name__, key)
            assert parameters[key].kind is inspect.Parameter.KEYWORD_ONLY, (function.__name__, key)
    with pytest.raises(TypeError):
        validate_shard("/cache/shard_00000")


def test_each_attempt_stages_under_a_path_it_owns_alone():
    """Deterministic staging is how two writers delete each other's work (R8 review, finding 6)."""
    from maxdiffusion.null_adapter_shards import STAGING_SUFFIX, _staging_path

    first = _staging_path("gs://bucket/cache/shard_00000", "gs://bucket/staging")
    second = _staging_path("gs://bucket/cache/shard_00000", "gs://bucket/staging")

    assert first != second
    for path in (first, second):
        assert path.startswith("gs://bucket/staging/shard_00000.") and path.endswith(STAGING_SUFFIX)


def test_the_marker_round_trips_through_json():
    marker = ShardMarker(
        schema_version=SHARD_SCHEMA_VERSION,
        count=1,
        names=(_NAMES[0],),
        files=canonical_files([_NAMES[0]]),
        sha256={_NAMES[0]: "e" * 64},
        header_fingerprint="f" * 64,
        quarantined={_NAMES[1]: "boom"},
    )

    restored = marker_from_json(marker.to_json())

    assert restored == marker
    assert json.loads(marker.to_json())["names"] == [_NAMES[0]]
    with pytest.raises(ValueError, match="schema_version"):
        marker_from_json(json.dumps({**json.loads(marker.to_json()), "schema_version": 99}))
    with pytest.raises(ValueError, match="marker fields"):
        marker_from_json(json.dumps({**json.loads(marker.to_json()), "stranger": 1}))


def test_an_absent_shard_directory_is_simply_invalid(tmp_path):
    report = _validate(tmp_path, shard="never_written")

    assert not report.valid and report.names == ()


def test_the_header_fingerprint_is_the_whole_header_not_one_of_its_fields():
    """The near-miss R10 shipped: ``base_context_fingerprint`` is 64 hex too, so it passes every type
    check on the way into ``resume_plan`` and then rejects every valid shard as another run's."""
    header = _header()

    assert header_fingerprint(header) == _fingerprint(header)
    assert header_fingerprint(header) != header.base_context_fingerprint
    # ... and it moves with any field, which is the property a resume is relying on.
    assert header_fingerprint(dataclasses.replace(header, code_sha="d" * 40)) != header_fingerprint(header)


def test_a_resume_handed_the_base_context_fingerprint_covers_nothing(tmp_path):
    _write(tmp_path)

    plan = _resume(tmp_path, _NAMES, ["shard_00000"], expected_header_fingerprint=_header().base_context_fingerprint)

    assert plan.covered == () and plan.todo == tuple(_NAMES)


@pytest.mark.parametrize(
    "paths, expected",
    [
        ((), 0),
        (("gs://b/run/shard_00000",), 1),
        (("gs://b/run/shard_00000", "gs://b/run/shard_00001"), 2),
        (("gs://b/run/shard_00007", "gs://b/run/shard_00001"), 8),  # the highest, not the count
        (("gs://b/run/shard_00003/",), 4),  # a trailing separator is not a new identity
        (("gs://b/run/a1/shard_00002",), 3),  # per-arm subdirectories still carry the index
        (("gs://b/run/videos", "gs://b/run/shard_00000"), 1),  # a stranger directory is ignored
        (("gs://b/run/shard_0001",), 0),  # not the canonical five-digit form
    ],
)
def test_the_next_shard_index_comes_from_the_paths_not_from_their_count(paths, expected):
    assert next_shard_index(paths) == expected


def test_a_resume_that_lost_one_shard_to_validation_does_not_reuse_a_live_identity(tmp_path):
    """Counting what a resume *found* hands the next attempt an identity that is already immutable."""
    _write(tmp_path, names=_NAMES[:2], shard="shard_00000")
    _write(tmp_path, names=_NAMES[2:], shard="shard_00001")
    with open(str(tmp_path / "cache" / "shard_00000" / canonical_files(_NAMES[:2])[_NAMES[0]]), "r+b") as handle:
        handle.write(b"\x00\x01\x02\x03")

    plan = _resume(tmp_path, _NAMES, ["shard_00000", "shard_00001"])
    published = [report.path for report in plan.shards if report.valid]

    assert len(published) == 1  # only shard_00001 survived validation
    assert next_shard_index([str(tmp_path / "cache" / s) for s in ("shard_00000", "shard_00001")]) == 2
