"""exp_04 R4b — the cached artifact record and its serialization (plan §4-P2).

A record **plus its shard provenance header** is what R4c replays from: the record carries the nulls,
``z_start``, ``expected_final_latent`` and its hash; the header carries the sigma grid, guidance
weight, model revision and base-context fingerprint. Neither half is sufficient alone, and this file
pins the half that makes such a check meaningful -- that what was written is exactly what comes back,
and that anything else is refused loudly rather than parsed optimistically.

What the round hangs on:

- **Round trip is bitwise, per field, per ``latent_dtype``.** fp16 is the default storage; the fp16
  fidelity gate (plan §4-P2/N8) can flip exactly ``{nulls, z_start, expected_final_latent}`` to fp32
  while ``z_i0``/``z_video`` stay fp16 because that is the *source* data's dtype.
- **The hash covers the stored bytes, and cannot go stale.** ``make_record`` casts before hashing and
  freezes the arrays; the validator recomputes the digest on every write, so neither a replaced hash
  nor a post-hoc mutation can reach a shard (Codex R4b review, finding 1).
- **The public codec is the production boundary.** It enforces the deployment geometry exactly and
  rejects non-finite data (finding 2). Tests that want small arrays go through the private
  ``_Geometry`` seam, which shares the whole validation path and differs only in the size table.
- **Provenance is validated identically on write and read** (finding 3), and every namespace the
  reader parses is closed (finding 4).
"""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile

import numpy as np
import pytest

from maxdiffusion.null_adapter_records import (
    LATENT_DTYPES,
    SCHEMA_VERSION,
    NullAdapterRecord,
    ProvenanceHeader,
    _Geometry,
    _make_record,
    _record_from_bytes,
    _record_to_bytes,
    header_from_json,
    header_to_json,
    make_record,
    record_from_bytes,
    record_to_bytes,
    sha256_of_array,
)


# The private seam: same codec, same validators, smaller arrays.
_TINY = _Geometry(z_video=(2, 3, 4, 6), z_i0=(2, 1, 4, 6), actions=(5, 7), nulls=(4, 3, 8), per_step_final_losses=(4,))
_PRODUCTION = _Geometry()


def _arrays(geometry=_TINY, seed=0):
    rng = np.random.default_rng(seed)
    shapes = geometry.shapes()
    return {field: rng.standard_normal(shapes[field], dtype=np.float32) for field in shapes}


def _record(latent_dtype="fp16", geometry=_TINY, seed=0, **overrides):
    fields = {
        "name": "droid_ep_000001/w0",
        "ordinal": 3,
        "split": "dev",
        "episode": "droid_ep_000001",
        "latent_dtype": latent_dtype,
        "noise_convention": "keyed",
        "arm": "A1",
        "final_future_mse": 0.0123,
        **_arrays(geometry, seed),
    }
    fields.update(overrides)
    return _make_record(geometry=geometry, **fields)


def _header(**overrides):
    fields = {
        "manifest_hash": "a" * 64,
        "code_sha": "b" * 40,
        "model_revision": "Wan2.2-TI2V-5B@refs/pr/1",
        "sigma_vector": np.linspace(1.0, 0.0, 26, dtype=np.float32),
        "guide_scale": 5.0,
        "base_context_fingerprint": "c" * 64,
        "optimization_config": {"inner_iters": 10, "lr": 0.01, "eps": 1e-8, "warm_start": True},
        "dtype_policy": "fp16",
        "l_null": 16,
    }
    fields.update(overrides)
    return ProvenanceHeader(**fields)


def _repack(blob, mutate):
    """Rebuild an archive after ``mutate`` edits its member dict, leaving the format identical."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    mutate(members)  # edits the dict in place; any return value is ignored
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, value in members.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(value), allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), payload.getvalue())
    return buffer.getvalue()


def _edit_meta(blob, mutate):
    def _mutate(members):
        meta = json.loads(bytes(members["__meta__"]).decode("utf-8"))
        mutate(meta)
        members["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

    return _repack(blob, _mutate)


@pytest.mark.parametrize("latent_dtype", ["fp16", "fp32"])
def test_round_trip_is_bitwise_for_every_field(latent_dtype):
    record = _record(latent_dtype=latent_dtype)

    restored = _record_from_bytes(_record_to_bytes(record, _TINY), _TINY)

    for field in dataclasses.fields(NullAdapterRecord):
        original, back = getattr(record, field.name), getattr(restored, field.name)
        if isinstance(original, np.ndarray):
            assert back.dtype == original.dtype, field.name
            np.testing.assert_array_equal(back.tobytes(), original.tobytes(), err_msg=field.name)
        else:
            assert back == original, field.name


@pytest.mark.parametrize("latent_dtype", ["fp16", "fp32"])
def test_stored_dtypes_follow_the_latent_dtype_policy(latent_dtype):
    """The fidelity-gate fallback flips exactly {nulls, z_start, expected_final_latent} (plan M5)."""
    restored = _record_from_bytes(_record_to_bytes(_record(latent_dtype=latent_dtype), _TINY), _TINY)

    for field in ("nulls", "z_start", "expected_final_latent"):
        assert getattr(restored, field).dtype == LATENT_DTYPES[latent_dtype], field
    for field in ("z_i0", "z_video"):
        assert getattr(restored, field).dtype == np.dtype("<f2"), field  # source dtype, never flipped
    assert restored.per_step_final_losses.dtype == np.dtype("<f4")


def test_make_record_hashes_the_stored_bytes_not_the_inputs():
    """fp32 in, fp16 stored: the hash must describe what a reader will actually read back."""
    arrays = _arrays(seed=1)
    record = _record(latent_dtype="fp16", **arrays)

    assert record.expected_final_latent.dtype == np.dtype("<f2")
    assert not np.array_equal(record.expected_final_latent.astype(np.float64), arrays["expected_final_latent"])
    _record_from_bytes(_record_to_bytes(record, _TINY), _TINY)  # the reader recomputes the hash and must agree
    assert record.expected_final_latent_sha256 != _record(latent_dtype="fp32", **arrays).expected_final_latent_sha256


def test_stored_arrays_are_read_only():
    """Frozen at the source, so the recorded hash cannot be invalidated behind the codec's back."""
    record = _record()

    for field in ("nulls", "z_start", "expected_final_latent", "z_i0", "z_video", "per_step_final_losses"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(record, field)[(0,) * getattr(record, field).ndim] = 0.0
    restored = _record_from_bytes(_record_to_bytes(record, _TINY), _TINY)
    with pytest.raises(ValueError, match="read-only"):
        restored.nulls[0, 0, 0] = 0.0


def test_writer_rejects_a_replaced_hash():
    record = dataclasses.replace(_record(), expected_final_latent_sha256="0" * 64)

    with pytest.raises(ValueError, match="expected_final_latent sha256 mismatch"):
        _record_to_bytes(record, _TINY)


def test_writer_rejects_an_array_mutated_after_make_record():
    """The freeze blocks the easy path; the writer's recomputed digest blocks the rest."""
    record = _record()
    thawed = record.expected_final_latent.copy()
    thawed[0, 0, 0, 0] = np.float16(thawed[0, 0, 0, 0] + np.float16(1.0))

    with pytest.raises(ValueError, match="expected_final_latent sha256 mismatch"):
        _record_to_bytes(dataclasses.replace(record, expected_final_latent=thawed), _TINY)


def test_public_codec_round_trips_the_production_geometry():
    record = _record(geometry=_PRODUCTION, seed=7)

    restored = record_from_bytes(record_to_bytes(record))

    assert restored.nulls.shape == (25, 16, 4096)
    assert restored.z_video.shape == (48, 9, 12, 20)
    np.testing.assert_array_equal(restored.z_video.tobytes(), record.z_video.tobytes())


@pytest.mark.parametrize("field", ["nulls", "z_video", "actions", "z_i0", "per_step_final_losses"])
def test_public_codec_rejects_off_geometry_tensors(field):
    """The artifact boundary is where the deployment schema is authoritative (review finding 2)."""
    arrays = _arrays(_PRODUCTION, seed=8)
    arrays[field] = arrays[field][..., :-1]

    with pytest.raises(ValueError, match=f"{field} must have shape"):
        make_record(
            name="n",
            ordinal=0,
            split="dev",
            episode="e",
            latent_dtype="fp16",
            noise_convention="keyed",
            arm="A1",
            final_future_mse=0.5,
            **arrays,
        )


def test_public_codec_rejects_a_tiny_record():
    """The private seam is for tests only: the same arrays must not survive the public entry point."""
    with pytest.raises(ValueError, match="must have shape"):
        make_record(
            name="n",
            ordinal=0,
            split="dev",
            episode="e",
            latent_dtype="fp16",
            noise_convention="keyed",
            arm="A1",
            final_future_mse=0.5,
            **_arrays(_TINY),
        )


@pytest.mark.parametrize("field", ["nulls", "expected_final_latent", "per_step_final_losses"])
def test_rejects_nonfinite_tensors(field):
    arrays = _arrays(seed=9)
    arrays[field] = arrays[field].copy()
    arrays[field].flat[0] = np.nan

    with pytest.raises(ValueError, match=f"{field} must be finite"):
        _record(**arrays)


def test_rejects_a_nonfinite_final_future_mse():
    with pytest.raises(ValueError, match="final_future_mse must be finite"):
        _record(final_future_mse=float("inf"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"latent_dtype": "bf16"}, "latent_dtype must be one of"),
        ({"noise_convention": "fresh"}, "noise_convention must be one of"),
        ({"ordinal": 1.5}, "ordinal must be an integer"),
        ({"nulls": np.zeros((4, 3))}, "nulls must have shape"),
        ({"z_start": np.zeros((2, 4, 4, 6))}, "z_start must have shape"),
        ({"z_i0": np.zeros((2, 2, 4, 6))}, "z_i0 must have shape"),
    ],
)
def test_make_record_rejects_malformed_fields(overrides, message):
    with pytest.raises(ValueError, match=message):
        _record(**overrides)


def test_serialization_is_byte_deterministic():
    """Identical content ⇒ identical bytes, so a shard's integrity marker (R8) means something.

    Comparing two back-to-back serializations is not enough: zip timestamps have two-second
    granularity, so a wall-clock timestamp would usually agree with itself. The archive's own
    ``date_time`` fields are asserted to be the fixed epoch, which is what actually makes the blob
    reproducible tomorrow as well as twice in a row.
    """
    record = _record()

    blob = _record_to_bytes(record, _TINY)

    assert blob == _record_to_bytes(record, _TINY)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        stamps = {info.date_time for info in archive.infolist()}
    assert stamps == {(1980, 1, 1, 0, 0, 0)}, stamps


def test_reader_rejects_a_truncated_array():
    record = _record()
    blob = _repack(_record_to_bytes(record, _TINY), lambda m: m.update(nulls=record.nulls[:-1]))

    with pytest.raises(ValueError, match="nulls is .* bytes"):
        _record_from_bytes(blob, _TINY)


def test_reader_rejects_an_array_stored_in_the_wrong_dtype():
    record = _record(latent_dtype="fp16")
    blob = _repack(_record_to_bytes(record, _TINY), lambda m: m.update(z_start=record.z_start.astype(np.float32)))

    with pytest.raises(ValueError, match="z_start is .* bytes"):
        _record_from_bytes(blob, _TINY)


def test_reader_rejects_tampered_expected_final_latent():
    record = _record()
    tampered = record.expected_final_latent.copy()
    tampered.flat[0] = np.float16(tampered.flat[0] + np.float16(1.0))
    blob = _repack(_record_to_bytes(record, _TINY), lambda m: m.update(expected_final_latent=tampered))

    with pytest.raises(ValueError, match="expected_final_latent sha256 mismatch"):
        _record_from_bytes(blob, _TINY)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m: m.pop("nulls"), "archive members do not match"),
        (lambda m: m.update(extra=np.zeros(2, np.float32)), "archive members do not match"),
    ],
)
def test_reader_closes_the_archive_namespace(mutate, message):
    blob = _repack(_record_to_bytes(_record(), _TINY), mutate)

    with pytest.raises(ValueError, match=message):
        _record_from_bytes(blob, _TINY)


@pytest.mark.filterwarnings("ignore:Duplicate name")  # the duplicate is the point of the test
def test_reader_rejects_duplicate_archive_members():
    blob = _record_to_bytes(_record(), _TINY)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        first = archive.namelist()[0]
        payload = archive.read(first)
    buffer = io.BytesIO(blob)
    with zipfile.ZipFile(buffer, "a", zipfile.ZIP_STORED) as archive:
        archive.writestr(zipfile.ZipInfo(first, date_time=(1980, 1, 1, 0, 0, 0)), payload)

    with pytest.raises(ValueError, match="archive members are duplicated"):
        _record_from_bytes(buffer.getvalue(), _TINY)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda meta: meta.update(extra_key=1), "record metadata keys do not match"),
        (lambda meta: meta.pop("arm"), "record metadata keys do not match"),
        (lambda meta: meta["shapes"].update(extra_field=[1]), "record shape keys do not match"),
        (lambda meta: meta.update(schema_version=SCHEMA_VERSION + 1), "unsupported record schema_version"),
        (lambda meta: meta.update(schema_version=True), "unsupported record schema_version"),
    ],
)
def test_reader_closes_the_metadata_namespace(mutate, message):
    blob = _edit_meta(_record_to_bytes(_record(), _TINY), mutate)

    with pytest.raises(ValueError, match=message):
        _record_from_bytes(blob, _TINY)


def test_hash_helper_canonicalizes_byte_order():
    """Equal values, opposite endianness, one digest -- the helper's documented contract."""
    little = np.arange(6, dtype=np.dtype("<f4")).reshape(2, 3)

    assert sha256_of_array(little.astype(np.dtype(">f4"))) == sha256_of_array(little)


def test_header_json_is_deterministic_and_round_trips_sigmas_exactly():
    header = _header()

    text = header_to_json(header)
    restored = header_from_json(text)

    assert text == header_to_json(header)
    assert restored.sigma_vector.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(restored.sigma_vector.tobytes(), header.sigma_vector.tobytes())
    assert restored.optimization_config == header.optimization_config


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema_version": SCHEMA_VERSION + 1}, "unsupported header schema_version"),
        ({"schema_version": True}, "header schema_version must be an integer"),
        ({"sigma_vector": np.zeros((2, 26), np.float32)}, "sigma_vector must be 1-D"),
        ({"sigma_vector": np.zeros(25, np.float32)}, "sigma_vector must be 1-D"),
        ({"sigma_vector": np.full(26, np.nan, np.float32)}, "sigma_vector must be finite"),
        ({"guide_scale": float("nan")}, "guide_scale must be finite"),
        ({"dtype_policy": "int8"}, "dtype_policy must be one of"),
        ({"l_null": 16.0}, "l_null must be an integer"),
        ({"l_null": True}, "l_null must be an integer"),
        ({"optimization_config": {"schedule": {"warmup": 100}}}, "optimization_config must be flat"),
        ({"optimization_config": {"lr": float("inf")}}, "must be finite"),
    ],
)
def test_header_writer_and_reader_share_one_validator(overrides, message):
    """Every case is asserted on BOTH paths: the reader used to accept what the writer refused."""
    with pytest.raises(ValueError, match=message):
        header_to_json(_header(**overrides))

    payload = json.loads(header_to_json(_header()))
    for key, value in overrides.items():
        payload[key] = value.tolist() if isinstance(value, np.ndarray) else value
    with pytest.raises(ValueError, match=message):
        header_from_json(json.dumps(payload))


def test_header_rejects_unknown_fields():
    payload = json.loads(header_to_json(_header()))
    payload["extra_field"] = 1

    with pytest.raises(ValueError, match="header fields do not match"):
        header_from_json(json.dumps(payload))
