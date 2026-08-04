"""Self-contained cached-artifact records for exp_04 null-text inversion (plan §4-P2).

A record plus its shard provenance header holds everything a reader needs to regenerate one
example's claimed output: the optimized per-step nulls, the ``z_start`` they were optimized from, the
``expected_final_latent`` they are claimed to reach and the hash of that latent's stored bytes
(record), together with the sigma grid, guidance weight, model revision and base-context fingerprint
they were produced under (header). ``verify_replay`` (R4c) takes that **pair** and nothing else -- in
particular never ``z_video``, which rides along only because the adapter trainer needs it. Shard
containers, staging prefixes and completion markers are R8's job; here the unit is one record, one
header and one bytes blob.

**This module is the production artifact boundary.** The public codec enforces the deployment
geometry exactly (``PRODUCTION_GEOMETRY``) and rejects non-finite data, because every later consumer
-- the verifier, the trainer, the evaluator -- trusts what comes out of ``record_from_bytes`` without
re-deriving it. The tiny-shape seam used by the tests is private and shares this module's entire
validation path; only the size table differs.

**Storage dtype.** ``latent_dtype`` defaults to ``fp16``. The fp16 fidelity gate (plan §4-P2/N8) can
flip it to ``fp32`` for exactly ``{nulls, z_start, expected_final_latent}``; ``z_i0`` and ``z_video``
stay fp16 regardless, because that is the dtype of the source data rather than a precision choice.
``make_record`` casts to the declared dtype *before* hashing, so the recorded sha256 always describes
the bytes a reader will actually read back, and it marks the stored arrays read-only so the hash
cannot go stale under the caller's feet.

**Byte discipline.** Every hash is taken over the array's contiguous bytes in the explicitly
little-endian form of its own dtype -- the convention R1 fixed for ``base_context_fingerprint``,
restated here rather than imported so record IO stays numpy-only (it must not need jax).

**Fail-closed reads.** ``record_from_bytes`` closes every namespace it parses (archive members,
metadata keys, shape keys, no duplicate members), checks the schema version, derives each array's
expected byte count from ``latent_dtype`` and the geometry, recomputes the hash, and re-runs the
writer's own validator. A reader is the last thing standing between a corrupted pivot and a training
target that silently means something else.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
LATENT_DTYPES = {"fp16": np.dtype("<f2"), "fp32": np.dtype("<f4")}
LATENT_SCOPED_FIELDS = ("nulls", "z_start", "expected_final_latent")
SOURCE_DTYPES = {"z_i0": np.dtype("<f2"), "z_video": np.dtype("<f2"), "actions": np.dtype("<f4")}
NOISE_CONVENTIONS = ("keyed", "global")
ARRAY_FIELDS = (*SOURCE_DTYPES, *LATENT_SCOPED_FIELDS, "per_step_final_losses")
PRODUCTION_SIGMA_LENGTH = 26  # build_rollout_sigmas(25, ...) -> N+1 entries
_META_KEYS = ("schema_version", "shapes")
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed so a record's bytes depend only on its content


@dataclasses.dataclass(frozen=True)
class _Geometry:
    """The size table the codec validates against; the public API only ever uses production."""

    z_video: tuple[int, ...] = (48, 9, 12, 20)
    z_i0: tuple[int, ...] = (48, 1, 12, 20)
    actions: tuple[int, ...] = (32, 7)
    nulls: tuple[int, ...] = (25, 16, 4096)
    per_step_final_losses: tuple[int, ...] = (25,)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "z_i0": self.z_i0,
            "actions": self.actions,
            "z_video": self.z_video,
            "nulls": self.nulls,
            "z_start": self.z_video,
            "expected_final_latent": self.z_video,
            "per_step_final_losses": self.per_step_final_losses,
        }


PRODUCTION_GEOMETRY = _Geometry()


@dataclasses.dataclass(frozen=True)
class ProvenanceHeader:
    """What every record in a shard was produced by; a reader rejects work it cannot match."""

    manifest_hash: str
    code_sha: str
    model_revision: str
    sigma_vector: np.ndarray
    guide_scale: float
    base_context_fingerprint: str
    optimization_config: dict[str, Any]
    dtype_policy: str
    l_null: int
    schema_version: int = SCHEMA_VERSION


@dataclasses.dataclass(frozen=True)
class NullAdapterRecord:
    """One cached example, per plan §4-P2."""

    name: str
    ordinal: int
    split: str
    episode: str
    z_i0: np.ndarray
    actions: np.ndarray
    z_video: np.ndarray
    latent_dtype: str
    nulls: np.ndarray
    z_start: np.ndarray
    expected_final_latent: np.ndarray
    expected_final_latent_sha256: str
    noise_convention: str
    arm: str
    per_step_final_losses: np.ndarray
    final_future_mse: float


def sha256_of_array(array: np.ndarray) -> str:
    """sha256 over the array's contiguous bytes, canonicalized to little-endian first.

    ``ascontiguousarray`` fixes the layout but preserves byte order, so a big-endian host would
    otherwise hash the same values to a different digest.
    """
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes()).hexdigest()


def _is_integral(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _stored_dtypes(latent_dtype: str) -> dict[str, np.dtype]:
    if latent_dtype not in LATENT_DTYPES:
        raise ValueError(f"latent_dtype must be one of {sorted(LATENT_DTYPES)}, got {latent_dtype!r}")
    return {
        **SOURCE_DTYPES,
        **dict.fromkeys(LATENT_SCOPED_FIELDS, LATENT_DTYPES[latent_dtype]),
        "per_step_final_losses": np.dtype("<f4"),
    }


def _validate(record: NullAdapterRecord, geometry: _Geometry = PRODUCTION_GEOMETRY) -> None:
    """The one validator, shared by the writer, the serializer and the reader."""
    dtypes = _stored_dtypes(record.latent_dtype)
    shapes = geometry.shapes()
    if record.noise_convention not in NOISE_CONVENTIONS:
        raise ValueError(f"noise_convention must be one of {list(NOISE_CONVENTIONS)}, got {record.noise_convention!r}")
    if not _is_integral(record.ordinal):
        raise ValueError(f"ordinal must be an integer, got {record.ordinal!r}")
    for field in ARRAY_FIELDS:
        array = getattr(record, field)
        if not isinstance(array, np.ndarray) or array.dtype != dtypes[field]:
            raise ValueError(
                f"{field} must be a {dtypes[field].name} array, got {getattr(array, 'dtype', type(array))}"
            )
        if array.shape != shapes[field]:
            raise ValueError(f"{field} must have shape {shapes[field]}, got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field} must be finite: a cached artifact may not carry NaN or inf")
    if not np.isfinite(record.final_future_mse):
        raise ValueError(f"final_future_mse must be finite, got {record.final_future_mse}")
    # Recomputed here, not just on read: a record whose hash no longer describes its bytes must never
    # reach a shard, whether the hash was replaced or the array was mutated after make_record.
    if sha256_of_array(record.expected_final_latent) != record.expected_final_latent_sha256:
        raise ValueError("expected_final_latent sha256 mismatch: the stored bytes are not the ones that were hashed")


def _validate_header(header: ProvenanceHeader) -> np.ndarray:
    """Shared by ``header_to_json`` and ``header_from_json``; returns the canonical sigma vector."""
    if not _is_integral(header.schema_version):
        raise ValueError(f"header schema_version must be an integer, got {header.schema_version!r}")
    if int(header.schema_version) != SCHEMA_VERSION:
        raise ValueError(f"unsupported header schema_version {header.schema_version!r}, expected {SCHEMA_VERSION}")
    sigma = np.ascontiguousarray(np.asarray(header.sigma_vector).astype(np.dtype("<f4")))
    if sigma.ndim != 1 or sigma.shape[0] != PRODUCTION_SIGMA_LENGTH:
        raise ValueError(f"sigma_vector must be 1-D with {PRODUCTION_SIGMA_LENGTH} entries, got shape {sigma.shape}")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("sigma_vector must be finite")
    if not np.isfinite(header.guide_scale):
        raise ValueError(f"guide_scale must be finite, got {header.guide_scale}")
    if header.dtype_policy not in LATENT_DTYPES:
        raise ValueError(f"dtype_policy must be one of {sorted(LATENT_DTYPES)}, got {header.dtype_policy!r}")
    if not _is_integral(header.l_null):
        raise ValueError(f"l_null must be an integer, got {header.l_null!r}")
    if not isinstance(header.optimization_config, dict):
        raise ValueError(f"optimization_config must be a dict, got {type(header.optimization_config)}")
    for key, value in header.optimization_config.items():
        if not isinstance(key, str) or not (value is None or isinstance(value, (str, bool, int, float))):
            raise ValueError("optimization_config must be flat: values must be str, int, float, bool or None")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"optimization_config[{key!r}] must be finite, got {value}")
    return sigma


def _freeze(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _make_record(*, geometry: _Geometry, **fields: Any) -> NullAdapterRecord:
    """Cast every tensor to its storage dtype, hash the stored bytes, freeze, validate fail-closed."""
    dtypes = _stored_dtypes(fields["latent_dtype"])

    def cast(field: str) -> np.ndarray:
        return _freeze(np.ascontiguousarray(np.asarray(fields[field]).astype(dtypes[field])))

    stored_expected = cast("expected_final_latent")
    record = NullAdapterRecord(
        name=str(fields["name"]),
        ordinal=fields["ordinal"],
        split=str(fields["split"]),
        episode=str(fields["episode"]),
        z_i0=cast("z_i0"),
        actions=cast("actions"),
        z_video=cast("z_video"),
        latent_dtype=fields["latent_dtype"],
        nulls=cast("nulls"),
        z_start=cast("z_start"),
        expected_final_latent=stored_expected,
        expected_final_latent_sha256=sha256_of_array(stored_expected),
        noise_convention=fields["noise_convention"],
        arm=str(fields["arm"]),
        per_step_final_losses=cast("per_step_final_losses"),
        final_future_mse=float(fields["final_future_mse"]),
    )
    _validate(record, geometry)
    return record


def make_record(**fields: Any) -> NullAdapterRecord:
    """Build a record at the production geometry (plan §4-P2)."""
    return _make_record(geometry=PRODUCTION_GEOMETRY, **fields)


def _scalar_fields(record: NullAdapterRecord) -> dict[str, Any]:
    return {f.name: getattr(record, f.name) for f in dataclasses.fields(record) if f.name not in ARRAY_FIELDS}


def _zip_arrays(members: dict[str, np.ndarray]) -> bytes:
    """npz with a fixed timestamp, so identical content always serializes to identical bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, array in members.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.ascontiguousarray(array), allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH), payload.getvalue())
    return buffer.getvalue()


def _record_to_bytes(record: NullAdapterRecord, geometry: _Geometry) -> bytes:
    _validate(record, geometry)
    meta = {"schema_version": SCHEMA_VERSION, "shapes": {f: list(getattr(record, f).shape) for f in ARRAY_FIELDS}}
    meta.update(_scalar_fields(record))
    members = {f: getattr(record, f) for f in ARRAY_FIELDS}
    members["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    return _zip_arrays(members)


def record_to_bytes(record: NullAdapterRecord) -> bytes:
    """Serialize a production-geometry record; refuses anything the reader would refuse."""
    return _record_to_bytes(record, PRODUCTION_GEOMETRY)


def _check_namespace(actual, expected, what: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{what} do not match the schema: unexpected {sorted(set(actual) ^ set(expected))}")


def _record_from_bytes(blob: bytes, geometry: _Geometry) -> NullAdapterRecord:
    """Parse a record, refusing anything whose bytes do not match its own declared schema."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError(f"archive members are duplicated: {sorted({n for n in names if names.count(n) > 1})}")
    _check_namespace(names, [f"{f}.npy" for f in (*ARRAY_FIELDS, "__meta__")], "archive members")

    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    meta = json.loads(bytes(members.pop("__meta__")).decode("utf-8"))
    expected_meta = (
        *_META_KEYS,
        *(f.name for f in dataclasses.fields(NullAdapterRecord) if f.name not in ARRAY_FIELDS),
    )
    _check_namespace(meta, expected_meta, "record metadata keys")
    _check_namespace(meta["shapes"], ARRAY_FIELDS, "record shape keys")
    if meta["schema_version"] != SCHEMA_VERSION or not _is_integral(meta["schema_version"]):
        raise ValueError(f"unsupported record schema_version {meta['schema_version']!r}, expected {SCHEMA_VERSION}")
    dtypes = _stored_dtypes(meta["latent_dtype"])

    for field in ARRAY_FIELDS:
        array, shape = members[field], tuple(meta["shapes"][field])
        expected_bytes = int(np.prod(shape)) * dtypes[field].itemsize
        if array.dtype != dtypes[field] or array.shape != shape or array.nbytes != expected_bytes:
            raise ValueError(
                f"{field} is {array.nbytes} bytes of {array.dtype} {array.shape}, but latent_dtype "
                f"{meta['latent_dtype']!r} implies {expected_bytes} bytes of {dtypes[field]} {shape}"
            )
        _freeze(array)

    scalars = {k: v for k, v in meta.items() if k not in _META_KEYS}
    record = NullAdapterRecord(**scalars, **members)
    _validate(record, geometry)
    return record


def record_from_bytes(blob: bytes) -> NullAdapterRecord:
    """Parse a production-geometry record, failing closed on anything off-contract."""
    return _record_from_bytes(blob, PRODUCTION_GEOMETRY)


def header_to_json(header: ProvenanceHeader) -> str:
    """Deterministic JSON: sorted keys, sigma vector as exact float32 values."""
    sigma = _validate_header(header)
    payload = {**dataclasses.asdict(header), "sigma_vector": [float(v) for v in sigma]}
    return json.dumps(payload, sort_keys=True)


def header_from_json(text: str) -> ProvenanceHeader:
    """Parse a provenance header under exactly the writer's contract."""
    payload = json.loads(text)
    _check_namespace(payload, [f.name for f in dataclasses.fields(ProvenanceHeader)], "header fields")
    # Validated in exactly the form the writer validates, then re-seated with the canonical vector.
    header = ProvenanceHeader(**payload)
    return dataclasses.replace(header, sigma_vector=_validate_header(header))
