"""Shard publication and validated resume for the exp_04 P2 cache (plan §4-P2; R8 + strengthening).

A cache job spans hours and preemptible hosts, so the only trustworthy answer to "which examples are
already cached?" is one derived from the artifacts. This module publishes shards so that question is
answerable, and answers it. The judgments about *what* to cache -- the fidelity gate and the
quarantine policy -- live in ``null_adapter_cache_policy``; here there is only storage.

**Layout — one file per record, plus a header and a marker.**::

    <shard>/header.json          the shared provenance header (R4b JSON), always present
    <shard>/record_00000.npz     one R4b record blob per example, in sorted-name order
    <shard>/_COMPLETE.json       the marker -- published LAST

One file per record rather than a tar: each record is already a self-contained, byte-deterministic
zip from the R4b codec, per-record files make a partial shard diagnosable, and object stores have no
partial-tar semantics worth relying on. **The name-to-file mapping is a canonical bijection**,
``sorted(names)[i] -> record_{i:05d}.npz``, recomputed on read and never taken from the marker: a
marker that could name its own files could point at a staging leftover from a different shard and
still validate, which is precisely how marker-last stops meaning anything (Codex R8 review,
finding 1).

**Publication order is the contract.** Everything is staged first, then published data-first with the
marker **last**: the marker's existence is the publish signal. ``tf.io.gfile.rename`` is a real
rename locally and a copy-then-delete on ``gs://``, i.e. per-file and *not* atomic across the shard --
which is why the order matters rather than the mechanism.

**Shards are immutable and single-writer.** Staging paths carry a per-attempt token so two writers
cannot delete each other's work; publication never overwrites (``overwrite=False``), so a racing
writer fails instead of corrupting; and a shard that already has a marker is never republished. An
*incomplete* directory (no marker) is not a boundary and may be swept, but only through
``discard_incomplete_shard``, so discarding is an explicit, auditable act rather than a side effect
of retrying (finding 6; ratification 4's conditions).

**Reading is total.** Every parse and read failure becomes an invalid ``ShardReport``; validation
never raises on artifact bytes, however hostile (finding 3). What it *does* raise on is a caller
error -- a missing expected fingerprint, arm or convention, which are required arguments because a
resume that does not check provenance is not a resume (finding 2).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import posixpath
import re
import uuid
from typing import Any, Mapping, Sequence

from maxdiffusion.null_adapter_records import (
    NOISE_CONVENTIONS,
    NullAdapterRecord,
    ProvenanceHeader,
    header_from_json,
    header_to_json,
    record_from_bytes,
    record_to_bytes,
)


SHARD_SCHEMA_VERSION = 1
MARKER_NAME = "_COMPLETE.json"
HEADER_NAME = "header.json"
RECORD_TEMPLATE = "record_{:05d}.npz"
STAGING_SUFFIX = ".staging"
REQUIRED_OPTIMIZATION_KEYS = ("inner_iters", "lr")  # exactly these -- the R6 carried contract
MAX_SHARD_BYTES = 8 * 2**30  # a declared ceiling; the runner owns the free-space floor (SOP)
_RECORD_FILENAME = re.compile(r"record_\d{5}\.npz\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclasses.dataclass(frozen=True)
class ShardMarker:
    """What a published shard claims about itself; its presence is the publish signal."""

    schema_version: int
    count: int
    names: tuple[str, ...]
    files: dict[str, str]
    sha256: dict[str, str]
    header_fingerprint: str
    quarantined: dict[str, str]

    def to_json(self) -> str:
        payload = {**dataclasses.asdict(self), "names": list(self.names)}
        return json.dumps(payload, sort_keys=True, allow_nan=False)


@dataclasses.dataclass(frozen=True)
class ShardReport:
    """Whether a shard may be trusted, and why not. Invalid is a value, not an exception: a resume
    walks over broken shards rather than dying on them."""

    path: str
    valid: bool
    names: tuple[str, ...]
    quarantined: dict[str, str]
    reasons: tuple[str, ...]
    header_fingerprint: str | None = None


@dataclasses.dataclass(frozen=True)
class ResumePlan:
    todo: tuple[str, ...]
    covered: tuple[str, ...]
    quarantined: dict[str, str]
    shards: tuple[ShardReport, ...]


def canonical_files(names: Sequence[str]) -> dict[str, str]:
    """The only legal name-to-file mapping: sorted position decides the filename."""
    return {name: RECORD_TEMPLATE.format(position) for position, name in enumerate(sorted(names))}


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate JSON keys would silently collapse: {duplicates}")
    return dict(pairs)


def _exact_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"marker {key} must be an integer, got {value!r}")
    return value


def _string_map(payload: Mapping[str, Any], key: str) -> dict[str, str]:
    value = payload[key]
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError(f"marker {key} must be a mapping of non-empty strings, got {value!r}")
    return dict(value)


def marker_from_json(text: str) -> ShardMarker:
    """Parse a marker under exactly the writer's contract; anything else is not a marker.

    Total and strict: duplicate JSON keys, wrong types (``True`` is not a version), negative counts,
    unsorted or duplicated names, non-canonical filenames, malformed digests and a name that is both
    published and quarantined are all refused here rather than downstream.
    """
    payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    fields = {field.name for field in dataclasses.fields(ShardMarker)}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError(
            f"marker fields do not match the schema: {sorted(payload) if isinstance(payload, dict) else payload!r}"
        )
    if _exact_int(payload, "schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError(f"unsupported marker schema_version {payload['schema_version']!r}")

    names = payload["names"]
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"marker names must be a list of non-empty strings, got {names!r}")
    if list(names) != sorted(set(names)):
        raise ValueError(f"marker names must be sorted and unique, got {names!r}")
    count = _exact_int(payload, "count")
    if count < 0 or count != len(names):
        raise ValueError(f"marker count {count} does not match its {len(names)} names")

    files, digests = _string_map(payload, "files"), _string_map(payload, "sha256")
    if files != canonical_files(names):
        raise ValueError("marker files are not the canonical sorted-position mapping")
    if set(digests) != set(names) or not all(_HEX64.fullmatch(digest) for digest in digests.values()):
        raise ValueError("marker sha256 must carry one 64-hex digest per name")
    fingerprint = payload["header_fingerprint"]
    if not isinstance(fingerprint, str) or not _HEX64.fullmatch(fingerprint):
        raise ValueError(f"marker header_fingerprint must be a 64-hex digest, got {fingerprint!r}")
    quarantined = _string_map(payload, "quarantined")
    if set(quarantined) & set(names):
        raise ValueError(f"a name cannot be both published and quarantined: {sorted(set(quarantined) & set(names))}")
    if not names and not quarantined:
        raise ValueError("a marker must publish records or record a quarantine")
    return ShardMarker(SHARD_SCHEMA_VERSION, count, tuple(names), files, digests, fingerprint, quarantined)


def _gfile():
    """Lazy: the unit logic (marker parsing) must import without TensorFlow."""
    from tensorflow.io import gfile

    return gfile


def _write_bytes(path: str, payload: bytes) -> None:
    gfile = _gfile()
    gfile.makedirs(posixpath.dirname(path))
    with gfile.GFile(path, "wb") as handle:
        handle.write(payload)


def _read_bytes(path: str) -> bytes:
    with _gfile().GFile(path, "rb") as handle:
        return handle.read()


def _publish(source: str, destination: str) -> None:
    """Move one staged artifact into the shard, refusing to overwrite anything already there.

    On ``gs://`` this is a copy-then-delete, so it is the *order* of these calls -- never their
    atomicity -- that makes the marker a valid publish signal; ``overwrite=False`` is what makes a
    second writer fail instead of quietly replacing the first one's bytes.
    """
    gfile = _gfile()
    gfile.makedirs(posixpath.dirname(destination))
    if gfile.exists(destination):
        raise FileExistsError(f"refusing to overwrite a published artifact: {destination}")
    gfile.rename(source, destination, overwrite=False)


def _shard_member(shard_path: str, filename: str) -> str:
    """Resolve one member of a shard, refusing anything that could reach outside it.

    The containment check is redundant *given* the filename pattern -- no string matching
    ``record_\\d{5}\\.npz`` can carry a separator or a dot segment -- and is kept anyway: this is the
    guard the reviewer's path-escape probe walked through, and defence in depth is cheap here. Its
    redundancy is why the mutation battery cannot kill it independently; that is recorded rather
    than hidden.
    """
    if not isinstance(filename, str) or not _RECORD_FILENAME.fullmatch(filename):
        raise ValueError(f"{filename!r} is not a canonical record filename")
    root = posixpath.normpath(shard_path)
    resolved = posixpath.normpath(posixpath.join(shard_path, filename))
    if posixpath.dirname(resolved) != root:
        raise ValueError(f"{filename!r} resolves outside the shard directory")
    return posixpath.join(shard_path, filename)


def _staging_path(shard_path: str, staging_prefix: str) -> str:
    """A staging directory this attempt owns alone: two writers must never share one."""
    token = uuid.uuid4().hex[:12]
    return posixpath.join(
        staging_prefix.rstrip("/"), f"{posixpath.basename(shard_path.rstrip('/'))}.{token}{STAGING_SUFFIX}"
    )


def _header_fingerprint(header: ProvenanceHeader) -> str:
    return hashlib.sha256(header_to_json(header).encode("utf-8")).hexdigest()


def _validate_write(records: Sequence[NullAdapterRecord], header: ProvenanceHeader, quarantined: Mapping[str, str]):
    if not records and not quarantined:
        raise ValueError("a shard must carry at least one record or one quarantined name")
    names = [record.name for record in records]
    if len(set(names)) != len(names):
        raise ValueError(f"record names within a shard must be unique, got {sorted(names)}")
    overlap = sorted(set(names) & set(quarantined))
    if overlap:
        raise ValueError(f"names cannot be both written and quarantined: {overlap}")
    if not all(isinstance(name, str) and name and isinstance(reason, str) for name, reason in quarantined.items()):
        raise ValueError("quarantined must map non-empty names to reason strings")
    if sorted(header.optimization_config) != sorted(REQUIRED_OPTIMIZATION_KEYS):
        raise ValueError(
            f"header optimization_config must declare exactly {list(REQUIRED_OPTIMIZATION_KEYS)}, "
            f"got {sorted(header.optimization_config)}"
        )
    for record in records:
        if record.latent_dtype != header.dtype_policy:
            raise ValueError(
                f"{record.name}: record latent_dtype {record.latent_dtype!r} disagrees with the "
                f"header's dtype_policy {header.dtype_policy!r}"
            )
        if (record.arm, record.noise_convention) != (records[0].arm, records[0].noise_convention):
            raise ValueError(
                f"{record.name}: a shard is one arm and one convention, but this record is "
                f"{record.arm!r}/{record.noise_convention!r} beside {records[0].arm!r}/"
                f"{records[0].noise_convention!r}"
            )


def _refuse_existing_destination(shard_path: str) -> None:
    gfile = _gfile()
    if gfile.exists(posixpath.join(shard_path, MARKER_NAME)):
        raise FileExistsError(f"{shard_path} is already published: a completed shard is never rewritten")
    if gfile.exists(shard_path) and gfile.listdir(shard_path):
        raise FileExistsError(
            f"{shard_path} holds an incomplete shard; call discard_incomplete_shard first so the "
            f"discard is an explicit act rather than a side effect of retrying"
        )


def discard_incomplete_shard(shard_path: str) -> bool:
    """Remove a shard directory that was never published. Refuses one that has a marker."""
    gfile = _gfile()
    if gfile.exists(posixpath.join(shard_path, MARKER_NAME)):
        raise FileExistsError(f"{shard_path} carries a completion marker and will not be discarded")
    if not gfile.exists(shard_path):
        return False
    gfile.rmtree(shard_path)
    return True


def write_shard(
    records: Sequence[NullAdapterRecord],
    header: ProvenanceHeader,
    shard_path: str,
    staging_prefix: str,
    *,
    quarantined: Mapping[str, str] | None = None,
) -> ShardMarker:
    """Publish one shard: stage each record in turn, move the data in, then the marker.

    Args:
      records: the examples to publish, all from one arm/convention and matching ``header``'s dtype
        policy. They go through the R4b public codec, which is where geometry and dtype are enforced.
      header: the shard's one provenance header -- carried even by a records-free diagnostic shard,
        so that shard can still be validated against the run's expected fingerprint (R8 ruling 3).
      shard_path: the published shard directory (local or ``gs://``); must not already exist.
      staging_prefix: where this attempt's staging directory is created, under a unique token.
      quarantined: ``name -> reason`` for examples this attempt tried and lost; recorded so the gap
        is visible, and never counted as covered.

    Returns:
      The ``ShardMarker`` as published.
    """
    records, quarantined = list(records), dict(quarantined or {})
    _validate_write(records, header, quarantined)
    _refuse_existing_destination(shard_path)

    files = canonical_files([record.name for record in records])
    staging = _staging_path(shard_path, staging_prefix)
    by_name = {record.name: record for record in records}
    digests: dict[str, str] = {}
    published_bytes = 0
    try:
        # One record resident at a time: the blob is serialized, written and dropped before the next
        # one is built, so peak memory is one record rather than the whole shard.
        for name in sorted(by_name):
            blob = record_to_bytes(by_name[name])
            published_bytes += len(blob)
            if published_bytes > MAX_SHARD_BYTES:
                raise ValueError(f"shard exceeds MAX_SHARD_BYTES ({published_bytes} > {MAX_SHARD_BYTES})")
            _write_bytes(posixpath.join(staging, files[name]), blob)
            digests[name] = hashlib.sha256(blob).hexdigest()
            del blob
        _write_bytes(posixpath.join(staging, HEADER_NAME), header_to_json(header).encode("utf-8"))
        marker = ShardMarker(
            schema_version=SHARD_SCHEMA_VERSION,
            count=len(records),
            names=tuple(sorted(by_name)),
            files=files,
            sha256=digests,
            header_fingerprint=_header_fingerprint(header),
            quarantined=quarantined,
        )
        _write_bytes(posixpath.join(staging, MARKER_NAME), marker.to_json().encode("utf-8"))

        # Data first, marker last: a reader that sees the marker sees everything it names.
        _publish(posixpath.join(staging, HEADER_NAME), posixpath.join(shard_path, HEADER_NAME))
        for name in marker.names:
            _publish(posixpath.join(staging, files[name]), _shard_member(shard_path, files[name]))
        _publish(posixpath.join(staging, MARKER_NAME), posixpath.join(shard_path, MARKER_NAME))
        return marker
    finally:
        gfile = _gfile()
        if gfile.exists(staging):
            gfile.rmtree(staging)


def _validate_expectations(fingerprint: str, arm: str, convention: str) -> None:
    if not isinstance(fingerprint, str) or not _HEX64.fullmatch(fingerprint):
        raise ValueError(f"expected_header_fingerprint must be a 64-hex digest, got {fingerprint!r}")
    if not isinstance(arm, str) or not arm:
        raise ValueError(f"expected_arm must be a non-empty string, got {arm!r}")
    if convention not in NOISE_CONVENTIONS:
        raise ValueError(f"expected_noise_convention must be one of {list(NOISE_CONVENTIONS)}, got {convention!r}")


def _validate_shard(shard_path: str, fingerprint: str, arm: str, convention: str) -> ShardReport:
    gfile = _gfile()
    marker_path = posixpath.join(shard_path, MARKER_NAME)
    if not gfile.exists(marker_path):
        return ShardReport(shard_path, False, (), {}, ("no completion marker: the shard was never published",))
    marker = marker_from_json(_read_bytes(marker_path).decode("utf-8"))

    reasons: list[str] = []
    header = None
    header_path = posixpath.join(shard_path, HEADER_NAME)
    if not gfile.exists(header_path):
        reasons.append("header.json is missing")
    else:
        header = header_from_json(_read_bytes(header_path).decode("utf-8"))
        if _header_fingerprint(header) != marker.header_fingerprint:
            reasons.append("header.json does not match the marker's header fingerprint")
    if marker.header_fingerprint != fingerprint:
        reasons.append("marker header fingerprint is not the expected one")

    for name in marker.names:
        path = _shard_member(shard_path, marker.files[name])
        if not gfile.exists(path):
            reasons.append(f"{name}: record file is missing")
            continue
        blob = _read_bytes(path)
        if hashlib.sha256(blob).hexdigest() != marker.sha256[name]:
            reasons.append(f"{name}: record sha256 does not match the marker")
            continue
        record = record_from_bytes(blob)
        if record.name != name:
            reasons.append(f"{name}: record file holds a different example")
        if header is not None and record.latent_dtype != header.dtype_policy:
            reasons.append(f"{name}: record latent_dtype disagrees with the header's dtype_policy")
        if (record.arm, record.noise_convention) != (arm, convention):
            reasons.append(f"{name}: record is {record.arm}/{record.noise_convention}, expected {arm}/{convention}")

    valid = not reasons
    return ShardReport(
        path=shard_path,
        valid=valid,
        names=marker.names if valid else (),
        quarantined=dict(marker.quarantined) if valid else {},
        reasons=tuple(reasons),
        header_fingerprint=marker.header_fingerprint,
    )


def validate_shard(
    shard_path: str, *, expected_header_fingerprint: str, expected_arm: str, expected_noise_convention: str
) -> ShardReport:
    """Decide whether a shard may be trusted, reading every byte it claims to have published.

    The three expectations are required, not optional: a resume that does not check which run wrote a
    shard is not a resume (R8 review, finding 2). They are validated eagerly -- a malformed
    *expectation* is a caller bug and raises -- while anything wrong with the *artifact*, down to
    hostile JSON, comes back as an invalid report.
    """
    _validate_expectations(expected_header_fingerprint, expected_arm, expected_noise_convention)
    try:
        return _validate_shard(shard_path, expected_header_fingerprint, expected_arm, expected_noise_convention)
    except Exception as error:  # noqa: BLE001 -- totality: no artifact may make validation raise
        return ShardReport(shard_path, False, (), {}, (f"shard could not be validated: {error!r}",))


def resume_plan(
    manifest: Sequence[str],
    shard_paths: Sequence[str],
    *,
    expected_header_fingerprint: str,
    expected_arm: str,
    expected_noise_convention: str,
) -> ResumePlan:
    """Exactly the manifest names no validated shard of *this run* has published.

    Quarantined names are not covered: they come back in ``todo`` and are listed in ``quarantined``,
    so a caller either re-attempts them or accepts the gap in writing. Every name a validated shard
    mentions -- published or quarantined -- must belong to the manifest and must be mentioned only
    once across shards; either violation means the wrong cohort is being resumed.
    """
    manifest = tuple(manifest)
    if len(set(manifest)) != len(manifest):
        raise ValueError("manifest names must be unique")

    reports = tuple(
        validate_shard(
            path,
            expected_header_fingerprint=expected_header_fingerprint,
            expected_arm=expected_arm,
            expected_noise_convention=expected_noise_convention,
        )
        for path in shard_paths
    )
    seen: dict[str, str] = {}
    covered: set[str] = set()
    quarantined: dict[str, str] = {}
    for report in reports:
        for name in (*report.names, *report.quarantined):
            if name in seen:
                raise ValueError(f"{name!r} appears in more than one validated shard: {seen[name]} and {report.path}")
            seen[name] = report.path
        covered.update(report.names)
        quarantined.update(report.quarantined)

    strangers = sorted(set(seen) - set(manifest))
    if strangers:
        raise ValueError(f"validated shards mention names outside the manifest: {strangers}")
    return ResumePlan(
        todo=tuple(name for name in manifest if name not in covered),
        covered=tuple(sorted(covered)),
        quarantined=quarantined,
        shards=reports,
    )
