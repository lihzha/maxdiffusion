"""Shard binding, manifest publication and artifact validation for exp_04's J0 cohorts (R9).

Split out of ``build_null_adapter_manifests`` in the R9 strengthening: the selection rules are pure
and belong next to each other, while everything here touches a filesystem or judges an artifact
somebody else wrote.

**Bindings fail closed.** A binding is what a manifest row pins its source shard to, so it may never
be guessed. On ``gs://`` the identity is the object generation read via ``gsutil stat``. The parse reads the two
fields it needs and refuses anything that would make either of them untrustworthy -- unrecognized
informational lines (creation time, hashes, ETag) are ignored so a gsutil version change is not an
outage, but a non-zero exit, **any** non-empty stderr, a
reauthentication marker in either stream (standing issue #6 -- a reauth prompt can appear while the
command still exits 0 and prints plausible-looking text), a missing/duplicated/non-decimal
``Generation`` or ``Content-Length``. Locally the identity is a streamed content sha256, because
mtime and size do not identify bytes.

**Publication is transactional**, in the shape R8 settled on: an attempt-unique staging directory,
data published first, a completion marker last, and no overwrite of an output that already carries
one. A manifest referenced by a launched job can therefore never change under it.

**Loading is strict.** Duplicate JSON keys, loose types, non-canonical episode ids, cohort sizes,
cross-cohort name uniqueness, DEV/TEST and TRAINFIT/TRAIN-2000 episode disjointness, the ≤2 windows
per TRAIN episode rule, the prescribed hash/ordinal ordering, and every row's binding against the
header are all checked here -- this is the boundary where corrupt cohort evidence would otherwise
enter an experiment.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import subprocess
import uuid
from typing import Any, Callable, Mapping, Sequence

from maxdiffusion.build_null_adapter_manifests import (
    ABSENT_META,
    COHORT_SIZES,
    HEADER_FIELDS,
    MANIFEST_SCHEMA_VERSION,
    MAX_WINDOWS_PER_TRAIN_EPISODE,
    ROW_FIELDS,
    Manifests,
    episode_id_for,
    episode_sort_key,
    listing_checksum,
    reject_duplicate_keys,
)


HEADER_NAME = "header.json"
MARKER_NAME = "_COMPLETE.json"
STAGING_SUFFIX = ".staging"
REAUTH_MARKERS = ("reauth", "reauthentication", "does not have storage.objects.get access")
_DECIMAL = re.compile(r"\d+\Z")
_GCS_FIELDS = ("generation", "content-length")
_STREAM_CHUNK = 8 << 20


def _gfile():
    from tensorflow.io import gfile

    return gfile


def _refuse_reauth(path: str, stdout: str, stderr: str) -> None:
    """Standing issue #6: a reauth prompt can arrive with a zero exit code and plausible output."""
    if stderr.strip():
        raise RuntimeError(f"gsutil stat for {path} wrote to stderr; refusing to bind: {stderr.strip()!r}")
    haystack = f"{stdout}\n{stderr}".lower()
    for marker in REAUTH_MARKERS:
        if marker in haystack:
            raise RuntimeError(f"gsutil stat for {path} looks like a reauthentication prompt: {marker!r}")


def _stream_sha256(path: str) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with _gfile().GFile(path, "rb") as handle:
        while True:
            chunk = handle.read(_STREAM_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def shard_binding(
    path: str,
    *,
    runner: Callable[..., Any] | None = None,
    hasher: Callable[[str], tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """What a manifest row pins its source shard to: ``{kind, generation, size}``.

    ``gs://`` binds to the object's generation; anything else binds to a streamed content sha256 --
    exact bytes, at the cost of a full read, which is why production runs against GCS.
    """
    if not path.startswith("gs://"):
        hasher = hasher or _stream_sha256
        digest, size = hasher(path)
        return {"kind": "local", "generation": f"sha256:{digest}", "size": int(size)}

    runner = runner or subprocess.run
    completed = runner(["gsutil", "stat", path], capture_output=True, text=True, check=False)
    stdout, stderr = completed.stdout or "", getattr(completed, "stderr", "") or ""
    if getattr(completed, "returncode", 1) != 0:
        raise RuntimeError(f"gsutil stat failed for {path}: {stderr.strip()!r}")
    _refuse_reauth(path, stdout, stderr)

    found: dict[str, list[str]] = {key: [] for key in _GCS_FIELDS}
    for line in stdout.splitlines():
        key, separator, value = line.partition(":")
        key = key.strip().lower()
        if separator and key in found:
            found[key].append(value.strip())
    for key, values in found.items():
        if len(values) != 1:
            raise RuntimeError(f"gsutil stat for {path} carried {len(values)} {key!r} fields, expected exactly one")
        if not _DECIMAL.fullmatch(values[0]):
            raise RuntimeError(f"gsutil stat for {path} gave a non-decimal {key!r}: {values[0]!r}")
    return {"kind": "gcs", "generation": found["generation"][0], "size": int(found["content-length"][0])}


def _payloads(manifests: Manifests, builder_sha: str) -> dict[str, dict[str, Any]]:
    missing = sorted(set(manifests.shard_paths) - set(manifests.bindings))
    if missing:
        raise ValueError(f"every source shard needs a binding; missing {missing}")
    header = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder_sha": builder_sha,
        "shard_listing_checksum": listing_checksum(manifests.listings),
        "cohort_sizes": {cohort: len(rows) for cohort, rows in sorted(manifests.cohorts.items())},
        "shard_bindings": {path: dict(manifests.bindings[path]) for path in manifests.shard_paths},
        "listings": {split: list(paths) for split, paths in sorted(manifests.listings.items())},
    }
    payloads: dict[str, dict[str, Any]] = {HEADER_NAME: header}
    for cohort, rows in manifests.cohorts.items():
        payloads[f"{cohort}.json"] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "cohort": cohort,
            "rows": [
                {
                    **row,
                    "shard_generation": manifests.bindings[row["shard_path"]]["generation"],
                    "shard_size": manifests.bindings[row["shard_path"]]["size"],
                }
                for row in rows
            ],
        }
    return payloads


def write_manifests(manifests: Manifests, out_dir: str, *, builder_sha: str) -> dict[str, str]:
    """Publish the manifest set: stage under an attempt-unique path, then data, then the marker.

    The listing checksum is computed here from the ordered listings the scans consumed -- it is not a
    caller's argument, because a digest anyone can supply attests to nothing.
    """
    payloads = _payloads(manifests, builder_sha)
    gfile = _gfile()
    if gfile.exists(posixpath.join(out_dir, MARKER_NAME)):
        raise FileExistsError(f"{out_dir} is already published: manifests are immutable, build into a new directory")
    if gfile.exists(out_dir) and gfile.listdir(out_dir):
        raise FileExistsError(f"{out_dir} is not empty; publish a fresh manifest set into a new directory")

    staging = f"{out_dir.rstrip('/')}.{uuid.uuid4().hex[:12]}{STAGING_SUFFIX}"
    written: dict[str, str] = {}
    try:
        gfile.makedirs(staging)
        for filename, payload in payloads.items():
            with gfile.GFile(posixpath.join(staging, filename), "w") as handle:
                handle.write(json.dumps(payload, sort_keys=True, indent=2))
        with gfile.GFile(posixpath.join(staging, MARKER_NAME), "w") as handle:
            handle.write(
                json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION, "files": sorted(payloads)}, sort_keys=True)
            )
        gfile.makedirs(out_dir)
        for filename in [*payloads, MARKER_NAME]:  # data first, the completion marker last
            source, destination = posixpath.join(staging, filename), posixpath.join(out_dir, filename)
            if gfile.exists(destination):
                raise FileExistsError(f"refusing to overwrite a published manifest: {destination}")
            gfile.rename(source, destination, overwrite=False)
            written[filename] = destination
    finally:
        if gfile.exists(staging):
            gfile.rmtree(staging)
    return {name: path for name, path in written.items() if name != MARKER_NAME}


def _strict_json(text: str, what: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except ValueError as error:
        raise ValueError(f"{what}: not parseable as strict JSON ({error})") from error


def _exact_int(value: Any, what: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be >= {minimum}, got {value}")
    return value


def _canonical_episode(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.isdigit() or str(int(value)) != value:
        raise ValueError(f"{what} must be a canonical decimal episode id, got {value!r}")
    return value


def load_manifest(path: str) -> dict[str, Any]:
    """Load and validate one cohort file: strict JSON, exact types, exact size, unique names."""
    with _gfile().GFile(path, "r") as handle:
        payload = _strict_json(handle.read(), path)
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "cohort", "rows"}:
        raise ValueError(f"{path}: manifest fields do not match the schema")
    if _exact_int(payload["schema_version"], f"{path}: schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported manifest schema_version {payload['schema_version']!r}")
    cohort, rows = payload["cohort"], payload["rows"]
    if cohort not in COHORT_SIZES:
        raise ValueError(f"{path}: unknown cohort {cohort!r}")
    if not isinstance(rows, list) or len(rows) != COHORT_SIZES[cohort]:
        raise ValueError(f"{path}: {cohort} must carry exactly {COHORT_SIZES[cohort]} rows, got {len(rows)}")

    names, episodes = [], []
    for index, row in enumerate(rows):
        where = f"{path}: row {index}"
        if not isinstance(row, Mapping) or set(row) != set(ROW_FIELDS):
            raise ValueError(f"{where}: row fields do not match {list(ROW_FIELDS)}")
        if row["split"] != cohort:
            raise ValueError(f"{where}: split {row['split']!r} does not match the cohort {cohort!r}")
        if not isinstance(row["name"], str) or not row["name"]:
            raise ValueError(f"{where}: name must be a non-empty string, got {row['name']!r}")
        if not isinstance(row["shard_path"], str) or not row["shard_path"]:
            raise ValueError(f"{where}: shard_path must be a non-empty string, got {row['shard_path']!r}")
        if not isinstance(row["shard_generation"], str) or not row["shard_generation"]:
            raise ValueError(f"{where}: shard_generation must be a non-empty string, got {row['shard_generation']!r}")
        _canonical_episode(row["episode"], f"{where}: episode")
        _exact_int(row["ordinal"], f"{where}: ordinal", minimum=0)
        _exact_int(row["shard_size"], f"{where}: shard_size", minimum=0)
        # The name is an identity source in its own right (the producer's window grammar), so a row
        # is only coherent if its stored episode is the one its own name spells out.
        try:
            embedded = episode_id_for(row["name"], ABSENT_META)
        except ValueError as error:
            raise ValueError(f"{where}: name is not a producer window name ({error})") from error
        if embedded != row["episode"]:
            raise ValueError(f"{where}: the name says episode {embedded} but the row says {row['episode']}")
        names.append(row["name"])
        episodes.append(row["episode"])
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate names {duplicates}")
    # One window per episode everywhere except TRAIN-2000, which takes up to two (plan §4-J0/P2).
    # Without this, a cohort of 64 rows could describe 63 episodes and still look complete.
    limit = MAX_WINDOWS_PER_TRAIN_EPISODE if cohort == "train2000" else 1
    over = sorted({episode for episode in episodes if episodes.count(episode) > limit})
    if over:
        raise ValueError(f"{path}: {cohort} takes at most {limit} windows/episode, but these repeat: {over}")
    return payload


def _check_ordering(cohort: str, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [(episode_sort_key(row["episode"]), row["ordinal"]) for row in rows]
    if keys != sorted(keys):
        raise ValueError(f"{cohort}: rows are not in the prescribed hash-then-ordinal order")


def _check_header(header: Any, cohorts: Mapping[str, Mapping[str, Any]], out_dir: str) -> None:
    if not isinstance(header, Mapping) or set(header) != set(HEADER_FIELDS):
        raise ValueError(f"{out_dir}: header fields do not match {list(HEADER_FIELDS)}")
    if _exact_int(header["schema_version"], f"{out_dir}: header schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{out_dir}: unsupported header schema_version {header['schema_version']!r}")
    if header["cohort_sizes"] != COHORT_SIZES:
        raise ValueError(f"{out_dir}: header cohort_sizes {header['cohort_sizes']!r} are not {COHORT_SIZES}")
    listings = header["listings"]
    if not isinstance(listings, Mapping) or not all(
        isinstance(paths, list) and all(isinstance(path, str) for path in paths) for paths in listings.values()
    ):
        raise ValueError(f"{out_dir}: header listings must map a split to an ordered list of paths")
    if header["shard_listing_checksum"] != listing_checksum(
        {split: tuple(paths) for split, paths in listings.items()}
    ):
        raise ValueError(f"{out_dir}: the header's shard_listing_checksum does not match its own listings")

    bindings = header["shard_bindings"]
    if not isinstance(bindings, Mapping):
        raise ValueError(f"{out_dir}: header shard_bindings must be a mapping")
    for cohort, payload in cohorts.items():
        for row in payload["rows"]:
            binding = bindings.get(row["shard_path"])
            if not isinstance(binding, Mapping):
                raise ValueError(f"{cohort}: {row['shard_path']} has no binding in the header")
            if (row["shard_generation"], row["shard_size"]) != (binding.get("generation"), binding.get("size")):
                raise ValueError(f"{cohort}: {row['name']} disagrees with the header binding for its shard")


def load_manifests(out_dir: str) -> dict[str, Any]:
    """Load every cohort plus the header, and check every cross-cohort invariant the gates rely on."""
    gfile = _gfile()
    if not gfile.exists(posixpath.join(out_dir, MARKER_NAME)):
        raise ValueError(f"{out_dir}: no completion marker; this manifest set was never published")
    loaded = {cohort: load_manifest(posixpath.join(out_dir, f"{cohort}.json")) for cohort in COHORT_SIZES}

    names = [row["name"] for payload in loaded.values() for row in payload["rows"]]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{out_dir}: a window appears in more than one cohort: {duplicates}")
    for left, right in (("dev64", "test64"), ("trainfit16", "train2000")):
        shared = sorted(
            {row["episode"] for row in loaded[left]["rows"]} & {row["episode"] for row in loaded[right]["rows"]}
        )
        if shared:
            raise ValueError(f"{out_dir}: {left} and {right} must be episode-disjoint, but share {shared}")
    per_episode: dict[str, int] = {}
    for row in loaded["train2000"]["rows"]:
        per_episode[row["episode"]] = per_episode.get(row["episode"], 0) + 1
    over = sorted(episode for episode, count in per_episode.items() if count > MAX_WINDOWS_PER_TRAIN_EPISODE)
    if over:
        raise ValueError(
            f"{out_dir}: TRAIN-2000 takes at most {MAX_WINDOWS_PER_TRAIN_EPISODE} windows/episode: {over}"
        )
    for cohort, payload in loaded.items():
        _check_ordering(cohort, payload["rows"])

    with gfile.GFile(posixpath.join(out_dir, HEADER_NAME), "r") as handle:
        header = _strict_json(handle.read(), posixpath.join(out_dir, HEADER_NAME))
    _check_header(header, loaded, out_dir)
    return {"header": header, **loaded}
