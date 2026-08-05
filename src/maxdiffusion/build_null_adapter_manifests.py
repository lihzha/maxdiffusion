"""J0 cohort-manifest builder for exp_04 (plan §4 "J0 — Cohort manifests"; R9 + strengthening).

The manifests decide which examples every later claim is about: the gates check coverage against
them, the noise convention keys off their names, and P2 caches exactly what they list. So this module
is deterministic and fail-closed, and the selection *rules* are pure so they can be pinned without
TFRecords. Binding, publication and artifact validation live in ``null_adapter_manifest_io``.

**Where an episode id comes from.** A window's identity is fixed upstream, and both sources are used:

- The **name**, minted by ``third_party/Wan2.2/scripts/make_droid_window_plan.py:162`` as
  ``f"ep{ep}_v{args.view}_s{start:05d}"`` -- real examples from this project's own DROID cache:
  ``ep0_v0_s00000`` (``docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json``) and
  ``ep30738_v0_s00132`` (``…/overfit100_results.md``). The cache lays these out as
  ``<out_root>/<split>/<name>`` (``precompute_features_droid_plan.py:117``) and the TFRecord producer
  stores the supplied name byte for byte (``wan_side_adapter_droid_cache_to_tfrecord.py``
  ``_make_example``, line 180), so a stored name *may* carry a ``<split>/`` prefix; everything before
  the last ``/`` is ignored.
- ``meta_json``, copied verbatim from the cache's ``meta.json``, which the cache builder writes with
  an explicit ``"episode_id": int(ep)`` (``precompute_features_droid_plan.py:156``). A cache
  directory without that file is represented by the producer as exactly ``b"{}"``.

Both are parsed and must agree. **Present-but-malformed metadata is a refusal, not an absence**: a
missing second source has one legitimate spelling (``{}``), so unparseable bytes, a non-object, or
duplicate keys mean the record is corrupt and cannot be identified -- otherwise a wrong-but-plausible
name would quietly become the truth. The canonical episode id is the decimal string of a
non-negative integer, and **that string is the sha256 preimage that orders the episodes**.

**The J0 contract is executable.** ``build_j0_manifests`` pins the plan's numbers as constants --
VAL's exact record count, TRAIN's 5,000-episode target, the 200-shard and 60 GiB caps -- lists shards
deterministically, scans, checks every target, selects, and only then publishes, all or nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import posixpath
import re
from typing import Any, Callable, Iterable, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
EPISODE_NAME = re.compile(r"ep(\d+)_v(\d+)_s(\d+)\Z")
# The producer's substitute for a cache directory with no meta.json, and the only spelling of
# "this record has no second identity source" that is accepted anywhere.
ABSENT_META = b"{}"
COHORT_SIZES = {"dev64": 64, "test64": 64, "trainfit16": 16, "train2000": 2000}
DEV_EPISODES, TEST_EPISODES, TRAINFIT_EPISODES = 64, 64, 16
TRAIN_WINDOW_TARGET = 2000
MAX_WINDOWS_PER_TRAIN_EPISODE = 2
ROW_FIELDS = ("split", "name", "episode", "ordinal", "shard_path", "shard_generation", "shard_size")
HEADER_FIELDS = (
    "schema_version",
    "builder_sha",
    "shard_listing_checksum",
    "cohort_sizes",
    "shard_bindings",
    "listings",
)
# Plan §4-J0's numbers, pinned here rather than passed in: they are the contract, not a parameter.
VAL_EXPECTED_RECORDS = 14_636
TRAIN_EPISODE_TARGET = 5_000
MAX_SCAN_SHARDS = 200
MAX_SCAN_BYTES = 60 * 2**30
LISTING_DOMAIN = b"exp04-j0-shard-listing-v1\0"


class CapExceeded(RuntimeError):
    """A bounded scan hit its shard or byte cap before reaching its episode target."""


@dataclasses.dataclass(frozen=True)
class Window:
    name: str
    episode: str
    ordinal: int
    shard_path: str


@dataclasses.dataclass(frozen=True)
class ScanResult:
    split: str
    listing: tuple[str, ...]
    windows: tuple[Window, ...]
    bindings: dict[str, dict[str, Any]]
    shards_opened: tuple[str, ...]
    episodes: int
    stopped_early: bool


@dataclasses.dataclass(frozen=True)
class Manifests:
    cohorts: dict[str, tuple[dict[str, Any], ...]]
    shard_paths: tuple[str, ...]
    listings: dict[str, tuple[str, ...]]
    bindings: dict[str, dict[str, Any]]


def reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict:
    """JSON object hook: duplicate keys are a corrupt document, not a last-one-wins convenience."""
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate JSON keys would silently collapse: {duplicates}")
    return dict(pairs)


def _episode_from_meta(name: str, payload: str) -> str | None:
    try:
        meta = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except ValueError as error:
        raise ValueError(f"{name}: meta_json is present but is not parseable JSON ({error})") from error
    if not isinstance(meta, Mapping):
        raise ValueError(f"{name}: meta_json must be a JSON object, got {type(meta).__name__}")
    if "episode_id" not in meta:
        return None
    value = meta["episode_id"]
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name}: meta_json episode_id must be an integer or digits, got {value!r}")
    if isinstance(value, str) and not value.isdigit():
        raise ValueError(f"{name}: meta_json episode_id must be an integer or digits, got {value!r}")
    if int(value) < 0:
        raise ValueError(f"{name}: meta_json episode_id must not be negative, got {value!r}")
    return str(int(value))


def _meta_payload(name: str, meta_json: Any) -> str | None:
    """The metadata text, or ``None`` for a legitimately absent second source.

    "Absent" has exactly one spelling -- ``b"{}"``, which is what the producer substitutes when a
    cache directory has no ``meta.json``. Empty bytes, whitespace and invalid UTF-8 are not that
    marker: they are corruption, and treating corruption as absence is what would let a
    wrong-but-plausible name become the record's identity unchallenged.
    """
    if isinstance(meta_json, str):
        raw = meta_json.encode("utf-8")
    elif isinstance(meta_json, (bytes, bytearray)):
        raw = bytes(meta_json)
    else:
        raise ValueError(f"{name}: meta_json must be the producer's bytes field, got {type(meta_json).__name__}")
    if raw == ABSENT_META:
        return None
    try:
        return raw.decode("utf-8")  # strict: no errors="replace" -- corrupt bytes are not text
    except UnicodeDecodeError as error:
        raise ValueError(f"{name}: meta_json is not valid UTF-8 ({error})") from error


def episode_id_for(name: str, meta_json: bytes | str) -> str:
    """The canonical episode id of one record: decimal digits, agreed by both identity sources."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"record name must be a non-empty string, got {name!r}")
    match = EPISODE_NAME.fullmatch(name.rsplit("/", 1)[-1])
    from_name = str(int(match.group(1))) if match else None

    payload = _meta_payload(name, meta_json)
    from_meta = _episode_from_meta(name, payload) if payload is not None else None

    if from_name is not None and from_meta is not None and from_name != from_meta:
        raise ValueError(f"{name}: the name says episode {from_name} but meta_json says {from_meta}")
    episode = from_name or from_meta
    if episode is None:
        raise ValueError(
            f"{name}: no episode id -- the name does not match ep<id>_v<view>_s<start> and meta_json "
            f"carries no episode_id"
        )
    return episode


def episode_sort_key(episode: str) -> str:
    """Ascending hex sha256 of the canonical decimal episode id (plan §4-J0's ordering)."""
    return hashlib.sha256(episode.encode("utf-8")).hexdigest()


def listing_checksum(listings: Mapping[str, Sequence[str]]) -> str:
    """Digest of the *ordered* shard listings the scans consumed.

    Order is part of the evidence: an early-stopped scan selects different episodes if the listing is
    permuted, so a checksum over a sorted set (or any caller-supplied string) would attest to nothing.
    Domain-separated, and each entry carries its split and its ordinal position.

    The position field is belt-and-braces: sha256 already streams the entries in order, and the NUL
    separators make the preimage unambiguous, so no two listings collide without it. It is kept
    because the digest is a provenance record a human will read a year from now, and its redundancy
    is why the mutation battery cannot kill it -- recorded rather than hidden.
    """
    digest = hashlib.sha256(LISTING_DOMAIN)
    for split in sorted(listings):
        for position, path in enumerate(listings[split]):
            digest.update(f"{split}\0{position}\0{path}\0".encode("utf-8"))
    return digest.hexdigest()


def _tfrecord_reader(shard_path: str) -> Iterable[tuple[str, int, bytes]]:
    """Default reader: only the three identity fields, so a scan never materializes the latents."""
    import tensorflow as tf

    features = {
        "name": tf.io.FixedLenFeature([], tf.string),
        "ordinal": tf.io.FixedLenFeature([], tf.int64),
        "meta_json": tf.io.FixedLenFeature([], tf.string, default_value=b"{}"),
    }
    for raw in tf.data.TFRecordDataset([shard_path]):
        parsed = tf.io.parse_single_example(raw, features)
        yield (
            parsed["name"].numpy().decode("utf-8"),
            int(parsed["ordinal"].numpy()),
            bytes(parsed["meta_json"].numpy()),
        )


def scan_split(
    shard_paths: Sequence[str],
    *,
    split: str = "",
    early_stop_episodes: int | None = None,
    max_shards: int | None = None,
    max_bytes: int | None = None,
    reader: Callable[[str], Iterable[tuple[str, int, bytes]]] | None = None,
    binder: Callable[[str], dict[str, Any]] | None = None,
) -> ScanResult:
    """Stream shards in the given order, extracting ``(name, episode, ordinal, shard_path)``.

    ``early_stop_episodes`` stops **the moment** the target distinct-episode count is reached, mid
    shard: an overshoot episode would enter the hash ordering and silently move the cohort boundaries.
    The shard-count cap is checked before the next shard is even statted, and its size against the
    byte cap before it is opened; either raises ``CapExceeded`` so the builder writes nothing.

    Each shard is bound (stat/identity) **before and after** it is read and the two must agree: a
    shard replaced between the stat that authorized it and the read that consumed it would otherwise
    be scanned under someone else's provenance.
    """
    reader = reader or _tfrecord_reader
    if binder is None:
        from maxdiffusion.null_adapter_manifest_io import shard_binding as binder  # lazy: keeps IO out
    windows: list[Window] = []
    bindings: dict[str, dict[str, Any]] = {}
    opened: list[str] = []
    episodes: set[str] = set()
    total_bytes = 0
    stopped_early = False

    for path in shard_paths:
        if early_stop_episodes is not None and len(episodes) >= early_stop_episodes:
            stopped_early = True
            break
        if max_shards is not None and len(opened) + 1 > max_shards:
            raise CapExceeded(
                f"shard cap {max_shards} reached after {len(opened)} shards with {len(episodes)} episodes"
            )
        before = binder(path)
        size = int(before["size"])
        if max_bytes is not None and total_bytes + size > max_bytes:
            raise CapExceeded(f"byte cap {max_bytes} would be exceeded by {path} ({total_bytes + size} bytes)")
        opened.append(path)
        bindings[path] = before
        total_bytes += size
        for name, ordinal, meta_json in reader(path):
            episode = episode_id_for(name, meta_json)
            windows.append(Window(name=name, episode=episode, ordinal=int(ordinal), shard_path=path))
            episodes.add(episode)
            if early_stop_episodes is not None and len(episodes) >= early_stop_episodes:
                stopped_early = True
                break
        after = binder(path)
        if (after.get("generation"), after.get("size")) != (before.get("generation"), before.get("size")):
            raise ValueError(f"{path} changed while it was being scanned: {before} then {after}")
        if stopped_early:
            break

    return ScanResult(
        split=split,
        listing=tuple(shard_paths),
        windows=tuple(windows),
        bindings=bindings,
        shards_opened=tuple(opened),
        episodes=len(episodes),
        stopped_early=stopped_early,
    )


def _group_by_episode(scan: ScanResult) -> tuple[list[str], dict[str, list[Window]]]:
    groups: dict[str, list[Window]] = {}
    seen: set[str] = set()
    for window in scan.windows:
        if window.name in seen:
            raise ValueError(f"{window.name!r} appears more than once in the {scan.split or 'scanned'} split")
        seen.add(window.name)
        groups.setdefault(window.episode, []).append(window)
    for windows in groups.values():
        windows.sort(key=lambda window: (window.ordinal, window.name))
    return sorted(groups, key=episode_sort_key), groups


def _row(split: str, window: Window) -> dict[str, Any]:
    return {
        "split": split,
        "name": window.name,
        "episode": window.episode,
        "ordinal": window.ordinal,
        "shard_path": window.shard_path,
    }


def select_cohorts(val_scan: ScanResult, train_scan: ScanResult) -> Manifests:
    """The plan's §4-J0 selection, exactly: pure, deterministic, fail-closed.

    Episodes are ordered by ascending hex sha256 of their canonical id. VAL's first 64 episodes are
    DEV-64 and the *next* 64 are TEST-64 -- episode-disjoint by construction, one lowest-ordinal
    window each. TRAIN's first 16 episodes are TRAINFIT-16; TRAIN-2000 then walks the episodes after
    those, taking ``min(2, available)`` lowest-ordinal windows each until exactly 2,000 windows are
    collected, and fails if the scanned pool runs out first.
    """
    val_order, val_groups = _group_by_episode(val_scan)
    train_order, train_groups = _group_by_episode(train_scan)
    needed = DEV_EPISODES + TEST_EPISODES
    if len(val_order) < needed:
        raise ValueError(f"the val scan holds {len(val_order)} episodes, fewer than the {needed} DEV+TEST need")
    if len(train_order) < TRAINFIT_EPISODES:
        raise ValueError(f"the train scan holds {len(train_order)} episodes, fewer than TRAINFIT-16 needs")

    cohorts: dict[str, tuple[dict[str, Any], ...]] = {
        "dev64": tuple(_row("dev64", val_groups[ep][0]) for ep in val_order[:DEV_EPISODES]),
        "test64": tuple(_row("test64", val_groups[ep][0]) for ep in val_order[DEV_EPISODES:needed]),
        "trainfit16": tuple(_row("trainfit16", train_groups[ep][0]) for ep in train_order[:TRAINFIT_EPISODES]),
    }

    train_rows: list[dict[str, Any]] = []
    for episode in train_order[TRAINFIT_EPISODES:]:
        if len(train_rows) >= TRAIN_WINDOW_TARGET:
            break
        available = train_groups[episode]
        # ``len(available)`` restates plan P2's "min(2, available)" literally. It is redundant with the
        # slice below -- which already yields fewer elements for a one-window episode -- and is kept
        # because a future refactor that indexes instead of slicing would otherwise silently break the
        # one-window rule (a documented equivalent mutant; R9 review ruling: KEEP).
        take = min(MAX_WINDOWS_PER_TRAIN_EPISODE, len(available), TRAIN_WINDOW_TARGET - len(train_rows))
        train_rows.extend(_row("train2000", window) for window in available[:take])
    if len(train_rows) != TRAIN_WINDOW_TARGET:
        raise ValueError(
            f"TRAIN-2000 underfilled: the scanned pool yielded {len(train_rows)} windows from "
            f"{max(len(train_order) - TRAINFIT_EPISODES, 0)} episodes, short of {TRAIN_WINDOW_TARGET}"
        )
    cohorts["train2000"] = tuple(train_rows)

    names = [row["name"] for rows in cohorts.values() for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"a window may appear in only one cohort row, but these repeat: {duplicates}")
    # No cohort-size assertion here: the pool checks above and the exact TRAIN-2000 count make the
    # sizes true by construction. Sizes are re-checked where they *can* be wrong -- on load.
    rows = [row for cohort_rows in cohorts.values() for row in cohort_rows]
    return Manifests(
        cohorts=cohorts,
        shard_paths=tuple(sorted({row["shard_path"] for row in rows})),
        listings={"val": val_scan.listing, "train": train_scan.listing},
        bindings={**val_scan.bindings, **train_scan.bindings},
    )


def _listed(shard_dir: str, lister: Callable[[str], Sequence[str]] | None) -> tuple[str, ...]:
    """Deterministic listing: sorted, absolute, non-empty -- the scan order is part of the evidence."""
    if lister is None:
        from tensorflow.io import gfile

        lister = lambda pattern: gfile.glob(pattern)  # noqa: E731
    paths = tuple(sorted(lister(posixpath.join(shard_dir.rstrip("/"), "*.tfrecord*"))))
    if not paths:
        raise ValueError(f"no shards found under {shard_dir}")
    return paths


def build_j0_manifests(
    val_shard_dir: str,
    train_shard_dir: str,
    out_dir: str,
    *,
    builder_sha: str,
    lister: Callable[[str], Sequence[str]] | None = None,
    reader: Callable[[str], Iterable[tuple[str, int, bytes]]] | None = None,
    binder: Callable[[str], dict[str, Any]] | None = None,
    publisher: Callable[..., dict[str, str]] | None = None,
) -> dict[str, Any]:
    """The J0 job: list, scan, check every plan target, select, and publish -- all or nothing.

    The plan's numbers are constants here, not arguments: VAL is scanned in full and must hold exactly
    ``VAL_EXPECTED_RECORDS`` records; TRAIN is scanned with the 5,000-episode early stop under the
    200-shard and 60 GiB caps, and a listing that runs out below that target is a failure, not a
    smaller cohort. Nothing is written until all of it holds.
    """
    val_listing = _listed(val_shard_dir, lister)
    train_listing = _listed(train_shard_dir, lister)

    val_scan = scan_split(val_listing, split="val", reader=reader, binder=binder)
    if len(val_scan.windows) != VAL_EXPECTED_RECORDS:
        raise ValueError(
            f"the val split holds {len(val_scan.windows)} records, not the expected {VAL_EXPECTED_RECORDS}: "
            f"the cohort definition assumes the published val split, so this is refused"
        )
    train_scan = scan_split(
        train_listing,
        split="train",
        early_stop_episodes=TRAIN_EPISODE_TARGET,
        max_shards=MAX_SCAN_SHARDS,
        max_bytes=MAX_SCAN_BYTES,
        reader=reader,
        binder=binder,
    )
    if train_scan.episodes < TRAIN_EPISODE_TARGET:
        raise ValueError(
            f"the train listing was exhausted at {train_scan.episodes} distinct episodes, short of the "
            f"{TRAIN_EPISODE_TARGET} target: nothing is written"
        )

    manifests = select_cohorts(val_scan, train_scan)
    if publisher is None:
        from maxdiffusion.null_adapter_manifest_io import write_manifests as publisher
    written = publisher(manifests, out_dir, builder_sha=builder_sha)
    return {
        "written": written,
        "val_records": len(val_scan.windows),
        "val_shards": len(val_scan.shards_opened),
        "train_episodes": train_scan.episodes,
        "train_shards": len(train_scan.shards_opened),
        "shard_listing_checksum": listing_checksum(manifests.listings),
    }
