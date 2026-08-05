"""exp_04 R9 — the J0 cohort builder: the manifests every later claim is about.

A silent change here would not break anything visibly; it would just make every downstream number
describe a different experiment. So the rules are pinned hard, and after the R9 strengthening the
*production contract* is pinned too:

- **The J0 entry point owns the plan's numbers.** VAL's exact record count, TRAIN's 5,000-episode
  target, the 200-shard and 60 GiB caps are constants, and every one of them can fail the build
  before a byte is written -- proved with a writer that raises if it is ever called.
- **Early stop is record-granular.** The scan stops on the record that reaches the target, mid shard;
  an overshoot episode would enter the hash ordering and move the cohort boundaries. Reader yields
  and binder calls are counted exactly.
- **Order is evidence.** The listing checksum is computed from the ordered listings the scans
  consumed, never supplied, and a reversed listing is a different digest.
- **Provenance fails closed.** The reviewer's two ``gsutil`` poison probes -- a clean-looking stdout
  with a reauth line on stderr, and ``Generation: Reauthentication required`` -- are regressions here.
- **Malformed metadata is a refusal, not an absence**: ``b"{}"`` is how a missing second source is
  spelled, so ``b"not json"`` next to a plausible name must not quietly become the truth.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from maxdiffusion.build_null_adapter_manifests import (
    COHORT_SIZES,
    DEV_EPISODES,
    MANIFEST_SCHEMA_VERSION,
    MAX_SCAN_BYTES,
    MAX_SCAN_SHARDS,
    TRAIN_EPISODE_TARGET,
    TRAIN_WINDOW_TARGET,
    VAL_EXPECTED_RECORDS,
    CapExceeded,
    Manifests,
    ScanResult,
    Window,
    build_j0_manifests,
    episode_id_for,
    episode_sort_key,
    listing_checksum,
    scan_split,
    select_cohorts,
)
from maxdiffusion.null_adapter_manifest_io import (
    HEADER_NAME,
    MARKER_NAME,
    load_manifest,
    load_manifests,
    shard_binding,
    write_manifests,
)


_TRAIN_EPISODE_OFFSET = 100_000  # val and train episodes are disjoint namespaces in DROID
_SHARD = "gs://bucket/val/shard-00000.tfrecord"
_LISTED = (_SHARD, "gs://bucket/val/shard-00001.tfrecord")  # listed, not necessarily opened
_BINDING = {"kind": "gcs", "generation": "17543557234567", "size": 1024}


def _name(episode, view=0, start=0):
    return f"ep{episode}_v{view}_s{start:05d}"


def _meta(episode, **extra):
    return json.dumps({"source": "droid_ctrl_world", "episode_id": int(episode), **extra}).encode("utf-8")


def _window(episode, ordinal=0, start=None, shard=_SHARD):
    start = ordinal if start is None else start
    return Window(name=_name(episode, start=start), episode=str(episode), ordinal=ordinal, shard_path=shard)


def _scan(windows, split="val"):
    return ScanResult(
        split=split,
        listing=_LISTED,
        windows=tuple(windows),
        bindings={_SHARD: dict(_BINDING)},
        shards_opened=(_SHARD,),
        episodes=len({w.episode for w in windows}),
        stopped_early=False,
    )


def _val_scan(episodes=range(200)):
    return _scan([_window(episode) for episode in episodes], split="val")


def _train_scan(episodes=range(2000), windows_per_episode=2):
    return _scan(
        [
            _window(_TRAIN_EPISODE_OFFSET + episode, ordinal=index, start=index * 4)
            for episode in episodes
            for index in range(windows_per_episode)
        ],
        split="train",
    )


# ---------------------------------------------------------------- identity


def test_the_producer_format_is_parsed_from_both_identity_sources():
    """Real names from this project's DROID cache (exp_02's manifest and results)."""
    assert episode_id_for("ep0_v0_s00000", _meta(0)) == "0"
    assert episode_id_for("ep30738_v0_s00132", _meta(30738)) == "30738"
    assert episode_id_for("ep4358_v0_s00040", b"{}") == "4358"  # the one legitimate "absent" spelling
    assert episode_id_for("train/ep4015_v0_s00000", b"{}") == "4015"  # a split-prefixed name
    assert episode_id_for("ep7_v1_s00004", b'{"source": "droid_ctrl_world"}') == "7"  # valid meta, no id
    assert episode_id_for("ep7_v1_s00004", b"{ }") == "7"  # valid JSON, no episode_id: name-only is fine
    assert episode_id_for("windows/00042", _meta(42)) == "42"  # unparseable name: meta carries it
    assert episode_id_for("ep007_v0_s00000", b"{}") == "7"  # canonical decimal, not the raw digits


@pytest.mark.parametrize(
    "name, meta, message",
    [
        ("ep7_v1_s00004", b"not json", "not parseable JSON"),  # the reviewer's regression
        ("ep7_v1_s00004", b"", "not parseable JSON"),  # empty bytes are not the absent-marker
        ("ep7_v1_s00004", b"   ", "not parseable JSON"),
        ("ep7_v1_s00004", b'{"episode_i\xff":8}', "not valid UTF-8"),  # the reviewer's probe
        ("ep7_v1_s00004", None, "must be the producer's bytes field"),
        ("ep7_v1_s00004", b"[1, 2]", "must be a JSON object"),
        ("ep7_v1_s00004", b'{"episode_id": 7, "episode_id": 8}', "duplicate JSON keys"),
        ("ep1_v0_s00000", _meta(2), "the name says episode 1 but meta_json says 2"),
        ("windows/00042", b"{}", "no episode id"),
        ("windows/00042", json.dumps({"episode_id": -3}).encode(), "must not be negative"),
        ("windows/00042", json.dumps({"episode_id": "abc"}).encode(), "must be an integer or digits"),
        ("windows/00042", json.dumps({"episode_id": True}).encode(), "must be an integer or digits"),
        ("", b"{}", "non-empty string"),
    ],
)
def test_an_unidentifiable_record_is_refused(name, meta, message):
    with pytest.raises(ValueError, match=message):
        episode_id_for(name, meta)


def test_episodes_are_ordered_by_the_hex_sha256_of_their_decimal_id():
    """Hand-computed digests, independently re-derived by the R9 reviewer."""
    assert episode_sort_key("0") == "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9"
    assert episode_sort_key("7") == "7902699be42c8a8e46fbbb4501726517e86b22c56a189f7625a6da49081b2451"
    assert episode_sort_key("30738") == "b51d6398dac7426234ec52dbe930937892cec74d244ea263a0b86705b06a8ed4"

    manifests = select_cohorts(_val_scan(range(128)), _train_scan(range(1016)))
    ordered = [row["episode"] for row in manifests.cohorts["dev64"]]
    assert ordered == sorted(ordered, key=lambda e: hashlib.sha256(e.encode("utf-8")).hexdigest())


# ---------------------------------------------------------------- selection


def test_the_cohorts_are_the_plan_s_slices_and_are_episode_disjoint():
    manifests = select_cohorts(_val_scan(range(200)), _train_scan(range(1016)))

    assert {cohort: len(rows) for cohort, rows in manifests.cohorts.items()} == COHORT_SIZES
    dev = [row["episode"] for row in manifests.cohorts["dev64"]]
    test = [row["episode"] for row in manifests.cohorts["test64"]]
    val_order = sorted({str(e) for e in range(200)}, key=episode_sort_key)
    assert dev == val_order[:DEV_EPISODES]
    assert test == val_order[DEV_EPISODES : 2 * DEV_EPISODES]
    assert not set(dev) & set(test)
    trainfit = {row["episode"] for row in manifests.cohorts["trainfit16"]}
    assert not trainfit & {row["episode"] for row in manifests.cohorts["train2000"]}


def test_the_selection_is_deterministic():
    first = select_cohorts(_val_scan(range(200)), _train_scan(range(1016)))
    second = select_cohorts(_val_scan(range(200)), _train_scan(range(1016)))

    assert first == second


def test_each_episode_contributes_its_lowest_ordinal_window():
    windows = [
        _window(5, ordinal=7, start=28),
        _window(5, ordinal=2, start=8),  # the lowest ordinal, listed second
        *[_window(episode) for episode in range(128) if episode != 5],
    ]
    manifests = select_cohorts(_scan(windows, "val"), _train_scan(range(1016)))

    rows = {row["episode"]: row for cohort in ("dev64", "test64") for row in manifests.cohorts[cohort]}
    assert rows["5"]["ordinal"] == 2 and rows["5"]["name"] == _name(5, start=8)


def test_train2000_takes_two_windows_per_episode_and_stops_at_exactly_two_thousand():
    manifests = select_cohorts(_val_scan(range(200)), _train_scan(range(1200)))

    counts: dict[str, int] = {}
    for row in manifests.cohorts["train2000"]:
        counts[row["episode"]] = counts.get(row["episode"], 0) + 1
    assert sum(counts.values()) == TRAIN_WINDOW_TARGET and set(counts.values()) <= {1, 2}


def test_one_window_episodes_contribute_what_they_have():
    """Plan P2's rule is min(2, available): an episode with a single window is not skipped."""
    train = []
    for episode in range(_TRAIN_EPISODE_OFFSET, _TRAIN_EPISODE_OFFSET + 1400):
        train.append(_window(episode, ordinal=0, start=0))
        if episode % 2 == 0:
            train.append(_window(episode, ordinal=1, start=4))

    manifests = select_cohorts(_val_scan(range(200)), _scan(train, "train"))

    counts: dict[str, int] = {}
    for row in manifests.cohorts["train2000"]:
        counts[row["episode"]] = counts.get(row["episode"], 0) + 1
    assert sum(counts.values()) == TRAIN_WINDOW_TARGET and 1 in set(counts.values())


def test_an_underfilled_train_pool_fails_closed():
    with pytest.raises(ValueError, match="TRAIN-2000 underfilled"):
        select_cohorts(_val_scan(range(200)), _train_scan(range(100)))


@pytest.mark.parametrize(
    "val, train, message",
    [(range(127), range(1016), "fewer than the 128"), (range(200), range(10), "fewer than TRAINFIT-16")],
)
def test_a_pool_too_small_for_a_cohort_fails_closed(val, train, message):
    with pytest.raises(ValueError, match=message):
        select_cohorts(_val_scan(val), _train_scan(train))


def test_a_duplicate_name_in_a_scan_is_a_hard_error():
    windows = [_window(1), _window(1), *[_window(e) for e in range(2, 200)]]

    with pytest.raises(ValueError, match="appears more than once"):
        select_cohorts(_scan(windows, "val"), _train_scan(range(1016)))


def test_a_name_shared_between_the_splits_is_a_hard_error():
    shared = _window(7)
    val = _scan([shared, *[_window(e) for e in range(1, 200) if e != 7]], "val")
    train = _scan(
        [
            shared,
            *[_window(_TRAIN_EPISODE_OFFSET + e, ordinal=i, start=i * 4) for e in range(1016) for i in range(2)],
        ],
        "train",
    )

    with pytest.raises(ValueError, match="may appear in only one cohort row"):
        select_cohorts(val, train)


# ---------------------------------------------------------------- scanning


def _fake_shards(count=4, per_shard=3, sizes=None, records_per_shard=None):
    """Shards plus a reader, a binder, and logs of what each of them was asked for."""
    paths = [f"gs://bucket/train/shard-{index:05d}.tfrecord" for index in range(count)]
    episodes = {}
    index = 0
    for path in paths:
        span = per_shard if records_per_shard is None else records_per_shard
        episodes[path] = list(range(index, index + span))
        index += span
    yielded: list[str] = []
    bound: list[str] = []

    def reader(path):
        for episode in episodes[path]:
            yielded.append(f"{path}:{episode}")
            yield _name(episode), 0, _meta(episode)

    sizes = sizes or dict.fromkeys(paths, 1024)

    def binder(path):
        bound.append(path)
        return {"kind": "gcs", "generation": "1", "size": sizes[path]}

    return paths, reader, binder, yielded, bound


def test_a_scan_extracts_identity_and_binds_every_shard():
    paths, reader, binder, yielded, bound = _fake_shards()

    result = scan_split(paths, split="val", reader=reader, binder=binder)

    assert result.episodes == 12 and len(result.windows) == 12
    assert result.shards_opened == tuple(paths) and result.listing == tuple(paths)
    assert bound == [path for path in paths for _ in range(2)]  # bound before AND after each read
    assert result.windows[0] == Window(name=_name(0), episode="0", ordinal=0, shard_path=paths[0])
    assert result.bindings[paths[0]]["size"] == 1024


def test_the_scan_stops_on_the_record_that_reaches_the_target():
    """The reviewer's probe: one shard, ten episodes, target six -- the overshoot must never be read."""
    paths, reader, binder, yielded, bound = _fake_shards(count=1, records_per_shard=10)

    result = scan_split(paths, early_stop_episodes=6, reader=reader, binder=binder)

    assert result.episodes == 6 and len(result.windows) == 6
    assert len(yielded) == 6  # the reader was never asked for records seven through ten
    assert [window.episode for window in result.windows] == [str(e) for e in range(6)]
    assert result.stopped_early


def test_early_stop_never_opens_or_stats_the_next_shard():
    paths, reader, binder, yielded, bound = _fake_shards(count=5, per_shard=3)

    result = scan_split(paths, early_stop_episodes=6, reader=reader, binder=binder)

    assert result.shards_opened == tuple(paths[:2])
    assert set(bound) == set(paths[:2])  # shard three was never even statted
    assert len(yielded) == 6 and result.stopped_early


@pytest.mark.parametrize(
    "caps, message, expected_binds",
    [
        ({"max_shards": 2}, "shard cap 2", 2),  # the third shard is not statted at all
        ({"max_bytes": 2048}, "byte cap 2048", 3),  # statted, then refused before opening
    ],
)
def test_a_cap_reached_before_the_target_fails_closed(caps, message, expected_binds):
    paths, reader, binder, yielded, bound = _fake_shards(count=5, per_shard=3)

    with pytest.raises(CapExceeded, match=message):
        scan_split(paths, early_stop_episodes=99, reader=reader, binder=binder, **caps)

    assert len(set(bound)) == expected_binds and len(yielded) == 6


def test_a_shard_replaced_while_it_is_being_scanned_is_refused():
    """The stat that authorized the shard and the read that consumed it must be the same object."""
    paths, reader, _, _, _ = _fake_shards(count=1, per_shard=3)
    generations = iter(["1", "2"])

    def shifting_binder(path):
        return {"kind": "gcs", "generation": next(generations), "size": 1024}

    with pytest.raises(ValueError, match="changed while it was being scanned"):
        scan_split(paths, reader=reader, binder=shifting_binder)


def test_a_scan_that_meets_its_target_within_the_caps_succeeds():
    paths, reader, binder, yielded, bound = _fake_shards(count=5, per_shard=3)

    result = scan_split(paths, early_stop_episodes=6, max_shards=3, max_bytes=4096, reader=reader, binder=binder)

    assert result.episodes == 6 and result.shards_opened == tuple(paths[:2])


# ---------------------------------------------------------------- listing checksum


def test_the_listing_checksum_binds_order_and_split():
    forward = {"val": ("a", "b"), "train": ("c", "d")}

    assert listing_checksum(forward) == listing_checksum({"train": ("c", "d"), "val": ("a", "b")})
    assert listing_checksum(forward) != listing_checksum({"val": ("b", "a"), "train": ("c", "d")})
    assert listing_checksum(forward) != listing_checksum({"val": ("c", "d"), "train": ("a", "b")})
    assert listing_checksum(forward) != listing_checksum({"val": ("a", "b", "c", "d"), "train": ()})
    # Without the per-split domain marker these two stream identical bytes, so the split really is
    # part of the preimage and not merely implied by the iteration order.
    assert listing_checksum({"val": (), "train": ("a", "b")}) != listing_checksum({"val": ("a", "b"), "train": ()})


# ---------------------------------------------------------------- bindings


class _Completed:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


_GSUTIL_STAT = """gs://bucket/shard-00000.tfrecord:
    Creation time:          Tue, 05 Aug 2026 01:02:03 GMT
    Content-Length:         3145728
    Content-Type:           application/octet-stream
    Generation:             1754355723456789
    Metageneration:         1
"""


def test_a_gcs_shard_binds_to_its_generation():
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return _Completed(stdout=_GSUTIL_STAT)

    binding = shard_binding("gs://bucket/shard-00000.tfrecord", runner=runner)

    assert binding == {"kind": "gcs", "generation": "1754355723456789", "size": 3145728}
    assert calls == [["gsutil", "stat", "gs://bucket/shard-00000.tfrecord"]]


@pytest.mark.parametrize(
    "completed, message",
    [
        # The reviewer's two poison probes, verbatim (standing issue #6).
        (_Completed(stdout=_GSUTIL_STAT, stderr="ReauthUnattendedError: Reauthentication required"), "stderr"),
        (
            _Completed(stdout=_GSUTIL_STAT.replace("1754355723456789", "Reauthentication required")),
            "reauthentication prompt",
        ),
        (_Completed(returncode=1, stderr="no such object"), "gsutil stat failed"),
        (_Completed(stdout="gs://bucket/x:\n  Content-Length: 5\n"), "0 'generation'"),
        (_Completed(stdout=_GSUTIL_STAT + "    Generation:  99\n"), "2 'generation'"),
        (_Completed(stdout=_GSUTIL_STAT.replace("3145728", "-5")), "non-decimal"),
        (_Completed(stdout="reauthentication required\n" + _GSUTIL_STAT), "reauthentication prompt"),
    ],
)
def test_a_gcs_binding_that_cannot_be_trusted_is_refused(completed, message):
    with pytest.raises(RuntimeError, match=message):
        shard_binding("gs://bucket/x", runner=lambda argv, **kwargs: completed)


def test_a_local_shard_binds_to_a_streamed_content_hash(tmp_path):
    path = tmp_path / "shard-00000.tfrecord"
    path.write_bytes(b"0123456789")

    binding = shard_binding(str(path))

    assert binding == {
        "kind": "local",
        "generation": f"sha256:{hashlib.sha256(b'0123456789').hexdigest()}",
        "size": 10,
    }
    path.write_bytes(b"9876543210")  # same size, same mtime granularity, different bytes
    assert shard_binding(str(path))["generation"] != binding["generation"]


# ---------------------------------------------------------------- publication and loading


def _built(tmp_path, out="manifests", **overrides):
    manifests = select_cohorts(_val_scan(range(200)), _train_scan(range(1016)))
    written = write_manifests(manifests, str(tmp_path / out), builder_sha=overrides.get("builder_sha", "a" * 40))
    return manifests, written


def test_manifests_round_trip_through_disk(tmp_path):
    manifests, written = _built(tmp_path)

    loaded = load_manifests(str(tmp_path / "manifests"))

    assert set(written) == {HEADER_NAME, *(f"{cohort}.json" for cohort in COHORT_SIZES)}
    assert loaded["header"]["builder_sha"] == "a" * 40
    assert loaded["header"]["shard_listing_checksum"] == listing_checksum(manifests.listings)
    assert loaded["header"]["listings"] == {split: list(paths) for split, paths in manifests.listings.items()}
    assert loaded["header"]["cohort_sizes"] == COHORT_SIZES
    for cohort, size in COHORT_SIZES.items():
        rows = loaded[cohort]["rows"]
        assert len(rows) == size
        assert [row["name"] for row in rows] == [row["name"] for row in manifests.cohorts[cohort]]
        assert all(row["shard_generation"] == _BINDING["generation"] for row in rows)


def test_the_written_json_is_deterministic_and_sorted(tmp_path):
    _built(tmp_path, out="first")
    _built(tmp_path, out="second")

    for filename in (HEADER_NAME, "dev64.json"):
        first = (tmp_path / "first" / filename).read_text()
        second = (tmp_path / "second" / filename).read_text()
        assert first == second
        keys = list(json.loads(first).keys())
        assert keys == sorted(keys), (filename, keys)


def test_publication_is_staged_and_marked(tmp_path):
    _built(tmp_path)

    assert os.path.exists(str(tmp_path / "manifests" / MARKER_NAME))
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".staging")]


def test_an_unmarked_directory_is_not_a_published_manifest_set(tmp_path):
    _built(tmp_path)
    os.remove(str(tmp_path / "manifests" / MARKER_NAME))

    with pytest.raises(ValueError, match="no completion marker"):
        load_manifests(str(tmp_path / "manifests"))


def test_writing_refuses_to_overwrite_a_published_manifest_set(tmp_path):
    manifests, _ = _built(tmp_path)

    with pytest.raises(FileExistsError, match="already published"):
        write_manifests(manifests, str(tmp_path / "manifests"), builder_sha="c" * 40)
    assert json.loads((tmp_path / "manifests" / HEADER_NAME).read_text())["builder_sha"] == "a" * 40


def test_writing_refuses_a_non_empty_output_directory(tmp_path):
    manifests = select_cohorts(_val_scan(range(200)), _train_scan(range(1016)))
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "stray.json").write_text("{}")

    with pytest.raises(FileExistsError, match="not empty"):
        write_manifests(manifests, str(tmp_path / "manifests"), builder_sha="a" * 40)


def test_writing_refuses_a_shard_without_a_binding(tmp_path):
    manifests = select_cohorts(_val_scan(range(200)), _train_scan(range(1016)))
    stripped = Manifests(manifests.cohorts, manifests.shard_paths, manifests.listings, {})

    with pytest.raises(ValueError, match="needs a binding"):
        write_manifests(stripped, str(tmp_path / "manifests"), builder_sha="a" * 40)
    assert not os.path.exists(str(tmp_path / "manifests" / HEADER_NAME))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p["rows"].pop(), "exactly 64 rows"),
        (lambda p: p["rows"].__setitem__(1, p["rows"][0]), "duplicate names"),
        (lambda p: p.__setitem__("schema_version", True), "must be an integer"),
        (lambda p: p.__setitem__("schema_version", 99), "schema_version"),
        (lambda p: p.__setitem__("cohort", "nope"), "unknown cohort"),
        (lambda p: p["rows"][0].pop("shard_generation"), "row fields"),
        (lambda p: p["rows"][0].__setitem__("split", "test64"), "does not match the cohort"),
        (lambda p: p["rows"][0].__setitem__("ordinal", "3"), "ordinal must be an integer"),
        (lambda p: p["rows"][0].__setitem__("ordinal", True), "ordinal must be an integer"),
        (lambda p: p["rows"][0].__setitem__("shard_size", -1), "shard_size must be >= 0"),
        (lambda p: p["rows"][0].__setitem__("shard_generation", {"g": 1}), "shard_generation must be"),
        (lambda p: p["rows"][0].__setitem__("episode", "007"), "canonical decimal episode id"),
        (lambda p: p["rows"][0].__setitem__("episode", 7), "canonical decimal episode id"),
        (lambda p: p["rows"][0].__setitem__("name", ""), "name must be a non-empty string"),
    ],
)
def test_loading_validates_one_cohort_file(tmp_path, mutate, message):
    _built(tmp_path)
    path = tmp_path / "manifests" / "dev64.json"
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_manifest(str(path))


@pytest.mark.parametrize("cohort", ["dev64", "test64", "trainfit16"])
def test_a_single_window_cohort_must_describe_as_many_episodes_as_it_has_rows(tmp_path, cohort):
    """The reviewer's case: row 2 reusing row 1's episode leaves 63 distinct episodes in a 64-row
    cohort -- complete-looking evidence over a silently smaller pool (plan §4-J0: one window each)."""
    _built(tmp_path)
    path = tmp_path / "manifests" / f"{cohort}.json"
    payload = json.loads(path.read_text())
    stolen = payload["rows"][1]["episode"]
    payload["rows"][2] = {
        **payload["rows"][2],
        "episode": stolen,
        "name": _name(int(stolen), start=98),  # a different window, so only the episode repeats
    }
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="at most 1 windows/episode"):
        load_manifest(str(path))


def test_a_row_whose_name_is_not_a_producer_window_name_is_refused(tmp_path):
    """The corrupted-window-name case: the name is the row's other identity source, so it has to be
    one -- a truncated or mangled name cannot be paired with anything."""
    _built(tmp_path)
    path = tmp_path / "manifests" / "dev64.json"
    payload = json.loads(path.read_text())
    payload["rows"][0]["name"] = payload["rows"][0]["name"].replace("_v", "-v")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="name is not a producer window name"):
        load_manifest(str(path))


def test_a_row_whose_name_disagrees_with_its_episode_is_refused(tmp_path):
    """Syntactically perfect, and about a different episode than the row claims."""
    _built(tmp_path)
    path = tmp_path / "manifests" / "dev64.json"
    payload = json.loads(path.read_text())
    row = payload["rows"][0]
    payload["rows"][0] = {**row, "name": _name(int(row["episode"]) + 500_000, start=0)}
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="the name says episode .* but the row says"):
        load_manifest(str(path))


def test_loading_refuses_duplicate_json_keys(tmp_path):
    _built(tmp_path)
    path = tmp_path / "manifests" / "dev64.json"
    path.write_text(path.read_text().replace('"cohort":', '"cohort": "dev64", "cohort":', 1))

    with pytest.raises(ValueError, match="duplicate JSON keys"):
        load_manifest(str(path))


def _corrupt(tmp_path, filename, mutate):
    path = tmp_path / "manifests" / filename
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload))


def test_loading_rejects_episode_overlap_between_paired_cohorts(tmp_path):
    for left, right in (("dev64", "test64"), ("trainfit16", "train2000")):
        target = tmp_path / f"{left}_{right}"
        _built(tmp_path, out=target.name)
        stolen = json.loads((target / f"{left}.json").read_text())["rows"][0]["episode"]
        path = target / f"{right}.json"
        payload = json.loads(path.read_text())
        payload["rows"][0]["episode"] = stolen
        payload["rows"][0]["name"] = _name(int(stolen), start=97)  # a *different* window of that episode
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="episode-disjoint|hash-then-ordinal"):
            load_manifests(str(target))


def test_loading_rejects_rows_out_of_the_prescribed_order(tmp_path):
    _built(tmp_path)

    def swap(payload):
        payload["rows"][0], payload["rows"][1] = payload["rows"][1], payload["rows"][0]

    _corrupt(tmp_path, "dev64.json", swap)

    with pytest.raises(ValueError, match="hash-then-ordinal order"):
        load_manifests(str(tmp_path / "manifests"))


@pytest.mark.parametrize(
    "filename, mutate, message",
    [
        (HEADER_NAME, lambda p: p.__setitem__("cohort_sizes", {"dev64": 64}), "cohort_sizes"),
        (HEADER_NAME, lambda p: p.pop("listings"), "header fields"),
        (HEADER_NAME, lambda p: p["listings"].__setitem__("val", list(reversed(p["listings"]["val"]))), "checksum"),
        (HEADER_NAME, lambda p: p["shard_bindings"].clear(), "no binding in the header"),
        (
            HEADER_NAME,
            lambda p: [b.__setitem__("generation", "99") for b in p["shard_bindings"].values()],
            "disagrees with the header binding",
        ),
    ],
)
def test_loading_validates_the_whole_manifest_set(tmp_path, filename, mutate, message):
    _built(tmp_path)
    _corrupt(tmp_path, filename, mutate)

    with pytest.raises(ValueError, match=message):
        load_manifests(str(tmp_path / "manifests"))


def test_loading_rejects_a_name_shared_across_cohorts(tmp_path):
    _built(tmp_path)
    dev = json.loads((tmp_path / "manifests" / "dev64.json").read_text())
    _corrupt(
        tmp_path,
        "test64.json",
        lambda p: p["rows"].__setitem__(
            0, {**p["rows"][0], "name": dev["rows"][0]["name"], "episode": dev["rows"][0]["episode"]}
        ),
    )

    with pytest.raises(ValueError, match="appears in more than one cohort"):
        load_manifests(str(tmp_path / "manifests"))


def test_loading_rejects_three_windows_from_one_train_episode(tmp_path):
    """A third window for the first episode, inserted so the prescribed ordering still holds -- so the
    per-episode cap is the only rule that can catch it."""
    _built(tmp_path)

    def mutate(payload):
        first = payload["rows"][0]
        payload["rows"][2] = {
            **first,
            "name": _name(int(first["episode"]), start=99),
            "ordinal": first["ordinal"] + 2,
        }

    _corrupt(tmp_path, "train2000.json", mutate)

    with pytest.raises(ValueError, match="at most 2 windows/episode"):
        load_manifests(str(tmp_path / "manifests"))


# ---------------------------------------------------------------- the J0 entry point


def _forbidden_publisher(*args, **kwargs):
    raise AssertionError("nothing may be published until every J0 target holds")


def _j0_world(
    val_records=VAL_EXPECTED_RECORDS,
    train_episodes=TRAIN_EPISODE_TARGET,
    train_shards=3,
    train_episodes_per_shard=None,
):
    """A synthetic J0 world: one val listing, one train listing, and the readers/binders for both."""
    val_shards = [f"gs://bucket/val/shard-{i:05d}.tfrecord" for i in range(2)]
    train_paths = [f"gs://bucket/train/shard-{i:05d}.tfrecord" for i in range(train_shards)]
    per_train = train_episodes_per_shard or max(1, -(-train_episodes // train_shards))
    contents: dict[str, list[int]] = {}
    for index, path in enumerate(val_shards):
        span = val_records // 2 + (val_records % 2 if index else 0)
        start = 0 if index == 0 else val_records // 2
        contents[path] = list(range(start, start + span))
    episode = _TRAIN_EPISODE_OFFSET
    for path in train_paths:
        contents[path] = list(range(episode, episode + per_train))
        episode += per_train

    def lister(pattern):
        return val_shards if "/val/" in pattern else train_paths

    def reader(path):
        for index, ep in enumerate(contents[path]):
            # Two windows per train episode, one per val episode.
            yield _name(ep), 0, _meta(ep)
            if path in train_paths:
                yield _name(ep, start=4), 1, _meta(ep)

    def binder(path):
        return {"kind": "gcs", "generation": "1", "size": 1024}

    return lister, reader, binder


def test_the_j0_entry_point_pins_the_plan_s_numbers():
    assert VAL_EXPECTED_RECORDS == 14_636
    assert TRAIN_EPISODE_TARGET == 5_000
    assert MAX_SCAN_SHARDS == 200
    assert MAX_SCAN_BYTES == 60 * 2**30
    assert MANIFEST_SCHEMA_VERSION == 1


def test_j0_builds_end_to_end(tmp_path):
    lister, reader, binder = _j0_world()

    report = build_j0_manifests(
        "gs://bucket/val",
        "gs://bucket/train",
        str(tmp_path / "out"),
        builder_sha="a" * 40,
        lister=lister,
        reader=reader,
        binder=binder,
    )

    assert report["val_records"] == VAL_EXPECTED_RECORDS
    assert report["train_episodes"] >= TRAIN_EPISODE_TARGET
    loaded = load_manifests(str(tmp_path / "out"))
    assert loaded["header"]["shard_listing_checksum"] == report["shard_listing_checksum"]
    assert {cohort: len(loaded[cohort]["rows"]) for cohort in COHORT_SIZES} == COHORT_SIZES


def test_j0_refuses_a_val_split_that_is_not_the_published_one(tmp_path):
    lister, reader, binder = _j0_world(val_records=VAL_EXPECTED_RECORDS - 1)

    with pytest.raises(ValueError, match=f"holds {VAL_EXPECTED_RECORDS - 1} records"):
        build_j0_manifests(
            "gs://bucket/val",
            "gs://bucket/train",
            str(tmp_path / "out"),
            builder_sha="a" * 40,
            lister=lister,
            reader=reader,
            binder=binder,
            publisher=_forbidden_publisher,
        )
    assert not os.path.exists(str(tmp_path / "out"))


def test_j0_refuses_a_train_listing_exhausted_below_the_target(tmp_path):
    lister, reader, binder = _j0_world(train_episodes=TRAIN_EPISODE_TARGET - 100)

    with pytest.raises(ValueError, match="exhausted at .* short of the 5000 target"):
        build_j0_manifests(
            "gs://bucket/val",
            "gs://bucket/train",
            str(tmp_path / "out"),
            builder_sha="a" * 40,
            lister=lister,
            reader=reader,
            binder=binder,
            publisher=_forbidden_publisher,
        )
    assert not os.path.exists(str(tmp_path / "out"))


@pytest.mark.parametrize("cap", ["shards", "bytes"])
def test_j0_refuses_a_train_scan_that_hits_a_cap(tmp_path, cap):
    lister, reader, binder = _j0_world(train_shards=MAX_SCAN_SHARDS + 5, train_episodes_per_shard=10)
    if cap == "bytes":
        lister, reader, _ = _j0_world(train_shards=4)

        def binder(path):  # each shard alone is a fifth of the byte cap, so shard five is refused
            return {"kind": "gcs", "generation": "1", "size": MAX_SCAN_BYTES // 3}

    with pytest.raises(CapExceeded):
        build_j0_manifests(
            "gs://bucket/val",
            "gs://bucket/train",
            str(tmp_path / "out"),
            builder_sha="a" * 40,
            lister=lister,
            reader=reader,
            binder=binder,
            publisher=_forbidden_publisher,
        )
    assert not os.path.exists(str(tmp_path / "out"))


def test_j0_refuses_an_empty_listing(tmp_path):
    with pytest.raises(ValueError, match="no shards found"):
        build_j0_manifests(
            "gs://bucket/val",
            "gs://bucket/train",
            str(tmp_path / "out"),
            builder_sha="a" * 40,
            lister=lambda pattern: [],
            publisher=_forbidden_publisher,
        )


def test_j0_lists_shards_deterministically(tmp_path):
    seen = []

    def unsorted_lister(pattern):
        paths = ["gs://b/val/shard-00001.tfrecord", "gs://b/val/shard-00000.tfrecord"]
        return paths if "/val/" in pattern else ["gs://b/train/s-1.tfrecord", "gs://b/train/s-0.tfrecord"]

    def reader(path):
        seen.append(path)
        return iter(())

    with pytest.raises(ValueError, match="holds 0 records"):
        build_j0_manifests(
            "gs://b/val",
            "gs://b/train",
            str(tmp_path / "out"),
            builder_sha="a" * 40,
            lister=unsorted_lister,
            reader=reader,
            binder=lambda path: {"kind": "gcs", "generation": "1", "size": 1},
            publisher=_forbidden_publisher,
        )
    assert seen == ["gs://b/val/shard-00000.tfrecord", "gs://b/val/shard-00001.tfrecord"]


# ---------------------------------------------------------------- real producer records


def test_the_scan_reads_real_producer_tfrecords(tmp_path):
    """The strongest fixture available: shards written by the production example builder itself."""
    producer = pytest.importorskip("maxdiffusion.data_preprocessing.wan_side_adapter_droid_cache_to_tfrecord")
    path = str(tmp_path / "shard-00000.tfrecord")
    with producer._TFRecordWriter(path) as writer:
        for episode in range(3):
            writer.write(
                producer._example_proto(
                    {
                        "name": producer._bytes_feature(_name(episode, start=episode * 4).encode("utf-8")),
                        "ordinal": producer._int64_feature(episode),
                        "z_i0": producer._bytes_feature(b"\x00" * 8),
                        "z_video": producer._bytes_feature(b"\x00" * 8),
                        "actions": producer._bytes_feature(b"\x00" * 8),
                        "meta_json": producer._bytes_feature(_meta(episode, view=0)),
                    }
                )
            )

    result = scan_split([path], split="val")

    assert result.episodes == 3
    assert [window.name for window in result.windows] == [_name(e, start=e * 4) for e in range(3)]
    assert [window.episode for window in result.windows] == ["0", "1", "2"]
    assert [window.ordinal for window in result.windows] == [0, 1, 2]
    assert result.bindings[path]["size"] == os.path.getsize(path)
