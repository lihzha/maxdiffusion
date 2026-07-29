"""CPU-only tests for exp_02 manifest assembly + fail-closed verification (cycle A, A2).

Covers the orchestration half of `maxdiffusion.data_preprocessing.build_overfit100_manifest`
against a fake IO layer -- plan v4 D5: the committed manifest carries complete provenance
(dual GCS fingerprints per episode, the embedded V1-fixture fingerprint, tool versions, the
builder commit, and the ordered draw log), and `verify_manifest` is what the cycle-B build
job calls to fail loudly on drift OR on malformed structure.

Strengthened after the Codex cycle-A review (A1-A4):
* A1 -- production builds refuse to run from a dirty/uncommitted implementation.
* A2 -- every downloaded object is bound to the fingerprint that was statted (md5 + size),
  and the annotation's embedded `episode_id` must match the drawn candidate.
* A3 -- per-object outcomes are classified `found` / `absent` / `error`; transient/tool
  errors abort the build instead of masquerading as a rejection reason, and errors on
  candidates prefetched past the stopping acceptance never affect the walk.
* A4 -- `validate_manifest_structure` is a fail-closed structural gate that runs BEFORE any
  remote stat, with one mutation test per invariant.

No network: `gsutil`/`ffprobe` appear only through their pure output parsers (fed captured
real output) and through a fake IO object.
"""

from __future__ import annotations

import copy

import pytest

from maxdiffusion.data_preprocessing.build_overfit100_manifest import (
    REASONS,
    SourceError,
    annotation_uri,
    build_manifest,
    implementation_provenance_errors,
    parse_ffprobe_json,
    parse_git_porcelain,
    validate_manifest_structure,
    verify_annotation_binding,
    verify_manifest,
    video_uri,
)
from maxdiffusion.data_preprocessing.extract_v1_fixture import Resolved, parse_gsutil_stat

_FIXTURE_FP = {
    "uri": "gs://v6_east1d/datasets/exp02_overfit100/fixtures/v1_cache_windows.npz",
    "generation": 1785299999999999,
    "md5": "ZmFrZWZpeHR1cmVtZDU1NTU1NQ==",
    "size_bytes": 691200,
    "names": ["ep0_v0_s00000", "ep0_v0_s00004", "ep0_v0_s00008"],
    "shapes": {"z_i0": [48, 1, 12, 20], "z_video": [48, 9, 12, 20]},
    "dtypes": {"z_i0": "float16", "z_video": "float16"},
}

_TOOLS = {
    "python": "3.12.8",
    "gsutil": "gsutil version: 5.37",
    "ffprobe": "ffprobe version 8.0",
    "ffmpeg": "ffmpeg version 8.0",
    "numpy": "2.5.1",
    "jax": "0.11.0",
}

_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _fp(uri, generation, size):
    return {"uri": uri, "generation": generation, "md5": f"md5-{generation}", "size": size}


def _probe(nb_frames):
    return {"width": 320, "height": 192, "nb_frames": nb_frames, "fps": 5.0, "pix_fmt": "yuv420p"}


def _ann(episode_id, success=1, texts=("do the thing",)):
    return {"episode_id": episode_id, "success": success, "texts": list(texts), "video_length": 88}


class FakeIO:
    """Serves the four IO seams from dicts, records calls, and can inject source errors.

    ``errors`` maps ``(seam, episode_id) -> message`` where seam is one of
    ``stat_annotation`` / ``fetch_annotation`` / ``stat_video`` / ``probe``.
    """

    def __init__(self, annotations, missing_videos=(), probes=None, errors=None):
        self.annotations = annotations
        self.missing_videos = set(missing_videos)
        self.probes = probes or {}
        self.errors = errors or {}
        self.statted_annotations, self.fetched, self.statted_videos, self.probed = [], [], [], []

    def _annotation_fp(self, episode_id):
        return _fp(annotation_uri(episode_id), 1785200000000000 + episode_id, 270000 + episode_id)

    def _video_fp(self, episode_id):
        return _fp(video_uri(episode_id), 1785000000000000 + episode_id, 300000 + episode_id)

    def stat_annotations(self, episode_ids):
        self.statted_annotations.extend(episode_ids)
        out = {}
        for episode_id in episode_ids:
            if ("stat_annotation", episode_id) in self.errors:
                out[episode_id] = Resolved.failed(self.errors[("stat_annotation", episode_id)])
            elif self.annotations.get(episode_id) is None:
                out[episode_id] = Resolved.absent()
            else:
                out[episode_id] = Resolved.found(self._annotation_fp(episode_id))
        return out

    def fetch_annotations(self, episode_ids, fingerprints):
        self.fetched.extend(episode_ids)
        out = {}
        for episode_id in episode_ids:
            if ("fetch_annotation", episode_id) in self.errors:
                out[episode_id] = Resolved.failed(self.errors[("fetch_annotation", episode_id)])
            else:
                out[episode_id] = Resolved.found(fingerprints[episode_id], payload=self.annotations[episode_id])
        return out

    def stat_videos(self, episode_ids):
        self.statted_videos.extend(episode_ids)
        out = {}
        for episode_id in episode_ids:
            if ("stat_video", episode_id) in self.errors:
                out[episode_id] = Resolved.failed(self.errors[("stat_video", episode_id)])
            elif episode_id in self.missing_videos:
                out[episode_id] = Resolved.absent()
            else:
                out[episode_id] = Resolved.found(self._video_fp(episode_id))
        return out

    def probe_video(self, episode_id, fingerprint):
        self.probed.append(episode_id)
        if ("probe", episode_id) in self.errors:
            return Resolved.failed(self.errors[("probe", episode_id)])
        return Resolved.found(fingerprint, payload=self.probes[episode_id])


def _default_world(**kwargs):
    """8 candidates: ids 10..17; 12 fails success, 14 has no usable text, 16 has no video."""
    annotations = {
        10: _ann(10, texts=("open the drawer", "", "pull the handle")),
        11: _ann(11),
        12: _ann(12, success=0),
        13: _ann(13),
        14: _ann(14, texts=("", "   ")),
        15: _ann(15),
        16: _ann(16),
        17: _ann(17),
    }
    probes = {ep: _probe(88 if ep % 2 else 128) for ep in annotations}
    probes[13] = _probe(32)  # too short -- rejected at the probe stage
    return FakeIO(annotations, missing_videos={16}, probes=probes, **kwargs)


def _build(io, n_target=3, order=(10, 11, 12, 13, 14, 15, 16, 17), **kwargs):
    return build_manifest(
        io,
        seed=0,
        n_target=n_target,
        fixture=_FIXTURE_FP,
        builder_commit=_COMMIT,
        created_utc="2026-07-28T10:00:00+00:00",
        tool_versions=_TOOLS,
        order=list(order),
        **kwargs,
    )


# ----------------------------------------------------------------------------------
# 1. Manifest assembly -- every provenance field present, in acceptance order.
# ----------------------------------------------------------------------------------


def test_manifest_top_level_provenance_fields():
    manifest = _build(_default_world())
    assert manifest["selection_seed"] == 0
    assert manifest["builder_commit"] == _COMMIT
    assert manifest["created_utc"] == "2026-07-28T10:00:00+00:00"
    assert manifest["tool_versions"] == _TOOLS
    assert manifest["fixture"] == _FIXTURE_FP
    assert set(manifest) >= {"episodes", "draw_log", "rejection_tally", "totals"}


def test_manifest_episode_records_carry_dual_fingerprints_and_geometry():
    manifest = _build(_default_world())
    episodes = manifest["episodes"]
    assert [e["episode_index"] for e in episodes] == [0, 1, 2]
    assert [e["episode_id"] for e in episodes] == [10, 11, 15]

    first = episodes[0]
    assert first["texts"] == ["open the drawer", "", "pull the handle"]
    assert first["chosen_text_index"] in (0, 2)
    assert first["chosen_text_raw"] == first["texts"][first["chosen_text_index"]]
    assert first["used_text"] == first["chosen_text_raw"].strip()
    assert first["annotation_fingerprint"] == _fp(annotation_uri(10), 1785200000000010, 270010)
    assert first["video_fingerprint"] == _fp(video_uri(10), 1785000000000010, 300010)
    assert first["ffprobe"] == _probe(128)
    assert first["n_windows"] == 24


def test_manifest_used_text_is_the_stripped_raw_text():
    io = FakeIO({20: _ann(20, texts=("  wipe the table  ",))}, probes={20: _probe(88)})
    manifest = _build(io, n_target=1, order=(20,))
    episode = manifest["episodes"][0]
    assert episode["chosen_text_raw"] == "  wipe the table  "
    assert episode["used_text"] == "wipe the table"


# ----------------------------------------------------------------------------------
# 2. Draw log / tally / totals -- the audit trail of the seeded walk.
# ----------------------------------------------------------------------------------


def test_draw_log_records_every_candidate_in_order_with_its_reason():
    manifest = _build(_default_world())
    assert manifest["draw_log"] == [
        {"episode_id": 10, "reason": "accepted"},
        {"episode_id": 11, "reason": "accepted"},
        {"episode_id": 12, "reason": "not_success"},
        {"episode_id": 13, "reason": "too_short"},
        {"episode_id": 14, "reason": "no_nonempty_text"},
        {"episode_id": 15, "reason": "accepted"},
    ]


def test_draw_log_stops_at_the_nth_acceptance():
    manifest = _build(_default_world())
    assert [d["episode_id"] for d in manifest["draw_log"]][-1] == 15
    assert 17 not in {d["episode_id"] for d in manifest["draw_log"]}


def test_missing_annotation_is_logged_and_skipped():
    io = FakeIO({30: None, 31: _ann(31)}, probes={31: _probe(88)})
    manifest = _build(io, n_target=1, order=(30, 31))
    assert manifest["draw_log"][0] == {"episode_id": 30, "reason": "missing_annotation"}
    assert manifest["episodes"][0]["episode_id"] == 31


def test_missing_video_is_logged_as_such():
    io = FakeIO({40: _ann(40), 41: _ann(41)}, missing_videos={40}, probes={41: _probe(88)})
    manifest = _build(io, n_target=1, order=(40, 41))
    assert manifest["draw_log"][0] == {"episode_id": 40, "reason": "missing_video"}


def test_rejection_tally_reconciles_with_the_draw_log():
    manifest = _build(_default_world())
    assert manifest["rejection_tally"] == {
        "accepted": 3,
        "not_success": 1,
        "too_short": 1,
        "no_nonempty_text": 1,
    }
    assert sum(manifest["rejection_tally"].values()) == len(manifest["draw_log"])


def test_totals_sum_windows_over_accepted_episodes():
    manifest = _build(_default_world())
    assert manifest["totals"]["episodes"] == 3
    assert manifest["totals"]["windows"] == sum(e["n_windows"] for e in manifest["episodes"])


def test_draw_log_reasons_stay_inside_the_declared_vocabulary():
    worlds = [
        (_default_world(), 3, (10, 11, 12, 13, 14, 15, 16, 17)),
        (FakeIO({30: None, 31: _ann(31)}, probes={31: _probe(88)}), 1, (30, 31)),
        (FakeIO({40: _ann(40), 41: _ann(41)}, missing_videos={40}, probes={41: _probe(88)}), 1, (40, 41)),
    ]
    seen = {draw["reason"] for io, n, order in worlds for draw in _build(io, n_target=n, order=order)["draw_log"]}
    assert seen == set(REASONS)


def test_block_size_never_changes_the_walk():
    reference = _build(_default_world(), block_size=1)
    for block_size in (2, 3, 7, 25, 100):
        manifest = _build(_default_world(), block_size=block_size)
        assert manifest["draw_log"] == reference["draw_log"]
        assert manifest["episodes"] == reference["episodes"]
        assert manifest["totals"] == reference["totals"]


def test_block_size_never_changes_the_dry_run_walk():
    reference = _build(_default_world(), block_size=1, dry_run=True)
    for block_size in (3, 25):
        assert _build(_default_world(), block_size=block_size, dry_run=True)["draw_log"] == reference["draw_log"]


def test_videos_are_probed_only_for_consumed_candidates():
    # A3 / seam a: the block prefetch reads METADATA only (stats, whose errors are deferred).
    # No MP4 is ever downloaded for a candidate the walk does not consume, at any block size.
    for block_size in (1, 3, 25):
        io = _default_world()
        _build(io, block_size=block_size)
        assert io.probed == [10, 11, 13, 15]  # exactly the provisional accepts that were consumed
        # Candidates rejected at annotation level never reach the video seam at all.
        assert 12 not in io.statted_videos and 14 not in io.statted_videos


def test_only_metadata_is_prefetched_past_the_stopping_acceptance():
    io = _default_world()
    _build(io, block_size=25)
    # 17 is in the same block as the stopping acceptance: it may be statted (cheap, deferred)
    # but must never be drawn nor downloaded.
    assert 17 not in io.probed
    assert 17 in io.statted_annotations


def test_build_raises_when_the_pool_is_exhausted_before_n():
    io = FakeIO({50: _ann(50, success=0)})
    with pytest.raises(RuntimeError, match="exhausted"):
        _build(io, n_target=1, order=(50,))


# ----------------------------------------------------------------------------------
# 3. --dry-run -- annotation-level decisions only, no video stats and no downloads.
# ----------------------------------------------------------------------------------


def test_dry_run_makes_no_video_stat_or_probe_calls():
    io = _default_world()
    manifest = _build(io, dry_run=True)
    assert io.statted_videos == [] and io.probed == []
    assert manifest["provisional"] is True
    assert [e["episode_id"] for e in manifest["episodes"]] == [10, 11, 13]  # 13 not yet probed
    for episode in manifest["episodes"]:
        assert episode["video_fingerprint"] is None
        assert episode["ffprobe"] is None
        assert episode["n_windows"] is None


def test_dry_run_still_binds_annotation_bytes():
    # Even the cheap path reads annotations through the stat -> fetch binding.
    io = _default_world()
    _build(io, dry_run=True)
    assert io.statted_annotations and io.fetched


def test_full_run_is_not_marked_provisional():
    assert _build(_default_world())["provisional"] is False


# ----------------------------------------------------------------------------------
# 4. A2 -- content is bound to the fingerprint that was recorded.
# ----------------------------------------------------------------------------------


def test_verify_annotation_binding_accepts_a_matching_episode_id():
    assert verify_annotation_binding(42, _ann(42)) == []


def test_verify_annotation_binding_rejects_a_mismatched_episode_id():
    errors = verify_annotation_binding(42, _ann(43))
    assert errors and any("episode_id" in e for e in errors)


def test_verify_annotation_binding_rejects_a_non_mapping():
    assert verify_annotation_binding(42, ["not", "an", "object"])


def test_build_aborts_when_the_annotation_names_a_different_episode():
    # A swapped/overwritten annotation must never be silently attributed to the drawn id.
    io = FakeIO({60: _ann(61), 61: _ann(61)}, probes={60: _probe(88), 61: _probe(88)})
    with pytest.raises(SourceError, match="episode_id"):
        _build(io, n_target=1, order=(60, 61))


def test_build_aborts_when_a_download_fails_its_md5_binding():
    # The IO layer surfaces the mismatch as a source error; the walk must not swallow it.
    io = _default_world(errors={("fetch_annotation", 10): "md5 mismatch for annotation 10"})
    with pytest.raises(SourceError, match="md5"):
        _build(io)


def test_build_aborts_when_a_probe_fails_after_a_successful_stat():
    # Seam f: stat says the MP4 exists but it cannot be downloaded/probed -> source error,
    # NOT a `missing_video` rejection that would silently shrink the corpus.
    io = _default_world(errors={("probe", 11): "ffprobe exited 1"})
    with pytest.raises(SourceError, match="ffprobe"):
        _build(io)


# ----------------------------------------------------------------------------------
# 5. A3 -- transient errors abort; errors past the stopping acceptance are irrelevant.
# ----------------------------------------------------------------------------------


def test_build_aborts_on_a_stat_error_before_the_nth_acceptance():
    io = _default_world(errors={("stat_annotation", 11): "ServiceException: 503 Backend Error"})
    with pytest.raises(SourceError, match="503"):
        _build(io)


def test_build_aborts_on_a_video_stat_error_before_the_nth_acceptance():
    io = _default_world(errors={("stat_video", 11): "ServiceException: 503 Backend Error"})
    with pytest.raises(SourceError, match="503"):
        _build(io)


def test_an_error_after_the_stopping_acceptance_does_not_affect_the_walk():
    # 17 is prefetched in the same block as the 3rd acceptance (15) but never consumed.
    clean = _build(_default_world())
    io = _default_world(errors={("stat_annotation", 17): "ServiceException: 503 Backend Error"})
    assert _build(io, block_size=25) == clean


def test_an_error_on_a_never_drawn_candidate_is_never_raised():
    io = _default_world(errors={("fetch_annotation", 16): "ServiceException: 500"})
    assert _build(io, block_size=25)["totals"]["episodes"] == 3


def test_transient_errors_never_enter_the_rejection_vocabulary():
    io = _default_world(errors={("stat_annotation", 11): "ServiceException: 503"})
    with pytest.raises(SourceError):
        _build(io)
    assert set(REASONS) == {
        "missing_annotation",
        "not_success",
        "no_nonempty_text",
        "missing_video",
        "too_short",
        "accepted",
    }


# ----------------------------------------------------------------------------------
# 6. A1 -- production builds refuse a dirty / uncommitted implementation.
# ----------------------------------------------------------------------------------


def test_parse_git_porcelain_reads_every_dirty_state():
    text = (
        "?? src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py\n"
        " M src/maxdiffusion/data_preprocessing/extract_v1_fixture.py\n"
        "A  src/maxdiffusion/tests/worklogs_yixun/test_overfit100_fixture.py\n"
        "R  old/path.py -> src/maxdiffusion/tests/worklogs_yixun/test_overfit100_selection.py\n"
    )
    assert parse_git_porcelain(text) == [
        "src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py",
        "src/maxdiffusion/data_preprocessing/extract_v1_fixture.py",
        "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_fixture.py",
        "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_selection.py",
    ]


def test_parse_git_porcelain_on_a_clean_tree():
    assert parse_git_porcelain("") == []


def test_implementation_provenance_is_clean_when_tracked_and_unmodified():
    paths = ("a.py", "b.py")
    assert implementation_provenance_errors(set(paths), [], paths=paths) == []


def test_implementation_provenance_rejects_an_untracked_file():
    paths = ("a.py", "b.py")
    errors = implementation_provenance_errors({"a.py"}, [], paths=paths)
    assert errors and any("b.py" in e and "HEAD" in e for e in errors)


def test_implementation_provenance_rejects_a_dirty_file():
    paths = ("a.py", "b.py")
    errors = implementation_provenance_errors(set(paths), ["b.py"], paths=paths)
    assert errors and any("b.py" in e for e in errors)


def test_implementation_provenance_ignores_dirt_outside_the_implementation():
    paths = ("a.py",)
    assert implementation_provenance_errors({"a.py"}, ["docs/notes.md"], paths=paths) == []


# ----------------------------------------------------------------------------------
# 7. A4 -- fail-closed structural validation, one mutation per invariant.
# ----------------------------------------------------------------------------------


def _valid_manifest():
    return _build(_default_world())


def test_validate_manifest_structure_accepts_a_real_manifest():
    assert validate_manifest_structure(_valid_manifest()) == []


def test_validate_manifest_structure_enforces_the_expected_episode_count():
    assert validate_manifest_structure(_valid_manifest(), expected_episodes=3) == []
    errors = validate_manifest_structure(_valid_manifest(), expected_episodes=100)
    assert errors and any("100" in e for e in errors)


_MUTATIONS = {
    "missing_totals": lambda m: m.pop("totals"),
    "missing_fixture": lambda m: m.pop("fixture"),
    "provisional_true": lambda m: m.update(provisional=True),
    "seed_not_int": lambda m: m.update(selection_seed="zero"),
    "commit_not_a_sha": lambda m: m.update(builder_commit="dirty-tree"),
    "tool_versions_missing_ffmpeg": lambda m: m["tool_versions"].pop("ffmpeg"),
    "tool_versions_blank": lambda m: m["tool_versions"].update(gsutil=""),
    "fixture_missing_generation": lambda m: m["fixture"].pop("generation"),
    "fixture_wrong_name_count": lambda m: m["fixture"].update(names=["only_one"]),
    "episode_missing_key": lambda m: m["episodes"][0].pop("used_text"),
    "episode_index_not_contiguous": lambda m: m["episodes"][1].update(episode_index=7),
    "duplicate_episode_id": lambda m: m["episodes"][1].update(episode_id=m["episodes"][0]["episode_id"]),
    "chosen_index_out_of_range": lambda m: m["episodes"][0].update(chosen_text_index=99),
    "chosen_index_on_empty_text": lambda m: m["episodes"][0].update(
        chosen_text_index=1, chosen_text_raw="", used_text=""
    ),
    "chosen_raw_mismatch": lambda m: m["episodes"][0].update(chosen_text_raw="something else"),
    "used_text_not_stripped": lambda m: m["episodes"][0].update(used_text="  padded  "),
    "empty_texts_list": lambda m: m["episodes"][0].update(texts=[]),
    "annotation_fingerprint_none": lambda m: m["episodes"][0].update(annotation_fingerprint=None),
    "video_fingerprint_none": lambda m: m["episodes"][1].update(video_fingerprint=None),
    "annotation_uri_wrong": lambda m: m["episodes"][0]["annotation_fingerprint"].update(uri="gs://other/x.json"),
    "video_uri_wrong": lambda m: m["episodes"][0]["video_fingerprint"].update(uri="gs://other/x.mp4"),
    "fingerprint_missing_md5": lambda m: m["episodes"][0]["annotation_fingerprint"].pop("md5"),
    "ffprobe_none": lambda m: m["episodes"][0].update(ffprobe=None),
    "ffprobe_too_short": lambda m: m["episodes"][0]["ffprobe"].update(nb_frames=32),
    "ffprobe_wrong_geometry": lambda m: m["episodes"][0]["ffprobe"].update(width=640),
    "n_windows_wrong": lambda m: m["episodes"][0].update(n_windows=999),
    "draw_reason_unknown": lambda m: m["draw_log"][2].update(reason="because"),
    "draw_log_duplicate_id": lambda m: m["draw_log"][1].update(episode_id=m["draw_log"][0]["episode_id"]),
    "draw_log_last_not_accepted": lambda m: m["draw_log"].append({"episode_id": 99, "reason": "not_success"}),
    "draw_log_accepted_mismatch": lambda m: m["draw_log"][0].update(reason="not_success"),
    "tally_disagrees": lambda m: m["rejection_tally"].update(accepted=99),
    "totals_episodes_wrong": lambda m: m["totals"].update(episodes=99),
    "totals_windows_wrong": lambda m: m["totals"].update(windows=1),
}


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_validate_manifest_structure_catches_every_mutation(mutation):
    manifest = copy.deepcopy(_valid_manifest())
    _MUTATIONS[mutation](manifest)
    assert validate_manifest_structure(manifest), f"mutation {mutation!r} was not caught"


def test_validate_manifest_structure_rejects_a_non_mapping():
    assert validate_manifest_structure(["not", "a", "manifest"])


# ----------------------------------------------------------------------------------
# 8. verify_manifest -- structure first (fail closed), then live fingerprints.
# ----------------------------------------------------------------------------------


def _stat_fn_from(manifest, overrides=None, missing=(), errors=()):
    """Fake `gsutil stat` that echoes the manifest's own fingerprints, with optional drift."""
    table = {_FIXTURE_FP["uri"]: {**_FIXTURE_FP, "size": _FIXTURE_FP["size_bytes"]}}
    for episode in manifest["episodes"]:
        for key in ("annotation_fingerprint", "video_fingerprint"):
            fingerprint = episode[key]
            table[fingerprint["uri"]] = dict(fingerprint)
    for uri, patch in (overrides or {}).items():
        table[uri] = {**table[uri], **patch}
    for uri in missing:
        table.pop(uri, None)

    def stat_fn(uris):
        resolved = {}
        for uri in uris:
            if uri in errors:
                resolved[uri] = Resolved.failed("ServiceException: 503 Backend Error")
            elif uri in table:
                resolved[uri] = Resolved.found(table[uri])
            else:
                resolved[uri] = Resolved.absent()
        return resolved

    return stat_fn


def test_verify_manifest_passes_when_nothing_drifted():
    manifest = _valid_manifest()
    assert verify_manifest(manifest, stat_fn=_stat_fn_from(manifest)) == []


def test_verify_manifest_detects_md5_drift():
    manifest = _valid_manifest()
    stat_fn = _stat_fn_from(manifest, overrides={video_uri(11): {"md5": "changed=="}})
    errors = verify_manifest(manifest, stat_fn=stat_fn)
    assert len(errors) == 1 and "md5" in errors[0] and video_uri(11) in errors[0]


def test_verify_manifest_detects_generation_drift():
    manifest = _valid_manifest()
    stat_fn = _stat_fn_from(manifest, overrides={annotation_uri(10): {"generation": 1}})
    errors = verify_manifest(manifest, stat_fn=stat_fn)
    assert len(errors) == 1 and "generation" in errors[0]


def test_verify_manifest_detects_size_drift():
    manifest = _valid_manifest()
    stat_fn = _stat_fn_from(manifest, overrides={video_uri(15): {"size": 12345}})
    errors = verify_manifest(manifest, stat_fn=stat_fn)
    assert len(errors) == 1 and "size" in errors[0]


def test_verify_manifest_detects_a_deleted_object():
    manifest = _valid_manifest()
    stat_fn = _stat_fn_from(manifest, missing=[video_uri(10)])
    errors = verify_manifest(manifest, stat_fn=stat_fn)
    assert len(errors) == 1 and "absent" in errors[0]


def test_verify_manifest_reports_a_stat_error_rather_than_calling_it_absent():
    manifest = _valid_manifest()
    stat_fn = _stat_fn_from(manifest, errors={video_uri(10)})
    errors = verify_manifest(manifest, stat_fn=stat_fn)
    assert len(errors) == 1 and "503" in errors[0]


def test_verify_manifest_checks_the_embedded_fixture_too():
    manifest = _valid_manifest()
    stat_fn = _stat_fn_from(manifest, overrides={_FIXTURE_FP["uri"]: {"md5": "otherhash=="}})
    errors = verify_manifest(manifest, stat_fn=stat_fn)
    assert len(errors) == 1 and _FIXTURE_FP["uri"] in errors[0]


def test_verify_manifest_reports_every_drifted_object():
    manifest = _valid_manifest()
    stat_fn = _stat_fn_from(
        manifest,
        overrides={video_uri(10): {"md5": "a=="}, annotation_uri(11): {"size": 7}},
        missing=[video_uri(15)],
    )
    assert len(verify_manifest(manifest, stat_fn=stat_fn)) == 3


def test_verify_manifest_fails_closed_before_touching_the_network():
    # A structurally broken manifest must be rejected WITHOUT any remote stat (A4).
    manifest = _valid_manifest()
    manifest["episodes"][0]["n_windows"] = 999
    calls = []

    def stat_fn(uris):
        calls.append(list(uris))
        return {uri: Resolved.found({"uri": uri}) for uri in uris}

    errors = verify_manifest(manifest, stat_fn=stat_fn)
    assert errors and calls == []


# ----------------------------------------------------------------------------------
# 9. Output parsers -- captured real `gsutil stat` / `ffprobe` output (probes 2026-07-28).
# ----------------------------------------------------------------------------------

_REAL_STAT_TWO = """gs://v6_east1d/datasets/droid_ctrl_world_aligned/annotation/train/5.json:
    Creation time:          Tue, 28 Jul 2026 04:55:13 GMT
    Update time:            Tue, 28 Jul 2026 04:55:13 GMT
    Storage class:          STANDARD
    Content-Length:         271540
    Content-Type:           application/json
    Hash (crc32c):          1kHF2Q==
    Hash (md5):             s/Lw9t3RHuszAmu6h7eosA==
    ETag:                   CKL7t6fK9JUDEAE=
    Generation:             1785214513577378
    Metageneration:         1
gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/5/0.mp4:
    Creation time:          Tue, 28 Jul 2026 13:54:55 GMT
    Update time:            Tue, 28 Jul 2026 13:54:55 GMT
    Storage class:          STANDARD
    Content-Length:         239431
    Content-Type:           video/mp4
    Hash (crc32c):          vKp+5w==
    Hash (md5):             /mWFoyi8P7XCpI2SpUYlWA==
    ETag:                   CK3mr/jC9ZUDEAE=
    Generation:             1785246895567661
    Metageneration:         1
"""

_REAL_FFPROBE = """{
    "programs": [

    ],
    "stream_groups": [

    ],
    "streams": [
        {
            "width": 320,
            "height": 192,
            "pix_fmt": "yuv420p",
            "r_frame_rate": "5/1",
            "nb_frames": "88",
            "nb_read_frames": "88"
        }
    ]
}
"""


def test_parse_gsutil_stat_handles_a_batched_multi_object_call():
    parsed = parse_gsutil_stat(_REAL_STAT_TWO)
    assert set(parsed) == {annotation_uri(5), video_uri(5)}
    assert parsed[annotation_uri(5)]["generation"] == 1785214513577378
    assert parsed[annotation_uri(5)]["size"] == 271540
    assert parsed[video_uri(5)]["md5"] == "/mWFoyi8P7XCpI2SpUYlWA=="
    assert parsed[video_uri(5)]["size"] == 239431


def test_parse_ffprobe_json_extracts_geometry_and_counted_frames():
    assert parse_ffprobe_json(_REAL_FFPROBE) == {
        "width": 320,
        "height": 192,
        "nb_frames": 88,
        "fps": 5.0,
        "pix_fmt": "yuv420p",
    }


def test_parse_ffprobe_json_prefers_the_counted_frame_total():
    text = _REAL_FFPROBE.replace('"nb_read_frames": "88"', '"nb_read_frames": "84"')
    assert parse_ffprobe_json(text)["nb_frames"] == 84


def test_parse_ffprobe_json_rejects_output_with_no_video_stream():
    with pytest.raises(ValueError):
        parse_ffprobe_json('{"streams": []}')


def test_parse_ffprobe_json_rejects_a_stream_missing_geometry():
    text = _REAL_FFPROBE.replace('"width": 320,\n', "")
    with pytest.raises(ValueError, match="width"):
        parse_ffprobe_json(text)
