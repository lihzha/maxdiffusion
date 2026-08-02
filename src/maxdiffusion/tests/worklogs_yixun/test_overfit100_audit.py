"""CPU-only tests for the exp_02 duplicate-condition audit (cycle B, deliverable B1/D6).

Plan v4 §1 makes the audit a *pre-registration* artifact: the build report lists the exact
duplicate-instruction groups and the minimum pairwise `z_i0` L2 distance among window pairs
**with different targets**, and §1/G4 then fixes the success-rule denominator at build time --
collided windows are flagged but never dropped post hoc. If this audit under-reports, exp_02
could claim "memorization" over a set whose conditions do not actually distinguish their
targets; if it over-reports (e.g. by counting pairs that share the same target, which are
harmless by construction), the pre-launch report cries wolf.

Everything here is local: tiny synthetic fixtures with hand-checkable answers, plus one
check against the committed manifest (the artifact the shuffled-control derangement in §1
was sized from). No accelerator, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from maxdiffusion.data_preprocessing.build_overfit100_dataset import (
    BuildError,
    duplicate_instruction_groups,
    min_pairwise_z_i0,
    target_key,
)

MANIFEST_PATH = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "worklogs_yixun"
    / "exp_02_overfit100_claude"
    / "overfit100_manifest.json"
)


def _episodes(texts):
    return [
        {"episode_index": i, "episode_id": 1000 + i, "used_text": text, "n_windows": 3} for i, text in enumerate(texts)
    ]


def _brute_force_min(vectors, keys):
    best = None
    pairs = 0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if keys[i] == keys[j]:
                continue
            pairs += 1
            distance = float(np.linalg.norm(vectors[i].astype(np.float64) - vectors[j].astype(np.float64)))
            if best is None or distance < best[0]:
                best = (distance, i, j)
    return best, pairs


# ----------------------------------------------------------------------------------
# 1. Exact-duplicate instruction groups.
# ----------------------------------------------------------------------------------


def test_duplicate_groups_finds_the_known_duplicates():
    groups = duplicate_instruction_groups(_episodes(["fold the cloth", "press button", "fold the cloth", "wipe"]))
    assert len(groups) == 1
    group = groups[0]
    assert group["used_text"] == "fold the cloth"
    assert group["count"] == 2
    assert group["episode_indices"] == [0, 2]
    assert group["episode_ids"] == [1000, 1002]


def test_duplicate_groups_is_empty_when_every_instruction_is_unique():
    assert duplicate_instruction_groups(_episodes(["a", "b", "c"])) == []


def test_duplicate_groups_are_ordered_by_size_then_text():
    groups = duplicate_instruction_groups(_episodes(["b", "a", "b", "a", "b", "c", "c"]))
    assert [(g["used_text"], g["count"]) for g in groups] == [("b", 3), ("a", 2), ("c", 2)]


def test_duplicate_groups_are_exact_matches_only():
    # Near-duplicates (case/whitespace) are NOT merged: the shuffled control compares the
    # exact stored strings, so the audit must use the same equality.
    groups = duplicate_instruction_groups(_episodes(["Fold the cloth", "fold the cloth", "fold the cloth "]))
    assert groups == []


def test_duplicate_groups_on_the_committed_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    groups = duplicate_instruction_groups(manifest["episodes"])
    # Plan §1 / cycle-A review: 6 exact-duplicate groups covering 22 of the 100 episodes --
    # the fact that forced the shuffled control to be a VALUE derangement.
    assert len(groups) == 6
    assert sum(g["count"] for g in groups) == 22
    assert max(g["count"] for g in groups) == 7
    for group in groups:
        assert len(group["episode_indices"]) == group["count"] == len(group["episode_ids"])


# ----------------------------------------------------------------------------------
# 2. Minimum pairwise z_i0 distance among windows with DIFFERENT targets.
# ----------------------------------------------------------------------------------


def test_min_pairwise_finds_the_planted_closest_pair():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(8, 6)).astype(np.float16) * 4.0
    vectors[5] = vectors[2] + np.float16(0.01)  # planted near-duplicate INPUT
    keys = [f"t{i}" for i in range(8)]  # every target distinct

    result = min_pairwise_z_i0(vectors, keys)

    assert result["argmin_pair"] == [2, 5]
    assert result["min_distance"] == pytest.approx(
        float(np.linalg.norm(vectors[2].astype(np.float64) - vectors[5].astype(np.float64))), rel=1e-3
    )
    assert result["n_windows"] == 8
    assert result["n_pairs_compared"] == 28


def test_min_pairwise_matches_brute_force():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(12, 5)).astype(np.float32)
    keys = [f"t{i}" for i in range(12)]
    (expected_distance, i, j), pairs = _brute_force_min(vectors, keys)
    result = min_pairwise_z_i0(vectors, keys)
    assert result["argmin_pair"] == [i, j]
    assert result["min_distance"] == pytest.approx(expected_distance, rel=1e-4)
    assert result["n_pairs_compared"] == pairs


def test_min_pairwise_excludes_pairs_that_share_a_target():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(6, 4)).astype(np.float32)
    vectors[1] = vectors[0] + 1e-4  # closest pair overall...
    keys = ["same", "same", "c", "d", "e", "f"]  # ...but the two share one target

    result = min_pairwise_z_i0(vectors, keys)

    assert result["argmin_pair"] != [0, 1]
    assert result["n_pairs_compared"] == 15 - 1  # the identical-target pair is excluded
    (expected_distance, i, j), _ = _brute_force_min(vectors, keys)
    assert result["argmin_pair"] == [i, j]
    assert result["min_distance"] == pytest.approx(expected_distance, rel=1e-4)


def test_min_pairwise_never_compares_a_window_with_itself():
    vectors = np.zeros((3, 4), dtype=np.float32)
    vectors[1, 0] = 1.0
    vectors[2, 0] = 2.0
    result = min_pairwise_z_i0(vectors, ["a", "b", "c"])
    assert result["min_distance"] == pytest.approx(1.0)
    assert result["argmin_pair"] in ([0, 1], [1, 2])
    assert result["n_pairs_compared"] == 3


def test_min_pairwise_is_invariant_to_the_chunk_size():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(37, 9)).astype(np.float32)
    keys = [f"t{i % 30}" for i in range(37)]  # a few shared targets, exercising the mask
    reference = min_pairwise_z_i0(vectors, keys, chunk_size=1024)
    for chunk_size in (1, 2, 7, 37, 64):
        result = min_pairwise_z_i0(vectors, keys, chunk_size=chunk_size)
        assert result["argmin_pair"] == reference["argmin_pair"]
        assert result["min_distance"] == pytest.approx(reference["min_distance"], rel=1e-6)
        assert result["n_pairs_compared"] == reference["n_pairs_compared"]
        assert [p["distance"] for p in result["smallest_pairs"]] == pytest.approx(
            [p["distance"] for p in reference["smallest_pairs"]], rel=1e-6
        )


def test_min_pairwise_histogram_is_the_n_smallest_ascending():
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(20, 6)).astype(np.float32)
    keys = [f"t{i}" for i in range(20)]
    result = min_pairwise_z_i0(vectors, keys, histogram_n=5)
    distances = [p["distance"] for p in result["smallest_pairs"]]
    assert len(distances) == 5
    assert distances == sorted(distances)
    assert distances[0] == pytest.approx(result["min_distance"])

    all_pairs = sorted(
        float(np.linalg.norm(vectors[i].astype(np.float64) - vectors[j].astype(np.float64)))
        for i in range(20)
        for j in range(i + 1, 20)
    )
    assert distances == pytest.approx(all_pairs[:5], rel=1e-4)


def test_min_pairwise_histogram_is_capped_by_the_available_pairs():
    vectors = np.eye(3, dtype=np.float32)
    result = min_pairwise_z_i0(vectors, ["a", "b", "c"], histogram_n=100)
    assert len(result["smallest_pairs"]) == 3


def test_min_pairwise_reports_no_pairs_for_a_degenerate_set():
    result = min_pairwise_z_i0(np.zeros((1, 4), dtype=np.float32), ["only"])
    assert result["n_pairs_compared"] == 0
    assert result["min_distance"] is None and result["argmin_pair"] is None


def test_min_pairwise_rejects_mismatched_target_keys():
    with pytest.raises(BuildError):
        min_pairwise_z_i0(np.zeros((3, 4), dtype=np.float32), ["a", "b"])


def test_min_pairwise_accepts_flattened_z_i0_geometry():
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(4, 48 * 1 * 12 * 20)).astype(np.float16)
    result = min_pairwise_z_i0(vectors, [f"t{i}" for i in range(4)])
    assert result["n_windows"] == 4 and result["n_pairs_compared"] == 6
    assert result["min_distance"] > 0.0


# ----------------------------------------------------------------------------------
# 3. Target identity -- "different targets" must be an exact-content test.
# ----------------------------------------------------------------------------------


def test_target_key_is_content_addressed():
    z = np.random.default_rng(6).normal(size=(4, 3, 2)).astype(np.float16)
    assert target_key(z) == target_key(z.copy())
    other = z.copy()
    other[0, 0, 0] = np.float16(float(other[0, 0, 0]) + 1.0)
    assert target_key(z) != target_key(other)


def test_target_key_distinguishes_a_transposed_view_with_the_same_values():
    z = np.arange(6, dtype=np.float16).reshape(2, 3)
    assert target_key(z) != target_key(z.T)
