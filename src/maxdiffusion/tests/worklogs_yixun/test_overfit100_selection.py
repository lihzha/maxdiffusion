"""CPU-only tests for the exp_02 episode-selection rules (cycle A, deliverable A2).

Covers the *pure* half of `maxdiffusion.data_preprocessing.build_overfit100_manifest`:
plan v4 D1 (seeded draw + acceptance predicate), D2 (instruction pick over the non-empty
texts only, keyed by `fold_in(selection_seed, episode_id)`), and D4's window arithmetic.

These are the rules the whole experiment's identity rests on -- which 100 DROID episodes
and which of their up-to-3 annotations become the memorization set -- so every branch is
pinned here before any network IO happens. No GCS, no ffprobe, no accelerator.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from maxdiffusion.data_preprocessing.build_overfit100_manifest import (
    MIN_FRAMES,
    N_EPISODES,
    candidate_order,
    decide_candidate,
    n_windows,
    nonempty_text_indices,
    pick_instruction_index,
)


def _annotation(success=1, texts=("pick up the mug",), episode_id=0):
    return {"episode_id": episode_id, "success": success, "texts": list(texts), "video_length": 88}


# ----------------------------------------------------------------------------------
# 1. D1 -- seeded candidate order: exact construction, determinism, seed sensitivity.
# ----------------------------------------------------------------------------------


def test_candidate_order_is_the_locked_construction():
    # The plan pins the draw to numpy's default_rng permutation over the full id range;
    # locking it here means a future refactor cannot silently reselect the dataset.
    expected = np.random.default_rng(0).permutation(N_EPISODES)
    np.testing.assert_array_equal(candidate_order(0), expected)


def test_candidate_order_is_a_permutation_of_all_episode_ids():
    order = candidate_order(0)
    assert order.shape == (N_EPISODES,)
    assert N_EPISODES == 69723
    np.testing.assert_array_equal(np.sort(order), np.arange(N_EPISODES))


def test_candidate_order_is_deterministic_per_seed():
    np.testing.assert_array_equal(candidate_order(0)[:64], candidate_order(0)[:64])


def test_candidate_order_differs_across_seeds():
    assert not np.array_equal(candidate_order(0)[:64], candidate_order(1)[:64])


# ----------------------------------------------------------------------------------
# 2. D1 -- acceptance decision matrix (reasons are the manifest's rejection vocabulary).
# ----------------------------------------------------------------------------------


def test_decide_missing_annotation():
    assert decide_candidate(None) == (False, "missing_annotation")


def test_decide_not_success():
    assert decide_candidate(_annotation(success=0)) == (False, "not_success")


def test_decide_no_nonempty_text():
    assert decide_candidate(_annotation(texts=("", "   ", "\n"))) == (False, "no_nonempty_text")


def test_decide_no_texts_key_at_all():
    assert decide_candidate({"episode_id": 3, "success": 1}) == (False, "no_nonempty_text")


def test_decide_missing_video():
    assert decide_candidate(_annotation(), video_exists=False) == (False, "missing_video")


def test_decide_too_short():
    assert decide_candidate(_annotation(), video_exists=True, nb_frames=MIN_FRAMES - 1) == (False, "too_short")


def test_decide_accepted_at_exact_minimum_length():
    assert decide_candidate(_annotation(), video_exists=True, nb_frames=MIN_FRAMES) == (True, "accepted")


def test_decide_accepted_full_probe():
    assert decide_candidate(_annotation(texts=("", "wipe the table")), video_exists=True, nb_frames=88) == (
        True,
        "accepted",
    )


def test_decide_is_provisional_before_the_video_probe():
    # Annotation-level pass with no probe results yet (the --dry-run path): provisional accept.
    assert decide_candidate(_annotation()) == (True, "accepted")


def test_decide_rejects_annotation_first_even_when_video_is_fine():
    assert decide_candidate(_annotation(success=0), video_exists=True, nb_frames=88) == (False, "not_success")


# ----------------------------------------------------------------------------------
# 3. D2 -- empty filter BEFORE the pick, and a stable pick keyed by (seed, episode_id).
# ----------------------------------------------------------------------------------


def test_nonempty_text_indices_uses_strip_semantics():
    assert nonempty_text_indices(["", "  ", "\t\n", "go"]) == [3]
    assert nonempty_text_indices(["a", "", "b"]) == [0, 2]
    assert nonempty_text_indices([]) == []


def test_pick_never_lands_on_a_whitespace_only_text():
    texts = ["", "   ", "put the banana in the box"]
    for episode_id in range(200):
        assert pick_instruction_index(0, episode_id, texts) == 2


def test_pick_only_lands_on_nonempty_indices_when_they_are_noncontiguous():
    texts = ["open the drawer", "  ", "close the drawer"]
    chosen = {pick_instruction_index(0, episode_id, texts) for episode_id in range(500)}
    assert chosen <= {0, 2}
    # Both reachable -- the pick is a real random choice, not a constant.
    assert chosen == {0, 2}


def test_pick_is_stable_regardless_of_call_order():
    texts = ["a", "b", "c"]
    first = [pick_instruction_index(0, ep, texts) for ep in (11, 22, 33)]
    # Interleave unrelated picks; a global-RNG implementation would drift here.
    _ = [pick_instruction_index(0, ep, ["x", "y"]) for ep in range(50)]
    second = [pick_instruction_index(0, ep, texts) for ep in (33, 11, 22)]
    assert first == [second[1], second[2], second[0]]


def test_pick_uses_fold_in_over_the_selection_seed():
    texts = ["a", "b", "c"]
    episode_id = 4242
    key = jax.random.fold_in(jax.random.key(0), episode_id)
    expected = int(jax.random.randint(key, (), 0, len(texts)))
    assert pick_instruction_index(0, episode_id, texts) == expected


def test_pick_depends_on_the_seed():
    texts = ["a", "b", "c", "d", "e", "f", "g", "h"]
    picks0 = [pick_instruction_index(0, ep, texts) for ep in range(64)]
    picks1 = [pick_instruction_index(1, ep, texts) for ep in range(64)]
    assert picks0 != picks1


def test_pick_on_single_nonempty_text_is_that_text():
    assert pick_instruction_index(0, 7, ["only one"]) == 0


def test_pick_raises_when_all_texts_are_empty():
    with pytest.raises(ValueError):
        pick_instruction_index(0, 7, ["", "  "])


# ----------------------------------------------------------------------------------
# 4. D4 -- window arithmetic: starts 0, 4, 8, ... with s + 33 <= nb_frames.
# ----------------------------------------------------------------------------------


def test_n_windows_matches_the_plan_examples():
    assert n_windows(88) == 14  # manifest-probe episode 0
    assert n_windows(128) == 24  # manifest-probe episode 1
    assert n_windows(33) == 1  # exactly one window
    assert n_windows(36) == 1  # not yet enough for a second start
    assert n_windows(37) == 2  # start 4 fits: 4 + 33 = 37


def test_n_windows_is_the_count_of_valid_starts():
    for nb_frames in range(MIN_FRAMES, 200):
        expected = len([s for s in range(0, nb_frames, 4) if s + MIN_FRAMES <= nb_frames])
        assert n_windows(nb_frames) == expected


def test_n_windows_rejects_too_short_clips():
    with pytest.raises(ValueError):
        n_windows(MIN_FRAMES - 1)
