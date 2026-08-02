"""CPU-only tests for the exp_02 window enumeration (cycle B, deliverable B1).

Covers the pure window half of `maxdiffusion.data_preprocessing.build_overfit100_dataset`:
plan v4 D4's "windows = 33 consecutive frames at starts s = 0, 4, 8, ... while s + 33 <=
nb_frames", the frame slice each start denotes, the record `name` that ties a built window
back to the exp_01 cache convention (`ep<ID>_v0_s<START>`), and the per-episode count
assertions that make a silently short/long decode abort the build.

The window grid IS the dataset's identity (1,629 windows over 100 episodes, fixed in the
committed manifest), so every edge is pinned here before any MP4 is decoded. No network,
no ffmpeg, no accelerator.
"""

from __future__ import annotations

import numpy as np
import pytest

from maxdiffusion.data_preprocessing.build_overfit100_dataset import (
    WINDOW_FRAMES,
    WINDOW_STRIDE,
    BuildError,
    check_frame_count,
    check_window_count,
    slice_window,
    window_frame_range,
    window_name,
    window_starts,
)
from maxdiffusion.data_preprocessing.build_overfit100_manifest import MIN_FRAMES, n_windows
from maxdiffusion.data_preprocessing.extract_v1_fixture import TARGET_NAMES

# ----------------------------------------------------------------------------------
# 1. D4 -- the window grid itself.
# ----------------------------------------------------------------------------------


def test_window_geometry_constants_match_the_plan():
    assert (WINDOW_FRAMES, WINDOW_STRIDE) == (33, 4)
    assert WINDOW_FRAMES == MIN_FRAMES


def test_window_starts_at_the_plan_edges():
    assert window_starts(33) == [0]  # exactly one window
    assert window_starts(36) == [0]  # 4 + 33 = 37 > 36, so no second start
    assert window_starts(37) == [0, 4]  # the first frame count that admits a second start


def test_window_starts_counts_match_the_manifest_examples():
    assert len(window_starts(88)) == 14
    assert len(window_starts(427)) == 99
    assert len(window_starts(133)) == 26  # manifest episode_index 0 (ep 25189)


def test_window_starts_agrees_with_the_cycle_a_count_everywhere():
    # `n_windows` (cycle A) sized the manifest; the enumeration used to BUILD must not
    # disagree with the count that was committed, or `n_windows` asserts would be vacuous.
    for nb_frames in range(MIN_FRAMES, 512):
        starts = window_starts(nb_frames)
        assert len(starts) == n_windows(nb_frames)
        assert starts == list(range(0, len(starts) * WINDOW_STRIDE, WINDOW_STRIDE))
        assert starts[-1] + WINDOW_FRAMES <= nb_frames
        assert starts[-1] + WINDOW_STRIDE + WINDOW_FRAMES > nb_frames  # maximal


def test_window_starts_rejects_a_clip_shorter_than_one_window():
    with pytest.raises(ValueError):
        window_starts(MIN_FRAMES - 1)


# ----------------------------------------------------------------------------------
# 2. D4 -- the frame slice a start denotes.
# ----------------------------------------------------------------------------------


def test_window_frame_range_is_a_half_open_33_frame_span():
    assert window_frame_range(0) == (0, 33)
    assert window_frame_range(4) == (4, 37)
    assert window_frame_range(96) == (96, 129)


def test_slice_window_takes_exactly_the_named_frames():
    frames = np.arange(50 * 2 * 3 * 3, dtype=np.uint8).reshape(50, 2, 3, 3)
    window = slice_window(frames, 4)
    assert window.shape == (33, 2, 3, 3)
    np.testing.assert_array_equal(window, frames[4:37])


def test_slice_window_refuses_a_start_that_runs_past_the_clip():
    frames = np.zeros((36, 2, 3, 3), dtype=np.uint8)
    assert slice_window(frames, 0).shape[0] == 33
    with pytest.raises(BuildError):
        slice_window(frames, 4)  # 4 + 33 = 37 > 36


# ----------------------------------------------------------------------------------
# 3. D6 -- record names follow the exp_01 cache convention exactly.
# ----------------------------------------------------------------------------------


def test_window_name_is_the_cache_convention():
    assert window_name(25189, 4) == "ep25189_v0_s00004"
    assert window_name(0, 0) == "ep0_v0_s00000"


def test_window_name_zero_pads_the_start_to_five_digits():
    assert window_name(7, 4) == "ep7_v0_s00004"
    assert window_name(7, 1234) == "ep7_v0_s01234"
    assert window_name(7, 12345) == "ep7_v0_s12345"


def test_window_name_does_not_truncate_a_six_digit_start():
    assert window_name(7, 123456) == "ep7_v0_s123456"


def test_window_name_reproduces_the_v1_fixture_names():
    # The V1 fixture windows are ep 0, view 0, starts 0/4/8 -- the names the gate looks up.
    assert tuple(window_name(0, start) for start in (0, 4, 8)) == TARGET_NAMES


def test_window_name_honours_a_non_default_view():
    assert window_name(3, 8, view=2) == "ep3_v2_s00008"


# ----------------------------------------------------------------------------------
# 4. Per-episode count assertions (a short/long decode must abort, never shrink the set).
# ----------------------------------------------------------------------------------


def test_check_frame_count_passes_on_the_manifest_value():
    check_frame_count(25189, decoded=133, expected=133)


def test_check_frame_count_raises_on_drift():
    with pytest.raises(BuildError) as excinfo:
        check_frame_count(25189, decoded=132, expected=133)
    assert "25189" in str(excinfo.value)
    assert "132" in str(excinfo.value) and "133" in str(excinfo.value)


def test_check_window_count_passes_on_the_manifest_value():
    check_window_count(25189, actual=26, expected=26)


def test_check_window_count_raises_on_drift():
    with pytest.raises(BuildError) as excinfo:
        check_window_count(25189, actual=25, expected=26)
    assert "25189" in str(excinfo.value)


def test_decoded_frames_drive_the_same_window_count_as_the_manifest():
    # End-to-end of the two asserts as the builder uses them: manifest nb_frames -> starts.
    nb_frames = 133
    frames = np.zeros((nb_frames, 4, 4, 3), dtype=np.uint8)
    check_frame_count(25189, decoded=len(frames), expected=nb_frames)
    starts = window_starts(nb_frames)
    check_window_count(25189, actual=len(starts), expected=26)
    assert all(slice_window(frames, start).shape[0] == WINDOW_FRAMES for start in starts)
