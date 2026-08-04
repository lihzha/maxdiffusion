"""exp_04 R1 — the inversion/replay sigma grid, pinned to hand-computed values.

Null-text inversion optimizes one null embedding *per sampler step*, so the sigma grid is part of
the cached-artifact contract: every ``nulls[25, 16, 4096]`` tensor is indexed by the step it was
optimized at, and a silently shifted grid invalidates every cached target without changing a shape.
This file pins the exact call the exp_04 arms make -- ``build_rollout_sigmas(25, 5.0, 0.0, 1.0)``,
the same grid the side-adapter train/eval path already uses -- against values derived by hand from
the closed form, never by re-running the implementation.

Closed form. ``build_rollout_sigmas`` takes ``linspace(sigma_max, sigma_min, N+1)[:-1]``, maps
``s -> shift*s / (1 + (shift-1)*s)``, and appends a literal 0.0. With N=25, shift=5, sigma_max=1.0,
sigma_min=0.0 the pre-shift value at index i is exactly ``(25-i)/25``, so

    sigma_i = 5*(25 - i) / (125 - 4*i)   for i in 0..24,      sigma_25 = 0.0

which is what ``_EXPECTED_SIGMAS`` spells out (i=24 -> 5/29 = 0.1724137931034483). The tolerance is
float32 round-off only: the implementation evaluates the map in float32, whose largest relative
deviation from the exact rational over this grid is 4.5e-7.

Accepted deviation (plan §8). The PyTorch reference (``third_party/Wan2.2/scripts``) starts its grid
at sigma_max = 0.999; this port starts at 1.0, the value the side-adapter path already trains and
evaluates with. Every exp_04 arm -- inversion, per-step optimization, replay, and all controls --
runs on this one grid, so the deviation is a constant shared by the method and its controls, never a
difference between them. It is registered in plan §8 and must not be "fixed" silently: changing it
invalidates every cached null.
"""

from __future__ import annotations

import numpy as np

from maxdiffusion.models.wan.side_adapter_wan import build_rollout_sigmas, rollout_timesteps_from_sigmas


_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX = 25, 5.0, 0.0, 1.0
_NUM_TRAIN_TIMESTEPS = 1000

# sigma_i = 5*(25 - i) / (125 - 4*i), i = 0..24; sigma_25 = 0.0 (the appended terminal sigma).
_EXPECTED_SIGMAS = np.array(
    [
        1.0,  # 125/125
        0.9917355371900827,  # 120/121
        0.9829059829059829,  # 115/117
        0.9734513274336283,  # 110/113
        0.963302752293578,  # 105/109
        0.9523809523809524,  # 100/105
        0.9405940594059405,  # 95/101
        0.9278350515463918,  # 90/97
        0.9139784946236559,  # 85/93
        0.898876404494382,  # 80/89
        0.8823529411764706,  # 75/85
        0.8641975308641975,  # 70/81
        0.8441558441558441,  # 65/77
        0.821917808219178,  # 60/73
        0.7971014492753623,  # 55/69
        0.7692307692307693,  # 50/65
        0.7377049180327869,  # 45/61
        0.7017543859649122,  # 40/57
        0.660377358490566,  # 35/53
        0.6122448979591837,  # 30/49
        0.5555555555555556,  # 25/45
        0.4878048780487805,  # 20/41
        0.40540540540540543,  # 15/37
        0.30303030303030304,  # 10/33
        0.1724137931034483,  # 5/29
        0.0,  # appended terminal sigma
    ],
    dtype=np.float64,
)


def test_expected_sigmas_transcription_matches_closed_form():
    """Guard the literals above against a typo -- closed form only, implementation untouched.

    ``rtol`` here is float64 round-off (a few ULP; the literals and the division below need not
    round identically), which is ~12 orders of magnitude tighter than any plausible transcription
    slip.
    """
    closed_form = np.array([5.0 * (25 - i) / (125 - 4 * i) for i in range(25)] + [0.0], dtype=np.float64)
    np.testing.assert_allclose(_EXPECTED_SIGMAS, closed_form, rtol=1e-15, atol=0.0)


def test_build_rollout_sigmas_matches_hand_computed_grid():
    sigmas = build_rollout_sigmas(_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX)

    assert sigmas.shape == (26,)
    assert sigmas.dtype == np.float32
    np.testing.assert_allclose(np.asarray(sigmas, dtype=np.float64), _EXPECTED_SIGMAS, rtol=1e-6, atol=1e-7)


def test_build_rollout_sigmas_endpoints_and_strict_descent():
    sigmas = np.asarray(build_rollout_sigmas(_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX))

    assert len(sigmas) == _STEPS + 1
    assert sigmas[0] == np.float32(1.0)
    assert sigmas[-1] == np.float32(0.0)
    assert np.all(np.diff(sigmas) < 0.0)
    # The one step the grid's tail hinges on: the last non-zero sigma is 5/29, not sigma_min.
    np.testing.assert_allclose(sigmas[-2], 0.1724137931034483, rtol=1e-6)


def test_rollout_timesteps_are_sigmas_times_num_train_timesteps():
    sigmas = build_rollout_sigmas(_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX)

    timesteps = np.asarray(rollout_timesteps_from_sigmas(sigmas, _NUM_TRAIN_TIMESTEPS))

    assert timesteps.shape == (_STEPS,)
    assert timesteps.dtype == np.float32
    np.testing.assert_array_equal(timesteps, np.asarray(sigmas)[:-1] * np.float32(_NUM_TRAIN_TIMESTEPS))
