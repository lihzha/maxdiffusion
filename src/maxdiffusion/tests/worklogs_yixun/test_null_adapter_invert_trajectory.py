"""exp_04 R2 — ``invert_trajectory``: the clean -> noise reverse-Euler recurrence (plan §3 step 1).

Inversion is the step that decides *what* the null embeddings are later optimized against: traj[i] is
the pivot at sigma_i, and every optimization target, cached artifact and replay check is indexed by
it. A recurrence that is off by one index, or that evaluates the velocity at the wrong point, still
produces a plausible-looking trajectory -- it just silently means something else. So this file does
not assert a round trip (which any self-consistent-but-wrong recurrence would also satisfy). It pins
the recurrence itself with oracles whose exact output is known analytically:

- a **constant** oracle, where the recurrence telescopes to ``z_video + sigma_i * c`` because the
  grid ends at sigma_N = 0 -- so every index is checked against a closed form, not against itself;
- a **linear** oracle ``a*z + b``, compared elementwise against a literal Python loop written out in
  this file (the scan must be an implementation detail, not a change of meaning);
- a **sigma-reporting** oracle, which returns the sigma it was handed, so the per-step increment
  ``(sigma_i - sigma_{i+1}) * sigma_i`` pins both the index (sigma_i, not sigma_{i+1}) and the sign;
- a **timestep-structure** oracle, which reports whether the per-token timestep it received had
  frame-0 tokens zeroed and all remaining tokens equal to sigma_i * 1000.

Parity (plan §8): the reference is ``compute_inversion_trajectory`` in
``third_party/Wan2.2/scripts/embedding_search.py:522-572`` (submodule pin f370228) -- same
``traj[N] = pin(z_video)``, same descending fill, same ``v`` evaluated at ``traj[i+1]`` with sigma_i,
same ``(sigma_i - sigma_{i+1}) * v`` step, same pin after every write. The reference pins with the
mask form ``(1-mask2)*z_I0 + mask2*latent``; this port uses ``apply_first_frame_pin``'s equivalent
frame-0 slice assignment, and its CFG mixing lives in the caller's ``velocity_fn`` (this function
takes one velocity, at inversion w=1 there is nothing to mix).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import NUM_TRAIN_TIMESTEPS, invert_trajectory
from maxdiffusion.models.wan.side_adapter_wan import build_rollout_sigmas


# The real geometry (48 x 9 x 12 x 20) shrunk to what a laptop runs: what is under test is the
# recurrence and the token layout, neither of which depends on the sizes.
_B, _C, _F, _H, _W = 2, 4, 3, 4, 6
_TOKENS_PER_FRAME = (_H // 2) * (_W // 2)
_SEQ_LEN = _F * _TOKENS_PER_FRAME
_SMALL_SIGMAS = (1.0, 0.6, 0.3, 0.0)
# Hardcoded, not imported: the expectations below must not be able to follow the module's
# constant if it drifts. This is the WAN scheduler's training horizon
# (``scheduler.config.num_train_timesteps``), and every cached pivot is indexed by it.
_NUM_TRAIN_TIMESTEPS = 1000
_ROLLOUT_SIGMAS = (25, 5.0, 0.0, 1.0)

# Values a numeric comparison cannot separate (+0/-0) or that a lossy copy would destroy; planted in
# the pin anchor, whose bits must survive every step. Same discipline as R1.
_EDGE_VALUES = (0.0, -0.0, np.float32(np.inf), np.float32(np.nan), np.float32(1e-45), -1.0)


def _f32_bits(x):
    """Raw float32 bit patterns: +0 != -0 and NaN == NaN. (Deliberately restated per test file.)"""
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x).astype(jnp.float32), jnp.uint32))


def _inputs(batch=_B, seed=0, dtype=jnp.float32):
    video_key, image_key = jax.random.split(jax.random.PRNGKey(seed))
    z_video = jax.random.normal(video_key, (batch, _C, _F, _H, _W), dtype=jnp.float32)
    z_i0 = jax.random.normal(image_key, (batch, _C, 1, _H, _W), dtype=jnp.float32)
    return z_video.astype(dtype), z_i0.astype(dtype)


def _sigmas(values=_SMALL_SIGMAS):
    return jnp.asarray(values, dtype=jnp.float32)


def _pin(z, z_i0):
    """The pin restated here, so the expectations do not lean on the module's own helper."""
    return z.at[:, :, :1].set(z_i0[:, :, :1])


def _constant_oracle(c):
    return lambda z, timestep_2d: jnp.full_like(z, c)


def _linear_oracle(a, b):
    return lambda z, timestep_2d: a * z + b


def _sigma_reporting_oracle(z, timestep_2d):
    """v = the sigma this call was handed (recovered from a future-frame token)."""
    sigma = (timestep_2d[:, -1] / _NUM_TRAIN_TIMESTEPS).astype(jnp.float32)
    return jnp.zeros_like(z) + sigma[:, None, None, None, None]


def _timestep_structure_oracle(z, timestep_2d):
    """v = [is the per-token timestep well-formed?, what value do the future tokens carry?]."""
    head = timestep_2d[:, :_TOKENS_PER_FRAME]
    tail = timestep_2d[:, _TOKENS_PER_FRAME:]
    well_formed = jnp.float32(timestep_2d.shape == (z.shape[0], _SEQ_LEN))
    well_formed = well_formed * jnp.all(head == 0.0).astype(jnp.float32)
    well_formed = well_formed * jnp.all(tail == tail[:, :1]).astype(jnp.float32)
    payload = jnp.zeros_like(z)
    payload = payload.at[:, 0].set(well_formed)
    return payload.at[:, 1].set(tail[0, 0])


def _increments(traj, sigmas):
    """Recover v_i from the trajectory: (traj[i] - traj[i+1]) / (sigma_i - sigma_{i+1})."""
    sigmas = np.asarray(sigmas, dtype=np.float64)
    traj = np.asarray(traj, dtype=np.float64)
    return [(traj[i] - traj[i + 1]) / (sigmas[i] - sigmas[i + 1]) for i in range(len(sigmas) - 1)]


def test_num_train_timesteps_matches_the_sampler_schedule():
    assert NUM_TRAIN_TIMESTEPS == _NUM_TRAIN_TIMESTEPS


@pytest.mark.parametrize("grid", ["small", "rollout"])
def test_constant_velocity_telescopes_to_the_closed_form(grid):
    """With v = c, traj[i] = z_video + sigma_i * c on every non-pinned entry (sigma_N = 0)."""
    z_video, z_i0 = _inputs()
    sigmas = _sigmas() if grid == "small" else build_rollout_sigmas(*_ROLLOUT_SIGMAS)
    c = 0.37

    traj = invert_trajectory(_constant_oracle(c), z_video, z_i0, sigmas)

    assert traj.shape == (len(sigmas), _B, _C, _F, _H, _W)
    pinned = _pin(z_video, z_i0)
    for i in range(len(sigmas)):
        expected = np.asarray(pinned[:, :, 1:]) + float(sigmas[i]) * c
        np.testing.assert_allclose(np.asarray(traj[i, :, :, 1:]), expected, rtol=1e-5, atol=1e-6)


def test_scan_matches_a_literal_python_loop_for_a_linear_oracle():
    """The scan is an implementation detail: elementwise equal to the loop written out here.

    Not bitwise: XLA contracts the scanned body's multiply-add into an FMA, which the eager loop
    evaluates as two rounded operations. The measured gap over the full 25-step grid is <= 2 ULP
    (max relative 1.9e-8); the tolerance below is ~50x that, and still ~7 orders of magnitude tighter
    than any index, sign or evaluation-point error, all of which move entries by O(1). Frame 0 is
    asserted bitwise separately (see the pin test).
    """
    z_video, z_i0 = _inputs(seed=3)
    sigmas = build_rollout_sigmas(*_ROLLOUT_SIGMAS)
    a, b = -0.31, 0.17

    traj = invert_trajectory(_linear_oracle(a, b), z_video, z_i0, sigmas)

    steps = len(sigmas) - 1
    expected = [None] * (steps + 1)
    current = _pin(z_video, z_i0)
    expected[steps] = current
    for i in range(steps - 1, -1, -1):
        velocity = a * current + b
        current = _pin(current + (sigmas[i] - sigmas[i + 1]) * velocity, z_i0)
        expected[i] = current

    for i in range(steps + 1):
        np.testing.assert_allclose(np.asarray(traj[i]), np.asarray(expected[i]), rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(_f32_bits(traj[i, :, :, :1]), _f32_bits(expected[i][:, :, :1]))


def test_each_step_uses_its_own_sigma_with_the_forward_sign():
    """Increment i must be (sigma_i - sigma_{i+1}) * sigma_i -- pins the index and the direction."""
    z_video, z_i0 = _inputs(seed=4)
    sigmas = build_rollout_sigmas(*_ROLLOUT_SIGMAS)

    traj = invert_trajectory(_sigma_reporting_oracle, z_video, z_i0, sigmas)

    recovered = _increments(traj, sigmas)
    for i, velocity in enumerate(recovered):
        # Every non-pinned entry carries sigma_i; sigma_{i+1} or a flipped sign both fail here.
        np.testing.assert_allclose(velocity[:, :, 1:], float(sigmas[i]), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("grid", ["small", "rollout"])
def test_per_token_timestep_zeroes_frame_zero_and_carries_sigma_times_num_train_timesteps(grid):
    z_video, z_i0 = _inputs(seed=5)
    sigmas = _sigmas() if grid == "small" else build_rollout_sigmas(*_ROLLOUT_SIGMAS)

    traj = invert_trajectory(_timestep_structure_oracle, z_video, z_i0, sigmas)

    recovered = _increments(traj, sigmas)
    for i, reported in enumerate(recovered):
        np.testing.assert_allclose(reported[:, 0, 1:], 1.0, rtol=1e-4)
        np.testing.assert_allclose(reported[:, 1, 1:], float(sigmas[i]) * _NUM_TRAIN_TIMESTEPS, rtol=1e-4, atol=1e-4)


def test_frame_zero_holds_the_anchor_bitwise_at_every_step():
    """Including signed zeros, NaN, inf and a subnormal: the pin is a copy, not an arithmetic blend."""
    z_video, z_i0 = _inputs(seed=6)
    edges = np.asarray(_EDGE_VALUES, dtype=np.float32)
    anchor = np.asarray(z_i0).copy()
    anchor[:, 0, 0, 0, : edges.size] = edges
    z_i0 = jnp.asarray(anchor)

    traj = invert_trajectory(_linear_oracle(0.4, -0.2), z_video, z_i0, _sigmas())

    for i in range(len(_SMALL_SIGMAS)):
        np.testing.assert_array_equal(_f32_bits(traj[i, :, :, :1]), _f32_bits(z_i0[:, :, :1]))


def test_trajectory_is_float32_even_for_bfloat16_inputs_and_velocities():
    z_video, z_i0 = _inputs(seed=7, dtype=jnp.bfloat16)

    traj = invert_trajectory(lambda z, t: (0.5 * z).astype(jnp.bfloat16), z_video, z_i0, _sigmas())

    assert traj.dtype == jnp.float32


def test_example_zero_is_independent_of_the_rest_of_the_batch():
    z_video, z_i0 = _inputs(batch=2, seed=8)
    oracle = _linear_oracle(0.23, -0.11)
    sigmas = build_rollout_sigmas(*_ROLLOUT_SIGMAS)

    batched = invert_trajectory(oracle, z_video, z_i0, sigmas)
    alone = invert_trajectory(oracle, z_video[:1], z_i0[:1], sigmas)

    np.testing.assert_array_equal(_f32_bits(batched[:, :1]), _f32_bits(alone))


@pytest.mark.parametrize(
    "sigmas, message",
    [
        ((0.0, 0.5, 1.0), "strictly descending"),  # ascending
        ((1.0, 0.5, 0.2), "must end at 0.0"),  # does not reach sigma = 0
        ((1.0, 0.5, 0.5, 0.0), "strictly descending"),  # not strictly descending
        ((1.0,), "at least one step"),  # single entry: no steps
        (1.0, "must be 1-D"),  # scalar
        (((1.0, 0.5), (0.5, 0.0)), "must be 1-D"),  # rank 2
        # Non-finite grids: +inf is "strictly descending" and would otherwise be accepted, after
        # which every step's arithmetic is invalid. NaN would trip the monotonicity check and -inf
        # the tail check, so matching on the finiteness message also pins that this guard runs first.
        ((np.inf, 1.0, 0.0), "must be finite"),
        ((1.0, np.nan, 0.0), "must be finite"),
        ((1.0, 0.5, -np.inf), "must be finite"),
    ],
)
def test_rejects_malformed_sigma_grids(sigmas, message):
    """Matched on the message: a downstream broadcasting error also raises ValueError, and a guard
    that lets a bad grid through to be caught by luck is not a guard."""
    z_video, z_i0 = _inputs()

    with pytest.raises(ValueError, match=message):
        invert_trajectory(_constant_oracle(1.0), z_video, z_i0, jnp.asarray(sigmas, dtype=jnp.float32))


@pytest.mark.parametrize(
    "video_shape, image_shape, message",
    [
        ((_B, _C, _F, _H, _W), (_B, _C, 2, _H, _W), "latent frames"),  # anchor neither 1 nor F frames
        ((_B, _C, _F, _H, _W), (_B + 1, _C, 1, _H, _W), "inconsistent with"),  # batch mismatch
        ((_B, _C, _F, _H, _W), (_B, _C + 1, 1, _H, _W), "inconsistent with"),  # channel mismatch
        ((_B, _C, _F, _H, _W), (_B, _C, 1, _H, _W + 1), "inconsistent with"),  # spatial mismatch
        ((_C, _F, _H, _W), (_C, 1, _H, _W), "z_video must be"),  # unbatched rank 4
    ],
)
def test_rejects_inconsistent_latent_shapes(video_shape, image_shape, message):
    """Also matched on the message -- see the sigma-grid test above for why."""
    z_video = jnp.zeros(video_shape, dtype=jnp.float32)
    z_i0 = jnp.zeros(image_shape, dtype=jnp.float32)

    with pytest.raises(ValueError, match=message):
        invert_trajectory(_constant_oracle(1.0), z_video, z_i0, _sigmas())


@pytest.mark.parametrize(
    "velocity_shape",
    [
        (),  # a scalar velocity broadcasts over everything
        (_C, _F, _H, _W),  # batchless: one example's velocity applied to all of them
        (1, _C, _F, _H, _W),  # singleton batch: example 0's velocity applied to example 1 as well
    ],
    ids=["scalar", "batchless", "singleton_batch"],
)
def test_rejects_velocity_outputs_that_do_not_match_the_latent_shape(velocity_shape):
    """A mis-shaped velocity broadcasts silently and corrupts every pivot plausibly.

    None of these raise on their own -- ``current + delta * v`` is perfectly legal NumPy-style
    broadcasting -- so the trajectory would come out the right shape, look reasonable, and mean
    something else. The seam has to fail closed instead.
    """
    z_video, z_i0 = _inputs(batch=_B)

    with pytest.raises(ValueError, match="velocity_fn returned shape"):
        invert_trajectory(lambda z, t: jnp.zeros(velocity_shape, dtype=jnp.float32), z_video, z_i0, _sigmas())


def test_accepts_a_velocity_that_matches_the_latent_shape_exactly():
    """The complement of the rejection above: the exact shape is not merely tolerated, it is used."""
    z_video, z_i0 = _inputs(batch=_B)

    traj = invert_trajectory(lambda z, t: jnp.full(z.shape, 0.25, dtype=jnp.float32), z_video, z_i0, _sigmas())

    assert traj.shape == (len(_SMALL_SIGMAS), _B, _C, _F, _H, _W)
