"""CPU-only core tests for the full-FT validation-loss evaluator (exp_01 Part II).

Cycle A "val-loss-core" under strict TDD. Covers the reviewable, weight-free core
of the held-out one-step-loss instrument (Query 8 / plan Part II, findings F1-F3):

* ``masked_velocity_mse_per_example`` -- the per-example twin of the scalar training
  objective, needed so the evaluator can report a mean AND a sample stderr over the
  val set. Characterized against the untouched scalar ``masked_velocity_mse``.
* ``per_example_rng`` -- deterministic ``(t_idx, eps)`` per DATASET POSITION so the
  loss curve across checkpoints is a pure model effect (F1).
* ``plan_batches`` -- exactly-once, position-ordered coverage with a masked padded
  tail (F2).
* ``aggregate`` -- validity-masked mean / sample-stderr (ddof=1) / count with a hard
  count assertion (F2).

CPU-only: no model weights, no TPU. The evaluator main (state build, restore loop,
output writers) is cycle B and is NOT exercised here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.eval_wan_full_ft_val_loss import aggregate, per_example_rng, plan_batches
from maxdiffusion.models.wan.side_adapter_wan import (
    masked_velocity_mse,
    masked_velocity_mse_per_example,
)

# Fixed shapes: batch B, channels C, latent frames F, height H, width W. B>1 so the
# per-example vector is non-trivial; F>1 so there are non-frame-0 (unmasked) frames.
_B, _C, _F, _H, _W = 4, 3, 4, 5, 6


def _fixed_vt(seed: int):
    """Deterministic ``(v_pred, v_target)`` float32 pair, both ``[B, C, F, H, W]``."""
    kp, kt = jax.random.split(jax.random.key(seed))
    v_pred = jax.random.normal(kp, (_B, _C, _F, _H, _W), dtype=jnp.float32)
    v_target = jax.random.normal(kt, (_B, _C, _F, _H, _W), dtype=jnp.float32)
    return v_pred, v_target


# ----------------------------------------------------------------------------------
# 1. masked_velocity_mse_per_example -- characterized against the scalar helper (F3).
# ----------------------------------------------------------------------------------


def test_per_example_shape_and_dtype():
    v_pred, v_target = _fixed_vt(seed=1)
    per_ex = masked_velocity_mse_per_example(v_pred, v_target)
    assert per_ex.shape == (_B,)
    assert per_ex.dtype == jnp.float32


def test_per_example_element_matches_scalar_on_b1_slice():
    # DEVIATION from plan F3's "bitwise per element" (documented in the cycle report):
    # plan F3 assumed the per-example path reproduces the scalar helper's reduction
    # TREE per row, giving bitwise equality. Empirically it does NOT on the XLA CPU
    # backend -- the frozen scalar helper reduces a standalone rank-5 [1,C,F,H,W],
    # while ANY batched/mapped reduction (axis-reduce, vmap, lax.map -- all probed)
    # differs by ~1 ULP (~2.4e-7). This is exactly Codex's ORIGINAL F3 caution that
    # "bitwise equality is not guaranteed by JAX/XLA". We therefore keep the clean,
    # efficient, independent vectorized helper (mask from v_target) and characterize
    # it against the scalar helper on each B=1 slice with a TIGHT (1-ULP-level)
    # closeness, not bitwise. The scalar training objective stays untouched.
    v_pred, v_target = _fixed_vt(seed=2)
    per_ex = np.asarray(masked_velocity_mse_per_example(v_pred, v_target))
    # The per-example losses must genuinely differ across rows (else the test is
    # vacuous); random inputs guarantee this.
    assert len({float(x) for x in per_ex}) == _B
    scalars = np.stack([np.asarray(masked_velocity_mse(v_pred[i : i + 1], v_target[i : i + 1], 1)) for i in range(_B)])
    np.testing.assert_allclose(per_ex, scalars, rtol=1e-6)
    # Pin that the residual is at the float32 reduction-order (~1 ULP) level, i.e. a
    # numeric-noise difference, not an algorithmic (mask/normalization) error.
    assert float(np.max(np.abs(per_ex - scalars))) < 1e-5


def test_per_example_mean_matches_batch_scalar_allclose():
    # The vector MEAN equals the batch scalar ALGEBRAICALLY, but NOT bit-for-bit:
    # the per-example path reduces (C,F,H,W) per row then means over B, whereas the
    # scalar helper does one global sum / (sum(mask)*B). XLA reduction trees differ,
    # so this is a tight allclose, deliberately not an exact-equality assertion.
    v_pred, v_target = _fixed_vt(seed=3)
    vector_mean = jnp.mean(masked_velocity_mse_per_example(v_pred, v_target))
    batch_scalar = masked_velocity_mse(v_pred, v_target, _B)
    np.testing.assert_allclose(np.asarray(vector_mean), np.asarray(batch_scalar), rtol=1e-6)


def test_per_example_invariant_to_frame0_perturbation():
    # Frame 0 is pinned to the image condition and masked out; a huge perturbation
    # of frame 0 in both v_pred and v_target must leave every element unchanged.
    v_pred, v_target = _fixed_vt(seed=5)
    base = masked_velocity_mse_per_example(v_pred, v_target)
    kp, kt = jax.random.split(jax.random.key(31))
    pred_pert = jax.random.normal(kp, (_B, _C, 1, _H, _W), dtype=jnp.float32) * 1000.0
    tgt_pert = jax.random.normal(kt, (_B, _C, 1, _H, _W), dtype=jnp.float32) * 1000.0
    v_pred_p = v_pred.at[:, :, :1].add(pred_pert)
    v_target_p = v_target.at[:, :, :1].add(tgt_pert)
    perturbed = masked_velocity_mse_per_example(v_pred_p, v_target_p)
    np.testing.assert_array_equal(np.asarray(base), np.asarray(perturbed))


def test_per_example_rejects_shape_mismatch_like_scalar():
    # Same shape-mismatch contract as the scalar helper: a broadcastable-but-
    # malformed v_pred is rejected, never silently masked against v_target's shape.
    v_target = jnp.ones((_B, _C, _F, _H, _W), dtype=jnp.float32)
    v_pred = jnp.ones((_B, _C, 1, _H, _W), dtype=jnp.float32)
    with pytest.raises(ValueError, match="shape"):
        masked_velocity_mse_per_example(v_pred, v_target)
    # Parity: the scalar helper rejects the same inputs.
    with pytest.raises(ValueError, match="shape"):
        masked_velocity_mse(v_pred, v_target, _B)


# ----------------------------------------------------------------------------------
# 2. per_example_rng -- deterministic, position-keyed (t, eps) (F1).
# ----------------------------------------------------------------------------------

_RNG_STEPS = 25
_RNG_SHAPE = (_C, _F, _H, _W)


def test_rng_same_seed_position_bitwise_identical_and_order_independent():
    t1, e1 = per_example_rng(0, 7, _RNG_STEPS, _RNG_SHAPE)
    t2, e2 = per_example_rng(0, 7, _RNG_STEPS, _RNG_SHAPE)
    np.testing.assert_array_equal(np.asarray(t1), np.asarray(t2))
    np.testing.assert_array_equal(np.asarray(e1), np.asarray(e2))
    # Interleaving a different position must not perturb the (seed, position) draw --
    # the function is stateless / order-independent.
    _ = per_example_rng(0, 999, _RNG_STEPS, _RNG_SHAPE)
    t3, e3 = per_example_rng(0, 7, _RNG_STEPS, _RNG_SHAPE)
    np.testing.assert_array_equal(np.asarray(t1), np.asarray(t3))
    np.testing.assert_array_equal(np.asarray(e1), np.asarray(e3))


def test_rng_distinct_positions_distinct_eps_and_t_in_range():
    epses = []
    for p in range(32):
        t, e = per_example_rng(0, p, _RNG_STEPS, _RNG_SHAPE)
        assert t.dtype == jnp.int32
        assert 0 <= int(t) < _RNG_STEPS  # t within [0, num_steps)
        epses.append(np.asarray(e))
    # Distinct positions -> distinct noise (statistically: pairwise not allclose).
    for a in range(6):
        for b in range(a + 1, 6):
            assert not np.allclose(epses[a], epses[b])


def test_rng_eps_shape_exact_and_t_scalar():
    shape = (7, 2, 5)
    t, e = per_example_rng(3, 11, 10, shape)
    assert e.shape == shape  # UNBATCHED example shape, exactly
    assert e.dtype == jnp.float32
    assert t.shape == ()
    assert t.dtype == jnp.int32


def test_rng_seed_changes_draw():
    _, e_seed0 = per_example_rng(0, 5, _RNG_STEPS, _RNG_SHAPE)
    _, e_seed1 = per_example_rng(1, 5, _RNG_STEPS, _RNG_SHAPE)
    assert not np.allclose(np.asarray(e_seed0), np.asarray(e_seed1))


def test_rng_num_steps_nonpositive_raises():
    with pytest.raises(ValueError, match="num_steps"):
        per_example_rng(0, 0, 0, (2, 2))
    with pytest.raises(ValueError, match="num_steps"):
        per_example_rng(0, 0, -3, (2, 2))


def test_rng_generic_num_steps_bounds_and_coverage():
    # Strengthen-phase F1 (cycle-A review): the earlier non-25 call sites checked
    # only shape/dtype, so a mutant hard-coding maxval=25 inside per_example_rng
    # survived. Deterministic non-25 contract (fixed seed -> fully deterministic
    # draws): with num_steps=3 over 64 positions, every t must lie in [0, 3) -- a
    # hard-coded maxval=25 sends ~88% of these draws out of range -- and the drawn
    # set must be exactly {0, 1, 2}, which also catches off-by-one bounds
    # (maxval=num_steps-1 can never draw 2; minval=1 can never draw 0).
    num_steps = 3
    draws = [int(per_example_rng(0, p, num_steps, (1,))[0]) for p in range(64)]
    assert all(0 <= t < num_steps for t in draws)
    assert set(draws) == {0, 1, 2}


# ----------------------------------------------------------------------------------
# 3. plan_batches -- exactly-once, position-ordered coverage with a masked tail (F2).
# ----------------------------------------------------------------------------------


def test_plan_batches_exact_multiple():
    plan = plan_batches(16, 8)
    assert len(plan) == 2
    for positions, validity in plan:
        assert len(positions) == 8
        assert len(validity) == 8
        assert all(validity)
    valid_positions = [p for positions, validity in plan for p, v in zip(positions, validity) if v]
    assert valid_positions == list(range(16))
    assert len(set(valid_positions)) == 16


def test_plan_batches_ragged_37_8():
    plan = plan_batches(37, 8)
    assert len(plan) == 5  # 4 full batches + 1 padded tail
    last_positions, last_validity = plan[-1]
    assert last_validity == [True, True, True, True, True, False, False, False]
    # Padded slots repeat the LAST REAL position (36), never a fresh/aliased index.
    assert last_positions == [32, 33, 34, 35, 36, 36, 36, 36]
    valid_positions = [p for positions, validity in plan for p, v in zip(positions, validity) if v]
    assert valid_positions == list(range(37))  # union of valid positions == range(total)
    assert len(set(valid_positions)) == 37  # no dupes among valid positions


def test_plan_batches_total_less_than_batch():
    plan = plan_batches(5, 8)
    assert len(plan) == 1
    positions, validity = plan[0]
    assert positions == [0, 1, 2, 3, 4, 4, 4, 4]
    assert validity == [True] * 5 + [False] * 3


def test_plan_batches_degenerate_raises():
    with pytest.raises(ValueError, match="total"):
        plan_batches(0, 8)
    with pytest.raises(ValueError, match="total"):
        plan_batches(-1, 8)
    with pytest.raises(ValueError, match="batch"):
        plan_batches(10, 0)
    with pytest.raises(ValueError, match="batch"):
        plan_batches(10, -2)


# ----------------------------------------------------------------------------------
# 4. aggregate -- validity-masked mean / sample-stderr / count (F2).
# ----------------------------------------------------------------------------------


def test_aggregate_golden_padded_excludes_duplicate_from_mean_and_stderr():
    # Golden: the unpadded real losses. Padded: append a duplicate carrying a
    # DIFFERENT (huge) value, masked invalid. If exclusion touched only the count
    # (not the mean/stderr), the padded mean/stderr would be wrong -- so exact
    # equality proves the duplicate is excluded from BOTH statistics, not merely
    # uncounted.
    real = [1.0, 2.0, 3.0, 4.0, 5.0]
    golden = aggregate(real, [True] * 5, 5)

    padded_losses = real + [999.0]
    padded_validity = [True] * 5 + [False]
    padded = aggregate(padded_losses, padded_validity, 5)

    assert padded["n"] == golden["n"] == 5
    assert padded["mean_loss"] == golden["mean_loss"]
    assert padded["stderr"] == golden["stderr"]

    # Sanity: had the padded duplicate been counted, the mean would move sharply.
    counted = aggregate(padded_losses, [True] * 6, 6)
    assert counted["mean_loss"] != golden["mean_loss"]


def test_aggregate_ddof1_and_mean_match_numpy():
    vals = [1.0, 2.0, 3.0, 4.0, 7.0]
    res = aggregate(vals, [True] * 5, 5)
    assert res["n"] == 5
    assert res["mean_loss"] == float(np.mean(vals))
    assert res["stderr"] == float(np.std(vals, ddof=1) / np.sqrt(5))


def test_aggregate_count_mismatch_raises_fewer_and_more():
    with pytest.raises(ValueError, match="expected_count"):
        aggregate([1.0, 2.0, 3.0], [True] * 3, 5)  # fewer than expected
    with pytest.raises(ValueError, match="expected_count"):
        aggregate([1.0, 2.0, 3.0], [True] * 3, 2)  # more than expected


def test_aggregate_accepts_list_of_per_batch_arrays():
    # Canonical contract: a list of per-batch 1-D loss arrays + matching per-batch
    # validity arrays. The final (padded) batch's invalid entry is excluded.
    batch_losses = [np.array([1.0, 2.0]), np.array([3.0, 999.0])]
    batch_validity = [np.array([True, True]), np.array([True, False])]
    res = aggregate(batch_losses, batch_validity, 3)
    assert res["n"] == 3
    assert res["mean_loss"] == float(np.mean([1.0, 2.0, 3.0]))
    assert res["stderr"] == float(np.std([1.0, 2.0, 3.0], ddof=1) / np.sqrt(3))


def test_aggregate_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        aggregate([1.0, 2.0, 3.0], [True, True], 2)
