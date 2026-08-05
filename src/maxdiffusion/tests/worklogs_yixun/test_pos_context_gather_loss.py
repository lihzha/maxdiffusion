"""exp_05 S6 — the regression objective's data side: gather + loss + the normalized metric.

Plan §4-P3': *sample t per example; cached z̄_t + per-token timestep + actions -> block-0 features
(stop-grad) -> head -> MSE(Ĉ_t, C*_t) fp32*, and the decidable training contract's metric is
**normalized MSE = MSE ÷ per-step target variance, computed once on the TRAIN cache**.

This round owns exactly that -- the tuple the trainer feeds and the two numbers it judges by. The
trainer class is S7 and the dispatch is S8; nothing here touches ``train_wan.py`` or an nnx module.

Three things carry the round:

- **The t-indexed row is THE row.** ``pos_embeds[t]`` and ``z_bar_states[t]`` must come back for the
  *same* t the timestep is built from. An off-by-one here trains the head to predict the context of
  the step before or after the latent it was shown -- a bug no downstream test could name.
- **fp32 at the loss.** The cache is fp16 and the model runs bf16; the objective is fp32 (plan §3's
  cast rule, inherited), and it stays fp32 through the mean the trainer accumulates.
- **The metric cannot silently divide by zero.** A step whose targets are constant across the TRAIN
  cache has zero variance, and ``MSE/0`` would hand the S7 stop rule an inf or a nan to compare
  against its running best. It is refused where it is computed.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import NUM_TRAIN_TIMESTEPS, N_HIST_FRAMES
from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L
from maxdiffusion.models.wan.side_adapter_wan import _build_per_token_timestep, rollout_timesteps_from_sigmas
from maxdiffusion.null_adapter_verify import canonical_sigmas
from maxdiffusion.pos_context_records import (
    POS_ARRAY_FIELDS,
    PRODUCTION_POS_GEOMETRY,
    _make_pos_record,
    _PosGeometry,
)
from maxdiffusion.pos_context_regression import (
    POS_STEPS,
    RegressionBatch,
    gather_training_tuple,
    normalized_regression_loss,
    per_example_regression_loss,
    regression_loss,
    sample_step_indices,
    target_variance_table,
)

# Every axis distinct and tiny, the S5 test convention -- but ``l_pos`` stays 8, because that is the
# deployed row count this round validates against.
_TINY = _PosGeometry(
    z_video=(2, 3, 2, 2),
    z_i0=(2, 1, 2, 2),
    actions=(4, 7),
    pos_embeds=(3, POS_L, 5),
    z_bar_states=(3, 2, 3, 2, 2),
    per_step_final_losses=(3,),
)
_TINY_SIGMAS = np.asarray([1.0, 0.7, 0.3, 0.0], dtype=np.float32)


def _fields(geometry=_TINY, seed=0, latent_dtype="fp32"):
    rng = np.random.default_rng(seed)
    shapes = geometry.shapes()
    arrays = {field: rng.standard_normal(shapes[field]).astype(np.float32) for field in POS_ARRAY_FIELDS}
    return {
        "name": f"episode-{seed:03d}",
        "ordinal": seed,
        "split": "train2000",
        "episode": f"ep-{seed}",
        "latent_dtype": latent_dtype,
        "noise_convention": "keyed",
        "arm": "B1",
        "final_future_mse": 0.125,
        **arrays,
    }


def _record(geometry=_TINY, seed=0, **overrides):
    return _make_pos_record(geometry=geometry, **{**_fields(geometry, seed), **overrides})


def _marked_record(geometry=_TINY, seed=0):
    """A record whose every step carries its own signature, so a gathered row names its own index."""
    shapes = geometry.shapes()
    steps = shapes["pos_embeds"][0]
    embeds = np.stack([np.full(shapes["pos_embeds"][1:], seed * 100 + index, np.float32) for index in range(steps)])
    states = np.stack([np.full(shapes["z_bar_states"][1:], seed * 100 - index, np.float32) for index in range(steps)])
    return _record(geometry, seed, pos_embeds=embeds, z_bar_states=states)


# --------------------------------------------------------------------------------------------------
# 1. The gather: the t-indexed row, its timestep, and everything that rides along.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("indices", [(0, 0), (2, 1), (1, 2), (0, 2)])
def test_the_gathered_rows_are_the_rows_the_indices_name(indices):
    """The off-by-one killer: ``pos_embeds[t]`` and ``z_bar_states[t]`` for the *same* t."""
    records = [_marked_record(seed=1), _marked_record(seed=2)]

    batch = gather_training_tuple(records, np.asarray(indices), sigmas=_TINY_SIGMAS)

    for position, (record, index) in enumerate(zip(records, indices)):
        assert np.array_equal(np.asarray(batch.target_context[position]), np.asarray(record.pos_embeds[index]))
        assert np.array_equal(np.asarray(batch.z_bar_t[position]), np.asarray(record.z_bar_states[index]))


def test_the_timestep_is_the_per_token_timestep_of_that_examples_step():
    """Built from exp_04's own helpers by import: sigma grid -> timestep -> per-token, history zeroed."""
    records = [_record(seed=1), _record(seed=2)]
    indices = np.asarray([2, 0])

    batch = gather_training_tuple(records, indices, sigmas=_TINY_SIGMAS)

    timesteps = rollout_timesteps_from_sigmas(jnp.asarray(_TINY_SIGMAS), NUM_TRAIN_TIMESTEPS)
    channels, f_lat, h_lat, w_lat = _TINY.z_video
    expected = _build_per_token_timestep(timesteps[indices], f_lat, h_lat, w_lat, n_hist=N_HIST_FRAMES)
    assert np.array_equal(np.asarray(batch.timestep_2d), np.asarray(expected))
    tokens_per_frame = (h_lat // 2) * (w_lat // 2)
    assert batch.timestep_2d.shape == (2, f_lat * tokens_per_frame)
    assert np.all(np.asarray(batch.timestep_2d)[:, : N_HIST_FRAMES * tokens_per_frame] == 0.0)
    assert np.all(np.asarray(batch.timestep_2d)[:, N_HIST_FRAMES * tokens_per_frame :] > 0.0)
    del channels


def test_the_actions_names_and_indices_ride_along_unchanged():
    records = [_record(seed=1), _record(seed=2)]
    indices = np.asarray([1, 2])

    batch = gather_training_tuple(records, indices, sigmas=_TINY_SIGMAS)

    assert batch.names == ("episode-001", "episode-002")
    assert np.array_equal(batch.step_indices, indices)
    for position, record in enumerate(records):
        assert np.array_equal(np.asarray(batch.actions[position]), np.asarray(record.actions, np.float32))


def test_every_gathered_tensor_is_finite_fp32():
    """The cache is fp16 and the model is bf16; the objective is fp32 and the cast happens once, here."""
    records = [_record(seed=1, latent_dtype="fp16"), _record(seed=2, latent_dtype="fp16")]

    batch = gather_training_tuple(records, np.asarray([0, 1]), sigmas=_TINY_SIGMAS)

    for field in ("z_bar_t", "timestep_2d", "actions", "target_context"):
        array = np.asarray(getattr(batch, field))
        assert array.dtype == np.float32, field
        assert np.all(np.isfinite(array)), field
    assert isinstance(batch, RegressionBatch) and dataclasses.is_dataclass(batch)


def test_the_same_records_and_indices_gather_bitwise_identically():
    """The seam is deterministic: the trainer owns the RNG, this owns nothing."""
    records = [_record(seed=1), _record(seed=2)]

    first = gather_training_tuple(records, np.asarray([2, 1]), sigmas=_TINY_SIGMAS)
    second = gather_training_tuple(records, np.asarray([2, 1]), sigmas=_TINY_SIGMAS)

    for field in ("z_bar_t", "timestep_2d", "actions", "target_context"):
        assert np.array_equal(np.asarray(getattr(first, field)), np.asarray(getattr(second, field))), field


def test_the_production_geometry_gathers_at_the_canonical_grid():
    """The default grid is exp_04's ``canonical_sigmas`` -- 25 steps, the 9x12x20 token layout."""
    record = _record(PRODUCTION_POS_GEOMETRY, seed=3, latent_dtype="fp16")

    batch = gather_training_tuple([record], np.asarray([24]))

    assert POS_STEPS == 25 == len(canonical_sigmas()) - 1
    assert batch.z_bar_t.shape == (1, 48, 9, 12, 20) and batch.target_context.shape == (1, POS_L, 4096)
    assert batch.timestep_2d.shape == (1, 9 * (12 // 2) * (20 // 2))
    assert np.array_equal(np.asarray(batch.target_context[0]), np.asarray(record.pos_embeds[24], np.float32))
    # The default grid is not merely *a* 25-step grid: the cached states were produced against this
    # one, so a head trained against any other would be conditioned on a timestep its target never saw.
    expected = _build_per_token_timestep(
        rollout_timesteps_from_sigmas(jnp.asarray(canonical_sigmas()), NUM_TRAIN_TIMESTEPS)[np.asarray([24])],
        9,
        12,
        20,
        n_hist=N_HIST_FRAMES,
    )
    assert np.array_equal(np.asarray(batch.timestep_2d), np.asarray(expected))


# --------------------------------------------------------------------------------------------------
# 2. The step-index seam (the trainer owns the key; this owns determinism).
# --------------------------------------------------------------------------------------------------


def test_the_same_key_samples_the_same_step_indices():
    first = sample_step_indices(jax.random.PRNGKey(0), 8)
    again = sample_step_indices(jax.random.PRNGKey(0), 8)
    other = sample_step_indices(jax.random.PRNGKey(1), 8)

    assert np.array_equal(np.asarray(first), np.asarray(again))
    assert not np.array_equal(np.asarray(first), np.asarray(other))
    assert first.shape == (8,) and jnp.issubdtype(first.dtype, jnp.integer)


def test_sampled_indices_stay_inside_the_grid():
    drawn = np.asarray(sample_step_indices(jax.random.PRNGKey(7), 512, steps=POS_STEPS))

    assert drawn.min() >= 0 and drawn.max() < POS_STEPS
    assert len(set(drawn.tolist())) > 1  # not a constant: it really samples


# --------------------------------------------------------------------------------------------------
# 3. The loss, against hand-computed oracles.
# --------------------------------------------------------------------------------------------------


def test_the_per_example_loss_is_the_analytic_mse_over_the_context():
    predicted = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
    target = jnp.asarray([[[1.0, 0.0], [3.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]])
    # example 0: (0 + 4 + 0 + 9)/4 = 3.25;  example 1: (4 + 4 + 4 + 4)/4 = 4.0
    assert np.allclose(np.asarray(per_example_regression_loss(predicted, target)), [3.25, 4.0])


def test_the_batch_loss_is_the_mean_of_the_per_example_losses():
    predicted = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]])
    target = jnp.asarray([[[1.0, 0.0], [3.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]])

    assert np.allclose(float(regression_loss(predicted, target)), (3.25 + 4.0) / 2)
    # Equal-sized examples, so the batch mean of means IS the element mean -- pinned, not assumed.
    assert np.allclose(float(regression_loss(predicted, target)), float(jnp.mean((predicted - target) ** 2)))


def test_the_loss_is_computed_in_fp32_from_bf16_inputs():
    """The model runs bf16 and the cache is fp16; the objective is fp32 (plan §3's cast rule). bf16
    arithmetic would both change the number and hand the trainer a bf16 loss to accumulate."""
    predicted = jnp.asarray(np.full((1, 2, 2), 1.0), dtype=jnp.bfloat16)
    target = jnp.asarray(np.full((1, 2, 2), 1.0 + 2**-7), dtype=jnp.bfloat16)  # exact in bf16
    exact = np.mean(
        (
            np.asarray(predicted, ml_dtypes.bfloat16).astype(np.float32)
            - np.asarray(target, ml_dtypes.bfloat16).astype(np.float32)
        )
        ** 2
    )

    loss = regression_loss(predicted, target)

    assert loss.dtype == jnp.float32
    assert float(loss) == float(exact) == 2.0**-14
    assert per_example_regression_loss(predicted, target).dtype == jnp.float32


def test_the_loss_survives_jit_with_its_shape_checks_intact():
    """S7 jits the train step: static shape checks must still run, value checks must not explode on a
    tracer that has no values yet."""
    predicted, target = jnp.asarray([[[1.0, 2.0]]]), jnp.asarray([[[0.0, 0.0]]])

    assert float(jax.jit(regression_loss)(predicted, target)) == float(regression_loss(predicted, target))
    with pytest.raises(ValueError, match="same shape"):
        jax.jit(regression_loss)(predicted, jnp.zeros((1, 1, 3)))


@pytest.mark.parametrize(
    "predicted, target, message",
    [
        (jnp.zeros((2, 8, 4)), jnp.zeros((2, 8, 5)), "same shape"),
        (jnp.zeros((2, 8, 4)), jnp.zeros((3, 8, 4)), "same shape"),
        (jnp.zeros((2, 8)), jnp.zeros((2, 8)), r"\[B, l_pos, D\]"),
    ],
)
def test_the_loss_refuses_shapes_it_cannot_mean_over(predicted, target, message):
    with pytest.raises(ValueError, match=message):
        regression_loss(predicted, target)


def test_the_loss_refuses_a_non_finite_prediction():
    """A nan loss silently poisons the running best the S7 stop rule compares against."""
    with pytest.raises(ValueError, match="finite"):
        regression_loss(jnp.asarray([[[np.nan, 0.0]]]), jnp.zeros((1, 1, 2)))


# --------------------------------------------------------------------------------------------------
# 4. The per-step variance table -- computed once, on the TRAIN cache.
# --------------------------------------------------------------------------------------------------


def _variance_record(values, geometry=_TINY, seed=0):
    """A record whose ``pos_embeds[i]`` is exactly ``values[i]`` broadcast over the context."""
    shapes = geometry.shapes()
    embeds = np.stack([np.full(shapes["pos_embeds"][1:], value, np.float32) for value in values])
    return _record(geometry, seed, pos_embeds=embeds)


def test_the_variance_table_is_the_per_step_population_variance_of_the_targets():
    """Analytic: step 0 over {1, 3} -> mean 2, var 1; step 1 over {0, 4} -> var 4; step 2 -> var 9."""
    cache = [_variance_record([1.0, 0.0, -3.0], seed=1), _variance_record([3.0, 4.0, 3.0], seed=2)]

    table = target_variance_table(cache)

    assert table.shape == (3,) and table.dtype == np.float32
    assert np.allclose(table, [1.0, 4.0, 9.0])


def _float32_chan(blocks: np.ndarray) -> np.ndarray:
    """The module's own algorithm with a float32 accumulator -- the variant this round rejects.

    Kept here rather than described in prose: a tolerance is only as honest as the wrong answer it
    actually excludes, and this is the wrong answer.
    """
    count, means, m2 = 0, np.zeros(0, np.float32), np.zeros(0, np.float32)
    for raw in blocks:
        block = raw.astype(np.float32)
        block_count = block.shape[1] * block.shape[2]
        block_mean = block.mean(axis=(1, 2))
        block_m2 = ((block - block_mean[:, None, None]) ** 2).sum(axis=(1, 2))
        if count == 0:
            means, m2 = block_mean, block_m2
        else:
            delta = block_mean - means
            total = count + block_count
            means = (means + delta * np.float32(block_count / total)).astype(np.float32)
            m2 = (m2 + block_m2 + delta**2 * np.float32(count * block_count / total)).astype(np.float32)
        count += block_count
    return (m2 / count).astype(np.float64)


def test_the_streaming_table_matches_a_direct_variance_over_the_whole_cache():
    """One pass over a ~15 GiB cache, so the accumulation is streaming -- and must still be the number
    a two-pass computation gives.

    ``rtol`` is not a taste: a correctly accumulated float64 result, returned as fp32, can differ from
    the float64 oracle by at most half an fp32 ulp (5.96e-8). Anything looser stops distinguishing
    this implementation from a float32 accumulator, which lands 1.55e-7 out on this very fixture.
    """
    cache = [_record(seed=seed) for seed in range(5)]
    stacked = np.stack([np.asarray(record.pos_embeds, np.float64) for record in cache])
    oracle = stacked.var(axis=(0, 2, 3))

    table = target_variance_table(cache)

    assert np.allclose(table, oracle, rtol=1e-7, atol=0.0)
    assert np.max(np.abs(_float32_chan(stacked) - oracle) / oracle) > 1e-7  # what the bound excludes


def _offset_record(offset, seed, geometry=_TINY):
    """A cache block far from zero with a spread only a few float32 ulps wide, stored exactly."""
    unit = np.spacing(np.float32(offset))
    deviations = np.random.default_rng(seed).integers(-8, 9, geometry.shapes()["pos_embeds"])
    return _record(geometry, seed, pos_embeds=(np.float32(offset) + deviations.astype(np.float32) * unit))


def test_the_table_holds_on_a_cache_where_float32_accumulation_visibly_fails():
    """**Why float64.** The estimator subtracts a large mean from large numbers. At an offset of 2**20
    the float32 spacing is 0.125 -- a sixth of this fixture's spread -- so a float32 accumulator's
    running mean carries a rounding error the size of the deviations it is measuring, and the variance
    comes out ~7% wrong. The float64 pass reproduces the two-pass oracle to within half an fp32 ulp.

    This is not a hypothetical regime: a step whose targets barely move is exactly where normalized
    MSE is most sensitive, and it is the number the S7 stop rule divides by.
    """
    cache = [_offset_record(2.0**20, seed) for seed in range(6)]
    stacked = np.stack([np.asarray(record.pos_embeds, np.float64) for record in cache])
    oracle = stacked.var(axis=(0, 2, 3))

    table = target_variance_table(cache)

    assert np.allclose(table, oracle, rtol=1e-7, atol=0.0)
    assert np.max(np.abs(_float32_chan(stacked) - oracle) / oracle) > 1e-2  # the same data, off by %


def test_the_table_does_not_depend_on_the_order_the_cache_is_read_in():
    cache = [_record(seed=seed) for seed in range(4)]

    assert np.allclose(target_variance_table(cache), target_variance_table(list(reversed(cache))), rtol=1e-6)


def test_a_step_with_no_variance_is_refused_rather_than_divided_by():
    """Constant targets at a step: ``MSE/0`` hands the stop rule an inf to compare against its best."""
    cache = [_variance_record([1.0, 2.0, 3.0], seed=1), _variance_record([1.0, 5.0, 6.0], seed=2)]

    with pytest.raises(ValueError, match="variance"):
        target_variance_table(cache)


def test_an_empty_cache_is_refused():
    with pytest.raises(ValueError, match="at least one record"):
        target_variance_table([])


def test_a_cache_whose_records_disagree_on_the_step_count_is_refused():
    other = _PosGeometry(
        z_video=(2, 3, 2, 2),
        z_i0=(2, 1, 2, 2),
        actions=(4, 7),
        pos_embeds=(4, POS_L, 5),
        z_bar_states=(4, 2, 3, 2, 2),
        per_step_final_losses=(4,),
    )

    with pytest.raises(ValueError, match="sampler-step count"):
        target_variance_table([_record(seed=1), _record(other, seed=2)])


def test_a_cache_carrying_a_context_that_is_not_l_pos_rows_is_refused():
    narrow = _PosGeometry(
        z_video=(2, 3, 2, 2),
        z_i0=(2, 1, 2, 2),
        actions=(4, 7),
        pos_embeds=(3, POS_L - 1, 5),
        z_bar_states=(3, 2, 3, 2, 2),
        per_step_final_losses=(3,),
    )

    with pytest.raises(ValueError, match=f"l_pos={POS_L}"):
        target_variance_table([_record(narrow, seed=1)])


# --------------------------------------------------------------------------------------------------
# 5. Normalized MSE -- the metric the S7 stop rule and checkpoint selection are decided on.
# --------------------------------------------------------------------------------------------------


def test_each_example_is_normalized_by_its_own_steps_variance():
    """Not the batch MSE over a mean variance: examples in one batch sit at different t."""
    predicted = jnp.asarray([[[1.0, 1.0]], [[0.0, 0.0]]])
    target = jnp.asarray([[[0.0, 0.0]], [[2.0, 2.0]]])  # per-example MSE = 1.0 and 4.0
    table = np.asarray([2.0, 4.0, 1.0], np.float32)

    value = normalized_regression_loss(predicted, target, np.asarray([0, 1]), table)

    assert float(value) == pytest.approx((1.0 / 2.0 + 4.0 / 4.0) / 2)  # 0.75
    # Deliberately distinguishable from normalizing the batch MSE by the batch's mean variance,
    # which these numbers make 2.5/3 -- a different number, and one no stop rule could interpret.
    assert float(value) != pytest.approx(float(jnp.mean(jnp.asarray([1.0, 4.0])) / np.mean(table[:2])))


def test_the_normalized_metric_reduces_to_the_loss_at_unit_variance():
    predicted, target = jnp.asarray([[[1.0, 1.0]]]), jnp.asarray([[[0.0, 0.0]]])
    table = np.ones((3,), np.float32)

    assert float(normalized_regression_loss(predicted, target, np.asarray([2]), table)) == pytest.approx(
        float(regression_loss(predicted, target))
    )


@pytest.mark.parametrize(
    "table, message",
    [
        (np.zeros((3,), np.float32), "positive"),
        (np.asarray([1.0, -1.0, 1.0], np.float32), "positive"),
        (np.asarray([np.inf, 1.0, 1.0], np.float32), "finite"),
        (np.ones((3, 2), np.float32), "one variance per sampler step"),
    ],
)
def test_the_normalized_metric_refuses_an_unusable_variance_table(table, message):
    with pytest.raises(ValueError, match=message):
        normalized_regression_loss(jnp.zeros((1, 1, 2)), jnp.zeros((1, 1, 2)), np.asarray([0]), table)


def test_the_normalized_metric_refuses_a_step_index_the_table_does_not_cover():
    with pytest.raises(ValueError, match="outside"):
        normalized_regression_loss(
            jnp.zeros((1, 1, 2)), jnp.zeros((1, 1, 2)), np.asarray([3]), np.ones((3,), np.float32)
        )


# --------------------------------------------------------------------------------------------------
# 6. Gather guards.
# --------------------------------------------------------------------------------------------------


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError, match="at least one record"):
        gather_training_tuple([], np.asarray([], np.int32), sigmas=_TINY_SIGMAS)


def test_one_index_per_record_or_nothing():
    with pytest.raises(ValueError, match="one step index per record"):
        gather_training_tuple([_record(seed=1), _record(seed=2)], np.asarray([0]), sigmas=_TINY_SIGMAS)


@pytest.mark.parametrize("index", [-1, 3, 99])
def test_a_step_index_outside_the_grid_is_refused(index):
    with pytest.raises(ValueError, match="outside"):
        gather_training_tuple([_record(seed=1)], np.asarray([index]), sigmas=_TINY_SIGMAS)


@pytest.mark.parametrize("indices", [np.asarray([1.0]), np.asarray([True])])
def test_a_step_index_that_is_not_an_integer_is_refused(indices):
    with pytest.raises(ValueError, match="integer"):
        gather_training_tuple([_record(seed=1)], indices, sigmas=_TINY_SIGMAS)


def test_a_sigma_grid_that_does_not_match_the_cached_step_count_is_refused():
    with pytest.raises(ValueError, match="sampler-step count"):
        gather_training_tuple([_record(seed=1)], np.asarray([0]), sigmas=np.asarray([1.0, 0.5, 0.0], np.float32))


def test_records_that_disagree_on_the_step_count_are_refused_at_the_gather_too():
    other = _PosGeometry(
        z_video=(2, 3, 2, 2),
        z_i0=(2, 1, 2, 2),
        actions=(4, 7),
        pos_embeds=(4, POS_L, 5),
        z_bar_states=(4, 2, 3, 2, 2),
        per_step_final_losses=(4,),
    )

    with pytest.raises(ValueError, match="sampler-step count"):
        gather_training_tuple([_record(seed=1), _record(other, seed=2)], np.asarray([0, 0]), sigmas=_TINY_SIGMAS)


def test_a_record_whose_states_do_not_match_its_contexts_is_refused():
    """The schema binds one state to one context; a record that lost that binding cannot be trained on."""

    class _Mangled:
        name, actions = "bad", np.zeros((4, 7), np.float32)
        pos_embeds = np.zeros((3, POS_L, 5), np.float32)
        z_bar_states = np.zeros((2, 2, 3, 2, 2), np.float32)  # one state short

    with pytest.raises(ValueError, match="one state per sampler step"):
        gather_training_tuple([_Mangled()], np.asarray([0]), sigmas=_TINY_SIGMAS)


def test_a_record_carrying_a_non_finite_state_is_refused():
    poisoned = np.asarray(_record(seed=1).z_bar_states, np.float32).copy()
    poisoned[0, 0, 0, 0, 0] = np.nan

    class _Poisoned:
        name, actions = "poisoned", np.zeros((4, 7), np.float32)
        pos_embeds = np.zeros((3, POS_L, 5), np.float32)
        z_bar_states = poisoned

    with pytest.raises(ValueError, match="finite"):
        gather_training_tuple([_Poisoned()], np.asarray([0]), sigmas=_TINY_SIGMAS)
