"""exp_05 S7 — the regression trainer's state machine.

Plan §4-P3' fixes four things this round owns end to end:

- **The split (F3/CLAUDE.md).** The ~5B backbone is frozen and excluded from the optimizer; only the
  pre_context adapter carries gradients and optimizer state. The block-0 features are taken under
  stop-grad, and the frozen module is *closed over* rather than passed as a differentiable argument,
  so "no frozen parameter is trained" is structural, not a convention.
- **Gradient accumulation preserving the logical batch (F3).** Microbatch x accumulation steps == the
  logical GBS. A microbatch that does not divide it is refused; the logical batch is never silently
  changed, and the accumulated update reproduces the full-batch update on a fixed toy.
- **The stop rule (F2), as a pure function over eval history.** DEV normalized MSE > 2x its running
  best for three CONSECUTIVE evals while train MSE is still falling -> stop and retain the prior best;
  otherwise the fixed budget. Selection is the best DEV normalized MSE.
- **Adapter-only checkpointing.** params / opt_state / step, exactly as the side-adapter trainer does
  it -- the input iterator is NOT serialized, and the resumed run rebuilds it at
  ``seed = config.seed + start_step``.

The model is a seam: the tests drive a toy predict function for the arithmetic oracles and a real
tiny WAN transformer + adapter stack for the freeze split, which is the only place the real
pre_context wiring can be observed.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxdiffusion.pos_context_regression import (
    RegressionBatch,
    per_example_regression_loss,
    regression_loss,
)
from maxdiffusion.trainers.wan_pos_context_regression_trainer import (
    DEFAULT_EVAL_EVERY,
    DEFAULT_LOGICAL_BATCH,
    STOP_FACTOR,
    STOP_PATIENCE,
    EvalRecord,
    RegressionTrainState,
    TrainingSchedule,
    WanPosContextRegressionTrainer,
    accumulation_plan,
    best_checkpoint_step,
    build_checkpoint_manager,
    build_pre_context_predict_fn,
    build_selection_manager,
    checked_training_batch,
    evaluate_dev,
    microbatches,
    preserve_selection,
    restore_adapter_checkpoint,
    resume_seed,
    save_adapter_checkpoint,
    should_evaluate,
    stop_verdict,
    train_step,
)

_L_POS, _DIM = 8, 5


class _Config:
    """A stand-in for ``HyperParameters``: undeclared keys raise ``ValueError``, not ``AttributeError``.

    This is the S4 trap (review finding 1) reproduced deliberately -- ``getattr(config, key, default)``
    never reaches its default against the real config object, so the trainer must read through
    ``optional_config_value``.
    """

    def __init__(self, **declared):
        self._declared = declared

    def get_keys(self):
        return dict(self._declared)

    def __getattr__(self, name):
        declared = object.__getattribute__(self, "_declared")
        if name not in declared:
            raise ValueError(f"Key {name} not in config")
        return declared[name]


def _batch(size=4, *, seed=0, l_pos=_L_POS, dim=_DIM):
    rng = np.random.default_rng(seed)
    return RegressionBatch(
        names=tuple(f"ep{index}" for index in range(size)),
        step_indices=np.arange(size, dtype=np.int32) % 3,
        z_bar_t=jnp.asarray(rng.standard_normal((size, 2, 3, 2, 2)), jnp.float32),
        timestep_2d=jnp.asarray(rng.standard_normal((size, 3)), jnp.float32),
        actions=jnp.asarray(rng.standard_normal((size, 4, 7)), jnp.float32),
        target_context=jnp.asarray(rng.standard_normal((size, l_pos, dim)), jnp.float32),
    )


def _toy_params(seed=0, l_pos=_L_POS, dim=_DIM):
    rng = np.random.default_rng(seed)
    return {"w": jnp.asarray(rng.standard_normal((l_pos, dim)), jnp.float32)}


def _toy_predict(params, batch):
    """Linear in ``params`` and example-dependent, so per-example gradients genuinely differ."""
    feature = jnp.mean(batch.z_bar_t, axis=(1, 2, 3, 4))
    return feature[:, None, None] * params["w"][None, :, :]


def _history(*pairs, first_step=1000, every=1000):
    return [
        EvalRecord(step=first_step + index * every, dev_normalized_mse=dev, train_mse=train)
        for index, (dev, train) in enumerate(pairs)
    ]


# --------------------------------------------------------------------------------------------------
# 1. The schedule: cadence and batching read from config, never from scattered literals.
# --------------------------------------------------------------------------------------------------


def test_the_schedule_is_read_from_config():
    config = _Config(
        max_train_steps=5000,
        eval_every=250,
        pos_logical_batch=128,
        pos_microbatch=32,
        seed=7,
        checkpoint_every=500,
    )

    schedule = TrainingSchedule.from_config(config)

    assert schedule.max_train_steps == 5000 and schedule.eval_every == 250
    assert schedule.logical_batch == 128 and schedule.microbatch == 32 and schedule.accumulation_steps == 4
    assert schedule.seed == 7


def test_the_schedule_defaults_are_the_plans_numbers_and_survive_an_undeclared_key():
    """``getattr(config, key, default)`` would raise here, which is the whole point of the helper."""
    schedule = TrainingSchedule.from_config(_Config(max_train_steps=30000, seed=0))

    assert (DEFAULT_LOGICAL_BATCH, DEFAULT_EVAL_EVERY) == (256, 1000)  # plan §4-P3' F2/F3
    assert schedule.logical_batch == 256 and schedule.eval_every == 1000
    assert schedule.microbatch == 256 and schedule.accumulation_steps == 1  # no accumulation by default


@pytest.mark.parametrize(
    "step, expected", [(1, False), (999, False), (1000, True), (1500, False), (2000, True), (30000, True)]
)
def test_evaluation_fires_on_the_configured_cadence(step, expected):
    schedule = TrainingSchedule.from_config(_Config(max_train_steps=30000, seed=0))

    assert should_evaluate(step, schedule) is expected


def test_the_last_step_always_evaluates_even_off_cadence():
    schedule = TrainingSchedule.from_config(_Config(max_train_steps=2500, eval_every=1000, seed=0))

    assert should_evaluate(2500, schedule) is True  # selection needs the final model measured
    assert should_evaluate(2400, schedule) is False


# --------------------------------------------------------------------------------------------------
# 2. Gradient accumulation: the logical batch is a contract, not a suggestion.
# --------------------------------------------------------------------------------------------------


def test_the_accumulation_plan_preserves_the_logical_batch():
    assert accumulation_plan(256, 64) == (64, 4)
    assert accumulation_plan(256, 256) == (256, 1)
    assert accumulation_plan(256, None) == (256, 1)


@pytest.mark.parametrize(
    "logical, microbatch, message",
    [
        (256, 24, "does not divide"),
        (256, 512, "larger than the logical batch"),
        (256, 0, "positive"),
        (0, 8, "positive"),
    ],
)
def test_a_microbatch_that_would_change_the_logical_batch_is_refused(logical, microbatch, message):
    """F3 is explicit: a gradient-accumulation fallback preserves the logical GBS, never silently
    changes it. Every unrepresentable request is an error, not a rounded-down batch."""
    with pytest.raises(ValueError, match=message):
        accumulation_plan(logical, microbatch)


def test_microbatches_partition_the_batch_exactly_once():
    batch = _batch(size=6)

    parts = microbatches(batch, 3)

    assert [len(part.names) for part in parts] == [2, 2, 2]
    assert sum((list(part.names) for part in parts), []) == list(batch.names)
    assert np.array_equal(
        np.concatenate([np.asarray(part.target_context) for part in parts]), np.asarray(batch.target_context)
    )


def test_a_batch_that_does_not_split_evenly_is_refused():
    with pytest.raises(ValueError, match="does not divide"):
        microbatches(_batch(size=5), 2)


def test_the_accumulated_update_reproduces_the_full_batch_update():
    """The equivalence oracle: four microbatches must move the parameters where one batch of the same
    8 examples would. Anything else is a silent change of the logical batch size.

    **Why a tolerance rather than bitwise.** The two computations sum the *same* per-example gradients
    in different orders -- one reduction over 8 examples versus four reductions over 2, then a mean --
    and fp32 addition is not associative. The gap is therefore bounded by rounding, not by semantics;
    ``atol=1e-6`` is several orders above the observed difference and far below any real disagreement
    (a dropped microbatch or a sum-instead-of-mean moves these leaves by ~1e-1, and both are in the
    battery). Making it bitwise would mean reimplementing one path in terms of the other, which would
    test nothing.
    """
    batch, params = _batch(size=8, seed=1), _toy_params()
    tx = optax.sgd(0.1)
    state = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)

    single, single_metrics = train_step(state, batch, predict_fn=_toy_predict, tx=tx, accumulation_steps=1)
    split, split_metrics = train_step(state, batch, predict_fn=_toy_predict, tx=tx, accumulation_steps=4)

    assert np.allclose(np.asarray(split.params["w"]), np.asarray(single.params["w"]), rtol=0, atol=1e-6)
    assert split_metrics["loss"] == pytest.approx(single_metrics["loss"], rel=1e-6)
    assert split.step == single.step == 1


def test_the_accumulated_loss_is_the_whole_batchs_loss():
    batch, params = _batch(size=8, seed=2), _toy_params()
    tx = optax.sgd(0.0)  # no movement: the metric is the only thing under test
    state = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)

    _, metrics = train_step(state, batch, predict_fn=_toy_predict, tx=tx, accumulation_steps=4)

    assert metrics["loss"] == pytest.approx(float(regression_loss(_toy_predict(params, batch), batch.target_context)))


# --------------------------------------------------------------------------------------------------
# 2b. JIT: the step K3 actually runs is the jitted one (S7 review, BLOCKER 1).
# --------------------------------------------------------------------------------------------------


def test_the_state_and_the_batch_are_pytrees():
    """Unregistered dataclasses cannot cross a ``jit`` boundary: the probe fails on ``state`` before
    tracing even starts."""
    params = _toy_params()
    tx = optax.sgd(0.1)
    state = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)
    batch = _batch(size=4)

    assert jax.tree.leaves(state), "the train state carries no pytree leaves"
    assert len(jax.tree.leaves(batch)) == 5  # step_indices + the four tensors
    doubled = jax.tree.map(lambda leaf: leaf * 2, batch)
    assert doubled.names == batch.names  # names ride as metadata, not as data
    assert np.array_equal(np.asarray(doubled.target_context), np.asarray(batch.target_context) * 2)


def test_the_train_step_runs_under_jit_and_agrees_with_the_eager_one():
    """The executing oracle the review asked for -- not a claim that it *should* be jittable."""
    batch, params = _batch(size=8, seed=4), _toy_params()
    tx = optax.adam(1e-2)
    state = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)
    step_fn = functools.partial(train_step, predict_fn=_toy_predict, tx=tx, accumulation_steps=2)

    eager_state, eager_metrics = step_fn(state, batch)
    jit_state, jit_metrics = jax.jit(step_fn)(state, batch)

    assert np.allclose(np.asarray(jit_state.params["w"]), np.asarray(eager_state.params["w"]), rtol=0, atol=1e-6)
    assert float(jit_metrics["loss"]) == pytest.approx(float(eager_metrics["loss"]), rel=1e-6)
    assert int(jit_state.step) == int(eager_state.step) == 1


def test_the_step_metrics_stay_jax_scalars():
    """``float()`` inside traced code raises a concretization error; the host loop does the conversion."""
    params = _toy_params()
    tx = optax.sgd(0.1)
    _, metrics = train_step(
        RegressionTrainState(params=params, opt_state=tx.init(params), step=0),
        _batch(size=4),
        predict_fn=_toy_predict,
        tx=tx,
        accumulation_steps=1,
    )

    assert all(isinstance(value, jax.Array) for value in metrics.values()), metrics


# --------------------------------------------------------------------------------------------------
# 2c. The logical batch is checked against what the iterator actually yields (BLOCKER 2).
# --------------------------------------------------------------------------------------------------


def test_a_batch_that_is_not_the_logical_batch_is_refused():
    """128 examples under ``logical=256, microbatch=64`` used to be accepted as four 32s -- a halved
    logical batch, silently, which is exactly what F3 forbids."""
    schedule = TrainingSchedule.from_config(
        _Config(max_train_steps=10, seed=0, pos_logical_batch=256, pos_microbatch=64)
    )

    with pytest.raises(ValueError, match="logical batch"):
        checked_training_batch(_batch(size=128), schedule)


def test_the_right_sized_batch_passes_through_unchanged():
    schedule = TrainingSchedule.from_config(_Config(max_train_steps=10, seed=0, pos_logical_batch=8, pos_microbatch=4))
    batch = _batch(size=8)

    assert checked_training_batch(batch, schedule) is batch


def test_a_schedule_whose_microbatch_contradicts_its_own_split_is_refused():
    """``accumulation_plan`` makes the two agree, so this guard only fires for a schedule assembled by
    hand -- which is exactly what S8's config wiring will be doing next. It is checked, not assumed."""
    schedule = dataclasses.replace(
        TrainingSchedule.from_config(_Config(max_train_steps=10, seed=0, pos_logical_batch=8, pos_microbatch=4)),
        microbatch=3,
    )

    with pytest.raises(ValueError, match="microbatch width"):
        checked_training_batch(_batch(size=8), schedule)


def test_the_loop_refuses_a_wrong_sized_iterator(tmp_path):
    config = _Config(max_train_steps=4, eval_every=2, seed=0, pos_logical_batch=8, pos_microbatch=4)
    trainer, dev_evaluator = _trainer(tmp_path, config, [0.5, 0.4])

    with pytest.raises(ValueError, match="logical batch"):
        trainer.run(lambda seed: itertools.cycle([_batch(size=4)]), dev_evaluator)


# --------------------------------------------------------------------------------------------------
# 3. The frozen split, on the real pre_context wiring.
# --------------------------------------------------------------------------------------------------


def _tiny_pre_context():
    """A real WAN transformer and a real pre_context adapter stack, both small enough for CPU."""
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning

    from maxdiffusion.models.wan.side_adapter_wan import NNXWanSideAdapterStack
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    channels, text_dim, f_lat, h_lat, w_lat = 4, 32, 2, 4, 6
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    set_mesh = getattr(jax, "set_mesh", None)
    with nn_partitioning.axis_rules(()), set_mesh(mesh) if set_mesh is not None else mesh:
        transformer = WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=channels,
            out_channels=channels,
            text_dim=text_dim,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
            dtype=jnp.bfloat16,
            weights_dtype=jnp.bfloat16,
        )
        adapters = NNXWanSideAdapterStack(
            rngs=nnx.Rngs(jax.random.key(1)),
            num_layers=1,
            model_dim=16,
            text_dim=text_dim,
            action_adapter_type="pre_context",
            action_dim=7,
            action_len=4,
            action_repr="delta",
            action_tokens=4,
            action_hidden=16,
            action_heads=2,
            side_adapter_layers="0",
            side_adapter_hidden=16,
            side_adapter_heads=2,
            pre_context_tokens=_L_POS,
            pre_context_heads=2,
            dtype=jnp.bfloat16,
            weights_dtype=jnp.bfloat16,
        )
    batch = RegressionBatch(
        names=("a", "b"),
        step_indices=np.zeros(2, np.int32),
        z_bar_t=jax.random.normal(jax.random.PRNGKey(2), (2, channels, f_lat, h_lat, w_lat), jnp.float32),
        timestep_2d=jnp.full((2, f_lat * (h_lat // 2) * (w_lat // 2)), 700.0, jnp.float32).at[:, :6].set(0.0),
        actions=jax.random.normal(jax.random.PRNGKey(3), (2, 4, 7), jnp.float32),
        target_context=jax.random.normal(jax.random.PRNGKey(4), (2, _L_POS, text_dim), jnp.float32),
    )
    return transformer, adapters, batch, mesh


def test_only_the_adapter_carries_gradients_and_optimizer_state():
    """**The freeze contract.** The 5B backbone is closed over, not passed as a differentiable
    argument, and the block-0 features arrive under stop-grad -- so a frozen parameter cannot receive
    a gradient even by accident, and ``tx.init`` never sees one."""
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning

    transformer, adapters, batch, mesh = _tiny_pre_context()
    set_mesh = getattr(jax, "set_mesh", None)

    with nn_partitioning.axis_rules(()), set_mesh(mesh) if set_mesh is not None else mesh:
        predict_fn, params = build_pre_context_predict_fn(transformer, adapters)
        tx = optax.adam(1e-2)  # stateful on purpose: SGD's EmptyState makes the opt-state claim vacuous
        state = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)
        frozen_before = jax.tree.leaves(nnx.split(transformer, nnx.Param, ...)[1])

        updated, metrics = train_step(state, batch, predict_fn=predict_fn, tx=tx, accumulation_steps=1)

        frozen_after = jax.tree.leaves(nnx.split(transformer, nnx.Param, ...)[1])

    assert frozen_before, "the fixture has no frozen side, so this proves nothing"
    assert jax.tree.structure(updated.params) == jax.tree.structure(params)
    # Every frozen leaf is bit-unchanged by a training step.
    for before, after in zip(frozen_before, frozen_after):
        assert np.array_equal(np.asarray(before, np.float32), np.asarray(after, np.float32))
    # Adam keeps two parameter-shaped slots (mu, nu) per trainable leaf, so the optimizer state's
    # parameter-shaped leaves are exactly the adapter's tree twice over -- and nothing else. Plain SGD
    # would have made this assertion vacuous: its EmptyState has no leaves at all (S7 review, MAJOR 5).
    adapter_shapes = sorted(tuple(leaf.shape) for leaf in jax.tree.leaves(params))
    opt_shapes = sorted(
        tuple(leaf.shape) for leaf in jax.tree.leaves(updated.opt_state) if getattr(leaf, "shape", ()) != ()
    )
    assert opt_shapes == sorted(adapter_shapes * 2), (len(opt_shapes), len(adapter_shapes))
    # ... and the adapter really moved: a freeze test that trains nothing proves nothing.
    assert any(
        not np.array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))
        for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(updated.params))
    )
    assert np.isfinite(metrics["loss"])


def test_the_prediction_depends_on_the_cached_state_and_on_the_actions():
    """Teacher forcing means both inputs are real: a head wired to ignore the actions, or to ignore the
    cached z-bar it is conditioned on, would still train to *a* number and still freeze correctly."""
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax.linen import partitioning as nn_partitioning

    transformer, adapters, batch, mesh = _tiny_pre_context()
    set_mesh = getattr(jax, "set_mesh", None)

    with nn_partitioning.axis_rules(()), set_mesh(mesh) if set_mesh is not None else mesh:
        predict_fn, params = build_pre_context_predict_fn(transformer, adapters)
        base = np.asarray(predict_fn(params, batch), np.float32)
        other_actions = np.asarray(
            predict_fn(params, dataclasses.replace(batch, actions=batch.actions * 3.0)), np.float32
        )
        other_state = np.asarray(
            predict_fn(params, dataclasses.replace(batch, z_bar_t=batch.z_bar_t * 3.0)), np.float32
        )

    assert not np.array_equal(base, other_actions), "the head ignores the action sequence"
    assert not np.array_equal(base, other_state), "the head ignores the cached state it conditions on"


def test_the_head_emits_the_deployed_eight_token_context():
    """No representation shim (plan §3 F1): the head's output IS the conditioning, so its shape is the
    target's shape."""
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax.linen import partitioning as nn_partitioning

    transformer, adapters, batch, mesh = _tiny_pre_context()
    set_mesh = getattr(jax, "set_mesh", None)

    with nn_partitioning.axis_rules(()), set_mesh(mesh) if set_mesh is not None else mesh:
        predict_fn, params = build_pre_context_predict_fn(transformer, adapters)
        predicted = predict_fn(params, batch)

    assert predicted.shape == batch.target_context.shape == (2, _L_POS, 32)


# --------------------------------------------------------------------------------------------------
# 4. The stop rule -- a pure function over the eval history.
# --------------------------------------------------------------------------------------------------


def test_the_rule_constants_are_the_plans():
    assert (STOP_FACTOR, STOP_PATIENCE) == (2.0, 3)


def test_three_consecutive_degraded_evals_while_train_falls_stops_the_run():
    history = _history((1.0, 5.0), (0.5, 4.0), (1.2, 3.0), (1.3, 2.0), (1.4, 1.0))

    verdict = stop_verdict(history)

    assert verdict.stop and verdict.streak == 3
    assert "consecutive" in verdict.reason
    assert (verdict.best_step, verdict.best_value) == (2000, 0.5)


def test_exactly_two_consecutive_degraded_evals_do_not_stop():
    history = _history((1.0, 5.0), (0.5, 4.0), (1.2, 3.0), (1.3, 2.0))

    verdict = stop_verdict(history)

    assert not verdict.stop and verdict.streak == 2


def test_a_train_mse_that_stops_falling_breaks_the_streak():
    """The rule is about *overfitting*: DEV rising while train still falls. A train loss that plateaus
    is a different failure and does not spend the patience budget."""
    history = _history((1.0, 5.0), (0.5, 4.0), (1.2, 3.0), (1.3, 3.0), (1.4, 2.0), (1.5, 1.0))

    verdict = stop_verdict(history)

    assert not verdict.stop and verdict.streak == 2  # the plateau reset it, only two since


def test_a_new_best_interrupts_the_streak():
    """Two degraded evals, then a new best, then one more degraded eval. Counting "three degraded"
    without the interruption would stop a run that just reached its best DEV number."""
    history = _history((1.0, 5.0), (0.5, 4.0), (1.2, 3.0), (1.3, 2.0), (0.4, 1.5), (1.0, 1.0))

    verdict = stop_verdict(history)

    assert not verdict.stop and verdict.streak == 1  # restarted at the eval after the new best
    assert (verdict.best_step, verdict.best_value) == (5000, 0.4)


def test_a_single_eval_cannot_trigger_the_rule():
    """There is no running best to be twice as bad as, and no previous train MSE to be falling from."""
    verdict = stop_verdict(_history((9.0, 9.0)))

    assert not verdict.stop and verdict.streak == 0 and verdict.best_step == 1000


def test_an_empty_history_decides_nothing():
    verdict = stop_verdict([])

    assert not verdict.stop and verdict.best_step is None and verdict.best_value is None


def test_exactly_twice_the_best_is_not_worse_than_twice_the_best():
    """``> 2x``, not ``>=``: the threshold is where the plan put it."""
    history = _history((1.0, 5.0), (2.0, 4.0), (2.0, 3.0), (2.0, 2.0))

    assert not stop_verdict(history).stop


def test_the_earliest_step_wins_a_tie_for_the_best():
    history = _history((0.5, 5.0), (0.5, 4.0), (0.5, 3.0))

    assert best_checkpoint_step(history) == 1000


def test_the_retained_checkpoint_is_the_prior_best_not_a_degraded_one():
    """The off-by-one killer: the run stops *at* the third bad eval, and the artifact kept is the one
    from before the degradation started."""
    history = _history((1.0, 5.0), (0.3, 4.0), (1.1, 3.0), (1.2, 2.0), (1.3, 1.0))

    verdict = stop_verdict(history)

    assert verdict.stop and verdict.best_step == 2000
    assert best_checkpoint_step(history) == 2000
    assert verdict.best_step not in (3000, 4000, 5000)  # none of the degraded evals


def test_the_rule_is_pure():
    history = _history((1.0, 5.0), (0.5, 4.0), (1.2, 3.0), (1.3, 2.0), (1.4, 1.0))
    snapshot = list(history)

    first, second = stop_verdict(history), stop_verdict(history)

    assert first == second and history == snapshot


# --------------------------------------------------------------------------------------------------
# 5. Checkpointing: adapter only, and the iterator is not part of it.
# --------------------------------------------------------------------------------------------------


def test_the_checkpoint_carries_params_opt_state_and_step_only(tmp_path):
    """Orbax partial resume, exactly as the side-adapter trainer does it: the dataloader cursor is not
    serialized, which is why the seed rule below exists.

    Asserted on what lands on disk rather than on a manager attribute -- the item set is a property of
    the checkpoint, and a future Orbax could rename the attribute without changing it.
    """
    from maxdiffusion.trainers.wan_pos_context_regression_trainer import CHECKPOINT_ITEMS

    params = _toy_params()
    tx = optax.sgd(0.1)
    manager = build_checkpoint_manager(str(tmp_path / "ckpt"))

    save_adapter_checkpoint(
        manager, RegressionTrainState(params=params, opt_state=tx.init(params), step=5), dev_metric=1.0
    )
    manager.wait_until_finished()

    written = {child.name for child in (tmp_path / "ckpt" / "5").iterdir() if child.is_dir()}
    assert set(CHECKPOINT_ITEMS) == {"params", "opt_state", "step"}
    assert written >= set(CHECKPOINT_ITEMS)
    assert not {name for name in written if "iter" in name}, written


def test_saving_and_restoring_round_trips_the_adapter_state(tmp_path):
    params = _toy_params(seed=3)
    tx = optax.adam(1e-3)
    state = RegressionTrainState(params=params, opt_state=tx.init(params), step=17)
    manager = build_checkpoint_manager(str(tmp_path / "ckpt"))

    save_adapter_checkpoint(manager, state, dev_metric=0.5)
    manager.wait_until_finished()
    empty = RegressionTrainState(
        params=jax.tree.map(jnp.zeros_like, params), opt_state=tx.init(jax.tree.map(jnp.zeros_like, params)), step=0
    )
    restored, start_step = restore_adapter_checkpoint(manager, empty)

    assert start_step == 17 and restored.step == 17
    assert np.array_equal(np.asarray(restored.params["w"]), np.asarray(params["w"]))
    assert jax.tree.structure(restored.opt_state) == jax.tree.structure(state.opt_state)


def test_restoring_from_an_empty_directory_starts_at_zero(tmp_path):
    params = _toy_params()
    tx = optax.sgd(0.1)
    state = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)

    restored, start_step = restore_adapter_checkpoint(build_checkpoint_manager(str(tmp_path / "ckpt")), state)

    assert start_step == 0 and restored is state


def test_the_restore_unit_round_trips_a_partial_run(tmp_path):
    """A UNIT check on save/restore only: two steps, save, restore, two more. It says nothing about the
    loop -- the integrated claim lives in ``test_the_interrupted_run_reproduces_the_uninterrupted_one``
    (S7 review, BLOCKER 4)."""
    tx = optax.adam(1e-2)
    params = _toy_params(seed=5)
    batches = [_batch(size=4, seed=seed) for seed in range(4)]

    state = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)
    for batch in batches:
        state, _ = train_step(state, batch, predict_fn=_toy_predict, tx=tx, accumulation_steps=1)
    uninterrupted = state

    manager = build_checkpoint_manager(str(tmp_path / "ckpt"))
    partial = RegressionTrainState(params=params, opt_state=tx.init(params), step=0)
    for batch in batches[:2]:
        partial, _ = train_step(partial, batch, predict_fn=_toy_predict, tx=tx, accumulation_steps=1)
    save_adapter_checkpoint(manager, partial, dev_metric=1.0)
    manager.wait_until_finished()

    resumed, start_step = restore_adapter_checkpoint(
        manager, RegressionTrainState(params=params, opt_state=tx.init(params), step=0)
    )
    assert start_step == 2
    for batch in batches[2:]:
        resumed, _ = train_step(resumed, batch, predict_fn=_toy_predict, tx=tx, accumulation_steps=1)

    assert np.allclose(np.asarray(resumed.params["w"]), np.asarray(uninterrupted.params["w"]), rtol=0, atol=1e-6)


def test_the_iterator_seed_is_rebuilt_from_the_resumed_step():
    """The dataloader cursor is not in the checkpoint, so the seed carries the position (CLAUDE.md)."""
    assert resume_seed(2026, 0) == 2026
    assert resume_seed(2026, 12000) == 14026


def test_resume_retention_keeps_the_latest_state(tmp_path):
    """**S7 review, BLOCKER 3.** Retention for the resume payload is by RECENCY. Ranking it by DEV
    would let ``latest_step()`` return an old "best" after the newest state was evicted, and a resumed
    run would silently rewind -- the one thing a checkpoint exists to prevent."""
    params = _toy_params()
    tx = optax.sgd(0.1)
    manager = build_checkpoint_manager(str(tmp_path / "ckpt"), max_to_keep=2)

    for step, dev in ((1, 0.1), (2, 0.9), (3, 0.9), (4, 0.9)):  # step 1 is by far the best DEV
        save_adapter_checkpoint(
            manager, RegressionTrainState(params=params, opt_state=tx.init(params), step=step), dev_metric=dev
        )
        manager.wait_until_finished()

    assert manager.latest_step() == 4
    assert sorted(manager.all_steps()) == [3, 4]  # the best-DEV step 1 was evicted, as recency demands
    restored, start_step = restore_adapter_checkpoint(
        manager, RegressionTrainState(params=params, opt_state=tx.init(params), step=0)
    )
    assert start_step == 4 and restored.step == 4


def test_the_selection_checkpoint_preserves_the_earliest_best(tmp_path):
    """**S7 review, BLOCKER 3.** Selection is a separate, immutable artifact: it is written only on a
    STRICT improvement, so a later tie can never overwrite the earliest best -- and recency eviction
    on the resume tree cannot delete it."""
    params = _toy_params()
    tx = optax.sgd(0.1)
    selection = build_selection_manager(str(tmp_path / "ckpt"))

    for step, dev in ((1, 0.9), (2, 0.4), (3, 0.4), (4, 0.8)):
        preserve_selection(
            selection, RegressionTrainState(params=params, opt_state=tx.init(params), step=step), dev_metric=dev
        )
        selection.wait_until_finished()

    assert selection.latest_step() == 2  # not 3, which merely tied
    assert list(selection.all_steps()) == [2]


def test_the_selection_checkpoint_is_restorable_on_its_own(tmp_path):
    """K3 hands K4 this artifact, so it has to carry a usable state, not just a number."""
    params = _toy_params(seed=8)
    tx = optax.adam(1e-3)
    selection = build_selection_manager(str(tmp_path / "ckpt"))

    preserve_selection(
        selection, RegressionTrainState(params=params, opt_state=tx.init(params), step=9), dev_metric=0.25
    )
    selection.wait_until_finished()
    restored, step = restore_adapter_checkpoint(
        selection, RegressionTrainState(params=jax.tree.map(jnp.zeros_like, params), opt_state=tx.init(params), step=0)
    )

    assert step == 9 and np.array_equal(np.asarray(restored.params["w"]), np.asarray(params["w"]))


# --------------------------------------------------------------------------------------------------
# 6. The loop: the state machine those parts add up to.
# --------------------------------------------------------------------------------------------------


def _sequence_evaluator(values):
    """A DEV evaluator that reads a scripted sequence -- the loop's decisions are what is under test."""
    remaining = list(values)

    def evaluate(state):
        return remaining.pop(0) if remaining else 1.0

    return evaluate


def _trainer(tmp_path, config, dev_values, *, lr=0.0):
    tx = optax.sgd(lr)
    params = _toy_params(seed=9)
    trainer = WanPosContextRegressionTrainer(
        config,
        predict_fn=_toy_predict,
        params=params,
        tx=tx,
        manager=build_checkpoint_manager(str(tmp_path / "ckpt")),
        selection_manager=build_selection_manager(str(tmp_path / "ckpt")),
    )
    return trainer, _sequence_evaluator(dev_values)


def test_the_loop_runs_the_whole_budget_when_nothing_triggers(tmp_path):
    config = _Config(max_train_steps=6, eval_every=2, seed=3, pos_logical_batch=4, pos_microbatch=4)
    trainer, dev_evaluator = _trainer(tmp_path, config, [0.9, 0.8, 0.7])

    report = trainer.run(lambda seed: iter([_batch(size=4, seed=seed + i) for i in range(6)]), dev_evaluator)

    assert report.steps_run == 6 and not report.verdict.stop
    assert [record.step for record in report.history] == [2, 4, 6]
    assert report.retained_step == 6  # the last is the best here


def test_the_loop_stops_on_the_rule_and_reports_the_prior_best(tmp_path):
    """One repeated batch and a real learning rate, so the train MSE genuinely falls at every eval --
    the rule's other conjunct has to hold for the DEV signal to be allowed to stop anything."""
    config = _Config(max_train_steps=20, eval_every=1, seed=3, pos_logical_batch=4, pos_microbatch=4)
    # DEV: a best at step 2, then three evals above 2x it while train MSE keeps falling.
    trainer, dev_evaluator = _trainer(tmp_path, config, [1.0, 0.2, 1.1, 1.2, 1.3, 0.05], lr=0.05)
    batch = _batch(size=4, seed=41)

    report = trainer.run(lambda seed: itertools.cycle([batch]), dev_evaluator)

    assert [record.train_mse for record in report.history] == sorted(
        (record.train_mse for record in report.history), reverse=True
    )
    assert report.verdict.stop and report.steps_run == 5
    assert report.retained_step == 2  # not 5, and not the never-reached 0.05
    assert len(report.history) == 5
    # The selection artifact and the report must name the same step: the loop writes the selection
    # checkpoint only on a strict improvement, and recency eviction on the resume tree cannot touch it.
    trainer.manager.wait_until_finished()
    trainer.selection_manager.wait_until_finished()
    assert trainer.selection_manager.latest_step() == report.retained_step
    assert trainer.manager.latest_step() == report.history[-1].step  # resume still points at the newest


def test_the_recorded_train_mse_is_the_window_since_the_previous_eval(tmp_path):
    """The rule asks whether train MSE is *still falling*, so the two numbers it compares must cover
    disjoint windows. A cumulative average would smear the most recent steps into every earlier one."""
    config = _Config(max_train_steps=4, eval_every=2, seed=0, pos_logical_batch=4, pos_microbatch=4)
    trainer, dev_evaluator = _trainer(tmp_path, config, [0.9, 0.8])  # lr=0.0: the losses are the batches'
    batches = [_batch(size=4, seed=seed) for seed in range(4)]
    params = trainer.state.params
    losses = [float(regression_loss(_toy_predict(params, batch), batch.target_context)) for batch in batches]

    report = trainer.run(lambda seed: iter(batches), dev_evaluator)

    assert report.history[0].train_mse == pytest.approx((losses[0] + losses[1]) / 2, rel=1e-6)
    assert report.history[1].train_mse == pytest.approx((losses[2] + losses[3]) / 2, rel=1e-6)
    assert report.history[1].train_mse != pytest.approx(sum(losses) / 4, rel=1e-6)  # not cumulative


def test_the_loop_rebuilds_the_iterator_at_the_resumed_seed(tmp_path):
    config = _Config(max_train_steps=2, eval_every=2, seed=100, pos_logical_batch=4, pos_microbatch=4)
    trainer, dev_evaluator = _trainer(tmp_path, config, [0.5])
    seen = []

    def make_iterator(seed):
        seen.append(seed)
        return iter([_batch(size=4, seed=0) for _ in range(4)])

    trainer.run(make_iterator, dev_evaluator, start_step=1)

    assert seen == [101]  # config.seed + start_step


def test_the_loop_evaluates_dev_with_the_normalized_metric():
    """The number the rule is decided on is normalized MSE, not raw MSE (plan §4-P3' F2)."""
    params = _toy_params(seed=11)
    state = RegressionTrainState(params=params, opt_state=None, step=0)
    batches = [_batch(size=4, seed=21)]
    table = np.full((3,), 4.0, np.float32)

    value = evaluate_dev(state, batches, predict_fn=_toy_predict, variance_table=table)
    raw = float(regression_loss(_toy_predict(params, batches[0]), batches[0].target_context))

    assert value == pytest.approx(raw / 4.0, rel=1e-6)


def test_dev_evaluation_weights_examples_not_batches():
    """**S7 review, MAJOR 6.** A short final batch must not count as much as a full one: the selection
    metric is a per-example mean over the whole DEV set, not a mean of batch means."""
    params = _toy_params(seed=13)
    state = RegressionTrainState(params=params, opt_state=None, step=0)
    big, small = _batch(size=4, seed=31), _batch(size=1, seed=32)
    table = np.asarray([2.0, 4.0, 8.0], np.float32)

    value = evaluate_dev(state, [big, small], predict_fn=_toy_predict, variance_table=table)

    def normalized(batch):
        losses = np.asarray(per_example_regression_loss(_toy_predict(params, batch), batch.target_context), np.float64)
        return losses / np.asarray(table)[np.asarray(batch.step_indices)]

    pooled = np.concatenate([normalized(big), normalized(small)])
    assert value == pytest.approx(float(pooled.mean()), rel=1e-6)
    batch_means = np.mean([normalized(big).mean(), normalized(small).mean()])
    assert value != pytest.approx(float(batch_means), rel=1e-6)  # the two really differ here


def test_dev_evaluation_needs_at_least_one_batch():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_dev(
            RegressionTrainState(params=_toy_params(), opt_state=None, step=0),
            [],
            predict_fn=_toy_predict,
            variance_table=np.ones((3,), np.float32),
        )


def _resumable_trainer(directory, steps):
    """A trainer bound to a checkpoint directory: build it twice on the same directory to resume."""
    config = _Config(max_train_steps=steps, eval_every=1, seed=5, pos_logical_batch=4, pos_microbatch=4)
    return WanPosContextRegressionTrainer(
        config,
        predict_fn=_toy_predict,
        params=_toy_params(seed=9),
        tx=optax.adam(1e-2),
        manager=build_checkpoint_manager(str(directory)),
        selection_manager=build_selection_manager(str(directory)),
    )


def _settle(trainer):
    """Flush and CLOSE both trees, then report what survived.

    Closing matters: production resumes in a *new process*, so only one writer is ever live on a
    checkpoint directory. Keeping the old manager open while the resumed one writes is a test-only
    hazard, and one worth not having.
    """
    trainer.manager.wait_until_finished()
    trainer.selection_manager.wait_until_finished()
    surviving = {
        "latest": trainer.manager.latest_step(),
        "selection": trainer.selection_manager.latest_step(),
    }
    trainer.manager.close()
    trainer.selection_manager.close()
    return surviving


def test_the_selection_is_written_only_when_the_eval_is_a_new_best(tmp_path, monkeypatch):
    """``preserve_selection`` refuses non-improvements itself, so the loop's guard is redundant for
    *correctness* -- but not for cost: without it every eval of a 30k-step run pays a read of the
    incumbent selection JSON (a GCS round trip) to be told nothing changed."""
    from maxdiffusion.trainers import wan_pos_context_regression_trainer as trainer_module

    calls = []
    original = trainer_module.preserve_selection
    monkeypatch.setattr(
        trainer_module,
        "preserve_selection",
        lambda manager, state, **kwargs: calls.append(int(state.step)) or original(manager, state, **kwargs),
    )
    config = _Config(max_train_steps=3, eval_every=1, seed=0, pos_logical_batch=4, pos_microbatch=4)
    trainer, dev_evaluator = _trainer(tmp_path, config, [0.5, 0.9, 0.8])  # only the first eval is a best

    report = trainer.run(lambda seed: itertools.cycle([_batch(size=4)]), dev_evaluator)

    assert calls == [1], calls
    assert report.retained_step == 1


def test_the_interrupted_run_reproduces_the_uninterrupted_one(tmp_path):
    """**S7 review, BLOCKER 4 -- the integrated oracle.** Not a manual restore of hand-picked batches:
    the same ``run`` method, interrupted at an eval boundary and resumed from its own checkpoint, ends
    with the same eval history, the same stop decision, the same retained step and the same parameters
    as a run that was never interrupted.

    One repeated batch, so the train MSE falls monotonically and the rule's second conjunct is live;
    the stream is seed-independent so "identical continuation" is well defined (the seed rule itself is
    pinned by ``test_the_loop_rebuilds_the_iterator_at_the_resumed_seed``).
    """
    batch = _batch(size=4, seed=17)
    dev_values = [1.0, 0.3, 0.9, 1.4, 1.5, 1.6]
    stream = lambda seed: itertools.repeat(batch)  # noqa: E731 -- deliberately seed-independent

    whole_trainer = _resumable_trainer(tmp_path / "whole", 6)
    whole = whole_trainer.run(stream, _sequence_evaluator(dev_values))
    whole_survivors = _settle(whole_trainer)

    first_trainer = _resumable_trainer(tmp_path / "split", 3)
    first = first_trainer.run(stream, _sequence_evaluator(dev_values))
    _settle(first_trainer)
    assert first.steps_run == 3 and len(first.history) == 3

    second_trainer = _resumable_trainer(tmp_path / "split", 6)
    resumed = second_trainer.run(stream, _sequence_evaluator(dev_values[3:]))
    resumed_survivors = _settle(second_trainer)

    assert [record.step for record in resumed.history] == [record.step for record in whole.history]
    assert [record.dev_normalized_mse for record in resumed.history] == [
        record.dev_normalized_mse for record in whole.history
    ]
    assert resumed.verdict == whole.verdict  # same stop decision, same streak, same running best
    assert resumed.retained_step == whole.retained_step == 2
    assert np.allclose(np.asarray(resumed.state.params["w"]), np.asarray(whole.state.params["w"]), rtol=0, atol=1e-6)
    assert resumed_survivors["selection"] == whole_survivors["selection"] == 2


def test_reopening_a_terminal_checkpoint_takes_no_steps(tmp_path):
    """**S7 follow-up, the remaining MAJOR.** ``run`` computed the restored verdict and then trained
    anyway. At production cadence a retry after a terminal checkpoint spends another 1,000 optimizer
    steps and advances the resume state *past* the decision the run already made.

    The iterator factory here raises if it is so much as called: a terminal reopen must return before
    any data pipeline is built, let alone a batch pulled.
    """
    batch = _batch(size=4, seed=29)
    stream = lambda seed: itertools.repeat(batch)  # noqa: E731

    first_trainer = _resumable_trainer(tmp_path / "ckpt", 6)
    terminal = first_trainer.run(stream, _sequence_evaluator([1.0, 0.3, 0.9, 1.4, 1.5, 1.6]))
    survivors = _settle(first_trainer)
    assert terminal.verdict.stop and terminal.steps_run == 5

    def explode(seed):
        raise AssertionError("a terminal run must not build an input iterator")

    reopened_trainer = _resumable_trainer(tmp_path / "ckpt", 30_000)  # a much larger budget, as a retry has
    reopened = reopened_trainer.run(explode, _sequence_evaluator([0.01]))
    after = _settle(reopened_trainer)

    assert reopened.steps_run == 0
    assert reopened.verdict == terminal.verdict and reopened.retained_step == terminal.retained_step
    assert [record.step for record in reopened.history] == [record.step for record in terminal.history]
    assert after == survivors  # the resume state and the selection artifact are untouched on disk


def test_the_stop_rule_state_survives_the_interruption(tmp_path):
    """The rule is a function of the whole history, so a resumed run that started from an empty history
    could not see a degradation streak that began before the interruption -- it would train straight
    past its own stop condition."""
    batch = _batch(size=4, seed=23)
    stream = lambda seed: itertools.repeat(batch)  # noqa: E731

    # Best at step 2, then two degraded evals -- one short of the stop -- before the interruption.
    first_trainer = _resumable_trainer(tmp_path / "ckpt", 4)
    first = first_trainer.run(stream, _sequence_evaluator([1.0, 0.3, 0.9, 1.4]))
    _settle(first_trainer)
    assert not first.verdict.stop and first.verdict.streak == 2

    second_trainer = _resumable_trainer(tmp_path / "ckpt", 6)
    resumed = second_trainer.run(stream, _sequence_evaluator([1.5]))
    _settle(second_trainer)

    assert [record.step for record in resumed.history] == [1, 2, 3, 4, 5]  # the earlier evals came back
    assert resumed.verdict.stop and resumed.steps_run == 1  # the third consecutive one, and it stopped
    assert resumed.retained_step == 2


def test_the_run_report_is_a_frozen_record():
    config = _Config(max_train_steps=1, eval_every=1, seed=0, pos_logical_batch=4, pos_microbatch=4)
    trainer = WanPosContextRegressionTrainer(
        config, predict_fn=_toy_predict, params=_toy_params(), tx=optax.sgd(0.0), manager=None
    )

    report = trainer.run(lambda seed: iter([_batch(size=4)]), lambda state: 0.5)

    assert dataclasses.is_dataclass(report) and report.history[0].step == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.steps_run = 5
