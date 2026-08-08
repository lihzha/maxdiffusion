"""exp_06 `rollout_adapter` — T3b-4 `loop-and-selection`: the decision state (plan §3d).

The contract: **the loop's decision state survives interruption and selects the right checkpoint.**

Everything numeric was settled earlier; what is left is the part that decides, and deciding is where
a training job loses work silently. Two failures from exp_05 are the reason this round exists: a
resumed run that restarts its stop rule from an empty history trains straight through a degradation
streak that began before the interruption, and a resume tree ranked by quality evicts the NEWEST
state so a resumed run rewinds.

**The standing restored-``state.step`` obligation is discharged here**, and only here — T3b-1 could
demonstrate the hazard on a pure function but could not exercise a state object, a checkpoint or a
restore. Two tests close it: an AST pin on the loop's own call site (the derivation is fed the loop's
counter, and ``state.step`` never reaches it), and an **interrupted-vs-uninterrupted run through the
REAL Orbax restore path** asserting identical draws, history, verdict, retained step and parameters.

Model, optimizer and DEV instrument are seams, as exp_05's S7 loop had them: this round's contract is
decision state, and a stub update makes every decision edge reachable and exact. Where a number is
compared it is compared bitwise; nothing here is inferred from a trained proxy.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import pos_rollout_loop as loop
from maxdiffusion.pos_rollout_loop import EvalRecord, LoopSchedule, RolloutTrainState

_MODULE_PATH = Path(loop.__file__).resolve()
_SHAPE = (4, 3, 4, 6)


def _schedule(**overrides):
    base = {
        "max_train_steps": 6,
        "eval_every": 2,
        "logical_batch": 4,
        "microbatch": 2,
        "seed": 0,
        "arm": "rollout",
        "k_b": 2,
        "num_steps": 25,
    }
    base.update(overrides)
    return LoopSchedule(**base)


class _Stream:
    """An input pipeline whose CONTENT depends on the seed it was opened with, and that records the
    seed of every open.

    A factory that discards its seed cannot fail: the loop could switch to ``batches(schedule.seed)``
    and every test would stay green (review, MAJOR). Here the batch at offset ``i`` of a stream opened
    at ``seed`` carries the cursor ``seed + i``, which is exactly the dataloader-position convention
    ``resume_seed`` implements — so a resumed segment reads on iff it was opened at
    ``schedule.seed + start_step``, and the concatenated cursors of two segments equal the
    uninterrupted run's iff it did.
    """

    def __init__(self, schedule):
        self.schedule = schedule
        self.seeds: list[int] = []

    def __call__(self, seed):
        self.seeds.append(int(seed))
        return (self.batch(int(seed) + offset) for offset in itertools.count())

    def batch(self, cursor: int):
        width = self.schedule.logical_batch
        return {
            "cursor": jnp.full((width,), float(cursor), jnp.float32),
            "z_video": jnp.full((width, *_SHAPE), float(cursor), jnp.float32),
            "actions": jnp.zeros((width, 4, 7), jnp.float32),
        }


def _batches(schedule):
    return _Stream(schedule)


def _stub_update(state, batch_parts, draw_parts, schedule, global_step):
    """A deterministic 'update' that depends on the draws AND the data, so either changing shows."""
    del schedule
    delta = sum(float(part.epsilon.sum()) for part in draw_parts) + sum(
        float(part["cursor"][0]) for part in batch_parts
    )
    return dataclasses.replace(state, params=jax.tree.map(lambda leaf: leaf + delta, state.params)), 1.0 / float(
        global_step
    )


def _recording_update(seen: list):
    """``_stub_update``, plus a record of which data cursor each step consumed."""

    def update(state, batch_parts, draw_parts, schedule, global_step):
        seen.append(int(batch_parts[0]["cursor"][0]))
        return _stub_update(state, batch_parts, draw_parts, schedule, global_step)

    return update


def _state():
    # A pytree of arrays, as real adapter params are: Orbax's StandardCheckpointHandler refuses a
    # bare array or scalar, so a stub shaped like the real thing is also the only one that saves.
    return RolloutTrainState(
        params={"w": jnp.zeros((2,), jnp.float32)}, opt_state={"mu": jnp.zeros((2,), jnp.float32)}, step=0
    )


def _run(schedule, dev_values, *, stream=None, update_fn=_stub_update, **kwargs):
    values = iter(dev_values)
    return loop.run_loop(
        _state(),
        schedule,
        batches=_batches(schedule) if stream is None else stream,
        update_fn=update_fn,
        dev_metric_fn=lambda state, step: next(values),
        **kwargs,
    )


def _function_node(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


# =============================================================================================
# 1. THE open obligation: the loop's own step keys the randomness, never a restored state.step.
# =============================================================================================


def test_the_loop_feeds_the_key_derivation_its_own_counter_not_a_restored_state_step():
    """The production-callsite pin. ``state.step`` is not resume-safe and must never key a draw."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    run = _function_node(source, "run_loop")
    draws = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "draw_step_for_batch"
    ]
    assert len(draws) == 1, "the loop must obtain its randomness from exactly one seam"
    passed = {keyword.arg: keyword.value for keyword in draws[0].keywords}
    assert isinstance(passed["global_step"], ast.Name), ast.unparse(draws[0])
    assert passed["global_step"].id == "global_step"

    # `global_step` is bound from the loop counter, and `state.step` is never read for any purpose
    # other than being overwritten.
    assignments = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "global_step" for t in node.targets)
    ]
    assert len(assignments) == 1
    assert "state.step" not in ast.unparse(assignments[0])
    for node in ast.walk(run):
        if isinstance(node, ast.Attribute) and node.attr == "step" and isinstance(node.value, ast.Name):
            assert node.value.id != "state", f"the loop must not read state.step: {ast.unparse(node)}"


def test_an_interrupted_run_reproduces_the_uninterrupted_one_through_the_real_restore(tmp_path):
    """THE integrated oracle — the last thing standing between exp_06 and a training job.

    Not a simulation: a real Orbax resume tree, a real save at an eval boundary, a real restore into
    a fresh state, and then the run continued. Every decision surface is compared — the draws the
    loop consumed, the eval history, the stop verdict, the retained step, and the parameters.
    """
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    schedule = _schedule(max_train_steps=6, eval_every=2)
    dev_values = [0.9, 0.8, 0.7]

    uninterrupted = _run(schedule, dev_values)

    # ...and now the same run, cut in half at an eval boundary and resumed from disk.
    directory = str(tmp_path / "resume")
    manager = loop.build_checkpoint_manager(directory)
    first = _run(_schedule(max_train_steps=4, eval_every=2), dev_values[:2], manager=manager)
    assert first.steps_run == 4

    resumed_manager = loop.build_checkpoint_manager(directory)
    second = loop.run_loop(
        _state(),
        schedule,
        batches=_batches(schedule),
        update_fn=_stub_update,
        dev_metric_fn=lambda state, step: dev_values[2],
        manager=resumed_manager,
    )

    assert second.steps_run == 2, "the resumed run must continue, not restart"
    assert first.draw_log + second.draw_log == uninterrupted.draw_log, "the resumed stream diverged"
    assert second.history == uninterrupted.history, "the eval history did not survive the interruption"
    assert second.verdict == uninterrupted.verdict
    assert second.retained_step == uninterrupted.retained_step
    assert np.allclose(np.asarray(second.state.params["w"]), np.asarray(uninterrupted.state.params["w"]))
    assert int(second.state.step) == int(uninterrupted.state.step) == 6


def test_a_resumed_run_carries_the_stop_rules_decision_state(tmp_path):
    # The streak that began before the interruption must still be counting after it.
    pytest.importorskip("orbax.checkpoint")
    directory = str(tmp_path / "decision")
    manager = loop.build_checkpoint_manager(directory)
    _run(_schedule(max_train_steps=4, eval_every=2), [0.1, 1.0], manager=manager)
    restored = loop.restore_eval_history(loop.build_checkpoint_manager(directory))
    assert [record.step for record in restored] == [2, 4]
    assert [record.dev_metric for record in restored] == [0.1, 1.0]


def _boom(*args, **kwargs):
    raise AssertionError(f"a terminal reopen must take no step: {args} {kwargs}")


def test_reopening_a_terminal_checkpoint_takes_no_step_and_never_builds_an_iterator(tmp_path):
    """**The defect exp_05's S7 hit and fixed, ported here** (review BLOCKER B-1).

    ``run_loop`` computed the restored verdict and then entered the loop anyway: the reviewer
    restored a terminal step-4 history and observed ``[('iterator', 4), ('update', 5), ('save', 5)]``.
    At production cadence a retry after a terminal checkpoint spends another 1,000 optimizer steps and
    advances the resume state PAST the decision the run already made — so the next retry resumes from
    a checkpoint the stop rule never sanctioned.

    The iterator factory, the update and the DEV instrument all raise if they are so much as called:
    a terminal reopen must return before any input pipeline is built, let alone a batch pulled.
    """
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = str(tmp_path / "terminal")
    manager, selection = loop.build_checkpoint_manager(directory), loop.build_selection_manager(directory)
    terminal = _run(
        _schedule(max_train_steps=8, eval_every=1),
        [0.1, 1.0, 1.0, 1.0],
        manager=manager,
        selection_manager=selection,
    )
    assert terminal.verdict.stop and terminal.steps_run == 4
    survivors = (tuple(manager.all_steps()), tuple(selection.all_steps()))

    reopened_manager = loop.build_checkpoint_manager(directory)
    reopened_selection = loop.build_selection_manager(directory)
    reopened = loop.run_loop(
        _state(),
        _schedule(max_train_steps=30_000, eval_every=1),  # a much larger budget, as a retry has
        batches=_boom,
        update_fn=_boom,
        dev_metric_fn=_boom,
        manager=reopened_manager,
        selection_manager=reopened_selection,
    )

    assert reopened.steps_run == 0 and reopened.draw_log == ()
    assert reopened.verdict == terminal.verdict
    assert reopened.retained_step == terminal.retained_step == 1
    assert [record.step for record in reopened.history] == [record.step for record in terminal.history]
    # The resume state and the selection artifact are untouched on disk.
    assert (tuple(reopened_manager.all_steps()), tuple(reopened_selection.all_steps())) == survivors


def test_the_resumed_production_call_receives_the_resumed_seed_and_continues_the_data(tmp_path):
    """The production callsite, not the helper (review, MAJOR).

    ``resume_seed`` tested in isolation proves nothing about the loop: with a fixture that discards
    its seed, ``run_loop`` could pass ``schedule.seed`` and both tests would stay green. So the stream
    here records what it was opened with AND yields seed-dependent data, and the assertion is on the
    resumed PRODUCTION call plus the data sequence it produced.
    """
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    schedule = _schedule(max_train_steps=6, eval_every=2, seed=7)
    whole_seen: list[int] = []
    whole_stream = _Stream(schedule)
    whole = _run(schedule, [0.9, 0.8, 0.7], stream=whole_stream, update_fn=_recording_update(whole_seen))
    assert whole_stream.seeds == [7] and whole_seen == [7, 8, 9, 10, 11, 12]

    directory = str(tmp_path / "cursor")
    first_seen: list[int] = []
    first_stream = _Stream(schedule)
    _run(
        _schedule(max_train_steps=4, eval_every=2, seed=7),
        [0.9, 0.8],
        stream=first_stream,
        update_fn=_recording_update(first_seen),
        manager=loop.build_checkpoint_manager(directory),
    )
    second_seen: list[int] = []
    second_stream = _Stream(schedule)
    second = _run(
        schedule,
        [0.7],
        stream=second_stream,
        update_fn=_recording_update(second_seen),
        manager=loop.build_checkpoint_manager(directory),
    )

    assert second_stream.seeds == [11], "the resumed call must receive schedule.seed + start_step, not schedule.seed"
    assert first_seen + second_seen == whole_seen, "the resumed segment did not continue the data sequence"
    assert np.allclose(np.asarray(second.state.params["w"]), np.asarray(whole.state.params["w"]))


# =============================================================================================
# 2. Cadence.
# =============================================================================================


def test_the_default_cadence_is_the_plans_thousand_updates():
    assert loop.DEFAULT_EVAL_EVERY == 1000, "plan §3d pins evaluation and selection every 1,000 updates"


def test_evaluation_happens_on_the_cadence_and_always_at_the_last_step():
    schedule = _schedule(max_train_steps=5, eval_every=2)
    assert [step for step in range(1, 6) if loop.should_evaluate(step, schedule)] == [2, 4, 5]
    assert not loop.should_evaluate(0, schedule)


def test_both_arms_evaluate_on_the_identical_cadence():
    for arm in ("rollout", "one_step"):
        report = _run(_schedule(arm=arm, max_train_steps=6, eval_every=2), [0.9, 0.8, 0.7])
        assert [record.step for record in report.history] == [2, 4, 6], arm


# =============================================================================================
# 3. Stop rule edges (generalized from S7 to the DEV estimand).
# =============================================================================================


def _records(pairs):
    return [EvalRecord(step=index + 1, dev_metric=dev, train_metric=train) for index, (dev, train) in enumerate(pairs)]


def test_the_stop_rule_needs_the_full_patience_of_consecutive_degraded_evals():
    falling = [(0.1, 1.0), (1.0, 0.9), (1.0, 0.8)]  # exactly two degraded evals
    assert not loop.stop_verdict(_records(falling)).stop
    assert loop.stop_verdict(_records(falling + [(1.0, 0.7)])).stop


def test_a_train_metric_that_stops_falling_breaks_the_streak():
    # A plateauing train loss is a different failure and must not spend the patience budget.
    history = _records([(0.1, 1.0), (1.0, 0.9), (1.0, 0.9), (1.0, 0.8), (1.0, 0.7)])
    assert not loop.stop_verdict(history).stop


def test_the_first_evals_cannot_trigger_a_stop_before_a_best_exists():
    assert not loop.stop_verdict(_records([(5.0, 1.0)])).stop
    assert loop.stop_verdict(_records([])).best_step is None


def test_the_retained_step_is_the_best_and_the_earliest_wins_a_tie():
    history = _records([(0.5, 1.0), (0.2, 0.9), (0.2, 0.8), (0.9, 0.7)])
    assert loop.best_checkpoint_step(history) == 2, "a later tie must not displace the earliest best"
    assert loop.stop_verdict(history).best_value == 0.2


def _scripted_update(metrics):
    """``_stub_update`` with the train metric scripted, so the recorded window is exactly known."""
    stream = iter(metrics)

    def update(state, batch_parts, draw_parts, schedule, global_step):
        state, _ = _stub_update(state, batch_parts, draw_parts, schedule, global_step)
        return state, next(stream)

    return update


def test_the_recorded_train_metric_is_the_window_mean_since_the_previous_eval():
    """**S7's disjoint-window semantics, inherited** (review BLOCKER B-3).

    The rule asks whether the train metric is *still falling*, so the two numbers it compares must
    cover disjoint windows. This loop overwrote ``train_metric`` on every update and recorded the
    last one: for updates ``[100, 0, 50, 0]`` it recorded ``[0, 0]`` where the window means are
    ``[50, 25]``. A cumulative average is the other wrong answer — it smears the earliest steps into
    every later window and can never fall as fast as the data does.
    """
    report = _run(
        _schedule(max_train_steps=4, eval_every=2), [0.9, 0.8], update_fn=_scripted_update([100.0, 0.0, 50.0, 0.0])
    )
    recorded = [record.train_metric for record in report.history]
    assert recorded == [50.0, 25.0]
    assert recorded != [0.0, 0.0], "the last minibatch of a window is not the window"
    assert recorded != [50.0, 37.5], "a cumulative average smears every earlier step into the later window"


def test_the_window_mean_is_the_number_the_stop_rule_actually_reads():
    """The oracle that makes B-3 a decision defect rather than a reporting one.

    Update metrics ``[100, 0, 50, 0, 25, 0, 12, 0]`` fall window-over-window (50, 25, 12.5, 6) while
    every *last* value is 0 — and 0 < 0 is false, so "the train metric is still falling" would never
    hold and the degradation streak could never reach patience. The same run therefore stops under
    the window mean and never stops under the last minibatch.
    """
    report = _run(
        _schedule(max_train_steps=8, eval_every=2),
        [0.1, 1.0, 1.0, 1.0],
        update_fn=_scripted_update([100.0, 0.0, 50.0, 0.0, 25.0, 0.0, 12.0, 0.0]),
    )
    assert [record.train_metric for record in report.history] == [50.0, 25.0, 12.5, 6.0]
    assert report.verdict.stop and report.steps_run == 8
    # ...and with the last-minibatch numbers the identical DEV history does not stop.
    assert not loop.stop_verdict(_records([(0.1, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)])).stop


# =============================================================================================
# 4. Checkpoint discipline: recency to resume, immutable earliest-best to select.
# =============================================================================================


def test_the_resume_tree_keeps_the_newest_state_not_the_best(tmp_path):
    """exp_05's BLOCKER 3, generalized: ranking this tree by quality rewinds a resumed run."""
    pytest.importorskip("orbax.checkpoint")
    directory = str(tmp_path / "recency")
    manager = loop.build_checkpoint_manager(directory, max_to_keep=1)
    for step, metric in ((1, 0.1), (2, 0.9)):
        loop.save_checkpoint(
            manager, dataclasses.replace(_state(), step=step), dev_metric=metric, history=(), arm="rollout", k_b=2
        )
    assert manager.latest_step() == 2, "the newest state must remain restorable after preemption"
    # ...and the retention BOUND is honoured, so the survivor is the newest rather than merely
    # present: `latest_step()` alone would pass a tree that kept everything, which is a different
    # (and on a long run, storage-fatal) policy than the one this tree declares.
    assert tuple(manager.all_steps()) == (2,), "max_to_keep must retain exactly the newest state"


def test_the_resume_seed_carries_the_dataloader_position():
    """The cursor is not serialized (CLAUDE.md), so the seed is what makes a resumed run read on.

    Without this the resumed segment replays the opening segment's DATA while its randomness moves
    on -- the mirror image of the state.step hazard, and just as silent.
    """
    assert loop.resume_seed(0, 0) == 0
    assert loop.resume_seed(7, 4000) == 4007
    assert loop.resume_seed(0, 1) != loop.resume_seed(0, 0), "a resumed run must not restart the data"


def test_the_selection_artifact_is_a_sibling_and_survives_recency_eviction(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    directory = str(tmp_path / "sel")
    assert loop.selection_dir(directory).endswith("_selection")
    assert not loop.selection_dir(directory).startswith(directory + "/"), "the trees must not nest"
    selection = loop.build_selection_manager(directory)
    assert loop.preserve_selection(
        selection, dataclasses.replace(_state(), step=1), dev_metric=0.5, arm="rollout", k_b=2
    )
    # A worse metric and an equal metric are both refused: ties keep the EARLIEST best.
    assert not loop.preserve_selection(
        selection, dataclasses.replace(_state(), step=2), dev_metric=0.9, arm="rollout", k_b=2
    )
    assert not loop.preserve_selection(
        selection, dataclasses.replace(_state(), step=3), dev_metric=0.5, arm="rollout", k_b=2
    )
    assert selection.latest_step() == 1
    assert loop.preserve_selection(
        selection, dataclasses.replace(_state(), step=4), dev_metric=0.4, arm="rollout", k_b=2
    )
    assert selection.latest_step() == 4


def test_the_selection_artifact_is_reconciled_from_the_restored_state_before_training(tmp_path):
    """**The crash window** (review BLOCKER B-2): the resume tree is written, then the process dies
    before the sibling is updated, so the best state exists only in the resume tree.

    Startup is the ONLY place that can repair it — after the first optimizer step those parameters
    are gone. The reviewer simulated exactly this: history best = step 2, and the only selection call
    after the restart was the WORSE step 4, so an empty sibling ships step 4 and a populated one goes
    stale. Here the first segment runs with the resume manager alone, which is what a crash between
    the two saves leaves behind.
    """
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = str(tmp_path / "crash")
    _run(_schedule(max_train_steps=2, eval_every=2), [0.5], manager=loop.build_checkpoint_manager(directory))
    selection = loop.build_selection_manager(directory)
    assert selection.latest_step() is None, "the crash window: the sibling was never written"

    report = _run(
        _schedule(max_train_steps=4, eval_every=2),
        [0.9],  # the next evaluation is WORSE than the best that was lost
        manager=loop.build_checkpoint_manager(directory),
        selection_manager=selection,
    )

    assert report.retained_step == 2
    selection.wait_until_finished()
    assert selection.latest_step() == 2, "an unreconciled sibling ships the worse step 4"
    payload = loop.read_checkpoint_json(selection, 2)
    assert payload[loop.DEV_METRIC] == 0.5 and payload["arm"] == "rollout"


def test_a_sibling_that_already_holds_the_recorded_best_is_accepted_and_left_alone(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = str(tmp_path / "consistent")
    _run(
        _schedule(max_train_steps=4, eval_every=2),
        [0.5, 0.9],
        manager=loop.build_checkpoint_manager(directory),
        selection_manager=loop.build_selection_manager(directory),
    )
    selection = loop.build_selection_manager(directory)
    assert selection.latest_step() == 2
    assert (
        loop.reconcile_selection(
            selection,
            dataclasses.replace(_state(), step=4),
            loop.restore_eval_history(loop.build_checkpoint_manager(directory)),
            arm="rollout",
            k_b=2,
        )
        == "consistent"
    )

    report = _run(
        _schedule(max_train_steps=6, eval_every=2),
        [0.8],
        manager=loop.build_checkpoint_manager(directory),
        selection_manager=selection,
    )
    assert report.retained_step == 2 and selection.latest_step() == 2


@pytest.mark.parametrize("stale", [False, True])
def test_reconciliation_fails_closed_when_the_sibling_cannot_be_repaired(tmp_path, stale):
    """The historical best is an EARLIER step, so the restored state cannot repair the sibling.

    Failing closed is the only honest option: the run would otherwise ship whatever the sibling holds
    (nothing, or a worse checkpoint) while the report names a step nobody can produce.
    """
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = str(tmp_path / f"closed_{stale}")
    _run(
        _schedule(max_train_steps=4, eval_every=2), [0.5, 0.9], manager=loop.build_checkpoint_manager(directory)
    )  # history best = step 2; the restored state is step 4
    selection = loop.build_selection_manager(directory)
    if stale:  # a sibling holding a real-but-not-best evaluation is not good enough either
        loop.save_checkpoint(
            selection, dataclasses.replace(_state(), step=4), dev_metric=0.9, history=(), arm="rollout", k_b=2
        )
        assert selection.latest_step() == 4

    with pytest.raises(RuntimeError, match="the historical best is step 2"):
        _run(
            _schedule(max_train_steps=6, eval_every=2),
            [0.7],
            manager=loop.build_checkpoint_manager(directory),
            selection_manager=selection,
        )


def test_reconciliation_refuses_a_sibling_the_history_cannot_account_for(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = str(tmp_path / "foreign")
    selection = loop.build_selection_manager(directory)
    loop.save_checkpoint(
        selection, dataclasses.replace(_state(), step=9), dev_metric=0.1, history=(), arm="rollout", k_b=2
    )
    with pytest.raises(RuntimeError, match="no evaluation history"):
        loop.reconcile_selection(selection, _state(), [], arm="rollout", k_b=2)
    history = [EvalRecord(step=2, dev_metric=0.5, train_metric=1.0)]
    with pytest.raises(RuntimeError, match="the sibling holds step 9"):
        loop.reconcile_selection(selection, dataclasses.replace(_state(), step=2), history, arm="rollout", k_b=2)


def test_reconciliation_is_a_no_op_without_a_selection_manager_or_a_history():
    assert loop.reconcile_selection(None, _state(), [], arm="rollout", k_b=2) == "no_selection"


@pytest.mark.parametrize("arm", ["rollout", "one_step"])
def test_checkpoint_metadata_carries_the_metric_the_arm_and_the_horizon(tmp_path, arm):
    pytest.importorskip("orbax.checkpoint")
    manager = loop.build_checkpoint_manager(str(tmp_path / f"meta_{arm}"))
    loop.save_checkpoint(
        manager,
        dataclasses.replace(_state(), step=3),
        dev_metric=0.25,
        history=[EvalRecord(step=3, dev_metric=0.25, train_metric=1.0)],
        arm=arm,
        k_b=4,
    )
    payload = loop.read_checkpoint_json(manager, 3)
    assert payload["arm"] == arm and payload["k_b"] == 4
    assert payload[loop.DEV_METRIC] == 0.25 and loop.DEV_METRIC == "dev_rollout_loss"
    assert payload["eval_history"] == [[3, 0.25, 1.0]]


# =============================================================================================
# 5. Arm parity at LOOP level, and config hygiene.
# =============================================================================================


def test_switching_arms_changes_only_the_loss_at_loop_level():
    """Loop-level parity: same draws, same cadence, same selection, same retention."""
    reports = {
        arm: _run(_schedule(arm=arm, max_train_steps=6, eval_every=2), [0.9, 0.8, 0.7])
        for arm in ("rollout", "one_step")
    }
    assert reports["rollout"].draw_log == reports["one_step"].draw_log, "an arm perturbed the stream"
    assert reports["rollout"].history == reports["one_step"].history
    assert reports["rollout"].verdict == reports["one_step"].verdict
    assert reports["rollout"].retained_step == reports["one_step"].retained_step
    # The arm reaches exactly one place in the loop: the update (and the metadata that records it).
    run = _function_node(_MODULE_PATH.read_text(encoding="utf-8"), "run_loop")
    branches = [node for node in ast.walk(run) if isinstance(node, ast.If) and "arm" in ast.unparse(node.test)]
    assert not branches, "the loop must not branch on the arm"


def test_the_schedule_reads_config_safely_and_refuses_an_unknown_arm():
    class _Config:
        def __init__(self, **values):
            self.__dict__.update(values)

    schedule = LoopSchedule.from_config(_Config(max_train_steps=10, seed=3))
    assert schedule.eval_every == loop.DEFAULT_EVAL_EVERY and schedule.arm == "rollout" and schedule.k_b == 2
    with pytest.raises(ValueError, match="unknown arm"):
        LoopSchedule.from_config(_Config(pos_rollout_arm="corrective_ss"))
    # issue #11: no three-argument getattr on a pyconfig HyperParameters anywhere in the module.
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, ast.unparse(node)


def test_the_loop_obtains_its_randomness_only_through_the_public_stream_seam():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("pos_rollout_stream")
        for alias in node.names
    }
    assert imported == {"draw_step_for_batch"}, "the loop must use the safe seam and nothing else"
    assert "jax.random" not in source
    for private in ("_draw_step_stream", "_rollout_epsilon", "_split_draws", "_one_step_timestep_indices"):
        assert private not in source, private


# =============================================================================================
# 6. Non-finite metrics may not enter a decision, a history, or a checkpoint (review BLOCKER 2).
# =============================================================================================


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_dev_metric_can_never_become_an_unbeatable_best(bad):
    """The defect: ``NaN`` compares false against everything, so ``stop_verdict([nan, 0.1])`` returned
    ``best_value=nan`` and NO later finite value could displace it — while ``preserve_selection``
    still replaced the sibling on the finite value. The report and the shipped artifact then
    disagree, the next reconciliation fails closed, and one bad number decided the whole run.
    """
    poisoned = [
        EvalRecord(step=1000, dev_metric=bad, train_metric=1.0),
        EvalRecord(step=2000, dev_metric=0.1, train_metric=0.9),
    ]
    with pytest.raises(ValueError, match="not finite"):
        loop.stop_verdict(poisoned)
    with pytest.raises(ValueError, match="not finite"):
        loop.best_checkpoint_step(poisoned)
    # The train metric is the other half of the rule and is refused identically.
    with pytest.raises(ValueError, match="the train metric at step 1000"):
        loop.stop_verdict([EvalRecord(step=1000, dev_metric=0.1, train_metric=bad)])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_neither_checkpoint_tree_can_be_written_with_a_non_finite_metric(tmp_path, bad):
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    manager = loop.build_checkpoint_manager(str(tmp_path / f"nf_{bad}"))
    with pytest.raises(ValueError, match="not finite"):
        loop.save_checkpoint(manager, _state(), dev_metric=bad, history=(), arm="rollout", k_b=2)
    selection = loop.build_selection_manager(str(tmp_path / f"nf_sel_{bad}"))
    with pytest.raises(ValueError, match="not finite"):
        loop.preserve_selection(selection, _state(), dev_metric=bad, arm="rollout", k_b=2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_the_loop_refuses_a_non_finite_metric_before_it_reaches_the_history(bad):
    with pytest.raises(ValueError, match="the DEV metric at step 2"):
        _run(_schedule(max_train_steps=4, eval_every=2), [bad])
    with pytest.raises(ValueError, match="the train window mean at step 2"):
        _run(_schedule(max_train_steps=4, eval_every=2), [0.5], update_fn=_scripted_update([bad, 1.0, 1.0, 1.0]))


def test_a_poisoned_history_cannot_be_restored_into_a_fresh_process(tmp_path):
    """A NaN that reached an artifact before this guard existed must not re-enter a resumed run."""
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")
    directory = str(tmp_path / "poisoned")
    manager = loop.build_checkpoint_manager(directory)
    import orbax.checkpoint as ocp

    manager.save(
        2,
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(_state().params),
            opt_state=ocp.args.StandardSave(_state().opt_state),
            step=ocp.args.JsonSave(
                {"step": 2, loop.DEV_METRIC: 0.5, "arm": "rollout", "k_b": 2, "eval_history": [[2, float("nan"), 1.0]]}
            ),
        ),
    )
    manager.wait_until_finished()
    with pytest.raises(ValueError, match="the restored DEV metric at step 2"):
        loop.restore_eval_history(loop.build_checkpoint_manager(directory))


def test_a_non_finite_metric_is_refused_BEFORE_selection_reads_its_incumbent(tmp_path):
    """`nan >= previous` is false, so a NaN reads as a strict improvement. The refusal must therefore
    precede the comparison, not merely precede the write."""
    pytest.importorskip("orbax.checkpoint")
    pytest.importorskip("tensorflow")

    class _Exploding:
        def latest_step(self):
            raise AssertionError("selection must refuse a non-finite metric before consulting the incumbent")

    with pytest.raises(ValueError, match="offered to selection"):
        loop.preserve_selection(_Exploding(), _state(), dev_metric=float("nan"), arm="rollout", k_b=2)
