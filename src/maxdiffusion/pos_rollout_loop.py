"""exp_06 `rollout_adapter` — T3b-4: the training loop and its selection discipline (plan §3d).

**One contract: the loop's decision state survives interruption and selects the right checkpoint.**

Everything numeric was settled in earlier rounds — T3b-1 owns the stream, T3b-2 the two arms, T3b-3
the DEV estimand. What is left is the part that decides things, and deciding is where a training job
loses work silently: a resumed run that restarts its stop rule from an empty history trains straight
through a degradation streak that began before the interruption, and a retention policy that ranks
the resume tree by quality evicts the newest state and rewinds the run.

**Two checkpoint trees, because they answer different questions** (exp_05's S7 BLOCKER 3, generalized
here rather than copied): the RESUME tree is kept by **recency**, so the newest state is always
restorable after preemption; the SELECTION artifact is a **sibling** path holding one immutable
checkpoint, written only on a STRICT improvement so a later tie cannot displace the earliest best.
They are siblings and never nested, because Orbax treats every subdirectory of a manager's root as
its own tree and two nested managers delete each other's in-flight temporary directories.

**The decision state rides inside the step item**, written atomically with the state it describes:
the eval history, the DEV metric, and — new here — the ARM and the horizon ``k``. A checkpoint that
cannot say which arm and which k produced it is not comparable against the other arm's, which is the
entire point of the matched-C0 design.

**THE restored-``state.step`` obligation is discharged here** (T1 reviewer's standing item, carried
open since T3b-1). The loop feeds the aux-key derivation **its own counter**, never
``state.step``: Orbax restores params/opt_state/step, and a freshly built state's ``step`` counts
from zero, so keying randomness on it makes a resumed segment re-consume the opening segment's
epsilons while the loader serves later data — silently, and identically in both arms, so no arm
comparison could reveal it. Two tests hold this down: an AST pin on this file's own call site, and an
interrupted-vs-uninterrupted run through the REAL restore path.

**Arm parity.** The arm reaches the loop as one parameter and is forwarded to one place: the loss.
Cadence, stream, selection rule, retention and metadata are arm-independent by construction.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

import jax

from maxdiffusion.pos_rollout_arms import ARMS
from maxdiffusion.pos_rollout_stream import draw_step_for_batch
from maxdiffusion.run_wan_null_inversion import optional_config_value

__all__ = [
    "DEFAULT_EVAL_EVERY",
    "DEV_METRIC",
    "STOP_FACTOR",
    "STOP_PATIENCE",
    "EvalRecord",
    "LoopSchedule",
    "RolloutTrainState",
    "RunReport",
    "StopVerdict",
    "best_checkpoint_step",
    "build_checkpoint_manager",
    "build_selection_manager",
    "preserve_selection",
    "restore_checkpoint",
    "restore_eval_history",
    "run_loop",
    "save_checkpoint",
    "selection_dir",
    "should_evaluate",
    "stop_verdict",
]

#: plan §3d: "evaluation and checkpoint selection every 1,000 updates, identically for R-B and
#: matched-C0". The YAML key that carries it lands at T4; until then this default IS the contract,
#: and a test pins the number so a drifting default cannot pass unnoticed.
DEFAULT_EVAL_EVERY = 1000
DEFAULT_LOGICAL_BATCH = 256
STOP_FACTOR = 2.0
STOP_PATIENCE = 3
CHECKPOINT_ITEMS = ("params", "opt_state", "step")
SELECTION_SUFFIX = "_selection"
#: The metric name in checkpoint metadata: T3b-3's fixed-draw DEV-64 rollout loss, NOT S7's
#: cached-target normalized MSE (which has no referent in exp_06).
DEV_METRIC = "dev_rollout_loss"


@dataclasses.dataclass(frozen=True)
class LoopSchedule:
    """Cadence, batching and arm, read from config once so no literal is scattered through the loop."""

    max_train_steps: int
    eval_every: int
    logical_batch: int
    microbatch: Any
    seed: int
    arm: str
    k_b: int
    num_steps: int

    @classmethod
    def from_config(cls, config: Any) -> "LoopSchedule":
        arm = str(optional_config_value(config, "pos_rollout_arm", ARMS[0]))
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; declared arms are {list(ARMS)}")
        return cls(
            max_train_steps=int(optional_config_value(config, "max_train_steps", 0)),
            eval_every=int(optional_config_value(config, "eval_every", DEFAULT_EVAL_EVERY)),
            logical_batch=int(optional_config_value(config, "pos_logical_batch", DEFAULT_LOGICAL_BATCH)),
            microbatch=optional_config_value(config, "pos_microbatch", None),
            seed=int(optional_config_value(config, "seed", 0)),
            arm=arm,
            k_b=int(optional_config_value(config, "pos_rollout_k", 2)),
            num_steps=int(optional_config_value(config, "side_adapter_sampling_steps", 25)),
        )


@dataclasses.dataclass(frozen=True)
class EvalRecord:
    step: int
    dev_metric: float
    train_metric: float


@dataclasses.dataclass(frozen=True)
class StopVerdict:
    stop: bool
    reason: str
    streak: int
    best_step: int | None
    best_value: float | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RolloutTrainState:
    params: Any
    opt_state: Any
    step: Any


@dataclasses.dataclass(frozen=True)
class RunReport:
    state: RolloutTrainState
    history: tuple[EvalRecord, ...]
    verdict: StopVerdict
    retained_step: int | None
    steps_run: int
    #: A per-step fingerprint of the randomness the loop consumed, so "the resumed run drew what the
    #: uninterrupted run drew" is a comparison over values rather than a hope.
    draw_log: tuple[tuple[int, int, int, float], ...]


def should_evaluate(step: int, schedule: LoopSchedule) -> bool:
    """DEV is measured on the cadence, and always at the last step: selection needs it measured."""
    return step >= 1 and (step % schedule.eval_every == 0 or step == schedule.max_train_steps)


def resume_seed(config_seed: int, start_step: int) -> int:
    """The dataloader cursor is not serialized, so the seed carries the position (CLAUDE.md)."""
    return int(config_seed) + int(start_step)


def stop_verdict(history: Sequence[EvalRecord], *, factor: float = STOP_FACTOR, patience: int = STOP_PATIENCE):
    """S7's rule, GENERALIZED to T3b-3's DEV estimand — pure, so its edges are testable.

    Stop when the DEV metric exceeds ``factor`` x its **running best over the earlier evals** on
    ``patience`` consecutive evals *while the train metric is still falling*. The conjunction is the
    point: a train loss that plateaus is a different failure and must not spend the patience budget,
    and the first eval can never trigger (no prior best, no previous train value). The retained
    checkpoint is the best DEV metric seen up to the decision — never one of the degraded evals.
    """
    best_step = best_value = None
    streak = 0
    for index, record in enumerate(history):
        degraded = best_value is not None and record.dev_metric > factor * best_value
        falling = index > 0 and record.train_metric < history[index - 1].train_metric
        streak = streak + 1 if (degraded and falling) else 0
        if best_value is None or record.dev_metric < best_value:
            best_step, best_value = record.step, record.dev_metric
        if streak >= patience:
            break
    stop = streak >= patience
    reason = (
        f"the DEV metric stayed above {factor}x its running best for {patience} consecutive evals "
        f"while the train metric kept falling; retaining step {best_step}"
        if stop
        else "the budget governs"
    )
    return StopVerdict(stop=stop, reason=reason, streak=streak, best_step=best_step, best_value=best_value)


def best_checkpoint_step(history: Sequence[EvalRecord]) -> int | None:
    """Selection = best DEV metric; the EARLIEST step wins a tie (strict-improvement rule)."""
    return stop_verdict(history).best_step


def selection_dir(ckpt_dir: str) -> str:
    """A SIBLING of the resume tree, never a child: nested Orbax roots delete each other's temps."""
    return f"{str(ckpt_dir).rstrip('/')}{SELECTION_SUFFIX}"


def _manager(ckpt_dir: str, *, max_to_keep: int):
    import orbax.checkpoint as ocp
    import tensorflow as tf

    tf.io.gfile.makedirs(ckpt_dir)
    return ocp.CheckpointManager(
        ckpt_dir,
        item_names=CHECKPOINT_ITEMS,
        item_handlers={
            "params": ocp.StandardCheckpointHandler(),
            "opt_state": ocp.StandardCheckpointHandler(),
            "step": ocp.JsonCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(create=True, max_to_keep=max_to_keep, enable_async_checkpointing=False),
    )


def build_checkpoint_manager(ckpt_dir: str, *, max_to_keep: int = 3):
    """The RESUME tree, kept by RECENCY.

    Ranking this tree by quality is the failure exp_05 caught: ``max_to_keep`` would then evict the
    NEWEST state and a resumed run would silently rewind. Quality is a different question and has its
    own tree below.
    """
    return _manager(ckpt_dir, max_to_keep=max_to_keep)


def build_selection_manager(ckpt_dir: str):
    """The SELECTION artifact: ONE immutable checkpoint, the EARLIEST best DEV metric."""
    return _manager(selection_dir(ckpt_dir), max_to_keep=1)


def save_checkpoint(manager, state: RolloutTrainState, *, dev_metric=None, history=(), arm: str, k_b: int) -> None:
    """Adapter params, optimizer state, and a step JSON carrying the whole decision state.

    The history/metric/arm/k live INSIDE the step item so they are written atomically with the state
    they describe and cannot influence retention. ``arm`` and ``k_b`` are recorded because a
    checkpoint that cannot say which arm and horizon produced it is not comparable with the other
    arm's — which is the matched-C0 design's whole purpose.
    """
    import orbax.checkpoint as ocp

    manager.save(
        int(state.step),
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(state.params),
            opt_state=ocp.args.StandardSave(state.opt_state),
            step=ocp.args.JsonSave(
                {
                    "step": int(state.step),
                    DEV_METRIC: None if dev_metric is None else float(dev_metric),
                    "arm": str(arm),
                    "k_b": int(k_b),
                    "eval_history": [
                        [int(record.step), float(record.dev_metric), float(record.train_metric)] for record in history
                    ],
                }
            ),
        ),
    )
    manager.wait_until_finished()


def read_checkpoint_json(manager, step: int) -> dict:
    import orbax.checkpoint as ocp

    return dict(manager.restore(step, args=ocp.args.Composite(step=ocp.args.JsonRestore())))["step"]


def preserve_selection(
    manager, state: RolloutTrainState, *, dev_metric: float, history=(), arm: str, k_b: int
) -> bool:
    """Write the selection checkpoint iff ``dev_metric`` STRICTLY improves. Ties keep the earliest."""
    incumbent = manager.latest_step()
    if incumbent is not None:
        previous = read_checkpoint_json(manager, incumbent).get(DEV_METRIC)
        if previous is not None and float(dev_metric) >= float(previous):
            return False
    save_checkpoint(manager, state, dev_metric=dev_metric, history=history, arm=arm, k_b=k_b)
    return True


def restore_checkpoint(manager, state: RolloutTrainState) -> tuple[RolloutTrainState, int]:
    import orbax.checkpoint as ocp

    latest = manager.latest_step()
    if latest is None:
        return state, 0
    restored = manager.restore(
        latest,
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(state.params),
            opt_state=ocp.args.StandardRestore(state.opt_state),
            step=ocp.args.JsonRestore(),
        ),
    )
    step = int(restored["step"]["step"])
    return dataclasses.replace(state, params=restored["params"], opt_state=restored["opt_state"], step=step), step


def restore_eval_history(manager) -> list[EvalRecord]:
    """The stop rule's decision state. Without it a resumed run restarts from an empty history and
    trains straight past a degradation streak that began before the interruption."""
    latest = manager.latest_step()
    if latest is None:
        return []
    rows = read_checkpoint_json(manager, latest).get("eval_history") or []
    return [EvalRecord(step=int(s), dev_metric=float(d), train_metric=float(t)) for s, d, t in rows]


def run_loop(
    state: RolloutTrainState,
    schedule: LoopSchedule,
    *,
    batches: Callable[[int], Any],
    update_fn: Callable[..., tuple[RolloutTrainState, float]],
    dev_metric_fn: Callable[..., float],
    manager=None,
    selection_manager=None,
) -> RunReport:
    """The loop. Model, optimizer and DEV instrument arrive as seams, exactly as S7's did.

    ``update_fn(state, microbatches, draws_parts, schedule, global_step) -> (state, train_metric)``
    is the arm's step; ``dev_metric_fn(state, global_step) -> float`` is T3b-3's fixed-draw estimand.
    Both are injected so this round's contract — the decision state — is testable without a model,
    and so the arm reaches exactly one place.
    """
    history: list[EvalRecord] = []
    start_step = 0
    if manager is not None:
        state, start_step = restore_checkpoint(manager, state)
        history = restore_eval_history(manager)

    verdict = stop_verdict(history)
    draw_log: list[tuple[int, int, int, float]] = []
    iterator = batches(resume_seed(schedule.seed, start_step))
    steps_run = 0

    for offset in range(schedule.max_train_steps - start_step):
        # THE loop's own counter. `state.step` is NOT resume-safe and must never key randomness.
        global_step = start_step + offset + 1
        batch = next(iterator)
        _, draw_parts, batch_parts = draw_step_for_batch(
            batch,
            seed=schedule.seed,
            global_step=global_step,
            logical_batch=schedule.logical_batch,
            microbatch=schedule.microbatch,
            num_steps=schedule.num_steps,
            k_b=schedule.k_b,
        )
        state, train_metric = update_fn(state, batch_parts, draw_parts, schedule, global_step)
        state = dataclasses.replace(state, step=global_step)
        steps_run += 1
        first = draw_parts[0]
        draw_log.append(
            (global_step, int(first.support_start), int(first.t_idx[0]), round(float(first.epsilon.sum()), 6))
        )

        if should_evaluate(global_step, schedule):
            history.append(
                EvalRecord(
                    step=global_step,
                    dev_metric=float(dev_metric_fn(state, global_step)),
                    train_metric=float(train_metric),
                )
            )
            if manager is not None:
                save_checkpoint(
                    manager,
                    state,
                    dev_metric=history[-1].dev_metric,
                    history=history,
                    arm=schedule.arm,
                    k_b=schedule.k_b,
                )
            if selection_manager is not None:
                preserve_selection(
                    selection_manager,
                    state,
                    dev_metric=history[-1].dev_metric,
                    history=history,
                    arm=schedule.arm,
                    k_b=schedule.k_b,
                )
            verdict = stop_verdict(history)
            if verdict.stop:
                break

    return RunReport(
        state=state,
        history=tuple(history),
        verdict=verdict,
        retained_step=best_checkpoint_step(history),
        steps_run=steps_run,
        draw_log=tuple(draw_log),
    )
