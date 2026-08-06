"""exp_05 S7 — teacher-forced regression trainer for the pre_context adapter (plan §4-P3').

The objective's data side is S6 (``pos_context_regression``); this module is the **state machine** on
top of it: the accumulated train step, the DEV cadence, the stop rule, and adapter-only checkpointing.
Dispatch and config keys are S8, so nothing here reads ``train_wan.py`` or constructs a pipeline.

Four contracts, each pinned by its own oracle in ``test_pos_context_trainer.py``:

- **The frozen split.** ``build_pre_context_predict_fn`` closes over the transformer's parameters and
  takes only the adapter's as an argument, so the ~5B backbone is not merely *untrained*, it is not
  differentiable from here; ``_first_block_self_attention_features`` additionally returns its features
  under ``stop_gradient``. The optimizer is initialized from the adapter tree alone.
- **Accumulation preserves the logical batch (F3).** ``microbatch x accumulation_steps == logical
  batch``; any request that cannot be expressed that way is refused rather than rounded.
- **The stop rule (F2) is pure.** ``stop_verdict`` reads an eval history and returns a decision; the
  loop only obeys it. That is what makes "3 consecutive" and "retain the prior best" testable without
  running 30k steps.
- **Adapter-only checkpoints.** ``params``/``opt_state``/``step`` -- the side-adapter trainer's exact
  item set. The input iterator is deliberately absent, so a resumed run rebuilds it at
  ``seed = config.seed + start_step``.

The model is a seam: production passes the real pre_context closure, tests pass a toy. The train step
is a pure function of ``(state, batch)`` with its seams bound by keyword, so S8/K3 can ``jax.jit`` it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterator, Sequence

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tensorflow as tf

from maxdiffusion.pos_context_regression import per_example_regression_loss, regression_loss
from maxdiffusion.run_wan_null_inversion import optional_config_value

# Plan §4-P3': the fixed budget's cadence (F2) and the logical global batch (F3).
DEFAULT_EVAL_EVERY = 1000
DEFAULT_LOGICAL_BATCH = 256
STOP_FACTOR = 2.0
STOP_PATIENCE = 3
CHECKPOINT_ITEMS = ("params", "opt_state", "step")
SELECTION_SUFFIX = "_selection"
DEV_METRIC = "dev_normalized_mse"


@dataclasses.dataclass(frozen=True)
class TrainingSchedule:
    """Cadence and batching, read from config once so no literal is scattered through the loop."""

    max_train_steps: int
    eval_every: int
    logical_batch: int
    microbatch: int
    accumulation_steps: int
    seed: int

    @classmethod
    def from_config(cls, config: Any) -> "TrainingSchedule":
        logical = int(optional_config_value(config, "pos_logical_batch", DEFAULT_LOGICAL_BATCH))
        microbatch, steps = accumulation_plan(logical, optional_config_value(config, "pos_microbatch", None))
        return cls(
            max_train_steps=int(optional_config_value(config, "max_train_steps", 0)),
            eval_every=int(optional_config_value(config, "eval_every", DEFAULT_EVAL_EVERY)),
            logical_batch=logical,
            microbatch=microbatch,
            accumulation_steps=steps,
            seed=int(optional_config_value(config, "seed", 0)),
        )


@dataclasses.dataclass(frozen=True)
class EvalRecord:
    step: int
    dev_normalized_mse: float
    train_mse: float


@dataclasses.dataclass(frozen=True)
class StopVerdict:
    stop: bool
    reason: str
    streak: int
    best_step: int | None
    best_value: float | None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RegressionTrainState:
    """The whole trainable state, and a pytree so ``jax.jit(train_step)`` can take and return it."""

    params: Any
    opt_state: Any
    step: Any  # a traced scalar inside jit, an int on the host; ``int(state.step)`` at the boundary


@dataclasses.dataclass(frozen=True)
class RunReport:
    state: RegressionTrainState
    history: tuple[EvalRecord, ...]
    verdict: StopVerdict
    retained_step: int | None
    steps_run: int


def accumulation_plan(logical_batch: int, microbatch: Any) -> tuple[int, int]:
    """``(microbatch, accumulation_steps)`` -- F3's fallback, which never changes the logical batch."""
    logical = int(logical_batch)
    if logical < 1:
        raise ValueError(f"the logical batch must be positive, got {logical}")
    if microbatch is None or int(microbatch) == logical:
        return logical, 1
    size = int(microbatch)
    if size < 1:
        raise ValueError(f"the microbatch must be positive, got {size}")
    if size > logical:
        raise ValueError(f"the microbatch {size} is larger than the logical batch {logical}")
    if logical % size:
        raise ValueError(
            f"the microbatch {size} does not divide the logical batch {logical}: accumulation preserves "
            f"the logical batch size, it never rounds it"
        )
    return size, logical // size


def should_evaluate(step: int, schedule: TrainingSchedule) -> bool:
    """DEV is measured on the cadence, and always at the last step: selection needs it measured."""
    return step >= 1 and (step % schedule.eval_every == 0 or step == schedule.max_train_steps)


def resume_seed(config_seed: int, start_step: int) -> int:
    """The dataloader cursor is not serialized, so the seed carries the position (CLAUDE.md)."""
    return int(config_seed) + int(start_step)


def microbatches(batch: Any, count: int) -> list[Any]:
    """Split one gathered batch into ``count`` equal parts, preserving order."""
    size = len(batch.names)
    if count < 1 or size % count:
        raise ValueError(f"{count} microbatches does not divide a batch of {size} examples")
    width = size // count
    # Every field of a gathered batch is example-major and sliceable, so the split stays correct if
    # the batch ever grows a field -- and it cannot silently drop one.
    windows = (slice(index * width, (index + 1) * width) for index in range(count))
    fields = [field.name for field in dataclasses.fields(batch)]
    return [
        dataclasses.replace(batch, **{name: getattr(batch, name)[window] for name in fields}) for window in windows
    ]


def checked_training_batch(batch: Any, schedule: TrainingSchedule) -> Any:
    """Refuse a batch that is not the configured logical batch (S7 review, BLOCKER 2).

    ``accumulation_plan`` only proves the *configuration* is expressible; nothing until here compared
    it with what the iterator actually yields. A 128-example batch under ``logical=256, microbatch=64``
    would otherwise be split into four 32s -- a halved logical batch, silently, which is exactly the
    substitution F3 forbids.
    """
    size = len(batch.names)
    if size != schedule.logical_batch:
        raise ValueError(
            f"the iterator yielded {size} examples but the logical batch is {schedule.logical_batch}: "
            f"accumulation preserves the logical batch, it never adapts to the data"
        )
    if size // schedule.accumulation_steps != schedule.microbatch:
        raise ValueError(
            f"{schedule.accumulation_steps} microbatches of {schedule.logical_batch} examples are not "
            f"the configured microbatch width {schedule.microbatch}"
        )
    return batch


def train_step(
    state: RegressionTrainState,
    batch: Any,
    *,
    predict_fn: Callable[[Any, Any], jax.Array],
    tx: optax.GradientTransformation,
    accumulation_steps: int = 1,
) -> tuple[RegressionTrainState, dict]:
    """One optimizer step over the logical batch, optionally accumulated over equal microbatches."""

    def loss_of(params, part):
        return regression_loss(predict_fn(params, part), part.target_context)

    grad_fn = jax.value_and_grad(loss_of)
    parts = microbatches(batch, accumulation_steps) if accumulation_steps > 1 else [batch]
    loss, grads = grad_fn(state.params, parts[0])
    for part in parts[1:]:
        part_loss, part_grads = grad_fn(state.params, part)
        loss = loss + part_loss
        grads = jax.tree.map(jnp.add, grads, part_grads)
    # Equal-sized microbatches, so the mean of the parts IS the whole batch's mean.
    scale = 1.0 / len(parts)
    loss, grads = loss * scale, jax.tree.map(lambda leaf: leaf * scale, grads)

    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    # JAX scalars, not floats: ``float()`` on a tracer is a concretization error, so the conversion
    # belongs to the host loop (S7 review, BLOCKER 1).
    metrics = {"loss": loss, "grad_norm": optax.tree.norm(grads)}
    return RegressionTrainState(params=params, opt_state=opt_state, step=state.step + 1), metrics


def evaluate_dev(
    state: RegressionTrainState,
    batches: Sequence[Any],
    *,
    predict_fn: Callable[[Any, Any], jax.Array],
    variance_table: Any,
) -> float:
    """The DEV number the rule is decided on: normalized MSE, averaged over the DEV batches."""
    batches = list(batches)
    if not batches:
        raise ValueError("a DEV evaluation needs at least one batch")
    table = jnp.asarray(variance_table, jnp.float32)
    # Sums and counts, divided once: a short final batch must not weigh as much as a full one, or the
    # number the checkpoint is selected on depends on how the DEV set happened to be chunked
    # (S7 review, MAJOR 6).
    total, examples = 0.0, 0
    for batch in batches:
        per_example = per_example_regression_loss(predict_fn(state.params, batch), batch.target_context)
        total += float(jnp.sum(per_example / table[jnp.asarray(batch.step_indices)]))
        examples += int(per_example.shape[0])
    return total / examples


def stop_verdict(history: Sequence[EvalRecord], *, factor: float = STOP_FACTOR, patience: int = STOP_PATIENCE):
    """F2's stop rule as a decision over the eval history -- pure, so its edges are testable.

    Stop when DEV normalized MSE exceeds ``factor`` x its **running best over the earlier evals** on
    ``patience`` consecutive evals *while train MSE is still falling*. The conjunction matters: a train
    loss that plateaus is a different failure and must not spend the patience budget, and the first
    eval can never trigger (no prior best, no previous train MSE). The retained checkpoint is the best
    DEV normalized MSE seen up to the decision -- never one of the degraded evals.
    """
    best_step = best_value = None
    streak = 0
    for index, record in enumerate(history):
        degraded = best_value is not None and record.dev_normalized_mse > factor * best_value
        falling = index > 0 and record.train_mse < history[index - 1].train_mse
        streak = streak + 1 if (degraded and falling) else 0
        if best_value is None or record.dev_normalized_mse < best_value:
            best_step, best_value = record.step, record.dev_normalized_mse
        if streak >= patience:
            break
    stop = streak >= patience
    reason = (
        f"DEV normalized MSE stayed above {factor}x its running best for {patience} consecutive evals "
        f"while train MSE kept falling; retaining step {best_step}"
        if stop
        else "the budget governs"
    )
    return StopVerdict(stop=stop, reason=reason, streak=streak, best_step=best_step, best_value=best_value)


def best_checkpoint_step(history: Sequence[EvalRecord]) -> int | None:
    """Selection = best DEV normalized MSE; the earliest step wins a tie."""
    return stop_verdict(history).best_step


def _manager(ckpt_dir: str, *, max_to_keep: int, keep_period: int | None = None):
    tf.io.gfile.makedirs(ckpt_dir)
    return ocp.CheckpointManager(
        ckpt_dir,
        item_names=CHECKPOINT_ITEMS,
        item_handlers={
            "params": ocp.StandardCheckpointHandler(),
            "opt_state": ocp.StandardCheckpointHandler(),
            "step": ocp.JsonCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            create=True, max_to_keep=max_to_keep, keep_period=keep_period, enable_async_checkpointing=True
        ),
    )


def build_checkpoint_manager(ckpt_dir: str, *, max_to_keep: int = 3, keep_period: int | None = None):
    """The RESUME tree: adapter state kept by **recency**, exactly like the side-adapter trainer.

    S7 review, BLOCKER 3: an earlier version ranked this tree by DEV with ``best_fn``, so ``max_to_keep``
    evicted the *newest* state and ``latest_step()`` could hand a resumed run an older one -- a silent
    rewind. Selection is a different question and now has its own tree (``build_selection_manager``).
    """
    return _manager(ckpt_dir, max_to_keep=max_to_keep, keep_period=keep_period)


def selection_dir(ckpt_dir: str) -> str:
    """The selection tree is a SIBLING of the resume tree, never a child of it.

    Orbax treats every subdirectory of a manager's root as its own; two managers whose roots nest
    therefore fight over the same tree, and one's cleanup pass deletes the other's unfinalized
    temporary directory mid-write (seen as ``FileNotFoundError`` renaming ``*.orbax-checkpoint-tmp``).
    """
    return f"{str(ckpt_dir).rstrip('/')}{SELECTION_SUFFIX}"


def build_selection_manager(ckpt_dir: str):
    """The SELECTION artifact: one immutable checkpoint, the earliest best DEV normalized MSE.

    Separate from the resume tree so recency eviction can never delete it, and written only on a
    *strict* improvement so a later tie cannot displace the earliest (plan §4-P3' F2 selection rule).
    """
    return _manager(selection_dir(ckpt_dir), max_to_keep=1)


def save_adapter_checkpoint(manager, state: RegressionTrainState, *, dev_metric=None, history=()) -> None:
    """Adapter params, its optimizer state, and the step JSON -- three items, as exp_04 settled it.

    The stop rule's decision state rides **inside the step item** rather than in Orbax metrics: it is
    written atomically with the state it describes, it cannot influence retention, and the restorable
    payload stays adapter-only (S7 review, BLOCKER 4).
    """
    manager.save(
        int(state.step),
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(state.params),
            opt_state=ocp.args.StandardSave(state.opt_state),
            step=ocp.args.JsonSave(
                {
                    "step": int(state.step),
                    DEV_METRIC: None if dev_metric is None else float(dev_metric),
                    "eval_history": [
                        [int(record.step), float(record.dev_normalized_mse), float(record.train_mse)]
                        for record in history
                    ],
                }
            ),
        ),
    )


def read_checkpoint_json(manager, step: int) -> dict:
    """The step item alone -- the cheap read the selection comparison and the resume both need."""
    return dict(manager.restore(step, args=ocp.args.Composite(step=ocp.args.JsonRestore())))["step"]


def preserve_selection(manager, state: RegressionTrainState, *, dev_metric: float, history=()) -> bool:
    """Write the selection checkpoint iff ``dev_metric`` is a STRICT improvement. Ties keep the earliest."""
    incumbent = manager.latest_step()
    if incumbent is not None:
        previous = read_checkpoint_json(manager, incumbent).get(DEV_METRIC)
        if previous is not None and float(dev_metric) >= float(previous):
            return False
    save_adapter_checkpoint(manager, state, dev_metric=dev_metric, history=history)
    return True


def restore_adapter_checkpoint(manager, state: RegressionTrainState) -> tuple[RegressionTrainState, int]:
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
    """The F2 decision state: without it a resumed run restarts the rule from an empty history and can
    train straight past a degradation streak that began before the interruption (BLOCKER 4)."""
    latest = manager.latest_step()
    if latest is None:
        return []
    rows = read_checkpoint_json(manager, latest).get("eval_history") or []
    return [
        EvalRecord(step=int(step), dev_normalized_mse=float(dev), train_mse=float(train)) for step, dev, train in rows
    ]


def build_pre_context_predict_fn(transformer, adapters) -> tuple[Callable[[Any, Any], jax.Array], Any]:
    """``(predict_fn, adapter_params)`` for the deployed pre_context path, backbone frozen.

    The transformer's split state is captured in the closure and never appears as an argument, so it
    cannot receive a gradient; the head's ``[B, l_pos, text_dim]`` output IS the conditioning (plan §3
    F1, no shim), returned in the model's activation dtype -- the fp32 cast belongs to the loss.
    """
    from flax import nnx

    from maxdiffusion.models.wan.side_adapter_wan import (
        _first_block_self_attention_features,
        _patchify_and_time_embed,
    )

    adapter_graphdef, adapter_params, adapter_rest = nnx.split(adapters, nnx.Param, ...)
    frozen = nnx.split(transformer, nnx.Param, ...)

    def predict(params, batch):
        model = nnx.merge(*frozen)
        adapter = nnx.merge(adapter_graphdef, params, adapter_rest)
        tokens, rotary_emb, _, timestep_proj, _, _ = _patchify_and_time_embed(model, batch.z_bar_t, batch.timestep_2d)
        features = _first_block_self_attention_features(model, tokens, timestep_proj, rotary_emb)
        return adapter.predict_pre_context(features, batch.actions)

    return predict, adapter_params


class WanPosContextRegressionTrainer:
    """The loop: accumulated steps, DEV evals on the cadence, the stop rule, adapter checkpoints.

    Two checkpoint trees, because they answer different questions (S7 review, BLOCKER 3): ``manager``
    keeps the latest state so a preempted job resumes exactly where it stopped, ``selection_manager``
    keeps the one artifact K4 will evaluate. A run started against a populated ``manager`` resumes its
    parameters, its step **and its eval history**, so the stop rule continues mid-stream rather than
    starting over (BLOCKER 4).
    """

    def __init__(self, config, *, predict_fn=None, params=None, tx=None, manager=None, selection_manager=None):
        """Constructible from the config ALONE, because that is all ``train_wan.train`` has (S8).

        The model and the optimizer are seams: a test (or S9's wiring) passes them, and a dispatched
        launch does not have them yet. The checkpoint trees, in contrast, are pure config -- they come
        from ``checkpoint_dir`` when it is set, and a run that never checkpoints simply has none.
        """
        self.config = config
        self.schedule = TrainingSchedule.from_config(config)
        self.predict_fn = predict_fn
        self.tx = tx
        checkpoint_dir = str(optional_config_value(config, "checkpoint_dir", "") or "")
        self.manager = (
            manager if manager is not None else (build_checkpoint_manager(checkpoint_dir) if checkpoint_dir else None)
        )
        if selection_manager is not None:
            self.selection_manager = selection_manager
        else:
            self.selection_manager = build_selection_manager(checkpoint_dir) if checkpoint_dir else None
        self.state = None if params is None else RegressionTrainState(params=params, opt_state=tx.init(params), step=0)

    def start_training(self) -> None:
        """The dispatch entry point (``train_wan.train`` calls this).

        The loop itself is finished -- ``run`` is what K3 executes -- but its two EXTERNAL seams are
        not: the pre_context model (transformer + adapter stack, i.e. ``build_pre_context_predict_fn``
        against real weights) and the K2 cache iterators that feed ``gather_training_tuple``. Both land
        in S9. Saying so precisely is the honest behavior; training on an empty seam is not.
        """
        raise NotImplementedError(
            "POS_CONTEXT_TI2V dispatches to this trainer and its config resolves, but the pre_context "
            "model and the K2 cache iterators are wired in S9. Until then a run is driven explicitly: "
            "construct with predict_fn/params/tx and call run(make_iterator, dev_evaluator)."
        )

    def run(
        self,
        make_iterator: Callable[[int], Iterator[Any]],
        dev_evaluator: Callable[[RegressionTrainState], float],
        *,
        start_step: int | None = None,
    ) -> RunReport:
        """Train to the budget, or until the stop rule says otherwise, resuming if a checkpoint exists."""
        state, history = self.state, []
        if self.manager is not None and start_step is None:
            state, restored_step = restore_adapter_checkpoint(self.manager, self.state)
            history = restore_eval_history(self.manager)
            start_step = restored_step
        start_step = 0 if start_step is None else int(start_step)
        state = dataclasses.replace(state, step=start_step)
        verdict = stop_verdict(history)
        if verdict.stop:
            # The restored history has already decided. Training on would spend another eval_every
            # steps -- 1,000 at production cadence -- and advance the resume state PAST the decision,
            # so the next retry would resume from a checkpoint the rule never sanctioned. Return
            # before the input pipeline is even built.
            print(
                f"[pos-regression] reopened a TERMINAL checkpoint at step {start_step}: {verdict.reason}. "
                f"No training step will be taken; the selected checkpoint is step {verdict.best_step}."
            )
            return RunReport(state, tuple(history), verdict, verdict.best_step, 0)
        iterator = make_iterator(resume_seed(self.schedule.seed, start_step))
        window: list[float] = []
        step = start_step
        for step in range(start_step + 1, self.schedule.max_train_steps + 1):
            state, metrics = train_step(
                state,
                checked_training_batch(next(iterator), self.schedule),
                predict_fn=self.predict_fn,
                tx=self.tx,
                accumulation_steps=self.schedule.accumulation_steps,
            )
            window.append(float(metrics["loss"]))  # the host boundary: metrics are JAX scalars until here
            if not should_evaluate(step, self.schedule):
                continue
            # Train MSE at an eval is the mean since the previous one: the rule compares like with like.
            history.append(EvalRecord(step, float(dev_evaluator(state)), sum(window) / len(window)))
            window = []
            verdict = stop_verdict(history)
            self._persist(state, history, verdict)
            if verdict.stop:
                break
        return RunReport(state, tuple(history), verdict, verdict.best_step, step - start_step)

    def _persist(self, state: RegressionTrainState, history, verdict: StopVerdict) -> None:
        """Latest state for resume; the selection artifact only when this eval is the strict new best."""
        latest = history[-1]
        if self.manager is not None:
            save_adapter_checkpoint(self.manager, state, dev_metric=latest.dev_normalized_mse, history=history)
        if self.selection_manager is not None and verdict.best_step == latest.step:
            preserve_selection(self.selection_manager, state, dev_metric=latest.dev_normalized_mse, history=history)
