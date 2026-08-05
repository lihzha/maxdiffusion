"""exp_05 S6 — the regression objective's data side: gather, loss, normalized metric.

Plan §4-P3' fixes the objective: *sample t per example; cached z̄_t + per-token timestep + actions ->
block-0 features (stop-grad) -> head -> MSE(Ĉ_t, C*_t) fp32*, judged by **normalized MSE = MSE ÷ the
per-step target variance computed once on the TRAIN cache**. This module owns exactly the parts that
are not the model: the training tuple assembled out of K2 records, the loss, and that metric.

Three decisions worth naming:

- **Nothing here samples anything by itself.** ``gather_training_tuple`` takes explicit indices and
  ``sample_step_indices`` takes an explicit key, so a batch is a pure function of what it was handed.
  RNG policy -- how t is drawn, reseeded, resumed -- belongs to the trainer (S7), which is also the
  only place that can get it wrong in a way that survives a restart.
- **The timestep is exp_04's, by import.** ``canonical_sigmas`` -> ``rollout_timesteps_from_sigmas``
  -> ``_build_per_token_timestep``: the cached states were produced against that grid, so rebuilding
  it here from anything else would train the head on a timestep the target never saw. The grid is an
  argument only so the toy geometries in the tests can drive the same code.
- **fp32 out.** The cache is fp16 and the model runs bf16. Gather emits fp32 and the loss computes in
  fp32; the single bf16 cast at the model boundary stays where §3's cast rule put it -- one component,
  in the trainer, not scattered through the data path.

The loss is jittable: shape checks are static and survive tracing, value checks (finiteness) are
skipped on tracers, where the values do not exist yet. The gather runs on concrete host arrays and
checks everything.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.wan.null_inversion_wan import NUM_TRAIN_TIMESTEPS, N_HIST_FRAMES
from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L
from maxdiffusion.models.wan.side_adapter_wan import _build_per_token_timestep, rollout_timesteps_from_sigmas
from maxdiffusion.null_adapter_verify import canonical_sigmas
from maxdiffusion.pos_context_records import PRODUCTION_POS_GEOMETRY

# The cached sampler-step count: one context and one state per step (plan §4-P2' schema).
POS_STEPS = PRODUCTION_POS_GEOMETRY.pos_embeds[0]


@dataclasses.dataclass(frozen=True)
class RegressionBatch:
    """One teacher-forced regression batch: what the head is shown, and what it must reproduce."""

    names: tuple[str, ...]
    step_indices: np.ndarray  # [B] int32, host-side: the metric normalizes per example by its own t
    z_bar_t: jax.Array  # [B, C, F, H, W] fp32 -- the cached state at t, first frame already pinned
    timestep_2d: jax.Array  # [B, seq_len] fp32 -- per-token timestep of sigma_t, history zeroed
    actions: jax.Array  # [B, A, 7] fp32
    target_context: jax.Array  # [B, l_pos, D] fp32 -- C*_t, the optimized conditional context


def _concrete(array: Any) -> np.ndarray | None:
    """The values, or ``None`` under ``jit`` -- a tracer carries a shape but no numbers yet."""
    try:
        return np.asarray(array)
    except jax.errors.TracerArrayConversionError:
        return None


def _checked_finite(array: Any, label: str) -> None:
    values = _concrete(array)
    if values is not None and not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{label} must be finite: a nan here silently poisons every metric downstream")


def _checked_indices(step_indices: Any, expected: int) -> np.ndarray:
    indices = np.asarray(step_indices)
    if indices.dtype == np.bool_ or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"step indices must be integers, got dtype {indices.dtype}")
    if indices.ndim != 1 or int(indices.shape[0]) != int(expected):
        raise ValueError(f"there must be exactly one step index per record, got {indices.shape} for {expected}")
    return indices.astype(np.int32)


def _checked_cached_example(record: Any, steps: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """One record's ``(pos_embeds, z_bar_states)`` as fp32, with the schema this round depends on."""
    embeds = np.asarray(record.pos_embeds, dtype=np.float32)
    states = np.asarray(record.z_bar_states, dtype=np.float32)
    if embeds.ndim != 3:
        raise ValueError(f"pos_embeds must be [N, l_pos, D], got shape {embeds.shape}")
    if int(embeds.shape[1]) != POS_L:
        raise ValueError(
            f"pos_embeds must carry l_pos={POS_L} rows -- the deployed head emits exactly that many "
            f"context tokens -- got {embeds.shape[1]}"
        )
    if steps is not None and int(embeds.shape[0]) != int(steps):
        raise ValueError(
            f"every record in a batch must agree on the sampler-step count: got {embeds.shape[0]} " f"beside {steps}"
        )
    if states.ndim != 5 or int(states.shape[0]) != int(embeds.shape[0]):
        raise ValueError(
            f"z_bar_states must carry one state per sampler step, got {states.shape} beside "
            f"{embeds.shape[0]} cached contexts"
        )
    _checked_finite(embeds, "pos_embeds")
    _checked_finite(states, "z_bar_states")
    return embeds, states, int(embeds.shape[0])


def sample_step_indices(key: jax.Array, batch_size: int, *, steps: int = POS_STEPS) -> jax.Array:
    """Draw one uniform ``t in [0, steps)`` per example. Deterministic in ``key``; the trainer owns it."""
    if int(batch_size) < 1:
        raise ValueError(f"a training batch needs at least one example, got {batch_size}")
    if int(steps) < 1:
        raise ValueError(f"the sampler grid needs at least one step, got {steps}")
    return jax.random.randint(key, (int(batch_size),), 0, int(steps), dtype=jnp.int32)


def gather_training_tuple(records: Sequence[Any], step_indices: Any, *, sigmas: Any = None) -> RegressionBatch:
    """Assemble ``(z̄_t, per-token timestep, actions, C*_t)`` for one batch of cached K2 records.

    ``records`` need only expose ``name``, ``actions``, ``pos_embeds`` and ``z_bar_states`` -- the four
    fields the objective reads -- so the trainer may stream lighter views than a full
    ``PosContextRecord``. Everything else about the schema is still checked here, because a state that
    is not the one belonging to the gathered context is a bug no downstream test can name.
    """
    records = list(records)
    if not records:
        raise ValueError("a training batch needs at least one record")
    indices = _checked_indices(step_indices, len(records))

    steps: int | None = None
    contexts, states, actions, names = [], [], [], []
    for record in records:
        embeds, cached_states, steps = _checked_cached_example(record, steps)
        contexts.append(embeds)
        states.append(cached_states)
        actions.append(np.asarray(record.actions, dtype=np.float32))
        names.append(str(record.name))

    if int(indices.min()) < 0 or int(indices.max()) >= steps:
        raise ValueError(f"step indices {indices.tolist()} fall outside the cached grid [0, {steps})")

    grid = canonical_sigmas() if sigmas is None else np.asarray(sigmas, dtype=np.float32)
    if grid.ndim != 1 or int(grid.shape[0]) != steps + 1:
        raise ValueError(
            f"the sigma grid must cover the cached sampler-step count: {steps} steps need {steps + 1} "
            f"sigmas, got shape {grid.shape}"
        )

    gathered_states = np.stack([state[index] for state, index in zip(states, indices)])
    gathered_contexts = np.stack([context[index] for context, index in zip(contexts, indices)])
    timesteps = rollout_timesteps_from_sigmas(jnp.asarray(grid), NUM_TRAIN_TIMESTEPS)
    _, f_lat, h_lat, w_lat = gathered_states.shape[1:]
    return RegressionBatch(
        names=tuple(names),
        step_indices=indices,
        z_bar_t=jnp.asarray(gathered_states, dtype=jnp.float32),
        timestep_2d=_build_per_token_timestep(
            timesteps[jnp.asarray(indices)], f_lat, h_lat, w_lat, n_hist=N_HIST_FRAMES
        ),
        actions=jnp.asarray(np.stack(actions), dtype=jnp.float32),
        target_context=jnp.asarray(gathered_contexts, dtype=jnp.float32),
    )


def per_example_regression_loss(predicted: Any, target: Any) -> jax.Array:
    """``[B]`` fp32 MSE over each example's ``[l_pos, D]`` context."""
    predicted, target = jnp.asarray(predicted), jnp.asarray(target)
    if predicted.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"the regression loss is defined on contexts shaped [B, l_pos, D], got {predicted.shape} "
            f"and {target.shape}"
        )
    if predicted.shape != target.shape:
        raise ValueError(f"prediction and target must have the same shape, got {predicted.shape} != {target.shape}")
    _checked_finite(predicted, "the predicted context")
    _checked_finite(target, "the target context")
    # fp32 before the subtraction: the residuals this objective is made of are small next to bf16's
    # spacing, and the cache itself is fp16.
    residual = predicted.astype(jnp.float32) - target.astype(jnp.float32)
    return jnp.mean(residual**2, axis=(1, 2))


def regression_loss(predicted: Any, target: Any) -> jax.Array:
    """The batch objective: the mean of the per-example MSEs (= the element mean, equal-sized rows)."""
    return jnp.mean(per_example_regression_loss(predicted, target))


def target_variance_table(records: Iterable[Any]) -> np.ndarray:
    """``[N]`` fp32: the population variance of ``C*_i`` over every element of the cache, per step i.

    Computed ONCE on the TRAIN cache (plan §4-P3'), so this is a single streaming pass -- the cache is
    ~15 GiB and never resident. Chan's parallel update in float64 rather than ``sum(x^2) - mean^2``:
    the naive form cancels catastrophically exactly where this metric matters, on steps whose targets
    barely move. A step with no variance is refused, not divided by.
    """
    steps: int | None = None
    count = 0
    means = m2 = np.zeros(0, dtype=np.float64)
    for record in records:
        embeds, _, steps = _checked_cached_example(record, steps)
        block = embeds.astype(np.float64)
        block_count = int(block.shape[1] * block.shape[2])
        block_mean = block.mean(axis=(1, 2))
        block_m2 = ((block - block_mean[:, None, None]) ** 2).sum(axis=(1, 2))
        if count == 0:
            means, m2 = block_mean, block_m2
        else:
            delta = block_mean - means
            total = count + block_count
            means = means + delta * (block_count / total)
            m2 = m2 + block_m2 + delta**2 * (count * block_count / total)
        count += block_count

    if steps is None:
        raise ValueError("the variance table needs at least one record: an empty cache measures nothing")
    variances = m2 / count
    degenerate = [index for index, value in enumerate(variances) if not np.isfinite(value) or value <= 0.0]
    if degenerate:
        raise ValueError(
            f"per-step target variance must be positive and finite, but steps {degenerate} have none: "
            f"normalized MSE would divide by zero and hand the stop rule an inf to compare against"
        )
    return variances.astype(np.float32)


def normalized_regression_loss(predicted: Any, target: Any, step_indices: Any, variance_table: Any) -> jax.Array:
    """The decidable metric (plan §4-P3' F2): each example's MSE over **its own** step's variance.

    Examples in one batch sit at different t, so normalizing the batch MSE by an average variance
    would report a number that no stop rule could interpret.
    """
    table = np.asarray(variance_table, dtype=np.float32)
    if table.ndim != 1:
        raise ValueError(f"the variance table carries one variance per sampler step, got shape {table.shape}")
    if not bool(np.all(np.isfinite(table))):
        raise ValueError("the variance table must be finite")
    if not bool(np.all(table > 0.0)):
        raise ValueError("every per-step target variance must be positive: the metric divides by it")

    per_example = per_example_regression_loss(predicted, target)
    indices = _checked_indices(step_indices, int(per_example.shape[0]))
    if int(indices.min()) < 0 or int(indices.max()) >= int(table.shape[0]):
        raise ValueError(f"step indices {indices.tolist()} fall outside the variance table [0, {table.shape[0]})")
    return jnp.mean(per_example / jnp.asarray(table)[jnp.asarray(indices)])
