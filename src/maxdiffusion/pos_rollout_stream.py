"""exp_06 `rollout_adapter` — T3b-1: the per-step training stream (plan §3b).

**One contract: every random quantity an optimizer step consumes is a pure function of
``(seed, LOOP global step)``** — identical for both arms, unchanged by microbatching, and
reproducible after a restart. Nothing here computes a loss, touches a checkpoint, or reads a
manifest; those are the later T3b sub-rounds (see the round's report for the proposed split).

Why this is its own contract. The comparison exp_06 exists to make is R-B against matched-C0 at
*identical data and identical randomness* (plan §3b: "C0 and R-B consume THE SAME batch stream so
the comparison is paired at the data level"). If an arm could perturb a draw — by consuming
randomness the other does not, by microbatching differently, or by resuming onto a different key —
the two arms would differ by more than their objective and the causal claim would be gone. That is a
property of the STREAM, provable with no model and no loss, so it is proved here where the oracles
can be exact.

**The three properties, and how each is enforced rather than hoped for.**

* **Arm-independent.** This module has no notion of an arm: no function takes one, and nothing
  branches on one. A draw cannot depend on something the code cannot see. A test pins the absence.
* **Accumulation-invariant.** Draws are keyed on the optimizer step, never on a microbatch index, so
  splitting a logical batch into microbatches cannot move a draw — and :func:`accumulation_plan`
  refuses any split that would silently change the logical batch (exp_05's S7 review, BLOCKER 2:
  configuration being *expressible* is not the same as the iterator *yielding* it).
* **Resume-stable.** The key is derived from the loop's global step. **The caller must pass the
  LOOP's step, never a restored ``state.step``** — the T1 reviewer's standing obligation for this
  round. Orbax restores params/opt_state/step, and a freshly built state's ``step`` is not the
  loop's position; keying on it silently re-runs one segment's randomness on another's data. This
  module cannot make that mistake for the caller (it has no access to a state object, and a test
  pins that), but it also cannot prevent the caller making it, so the hazard is demonstrated
  explicitly in the tests rather than only described here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from maxdiffusion.pos_rollout_support import exp03_aux_key, rollout_support

# The raw per-shape draw helpers are PRIVATE (T3b-1 review, MAJOR). Exporting them beside the safe
# seam proved only that a safe path EXISTS; a caller could still draw at a microbatch width and get a
# different epsilon at every accumulation factor -- the failure this module's own test exhibits. The
# public surface is `draw_step_for_batch`, which validates the batch it is handed, derives the shape
# itself, draws ONCE and splits.
__all__ = [
    "POS_ROLLOUT_EPSILON_PURPOSE",
    "POS_ROLLOUT_TIMESTEP_PURPOSE",
    "StepDraws",
    "accumulation_plan",
    "checked_logical_batch",
    "draw_step_for_batch",
    "microbatch_slices",
    "split_batch",
]

# Declared in `pos_rollout_support.EXP03_AUX_PURPOSES` at T3b-1. Named `rollout_epsilon` rather than
# `epsilon` deliberately: T1's suite uses `"epsilon"` as its example of an UNDECLARED purpose, so
# taking that name would have silently defanged an inherited tripwire.
POS_ROLLOUT_EPSILON_PURPOSE = "rollout_epsilon"

# matched-C0 samples a PER-EXAMPLE sigma index. The settled trainer draws it from a split rng; left
# outside the stream it would be accumulation- and resume-dependent, which would confound exactly the
# R-B-vs-C0 comparison Yixun's decision 1 exists to make causal. So it is a stream draw like any
# other (T3b-1 strengthen, review BLOCKER 1).
POS_ROLLOUT_TIMESTEP_PURPOSE = "one_step_index"


@dataclasses.dataclass(frozen=True)
class StepDraws:
    """Everything one optimizer step draws: the support window, the noise, and C0's sigma index.

    Returned as one value so a caller cannot take the support from one step and the epsilon from
    another — the pairing is the object, not a convention.
    """

    support_start: jax.Array
    support_end: jax.Array
    epsilon: jax.Array
    #: matched-C0's per-example sigma index. R-B does not read it; that both arms are HANDED the
    #: same draws is what keeps them paired, not that both consume every field.
    t_idx: jax.Array


def _rollout_epsilon(*, seed: int, global_step, shape, dtype=jnp.float32, salt: int = 0) -> jax.Array:
    """The step's fresh noise, keyed on ``(seed, global_step)`` through T1's derivation.

    ``global_step`` is the LOOP's step (see the module docstring). Because the key is derived rather
    than split, drawing it does not advance any stream, so an arm that draws epsilon leaves every
    other draw exactly where an arm that does not would have left it.
    """
    key = exp03_aux_key(seed=seed, global_step=global_step, purpose=POS_ROLLOUT_EPSILON_PURPOSE, salt=salt)
    return jax.random.normal(key, tuple(shape), dtype=dtype)


def _one_step_timestep_indices(*, seed: int, global_step, batch_size: int, num_steps: int, salt: int = 0) -> jax.Array:
    """matched-C0's per-example sigma index, keyed on ``(seed, global_step)`` like every other draw.

    Uniform over the same ``{0 .. num_steps-1}`` the settled trainer samples (its default
    ``side_adapter_t_sampling`` is ``uniform``); drawn at the LOGICAL batch width so that splitting
    for accumulation is a view of one draw rather than several.
    """
    key = exp03_aux_key(seed=seed, global_step=global_step, purpose=POS_ROLLOUT_TIMESTEP_PURPOSE, salt=salt)
    return jax.random.randint(key, (int(batch_size),), 0, int(num_steps))


def _draw_step_stream(
    *,
    seed: int,
    global_step,
    num_steps: int,
    k_b: int,
    shape,
    dtype=jnp.float32,
    support_salt: int = 0,
    epsilon_salt: int = 0,
    timestep_salt: int = 0,
) -> StepDraws:
    """THE per-step draw: one support window and one epsilon, both per batch (plan §3b).

    One call per optimizer step, whatever the arm and whatever the accumulation factor. The support
    is T1's characterized primitive (legal range, terminal sigma excluded, per-batch granularity);
    the epsilon is fresh per step and independent of it by purpose.
    """
    start, end = rollout_support(
        seed=seed,
        global_step=global_step,
        num_steps=num_steps,
        k_b=k_b,
        support_salt=support_salt,
    )
    epsilon = _rollout_epsilon(seed=seed, global_step=global_step, shape=shape, dtype=dtype, salt=epsilon_salt)
    t_idx = _one_step_timestep_indices(
        seed=seed, global_step=global_step, batch_size=int(tuple(shape)[0]), num_steps=num_steps, salt=timestep_salt
    )
    return StepDraws(support_start=start, support_end=end, epsilon=epsilon, t_idx=t_idx)


def _split_draws(draws: StepDraws, accumulation_steps: int) -> tuple[StepDraws, ...]:
    """Microbatch VIEWS of one logical-batch draw — never a fresh draw per microbatch.

    The per-example fields are sliced; the per-batch support is shared unchanged, which is what
    ``one support draw per batch`` (plan §3b) means once accumulation exists.
    """
    windows = microbatch_slices(int(draws.epsilon.shape[0]), accumulation_steps)
    return tuple(
        StepDraws(
            support_start=draws.support_start,
            support_end=draws.support_end,
            epsilon=draws.epsilon[window],
            t_idx=draws.t_idx[window],
        )
        for window in windows
    )


def draw_step_for_batch(
    batch: Mapping[str, Any],
    *,
    seed: int,
    global_step,
    logical_batch: int,
    microbatch: Any,
    num_steps: int,
    k_b: int,
    dtype=jnp.float32,
) -> tuple[StepDraws, tuple[StepDraws, ...], tuple[dict[str, Any], ...]]:
    """THE public seam: the only sanctioned way for a loop to obtain a step's randomness.

    Takes the ACTUAL batch, refuses it if it is not the declared logical batch, derives the example
    shape from it, draws once at that width and splits both the draws and the batch. A caller cannot
    reach a per-microbatch draw from here, which is the point (T3b-1 review, MAJOR): proving a safe
    path exists is not the same as making the unsafe one unreachable.
    """
    microbatch_size, accumulation_steps = accumulation_plan(logical_batch, microbatch)
    checked_logical_batch(
        batch, logical_batch=logical_batch, accumulation_steps=accumulation_steps, microbatch=microbatch_size
    )
    example_shape = tuple(jnp.shape(batch["z_video"])[1:])
    draws = _draw_step_stream(
        seed=seed,
        global_step=global_step,
        num_steps=num_steps,
        k_b=k_b,
        shape=(int(logical_batch), *example_shape),
        dtype=dtype,
    )
    return draws, _split_draws(draws, accumulation_steps), split_batch(batch, accumulation_steps)


def _draw_logical_step(
    *,
    seed: int,
    global_step,
    logical_batch: int,
    microbatch: Any,
    num_steps: int,
    k_b: int,
    example_shape,
    dtype=jnp.float32,
) -> tuple[StepDraws, tuple[StepDraws, ...]]:
    """THE orchestration seam: draw ONCE at the checked logical-batch width, then split.

    This is the only sanctioned way for a loop to obtain a step's randomness. Drawing per microbatch
    instead would produce a different epsilon at every accumulation factor — the reviewer measured
    exactly that — so accumulation would silently change the objective. Returning both the logical
    draw and its views lets a caller assert they reconstruct it.
    """
    _, accumulation_steps = accumulation_plan(logical_batch, microbatch)
    draws = _draw_step_stream(
        seed=seed,
        global_step=global_step,
        num_steps=num_steps,
        k_b=k_b,
        shape=(int(logical_batch), *tuple(example_shape)),
        dtype=dtype,
    )
    return draws, _split_draws(draws, accumulation_steps)


def accumulation_plan(logical_batch: int, microbatch: Any) -> tuple[int, int]:
    """``(microbatch, accumulation_steps)`` — a split that NEVER changes the logical batch.

    Generalized from exp_05's S7 (which proved the rule on its own batch type). Every rejection here
    is a case where accumulating would have quietly trained on a different batch size than the one
    the run declares, which would break the matched-C0 comparison as surely as a different objective
    would.
    """
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


def microbatch_slices(logical_batch: int, accumulation_steps: int) -> tuple[slice, ...]:
    """The contiguous, order-preserving windows a logical batch is accumulated over."""
    logical, count = int(logical_batch), int(accumulation_steps)
    if count < 1 or logical < 1 or logical % count:
        raise ValueError(f"{count} microbatches do not divide a logical batch of {logical}")
    width = logical // count
    return tuple(slice(index * width, (index + 1) * width) for index in range(count))


def split_batch(batch: Mapping[str, Any], accumulation_steps: int) -> tuple[dict[str, Any], ...]:
    """Split every field of an example-major batch into ``accumulation_steps`` ordered parts.

    Field-agnostic on purpose: a batch that grows a field later is split correctly rather than
    silently carrying the whole field into every microbatch.
    """
    if not batch:
        raise ValueError("cannot split an empty batch")
    sizes = {int(jnp.shape(value)[0]) for value in batch.values()}
    if len(sizes) != 1:
        raise ValueError(f"batch fields disagree on the example count: {sorted(sizes)}")
    windows = microbatch_slices(sizes.pop(), accumulation_steps)
    return tuple({name: value[window] for name, value in batch.items()} for window in windows)


def checked_logical_batch(
    batch: Mapping[str, Any], *, logical_batch: int, accumulation_steps: int, microbatch: int
) -> Mapping[str, Any]:
    """Refuse a batch that is not the configured logical batch (S7 review, BLOCKER 2, generalized).

    ``accumulation_plan`` proves the CONFIGURATION is expressible; nothing until here compares it
    with what the iterator actually yields. A 128-example batch under ``logical=256, microbatch=64``
    would otherwise be split into four 32s — a halved logical batch, silently.
    """
    if not batch:
        raise ValueError("cannot validate an empty batch")
    sizes = {int(jnp.shape(value)[0]) for value in batch.values()}
    if len(sizes) != 1:
        raise ValueError(f"batch fields disagree on the example count: {sorted(sizes)}")
    size = sizes.pop()
    if size != int(logical_batch):
        raise ValueError(
            f"the iterator yielded {size} examples but the logical batch is {int(logical_batch)}: "
            f"accumulation preserves the logical batch, it never adapts to the data"
        )
    if size // int(accumulation_steps) != int(microbatch):
        raise ValueError(
            f"{int(accumulation_steps)} microbatches of {size} examples are not the configured "
            f"microbatch width {int(microbatch)}"
        )
    return batch
