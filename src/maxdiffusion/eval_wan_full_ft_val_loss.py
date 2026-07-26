"""Pure helpers for the full-finetune (FULL_FT_TI2V) validation-loss evaluator.

exp_01 Part II (Query 8), T1: per-checkpoint one-step validation loss -- the exact
training objective (velocity MSE, frame-0 masked) over ALL held-out windows, with
per-example ``(t, eps)`` held FIXED across checkpoints so the loss curve is a pure
model effect. This module holds only the PURE, CPU-testable pieces of that
evaluator:

* :func:`per_example_rng` -- deterministic per-position ``(t_idx, eps)`` draw (D1/F1),
* :func:`plan_batches`   -- exactly-once position coverage with a masked padded tail (D2/F2),
* :func:`aggregate`      -- validity-masked mean / sample-stderr / count (D2/F2).

The config-driven evaluator main (state build, per-checkpoint restore loop, output
writers) lands in cycle B; this cycle contributes pure functions only.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


def per_example_rng(
    seed: int,
    position: int,
    num_steps: int,
    example_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Deterministic ``(t_idx, eps)`` for one validation example, keyed by POSITION.

    Counter-based (stateless) draw: ``k = fold_in(key(seed), position)``, split into
    ``(k_t, k_eps)``; ``t_idx = randint(k_t, (), 0, num_steps, int32)`` and
    ``eps = normal(k_eps, example_shape, float32)`` generated at the UNBATCHED
    example shape (the caller stacks per-example ``eps`` into a batch).

    DISTRIBUTION PARITY, not draw replay. Training samples a uniform step index
    ``randint([0, num_steps))`` and independent fresh Gaussian noise per example
    (see ``_sample_step_indices`` / ``_build_noise`` in
    ``wan_ti2v_side_adapter_trainer``). Held-out examples have no training draws to
    replay, so this reproduces the same MARGINAL law -- independent uniform ``t`` and
    standard-normal ``eps`` per example -- deliberately NOT training's stateful
    per-step key sequence. Being a pure function of ``(seed, position)``, the draw is
    identical across checkpoints, batches, batch order, and hosts, which is what
    makes the per-checkpoint loss comparison a pure model effect.

    ``position`` is the 0-based DATASET POSITION (the reader coordinate); the stored
    per-record ``ordinal`` field is NEVER used as this index. Raises ``ValueError``
    if ``num_steps <= 0``. Returns ``t_idx`` as an int32 scalar and ``eps`` as a
    float32 array of shape ``example_shape``.
    """
    if num_steps <= 0:
        raise ValueError(f"per_example_rng: num_steps must be positive, got {num_steps}")
    k = jax.random.fold_in(jax.random.key(seed), position)
    k_t, k_eps = jax.random.split(k)
    t_idx = jax.random.randint(k_t, (), 0, num_steps, dtype=jnp.int32)
    eps = jax.random.normal(k_eps, example_shape, jnp.float32)
    return t_idx, eps


def plan_batches(total: int, batch: int) -> list[tuple[list[int], list[bool]]]:
    """Plan fixed, position-ordered evaluation batches covering ``0..total-1`` once.

    Returns one ``(positions, validity)`` pair per batch in position order, with no
    shuffling and no repetition of the dataset. Every batch has exactly ``batch``
    entries; the final partial batch is padded by REPEATING THE LAST REAL POSITION
    with ``validity == False`` for the pads. Repeating a real record is safe because
    the transformer forward and the per-example loss are batch-independent -- the pad
    only fills the fixed-shape batch and is dropped in aggregation.

    Invariants (checked by the tests): the union of positions whose validity is True
    is exactly ``range(total)``, each real position appears once and only in a valid
    slot, and padded slots alias the last real position (never a fresh index).

    Raises ``ValueError`` if ``total`` or ``batch`` is non-positive.
    """
    if total <= 0:
        raise ValueError(f"plan_batches: total must be positive, got {total}")
    if batch <= 0:
        raise ValueError(f"plan_batches: batch must be positive, got {batch}")
    plan: list[tuple[list[int], list[bool]]] = []
    for start in range(0, total, batch):
        positions = list(range(start, min(start + batch, total)))
        validity = [True] * len(positions)
        while len(positions) < batch:
            positions.append(positions[-1])  # repeat the last REAL position
            validity.append(False)
        plan.append((positions, validity))
    return plan


def _concat_1d(x, dtype) -> np.ndarray:
    """Normalize the aggregate() input into a flat 1-D numpy array of ``dtype``.

    Accepts the canonical contract -- a list/tuple of per-batch 1-D array-likes,
    concatenated in the given order -- and, for convenience, a single already-1-D
    array-like (treated as one batch). A flat Python list of scalars also flattens
    correctly (each scalar becomes a length-1 segment).
    """
    if isinstance(x, (list, tuple)):
        parts = [np.asarray(p, dtype=dtype).reshape(-1) for p in x]
        if not parts:
            return np.zeros((0,), dtype=dtype)
        return np.concatenate(parts)
    return np.asarray(x, dtype=dtype).reshape(-1)


def aggregate(
    per_example_losses,
    validity,
    expected_count: int,
) -> dict:
    """Validity-masked mean / sample-stderr / count over per-example losses.

    Input contract (chosen, per plan D2): ``per_example_losses`` is a list of
    per-batch 1-D loss arrays and ``validity`` is the matching list of per-batch 1-D
    boolean masks -- exactly what the evaluator collects from its per-batch loop
    (each jitted eval step returns a ``[batch]`` loss vector; each batch's validity
    comes from :func:`plan_batches`). A single already-concatenated 1-D array is also
    accepted for either argument. Both are flattened in the given order and must have
    equal total length.

    ONLY mask-valid entries are counted: padded duplicates (``validity == False``)
    are excluded from the count, the mean, AND the stderr. ``stderr`` is the sample
    standard deviation (``ddof=1``) divided by ``sqrt(n)``. Raises ``ValueError`` if
    the flattened losses and validity differ in length, or if the number of valid
    entries ``n`` does not equal ``expected_count`` (catches both a short read and a
    duplicated/over-long source). Reductions run in float64 for host-side stability.

    Returns ``{"mean_loss": float, "stderr": float, "n": int}``.
    """
    losses = _concat_1d(per_example_losses, np.float64)
    valid = _concat_1d(validity, bool)
    if losses.shape != valid.shape:
        raise ValueError(f"aggregate: losses length {losses.shape[0]} != validity length {valid.shape[0]}")
    valid_losses = losses[valid]
    n = int(valid_losses.shape[0])
    if n != expected_count:
        raise ValueError(f"aggregate: counted {n} valid entries != expected_count {expected_count}")
    mean_loss = float(np.mean(valid_losses))
    if n >= 2:
        stderr = float(np.std(valid_losses, ddof=1) / np.sqrt(n))
    else:
        stderr = float("nan")
    return {"mean_loss": mean_loss, "stderr": stderr, "n": n}


# Re-exported for the cycle-B evaluator; keeps ``Sequence`` referenced for typing.
__all__: Sequence[str] = ("per_example_rng", "plan_batches", "aggregate")
