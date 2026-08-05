"""What the exp_04 cache job drops and what dtype it stores at (plan §4-P2; R8 strengthening).

Two pure judgments, split out of ``null_adapter_shards`` so the shard writer is only storage:

**Quarantine.** A pathological example must not kill a multi-hour cache job, and a systemic failure
must not become a data gap. Those two requirements meet at the exception type: only
``ExampleDivergenceError`` is quarantineable. **The runner is contractually required to raise exactly
that type, at the orchestration boundary, for per-example pathology** -- R6's hard trace-finiteness
error (``tracking_losses must be finite``), a per-example NaN, an optimization that diverged -- and to
let everything else through untouched. A configuration error, an OOM, a preempted host or a bug is
not an example's fault, and turning it into a recorded gap would let infrastructure quietly shrink
the cohort (Codex R8 review, finding 5).

Recovery follows the R8 rulings: retry each example alone; quarantine only those that diverge alone;
**re-run the surviving batch** so the composition failure is proven gone rather than assumed gone
(ruling 2); re-raise when nothing diverged alone (a composition/capacity bug) and when *everything*
did (never an empty shard -- ruling 3).

**The fidelity gate.** Plan §4-P2's fp16 decision, over the predeclared **first eight DEV manifest
examples, derived here from the manifest** rather than accepted from the caller: a caller who could
choose the eight could choose the verdict (finding 4). Worst-example maxima, inclusive thresholds.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Mapping, Sequence


FIDELITY_MAX_SSIM_DROP = 0.01  # plan §4-P2: worst-example maxima, not means
FIDELITY_MAX_MSE_INCREASE = 0.05
# The thresholds are inclusive ("<= 0.01", "<= 5%"), and 0.9 - 0.89 is 0.010000000000000009 in binary
# floating point -- the trap R6's plateau boundary hit. Differences within this of a threshold count
# as being at it; it is seven orders below any fidelity difference that could matter.
FIDELITY_BOUNDARY_ATOL = 1e-9
FIDELITY_SUBSET_SIZE = 8  # "the first 8 DEV-manifest examples", plan §4-P2
FIDELITY_METRICS = ("future_ssim", "future_mse")


class ExampleDivergenceError(Exception):
    """One example is pathological; the run may drop it and continue.

    **The runner raises this**, at the batch-orchestration boundary, and only for per-example
    pathology: a non-finite optimizer trace, a NaN latent, a diverged optimization. Nothing else may
    be wrapped in it, and nothing else is quarantineable -- see the module docstring.
    """


@dataclasses.dataclass(frozen=True)
class FidelityVerdict:
    passed: bool
    latent_dtype: str
    subset: tuple[str, ...]
    worst_ssim_drop: float
    worst_mse_increase: float
    per_example: dict[str, dict[str, float]]
    reasons: tuple[str, ...]


def _checked_names(names: Sequence[str], label: str) -> tuple[str, ...]:
    names = tuple(names)
    if not names:
        raise ValueError(f"{label} must carry at least one name")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique, got {list(names)}")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{label} names must be non-empty strings, got {list(names)}")
    return names


def _checked_results(results: Any, names: Sequence[str]) -> dict[str, Any]:
    if not isinstance(results, Mapping) or set(results) != set(names):
        raise ValueError(
            f"the batch runner did not return a result for exactly {list(names)}: "
            f"got {sorted(results) if isinstance(results, Mapping) else type(results)}"
        )
    return dict(results)


def quarantine_batch_failures(
    run_fn: Callable[[tuple[str, ...]], Mapping[str, Any]], names: Sequence[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run a batch; if it fails, drop only the examples that diverge on their own.

    ``run_fn(names) -> {name: result}`` must return a result for exactly the names it was given: a
    runner that silently dropped one would otherwise look like a success with a hole in it.

    The happy path is a single batched call. On failure each example is retried alone; those that
    raise ``ExampleDivergenceError`` are quarantined with ``"<ExceptionType>: <message>"`` as the
    reason, and any other exception propagates immediately. Then the survivors are **re-run as a
    batch** and those results are what comes back -- if that re-run fails, the failure was never the
    quarantined examples' and it propagates too.

    Raises:
      The original batch error when no example diverges alone (a composition bug: per-example
      independence is a plan §3 contract) and when every example diverges (an all-quarantined batch
      is not a boundary, so it must never reach a marker).
    """
    names = _checked_names(names, "batch")
    try:
        batch = run_fn(names)
    except Exception as batch_error:  # noqa: BLE001 -- triage below decides what is survivable
        quarantined: dict[str, str] = {}
        survivors: list[str] = []
        for name in names:
            try:
                run_fn((name,))
            except ExampleDivergenceError as error:
                quarantined[name] = f"{type(error).__name__}: {error}"
            else:
                survivors.append(name)
        if not quarantined or not survivors:
            raise batch_error
        # Ruling 2: prove the composition failure is gone instead of assuming the singletons did it.
        return _checked_results(run_fn(tuple(survivors)), survivors), quarantined
    return _checked_results(batch, names), {}


def _fidelity_metrics(table: Mapping[str, Any], name: str, label: str) -> dict[str, float]:
    entry = table.get(name)
    if not isinstance(entry, Mapping) or any(metric not in entry for metric in FIDELITY_METRICS):
        raise ValueError(f"{label}[{name!r}] must carry {list(FIDELITY_METRICS)}, got {sorted(entry or {})}")
    values = {metric: float(entry[metric]) for metric in FIDELITY_METRICS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f"{label}[{name!r}] metrics must be finite, got {values}")
    return values


def fidelity_gate(
    dev_manifest: Sequence[str],
    fp32_metrics: Mapping[str, Mapping[str, float]],
    fp16_metrics: Mapping[str, Mapping[str, float]],
) -> FidelityVerdict:
    """Plan §4-P2's fp16 fidelity gate over the manifest's predeclared first eight DEV examples.

    Both tables are ``name -> {"future_ssim", "future_mse"}`` for the *same* replay, once in memory
    at fp32 and once after the fp16 round-trip, and both must cover exactly that derived subset --
    the subset is never a caller's argument. The cache stays fp16 only if **every** example satisfies
    ``ssim_drop <= 0.01`` and ``mse_increase <= 5%``; otherwise the caller flips ``latent_dtype`` to
    fp32 for exactly ``{nulls, z_start, expected_final_latent}`` (plan M5).

    What this proves is exactly the deltas in the metric tables it was handed, and nothing about how
    those numbers were produced. In particular it says nothing about writer order: R6 measured that
    at fp16 storage a replay-before-cast mistake moves the endpoint by 2e-3..8e-3, which is inside
    any tolerance an fp16 record needs from ``verify_replay``, so that order is pinned separately, at
    fp32, by R6's writer-order test.
    """
    manifest = _checked_names(dev_manifest, "the DEV manifest")
    if len(manifest) < FIDELITY_SUBSET_SIZE:
        raise ValueError(f"the DEV manifest must carry at least {FIDELITY_SUBSET_SIZE} names, got {len(manifest)}")
    subset = manifest[:FIDELITY_SUBSET_SIZE]
    for label, table in (("fp32", fp32_metrics), ("fp16", fp16_metrics)):
        if set(table) != set(subset):
            raise ValueError(f"the {label} table must cover exactly the predeclared first {FIDELITY_SUBSET_SIZE}")

    per_example: dict[str, dict[str, float]] = {}
    for name in subset:
        exact = _fidelity_metrics(fp32_metrics, name, "fp32")
        stored = _fidelity_metrics(fp16_metrics, name, "fp16")
        if exact["future_mse"] <= 0.0:
            raise ValueError(f"fp32[{name!r}] future_mse must be positive, got {exact['future_mse']}")
        if stored["future_mse"] < 0.0:
            raise ValueError(f"fp16[{name!r}] future_mse must not be negative, got {stored['future_mse']}")
        per_example[name] = {
            "ssim_drop": exact["future_ssim"] - stored["future_ssim"],
            "mse_increase": (stored["future_mse"] - exact["future_mse"]) / exact["future_mse"],
        }

    worst_ssim = max(entry["ssim_drop"] for entry in per_example.values())
    worst_mse = max(entry["mse_increase"] for entry in per_example.values())
    reasons = []
    if worst_ssim > FIDELITY_MAX_SSIM_DROP + FIDELITY_BOUNDARY_ATOL:
        reasons.append("ssim_drop")
    if worst_mse > FIDELITY_MAX_MSE_INCREASE + FIDELITY_BOUNDARY_ATOL:
        reasons.append("mse_increase")
    return FidelityVerdict(
        passed=not reasons,
        latent_dtype="fp16" if not reasons else "fp32",
        subset=subset,
        worst_ssim_drop=worst_ssim,
        worst_mse_increase=worst_mse,
        per_example=per_example,
        reasons=tuple(reasons),
    )
