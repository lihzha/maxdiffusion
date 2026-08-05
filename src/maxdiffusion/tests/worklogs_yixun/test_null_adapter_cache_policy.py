"""exp_04 R8 — ``null_adapter_cache_policy``: what the cache job drops, and what it stores at.

Two judgments live here, both pure and both plan-affecting:

- **Quarantine** decides what a failing example costs. The reviewer's conditional ratification turns
  on one distinction: a *pathological example* may become a recorded gap, while a configuration
  error, an OOM or a preempted host may not -- those are systemic and must stop the job. So only
  ``ExampleDivergenceError`` is quarantineable, and the runner is contractually required to raise
  that type (and nothing else) for per-example pathology.
- **The fidelity gate** decides the storage dtype from the predeclared first eight DEV examples --
  derived here from the manifest, never chosen by the caller, because a cherry-picked subset would
  make the fp16 decision unfalsifiable.

Both were split out of ``null_adapter_shards`` in the R8 strengthening: they judge, they do no IO,
and keeping them next to the shard writer made a 450-line module out of two 100-line ideas.
"""

from __future__ import annotations

import numpy as np
import pytest

from maxdiffusion.null_adapter_cache_policy import (
    FIDELITY_BOUNDARY_ATOL,
    FIDELITY_MAX_MSE_INCREASE,
    FIDELITY_MAX_SSIM_DROP,
    FIDELITY_SUBSET_SIZE,
    ExampleDivergenceError,
    fidelity_gate,
    quarantine_batch_failures,
)


_NAMES = tuple(f"droid_ep_{index:06d}/w0" for index in range(3))
_DEV = tuple(f"droid_ep_{index:06d}/w0" for index in range(12))


def _diverging_run(bad_names, error_type=ExampleDivergenceError):
    """A runner that raises for any batch containing a bad example -- the shape R6's hard trace
    failure has when the orchestration boundary wraps it."""
    bad = set(bad_names)

    def run_fn(names):
        hit = sorted(bad & set(names))
        if hit:
            raise error_type(f"tracking_losses must be finite ({hit[0]})")
        return {name: {"result": name} for name in names}

    return run_fn


def test_a_clean_batch_runs_once_and_quarantines_nothing():
    calls = []

    def run_fn(names):
        calls.append(tuple(names))
        return {name: name for name in names}

    results, quarantined = quarantine_batch_failures(run_fn, _NAMES)

    assert results == {name: name for name in _NAMES} and quarantined == {}
    assert calls == [_NAMES]  # the happy path costs exactly one pass


def test_one_diverging_example_is_quarantined_and_the_survivors_are_rerun_as_a_batch():
    """Ruling 2's strengthening: the survivor batch is re-run, so the composition failure is proven
    gone rather than assumed gone -- and the returned results come from that batched re-run."""
    calls = []
    inner = _diverging_run({_NAMES[1]})

    def run_fn(names):
        calls.append(tuple(names))
        return inner(names)

    results, quarantined = quarantine_batch_failures(run_fn, _NAMES)

    assert set(results) == {_NAMES[0], _NAMES[2]}
    assert set(quarantined) == {_NAMES[1]}
    assert "tracking_losses must be finite" in quarantined[_NAMES[1]]
    assert calls[0] == _NAMES  # the failing batch ...
    assert calls[1:4] == [(_NAMES[0],), (_NAMES[1],), (_NAMES[2],)]  # ... the singleton triage ...
    assert calls[-1] == (_NAMES[0], _NAMES[2])  # ... and the survivor re-run


def test_a_survivor_batch_that_still_fails_is_not_papered_over():
    state = {"survivor_attempts": 0}

    def run_fn(names):
        if _NAMES[1] in names and len(names) == 1:
            raise ExampleDivergenceError("diverged")
        if len(names) == 1:
            return {names[0]: names[0]}
        state["survivor_attempts"] += 1
        raise RuntimeError("host ran out of memory")

    with pytest.raises(RuntimeError, match="host ran out of memory"):
        quarantine_batch_failures(run_fn, _NAMES)
    assert state["survivor_attempts"] == 2  # the original batch and the survivor re-run


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError, OSError, ValueError, KeyboardInterrupt, SystemExit])
def test_only_example_divergence_is_quarantineable(error_type):
    """A config error, an OOM or a preempted host is systemic: turning it into a data gap would let
    an infrastructure failure quietly shrink the cohort (Codex R8 review, finding 5)."""
    with pytest.raises(error_type):
        quarantine_batch_failures(_diverging_run({_NAMES[1]}, error_type), _NAMES)


def test_a_batch_only_failure_is_re_raised():
    """Ratified in R8: per-example independence is a plan §3 contract, so a failure no singleton
    reproduces is a composition/capacity bug and must propagate."""
    state = {"first": True}

    def run_fn(names):
        if len(names) > 1 and state["first"]:
            state["first"] = False
            raise ExampleDivergenceError("only when batched")
        return {name: name for name in names}

    with pytest.raises(ExampleDivergenceError, match="only when batched"):
        quarantine_batch_failures(run_fn, _NAMES)


def test_an_all_diverging_batch_is_re_raised_rather_than_becoming_an_empty_shard():
    """Ruling 3: an all-quarantined result is never a completed boundary, so it never gets written."""
    with pytest.raises(ExampleDivergenceError):
        quarantine_batch_failures(_diverging_run(set(_NAMES)), _NAMES)


def test_a_runner_that_returns_the_wrong_namespace_is_a_hard_error():
    def dropping(names):
        return {name: name for name in names[:-1]}

    with pytest.raises(ValueError, match="did not return a result"):
        quarantine_batch_failures(dropping, _NAMES)

    def stranger(names):
        return {**{name: name for name in names}, "stranger": 1}

    with pytest.raises(ValueError, match="did not return a result"):
        quarantine_batch_failures(stranger, _NAMES)


def test_the_batch_must_be_a_non_empty_unique_name_sequence():
    for names, message in (((), "at least one"), ((_NAMES[0], _NAMES[0]), "unique")):
        with pytest.raises(ValueError, match=message):
            quarantine_batch_failures(lambda batch: {}, names)


def test_the_divergence_error_is_the_documented_runner_seam():
    assert issubclass(ExampleDivergenceError, Exception)
    assert not issubclass(ExampleDivergenceError, (KeyboardInterrupt, SystemExit))
    assert "runner" in (ExampleDivergenceError.__doc__ or "").lower()


def _metrics(ssim, mse, names):
    return {name: {"future_ssim": ssim, "future_mse": mse} for name in names}


def _subset(names=_DEV):
    return tuple(names[:FIDELITY_SUBSET_SIZE])


def test_the_subset_is_the_manifest_s_first_eight_not_the_caller_s_choice():
    """The estimand is fixed by plan §4-P2; a caller who could pick the eight could pick the verdict."""
    assert FIDELITY_SUBSET_SIZE == 8
    verdict = fidelity_gate(_DEV, _metrics(0.9, 1.0, _subset()), _metrics(0.895, 1.02, _subset()))

    assert verdict.passed and verdict.latent_dtype == "fp16"
    assert tuple(sorted(verdict.per_example)) == tuple(sorted(_subset()))
    assert verdict.subset == _subset()


def test_a_cherry_picked_table_is_refused():
    cherry = _DEV[4:12]  # eight real DEV names -- just not the predeclared eight

    with pytest.raises(ValueError, match="predeclared first"):
        fidelity_gate(_DEV, _metrics(0.9, 1.0, cherry), _metrics(0.9, 1.0, cherry))


@pytest.mark.parametrize(
    "manifest, message",
    [
        (_DEV[:4], "at least"),
        ((), "at least"),
        ((*_DEV[:7], _DEV[0]), "unique"),
    ],
)
def test_an_unusable_dev_manifest_is_refused(manifest, message):
    with pytest.raises(ValueError, match=message):
        fidelity_gate(manifest, _metrics(0.9, 1.0, _DEV), _metrics(0.9, 1.0, _DEV))


def test_the_plan_s_fidelity_thresholds_are_pinned_literally():
    """Literal, not derived: a boundary test written in terms of the constants moves with them."""
    assert (FIDELITY_MAX_SSIM_DROP, FIDELITY_MAX_MSE_INCREASE) == (0.01, 0.05)
    assert FIDELITY_BOUNDARY_ATOL == 1e-9


@pytest.mark.parametrize(
    "ssim_fp16, mse_fp16, passes, reason",
    [
        (0.89, 1.0, True, None),  # exactly a 0.01 drop passes (<=), despite binary floating point
        (0.889999, 1.0, False, "ssim_drop"),
        (0.9, 1.05, True, None),  # exactly +5% passes
        (0.9, 1.0500001, False, "mse_increase"),
    ],
)
def test_the_thresholds_are_inclusive_at_the_boundary(ssim_fp16, mse_fp16, passes, reason):
    verdict = fidelity_gate(_DEV, _metrics(0.9, 1.0, _subset()), _metrics(ssim_fp16, mse_fp16, _subset()))

    assert verdict.passed is passes
    assert verdict.latent_dtype == ("fp16" if passes else "fp32")
    if reason:
        assert reason in verdict.reasons


def test_the_gate_is_worst_example_not_mean():
    fp32 = _metrics(0.9, 1.0, _subset())
    fp16 = _metrics(0.9, 1.0, _subset())
    fp16[_subset()[3]] = {"future_ssim": 0.86, "future_mse": 1.3}  # 0.04 drop, +30% MSE, on one

    verdict = fidelity_gate(_DEV, fp32, fp16)

    assert not verdict.passed and verdict.latent_dtype == "fp32"
    assert verdict.worst_ssim_drop == pytest.approx(0.04) and verdict.worst_mse_increase == pytest.approx(0.3)
    assert sorted(verdict.reasons) == ["mse_increase", "ssim_drop"]
    # A mean-based gate would have passed both conditions on exactly this table -- that is the point.
    assert float(np.mean([e["ssim_drop"] for e in verdict.per_example.values()])) < FIDELITY_MAX_SSIM_DROP
    assert float(np.mean([e["mse_increase"] for e in verdict.per_example.values()])) < FIDELITY_MAX_MSE_INCREASE


def test_the_mse_condition_is_relative_not_absolute():
    """+0.08 on a 4.0 baseline is 2%; an absolute threshold would reject it, and the 1.0 baselines
    used elsewhere here cannot tell the two apart."""
    verdict = fidelity_gate(_DEV, _metrics(0.9, 4.0, _subset()), _metrics(0.9, 4.08, _subset()))

    assert verdict.passed and verdict.worst_mse_increase == pytest.approx(0.02)
    assert fidelity_gate(_DEV, _metrics(0.9, 4.0, _subset()), _metrics(0.9, 4.3, _subset())).reasons == (
        "mse_increase",
    )


def test_an_fp16_that_improves_a_metric_is_not_penalised():
    verdict = fidelity_gate(_DEV, _metrics(0.9, 1.0, _subset()), _metrics(0.95, 0.5, _subset()))

    assert verdict.passed and verdict.worst_ssim_drop < 0.0 and verdict.worst_mse_increase < 0.0


@pytest.mark.parametrize(
    "fp32, fp16, message",
    [
        (_metrics(0.9, 1.0, _DEV[:7]), _metrics(0.9, 1.0, _DEV[:7]), "predeclared first"),
        (_metrics(0.9, 0.0, _DEV[:8]), _metrics(0.9, 0.0, _DEV[:8]), "must be positive"),
        (_metrics(0.9, -1.0, _DEV[:8]), _metrics(0.9, 1.0, _DEV[:8]), "must be positive"),
        (_metrics(0.9, 1.0, _DEV[:8]), _metrics(0.9, -0.5, _DEV[:8]), "must not be negative"),
        (_metrics(0.9, float("nan"), _DEV[:8]), _metrics(0.9, 1.0, _DEV[:8]), "finite"),
        (_metrics(0.9, 1.0, _DEV[:8]), _metrics(float("inf"), 1.0, _DEV[:8]), "finite"),
        ({name: {"future_ssim": 0.9} for name in _DEV[:8]}, _metrics(0.9, 1.0, _DEV[:8]), "future_mse"),
    ],
)
def test_the_fidelity_gate_fails_closed_on_unusable_input(fp32, fp16, message):
    with pytest.raises(ValueError, match=message):
        fidelity_gate(_DEV, fp32, fp16)


def test_the_docstring_attributes_the_writer_order_limit_to_r6():
    """The fp16 non-discrimination measurement was made in R6, and the gate must say what it proves."""
    doc = fidelity_gate.__doc__

    assert "R6" in doc and "R7" not in doc
    assert "metric table" in doc
