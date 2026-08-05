"""exp_04 R5 — the gate module: the experiment's verdicts are code, not prose (plan §3, N4/M7).

Every claim exp_04 will make passes through here, so the semantics that decide "pass" are pinned
harder than the numbers they consume. What the round hangs on:

- **Imputation is claim-penalizing, and its direction is the point.** A missing observation is never
  dropped -- dropping is what turns a broken run into a good-looking one. Invalid pairs push the
  aggregate *against* the claim: ratio -> 1.0 and improved -> false for the method-vs-control gates,
  ΔSSIM -> -1.0 for the paired difference. A missing *baseline* can never award the adapter an
  advantage, which is the asymmetry M7/P1 exists to force.
- **Validity comes from both metrics of both sides.** A missing SSIM invalidates its observation just
  as a missing MSE does; the review found a 7/64-missing-method-SSIM table reporting zero invalidity,
  so the boundary cases below are checked for each side and each metric.
- **The estimand is not the caller's to choose.** Public gates take a ``NoiseConvention`` and derive
  the k-set. The same probe can pass at ``k={0}`` and fail under the declared keyed ``{0,1,2}``.
- **The JSON boundary is exact.** Duplicate keys are rejected rather than collapsed, undefined ratios
  serialize as ``null``, and output is strict, sorted JSON.
- **The bootstrap is pinned to its seed**, not merely to itself: a golden CI fixes 20260804.

The tables are synthetic by design: hand-built, so the right answer is arithmetic rather than opinion.
"""

from __future__ import annotations

import json
import math

import pytest

from maxdiffusion.null_adapter_gates import (
    GLOBAL_K_SET,
    KEYED_K_SET,
    GateVerdict,
    NoiseConvention,
    Target,
    _gate_g1,
    _select_target,
    gate_g1,
    gate_g2,
    gate_g3_vs_baseline,
    gate_g3_vs_null_only,
    k_set_for,
    parse_table,
    select_target,
    verdicts_to_json,
)


_GLOBAL = NoiseConvention.GLOBAL  # k-set {0}: the shape most fixtures below use
_K = GLOBAL_K_SET


def _names(count, prefix="droid_ep_"):
    return [f"{prefix}{i:06d}/w0" for i in range(count)]


def _table(values, k_set=_K):
    """``values``: name -> (mse, ssim); a name mapped to ``None`` is absent from the table."""
    return {
        name: {str(k): {"future_mse": value[0], "future_ssim": value[1]} for k in k_set}
        for name, value in values.items()
        if value is not None
    }


def _uniform(names, mse, ssim, k_set=_K):
    return _table(dict.fromkeys(names, (mse, ssim)), k_set)


def _g1_tables(names, *, method_mse=1.0, control_mse=10.0, method_ssim=0.9):
    return {"method": _uniform(names, method_mse, method_ssim), "control": _uniform(names, control_mse, 0.5)}


def _g3_tables(names, *, adapter_ssim=0.60, baseline_ssim=0.50):
    return {"method": _uniform(names, 1.0, adapter_ssim), "control": _uniform(names, 1.0, baseline_ssim)}


def _probe_table(names, ssim, k_set=_K):
    return _uniform(names, 1.0, ssim, k_set)


def _verdict(passed):
    return GateVerdict(passed=passed, reasons=() if passed else ("stub",), numbers={"mean_ssim": 1.0})


def test_a_clean_table_passes_g1():
    names = _names(10)
    tables = _g1_tables(names)

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert verdict.passed, verdict.reasons
    assert verdict.numbers["median_ratio"] == pytest.approx(10.0)
    assert verdict.numbers["fraction_improved"] == pytest.approx(1.0)
    assert verdict.numbers["mean_ssim"] == pytest.approx(0.9)
    assert verdict.numbers["invalid_pairs"] == 0
    assert verdict.numbers["k_set"] == list(GLOBAL_K_SET)


@pytest.mark.parametrize(
    "kwargs, reason",
    [({"control_mse": 3.0}, "median_ratio"), ({"method_ssim": 0.78}, "mean_ssim")],
)
def test_each_g1_condition_is_individually_decisive(kwargs, reason):
    names = _names(10)
    tables = _g1_tables(names, **kwargs)

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert not verdict.passed
    assert verdict.reasons == (reason,), verdict.reasons


def test_g1_fraction_improved_is_decisive_on_its_own():
    names = _names(10)
    tables = _g1_tables(names)
    for name in names[:3]:  # three examples where the method is worse: ratio 0.5, median still 10
        tables["method"][name] = {"0": {"future_mse": 20.0, "future_ssim": 0.9}}

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert not verdict.passed
    assert verdict.reasons == ("fraction_improved",), verdict.reasons
    assert verdict.numbers["median_ratio"] == pytest.approx(10.0)


def test_g1_ci_lower_bound_is_decisive_on_its_own():
    """Mean above the bar, spread wide enough that the bootstrap's 2.5th percentile is not."""
    names = _names(10)
    tables = _g1_tables(names)
    for index, name in enumerate(names):
        tables["method"][name] = {"0": {"future_mse": 1.0, "future_ssim": 0.30 if index < 2 else 0.95}}

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert verdict.numbers["mean_ssim"] >= 0.80
    assert verdict.reasons == ("ssim_ci_low",), verdict.reasons


def test_a_fraction_exactly_at_the_threshold_passes():
    """80% improved is a pass, not a near miss: the comparison is >=, and the boundary is the target."""
    names = _names(10)
    tables = _g1_tables(names)
    for name in names[:2]:  # exactly 8/10 improved
        tables["method"][name] = {"0": {"future_mse": 20.0, "future_ssim": 0.9}}

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert verdict.numbers["fraction_improved"] == pytest.approx(0.80)
    assert "fraction_improved" not in verdict.reasons, verdict.reasons


def test_g2_uses_its_own_ssim_thresholds():
    names = _names(10)
    tables = _g1_tables(names, method_ssim=0.78)

    assert gate_g2(tables["method"], tables["control"], names, _GLOBAL).passed
    assert not gate_g1(tables["method"], tables["control"], names, _GLOBAL).passed


@pytest.mark.parametrize("gate", [gate_g1, gate_g2])
def test_coverage_failures_are_reported_by_kind(gate):
    names = _names(4)
    tables = _g1_tables(names)
    missing = {name: entry for name, entry in tables["method"].items() if name != names[0]}
    stranger = {**tables["method"], "stranger/w0": {"0": {"future_mse": 1.0, "future_ssim": 0.9}}}

    missing_verdict = gate(missing, tables["control"], names, _GLOBAL)
    extra_verdict = gate(stranger, tables["control"], names, _GLOBAL)
    duplicate_verdict = gate(tables["method"], tables["control"], names + [names[0]], _GLOBAL)

    assert missing_verdict.reasons == ("coverage",)
    assert missing_verdict.numbers["missing_names"]["method"] == [names[0]]
    assert extra_verdict.numbers["extra_names"]["method"] == ["stranger/w0"]
    assert duplicate_verdict.numbers["duplicate_names"] == [names[0]]


def test_seed_reduction_averages_the_declared_k_set():
    names = _names(4)
    method = {
        name: {"0": {"future_mse": 1.0, "future_ssim": 0.8}, "1": {"future_mse": 3.0, "future_ssim": 1.0}}
        for name in names
    }
    control = _uniform(names, 10.0, 0.5, k_set=(0, 1))

    verdict = _gate_g1(method, control, names, (0, 1))

    assert verdict.numbers["mean_ssim"] == pytest.approx(0.9)  # mean of 0.8 and 1.0
    assert verdict.numbers["median_ratio"] == pytest.approx(5.0)  # 10.0 / mean(1.0, 3.0)


def test_a_missing_seed_invalidates_the_whole_example():
    """Fail-closed at reduction: a k-set is a declaration, not a wish."""
    names = _names(10)
    method = _uniform(names, 1.0, 0.9, k_set=(0, 1))
    control = _uniform(names, 10.0, 0.5, k_set=(0, 1))
    method[names[0]].pop("1")

    verdict = _gate_g1(method, control, names, (0, 1))

    assert verdict.numbers["invalid_pairs"] == 1
    assert verdict.numbers["fraction_improved"] == pytest.approx(0.9)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
@pytest.mark.parametrize("metric", ["future_mse", "future_ssim"])
def test_nonfinite_and_missing_metrics_are_invalid(bad, metric):
    names = _names(10)
    tables = _g1_tables(names)
    entry = {"future_mse": 1.0, "future_ssim": 0.9}
    if bad is None:
        entry.pop(metric)
    else:
        entry[metric] = bad
    tables["method"][names[0]] = {"0": entry}

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert verdict.numbers["invalid_pairs"] == 1


@pytest.mark.parametrize("side", ["method", "control"])
@pytest.mark.parametrize("metric", ["future_mse", "future_ssim"])
@pytest.mark.parametrize("broken, auto_fails", [(6, False), (7, True)])
def test_invalidity_propagates_from_either_metric_of_either_side(side, metric, broken, auto_fails):
    """6/64 survives, 7/64 does not -- and a missing SSIM counts exactly like a missing MSE.

    The review's finding: pair validity used to read only the two MSEs, so seven missing method SSIMs
    reported zero invalidity and the gate passed on 57 examples' worth of evidence.
    """
    names = _names(64)
    tables = _g1_tables(names)
    for name in names[:broken]:
        entry = dict(tables[side][name]["0"])
        entry.pop(metric)
        tables[side][name] = {"0": entry}

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert verdict.numbers["invalid_pairs"] == broken
    assert verdict.numbers["invalid_fraction"] == pytest.approx(broken / 64)
    assert ("invalid_fraction" in verdict.reasons) is auto_fails, verdict.reasons
    assert verdict.numbers["fraction_improved"] == pytest.approx((64 - broken) / 64)


def test_g1_imputation_direction_differs_for_method_and_control():
    """An invalid method costs it the SSIM mean; an invalid control does not hand it a gift."""
    names = _names(10)
    method_bad = _g1_tables(names)
    method_bad["method"][names[0]] = {"0": {"future_mse": float("nan"), "future_ssim": 0.9}}
    control_bad = _g1_tables(names)
    control_bad["control"][names[0]] = {"0": {"future_mse": float("nan"), "future_ssim": 0.5}}

    method_verdict = gate_g1(method_bad["method"], method_bad["control"], names, _GLOBAL)
    control_verdict = gate_g1(control_bad["method"], control_bad["control"], names, _GLOBAL)

    assert method_verdict.numbers["mean_ssim"] == pytest.approx(0.81)  # 9 x 0.9, the tenth imputed to 0
    assert control_verdict.numbers["mean_ssim"] == pytest.approx(0.9)  # the method's measurement stands
    for verdict in (method_verdict, control_verdict):
        assert verdict.numbers["invalid_pairs"] == 1
        assert verdict.numbers["fraction_improved"] == pytest.approx(0.9)
        assert verdict.numbers["per_example_ratio"][0] is None  # undefined, not a favourable number
    assert method_verdict.numbers["imputed_method_ssim"] == 1
    assert control_verdict.numbers["imputed_method_ssim"] == 0


def test_g3_imputes_against_the_adapter_whichever_side_is_missing():
    """The asymmetry M7 exists to force: a missing baseline must never look like an adapter win."""
    names = _names(10)
    adapter_bad = _g3_tables(names)
    adapter_bad["method"][names[0]] = {"0": {"future_mse": 1.0, "future_ssim": float("nan")}}
    baseline_bad = _g3_tables(names)
    baseline_bad["control"][names[0]] = {"0": {"future_mse": 1.0, "future_ssim": float("nan")}}

    for tables in (adapter_bad, baseline_bad):
        verdict = gate_g3_vs_null_only(tables["method"], tables["control"], names, _GLOBAL)
        assert verdict.numbers["per_example_delta"][0] == pytest.approx(-1.0), verdict.numbers
        assert verdict.numbers["fraction_improved"] == pytest.approx(0.9)
        assert verdict.numbers["mean_delta"] == pytest.approx((9 * 0.10 - 1.0) / 10)
        assert not verdict.passed


def test_ssim_is_clipped_to_the_unit_interval_before_differencing():
    names = _names(4)
    tables = _g3_tables(names)
    tables["method"][names[0]] = {"0": {"future_mse": 1.0, "future_ssim": 1.4}}  # clipped to 1.0
    tables["control"][names[0]] = {"0": {"future_mse": 1.0, "future_ssim": -0.3}}  # clipped to 0.0

    verdict = gate_g3_vs_null_only(tables["method"], tables["control"], names, _GLOBAL)

    assert verdict.numbers["per_example_delta"][0] == pytest.approx(1.0)  # not 1.7


def test_the_conventions_k_set_is_derived_not_supplied():
    """A probe that passes at k={0} can fail under the declared keyed reduction -- and must."""
    names = _names(10)
    method = {
        name: {
            "0": {"future_mse": 1.0, "future_ssim": 0.60},
            "1": {"future_mse": 1.0, "future_ssim": 0.30},
            "2": {"future_mse": 1.0, "future_ssim": 0.30},
        }
        for name in names
    }
    baseline = _uniform(names, 1.0, 0.50, k_set=(0, 1, 2))

    assert gate_g3_vs_null_only(method, baseline, names, NoiseConvention.GLOBAL).passed
    keyed = gate_g3_vs_null_only(method, baseline, names, NoiseConvention.KEYED)
    assert not keyed.passed
    assert keyed.numbers["k_set"] == list(KEYED_K_SET)
    assert keyed.numbers["mean_delta"] == pytest.approx(-0.10)


@pytest.mark.parametrize("bad", ["fresh", "KEYED", 0, None, (0, 1)])
def test_an_unknown_convention_is_refused(bad):
    with pytest.raises(ValueError):
        k_set_for(bad)


def test_conventions_carry_the_plan_s_fixed_k_sets():
    assert k_set_for(NoiseConvention.KEYED) == (0, 1, 2) == KEYED_K_SET
    assert k_set_for(NoiseConvention.GLOBAL) == (0,) == GLOBAL_K_SET
    assert k_set_for("keyed") == KEYED_K_SET  # the string a record stores also resolves


@pytest.mark.parametrize("invalid, auto_fails", [(1, False), (2, True)])
def test_the_invalidity_threshold_is_strictly_greater_than_ten_percent(invalid, auto_fails):
    """Exactly 10% survives; one more pair does not. The boundary is the mutation target."""
    names = _names(10)
    tables = _g1_tables(names, method_ssim=0.99)
    for name in names[:invalid]:
        tables["method"][name] = {"0": {"future_mse": float("nan"), "future_ssim": 0.99}}

    verdict = gate_g1(tables["method"], tables["control"], names, _GLOBAL)

    assert ("invalid_fraction" in verdict.reasons) is auto_fails, verdict.reasons
    assert verdict.numbers["invalid_fraction"] == pytest.approx(invalid / 10)


def test_a_confidence_interval_touching_zero_does_not_exclude_zero():
    names = _names(8)
    flat = _g3_tables(names, adapter_ssim=0.5, baseline_ssim=0.5)  # every delta exactly 0.0

    verdict = gate_g3_vs_null_only(flat["method"], flat["control"], names, _GLOBAL)

    assert verdict.numbers["ci"] == [0.0, 0.0]
    assert verdict.reasons == ("mean_delta", "ci_excludes_zero"), verdict.reasons


def test_bootstrap_is_deterministic_and_degenerate_on_constant_data():
    names = _names(8)
    tables = _g3_tables(names)

    first = gate_g3_vs_null_only(tables["method"], tables["control"], names, _GLOBAL)
    second = gate_g3_vs_null_only(tables["method"], tables["control"], names, _GLOBAL)

    assert first.numbers["ci"] == second.numbers["ci"]
    assert first.numbers["ci"][0] == pytest.approx(0.10)  # every resample of a constant is the constant
    assert first.numbers["bootstrap_seed"] == 20260804
    assert first.numbers["bootstrap_resamples"] == 10000


def test_the_bootstrap_ci_is_pinned_to_the_declared_seed():
    """A golden interval, computed once under seed 20260804 with 10k example-level resamples.

    Reporting the seed while resampling under another one left every other test green, so the seed
    is pinned by its consequences here, not by the constant the verdict advertises.
    """
    names = _names(64)
    method = _table({name: (1.0, 0.30 + 0.01 * index) for index, name in enumerate(names)})
    baseline = _uniform(names, 1.0, 0.50)

    verdict = gate_g3_vs_null_only(method, baseline, names, _GLOBAL)

    assert verdict.numbers["mean_delta"] == pytest.approx(0.115, abs=5e-4)
    assert verdict.numbers["ci"][0] == pytest.approx(0.06953125, abs=5e-8)
    assert verdict.numbers["ci"][1] == pytest.approx(0.159375, abs=5e-8)


def test_g3_vs_null_only_conditions_are_individually_decisive():
    names = _names(10)
    thin = _g3_tables(names, adapter_ssim=0.53)  # delta 0.03 < 0.05 margin
    straddling = {
        "method": _table({name: (1.0, 0.9 if index % 2 else 0.1) for index, name in enumerate(names)}),
        "control": _uniform(names, 1.0, 0.4),
    }

    thin_verdict = gate_g3_vs_null_only(thin["method"], thin["control"], names, _GLOBAL)
    straddle_verdict = gate_g3_vs_null_only(straddling["method"], straddling["control"], names, _GLOBAL)

    assert thin_verdict.reasons == ("mean_delta",), thin_verdict.reasons
    assert straddle_verdict.numbers["mean_delta"] >= 0.05
    assert straddle_verdict.reasons == ("ci_excludes_zero",), straddle_verdict.reasons


def test_g3_vs_baseline_adds_the_improved_fraction_condition():
    names = _names(10)
    lopsided = {
        "method": _table({name: (1.0, 0.80 if index < 5 else 0.48) for index, name in enumerate(names)}),
        "control": _uniform(names, 1.0, 0.50),
    }

    verdict = gate_g3_vs_baseline(lopsided["method"], lopsided["control"], names, _GLOBAL)

    assert verdict.numbers["mean_delta"] >= 0.02
    assert verdict.numbers["fraction_improved"] == pytest.approx(0.5)
    assert verdict.reasons == ("fraction_improved",), verdict.reasons


@pytest.mark.parametrize(
    "g1_passed, probe_ssim, g2_passed, expected",
    [
        (True, 0.80, False, Target.A1_KEYED),
        (True, 0.69, True, Target.A2_GLOBAL),  # below the 0.70 floor
        (True, 0.69, False, Target.STOP),
        (False, 0.99, True, Target.A2_GLOBAL),
        (False, 0.99, False, Target.STOP),
    ],
)
def test_select_target_truth_table(g1_passed, probe_ssim, g2_passed, expected):
    names = _names(8)

    selection = select_target(
        _verdict(g1_passed), _probe_table(names, probe_ssim), names, _GLOBAL, _verdict(g2_passed)
    )

    assert selection.target is expected, selection.reasons
    assert selection.reasons  # a selection always says why


def test_select_target_reduces_the_probe_table_under_the_gate_machinery():
    """Pre-aggregated scalars used to walk in unchecked; now the probe faces the same rules."""
    names = _names(10)
    probe = _probe_table(names, 0.95)
    probe[names[0]] = {"0": {"future_mse": 1.0, "future_ssim": float("nan")}}
    probe.pop(names[1])

    selection = select_target(_verdict(True), probe, names, _GLOBAL, _verdict(False))

    assert selection.target is Target.STOP
    assert selection.numbers["missing_names"]["probe"] == [names[1]]
    assert any("coverage" in reason for reason in selection.reasons), selection.reasons


@pytest.mark.parametrize("broken, expected", [(1, Target.A1_KEYED), (2, Target.STOP)])
def test_select_target_imputes_invalid_probe_examples_against_the_claim(broken, expected):
    """The probe faces the gates' own ceiling: 10% survives (imputed), more than 10% does not."""
    names = _names(10)
    probe = _probe_table(names, 0.95)
    for name in names[:broken]:
        probe[name] = {"0": {"future_mse": 1.0, "future_ssim": float("nan")}}

    selection = select_target(_verdict(True), probe, names, _GLOBAL, _verdict(False))

    assert selection.numbers["probe_invalid_pairs"] == broken
    assert selection.numbers["probe_mean_ssim"] == pytest.approx((10 - broken) * 0.95 / 10)  # imputed to 0
    assert selection.target is expected, selection.reasons


@pytest.mark.parametrize("out_of_range", [1.4, -0.2])
def test_select_target_rejects_out_of_range_probe_ssim(out_of_range):
    """An SSIM above 1.0 would inflate the floor comparison; it is an invalid observation, not data.

    Two such examples take the probe over the invalidity ceiling, so the out-of-range values can
    neither be averaged in nor quietly dropped.
    """
    names = _names(10)
    probe = _probe_table(names, 0.95)
    for name in names[:2]:
        probe[name] = {"0": {"future_mse": 1.0, "future_ssim": out_of_range}}

    selection = select_target(_verdict(True), probe, names, _GLOBAL, _verdict(False))

    assert selection.numbers["probe_invalid_pairs"] == 2
    assert selection.numbers["probe_mean_ssim"] == pytest.approx(0.76)  # 8 x 0.95, two imputed to 0
    assert selection.target is Target.STOP
    assert any("invalidity ceiling" in reason for reason in selection.reasons), selection.reasons


def test_a_failed_transfer_conjunct_always_reports_the_floor_too():
    """With SSIM in [0, 1] the floor implies the transfer test, so no write-up can ever honestly say
    "rejected on transfer alone" -- asserted here so a future refactor cannot make it sayable."""
    names = _names(8)

    for probe_ssim in (0.0, 0.35, 0.69):
        selection = select_target(_verdict(True), _probe_table(names, probe_ssim), names, _GLOBAL, _verdict(False))
        transfer = [r for r in selection.reasons if "transfer" in r]
        floor = [r for r in selection.reasons if "floor" in r]
        assert not transfer or floor, selection.reasons


def test_the_relative_transfer_conjunct_is_still_pinned():
    """Only reachable with an out-of-range A1 mean, which the public API cannot produce -- so the
    branch is exercised through the private seam to keep it from rotting."""
    names = _names(8)

    selection = _select_target(_verdict(True), _probe_table(names, 0.75), names, _K, _verdict(False), a1_mean_ssim=2.0)

    assert selection.target is Target.STOP
    assert any("transfer" in reason for reason in selection.reasons), selection.reasons


def test_parse_table_rejects_duplicate_keys_before_coverage_can_see_them():
    """json.load would keep the last value and report a table that covers the manifest exactly once."""
    raw = '{"a/w0": {"0": {"future_mse": 1.0, "future_ssim": 0.9}}, "a/w0": {"0": {"future_mse": 9.0, "future_ssim": 0.1}}}'

    with pytest.raises(ValueError, match="duplicate JSON keys"):
        parse_table(raw)

    assert set(parse_table('{"a/w0": {"0": {"future_mse": 1.0, "future_ssim": 0.9}}}')) == {"a/w0"}


def test_verdicts_serialize_to_strict_deterministic_json():
    names = _names(6)
    tables = _g1_tables(names)
    tables["method"][names[0]] = {"0": {"future_mse": float("nan"), "future_ssim": 0.9}}
    verdicts = {
        "G1": gate_g1(tables["method"], tables["control"], names, _GLOBAL),
        "selection": select_target(_verdict(True), _probe_table(names, 0.9), names, _GLOBAL, _verdict(True)),
    }

    text = verdicts_to_json(verdicts)

    assert text == verdicts_to_json(verdicts)
    assert "NaN" not in text and "Infinity" not in text
    restored = json.loads(text, parse_constant=_reject_constant)
    assert restored["G1"]["numbers"]["per_example_ratio"][0] is None  # undefined ratio -> null
    assert list(restored) == sorted(restored)


def test_serialization_refuses_nonfinite_numbers_outright():
    """Nothing the public gates produce carries NaN, but the serializer still refuses to emit one:
    a results file with a bare NaN token is not JSON, and would fail at whoever reads it next."""
    poisoned = GateVerdict(passed=True, reasons=(), numbers={"mean_ssim": float("nan")})

    with pytest.raises(ValueError):
        verdicts_to_json({"G1": poisoned})


def _reject_constant(token):
    raise AssertionError(f"strict JSON must not contain {token}")


def test_probe_relative_is_null_rather_than_infinite():
    names = _names(4)

    selection = _select_target(_verdict(True), _probe_table(names, 0.9), names, _K, _verdict(False), a1_mean_ssim=0.0)

    assert selection.numbers["probe_relative"] is None
    json.loads(verdicts_to_json({"selection": selection}), parse_constant=_reject_constant)


def test_numbers_are_json_friendly():
    names = _names(6)
    tables = _g1_tables(names)

    numbers = gate_g1(tables["method"], tables["control"], names, _GLOBAL).numbers

    for key, value in numbers.items():
        assert isinstance(value, (int, float, list, tuple, str, dict)), (key, type(value))
        if isinstance(value, float):
            assert math.isfinite(value), key
