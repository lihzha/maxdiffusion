"""Gate statistics for exp_04 — the verdicts are code, not prose (plan §3, N4/M7).

Host-only, numpy-only, deterministic: metric tables in, gate verdicts out. Nothing here prints or
touches jax, so a verdict can be recomputed from the stored JSON months later and come out identical.

**The tables.** ``table[name][k] -> {"future_mse": float, "future_ssim": float}``, one per method,
plus the manifest the evaluation was defined over. Seed keys may be ints or the strings JSON returns.

**The estimand is not the caller's to choose.** Public gates take a ``NoiseConvention`` and derive
the k-set from it (``keyed -> {0,1,2}``, ``global -> {0}``, plan §3). A gate evaluated at a subset of
the declared seeds is a different experiment, and the difference is not visually obvious: the same
probe can pass at ``k={0}`` and fail under the full keyed reduction. The private ``_``-prefixed
entry points take an explicit k-set for tests only.

**Seed reduction.** An example's observation is the mean over the declared k-set of *both* primary
metrics. If any seed is missing, nonfinite, or (where a range is declared) out of range, that
observation is invalid -- fail-closed, because a k-set is a statement of what was run.

**Validity propagates from both metrics of both sides.** An observation is valid only if its MSE and
its SSIM are; a pair is valid only if both observations are. The one asymmetry is the absolute-SSIM
condition, which concerns the method alone: a *measured* method SSIM stays in that mean even when the
control is the invalid side.

**Imputation is claim-penalizing** (M7/P1). Nothing is ever dropped -- dropping is how a broken run
starts looking good. Instead the aggregate moves against the claim under test:

- method-vs-control (G1/G2): an invalid pair contributes ``ratio = 1.0`` to the median and
  ``improved = False``; an invalid *method* observation additionally contributes ``ssim = 0.0``.
- paired difference (G3): an invalid pair on *either* side contributes ``ΔSSIM = -1.0`` and
  ``improved = False``. SSIM is clipped to [0, 1] before differencing, which is what makes -1.0 the
  worst attainable value -- so a missing baseline can never award the adapter an advantage.

Beyond that a gate auto-fails if **more than** 10% of pairs are invalid (strictly greater), and on
any coverage problem -- every manifest name exactly once, no strangers, no duplicates, including
duplicate keys in the raw JSON, which ``parse_table`` rejects rather than silently collapsing.

**Bootstrap.** 10,000 paired resamples over examples, ``np.random.default_rng(20260804)``, percentile
CI at 2.5/97.5. Same table, same interval, always; a golden CI in the tests pins the seed itself.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260804
CI_PERCENTILES = (2.5, 97.5)
MAX_INVALID_FRACTION = 0.10
IMPUTED_RATIO = 1.0  # what an invalid pair contributes to the median
IMPUTED_SSIM = 0.0  # what an invalid method observation contributes to the absolute-SSIM mean
IMPUTED_DELTA = -1.0  # what an invalid pair contributes to the paired difference
SSIM_RANGE = (0.0, 1.0)


class NoiseConvention(enum.Enum):
    """Plan §3's two deployment conventions, each with its own fixed k-set."""

    KEYED = "keyed"
    GLOBAL = "global"


class Target(enum.Enum):
    """Deployment target chosen by the plan's §4-P1 selection rule."""

    A1_KEYED = "A1/keyed"
    A2_GLOBAL = "A2/global"
    STOP = "stop"


KEYED_K_SET = (0, 1, 2)
GLOBAL_K_SET = (0,)
_K_SETS = {NoiseConvention.KEYED: KEYED_K_SET, NoiseConvention.GLOBAL: GLOBAL_K_SET}


@dataclasses.dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reasons: tuple[str, ...]
    numbers: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class TargetSelection:
    target: Target
    reasons: tuple[str, ...]
    numbers: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class _Observation:
    """One example's reduction over the declared k-set."""

    mse: float | None
    ssim: float | None

    @property
    def valid(self) -> bool:
        return self.mse is not None and self.ssim is not None


def k_set_for(convention: NoiseConvention | str) -> tuple[int, ...]:
    """The fixed k-set of a convention; the caller declares the convention, never the seeds."""
    if isinstance(convention, str):
        try:
            convention = NoiseConvention(convention)
        except ValueError as error:
            raise ValueError(f"unknown noise_convention {convention!r}") from error
    if not isinstance(convention, NoiseConvention):
        raise ValueError(f"noise_convention must be a NoiseConvention, got {type(convention)}")
    return _K_SETS[convention]


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate JSON keys would silently collapse: {duplicates}")
    return dict(pairs)


def parse_table(text: str) -> dict:
    """Parse a metric table, refusing duplicate keys instead of letting the last one win."""
    table = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(table, Mapping) or not all(isinstance(value, Mapping) for value in table.values()):
        raise ValueError("not a metric table: expected name -> seed -> metrics")
    return dict(table)


def load_table(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return parse_table(handle.read())


def _metric(entry: Mapping, metric: str, value_range: tuple[float, float] | None) -> float | None:
    value = entry.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    if value_range is not None and not value_range[0] <= value <= value_range[1]:
        return None
    return float(value)


def _observe(
    table: Mapping, name: str, k_set: Sequence[int], *, ssim_range: tuple[float, float] | None = None
) -> _Observation:
    """Reduce one example over the declared k-set; any bad seed invalidates the whole observation."""
    entry = table.get(name)
    if not isinstance(entry, Mapping):
        return _Observation(None, None)
    mses: list[float] | None = []
    ssims: list[float] | None = []
    for k in k_set:
        observation = entry.get(str(k), entry.get(k))
        if not isinstance(observation, Mapping):
            return _Observation(None, None)
        mse = _metric(observation, "future_mse", None)
        ssim = _metric(observation, "future_ssim", ssim_range)
        mses = None if mse is None or mses is None else [*mses, mse]
        ssims = None if ssim is None or ssims is None else [*ssims, ssim]
    return _Observation(
        None if mses is None else float(np.mean(mses)),
        None if ssims is None else float(np.mean(ssims)),
    )


def _coverage(manifest: Sequence[str], **tables: Mapping) -> dict[str, Any]:
    duplicates = sorted({name for name in manifest if list(manifest).count(name) > 1})
    missing = {label: sorted(set(manifest) - set(table)) for label, table in tables.items()}
    extra = {label: sorted(set(table) - set(manifest)) for label, table in tables.items()}
    ok = not duplicates and not any(missing.values()) and not any(extra.values())
    return {
        "coverage_ok": ok,
        "duplicate_names": duplicates,
        "missing_names": missing,
        "extra_names": extra,
        "examples": len(manifest),
    }


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    low, high = np.percentile(values[draws].mean(axis=1), CI_PERCENTILES)
    return float(low), float(high)


def _clip(value: float) -> float:
    return min(max(value, SSIM_RANGE[0]), SSIM_RANGE[1])


def _invalidity(invalid: int, total: int) -> tuple[float, str | None]:
    fraction = invalid / total if total else 0.0
    return fraction, "invalid_fraction" if fraction > MAX_INVALID_FRACTION else None


def _ratio_gate(
    method: Mapping,
    control: Mapping,
    manifest: Sequence[str],
    k_set: Sequence[int],
    *,
    min_median_ratio: float,
    min_fraction_improved: float,
    min_mean_ssim: float,
    min_ssim_ci_low: float,
) -> GateVerdict:
    """G1/G2 shape: method vs control on future MSE, plus an absolute SSIM floor on the method."""
    coverage = _coverage(manifest, method=method, control=control)
    if not coverage["coverage_ok"]:
        return GateVerdict(False, ("coverage",), coverage)

    ratios: list[float | None] = []
    improved, ssims, invalid, imputed_ssim = [], [], 0, 0
    for name in manifest:
        observed = _observe(method, name, k_set)
        reference = _observe(control, name, k_set)
        method_valid = observed.valid and observed.mse > 0.0
        pair_valid = method_valid and reference.valid
        ratio = reference.mse / observed.mse if pair_valid else None
        ratios.append(ratio if ratio is not None and math.isfinite(ratio) else None)
        improved.append(bool(pair_valid and observed.mse < reference.mse))
        if observed.valid:  # a control-only failure never costs the method its own measurement
            ssims.append(observed.ssim)
        else:
            ssims.append(IMPUTED_SSIM)
            imputed_ssim += 1
        invalid += int(not pair_valid)

    ssim_values = np.asarray(ssims, dtype=float)
    ci_low, ci_high = _bootstrap_ci(ssim_values)
    median_ratio = float(np.median([IMPUTED_RATIO if r is None else r for r in ratios]))
    fraction_improved = float(np.mean(improved))
    mean_ssim = float(ssim_values.mean())
    invalid_fraction, invalidity = _invalidity(invalid, len(manifest))

    reasons = []
    if median_ratio < min_median_ratio:
        reasons.append("median_ratio")
    if fraction_improved < min_fraction_improved:
        reasons.append("fraction_improved")
    if mean_ssim < min_mean_ssim:
        reasons.append("mean_ssim")
    if ci_low < min_ssim_ci_low:
        reasons.append("ssim_ci_low")
    if invalidity:
        reasons.append(invalidity)

    numbers = {
        **coverage,
        "median_ratio": median_ratio,
        "per_example_ratio": ratios,  # None where the ratio is undefined; the median sees 1.0
        "imputed_ratio_value": IMPUTED_RATIO,
        "fraction_improved": fraction_improved,
        "mean_ssim": mean_ssim,
        "ci": [ci_low, ci_high],
        "invalid_pairs": invalid,
        "invalid_fraction": invalid_fraction,
        "imputed_method_ssim": imputed_ssim,
        "k_set": list(k_set),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    return GateVerdict(not reasons, tuple(reasons), numbers)


def gate_g1(method: Mapping, control: Mapping, manifest: Sequence[str], convention: NoiseConvention) -> GateVerdict:
    """A1 vs A0 on DEV-64 (plan §4-P1): ratio >= 5, >= 80% improved, SSIM >= 0.80, CI low >= 0.75."""
    return _gate_g1(method, control, manifest, k_set_for(convention))


def _gate_g1(method: Mapping, control: Mapping, manifest: Sequence[str], k_set: Sequence[int]) -> GateVerdict:
    return _ratio_gate(
        method,
        control,
        manifest,
        k_set,
        min_median_ratio=5.0,
        min_fraction_improved=0.80,
        min_mean_ssim=0.80,
        min_ssim_ci_low=0.75,
    )


def gate_g2(method: Mapping, control: Mapping, manifest: Sequence[str], convention: NoiseConvention) -> GateVerdict:
    """A2 vs A2-0 from eps_0 (plan §4-P1): same shape, SSIM >= 0.75 with CI low >= 0.70."""
    return _gate_g2(method, control, manifest, k_set_for(convention))


def _gate_g2(method: Mapping, control: Mapping, manifest: Sequence[str], k_set: Sequence[int]) -> GateVerdict:
    return _ratio_gate(
        method,
        control,
        manifest,
        k_set,
        min_median_ratio=5.0,
        min_fraction_improved=0.80,
        min_mean_ssim=0.75,
        min_ssim_ci_low=0.70,
    )


def _gate_g3(
    method: Mapping,
    baseline: Mapping,
    manifest: Sequence[str],
    k_set: Sequence[int],
    *,
    margin: float,
    min_fraction_improved: float | None,
) -> GateVerdict:
    """Paired SSIM difference on TEST-64 (plan §4-P3), imputing -1.0 against the adapter."""
    coverage = _coverage(manifest, method=method, baseline=baseline)
    if not coverage["coverage_ok"]:
        return GateVerdict(False, ("coverage",), coverage)

    deltas, improved, invalid = [], [], 0
    for name in manifest:
        observed = _observe(method, name, k_set)
        reference = _observe(baseline, name, k_set)
        if not observed.valid or not reference.valid:
            deltas.append(IMPUTED_DELTA)  # the worst value clipping makes attainable
            improved.append(False)
            invalid += 1
            continue
        delta = _clip(observed.ssim) - _clip(reference.ssim)  # clip, then difference
        deltas.append(delta)
        improved.append(delta > 0.0)

    delta_values = np.asarray(deltas, dtype=float)
    ci_low, ci_high = _bootstrap_ci(delta_values)
    mean_delta = float(delta_values.mean())
    fraction_improved = float(np.mean(improved))
    invalid_fraction, invalidity = _invalidity(invalid, len(manifest))

    reasons = []
    if mean_delta < margin:
        reasons.append("mean_delta")
    if not ci_low > 0.0:
        reasons.append("ci_excludes_zero")
    if min_fraction_improved is not None and fraction_improved < min_fraction_improved:
        reasons.append("fraction_improved")
    if invalidity:
        reasons.append(invalidity)

    numbers = {
        **coverage,
        "mean_delta": mean_delta,
        "ci": [ci_low, ci_high],
        "fraction_improved": fraction_improved,
        "per_example_delta": [float(d) for d in deltas],
        "invalid_pairs": invalid,
        "invalid_fraction": invalid_fraction,
        "margin": float(margin),
        "k_set": list(k_set),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    return GateVerdict(not reasons, tuple(reasons), numbers)


def gate_g3_vs_null_only(
    method: Mapping, baseline: Mapping, manifest: Sequence[str], convention: NoiseConvention
) -> GateVerdict:
    """Adapter vs null-only: +0.05 mean SSIM, CI excluding 0; no improved-fraction condition."""
    return _gate_g3(method, baseline, manifest, k_set_for(convention), margin=0.05, min_fraction_improved=None)


def gate_g3_vs_baseline(
    method: Mapping, baseline: Mapping, manifest: Sequence[str], convention: NoiseConvention
) -> GateVerdict:
    """Adapter vs pre_context at matched noise: +0.02, CI excluding 0, and >= 60% improved."""
    return _gate_g3(method, baseline, manifest, k_set_for(convention), margin=0.02, min_fraction_improved=0.60)


def select_target(
    g1: GateVerdict,
    a1_probe_table: Mapping,
    manifest: Sequence[str],
    convention: NoiseConvention,
    g2: GateVerdict,
) -> TargetSelection:
    """The plan's §4-P1 target-selection rule, reducing the A1-probe table under this module's rules."""
    return _select_target(g1, a1_probe_table, manifest, k_set_for(convention), g2)


def _select_target(
    g1: GateVerdict,
    a1_probe_table: Mapping,
    manifest: Sequence[str],
    k_set: Sequence[int],
    g2: GateVerdict,
    *,
    a1_mean_ssim: float | None = None,
) -> TargetSelection:
    """A1 must clear both probe conditions; the probe table goes through the same machinery as a gate.

    ``a1_mean_ssim`` defaults to G1's own method mean, which was computed here under the same
    reduction. The override exists for tests that need an out-of-range A1 mean to reach the relative
    conjunct at all -- see the note on that test.
    """
    coverage = _coverage(manifest, probe=a1_probe_table)
    a1_mean = g1.numbers.get("mean_ssim") if a1_mean_ssim is None else float(a1_mean_ssim)

    probe_values, invalid = [], 0
    for name in manifest:
        observed = _observe(a1_probe_table, name, k_set, ssim_range=SSIM_RANGE)
        if observed.valid:
            probe_values.append(observed.ssim)
        else:
            probe_values.append(IMPUTED_SSIM)
            invalid += 1
    probe_mean = float(np.mean(probe_values)) if probe_values else 0.0
    invalid_fraction, invalidity = _invalidity(invalid, len(manifest))
    numbers = {
        **coverage,
        "probe_mean_ssim": probe_mean,
        "a1_mean_ssim": a1_mean,
        "probe_relative": (probe_mean / a1_mean) if a1_mean else None,  # null rather than infinity
        "probe_invalid_pairs": invalid,
        "probe_invalid_fraction": invalid_fraction,
        "k_set": list(k_set),
    }

    probe_usable = coverage["coverage_ok"] and not invalidity
    relative_ok = probe_usable and a1_mean is not None and probe_mean >= 0.7 * a1_mean
    floor_ok = probe_usable and probe_mean >= 0.70

    if g1.passed and relative_ok and floor_ok:
        return TargetSelection(Target.A1_KEYED, ("G1 passed; A1-probe cleared transfer and floor",), numbers)

    reasons = []
    if not g1.passed:
        reasons.append(f"G1 failed ({', '.join(g1.reasons) or 'unspecified'})")
    if not coverage["coverage_ok"]:
        reasons.append("A1-probe table failed coverage")
    if invalidity:
        reasons.append("A1-probe exceeded the invalidity ceiling")
    if not relative_ok:
        reasons.append("A1-probe below 0.7x the A1 mean (transfer)")
    if not floor_ok:
        reasons.append("A1-probe below the 0.70 absolute floor")
    if g2.passed:
        return TargetSelection(
            Target.A2_GLOBAL, (*reasons, "G2 passed: falling back to the global convention"), numbers
        )
    return TargetSelection(Target.STOP, (*reasons, f"G2 failed ({', '.join(g2.reasons) or 'unspecified'})"), numbers)


def verdicts_to_json(verdicts: Mapping[str, GateVerdict | TargetSelection]) -> str:
    """Strict, deterministic JSON: sorted keys and no NaN/Infinity tokens the standard would reject."""

    def _encode(verdict: Any) -> dict[str, Any]:
        payload = {"reasons": list(verdict.reasons), "numbers": verdict.numbers}
        if isinstance(verdict, GateVerdict):
            return {"passed": verdict.passed, **payload}
        return {"target": verdict.target.value, **payload}

    return json.dumps({name: _encode(v) for name, v in verdicts.items()}, sort_keys=True, allow_nan=False)
