"""exp_05 S9 (partial) — restoring the selected checkpoint and clearing the pre-K4 DEV gate.

Plan §4-P3' Eval/F2. Two pieces are exp_05's own and are here; the third -- driving the regressed
adapter through the evaluator's ``pre_context`` mode -- belongs to a file exp_04 has not written yet
(``generate_wan_null_adapter.py``, its R14/R15). exp_05's plan §6 says to stall at that dependency
rather than duplicate the module, so ``regressed_pre_context_eval`` names the round it waits for.

- **Restore is metadata-checked and single-sourced.** ``restore_selected_adapter`` takes the SELECTION
  manager -- the immutable, strictly-improving artifact S7 publishes -- and refuses anything whose
  step, DEV metric or ``l_pos`` disagrees with what the K3 report named. There is no parameter through
  which the resume tree could reach it, so "fall back to the latest checkpoint" is not a behavior this
  code can have: evaluating a checkpoint the stop rule did not select produces a plausible number for
  the wrong model, which is the worst failure available here.
- **The gate is exp_04's, by import.** Plan §4-P3' says H3 ≡ G3 forms, so the +0.05 adapter-vs-null-only
  comparison is literally ``gate_g3_vs_null_only``, with its coverage and imputation rules. The gate is
  defined on DEV and refuses any other cohort -- TEST is not a tuning set and is not touched until it
  passes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L
from maxdiffusion.null_adapter_gates import GateVerdict, NoiseConvention, gate_g3_vs_null_only
from maxdiffusion.trainers.wan_pos_context_regression_trainer import (
    DEV_METRIC,
    read_checkpoint_json,
    restore_adapter_checkpoint,
)

DEV_COHORT = "dev64"
TEST_COHORT = "test64"
REGRESSED_METHOD = "pre_context_regressed"
# Stamped on every certificate the gate issues; a plain mapping cannot forge the pair of it
# and the selection binding without going through the gate.
CERTIFICATE = "exp05.pre_k4_dev_gate.v1"


def selection_manifest(selection_manager) -> dict:
    """What the selection artifact says about itself: step, DEV metric, ``l_pos``, eval history."""
    step = selection_manager.latest_step()
    if step is None:
        raise ValueError(
            "there is no selection artifact in this tree: K3 publishes one at its best DEV normalized "
            "MSE, and an evaluation without it has no defined subject"
        )
    return read_checkpoint_json(selection_manager, step)


def restore_selected_adapter(
    selection_manager,
    template: Any,
    *,
    expected_step: int,
    expected_dev_metric: float,
    expected_l_pos: int = POS_L,
) -> tuple[Any, dict]:
    """The adapter K4 evaluates: the selection artifact, checked against what the K3 report named.

    ``template`` supplies the parameter/optimizer tree structure to restore into, exactly as the
    trainer's own restore does.
    """
    manifest = selection_manifest(selection_manager)
    if int(manifest.get("step", -1)) != int(expected_step):
        raise ValueError(
            f"the selection artifact is at step {manifest.get('step')!r} but the report named "
            f"{expected_step}: the evaluation would measure a checkpoint the run did not select"
        )
    l_pos = manifest.get("l_pos")
    if l_pos is None or int(l_pos) != int(expected_l_pos):
        raise ValueError(
            f"the selection artifact declares l_pos={l_pos!r}, not the deployed {expected_l_pos}: the "
            f"head's output IS the conditioning, so another row count cannot drive the deployed forward"
        )
    # S9 review, BLOCKER 1: NOT optional. Two runs can select the same step; the DEV metric is what
    # tells their artifacts apart, so a restore that skipped it verified almost nothing.
    if not math.isfinite(float(expected_dev_metric)):
        raise ValueError(
            f"the reported {DEV_METRIC} must be finite, got {expected_dev_metric!r}: a nonfinite "
            f"expectation cannot be compared with anything"
        )
    stored = manifest.get(DEV_METRIC)
    if stored is None:
        raise ValueError(
            f"the selection artifact carries no {DEV_METRIC}: it cannot be matched against the report "
            f"and therefore cannot be evaluated"
        )
    if not math.isfinite(float(stored)):
        raise ValueError(f"the selection artifact's {DEV_METRIC} is {stored!r}, which is not finite")
    if not math.isclose(float(stored), float(expected_dev_metric), rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(
            f"the selection artifact's {DEV_METRIC} is {stored!r}, not the reported "
            f"{expected_dev_metric!r}: the artifact and the report disagree about which run this is"
        )
    state, _ = restore_adapter_checkpoint(selection_manager, template)
    return state, manifest


def closed_loop_shift(replay: Mapping, adapter: Mapping, manifest: Sequence[str], *, convention=NoiseConvention.KEYED):
    """Serialized-target replay minus the adapter, in exp_04's G3 form -- a measurement, not a gate.

    Plan §4-P3' calls this gap the closed-loop-shift measurement: how much of the cached targets'
    quality survives being predicted rather than replayed.
    """
    return gate_g3_vs_null_only(replay, adapter, manifest, convention)


IDENTITY_FIELDS = ("step", DEV_METRIC, "l_pos")


def _identity(selection: Mapping) -> dict:
    """The three fields that name a checkpoint: which step, of which run, at which row count."""
    missing = [field for field in IDENTITY_FIELDS if selection.get(field) is None]
    if missing:
        raise ValueError(f"a gate certificate must be bound to a verified selection manifest; missing {missing}")
    return {
        "step": int(selection["step"]),
        DEV_METRIC: float(selection[DEV_METRIC]),
        "l_pos": int(selection["l_pos"]),
    }


def pre_k4_dev_gate(
    adapter: Mapping,
    null_only: Mapping,
    replay: Mapping,
    manifest: Sequence[str],
    *,
    cohort: str,
    selection: Mapping,
    convention: NoiseConvention = NoiseConvention.KEYED,
) -> dict:
    """Plan §4-P3' F2: before TEST is touched, the selected checkpoint must beat null-only on DEV.

    The decision is exp_04's ``gate_g3_vs_null_only`` (+0.05 mean SSIM, CI excluding zero) on
    noise-matched tables, with coverage and imputation inherited. The replay comparison rides along as
    the closed-loop-shift measurement and decides nothing.
    """
    if cohort != DEV_COHORT:
        raise ValueError(
            f"the pre-K4 gate is defined on {DEV_COHORT}, got {cohort!r}: TEST is never a tuning set, "
            f"and no split but DEV may be measured before the gate passes"
        )
    verdict: GateVerdict = gate_g3_vs_null_only(adapter, null_only, manifest, convention)
    return {
        "certificate": CERTIFICATE,
        "selection": _identity(selection),
        "passed": bool(verdict.passed),
        "gate": verdict,
        "closed_loop_shift": closed_loop_shift(replay, adapter, manifest, convention=convention),
        "cohort": cohort,
        "convention": convention.value,
    }


def k4_comparison_row(manifest: Mapping, gate: Mapping, *, cohort: str) -> dict:
    """One publishable row for K4's comparison table, bound to the artifact it measured.

    ``gate`` must be a certificate ``pre_k4_dev_gate`` issued (S9 review, BLOCKER 2): a mapping that
    merely says ``passed`` is not evidence, and neither is a real certificate for another checkpoint.
    The row may only name the artifact the gate actually measured.
    """
    if not isinstance(gate, Mapping) or gate.get("certificate") != CERTIFICATE or gate.get("cohort") != DEV_COHORT:
        raise ValueError(
            f"a K4 row needs the certificate pre_k4_dev_gate issues on {DEV_COHORT}, not an arbitrary "
            f"mapping: {sorted(gate)!r}"
            if isinstance(gate, Mapping)
            else "a K4 row needs a gate certificate"
        )
    if _identity(gate.get("selection") or {}) != _identity(manifest):
        raise ValueError(
            f"this certificate was issued for a different checkpoint "
            f"({gate['selection']} vs {_identity(manifest)}): the row must name the artifact the gate passed"
        )
    if cohort != DEV_COHORT and not gate.get("passed"):
        raise ValueError(
            f"the pre-K4 DEV gate did not pass, so {cohort!r} may not be reported: proceeding to TEST "
            f"is exactly what the gate decides"
        )
    return {
        "method": REGRESSED_METHOD,
        "cohort": cohort,
        "selected_step": int(manifest["step"]),
        "dev_normalized_mse": float(manifest[DEV_METRIC]),
        "l_pos": int(manifest["l_pos"]),
        "gate_passed": bool(gate.get("passed")),
        "gate_convention": str(gate.get("convention", "")),
        "mean_ssim_delta_vs_null_only": float(gate["gate"].numbers.get("mean_delta", float("nan"))),
        "closed_loop_shift": float(gate["closed_loop_shift"].numbers.get("mean_delta", float("nan"))),
    }


def regressed_pre_context_eval(selection_manager, config):  # pragma: no cover -- the stall boundary
    """Drive the restored adapter through the evaluator's ``pre_context`` mode.

    Not implemented here on purpose: the evaluator is ``generate_wan_null_adapter.py``, exp_04's R14
    (``evaluator-restore``) and R15 (``evaluator-modes-noise``) deliverable, and it does not exist in
    any ref yet. exp_05's plan §6 dependency matrix routes this through merge-2 at the R15 boundary and
    says exp_05 stalls there rather than duplicating the file.
    """
    raise NotImplementedError(
        "the regressed pre_context evaluation runs inside generate_wan_null_adapter.py, which is "
        "exp_04's R14/R15 deliverable and does not exist yet; exp_05 stalls at the §6 dependency "
        "matrix (merge-2 at the R15 boundary) rather than authoring exp_04's evaluator"
    )
