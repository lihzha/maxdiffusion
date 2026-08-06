"""exp_05 S9 (partial) — the selected checkpoint is restorable and gate-able.

**Round scope note.** S9's third piece -- wiring the regressed checkpoint into the evaluator's
``pre_context`` mode -- is blocked: `src/maxdiffusion/generate_wan_null_adapter.py` does not exist in
any ref. It is exp_04's R14/R15 deliverable (exp_04 plan §5 item 10, §6), and exp_04's rounds stop at
R11. exp_05's own plan §6 says to stall at that matrix rather than duplicate the file, so this round
builds the two pieces that are exp_05-owned and dependency-free, and names the boundary where the
third begins.

What is here:

- **Restore, metadata-checked.** K4 evaluates the artifact the K3 report named. A selection tree that
  is empty, or whose step / DEV metric / ``l_pos`` disagree with the report, is refused -- and the
  function is *given* the selection manager, never the resume manager, so "fall back to the latest
  checkpoint" is not an option it can take. Evaluating the wrong checkpoint is the silent disaster
  this round exists to prevent.
- **The pre-K4 DEV gate (plan §4-P3' F2).** exp_04's ``gate_g3_vs_null_only`` verbatim -- the +0.05
  form -- on noise-matched DEV-64 tables, with its coverage and imputation rules inherited by import.
  The gate refuses a TEST cohort outright: TEST is not a tuning set, and it is not touched until the
  gate passes.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L
from maxdiffusion.null_adapter_gates import NoiseConvention
from maxdiffusion.pos_context_eval import (
    DEV_COHORT,
    DEV_METRIC,
    TEST_COHORT,
    closed_loop_shift,
    k4_comparison_row,
    pre_k4_dev_gate,
    restore_selected_adapter,
    selection_manifest,
)
from maxdiffusion.trainers.wan_pos_context_regression_trainer import (
    RegressionTrainState,
    build_selection_manager,
    preserve_selection,
    save_adapter_checkpoint,
)

_NAMES = tuple(f"ep{index}_v0_s{index * 4:05d}" for index in range(8))
_K = (0, 1, 2)


def _params(seed=0):
    return {"w": jnp.asarray(np.random.default_rng(seed).standard_normal((POS_L, 4)), jnp.float32)}


def _state(step, seed=0):
    params = _params(seed)
    return RegressionTrainState(params=params, opt_state=optax.adam(1e-3).init(params), step=step)


def _published(tmp_path, *, step=2000, dev=0.25, l_pos=POS_L, seed=0):
    """A selection artifact as K3 publishes one."""
    manager = build_selection_manager(str(tmp_path / "ckpt"))
    preserve_selection(manager, _state(step, seed), dev_metric=dev, l_pos=l_pos)
    manager.wait_until_finished()
    return manager


def _template():
    params = _params(99)
    return RegressionTrainState(
        params=jax.tree.map(jnp.zeros_like, params), opt_state=optax.adam(1e-3).init(params), step=0
    )


def _table(values, k_set=_K):
    """``name -> (mse, ssim)``; a name mapped to ``None`` is absent from the table."""
    return {
        name: {str(k): {"future_mse": value[0], "future_ssim": value[1]} for k in k_set}
        for name, value in values.items()
        if value is not None
    }


def _selection(step=2000, dev=0.25, l_pos=POS_L):
    """The manifest a restore would have verified -- what a gate certificate is bound to."""
    return {"step": step, "dev_normalized_mse": dev, "l_pos": l_pos}


def _tables(adapter_ssim=0.80, null_ssim=0.70, replay_ssim=0.85):
    values = lambda ssim: dict.fromkeys(_NAMES, (0.1, ssim))  # noqa: E731
    return _table(values(adapter_ssim)), _table(values(null_ssim)), _table(values(replay_ssim))


# --------------------------------------------------------------------------------------------------
# 1. Restore: the artifact the report named, or nothing.
# --------------------------------------------------------------------------------------------------


def test_the_selection_manifest_carries_what_the_report_named(tmp_path):
    manager = _published(tmp_path, step=3000, dev=0.125)

    manifest = selection_manifest(manager)

    assert manifest["step"] == 3000 and manifest["dev_normalized_mse"] == pytest.approx(0.125)
    assert manifest["l_pos"] == POS_L


def test_the_selected_checkpoint_restores_into_the_deployed_parameter_tree(tmp_path):
    manager = _published(tmp_path, step=2000, dev=0.25, seed=5)

    state, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)

    assert state.step == 2000 and manifest["step"] == 2000
    assert np.array_equal(np.asarray(state.params["w"]), np.asarray(_params(5)["w"]))


def test_an_empty_selection_tree_is_refused_rather_than_falling_back(tmp_path):
    """**The silent disaster.** K3 publishes a selection artifact; if it did not, the evaluation has
    no defined subject. Falling back to the newest checkpoint would evaluate a model the stop rule
    explicitly did not select, and the number would look perfectly reasonable."""
    empty = build_selection_manager(str(tmp_path / "ckpt"))

    with pytest.raises(ValueError, match="no selection artifact"):
        restore_selected_adapter(empty, _template(), expected_step=2000, expected_dev_metric=0.25)


@pytest.mark.parametrize("stored, expected", [(2000, 3000), (3000, 2000)])
def test_a_step_that_disagrees_with_the_report_is_refused(tmp_path, stored, expected):
    """Both directions: a checkpoint LATER than the report is as wrong as an earlier one -- it is a
    checkpoint the stop rule did not select either way."""
    manager = _published(tmp_path / f"s{stored}", step=stored)

    with pytest.raises(ValueError, match="step"):
        restore_selected_adapter(manager, _template(), expected_step=expected, expected_dev_metric=0.25)


def test_a_dev_metric_that_disagrees_with_the_report_is_refused(tmp_path):
    manager = _published(tmp_path, step=2000, dev=0.25)

    with pytest.raises(ValueError, match="dev_normalized_mse"):
        restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.9)


def test_a_matching_dev_metric_is_accepted(tmp_path):
    manager = _published(tmp_path, step=2000, dev=0.25)

    state, _ = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)

    assert state.step == 2000


def test_the_dev_metric_check_cannot_be_skipped(tmp_path):
    """**S9 review, BLOCKER 1 -- the reviewer's probe.** With the metric optional, a step+l_pos-only
    call restored an artifact whose DEV number nobody compared with the report's. The metric is the
    only field that distinguishes two checkpoints at the same step of two different runs."""
    import inspect

    manager = _published(tmp_path, step=2000, dev=0.25)
    parameter = inspect.signature(restore_selected_adapter).parameters["expected_dev_metric"]

    assert parameter.default is inspect.Parameter.empty, "the DEV metric check is optional again"
    with pytest.raises(TypeError, match="expected_dev_metric"):
        restore_selected_adapter(manager, _template(), expected_step=2000)


@pytest.mark.parametrize("reported", [float("nan"), float("inf"), -float("inf")])
def test_a_nonfinite_reported_metric_is_refused(tmp_path, reported):
    """``nan != nan``, so a nonfinite expectation would compare unequal forever -- or, worse, be
    silently accepted by a looser comparison. It is refused as unusable."""
    manager = _published(tmp_path, step=2000, dev=0.25)

    with pytest.raises(ValueError, match="finite"):
        restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=reported)


def test_a_nonfinite_stored_metric_is_refused(tmp_path):
    manager = _published(tmp_path, step=2000, dev=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)


def test_an_artifact_without_a_stored_metric_is_refused(tmp_path):
    manager = build_selection_manager(str(tmp_path / "ckpt"))
    save_adapter_checkpoint(manager, _state(2000), l_pos=POS_L)  # no dev_metric written
    manager.wait_until_finished()

    with pytest.raises(ValueError, match=DEV_METRIC):
        restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)


def test_an_l_pos_that_is_not_the_deployed_row_count_is_refused(tmp_path):
    """The head's output IS the conditioning; a checkpoint trained at another width cannot drive the
    deployed forward, and the failure would otherwise appear as a silent shape error mid-rollout."""
    manager = _published(tmp_path, step=2000, l_pos=POS_L + 1)

    with pytest.raises(ValueError, match="l_pos"):
        restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)


def test_an_artifact_without_the_metadata_contract_is_refused(tmp_path):
    """A checkpoint written before this contract cannot be verified, so it cannot be evaluated."""
    manager = build_selection_manager(str(tmp_path / "ckpt"))
    preserve_selection(manager, _state(2000), dev_metric=0.25)  # no l_pos
    manager.wait_until_finished()

    with pytest.raises(ValueError, match="l_pos"):
        restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)


def test_restore_takes_exactly_one_checkpoint_source():
    """**S9 review, MAJOR 3.** Excluding the literal name "manager" was too weak an oracle: any second
    source -- ``fallback_manager``, ``resume_manager``, a ``**kwargs`` catch-all -- would have passed
    it. The whole signature is pinned, names and kinds, so widening it fails here."""
    import inspect

    signature = inspect.signature(restore_selected_adapter)

    assert [(name, parameter.kind.name) for name, parameter in signature.parameters.items()] == [
        ("selection_manager", "POSITIONAL_OR_KEYWORD"),
        ("template", "POSITIONAL_OR_KEYWORD"),
        ("expected_step", "KEYWORD_ONLY"),
        ("expected_dev_metric", "KEYWORD_ONLY"),
        ("expected_l_pos", "KEYWORD_ONLY"),
    ]
    assert not [p for p in signature.parameters.values() if p.kind.name.startswith("VAR_")], "a catch-all crept in"


# --------------------------------------------------------------------------------------------------
# 2. The pre-K4 DEV gate (plan §4-P3' F2).
# --------------------------------------------------------------------------------------------------


def test_the_gate_is_exp_04s_plus_point_oh_five_form_on_dev():
    adapter, null_only, replay = _tables(adapter_ssim=0.80, null_ssim=0.70)

    verdict = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    assert verdict["passed"] is True
    assert verdict["gate"].numbers["margin"] == 0.05
    assert verdict["cohort"] == DEV_COHORT


def test_an_adapter_that_does_not_clear_the_margin_does_not_pass():
    adapter, null_only, replay = _tables(adapter_ssim=0.72, null_ssim=0.70)  # +0.02 only

    verdict = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    assert verdict["passed"] is False


def test_the_gate_refuses_a_test_cohort():
    """TEST is never a tuning set (plan §4-P3', predeclared): the gate is defined on DEV, and it says
    so instead of quietly measuring the split the experiment must not look at yet."""
    adapter, null_only, replay = _tables()

    with pytest.raises(ValueError, match="TEST"):
        pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=TEST_COHORT, selection=_selection())


def test_the_gate_refuses_any_cohort_that_is_not_dev():
    adapter, null_only, replay = _tables()

    with pytest.raises(ValueError, match="dev64"):
        pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort="train2000", selection=_selection())


def test_the_gate_inherits_exp_04s_coverage_and_imputation():
    """A missing example is not silently dropped: exp_04's rules impute the worst value, so a run that
    only measured half the cohort cannot pass by averaging what worked."""
    adapter, null_only, replay = _tables(adapter_ssim=0.90, null_ssim=0.70)
    for name in _NAMES[4:]:
        adapter.pop(name)

    verdict = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    assert verdict["passed"] is False
    assert verdict["gate"].numbers["coverage_ok"] is False
    assert sorted(verdict["gate"].numbers["missing_names"]["method"]) == sorted(_NAMES[4:])
    assert verdict["gate"].reasons == ("coverage",)  # refused on coverage, not scored on the half


def test_the_closed_loop_shift_is_measured_but_does_not_decide():
    """The serialized-replay-vs-adapter gap is K4's shift measurement (plan §4-P3'); only the
    adapter-vs-null-only gate decides whether TEST may be touched."""
    adapter, null_only, replay = _tables(adapter_ssim=0.80, null_ssim=0.70, replay_ssim=0.90)

    verdict = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())
    shift = closed_loop_shift(replay, adapter, _NAMES)

    assert verdict["passed"] is True  # decided by the gate alone
    assert shift.numbers["mean_delta"] == pytest.approx(0.10, abs=1e-6)
    assert verdict["closed_loop_shift"].numbers["mean_delta"] == pytest.approx(0.10, abs=1e-6)


def test_the_gate_is_noise_matched_through_the_convention():
    """Same k-set on both sides: a comparison across conventions is not noise-matched and the tables
    would not even line up."""
    adapter, null_only, replay = _tables()
    single = _table(dict.fromkeys(_NAMES, (0.1, 0.7)), k_set=(0,))

    matched = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())
    global_side = pre_k4_dev_gate(
        _table(dict.fromkeys(_NAMES, (0.1, 0.8)), k_set=(0,)),
        single,
        single,
        _NAMES,
        cohort=DEV_COHORT,
        selection=_selection(),
        convention=NoiseConvention.GLOBAL,
    )

    assert matched["convention"] == "keyed" and global_side["convention"] == "global"
    assert matched["passed"] and global_side["passed"]


# --------------------------------------------------------------------------------------------------
# 3. The K4 comparison row.
# --------------------------------------------------------------------------------------------------


def test_the_comparison_row_names_the_artifact_it_measured(tmp_path):
    manager = _published(tmp_path, step=2000, dev=0.25)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)
    adapter, null_only, replay = _tables()
    verdict = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    row = k4_comparison_row(manifest, verdict, cohort=DEV_COHORT)

    assert row["method"] == "pre_context_regressed" and row["cohort"] == DEV_COHORT
    assert row["selected_step"] == 2000 and row["dev_normalized_mse"] == pytest.approx(0.25)
    assert row["l_pos"] == POS_L and row["gate_passed"] is True
    assert json.loads(json.dumps(row)) == row  # publishable as it stands


def test_a_forged_gate_result_cannot_unlock_a_test_row(tmp_path):
    """**S9 review, BLOCKER 2 -- the reviewer's probe.** ``{"passed": True, "cohort": "test64"}`` used
    to be enough to publish a TEST row. The row now demands the certificate the gate itself returns."""
    manager = _published(tmp_path, step=2000, dev=0.25)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)

    for forged in ({"passed": True, "cohort": "test64"}, {"passed": True, "cohort": DEV_COHORT}, {"passed": True}):
        with pytest.raises(ValueError, match="certificate"):
            k4_comparison_row(manifest, forged, cohort=TEST_COHORT)


def test_a_correctly_bound_forgery_without_the_stamp_is_refused(tmp_path):
    """The hardest forgery: right cohort, right checkpoint binding, no certificate stamp. Only the
    stamp separates a mapping someone assembled from one the gate issued."""
    manager = _published(tmp_path, step=2000, dev=0.25)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)
    binding = {"step": 2000, DEV_METRIC: 0.25, "l_pos": POS_L}

    with pytest.raises(ValueError, match="certificate"):
        k4_comparison_row(manifest, {"passed": True, "cohort": DEV_COHORT, "selection": binding}, cohort=TEST_COHORT)


def test_a_stamped_certificate_claiming_another_cohort_is_refused(tmp_path):
    """And the mirror: the stamp copied onto a mapping that claims it gated TEST. The gate only ever
    issues DEV certificates, so a stamped non-DEV one is by construction hand-made."""
    manager = _published(tmp_path, step=2000, dev=0.25)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)
    adapter, null_only, replay = _tables()
    real = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    with pytest.raises(ValueError, match="certificate"):
        k4_comparison_row(manifest, {**real, "cohort": TEST_COHORT}, cohort=TEST_COHORT)


@pytest.mark.parametrize("dropped", ["step", DEV_METRIC, "l_pos"])
def test_the_gate_refuses_a_selection_that_cannot_identify_a_checkpoint(dropped):
    """A certificate is only as good as its binding: every identity field must be present when it is
    issued, or the artifact it names is ambiguous."""
    adapter, null_only, replay = _tables()
    partial = {key: value for key, value in _selection().items() if key != dropped}

    with pytest.raises(ValueError, match="must be bound"):
        pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=partial)


def test_a_certificate_for_another_checkpoint_cannot_unlock_a_test_row(tmp_path):
    """**S9 review, BLOCKER 2, second probe.** A real, passing certificate -- for a different
    checkpoint. The row must name the artifact the gate actually measured."""
    manager = _published(tmp_path, step=2000, dev=0.25)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)
    adapter, null_only, replay = _tables()
    other = pre_k4_dev_gate(
        adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection(step=9000, dev=0.01)
    )

    assert other["passed"] is True
    with pytest.raises(ValueError, match="different checkpoint"):
        k4_comparison_row(manifest, other, cohort=TEST_COHORT)


@pytest.mark.parametrize("field, value", [("step", 9000), ("dev_normalized_mse", 0.99), ("l_pos", POS_L + 1)])
def test_every_identity_field_binds_the_certificate(tmp_path, field, value):
    manager = _published(tmp_path, step=2000, dev=0.25)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)
    adapter, null_only, replay = _tables()
    certificate = pre_k4_dev_gate(
        adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection={**_selection(), field: value}
    )

    with pytest.raises(ValueError, match="different checkpoint"):
        k4_comparison_row(manifest, certificate, cohort=TEST_COHORT)


def test_a_certificate_from_a_non_dev_cohort_cannot_exist():
    """The certificate carries its cohort, and the gate only issues DEV ones -- so "cohort == dev64"
    is a property of the object, not a promise the caller makes."""
    adapter, null_only, replay = _tables()

    certificate = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    assert certificate["cohort"] == DEV_COHORT
    with pytest.raises(ValueError, match="TEST"):
        pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=TEST_COHORT, selection=_selection())


def test_a_test_row_requires_the_dev_gate_to_have_passed(tmp_path):
    manager = _published(tmp_path, step=2000)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)
    adapter, null_only, replay = _tables(adapter_ssim=0.71, null_ssim=0.70)
    failed = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    with pytest.raises(ValueError, match="did not pass"):
        k4_comparison_row(manifest, failed, cohort=TEST_COHORT)


def test_a_test_row_is_allowed_once_the_gate_has_passed(tmp_path):
    manager = _published(tmp_path, step=2000)
    _, manifest = restore_selected_adapter(manager, _template(), expected_step=2000, expected_dev_metric=0.25)
    adapter, null_only, replay = _tables()
    passed = pre_k4_dev_gate(adapter, null_only, replay, _NAMES, cohort=DEV_COHORT, selection=_selection())

    row = k4_comparison_row(manifest, passed, cohort=TEST_COHORT)

    assert row["cohort"] == TEST_COHORT and row["gate_passed"] is True


# --------------------------------------------------------------------------------------------------
# 4. The boundary this round stops at.
# --------------------------------------------------------------------------------------------------


def test_the_rollout_wiring_names_the_round_that_owns_it():
    """``generate_wan_null_adapter.py`` is exp_04's R14/R15 file and does not exist yet. exp_05's plan
    §6 says to stall at that dependency rather than author exp_04's module, so the entry point says
    which round it is waiting for instead of silently doing something else."""
    from maxdiffusion.pos_context_eval import regressed_pre_context_eval

    with pytest.raises(NotImplementedError, match="R14"):
        regressed_pre_context_eval(None, None)


def test_the_evaluator_file_is_still_exp_04s_to_write():
    """A guard on the stall itself: if this fails, either exp_04 shipped the evaluator (merge, then
    finish S9) or exp_05 started writing it (stop -- that is the duplication the plan forbids)."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[4]
    evaluator = repo_root / "src" / "maxdiffusion" / "generate_wan_null_adapter.py"

    assert repo_root.joinpath("src", "maxdiffusion", "train_wan.py").exists(), "the repo root moved"
    assert not evaluator.exists(), "exp_04's evaluator has landed: merge it and wire the pre_context mode"
