"""exp_06 T7 `fit-probe-mode`: M1 measures the cost, and authorizes only what it measured (§4-P1).

The contract: **M1 measures what a training job would cost, and authorizes only the cells it
measured** — on the program it measured them on.

Review pass 3 rejected the first version of this round on four blockers, and this file is organised
around reproducing each of them before it is fixed:

* **T7-1 — provenance was STAMPED, not BOUND.** The reviewer published an authorization carrying a
  wrong SHA, a foreign model and the wrong device kind, and production accepted it, because
  ``assert_cell_authorized`` was called with no current context at all. Section 3 executes that
  attack against the derived binding.
* **T7-2 — the cell omitted the ARM**, so a matched-C0 measurement authorized R-B despite a
  different forward/backward graph. Section 1 pins the arm into the cell and section 3 shows the
  cross-arm authorization refused.
* **T7-3 — duplicate contradictory trials authorized a refused cell.** Section 2 publishes the
  reviewer's exact pair (one fitting trial, one at 96.9% HBM with a reservation failure) and
  requires the cell refused.
* **T7-4 — ``run_fit_probe`` hid missing ORCHESTRATION**, and the launcher had no probe mode, so M1
  could not be run at all. Section 5 walks the ladder end to end on the host, with only the device
  telemetry adapter stubbed.

Everything about a device is a seam: this suite runs on a laptop and pins the arithmetic, the
verdict rule, the provenance derivation and the authorization contract. What it cannot do is tell
you the numbers, and the module says so rather than shipping a projection it invented.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path

import pytest

from maxdiffusion import pos_rollout_fit_probe as probe

_MODULE_PATH = Path(probe.__file__).resolve()
_CONFIG_PATH = _MODULE_PATH.parent / "configs" / "base_wan_5b_pos_rollout.yml"
_CAPACITY = 32 * 1024**3


class _Device:
    """A device stand-in: the runtime's own report is what the derivation reads."""

    def __init__(self, kind="v6e"):
        self.device_kind = kind


def _config(**overrides):
    import yaml

    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values.update(overrides)

    class _Config:
        def __init__(self, mapping):
            self.__dict__.update(mapping)

        def get_keys(self):
            return dict(self.__dict__)

    return _Config(values)


def _context(config=None, devices=None, **overrides):
    """A DERIVED context. Overrides exist only to build the 'measured elsewhere' counterexamples.

    ``environ={}`` deliberately: this checkout HAS git objects, so the SHA derives from HEAD, and
    handing it a different COMMIT would (correctly) be refused as two different programs.
    """
    derived = probe.derive_probe_context(
        config or _config(), devices=devices or [_Device() for _ in range(8)], environ={}
    )
    return dataclasses.replace(derived, **overrides) if overrides else derived


def _measurement(context=None, *, arm="rollout", microbatch=32, k_b=2, **overrides):
    context = context or _context()
    values = {
        "cell": probe.FitCell(arm=arm, microbatch=microbatch, k_b=k_b),
        "context_digest": context.digest(),
        "compile_seconds": 480.0,
        "step_seconds": 3.5,
        "eval_seconds": 600.0,
        "checkpoint_seconds": 90.0,
        "peak_bytes": 20 * 1024**3,
        "capacity_bytes": _CAPACITY,
        "reservation_failures": 0,
    }
    values.update(overrides)
    return probe.CellMeasurement(**values)


def _publish(tmp_path, measurements, *, context=None, name="authorization.json", steps=10_000):
    context = context or _context()
    evidence = probe.build_evidence(
        context, measurements, max_train_steps=steps, eval_every=1_000, checkpoint_every=1_000
    )
    return probe.publish_authorization(str(tmp_path / name), evidence)


# =============================================================================================
# 1. The cell: an ARM at a microbatch and a horizon (T7-2).
# =============================================================================================


def test_a_cell_is_the_arm_the_microbatch_and_the_horizon():
    """The reviewer's finding: ``(microbatch, k)`` let a C0 measurement authorize R-B. The two arms
    build different forward/backward graphs — one unrolls k sampler steps under remat and
    differentiates through them — so one's peak is not evidence about the other."""
    cell = probe.FitCell(arm="rollout", microbatch=32, k_b=2)
    assert dataclasses.is_dataclass(cell) and cell.__dataclass_params__.frozen
    assert list(inspect.signature(probe.FitCell).parameters) == ["arm", "microbatch", "k_b"]
    assert cell != probe.FitCell(arm="one_step", microbatch=32, k_b=2), "the arm is part of the identity"
    assert hash(cell) == hash(probe.FitCell("rollout", 32, 2))
    with pytest.raises(ValueError, match="unknown arm"):
        probe.FitCell(arm="corrective_ss", microbatch=32, k_b=2)
    for bad in ({"microbatch": 0}, {"k_b": 0}, {"microbatch": -8}):
        with pytest.raises(ValueError, match="positive microbatch and horizon"):
            probe.FitCell(**{"arm": "rollout", "microbatch": 32, "k_b": 2, **bad})


def test_the_declared_ladder_is_arm_by_microbatch_by_k():
    from maxdiffusion import pos_rollout_arms

    assert probe.LADDER_K == (2, 4), "plan §4-P1: k in {2, 4}"
    assert probe.LADDER_MICROBATCH == (8, 16, 32, 64)
    assert probe.LADDER_ARMS == pos_rollout_arms.ARMS, "the ladder measures every declared arm"
    cells = probe.ladder()
    assert len(cells) == 2 * 4 * 2 == 16
    assert probe.FitCell("rollout", 32, 2) in cells and probe.FitCell("one_step", 64, 4) in cells


def test_a_reservation_failure_is_counted_and_refuses_the_cell():
    """Swallowing them is how a cell that only fits when the neighbours are idle gets authorized."""
    verdict = probe.cell_verdict(_measurement(reservation_failures=2))
    assert not verdict.fits and "reservation_failures" in verdict.reasons
    assert verdict.numbers["reservation_failures"] == 2 and verdict.numbers["arm"] == "rollout"


def test_the_headroom_rule_refuses_rather_than_warns():
    """Plan §4-P1: steady-state peak <= 90% of capacity. exp_03's C arm missed at 31.28G/31.25G —
    a 0.1% miss — so a rule that warns and proceeds is a rule that loses a 64-chip reservation."""
    assert probe.HEADROOM_FRACTION == 0.90
    ok = probe.cell_verdict(_measurement(peak_bytes=int(_CAPACITY * 0.899)))
    assert ok.fits and not ok.reasons
    edge = probe.cell_verdict(_measurement(peak_bytes=int(_CAPACITY * 0.901)))
    assert not edge.fits and "headroom" in edge.reasons
    assert edge.numbers["peak_fraction"] == pytest.approx(0.901, abs=1e-3)
    assert not probe.cell_verdict(_measurement(peak_bytes=int(_CAPACITY * 0.99))).fits


def test_a_measurement_must_be_complete_and_finite():
    for override, message in (
        ({"step_seconds": 0.0}, "steady-state step time"),
        ({"step_seconds": float("inf")}, "steady-state step time"),
        ({"compile_seconds": -1.0}, "compile time"),
        ({"eval_seconds": -1.0}, "evaluation overhead"),
        ({"checkpoint_seconds": float("nan")}, "checkpoint overhead"),
        ({"peak_bytes": 0}, "peak"),
        ({"capacity_bytes": 0}, "capacity"),
        ({"reservation_failures": -1}, "reservation failures"),
        ({"context_digest": "not-a-digest"}, "digest of the context"),
    ):
        with pytest.raises(ValueError, match=message):
            probe.cell_verdict(_measurement(**override))


# =============================================================================================
# 2. Aggregation and projection: the two ways a number nobody measured used to travel.
# =============================================================================================


def test_contradictory_trials_of_one_cell_refuse_it(tmp_path):
    """T7-3, the reviewer's executed attack: the same cell published once fitting and once at 96.9%
    HBM with a reservation failure appeared in BOTH lists and was authorized, because publication
    appended trials independently while assertion returned on the first authorized occurrence."""
    context = _context()
    published = _publish(
        tmp_path,
        [
            _measurement(context),
            _measurement(context, peak_bytes=int(_CAPACITY * 0.969), reservation_failures=1),
        ],
        context=context,
    )
    cell = {"arm": "rollout", "microbatch": 32, "k_b": 2}
    assert published["authorized_cells"] == [], "a cell that missed on any trial is a cell that missed"
    assert published["refused_cells"] == [{**cell, "reasons": ["headroom", "reservation_failures"]}]
    assert published["measured_cells"] == [cell], "one aggregated verdict per cell, not one per trial"
    with pytest.raises(ValueError, match="measured that cell and refused it"):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=context)


def test_repeated_trials_aggregate_to_the_worst_of_each():
    context = _context()
    aggregated = probe.aggregate_trials(
        [
            _measurement(context, peak_bytes=10 * 1024**3, step_seconds=3.0, reservation_failures=1),
            _measurement(context, peak_bytes=12 * 1024**3, step_seconds=4.0, reservation_failures=2),
            _measurement(context, arm="one_step", peak_bytes=8 * 1024**3),
        ]
    )
    assert len(aggregated) == 2, "one measurement per cell, in first-seen order"
    worst = aggregated[0]
    assert worst.peak_bytes == 12 * 1024**3 and worst.step_seconds == 4.0
    assert worst.reservation_failures == 3, "reservation failures are totalled, never averaged away"
    assert aggregated[1].cell.arm == "one_step"


def test_trials_measured_under_two_contexts_cannot_be_averaged():
    here, elsewhere = _context(), _context(code_sha="0" * 40)
    with pytest.raises(ValueError, match="different contexts"):
        probe.aggregate_trials([_measurement(here), _measurement(elsewhere)])


def test_the_projection_counts_checkpoints_on_their_own_cadence():
    """T7 MAJOR: the launcher exposes CHECKPOINT_EVERY independently of EVAL_EVERY, and the first
    version projected checkpoints on the evaluation cadence — a run that checkpoints four times as
    often as it evaluates was projected as if it did not."""
    projection = probe.project_wall_clock(
        _measurement(step_seconds=3.0, eval_seconds=600.0, checkpoint_seconds=90.0),
        max_train_steps=10_000,
        eval_every=1_000,
        checkpoint_every=250,
    )
    assert projection["evaluations"] == 10 and projection["checkpoints"] == 40
    assert projection["train_seconds"] == pytest.approx(30_000.0)
    assert projection["eval_seconds_total"] == pytest.approx(6_000.0)
    assert projection["checkpoint_seconds_total"] == pytest.approx(3_600.0)
    assert projection["total_seconds"] == pytest.approx(30_000 + 6_000 + 3_600 + 480)
    assert projection["total_hours"] == pytest.approx(projection["total_seconds"] / 3600.0)


def test_a_projection_cannot_be_built_from_costs_nobody_measured():
    """The reviewer produced a finite 6.55-hour projection from a negative evaluation cost and a
    negative checkpoint cost. The overheads are now MEASUREMENT fields, validated finite and
    non-negative, and there is no argument through which a caller can supply one."""
    parameters = inspect.signature(probe.project_wall_clock).parameters
    assert "eval_seconds" not in parameters and "checkpoint_seconds" not in parameters
    for name in ("max_train_steps", "eval_every", "checkpoint_every"):
        assert parameters[name].default is inspect.Parameter.empty, f"{name} must be given, not defaulted"
    for override, message in (({"eval_seconds": -600.0}, "evaluation"), ({"checkpoint_seconds": -90.0}, "checkpoint")):
        with pytest.raises(ValueError, match=message):
            probe.project_wall_clock(
                _measurement(**override), max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
            )
    for bad in ({"max_train_steps": 0}, {"eval_every": -1}, {"checkpoint_every": 0}, {"eval_every": 2.5}):
        with pytest.raises(ValueError, match="positive whole number"):
            probe.project_wall_clock(
                _measurement(), **{"max_train_steps": 10_000, "eval_every": 1_000, "checkpoint_every": 1_000, **bad}
            )


def test_the_projection_is_refused_for_a_cell_that_does_not_fit():
    with pytest.raises(ValueError, match="does not fit"):
        probe.project_wall_clock(
            _measurement(peak_bytes=int(_CAPACITY * 0.95)),
            max_train_steps=10_000,
            eval_every=1_000,
            checkpoint_every=1_000,
        )


# =============================================================================================
# 3. PROVENANCE: derived from the running program, and BOUND to the measurement (T7-1).
# =============================================================================================


def test_the_context_is_derived_from_the_running_program():
    config = _config()
    context = probe.derive_probe_context(config, devices=[_Device("v6e") for _ in range(8)], environ={})
    assert len(context.code_sha) == 40 and context.code_sha == probe.derive_code_sha(environ={})
    assert context.device_kind == "v6e" and context.device_count == 8
    assert dict(context.geometry)["height"] == 192 and dict(context.geometry)["num_frames"] == 32
    assert context.model_revision.startswith("Wan-AI/Wan2.2-TI2V-5B-Diffusers@")
    assert context.recipe_fingerprint == probe.recipe_fingerprint(config)
    assert len(context.digest()) == 64


def test_a_footprint_key_moves_the_recipe_fingerprint_and_a_cell_key_does_not():
    """The fingerprint is the rest of the footprint-bearing recipe: dtype, remat, attention,
    parallelism, logical batch, sampling geometry, adapter shape. The cell's own three fields are
    deliberately outside it — the ladder varies them and authorizes them per cell."""
    base = probe.recipe_fingerprint(_config())
    for key, value in (
        ("activations_dtype", "float32"),
        ("remat_policy", "MINIMAL"),
        ("pos_logical_batch", 512),
        ("side_adapter_sampling_steps", 40),
        ("ici_fsdp_parallelism", 4),
        ("per_device_batch_size", 2.0),
    ):
        assert probe.recipe_fingerprint(_config(**{key: value})) != base, f"{key} changes the footprint"
    for key, value in (("pos_rollout_arm", "one_step"), ("pos_microbatch", 64), ("pos_rollout_k", 4)):
        assert probe.recipe_fingerprint(_config(**{key: value})) == base, f"{key} is the CELL, not the recipe"


def test_the_code_sha_is_derived_and_a_disagreement_is_fatal():
    """A tarball worker has no git objects and the launcher's COMMIT is the provenance; a checkout
    has git. When both exist and disagree, they are two different programs."""
    assert probe.derive_code_sha(environ={"COMMIT": "c" * 40}, module_file="/nonexistent/x.py") == "c" * 40
    with pytest.raises(ValueError, match="no 40-hex code SHA"):
        probe.derive_code_sha(environ={}, module_file="/nonexistent/x.py")
    with pytest.raises(ValueError, match="no 40-hex code SHA"):
        probe.derive_code_sha(environ={"COMMIT": "not-a-sha"}, module_file="/nonexistent/x.py")
    head = probe.derive_code_sha(environ={})
    with pytest.raises(ValueError, match="two of them"):
        probe.derive_code_sha(environ={"COMMIT": "0" * 40})
    assert probe.derive_code_sha(environ={"COMMIT": head}) == head


def test_an_authorization_measured_on_another_program_does_not_authorize_this_one(tmp_path):
    """T7-1 as the reviewer executed it: an authorization carrying a wrong SHA, a foreign model and
    the wrong device kind used to be accepted, because production passed no context to compare."""
    here = _context()
    for field, value in (
        ("code_sha", "0" * 40),
        ("model_revision", "Some-Other/Model@" + "9" * 40),
        ("device_kind", "v5p"),
        ("device_count", 256),
        ("recipe_fingerprint", "f" * 64),
    ):
        elsewhere = _context(**{field: value})
        published = _publish(tmp_path, [_measurement(elsewhere)], context=elsewhere, name=f"{field}.json")
        with pytest.raises(ValueError, match="measured a different program") as excinfo:
            probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=here)
        assert field in str(excinfo.value), "the refusal names what differs, so it says what to re-measure"
    matching = _publish(tmp_path, [_measurement(here)], context=here, name="ok.json")
    probe.assert_cell_authorized(matching, probe.FitCell("rollout", 32, 2), context=here)


def test_the_current_context_is_required_and_cannot_be_a_hand_written_mapping(tmp_path):
    context = _context()
    published = _publish(tmp_path, [_measurement(context)], context=context)
    with pytest.raises(TypeError):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2))
    forged = {**context.as_payload()}
    with pytest.raises(ValueError, match="derive_probe_context"):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=forged)


def test_publication_takes_no_provenance_arguments_at_all():
    """STAMPED != BOUND, as an API fact: there is no argument through which a claim can arrive."""
    parameters = list(inspect.signature(probe.publish_authorization).parameters)
    assert parameters == ["path", "evidence"]
    for banned in ("code_sha", "model_revision", "device_kind", "device_count", "geometry"):
        assert banned not in parameters
    with pytest.raises(ValueError, match="build_evidence"):
        probe.publish_authorization("unused", {"authorized_cells": [{"arm": "rollout", "microbatch": 32, "k_b": 2}]})


def test_a_measurement_from_another_context_cannot_be_published_beside_a_nicer_claim():
    here, elsewhere = _context(), _context(device_kind="v5p")
    with pytest.raises(ValueError, match="not under"):
        probe.build_evidence(
            here, [_measurement(elsewhere)], max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
        )
    with pytest.raises(ValueError, match="derive_probe_context"):
        probe.build_evidence(
            here.as_payload(), [_measurement(here)], max_train_steps=10_000, eval_every=1_000, checkpoint_every=1_000
        )


# =============================================================================================
# 4. THE AUTHORIZATION ARTIFACT — the point of the round.
# =============================================================================================


def test_the_authorization_names_exactly_the_cells_that_were_measured_and_fit(tmp_path):
    context = _context()
    published = _publish(
        tmp_path,
        [
            _measurement(context, microbatch=16, peak_bytes=12 * 1024**3),
            _measurement(context, microbatch=32, peak_bytes=20 * 1024**3),
            _measurement(context, microbatch=64, peak_bytes=31 * 1024**3),  # 96.9% -- measured, REFUSED
            _measurement(context, arm="one_step", microbatch=64, peak_bytes=18 * 1024**3),
        ],
        context=context,
    )
    assert published["authorized_cells"] == [
        {"arm": "rollout", "microbatch": 16, "k_b": 2},
        {"arm": "rollout", "microbatch": 32, "k_b": 2},
        {"arm": "one_step", "microbatch": 64, "k_b": 2},
    ]
    assert published["refused_cells"] == [{"arm": "rollout", "microbatch": 64, "k_b": 2, "reasons": ["headroom"]}]
    assert len(published["measured_cells"]) == 4
    assert published["context"] == context.as_payload() and published["context_digest"] == context.digest()
    assert published["headroom_fraction"] == probe.HEADROOM_FRACTION
    assert len(published["projections"]) == 3, "only a cell that fits gets a wall-clock"


@pytest.mark.parametrize(
    "cell, allowed",
    [
        (probe.FitCell("rollout", 32, 2), True),
        (probe.FitCell("one_step", 32, 2), False),  # THE T7-2 CASE: the other arm was never measured
        (probe.FitCell("rollout", 64, 2), False),  # measured and refused
        (probe.FitCell("rollout", 32, 4), False),  # never measured
        (probe.FitCell("rollout", 8, 2), False),  # never measured
    ],
)
def test_only_a_measured_and_fitting_cell_of_this_arm_is_authorized(tmp_path, cell, allowed):
    context = _context()
    published = _publish(
        tmp_path,
        [
            _measurement(context, microbatch=32, peak_bytes=20 * 1024**3),
            _measurement(context, microbatch=64, peak_bytes=31 * 1024**3),
        ],
        context=context,
        name=f"auth_{cell.arm}_{cell.microbatch}_{cell.k_b}.json",
    )
    if allowed:
        probe.assert_cell_authorized(published, cell, context=context)
    else:
        with pytest.raises(ValueError, match="M1 did not authorize"):
            probe.assert_cell_authorized(published, cell, context=context)


def test_an_unmeasured_cell_and_a_refused_cell_are_refused_DIFFERENTLY(tmp_path):
    """An operator who asked for a cell nobody measured needs a different instruction from one whose
    cell was measured and missed."""
    context = _context()
    published = _publish(tmp_path, [_measurement(context, microbatch=64, peak_bytes=31 * 1024**3)], context=context)
    with pytest.raises(ValueError, match="measured that cell and refused it"):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 64, 2), context=context)
    with pytest.raises(ValueError, match="never measured"):
        probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 4), context=context)


def test_the_authorization_is_digest_verified_and_published_once(tmp_path):
    path = str(tmp_path / "auth.json")
    context = _context()
    first = _publish(tmp_path, [_measurement(context)], context=context, name="auth.json")
    assert probe.load_authorization(path) == first
    with pytest.raises(ValueError, match="already published"):
        _publish(
            tmp_path, [_measurement(context, microbatch=64, peak_bytes=1024**3)], context=context, name="auth.json"
        )
    stored = json.loads(Path(path).read_text())
    stored["payload"]["authorized_cells"].append({"arm": "rollout", "microbatch": 64, "k_b": 4})
    Path(path).write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="digest"):
        probe.load_authorization(path)


def test_an_identical_republication_is_adopted_rather_than_refused(tmp_path):
    """Issue #10: the queue's auto-retry re-runs a job from the top, and a probe that measured the
    same numbers must adopt its own publication instead of dying on it."""
    context = _context()
    first = _publish(tmp_path, [_measurement(context)], context=context, name="same.json")
    again = _publish(tmp_path, [_measurement(context)], context=context, name="same.json")
    assert again["sha256"] == first["sha256"] and again["authorized_cells"] == first["authorized_cells"]


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda p: p.update(protocol="exp06.fit_authorization.v1"), "is not exp06.fit_authorization.v2"),
        (lambda p: p.pop("context"), "carries none"),
        (lambda p: p.update(context_digest="0" * 64), "does not describe the recorded context"),
        (
            lambda p: p["authorized_cells"].append({"arm": "rollout", "microbatch": 64, "k_b": 2}),
            "unaccounted for",
        ),
        (lambda p: p["authorized_cells"].append(dict(p["authorized_cells"][0])), "same cell twice"),
        (lambda p: p["refused_cells"].append({**p["authorized_cells"][0], "reasons": ["headroom"]}), "BOTH"),
        (lambda p: p.update(measured_cells="everything"), "must be a list"),
        (lambda p: p["measured_cells"].append({"microbatch": 32, "k_b": 2}), "not a cell this probe"),
    ],
)
def test_a_malformed_authorization_is_refused_on_load(tmp_path, damage, message):
    """T7-1's last clause: validate the WHOLE schema and the measured/authorized/refused consistency
    on load. A contradictory artifact would otherwise answer a question it never settled."""
    import hashlib

    context = _context()
    _publish(tmp_path, [_measurement(context)], context=context, name="damaged.json")
    path = tmp_path / "damaged.json"
    stored = json.loads(path.read_text())
    damage(stored["payload"])
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match=message):
        probe.load_authorization(str(path))


def test_a_hand_written_mapping_is_not_an_authorization(tmp_path):
    """exp_05's S9 lesson: a mapping that merely says the right words is not evidence."""
    context = _context()
    for forgery in ({"authorized_cells": [{"arm": "rollout", "microbatch": 32, "k_b": 2}]}, {}, None):
        with pytest.raises(ValueError, match="authorization the fit probe published"):
            probe.assert_cell_authorized(forgery, probe.FitCell("rollout", 32, 2), context=context)


# =============================================================================================
# 5. THE PROBE ITSELF: orchestration on the host, telemetry on the device (T7-4).
# =============================================================================================


def _stub_measurer(peaks=None, *, calls=None):
    def measure(*, cell, context, config):
        if calls is not None:
            calls.append(cell)
        peak = (peaks or {}).get((cell.arm, cell.microbatch, cell.k_b), 20 * 1024**3)
        return probe.CellMeasurement(
            cell=cell,
            context_digest=context.digest(),
            compile_seconds=480.0,
            step_seconds=3.5 * cell.k_b,
            eval_seconds=600.0,
            checkpoint_seconds=90.0,
            peak_bytes=peak,
            capacity_bytes=_CAPACITY,
            reservation_failures=0,
        )

    return measure


def test_the_probe_walks_the_ladder_aggregates_projects_and_publishes(tmp_path):
    """T7-4: the first version did none of this — it walked nothing, invoked no measurement seam,
    aggregated nothing, projected nothing and published nothing, so M1 could not be run at all."""
    path = str(tmp_path / "m1.json")
    config = _config(pos_fit_authorization=path)
    calls = []
    published = probe.run_fit_probe(
        config,
        measurer=_stub_measurer({("rollout", 64, 4): 31 * 1024**3}, calls=calls),
        devices=[_Device() for _ in range(8)],
        trials=2,
    )
    assert len(calls) == 16 * 2, "every declared cell, every trial"
    assert calls[0] == probe.FitCell("rollout", 8, 2) and calls[-1] == probe.FitCell("one_step", 64, 4)
    assert len(published["measured_cells"]) == 16, "trials aggregate to one verdict per cell"
    assert {"arm": "rollout", "microbatch": 64, "k_b": 4, "reasons": ["headroom"]} in published["refused_cells"]
    assert len(published["authorized_cells"]) == 15
    assert len(published["projections"]) == 15
    assert published["context"]["device_kind"] == "v6e" and published["context"]["device_count"] == 8
    assert probe.load_authorization(path)["sha256"] == published["sha256"], "it published where it was told"


def test_the_probe_derives_the_context_and_the_measurer_does_not_get_to_choose_it(tmp_path):
    """``measurer`` is a host stand-in, not a provenance channel: the context is derived here, and a
    measurement that comes back describing another cell or another context is refused."""
    config = _config(pos_fit_authorization=str(tmp_path / "x.json"))
    elsewhere = _context(code_sha="0" * 40)

    def forging(*, cell, context, config):
        return _measurement(elsewhere, arm=cell.arm, microbatch=cell.microbatch, k_b=cell.k_b)

    with pytest.raises(ValueError, match="bound to another context"):
        probe.run_fit_probe(config, measurer=forging, devices=[_Device()], cells=[probe.FitCell("rollout", 32, 2)])

    def wrong_cell(*, cell, context, config):
        return _measurement(context, arm="one_step", microbatch=8, k_b=4)

    with pytest.raises(ValueError, match="describing"):
        probe.run_fit_probe(config, measurer=wrong_cell, devices=[_Device()], cells=[probe.FitCell("rollout", 32, 2)])

    def not_a_measurement(*, cell, context, config):
        return {"peak_bytes": 1}

    with pytest.raises(ValueError, match="not a CellMeasurement"):
        probe.run_fit_probe(
            config, measurer=not_a_measurement, devices=[_Device()], cells=[probe.FitCell("rollout", 32, 2)]
        )


def test_the_probe_refuses_to_publish_nowhere(tmp_path):
    with pytest.raises(ValueError, match="publishes to"):
        probe.run_fit_probe(
            _config(pos_fit_authorization=""),
            measurer=_stub_measurer(),
            devices=[_Device()],
            cells=[probe.FitCell("rollout", 32, 2)],
        )


def test_only_the_per_cell_telemetry_is_a_device_boundary():
    """The pass-2 distinction, applied: naming the boundary is accepted for DEVICE work and does not
    extend to ORCHESTRATION. The one thing left unimplemented is the telemetry adapter."""
    with pytest.raises(NotImplementedError) as excinfo:
        probe.measure_cell_on_device(cell=probe.FitCell("rollout", 32, 2), context=_context(), config=_config())
    message = str(excinfo.value)
    for token in ("TPU", "peak", "STEADY-STATE", "reservation"):
        assert token in message, message
    assert "orchestration around it" in message, "it must say what IS implemented"
    source = inspect.getsource(probe.run_fit_probe)
    for token in ("derive_probe_context", "ladder()", "aggregate", "build_evidence", "publish_authorization"):
        assert token in source or token.rstrip("()") in source, f"run_fit_probe must {token}"


def test_the_entry_point_initializes_pyconfig_and_runs_the_probe(monkeypatch, tmp_path):
    """``main`` is the shell's door into M1: argv[1] is the config, and the probe runs. The real
    ``pyconfig`` drags in ``transformers`` and the whole model stack, so it is stubbed here — the
    property under test is that ``main`` initializes it and hands ``pyconfig.config`` to the probe,
    not that pyconfig works."""
    import sys
    import types

    config = _config(pos_fit_authorization=str(tmp_path / "main.json"))
    seen = {}
    stub = types.ModuleType("maxdiffusion.pyconfig")
    stub.initialize = lambda argv: seen.update(argv=list(argv))
    stub.config = config
    monkeypatch.setitem(sys.modules, "maxdiffusion.pyconfig", stub)
    monkeypatch.setattr(probe, "run_fit_probe", lambda cfg: seen.update(config=cfg) or {"ok": True})
    assert probe.main(["pos_rollout_fit_probe.py", str(_CONFIG_PATH), "pos_microbatch=8"]) == {"ok": True}
    assert seen["argv"][1] == str(_CONFIG_PATH) and seen["config"] is config


# =============================================================================================
# 6. The structural half: a training run cannot reach an unauthorized cell.
# =============================================================================================


def _trainer_config(tmp_path, **overrides):
    values = {
        "pos_recipe_lock": str(tmp_path / "recipe_lock.json"),
        "pos_resume_parent": str(tmp_path / "attempts"),
        "checkpoint_dir": str(tmp_path / "attempts" / "att-NOW" / "checkpoints"),
    }
    values.update(overrides)
    return _config(**values)


def test_the_trainer_refuses_to_start_without_an_authorization(tmp_path):
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    with pytest.raises(ValueError, match="published fit-probe authorization"):
        WanPosRolloutTrainer(_trainer_config(tmp_path, pos_fit_authorization="")).start_training()


def test_the_trainer_derives_its_own_context_and_requires_exact_binding(tmp_path):
    """The trainer no longer calls ``assert_cell_authorized`` with nothing to compare against: it
    derives the running program's context itself, so the comparison is between two independent
    derivations rather than between a claim and silence."""
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    config = _trainer_config(tmp_path, pos_fit_authorization=str(tmp_path / "auth.json"))
    trainer = WanPosRolloutTrainer(config)
    running = trainer.running_context()
    assert running == probe.derive_probe_context(config), "derived twice from the same program, equal"

    _publish(tmp_path, [_measurement(running)], context=running, name="auth.json")
    assert trainer.authorized_cell() == probe.FitCell("rollout", 32, 2)

    foreign = dataclasses.replace(running, code_sha="0" * 40)
    _publish(tmp_path, [_measurement(foreign)], context=foreign, name="foreign.json")
    other = WanPosRolloutTrainer(_trainer_config(tmp_path, pos_fit_authorization=str(tmp_path / "foreign.json")))
    with pytest.raises(ValueError, match="measured a different program"):
        other.authorized_cell()


def test_the_trainer_refuses_a_cell_M1_did_not_authorize(tmp_path):
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer

    path = str(tmp_path / "auth.json")
    config = _trainer_config(tmp_path, pos_fit_authorization=path)
    running = probe.derive_probe_context(config)
    _publish(tmp_path, [_measurement(running, microbatch=32)], context=running, name="auth.json")
    # The authorized cell reaches the (still unwired) model boundary rather than the fit refusal.
    with pytest.raises(NotImplementedError, match="model/data wiring"):
        WanPosRolloutTrainer(_trainer_config(tmp_path, pos_fit_authorization=path, pos_microbatch=32)).start_training()
    for override in ({"pos_microbatch": 64}, {"pos_rollout_k": 4}, {"pos_rollout_arm": "one_step"}):
        with pytest.raises(ValueError, match="M1 did not authorize"):
            WanPosRolloutTrainer(_trainer_config(tmp_path, pos_fit_authorization=path, **override)).start_training()


def test_the_authorization_check_runs_before_anything_expensive():
    """The refusal must come from configuration, not after a pipeline load."""
    from maxdiffusion.trainers import wan_pos_rollout_trainer

    source = Path(wan_pos_rollout_trainer.__file__).read_text(encoding="utf-8")
    node = next(
        item
        for item in ast.walk(ast.parse(source))
        if isinstance(item, ast.FunctionDef) and item.name == "start_training"
    )
    calls = [ast.unparse(sub) for sub in ast.walk(node) if isinstance(sub, ast.Call)]
    raises = [index for index, sub in enumerate(ast.walk(node)) if isinstance(sub, ast.Raise)]
    assert any("authorized_cell" in call for call in calls)
    assert any("assert_paired_recipe" in call for call in calls), "the pair lock is part of starting"
    assert any("resume_source" in call for call in calls), "the resume adoption is part of starting"
    assert raises, "start_training still names its unwired boundary"


# =============================================================================================
# 7. The honesty the module owes, pinned.
# =============================================================================================


def test_the_module_states_what_it_cannot_know():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    flowed = " ".join(source.split())
    assert "2.713" in flowed and "31.28" in flowed, "exp_03's measured points are the honest anchor"
    assert "UNKNOWN" in flowed
    assert "exploratory" in flowed, "k=4 is exploratory only (plan §3/§10)"
    assert "never seen a TPU" in flowed
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"issue #11: {ast.unparse(node)}"
