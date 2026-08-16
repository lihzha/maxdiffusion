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
import functools
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

import pytest

from maxdiffusion import pos_rollout_fit_probe as probe

# ---------------------------------------------------------------------------------------------
# Round F1b: the probe now shares the PRODUCTION optimizer, so `max_utils` must import — and it
# imports four third-party packages this environment does not have. They are stubbed here (only the
# missing leaves; the real `google` namespace package is left alone because orbax needs
# `google.protobuf` through it). None is exp_06 code, none is on the measurement path, and all are
# installed on the worker where M1 actually runs.
# ---------------------------------------------------------------------------------------------
import importlib.machinery


def _stub_leaf(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__file__ = f"<stub {name}>"

    class _Stub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"{name} is stubbed: it is not part of the fit probe")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError(f"{name} is stubbed: it is not part of the fit probe")

    def _getattr(attribute, _stub=_Stub):
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        return _stub

    module.__getattr__ = _getattr
    return module


def _install_import_shims() -> list[str]:
    """Insert the stubs that are actually missing. Returns the names installed."""
    installed = []
    for name in ("transformers", "safetensors"):
        if name not in sys.modules:
            try:
                __import__(name)
            except ImportError:
                sys.modules[name] = _stub_leaf(name)
                installed.append(name)

    if "tensorboardX" not in sys.modules:
        try:
            __import__("tensorboardX")
        except ImportError:
            package = _stub_leaf("tensorboardX")
            writer = _stub_leaf("tensorboardX.writer")

            class SummaryWriter:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("tensorboardX is stubbed")

            writer.SummaryWriter = SummaryWriter
            package.writer = writer
            sys.modules["tensorboardX"] = package
            sys.modules["tensorboardX.writer"] = writer
            installed.append("tensorboardX")

    if "google.cloud.storage" not in sys.modules:
        try:
            from google.cloud import storage  # noqa: F401
        except ImportError:
            try:
                import google as namespace
            except ImportError:
                namespace = types.ModuleType("google")
                namespace.__path__ = []
                sys.modules["google"] = namespace
            cloud = sys.modules.get("google.cloud")
            if cloud is None:
                cloud = types.ModuleType("google.cloud")
                cloud.__path__ = []
                cloud.__spec__ = importlib.machinery.ModuleSpec("google.cloud", None, is_package=True)
                sys.modules["google.cloud"] = cloud
                namespace.cloud = cloud
            storage = _stub_leaf("google.cloud.storage")

            class Client:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("google.cloud.storage is stubbed")

            storage.Client = Client
            sys.modules["google.cloud.storage"] = storage
            cloud.storage = storage
            installed.append("google.cloud.storage")
    return installed


_MODULE_PATH = Path(probe.__file__).resolve()
_CONFIG_PATH = _MODULE_PATH.parent / "configs" / "base_wan_5b_pos_rollout.yml"
_CAPACITY = 32 * 1024**3


class _Device:
    """A device stand-in: the runtime's own report is what the derivation reads."""

    def __init__(self, kind="v6e"):
        self.device_kind = kind


@functools.lru_cache(maxsize=1)
def _local_model_dir() -> str:
    """A model directory with contents, because provenance is now CONTENT-BOUND (review F1, LS-10).

    ``derive_model_revision`` fails closed on a name it cannot resolve to an immutable revision, so a
    test config naming ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` on a laptop without the snapshot correctly
    refuses. Every test config therefore names a real directory, and its manifest identifies it.
    """
    directory = Path(tempfile.mkdtemp(prefix="exp06_probe_model_")) / "snapshot"
    (directory / "transformer").mkdir(parents=True)
    (directory / "transformer" / "weights.safetensors").write_bytes(b"w" * 512)
    (directory / "model_index.json").write_text('{"_class_name": "test"}')
    return str(directory)


def _config(**overrides):
    import yaml

    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values["pretrained_model_name_or_path"] = _local_model_dir()
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
        "context_digest": context.binding_digest(),
        "compile_seconds": 480.0,
        "step_seconds": 3.5,
        "eval_seconds": 600.0,
        "checkpoint_seconds": 90.0,
        "peak_bytes": 20 * 1024**3,
        "capacity_bytes": _CAPACITY,
        "reservation_failures": 0,
        # F10: the shape M1-9 actually measured on v6e, and the shape that authorizes under Yixun's
        # Option A. The compiled analysis IS the reported peak and the bound; the runtime watermark
        # sits far below it (the PJRT mark never sees XLA's temp arena) and is kept as a cross-check.
        "peak_source": probe.PEAK_SOURCE_ANALYSIS,
        "peak_attribution": probe.PEAK_ATTRIBUTION_NONE,
        "watermark_bytes": 4 * 1024**3,
        "watermark_before_bytes": 3 * 1024**3,
    }
    values.update(overrides)
    # The analysis TRACKS the peak unless a case sets it: every `peak_bytes=` case in this file was
    # written to move the number the headroom rule reads, and since F10 that number is the analysis.
    values.setdefault("analysis_bytes", values["peak_bytes"])
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
    # F7: a different LABEL is the same program; a different MANIFEST is the different one.
    here, elsewhere = _context(), _context(manifest_digest="0" * 64)
    with pytest.raises(ValueError, match="different contexts"):
        probe.aggregate_trials([_measurement(here), _measurement(elsewhere)])


def test_the_projection_counts_the_events_production_performs():
    """Review F1, LS-8. The cadence AND the final step, and checkpoints on the evaluation cadence
    because that is where ``pos_rollout_loop.run_loop`` writes them. The earlier version of this test
    asserted an INDEPENDENT checkpoint cadence, which production never honoured."""
    projection = probe.project_wall_clock(
        _measurement(step_seconds=3.0, eval_seconds=600.0, checkpoint_seconds=90.0),
        max_train_steps=10_000,
        eval_every=1_000,
        checkpoint_every=1_000,
    )
    assert projection["evaluations"] == 10 and projection["checkpoints"] == 10
    assert projection["train_seconds"] == pytest.approx(30_000.0)
    assert projection["eval_seconds_total"] == pytest.approx(6_000.0)
    assert projection["checkpoint_seconds_total"] == pytest.approx(900.0)
    assert projection["total_seconds"] == pytest.approx(30_000 + 6_000 + 900 + 480)
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
    assert "@manifest:" in context.model_revision, "a local snapshot is identified by its contents"
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


def test_the_code_sha_is_derived_and_a_disagreement_is_fatal(monkeypatch):
    """A tarball worker has no git objects and the launcher's COMMIT is the LABEL; a checkout has git.
    When both exist and disagree, they are two different programs.

    **Updated by review F5b (BLOCKER 2), and the update is the finding.** A commit is not the running
    bytes, so ``COMMIT`` alone no longer stands behind a measurement: a git-less deployment must bind
    a content manifest (hence the argument here), and a process that DECLARES a commit from a tree
    with uncommitted measurement code is refused. The dirty-tree state is pinned per branch rather
    than read from the working tree, so this test says the same thing before and after a ceremony
    commit."""
    manifest = "c" * 64
    for environ, expected in (({"COMMIT": "c" * 40}, "c" * 40),):
        assert probe.derive_code_sha(environ=environ, module_file="/nonexistent/x.py", manifest=manifest) == expected
    with pytest.raises(ValueError, match="no 40-hex code SHA"):
        probe.derive_code_sha(environ={}, module_file="/nonexistent/x.py", manifest=manifest)
    with pytest.raises(ValueError, match="no 40-hex code SHA"):
        probe.derive_code_sha(environ={"COMMIT": "not-a-sha"}, module_file="/nonexistent/x.py", manifest=manifest)
    # F5b: a git-less deployment that binds no manifest is a claim with nothing behind it.
    with pytest.raises(ValueError, match="not identified at all"):
        probe.derive_code_sha(environ={"COMMIT": "c" * 40}, module_file="/nonexistent/x.py")

    head = probe.derive_code_sha(environ={})
    with pytest.raises(ValueError, match="two of them"):
        probe.derive_code_sha(environ={"COMMIT": "0" * 40})
    monkeypatch.setattr(probe, "_git_dirty_paths", lambda start: ())
    assert probe.derive_code_sha(environ={"COMMIT": head}) == head
    monkeypatch.setattr(probe, "_git_dirty_paths", lambda start: ("src/maxdiffusion/pos_rollout_arms.py",))
    with pytest.raises(ValueError, match="uncommitted"):
        probe.derive_code_sha(environ={"COMMIT": head})


def test_an_authorization_measured_on_another_program_does_not_authorize_this_one(tmp_path):
    """T7-1 as the reviewer executed it: an authorization carrying a foreign model, foreign bytes and
    the wrong device kind used to be accepted, because production passed no context to compare.

    **`code_sha` left this list in F7b, deliberately.** A commit is a LABEL, and refusing on it alone
    blocked a legitimate M2 launch after a docs-only ledger commit — the M1-6 failure, one step later
    and with a reservation held. `manifest_digest` takes its place here, because that is the field
    that actually says "different bytes"; the label case is asserted below as a PASS with a logged
    drift, so the change is recorded rather than merely absent."""
    here = _context()
    for field, value in (
        ("manifest_digest", "0" * 64),
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

    # F7b: the LABEL alone is not a different program, and the gate says so out loud rather than
    # refusing a launch whose bytes it has just verified are identical.
    relabelled = _context(code_sha="5631a36" + "0" * 33)
    published = _publish(tmp_path, [_measurement(relabelled)], context=relabelled, name="label.json")
    probe.assert_cell_authorized(published, probe.FitCell("rollout", 32, 2), context=here)


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
        # The expected text names the CURRENT protocol constant rather than a literal: F5 bumped it to
        # v3 (the payload gained `cell_provenance`), and a test pinning the old spelling of the
        # refusal fails for the bump rather than for the defect it exists to catch.
        (lambda p: p.update(protocol="exp06.fit_authorization.v1"), f"is not {probe.AUTHORIZATION_PROTOCOL}"),
        (lambda p: p.pop("cell_provenance"), "records none"),
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
            context_digest=context.binding_digest(),
            compile_seconds=480.0,
            step_seconds=3.5 * cell.k_b,
            eval_seconds=600.0,
            checkpoint_seconds=90.0,
            peak_bytes=peak,
            capacity_bytes=_CAPACITY,
            reservation_failures=0,
            peak_source=probe.PEAK_SOURCE_ANALYSIS,
            peak_attribution=probe.PEAK_ATTRIBUTION_NONE,
            analysis_bytes=peak,
            watermark_bytes=4 * 1024**3,
            watermark_before_bytes=3 * 1024**3,
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
    # F9b changed the ORDER, and only the order: the runtime watermark is monotone with no reset, so
    # a cell above the headroom floor poisons the standing bound of every cell measured after it.
    # `rollout` mb=8 is the sole such cell and now runs LAST. See `probe.LADDER_ORDER`.
    assert calls[0] == probe.FitCell("one_step", 8, 2) and calls[-1] == probe.FitCell("rollout", 8, 4)
    assert calls == [cell for cell in probe.ladder() for _ in range(2)], "the declared order, verbatim"
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
    elsewhere = _context(manifest_digest="0" * 64)

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


def test_the_orchestration_is_all_present_and_the_seam_is_named():
    """Review F1, LS-6: the measurer no longer raises. What remains device-specific is the telemetry
    source and the pretrained-weights source, and the orchestration names both."""
    source = inspect.getsource(probe.run_fit_probe)
    for token in ("derive_probe_context", "ladder()", "aggregate", "build_evidence", "publish_authorization"):
        assert token in source or token.rstrip("()") in source, f"run_fit_probe must {token}"
    measurement = inspect.getsource(probe.measure_cell_on_device) + inspect.getsource(probe._measure_under_mesh)
    for token in ("DeviceTelemetry", "ProductionModelSource", "build_probe_program", "steady_state", "scope"):
        assert token in measurement, f"the measurer must reach {token}"


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

    foreign = dataclasses.replace(running, manifest_digest="0" * 64)
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
    # The authorized cell PASSES the gate and stops at the next configuration check instead (W2: the
    # boundary is no longer a NotImplementedError -- `start_training` is wired, and this YAML's
    # `global_batch_size_to_load` is the un-derived placeholder, not this run's device arithmetic).
    with pytest.raises(ValueError, match="pos_logical_batch"):
        WanPosRolloutTrainer(_trainer_config(tmp_path, pos_fit_authorization=path, pos_microbatch=32)).start_training()
    for override in ({"pos_microbatch": 64}, {"pos_rollout_k": 4}, {"pos_rollout_arm": "one_step"}):
        with pytest.raises(ValueError, match="M1 did not authorize"):
            WanPosRolloutTrainer(_trainer_config(tmp_path, pos_fit_authorization=path, **override)).start_training()


def test_the_authorization_check_runs_before_anything_expensive():
    """The refusal must come from configuration, not after a pipeline load.

    W2 wired the expensive half, so "the gate is called" stopped being the property that matters:
    a gate called AFTER the pipeline load is a gate that costs a reservation. The assertion is now
    on the ORDER of ``start_training``'s own statements.
    """
    from maxdiffusion.trainers import wan_pos_rollout_trainer

    source = Path(wan_pos_rollout_trainer.__file__).read_text(encoding="utf-8")
    node = next(
        item
        for item in ast.walk(ast.parse(source))
        if isinstance(item, ast.FunctionDef) and item.name == "start_training"
    )
    statements = [ast.unparse(statement) for statement in node.body]
    position = {
        name: next(index for index, text in enumerate(statements) if name in text)
        for name in ("authorized_context", "assert_paired_recipe", "resume_source", "load_backbone")
    }
    assert position["authorized_context"] == 0, "M1's gate is the FIRST executable statement (LOW 4)"
    assert position["authorized_context"] < position["load_backbone"], "M1's gate must precede the 5B load"
    assert position["assert_paired_recipe"] < position["load_backbone"], "the pair lock is part of starting"
    assert position["resume_source"] < position["load_backbone"], "the resume adoption is part of starting"


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


# =============================================================================================
# 8. ROUND F1 — the six M1-critical findings of the launch-surface re-review.
# =============================================================================================


def test_the_fingerprint_covers_the_whole_recipe_and_its_exclusions_are_argued(tmp_path):
    """LS-4. The reviewer measured the old 24-key allowlist going blind on ``action_tokens``,
    ``pre_context_tokens``, ``flash_block_sizes`` and ``latent_frames`` — every one of which feeds
    the adapter construction the probe is measuring. Inclusion is now the default."""
    for key, reason in probe.FINGERPRINT_EXCLUSIONS.items():
        assert reason in probe.FINGERPRINT_EXCLUSION_REASONS, f"{key} is excluded for an unreviewed reason"
    declared = set(probe.config_keys(_config()))
    # `tensorboard_dir` / `metrics_dir` are DERIVED by pyconfig.user_init from run_name + output_dir,
    # so they are absent from the YAML and present at runtime; excluding them is still required.
    derived_only = {"tensorboard_dir", "metrics_dir"}
    assert (
        set(probe.FINGERPRINT_EXCLUSIONS) - declared <= derived_only
    ), f"an exclusion names a key nothing declares: {sorted(set(probe.FINGERPRINT_EXCLUSIONS) - declared - derived_only)}"
    assert set(probe.config_recipe(_config())) == declared - set(probe.FINGERPRINT_EXCLUSIONS)


@pytest.mark.parametrize(
    "key, value",
    [
        ("action_tokens", 64),
        ("action_dim", 14),
        ("action_len", 64),
        ("pre_context_tokens", 64),
        ("pre_context_heads", 64),
        ("action_hidden", 4096),
        ("action_heads", 64),
        ("flash_block_sizes", {"block_q": 1024}),
        ("logical_axis_rules", [["activation_batch", "data"]]),
        ("latent_frames", 99),
        ("latent_channels", 99),
        ("latent_height", 99),
        ("text_dim", 9999),
        ("precision", "HIGHEST"),
        ("remat_policy", "MINIMAL"),
        ("activations_dtype", "float32"),
        ("ici_fsdp_parallelism", 4),
        ("pretrained_model_name_or_path", "Some-Other/Model"),
    ],
)
def test_every_graph_or_hbm_bearing_key_moves_the_fingerprint(key, value):
    """The reviewer's list, verbatim, plus the neighbours that share its failure mode."""
    assert probe.recipe_fingerprint(_config(**{key: value})) != probe.recipe_fingerprint(_config()), key


@pytest.mark.parametrize("key", ["pos_rollout_arm", "pos_microbatch", "pos_rollout_k", "max_train_steps", "run_name"])
def test_the_cell_the_schedule_and_the_destinations_stay_out_of_the_fingerprint(key):
    """An over-bound fingerprint refuses M3 for measuring at M2's length; these are the exclusions."""
    values = {
        "pos_rollout_arm": "one_step",
        "pos_microbatch": 64,
        "pos_rollout_k": 4,
        "max_train_steps": 30_000,
        "run_name": "elsewhere",
    }
    assert probe.recipe_fingerprint(_config(**{key: values[key]})) == probe.recipe_fingerprint(_config()), key


def test_a_nested_container_cannot_hide_a_change_from_the_fingerprint():
    """``flash_block_sizes`` is a dict and ``logical_axis_rules`` a list of lists: rendering them
    canonically is what makes the fingerprint a fingerprint of the WHOLE recipe."""
    left = probe.recipe_fingerprint(_config(flash_block_sizes={"block_q": 512, "block_kv": 1024}))
    right = probe.recipe_fingerprint(_config(flash_block_sizes={"block_kv": 1024, "block_q": 512}))
    assert left == right, "key order is not a change"
    assert left != probe.recipe_fingerprint(_config(flash_block_sizes={"block_q": 512, "block_kv": 2048}))


def test_loading_re_decides_the_authorization_from_its_own_measurements(tmp_path):
    """LS-5, the reviewer's executed attack: it changed an authorized cell's recorded measurement to
    a capacity-level peak plus a reservation failure, recomputed the digest, and BOTH loading and the
    assertion accepted it — because the loader validated three lists of cell names and never looked
    at the numbers underneath them."""
    import hashlib

    context = _context()
    path = tmp_path / "revalidate.json"
    _publish(tmp_path, [_measurement(context)], context=context, name="revalidate.json")

    stored = json.loads(path.read_text())
    stored["payload"]["measurements"][0]["peak_bytes"] = _CAPACITY
    stored["payload"]["measurements"][0]["reservation_failures"] = 3
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="does not reproduce it"):
        probe.load_authorization(str(path))


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda p: p["measurements"][0].update(peak_bytes=_CAPACITY), "does not reproduce it"),
        (lambda p: p["measurements"][0].update(reservation_failures=5), "does not reproduce it"),
        (lambda p: p["measurements"][0].update(step_seconds=0.001), "does not reproduce it"),
        (lambda p: p["projections"][0].update(total_hours=0.5), "does not reproduce it"),
        (lambda p: p["projections"].clear(), "does not reproduce it"),
        (lambda p: p.update(headroom_fraction=0.99), "refuses above 0.9"),
        (lambda p: p.update(measurements=[]), "records none"),
        (lambda p: p.pop("projection_inputs"), "records none"),
        (lambda p: p["measurements"][0].update(context_digest="0" * 64), "measure a different program"),
        (lambda p: p["measurements"].append(dict(p["measurements"][0])), "recorded more than once"),
        (lambda p: p["measurements"][0].pop("eval_seconds"), "not a recorded cell measurement"),
    ],
)
def test_every_recorded_number_is_re_decided_on_load(tmp_path, damage, message):
    import hashlib

    context = _context()
    name = f"dmg_{abs(hash(message)) % 10**6}_{id(damage) % 10**6}.json"
    _publish(tmp_path, [_measurement(context)], context=context, name=name)
    path = tmp_path / name
    stored = json.loads(path.read_text())
    damage(stored["payload"])
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match=message):
        probe.load_authorization(str(path))


def test_the_evaluation_count_is_the_loops_own_predicate():
    """LS-8: ``should_evaluate`` fires on the cadence AND at the final step, so 1,001 steps at
    cadence 1,000 evaluates TWICE. The closed form is checked against production's predicate."""
    from maxdiffusion.pos_rollout_loop import LoopSchedule, should_evaluate

    for steps, cadence in ((1_001, 1_000), (10_000, 1_000), (10_001, 1_000), (999, 1_000), (2_500, 500), (7, 3)):
        schedule = LoopSchedule(
            max_train_steps=steps,
            eval_every=cadence,
            logical_batch=256,
            microbatch=32,
            seed=0,
            arm="rollout",
            k_b=2,
            num_steps=25,
        )
        expected = sum(1 for step in range(1, steps + 1) if should_evaluate(step, schedule))
        assert probe.evaluation_count(steps, cadence) == expected, (steps, cadence)


def test_a_checkpoint_cadence_production_ignores_is_refused_not_projected():
    """LS-8's second half: ``checkpoint_every`` is not consumed by ``LoopSchedule`` at all — the loop
    writes checkpoints inside the evaluation branch. Projecting an independent cadence describes a
    run that cannot be started."""
    from maxdiffusion import pos_rollout_loop

    assert "checkpoint_every" not in inspect.getsource(pos_rollout_loop.LoopSchedule.from_config)
    with pytest.raises(ValueError, match="never reads an independent checkpoint cadence"):
        probe.project_wall_clock(_measurement(), max_train_steps=10_000, eval_every=1_000, checkpoint_every=250)
    projection = probe.project_wall_clock(
        _measurement(step_seconds=3.0), max_train_steps=1_001, eval_every=1_000, checkpoint_every=1_000
    )
    assert projection["evaluations"] == 2 and projection["checkpoints"] == 2


def test_model_provenance_fails_closed_or_is_content_bound(tmp_path):
    """LS-10: ``@local-dir`` identified every local directory alike and ``@no-local-snapshot:<Error>``
    made two unresolved models compare EQUAL — so each could authorize the other's cells."""
    with pytest.raises(ValueError, match="no local snapshot"):
        probe.derive_model_revision(_config(pretrained_model_name_or_path="No-Such-Org/No-Such-Model"))
    body = ast.parse(textwrap.dedent(inspect.getsource(probe.derive_model_revision))).body[0]
    code = "\n".join(
        ast.unparse(node)
        for node in body.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    )
    assert "no-local-snapshot" not in code and "local-dir" not in code, "no fallback string survives in the code"

    left, right = tmp_path / "modelA", tmp_path / "modelB"
    for directory, payload in ((left, b"weights-A"), (right, b"weights-B-and-longer")):
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(payload)
    revisions = [probe.derive_model_revision(_config(pretrained_model_name_or_path=str(d))) for d in (left, right)]
    assert revisions[0] != revisions[1], "two different models must not share an identity"
    assert all("@manifest:" in revision for revision in revisions)
    assert probe.derive_model_revision(_config(pretrained_model_name_or_path=str(left))) == revisions[0]

    (left / "extra.safetensors").write_bytes(b"another shard")
    assert probe.derive_model_revision(_config(pretrained_model_name_or_path=str(left))) != revisions[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        probe.snapshot_manifest_digest(str(empty))


# ---------------------------------------------------------------------------------------------
# LS-6 + LS-7: the REAL entrypoint, through the REAL config parser, with only the device and the
# weights source controlled.
# ---------------------------------------------------------------------------------------------

_TRANSFORMERS_STUB = """\
class _Missing:
    def __init__(self, *a, **k):
        raise RuntimeError("stubbed: the text encoder is not part of the fit probe")

    @classmethod
    def from_pretrained(cls, *a, **k):
        raise RuntimeError("stubbed: the text encoder is not part of the fit probe")


def __getattr__(name):
    return _Missing


__version__ = "stub"
"""
_TENSORBOARDX_STUB = "from . import writer  # noqa: F401\n__version__ = 'stub'\n"
_TENSORBOARDX_WRITER = (
    "class SummaryWriter:\n    def __init__(self, *a, **k):\n        raise RuntimeError('stubbed')\n"
)
_GOOGLE_INIT = "__path__ = __import__('pkgutil').extend_path(__path__, __name__)\n"
_GOOGLE_CLOUD = "from . import storage  # noqa: F401\n"
_GOOGLE_STORAGE = "class Client:\n    def __init__(self, *a, **k):\n        raise RuntimeError('stubbed')\n"
_SAFETENSORS_STUB = "def safe_open(*a, **k):\n    raise RuntimeError('stubbed')\n"

_SITECUSTOMIZE = '''\
"""Replace ONLY the two seams a host cannot provide, before the real entrypoint runs.

A TEST harness, not production configuration: nothing in ``pos_rollout_fit_probe`` reads an
environment variable to decide how to measure. The real ``main``, ``pyconfig.initialize``,
``run_fit_probe``, ``build_probe_program``, compile, steps, scoring pass and checkpoint write all run
untouched.

* ``DeviceTelemetry.peak_and_capacity`` — a CPU backend reports no memory statistics, and production
  correctly REFUSES rather than inventing a peak.
* ``ProductionModelSource`` — the pretrained 5B pipeline needs the snapshot and an accelerator. The
  replacement builds the SAME two module types from the SAME config keys at test dimensions.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import pos_rollout_fit_probe as _probe


def _peak_and_capacity(self):
    return int(os.environ["F1_PEAK_BYTES"]), int(os.environ["F1_CAPACITY_BYTES"])


def _reset_peak(self):
    """This controlled backend HAS a reset facility, so its peak is attributable to the cell."""
    return True


class _TinyModelSource:
    def mesh(self, config):
        return jax.sharding.Mesh(
            np.array(jax.devices()).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor")
        )

    def load(self, config):
        """The WEIGHTS seam at test dimensions — the same record production's loader returns (W3).

        It no longer builds the adapter: the shared finalizer does that, so a host test exercises the
        very construction M1 and the trainer both enter.
        """
        import types as _types

        import jax
        import jax.numpy as jnp
        from flax import nnx
        from flax.linen import partitioning as nn_partitioning

        from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
        from maxdiffusion.pos_rollout_update import LoadedBackbone

        declared_ = _probe.declared
        mesh = self.mesh(config)
        with nn_partitioning.axis_rules(()), mesh:
            transformer = WanModel(
                rngs=nnx.Rngs(jax.random.key(0)),
                num_attention_heads=2,
                attention_head_dim=8,
                in_channels=int(declared_(config, "latent_channels")),
                out_channels=int(declared_(config, "latent_channels")),
                text_dim=int(declared_(config, "text_dim")),
                freq_dim=16,
                ffn_dim=32,
                num_layers=1,
                attention="dot_product",
                rope_max_seq_len=64,
                scan_layers=False,
                dtype=jnp.float32,
                weights_dtype=jnp.float32,
            )
        scheduler = _types.SimpleNamespace(
            config=_types.SimpleNamespace(sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000)
        )
        return LoadedBackbone(
            transformer=transformer,
            mesh=mesh,
            null_context=jnp.full(
                (1, int(declared_(config, "wan_max_sequence_length")), int(declared_(config, "text_dim"))),
                0.25,
                jnp.float32,
            ),
            scheduler=scheduler,
        )


if os.environ.get("F1_PEAK_BYTES"):
    _probe.DeviceTelemetry.peak_and_capacity = _peak_and_capacity
    _probe.DeviceTelemetry.reset_peak = _reset_peak
    _probe.ProductionModelSource = _TinyModelSource
'''

_TINY_OVERRIDES = {
    "latent_channels": 4,
    "latent_frames": 2,
    "latent_height": 4,
    "latent_width": 6,
    "text_dim": 32,
    "wan_max_sequence_length": 8,
    "action_tokens": 4,
    "action_hidden": 16,
    "action_heads": 2,
    "pre_context_tokens": 4,
    "pre_context_heads": 2,
    "side_adapter_layers": "0",
    "side_adapter_hidden": 16,
    "side_adapter_heads": 2,
    "side_adapter_sampling_steps": 4,
    "pos_logical_batch": 8,
    "pos_microbatch": 8,
    "weights_dtype": "float32",
    "activations_dtype": "float32",
    "skip_jax_distributed_system": True,
    "max_train_steps": 100,
    "eval_every": 50,
    "checkpoint_every": 50,
}


def _controlled_backend(tmp_path: Path) -> Path:
    """The stub tree: four absent third-party packages, plus the two-seam sitecustomize."""
    stubs = tmp_path / "stubs"
    (stubs / "tensorboardX").mkdir(parents=True)
    (stubs / "google" / "cloud").mkdir(parents=True)
    (stubs / "transformers").mkdir(parents=True)
    (stubs / "transformers" / "__init__.py").write_text(_TRANSFORMERS_STUB)
    (stubs / "tensorboardX" / "__init__.py").write_text(_TENSORBOARDX_STUB)
    (stubs / "tensorboardX" / "writer.py").write_text(_TENSORBOARDX_WRITER)
    (stubs / "google" / "__init__.py").write_text(_GOOGLE_INIT)
    (stubs / "google" / "cloud" / "__init__.py").write_text(_GOOGLE_CLOUD)
    (stubs / "google" / "cloud" / "storage.py").write_text(_GOOGLE_STORAGE)
    (stubs / "safetensors.py").write_text(_SAFETENSORS_STUB)
    (stubs / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    return stubs


def _tiny_config_file(tmp_path: Path, **overrides) -> Path:
    import yaml

    values = yaml.safe_load(_CONFIG_PATH.read_text())
    values.update(_TINY_OVERRIDES)
    values["pretrained_model_name_or_path"] = _local_model_dir()
    # The checkpoint overhead is measured where production writes it (review F1b, BLOCKER 2), so the
    # probe needs a destination; here it is a local one, and the probe removes its own scratch write.
    values["checkpoint_dir"] = str(tmp_path / "checkpoints")
    values.update(overrides)
    path = tmp_path / "tiny.yml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    return path


def _run_real_entrypoint(tmp_path: Path, *, peak: int, capacity: int, **cli):
    """``python src/maxdiffusion/pos_rollout_fit_probe.py <config.yml> key=value ...`` — for real."""
    repo = _MODULE_PATH.parents[2]
    stubs = _controlled_backend(tmp_path)
    config = _tiny_config_file(tmp_path)
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path / "home"),
        "PYTHONPATH": f"{stubs}:{repo / 'src'}",
        "JAX_PLATFORMS": "cpu",
        "F1_PEAK_BYTES": str(peak),
        "F1_CAPACITY_BYTES": str(capacity),
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    (tmp_path / "home").mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, str(repo / "src" / "maxdiffusion" / "pos_rollout_fit_probe.py"), str(config)]
        + [f"{key}={value}" for key, value in cli.items()],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def test_the_real_m1_entrypoint_measures_and_publishes(tmp_path):
    """LS-6 + LS-7 together, and the reason this round exists.

    The previous version's ``measure_cell_on_device`` raised unconditionally while being the DEFAULT
    measurer of this entrypoint, so the job the launcher exists to start was guaranteed to fail
    before measuring its first cell — and the "real-Python" launcher test passed anyway, because it
    never ran the real config parser or the real entrypoint.

    This runs the actual command the launcher emits: real ``pyconfig.initialize`` over the real YAML
    schema, real ``main``, real program build, real compile, real optimizer steps, real forward-only
    scoring pass, real Orbax checkpoint write, real publication. Only the memory-statistics source
    and the pretrained-weights source are controlled, and both are named in the sitecustomize.

    **F10 note on the controlled watermark.** The peak this backend reports is 1 KiB, BELOW the
    compiled analysis of every cell (this tiny program's analyses are a few hundred KiB), which is
    the shape M1-9 measured on v6e at scale (watermarks 4.2-4.9 GiB under 10-30 GiB analyses). Until
    F10 this test drove the peak to 20 GiB precisely because only a runtime-sourced peak could
    authorize; under Yixun's Option A that same number is a watermark towering over the cell's own
    bound, and the next test is what it now demonstrates.
    """
    authorization = tmp_path / "m1.json"
    proc = _run_real_entrypoint(tmp_path, peak=1024, capacity=_CAPACITY, pos_fit_authorization=str(authorization))
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "[M1] published" in proc.stdout, proc.stdout[-3000:]
    assert authorization.exists(), "M1 published nothing"

    published = probe.load_authorization(str(authorization))
    assert published["protocol"] == "exp06.fit_authorization.v7"
    assert len(published["authorized_cells"]) == 4, published["authorized_cells"]
    assert {entry["arm"] for entry in published["authorized_cells"]} == {"rollout", "one_step"}
    assert {entry["k_b"] for entry in published["authorized_cells"]} == {2, 4}
    for measurement in published["measurements"]:
        assert measurement["step_seconds"] > 0.0, "a measured step time, not a placeholder"
        assert measurement["compile_seconds"] > 0.0
        assert measurement["checkpoint_seconds"] > 0.0, "a real Orbax write was timed"
        assert measurement["eval_seconds"] > 0.0
        # F10: the bound each cell was authorized on is its own compiled analysis, and the runtime
        # reading is published beside it, below it, as the cross-check that did not fire.
        assert measurement["peak_source"] == probe.PEAK_SOURCE_ANALYSIS
        assert measurement["analysis_bytes"] == measurement["peak_bytes"] > 0
        assert measurement["watermark_bytes"] == 1024 < measurement["analysis_bytes"]
    assert published["context"]["device_kind"] == "cpu"
    assert "@manifest:" in published["context"]["model_revision"]
    assert len(published["projections"]) == 4


def test_the_real_entrypoint_refuses_on_the_MARK_and_on_the_HEADROOM_rule(tmp_path):
    """The same real path, driven into each of F10's two refusals by the controlled backend alone.

    Run 1 reports a watermark of 97% of capacity while every cell's compiled analysis is a few
    hundred KiB: the mark contradicts the bound, and every cell is refused as inconsistent. Run 2
    puts the watermark back under the analyses and shrinks the CAPACITY — derived from run 1's
    published analyses rather than hard-coded, so a change in this tiny program's footprint moves
    the threshold with it — until the analyses themselves are over the 90% rule.
    """
    first, second = tmp_path / "mark", tmp_path / "headroom"  # one stub tree per real run
    first.mkdir()
    second.mkdir()
    marked = first / "m1_marked.json"
    proc = _run_real_entrypoint(
        first, peak=int(_CAPACITY * 0.97), capacity=_CAPACITY, pos_fit_authorization=str(marked)
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    published = probe.load_authorization(str(marked))
    assert published["authorized_cells"] == [], "a mark above the bound authorizes nothing"
    assert len(published["refused_cells"]) == 4
    assert all(entry["reasons"] == ["watermark_exceeds_analysis"] for entry in published["refused_cells"])
    assert published["projections"] == [], "a cell that does not fit gets no wall-clock"

    smallest = min(measurement["analysis_bytes"] for measurement in published["measurements"])
    over_floor = second / "m1_headroom.json"
    proc = _run_real_entrypoint(
        second,
        peak=1024,
        capacity=int(smallest / 0.95),  # every analysis is now >= 95% of the device
        pos_fit_authorization=str(over_floor),
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    published = probe.load_authorization(str(over_floor))
    assert published["authorized_cells"] == [], "95% of capacity is a miss"
    assert len(published["refused_cells"]) == 4
    assert all(entry["reasons"] == ["headroom"] for entry in published["refused_cells"])
    assert published["projections"] == []


def test_the_only_device_specific_surface_is_the_telemetry_and_the_weights(tmp_path):
    """What "controlled backend" means, pinned: two seams, both named, everything else real."""
    assert not inspect.isabstract(probe.DeviceTelemetry)
    telemetry = probe.DeviceTelemetry()

    class _NoStats:
        device_kind = "cpu"

        def memory_stats(self):
            return None

    probe_telemetry = probe.DeviceTelemetry()
    probe_telemetry.devices = lambda: [_NoStats()]
    with pytest.raises(ValueError, match="reports no memory statistics"):
        probe_telemetry.peak_and_capacity()

    class _Partial(_NoStats):
        def memory_stats(self):
            return {"peak_bytes_in_use": 1}

    probe_telemetry.devices = lambda: [_Partial()]
    with pytest.raises(ValueError, match="bytes_limit"):
        probe_telemetry.peak_and_capacity()

    class _Full(_NoStats):
        def memory_stats(self):
            return {"peak_bytes_in_use": 7, "bytes_limit": 11}

    probe_telemetry.devices = lambda: [_Full(), _Full()]
    assert probe_telemetry.peak_and_capacity() == (7, 11)
    assert callable(telemetry.devices)


def test_the_measurer_is_no_longer_a_guaranteed_failure():
    """The blocker, as an API fact: nothing on the measurement path raises NotImplementedError."""
    for function in (probe.measure_cell_on_device, probe.build_probe_program, probe.run_fit_probe):
        assert "NotImplementedError" not in inspect.getsource(function), function.__name__
    assert inspect.signature(probe.run_fit_probe).parameters["measurer"].default is probe.measure_cell_on_device


def test_the_script_entry_delegates_to_the_package_module():
    """Run as a script, ``__main__`` holds its own copies of the two seams; the launcher's command
    must reach the package's, or a configured backend silently would not apply."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tail = source.split('if __name__ == "__main__":')[-1]
    assert "from maxdiffusion.pos_rollout_fit_probe import main" in tail
    assert textwrap.dedent(tail).count("_main(sys.argv)") == 1


def test_a_cell_whose_microbatch_cannot_divide_the_logical_batch_is_dropped(tmp_path):
    """Accumulation preserves the logical batch by construction, so such a cell is not a program that
    can run; the probe names it and moves on rather than dying partway through the ladder."""
    config = _config(pos_fit_authorization=str(tmp_path / "drop.json"), pos_logical_batch=16)
    published = probe.run_fit_probe(config, measurer=_stub_measurer(), devices=[_Device()], trials=1)
    assert {entry["microbatch"] for entry in published["measured_cells"]} == {8, 16}
    with pytest.raises(ValueError, match="no declared cell"):
        probe.run_fit_probe(
            _config(pos_fit_authorization=str(tmp_path / "none.json"), pos_logical_batch=3),
            measurer=_stub_measurer(),
            devices=[_Device()],
            trials=1,
        )


# ---------------------------------------------------------------------------------------------
# Battery strengthening: three mutants survived the first F1 run, each naming a property this file
# asserted around rather than at. Fixed, not ratified.
# ---------------------------------------------------------------------------------------------


def test_the_local_manifest_binds_file_SIZES_not_only_names(tmp_path):
    """Battery F16. The first provenance test used directories with different file NAMES, so a
    manifest that recorded every size as zero still told them apart. A truncated or swapped shard
    keeps its name — that is exactly the mutation a content-bound identity has to see."""
    left, right = tmp_path / "same_names_A", tmp_path / "same_names_B"
    for directory, payload in ((left, b"w" * 4096), (right, b"w" * 4095)):
        (directory / "transformer").mkdir(parents=True)
        (directory / "transformer" / "weights.safetensors").write_bytes(payload)
        (directory / "model_index.json").write_text("{}")
    assert sorted(p.name for p in left.rglob("*")) == sorted(p.name for p in right.rglob("*"))
    assert probe.snapshot_manifest_digest(str(left)) != probe.snapshot_manifest_digest(
        str(right)
    ), "one truncated byte in one shard, identical names: the manifest must still separate them"


def test_a_resolved_snapshot_must_be_named_by_an_immutable_commit(tmp_path, monkeypatch):
    """Battery F17. The remote branch was never exercised because every test config names a local
    directory, so the check that a resolved leaf IS a commit was untested."""
    import sys
    import types

    def _hub(resolved):
        module = types.ModuleType("huggingface_hub")
        module.snapshot_download = lambda name, **kwargs: resolved
        return module

    monkeypatch.setitem(sys.modules, "huggingface_hub", _hub(str(tmp_path / "snapshots" / "main")))
    with pytest.raises(ValueError, match="not a snapshot commit"):
        probe.derive_model_revision(_config(pretrained_model_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers"))

    commit = "b" * 40
    monkeypatch.setitem(sys.modules, "huggingface_hub", _hub(str(tmp_path / "snapshots" / commit)))
    revision = probe.derive_model_revision(_config(pretrained_model_name_or_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers"))
    assert revision == f"Wan-AI/Wan2.2-TI2V-5B-Diffusers@{commit}"


def test_the_steady_state_time_excludes_the_warm_up_steps(monkeypatch):
    """Battery F20. The whole point of ``WARMUP_STEPS`` is that the first calls are not steady, and
    nothing asserted that the timed region starts after them — a mutant that timed the warm-up too
    survived. Here the first ``1 + WARMUP_STEPS`` calls are made slow and the rest fast."""
    import time as _time

    calls = {"n": 0}
    slow, fast = 0.05, 0.001

    def _step(params, opt_state, batch, draws):
        calls["n"] += 1
        _time.sleep(slow if calls["n"] <= 1 + probe.WARMUP_STEPS else fast)
        return params, opt_state, 0.0

    def _fake_program(config, cell, *, model_source=None):
        return probe.ProbeProgram(step=_step, score=lambda *a: 0.0, params={}, opt_state={}, batch={}, draws=())

    monkeypatch.setattr(probe, "build_probe_program", _fake_program)
    monkeypatch.setattr(probe, "_program_bytes", lambda program, params, opt_state: None)
    monkeypatch.setattr(probe, "_time_one_checkpoint", lambda config, cell, params, opt_state: 0.01)

    class _Telemetry(probe.DeviceTelemetry):
        def reset_peak(self):
            return True  # a backend WITH a reset facility; the fail-closed path has its own test

        def peak_and_capacity(self):
            return 1024, 4096

    class _Source:
        def mesh(self, config):
            return None

    measurement = probe.measure_cell_on_device(
        cell=probe.FitCell("rollout", 8, 2),
        context=_context(),
        config=_config(),
        telemetry=_Telemetry(),
        model_source=_Source(),
    )
    assert calls["n"] == 1 + probe.WARMUP_STEPS + probe.TIMED_STEPS
    assert (
        measurement.step_seconds < slow / 2
    ), f"step_seconds={measurement.step_seconds} includes the warm-up; the timed region must start after it"
    assert measurement.step_seconds >= fast
    assert measurement.compile_seconds >= slow, "the compile measurement DOES include the first call"


def test_the_probe_program_is_built_from_the_production_stream_seam(tmp_path, monkeypatch):
    """Battery F22 + F23. The program's batch and draws must be what a real optimizer step gets:
    drawn ONCE at the logical-batch width by ``draw_step_for_batch`` and split, not fabricated at the
    microbatch width and not zeroed. Both mutations survived a suite that only ever looked at timings.
    """
    import jax.numpy as jnp

    _install_import_shims()
    from maxdiffusion.pos_rollout_stream import draw_step_for_batch

    logical, microbatch = 16, 8
    built = {}

    class _Source:
        def mesh(self, config):
            return None

        def load(self, config):
            from flax import nnx

            import jax

            from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

            transformer = WanModel(
                rngs=nnx.Rngs(jax.random.key(0)),
                num_attention_heads=2,
                attention_head_dim=8,
                in_channels=4,
                out_channels=4,
                text_dim=32,
                freq_dim=16,
                ffn_dim=32,
                num_layers=1,
                attention="dot_product",
                rope_max_seq_len=64,
                scan_layers=False,
                dtype=jnp.float32,
                weights_dtype=jnp.float32,
            )
            import types as _types

            from maxdiffusion.pos_rollout_update import LoadedBackbone

            built["ok"] = True
            return LoadedBackbone(
                transformer=transformer,
                mesh=None,
                null_context=jnp.full((1, 8, 32), 0.25, jnp.float32),
                scheduler=_types.SimpleNamespace(
                    config=_types.SimpleNamespace(sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000)
                ),
            )

    tiny = _config(
        pos_logical_batch=logical,
        pos_microbatch=microbatch,
        action_len=4,
        action_tokens=4,
        action_hidden=16,
        action_heads=2,
        pre_context_tokens=4,
        pre_context_heads=2,
        side_adapter_layers="0",
        side_adapter_hidden=16,
        side_adapter_heads=2,
        latent_channels=4,
        latent_frames=2,
        latent_height=4,
        latent_width=6,
        text_dim=32,
        wan_max_sequence_length=8,
        side_adapter_sampling_steps=4,
        weights_dtype="float32",
        activations_dtype="float32",
    )
    program = probe.build_probe_program(tiny, probe.FitCell("rollout", microbatch, 2), model_source=_Source())
    assert built["ok"]
    assert isinstance(program.batch, tuple) and len(program.batch) == logical // microbatch
    assert all(part["z_video"].shape[0] == microbatch for part in program.batch), "each part is a MICROBATCH"
    assert program.eval_batch["z_video"].shape[0] == 1, "the DEV unit is batch-ONE"

    # ...and it is the split of a LOGICAL-width draw, computed here independently.
    expected_batch = {
        "z_video": jnp.zeros((logical, 4, 2, 4, 6), jnp.float32),
        "z_i0": jnp.zeros((logical, 4, 1, 4, 6), jnp.float32),
        "actions": jnp.zeros(
            (logical, int(probe.declared(tiny, "action_len")), int(probe.declared(tiny, "action_dim"))), jnp.float32
        ),
    }
    _, micro_draws, _ = draw_step_for_batch(
        expected_batch,
        seed=int(probe.declared(tiny, "seed")),
        global_step=1,
        logical_batch=logical,
        microbatch=microbatch,
        num_steps=4,
        k_b=2,
    )
    expected = tuple(
        tuple(getattr(part, field) for field in ("support_start", "support_end", "epsilon", "t_idx"))
        for part in micro_draws
    )
    assert len(program.draws) == len(expected) == logical // microbatch
    for got_part, want_part in zip(program.draws, expected):
        for got, want in zip(got_part, want_part):
            assert jnp.array_equal(jnp.asarray(got), jnp.asarray(want)), "the draws are the stream's, unmodified"
    assert float(jnp.abs(jnp.asarray(program.draws[0][2])).sum()) > 0.0, "epsilon is not zeroed"


# =============================================================================================
# 9. ROUND F1b — the M1-readiness review's six findings.
# =============================================================================================


def test_the_probe_builds_the_adapter_the_experiment_trains():
    """F1b BLOCKER 1, the one that decides whether M1 measures the approved experiment.

    The config is generated from the side-adapter YAML and inherited its `side_adapter` default,
    while the pilot trains the UNCHANGED **pre_context** adapter (~128M) — a different architecture
    with a different footprint. `ProductionModelSource` consumes the value directly, so M1 would have
    measured a model the experiment never runs and authorized cells on it.
    """
    import yaml

    declared = yaml.safe_load(_CONFIG_PATH.read_text())["action_adapter_type"]
    assert declared == "pre_context", "plan §3: the objective is the only manipulated variable"
    from maxdiffusion import pos_rollout_update

    factory = inspect.getsource(pos_rollout_update.build_adapter_stack)
    assert 'getattr(config, "action_adapter_type")' in factory, "the shared factory reads it from the config"
    assert "build_adapter_stack" in inspect.getsource(pos_rollout_update.build_training_program), "and M1 uses it"
    assert probe.recipe_fingerprint(_config(action_adapter_type="side_adapter")) != probe.recipe_fingerprint(
        _config()
    ), "and the fingerprint moves with it, so an authorization cannot cross the two architectures"


def test_the_timed_unit_is_one_logical_optimizer_update(tmp_path):
    """F1b BLOCKER 2. `step_seconds` is multiplied by `max_train_steps`, which counts LOGICAL
    updates — so timing one microbatch understated GBS-256 computation by 4-32x and never made the
    accumulation state resident."""
    from maxdiffusion import pos_rollout_update

    _install_import_shims()
    logical, microbatch = 16, 4
    tiny = _tiny_probe_config(tmp_path, pos_logical_batch=logical, pos_microbatch=microbatch)
    program = probe.build_probe_program(tiny, probe.FitCell("rollout", microbatch, 2), model_source=_TinySource())
    assert len(program.batch) == logical // microbatch == 4, "every microbatch of the logical batch"
    assert len(program.draws) == len(program.batch)

    # ...and it is the SHARED primitive, not a copy living in the probe.
    assert "build_training_program" in inspect.getsource(probe.build_probe_program)
    assert "optax.adamw(" not in inspect.getsource(probe.build_probe_program), "no private optimizer"
    assert "max_utils.create_optimizer" in inspect.getsource(pos_rollout_update.build_optimizer)


def test_the_shared_update_accumulates_every_microbatch_before_one_optimizer_step():
    """The primitive itself: N microbatches in, ONE optimizer update out, gradient = the mean."""
    import jax.numpy as jnp
    import optax

    from maxdiffusion.pos_rollout_update import build_logical_update

    seen = []

    def loss_fn(params, batch, context, *, frozen_state, draws):
        seen.append(batch["x"])
        return (params["w"] * batch["x"]).sum(), {}

    optimizer = optax.sgd(learning_rate=1.0)
    params = {"w": jnp.zeros((3,), jnp.float32)}
    opt_state = optimizer.init(params)
    update = build_logical_update(loss_fn, optimizer, context=None)
    micro = ({"x": jnp.ones((3,), jnp.float32)}, {"x": jnp.ones((3,), jnp.float32) * 3.0})
    nothing = (jnp.int32(0), jnp.int32(2), jnp.zeros((1,), jnp.float32), jnp.zeros((1,), jnp.int32))
    # F3: `frozen_state` is the SECOND POSITIONAL argument of the update and reaches the loss
    # keyword-only; this primitive threads it and never differentiates it.
    new_params, _, loss = update(params, None, opt_state, micro, (nothing, nothing))
    # F4 changed what this line can mean. The accumulation is a `lax.scan`, so the body is TRACED
    # ONCE however many microbatches it runs — that is the entire point of the rewrite (the Python
    # `for` it replaced emitted one 5B forward+backward into the jaxpr per microbatch, and XLA
    # compiling 32 of them killed four TPU hosts). So a trace count of 2 is no longer evidence that
    # both microbatches contributed; it would now be evidence the graph defect was back.
    assert len(seen) == 1, "ONE gradient block must be traced, not one per microbatch (F4)"
    # The contract the old trace count stood for is asserted numerically, and STRICTLY: the gradient
    # of the MEAN loss is the mean of the gradients, (1 + 3) / 2 = 2, so sgd(1.0) steps -2. Dropping
    # either microbatch lands on -1 or -3 instead, so a lost microbatch still fails here.
    assert jnp.allclose(new_params["w"], jnp.full((3,), -2.0)), new_params
    assert float(loss) == pytest.approx(((1 * 0) + (3 * 0)) / 2)


def test_the_evaluation_unit_is_the_dev_instruments_own_batch_one(tmp_path):
    """F1b BLOCKER 2. `DevBatchReader.read` returns ONE example's tensors and the cohort is scored
    example by example, so scaling a microbatch by ceil(64/microbatch) timed a different program."""
    _install_import_shims()
    tiny = _tiny_probe_config(tmp_path, pos_logical_batch=16, pos_microbatch=8)
    program = probe.build_probe_program(tiny, probe.FitCell("rollout", 8, 2), model_source=_TinySource())
    assert program.eval_batch["z_video"].shape[0] == 1
    assert program.eval_batch["actions"].shape[0] == 1
    body = inspect.getsource(probe._measure_under_mesh)
    assert "DEV_COHORT_SIZE // int(cell.microbatch)" not in body, "no microbatch scaling survives"
    assert "* DEV_COHORT_SIZE" in body, "one batch-one pass, times the cohort"


def test_the_checkpoint_unit_is_written_where_production_writes(tmp_path):
    """F1b BLOCKER 2. A local temporary directory measures serialization; production writes the
    loop's payload to a `gs://` tree, and the storage class dominates the cost."""
    source = inspect.getsource(probe._time_one_checkpoint)
    assert "tempfile" not in source, "the probe no longer writes to a private tmpdir"
    assert "save_checkpoint" in source and "build_checkpoint_manager" in source, "the loop's own payload"
    with pytest.raises(ValueError, match="cannot be measured where production writes it"):
        probe._time_one_checkpoint(_config(checkpoint_dir=""), probe.FitCell("rollout", 8, 2), {}, {})


def test_a_peak_this_cell_did_not_set_is_refused_not_reported():
    """F1b BLOCKER 3, the contamination that made 32 sequential cells share one number: nothing reset
    the runtime's LIFETIME high-water mark between model load, compile, warm-up and the trials.

    **F9 revised the middle case, and the revision is a strengthening.** A standing mark bounds this
    cell whether or not this cell set it (``classify_peak`` carries the proof: the mark is monotone,
    so it dominates every instant inside this cell's window, including the cell's own peak). What it
    cannot be is a number smaller than this cell's own compiled analysis — that is not a ceiling —
    and it cannot be reported with nothing cell-local to check it against. So: no analysis at all
    still refuses; a 30-GiB standing mark over a 7-GiB analysis now reports **30 GiB**, the ceiling,
    rather than the 7-GiB floor it used to report. The number went UP and the cell got HARDER to
    authorize; what changed is that it is now judged on a bound instead of refused on provenance.
    """

    class _NoReset(probe.DeviceTelemetry):
        def __init__(self, peak):
            self._peak = peak

        def reset_peak(self):
            return False

        def peak_and_capacity(self):
            return self._peak, 32 * 1024**3

    telemetry = _NoReset(30 * 1024**3)
    before = telemetry.begin_steady_state()
    assert before["peak_before"] == 30 * 1024**3 and before["reset"] is False
    with pytest.raises(ValueError, match="no per-cell steady-state peak could be obtained"):
        telemetry.end_steady_state(before, program_bytes=None)

    # The standing mark DOMINATES this cell's analysis, so it is the reported ceiling...
    peak, capacity, source = telemetry.end_steady_state(before, program_bytes=7 * 1024**3)
    assert (peak, capacity) == (30 * 1024**3, 32 * 1024**3) and source == probe.PEAK_SOURCE_RUNTIME_RAISED
    assert probe.cell_verdict(
        _measurement(
            peak_bytes=peak,
            capacity_bytes=capacity,
            peak_source=source,
            analysis_bytes=7 * 1024**3,
            watermark_bytes=peak,
            peak_attribution=probe.PEAK_ATTRIBUTION_STANDING,
        )
        # F10: 30 GiB of watermark over a 7 GiB bound is the INCONSISTENCY, not the headroom rule --
        # the bound itself is 21.9% of the device. The mark and the analysis disagree about the same
        # cell by a factor of four, and the contract refuses that rather than picking a winner.
    ).reasons == ("watermark_exceeds_analysis",)

    # ...and a standing mark BELOW this cell's own analysis is discarded: the two disagree, so the
    # analysis is what gets reported, and per review W1 A3 an analysis is NOT authorizing.
    peak, _, source = telemetry.end_steady_state(before, program_bytes=31 * 1024**3)
    assert peak == 31 * 1024**3 and source == probe.PEAK_SOURCE_ANALYSIS

    telemetry._peak = 31 * 1024**3
    peak, _, source = telemetry.end_steady_state(before, program_bytes=None)
    assert peak == 31 * 1024**3 and source == probe.PEAK_SOURCE_RUNTIME_RAISED


def test_the_ANALYSIS_is_the_authorization_bound_and_the_refused_capacity_still_is_not():
    """**Review W1/A3, INVERTED by F10 (Yixun's Option A, plan v2.9 §4-P1)** — and the history is
    kept because it is the reason the inversion needed a decision.

    W1/A3 refused to authorize on the analysis: it was a LOWER bound (it could not see the frozen 5B,
    captured as a constant), and a "<= 90% of capacity" rule is a ceiling test. Both halves of that
    premise have since moved. F3 made the backbone an explicit argument of the compiled update, so
    the analysis counts it; and M1-9 measured the pair on v6e, where the analysis ran 2-7x ABOVE the
    allocator's watermark on every one of twelve cells. The conservative number is now the analysis,
    the old rule authorized nothing, and the bound is the analysis with the watermark cross-checking
    it. What is still never an authorization is the capacity a REFUSED allocation hit.
    """
    bound = _measurement(peak_bytes=7 * 1024**3)
    verdict = probe.cell_verdict(bound)
    assert verdict.fits, "an analysis under the floor, with nothing contradicting it, authorizes"
    assert verdict.numbers["peak_source"] == probe.PEAK_SOURCE_ANALYSIS
    assert verdict.numbers["authorized_bytes"] == 7 * 1024**3

    # ...and the same bound ABOVE the headroom rule refuses on the rule.
    high = probe.cell_verdict(_measurement(peak_bytes=int(_CAPACITY * 0.95)))
    assert not high.fits and high.reasons == ("headroom",)

    # The one source that authorizes nothing: a cell that hit a refused allocation reports the
    # capacity, and a capacity is not a footprint.
    missed = probe.cell_verdict(
        _measurement(peak_bytes=_CAPACITY, capacity_bytes=_CAPACITY, peak_source=probe.PEAK_SOURCE_REFUSED)
    )
    assert not missed.fits and "peak_source" in missed.reasons
    for source in probe.AUTHORIZING_PEAK_SOURCES:
        attribution = (
            probe.PEAK_ATTRIBUTION_NONE if source == probe.PEAK_SOURCE_ANALYSIS else probe.PEAK_ATTRIBUTION_RAISED
        )
        assert probe.cell_verdict(_measurement(peak_source=source, peak_attribution=attribution)).fits, source
    assert probe.PEAK_SOURCE_ANALYSIS in probe.AUTHORIZING_PEAK_SOURCES
    assert probe.PEAK_SOURCE_REFUSED not in probe.AUTHORIZING_PEAK_SOURCES


def test_the_peak_source_is_required_recorded_and_re_decided(tmp_path):
    """It has no default: a default would be a claim about provenance that nobody made."""
    import dataclasses as dc

    field = {f.name: f for f in dc.fields(probe.CellMeasurement)}["peak_source"]
    assert field.default is dc.MISSING and field.default_factory is dc.MISSING

    context = _context()
    published = _publish(tmp_path, [_measurement(context)], context=context, name="src.json")
    assert published["measurements"][0]["peak_source"] == probe.PEAK_SOURCE_ANALYSIS
    loaded = probe.load_authorization(str(tmp_path / "src.json"))
    assert loaded["measurements"][0]["peak_source"] == probe.PEAK_SOURCE_ANALYSIS

    # An artifact whose measurements carry NO BOUND authorizes nothing when re-decided (F10): the
    # provenance is recorded either way, and it is the missing analysis that refuses the cell.
    unbounded_context = _context()
    unbounded = _publish(
        tmp_path,
        [
            _measurement(
                unbounded_context,
                analysis_bytes=None,
                peak_source=probe.PEAK_SOURCE_RUNTIME_RESET,
                peak_attribution=probe.PEAK_ATTRIBUTION_RESET,
            )
        ],
        context=unbounded_context,
        name="unbounded.json",
    )
    assert unbounded["authorized_cells"] == []
    assert unbounded["refused_cells"][0]["reasons"] == ["analysis_missing"]
    with pytest.raises(ValueError, match="M1 did not authorize"):
        probe.assert_cell_authorized(unbounded, probe.FitCell("rollout", 32, 2), context=unbounded_context)


def test_an_unrecognised_peak_source_is_refused_on_load():
    with pytest.raises(ValueError, match="not a peak source this probe produces"):
        probe.cell_verdict(_measurement(peak_source="a number I liked"))


def test_a_mixed_cell_is_only_as_good_as_its_weakest_evidence():
    """Trials aggregate conservatively in provenance too: one analysis-labelled trial and the cell
    is labelled with the analysis. **F10: the label no longer refuses the cell** — what refuses a
    cell is its worst NUMBERS, and the aggregate takes the worst of every one of them (here the
    runtime trial's 25 GiB mark against a 20 GiB bound)."""
    context = _context()
    aggregated = probe.aggregate_trials(
        [
            _measurement(
                context,
                peak_source=probe.PEAK_SOURCE_RUNTIME_RESET,
                peak_attribution=probe.PEAK_ATTRIBUTION_RESET,
                peak_bytes=25 * 1024**3,
                analysis_bytes=20 * 1024**3,
                watermark_bytes=25 * 1024**3,
            ),
            _measurement(context, peak_source=probe.PEAK_SOURCE_ANALYSIS),
        ]
    )
    assert aggregated[0].peak_source == probe.PEAK_SOURCE_ANALYSIS
    assert probe.cell_verdict(aggregated[0]).reasons == ("watermark_exceeds_analysis",)


def test_a_backend_that_can_reset_gives_an_attributable_peak():
    class _Resets(probe.DeviceTelemetry):
        def __init__(self):
            self.resets = 0

        def reset_peak(self):
            self.resets += 1
            return True

        def peak_and_capacity(self):
            return 5 * 1024**3, 32 * 1024**3

    telemetry = _Resets()
    before = telemetry.begin_steady_state()
    assert telemetry.resets == 1 and before["peak_before"] == 0
    peak, _, source = telemetry.end_steady_state(before, program_bytes=None)
    assert peak == 5 * 1024**3 and "after reset" in source


def test_the_reset_probe_tries_supported_facilities_and_reports_honestly():
    class _Device:
        device_kind = "v6e"

        def __init__(self):
            self.cleared = 0

        def clear_memory_stats(self):
            self.cleared += 1

    telemetry = probe.DeviceTelemetry()
    devices = [_Device(), _Device()]
    telemetry.devices = lambda: devices
    assert telemetry.reset_peak() is True and [d.cleared for d in devices] == [1, 1]

    class _Bare:
        device_kind = "v6e"

    telemetry.devices = lambda: [_Bare()]
    assert telemetry.reset_peak() is False, "a backend without the facility must say so"


@pytest.mark.parametrize(
    "error, exhausted, why",
    [
        (RuntimeError("boom in program build"), False, "the reviewer's probe: 'boom' contains 'oom'"),
        (OSError("No room left on device"), False, "the reviewer's probe: a FULL DISK is not HBM"),
        (OSError("Input/output error"), False, "an I/O failure is not an allocation refusal"),
        (KeyError("zoom"), False, "a lookup failure is not an allocation refusal"),
        (ValueError("groom the tensors"), False, "nor is a shape error"),
        (RuntimeError("RESOURCE_EXHAUSTED: Attempting to allocate 31.28G"), True, "the real thing"),
        (RuntimeError("Out of memory allocating 4GB on device"), True, "and its other spelling"),
        (RuntimeError("XlaRuntimeError: OOM when allocating"), True, "word-boundary OOM"),
        (MemoryError("cannot allocate"), False, "a bare MemoryError with no allocation phrase"),
    ],
)
def test_only_a_real_allocation_refusal_is_classified_as_one(error, exhausted, why):
    """F1b MAJOR 4. Under the bare `"OOM"` substring a model bug and a full disk both became
    'measured' HBM refusals, and M1 would have exited successfully having authorized nothing it
    understood."""
    assert probe._is_resource_exhausted(error) is exhausted, why


def test_a_structured_status_code_is_preferred_over_the_message():
    class _Structured(RuntimeError):
        status_code = "RESOURCE_EXHAUSTED"

    assert probe._is_resource_exhausted(_Structured("no message about memory at all")) is True


def test_local_model_identity_is_bound_to_the_BYTES(tmp_path):
    """F1b MAJOR 5. `(relpath, size)` is metadata: an in-place byte change, or a swap for another
    checkpoint of the same shape — including a same-length config edit that alters the graph — left
    the identity, and therefore the authorization, standing."""
    directory = tmp_path / "model"
    directory.mkdir()
    shard = directory / "model.safetensors"
    shard.write_bytes(b"A" * 4096)
    before = probe.snapshot_manifest_digest(str(directory))

    shard.write_bytes(b"B" * 4096)  # same path, same size, every byte different
    assert probe.snapshot_manifest_digest(str(directory)) != before, "an in-place byte change must be seen"

    shard.write_bytes(b"A" * 4096)
    assert probe.snapshot_manifest_digest(str(directory)) == before, "and the identity is stable"

    config = directory / "config.json"
    config.write_text('{"num_layers": 30}')
    with_config = probe.snapshot_manifest_digest(str(directory))
    config.write_text('{"num_layers": 40}')  # same length, different graph
    assert probe.snapshot_manifest_digest(str(directory)) != with_config


def test_a_local_model_too_large_to_hash_must_be_named_by_its_commit(tmp_path, monkeypatch):
    """The ceiling, and why it is the honest place to stop: at 5B scale a full re-hash on every probe
    and every training start costs more than the thing it identifies, and the remote branch already
    carries an immutable id."""
    monkeypatch.setattr(probe, "LOCAL_SNAPSHOT_MAX_BYTES", 1024)
    directory = tmp_path / "big"
    directory.mkdir()
    (directory / "shard.safetensors").write_bytes(b"x" * 4096)
    with pytest.raises(ValueError, match="above the .* ceiling"):
        probe.snapshot_manifest_digest(str(directory))


def test_the_reconstruction_check_compares_canonical_bytes(tmp_path):
    """F1b MINOR 6. Python reads JSON `2` and `2.0` as equal numbers, so retyping a recorded duration
    and re-hashing passed a check whose docstring claimed byte-identity."""
    import hashlib

    context = _context()
    name = "canonical.json"
    _publish(tmp_path, [_measurement(context, step_seconds=2.0)], context=context, name=name)
    path = tmp_path / name
    stored = json.loads(path.read_text())
    assert stored["payload"]["measurements"][0]["step_seconds"] == 2.0
    stored["payload"]["measurements"][0]["step_seconds"] = 2  # int, not float
    stored["sha256"] = hashlib.sha256(json.dumps(stored["payload"], sort_keys=True).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="does not reproduce it"):
        probe.load_authorization(str(path))
    assert "canonical" in inspect.getsource(probe.load_authorization), "the claim matches the check"


# ---------------------------------------------------------------------------------------------
# F1b fixtures: the tiny real model, shared by the tests above.
# ---------------------------------------------------------------------------------------------


class _TinySource:
    """The weights seam at test dimensions — the same record, the same config keys."""

    def mesh(self, config):
        import jax
        import numpy as np

        return jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))

    def load(self, config):
        """The WEIGHTS seam at test dimensions — the same record production's loader returns (W3).

        It no longer builds the adapter: the shared finalizer does that, so a host test exercises the
        very construction M1 and the trainer both enter.
        """
        import types as _types

        import jax
        import jax.numpy as jnp
        from flax import nnx
        from flax.linen import partitioning as nn_partitioning

        from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
        from maxdiffusion.pos_rollout_update import LoadedBackbone

        declared_ = probe.declared
        mesh = self.mesh(config)
        with nn_partitioning.axis_rules(()), mesh:
            transformer = WanModel(
                rngs=nnx.Rngs(jax.random.key(0)),
                num_attention_heads=2,
                attention_head_dim=8,
                in_channels=int(declared_(config, "latent_channels")),
                out_channels=int(declared_(config, "latent_channels")),
                text_dim=int(declared_(config, "text_dim")),
                freq_dim=16,
                ffn_dim=32,
                num_layers=1,
                attention="dot_product",
                rope_max_seq_len=64,
                scan_layers=False,
                dtype=jnp.float32,
                weights_dtype=jnp.float32,
            )
        scheduler = _types.SimpleNamespace(
            config=_types.SimpleNamespace(sigma_min=0.0, sigma_max=1.0, num_train_timesteps=1000)
        )
        return LoadedBackbone(
            transformer=transformer,
            mesh=mesh,
            null_context=jnp.full(
                (1, int(declared_(config, "wan_max_sequence_length")), int(declared_(config, "text_dim"))),
                0.25,
                jnp.float32,
            ),
            scheduler=scheduler,
        )


def _tiny_probe_config(tmp_path: Path, **overrides):
    values = {
        "checkpoint_dir": str(tmp_path / "ckpt"),
        "latent_channels": 4,
        "latent_frames": 2,
        "latent_height": 4,
        "latent_width": 6,
        "text_dim": 32,
        "wan_max_sequence_length": 8,
        "action_tokens": 4,
        "action_hidden": 16,
        "action_heads": 2,
        "pre_context_tokens": 4,
        "pre_context_heads": 2,
        "side_adapter_layers": "0",
        "side_adapter_hidden": 16,
        "side_adapter_heads": 2,
        "side_adapter_sampling_steps": 4,
        "weights_dtype": "float32",
        "activations_dtype": "float32",
    }
    values.update(overrides)
    return _config(**values)


# ---------------------------------------------------------------------------------------------
# F1b battery strengthening: three mutants survived the first run, each because the test asserted a
# SOURCE STRING rather than the behaviour. Fixed, not ratified.
# ---------------------------------------------------------------------------------------------


def test_the_probe_obtains_its_optimizer_from_the_shared_primitive(tmp_path, monkeypatch):
    """Battery G07. The first test grepped for ``optax.adamw(`` — a private optimizer spelled
    ``_o.adamw(`` walked straight past it. What matters is that the probe CALLS the shared builder."""
    from maxdiffusion import pos_rollout_update

    _install_import_shims()
    calls = []
    real = pos_rollout_update.build_optimizer

    def _recording(config, *, num_steps):
        calls.append(int(num_steps))
        return real(config, num_steps=num_steps)

    monkeypatch.setattr(probe, "build_optimizer", _recording, raising=False)
    monkeypatch.setattr(pos_rollout_update, "build_optimizer", _recording)
    tiny = _tiny_probe_config(tmp_path, pos_logical_batch=8, pos_microbatch=8, max_train_steps=1234)
    probe.build_probe_program(tiny, probe.FitCell("rollout", 8, 2), model_source=_TinySource())
    assert calls == [1234], "the probe must build its optimizer through the shared production builder"


def test_the_shared_optimizer_carries_the_configured_schedule_and_clipping():
    """Battery G08. Grepping for ``max_utils.create_optimizer`` proves nothing about the object that
    comes back; these are the two properties a substituted ``optax.adamw`` silently loses."""
    import jax.numpy as jnp

    from maxdiffusion.pos_rollout_update import build_optimizer

    _install_import_shims()
    config = _config(
        learning_rate=1.0,
        warmup_steps_fraction=0.5,
        learning_rate_schedule_steps=-1,
        opt_enable_grad_global_norm_clipping=True,
        max_grad_norm=1e-3,
    )
    optimizer, schedule = build_optimizer(config, num_steps=100)
    assert float(schedule(0)) < 1.0, "a warmup schedule does not start at the peak learning rate"
    assert float(schedule(0)) < float(schedule(50)), "and it rises through warmup"

    params = {"w": jnp.zeros((4,), jnp.float32)}
    state = optimizer.init(params)
    huge = {"w": jnp.full((4,), 1e6, jnp.float32)}
    updates, _ = optimizer.update(huge, state, params)
    assert float(jnp.linalg.norm(updates["w"])) < 1.0, (
        "a 1e6-norm gradient must be clipped by the configured global-norm clip; an unclipped adamw "
        "would step by roughly the learning rate in every coordinate"
    )


def test_a_storage_error_that_merely_SPELLS_oom_is_not_an_allocation_refusal():
    """Battery G18. Both of the reviewer's probes are caught by the phrase regex alone, so nothing
    exercised the type guard behind it — and the guard is what separates a DISK that reports 'OOM'
    from HBM exhaustion. An OSError is never an allocation refusal, whatever it spells."""
    assert probe._is_resource_exhausted(OSError("OOM while flushing shard to disk")) is False
    assert probe._is_resource_exhausted(OSError("RESOURCE_EXHAUSTED writing to bucket")) is False
    assert probe._is_resource_exhausted(IOError("OUT OF MEMORY on the storage node")) is False
    # ...while the same phrases from the runtime still classify.
    assert probe._is_resource_exhausted(RuntimeError("OOM while allocating on device")) is True


def test_the_manifest_records_are_length_framed(tmp_path):
    """Review W1, A5 — the reviewer's collision, verbatim.

    ``path + NUL + bytes`` is a chosen serialization with an ambiguous parse: ONE file named ``a``
    holding ``Xb\\0Y`` produced the same byte stream as TWO files ``a=X`` and ``b=Y``. A snapshot
    reshuffled at equal total bytes was therefore invisible to the identity. Every record now
    declares its own lengths and the file count is framed, so exactly one tree maps to one stream.
    """
    left, right = tmp_path / "one_file", tmp_path / "two_files"
    left.mkdir()
    right.mkdir()
    (left / "a").write_bytes(b"X" + b"b" + b"\0" + b"Y")
    (right / "a").write_bytes(b"X")
    (right / "b").write_bytes(b"Y")
    assert probe.snapshot_manifest_digest(str(left)) != probe.snapshot_manifest_digest(str(right))

    # The same ambiguity one level down: a path whose NAME carries the separator.
    tricky, plain = tmp_path / "tricky", tmp_path / "plain"
    tricky.mkdir()
    plain.mkdir()
    (tricky / "ab").write_bytes(b"YZ")
    (plain / "a").write_bytes(b"b")
    (plain / "b").write_bytes(b"YZ")
    assert probe.snapshot_manifest_digest(str(tricky)) != probe.snapshot_manifest_digest(str(plain))

    # ...and the framing is versioned, so a future change to the record layout is a new identity.
    assert "exp06.snapshot.v2" in inspect.getsource(probe.snapshot_manifest_digest)


def test_the_adapter_factory_is_shared_and_carries_the_production_dtypes():
    """Review W1, A2's concrete half. The probe rebuilt this construction by hand and omitted
    ``dtype``, ``weights_dtype`` and ``precision`` — agreeing with production only by coincidence of
    the pinned defaults, so the first debugging run at float32 would have measured the wrong model."""
    from maxdiffusion import pos_rollout_update

    factory = inspect.getsource(pos_rollout_update.build_adapter_stack)
    # Read the settled trainer's construction from DISK: importing it pulls the full pipeline stack,
    # and the property under test is that the two argument lists agree.
    settled = (_MODULE_PATH.parent / "trainers" / "wan_ti2v_side_adapter_trainer.py").read_text()
    production = settled[settled.index("def _build_adapters") : settled.index("def _compute_null_context")]
    for argument in (
        "num_layers",
        "model_dim",
        "text_dim",
        "action_adapter_type",
        "action_dim",
        "action_len",
        "action_repr",
        "action_tokens",
        "action_hidden",
        "action_heads",
        "side_adapter_layers",
        "side_adapter_hidden",
        "side_adapter_heads",
        "pre_context_tokens",
        "pre_context_heads",
        "dtype",
        "weights_dtype",
        "precision",
    ):
        assert f"{argument}=" in factory, f"the shared factory must pass {argument}"
        assert f"{argument}=" in production, f"...which production passes too: {argument}"
    assert "build_adapter_stack" in inspect.getsource(pos_rollout_update.build_training_program)


def test_the_shared_factory_really_builds_the_configured_adapter(tmp_path):
    """Not a source scan: build it, and read the dtypes back off the constructed module."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from maxdiffusion.models.wan.side_adapter_wan import NNXWanSideAdapterStack
    from maxdiffusion.pos_rollout_update import build_adapter_stack

    _install_import_shims()
    config = _tiny_probe_config(tmp_path, activations_dtype="float32", weights_dtype="float32")
    transformer = _TinySource().load(config).transformer
    adapters = build_adapter_stack(config, transformer)
    assert isinstance(adapters, NNXWanSideAdapterStack)
    assert adapters.action_adapter_type == "pre_context", "the approved architecture"
    assert jnp.dtype(adapters.dtype) == jnp.dtype("float32"), "the ACTIVATION dtype reaches the module"
    # `weights_dtype` is consumed at construction rather than stored, so it is read off a parameter.
    leaves = [leaf for leaf in jax.tree.leaves(nnx.state(adapters, nnx.Param)) if hasattr(leaf, "dtype")]
    assert leaves and all(jnp.dtype(leaf.dtype) == jnp.dtype("float32") for leaf in leaves)

    bf16 = build_adapter_stack(
        _tiny_probe_config(tmp_path, activations_dtype="bfloat16", weights_dtype="bfloat16"), transformer
    )
    assert jnp.dtype(bf16.dtype) == jnp.dtype("bfloat16"), "and a different config builds a different model"


# ---------------------------------------------------------------------------------------------
# W1 battery strengthening: four mutants survived, each because the assertion was satisfied by a
# neighbouring property rather than the one under test. Fixed, not ratified.
# ---------------------------------------------------------------------------------------------


def test_the_record_framing_holds_when_the_file_COUNT_cannot_tell_the_trees_apart(tmp_path):
    """Battery W09/W11. The reviewer's collision has one file on one side and two on the other, so
    the framed COUNT alone separates it — and a mutant that dropped the per-record framing survived.

    These two trees have the SAME number of files and the SAME total bytes, and under an unframed
    ``path + NUL + content`` serialization they produce byte-identical streams:

        A: {"a": b"b\\0Y", "z": b""}   ->  a\\0b\\0Y  +  z\\0   ==  "a\\0b\\0Yz\\0"
        B: {"a": b"",      "b": b"Yz\\0"} ->  a\\0     +  b\\0Yz\\0 ==  "a\\0b\\0Yz\\0"

    Only the per-record length header separates them, which is exactly what A5 asked for.
    """
    left, right = tmp_path / "count_A", tmp_path / "count_B"
    left.mkdir()
    right.mkdir()
    (left / "a").write_bytes(b"b\0Y")
    (left / "z").write_bytes(b"")
    (right / "a").write_bytes(b"")
    (right / "b").write_bytes(b"Yz\0")

    assert len(list(left.rglob("*"))) == len(list(right.rglob("*"))) == 2, "the count cannot separate them"
    assert sum(p.stat().st_size for p in left.rglob("*")) == sum(p.stat().st_size for p in right.rglob("*"))
    unframed_left = b"".join(sorted(p.name.encode() + b"\0" + p.read_bytes() for p in left.rglob("*")))
    unframed_right = b"".join(sorted(p.name.encode() + b"\0" + p.read_bytes() for p in right.rglob("*")))
    assert unframed_left == unframed_right, "the unframed serializations really do collide"

    assert probe.snapshot_manifest_digest(str(left)) != probe.snapshot_manifest_digest(str(right))


def test_a_file_that_changes_size_while_it_is_hashed_is_refused(tmp_path, monkeypatch):
    """Battery W12. The size in the record header is a claim about the bytes that follow it; if the
    file is being written underneath the probe, the record describes neither the old nor the new
    file, and a manifest that shipped it would identify a model that never existed."""
    import pathlib

    directory = tmp_path / "racing"
    directory.mkdir()
    target = directory / "shard.safetensors"
    target.write_bytes(b"x" * 4096)

    real_stat = pathlib.Path.stat

    def _lying_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self.name == "shard.safetensors":

            class _Bigger:
                st_size = result.st_size + 512

                def __getattr__(self, name):
                    return getattr(result, name)

            return _Bigger()
        return result

    monkeypatch.setattr(pathlib.Path, "stat", _lying_stat)
    with pytest.raises(ValueError, match="changed size while it was being hashed"):
        probe.snapshot_manifest_digest(str(directory))


def test_M1_and_the_shared_factory_build_the_SAME_adapter(tmp_path):
    """Battery W17. Asserting that ``build_adapter_stack`` appears in the source let a hand-rebuilt
    construction survive next to it; what matters is that the object M1 gets is the object the
    factory makes, parameter tree and dtypes included."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from maxdiffusion.pos_rollout_update import build_adapter_stack

    _install_import_shims()
    config = _tiny_probe_config(tmp_path)
    backbone = _TinySource().load(config)
    transformer, from_source = backbone.transformer, build_adapter_stack(config, backbone.transformer)
    from_factory = build_adapter_stack(config, transformer)

    def _shapes(module):
        return jax.tree.map(
            lambda leaf: (tuple(jnp.shape(leaf)), str(jnp.dtype(leaf.dtype))), nnx.state(module, nnx.Param)
        )

    assert jax.tree.structure(_shapes(from_source)) == jax.tree.structure(
        _shapes(from_factory)
    ), "M1's adapter and the shared factory's adapter must have the same parameter tree"
    assert _shapes(from_source) == _shapes(from_factory), "...with the same shapes and dtypes"
    assert from_source.action_adapter_type == from_factory.action_adapter_type == "pre_context"


def test_the_adapter_only_optimizer_never_touches_the_frozen_backbone(tmp_path):
    """The S7 freeze-split discipline, on the tiny fixture: optimizer state is built for the ADAPTER
    parameters alone, and the frozen transformer is not in the trainable tree at all."""
    import jax

    from maxdiffusion.pos_rollout_arms import build_arm
    from maxdiffusion.pos_rollout_update import build_optimizer

    from maxdiffusion.pos_rollout_update import build_adapter_stack

    _install_import_shims()
    config = _tiny_probe_config(tmp_path)
    backbone = _TinySource().load(config)
    transformer = backbone.transformer
    adapters = build_adapter_stack(config, transformer)
    _, adapter_params, _ = build_arm("rollout", transformer, adapters)
    optimizer, _ = build_optimizer(config, num_steps=100)
    opt_state = optimizer.init(adapter_params)

    trainable = jax.tree.leaves(adapter_params)
    assert trainable, "the adapter has parameters"
    from flax import nnx

    frozen = jax.tree.leaves(nnx.state(transformer, nnx.Param))
    assert frozen, "the transformer has parameters too"
    assert len(trainable) < len(frozen), "the adapter is the small half of the split"
    trainable_ids = {id(leaf) for leaf in trainable}
    assert not (trainable_ids & {id(leaf) for leaf in frozen}), "no frozen leaf is in the trainable tree"
    assert len(jax.tree.leaves(opt_state)) > 0
    for leaf in jax.tree.leaves(opt_state):
        assert not any(leaf is frozen_leaf for frozen_leaf in frozen), "no optimizer slot for a frozen leaf"


def test_the_SIZE_in_each_record_header_is_load_bearing(tmp_path):
    """Battery W11. With the path length framed but the SIZE dropped, a record's content can still
    absorb the next record's header — the framing has to say where the bytes END, not only where the
    path does. These two trees have equal file counts, equal total bytes, equal path lengths, and are
    byte-identical under ``len(path) + NL + path + content``:

        A: {"a": b"",      "b": b"1\\nz"}  ->  1\\na      + 1\\nb1\\nz
        B: {"a": b"1\\nb", "z": b""}       ->  1\\na1\\nb  + 1\\nz
    """
    left, right = tmp_path / "size_A", tmp_path / "size_B"
    left.mkdir()
    right.mkdir()
    (left / "a").write_bytes(b"")
    (left / "b").write_bytes(b"1\nz")
    (right / "a").write_bytes(b"1\nb")
    (right / "z").write_bytes(b"")

    def _sizeless(root):
        parts = []
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            name = str(path.relative_to(root)).encode()
            parts.append(f"{len(name)}\n".encode() + name + path.read_bytes())
        return b"".join(parts)

    assert _sizeless(left) == _sizeless(right), "the size-less serializations really do collide"
    assert probe.snapshot_manifest_digest(str(left)) != probe.snapshot_manifest_digest(str(right))


def test_M1_CALLS_the_shared_factory_rather_than_merely_importing_it():
    """Battery W17. ``build_adapter_stack`` appears in ``ProductionModelSource.build`` as an IMPORT,
    so a hand-rebuilt construction sitting next to that import passed a substring check. The property
    is that the factory is CALLED and that no adapter class is constructed here directly."""
    import ast as _ast

    from maxdiffusion import pos_rollout_update

    source = textwrap.dedent(inspect.getsource(pos_rollout_update.build_training_program))
    tree = _ast.parse(source)
    called = {
        node.func.id for node in _ast.walk(tree) if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    }
    assert "build_adapter_stack" in called, "the shared finalizer must CALL the shared factory"
    assert "NNXWanSideAdapterStack" not in called, "and must not construct the adapter itself"
    assert "NNXWanSideAdapterStack" not in source, "not even under another name"
