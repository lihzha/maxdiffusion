"""exp_06 `rollout_adapter` — T5a: restore the selected checkpoint, and reproduce the baseline.

**One contract: a checkpoint can be restored and scored, and the historical baseline is reproduced
before anything new is measured.**

**Restore reaches the selection artifact, structurally.** exp_05's S9 established the discipline —
evaluate the immutable earliest-best checkpoint the stop rule selected, never the resume tree's
latest, because evaluating an unselected checkpoint produces a plausible number for the wrong model.
S9 took a *manager* and documented that no parameter could reach the resume tree; that was a claim
about what callers would pass. :func:`restore_selected_checkpoint` takes the checkpoint ROOT and
derives the sibling itself, so handing it the resume root reaches an empty ``<root>_selection`` tree
and is refused. The metadata match is widened to exp_06's own identity: step, DEV metric, **arm and
horizon**, because a matched-C0 artifact scored as R-B would invert the experiment's claim.

**The anchor protocol (plan §3c) is the round's headline, and it runs before any new-arm scoring.**
The deployed pre_context checkpoint's recorded 4-sample validation — mean SSIM **0.2946**, latent MSE
1.496, pixel MSE 0.0983 at step 30,000 — is reproduced within a predeclared band. This is the wiring
proof: if exp_06's evaluation path cannot recover the number on the checkpoint that produced it, then
every later exp_06 number is measured with an uncalibrated instrument, and a "+0.05 paired ΔSSIM"
would be a statement about the evaluator rather than the objective. All three recorded metrics are
compared, not SSIM alone: a latent path that is wrong while SSIM happens to land is exactly the
coincidence a single-metric check waves through.

**THE ANCHOR IS A WIRING CHECK AND MUST NEVER BE QUOTED AS A QUALITY BASELINE.** The run's own
``summary.csv`` shows its four samples are four windows of a SINGLE episode —
``ep10099_v0_s00000/_s00004/_s00008/_s00012``, per-sample SSIM 0.175–0.484 — so 0.2946 is a mean over
four correlated windows of one clip, not an estimate of the deployed adapter's quality on DROID. It
answers exactly one question: does exp_06's evaluation path compute what the deployed path computed?
The quality baseline every later table carries is the DEV-64 benchmark row (:func:`freeze_benchmark_row`).

**The tolerance is a constant and takes no argument.** :data:`ANCHOR_REL_TOLERANCE` is 2% relative.
Its basis is DISCRIMINATION, not a confidence interval: this is a REPRODUCTION check, so between-sample
variance is not the right frame — but it gives the number meaning. The run's per-sample spread is
SSIM sd 0.1336, a SEM of 22.7% of the mean (latent 18.3%, pixel 16.4%), so **the 2% band is ~8× tighter
than the between-sample SEM and cannot be satisfied by accident**, while still absorbing float and
hardware nondeterminism. Every wiring failure this path can have is order-unity anyway: the
frozen-context control sits near 0.25, the per-clip oracle near 0.92, and a missing frame-0 pin or a
wrong noise convention is worse than either.

**What is NOT re-implemented is pinned as hard as what is.** The prediction is T3a's
:func:`~maxdiffusion.pos_rollout_step.cfg_rollout`; the SSIM is the deployed definition, copied
verbatim and held by a drift tripwire; the initial noise follows deployment's sequential-split
convention in the video latents' dtype with frame 0 pinned. Plan §5-5's standing ruling applies: the
settled evaluator is NOT rewired, and parity is discharged BY TEST. A wiring proof written against a
private copy of the wiring proves nothing.

**DECLARED DEVIATION — non-finite SSIM.** The deployed summary averages only the finite SSIMs so that
a worker without scikit-image still yields a number. :func:`summarize_samples` REFUSES instead: a mean
over whichever samples happened to work is a different estimand, and exp_04's rule is that a missing
example is refused, never imputed. This cannot change the anchor value (all four anchor samples are
finite, so the two aggregations coincide); it changes what a broken worker does, which is the point.

**STAMPED ≠ BOUND (review pass 2, and the reason this module was reworked).** Provenance used to be
CARRIED beside a measurement rather than DERIVED from it, so a certificate repeated what its caller
typed. The reviewer executed the consequences: the recorded means with FOUR UNRELATED sample names
returned ``reproduced=True``; a certificate was issued for checkpoint ``{"run_name":
"some-other-run", "step": 1}``; and a ``num_steps=1`` rollout was certifiable as ``num_steps=25``.
Now every one of those facts is derived from the artifact that produced it:

* :class:`CheckpointIdentity` is built only by a restore, and its ``run_name`` comes from the
  checkpoint ROOT's own path rather than from an argument.
* :class:`RolloutExecution` carries the horizon that actually ran. :func:`rollout_prediction` has no
  ``num_steps`` parameter at all — the deployed grid is enforced at the scoring boundary.
* :class:`Measurement` is digest-bound and is the ONLY thing :func:`reproduce_anchor` and
  :func:`anchor_certificate` will read; the anchor additionally requires its exact four historical
  sample names IN ORDER, plus the historical run and step.
* :class:`ScoreTable` is the same discipline for per-example tables: cohort, condition, arm,
  checkpoint, horizon and per-row action/draw identities, under one digest. The gates in
  :mod:`maxdiffusion.pos_rollout_gates` accept these and never a naked mapping.

**The evaluator RUNS (review pass 2, T5a BLOCKER 1).** :func:`main` dispatches the four phases of
plan §3c — ``anchor → benchmark → gates → confirm`` — and each is implemented end to end. The
Planner's ruling is respected literally: "name the boundary in the error" was accepted for DEVICE
work and does NOT extend to the orchestration, so the only seam left is
:class:`DeviceBackend`/:func:`load_device_backend`, which is where real Wan weights and a real VAE
are genuinely required. The ordering is enforced by the code rather than by the launcher: every
new-arm phase must load the anchor's own certificate and find it reproduced (:func:`require_anchor`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.wan.side_adapter_wan import apply_first_frame_pin
from maxdiffusion.pos_rollout_dev_instrument import load_dev_cohort
from maxdiffusion.pos_rollout_loop import (
    DEV_METRIC,
    build_selection_manager,
    read_checkpoint_json,
    restore_checkpoint,
)
from maxdiffusion.pos_rollout_step import cfg_rollout
from maxdiffusion.pos_rollout_support import storage_exists, storage_read_bytes, storage_write_bytes

__all__ = [
    "ANCHOR_PROTOCOL",
    "ANCHOR_REL_TOLERANCE",
    "DEPLOYED_SAMPLING_STEPS",
    "ANCHOR_METRICS",
    "EVAL_NOISE_SEED",
    "EVAL_PHASES",
    "J0_TEST64_SHA256",
    "BENCHMARK_PROTOCOL",
    "MEASUREMENT_PROTOCOL",
    "RUN_REPORT_PROTOCOL",
    "SCORE_TABLE_PROTOCOL",
    "load_run_report",
    "main",
    "publish_run_report",
    "restore_from_report",
    "HISTORICAL_ANCHOR",
    "AnchorRecord",
    "AnchorVerdict",
    "CheckpointIdentity",
    "DeviceBackend",
    "Measurement",
    "RolloutExecution",
    "ScoreTable",
    "anchor_certificate",
    "anchor_sample_keys",
    "assert_no_test_examples",
    "build_score_table",
    "draw_key_digest",
    "evaluation_draw_key",
    "freeze_benchmark_row",
    "frame_ssim",
    "initial_latents",
    "load_benchmark_row",
    "load_certificate",
    "load_device_backend",
    "load_score_table",
    "protocol_root_for",
    "publish_certificate",
    "publish_score_table",
    "reproduce_anchor",
    "require_anchor",
    "restore_anchor_checkpoint",
    "restore_selected_checkpoint",
    "rollout_prediction",
    "run_anchor_phase",
    "run_benchmark_phase",
    "run_confirm_phase",
    "run_evaluation",
    "run_gates_phase",
    "sample_metrics",
    "summarize_samples",
]

ANCHOR_PROTOCOL = "exp06.anchor.v1"
BENCHMARK_PROTOCOL = "exp06.dev64_benchmark.v1"
RUN_REPORT_PROTOCOL = "exp06.run_report.v1"
MEASUREMENT_PROTOCOL = "exp06.measurement.v1"
SCORE_TABLE_PROTOCOL = "exp06.score_table.v1"
#: Plan §3c's protocol, in order. The launcher names a phase; the ORDER is enforced here.
EVAL_PHASES = ("anchor", "benchmark", "gates", "confirm")
#: The evaluation's own noise seed. Fixed for the life of the experiment and unrelated to a run's
#: training seed: a condition comparison must not move when a run's seed does.
EVAL_NOISE_SEED = 20260806
#: Relative band, both sides. A constant, never an argument — see the module docstring for why 2%.
#: The deployed sigma grid. A rollout measured at another horizon is a different measurement, so this
#: is checked rather than merely recorded (T5a review item 7).
DEPLOYED_SAMPLING_STEPS = 25
ANCHOR_REL_TOLERANCE = 0.02
ANCHOR_METRICS = ("mean_latent_mse", "mean_pixel_mse", "mean_ssim")
#: exp_04's published J0 TEST-64 manifest. Pinned by DIGEST because this file is read to REFUSE:
#: a guard that accepts whatever manifest it is handed is defanged by handing it an empty one.
J0_TEST64_SHA256 = "878576867003aafd1547e500924c51a40ab1bba80e30496c1c6c485b64bd519b"


@dataclasses.dataclass(frozen=True)
class AnchorRecord:
    """The published 4-sample validation of the deployed pre_context checkpoint.

    Transcribed once from `master_experiment_tracker.md` ("Run:
    wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110", validation @ step_030000) so the number
    exp_06 calibrates against has exactly one definition in the tree, and VERIFIED against the run's
    own ``summary.json``: 0.29460108026184817 / 1.4960926324129105 / 0.0983371902257204 over n=4.

    **A wiring reference, not a quality baseline** — the four samples are windows of one episode; see
    the module docstring. Nothing in exp_06 may quote it as the deployed adapter's DROID quality.
    """

    run_name: str
    checkpoint_step: int
    num_samples: int
    mean_latent_mse: float
    mean_pixel_mse: float
    mean_ssim: float
    artifacts: str
    #: The FOUR samples, in the order the deployed run's ``summary.csv`` recorded them. Transcribed
    #: from that file, and the reason the anchor is a wiring check: they are four windows of ONE
    #: episode. Bound rather than described, because the reviewer passed the recorded means with four
    #: unrelated names and the reproduction returned True.
    sample_names: tuple = ()


HISTORICAL_ANCHOR = AnchorRecord(
    run_name="wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110",
    checkpoint_step=30000,
    num_samples=4,
    mean_latent_mse=1.496,
    mean_pixel_mse=0.0983,
    mean_ssim=0.2946,
    artifacts=(
        "gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter/"
        "wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110/validation/step_030000/"
    ),
    sample_names=("ep10099_v0_s00000", "ep10099_v0_s00004", "ep10099_v0_s00008", "ep10099_v0_s00012"),
)


@dataclasses.dataclass(frozen=True)
class AnchorVerdict:
    reproduced: bool
    deviations: dict
    failed: tuple
    measurement: "Measurement"


def reproduce_anchor(measurement: "Measurement") -> AnchorVerdict:
    """Plan §3c(a): did this evaluation path recover the published baseline? No knobs, no mappings.

    Everything compared here is DERIVED from the measurement — its restored checkpoint identity, the
    horizon that executed, and the samples :func:`summarize_samples` actually aggregated. A caller
    contributes nothing but the measurement itself.
    """
    if not isinstance(measurement, Measurement):
        raise TypeError(
            "the anchor is reproduced from a Measurement, never from a mapping: a mapping is a caller's "
            "description of a measurement, and the reviewer passed the recorded means with four "
            "unrelated sample names to prove the difference"
        )
    checkpoint = measurement.checkpoint
    if checkpoint.run_name != HISTORICAL_ANCHOR.run_name:
        raise ValueError(
            f"the anchor is the historical run {HISTORICAL_ANCHOR.run_name!r}; this measurement was "
            f"taken on {checkpoint.run_name!r}. The run name is derived from the checkpoint root that "
            f"was opened, so this is a statement about which artifact was restored."
        )
    if int(checkpoint.step) != HISTORICAL_ANCHOR.checkpoint_step:
        raise ValueError(
            f"the anchor is step {HISTORICAL_ANCHOR.checkpoint_step}, this measurement restored step "
            f"{checkpoint.step}: another checkpoint has no published baseline"
        )
    if int(measurement.num_samples) != HISTORICAL_ANCHOR.num_samples:
        raise ValueError(
            f"the anchor is the recorded {HISTORICAL_ANCHOR.num_samples} samples, got "
            f"{measurement.num_samples}: a different sample count is a different measurement"
        )
    names = list(measurement.sample_names)
    recorded_names = list(HISTORICAL_ANCHOR.sample_names)
    if sorted(names) != sorted(recorded_names):
        raise ValueError(
            f"the anchor is the recorded samples {recorded_names}, this measurement scored {names}: the "
            f"recorded means are a mean over THOSE four windows, and four other samples that happen to "
            f"average to the same number are a different measurement"
        )
    if names != recorded_names:
        raise ValueError(
            f"the anchor samples are scored in the order the val directory yielded them "
            f"{recorded_names}, got {names}: deployment's noise is a SEQUENTIAL split, so sample i "
            f"depends on how many preceded it and the order is part of the measurement"
        )
    if int(measurement.num_steps) != DEPLOYED_SAMPLING_STEPS:
        raise ValueError(
            f"the anchor was measured on the deployed {DEPLOYED_SAMPLING_STEPS}-step rollout, this "
            f"measurement executed {measurement.num_steps}: another horizon is a different measurement"
        )
    deviations, failed = {}, []
    for metric in ANCHOR_METRICS:
        value = measurement.means[metric]
        recorded = float(getattr(HISTORICAL_ANCHOR, metric))
        deviations[metric] = (float(value) - recorded) / recorded
        if abs(deviations[metric]) > ANCHOR_REL_TOLERANCE:
            failed.append(metric)
    return AnchorVerdict(reproduced=not failed, deviations=deviations, failed=tuple(failed), measurement=measurement)


def anchor_certificate(verdict: AnchorVerdict) -> dict:
    """What an anchor run publishes — derived ENTIRELY from the measurement the verdict carries.

    There is no ``checkpoint``, ``code_sha``, ``model_revision`` or ``num_steps`` argument any more.
    Each of those was a caller's assertion, and each was forged by the reviewer.
    """
    if not isinstance(verdict, AnchorVerdict):
        raise TypeError("an anchor certificate is issued from an AnchorVerdict that reproduce_anchor returned")
    measurement = verdict.measurement
    return {
        "protocol": ANCHOR_PROTOCOL,
        "reproduced": bool(verdict.reproduced),
        "failed": list(verdict.failed),
        "deviations": {name: float(value) for name, value in verdict.deviations.items()},
        "measured": dict(measurement.payload),
        "measurement_sha256": measurement.digest,
        "recorded": {metric: float(getattr(HISTORICAL_ANCHOR, metric)) for metric in ANCHOR_METRICS},
        "tolerance": ANCHOR_REL_TOLERANCE,
        "checkpoint": measurement.checkpoint.payload(),
        "sample_names": list(measurement.sample_names),
        "code_sha": measurement.code_sha,
        "model_revision": measurement.model_revision,
        "num_steps": int(measurement.num_steps),
    }


def _plain(value):
    return float(value) if isinstance(value, float) else value


# ---------------------------------------------------------------------------------------------
# Restore: the selection artifact, reached from the ROOT — and the identity it yields.
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CheckpointIdentity:
    """WHICH artifact an evaluation opened, derived from the artifact — never described by a caller.

    ``run_name`` is read off the checkpoint ROOT's own path (the run directory that contains
    ``checkpoints``); ``step`` is what Orbax actually restored; ``arm``/``k_b``/``dev_metric`` come
    from the checkpoint's own step JSON. The reviewer's executed attack was
    ``anchor_certificate(..., checkpoint={"run_name": "some-other-run", "step": 1})`` — a certificate
    that repeated a caller's typing. Nothing in this class can be typed into a certificate: the two
    constructors below are restores, and the anchor additionally requires the run name it derives to
    BE the historical run's.
    """

    run_name: str
    step: int
    root: str
    source: str
    arm: str | None = None
    k_b: int | None = None
    dev_metric: float | None = None

    def payload(self) -> dict:
        return {
            "run_name": self.run_name,
            "step": int(self.step),
            "root": self.root,
            "source": self.source,
            "arm": self.arm,
            "k_b": None if self.k_b is None else int(self.k_b),
            "dev_metric": None if self.dev_metric is None else float(self.dev_metric),
        }


def _run_name_from_root(ckpt_dir: str) -> str:
    """The run a checkpoint root belongs to, read off the path the job was pointed at.

    Every exp_06 and exp_05 root is ``<...>/<run_name>/checkpoints`` (the launchers build exactly
    that), so the run name is a FACT ABOUT THE ARTIFACT rather than a second thing to be asserted.
    A root that does not follow the convention still yields its own last segment, which is the most
    a path can honestly say — and the anchor then refuses it unless it IS the historical run.
    """
    parts = [part for part in str(ckpt_dir).rstrip("/").split("/") if part and part != ".."]
    if not parts:
        raise ValueError(f"{ckpt_dir!r} names no checkpoint root, so no run identity can be derived from it")
    if len(parts) >= 2 and parts[-1].startswith("checkpoints"):
        return parts[-2]
    return parts[-1]


def publish_run_report(path: str, report, *, arm: str, k_b: int, num_steps: int) -> dict:
    """Publish the training run's own account of what it selected, from the loop's ``RunReport``.

    The evaluator used to take ``expected_step``/``expected_dev_metric``/``expected_arm``/
    ``expected_k_b`` as loose keyword arguments, which made the artifact-vs-report check only as
    strong as the caller's memory of the report (T5a review item 6). The report is now an artifact
    with a digest, produced from the object the loop returned rather than retyped.
    """
    if report.retained_step is None:
        raise ValueError(
            "this run selected no checkpoint (its history is empty), so there is nothing for an "
            "evaluation to measure; publishing a report that names no step would invite one"
        )
    selected = [record for record in report.history if int(record.step) == int(report.retained_step)]
    if not selected:
        raise ValueError(
            f"the report retains step {report.retained_step} but its history has no such evaluation: "
            f"the retained step must be one the run actually measured"
        )
    if int(num_steps) != DEPLOYED_SAMPLING_STEPS:
        raise ValueError(f"runs are evaluated on the deployed {DEPLOYED_SAMPLING_STEPS}-step grid, got {num_steps!r}")
    return publish_certificate(
        path,
        {
            "protocol": RUN_REPORT_PROTOCOL,
            "retained_step": int(report.retained_step),
            DEV_METRIC: float(selected[0].dev_metric),
            "arm": str(arm),
            "k_b": int(k_b),
            "num_steps": int(num_steps),
            "steps_run": int(report.steps_run),
            "stopped": bool(report.verdict.stop),
            "stop_reason": str(report.verdict.reason),
        },
    )


def load_run_report(path: str) -> dict:
    """Read a published run report and re-verify its digest: an edited report blesses a wrong model."""
    stored = json.loads(storage_read_bytes(path).decode("utf-8"))
    payload = stored.get("payload")
    if not isinstance(payload, Mapping) or _digest(payload) != stored.get("sha256"):
        raise ValueError(
            f"{path}: the recorded digest {stored.get('sha256')!r} does not describe its payload — the "
            f"run report has been edited since it was published, and it is what names the checkpoint"
        )
    if payload.get("protocol") != RUN_REPORT_PROTOCOL:
        raise ValueError(f"{path} is not a {RUN_REPORT_PROTOCOL} artifact: {payload.get('protocol')!r}")
    return dict(payload)


def restore_from_report(ckpt_dir: str, template: Any, *, report_path: str) -> tuple[Any, CheckpointIdentity]:
    """The production restore: the run's published report decides what the artifact must be."""
    report = load_run_report(report_path)
    return restore_selected_checkpoint(
        ckpt_dir,
        template,
        expected_step=int(report["retained_step"]),
        expected_dev_metric=float(report[DEV_METRIC]),
        expected_arm=str(report["arm"]),
        expected_k_b=int(report["k_b"]),
    )


def restore_anchor_checkpoint(ckpt_dir: str, template: Any) -> tuple[Any, CheckpointIdentity]:
    """Restore the DEPLOYED pre_context checkpoint the anchor is measured on — that one, and no other.

    The historical run predates exp_06's selection sibling, so it is restored from its own resume
    tree; what makes that safe is that BOTH coordinates are bound rather than accepted. The run name
    is derived from the root and must be the historical run's, and the step is not "latest" but
    exactly :data:`HISTORICAL_ANCHOR.checkpoint_step`. Pointing this at another run, or at the same
    run's later checkpoints, is refused before any weights move.
    """
    run_name = _run_name_from_root(ckpt_dir)
    if run_name != HISTORICAL_ANCHOR.run_name:
        raise ValueError(
            f"{ckpt_dir} belongs to run {run_name!r}; the anchor is the deployed run "
            f"{HISTORICAL_ANCHOR.run_name!r}. The published 0.2946 is a fact about THAT checkpoint, so "
            f"reproducing it anywhere else certifies nothing."
        )
    import orbax.checkpoint as ocp

    manager = ocp.CheckpointManager(
        ckpt_dir,
        item_names=("params", "opt_state", "step"),
        item_handlers={
            "params": ocp.StandardCheckpointHandler(),
            "opt_state": ocp.StandardCheckpointHandler(),
            "step": ocp.JsonCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    step = HISTORICAL_ANCHOR.checkpoint_step
    if step not in set(manager.all_steps()):
        raise ValueError(
            f"{ckpt_dir} carries no step {step}: the anchor is the recorded validation of that exact "
            f"checkpoint, and the run's latest checkpoint is not a substitute for it"
        )
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(template.params),
            opt_state=ocp.args.StandardRestore(template.opt_state),
            step=ocp.args.JsonRestore(),
        ),
    )
    restored_step = int(restored["step"]["step"])
    if restored_step != step:
        raise ValueError(f"{ckpt_dir} step {step} restored a checkpoint declaring step {restored_step}")
    state = dataclasses.replace(template, params=restored["params"], opt_state=restored["opt_state"], step=step)
    return state, CheckpointIdentity(run_name=run_name, step=step, root=str(ckpt_dir), source="historical")


def restore_selected_checkpoint(
    ckpt_dir: str,
    template: Any,
    *,
    expected_step: int,
    expected_dev_metric: float,
    expected_arm: str,
    expected_k_b: int,
) -> tuple[Any, CheckpointIdentity]:
    """The checkpoint an evaluation measures: the sibling selection artifact, metadata-checked.

    ``ckpt_dir`` is the RUN's checkpoint root; the selection sibling is derived from it here, so
    there is no argument through which the resume tree can be reached — not by convention, by
    construction. Every field the run report named must match: two runs can select the same step, and
    the metric, the arm and the horizon are what tell their artifacts apart.
    """
    manager = build_selection_manager(ckpt_dir)
    step = manager.latest_step()
    if step is None:
        raise ValueError(
            f"there is no selection artifact under {ckpt_dir}: the loop publishes one at its earliest "
            f"best DEV metric, and an evaluation without it has no defined subject. The resume tree's "
            f"latest checkpoint is NOT a substitute — it is whatever step preemption last wrote."
        )
    manifest = read_checkpoint_json(manager, step)
    if int(manifest.get("step", -1)) != int(expected_step):
        raise ValueError(
            f"the selection artifact is at step {manifest.get('step')!r} but the report named "
            f"{expected_step}: the evaluation would measure a checkpoint the run did not select"
        )
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
    if not math.isfinite(float(stored)) or not math.isclose(
        float(stored), float(expected_dev_metric), rel_tol=1e-9, abs_tol=0.0
    ):
        raise ValueError(
            f"the selection artifact's {DEV_METRIC} is {stored!r}, not the reported "
            f"{expected_dev_metric!r}: the artifact and the report disagree about which run this is"
        )
    if str(manifest.get("arm")) != str(expected_arm):
        raise ValueError(
            f"the selection artifact declares arm {manifest.get('arm')!r}, not {expected_arm!r}: "
            f"scoring matched-C0 as R-B would invert the comparison exp_06 exists to make"
        )
    if int(manifest.get("k_b", -1)) != int(expected_k_b):
        raise ValueError(
            f"the selection artifact declares k_b={manifest.get('k_b')!r}, not {expected_k_b}: the "
            f"horizon is part of the arm's identity"
        )
    state, _ = restore_checkpoint(manager, template)
    return state, CheckpointIdentity(
        run_name=_run_name_from_root(ckpt_dir),
        step=int(manifest["step"]),
        root=str(ckpt_dir),
        source="selection",
        arm=str(manifest.get("arm")),
        k_b=int(manifest.get("k_b")),
        dev_metric=float(stored),
    )


# ---------------------------------------------------------------------------------------------
# Scoring: deployment's conventions, composed from the committed pieces.
# ---------------------------------------------------------------------------------------------


def anchor_sample_keys(seed: int, count: int) -> tuple:
    """Deployment's per-sample keys: ``key(seed)`` split SEQUENTIALLY, one split per sample.

    Not a per-index fold: the deployed loop threads ``rng`` through ``jax.random.split``, so sample i
    depends on how many samples preceded it. Reproducing the anchor means reproducing that.
    """
    rng = jax.random.key(int(seed))
    keys = []
    for _ in range(int(count)):
        rng, sample_rng = jax.random.split(rng)
        keys.append(sample_rng)
    return tuple(keys)


def initial_latents(key, z_video: jax.Array, z_i0: jax.Array) -> jax.Array:
    """Deployment's initial state: normal noise in the VIDEO LATENTS' dtype, frame 0 pinned."""
    z = jax.random.normal(key, z_video.shape, dtype=z_video.dtype)
    return apply_first_frame_pin(z, z_i0)


def evaluation_draw_key(name: str) -> jax.Array:
    """THE pinned initial-noise key for one example — a function of its NAME and nothing else.

    §3e scores each example three times on IDENTICAL noise, differing only in the actions fed. That
    is only true if the draw is keyed on the RECEIVER; keying the wrong-action row on the donor would
    change the noise between conditions and the gate would be measuring noise. Because the key
    depends on nothing but the name, "identical noise" is checkable from the published table
    (:func:`draw_key_digest`) rather than asserted in a comment.
    """
    tag = int.from_bytes(hashlib.sha256(str(name).encode("utf-8")).digest()[:4], "big")
    return jax.random.fold_in(jax.random.key(EVAL_NOISE_SEED), tag)


def draw_key_digest(key) -> str:
    """A digest of the key's own BITS, so a table records the draw it used rather than a name."""
    return hashlib.sha256(np.asarray(jax.random.key_data(key), dtype=np.uint32).tobytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class RolloutExecution:
    """What a rollout ACTUALLY did. Produced only by :func:`rollout_prediction`.

    The reviewer executed ``rollout_prediction(..., num_steps=1)`` and then certified the result as
    ``num_steps=25``: the horizon was a caller's word about a measurement. It now travels WITH the
    prediction, so a certificate and a benchmark row can only restate what ran.
    """

    z_pred: Any
    num_steps: int
    grid_size: int
    guide_scale: float
    draw_key_sha256: str


def rollout_prediction(
    *, velocity_fn, sigmas, timesteps, context, z_i0, z_video, key, guide_scale
) -> RolloutExecution:
    """The prediction a score is computed from — T3a's rollout, at the DEPLOYED horizon only.

    **There is no ``num_steps`` parameter.** The comparison exp_06 makes is against a baseline
    measured on the deployed 25-step grid, so a rollout at any other horizon is not a cheaper version
    of this measurement, it is a different one. The sigma grid is checked for the same reason: 25
    steps taken on a grid built for 10 is not a 25-step rollout.
    """
    sigmas = jnp.asarray(sigmas)
    timesteps = jnp.asarray(timesteps)
    if int(sigmas.shape[0]) != DEPLOYED_SAMPLING_STEPS + 1:
        raise ValueError(
            f"the deployed grid has {DEPLOYED_SAMPLING_STEPS + 1} sigmas (N+1, terminal 0), got "
            f"{int(sigmas.shape[0])}: a {DEPLOYED_SAMPLING_STEPS}-step rollout on another grid is a "
            f"different measurement and would be indistinguishable in the published table"
        )
    if int(timesteps.shape[0]) < DEPLOYED_SAMPLING_STEPS:
        raise ValueError(f"the grid carries {int(timesteps.shape[0])} timesteps, fewer than {DEPLOYED_SAMPLING_STEPS}")
    z_pred = cfg_rollout(
        initial_latents(key, z_video, z_i0),
        velocity_fn=velocity_fn,
        sigmas=sigmas,
        timesteps=timesteps,
        context=context,
        z_i0=z_i0,
        start=0,
        num_steps=DEPLOYED_SAMPLING_STEPS,
    )
    return RolloutExecution(
        z_pred=z_pred,
        num_steps=DEPLOYED_SAMPLING_STEPS,
        grid_size=int(sigmas.shape[0]),
        guide_scale=float(guide_scale),
        draw_key_sha256=draw_key_digest(key),
    )


def frame_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """VERBATIM copy of the deployed ``generate_wan_side_adapter._frame_ssim`` (plan §5-5).

    Copied rather than imported because importing that module pulls the full training stack; a drift
    tripwire asserts every line below still appears in the deployed file, so the copy cannot go stale
    silently. Callers get the refusal: :func:`summarize_samples` rejects a non-finite value instead of
    filtering it.
    """
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")
    vals = []
    for p, t in zip(pred, target):
        h, w = p.shape[:2]
        win = min(7, h, w)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return float("nan")
        vals.append(
            float(
                structural_similarity(
                    t,
                    p,
                    channel_axis=-1,
                    data_range=1.0,
                    win_size=win,
                )
            )
        )
    return float(np.mean(vals))


def sample_metrics(execution: RolloutExecution, z_video, *, decode_fn) -> dict:
    """One sample's latent and pixel metrics, in deployment's definitions.

    It takes an EXECUTION rather than a bare array, so the horizon that ran travels into the row and
    from there into the summary: there is no point at which a caller can restate it.

    ``decode_fn`` is deployment's ``pipeline._decode_latents_to_video(pipeline._denormalize_latents(x))``
    — the VAE seam, injected because it needs real weights. BOTH sides go through the same callable,
    so a decode that normalizes differently cannot flatter the prediction.
    """
    if not isinstance(execution, RolloutExecution):
        raise TypeError(
            "a sample is scored from the RolloutExecution that produced it, never from a bare array: "
            "the array does not know how many steps it took, and that is exactly what got restated"
        )
    z_pred = jnp.asarray(execution.z_pred)
    z_video = jnp.asarray(z_video)
    diff = z_pred.astype(jnp.float32) - z_video.astype(jnp.float32)
    pred0 = np.asarray(decode_fn(z_pred.astype(jnp.float32))[0], dtype=np.float32)
    gt0 = np.asarray(decode_fn(z_video.astype(jnp.float32))[0], dtype=np.float32)
    ssim = frame_ssim(pred0, gt0)
    if not math.isfinite(ssim):
        raise ValueError(
            "the SSIM for this sample is not finite: the deployed helper returns NaN when scikit-image "
            "is missing or the frames are too small to window, and a NaN that reaches a summary is a "
            "silently dropped sample rather than a measured one"
        )
    return {
        "latent_mse": float(jnp.mean(diff**2)),
        "pixel_mse": float(np.mean((pred0 - gt0) ** 2)),
        "pixel_mae": float(np.mean(np.abs(pred0 - gt0))),
        "ssim_avg": ssim,
        "num_steps": int(execution.num_steps),
        "draw_key_sha256": execution.draw_key_sha256,
    }


@dataclasses.dataclass(frozen=True)
class Measurement:
    """A digest-bound aggregate: WHAT was scored, on WHICH checkpoint, at WHICH horizon.

    Produced only by :func:`summarize_samples`, which is the only place that sees both the scored
    rows and the restored identity. Everything downstream — the anchor verdict, the certificate —
    reads this object, so provenance is derived once, where it is known.
    """

    payload: dict
    digest: str
    checkpoint: CheckpointIdentity

    @property
    def means(self) -> dict:
        return {metric: float(self.payload[metric]) for metric in ANCHOR_METRICS}

    @property
    def sample_names(self) -> list:
        return list(self.payload["sample_names"])

    @property
    def num_samples(self) -> int:
        return int(self.payload["num_samples"])

    @property
    def num_steps(self) -> int:
        return int(self.payload["num_steps"])

    @property
    def code_sha(self) -> str:
        return str(self.payload["code_sha"])

    @property
    def model_revision(self) -> str:
        return str(self.payload["model_revision"])


def summarize_samples(
    rows: Sequence[Mapping],
    *,
    checkpoint: CheckpointIdentity,
    code_sha: str,
    model_revision: str,
    test_manifest_path: str,
) -> Measurement:
    """Deployment's aggregation, minus its non-finite filter (the declared deviation, see module doc).

    The TEST screen runs HERE rather than beside the caller, so there is no route from sample rows to
    an anchor summary that skipped it, and the names it screened are the names the certificate later
    publishes. The horizon comes from the ROWS — every one of which was produced by a
    :class:`RolloutExecution` — and rows that executed different horizons are refused rather than
    averaged into a number no single rollout ever produced.
    """
    if not isinstance(checkpoint, CheckpointIdentity):
        raise TypeError(
            "a measurement is bound to the CheckpointIdentity a restore produced, never to a caller's "
            "mapping: the reviewer certified checkpoint {'run_name': 'some-other-run', 'step': 1}"
        )
    if not rows:
        raise ValueError("a summary over no samples is not a measurement")
    names = [str(row.get("name", "")) for row in rows]
    if not all(names):
        raise ValueError("every scored sample must carry its name: an unnamed sample cannot be screened")
    assert_no_test_examples(names, test_manifest_path=test_manifest_path)
    bad = [index for index, row in enumerate(rows) if not math.isfinite(float(row.get("ssim_avg", float("nan"))))]
    if bad:
        raise ValueError(
            f"samples {bad} have a non-finite ssim: refusing to average the ones that happened to work. "
            f"A missing SSIM backend produces NaN silently, and a mean over a subset is a different estimand."
        )
    horizons = {int(row.get("num_steps", -1)) for row in rows}
    if horizons != {DEPLOYED_SAMPLING_STEPS}:
        raise ValueError(
            f"these samples executed horizons {sorted(horizons)}; the deployed grid is "
            f"{DEPLOYED_SAMPLING_STEPS} steps. The horizon is taken from the rollouts that ran, so a "
            f"mixed or short summary is refused here rather than restated as 25 in a certificate."
        )
    payload = {
        "protocol": MEASUREMENT_PROTOCOL,
        "mean_latent_mse": float(np.mean([float(row["latent_mse"]) for row in rows])),
        "mean_pixel_mse": float(np.mean([float(row["pixel_mse"]) for row in rows])),
        "mean_ssim": float(np.mean([float(row["ssim_avg"]) for row in rows])),
        "num_samples": len(rows),
        "num_steps": DEPLOYED_SAMPLING_STEPS,
        "checkpoint": checkpoint.payload(),
        "checkpoint_step": int(checkpoint.step),
        "sample_names": names,
        "code_sha": str(code_sha),
        "model_revision": str(model_revision),
    }
    return Measurement(payload=payload, digest=_digest(payload), checkpoint=checkpoint)


def assert_no_test_examples(names: Sequence[str], *, test_manifest_path: str) -> None:
    """The anchor's four samples come from the val DIRECTORY in file order, and exp_04 drew BOTH
    DEV-64 and TEST-64 from that directory — so "the first four records" can legitimately land on a
    TEST example. Names are read here only to REFUSE; nothing about TEST is ever scored.
    """
    from maxdiffusion.null_adapter_manifest_io import load_manifest

    digest = hashlib.sha256(storage_read_bytes(test_manifest_path)).hexdigest()
    if digest != J0_TEST64_SHA256:
        raise ValueError(
            f"{test_manifest_path} is not exp_04's published TEST-64 manifest ({J0_TEST64_SHA256}, found "
            f"{digest}): this file is read to REFUSE, so an unpinned one would defang the screen"
        )
    reserved = {str(row["name"]) for row in load_manifest(test_manifest_path)["rows"]}
    intruders = sorted(set(map(str, names)) & reserved)
    if intruders:
        raise ValueError(
            f"these anchor samples are TEST-64 examples: {intruders}. TEST is confirmation only "
            f"(plan §3d); the anchor must be measured on samples that are not held out."
        )


# ---------------------------------------------------------------------------------------------
# Publication: frozen once, adopted on republication, never re-derived (issue #10).
# ---------------------------------------------------------------------------------------------


def _digest(payload: Mapping) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def publish_certificate(path: str, payload: Mapping) -> dict:
    """Write once. An identical republication is ADOPTED; a different one is refused.

    Issue #10's rule, in the small: a queue retry re-runs a phase, and an artifact that silently
    rewrites itself makes the second run's number the record of the first run's claim.
    """
    published = {**dict(payload), "sha256": _digest(payload)}
    if storage_exists(path):
        existing = json.loads(storage_read_bytes(path).decode("utf-8"))
        # Adopt only a file that is INTERNALLY consistent: comparing two self-declared digests would
        # adopt a hand-edited payload whose header still carries the original hash.
        if _digest(existing.get("payload") or {}) != existing.get("sha256"):
            raise ValueError(
                f"{path}: the published digest {existing.get('sha256')!r} does not describe its own "
                f"payload — the artifact was edited after publication and cannot be adopted"
            )
        if existing.get("sha256") != published["sha256"]:
            raise ValueError(
                f"{path} was already published with digest {existing.get('sha256')!r}; this run produced "
                f"{published['sha256']!r}. A frozen artifact is adopted, never rewritten."
            )
        return {**existing["payload"], "sha256": existing["sha256"]}
    storage_write_bytes(
        path, json.dumps({"payload": dict(payload), "sha256": published["sha256"]}, sort_keys=True).encode("utf-8")
    )
    return published


# ---------------------------------------------------------------------------------------------
# The per-example table, as an ARTIFACT (review pass 2, T5b MAJOR 4: the gates accepted naked
# scalar tables, so a future scorer could key wrong-action noise on the donor and pass every test).
# ---------------------------------------------------------------------------------------------

#: What a table row must carry beyond its numbers. Each one is an IDENTITY the gate cross-checks:
#: which example's actions were fed, their digest, and the digest of the noise key that was drawn.
SCORE_ROW_FIELDS = ("ssim", "mse", "actions_from", "actions_sha256", "draw_key_sha256", "num_steps")


@dataclasses.dataclass(frozen=True)
class ScoreTable:
    """One arm × one condition, over one cohort — digest-bound, with per-row identities.

    A gate that reads ``{name: {"ssim": …}}`` can only check the numbers it is handed. This carries
    the cohort and its manifest digest, the arm, the condition, the restored checkpoint, the horizon
    that executed, and per example the actions actually fed and the noise key actually drawn — so
    "true and wrong differ only in their actions" is verified from the artifact.
    """

    payload: dict
    digest: str

    @property
    def cohort(self) -> str:
        return str(self.payload["cohort"])

    @property
    def manifest_sha256(self) -> str:
        return str(self.payload["manifest_sha256"])

    @property
    def condition(self) -> str:
        return str(self.payload["condition"])

    @property
    def arm(self) -> str:
        return str(self.payload["arm"])

    @property
    def num_steps(self) -> int:
        return int(self.payload["num_steps"])

    @property
    def checkpoint(self) -> dict:
        return dict(self.payload["checkpoint"])

    @property
    def derangement_sha256(self):
        return self.payload.get("derangement_sha256")

    @property
    def rows(self) -> dict:
        return {name: (dict(row) if isinstance(row, Mapping) else row) for name, row in self.payload["rows"].items()}

    @property
    def names(self) -> list:
        return list(self.payload["rows"])

    def scalar_table(self) -> dict:
        """The ``{name: {"ssim", "mse"}}`` view exp_04's decision function consumes. Read-only."""
        return {name: {"ssim": row["ssim"], "mse": row["mse"]} for name, row in self.payload["rows"].items()}


def build_score_table(
    *,
    rows: Mapping,
    cohort,
    condition: str,
    arm: str,
    checkpoint: CheckpointIdentity,
    num_steps: int,
    derangement_sha256: str | None = None,
    allow_incomplete: bool = False,
) -> ScoreTable:
    """The table constructor. Every row must carry its identities; the horizon must be the deployed one.

    ``allow_incomplete`` exists because a coverage FAILURE is a real outcome that must still produce a
    quotable artifact (the gate then penalizes the claim). It never relaxes an identity.
    """
    if not isinstance(checkpoint, CheckpointIdentity):
        raise TypeError("a score table is bound to the CheckpointIdentity a restore produced, not to a mapping")
    if int(num_steps) != DEPLOYED_SAMPLING_STEPS:
        raise ValueError(
            f"tables are measured on the deployed {DEPLOYED_SAMPLING_STEPS}-step rollout, got {num_steps!r}"
        )
    names = list(cohort.names)
    if not allow_incomplete and sorted(map(str, rows)) != sorted(names):
        raise ValueError(
            f"the table does not cover the {cohort.cohort} cohort exactly; symmetric difference "
            f"{sorted(set(names) ^ set(map(str, rows)))}"
        )
    if set(map(str, rows)) - set(names):
        raise ValueError(f"the table carries examples outside the cohort: {sorted(set(map(str, rows)) - set(names))}")
    clean = {}
    for name, row in rows.items():
        missing = [field for field in SCORE_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"row {name!r} carries no {missing}: a table row without its identities is a claim")
        if int(row["num_steps"]) != DEPLOYED_SAMPLING_STEPS:
            raise ValueError(f"row {name!r} executed {row['num_steps']} steps, not {DEPLOYED_SAMPLING_STEPS}")
        clean[str(name)] = {
            "ssim": None if row["ssim"] is None else float(row["ssim"]),
            "mse": None if row["mse"] is None else float(row["mse"]),
            "actions_from": None if row["actions_from"] is None else str(row["actions_from"]),
            "actions_sha256": str(row["actions_sha256"]),
            "draw_key_sha256": str(row["draw_key_sha256"]),
            "num_steps": int(row["num_steps"]),
        }
    payload = {
        "protocol": SCORE_TABLE_PROTOCOL,
        "cohort": str(cohort.cohort),
        "manifest_path": str(cohort.manifest_path),
        "manifest_sha256": str(cohort.manifest_sha256),
        "condition": str(condition),
        "arm": str(arm),
        "checkpoint": checkpoint.payload(),
        "num_steps": int(num_steps),
        "example_count": len(clean),
        "rows": clean,
    }
    if derangement_sha256 is not None:
        payload["derangement_sha256"] = str(derangement_sha256)
    return ScoreTable(payload=payload, digest=_digest(payload))


def publish_score_table(path: str, table: ScoreTable) -> dict:
    if not isinstance(table, ScoreTable):
        raise TypeError("only a built ScoreTable is publishable; a mapping has no identities to publish")
    return publish_certificate(path, table.payload)


def load_score_table(path: str) -> ScoreTable:
    """Read a published table back, digest-verified and schema-validated, as the artifact it is."""
    payload = _load_payload(path, protocol=SCORE_TABLE_PROTOCOL)
    return build_score_table(
        rows=payload["rows"],
        cohort=_LoadedCohort(payload["cohort"], payload["manifest_path"], payload["manifest_sha256"], payload["rows"]),
        condition=payload["condition"],
        arm=payload["arm"],
        checkpoint=CheckpointIdentity(**payload["checkpoint"]),
        num_steps=payload["num_steps"],
        derangement_sha256=payload.get("derangement_sha256"),
        allow_incomplete=True,
    )


@dataclasses.dataclass(frozen=True)
class _LoadedCohort:
    """The cohort identity a published table already carries — enough to rebuild the artifact.

    Deliberately NOT a substitute for :class:`DevCohort`: it cannot draw, cannot read, and every gate
    re-checks a loaded table's cohort/digest against the manifest the run actually loaded.
    """

    cohort: str
    manifest_path: str
    manifest_sha256: str
    _rows: Mapping

    @property
    def names(self) -> list:
        return list(self._rows)


def freeze_benchmark_row(path: str, *, table: ScoreTable) -> dict:
    """Plan §3c(b): the deployed checkpoint's DEV-64 table, frozen as the row every later table carries.

    Derived from a bound :class:`ScoreTable` — cohort, manifest digest, checkpoint, horizon and the
    per-example values all come from the artifact that measured them. The previous signature took the
    table, the checkpoint, the code SHA and the horizon as four independent caller assertions.
    """
    if not isinstance(table, ScoreTable):
        raise TypeError(
            "the benchmark row is frozen from a ScoreTable, never from a caller's per-example mapping: "
            "the row every later table compares against must know which checkpoint produced it"
        )
    if table.condition != "true":
        raise ValueError(
            f"the benchmark row is the deployed checkpoint under TRUE actions, got condition "
            f"{table.condition!r}: a baseline measured under wrong or zero actions is not the baseline"
        )
    values = {name: row["ssim"] for name, row in table.rows.items()}
    unusable = [name for name, value in values.items() if value is None or not math.isfinite(float(value))]
    if unusable:
        raise ValueError(f"every benchmark value must be finite; {unusable} are not")
    return publish_certificate(
        path,
        {
            "protocol": BENCHMARK_PROTOCOL,
            "cohort": table.cohort,
            "manifest_path": str(table.payload["manifest_path"]),
            "manifest_sha256": table.manifest_sha256,
            "example_count": len(values),
            "per_example": {name: float(value) for name, value in values.items()},
            "mean_ssim": float(np.mean([float(value) for value in values.values()])),
            "checkpoint": table.checkpoint,
            "num_steps": table.num_steps,
            "score_table_sha256": table.digest,
        },
    )


def load_benchmark_row(path: str) -> dict:
    """Read a frozen row: digest-verified AND semantically validated.

    The digest only proves nobody edited the file after publication. What a later table needs to know
    is that this row IS a benchmark row: the right protocol, a count that matches its own table, a
    mean that matches its own values, and the deployed horizon (T5a review MAJOR 4).
    """
    payload = _load_payload(path, protocol=BENCHMARK_PROTOCOL)
    values = payload.get("per_example")
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{path}: a benchmark row carries a per-example table")
    if int(payload.get("example_count", -1)) != len(values):
        raise ValueError(
            f"{path}: declares {payload.get('example_count')} examples and carries {len(values)}: the "
            f"count and the table must be the same fact"
        )
    if int(payload.get("num_steps", -1)) != DEPLOYED_SAMPLING_STEPS:
        raise ValueError(
            f"{path}: the benchmark row records horizon {payload.get('num_steps')!r}, not the deployed "
            f"{DEPLOYED_SAMPLING_STEPS}: it cannot be the row later tables compare against"
        )
    numbers = [float(value) for value in values.values()]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"{path}: the frozen row carries non-finite values")
    if not math.isclose(float(payload["mean_ssim"]), float(np.mean(numbers)), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"{path}: the recorded mean {payload['mean_ssim']!r} is not the mean of its own table "
            f"({float(np.mean(numbers))!r})"
        )
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("run_name"):
        raise ValueError(f"{path}: a benchmark row names the checkpoint it measured")
    return {**dict(payload), "sha256": _digest(payload)}


def _load_payload(path: str, *, protocol: str) -> dict:
    """Read a published artifact, re-verify its digest, and check it is the protocol asked for."""
    if not storage_exists(path):
        raise ValueError(
            f"there is no {protocol} artifact at {path}. A phase READS its prerequisites rather than "
            f"assuming them, so a missing one stops the job here instead of after the model is loaded."
        )
    stored = json.loads(storage_read_bytes(path).decode("utf-8"))
    payload = stored.get("payload") if isinstance(stored, Mapping) else None
    if not isinstance(payload, Mapping) or _digest(payload) != stored.get("sha256"):
        raise ValueError(
            f"{path}: the recorded digest {stored.get('sha256') if isinstance(stored, Mapping) else None!r} does "
            f"not describe its payload — the artifact has been edited since it was published"
        )
    if payload.get("protocol") != protocol:
        raise ValueError(
            f"{path} is not a {protocol} artifact: its protocol is {payload.get('protocol')!r}. A payload "
            f"that merely carries the right-looking keys is not the artifact it imitates."
        )
    return dict(payload)


def load_certificate(path: str, *, protocol: str) -> dict:
    """The public spelling of :func:`_load_payload`, for the phases and the gates module."""
    return _load_payload(path, protocol=protocol)


# ---------------------------------------------------------------------------------------------
# The device seam — the ONLY thing here that needs a TPU and real weights.
# ---------------------------------------------------------------------------------------------


class DeviceBackend:
    """Real weights and a real VAE, composed. **No identity passes through it.**

    ``velocity_for(params, actions, adapter_enabled)`` returns the ``velocity_fn`` T3a's rollout
    consumes; ``decode_fn`` is deployment's
    ``pipeline._decode_latents_to_video(pipeline._denormalize_latents(x))``. Everything else about a
    measurement — which example, whose actions, which noise key, which checkpoint, which horizon —
    is derived by the caller from the cohort, the derangement artifact and the restore, so a backend
    that returned any number it liked still could not misdescribe WHAT was measured. That is the
    boundary the Planner drew: device work may be injected; orchestration may not.
    """

    def __init__(
        self,
        *,
        velocity_for,
        decode_fn,
        sigmas,
        timesteps,
        context,
        guide_scale,
        template=None,
        params=None,
        scope=None,
    ):
        self.velocity_for = velocity_for
        self.decode_fn = decode_fn
        self.sigmas = sigmas
        self.timesteps = timesteps
        self.context = context
        self.guide_scale = float(guide_scale)
        self.template = template
        self.params = params
        #: ``() -> iterable of context managers`` entered around every score. The Wan blocks call
        #: ``with_sharding_constraint``, which needs a mesh in context, and the deployed evaluator
        #: wraps its rollout in ``with mesh, axis_rules(...)`` for exactly that reason.
        self.scope = scope

    def bound(self, params) -> "DeviceBackend":
        """The same device, with the restored adapter parameters bound. Immutable, so a phase cannot
        score half its cohort with one checkpoint and half with another."""
        return DeviceBackend(
            velocity_for=self.velocity_for,
            decode_fn=self.decode_fn,
            sigmas=self.sigmas,
            timesteps=self.timesteps,
            context=self.context,
            guide_scale=self.guide_scale,
            template=self.template,
            params=params,
            scope=self.scope,
        )

    def score(self, *, z_i0, z_video, actions, key, adapter_enabled: bool = True):
        import contextlib

        if self.params is None:
            raise ValueError("this backend has no restored parameters bound; call .bound(params) after the restore")
        with contextlib.ExitStack() as stack:
            for manager in () if self.scope is None else self.scope():
                stack.enter_context(manager)
            velocity_fn = self.velocity_for(self.params, actions, adapter_enabled)
            execution = rollout_prediction(
                velocity_fn=velocity_fn,
                sigmas=self.sigmas,
                timesteps=self.timesteps,
                context=self.context,
                z_i0=z_i0,
                z_video=z_video,
                key=key,
                guide_scale=self.guide_scale,
            )
        return execution, sample_metrics(execution, z_video, decode_fn=self.decode_fn)


def load_device_backend(config):  # pragma: no cover -- real Wan weights, a real VAE, a real device
    """Build the backend from the deployed pipeline: transformer, VAE, null context, sigma grid.

    This is the one function in exp_06's evaluator that cannot run without weights on a device. It is
    IMPLEMENTED rather than stubbed — the failure an operator sees here is a missing model or a
    missing device, not missing code.
    """
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning

    from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid
    from maxdiffusion.pos_rollout_step import build_cfg_velocity_fn
    from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import WanTI2VSideAdapterTrainer

    trainer = WanTI2VSideAdapterTrainer(config)
    pipeline = trainer._load_wan_pipeline()
    mesh = pipeline.mesh
    null_context = trainer._compute_null_context(pipeline, mesh)
    adapters = trainer._build_adapters(pipeline.transformer)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        make_velocity_fn, adapter_params = build_cfg_velocity_fn(pipeline.transformer, adapters)
        frozen = nnx.split(pipeline.transformer, nnx.Param, ...)

    def velocity_for(params, actions, adapter_enabled):
        if adapter_enabled:
            return make_velocity_fn(params, actions=actions, guide_scale=float(config.side_adapter_guide_scale))
        model = nnx.merge(*frozen)

        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            return model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=True,
            )

        return velocity_fn

    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=int(config.side_adapter_sampling_steps),
        flow_shift=float(config.flow_shift),
        sigma_min=0.0,
        sigma_max=1.0,
        num_train_timesteps=int(config.num_train_timesteps),
    )
    from maxdiffusion.pos_rollout_loop import RolloutTrainState

    return DeviceBackend(
        velocity_for=velocity_for,
        decode_fn=lambda x: pipeline._decode_latents_to_video(pipeline._denormalize_latents(x)),
        sigmas=sigmas,
        timesteps=timesteps,
        context=null_context,
        guide_scale=float(config.side_adapter_guide_scale),
        template=RolloutTrainState(params=adapter_params, opt_state=None, step=0),
        # The Wan blocks constrain their activations' sharding, so the mesh must be in context for
        # every forward -- the deployed evaluator wraps its rollout in exactly this pair.
        scope=lambda: (mesh, nn_partitioning.axis_rules(config.logical_axis_rules)),
    )


# ---------------------------------------------------------------------------------------------
# The protocol: four phases, in order, and the order is enforced HERE (plan §3c).
# ---------------------------------------------------------------------------------------------

_ATTEMPT_ROOT = re.compile(r"^eval_(?P<phase>anchor|benchmark|gates|confirm)_(?P<attempt>.+)$")


def protocol_root_for(artifact_root: str, *, phase: str) -> str:
    """The RUN-level root the phases share, derived from this attempt's own artifact root.

    Issue #13 gives every phase its own attempt-scoped root (``…/<run>/eval_<phase>_<attempt>``), so
    a later phase cannot guess an earlier one's directory — but the protocol's cross-phase artifacts
    (the anchor certificate, the frozen benchmark row, the DEV certificate) must live somewhere both
    can name. That somewhere is one level up, and the derivation FAILS CLOSED: a root that is not
    attempt-scoped for the phase being run is refused rather than trimmed hopefully.
    """
    trimmed = str(artifact_root).rstrip("/")
    head, _, last = trimmed.rpartition("/")
    matched = _ATTEMPT_ROOT.match(last)
    if not head or not matched or matched.group("phase") != str(phase):
        raise ValueError(
            f"{artifact_root!r} is not an attempt-scoped root for phase {phase!r}: exp_06 publishes into "
            f"<run_root>/eval_{phase}_<attempt> (issue #13), and the protocol's shared artifacts live in "
            f"<run_root>. Refusing to guess where those are."
        )
    return head


def anchor_certificate_path(protocol_root: str) -> str:
    return f"{str(protocol_root).rstrip('/')}/anchor_certificate.json"


def benchmark_row_path(protocol_root: str) -> str:
    return f"{str(protocol_root).rstrip('/')}/benchmark_row.json"


def dev_certificate_path(protocol_root: str) -> str:
    return f"{str(protocol_root).rstrip('/')}/dev_certificate.json"


def require_anchor(protocol_root: str) -> dict:
    """Plan §3c: nothing new is measured until the wiring proof has passed. Enforced in code.

    A launcher can be invoked with the phases in any order, or with one phase retried alone. What
    makes the ordering real is that every new-arm phase must LOAD the anchor's own certificate and
    find it reproduced — on the historical run, at the historical step, on the deployed grid.
    """
    path = anchor_certificate_path(protocol_root)
    if not storage_exists(path):
        raise ValueError(
            f"there is no anchor certificate at {path}: plan §3c runs the anchor protocol BEFORE any "
            f"new arm is scored, because a +0.05 paired ΔSSIM measured with an uncalibrated evaluator "
            f"is a statement about the evaluator. Run POS_EVAL_PHASE=anchor first."
        )
    certificate = load_certificate(path, protocol=ANCHOR_PROTOCOL)
    if not certificate.get("reproduced"):
        raise ValueError(
            f"{path}: the anchor did not reproduce ({certificate.get('failed')}), so this evaluation path "
            f"has not been shown to compute what the deployed path computed. No arm may be scored with it."
        )
    checkpoint = certificate.get("checkpoint") or {}
    if checkpoint.get("run_name") != HISTORICAL_ANCHOR.run_name or int(checkpoint.get("step", -1)) != (
        HISTORICAL_ANCHOR.checkpoint_step
    ):
        raise ValueError(f"{path}: the certificate names checkpoint {checkpoint!r}, not the historical anchor's")
    if int(certificate.get("num_steps", -1)) != DEPLOYED_SAMPLING_STEPS:
        raise ValueError(f"{path}: the anchor was certified at horizon {certificate.get('num_steps')!r}")
    return certificate


def _required(config, key: str) -> str:
    from maxdiffusion.run_wan_null_inversion import optional_config_value

    value = str(optional_config_value(config, key, "") or "")
    if not value:
        raise ValueError(
            f"{key} is empty. This phase cannot run without it, and defaulting it would publish a "
            f"certificate about a job nobody configured."
        )
    return value


def _provenance(config) -> tuple[str, str]:
    """``(code_sha, model_revision)`` — refused when absent: an unprovenanced certificate is a claim."""
    import os

    from maxdiffusion.run_wan_null_inversion import optional_config_value

    code_sha = str(optional_config_value(config, "code_sha", "") or os.environ.get("COMMIT", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError(
            f"code_sha is {code_sha!r}: every certificate exp_06 publishes carries the 40-hex commit the "
            f"worker ran, and a table that cannot say which code produced it is not reproducible"
        )
    model = str(optional_config_value(config, "pretrained_model_name_or_path", "") or "")
    if not model:
        raise ValueError("pretrained_model_name_or_path is empty: a measurement names the model it measured")
    return code_sha, model


def read_anchor_samples(config, *, count: int, start_index: int = 0) -> list:
    """The anchor's samples, from the val DIRECTORY in file order — deployment's own reading.

    The historical protocol read ``num_eval_videos`` records from ``eval_data_dir`` starting at
    ``validation_start_index``; reproducing its number means reproducing that. exp_06's YAML empties
    ``eval_data_dir`` on purpose (selection binds to the DEV manifest, never a directory — plan §3d),
    so the anchor phase requires the operator to name the historical validation directory explicitly.
    """
    import tensorflow as tf

    from maxdiffusion.run_wan_null_inversion import optional_config_value

    # NOT ``getattr(config, key, default)`` — issue #11: ``HyperParameters.__getattr__`` raises
    # ValueError, which getattr's default never swallows. It has killed two TPU jobs this campaign.
    data_dir = str(optional_config_value(config, "eval_data_dir", "") or "")
    if not data_dir:
        raise ValueError(
            "eval_data_dir is empty. The anchor is reproduced on the four val-directory records the "
            "deployed run scored, so the anchor phase must be given that directory explicitly "
            "(eval_data_dir=gs://…). exp_06's training config empties it by design (plan §3d)."
        )
    files = sorted(tf.io.gfile.glob(data_dir.rstrip("/") + "/*.tfrecord")) or sorted(
        tf.io.gfile.glob(data_dir.rstrip("/") + "/*.tfrecord-*")
    )
    if not files:
        raise FileNotFoundError(f"no TFRecord shards under {data_dir}")
    features = {
        "name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "z_i0": tf.io.FixedLenFeature([], tf.string),
        "z_video": tf.io.FixedLenFeature([], tf.string),
        "actions": tf.io.FixedLenFeature([], tf.string),
    }
    channels, frames = int(config.latent_channels), int(config.latent_frames)
    height, width = int(config.latent_height), int(config.latent_width)
    action_len, action_dim = int(config.action_len), int(config.action_dim)
    dataset = tf.data.TFRecordDataset(files).map(lambda raw: tf.io.parse_single_example(raw, features))
    dataset = dataset.skip(max(0, int(start_index))).take(int(count))
    samples = []
    for index, raw in enumerate(dataset.as_numpy_iterator()):
        samples.append(
            {
                "name": raw["name"].decode("utf-8") or f"sample_{int(start_index) + index:06d}",
                "z_i0": np.frombuffer(raw["z_i0"], np.float16).reshape(channels, 1, height, width).astype(np.float32),
                "z_video": (
                    np.frombuffer(raw["z_video"], np.float16)
                    .reshape(channels, frames, height, width)
                    .astype(np.float32)
                ),
                "actions": np.frombuffer(raw["actions"], np.float32).reshape(action_len, action_dim),
            }
        )
    if len(samples) != int(count):
        raise ValueError(f"{data_dir} yielded {len(samples)} records from index {start_index}, not {count}")
    return samples


def run_anchor_phase(config, *, backend, artifact_root: str, protocol_root: str) -> dict:
    """Restore the deployed checkpoint, score its four recorded samples, and publish the verdict."""
    from maxdiffusion.run_wan_null_inversion import optional_config_value

    code_sha, model_revision = _provenance(config)
    state, identity = restore_anchor_checkpoint(_required(config, "checkpoint_dir"), backend.template)
    scoring = backend.bound(state.params)
    samples = read_anchor_samples(
        config,
        count=HISTORICAL_ANCHOR.num_samples,
        start_index=int(optional_config_value(config, "validation_start_index", 0) or 0),
    )
    keys = anchor_sample_keys(int(optional_config_value(config, "validation_seed", 0) or 0), len(samples))
    rows = []
    for sample, key in zip(samples, keys):
        z_i0 = jnp.asarray(sample["z_i0"])[None]
        z_video = jnp.asarray(sample["z_video"])[None]
        _, metrics = scoring.score(z_i0=z_i0, z_video=z_video, actions=jnp.asarray(sample["actions"])[None], key=key)
        rows.append({"name": sample["name"], **metrics})
    measurement = summarize_samples(
        rows,
        checkpoint=identity,
        code_sha=code_sha,
        model_revision=model_revision,
        test_manifest_path=_required(config, "pos_test_manifest"),
    )
    verdict = reproduce_anchor(measurement)
    certificate = publish_certificate(anchor_certificate_path(protocol_root), anchor_certificate(verdict))
    publish_certificate(f"{artifact_root}/anchor_attempt.json", anchor_certificate(verdict))
    return {"phase": "anchor", "certificate": certificate, "verdict": verdict, "measurement": measurement}


def _score_cohort_tables(config, *, backend, cohort, conditions, arm: str, identity, gates) -> dict:
    """Produce this arm's condition tables through the ONE producer (plan §3e's plan, consumed)."""
    derangement = gates.cohort_derangement(cohort)
    plan = gates.action_use_plan(cohort, derangement=derangement)
    del config
    tables = {
        condition: gates.score_condition_table(
            plan,
            condition=condition,
            cohort=cohort,
            derangement=derangement,
            checkpoint=identity,
            arm=arm,
            backend=backend,
        )
        for condition in conditions
    }
    return {"derangement": derangement, "plan": plan, "tables": tables}


def run_benchmark_phase(config, *, backend, artifact_root: str, protocol_root: str) -> dict:
    """Plan §3c(b): freeze the deployed checkpoint's DEV-64 table — the row every table carries."""
    from maxdiffusion import pos_rollout_gates as gates

    state, identity = restore_anchor_checkpoint(_required(config, "checkpoint_dir"), backend.template)
    cohort = load_dev_cohort(_required(config, "pos_dev_manifest"))
    produced = _score_cohort_tables(
        config,
        backend=backend.bound(state.params),
        cohort=cohort,
        conditions=("true",),
        arm="historical",
        identity=identity,
        gates=gates,
    )
    table = produced["tables"]["true"]
    publish_score_table(f"{artifact_root}/tables/historical_true.json", table)
    row = freeze_benchmark_row(benchmark_row_path(protocol_root), table=table)
    return {"phase": "benchmark", "benchmark_row": row, "table": table}


def run_gates_phase(config, *, backend, artifact_root: str, protocol_root: str) -> dict:
    """Plan §3c/§3e on DEV-64: the paired primary gate, the action-use battery, and the certificate."""
    from maxdiffusion import pos_rollout_gates as gates

    cohort = load_dev_cohort(_required(config, "pos_dev_manifest"))
    arm_state, arm_identity = restore_from_report(
        _required(config, "checkpoint_dir"), backend.template, report_path=_required(config, "pos_run_report")
    )
    control_state, control_identity = restore_from_report(
        _required(config, "pos_control_checkpoint_dir"),
        backend.template,
        report_path=_required(config, "pos_control_run_report"),
    )
    if control_identity.root == arm_identity.root:
        raise ValueError(
            f"both arms restored from {arm_identity.root}: matched-C0 exists to be a DIFFERENT run, and "
            f"comparing a checkpoint with itself would report a delta of exactly zero as a finding"
        )
    conditions = ("true", "wrong", "zero", "adapter_disabled")
    produced = _score_cohort_tables(
        config,
        backend=backend.bound(arm_state.params),
        cohort=cohort,
        conditions=conditions,
        arm="rollout",
        identity=arm_identity,
        gates=gates,
    )
    control = _score_cohort_tables(
        config,
        backend=backend.bound(control_state.params),
        cohort=cohort,
        conditions=("true", "wrong", "zero"),
        arm="control",
        identity=control_identity,
        gates=gates,
    )
    for label, tables in (("rollout", produced["tables"]), ("control", control["tables"])):
        for condition, table in tables.items():
            publish_score_table(f"{artifact_root}/tables/{label}_{condition}.json", table)
    certificate = gates.dev_certificate(
        dev_certificate_path(protocol_root),
        rollout=produced["tables"]["true"],
        control=control["tables"]["true"],
        cohort=cohort,
    )
    report = gates.action_use_report(
        cohort,
        derangement=produced["derangement"],
        tables=produced["tables"],
        control_tables=control["tables"],
    )
    publish_certificate(f"{artifact_root}/action_use_report.json", gates.report_payload(report))
    return {"phase": "gates", "certificate": certificate, "action_use": report}


def run_confirm_phase(config, *, backend, artifact_root: str, protocol_root: str) -> dict:
    """The single TEST door: both gates, on an independently derived TEST derangement (plan §3e)."""
    from maxdiffusion import pos_rollout_gates as gates

    # The prerequisite is checked BEFORE anything expensive: review pass 3's finding was that no phase
    # transported its prerequisites, so a job could load a model and score a cohort before discovering
    # that the DEV gate it depends on was never issued.
    gates.load_dev_certificate(dev_certificate_path(protocol_root))
    cohort = gates.load_test_cohort(_required(config, "pos_test_manifest"))
    arm_state, arm_identity = restore_from_report(
        _required(config, "checkpoint_dir"), backend.template, report_path=_required(config, "pos_run_report")
    )
    control_state, control_identity = restore_from_report(
        _required(config, "pos_control_checkpoint_dir"),
        backend.template,
        report_path=_required(config, "pos_control_run_report"),
    )
    produced = _score_cohort_tables(
        config,
        backend=backend.bound(arm_state.params),
        cohort=cohort,
        conditions=("true", "wrong", "zero", "adapter_disabled"),
        arm="rollout",
        identity=arm_identity,
        gates=gates,
    )
    control = _score_cohort_tables(
        config,
        backend=backend.bound(control_state.params),
        cohort=cohort,
        conditions=("true", "wrong", "zero"),
        arm="control",
        identity=control_identity,
        gates=gates,
    )
    confirmation = gates.confirm_on_test(
        dev_certificate_path(protocol_root),
        test_cohort=cohort,
        derangement=produced["derangement"],
        tables=produced["tables"],
        control_tables=control["tables"],
    )
    publish_certificate(f"{artifact_root}/test_confirmation.json", gates.confirmation_payload(confirmation))
    return {"phase": "confirm", "confirmation": confirmation}


_PHASE_RUNNERS = {
    "anchor": run_anchor_phase,
    "benchmark": run_benchmark_phase,
    "gates": run_gates_phase,
    "confirm": run_confirm_phase,
}


def run_evaluation(config, *, backend=None) -> dict:
    """Dispatch plan §3c's protocol. The phase names WHAT runs; this function enforces the ORDER."""
    from maxdiffusion.run_wan_null_inversion import optional_config_value

    phase = str(optional_config_value(config, "pos_eval_phase", "") or "")
    if phase not in EVAL_PHASES:
        raise ValueError(
            f"{phase!r} is not a phase this evaluator wires; the protocol is {list(EVAL_PHASES)} in that "
            f"order (plan §3c). An empty phase is a wrapper bug, not a request for the default."
        )
    artifact_root = str(optional_config_value(config, "base_output_directory", "") or "").rstrip("/")
    protocol_root = protocol_root_for(artifact_root, phase=phase)
    if phase != "anchor":
        require_anchor(protocol_root)
    if backend is None:  # pragma: no cover -- real weights
        backend = load_device_backend(config)
    return _PHASE_RUNNERS[phase](config, backend=backend, artifact_root=artifact_root, protocol_root=protocol_root)


def main(argv):  # pragma: no cover -- the shell entry point
    """``python src/maxdiffusion/eval_wan_pos_rollout.py <config.yml> key=value ...``."""
    from maxdiffusion import pyconfig

    pyconfig.initialize(argv)
    return run_evaluation(pyconfig.config)


if __name__ == "__main__":  # pragma: no cover -- the shell entry point
    import sys

    main(sys.argv)
