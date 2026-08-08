"""exp_06 T7: the M1 fit probe — measure the cost, and authorize only what was measured (§4-P1).

**One contract: M1 measures what a training job would cost, and authorizes only the cells it
measured.** The second clause is the round. A fit probe that produces a table nobody is obliged to
consult is a document; a probe whose verdict is the only thing that can unlock a training cell is a
gate, and :func:`assert_cell_authorized` is called from ``WanPosRolloutTrainer.start_training``
before anything expensive — so a cell M1 never measured has no route to a training run.

**Review pass 3 rebuilt this module around one rule: STAMPED ≠ BOUND.** The first version took the
measurements from one argument and the SHA / model / device / geometry from another, so its digest
proved only that the caller's *claims* were not edited afterwards; the reviewer published an
authorization carrying a wrong SHA, a foreign model and the wrong device kind, and production
accepted it — because production never passed a context to compare against. Four structural changes:

1. **Provenance is DERIVED, never supplied.** :func:`derive_probe_context` reads the running
   program: the code SHA from this checkout's git HEAD (cross-checked against the launcher's
   ``COMMIT``), the model revision from the *resolved local snapshot*, the device kind and count from
   ``jax.devices()``, the tensor geometry and the footprint-bearing recipe from the config the job is
   actually running. :func:`publish_authorization` takes NO provenance arguments at all.
2. **A measurement is bound to the context it was measured under.** ``CellMeasurement`` carries the
   context digest; :func:`build_evidence` refuses any measurement whose digest is not the context
   being published. A number measured elsewhere cannot be published here.
3. **The cell carries the ARM** — ``(arm, microbatch, k_b)`` — and the context carries the
   ``recipe_fingerprint`` of everything else that decides the footprint (dtypes, remat, attention,
   parallelism, logical batch, sampling geometry, adapter shape). A matched-C0 measurement no longer
   authorizes R-B: they are different forward/backward graphs and now different cells.
4. **Training derives its own context independently** and requires exact binding, so the comparison
   is between two derivations of the same running program rather than between a claim and nothing.

**Repeated trials aggregate CONSERVATIVELY (T7-3).** The reviewer published one cell twice — once
fitting, once at 96.9% HBM with a reservation failure — and it appeared in both lists and was
authorized. :func:`aggregate_trials` takes the WORST of every trial (max peak, max step time, summed
reservation failures, min capacity), so a cell that missed once is a cell that missed; the published
lists are unique and disjoint, and :func:`load_authorization` re-checks that on the way in.

**What is honestly UNKNOWN before this probe has run.** Plan §10 says cost and HBM beyond the k=2
reference are UNKNOWN, and the two measured points nearby are exp_03's: its B arm cost **2.713x** the
one-step baseline on v6e-64 (a full-FT context, not this one), and its C arm missed fit at **31.28G
against 31.25G** of capacity. The adapter's backward still runs VJPs through the frozen 5B to both
the context and the rollout state, so k=4 may cost more than 2x k=2 and may cost more than 4x — which
is exactly why **k=4 is exploratory only** and may never be the headline.

**measure -> aggregate -> project -> authorize:**

* :func:`cell_verdict` decides ONE cell against the headroom rule — steady-state peak <= 90% of
  capacity — and **refuses rather than warns**. exp_03's C arm missed by 0.1%; a rule that warns and
  proceeds is a rule that loses a 64-chip reservation.
* :func:`project_wall_clock` counts evaluations and checkpoints on their OWN cadences (the launcher
  exposes ``EVAL_EVERY`` and ``CHECKPOINT_EVERY`` independently) and takes every overhead from the
  MEASUREMENT rather than from an argument, because the reviewer turned two negative overheads into
  a finite 6.55-hour projection. Every duration is validated finite and non-negative.
* :func:`run_fit_probe` is the ORCHESTRATION and it is host-testable: derive the context, walk the
  ladder, collect trials, aggregate, project, publish. Only :func:`measure_cell_on_device` — compile
  the arm, run to a steady-state step time, read the per-device peak HBM and capacity, count the
  runtime's reservation failures — is device-specific, and it names that boundary.

**What this module has never done, stated plainly:** it has never seen a TPU. Everything below the
device measurer is arithmetic, a verdict rule, a provenance derivation and an artifact contract, and
every test in this repository exercises them on the host.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from maxdiffusion.pos_rollout_support import storage_exists, storage_read_bytes, storage_write_bytes

__all__ = [
    "AUTHORIZATION_PROTOCOL",
    "FOOTPRINT_KEYS",
    "GEOMETRY_KEYS",
    "HEADROOM_FRACTION",
    "LADDER_ARMS",
    "LADDER_K",
    "LADDER_MICROBATCH",
    "TRIALS_PER_CELL",
    "CellMeasurement",
    "CellVerdict",
    "FitCell",
    "ProbeContext",
    "ProbeEvidence",
    "aggregate_trials",
    "assert_cell_authorized",
    "build_evidence",
    "cell_verdict",
    "declared",
    "derive_code_sha",
    "derive_device_signature",
    "derive_model_revision",
    "derive_probe_context",
    "ladder",
    "latent_geometry",
    "load_authorization",
    "main",
    "measure_cell_on_device",
    "project_wall_clock",
    "publish_authorization",
    "recipe_fingerprint",
    "run_fit_probe",
]

AUTHORIZATION_PROTOCOL = "exp06.fit_authorization.v2"
#: Plan §4-P1. Steady state, not transient: a peak measured during compilation is not this number.
HEADROOM_FRACTION = 0.90
#: k=2 is the predeclared primary; k=4 is EXPLORATORY and runnable only in a cell M1 measured.
LADDER_K = (2, 4)
#: Powers-of-two divisors of GBS 256 spanning the default 32 (the reviewer accepted this rationale).
LADDER_MICROBATCH = (8, 16, 32, 64)
#: BOTH arms are measured. Their forward/backward graphs differ, so one's peak is not the other's
#: evidence -- the flaw the reviewer found when the cell was only ``(microbatch, k)``.
LADDER_ARMS = ("rollout", "one_step")
#: Repeats per cell. One trial cannot show a cell that only fits when the neighbours are idle.
TRIALS_PER_CELL = 2

#: Everything OUTSIDE the cell that decides the memory footprint or the shape of the graph. A change
#: to any of these is a different program, so it changes the fingerprint and refuses the cell. The
#: cell's own three fields are deliberately absent: the ladder varies them, and they are authorized
#: per cell rather than globally.
FOOTPRINT_KEYS = (
    "weights_dtype",
    "activations_dtype",
    "scan_layers",
    "remat_policy",
    "attention",
    "use_memory_efficient_attention",
    "flash_min_seq_length",
    "per_device_batch_size",
    "pos_logical_batch",
    "side_adapter_sampling_steps",
    "side_adapter_guide_scale",
    "side_adapter_layers",
    "side_adapter_hidden",
    "side_adapter_heads",
    "num_attention_heads",
    "attention_head_dim",
    "ici_data_parallelism",
    "ici_fsdp_parallelism",
    "ici_context_parallelism",
    "ici_tensor_parallelism",
    "dcn_data_parallelism",
    "dcn_fsdp_parallelism",
    "dcn_context_parallelism",
    "dcn_tensor_parallelism",
)
#: The tensor geometry every activation is sized by.
GEOMETRY_KEYS = ("height", "width", "num_frames", "latent_height", "latent_width")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def declared(config, key: str):
    """Read a key the exp_06 config DECLARES. Two-argument ``getattr`` only (issue #11).

    ``pyconfig.HyperParameters.__getattr__`` raises ``ValueError`` for an unknown key, so the
    three-argument form never falls back -- it propagates, and it has already killed two TPU jobs in
    this campaign. Every key this module reads is in the checked-in YAML, so a missing one is a
    broken deployment and says so.
    """
    try:
        return getattr(config, key)
    except (AttributeError, ValueError) as error:
        raise ValueError(
            f"the exp_06 config must declare {key!r}: the fit probe binds the footprint-bearing recipe, "
            f"and a key it cannot read is a recipe it cannot fingerprint ({type(error).__name__}: {error})"
        ) from error


def _digest(payload: Mapping) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# =================================================================================================
# The cell and the context: what is authorized, and the running program it was measured on.
# =================================================================================================


@dataclasses.dataclass(frozen=True)
class FitCell:
    """The unit M1 authorizes: an ARM at a microbatch and a horizon.

    The arm is here because R-B and matched-C0 build different forward/backward graphs — one unrolls
    k sampler steps under remat and differentiates through them, the other does not — so a peak
    measured for one is not evidence about the other. The reviewer's exact finding.
    """

    arm: str
    microbatch: int
    k_b: int

    def __post_init__(self) -> None:
        if self.arm not in LADDER_ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; exp_06 declares {list(LADDER_ARMS)}")
        if int(self.microbatch) <= 0 or int(self.k_b) <= 0:
            raise ValueError(f"a cell needs a positive microbatch and horizon, got {self.microbatch}/{self.k_b}")

    def as_payload(self) -> dict:
        return {"arm": str(self.arm), "microbatch": int(self.microbatch), "k_b": int(self.k_b)}

    @classmethod
    def from_payload(cls, payload: Mapping) -> "FitCell":
        return cls(arm=str(payload["arm"]), microbatch=int(payload["microbatch"]), k_b=int(payload["k_b"]))


def ladder(
    *,
    arms: Sequence[str] = LADDER_ARMS,
    microbatches: Sequence[int] = LADDER_MICROBATCH,
    horizons: Sequence[int] = LADDER_K,
) -> tuple[FitCell, ...]:
    return tuple(FitCell(arm=a, microbatch=m, k_b=k) for a in arms for m in microbatches for k in horizons)


@dataclasses.dataclass(frozen=True)
class ProbeContext:
    """The running program a measurement is a measurement OF. Every field is derived, none supplied."""

    code_sha: str
    model_revision: str
    device_kind: str
    device_count: int
    geometry: tuple
    recipe_fingerprint: str

    def as_payload(self) -> dict:
        return {
            "code_sha": str(self.code_sha),
            "model_revision": str(self.model_revision),
            "device_kind": str(self.device_kind),
            "device_count": int(self.device_count),
            "geometry": [[str(key), _plain(value)] for key, value in self.geometry],
            "recipe_fingerprint": str(self.recipe_fingerprint),
        }

    def digest(self) -> str:
        return _digest(self.as_payload())

    @classmethod
    def from_payload(cls, payload: Mapping) -> "ProbeContext":
        return cls(
            code_sha=str(payload["code_sha"]),
            model_revision=str(payload["model_revision"]),
            device_kind=str(payload["device_kind"]),
            device_count=int(payload["device_count"]),
            geometry=tuple((str(key), value) for key, value in payload["geometry"]),
            recipe_fingerprint=str(payload["recipe_fingerprint"]),
        )

    def differences(self, other: "ProbeContext") -> tuple:
        """Which derived fields disagree — so a refusal tells the operator what to re-measure."""
        mine, theirs = self.as_payload(), other.as_payload()
        return tuple(sorted(field for field in mine if mine[field] != theirs[field]))


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def recipe_fingerprint(config) -> str:
    """A digest of every footprint-bearing config value OUTSIDE the cell (T7-2).

    Dtype, remat, attention, parallelism, logical batch, sampling geometry and adapter shape all move
    the peak; none of them was checked before, except as a field the caller stamped on the artifact.
    """
    return _digest({key: _plain(declared(config, key)) for key in FOOTPRINT_KEYS})


def latent_geometry(config) -> tuple:
    return tuple((key, _plain(declared(config, key))) for key in GEOMETRY_KEYS)


def _git_head(start: pathlib.Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    head = completed.stdout.strip()
    return head if completed.returncode == 0 and _SHA_RE.match(head) else ""


def derive_code_sha(*, module_file: str | None = None, environ: Mapping | None = None) -> str:
    """The SHA of the code that is RUNNING, from this module's own checkout.

    On a TPU worker the code arrives as an uploaded tarball with no git objects, so the launcher's
    validated ``COMMIT`` is the provenance there; in a checkout git is. When both exist they must
    agree — a disagreement means the tarball and the checkout are different programs, which is
    exactly the case a stamped SHA would have hidden.
    """
    environ = os.environ if environ is None else environ
    stamped = str(environ.get("COMMIT", "")).strip()
    head = _git_head(pathlib.Path(module_file or __file__).resolve().parent)
    if head and stamped and head != stamped:
        raise ValueError(
            f"the running checkout is at {head} but COMMIT declares {stamped}: an HBM measurement is a "
            f"measurement OF A PROGRAM, and these are two of them"
        )
    sha = head or stamped
    if not _SHA_RE.match(sha):
        raise ValueError(
            f"no 40-hex code SHA could be derived (git HEAD {head!r}, COMMIT {stamped!r}): every exp_06 "
            f"artifact is provenance-bound, and an unprovenanced measurement authorizes nothing"
        )
    return sha


def derive_device_signature(devices: Sequence | None = None) -> tuple[str, int]:
    """The devices this process actually has, from the runtime rather than from a caller."""
    if devices is None:
        import jax

        devices = jax.devices()
    if not devices:
        raise ValueError("this process has no devices; there is nothing to measure a peak on")
    kinds = sorted({_device_kind(device) for device in devices})
    if len(kinds) != 1:
        raise ValueError(f"a heterogeneous device set cannot carry one peak measurement: {kinds}")
    return kinds[0], len(devices)


def _device_kind(device) -> str:
    """``v6e`` / ``TPU v6 lite`` / ``cpu`` — whatever the runtime calls the thing that holds the peak.

    Two-argument ``getattr`` only: the three-argument form is forbidden across exp_06 (issue #11), and
    the absence of an attribute here is a real condition rather than a default worth inventing.
    """
    for attribute in ("device_kind", "platform"):
        try:
            value = str(getattr(device, attribute))
        except AttributeError:
            continue
        if value:
            return value
    raise ValueError(f"{device!r} reports neither a device_kind nor a platform; it cannot be provenance")


def derive_model_revision(config) -> str:
    """The RESOLVED local snapshot, not the name the config asked for.

    ``Wan-AI/Wan2.2-TI2V-5B-Diffusers`` is a moving target; the thing whose activations were measured
    is the snapshot on this machine's disk. When no snapshot is resolvable the derivation says so
    rather than inventing one, and an authorization derived without a snapshot then refuses a
    training job that has one — conservative in the direction that costs a re-measure, not a run.
    """
    name = str(declared(config, "pretrained_model_name_or_path"))
    if os.path.isdir(name):
        return f"{name}@local-dir"
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(name, local_files_only=True)
    except Exception as error:  # noqa: BLE001 -- any failure to resolve is a state, not a crash
        return f"{name}@no-local-snapshot:{type(error).__name__}"
    return f"{name}@{pathlib.Path(resolved).name}"


def derive_probe_context(config, *, devices: Sequence | None = None, environ: Mapping | None = None) -> ProbeContext:
    """Everything a measurement is a measurement of, derived from the running program."""
    kind, count = derive_device_signature(devices)
    return ProbeContext(
        code_sha=derive_code_sha(environ=environ),
        model_revision=derive_model_revision(config),
        device_kind=kind,
        device_count=count,
        geometry=latent_geometry(config),
        recipe_fingerprint=recipe_fingerprint(config),
    )


# =================================================================================================
# Measurement, verdict, aggregation, projection.
# =================================================================================================


@dataclasses.dataclass(frozen=True)
class CellMeasurement:
    """What a probe run reports for one cell, BOUND to the context it was measured under.

    ``context_digest`` is the binding: :func:`build_evidence` refuses a measurement whose digest is
    not the context being published, so a number measured on another machine, another commit or
    another model cannot be carried into this artifact alongside a nicer-looking claim.
    """

    cell: FitCell
    context_digest: str
    compile_seconds: float
    step_seconds: float
    eval_seconds: float
    checkpoint_seconds: float
    peak_bytes: int
    capacity_bytes: int
    reservation_failures: int

    def as_payload(self) -> dict:
        return {
            "cell": self.cell.as_payload(),
            "context_digest": str(self.context_digest),
            "compile_seconds": float(self.compile_seconds),
            "step_seconds": float(self.step_seconds),
            "eval_seconds": float(self.eval_seconds),
            "checkpoint_seconds": float(self.checkpoint_seconds),
            "peak_bytes": int(self.peak_bytes),
            "capacity_bytes": int(self.capacity_bytes),
            "reservation_failures": int(self.reservation_failures),
        }


@dataclasses.dataclass(frozen=True)
class CellVerdict:
    fits: bool
    reasons: tuple
    numbers: dict


def _duration(value, what: str, *, strictly_positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be a finite number of seconds, got {value!r}: it was never measured")
    if number < 0.0 or (strictly_positive and number == 0.0):
        raise ValueError(
            f"{what} must be {'positive' if strictly_positive else 'non-negative'}, got {value!r}: a negative "
            f"cost subtracts from a projection, which is how a run that does not fit acquires a plausible "
            f"wall-clock"
        )
    return number


def _checked(measurement: CellMeasurement) -> None:
    """A missing measurement is not a zero. Each of these would otherwise flatter the cell."""
    _duration(measurement.step_seconds, "the steady-state step time", strictly_positive=True)
    _duration(measurement.compile_seconds, "the compile time")
    _duration(measurement.eval_seconds, "the evaluation overhead")
    _duration(measurement.checkpoint_seconds, "the checkpoint overhead")
    if int(measurement.peak_bytes) <= 0:
        raise ValueError(f"the per-device peak must be positive, got {measurement.peak_bytes!r}")
    if int(measurement.capacity_bytes) <= 0:
        raise ValueError(f"the device capacity must be positive, got {measurement.capacity_bytes!r}")
    if int(measurement.reservation_failures) < 0:
        raise ValueError(f"reservation failures are counted, got {measurement.reservation_failures!r}")
    if not _DIGEST_RE.match(str(measurement.context_digest)):
        raise ValueError(
            f"a measurement carries the digest of the context it was measured under; "
            f"{measurement.context_digest!r} is not one"
        )


def cell_verdict(measurement: CellMeasurement) -> CellVerdict:
    """Does this cell fit? Headroom rule + reservation failures, and it REFUSES rather than warns."""
    _checked(measurement)
    fraction = int(measurement.peak_bytes) / int(measurement.capacity_bytes)
    reasons = []
    if fraction > HEADROOM_FRACTION:
        reasons.append("headroom")
    if int(measurement.reservation_failures) > 0:
        reasons.append("reservation_failures")
    return CellVerdict(
        fits=not reasons,
        reasons=tuple(reasons),
        numbers={
            **measurement.cell.as_payload(),
            "peak_bytes": int(measurement.peak_bytes),
            "capacity_bytes": int(measurement.capacity_bytes),
            "peak_fraction": fraction,
            "headroom_fraction": HEADROOM_FRACTION,
            "reservation_failures": int(measurement.reservation_failures),
            "compile_seconds": float(measurement.compile_seconds),
            "step_seconds": float(measurement.step_seconds),
            "eval_seconds": float(measurement.eval_seconds),
            "checkpoint_seconds": float(measurement.checkpoint_seconds),
            "trials": 1,
        },
    )


def aggregate_trials(measurements: Sequence[CellMeasurement]) -> tuple[CellMeasurement, ...]:
    """One measurement per cell, taking the WORST of every trial (T7-3).

    The reviewer published the same cell twice — once fitting, once at 96.9% with a reservation
    failure — and it landed in both the authorized and the refused list and was authorized, because
    publication appended each trial independently while assertion returned on the first authorized
    occurrence. Aggregating conservatively removes the contradiction rather than adjudicating it: a
    cell that missed on any trial is a cell that missed.
    """
    if not measurements:
        raise ValueError("an aggregation over no trials measures nothing; run the ladder first")
    order: list[FitCell] = []
    grouped: dict[FitCell, list[CellMeasurement]] = {}
    for measurement in measurements:
        _checked(measurement)
        if measurement.cell not in grouped:
            grouped[measurement.cell] = []
            order.append(measurement.cell)
        grouped[measurement.cell].append(measurement)
    aggregated = []
    for cell in order:
        trials = grouped[cell]
        digests = {str(trial.context_digest) for trial in trials}
        if len(digests) != 1:
            raise ValueError(
                f"{cell} was measured under {len(digests)} different contexts; those are measurements of "
                f"different programs and cannot be averaged into one cell"
            )
        aggregated.append(
            CellMeasurement(
                cell=cell,
                context_digest=trials[0].context_digest,
                compile_seconds=max(float(trial.compile_seconds) for trial in trials),
                step_seconds=max(float(trial.step_seconds) for trial in trials),
                eval_seconds=max(float(trial.eval_seconds) for trial in trials),
                checkpoint_seconds=max(float(trial.checkpoint_seconds) for trial in trials),
                peak_bytes=max(int(trial.peak_bytes) for trial in trials),
                capacity_bytes=min(int(trial.capacity_bytes) for trial in trials),
                reservation_failures=sum(int(trial.reservation_failures) for trial in trials),
            )
        )
    return tuple(aggregated)


def _positive_int(value, what: str) -> int:
    number = int(value)
    if number != float(value) or number <= 0:
        raise ValueError(f"{what} must be a positive whole number of steps, got {value!r}")
    return number


def project_wall_clock(
    measurement: CellMeasurement,
    *,
    max_train_steps: int,
    eval_every: int,
    checkpoint_every: int,
) -> dict:
    """The wall-clock a run in this cell would take, INCLUDING eval and checkpoint overhead (F11).

    **The overheads come from the MEASUREMENT, not from arguments.** The first version accepted them
    as caller-supplied floats and validated nothing, so the reviewer produced a finite 6.55-hour
    projection from a negative evaluation cost and a negative checkpoint cost. It also counted
    checkpoints on the evaluation cadence, although the launcher exposes ``CHECKPOINT_EVERY``
    independently — a run that checkpoints four times as often as it evaluates was projected as if it
    did not.
    """
    verdict = cell_verdict(measurement)
    if not verdict.fits:
        raise ValueError(
            f"cell {measurement.cell} does not fit ({verdict.reasons}); projecting a wall-clock for it "
            f"would put a number nobody can realize into a launch plan"
        )
    steps = _positive_int(max_train_steps, "max_train_steps")
    eval_cadence = _positive_int(eval_every, "eval_every")
    checkpoint_cadence = _positive_int(checkpoint_every, "checkpoint_every")
    evaluations = steps // eval_cadence
    checkpoints = steps // checkpoint_cadence
    train = float(measurement.step_seconds) * steps
    evals = float(measurement.eval_seconds) * evaluations
    checkpointing = float(measurement.checkpoint_seconds) * checkpoints
    total = train + evals + checkpointing + float(measurement.compile_seconds)
    return {
        **verdict.numbers,
        "max_train_steps": steps,
        "eval_every": eval_cadence,
        "checkpoint_every": checkpoint_cadence,
        "evaluations": evaluations,
        "checkpoints": checkpoints,
        "train_seconds": train,
        "eval_seconds_total": evals,
        "checkpoint_seconds_total": checkpointing,
        "compile_seconds": float(measurement.compile_seconds),
        "total_seconds": total,
        "total_hours": total / 3600.0,
    }


# =================================================================================================
# The authorization artifact.
# =================================================================================================


@dataclasses.dataclass(frozen=True)
class ProbeEvidence:
    """What M1 learned, ready to publish: one context, one measurement per cell, and the verdicts."""

    context: ProbeContext
    measurements: tuple
    projections: tuple

    def as_payload(self) -> dict:
        authorized, measured, refused = [], [], []
        for measurement in self.measurements:
            verdict = cell_verdict(measurement)
            payload = measurement.cell.as_payload()
            measured.append(payload)
            if verdict.fits:
                authorized.append(payload)
            else:
                refused.append({**payload, "reasons": list(verdict.reasons)})
        return {
            "protocol": AUTHORIZATION_PROTOCOL,
            "context": self.context.as_payload(),
            "context_digest": self.context.digest(),
            "authorized_cells": authorized,
            "measured_cells": measured,
            "refused_cells": refused,
            "headroom_fraction": HEADROOM_FRACTION,
            "measurements": [measurement.as_payload() for measurement in self.measurements],
            "projections": [dict(projection) for projection in self.projections],
        }


def build_evidence(
    context: ProbeContext,
    measurements: Sequence[CellMeasurement],
    *,
    max_train_steps: int,
    eval_every: int,
    checkpoint_every: int,
) -> ProbeEvidence:
    """Bind the trials to the context they were measured under, aggregate them, and project."""
    if not isinstance(context, ProbeContext):
        raise ValueError(
            "the context must be one derive_probe_context() produced from the running program; a mapping "
            "that names a SHA is the caller's claim, not the program's provenance"
        )
    if not measurements:
        raise ValueError("an authorization over no measurements authorizes nothing; run the ladder first")
    wanted = context.digest()
    foreign = sorted({str(m.context_digest) for m in measurements if str(m.context_digest) != wanted})
    if foreign:
        raise ValueError(
            f"these measurements were made under context digest(s) {foreign}, not under {wanted}: a number "
            f"measured on another commit, model or device cannot be published as evidence about this one"
        )
    aggregated = aggregate_trials(measurements)
    projections = [
        project_wall_clock(
            measurement,
            max_train_steps=max_train_steps,
            eval_every=eval_every,
            checkpoint_every=checkpoint_every,
        )
        for measurement in aggregated
        if cell_verdict(measurement).fits
    ]
    return ProbeEvidence(context=context, measurements=aggregated, projections=tuple(projections))


def publish_authorization(path: str, evidence: ProbeEvidence) -> dict:
    """The artifact M2/M3 must present to run: exactly the cells M1 measured AND found to fit.

    A measured-and-refused cell is recorded too, with its reasons, so an operator who asked for it
    learns that it was measured and missed rather than that it was forgotten. Published once and
    adopted on an identical republication, refused on a different one (issue #10).

    **It takes no provenance arguments.** Everything published here comes from the evidence, whose
    context was derived and whose measurements are bound to that context's digest.
    """
    if not isinstance(evidence, ProbeEvidence):
        raise ValueError(
            "publication takes the evidence build_evidence() produced; separate measurement and "
            "provenance arguments are what let the reviewer publish a wrong SHA next to real numbers"
        )
    payload = evidence.as_payload()
    published = {**payload, "sha256": _digest(payload)}
    if storage_exists(path):
        existing = json.loads(storage_read_bytes(path).decode("utf-8"))
        if _digest(existing.get("payload") or {}) != existing.get("sha256"):
            raise ValueError(f"{path}: the published digest does not describe its payload; it was edited")
        if existing["sha256"] != published["sha256"]:
            raise ValueError(
                f"{path} was already published with digest {existing['sha256']!r}; this probe produced "
                f"{published['sha256']!r}. An authorization is adopted, never rewritten (issue #10)."
            )
        return {**existing["payload"], "sha256": existing["sha256"]}
    storage_write_bytes(
        path, json.dumps({"payload": payload, "sha256": published["sha256"]}, sort_keys=True).encode("utf-8")
    )
    return published


def _cell_list(payload: Mapping, field: str) -> list[FitCell]:
    entries = payload.get(field)
    if not isinstance(entries, list):
        raise ValueError(f"the authorization's {field} must be a list of cells, got {type(entries).__name__}")
    cells = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{field} contains {entry!r}, which is not a cell")
        try:
            cells.append(FitCell.from_payload(entry))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{field} contains {entry!r}, which is not a cell this probe authorizes: {error}"
            ) from error
    if len(set(cells)) != len(cells):
        raise ValueError(f"{field} lists the same cell twice; a cell has ONE aggregated verdict, not several")
    return cells


def load_authorization(path: str) -> dict:
    """Read an authorization, re-verify its digest, and validate the WHOLE schema.

    An edited artifact authorizes an unmeasured cell, so the digest is checked; but a *malformed*
    one -- a cell in both lists, a refusal missing from the measured list, a context without a
    fingerprint -- would have been read straight through before, and the assertion below would then
    have answered a question the artifact never actually settled.
    """
    stored = json.loads(storage_read_bytes(path).decode("utf-8"))
    payload = stored.get("payload")
    if not isinstance(payload, Mapping) or _digest(payload) != stored.get("sha256"):
        raise ValueError(
            f"{path}: the recorded digest {stored.get('sha256')!r} does not describe its payload — the "
            f"fit authorization has been edited, and it is the only thing standing between a launch and "
            f"a cell nobody measured"
        )
    if payload.get("protocol") != AUTHORIZATION_PROTOCOL:
        raise ValueError(f"{path}: protocol {payload.get('protocol')!r} is not {AUTHORIZATION_PROTOCOL}")
    context = payload.get("context")
    if not isinstance(context, Mapping):
        raise ValueError(f"{path}: an authorization carries the context it was measured under; this one carries none")
    try:
        rebuilt = ProbeContext.from_payload(context)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: the recorded context is not a probe context: {error}") from error
    if rebuilt.digest() != payload.get("context_digest"):
        raise ValueError(
            f"{path}: the recorded context digest {payload.get('context_digest')!r} does not describe the "
            f"recorded context — the two would then bind different programs"
        )
    authorized, measured, refused = (
        _cell_list(payload, field) for field in ("authorized_cells", "measured_cells", "refused_cells")
    )
    overlap = sorted(str(cell) for cell in set(authorized) & set(refused))
    if overlap:
        raise ValueError(
            f"{path}: {overlap} are listed as BOTH authorized and refused; the artifact contradicts itself"
        )
    if set(authorized) | set(refused) != set(measured):
        raise ValueError(
            f"{path}: the authorized and refused cells do not cover the measured ones — "
            f"{sorted(str(c) for c in set(measured) ^ (set(authorized) | set(refused)))} is unaccounted for"
        )
    return {**dict(payload), "sha256": stored["sha256"]}


def assert_cell_authorized(authorization: Any, cell: FitCell, *, context: ProbeContext) -> None:
    """The gate M2/M3 cannot get around: M1 measured THIS cell, on THIS program, and it fit.

    ``context`` is required and must be a derived :class:`ProbeContext`. That is the whole fix for
    the reviewer's executed attack: production used to call this with nothing to compare against, so
    an authorization carrying a foreign SHA, a foreign model and the wrong device kind was accepted.
    """
    if not isinstance(authorization, Mapping) or authorization.get("protocol") != AUTHORIZATION_PROTOCOL:
        raise ValueError(
            "a training run needs the authorization the fit probe published; a mapping that lists cells "
            "is not evidence that anything was measured"
        )
    if not isinstance(context, ProbeContext):
        raise ValueError(
            "the current context must be one derive_probe_context() produced from THIS running program; "
            "a mapping that repeats the authorization's own claims proves nothing about this job"
        )
    recorded = authorization.get("context")
    if not isinstance(recorded, Mapping):
        raise ValueError("this authorization records no context, so it cannot be bound to any program")
    measured_under = ProbeContext.from_payload(recorded)
    if measured_under.digest() != context.digest():
        raise ValueError(
            f"this authorization measured a different program: {list(measured_under.differences(context))} "
            f"differ (measured on {measured_under.code_sha[:12]}/{measured_under.model_revision}/"
            f"{measured_under.device_kind}x{measured_under.device_count}, running "
            f"{context.code_sha[:12]}/{context.model_revision}/{context.device_kind}x{context.device_count}). "
            f"An HBM peak is a measurement OF A PROGRAM, and that is not this one."
        )
    wanted = cell.as_payload()
    if wanted in [dict(entry) for entry in authorization.get("authorized_cells", [])]:
        return
    refused = [
        entry
        for entry in authorization.get("refused_cells", [])
        if dict(entry, reasons=None) == dict(wanted, reasons=None)
    ]
    if refused:
        raise ValueError(
            f"M1 did not authorize {cell.arm} at microbatch={cell.microbatch} k={cell.k_b}: it measured that "
            f"cell and refused it ({refused[0].get('reasons')}). Choose an authorized cell or re-run M1 on "
            f"code that changes the footprint."
        )
    raise ValueError(
        f"M1 did not authorize {cell.arm} at microbatch={cell.microbatch} k={cell.k_b}: that cell was never "
        f"measured. M1 authorizes only the cells it measured (plan §4-P1); authorized are "
        f"{authorization.get('authorized_cells')}."
    )


# =================================================================================================
# The probe itself: orchestration on the host, telemetry on the device.
# =================================================================================================


def measure_cell_on_device(*, cell: FitCell, context: ProbeContext, config) -> CellMeasurement:  # pragma: no cover
    """Compile one cell, run it to a steady state, and read the runtime's telemetry. DEVICE ONLY.

    This is the one thing in this module that cannot run on a host, and it is narrow on purpose:
    build the arm's train step at ``cell``, compile it, run enough steps that the step time is
    steady rather than warm-up, read the per-device peak HBM and the device capacity from the
    runtime's memory stats, time one DEV evaluation and one checkpoint write, and count the
    reservation failures the runtime reports. Everything around it — the ladder, the trials, the
    aggregation, the verdict, the projection and the publication — is orchestration and runs above.
    """
    del cell, context, config
    raise NotImplementedError(
        "the per-cell TPU telemetry is the device boundary: compiling the arm at this cell, running to a "
        "STEADY-STATE step time, reading the per-device peak HBM and capacity from the runtime's memory "
        "stats, timing one DEV evaluation and one checkpoint write, and counting the runtime's reservation "
        "failures. The orchestration around it (ladder, trials, aggregation, verdict, projection, "
        "publication) is implemented and host-tested — only this adapter needs a TPU."
    )


def run_fit_probe(
    config,
    *,
    measurer: Callable[..., CellMeasurement] = measure_cell_on_device,
    cells: Sequence[FitCell] | None = None,
    trials: int = TRIALS_PER_CELL,
    devices: Sequence | None = None,
) -> dict:
    """M1, end to end: derive the context, walk the ladder, aggregate, project, publish.

    The reviewer's finding was that the previous version hid missing ORCHESTRATION behind a device
    boundary — it walked nothing, aggregated nothing and published nothing — so M1 could not be run
    from this launch surface at all. This walks it.

    ``measurer`` is a seam only in the sense that a test can pass a host stand-in; it cannot forge
    provenance, because the CONTEXT is derived here and every measurement it returns is checked
    against the requested cell and that context's digest before it is allowed into the evidence.
    """
    context = derive_probe_context(config, devices=devices)
    requested = tuple(cells) if cells is not None else ladder()
    if not requested:
        raise ValueError("a probe over no cells authorizes nothing")
    if int(trials) < 1:
        raise ValueError(f"each cell needs at least one trial, got {trials!r}")
    measurements: list[CellMeasurement] = []
    for cell in requested:
        for trial in range(int(trials)):
            measurement = measurer(cell=cell, context=context, config=config)
            if not isinstance(measurement, CellMeasurement):
                raise ValueError(
                    f"the measurer returned {type(measurement).__name__} for {cell}, not a CellMeasurement"
                )
            if measurement.cell != cell:
                raise ValueError(
                    f"trial {trial} of {cell} came back describing {measurement.cell}: a measurement is "
                    f"evidence about the cell it was taken on"
                )
            if str(measurement.context_digest) != context.digest():
                raise ValueError(
                    f"trial {trial} of {cell} came back bound to another context: the probe derives the "
                    f"context, the measurer does not get to choose it"
                )
            measurements.append(measurement)
    evidence = build_evidence(
        context,
        measurements,
        max_train_steps=int(declared(config, "max_train_steps")),
        eval_every=int(declared(config, "eval_every")),
        checkpoint_every=int(declared(config, "checkpoint_every")),
    )
    path = str(declared(config, "pos_fit_authorization"))
    if not path:
        raise ValueError(
            "pos_fit_authorization must name the path M1 publishes to; a probe that measures a ladder and "
            "publishes nowhere leaves M2/M3 with no route to a run"
        )
    published = publish_authorization(path, evidence)
    print(
        f"[M1] context {context.digest()[:16]} on {context.device_kind}x{context.device_count} @ {context.code_sha[:12]}"
    )
    for entry in published["measured_cells"]:
        state = "AUTHORIZED" if entry in published["authorized_cells"] else "refused"
        print(f"[M1] {entry['arm']:<9s} microbatch={entry['microbatch']:<3d} k={entry['k_b']}  {state}")
    for projection in published["projections"]:
        print(
            f"[M1] projection {projection['arm']} m={projection['microbatch']} k={projection['k_b']}: "
            f"{projection['total_hours']:.2f}h at {projection['peak_fraction']:.1%} of capacity"
        )
    print(f"[M1] published {path} sha256={published['sha256']}")
    return published


def main(argv):
    """``python src/maxdiffusion/pos_rollout_fit_probe.py <config.yml> key=value ...`` — the M1 job."""
    from maxdiffusion import pyconfig

    pyconfig.initialize(argv)
    return run_fit_probe(pyconfig.config)


if __name__ == "__main__":  # pragma: no cover -- the shell entry point
    import sys

    main(sys.argv)
