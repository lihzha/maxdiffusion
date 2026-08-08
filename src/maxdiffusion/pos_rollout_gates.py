"""exp_06 `rollout_adapter` — T5b: the two gates that decide the experiment (plan §3c, §3e).

**The primary gate (§3c)** asks the question Yixun's decision 1 exists to make answerable: did the
rollout objective beat its matched control? Mean PAIRED per-example ΔSSIM(R-B − matched-C0) on DEV-64
**≥ +0.05 AND paired-bootstrap CI-low > 0** (10,000 resamples, seed 20260804). The decision function
is **exp_04's ``gate_g3_vs_null_only``, by import** — same margin, same CI rule, same coverage checks,
same claim-penalizing imputation. Two experiments that re-derive the same bootstrap end up with two
definitions of the same claim, and the difference surfaces as a disagreement nobody can adjudicate.

**The action-use gate (§3e)** is the one that makes this an action-conditioned claim rather than a
reconstruction claim. SSIM cannot tell a world model from a video autoencoder that ignores its
actions: an adapter that learned to continue plausible robot footage from the first frame scores well
and knows nothing about what the robot was asked to do. So each example is scored three times on
**identical noise** — its own actions, another example's, and zeros — and the gate is that true beats
wrong (CI-low > 0; NO +0.05 margin, which belongs to the primary gate alone).

**STAMPED ≠ BOUND — what this module was reworked for.** The reviewer executed five forgeries against
its first version, and every one of them was the same defect: evidence that was CARRIED rather than
DERIVED. ``{"certificate": GATE_CERTIFICATE, "passed": True}`` unlocked TEST;
``dev_certificate(GateVerdict(True, (), {}), …)`` issued a pass with ``mean_delta=NaN``; a TEST-seeded
mapping passed as DEV's derangement; byte-identical donors passed because nothing ever received
action bytes; and the identical-noise contract lived in ``action_use_plan``, **which nothing
consumed**. The rework:

* :class:`DerangementArtifact` — cohort, seed, permutation, **per-example action digests** and a
  fingerprint over all of it. It is BUILT by reading the cohort's own actions through
  :class:`CohortBatchReader`, so "no byte-identical donor" is a measurement rather than a promise, and
  it is fully re-validated everywhere it is used (planning, scoring, gating).
* :func:`score_condition_table` — the ONE table producer, and the only consumer of the plan. It emits
  receiver/donor identities, the digest of the actions actually fed, the digest of the noise key
  actually drawn, the restored checkpoint and the executed horizon. A future scorer that keys the
  wrong-action row on the DONOR now produces a table the gate refuses.
* Gates accept only :class:`~maxdiffusion.eval_wan_pos_rollout.ScoreTable` artifacts, never naked
  mappings, and :func:`dev_certificate` **computes the primary gate itself** from those artifacts
  before publishing a strict, digest-verified schema that :func:`load_dev_certificate` re-validates.

**The derangement repairs by BIJECTIVE SWAP and fails closed** (plan v2.6). A non-bijective repair
would use one example's actions twice and another's never, so the wrong-action condition would differ
from the true one in its marginal distribution as well as its pairing — and the gate would stop
isolating action USE.

**Reported, never gated:** Δ(true − zero), the adapter-disabled diagnostic row, and the same battery
run on matched-C0 — including **both** of C0's deltas (§3e wants the full battery on the control).
The reported block carries no ``passed`` key at all.

**TEST is one door, and it runs BOTH gates.** :func:`confirm_on_test` loads a DEV certificate from
disk, re-validates its schema and its computed pass conditions, and then confirms the primary gate
AND the action-use gate on TEST-64 with an **independently derived** TEST derangement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from maxdiffusion.eval_wan_pos_rollout import (
    DEPLOYED_SAMPLING_STEPS,
    J0_TEST64_SHA256,
    CheckpointIdentity,
    ScoreTable,
    build_score_table,
    draw_key_digest,
    evaluation_draw_key,
    load_certificate,
    publish_certificate,
)
from maxdiffusion.null_adapter_gates import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MAX_INVALID_FRACTION,
    GateVerdict,
    NoiseConvention,
    gate_g3_vs_null_only,
)
from maxdiffusion.pos_rollout_dev_instrument import DevCohort
from maxdiffusion.pos_rollout_support import storage_read_bytes

__all__ = [
    "ACTION_CONDITIONS",
    "ACTION_USE_MARGIN",
    "ACTION_USE_PROTOCOL",
    "DERANGEMENT_PROTOCOL",
    "DERANGEMENT_SEED",
    "GATE_CERTIFICATE",
    "GATE_PROTOCOL",
    "PLAN_PROTOCOL",
    "PRIMARY_MARGIN",
    "TEST_CONFIRMATION_PROTOCOL",
    "ActionUsePlan",
    "CohortBatchReader",
    "DerangementArtifact",
    "TestCohort",
    "action_use_gate",
    "action_use_plan",
    "action_use_report",
    "as_gate_table",
    "cohort_derangement",
    "confirm_on_test",
    "confirmation_payload",
    "derangement_fingerprint",
    "dev_certificate",
    "load_dev_certificate",
    "load_test_cohort",
    "primary_gate",
    "report_payload",
    "score_condition_table",
]

#: Plan §3c. Not a parameter: a margin a caller can pass is a margin that moves when the run misses.
PRIMARY_MARGIN = 0.05
#: Plan §3e gates on the CI alone — "true beats wrong", not "beats it by the primary margin".
ACTION_USE_MARGIN = 0.0
DERANGEMENT_SEED = 20260804
#: The conditions §3e evaluates. ``adapter_disabled`` is the diagnostic row: the frozen backbone with
#: no adapter at all, which is what "the adapter contributed nothing" would look like.
ACTION_CONDITIONS = ("true", "wrong", "zero")
SCORED_CONDITIONS = ("true", "wrong", "zero", "adapter_disabled")
GATE_PROTOCOL = "exp06.gates.v1"
DERANGEMENT_PROTOCOL = "exp06.derangement.v1"
PLAN_PROTOCOL = "exp06.action_use_plan.v1"
ACTION_USE_PROTOCOL = "exp06.action_use.v1"
TEST_CONFIRMATION_PROTOCOL = "exp06.test_confirmation.v1"
#: Stamped on the DEV certificate; TEST confirmation will not proceed without one carrying it — and,
#: since the reviewer unlocked TEST with a mapping carrying nothing else, will not proceed on it alone.
GATE_CERTIFICATE = "exp06.dev_primary_gate.v1"
#: exp_06 scores one pinned draw per example (§3d), which is exp_04's single-seed reduction.
_ONE_DRAW = NoiseConvention.GLOBAL


def as_gate_table(table: ScoreTable) -> dict:
    """exp_06's per-example rows in the shape exp_04's decision function reads.

    Nothing is invented: exp_04 reduces over a k-set and requires both metrics, and exp_06 supplies
    the SSIM and the latent MSE its own scorer measured for the same pinned draw. A missing or
    non-finite value stays missing, so the imported gate imputes it against the claim.
    """
    if not isinstance(table, ScoreTable):
        raise TypeError(
            "a gate reads a built ScoreTable, never a mapping of numbers: the mapping cannot say which "
            "checkpoint, which actions or which noise produced it, and that is what the gate checks"
        )
    return {
        str(name): {"0": {"future_ssim": row.get("ssim"), "future_mse": row.get("mse")}}
        for name, row in table.rows.items()
        # A row that is not a record at all becomes MISSING coverage rather than an imputed perfect
        # SSIM: a malformed entry must cost the claim, never flatter it (B5's rule).
        if isinstance(row, Mapping)
    }


# ---------------------------------------------------------------------------------------------
# Cohorts and their canonical reader. TEST lives HERE, never in the DEV instrument.
# ---------------------------------------------------------------------------------------------


class TestCohort:
    """The TEST-64 confirmation cohort — **constructing one IS loading the approved manifest.**

    Deliberately a separate class from :class:`DevCohort` rather than a relaxation of it. The DEV
    instrument's whole contract is that it is structurally unable to see TEST (T3b-3); widening it to
    carry a cohort argument would undo that in order to save thirty lines here. This is the sanctioned
    second TEST read, behind the DEV certificate, and it is pinned to exp_04's published digest.
    """

    __slots__ = ("_rows", "_by_name", "cohort", "manifest_sha256", "manifest_path")

    def __init__(self, path: str):
        from maxdiffusion.null_adapter_manifest_io import MANIFEST_SCHEMA_VERSION

        raw = storage_read_bytes(path)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != J0_TEST64_SHA256:
            raise ValueError(
                f"{path} is not exp_04's published TEST-64 manifest ({J0_TEST64_SHA256}, found {digest}): a "
                f"confirmation cohort chosen at confirmation time confirms nothing"
            )
        payload = json.loads(raw.decode("utf-8"))
        if int(payload["schema_version"]) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported manifest schema_version {payload['schema_version']!r}")
        if str(payload["cohort"]) != "test64":
            raise ValueError(f"{path} declares cohort {payload['cohort']!r}, not 'test64'")
        self.cohort = "test64"
        self.manifest_sha256 = digest
        self.manifest_path = str(path)
        self._rows = tuple(payload["rows"])
        self._by_name = {str(row["name"]): row for row in self._rows}

    @property
    def names(self) -> tuple:
        return tuple(str(row["name"]) for row in self._rows)

    @property
    def rows(self) -> tuple:
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def row(self, name: str) -> dict:
        if str(name) not in self._by_name:
            raise ValueError(f"{name!r} is not in the test64 cohort")
        return self._by_name[str(name)]


def load_test_cohort(path: str) -> TestCohort:
    return TestCohort(path)


class CohortBatchReader:
    """The evaluator's canonical row reader: it opens the row the approved manifest bound.

    **There is no injectable seam**, for the reason review pass 1 established one layer lower: a
    ``reader`` that echoes the requested identity while returning other tensors defeats every check
    that believes it. The decoder is resolved from the module at construction; a test that needs
    different bytes patches ``run_wan_null_inversion._tfrecord_reader``, which is a property of the
    test process rather than an argument any production caller can reach.
    """

    __slots__ = ("cohort", "_read_batch", "_cache")

    def __init__(self, cohort):
        if not isinstance(cohort, (DevCohort, TestCohort)):
            raise TypeError(
                "a reader reads a LOADED cohort's own rows; a hand-built cohort is exactly what the "
                "manifest binding exists to refuse"
            )
        from maxdiffusion import run_wan_null_inversion

        self.cohort = cohort
        self._cache: dict = {}
        self._read_batch = run_wan_null_inversion.build_read_batch(
            {str(row["name"]): row for row in cohort.rows}, reader=run_wan_null_inversion._tfrecord_reader
        )

    def read(self, name: str) -> dict:
        name = str(name)
        if name in self._cache:
            return self._cache[name]
        self.cohort.row(name)  # membership, against the digest-pinned manifest's content
        batch, fields = self._read_batch((name,))
        if tuple(batch.names) != (name,):
            raise ValueError(f"the canonical reader returned {tuple(batch.names)} when asked for {name!r}")
        example = {
            "z_i0": np.asarray(batch.z_i0, np.float32),
            "z_video": np.asarray(batch.z_video, np.float32),
            "actions": np.asarray(fields[name]["actions"], np.float32)[None, ...],
        }
        self._cache[name] = example
        return example


def _actions_digest(actions) -> str:
    """The digest of the bytes actually fed. One definition, used by the artifact and the producer."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(actions, np.float32)).tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# §3e: the cohort-level seeded derangement, as an ARTIFACT.
# ---------------------------------------------------------------------------------------------


def _cohort_seed(cohort: str) -> int:
    """Independent per cohort, so DEV's assignment tells you nothing about TEST's."""
    tag = hashlib.sha256(f"{DERANGEMENT_SEED}:{cohort}".encode("utf-8")).digest()[:8]
    return int.from_bytes(tag, "big")


@dataclasses.dataclass(frozen=True)
class DerangementArtifact:
    """WHO received WHOSE actions — with the action digests that make it checkable.

    ``action_sha256`` is the digest of each example's OWN action bytes, read from the cohort's own
    records. It is the thing that was missing: without it, "no byte-identical donor" was a claim about
    a name list, and the reviewer passed a mapping whose donors carried identical actions.
    """

    protocol: str
    cohort: str
    seed: int
    permutation: dict
    action_sha256: dict
    fingerprint: str

    def donor(self, name: str) -> str:
        return str(self.permutation[str(name)])

    def payload(self) -> dict:
        return {
            "protocol": self.protocol,
            "cohort": self.cohort,
            "seed": int(self.seed),
            "permutation": dict(self.permutation),
            "action_sha256": dict(self.action_sha256),
            "fingerprint": self.fingerprint,
        }


def derangement_fingerprint(artifact) -> str:
    """The hash persisted beside the permutation, over EVERYTHING that defines the assignment."""
    if isinstance(artifact, DerangementArtifact):
        body = {
            "cohort": artifact.cohort,
            "seed": int(artifact.seed),
            "permutation": dict(artifact.permutation),
            "action_sha256": dict(artifact.action_sha256),
        }
    else:
        raise TypeError("a fingerprint is taken of a DerangementArtifact, not of a bare mapping of names")
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def cohort_derangement(cohort) -> DerangementArtifact:
    """A fixed-point-free assignment ``name -> donor``, seeded per cohort and legal BY MEASUREMENT.

    Legal means: no example receives its own actions (a fixed point IS the true-action condition), and
    no example receives BYTE-IDENTICAL actions from another example (a fixed point the name check
    cannot see). The action bytes are READ HERE, from the cohort's own rows, so both properties are
    measured rather than asserted about a caller's mapping.

    The repair preserves the PERMUTATION by swapping rather than overwriting: an assignment where one
    example's actions are used twice and another's never is no longer a rearrangement of the same
    action set, so the wrong-action condition would differ from the true one in its marginal
    distribution as well as its pairing. Where no legal swap exists this fails closed.
    """
    reader = CohortBatchReader(cohort)
    names = [str(name) for name in cohort.names]
    digests = {name: _actions_digest(reader.read(name)["actions"]) for name in names}
    order = np.random.default_rng(_cohort_seed(str(cohort.cohort))).permutation(len(names))
    assignment = {name: names[order[index]] for index, name in enumerate(names)}

    def legal(receiver: str, donor: str) -> bool:
        return donor != receiver and digests[donor] != digests[receiver]

    for receiver in names:
        if legal(receiver, assignment[receiver]):
            continue
        for offset in range(1, len(names)):
            other = names[(names.index(receiver) + offset) % len(names)]
            if legal(receiver, assignment[other]) and legal(other, assignment[receiver]):
                assignment[receiver], assignment[other] = assignment[other], assignment[receiver]
                break
        else:
            raise ValueError(
                f"no legal wrong-action assignment exists for {receiver!r}: every remaining donor is "
                f"itself or carries byte-identical actions. The cohort cannot support this gate."
            )
    artifact = DerangementArtifact(
        protocol=DERANGEMENT_PROTOCOL,
        cohort=str(cohort.cohort),
        seed=DERANGEMENT_SEED,
        permutation=assignment,
        action_sha256=digests,
        fingerprint="",
    )
    artifact = dataclasses.replace(artifact, fingerprint=derangement_fingerprint(artifact))
    _validate_derangement(artifact, names, cohort=str(cohort.cohort))
    return artifact


def _validate_derangement(artifact, names: Sequence[str], *, cohort: str) -> None:
    """Everything a derangement must be, re-checked wherever one is used. Fail closed, in this order.

    The cohort comes first because "this is DEV's derangement" is the question a TEST-seeded mapping
    answered wrongly; the fingerprint comes next because it is what catches every hand-edit of the
    permutation or the digests at once.
    """
    if not isinstance(artifact, DerangementArtifact):
        raise TypeError(
            "the wrong-action assignment is a DerangementArtifact carrying the cohort, seed, permutation "
            "and action digests; any name permutation used to be accepted as 'the derangement'"
        )
    if artifact.protocol != DERANGEMENT_PROTOCOL:
        raise ValueError(f"{artifact.protocol!r} is not a {DERANGEMENT_PROTOCOL} artifact")
    if artifact.cohort != str(cohort):
        raise ValueError(
            f"this derangement is {artifact.cohort!r}'s, and it is being used for {cohort!r}: the "
            f"assignment is derived independently per cohort, so a TEST mapping is not DEV's"
        )
    if int(artifact.seed) != DERANGEMENT_SEED:
        raise ValueError(f"the derangement seed is {artifact.seed!r}, not the pinned {DERANGEMENT_SEED}")
    if derangement_fingerprint(artifact) != artifact.fingerprint:
        raise ValueError(
            "the derangement's fingerprint does not describe its own permutation and action digests: it "
            "has been edited since it was derived, and it is what says which actions were fed"
        )
    names = [str(name) for name in names]
    if set(artifact.permutation) != set(names) or set(artifact.action_sha256) != set(names):
        raise ValueError(
            f"the derangement does not cover the cohort: {len(artifact.permutation)} entries and "
            f"{len(artifact.action_sha256)} digests for {len(names)} examples"
        )
    fixed = sorted(name for name, donor in artifact.permutation.items() if donor == name)
    if fixed:
        raise ValueError(f"the derangement has a fixed point at {fixed}: those examples get their TRUE actions")
    if sorted(artifact.permutation.values()) != sorted(names):
        raise ValueError("the derangement is not a permutation: some example's actions are used twice")
    identical = sorted(
        name
        for name, donor in artifact.permutation.items()
        if artifact.action_sha256[donor] == artifact.action_sha256[name]
    )
    if identical:
        raise ValueError(f"{identical} were assigned byte-identical actions, which is not a wrong-action row")


# ---------------------------------------------------------------------------------------------
# The plan, and the ONE producer that consumes it.
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ActionUsePlan:
    """What to evaluate, per example: four conditions on ONE draw — the evaluated example's own."""

    protocol: str
    cohort: str
    manifest_sha256: str
    derangement_sha256: str
    entries: tuple

    def payload(self) -> dict:
        return {
            "protocol": self.protocol,
            "cohort": self.cohort,
            "manifest_sha256": self.manifest_sha256,
            "derangement_sha256": self.derangement_sha256,
            "entries": [dict(entry) for entry in self.entries],
        }


def action_use_plan(cohort, *, derangement: DerangementArtifact) -> ActionUsePlan:
    """``draw_key_name`` is the RECEIVER in every condition, never the donor.

    The draw is keyed on the example's name, so keying the wrong row on the donor would change the
    noise between conditions and the gate would measure noise instead of actions. This used to be the
    plan's only claim to fame — and nothing consumed the plan. :func:`score_condition_table` does.
    """
    names = [str(name) for name in cohort.names]
    _validate_derangement(derangement, names, cohort=str(cohort.cohort))
    entries = tuple(
        {
            "name": name,
            "conditions": {
                "true": {"actions_from": name, "draw_key_name": name},
                "wrong": {"actions_from": derangement.donor(name), "draw_key_name": name},
                "zero": {"actions_from": None, "draw_key_name": name},
                "adapter_disabled": {"actions_from": name, "draw_key_name": name},
            },
        }
        for name in names
    )
    return ActionUsePlan(
        protocol=PLAN_PROTOCOL,
        cohort=str(cohort.cohort),
        manifest_sha256=str(cohort.manifest_sha256),
        derangement_sha256=derangement.fingerprint,
        entries=entries,
    )


def score_condition_table(
    plan: ActionUsePlan,
    *,
    condition: str,
    cohort,
    derangement: DerangementArtifact,
    checkpoint: CheckpointIdentity,
    arm: str,
    backend,
) -> ScoreTable:
    """THE table producer — the only consumer of the plan, and the only way a table exists.

    For each planned example it resolves the receiver, reads the actions the condition calls for from
    the cohort's own records, draws the RECEIVER's pinned key, asks the device for a score, and writes
    the identities down: whose actions, their digest, the digest of the key that was drawn, and the
    horizon that executed. The digests are checked against the derangement artifact as they are
    produced, so a donor whose bytes are not the ones the artifact recorded stops the run here rather
    than becoming an unfalsifiable row in a published table.
    """
    if not isinstance(plan, ActionUsePlan):
        raise TypeError("a table is produced from an ActionUsePlan; a loose list of dicts is not a plan")
    if condition not in SCORED_CONDITIONS:
        raise ValueError(f"{condition!r} is not one of {list(SCORED_CONDITIONS)}")
    names = [str(name) for name in cohort.names]
    _validate_derangement(derangement, names, cohort=str(cohort.cohort))
    if plan.cohort != str(cohort.cohort) or plan.manifest_sha256 != str(cohort.manifest_sha256):
        raise ValueError(
            f"this plan is for cohort {plan.cohort!r}/{plan.manifest_sha256[:12]}, and it is being run "
            f"against {cohort.cohort!r}/{str(cohort.manifest_sha256)[:12]}"
        )
    if plan.derangement_sha256 != derangement.fingerprint:
        raise ValueError("the plan was built from a different derangement than the one supplied to score it")

    reader = CohortBatchReader(cohort)
    rows = {}
    for entry in plan.entries:
        receiver = str(entry["name"])
        spec = entry["conditions"][condition]
        if str(spec["draw_key_name"]) != receiver:
            raise ValueError(
                f"the {condition!r} row for {receiver!r} plans its draw on {spec['draw_key_name']!r}: the "
                f"pinned draw is keyed on the RECEIVER in every condition, or the comparison is of noise"
            )
        example = reader.read(receiver)
        donor = spec["actions_from"]
        if condition == "wrong":
            if str(donor) != derangement.donor(receiver):
                raise ValueError(f"the plan feeds {receiver!r} actions from {donor!r}, not its recorded donor")
            actions = reader.read(str(donor))["actions"]
        elif donor is None:
            actions = np.zeros_like(example["actions"])
        else:
            if str(donor) != receiver:
                raise ValueError(f"the {condition!r} row for {receiver!r} may not be fed {donor!r}'s actions")
            actions = example["actions"]
        digest = _actions_digest(actions)
        if donor is not None and digest != derangement.action_sha256[str(donor)]:
            raise ValueError(
                f"the actions read for {donor!r} do not match the digest the derangement recorded: the "
                f"records moved under the assignment, so the wrong-action row would be unverifiable"
            )
        key = evaluation_draw_key(receiver)
        execution, metrics = backend.score(
            z_i0=example["z_i0"],
            z_video=example["z_video"],
            actions=actions,
            key=key,
            adapter_enabled=condition != "adapter_disabled",
        )
        if int(execution.num_steps) != DEPLOYED_SAMPLING_STEPS:
            raise ValueError(f"the rollout for {receiver!r} executed {execution.num_steps} steps")
        if execution.draw_key_sha256 != draw_key_digest(key):
            raise ValueError(f"the rollout for {receiver!r} did not use the pinned draw it was given")
        rows[receiver] = {
            "ssim": metrics.get("ssim_avg"),
            "mse": metrics.get("latent_mse"),
            "actions_from": None if donor is None else str(donor),
            "actions_sha256": digest,
            "draw_key_sha256": execution.draw_key_sha256,
            "num_steps": int(execution.num_steps),
        }
    return build_score_table(
        rows=rows,
        cohort=cohort,
        condition=condition,
        arm=arm,
        checkpoint=checkpoint,
        num_steps=DEPLOYED_SAMPLING_STEPS,
        derangement_sha256=derangement.fingerprint if condition == "wrong" else None,
    )


# ---------------------------------------------------------------------------------------------
# The gates. They read artifacts, and they compute their own decisions.
# ---------------------------------------------------------------------------------------------


def _cohort_names(cohort) -> list:
    if not isinstance(cohort, (DevCohort, TestCohort)):
        raise TypeError(
            "a gate is decided over a LOADED cohort's manifest, never a caller's name list: the manifest "
            "digest is what makes the verdict quotable"
        )
    return [str(name) for name in cohort.names]


def _agree(left: ScoreTable, right: ScoreTable, cohort) -> None:
    """Two tables may only be compared when they describe the same measurement of two arms."""
    for table in (left, right):
        if not isinstance(table, ScoreTable):
            raise TypeError("a gate compares two built ScoreTables; a mapping of numbers is not one")
        if table.cohort != str(cohort.cohort) or table.manifest_sha256 != str(cohort.manifest_sha256):
            raise ValueError(
                f"a {table.cohort!r} table is being decided against the {cohort.cohort!r} cohort: a score "
                f"is only quotable against the cohort it was measured on"
            )
        if table.num_steps != DEPLOYED_SAMPLING_STEPS:
            raise ValueError(
                f"a table measured at horizon {table.num_steps} cannot decide a {DEPLOYED_SAMPLING_STEPS}-step claim"
            )


def primary_gate(*, rollout: ScoreTable, control: ScoreTable, cohort) -> GateVerdict:
    """Plan §3c: mean paired ΔSSIM(R-B − matched-C0) ≥ +0.05 with CI-low > 0, on DEV-64.

    Keyword-only on purpose: these two tables are the experiment's claim and its control, and swapping
    them by position reports the control beating the arm with a straight face.
    """
    names = _cohort_names(cohort)
    _agree(rollout, control, cohort)
    for label, table in (("rollout", rollout), ("control", control)):
        if table.condition != "true":
            raise ValueError(
                f"the {label} table is the {table.condition!r} condition: the primary gate compares the "
                f"two arms under TRUE actions, and any other condition answers a different question"
            )
    if rollout.checkpoint == control.checkpoint:
        raise ValueError(
            f"both tables were measured on checkpoint {rollout.checkpoint!r}: matched-C0 is a different "
            f"run, and comparing a checkpoint with itself reports a delta of zero as a finding"
        )
    verdict = gate_g3_vs_null_only(as_gate_table(rollout), as_gate_table(control), names, _ONE_DRAW)
    return GateVerdict(
        verdict.passed,
        verdict.reasons,
        {
            **verdict.numbers,
            "cohort": str(cohort.cohort),
            "manifest_sha256": str(cohort.manifest_sha256),
            "rollout_table_sha256": rollout.digest,
            "control_table_sha256": control.digest,
            "rollout_checkpoint": rollout.checkpoint,
            "control_checkpoint": control.checkpoint,
            "num_steps": rollout.num_steps,
        },
    )


def dev_certificate(path: str, *, rollout: ScoreTable, control: ScoreTable, cohort) -> dict:
    """Issue the DEV certificate by COMPUTING the gate from the bound tables, then publishing it.

    It no longer takes a verdict. The reviewer passed ``GateVerdict(True, (), {})`` and got a passing
    certificate with ``mean_delta=NaN`` and an empty CI, because "the gate passed" was a caller's
    word. Here the only inputs are the two artifacts and the cohort, and the decision is made here.
    """
    verdict = primary_gate(rollout=rollout, control=control, cohort=cohort)
    numbers = verdict.numbers
    payload = {
        "protocol": GATE_PROTOCOL,
        "certificate": GATE_CERTIFICATE,
        "passed": bool(verdict.passed),
        "reasons": list(verdict.reasons),
        "cohort": str(cohort.cohort),
        "manifest_sha256": str(cohort.manifest_sha256),
        "margin": PRIMARY_MARGIN,
        "num_steps": int(rollout.num_steps),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "coverage_ok": bool(numbers.get("coverage_ok")),
        "example_count": len(_cohort_names(cohort)),
        "mean_delta": float(numbers.get("mean_delta", float("nan"))),
        "ci": [float(value) for value in numbers.get("ci", [])],
        "invalid_fraction": float(numbers.get("invalid_fraction", 1.0)),
        "rollout_table_sha256": rollout.digest,
        "control_table_sha256": control.digest,
        "rollout_checkpoint": rollout.checkpoint,
        "control_checkpoint": control.checkpoint,
    }
    return publish_certificate(path, payload)


_CERTIFICATE_FIELDS = (
    "certificate",
    "passed",
    "reasons",
    "cohort",
    "manifest_sha256",
    "margin",
    "num_steps",
    "bootstrap",
    "coverage_ok",
    "mean_delta",
    "ci",
    "rollout_table_sha256",
    "control_table_sha256",
)


def load_dev_certificate(path: str) -> dict:
    """Read a DEV certificate and re-decide whether it could have been issued.

    Digest verification only proves nobody edited the file. What TEST needs to know is that this
    artifact IS the DEV primary gate's verdict: the pinned protocol and marker, the pinned margin and
    bootstrap, complete coverage, a finite mean and CI, and a ``passed`` flag that agrees with its own
    numbers. ``{"certificate": GATE_CERTIFICATE, "passed": True}`` satisfies none of it.
    """
    payload = load_certificate(path, protocol=GATE_PROTOCOL)
    missing = [field for field in _CERTIFICATE_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{path}: this is not a DEV gate certificate — it carries no {missing}")
    if payload["certificate"] != GATE_CERTIFICATE:
        raise ValueError(f"{path}: certificate marker {payload['certificate']!r}, expected {GATE_CERTIFICATE!r}")
    if float(payload["margin"]) != PRIMARY_MARGIN:
        raise ValueError(f"{path}: issued against margin {payload['margin']!r}, not the pinned {PRIMARY_MARGIN}")
    bootstrap = payload["bootstrap"]
    if int(bootstrap.get("seed", -1)) != BOOTSTRAP_SEED or int(bootstrap.get("resamples", -1)) != BOOTSTRAP_RESAMPLES:
        raise ValueError(f"{path}: issued against bootstrap {bootstrap!r}, not the pinned constants")
    if int(payload["num_steps"]) != DEPLOYED_SAMPLING_STEPS:
        raise ValueError(f"{path}: issued at horizon {payload['num_steps']!r}, not the deployed grid")
    ci = [float(value) for value in payload["ci"]]
    mean_delta = float(payload["mean_delta"])
    if len(ci) != 2 or not all(np.isfinite(ci)) or not np.isfinite(mean_delta):
        raise ValueError(f"{path}: mean_delta {mean_delta!r} and CI {ci!r} must be finite to have decided anything")
    decided = bool(payload["coverage_ok"]) and mean_delta >= PRIMARY_MARGIN and ci[0] > 0.0 and not payload["reasons"]
    if bool(payload["passed"]) != decided:
        raise ValueError(
            f"{path}: it records passed={payload['passed']} but its own numbers decide {decided} "
            f"(coverage={payload['coverage_ok']}, mean_delta={mean_delta}, ci_low={ci[0]}, "
            f"reasons={payload['reasons']}). A certificate must agree with the gate it claims to be."
        )
    return payload


def _action_use_numbers(verdict_numbers: Mapping, cohort, derangement: DerangementArtifact) -> dict:
    """The numbers BOTH return paths of the action-use gate carry (review pass 2, T5b MAJOR 5).

    The early coverage return used to omit the permutation, its hash, the cohort and the manifest, so
    a failing verdict could not be published and ``action_use_report`` raised ``KeyError`` on it.
    """
    return {
        **dict(verdict_numbers),
        "margin": ACTION_USE_MARGIN,
        "derangement": dict(derangement.permutation),
        "derangement_sha256": derangement.fingerprint,
        "derangement_seed": int(derangement.seed),
        "cohort": str(cohort.cohort),
        "manifest_sha256": str(cohort.manifest_sha256),
    }


def action_use_gate(*, true_table: ScoreTable, wrong_table: ScoreTable, cohort, derangement) -> GateVerdict:
    """Plan §3e: mean paired ΔSSIM(true − wrong) with CI-low > 0, and the shuffle written down.

    The COMPUTATION is exp_04's — same paired deltas, same bootstrap, same coverage and imputation —
    but the DECISION is stated here, because §3e's rule genuinely differs from §3c's: there is no
    +0.05 margin. What is checked BEFORE any of that is the pairing itself: identical draw keys across
    conditions, and wrong-action bytes that are the recorded donor's.
    """
    names = _cohort_names(cohort)
    _validate_derangement(derangement, names, cohort=str(cohort.cohort))
    _agree(true_table, wrong_table, cohort)
    if true_table.condition != "true" or wrong_table.condition != "wrong":
        raise ValueError(
            f"the action-use gate compares the 'true' and 'wrong' conditions, got "
            f"{true_table.condition!r} and {wrong_table.condition!r}"
        )
    if wrong_table.derangement_sha256 != derangement.fingerprint:
        raise ValueError(
            f"the wrong-action table was produced under derangement "
            f"{str(wrong_table.derangement_sha256)[:12]}, and it is being judged against "
            f"{derangement.fingerprint[:12]}: the shuffle a table was scored under is part of the table"
        )
    if true_table.checkpoint != wrong_table.checkpoint:
        raise ValueError(
            "the two conditions were measured on different checkpoints, so their difference is not action use"
        )
    true_rows, wrong_rows = true_table.rows, wrong_table.rows
    for name in names:
        if name not in true_rows or name not in wrong_rows:
            continue  # a hole is a COVERAGE failure below, and coverage is claim-penalizing
        if true_rows[name]["draw_key_sha256"] != wrong_rows[name]["draw_key_sha256"]:
            raise ValueError(
                f"{name!r} was scored on different draws in the true and wrong conditions: §3e's whole "
                f"construction is that the two rows differ ONLY in the actions fed, so this comparison "
                f"would be measuring noise. (Keying the wrong row on the DONOR is how this happens.)"
            )
        if wrong_rows[name]["actions_from"] != derangement.donor(name):
            raise ValueError(f"{name!r} was fed {wrong_rows[name]['actions_from']!r}'s actions, not its donor's")
        if wrong_rows[name]["actions_sha256"] != derangement.action_sha256[derangement.donor(name)]:
            raise ValueError(f"the wrong-action bytes for {name!r} are not the ones the derangement recorded")
        if wrong_rows[name]["actions_sha256"] == true_rows[name]["actions_sha256"]:
            raise ValueError(f"{name!r} was fed byte-identical actions in both conditions, which is not a contrast")
    verdict = gate_g3_vs_null_only(as_gate_table(true_table), as_gate_table(wrong_table), names, _ONE_DRAW)
    numbers = _action_use_numbers(verdict.numbers, cohort, derangement)
    if not verdict.numbers.get("coverage_ok"):
        return GateVerdict(False, ("coverage",), numbers)
    reasons = []
    if not verdict.numbers["ci"][0] > ACTION_USE_MARGIN:
        reasons.append("ci_excludes_zero")
    if verdict.numbers["invalid_fraction"] > MAX_INVALID_FRACTION:
        reasons.append("invalid_fraction")
    return GateVerdict(not reasons, tuple(reasons), numbers)


def _mean_delta(left: ScoreTable, right: ScoreTable, names: Sequence[str]) -> float:
    """A reported difference. Deliberately not a gate: it returns a number and never a verdict."""
    rows_left, rows_right = left.rows, right.rows
    return float(np.mean([float(rows_left[name]["ssim"]) - float(rows_right[name]["ssim"]) for name in names]))


def _complete(table: ScoreTable, names: Sequence[str]) -> bool:
    rows = table.rows
    return all(
        name in rows and rows[name]["ssim"] is not None and np.isfinite(float(rows[name]["ssim"])) for name in names
    )


def action_use_report(cohort, *, derangement, tables: Mapping, control_tables: Mapping) -> dict:
    """The §3e gate plus everything §3e reports and refuses to gate on.

    ``control_tables`` (matched-C0 under true, wrong AND zero actions) is REQUIRED, not optional:
    skipping it is how "the adapter uses its actions" gets published without the comparison that says
    whether rollout training uses them MORE than one-step training does. Plan §3e asks for the full
    battery on the control, so both of C0's deltas are reported — and neither may become a pass field.
    """
    names = _cohort_names(cohort)
    if set(tables) != set(SCORED_CONDITIONS):
        raise ValueError(f"the arm's battery is {list(SCORED_CONDITIONS)}, got {sorted(tables)}")
    if set(control_tables) != set(ACTION_CONDITIONS):
        raise ValueError(
            f"control_tables must carry matched-C0 under {list(ACTION_CONDITIONS)} actions, got "
            f"{sorted(control_tables)}: plan §3e runs the FULL battery on the control, so it cannot be "
            f"trimmed to the two rows that make the arm look good"
        )
    gate = action_use_gate(
        true_table=tables["true"], wrong_table=tables["wrong"], cohort=cohort, derangement=derangement
    )
    coverage_ok = bool(gate.numbers.get("coverage_ok")) and all(
        _complete(table, names) for table in (*tables.values(), *control_tables.values())
    )
    reported = {"coverage_ok": coverage_ok}
    if coverage_ok:
        rollout_effect = _mean_delta(tables["true"], tables["wrong"], names)
        control_effect = _mean_delta(control_tables["true"], control_tables["wrong"], names)
        reported.update(
            {
                "mean_delta_true_minus_wrong": rollout_effect,
                "mean_delta_true_minus_zero": _mean_delta(tables["true"], tables["zero"], names),
                "mean_ssim_adapter_disabled": float(
                    np.mean([float(tables["adapter_disabled"].rows[name]["ssim"]) for name in names])
                ),
                "control_mean_delta_true_minus_wrong": control_effect,
                "control_mean_delta_true_minus_zero": _mean_delta(
                    control_tables["true"], control_tables["zero"], names
                ),
                "rollout_uses_actions_more_than_control": bool(rollout_effect > control_effect),
            }
        )
    return {
        "protocol": ACTION_USE_PROTOCOL,
        "cohort": str(cohort.cohort),
        "manifest_sha256": str(cohort.manifest_sha256),
        "derangement_sha256": gate.numbers["derangement_sha256"],
        "table_sha256": {condition: table.digest for condition, table in tables.items()},
        "control_table_sha256": {condition: table.digest for condition, table in control_tables.items()},
        "gate": gate,
        "reported": reported,
    }


def _verdict_payload(verdict: GateVerdict) -> dict:
    return {
        "passed": bool(verdict.passed),
        "reasons": list(verdict.reasons),
        "numbers": json.loads(json.dumps(verdict.numbers, default=float)),
    }


def report_payload(report: Mapping) -> dict:
    """The action-use report as a publishable artifact — the verdict flattened, the block unchanged."""
    return {**{key: value for key, value in report.items() if key != "gate"}, "gate": _verdict_payload(report["gate"])}


# ---------------------------------------------------------------------------------------------
# TEST confirmation — the only door, locked by the DEV certificate, and it runs BOTH gates.
# ---------------------------------------------------------------------------------------------


def confirm_on_test(
    certificate_path: str, *, test_cohort: TestCohort, derangement, tables: Mapping, control_tables: Mapping
) -> dict:
    """Plan §3c AND §3e on TEST-64, reachable only after the DEV gate has passed.

    One door, two gates. The previous version ran only the primary gate, so an adapter could be
    confirmed on held-out data without anyone re-checking that it uses its actions there — which is
    the claim the experiment exists to make. The certificate is LOADED and re-decided
    (:func:`load_dev_certificate`), the cohort is digest-pinned by construction, and the TEST
    derangement is derived independently from the TEST cohort's own actions.
    """
    if not isinstance(test_cohort, TestCohort):
        raise TypeError("TEST confirmation is decided over a loaded TestCohort, whose construction pins the manifest")
    certificate = load_dev_certificate(certificate_path)
    if not certificate["passed"]:
        raise ValueError(
            f"the DEV primary gate did not pass ({certificate['reasons']}), so TEST may not be scored: "
            f"proceeding to TEST is exactly what the DEV gate decides"
        )
    if certificate["cohort"] == test_cohort.cohort:
        raise ValueError("the DEV certificate names the TEST cohort: the gate that unlocks TEST is decided on DEV")
    names = _cohort_names(test_cohort)
    _validate_derangement(derangement, names, cohort=test_cohort.cohort)
    if set(tables) != set(SCORED_CONDITIONS) or set(control_tables) != set(ACTION_CONDITIONS):
        raise ValueError(
            f"TEST confirmation runs the same battery as DEV: {list(SCORED_CONDITIONS)} for the arm and "
            f"{list(ACTION_CONDITIONS)} for matched-C0, got {sorted(tables)} and {sorted(control_tables)}"
        )
    primary = primary_gate(rollout=tables["true"], control=control_tables["true"], cohort=test_cohort)
    action_use = action_use_gate(
        true_table=tables["true"], wrong_table=tables["wrong"], cohort=test_cohort, derangement=derangement
    )
    # The confirmation carries the DEV certificate that unlocked it, so a TEST row can never be quoted
    # without the precondition that made scoring it legitimate (exp_05's S9 row lesson).
    return {
        "protocol": TEST_CONFIRMATION_PROTOCOL,
        "cohort": test_cohort.cohort,
        "manifest_sha256": test_cohort.manifest_sha256,
        "derangement_sha256": derangement.fingerprint,
        "dev_certificate_sha256": hashlib.sha256(
            json.dumps(dict(certificate), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "primary": primary,
        "action_use": action_use,
        "confirmed": bool(primary.passed and action_use.passed),
    }


def confirmation_payload(confirmation: Mapping) -> dict:
    return {
        **{key: value for key, value in confirmation.items() if key not in ("primary", "action_use")},
        "primary": _verdict_payload(confirmation["primary"]),
        "action_use": _verdict_payload(confirmation["action_use"]),
    }
