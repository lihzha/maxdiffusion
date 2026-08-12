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

**THE TRUST BOUNDARY OF A BANKED CELL, stated rather than implied (review F5b, BLOCKER 1).** F5 let a
restart adopt a cell a prior attempt published instead of re-measuring it. Codex's finding was that
the artifact's digest proves CONSISTENCY, not LEGALITY: every hashed value -- the context, the run
identity, the peak, the peak source -- is supplied by whoever wrote the object, so a writer who can
reach the bucket can fabricate a cheap cell, recompute the digest, and be adopted. That was true, and
the battery probe that claimed otherwise was wrong (it watched the run-level digest move and called
propagation a refusal); both are corrected.

What this round DOES provide, and what it does not:

* **Integrity** -- a cell object is content-addressed and digest-verified end to end, so truncation,
  a torn concurrent write, or an edit without a re-hash is refused.
* **Program binding** -- adoption compares the artifact's recorded context byte-for-byte against the
  context THIS process derived, and that context now carries :func:`deployed_manifest_digest`: the
  sha256 of the running bytes, not a commit label. Cells measured by other code, on another topology,
  against another model or under another recipe are refused, and the refusal names the field.
* **NOT authenticated, and the bar is LOWER than it first looks (review F5c).** There is no signature
  and no publication authority. Every field the loader checks -- the context, the manifest digest, the
  run identity, the peaks -- is *inside the artifact*, so ANY WRITER WITH BUCKET WRITE ACCESS WHO CAN
  READ ONE CURRENT ARTIFACT CAN FABRICATE A MEASUREMENT: copy that artifact's context verbatim
  (the manifest digest is public in the payload), replace the trials with one-byte peaks, recompute
  the payload digest and the marker, and adoption accepts it. **Possession of the deployed source tree
  is NOT required** -- the F5b docstring said it was, and that was wrong: the manifest is recomputed
  locally but only ever EQUALITY-compared against a value the forger controls. The reviewer executed
  this end to end through the real publication, loader and adoption functions (``adopting rollout ...
  peak 1 bytes``, measurer skipped). Probe ``F5-8`` keeps it visible, and reports it **DECLARED**
  rather than refused, because reporting it as a refusal is what this module did for two rounds.

  **The trust anchor is therefore the bucket ACL** -- lab-internal writers only -- which is the same
  anchor the final authorization table has always rested on, and the same one every published artifact
  in this campaign rests on. What the manifest binding *does* buy is the accident case, which is the
  one that has actually cost this campaign time: a dirty tree, a stale tarball, a hand-edited module,
  a cross-code adoption. It buys nothing against a deliberate writer. Real authentication (workload
  identity / KMS signing at publication, verified at adoption) is infrastructure, not a code change,
  and it is escalated to Yixun as a policy decision in the pre-launch package rather than skipped
  silently or half-faked with an in-repo shared secret those same bucket writers could read.

**What this module has never done, stated plainly:** it has never seen a TPU. Everything below the
device measurer is arithmetic, a verdict rule, a provenance derivation and an artifact contract, and
every test in this repository exercises them on the host.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from maxdiffusion.pos_rollout_support import (
    storage_exists,
    storage_list_children,
    storage_publish_bytes,
    storage_read_bytes,
    storage_write_bytes,
)

__all__ = [
    "ADOPTED_PREFIX",
    "ADOPTION_SCAN_DEPTH",
    "ADOPTION_SCAN_LIMIT",
    "AUTHORIZATION_PROTOCOL",
    "CELLS_DIRNAME",
    "CELL_PROTOCOL",
    "DIGEST_SUFFIX",
    "PROVENANCE_MEASURED",
    "CellArtifact",
    "adopt_published_cell",
    "adoption_candidates",
    "cell_artifact_name",
    "cell_publication_dir",
    "cell_content_path",
    "cell_marker_path",
    "derive_job_identity",
    "load_cell_artifact",
    "publish_cell",
    "DEV_COHORT_SIZE",
    "FINGERPRINT_EXCLUSIONS",
    "FINGERPRINT_EXCLUSION_REASONS",
    "GEOMETRY_KEYS",
    "TIMED_STEPS",
    "WARMUP_STEPS",
    "DeviceTelemetry",
    "ProbeProgram",
    "ProductionModelSource",
    "build_probe_program",
    "config_keys",
    "config_recipe",
    "evaluation_count",
    "snapshot_manifest_digest",
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

#: ``v4`` because F6 added ``excluded_cells`` / ``exclusion_reason`` / ``skipped_cells`` (v3 added
#: ``cell_provenance``). A v3 loader reading a v4 table would treat a DECLARED-unreachable cell as one
#: that was simply never measured -- the same class of confusion v3 was cut for -- so the version is
#: the fail-closed signal, exactly as before.
AUTHORIZATION_PROTOCOL = "exp06.fit_authorization.v4"
#: Plan §4-P1. Steady state, not transient: a peak measured during compilation is not this number.
HEADROOM_FRACTION = 0.90

# --- F5: per-cell publication and adoption -----------------------------------------------------
#: The per-cell artifact's own protocol, versioned separately from the run-level table.
CELL_PROTOCOL = "exp06.fit_cell.v1"
#: The directory of per-cell artifacts, beside the run-level authorization in the attempt root.
CELLS_DIRNAME = "cells"
#: The commit marker. Content is renamed into place FIRST and this sidecar written LAST, so a cell
#: without a sidecar is a publication that did not finish and is not adoptable.
DIGEST_SUFFIX = ".digest"
#: How deep under an adoption root a ``cells/`` directory is looked for. Two layouts are in use and
#: both are this job's own tree: the launcher derives ``<OUTPUT_DIR>/<RUN_NAME>/fit_probe/attempts/
#: att-X/cells`` (depth 5 from ``OUTPUT_DIR``, depth 2 from the attempts root), and the submit wrapper
#: puts an attempt level ABOVE ``OUTPUT_DIR`` as well, which puts ``cells`` at depth 6 from the M1
#: root. A bound exists at all because an adoption root pointed at a bucket root must not walk the
#: bucket.
ADOPTION_SCAN_DEPTH = 6
#: The second bound: directories visited before the scan gives up and says so. Adoption is an
#: optimization in front of a 3.5-hour ladder, so it stops rather than becoming the cost.
ADOPTION_SCAN_LIMIT = 4096
#: What the run-level table records for a cell this attempt measured itself.
PROVENANCE_MEASURED = "measured"
#: ...and the prefix for one it adopted, followed by the artifact path it adopted.
ADOPTED_PREFIX = "adopted from "

#: Where a reported peak came from. The first two are attributable UPPER bounds on this cell's
#: footprint; the third is a cell-local LOWER bound, good enough to refuse a cell and never good
#: enough to authorize one (review W1, A3).
PEAK_SOURCE_RUNTIME_RESET = "runtime high-water mark after reset"
PEAK_SOURCE_RUNTIME_RAISED = "runtime high-water mark raised by this region"
PEAK_SOURCE_ANALYSIS = "compiled memory analysis"
#: A cell whose allocation was refused reports the capacity it hit; it is never an authorization.
PEAK_SOURCE_REFUSED = "device capacity after a refused allocation"
AUTHORIZING_PEAK_SOURCES = (PEAK_SOURCE_RUNTIME_RESET, PEAK_SOURCE_RUNTIME_RAISED)
PEAK_SOURCES = AUTHORIZING_PEAK_SOURCES + (PEAK_SOURCE_ANALYSIS, PEAK_SOURCE_REFUSED)

#: k=2 is the predeclared primary; k=4 is EXPLORATORY and runnable only in a cell M1 measured.
LADDER_K = (2, 4)
#: Powers-of-two divisors of GBS 256 spanning the default 32 (the reviewer accepted this rationale).
LADDER_MICROBATCH = (8, 16, 32, 64)
#: BOTH arms are measured. Their forward/backward graphs differ, so one's peak is not the other's
#: evidence -- the flaw the reviewer found when the cell was only ``(microbatch, k)``.
LADDER_ARMS = ("rollout", "one_step")
#: Repeats per cell. One trial cannot show a cell that only fits when the neighbours are idle.
TRIALS_PER_CELL = 2

#: THE RECIPE IS EVERYTHING EXCEPT THESE, each excluded for a written reason (review F1, LS-4).
#:
#: The first version listed 24 keys it believed were footprint-bearing, and the reviewer measured
#: what that list could not see: changing ``action_tokens``, ``pre_context_tokens``,
#: ``flash_block_sizes`` or ``latent_frames`` left the digest UNCHANGED, although every one of them
#: feeds the adapter construction at ``wan_ti2v_side_adapter_trainer._build_adapters``. An allowlist
#: of what to include is a list somebody has to remember to extend; this is a **denylist**, so a key
#: added to the YAML tomorrow is bound by default and a key that leaves the fingerprint has to be
#: argued for here, in writing, in the diff a reviewer reads.
_CELL = "the cell: the ladder varies it and M1 authorizes it per cell rather than globally"
_DESTINATION = "a destination or run identity: it names where output goes, never what is computed"
_SCHEDULE = "schedule length or cadence: it decides how MANY steps run, never what one step compiles to"
#: F6. A fourth category, added as a reviewed decision rather than an edit (the rule this dict states
#: about itself). The exclusion declaration decides WHICH cells the ladder visits and why; it does not
#: change what any one cell compiles to. Keeping it OUT of the fingerprint is load-bearing: a cell is
#: identified by its own recipe, so declaring cell X unreachable must not invalidate the banked
#: artifacts of cells Y -- which is precisely what F5 exists to keep. The run-level table digest covers
#: the declaration instead, so a reader of the authorization still sees it.
_EXCLUSION = (
    "the cell-exclusion declaration: it decides WHICH cells the ladder visits, never what one cell compiles to"
)

FINGERPRINT_EXCLUSIONS = {
    "pos_rollout_arm": _CELL,
    "pos_microbatch": _CELL,
    "pos_rollout_k": _CELL,
    "run_name": _DESTINATION,
    "output_dir": _DESTINATION,
    "base_output_directory": _DESTINATION,
    "checkpoint_dir": _DESTINATION,
    "tensorboard_dir": _DESTINATION,
    "metrics_dir": _DESTINATION,
    "jax_cache_dir": _DESTINATION,
    "pos_resume_parent": _DESTINATION,
    "pos_recipe_lock": _DESTINATION,
    "pos_fit_authorization": _DESTINATION,
    "pos_fit_adoption_root": _DESTINATION,
    "pos_fit_excluded_cells": _EXCLUSION,
    "pos_fit_exclusion_reason": _EXCLUSION,
    "pos_run_report": _DESTINATION,
    "pos_anchor_certificate": _DESTINATION,
    "pos_benchmark_row": _DESTINATION,
    "pos_dev_certificate": _DESTINATION,
    "pos_eval_phase": _DESTINATION,
    "wandb_project": _DESTINATION,
    "wandb_entity": _DESTINATION,
    "max_train_steps": _SCHEDULE,
    "max_train_samples": _SCHEDULE,
    "num_train_epochs": _SCHEDULE,
    "eval_every": _SCHEDULE,
    "checkpoint_every": _SCHEDULE,
    "learning_rate_schedule_steps": _SCHEDULE,
    "warmup_steps_fraction": _SCHEDULE,
}
#: The categories a key may be excluded under. A new reason string is a reviewed decision, not an
#: edit: the test refuses any exclusion whose reason is not one of these three.
FINGERPRINT_EXCLUSION_REASONS = (_CELL, _DESTINATION, _SCHEDULE, _EXCLUSION)

#: The tensor geometry every activation is sized by. Recorded SEPARATELY in the context (and human
#: readable there) as well as being inside the fingerprint, so a refusal can name the shapes.
GEOMETRY_KEYS = ("height", "width", "num_frames", "latent_height", "latent_width", "latent_channels")

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
    #: F5b, BLOCKER 2: the RUNNING BYTES. ``code_sha`` is the commit this program says it is;
    #: this is what is on disk. Adoption compares the whole context byte-for-byte, so binding the
    #: manifest here is what makes "measured by other code" a refusal rather than a hope.
    manifest_digest: str
    model_revision: str
    device_kind: str
    device_count: int
    geometry: tuple
    recipe_fingerprint: str

    def as_payload(self) -> dict:
        return {
            "code_sha": str(self.code_sha),
            "manifest_digest": str(self.manifest_digest),
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
            manifest_digest=str(payload["manifest_digest"]),
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
    """A canonical, JSON-stable rendering — nested containers included, so a dict key cannot hide.

    ``flash_block_sizes`` is a dict and ``logical_axis_rules`` is a list of lists; rendering either
    with ``str()`` was survivable, but rendering them canonically is what lets the fingerprint be a
    fingerprint of the whole recipe rather than of the parts that happened to be scalars.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def config_recipe(config) -> dict:
    """Every declared config value except :data:`FINGERPRINT_EXCLUSIONS` — the canonical recipe."""
    return {
        str(key): _plain(value)
        for key, value in sorted(config_keys(config).items())
        if str(key) not in FINGERPRINT_EXCLUSIONS
    }


def config_keys(config) -> dict:
    """Every key this config declares, on each config object the probe sees.

    ``HyperParameters`` exposes ``get_keys()`` -- the only DECLARED way to ask it what it has -- and
    a plain object exposes ``__dict__``. Two-argument ``getattr`` only (issue #11).
    """
    try:
        return dict(config.get_keys())
    except (AttributeError, TypeError, ValueError):
        pass
    if isinstance(config, Mapping):
        return dict(config)
    return dict(vars(config))


def recipe_fingerprint(config) -> str:
    """A digest of the WHOLE declared recipe outside the cell and the destinations (LS-4).

    The reviewer measured the old curated list going blind on ``action_tokens``,
    ``pre_context_tokens``, ``flash_block_sizes`` and ``latent_frames`` — every one of which feeds
    the adapter the probe is measuring. Inclusion is now the default and exclusion is the thing that
    has to be written down, so the failure mode has flipped from "silently unbound" to "noisily
    over-bound", which costs a re-measure rather than a wrong authorization.
    """
    return _digest(config_recipe(config))


#: Why a cell was left out of the measured set, when it was not DECLARED unreachable.
SKIPPED_NON_DIVIDING = "microbatch does not divide the logical batch, so this cell is not a runnable program"


def declared_exclusion_reason(config) -> str:
    """The one reason string covering this run's declared exclusions. Empty when there are none."""
    return str(declared(config, "pos_fit_exclusion_reason") or "").strip()


def parse_excluded_cells(config) -> tuple:
    """Cells this run DECLARES unreachable — parsed strictly, or refused loudly (F6).

    M1-4 attempt 1 banked 12 of 16 cells and then died at ``one_step microbatch=32 k=2`` with
    ``bad_smem_address`` — the same cell and the same chip fault as M1-3 attempt 2, on a different VM
    on a different day. **2/2 on one cell is deterministic**, not a zone event: an XLA codegen fault
    under the F4 scan at chunk width >= 32 for the one_step loss on v6e-8. Four unreachable cells were
    holding twelve good ones hostage, because the table only publishes when the ladder finishes.

    A declared exclusion lets the table publish and still account for every cell. It is deliberately
    unpleasant to declare: every entry must name an arm the experiment runs and a microbatch and
    horizon the ladder actually visits, duplicates are refused, and a non-empty list without
    ``pos_fit_exclusion_reason`` is refused — an undocumented exclusion is a cell that quietly stopped
    being measured, which is the failure this mechanism exists to make impossible rather than easy.
    """
    raw = declared(config, "pos_fit_excluded_cells")
    if isinstance(raw, (list, tuple)):
        entries = [str(item).strip() for item in raw]
        blank = not entries
    else:
        text = str(raw or "").strip()
        # A BLANK declaration is no declaration. A declaration made of punctuation is a malformed one:
        # `pos_fit_excluded_cells=","` used to parse as "no exclusions", so a list written to keep the
        # probe away from the deterministic-fault cell would have walked straight into it (F6b MINOR).
        blank = not text
        entries = [chunk.strip() for chunk in text.replace("\n", ",").split(",")] if text else []
    if blank:
        return ()
    # Review F6b, MINOR: empty chunks used to be discarded before validation, so
    # `pos_fit_excluded_cells=","` parsed as NO exclusions -- a declaration written to keep the probe
    # away from the deterministic-fault cell would have walked straight into it. "Malformed = loud"
    # has to include the malformation that looks like nothing.
    if any(not entry for entry in entries):
        raise ValueError(
            f"pos_fit_excluded_cells={raw!r} contains an empty token (a leading, trailing or doubled "
            f"comma). An exclusion list is read by a machine that will otherwise silently measure a cell "
            f"you meant to declare unreachable; write the entries with single commas and no trailing one."
        )
    if not declared_exclusion_reason(config):
        raise ValueError(
            f"pos_fit_excluded_cells declares {entries} but pos_fit_exclusion_reason is empty. An exclusion "
            f"removes a cell from the measured ladder, and the table records WHY; an undocumented one is a "
            f"cell that quietly stopped being measured."
        )
    cells: list[FitCell] = []
    for entry in entries:
        fields = entry.split(":")
        if len(fields) != 3:
            raise ValueError(f"{entry!r} is not an exclusion: it needs three fields, 'arm:microbatch:k'")
        arm, microbatch, horizon = (field.strip() for field in fields)
        if arm not in LADDER_ARMS:
            raise ValueError(f"{entry!r} names arm {arm!r}; exp_06 declares {list(LADDER_ARMS)}")
        for label, value in (("microbatch", microbatch), ("k", horizon)):
            if not value.isdigit() or int(value) <= 0:
                raise ValueError(f"{entry!r}: {label} must be a positive whole number, got {value!r}")
        cell = FitCell(arm=arm, microbatch=int(microbatch), k_b=int(horizon))
        if cell.microbatch not in LADDER_MICROBATCH or cell.k_b not in LADDER_K:
            raise ValueError(
                f"{entry!r} is not a cell this ladder visits (microbatches {list(LADDER_MICROBATCH)}, "
                f"horizons {list(LADDER_K)}): excluding a cell that never runs hides a typo rather than a fault"
            )
        if cell in cells:
            raise ValueError(f"{entry!r} is declared twice; one cell has one exclusion")
        cells.append(cell)
    return tuple(cells)


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


#: Excluded from the deployed manifest, with a reason: a test cannot change what a measurement costs,
#: and hashing the test tree would make every red-first round invalidate every banked cell.
MANIFEST_EXCLUDED_DIR = "tests"
_MANIFEST_CACHE: dict = {}


def deployed_manifest_digest(root: str | None = None) -> str:
    """The RUNNING BYTES of the deployed package: sha256 over every ``.py`` outside ``tests/``.

    Review F5b, BLOCKER 2. ``code_sha`` names a commit; this names what is actually on disk, and it is
    what :class:`ProbeContext` binds and adoption compares. A dirty checkout, a stale tarball, a
    hand-edited module on a worker and a caller-supplied ``COMMIT`` all move it or fail to reproduce
    it, and none of them can be argued with.

    Length-framed records in sorted path order — the serialization discipline
    :func:`snapshot_manifest_digest` earned in review F1b/W1, so exactly one tree of files maps to any
    given stream and a reshuffle at equal total bytes is not invisible. Cached per process for the
    default root: the bytes cannot change under a running process, and re-hashing the package on
    every context derivation would be a cost with no answer attached.
    """
    package = pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parent
    key = str(package)
    if root is None and key in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[key]
    files = sorted(
        path
        for path in package.rglob("*.py")
        if MANIFEST_EXCLUDED_DIR not in path.relative_to(package).parts and "__pycache__" not in path.parts
    )
    if not files:
        raise ValueError(
            f"{package} holds no deployed Python: a probe that cannot identify the code it is running "
            f"cannot publish a measurement OF that code"
        )
    digest = hashlib.sha256()
    digest.update(f"exp06.deployed_manifest.v1\n{len(files)}\n".encode("utf-8"))
    for path in files:
        relative = str(path.relative_to(package)).encode("utf-8")
        body = path.read_bytes()
        digest.update(f"{len(relative)} {len(body)}\n".encode("utf-8"))
        digest.update(relative)
        digest.update(body)
    computed = digest.hexdigest()
    if root is None:
        _MANIFEST_CACHE[key] = computed
    return computed


def _git_dirty_paths(start: pathlib.Path) -> tuple:
    """The manifest-bearing files git reports as modified — empty when the deployment is clean."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    dirty = []
    for line in completed.stdout.splitlines():
        name = line[3:].strip().strip('"')
        # Scoped to what the manifest hashes: a dirty test file cannot change a measurement, and
        # blocking on one would make this round's own red-first workflow unrunnable.
        if name.endswith(".py") and f"{MANIFEST_EXCLUDED_DIR}/" not in name:
            dirty.append(name)
    return tuple(sorted(dirty))


def derive_code_sha(
    *, module_file: str | None = None, environ: Mapping | None = None, manifest: str | None = None
) -> str:
    """The commit this program CLAIMS to be, cross-checked — never the thing adoption trusts.

    **Review F5b, BLOCKER 2: a commit is not the running bytes.** This read ``git rev-parse HEAD``
    and, failing that, the launcher's ``COMMIT`` — and the F5 delta was its own counterexample: its
    running bytes were uncommitted while the SHA it derived read ``a3ba5c0``. A dirty measurement-code
    change could therefore adopt cells measured before that change. What identifies the code now is
    :func:`deployed_manifest_digest`, bound into :class:`ProbeContext` beside this field; ``code_sha``
    survives as the label the launcher's prerequisite check compares, and two rules keep it honest:

    * **A process that DECLARES a commit must BE that commit.** ``COMMIT`` is the launcher's assertion
      "this job runs commit X"; a process making it from a tree with uncommitted measurement code is
      publishing provenance it cannot support, and is refused loudly rather than warned. A process
      that declares nothing (a developer's checkout) is identified by its manifest alone — it still
      runs, and it still cannot masquerade as a commit.

      **This is NOT a whole-checkout cleanliness check, and must not be described as one** (review
      F5c). Its scope is exactly the manifest's: ``.py`` files under ``maxdiffusion`` outside
      ``tests/``. A dirty YAML does not refuse here — its LOADED VALUES are bound by
      ``recipe_fingerprint`` instead, which is the binding that actually decides the footprint — and a
      dirty test file does not refuse either, because a test cannot change what a measurement costs.
    * **A deployment without git identifies itself by CONTENT.** On a worker the code arrives as a
      tarball with no git objects, and ``COMMIT`` is then an environment variable anybody can set. It
      may label the artifact; it may not be the only thing standing behind it, so the derivation
      refuses unless a manifest is bound alongside it.
    """
    environ = os.environ if environ is None else environ
    stamped = str(environ.get("COMMIT", "")).strip()
    start = pathlib.Path(module_file or __file__).resolve().parent
    head = _git_head(start)
    if head and stamped and head != stamped:
        raise ValueError(
            f"the running checkout is at {head} but COMMIT declares {stamped}: an HBM measurement is a "
            f"measurement OF A PROGRAM, and these are two of them"
        )
    if stamped and head:
        dirty = _git_dirty_paths(start)
        if dirty:
            raise ValueError(
                f"COMMIT declares {stamped} but this tree has uncommitted measurement code: "
                f"{list(dirty[:5])}{' ...' if len(dirty) > 5 else ''}. A run that publishes under a commit "
                f"must BE that commit -- otherwise the authorization names bytes nobody can retrieve, and a "
                f"later attempt adopts cells measured by code that no longer exists. Commit the tree (or "
                f"clear COMMIT to run identified by content alone) and re-launch."
            )
    if stamped and not head and not str(manifest or ""):
        raise ValueError(
            f"COMMIT declares {stamped} but there is no git checkout to verify it against and no content "
            f"manifest was bound. An environment variable is a claim, not an identity: a deployment without "
            f"git is identified by the bytes it is running, or it is not identified at all."
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


#: A resolved HF snapshot directory is named by its commit. Anything else is not an immutable id.
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")


#: Above this, hashing a local directory's bytes costs more than the measurement it identifies, and
#: the only honest identity left is the immutable snapshot commit. Production names a HF repo and
#: takes the remote branch; a local directory is a development affordance, and it is bounded.
LOCAL_SNAPSHOT_MAX_BYTES = 2 * 1024**3


def snapshot_manifest_digest(directory: str) -> str:
    """A CONTENT-bound identity for a local model directory: the file bytes, hashed.

    Review F1b, MAJOR 5: this hashed ``(relative path, size)`` only, so an in-place byte change — or a
    swap for another checkpoint of the same shape, including a same-length transformer config edit
    that changes the graph — left the identity unchanged and the authorization standing. Paths and
    sizes are metadata; the reviewer's instruction was to bind the contents.

    Bytes are hashed in sorted path order, in bounded chunks. Above
    :data:`LOCAL_SNAPSHOT_MAX_BYTES` the derivation refuses and asks for a resolved snapshot commit,
    because at 5B-parameter scale a full re-hash on every probe and every training start would cost
    more than the thing it identifies — and the remote branch already has an immutable id.
    """
    root = pathlib.Path(directory)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(
            f"{directory} contains no files, so there is no model here to identify: a probe cannot "
            f"measure the footprint of a model that is not on this machine"
        )
    total = sum(path.stat().st_size for path in files)
    if total > LOCAL_SNAPSHOT_MAX_BYTES:
        raise ValueError(
            f"{directory} holds {total} bytes, above the {LOCAL_SNAPSHOT_MAX_BYTES}-byte ceiling for "
            f"content-hashing a local model. Identify it by a resolved snapshot commit instead "
            f"(prefetch the repo and name it), because a metadata-only identity would not notice an "
            f"in-place byte change and an authorization must not survive one."
        )
    # LENGTH-FRAMED records (review F1b/W1, A5). Concatenating `path + NUL + bytes` is a chosen
    # serialization with an ambiguous parse: one file named `a` holding `Xb\0Y` produced the same
    # stream as two files `a=X` and `b=Y`, so a reshuffle of a snapshot's contents at equal total
    # bytes was invisible. Every record now declares its own lengths, and the count is framed too, so
    # exactly one tree of files maps to any given stream.
    digest = hashlib.sha256()
    digest.update(f"exp06.snapshot.v2\n{len(files)}\n".encode("utf-8"))
    for path in files:
        relative = str(path.relative_to(root)).encode("utf-8")
        size = path.stat().st_size
        digest.update(f"{len(relative)} {size}\n".encode("utf-8"))
        digest.update(relative)
        with open(path, "rb") as handle:
            read = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                read += len(chunk)
        if read != size:
            raise ValueError(f"{path} changed size while it was being hashed ({size} -> {read})")
    return digest.hexdigest()


def derive_model_revision(config) -> str:
    """An IMMUTABLE identity for the model being measured, or a refusal (review F1, LS-10).

    The previous version had two soft spots and the reviewer named both: every local directory
    identified as ``@local-dir`` regardless of contents, and a resolution failure became
    ``@no-local-snapshot:<Exception>`` — so two unresolved models, or two entirely different local
    ones, compared EQUAL and each could authorize the other's cells.

    Now: a remote name resolves to its snapshot COMMIT (immutable by construction) or the derivation
    fails closed; a local directory is identified by a manifest of its contents. There is no fallback
    string, because a fallback string is a stamp — the thing this whole module exists not to accept.
    """
    name = str(declared(config, "pretrained_model_name_or_path"))
    if os.path.isdir(name):
        return f"{name}@manifest:{snapshot_manifest_digest(name)}"
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(name, local_files_only=True)
    except Exception as error:  # noqa: BLE001 -- report the cause, then fail closed
        raise ValueError(
            f"no local snapshot of {name!r} could be resolved ({type(error).__name__}: {error}). An HBM "
            f"measurement is a measurement OF A MODEL; without an immutable revision this probe would be "
            f"publishing numbers about a model it cannot name. Prefetch the snapshot on this host "
            f"(bash_scripts/prefetch_hf_snapshot.sh) and re-run."
        ) from error
    commit = pathlib.Path(resolved).name
    if not _REVISION_RE.match(commit):
        raise ValueError(
            f"{name!r} resolved to {resolved!r}, whose leaf {commit!r} is not a snapshot commit: an "
            f"identity that is not immutable cannot bind a measurement to the model it measured"
        )
    return f"{name}@{commit}"


def derive_probe_context(config, *, devices: Sequence | None = None, environ: Mapping | None = None) -> ProbeContext:
    """Everything a measurement is a measurement of, derived from the running program."""
    kind, count = derive_device_signature(devices)
    manifest = deployed_manifest_digest()
    return ProbeContext(
        code_sha=derive_code_sha(environ=environ, manifest=manifest),
        manifest_digest=manifest,
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
    #: WHERE the peak came from, and REQUIRED — there is no default, because a default would be a
    #: claim about provenance that nobody made. A compiled-memory analysis is a LOWER bound on the
    #: footprint and a "<= 90% of capacity" rule needs an UPPER one, so this field is what lets
    #: `cell_verdict` refuse to authorize on a floor (review W1, A3). Re-decided on load.
    peak_source: str

    def as_payload(self) -> dict:
        return {
            "cell": self.cell.as_payload(),
            "peak_source": str(self.peak_source),
            "context_digest": str(self.context_digest),
            "compile_seconds": float(self.compile_seconds),
            "step_seconds": float(self.step_seconds),
            "eval_seconds": float(self.eval_seconds),
            "checkpoint_seconds": float(self.checkpoint_seconds),
            "peak_bytes": int(self.peak_bytes),
            "capacity_bytes": int(self.capacity_bytes),
            "reservation_failures": int(self.reservation_failures),
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "CellMeasurement":
        """Deserialize a recorded measurement so its verdict can be RECOMPUTED on load (LS-5)."""
        try:
            return cls(
                cell=FitCell.from_payload(payload["cell"]),
                context_digest=str(payload["context_digest"]),
                compile_seconds=float(payload["compile_seconds"]),
                step_seconds=float(payload["step_seconds"]),
                eval_seconds=float(payload["eval_seconds"]),
                checkpoint_seconds=float(payload["checkpoint_seconds"]),
                peak_bytes=int(payload["peak_bytes"]),
                capacity_bytes=int(payload["capacity_bytes"]),
                reservation_failures=int(payload["reservation_failures"]),
                peak_source=str(payload["peak_source"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{payload!r} is not a recorded cell measurement: {error}") from error


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
    if str(measurement.peak_source) not in PEAK_SOURCES:
        raise ValueError(
            f"{measurement.peak_source!r} is not a peak source this probe produces ({list(PEAK_SOURCES)}); "
            f"an unrecognised source cannot be judged against the authorization rule"
        )
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
    if str(measurement.peak_source) not in AUTHORIZING_PEAK_SOURCES:
        # A floor cannot clear a ceiling rule. `Compiled.memory_analysis()` is cell-local, which is
        # why it is trusted to REFUSE a cell, but it omits whatever the program closed over as a
        # constant -- so a cell whose only evidence is the analysis may be far larger than it looks,
        # and authorizing it would be authorizing a lower bound against a 90% ceiling (review W1, A3).
        reasons.append("peak_source")
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
            "peak_source": str(measurement.peak_source),
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
                peak_source=(
                    trials[0].peak_source
                    if len({str(trial.peak_source) for trial in trials}) == 1
                    else PEAK_SOURCE_ANALYSIS  # a mixed cell is only as good as its weakest evidence
                ),
            )
        )
    return tuple(aggregated)


def _positive_int(value, what: str) -> int:
    number = int(value)
    if number != float(value) or number <= 0:
        raise ValueError(f"{what} must be a positive whole number of steps, got {value!r}")
    return number


def evaluation_count(max_train_steps: int, eval_every: int) -> int:
    """How many times the LOOP evaluates — the cadence AND the final step (LS-8).

    ``pos_rollout_loop.should_evaluate`` is ``step >= 1 and (step % eval_every == 0 or step ==
    max_train_steps)``; this closed form is asserted against that predicate over a spread of lengths
    and cadences, so the projection's count is checked against production rather than restated from
    it.
    """
    steps = _positive_int(max_train_steps, "max_train_steps")
    cadence = _positive_int(eval_every, "eval_every")
    return steps // cadence + (1 if steps % cadence else 0)


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
    projection from a negative evaluation cost and a negative checkpoint cost.

    **And the arithmetic now matches the run being projected (review F1, LS-8).** Two corrections,
    both measured by the reviewer against the actual loop:

    * ``pos_rollout_loop.should_evaluate`` fires on the cadence **and at the final step**, so a
      10,001-step run at cadence 1,000 evaluates eleven times, not ten. Flooring understated every
      projection whose length was not an exact multiple of its cadence.
    * ``checkpoint_every`` is **not consumed by the loop at all**: ``run_loop`` writes its
      checkpoints inside the evaluation branch. Projecting an independent cadence was projecting a
      run nobody can start, so a config whose two cadences disagree is REFUSED here rather than
      quietly projected — the launcher may expose the key, but the probe will not pretend production
      honours it.
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
    if checkpoint_cadence != eval_cadence:
        raise ValueError(
            f"checkpoint_every={checkpoint_cadence} differs from eval_every={eval_cadence}, but "
            f"pos_rollout_loop.run_loop writes its checkpoints inside the evaluation branch — it never "
            f"reads an independent checkpoint cadence. Projecting one would describe a run that cannot "
            f"be started; set the two equal, or wire the cadence into the loop first."
        )
    evaluations = evaluation_count(steps, eval_cadence)
    checkpoints = evaluations
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
    """What M1 learned: one context, one aggregated measurement per cell, and the run it projects for.

    **Every derived field is computed in** :meth:`as_payload` **and none is stored** — the cell
    lists, the verdicts and the projections are all functions of ``(context, measurements,
    projection_inputs)``. That is what makes :func:`load_authorization` able to REVALIDATE rather
    than trust: it rebuilds this object from the recorded inputs and requires the whole payload to
    come out identical. The reviewer edited a recorded measurement to capacity-level peak plus a
    reservation failure, re-hashed, and the old loader (which validated only the three cell lists)
    authorized the cell anyway.
    """

    context: ProbeContext
    measurements: tuple
    projection_inputs: tuple
    #: F5. One ``(cell, text)`` pair per aggregated cell, in the same order: ``"measured"`` for a cell
    #: this attempt measured, ``"adopted from <path>"`` for one it took from a prior attempt's
    #: publication. It is a RECORD, not a derivation — the artifact a cell was adopted from is not
    #: recomputable from the numbers — so it is digest-bound like everything else here and no more.
    #: Left empty it fills itself in as all-measured, which is what an uninterrupted run is.
    provenance: tuple = ()
    #: F6. ``((cell, reason), ...)`` for cells DECLARED unreachable, and for cells the divisibility
    #: rule dropped. Neither was measured; both are recorded, because a table that is silent about a
    #: cell is a table a reader cannot distinguish from one that forgot it.
    exclusions: tuple = ()
    skipped: tuple = ()

    def _assert_one_status_per_cell(self) -> None:
        """A cell has ONE status. Review F6b, MAJOR 1.

        The serializer emitted the four lists independently, so an edited-and-rehashed table could
        name a cell BOTH authorized and excluded — and :func:`assert_cell_authorized` returns on the
        authorized list before it ever looks at exclusions, so the smuggled cell would have run. This
        lives on the evidence rather than in the loader because BOTH paths build one of these: the
        probe when it publishes, the loader when it re-decides.
        """
        buckets = (
            ("measured", [entry.cell for entry in self.measurements]),
            ("excluded", [cell for cell, _ in self.exclusions]),
            ("skipped", [cell for cell, _ in self.skipped]),
        )
        for name, cells in buckets:
            duplicates = sorted({str(cell) for cell in cells if cells.count(cell) > 1})
            if duplicates:
                raise ValueError(f"{name} names {duplicates} twice; a cell has one status, not several")
        for (left, first), (right, second) in (
            (buckets[0], buckets[1]),
            (buckets[0], buckets[2]),
            (buckets[1], buckets[2]),
        ):
            both = sorted({str(cell) for cell in set(first) & set(second)})
            if both:
                raise ValueError(
                    f"{both} are recorded as both {left} and {right}: a cell has ONE status, and a table "
                    f"that gives it two settles nothing about which one a training run may quote"
                )
        for cell, why in self.exclusions:
            if not str(why).strip():
                raise ValueError(
                    f"{cell} is recorded as excluded with no reason. An exclusion removes a cell from the "
                    f"measured ladder; an undocumented one is a cell that quietly stopped being measured"
                )

    def __post_init__(self) -> None:
        if not self.provenance:
            object.__setattr__(self, "provenance", tuple((m.cell, PROVENANCE_MEASURED) for m in self.measurements))

    def projections(self) -> tuple:
        inputs = dict(self.projection_inputs)
        return tuple(
            project_wall_clock(
                measurement,
                max_train_steps=inputs["max_train_steps"],
                eval_every=inputs["eval_every"],
                checkpoint_every=inputs["checkpoint_every"],
            )
            for measurement in self.measurements
            if cell_verdict(measurement).fits
        )

    def as_payload(self) -> dict:
        self._assert_one_status_per_cell()
        recorded = [cell for cell, _ in self.provenance]
        if recorded != [measurement.cell for measurement in self.measurements]:
            raise ValueError(
                f"the provenance record names {[str(cell) for cell in recorded]} but the measurements are "
                f"{[str(m.cell) for m in self.measurements]}: a table whose account of where its numbers "
                f"came from does not line up with its numbers settles nothing about either"
            )
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
            "cell_provenance": [{**cell.as_payload(), "provenance": str(text)} for cell, text in self.provenance],
            "excluded_cells": [{**cell.as_payload(), "reason": str(why)} for cell, why in self.exclusions],
            "skipped_cells": [{**cell.as_payload(), "reason": str(why)} for cell, why in self.skipped],
            "exclusion_reason": str(self.exclusions[0][1]) if self.exclusions else "",
            "context": self.context.as_payload(),
            "context_digest": self.context.digest(),
            "authorized_cells": authorized,
            "measured_cells": measured,
            "refused_cells": refused,
            "headroom_fraction": HEADROOM_FRACTION,
            "projection_inputs": dict(self.projection_inputs),
            "measurements": [measurement.as_payload() for measurement in self.measurements],
            "projections": [dict(projection) for projection in self.projections()],
        }


def build_evidence(
    context: ProbeContext,
    measurements: Sequence[CellMeasurement],
    *,
    max_train_steps: int,
    eval_every: int,
    checkpoint_every: int,
    provenance: Mapping | None = None,
    exclusions: Sequence | None = None,
    skipped: Sequence | None = None,
) -> ProbeEvidence:
    """Bind the trials to the context they were measured under, aggregate them, and project.

    ``provenance`` maps each cell to how this attempt came by it (F5). Omitted, every cell is
    recorded as measured — which is what a run with no adoption root did and still does.

    ``exclusions`` and ``skipped`` are the cells that were NOT measured and why (F6): declared
    unreachable, or dropped because their microbatch does not divide the logical batch. They are
    carried so the published table accounts for every cell of the ladder it ran rather than being
    silent about the ones that are missing.
    """
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
    if provenance is None:
        record = tuple((entry.cell, PROVENANCE_MEASURED) for entry in aggregated)
    else:
        missing = sorted(str(entry.cell) for entry in aggregated if entry.cell not in provenance)
        if missing:
            raise ValueError(f"no provenance was recorded for {missing}: every published cell says where it came from")
        record = tuple((entry.cell, str(provenance[entry.cell])) for entry in aggregated)
    declared_out = tuple((cell, str(why)) for cell, why in (exclusions or ()))
    dropped = tuple((cell, str(why)) for cell, why in (skipped or ()))
    overlap = {cell for cell, _ in declared_out} & {entry.cell for entry in aggregated}
    if overlap:
        raise ValueError(
            f"{sorted(str(cell) for cell in overlap)} are recorded as BOTH measured and excluded; an excluded "
            f"cell is one this run never built, so a measurement of it is a contradiction"
        )
    evidence = ProbeEvidence(
        context=context,
        measurements=aggregated,
        projection_inputs=(
            ("max_train_steps", _positive_int(max_train_steps, "max_train_steps")),
            ("eval_every", _positive_int(eval_every, "eval_every")),
            ("checkpoint_every", _positive_int(checkpoint_every, "checkpoint_every")),
        ),
        provenance=record,
        exclusions=declared_out,
        skipped=dropped,
    )
    evidence.projections()  # project now, so a cadence production cannot honour fails HERE (LS-8)
    return evidence


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
    """Read an authorization and RE-DECIDE it from the measurements it records (review F1, LS-5).

    The digest proves the bytes were not edited *after* publication. It proves nothing about whether
    the published lists follow from the published numbers — and the reviewer showed exactly that gap:
    it changed an authorized cell's recorded measurement to a capacity-level peak plus a reservation
    failure, recomputed the (unkeyed) digest, and both loading and :func:`assert_cell_authorized`
    accepted the cell. The old loader validated three lists of cell names and never looked at the
    measurements underneath them.

    So loading now **rebuilds the whole artifact**: it deserializes the context, every measurement
    and the projection inputs, reconstructs a :class:`ProbeEvidence` from them, and requires the
    payload that object produces to be byte-identical to the payload on disk. Every verdict, every
    cell list, every projection and the pinned headroom constant are therefore recomputed from the
    recorded numbers rather than read off the artifact. An authorization whose lists do not follow
    from its own measurements does not load at all.
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
    if payload.get("headroom_fraction") != HEADROOM_FRACTION:
        raise ValueError(
            f"{path}: it was decided against a headroom of {payload.get('headroom_fraction')!r}, but this "
            f"code refuses above {HEADROOM_FRACTION}. Plan §4-P1 pins the rule; an artifact decided under "
            f"another one is not evidence about what this code will run."
        )
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

    # The three cell lists are shape-checked FIRST, so a malformed one is diagnosed precisely rather
    # than surfacing as "the re-decision does not reproduce the artifact".
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

    recorded = payload.get("measurements")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError(
            f"{path}: an authorization records the measurements it was decided from; this one records none"
        )
    measurements = [CellMeasurement.from_payload(entry) for entry in recorded]
    foreign = sorted({m.context_digest for m in measurements if m.context_digest != rebuilt.digest()})
    if foreign:
        raise ValueError(
            f"{path}: measurements bound to context digest(s) {foreign} were published under "
            f"{rebuilt.digest()} — they measure a different program from the one this artifact claims"
        )
    cells = [measurement.cell for measurement in measurements]
    if len(set(cells)) != len(cells):
        raise ValueError(
            f"{path}: a cell is recorded more than once; trials aggregate to ONE record per cell, and two "
            f"records for one cell is the shape the contradictory-duplicate defect took"
        )
    inputs = payload.get("projection_inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"{path}: an authorization records the run it projected for; this one records none")
    for field in ("excluded_cells", "skipped_cells"):
        if not isinstance(payload.get(field), list):
            raise ValueError(
                f"{path}: a {AUTHORIZATION_PROTOCOL} table accounts for every cell of the ladder it ran -- "
                f"authorized, refused, DECLARED-excluded or skipped -- and this one records no {field}"
            )
    statuses: dict = {}
    for field in ("authorized_cells", "refused_cells", "excluded_cells", "skipped_cells"):
        for entry in payload.get(field) or []:
            try:
                cell = FitCell.from_payload(entry)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}: {field} contains {entry!r}, which is not a cell: {error}") from error
            if field != "refused_cells" or cell not in statuses.get("authorized_cells", ()):
                statuses.setdefault(field, []).append(cell)
    for field, cells in statuses.items():
        duplicates = sorted({str(cell) for cell in cells if cells.count(cell) > 1})
        if duplicates:
            raise ValueError(f"{path}: {field} names {duplicates} twice; a cell has one status, not several")
    for left, right in (
        ("authorized_cells", "excluded_cells"),
        ("authorized_cells", "skipped_cells"),
        ("refused_cells", "excluded_cells"),
        ("refused_cells", "skipped_cells"),
        ("excluded_cells", "skipped_cells"),
    ):
        both = sorted({str(cell) for cell in set(statuses.get(left, ())) & set(statuses.get(right, ()))})
        if both:
            raise ValueError(
                f"{path}: {both} are listed as both {left} and {right}. A cell has ONE status; this table "
                f"gives it two, and the gate answers from whichever list it reads first."
            )
    for entry in payload.get("excluded_cells") or []:
        if not str(entry.get("reason", "")).strip():
            raise ValueError(
                f"{path}: {entry!r} is excluded with no reason recorded. An exclusion removes a cell from "
                f"the measured ladder, and a table that cannot say why is not evidence about the run."
            )
    recorded_provenance = payload.get("cell_provenance")
    if not isinstance(recorded_provenance, list):
        raise ValueError(
            f"{path}: a {AUTHORIZATION_PROTOCOL} table records, per cell, whether this attempt measured it or "
            f"adopted a prior attempt's publication; this one records none"
        )
    try:
        evidence = ProbeEvidence(
            context=rebuilt,
            measurements=tuple(measurements),
            projection_inputs=tuple(
                (key, int(inputs[key])) for key in ("max_train_steps", "eval_every", "checkpoint_every")
            ),
            provenance=tuple((FitCell.from_payload(entry), str(entry["provenance"])) for entry in recorded_provenance),
            exclusions=tuple(
                (FitCell.from_payload(entry), str(entry["reason"])) for entry in payload.get("excluded_cells") or []
            ),
            skipped=tuple(
                (FitCell.from_payload(entry), str(entry["reason"])) for entry in payload.get("skipped_cells") or []
            ),
        )
        recomputed = evidence.as_payload()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: the recorded evidence cannot be re-decided: {error}") from error
    # Canonical SERIALIZED bytes, not mapping equality (review F1b, MINOR 6): Python reads JSON `2`
    # and `2.0` as equal numbers, so retyping a recorded duration and re-hashing passed a check whose
    # docstring claimed byte-identity. The digest is computed over exactly these bytes, so comparing
    # them is both literally true and the same comparison the digest makes.
    canonical = json.dumps(recomputed, sort_keys=True).encode("utf-8")
    if canonical != json.dumps(dict(payload), sort_keys=True).encode("utf-8"):
        differing = sorted(
            key
            for key in set(recomputed) | set(payload)
            if json.dumps(recomputed.get(key), sort_keys=True) != json.dumps(dict(payload).get(key), sort_keys=True)
        )
        raise ValueError(
            f"{path}: re-deciding this artifact from its own recorded measurements does not reproduce it — "
            f"{differing} differ. The published verdicts do not follow from the published numbers, so the "
            f"artifact settles nothing (review F1, LS-5)."
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
    excluded = [
        entry
        for entry in authorization.get("excluded_cells", [])
        if {key: value for key, value in entry.items() if key != "reason"} == wanted
    ]
    if excluded:
        raise ValueError(
            f"M1 did not measure {cell.arm} at microbatch={cell.microbatch} k={cell.k_b}: this run DECLARED "
            f"that cell EXCLUDED and never built it -- {excluded[0].get('reason')!r}. It was left out on "
            f"purpose, so there is no measurement to appeal to and re-running M1 unchanged will not produce "
            f"one. Choose an authorized cell, or change the declaration and re-measure."
        )
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
# F5: a cell is published the moment it is measured, and adopted the moment it is verified.
#
# The ladder is ~3.5 hours and it published ONLY at the end, so a zone that kills the VM at 90
# minutes threw away every cell. It happened repeatedly: one attempt measured 24 of 32 cells and
# published nothing, and five attempts re-measured the same cells byte-identically. Publication
# per cell banks the work; adoption is what makes the banked work count on the next attempt.
#
# ADOPTION IS BOUND TO THE CONTEXT DIGEST, and that is a deliberately coarse policy stated here so
# nobody has to infer it: the digest carries ``code_sha``, so ANY commit — a comment, a docs-only
# descendant, a launcher tweak — makes every published cell unadoptable and the ladder re-measures
# from scratch. It over-refuses. The alternative is a curated list of "code that changes the
# footprint", which is a list somebody has to remember to extend, and the cost of forgetting is an
# HBM authorization for a program nobody measured. Over-refusing costs TPU minutes; under-refusing
# costs a 64-chip reservation. The same trade was already made for FINGERPRINT_EXCLUSIONS.
# =================================================================================================


def derive_job_identity(config) -> str:
    """WHICH RUN measured a cell — the guard against another job's cells leaking in.

    The queue exposes no job id to the workload, so the run identity is ``run_name``: the launcher
    scopes every root by it, it is fixed for the life of a submission (the submit wrapper bakes it
    into the job's environment) and it differs between submissions. Two attempts of one job share it;
    two different experiments do not.

    What it does NOT distinguish is two submissions made under the same ``run_name`` against the same
    root at the same commit with the same recipe on the same topology — and those are the same
    program by every check this module has, so adopting between them is sound rather than tolerated.
    An empty ``run_name`` is not a wildcard: :func:`adopt_published_cell` refuses on both sides.
    """
    return str(declared(config, "run_name") or "")


def cell_artifact_name(cell: FitCell) -> str:
    """``rollout_m8_k2.json`` — the same ``<arm>_m<microbatch>_k<k>`` spelling the probe checkpoint uses."""
    return f"{cell.arm}_m{int(cell.microbatch)}_k{int(cell.k_b)}.json"


def cell_publication_dir(authorization_path: str) -> str:
    """``cells/`` beside the run-level table, so a cell is attempt-scoped exactly as it is (issue #13)."""
    parent = str(authorization_path).rpartition("/")[0]
    return f"{parent}/{CELLS_DIRNAME}" if parent else CELLS_DIRNAME


def cell_marker_path(authorization_path: str, cell: FitCell) -> str:
    """The cell's COMMIT MARKER -- the one stable, discoverable name, holding one content digest.

    Everything else about a banked cell is content-addressed and immutable; this is the only name a
    second publisher can take, and taking it is safe because whatever digest it holds names a whole
    object that verifies against it (review F5b, MAJOR).
    """
    return f"{cell_publication_dir(authorization_path)}/{cell_artifact_name(cell)}{DIGEST_SUFFIX}"


def _content_for_marker(marker_path: str, digest: str) -> str:
    """``.../rollout_m8_k2.<full 64-hex digest>.json`` beside the marker that commits it.

    Review F5c, MINOR: this truncated the digest to 12 hex characters, so an object was named by 48
    bits of itself rather than by itself — and two payloads sharing that prefix would contend for one
    name, re-expressing the very ``marker-A -> content-B`` tear content-addressing exists to remove.
    A long filename is a small price for a name that is actually the content's identity.
    """
    base = str(marker_path)
    if base.endswith(DIGEST_SUFFIX):
        base = base[: -len(DIGEST_SUFFIX)]
    stem = base[: -len(".json")] if base.endswith(".json") else base
    return f"{stem}.{str(digest)}.json"


def cell_content_path(authorization_path: str, cell: FitCell, digest: str) -> str:
    return _content_for_marker(cell_marker_path(authorization_path, cell), digest)


@dataclasses.dataclass(frozen=True)
class CellArtifact:
    """One measured cell, banked: its TRIALS, and the running program they were measured on.

    The trials are stored individually rather than aggregated because aggregation is conservative and
    order-independent (:func:`aggregate_trials` takes the worst of each field): storing the aggregate
    would make an adopted cell's contribution to the table a different computation from a measured
    cell's, and the whole claim of this round is that it is the same table.
    """

    cell: FitCell
    context: ProbeContext
    job_identity: str
    trials: tuple

    def as_payload(self) -> dict:
        return {
            "protocol": CELL_PROTOCOL,
            "cell": self.cell.as_payload(),
            "job_identity": str(self.job_identity),
            "context": self.context.as_payload(),
            "context_digest": self.context.digest(),
            # Redundant with the context above, and recorded anyway: these are the three fields an
            # operator reads in a refusal message, and a refusal that can name them without rebuilding
            # a context object is a refusal that still works when the context payload is unparseable.
            "code_sha": str(self.context.code_sha),
            "device_count": int(self.context.device_count),
            "recipe_fingerprint": str(self.context.recipe_fingerprint),
            "trial_count": len(self.trials),
            "trials": [trial.as_payload() for trial in self.trials],
        }

    @classmethod
    def from_payload(cls, payload: Mapping) -> "CellArtifact":
        """Rebuild a cell artifact, refusing anything internally inconsistent BEFORE it is compared.

        Every check here is about the artifact agreeing with itself. Whether it agrees with THIS
        process is :func:`adopt_published_cell`'s question, and separating the two is what lets a
        refusal say which kind of wrong it is.
        """
        if not isinstance(payload, Mapping):
            raise ValueError(f"{payload!r} is not a cell artifact payload")
        if payload.get("protocol") != CELL_PROTOCOL:
            raise ValueError(f"protocol {payload.get('protocol')!r} is not {CELL_PROTOCOL}")
        cell = FitCell.from_payload(payload["cell"])
        context = ProbeContext.from_payload(payload["context"])
        if context.digest() != payload.get("context_digest"):
            raise ValueError(
                f"the recorded context digest {payload.get('context_digest')!r} does not describe the recorded "
                f"context: the two would bind different programs"
            )
        for field, value in (
            ("code_sha", context.code_sha),
            ("device_count", context.device_count),
            ("recipe_fingerprint", context.recipe_fingerprint),
        ):
            if payload.get(field) != value:
                raise ValueError(f"the recorded {field} {payload.get(field)!r} contradicts the recorded context")
        trials = tuple(CellMeasurement.from_payload(entry) for entry in payload["trials"])
        if not trials:
            raise ValueError("a cell artifact records the trials it was measured over; this one records none")
        if int(payload.get("trial_count", -1)) != len(trials):
            raise ValueError(f"it claims {payload.get('trial_count')!r} trials and records {len(trials)}")
        for trial in trials:
            _checked(trial)
            if trial.cell != cell:
                raise ValueError(f"a trial describes {trial.cell} in an artifact about {cell}")
            if str(trial.context_digest) != context.digest():
                raise ValueError(f"a trial of {cell} is bound to a context this artifact does not record")
        return cls(cell=cell, context=context, job_identity=str(payload.get("job_identity", "")), trials=trials)


def publish_cell_content(marker_path: str, artifact: CellArtifact) -> str:
    """Write the immutable CONTENT OBJECT for this artifact and return its digest.

    Content-addressed: the object's name carries the digest of the payload inside it, so two writers
    with different measurements write two different objects and neither can overwrite the other. That
    is what removes the tear the reviewer found — there is no name for two payloads to contend for.
    """
    if not isinstance(artifact, CellArtifact):
        raise ValueError(f"publish_cell_content takes a CellArtifact, not {type(artifact).__name__}")
    payload = artifact.as_payload()
    digest = _digest(payload)
    body = json.dumps({"payload": payload, "sha256": digest}, sort_keys=True).encode("utf-8")
    storage_publish_bytes(_content_for_marker(marker_path, digest), body)
    return digest


def commit_cell_marker(marker_path: str, artifact: CellArtifact) -> str:
    """Point the marker at one exact content object. This is the ONLY mutable name in the scheme.

    A concurrent second writer may replace it, and that is safe by construction: whatever digest the
    marker ends up holding names a whole, self-verifying object that some publisher finished writing
    before it committed. Last-writer-wins on the marker, never a mixed pair — which is precisely the
    guarantee the review asked for and the previous two-object overwrite could not give.
    """
    digest = _digest(artifact.as_payload())
    storage_publish_bytes(marker_path, f"{digest}\n".encode("utf-8"))
    return digest


def publish_cell(marker_path: str, artifact: CellArtifact) -> dict:
    """Bank one finished cell: immutable content object first, then the single commit marker.

    **Review F5b, MAJOR — two publishers could tear the pair permanently.** The previous shape wrote
    a fixed-name content file and then a digest sidecar, so two writers could finish as
    ``content-B + digest-A`` and the cell was unadoptable forever. Content objects are now named by
    their own digest and the marker names one of them, so:

    * no two payloads contend for a name — a torn interleaving is not expressible;
    * whatever the marker holds is a complete object that verifies against it;
    * a publisher that finds a BROKEN pair (marker naming an object that is missing, unreadable or
      does not hash to it) **repairs** it instead of returning early. Returning early was how the
      earlier version left a poisoned cache; the review found the window my first fix did not close.

    A complete, verifying pair is kept rather than rewritten (issue #10), and this attempt's own
    in-memory measurement decides this attempt's table either way. Nothing here raises for a storage
    condition: banking is a cache in front of a 3.5-hour ladder and must never be the thing that ends
    one.

    **What publication does not do:** it does not authenticate. The object carries an integrity digest
    and the running program's manifest; it carries no signature, so the writer set is whoever the
    bucket ACL admits. See the module docstring for the boundary and the escalation.
    """
    if not isinstance(artifact, CellArtifact):
        raise ValueError(f"publish_cell takes a CellArtifact, not {type(artifact).__name__}")
    payload = artifact.as_payload()
    digest = _digest(payload)
    if storage_exists(marker_path):
        recorded = _marker_digest(marker_path)
        intact = bool(recorded) and _content_is_intact(marker_path, recorded)
        if intact:
            if recorded != digest:
                print(
                    f"[M1] {marker_path} already commits {recorded[:12]} and this measurement is {digest[:12]}; "
                    f"keeping the published artifact -- a published artifact is adopted, never rewritten "
                    f"(issue #10). This attempt's own measurement still decides its table.",
                    flush=True,
                )
            return {**payload, "sha256": recorded if recorded == digest else digest}
        print(
            f"[M1] {marker_path} commits {recorded[:12] if recorded else '(unreadable)'} but that content object "
            f"is missing or does not hash to it; REPAIRing the pair from this attempt's own measurement rather "
            f"than leaving the cell unadoptable.",
            flush=True,
        )
    publish_cell_content(marker_path, artifact)
    commit_cell_marker(marker_path, artifact)
    return {**payload, "sha256": digest}


def _marker_digest(marker_path: str) -> str:
    try:
        recorded = storage_read_bytes(marker_path).decode("utf-8").strip()
    except Exception:  # noqa: BLE001 -- an unreadable marker is a broken pair, not a crash
        return ""
    return recorded if _DIGEST_RE.match(recorded) else ""


def _content_is_intact(marker_path: str, digest: str) -> bool:
    try:
        _read_verified_content(marker_path, digest)
    except Exception:  # noqa: BLE001 -- any failure to reproduce the committed object is "broken"
        return False
    return True


def _read_verified_content(marker_path: str, digest: str) -> Mapping:
    """The payload the marker commits to, or a refusal naming which link of the chain broke."""
    path = _content_for_marker(marker_path, digest)
    if not storage_exists(path):
        raise ValueError(
            f"{marker_path} commits {digest[:12]} but {path} does not exist: the marker is the commit, and "
            f"a commit that names no object is not a publication"
        )
    parsed = json.loads(storage_read_bytes(path).decode("utf-8"))
    stored = dict(parsed) if isinstance(parsed, Mapping) else {}
    payload = stored.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} holds no cell payload")
    recomputed = _digest(payload)
    if recomputed != digest or stored.get("sha256") != digest:
        raise ValueError(
            f"{path}: the content hashes to {recomputed[:12]} and records {str(stored.get('sha256'))[:12]}, but "
            f"the marker commits {digest[:12]} -- the object is not the one that was committed"
        )
    return payload


def load_cell_artifact(marker_path: str) -> CellArtifact:
    """Read a banked cell through its commit marker, or refuse it by name.

    The marker is the only entry point: it names the exact content object, that object is verified to
    hash to what the marker names, and only then is the payload parsed. A publication that did not
    reach its marker is invisible here, which is the safe way round.
    """
    if not storage_exists(marker_path):
        raise ValueError(
            f"{marker_path} does not exist. The marker is written last and is the commit, so its absence "
            f"means the publication did not finish -- and a half-published cell is exactly the artifact an "
            f"adopt-if-published design must not adopt"
        )
    digest = _marker_digest(marker_path)
    if not digest:
        raise ValueError(f"{marker_path} does not hold a sha256 digest; it commits nothing")
    payload = _read_verified_content(marker_path, digest)
    try:
        return CellArtifact.from_payload(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{marker_path}: {error}") from error


def _directories_under(root: str, *, depth: int, limit: int) -> tuple:
    """Every ``cells/`` directory at or above ``depth`` under ``root``, breadth first and bounded."""
    frontier, found, visited = [(str(root).rstrip("/"), 0)], [], 0
    while frontier:
        if visited >= limit:
            print(
                f"[M1] adoption scan stopped after {limit} directories under {root}: it is bounded on purpose, "
                f"and a root this wide is not an attempt tree. Point pos_fit_adoption_root at this job's M1 root.",
                flush=True,
            )
            break
        current, level = frontier.pop(0)
        visited += 1
        try:
            names = storage_list_children(current)
        except Exception as error:  # noqa: BLE001 -- a listing that fails is an empty listing here
            print(f"[M1] adoption scan: cannot list {current} ({type(error).__name__}: {error}); skipping", flush=True)
            continue
        for name in names:
            child = f"{current}/{name}"
            if name == CELLS_DIRNAME:
                found.append(child)
            elif level + 1 < depth:
                frontier.append((child, level + 1))
    return tuple(sorted(found))


def adoption_candidates(
    root: str, cell: FitCell, *, depth: int = ADOPTION_SCAN_DEPTH, limit: int = ADOPTION_SCAN_LIMIT
) -> tuple:
    """Published artifacts for this cell under an adoption root, sorted, oldest attempt first."""
    if not str(root or ""):
        return ()
    name = f"{cell_artifact_name(cell)}{DIGEST_SUFFIX}"
    return tuple(
        path
        for path in (f"{directory}/{name}" for directory in _directories_under(root, depth=depth, limit=limit))
        if storage_exists(path)
    )


def _adoption_refusal(artifact: CellArtifact, *, cell: FitCell, context: ProbeContext, job_identity: str, trials: int):
    """Why THIS process may not use that artifact — or ``""`` if it may. Every branch re-measures."""
    if artifact.cell != cell:
        return f"it records {artifact.cell} and this is {cell}"
    if str(artifact.job_identity) != str(job_identity):
        return (
            f"it was published by job/run {artifact.job_identity!r} and this job is {job_identity!r}: another "
            f"run's cells sitting under a shared root are not this run's evidence"
        )
    if artifact.context.digest() != context.digest():
        return (
            f"it was measured under a different program -- {list(artifact.context.differences(context))} differ "
            f"(measured on {artifact.context.code_sha[:12]}/{artifact.context.device_kind}x"
            f"{artifact.context.device_count}, running {context.code_sha[:12]}/{context.device_kind}x"
            f"{context.device_count}). Adoption is bound to the context digest, code_sha included."
        )
    if len(artifact.trials) != int(trials):
        return (
            f"it records {len(artifact.trials)} trial(s) and this ladder runs {int(trials)}: a cell repeats "
            f"because one trial cannot show a cell that only fits when the neighbours are idle"
        )
    return ""


def adopt_published_cell(
    cell: FitCell, *, context: ProbeContext, job_identity: str, root: str, trials: int
) -> tuple | None:
    """The trials of an already-published, fully verified measurement of this cell — or ``None``.

    ``None`` means measure it. Nothing here can fail the run: every refusal costs the cell's own
    measurement time, which is what the probe was going to spend anyway.

    **What adoption verifies, and what it cannot.** It verifies integrity (content-addressed object,
    digest-checked against its marker) and program identity (the artifact's recorded context, manifest
    of the running bytes included, byte-for-byte against this process's). It does NOT verify
    authorship: these artifacts are not authenticated, and a bucket writer holding the deployed tree
    could fabricate one. That residual is accepted pending Yixun's decision on signing; the trust
    anchor today is the bucket ACL.
    """
    if not str(root or ""):
        return None
    if not str(job_identity or ""):
        print(
            "[M1] adoption is off: this job declares no run_name, and an empty run identity would match "
            "every other empty one rather than naming a job",
            flush=True,
        )
        return None
    for path in reversed(adoption_candidates(root, cell)):
        try:
            artifact = load_cell_artifact(path)
        except Exception as error:  # noqa: BLE001 -- adoption is an optimization and may never fail a run
            # Deliberately every exception, not a curated tuple: the storage layer's own errors
            # (tensorflow's NotFoundError family) are not in the ValueError/OSError hierarchy, and a
            # read race against another attempt's publication must cost this cell a re-measure rather
            # than the 3.5-hour ladder.
            print(f"[M1] not adopting {path}: {type(error).__name__}: {error}", flush=True)
            continue
        refusal = _adoption_refusal(artifact, cell=cell, context=context, job_identity=job_identity, trials=trials)
        if refusal:
            print(f"[M1] not adopting {path}: {refusal}", flush=True)
            continue
        print(
            f"[M1] adopting {cell.arm} microbatch={cell.microbatch} k={cell.k_b} from {path} "
            f"({len(artifact.trials)} trials, peak {max(t.peak_bytes for t in artifact.trials)} bytes)",
            flush=True,
        )
        return artifact.trials, path
    return None


# =================================================================================================
# The probe itself: orchestration on the host, telemetry on the device.
# =================================================================================================


#: Steps discarded before timing: the first call compiles, the next few are not yet steady.
WARMUP_STEPS = 2
#: Steps averaged for the steady-state step time.
TIMED_STEPS = 3
#: DEV-64 is what the loop evaluates on (plan §3d); the probe measures ONE pass and scales by the
#: number of passes the cohort needs at this cell's microbatch. Recorded in the projection so the
#: scaling is visible rather than folded into a single number.
DEV_COHORT_SIZE = 64


class DeviceTelemetry:
    """THE device boundary, and now the only one: what the runtime says about its own memory.

    Everything else in the measurement path — building the arm's program, compiling it, stepping it,
    timing it, writing a checkpoint — is ordinary JAX that runs on whatever backend is present. This
    class is the part whose answers only a real accelerator has, and it **fails closed**: a backend
    that reports no memory statistics (CPU does not) cannot produce a peak, and a probe that cannot
    read a peak must refuse rather than invent one.
    """

    def devices(self) -> Sequence:
        import jax

        return jax.devices()

    def memory_stats(self, device) -> Mapping | None:
        try:
            return device.memory_stats()
        except AttributeError:
            return None

    def peak_and_capacity(self) -> tuple[int, int]:
        """``(worst per-device peak, smallest per-device capacity)`` across this process's devices."""
        stats = [(device, self.memory_stats(device)) for device in self.devices()]
        blind = [str(device) for device, entry in stats if not entry]
        if blind or not stats:
            raise ValueError(
                f"this backend reports no memory statistics for {blind or 'any device'}: M1's whole output is "
                f"a per-device peak against a per-device capacity, and a probe that cannot read them has "
                f"nothing to authorize. Run M1 on the accelerator the training job will use."
            )
        missing = sorted(
            {key for _, entry in stats for key in ("peak_bytes_in_use", "bytes_limit") if key not in entry}
        )
        if missing:
            raise ValueError(f"the runtime's memory statistics do not report {missing}; the peak cannot be read")
        return (
            max(int(entry["peak_bytes_in_use"]) for _, entry in stats),
            min(int(entry["bytes_limit"]) for _, entry in stats),
        )

    def reset_peak(self) -> bool:
        """Ask the backend to clear its high-water mark. ``True`` when a facility exists and worked."""
        cleared = False
        for device in self.devices():
            for name in ("clear_memory_stats", "reset_memory_stats"):
                try:
                    getattr(device, name)()
                except (AttributeError, NotImplementedError, TypeError):
                    continue
                except Exception:  # noqa: BLE001 -- a backend that refuses is a backend without one
                    continue
                cleared = True
                break
        return cleared

    def begin_steady_state(self) -> dict:
        """Open the region whose peak may be attributed to this cell."""
        reset = self.reset_peak()
        peak, capacity = self.peak_and_capacity()
        return {"reset": reset, "peak_before": 0 if reset else int(peak), "capacity": int(capacity)}

    def end_steady_state(self, before: Mapping, *, program_bytes: int | None = None) -> tuple[int, int, str]:
        """``(peak, capacity, source)`` for THIS cell, or a refusal.

        Two attributable sources, and the larger wins because a peak is a ceiling:

        * the runtime's high-water mark **when this region raised it** (after a reset, always; without
          one, only when it exceeded everything that came before — otherwise the mark belongs to some
          earlier cell or to the pipeline load and says nothing about this program);
        * the compiled executable's own memory analysis, which is per-program by construction.

        With neither, there is no cell-local number and the measurement fails closed. The alternative
        — reporting the lifetime mark — is what let a 64-wide cell's peak authorize an 8-wide one.
        """
        peak, capacity = self.peak_and_capacity()
        sources = {}
        if bool(before.get("reset")) or int(peak) > int(before.get("peak_before", 0)):
            sources[
                "runtime high-water mark" + (" after reset" if before.get("reset") else " raised by this region")
            ] = int(peak)
        if program_bytes:
            sources[PEAK_SOURCE_ANALYSIS] = int(program_bytes)
        if not sources:
            raise ValueError(
                f"no per-cell steady-state peak could be obtained: this backend offers no way to reset the "
                f"high-water mark, this region did not raise it above the {int(before.get('peak_before', 0))} "
                f"bytes already standing, and the compiled program reports no memory analysis. The lifetime "
                f"mark belongs to an earlier cell or to the model load, so reporting it would authorize this "
                f"cell on somebody else's measurement."
            )
        name = max(sources, key=lambda key: sources[key])
        return sources[name], int(capacity), name


#: Structured first, then bounded exact phrases. The reviewer classified BOTH ``"boom in program
#: build"`` and ``"No room left on device"`` as HBM exhaustion under a bare ``"OOM"`` substring — so a
#: model bug or a FULL DISK became an apparently measured allocation refusal and M1 exited
#: successfully having authorized nothing it understood.
_EXHAUSTED_PHRASES = re.compile(
    r"\bRESOURCE_EXHAUSTED\b|\bOUT OF MEMORY\b|\bOOM\b|\bXLA_?RUNTIME_?ERROR: *RESOURCE", re.I
)


def _is_resource_exhausted(error: BaseException) -> bool:
    """A refused ALLOCATION, and nothing else.

    ``"boom in program build"`` contains ``oom``; ``"No room left on device"`` contains ``room``. Both
    were classified as HBM exhaustion and would have been published as measured refusals. Word
    boundaries and a structured status check replace the substring.
    """
    status = None
    for attribute in ("status_code", "code", "_code"):
        try:
            status = str(getattr(error, attribute))
        except AttributeError:
            continue
        break
    if status and "RESOURCE_EXHAUSTED" in status.upper():
        return True
    if not isinstance(error, (MemoryError, RuntimeError)):
        # A disk error (OSError), a KeyError, a shape error: none of them is an allocation refusal,
        # whatever their message happens to spell. RuntimeError covers jax.errors.JaxRuntimeError.
        return False
    return bool(_EXHAUSTED_PHRASES.search(f"{type(error).__name__}: {error}"))


def build_probe_program(config, cell: FitCell, *, model_source=None):
    """Build EXACTLY the program one training step at this cell runs, over a synthetic batch.

    **W3 (final review, BLOCKER A2): this now shares the whole finalization boundary with the live
    trainer, not merely the three factories.** The previous version built its own adapter, optimizer
    and jitted update through the shared factories and then finished the program differently: it
    returned an UNSHARDED parameter tree and compiled outside the deployed ``logical_axis_rules``,
    while the trainer replicated its parameters and compiled inside them. Measured on an 8-device
    mesh, the trainer's leaves carried ``NamedSharding(mesh, P())`` and M1's carried
    ``SingleDeviceSharding(CpuDevice(0))``. Since the Wan blocks translate logical axes during the
    forward, that is a different compiled program -- and the per-device HBM peak M1 authorizes is
    exactly the quantity it changes. It also closed over a ZERO null context and built its own grid,
    where training uses the loader's real T5 embedding and the scheduler.

    So everything that decides what is compiled now comes from
    :func:`~maxdiffusion.pos_rollout_update.build_training_program`, and this function adds only what
    is genuinely probe-specific: a synthetic batch (a footprint does not depend on WHICH examples the
    loader would have handed it) and the batch-one slice the DEV instrument's unit is timed at.

    ``model_source`` is the WEIGHTS seam: production loads the pretrained 5B pipeline, which needs
    both the snapshot and an accelerator. A host test supplies a source returning the same
    ``LoadedBackbone`` record from the same config keys at test dimensions.
    """
    import jax.numpy as jnp

    from maxdiffusion.pos_rollout_stream import draw_step_for_batch
    from maxdiffusion.pos_rollout_update import build_training_program, draws_to_arrays

    source = model_source or ProductionModelSource()
    backbone = source.load(config)
    num_steps = int(declared(config, "side_adapter_sampling_steps"))
    program = build_training_program(config, backbone, arm=str(cell.arm), k_b=int(cell.k_b), num_steps=num_steps)

    microbatch = int(cell.microbatch)
    latents = (
        int(declared(config, "latent_channels")),
        int(declared(config, "latent_frames")),
        int(declared(config, "latent_height")),
        int(declared(config, "latent_width")),
    )
    logical_batch = int(declared(config, "pos_logical_batch"))
    # Built at the LOGICAL width and split by the production stream seam, exactly as a training step
    # does: the accumulation plan is part of what is being measured, and tiling a microbatch up would
    # have let the probe's own arithmetic decide the split instead of the seam's.
    batch = {
        "z_video": jnp.zeros((logical_batch, *latents), jnp.float32),
        "z_i0": jnp.zeros((logical_batch, latents[0], 1, latents[2], latents[3]), jnp.float32),
        "actions": jnp.zeros(
            (logical_batch, int(declared(config, "action_len")), int(declared(config, "action_dim"))), jnp.float32
        ),
    }
    _, micro_draws, micro_batches = draw_step_for_batch(
        batch,
        seed=int(declared(config, "seed")),
        global_step=1,
        logical_batch=logical_batch,
        microbatch=microbatch,
        num_steps=num_steps,
        k_b=int(cell.k_b),
    )

    # THE SHARED DEV SCORER, timed as-is. The private one here jitted `loss_fn(...)[0]`, and that
    # subscript let XLA prune the aux norms and statistics the instrument actually computes —
    # understating the evaluation term of the wall projection (final re-ruling, MAJOR 3).
    one = {key: value[:1] for key, value in micro_batches[0].items()}
    one_draws = tuple(_first_example(array, microbatch) for array in draws_to_arrays(micro_draws[0]))
    return ProbeProgram(
        step=program.step,
        score=program.score,
        params=program.params,
        opt_state=program.opt_state,
        batch=tuple(micro_batches),
        draws=tuple(draws_to_arrays(draws) for draws in micro_draws),
        eval_batch=one,
        eval_draws=one_draws,
        scope=program.scope,
        context=program.context,
        frozen=program.frozen,
    )


def _draws_from(values):
    from maxdiffusion.pos_rollout_update import draws_from_arrays

    return draws_from_arrays(values)


def _first_example(array, microbatch: int):
    """The batch-one slice of a per-example draw; a scalar or per-batch value is passed through."""
    shape = getattr(array, "shape")
    return array[:1] if shape[:1] == (int(microbatch),) else array


@dataclasses.dataclass(frozen=True)
class ProbeProgram:
    """The compiled thing M1 measures: one optimizer step, one scoring pass, and their inputs."""

    step: Any
    score: Any
    params: Any
    opt_state: Any
    #: EVERY microbatch of the logical batch: the timed unit is one accumulated optimizer update.
    batch: Any
    draws: Any
    #: The DEV instrument's own unit — one example — timed separately (review F1b, BLOCKER 2).
    eval_batch: Any = None
    eval_draws: Any = None
    #: THE deployed execution scope (mesh + logical axis rules), from the shared program (W3). The
    #: measurement used to run under `with mesh` alone, which can compile a different sharding.
    scope: Any = contextlib.nullcontext
    #: The ``ArmContext`` the step was compiled against — the shared one, exposed so an oracle can
    #: compare it with the trainer's rather than dig it out of a closure (W3).
    context: Any = None
    #: The frozen backbone this program's step is compiled against, from the shared program (F3).
    #: Exposed for exactly the reason ``context`` is: since F3 the 5B is an ARGUMENT of the update
    #: rather than a constant inside it, so it now decides what is compiled — and an oracle comparing
    #: M1's program with the trainer's must be able to read it on BOTH sides rather than borrow one
    #: side's and assume the other matches.
    frozen: Any = None


class ProductionModelSource:
    """The WEIGHTS seam — and, since W3, nothing more than that.

    ``load`` returns the shared :class:`~maxdiffusion.pos_rollout_update.LoadedBackbone` by calling
    the shared :func:`~maxdiffusion.pos_rollout_update.load_backbone`, which is the same function the
    live trainer calls. It used to return ``(transformer, adapters, num_train_timesteps)`` and leave
    M1 to invent a null context and a grid; the reviewer's finding was that inventing them made M1
    measure a program deployment never runs.
    """

    def mesh(self, config):  # pragma: no cover -- needs an accelerator
        """Deprecated by W3: the mesh is the PIPELINE's, carried on the backbone. Kept only because
        the measurement path still asks a source for one when it has no backbone yet."""
        import jax

        from maxdiffusion import max_utils

        return jax.sharding.Mesh(max_utils.create_device_mesh(config), declared(config, "mesh_axes"))

    def load(self, config):  # pragma: no cover -- needs the snapshot and an accelerator
        from maxdiffusion.pos_rollout_update import load_backbone

        return load_backbone(config)


def measure_cell_on_device(
    *,
    cell: FitCell,
    context: ProbeContext,
    config,
    telemetry: DeviceTelemetry | None = None,
    model_source=None,
) -> CellMeasurement:
    """Compile one cell, run it to a steady state, and read the runtime's telemetry.

    The reviewer's blocker was blunt: this function unconditionally raised while being the default
    measurer of the M1 entrypoint, so **the job the launcher exists to start was guaranteed to fail
    before measuring its first cell**. Hardware-specific code may be untestable on the host; it
    cannot be absent at the last gate before launching that exact job.

    What it does now, in order: build the arm's real train step at this cell; compile it and time the
    compile; run :data:`WARMUP_STEPS` and discard them; time :data:`TIMED_STEPS` and take the mean as
    the steady-state step time; run one DEV-cohort evaluation pass and scale it to the cohort; write
    one real checkpoint and time it; read the per-device peak and capacity; and count every refused
    allocation seen along the way. A ``RESOURCE_EXHAUSTED`` at any point is counted as a reservation
    failure and the cell is reported at capacity — measured and refused, which is a result, not a
    crash.
    """
    import time

    telemetry = telemetry or DeviceTelemetry()
    source = model_source or ProductionModelSource()
    started = time.perf_counter()
    try:
        # The scope has to be in context for the FORWARD, not only for construction: the adapter's
        # first-block feature extraction issues a sharding constraint against a LOGICAL axis name,
        # and the RULES are what translate it. Production runs its step under mesh + axis rules; W3
        # made the probe enter the very same scope object, built by the shared program.
        return _measure_under_mesh(
            cell=cell, context=context, config=config, telemetry=telemetry, source=source, started=started
        )
    except Exception as error:  # noqa: BLE001 -- a refused allocation is a RESULT about this cell
        if not _is_resource_exhausted(error):
            raise
        # The cell missed. Everything recorded here is measured: the elapsed time really elapsed, and
        # the peak is reported AT capacity because the runtime refused an allocation at that ceiling.
        # No number is invented, and the reservation failure refuses the cell on its own.
        capacity = _capacity_after_refusal(telemetry)
        elapsed = max(time.perf_counter() - started, 1e-9)
        return CellMeasurement(
            cell=cell,
            context_digest=context.digest(),
            compile_seconds=elapsed,
            step_seconds=elapsed,
            eval_seconds=0.0,
            checkpoint_seconds=0.0,
            peak_bytes=capacity,
            capacity_bytes=capacity,
            reservation_failures=1,
            peak_source=PEAK_SOURCE_REFUSED,
        )


def _measure_under_mesh(*, cell, context, config, telemetry, source, started) -> CellMeasurement:
    """The timed body of :func:`measure_cell_on_device`, inside the SCOPE production runs under."""
    import time

    import jax

    # ANNOUNCE THE CELL BEFORE COMPILING IT, not after measuring it (F3).
    # Every other `[M1]` line in this file reports a FINISHED cell, so a probe that hangs inside its
    # first XLA compile prints nothing at all. That is precisely what happened: three M1 attempts on
    # v6e-8 each ran ~2h and died on `TPU_VM_HEALTH_TIMEOUT` with no `[M1]` line ever emitted, and
    # the logs could not say whether the job was compiling, loading, or wedged. Unbuffered stdout was
    # not the missing piece -- `PYTHONUNBUFFERED=1` was already exported by the launcher -- there was
    # simply no statement to flush. One line before the long-running call is what makes the next
    # failure diagnosable from the log alone.
    print(f"[M1] entering {cell.arm} microbatch={cell.microbatch} k={cell.k_b}: building and compiling", flush=True)
    program = build_probe_program(config, cell, model_source=source)
    # THE DEPLOYED SCOPE — mesh AND logical axis rules — around everything that is measured (W3).
    # Compilation, the timed steps, the scoring pass and the checkpoint all happen inside it, because
    # a bare mesh lets the forward's logical-axis constraints translate differently and the peak this
    # function returns is then the peak of another program.
    with program.scope():
        params, opt_state = program.params, program.opt_state
        params, opt_state, loss = program.step(params, opt_state, program.batch, program.draws)
        jax.block_until_ready(loss)
        compile_seconds = time.perf_counter() - started

        for _ in range(WARMUP_STEPS):
            params, opt_state, loss = program.step(params, opt_state, program.batch, program.draws)
        jax.block_until_ready(loss)

        # --- the steady-state region, isolated (review F1b, BLOCKER 3) -------------------------------
        # The runtime's `peak_bytes_in_use` is a LIFETIME high-water mark: nothing separated model load,
        # compilation, warm-up and the previous 31 cells from it, so a later cell inherited an earlier
        # one's peak and a load transient read as steady state. Two supported facilities are used and the
        # LARGER is reported: a reset where the backend offers one plus the high-water mark attributable
        # to this region, and the compiled executable's OWN memory analysis, which is cell-local by
        # construction. If neither yields a number attributable to this cell, the measurement fails
        # closed rather than reporting somebody else's peak.
        before = telemetry.begin_steady_state()

        timed_started = time.perf_counter()
        for _ in range(TIMED_STEPS):
            params, opt_state, loss = program.step(params, opt_state, program.batch, program.draws)
        jax.block_until_ready(loss)
        step_seconds = (time.perf_counter() - timed_started) / TIMED_STEPS

        peak, capacity, peak_source = telemetry.end_steady_state(
            before, program_bytes=_program_bytes(program, params, opt_state)
        )

        # The DEV instrument scores ONE example at a time, so the evaluation unit is a batch-one pass.
        jax.block_until_ready(program.score(params, program.eval_batch, program.eval_draws))
        eval_started = time.perf_counter()
        jax.block_until_ready(program.score(params, program.eval_batch, program.eval_draws))
        eval_seconds = (time.perf_counter() - eval_started) * DEV_COHORT_SIZE

        checkpoint_seconds = _time_one_checkpoint(config, cell, params, opt_state)

        print(
            f"[M1] {cell.arm} microbatch={cell.microbatch} k={cell.k_b}: peak {peak} bytes "
            f"({peak_source}), step {step_seconds:.3f}s over {len(program.batch)} microbatches"
        )
        return CellMeasurement(
            cell=cell,
            context_digest=context.digest(),
            compile_seconds=compile_seconds,
            step_seconds=step_seconds,
            eval_seconds=eval_seconds,
            checkpoint_seconds=checkpoint_seconds,
            peak_bytes=peak,
            capacity_bytes=capacity,
            reservation_failures=0,
            peak_source=peak_source,
        )


def _program_bytes(program: "ProbeProgram", params, opt_state) -> int | None:
    """The compiled update's OWN footprint — arguments + temporaries + output, per executable.

    ``Compiled.memory_analysis()`` is a supported XLA facility and it is cell-local by construction:
    it describes this program, not this process's history. The caller still takes the maximum of it
    and the attributable runtime high-water mark, because a static account is not a run.

    **F3 made this number strictly better, and the caveat that used to stand here is now obsolete.**
    It read "weights closed over as constants are not arguments" — true when written, and a symptom
    of the defect that killed M1: the frozen 5B was a captured CONSTANT, so ``argument_size_in_bytes``
    could not see the ten gigabytes the program really carried, and no oracle in this file could have
    reported them. The backbone is now an explicit argument of the compiled update, so those bytes
    are counted where they belong. Nothing is double-counted (the caller MAXes rather than sums), and
    an authorization measured before F3 is not comparable with one measured after — already enforced,
    because every authorization carries the ``code_sha`` it was measured on and the launcher refuses
    a training job whose SHA differs.
    """
    try:
        compiled = program.step.lower(params, opt_state, program.batch, program.draws).compile()
        analysis = compiled.memory_analysis()
    except Exception:  # noqa: BLE001 -- an unavailable analysis is a missing source, not a failure
        return None
    if analysis is None:
        return None
    total = 0
    for field in ("argument_size_in_bytes", "temp_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes"):
        try:
            total += int(getattr(analysis, field) or 0)
        except AttributeError:
            continue
    return total or None


def _capacity_after_refusal(telemetry: DeviceTelemetry) -> int:
    """After a refused allocation the peak is not meaningful; the CAPACITY still decides the verdict."""
    try:
        return telemetry.peak_and_capacity()[1]
    except ValueError:
        raise ValueError(
            "this cell exhausted the device AND the backend reports no memory statistics, so the miss "
            "cannot be recorded with a number. Run M1 on the accelerator the training job will use."
        ) from None


def _time_one_checkpoint(config, cell: FitCell, params, opt_state) -> float:
    """One PRODUCTION checkpoint write — the loop's own payload, to the configured storage class.

    Review F1b, BLOCKER 2: this used to write to a local ``TemporaryDirectory``. On a worker the
    training job writes ``params + opt_state + step`` through ``pos_rollout_loop.save_checkpoint``
    into a ``gs://`` tree, and the cost of that is dominated by the storage class, not by
    serialization — so a local tmpdir measured the wrong thing by roughly the network.

    The write goes to a probe-scoped path under the CONFIGURED checkpoint parent, so the storage
    class is production's, and it is removed afterwards. When no checkpoint destination is configured
    the measurement fails closed rather than substituting a local directory.
    """
    import time

    from maxdiffusion.pos_rollout_loop import RolloutTrainState, build_checkpoint_manager, save_checkpoint
    from maxdiffusion.pos_rollout_support import is_remote

    parent = str(declared(config, "checkpoint_dir") or "")
    if not parent:
        raise ValueError(
            "checkpoint_dir is empty, so the checkpoint overhead cannot be measured where production "
            "writes it. A local temporary directory measures serialization and not the storage class, "
            "and the projection would then understate every checkpoint of the run."
        )
    destination = f"{parent.rstrip('/')}/_m1_probe/{cell.arm}_m{cell.microbatch}_k{cell.k_b}"
    manager = build_checkpoint_manager(destination, max_to_keep=1)
    state = RolloutTrainState(params=params, opt_state=opt_state, step=0)
    started = time.perf_counter()
    save_checkpoint(manager, state, dev_metric=0.0, history=(), arm=str(cell.arm), k_b=int(cell.k_b))
    manager.wait_until_finished()
    elapsed = time.perf_counter() - started
    _remove_probe_checkpoint(destination, remote=is_remote(destination))
    return elapsed


def _remove_probe_checkpoint(destination: str, *, remote: bool) -> None:
    """Best effort: the probe's own scratch write must not be left behind in a run's tree."""
    try:
        from tensorflow.io import gfile

        if gfile.exists(destination):
            gfile.rmtree(destination)
    except Exception:  # noqa: BLE001 -- a probe that cannot tidy up must still report its measurement
        if not remote:
            import shutil

            shutil.rmtree(destination, ignore_errors=True)


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

    Cells whose microbatch does not divide ``pos_logical_batch`` are DROPPED rather than measured:
    accumulation preserves the logical batch by construction, so such a cell is not a program that
    can run, and a probe that tried would die on the accumulation seam partway through the ladder.
    They are named in the log and remain "never measured" at the gate, which is exactly true.
    """
    context = derive_probe_context(config, devices=devices)
    requested = tuple(cells) if cells is not None else ladder()
    logical_batch = int(declared(config, "pos_logical_batch"))

    # F6: DECLARED-unreachable cells are removed BEFORE anything is built. Not measured, not compiled,
    # not adopted -- the whole point is that compiling one of them killed the VM twice on the same
    # cell. They are recorded in the table instead, with the declared reason.
    declared_out = parse_excluded_cells(config)
    why_excluded = declared_exclusion_reason(config)
    exclusions = tuple((cell, why_excluded) for cell in declared_out if cell in requested)
    for cell, _ in exclusions:
        print(f"[M1] EXCLUDED {cell.arm} microbatch={cell.microbatch} k={cell.k_b}: {why_excluded}", flush=True)
    survivors = tuple(cell for cell in requested if cell not in declared_out)

    runnable = tuple(cell for cell in survivors if logical_batch % int(cell.microbatch) == 0)
    skipped = tuple((cell, SKIPPED_NON_DIVIDING) for cell in survivors if cell not in runnable)
    for cell, _ in skipped:
        print(
            f"[M1] skipping {cell.arm} microbatch={cell.microbatch} k={cell.k_b}: "
            f"{cell.microbatch} does not divide the logical batch {logical_batch}"
        )
    requested = runnable
    if not requested:
        # Name the actual cause rather than a merged one: "nothing divides the logical batch" and
        # "every cell was declared unreachable" are different operator mistakes with different fixes.
        if exclusions and not skipped:
            raise ValueError(
                "every cell of this ladder is declared in pos_fit_excluded_cells, so this probe would "
                "publish a table that authorizes nothing. The exclusion list removes cells from a ladder; "
                "it cannot be the whole ladder."
            )
        if skipped and not exclusions:
            raise ValueError(
                f"no declared cell has a microbatch dividing pos_logical_batch={logical_batch}, so this probe "
                f"would authorize nothing; the ladder is {LADDER_MICROBATCH}"
            )
        raise ValueError(
            f"every cell of this ladder was excluded or skipped ({len(exclusions)} declared unreachable, "
            f"{len(skipped)} with a microbatch not dividing pos_logical_batch={logical_batch}), so this probe "
            f"would publish a table that authorizes nothing; the ladder is {LADDER_MICROBATCH}"
        )
    if int(trials) < 1:
        raise ValueError(f"each cell needs at least one trial, got {trials!r}")
    # The destination is checked BEFORE the first cell is measured (F5). It used to be read after the
    # ladder, so a probe pointed nowhere spent the whole 3.5 hours to discover it -- and F5's per-cell
    # publication needs the path from the first cell anyway.
    path = str(declared(config, "pos_fit_authorization"))
    if not path:
        raise ValueError(
            "pos_fit_authorization must name the path M1 publishes to; a probe that measures a ladder and "
            "publishes nowhere leaves M2/M3 with no route to a run"
        )
    adoption_root = str(declared(config, "pos_fit_adoption_root") or "")
    job_identity = derive_job_identity(config)
    print(
        f"[M1] banking cells in {cell_publication_dir(path)}; adoption root "
        f"{adoption_root or '(none -- every cell will be measured)'}; run identity {job_identity!r}",
        flush=True,
    )
    measurements: list[CellMeasurement] = []
    provenance: dict[FitCell, str] = {}
    for cell in requested:
        adopted = adopt_published_cell(
            cell, context=context, job_identity=job_identity, root=adoption_root, trials=int(trials)
        )
        if adopted is not None:
            banked, source = adopted
            measurements.extend(banked)
            # NOT republished into this attempt's own cells/ tree: a second copy of one measurement
            # would read like a second measurement to anything that counts artifacts.
            provenance[cell] = f"{ADOPTED_PREFIX}{source}"
            continue
        taken: list[CellMeasurement] = []
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
            taken.append(measurement)
        # BANK IT NOW. The next line of this loop is another cell's compile, and the zone has killed
        # this job seven times inside one; a cell that is only in memory when that happens is a cell
        # the campaign pays for again.
        publish_cell(
            cell_marker_path(path, cell),
            CellArtifact(cell=cell, context=context, job_identity=job_identity, trials=tuple(taken)),
        )
        measurements.extend(taken)
        provenance[cell] = PROVENANCE_MEASURED
    evidence = build_evidence(
        context,
        measurements,
        max_train_steps=int(declared(config, "max_train_steps")),
        eval_every=int(declared(config, "eval_every")),
        checkpoint_every=int(declared(config, "checkpoint_every")),
        provenance=provenance,
        exclusions=exclusions,
        skipped=skipped,
    )
    published = publish_authorization(path, evidence)
    print(
        f"[M1] context {context.digest()[:16]} on {context.device_kind}x{context.device_count} @ {context.code_sha[:12]}"
    )
    banked = {(row["arm"], row["microbatch"], row["k_b"]): row["provenance"] for row in published["cell_provenance"]}
    for entry in published["measured_cells"]:
        state = "AUTHORIZED" if entry in published["authorized_cells"] else "refused"
        source = banked.get((entry["arm"], entry["microbatch"], entry["k_b"]), PROVENANCE_MEASURED)
        print(f"[M1] {entry['arm']:<9s} microbatch={entry['microbatch']:<3d} k={entry['k_b']}  {state}  [{source}]")
    for entry in published["excluded_cells"]:
        print(f"[M1] {entry['arm']:<9s} microbatch={entry['microbatch']:<3d} k={entry['k_b']}  EXCLUDED  [declared]")
    adopted = sum(1 for text in banked.values() if text.startswith(ADOPTED_PREFIX))
    print(
        f"[M1] {len(banked) - adopted} cell(s) measured this attempt, {adopted} adopted from prior attempts, "
        f"{len(published['excluded_cells'])} declared EXCLUDED, {len(published['skipped_cells'])} skipped"
    )
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

    # Delegate to the PACKAGE module rather than running this file's own copy. Executed as a script,
    # `__main__` and `maxdiffusion.pos_rollout_fit_probe` are two different module objects holding
    # two different `ProductionModelSource` classes and two different `DeviceTelemetry` classes — so
    # anything that configures the package (a test's controlled backend, a future production
    # override) would silently not apply to the code the launcher actually runs.
    from maxdiffusion.pos_rollout_fit_probe import main as _main

    _main(sys.argv)
