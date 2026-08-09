"""exp_06 `rollout_adapter` — T3b-3: the DEV-64 selection instrument (plan §3d).

**One contract: the selection estimand is deterministic, manifest-bound, and structurally cannot see
TEST.** Selection decides which checkpoint the experiment ships, so an estimand that drifts between
processes, or that quietly scores a different cohort, does not merely add noise — it makes the
selected artifact unreproducible and its number unquotable.

**Fixed draws (the exp_02 D2 instrument pattern).** Every ``(example, replicate)`` pair has ONE
pinned ``(support, epsilon, t_idx)``, derived from a FIXED instrument seed, the example's NAME, and a
predeclared replicate id. **No training step, optimizer step or evaluation index touches the key.**
An earlier version folded the evaluation index in, so a checkpoint measured at step 3,000 met
different noise than one measured at step 4,000 — which defeats the whole purpose: the stop rule and
the best-checkpoint choice would have been comparing scores that differ by their DRAWS as well as
their parameters, i.e. carrying evaluation noise into a selection decision (T3b-3 review, BLOCKER 3).
Two checkpoints scored by this instrument now differ **only by their parameters**, and the same
checkpoint scores identically whenever it is measured.

**Arm-agnostic by construction.** The draws know nothing about arms: R-B and matched-C0 are scored by
the same instrument on the same draws, and only the loss differs. If the instrument could vary with
the arm, "R-B selected a better checkpoint" would be unfalsifiable.

**A dedicated purpose.** The instrument derives from its own purpose, not from the training stream's
(``rollout_epsilon``/``index_support_rollout``/``one_step_index``). Sharing one would couple
selection to training randomness — the same checkpoint would score differently depending on which
step it was evaluated at.

**Construction IS verification.** A ``DevCohort`` is built from a PATH and nothing else: the
constructor reads the file, hashes it, and refuses anything whose digest is not
:data:`J0_DEV64_SHA256`. Two earlier designs failed here, both to attacks the reviewer EXECUTED
rather than hypothesized. The first validated that a caller had passed the string ``"dev64"`` -- a
claim about a LABEL -- so a DEV label wrapped around a genuine TEST name produced a draw (BLOCKER 1).
The second made the class a loader-issued capability but kept the issue secret in a module attribute
(``instrument._ISSUE_TOKEN`` handed it straight back) and let the caller supply the expected digest,
so a 64-row DEV-labelled manifest whose first row was the real TEST row loaded under its own hash
(BLOCKER A-B1). Both existed because verification and construction were separable. They are not
separable now, and there is no secret left to leak because none is needed.

**TEST-64 is structurally unreachable, not merely unused.** The S7-era hazard was a config pointing
at the whole validation directory, so "we don't pass TEST" is exactly the assurance that failed. Here
there is no API that accepts a bare example name: draws are produced from a :class:`DevCohort`, and a
``DevCohort`` requires the approved manifest's bytes. A training-time caller holding the TEST manifest
cannot construct the object the instrument requires -- nor the batches, since :class:`DevBatchReader`
opens the cohort's own rows and verifies what comes back (BLOCKER A-B2). Every route in is tried by
the tests and refused.

**Not this round:** the loop and its cadence/stop rule/checkpoints (T3b-4), YAML (T4), the evaluator
and its gates (T5a/b). This module computes a per-example number and the provenance that makes it
quotable; deciding anything with it is T3b-4's.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from maxdiffusion.pos_rollout_stream import StepDraws
from maxdiffusion.pos_rollout_support import exp03_aux_key, require_finite, storage_read_bytes

__all__ = [
    "DEV_COHORT",
    "DEV_COHORT_SIZE",
    "FORBIDDEN_COHORTS",
    "POS_ROLLOUT_DEV_PURPOSE",
    "DEFAULT_REPLICATES",
    "INSTRUMENT_SEED",
    "J0_DEV64_SHA256",
    "DevBatchReader",
    "DevCohort",
    "instrument_provenance",
    "load_dev_cohort",
    "score_dev_cohort",
]

#: The ONLY cohort this instrument will score. Selection runs during training; TEST is confirmation
#: only (plan §3d), so the name is a constant here rather than a caller's argument.
DEV_COHORT = "dev64"
#: Pinned like the digest. A caller-settable size is a knob that can only ever weaken a constant.
DEV_COHORT_SIZE = 64
#: Named so a refusal can say what it refused, and so the guard is greppable from a worker log.
FORBIDDEN_COHORTS = ("test64",)

#: Declared additively in ``pos_rollout_support.EXP03_AUX_PURPOSES``. Dedicated to selection so the
#: estimand never moves when the training stream does.
POS_ROLLOUT_DEV_PURPOSE = "dev_instrument"

#: The digest of exp_04's published J0 DEV-64 manifest. Selection binds to THIS file; a score
#: measured against anything else is not comparable and the loader refuses to produce one.
J0_DEV64_SHA256 = "3c59d023f3b782542ecae443b8d83008e7d8dfd801347f41adfab75218340836"

#: The instrument's own seed. Fixed for the life of the experiment and deliberately unrelated to the
#: training seed: selection draws must not move when a run's seed does.
INSTRUMENT_SEED = 20260804

#: Predeclared replicate ids. One draw per example is the estimand; more replicates would be a
#: declared change to the instrument, never an incidental consequence of when a checkpoint was
#: measured.
DEFAULT_REPLICATES = (0,)


def _name_id(text: str) -> int:
    """A stable 32-bit id for a string, hashed like T1's purpose ids (never positional)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _dev_draw_key(*, name: str, field: str, replicate: int) -> jax.Array:
    """The instrument's key for one ``(example, field, replicate)``. PRIVATE, and step-free.

    T1's derivation is the root — same offset, same fold discipline, its own purpose — with the
    fixed instrument seed and a predeclared REPLICATE id in the position T1 uses for a step. Nothing
    about a training run reaches it. It is private and takes no bare cohort: callers reach it only
    through :meth:`DevCohort.draw`, i.e. after membership has been validated (review BLOCKER 1).
    """
    root = exp03_aux_key(seed=INSTRUMENT_SEED, global_step=int(replicate), purpose=POS_ROLLOUT_DEV_PURPOSE)
    return jax.random.fold_in(jax.random.fold_in(root, _name_id(name)), _name_id(field))


def _read_bytes(path: str) -> bytes:
    """One reader for every artifact, remote or local (review pass 2, T5a BLOCKER 2)."""
    return storage_read_bytes(path)


class DevCohort:
    """The only object the instrument will draw or score for — and **constructing one IS loading the
    approved manifest.**

    Three designs failed before this one, each in the same way. A public dataclass validating that
    the caller passed the label ``"dev64"`` was a claim about a LABEL: the reviewer wrapped that
    label around a genuine TEST name and got a draw (BLOCKER 1). A loader-issued capability moved the
    check but kept a secret in a module attribute, and ``instrument._ISSUE_TOKEN`` handed it straight
    back; worse, the caller-settable digest let a 64-row DEV-labelled manifest whose first row was
    the real TEST row be loaded with its own computed hash (BLOCKER A-B1). Both attacks existed
    because verification and construction were separable.

    They are not separable here. There is no token, no secret and no digest argument: this
    constructor reads the file, hashes it, and refuses anything but :data:`J0_DEV64_SHA256`. The only
    input a caller controls is a path, so producing a cohort requires producing the approved
    manifest's bytes — and the forgery has nowhere to go. (No in-language guard survives
    ``object.__new__`` plus slot assignment; an attacker with that reach can rewrite the pinned
    digest just as easily, so that is a bound on Python, not a residual of this design.)

    It carries the manifest's ROWS, not just names: identity for scoring is the row (name, shard,
    ordinal) the approved manifest specifies, so what gets read is decided by the manifest and never
    by a caller (BLOCKER 2, BLOCKER A-B2).
    """

    __slots__ = ("_rows", "_by_name", "cohort", "manifest_sha256", "manifest_path")

    def __init__(self, path: str):
        from maxdiffusion.null_adapter_manifest_io import MANIFEST_SCHEMA_VERSION

        raw = _read_bytes(path)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != J0_DEV64_SHA256:
            raise ValueError(
                f"manifest hash mismatch for {path}: the DEV instrument binds to the published J0 "
                f"DEV-64 manifest {J0_DEV64_SHA256}, found {digest}. A score is only quotable against "
                f"the cohort it was measured on, so there is no argument that relaxes this."
            )
        # Parsed from the SAME buffer that was hashed: validating one read and hashing another
        # permits payload/digest disagreement (review BLOCKER 2).
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "cohort", "rows"}:
            raise ValueError(f"{path}: manifest fields do not match the schema")
        if int(payload["schema_version"]) != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported manifest schema_version {payload['schema_version']!r}")

        cohort = str(payload["cohort"])
        if cohort in FORBIDDEN_COHORTS:
            raise ValueError(
                f"refusing to load cohort {cohort!r} for training-time selection: TEST is confirmation only "
                f"(plan §3d), and the DEV instrument scores {DEV_COHORT!r}"
            )
        if cohort != DEV_COHORT:
            raise ValueError(f"the DEV instrument scores {DEV_COHORT!r} only, refusing cohort {cohort!r}")

        rows = payload["rows"]
        if not isinstance(rows, list) or len(rows) != DEV_COHORT_SIZE:
            raise ValueError(f"the {cohort} cohort must carry exactly {DEV_COHORT_SIZE} examples, got {len(rows)}")
        names = [str(row["name"]) for row in rows]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: the cohort carries duplicate example names")
        for row in rows:
            if str(row["split"]) != cohort:
                raise ValueError(f"{path}: row {row['name']!r} declares split {row['split']!r}, not {cohort!r}")

        self.cohort = cohort
        self.manifest_sha256 = digest
        self.manifest_path = str(path)
        self._rows = tuple(rows)
        self._by_name = {row["name"]: row for row in self._rows}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(row["name"] for row in self._rows)

    @property
    def rows(self) -> tuple[dict, ...]:
        return self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def row(self, name: str) -> dict:
        """The manifest row for ``name`` — the only way to learn WHAT to read for an example."""
        if name not in self._by_name:
            raise ValueError(f"{name!r} is not in the {self.cohort} cohort; the instrument scores its manifest only")
        return self._by_name[name]

    def draw(self, name: str, *, num_steps: int, k_b: int, example_shape, replicate: int = 0, dtype=jnp.float32):
        """THE pinned draw for one ``(example, replicate)`` — membership-checked, step-free.

        Keys are derived only here, only after :meth:`row` has validated membership against the
        approved manifest's content.
        """
        self.row(name)
        if int(replicate) not in DEFAULT_REPLICATES:
            raise ValueError(f"replicate {replicate!r} is not a predeclared replicate {list(DEFAULT_REPLICATES)}")
        start = jax.random.randint(
            _dev_draw_key(name=name, field="support", replicate=replicate), (), 0, int(num_steps) - int(k_b)
        )
        epsilon = jax.random.normal(
            _dev_draw_key(name=name, field="epsilon", replicate=replicate), (1, *tuple(example_shape)), dtype=dtype
        )
        t_idx = jax.random.randint(
            _dev_draw_key(name=name, field="index", replicate=replicate), (1,), 0, int(num_steps)
        )
        return StepDraws(support_start=start, support_end=start + int(k_b), epsilon=epsilon, t_idx=t_idx)


def load_dev_cohort(path: str) -> DevCohort:
    """Load exp_04's published J0 DEV-64 manifest, fail-closed. One argument, and no knobs."""
    return DevCohort(path)


class DevBatchReader:
    """**The instrument's own canonical row reader.** It opens the row's ``shard_path``/``ordinal``
    and verifies the decoded example's identity before anything is scored.

    **There is no injectable seam. That was the defect.** Two designs failed here, one layer apart.
    The first took ``batch_loader(row)`` from the caller and could not require it to use the row:
    ``lambda row: test_batch`` scored TEST content as DEV (BLOCKER A-B2). The second replaced it with
    this reader but kept ``reader``/``binder`` injectable "to exp_04's standard" — and the reviewer
    executed the same attack one layer lower: a ``reader`` echoing every genuine DEV name and its
    declared ordinal while returning tensors filled with 999, plus a ``binder`` echoing the
    manifest's generation and size, produced ``metric 999.0`` stamped ``cohort dev64`` with the
    genuine digest over all 64 examples. Equivalently it can hand back TEST tensors relabelled with
    the requested DEV identity, and every check believes it.

    **The lesson, recorded because it took three rounds:** matching a prior experiment's tolerance is
    not the same as closing the path. "exp_04's standard, no better" is not a justification when the
    hole is reachable.

    So this constructor takes a cohort and nothing else. Reading is exp_04's manifest-bound
    ``build_read_batch`` over the instrument's own TFRecord decoder and exp_04's own shard binding;
    it refuses a name outside the cohort, a record whose DECODED ordinal disagrees with its row, a
    shard whose generation/size has moved since J0 bound it, a duplicate, a name the shard did not
    yield, and any payload that is not production geometry and finite. A test that needs different
    bytes monkeypatches the decoder MODULE, which is a property of the test process rather than an
    argument any production caller can reach.
    """

    __slots__ = ("cohort", "_read_batch")

    def __init__(self, cohort: DevCohort):
        if not isinstance(cohort, DevCohort):
            raise TypeError(
                "a DevBatchReader reads a loaded DevCohort's own rows; a hand-built cohort is exactly "
                "what the manifest binding exists to refuse"
            )
        from maxdiffusion import run_wan_null_inversion

        self.cohort = cohort
        # Resolved from the module at construction, never from an argument: a caller cannot supply a
        # decoder, and a test that needs one patches `run_wan_null_inversion._tfrecord_reader`.
        self._read_batch = run_wan_null_inversion.build_read_batch(
            {str(row["name"]): row for row in cohort.rows}, reader=run_wan_null_inversion._tfrecord_reader
        )

    def read(self, name: str) -> dict:
        """The batch for ONE cohort example, decoded from the row the approved manifest bound."""
        self.cohort.row(str(name))  # membership, against the digest-pinned manifest's content
        batch, fields = self._read_batch((str(name),))
        if tuple(batch.names) != (str(name),):
            raise ValueError(f"the canonical reader returned {tuple(batch.names)} when asked for {name!r}")
        return {
            "z_i0": np.asarray(batch.z_i0, np.float32),
            "z_video": np.asarray(batch.z_video, np.float32),
            "actions": np.asarray(fields[str(name)]["actions"], np.float32)[None, ...],
        }


def instrument_provenance(
    cohort: DevCohort, *, k_b: int, eval_index: int | None = None, arm: str | None = None
) -> dict:
    """What every emitted score carries, so it can never be quoted against a different cohort.

    ``eval_index`` is recorded as WHEN a score was measured. It is deliberately absent from the draw
    derivation (review BLOCKER 3) — provenance may know the training step; the estimand may not.
    """
    return {
        "cohort": cohort.cohort,
        "manifest_path": cohort.manifest_path,
        "manifest_sha256": cohort.manifest_sha256,
        "example_count": len(cohort),
        "instrument_purpose": POS_ROLLOUT_DEV_PURPOSE,
        "instrument_seed": INSTRUMENT_SEED,
        "replicates": list(DEFAULT_REPLICATES),
        "k_b": int(k_b),
        "measured_at_step": None if eval_index is None else int(eval_index),
        "arm": arm,
    }


def score_dev_cohort(
    cohort: DevCohort,
    loss_fn,
    *,
    params,
    context,
    example_shape,
    eval_index: int | None = None,
    arm: str | None = None,
    dtype=jnp.float32,
    replicate: int = 0,
) -> dict:
    """Score every DEV example on its pinned draw; return the mean and its provenance.

    Batches come from the instrument's own :class:`DevBatchReader`, which opens the approved
    manifest's row and verifies the decoded example's identity. There is no caller callback and no
    caller mapping: both were forgeable (review BLOCKERs 2 and A-B2), and a forged batch scored under
    genuine DEV provenance is not a noisy measurement, it is a fabricated one.

    The reader is built here rather than accepted, and the read is dispatched through the CLASS
    rather than the instance, so neither a callback, a decoder, nor a subclass can substitute for
    reading the row the approved manifest names.
    """
    # The reader is CONSTRUCTED here, from this cohort. It is not a parameter, so there is nothing to
    # substitute: neither a callback (BLOCKER A-B2) nor a decoder under it (review pass 1, BLOCKER 1).
    batch_reader = DevBatchReader(cohort)
    per_example: dict[str, float] = {}
    for row in cohort.rows:
        name = str(row["name"])
        draws = cohort.draw(
            name,
            num_steps=context.num_steps,
            k_b=context.k_b,
            example_shape=example_shape,
            replicate=replicate,
            dtype=dtype,
        )
        batch = DevBatchReader.read(batch_reader, name)
        value, _ = loss_fn(params, batch, context, draws=draws)
        # A non-finite per-example score would travel into the mean, into the stop rule, and into an
        # unbeatable running best (review BLOCKER 2). It is refused where it is produced.
        per_example[name] = require_finite(value, f"the DEV score for {name}")
    ordered = [per_example[str(row["name"])] for row in cohort.rows]
    return {
        "metric": require_finite(sum(ordered) / len(ordered), "the aggregate DEV metric"),
        "per_example": per_example,
        **instrument_provenance(cohort, k_b=context.k_b, eval_index=eval_index, arm=arm),
    }


def forbidden_cohort_names(paths: Sequence[str]) -> tuple[str, ...]:
    """Helper for callers that hold several manifests: which of these may selection NOT read."""
    from maxdiffusion.null_adapter_manifest_io import load_manifest

    return tuple(path for path in paths if str(load_manifest(path)["cohort"]) != DEV_COHORT)
