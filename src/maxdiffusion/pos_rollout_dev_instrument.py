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

**Capabilities, not labels.** A ``DevCohort`` cannot be constructed — only ISSUED by
:func:`load_dev_cohort`, which requires the approved manifest's digest and validates every row
against the bytes it hashed. The previous design checked that a caller had passed the string
``"dev64"``, which is a claim about a label rather than about content: the reviewer duly wrapped a
DEV label around a genuine TEST name and obtained a draw (BLOCKER 1). Guarding a claim is not the
same as making the wrong thing unconstructible, and this module now does the latter.

**TEST-64 is structurally unreachable, not merely unused.** The S7-era hazard was a config pointing
at the whole validation directory, so "we don't pass TEST" is exactly the assurance that failed. Here
there is no API that accepts a bare example name: draws are produced from a :class:`DevCohort`, a
``DevCohort`` can only be built by :func:`load_dev_cohort`, and that refuses any manifest whose
cohort is not ``dev64``. A training-time caller holding the TEST manifest cannot construct the object
the instrument requires. The tests try it and are refused.

**Not this round:** the loop and its cadence/stop rule/checkpoints (T3b-4), YAML (T4), the evaluator
and its gates (T5a/b). This module computes a per-example number and the provenance that makes it
quotable; deciding anything with it is T3b-4's.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Sequence

import jax
import jax.numpy as jnp

from maxdiffusion.pos_rollout_stream import StepDraws
from maxdiffusion.pos_rollout_support import exp03_aux_key

__all__ = [
    "DEV_COHORT",
    "FORBIDDEN_COHORTS",
    "POS_ROLLOUT_DEV_PURPOSE",
    "DEFAULT_REPLICATES",
    "INSTRUMENT_SEED",
    "J0_DEV64_SHA256",
    "DevCohort",
    "instrument_provenance",
    "load_dev_cohort",
    "score_dev_cohort",
]

#: The ONLY cohort this instrument will score. Selection runs during training; TEST is confirmation
#: only (plan §3d), so the name is a constant here rather than a caller's argument.
DEV_COHORT = "dev64"
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

# Only `load_dev_cohort` holds this; `DevCohort.__init__` refuses without it, so the class is a
# capability the loader ISSUES rather than a struct any caller can fill in (review BLOCKER 1).
_ISSUE_TOKEN = object()


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


class DevCohort:
    """A loader-ISSUED capability: the only object the instrument will draw or score for.

    Not a dataclass and not publicly constructible — :func:`load_dev_cohort` holds the only issue
    token. The previous design was a public frozen dataclass validating that the caller had passed
    the label ``"dev64"``; the reviewer wrapped that label around a genuine TEST name and got a draw
    (review BLOCKER 1). A capability cannot be forged, so the attack has no entry point rather than a
    guarded one.

    It carries the manifest's ROWS, not just names: identity for scoring is the row (name, shard,
    ordinal) the approved manifest specifies, so what gets read is decided by the manifest and not by
    a caller's dictionary keys (review BLOCKER 2).
    """

    __slots__ = ("_rows", "_by_name", "cohort", "manifest_sha256", "manifest_path")

    def __init__(self, token, *, cohort, rows, manifest_sha256, manifest_path):
        if token is not _ISSUE_TOKEN:
            raise TypeError(
                "DevCohort is issued by load_dev_cohort, never constructed: a hand-built cohort is a "
                "label, and this instrument binds to the approved manifest's CONTENT"
            )
        self.cohort = cohort
        self.manifest_sha256 = manifest_sha256
        self.manifest_path = manifest_path
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


def load_dev_cohort(path: str, *, expected_sha256: str = J0_DEV64_SHA256, expected_size: int = 64) -> DevCohort:
    """Issue a cohort from exp_04's published J0 DEV-64 manifest, fail-closed.

    The digest is REQUIRED (it defaults to the approved manifest's, so binding is the easy path and
    opting out is impossible): any schema-valid file labelled ``dev64`` was previously accepted.
    The bytes are read ONCE and both hashed and parsed from that same buffer — validating one read
    and hashing another permits payload/digest disagreement (review BLOCKER 2).
    """
    from maxdiffusion.null_adapter_manifest_io import MANIFEST_SCHEMA_VERSION

    try:
        from tensorflow.io import gfile

        raw = gfile.GFile(path, "rb").read()
    except Exception:  # noqa: BLE001 - a local path is the common case and must not need tensorflow
        raw = pathlib.Path(path).read_bytes()

    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"manifest hash mismatch for {path}: expected {expected_sha256}, found {digest}. A score is only "
            f"quotable against the cohort it was measured on."
        )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "cohort", "rows"}:
        raise ValueError(f"{path}: manifest fields do not match the schema")
    if int(payload["schema_version"]) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported manifest schema_version {payload['schema_version']!r}")

    cohort = str(payload["cohort"])
    if cohort in FORBIDDEN_COHORTS:
        raise ValueError(
            f"refusing to load cohort {cohort!r} for training-time selection: TEST is confirmation only "
            f"(plan \u00a73d), and the DEV instrument scores {DEV_COHORT!r}"
        )
    if cohort != DEV_COHORT:
        raise ValueError(f"the DEV instrument scores {DEV_COHORT!r} only, refusing cohort {cohort!r}")

    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != int(expected_size):
        raise ValueError(f"the {cohort} cohort must carry exactly {int(expected_size)} examples, got {len(rows)}")
    names = [str(row["name"]) for row in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: the cohort carries duplicate example names")
    for row in rows:
        if str(row["split"]) != cohort:
            raise ValueError(f"{path}: row {row['name']!r} declares split {row['split']!r}, not {cohort!r}")
    return DevCohort(_ISSUE_TOKEN, cohort=cohort, rows=rows, manifest_sha256=digest, manifest_path=str(path))


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
    batch_loader,
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

    ``batch_loader(row) -> batch`` is driven BY THE COHORT: the instrument hands it the approved
    manifest's row and takes what comes back. It does NOT accept a caller-supplied mapping keyed by
    name — that design let TEST tensors be filed under DEV keys and stamped with DEV provenance
    (review BLOCKER 2). Verifying caller tensors is checkable but forgeable; sourcing them from the
    validated manifest is not.
    """
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
        batch = batch_loader(dict(row))
        value, _ = loss_fn(params, batch, context, draws=draws)
        per_example[name] = float(value)
    ordered = [per_example[str(row["name"])] for row in cohort.rows]
    return {
        "metric": float(sum(ordered) / len(ordered)),
        "per_example": per_example,
        **instrument_provenance(cohort, k_b=context.k_b, eval_index=eval_index, arm=arm),
    }


def forbidden_cohort_names(paths: Sequence[str]) -> tuple[str, ...]:
    """Helper for callers that hold several manifests: which of these may selection NOT read."""
    from maxdiffusion.null_adapter_manifest_io import load_manifest

    return tuple(path for path in paths if str(load_manifest(path)["cohort"]) != DEV_COHORT)
