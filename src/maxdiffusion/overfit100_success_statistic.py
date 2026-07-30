"""exp_02 overfit100 success statistic -- the plan's verdict rule as a PURE unit (D11/G4).

The plan predeclares exactly how the memorization claim is decided
(`docs/worklogs_yixun/exp_02_overfit100_claude/plan_overfit100.md` §3 D11). This module is
that rule, and nothing else: pure functions over the evaluator's aggregation rows, no IO in
the computation, no jax, no tf -- so the statistic is executable and unit-tested instead of
being computed by hand from a spreadsheet.

The rule, verbatim:

* ``C3_100`` = the S3 run's **segment-final** checkpoints (2500; +5000/7500/10000 if the run
  is extended). S2 checkpoints never enter the statistic -- they are simply not in the
  ``segment_final_checkpoints`` argument.
* ``m_corr(w, c) = median_{seed in {0,1,2}} SSIM(w, c, seed | context_mode="correct")``.
  The MEDIAN, over exactly the declared seeds, and **correct mode only**: ``null`` and
  ``shuffled`` rollouts are reported context (:func:`ablation_summary`) and can never move
  the statistic.
* ``fraction(c) = frac{w : m_corr(w, c) >= threshold}`` over a **denominator fixed at build
  time** -- the canonical windows passed in as ``canonical_windows``. Collision-flagged
  windows are recorded in the output and STAY in the denominator; there is deliberately no
  code path that drops a window.
* **Headline claim** ("canonical-window memorization"): established iff
  ``max_{c in C3_100} fraction(c) >= 0.90`` at threshold **0.95**; **partial** if that holds
  only at threshold **0.90**; otherwise **none**.
* ``c*`` = ``argmax_c fraction(c)`` at the HEADLINE threshold (the plan's "argmax of that
  fraction"), ties broken by higher mean ``m_corr``, then by the **earlier** step. The
  partial tier's own argmax is reported separately as ``c_star_partial`` for transparency,
  but ``c*`` -- and therefore the full-set gate -- is always the headline-threshold argmax.
* **Two-tier claim**: the stronger "full-set memorization" additionally requires
  ``frac{SSIM(w, c*, seed 0 | correct) >= 0.90} >= 0.90`` over ALL built windows. It is
  evaluable ONLY from a role-validated ``s3_full_set`` pass with COMPLETE coverage of those
  windows at ``c*``; otherwise the gate is ``evaluable: False`` and only the canonical-window
  claim is ever made.

**Cycle-D strengthening (Codex review D1-D3): nothing pass-derived, nothing operator-trusted.**

* The canonical denominator is derived from the AUTHENTICATED manifest
  (:func:`canonical_cohort_from_manifest`), independently of what any pass selected -- a sparse
  ten-window artifact can no longer produce denominator 10. ``require_cohort`` enforces that
  inside :func:`evaluate_success`, and the CLI additionally verifies each artifact's recorded
  ``manifest_sha256`` against the manifest file it was handed.
* Every artifact declares a **pass role** (:data:`PASS_ROLES`) and is validated against that
  role's D11 contract by :func:`validate_artifact_role`: exact seeds, exact modes, exact cohort
  scope, ``25`` sampling steps, and a COMPLETE ``(window, seed, mode)`` row grid. ``C3_100`` is
  only the role-validated ``s3_segment_final`` artifacts (:func:`segment_final_checkpoints_from_artifacts`);
  :func:`assert_artifacts_consistent` additionally refuses mixed run name / manifest hash /
  dataset / commit and any non-25-step artifact.
* The manifest cohort helpers are the SINGLE definition of the canonical-window math:
  ``generate_wan_side_adapter`` imports them from here (pinned by a parity test), so the
  evaluator and the verdict can never disagree about what the cohort is.

Fail-closed by default: at a segment-final checkpoint every canonical window must carry
exactly the declared seeds, or :func:`m_corr` raises (a median over 2 of 3 seeds is not the
defined statistic). ``strict=False`` keeps going but marks such windows unmeasured, counts
them as NOT passing (the denominator never shrinks), and lists them in the output.

Rows may be supplied from several aggregation artifacts (the 3-seed canonical pass and the
1-seed all-window pass are separate runs). An identical measurement reported twice is
collapsed; a **conflicting** duplicate -- same ``(window, checkpoint, seed, mode)``, different
SSIM -- is refused, because silently averaging two disagreeing measurements would corrupt the
median.

A thin ``__main__`` CLI (json in -> verdict json out) sits at the bottom of the file, outside
the pure core, so the verdict is machine-written from the artifacts rather than transcribed.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from typing import Iterable, Mapping, Sequence

VERDICT_SCHEMA = "overfit100_success_verdict_v1"
# The aggregation artifact the evaluator writes and this module consumes. v2 (cycle-D
# strengthening) adds the pass role, the manifest hash, and the manifest-derived cohort with
# explicit covered/missing sets -- a v1 artifact is refused rather than reinterpreted.
AGGREGATION_SCHEMA = "overfit100_eval_aggregation_v2"

CORRECT_MODE = "correct"
ABLATION_MODES = ("null", "shuffled")
DEFAULT_SEEDS = (0, 1, 2)

# D11's coverage matrix as four named cells. Every artifact declares exactly one.
PASS_ROLES = ("s2_gate", "s3_intermediate", "s3_segment_final", "s3_full_set")
S3_ROLES = ("s3_intermediate", "s3_segment_final", "s3_full_set")
SEGMENT_FINAL_ROLE = "s3_segment_final"
FULL_SET_ROLE = "s3_full_set"
# D9/D11: the rollout is a 25-step sampler. A pass at any other step count is not comparable
# with the cohort and is refused outright.
REQUIRED_SAMPLING_STEPS = 25
# The provenance fields every artifact in one verdict must agree on (D2).
SHARED_PROVENANCE_FIELDS = ("run_name", "manifest_sha256", "eval_data_dir", "commit")

HEADLINE_THRESHOLD = 0.95
PARTIAL_THRESHOLD = 0.90
CLAIM_FRACTION = 0.90
FULL_SET_SEED = 0
FULL_SET_THRESHOLD = 0.90
FULL_SET_FRACTION = 0.90

_REQUIRED_ROW_FIELDS = ("episode_id", "window_start", "checkpoint_step", "seed", "context_mode", "ssim")
_TIE_TOL = 1e-12
_VALUE_TOL = 1e-9

WindowKey = tuple[int, int]


# --------------------------------------------------------------------------------------
# D1: the FIXED cohorts, derived from the authenticated manifest -- the single definition of
# the canonical-window math (``generate_wan_side_adapter`` imports these).
# --------------------------------------------------------------------------------------

WINDOW_STRIDE = 4


def canonical_window_start(n_windows: int) -> int:
    """The plan's canonical (median) window of an episode: ``4 * floor((n_w - 1) / 2)``.

    ``floor``, not round-half-up: an episode with an even window count takes the LOWER of its two
    middle windows, so the cohort is a pure function of the manifest's ``n_windows`` with no
    tie-break convention to remember.
    """
    count = int(n_windows)
    if count < 1:
        raise ValueError(f"canonical_window_start needs a positive window count; got {n_windows!r}")
    return WINDOW_STRIDE * ((count - 1) // 2)


def manifest_episode_rows(manifest: Mapping, *, episode_indices=None) -> list[dict]:
    """The manifest's episodes as ``{episode_index, episode_id, n_windows}``, sorted by index.

    ``episode_indices`` restricts to one built set (e.g. ``train10``); every requested index must
    exist in the manifest, so a set claiming an unreviewed episode cannot produce a cohort.
    """
    entries = manifest.get("episodes") or []
    if not entries:
        raise ValueError("manifest lists no episodes; cannot derive a cohort")
    rows: dict[int, dict] = {}
    for entry in entries:
        index = int(entry["episode_index"])
        if index in rows:
            raise ValueError(f"manifest repeats episode_index {index}")
        if "n_windows" not in entry:
            raise ValueError(
                f"manifest episode_index {index} carries no n_windows; the canonical window is "
                f"4 * floor((n_windows - 1) / 2) and cannot be derived without it"
            )
        rows[index] = {
            "episode_index": index,
            "episode_id": int(entry["episode_id"]),
            "n_windows": int(entry["n_windows"]),
        }
    if episode_indices is None:
        wanted = sorted(rows)
    else:
        wanted = sorted(int(i) for i in episode_indices)
        missing = [index for index in wanted if index not in rows]
        if missing:
            raise ValueError(
                f"episode_index {missing} not defined by the manifest (it defines 0..{max(rows)}); refusing to "
                f"derive a cohort that includes an episode the experiment never selected"
            )
    return [rows[index] for index in wanted]


def canonical_cohort_from_manifest(manifest: Mapping, *, episode_indices=None) -> tuple[WindowKey, ...]:
    """The FIXED canonical denominator: one ``(episode_id, canonical_start)`` key per episode."""
    return tuple(
        (row["episode_id"], canonical_window_start(row["n_windows"]))
        for row in manifest_episode_rows(manifest, episode_indices=episode_indices)
    )


def all_window_keys_from_manifest(manifest: Mapping, *, episode_indices=None) -> tuple[WindowKey, ...]:
    """Every BUILT window key, in ``(episode_index, window_start)`` order (the full-set cohort)."""
    keys: list[WindowKey] = []
    for row in manifest_episode_rows(manifest, episode_indices=episode_indices):
        keys.extend((row["episode_id"], WINDOW_STRIDE * slot) for slot in range(row["n_windows"]))
    return tuple(keys)


# --------------------------------------------------------------------------------------
# D2: pass roles -- D11's coverage matrix, validated instead of trusted.
# --------------------------------------------------------------------------------------


def role_requirements(role: str) -> dict:
    """The D11 contract of one pass role: exact seeds, exact modes, and the cohort scope.

    * ``s3_segment_final`` -- 3 seeds x 3 modes over the FULL canonical cohort (the only role
      that may enter ``C3_100``);
    * ``s3_intermediate``  -- 1 seed, correct mode, full canonical cohort;
    * ``s3_full_set``      -- seed 0, correct mode, EVERY built window (the stronger tier's input);
    * ``s2_gate``          -- 3 seeds over the (train10) cohort; correct mode mandatory, the two
      ablations optional because D11 runs them only at the gate's final checkpoint.
    """
    if role not in PASS_ROLES:
        raise ValueError(f"unknown pass role {role!r}; valid roles are {list(PASS_ROLES)}")
    return {
        "s2_gate": {
            "seeds": (0, 1, 2),
            "modes": (CORRECT_MODE,),
            "optional_modes": ABLATION_MODES,
            "scope": "canonical",
        },
        "s3_intermediate": {"seeds": (0,), "modes": (CORRECT_MODE,), "optional_modes": (), "scope": "canonical"},
        "s3_segment_final": {
            "seeds": (0, 1, 2),
            "modes": (CORRECT_MODE,) + ABLATION_MODES,
            "optional_modes": (),
            "scope": "canonical",
        },
        "s3_full_set": {"seeds": (0,), "modes": (CORRECT_MODE,), "optional_modes": (), "scope": "all_windows"},
    }[role]


def pass_role_plan_reasons(
    role: str,
    *,
    seeds,
    modes,
    sampling_steps,
    covered_canonical,
    covered_all,
    cohort,
    all_window_keys,
) -> list[str]:
    """Why a PLANNED pass does not satisfy its role (empty list = it does).

    Pure over the plan, so the evaluator can refuse a mislabeled pass in SECONDS -- before the
    5B load -- and the aggregator can re-derive the same verdict from the written artifact
    without trusting any flag the pass recorded about itself.
    """
    spec = role_requirements(role)
    reasons: list[str] = []
    if int(sampling_steps) != REQUIRED_SAMPLING_STEPS:
        reasons.append(
            f"sampling_steps={int(sampling_steps)} but role {role} requires exactly "
            f"{REQUIRED_SAMPLING_STEPS} (D9's rollout sampler; another step count is not comparable)"
        )
    seed_set = tuple(sorted({int(s) for s in seeds}))
    if seed_set != tuple(sorted(spec["seeds"])):
        reasons.append(f"seeds={list(seed_set)} but role {role} requires exactly {list(spec['seeds'])}")
    mode_set = {str(m) for m in modes}
    required_modes = set(spec["modes"])
    allowed_modes = required_modes | set(spec["optional_modes"])
    if not required_modes <= mode_set:
        reasons.append(f"modes={sorted(mode_set)} but role {role} requires {sorted(required_modes)}")
    extra_modes = sorted(mode_set - allowed_modes)
    if extra_modes:
        reasons.append(f"modes {extra_modes} are not part of role {role} (allowed: {sorted(allowed_modes)})")
    cohort_keys = _normalize_keys(cohort, what="cohort")
    all_keys = _normalize_keys(all_window_keys, what="all_window_keys")
    if spec["scope"] == "canonical":
        covered = set(_normalize_keys(covered_canonical, what="covered_canonical"))
        missing = [list(key) for key in cohort_keys if key not in covered]
        extra = [list(key) for key in sorted(covered - set(cohort_keys))]
        if missing:
            reasons.append(
                f"role {role} must cover the whole canonical cohort ({len(cohort_keys)} windows); "
                f"{len(missing)} missing, e.g. {missing[:3]}"
            )
        if extra:
            reasons.append(f"covered canonical windows {extra[:3]} are not in the derived cohort")
    else:
        covered = set(_normalize_keys(covered_all, what="covered_windows"))
        missing = [list(key) for key in all_keys if key not in covered]
        extra = [list(key) for key in sorted(covered - set(all_keys))]
        if missing:
            reasons.append(
                f"role {role} must cover every built window ({len(all_keys)} windows); "
                f"{len(missing)} missing, e.g. {missing[:3]}"
            )
        if extra:
            reasons.append(f"covered windows {extra[:3]} are not built windows of the derived cohort")
    return reasons


def _artifact_grid_reasons(artifact: Mapping, *, role: str, keys: Sequence[WindowKey]) -> list[str]:
    """Every ``(window, CHECKPOINT, seed, mode)`` cell the role promises must have a row.

    **E1 (close-out review): the grid is CHECKPOINT-BOUND.** Keying it on ``(window, seed, mode)``
    alone let an artifact declare step 2500, carry complete correct-mode rows at 2500 and its
    null/shuffled rows at step 1000, and still validate as a segment final -- producing an
    ``established`` headline with ZERO contemporaneous controls at the checkpoint being judged.
    Two rules close that:

    1. every required cell must exist AT THE ARTIFACT'S DECLARED ``checkpoint_step``;
    2. a row carrying any OTHER ``checkpoint_step`` is REJECTED -- an artifact is one checkpoint's
       evidence, so mixed-checkpoint rows are a provenance error, not a partial pass.
    """
    del role  # the caller already resolved the role's scope into ``keys``
    checkpoint = int(artifact.get("checkpoint_step", -1))
    modes = [str(m) for m in artifact.get("context_modes") or ()]
    seeds = [int(s) for s in artifact.get("rollout_seeds") or ()]
    present: set[tuple] = set()
    foreign: dict[int, int] = {}
    for row in artifact.get("rows") or ():
        step = int(_field(row, "checkpoint_step"))
        if step != checkpoint:
            foreign[step] = foreign.get(step, 0) + 1
            continue
        present.add((window_key(row), step, int(_field(row, "seed")), str(_field(row, "context_mode"))))
    reasons: list[str] = []
    if foreign:
        reasons.append(
            f"{sum(foreign.values())} row(s) carry checkpoint_step {sorted(foreign)} but the artifact declares "
            f"{checkpoint}; an artifact is ONE checkpoint's evidence, so mixed-checkpoint rows are refused"
        )
    missing = [
        [list(key), checkpoint, seed, mode]
        for key in keys
        for seed in seeds
        for mode in modes
        if (key, checkpoint, seed, mode) not in present
    ]
    if missing:
        reasons.append(
            f"{len(missing)} declared (window, checkpoint, seed, mode) row(s) are absent at checkpoint {checkpoint}, "
            f"e.g. {missing[:3]}; the artifact's coverage claim is not backed by rows AT ITS OWN CHECKPOINT"
        )
    return reasons


def validate_artifact_role(artifact: Mapping, *, canonical_cohort, all_window_keys) -> dict:
    """Validate ONE artifact against its declared role and the manifest-derived cohorts.

    Returns ``{"ok", "role", "checkpoint_step", "reasons"}``. Nothing the artifact says about
    itself is trusted: the schema tag, the role, the seeds/modes, the cohort, the covered sets AND
    the row grid are all re-checked here, so a pass cannot label itself segment-final without
    having done the segment-final work.
    """
    reasons: list[str] = []
    schema = str(artifact.get("schema", ""))
    if schema != AGGREGATION_SCHEMA:
        reasons.append(f"schema {schema!r} is not {AGGREGATION_SCHEMA!r}")
    role = str(artifact.get("eval_pass_role", "") or "")
    if role not in PASS_ROLES:
        reasons.append(f"eval_pass_role {role!r} is not one of {list(PASS_ROLES)}")
        return {"ok": False, "role": role, "checkpoint_step": artifact.get("checkpoint_step"), "reasons": reasons}
    cohort = _normalize_keys(canonical_cohort, what="canonical_cohort")
    all_keys = _normalize_keys(all_window_keys, what="all_window_keys")
    recorded_cohort = _normalize_keys(artifact.get("canonical_cohort") or (), what="artifact canonical_cohort")
    if set(recorded_cohort) != set(cohort) or len(recorded_cohort) != len(cohort):
        reasons.append(
            f"the artifact's canonical_cohort ({len(recorded_cohort)} windows) is not the manifest-derived cohort "
            f"({len(cohort)} windows)"
        )
    reasons.extend(
        pass_role_plan_reasons(
            role,
            seeds=artifact.get("rollout_seeds") or (),
            modes=artifact.get("context_modes") or (),
            sampling_steps=artifact.get("sampling_steps", 0),
            covered_canonical=artifact.get("covered_canonical_windows") or (),
            covered_all=artifact.get("covered_windows") or (),
            cohort=cohort,
            all_window_keys=all_keys,
        )
    )
    scope_keys = cohort if role_requirements(role)["scope"] == "canonical" else all_keys
    reasons.extend(_artifact_grid_reasons(artifact, role=role, keys=scope_keys))
    return {
        "ok": not reasons,
        "role": role,
        "checkpoint_step": int(artifact.get("checkpoint_step", -1)),
        "reasons": reasons,
    }


def assert_artifacts_consistent(artifacts: Sequence[Mapping]) -> dict:
    """One verdict may only be built from artifacts of the SAME run, manifest, dataset and commit.

    Mixed provenance would silently splice measurements from different experiments into one
    statistic, and a non-25-step artifact is not comparable at all -- both are refused (D2).
    Returns the shared provenance for the verdict's own record.
    """
    if not artifacts:
        raise ValueError("no aggregation artifacts supplied")
    shared: dict = {}
    for index, artifact in enumerate(artifacts):
        steps = int(artifact.get("sampling_steps", 0))
        if steps != REQUIRED_SAMPLING_STEPS:
            raise ValueError(
                f"aggregation artifact {index} was rolled out with sampling_steps={steps}, not "
                f"{REQUIRED_SAMPLING_STEPS}; refusing to mix step counts in one verdict"
            )
        for field in SHARED_PROVENANCE_FIELDS:
            value = artifact.get(field)
            if field not in shared:
                shared[field] = value
            elif shared[field] != value:
                raise ValueError(
                    f"aggregation artifacts disagree on {field}: {shared[field]!r} vs {value!r}. One verdict must be "
                    f"built from one run's passes (same manifest, dataset, and eval commit)."
                )
    shared["sampling_steps"] = REQUIRED_SAMPLING_STEPS
    return shared


def validate_artifacts(artifacts: Sequence[Mapping], *, canonical_cohort, all_window_keys) -> list[dict]:
    """Validate every artifact once; the result list is positionally aligned with ``artifacts``."""
    return [
        validate_artifact_role(artifact, canonical_cohort=canonical_cohort, all_window_keys=all_window_keys)
        for artifact in artifacts
    ]


def admitted_artifacts(artifacts: Sequence[Mapping], results: Sequence[Mapping], *, role: str) -> list[Mapping]:
    """The artifacts of ``role`` that FULLY validated -- whole-artifact admission (E1).

    The statistic consumes admitted ARTIFACTS, never rows filtered by an artifact's self-declared
    label: an artifact that claims a role it does not satisfy contributes nothing at all, so its
    numbers can neither enter a median nor collide with a valid pass's measurements.
    """
    if len(artifacts) != len(results):
        raise ValueError(f"validation results ({len(results)}) do not align with artifacts ({len(artifacts)})")
    return [artifact for artifact, result in zip(artifacts, results) if result["role"] == role and result["ok"]]


def _rejection_notes(results: Sequence[Mapping], *, role: str) -> list[str]:
    notes = []
    for index, result in enumerate(results):
        if result["role"] != role:
            notes.append(f"artifact {index}: role {result['role']!r}")
        elif not result["ok"]:
            notes.append(f"artifact {index} (step {result['checkpoint_step']}): {'; '.join(result['reasons'])}")
    return notes


def segment_final_checkpoints_from_artifacts(
    artifacts: Sequence[Mapping], *, canonical_cohort, all_window_keys, results=None
):
    """``C3_100`` = the steps of the ROLE-VALIDATED ``s3_segment_final`` artifacts, sorted.

    Anything else -- S2 gates, intermediate passes, full-set passes, and any artifact that
    declares ``s3_segment_final`` without satisfying it (including one whose rows are not all at
    its declared checkpoint, E1) -- is excluded, and an empty result is a loud failure naming why
    each candidate was rejected.
    """
    if results is None:
        results = validate_artifacts(artifacts, canonical_cohort=canonical_cohort, all_window_keys=all_window_keys)
    admitted = admitted_artifacts(artifacts, results, role=SEGMENT_FINAL_ROLE)
    steps = sorted({int(artifact["checkpoint_step"]) for artifact in admitted})
    if not steps:
        raise ValueError(
            f"no role-validated {SEGMENT_FINAL_ROLE} artifact was supplied, so C3_100 is empty and no headline claim "
            f"can be made. Rejections: " + " | ".join(_rejection_notes(results, role=SEGMENT_FINAL_ROLE))
        )
    return steps


def full_set_input_from_artifacts(artifacts: Sequence[Mapping], *, canonical_cohort, all_window_keys, results=None):
    """The stronger tier's input: the role-validated ``s3_full_set`` artifacts' windows + rows.

    Returns ``(None, reasons)`` when no such artifact validates -- the tier is then simply not
    evaluable, never scored from a partial or mixed-checkpoint pass.
    """
    if results is None:
        results = validate_artifacts(artifacts, canonical_cohort=canonical_cohort, all_window_keys=all_window_keys)
    admitted = admitted_artifacts(artifacts, results, role=FULL_SET_ROLE)
    rows = rows_from_artifacts(admitted)
    reasons = [
        f"artifact {index}: {'; '.join(result['reasons'])}"
        for index, result in enumerate(results)
        if result["role"] == FULL_SET_ROLE and not result["ok"]
    ]
    if not rows:
        return None, reasons
    return {"windows": _normalize_keys(all_window_keys, what="all_window_keys"), "rows": rows}, reasons


# --------------------------------------------------------------------------------------
# Row plumbing.
# --------------------------------------------------------------------------------------


def window_key(row: Mapping) -> WindowKey:
    """The stable window identity ``(episode_id, window_start)`` -- never the record order."""
    return (int(_field(row, "episode_id")), int(_field(row, "window_start")))


def _field(row: Mapping, name: str):
    if name not in row:
        raise ValueError(f"aggregation row is missing the required field {name!r}: {sorted(row)}")
    return row[name]


def _normalize_keys(keys: Iterable, *, what: str) -> tuple[WindowKey, ...]:
    out: list[WindowKey] = []
    for key in keys:
        pair = tuple(int(v) for v in key)
        if len(pair) != 2:
            raise ValueError(f"{what} entries must be (episode_id, window_start) pairs; got {key!r}")
        out.append((pair[0], pair[1]))
    duplicates = sorted({k for k in out if out.count(k) > 1})
    if duplicates:
        raise ValueError(
            f"{what} contains duplicate window keys {duplicates}; the denominator is fixed at build time, so every "
            f"window must appear exactly once"
        )
    return tuple(out)


def _collect(rows: Iterable[Mapping], *, mode: str, checkpoint: int, keys: Sequence[WindowKey], seeds=None) -> dict:
    """``{window_key: {seed: ssim}}`` for one mode at one checkpoint, restricted to ``keys``."""
    wanted = set(keys)
    allowed = None if seeds is None else {int(s) for s in seeds}
    out: dict[WindowKey, dict[int, float]] = {key: {} for key in wanted}
    for row in rows:
        if str(_field(row, "context_mode")) != mode or int(_field(row, "checkpoint_step")) != int(checkpoint):
            continue
        key = window_key(row)
        if key not in wanted:
            continue
        seed = int(_field(row, "seed"))
        if allowed is not None and seed not in allowed:
            raise ValueError(
                f"aggregation rows contain seed {seed} for window {key} at checkpoint {checkpoint}, which is not in "
                f"the declared seed set {sorted(allowed)}; the statistic is defined over exactly those seeds."
            )
        value = float(_field(row, "ssim"))
        if not math.isfinite(value):
            raise ValueError(f"non-finite SSIM {value!r} for window {key} seed {seed} at checkpoint {checkpoint}")
        if seed in out[key]:
            if abs(out[key][seed] - value) > _VALUE_TOL:
                raise ValueError(
                    f"conflicting duplicate measurement for window {key} checkpoint {checkpoint} seed {seed} mode "
                    f"{mode!r}: {out[key][seed]} vs {value}. Identical repeats (the same window reported by two "
                    f"artifacts) are collapsed, but two different values cannot both be the measurement."
                )
            continue
        out[key][seed] = value
    return out


# --------------------------------------------------------------------------------------
# The statistic.
# --------------------------------------------------------------------------------------


def m_corr(rows, *, checkpoint: int, windows, seeds=DEFAULT_SEEDS, strict: bool = True) -> dict:
    """``m_corr(w, c)`` = median over ``seeds`` of the CORRECT-mode SSIM, per window.

    Returns ``{window_key: median}`` for every window in ``windows`` (the fixed denominator).
    ``strict=True`` (default) raises when a window lacks any declared seed -- a median over a
    partial seed set is not the plan's statistic. ``strict=False`` yields ``None`` for such a
    window; callers must then count it as NOT passing.
    """
    keys = _normalize_keys(windows, what="windows")
    seed_tuple = tuple(int(s) for s in seeds)
    if not seed_tuple:
        raise ValueError("m_corr requires a non-empty seed set")
    collected = _collect(rows, mode=CORRECT_MODE, checkpoint=checkpoint, keys=keys, seeds=seed_tuple)
    out: dict[WindowKey, float | None] = {}
    missing: list[str] = []
    for key in keys:
        per_seed = collected[key]
        absent = [s for s in seed_tuple if s not in per_seed]
        if absent:
            missing.append(f"episode {key[0]} window_start {key[1]} missing seed(s) {absent}")
            out[key] = None
            continue
        out[key] = float(statistics.median([per_seed[s] for s in seed_tuple]))
    if missing and strict:
        raise ValueError(
            f"incomplete correct-mode coverage at checkpoint {checkpoint} over the declared seeds {list(seed_tuple)}: "
            + "; ".join(missing[:8])
            + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
        )
    return out


def fraction_at(m_map: Mapping, threshold: float, *, denominator: int) -> float:
    """``frac{w : m_corr(w) >= threshold}`` over a FIXED denominator (unmeasured => not passing)."""
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError(f"fraction_at needs a positive denominator (the fixed window count); got {denominator}")
    passing = sum(1 for value in m_map.values() if value is not None and float(value) >= float(threshold))
    return passing / denominator


def ablation_summary(rows, *, checkpoints, windows, seeds=DEFAULT_SEEDS) -> dict:
    """Reported context (never an input to the rule): per-ablation-mode SSIM and its gap.

    ``mean_gap_vs_correct`` averages ``SSIM(correct) - SSIM(mode)`` over the ``(window, seed,
    checkpoint)`` triples where BOTH were measured, so the gap is paired rather than a
    difference of two differently-covered means.
    """
    keys = _normalize_keys(windows, what="windows")
    out: dict[str, dict] = {}
    for mode in ABLATION_MODES:
        values: list[float] = []
        gaps: list[float] = []
        for checkpoint in checkpoints:
            ablation = _collect(rows, mode=mode, checkpoint=checkpoint, keys=keys)
            correct = _collect(rows, mode=CORRECT_MODE, checkpoint=checkpoint, keys=keys)
            for key in keys:
                for seed, value in sorted(ablation[key].items()):
                    values.append(value)
                    if seed in correct[key]:
                        gaps.append(correct[key][seed] - value)
        out[mode] = {
            "n_rows": len(values),
            "mean_ssim": (sum(values) / len(values)) if values else None,
            "mean_gap_vs_correct": (sum(gaps) / len(gaps)) if gaps else None,
            "n_paired": len(gaps),
        }
    del seeds  # the summary is over whatever ablation seeds were run, not the statistic's set
    return out


def _pick_c_star(candidates: Sequence[Mapping]) -> tuple[int, str]:
    """argmax fraction -> higher mean m_corr -> EARLIER step. Deterministic, order-free."""
    ordered = sorted(candidates, key=lambda entry: int(entry["checkpoint_step"]))
    best = ordered[0]
    for entry in ordered[1:]:
        if entry["fraction"] > best["fraction"] + _TIE_TOL:
            best = entry
        elif (
            math.isclose(entry["fraction"], best["fraction"], rel_tol=0.0, abs_tol=_TIE_TOL)
            and _mean(entry) > _mean(best) + _TIE_TOL
        ):
            best = entry
    fraction_ties = [
        e for e in ordered if math.isclose(e["fraction"], best["fraction"], rel_tol=0.0, abs_tol=_TIE_TOL)
    ]
    if len(fraction_ties) == 1:
        reason = "fraction"
    else:
        mean_ties = [e for e in fraction_ties if math.isclose(_mean(e), _mean(best), rel_tol=0.0, abs_tol=_TIE_TOL)]
        reason = "mean_m_corr" if len(mean_ties) == 1 else "earlier_step"
    return int(best["checkpoint_step"]), reason


def _mean(entry: Mapping) -> float:
    value = entry.get("mean_m_corr")
    return float("-inf") if value is None else float(value)


def evaluate_success(
    rows,
    *,
    canonical_windows,
    segment_final_checkpoints,
    seeds=DEFAULT_SEEDS,
    headline_threshold: float = HEADLINE_THRESHOLD,
    partial_threshold: float = PARTIAL_THRESHOLD,
    claim_fraction: float = CLAIM_FRACTION,
    full_set=None,
    full_set_reasons=(),
    full_set_seed: int = FULL_SET_SEED,
    full_set_threshold: float = FULL_SET_THRESHOLD,
    full_set_fraction: float = FULL_SET_FRACTION,
    flagged_windows=(),
    strict: bool = True,
    require_cohort=None,
) -> dict:
    """The plan's two-tier verdict over the aggregation rows. Returns a JSON-ready dict.

    ``require_cohort`` (D1) is the manifest-derived cohort: when supplied, ``canonical_windows``
    must equal it exactly, so no caller -- CLI or otherwise -- can score an S3 claim against a
    shrunken denominator. ``full_set`` is ``{"windows": [...], "rows": [...]}`` from a
    ROLE-VALIDATED ``s3_full_set`` pass (:func:`full_set_input_from_artifacts`); the stronger tier
    is scored from those rows only, and only when they cover every window.
    """
    keys = _normalize_keys(canonical_windows, what="canonical_windows")
    if not keys:
        raise ValueError("canonical_windows is the FIXED denominator and cannot be empty")
    if require_cohort is not None:
        required = _normalize_keys(require_cohort, what="require_cohort")
        if set(keys) != set(required) or len(keys) != len(required):
            raise ValueError(
                f"the verdict's denominator ({len(keys)} canonical windows) is not the required manifest-derived "
                f"cohort ({len(required)} windows). The denominator is FIXED at build time (plan §1/G4): a pass that "
                f"covered fewer windows cannot shrink it -- report missing coverage instead."
            )
    checkpoints = sorted({int(c) for c in segment_final_checkpoints})
    if not checkpoints:
        raise ValueError("segment_final_checkpoints (C3_100) cannot be empty; S2 checkpoints never enter the rule")
    flagged = _normalize_keys(flagged_windows, what="flagged_windows")
    unknown = [key for key in flagged if key not in set(keys)]
    if unknown:
        raise ValueError(
            f"flagged_windows {unknown} are not in the canonical window set; flagging records a collision, it never "
            f"changes the denominator, so a flag outside the denominator is a bookkeeping error."
        )
    seed_tuple = tuple(int(s) for s in seeds)
    denominator = len(keys)

    per_checkpoint = []
    coverage_complete = True
    for checkpoint in checkpoints:
        values = m_corr(rows, checkpoint=checkpoint, windows=keys, seeds=seed_tuple, strict=strict)
        measured = [v for v in values.values() if v is not None]
        unmeasured = [key for key in keys if values[key] is None]
        coverage_complete = coverage_complete and not unmeasured
        per_checkpoint.append(
            {
                "checkpoint_step": checkpoint,
                "n_windows": denominator,
                "n_measured": len(measured),
                "fraction": fraction_at(values, headline_threshold, denominator=denominator),
                "fraction_partial": fraction_at(values, partial_threshold, denominator=denominator),
                "mean_m_corr": (sum(measured) / len(measured)) if measured else None,
                "unmeasured_windows": [list(key) for key in unmeasured],
                "m_corr": {str(list(key)): values[key] for key in keys},
            }
        )

    headline_fraction = max(entry["fraction"] for entry in per_checkpoint)
    partial_fraction = max(entry["fraction_partial"] for entry in per_checkpoint)
    established = headline_fraction >= claim_fraction
    partial = (not established) and partial_fraction >= claim_fraction
    verdict = "established" if established else ("partial" if partial else "none")

    c_star, tie_break = _pick_c_star(per_checkpoint)
    partial_candidates = [{**entry, "fraction": entry["fraction_partial"]} for entry in per_checkpoint]
    c_star_partial, tie_break_partial = _pick_c_star(partial_candidates)

    full_set_claim = _full_set_claim(
        full_set,
        c_star=c_star,
        seed=int(full_set_seed),
        threshold=float(full_set_threshold),
        required_fraction=float(full_set_fraction),
        headline_established=established,
        reasons=list(full_set_reasons or ()),
    )

    return {
        "schema": VERDICT_SCHEMA,
        "verdict": verdict,
        "context_mode": CORRECT_MODE,
        "seeds": list(seed_tuple),
        "segment_final_checkpoints": checkpoints,
        "denominator": denominator,
        "canonical_windows": [list(key) for key in keys],
        "flagged_windows": [list(key) for key in flagged],
        "claim_fraction": float(claim_fraction),
        "coverage_complete": coverage_complete,
        "headline": {
            "claim": "canonical-window memorization",
            "threshold": float(headline_threshold),
            "fraction": headline_fraction,
            "established": established,
        },
        "partial": {
            "claim": "canonical-window memorization (partial)",
            "threshold": float(partial_threshold),
            "fraction": partial_fraction,
            "established": partial,
        },
        "c_star": c_star,
        "c_star_tie_break": tie_break,
        "c_star_partial": c_star_partial,
        "c_star_partial_tie_break": tie_break_partial,
        "per_checkpoint": per_checkpoint,
        "full_set_claim": full_set_claim,
        "ablation_summary": ablation_summary(rows, checkpoints=checkpoints, windows=keys, seeds=seed_tuple),
    }


def _full_set_claim(full_set, *, c_star, seed, threshold, required_fraction, headline_established, reasons) -> dict:
    """The stronger tier: seed-``seed`` correct-mode fraction over ALL built windows at ``c*``.

    D3: scored ONLY from a role-validated ``s3_full_set`` pass's own rows, and only when that pass
    covers EVERY built window at ``c*``. Incomplete coverage is *not* a low fraction -- it makes
    the tier unevaluable, so a partial pass can never understate (or overstate) the claim.
    """
    base = {
        "claim": "full-set memorization",
        "seed": seed,
        "threshold": threshold,
        "required_fraction": required_fraction,
        "checkpoint_step": c_star,
        "evaluable": False,
        "established": False,
        "fraction": None,
        "n_windows": 0,
        "n_measured": 0,
    }
    if not full_set or not full_set.get("windows"):
        detail = (" Rejected candidates: " + " | ".join(reasons)) if reasons else ""
        base["reason"] = (
            f"no role-validated {FULL_SET_ROLE} pass was supplied, so the stronger tier is not evaluable and only "
            f"the canonical-window claim is made (plan D11's two-tier structure).{detail}"
        )
        return base
    keys = _normalize_keys(full_set["windows"], what="full_set windows")
    collected = _collect(full_set.get("rows") or (), mode=CORRECT_MODE, checkpoint=c_star, keys=keys, seeds=None)
    measured = {key: values[seed] for key, values in collected.items() if seed in values}
    base["n_windows"] = len(keys)
    base["n_measured"] = len(measured)
    if len(measured) != len(keys):
        base["unmeasured_windows"] = [list(key) for key in keys if key not in measured][:16]
        base["reason"] = (
            f"incomplete {FULL_SET_ROLE} coverage at c*={c_star}: {len(measured)} of {len(keys)} built windows carry a "
            f"seed-{seed} correct-mode row. The tier requires the whole set, so it is not evaluable rather than scored "
            f"on a subset."
        )
        return base
    passing = sum(1 for value in measured.values() if value >= threshold)
    fraction = passing / len(keys)  # the FIXED all-window denominator
    base.update(
        {
            "evaluable": True,
            "fraction": fraction,
            "established": bool(headline_established and fraction >= required_fraction),
        }
    )
    return base


def rows_from_artifacts(artifacts: Iterable[Mapping]) -> list[dict]:
    """Concatenate the ``rows`` of several aggregation artifacts, in the given order.

    E1: there is deliberately NO role filter here. Selecting rows by an artifact's self-declared
    label is exactly the weaker mechanism the close-out review rejected -- callers pass the output
    of :func:`admitted_artifacts`, i.e. whole artifacts that passed validation, so an artifact that
    merely claims a role contributes nothing.
    """
    out: list[dict] = []
    for index, artifact in enumerate(artifacts):
        if "rows" not in artifact:
            raise ValueError(f"aggregation artifact {index} has no 'rows' key: {sorted(artifact)}")
        out.extend(artifact["rows"])
    return out


# --------------------------------------------------------------------------------------
# Thin CLI (the only IO in this file): artifacts in -> verdict json out.
# --------------------------------------------------------------------------------------


def _sha256_file(path) -> str:
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def verdict_from_artifact_files(paths, manifest_path, out_path=None, *, episode_indices=None, **kwargs) -> dict:
    """The machine verdict: authenticated manifest + role-validated artifacts in, verdict JSON out.

    The chain, end to end (D1-D3), with nothing operator-supplied:

    1. read the artifacts and the MANIFEST, and require every artifact's recorded
       ``manifest_sha256`` to equal ``sha256(manifest bytes)`` -- the same hash cycle C bound to
       the published dataset through ``_SUCCESS.manifest_sha256``, so the cohort below is
       authenticated rather than asserted;
    2. refuse mixed run/manifest/dataset/commit provenance and non-25-step passes;
    3. DERIVE the canonical cohort and the all-window cohort from that manifest;
    4. take ``C3_100`` from the role-validated ``s3_segment_final`` artifacts only, and the
       stronger tier's input from the role-validated ``s3_full_set`` artifacts only;
    5. score with ``require_cohort`` set, so the denominator cannot be anything but the derived
       cohort.
    """
    artifacts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            artifacts.append(json.load(handle))
    shared = assert_artifacts_consistent(artifacts)

    manifest_sha256 = _sha256_file(manifest_path)
    recorded = str(shared.get("manifest_sha256") or "")
    if recorded != manifest_sha256:
        raise ValueError(
            f"the artifacts record manifest_sha256={recorded!r} but sha256({manifest_path})={manifest_sha256}. The "
            f"cohort must be derived from the manifest the passes actually used; refusing to score against a "
            f"different manifest."
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    cohort = canonical_cohort_from_manifest(manifest, episode_indices=episode_indices)
    all_keys = all_window_keys_from_manifest(manifest, episode_indices=episode_indices)
    totals = manifest.get("totals") or {}
    if episode_indices is None and totals:
        # Self-consistency of the authenticated manifest: a truncated episode list would otherwise
        # silently shrink both cohorts.
        if int(totals.get("episodes", len(cohort))) != len(cohort):
            raise ValueError(
                f"manifest totals.episodes={totals.get('episodes')} but {len(cohort)} canonical windows derived"
            )
        if int(totals.get("windows", len(all_keys))) != len(all_keys):
            raise ValueError(
                f"manifest totals.windows={totals.get('windows')} but {len(all_keys)} built window keys derived"
            )

    # E1: validate ONCE, then admit whole artifacts by role. Every downstream input -- the
    # statistic's rows, C3_100, and the stronger tier -- comes from artifacts that FULLY validated.
    results = validate_artifacts(artifacts, canonical_cohort=cohort, all_window_keys=all_keys)
    admitted_segment_final = admitted_artifacts(artifacts, results, role=SEGMENT_FINAL_ROLE)
    checkpoints = kwargs.pop("segment_final_checkpoints", None)
    if checkpoints is None:
        checkpoints = segment_final_checkpoints_from_artifacts(
            artifacts, canonical_cohort=cohort, all_window_keys=all_keys, results=results
        )
    full_set, full_set_reasons = full_set_input_from_artifacts(
        artifacts, canonical_cohort=cohort, all_window_keys=all_keys, results=results
    )
    flagged = kwargs.pop("flagged_windows", None)
    if flagged is None:
        flagged = []
        for artifact in artifacts:
            for key in artifact.get("flagged_windows", ()):
                if list(key) not in flagged:
                    flagged.append(list(key))

    verdict = evaluate_success(
        # The canonical statistic reads the rows of the ADMITTED segment-final artifacts only: S2
        # gates, intermediate passes, the 1-seed full-set pass and any artifact that fails its own
        # role contract contribute nothing, so nothing outside C3_100 can reach the median (D2/E1)
        # and the tiers never collide on a shared window measurement.
        rows_from_artifacts(admitted_segment_final),
        canonical_windows=cohort,
        segment_final_checkpoints=checkpoints,
        flagged_windows=flagged,
        full_set=full_set,
        full_set_reasons=full_set_reasons,
        require_cohort=cohort,
        **kwargs,
    )
    verdict["manifest_path"] = str(manifest_path)
    verdict["manifest_sha256"] = manifest_sha256
    verdict["provenance"] = shared
    verdict["artifact_roles"] = results
    verdict["admitted_artifacts"] = {
        SEGMENT_FINAL_ROLE: len(admitted_segment_final),
        FULL_SET_ROLE: len(admitted_artifacts(artifacts, results, role=FULL_SET_ROLE)),
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    return verdict


def main(argv: Sequence[str]) -> int:
    if len(argv) < 4:
        raise SystemExit(
            "usage: overfit100_success_statistic.py <verdict_out.json> <overfit100_manifest.json> "
            "<aggregation.json> [more.json ...]"
        )
    verdict = verdict_from_artifact_files(list(argv[3:]), manifest_path=argv[2], out_path=argv[1])
    print(json.dumps({key: verdict[key] for key in ("verdict", "c_star", "headline", "full_set_claim")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
