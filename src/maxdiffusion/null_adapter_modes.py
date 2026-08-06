"""The four J1 execution modes for exp_04 null-text inversion (plan §5 item 5; round R10).

``run_wan_null_inversion`` decides *what* to run; this module runs it. The split keeps both files
reviewable and, more importantly, keeps every mode body testable: the model and the filesystem reach
these functions only through two small seams of callables, so the happy path, the quarantine path and
the verification-failure path all have real unit tests with fakes.

- ``capacity`` -- the plan §4-P1 arms over a cohort, then the cohort decode, the pixel fill, **the
  gates and the target selection**, and only then the records, videos and report.
- ``adequacy_probe`` -- §4-P1's F1/M3 grid over the first eight DEV names, the full evidence traces,
  the adoption verdict and the L_null diagnostic.
- ``cache`` -- §4-P2 target caching of the *selected* arm, at the dtype its own fidelity gate
  decided, resume-driven: validated shards are skipped, never rewritten.
- ``verify_replay`` -- §5's independent check: every shard validated, every record replayed, exact
  cohort coverage required. Anything short of that is a non-zero exit.

**Publication is the last thing that happens, and it is ordered.** Records are immutable once
published, so nothing may become immutable before the run knows whether it succeeded. R10's first cut
wrote each batch's shards inside the arm loop, which meant a decode failure at example 63 left 62
examples' worth of completed shards that a retry could never overwrite -- the job was
unre-runnable and the cohort permanently half-published (review, finding 2). The order is now:
arms -> bounded-batch cohort decode -> pixel fill -> gates and selection -> publish.

**Gating order is a rule, not a convention.** Latent tables are never written as gate input before
the pixel fill: R6 leaves ``future_ssim`` absent, and the gates read absent as *invalid*, so a table
published un-filled would make every verdict a coverage failure that looks like a result.

**Gates see the declared cohort, not the surviving one.** Quarantined names are removed from the
work but stay in the manifest the gates are evaluated against, so a run that lost examples fails on
coverage instead of quietly certifying a smaller experiment.
"""

from __future__ import annotations

import dataclasses
import json
import os
import posixpath
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from maxdiffusion.models.wan.null_direct_opt_wan import (
    A3_ITERS,
    direct_optimize_nulls,
    direct_rollout,
    endpoint_future_mse,
    measure_single_update,
)
from maxdiffusion.models.wan.null_inversion_wan import base_context_fingerprint, global_noise
from maxdiffusion.null_adapter_cache_policy import (
    FIDELITY_SUBSET_SIZE,
    fidelity_gate,
    quarantine_batch_failures,
)
from maxdiffusion.null_adapter_gates import (
    NoiseConvention,
    Target,
    gate_g1,
    gate_g2,
    select_target,
    verdicts_to_json,
)
from maxdiffusion.null_adapter_pixels import (
    comparison_video_frames,
    decode_and_score,
    fill_pixel_metrics,
    save_video_mp4,
)
from maxdiffusion.null_adapter_records import (
    PRODUCTION_GEOMETRY,
    ProvenanceHeader,
    record_from_bytes,
    record_to_bytes,
)
from maxdiffusion.null_adapter_runner_core import (
    ADEQUACY_GRID,
    DEFAULT_RECIPE,
    METHODS,
    RECORD_ARMS,
    AdequacyReport,
    CapacityParams,
    adopt_recipe,
    build_capacity_records,
    emit_metric_tables,
    run_adequacy_probe,
    run_capacity_example_batch,
)
from maxdiffusion.null_adapter_shards import header_fingerprint, next_shard_index, write_shard
from maxdiffusion.null_adapter_verify import canonical_sigmas, verify_replay
from maxdiffusion.run_wan_null_inversion import batching_plan, guard_example_divergence


RECORD_ARM_ORDER = ("a1", "a2")
TABLES_NAME = "gate_tables.json"
REPORT_NAME = "run_report.json"
SELECTION_NAME = "selection.json"
ADEQUACY_NAME = "adequacy_report.json"
A3_MEASUREMENT_NAME = "a3_measurement.json"
DIRECT_OPT_NAME = "a3_direct_opt.json"
DIRECT_OPT_ARRAYS_NAME = "a3_nulls.npz"
# Plan §4-P1b: J1b is A3 on the first eight DEV examples, from the canonical eps_0, at production
# geometry. The boundary is enforced rather than assumed -- this job is separately approved, and an
# approval is for a specific experiment.
DIRECT_OPT_COHORT = "dev64"
DIRECT_OPT_EXAMPLES = 8
PRODUCTION_L_NULL = 16
PRODUCTION_CONTEXT_DIM = 4096
PRODUCTION_STEPS = 25
VERIFY_NAME = "verify_report.json"
VIDEO_FPS = 16
# Plan §4-P1's adequacy probe is a fixed experiment: the first eight DEV examples, the six approved
# cells. Both are preflighted, because a probe run on three arbitrary examples and an arbitrary grid
# answers a question nobody asked and costs the same (review, finding 3).
ADEQUACY_COHORT = "dev64"
# The §4-P1 selection rule is defined on DEV-64; a selection made anywhere else is not that rule.
SELECTION_COHORT = "dev64"
ADEQUACY_EXAMPLES = 8
# "If the projection exceeds +2 h, stop and surface instead of running" (plan §4-P1).
RERUN_BUDGET_SECONDS = 2 * 3600
# The re-run pays for both optimized arms, which is what the +2 h budget was sized against.
RERUN_ARMS = 2
# Which record arm a selection verdict deploys; STOP deploys nothing, and must not fall through to a
# default (R10 shipped ``arm="a1"`` as a keyword default, so ``global`` still cached A1/keyed).
TARGET_ARMS = {Target.A1_KEYED: "a1", Target.A2_GLOBAL: "a2", Target.STOP: None}


@dataclasses.dataclass(frozen=True)
class Backend:
    """Everything a mode needs from the model. Fakes satisfy this in tests; R10's glue supplies it."""

    velocity_fn: Callable[..., Any]
    decode_fn: Callable[[Any], Any]
    read_batch: Callable[[Sequence[str]], tuple[Any, Mapping[str, Mapping[str, Any]]]]
    base_context: Any
    model_revision: str


@dataclasses.dataclass(frozen=True)
class Sinks:
    """Everything a mode needs from the filesystem, injectable for the same reason."""

    write_shard: Callable[..., Any]
    write_json: Callable[[str, Any], str]
    save_video: Callable[..., str]
    read_shard: Callable[[str], tuple[Any, tuple[Any, ...]]]
    resume_plan: Callable[..., Any]
    validate_shard: Callable[..., Any]
    read_json: Callable[[str], Any]
    read_marker: Callable[[str], Any]
    write_arrays: Callable[..., str]


def header_for(plan: Mapping[str, Any], backend: Backend, *, manifest_hash: str, code_sha: str) -> ProvenanceHeader:
    """The one provenance header a run's shards share (R6's exact optimization_config contract)."""
    return ProvenanceHeader(
        manifest_hash=manifest_hash,
        code_sha=code_sha,
        model_revision=backend.model_revision,
        sigma_vector=canonical_sigmas(),
        guide_scale=float(plan["params"]["guide_scale"]),
        base_context_fingerprint=base_context_fingerprint(backend.base_context),
        optimization_config=dict(plan["optimization_config"]),
        dtype_policy=str(plan["latent_dtype"]),
        l_null=int(plan["params"]["l_null"]),
    )


def merge_tables(tables: Sequence[Mapping[str, Mapping]]) -> dict[str, dict]:
    """Per-batch gate tables into one cohort table; a name may be contributed by only one batch."""
    merged: dict[str, dict] = {method: {} for method in METHODS}  # noqa: C420 -- distinct inner dicts
    for table in tables:
        for method, rows in table.items():
            if method not in merged:
                raise ValueError(f"unknown method {method!r} in a metric table")
            for name, entry in rows.items():
                if name in merged[method]:
                    raise ValueError(f"{name!r} appears in two batches of the {method} table")
                merged[method][name] = entry
    return merged


def merge_latents(results: Sequence[Any]) -> dict[str, np.ndarray]:
    """Per-batch final latents into cohort arrays: probes stack on their batch axis, not their seed."""
    merged: dict[str, np.ndarray] = {}
    methods = sorted({method for result in results for method in result.final_latents})
    for method in methods:
        arrays = [np.asarray(result.final_latents[method]) for result in results]
        axis = 1 if arrays and arrays[0].ndim == 6 else 0
        merged[method] = np.concatenate(arrays, axis=axis)
    return merged


def _batch_results(backend: Backend, names: Sequence[str], params: CapacityParams, *, arms=None):
    """One batch of arms under the quarantine seam; survivors come back as one ``ArmResults``."""

    def run_fn(subset):
        batch, _ = backend.read_batch(subset)
        results = run_capacity_example_batch(backend.velocity_fn, batch, backend.base_context, params, arms=arms)
        # Every name maps to the batch's single ArmResults: the quarantine seam speaks in names, and
        # a survivor re-run returns a fresh ArmResults covering exactly the survivors.
        return dict.fromkeys(subset, results)

    # The divergence seam sits INSIDE the quarantine call: R6 raises a plain ValueError for a
    # non-finite trace, and only what comes out of the guard as ExampleDivergenceError is allowed to
    # become a recorded gap (R8's ratified policy).
    results, quarantined = quarantine_batch_failures(guard_example_divergence(run_fn), names)
    arm_results = next(iter(results.values())) if results else None
    return arm_results, quarantined


def _future_mse(latents: Any, z_video: Any) -> np.ndarray:
    """Per-example MSE over the non-pinned latent frames -- the plan's primary latent metric."""
    predicted = np.asarray(latents, np.float32)[..., 1:, :, :]
    target = np.asarray(z_video, np.float32)[..., 1:, :, :]
    return np.mean((predicted - target) ** 2, axis=(-4, -3, -2, -1))


def decode_cohort(
    decode_fn: Callable[[Any], Any],
    read_batch: Callable[[Sequence[str]], tuple[Any, Mapping]],
    latents_by_arm: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    batch_size: int,
) -> dict[str, dict]:
    """Decode and score the whole cohort, ``batch_size`` examples at a time.

    Plan §4-P1 asks for a full-cohort decode, which is a statement about *coverage*, not about
    issuing one VAE call with B=64: at production geometry that is 64x33 frames of decoder
    activations in one shot, and R10's first cut did exactly that regardless of ``null_batch_size``
    (review, finding 2). The cohort is decoded whole, in bounded pieces, and each piece re-reads only
    its own ground truth.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"decode batch size must be an integer >= 1, got {batch_size!r}")
    names = tuple(names)
    metrics: dict[str, dict] = {}
    for start in range(0, len(names), batch_size):
        chunk = names[start : start + batch_size]
        batch, _ = read_batch(chunk)
        sliced = {
            method: (value[:, start : start + len(chunk)] if value.ndim == 6 else value[start : start + len(chunk)])
            for method, value in latents_by_arm.items()
        }
        for method, rows in decode_and_score(decode_fn, sliced, np.asarray(batch.z_video), chunk).items():
            metrics.setdefault(method, {}).update(rows)
    return metrics


def evaluate_gates(filled: Mapping[str, Mapping], manifest: Sequence[str]) -> dict[str, Any]:
    """G1, G2 and the §4-P1 target-selection rule, over the **declared** cohort.

    The conventions are read off the tables rather than off config: the single-noise arms carry the
    one seed key ``"0"`` and reduce under ``GLOBAL``; the probes carry ``{0,1,2}`` and reduce under
    ``KEYED`` (R6's emission contract). Passing config's deployment convention here would evaluate
    A1-vs-A0 at three seeds two of the arms never ran.
    """
    g1 = gate_g1(filled["a1"], filled["a0"], manifest, NoiseConvention.GLOBAL)
    g2 = gate_g2(filled["a2"], filled["a2_0"], manifest, NoiseConvention.GLOBAL)
    selection = select_target(g1, filled["a1_probe"], manifest, NoiseConvention.KEYED, g2)
    return {"g1": g1, "g2": g2, "selection": selection}


def selection_payload(
    verdicts: Mapping[str, Any], plan: Mapping[str, Any], *, manifest_hash: str = ""
) -> dict[str, Any]:
    """The J1 artifact P2 and the verifier are required to read their arm from.

    It carries its own provenance -- which cohort chose the arm, which manifest that cohort was cut
    from, and whether the run that produced it was a smoke -- because downstream this file *is* the
    authority to cache two thousand examples (follow-up review, finding 2).
    """
    selection = verdicts["selection"]
    arm = TARGET_ARMS[selection.target]
    label, convention = RECORD_ARMS[arm] if arm else (None, None)
    return {
        "cohort": plan["cohort"],
        "manifest_hash": manifest_hash,
        "smoke_examples": int(plan.get("smoke_examples", 0)),
        "manifest": list(plan["names"]),
        "target": selection.target.value,
        "arm": arm,
        "label": label,
        "noise_convention": convention,
        "reasons": list(selection.reasons),
        "gates": json.loads(verdicts_to_json(dict(verdicts))),
    }


def selected_arm(
    selection: Mapping[str, Any], *, expected_manifest_hash: str | None = None, allow_smoke: bool = False
) -> str:
    """The record arm a J1 selection deployed, or a refusal -- never a default.

    P2 and the verifier both used to take ``arm="a1"`` as a keyword default that ``main`` never
    overrode, so a run selecting A2/global cached and verified A1/keyed and said so in its own
    provenance (review, findings 4 and 6). There is no default here.

    The artifact must also be *this* experiment's. An unbound selection is a file that says "A1", and
    a two-example smoke, a run against a rebuilt manifest and a copy from another experiment all say
    it just as convincingly -- which is enough to authorize caching every window in TRAIN-2000
    (follow-up review, finding 2). So the selection has to have been made on the DEV cohort, against
    the manifest this job loaded, by a run that was not a smoke.
    """
    if not isinstance(selection, Mapping):
        raise ValueError(f"the J1 selection artifact must be a JSON object, got {type(selection).__name__}")
    target, arm = selection.get("target"), selection.get("arm")
    if target == Target.STOP.value or arm is None:
        raise ValueError(
            f"the J1 selection stopped after P1 ({'; '.join(selection.get('reasons', [])) or 'no reason recorded'}): "
            f"there is no selected arm to cache or verify"
        )
    if arm not in RECORD_ARMS or RECORD_ARMS[arm][0] != selection.get("label"):
        raise ValueError(f"the J1 selection artifact names an unusable arm: {arm!r}/{selection.get('label')!r}")
    if target != f"{RECORD_ARMS[arm][0]}/{RECORD_ARMS[arm][1]}":
        raise ValueError(f"the J1 selection artifact's target {target!r} disagrees with its arm {arm!r}")

    if selection.get("cohort") != SELECTION_COHORT:
        raise ValueError(
            f"the J1 selection was made on {selection.get('cohort')!r}, but the §4-P1 rule is defined on "
            f"{SELECTION_COHORT!r}: this artifact does not authorize a cache"
        )
    digest = selection.get("manifest_hash")
    if not isinstance(digest, str) or not digest:
        raise ValueError(
            "the J1 selection artifact carries no manifest binding: an unbound selection cannot be shown "
            "to describe the cohort this job loaded"
        )
    if expected_manifest_hash is not None and digest != expected_manifest_hash:
        raise ValueError(
            f"the J1 selection was made against a different {SELECTION_COHORT} manifest "
            f"({digest[:12]}... vs {expected_manifest_hash[:12]}...): it does not authorize this job"
        )
    if not allow_smoke and int(selection.get("smoke_examples") or 0):
        raise ValueError(
            f"the J1 selection came from a {selection['smoke_examples']}-example smoke run: a smoke chooses "
            f"an arm on a cohort too small to have gated anything, and cannot authorize a cache"
        )
    return str(arm)


def _load_selection(sinks: Sinks, artifact_dir: str, selection_uri: str | None) -> dict[str, Any]:
    uri = selection_uri or posixpath.join(artifact_dir, SELECTION_NAME)
    payload = sinks.read_json(uri)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{uri} is not a J1 selection artifact")
    return dict(payload)


def run_capacity(
    plan: Mapping[str, Any],
    backend: Backend,
    sinks: Sinks,
    *,
    artifact_dir: str,
    staging_dir: str,
    manifest_hash: str,
    code_sha: str,
    decode_batch_size: int = 8,
    adopted_recipe: Mapping[str, Any] | None = None,
    a3_measure: bool = False,
    measure: Callable[..., Any] = measure_single_update,
) -> dict[str, Any]:
    """Plan §4-P1: every arm over the cohort, decoded, gated, selected, and only then recorded."""
    started = time.time()
    plan = apply_adopted_recipe(plan, adopted_recipe) if adopted_recipe else plan
    params = CapacityParams(**plan["params"])
    header = header_for(plan, backend, manifest_hash=manifest_hash, code_sha=code_sha)

    # --- arms. Nothing here is published: a failure past this point must leave nothing immutable.
    batches: list[tuple[Any, dict[str, str]]] = []
    names: list[str] = []
    quarantined: dict[str, str] = {}
    for batch_names in plan["batches"]:
        arm_results, lost = _batch_results(backend, batch_names, params)
        quarantined.update(lost)
        if arm_results is None:
            continue
        batches.append((arm_results, lost))
        names.extend(arm_results.names)
    if not names:
        raise RuntimeError("every batch was quarantined: there is nothing to gate")

    # --- decode the declared cohort in bounded batches, then fill: never gate before the fill.
    latents = merge_latents([result for result, _ in batches])
    pixels = decode_cohort(backend.decode_fn, backend.read_batch, latents, names, batch_size=decode_batch_size)
    filled = fill_pixel_metrics(merge_tables([emit_metric_tables(result) for result, _ in batches]), pixels)

    # --- plan §4-P1 item (iii): one A3 update, measured inside J1. It is what decides whether the
    # separately-approved J1b is proposed at all, so it runs on real data from this very cohort.
    a3_report = None
    if a3_measure:
        probe_names = tuple(names[:1])
        z_start, z_i0, z_video, null_init = a3_inputs(backend, plan, probe_names)
        a3_report = measurement_payload(
            measure(
                backend.velocity_fn,
                z_start,
                z_i0,
                z_video,
                canonical_sigmas(),
                null_init,
                backend.base_context,
                lr=float(plan["params"]["lr"]),
                guide_scale=float(plan["params"]["guide_scale"]),
            ),
            plan,
            header,
            probe_names,
        )

    # --- gates and selection, against the cohort the run declared rather than the one it survived.
    verdicts = evaluate_gates(filled, plan["names"])
    selection = selection_payload(verdicts, plan, manifest_hash=manifest_hash)

    # --- publication.
    sinks.write_json(posixpath.join(artifact_dir, TABLES_NAME), filled)
    sinks.write_json(posixpath.join(artifact_dir, SELECTION_NAME), selection)
    if a3_report is not None:
        sinks.write_json(posixpath.join(artifact_dir, A3_MEASUREMENT_NAME), a3_report)
    shards = []
    for index, (arm_results, lost) in enumerate(batches):
        batch, fields = backend.read_batch(arm_results.names)
        for arm in RECORD_ARM_ORDER:
            records = build_capacity_records(
                backend.velocity_fn, arm_results, batch, backend.base_context, header, fields, arm=arm
            )
            shard = posixpath.join(artifact_dir, RECORD_ARMS[arm][0].lower(), f"shard_{index:05d}")
            sinks.write_shard(records, header, shard, staging_dir, quarantined=lost)
            shards.append(shard)

    videos = {}
    for index, name in enumerate(plan["decode_subset"]):
        if name not in names:
            continue
        position = names.index(name)
        one, _ = backend.read_batch((name,))
        stacked = comparison_video_frames(
            np.asarray(backend.decode_fn(np.asarray(one.z_video)))[0],
            np.asarray(backend.decode_fn(latents["a1"][position : position + 1]))[0],
        )
        videos[name] = sinks.save_video(stacked, posixpath.join(artifact_dir, "videos", f"{index:02d}.mp4"), VIDEO_FPS)

    report = {
        "mode": "capacity",
        "cohort": plan["cohort"],
        "declared": len(plan["names"]),
        "examples": len(names),
        "smoke_examples": plan.get("smoke_examples", 0),
        "quarantined": quarantined,
        "recipe": dict(plan["optimization_config"]),
        "target": selection["target"],
        "a3_measurement": None if a3_report is None else {
            "verdict": a3_report["verdict"],
            "projection_hours": a3_report["projection_hours"],
            "fits_budget": a3_report["fits_budget"],
            "preliminary": a3_report["preliminary"],
        },
        "gates": {name: verdicts[name].reasons for name in ("g1", "g2")},
        "shards": shards,
        "videos": sorted(videos),
        "tables": sorted(filled),
        "seconds": round(time.time() - started, 3),
    }
    sinks.write_json(posixpath.join(artifact_dir, REPORT_NAME), report)
    return report


def rerun_projection_seconds(adopted: Mapping[str, Any], cohort_examples: int) -> float:
    """What re-running the arms at the adopted recipe would cost, from the probe's own measurement."""
    per_example = float(adopted.get("projection_seconds_per_example", 0.0))
    if not np.isfinite(per_example) or per_example < 0.0:
        raise ValueError(f"the adequacy artifact's projection_seconds_per_example is unusable: {per_example!r}")
    return per_example * int(cohort_examples) * RERUN_ARMS


def apply_adopted_recipe(plan: Mapping[str, Any], adopted: Mapping[str, Any]) -> dict[str, Any]:
    """§4-P1's re-run seam: gate on the adopted recipe, or stop before spending the time.

    "If a recipe is adopted, re-run A1/A2 on the full DEV cohort under it **before** gating" -- so
    the adopted recipe has to reach the arms *and* the header, or ``build_capacity_records`` refuses
    the records for advertising a recipe they were not produced at. "If the projection exceeds +2 h,
    stop and surface instead of running" is the other half, and it is a raise rather than a warning
    because the alternative is discovering the overrun by watching a TPU bill.
    """
    if not adopted.get("adopted"):
        return dict(plan)
    projected = rerun_projection_seconds(adopted, len(plan["names"]))
    if projected > RERUN_BUDGET_SECONDS:
        raise RuntimeError(
            f"re-running the cohort at the adopted recipe projects {projected / 3600:.2f} h, over the "
            f"{RERUN_BUDGET_SECONDS / 3600:.0f} h budget: surface the adequacy result instead of running it"
        )
    recipe = {"inner_iters": int(adopted["inner_iters"]), "lr": float(adopted["lr"])}
    updated = dict(plan)
    updated["params"] = {**plan["params"], **recipe}
    updated["optimization_config"] = dict(recipe)  # the header must describe what actually ran
    updated["adopted_recipe"] = {**recipe, "projected_seconds": projected}
    return updated


def natural_context_length(base_context: Any) -> int:
    """L_nat: how many rows of the padded T5("") context are not padding (plan §4-P1's ablation).

    The encoder writes zeros past the attention mask, so the last non-zero row is the natural length.
    This is a **diagnostic-only** number: L_null stays 16 for P2/P3 whatever it says (plan §4-P1,
    N5), and revisiting that is a Yixun decision that reopens the plan.
    """
    rows = np.asarray(base_context, np.float32)
    if rows.ndim != 2:
        raise ValueError(f"base_context must be [S, D], got shape {rows.shape}")
    occupied = np.flatnonzero(np.any(rows != 0.0, axis=1))
    if occupied.size == 0:
        raise ValueError('the T5("") context is entirely zero: there is no natural length to ablate')
    return int(occupied[-1]) + 1


def _preflight_adequacy(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Everything wrong with a probe request, decided before a single example is read."""
    if plan["cohort"] != ADEQUACY_COHORT:
        raise ValueError(f"the adequacy probe is defined on {ADEQUACY_COHORT}, not {plan['cohort']!r}")
    if len(plan["names"]) < ADEQUACY_EXAMPLES:
        raise ValueError(
            f"the adequacy probe needs the first {ADEQUACY_EXAMPLES} DEV names, but the cohort carries "
            f"{len(plan['names'])}"
        )
    grid = tuple(tuple(cell) for cell in plan["grid"])
    if set(grid) != set(ADEQUACY_GRID):
        raise ValueError(
            f"the adequacy grid is the approved {sorted(ADEQUACY_GRID)}, got {sorted(grid)}; a probe on "
            f"another grid answers a different question at the same price"
        )
    if tuple(DEFAULT_RECIPE) not in set(grid):
        raise ValueError(f"the adequacy grid must contain the default recipe {DEFAULT_RECIPE}, got {sorted(grid)}")
    return tuple(plan["names"][:ADEQUACY_EXAMPLES])


def _score_payload(score: Any, seconds: float | None = None) -> dict[str, Any]:
    """One grid cell, with the evidence it produced -- not just the number it reduced to."""
    payload = {
        "inner_iters": score.inner_iters,
        "lr": score.lr,
        "score": score.score,
        "per_example": list(score.per_example),
        "tracking_losses": np.asarray(score.tracking_losses).tolist(),
        "grad_norms": np.asarray(score.grad_norms).tolist(),
        "final_losses": np.asarray(score.final_losses).tolist(),
    }
    if seconds is not None:
        payload["seconds"] = round(seconds, 3)
    return payload


def run_adequacy(plan: Mapping[str, Any], backend: Backend, sinks: Sinks, *, artifact_dir: str) -> dict[str, Any]:
    """Plan §4-P1's adequacy probe: the fixed first-eight DEV examples on the approved six-cell grid.

    The probe exists to produce evidence, so it persists the evidence: per-example scores, the
    optimizer's ``[N, J, B]`` tracking-loss and grad-norm traces, the ``[B, N]`` post-inner-loop
    final losses, the adoption numbers, and the per-cell wall time capacity's re-run projection is
    computed from. R10's first cut kept only the medians, which is the one part of an adequacy probe
    that cannot be recomputed later (review, finding 3).
    """
    started = time.time()
    names = _preflight_adequacy(plan)
    batch, _ = backend.read_batch(names)
    guide_scale, l_null = float(plan["params"]["guide_scale"]), int(plan["params"]["l_null"])

    scores, timings = [], []
    for cell in plan["grid"]:
        cell_started = time.time()
        report = run_adequacy_probe(
            backend.velocity_fn, batch, backend.base_context, (tuple(cell),), guide_scale=guide_scale, l_null=l_null
        )
        timings.append(time.time() - cell_started)
        scores.extend(report.scores)
    report = AdequacyReport(names, tuple(scores))
    adoption = adopt_recipe(report)

    per_cell = {(score.inner_iters, score.lr): seconds for score, seconds in zip(scores, timings)}
    adopted_seconds = per_cell[(adoption.inner_iters, adoption.lr)]
    payload = {
        "mode": "adequacy_probe",
        "cohort": plan["cohort"],
        "names": list(names),
        "grid": [list(cell) for cell in plan["grid"]],
        "guide_scale": guide_scale,
        "l_null": l_null,
        "scores": [_score_payload(score, seconds) for score, seconds in zip(scores, timings)],
        "adopted": {
            "inner_iters": adoption.inner_iters,
            "lr": adoption.lr,
            "adopted": adoption.adopted,
            "projection_seconds_per_example": adopted_seconds / len(names),
        },
        "plateau": adoption.plateau,
        "reasons": list(adoption.reasons),
        "numbers": adoption.numbers,
        "l_null_ablation": _l_null_ablation(backend, batch, guide_scale, l_null),
        "seconds": round(time.time() - started, 3),
    }
    sinks.write_json(posixpath.join(artifact_dir, ADEQUACY_NAME), payload)
    return payload


def _l_null_ablation(backend: Backend, batch: Any, guide_scale: float, l_null: int) -> dict[str, Any]:
    """Plan N5's diagnostic: the default recipe at L=16 beside the same recipe at L_nat."""

    def score_at(value: int) -> float:
        probe = run_adequacy_probe(
            backend.velocity_fn, batch, backend.base_context, (DEFAULT_RECIPE,), guide_scale=guide_scale, l_null=value
        )
        return probe.scores[0].score

    l_nat = natural_context_length(backend.base_context)
    at_l_null = score_at(l_null)
    # When the padded context happens to be exactly L_null rows there is no second cell to run, and
    # paying for an identical one would only make the ablation look like it said something.
    return {
        "diagnostic_only": True,
        "recipe": list(DEFAULT_RECIPE),
        "l_null": {"l": l_null, "score": at_l_null},
        "l_nat": {"l": l_nat, "score": at_l_null if l_nat == l_null else score_at(l_nat)},
    }


def measure_fidelity(
    backend: Backend,
    names: Sequence[str],
    params: CapacityParams,
    header: ProvenanceHeader,
    *,
    arm: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Plan §4-P2's fp16 question, measured: the same replay scored in memory and after serialization.

    Returns the two tables ``fidelity_gate`` compares -- ``name -> {future_ssim, future_mse}`` -- the
    first from the fp32 arrays the arm produced, the second from the bytes an fp16 record actually
    stores, round-tripped through the R4b codec so the numbers describe what a reader would read.
    """
    names = tuple(names)
    batch, fields = backend.read_batch(names)
    results = run_capacity_example_batch(backend.velocity_fn, batch, backend.base_context, params, arms=(arm,))
    z_video = np.asarray(batch.z_video, np.float32)

    exact = np.asarray(results.final_latents[arm], np.float32)
    stored_header = dataclasses.replace(header, dtype_policy="fp16")
    records = build_capacity_records(
        backend.velocity_fn, results, batch, backend.base_context, stored_header, fields, arm=arm
    )
    stored = np.stack(
        [
            np.asarray(record_from_bytes(record_to_bytes(record)).expected_final_latent, np.float32)
            for record in records
        ]
    )

    scored = decode_and_score(backend.decode_fn, {arm: exact}, z_video, names)[arm]
    scored_stored = decode_and_score(backend.decode_fn, {arm: stored}, z_video, names)[arm]
    exact_mse, stored_mse = _future_mse(exact, z_video), _future_mse(stored, z_video)
    fp32 = {
        name: {"future_ssim": scored[name]["0"]["future_ssim"], "future_mse": float(exact_mse[index])}
        for index, name in enumerate(names)
    }
    fp16 = {
        name: {"future_ssim": scored_stored[name]["0"]["future_ssim"], "future_mse": float(stored_mse[index])}
        for index, name in enumerate(names)
    }
    return fp32, fp16


def a3_inputs(backend: Backend, plan: Mapping[str, Any], names: Sequence[str]):
    """The canonical A3 problem for ``names``: eps_0 as the start, the batch's own condition/target.

    ``z_start`` is ``global_noise(0)`` broadcast over the batch -- plan §3's single canonical noise,
    which is what A3 optimizes from and what A2's deployment convention deploys from. It is built
    here rather than read from anywhere so the two arms cannot drift apart.
    """
    names = tuple(names)
    batch, _ = backend.read_batch(names)
    l_null = int(plan["params"]["l_null"])
    z_start = np.broadcast_to(np.asarray(global_noise(0), np.float32), (len(names), *PRODUCTION_GEOMETRY.z_video))
    return (
        np.asarray(z_start),
        np.asarray(batch.z_i0, np.float32),
        np.asarray(batch.z_video, np.float32),
        np.asarray(backend.base_context, np.float32)[:l_null],
    )


def measurement_payload(report: Any, plan: Mapping[str, Any], header: ProvenanceHeader, names) -> dict[str, Any]:
    """One A3 measurement, bound to the run that produced it."""
    return {
        "mode": "a3_measurement",
        "names": list(names),
        "verdict": report.verdict,
        "reasons": list(report.reasons),
        "lower_seconds": report.lower_seconds,
        "compile_seconds": report.compile_seconds,
        "step_seconds": report.step_seconds,
        "setup_seconds": report.setup_seconds,
        "peak_hbm_bytes": report.peak_hbm_bytes,
        "current_hbm_bytes": report.current_hbm_bytes,
        "device_memory": [dict(entry) for entry in report.device_memory],
        "loss": report.loss,
        "grad_norm": report.grad_norm,
        "batch": report.batch,
        "iters": report.iters,
        "job_batch": report.job_batch,
        "compute_seconds": report.compute_seconds,
        "write_allowance_seconds": report.write_allowance_seconds,
        "projection_seconds": report.projection_seconds,
        "projection_hours": report.projection_hours,
        "fits_budget": report.fits_budget,
        "preliminary": report.preliminary,
        "budgets": dict(report.budgets),
        "provenance": {
            "cohort": plan["cohort"],
            "manifest_hash": header.manifest_hash,
            "code_sha": header.code_sha,
            "model_revision": header.model_revision,
            "base_context_fingerprint": header.base_context_fingerprint,
            "guide_scale": float(header.guide_scale),
            "l_null": int(header.l_null),
        },
    }


def _preflight_direct_opt(plan: Mapping[str, Any], backend: Backend) -> tuple[str, ...]:
    """J1b's production boundary, checked before a separately-approved job spends anything."""
    if plan["cohort"] != DIRECT_OPT_COHORT:
        raise ValueError(f"A3 is defined on {DIRECT_OPT_COHORT}, not {plan['cohort']!r}")
    if len(plan["names"]) < DIRECT_OPT_EXAMPLES:
        raise ValueError(
            f"A3 needs the first {DIRECT_OPT_EXAMPLES} DEV names, but the cohort carries {len(plan['names'])}"
        )
    l_null = int(plan["params"]["l_null"])
    if l_null != PRODUCTION_L_NULL:
        raise ValueError(f"A3 runs at the pinned L_null={PRODUCTION_L_NULL}, got {l_null}")
    context = np.asarray(backend.base_context)
    if context.ndim != 2 or context.shape[1] != PRODUCTION_CONTEXT_DIM:
        raise ValueError(f"A3 expects a [S, {PRODUCTION_CONTEXT_DIM}] context, got shape {context.shape}")
    if len(canonical_sigmas()) - 1 != PRODUCTION_STEPS:
        raise ValueError(f"A3 runs on the canonical {PRODUCTION_STEPS}-step grid")
    return tuple(plan["names"][:DIRECT_OPT_EXAMPLES])


def run_direct_opt(
    plan: Mapping[str, Any],
    backend: Backend,
    sinks: Sinks,
    *,
    artifact_dir: str,
    manifest_hash: str,
    code_sha: str,
    iters: int = A3_ITERS,
    measure: Callable[..., Any] = measure_single_update,
) -> dict[str, Any]:
    """Plan §4-P1b: the separately-approved J1b job -- A3 over the first eight DEV examples.

    **The fit probe comes first.** J1's measurement is B=1, which the plan and the R11 review both
    treat as a preliminary compute estimate: B=8 has a different compile, execution, sharding and HBM
    profile. So this job opens by measuring one update at its own batch size and continues to the
    300 iterations only if that projection fits. A job that does not fit stops, says so, and exits
    non-zero rather than spending four hours discovering it.
    """
    started = time.time()
    names = _preflight_direct_opt(plan, backend)
    header = header_for(plan, backend, manifest_hash=manifest_hash, code_sha=code_sha)
    # Wall-clocked, because it is real J1b time the projection must carry: eps_0 construction plus
    # staging eight examples' worth of latents off the manifest's shards.
    setup_started = time.time()
    z_start, z_i0, z_video, null_init = a3_inputs(backend, plan, names)
    setup_seconds = time.time() - setup_started
    sigmas = canonical_sigmas()
    recipe = {"lr": float(plan["params"]["lr"]), "guide_scale": float(plan["params"]["guide_scale"])}

    probe = measure(
        backend.velocity_fn,
        z_start,
        z_i0,
        z_video,
        sigmas,
        null_init,
        backend.base_context,
        iters=int(iters),
        # The batch the job runs at -- not a multiplier. One joint update covers all eight examples,
        # so 300 iterations is 300 updates.
        job_batch=len(names),
        setup_seconds=setup_seconds,
        require_single_example=False,
        **recipe,
    )
    payload = {
        "mode": "direct_opt",
        "cohort": plan["cohort"],
        "names": list(names),
        "iters": int(iters),
        "fit_probe": measurement_payload(probe, plan, header, names),
        "continued": bool(probe.fits_budget),
        "seconds": round(time.time() - started, 3),
    }
    if not probe.fits_budget:
        payload["reasons"] = [
            f"the B={len(names)} fit probe projects {probe.projection_hours:.2f} h "
            f"({probe.verdict}); J1b does not run"
        ]
        sinks.write_json(posixpath.join(artifact_dir, DIRECT_OPT_NAME), payload)
        return payload

    nulls, losses, grad_norms = direct_optimize_nulls(
        backend.velocity_fn, z_start, z_i0, z_video, sigmas, null_init, backend.base_context,
        iters=int(iters), **recipe,
    )
    # The post-update endpoint: ``losses`` records the loss BEFORE each update, so the value after
    # the last one is not in it -- and that is the number A3 is judged on.
    final_endpoint = endpoint_future_mse(
        direct_rollout(
            backend.velocity_fn, nulls, z_start, z_i0, sigmas, backend.base_context,
            guide_scale=recipe["guide_scale"],
        ),
        z_video,
    )
    arrays = sinks.write_arrays(
        posixpath.join(artifact_dir, DIRECT_OPT_ARRAYS_NAME),
        nulls=np.asarray(nulls, np.float32),
        losses=np.asarray(losses, np.float32),
        grad_norms=np.asarray(grad_norms, np.float32),
        final_endpoint=np.asarray(final_endpoint, np.float32),
    )
    payload.update(
        arrays=arrays,
        initial_loss=[float(value) for value in np.asarray(losses[0])],
        final_loss=[float(value) for value in np.asarray(losses[-1])],
        final_endpoint=[float(value) for value in np.asarray(final_endpoint)],
        grad_norm_first=[float(value) for value in np.asarray(grad_norms[0])],
        grad_norm_last=[float(value) for value in np.asarray(grad_norms[-1])],
        provenance=measurement_payload(probe, plan, header, names)["provenance"],
        seconds=round(time.time() - started, 3),
    )
    sinks.write_json(posixpath.join(artifact_dir, DIRECT_OPT_NAME), payload)
    return payload


def run_cache(
    plan: Mapping[str, Any],
    backend: Backend,
    sinks: Sinks,
    *,
    artifact_dir: str,
    staging_dir: str,
    manifest_hash: str,
    code_sha: str,
    dev_manifest: Sequence[str],
    selection_uri: str | None = None,
    selection_digest: str | None = None,
    existing_shards: Sequence[str] = (),
    fidelity: Callable[..., tuple[Mapping, Mapping]] = measure_fidelity,
) -> dict[str, Any]:
    """Plan §4-P2: cache the arm J1 selected, at the dtype its own fidelity gate decides.

    The order is the plan's, and each step is a precondition of the next: read the selection, decide
    the dtype on the first eight DEV examples, build the header from that verdict, and only then
    resume and write. R10's first cut had the arm defaulting to A1, the fidelity metrics arriving
    from a caller that never sent them, and the verdict evaluated *after* the records it should have
    described were already published (review, finding 4).
    """
    started = time.time()
    # allow_smoke stays False: a smoke selection may not authorize a cache.
    arm = selected_arm(
        _load_selection(sinks, artifact_dir, selection_uri), expected_manifest_hash=selection_digest
    )
    label, convention = RECORD_ARMS[arm]
    params = CapacityParams(**plan["params"])

    # The fidelity gate runs before any cohort caching, and its verdict is what the header declares.
    probe_names = tuple(dev_manifest)[:FIDELITY_SUBSET_SIZE]
    fp32_metrics, fp16_metrics = fidelity(
        backend,
        probe_names,
        params,
        header_for(plan, backend, manifest_hash=manifest_hash, code_sha=code_sha),
        arm=arm,
    )
    verdict = fidelity_gate(dev_manifest, fp32_metrics, fp16_metrics)
    header = header_for(
        {**plan, "latent_dtype": verdict.latent_dtype}, backend, manifest_hash=manifest_hash, code_sha=code_sha
    )

    resume = sinks.resume_plan(
        plan["names"],
        list(existing_shards),
        expected_header_fingerprint=header_fingerprint(header),
        expected_arm=label,
        expected_noise_convention=convention,
    )
    todo = tuple(resume.todo)
    quarantined, shards, written = dict(resume.quarantined), [], []
    first_index = next_shard_index(existing_shards)
    for index, batch_names in enumerate(batching_plan(todo, len(plan["batches"][0])) if todo else ()):
        arm_results, lost = _batch_results(backend, batch_names, params, arms=(arm,))
        quarantined.update(lost)
        if arm_results is None:
            continue
        batch, fields = backend.read_batch(arm_results.names)
        records = build_capacity_records(
            backend.velocity_fn, arm_results, batch, backend.base_context, header, fields, arm=arm
        )
        shard = posixpath.join(artifact_dir, f"shard_{first_index + index:05d}")
        sinks.write_shard(records, header, shard, staging_dir, quarantined=lost)
        shards.append(shard)
        written.extend(arm_results.names)
    # A retry that succeeded this attempt is no longer a gap, whichever attempt first lost it.
    superseded = sorted(set(quarantined) & set(written))
    for name in superseded:
        del quarantined[name]

    payload = {
        "mode": "cache",
        "arm": label,
        "noise_convention": convention,
        "latent_dtype": verdict.latent_dtype,
        "fidelity": {
            "passed": verdict.passed,
            "latent_dtype": verdict.latent_dtype,
            "subset": list(verdict.subset),
            "worst_ssim_drop": verdict.worst_ssim_drop,
            "worst_mse_increase": verdict.worst_mse_increase,
            "reasons": list(verdict.reasons),
        },
        "already_covered": len(resume.covered),
        "written": len(written),
        "todo": list(todo),
        "quarantined": quarantined,
        "superseded": sorted({*resume.superseded, *superseded}),
        "shards": shards,
        "seconds": round(time.time() - started, 3),
    }
    sinks.write_json(posixpath.join(artifact_dir, REPORT_NAME), payload)
    return payload


def run_verify(
    plan: Mapping[str, Any],
    backend: Backend,
    sinks: Sinks,
    *,
    artifact_dir: str,
    shard_paths: Sequence[str],
    atol: float,
    selection_uri: str | None = None,
    selection_digest: str | None = None,
) -> dict[str, Any]:
    """Plan §5: certify the cache, or say exactly why it cannot be certified.

    "Nothing to check" is not a pass. R10's first cut returned exit 0 for an empty shard list, read
    records straight out of a directory without validating the shard around them, let a duplicate
    name overwrite an earlier verdict, and never asked whether the cohort was covered -- so a cache
    missing half its examples, or carrying a replaced record beside a stale marker, verified clean
    (review, finding 6). Everything here is a failure that produces exit 1: an unvalidated shard, a
    missing name, a duplicate, a recorded quarantine, and of course a replay that does not reproduce.
    """
    started = time.time()
    # A smoke cache is still a cache, and verification's own coverage check is against its own plan,
    # so a smoke-flagged selection may name the arm here even though it may not authorize caching.
    arm = selected_arm(
        _load_selection(sinks, artifact_dir, selection_uri),
        expected_manifest_hash=selection_digest,
        allow_smoke=True,
    )
    label, convention = RECORD_ARMS[arm]
    failures: list[str] = []
    verdicts: dict[str, str] = {}
    shard_paths = tuple(shard_paths)

    if not shard_paths:
        failures.append("no published shards: there is nothing to verify")
    reports = []
    if shard_paths:
        # The expectation has to start somewhere: a verify run is a separate job from the cache run
        # and does not carry that run's header. It is taken from the first shard and then required of
        # every shard, which is not circular -- ``validate_shard`` independently requires each shard's
        # own ``header.json`` to hash to its own marker's fingerprint, so a doctored header fails
        # against its own marker before it can become anybody's expectation.
        # The MARKER, not the shard: reading records out of a shard that has not been validated is
        # exactly the boundary R8 drew, and bootstrapping the expectation from ``read_shard`` crossed
        # it for shard 0 (follow-up review, finding 4). ``marker_from_json`` is strict, and the marker
        # names no record bytes.
        try:
            expected = sinks.read_marker(shard_paths[0]).header_fingerprint
        except Exception as error:  # noqa: BLE001 -- an unreadable marker is a verdict, not a crash
            failures.append(f"shard {shard_paths[0]}: marker unreadable ({type(error).__name__}: {error})")
            expected = None
        for path in shard_paths if expected is not None else ():
            report = sinks.validate_shard(
                path,
                expected_header_fingerprint=expected,
                expected_arm=label,
                expected_noise_convention=convention,
            )
            reports.append(report)
            if not report.valid:
                failures.append(f"shard {path}: {'; '.join(report.reasons) or 'invalid'}")

    seen: dict[str, str] = {}
    for report in reports:
        if not report.valid:
            continue
        for name, reason in report.quarantined.items():
            failures.append(f"quarantined {name}: {reason}")
        header, records = sinks.read_shard(report.path)
        for record in records:
            if record.name in seen:
                failures.append(f"duplicate {record.name}: published by {seen[record.name]} and {report.path}")
                continue
            seen[record.name] = report.path
            try:
                verify_replay(
                    record,
                    header,
                    backend.velocity_fn,
                    backend.base_context,
                    expected_model_revision=backend.model_revision,
                    expected_guide_scale=float(plan["params"]["guide_scale"]),
                    expected_noise_convention=convention,
                    expected_arm=label,
                    atol=float(atol),
                )
            except Exception as error:  # noqa: BLE001 -- a verifier failure is a result, not a crash
                verdicts[record.name] = f"{type(error).__name__}: {error}"
                failures.append(record.name)
            else:
                verdicts[record.name] = "ok"

    missing = [name for name in plan["names"] if name not in seen]
    strangers = sorted(set(seen) - set(plan["names"]))
    failures.extend(f"missing {name}" for name in missing)
    failures.extend(f"not in the cohort: {name}" for name in strangers)

    payload = {
        "mode": "verify_replay",
        "arm": label,
        "cohort": plan["cohort"],
        "declared": len(plan["names"]),
        "records": len(verdicts),
        "shards": list(shard_paths),
        "invalid_shards": [report.path for report in reports if not report.valid],
        "missing": missing,
        "failures": sorted(failures),
        "verdicts": verdicts,
        "seconds": round(time.time() - started, 3),
    }
    sinks.write_json(posixpath.join(artifact_dir, VERIFY_NAME), payload)
    return payload


def execute(
    mode: str, plan: Mapping[str, Any], backend: Backend, sinks: Sinks, **kwargs
) -> tuple[dict[str, Any], int]:
    """Dispatch one mode and return ``(report, exit_code)``."""
    if mode == "capacity":
        return run_capacity(plan, backend, sinks, **kwargs), 0
    if mode == "adequacy_probe":
        return run_adequacy(plan, backend, sinks, artifact_dir=kwargs["artifact_dir"]), 0
    if mode == "cache":
        return run_cache(plan, backend, sinks, **kwargs), 0
    if mode == "direct_opt":
        report = run_direct_opt(plan, backend, sinks, **kwargs)
        # A fit probe that refuses is a stop, not a result to be scrolled past.
        return report, (0 if report["continued"] else 1)
    if mode == "verify_replay":
        report = run_verify(plan, backend, sinks, **kwargs)
        return report, (1 if report["failures"] else 0)
    raise ValueError(f"unknown mode {mode!r}")


def _gfile():
    from tensorflow.io import gfile

    return gfile


def write_json(path: str, payload: Any) -> str:
    """Strict, sorted JSON -- the gate tables have to survive ``null_adapter_gates.parse_table``."""
    gfile = _gfile()
    gfile.makedirs(posixpath.dirname(path))
    with gfile.GFile(path, "w") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False, indent=2))
    return path


def read_json(path: str) -> Any:
    """Read one published JSON artifact -- the J1 selection, in practice."""
    gfile = _gfile()
    if not gfile.exists(path):
        raise FileNotFoundError(f"{path} does not exist: this job requires the J1 selection artifact")
    with gfile.GFile(path, "r") as handle:
        return json.loads(handle.read())


def write_arrays(path: str, **arrays: Any) -> str:
    """Publish A3's tensors as a single npz, staged through the same transactional upload as videos."""
    import io

    buffer = io.BytesIO()
    np.savez(buffer, **{name: np.asarray(value) for name, value in arrays.items()})
    gfile = _gfile()
    gfile.makedirs(posixpath.dirname(path))
    staged = f"{path}.partial"
    if gfile.exists(staged):
        gfile.remove(staged)
    with gfile.GFile(staged, "wb") as handle:
        handle.write(buffer.getvalue())
    gfile.rename(staged, path, overwrite=True)
    return path


def read_marker(shard_path: str) -> Any:
    """A shard's completion marker alone -- which run wrote it, without touching a single record."""
    from maxdiffusion.null_adapter_shards import MARKER_NAME, marker_from_json

    with _gfile().GFile(posixpath.join(shard_path, MARKER_NAME), "r") as handle:
        return marker_from_json(handle.read())


def read_shard(shard_path: str) -> tuple[Any, tuple[Any, ...]]:
    """The published pair a verifier consumes: the shard's header and every record it names."""
    from maxdiffusion.null_adapter_records import header_from_json

    from maxdiffusion.null_adapter_shards import HEADER_NAME, MARKER_NAME, marker_from_json

    gfile = _gfile()
    with gfile.GFile(posixpath.join(shard_path, MARKER_NAME), "r") as handle:
        marker = marker_from_json(handle.read())
    with gfile.GFile(posixpath.join(shard_path, HEADER_NAME), "r") as handle:
        header = header_from_json(handle.read())
    records = []
    for name in marker.names:
        with gfile.GFile(posixpath.join(shard_path, marker.files[name]), "rb") as handle:
            records.append(record_from_bytes(handle.read()))
    return header, tuple(records)


def upload_artifact(local_path: str, remote_path: str, *, gfile=None) -> str:
    """Copy one finished local file into place, through a staged name so nothing partial is visible."""
    gfile = gfile or _gfile()
    gfile.makedirs(posixpath.dirname(remote_path))
    staged = f"{remote_path}.partial"
    if gfile.exists(staged):
        gfile.remove(staged)
    gfile.copy(local_path, staged, overwrite=True)
    gfile.rename(staged, remote_path, overwrite=True)
    return remote_path


def publish_video(frames: Any, path: str, fps: int = VIDEO_FPS, *, gfile=None, workdir=None) -> str:
    """Encode locally, then publish to wherever ``path`` points -- including ``gs://``.

    ``save_video_mp4`` is an ``os.makedirs``/``os.replace`` writer, so handing it the configured
    ``gs://…/videos/00.mp4`` created a local directory literally named ``gs:`` and published nothing
    (review, finding 8). The encode stays exactly where it was -- one ffmpeg subprocess, on one host,
    after the compute -- and only the finished bytes travel. The PNG fallback travels too: it is the
    path taken on a host without ffmpeg, which is the host most likely to need the diagnostic.
    """
    if not str(path).startswith("gs://"):
        return save_video_mp4(frames, path, fps=fps)
    gfile = gfile or _gfile()
    with (workdir or tempfile.TemporaryDirectory)() as tmp:
        local = os.path.join(tmp, posixpath.basename(path))
        produced = save_video_mp4(frames, local, fps=fps)
        if produced == local:
            return upload_artifact(produced, path, gfile=gfile)
        remote_dir = f"{posixpath.splitext(path)[0]}_frames"
        for entry in sorted(os.listdir(produced)):
            upload_artifact(os.path.join(produced, entry), posixpath.join(remote_dir, entry), gfile=gfile)
        return remote_dir


def default_sinks() -> Sinks:
    """The production seam: real shard publication, real JSON, real videos, real resume."""
    from maxdiffusion.null_adapter_shards import resume_plan, validate_shard

    return Sinks(
        write_shard=write_shard,
        write_json=write_json,
        save_video=lambda frames, path, fps: publish_video(frames, path, fps),
        read_shard=read_shard,
        resume_plan=resume_plan,
        validate_shard=validate_shard,
        read_json=read_json,
        read_marker=read_marker,
        write_arrays=write_arrays,
    )
