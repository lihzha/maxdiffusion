"""The positive-slot J1 execution path for exp_05 (plan §5 items 2-3, §4-P1'; round S4).

The sibling of ``null_adapter_modes``, and deliberately a **separate module**: plan §6's merge policy
makes ``null_adapter_modes.py`` a class-(c) dual-touch file, and every line of positive-slot logic
that lives there is a line the next exp_04 -> exp_05 merge has to reconcile. Keeping the whole B-arm
path here reduces exp_05's edit surface on shared files to a single additive dispatch in
``run_wan_null_inversion.main`` -- ``null_adapter_modes.py`` is **not edited at all** -- while
everything else is reused from exp_04 **by import**: the ``Backend``/``Sinks`` seams, the gate
functions, the quarantine and divergence seams, the batch fingerprints, the metric tables, the
bounded cohort decode and the pixel fill.

**The B-arms (plan §4-P1'), and why the pivots are computed here.** exp_04's A-arms invert at the
512-row T5("") context; exp_05 inverts at the **8-token** ``pos_context_from_t5(base)`` context,
because that is the representation the deployed adapter conditions on (S1). The trajectories are
therefore *different tensors* than exp_04's, which is why plan §3 forbids reusing exp_04's pivots and
why ``run_pos_capacity_example_batch`` computes its own. The pin for that is a test that captures the
inversion call's context shape and requires ``[B, 8, D]``.

    B0        frozen-C replay from traj[0]   -- the matched control; CFG is ACTIVE (S3), not collapsed
    B1        optimize + replay from traj[0]
    B1-probe  B1's contexts from keyed{0,1,2}
    B2        optimize + replay from eps_0
    B2-0      frozen-C replay from eps_0
    B2-probe  B2's contexts from keyed{0,1,2}

**THE S4 MUST -- the cast rule's open half, DISCHARGED here.** S2/S3 proved the operators do *not*
cast the context; nothing yet proved the runner-built ``velocity_fn`` *does*.
``casting_velocity_fn`` is that closure, and ``test_pos_context_runner.py`` pins it at bf16 for
**both** branches -- the 8-token conditional and the 512-row unconditional. (exp_04's ``_load_backend``
closure already casts both, at ``run_wan_null_inversion.py:537``; this wrapper makes the guarantee
exp_05's own, tested rather than inherited, and is idempotent when composed with it.)

**Publication.** The ordering is arms, decode, fill, gates, selection, records, and only then the
run-level JSON -- and the records go through the S5 codec with the S5 ``PosProvenanceHeader`` (``l_pos``,
never ``l_null``). The **record writer itself is not exp_04's**: ``null_adapter_shards.write_shard``
hard-codes exp_04's ``record_to_bytes``/``header_to_json``, so it cannot serialize a
``PosContextRecord``, and exp_05 may not edit it. ``pos_default_sinks`` therefore supplies every
exp_04 sink except that one, which fails loudly pending its own round -- a positive capacity run
gates, selects and reports, and refuses to *publish* records rather than pretending to.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from maxdiffusion.models.wan.null_inversion_wan import (
    base_context_fingerprint,
    global_noise,
    invert_trajectory,
)
from maxdiffusion.models.wan.pos_context_inversion_wan import (
    POS_L,
    optimize_positive_embeddings,
    pos_context_from_t5,
    replay_with_positive,
)
from maxdiffusion.null_adapter_cache_policy import quarantine_batch_failures
from maxdiffusion.null_adapter_gates import (
    NoiseConvention,
    Target,
    gate_g1,
    gate_g2,
    select_target,
    verdicts_to_json,
)
from maxdiffusion.null_adapter_modes import (
    ADEQUACY_COHORT,
    ADEQUACY_EXAMPLES,
    ADEQUACY_NAME,
    REPORT_NAME,
    SELECTION_COHORT,
    SELECTION_NAME,
    TABLES_NAME,
    Backend,
    Sinks,
    _score_payload,
    decode_cohort,
    default_sinks,
    merge_latents,
)
from maxdiffusion.null_adapter_pixels import fill_pixel_metrics
from maxdiffusion.null_adapter_runner_core import (
    ADEQUACY_GRID,
    DEFAULT_RECIPE,
    PROBE_K_SET,
    RECORD_FIELDS,
    SINGLE_SEED_KEY,
    AdequacyReport,
    CapacityParams,
    RecipeScore,
    adopt_recipe,
    _checked_trace,
    _keyed_batch,
    _metric_tables,
    _tracking_losses,
    _validate as _validate_batch,
    batch_fingerprint,
)
from maxdiffusion.null_adapter_verify import canonical_sigmas
from maxdiffusion.pos_context_records import (
    PRODUCTION_POS_GEOMETRY,
    PosProvenanceHeader,
    _validate_pos_header,
    make_pos_record,
)
from maxdiffusion.run_wan_null_inversion import guard_example_divergence


POS_METHODS = ("b0", "b1", "b1_probe", "b2", "b2_0", "b2_probe")
POS_PROBE_METHODS = {"b1_probe": "b1", "b2_probe": "b2"}
# Which record arm a selection verdict deploys, in the positive slot's names.
POS_RECORD_ARMS = {"b1": ("B1", "keyed"), "b2": ("B2", "global")}
POS_RECORD_ARM_ORDER = ("b1", "b2")
POS_TARGET_ARMS = {Target.A1_KEYED: "b1", Target.A2_GLOBAL: "b2", Target.STOP: None}


def casting_velocity_fn(velocity_fn: Callable[..., Any], activations_dtype: Any) -> Callable[..., Any]:
    """**THE S4 MUST**: the runner-built closure casts BOTH branches' contexts to the activation dtype.

    The one cast rule (``pos_context_inversion_wan``'s module docstring) says the optimize and replay
    operators pass fp32 contexts through untouched and the runner-built ``velocity_fn`` performs the
    activation-dtype cast immediately before the transformer call. This is that closure. It casts on
    every call, so the 8-token conditional context and the 512-row unconditional context are treated
    identically -- an asymmetric cast would condition one branch on a representation the deployed
    adapter never emits, which is the exact failure S1 measured.

    Composing it over exp_04's ``_load_backend`` closure (which already casts) is a no-op, because the
    cast is idempotent; the wrapper exists so exp_05 owns a *tested* one rather than inheriting an
    untested guarantee.
    """
    import jax.numpy as jnp

    def cast_then_velocity(latents, timestep_2d, encoder_hidden_states):
        return velocity_fn(latents, timestep_2d, jnp.asarray(encoder_hidden_states).astype(activations_dtype))

    return cast_then_velocity


def pos_header_for(
    plan: Mapping[str, Any], backend: Backend, *, manifest_hash: str, code_sha: str
) -> PosProvenanceHeader:
    """The one provenance header a positive run's shards share -- ``l_pos``, never ``l_null`` (S4)."""
    return PosProvenanceHeader(
        manifest_hash=manifest_hash,
        code_sha=code_sha,
        model_revision=backend.model_revision,
        sigma_vector=canonical_sigmas(),
        guide_scale=float(plan["params"]["guide_scale"]),
        base_context_fingerprint=base_context_fingerprint(backend.base_context),
        optimization_config=dict(plan["optimization_config"]),
        dtype_policy=str(plan["latent_dtype"]),
        l_pos=int(plan["params"].get("l_pos", POS_L)),
    )


def emit_pos_metric_tables(arm_results: Mapping[str, Any]) -> dict[str, dict]:
    """The per-method tables the gates consume, over the B-methods (exp_04's emission contract)."""
    names, metrics = arm_results["names"], arm_results["metrics"]
    tables = {}
    for method in POS_METHODS:
        table = metrics.get(method)
        if not isinstance(table, Mapping) or sorted(table) != sorted(names):
            raise ValueError(f"the {method!r} table does not cover the batch exactly: {sorted(table or {})}")
        keys = [str(k) for k in PROBE_K_SET] if method in POS_PROBE_METHODS else [SINGLE_SEED_KEY]
        for name in names:
            if sorted(table[name]) != sorted(keys):
                raise ValueError(f"the {method!r} table has the wrong seed keys for {name!r}: {sorted(table[name])}")
        tables[method] = {name: {key: dict(table[name][key]) for key in keys} for name in names}
    return tables


def _normalized_root(path: Any) -> str:
    """One spelling per root: ``gs://b/x/`` and ``gs://b/x`` are the same directory (follow-up 1)."""
    return str(path or "").rstrip("/")


def positive_roots(config: Any) -> tuple[str, str]:
    """The positive slot's artifact and staging roots -- **normalized, required, and never the null's**.

    Resolvable *before* any storage operation, which is the point: ``main`` used to run exp_04's
    free-space check and staging sweep against ``null_artifact_dir``/``null_staging_dir`` whatever the
    slot, so the checked-in positive YAML (whose null roots are empty) died in ``os.statvfs("")``
    before the plan existed, and a positive run's own staging was never swept (follow-up finding 1).

    Comparison is on the normalized form, so a trailing slash cannot smuggle a positive root onto the
    null slot's tree.
    """
    from maxdiffusion.run_wan_null_inversion import optional_config_value

    artifact_dir = _normalized_root(optional_config_value(config, "pos_artifact_dir", ""))
    staging_dir = _normalized_root(optional_config_value(config, "pos_staging_dir", ""))
    missing = [key for key, value in (("pos_artifact_dir", artifact_dir), ("pos_staging_dir", staging_dir)) if not value]
    if missing:
        raise ValueError(
            f"a positive-slot run needs its own {' and '.join(missing)}: writing B-arm artifacts into "
            f"the null slot's root would put two experiments' selection.json in one directory, and an "
            f"unset root cannot even be free-space checked"
        )
    null_roots = {
        _normalized_root(optional_config_value(config, key, ""))
        for key in ("null_artifact_dir", "null_staging_dir")
    } - {""}
    collisions = sorted({artifact_dir, staging_dir} & null_roots)
    if collisions:
        raise ValueError(
            f"pos_artifact_dir/pos_staging_dir must not be the null slot's roots, got {collisions} "
            f"(compared after normalization, so a trailing slash does not hide a collision)"
        )
    return artifact_dir, staging_dir


def positive_plan(config: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    """The positive slot's plan additions -- ``l_pos``, its ablation, and the slot-isolated roots.

    ``plan_run`` is exp_04's and stays untouched (its output is byte-identical for a null run), so the
    positive-only parameters are layered on here. ``l_pos`` reaches the inversion context, the warm
    start, the frozen-context arms and the optimizer initialization from this one number: S4 shipped
    them hard-wired at eight, so a config asking for ``pos_L=1`` silently ran eight (review, finding 4).

    The artifact roots are **required** and may not be the null slot's: a positive run writing B1/B2
    shards under ``null_artifact_dir`` puts two experiments' ``selection.json`` in one directory
    (finding 3).
    """
    from maxdiffusion.run_wan_null_inversion import optional_config_value

    l_pos = int(optional_config_value(config, "pos_L", POS_L))
    if l_pos < 1:
        raise ValueError(f"pos_L must be a positive integer, got {l_pos}")
    artifact_dir, staging_dir = positive_roots(config)

    ablation = tuple(
        int(cell) for cell in str(optional_config_value(config, "pos_ablation_L", "") or "").split(",") if cell.strip()
    )
    if any(cell < 1 for cell in ablation):
        raise ValueError(f"pos_ablation_L cells must be positive integers, got {ablation}")
    updated = dict(plan)
    updated["embedding_slot"] = "positive"
    updated["params"] = {**plan["params"], "l_pos": l_pos}
    updated["ablation_l_pos"] = ablation
    updated["artifact_dir"] = artifact_dir
    updated["staging_dir"] = staging_dir
    return updated


def run_pos_capacity_example_batch(
    velocity_fn: Callable[..., Any],
    batch: Any,
    base_context: Any,
    params: CapacityParams = CapacityParams(),
    *,
    l_pos: int = POS_L,
) -> dict[str, Any]:
    """Every plan §4-P1' B-arm on one batch, inverting **at the 8-token context**, exactly once.

    The inversion context is ``pos_context_from_t5(base_context)`` broadcast over the batch -- not the
    512-row T5(""). That is the whole reason exp_05 cannot reuse exp_04's pivots (plan §3): a
    different conditional representation is a different trajectory, and a study that borrowed the
    null slot's pivots would be measuring the positive slot's contexts against someone else's
    reference points.
    """
    import jax.numpy as jnp

    names = tuple(batch.names)
    z_i0 = jnp.asarray(batch.z_i0, jnp.float32)
    z_video = jnp.asarray(batch.z_video, jnp.float32)
    context = jnp.asarray(base_context, jnp.float32)
    sigmas = jnp.asarray(canonical_sigmas())
    steps = sigmas.shape[0] - 1
    # ONE l_pos reaches all four places it has to (S4 review, finding 4): the inversion context, the
    # warm start the optimizer initializes from, the frozen-context arms, and the recipe below.
    warm = pos_context_from_t5(context, l_pos)
    cond = jnp.broadcast_to(warm, (len(names), *warm.shape))

    traj = invert_trajectory(lambda z, t: velocity_fn(z, t, cond), z_video, z_i0, sigmas)
    eps_0 = jnp.broadcast_to(global_noise(0), (len(names), *PRODUCTION_POS_GEOMETRY.z_video))
    frozen = jnp.broadcast_to(warm, (steps, len(names), *warm.shape))
    recipe = {
        "inner_iters": params.inner_iters,
        "lr": params.lr,
        "guide_scale": params.guide_scale,
        "pos_init": warm,
    }

    def replay(z_start, embeds):
        return replay_with_positive(
            velocity_fn, z_start, z_i0, sigmas, embeds, context, guide_scale=params.guide_scale
        )

    def optimize(pivots):
        embeds, states, _, losses, norms = optimize_positive_embeddings(
            velocity_fn, pivots, z_i0, sigmas, context, **recipe
        )
        traces = {
            "tracking_losses": _checked_trace(losses, "tracking_losses"),
            "grad_norms": _checked_trace(norms, "grad_norms"),
        }
        return embeds, states, traces

    embeds, states, z_start, per_step, diagnostics = {}, {}, {}, {}, {}
    for arm, pivots, start in (("b1", traj, traj[0]), ("b2", traj.at[0].set(eps_0), eps_0)):
        embeds[arm], states[arm], diagnostics[arm] = optimize(pivots)
        z_start[arm] = start
        # The optimizer's own objective at the locked contexts, against the pivots it tracked.
        full = jnp.concatenate([states[arm], replay(start, embeds[arm])[None]], axis=0)
        per_step[arm] = _checked_trace(_tracking_losses(full, pivots), "per_step_final_losses")

    latents = {
        "b0": replay(traj[0], frozen),
        "b1": replay(traj[0], embeds["b1"]),
        "b2": replay(eps_0, embeds["b2"]),
        "b2_0": replay(eps_0, frozen),
        "b1_probe": jnp.stack([replay(_keyed_batch(names, k), embeds["b1"]) for k in PROBE_K_SET]),
        "b2_probe": jnp.stack([replay(_keyed_batch(names, k), embeds["b2"]) for k in PROBE_K_SET]),
    }
    return {
        "names": names,
        "metrics": _metric_tables(names, latents, z_video),
        "final_latents": {method: np.asarray(value) for method, value in latents.items()},
        "pos_embeds": {arm: np.asarray(value) for arm, value in embeds.items()},
        "z_bar_states": {arm: np.asarray(value) for arm, value in states.items()},
        "z_start": {arm: np.asarray(value) for arm, value in z_start.items()},
        "per_step_final_losses": per_step,
        "diagnostics": diagnostics,
        "params": params,
        "l_pos": int(l_pos),
        "base_context_fingerprint": base_context_fingerprint(context),
        "batch_fingerprint": batch_fingerprint(names, batch.z_i0, batch.z_video),
    }


def build_pos_capacity_records(
    velocity_fn: Callable[..., Any],
    arm_results: Mapping[str, Any],
    batch: Any,
    base_context: Any,
    header: PosProvenanceHeader,
    example_fields: Mapping[str, Mapping[str, Any]],
    *,
    arm: str,
) -> list[Any]:
    """Cast -> replay -> record, in that order (exp_04 R4c's writer contract, on the S5 schema).

    The stored dtypes are applied *first* and the replay runs from the cast values, so
    ``expected_final_latent`` describes an endpoint the record's own bytes actually reach. The
    per-step states ride along at the same dtype -- they are the teacher-forcing targets, and a state
    cached at a precision the replay never saw would be a training target for a trajectory nobody ran.
    """
    import jax.numpy as jnp

    from maxdiffusion.pos_context_records import LATENT_DTYPES

    # ---- the writer preflight, BEFORE any replay (S4 review, finding 2). ``arm_results`` binds the
    # recipe, the context and the batch it was produced from, and every one of those must agree with
    # the header and with the arrays passed in. Otherwise embeddings optimized at (J=1, lr=1e-2, w=5)
    # can be published under a header advertising (J=50, lr=3e-2, w=7), and a batch that merely reuses
    # the names carries someone else's z_video into the cache -- both of which still *verify*, because
    # a verifier checks a record against itself. That is exactly why it has to be caught here.
    if arm not in POS_RECORD_ARMS:
        raise ValueError(f"arm must be one of {sorted(POS_RECORD_ARMS)}, got {arm!r}")
    if header.embedding_slot != "positive":
        raise ValueError(f"a positive-slot record needs a positive-slot header, got {header.embedding_slot!r}")
    header_sigmas = _validate_pos_header(header)  # the header's own contract, reused not restated
    recipe = arm_results["params"]
    names, z_i0_checked, z_video_checked, context = _validate_batch(batch, base_context, recipe)

    if not np.array_equal(header_sigmas, canonical_sigmas()):
        raise ValueError("header sigma_vector does not match the canonical grid: these records could never verify")
    if base_context_fingerprint(context) != header.base_context_fingerprint:
        raise ValueError('header base_context_fingerprint does not match this base_context: a different T5("")')
    if arm_results["base_context_fingerprint"] != header.base_context_fingerprint:
        raise ValueError("the header's base-context fingerprint is not the one these arms were run against")
    if tuple(arm_results["names"]) != names:
        raise ValueError(
            f"arm_results and batch describe different examples: {list(arm_results['names'])} vs {list(names)}"
        )
    if arm_results["batch_fingerprint"] != batch_fingerprint(names, z_i0_checked, z_video_checked):
        raise ValueError("batch fingerprint does not match arm_results: the same names carry different tensors")
    if float(header.guide_scale) != float(recipe.guide_scale):
        raise ValueError(f"header guide_scale {header.guide_scale} does not match the run's {recipe.guide_scale}")
    declared = {key: header.optimization_config.get(key) for key in ("inner_iters", "lr")}
    if declared != {"inner_iters": recipe.inner_iters, "lr": float(recipe.lr)}:
        raise ValueError(
            f"header optimization_config {declared} does not match the run's recipe "
            f"{{'inner_iters': {recipe.inner_iters}, 'lr': {float(recipe.lr)}}}"
        )
    produced_l_pos = int(arm_results["l_pos"])
    stored_rows = int(np.asarray(arm_results["pos_embeds"][arm]).shape[2])
    if not int(header.l_pos) == produced_l_pos == stored_rows:
        raise ValueError(
            f"header l_pos {header.l_pos} must equal the l_pos the arms ran at ({produced_l_pos}) and the "
            f"optimized context's stored rows ({stored_rows})"
        )
    wrong = [name for name in names if sorted(example_fields.get(name, {})) != sorted(RECORD_FIELDS)]
    if wrong:
        raise ValueError(f"example_fields must carry exactly {list(RECORD_FIELDS)} for every name; wrong for {wrong}")

    latent_dtype = LATENT_DTYPES[str(header.dtype_policy)]
    embeds = np.asarray(arm_results["pos_embeds"][arm]).astype(latent_dtype)
    states = np.asarray(arm_results["z_bar_states"][arm]).astype(latent_dtype)
    z_start = np.asarray(arm_results["z_start"][arm]).astype(latent_dtype)
    sigmas = jnp.asarray(header.sigma_vector, jnp.float32)

    final = replay_with_positive(
        velocity_fn,
        jnp.asarray(z_start, jnp.float32),
        jnp.asarray(batch.z_i0, jnp.float32),
        sigmas,
        jnp.asarray(embeds, jnp.float32),
        jnp.asarray(base_context, jnp.float32),
        guide_scale=float(header.guide_scale),
    )
    final = np.asarray(final)
    label, convention = POS_RECORD_ARMS[arm]
    losses = np.asarray(arm_results["per_step_final_losses"][arm], np.float32)
    return [
        make_pos_record(
            name=name,
            **{field: example_fields[name][field] for field in ("ordinal", "split", "episode")},
            actions=example_fields[name]["actions"],
            z_i0=np.asarray(batch.z_i0)[index],
            z_video=np.asarray(batch.z_video)[index],
            latent_dtype=str(header.dtype_policy),
            pos_embeds=embeds[:, index],
            z_bar_states=states[:, index],
            z_start=z_start[index],
            expected_final_latent=final[index],
            noise_convention=convention,
            arm=label,
            per_step_final_losses=losses[index],
            final_future_mse=float(np.mean((final[index, :, 1:] - np.asarray(batch.z_video)[index, :, 1:]) ** 2)),
        )
        for index, name in enumerate(names)
    ]



# --------------------------------------------------------------------------------------------------
# The adequacy probe and the L_pos ablation (plan §4-P1'; S4 review, finding 4).
# --------------------------------------------------------------------------------------------------


def _pos_pivots(velocity_fn, batch, context, sigmas, l_pos: int):
    """The 8-token inversion trajectory the arms and the probe share (plan §3: computed, never reused)."""
    import jax.numpy as jnp

    warm = pos_context_from_t5(context, l_pos)
    cond = jnp.broadcast_to(warm, (len(batch.names), *warm.shape))
    traj = invert_trajectory(
        lambda z, t: velocity_fn(z, t, cond),
        jnp.asarray(batch.z_video, jnp.float32),
        jnp.asarray(batch.z_i0, jnp.float32),
        sigmas,
    )
    return traj, warm


def _preflight_pos_adequacy(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Everything wrong with a probe request, decided **before a single example is read**.

    The same fixed experiment exp_04 preflights: the first eight DEV names on the approved six-cell
    grid. A probe run on three arbitrary examples and an arbitrary grid answers a question nobody
    asked and costs the same.
    """
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


def _pos_probe_cell(velocity_fn, batch, context, traj, warm, cell, *, guide_scale: float) -> RecipeScore:
    """One (J, lr) cell on the positive optimizer, with the evidence it produced.

    ``final_losses`` is the ``[B, N]`` post-inner-loop tracking loss -- the optimizer's own objective
    at the locked contexts, which is what the score reduces; ``per_example`` is each example's mean
    over the sampler steps and the score is their median (exp_04's M3 rule, on the positive slot).
    """
    import jax.numpy as jnp

    inner_iters, lr = int(cell[0]), float(cell[1])
    embeds, states, z_final, losses, norms = optimize_positive_embeddings(
        velocity_fn,
        traj,
        jnp.asarray(batch.z_i0, jnp.float32),
        jnp.asarray(canonical_sigmas()),
        context,
        inner_iters=inner_iters,
        lr=lr,
        guide_scale=guide_scale,
        pos_init=warm,
    )
    del embeds
    full = jnp.concatenate([states, z_final[None]], axis=0)
    final = np.asarray(_checked_trace(_tracking_losses(full, traj), "final_losses"))
    per_example = tuple(float(value) for value in final.mean(axis=1))
    return RecipeScore(
        inner_iters=inner_iters,
        lr=lr,
        score=float(np.median(per_example)),
        per_example=per_example,
        tracking_losses=np.asarray(_checked_trace(losses, "tracking_losses")),
        grad_norms=np.asarray(_checked_trace(norms, "grad_norms")),
        final_losses=final,
    )


def _pos_l_ablation(velocity_fn, batch, context, plan: Mapping[str, Any], *, recipe, arm: str) -> dict[str, Any]:
    """Plan §4's DIAGNOSTIC-only ``L_pos in {1, 8}`` sweep.

    It reports reconstruction quality at each row count and **produces nothing publishable**: the
    records, the header and the shard writer are all fixed at ``plan["params"]["l_pos"]``, so a
    one-row diagnostic can never become a cached target (guarded by test and by the writer's own
    ``header.l_pos == produced == stored`` preflight).

    **It runs at the ADOPTED recipe and the SELECTED arm's calculation, and says so in its evidence.**
    Run at the default recipe instead, it answers "does L matter at (10, 1e-2)?" while K1 proceeds at
    whatever the probe adopted -- a diagnostic about a configuration the experiment is not using
    (follow-up finding 3). Both bindings are persisted so a reader can see which comparison was made.
    """
    import jax.numpy as jnp

    if arm not in POS_RECORD_ARMS:
        raise ValueError(f"the ablation must mirror a record arm's calculation, got {arm!r}")
    guide_scale = float(plan["params"]["guide_scale"])
    recipe = {"inner_iters": int(recipe["inner_iters"]), "lr": float(recipe["lr"])}
    sigmas = jnp.asarray(canonical_sigmas())
    cells = {}
    for l_pos in plan.get("ablation_l_pos", ()):
        traj, warm = _pos_pivots(velocity_fn, batch, context, sigmas, int(l_pos))
        if arm == "b2":  # B2's calculation starts from the single canonical noise, not from traj[0]
            eps_0 = jnp.broadcast_to(global_noise(0), (len(batch.names), *PRODUCTION_POS_GEOMETRY.z_video))
            traj = traj.at[0].set(eps_0)
        _, states, z_final, _, _ = optimize_positive_embeddings(
            velocity_fn, traj, jnp.asarray(batch.z_i0, jnp.float32), sigmas, context,
            guide_scale=guide_scale, pos_init=warm, **recipe,
        )
        full = jnp.concatenate([states, z_final[None]], axis=0)
        final = np.asarray(_tracking_losses(full, traj))
        cells[str(int(l_pos))] = {
            "l_pos": int(l_pos),
            "final_tracking_loss": float(np.median(final.mean(axis=1))),
            "per_example": [float(v) for v in final.mean(axis=1)],
        }
    return {
        "diagnostic_only": True,
        "published_l_pos": int(plan["params"]["l_pos"]),
        "recipe": recipe,  # the ADOPTED one, bound into the evidence
        "arm": arm,
        "cells": cells,
    }


def run_pos_adequacy(
    plan: Mapping[str, Any], backend: Backend, sinks: Sinks, *, artifact_dir: str, manifest_hash: str = ""
) -> dict[str, Any]:
    """Plan §4-P1's adequacy probe on the POSITIVE slot: first-eight DEV, the approved six-cell grid.

    Mirrors ``null_adapter_modes.run_adequacy`` -- same preflight, same adoption rule (exp_04's
    ``adopt_recipe``, reused), same persisted evidence (per-example scores, the optimizer's ``[N,J,B]``
    traces, the ``[B,N]`` final losses, the per-cell wall time capacity's re-run projection needs) --
    over ``optimize_positive_embeddings`` and the 8-token pivots.
    """
    import jax.numpy as jnp

    started = time.time()
    names = _preflight_pos_adequacy(plan)  # before any data is read
    batch, _ = backend.read_batch(names)
    context = jnp.asarray(backend.base_context, jnp.float32)
    guide_scale, l_pos = float(plan["params"]["guide_scale"]), int(plan["params"]["l_pos"])
    traj, warm = _pos_pivots(backend.velocity_fn, batch, context, jnp.asarray(canonical_sigmas()), l_pos)

    scores, timings = [], []
    for cell in plan["grid"]:
        cell_started = time.time()
        scores.append(_pos_probe_cell(backend.velocity_fn, batch, context, traj, warm, cell, guide_scale=guide_scale))
        timings.append(time.time() - cell_started)
    adoption = adopt_recipe(AdequacyReport(names, tuple(scores)))
    per_cell = {(score.inner_iters, score.lr): seconds for score, seconds in zip(scores, timings)}

    payload = {
        "mode": "adequacy_probe",
        "embedding_slot": "positive",
        "cohort": plan["cohort"],
        "names": list(names),
        "grid": [list(cell) for cell in plan["grid"]],
        "guide_scale": guide_scale,
        "l_pos": l_pos,
        "scores": [_score_payload(score, seconds) for score, seconds in zip(scores, timings)],
        "adopted": {
            "inner_iters": adoption.inner_iters,
            "lr": adoption.lr,
            "adopted": adoption.adopted,
            "projection_seconds_per_example": per_cell[(adoption.inner_iters, adoption.lr)] / len(names),
        },
        "plateau": adoption.plateau,
        "reasons": list(adoption.reasons),
        "numbers": adoption.numbers,
        "manifest_hash": manifest_hash,
        "l_pos_ablation": _pos_l_ablation(
            backend.velocity_fn, batch, context, plan,
            recipe={"inner_iters": adoption.inner_iters, "lr": adoption.lr},
            arm=str(plan.get("ablation_arm", POS_RECORD_ARM_ORDER[0])),
        ),
        "seconds": round(time.time() - started, 3),
    }
    sinks.write_json(posixpath.join(artifact_dir, ADEQUACY_NAME), payload)
    return payload


# --------------------------------------------------------------------------------------------------
# The positive selection artifact (S4 review, finding 3).
# --------------------------------------------------------------------------------------------------


def pos_selection_payload(verdicts: Mapping[str, Any], plan: Mapping[str, Any], *, manifest_hash: str = "") -> dict:
    """The K1 artifact K2 and the verifier read their arm from -- in the POSITIVE slot's own names.

    S4 serialized exp_04's payload verbatim, so a passing B1 selection said ``target="A1/keyed"``
    beside ``arm="b1"`` and ``label="B1"`` -- an artifact that contradicts itself and names an arm this
    experiment never ran (review, finding 3). Here the target is ``"B1/keyed"``/``"B2/global"``, the
    gate labels are H1/H2, and the provenance fields exp_04 requires (cohort, manifest binding, smoke
    flag) are carried unchanged, because ``pos_selected_arm`` checks exactly them.
    """
    selection = verdicts["selection"]
    arm = POS_TARGET_ARMS[selection.target]
    label, convention = POS_RECORD_ARMS[arm] if arm else (None, None)
    gates = json.loads(verdicts_to_json({"h1": verdicts["g1"], "h2": verdicts["g2"], "selection": selection}))
    return {
        "embedding_slot": "positive",
        "cohort": plan["cohort"],
        "manifest_hash": manifest_hash,
        "smoke_examples": int(plan.get("smoke_examples", 0)),
        "manifest": list(plan["names"]),
        "target": f"{label}/{convention}" if arm else Target.STOP.value,
        "arm": arm,
        "label": label,
        "noise_convention": convention,
        "l_pos": int(plan["params"].get("l_pos", POS_L)),
        "reasons": list(selection.reasons),
        "gates": gates,
    }


def pos_selected_arm(
    selection: Mapping[str, Any], *, expected_manifest_hash: str | None = None, allow_smoke: bool = False
) -> str:
    """The record arm a K1 selection deployed, or a refusal -- never a default (exp_04's rule, B-names).

    The artifact must be *this* experiment's: made on the positive slot, on the DEV cohort, against
    the manifest this job loaded, by a run that was not a smoke. An unbound selection is a file that
    says "B1", and a two-example smoke says it just as convincingly.
    """
    if not isinstance(selection, Mapping):
        raise ValueError(f"the K1 selection artifact must be a JSON object, got {type(selection).__name__}")
    if selection.get("embedding_slot") != "positive":
        raise ValueError(
            f"this selection was made in the {selection.get('embedding_slot')!r} slot: it does not "
            f"authorize a positive-slot cache"
        )
    target, arm = selection.get("target"), selection.get("arm")
    if target == Target.STOP.value or arm is None:
        raise ValueError(
            f"the K1 selection stopped after P1' ({'; '.join(selection.get('reasons', [])) or 'no reason recorded'}): "
            f"there is no selected arm to cache or verify"
        )
    if arm not in POS_RECORD_ARMS or POS_RECORD_ARMS[arm][0] != selection.get("label"):
        raise ValueError(f"the K1 selection artifact names an unusable arm: {arm!r}/{selection.get('label')!r}")
    if target != f"{POS_RECORD_ARMS[arm][0]}/{POS_RECORD_ARMS[arm][1]}":
        raise ValueError(f"the K1 selection artifact's target {target!r} disagrees with its arm {arm!r}")
    if selection.get("cohort") != SELECTION_COHORT:
        raise ValueError(
            f"the K1 selection was made on {selection.get('cohort')!r}, but the §4-P1' rule is defined on "
            f"{SELECTION_COHORT!r}: this artifact does not authorize a cache"
        )
    digest = selection.get("manifest_hash")
    if not isinstance(digest, str) or not digest:
        raise ValueError("the K1 selection artifact carries no manifest binding")
    if expected_manifest_hash is not None and digest != expected_manifest_hash:
        raise ValueError(
            f"the K1 selection was made against a different {SELECTION_COHORT} manifest "
            f"({digest[:12]}... vs {expected_manifest_hash[:12]}...): it does not authorize this job"
        )
    if not allow_smoke and int(selection.get("smoke_examples") or 0):
        raise ValueError(
            f"the K1 selection came from a {selection['smoke_examples']}-example smoke run: it cannot "
            f"authorize a cache"
        )
    return str(arm)


def pos_adoption(
    uri: str, plan: Mapping[str, Any], *, exists: Callable[[str], bool], read_json: Callable[[str], Any],
    manifest_hash: str | None = None,
) -> dict | None:
    """The adequacy artifact this run may adopt from -- **checked to be this slot's, and this job's**.

    exp_04's ``load_adoption`` only asks whether an adoption block is parseable, which is right for a
    single-slot driver. Pointed at a positive URI, it will just as happily consume a *null-slot*
    adequacy artifact and re-run the B-arms at a recipe chosen for a different experiment
    (follow-up finding 2). Everything the artifact claims about itself is therefore checked before a
    single number is applied: the slot, the mode, the cohort, ``l_pos``, the guidance weight, and the
    manifest the probe was run against.
    """
    if not uri or not exists(uri):
        return None
    payload = read_json(uri)  # a read failure propagates: the artifact exists, so it must be readable
    if not isinstance(payload, Mapping):
        raise ValueError(f"{uri} is not a positive adequacy artifact")
    if payload.get("embedding_slot") != "positive":
        raise ValueError(
            f"{uri} was produced in the {payload.get('embedding_slot')!r} slot: a positive run will not "
            f"adopt a recipe chosen for another experiment"
        )
    if payload.get("mode") != "adequacy_probe":
        raise ValueError(f"{uri} is a {payload.get('mode')!r} artifact, not an adequacy probe")
    if payload.get("cohort") != plan["cohort"]:
        raise ValueError(f"{uri} was probed on {payload.get('cohort')!r}, this run is on {plan['cohort']!r}")
    if int(payload.get("l_pos", -1)) != int(plan["params"]["l_pos"]):
        raise ValueError(
            f"{uri} was probed at l_pos={payload.get('l_pos')!r}, this run runs at "
            f"{plan['params']['l_pos']}: the recipe was chosen for a different representation"
        )
    if float(payload.get("guide_scale", float("nan"))) != float(plan["params"]["guide_scale"]):
        raise ValueError(f"{uri} was probed at w={payload.get('guide_scale')!r}, this run uses {plan['params']['guide_scale']}")
    digest = payload.get("manifest_hash")
    if manifest_hash is not None and digest != manifest_hash:
        raise ValueError(
            f"{uri} was probed against a different manifest ({str(digest)[:12]}... vs "
            f"{str(manifest_hash)[:12]}...): it does not authorize this job"
        )
    adoption = payload.get("adopted")
    if not isinstance(adoption, Mapping) or not {"inner_iters", "lr", "adopted"} <= set(adoption):
        raise ValueError(f"{uri} exists but carries no usable adoption block")
    return dict(adoption)


def _pos_batch_results(backend: Backend, names: Sequence[str], params: CapacityParams, *, l_pos: int = POS_L):
    """One batch of B-arms under exp_04's quarantine + divergence seams, unchanged."""

    def run_fn(subset):
        batch, _ = backend.read_batch(subset)
        results = run_pos_capacity_example_batch(
            backend.velocity_fn, batch, backend.base_context, params, l_pos=l_pos
        )
        return dict.fromkeys(subset, results)

    results, quarantined = quarantine_batch_failures(guard_example_divergence(run_fn), names)
    return (next(iter(results.values())) if results else None), quarantined


def evaluate_pos_gates(filled: Mapping[str, Mapping], manifest: Sequence[str]) -> dict[str, Any]:
    """H1/H2 and the target-selection rule -- **exp_04's gate functions**, on the B tables.

    Plan §4-P1' says H1/H2 are "≡ exp_04's G1/G2 forms verbatim", so they are literally the same
    functions here; only the arm names differ. The conventions are read off the tables exactly as
    exp_04 does: single-noise arms at seed key "0" under GLOBAL, probes at {0,1,2} under KEYED.
    """
    h1 = gate_g1(filled["b1"], filled["b0"], manifest, NoiseConvention.GLOBAL)
    h2 = gate_g2(filled["b2"], filled["b2_0"], manifest, NoiseConvention.GLOBAL)
    selection = select_target(h1, filled["b1_probe"], manifest, NoiseConvention.KEYED, h2)
    return {"g1": h1, "g2": h2, "selection": selection}


def run_capacity_positive(
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
) -> dict[str, Any]:
    """Plan §4-P1': every B-arm over the cohort, decoded, gated, selected, and only then recorded."""
    from maxdiffusion.null_adapter_modes import apply_adopted_recipe

    started = time.time()
    plan = apply_adopted_recipe(plan, adopted_recipe) if adopted_recipe else plan
    params = CapacityParams(**{k: v for k, v in plan["params"].items() if k != "l_pos"})
    l_pos = int(plan["params"].get("l_pos", POS_L))
    header = pos_header_for(plan, backend, manifest_hash=manifest_hash, code_sha=code_sha)

    batches: list[tuple[Mapping[str, Any], dict[str, str]]] = []
    names: list[str] = []
    quarantined: dict[str, str] = {}
    for batch_names in plan["batches"]:
        arm_results, lost = _pos_batch_results(backend, batch_names, params, l_pos=l_pos)
        quarantined.update(lost)
        if arm_results is None:
            continue
        batches.append((arm_results, lost))
        names.extend(arm_results["names"])
    if not names:
        raise RuntimeError("every batch was quarantined: there is nothing to gate")

    latents = merge_latents([_LatentsView(result) for result, _ in batches])
    pixels = decode_cohort(backend.decode_fn, backend.read_batch, latents, names, batch_size=decode_batch_size)
    tables = _merge_pos_tables([emit_pos_metric_tables(result) for result, _ in batches])
    filled = fill_pixel_metrics(tables, pixels)

    verdicts = evaluate_pos_gates(filled, plan["names"])
    selection = pos_selection_payload(verdicts, plan, manifest_hash=manifest_hash)

    # --- publication. The run-level JSONs are written **after** every shard has landed: with a
    # writer that can fail, exp_04's order would leave an authoritative-looking selection.json from a
    # run that never published a record (S4 review, finding 5). The decision is still made before
    # anything is written -- it is only its *visibility* that waits for the records.
    #
    # **The exact claim** (follow-up, narrowing S4's "zero artifacts on any failure"): a failure on the
    # FIRST shard leaves the artifact root empty. A failure on a LATER shard leaves the earlier shards
    # published -- which is safe, because a shard is immutable and carries its own completion marker,
    # but it does mean a retry into the same root must RESUME rather than republish: exp_04's R8
    # resume discipline reads the markers, skips what a validated shard already covers, and takes its
    # next shard identity from ``next_shard_index`` so it cannot reuse a published one. What is
    # guaranteed unconditionally is that no ``selection.json``/``run_report.json`` ever describes a run
    # whose records did not all land.
    shards = []
    for index, (arm_results, lost) in enumerate(batches):
        batch, fields = backend.read_batch(arm_results["names"])
        for record_arm in POS_RECORD_ARM_ORDER:
            records = build_pos_capacity_records(
                backend.velocity_fn, arm_results, batch, backend.base_context, header, fields, arm=record_arm
            )
            shard = posixpath.join(artifact_dir, POS_RECORD_ARMS[record_arm][0].lower(), f"shard_{index:05d}")
            sinks.write_shard(records, header, shard, staging_dir, quarantined=lost)
            shards.append(shard)
    sinks.write_json(posixpath.join(artifact_dir, TABLES_NAME), filled)
    sinks.write_json(posixpath.join(artifact_dir, SELECTION_NAME), selection)

    report = {
        "mode": "capacity",
        "embedding_slot": "positive",
        "cohort": plan["cohort"],
        "declared": len(plan["names"]),
        "examples": len(names),
        "smoke_examples": plan.get("smoke_examples", 0),
        "quarantined": quarantined,
        "recipe": dict(plan["optimization_config"]),
        "l_pos": int(header.l_pos),
        "target": selection["target"],
        "gates": {name: verdicts[name].reasons for name in ("g1", "g2")},
        "shards": shards,
        "tables": sorted(filled),
        "seconds": round(time.time() - started, 3),
    }
    sinks.write_json(posixpath.join(artifact_dir, REPORT_NAME), report)
    return report


class _LatentsView:
    """Adapts a positive arm-result mapping to the ``.final_latents`` attribute ``merge_latents`` reads."""

    def __init__(self, result: Mapping[str, Any]):
        self.final_latents = result["final_latents"]


def _merge_pos_tables(tables: Sequence[Mapping[str, Mapping]]) -> dict[str, dict]:
    """Per-batch tables into one cohort table; a name may be contributed by only one batch."""
    merged: dict[str, dict] = {method: {} for method in POS_METHODS}  # noqa: C420 -- distinct inner dicts
    for table in tables:
        for method, rows in table.items():
            if method not in merged:
                raise ValueError(f"unknown method {method!r} in a metric table")
            for name, entry in rows.items():
                if name in merged[method]:
                    raise ValueError(f"{name!r} appears in two batches of the {method} table")
                merged[method][name] = entry
    return merged


def pos_execute(
    mode: str, plan: Mapping[str, Any], backend: Backend, sinks: Sinks, **kwargs
) -> tuple[dict[str, Any], int]:
    """Dispatch one positive-slot mode. Only ``capacity`` (K1) is wired in S4; the rest say so."""
    if mode == "capacity":
        return run_capacity_positive(plan, backend, sinks, **kwargs), 0
    if mode == "adequacy_probe":
        return (
            run_pos_adequacy(
                plan, backend, sinks,
                artifact_dir=kwargs["artifact_dir"], manifest_hash=str(kwargs.get("manifest_hash", "")),
            ),
            0,
        )
    raise ValueError(
        f"the positive slot wires {('capacity', 'adequacy_probe')} in S4; {mode!r} belongs to a later round and must not "
        f"silently fall through to the null-slot implementation"
    )


def pos_write_shard(
    records: Sequence[Any],
    header: PosProvenanceHeader,
    shard_path: str,
    staging_prefix: str,
    *,
    quarantined: Mapping[str, str] | None = None,
) -> Any:
    """Publish one positive-slot shard: stage every record, then the header, then the marker LAST.

    exp_04's ``write_shard`` cannot serve this schema -- it hard-codes its own ``record_to_bytes`` and
    ``header_to_json`` -- and exp_05 may not edit it (plan §6). So the *writer* is restated here over
    the S5 codec while every rule it enforces is exp_04's, reused by import: the canonical
    sorted-position filename bijection, the ``MAX_SHARD_BYTES`` ceiling, one record resident at a
    time, a staging directory this attempt alone owns, an immutable destination, and **data first,
    marker last** so a reader that sees the marker sees everything the marker names.

    This is the S4b core the S4 review ruled K1-blocking: with it present the interim's "refuse at the
    end of an expensive run" failure mode is gone.
    """
    from maxdiffusion.null_adapter_shards import (
        HEADER_NAME,
        MARKER_NAME,
        MAX_SHARD_BYTES,
        SHARD_SCHEMA_VERSION,
        ShardMarker,
        _gfile,
        _publish,
        _refuse_existing_destination,
        _shard_member,
        _staging_path,
        _write_bytes,
        canonical_files,
    )
    from maxdiffusion.pos_context_records import pos_header_fingerprint, pos_header_to_json, pos_record_to_bytes

    records, quarantined = list(records), dict(quarantined or {})
    _validate_pos_write(records, header, quarantined)
    _refuse_existing_destination(shard_path)

    files = canonical_files([record.name for record in records])
    staging = _staging_path(shard_path, staging_prefix)
    by_name = {record.name: record for record in records}
    digests: dict[str, str] = {}
    published_bytes = 0
    try:
        for name in sorted(by_name):
            blob = pos_record_to_bytes(by_name[name])
            published_bytes += len(blob)
            if published_bytes > MAX_SHARD_BYTES:
                raise ValueError(f"shard exceeds MAX_SHARD_BYTES ({published_bytes} > {MAX_SHARD_BYTES})")
            _write_bytes(posixpath.join(staging, files[name]), blob)
            digests[name] = hashlib.sha256(blob).hexdigest()
            del blob
        _write_bytes(posixpath.join(staging, HEADER_NAME), pos_header_to_json(header).encode("utf-8"))
        marker = ShardMarker(
            schema_version=SHARD_SCHEMA_VERSION,
            count=len(records),
            names=tuple(sorted(by_name)),
            files=files,
            sha256=digests,
            header_fingerprint=pos_header_fingerprint(header),
            quarantined=quarantined,
        )
        _write_bytes(posixpath.join(staging, MARKER_NAME), marker.to_json().encode("utf-8"))

        _publish(posixpath.join(staging, HEADER_NAME), posixpath.join(shard_path, HEADER_NAME))
        for name in marker.names:
            _publish(posixpath.join(staging, files[name]), _shard_member(shard_path, files[name]))
        _publish(posixpath.join(staging, MARKER_NAME), posixpath.join(shard_path, MARKER_NAME))
        return marker
    finally:
        gfile = _gfile()
        if gfile.exists(staging):
            gfile.rmtree(staging)


def _validate_pos_write(records: Sequence[Any], header: PosProvenanceHeader, quarantined: Mapping[str, str]) -> None:
    """What may be published together: one slot, one dtype policy, one arm, no name in two places."""
    if header.embedding_slot != "positive":
        raise ValueError(f"a positive shard needs a positive-slot header, got {header.embedding_slot!r}")
    names = [record.name for record in records]
    if len(set(names)) != len(names):
        raise ValueError(f"a shard may not carry a name twice: {sorted({n for n in names if names.count(n) > 1})}")
    overlap = sorted(set(names) & set(quarantined))
    if overlap:
        raise ValueError(f"these names are both published and recorded as quarantined: {overlap}")
    for record in records:
        if record.latent_dtype != header.dtype_policy:
            raise ValueError(
                f"{record.name!r} is stored at {record.latent_dtype!r} but the header declares "
                f"{header.dtype_policy!r}"
            )
        if record.pos_embeds.shape[1] != int(header.l_pos):
            raise ValueError(
                f"{record.name!r} stores {record.pos_embeds.shape[1]} context rows but the header "
                f"declares l_pos={header.l_pos}"
            )
    arms = {record.arm for record in records}
    if len(arms) > 1:
        raise ValueError(f"a shard may carry only one arm, got {sorted(arms)}")


def pos_default_sinks() -> Sinks:
    """exp_04's production sinks, with the record writer replaced by this slot's own (S4b)."""
    import dataclasses

    return dataclasses.replace(default_sinks(), write_shard=pos_write_shard)
