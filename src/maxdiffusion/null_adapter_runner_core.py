"""Capacity-study orchestration for exp_04 null-text inversion (plan §4-P1, round R6).

Pure functions over injected arguments: the caller supplies ``velocity_fn`` (the frozen WAN backbone
wrapped to the R3/R4a seam), the T5("") context and host arrays. Nothing here loads a pipeline, reads
``pyconfig`` or touches GCS -- that wiring is R10's, and keeping it out is what lets the arms be
tested against toy oracles at full production geometry.

**The arms** (plan §4-P1). The inversion pivots are computed **once per batch** and every arm shares
them::

    A0        frozen replay  from traj[0]           (base-row nulls: v_unc == v_cond, unguided)
    A1        optimize       from traj[0]           + replay from traj[0]
    A1-probe  A1's nulls     from keyed(name, k)    k in {0, 1, 2}   -- does the basin transfer?
    A2        optimize       from eps_0             + replay from eps_0
    A2-0      frozen replay  from eps_0             (A2's matched control)
    A2-probe  A2's nulls     from keyed(name, k)    k in {0, 1, 2}

A2 keeps A1's *targets*: only the starting pivot is replaced by ``eps_0 = global_noise(0)``, so both
arms track the same inversion trajectory from different basins. Batch assembly follows R1's rules --
keyed draws are stacked per name, the global draw is broadcast.

**Metrics.** Future-frame (non-pinned) latent MSE against ``z_video`` is primary; the full-tensor MSE
rides along as the plan's secondary number. Latent frame 0 is the pinned image condition, identical
in every arm by construction, so including it would measure the pin rather than the method.

**The tables** ``emit_metric_tables`` returns are exactly what ``null_adapter_gates`` consumes:
``table[name][seed-key] -> {"future_mse": ..., "full_mse": ...}``. **``future_ssim`` is deliberately
absent** -- decoding lands in R7 -- and under the gates' semantics a missing primary metric makes the
observation *invalid*, not merely unscored. Every gate fed these tables as they stand will fail on
its invalidity ceiling. **R7 must fill ``future_ssim`` in before any gate verdict means anything.**

**Results are bound to the run that produced them.** ``ArmResults`` carries the exact
``CapacityParams``, the base-context fingerprint and a content fingerprint of the batch, and
``build_capacity_records`` refuses -- before any forward -- a header or an input batch that
disagrees with any of them. ``verify_replay`` checks a record against *itself*, so without this bind
nulls optimized at one recipe verify happily under a header advertising another, and a batch that
reuses the names publishes someone else's ``z_video`` with the first run's metrics. The optimizer's
``[N, J, B]`` loss and grad-norm traces are retained too (plan §4-P1 requires both), and a
non-finite trace fails the run rather than riding along as evidence.

**The writer contract (R4c's deferred pin).** ``build_capacity_records`` casts every tensor to its
storage dtype **first**, replays from the cast values, and only then records: cast -> replay ->
record. ``z_i0`` is stored at the source fp16 under every dtype policy, so an ``expected_final_latent``
computed from the pre-cast fp32 inputs describes an endpoint the record's own bytes do not reach, and
``null_adapter_verify.verify_replay`` rejects it. The provenance header is the contract: the guidance
weight, the sigma grid, the storage dtype and ``l_null`` all come from it, and a header that
disagrees with the run is refused before anything is built.

**Geometry is fail-closed at every public array entry** (48/9/12/20 latents, 48/1/12/20 first-frame
condition, 512x4096 context, the 25-step canonical grid). The latent sizes are taken from the codec's
own ``PRODUCTION_GEOMETRY`` so the runner and the artifact boundary cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from maxdiffusion.models.wan.null_inversion_wan import (
    base_context_fingerprint,
    global_noise,
    invert_trajectory,
    keyed_noise,
    optimize_null_embeddings,
    replay_with_nulls,
)
from maxdiffusion.null_adapter_records import (
    LATENT_DTYPES,
    PRODUCTION_GEOMETRY,
    SOURCE_DTYPES,
    NullAdapterRecord,
    ProvenanceHeader,
    _validate_header,
    make_record,
    sha256_of_array,
)
from maxdiffusion.null_adapter_verify import canonical_sigmas


# 512 is the WAN T5 sequence length (``wan_max_sequence_length``); 4096 is its embedding width, which
# is also the codec's null-token width, so the two cannot drift apart unnoticed.
PRODUCTION_CONTEXT_SHAPE = (512, PRODUCTION_GEOMETRY.nulls[2])
PROBE_K_SET = (0, 1, 2)
SINGLE_SEED_KEY = "0"
METHODS = ("a0", "a1", "a1_probe", "a2", "a2_0", "a2_probe")
PROBE_METHODS = {"a1_probe": "a1", "a2_probe": "a2"}
# The record arms and what they mean downstream: the label the verifier is told to expect, and the
# deployment noise convention the plan's §4-P1 selection rule attaches to the arm (A1 is the keyed
# target, A2 the single-canonical-noise fallback).
RECORD_ARMS = {"a1": ("A1", "keyed"), "a2": ("A2", "global")}
RECORD_FIELDS = ("ordinal", "split", "episode", "actions")
PRIMARY_METRIC, SECONDARY_METRIC = "future_mse", "full_mse"

DEFAULT_RECIPE = (10, 0.01)
ADEQUACY_GRID = tuple((inner_iters, lr) for lr in (0.01, 0.03) for inner_iters in (10, 25, 50))
ADOPTION_FACTOR = 0.5
PLATEAU_PAIR = (25, 50)
PLATEAU_MIN_IMPROVEMENT = 0.10
PLATEAU_BOUNDARY_ATOL = 1e-9  # see _plateau: keeps the exact-10% case off the "below 10%" side
RECONSTRUCTION_LIMITED, RECIPE_LIMITED, UNDETERMINED = "reconstruction-limited", "recipe-limited", "undetermined"


@dataclasses.dataclass(frozen=True)
class CapacityBatch:
    """One batch of manifest examples at the production latent geometry."""

    names: tuple[str, ...]
    z_i0: Any
    z_video: Any


@dataclasses.dataclass(frozen=True)
class CapacityParams:
    """The optimization recipe every arm in a run shares (plan §3 defaults)."""

    inner_iters: int = 10
    lr: float = 0.01
    guide_scale: float = 5.0
    l_null: int = 16


@dataclasses.dataclass(frozen=True)
class ArmResults:
    """Everything one batch produced. Arrays are host arrays, so the result is jax-free downstream.

    ``final_latents`` is keyed by method: ``[B, C, F, H, W]`` for the single-noise arms and
    ``[len(PROBE_K_SET), B, C, F, H, W]`` for the probes. ``nulls`` / ``z_start`` /
    ``per_step_final_losses`` / ``diagnostics`` are keyed by the two *record* arms only ("a1", "a2"),
    which are the arms whose outputs can be cached; ``per_step_final_losses`` is the ``[B, N]``
    post-inner-loop tracking loss and ``diagnostics[arm]`` carries the optimizer's own ``[N, J, B]``
    ``tracking_losses`` and ``grad_norms`` traces (plan §4-P1 requires both to be logged).

    The last three fields **bind the result to the run that produced it**: the exact recipe, the
    fingerprint of the T5("") context the nulls were optimized against, and a content fingerprint of
    the batch. Without them a record writer can only check that names line up, and results from one
    run can be published under another run's header (Codex R6 review, finding 1).
    """

    names: tuple[str, ...]
    metrics: dict[str, dict[str, dict[str, dict[str, float]]]]
    final_latents: dict[str, np.ndarray]
    nulls: dict[str, np.ndarray]
    z_start: dict[str, np.ndarray]
    per_step_final_losses: dict[str, np.ndarray]
    diagnostics: dict[str, dict[str, np.ndarray]]
    params: CapacityParams
    base_context_fingerprint: str
    batch_fingerprint: str


@dataclasses.dataclass(frozen=True)
class RecipeScore:
    """One (J, lr) cell of the adequacy grid; ``score`` is the median of ``per_example`` (plan M3).

    ``tracking_losses`` and ``grad_norms`` are the optimizer's ``[N, J, B]`` traces -- the
    per-inner-iteration diagnostics the adequacy probe exists to produce -- and ``final_losses`` is
    the distinct ``[B, N]`` post-inner-loop tracking loss the score is computed from.
    """

    inner_iters: int
    lr: float
    score: float
    per_example: tuple[float, ...]
    tracking_losses: np.ndarray
    grad_norms: np.ndarray
    final_losses: np.ndarray


@dataclasses.dataclass(frozen=True)
class AdequacyReport:
    names: tuple[str, ...]
    scores: tuple[RecipeScore, ...]


@dataclasses.dataclass(frozen=True)
class RecipeAdoption:
    inner_iters: int
    lr: float
    adopted: bool
    plateau: str
    reasons: tuple[str, ...]
    numbers: dict[str, Any]


def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _checked(array: Any, label: str, shape: tuple[int, ...]) -> jnp.ndarray:
    """Fail closed at the production shape, in float32, before anything reaches the model."""
    value = jnp.asarray(array)
    if value.shape != shape:
        raise ValueError(f"{label} must have the production shape {shape}, got {value.shape}")
    if not bool(jnp.all(jnp.isfinite(value))):
        raise ValueError(f"{label} must be finite")
    return value.astype(jnp.float32)


def _validate_recipe(inner_iters: Any, lr: Any) -> None:
    """The R3 contract, restated here so an unusable recipe costs nothing instead of an inversion.

    ``optimize_null_embeddings`` rejects these too, but only once the arms are already running: the
    inversion has been paid for by then (Codex R6 review, finding 4).
    """
    if not _is_int(inner_iters) or inner_iters < 1:
        raise ValueError(f"inner_iters must be an integer >= 1, got {inner_iters!r}")
    if not np.isfinite(lr) or float(lr) < 0.0:
        raise ValueError(f"lr must be finite and non-negative, got {lr}")


def _validate(batch: CapacityBatch, base_context: Any, params: CapacityParams):
    names = tuple(batch.names)
    if not names:
        raise ValueError("batch must carry at least one example")
    if len(set(names)) != len(names) or not all(isinstance(name, str) and name for name in names):
        # Names key the noise draws and the metric tables; a duplicate silently collapses an example.
        raise ValueError(f"batch names must be unique non-empty strings, got {list(names)}")
    context = _checked(base_context, "base_context", PRODUCTION_CONTEXT_SHAPE)
    z_i0 = _checked(batch.z_i0, "z_i0", (len(names), *PRODUCTION_GEOMETRY.z_i0))
    z_video = _checked(batch.z_video, "z_video", (len(names), *PRODUCTION_GEOMETRY.z_video))
    if not _is_int(params.l_null) or not 1 <= params.l_null <= context.shape[0]:
        raise ValueError(f"l_null must be an integer in [1, {context.shape[0]}], got {params.l_null!r}")
    if not np.isfinite(params.guide_scale):
        raise ValueError(f"guide_scale must be finite, got {params.guide_scale}")
    _validate_recipe(params.inner_iters, params.lr)
    return names, z_i0, z_video, context


def batch_fingerprint(names: Sequence[str], z_i0: Any, z_video: Any) -> str:
    """Content fingerprint of a batch: sha256 over its names and their tensors, in name order.

    Name order rather than batch order, so re-ordering a batch is the same batch. Each tensor enters
    through ``null_adapter_records.sha256_of_array``, i.e. the codec's own byte discipline
    (contiguous, explicitly little-endian), and each name is NUL-terminated so no two name lists can
    concatenate to the same preimage. Float32 canonicalization matches what ``_validate`` returns, so
    the fingerprint of a batch is stable regardless of the dtype it arrived in.
    """
    digest = hashlib.sha256()
    z_i0, z_video = np.asarray(z_i0, np.float32), np.asarray(z_video, np.float32)
    for index in sorted(range(len(names)), key=lambda position: names[position]):
        digest.update(names[index].encode("utf-8") + b"\0")
        for array in (z_i0[index], z_video[index]):
            digest.update(bytes.fromhex(sha256_of_array(array)))
    return digest.hexdigest()


def _checked_trace(array: Any, label: str) -> np.ndarray:
    """Diagnostics are evidence, so a NaN in them fails the run instead of riding along in a report."""
    value = np.asarray(array)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite: a diagnostic trace carrying NaN or inf is not evidence")
    return value


def _mse(latents: jnp.ndarray, target: jnp.ndarray, *, drop_frame_zero: bool) -> jnp.ndarray:
    """Per-example MSE over the trailing ``[C, F, H, W]`` axes, so probe stacks reduce the same way."""
    if drop_frame_zero:
        latents, target = latents[..., 1:, :, :], target[..., 1:, :, :]
    return jnp.mean((latents - target) ** 2, axis=(-4, -3, -2, -1))


def _metric_tables(names: Sequence[str], latents: Mapping[str, jnp.ndarray], z_video: jnp.ndarray) -> dict:
    tables = {}
    for method, value in latents.items():
        stacked = value if value.ndim == 6 else value[None]  # [K, B, C, F, H, W]
        keys = [str(k) for k in PROBE_K_SET] if value.ndim == 6 else [SINGLE_SEED_KEY]
        future = np.asarray(_mse(stacked, z_video[None], drop_frame_zero=True))
        full = np.asarray(_mse(stacked, z_video[None], drop_frame_zero=False))
        tables[method] = {
            name: {
                key: {PRIMARY_METRIC: float(future[seed, index]), SECONDARY_METRIC: float(full[seed, index])}
                for seed, key in enumerate(keys)
            }
            for index, name in enumerate(names)
        }
    return tables


def _tracking_losses(z_bar: jnp.ndarray, traj: jnp.ndarray) -> jnp.ndarray:
    """``[B, N]`` post-inner-loop tracking loss: the optimizer's own objective at the locked nulls.

    Not ``losses[:, -1]`` from ``optimize_null_embeddings``, which is logged *before* the last update.
    """
    return jnp.mean((z_bar[1:] - traj[1:]) ** 2, axis=(2, 3, 4, 5)).T


def _keyed_batch(names: Sequence[str], k: int) -> jnp.ndarray:
    return jnp.stack([keyed_noise(name, k) for name in names])


def _requested_methods(arms: Sequence[str] | None) -> tuple[str, ...]:
    """Which of ``METHODS`` to compute, in ``METHODS`` order; ``None`` means the whole §4-P1 study."""
    if arms is None:
        return METHODS
    requested = tuple(arms)
    unknown = sorted(set(requested) - set(METHODS))
    if not requested or unknown:
        raise ValueError(f"arms must be a non-empty subset of {list(METHODS)}, got {list(requested)}")
    return tuple(method for method in METHODS if method in set(requested))


def run_capacity_example_batch(
    velocity_fn: Callable[..., jnp.ndarray],
    batch: CapacityBatch,
    base_context: Any,
    params: CapacityParams = CapacityParams(),
    *,
    arms: Sequence[str] | None = None,
) -> ArmResults:
    """Run the plan §4-P1 arms on one batch, inverting exactly once.

    Args:
      velocity_fn: ``(latents, timestep_2d, encoder_hidden_states) -> v``, the R3/R4a seam. Inversion
        runs at w=1, where CFG collapses to the conditional branch, so it is called with the base
        context alone rather than through a second seam.
      batch: names plus ``z_i0`` ``[B, 48, 1, 12, 20]`` and ``z_video`` ``[B, 48, 9, 12, 20]``.
      base_context: the ``[512, 4096]`` T5("") context.
      params: the shared recipe (J, lr, w, L_null).
      arms: which of ``METHODS`` to compute, defaulting to all six. **The capacity study wants all
        six** -- the gates compare them against each other. P2's caching job wants exactly one, and
        running the other five there buys a second full null optimization plus eight replays per
        batch for outputs nothing reads (R10 review, finding 4). Only what a requested arm needs is
        computed: the inversion is always paid for, ``eps_0``, the frozen nulls and either
        optimization are not.

    Returns:
      ``ArmResults`` -- metric tables in the gates' schema plus the artifacts records are built from.
      Every per-arm dictionary carries exactly the requested arms, so a caller that asks for one arm
      and then reads another gets a ``KeyError`` rather than a stale array from somewhere else.
    """
    methods = _requested_methods(arms)
    names, z_i0, z_video, context = _validate(batch, base_context, params)
    sigmas = jnp.asarray(canonical_sigmas())
    steps = sigmas.shape[0] - 1
    null_init = context[: params.l_null]
    cond = jnp.broadcast_to(context, (len(names), *context.shape))

    traj = invert_trajectory(lambda z, t: velocity_fn(z, t, cond), z_video, z_i0, sigmas)

    # One recipe object for both optimized arms: A1 and A2 must differ only in where they start, or
    # the G2 comparison would be confounded by the hyperparameters as well as by the basin.
    recipe = {"inner_iters": params.inner_iters, "lr": params.lr, "guide_scale": params.guide_scale}

    def replay(z_start, nulls):
        return replay_with_nulls(velocity_fn, z_start, z_i0, sigmas, nulls, context, guide_scale=params.guide_scale)

    def optimize(pivots):
        nulls, z_bar, losses, norms = optimize_null_embeddings(
            velocity_fn, pivots, z_i0, sigmas, null_init, context, **recipe
        )
        traces = {
            "tracking_losses": _checked_trace(losses, "tracking_losses"),
            "grad_norms": _checked_trace(norms, "grad_norms"),
        }
        return nulls, traces, _checked_trace(_tracking_losses(z_bar, traj), "per_step_final_losses")

    wanted = set(methods)
    frozen = jnp.broadcast_to(null_init, (steps, params.l_null, context.shape[1])) if {"a0", "a2_0"} & wanted else None
    eps_0 = (
        jnp.broadcast_to(global_noise(0), (len(names), *PRODUCTION_GEOMETRY.z_video))
        if {"a2", "a2_0", "a2_probe"} & wanted
        else None
    )

    nulls, z_start, per_step, diagnostics = {}, {}, {}, {}
    if {"a1", "a1_probe"} & wanted:
        nulls["a1"], diagnostics["a1"], per_step["a1"] = optimize(traj)
        z_start["a1"] = traj[0]
    if {"a2", "a2_probe"} & wanted:  # both are in eps_0's trigger set, so it exists here
        # Same targets, eps_0 as the starting pivot.
        nulls["a2"], diagnostics["a2"], per_step["a2"] = optimize(traj.at[0].set(eps_0))
        z_start["a2"] = eps_0

    builders = {
        "a0": lambda: replay(traj[0], frozen),
        "a1": lambda: replay(traj[0], nulls["a1"]),
        "a2": lambda: replay(z_start["a2"], nulls["a2"]),
        "a2_0": lambda: replay(eps_0, frozen),
        "a1_probe": lambda: jnp.stack([replay(_keyed_batch(names, k), nulls["a1"]) for k in PROBE_K_SET]),
        "a2_probe": lambda: jnp.stack([replay(_keyed_batch(names, k), nulls["a2"]) for k in PROBE_K_SET]),
    }
    latents = {method: builders[method]() for method in methods}
    return ArmResults(
        names=names,
        metrics=_metric_tables(names, latents, z_video),
        final_latents={method: np.asarray(value) for method, value in latents.items()},
        nulls={arm: np.asarray(value) for arm, value in nulls.items()},
        z_start={arm: np.asarray(value) for arm, value in z_start.items()},
        per_step_final_losses=per_step,
        diagnostics=diagnostics,
        params=params,
        base_context_fingerprint=base_context_fingerprint(context),
        batch_fingerprint=batch_fingerprint(names, z_i0, z_video),
    )


def emit_metric_tables(arm_results: ArmResults) -> dict[str, dict]:
    """The per-method tables ``null_adapter_gates`` consumes, as plain JSON-serializable dicts.

    Single-noise arms (a0/a1/a2/a2_0) carry the one seed key ``"0"`` and are gated under
    ``NoiseConvention.GLOBAL``; the probes carry ``PROBE_K_SET`` and are reduced under
    ``NoiseConvention.KEYED``. ``future_ssim`` is absent until R7 decodes -- see the module docstring:
    the gates read that as *invalid*, so no verdict from these tables is meaningful before R7.
    """
    tables = {}
    for method in METHODS:
        table = arm_results.metrics.get(method)
        if not isinstance(table, Mapping) or sorted(table) != sorted(arm_results.names):
            raise ValueError(f"the {method!r} table does not cover the batch exactly: {sorted(table or {})}")
        keys = [str(k) for k in PROBE_K_SET] if method in PROBE_METHODS else [SINGLE_SEED_KEY]
        for name in arm_results.names:
            if sorted(table[name]) != sorted(keys):
                raise ValueError(f"the {method!r} table has the wrong seed keys for {name!r}: {sorted(table[name])}")
        tables[method] = {
            name: {key: {metric: float(value) for metric, value in table[name][key].items()} for key in keys}
            for name in arm_results.names
        }
    return tables


def build_capacity_records(
    velocity_fn: Callable[..., jnp.ndarray],
    arm_results: ArmResults,
    batch: CapacityBatch,
    base_context: Any,
    header: ProvenanceHeader,
    example_fields: Mapping[str, Mapping[str, Any]],
    *,
    arm: str,
) -> list[NullAdapterRecord]:
    """Cache one arm's optimized nulls as verifiable records: **cast -> replay -> record**.

    The header supplies the storage dtype, the guidance weight, the sigma grid and ``l_null``, and is
    checked against this run before anything is built -- a record that could not verify under its own
    header is never written. ``example_fields`` carries the per-example manifest data the arms do not
    see (``ordinal``, ``split``, ``episode``, ``actions``).

    ``final_future_mse`` is the arm's own fp32 measurement, i.e. the same number the metric tables
    report, rather than a re-measurement of the fp16-rounded stored tensors.

    **Nothing here is taken on the caller's word.** ``arm_results`` binds the recipe, the context and
    the batch it was produced from, and every one of those is required to agree with the header and
    with the arrays passed in *before* a single forward: otherwise nulls optimized at (J=1, lr=1e-2,
    w=5) can be published under a header advertising (J=50, lr=3e-2, w=7), and a batch that merely
    reuses the names carries someone else's ``z_video`` and metrics into the cache -- both of which
    still verify, because ``verify_replay`` deliberately checks the record against itself.
    """
    if arm not in RECORD_ARMS:
        raise ValueError(f"arm must be one of {sorted(RECORD_ARMS)}, got {arm!r}")
    label, convention = RECORD_ARMS[arm]
    header_sigmas = _validate_header(header)  # the writer's own header contract, reused not restated
    recipe = arm_results.params
    names, z_i0, z_video, context = _validate(batch, base_context, recipe)

    if not np.array_equal(header_sigmas, canonical_sigmas()):
        raise ValueError("header sigma_vector does not match the canonical grid: these records could never verify")
    if base_context_fingerprint(context) != header.base_context_fingerprint:
        raise ValueError('header base_context_fingerprint does not match this base_context: a different T5("")')
    if arm_results.base_context_fingerprint != header.base_context_fingerprint:
        raise ValueError("arm_results were produced against a different base_context than the header declares")
    if tuple(arm_results.names) != names:
        raise ValueError(
            f"arm_results and batch describe different examples: {list(arm_results.names)} vs {list(names)}"
        )
    if arm_results.batch_fingerprint != batch_fingerprint(names, z_i0, z_video):
        raise ValueError("batch fingerprint does not match arm_results: the same names carry different tensors")
    if float(header.guide_scale) != float(recipe.guide_scale):
        raise ValueError(f"header guide_scale {header.guide_scale} does not match the run's {recipe.guide_scale}")
    if header.l_null != recipe.l_null:
        raise ValueError(f"header l_null {header.l_null} does not match the run's l_null {recipe.l_null}")
    declared = {key: header.optimization_config.get(key) for key in ("inner_iters", "lr")}
    if declared != {"inner_iters": recipe.inner_iters, "lr": float(recipe.lr)}:
        raise ValueError(
            f"header optimization_config {declared} does not match the run's recipe "
            f"{{'inner_iters': {recipe.inner_iters}, 'lr': {float(recipe.lr)}}}"
        )
    missing = [name for name in names if sorted(example_fields.get(name, {})) != sorted(RECORD_FIELDS)]
    if missing:
        raise ValueError(
            f"example_fields must carry exactly {list(RECORD_FIELDS)} for every name; wrong for {missing}"
        )
    nulls = np.asarray(arm_results.nulls[arm])
    if nulls.shape[2] != header.l_null:
        raise ValueError(f"header l_null {header.l_null} does not match the optimized nulls of shape {nulls.shape}")

    # Cast first: the record's own bytes are what the replay must start from (R4c's writer contract).
    latent_dtype = LATENT_DTYPES[header.dtype_policy]
    stored_nulls = nulls.astype(latent_dtype)
    stored_z_start = np.asarray(arm_results.z_start[arm]).astype(latent_dtype)
    stored_z_i0 = np.asarray(z_i0).astype(SOURCE_DTYPES["z_i0"])  # fp16 under every dtype policy
    expected = np.asarray(
        replay_with_nulls(
            velocity_fn,
            jnp.asarray(stored_z_start, jnp.float32),
            jnp.asarray(stored_z_i0, jnp.float32),
            jnp.asarray(header_sigmas, jnp.float32),
            jnp.asarray(stored_nulls, jnp.float32),
            context,
            guide_scale=float(header.guide_scale),
        )
    )
    return [
        make_record(
            name=name,
            **{field: example_fields[name][field] for field in RECORD_FIELDS},
            z_i0=stored_z_i0[index],
            z_video=np.asarray(z_video[index]),
            latent_dtype=header.dtype_policy,
            nulls=stored_nulls[:, index],
            z_start=stored_z_start[index],
            expected_final_latent=expected[index],
            noise_convention=convention,
            arm=label,
            per_step_final_losses=np.asarray(arm_results.per_step_final_losses[arm][index], np.float32),
            final_future_mse=arm_results.metrics[arm][name][SINGLE_SEED_KEY][PRIMARY_METRIC],
        )
        for index, name in enumerate(names)
    ]


def _validate_grid(grid: Sequence[tuple[int, float]]) -> tuple[tuple[int, float], ...]:
    recipes = tuple((inner_iters, float(lr)) for inner_iters, lr in grid)
    if not recipes or len(set(recipes)) != len(recipes):
        raise ValueError(f"the adequacy grid must be non-empty with distinct recipes, got {list(grid)}")
    for inner_iters, lr in recipes:  # each cell must be runnable before the shared inversion is paid for
        _validate_recipe(inner_iters, lr)
    return recipes


def run_adequacy_probe(
    velocity_fn: Callable[..., jnp.ndarray],
    probe_batch: CapacityBatch,
    base_context: Any,
    grid: Sequence[tuple[int, float]] = ADEQUACY_GRID,
    *,
    guide_scale: float,
    l_null: int,
) -> AdequacyReport:
    """Score each (J, lr) recipe on the same examples and the same pivots (plan §4-P1 F1/M3).

    Per example, ``s_e`` is the mean over sampler steps of the post-inner-loop tracking loss; a
    recipe's score is the median of ``s_e`` over the probe examples. The inversion is shared across
    the whole grid -- the pivots do not depend on the recipe, and re-deriving them per cell would
    compare recipes at different (numerically re-derived) targets as well as at different recipes.
    """
    recipes = _validate_grid(grid)
    params = CapacityParams(guide_scale=guide_scale, l_null=l_null)
    names, z_i0, z_video, context = _validate(probe_batch, base_context, params)
    sigmas = jnp.asarray(canonical_sigmas())
    cond = jnp.broadcast_to(context, (len(names), *context.shape))

    traj = invert_trajectory(lambda z, t: velocity_fn(z, t, cond), z_video, z_i0, sigmas)
    scores = []
    for inner_iters, lr in recipes:
        recipe = {"inner_iters": inner_iters, "lr": lr, "guide_scale": guide_scale}
        _, z_bar, losses, norms = optimize_null_embeddings(
            velocity_fn, traj, z_i0, sigmas, context[:l_null], context, **recipe
        )
        final_losses = _checked_trace(_tracking_losses(z_bar, traj), "final_losses")
        per_example = final_losses.mean(axis=1)
        scores.append(
            RecipeScore(
                inner_iters=inner_iters,
                lr=lr,
                score=float(np.median(per_example)),
                per_example=tuple(map(float, per_example)),
                tracking_losses=_checked_trace(losses, "tracking_losses"),
                grad_norms=_checked_trace(norms, "grad_norms"),
                final_losses=final_losses,
            )
        )
    return AdequacyReport(names, tuple(scores))


def _plateau(scores: Mapping[tuple[int, float], RecipeScore], lr: float) -> tuple[str, float | None]:
    coarse, fine = (scores.get((inner_iters, lr)) for inner_iters in PLATEAU_PAIR)
    usable = coarse is not None and fine is not None and math.isfinite(coarse.score) and math.isfinite(fine.score)
    if not usable or coarse.score <= 0.0:  # no J=25/J=50 pair at this lr, or nothing to divide by
        return UNDETERMINED, None
    improvement = (coarse.score - fine.score) / coarse.score
    # Boundary-aware: the plan gives "reconstruction-limited" to improvements strictly *below* 10%, so
    # exactly 10% must land on the other side -- and (1.0 - 0.9)/1.0 evaluates to 0.09999999999999998
    # in binary floating point. Anything within PLATEAU_BOUNDARY_ATOL of the threshold is read as
    # being at it. The epsilon is 1e-9 on a 0.1 threshold, i.e. eight orders below the smallest
    # difference these float32-derived scores can meaningfully express.
    return (
        RECONSTRUCTION_LIMITED if improvement < PLATEAU_MIN_IMPROVEMENT - PLATEAU_BOUNDARY_ATOL else RECIPE_LIMITED
    ), improvement


def adopt_recipe(report: AdequacyReport) -> RecipeAdoption:
    """The plan's decidable adoption rule (M3), plus the plateau reading of a failure.

    Adopt the lowest-scoring recipe among those at or below ``ADOPTION_FACTOR`` x the default
    recipe's score; break ties towards the cheaper recipe (lower J, then lower lr); keep the default
    if nothing qualifies. The plateau is then read off the adopted recipe's learning-rate column:
    less than ``PLATEAU_MIN_IMPROVEMENT`` from J=25 to J=50 means more optimization is not the
    missing ingredient ("reconstruction-limited"), otherwise the recipe was the limit.
    """
    scores = {(score.inner_iters, score.lr): score for score in report.scores}
    if len(scores) != len(report.scores):
        raise ValueError(f"the adequacy report contains duplicate recipes: {[s.inner_iters for s in report.scores]}")
    if DEFAULT_RECIPE not in scores:
        raise ValueError(f"the adequacy report must contain the default recipe {DEFAULT_RECIPE}")
    default = scores[DEFAULT_RECIPE]
    if not math.isfinite(default.score):
        raise ValueError(f"the default recipe's score must be finite, got {default.score}")

    threshold = ADOPTION_FACTOR * default.score
    qualifying = [s for s in report.scores if math.isfinite(s.score) and s.score <= threshold]
    chosen = min(qualifying, key=lambda s: (s.score, s.inner_iters, s.lr)) if qualifying else default
    plateau, improvement = _plateau(scores, chosen.lr)
    verb = "adopted" if qualifying else "no recipe qualified; keeping the default"
    reasons = (f"{verb} J={chosen.inner_iters}, lr={chosen.lr:g} at score {chosen.score:.6g}", f"plateau: {plateau}")
    numbers = {
        "threshold": threshold,
        "default_score": default.score,
        "adopted_score": chosen.score,
        "plateau_improvement": improvement,
        "scores": {f"J{s.inner_iters}_lr{s.lr:g}": s.score for s in report.scores},
    }
    return RecipeAdoption(chosen.inner_iters, chosen.lr, bool(qualifying), plateau, reasons, numbers)
