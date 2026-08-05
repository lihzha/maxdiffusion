"""exp_03 Mechanism B — the SIGMA-TRAJECTORY trace (plan v3.1, review P7).

D1 measures temporal decay of decoded frames. It cannot, on its own, establish that *sampler
compounding* was reduced — that is a statement about the denoising trajectory, and this is the
instrument for it. At every step of the 25-step rollout the trace records how far the sampler's
state has drifted from the ideal interpolant it would be on if the velocity were perfect::

    error_i = || z_{sigma_i} - ((1 - sigma_i) * z_gt + sigma_i * eps) ||^2 / numel

with frame 0 pinned on both sides, exactly as the rollout pins it. Two properties make this
readable rather than merely suggestive:

* **It starts at exactly zero.** The eval's initial state is ``eps`` (``sigma_0 = 1``), which *is*
  the interpolant at ``sigma_0``, so index 0 has no arithmetic in it at all.
* **A perfect velocity keeps it zero — to the arithmetic's precision.** With ``v = eps - z_gt`` the
  Euler rule maps the interpolant at ``sigma_i`` onto the interpolant at ``sigma_{i+1}``
  *algebraically*. In float32 that is ~1e-14 and the growth in the trace really is model error and
  nothing else. In **bfloat16** — the production ``weights_dtype`` — the incremental accumulation
  does not reproduce the interpolant exactly, so a finite-precision FLOOR exists. It is therefore
  measured rather than assumed: :func:`oracle_floor` re-runs each window with the exact velocity and
  the result travels in the JSON as ``floor``, in the same entry as the ``error`` it bounds.

  **Predeclared reporting choice:** the reported metric is the **RAW** error, with the floor
  alongside; subtracting it is an analysis decision to be stated when the numbers are read, not a
  correction applied silently here. The reference interpolant is evaluated in float32
  (:data:`TRACE_REFERENCE_DTYPE`) so the reference contributes no rounding of its own.

**It traces the operator the eval runs.** The step comes from
``models/wan/overfit100_sampling.py`` — the same function ``generate_wan_side_adapter`` calls and
the exp_03 trainers unroll (plan §3, one-sampler rule). The eval rollout itself is NOT modified;
this is its own harness on the shared step.

**The design is approval-pinned** (:func:`assert_approved_trace_design`, checked before the ~5B
load): cohort seed 0, exactly 30 windows, exactly the settled 25-step sampler, with the resulting
cohort and grid lengths verified afterwards. Constants in a module do not constrain a run whose
config supplies the values, and a trace taken on another cohort or another sampler is a different
measurement that would corrupt the cross-arm comparison instead of announcing itself.

**Isolation, inherited from the exp_02 probe review.** Same 30-window seeded cohort (reusing
``probe_overfit100_sampling_steps.probe_cohort``), same per-window key
``window_fold_key(0, episode_id, window_start)``, one immutable JSON per checkpoint under the run's
canonical ``validation_probe_sampling/`` root, and a hard refusal of any path component that names a
verdict role directory. It writes no role, no cohort denominator and no aggregation schema, so it
cannot be mistaken for admissible evidence.
"""

from __future__ import annotations

import statistics
from typing import Sequence

import jax
import jax.numpy as jnp
from absl import app
from flax.linen import partitioning as nn_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.probe_overfit100_sampling_steps as probe
from maxdiffusion import max_logging, pyconfig
from maxdiffusion.models.wan.overfit100_sampling import (
    overfit100_sampler_grid,
    overfit100_sampler_step,
)
from maxdiffusion.models.wan.side_adapter_wan import apply_first_frame_pin

TRACE_SCHEMA = "exp03_sigma_trajectory_trace_v1"
TRACE_OUTPUT_DIR = probe.PROBE_OUTPUT_DIR  # the same non-role diagnostic root
TRACE_CONTEXT_MODE = "correct"
TRACE_SEED = 0  # the eval's own seed-0 keying, so the trace lines up with the landed rows
TRACE_NUM_WINDOWS = 30  # the exp_02 probe's cohort, reused so the two diagnostics are comparable
TRACE_SAMPLING_STEPS = 25  # the settled eval sampler; a trace of another sampler is another metric

# The reference interpolant is evaluated in FLOAT32 regardless of the rollout's latent dtype, and
# it is documented here because it is a choice, not an accident: the reference then carries no
# rounding of its own, so the trace is "how far the (possibly bf16) rollout drifted from the exact
# trajectory" rather than a difference of two approximations.
TRACE_REFERENCE_DTYPE = jnp.float32


def assert_approved_trace_design(*, seed: int, num_windows: int, sampling_steps: int) -> None:
    """Refuse anything but the approved trace (seed 0, 30 windows, the 25-step eval sampler).

    The exp_02 probe review's lesson, applied before the ~5B load: constants in a module do not
    constrain a run whose config supplies the values. A trace taken on a different cohort or a
    different sampler is a different measurement, and Mechanism B's readings are compared across
    arms -- so a silent substitution would corrupt the comparison rather than announce itself.
    """
    if int(seed) != TRACE_SEED:
        raise ValueError(
            f"the sigma trace is pinned to cohort seed {TRACE_SEED} (the eval's own seed-0 keying), got {int(seed)}; "
            f"another seed selects other windows and the arms would no longer be comparable"
        )
    if int(num_windows) != TRACE_NUM_WINDOWS:
        raise ValueError(
            f"the sigma trace is pinned to the {TRACE_NUM_WINDOWS}-window probe cohort, got {int(num_windows)}"
        )
    if int(sampling_steps) != TRACE_SAMPLING_STEPS:
        raise ValueError(
            f"the sigma trace is pinned to the settled {TRACE_SAMPLING_STEPS}-step eval sampler, got "
            f"{int(sampling_steps)}; tracing another sampler measures another trajectory"
        )


def trace_grid(*, num_inference_steps: int, flow_shift: float, sigma_min: float = 0.0, sigma_max: float = 1.0):
    """The eval's ``(sigmas, timesteps)`` — built by the shared sampler module, never re-derived."""
    return overfit100_sampler_grid(
        num_inference_steps=num_inference_steps,
        flow_shift=flow_shift,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_train_timesteps=1000,
    )


def ideal_interpolant(z_gt: jax.Array, eps: jax.Array, sigma: float, z_i0: jax.Array) -> jax.Array:
    """``pin((1 - sigma) * z_gt + sigma * eps)`` — where a perfect sampler would be at ``sigma``.

    Evaluated in :data:`TRACE_REFERENCE_DTYPE` (float32), so the reference is the exact trajectory
    to the precision of the trace's own arithmetic even when the rollout runs in bfloat16.
    """
    value = (1.0 - float(sigma)) * z_gt.astype(TRACE_REFERENCE_DTYPE) + float(sigma) * eps.astype(
        TRACE_REFERENCE_DTYPE
    )
    return apply_first_frame_pin(value, z_i0.astype(TRACE_REFERENCE_DTYPE))


def interpolant_error(z: jax.Array, z_gt: jax.Array, eps: jax.Array, sigma: float, z_i0: jax.Array) -> float:
    """``|| z - ideal ||^2 / numel`` at one sigma."""
    diff = z.astype(TRACE_REFERENCE_DTYPE) - ideal_interpolant(z_gt, eps, sigma, z_i0)
    return float(jnp.mean(diff**2))


def trace_noise_key(*, episode_id: int, window_start: int) -> jax.Array:
    """The EVAL's per-window key — identical draw, so the trace starts where the rollout starts."""
    return gen.window_fold_key(TRACE_SEED, int(episode_id), int(window_start))


def trace_noise(*, episode_id: int, window_start: int, shape, dtype=jnp.float32) -> jax.Array:
    """The fixed epsilon for one window (the rollout's own initial noise)."""
    return jax.random.normal(trace_noise_key(episode_id=episode_id, window_start=window_start), shape, dtype=dtype)


def sigma_trace(
    velocity_fn,
    *,
    z_gt: jax.Array,
    z_i0: jax.Array,
    eps: jax.Array,
    context: jax.Array,
    sigmas: jax.Array,
    timesteps: jax.Array,
) -> list[float]:
    """Interpolant error at every grid index, ``0 .. N`` (``N + 1`` values).

    The rollout is re-run here rather than instrumented in place: the eval path stays untouched, and
    the loop is a plain Python loop because it runs once per window over 25 steps and must hand back
    a value per step.
    """
    num_steps = int(sigmas.shape[0]) - 1
    z = apply_first_frame_pin(eps, z_i0)
    errors = [interpolant_error(z, z_gt, eps, float(sigmas[0]), z_i0)]
    for index in range(num_steps):
        z = overfit100_sampler_step(
            z,
            index,
            velocity_fn=velocity_fn,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
        )
        errors.append(interpolant_error(z, z_gt, eps, float(sigmas[index + 1]), z_i0))
    return errors


def batch_replicas(config, mesh=None) -> int:
    """How many identical rows ONE trace forward needs for the batch axis to divide evenly.

    Derived from the run's own ``logical_axis_rules`` -- the mesh axes that ``batch`` maps to --
    rather than guessed: on the v6e-8 probe that is ``data x fsdp = 8``. Falls back to the device
    count when the rule is absent, and returns 1 on a single-device host, which is what makes this a
    no-op there and leaves any trace already taken byte-identical.
    """
    axes: tuple = ()
    for rule in getattr(config, "logical_axis_rules", ()) or ():
        if len(rule) == 2 and rule[0] == "batch":
            axes = tuple(rule[1]) if isinstance(rule[1], (list, tuple)) else (rule[1],)
            break
    if mesh is None or not axes:
        return max(1, int(jax.device_count()))
    size = 1
    for axis in axes:
        size *= int(dict(mesh.shape).get(axis, 1))
    return max(1, size)


def _tile_rows(array, replicas: int):
    """``array`` with its single leading row repeated ``replicas`` times."""
    if array is None or int(array.shape[0]) != 1 or replicas <= 1:
        return array
    return jnp.broadcast_to(array, (replicas,) + tuple(array.shape[1:]))


def jitted_tiled_forward(state, *, replicas: int = 1):
    """ONE compiled forward for the whole trace: the window TILED to ``replicas`` rows.

    THE Job 8f failure, and the wall-time hazard behind it. Every other phase of S1.5 runs the
    global batch -- eight examples, one per chip -- but the sigma trace is inherently one window at
    a time, and the transformer annotates its activations ``P(('data','fsdp'), 'context',
    'tensor')``. One row cannot be split eight ways, so the first trace forward the probe ever
    reached raised ``IndivisibleError`` at ``transformer_wan.py:401`` with the entire
    checkpoint-state report already computed behind it.

    Tiling rather than re-annotating: the constraint is production's, it is right for production,
    and a diagnostic has no business relaxing the sharding of the model it exists to measure.

    And compiling does NOT make the tiling redundant, which is the subtler half of why it stays.
    Inside one jit the batch-1 constraint stops raising -- GSPMD satisfies it by replicating a
    size-1 dimension -- so an untiled compiled trace would run perfectly happily under PARTIAL
    REPLICATION. That is a different sharding regime from the genuinely batch-partitioned one
    production runs under: the trace would be measuring the model through an execution the
    experiment never uses, and saying nothing about it. Tiling puts one real row on each chip, which
    is the regime the numbers are meant to describe.

    COMPILED, not eager, and that is the second half of the fix. The trace is ~750 model forwards
    per state, each traversing 40 ``nnx.remat``-wrapped blocks; run op by op that is a dispatch
    storm of unbounded duration with no way to size a retry. Compiled once and reused for every step
    of every window it is one compilation plus 750 executions of a graph XLA has already scheduled.

    The parameters are an ARGUMENT, never a closure capture: a jitted function closing over an
    ``nnx`` module turns ~5B weights into compile-time literals. This is the shape every other
    compiled function here uses -- ``nnx.merge(graphdef, params, rest)`` inside, ``graphdef`` static
    and ``params`` traced -- and the inputs are fed exactly as the replay feeds them, at their
    natural shardings, with no ``in_shardings`` of its own.

    No ``out_shardings``, deliberately: unlike a gradient this output is latent-shaped -- eight rows
    of one window, under 2 MB -- so however XLA lays it out costs nothing worth pinning.

    Reading row 0 (in :func:`velocity_fn_for`, OUTSIDE this compiled boundary) is EXACT because
    nothing in this transformer mixes information across batch rows, verified in the code rather
    than assumed: the normalizations are ``FP32LayerNorm`` and ``RMSNorm`` reducing over the FEATURE
    axis (per token -- there are no batch statistics anywhere); attention folds heads into the
    leading axis and its einsums contract over sequence and head-dim only, with the softmax over one
    example's own keys; rotary embeddings are a batch-1 broadcast; dropout is configured to 0.0 and
    the probe passes ``deterministic=True`` besides; and the only collectives in the attention
    module ride the ``context`` (sequence) axis, inside kernels this 540-token path does not take.
    The remaining caveat is arithmetic rather than semantics: XLA may schedule reductions
    differently for a different batch shape, so row 0 could in principle differ from a batch-1
    reference in the last bits. Measured on an 8-device mesh it does not -- it is bitwise.
    """
    graphdef = state.graphdef

    def _forward(params, st, hidden_states, timestep, encoder_hidden_states):
        transformer = gen.nnx.merge(graphdef, params, st.rest_of_state)
        return transformer(
            hidden_states=_tile_rows(hidden_states, replicas),
            timestep=_tile_rows(timestep, replicas),
            encoder_hidden_states=_tile_rows(encoder_hidden_states, replicas),
            deterministic=True,
        )

    return jax.jit(_forward)


def velocity_fn_for(forward, state):
    """The sampler-facing velocity: call the compiled tiled forward, take row 0 OUTSIDE it.

    The slice lives here rather than inside the compiled function so that the compiled thing is
    exactly the forward -- one graph, reused ~750 times -- while the sampler goes on receiving the
    single window it handed in.
    """

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        return forward(state.params, state, hidden_states, timestep, encoder_hidden_states)[:1]

    return velocity_fn


def oracle_velocity_fn(z_gt: jax.Array, eps: jax.Array):
    """The PERFECT velocity ``eps - z_gt``, in the latents' own dtype.

    Built in the latents' dtype on purpose: the floor it produces is the floor of the *production*
    arithmetic, not of an idealized float32 rehearsal of it.
    """
    exact = (eps - z_gt).astype(z_gt.dtype)

    def velocity_fn(hidden_states, timestep, encoder_hidden_states):
        del timestep, encoder_hidden_states
        return exact.astype(hidden_states.dtype)

    return velocity_fn


def oracle_floor(*, z_gt, z_i0, eps, context, sigmas, timesteps) -> list[float]:
    """The finite-precision FLOOR: what the trace reads when the velocity is exactly right.

    In float32 this is ~1e-14 and the trace is "model error, and nothing else". In bfloat16 -- the
    production ``weights_dtype`` -- the incremental Euler accumulation does NOT reproduce the
    interpolant exactly, so a nonzero floor exists and must be visible next to the number it
    contaminates. Costs one network-free rollout per window.
    """
    return sigma_trace(
        oracle_velocity_fn(z_gt, eps),
        z_gt=z_gt,
        z_i0=z_i0,
        eps=eps,
        context=context,
        sigmas=sigmas,
        timesteps=timesteps,
    )


def trace_rows(velocity_fn, *, z_gt, z_i0, eps, context, sigmas, timesteps) -> list[dict]:
    """:func:`sigma_trace` labelled by grid index and sigma, WITH the finite-precision floor.

    ``error`` is the RAW error -- the predeclared reported metric -- and ``floor`` travels beside it
    in the same entry (so the two can never be misaligned). Subtracting the floor is an analysis
    decision to be made and stated when the numbers are read, not a silent correction applied here.
    """
    errors = sigma_trace(
        velocity_fn, z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps
    )
    floor = oracle_floor(z_gt=z_gt, z_i0=z_i0, eps=eps, context=context, sigmas=sigmas, timesteps=timesteps)
    return [
        {"index": index, "sigma": float(sigmas[index]), "error": float(error), "floor": float(floor[index])}
        for index, error in enumerate(errors)
    ]


def trace_cohort(manifest_path: str, *, seed: int = TRACE_SEED, num_windows: int = TRACE_NUM_WINDOWS) -> list[dict]:
    """The exp_02 sampling probe's cohort selection, REUSED (not re-implemented)."""
    return probe.probe_cohort(manifest_path, seed=int(seed), num_windows=int(num_windows))


def trace_output_path(config, checkpoint_step: int) -> str:
    """``<output_dir>/<run_name>/validation_probe_sampling/sigma_trace_ckpt<step>.json``.

    Canonical only, with the exp_02 probe review's rule enforced: a diagnostic must never be written
    where the verdict's evidence lives, so any path component naming a ``step_<n>_<role>`` directory
    is refused outright.
    """
    root = f"{str(config.output_dir).rstrip('/')}/{config.run_name}/{TRACE_OUTPUT_DIR}"
    path = f"{root.rstrip('/')}/sigma_trace_ckpt{int(checkpoint_step)}.json"
    offenders = [part for part in path.split("/") if part.startswith("step_")]
    if offenders:
        raise ValueError(
            f"refusing the trace output path {path!r}: component(s) {offenders} name a verdict role directory. A "
            f"diagnostic must never be written where the verdict's evidence lives -- fix output_dir/run_name."
        )
    parts = path.split("/")
    if len(parts) < 2 or parts[-2] != TRACE_OUTPUT_DIR:
        raise ValueError(f"refusing the trace output path {path!r}: it is not directly under {TRACE_OUTPUT_DIR}/")
    return path


def mean_trace(rows: Sequence[dict]) -> list[dict]:
    """The cohort mean error at each grid index — the curve the arms are compared on."""
    if not rows:
        return []
    lengths = {len(row["trace"]) for row in rows}
    if len(lengths) != 1:
        raise ValueError(f"windows traced different step counts {sorted(lengths)}; the mean curve is not defined")
    length = lengths.pop()
    out = []
    for index in range(length):
        entries = [row["trace"][index] for row in rows]
        sigmas = {round(float(entry["sigma"]), 9) for entry in entries}
        if len(sigmas) != 1:
            raise ValueError(f"windows disagree about sigma at index {index}: {sorted(sigmas)}")
        entry = {
            "index": index,
            "sigma": float(entries[0]["sigma"]),
            "error": statistics.fmean(float(item["error"]) for item in entries),
        }
        if all("floor" in item for item in entries):
            entry["floor"] = statistics.fmean(float(item["floor"]) for item in entries)
        out.append(entry)
    return out


def trace_artifact(config, *, checkpoint_step: int, cohort, rows) -> dict:
    """The diagnostic JSON. Carries NO verdict fields — it is not admissible evidence."""
    return {
        "schema": TRACE_SCHEMA,
        "kind": "diagnostic",
        "note": (
            "Sigma-trajectory diagnostic (exp_03 Mechanism B). NOT an evaluation artifact: no role, no "
            "cohort denominator, no aggregation schema. It cannot enter the success statistic."
        ),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_dir": str(getattr(config, "checkpoint_dir", "")),
        "run_name": str(getattr(config, "run_name", "")),
        "commit": gen._eval_code_commit(),
        "eval_data_dir": str(getattr(config, "eval_data_dir", "")),
        "model_manifest_path": str(getattr(config, "model_manifest_path", "")),
        "context_mode": TRACE_CONTEXT_MODE,
        "rollout_seed": TRACE_SEED,
        "cohort_seed": int(getattr(config, "seed", 0)),
        "sampling_steps": TRACE_SAMPLING_STEPS,
        "reference_dtype": str(jnp.dtype(TRACE_REFERENCE_DTYPE)),
        "latent_dtype": str(getattr(config, "weights_dtype", "")),
        "reported_metric": (
            "RAW interpolant error; 'floor' beside it is the finite-precision oracle floor at the same "
            "index (a perfect velocity's reading). Subtract only as a stated analysis choice."
        ),
        "cohort": [str(window["name"]) for window in cohort],
        "mean_trace": mean_trace(rows),
        "rows": list(rows),
    }


def write_trace_artifact(path: str, payload: dict) -> None:
    """Immutable write, reusing the eval path's compare-or-refuse writer."""
    gen._write_json_immutable(path, payload)


def _log_mean_trace(curve: Sequence[dict]) -> None:
    max_logging.log("[exp03_sigma_trace] mean interpolant error per sigma step")
    for entry in curve:
        max_logging.log(
            f"[exp03_sigma_trace]   i={entry['index']:>3} sigma={entry['sigma']:.4f} "
            f"error={entry['error']:.6f} floor={entry.get('floor', float('nan')):.6f}"
        )


def run_trace(config) -> dict:
    """Trace the cohort at one checkpoint and write the diagnostic JSON."""
    manifest_path = str(getattr(config, "model_manifest_path", ""))
    # Cheap refusal BEFORE the ~5B load: the approved trace, or nothing.
    assert_approved_trace_design(
        seed=int(getattr(config, "seed", -1)),
        num_windows=int(getattr(config, "probe_num_windows", -1)),
        sampling_steps=int(getattr(config, "side_adapter_sampling_steps", -1)),
    )
    cohort = trace_cohort(manifest_path, seed=TRACE_SEED, num_windows=TRACE_NUM_WINDOWS)
    if len(cohort) != TRACE_NUM_WINDOWS:
        raise ValueError(f"the cohort selection returned {len(cohort)} windows, not {TRACE_NUM_WINDOWS}")
    (
        trainer,
        pipeline,
        mesh,
        state,
        state_shardings,
        null_context,
        checkpoint_step,
    ) = gen._restore_overfit100_validation_state(config)
    scheduler, _ = trainer._create_scheduler()
    sigmas, timesteps = overfit100_sampler_grid(
        num_inference_steps=TRACE_SAMPLING_STEPS,
        flow_shift=config.flow_shift,
        sigma_min=scheduler.config.sigma_min,
        sigma_max=scheduler.config.sigma_max,
        num_train_timesteps=scheduler.config.num_train_timesteps,
    )
    if int(sigmas.shape[0]) != TRACE_SAMPLING_STEPS + 1 or int(timesteps.shape[0]) != TRACE_SAMPLING_STEPS:
        raise ValueError(
            f"the grid came back as {int(sigmas.shape[0])} sigmas / {int(timesteps.shape[0])} timesteps; "
            f"the pinned {TRACE_SAMPLING_STEPS}-step trace needs {TRACE_SAMPLING_STEPS + 1} / {TRACE_SAMPLING_STEPS}"
        )
    replicated = NamedSharding(mesh, P())
    weights_dtype = gen._dtype(config.weights_dtype)
    # One window is one row of a mesh-sized batch, through ONE compiled forward: the standalone
    # entry point has the same batch-1 forward the probe's does and would hit the same
    # IndivisibleError on the same mesh, and the same dispatch storm if it were left eager.
    forward = jitted_tiled_forward(state, replicas=batch_replicas(config, mesh))
    velocity_fn = velocity_fn_for(forward, state)

    samples = gen.read_overfit100_samples(config, cohort)
    if jax.process_index() == 0:
        max_logging.log(
            f"[exp03_sigma_trace] step={checkpoint_step} windows={len(samples)} "
            f"steps={config.side_adapter_sampling_steps} seed={TRACE_SEED} mode={TRACE_CONTEXT_MODE}"
        )

    rows: list[dict] = []
    for sample in samples:
        z_i0 = jax.device_put(jnp.asarray(sample.z_i0[None]).astype(weights_dtype), replicated)
        z_gt = jax.device_put(jnp.asarray(sample.z_video[None]).astype(weights_dtype), replicated)
        context = gen.overfit100_context_for_mode(
            state.context_table, null_context, TRACE_CONTEXT_MODE, sample.episode_index, None
        ).astype(weights_dtype)
        context = jnp.broadcast_to(context, (z_gt.shape[0], context.shape[1], context.shape[2]))
        eps = trace_noise(
            episode_id=sample.episode_id, window_start=sample.window_start, shape=z_gt.shape, dtype=z_gt.dtype
        )
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            window_trace = trace_rows(
                velocity_fn,
                z_gt=z_gt,
                z_i0=z_i0,
                eps=eps,
                context=context,
                sigmas=sigmas,
                timesteps=timesteps,
            )
        if jax.process_index() != 0:
            continue
        rows.append(
            {
                "name": sample.name,
                "episode_id": int(sample.episode_id),
                "episode_index": int(sample.episode_index),
                "window_start": int(sample.window_start),
                "trace": window_trace,
            }
        )
        max_logging.log(
            f"[exp03_sigma_trace] {sample.name} first={window_trace[0]['error']:.6f} "
            f"last={window_trace[-1]['error']:.6f} floor_last={window_trace[-1]['floor']:.6f}"
        )

    if jax.process_index() != 0:
        return {}
    payload = trace_artifact(config, checkpoint_step=checkpoint_step, cohort=cohort, rows=rows)
    path = trace_output_path(config, checkpoint_step)
    write_trace_artifact(path, payload)
    _log_mean_trace(payload["mean_trace"])
    max_logging.log(f"[exp03_sigma_trace] wrote {path} ({len(rows)} windows)")
    return payload


def run(argv: Sequence[str]):
    pyconfig.initialize(argv)
    config = pyconfig.config
    if str(getattr(config, "model_type", "")) != gen.OVERFIT100_MODEL_TYPE:
        raise ValueError(
            f"the sigma-trajectory trace requires model_type == {gen.OVERFIT100_MODEL_TYPE!r}; "
            f"got {config.model_type!r}"
        )
    return run_trace(config)


def main(argv: Sequence[str]) -> None:  # pragma: no cover - entry point
    run(argv)


if __name__ == "__main__":  # pragma: no cover
    app.run(main)
