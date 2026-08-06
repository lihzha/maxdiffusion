"""A3: joint direct optimization of every step's nulls, plus its J1 cost measurement (plan §4-P1b).

A1 and A2 optimize ∅_i one sampler step at a time, each against a *pivot* the inversion supplied, and
never differentiate through more than a single Euler step. **A3 throws the pivots away.** It treats
the whole ``[N, B, L, D]`` null tensor as one parameter and asks the endpoint question directly:
choose all of them at once so that a full CFG rollout from ε₀ lands on ``z_video``. The rollout is
therefore a differentiable function of every null simultaneously, which is the entire reason this
lives beside ``null_inversion_wan`` rather than inside it.

**One optimizer, not one per step.** ``optimize_null_embeddings`` deliberately constructs a *fresh*
Adam per sampler step, because there each step is its own small problem and the reference does the
same. Here there is one problem, so there is one ``optax.adam``: its moments and -- decisively -- its
bias correction advance with the iteration count. Carrying the contrast explicitly matters because
the two implementations differ by a single line and converge differently, and both produce numbers
that look like A3.

**Memory is the reason A3 is conditional.** A differentiable 25-step rollout stores activations for
every step, and at production geometry against a frozen 5B backbone that is what decides whether the
job is runnable at all. ``@jax.remat`` on the per-step body inside ``lax.scan`` trades recomputation
for that storage -- the exp_03-precedent pattern. Whether it is *observable* from outside is a
question about memory, not about numbers: remat is semantics-preserving by construction, so no test
here can distinguish it numerically, and the round records that rather than inventing a proxy.

**The measurement is what authorizes J1b.** ``measure_single_update`` runs inside J1 (plan §4-P1
item iii): it compiles and executes exactly one update for one example, separates compile time from
step time, reads peak HBM where the backend offers it, enforces the numerical stops, and projects the
full job. J1b is proposed only if that projection fits four hours on a v6e-8 -- so a budget that does
not stop, or projection arithmetic that is off by a factor, spends TPU hours on a job nobody sized.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from maxdiffusion.models.wan.null_inversion_wan import (
    ADAM_B1,
    ADAM_B2,
    ADAM_EPS,
    N_HIST_FRAMES,
    NUM_TRAIN_TIMESTEPS,
    _build_per_token_timestep,
    _checked_velocity,
    _validate_sigmas,
    apply_first_frame_pin,
    embed_null_tokens,
    rollout_timesteps_from_sigmas,
)


# Plan §4-P1b's sizing constants and §4-P1 item (iii)'s numerical stops.
A3_ITERS = 300  # "~300 Adam iterations" over the joint parameter
A3_EXAMPLES = 8  # J1b is A3 on 8 DEV examples -- ONE joint batch, not eight serial runs
# The endpoint evaluation and the ~50 MB array publication that follow the last update. Declared
# rather than measured: it happens once, after the loop, and a projection that omits it is a
# projection of the loop rather than of the job.
A3_WRITE_ALLOWANCE_SECONDS = 300.0
COMPILE_BUDGET_SECONDS = 30 * 60
UPDATE_BUDGET_SECONDS = 120
J1B_BUDGET_SECONDS = 4 * 3600
# What a failing measurement says. "ok" means the measurement itself succeeded -- it does not mean
# J1b is proposed; that is ``fits_budget``, and the two are deliberately separate.
VERDICT_OK, VERDICT_COMPILE, VERDICT_UPDATE = "ok", "compile-budget", "update-budget"
VERDICT_OOM, VERDICT_NONFINITE = "oom", "nonfinite"
# An allocation failure is a measurement result: it is exactly the thing being measured. Anything
# else is a bug, and reporting it as "A3 does not fit" would blame the machine for the code.
# Matched by exception type OR by allocation wording at a **word boundary**: a bare "OOM" substring
# classified ``RuntimeError("BOOM: model kernel bug")`` as an out-of-memory verdict (R11 review,
# finding 5), which is precisely how a kernel bug becomes "A3 needs a bigger machine".
OOM_TYPES = ("XlaRuntimeError", "ResourceExhaustedError", "OutOfMemoryError", "MemoryError")
OOM_PATTERN = re.compile(r"\b(?:RESOURCE_EXHAUSTED|OOM|out\s+of\s+memory)\b", re.IGNORECASE)
# Peak counters and current-allocation counters are different measurements. ``bytes_in_use`` is what
# is allocated right now; reporting it as a peak understates the high-water mark a job must fit
# under (finding 6), so the two are read from disjoint key sets and reported separately.
PEAK_MEMORY_KEYS = ("peak_bytes_in_use", "peak_memory_in_use")
CURRENT_MEMORY_KEYS = ("bytes_in_use", "memory_in_use")
# The jaxpr primitives that mean "this backward pass rematerializes". Reviewed allowlist: on JAX
# 0.10.2 the recursive primitive is ``remat2`` nested inside ``scan``. Per the R11 remat ruling the
# pin FAILS CLOSED on drift -- an unknown renamed primitive must be inspected in the lowered backward
# graph and added deliberately, never tolerated automatically.
REMAT_PRIMITIVES = frozenset({"remat2", "remat", "checkpoint"})


@dataclasses.dataclass(frozen=True)
class MeasurementReport:
    """What one A3 update cost, and whether that authorizes proposing J1b.

    Timing follows the R11 review's methodology. The kernel takes its nulls, latents and optimizer
    state as **operands**, so what is compiled is an actual optimizer step rather than a zero-argument
    program with the data folded in as constants (the reviewer's lowered toy had ``@main()`` and 85
    embedded constants). ``lower()`` and ``compile()`` are timed separately and **without executing**;
    then the compiled executable runs exactly once, synchronized.

    ``verdict`` is about the measurement; ``fits_budget`` is about the job. A measurement can succeed
    and still not authorize J1b, and a measurement that stopped never authorizes it.

    **The budget fields are verdicts on measured values, not interrupts.** A synchronous XLA
    compilation cannot be cancelled from inside the process that is blocked in it, so the plan's
    hard 1,800 s / 120 s aborts belong to the launcher's external watchdog; these numbers are how the
    run *reports* that it went over, not how it is stopped. The runbook says so too.

    Memory is reported per device with the key each number came from. Peak and current allocation are
    different measurements and are kept apart; ``None`` means the backend exposes no such counter and
    is unavailable evidence -- never a fit certificate.
    """

    verdict: str
    reasons: tuple[str, ...]
    lower_seconds: float
    compile_seconds: float
    step_seconds: float | None
    setup_seconds: float
    peak_hbm_bytes: int | None
    current_hbm_bytes: int | None
    device_memory: tuple[dict[str, Any], ...]
    loss: float | None
    grad_norm: float | None
    batch: int
    iters: int
    job_batch: int
    compute_seconds: float
    write_allowance_seconds: float
    projection_seconds: float
    projection_hours: float
    fits_budget: bool
    preliminary: bool
    budgets: dict[str, float]

def _is_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _validate_recipe(iters: Any, lr: Any, guide_scale: Any) -> None:
    """The R3 contract, restated: an unusable recipe must cost nothing rather than a rollout."""
    if not _is_int(iters):
        raise ValueError(f"iters must be an integer, got {iters!r}")
    if int(iters) < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    if not np.isfinite(lr) or float(lr) < 0.0:
        # lr = 0 is legal and means "frozen nulls", which the measurement's dry-cost case relies on.
        raise ValueError(f"lr must be finite and non-negative, got {lr}")
    if not np.isfinite(guide_scale):
        raise ValueError(f"guide_scale must be finite, got {guide_scale}")


def _validated_geometry(z_start, z_i0, sigmas, base_context, nulls_or_init, *, is_init: bool):
    """Fail closed at every public array entry, in float32, before anything reaches the model."""
    _validate_sigmas(sigmas)
    sigmas = jnp.asarray(sigmas, dtype=jnp.float32)
    z_start = jnp.asarray(z_start).astype(jnp.float32)
    z_i0 = jnp.asarray(z_i0).astype(jnp.float32)
    base_context = jnp.asarray(base_context).astype(jnp.float32)
    value = jnp.asarray(nulls_or_init).astype(jnp.float32)
    steps = int(sigmas.shape[0]) - 1

    if z_start.ndim != 5:
        raise ValueError(f"z_start must be [B, C, F, H, W], got shape {z_start.shape}")
    batch, channels, f_lat, h_lat, w_lat = z_start.shape
    if z_i0.ndim != 5 or (z_i0.shape[0], z_i0.shape[1], z_i0.shape[3:]) != (batch, channels, (h_lat, w_lat)):
        raise ValueError(f"z_i0 shape {z_i0.shape} is inconsistent with z_start shape {z_start.shape}")
    if z_i0.shape[2] not in (1, f_lat):
        raise ValueError(f"z_i0 must carry 1 or {f_lat} latent frames, got {z_i0.shape[2]}")
    if base_context.ndim == 3:
        if base_context.shape[0] != 1:
            raise ValueError(f"base_context must have a unit leading axis when rank-3, got {base_context.shape}")
        base_context = base_context[0]
    if base_context.ndim != 2:
        raise ValueError(f"base_context must be [S, D] or [1, S, D], got shape {base_context.shape}")

    if is_init:
        if value.ndim == 2:
            value = jnp.broadcast_to(value, (batch, *value.shape))
        if value.ndim != 3 or value.shape[0] != batch:
            raise ValueError(f"null_init shape {value.shape} is inconsistent with a batch of {batch}")
        value = jnp.broadcast_to(value, (steps, *value.shape))
    else:
        if value.ndim == 3:
            value = jnp.broadcast_to(value[:, None], (value.shape[0], batch, *value.shape[1:]))
        if value.ndim != 4:
            raise ValueError(f"nulls must be [N, L, D] or [N, B, L, D], got shape {value.shape}")
        if value.shape[0] != steps:
            raise ValueError(f"nulls must carry one entry per sampler step: got {value.shape[0]}, expected {steps}")
        if value.shape[1] != batch:
            raise ValueError(f"nulls batch {value.shape[1]} does not match z_start batch {batch}")
    if value.shape[-1] != base_context.shape[1]:
        raise ValueError(f"nulls shape {value.shape} is inconsistent with base_context {base_context.shape}")
    if value.shape[-2] > base_context.shape[0]:
        raise ValueError(f"nulls length {value.shape[-2]} exceeds context length {base_context.shape[0]}")
    return sigmas, z_start, z_i0, base_context, value, (batch, f_lat, h_lat, w_lat)


def endpoint_future_mse(z_final: jax.Array, z_video: jax.Array) -> jax.Array:
    """Per-example MSE over the non-pinned latent frames -- A3's whole objective.

    Latent frame 0 is the pinned image condition and is identical in every arm by construction, so
    including it would measure the pin rather than the method (the same rule R6's primary metric
    follows). ``[B]`` float32.
    """
    predicted = jnp.asarray(z_final, jnp.float32)
    target = jnp.asarray(z_video, jnp.float32)
    if predicted.shape != target.shape:
        raise ValueError(f"z_video shape {target.shape} does not match the rollout endpoint {predicted.shape}")
    if predicted.ndim != 5:
        raise ValueError(f"the endpoint must be [B, C, F, H, W], got shape {predicted.shape}")
    return jnp.mean((predicted[:, :, 1:] - target[:, :, 1:]) ** 2, axis=(1, 2, 3, 4))


def direct_rollout(
    velocity_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    nulls: jax.Array,
    z_start: jax.Array,
    z_i0: jax.Array,
    sigmas: jax.Array,
    base_context: jax.Array,
    *,
    guide_scale: float,
    return_trajectory: bool = False,
):
    """The CFG replay of ``replay_with_nulls``, written to be differentiated through end to end.

    Same recurrence, same pins, same per-step ``v_cond`` recomputation -- ``z`` moves every step, so
    a cached conditional velocity would be evaluated at a stale latent. What differs is that nothing
    is stopped: the gradient of the endpoint reaches ``nulls[i]`` for **every** i, which is what
    makes A3 joint rather than a slower per-step method.

    ``@jax.remat`` wraps the scanned body so the backward pass recomputes each step's activations
    instead of storing twenty-five steps of them. It is semantics-preserving, so it changes what the
    rollout *costs*, never what it returns.
    """
    if not np.isfinite(guide_scale):
        raise ValueError(f"guide_scale must be finite, got {guide_scale}")
    sigmas, z_start, z_i0, base_context, nulls, geometry = _validated_geometry(
        z_start, z_i0, sigmas, base_context, nulls, is_init=False
    )
    return _scan_rollout(
        velocity_fn, nulls, z_start, z_i0, sigmas, base_context, guide_scale, geometry,
        return_trajectory=return_trajectory,
    )


def _scan_rollout(
    velocity_fn, nulls, z_start, z_i0, sigmas, base_context, guide_scale, geometry, *, return_trajectory=False
):
    """The rollout itself, over already-canonicalized inputs.

    Split out because ``_validate_sigmas`` reads the grid with numpy and so cannot run under a trace:
    the timing kernel validates once, outside ``jit``, and traces this.
    """
    batch, f_lat, h_lat, w_lat = geometry
    steps = int(sigmas.shape[0]) - 1
    timesteps = rollout_timesteps_from_sigmas(sigmas, NUM_TRAIN_TIMESTEPS)[:steps]
    cond_context = jnp.broadcast_to(base_context, (batch, *base_context.shape))
    weight = jnp.asarray(guide_scale, dtype=jnp.float32)

    @jax.remat  # recompute activations in the backward pass rather than storing every step's
    def body(current, step):
        step_t, dsigma, null_i = step
        timestep_2d = _build_per_token_timestep(
            jnp.broadcast_to(step_t, (batch,)), f_lat, h_lat, w_lat, n_hist=N_HIST_FRAMES
        )
        v_cond = _checked_velocity(velocity_fn, current, timestep_2d, cond_context)
        v_unc = _checked_velocity(velocity_fn, current, timestep_2d, embed_null_tokens(null_i, base_context))
        v_cfg = v_unc + weight * (v_cond - v_unc)
        stepped = apply_first_frame_pin(current + dsigma * v_cfg, z_i0)
        return stepped, stepped

    start = apply_first_frame_pin(z_start, z_i0)
    z_final, stepped = lax.scan(body, start, (timesteps, sigmas[1:] - sigmas[:steps], nulls))
    if return_trajectory:
        return z_final, jnp.concatenate([start[None], stepped], axis=0)
    return z_final


def _objective(velocity_fn, z_start, z_i0, z_video, sigmas, base_context, weight):
    """Σ-over-batch of the per-example endpoint loss, so per-example gradients stay independent."""

    def objective(nulls):
        z_final = direct_rollout(
            velocity_fn, nulls, z_start, z_i0, sigmas, base_context, guide_scale=float(weight)
        )
        per_example = endpoint_future_mse(z_final, z_video)
        return jnp.sum(per_example), per_example

    return objective


def direct_optimize_nulls(
    velocity_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    z_start: jax.Array,
    z_i0: jax.Array,
    z_video: jax.Array,
    sigmas: jax.Array,
    null_init: jax.Array,
    base_context: jax.Array,
    *,
    iters: int,
    lr: float,
    guide_scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Plan §4-P1b's A3: optimize every step's nulls jointly against the rollout endpoint.

    Args:
      velocity_fn: ``(latents, timestep_2d, encoder_hidden_states) -> v``, the R3/R4a seam.
      z_start: ``[B, C, F, H, W]`` starting latents -- ε₀ in the plan's A3.
      z_i0: ``[B, C, 1, H, W]`` (or ``[B, C, F, H, W]``) first-frame condition.
      z_video: ``[B, C, F, H, W]`` target latents; only frames 1.. enter the loss.
      sigmas: the ``N+1`` descending grid ending at 0.0; validated eagerly, so it must be concrete.
      null_init: ``[L, D]`` or ``[B, L, D]`` starting nulls, broadcast to all ``N`` steps.
      base_context: ``[S, D]`` or ``[1, S, D]`` T5("") context whose leading ``L`` rows are replaced.
      iters: Adam iterations over the joint parameter (plan: ~300).
      lr: Adam learning rate. ``0.0`` is legal and freezes the parameter.
      guide_scale: deployment CFG weight ``w``. At ``w = 1`` the null branch cancels out of ``v_cfg``
        and every gradient is identically zero -- algebra, not a bug.

    Returns:
      ``(nulls, losses, grad_norms)`` -- ``[N, B, L, D]`` optimized nulls and ``[iters, B]``
      per-example losses and gradient norms, all float32. ``losses[j]`` is the loss *before* the
      j-th update, matching R3's convention, so the post-optimization loss is not in this array.
    """
    _validate_recipe(iters, lr, guide_scale)
    sigmas, z_start, z_i0, base_context, nulls, _ = _validated_geometry(
        z_start, z_i0, sigmas, base_context, null_init, is_init=True
    )
    z_video = jnp.asarray(z_video).astype(jnp.float32)
    if z_video.shape != z_start.shape:
        raise ValueError(f"z_video shape {z_video.shape} does not match z_start shape {z_start.shape}")

    objective = _objective(velocity_fn, z_start, z_i0, z_video, sigmas, base_context, guide_scale)
    # ONE optimizer for the whole run: A3 is a single joint problem, so Adam's moments and its bias
    # correction advance with the iteration count. R3 constructs a fresh optimizer per sampler step
    # for the opposite reason -- see the module docstring.
    optimizer = optax.adam(lr, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS, eps_root=0.0)

    def iteration(state, _):
        value, opt_state = state
        (_, per_example), grads = jax.value_and_grad(objective, has_aux=True)(value)
        updates, opt_state = optimizer.update(grads, opt_state, value)
        grad_norm = jnp.sqrt(jnp.sum(grads**2, axis=(0, 2, 3)))  # per example, across every step
        return (optax.apply_updates(value, updates), opt_state), (per_example, grad_norm)

    (final, _), (losses, grad_norms) = lax.scan(
        iteration, (nulls, optimizer.init(nulls)), length=int(iters)
    )
    return final, losses, grad_norms


def jaxpr_primitives(jaxpr: Any) -> set[str]:
    """Every primitive name in a jaxpr, recursing into nested ones (``scan`` bodies included).

    Substring matching on ``str(jaxpr)`` happens to work today, but it also matches a variable named
    ``remat_x``; walking the equations is what makes the remat pin a statement about structure.
    """
    names: set[str] = set()
    stack = [jaxpr]
    seen: set[int] = set()
    while stack:
        current = getattr(stack.pop(), "jaxpr", None) or None
        if current is None:
            continue
        if id(current) in seen:
            continue
        seen.add(id(current))
        for eqn in getattr(current, "eqns", ()):
            names.add(eqn.primitive.name)
            for value in eqn.params.values():
                for candidate in value if isinstance(value, (tuple, list)) else (value,):
                    if hasattr(candidate, "eqns") or hasattr(candidate, "jaxpr"):
                        stack.append(candidate if hasattr(candidate, "jaxpr") else _Closed(candidate))
    return names


class _Closed:
    """Adapter so a bare ``Jaxpr`` walks the same path as a ``ClosedJaxpr``."""

    def __init__(self, jaxpr):
        self.jaxpr = jaxpr


def rematerializes(fn: Callable[..., Any], *args: Any) -> tuple[bool, set[str]]:
    """Whether ``jax.grad(fn)``'s jaxpr rematerializes, and every primitive it used.

    Fails closed by construction: the caller gets the observed primitive set, so a renamed primitive
    surfaces as "no remat, here is what I saw" and has to be inspected rather than absorbed.
    """
    primitives = jaxpr_primitives(jax.make_jaxpr(jax.grad(fn))(*args))
    return bool(primitives & REMAT_PRIMITIVES), primitives


def _is_oom(error: BaseException) -> bool:
    """An allocation failure, by exception type or allocation wording at a word boundary."""
    if type(error).__name__ in OOM_TYPES:
        return True
    return bool(OOM_PATTERN.search(str(error)))


def _device_memory(devices: Sequence[Any] | None = None) -> tuple[dict[str, Any], ...]:
    """Per-device peak and current allocation, each labelled with the key it was read from."""
    try:
        devices = list(devices if devices is not None else jax.local_devices())
    except Exception:  # noqa: BLE001 -- device introspection must not fail a timing measurement
        return ()
    entries = []
    for device in devices:
        try:
            stats = device.memory_stats()
        except Exception:  # noqa: BLE001
            stats = None
        entry: dict[str, Any] = {
            "device": str(device),
            "peak_bytes": None,
            "peak_key": None,
            "current_bytes": None,
            "current_key": None,
        }
        if isinstance(stats, Mapping):
            for key in PEAK_MEMORY_KEYS:
                if key in stats:
                    entry["peak_bytes"], entry["peak_key"] = int(stats[key]), key
                    break
            for key in CURRENT_MEMORY_KEYS:
                if key in stats:
                    entry["current_bytes"], entry["current_key"] = int(stats[key]), key
                    break
        entries.append(entry)
    return tuple(entries)


def _across_devices(entries: Sequence[Mapping[str, Any]], field: str) -> int | None:
    """The high-water mark across every addressable device; ``None`` when nobody reported one."""
    values = [entry[field] for entry in entries if entry.get(field) is not None]
    return max(values) if values else None


def _update_kernel(velocity_fn, sigmas, geometry, *, lr, guide_scale):
    """One Adam update as a function of its operands -- the thing J1b actually runs, 300 times.

    ``nulls``, the latents, the target, the base context and the optimizer state are all **arguments**,
    so the compiled program is a real optimizer step. Only ``velocity_fn`` (the frozen model) and the
    validated sigma grid are closed over: the model is state rather than data under optimization, and
    the grid must stay concrete because ``_validate_sigmas`` reads it with numpy.
    """
    optimizer = optax.adam(lr, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS, eps_root=0.0)
    weight = float(guide_scale)

    def kernel(nulls, z_start, z_i0, z_video, base_context, opt_state):
        def objective(value):
            z_final = _scan_rollout(
                velocity_fn, value, z_start, z_i0, sigmas, base_context, weight, geometry
            )
            per_example = endpoint_future_mse(z_final, z_video)
            return jnp.sum(per_example), per_example

        (_, per_example), grads = jax.value_and_grad(objective, has_aux=True)(nulls)
        updates, opt_state = optimizer.update(grads, opt_state, nulls)
        grad_norm = jnp.sqrt(jnp.sum(grads**2, axis=(0, 2, 3)))
        return optax.apply_updates(nulls, updates), opt_state, per_example, grad_norm

    return optimizer, jax.jit(kernel)


def measure_single_update(
    velocity_fn: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    z_start: jax.Array,
    z_i0: jax.Array,
    z_video: jax.Array,
    sigmas: jax.Array,
    null_init: jax.Array,
    base_context: jax.Array,
    *,
    lr: float,
    guide_scale: float,
    iters: int = A3_ITERS,
    job_batch: int = A3_EXAMPLES,
    write_allowance: float = A3_WRITE_ALLOWANCE_SECONDS,
    setup_seconds: float = 0.0,
    compile_budget: float = COMPILE_BUDGET_SECONDS,
    update_budget: float = UPDATE_BUDGET_SECONDS,
    projection_budget: float = J1B_BUDGET_SECONDS,
    clock: Callable[[], float] | None = None,
    devices: Sequence[Any] | None = None,
    require_single_example: bool = True,
) -> MeasurementReport:
    """Plan §4-P1 item (iii): compile ONE A3 update, execute it ONCE, and price the full job.

    This runs inside J1 and decides whether J1b is even proposed, so it measures rather than
    estimates: lowering and compilation are timed on their own, without executing, and then the
    compiled executable is invoked exactly once under ``block_until_ready``.

    **The step is one JOINT update over the whole batch, so the projection does not multiply by the
    batch.** J1b optimizes all eight examples as a single ``[N, 8, L, D]`` parameter; running 300
    iterations means 300 updates, not 2,400. Multiplying by the example count inflated a fitting
    10-second update into a 24,000-second refusal (R11 follow-up). The projected wall clock is::

        lower + compile + setup + iters x step + write_allowance

    Compilation is in it because J1b is a separate job with its own B=8 executable and no configured
    persistent compilation cache, so its compile time *is* wall time (review finding 3). ``setup`` is
    the caller's measured non-kernel time -- eps_0 construction and data staging -- and
    ``write_allowance`` covers the endpoint evaluation and array publication after the last update.

    ``job_batch`` is what the *job* will run at, and is used only to mark a smaller measurement
    ``preliminary``: a B=1 measurement is a preliminary compute estimate, never a B=8 HBM
    certification, because B=8 has a different compile, execution, sharding and memory profile. The
    J1b mode therefore opens with its own B=8 fit probe before committing to 300 iterations.
    """
    _validate_recipe(iters, lr, guide_scale)
    if not _is_int(job_batch) or int(job_batch) < 1:
        # examples=0 projected zero seconds and "fits"; examples=-1 projected -300 (finding 3).
        raise ValueError(f"job_batch must be an integer >= 1, got {job_batch!r}")
    for name, budget in (
        ("compile_budget", compile_budget),
        ("update_budget", update_budget),
        ("projection_budget", projection_budget),
        ("setup_seconds", setup_seconds),
        ("write_allowance", write_allowance),
    ):
        if not np.isfinite(budget) or float(budget) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {budget!r}")

    clock = clock or time.perf_counter
    sigmas, z_start, z_i0, base_context, nulls, geometry = _validated_geometry(
        z_start, z_i0, sigmas, base_context, null_init, is_init=True
    )
    z_video = jnp.asarray(z_video).astype(jnp.float32)
    if z_video.shape != z_start.shape:
        raise ValueError(f"z_video shape {z_video.shape} does not match z_start shape {z_start.shape}")
    batch = int(z_start.shape[0])
    if require_single_example and batch != 1:
        raise ValueError(f"the A3 measurement is defined on exactly one example, got a batch of {batch}")

    budgets = {
        "compile_seconds": float(compile_budget),
        "update_seconds": float(update_budget),
        "projection_seconds": float(projection_budget),
    }

    def report(**fields) -> MeasurementReport:
        step = fields.get("step_seconds")
        overhead = float(fields.get("lower_seconds", 0.0)) + float(fields.get("compile_seconds", 0.0))
        # iters x step: ONE joint update per iteration, whatever the batch size.
        compute = (
            overhead + float(setup_seconds) + float(iters) * float(step) if step is not None else float("inf")
        )
        projection = compute + float(write_allowance) if step is not None else float("inf")
        authorized = fields.get("verdict", VERDICT_OK) == VERDICT_OK and projection <= projection_budget
        defaults = {
            "verdict": VERDICT_OK,
            "reasons": (),
            "lower_seconds": 0.0,
            "compile_seconds": 0.0,
            "step_seconds": None,
            "setup_seconds": float(setup_seconds),
            "peak_hbm_bytes": None,
            "current_hbm_bytes": None,
            "device_memory": (),
            "loss": None,
            "grad_norm": None,
            "batch": batch,
            "iters": int(iters),
            "job_batch": int(job_batch),
            "compute_seconds": compute,
            "write_allowance_seconds": float(write_allowance),
            "projection_seconds": projection,
            "projection_hours": projection / 3600.0,
            "fits_budget": bool(authorized),
            "preliminary": batch < int(job_batch),
            "budgets": budgets,
        }
        return MeasurementReport(**{**defaults, **fields})

    optimizer, kernel = _update_kernel(velocity_fn, sigmas, geometry, lr=lr, guide_scale=guide_scale)
    operands = (nulls, z_start, z_i0, z_video, base_context, optimizer.init(nulls))

    try:
        started = clock()
        lowered = kernel.lower(*operands)
        lower_seconds = clock() - started
        started = clock()
        compiled = lowered.compile()  # compiled, NOT executed
        compile_seconds = clock() - started
    except Exception as error:  # noqa: BLE001 -- an allocation failure while compiling is a result
        if not _is_oom(error):
            raise
        return report(verdict=VERDICT_OOM, reasons=(f"{type(error).__name__}: {error}",))

    if compile_seconds > compile_budget:
        return report(
            verdict=VERDICT_COMPILE,
            reasons=(
                f"compilation took {compile_seconds:.1f}s, over the {compile_budget:.0f}s budget "
                f"(reported, not interrupted: see the docstring on the external watchdog)",
            ),
            lower_seconds=lower_seconds,
            compile_seconds=compile_seconds,
            device_memory=_device_memory(devices),
        )

    try:
        started = clock()
        outputs = jax.block_until_ready(compiled(*operands))  # exactly one synchronized execution
        step_seconds = clock() - started
    except Exception as error:  # noqa: BLE001
        if not _is_oom(error):
            raise
        return report(
            verdict=VERDICT_OOM,
            reasons=(f"{type(error).__name__}: {error}",),
            lower_seconds=lower_seconds,
            compile_seconds=compile_seconds,
        )

    updated, _, per_example, grad_norm = outputs
    entries = _device_memory(devices)
    measured = {
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "step_seconds": step_seconds,
        "peak_hbm_bytes": _across_devices(entries, "peak_bytes"),
        "current_hbm_bytes": _across_devices(entries, "current_bytes"),
        "device_memory": entries,
        "loss": float(jnp.sum(per_example)),
        "grad_norm": float(jnp.sum(grad_norm)),
    }

    # Finiteness is checked AFTER synchronization, on the values the update actually produced. A
    # velocity returning NaN previously reported verdict="ok", loss=nan and fits_budget=True -- a
    # measurement of nothing, authorizing a job (finding 5).
    nonfinite = [
        name
        for name, value in (("losses", per_example), ("grad_norms", grad_norm), ("nulls", updated))
        if not bool(jnp.all(jnp.isfinite(value)))
    ]
    if nonfinite:
        return report(
            verdict=VERDICT_NONFINITE,
            reasons=(f"the update produced non-finite {', '.join(nonfinite)}: this measures nothing",),
            **measured,
        )
    if step_seconds > update_budget:
        return report(
            verdict=VERDICT_UPDATE,
            reasons=(
                f"one update took {step_seconds:.1f}s, over the {update_budget:.0f}s budget "
                f"(reported, not interrupted: see the docstring on the external watchdog)",
            ),
            **measured,
        )
    return report(**measured)
