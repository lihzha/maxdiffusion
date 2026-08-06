"""exp_04 R11 — A3: joint direct optimization of every step's nulls, and its cost measurement.

A3 asks a different question from A1/A2. Those optimize ∅_i one sampler step at a time against a
*pivot* the inversion supplied; A3 throws the pivots away and asks the endpoint question directly --
choose all ``[N, B, L, D]`` nulls at once so that a full 25-step CFG rollout from ε₀ lands on
``z_video``. That makes the rollout a differentiable function of every null simultaneously, which is
why this module exists separately: the per-step optimizer never differentiates through more than one
Euler step, and this one differentiates through all of them.

Three properties carry the round:

- **Every step's nulls receive gradient.** The whole claim of A3 is joint optimization. If the
  gradient path through one step were severed the run would still produce numbers, still converge on
  the remaining steps, and still look like A3.
- **One optimizer, not one per iteration.** This is a single joint problem, so Adam's moments and its
  bias correction carry across iterations -- the exact opposite of R3, which deliberately builds a
  fresh optimizer per sampler step. A fresh-per-iteration implementation is a different algorithm
  that converges differently and would be reported as A3.
- **The measurement is what authorizes J1b.** It runs *inside* J1, and its numbers decide whether a
  separately-approved job is even proposed. Budgets that do not stop, or projection arithmetic that
  is off, spend TPU hours on a job nobody sized.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from maxdiffusion.models.wan.null_direct_opt_wan import (
    A3_EXAMPLES,
    A3_ITERS,
    A3_WRITE_ALLOWANCE_SECONDS,
    COMPILE_BUDGET_SECONDS,
    J1B_BUDGET_SECONDS,
    REMAT_PRIMITIVES,
    UPDATE_BUDGET_SECONDS,
    MeasurementReport,
    direct_optimize_nulls,
    direct_rollout,
    endpoint_future_mse,
    jaxpr_primitives,
    measure_single_update,
    rematerializes,
)
from maxdiffusion.models.wan.null_inversion_wan import replay_with_nulls


_B, _C, _F, _H, _W = 2, 2, 3, 2, 2
_S, _D, _L = 4, 3, 2
_SIGMAS = jnp.asarray((1.0, 0.5, 0.0), jnp.float32)  # N = 2 steps
_STEPS = int(_SIGMAS.shape[0]) - 1
_GUIDE = 5.0  # emphatically not 1.0: at w = 1 the null branch cancels and every gradient is zero
_ADAM = (0.9, 0.999, 1e-8)

_RNG = np.random.default_rng(20260806)
_PATTERN = jnp.asarray(_RNG.standard_normal((_C, _F, _H, _W), dtype=np.float32))
_CTX_WEIGHTS = jnp.asarray(_RNG.standard_normal((_L, _D), dtype=np.float32))
_C_CTX, _C_Z, _C_T = 0.3, 0.05, 1e-4


def _velocity_fn(z, timestep_2d, context):
    """A differentiable oracle that reads the nulls, the latent and the per-token timestep."""
    scale = jnp.sum(context[:, :_L] * _CTX_WEIGHTS, axis=(1, 2))
    return (
        _C_CTX * scale[:, None, None, None, None] * _PATTERN
        + _C_Z * z
        + _C_T * jnp.mean(timestep_2d)
    )


def _forbidden_velocity(*args, **kwargs):
    raise AssertionError("arguments must be validated before any compute")


def _inputs(batch=_B, seed=3):
    rng = np.random.default_rng(seed)
    z_start = jnp.asarray(rng.standard_normal((batch, _C, _F, _H, _W), dtype=np.float32))
    z_i0 = jnp.asarray(rng.standard_normal((batch, _C, 1, _H, _W), dtype=np.float32))
    z_video = jnp.asarray(rng.standard_normal((batch, _C, _F, _H, _W), dtype=np.float32))
    base_context = jnp.asarray(rng.standard_normal((_S, _D), dtype=np.float32))
    return z_start, z_i0, z_video, base_context


def _nulls(batch=_B, seed=11):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((_STEPS, batch, _L, _D), dtype=np.float32))


def _optimize(batch=_B, iters=2, lr=1e-2, guide_scale=_GUIDE, **overrides):
    z_start, z_i0, z_video, base_context = _inputs(batch)
    fields = {
        "velocity_fn": _velocity_fn,
        "z_start": z_start,
        "z_i0": z_i0,
        "z_video": z_video,
        "sigmas": _SIGMAS,
        "null_init": base_context[:_L],
        "base_context": base_context,
        "iters": iters,
        "lr": lr,
        "guide_scale": guide_scale,
    }
    fields.update(overrides)
    velocity_fn = fields.pop("velocity_fn")
    return direct_optimize_nulls(
        velocity_fn,
        fields.pop("z_start"),
        fields.pop("z_i0"),
        fields.pop("z_video"),
        fields.pop("sigmas"),
        fields.pop("null_init"),
        fields.pop("base_context"),
        **fields,
    )


# --------------------------------------------------------------------------- the differentiable rollout


def test_every_steps_nulls_receive_a_finite_non_zero_gradient():
    """A3's entire claim. A severed path through one step still converges on the others and still
    produces a plausible loss curve -- it is simply no longer joint optimization."""
    z_start, z_i0, z_video, base_context = _inputs()
    nulls = _nulls()

    def loss(value):
        z_final = direct_rollout(
            _velocity_fn, value, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE
        )
        return jnp.sum(endpoint_future_mse(z_final, z_video))

    grads = jax.grad(loss)(nulls)
    per_step = np.asarray(jnp.sqrt(jnp.sum(grads**2, axis=(1, 2, 3))))

    assert grads.shape == nulls.shape
    assert np.all(np.isfinite(per_step))
    assert per_step.shape == (_STEPS,)
    assert float(per_step.min()) > 0.0, per_step  # EVERY step, not just the last


def test_each_step_moves_the_endpoint_on_its_own():
    """Per-step, not just in aggregate: perturbing one step's nulls must move the endpoint."""
    z_start, z_i0, z_video, base_context = _inputs()
    nulls = _nulls()
    baseline = direct_rollout(_velocity_fn, nulls, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE)

    for step in range(_STEPS):
        bumped = nulls.at[step].add(0.5)
        moved = direct_rollout(_velocity_fn, bumped, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE)
        assert float(jnp.max(jnp.abs(moved - baseline))) > 1e-6, step


def test_the_pin_is_applied_at_every_step_of_the_rollout():
    z_start, z_i0, _, base_context = _inputs()

    z_final, trajectory = direct_rollout(
        _velocity_fn, _nulls(), z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE, return_trajectory=True
    )

    assert trajectory.shape == (_STEPS + 1, _B, _C, _F, _H, _W)
    for index, frame in enumerate(trajectory):
        np.testing.assert_allclose(np.asarray(frame[:, :, :1]), np.asarray(z_i0), rtol=0, atol=0, err_msg=index)
    np.testing.assert_array_equal(np.asarray(z_final), np.asarray(trajectory[-1]))


def test_the_rollout_is_float32():
    z_start, z_i0, _, base_context = _inputs()

    z_final = direct_rollout(
        _velocity_fn, _nulls().astype(jnp.float16), z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE
    )

    assert z_final.dtype == jnp.float32


# --------------------------------------------------------------------------- the endpoint objective


def test_the_endpoint_loss_is_the_future_frame_mse():
    z_final = jnp.zeros((1, 1, 3, 1, 2), jnp.float32)
    z_video = jnp.asarray([[[[[1.0, 1.0]], [[2.0, 2.0]], [[4.0, 4.0]]]]], jnp.float32)

    # Frames 1 and 2 only: mean of (2^2, 2^2, 4^2, 4^2) = 10.0. Frame 0's 1.0s are excluded.
    np.testing.assert_allclose(np.asarray(endpoint_future_mse(z_final, z_video)), [10.0], rtol=1e-6)


def test_frame_zero_cannot_influence_the_endpoint_loss():
    """It is the pinned image condition, identical in every arm by construction: including it would
    measure the pin rather than the method."""
    z_final = jnp.zeros((1, 1, 3, 1, 2), jnp.float32)
    z_video = jnp.asarray([[[[[1.0, 1.0]], [[2.0, 2.0]], [[4.0, 4.0]]]]], jnp.float32)
    poisoned = z_video.at[:, :, 0].set(1e6)

    np.testing.assert_allclose(
        np.asarray(endpoint_future_mse(z_final, z_video)),
        np.asarray(endpoint_future_mse(z_final, poisoned)),
        rtol=0,
        atol=0,
    )


def test_the_endpoint_loss_is_per_example():
    z_final = jnp.zeros((2, 1, 3, 1, 1), jnp.float32)
    z_video = jnp.asarray([[[[[0.0]], [[1.0]], [[1.0]]]], [[[[0.0]], [[2.0]], [[2.0]]]]], jnp.float32)

    np.testing.assert_allclose(np.asarray(endpoint_future_mse(z_final, z_video)), [1.0, 4.0], rtol=1e-6)


# --------------------------------------------------------------------------- the joint optimizer


def test_one_adam_state_carries_across_iterations():
    """The contrast with R3, which builds a fresh optimizer per sampler step *by design*.

    A3 is one joint problem, so the moments and -- decisively -- Adam's bias correction advance with
    the iteration count. A fresh-per-iteration implementation replays t=1 forever and is a different
    algorithm; this hand-rolled two-iteration reference is what tells them apart.
    """
    z_start, z_i0, z_video, base_context = _inputs()
    null_init = base_context[:_L]
    lr = 1e-2

    def loss(value):
        z_final = direct_rollout(
            _velocity_fn, value, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE
        )
        return jnp.sum(endpoint_future_mse(z_final, z_video))

    start = jnp.broadcast_to(null_init, (_STEPS, _B, _L, _D))
    optimizer = optax.adam(lr, b1=_ADAM[0], b2=_ADAM[1], eps=_ADAM[2], eps_root=0.0)
    value, state = start, optimizer.init(start)
    for _ in range(2):  # ONE optimizer, stepped twice
        grads = jax.grad(loss)(value)
        updates, state = optimizer.update(grads, state, value)
        value = optax.apply_updates(value, updates)

    nulls, _, _ = _optimize(iters=2, lr=lr)

    np.testing.assert_allclose(np.asarray(nulls), np.asarray(value), rtol=1e-5, atol=1e-7)


def test_a_fresh_optimizer_each_iteration_would_land_somewhere_else():
    """Guards the test above from being vacuous: the two algorithms really do differ here."""
    z_start, z_i0, z_video, base_context = _inputs()
    null_init = base_context[:_L]
    lr = 1e-2

    def loss(value):
        z_final = direct_rollout(
            _velocity_fn, value, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE
        )
        return jnp.sum(endpoint_future_mse(z_final, z_video))

    value = jnp.broadcast_to(null_init, (_STEPS, _B, _L, _D))
    for _ in range(2):  # a NEW optimizer every iteration -- R3's per-step discipline, misapplied
        optimizer = optax.adam(lr, b1=_ADAM[0], b2=_ADAM[1], eps=_ADAM[2], eps_root=0.0)
        state = optimizer.init(value)
        grads = jax.grad(loss)(value)
        updates, state = optimizer.update(grads, state, value)
        value = optax.apply_updates(value, updates)

    nulls, _, _ = _optimize(iters=2, lr=lr)

    assert float(jnp.max(jnp.abs(nulls - value))) > 1e-9


def test_the_optimizer_returns_the_declared_shapes():
    nulls, losses, grad_norms = _optimize(iters=3)

    assert nulls.shape == (_STEPS, _B, _L, _D) and nulls.dtype == jnp.float32
    assert losses.shape == (3, _B) and losses.dtype == jnp.float32
    assert grad_norms.shape == (3, _B) and grad_norms.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(losses))) and bool(jnp.all(jnp.isfinite(grad_norms)))


def test_the_objective_descends_on_a_toy():
    _, losses, _ = _optimize(iters=12, lr=5e-2)

    assert float(losses[-1].sum()) < float(losses[0].sum())


def test_batch_composition_does_not_change_an_examples_result():
    """The plan §3 batching contract: the objective sums per-example losses, so each example's
    gradient is exactly the gradient of its own loss."""
    z_start, z_i0, z_video, base_context = _inputs(batch=_B)
    kwargs = {"iters": 3, "lr": 1e-2, "guide_scale": _GUIDE}

    together, _, _ = direct_optimize_nulls(
        _velocity_fn, z_start, z_i0, z_video, _SIGMAS, base_context[:_L], base_context, **kwargs
    )
    alone, _, _ = direct_optimize_nulls(
        _velocity_fn,
        z_start[:1],
        z_i0[:1],
        z_video[:1],
        _SIGMAS,
        base_context[:_L],
        base_context,
        **kwargs,
    )

    np.testing.assert_allclose(np.asarray(together[:, :1]), np.asarray(alone), rtol=1e-5, atol=1e-7)


def test_nulls_start_from_the_supplied_initializer():
    z_start, z_i0, z_video, base_context = _inputs()

    nulls, _, _ = direct_optimize_nulls(
        _velocity_fn, z_start, z_i0, z_video, _SIGMAS, base_context[:_L], base_context,
        iters=1, lr=0.0, guide_scale=_GUIDE,
    )

    # lr = 0 freezes the parameter, so what comes back is exactly the broadcast initializer.
    np.testing.assert_allclose(
        np.asarray(nulls), np.asarray(jnp.broadcast_to(base_context[:_L], (_STEPS, _B, _L, _D))), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"iters": 0}, "iters must be >= 1"),
        ({"iters": 1.5}, "iters must be an integer"),
        ({"iters": True}, "iters must be an integer"),
        ({"lr": -1e-3}, "lr must be finite and non-negative"),
        ({"lr": float("nan")}, "lr must be finite and non-negative"),
        ({"guide_scale": float("inf")}, "guide_scale must be finite"),
    ],
)
def test_an_unusable_recipe_is_refused_before_any_compute(overrides, message):
    with pytest.raises(ValueError, match=message):
        _optimize(velocity_fn=_forbidden_velocity, **overrides)


def test_a_null_initializer_longer_than_the_context_is_refused():
    z_start, z_i0, z_video, base_context = _inputs()

    with pytest.raises(ValueError, match="exceeds context length"):
        direct_optimize_nulls(
            _forbidden_velocity,
            z_start,
            z_i0,
            z_video,
            _SIGMAS,
            jnp.zeros((_S + 1, _D), jnp.float32),
            base_context,
            iters=1,
            lr=1e-2,
            guide_scale=_GUIDE,
        )


def test_a_target_that_does_not_match_the_start_is_refused():
    z_start, z_i0, z_video, base_context = _inputs()

    with pytest.raises(ValueError, match="z_video"):
        direct_optimize_nulls(
            _forbidden_velocity,
            z_start,
            z_i0,
            z_video[:, :, :-1],
            _SIGMAS,
            base_context[:_L],
            base_context,
            iters=1,
            lr=1e-2,
            guide_scale=_GUIDE,
        )


# --------------------------------------------------------------------------- the J1 measurement


def _clock(*values):
    ticks = iter(values)
    return lambda: next(ticks)


class _FakeDevice:
    def __init__(self, name="tpu:0", stats=None):
        self._name, self._stats = name, stats

    def __str__(self):
        return self._name

    def memory_stats(self):
        return self._stats


# lower 0.5 s, compile 3.0 s, one execution 1.0 s
_TICKS = (0.0, 0.5, 1.0, 4.0, 10.0, 11.0)
_DEVICES = (_FakeDevice("tpu:0", {"peak_bytes_in_use": 1234, "bytes_in_use": 900}),)


def _measure(**overrides):
    z_start, z_i0, z_video, base_context = _inputs(batch=1)
    fields = {
        "lr": 1e-2,
        "guide_scale": _GUIDE,
        "clock": _clock(*_TICKS),
        "devices": _DEVICES,
    }
    fields.update(overrides)
    velocity_fn = fields.pop("velocity_fn", _velocity_fn)
    return measure_single_update(
        velocity_fn, z_start, z_i0, z_video, _SIGMAS, base_context[:_L], base_context, **fields
    )


def test_lowering_compilation_and_the_single_execution_are_timed_separately():
    """The reviewer's methodology: compile without executing, then execute exactly once.

    R11's first cut called a zero-argument ``jax.jit`` twice and subtracted -- two executions, and a
    compiled program with the data folded in as 85 constants rather than an optimizer step.
    """
    report = _measure()

    assert isinstance(report, MeasurementReport)
    assert report.verdict == "ok" and report.reasons == ()
    assert report.lower_seconds == pytest.approx(0.5)
    assert report.compile_seconds == pytest.approx(3.0)  # compile alone, nothing executed
    assert report.step_seconds == pytest.approx(1.0)  # one synchronized execution
    assert report.batch == 1


def test_the_compiled_program_takes_the_optimizer_state_and_data_as_operands():
    """A zero-operand jit specializes on the data; what it compiles is not the step J1b will run."""
    z_start, z_i0, z_video, base_context = _inputs(batch=1)
    from maxdiffusion.models.wan.null_direct_opt_wan import _update_kernel, _validated_geometry

    sigmas, z_start, z_i0, base_context, nulls, geometry = _validated_geometry(
        z_start, z_i0, _SIGMAS, base_context, base_context[:_L], is_init=True
    )
    optimizer, kernel = _update_kernel(_velocity_fn, sigmas, geometry, lr=1e-2, guide_scale=_GUIDE)
    text = kernel.lower(nulls, z_start, z_i0, z_video, base_context, optimizer.init(nulls)).as_text()

    signature = next(line for line in text.splitlines() if "@main" in line)
    # Six operands reach the kernel: the parameter, three latents, the context and the Adam state --
    # not a zero-argument @main() with the data folded in as constants.
    assert "()" not in signature, signature
    assert signature.count("tensor<") >= 6, signature


def test_the_projection_includes_compilation_setup_and_the_write_allowance():
    """J1b is a separate job with its own B=8 executable and no configured compilation cache, so its
    compile time is wall time. Leaving it out approved a 15,000 s job as fitting four hours."""
    report = _measure(setup_seconds=7.0)

    overhead = 0.5 + 3.0 + 7.0
    assert report.setup_seconds == pytest.approx(7.0)
    assert report.compute_seconds == pytest.approx(overhead + 1.0 * A3_ITERS)
    assert report.write_allowance_seconds == pytest.approx(A3_WRITE_ALLOWANCE_SECONDS)
    assert report.projection_seconds == pytest.approx(report.compute_seconds + A3_WRITE_ALLOWANCE_SECONDS)
    assert report.projection_hours == pytest.approx(report.projection_seconds / 3600.0)


def test_a_batched_update_is_one_update_not_one_per_example():
    """The reviewer's case: a 10-second JOINT update over all eight examples.

    300 iterations of it is 300 updates, so the compute is 3,000 s plus 3.5 s of lower+compile --
    a job that comfortably fits. Multiplying by the batch turned that into 24,000 s and refused it.
    """
    report = _measure(clock=_clock(0.0, 0.5, 1.0, 4.0, 10.0, 20.0), write_allowance=0.0)

    assert report.step_seconds == pytest.approx(10.0)
    assert report.compute_seconds == pytest.approx(3003.5)
    assert report.projection_seconds == pytest.approx(3003.5)
    assert report.fits_budget is True  # ... and it AUTHORIZES


def test_the_write_allowance_is_carried_into_the_projection():
    report = _measure(clock=_clock(0.0, 0.5, 1.0, 4.0, 10.0, 20.0), write_allowance=120.0)

    assert report.projection_seconds == pytest.approx(3003.5 + 120.0)
    assert report.fits_budget is True


def test_a_projection_that_only_fits_without_compilation_is_refused():
    """The reviewer's probe: a realistic compile plus exactly the budget's worth of updates."""
    step = J1B_BUDGET_SECONDS / A3_ITERS  # updates alone == exactly 4 h

    report = _measure(clock=_clock(0.0, 0.0, 0.0, 600.0, 10.0, 10.0 + step), write_allowance=0.0)

    assert report.compile_seconds == pytest.approx(600.0)
    assert report.projection_seconds == pytest.approx(J1B_BUDGET_SECONDS + 600.0)
    assert report.fits_budget is False


def test_a_projection_exactly_at_four_hours_including_overhead_still_fits():
    step = (J1B_BUDGET_SECONDS - 3.5 - 1e-6) / A3_ITERS  # 0.5 lower + 3.0 compile

    report = _measure(clock=_clock(0.0, 0.5, 1.0, 4.0, 10.0, 10.0 + step), write_allowance=0.0)

    assert report.projection_seconds == pytest.approx(J1B_BUDGET_SECONDS)
    assert report.projection_seconds <= J1B_BUDGET_SECONDS
    assert report.fits_budget is True


def test_a_projection_over_four_hours_does_not_propose_j1b():
    over = (J1B_BUDGET_SECONDS / A3_ITERS) + 1.0

    report = _measure(clock=_clock(0.0, 0.5, 1.0, 4.0, 10.0, 10.0 + over))

    assert report.projection_seconds > J1B_BUDGET_SECONDS
    assert report.fits_budget is False
    assert report.verdict == "ok"  # the measurement succeeded; it simply does not authorize J1b


@pytest.mark.parametrize("job_batch", [0, -1, 1.5, True, "8"])
def test_a_nonsensical_job_batch_is_refused(job_batch):
    """``examples=0`` projected zero seconds and fit; ``examples=-1`` projected -300 and also fit."""
    with pytest.raises(ValueError, match="job_batch must be an integer >= 1"):
        _measure(job_batch=job_batch)


def test_a_negative_write_allowance_is_refused():
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _measure(write_allowance=-1.0)


@pytest.mark.parametrize(
    "budget", [{"compile_budget": -1.0}, {"update_budget": float("nan")}, {"projection_budget": -5.0}]
)
def test_a_nonsensical_budget_is_refused(budget):
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _measure(**budget)


def test_a_compile_over_thirty_minutes_is_reported_as_a_stop():
    report = _measure(clock=_clock(0.0, 0.5, 1.0, 1.0 + COMPILE_BUDGET_SECONDS + 1.0, 0.0, 0.0))

    assert report.verdict == "compile-budget"
    assert any("compilation" in reason for reason in report.reasons)
    assert any("watchdog" in reason for reason in report.reasons)  # honest about what it cannot do
    assert report.fits_budget is False and report.step_seconds is None


def test_an_update_over_two_minutes_is_reported_as_a_stop():
    report = _measure(clock=_clock(0.0, 0.5, 1.0, 4.0, 10.0, 10.0 + UPDATE_BUDGET_SECONDS + 1.0))

    assert report.verdict == "update-budget"
    assert any("update" in reason for reason in report.reasons)
    assert report.fits_budget is False


def test_an_update_exactly_at_the_budget_is_not_a_stop():
    report = _measure(clock=_clock(0.0, 0.5, 1.0, 4.0, 10.0, 10.0 + UPDATE_BUDGET_SECONDS))

    assert report.verdict == "ok"


def test_a_stopped_measurement_never_proposes_j1b():
    report = _measure(update_budget=0.5)

    assert report.verdict == "update-budget"
    assert report.step_seconds == pytest.approx(1.0)
    assert report.projection_seconds <= J1B_BUDGET_SECONDS  # the arithmetic alone would say "fits"
    assert report.fits_budget is False  # ... but the measurement stopped, so it does not


def test_a_non_finite_update_never_authorizes_anything():
    """A velocity returning NaN reported verdict="ok", loss=nan and fits_budget=True: a measurement
    of nothing, authorizing a job (R11 review, finding 5)."""

    def nan_velocity(z, timestep_2d, context):
        return jnp.full_like(z, jnp.nan)

    report = _measure(velocity_fn=nan_velocity)

    assert report.verdict == "nonfinite"
    assert report.fits_budget is False
    assert any("non-finite" in reason for reason in report.reasons)
    assert any("losses" in reason for reason in report.reasons)


def test_an_out_of_memory_failure_becomes_a_verdict_not_a_crash():
    def oom(*args, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while allocating 34359738368 bytes")

    report = _measure(velocity_fn=oom, clock=_clock(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))

    assert report.verdict == "oom"
    assert any("RESOURCE_EXHAUSTED" in reason for reason in report.reasons)
    assert report.fits_budget is False and report.step_seconds is None


@pytest.mark.parametrize(
    "message",
    ["BOOM: model kernel bug", "ZOOM level invalid", "the loomed shape is wrong"],
)
def test_a_bug_whose_message_merely_contains_oom_is_not_laundered(message):
    """The reviewer's probe: a bare substring classified ``BOOM`` as an allocation failure, which is
    how a kernel bug becomes "A3 needs a bigger machine"."""

    def broken(*args, **kwargs):
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message.split(":")[0]):
        _measure(velocity_fn=broken)


def test_a_failure_that_is_not_an_out_of_memory_propagates():
    def broken(*args, **kwargs):
        raise ValueError("z_video must have the production shape")

    with pytest.raises(ValueError, match="production shape"):
        _measure(velocity_fn=broken)


def test_the_measurement_is_one_example_only_unless_told_otherwise():
    z_start, z_i0, z_video, base_context = _inputs(batch=2)

    with pytest.raises(ValueError, match="exactly one example"):
        measure_single_update(
            _velocity_fn, z_start, z_i0, z_video, _SIGMAS, base_context[:_L], base_context,
            lr=1e-2, guide_scale=_GUIDE,
        )
    # ... and J1b's own fit probe opts out, because it is measuring the B=8 shape deliberately.
    probe = measure_single_update(
        _velocity_fn, z_start, z_i0, z_video, _SIGMAS, base_context[:_L], base_context,
        lr=1e-2, guide_scale=_GUIDE, job_batch=2, require_single_example=False, clock=_clock(*_TICKS),
        devices=_DEVICES,
    )
    assert probe.batch == 2 and probe.preliminary is False


def test_a_single_example_measurement_is_marked_preliminary():
    """B=1 timing is a compute estimate; B=8 has a different compile, sharding and HBM profile."""
    assert _measure().preliminary is True


def test_memory_is_reported_per_device_with_the_key_it_came_from():
    devices = (
        _FakeDevice("tpu:0", {"peak_bytes_in_use": 100, "bytes_in_use": 40}),
        _FakeDevice("tpu:1", {"peak_bytes_in_use": 700, "bytes_in_use": 90}),
    )

    report = _measure(devices=devices)

    assert report.peak_hbm_bytes == 700  # the maximum across addressable devices, not device 0
    assert report.current_hbm_bytes == 90
    assert [entry["device"] for entry in report.device_memory] == ["tpu:0", "tpu:1"]
    assert {entry["peak_key"] for entry in report.device_memory} == {"peak_bytes_in_use"}
    assert {entry["current_key"] for entry in report.device_memory} == {"bytes_in_use"}


def test_current_allocation_is_never_reported_as_a_peak():
    """``bytes_in_use`` is what is allocated right now; calling it a peak understates the high-water
    mark a job has to fit under."""
    report = _measure(devices=(_FakeDevice("tpu:0", {"bytes_in_use": 4096}),))

    assert report.peak_hbm_bytes is None  # no peak counter exists on this backend
    assert report.current_hbm_bytes == 4096
    assert report.device_memory[0]["peak_key"] is None


def test_missing_memory_stats_are_unavailable_evidence_not_zero():
    report = _measure(devices=(_FakeDevice("cpu:0", None),))

    assert report.peak_hbm_bytes is None and report.current_hbm_bytes is None
    assert report.device_memory[0]["peak_bytes"] is None


def test_the_report_carries_the_budgets_it_was_judged_against():
    report = _measure()

    assert report.budgets == {
        "compile_seconds": COMPILE_BUDGET_SECONDS,
        "update_seconds": UPDATE_BUDGET_SECONDS,
        "projection_seconds": J1B_BUDGET_SECONDS,
    }


def test_the_plan_budgets_are_pinned():
    assert COMPILE_BUDGET_SECONDS == 30 * 60
    assert UPDATE_BUDGET_SECONDS == 120
    assert J1B_BUDGET_SECONDS == 4 * 3600
    assert A3_ITERS == 300 and A3_EXAMPLES == 8


def test_the_measurement_really_runs_one_update():
    report = _measure()

    assert report.loss is not None and np.isfinite(report.loss)
    assert report.grad_norm is not None and np.isfinite(report.grad_norm)


# --------------------------------------------------------------------------- the real transformer


def test_a3_runs_through_a_real_tiny_wan_transformer():
    """One tiny-model case at the deployment guidance weight, per the round's contract."""
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning

    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    set_mesh = getattr(jax, "set_mesh", None)
    context_mesh = set_mesh(mesh) if set_mesh is not None else mesh
    channels, text_dim = 4, 32
    sigmas = jnp.asarray((1.0, 0.7, 0.4, 0.15, 0.0), jnp.float32)  # N = 4
    z_start = jax.random.normal(jax.random.PRNGKey(0), (1, channels, _F, _H, _W), jnp.float32)
    z_i0 = jax.random.normal(jax.random.PRNGKey(1), (1, channels, 1, _H, _W), jnp.float32)
    z_video = jax.random.normal(jax.random.PRNGKey(2), (1, channels, _F, _H, _W), jnp.float32)
    base_context = jax.random.normal(jax.random.PRNGKey(3), (_S, text_dim), jnp.float32)

    with nn_partitioning.axis_rules(()), context_mesh:
        model = WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=channels,
            out_channels=channels,
            text_dim=text_dim,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
        )

        def velocity_fn(z, timestep_2d, context):
            return model(hidden_states=z, timestep=timestep_2d, encoder_hidden_states=context)

        nulls, losses, grad_norms = direct_optimize_nulls(
            velocity_fn,
            z_start,
            z_i0,
            z_video,
            sigmas,
            base_context[:_L],
            base_context,
            iters=2,
            lr=1e-2,
            guide_scale=_GUIDE,
        )

    assert nulls.shape == (4, 1, _L, text_dim)
    assert bool(jnp.all(jnp.isfinite(losses))) and bool(jnp.all(jnp.isfinite(grad_norms)))
    assert float(jnp.min(grad_norms)) > 0.0, grad_norms
    assert float(losses[-1].sum()) <= float(losses[0].sum())


# --------------------------------------------------------------------------- remat


def test_the_backward_pass_rematerializes_rather_than_storing_every_step():
    """Memory is the reason A3 is a *conditional* job, so the remat is load-bearing.

    It cannot be caught numerically -- ``jax.remat`` is semantics-preserving by construction, and a
    side-by-side check of this rollout with and without it gives bit-identical gradients. It is
    caught structurally: the pin walks the gradient jaxpr's equations recursively (the primitive is
    nested inside ``scan``) and matches against a reviewed allowlist, rather than grepping the
    printed text -- which would also match a variable named ``remat_x``.

    **It fails closed.** Per the R11 remat ruling, an unknown renamed primitive is not tolerated: the
    assertion reports what it actually saw so the new lowered backward graph gets inspected before
    the allowlist is extended.
    """
    z_start, z_i0, z_video, base_context = _inputs()

    def loss(value):
        z_final = direct_rollout(
            _velocity_fn, value, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE
        )
        return jnp.sum(endpoint_future_mse(z_final, z_video))

    rematerialized, primitives = rematerializes(loss, _nulls())

    assert rematerialized, (
        f"no rematerialization primitive in the backward pass. Allowlist {sorted(REMAT_PRIMITIVES)}; "
        f"observed {sorted(primitives)}. Inspect the lowered backward graph before extending it."
    )
    assert "scan" in primitives  # ... and it is still a scan, not an unrolled 25-deep graph


def test_the_remat_pin_walks_nested_jaxprs_rather_than_matching_text():
    """The primitive lives inside the scan body, so a non-recursive walker would never see it."""
    z_start, z_i0, z_video, base_context = _inputs()

    def loss(value):
        return jnp.sum(endpoint_future_mse(
            direct_rollout(_velocity_fn, value, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE),
            z_video,
        ))

    top_level = jax.make_jaxpr(jax.grad(loss))(_nulls())
    shallow = {eqn.primitive.name for eqn in top_level.jaxpr.eqns}
    deep = jaxpr_primitives(top_level)

    assert not (shallow & REMAT_PRIMITIVES), "the fixture no longer nests remat; the walker is untested"
    assert deep & REMAT_PRIMITIVES
    assert shallow < deep


# --------------------------------------------------------------------------- the recurrence oracle


def test_the_rollout_is_bitwise_identical_to_the_reviewed_replay_operator():
    """The independent oracle: R4a's ``replay_with_nulls``, which has its own review and tests.

    A3's rollout must be the *same* recurrence, only differentiable. Reversing every ``dsigma`` to
    the wrong sign survived all forty of R11's first-cut tests -- the finite-difference check
    included, because it differentiates the same wrong forward. Only a comparison against a
    recurrence this round did not write can see it, so this pins the endpoint **and every trajectory
    element**, with a distinct null per step so a shared-null bug cannot hide either.
    """
    z_start, z_i0, _, base_context = _inputs()
    nulls = _nulls()  # distinct per step and per example
    assert float(jnp.max(jnp.abs(nulls[0] - nulls[1]))) > 0.0

    mine, my_trajectory = direct_rollout(
        _velocity_fn, nulls, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE, return_trajectory=True
    )
    theirs, their_trajectory = replay_with_nulls(
        _velocity_fn, z_start, z_i0, _SIGMAS, nulls, base_context, guide_scale=_GUIDE, return_trajectory=True
    )

    np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    np.testing.assert_array_equal(np.asarray(my_trajectory), np.asarray(their_trajectory))


@pytest.mark.parametrize("weight", [0.0, 1.0, 2.5, 7.5])
def test_the_recurrences_agree_at_every_guidance_weight(weight):
    """CFG algebra included: w = 1 collapses the null branch, w = 0 is the unguided limit."""
    z_start, z_i0, _, base_context = _inputs()
    nulls = _nulls()

    mine = direct_rollout(_velocity_fn, nulls, z_start, z_i0, _SIGMAS, base_context, guide_scale=weight)
    theirs = replay_with_nulls(_velocity_fn, z_start, z_i0, _SIGMAS, nulls, base_context, guide_scale=weight)

    np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


def test_the_two_recurrences_agree_on_a_longer_grid():
    """A two-step grid can hide an index error that a five-step grid cannot."""
    sigmas = jnp.asarray((1.0, 0.8, 0.55, 0.3, 0.1, 0.0), jnp.float32)
    rng = np.random.default_rng(5)
    nulls = jnp.asarray(rng.standard_normal((5, _B, _L, _D), dtype=np.float32))
    z_start, z_i0, _, base_context = _inputs()

    mine, my_trajectory = direct_rollout(
        _velocity_fn, nulls, z_start, z_i0, sigmas, base_context, guide_scale=_GUIDE, return_trajectory=True
    )
    theirs, their_trajectory = replay_with_nulls(
        _velocity_fn, z_start, z_i0, sigmas, nulls, base_context, guide_scale=_GUIDE, return_trajectory=True
    )

    np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))
    np.testing.assert_array_equal(np.asarray(my_trajectory), np.asarray(their_trajectory))


def test_the_gradient_matches_finite_differences_through_the_whole_rollout():
    """An independent check of the gradient *path*, not just of its sign or magnitude.

    Every other differentiability test here compares ``direct_rollout`` against itself, so a wrong
    but self-consistent path -- a ``stop_gradient`` on the conditional branch, say, which leaves the
    forward pass bit-identical -- satisfies all of them. Central differences do not go through the
    autodiff graph at all, so they do.
    """
    z_start, z_i0, z_video, base_context = _inputs(batch=1)
    nulls = _nulls(batch=1)

    def loss(value):
        z_final = direct_rollout(
            _velocity_fn, value, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE
        )
        return float(jnp.sum(endpoint_future_mse(z_final, z_video)))

    analytic = np.asarray(jax.grad(lambda v: jnp.sum(endpoint_future_mse(
        direct_rollout(_velocity_fn, v, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE), z_video
    )))(nulls))

    eps = 1e-3
    for step in range(_STEPS):
        for row in range(_L):
            index = (step, 0, row, 0)
            up = loss(nulls.at[index].add(eps))
            down = loss(nulls.at[index].add(-eps))
            numeric = (up - down) / (2 * eps)
            assert numeric == pytest.approx(float(analytic[index]), rel=2e-2, abs=1e-5), index


def test_grad_norms_are_per_example_across_every_step():
    """``grads`` is ``[N, B, L, D]``: reducing the wrong pair of axes gives a per-*step* norm that has
    the same shape as the per-example one whenever N happens to equal B."""
    z_start, z_i0, z_video, base_context = _inputs(batch=1)

    _, _, grad_norms = direct_optimize_nulls(
        _velocity_fn, z_start, z_i0, z_video, _SIGMAS, base_context[:_L], base_context,
        iters=1, lr=1e-2, guide_scale=_GUIDE,
    )

    assert grad_norms.shape == (1, 1)  # one iteration, one example -- not (1, _STEPS)
    grads = jax.grad(lambda v: jnp.sum(endpoint_future_mse(
        direct_rollout(_velocity_fn, v, z_start, z_i0, _SIGMAS, base_context, guide_scale=_GUIDE), z_video
    )))(jnp.broadcast_to(base_context[:_L], (_STEPS, 1, _L, _D)))
    expected = jnp.sqrt(jnp.sum(grads**2, axis=(0, 2, 3)))

    np.testing.assert_allclose(np.asarray(grad_norms[0]), np.asarray(expected), rtol=1e-5, atol=1e-7)


def test_an_examples_gradient_does_not_depend_on_who_it_is_batched_with():
    """The Σ-objective, measured where it is actually visible.

    Comparing optimized *nulls* cannot see this: Adam divides by sqrt(v), so scaling every gradient
    by 1/B leaves the updates almost unchanged. The gradient norms are not normalized, and they are.
    """
    z_start, z_i0, z_video, base_context = _inputs(batch=_B)
    kwargs = {"iters": 1, "lr": 1e-2, "guide_scale": _GUIDE}

    _, _, together = direct_optimize_nulls(
        _velocity_fn, z_start, z_i0, z_video, _SIGMAS, base_context[:_L], base_context, **kwargs
    )
    _, _, alone = direct_optimize_nulls(
        _velocity_fn, z_start[:1], z_i0[:1], z_video[:1], _SIGMAS, base_context[:_L], base_context, **kwargs
    )

    np.testing.assert_allclose(np.asarray(together[:, :1]), np.asarray(alone), rtol=1e-5, atol=1e-7)


def test_the_compiled_executable_is_invoked_exactly_once(monkeypatch):
    """"Compile + execute exactly one update" is a count, so it is asserted as one.

    R11's first cut called the jitted function twice and subtracted; the injected clock cannot see an
    extra execution hidden inside the compile phase, but the synchronization barrier can.
    """
    executions = []
    real = jax.block_until_ready

    def counting(value):
        executions.append(1)
        return real(value)

    monkeypatch.setattr(jax, "block_until_ready", counting)
    report = _measure()

    assert report.verdict == "ok"
    assert len(executions) == 1, f"the measurement executed the kernel {len(executions)} times"


def test_the_remat_pin_reports_absence_when_there_is_no_remat():
    """The negative control that makes the pin mean something: a rollout without the decorator must
    come back as "not rematerialized, here is what I saw", never as an automatic pass."""
    z_start, z_i0, z_video, base_context = _inputs()
    nulls = _nulls()

    def plain_loss(value):
        # The same recurrence shape, deliberately WITHOUT @jax.remat.
        def body(carry, step):
            return carry + jnp.sum(step) * 0.0 + carry * 0.0, None

        total, _ = jax.lax.scan(body, jnp.zeros(()), value.reshape(value.shape[0], -1))
        return total + jnp.sum(value**2)

    rematerialized, primitives = rematerializes(plain_loss, nulls)

    assert rematerialized is False
    assert not (primitives & REMAT_PRIMITIVES)
    assert primitives, "the walker returned nothing at all, so it is not walking"
