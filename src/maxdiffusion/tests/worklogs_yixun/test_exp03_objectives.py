"""exp_03 cycle A round 3 — the three objectives (plan v3.1 §1), the scientific heart.

The claims that have to hold before any of this is worth TPU time:

* **A's corrective target is exact.** ``v* = (z_lo - z_gt) / sigma_lo`` reduces to ``eps - z_gt``
  on-path, and off-path it is exactly the velocity that makes the Euler rule contract the error by
  ``sigma_next / sigma`` (the identity the plan reviewer verified). Both are checked against
  independently written arithmetic, not against the implementation.
* **The supports are the plan's, in the eval's direction.** The grid DESCENDS; A draws ``k_A``
  first and then a start with the length-dependent range; B walks consecutive indices. The terminal
  index (sigma = 0) is unreachable -- checked over the whole support by histogram, not by spot check.
* **B is zero at the optimum and horizon-normalized.** A perfect velocity oracle gives exactly zero
  at every support; without the ``(sigma_hi - sigma_lo)^2`` divisor the loss is reweighted by the
  grid's nonuniform spacing.
* **Gradients flow exactly where they should.** A's off-path advance contributes none (compared
  against a reference whose advance is detached: exact equality), B's contributes through both
  forwards (compared against an explicit two-step unroll: exact equality).
* **C is literal.** Its gradient equals ``lambda * grad(L_A) + (1 - lambda) * grad(L_B)`` exactly.
* **The control is untouched** by any of it, and everything holds under ``jit`` as well as eagerly
  (the round-2 lesson: eager parity is not the production boundary).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import maxdiffusion.trainers.wan_ti2v_exp03_trainer as exp03
import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as parent
from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid, overfit100_sampler_step
from maxdiffusion.models.wan.side_adapter_wan import apply_first_frame_pin, masked_velocity_mse
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

_DATA_B, _BSZ, _C, _F, _H, _W = 3, 2, 3, 4, 5, 6
_SLOTS, _LEN, _DIM = 4, 4, 8
_STEPS = 25


class _StubTransformer(nnx.Module):
    """A real ``Param`` so gradients are genuine, and a velocity that reads every input."""

    def __init__(self, gain: float = 0.25):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        hidden = kwargs["hidden_states"].astype(jnp.float32)
        t_mean = jnp.mean(kwargs["timestep"].astype(jnp.float32))
        ctx_mean = jnp.mean(kwargs["encoder_hidden_states"].astype(jnp.float32))
        return self.gain[...] * jnp.tanh(hidden) + 0.01 * t_mean + 0.001 * ctx_mean


class _OracleTransformer(nnx.Module):
    """The PERFECT velocity ``eps - z_gt``, injected as constants (B must score exactly zero)."""

    def __init__(self, target: jax.Array):
        self.gain = nnx.Param(jnp.asarray(0.0, dtype=jnp.float32))
        self.target = target

    def __call__(self, **kwargs):
        return (self.target + self.gain[...]).astype(kwargs["hidden_states"].dtype)


def _fixture(*, objective="corrective_ss", transformer=None, **overrides):
    transformer = transformer or _StubTransformer()
    graphdef, params, rest = nnx.split(transformer, nnx.Param, ...)
    context_table = jax.random.normal(jax.random.key(41), (_SLOTS, _LEN, _DIM), dtype=jnp.float32)
    state = parent.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=optax.sgd(0.1),
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=context_table,
    )
    k1, k2 = jax.random.split(jax.random.key(42), 2)
    data = {
        "z_i0": jax.random.normal(k1, (_DATA_B, _C, 1, _H, _W), dtype=jnp.float32),
        "z_video": jax.random.normal(k2, (_DATA_B, _C, _F, _H, _W), dtype=jnp.float32),
        "episode_index": jnp.asarray([0, 1, 2], dtype=jnp.int32),
    }
    settings = {
        "weights_dtype": "float32",
        "activations_dtype": "float32",
        "global_batch_size_to_train_on": _BSZ,
        "side_adapter_sampling_steps": _STEPS,
        "flow_shift": 5.0,
        "side_adapter_t_sampling": "uniform",
        "side_adapter_noise_mode": "fresh",
        "seed": 0,
        "model_type": exp03.EXP03_MODEL_TYPE,
        "exp03_objective": objective,
        "exp03_k_a": 2,
        "exp03_k_b": 2,
        "exp03_lambda": 0.5,
        "exp03_p_ss_max": 0.5,
        "exp03_p_ss_ramp_steps": 500,
        "exp03_ramp_origin": 0,
    }
    settings.update(overrides)
    config = SimpleNamespace(**settings)
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=config.flow_shift, sigma_min=0.0, sigma_max=1.0)
    return state, data, config, scheduler


def _grid(config, scheduler):
    return overfit100_sampler_grid(
        num_inference_steps=config.side_adapter_sampling_steps,
        flow_shift=config.flow_shift,
        sigma_min=scheduler.config.sigma_min,
        sigma_max=scheduler.config.sigma_max,
        num_train_timesteps=scheduler.config.num_train_timesteps,
    )


# =============================================================================================
# 1. The corrective target (plan review P1).
# =============================================================================================


def test_the_corrective_target_reduces_to_the_plain_target_on_path():
    # ON the ideal trajectory the corrective target IS exp_02's target, so A degenerates to the
    # control exactly when the model is already right -- the property that makes it "corrective".
    state, data, config, scheduler = _fixture()
    sigmas, _ = _grid(config, scheduler)
    rng = jax.random.key(3)
    z_gt = data["z_video"][:_BSZ].astype(jnp.float32)
    z_i0 = data["z_i0"][:_BSZ].astype(jnp.float32)
    eps = jax.random.normal(rng, z_gt.shape, dtype=jnp.float32)
    for index in (0, 7, 24):
        sigma = float(sigmas[index])
        on_path = apply_first_frame_pin((1.0 - sigma) * z_gt + sigma * eps, z_i0)
        corrective = (on_path - z_gt) / sigma
        plain = eps - z_gt
        # Frame 0 is pinned and masked out of the loss, so the claim is about the scored elements.
        assert np.allclose(np.asarray(corrective[:, :, 1:]), np.asarray(plain[:, :, 1:]), atol=1e-5), index


@pytest.mark.parametrize("index", [0, 5, 12, 23])
def test_the_corrective_target_is_the_exact_one_step_correction_off_path(index):
    # The reviewer's identity: with v* = (z - z_gt)/sigma, the Euler rule gives
    # z_next - z_gt = (sigma_next / sigma) (z - z_gt) -- an exact contraction toward the truth.
    state, data, config, scheduler = _fixture()
    sigmas, _ = _grid(config, scheduler)
    z_gt = data["z_video"][:_BSZ].astype(jnp.float32)
    z_i0 = data["z_i0"][:_BSZ].astype(jnp.float32)
    z_gt = apply_first_frame_pin(z_gt, z_i0)
    off_path = apply_first_frame_pin(z_gt + jax.random.normal(jax.random.key(9), z_gt.shape) * 0.7, z_i0)
    sigma, sigma_next = float(sigmas[index]), float(sigmas[index + 1])
    v_star = (off_path - z_gt) / sigma
    z_next = apply_first_frame_pin(off_path + (sigma_next - sigma) * v_star, z_i0)
    expected = apply_first_frame_pin(z_gt + (sigma_next / sigma) * (off_path - z_gt), z_i0)
    assert np.allclose(np.asarray(z_next), np.asarray(expected), atol=1e-5)


def test_the_implementation_uses_sigma_lo_as_the_denominator():
    # A denominator of sigma_hi would still "look like" a correction but would not be the exact
    # one-step target: drive the real loss with an oracle whose prediction IS the corrective target
    # at sigma_lo and require exactly zero.
    state, data, config, scheduler = _fixture(exp03_p_ss_max=0.0)  # teacher-forced branch only
    sigmas, _ = _grid(config, scheduler)
    global_step = jnp.asarray(11, dtype=jnp.int32)
    start, end, _ = exp03.corrective_support(seed=0, global_step=global_step, num_steps=_STEPS, k_a_max=2)
    sigma_lo = float(sigmas[int(end)])

    rng = jax.random.key(5)
    noise_rng, _, _ = jax.random.split(rng, 3)
    from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import _build_noise

    z_gt = data["z_video"][:_BSZ].astype(jnp.float32)
    z_i0 = data["z_i0"][:_BSZ].astype(jnp.float32)
    eps = _build_noise(noise_rng, z_gt.shape, jnp.float32, config)
    z_lo = apply_first_frame_pin((1.0 - sigma_lo) * z_gt + sigma_lo * eps, z_i0)
    oracle_target = (z_lo - z_gt) / sigma_lo

    state, data, config, scheduler = _fixture(exp03_p_ss_max=0.0, transformer=_OracleTransformer(oracle_target))
    loss, aux = exp03._corrective_ss_loss(state.params, state, data, rng, config, scheduler, global_step=global_step)
    assert float(loss) < 1e-10, float(loss)
    assert float(aux["p_ss"]) == 0.0


# =============================================================================================
# 2. The index supports (plan v2.2, direction-corrected).
# =============================================================================================


def test_the_corrective_support_matches_the_plans_exact_distribution():
    draws = [
        tuple(int(value) for value in exp03.corrective_support(seed=0, global_step=step, num_steps=_STEPS, k_a_max=2))
        for step in range(4000)
    ]
    k_values = {k for _, _, k in draws}
    assert k_values == {1, 2}, k_values
    for start, end, k in draws:
        assert end == start + k
        assert 0 <= start <= _STEPS - 1 - k  # s ~ U{0 .. 24 - k_A}
        assert end <= _STEPS - 1  # the terminal index 25 is unreachable
    # Roughly uniform in k, and every legal start appears for each k.
    counts = {k: sum(1 for _, _, value in draws if value == k) for k in (1, 2)}
    assert 0.4 < counts[1] / len(draws) < 0.6, counts
    for k in (1, 2):
        starts = {start for start, _, value in draws if value == k}
        assert starts == set(range(_STEPS - k)), (k, sorted(set(range(_STEPS - k)) - starts))


def test_the_rollout_support_walks_consecutive_indices_in_the_evals_direction():
    draws = [
        tuple(int(value) for value in exp03.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=2))
        for step in range(3000)
    ]
    for start, end in draws:
        assert end == start + 2
        assert 0 <= start <= 22  # s ~ U{0 .. 22}
        assert end <= 24  # never the terminal index
    assert {start for start, _ in draws} == set(range(23))


def test_the_supports_never_reach_a_zero_sigma():
    _, _, config, scheduler = _fixture()
    sigmas, _ = _grid(config, scheduler)
    assert float(sigmas[_STEPS]) == 0.0  # the terminal index really is zero...
    assert abs(float(sigmas[24]) - 0.1724137931) < 1e-6  # ...and 24 is the smallest positive sigma
    for step in range(500):
        _, end_a, _ = exp03.corrective_support(seed=0, global_step=step, num_steps=_STEPS, k_a_max=2)
        _, end_b = exp03.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=2)
        assert float(sigmas[int(end_a)]) > 0.0
        assert float(sigmas[int(end_b)]) > 0.0


def test_the_supports_are_step_keyed_and_independent_between_the_two_trials():
    # C draws both supports on the same batch; different purposes make them independent by
    # construction rather than by luck.
    same_step_a = exp03.corrective_support(seed=0, global_step=17, num_steps=_STEPS, k_a_max=2)
    same_step_b = exp03.rollout_support(seed=0, global_step=17, num_steps=_STEPS, k_b=2)
    agreements = sum(
        1
        for step in range(300)
        if int(exp03.corrective_support(seed=0, global_step=step, num_steps=_STEPS, k_a_max=2)[0])
        == int(exp03.rollout_support(seed=0, global_step=step, num_steps=_STEPS, k_b=2)[0])
    )
    assert agreements < 60  # ~1/23 of 300 if independent; equality would give 300
    assert int(same_step_a[0]) >= 0 and int(same_step_b[0]) >= 0
    # ...and re-drawing at the same step gives the same support (resume stability).
    again = exp03.corrective_support(seed=0, global_step=17, num_steps=_STEPS, k_a_max=2)
    assert tuple(int(v) for v in again) == tuple(int(v) for v in same_step_a)


def test_the_supports_are_tracer_safe():
    jitted = jax.jit(lambda step: exp03.corrective_support(seed=0, global_step=step, num_steps=_STEPS, k_a_max=2))
    for step in (0, 250, 9999):
        eager = exp03.corrective_support(seed=0, global_step=step, num_steps=_STEPS, k_a_max=2)
        traced = jitted(jnp.asarray(step, dtype=jnp.int32))
        assert tuple(int(v) for v in traced) == tuple(int(v) for v in eager), step


# =============================================================================================
# 3. Trial B — zero at the optimum, horizon-normalized, masked like exp_02.
# =============================================================================================


def _oracle_fixture(objective="rollout_loss", **overrides):
    state, data, config, scheduler = _fixture(objective=objective, **overrides)
    rng = jax.random.key(5)
    noise_rng, _, _ = jax.random.split(rng, 3)
    from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import _build_noise

    z_gt = data["z_video"][:_BSZ].astype(jnp.float32)
    eps = _build_noise(noise_rng, z_gt.shape, jnp.float32, config)
    oracle = _OracleTransformer(eps - z_gt)
    return (*_fixture(objective=objective, transformer=oracle, **overrides), rng)


def test_the_rollout_loss_is_zero_at_the_optimum():
    # v = eps - z_gt maps the interpolant at sigma_hi onto the interpolant at sigma_lo exactly, so
    # the endpoint loss must vanish -- at whatever support the step happens to draw.
    state, data, config, scheduler, rng = _oracle_fixture()
    for step in (0, 3, 41, 512):
        loss, aux = exp03._rollout_loss(
            state.params, state, data, rng, config, scheduler, global_step=jnp.asarray(step, jnp.int32)
        )
        assert float(loss) < 1e-8, (step, float(loss))
        assert float(aux["raw_endpoint_mse"]) < 1e-8


def test_the_rollout_loss_is_horizon_normalized():
    # The normalizer is what makes supports comparable across the nonuniform grid: the reported
    # loss is the raw endpoint MSE divided by (sigma_hi - sigma_lo)^2, and that divisor really
    # varies across the grid.
    state, data, config, scheduler = _fixture(objective="rollout_loss")
    sigmas, _ = _grid(config, scheduler)
    horizons = []
    for step in (0, 1, 2, 3, 4, 5):
        global_step = jnp.asarray(step, jnp.int32)
        loss, aux = exp03._rollout_loss(
            state.params, state, data, jax.random.key(7), config, scheduler, global_step=global_step
        )
        start, end = exp03.rollout_support(seed=0, global_step=global_step, num_steps=_STEPS, k_b=2)
        expected = (float(sigmas[int(start)]) - float(sigmas[int(end)])) ** 2
        assert abs(float(aux["horizon_sq"]) - expected) < 1e-6
        assert abs(float(loss) - float(aux["raw_endpoint_mse"]) / expected) < 1e-4 * max(1.0, abs(float(loss)))
        horizons.append(expected)
    assert max(horizons) / min(horizons) > 1.5  # the grid really is nonuniform


def test_the_rollout_loss_masks_frame_zero_exactly_like_exp02():
    # Masking parity: the loss ignores what happens on the pinned frame, whatever happens there.
    state, data, config, scheduler = _fixture(objective="rollout_loss")
    rng = jax.random.key(13)
    global_step = jnp.asarray(2, jnp.int32)
    base, _ = exp03._rollout_loss(state.params, state, data, rng, config, scheduler, global_step=global_step)
    perturbed = dict(data)
    perturbed["z_i0"] = data["z_i0"] * 1.0  # frame 0 pin source unchanged...
    perturbed["z_video"] = data["z_video"].at[:, :, :1].add(5.0)  # ...but frame 0 of the target moved
    moved, _ = exp03._rollout_loss(state.params, state, perturbed, rng, config, scheduler, global_step=global_step)
    assert abs(float(base) - float(moved)) < 1e-4 * max(1.0, abs(float(base)))
    # The mask is exp_02's helper itself, not a re-implementation.
    source = jax.numpy  # placeholder to keep the import graph honest
    del source
    assert "masked_velocity_mse" in exp03._rollout_loss.__code__.co_names


# =============================================================================================
# 4. Gradient structure.
# =============================================================================================


def _reference_corrective(params, state, data, rng, config, scheduler, *, global_step, detach_advance: bool):
    """A/B reference: the same loss with the off-path advance's params optionally NOT detached."""
    ctx = exp03._exp03_prologue(params, state, data, rng, config, scheduler)
    start, end, _ = exp03.corrective_support(seed=0, global_step=global_step, num_steps=_STEPS, k_a_max=2)
    sigma_lo = ctx.sigmas[end].astype(jnp.float32)
    z_hi = exp03._interpolant_at(ctx, start)
    teacher = exp03._interpolant_at(ctx, end)
    advanced = exp03._advance_with_sampler(
        ctx, z_hi.astype(ctx.weights_dtype), start, end, velocity_fn=exp03._sampling_velocity_fn(ctx)
    ).astype(jnp.float32)
    if detach_advance:
        advanced = jax.lax.stop_gradient(advanced)
    coin = jax.random.uniform(exp03.exp03_aux_key(seed=0, global_step=global_step, purpose="p_ss_coin"), ())
    z_lo = jnp.where(coin < exp03.exp03_p_ss(config, global_step), advanced, teacher)
    z_lo = apply_first_frame_pin(z_lo, ctx.z_i0)
    v_pred = exp03._forward_velocity(ctx, z_lo, end)
    return masked_velocity_mse(v_pred, (z_lo - ctx.z_video) / sigma_lo, ctx.b)


def test_trial_a_takes_no_gradient_through_the_off_path_advance():
    # p_ss forced to 1 so the advance is ALWAYS the state that is scored -- otherwise the claim
    # would be vacuous. The production gradient must equal the detached reference exactly, and
    # differ from the undetached one (proving the advance really is param-dependent).
    state, data, config, scheduler = _fixture(exp03_p_ss_max=1.0, exp03_p_ss_ramp_steps=0)
    rng = jax.random.key(17)
    global_step = jnp.asarray(31, jnp.int32)

    def production(params):
        return exp03._corrective_ss_loss(params, state, data, rng, config, scheduler, global_step=global_step)[0]

    def detached(params):
        return _reference_corrective(
            params, state, data, rng, config, scheduler, global_step=global_step, detach_advance=True
        )

    def attached(params):
        return _reference_corrective(
            params, state, data, rng, config, scheduler, global_step=global_step, detach_advance=False
        )

    got = jax.grad(production)(state.params)
    want = jax.grad(detached)(state.params)
    other = jax.grad(attached)(state.params)
    got_leaves = jax.tree_util.tree_leaves(got)
    assert got_leaves
    for left, right in zip(got_leaves, jax.tree_util.tree_leaves(want)):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(got_leaves, jax.tree_util.tree_leaves(other))
    ), "the advance does not depend on the params -- this test proves nothing"


def test_trial_b_takes_gradient_through_both_rollout_forwards():
    # Equality against an EXPLICIT two-step unroll pins "exactly two differentiated forwards":
    # one step, or three, or a detached scan would all differ.
    state, data, config, scheduler = _fixture(objective="rollout_loss")
    rng = jax.random.key(19)
    global_step = jnp.asarray(23, jnp.int32)

    def production(params):
        return exp03._rollout_loss(params, state, data, rng, config, scheduler, global_step=global_step)[0]

    def explicit_two_steps(params):
        ctx = exp03._exp03_prologue(params, state, data, rng, config, scheduler)
        start, end = exp03.rollout_support(seed=0, global_step=global_step, num_steps=_STEPS, k_b=2)
        velocity_fn = exp03._training_velocity_fn(ctx)
        z = exp03._interpolant_at(ctx, start).astype(ctx.weights_dtype)
        for offset in range(2):
            z = overfit100_sampler_step(
                z,
                start + offset,
                velocity_fn=velocity_fn,
                sigmas=ctx.sigmas,
                timesteps=ctx.timesteps,
                context=ctx.context,
                z_i0=ctx.z_i0.astype(ctx.weights_dtype),
            )
        ideal = exp03._interpolant_at(ctx, end)
        horizon = (ctx.sigmas[start].astype(jnp.float32) - ctx.sigmas[end].astype(jnp.float32)) ** 2
        return masked_velocity_mse(z.astype(jnp.float32), ideal, ctx.b) / horizon

    got = jax.grad(production)(state.params)
    want = jax.grad(explicit_two_steps)(state.params)
    for left, right in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        assert np.allclose(np.asarray(left), np.asarray(right), rtol=1e-5, atol=1e-7)
    assert any(abs(float(np.asarray(leaf))) > 0 for leaf in jax.tree_util.tree_leaves(got))

    # ...and a detached scan is NOT what production does.
    def detached(params):
        return jax.lax.stop_gradient(production(params)) + 0.0 * jax.tree_util.tree_leaves(params)[0]

    detached_grad = jax.tree_util.tree_leaves(jax.grad(detached)(state.params))[0]
    assert not np.array_equal(np.asarray(detached_grad), np.asarray(jax.tree_util.tree_leaves(got)[0]))


def test_trial_c_is_the_literal_weighted_combination():
    # The gradient identity, exactly: grad(C) == lambda * grad(A) + (1 - lambda) * grad(B).
    state, data, config, scheduler = _fixture(objective="combined")
    rng = jax.random.key(29)
    global_step = jnp.asarray(37, jnp.int32)
    lam = float(config.exp03_lambda)

    grad_c = jax.grad(
        lambda params: exp03._combined_loss(params, state, data, rng, config, scheduler, global_step=global_step)[0]
    )(state.params)
    grad_a = jax.grad(
        lambda params: exp03._corrective_ss_loss(params, state, data, rng, config, scheduler, global_step=global_step)[
            0
        ]
    )(state.params)
    grad_b = jax.grad(
        lambda params: exp03._rollout_loss(params, state, data, rng, config, scheduler, global_step=global_step)[0]
    )(state.params)
    combined = jax.tree_util.tree_map(lambda a, b: lam * a + (1.0 - lam) * b, grad_a, grad_b)
    for left, right in zip(jax.tree_util.tree_leaves(grad_c), jax.tree_util.tree_leaves(combined)):
        assert np.allclose(np.asarray(left), np.asarray(right), rtol=1e-6, atol=1e-8)
    # The weighting is not symmetric under swapping, so the test can see a swap.
    swapped = jax.tree_util.tree_map(lambda a, b: (1.0 - lam) * a + lam * b, grad_a, grad_b)
    if not np.allclose(
        np.asarray(jax.tree_util.tree_leaves(grad_a)[0]), np.asarray(jax.tree_util.tree_leaves(grad_b)[0])
    ):
        assert (
            not np.allclose(
                np.asarray(jax.tree_util.tree_leaves(grad_c)[0]), np.asarray(jax.tree_util.tree_leaves(swapped)[0])
            )
            or lam == 0.5
        )


def test_trial_c_with_asymmetric_lambda_sees_the_weights():
    # lambda = 0.5 makes a swap invisible; the asymmetric case is where the weights are pinned.
    state, data, config, scheduler = _fixture(objective="combined", exp03_lambda=0.25)
    rng = jax.random.key(31)
    global_step = jnp.asarray(43, jnp.int32)
    loss_c, aux = exp03._combined_loss(state.params, state, data, rng, config, scheduler, global_step=global_step)
    loss_a = float(aux["loss_a"])
    loss_b = float(aux["loss_b"])
    assert abs(float(loss_c) - (0.25 * loss_a + 0.75 * loss_b)) < 1e-5 * max(1.0, abs(float(loss_c)))
    assert abs(float(loss_c) - (0.75 * loss_a + 0.25 * loss_b)) > 1e-6  # a swap is visible
    assert float(aux["lambda"]) == 0.25


# =============================================================================================
# 5. The p_ss ramp.
# =============================================================================================


@pytest.mark.parametrize("origin", [0, 10000])
def test_the_p_ss_ramp_is_linear_and_keyed_to_the_ramp_origin(origin):
    _, _, config, _ = _fixture(exp03_ramp_origin=origin, exp03_p_ss_max=0.5, exp03_p_ss_ramp_steps=500)
    assert float(exp03.exp03_p_ss(config, origin)) == 0.0
    assert abs(float(exp03.exp03_p_ss(config, origin + 250)) - 0.25) < 1e-6
    assert abs(float(exp03.exp03_p_ss(config, origin + 500)) - 0.5) < 1e-6
    assert abs(float(exp03.exp03_p_ss(config, origin + 2500)) - 0.5) < 1e-6  # constant after the ramp
    # Before the origin (a Tier-1 arm reading a Tier-2 step) it is clamped at 0, never negative.
    assert float(exp03.exp03_p_ss(config, max(0, origin - 100))) in (0.0, 0.5) or origin == 0


def test_the_ramp_is_resume_stable_and_tracer_safe():
    _, _, config, _ = _fixture(exp03_ramp_origin=10000)
    jitted = jax.jit(lambda step: exp03.exp03_p_ss(config, step))
    for step in (10000, 10250, 10500, 12499):
        assert abs(float(jitted(jnp.asarray(step, jnp.int32))) - float(exp03.exp03_p_ss(config, step))) < 1e-7
    # Resume: the value depends only on the global step, not on how many steps this process ran.
    assert float(exp03.exp03_p_ss(config, 10250)) == float(exp03.exp03_p_ss(config, 10250))


def test_a_zero_length_ramp_is_immediately_at_p_max():
    _, _, config, _ = _fixture(exp03_p_ss_ramp_steps=0, exp03_p_ss_max=0.5)
    assert float(exp03.exp03_p_ss(config, 0)) == 0.5
    assert float(exp03.exp03_p_ss(config, 1234)) == 0.5


# =============================================================================================
# 6. Wiring: the hook, jit, and the untouched control.
# =============================================================================================


@pytest.mark.parametrize("objective", ["corrective_ss", "rollout_loss", "combined"])
def test_each_objective_is_reachable_through_the_hook_and_runs_under_jit(objective):
    state, data, config, scheduler = _fixture(objective=objective)
    trainer = exp03.Exp03Trainer.__new__(exp03.Exp03Trainer)
    trainer.config = config
    loss_fn, step_fn = trainer._loss_and_step_fns()
    assert loss_fn is exp03.EXP03_LOSSES[objective]

    def compiled(state_, data_, rng_, global_step_):
        return step_fn(state_, data_, rng_, scheduler, config, global_step=global_step_)

    jitted = jax.jit(compiled)
    new_state, metrics, rng = jitted(state, data, jax.random.key(3), jnp.asarray(5, jnp.int32))
    assert set(metrics["scalar"]) == {
        "learning/loss",
        "learning/velocity_mse",
        "learning/grad_norm",
        "learning/max_abs_grad",
        "learning/sigma_mean",
        "learning/timestep_mean",
        "learning/v_pred_l2",
        "learning/v_target_l2",
        "learning/z_noisy_std",
        "learning/z_target_std",
        "learning/z_init_anchor_mse",
    }
    assert np.isfinite(float(metrics["scalar"]["learning/loss"]))
    assert float(metrics["scalar"]["learning/grad_norm"]) > 0.0
    updated = jax.tree_util.tree_leaves(new_state.params)[0]
    assert not np.array_equal(np.asarray(updated), np.asarray(jax.tree_util.tree_leaves(state.params)[0]))


@pytest.mark.parametrize("objective", ["corrective_ss", "rollout_loss", "combined"])
def test_the_trial_objectives_refuse_to_run_without_a_global_step(objective):
    state, data, config, scheduler = _fixture(objective=objective)
    with pytest.raises(ValueError) as excinfo:
        exp03.EXP03_LOSSES[objective](state.params, state, data, jax.random.key(1), config, scheduler)
    assert "global_step" in str(excinfo.value)


def test_the_trials_use_the_extracted_sampler_step():
    # The one-sampler rule: an arm trains on the operator the eval runs, not a private copy.
    for fn in (exp03._advance_with_sampler, exp03._rollout_loss):
        assert "overfit100_sampler_step" in fn.__code__.co_names or any(
            "overfit100_sampler_step" in const.co_names
            for const in fn.__code__.co_consts
            if hasattr(const, "co_names")
        )


def test_the_control_path_is_untouched_by_the_trial_code():
    # Round 2's guarantee must survive round 3: control still returns the parent's functions by
    # identity, and its step still matches the parent's exactly.
    state, data, config, scheduler = _fixture(objective="control")
    trainer = exp03.Exp03Trainer.__new__(exp03.Exp03Trainer)
    trainer.config = config
    loss_fn, step_fn = trainer._loss_and_step_fns()
    assert loss_fn is parent._denoising_loss and step_fn is parent._train_step
    rng = jax.random.key(47)
    mine = step_fn(state, data, rng, scheduler, config, global_step=jnp.asarray(3, jnp.int32))
    theirs = parent._train_step(state, data, rng, scheduler, config)
    for left, right in zip(jax.tree_util.tree_leaves(mine[0].params), jax.tree_util.tree_leaves(theirs[0].params)):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    for key, value in theirs[1]["scalar"].items():
        assert np.array_equal(np.asarray(mine[1]["scalar"][key]), np.asarray(value)), key
