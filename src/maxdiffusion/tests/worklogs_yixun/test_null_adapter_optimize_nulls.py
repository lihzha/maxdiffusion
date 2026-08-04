"""exp_04 R3 — ``optimize_null_embeddings``: per-step null-text optimization (plan §3 step 2).

This is Algorithm 1 of Mokady et al. as the reference implements it
(``third_party/Wan2.2/scripts/embedding_search.py:575-678``, submodule pin f370228): for each
sampler step i, hold the pivot z̄_i fixed and run ``inner_iters`` Adam steps on ∅_i so that one CFG
Euler step from z̄_i lands on the inversion pivot traj[i+1]; then lock ∅_i, take one more forward with
it to advance z̄_{i+1}, and warm-start ∅_{i+1} ← ∅_i.

Four properties carry the round, and each has a test that fails without it:

- **The algorithm itself.** ``test_matches_the_hand_rolled_reference`` writes the whole loop out
  literally -- explicit Python loops, hand-rolled Adam (b1 0.9, b2 0.999, eps 1e-8, bias correction)
  -- and compares all four outputs. That single test pins the objective, the CFG mixing, both pins,
  the locked-∅ advance, the warm start, fresh-Adam-per-step, and the optax/torch Adam parity that
  plan §8's deviation register requires.
- **Convexity/descent.** With a velocity linear in the context the inner problem is a convex
  quadratic in ∅, so the final recorded loss must sit below the initial one for every step and
  example. No monotonicity is asserted (plan F13): Adam is not required to descend at every iterate.
- **CFG algebra.** At w=1 the ∅ branch cancels out of v_cfg exactly, so the gradient must be exactly
  zero and ∅ must never move. At w=5 the Euler increment must equal v_unc + 5(v_cond − v_unc).
- **Call structure.** v_cond is ∅-independent, so recomputing it inside the inner loop changes no
  output -- it is only observable as wasted forwards. The counting tests therefore run under
  ``jax.disable_jit()``, where ``lax.scan`` executes its body eagerly once per iteration, and count
  real calls: exactly 1 conditional + J unconditional + 1 advance per outer step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import optimize_null_embeddings


# Small enough that the hand-rolled reference is cheap, real enough to keep every axis distinct.
_B, _C, _F, _H, _W = 2, 2, 2, 4, 6
_S, _D, _L = 8, 4, 3  # context rows, context dim, null-token count
_SIGMAS = (1.0, 0.6, 0.0)  # N = 2 steps
_NUM_TRAIN_TIMESTEPS = 1000  # hardcoded, not imported -- see the R2 test for why
_ADAM = (0.9, 0.999, 1e-8)  # torch.optim.Adam defaults, which optax must reproduce
# The oracle's sensitivity to the evaluation point and to the timestep tensor. Both are small enough
# not to swamp the context term (which is what is being optimized) and large enough that feeding the
# model a wrong z or a wrong timestep moves the trajectory far outside every tolerance here.
_C_Z, _C_T = 0.05, 1e-4


def _f32_bits(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x).astype(jnp.float32), jnp.uint32))


def _pin(z, z_i0):
    return z.at[:, :, :1].set(z_i0[:, :, :1])


def _timestep_2d(sigma, batch=_B, f_lat=_F, h_lat=_H, w_lat=_W):
    """The per-token timestep, restated independently of the sampler helper (pinned in R2)."""
    tokens_per_frame = (h_lat // 2) * (w_lat // 2)
    full = jnp.full((batch, f_lat * tokens_per_frame), sigma * _NUM_TRAIN_TIMESTEPS, jnp.float32)
    return full.at[:, :tokens_per_frame].set(0.0)


def _inputs(batch=_B, seed=0, steps=len(_SIGMAS) - 1):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    traj = jax.random.normal(keys[0], (steps + 1, batch, _C, _F, _H, _W), jnp.float32)
    z_i0 = jax.random.normal(keys[1], (batch, _C, 1, _H, _W), jnp.float32)
    base_context = jax.random.normal(keys[2], (_S, _D), jnp.float32)
    null_init = base_context[:_L] + 0.1 * jax.random.normal(keys[3], (_L, _D), jnp.float32)
    return traj, z_i0, base_context, null_init


def _splice(nulls, base_context):
    """Broadcast-and-splice, restated here so the literal reference imports no production helper.

    Codex R3 review, finding 3: sharing ``embed_null_tokens`` with the implementation would let a
    splice regression cancel out of the principal comparison below.
    """
    context = jnp.broadcast_to(base_context, (nulls.shape[0], *base_context.shape))
    return context.at[:, : nulls.shape[1]].set(nulls)


def _coupled_velocity(seed=11, context_scale=1.0):
    """v = A(context rows) * pattern + c_z * z + c_t * mean(timestep per example).

    Still affine in ∅ -- z̄_i and the timestep are constants across an inner loop -- so each inner
    problem stays a convex quadratic. But the velocity now genuinely reads the evaluation point and
    the per-token timestep, so feeding the model the wrong latent or a corrupted timestep changes the
    trajectory instead of cancelling out (Codex R3 review, finding 1: with a z-and-timestep-blind
    oracle, all-zero timesteps and zeroed latents both left the suite green).
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    weights = jax.random.normal(keys[0], (_L, _D), jnp.float32) * context_scale
    pattern = jax.random.normal(keys[1], (_C, _F, _H, _W), jnp.float32)

    def velocity_fn(z, timestep_2d, context):
        scale = jnp.sum(context[:, :_L] * weights, axis=(1, 2))
        per_example_t = jnp.mean(timestep_2d, axis=1)  # per example, so the batch stays independent
        return scale[:, None, None, None, None] * pattern + _C_Z * z + _C_T * per_example_t[:, None, None, None, None]

    return velocity_fn


def _reference_optimize(
    velocity_fn, traj, z_i0, sigmas, null_init, base_context, inner_iters, lr, guide_scale, eps=_ADAM[2]
):
    """The whole algorithm written out: literal loops, hand-rolled Adam, no optax, no scan.

    Independent of the implementation all the way down: its own timestep construction, its own pin,
    its own splice, its own Adam. ``eps`` is a parameter only so a test can show the fixture actually
    separates 1e-8 from a wrong epsilon.
    """
    b1, b2 = _ADAM[:2]
    batch = traj.shape[1]
    nulls = jnp.broadcast_to(jnp.asarray(null_init), (batch, _L, _D))
    z_bar = _pin(traj[0], z_i0)
    trajectory, all_nulls, all_losses, all_norms = [z_bar], [], [], []

    for i in range(len(sigmas) - 1):
        timestep_2d = _timestep_2d(float(sigmas[i]), batch=batch)
        dsigma = float(sigmas[i + 1]) - float(sigmas[i])
        target = traj[i + 1]
        v_cond = velocity_fn(z_bar, timestep_2d, jnp.broadcast_to(base_context, (batch, _S, _D)))

        def losses_of(n, z_bar=z_bar, timestep_2d=timestep_2d, v_cond=v_cond, dsigma=dsigma, target=target):
            v_unc = velocity_fn(z_bar, timestep_2d, _splice(n, base_context))
            z_hat = _pin(z_bar + dsigma * (v_unc + guide_scale * (v_cond - v_unc)), z_i0)
            per_example = jnp.mean((z_hat - target) ** 2, axis=(1, 2, 3, 4))
            return jnp.sum(per_example), per_example

        mu = jnp.zeros_like(nulls)  # fresh Adam state for every outer step
        nu = jnp.zeros_like(nulls)
        step_losses, step_norms = [], []
        for j in range(inner_iters):
            (_, per_example), grads = jax.value_and_grad(losses_of, has_aux=True)(nulls)
            step_losses.append(per_example)
            step_norms.append(jnp.sqrt(jnp.sum(grads**2, axis=(1, 2))))
            mu = b1 * mu + (1.0 - b1) * grads
            nu = b2 * nu + (1.0 - b2) * grads**2
            mu_hat = mu / (1.0 - b1 ** (j + 1))
            nu_hat = nu / (1.0 - b2 ** (j + 1))
            nulls = nulls - lr * mu_hat / (jnp.sqrt(nu_hat) + eps)

        all_nulls.append(nulls)  # ∅_i is locked AFTER the inner loop ...
        v_unc = velocity_fn(z_bar, timestep_2d, _splice(nulls, base_context))
        z_bar = _pin(z_bar + dsigma * (v_unc + guide_scale * (v_cond - v_unc)), z_i0)  # ... and used here
        trajectory.append(z_bar)
        all_losses.append(jnp.stack(step_losses))
        all_norms.append(jnp.stack(step_norms))

    return jnp.stack(all_nulls), jnp.stack(trajectory), jnp.stack(all_losses), jnp.stack(all_norms)


def _run(velocity_fn, traj, z_i0, base_context, null_init, *, inner_iters=3, lr=1e-2, guide_scale=5.0, sigmas=_SIGMAS):
    return optimize_null_embeddings(
        velocity_fn,
        traj,
        z_i0,
        jnp.asarray(sigmas, jnp.float32),
        null_init,
        base_context,
        inner_iters=inner_iters,
        lr=lr,
        guide_scale=guide_scale,
    )


def test_matches_the_hand_rolled_reference():
    """Pins objective, CFG mixing, both pins, locked-∅ advance, warm start, and Adam parity at once.

    Tolerance, not bitwise: the module runs the loop inside ``lax.scan`` with optax, the reference
    runs it eagerly with hand-written updates, so the two differ by FMA-level rounding (R2 saw the
    same). The gap is largest in ``z_bar``, where an FMA-sized difference in ∅ is amplified by the
    CFG factor (1 - w) = -4: measured worst case 5.9e-6 absolute / 7.6e-5 relative, against which the
    tolerance below keeps ~13x headroom. Every structural error in the mutation battery moves these
    outputs by O(1), i.e. four orders of magnitude more.
    """
    traj, z_i0, base_context, null_init = _inputs()
    velocity_fn = _coupled_velocity()

    nulls, z_bar, losses, grad_norms = _run(velocity_fn, traj, z_i0, base_context, null_init)
    expected = _reference_optimize(velocity_fn, traj, z_i0, _SIGMAS, null_init, base_context, 3, 1e-2, 5.0)

    for got, want, name in zip((nulls, z_bar, losses, grad_norms), expected, ("nulls", "z_bar", "losses", "norms")):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-3, atol=1e-5, err_msg=name)


def test_inner_loop_reduces_every_step_and_example_on_a_convex_problem():
    """Contract is final <= initial (plan F13): Adam need not descend at every single iterate."""
    traj, z_i0, base_context, null_init = _inputs(seed=2)

    _, _, losses, _ = _run(_coupled_velocity(), traj, z_i0, base_context, null_init, inner_iters=8, lr=1e-2)

    losses = np.asarray(losses)
    assert np.all(np.isfinite(losses))
    assert np.all(losses[:, -1, :] < losses[:, 0, :]), losses


def test_guide_scale_one_cancels_the_null_branch_exactly():
    """v_cfg = v_unc + 1*(v_cond - v_unc) = v_cond: ∅ leaves the graph, so its gradient is 0."""
    traj, z_i0, base_context, null_init = _inputs(seed=3)

    nulls, _, _, grad_norms = _run(_coupled_velocity(), traj, z_i0, base_context, null_init, guide_scale=1.0)

    np.testing.assert_array_equal(np.asarray(grad_norms), np.zeros_like(np.asarray(grad_norms)))
    warm_started = np.broadcast_to(np.asarray(null_init), (len(_SIGMAS) - 1, _B, _L, _D))
    np.testing.assert_array_equal(_f32_bits(nulls), _f32_bits(jnp.asarray(warm_started)))


def test_cfg_mixing_uses_the_deployment_weight():
    """With cond/unc branches returning distinct constants, the advance must be v_unc + w(v_cond-v_unc)."""
    traj, z_i0, base_context, null_init = _inputs(seed=4)
    guide_scale, c_cond, c_unc = 5.0, 0.75, -0.25

    def branch_velocity(z, timestep_2d, context):
        is_cond = jnp.all(context[:, :_L] == base_context[:_L])
        return jnp.where(is_cond, c_cond, c_unc) * jnp.ones_like(z)

    _, z_bar, _, _ = _run(branch_velocity, traj, z_i0, base_context, null_init, lr=0.0, guide_scale=guide_scale)

    v_cfg = c_unc + guide_scale * (c_cond - c_unc)
    expected = _pin(traj[0], z_i0)
    for i in range(len(_SIGMAS) - 1):
        expected = _pin(expected + (_SIGMAS[i + 1] - _SIGMAS[i]) * v_cfg, z_i0)
        np.testing.assert_allclose(np.asarray(z_bar[i + 1]), np.asarray(expected), rtol=1e-6, atol=1e-7)


def _counting_velocity(base_context, inner):
    """Counts real calls per context kind. Only meaningful under ``jax.disable_jit()``.

    Under ``disable_jit`` the scan bodies run eagerly, but the inner-loop calls still go through
    ``value_and_grad`` and therefore arrive as autodiff tracers, whose values cannot be read on the
    host. Those are exactly the unconditional calls, so they are classified by that fact; the two
    concrete calls per step (the cached conditional and the locked-∅ advance) are compared by value
    and their contexts recorded.
    """
    seen = {"cond": 0, "unc": 0, "concrete_unc_contexts": [], "calls": []}

    def velocity_fn(z, timestep_2d, context):
        if isinstance(context, jax.core.Tracer):
            kind = "inner"  # inside value_and_grad: only the null branch is differentiated
            seen["unc"] += 1
        elif np.array_equal(
            np.asarray(context[:, :_L]), np.asarray(jnp.broadcast_to(base_context[:_L], (_B, _L, _D)))
        ):
            kind = "cond"
            seen["cond"] += 1
        else:
            kind = "advance"
            seen["unc"] += 1
            seen["concrete_unc_contexts"].append(np.asarray(context[:, :_L]))
        # The timestep is closed over rather than differentiated, so it stays concrete even in the
        # inner calls -- which is what lets the timestep-content test below read every one of them.
        seen["calls"].append((kind, np.asarray(timestep_2d)))
        return inner(z, timestep_2d, context)

    return velocity_fn, seen


def test_every_forward_receives_the_step_s_own_per_token_timestep():
    """Frame-0 tokens carry 0, every later token carries sigma_i * 1000 -- checked at both steps,
    and for the inner calls as well as the locked-∅ advance (Codex R3 review, finding 1)."""
    traj, z_i0, base_context, null_init = _inputs(seed=13)
    velocity_fn, seen = _counting_velocity(base_context, _coupled_velocity())
    inner_iters = 2
    tokens_per_frame = (_H // 2) * (_W // 2)

    with jax.disable_jit():
        _run(velocity_fn, traj, z_i0, base_context, null_init, inner_iters=inner_iters)

    per_step = inner_iters + 2  # 1 conditional + J inner + 1 advance
    assert len(seen["calls"]) == per_step * (len(_SIGMAS) - 1)
    for i in range(len(_SIGMAS) - 1):
        kinds = [kind for kind, _ in seen["calls"][i * per_step : (i + 1) * per_step]]
        assert kinds == ["cond"] + ["inner"] * inner_iters + ["advance"], kinds
        for kind, timestep_2d in seen["calls"][i * per_step : (i + 1) * per_step]:
            assert timestep_2d.shape == (_B, _F * tokens_per_frame), kind
            np.testing.assert_array_equal(timestep_2d[:, :tokens_per_frame], 0.0)
            np.testing.assert_allclose(
                timestep_2d[:, tokens_per_frame:], _SIGMAS[i] * _NUM_TRAIN_TIMESTEPS, rtol=1e-6, atol=1e-4
            )


def test_conditional_velocity_is_computed_once_per_outer_step():
    """v_cond does not depend on ∅, so recomputing it inside the inner loop is invisible in the
    outputs and visible only here: it would turn J+1 forwards per step into 2J+1."""
    traj, z_i0, base_context, null_init = _inputs(seed=5)
    velocity_fn, seen = _counting_velocity(base_context, _coupled_velocity())
    inner_iters, steps = 4, len(_SIGMAS) - 1

    with jax.disable_jit():
        _run(velocity_fn, traj, z_i0, base_context, null_init, inner_iters=inner_iters)

    assert seen["cond"] == steps, seen
    assert seen["unc"] == steps * (inner_iters + 1), seen  # J inner iterations + 1 locked-∅ advance


def test_zero_learning_rate_freezes_the_nulls_and_the_advance_uses_them():
    """lr=0 turns the whole thing into a frozen-∅ replay -- and the advance must use that ∅."""
    traj, z_i0, base_context, null_init = _inputs(seed=6)
    velocity_fn, seen = _counting_velocity(base_context, _coupled_velocity())

    with jax.disable_jit():
        nulls, _, _, _ = _run(velocity_fn, traj, z_i0, base_context, null_init, lr=0.0)

    warm = np.broadcast_to(np.asarray(null_init), (_B, _L, _D))
    np.testing.assert_array_equal(_f32_bits(nulls), _f32_bits(jnp.stack([warm] * (len(_SIGMAS) - 1))))
    assert len(seen["concrete_unc_contexts"]) == len(_SIGMAS) - 1  # one advance per outer step
    for context in seen["concrete_unc_contexts"]:
        np.testing.assert_array_equal(context, warm)


def test_next_step_warm_starts_and_resets_the_optimizer_state():
    """Step i+1 continues from step i's locked ∅, with fresh Adam moments.

    Asserted by composition rather than by capture (the inner-loop contexts are autodiff tracers):
    re-running the tail of the problem as its own call -- same pivot, same warm-started ∅ -- must
    reproduce the full run's tail exactly. That holds only if the nulls carry over AND the optimizer
    state does not.
    """
    traj, z_i0, base_context, null_init = _inputs(seed=7)
    velocity_fn = _coupled_velocity()

    nulls, z_bar, losses, grad_norms = _run(velocity_fn, traj, z_i0, base_context, null_init, inner_iters=4)
    tail_traj = jnp.concatenate([z_bar[1][None], traj[2:]], axis=0)
    tail = _run(velocity_fn, tail_traj, z_i0, base_context, nulls[0], inner_iters=4, sigmas=_SIGMAS[1:])

    assert not np.allclose(np.asarray(nulls[0]), np.broadcast_to(np.asarray(null_init), (_B, _L, _D)))
    for got, want, name in zip(tail, (nulls[1:], z_bar[1:], losses[1:], grad_norms[1:]), ("n", "z", "l", "g")):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7, err_msg=name)


def test_output_shapes_dtypes_and_traces():
    traj, z_i0, base_context, null_init = _inputs(seed=8)
    inner_iters, steps = 3, len(_SIGMAS) - 1

    nulls, z_bar, losses, grad_norms = _run(_coupled_velocity(), traj, z_i0, base_context, null_init, inner_iters=3)

    assert nulls.shape == (steps, _B, _L, _D)
    assert z_bar.shape == (steps + 1, _B, _C, _F, _H, _W)
    assert losses.shape == grad_norms.shape == (steps, inner_iters, _B)
    for array in (nulls, z_bar, losses, grad_norms):
        assert array.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(array)))
    np.testing.assert_array_equal(_f32_bits(z_bar[0]), _f32_bits(_pin(traj[0], z_i0)))


def test_example_zero_is_independent_of_the_rest_of_the_batch():
    """The objective sums per-example losses, so example 0's ∅ must not feel example 1 at all."""
    traj, z_i0, base_context, null_init = _inputs(batch=_B, seed=9)
    velocity_fn = _coupled_velocity()
    batched_init = jnp.broadcast_to(null_init, (_B, _L, _D))

    batched = _run(velocity_fn, traj, z_i0, base_context, batched_init)
    alone = _run(velocity_fn, traj[:, :1], z_i0[:1], base_context, batched_init[:1])

    for got, want, name in zip(batched, alone, ("nulls", "z_bar", "losses", "grad_norms")):
        got = np.asarray(got)[:, :1] if name in ("nulls", "z_bar") else np.asarray(got)[:, :, :1]
        np.testing.assert_allclose(got, np.asarray(want), rtol=1e-6, atol=1e-8, err_msg=name)


def test_gradients_flow_through_a_real_tiny_wan_transformer():
    """Port of the reference smoke script's exit criterion: ∅ receives a non-zero gradient.

    Skipped where the transformer's own import chain (torch/chex/einops/aqt) is unavailable; the
    rest of this file stays runnable in a bare JAX environment.
    """
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    set_mesh = getattr(jax, "set_mesh", None)
    context_mesh = set_mesh(mesh) if set_mesh is not None else mesh
    channels, text_dim = 4, 32
    traj = jax.random.normal(jax.random.PRNGKey(0), (2, 1, channels, _F, _H, _W), jnp.float32)
    z_i0 = jax.random.normal(jax.random.PRNGKey(1), (1, channels, 1, _H, _W), jnp.float32)
    base_context = jax.random.normal(jax.random.PRNGKey(2), (_S, text_dim), jnp.float32)
    null_init = base_context[:_L]

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

        _, _, losses, grad_norms = optimize_null_embeddings(
            velocity_fn,
            traj,
            z_i0,
            jnp.asarray((1.0, 0.0), jnp.float32),
            null_init,
            base_context,
            inner_iters=2,
            lr=1e-2,
            guide_scale=5.0,
        )

    assert bool(jnp.all(jnp.isfinite(grad_norms))) and bool(jnp.all(jnp.isfinite(losses)))
    assert float(jnp.min(grad_norms)) > 0.0, grad_norms


def test_adam_epsilon_is_pinned_where_it_actually_matters():
    """Adam's update is ~ -lr*sign(g) unless |g| ~ eps, so eps is only pinned in the tiny-gradient
    regime. This fixture scales the context term down until the gradient is small enough that
    eps=1e-8 and eps=1e-4 give visibly different iterates, then requires the implementation to match
    the 1e-8 recipe exactly (Codex R3 review, finding 2: ADAM_EPS=1e-4 previously left all green).
    """
    traj, z_i0, base_context, null_init = _inputs(seed=14)
    velocity_fn = _coupled_velocity(context_scale=1e-5)  # drives the ∅ gradient into the eps regime

    nulls, _, _, grad_norms = _run(velocity_fn, traj, z_i0, base_context, null_init, inner_iters=4, lr=1e-3)
    with_correct_eps = _reference_optimize(
        velocity_fn, traj, z_i0, _SIGMAS, null_init, base_context, 4, 1e-3, 5.0, eps=1e-8
    )
    with_wrong_eps = _reference_optimize(
        velocity_fn, traj, z_i0, _SIGMAS, null_init, base_context, 4, 1e-3, 5.0, eps=1e-4
    )

    assert float(jnp.max(grad_norms)) < 1e-4, grad_norms  # the fixture really is in that regime
    np.testing.assert_allclose(np.asarray(nulls), np.asarray(with_correct_eps[0]), rtol=1e-5, atol=1e-9)
    # ... and the two epsilons are genuinely distinguishable here, so the assertion above has teeth.
    assert not np.allclose(np.asarray(with_correct_eps[0]), np.asarray(with_wrong_eps[0]), rtol=1e-2, atol=1e-9)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sigmas": (1.0, 0.6, 0.3, 0.0)}, "traj length"),  # grid longer than the trajectory
        ({"inner_iters": 0}, "inner_iters must be"),
        ({"inner_iters": 1.5}, "inner_iters must be an integer"),  # no silent truncation to 1
        ({"inner_iters": 2.0}, "inner_iters must be an integer"),
        ({"guide_scale": float("inf")}, "guide_scale must be finite"),
        ({"guide_scale": float("nan")}, "guide_scale must be finite"),
        ({"lr": -1e-3}, "lr must be finite and non-negative"),  # negative lr would ascend
        ({"lr": float("inf")}, "lr must be finite and non-negative"),
        ({"lr": float("nan")}, "lr must be finite and non-negative"),
    ],
)
def test_rejects_malformed_arguments(kwargs, message):
    traj, z_i0, base_context, null_init = _inputs(seed=10)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), traj, z_i0, base_context, null_init, **kwargs)


def test_rejects_more_null_tokens_than_context_rows():
    traj, z_i0, base_context, _ = _inputs(seed=11)
    too_many = jnp.zeros((_S + 1, _D), jnp.float32)

    with pytest.raises(ValueError, match="exceeds"):
        _run(_coupled_velocity(), traj, z_i0, base_context, too_many)


def test_rejects_a_velocity_with_the_wrong_shape():
    traj, z_i0, base_context, null_init = _inputs(seed=12)

    with pytest.raises(ValueError, match="velocity_fn returned shape"):
        _run(lambda z, t, c: jnp.zeros((), jnp.float32), traj, z_i0, base_context, null_init)
