"""exp_05 S2 — ``optimize_positive_embeddings``: per-step POSITIVE-context optimization (plan §3 step 2).

The branch swap of exp_04's null-text optimizer, ported from ``optimize_positive_embeddings``
(``third_party/Wan2.2/scripts/embedding_search.py:681-788``, submodule pin f370228): gradients flow
through ``v_cond``, so ``v_uncond`` is what gets cached once per outer step, and the optimized tensor
is the deployed **8-token** context passed *directly* as the whole ``encoder_hidden_states`` (S1's
convention) rather than spliced into a 512-row T5 context.

Five properties carry the round, each with a test that fails without it:

- **The asymmetry.** ``test_uncond_is_cached_and_cond_carries_the_eight_token_context`` counts real
  forwards under ``jax.disable_jit()``: exactly N unconditional calls at ``[B, S, D]`` and N·(J+1)
  conditional calls at ``[B, 8, D]``. Recomputing ``v_unc`` inside the inner loop changes no output --
  it is only observable here -- and a swapped slot is visible in the captured shapes.
- **The algorithm.** ``test_matches_the_hand_rolled_reference`` writes the whole loop out literally
  (explicit loops, hand-rolled Adam, its own truncation/pin/timestep, no production helper) and
  compares all four outputs, pinning objective, CFG mixing, both pins, the locked-C advance, the warm
  start, fresh-Adam-per-step and optax/torch Adam parity in one place.
- **CFG algebra -- the sign difference from the null slot.** At w=1 ``v_cfg = v_cond``, so the
  positive context stays in the graph and its gradient must be **non-zero** (exp_04's nulls are
  exactly zero there). That single assertion is what kills a C-in-the-uncond-slot implementation.
- **State recording, at the cache schema's cardinality.** ``z_bar_states`` is a first-class output the
  teacher-forced trainer consumes: **N** pre-step states aligned one-to-one with the N optimized
  contexts, with the terminal ``z_bar_N`` returned separately as ``z_final``. Checked against what the
  model actually saw -- every forward of step i receives exactly ``z_bar_states[i]``, bit for bit,
  post-pin -- and against a simulated S4/K2 consumer that zips states with contexts.
- **THE CAST SEAM.** One rule, stated in ``pos_context_inversion_wan``'s module docstring: the
  optimizer and the replay operator pass the fp32 context through unchanged, and the runner-built
  real-backbone ``velocity_fn`` casts to the activation dtype immediately before the transformer call.
  Only the optimizer's half is provable here -- pinned by reading the dtype and the bits of every
  captured context; the wiring half is S4's required closure test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bit_test_helpers import f32_bits as _f32_bits
from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L, optimize_positive_embeddings


# Small enough that the hand-rolled reference is cheap, real enough to keep every axis distinct.
_B, _C, _F, _H, _W = 2, 2, 2, 4, 6
# Context rows of the T5("") uncond branch, its width, and the deployed positive-token count. All
# hardcoded rather than imported (exp_04 R2's rule); ``_S != _POS_L`` is what makes the two branches
# distinguishable by shape alone.
_S, _D, _POS_L = 16, 4, 8
_SIGMAS = (1.0, 0.6, 0.0)  # N = 2 steps
_NUM_TRAIN_TIMESTEPS = 1000
_ADAM = (0.9, 0.999, 1e-8)  # torch.optim.Adam defaults, which optax must reproduce
# The oracle's sensitivity to the evaluation point and to the timestep tensor (exp_04 R3's values):
# small enough not to swamp the context term, large enough that a wrong z or timestep leaves every
# tolerance here.
_C_Z, _C_T = 0.05, 1e-4


def _pin(z, z_i0):
    return z.at[:, :, :1].set(z_i0[:, :, :1])


def _timestep_2d(sigma, batch=_B, f_lat=_F, h_lat=_H, w_lat=_W):
    """The per-token timestep, restated independently of the sampler helper (pinned in exp_04 R2)."""
    tokens_per_frame = (h_lat // 2) * (w_lat // 2)
    full = jnp.full((batch, f_lat * tokens_per_frame), sigma * _NUM_TRAIN_TIMESTEPS, jnp.float32)
    return full.at[:, :tokens_per_frame].set(0.0)


def _inputs(batch=_B, seed=0, steps=len(_SIGMAS) - 1):
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    traj = jax.random.normal(keys[0], (steps + 1, batch, _C, _F, _H, _W), jnp.float32)
    z_i0 = jax.random.normal(keys[1], (batch, _C, 1, _H, _W), jnp.float32)
    base_context = jax.random.normal(keys[2], (_S, _D), jnp.float32)
    return traj, z_i0, base_context


def _coupled_velocity(seed=11, context_scale=1.0):
    """v = <context, weights> * pattern + c_z * z + c_t * mean(timestep per example).

    Reads the *whole* context at either sequence length, so the conditional (8 rows) and the
    unconditional (16 rows) branches genuinely differ, and stays affine in C -- z̄_i and the timestep
    are constants across an inner loop -- so each inner problem is a convex quadratic. Coupling to z
    and to the timestep is exp_04 R3's finding 1: a z-and-timestep-blind oracle left zeroed latents
    and all-zero timesteps undetected.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    weights = jax.random.normal(keys[0], (_S, _D), jnp.float32) * context_scale
    pattern = jax.random.normal(keys[1], (_C, _F, _H, _W), jnp.float32)

    def velocity_fn(z, timestep_2d, context):
        scale = jnp.sum(context * weights[: context.shape[1]], axis=(1, 2))
        per_example_t = jnp.mean(timestep_2d, axis=1)  # per example, so the batch stays independent
        return scale[:, None, None, None, None] * pattern + _C_Z * z + _C_T * per_example_t[:, None, None, None, None]

    return velocity_fn


def _reference_optimize(
    velocity_fn,
    traj,
    z_i0,
    sigmas,
    base_context,
    inner_iters,
    lr,
    guide_scale,
    eps=_ADAM[2],
    pos_init=None,
    reduction="sum",
):
    """The whole algorithm written out: literal loops, hand-rolled Adam, no optax, no scan.

    Independent of the implementation all the way down -- its own truncation of T5("") to the warm
    start, its own timestep construction, its own pin, its own Adam. ``eps`` and ``reduction`` are
    parameters only so a test can show the fixture separates 1e-8 from a wrong epsilon, and the
    summed objective from a batch-averaged one.
    """
    b1, b2 = _ADAM[:2]
    batch = traj.shape[1]
    warm = base_context[:_POS_L] if pos_init is None else jnp.asarray(pos_init)
    pos = jnp.broadcast_to(warm, (batch, _POS_L, _D))
    uncond = jnp.broadcast_to(base_context, (batch, _S, _D))
    z_bar = _pin(traj[0], z_i0)
    trajectory, all_pos, all_losses, all_norms = [z_bar], [], [], []

    for i in range(len(sigmas) - 1):
        timestep_2d = _timestep_2d(float(sigmas[i]), batch=batch)
        dsigma = float(sigmas[i + 1]) - float(sigmas[i])
        target = traj[i + 1]
        v_unc = velocity_fn(z_bar, timestep_2d, uncond)  # cached: C-independent

        def losses_of(c, z_bar=z_bar, timestep_2d=timestep_2d, v_unc=v_unc, dsigma=dsigma, target=target):
            v_cond = velocity_fn(z_bar, timestep_2d, c)
            z_hat = _pin(z_bar + dsigma * (v_unc + guide_scale * (v_cond - v_unc)), z_i0)
            per_example = jnp.mean((z_hat - target) ** 2, axis=(1, 2, 3, 4))
            total = jnp.mean(per_example) if reduction == "mean" else jnp.sum(per_example)
            return total, per_example

        mu = jnp.zeros_like(pos)  # fresh Adam state for every outer step
        nu = jnp.zeros_like(pos)
        step_losses, step_norms = [], []
        for j in range(inner_iters):
            (_, per_example), grads = jax.value_and_grad(losses_of, has_aux=True)(pos)
            step_losses.append(per_example)
            step_norms.append(jnp.sqrt(jnp.sum(grads**2, axis=(1, 2))))
            mu = b1 * mu + (1.0 - b1) * grads
            nu = b2 * nu + (1.0 - b2) * grads**2
            mu_hat = mu / (1.0 - b1 ** (j + 1))
            nu_hat = nu / (1.0 - b2 ** (j + 1))
            pos = pos - lr * mu_hat / (jnp.sqrt(nu_hat) + eps)

        all_pos.append(pos)  # C_i is locked AFTER the inner loop ...
        v_cond = velocity_fn(z_bar, timestep_2d, pos)
        z_bar = _pin(z_bar + dsigma * (v_unc + guide_scale * (v_cond - v_unc)), z_i0)  # ... and used here
        trajectory.append(z_bar)
        all_losses.append(jnp.stack(step_losses))
        all_norms.append(jnp.stack(step_norms))

    return jnp.stack(all_pos), jnp.stack(trajectory), jnp.stack(all_losses), jnp.stack(all_norms)


def _run(velocity_fn, traj, z_i0, base_context, *, inner_iters=3, lr=1e-2, guide_scale=5.0, sigmas=_SIGMAS, **kwargs):
    return optimize_positive_embeddings(
        velocity_fn,
        traj,
        z_i0,
        jnp.asarray(sigmas, jnp.float32),
        base_context,
        inner_iters=inner_iters,
        lr=lr,
        guide_scale=guide_scale,
        **kwargs,
    )


def _full_trajectory(z_bar_states, z_final):
    """Recompose the N+1 trajectory the hand-rolled reference builds.

    The implementation returns the N schema-facing pre-step states and the terminal state under
    separate names (S2 review, finding 1), so the tests that reason about the whole trajectory say so
    explicitly here rather than indexing past the end of the cache.
    """
    return jnp.concatenate([jnp.asarray(z_bar_states), jnp.asarray(z_final)[None]], axis=0)


def _recording_velocity(inner):
    """Record every real forward. Only meaningful under ``jax.disable_jit()``.

    Contexts reaching the differentiated inner iterations are autodiff tracers, whose *values* cannot
    be read on the host -- but their shape and dtype can, which is exactly what the asymmetry and
    cast-seam assertions need. ``z`` and the timestep are closed over rather than differentiated, so
    they stay concrete in every call.
    """
    seen = []

    def velocity_fn(z, timestep_2d, context):
        seen.append(
            {
                "kind": "cond" if context.shape[1] == _POS_L else "uncond",
                "shape": tuple(context.shape),
                "dtype": jnp.asarray(context).dtype,
                "context": None if isinstance(context, jax.core.Tracer) else np.asarray(context),
                "z": np.asarray(z),
                "timestep": np.asarray(timestep_2d),
            }
        )
        return inner(z, timestep_2d, context)

    return velocity_fn, seen


# --------------------------------------------------------------------------------------------------
# 1. The asymmetry: cached v_unc, N*(J+1) conditional forwards, and the two branch shapes.
# --------------------------------------------------------------------------------------------------


def test_uncond_is_cached_and_cond_carries_the_eight_token_context():
    """v_unc is C-independent, so it is computed once per outer step -- invisible in the outputs and
    visible only in the call counts (recomputing it would make 2N+... forwards). The captured shapes
    pin which branch each context reaches: the deployed 8-token C conditions, T5("") does not."""
    traj, z_i0, base_context = _inputs(seed=5)
    velocity_fn, seen = _recording_velocity(_coupled_velocity())
    inner_iters, steps = 4, len(_SIGMAS) - 1

    with jax.disable_jit():
        _run(velocity_fn, traj, z_i0, base_context, inner_iters=inner_iters)

    kinds = [call["kind"] for call in seen]
    assert kinds.count("uncond") == steps, kinds
    assert kinds.count("cond") == steps * (inner_iters + 1), kinds  # J inner iterations + 1 advance
    per_step = inner_iters + 2
    for i in range(steps):  # ... and in this order: the cached uncond first, then the inner loop
        assert kinds[i * per_step : (i + 1) * per_step] == ["uncond"] + ["cond"] * (inner_iters + 1)
    for call in seen:
        assert call["shape"] == ((_B, _POS_L, _D) if call["kind"] == "cond" else (_B, _S, _D)), call["shape"]


def test_every_forward_receives_the_step_s_own_per_token_timestep():
    """Frame-0 tokens carry 0, every later token carries sigma_i * 1000 -- at both steps and in the
    inner calls as well as the advance (exp_04 R3 review, finding 1)."""
    traj, z_i0, base_context = _inputs(seed=13)
    velocity_fn, seen = _recording_velocity(_coupled_velocity())
    inner_iters = 2
    tokens_per_frame = (_H // 2) * (_W // 2)

    with jax.disable_jit():
        _run(velocity_fn, traj, z_i0, base_context, inner_iters=inner_iters)

    per_step = inner_iters + 2
    for i in range(len(_SIGMAS) - 1):
        for call in seen[i * per_step : (i + 1) * per_step]:
            assert call["timestep"].shape == (_B, _F * tokens_per_frame), call["kind"]
            np.testing.assert_array_equal(call["timestep"][:, :tokens_per_frame], 0.0)
            np.testing.assert_allclose(
                call["timestep"][:, tokens_per_frame:], _SIGMAS[i] * _NUM_TRAIN_TIMESTEPS, rtol=1e-6, atol=1e-4
            )


# --------------------------------------------------------------------------------------------------
# 2. The algorithm, against a literal hand-rolled reference.
# --------------------------------------------------------------------------------------------------


def test_matches_the_hand_rolled_reference():
    """Pins objective, CFG mixing, both pins, the locked-C advance, the warm start and Adam parity.

    Tolerance, not bitwise: the module runs the loop inside ``lax.scan`` with optax, the reference
    runs it eagerly with hand-written updates, so the two differ by FMA-level rounding (exp_04 R2/R3
    saw the same). Measured worst cases: 1.0e-5 relative on ``z_bar`` (8.4e-5 absolute, where an
    FMA-sized difference in C is amplified by the CFG factor w = 5) and 2.1e-6 relative on ``losses``
    (1.4e-3 absolute on a scale of 7.3e2) -- so ``rtol`` keeps ~100x headroom. Every structural error
    in the mutation battery moves these outputs by O(1), five orders of magnitude more.
    """
    traj, z_i0, base_context = _inputs()
    velocity_fn = _coupled_velocity()

    pos_embeds, z_bar_states, z_final, losses, grad_norms = _run(velocity_fn, traj, z_i0, base_context)
    expected = _reference_optimize(velocity_fn, traj, z_i0, _SIGMAS, base_context, 3, 1e-2, 5.0)
    z_bar = _full_trajectory(z_bar_states, z_final)  # the reference builds one N+1 trajectory

    for got, want, name in zip((pos_embeds, z_bar, losses, grad_norms), expected, ("pos", "z_bar", "losses", "norms")):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-3, atol=1e-5, err_msg=name)


def test_inner_loop_reduces_every_step_and_example_on_a_convex_problem():
    """Contract is final <= initial (exp_04 plan F13): Adam need not descend at every iterate."""
    traj, z_i0, base_context = _inputs(seed=2)

    _, _, _, losses, _ = _run(_coupled_velocity(), traj, z_i0, base_context, inner_iters=8)

    losses = np.asarray(losses)
    assert np.all(np.isfinite(losses))
    assert np.all(losses[:, -1, :] < losses[:, 0, :]), losses


# --------------------------------------------------------------------------------------------------
# 3. CFG algebra — the sign difference from the null slot.
# --------------------------------------------------------------------------------------------------


def test_guide_scale_one_keeps_the_positive_branch_in_the_graph():
    """THE difference from exp_04. At w=1, v_cfg = v_unc + 1*(v_cond - v_unc) = v_cond, so the *null*
    branch cancels (exp_04's gradient is exactly zero there) while the positive branch is the whole
    velocity -- its gradient must be non-zero and C must actually move. An implementation that put C
    in the unconditional slot would produce exactly zero here."""
    traj, z_i0, base_context = _inputs(seed=3)

    pos_embeds, _, _, _, grad_norms = _run(_coupled_velocity(), traj, z_i0, base_context, guide_scale=1.0)

    assert float(jnp.min(grad_norms)) > 0.0, grad_norms
    warm = np.broadcast_to(np.asarray(base_context[:_POS_L]), (len(_SIGMAS) - 1, _B, _POS_L, _D))
    assert not np.any(np.asarray(pos_embeds) == warm), pos_embeds


def test_cfg_mixing_uses_the_deployment_weight():
    """With the two branches returning distinct constants, the advance must be v_unc + w(v_cond-v_unc)."""
    traj, z_i0, base_context = _inputs(seed=4)
    guide_scale, c_cond, c_unc = 5.0, 0.75, -0.25

    def branch_velocity(z, timestep_2d, context):
        return (c_cond if context.shape[1] == _POS_L else c_unc) * jnp.ones_like(z)

    _, z_bar_states, z_final, _, _ = _run(branch_velocity, traj, z_i0, base_context, lr=0.0, guide_scale=guide_scale)

    z_bar = _full_trajectory(z_bar_states, z_final)
    v_cfg = c_unc + guide_scale * (c_cond - c_unc)
    expected = _pin(traj[0], z_i0)
    for i in range(len(_SIGMAS) - 1):
        expected = _pin(expected + (_SIGMAS[i + 1] - _SIGMAS[i]) * v_cfg, z_i0)
        np.testing.assert_allclose(np.asarray(z_bar[i + 1]), np.asarray(expected), rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------------------------------
# 4. Locked-C advance, warm start, and the recorded states.
# --------------------------------------------------------------------------------------------------


def test_next_step_warm_starts_from_the_locked_context_with_fresh_adam_state():
    """Step i+1 continues from step i's locked C, with fresh Adam moments.

    Asserted by composition (the inner-loop contexts are autodiff tracers): re-running the tail of the
    problem as its own call -- same pivot, same warm start -- must reproduce the full run's tail
    exactly. That holds only if C carries over AND the optimizer state does not.
    """
    traj, z_i0, base_context = _inputs(seed=7)
    velocity_fn = _coupled_velocity()

    pos_embeds, states, z_final, losses, grad_norms = _run(velocity_fn, traj, z_i0, base_context, inner_iters=4)
    tail_traj = jnp.concatenate([states[1][None], traj[2:]], axis=0)
    tail = _run(velocity_fn, tail_traj, z_i0, base_context, inner_iters=4, sigmas=_SIGMAS[1:], pos_init=pos_embeds[0])

    warm = np.broadcast_to(np.asarray(base_context[:_POS_L]), (_B, _POS_L, _D))
    assert not np.allclose(np.asarray(pos_embeds[0]), warm)  # the warm start really did move
    wanted = (pos_embeds[1:], states[1:], z_final, losses[1:], grad_norms[1:])
    for got, want, name in zip(tail, wanted, ("p", "states", "final", "l", "g")):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-7, err_msg=name)


def test_recorded_states_are_exactly_what_the_forwards_received():
    """``z_bar_states`` is a first-class output (the teacher-forced trainer consumes it), so it is
    pinned against what the model actually saw: every forward of step i gets ``z_bar_states[i]`` bit
    for bit, post-pin. A state recorded pre-pin, or shifted by one step, fails here."""
    traj, z_i0, base_context = _inputs(seed=15)
    velocity_fn, seen = _recording_velocity(_coupled_velocity())
    inner_iters, steps = 2, len(_SIGMAS) - 1

    with jax.disable_jit():
        _, states, z_final, _, _ = _run(velocity_fn, traj, z_i0, base_context, inner_iters=inner_iters)

    assert states.dtype == z_final.dtype == jnp.float32
    assert states.shape[0] == steps  # the N pre-step states, NOT the N+1 trajectory
    np.testing.assert_array_equal(_f32_bits(states[0]), _f32_bits(_pin(traj[0], z_i0)))
    per_step = inner_iters + 2
    assert len(seen) == per_step * steps
    for i in range(steps):
        for call in seen[i * per_step : (i + 1) * per_step]:
            np.testing.assert_array_equal(
                _f32_bits(call["z"]), _f32_bits(states[i]), err_msg=f"step {i} {call['kind']}"
            )
    for state in _full_trajectory(states, z_final):  # every recorded state is pinned, terminal included
        np.testing.assert_array_equal(_f32_bits(state[:, :, :1]), _f32_bits(z_i0))


def test_states_and_contexts_zip_one_to_one_for_the_teacher_forced_cache():
    """The K2 cache schema (plan §3/P2'), simulated from the S4 consumer's side.

    The trainer pairs ``(z_bar_states[i], pos_embeds[i])`` positionally and never sees a trajectory,
    so the two must have the same length N *and* the pairing must be the real one: replaying the CFG
    Euler step from state i under context i has to land on state i+1 -- and, for the last pair, on
    ``z_final``. Returning the N+1 trajectory in the states slot (the ``[:-1]``-slice-forgotten defect
    this schema exists to prevent) misaligns every pair from the first step on.
    """
    traj, z_i0, base_context = _inputs(seed=20)
    velocity_fn, guide_scale = _coupled_velocity(), 5.0
    steps = len(_SIGMAS) - 1

    pos_embeds, states, z_final, _, _ = _run(velocity_fn, traj, z_i0, base_context, guide_scale=guide_scale)

    assert states.shape[0] == pos_embeds.shape[0] == steps
    assert z_final.shape == states.shape[1:]
    uncond = jnp.broadcast_to(base_context, (_B, _S, _D))
    for i in range(steps):
        timestep_2d = _timestep_2d(_SIGMAS[i])
        v_unc = velocity_fn(states[i], timestep_2d, uncond)
        v_cond = velocity_fn(states[i], timestep_2d, pos_embeds[i])
        advanced = _pin(states[i] + (_SIGMAS[i + 1] - _SIGMAS[i]) * (v_unc + guide_scale * (v_cond - v_unc)), z_i0)
        want = states[i + 1] if i + 1 < steps else z_final
        np.testing.assert_allclose(np.asarray(advanced), np.asarray(want), rtol=1e-5, atol=1e-6, err_msg=f"pair {i}")


def test_output_shapes_dtypes_and_traces():
    traj, z_i0, base_context = _inputs(seed=8)
    inner_iters, steps = 3, len(_SIGMAS) - 1

    outputs = _run(_coupled_velocity(), traj, z_i0, base_context, inner_iters=inner_iters)
    pos_embeds, states, z_final, losses, grad_norms = outputs

    assert POS_L == _POS_L  # the deployed ``pre_context_tokens``; the target's row count
    assert pos_embeds.shape == (steps, _B, _POS_L, _D)
    assert states.shape == (steps, _B, _C, _F, _H, _W)  # N states, one per optimized context
    assert z_final.shape == (_B, _C, _F, _H, _W)  # the terminal state, named rather than sliced
    assert losses.shape == grad_norms.shape == (steps, inner_iters, _B)
    for array in outputs:
        assert array.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(array)))


# --------------------------------------------------------------------------------------------------
# 5. THE CAST SEAM (S1's discovered contract) and the batching contract.
# --------------------------------------------------------------------------------------------------


def test_the_optimizer_hands_velocity_fn_an_uncast_fp32_context():
    """THE CAST-SEAM DECISION, pinned: ``velocity_fn`` owns the activation-dtype cast, so the
    optimizer passes C through in fp32, unmodified -- exactly the tensor it is optimizing and exactly
    the tensor it returns. An optimizer that cast C to bf16 (or to anything else) before the seam
    would both change these dtypes and break the bitwise equality with ``pos_embeds``.

    The uncond slot is pinned the same way: it receives T5("") broadcast over the batch, untouched.
    """
    traj, z_i0, base_context = _inputs(seed=16)
    velocity_fn, seen = _recording_velocity(_coupled_velocity())
    inner_iters, steps = 2, len(_SIGMAS) - 1

    with jax.disable_jit():
        pos_embeds, _, _, _, _ = _run(velocity_fn, traj, z_i0, base_context, inner_iters=inner_iters)

    assert pos_embeds.dtype == jnp.float32
    for call in seen:  # tracers carry a dtype even where their value cannot be read
        assert call["dtype"] == jnp.float32, call
    per_step = inner_iters + 2
    for i in range(steps):
        step_calls = seen[i * per_step : (i + 1) * per_step]
        np.testing.assert_array_equal(  # the cached uncond forward: T5(""), broadcast, untouched
            _f32_bits(step_calls[0]["context"]), _f32_bits(jnp.broadcast_to(base_context, (_B, _S, _D)))
        )
        np.testing.assert_array_equal(  # the advance: the locked C, bit for bit what is returned
            _f32_bits(step_calls[-1]["context"]), _f32_bits(pos_embeds[i]), err_msg=f"step {i}"
        )


def test_example_zero_is_independent_of_the_rest_of_the_batch():
    """The objective sums per-example losses, so example 0's C must not feel example 1 at all --
    bitwise, which is what separates the sum from a batch-averaged objective (whose per-example
    gradients differ by the 1/B factor Adam's epsilon does not fully cancel). Per-example distinct
    warm starts, per S1's lesson that a shared context hides collapse and permutation defects."""
    traj, z_i0, base_context = _inputs(seed=9)
    velocity_fn = _coupled_velocity()
    pos_init = base_context[:_POS_L] + jnp.stack(
        [0.1 * jax.random.normal(jax.random.PRNGKey(17 + b), (_POS_L, _D), jnp.float32) for b in range(_B)]
    )

    batched = _run(velocity_fn, traj, z_i0, base_context, pos_init=pos_init)
    alone = _run(velocity_fn, traj[:, :1], z_i0[:1], base_context, pos_init=pos_init[:1])

    assert not np.allclose(np.asarray(pos_init[0]), np.asarray(pos_init[1]))  # the fixture's point
    # Each output carries the batch on a different axis: 1 for the [N, B, ...] stacks, 0 for the
    # terminal state, 2 for the [N, J, B] traces.
    batch_axis = {"pos": 1, "states": 1, "final": 0, "losses": 2, "grad_norms": 2}
    for got, want, name in zip(batched, alone, ("pos", "states", "final", "losses", "grad_norms")):
        got = np.take(np.asarray(got), [0], axis=batch_axis[name])
        np.testing.assert_array_equal(_f32_bits(got), _f32_bits(jnp.asarray(want)), err_msg=name)


def test_the_objective_is_the_batch_sum_not_the_batch_mean():
    """Adam is scale-invariant except through its epsilon, so sum-vs-mean is only distinguishable in
    the small-gradient regime -- where it is a real difference in the iterates, and where the batching
    contract (each example's gradient is the gradient of its own loss) actually bites."""
    traj, z_i0, base_context = _inputs(seed=18)
    velocity_fn = _coupled_velocity(context_scale=1e-5)  # drives the C gradient into the eps regime

    pos_embeds, _, _, _, _ = _run(velocity_fn, traj, z_i0, base_context, inner_iters=4, lr=1e-3)
    as_sum = _reference_optimize(velocity_fn, traj, z_i0, _SIGMAS, base_context, 4, 1e-3, 5.0, reduction="sum")
    as_mean = _reference_optimize(velocity_fn, traj, z_i0, _SIGMAS, base_context, 4, 1e-3, 5.0, reduction="mean")

    np.testing.assert_allclose(np.asarray(pos_embeds), np.asarray(as_sum[0]), rtol=1e-5, atol=1e-9)
    assert not np.allclose(np.asarray(as_sum[0]), np.asarray(as_mean[0]), rtol=1e-3, atol=1e-9)


def test_adam_epsilon_is_pinned_where_it_actually_matters():
    """Adam's update is ~ -lr*sign(g) unless |g| ~ eps, so eps is only pinned in the tiny-gradient
    regime. This fixture scales the context term down until eps=1e-8 and eps=1e-4 give visibly
    different iterates, then requires the implementation to match the 1e-8 recipe (exp_04 R3 review,
    finding 2: ADAM_EPS=1e-4 previously left all green). eps_root=0.0 is part of the same recipe."""
    traj, z_i0, base_context = _inputs(seed=14)
    velocity_fn = _coupled_velocity(context_scale=1e-5)

    pos_embeds, _, _, _, grad_norms = _run(velocity_fn, traj, z_i0, base_context, inner_iters=4, lr=1e-3)
    with_correct_eps = _reference_optimize(velocity_fn, traj, z_i0, _SIGMAS, base_context, 4, 1e-3, 5.0, eps=1e-8)
    with_wrong_eps = _reference_optimize(velocity_fn, traj, z_i0, _SIGMAS, base_context, 4, 1e-3, 5.0, eps=1e-4)

    assert float(jnp.max(grad_norms)) < 1e-4, grad_norms  # the fixture really is in that regime
    np.testing.assert_allclose(np.asarray(pos_embeds), np.asarray(with_correct_eps[0]), rtol=1e-5, atol=1e-9)
    assert not np.allclose(np.asarray(with_correct_eps[0]), np.asarray(with_wrong_eps[0]), rtol=1e-2, atol=1e-9)


# --------------------------------------------------------------------------------------------------
# 6. Fail-closed argument validation, and the real backbone.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sigmas": (1.0, 0.6, 0.3, 0.0)}, "traj length"),  # grid longer than the trajectory
        ({"sigmas": (0.6, 1.0, 0.0)}, "strictly descending"),
        ({"sigmas": (1.0, 0.6, 0.1)}, "must end at 0.0"),
        ({"inner_iters": 0}, "inner_iters must be"),
        ({"inner_iters": 1.5}, "inner_iters must be an integer"),  # no silent truncation to 1
        ({"inner_iters": 2.0}, "inner_iters must be an integer"),
        ({"inner_iters": True}, "inner_iters must be an integer"),  # bool is an int subclass
        ({"guide_scale": float("inf")}, "guide_scale must be finite"),
        ({"guide_scale": float("nan")}, "guide_scale must be finite"),
        ({"lr": -1e-3}, "lr must be finite and non-negative"),  # a negative lr ascends the objective
        ({"lr": float("inf")}, "lr must be finite and non-negative"),
        ({"lr": float("nan")}, "lr must be finite and non-negative"),
    ],
)
def test_rejects_malformed_arguments(kwargs, message):
    traj, z_i0, base_context = _inputs(seed=10)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), traj, z_i0, base_context, **kwargs)


@pytest.mark.parametrize(
    "mangle, message",
    [
        (lambda t, z, c: (t[0], z, c), "traj must be"),  # rank-5: a single pivot, not a trajectory
        (lambda t, z, c: (t, z[:, :, :, :1], c), "z_i0 shape"),  # spatial mismatch
        (lambda t, z, c: (t, z[:1], c), "z_i0 shape"),  # batch mismatch
        (lambda t, z, c: (t, jnp.broadcast_to(z, (_B, _C, 3, _H, _W)), c), "z_i0 must carry"),
        (lambda t, z, c: (t, z, c[None].repeat(3, axis=0)), "unit leading axis"),  # a batch of T5s
        (lambda t, z, c: (t, z, c[0]), "base_context must be"),  # rank-1
        (lambda t, z, c: (t, z, c[None, None]), "base_context must be"),  # rank-4
    ],
)
def test_rejects_inconsistent_geometry(mangle, message):
    traj, z_i0, base_context = _inputs(seed=11)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), *mangle(traj, z_i0, base_context))


@pytest.mark.parametrize(
    "pos_init, message",
    [
        (jnp.zeros((_POS_L, _D + 1), jnp.float32), "pos_init"),  # a different context width
        (jnp.zeros((_B + 1, _POS_L, _D), jnp.float32), "pos_init"),  # a different batch
        (jnp.zeros((_POS_L,), jnp.float32), "pos_init"),  # rank-1
    ],
)
def test_rejects_an_inconsistent_warm_start(pos_init, message):
    traj, z_i0, base_context = _inputs(seed=12)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), traj, z_i0, base_context, pos_init=pos_init)


def test_rejects_a_velocity_with_the_wrong_shape():
    traj, z_i0, base_context = _inputs(seed=12)

    with pytest.raises(ValueError, match="velocity_fn returned shape"):
        _run(lambda z, t, c: jnp.zeros((), jnp.float32), traj, z_i0, base_context)


def test_the_warm_start_is_the_truncated_t5_context_and_l_pos_is_settable():
    """C_0 = pos_context_from_t5(T5("")) -- the plan §3 warm start -- with plan §4's L_pos ablation
    reachable by handing the same constructor's output in as ``pos_init``."""
    from maxdiffusion.models.wan.pos_context_inversion_wan import pos_context_from_t5

    traj, z_i0, base_context = _inputs(seed=19)
    velocity_fn = _coupled_velocity()

    default = _run(velocity_fn, traj, z_i0, base_context, lr=0.0)
    explicit = _run(velocity_fn, traj, z_i0, base_context, lr=0.0, pos_init=pos_context_from_t5(base_context))
    ablated = _run(velocity_fn, traj, z_i0, base_context, lr=0.0, pos_init=pos_context_from_t5(base_context, l_pos=1))

    np.testing.assert_array_equal(_f32_bits(default[0]), _f32_bits(jnp.asarray(explicit[0])))
    warm = np.broadcast_to(np.asarray(base_context[:_POS_L]), (len(_SIGMAS) - 1, _B, _POS_L, _D))
    np.testing.assert_array_equal(_f32_bits(default[0]), _f32_bits(jnp.asarray(warm)))  # lr=0 freezes C
    assert ablated[0].shape == (len(_SIGMAS) - 1, _B, 1, _D)


@pytest.mark.parametrize("guide_scale", [5.0, 1.0])
def test_gradients_flow_through_a_real_tiny_wan_transformer(guide_scale):
    """Port of the reference smoke script's exit criterion, at both guidance weights: the positive
    context receives a non-zero gradient **including at w=1**, where exp_04's null gradient is exactly
    zero. That is the positive slot's distinguishing property, on the real backbone.

    Skipped where the transformer's own import chain (torch/chex/einops/aqt) is unavailable; the rest
    of this file stays runnable in a bare JAX environment.
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

        pos_embeds, states, z_final, losses, grad_norms = optimize_positive_embeddings(
            velocity_fn,
            traj,
            z_i0,
            jnp.asarray((1.0, 0.0), jnp.float32),
            base_context,
            inner_iters=2,
            lr=1e-2,
            guide_scale=guide_scale,
        )

    assert pos_embeds.shape == (1, 1, _POS_L, text_dim)  # N = 1 step: one context, one state
    assert states.shape == (1, 1, channels, _F, _H, _W) and z_final.shape == (1, channels, _F, _H, _W)
    assert bool(jnp.all(jnp.isfinite(grad_norms))) and bool(jnp.all(jnp.isfinite(losses)))
    assert float(jnp.min(grad_norms)) > 0.0, grad_norms
