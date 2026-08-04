"""exp_04 R4a — ``replay_with_nulls``: the deployment CFG sampler (plan §3 step 3).

This is the operator every downstream claim is measured through: A0/A1/A2 reconstruction, the cached
artifact's ``expected_final_latent``, and the eventual adapter-vs-baseline comparison all reduce to
"replay from ``z_start`` with these nulls and see where it lands". Ported from
``regenerate_with_null_embeds`` (``third_party/Wan2.2/scripts/embedding_search.py:791-819``,
submodule pin f370228).

One asymmetry against R3 is deliberate and is tested here: the optimizer caches ``v_cond`` once per
outer step because it holds ``z_bar_i`` fixed across the inner loop, whereas replay **moves** ``z``
at every step, so ``v_cond`` must be recomputed each step. Caching it here would be a correctness
bug, not a perf win -- hence the call-structure test (N conditional + N unconditional forwards).

The A0 identity is the other load-bearing property: if the nulls are the base context's own leading
rows, then ``v_unc == v_cond`` and the CFG combine ``v_unc + w(v_cond - v_unc)`` collapses to
``v_cond`` for *any* w. That is what makes the frozen-∅ control A0 well defined without a separate
code path, and it is asserted across three guidance weights (to a tolerance whose provenance is
documented in that test, together with the contrast that keeps it honest).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion.models.wan.null_inversion_wan import replay_with_nulls


_B, _C, _F, _H, _W = 2, 2, 2, 4, 6
_S, _D, _L = 8, 4, 3
_SIGMAS = (1.0, 0.6, 0.3, 0.0)  # N = 3 steps
_NUM_TRAIN_TIMESTEPS = 1000
_C_Z, _C_T = 0.05, 1e-4  # the oracle's coupling to the evaluation point and to the timestep
_EDGE_VALUES = (0.0, -0.0, np.float32(np.inf), np.float32(np.nan), np.float32(1e-45), -1.0)


def _f32_bits(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x).astype(jnp.float32), jnp.uint32))


def _pin(z, z_i0):
    return z.at[:, :, :1].set(z_i0[:, :, :1])


def _timestep_2d(sigma, batch=_B):
    tokens_per_frame = (_H // 2) * (_W // 2)
    full = jnp.full((batch, _F * tokens_per_frame), sigma * _NUM_TRAIN_TIMESTEPS, jnp.float32)
    return full.at[:, :tokens_per_frame].set(0.0)


def _splice(nulls, base_context):
    """Broadcast-and-splice, restated locally so the reference below imports no production helper."""
    if nulls.ndim == 2:
        nulls = jnp.broadcast_to(nulls, (_B, *nulls.shape))
    context = jnp.broadcast_to(base_context, (nulls.shape[0], *base_context.shape))
    return context.at[:, : nulls.shape[1]].set(nulls)


def _inputs(batch=_B, seed=0, steps=len(_SIGMAS) - 1, dtype=jnp.float32):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    z_start = jax.random.normal(keys[0], (batch, _C, _F, _H, _W), jnp.float32).astype(dtype)
    z_i0 = jax.random.normal(keys[1], (batch, _C, 1, _H, _W), jnp.float32).astype(dtype)
    base_context = jax.random.normal(keys[2], (_S, _D), jnp.float32)
    nulls = jax.random.normal(keys[3], (steps, batch, _L, _D), jnp.float32)
    return z_start, z_i0, base_context, nulls


def _coupled_velocity(seed=21):
    """v = A(context rows)*pattern + c_z*z + c_t*mean(timestep): reads context, latent and timestep.

    Blind oracles were the R3 review's MAJOR finding; the same discipline applies here, otherwise a
    wrong evaluation point or a corrupted timestep would be invisible to the literal-loop test.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    weights = jax.random.normal(keys[0], (_L, _D), jnp.float32)
    pattern = jax.random.normal(keys[1], (_C, _F, _H, _W), jnp.float32)

    def velocity_fn(z, timestep_2d, context):
        scale = jnp.sum(context[:, :_L] * weights, axis=(1, 2))
        per_example_t = jnp.mean(timestep_2d, axis=1)
        return scale[:, None, None, None, None] * pattern + _C_Z * z + _C_T * per_example_t[:, None, None, None, None]

    return velocity_fn


def _reference_replay(velocity_fn, z_start, z_i0, sigmas, nulls, base_context, guide_scale):
    """The sampler written out: literal loop, own timestep, own pin, own splice."""
    z = _pin(z_start, z_i0)
    trajectory = [z]
    for i in range(len(sigmas) - 1):
        timestep_2d = _timestep_2d(float(sigmas[i]), batch=z.shape[0])
        cond_context = jnp.broadcast_to(base_context, (z.shape[0], *base_context.shape))
        v_cond = velocity_fn(z, timestep_2d, cond_context)  # recomputed every step: z has moved
        v_unc = velocity_fn(z, timestep_2d, _splice(nulls[i], base_context))
        dsigma = float(sigmas[i + 1]) - float(sigmas[i])
        z = _pin(z + dsigma * (v_unc + guide_scale * (v_cond - v_unc)), z_i0)
        trajectory.append(z)
    return z, jnp.stack(trajectory)


def _run(velocity_fn, z_start, z_i0, base_context, nulls, *, guide_scale=5.0, sigmas=_SIGMAS, **kwargs):
    return replay_with_nulls(
        velocity_fn,
        z_start,
        z_i0,
        jnp.asarray(sigmas, jnp.float32),
        nulls,
        base_context,
        guide_scale=guide_scale,
        **kwargs,
    )


def test_scan_matches_a_literal_python_loop():
    z_start, z_i0, base_context, nulls = _inputs()
    velocity_fn = _coupled_velocity()

    z_final, traj = _run(velocity_fn, z_start, z_i0, base_context, nulls, return_trajectory=True)
    expected_final, expected_traj = _reference_replay(velocity_fn, z_start, z_i0, _SIGMAS, nulls, base_context, 5.0)

    np.testing.assert_allclose(np.asarray(z_final), np.asarray(expected_final), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(traj), np.asarray(expected_traj), rtol=1e-5, atol=1e-7)


def test_cfg_combine_is_analytic_at_the_deployment_weight():
    """With distinct constants per branch, each step's increment must be dsigma*(v_unc + w*(v_cond-v_unc))."""
    z_start, z_i0, base_context, nulls = _inputs(seed=1)
    guide_scale, c_cond, c_unc = 5.0, 0.75, -0.25

    def branch_velocity(z, timestep_2d, context):
        is_cond = jnp.all(context[:, :_L] == base_context[:_L])
        return jnp.where(is_cond, c_cond, c_unc) * jnp.ones_like(z)

    _, traj = _run(
        branch_velocity, z_start, z_i0, base_context, nulls, guide_scale=guide_scale, return_trajectory=True
    )

    # Tolerance covers fp32 accumulation over the grid (worst observed 1.2e-7 absolute on an entry
    # near zero); swapping the two branches would instead move v_cfg from 4.75 to -4.25.
    v_cfg = c_unc + guide_scale * (c_cond - c_unc)
    expected = _pin(z_start, z_i0)
    for i in range(len(_SIGMAS) - 1):
        expected = _pin(expected + (_SIGMAS[i + 1] - _SIGMAS[i]) * v_cfg, z_i0)
        np.testing.assert_allclose(np.asarray(traj[i + 1]), np.asarray(expected), rtol=1e-5, atol=1e-6)


def test_base_row_nulls_make_guidance_inert():
    """A0 identity: nulls == the context's own leading rows ⇒ v_unc == v_cond ⇒ w cannot matter.

    Not asserted bitwise. The two branches receive bit-identical contexts, but XLA is free to
    schedule the conditional reduction (whose input is loop-invariant) differently from the
    unconditional one, so ``v_cond - v_unc`` can come out one ULP off instead of exactly zero -- and
    the CFG combine multiplies that by w. The measured deviation is exactly linear in w, which is
    the signature of that mechanism rather than of leaking guidance: max relative 6.3e-6 at w=5,
    1.9e-5 at w=13, 7.7e-5 at w=50. Real guidance, by contrast, separates these runs by 59.9
    absolute -- the contrast assertion at the end keeps that gap in the test.
    """
    z_start, z_i0, base_context, nulls = _inputs(seed=2)
    steps = len(_SIGMAS) - 1
    frozen = jnp.broadcast_to(base_context[:_L], (steps, _B, _L, _D))
    velocity_fn = _coupled_velocity()

    at_w1 = _run(velocity_fn, z_start, z_i0, base_context, frozen, guide_scale=1.0)
    at_w5 = _run(velocity_fn, z_start, z_i0, base_context, frozen, guide_scale=5.0)
    at_w13 = _run(velocity_fn, z_start, z_i0, base_context, frozen, guide_scale=13.0)

    np.testing.assert_allclose(np.asarray(at_w1), np.asarray(at_w5), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(at_w1), np.asarray(at_w13), rtol=1e-4, atol=1e-6)
    free_w1 = _run(velocity_fn, z_start, z_i0, base_context, nulls, guide_scale=1.0)
    free_w13 = _run(velocity_fn, z_start, z_i0, base_context, nulls, guide_scale=13.0)
    assert float(jnp.max(jnp.abs(free_w1 - free_w13))) > 1.0


def test_frame_zero_holds_the_anchor_bitwise_at_every_step():
    z_start, z_i0, base_context, nulls = _inputs(seed=3)
    edges = np.asarray(_EDGE_VALUES, dtype=np.float32)
    anchor = np.asarray(z_i0).copy()
    anchor[:, 0, 0, 0, : edges.size] = edges
    z_i0 = jnp.asarray(anchor)

    _, traj = _run(_coupled_velocity(), z_start, z_i0, base_context, nulls, return_trajectory=True)

    for i in range(len(_SIGMAS)):
        np.testing.assert_array_equal(_f32_bits(traj[i, :, :, :1]), _f32_bits(z_i0[:, :, :1]))


def test_conditional_velocity_is_recomputed_at_every_step():
    """The optimizer caches v_cond because z_bar_i is fixed; here z moves, so caching would be wrong."""
    z_start, z_i0, base_context, nulls = _inputs(seed=4)
    steps = len(_SIGMAS) - 1
    seen = {"cond": 0, "unc": 0, "latents": []}
    inner = _coupled_velocity()
    base_rows = np.broadcast_to(np.asarray(base_context[:_L]), (_B, _L, _D))

    def counting_velocity(z, timestep_2d, context):
        # Replay takes no gradients, so every context is concrete even inside the scan body.
        kind = "cond" if np.array_equal(np.asarray(context[:, :_L]), base_rows) else "unc"
        seen[kind] += 1
        seen["latents"].append(np.asarray(z))
        return inner(z, timestep_2d, context)

    with jax.disable_jit():
        _run(counting_velocity, z_start, z_i0, base_context, nulls)

    assert seen["cond"] == steps, seen
    assert seen["unc"] == steps, seen
    # Both forwards of a step see the same latent, and it changes between steps (z is not cached).
    for i in range(steps):
        np.testing.assert_array_equal(seen["latents"][2 * i], seen["latents"][2 * i + 1])
        if i:
            assert not np.array_equal(seen["latents"][2 * i], seen["latents"][2 * (i - 1)])


def test_trajectory_shape_endpoints_and_dtype():
    z_start, z_i0, base_context, nulls = _inputs(seed=5, dtype=jnp.bfloat16)

    z_final, traj = _run(_coupled_velocity(), z_start, z_i0, base_context, nulls, return_trajectory=True)

    assert z_final.dtype == jnp.float32 and traj.dtype == jnp.float32
    assert traj.shape == (len(_SIGMAS), _B, _C, _F, _H, _W)
    np.testing.assert_array_equal(_f32_bits(traj[0]), _f32_bits(_pin(z_start.astype(jnp.float32), z_i0)))
    np.testing.assert_array_equal(_f32_bits(traj[-1]), _f32_bits(z_final))


def test_unbatched_nulls_are_broadcast_over_the_batch():
    z_start, z_i0, base_context, nulls = _inputs(seed=6)
    velocity_fn = _coupled_velocity()

    shared = nulls[:, 0]  # [N, L, D]
    from_shared = _run(velocity_fn, z_start, z_i0, base_context, shared)
    from_tiled = _run(velocity_fn, z_start, z_i0, base_context, jnp.broadcast_to(shared[:, None], nulls.shape))

    np.testing.assert_array_equal(_f32_bits(from_shared), _f32_bits(from_tiled))


def test_batched_replay_equals_the_per_example_singleton_replays():
    """Each example replays independently of the rest of the batch.

    This is the path R4c's artifact verification will take: a published record carries one example,
    so verifying it means replaying at B=1 and expecting exactly what the B=2 cohort run produced.
    Asserted bitwise on the full trajectory, not just the endpoint.
    """
    z_start, z_i0, base_context, nulls = _inputs(seed=10)
    velocity_fn = _coupled_velocity()
    assert not np.array_equal(np.asarray(nulls[:, 0]), np.asarray(nulls[:, 1]))  # genuinely per example

    z_final, traj = _run(velocity_fn, z_start, z_i0, base_context, nulls, return_trajectory=True)

    for b in range(_B):
        alone_final, alone_traj = _run(
            velocity_fn,
            z_start[b : b + 1],
            z_i0[b : b + 1],
            base_context,
            nulls[:, b : b + 1],
            return_trajectory=True,
        )
        np.testing.assert_array_equal(_f32_bits(z_final[b : b + 1]), _f32_bits(alone_final))
        np.testing.assert_array_equal(_f32_bits(traj[:, b : b + 1]), _f32_bits(alone_traj))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda n: n[:-1], "nulls must carry one entry per sampler step"),
        (lambda n: jnp.concatenate([n, n[:1]]), "nulls must carry one entry per sampler step"),
        (lambda n: n[0, 0], "nulls must be"),  # rank 2
        (lambda n: n[..., :-1], "inconsistent with base_context"),  # wrong feature dim
        # A rank-4 tensor with a unit batch is NOT a broadcast request: it would silently drive both
        # examples with example 0's nulls, since the velocity seam broadcasts and the shape guard
        # still sees v.shape == z.shape (Codex R4a review, finding 1).
        (lambda n: n[:, :1], "nulls batch"),
        (lambda n: jnp.zeros((n.shape[0], _B, _S + 1, _D), jnp.float32), "exceeds context length"),
    ],
)
def test_rejects_malformed_nulls(mutate, message):
    z_start, z_i0, base_context, nulls = _inputs(seed=7)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), z_start, z_i0, base_context, mutate(nulls))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"guide_scale": float("inf")}, "guide_scale must be finite"),
        ({"guide_scale": float("nan")}, "guide_scale must be finite"),
        ({"sigmas": (1.0, 0.6, 0.3)}, "must end at 0.0"),
        ({"sigmas": (np.inf, 0.6, 0.3, 0.0)}, "must be finite"),
    ],
)
def test_rejects_malformed_arguments(kwargs, message):
    z_start, z_i0, base_context, nulls = _inputs(seed=8)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), z_start, z_i0, base_context, nulls, **kwargs)


def test_rejects_a_velocity_with_the_wrong_shape():
    z_start, z_i0, base_context, nulls = _inputs(seed=9)

    with pytest.raises(ValueError, match="velocity_fn returned shape"):
        _run(lambda z, t, c: jnp.zeros((), jnp.float32), z_start, z_i0, base_context, nulls)
