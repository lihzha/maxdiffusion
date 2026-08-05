"""exp_05 S3 — ``replay_with_positive``: deployment CFG replay under per-step positive contexts.

Plan §3 step 3, ported from ``regenerate_with_positive_embeds``
(``third_party/Wan2.2/scripts/embedding_search.py:822-853``, submodule pin f370228). The conditional
branch takes the optimized 8-token ``C_i`` **directly** as the whole ``encoder_hidden_states``; the
unconditional branch keeps the frozen 512-row T5("").

Four properties carry the round:

- **Both velocities are fresh at every step**, exactly as in exp_04's ``replay_with_nulls``. Neither
  can be cached: replay *moves* ``z`` after every step, and both branches are evaluated at the current
  ``z_i``. (The optimizer caches ``v_unc`` only because it holds ``z_bar_i`` fixed across its inner
  loop -- a property replay does not have.) Pinned by the call-structure test: 2N forwards, strictly
  alternating ``[B, 8, D]`` and ``[B, S, D]``.
- **THE MUST CONTRACT (S2 review, finding 2), discharged here.** S1 could only characterize
  conditional-velocity parity *at the seam*, against a direct transformer call, because this operator
  did not exist. ``test_the_deployed_forward_is_the_replay_operator_s_own_v_cond`` re-runs that
  fixture through the ACTUAL ``replay_with_positive``: the deployed ``wan_pre_context_adapter_forward``
  with a head emitting ``C`` must equal the velocity this operator computes for its conditional
  branch -- bitwise at fp32, and bitwise at the deployed bf16 activation dtype when the injected
  ``velocity_fn`` performs the cast (the one rule's composition).
- **The B0 control is an ACTIVE-CFG control.** exp_04's A0 sets the nulls to the base rows, making
  ``v_unc == v_cond`` so CFG collapses and the output is w-invariant. The positive slot cannot do
  that: even at ``C = pos_context_from_t5(T5(""))`` the two branches are different forwards at
  different sequence lengths (8 vs 512 -- the S1 finding), so B0's output **does** depend on w. Pinned
  as a difference, deliberately contrasting exp_04's invariance test.
- **The operator passes contexts through unchanged** (the module's one cast rule): fp32 in, fp32 to
  ``velocity_fn``, and the activation-dtype cast is the injected closure's job.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bit_test_helpers import bits as _bits, f32_bits as _f32_bits
from maxdiffusion.models.wan.pos_context_inversion_wan import POS_L, pos_context_from_t5, replay_with_positive


_B, _C, _F, _H, _W = 2, 2, 2, 4, 6
_S, _D, _POS_L = 16, 4, 8  # uncond context rows, width, deployed positive-token count (hardcoded)
_SIGMAS = (1.0, 0.6, 0.0)  # N = 2 steps
_NUM_TRAIN_TIMESTEPS = 1000
_C_Z, _C_T = 0.05, 1e-4  # the oracle's coupling to the evaluation point and the timestep


def _raw_bits(x):
    x = jnp.asarray(x)
    return _bits(x, x.dtype, jnp.uint16 if x.dtype.itemsize == 2 else jnp.uint32)


def _pin(z, z_i0):
    return z.at[:, :, :1].set(z_i0[:, :, :1])


def _timestep_2d(sigma, batch=_B, f_lat=_F, h_lat=_H, w_lat=_W):
    tokens_per_frame = (h_lat // 2) * (w_lat // 2)
    full = jnp.full((batch, f_lat * tokens_per_frame), sigma * _NUM_TRAIN_TIMESTEPS, jnp.float32)
    return full.at[:, :tokens_per_frame].set(0.0)


def _inputs(batch=_B, seed=0, steps=len(_SIGMAS) - 1):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    z_start = jax.random.normal(keys[0], (batch, _C, _F, _H, _W), jnp.float32)
    z_i0 = jax.random.normal(keys[1], (batch, _C, 1, _H, _W), jnp.float32)
    base_context = jax.random.normal(keys[2], (_S, _D), jnp.float32)
    pos_embeds = jax.random.normal(keys[3], (steps, batch, _POS_L, _D), jnp.float32)
    return z_start, z_i0, base_context, pos_embeds


def _coupled_velocity(seed=11):
    """v = <context, weights> * pattern + c_z * z + c_t * mean(timestep), at either sequence length."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    weights = jax.random.normal(keys[0], (_S, _D), jnp.float32)
    pattern = jax.random.normal(keys[1], (_C, _F, _H, _W), jnp.float32)

    def velocity_fn(z, timestep_2d, context):
        scale = jnp.sum(context * weights[: context.shape[1]], axis=(1, 2))
        per_example_t = jnp.mean(timestep_2d, axis=1)
        return scale[:, None, None, None, None] * pattern + _C_Z * z + _C_T * per_example_t[:, None, None, None, None]

    return velocity_fn


def _reference_replay(velocity_fn, z_start, z_i0, sigmas, pos_embeds, base_context, guide_scale):
    """The whole operator written out: literal loop, own pin, own timestep, no production helper."""
    batch = z_start.shape[0]
    uncond = jnp.broadcast_to(base_context, (batch, _S, _D))
    z = _pin(z_start, z_i0)
    trajectory = [z]
    for i in range(len(sigmas) - 1):
        timestep_2d = _timestep_2d(float(sigmas[i]), batch=batch)
        context = jnp.broadcast_to(pos_embeds[i], (batch, pos_embeds.shape[-2], _D))
        v_cond = velocity_fn(z, timestep_2d, context)
        v_unc = velocity_fn(z, timestep_2d, uncond)
        v_cfg = v_unc + guide_scale * (v_cond - v_unc)
        z = _pin(z + (float(sigmas[i + 1]) - float(sigmas[i])) * v_cfg, z_i0)
        trajectory.append(z)
    return z, jnp.stack(trajectory)


def _run(velocity_fn, z_start, z_i0, base_context, pos_embeds, *, guide_scale=5.0, sigmas=_SIGMAS, **kwargs):
    return replay_with_positive(
        velocity_fn,
        z_start,
        z_i0,
        jnp.asarray(sigmas, jnp.float32),
        pos_embeds,
        base_context,
        guide_scale=guide_scale,
        **kwargs,
    )


def _recording_velocity(inner):
    seen = []

    def velocity_fn(z, timestep_2d, context):
        seen.append(
            {
                "kind": "cond" if context.shape[1] == _POS_L else "uncond",
                "shape": tuple(context.shape),
                "dtype": jnp.asarray(context).dtype,
                "context": np.asarray(context),
                "z": np.asarray(z),
                "timestep": np.asarray(timestep_2d),
            }
        )
        return inner(z, timestep_2d, context)

    return velocity_fn, seen


# --------------------------------------------------------------------------------------------------
# 1. The algorithm.
# --------------------------------------------------------------------------------------------------


def test_matches_the_hand_rolled_reference():
    """Pins the CFG combine, the dsigma sign, the per-step context indexing and every pin at once.

    Tolerance, not bitwise: the module runs the loop in ``lax.scan``, the reference runs it eagerly,
    so the two differ by FMA-level rounding -- measured worst case 9.3e-7 relative / 7.6e-6 absolute
    on a scale of 3.0e+1, i.e. ~1000x headroom under ``rtol``. Every mutant in the battery moves these
    outputs by O(1).
    """
    z_start, z_i0, base_context, pos_embeds = _inputs()
    velocity_fn = _coupled_velocity()

    z_final, trajectory = _run(velocity_fn, z_start, z_i0, base_context, pos_embeds, return_trajectory=True)
    want_final, want_traj = _reference_replay(velocity_fn, z_start, z_i0, _SIGMAS, pos_embeds, base_context, 5.0)

    np.testing.assert_allclose(np.asarray(z_final), np.asarray(want_final), rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(np.asarray(trajectory), np.asarray(want_traj), rtol=1e-3, atol=1e-5)


def test_cfg_mixing_uses_the_deployment_weight():
    """With the branches returning distinct constants, each step must be v_unc + w(v_cond - v_unc)."""
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=4)
    guide_scale, c_cond, c_unc = 5.0, 0.75, -0.25

    def branch_velocity(z, timestep_2d, context):
        return (c_cond if context.shape[1] == _POS_L else c_unc) * jnp.ones_like(z)

    _, trajectory = _run(
        branch_velocity, z_start, z_i0, base_context, pos_embeds, guide_scale=guide_scale, return_trajectory=True
    )

    v_cfg = c_unc + guide_scale * (c_cond - c_unc)
    expected = _pin(z_start, z_i0)
    for i in range(len(_SIGMAS) - 1):
        expected = _pin(expected + (_SIGMAS[i + 1] - _SIGMAS[i]) * v_cfg, z_i0)
        np.testing.assert_allclose(np.asarray(trajectory[i + 1]), np.asarray(expected), rtol=1e-6, atol=1e-7)


def test_the_frozen_context_b0_control_still_has_active_cfg():
    """B0 is a frozen-C control, NOT an unguided sampler (plan §4, the S1 finding).

    exp_04's A0 puts the base rows back in the null slot, so ``v_unc == v_cond``, CFG collapses and
    the replay is w-invariant. Here the branches are different forwards at different sequence lengths
    (``POS_L`` vs the full T5 length), so even the frozen warm-start context leaves CFG active: two
    guidance weights must give different trajectories -- measured separation max |Δ| 7.9e+1 on a
    tensor of scale 4.3e+1, so this is a structural difference, not a rounding artifact. Anything
    else would mean the operator is not really running two branches.
    """
    z_start, z_i0, base_context, _ = _inputs(seed=5)
    velocity_fn = _coupled_velocity()
    steps = len(_SIGMAS) - 1
    frozen = jnp.broadcast_to(pos_context_from_t5(base_context), (steps, _B, _POS_L, _D))

    at_w1 = _run(velocity_fn, z_start, z_i0, base_context, frozen, guide_scale=1.0)
    at_w5 = _run(velocity_fn, z_start, z_i0, base_context, frozen, guide_scale=5.0)
    optimized = _run(velocity_fn, z_start, z_i0, base_context, frozen + 0.5, guide_scale=5.0)

    assert not np.allclose(np.asarray(at_w1), np.asarray(at_w5), rtol=1e-3), "B0 must not be w-invariant"
    assert not np.allclose(np.asarray(at_w5), np.asarray(optimized), rtol=1e-3)  # C itself steers the replay


# --------------------------------------------------------------------------------------------------
# 2. THE MUST CONTRACT — S1's parity fixture, through the actual operator.
# --------------------------------------------------------------------------------------------------


def _replay_v_cond_vs_deployed(dtype, cast_in_velocity_fn):
    """Run the real operator on a tiny WanModel and compare its conditional branch to deployment.

    The deployed forward is evaluated on exactly the ``(z, timestep)`` the operator's own ``v_cond``
    call received, so what is compared is this operator's conditioning against
    ``wan_pre_context_adapter_forward``'s -- not two independently constructed calls. ``_ConstantContextHead``
    and the batched fixture contexts are imported from the S1 test module, so the fixture really is
    S1's (Codex S1 review, verification 1).
    """
    pytest.importorskip("torch")
    pytest.importorskip("aqt")
    from flax import nnx
    from flax.linen import partitioning as nn_partitioning
    from maxdiffusion.models.wan.side_adapter_wan import wan_action_adapter_forward
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
    from test_pos_context_truncate_pad import _ConstantContextHead

    text_dim = 32
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    set_mesh = getattr(jax, "set_mesh", None)
    mesh_ctx = set_mesh(mesh) if set_mesh is not None else mesh
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    z_i0 = jax.random.normal(keys[0], (_B, _C, 1, _H, _W), jnp.float32)
    z_start = _pin(jax.random.normal(keys[1], (_B, _C, _F, _H, _W), jnp.float32), z_i0)
    base_context = jax.random.normal(keys[2], (_S, text_dim), jnp.float32)
    contexts = jnp.stack(  # one DISTINCT context per example (the S1 lesson)
        [pos_context_from_t5(base_context) + 0.3 * float(b + 1) for b in range(_B)]
    )
    seen = {}

    with nn_partitioning.axis_rules(()), mesh_ctx:
        model = WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=_C,
            out_channels=_C,
            text_dim=text_dim,
            freq_dim=16,
            ffn_dim=32,
            num_layers=2,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
            dtype=dtype,
            weights_dtype=dtype,
        )

        def velocity_fn(z, timestep_2d, context):
            # The runner-built closure owns the activation-dtype cast (the module's one rule); the
            # operator must have handed the context over untouched, which is asserted below.
            entry_dtype = jnp.asarray(context).dtype
            if cast_in_velocity_fn:
                context = context.astype(dtype)
            velocity = model(hidden_states=z, timestep=timestep_2d, encoder_hidden_states=context)
            if context.shape[1] == POS_L:
                seen.update(z=z, timestep=timestep_2d, velocity=velocity, entry_dtype=entry_dtype)
            return velocity

        with jax.disable_jit():  # so the captured z/timestep are concrete
            _run(velocity_fn, z_start, z_i0, base_context, contexts[None], sigmas=(0.7, 0.0))

        deployed = wan_action_adapter_forward(
            model,
            _ConstantContextHead(contexts),
            hidden_states=seen["z"],
            timestep=seen["timestep"],
            encoder_hidden_states=None,  # the pre_context path ignores it
            actions=None,
            deterministic=True,
            rngs=None,
        )
    assert seen["entry_dtype"] == jnp.float32  # the operator did not cast -- the closure did
    return np.asarray(seen["velocity"]), np.asarray(deployed)


def test_the_deployed_forward_is_the_replay_operator_s_own_v_cond():
    """**THE MUST CONTRACT, DISCHARGED** (S2 review, finding 2; Codex S1 review, verification 1).

    S1 pinned this against a direct transformer call because ``replay_with_positive`` did not exist.
    Now it is pinned against the operator itself: the velocity the operator computes for its
    conditional branch is bit-for-bit the deployed ``wan_pre_context_adapter_forward`` output for a
    head emitting the same C. Serialized targets and the trained adapter therefore realize the same
    conditioning, so K4's closed-loop gap is an adapter gap and not a representation gap.
    """
    from_operator, deployed = _replay_v_cond_vs_deployed(jnp.float32, cast_in_velocity_fn=False)

    assert float(np.std(deployed)) > 1e-3, deployed  # a degenerate output would make this vacuous
    np.testing.assert_array_equal(_f32_bits(from_operator), _f32_bits(deployed))


def test_parity_at_bf16_holds_for_the_operator_plus_casting_velocity_fn():
    """The one rule's composition at the deployed activation dtype: the operator passes fp32 through,
    the injected ``velocity_fn`` casts, and the pair together reproduces deployment bit for bit --
    while the same operator with a NON-casting closure does not (S1 measured the cast is no no-op)."""
    cast, deployed = _replay_v_cond_vs_deployed(jnp.bfloat16, cast_in_velocity_fn=True)
    uncast, _ = _replay_v_cond_vs_deployed(jnp.bfloat16, cast_in_velocity_fn=False)

    np.testing.assert_array_equal(_raw_bits(cast), _raw_bits(deployed))
    assert not np.array_equal(_raw_bits(uncast), _raw_bits(deployed))


# --------------------------------------------------------------------------------------------------
# 3. Structure, batching and dtypes.
# --------------------------------------------------------------------------------------------------


def test_both_velocities_are_recomputed_at_every_step_in_the_deployed_order():
    """2N forwards, strictly alternating cond/uncond -- neither branch may be cached across steps,
    because both are evaluated at the current z, which moves at every step. Each pair is also checked
    to receive that step's own z (the trajectory entry) and its own per-token timestep."""
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=6)
    velocity_fn, seen = _recording_velocity(_coupled_velocity())
    steps = len(_SIGMAS) - 1
    tokens_per_frame = (_H // 2) * (_W // 2)

    with jax.disable_jit():
        _, trajectory = _run(velocity_fn, z_start, z_i0, base_context, pos_embeds, return_trajectory=True)

    assert [call["kind"] for call in seen] == ["cond", "uncond"] * steps
    for i in range(steps):
        for call in seen[2 * i : 2 * i + 2]:
            assert call["shape"] == ((_B, _POS_L, _D) if call["kind"] == "cond" else (_B, _S, _D))
            assert call["dtype"] == jnp.float32  # contexts reach the seam uncast (the one rule)
            np.testing.assert_array_equal(_f32_bits(call["z"]), _f32_bits(trajectory[i]), err_msg=f"step {i}")
            np.testing.assert_array_equal(call["timestep"][:, :tokens_per_frame], 0.0)
            np.testing.assert_allclose(
                call["timestep"][:, tokens_per_frame:], _SIGMAS[i] * _NUM_TRAIN_TIMESTEPS, rtol=1e-6, atol=1e-4
            )
        np.testing.assert_array_equal(_f32_bits(seen[2 * i]["context"]), _f32_bits(pos_embeds[i]))
        np.testing.assert_array_equal(
            _f32_bits(seen[2 * i + 1]["context"]), _f32_bits(jnp.broadcast_to(base_context, (_B, _S, _D)))
        )


def test_every_step_is_pinned_and_the_trajectory_endpoints_are_the_run_s_own():
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=7)

    z_final, trajectory = _run(_coupled_velocity(), z_start, z_i0, base_context, pos_embeds, return_trajectory=True)

    assert trajectory.shape == (len(_SIGMAS), _B, _C, _F, _H, _W)
    np.testing.assert_array_equal(_f32_bits(trajectory[0]), _f32_bits(_pin(z_start, z_i0)))
    np.testing.assert_array_equal(_f32_bits(trajectory[-1]), _f32_bits(z_final))
    for state in trajectory:
        np.testing.assert_array_equal(_f32_bits(state[:, :, :1]), _f32_bits(z_i0))


def test_every_example_s_whole_trajectory_is_independent_of_the_rest_of_the_batch():
    """Per-example distinct embeds (the S1 lesson), checked for EVERY example over the FULL trajectory.

    Comparing only example 0's endpoint -- as this test first did -- is blind to a collapse onto row
    0: example 0 is precisely the example such a defect leaves untouched, and an endpoint can agree
    where the path did not. So every example's whole trajectory is compared bitwise against its own
    singleton run (S3 review, finding 1; the R4a/S1 standard).
    """
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=8)
    velocity_fn = _coupled_velocity()
    assert not np.allclose(np.asarray(pos_embeds[:, 0]), np.asarray(pos_embeds[:, 1]))  # the fixture's point

    batched_final, batched_traj = _run(velocity_fn, z_start, z_i0, base_context, pos_embeds, return_trajectory=True)

    for b in range(_B):
        one = slice(b, b + 1)
        alone_final, alone_traj = _run(
            velocity_fn, z_start[one], z_i0[one], base_context, pos_embeds[:, one], return_trajectory=True
        )
        np.testing.assert_array_equal(_f32_bits(batched_traj[:, one]), _f32_bits(alone_traj), err_msg=f"traj {b}")
        np.testing.assert_array_equal(_f32_bits(batched_final[one]), _f32_bits(alone_final), err_msg=f"final {b}")


def test_a_shared_per_step_context_may_be_broadcast_over_the_batch():
    """``[N, L, D]`` is the "one context per step, shared by the batch" convention (exp_04 R4a).

    Compared over the full trajectory against contexts repeated explicitly, so the broadcast is pinned
    as a genuine per-example repeat rather than as something that merely agrees at the endpoint.
    """
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=9)
    shared = pos_embeds[:, 0]
    repeated = jnp.broadcast_to(shared[:, None], pos_embeds.shape)

    broadcast_final, broadcast_traj = _run(
        _coupled_velocity(), z_start, z_i0, base_context, shared, return_trajectory=True
    )
    explicit_final, explicit_traj = _run(
        _coupled_velocity(), z_start, z_i0, base_context, repeated, return_trajectory=True
    )

    np.testing.assert_array_equal(_f32_bits(broadcast_traj), _f32_bits(explicit_traj))
    np.testing.assert_array_equal(_f32_bits(broadcast_final), _f32_bits(explicit_final))


def test_latent_arithmetic_is_float32_even_from_bfloat16_inputs():
    """The reference calls ``.float()`` on everything before stepping; a bf16 cache must not make the
    replay itself bf16."""
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=10)
    as_bf16 = [x.astype(jnp.bfloat16) for x in (z_start, z_i0, base_context, pos_embeds)]

    from_bf16 = _run(_coupled_velocity(), *as_bf16)
    upcast = _run(_coupled_velocity(), *[x.astype(jnp.float32) for x in as_bf16])

    assert from_bf16.dtype == jnp.float32
    np.testing.assert_array_equal(_f32_bits(from_bf16), _f32_bits(upcast))


# --------------------------------------------------------------------------------------------------
# 4. Fail-closed validation.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sigmas": (1.0, 0.6, 0.3, 0.0)}, "one entry per sampler step"),  # grid longer than the cache
        ({"sigmas": (1.0,)}, "at least one step"),
        ({"sigmas": (0.6, 1.0, 0.0)}, "strictly descending"),
        ({"sigmas": (1.0, 0.6, 0.1)}, "must end at 0.0"),
        ({"guide_scale": float("inf")}, "guide_scale must be finite"),
        ({"guide_scale": float("nan")}, "guide_scale must be finite"),
    ],
)
def test_rejects_malformed_arguments(kwargs, message):
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=11)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), z_start, z_i0, base_context, pos_embeds, **kwargs)


@pytest.mark.parametrize(
    "mangle, message",
    [
        # A [N, 1, L, D] cache at B = 2 must be REJECTED, not broadcast: it would drive the whole
        # batch with example 0's contexts while every shape guard still passed (exp_04 R4a's lesson).
        (lambda z, i, c, p: (z, i, c, p[:, :1]), "batch"),
        (lambda z, i, c, p: (z, i, c, p[:1]), "one entry per sampler step"),
        (lambda z, i, c, p: (z, i, c, p[..., :1]), "inconsistent with base_context"),
        (lambda z, i, c, p: (z, i, c, p[0, 0]), "pos_embeds must be"),  # rank-2
        (lambda z, i, c, p: (z[0], i, c, p), "z_start must be"),  # rank-4
        (lambda z, i, c, p: (z, i[:1], c, p), "z_i0 shape"),
        (lambda z, i, c, p: (z, jnp.broadcast_to(i, (_B, _C, 3, _H, _W)), c, p), "z_i0 must carry"),
        (lambda z, i, c, p: (z, i, c[None].repeat(3, axis=0), p), "unit leading axis"),
        (lambda z, i, c, p: (z, i, c[0], p), "base_context must be"),
    ],
)
def test_rejects_inconsistent_geometry(mangle, message):
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=12)
    z_start, z_i0, base_context, pos_embeds = mangle(z_start, z_i0, base_context, pos_embeds)

    with pytest.raises(ValueError, match=message):
        _run(_coupled_velocity(), z_start, z_i0, base_context, pos_embeds)


def test_rejects_a_velocity_with_the_wrong_shape():
    z_start, z_i0, base_context, pos_embeds = _inputs(seed=13)

    with pytest.raises(ValueError, match="velocity_fn returned shape"):
        _run(lambda z, t, c: jnp.zeros((), jnp.float32), z_start, z_i0, base_context, pos_embeds)
