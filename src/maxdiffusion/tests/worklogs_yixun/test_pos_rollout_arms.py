"""exp_06 `rollout_adapter` — T3b-2 `arm-losses`: both objectives, one stream (plan §3, §3b).

The contract: **both arms are the same function on the same stream; the arm selects only the loss.**
exp_06 claims R-B beats matched-C0 *because of the objective*, so every other difference between the
arms is a confound. This file's job is to leave nowhere for one to hide.

**matched-C0 carries T2's double-equivalence obligation, extended.** It must equal the settled inline
``_denoising_loss`` bitwise in VALUE and in GRADIENT. The gradient half is not decoration: a control
arm whose gradients were quietly wrong would make R-B look better, and no value comparison would
show it (the S16 lesson from T3a).

**An overclaim, corrected by the battery.** An earlier draft of this file said the gradient
comparison "pins the stop-gradient pattern in place". It does not, and the mutants proved it:
removing either of C0's stop-grads was caught only by the structural test. The reason is worth
stating because it is the same fact that makes the pattern correct here — in a ONE-STEP objective
``z_t`` is built from data and noise alone, so it and ``v_uncond`` carry no parameter dependence and
cutting their gradients cuts nothing. They are inert, which is exactly why they are free here and
truncating in a rollout. ``test_the_one_step_stop_gradients_are_inert`` proves the inertness instead
of assuming it, and the pattern is pinned structurally.

If C0 and the settled objective disagreed, this round would STOP and report: a control that is not
the deployed objective invalidates the causal claim Yixun's decision 1 exists to support.

The verbatim copies below are held to their source by an AST drift tripwire (T2's technique:
contiguous-subsequence match on unparsed statements — whitespace-proof, edit-lethal). The settled
trainer is not edited.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from maxdiffusion import pos_rollout_arms as arms
from maxdiffusion import pos_rollout_stream as stream
from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    apply_first_frame_pin,
    wan_action_adapter_forward,
)
from maxdiffusion.pos_rollout_losses import rollout_endpoint_loss
from maxdiffusion.pos_rollout_step import build_cfg_velocity_fn
from maxdiffusion.tests.worklogs_yixun.test_pos_rollout_step import (
    _B,
    _C,
    _F,
    _GUIDE,
    _H,
    _NUM_TRAIN_TIMESTEPS,
    _STEPS,
    _W,
    _grid,
    _inputs,
    _mesh_context,
    _requires_backend,
    _tiny_cfg_stack,
    _tree_norm,
)

_MODULE_PATH = Path(arms.__file__).resolve()
_TRAINER_PATH = _MODULE_PATH.parent / "trainers" / "wan_ti2v_side_adapter_trainer.py"
_THIS_PATH = Path(__file__).resolve()
_K = 2


def _function_node(source: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _context(sigmas, timesteps, null_context, guide_scale=_GUIDE):
    return arms.ArmContext(
        sigmas=sigmas,
        timesteps=timesteps,
        null_context=null_context,
        guide_scale=guide_scale,
        weights_dtype=jnp.float32,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
        num_steps=_STEPS,
        k_b=_K,
    )


def _batch(data):
    return {"z_video": data["z"], "z_i0": data["z_i0"], "actions": data["actions"]}


class _Config:
    def __init__(self, guide_scale=_GUIDE):
        self.side_adapter_guide_scale = guide_scale


class _State:
    def __init__(self, null_context):
        self.null_context = null_context


# =============================================================================================
# The verbatim reference: the OBJECTIVE of trainers/wan_ti2v_side_adapter_trainer._denoising_loss
# (lines 153-159 and 162-196 at d14ed71). The sampling above it differs by design -- exp_06 keys its
# epsilon and sigma index on the loop step for resume-stability -- so eps and t_idx arrive as
# arguments and the two contiguous runs of objective statements are copied unchanged.
# =============================================================================================


def _settled_denoising_objective(
    *,
    params,
    adapters,
    transformer,
    state,
    config,
    scheduler,
    sigmas,
    t_idx,
    eps,
    z_video_f32,
    z_i0_f32,
    actions,
    weights_dtype,
    dropout_rng,
    b,
    f_lat,
    h_lat,
    w_lat,
):
    sigma_t = sigmas[t_idx]
    step_t = sigma_t * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    null_context = jnp.broadcast_to(
        state.null_context.astype(weights_dtype),
        (b, state.null_context.shape[1], state.null_context.shape[2]),
    )

    sigma_b = sigma_t.reshape((b, 1, 1, 1, 1))
    z_t_f32 = (1.0 - sigma_b) * z_video_f32 + sigma_b * eps
    z_t_f32 = apply_first_frame_pin(z_t_f32, z_i0_f32)
    z_t = z_t_f32.astype(weights_dtype)

    v_cond = wan_action_adapter_forward(
        transformer,
        adapters,
        hidden_states=z_t,
        timestep=timestep_2d,
        encoder_hidden_states=null_context,
        actions=actions,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )
    if abs(config.side_adapter_guide_scale - 1.0) > 1e-6:
        # Match ../Wan2.2: the unconditional CFG branch is frozen and run
        # without gradient through either the branch or its latent input.
        v_uncond = transformer(
            hidden_states=jax.lax.stop_gradient(z_t),
            timestep=timestep_2d,
            encoder_hidden_states=null_context,
            deterministic=True,
        )
        v_uncond = jax.lax.stop_gradient(v_uncond)
        v_pred = v_uncond + config.side_adapter_guide_scale * (v_cond - v_uncond)
    else:
        v_pred = v_cond

    v_target = eps - z_video_f32
    mask = jnp.ones((1, z_video_f32.shape[1], f_lat, h_lat, w_lat), dtype=jnp.float32)
    mask = mask.at[:, :, :1, :, :].set(0.0)
    diff = (v_pred.astype(jnp.float32) - v_target.astype(jnp.float32)) * mask
    n_valid = jnp.maximum(jnp.sum(mask) * b, 1.0)
    loss = jnp.sum(diff**2) / n_valid
    del params
    return loss


def _settled_reference_loss(params, transformer, adapters_graphdef, adapters_rest, data, sigmas, t_idx, dropout_rng):
    adapters = nnx.merge(adapters_graphdef, params, adapters_rest)
    z_video_f32 = data["z"].astype(jnp.float32)
    z_i0_f32 = data["z_i0"].astype(jnp.float32)
    actions = data["actions"].astype(jnp.float32)
    b, _, f_lat, h_lat, w_lat = z_video_f32.shape
    return _settled_denoising_objective(
        params=params,
        adapters=adapters,
        transformer=transformer,
        state=_State(data["null_context"]),
        config=_Config(),
        scheduler=type("S", (), {"config": type("C", (), {"num_train_timesteps": _NUM_TRAIN_TIMESTEPS})})(),
        sigmas=sigmas,
        t_idx=t_idx,
        eps=data["epsilon"],
        z_video_f32=z_video_f32,
        z_i0_f32=z_i0_f32,
        actions=actions,
        weights_dtype=jnp.float32,
        dropout_rng=dropout_rng,
        b=b,
        f_lat=f_lat,
        h_lat=h_lat,
        w_lat=w_lat,
    )


def _fixture(seed=31):
    data = _inputs(seed)
    data["draws"] = stream._draw_step_stream(
        seed=0, global_step=7, num_steps=_STEPS, k_b=_K, shape=(_B, _C, _F, _H, _W)
    )
    data["epsilon"] = data["draws"].epsilon
    data["null_context"] = data["null_context"][:1]  # deployed shape: ONE null context, broadcast
    sigmas, timesteps = _grid()
    return data, sigmas, timesteps


# =============================================================================================
# 1. matched-C0: the double equivalence, in VALUE and in GRADIENT.
# =============================================================================================


def test_the_one_step_arm_equals_the_settled_denoising_objective_in_value_and_gradient():
    """THE control's obligation: the deployed objective, in value and in gradient.

    Note what this does and does not show: it proves C0 computes and differentiates the same function
    the settled trainer does. It does NOT pin the stop-gradient syntax -- those lines are provably
    inert in a one-step objective (see the inertness test), so only the AST assertion pins them.
    """
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data, sigmas, timesteps = _fixture()
    context = _context(sigmas, timesteps, data["null_context"])
    graphdef, _, rest = nnx.split(adapters, nnx.Param, ...)
    rules, mesh = _mesh_context()
    with rules, mesh:
        loss_fn, params = arms.build_arm("one_step", transformer, adapters)
        for t_idx in (jnp.zeros((_B,), jnp.int32), jnp.asarray([3, 17], jnp.int32)):
            dropout = jax.random.key(5)

            def mine(p, t=t_idx, d=dropout):
                return loss_fn(
                    p,
                    _batch(data),
                    context,
                    draws=dataclasses.replace(data["draws"], t_idx=t),
                    seed=0,
                    global_step=7,
                    dropout_rng=d,
                )[0]

            def settled(p, t=t_idx, d=dropout):
                return _settled_reference_loss(p, transformer, graphdef, rest, data, sigmas, t, d)

            assert np.array_equal(
                np.asarray(mine(params)), np.asarray(settled(params))
            ), "VALUE differs from the settled objective"
            got, want = jax.grad(mine)(params), jax.grad(settled)(params)
            difference = _tree_norm(jax.tree.map(lambda a, b: a - b, got, want))
            assert (
                difference == 0.0
            ), f"GRADIENT differs from the settled objective (relL2 {difference / _tree_norm(want):.3e})"


def test_the_settled_denoising_objective_has_not_drifted_from_the_copy_pinned_here():
    # Two contiguous runs of statements (the eps draw between them differs by design). Structural,
    # so reformatting is free and an edit is lethal.
    settled = [
        ast.unparse(node) for node in _function_node(_TRAINER_PATH.read_text(encoding="utf-8"), "_denoising_loss").body
    ]
    mine = [
        ast.unparse(node)
        for node in _function_node(_THIS_PATH.read_text(encoding="utf-8"), "_settled_denoising_objective").body
    ]
    mine = [statement for statement in mine if not statement.startswith(("del ", "return "))]
    runs, current = [], []
    for statement in mine:
        current.append(statement)
    # Split the copy at the point the settled body draws its epsilon.
    boundary = next(index for index, statement in enumerate(current) if statement.startswith("sigma_b ="))
    for run in (current[:boundary], current[boundary:]):
        assert run, "empty run"
        assert any(
            settled[index : index + len(run)] == run for index in range(len(settled) - len(run) + 1)
        ), f"this run no longer matches a contiguous span of _denoising_loss:\n{run[0]}"
    del runs


def test_production_dropout_is_zero_and_the_dropout_key_is_therefore_inert():
    """BLOCKER 2, closed by proof rather than by plumbing.

    matched-C0 keeps the deployed ``deterministic=False`` forward and accepts a ``dropout_rng``; R-B
    discards it. That would make the arms differ in randomness — except that production dropout is
    ZERO, so the key changes nothing. Proven on both halves (value and gradient), and the config
    value is pinned so the proof cannot rot: if the rate ever becomes nonzero, the key must become a
    stream draw like the others, and this test is what will say so.
    """
    _requires_backend()
    config_text = (_MODULE_PATH.parent / "configs" / "base_wan_5b_side_adapter.yml").read_text(encoding="utf-8")
    assert "dropout: 0.0" in config_text, "production dropout is no longer zero; the key must become a stream draw"
    assert arms.PRODUCTION_DROPOUT_RATE == 0.0

    transformer, adapters = _tiny_cfg_stack()
    data, sigmas, timesteps = _fixture()
    context = _context(sigmas, timesteps, data["null_context"])
    rules, mesh = _mesh_context()
    with rules, mesh:
        loss_fn, params = arms.build_arm("one_step", transformer, adapters)

        def with_key(key):
            def value(p):
                return loss_fn(p, _batch(data), context, draws=data["draws"], dropout_rng=key)[0]

            return float(value(params)), jax.grad(value)(params)

        first_value, first_grad = with_key(jax.random.key(5))
        other_value, other_grad = with_key(jax.random.key(9999))
        none_value, none_grad = with_key(None)
    assert first_value == other_value == none_value, "the dropout key moved the loss"
    for other in (other_grad, none_grad):
        assert (
            _tree_norm(jax.tree.map(lambda a, b: a - b, first_grad, other)) == 0.0
        ), "the dropout key moved the gradient"


def test_a_nonzero_dropout_is_REFUSED_not_merely_proven_inert():
    """A2: enforcement. Inertness at zero does not guarantee zero (review BLOCKER A2).

    A command-line override beats YAML and reaches ``WanModel`` directly, so the old test could stay
    green while a launch trained with dropout on. Both routes are now fail-closed: the declared rate
    and the constructed module.
    """
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    with pytest.raises(ValueError, match="dropout must be 0.0"):
        arms.build_arm("rollout", transformer, adapters, dropout_rate=0.1)
    with pytest.raises(ValueError, match="dropout must be 0.0"):
        arms.assert_dropout_is_zero(dropout_rate=0.1)
    # The zero path still builds, so the guard is not simply always-on.
    rules, mesh = _mesh_context()
    with rules, mesh:
        assert arms.build_arm("one_step", transformer, adapters, dropout_rate=0.0)[1] is not None
    # ...and a CONSTRUCTED module carrying nonzero dropout is refused even if the rate is not passed.
    from flax import nnx

    class _Carrier(nnx.Module):
        def __init__(self):
            self.drop = nnx.Dropout(0.25, deterministic=False)

    with pytest.raises(ValueError, match="the constructed module has rate"):
        arms.assert_dropout_is_zero(_Carrier())


def test_the_one_step_stop_gradients_are_inert_which_is_why_they_are_correct_here():
    """The fact behind the correction above: in a one-step objective these stop-grads cut nothing.

    ``z_t`` depends on the batch and the noise, never on the adapter, so ``stop_gradient`` on it and
    on ``v_uncond`` removes a gradient path that does not exist. Proven by building the same loss
    without them and comparing gradients bitwise. This is ALSO the sharpest available statement of
    why T3a's rollout forward must not copy the pattern: there ``z_i`` does depend on the adapter,
    so the same two lines delete a real path (T3a measured 46-65% of the gradient).
    """
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data, sigmas, timesteps = _fixture()
    context = _context(sigmas, timesteps, data["null_context"])
    rules, mesh = _mesh_context()
    with rules, mesh:
        make_velocity_fn, params = arms.build_one_step_velocity_fn(transformer, adapters)
        graphdef, _, rest = nnx.split(adapters, nnx.Param, ...)

        def without_stop_grads(p):
            model = nnx.merge(*nnx.split(transformer, nnx.Param, ...))
            adapter = nnx.merge(graphdef, p, rest)

            def velocity_fn(hidden_states, timestep, encoder_hidden_states):
                v_cond = wan_action_adapter_forward(
                    model,
                    adapter,
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    actions=data["actions"].astype(jnp.float32),
                    deterministic=False,
                    rngs=nnx.Rngs(dropout=jax.random.key(5)),
                )
                v_uncond = model(
                    hidden_states=hidden_states,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    deterministic=True,
                )
                return v_uncond + _GUIDE * (v_cond - v_uncond)

            return arms.one_step_denoising_loss(
                p, lambda *a, **k: velocity_fn, _batch(data), context, draws=data["draws"]
            )[0]

        def with_stop_grads(p):
            return arms.one_step_denoising_loss(
                p, make_velocity_fn, _batch(data), context, draws=data["draws"], dropout_rng=jax.random.key(5)
            )[0]

        kept, dropped = jax.grad(with_stop_grads)(params), jax.grad(without_stop_grads)(params)
    assert _tree_norm(jax.tree.map(lambda a, b: a - b, kept, dropped)) == 0.0


def test_the_one_step_arm_keeps_the_deployed_stop_gradients():
    # They are correct for a one-step objective and forbidden in R-B; the gradient equivalence above
    # is what proves they are still there, this pins where.
    builder = _function_node(_MODULE_PATH.read_text(encoding="utf-8"), "build_one_step_velocity_fn")
    unparsed = ast.unparse(builder)
    assert "jax.lax.stop_gradient(hidden_states)" in unparsed
    assert "v_uncond = jax.lax.stop_gradient(v_uncond)" in unparsed


# =============================================================================================
# 2. R-B: composition of the committed pieces, value and gradient.
# =============================================================================================


def test_the_rollout_arm_is_exactly_t2s_kernel_over_t3as_velocity():
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data, sigmas, timesteps = _fixture()
    context = _context(sigmas, timesteps, data["null_context"])
    rules, mesh = _mesh_context()
    with rules, mesh:
        loss_fn, params = arms.build_arm("rollout", transformer, adapters)
        make_velocity_fn, reference_params = build_cfg_velocity_fn(transformer, adapters)

        def mine(p):
            return loss_fn(p, _batch(data), context, draws=data["draws"], seed=0, global_step=7)[0]

        def reference(p):
            value, _ = rollout_endpoint_loss(
                z_video_f32=data["z"].astype(jnp.float32),
                z_i0_f32=data["z_i0"].astype(jnp.float32),
                eps_f32=data["epsilon"].astype(jnp.float32),
                sigmas=sigmas,
                timesteps=timesteps,
                context=jnp.broadcast_to(data["null_context"], (_B, *data["null_context"].shape[1:])),
                velocity_fn=make_velocity_fn(p, actions=data["actions"].astype(jnp.float32), guide_scale=_GUIDE),
                weights_dtype=jnp.float32,
                num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
                support_start=data["draws"].support_start,
                support_end=data["draws"].support_end,
                k_b=_K,
            )
            return value

        assert jax.tree.structure(params) == jax.tree.structure(reference_params)
        assert np.array_equal(np.asarray(mine(params)), np.asarray(reference(params)))
        got, want = jax.grad(mine)(params), jax.grad(reference)(params)
        assert _tree_norm(jax.tree.map(lambda a, b: a - b, got, want)) == 0.0


def test_the_rollout_arm_carries_no_stop_gradient_of_its_own():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    rollout_paths = [
        ast.unparse(_function_node(source, name)) for name in ("rollout_arm_loss", "one_step_denoising_loss")
    ]
    assert "stop_gradient" not in rollout_paths[0], "R-B must not stop_gradient (plan §3a)"
    assert "stop_gradient" not in rollout_paths[1], "C0's stop-grads belong in its velocity builder, not its loss"
    # Dynamic attribute access is how an AST guard gets walked past (T3a S03b/S05/S16).
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in {"getattr", "eval", "exec"}:
            raise AssertionError(f"dynamic attribute access is forbidden here: {ast.unparse(node)}")
        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            raise AssertionError("__dict__ lookup is forbidden here")


# =============================================================================================
# 3. Arm isolation: the arm selects ONLY the loss.
# =============================================================================================


class _RecordingBatch(dict):
    """A batch that remembers which fields were read, so 'identical consumption' is measurable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads: list[str] = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


def test_both_arms_expose_the_identical_parameter_tree():
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    rules, mesh = _mesh_context()
    with rules, mesh:
        _, rollout_params = arms.build_arm("rollout", transformer, adapters)
        _, one_step_params = arms.build_arm("one_step", transformer, adapters)
    assert jax.tree.structure(rollout_params) == jax.tree.structure(one_step_params)
    for left, right in zip(jax.tree.leaves(rollout_params), jax.tree.leaves(one_step_params)):
        assert np.array_equal(np.asarray(left), np.asarray(right)), "the arms start from different weights"


def test_switching_arms_changes_the_loss_and_nothing_else():
    """The experimental control, as an oracle rather than an intention.

    Same batch fields consumed, same stream draws, same parameter tree; only the number differs. If
    an arm could touch the stream or read a different field, the M3 comparison would be confounded
    and no downstream metric would reveal it.
    """
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data, sigmas, timesteps = _fixture()
    context = _context(sigmas, timesteps, data["null_context"])
    rules, mesh = _mesh_context()
    reads = {}
    batches = {}
    values = {}
    with rules, mesh:
        for arm in arms.ARMS:
            loss_fn, params = arms.build_arm(arm, transformer, adapters)
            batch = _RecordingBatch(_batch(data))
            value, _ = loss_fn(
                params,
                batch,
                context,
                draws=data["draws"],
                seed=0,
                global_step=7,
                dropout_rng=jax.random.key(5),
            )
            reads[arm] = sorted(set(batch.reads))
            batches[arm] = list(batch.reads)
            values[arm] = float(value)
    assert reads["rollout"] == reads["one_step"], reads
    assert reads["rollout"] == ["actions", "z_i0", "z_video"]
    # Set equality, deliberately NOT the multiset: reading a field twice (e.g. once for the tensor
    # and once for its leading dimension) is inert, and pinning read COUNTS would fail on a harmless
    # refactor while catching nothing real. What matters is that neither arm reaches for a field the
    # other does not -- an arm quietly ignoring `actions`, say, would confound the whole comparison.
    assert set(batches["rollout"]) == set(batches["one_step"]), batches
    assert values["rollout"] != values["one_step"], "the arms must differ in the loss, or nothing is being compared"
    # The stream is computed outside the arms and neither draws its own randomness.
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "jax.random" not in source
    for name in ("rollout_arm_loss", "one_step_denoising_loss"):
        parameters = inspect.signature(getattr(arms, name)).parameters
        assert "draws" in parameters, name
        assert parameters["draws"].kind is inspect.Parameter.KEYWORD_ONLY
        assert not {"epsilon", "t_idx", "rng", "key"} & set(
            parameters
        ), f"{name} must take its randomness from the stream"


def test_the_stream_draw_is_identical_whichever_arm_will_consume_it():
    # T3b-1 guarantees this structurally; here it is checked at the seam the arms actually use.
    for global_step in (0, 7, 4001):
        draws = [
            stream._draw_step_stream(
                seed=0, global_step=global_step, num_steps=_STEPS, k_b=_K, shape=(_B, _C, _F, _H, _W)
            )
            for _ in arms.ARMS
        ]
        assert np.array_equal(np.asarray(draws[0].epsilon), np.asarray(draws[1].epsilon))
        assert int(draws[0].support_start) == int(draws[1].support_start)


def test_an_unknown_arm_is_refused_and_the_declared_set_is_the_two():
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    assert arms.ARMS == ("rollout", "one_step")
    with pytest.raises(ValueError, match="unknown arm"):
        arms.build_arm("corrective_ss", transformer, adapters)


def test_the_one_step_arm_refuses_to_invent_its_sigma_index():
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data, sigmas, timesteps = _fixture()
    rules, mesh = _mesh_context()
    with rules, mesh:
        loss_fn, params = arms.build_arm("one_step", transformer, adapters)
        with pytest.raises(ValueError, match="t_idx"):
            loss_fn(
                params,
                _batch(data),
                _context(sigmas, timesteps, data["null_context"]),
                draws=dataclasses.replace(data["draws"], t_idx=None),
                seed=0,
                global_step=7,
            )


# =============================================================================================
# 4. Accumulation equivalence, on real gradients.
# =============================================================================================


@pytest.mark.parametrize("arm", ["rollout", "one_step"])
def test_the_accumulated_update_equals_the_full_batch_update(arm):
    """Accumulation must be a memory strategy, never a different objective.

    Compared on the GRADIENT rather than on an end state: applying the T3a method note, the
    quantity in dispute is available exactly, so nothing is inferred from a trained-model proxy.
    Not bitwise -- summing two half-batch reductions is a different reduction ORDER from one
    full-batch reduction, so float32 disagrees in the last bits (T2's half-ulp precedent). The
    tolerance is on the RELATIVE L2 of the difference and is orders of magnitude tighter than any
    real accumulation bug, which changes the logical batch or drops a microbatch entirely.
    """
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data, sigmas, timesteps = _fixture()
    context = _context(sigmas, timesteps, data["null_context"])
    rules, mesh = _mesh_context()
    with rules, mesh:
        loss_fn, params = arms.build_arm(arm, transformer, adapters)

        def gradient_of(batch, part_draws):
            def value(p):
                return loss_fn(
                    p, batch, context, draws=part_draws, seed=0, global_step=7, dropout_rng=jax.random.key(5)
                )[0]

            return jax.grad(value)(params)

        full = gradient_of(_batch(data), data["draws"])
        parts = stream.split_batch(_batch(data), 2)
        part_draws = stream._split_draws(data["draws"], 2)
        accumulated = jax.tree.map(
            lambda *leaves: sum(leaves) / len(leaves),
            *[gradient_of(part, drawn) for part, drawn in zip(parts, part_draws)],
        )

    relative = _tree_norm(jax.tree.map(lambda a, b: a - b, full, accumulated)) / _tree_norm(full)
    assert relative < 1e-5, f"{arm}: accumulated update differs from the full-batch update by {relative:.3e}"


def test_the_arms_module_reads_no_config():
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id not in {"config", "cfg", "scheduler"}, ast.unparse(node)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "getattr":
            assert len(node.args) < 3, f"three-argument getattr is forbidden (issue #11): {ast.unparse(node)}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            assert not names & {"config", "cfg", "scheduler"}, node.name


def test_the_arms_compose_the_committed_kernels_rather_than_reimplementing_them():
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and "pos_rollout" in (node.module or "")
        for alias in node.names
    }
    assert {
        "rollout_endpoint_loss",
        "build_cfg_velocity_fn",
        "masked_velocity_mse",
        "build_noisy_pinned_latents",
    } <= imported
    assert arms.rollout_endpoint_loss is rollout_endpoint_loss
    assert arms.build_cfg_velocity_fn is build_cfg_velocity_fn
    for forbidden in ("def rollout_endpoint_loss", "def masked_velocity_mse", "sigmas[index + 1]", "jax.lax.scan"):
        assert forbidden not in source, forbidden
