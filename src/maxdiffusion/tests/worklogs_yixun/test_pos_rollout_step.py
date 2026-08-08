"""exp_06 `rollout_adapter` — T3a `cfg-rollout-step`: the §3a gradient contract, proven.

This is the round the experiment's correctness hinges on. T1 gave exp_06 the shared sampler step, T2
the endpoint kernel that unrolls it; T3a supplies the velocity the unroll differentiates, and must
show that the gradients it produces are the ones deployment's math implies — not merely that a
number comes out.

Everything here runs against the REAL architecture: a real ``WanModel`` and a real
``NNXWanSideAdapterStack`` in its ``pre_context`` configuration (the fixture construction is exp_05's
``test_pos_context_trainer._tiny_pre_context``, re-tuned to float32 so the parity claim sits inside
T2's proven-equivalent domain). A stand-in forward could satisfy every assertion below while the
frozen split and the block-0 stop-grad — both properties of the real module tree — went untested.

**The three clauses and how each is proven.**

* **(i) adapter parameters only.** The gradient tree is compared leaf-for-leaf against the adapter's
  own parameter tree, the frozen backbone's leaves are shown bit-unchanged across a gradient step,
  and — structurally — the backbone is shown never to be an argument of anything differentiated.
* **(ii) both branches differentiate the rollout state.** Three oracles, because each reaches
  somewhere the others do not:
  - **The isolated-unconditional finite difference** — the plan's §3a obligation, discharged
    directly. Differencing the WHOLE loss cannot resolve the unconditional branch's state
    dependence in float32: that term is ~1-7% of the gradient and the forward noise swamps it. But
    the term does not have to be inferred from the whole loss, because it is available in closed
    form — the unconditional branch contributes exactly
    ``(1 - w) * (sigma[s+2] - sigma[s+1]) * v_unc(z_1(theta))`` to the second step. Central-
    differencing THAT scalar and comparing it against ``<grad_full - grad_cut_uncond, d>`` gives an
    autodiff-independent measurement of precisely the disputed quantity: 110-1039x discrimination.
  - **A whole-loss finite-difference oracle** discriminates *inter-step path present* from
    *inter-step path truncated*: probing along the direction in which those two hypotheses disagree,
    the central difference lands 5-16x closer to the module's gradient than to the truncated one.
  - **An exact gradient-contrast oracle** (no numerical floor at all) covers the conditional branch
    and, critically, kills truncations obfuscated past every AST guard (T2's N08 lesson) — a
    mutated module coincides with the truncated reference, and the margin vanishes.
  The design measurements behind the constants — the h-sweep and the per-seed discrimination tables
  — are recorded in the round's report.

**On precision.** All of the above runs in float32. ``jax.enable_x64()`` IS available in this JAX
(0.10.2) as a scoped, restoring context manager — an earlier note in this campaign said otherwise,
having checked only the removed ``jax.experimental.enable_x64``; the correction is recorded here so
a later round does not inherit a false constraint. It is not used: the isolated oracle above makes
float64 unnecessary, and flipping global dtype promotion mid-suite perturbs tracing and caching for
everything that runs after it.
* **(iii) the architecture's block-0 stop-grad stays.** A tripwire on the inherited module, because
  "held fixed" is a claim about code someone could helpfully "fix".

**Parity** with the deployed evaluator is proven the way T1 established: a verbatim copy of
``generate_wan_side_adapter._rollout_sample``'s loop body, bitwise-compared over the full 25-step
production rollout at both CFG branches, with an AST drift tripwire binding the copy to the file it
came from. The settled evaluator is NOT rewired (standing ruling, plan §5-5).
"""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxdiffusion import pos_rollout_step as step_module
from maxdiffusion.models.wan.overfit100_sampling import overfit100_sampler_grid, overfit100_sampler_step
from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    apply_first_frame_pin,
    wan_action_adapter_forward,
)
from maxdiffusion.pos_rollout_step import build_cfg_velocity_fn, cfg_rollout, combine_cfg

_MODULE_PATH = Path(step_module.__file__).resolve()
_PACKAGE_ROOT = _MODULE_PATH.parent
_EVALUATOR_PATH = _PACKAGE_ROOT / "generate_wan_side_adapter.py"
_TRAINER_PATH = _PACKAGE_ROOT / "trainers" / "wan_ti2v_side_adapter_trainer.py"
_SIDE_ADAPTER_PATH = _PACKAGE_ROOT / "models" / "wan" / "side_adapter_wan.py"
_THIS_PATH = Path(__file__).resolve()

# Production rollout grid and the deployed guide scale (base_wan_5b_side_adapter.yml).
_STEPS, _SHIFT, _SIGMA_MIN, _SIGMA_MAX, _NUM_TRAIN_TIMESTEPS = 25, 5.0, 0.0, 1.0, 1000
_GUIDE = 5.0
# Tiny real-architecture geometry (exp_05's fixture), at float32 for the bitwise parity claim.
_C, _TEXT, _F, _H, _W, _L_POS, _B = 4, 32, 2, 4, 6, 4, 2

# The finite-difference oracle's operating point, chosen by the sweep in the round's report:
# the production grid, a 2-step window at index 12, and a relative step of 3e-3.
_FD_START, _FD_K, _FD_H_REL = 12, 2, 3e-3
_FD_SEEDS = (5, 17, 29, 41, 53)
# The ISOLATED unconditional oracle tolerates a much smaller step, because the quantity it
# differences is the term itself rather than a loss the term is a percent of.
_ISO_H_REL = 1e-3

# Thresholds. **These are fitted guardrails, not independent validation margins**: the operating
# point (start index, horizon, step size) and the constants below were chosen from measurements on
# these same seeds, so "headroom" here means "distance from the worst seed we looked at", not
# out-of-sample confidence. That is acceptable because the load-bearing claim is carried by
# `test_the_isolated_unconditional_term_matches_its_own_finite_difference`, which is a DIRECT
# measurement of the quantity in dispute and clears its threshold by ~5x rather than by inches.
# Worst-seed measurements, in order: 5.03 / 0.341 / 0.023 / 0.051 / 109.9.
_MIN_FD_RATIO, _MIN_REL_CARRY, _MIN_REL_UNCOND, _MIN_REL_COND = 3.0, 0.15, 0.010, 0.020
_MIN_ISO_RATIO = 20.0


def _requires_backend():
    pytest.importorskip("torch")
    pytest.importorskip("aqt")


def _mesh_context():
    from flax.linen import partitioning as nn_partitioning

    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    set_mesh = getattr(jax, "set_mesh", None)
    return nn_partitioning.axis_rules(()), (set_mesh(mesh) if set_mesh is not None else mesh)


@functools.lru_cache(maxsize=1)
def _tiny_cfg_stack():
    """A real WAN transformer + real pre_context adapter stack, float32, small enough for CPU.

    Construction copied from exp_05's ``test_pos_context_trainer._tiny_pre_context`` so the frozen
    split and gradient claims are made against the deployed architecture. Only the dtypes differ
    (float32 rather than bfloat16): the parity claim is asserted bitwise and T2's precondition puts
    the loss inputs in float32.
    """
    from flax import nnx

    from maxdiffusion.models.wan.side_adapter_wan import NNXWanSideAdapterStack
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel

    rules, mesh = _mesh_context()
    with rules, mesh:
        transformer = WanModel(
            rngs=nnx.Rngs(jax.random.key(0)),
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=_C,
            out_channels=_C,
            text_dim=_TEXT,
            freq_dim=16,
            ffn_dim=32,
            num_layers=1,
            attention="dot_product",
            rope_max_seq_len=64,
            scan_layers=False,
            dtype=jnp.float32,
            weights_dtype=jnp.float32,
        )
        adapters = NNXWanSideAdapterStack(
            rngs=nnx.Rngs(jax.random.key(1)),
            num_layers=1,
            model_dim=16,
            text_dim=_TEXT,
            action_adapter_type="pre_context",
            action_dim=7,
            action_len=4,
            action_repr="delta",
            action_tokens=4,
            action_hidden=16,
            action_heads=2,
            side_adapter_layers="0",
            side_adapter_hidden=16,
            side_adapter_heads=2,
            pre_context_tokens=_L_POS,
            pre_context_heads=2,
            dtype=jnp.float32,
            weights_dtype=jnp.float32,
        )
    return transformer, adapters


def _grid():
    return overfit100_sampler_grid(
        num_inference_steps=_STEPS,
        flow_shift=_SHIFT,
        sigma_min=_SIGMA_MIN,
        sigma_max=_SIGMA_MAX,
        num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
    )


def _inputs(seed):
    keys = jax.random.split(jax.random.key(seed), 5)
    return {
        "z": jax.random.normal(keys[0], (_B, _C, _F, _H, _W), jnp.float32),
        "z_i0": jax.random.normal(keys[1], (_B, _C, 1, _H, _W), jnp.float32),
        "null_context": jax.random.normal(keys[2], (_B, 7, _TEXT), jnp.float32),
        "actions": jax.random.normal(keys[3], (_B, 4, 7), jnp.float32),
        "probe": jax.random.normal(keys[4], (_B, _C, _F - 1, _H, _W), jnp.float32),
    }


def _tree_norm(tree):
    return float(jnp.sqrt(sum(jnp.sum(leaf.astype(jnp.float32) ** 2) for leaf in jax.tree.leaves(tree))))


def _tree_dot(left, right):
    return float(
        sum(
            jnp.sum(a.astype(jnp.float32) * b.astype(jnp.float32))
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
        )
    )


def _function_node(source: str, name: str, *, within: str | None = None) -> ast.FunctionDef:
    scope = ast.parse(source)
    if within is not None:
        scope = _function_node(source, within)
    for node in ast.walk(scope):
        if isinstance(node, ast.FunctionDef) and node.name == name and node is not scope:
            return node
    raise AssertionError(f"{name} not found{'' if within is None else f' inside {within}'}")


# =============================================================================================
# The verbatim reference: the loop body of generate_wan_side_adapter._rollout_sample, copied from
# exp_06's own tree (src/maxdiffusion/generate_wan_side_adapter.py:136-158). Do not refactor it --
# its value is that it is the deployed code. The eval module cannot be imported in a test venv
# (it pulls `transformers`), so the copy is bound to its source by an AST drift tripwire.
# =============================================================================================


def _deployed_rollout_reference(*, transformer, adapters, sigmas, timesteps, null_context, z_i0, actions, config, z):
    b, _, f_lat, h_lat, w_lat = z.shape

    def _body(i, current):
        step_t = jnp.broadcast_to(timesteps[i], (b,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
        v_cond = wan_action_adapter_forward(
            transformer,
            adapters,
            hidden_states=current,
            timestep=timestep_2d,
            encoder_hidden_states=null_context,
            actions=actions,
            deterministic=True,
        )
        if abs(config.side_adapter_guide_scale - 1.0) > 1e-6:
            v_uncond = transformer(
                hidden_states=current,
                timestep=timestep_2d,
                encoder_hidden_states=null_context,
                deterministic=True,
            )
            v = v_uncond + config.side_adapter_guide_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return apply_first_frame_pin(current + (sigmas[i + 1] - sigmas[i]).astype(current.dtype) * v, z_i0)

    return jax.lax.fori_loop(0, _STEPS, _body, z)


class _Config:
    def __init__(self, guide_scale):
        self.side_adapter_guide_scale = guide_scale


# =============================================================================================
# 1. Deployed parity.
# =============================================================================================


@pytest.mark.parametrize("guide_scale", [1.0, _GUIDE])
def test_the_cfg_rollout_reproduces_the_deployed_evaluator_bitwise(guide_scale):
    """The full 25-step production rollout, at both CFG branches, on the real architecture.

    Exact equality, not ``allclose``: the evaluator's bitwise reproducibility is a measured asset of
    this campaign, and every downstream claim about "the training trajectory is the evaluation
    trajectory" reduces to this comparison.
    """
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data = _inputs(7)
    sigmas, timesteps = _grid()
    rules, mesh = _mesh_context()
    with rules, mesh:
        z0 = apply_first_frame_pin(data["z"], data["z_i0"])
        want = _deployed_rollout_reference(
            transformer=transformer,
            adapters=adapters,
            sigmas=sigmas,
            timesteps=timesteps,
            null_context=data["null_context"],
            z_i0=data["z_i0"],
            actions=data["actions"],
            config=_Config(guide_scale),
            z=z0,
        )
        make_velocity_fn, params = build_cfg_velocity_fn(transformer, adapters)
        got = cfg_rollout(
            z0,
            velocity_fn=make_velocity_fn(params, actions=data["actions"], guide_scale=guide_scale),
            sigmas=sigmas,
            timesteps=timesteps,
            context=data["null_context"],
            z_i0=data["z_i0"],
            start=0,
            num_steps=_STEPS,
        )
    assert got.dtype == want.dtype == jnp.float32
    assert np.array_equal(np.asarray(got), np.asarray(want))


def test_the_deployed_evaluator_body_has_not_drifted_from_the_copy_pinned_here():
    # The parity test compares against a VERBATIM copy of the evaluator's loop body; that copy is
    # only evidence while it still matches the file it came from. Structural, so formatting is free.
    deployed = _function_node(_EVALUATOR_PATH.read_text(encoding="utf-8"), "_body", within="_rollout_sample")
    reference = _function_node(_THIS_PATH.read_text(encoding="utf-8"), "_body", within="_deployed_rollout_reference")
    assert ast.unparse(deployed) == ast.unparse(reference)


def test_the_settled_evaluator_was_not_rewired():
    # Standing Planner ruling (plan §5-5): parity is discharged BY TEST, so the settled evaluator
    # keeps its inline sampler and this round leaves it alone.
    source = _EVALUATOR_PATH.read_text(encoding="utf-8")
    assert "pos_rollout_step" not in source
    assert "cfg_rollout" not in source
    assert "sigmas[i + 1] - sigmas[i]" in source, "the evaluator's inline step is what parity is measured against"


# =============================================================================================
# 2. The CFG combination itself.
# =============================================================================================


def test_the_cfg_combination_is_the_deployed_form_order_and_sign():
    key_a, key_b = jax.random.split(jax.random.key(3))
    v_uncond = jax.random.normal(key_a, (2, 3), jnp.float32)
    v_cond = jax.random.normal(key_b, (2, 3), jnp.float32)
    got = combine_cfg(v_uncond, v_cond, _GUIDE)
    assert np.array_equal(np.asarray(got), np.asarray(v_uncond + _GUIDE * (v_cond - v_uncond)))
    # The base is the UNCONDITIONAL velocity and the guided direction is cond-minus-uncond; swapping
    # the roles is a different operator, not a re-parametrisation.
    assert not np.allclose(np.asarray(got), np.asarray(combine_cfg(v_cond, v_uncond, _GUIDE)))
    # w = 0 recovers the unconditional velocity exactly. w = 1 recovers the conditional one only up
    # to float32 rounding -- `v_u + 1*(v_c - v_u)` is not bitwise `v_c` -- and THAT is precisely why
    # the deployed evaluator branches on |w - 1| <= 1e-6 instead of always computing the
    # combination. The module keeps that branch, and the test below shows it returns v_cond exactly.
    assert np.array_equal(np.asarray(combine_cfg(v_uncond, v_cond, 0.0)), np.asarray(v_uncond))
    assert np.allclose(np.asarray(combine_cfg(v_uncond, v_cond, 1.0)), np.asarray(v_cond), rtol=1e-6)
    assert not np.array_equal(np.asarray(combine_cfg(v_uncond, v_cond, 1.0)), np.asarray(v_cond))
    # The algebraically-equal `w*v_cond + (1-w)*v_uncond` is NOT bitwise equal in float32, which is
    # why the deployed form is written out rather than "simplified".
    assert not np.array_equal(np.asarray(got), np.asarray(_GUIDE * v_cond + (1.0 - _GUIDE) * v_uncond))


def test_the_identity_guide_scale_returns_the_conditional_velocity_exactly():
    # The deployed evaluator skips the second forward when |w - 1| <= 1e-6; the module keeps that
    # branch so parity holds at w = 1, and the tolerance is the same constant.
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data = _inputs(9)
    sigmas, timesteps = _grid()
    rules, mesh = _mesh_context()
    with rules, mesh:
        make_velocity_fn, params = build_cfg_velocity_fn(transformer, adapters)
        timestep_2d = _build_per_token_timestep(jnp.broadcast_to(timesteps[3], (_B,)), _F, _H, _W, n_hist=1)
        guided = make_velocity_fn(params, actions=data["actions"], guide_scale=1.0)(
            data["z"], timestep_2d, data["null_context"]
        )
        expected = wan_action_adapter_forward(
            transformer,
            adapters,
            hidden_states=data["z"],
            timestep=timestep_2d,
            encoder_hidden_states=data["null_context"],
            actions=data["actions"],
            deterministic=True,
        )
    assert np.array_equal(np.asarray(guided), np.asarray(expected))
    assert step_module.CFG_IDENTITY_TOLERANCE == 1e-6


# =============================================================================================
# 3. Clause (i): the gradient tree contains adapter parameters ONLY.
# =============================================================================================


def _scalar_loss(params, make_velocity_fn, data, sigmas, timesteps, *, start=_FD_START, steps=_FD_K, guide=_GUIDE):
    out = cfg_rollout(
        apply_first_frame_pin(data["z"], data["z_i0"]),
        velocity_fn=make_velocity_fn(params, actions=data["actions"], guide_scale=guide),
        sigmas=sigmas,
        timesteps=timesteps,
        context=data["null_context"],
        z_i0=data["z_i0"],
        start=start,
        num_steps=steps,
    )
    return jnp.sum(data["probe"] * out[:, :, 1:])


def test_only_adapter_parameters_appear_in_the_gradient_tree():
    """The freeze contract, on the real module tree.

    The backbone is closed over rather than passed, so this is not "the filter worked" -- there is no
    combined tree that a filter could have got wrong. What is checked is that the seam holds: the
    gradient has exactly the adapter's structure, every frozen leaf is bit-unchanged, and the
    gradient is not trivially zero (a freeze test that differentiates nothing proves nothing).
    """
    _requires_backend()
    from flax import nnx

    transformer, adapters = _tiny_cfg_stack()
    data = _inputs(11)
    sigmas, timesteps = _grid()
    rules, mesh = _mesh_context()
    with rules, mesh:
        frozen_before = [
            np.asarray(leaf, np.float32).copy() for leaf in jax.tree.leaves(nnx.split(transformer, nnx.Param, ...)[1])
        ]
        adapter_params = nnx.split(adapters, nnx.Param, ...)[1]
        make_velocity_fn, params = build_cfg_velocity_fn(transformer, adapters)
        grads = jax.grad(_scalar_loss)(params, make_velocity_fn, data, sigmas, timesteps)
        frozen_after = jax.tree.leaves(nnx.split(transformer, nnx.Param, ...)[1])

    assert frozen_before, "the fixture has no frozen side, so this proves nothing"
    assert jax.tree.structure(grads) == jax.tree.structure(adapter_params)
    assert len(jax.tree.leaves(grads)) == len(jax.tree.leaves(adapter_params))
    for before, after in zip(frozen_before, frozen_after):
        assert np.array_equal(before, np.asarray(after, np.float32))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(grads))
    assert _tree_norm(grads) > 0.0


def test_the_frozen_backbone_is_never_an_argument_of_a_differentiated_function():
    """Structural clause (i): the closure seam, checked in the source rather than inferred.

    ``build_cfg_velocity_fn`` binds the backbone's split state to a local name; the function it
    returns takes ``params`` and never the backbone. If the backbone were ever threaded through an
    argument, a caller could differentiate it -- which is the failure mode the closure prevents.
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    builder = _function_node(source, "build_cfg_velocity_fn")
    splits = [
        node for node in ast.walk(builder) if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "split"
    ]
    assert splits, "the backbone must be split (and captured) inside the builder"
    inner = _function_node(source, "make_velocity_fn", within="build_cfg_velocity_fn")
    positional = [argument.arg for argument in inner.args.args]
    assert positional == ["params"], positional
    keyword_only = {argument.arg for argument in inner.args.kwonlyargs}
    assert not keyword_only & {"transformer", "adapters", "frozen", "model"}
    # ...and no function in the module takes the backbone as a differentiable-looking argument.
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "build_cfg_velocity_fn":
            names = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            assert "transformer" not in names, f"{node.name} takes the frozen backbone as an argument"


# =============================================================================================
# 4. Clause (ii): BOTH branches differentiate the rollout state.
# =============================================================================================


def _truncated_velocity_builder(transformer, adapters, *, cut_uncond=False, cut_cond=False, cut_v_uncond=False):
    """The module's velocity, deliberately mis-built -- the hypotheses the oracles discriminate.

    ``cut_uncond``/``cut_cond`` stop-gradient the rollout state entering one branch;
    ``cut_v_uncond`` stop-gradients the unconditional velocity itself. ``cut_uncond +
    cut_v_uncond`` together are exactly the one-step trainer's pattern at
    ``wan_ti2v_side_adapter_trainer.py:180-187``.
    """
    from flax import nnx

    adapter_graphdef, adapter_params, adapter_rest = nnx.split(adapters, nnx.Param, ...)
    frozen = nnx.split(transformer, nnx.Param, ...)

    def make_velocity_fn(params, *, actions, guide_scale):
        model = nnx.merge(*frozen)
        adapter = nnx.merge(adapter_graphdef, params, adapter_rest)

        def velocity_fn(hidden_states, timestep, encoder_hidden_states):
            z_cond = jax.lax.stop_gradient(hidden_states) if cut_cond else hidden_states
            z_uncond = jax.lax.stop_gradient(hidden_states) if cut_uncond else hidden_states
            v_cond = wan_action_adapter_forward(
                model,
                adapter,
                hidden_states=z_cond,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                actions=actions,
                deterministic=True,
            )
            v_uncond = model(
                hidden_states=z_uncond,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=True,
            )
            if cut_v_uncond:
                v_uncond = jax.lax.stop_gradient(v_uncond)
            return combine_cfg(v_uncond, v_cond, guide_scale)

        return make_velocity_fn_inner(velocity_fn)

    def make_velocity_fn_inner(velocity_fn):
        return velocity_fn

    return make_velocity_fn, adapter_params


def _rollout_with_cut_carry(z, *, velocity_fn, sigmas, timesteps, context, z_i0, start, num_steps):
    """The 'inter-step path truncated' hypothesis: each step differentiates its own forward only."""
    return jax.lax.fori_loop(
        start,
        start + num_steps,
        lambda index, current: overfit100_sampler_step(
            jax.lax.stop_gradient(current),
            index,
            velocity_fn=velocity_fn,
            sigmas=sigmas,
            timesteps=timesteps,
            context=context,
            z_i0=z_i0,
        ),
        z,
    )


def _isolated_unconditional_term(params, make_velocity_fn, transformer, data, sigmas, timesteps, *, start=_FD_START):
    """The second-step contribution of the UNCONDITIONAL branch, in closed form.

    One rollout step produces ``z_1(theta)``. The next step's velocity is
    ``v = (1 - w) * v_unc(z_1) + w * v_cond(z_1, theta)``, and the Euler update multiplies it by
    ``sigma[s+2] - sigma[s+1]`` before the frame-0 pin. So the unconditional branch's entire
    contribution to the endpoint -- the piece a ``stop_gradient`` on its ``z`` input deletes -- is

        ``(1 - w) * (sigma[s+2] - sigma[s+1]) * v_unc(z_1(theta))``

    evaluated on the non-frame-0 slice the loss reads. ``theta`` enters ONLY through ``z_1``: the
    backbone is frozen and its own parameters are not differentiated. Differencing this scalar
    therefore measures the disputed term itself instead of hunting for it inside a loss it is a few
    percent of -- which is what makes an autodiff-independent proof possible in float32.
    """
    z0 = apply_first_frame_pin(data["z"], data["z_i0"])
    z1 = overfit100_sampler_step(
        z0,
        start,
        velocity_fn=make_velocity_fn(params, actions=data["actions"], guide_scale=_GUIDE),
        sigmas=sigmas,
        timesteps=timesteps,
        context=data["null_context"],
        z_i0=data["z_i0"],
    )
    timestep_2d = _build_per_token_timestep(
        jnp.broadcast_to(timesteps[start + 1], (z0.shape[0],)), _F, _H, _W, n_hist=1
    )
    v_uncond = transformer(
        hidden_states=z1,
        timestep=timestep_2d,
        encoder_hidden_states=data["null_context"],
        deterministic=True,
    )
    coefficient = (1.0 - _GUIDE) * (sigmas[start + 2] - sigmas[start + 1]).astype(jnp.float32)
    return jnp.sum(data["probe"] * (coefficient * v_uncond)[:, :, 1:])


@functools.lru_cache(maxsize=1)
def _discrimination_table():
    """One expensive pass: per-seed FD and exact-contrast measurements for clause (ii)."""
    from flax import nnx

    del nnx
    transformer, adapters = _tiny_cfg_stack()
    sigmas, timesteps = _grid()
    rules, mesh = _mesh_context()
    rows = []
    with rules, mesh:
        make_velocity_fn, params = build_cfg_velocity_fn(transformer, adapters)
        cuts = {
            "uncond": _truncated_velocity_builder(transformer, adapters, cut_uncond=True)[0],
            "cond": _truncated_velocity_builder(transformer, adapters, cut_cond=True)[0],
            "one_step_pattern": _truncated_velocity_builder(transformer, adapters, cut_uncond=True, cut_v_uncond=True)[
                0
            ],
        }
        for seed in _FD_SEEDS:
            data = _inputs(seed)

            def loss(p, maker=make_velocity_fn, roll=cfg_rollout, d=data):
                out = roll(
                    apply_first_frame_pin(d["z"], d["z_i0"]),
                    velocity_fn=maker(p, actions=d["actions"], guide_scale=_GUIDE),
                    sigmas=sigmas,
                    timesteps=timesteps,
                    context=d["null_context"],
                    z_i0=d["z_i0"],
                    start=_FD_START,
                    num_steps=_FD_K,
                )
                return jnp.sum(d["probe"] * out[:, :, 1:])

            grad_full = jax.grad(loss)(params)
            grad_carry = jax.grad(functools.partial(loss, roll=_rollout_with_cut_carry))(params)
            grad_cut = {name: jax.grad(functools.partial(loss, maker=maker))(params) for name, maker in cuts.items()}
            relative = {
                name: _tree_norm(jax.tree.map(lambda a, b: a - b, grad_full, gradient)) / _tree_norm(grad_full)
                for name, gradient in grad_cut.items()
            }
            relative["carry"] = _tree_norm(jax.tree.map(lambda a, b: a - b, grad_full, grad_carry)) / _tree_norm(
                grad_full
            )

            # --- The isolated-unconditional oracle (the plan's §3a obligation, discharged directly).
            # `grad_full - grad_cut_uncond` is, in exact arithmetic, the gradient of the closed-form
            # term in `_isolated_unconditional_term`. Probe along that difference and central-
            # difference the term itself: no other gradient path contributes, so the float32 noise
            # that swamps a whole-loss difference is simply not in the measurement.
            isolated_difference = jax.tree.map(lambda a, b: a - b, grad_full, grad_cut["uncond"])
            isolated_scale = _tree_norm(isolated_difference)
            isolated_direction = jax.tree.map(lambda leaf: leaf / isolated_scale, isolated_difference)

            def isolated(p, d=data):
                return _isolated_unconditional_term(p, make_velocity_fn, transformer, d, sigmas, timesteps)

            iso_h = _ISO_H_REL * _tree_norm(params) / _tree_norm(isolated_direction)
            iso_fd = (
                float(isolated(jax.tree.map(lambda p, q: p + iso_h * q, params, isolated_direction)))
                - float(isolated(jax.tree.map(lambda p, q: p - iso_h * q, params, isolated_direction)))
            ) / (2.0 * iso_h)
            iso_target = _tree_dot(isolated_difference, isolated_direction)
            iso_autodiff = _tree_dot(jax.grad(isolated)(params), isolated_direction)

            # Probe along the direction in which the two hypotheses DISAGREE: a random direction
            # projects that disagreement unreliably (it can be near-orthogonal), which is what makes
            # a random-direction FD look seed-lucky.
            difference = jax.tree.map(lambda a, b: a - b, grad_full, grad_carry)
            scale = _tree_norm(difference)
            direction = jax.tree.map(lambda leaf: leaf / scale, difference)
            h = _FD_H_REL * _tree_norm(params) / _tree_norm(direction)
            plus = float(loss(jax.tree.map(lambda p, q: p + h * q, params, direction)))
            minus = float(loss(jax.tree.map(lambda p, q: p - h * q, params, direction)))
            finite_difference = (plus - minus) / (2.0 * h)
            projected_full = _tree_dot(grad_full, direction)
            projected_carry = _tree_dot(grad_carry, direction)
            rows.append(
                {
                    "seed": seed,
                    "fd": finite_difference,
                    "full": projected_full,
                    "carry": projected_carry,
                    "err_full": abs(finite_difference - projected_full),
                    "err_carry": abs(finite_difference - projected_carry),
                    "gap": abs(projected_full - projected_carry),
                    "relative": relative,
                    "iso_fd": iso_fd,
                    "iso_target": iso_target,
                    "iso_autodiff": iso_autodiff,
                    "iso_err": abs(iso_fd - iso_target),
                    "iso_ratio": abs(iso_target) / max(abs(iso_fd - iso_target), 1e-30),
                }
            )
    return tuple(rows)


def test_the_isolated_unconditional_term_matches_its_own_finite_difference():
    """**The §3a obligation, discharged directly: the unconditional branch's state dependence.**

    The gap between the module's gradient and a stop-gradient-on-the-unconditional-branch variant is
    claimed to BE the derivative of ``(1 - w) * (sigma[s+2] - sigma[s+1]) * v_unc(z_1(theta))``. That
    claim is checked two ways here: against autodiff on the closed-form term (an algebraic identity
    check -- if the term were mis-derived, this fails), and against a CENTRAL FINITE DIFFERENCE of
    the same term, which uses no autodiff at all and is therefore the actual proof.

    This is the oracle that reaches what a whole-loss difference cannot: the term is a few percent of
    the loss gradient and float32 forward noise hides it there, but differencing the term on its own
    resolves it to better than a percent.
    """
    _requires_backend()
    for row in _discrimination_table():
        # The algebraic identity: grad_full - grad_cut_uncond IS the gradient of the isolated term.
        assert np.allclose(row["iso_autodiff"], row["iso_target"], rtol=1e-4), row
        # The autodiff-independent proof: the finite difference of that term reproduces the gap.
        assert row["iso_ratio"] >= _MIN_ISO_RATIO, row
        assert abs(row["iso_target"]) > 0.0


def test_the_finite_difference_selects_the_untruncated_gradient_on_every_seed():
    """THE finite-difference oracle: autodiff-independent, and a discriminator by construction.

    Its job is not to reproduce the gradient to 1e-6 -- in float32, on a real transformer, it cannot.
    Its job is to decide between two hypotheses: the inter-step path is differentiated, or it is not.
    So the perturbation runs along the direction in which those hypotheses disagree, and the
    assertion is that the central difference lands nearer the module's gradient than the truncated
    one, by a margin, on every seed.
    """
    _requires_backend()
    for row in _discrimination_table():
        assert row["err_full"] < row["err_carry"], row
        assert row["gap"] / row["err_full"] >= _MIN_FD_RATIO, row


def test_the_state_dependence_of_each_cfg_branch_is_present_in_the_gradient():
    """The exact contrast oracle -- no numerical floor, which is why it reaches where the FD cannot.

    Stopping the gradient on the rollout state entering EITHER branch changes the gradient by a
    measurable relative margin. Because this compares the module's own gradient against a deliberate
    truncation, it also kills a truncation obfuscated past an AST guard: if the module were already
    truncated, the two would coincide and the margin would vanish.
    """
    _requires_backend()
    for row in _discrimination_table():
        assert row["relative"]["carry"] >= _MIN_REL_CARRY, row
        assert row["relative"]["uncond"] >= _MIN_REL_UNCOND, row
        assert row["relative"]["cond"] >= _MIN_REL_COND, row
        assert row["relative"]["one_step_pattern"] >= _MIN_REL_UNCOND, row


def test_the_module_applies_no_stop_gradient_anywhere():
    """The cheap guard. It is NOT the load-bearing one -- see the note about obfuscation below.

    Checked on the AST, not the raw source, because the module's docstring legitimately discusses the
    stop-gradient it does not perform. An AST guard is evadable by construction (``getattr(jax.lax,
    "stop_" + "gradient")`` never spells the name), which is why the exact-contrast oracle above is
    the real gate; this test additionally forbids the dynamic-attribute route so the two guards do
    not share a blind spot.
    """
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "stop_gradient":
            raise AssertionError(f"clause (ii) forbids a stop_gradient here: {ast.unparse(node)}")
        if isinstance(node, ast.Name) and node.id == "stop_gradient":
            raise AssertionError("clause (ii) forbids a stop_gradient here")
        # No dynamic attribute lookup at all: this module has no legitimate use for one, and it is
        # the documented way an AST guard gets walked past (T2's N08).
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in {"getattr", "eval", "exec"}:
            raise AssertionError(f"dynamic attribute access is forbidden here: {ast.unparse(node)}")


def test_the_one_step_trainers_stop_gradient_pattern_is_the_one_not_copied():
    """T1-6, kept as a tripwire so the module docstring's claim stays checkable.

    The deployed one-step trainer stop-gradients both the noisy latent entering the unconditional
    branch and the unconditional velocity itself. That is correct there -- its ``z_t`` is
    parameter-independent -- and gradient-truncating for a rollout. If the inherited site ever
    changes, this round's rationale needs re-reading, so it is pinned rather than described.
    """
    body = _function_node(_TRAINER_PATH.read_text(encoding="utf-8"), "_denoising_loss")
    unparsed = ast.unparse(body)
    assert "jax.lax.stop_gradient(z_t)" in unparsed
    assert "v_uncond = jax.lax.stop_gradient(v_uncond)" in unparsed


# =============================================================================================
# 5. Clause (iii): the architecture's block-0 stop-grad remains.
# =============================================================================================


def test_the_architectures_block_zero_stop_gradient_is_still_in_place():
    """Architecture held fixed FOR ISOLATION of the objective variable -- so this is not ours to fix.

    The pre_context head is conditioned on first-block features that arrive detached. Removing that
    stop-grad would change the architecture mid-experiment and confound the objective comparison.
    """
    source = _SIDE_ADAPTER_PATH.read_text(encoding="utf-8")
    node = _function_node(source, "_first_block_self_attention_features")
    returns = [child for child in ast.walk(node) if isinstance(child, ast.Return)]
    assert len(returns) == 1
    assert ast.unparse(returns[0]) == "return jax.lax.stop_gradient(features)"


def test_the_module_routes_through_the_shared_adapter_forward():
    # Clause (iii) only holds because the velocity goes through the deployed adapter path; a
    # hand-rolled forward could bypass `_first_block_self_attention_features` entirely.
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "wan_action_adapter_forward" in source
    assert "_first_block_self_attention_features" not in source
    assert "predict_pre_context" not in source


# =============================================================================================
# 6. Composition: the seams T1/T2/T3a were split along still fit.
# =============================================================================================


def test_the_velocity_fn_plugs_into_the_t2_endpoint_kernel_and_is_differentiable_there():
    _requires_backend()
    from maxdiffusion.pos_rollout_losses import rollout_endpoint_loss

    transformer, adapters = _tiny_cfg_stack()
    data = _inputs(13)
    sigmas, timesteps = _grid()
    rules, mesh = _mesh_context()
    with rules, mesh:
        make_velocity_fn, params = build_cfg_velocity_fn(transformer, adapters)
        eps = jax.random.normal(jax.random.key(23), (_B, _C, _F, _H, _W), jnp.float32)

        def loss(p):
            value, _ = rollout_endpoint_loss(
                z_video_f32=data["z"],
                z_i0_f32=data["z_i0"],
                eps_f32=eps,
                sigmas=sigmas,
                timesteps=timesteps,
                context=data["null_context"],
                velocity_fn=make_velocity_fn(p, actions=data["actions"], guide_scale=_GUIDE),
                weights_dtype=jnp.float32,
                num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
                support_start=jnp.asarray(3, jnp.int32),
                support_end=jnp.asarray(5, jnp.int32),
                k_b=2,
            )
            return value

        value = float(loss(params))
        grads = jax.grad(loss)(params)
    assert np.isfinite(value)
    assert jax.tree.structure(grads) == jax.tree.structure(params)
    assert _tree_norm(grads) > 0.0


def test_the_velocity_fn_has_the_shared_sampler_signature():
    # T1's sampler and T2's kernel both call `velocity_fn(hidden_states=..., timestep=...,
    # encoder_hidden_states=...)`; the seam is that signature and nothing else.
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    rules, mesh = _mesh_context()
    with rules, mesh:
        make_velocity_fn, params = build_cfg_velocity_fn(transformer, adapters)
        velocity_fn = make_velocity_fn(params, actions=_inputs(3)["actions"], guide_scale=_GUIDE)
    assert list(inspect.signature(velocity_fn).parameters) == [
        "hidden_states",
        "timestep",
        "encoder_hidden_states",
    ]
    outer = inspect.signature(make_velocity_fn).parameters
    assert list(outer)[0] == "params"
    for required in ("actions", "guide_scale"):
        assert outer[required].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_rollout_pins_frame_zero_at_every_step():
    _requires_backend()
    transformer, adapters = _tiny_cfg_stack()
    data = _inputs(17)
    sigmas, timesteps = _grid()
    rules, mesh = _mesh_context()
    with rules, mesh:
        make_velocity_fn, params = build_cfg_velocity_fn(transformer, adapters)
        out = cfg_rollout(
            apply_first_frame_pin(data["z"], data["z_i0"]),
            velocity_fn=make_velocity_fn(params, actions=data["actions"], guide_scale=_GUIDE),
            sigmas=sigmas,
            timesteps=timesteps,
            context=data["null_context"],
            z_i0=data["z_i0"],
            start=0,
            num_steps=_STEPS,
        )
    assert np.array_equal(np.asarray(out[:, :, :1]), np.asarray(data["z_i0"]))


def test_the_rollout_delegates_every_step_to_the_shared_sampler():
    # One Euler update and one frame-0 pin in the experiment: this module must not re-implement them.
    source = _MODULE_PATH.read_text(encoding="utf-8")
    assert "overfit100_sampler_step" in source
    assert "sigmas[index + 1]" not in source
    assert "apply_first_frame_pin" not in source
    assert "_build_per_token_timestep" not in source
