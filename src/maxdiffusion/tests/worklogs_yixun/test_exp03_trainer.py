"""exp_03 cycle A round 2 — the trainer skeleton, ctrl0-capable.

What has to be true before ctrl0 can be a *replication guard* rather than a second implementation:

1. **The parent's step did not move.** Introducing the ``_loss_and_step_fns`` hook refactored
   ``_train_step`` into a factory; a verbatim copy of the pre-refactor step is compared against the
   refactored one on fixed inputs with exact equality — parameters, metrics and the returned rng.
   exp_02's runs are settled history; their semantics may not shift under them.
2. **The control arm IS the parent's objective**, by identity, not by a copy that looks equal.
3. **The shared RNG stream is preserved exactly**, including across a resume boundary, and the new
   auxiliary keys cannot advance it.
4. **Nothing else changed**: same state class, same Orbax item shapes, same preflights.
5. Unimplemented objectives fail at startup rather than silently training the control.

Everything runs on CPU against a tiny stub transformer with a real ``nnx.Param`` (the exp_01
harness pattern), so the optimizer step, the gradient and the state update are genuine.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
import pytest
import yaml
from flax import nnx

import maxdiffusion.train_wan as train_wan
import maxdiffusion.trainers.wan_ti2v_exp03_trainer as exp03
import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as parent
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

_REPO = Path(parent.__file__).parents[3]
_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_exp03.yml"
_OVERFIT100_CONFIG = _REPO / "src/maxdiffusion/configs/base_wan_5b_overfit100.yml"
_LAUNCHER = _REPO / "bash_scripts/train_wan_exp03.sh"
_OVERFIT100_LAUNCHER = _REPO / "bash_scripts/train_wan_overfit100.sh"

_DATA_B, _BSZ, _C, _F, _H, _W = 3, 2, 3, 4, 5, 6
_SLOTS, _LEN, _DIM = 4, 4, 8


class _StubTransformer(nnx.Module):
    """Tiny stand-in with a real Param, so gradients and the optimizer step are genuine."""

    def __init__(self):
        self.gain = nnx.Param(jnp.asarray(0.5, dtype=jnp.float32))

    def __call__(self, **kwargs):
        hidden = kwargs["hidden_states"].astype(jnp.float32)
        t_mean = jnp.mean(kwargs["timestep"].astype(jnp.float32))
        ctx_mean = jnp.mean(kwargs["encoder_hidden_states"].astype(jnp.float32))
        return self.gain[...] * hidden + 0.01 * t_mean + 0.001 * ctx_mean


def _fixture(tx=None):
    transformer = _StubTransformer()
    graphdef, params, rest = nnx.split(transformer, nnx.Param, ...)
    context_table = jax.random.normal(jax.random.key(41), (_SLOTS, _LEN, _DIM), dtype=jnp.float32)
    state = parent.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx or optax.sgd(0.1),
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=context_table,
    )
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    data = {
        "z_i0": jax.random.normal(k1, (_DATA_B, _C, 1, _H, _W), dtype=jnp.float32),
        "z_video": jax.random.normal(k2, (_DATA_B, _C, _F, _H, _W), dtype=jnp.float32),
        "episode_index": jnp.asarray([0, 1, 2], dtype=jnp.int32),
        "actions": jax.random.normal(k3, (_DATA_B, 32, 7), dtype=jnp.float32),
    }
    config = SimpleNamespace(
        weights_dtype="float32",
        activations_dtype="float32",
        global_batch_size_to_train_on=_BSZ,
        side_adapter_sampling_steps=4,
        flow_shift=5.0,
        side_adapter_t_sampling="uniform",
        side_adapter_noise_mode="fresh",
        seed=0,
        model_type=exp03.EXP03_MODEL_TYPE,
        exp03_objective="control",
        exp03_k_a=2,
        exp03_k_b=2,
        exp03_lambda=0.5,
        exp03_p_ss_max=0.5,
        exp03_p_ss_ramp_steps=500,
    )
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=config.flow_shift, sigma_min=0.0, sigma_max=1.0)
    return state, data, config, scheduler


# =============================================================================================
# 1. Hook parity — the refactored parent step is the pre-refactor step, exactly.
# =============================================================================================


def _pre_refactor_train_step(state, data: dict, rng, scheduler, config):
    """VERBATIM copy of wan_ti2v_overfit100_trainer._train_step at e4a11a4 (pre-hook).

    It calls the module-level ``_denoising_loss`` directly, which is exactly what the refactor
    replaced with a parameter. Kept unrefactored: its value is that it is the old code.
    """
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        return parent._denoising_loss(params, state, data, loss_rng, config, scheduler)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    max_abs_grad = jax.tree_util.tree_reduce(
        lambda m, arr: jnp.maximum(m, jnp.max(jnp.abs(arr))), grads, initializer=-1.0
    )
    state = state.apply_gradients(grads=grads)
    metrics = {
        "scalar": {
            "learning/loss": loss,
            "learning/velocity_mse": aux["velocity_mse"],
            "learning/grad_norm": grad_norm,
            "learning/max_abs_grad": max_abs_grad,
            "learning/sigma_mean": aux["sigma_mean"],
            "learning/timestep_mean": aux["timestep_mean"],
            "learning/v_pred_l2": aux["v_pred_l2"],
            "learning/v_target_l2": aux["v_target_l2"],
            "learning/z_noisy_std": aux["z_noisy_std"],
            "learning/z_target_std": aux["z_target_std"],
            "learning/z_init_anchor_mse": aux["z_init_anchor_mse"],
        },
        "scalars": {},
    }
    return state, metrics, rng


def _assert_steps_identical(got, want):
    got_state, got_metrics, got_rng = got
    want_state, want_metrics, want_rng = want
    got_leaves = jax.tree_util.tree_leaves(got_state.params)
    want_leaves = jax.tree_util.tree_leaves(want_state.params)
    assert len(got_leaves) == len(want_leaves) and got_leaves
    for left, right in zip(got_leaves, want_leaves):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert set(got_metrics["scalar"]) == set(want_metrics["scalar"])
    for key, value in want_metrics["scalar"].items():
        assert np.array_equal(np.asarray(got_metrics["scalar"][key]), np.asarray(value)), key
    assert np.array_equal(np.asarray(jax.random.key_data(got_rng)), np.asarray(jax.random.key_data(want_rng)))


def test_the_refactored_parent_step_is_the_pre_refactor_step_exactly():
    state, data, config, scheduler = _fixture()
    rng = jax.random.key(7)
    _assert_steps_identical(
        parent._train_step(state, data, rng, scheduler, config),
        _pre_refactor_train_step(state, data, rng, scheduler, config),
    )


def test_the_factory_applied_to_the_plain_loss_is_the_parents_step():
    state, data, config, scheduler = _fixture()
    rng = jax.random.key(11)
    rebuilt = parent._make_train_step(parent._denoising_loss)
    _assert_steps_identical(
        rebuilt(state, data, rng, scheduler, config),
        _pre_refactor_train_step(state, data, rng, scheduler, config),
    )


def test_the_hook_is_what_the_loop_uses():
    # The refactor is pointless if start_training still names the module-level functions.
    source = inspect.getsource(parent.WanTI2VOverfit100Trainer.start_training)
    assert "self._loss_and_step_fns()" in source
    # The hook's step is what gets compiled (through the 4-argument production adapter), and the
    # module-level function is never re-bound behind the hook's back.
    assert "return train_step_fn(state, data, rng, scheduler, config, global_step=global_step)" in source
    assert "functools.partial(_train_step" not in source
    assert "jax.jit(\n            _jit_train_step," in source


def test_the_step_still_binds_this_modules_loss_late(monkeypatch):
    # exp_02's suite pins that the step calls THIS module's _denoising_loss by name (it patches the
    # name and asserts the spy is hit). The factory must not freeze that binding at import time --
    # settled behaviour, preserved deliberately, and the control arm inherits it.
    state, data, config, scheduler = _fixture()
    seen = []
    real = parent._denoising_loss

    def spy(params, st, batch, rng, cfg, sched, **kwargs):
        seen.append("called")
        return real(params, st, batch, rng, cfg, sched, **kwargs)

    monkeypatch.setattr(parent, "_denoising_loss", spy)
    parent._train_step(state, data, jax.random.key(5), scheduler, config)
    _, control_step = _exp03_trainer(config)._loss_and_step_fns()
    control_step(state, data, jax.random.key(5), scheduler, config)
    assert seen == ["called", "called"]


def test_the_loss_call_shape_is_exp02s_when_no_step_is_threaded():
    # exp_02's suite spies on _denoising_loss with its ORIGINAL six-argument signature. Threading
    # the step must not break that contract, so the step is passed only when there is one.
    state, data, config, scheduler = _fixture()
    seen = []

    def strict_spy(params, st, batch, rng, cfg, sched):  # NO **kwargs, on purpose
        seen.append("legacy")
        return parent._denoising_loss(params, st, batch, rng, cfg, sched)

    step_fn = parent._make_train_step(strict_spy)
    step_fn(state, data, jax.random.key(2), scheduler, config)  # legacy shape
    assert seen == ["legacy"]

    def step_aware_spy(params, st, batch, rng, cfg, sched, *, global_step=None):
        seen.append(("threaded", global_step))
        return parent._denoising_loss(params, st, batch, rng, cfg, sched)

    parent._make_train_step(step_aware_spy)(
        state, data, jax.random.key(2), scheduler, config, global_step=jnp.asarray(41, jnp.int32)
    )
    assert seen[-1][0] == "threaded" and int(seen[-1][1]) == 41


def test_the_parent_hook_returns_the_plain_objective():
    trainer = parent.WanTI2VOverfit100Trainer.__new__(parent.WanTI2VOverfit100Trainer)
    loss_fn, step_fn = trainer._loss_and_step_fns()
    assert loss_fn is parent._denoising_loss
    assert step_fn is parent._train_step


# ---------------------------------------------------------------------------------------------
# 1b. The JIT certificate — parity where production actually runs, not only eagerly.
# ---------------------------------------------------------------------------------------------


def _production_jit(step_fn, scheduler, config, *, counter=None):
    """Compile a step exactly the way ``start_training`` does: 4 positional args, step last."""

    def _jit_train_step(state, data, rng, global_step):
        if counter is not None:
            counter.append(1)  # the body runs once per TRACE, so this counts traces
        return step_fn(state, data, rng, scheduler, config, global_step=global_step)

    return jax.jit(_jit_train_step)


def _pre_refactor_jit(scheduler, config, *, counter=None):
    """The verbatim pre-refactor step behind the same 4-argument boundary (it ignores the step)."""

    def _jit_train_step(state, data, rng, global_step):
        if counter is not None:
            counter.append(1)
        del global_step
        return _pre_refactor_train_step(state, data, rng, scheduler, config)

    return jax.jit(_jit_train_step)


def _assert_states_identical(got, want):
    """Params, the FULL AdamW optimizer state, and the step counter — all exact."""
    for label, left, right in (
        ("params", got.params, want.params),
        ("opt_state", got.opt_state, want.opt_state),
    ):
        got_leaves = jax.tree_util.tree_leaves(left)
        want_leaves = jax.tree_util.tree_leaves(right)
        assert len(got_leaves) == len(want_leaves) and got_leaves, label
        for a, b in zip(got_leaves, want_leaves):
            assert np.array_equal(np.asarray(a), np.asarray(b)), label
    assert int(got.step) == int(want.step)


def test_the_compiled_steps_agree_over_repeated_cached_calls():
    # The production boundary: jitted, AdamW (so mu/nu/count all travel), several cached calls in
    # sequence so a divergence in the carried optimizer state would compound into view.
    state, data, config, scheduler = _fixture(tx=optax.adamw(1e-3))
    _, control_step = _exp03_trainer(config)._loss_and_step_fns()
    my_traces: list[int] = []
    their_traces: list[int] = []
    mine = _production_jit(control_step, scheduler, config, counter=my_traces)
    theirs = _pre_refactor_jit(scheduler, config, counter=their_traces)

    my_state, their_state = state, state
    my_rng = their_rng = jax.random.key(19)
    for step in range(4):
        global_step = jnp.asarray(step, dtype=jnp.int32)
        my_state, my_metrics, my_rng = mine(my_state, data, my_rng, global_step)
        their_state, their_metrics, their_rng = theirs(their_state, data, their_rng, global_step)
        _assert_states_identical(my_state, their_state)
        assert set(my_metrics["scalar"]) == set(their_metrics["scalar"])
        for key, value in their_metrics["scalar"].items():
            assert np.array_equal(np.asarray(my_metrics["scalar"][key]), np.asarray(value)), (step, key)
        assert np.array_equal(np.asarray(jax.random.key_data(my_rng)), np.asarray(jax.random.key_data(their_rng)))
    assert int(my_state.step) == 4  # the optimizer really stepped four times
    # Each compiled EXACTLY once across the four calls: no per-step retrace from the closure, and
    # the dynamic global step does not specialize the cache.
    assert sum(my_traces) == 1, sum(my_traces)
    assert sum(their_traces) == 1, sum(their_traces)


def test_the_compiled_steps_have_the_same_jaxpr():
    state, data, config, scheduler = _fixture(tx=optax.adamw(1e-3))
    _, control_step = _exp03_trainer(config)._loss_and_step_fns()
    args = (state, data, jax.random.key(21), jnp.asarray(3, dtype=jnp.int32))

    def mine(s, d, r, g):
        return control_step(s, d, r, scheduler, config, global_step=g)

    def theirs(s, d, r, g):
        del g
        return _pre_refactor_train_step(s, d, r, scheduler, config)

    my_jaxpr = jax.make_jaxpr(mine)(*args)
    their_jaxpr = jax.make_jaxpr(theirs)(*args)
    my_primitives = sorted(str(eqn.primitive) for eqn in my_jaxpr.jaxpr.eqns)
    their_primitives = sorted(str(eqn.primitive) for eqn in their_jaxpr.jaxpr.eqns)
    assert my_primitives == their_primitives
    assert len(my_jaxpr.jaxpr.eqns) == len(their_jaxpr.jaxpr.eqns)


def test_the_control_step_ignores_the_threaded_global_step():
    # The control must be stepwise-stationary: the same inputs at different global steps give the
    # same result, bit for bit. (A trial will NOT have this property -- that is the point.)
    state, data, config, scheduler = _fixture(tx=optax.adamw(1e-3))
    _, control_step = _exp03_trainer(config)._loss_and_step_fns()
    compiled = _production_jit(control_step, scheduler, config)
    rng = jax.random.key(23)
    first = compiled(state, data, rng, jnp.asarray(0, dtype=jnp.int32))
    later = compiled(state, data, rng, jnp.asarray(10_000, dtype=jnp.int32))
    _assert_states_identical(first[0], later[0])
    for key, value in first[1]["scalar"].items():
        assert np.array_equal(np.asarray(value), np.asarray(later[1]["scalar"][key])), key
    # ...and eager equals compiled, with and without the argument.
    eager = control_step(state, data, rng, scheduler, config)
    _assert_states_identical(first[0], eager[0])


def test_the_production_loop_threads_the_loops_step_not_the_states():
    import ast
    import textwrap

    source = inspect.getsource(parent.WanTI2VOverfit100Trainer.start_training)
    assert "p_train_step(state, batch, rng, jnp.asarray(step, dtype=jnp.int32))" in source
    assert "in_shardings=(state_shardings, data_shardings, None, None)" in source
    # And the loop never READS state.step -- checked structurally, so the comment explaining why
    # is not mistaken for the thing it warns against. (Restore brings back params and opt_state;
    # state.step is whatever the freshly built state had, so a step-keyed draw would restart.)
    tree = ast.parse(textwrap.dedent(source))
    reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "step"
        and isinstance(node.value, ast.Name)
        and node.value.id == "state"
    ]
    assert not reads


# =============================================================================================
# 2. The control arm is the parent's objective, by identity.
# =============================================================================================


def _exp03_trainer(config):
    trainer = exp03.Exp03Trainer.__new__(exp03.Exp03Trainer)
    trainer.config = config
    return trainer


def test_the_control_arm_returns_the_parents_functions_by_identity():
    _, _, config, _ = _fixture()
    loss_fn, step_fn = _exp03_trainer(config)._loss_and_step_fns()
    assert loss_fn is parent._denoising_loss
    assert step_fn is parent._train_step


def test_the_control_arm_step_equals_the_parent_step_exactly():
    state, data, config, scheduler = _fixture()
    rng = jax.random.key(13)
    _, step_fn = _exp03_trainer(config)._loss_and_step_fns()
    _assert_steps_identical(
        step_fn(state, data, rng, scheduler, config),
        _pre_refactor_train_step(state, data, rng, scheduler, config),
    )


@pytest.mark.parametrize("objective", ["corrective_ss", "rollout_loss", "combined"])
def test_every_declared_objective_now_resolves_to_its_implementation(objective):
    # Round 3 landed the three trials: each resolves to ITS loss (never silently to the control's).
    _, _, config, _ = _fixture()
    config.exp03_objective = objective
    loss_fn, step_fn = _exp03_trainer(config)._loss_and_step_fns()
    assert loss_fn is exp03.EXP03_LOSSES[objective]
    assert loss_fn is not parent._denoising_loss
    assert step_fn is not parent._train_step
    assert set(exp03.EXP03_IMPLEMENTED_OBJECTIVES) == set(exp03.EXP03_OBJECTIVES)


def test_a_declared_but_unimplemented_objective_refuses_with_notimplementederror(monkeypatch):
    # The REAL path: an objective the config surface accepts (validate_exp03_config reads
    # EXP03_OBJECTIVES) but the dispatch table has no entry for. It must raise NotImplementedError
    # naming the arm -- not a bare KeyError, and never a silent fallback to the control.
    _, _, config, _ = _fixture()
    monkeypatch.setattr(exp03, "EXP03_OBJECTIVES", exp03.EXP03_OBJECTIVES + ("future_arm",))
    config.exp03_objective = "future_arm"
    assert exp03.validate_exp03_config(config) == "future_arm"  # the config surface accepts it...
    with pytest.raises(NotImplementedError) as excinfo:
        _exp03_trainer(config)._loss_and_step_fns()  # ...and the dispatch refuses it
    message = str(excinfo.value)
    assert "future_arm" in message and "control" in message


def test_the_implemented_set_is_derived_from_the_dispatch_table():
    # Not hand-maintained: what is implemented is what can be run.
    assert set(exp03.EXP03_IMPLEMENTED_OBJECTIVES) == {"control", *exp03.EXP03_LOSSES}
    assert set(exp03.EXP03_IMPLEMENTED_OBJECTIVES) == set(exp03.EXP03_OBJECTIVES)


def test_no_unused_rng_purpose_is_declared():
    # self_gen_noise was removed: A's off-path state uses the SAME epsilon as its teacher-forced
    # branch, and a declared-but-undrawn purpose misstates that design.
    assert "self_gen_noise" not in exp03.EXP03_AUX_PURPOSES
    source = Path(exp03.__file__).read_text()
    for purpose in exp03.EXP03_AUX_PURPOSES:
        assert source.count(f'purpose="{purpose}"') >= 1, f"{purpose} is declared but never drawn"


def test_an_unknown_objective_is_a_config_error_not_a_silent_control_run():
    _, _, config, _ = _fixture()
    config.exp03_objective = "definitely_not_an_objective"
    with pytest.raises(ValueError):
        _exp03_trainer(config)._loss_and_step_fns()


@pytest.mark.parametrize(
    "key,value",
    [
        ("exp03_k_a", 3),
        ("exp03_k_a", 0),
        ("exp03_k_b", 3),
        ("exp03_lambda", 1.5),
        ("exp03_lambda", -0.1),
        ("exp03_p_ss_max", 1.5),
        ("exp03_p_ss_ramp_steps", -1),
    ],
)
def test_out_of_contract_knobs_are_refused(key, value):
    _, _, config, _ = _fixture()
    setattr(config, key, value)
    with pytest.raises(ValueError):
        exp03.validate_exp03_config(config)


# =============================================================================================
# 3. RNG discipline — the ctrl0 replication keystone.
# =============================================================================================


def _shared_stream_draws(step_fn, *, seed, steps, start_step=0):
    """Drive the REAL train step and record what the shared stream produced, per step.

    Mirrors ``start_training``: the stream is ``key(seed + 1)`` created at segment start and
    advanced by the step function itself, once per step. ``start_step`` exists to model a resume:
    exp_02 re-creates the key on restart and runs ``range(start_step, max_train_steps)``, so a
    resumed segment's k-th step uses the k-th key of a FRESH stream. That is settled behaviour and
    ctrl0 must reproduce it, not improve on it.
    """
    state, data, config, scheduler = _fixture()
    rng = jax.random.key(seed + 1)
    seen = []
    for _ in range(start_step, steps):
        state, metrics, rng = step_fn(state, data, rng, scheduler, config)
        seen.append(
            (
                np.asarray(jax.random.key_data(rng)).copy(),
                float(metrics["scalar"]["learning/loss"]),
            )
        )
    return seen


def test_the_new_trainers_shared_stream_is_the_parents_exactly():
    _, _, config, _ = _fixture()
    _, exp03_step = _exp03_trainer(config)._loss_and_step_fns()
    mine = _shared_stream_draws(exp03_step, seed=0, steps=5)
    theirs = _shared_stream_draws(_pre_refactor_train_step, seed=0, steps=5)
    assert len(mine) == 5
    for (my_key, my_loss), (their_key, their_loss) in zip(mine, theirs):
        assert np.array_equal(my_key, their_key)
        assert my_loss == their_loss


def test_the_shared_stream_matches_across_a_resume_boundary():
    # A segment preempted at step 3 and restarted: exp_02 rebuilds key(seed+1) and runs the
    # remaining steps off a fresh stream. Both trainers must do the same thing, step for step.
    _, _, config, _ = _fixture()
    _, exp03_step = _exp03_trainer(config)._loss_and_step_fns()
    for start_step in (0, 3):
        mine = _shared_stream_draws(exp03_step, seed=0, steps=5, start_step=start_step)
        theirs = _shared_stream_draws(_pre_refactor_train_step, seed=0, steps=5, start_step=start_step)
        assert len(mine) == 5 - start_step
        for (my_key, my_loss), (their_key, their_loss) in zip(mine, theirs):
            assert np.array_equal(my_key, their_key)
            assert my_loss == their_loss


def test_the_auxiliary_key_is_derived_not_split_so_it_cannot_advance_the_stream():
    # THE ctrl0 property: drawing auxiliary randomness between steps leaves the shared stream in
    # exactly the state it would have been in. Interleave aux draws into a run and compare.
    _, _, config, _ = _fixture()
    _, step_fn = _exp03_trainer(config)._loss_and_step_fns()
    state, data, config, scheduler = _fixture()

    plain = _shared_stream_draws(step_fn, seed=0, steps=4)

    rng = jax.random.key(0 + 1)
    interleaved = []
    for step in range(4):
        for purpose in exp03.EXP03_AUX_PURPOSES:
            aux = exp03.exp03_aux_key(seed=0, global_step=step, purpose=purpose)
            jax.random.normal(aux, (3,))  # actually consume it
        state, metrics, rng = step_fn(state, data, rng, scheduler, config)
        interleaved.append((np.asarray(jax.random.key_data(rng)).copy(), float(metrics["scalar"]["learning/loss"])))
    for (a_key, a_loss), (b_key, b_loss) in zip(interleaved, plain):
        assert np.array_equal(a_key, b_key)
        assert a_loss == b_loss
    # ...and the DERIVATION never touches the training key: no split appears inside exp03_aux_key
    # itself. (The trial objectives do split the SHARED stream -- in exp_02's exact order -- which
    # is the point: their epsilon and dropout keys are the control's.)
    import ast

    tree = ast.parse(Path(exp03.__file__).read_text())
    fold_fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "exp03_aux_key"
    )
    splits = [
        node for node in ast.walk(fold_fn) if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "split"
    ]
    assert not splits


def test_the_auxiliary_key_is_tracer_safe_and_agrees_with_eager():
    # It is computed INSIDE the compiled train step, where the global step is a tracer. An int()
    # coercion there raises; the values must also be the ones eager code would produce.
    for step in (0, 1, 250, 12_499):
        eager = exp03.exp03_aux_key(seed=0, global_step=step, purpose="index_support")
        compiled = jax.jit(lambda value: exp03.exp03_aux_key(seed=0, global_step=value, purpose="index_support"))(
            jnp.asarray(step, dtype=jnp.int32)
        )
        assert np.array_equal(np.asarray(jax.random.key_data(eager)), np.asarray(jax.random.key_data(compiled))), step
    # ...and a draw made from it is likewise identical across the boundary.
    eager_draw = jax.random.normal(exp03.exp03_aux_key(seed=0, global_step=7, purpose="p_ss_coin"), (4,))
    compiled_draw = jax.jit(
        lambda value: jax.random.normal(exp03.exp03_aux_key(seed=0, global_step=value, purpose="p_ss_coin"), (4,))
    )(jnp.asarray(7, dtype=jnp.int32))
    assert np.array_equal(np.asarray(eager_draw), np.asarray(compiled_draw))
    # The fold really is on an array, and no int() coercion of the step reaches it: the only
    # int(global_step) allowed is inside the concrete-value guard (checked structurally below).
    import ast

    tree = ast.parse(Path(exp03.__file__).read_text())
    fold_fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "exp03_aux_key"
    )
    guard_lines = {node.lineno for node in ast.walk(fold_fn) if isinstance(node, ast.If)}
    coercions = [
        node
        for node in ast.walk(fold_fn)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "int"
        and any(getattr(arg, "id", "") == "global_step" for arg in node.args)
        and node.lineno not in guard_lines
    ]
    assert not coercions, "int(global_step) outside the concrete-value guard would raise on a tracer"
    assert "jnp.asarray(global_step" in Path(exp03.__file__).read_text()


def _aux_recording_loss(params, state, data, rng, config, scheduler, *, global_step=None):
    """A stand-in for a round-3 objective: its value IS a draw keyed on the global step.

    Deliberately free of the shared stream, so what this records is exactly the step-keyed
    auxiliary draw. (The shared stream restarts on resume by design -- that is exp_02's settled
    behaviour -- so mixing it in would mask the property under test.)
    """
    key = exp03.exp03_aux_key(seed=int(config.seed), global_step=global_step, purpose="p_ss_coin")
    draw = jax.random.normal(key, ())
    gain = jax.tree_util.tree_leaves(params)[0].astype(jnp.float32)
    loss = draw + 0.0 * jnp.sum(gain)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    aux = {
        "velocity_mse": loss,
        "sigma_mean": zero,
        "timestep_mean": zero,
        "v_pred_l2": zero,
        "v_target_l2": zero,
        "z_noisy_std": zero,
        "z_target_std": zero,
        "z_init_anchor_mse": zero,
    }
    return loss, aux


def _run_segment(compiled, state, data, rng, steps):
    """``for step in range(*steps)`` exactly as start_training does; returns the per-step draws."""
    draws = []
    for step in range(*steps):
        state, metrics, rng = compiled(state, data, rng, jnp.asarray(step, dtype=jnp.int32))
        draws.append((step, float(metrics["scalar"]["learning/loss"])))
    return state, rng, draws


def test_the_step_keyed_draws_survive_a_real_save_and_restore(tmp_path):
    # THE resume-safety property, end to end through the PRODUCTION restore path: an uninterrupted
    # run and a run preempted at step 3 and restored must make the same auxiliary draw at the same
    # GLOBAL step. This is why the loop's step is threaded rather than state.step.
    state, data, config, scheduler = _fixture(tx=optax.adamw(1e-3))
    step_fn = parent._make_train_step(_aux_recording_loss)
    compiled = _production_jit(step_fn, scheduler, config)

    _, _, uninterrupted = _run_segment(compiled, state, data, jax.random.key(config.seed + 1), (0, 6))

    trainer = _exp03_trainer(
        SimpleNamespace(**{**vars(config), "checkpoint_max_to_keep": None, "save_final_checkpoint": True})
    )
    manager = trainer._build_checkpoint_manager(str(tmp_path / "ckpt"))
    mid_state, _, first_half = _run_segment(compiled, state, data, jax.random.key(config.seed + 1), (0, 3))
    trainer._save_checkpoint(manager, 3, mid_state)
    manager.wait_until_finished()

    # A genuine restart: a freshly built state, the production restore, the loop resuming at the
    # restored step with a fresh key -- exp_02's actual resume semantics.
    fresh_state, _, _, _ = _fixture(tx=optax.adamw(1e-3))
    restored, start_step = trainer._maybe_restore(manager, fresh_state)
    assert start_step == 3
    _, _, second_half = _run_segment(compiled, restored, data, jax.random.key(config.seed + 1), (start_step, 6))

    assert [step for step, _ in first_half + second_half] == list(range(6))
    assert first_half + second_half == uninterrupted
    # ...and the draws really vary with the step, so the equality above is not vacuous.
    assert len({value for _, value in uninterrupted}) == 6

    # The counterfactual that motivates threading the LOOP's step: state.step is NOT the global
    # step after a restore, so keying on it would have restarted the draws at 0.
    assert int(restored.step) == 0 != start_step


def test_the_compiled_step_can_key_an_objective_on_the_global_step():
    # A round-3 objective compiles and gets DIFFERENT draws at different steps, from one trace.
    state, data, config, scheduler = _fixture(tx=optax.adamw(1e-3))
    traces: list[int] = []
    compiled = _production_jit(parent._make_train_step(_aux_recording_loss), scheduler, config, counter=traces)
    rng = jax.random.key(3)
    first = compiled(state, data, rng, jnp.asarray(0, dtype=jnp.int32))[1]["scalar"]["learning/loss"]
    later = compiled(state, data, rng, jnp.asarray(1, dtype=jnp.int32))[1]["scalar"]["learning/loss"]
    assert float(first) != float(later)
    assert sum(traces) == 1  # the step is dynamic, not a compile-time constant


def test_the_auxiliary_key_is_deterministic_purpose_scoped_and_step_keyed():
    first = exp03.exp03_aux_key(seed=0, global_step=10, purpose="p_ss_coin")
    again = exp03.exp03_aux_key(seed=0, global_step=10, purpose="p_ss_coin")
    assert np.array_equal(np.asarray(jax.random.key_data(first)), np.asarray(jax.random.key_data(again)))
    variants = {
        "step": exp03.exp03_aux_key(seed=0, global_step=11, purpose="p_ss_coin"),
        "purpose": exp03.exp03_aux_key(seed=0, global_step=10, purpose="index_support"),
        "seed": exp03.exp03_aux_key(seed=1, global_step=10, purpose="p_ss_coin"),
    }
    for label, key in variants.items():
        assert not np.array_equal(np.asarray(jax.random.key_data(first)), np.asarray(jax.random.key_data(key))), label
    # Resume stability: the value at a step does not depend on how many draws preceded it.
    assert np.array_equal(
        np.asarray(jax.random.normal(exp03.exp03_aux_key(seed=0, global_step=250, purpose="index_support"), (5,))),
        np.asarray(jax.random.normal(exp03.exp03_aux_key(seed=0, global_step=250, purpose="index_support"), (5,))),
    )


def test_purpose_ids_are_name_hashed_so_adding_one_renumbers_nothing():
    ids = {purpose: exp03._purpose_id(purpose) for purpose in exp03.EXP03_AUX_PURPOSES}
    assert len(set(ids.values())) == len(ids)
    # Hash of the NAME, not the position: this is what makes a future purpose additive.
    import hashlib

    for purpose, value in ids.items():
        assert value == int.from_bytes(hashlib.sha256(purpose.encode("utf-8")).digest()[:4], "big")
    with pytest.raises(ValueError):
        exp03._purpose_id("undeclared_purpose")
    with pytest.raises(ValueError):
        exp03.exp03_aux_key(seed=0, global_step=-1, purpose="p_ss_coin")


def test_the_auxiliary_root_is_not_the_training_key():
    # key(seed + 1) is the training stream; the aux root must be a different key entirely.
    assert exp03.EXP03_AUX_SEED_OFFSET != 1
    training_root = jax.random.key(0 + 1)
    aux_root = jax.random.key(0 + exp03.EXP03_AUX_SEED_OFFSET)
    assert not np.array_equal(
        np.asarray(jax.random.key_data(training_root)), np.asarray(jax.random.key_data(aux_root))
    )


# =============================================================================================
# 4. State / checkpoint compatibility.
# =============================================================================================


def test_the_state_class_and_orbax_item_shapes_are_unchanged():
    # The exp_03 trainer inherits the state and the save/restore path; it defines neither, so the
    # Orbax item layout cannot drift away from exp_02's checkpoints.
    assert "Overfit100TrainState" not in vars(exp03)
    for name in ("_save_checkpoint", "_maybe_restore", "_build_checkpoint_manager", "_checkpoint_manager_options"):
        assert name not in vars(exp03.Exp03Trainer), f"{name} is overridden; the checkpoint layout could drift"
        assert getattr(exp03.Exp03Trainer, name) is getattr(parent.WanTI2VOverfit100Trainer, name), name
    save_source = inspect.getsource(exp03.Exp03Trainer._save_checkpoint)
    for item in ("params=ocp.args.StandardSave", "opt_state=ocp.args.StandardSave", "step=ocp.args.JsonSave"):
        assert item in save_source, item


def test_the_preflights_are_inherited_unchanged():
    for name in (
        "_validate_probe_config",
        "_validate_overfit100_config",
        "_validate_pinned_snapshot",
        "_preflight_dataset",
        "_build_context_table",
        "_build_optimizer",
        "_load_dataset",
        "_shard_state",
    ):
        assert getattr(exp03.Exp03Trainer, name) is getattr(parent.WanTI2VOverfit100Trainer, name), name


def test_the_checkpoint_roundtrips_through_the_inherited_path(tmp_path):
    # A real Orbax save/restore of the real state structure at synthetic shapes, through the
    # PRODUCTION methods. The in-memory params are corrupted before restore, so "restored == saved"
    # is a real claim rather than "== whatever was already in memory".
    state, _, config, _ = _fixture(tx=optax.adamw(1e-4))
    trainer = _exp03_trainer(
        SimpleNamespace(**{**vars(config), "checkpoint_max_to_keep": None, "save_final_checkpoint": True})
    )
    manager = trainer._build_checkpoint_manager(str(tmp_path / "ckpt"))
    trainer._save_checkpoint(manager, 7, state)
    manager.wait_until_finished()
    assert manager.latest_step() == 7

    corrupted = state.replace(params=jax.tree_util.tree_map(lambda leaf: leaf + 99.0, state.params))
    restored, start_step = trainer._maybe_restore(manager, corrupted)
    assert start_step == 7
    for left, right in zip(jax.tree_util.tree_leaves(restored.params), jax.tree_util.tree_leaves(state.params)):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    for left, right in zip(jax.tree_util.tree_leaves(restored.opt_state), jax.tree_util.tree_leaves(state.opt_state)):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    # ...and the context table is NOT restored -- it is rebuilt every start, exp_02's rule.
    assert np.array_equal(np.asarray(restored.context_table), np.asarray(corrupted.context_table))


def test_the_from_scratch_path_starts_at_step_zero_through_the_same_code(tmp_path):
    # Tier 2 starts from the pretrained init with an EMPTY checkpoint dir: the same _maybe_restore
    # returns the state untouched at step 0. No exp_03-specific branch exists to diverge.
    state, _, config, _ = _fixture(tx=optax.adamw(1e-4))
    trainer = _exp03_trainer(
        SimpleNamespace(**{**vars(config), "checkpoint_max_to_keep": None, "save_final_checkpoint": True})
    )
    manager = trainer._build_checkpoint_manager(str(tmp_path / "empty"))
    restored, start_step = trainer._maybe_restore(manager, state)
    assert start_step == 0
    assert restored is state


def test_the_exp03_trainer_defines_no_restore_or_dataset_branch():
    # Tier 1 (seeded checkpoint) and Tier 2 (pretrained init) differ only in what is on disk, so
    # the subclass must contribute no restore/dataset code of its own -- checked structurally, so a
    # mention in the docstring is not mistaken for an implementation.
    import ast

    tree = ast.parse(Path(exp03.__file__).read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(name.startswith("orbax") for name in imported), imported
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert defined & {"_loss_and_step_fns", "start_training"}
    assert not defined & {
        "_maybe_restore",
        "_save_checkpoint",
        "_load_dataset",
        "_build_checkpoint_manager",
        "_build_optimizer",
        "_shard_state",
        "_build_context_table",
    }, defined


# =============================================================================================
# 5. Dispatch, config and launcher.
# =============================================================================================


def test_model_type_routes_to_the_exp03_trainer():
    source = inspect.getsource(train_wan.train)
    assert '"EXP03_TI2V"' in source
    assert "from maxdiffusion.trainers.wan_ti2v_exp03_trainer import Exp03Trainer" in source
    assert "trainer = Exp03Trainer(config)" in source
    assert exp03.EXP03_MODEL_TYPE == "EXP03_TI2V"


def test_the_config_is_the_overfit100_config_plus_the_exp03_keys():
    exp03_cfg = yaml.safe_load(_CONFIG.read_text())
    base_cfg = yaml.safe_load(_OVERFIT100_CONFIG.read_text())
    added = set(exp03_cfg) - set(base_cfg)
    assert added == {
        "exp03_objective",
        "exp03_k_a",
        "exp03_k_b",
        "exp03_lambda",
        "exp03_p_ss_max",
        "exp03_p_ss_ramp_steps",
        "exp03_ramp_origin",
        "exp03_snapshot_before_step",
        "s1_5_num_batches",
        "s1_5_support_draws",
        "exp03_support_salt",
    }
    assert not set(base_cfg) - set(exp03_cfg), "keys were dropped from the exp_02 config"
    assert exp03_cfg["model_type"] == "EXP03_TI2V"
    # Every other key is exp_02's, unchanged: the objective is the experiment variable.
    differing = {key for key in base_cfg if exp03_cfg[key] != base_cfg[key]}
    assert differing == {"model_type"}, differing
    # Defaults are the plan's.
    assert exp03_cfg["exp03_objective"] == "control"
    assert (exp03_cfg["exp03_k_a"], exp03_cfg["exp03_k_b"]) == (2, 2)
    assert exp03_cfg["exp03_lambda"] == 0.5
    assert exp03_cfg["exp03_p_ss_max"] == 0.5
    assert exp03_cfg["exp03_p_ss_ramp_steps"] == 500
    assert exp03_cfg["exp03_ramp_origin"] == 0  # Tier 2 default; Tier-1 arms pass 10000


def test_the_config_types_survive_pyconfig_override_rules():
    # pyconfig coerces an override to the YAML value's type, so a knob typed as int here can never
    # be given a float on the command line by accident (and vice versa).
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert type(cfg["exp03_objective"]) is str
    assert type(cfg["exp03_k_a"]) is int and type(cfg["exp03_k_b"]) is int
    assert type(cfg["exp03_lambda"]) is float and type(cfg["exp03_p_ss_max"]) is float
    assert type(cfg["exp03_p_ss_ramp_steps"]) is int and type(cfg["exp03_ramp_origin"]) is int


def test_the_launcher_passes_every_exp03_knob_through():
    text = _LAUNCHER.read_text()
    assert "src/maxdiffusion/configs/base_wan_5b_exp03.yml" in text
    assert "src/maxdiffusion/configs/base_wan_5b_overfit100.yml" not in text
    for env, key in (
        ("EXP03_OBJECTIVE", "exp03_objective"),
        ("EXP03_K_A", "exp03_k_a"),
        ("EXP03_K_B", "exp03_k_b"),
        ("EXP03_LAMBDA", "exp03_lambda"),
        ("EXP03_P_SS_MAX", "exp03_p_ss_max"),
        ("EXP03_P_SS_RAMP_STEPS", "exp03_p_ss_ramp_steps"),
        ("EXP03_RAMP_ORIGIN", "exp03_ramp_origin"),
        ("EXP03_SNAPSHOT_BEFORE_STEP", "exp03_snapshot_before_step"),
    ):
        assert f'{env}="${{{env}:-' in text, env  # env-overridable with a default
        assert f'{key}="${{{env}}}"' in text, key  # ...and actually forwarded
        assert f'echo "{env}=' in text, env  # ...and logged, so the run records what it ran


def test_the_launcher_keeps_the_overfit100_safety_apparatus():
    text = _LAUNCHER.read_text()
    for required in (
        "prefetch_hf_snapshot.sh",
        "local_files_only=True",
        "export COMMIT",
        "MODEL_REVISION",
        "expected_model_revision=",
        "model_manifest_path=",
    ):
        assert required in text, required
    # The training arm needs no ffmpeg (loss-arm precedent) -- and says so.
    assert "# >>> ffmpeg ensure" not in text
    assert "ffmpeg" in text


# DOTALL: a joined multiline value (LIBTPU_INIT_ARGS) is one string with newlines in it.
_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)$", re.DOTALL)


def _logical_lines(text: str) -> list[str]:
    """Join backslash-continued lines, so a multiline value is ONE entry.

    ``LIBTPU_INIT_ARGS`` spans ~25 continued lines of XLA flags. Parsing line by line would let any
    one of those flags drift between the two launchers unnoticed, which is exactly the hole this
    closes: the whole value is compared verbatim, as a single string.
    """
    joined: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if buffer:
            buffer += "\n" + line
        else:
            buffer = line
        if line.endswith("\\"):
            continue
        joined.append(buffer)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def _launcher_defaults(text: str) -> dict[str, str]:
    """Every uppercase assignment -> its value, VERBATIM.

    Covers three shapes the earlier suffix-based parser missed and that a clone can silently drift:
    ``export``-prefixed environment defaults (``export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"``),
    plain literal exports (``export PYTHONUNBUFFERED=1``), and multiline continued values
    (``LIBTPU_INIT_ARGS``). The value is kept as written -- including the ``${NAME:-...}`` wrapper --
    so a changed default, a changed XLA flag, or a changed fallback all show up as a value mismatch.
    """
    out: dict[str, str] = {}
    for line in _logical_lines(text):
        match = _ASSIGNMENT.match(line.strip())
        if match:
            out[match.group(1)] = match.group(2)
    return out


def _launcher_overrides(text: str) -> dict[str, str]:
    """Every ``key=value`` the launcher passes to train_wan.py, conditionals included verbatim."""
    block = text[text.index("python src/maxdiffusion/train_wan.py") :]
    out: dict[str, str] = {}
    for raw in block.splitlines()[1:]:
        line = raw.strip()
        if line.endswith("\\"):
            line = line[:-1].strip()
        if not line or line.endswith(".yml"):
            continue
        if line.startswith("${"):
            out[line] = ""  # a conditional override: compared as a whole, so dropping it shows up
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key] = value
    return out


# The ONLY differences the exp_03 launcher is allowed to have. Deliberately tight: anything else
# that differs is drift between two launchers that must otherwise stay the same recipe.
_ALLOWED_DEFAULT_DELTAS = {
    "RUN_NAME",
    "OUTPUT_DIR",
    "EXP03_OBJECTIVE",
    "EXP03_K_A",
    "EXP03_K_B",
    "EXP03_LAMBDA",
    "EXP03_P_SS_MAX",
    "EXP03_P_SS_RAMP_STEPS",
    "EXP03_RAMP_ORIGIN",
    "EXP03_SNAPSHOT_BEFORE_STEP",
}
_ALLOWED_OVERRIDE_DELTAS = {
    "exp03_objective",
    "exp03_k_a",
    "exp03_k_b",
    "exp03_lambda",
    "exp03_p_ss_max",
    "exp03_p_ss_ramp_steps",
    "exp03_ramp_origin",
    "exp03_snapshot_before_step",
}


def _assert_maps_agree(base: dict, mine: dict, allowlist: set, what: str):
    """Bidirectional comparison outside the allowlist: same keys AND same values."""
    base_keys = set(base) - allowlist
    my_keys = set(mine) - allowlist
    assert my_keys == base_keys, (
        f"{what} drifted: only in exp03 {sorted(my_keys - base_keys)}, "
        f"missing from exp03 {sorted(base_keys - my_keys)}"
    )
    differing = {key: (base[key], mine[key]) for key in base_keys if base[key] != mine[key]}
    assert not differing, f"{what} values drifted: {differing}"


def test_the_launcher_defaults_do_not_drift_from_the_overfit100_launcher():
    # Not a suffix filter: the parsed DEFAULT map is compared in both directions, so changing a
    # shared default (LEARNING_RATE, WARMUP_STEPS, PER_DEVICE_BATCH_SIZE, ...) is a failure.
    base = _launcher_defaults(_OVERFIT100_LAUNCHER.read_text())
    mine = _launcher_defaults(_LAUNCHER.read_text())
    # The parser really parsed -- including the shapes a suffix filter misses.
    assert "LEARNING_RATE" in base and "MAX_TRAIN_STEPS" in base
    assert "JAX_PLATFORMS" in base and "PYTHONUNBUFFERED" in base  # export-prefixed
    assert "LIBTPU_INIT_ARGS" in base  # multiline
    assert base["LIBTPU_INIT_ARGS"].count("--xla") > 10, "the continued XLA flags were not joined"
    assert "\n" in base["LIBTPU_INIT_ARGS"]  # ...as ONE entry
    assert set(mine) - set(base) == {
        "EXP03_OBJECTIVE",
        "EXP03_K_A",
        "EXP03_K_B",
        "EXP03_LAMBDA",
        "EXP03_P_SS_MAX",
        "EXP03_P_SS_RAMP_STEPS",
        "EXP03_RAMP_ORIGIN",
        "EXP03_SNAPSHOT_BEFORE_STEP",
    }
    _assert_maps_agree(base, mine, _ALLOWED_DEFAULT_DELTAS, "launcher defaults")


def test_every_command_line_override_survives_the_clone():
    # Includes the OPTIONAL tfrecord_shuffle_buffer_size conditional, which a suffix filter missed.
    base = _launcher_overrides(_OVERFIT100_LAUNCHER.read_text())
    mine = _launcher_overrides(_LAUNCHER.read_text())
    assert any("tfrecord_shuffle_buffer_size" in key for key in base), "the parser missed the conditional"
    assert "learning_rate" in base and "hardware" in base
    _assert_maps_agree(base, mine, _ALLOWED_OVERRIDE_DELTAS, "launcher overrides")
    assert set(mine) - set(base) == _ALLOWED_OVERRIDE_DELTAS
