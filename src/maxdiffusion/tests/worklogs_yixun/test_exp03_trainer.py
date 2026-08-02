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
    assert "functools.partial(train_step_fn" in source
    assert "functools.partial(_train_step" not in source


def test_the_step_still_binds_this_modules_loss_late(monkeypatch):
    # exp_02's suite pins that the step calls THIS module's _denoising_loss by name (it patches the
    # name and asserts the spy is hit). The factory must not freeze that binding at import time --
    # settled behaviour, preserved deliberately, and the control arm inherits it.
    state, data, config, scheduler = _fixture()
    seen = []
    real = parent._denoising_loss

    def spy(params, st, batch, rng, cfg, sched):
        seen.append("called")
        return real(params, st, batch, rng, cfg, sched)

    monkeypatch.setattr(parent, "_denoising_loss", spy)
    parent._train_step(state, data, jax.random.key(5), scheduler, config)
    _, control_step = _exp03_trainer(config)._loss_and_step_fns()
    control_step(state, data, jax.random.key(5), scheduler, config)
    assert seen == ["called", "called"]


def test_the_parent_hook_returns_the_plain_objective():
    trainer = parent.WanTI2VOverfit100Trainer.__new__(parent.WanTI2VOverfit100Trainer)
    loss_fn, step_fn = trainer._loss_and_step_fns()
    assert loss_fn is parent._denoising_loss
    assert step_fn is parent._train_step


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
def test_an_unimplemented_objective_refuses_to_start(objective):
    _, _, config, _ = _fixture()
    config.exp03_objective = objective
    with pytest.raises(NotImplementedError) as excinfo:
        _exp03_trainer(config)._loss_and_step_fns()
    message = str(excinfo.value)
    assert objective in message and "round 3" in message


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
    # ...and the derivation never touches the training key: no split of it appears in the source.
    source = Path(exp03.__file__).read_text()
    assert "jax.random.split" not in source


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


def test_the_config_types_survive_pyconfig_override_rules():
    # pyconfig coerces an override to the YAML value's type, so a knob typed as int here can never
    # be given a float on the command line by accident (and vice versa).
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert type(cfg["exp03_objective"]) is str
    assert type(cfg["exp03_k_a"]) is int and type(cfg["exp03_k_b"]) is int
    assert type(cfg["exp03_lambda"]) is float and type(cfg["exp03_p_ss_max"]) is float
    assert type(cfg["exp03_p_ss_ramp_steps"]) is int


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


def test_the_launcher_is_otherwise_the_overfit100_launcher():
    # Guards against the clone drifting: every non-exp03 command-line override the exp_02 launcher
    # passes must still be passed here.
    base = _OVERFIT100_LAUNCHER.read_text()
    text = _LAUNCHER.read_text()
    overrides = [line.strip() for line in base.splitlines() if line.strip().endswith('}" \\') and "=" in line]
    missing = [line for line in overrides if line not in text]
    assert not missing, missing
