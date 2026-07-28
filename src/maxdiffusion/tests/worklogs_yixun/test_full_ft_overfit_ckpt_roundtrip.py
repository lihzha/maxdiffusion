"""CPU-only Orbax round-trip + restore tests for the full-FT generate path (round 4).

exp_01 round 4 ("ckpt-generation") behaviorally proves the three things the round-3
Codex review made binding for this round (its "Notes for round 4"):

  (a) the generic ``params``/``opt_state``/``step`` restore into ``FullFTTrainState``
      -- a real tiny Orbax round trip through the trainer's Composite layout
      (``_build_checkpoint_manager`` / ``_save_checkpoint``) on ``tmp_path``, with the
      in-memory params deliberately corrupted before restore so "restored == SAVED"
      is a real claim (not "== init", not "== corrupted");
  (b) the ``checkpoint_step == 0`` bypass -- Orbax is NEVER consulted (the checkpoint
      manager is a spy that fails if built), the state keeps its freshly-built
      (pretrained-stand-in) params, and the reported step is 0;
  (c) that the full-FT rollout actually CONSUMES the restored params -- the recording
      stub transformer's velocity depends on its ``Param``, so the rollout output
      after restore matches a rollout at the saved param value and DIFFERS from the
      init-param rollout.

Plus the restore-path negative: a requested ``checkpoint_step=N`` that names no saved
checkpoint raises an actionable error (naming N and the available steps), and an empty
dir raises the mode-correct "No full-FT/adapter checkpoints found" message (F4).

The strengthen phase (Codex round-4 review) added:
  * F1 -- in cohort mode the restored JSON step is written INTO ``FullFTTrainState.step``
    (asserted beside the returned scalar), completing the generic params/opt_state/STEP
    restore of the round-3 binding note;
  * F2 -- behavioral coverage of the PRODUCTION ``_build_full_ft_validation_state``
    (stub transformer through the real builder: genuine rollout-ready ``FullFTTrainState``,
    real optimizer, ``_build_adapters`` never consulted) and the PRODUCTION
    ``_restore_validation_state`` dispatcher (full-FT arm selects the full-FT builder,
    never the adapter one, and the manager-free step-0 bypass holds THROUGH the
    production call site -- killing a ``cohort_mode=False`` call-site mutant; the
    adapter arm selects the adapter builder and passes ``cohort_mode=False``);
  * F4 -- the cohort-mode empty-dir message says "full-FT" while the adapter message
    stays byte-identical, both pinned exactly.

CPU-only: a tiny ``nnx.Module`` stub transformer with a real ``Param`` and a real
``optax.adamw`` -- no 5B weights, no pipeline, no mesh. The darwin grain import stub
lives in ``conftest.py``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.trainers.wan_ti2v_full_ft_trainer as full_ft
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

_INIT_GAIN = 0.5


class _CkptStubTransformer(nnx.Module):
    """Stand-in for the (trainable) WAN transformer: a real ``Param`` (``gain``) whose
    value both round-trips through the checkpoint AND drives the rollout velocity, so
    "the rollout consumed the restored param" is provable from the rollout output."""

    def __init__(self, gain=_INIT_GAIN):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        # velocity shaped like hidden_states [b, c, f, h, w]; depends on gain.
        return self.gain[...] * kwargs["hidden_states"].astype(jnp.float32)


def _make_state(gain, tx):
    t = _CkptStubTransformer(gain)
    graphdef, params, rest = nnx.split(t, nnx.Param, ...)
    null_context = jnp.zeros((1, 2, 8), dtype=jnp.float32)
    return full_ft.FullFTTrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx,
        graphdef=graphdef,
        rest_of_state=rest,
        null_context=null_context,
    )


def _ckpt_config(*, checkpoint_step, ckpt_dir):
    return SimpleNamespace(
        model_type="FULL_FT_TI2V",
        checkpoint_step=checkpoint_step,
        checkpoint_dir=ckpt_dir,
        output_dir="",
        run_name="",
        checkpoint_keep_period=-1,
    )


def _first_leaf(tree):
    return np.asarray(jax.tree_util.tree_leaves(tree)[0])


def _rollout_inputs():
    b, c, f, h, w = 1, 2, 3, 4, 4
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    data = {
        "z_i0": jax.random.normal(k1, (b, c, 1, h, w), dtype=jnp.float32),
        "z_video": jax.random.normal(k2, (b, c, f, h, w), dtype=jnp.float32),
        "actions": jax.random.normal(k3, (b, 32, 7), dtype=jnp.float32),
    }
    config = SimpleNamespace(
        model_type="FULL_FT_TI2V",
        weights_dtype="float32",
        side_adapter_sampling_steps=2,
        flow_shift=5.0,
        side_adapter_guide_scale=1.0,
    )
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=5.0, sigma_min=0.0, sigma_max=1.0)
    return data, config, scheduler, jax.random.key(0)


# (1) Generic params/opt_state/step round trip into FullFTTrainState (review note a).


def test_ckpt_roundtrip_restores_saved_params_optstate_and_step(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    step_value = 7
    trainer = full_ft.WanTI2VFullFTTrainer(_ckpt_config(checkpoint_step=step_value, ckpt_dir=ckpt_dir))

    # A saved state whose params AND optimizer moments have MOVED off their init values
    # (adamw init mu/nu are zeros, so one grad step makes "moments restored" non-vacuous).
    state = _make_state(_INIT_GAIN, optax.adamw(0.1))
    state = state.apply_gradients(grads=jax.tree.map(jnp.ones_like, state.params))
    saved_params = jax.tree.map(np.asarray, state.params)
    saved_mu, saved_nu = full_ft._adam_moment_trees(state.opt_state)
    saved_mu0, saved_nu0 = _first_leaf(saved_mu), _first_leaf(saved_nu)
    assert saved_mu0 != 0.0 and saved_nu0 != 0.0  # non-vacuity: saved moments are nonzero
    assert not np.array_equal(_first_leaf(saved_params), np.asarray(_INIT_GAIN))  # param moved off init

    mgr = trainer._build_checkpoint_manager(ckpt_dir)
    trainer._save_checkpoint(mgr, step_value, state)
    mgr.wait_until_finished()
    mgr.close()

    # DELIBERATELY corrupt the in-memory params before restore -- restore must overwrite.
    state = state.replace(params=jax.tree.map(lambda p: p + 100.0, state.params))
    assert not np.allclose(_first_leaf(state.params), _first_leaf(saved_params))

    # Reconstruct a FRESH state (init gain, ZERO moments, step 0) and restore into it.
    fresh = _make_state(_INIT_GAIN, optax.adamw(0.1))
    assert _first_leaf(full_ft._adam_moment_trees(fresh.opt_state)[0]) == 0.0  # zeros pre-restore
    assert int(fresh.step) == 0  # non-vacuity for F1: the fresh state starts at step 0

    config = _ckpt_config(checkpoint_step=step_value, ckpt_dir=ckpt_dir)
    restored, restored_step = gen._restore_checkpoint_state(config, fresh, ckpt_dir, cohort_mode=True)

    # Params are the SAVED values -- not the corrupted (+100) copy, not the init 0.5.
    for r, s in zip(jax.tree_util.tree_leaves(restored.params), jax.tree_util.tree_leaves(saved_params)):
        np.testing.assert_array_equal(np.asarray(r), np.asarray(s))
    assert not np.allclose(_first_leaf(restored.params), np.asarray(_INIT_GAIN))
    # Optimizer moments restored to the saved (nonzero) ones, not the fresh zeros.
    r_mu, r_nu = full_ft._adam_moment_trees(restored.opt_state)
    np.testing.assert_array_equal(_first_leaf(r_mu), saved_mu0)
    np.testing.assert_array_equal(_first_leaf(r_nu), saved_nu0)
    assert _first_leaf(r_mu) != 0.0
    # Step restored -- BOTH the returned scalar and (F1, round-3 binding note: the
    # generic restore covers params/opt_state/STEP) the state's own step field.
    assert restored_step == step_value
    assert int(restored.step) == step_value


# (2) The full-FT rollout consumes the RESTORED params, not the init ones (review note c).


def test_rollout_consumes_restored_params_not_init(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    trainer = full_ft.WanTI2VFullFTTrainer(_ckpt_config(checkpoint_step=5, ckpt_dir=ckpt_dir))

    saved = _make_state(_INIT_GAIN, optax.adamw(0.3))
    saved = saved.apply_gradients(grads=jax.tree.map(jnp.ones_like, saved.params))
    saved_gain = float(_first_leaf(saved.params))
    assert saved_gain != _INIT_GAIN  # non-vacuity: saved gain differs from init

    mgr = trainer._build_checkpoint_manager(ckpt_dir)
    trainer._save_checkpoint(mgr, 5, saved)
    mgr.wait_until_finished()
    mgr.close()

    fresh = _make_state(_INIT_GAIN, optax.adamw(0.3))  # gain == init 0.5, zero moments
    # cohort_mode deliberately left at its default (False) here: this test doubles as
    # coverage of the shared non-cohort branch (requested > 0) restoring into a
    # FullFTTrainState; the cohort branch is pinned by the round-trip test above.
    restored, _ = gen._restore_checkpoint_state(_ckpt_config(checkpoint_step=5, ckpt_dir=ckpt_dir), fresh, ckpt_dir)

    data, config, scheduler, rng = _rollout_inputs()
    z_restored, _ = gen._rollout_sample(restored, data, rng, scheduler, config)
    z_init, _ = gen._rollout_sample(fresh, data, rng, scheduler, config)  # gain == init
    z_saved_gain, _ = gen._rollout_sample(_make_state(saved_gain, optax.adamw(0.3)), data, rng, scheduler, config)

    # The rollout output moved OFF the init-param output (restore actually took effect)...
    assert not np.allclose(np.asarray(z_restored), np.asarray(z_init))
    # ...and matches a rollout whose param is EXACTLY the saved gain (pinning the value).
    np.testing.assert_allclose(np.asarray(z_restored), np.asarray(z_saved_gain), rtol=1e-6, atol=0)
    assert np.all(np.isfinite(np.asarray(z_restored)))


# (3) checkpoint_step == 0 bypasses Orbax entirely (review note b).


def test_checkpoint_step_zero_bypasses_orbax_entirely(monkeypatch):
    consulted = []

    class _SpyManager:
        def __init__(self, *args, **kwargs):
            consulted.append("constructed")

        def latest_step(self):
            consulted.append("latest_step")
            return 999

        def all_steps(self):
            consulted.append("all_steps")
            return [999]

        def restore(self, *args, **kwargs):
            raise AssertionError("Orbax restore called under checkpoint_step=0 bypass")

    monkeypatch.setattr(gen.ocp, "CheckpointManager", _SpyManager)
    fresh = _make_state(_INIT_GAIN, optax.adamw(0.1))
    config = _ckpt_config(checkpoint_step=0, ckpt_dir="/does/not/exist")

    restored, step = gen._restore_checkpoint_state(config, fresh, "/does/not/exist", cohort_mode=True)

    assert step == 0  # step-0 baseline
    assert restored is fresh  # state returned UNCHANGED -- the pretrained stand-in
    assert float(_first_leaf(restored.params)) == _INIT_GAIN  # init params kept, not restored
    assert consulted == []  # the checkpoint manager was NEVER built or consulted


# (4) Restore-path negatives: a requested step with no checkpoint is actionable.


def test_missing_requested_step_raises_actionable_error(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    trainer = full_ft.WanTI2VFullFTTrainer(_ckpt_config(checkpoint_step=7, ckpt_dir=ckpt_dir))
    state = _make_state(_INIT_GAIN, optax.adamw(0.1))
    mgr = trainer._build_checkpoint_manager(ckpt_dir)
    trainer._save_checkpoint(mgr, 7, state)  # only step 7 exists
    mgr.wait_until_finished()
    mgr.close()

    fresh = _make_state(_INIT_GAIN, optax.adamw(0.1))
    config = _ckpt_config(checkpoint_step=999, ckpt_dir=ckpt_dir)
    with pytest.raises(ValueError) as ei:
        gen._restore_checkpoint_state(config, fresh, ckpt_dir, cohort_mode=True)
    msg = str(ei.value)
    assert "999" in msg  # names the requested (missing) step
    assert "7" in msg  # names the available step(s) seen
    assert ckpt_dir in msg  # actionable: names the directory


def test_no_checkpoints_at_all_raises_full_ft_message_in_cohort_mode(tmp_path):
    # (F4) The cohort-mode message must name the actual mode: "full-FT", not "adapter".
    empty = str(tmp_path / "empty")
    os.makedirs(empty, exist_ok=True)
    fresh = _make_state(_INIT_GAIN, optax.adamw(0.1))
    config = _ckpt_config(checkpoint_step=-1, ckpt_dir=empty)
    with pytest.raises(ValueError, match=r"No full-FT checkpoints found in"):
        gen._restore_checkpoint_state(config, fresh, empty, cohort_mode=True)
    with pytest.raises(ValueError) as ei:
        gen._restore_checkpoint_state(config, fresh, empty, cohort_mode=True)
    assert str(ei.value) == f"No full-FT checkpoints found in {empty}"  # exact message


def test_no_checkpoints_adapter_message_unchanged_outside_cohort_mode(tmp_path):
    # (F4 guard) Outside cohort mode the historical adapter message is byte-identical.
    empty = str(tmp_path / "empty")
    os.makedirs(empty, exist_ok=True)
    fresh = _make_state(_INIT_GAIN, optax.adamw(0.1))
    config = _ckpt_config(checkpoint_step=-1, ckpt_dir=empty)
    with pytest.raises(ValueError) as ei:
        gen._restore_checkpoint_state(config, fresh, empty)
    assert str(ei.value) == f"No adapter checkpoints found in {empty}"  # exact HEAD message


# --------------------------------------------------------------------------------------
# (5) F2: the PRODUCTION full-FT builder and _restore_validation_state dispatcher.
# --------------------------------------------------------------------------------------


class _FakeMesh:
    """Context-manager mesh stand-in; the real mesh is only entered by the builder."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _builder_config():
    # Everything _build_full_ft_validation_state consumes when the pipeline-load and
    # _shard_state seams are stubbed: axis rules + the REAL optimizer factory's keys.
    return SimpleNamespace(
        model_type="FULL_FT_TI2V",
        logical_axis_rules=(),
        max_train_steps=4,
        learning_rate=1.0e-5,
        learning_rate_schedule_steps=-1,
        warmup_steps_fraction=0.5,
        adam_b1=0.9,
        adam_b2=0.999,
        adam_eps=1.0e-8,
        adam_weight_decay=1.0e-2,
        opt_enable_grad_global_norm_clipping=True,
        max_grad_norm=1.0,
        opt_enable_grad_clipping=False,
        max_grad_value=1.0,
    )


def test_build_full_ft_validation_state_produces_rollout_ready_state(monkeypatch):
    # (F2) Run the PRODUCTION builder with a stub transformer behind the pipeline-load
    # seam: it must produce a genuine FullFTTrainState (transformer split as params,
    # REAL optimizer state, the computed null context), never touch _build_adapters,
    # route the created state through the trainer's _shard_state, drop the pipeline's
    # heavy modules -- and the built state must drive the full-FT rollout.
    stub_gain = 0.7
    fake_null = jnp.full((1, 2, 8), 3.0, dtype=jnp.float32)
    pipeline = SimpleNamespace(
        mesh=_FakeMesh(),
        transformer=_CkptStubTransformer(stub_gain),
        text_encoder=object(),
        tokenizer=object(),
    )
    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_load_wan_pipeline", lambda self: pipeline)
    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_compute_null_context", lambda self, p, m: fake_null)

    def _no_adapters(self, transformer):
        raise AssertionError("_build_adapters consulted in the full-FT validation build")

    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_build_adapters", _no_adapters)
    shard_seen = {}

    def _spy_shard(self, mesh, state):
        shard_seen["state"] = state
        return state, "SHARDINGS"

    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_shard_state", _spy_shard)

    trainer, out_pipeline, out_mesh, state, shardings = gen._build_full_ft_validation_state(_builder_config())

    assert isinstance(trainer, full_ft.WanTI2VFullFTTrainer)
    assert out_pipeline is pipeline and out_mesh is pipeline.mesh
    assert shardings == "SHARDINGS"
    # A genuine FullFTTrainState: transformer params ARE the trainable params, no
    # adapter-side fields exist, and it is the object _shard_state saw.
    assert isinstance(state, full_ft.FullFTTrainState)
    assert not hasattr(state, "transformer_params")  # not the adapter TrainState shape
    assert state is shard_seen["state"]
    assert _first_leaf(state.params) == np.float32(stub_gain)  # the stub transformer's Param
    assert state.null_context is fake_null
    assert full_ft._adam_moment_trees(state.opt_state)[0] is not None  # REAL adamw state
    # Memory discipline mirrored from the adapter builder: heavy modules dropped.
    assert not hasattr(out_pipeline, "transformer")
    assert not hasattr(out_pipeline, "text_encoder") and not hasattr(out_pipeline, "tokenizer")
    # And the built state is ROLLOUT-READY through the full-FT branch.
    data, config, scheduler, rng = _rollout_inputs()
    z, metrics = gen._rollout_sample(state, data, rng, scheduler, config)
    assert np.all(np.isfinite(np.asarray(z))) and "latent_mse" in metrics


def test_restore_validation_state_full_ft_step0_holds_through_production_path(monkeypatch):
    # (F2) THE production-call-site proof: dispatcher selects the full-FT builder (the
    # adapter builder is booby-trapped), and with checkpoint_step=0 the REAL
    # _restore_checkpoint_state runs inside it in cohort mode -- so Orbax is never
    # consulted (manager construction is booby-trapped) and the built state comes back
    # unchanged at step 0. A `cohort_mode=False` mutant at the call site would build
    # the manager and fail here.
    built_state = _make_state(_INIT_GAIN, optax.adamw(0.1))
    built = ("TRAINER", "PIPELINE", "MESH", built_state, "SHARDINGS")
    builder_calls = []

    def _fake_full_ft_builder(config):
        builder_calls.append(config)
        return built

    monkeypatch.setattr(gen, "_build_full_ft_validation_state", _fake_full_ft_builder)

    def _adapter_builder(config):
        raise AssertionError("adapter builder selected for model_type=FULL_FT_TI2V")

    monkeypatch.setattr(gen, "_build_side_adapter_validation_state", _adapter_builder)

    class _NoManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Orbax CheckpointManager built under the step-0 bypass")

    monkeypatch.setattr(gen.ocp, "CheckpointManager", _NoManager)

    config = _ckpt_config(checkpoint_step=0, ckpt_dir="/fake/ckpts")
    trainer, pipeline, mesh, state, shardings, step = gen._restore_validation_state(config)

    assert builder_calls == [config]  # the full-FT builder got the exact config, once
    assert (trainer, pipeline, mesh, shardings) == ("TRAINER", "PIPELINE", "MESH", "SHARDINGS")
    assert state is built_state  # unchanged pretrained stand-in
    assert float(_first_leaf(state.params)) == _INIT_GAIN
    assert step == 0


def test_restore_validation_state_adapter_arm_uses_adapter_builder_non_cohort(monkeypatch):
    # (F2 guard) For a non-full-FT model_type the dispatcher selects the ADAPTER
    # builder (full-FT builder booby-trapped) and calls the restore helper with
    # cohort_mode=False and the config-derived ckpt_dir -- the adapter semantics of
    # round 4's hard rule, now pinned at the production call site.
    built = ("TRAINER", "PIPELINE", "MESH", "STATE", "SHARDINGS")
    monkeypatch.setattr(gen, "_build_side_adapter_validation_state", lambda config: built)

    def _full_ft_builder(config):
        raise AssertionError("full-FT builder selected for an adapter model_type")

    monkeypatch.setattr(gen, "_build_full_ft_validation_state", _full_ft_builder)
    seen = {}

    def _spy_restore(config, state, ckpt_dir, *, cohort_mode=False):
        seen["cohort_mode"] = cohort_mode
        seen["state"] = state
        seen["ckpt_dir"] = ckpt_dir
        return "RESTORED_STATE", 42

    monkeypatch.setattr(gen, "_restore_checkpoint_state", _spy_restore)

    config = SimpleNamespace(
        model_type="SIDE_ADAPTER_TI2V", checkpoint_dir="/adapter/ckpts", output_dir="", run_name=""
    )
    out = gen._restore_validation_state(config)

    assert out == ("TRAINER", "PIPELINE", "MESH", "RESTORED_STATE", "SHARDINGS", 42)
    assert seen["cohort_mode"] is False  # adapter arm NEVER runs in cohort mode
    assert seen["state"] == "STATE"
    assert seen["ckpt_dir"] == "/adapter/ckpts"
