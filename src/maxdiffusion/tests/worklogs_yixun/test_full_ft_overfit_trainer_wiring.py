"""CPU-only wiring tests for the full-FT overfit trainer (exp_01 round 3).

These pin the *trainer plumbing* around the round-2 objective without loading any
5B weights, pipeline, or mesh: the probe-config asserts (guide scale exactly 1.0
with non-finite rejected, noise mode == "fresh"), the shard-selection that -- unlike
the side-adapter parent -- KEEPS the computed FSDP shardings for params/opt_state,
the large-leaf sharding audit (top-k selection, PATH-MATCHED Adam mu/nu twins, the
>100 MB-replicated raise), physical per-host shard byte accounting (no replica
dedupe), the per-dtype startup summaries that distinguish the mixed-dtype primary
run from the fp32 control (plan §2.4), and the ``FULL_FT_TI2V`` dispatch.

The strengthen phase (Codex round-3 review) added BEHAVIORAL coverage (F4) beside
the original pure/source checks: validation-before-load ordering via a raising
pipeline-load spy, the ``_shard_state`` dataflow via object-identity spies (the
helper's output IS what gets audited/placed/returned), the integrated audit raise
propagating out of ``_shard_state``, and an executed ``train_wan.train`` dispatch
onto a recording fake trainer. Still no ``WanPipeline`` build anywhere. The darwin
grain import stub lives in ``conftest.py``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.sharding import PartitionSpec as P

import maxdiffusion.trainers.wan_ti2v_full_ft_trainer as full_ft
import maxdiffusion.trainers.wan_ti2v_side_adapter_trainer as trainer_mod
from maxdiffusion import max_utils

# PartitionSpec / NamedSharding are already jax leaves; keep an explicit is_leaf so
# spec trees never get flattened into their axis names during comparisons.
_IS_SPEC = lambda x: isinstance(x, (P, jax.sharding.NamedSharding))  # noqa: E731


def _sds(shape, dtype=jnp.float32):
    """A weightless stand-in leaf carrying only shape+dtype (no HBM/host alloc)."""
    return jax.ShapeDtypeStruct(shape, dtype)


class _FakeShard:
    """An addressable-shard stand-in (index + .data.shape), as jax.Array exposes."""

    def __init__(self, index, shape):
        self.index = index
        self.data = SimpleNamespace(shape=shape)


class _FakeShardedLeaf:
    """Array-like leaf with controllable addressable_shards for byte accounting."""

    def __init__(self, shape, shards):
        self.shape = shape
        self.dtype = jnp.dtype(jnp.float32)
        self.addressable_shards = shards


class _FakeMesh:
    """Context-manager mesh stand-in: only ``with mesh`` + ``.size`` are consumed."""

    def __init__(self, size=1):
        self.size = size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _wiring_config(**overrides):
    base = {
        "side_adapter_guide_scale": 1.0,
        "side_adapter_noise_mode": "fresh",
        "logical_axis_rules": (),  # nn_partitioning.axis_rules accepts an empty rule set
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------------------
# (1)+(2) probe-config asserts: guide scale must be exactly 1.0 (§2.1, F5: non-finite
# rejected), noise must be "fresh" (F1-BLOCKER from the plan).
# --------------------------------------------------------------------------------------


def _probe_config(guide_scale=1.0, noise_mode="fresh"):
    return SimpleNamespace(side_adapter_guide_scale=guide_scale, side_adapter_noise_mode=noise_mode)


def test_validate_probe_config_passes_at_guide_scale_1_and_fresh():
    # Must NOT raise for the sanctioned probe recipe.
    full_ft.WanTI2VFullFTTrainer._validate_probe_config(_probe_config(guide_scale=1.0, noise_mode="fresh"))


def test_validate_probe_config_rejects_guide_scale_not_1():
    with pytest.raises((ValueError, AssertionError), match=r"2\.1"):
        full_ft.WanTI2VFullFTTrainer._validate_probe_config(_probe_config(guide_scale=5.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 1.0000001])
def test_validate_probe_config_rejects_nonfinite_and_near_one_guide_scale(bad):
    # F5 regression: NaN slipped through the old abs(gs-1)>1e-6 tolerance (NaN>x is
    # False). The guard now rejects non-finite explicitly and compares exactly to 1.0.
    with pytest.raises(ValueError, match=r"2\.1"):
        full_ft.WanTI2VFullFTTrainer._validate_probe_config(_probe_config(guide_scale=bad))


def test_validate_probe_config_rejects_non_fresh_noise():
    # F1 BLOCKER: fixed noise looks fine on train loss but breaks generations.
    with pytest.raises((ValueError, AssertionError), match="fresh"):
        full_ft.WanTI2VFullFTTrainer._validate_probe_config(_probe_config(noise_mode="fixed"))


def test_start_training_validates_before_pipeline_load(monkeypatch):
    # (F4-i) BEHAVIORAL ordering: with a bad guide scale, start_training must abort in
    # _validate_probe_config BEFORE any pipeline/weights load. The load spy raises a
    # DIFFERENT error type, so reordering would surface RuntimeError, not ValueError.
    def _spy_load(self):
        raise RuntimeError("pipeline load reached before probe-config validation")

    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_load_wan_pipeline", _spy_load)
    trainer = full_ft.WanTI2VFullFTTrainer(_wiring_config(side_adapter_guide_scale=5.0))
    with pytest.raises(ValueError, match=r"2\.1"):
        trainer.start_training()


# --------------------------------------------------------------------------------------
# (3) shard-selection: full-FT KEEPS the computed shardings (no replicate replacement);
# behavioral dataflow through _shard_state proves the helper's output is what gets
# audited, placed, and returned (F4-ii/iii), beside the original pure/source checks.
# --------------------------------------------------------------------------------------


def test_full_ft_state_shardings_keeps_computed_not_replicated():
    sharded = P("fsdp")
    computed = full_ft.FullFTTrainState(
        step=P(),
        apply_fn=None,
        params={"w": sharded, "b": sharded},
        tx=None,
        opt_state=({"mu": {"w": sharded}, "nu": {"w": sharded}},),
        graphdef=None,
        rest_of_state=None,
        null_context=P(),
    )
    out = full_ft._full_ft_state_shardings(computed, computed)
    # KEPT: params/opt_state specs are exactly the computed (sharded) ones.
    assert jax.tree_util.tree_leaves(out.params, is_leaf=_IS_SPEC) == [sharded, sharded]
    assert all(s == sharded for s in jax.tree_util.tree_leaves(out.opt_state, is_leaf=_IS_SPEC))
    # CONTRAST: the parent's replacement would have replicated them to P().
    parent_style = trainer_mod._replicated_sharding_tree(computed.params, P())
    assert jax.tree_util.tree_leaves(parent_style, is_leaf=_IS_SPEC) == [P(), P()]
    assert jax.tree_util.tree_leaves(out.params, is_leaf=_IS_SPEC) != [P(), P()]


def test_shard_state_retains_actual_tpu_and_audits_without_replicating():
    src = inspect.getsource(full_ft.WanTI2VFullFTTrainer._shard_state)
    # Real path keeps the actual-TPU sharding and the no-replicate selection...
    assert "_apply_actual_sharding_for_tpu" in src
    assert "_full_ft_state_shardings" in src
    # ...runs the large-leaf audit...
    assert "audit_large_leaves" in src or "_log_and_audit_shardings" in src
    # ...and, unlike the parent, never replicates params/opt_state.
    assert "_replicated_sharding_tree" not in src


def test_shard_state_dataflow_helper_output_is_audited_and_placed(monkeypatch):
    # (F4-ii) BEHAVIORAL dataflow, asserted on object identity rather than tokens:
    # computed (nnx.get_named_sharding) -> _apply_actual_sharding_for_tpu -> the
    # keep-computed helper -> ONE shardings object seen by the audit, _to_target_if_cpu,
    # jax.device_put, and the return -- still carrying the computed FSDP specs.
    state = full_ft.FullFTTrainState(
        step=0,
        apply_fn=None,
        params={"w": _sds((4, 4))},
        tx=None,
        opt_state={"mu": _sds((4, 4))},
        graphdef=None,
        rest_of_state=None,
        null_context=None,
    )
    computed = state.replace(step=P(), params={"w": P("fsdp")}, opt_state={"mu": P("fsdp")})
    seen = {}

    monkeypatch.setattr(full_ft.nnx, "get_named_sharding", lambda s, m: computed)

    def _spy_actual(s, c):
        seen["actual_in"] = c
        return c

    monkeypatch.setattr(full_ft, "_apply_actual_sharding_for_tpu", _spy_actual)

    def _spy_audit(self, s, shardings, mesh):
        seen["audited"] = shardings

    monkeypatch.setattr(full_ft.WanTI2VFullFTTrainer, "_log_and_audit_shardings", _spy_audit)

    def _spy_to_target(s, shardings):
        seen["to_target"] = shardings
        return s

    monkeypatch.setattr(full_ft, "_to_target_if_cpu", _spy_to_target)

    def _spy_device_put(s, shardings):
        seen["placed"] = shardings
        seen["placed_state"] = s
        return ("PLACED", s)

    monkeypatch.setattr(full_ft.jax, "device_put", _spy_device_put)

    trainer = full_ft.WanTI2VFullFTTrainer(_wiring_config())
    out_state, out_shardings = trainer._shard_state(_FakeMesh(size=8), state)

    # The computed shardings flowed into the actual-TPU retention step...
    assert seen["actual_in"] is computed
    # ...and ONE post-helper shardings object reaches audit, host-transfer, placement,
    # and the return -- whose params/opt_state ARE the computed (FSDP, not P()) specs.
    assert seen["audited"] is seen["to_target"] is seen["placed"] is out_shardings
    assert out_shardings.params is computed.params
    assert out_shardings.opt_state is computed.opt_state
    assert out_shardings.params["w"] == P("fsdp")
    # The state placed is the one _to_target_if_cpu returned; the device_put result is
    # what _shard_state hands back.
    assert seen["placed_state"] is state
    assert out_state[0] == "PLACED" and out_state[1] is state


def test_shard_state_propagates_audit_raise_on_replicated_large_leaf(monkeypatch):
    # (F4-iii) INTEGRATED: a >100MB fully-replicated param on a multi-device mesh must
    # raise out of _shard_state itself (real _apply_actual_sharding_for_tpu, real
    # helper, real audit) -- before any device placement can happen.
    state = full_ft.FullFTTrainState(
        step=0,
        apply_fn=None,
        params={"big": _sds((6000, 6000))},  # 137 MB fp32
        tx=None,
        opt_state={},
        graphdef=None,
        rest_of_state=None,
        null_context=None,
    )
    computed = state.replace(step=P(), params={"big": P()}, opt_state={})
    monkeypatch.setattr(full_ft.nnx, "get_named_sharding", lambda s, m: computed)

    def _fail_if_placed(*args, **kwargs):
        raise AssertionError("device_put reached: the audit failed to raise")

    monkeypatch.setattr(full_ft.jax, "device_put", _fail_if_placed)
    trainer = full_ft.WanTI2VFullFTTrainer(_wiring_config())
    with pytest.raises(ValueError, match="replicated"):
        trainer._shard_state(_FakeMesh(size=64), state)


# --------------------------------------------------------------------------------------
# (4) large-leaf sharding audit: top-k selection by bytes, path-matched Adam twins (F3),
# physical shard-byte accounting (F2), and the >100 MB-replicated raise.
# --------------------------------------------------------------------------------------


def test_audit_selects_top8_params_by_bytes():
    # 10 leaves of strictly increasing size; top_k=8 keeps the 8 largest, descending.
    params = {f"p{i:02d}": _sds((i + 1, 100, 100)) for i in range(10)}
    specs = {"params": {k: P("fsdp") for k in params}, "opt_state": {}}
    out = full_ft.audit_large_leaves(params, {}, specs, mesh_size=1, top_k=8)
    leading = [e["shape"][0] for e in out["params"]]
    assert leading == [10, 9, 8, 7, 6, 5, 4, 3]
    top = out["params"][0]
    assert top["shape"] == (10, 100, 100)
    assert top["dtype"] == "float32"
    assert top["bytes"] == 10 * 100 * 100 * 4
    assert top["spec"] == P("fsdp")
    # No ScaleByAdamState in an empty opt tree -> no twins (never invented).
    assert out["opt_twins"] == []


def test_audit_opt_twins_are_path_matched_not_magnitude_matched():
    # (F3) The twins must be the mu AND nu of each selected param, path-matched inside
    # the real ScaleByAdamState. The opt tree is skewed to DEFEAT magnitude matching:
    # the globally-largest opt leaves are a non-moment "extra" leaf (36 MB) and the
    # SMALL param's mu (16 MB) -- magnitude sorting would return those, path matching
    # must return big's mu/nu (4 MB each) instead.
    params = {"big": _sds((1000, 1000)), "small": _sds((10,))}
    adam = optax.ScaleByAdamState(
        count=_sds((), jnp.int32),
        mu={"big": _sds((1000, 1000)), "small": _sds((2000, 2000))},  # small's mu skewed huge
        nu={"big": _sds((1000, 1000)), "small": _sds((10,))},
    )
    opt = (adam, {"extra_huge": _sds((3000, 3000))})  # non-moment state dwarfs everything
    specs = {
        "params": {"big": P("fsdp"), "small": P()},
        "opt_state": (
            optax.ScaleByAdamState(
                count=P(),
                mu={"big": P("fsdp"), "small": P("fsdp")},
                nu={"big": P("fsdp"), "small": P()},
            ),
            {"extra_huge": P("fsdp")},
        ),
    }
    out = full_ft.audit_large_leaves(params, opt, specs, mesh_size=64, top_k=1)
    assert [e["path"] for e in out["params"]] == ["['big']"]
    # Exactly big's two moments, labeled, in mu-then-nu order -- nothing magnitude-picked.
    assert [(e["path"], e["moment"]) for e in out["opt_twins"]] == [("['big']", "mu"), ("['big']", "nu")]
    twin_shapes = {e["shape"] for e in out["opt_twins"]}
    assert (2000, 2000) not in twin_shapes and (3000, 3000) not in twin_shapes
    assert twin_shapes == {(1000, 1000)}
    # And each twin carries its own path-matched spec from the mirrored spec tree.
    assert [e["spec"] for e in out["opt_twins"]] == [P("fsdp"), P("fsdp")]


def test_audit_raises_on_large_replicated_param_multidevice():
    # 6000x6000 f32 = ~137 MB > 100 MB threshold, fully replicated on a 64-device mesh.
    params = {"big": _sds((6000, 6000))}
    specs = {"params": {"big": P()}, "opt_state": {}}
    with pytest.raises(ValueError, match="replicated"):
        full_ft.audit_large_leaves(params, {}, specs, mesh_size=64)


def test_audit_raises_on_large_replicated_opt_leaf_multidevice():
    # The param is safely sharded, but a huge optimizer leaf is replicated. The raise
    # scans ALL opt leaves (kept as-is under F3 -- not only the path-matched twins).
    params = {"w": _sds((10, 10))}
    opt = {"mu": _sds((6000, 6000))}
    specs = {"params": {"w": P("fsdp")}, "opt_state": {"mu": P()}}
    with pytest.raises(ValueError, match="replicated"):
        full_ft.audit_large_leaves(params, opt, specs, mesh_size=64)


def test_audit_passes_when_sharded_or_single_device():
    big = {"big": _sds((6000, 6000))}
    sharded = {"params": {"big": P("fsdp")}, "opt_state": {}}
    replicated = {"params": {"big": P()}, "opt_state": {}}
    # Sharded on a multi-device mesh: fine.
    full_ft.audit_large_leaves(big, {}, sharded, mesh_size=64)
    # Replicated but mesh spans a single device: nothing to shard over, so fine.
    full_ft.audit_large_leaves(big, {}, replicated, mesh_size=1)


def test_audit_ignores_small_replicated_leaf():
    # A tiny replicated leaf (<100 MB) must NOT trip the audit even on 64 devices.
    params = {"small": _sds((10, 10))}
    specs = {"params": {"small": P()}, "opt_state": {}}
    out = full_ft.audit_large_leaves(params, {}, specs, mesh_size=64)
    assert out["params"][0]["shape"] == (10, 10)


def test_leaf_bytes_sums_all_addressable_shards_without_index_dedupe():
    # (F2) Two REPLICAS -- identical shard index -- both occupy HBM: physical must be
    # 80 for a 40-byte logical leaf (the old index-dedupe reported 40).
    replicated = _FakeShardedLeaf((10,), [_FakeShard((slice(None),), (10,)), _FakeShard((slice(None),), (10,))])
    assert full_ft._leaf_bytes(replicated) == (40, 80)
    # Two DISTINCT shards of 5 elements: physical equals logical (40) -- the sum walks
    # shard shapes, it does not multiply by device count.
    sharded = _FakeShardedLeaf((10,), [_FakeShard((slice(0, 5),), (5,)), _FakeShard((slice(5, 10),), (5,))])
    assert full_ft._leaf_bytes(sharded) == (40, 40)
    # No addressable_shards (ShapeDtypeStruct / host array): falls back to logical.
    assert full_ft._leaf_bytes(_sds((10,))) == (40, 40)
    # And the tree totals aggregate the physical number.
    assert full_ft._tree_byte_totals({"x": replicated}) == (40, 80)


# --------------------------------------------------------------------------------------
# (5) fp32-control precondition (plan §2.4): per-dtype summaries -- NOT a sampled leaf
# (F1) -- distinguish the mixed-dtype primary run from the fp32 control, and the Adam
# moments follow their param's dtype leaf-by-leaf. Uses the real max_utils optimizer.
# --------------------------------------------------------------------------------------


def _opt_config():
    return SimpleNamespace(
        adam_b1=0.9,
        adam_b2=0.999,
        adam_eps=1.0e-8,
        adam_weight_decay=1.0e-2,
        opt_enable_grad_global_norm_clipping=True,
        max_grad_norm=1.0,
        opt_enable_grad_clipping=False,
        max_grad_value=1.0,
    )


def _real_opt_state(params):
    tx = max_utils.create_optimizer(_opt_config(), optax.constant_schedule(1.0e-5))
    return tx.init(params)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_adam_moments_follow_param_dtype(dtype):
    # Uniform tree: the §2.4 control knob (weights_dtype) drives params AND moments.
    params = {"w": jnp.ones((4, 4), dtype=dtype), "b": jnp.zeros((4,), dtype=dtype)}
    mu, nu = full_ft._adam_moment_trees(_real_opt_state(params))
    expected = jnp.dtype(dtype).name
    assert set(full_ft._dtype_summary(params)) == {expected}
    assert set(full_ft._dtype_summary(mu)) == {expected}
    assert set(full_ft._dtype_summary(nu)) == {expected}


def test_dtype_summary_distinguishes_mixed_primary_from_fp32_control():
    # (F1) Primary-style MIXED tree, shaped like the real loader output: the
    # alphabetically-first leaf is a float32 norm (exactly the case where first-leaf
    # sampling logged float32/float32/float32 and hid the bfloat16 bulk).
    primary = {
        "a_norm_scale": jnp.ones((8,), jnp.float32),  # loader keeps norms f32
        "blocks_attn_w": jnp.ones((8, 8), jnp.bfloat16),  # bulk in weights_dtype
        "blocks_mlp_w": jnp.ones((4, 4), jnp.bfloat16),
    }
    control = jax.tree.map(lambda x: x.astype(jnp.float32), primary)  # fp32-state control
    p_sum = full_ft._dtype_summary(primary)
    c_sum = full_ft._dtype_summary(control)
    # Exact per-dtype (leaf_count, total_bytes), stably ordered by dtype name.
    assert p_sum == {"bfloat16": (2, 8 * 8 * 2 + 4 * 4 * 2), "float32": (1, 8 * 4)}
    assert c_sum == {"float32": (3, 8 * 4 + 8 * 8 * 4 + 4 * 4 * 4)}
    assert list(p_sum) == ["bfloat16", "float32"]
    # The formatted log lines DIFFER -- primary is visibly mixed, control is pure f32.
    assert full_ft._format_dtype_summary(p_sum) != full_ft._format_dtype_summary(c_sum)
    assert "bfloat16" in full_ft._format_dtype_summary(p_sum)
    assert "bfloat16" not in full_ft._format_dtype_summary(c_sum)
    # Real-optax moments mirror each tree's per-dtype breakdown exactly.
    p_mu, p_nu = full_ft._adam_moment_trees(_real_opt_state(primary))
    c_mu, c_nu = full_ft._adam_moment_trees(_real_opt_state(control))
    assert full_ft._dtype_summary(p_mu) == p_sum
    assert full_ft._dtype_summary(p_nu) == p_sum
    assert full_ft._dtype_summary(c_mu) == c_sum
    assert full_ft._dtype_summary(c_nu) == c_sum


def test_moment_dtypes_agree_with_params_per_path():
    # (F1) Leaf-by-leaf: each param's mu and nu carry THAT param's dtype -- pinned on a
    # genuinely mixed real-optax tree, not via any first-leaf shortcut.
    params = {"norm": jnp.ones((3,), jnp.float32), "w": jnp.ones((4, 4), jnp.bfloat16)}
    mu, nu = full_ft._adam_moment_trees(_real_opt_state(params))
    p_leaves = jax.tree_util.tree_leaves_with_path(params)
    mu_by = {jax.tree_util.keystr(p): leaf for p, leaf in jax.tree_util.tree_leaves_with_path(mu)}
    nu_by = {jax.tree_util.keystr(p): leaf for p, leaf in jax.tree_util.tree_leaves_with_path(nu)}
    assert len(p_leaves) == 2
    for path, leaf in p_leaves:
        key = jax.tree_util.keystr(path)
        assert mu_by[key].dtype == leaf.dtype
        assert nu_by[key].dtype == leaf.dtype
    # Non-vacuity: the tree really is mixed, so per-path agreement is a real claim.
    assert {leaf.dtype for _, leaf in p_leaves} == {jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)}


# --------------------------------------------------------------------------------------
# (6) dispatch: FULL_FT_TI2V -> WanTI2VFullFTTrainer -- source-wired AND actually
# executed through train_wan.train (F4-iv).
# --------------------------------------------------------------------------------------


def test_dispatch_full_ft_branch_and_class():
    # Class is importable from the module train_wan references, and subclasses the parent.
    assert issubclass(full_ft.WanTI2VFullFTTrainer, trainer_mod.WanTI2VSideAdapterTrainer)
    train_wan_path = Path(full_ft.__file__).parents[1] / "train_wan.py"
    src = train_wan_path.read_text()
    assert 'config.model_type == "FULL_FT_TI2V"' in src
    assert "WanTI2VFullFTTrainer" in src
    # The token appears inside the dispatch function, mapping the branch to the class.
    dispatch = src[src.index("def train(") :]
    assert "FULL_FT_TI2V" in dispatch and "WanTI2VFullFTTrainer" in dispatch


def test_train_wan_dispatch_executes_full_ft_trainer(monkeypatch):
    # (F4-iv) BEHAVIORAL dispatch: actually run train_wan.train with FULL_FT_TI2V and a
    # recording fake patched in place of the trainer class (train_wan imports it from
    # the full_ft module at call time). It must be instantiated with the exact config
    # object and have start_training called exactly once.
    import maxdiffusion.train_wan as train_wan

    calls = []

    class _FakeTrainer:
        def __init__(self, config):
            calls.append(("init", config))

        def start_training(self):
            calls.append(("start", None))

    monkeypatch.setattr(full_ft, "WanTI2VFullFTTrainer", _FakeTrainer)
    config = SimpleNamespace(model_type="FULL_FT_TI2V")
    train_wan.train(config)
    assert calls == [("init", config), ("start", None)]


def test_full_ft_state_is_used_and_binds_module_steps():
    # start_training must build the round-2 FullFTTrainState and jit the module-level
    # _train_step/_eval_step (bound directly per the round-2 review notes).
    src = inspect.getsource(full_ft.WanTI2VFullFTTrainer.start_training)
    assert "FullFTTrainState.create" in src
    assert "_validate_probe_config" in src
    assert "_train_step" in src and "_eval_step" in src
    # No adapter is built in the full-FT path.
    assert "_build_adapters" not in src
