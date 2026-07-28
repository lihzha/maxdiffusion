"""CPU-only forward-selection + cohort-reader tests for the generate path (round 4).

exp_01 round 4 ("ckpt-generation"), second test file. Two concerns:

(5) FORWARD SELECTION in ``_rollout_sample`` -- for ``model_type == "FULL_FT_TI2V"`` the
    loop body calls the PLAIN transformer (no adapter forward, no ``actions`` kwarg);
    for the side-adapter model_type the adapter path is byte-identical (guard test).
    ``_rollout_sample`` drives the sampling loop with ``jax.lax.fori_loop``, which traces
    the body ONCE, so a Python recording list captures exactly one body invocation --
    that single record tells us WHICH forward the body issues and with what kwargs
    (the meaningful "one plain-transformer call, zero adapter calls per step" contract).

(6) The ``validation_ordinals`` cohort reader (plan §2.3). CONTRACT (chosen + tested):
    a comma-separated list of 0-based dataset POSITIONS (the same coordinate as the
    contiguous ``validation_start_index``, NOT the stored ``ordinal`` field) selects
    exactly those records, IN THE LISTED ORDER (so the per-sample rollout seed, which
    ``run()`` derives by splitting an rng in iteration order, is user-controlled and
    deterministic). Empty / missing key falls back to the contiguous
    ``skip(start_index).take(count)`` read byte-identically. An out-of-range position
    raises an actionable error naming the offending ordinal and the dataset size seen.

(7) Artifact provenance (strengthen F3): ``_validation_config_artifact`` -- the dict
    ``run()`` writes to ``config.json``. Adapter modes stay byte-identical to the HEAD
    artifact (transcribed + compared through the same JSON serialization). FULL_FT_TI2V
    drops the misleading ``action_adapter_type``, records ``model_type`` and the
    RESOLVED ordered ``validation_ordinals`` (whose list order IS the per-sample
    seed-assignment order), and omits ``validation_start_index`` when ordinals are in
    use (it is ignored then) while keeping it for the contiguous fallback.

CPU-only: tiny ``nnx.Module`` stubs, fake in-memory records; no weights/pipeline/mesh.
The darwin grain import stub lives in ``conftest.py``.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.trainers.wan_ti2v_full_ft_trainer as full_ft
import maxdiffusion.trainers.wan_ti2v_side_adapter_trainer as trainer_mod
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

# Module-level so the record survives nnx.split/merge and fori_loop tracing.
_TRANSFORMER_CALLS: list[dict] = []


@pytest.fixture(autouse=True)
def _clear_calls():
    _TRANSFORMER_CALLS.clear()
    yield
    _TRANSFORMER_CALLS.clear()


class _RecordingStubTransformer(nnx.Module):
    """Plain-transformer stand-in: records each call's kwargs, returns velocity."""

    def __init__(self, gain=0.3):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        _TRANSFORMER_CALLS.append(kwargs)
        return self.gain[...] * kwargs["hidden_states"].astype(jnp.float32)


class _TrivialAdapter(nnx.Module):
    """Minimal adapter module so a side-adapter TrainState can be split/merged."""

    def __init__(self):
        self.p = nnx.Param(jnp.zeros((1,), dtype=jnp.float32))


def _rollout_config(*, model_type, guide_scale=1.0):
    return SimpleNamespace(
        model_type=model_type,
        weights_dtype="float32",
        side_adapter_sampling_steps=2,
        flow_shift=5.0,
        side_adapter_guide_scale=guide_scale,
    )


def _rollout_data():
    b, c, f, h, w = 1, 2, 3, 4, 4
    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    return {
        "z_i0": jax.random.normal(k1, (b, c, 1, h, w), dtype=jnp.float32),
        "z_video": jax.random.normal(k2, (b, c, f, h, w), dtype=jnp.float32),
        "actions": jax.random.normal(k3, (b, 32, 7), dtype=jnp.float32),
    }


def _scheduler():
    return FlaxFlowMatchScheduler(dtype=jnp.float32, shift=5.0, sigma_min=0.0, sigma_max=1.0)


# --------------------------------------------------------------------------------------
# (5) Forward selection.
# --------------------------------------------------------------------------------------


def test_full_ft_rollout_calls_plain_transformer_no_adapter_no_actions(monkeypatch):
    adapter_calls = []

    def _spy_adapter(transformer, adapters, **kwargs):
        adapter_calls.append(kwargs)
        return kwargs["hidden_states"] * 0.0

    monkeypatch.setattr(gen, "wan_action_adapter_forward", _spy_adapter)

    t = _RecordingStubTransformer()
    graphdef, params, rest = nnx.split(t, nnx.Param, ...)
    state = full_ft.FullFTTrainState(
        step=0,
        apply_fn=None,
        params=params,
        tx=None,
        opt_state=None,
        graphdef=graphdef,
        rest_of_state=rest,
        null_context=jnp.zeros((1, 2, 8), dtype=jnp.float32),
    )
    z, metrics = gen._rollout_sample(
        state, _rollout_data(), jax.random.key(0), _scheduler(), _rollout_config(model_type="FULL_FT_TI2V")
    )

    # The PLAIN transformer is what the body calls (fori_loop traces the body once)...
    assert len(_TRANSFORMER_CALLS) == 1
    # ...and the adapter forward is NEVER referenced in the full-FT branch.
    assert adapter_calls == []
    call = _TRANSFORMER_CALLS[0]
    # Exactly the plain-transformer kwargs: no actions, nothing adapter/action-related.
    assert set(call.keys()) == {"hidden_states", "timestep", "encoder_hidden_states", "deterministic"}
    assert "actions" not in call
    assert not any("adapter" in k or "action" in k for k in call)
    assert call["deterministic"] is True
    assert np.all(np.isfinite(np.asarray(z)))
    assert "latent_mse" in metrics  # the rollout still produced its metrics dict


def test_side_adapter_rollout_path_is_preserved(monkeypatch):
    # GUARD: the non-full-FT path must be untouched -- the adapter forward is called
    # (WITH actions) and, at guide_scale == 1.0, the plain transformer is NOT called
    # (the CFG bypass, byte-identical to the pre-round-4 behavior).
    adapter_calls = []

    def _spy_adapter(transformer, adapters, **kwargs):
        adapter_calls.append(kwargs)
        return kwargs["hidden_states"] * 0.0

    monkeypatch.setattr(gen, "wan_action_adapter_forward", _spy_adapter)

    t = _RecordingStubTransformer()
    t_graphdef, t_params, t_rest = nnx.split(t, nnx.Param, ...)
    a = _TrivialAdapter()
    a_graphdef, a_params, a_rest = nnx.split(a, nnx.Param, ...)
    state = trainer_mod.TrainState(
        step=0,
        apply_fn=None,
        params=a_params,
        tx=None,
        opt_state=None,
        graphdef=a_graphdef,
        rest_of_state=a_rest,
        transformer_graphdef=t_graphdef,
        transformer_params=t_params,
        transformer_rest=t_rest,
        null_context=jnp.zeros((1, 2, 8), dtype=jnp.float32),
    )
    z, _ = gen._rollout_sample(
        state,
        _rollout_data(),
        jax.random.key(0),
        _scheduler(),
        _rollout_config(model_type="SIDE_ADAPTER_TI2V", guide_scale=1.0),
    )

    assert len(adapter_calls) == 1  # adapter forward drives the side-adapter body
    assert "actions" in adapter_calls[0]  # actions ARE fed to the adapter path
    assert _TRANSFORMER_CALLS == []  # guide_scale == 1.0 -> no v_uncond plain call
    assert np.all(np.isfinite(np.asarray(z)))


# --------------------------------------------------------------------------------------
# (6) validation_ordinals cohort reader.
# --------------------------------------------------------------------------------------


def _fake_raw_records(n):
    """Fake parsed records; the stored ``ordinal`` field (1000+pos) is DELIBERATELY
    distinct from the dataset position so selection-by-position is provable."""
    return [
        {
            "name": f"r{i}".encode(),
            "ordinal": 1000 + i,
            "z_i0": b"",
            "z_video": b"",
            "actions": b"",
            "meta_json": b"{}",
        }
        for i in range(n)
    ]


def test_parse_ordinals():
    assert gen._parse_ordinals("") == []
    assert gen._parse_ordinals("   ") == []
    assert gen._parse_ordinals("0,5,3") == [0, 5, 3]
    assert gen._parse_ordinals(" 0 , 5 , 3 ") == [0, 5, 3]  # whitespace tolerant
    with pytest.raises(ValueError):
        gen._parse_ordinals("0,notanint")
    with pytest.raises(ValueError):
        gen._parse_ordinals("-1")  # negative positions are invalid


def test_select_records_ordinals_in_listed_order():
    recs = _fake_raw_records(8)
    out = gen._select_eval_records(iter(recs), ordinals=[0, 5, 3], start_index=0, count=1)
    assert [pos for pos, _ in out] == [0, 5, 3]  # LISTED order, not sorted
    assert [raw is recs[pos] for pos, raw in out] == [True, True, True]


def test_select_records_empty_ordinals_is_contiguous_and_byte_identical():
    recs = _fake_raw_records(8)
    out = gen._select_eval_records(iter(recs), ordinals=[], start_index=2, count=3)
    # Exactly ds.skip(2).take(3): positions 2,3,4, the SAME record objects, in order.
    assert [pos for pos, _ in out] == [2, 3, 4]
    assert [raw for _, raw in out] == [recs[2], recs[3], recs[4]]
    assert all(raw is recs[pos] for pos, raw in out)


def test_select_records_ordinals_early_stop_does_not_drain_iterator():
    recs = _fake_raw_records(8)
    it = iter(recs)
    out = gen._select_eval_records(it, ordinals=[0, 2], start_index=0, count=1)
    assert [pos for pos, _ in out] == [0, 2]
    # It stopped once the max requested ordinal (2) was seen; 3..7 remain unconsumed.
    assert len(list(it)) == 5


def test_select_records_out_of_range_ordinal_raises_naming_ordinal_and_size():
    recs = _fake_raw_records(8)
    with pytest.raises(ValueError) as ei:
        gen._select_eval_records(iter(recs), ordinals=[0, 99], start_index=0, count=1)
    msg = str(ei.value)
    assert "99" in msg  # the offending ordinal
    assert "8" in msg  # the dataset size actually seen


def _reader_config(**overrides):
    base = {
        "latent_channels": 2,
        "latent_frames": 3,
        "latent_height": 4,
        "latent_width": 4,
        "action_len": 32,
        "action_dim": 7,
        "eval_data_dir": "gs://fake/train",
        "validation_ordinals": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_parsed_record(position):
    c, f, h, w, al, ad = 2, 3, 4, 4, 32, 7
    return {
        "name": b"",  # empty -> the default name encodes the dataset position
        "ordinal": 1000 + position,  # stored field: distinct from position
        "z_i0": np.full((c, 1, h, w), position, dtype=np.float16).tobytes(),
        "z_video": np.full((c, f, h, w), position, dtype=np.float16).tobytes(),
        "actions": np.full((al, ad), position, dtype=np.float32).tobytes(),
        "meta_json": b"{}",
    }


def test_read_eval_samples_honors_ordinals_in_listed_order(monkeypatch):
    recs = [_fake_parsed_record(i) for i in range(8)]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs))
    samples = gen._read_eval_samples(_reader_config(validation_ordinals="0,5,3"), count=1, start_index=0)
    # Selected by dataset POSITION, in the listed order 0,5,3 (buffer value == position).
    assert [int(s.z_video[0, 0, 0, 0]) for s in samples] == [0, 5, 3]
    # The stored ordinal FIELD (1000+pos) rides through, distinct from the position.
    assert [s.ordinal for s in samples] == [1000, 1005, 1003]
    # Default name uses the dataset position (matches the old sample_{pos:06d}).
    assert [s.name for s in samples] == ["sample_000000", "sample_000005", "sample_000003"]


def test_read_eval_samples_empty_ordinals_falls_back_to_contiguous(monkeypatch):
    recs = [_fake_parsed_record(i) for i in range(8)]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs))
    samples = gen._read_eval_samples(_reader_config(validation_ordinals=""), count=3, start_index=2)
    assert [int(s.z_video[0, 0, 0, 0]) for s in samples] == [2, 3, 4]
    assert [s.name for s in samples] == ["sample_000002", "sample_000003", "sample_000004"]


def test_read_eval_samples_missing_ordinals_key_defaults_contiguous(monkeypatch):
    # Round 5 has NOT added validation_ordinals to any yml yet: the reader must reach it
    # via getattr(config, "validation_ordinals", "") and fall back to the contiguous read.
    config = SimpleNamespace(
        latent_channels=2,
        latent_frames=3,
        latent_height=4,
        latent_width=4,
        action_len=32,
        action_dim=7,
        eval_data_dir="gs://fake/train",
    )
    assert not hasattr(config, "validation_ordinals")
    recs = [_fake_parsed_record(i) for i in range(4)]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs))
    samples = gen._read_eval_samples(config, count=2, start_index=1)
    assert [int(s.z_video[0, 0, 0, 0]) for s in samples] == [1, 2]


# --------------------------------------------------------------------------------------
# (7) F3: config.json artifact provenance -- adapter byte-identical, full-FT records
# model_type + the resolved cohort.
# --------------------------------------------------------------------------------------


def _artifact_config(**overrides):
    base = {
        "run_name": "runA",
        "checkpoint_dir": "gs://b/ck",
        "eval_data_dir": "gs://b/train",
        "validation_start_index": 4,
        "side_adapter_sampling_steps": 25,
        "side_adapter_guide_scale": 5.0,
        "action_adapter_type": "pre_context",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# The HEAD (pre-round-4) artifact for _artifact_config(), transcribed field by field.
_HEAD_ARTIFACT = {
    "run_name": "runA",
    "checkpoint_dir": "gs://b/ck",
    "checkpoint_step": 2500,
    "eval_data_dir": "gs://b/train",
    "num_eval_videos": 3,
    "validation_start_index": 4,
    "seed": 11,
    "side_adapter_sampling_steps": 25,
    "side_adapter_guide_scale": 5.0,
    "action_adapter_type": "pre_context",
}


def test_validation_config_artifact_adapter_mode_byte_identical_to_head():
    # Adapter model_type AND a config missing model_type entirely: both must produce
    # the HEAD artifact exactly -- compared as dicts and through the same
    # json.dumps(indent=2, sort_keys=True) serialization _write_json uses, so the
    # claim is byte-identity of the payload written to config.json.
    for config in (_artifact_config(model_type="SIDE_ADAPTER_TI2V"), _artifact_config()):
        out = gen._validation_config_artifact(config, checkpoint_step=2500, num_samples=3, seed=11)
        assert out == _HEAD_ARTIFACT
        assert json.dumps(out, indent=2, sort_keys=True) == json.dumps(_HEAD_ARTIFACT, indent=2, sort_keys=True)


def test_validation_config_artifact_full_ft_records_cohort_provenance():
    config = _artifact_config(model_type="FULL_FT_TI2V", validation_ordinals=" 0, 5 ,3 ")
    out = gen._validation_config_artifact(config, checkpoint_step=0, num_samples=3, seed=11)
    assert out["model_type"] == "FULL_FT_TI2V"
    # RESOLVED positions in LISTED order -- which IS the per-sample seed-assignment
    # order (run() splits the rollout rng sequentially over samples in this order).
    assert out["validation_ordinals"] == [0, 5, 3]
    assert "action_adapter_type" not in out  # no adapter exists in a full-FT run
    assert "validation_start_index" not in out  # ignored when ordinals select the cohort
    # Base provenance rides through (incl. the step-0 baseline's checkpoint_step=0).
    assert out["checkpoint_step"] == 0 and out["num_eval_videos"] == 3 and out["seed"] == 11
    assert out["run_name"] == "runA" and out["eval_data_dir"] == "gs://b/train"


def test_validation_config_artifact_full_ft_contiguous_keeps_start_index():
    # Empty ordinals (and, for round-5 forward-compat, a MISSING key): the contiguous
    # selector is in effect, so validation_start_index is meaningful and stays.
    for config in (
        _artifact_config(model_type="FULL_FT_TI2V", validation_ordinals=""),
        _artifact_config(model_type="FULL_FT_TI2V"),
    ):
        out = gen._validation_config_artifact(config, checkpoint_step=2500, num_samples=3, seed=11)
        assert out["model_type"] == "FULL_FT_TI2V"
        assert out["validation_ordinals"] == []
        assert out["validation_start_index"] == 4
        assert "action_adapter_type" not in out


def test_run_writes_artifact_via_shared_builder():
    # Deletion guard: run() must write config.json through _validation_config_artifact
    # (the behavioral content tests above would otherwise not bind the production path).
    src = inspect.getsource(gen.run)
    assert "_validation_config_artifact" in src
