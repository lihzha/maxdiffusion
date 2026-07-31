"""CPU-only tests for the exp_02 OVERFIT100 one-step val-loss mode (plan D11 "one-step
instrument", §4 ``eval_wan_full_ft_val_loss.py``).

The exp_01 evaluator's aggregation core (``plan_batches`` / ``aggregate`` / the per-checkpoint
restore loop) is REUSED UNCHANGED; this round adds the exp_02 mode around it:

  (A) PER-WINDOW RNG -- exp_01 keys the fixed ``(t, eps)`` draw on the dataset POSITION.
      exp_02 must not: the memorization eval reads the TRAIN set through selections and
      subsets, so a position key would silently change the draw when the record order
      changes. The key is the stable window identity ``(episode_id, window_start)``, folded
      into the seed. Pinned: identical draws under an arbitrary REORDERING of the dataset,
      distinct draws per window and per seed, and inequality with the position-keyed draw
      (so a regression to ``per_example_rng(seed, position, ...)`` fails here).
  (B) SCHEMA-V2 LOAD -- the OVERFIT100 branch of ``load_all_records`` reads
      ``name``/``episode_id``/``episode_index``/``window_start`` (cycle-C review judgment 7)
      and no ``actions``, keeps the EOF drain + count assert, and rejects a duplicate name.
  (C) CONTEXT IN THE LOSS -- the batch carries ``episode_index`` and the jitted step gathers
      ``state.context_table[episode_index]``: row-distinct per example, and the resulting
      per-example loss is EXACTLY the training loss (parity against the trainer's own
      ``_denoising_loss`` with the same ``(t, eps)``).
  (D) THE MODEL GATE -- extended to OVERFIT100_TI2V (with the exp_02 config invariants
      enforced), still refusing every other model type; the exp_01 arm is unchanged.
  (E) OUTPUTS -- the 9-column aggregate schema is untouched; the exp_02 mode additionally
      writes a per-window artifact carrying ``episode_id`` / ``window_start``.

Stub transformer, fake records; no weights, no GCS, no mesh.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import maxdiffusion.eval_wan_full_ft_val_loss as ev
import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.trainers.wan_ti2v_overfit100_trainer as overfit100
from maxdiffusion.models.wan.side_adapter_wan import build_rollout_sigmas
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

_C, _F, _H, _W = 2, 3, 4, 4
_L, _D = 2, 8
_SIGMAS = np.asarray(build_rollout_sigmas(4, 5.0, 0.0, 1.0))

_CALLS: list[dict] = []


@pytest.fixture(autouse=True)
def _clear_calls():
    _CALLS.clear()
    yield
    _CALLS.clear()


class _RecStub(nnx.Module):
    def __init__(self, gain=0.3):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        _CALLS.append(kwargs)
        return self.gain[...] * kwargs["hidden_states"].astype(jnp.float32)


def _fake_raw(*, episode_index, episode_id, window_start, fill=None):
    fill = float(window_start if fill is None else fill)
    return {
        "name": gen.overfit100_window_name(episode_id, window_start).encode(),
        "episode_id": int(episode_id),
        "episode_index": int(episode_index),
        "window_start": int(window_start),
        "z_i0": np.full((_C, 1, _H, _W), fill, dtype=np.float16).tobytes(),
        "z_video": np.full((_C, _F, _H, _W), fill, dtype=np.float16).tobytes(),
        "instruction": f"text {episode_index}".encode(),
    }


def _loaded(specs):
    """Already-loaded record dicts as the OVERFIT100 branch of load_all_records returns."""
    out = []
    for position, (episode_index, episode_id, window_start) in enumerate(specs):
        out.append(
            {
                "position": position,
                "name": gen.overfit100_window_name(episode_id, window_start),
                "episode_id": int(episode_id),
                "episode_index": int(episode_index),
                "window_start": int(window_start),
                "z_i0": np.full((_C, 1, _H, _W), float(window_start), dtype=np.float16),
                "z_video": np.full((_C, _F, _H, _W), float(window_start), dtype=np.float16),
            }
        )
    return out


def _loader_config(expected, **overrides):
    base = {
        "model_type": "OVERFIT100_TI2V",
        "latent_channels": _C,
        "latent_frames": _F,
        "latent_height": _H,
        "latent_width": _W,
        "validation_expected_count": expected,
        "eval_data_dir": "gs://fake/train100",
        "num_text_slots": 4,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _stub_overfit100_state(gain=0.3, *, slots=4):
    stub = _RecStub(gain)
    graphdef, params, rest = nnx.split(stub, nnx.Param, ...)
    table = jnp.stack([jnp.full((_L, _D), float(i) + 1.0, dtype=jnp.float32) for i in range(slots)])
    return overfit100.Overfit100TrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=optax.adamw(0.1),
        graphdef=graphdef,
        rest_of_state=rest,
        context_table=table,
    )


def _loss_config(**overrides):
    base = {
        "model_type": "OVERFIT100_TI2V",
        "weights_dtype": "float32",
        "activations_dtype": "float32",
        "side_adapter_sampling_steps": 4,
        "flow_shift": 5.0,
        "side_adapter_guide_scale": 1.0,
        "side_adapter_noise_mode": "fresh",
        "side_adapter_t_sampling": "uniform",
        "global_batch_size_to_train_on": 2,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ======================================================================================
# (A) The per-window RNG key.
# ======================================================================================


def test_per_window_rng_is_deterministic_and_window_specific():
    a = ev.per_window_rng(0, 25189, 48, 25, (_C, _F, _H, _W))
    b = ev.per_window_rng(0, 25189, 48, 25, (_C, _F, _H, _W))
    assert int(a[0]) == int(b[0])
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
    # A different window (same episode, different start) draws differently...
    c = ev.per_window_rng(0, 25189, 52, 25, (_C, _F, _H, _W))
    assert not np.array_equal(np.asarray(a[1]), np.asarray(c[1]))
    # ...and so does a different episode at the same start, and a different seed.
    d = ev.per_window_rng(0, 30000, 48, 25, (_C, _F, _H, _W))
    e = ev.per_window_rng(1, 25189, 48, 25, (_C, _F, _H, _W))
    assert not np.array_equal(np.asarray(a[1]), np.asarray(d[1]))
    assert not np.array_equal(np.asarray(a[1]), np.asarray(e[1]))


def test_per_window_rng_is_not_the_position_keyed_draw():
    # A regression to exp_01's position key must be visible: for the same (seed, num_steps,
    # shape) the two draws differ (they fold in different values).
    window = ev.per_window_rng(0, 25189, 48, 25, (_C, _F, _H, _W))
    position = ev.per_example_rng(0, 0, 25, (_C, _F, _H, _W))
    assert not np.array_equal(np.asarray(window[1]), np.asarray(position[1]))


def test_per_window_rng_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ev.per_window_rng(0, 1, 0, 0, (2, 2))
    with pytest.raises(ValueError):
        ev.per_window_rng(0, -1, 0, 25, (2, 2))  # negative episode_id cannot be folded in
    with pytest.raises(ValueError):
        ev.per_window_rng(0, 1, -4, 25, (2, 2))


def test_window_fold_key_is_stable_and_collision_free_across_the_cohort():
    keys = {
        (eid, start): bytes(jax.random.key_data(gen.window_fold_key(0, eid, start)).tobytes())
        for eid in (100, 25189, 69722)
        for start in (0, 4, 196)
    }
    assert len(set(keys.values())) == len(keys)
    # Stable across calls.
    assert bytes(jax.random.key_data(gen.window_fold_key(0, 25189, 48)).tobytes()) == bytes(
        jax.random.key_data(gen.window_fold_key(0, 25189, 48)).tobytes()
    )


def test_assemble_overfit100_batch_is_invariant_to_dataset_reordering():
    specs = [(i // 2, 100 + i // 2, 4 * (i % 2)) for i in range(8)]
    records = _loaded(specs)
    shuffled_specs = [specs[i] for i in (5, 0, 7, 2, 1, 6, 3, 4)]
    shuffled = _loaded(shuffled_specs)

    def draws(recs, positions):
        _, z_video, eps, sigma_t, episode_index = ev.assemble_overfit100_batch(recs, positions, 0, 4, _SIGMAS)
        return {
            (int(recs[p]["episode_id"]), int(recs[p]["window_start"])): (float(sigma_t[i]), np.asarray(eps[i]))
            for i, p in enumerate(positions)
        }

    a = draws(records, list(range(8)))
    b = draws(shuffled, list(range(8)))
    assert set(a) == set(b)
    for key in a:
        assert a[key][0] == b[key][0]
        np.testing.assert_array_equal(a[key][1], b[key][1])


def test_assemble_overfit100_batch_stacks_context_indices_and_latents():
    records = _loaded([(3, 103, 0), (1, 101, 8)])
    z_i0, z_video, eps, sigma_t, episode_index = ev.assemble_overfit100_batch(records, [0, 1], 0, 4, _SIGMAS)
    assert z_i0.shape == (2, _C, 1, _H, _W) and z_video.shape == (2, _C, _F, _H, _W)
    assert eps.shape == (2, _C, _F, _H, _W) and sigma_t.shape == (2,)
    assert list(np.asarray(episode_index)) == [3, 1]
    assert np.asarray(episode_index).dtype == np.int32
    assert float(z_video[1, 0, 0, 0, 0]) == pytest.approx(8.0)


def test_assemble_overfit100_batch_is_independent_of_batch_composition():
    records = _loaded([(i // 2, 100 + i // 2, 4 * (i % 2)) for i in range(8)])
    _, _, eps_all, sig_all, _ = ev.assemble_overfit100_batch(records, list(range(8)), 0, 4, _SIGMAS)
    _, _, eps_a, sig_a, _ = ev.assemble_overfit100_batch(records, [0, 1, 2, 3], 0, 4, _SIGMAS)
    _, _, eps_b, sig_b, _ = ev.assemble_overfit100_batch(records, [4, 5, 6, 7], 0, 4, _SIGMAS)
    np.testing.assert_array_equal(np.concatenate([eps_a, eps_b]), eps_all)
    np.testing.assert_array_equal(np.concatenate([sig_a, sig_b]), sig_all)


# ======================================================================================
# (B) The schema-v2 load.
# ======================================================================================


def test_load_all_records_overfit100_branch_reads_the_aggregation_fields(monkeypatch):
    raws = [_fake_raw(episode_index=i // 2, episode_id=100 + i // 2, window_start=4 * (i % 2)) for i in range(6)]
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(raws))
    out = ev.load_all_records(_loader_config(6))
    assert [r["position"] for r in out] == list(range(6))
    assert [r["episode_index"] for r in out] == [0, 0, 1, 1, 2, 2]
    assert [r["window_start"] for r in out] == [0, 4, 0, 4, 0, 4]
    assert [r["episode_id"] for r in out] == [100, 100, 101, 101, 102, 102]
    assert out[0]["name"] == "ep100_v0_s00000"
    assert out[0]["z_video"].shape == (_C, _F, _H, _W)
    assert "actions" not in out[0] and "ordinal" not in out[0]


def test_load_all_records_overfit100_count_mismatch_names_both_numbers(monkeypatch):
    raws = [_fake_raw(episode_index=0, episode_id=100, window_start=4 * i) for i in range(3)]
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(raws))
    with pytest.raises(ValueError) as ei:
        ev.load_all_records(_loader_config(7))
    msg = str(ei.value)
    assert "3" in msg and "7" in msg


def test_load_all_records_overfit100_refuses_a_duplicate_window(monkeypatch):
    dup = _fake_raw(episode_index=0, episode_id=100, window_start=0)
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter([dup, dict(dup)]))
    with pytest.raises(ValueError) as ei:
        ev.load_all_records(_loader_config(2))
    assert "duplicate" in str(ei.value).lower()


def test_load_all_records_overfit100_refuses_an_out_of_range_episode_index(monkeypatch):
    raws = [_fake_raw(episode_index=9, episode_id=109, window_start=0)]
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(raws))
    with pytest.raises(ValueError) as ei:
        ev.load_all_records(_loader_config(1, num_text_slots=4))
    msg = str(ei.value)
    assert "9" in msg and "4" in msg


def test_load_all_records_full_ft_arm_still_uses_the_exp01_reader(monkeypatch):
    def _boom(config):
        raise AssertionError("the OVERFIT100 reader must not run for FULL_FT_TI2V")

    monkeypatch.setattr(gen, "_iter_overfit100_records", _boom)
    exp01 = [
        {
            "name": b"r0",
            "ordinal": 1000,
            "z_i0": np.zeros((_C, 1, _H, _W), np.float16).tobytes(),
            "z_video": np.zeros((_C, _F, _H, _W), np.float16).tobytes(),
            "actions": np.zeros((32, 7), np.float32).tobytes(),
            "meta_json": b"{}",
        }
    ]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(exp01))
    out = ev.load_all_records(_loader_config(1, model_type="FULL_FT_TI2V"))
    assert out[0]["ordinal"] == 1000


# ======================================================================================
# (C) Context gather inside the loss + training parity.
# ======================================================================================


def test_eval_step_gathers_row_distinct_context_per_example():
    state = _stub_overfit100_state()
    records = _loaded([(3, 103, 0), (1, 101, 8)])
    z_i0, z_video, eps, sigma_t, episode_index = ev.assemble_overfit100_batch(records, [0, 1], 0, 4, _SIGMAS)
    losses = ev._eval_batch_per_example_loss_overfit100(
        state,
        jnp.asarray(z_i0),
        jnp.asarray(z_video),
        jnp.asarray(eps),
        jnp.asarray(sigma_t),
        jnp.asarray(episode_index),
        config=_loss_config(),
        num_train_timesteps=1000,
    )
    assert losses.shape == (2,)
    assert len(_CALLS) == 1
    context = np.asarray(_CALLS[0]["encoder_hidden_states"])
    assert context.shape == (2, _L, _D)
    # Row-distinct: example 0 -> table row 3 (value 4.0), example 1 -> row 1 (value 2.0).
    assert context[0, 0, 0] == pytest.approx(4.0)
    assert context[1, 0, 0] == pytest.approx(2.0)
    assert set(_CALLS[0]) == {"hidden_states", "timestep", "encoder_hidden_states", "deterministic"}
    assert "actions" not in _CALLS[0]


def test_eval_loss_matches_the_training_loss_exactly_including_context(monkeypatch):
    # PARITY with the trainer's own module-level _denoising_loss: same shared helpers, same
    # gathered context, same (t, eps) -> the eval per-example mean IS the training scalar.
    state = _stub_overfit100_state()
    config = _loss_config(global_batch_size_to_train_on=2)
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=5.0, sigma_min=0.0, sigma_max=1.0)
    records = _loaded([(3, 103, 0), (1, 101, 8)])
    z_i0, z_video, eps, sigma_t, episode_index = ev.assemble_overfit100_batch(records, [0, 1], 0, 4, _SIGMAS)

    sigmas = build_rollout_sigmas(4, 5.0, scheduler.config.sigma_min, scheduler.config.sigma_max)
    t_idx = jnp.asarray([int(np.argmin(np.abs(np.asarray(sigmas) - s))) for s in np.asarray(sigma_t)])
    monkeypatch.setattr(overfit100, "_sample_step_indices", lambda rng, b, n, sig, cfg: t_idx)
    monkeypatch.setattr(overfit100, "_build_noise", lambda rng, shape, dtype, cfg: jnp.asarray(eps))

    train_loss, _ = overfit100._denoising_loss(
        state.params,
        state,
        {
            "z_i0": jnp.asarray(z_i0),
            "z_video": jnp.asarray(z_video),
            "episode_index": jnp.asarray(episode_index),
        },
        jax.random.key(0),
        config,
        scheduler,
    )
    eval_losses = ev._eval_batch_per_example_loss_overfit100(
        state,
        jnp.asarray(z_i0),
        jnp.asarray(z_video),
        jnp.asarray(eps),
        jnp.asarray(sigma_t),
        jnp.asarray(episode_index),
        config=config,
        num_train_timesteps=int(scheduler.config.num_train_timesteps),
    )
    assert float(jnp.mean(eval_losses)) == pytest.approx(float(train_loss), rel=1e-6, abs=1e-7)


def test_eval_step_uses_the_shared_noisy_pinned_latents_helper():
    # Frame 0 of the noisy latents must be the pin (z_i0), not the interpolation.
    state = _stub_overfit100_state()
    records = _loaded([(0, 100, 0)])
    z_i0, z_video, eps, sigma_t, episode_index = ev.assemble_overfit100_batch(records, [0], 0, 4, _SIGMAS)
    ev._eval_batch_per_example_loss_overfit100(
        state,
        jnp.asarray(z_i0),
        jnp.asarray(z_video),
        jnp.asarray(eps),
        jnp.asarray(sigma_t),
        jnp.asarray(episode_index),
        config=_loss_config(global_batch_size_to_train_on=1),
        num_train_timesteps=1000,
    )
    hidden = np.asarray(_CALLS[0]["hidden_states"])
    np.testing.assert_allclose(hidden[:, :, 0], np.asarray(z_i0)[:, :, 0], rtol=0, atol=0)


# ======================================================================================
# (D) The model-type gate.
# ======================================================================================


def _gate_config(**overrides):
    base = {
        "model_type": "OVERFIT100_TI2V",
        "side_adapter_guide_scale": 1.0,
        "side_adapter_noise_mode": "fresh",
        "num_text_slots": 100,
        "text_encode_batch": 8,
        "expected_windows": 1629,
        "checkpoint_steps": [250],
        "model_manifest_path": "docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_gate_accepts_overfit100_and_full_ft():
    ev._assert_full_ft(_gate_config())
    ev._assert_full_ft(
        SimpleNamespace(model_type="FULL_FT_TI2V", side_adapter_guide_scale=1.0, side_adapter_noise_mode="fresh")
    )


def test_gate_still_refuses_other_model_types():
    with pytest.raises(ValueError) as ei:
        ev._assert_full_ft(_gate_config(model_type="SIDE_ADAPTER_TI2V"))
    msg = str(ei.value)
    assert "FULL_FT_TI2V" in msg and "OVERFIT100_TI2V" in msg and "SIDE_ADAPTER_TI2V" in msg


def test_gate_enforces_the_probe_invariants_in_overfit100_mode():
    with pytest.raises(ValueError, match="guide_scale"):
        ev._assert_full_ft(_gate_config(side_adapter_guide_scale=5.0))
    with pytest.raises(ValueError, match="fresh"):
        ev._assert_full_ft(_gate_config(side_adapter_noise_mode="fixed"))


def test_gate_enforces_the_exp02_config_invariants():
    with pytest.raises(ValueError, match="num_text_slots"):
        ev._assert_full_ft(_gate_config(num_text_slots=0))
    with pytest.raises(ValueError, match="model_manifest_path"):
        ev._assert_full_ft(_gate_config(model_manifest_path=""))


def test_state_builder_dispatch_is_model_type_driven(monkeypatch):
    def _boom(config):
        raise AssertionError("the full-FT builder must not run in OVERFIT100 mode")

    monkeypatch.setattr(gen, "_build_full_ft_validation_state", _boom)
    pipe = SimpleNamespace(vae=1, vae_cache=2, text_encoder=3, tokenizer=4)
    monkeypatch.setattr(
        gen,
        "_build_overfit100_validation_state",
        lambda config: ("TR", pipe, "MESH", "STATE", "SH", "NULL"),
    )
    trainer, out_pipe, mesh, state, shardings = ev._build_and_free_state(_gate_config())
    assert (trainer, mesh, state, shardings) == ("TR", "MESH", "STATE", "SH")
    # The loss evaluator never decodes or embeds -> the heavy rollout modules are freed.
    for attr in ("vae", "vae_cache", "text_encoder", "tokenizer"):
        assert not hasattr(out_pipe, attr)


# ======================================================================================
# (E) Outputs.
# ======================================================================================


def test_aggregate_column_schema_is_unchanged():
    assert tuple(ev._VAL_LOSS_COLUMNS) == (
        "checkpoint_step",
        "mean_loss",
        "stderr",
        "n",
        "validation_seed",
        "dataset_path",
        "checkpoint_path",
        "train_commit",
        "eval_commit",
    )


def test_per_window_columns_carry_the_window_identity():
    assert tuple(ev._PER_WINDOW_COLUMNS) == (
        "checkpoint_step",
        "name",
        "episode_id",
        "episode_index",
        "window_start",
        "loss",
        "sigma_t",
        "validation_seed",
    )


def test_make_per_window_rows_pairs_losses_with_records():
    records = _loaded([(3, 103, 0), (1, 101, 8)])
    rows = ev.make_per_window_rows(
        records,
        positions=[0, 1],
        validity=[True, True],
        losses=np.asarray([0.5, 0.25]),
        sigma_t=np.asarray([0.3, 0.7]),
        checkpoint_step=2500,
        seed=0,
    )
    assert [r["name"] for r in rows] == ["ep103_v0_s00000", "ep101_v0_s00008"]
    assert [r["episode_id"] for r in rows] == [103, 101]
    assert [r["window_start"] for r in rows] == [0, 8]
    assert [r["loss"] for r in rows] == [0.5, 0.25]
    assert [r["episode_index"] for r in rows] == [3, 1]
    assert all(r["checkpoint_step"] == 2500 for r in rows)
    assert list(rows[0].keys()) == list(ev._PER_WINDOW_COLUMNS)


def test_make_per_window_rows_drops_padded_slots():
    records = _loaded([(0, 100, 0), (1, 101, 0)])
    rows = ev.make_per_window_rows(
        records,
        positions=[0, 1, 1],
        validity=[True, True, False],
        losses=np.asarray([0.5, 0.25, 0.25]),
        sigma_t=np.asarray([0.3, 0.7, 0.7]),
        checkpoint_step=2500,
        seed=0,
    )
    assert len(rows) == 2  # the padded duplicate is excluded


def test_write_per_window_outputs_writes_json_and_csv(tmp_path):
    records = _loaded([(0, 100, 0)])
    rows = ev.make_per_window_rows(
        records,
        positions=[0],
        validity=[True],
        losses=np.asarray([0.5]),
        sigma_t=np.asarray([0.3]),
        checkpoint_step=2500,
        seed=0,
    )
    ev.write_per_window_outputs(rows, str(tmp_path))
    payload = json.loads((tmp_path / "val_loss_per_window.json").read_text())
    assert payload == rows
    reader = csv.DictReader(io.StringIO((tmp_path / "val_loss_per_window.csv").read_text()))
    assert reader.fieldnames == list(ev._PER_WINDOW_COLUMNS)
    assert [r["name"] for r in reader] == ["ep100_v0_s00000"]


# ======================================================================================
# (eval-ffmpeg strengthening, finding 1) EXECUTABLE proof that an OVERFIT100 loss
# evaluation needs no video codec.
#
# The previous claim was a source-token scan of the module, which a reviewer correctly
# called out as weak: it cannot see `subprocess.run(["ffmpeg", ...])`, an aliased import, or
# a transitive call through the imported generator module. This drives the REAL
# ``evaluate()`` loop -- record load, state build, per-checkpoint restore, jitted batches,
# aggregation, json/csv outputs -- with every codec entry point booby-trapped and
# ffmpeg/ffprobe stripped from PATH. If any decoder were reachable, the run would raise.
# ======================================================================================

MESH_AXES = ("data", "fsdp", "context", "tensor")


class _CodecTrap(Exception):
    """Raised by any booby-trapped codec entry point."""


def _install_codec_traps(monkeypatch, calls):
    """Booby-trap every path that could reach ffmpeg/ffprobe, directly or transitively."""
    import subprocess

    from maxdiffusion import utils as mx_utils
    from maxdiffusion.data_preprocessing import build_overfit100_dataset as builder

    def _trap(name):
        def _boom(*args, **kwargs):
            calls.append(name)
            raise _CodecTrap(f"{name} was reached by the loss evaluator")

        return _boom

    # Direct decoders / encoders, in both the generator module and the builder.
    monkeypatch.setattr(gen, "_save_video", _trap("gen._save_video"))
    monkeypatch.setattr(gen, "overfit100_aux_rgb", _trap("gen.overfit100_aux_rgb"))
    monkeypatch.setattr(gen, "export_to_video", _trap("gen.export_to_video"), raising=False)
    monkeypatch.setattr(mx_utils, "export_to_video", _trap("utils.export_to_video"), raising=False)
    monkeypatch.setattr(builder, "decode_mp4_frames", _trap("builder.decode_mp4_frames"))
    monkeypatch.setattr(builder, "fetch_pinned", _trap("builder.fetch_pinned"))

    # ...and the spawn layer itself, so an inline argv (the thing a token scan cannot see) is
    # caught too. Non-codec spawns (e.g. `git rev-parse` for the eval commit) pass through.
    real_run, real_popen = subprocess.run, subprocess.Popen

    def _guard(delegate, label):
        def _checked(popenargs, *args, **kwargs):
            argv = popenargs if isinstance(popenargs, (list, tuple)) else [popenargs]
            program = str(argv[0]) if argv else ""
            if os.path.basename(program) in ("ffmpeg", "ffprobe"):
                calls.append(f"{label}:{os.path.basename(program)}")
                raise _CodecTrap(f"{label} spawned {program}")
            return delegate(popenargs, *args, **kwargs)

        return _checked

    monkeypatch.setattr(subprocess, "run", _guard(real_run, "subprocess.run"))
    monkeypatch.setattr(subprocess, "Popen", _guard(real_popen, "subprocess.Popen"))


def _strip_ffmpeg_from_path(monkeypatch, tmp_path):
    """No ffmpeg/ffprobe anywhere a lookup could find one."""
    import shutil

    empty = tmp_path / "empty_bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda binary, *a, **kw: None if binary in ("ffmpeg", "ffprobe") else real_which(binary)
    )
    return empty


def _loss_eval_config(tmp_path, **overrides):
    base = {
        "model_type": "OVERFIT100_TI2V",
        "latent_channels": _C,
        "latent_frames": _F,
        "latent_height": _H,
        "latent_width": _W,
        "validation_expected_count": 4,
        "validation_checkpoint_steps": "2500",
        "eval_data_dir": "gs://fake/train10",
        "checkpoint_dir": str(tmp_path / "ck"),
        "output_dir": str(tmp_path / "out"),
        "run_name": "loss-probe",
        "validation_loss_output_dir": str(tmp_path / "out" / "validation_loss"),
        "side_adapter_guide_scale": 1.0,
        "side_adapter_noise_mode": "fresh",
        "validation_seed": 0,
        "flow_shift": 5.0,
        "side_adapter_sampling_steps": 25,
        "global_batch_size_to_train_on": 2,
        "data_sharding": [["data", "fsdp", "context", "tensor"]],
        "logical_axis_rules": (),
        "weights_dtype": "float32",
        "activations_dtype": "float32",
        "seed": 0,
        "num_text_slots": 4,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _install_loss_eval_stubs(monkeypatch, *, n_records=4, slots=4):
    """Real loop, stub weights: records, a 1-device mesh, a genuine state + sharding tree."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    raws = [
        _fake_raw(episode_index=i // 2, episode_id=100 + i // 2, window_start=4 * (i % 2)) for i in range(n_records)
    ]
    monkeypatch.setattr(gen, "_iter_overfit100_records", lambda config: iter(raws))

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), MESH_AXES)
    state = _stub_overfit100_state(slots=slots)
    shardings = jax.tree.map(lambda _: NamedSharding(mesh, P()), state)
    trainer = SimpleNamespace(
        _create_scheduler=lambda: (
            FlaxFlowMatchScheduler(dtype=jnp.float32, shift=5.0, sigma_min=0.0, sigma_max=1.0),
            None,
        )
    )
    pipeline = SimpleNamespace(vae=object(), vae_cache=object(), text_encoder=object(), tokenizer=object())
    monkeypatch.setattr(
        gen,
        "_build_overfit100_validation_state",
        lambda config: (trainer, pipeline, mesh, state, shardings, jnp.zeros((1, _L, _D))),
    )
    monkeypatch.setattr(
        gen,
        "_restore_checkpoint_state",
        lambda config, st, ckpt, **kw: (
            st.replace(step=int(kw.get("requested_step", 0))),
            int(kw.get("requested_step", 0)),
        ),
    )
    return pipeline


def test_overfit100_loss_evaluation_completes_with_every_codec_entry_point_booby_trapped(tmp_path, monkeypatch):
    # The executable form of "this arm needs no ffmpeg": the REAL evaluate() loop runs to its
    # json/csv artifacts while every decoder -- and the spawn layer itself -- would raise.
    calls: list[str] = []
    _install_codec_traps(monkeypatch, calls)
    _strip_ffmpeg_from_path(monkeypatch, tmp_path)
    monkeypatch.setenv("TRAIN_COMMIT", "a" * 40)
    monkeypatch.setenv("COMMIT", "b" * 40)
    monkeypatch.delenv("SMOKE_LIMIT", raising=False)
    pipeline = _install_loss_eval_stubs(monkeypatch)

    config = _loss_eval_config(tmp_path)
    ev._assert_full_ft(
        SimpleNamespace(
            model_type="OVERFIT100_TI2V",
            side_adapter_guide_scale=1.0,
            side_adapter_noise_mode="fresh",
            num_text_slots=4,
            text_encode_batch=8,
            expected_windows=167,
            checkpoint_steps=[250],
            model_manifest_path="docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json",
        )
    )
    rows, out_dir = ev.evaluate(config)

    # (1) No codec entry point was reached -- direct, aliased, or via a subprocess spawn.
    assert calls == [], f"the loss evaluator reached a codec path: {calls}"
    # (2) The loop really ran: one aggregate row + the per-window artifact for all 4 windows.
    assert len(rows) == 1 and rows[0]["checkpoint_step"] == 2500 and rows[0]["n"] == 4
    written = sorted(os.listdir(out_dir))
    assert "val_loss.json" in written and "val_loss.csv" in written
    assert "val_loss_per_window.json" in written and "val_loss_per_window.csv" in written
    per_window = json.loads((Path(out_dir) / "val_loss_per_window.json").read_text())
    assert len(per_window) == 4
    assert [row["name"] for row in per_window] == [
        "ep100_v0_s00000",
        "ep100_v0_s00004",
        "ep101_v0_s00000",
        "ep101_v0_s00004",
    ]
    assert all(np.isfinite(row["loss"]) for row in per_window)
    # (3) The VAE was released before the loop, so latents could not be decoded even in principle.
    for attr in ("vae", "vae_cache", "text_encoder", "tokenizer"):
        assert not hasattr(pipeline, attr)


def test_the_codec_traps_are_not_vacuous(tmp_path, monkeypatch):
    # A booby trap that never fires proves nothing, so prove the traps DO catch both shapes of
    # call: a direct decoder invocation and an inline `subprocess.run(["ffmpeg", ...])` argv --
    # exactly the transitive/inline cases the source scan cannot see.
    import subprocess

    from maxdiffusion.data_preprocessing import build_overfit100_dataset as builder

    calls: list[str] = []
    _install_codec_traps(monkeypatch, calls)
    with pytest.raises(_CodecTrap):
        builder.decode_mp4_frames("/nonexistent.mp4")
    with pytest.raises(_CodecTrap):
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
    with pytest.raises(_CodecTrap):
        subprocess.run(["/usr/local/bin/ffprobe", "-version"], capture_output=True)
    assert calls == ["builder.decode_mp4_frames", "subprocess.run:ffmpeg", "subprocess.run:ffprobe"]
    # ...while a NON-codec spawn still passes through untouched.
    assert subprocess.run(["echo", "ok"], capture_output=True, text=True).stdout.strip() == "ok"


def test_loss_eval_source_scan_remains_as_supplementary_evidence():
    # Kept per the reviewer: cheap, and it documents the intent. It is no longer the only proof.
    source = Path(ev.__file__).read_text()
    for forbidden in ("decode_mp4_frames", "overfit100_aux_rgb", "_save_video", "export_to_video", "ffprobe"):
        assert forbidden not in source, f"{forbidden} appeared in the loss evaluator; it now needs ffmpeg"
