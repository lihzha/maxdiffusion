"""CPU-only integration tests for the full-FT validation-loss EVALUATOR (exp_01 Part II).

Cycle B "val-loss-evaluator" under strict TDD. This is the config-driven evaluator that
sits on top of cycle A's pure functions (``per_example_rng`` / ``plan_batches`` /
``aggregate`` -- their contracts are NOT retested here). It covers the seven cycle-B
concerns and the four mutants from the SOP:

  1. ``load_all_records`` -- EOF drain + count assert BEFORE any state/restore work
     (the ordering is proven by a build spy), fewer- AND more-than-expected failure with
     both numbers, ``validation_expected_count <= 0`` refused, and positions enumerated
     independently of the stored ``ordinal`` (mutant ii: assert-after-restore -> red).
  2. Cross-checkpoint / cross-batch / cross-order ``(t, eps)`` identity keyed by dataset
     POSITION, with stored ordinals deliberately unrelated to positions (mutant i:
     RNG-by-stored-ordinal -> red).
  3. The jitted eval step's objective: one plain transformer call, no actions/adapter/CFG,
     null-context broadcast in activations dtype, frame-0 pin, per-example loss == a
     hand-computed masked MSE against ``eps - z_video``.
  4. The checkpoint loop: restore called with each requested step in order, restored-step
     mismatch is fatal, rows collected per step, and the padded tail is excluded from the
     aggregate via the plan's validity (mutant iii: validity ignored -> red).
  5. The ``requested_step`` kwarg on ``generate_wan_side_adapter._restore_checkpoint_state``:
     ``None`` is byte-identical to the config path; an int OVERRIDES the config-derived
     step (mutant iv: requested_step ignored -> red), via a real tiny Orbax round trip.
  6. ``write_outputs`` -- the EXACT 9-column schema in JSON and CSV; the plot is guarded by
     the matplotlib import (skipped when absent, written when present); ``plot-only`` CLI
     regenerates from the JSON; the SMOKE path writes to ``validation_loss_smoke/`` and
     SKIPS the ``n == expected`` assertion.
  7. VAE / vae_cache / text_encoder / tokenizer are dropped right after state construction
     (plan F5) -- no rollouts happen in T1.
  8. STRENGTHEN (cycle-B Codex review): F1 -- the ``_loss_to_host`` collective-gather seam
     (two-host v6e-8: the jitted [B] output is sharded over the GLOBAL mesh, so a direct
     ``np.asarray`` would raise; sharded values must route through ``process_allgather``,
     and the loop must invoke the seam per batch -- mutant alpha); F2 -- TRAIN_COMMIT is
     mandatory at BOTH layers (wrapper rejects before python; ``_require_train_commit``
     refuses FULL-mode artifact writes on empty/"unknown"; smoke exempt, documented --
     mutant beta) with exact propagation into both writers' rows; F3 -- the sigma-grid
     wiring seam ``_build_sigma_grid`` pins ALL FOUR ``build_rollout_sigmas`` args +
     evaluate-level wiring capture (mutant gamma), plus static wrapper env->override
     mapping tests (grep-style + fake-python argv capture + ``bash -n``).

CPU-only: tiny ``nnx.Module`` stub transformers, fake in-memory records via the
``_iter_parsed_records`` monkeypatch seam; no 5B weights, no pipeline, no mesh (except a
1-device CPU mesh in the evaluate wiring test). The darwin grain import stub lives in
``conftest.py``.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh

import maxdiffusion.eval_wan_full_ft_val_loss as ev
import maxdiffusion.generate_wan_side_adapter as gen
import maxdiffusion.trainers.wan_ti2v_full_ft_trainer as full_ft
from maxdiffusion.models.wan.side_adapter_wan import build_rollout_sigmas
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

# Small latent geometry: C channels, F latent frames (>1 so a non-frame-0 signal exists),
# H x W spatial. Keeps every CPU forward trivially cheap.
_C, _F, _H, _W = 2, 3, 4, 4
_EXAMPLE_SHAPE = (_C, _F, _H, _W)
_SIGMAS = np.asarray(build_rollout_sigmas(25, 5.0, 0.0, 1.0))

# The EXACT 9-column schema (plan D5/F6) both writers must emit, in order.
_COLUMNS = [
    "checkpoint_step",
    "mean_loss",
    "stderr",
    "n",
    "validation_seed",
    "dataset_path",
    "checkpoint_path",
    "train_commit",
    "eval_commit",
]

# Repo root (this file lives at src/maxdiffusion/tests/worklogs_yixun/) and the wrapper
# under static test (F3).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_WRAPPER = str(_REPO_ROOT / "bash_scripts" / "eval_wan_full_ft_val_loss.sh")


# --------------------------------------------------------------------------------------
# Fixtures / fakes.
# --------------------------------------------------------------------------------------


def _fake_raw(index, *, ordinal=None, fill=None):
    """A fake PARSED record (bytes fields), as ``_iter_parsed_records`` yields them.

    The stored ``ordinal`` is DELIBERATELY unrelated to the dataset position (1000+7*index
    by default) so any code that (mis)uses it as an index is caught. ``fill`` is the numeric
    value packed into z_i0/z_video so the loaded record's data is identifiable per position.
    """
    fill = index if fill is None else fill
    ordinal = (1000 + 7 * index) if ordinal is None else ordinal
    return {
        "name": f"r{index}".encode(),
        "ordinal": ordinal,
        "z_i0": np.full((_C, 1, _H, _W), fill, dtype=np.float16).tobytes(),
        "z_video": np.full((_C, _F, _H, _W), fill, dtype=np.float16).tobytes(),
        "actions": np.full((32, 7), fill, dtype=np.float32).tobytes(),
        "meta_json": b"{}",
    }


def _loaded_records(n, *, fills=None):
    """A list of already-LOADED record dicts (position == index), as ``load_all_records``
    returns them: z_i0/z_video full of the position value, ordinal distinct from position."""
    fills = list(range(n)) if fills is None else fills
    return [
        {
            "position": i,
            "ordinal": 1000 + 7 * i,
            "z_i0": np.full((_C, 1, _H, _W), fills[i], dtype=np.float16),
            "z_video": np.full((_C, _F, _H, _W), fills[i], dtype=np.float16),
        }
        for i in range(n)
    ]


def _loader_config(expected):
    return SimpleNamespace(
        latent_channels=_C,
        latent_frames=_F,
        latent_height=_H,
        latent_width=_W,
        validation_expected_count=expected,
        eval_data_dir="gs://fake/val",
    )


def _eval_config(**overrides):
    base = {
        "model_type": "FULL_FT_TI2V",
        "latent_channels": _C,
        "latent_frames": _F,
        "latent_height": _H,
        "latent_width": _W,
        "validation_expected_count": 7,
        "validation_checkpoint_steps": "2500,5000",
        "eval_data_dir": "gs://fake/val",
        "checkpoint_dir": "/fake/ckpts",
        "output_dir": "gs://out",
        "run_name": "runX",
        "validation_loss_output_dir": "",
        "side_adapter_guide_scale": 1.0,
        "side_adapter_noise_mode": "fresh",
        "validation_seed": 0,
        # Keys evaluate() itself consumes (the wiring test drives it end-to-end to the loop).
        "flow_shift": 5.0,
        "side_adapter_sampling_steps": 25,
        "global_batch_size_to_train_on": 8,
        "data_sharding": [["data", "fsdp", "context", "tensor"]],
        "logical_axis_rules": (),
        "weights_dtype": "float32",
        "activations_dtype": "float32",
        "seed": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _RecStub(nnx.Module):
    """Plain-transformer stand-in: records each call's kwargs, velocity == gain * hidden."""

    def __init__(self, gain=0.3):
        self.gain = nnx.Param(jnp.asarray(gain, dtype=jnp.float32))

    def __call__(self, **kwargs):
        _CALLS.append(kwargs)
        return self.gain[...] * kwargs["hidden_states"].astype(jnp.float32)


_CALLS: list[dict] = []


@pytest.fixture(autouse=True)
def _clear_calls():
    _CALLS.clear()
    yield
    _CALLS.clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Hermetic env: the evaluator reads SMOKE_LIMIT / TRAIN_COMMIT / COMMIT from the process
    # environment, so a developer shell must never leak into these tests. Tests that need a
    # value set it explicitly on top of this cleanup.
    for var in ("SMOKE_LIMIT", "TRAIN_COMMIT", "COMMIT"):
        monkeypatch.delenv(var, raising=False)


def _stub_state(gain=0.3, *, null_l=2, null_d=8, tx=None):
    t = _RecStub(gain)
    graphdef, params, rest = nnx.split(t, nnx.Param, ...)
    null_context = jnp.arange(null_l * null_d, dtype=jnp.float32).reshape(1, null_l, null_d)
    return full_ft.FullFTTrainState.create(
        apply_fn=graphdef.apply,
        params=params,
        tx=tx or optax.adamw(0.1),
        graphdef=graphdef,
        rest_of_state=rest,
        null_context=null_context,
    )


def _first_leaf(tree):
    return np.asarray(jax.tree_util.tree_leaves(tree)[0])


def _ckpt_config(*, checkpoint_step, ckpt_dir):
    return SimpleNamespace(
        model_type="FULL_FT_TI2V",
        checkpoint_step=checkpoint_step,
        checkpoint_dir=ckpt_dir,
        output_dir="",
        run_name="",
        checkpoint_keep_period=-1,
    )


# ======================================================================================
# 1. load_all_records -- drain, count assert BEFORE state work, position enumeration.
# ======================================================================================


def test_load_all_records_positions_enumerated_independently_of_stored_ordinal(monkeypatch):
    recs = [_fake_raw(i) for i in range(7)]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs))
    out = ev.load_all_records(_loader_config(7))
    assert [r["position"] for r in out] == list(range(7))  # POSITION == enumeration index
    assert [r["ordinal"] for r in out] == [1000 + 7 * i for i in range(7)]  # stored != position
    # The loaded data at position p carries the position value (proves position<->data pairing).
    assert all(int(r["z_video"].flat[0]) == r["position"] for r in out)
    assert all(int(r["z_i0"].flat[0]) == r["position"] for r in out)
    assert out[0]["z_video"].shape == _EXAMPLE_SHAPE and out[0]["z_i0"].shape == (_C, 1, _H, _W)


def test_load_all_records_fewer_than_expected_raises_both_numbers(monkeypatch):
    recs = [_fake_raw(i) for i in range(5)]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs))
    with pytest.raises(ValueError) as ei:
        ev.load_all_records(_loader_config(7))
    msg = str(ei.value)
    assert "5" in msg and "7" in msg  # both the actual and the expected count


def test_load_all_records_more_than_expected_raises_both_numbers(monkeypatch):
    recs = [_fake_raw(i) for i in range(9)]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs))
    with pytest.raises(ValueError) as ei:
        ev.load_all_records(_loader_config(7))
    msg = str(ei.value)
    assert "9" in msg and "7" in msg


def test_load_all_records_drains_reader_to_eof(monkeypatch):
    # more-than-expected is ONLY catchable if the reader is drained past the expected count;
    # a short-circuit at ``expected`` would leave records unconsumed and miss over-long sources.
    it = iter([_fake_raw(i) for i in range(9)])
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: it)
    with pytest.raises(ValueError):
        ev.load_all_records(_loader_config(7))
    assert list(it) == []  # the iterator was consumed to EOF


def test_load_all_records_rejects_nonpositive_expected(monkeypatch):
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter([_fake_raw(0)]))
    for bad in (0, -1):
        with pytest.raises(ValueError, match="validation_expected_count"):
            ev.load_all_records(_loader_config(bad))


class _BuildReached(Exception):
    """Sentinel raised by the build spy to prove the code reached state construction."""


def test_evaluate_asserts_count_before_building_state(monkeypatch):
    # MUTANT (ii): moving the count assertion after the first restore would let the build
    # run on a wrong-count dataset. The spy raises the moment build is reached, so:
    #   (a) matching count -> build IS reached (non-vacuity: the sentinel fires);
    #   (b) wrong count    -> load raises FIRST, naming both numbers; build never runs.
    monkeypatch.setenv("TRAIN_COMMIT", "TSHA")  # full mode: satisfy the F2 commit guard

    def _spy_build(config):
        raise _BuildReached()

    monkeypatch.setattr(gen, "_build_full_ft_validation_state", _spy_build)

    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter([_fake_raw(i) for i in range(7)]))
    with pytest.raises(_BuildReached):
        ev.evaluate(_eval_config(validation_expected_count=7))

    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter([_fake_raw(i) for i in range(5)]))
    with pytest.raises(ValueError) as ei:
        ev.evaluate(_eval_config(validation_expected_count=7))
    assert "5" in str(ei.value) and "7" in str(ei.value)  # the COUNT error, not the build sentinel


# ======================================================================================
# 2. assemble_batch -- deterministic (t, eps) keyed by POSITION (mutant i).
# ======================================================================================


def test_assemble_batch_rng_indexed_by_position_not_stored_ordinal():
    seed, num_steps = 0, 25
    recs = _loaded_records(8)
    positions = [0, 5, 3, 1]  # arbitrary order, incl. non-contiguous
    z_i0, z_video, eps, sigma_t = ev.assemble_batch(recs, positions, seed, num_steps, _SIGMAS)
    assert eps.shape == (len(positions), *_EXAMPLE_SHAPE) and sigma_t.shape == (len(positions),)
    for i, p in enumerate(positions):
        t_ref, eps_ref = ev.per_example_rng(seed, p, num_steps, _EXAMPLE_SHAPE)
        np.testing.assert_array_equal(eps[i], np.asarray(eps_ref))  # eps drawn by POSITION p
        np.testing.assert_array_equal(sigma_t[i], _SIGMAS[int(t_ref)])  # sigma by POSITION p
        # data comes from records[p] ...
        assert int(z_video[i].flat[0]) == p and int(z_i0[i].flat[0]) == p
        # ... but the draw is NOT keyed by the stored ordinal (which would give other bits).
        _, eps_ord = ev.per_example_rng(seed, recs[p]["ordinal"], num_steps, _EXAMPLE_SHAPE)
        assert not np.allclose(eps[i], np.asarray(eps_ord))


def test_assemble_batch_rng_independent_of_batch_size():
    seed, num_steps = 0, 25
    recs = _loaded_records(8)
    _, _, eps8, sig8 = ev.assemble_batch(recs, list(range(8)), seed, num_steps, _SIGMAS)
    _, _, epsA, sigA = ev.assemble_batch(recs, [0, 1, 2, 3], seed, num_steps, _SIGMAS)
    _, _, epsB, sigB = ev.assemble_batch(recs, [4, 5, 6, 7], seed, num_steps, _SIGMAS)
    np.testing.assert_array_equal(eps8[:4], epsA)  # B=8 vs B=4 rebatch -> identical per position
    np.testing.assert_array_equal(eps8[4:], epsB)
    np.testing.assert_array_equal(sig8[:4], sigA)
    np.testing.assert_array_equal(sig8[4:], sigB)


def test_rng_follows_position_across_reader_reorder(monkeypatch):
    # Two DIFFERENT reader contents at the same positions: the (t, eps) per position is the
    # SAME (a pure function of position), while the DATA at each position differs. Proves the
    # RNG follows dataset position, never the record content / iteration order.
    seed, num_steps = 0, 25
    recs_a_raw = [_fake_raw(i, fill=i, ordinal=1000 + 7 * i) for i in range(8)]
    recs_b_raw = [_fake_raw(i, fill=100 + i, ordinal=5000 + i) for i in range(8)]
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs_a_raw))
    recs_a = ev.load_all_records(_loader_config(8))
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter(recs_b_raw))
    recs_b = ev.load_all_records(_loader_config(8))

    posn = list(range(8))
    _, zv_a, eps_a, sig_a = ev.assemble_batch(recs_a, posn, seed, num_steps, _SIGMAS)
    _, zv_b, eps_b, sig_b = ev.assemble_batch(recs_b, posn, seed, num_steps, _SIGMAS)
    np.testing.assert_array_equal(eps_a, eps_b)  # identical (t, eps) per position ...
    np.testing.assert_array_equal(sig_a, sig_b)
    assert [int(zv_a[i].flat[0]) for i in range(8)] == list(range(8))  # ... over DIFFERENT data
    assert [int(zv_b[i].flat[0]) for i in range(8)] == [100 + i for i in range(8)]


# ======================================================================================
# 3. _eval_batch_per_example_loss -- one plain call, objective parity, frame-0 pin.
# ======================================================================================


def test_eval_step_objective_one_plain_call_no_actions_and_masked_mse():
    b = 4
    gain = 0.3
    state = _stub_state(gain, null_l=2, null_d=8)
    k = jax.random.split(jax.random.key(1), 4)
    z_i0 = jax.random.normal(k[0], (b, _C, 1, _H, _W), dtype=jnp.float32)
    z_video = jax.random.normal(k[1], (b, _C, _F, _H, _W), dtype=jnp.float32)
    eps = jax.random.normal(k[2], (b, _C, _F, _H, _W), dtype=jnp.float32)
    sigma_t = jax.random.uniform(k[3], (b,), dtype=jnp.float32)
    config = SimpleNamespace(weights_dtype="float32", activations_dtype="float32")
    ntt = 1000

    loss = np.asarray(
        ev._eval_batch_per_example_loss(state, z_i0, z_video, eps, sigma_t, config=config, num_train_timesteps=ntt)
    )
    assert loss.shape == (b,) and loss.dtype == np.float32

    # Exactly ONE plain transformer call -- no adapter forward, no CFG uncond branch.
    assert len(_CALLS) == 1
    call = _CALLS[0]
    assert set(call.keys()) == {"hidden_states", "timestep", "encoder_hidden_states", "deterministic"}
    assert "actions" not in call
    assert not any("adapter" in key or "action" in key for key in call)
    assert call["deterministic"] is True

    # null context broadcast to [B, L, D] in the activations dtype.
    nc = np.asarray(call["encoder_hidden_states"])
    assert nc.shape == (b, 2, 8) and call["encoder_hidden_states"].dtype == jnp.float32

    # z_t frame 0 is pinned to z_i0 (bitwise; weights_dtype float32 makes the cast identity).
    hs = np.asarray(call["hidden_states"])
    np.testing.assert_array_equal(hs[:, :, :1], np.asarray(z_i0))

    # timestep encodes sigma_t * num_train_timesteps on the future tokens, 0 on frame-0 tokens.
    ts = np.asarray(call["timestep"])
    tpf = (_H // 2) * (_W // 2)
    assert ts.shape == (b, _F * tpf)
    step_t = np.asarray(sigma_t) * ntt
    np.testing.assert_allclose(ts[:, tpf:], np.broadcast_to(step_t[:, None], (b, _F * tpf - tpf)), rtol=1e-5)
    assert np.all(ts[:, :tpf] == 0.0)

    # Per-example loss == hand-computed frame-0-masked MSE of (gain*z_t) vs (eps - z_video).
    sig = np.asarray(sigma_t).reshape(b, 1, 1, 1, 1)
    z_t = (1.0 - sig) * np.asarray(z_video) + sig * np.asarray(eps)
    z_t[:, :, :1] = np.asarray(z_i0)  # frame-0 pin
    diff = gain * z_t - (np.asarray(eps) - np.asarray(z_video))
    diff[:, :, :1] = 0.0  # frame-0 masked out
    denom = _C * (_F - 1) * _H * _W
    expected = (diff**2).reshape(b, -1).sum(axis=1) / denom
    np.testing.assert_allclose(loss, expected, rtol=1e-5, atol=1e-6)


# ======================================================================================
# 4. _evaluate_all_checkpoints -- restore-in-order, mismatch fatal, padded-tail excluded.
# ======================================================================================


def _fake_restore_factory(seen):
    def _restore(config, state, ckpt_dir, *, cohort_mode=False, requested_step=None):
        seen.append(requested_step)
        return SimpleNamespace(step=requested_step), requested_step

    return _restore


def _position_loss_eval_step(state, z_i0, z_video, eps, sigma_t):
    # Per-example loss == the position value packed into z_video (fill == position).
    return jnp.asarray(z_video[:, 0, 0, 0, 0], dtype=jnp.float32)


def test_loop_restores_each_step_in_order_and_excludes_padded_tail(monkeypatch):
    # MUTANT (iii): if the loop fed all-True validity to aggregate, the 3 padded duplicates
    # of position 36 would be counted -> n=40 != 37 -> aggregate raises. Honoring validity
    # gives n=37 and the golden mean over positions 0..36.
    recs = _loaded_records(37)
    steps = [2500, 5000, 7500]
    seen: list[int] = []
    monkeypatch.setattr(gen, "_restore_checkpoint_state", _fake_restore_factory(seen))

    rows = ev._evaluate_all_checkpoints(
        _eval_config(),
        recs,
        steps,
        "/fake/ckpts",
        "INIT_STATE",
        _position_loss_eval_step,
        seed=0,
        num_steps=25,
        sigmas=_SIGMAS,
        batch=8,
        expected_count=37,
        dataset_path="gs://fake/val",
        checkpoint_path="/fake/ckpts",
        train_commit="TRAIN_SHA",
        eval_commit="EVAL_SHA",
    )

    assert seen == steps  # restore called with each requested_step, in order
    assert [r["checkpoint_step"] for r in rows] == steps
    golden_mean = float(np.mean(range(37)))
    for r in rows:
        assert r["n"] == 37  # padded tail excluded
        assert r["mean_loss"] == golden_mean
        assert r["validation_seed"] == 0
        assert r["dataset_path"] == "gs://fake/val" and r["checkpoint_path"] == "/fake/ckpts"
        assert r["train_commit"] == "TRAIN_SHA" and r["eval_commit"] == "EVAL_SHA"
        assert list(r.keys()) == _COLUMNS


def test_loop_raises_on_returned_step_mismatch(monkeypatch):
    def _bad(config, state, ckpt_dir, *, cohort_mode=False, requested_step=None):
        return SimpleNamespace(step=requested_step), 999  # returned scalar disagrees with request

    monkeypatch.setattr(gen, "_restore_checkpoint_state", _bad)
    with pytest.raises(ValueError, match="mismatch"):
        ev._evaluate_all_checkpoints(
            _eval_config(),
            _loaded_records(8),
            [2500],
            "/d",
            "I",
            lambda *a: jnp.zeros((8,), jnp.float32),
            seed=0,
            num_steps=25,
            sigmas=_SIGMAS,
            batch=8,
            expected_count=8,
            dataset_path="",
            checkpoint_path="",
            train_commit="",
            eval_commit="",
        )


def test_loop_raises_on_state_step_mismatch(monkeypatch):
    def _bad(config, state, ckpt_dir, *, cohort_mode=False, requested_step=None):
        return SimpleNamespace(step=111), requested_step  # returned scalar OK, state.step wrong

    monkeypatch.setattr(gen, "_restore_checkpoint_state", _bad)
    with pytest.raises(ValueError, match="mismatch"):
        ev._evaluate_all_checkpoints(
            _eval_config(),
            _loaded_records(8),
            [2500],
            "/d",
            "I",
            lambda *a: jnp.zeros((8,), jnp.float32),
            seed=0,
            num_steps=25,
            sigmas=_SIGMAS,
            batch=8,
            expected_count=8,
            dataset_path="",
            checkpoint_path="",
            train_commit="",
            eval_commit="",
        )


def test_loop_reuses_identical_rng_across_checkpoints(monkeypatch):
    # Cross-checkpoint (t, eps) identity THROUGH the loop: capture the eps/sigma fed to the
    # eval step per batch per checkpoint; the two passes must be bitwise-identical per batch.
    recs = _loaded_records(20)
    monkeypatch.setattr(
        gen,
        "_restore_checkpoint_state",
        lambda config, state, d, *, cohort_mode=False, requested_step=None: (
            SimpleNamespace(step=requested_step),
            requested_step,
        ),
    )
    recorded: list[tuple[int, np.ndarray, np.ndarray]] = []

    def _rec_eval(state, z_i0, z_video, eps, sigma_t):
        recorded.append((int(state.step), np.asarray(eps).copy(), np.asarray(sigma_t).copy()))
        return jnp.zeros((z_video.shape[0],), jnp.float32)

    ev._evaluate_all_checkpoints(
        _eval_config(),
        recs,
        [2500, 5000],
        "/d",
        "I",
        _rec_eval,
        seed=0,
        num_steps=25,
        sigmas=_SIGMAS,
        batch=8,
        expected_count=20,
        dataset_path="",
        checkpoint_path="",
        train_commit="",
        eval_commit="",
    )
    by_step: dict[int, list] = {}
    for step, eps, sig in recorded:
        by_step.setdefault(step, []).append((eps, sig))
    a, b = by_step[2500], by_step[5000]
    assert len(a) == len(b) == 3  # ceil(20/8) == 3 batches
    for (eps_a, sig_a), (eps_b, sig_b) in zip(a, b):
        np.testing.assert_array_equal(eps_a, eps_b)  # SAME noise across checkpoints
        np.testing.assert_array_equal(sig_a, sig_b)  # SAME sigma across checkpoints


def test_loop_smoke_limits_batches_and_checkpoint_and_skips_count_assert(monkeypatch):
    # SMOKE (plan F5): only the first N batches and the FIRST checkpoint, and the n==expected
    # assertion is SKIPPED (aggregate counts whatever the smoke subset yields, here 16 != 37).
    recs = _loaded_records(37)
    monkeypatch.setattr(
        gen,
        "_restore_checkpoint_state",
        lambda config, state, d, *, cohort_mode=False, requested_step=None: (
            SimpleNamespace(step=requested_step),
            requested_step,
        ),
    )
    rows = ev._evaluate_all_checkpoints(
        _eval_config(),
        recs,
        [2500, 5000, 7500],
        "/d",
        "I",
        lambda s, zi, zv, e, st: jnp.zeros((zv.shape[0],), jnp.float32),
        seed=0,
        num_steps=25,
        sigmas=_SIGMAS,
        batch=8,
        expected_count=37,
        dataset_path="",
        checkpoint_path="",
        train_commit="",
        eval_commit="",
        smoke_limit=2,
    )
    assert len(rows) == 1  # only the first checkpoint
    assert rows[0]["checkpoint_step"] == 2500
    assert rows[0]["n"] == 16  # 2 batches x 8, NOT the full 37 (assertion skipped)


# ======================================================================================
# 5. _restore_checkpoint_state requested_step kwarg (real Orbax; mutant iv).
# ======================================================================================


def test_requested_step_kwarg_overrides_config(tmp_path):
    ckpt_dir = str(tmp_path / "ck")
    trainer = full_ft.WanTI2VFullFTTrainer(_ckpt_config(checkpoint_step=-1, ckpt_dir=ckpt_dir))
    mgr = trainer._build_checkpoint_manager(ckpt_dir)
    trainer._save_checkpoint(mgr, 3, _stub_state(gain=0.11))
    trainer._save_checkpoint(mgr, 7, _stub_state(gain=0.77))
    mgr.wait_until_finished()
    mgr.close()

    # MUTANT (iv): config says 3, requested_step says 7 -> must restore 7 (kwarg wins).
    r7, s7 = gen._restore_checkpoint_state(
        _ckpt_config(checkpoint_step=3, ckpt_dir=ckpt_dir),
        _stub_state(gain=0.5),
        ckpt_dir,
        cohort_mode=True,
        requested_step=7,
    )
    assert s7 == 7 and int(r7.step) == 7
    assert float(_first_leaf(r7.params)) == pytest.approx(0.77, rel=1e-6)

    # The other direction: config says 7, requested_step says 3 -> restore 3.
    r3, s3 = gen._restore_checkpoint_state(
        _ckpt_config(checkpoint_step=7, ckpt_dir=ckpt_dir),
        _stub_state(gain=0.5),
        ckpt_dir,
        cohort_mode=True,
        requested_step=3,
    )
    assert s3 == 3 and float(_first_leaf(r3.params)) == pytest.approx(0.11, rel=1e-6)


def test_requested_step_none_is_config_driven_byte_identical(tmp_path):
    ckpt_dir = str(tmp_path / "ck")
    trainer = full_ft.WanTI2VFullFTTrainer(_ckpt_config(checkpoint_step=-1, ckpt_dir=ckpt_dir))
    mgr = trainer._build_checkpoint_manager(ckpt_dir)
    trainer._save_checkpoint(mgr, 3, _stub_state(gain=0.11))
    trainer._save_checkpoint(mgr, 7, _stub_state(gain=0.77))
    mgr.wait_until_finished()
    mgr.close()

    # requested_step defaulted to None -> the config's checkpoint_step drives selection.
    r, s = gen._restore_checkpoint_state(
        _ckpt_config(checkpoint_step=7, ckpt_dir=ckpt_dir), _stub_state(gain=0.5), ckpt_dir, cohort_mode=True
    )
    assert s == 7 and float(_first_leaf(r.params)) == pytest.approx(0.77, rel=1e-6)


def test_requested_step_zero_is_pretrained_baseline_bypass(monkeypatch):
    # requested_step=0 in cohort mode is the pretrained baseline: Orbax is never built.
    class _NoManager:
        def __init__(self, *a, **k):
            raise AssertionError("Orbax CheckpointManager built under requested_step=0 bypass")

    monkeypatch.setattr(gen.ocp, "CheckpointManager", _NoManager)
    fresh = _stub_state(gain=0.5)
    r, s = gen._restore_checkpoint_state(
        _ckpt_config(checkpoint_step=999, ckpt_dir="/nope"), fresh, "/nope", cohort_mode=True, requested_step=0
    )
    assert s == 0 and r is fresh


# ======================================================================================
# 6. write_outputs / plot_rows / plot-only CLI / smoke output dir.
# ======================================================================================


def _sample_rows():
    return [
        ev._make_row(
            step,
            {"mean_loss": mean, "stderr": err, "n": 37},
            seed=0,
            dataset_path="gs://d",
            checkpoint_path="gs://c",
            train_commit="TRAIN",
            eval_commit="EVAL",
        )
        for step, mean, err in [(2500, 0.5, 0.01), (5000, 0.4, 0.008)]
    ]


def test_write_outputs_json_and_csv_9_columns_exact(tmp_path):
    rows = _sample_rows()
    out = str(tmp_path / "vl")
    ev.write_outputs(rows, out)

    with open(f"{out}/val_loss.json") as fh:
        data = json.load(fh)
    assert isinstance(data, list) and len(data) == 2
    for row in data:
        assert list(row.keys()) == _COLUMNS  # exact names AND order
    assert data[0]["checkpoint_step"] == 2500 and data[0]["mean_loss"] == 0.5
    assert data[1]["checkpoint_step"] == 5000
    # F2: the commit provenance propagates EXACTLY into every JSON row.
    assert all(row["train_commit"] == "TRAIN" and row["eval_commit"] == "EVAL" for row in data)

    with open(f"{out}/val_loss.csv", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        body = list(reader)
    assert header == _COLUMNS  # exact names AND order
    assert len(body) == 2 and body[0][0] == "2500" and body[1][0] == "5000"
    # F2: ... and into the exact CSV cells (columns 8/9 of the 9-column schema).
    assert all(cells[7] == "TRAIN" and cells[8] == "EVAL" for cells in body)


def test_write_outputs_writes_plot_when_matplotlib_present(tmp_path):
    pytest.importorskip("matplotlib")
    out = str(tmp_path / "vl")
    ev.write_outputs(_sample_rows(), out)
    assert os.path.exists(f"{out}/val_loss_plot.png")


def test_write_outputs_json_csv_survive_absent_matplotlib(tmp_path):
    # The plot is import-guarded: JSON/CSV are always written; the PNG is skipped (not fatal)
    # when matplotlib is unavailable (the recorded plot-only regeneration covers it later).
    out = str(tmp_path / "vl")
    ev.write_outputs(_sample_rows(), out)
    assert os.path.exists(f"{out}/val_loss.json") and os.path.exists(f"{out}/val_loss.csv")
    try:
        import matplotlib  # noqa: F401

        has_mpl = True
    except Exception:
        has_mpl = False
    if not has_mpl:
        assert not os.path.exists(f"{out}/val_loss_plot.png")


def test_plot_only_cli_regenerates_png_from_json(tmp_path):
    pytest.importorskip("matplotlib")
    out = str(tmp_path / "vl")
    ev.write_outputs(_sample_rows(), out)
    png = str(tmp_path / "regen.png")
    ev.main(["prog", "plot-only", f"{out}/val_loss.json", png])
    assert os.path.exists(png)


def test_plot_only_cli_bad_arity_raises():
    with pytest.raises(ValueError, match="plot-only"):
        ev.main(["prog", "plot-only", "only-one-arg"])


def test_resolve_output_dir_default_explicit_and_smoke():
    cfg = SimpleNamespace(output_dir="gs://out", run_name="runX", validation_loss_output_dir="")
    assert ev._resolve_output_dir(cfg, smoke=False).endswith("/runX/validation_loss")
    assert ev._resolve_output_dir(cfg, smoke=True).endswith("/runX/validation_loss_smoke")
    cfg2 = SimpleNamespace(output_dir="gs://out", run_name="runX", validation_loss_output_dir="gs://explicit/vl")
    assert ev._resolve_output_dir(cfg2, smoke=False) == "gs://explicit/vl"
    assert ev._resolve_output_dir(cfg2, smoke=True) == "gs://explicit/vl_smoke"


# ======================================================================================
# 7. VAE / text-encoder deletion after state construction (plan F5).
# ======================================================================================


def test_free_rollout_modules_deletes_the_four_and_keeps_others():
    pipe = SimpleNamespace(
        vae=object(), vae_cache=object(), text_encoder=object(), tokenizer=object(), transformer=object()
    )
    ev._free_rollout_modules(pipe)
    for attr in ("vae", "vae_cache", "text_encoder", "tokenizer"):
        assert not hasattr(pipe, attr)
    assert hasattr(pipe, "transformer")  # unrelated attrs untouched


def test_free_rollout_modules_tolerates_missing_attrs():
    pipe = SimpleNamespace(vae=object())  # only vae present
    ev._free_rollout_modules(pipe)  # must not raise on the absent ones
    assert not hasattr(pipe, "vae")


def test_build_and_free_state_drops_rollout_modules_after_construction(monkeypatch):
    pipe = SimpleNamespace(vae=object(), vae_cache=object(), text_encoder=object(), tokenizer=object())
    monkeypatch.setattr(gen, "_build_full_ft_validation_state", lambda config: ("TR", pipe, "MESH", "STATE", "SH"))
    trainer, out_pipe, mesh, state, shardings = ev._build_and_free_state(_eval_config())
    assert out_pipe is pipe and (trainer, mesh, state, shardings) == ("TR", "MESH", "STATE", "SH")
    for attr in ("vae", "vae_cache", "text_encoder", "tokenizer"):
        assert not hasattr(out_pipe, attr)  # gone right after state construction


# ======================================================================================
# Misc: config parsing, main guards, commit resolution.
# ======================================================================================


def test_parse_checkpoint_steps_ok_and_rejections():
    assert ev._parse_checkpoint_steps("2500,5000,7500") == [2500, 5000, 7500]
    assert ev._parse_checkpoint_steps(" 2500 , 5000 ") == [2500, 5000]  # whitespace tolerant
    for bad in ("", "   ", "2500,0", "2500,-1", "2500,x"):
        with pytest.raises(ValueError):
            ev._parse_checkpoint_steps(bad)


def test_assert_full_ft_rejects_wrong_model_type():
    with pytest.raises(ValueError, match="FULL_FT_TI2V"):
        ev._assert_full_ft(SimpleNamespace(model_type="SIDE_ADAPTER_TI2V"))


def test_assert_full_ft_enforces_probe_config():
    with pytest.raises(ValueError, match="guide_scale"):
        ev._assert_full_ft(
            SimpleNamespace(model_type="FULL_FT_TI2V", side_adapter_guide_scale=5.0, side_adapter_noise_mode="fresh")
        )
    # A valid full-FT config passes both guards.
    ev._assert_full_ft(
        SimpleNamespace(model_type="FULL_FT_TI2V", side_adapter_guide_scale=1.0, side_adapter_noise_mode="fresh")
    )


def test_resolve_commits_reads_env(monkeypatch):
    monkeypatch.setenv("TRAIN_COMMIT", "TSHA")
    monkeypatch.setenv("COMMIT", "ESHA")
    train, evc = ev._resolve_commits(SimpleNamespace())
    assert train == "TSHA" and evc == "ESHA"
    monkeypatch.delenv("TRAIN_COMMIT")
    train2, _ = ev._resolve_commits(SimpleNamespace())
    assert train2 == "unknown"  # default when neither env nor config provides it


# ======================================================================================
# 8. STRENGTHEN (cycle-B Codex review): F1 gather seam, F2 mandatory TRAIN_COMMIT,
#    F3 sigma-grid wiring + static wrapper mapping.
# ======================================================================================

# ---- F1: the _loss_to_host collective-gather seam (two-host v6e-8). ----


def test_loss_to_host_fully_addressable_skips_collective(monkeypatch):
    gathered = []
    monkeypatch.setattr(
        ev, "multihost_utils", SimpleNamespace(process_allgather=lambda *a, **k: gathered.append(a) or np.zeros(1))
    )
    x = jnp.arange(8, dtype=jnp.float32)  # single-process CPU array: fully addressable
    out = ev._loss_to_host(x)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.float32))
    assert gathered == []  # no collective on the addressable path
    y = np.arange(3.0)  # plain numpy: no is_fully_addressable attr -> host path
    np.testing.assert_array_equal(ev._loss_to_host(y), y)
    assert gathered == []


def test_loss_to_host_non_addressable_routes_through_collective(monkeypatch):
    # F1: a value the local process cannot fully address (the jitted [B] output sharded over
    # the GLOBAL two-host mesh) MUST go through process_allgather(..., tiled=True) -- a direct
    # np.asarray would raise on the real topology, so the fake pins the gather routing.
    seen = {}

    def _fake_gather(value, tiled=False):
        seen["value"] = value
        seen["tiled"] = tiled
        return np.arange(8, dtype=np.float32)

    monkeypatch.setattr(ev, "multihost_utils", SimpleNamespace(process_allgather=_fake_gather))

    class _FakeSharded:
        is_fully_addressable = False

    fake = _FakeSharded()
    out = ev._loss_to_host(fake)
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.float32))
    assert seen["value"] is fake  # the sharded value itself was gathered
    assert seen["tiled"] is True  # tiled=True: concatenate shards, not stack processes


def test_loop_invokes_gather_seam_per_batch(monkeypatch):
    # MUTANT (alpha): bypassing the seam (direct np.asarray in the loop) leaves the spy at
    # zero invocations. Correct behavior: one _loss_to_host call per batch per checkpoint.
    recs = _loaded_records(20)
    monkeypatch.setattr(
        gen,
        "_restore_checkpoint_state",
        lambda config, state, d, *, cohort_mode=False, requested_step=None: (
            SimpleNamespace(step=requested_step),
            requested_step,
        ),
    )
    calls = []
    real = ev._loss_to_host

    def _spy(loss):
        calls.append(loss)
        return real(loss)

    monkeypatch.setattr(ev, "_loss_to_host", _spy)
    rows = ev._evaluate_all_checkpoints(
        _eval_config(),
        recs,
        [2500, 5000],
        "/d",
        "I",
        lambda s, zi, zv, e, st: jnp.zeros((zv.shape[0],), jnp.float32),
        seed=0,
        num_steps=25,
        sigmas=_SIGMAS,
        batch=8,
        expected_count=20,
        dataset_path="",
        checkpoint_path="",
        train_commit="T",
        eval_commit="E",
    )
    assert len(calls) == 6  # ceil(20/8)=3 batches x 2 checkpoints: the seam runs PER BATCH
    assert rows[0]["n"] == rows[1]["n"] == 20


# ---- F2: TRAIN_COMMIT mandatory (module layer + wrapper layer + propagation). ----


def test_require_train_commit_full_mode_rejects_empty_and_unknown():
    for bad in ("", "unknown"):
        with pytest.raises(ValueError, match="TRAIN_COMMIT"):
            ev._require_train_commit(bad, smoke=False)
    ev._require_train_commit("abc123", smoke=False)  # a real SHA passes


def test_require_train_commit_smoke_mode_relaxed():
    # Documented relaxation: smoke outputs go to the isolated validation_loss_smoke/ dir and
    # are fit-probe evidence only, never T1 acceptance artifacts (see the helper docstring).
    ev._require_train_commit("unknown", smoke=True)
    ev._require_train_commit("", smoke=True)


def test_evaluate_full_mode_refuses_missing_train_commit_before_any_work(monkeypatch):
    # MUTANT (beta): tolerating an empty train_commit in full mode would reach the reader /
    # builder. Correct behavior: ValueError BEFORE the drain and BEFORE any state work.
    def _no_reader(config):
        raise AssertionError("reader consulted despite missing train_commit")

    def _no_build(config):
        raise AssertionError("state built despite missing train_commit")

    monkeypatch.setattr(gen, "_iter_parsed_records", _no_reader)
    monkeypatch.setattr(gen, "_build_full_ft_validation_state", _no_build)
    with pytest.raises(ValueError, match="TRAIN_COMMIT"):
        ev.evaluate(_eval_config())  # _clean_env guarantees TRAIN_COMMIT is unset -> "unknown"


def test_evaluate_smoke_mode_relaxes_train_commit(monkeypatch):
    # Same missing TRAIN_COMMIT, but SMOKE_LIMIT set: the guard is relaxed and evaluation
    # proceeds to the state build (the sentinel fires), per the documented smoke exemption.
    monkeypatch.setenv("SMOKE_LIMIT", "2")
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter([_fake_raw(i) for i in range(7)]))

    def _spy_build(config):
        raise _BuildReached()

    monkeypatch.setattr(gen, "_build_full_ft_validation_state", _spy_build)
    with pytest.raises(_BuildReached):
        ev.evaluate(_eval_config())


def test_wrapper_rejects_missing_train_commit_before_python(tmp_path):
    # F2 layer 1: the wrapper fails fast (bash ${TRAIN_COMMIT:?...}) with RUN_NAME set but
    # TRAIN_COMMIT missing -- and python is NEVER invoked (the fake python leaves a marker).
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    marker = tmp_path / "python_ran"
    fake_python = fakebin / "python"
    fake_python.write_text(f"#!/bin/bash\ntouch {marker}\n")
    fake_python.chmod(0o755)
    env = {"PATH": f"{fakebin}:/usr/bin:/bin", "HOME": str(tmp_path), "RUN_NAME": "r1"}
    proc = subprocess.run(["bash", _WRAPPER], env=env, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode != 0
    assert "TRAIN_COMMIT" in proc.stderr  # the clear pre-python error names the variable
    assert not marker.exists()  # python was never reached


# ---- F3: sigma-grid wiring seam + evaluate-level wiring + static wrapper mapping. ----


def test_build_sigma_grid_wiring_pins_all_four_args(monkeypatch):
    # MUTANT (gamma): dropping/hardcoding flow_shift (or swapping sigma_min/max) breaks BOTH
    # the recorded call args and the value equality against the directly-built grid.
    calls = []

    def _spy(num, shift, smin, smax):
        calls.append((num, shift, smin, smax))
        return build_rollout_sigmas(num, shift, smin, smax)

    monkeypatch.setattr(ev, "build_rollout_sigmas", _spy)
    cfg = SimpleNamespace(side_adapter_sampling_steps=13, flow_shift=3.0)
    sched = SimpleNamespace(config=SimpleNamespace(sigma_min=0.25, sigma_max=0.75, num_train_timesteps=777))
    grid = ev._build_sigma_grid(cfg, sched)
    assert calls == [(13, 3.0, 0.25, 0.75)]  # ALL FOUR args, from their contract sources
    np.testing.assert_array_equal(grid, np.asarray(build_rollout_sigmas(13, 3.0, 0.25, 0.75)))
    assert grid.shape == (14,) and isinstance(grid, np.ndarray)  # N+1 grid on host


def test_assemble_batch_passes_num_steps_through_to_rng():
    # Wiring-level pin of uniform t over a NON-25 grid: t_idx must come from
    # per_example_rng(seed, p, num_steps) and sigma_t from THAT grid (cycle-A pins the
    # randint bounds/support; this pins the evaluator passes num_steps through unchanged).
    num_steps = 13
    sigmas13 = np.asarray(build_rollout_sigmas(num_steps, 3.0, 0.25, 0.75))
    recs = _loaded_records(6)
    _, _, eps, sigma_t = ev.assemble_batch(recs, list(range(6)), 0, num_steps, sigmas13)
    for i in range(6):
        t_ref, eps_ref = ev.per_example_rng(0, i, num_steps, _EXAMPLE_SHAPE)
        assert 0 <= int(t_ref) < num_steps
        np.testing.assert_array_equal(eps[i], np.asarray(eps_ref))
        np.testing.assert_array_equal(sigma_t[i], sigmas13[int(t_ref)])


def test_evaluate_wires_sigma_grid_scheduler_and_loop_inputs(monkeypatch, tmp_path):
    # Evaluate-level wiring capture: the sigma grid inside evaluate() is built from the
    # config's sampling steps + flow_shift and the SCHEDULER's sigma_min/max (distinct
    # values so any swap/hardcode is caught), and the loop receives exactly the resolved
    # num_steps/sigmas/batch/seed/commits/paths. The loop itself is faked; outputs are
    # written for real (write path covered end-to-end).
    monkeypatch.setenv("TRAIN_COMMIT", "TSHA")
    monkeypatch.setenv("COMMIT", "ESHA")
    monkeypatch.setattr(gen, "_iter_parsed_records", lambda config: iter([_fake_raw(i) for i in range(7)]))

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=3.0, sigma_min=0.25, sigma_max=0.75)
    fake_trainer = SimpleNamespace(_create_scheduler=lambda: (scheduler, None))
    pipe = SimpleNamespace(vae=object(), vae_cache=object(), text_encoder=object(), tokenizer=object())
    monkeypatch.setattr(
        gen, "_build_full_ft_validation_state", lambda config: (fake_trainer, pipe, mesh, "STATE", None)
    )

    sigma_calls = []

    def _spy_sigmas(num, shift, smin, smax):
        sigma_calls.append((num, shift, smin, smax))
        return build_rollout_sigmas(num, shift, smin, smax)

    monkeypatch.setattr(ev, "build_rollout_sigmas", _spy_sigmas)

    captured = {}

    def _fake_loop(config, records, steps, ckpt_dir, state, eval_step_fn, **kw):
        captured.update(kw)
        captured["steps"] = steps
        captured["ckpt_dir"] = ckpt_dir
        captured["state"] = state
        captured["n_records"] = len(records)
        return _sample_rows()

    monkeypatch.setattr(ev, "_evaluate_all_checkpoints", _fake_loop)

    out_root = str(tmp_path / "vl")
    cfg = _eval_config(
        validation_loss_output_dir=out_root,
        flow_shift=3.0,
        side_adapter_sampling_steps=13,
        global_batch_size_to_train_on=4,
    )
    rows, out_dir = ev.evaluate(cfg)

    assert sigma_calls == [(13, 3.0, 0.25, 0.75)]  # config steps+shift, SCHEDULER sigma_min/max
    assert captured["num_steps"] == 13
    np.testing.assert_array_equal(captured["sigmas"], np.asarray(build_rollout_sigmas(13, 3.0, 0.25, 0.75)))
    assert captured["batch"] == 4 and captured["seed"] == 0
    assert captured["steps"] == [2500, 5000] and captured["expected_count"] == 7
    assert captured["train_commit"] == "TSHA" and captured["eval_commit"] == "ESHA"
    assert captured["dataset_path"] == "gs://fake/val" and captured["ckpt_dir"] == "/fake/ckpts"
    assert captured["checkpoint_path"] == "/fake/ckpts"
    assert captured["state"] == "STATE" and captured["n_records"] == 7
    assert captured["smoke_limit"] is None  # full mode
    # The rollout-only modules were freed right after the (faked) build (plan F5)...
    assert not hasattr(pipe, "vae") and not hasattr(pipe, "text_encoder")
    # ...and the returned rows were written for real to the resolved output dir.
    assert out_dir == out_root and rows == _sample_rows()
    assert os.path.exists(f"{out_root}/val_loss.json") and os.path.exists(f"{out_root}/val_loss.csv")


def test_evaluate_uses_scheduler_num_train_timesteps_source():
    # Deletion guard for the step_t scale source: the jitted step must receive the
    # SCHEDULER's num_train_timesteps (behaviorally pinned as sigma*ntt on the timestep
    # tokens by the eval-step objective test); this pins evaluate() wiring it from
    # scheduler.config, not a constant.
    import inspect

    src = inspect.getsource(ev.evaluate)
    assert "int(scheduler.config.num_train_timesteps)" in src
    assert "num_train_timesteps=num_train_timesteps" in src
    assert "_build_sigma_grid(config, scheduler)" in src  # the F3 seam is the ONLY grid source


# ---- F3 (static): wrapper env -> config override mapping, grep-style + fake-python. ----

_WRAPPER_NEEDLES = [
    # required env
    'RUN_NAME="${RUN_NAME:?',
    'TRAIN_COMMIT="${TRAIN_COMMIT:?',
    # defaults (exact values from the report table / plan)
    'OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft}"',
    'EVAL_DATA_DIR="${EVAL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"',
    'MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"',
    'CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"',
    'CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-2500,5000,7500,10000,12500,15000,17500,20000}"',
    'EXPECTED_COUNT="${EXPECTED_COUNT:-14636}"',
    'VALIDATION_SEED="${VALIDATION_SEED:-0}"',
    'PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"',
    'SMOKE_LIMIT="${SMOKE_LIMIT:-}"',
    # env-channel exports the module reads via os.environ
    "export SMOKE_LIMIT",
    "export TRAIN_COMMIT",
    "export COMMIT",
    # env -> pyconfig override mapping (one line per report-table row)
    "python src/maxdiffusion/eval_wan_full_ft_val_loss.py",
    "src/maxdiffusion/configs/base_wan_5b_full_ft.yml",
    'run_name="${RUN_NAME}"',
    'pretrained_model_name_or_path="${MODEL_DIR}"',
    'eval_data_dir="${EVAL_DATA_DIR}"',
    'output_dir="${OUTPUT_DIR}"',
    'base_output_directory="${OUTPUT_DIR}"',
    'checkpoint_dir="${CHECKPOINT_DIR}"',
    'validation_checkpoint_steps="${CHECKPOINT_STEPS}"',
    'validation_expected_count="${EXPECTED_COUNT}"',
    'validation_loss_output_dir="${VALIDATION_LOSS_OUTPUT_DIR}"',
    'validation_seed="${VALIDATION_SEED}"',
    'per_device_batch_size="${PER_DEVICE_BATCH_SIZE}"',
    "hardware=tpu",
]


@pytest.mark.parametrize("needle", _WRAPPER_NEEDLES, ids=lambda n: n[:48])
def test_wrapper_static_env_to_override_mapping(needle):
    with open(_WRAPPER) as fh:
        txt = fh.read()
    assert needle in txt


def test_wrapper_never_references_adapter_paths():
    with open(_WRAPPER) as fh:
        txt = fh.read()
    assert "generate_wan_side_adapter" not in txt  # the evaluator module, not the rollout script
    assert "base_wan_5b_side_adapter.yml" not in txt  # never the adapter config


def test_wrapper_bash_syntax_ok():
    proc = subprocess.run(["bash", "-n", _WRAPPER], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_wrapper_maps_env_to_overrides_via_fake_python(tmp_path):
    # Behavioral mapping proof (launcher-test precedent): a fake `python` captures the exact
    # argv the wrapper submits plus the exported env channel; every report-table row is
    # asserted against the capture.
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    capture = tmp_path / "argv.txt"
    fake_python = fakebin / "python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\n'
        '{ env | grep -E "^(TRAIN_COMMIT|COMMIT|SMOKE_LIMIT)="; } >> "$CAPTURE" || true\n'
    )
    fake_python.chmod(0o755)
    env = {
        "PATH": f"{fakebin}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "RUN_NAME": "r1",
        "TRAIN_COMMIT": "abc123",
        "CAPTURE": str(capture),
    }
    proc = subprocess.run(["bash", _WRAPPER], env=env, cwd=str(tmp_path), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = capture.read_text().splitlines()
    assert lines[0] == "src/maxdiffusion/eval_wan_full_ft_val_loss.py"  # the evaluator module
    for expected in [
        "src/maxdiffusion/configs/base_wan_5b_full_ft.yml",
        "run_name=r1",
        "pretrained_model_name_or_path=Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "eval_data_dir=gs://v6_east1d/datasets/droid_wan_side_adapter/val",
        "output_dir=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft",
        "base_output_directory=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft",
        "checkpoint_dir=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft/r1/checkpoints",
        "validation_checkpoint_steps=2500,5000,7500,10000,12500,15000,17500,20000",
        "validation_expected_count=14636",
        "validation_loss_output_dir=",
        "validation_seed=0",
        "per_device_batch_size=4",
        "hardware=tpu",
    ]:
        assert expected in lines, expected
    # The env channel: TRAIN_COMMIT propagates verbatim; COMMIT defaults to "unknown" in a
    # non-git cwd; SMOKE_LIMIT is exported (empty) so the module sees a FULL run.
    assert "TRAIN_COMMIT=abc123" in lines
    assert "COMMIT=unknown" in lines
    assert "SMOKE_LIMIT=" in lines
    assert not any("generate_wan_side_adapter" in ln for ln in lines)
