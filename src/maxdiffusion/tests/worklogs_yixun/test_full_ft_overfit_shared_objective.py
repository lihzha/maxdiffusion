"""CPU-only characterization tests for the shared full-FT / side-adapter objective.

exp_01 round 1 extracts the inline math of ``_denoising_loss`` into
``build_noisy_pinned_latents`` and ``masked_velocity_mse`` (in ``side_adapter_wan``)
so the side-adapter and full-finetune trainers share one objective -- parity by
shared code. The ``_reference_*`` functions below transcribe the *pre-refactor*
equations verbatim; the tests assert that (1) the helpers and (2) the refactored
trainer ``_denoising_loss`` itself (fixed rngs, stub adapter/transformer modules,
production ``guide_scale=5.0`` CFG path) reproduce them exactly. CPU-only: no
model weights, no TPU.
"""

from __future__ import annotations

# The darwin-only grain import stub (installed before importing the trainer, which
# pulls tensorflow + grain via input_pipeline_interface) now lives in conftest.py so
# both worklog test files share one copy.
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import maxdiffusion.trainers.wan_ti2v_side_adapter_trainer as trainer_mod
from maxdiffusion.models.wan.side_adapter_wan import (
    build_noisy_pinned_latents,
    build_rollout_sigmas,
    masked_velocity_mse,
    _build_per_token_timestep,
)
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

# Small, fixed shapes: batch B, channels C, latent frames F, height H, width W.
_B, _C, _F, _H, _W = 2, 3, 4, 5, 6


def _fixed_inputs(seed: int):
    """Deterministic float32 inputs shaped like the real cached-DROID latents."""
    k_video, k_i0, k_eps, k_sigma = jax.random.split(jax.random.key(seed), 4)
    z_video = jax.random.normal(k_video, (_B, _C, _F, _H, _W), dtype=jnp.float32)
    z_i0 = jax.random.normal(k_i0, (_B, _C, 1, _H, _W), dtype=jnp.float32)
    eps = jax.random.normal(k_eps, (_B, _C, _F, _H, _W), dtype=jnp.float32)
    # sigma in (0, 1), distinct per example (the fresh-noise regime).
    sigma_t = jax.random.uniform(k_sigma, (_B,), dtype=jnp.float32, minval=0.05, maxval=0.95)
    return z_video, z_i0, eps, sigma_t


def _reference_z_t(z_video, z_i0, eps, sigma_t):
    """Transcription of the pre-refactor ``z_t`` (noisy latents + frame-0 pin)."""
    b = z_video.shape[0]
    sigma_b = sigma_t.reshape((b, 1, 1, 1, 1))
    z_t = (1.0 - sigma_b) * z_video + sigma_b * eps
    # frame-0 pin: z_i0 carries a single latent frame.
    z_t = z_t.at[:, :, :1].set(z_i0[:, :, :1])
    return z_t


def _reference_loss(v_pred, v_target, batch_size):
    """Independent transcription of the pre-refactor masked velocity MSE."""
    _, c, f, h, w = v_pred.shape
    mask = jnp.ones((1, c, f, h, w), dtype=jnp.float32)
    mask = mask.at[:, :, :1, :, :].set(0.0)
    diff = (v_pred.astype(jnp.float32) - v_target.astype(jnp.float32)) * mask
    n_valid = jnp.maximum(jnp.sum(mask) * batch_size, 1.0)
    return jnp.sum(diff**2) / n_valid


# (a) Characterization: helpers == independently-transcribed reference equations.


def test_build_noisy_pinned_latents_matches_reference():
    z_video, z_i0, eps, sigma_t = _fixed_inputs(seed=1)
    got = build_noisy_pinned_latents(z_video, z_i0, eps, sigma_t)
    ref = _reference_z_t(z_video, z_i0, eps, sigma_t)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


def test_masked_velocity_mse_matches_reference():
    z_video, _, eps, _ = _fixed_inputs(seed=2)
    v_pred = jax.random.normal(jax.random.key(7), z_video.shape, dtype=jnp.float32)
    v_target = eps - z_video
    got = masked_velocity_mse(v_pred, v_target, _B)
    ref = _reference_loss(v_pred, v_target, _B)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))


# (b) Pin correctness.


def test_first_frame_is_pinned_exactly():
    z_video, z_i0, eps, sigma_t = _fixed_inputs(seed=3)
    got = build_noisy_pinned_latents(z_video, z_i0, eps, sigma_t)
    np.testing.assert_array_equal(np.asarray(got[:, :, 0]), np.asarray(z_i0[:, :, 0]))


def test_non_first_frames_follow_unpinned_formula():
    z_video, z_i0, eps, sigma_t = _fixed_inputs(seed=4)
    got = build_noisy_pinned_latents(z_video, z_i0, eps, sigma_t)
    b = z_video.shape[0]
    sigma_b = sigma_t.reshape((b, 1, 1, 1, 1))
    unpinned = (1.0 - sigma_b) * z_video + sigma_b * eps
    np.testing.assert_array_equal(np.asarray(got[:, :, 1:]), np.asarray(unpinned[:, :, 1:]))


# (c) Mask semantics: frame 0 contributes nothing to the loss.


def test_loss_invariant_to_frame0_perturbation():
    z_video, _, eps, _ = _fixed_inputs(seed=5)
    v_pred = jax.random.normal(jax.random.key(11), z_video.shape, dtype=jnp.float32)
    v_target = eps - z_video
    base = masked_velocity_mse(v_pred, v_target, _B)

    kp, kt = jax.random.split(jax.random.key(13))
    pred_pert = jax.random.normal(kp, (_B, _C, 1, _H, _W), dtype=jnp.float32) * 1000.0
    tgt_pert = jax.random.normal(kt, (_B, _C, 1, _H, _W), dtype=jnp.float32) * 1000.0
    v_pred_p = v_pred.at[:, :, :1].add(pred_pert)
    v_target_p = v_target.at[:, :, :1].add(tgt_pert)
    perturbed = masked_velocity_mse(v_pred_p, v_target_p, _B)

    # Frame 0 is masked out, so the loss must be bit-for-bit unchanged.
    assert float(base) == float(perturbed)


# (d) Normalization: explicit hand-computed n_valid.


def test_normalization_hand_computed():
    b, c, f, h, w = _B, _C, _F, _H, _W
    count_nonframe0 = c * (f - 1) * h * w  # unmasked elements per example
    n_valid = b * count_nonframe0
    v_target = jnp.zeros((b, c, f, h, w), dtype=jnp.float32)
    for fill in (1.0, 2.5):
        v_pred = jnp.full((b, c, f, h, w), fill, dtype=jnp.float32)
        # Every unmasked element contributes (fill - 0) ** 2; frame 0 is masked.
        sum_sq = fill**2 * b * count_nonframe0
        expected = sum_sq / n_valid  # == fill ** 2
        got = masked_velocity_mse(v_pred, v_target, b)
        np.testing.assert_allclose(np.asarray(got), expected, rtol=0, atol=1e-6)


# (F2) Shape contract: a broadcastable-but-mismatched v_pred must be rejected,
# never silently reduced against a mask built from the wrong tensor. (The old
# inline code built the mask from z_video == v_target; a one-frame v_pred vs a
# four-frame v_target returned 1.0 there but 0.0 with a v_pred-derived mask.)


def test_masked_velocity_mse_rejects_shape_mismatch():
    v_target = jnp.ones((_B, _C, _F, _H, _W), dtype=jnp.float32)
    v_pred = jnp.ones((_B, _C, 1, _H, _W), dtype=jnp.float32)  # broadcastable, malformed
    with pytest.raises(ValueError, match="shape"):
        masked_velocity_mse(v_pred, v_target, _B)


# (e) Dtype contract.


def test_helpers_return_float32():
    z_video, z_i0, eps, sigma_t = _fixed_inputs(seed=6)
    z_t = build_noisy_pinned_latents(z_video, z_i0, eps, sigma_t)
    assert z_t.dtype == jnp.float32

    v_pred = jax.random.normal(jax.random.key(17), z_video.shape, dtype=jnp.float32)
    v_target = eps - z_video
    loss = masked_velocity_mse(v_pred, v_target, _B)
    assert loss.dtype == jnp.float32
    assert loss.shape == ()


# (F3a) The float32 casts are real: bf16 inputs must reproduce the value the f32
# reference computes on the bf16-rounded (exactly f32-representable) numbers.
# Without the internal upcasts, bf16 arithmetic would round differently.


def test_helpers_compute_in_float32_for_bf16_inputs():
    z_video, z_i0, eps, sigma_t = _fixed_inputs(seed=8)
    zb, ib, eb, sb = (x.astype(jnp.bfloat16) for x in (z_video, z_i0, eps, sigma_t))
    got = build_noisy_pinned_latents(zb, ib, eb, sb)
    ref = _reference_z_t(
        zb.astype(jnp.float32), ib.astype(jnp.float32), eb.astype(jnp.float32), sb.astype(jnp.float32)
    )
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))

    v_pred = jax.random.normal(jax.random.key(19), z_video.shape, dtype=jnp.float32)
    pb, tb = v_pred.astype(jnp.bfloat16), (eps - z_video).astype(jnp.bfloat16)
    got_loss = masked_velocity_mse(pb, tb, _B)
    ref_loss = _reference_loss(pb.astype(jnp.float32), tb.astype(jnp.float32), _B)
    assert got_loss.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got_loss), np.asarray(ref_loss))


# (F3b) The explicit ``batch_size`` argument governs n_valid, not the tensor's
# leading dimension (the trainer passes the sliced batch size; keep it explicit).


def test_normalization_uses_explicit_batch_size_argument():
    b_data, batch_size_arg = _B, 2 * _B  # deliberately different
    count_nonframe0 = _C * (_F - 1) * _H * _W
    v_target = jnp.zeros((b_data, _C, _F, _H, _W), dtype=jnp.float32)
    v_pred = jnp.ones((b_data, _C, _F, _H, _W), dtype=jnp.float32)
    sum_sq = b_data * count_nonframe0  # each unmasked element contributes 1.0
    n_valid = count_nonframe0 * batch_size_arg  # the ARGUMENT, not shape[0]
    got = masked_velocity_mse(v_pred, v_target, batch_size_arg)
    assert float(got) == sum_sq / n_valid  # == 0.5, not 1.0


# (F1) Characterization of the REFACTORED trainer ``_denoising_loss`` itself:
# fixed rngs, tiny shapes, stub adapter/transformer modules merged through the
# real TrainState fields, and the production ``guide_scale=5.0`` CFG path (the
# stop-gradient uncond branch; stop_gradient changes gradients, never values).
# ``_reference_denoising_loss`` transcribes the complete pre-refactor body; the
# trainer must reproduce loss and every aux entry bit-for-bit.


class _StubTransformer(nnx.Module):
    """Deterministic stand-in for the frozen WAN transformer (uncond branch)."""

    def __init__(self):
        self.gain = nnx.Param(jnp.asarray(0.5, dtype=jnp.float32))

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, deterministic):
        t_mean = jnp.mean(timestep.astype(jnp.float32))
        ctx_mean = jnp.mean(encoder_hidden_states.astype(jnp.float32))
        return self.gain[...] * hidden_states.astype(jnp.float32) + 0.01 * t_mean + 0.001 * ctx_mean


class _StubAdapters(nnx.Module):
    """Stand-in adapter stack; its param flows through state.params and nnx.merge."""

    def __init__(self):
        self.bias = nnx.Param(jnp.asarray(0.25, dtype=jnp.float32))


def _stub_adapter_forward(
    transformer, adapters, *, hidden_states, timestep, encoder_hidden_states, actions, deterministic, rngs
):
    """Deterministic stand-in for ``wan_action_adapter_forward`` (cond branch)."""
    base = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        deterministic=True,
    )
    return base + adapters.bias[...] + 0.05 * jnp.mean(actions.astype(jnp.float32))


def _make_trainer_fixture(guide_scale):
    adapters = _StubAdapters()
    transformer = _StubTransformer()
    graphdef, params, rest = nnx.split(adapters, nnx.Param, ...)
    t_graphdef, t_params, t_rest = nnx.split(transformer, nnx.Param, ...)
    null_context = jax.random.normal(jax.random.key(41), (1, 4, 8), dtype=jnp.float32)
    state = trainer_mod.TrainState(
        step=0,
        apply_fn=None,
        params=params,
        tx=None,
        opt_state=None,
        graphdef=graphdef,
        rest_of_state=rest,
        transformer_graphdef=t_graphdef,
        transformer_params=t_params,
        transformer_rest=t_rest,
        null_context=null_context,
    )
    # Data carries 3 rows but global_batch_size_to_train_on is 2: the trainer
    # must slice. h=4, w=6 gives a 2x3 token grid for the per-token timestep.
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    data = {
        "z_i0": jax.random.normal(k1, (3, _C, 1, 4, 6), dtype=jnp.float32),
        "z_video": jax.random.normal(k2, (3, _C, _F, 4, 6), dtype=jnp.float32),
        "actions": jax.random.normal(k3, (3, 5, 7), dtype=jnp.float32),
    }
    config = SimpleNamespace(
        weights_dtype="float32",
        global_batch_size_to_train_on=2,
        side_adapter_sampling_steps=4,
        flow_shift=5.0,
        side_adapter_guide_scale=guide_scale,
        side_adapter_t_sampling="uniform",
        side_adapter_noise_mode="fresh",
        seed=0,
    )
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=config.flow_shift, sigma_min=0.0, sigma_max=1.0)
    return transformer, adapters, state, data, config, scheduler


def _reference_denoising_loss(transformer, adapters, null_context, data, rng, config, scheduler):
    """Verbatim transcription of the pre-refactor ``_denoising_loss`` body."""
    noise_rng, step_rng, _ = jax.random.split(rng, 3)
    bsz = config.global_batch_size_to_train_on
    z_i0_f32 = data["z_i0"][:bsz].astype(jnp.float32)
    z_video_f32 = data["z_video"][:bsz].astype(jnp.float32)
    actions = data["actions"][:bsz].astype(jnp.float32)  # weights_dtype == float32
    b, _, f_lat, h_lat, w_lat = z_video_f32.shape
    num_steps = int(config.side_adapter_sampling_steps)
    sigmas = build_rollout_sigmas(num_steps, config.flow_shift, scheduler.config.sigma_min, scheduler.config.sigma_max)
    t_idx = jax.random.randint(step_rng, (b,), 0, num_steps)  # "uniform" t sampling
    sigma_t = sigmas[t_idx]
    step_t = sigma_t * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    null_context = jnp.broadcast_to(
        null_context.astype(jnp.float32), (b, null_context.shape[1], null_context.shape[2])
    )
    eps = jax.random.normal(noise_rng, z_video_f32.shape, dtype=jnp.float32)  # "fresh" noise
    sigma_b = sigma_t.reshape((b, 1, 1, 1, 1))
    z_t = (1.0 - sigma_b) * z_video_f32 + sigma_b * eps
    z_t = z_t.at[:, :, :1].set(z_i0_f32[:, :, :1])  # frame-0 pin
    v_cond = _stub_adapter_forward(
        transformer,
        adapters,
        hidden_states=z_t,
        timestep=timestep_2d,
        encoder_hidden_states=null_context,
        actions=actions,
        deterministic=False,
        rngs=None,
    )
    if abs(config.side_adapter_guide_scale - 1.0) > 1e-6:
        v_uncond = transformer(
            hidden_states=z_t, timestep=timestep_2d, encoder_hidden_states=null_context, deterministic=True
        )
        v_pred = v_uncond + config.side_adapter_guide_scale * (v_cond - v_uncond)
    else:
        v_pred = v_cond
    v_target = eps - z_video_f32
    mask = jnp.ones((1, z_video_f32.shape[1], f_lat, h_lat, w_lat), dtype=jnp.float32)
    mask = mask.at[:, :, :1, :, :].set(0.0)
    diff = (v_pred.astype(jnp.float32) - v_target.astype(jnp.float32)) * mask
    n_valid = jnp.maximum(jnp.sum(mask) * b, 1.0)
    loss = jnp.sum(diff**2) / n_valid
    aux = {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(sigma_t.astype(jnp.float32)),
        "timestep_mean": jnp.mean(step_t.astype(jnp.float32)),
        "v_pred_l2": jnp.linalg.norm(v_pred.astype(jnp.float32)),
        "v_target_l2": jnp.linalg.norm(v_target.astype(jnp.float32)),
        "z_noisy_std": jnp.std(z_t),
        "z_target_std": jnp.std(z_video_f32),
        "z_init_anchor_mse": jnp.mean((z_t[:, :, :1] - z_i0_f32[:, :, :1]) ** 2),
    }
    return loss, aux


@pytest.mark.parametrize("guide_scale", [1.0, 5.0])
def test_trainer_denoising_loss_matches_pre_refactor_reference(monkeypatch, guide_scale):
    transformer, adapters, state, data, config, scheduler = _make_trainer_fixture(guide_scale)
    monkeypatch.setattr(trainer_mod, "wan_action_adapter_forward", _stub_adapter_forward)
    rng = jax.random.key(123)
    loss, aux = trainer_mod._denoising_loss(state.params, state, data, rng, config, scheduler)
    ref_loss, ref_aux = _reference_denoising_loss(
        transformer, adapters, state.null_context, data, rng, config, scheduler
    )
    assert set(aux) == set(ref_aux)
    np.testing.assert_array_equal(np.asarray(loss), np.asarray(ref_loss))
    for key in ref_aux:
        np.testing.assert_array_equal(np.asarray(aux[key]), np.asarray(ref_aux[key]), err_msg=key)
    assert float(aux["velocity_mse"]) == float(loss)
    # The frame-0 pin held inside the production function, not just the reference.
    assert float(aux["z_init_anchor_mse"]) == 0.0


def test_trainer_denoising_loss_cfg_branch_is_exercised(monkeypatch):
    monkeypatch.setattr(trainer_mod, "wan_action_adapter_forward", _stub_adapter_forward)
    rng = jax.random.key(123)
    losses = {}
    for guide_scale in (1.0, 5.0):
        _, _, state, data, config, scheduler = _make_trainer_fixture(guide_scale)
        loss, _ = trainer_mod._denoising_loss(state.params, state, data, rng, config, scheduler)
        losses[guide_scale] = float(loss)
    # The stub cond/uncond branches differ, so an exercised CFG combine must move the loss.
    assert losses[1.0] != losses[5.0]


def test_trainer_denoising_loss_calls_shared_helpers(monkeypatch):
    calls = {"build_noisy_pinned_latents": 0, "masked_velocity_mse": 0}

    def counting_z_t(*args, **kwargs):
        calls["build_noisy_pinned_latents"] += 1
        return build_noisy_pinned_latents(*args, **kwargs)

    def counting_mse(*args, **kwargs):
        calls["masked_velocity_mse"] += 1
        return masked_velocity_mse(*args, **kwargs)

    monkeypatch.setattr(trainer_mod, "wan_action_adapter_forward", _stub_adapter_forward)
    monkeypatch.setattr(trainer_mod, "build_noisy_pinned_latents", counting_z_t)
    monkeypatch.setattr(trainer_mod, "masked_velocity_mse", counting_mse)
    _, _, state, data, config, scheduler = _make_trainer_fixture(5.0)
    trainer_mod._denoising_loss(state.params, state, data, jax.random.key(123), config, scheduler)
    # Parity with the full-FT trainer is BY SHARED CODE: each helper used exactly once.
    assert calls == {"build_noisy_pinned_latents": 1, "masked_velocity_mse": 1}
