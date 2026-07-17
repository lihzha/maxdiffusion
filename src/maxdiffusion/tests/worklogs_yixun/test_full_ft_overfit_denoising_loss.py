"""CPU-only fixed-RNG integration tests for the full-FT overfit denoising loss.

exp_01 round 2: ``wan_ti2v_full_ft_trainer._denoising_loss`` is the side-adapter
objective *minus the adapter* -- exactly one plain WAN transformer forward, null
text context, the frame-0 pin, fresh per-example noise, target ``eps - z_video``,
and NO actions anywhere. These tests drive that loss on a tiny call-recording
stub transformer (an ``nnx.Module`` with a real ``Param`` so ``_train_step``'s
optimizer step is genuine) through the actual ``FullFTTrainState``, a real
``FlaxFlowMatchScheduler``, and fixed rngs, asserting every contract against
independently transcribed reference equations. Data batch is 3 while
``global_batch_size_to_train_on`` is 2, so the slice is pinned. CPU-only: no
model weights, no TPU. The darwin grain import stub lives in ``conftest.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

import maxdiffusion.trainers.wan_ti2v_full_ft_trainer as full_ft
from maxdiffusion.models.wan.side_adapter_wan import (
    build_rollout_sigmas,
    masked_velocity_mse,
    _build_per_token_timestep,
    _dtype,
)
from maxdiffusion.schedulers import FlaxFlowMatchScheduler

# Tiny fixed shapes: data batch 3 (sliced to bsz 2), channels C, latent frames F,
# height H, width W.  H/W stay small; _build_per_token_timestep only uses //2.
_DATA_B, _BSZ, _C, _F, _H, _W = 3, 2, 3, 4, 5, 6
_STUB_GAIN = 0.5

# The recording stub appends every call's kwargs here. Module-level so it survives
# nnx.split/merge -- a method's global namespace is untouched by nnx graph ops.
_STUB_CALLS: list[dict] = []


@pytest.fixture(autouse=True)
def _clear_stub_calls():
    _STUB_CALLS.clear()
    yield
    _STUB_CALLS.clear()


def _stub_velocity(hidden_states, timestep, encoder_hidden_states, gain):
    """The stub transformer's deterministic velocity map (shared with the ref)."""
    t_mean = jnp.mean(timestep.astype(jnp.float32))
    ctx_mean = jnp.mean(encoder_hidden_states.astype(jnp.float32))
    return gain * hidden_states.astype(jnp.float32) + 0.01 * t_mean + 0.001 * ctx_mean


class _RecordingStubTransformer(nnx.Module):
    """Stand-in for the (now trainable) WAN transformer.

    A real ``Param`` (``gain``) so gradients/optimizer steps are genuine, and it
    records every call's kwargs for strict inspection (exactly one call, null
    context, no ``actions``). Returns a velocity-shaped array (same shape as
    ``hidden_states``), matching ``masked_velocity_mse``'s shape contract.
    """

    def __init__(self):
        self.gain = nnx.Param(jnp.asarray(_STUB_GAIN, dtype=jnp.float32))

    def __call__(self, **kwargs):
        _STUB_CALLS.append(kwargs)
        return _stub_velocity(
            kwargs["hidden_states"], kwargs["timestep"], kwargs["encoder_hidden_states"], self.gain[...]
        )


def _make_fixture(*, weights_dtype="float32", activations_dtype="float32", data_dtype=jnp.float32, tx=None):
    transformer = _RecordingStubTransformer()
    graphdef, params, rest = nnx.split(transformer, nnx.Param, ...)
    null_context = jax.random.normal(jax.random.key(41), (1, 4, 8), dtype=jnp.float32)
    if tx is None:
        state = full_ft.FullFTTrainState(
            step=0,
            apply_fn=None,
            params=params,
            tx=None,
            opt_state=None,
            graphdef=graphdef,
            rest_of_state=rest,
            null_context=null_context,
        )
    else:
        state = full_ft.FullFTTrainState.create(
            apply_fn=graphdef.apply,
            params=params,
            tx=tx,
            graphdef=graphdef,
            rest_of_state=rest,
            null_context=null_context,
        )
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    data = {
        "z_i0": jax.random.normal(k1, (_DATA_B, _C, 1, _H, _W), dtype=jnp.float32).astype(data_dtype),
        "z_video": jax.random.normal(k2, (_DATA_B, _C, _F, _H, _W), dtype=jnp.float32).astype(data_dtype),
        # actions are present in the batch but MUST be ignored by the full-FT loss.
        "actions": jax.random.normal(k3, (_DATA_B, 32, 7), dtype=jnp.float32),
    }
    config = SimpleNamespace(
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
        global_batch_size_to_train_on=_BSZ,
        side_adapter_sampling_steps=4,
        flow_shift=5.0,
        side_adapter_t_sampling="uniform",
        side_adapter_noise_mode="fresh",
        seed=0,
    )
    scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32, shift=config.flow_shift, sigma_min=0.0, sigma_max=1.0)
    return transformer, state, data, config, scheduler


def _reference(state, data, rng, config, scheduler):
    """Transcribe the full-FT ``_denoising_loss`` math (fresh noise, uniform t, no CFG)."""
    noise_rng, step_rng, _dropout_rng = jax.random.split(rng, 3)
    weights_dtype = _dtype(config.weights_dtype)
    activations_dtype = _dtype(config.activations_dtype)
    bsz = config.global_batch_size_to_train_on
    z_i0_f32 = data["z_i0"][:bsz].astype(jnp.float32)
    z_video_f32 = data["z_video"][:bsz].astype(jnp.float32)
    b, _, f_lat, h_lat, w_lat = z_video_f32.shape
    num_steps = int(config.side_adapter_sampling_steps)
    sigmas = build_rollout_sigmas(num_steps, config.flow_shift, scheduler.config.sigma_min, scheduler.config.sigma_max)
    t_idx = jax.random.randint(step_rng, (b,), 0, num_steps)  # side_adapter_t_sampling == "uniform"
    sigma_t = sigmas[t_idx]
    step_t = sigma_t * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    eps = jax.random.normal(noise_rng, z_video_f32.shape, dtype=jnp.float32)  # side_adapter_noise_mode == "fresh"
    sigma_b = sigma_t.reshape((b, 1, 1, 1, 1))
    z_t_f32 = (1.0 - sigma_b) * z_video_f32 + sigma_b * eps
    z_t_f32 = z_t_f32.at[:, :, :1].set(z_i0_f32[:, :, :1])  # frame-0 pin
    null_ctx = jnp.broadcast_to(
        state.null_context.astype(activations_dtype),
        (b, state.null_context.shape[1], state.null_context.shape[2]),
    )
    v_target = eps - z_video_f32
    return SimpleNamespace(
        b=b,
        weights_dtype=weights_dtype,
        activations_dtype=activations_dtype,
        num_steps=num_steps,
        sigmas=sigmas,
        t_idx=t_idx,
        sigma_t=sigma_t,
        step_t=step_t,
        timestep_2d=timestep_2d,
        eps=eps,
        z_i0_f32=z_i0_f32,
        z_video_f32=z_video_f32,
        z_t_f32=z_t_f32,
        null_ctx=null_ctx,
        v_target=v_target,
    )


_RNG = jax.random.key(123)


# (1) Fresh per-example noise: eps rows differ, and that fresh eps is the one the
# loss actually consumes (aux v_target_l2 == ||fresh_eps - z_video||).


def test_fresh_noise_is_per_example_and_drives_the_loss():
    _, state, data, config, scheduler = _make_fixture()
    ref = _reference(state, data, _RNG, config, scheduler)
    # Fresh (not fixed/broadcast) noise: the two batch rows must be distinct.
    assert not np.allclose(np.asarray(ref.eps[0]), np.asarray(ref.eps[1]))
    _, aux = full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    v_target_l2 = jnp.linalg.norm((ref.eps - ref.z_video_f32).astype(jnp.float32))
    np.testing.assert_array_equal(np.asarray(aux["v_target_l2"]), np.asarray(v_target_l2))


# (2) t/sigma selection matches build_rollout_sigmas indexing: the sigma grid is
# transcribed inline (linspace + shift warp + terminal 0, independent of the
# helper), the sigma used equals sigmas[t_idx], both surface through aux, and
# (F1) the transformer's recorded per-token timestep equals the reference
# timestep_2d bitwise WITH its TI2V structure asserted independently of
# _build_per_token_timestep: the frame-0 history-token block is 0 ("clean"
# frame) and every future-token position carries its example's own step_t.


def test_sigma_and_timestep_match_build_rollout_sigmas_indexing():
    _, state, data, config, scheduler = _make_fixture()
    ref = _reference(state, data, _RNG, config, scheduler)
    # Transcription of the sigma computation itself, not a call into the helper.
    lin = jnp.linspace(scheduler.config.sigma_max, scheduler.config.sigma_min, ref.num_steps + 1, dtype=jnp.float32)
    lin = lin[:-1]
    shifted = config.flow_shift * lin / (1.0 + (config.flow_shift - 1.0) * lin)
    grid = jnp.concatenate([shifted, jnp.zeros((1,), dtype=jnp.float32)], axis=0)
    np.testing.assert_array_equal(np.asarray(ref.sigmas), np.asarray(grid))
    # sigma_t IS the grid indexed by the uniform t sample -- pin the indexing.
    np.testing.assert_array_equal(np.asarray(ref.sigma_t), np.asarray(grid[ref.t_idx]))
    _, aux = full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    np.testing.assert_array_equal(np.asarray(aux["sigma_mean"]), np.asarray(jnp.mean(ref.sigma_t)))
    np.testing.assert_array_equal(np.asarray(aux["timestep_mean"]), np.asarray(jnp.mean(ref.step_t)))
    # (F1) Bitwise: the per-token timestep the transformer actually received.
    timestep_kwarg = _STUB_CALLS[0]["timestep"]
    np.testing.assert_array_equal(np.asarray(timestep_kwarg), np.asarray(ref.timestep_2d))
    # (F1) Structure, transcribed (not via _build_per_token_timestep): n_hist=1
    # means the first frame's token block gets timestep 0, the rest get step_t.
    tokens_per_frame = (_H // 2) * (_W // 2)
    seq_len = _F * tokens_per_frame
    assert timestep_kwarg.shape == (ref.b, seq_len)
    np.testing.assert_array_equal(
        np.asarray(timestep_kwarg[:, :tokens_per_frame]), np.zeros((ref.b, tokens_per_frame), dtype=np.float32)
    )
    expected_future = np.broadcast_to(np.asarray(ref.step_t)[:, None], (ref.b, seq_len - tokens_per_frame))
    np.testing.assert_array_equal(np.asarray(timestep_kwarg[:, tokens_per_frame:]), expected_future)
    # Non-vacuity: the examples carry DIFFERENT step_t, so "own step_t" is real.
    assert float(ref.step_t[0]) != float(ref.step_t[1])


# (3) Target is exactly eps - z_video: recover v_pred from the recorded stub inputs
# and confirm the returned loss == masked_velocity_mse(v_pred, eps - z_video, b).


def test_target_is_exactly_eps_minus_z_video():
    _, state, data, config, scheduler = _make_fixture()
    ref = _reference(state, data, _RNG, config, scheduler)
    loss, aux = full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    assert len(_STUB_CALLS) == 1
    call = _STUB_CALLS[0]
    v_pred = _stub_velocity(call["hidden_states"], call["timestep"], call["encoder_hidden_states"], _STUB_GAIN)
    loss_ref = masked_velocity_mse(v_pred, ref.v_target, ref.b)
    np.testing.assert_array_equal(np.asarray(loss), np.asarray(loss_ref))
    # aux surfaces the same velocity target norm, and velocity_mse mirrors the loss.
    np.testing.assert_array_equal(np.asarray(aux["v_target_l2"]), np.asarray(jnp.linalg.norm(ref.v_target)))
    assert float(aux["velocity_mse"]) == float(loss)


# (4) Frame-0 pin present: the hidden_states the stub receives has frame 0 bitwise
# equal to z_i0 (after the weights_dtype cast), and aux z_init_anchor_mse == 0.


def test_frame0_pin_reaches_transformer_and_anchor_mse_is_zero():
    _, state, data, config, scheduler = _make_fixture()
    ref = _reference(state, data, _RNG, config, scheduler)
    _, aux = full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    hidden_states = _STUB_CALLS[0]["hidden_states"]
    expected_frame0 = ref.z_i0_f32[:, :, 0].astype(ref.weights_dtype)
    np.testing.assert_array_equal(np.asarray(hidden_states[:, :, 0]), np.asarray(expected_frame0))
    assert float(aux["z_init_anchor_mse"]) == 0.0


# (5) Null context: encoder_hidden_states received == state.null_context broadcast
# to the sliced batch (in activations dtype), bitwise.


def test_encoder_hidden_states_is_null_context_broadcast():
    _, state, data, config, scheduler = _make_fixture()
    ref = _reference(state, data, _RNG, config, scheduler)
    full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    encoder_hidden_states = _STUB_CALLS[0]["encoder_hidden_states"]
    np.testing.assert_array_equal(np.asarray(encoder_hidden_states), np.asarray(ref.null_ctx))


# (5b/F2) The null-context cast follows activations_dtype, NOT weights_dtype (the
# accepted round-2 deviation). Split the dtypes so a revert to the reference's
# weights_dtype cast would produce float32 context here and fail both asserts.


def test_null_context_cast_follows_activations_dtype_not_weights_dtype():
    _, state, data, config, scheduler = _make_fixture(weights_dtype="float32", activations_dtype="bfloat16")
    assert state.null_context.dtype == jnp.float32  # stored f32; the loss casts it
    full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    encoder_hidden_states = _STUB_CALLS[0]["encoder_hidden_states"]
    assert encoder_hidden_states.dtype == jnp.bfloat16  # activations_dtype, not f32
    expected = jnp.broadcast_to(
        state.null_context.astype(jnp.bfloat16),
        (_BSZ, state.null_context.shape[1], state.null_context.shape[2]),
    )
    np.testing.assert_array_equal(np.asarray(encoder_hidden_states), np.asarray(expected))


# (6) Exactly ONE plain-transformer call, with no actions and nothing
# adapter-related in its kwargs (the strict integration contract).


def test_exactly_one_plain_transformer_call_without_actions():
    _, state, data, config, scheduler = _make_fixture()
    full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    assert len(_STUB_CALLS) == 1
    call = _STUB_CALLS[0]
    assert set(call.keys()) == {"hidden_states", "timestep", "encoder_hidden_states", "deterministic", "rngs"}
    assert "actions" not in call
    assert not any("adapter" in k or "action" in k for k in call)
    assert call["deterministic"] is False


# (7) Sliced batch size: data batch is 3, bsz is 2. Tensors are sliced before use,
# and masked_velocity_mse is invoked with the sliced batch size (2), not 3.


def test_tensors_sliced_and_masked_mse_gets_sliced_batch_size(monkeypatch):
    seen = {}
    real_masked_velocity_mse = full_ft.masked_velocity_mse

    def spy(v_pred, v_target, batch_size):
        seen["batch_size"] = batch_size
        seen["v_pred_leading"] = v_pred.shape[0]
        seen["v_target_leading"] = v_target.shape[0]
        return real_masked_velocity_mse(v_pred, v_target, batch_size)

    monkeypatch.setattr(full_ft, "masked_velocity_mse", spy)
    _, state, data, config, scheduler = _make_fixture()
    assert data["z_video"].shape[0] == _DATA_B  # unsliced batch is 3
    full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    assert seen["batch_size"] == _BSZ  # the explicit sliced batch size (2)
    assert seen["v_pred_leading"] == _BSZ
    assert seen["v_target_leading"] == _BSZ
    assert _STUB_CALLS[0]["hidden_states"].shape[0] == _BSZ  # sliced before the forward


# (8) One _train_step changes the transformer Param and returns finite metrics.


def test_train_step_updates_params_and_reports_finite_metrics():
    _, state, data, config, scheduler = _make_fixture(tx=optax.sgd(0.1))
    gain_before = np.asarray(jax.tree_util.tree_leaves(state.params)[0])
    new_state, metrics, _ = full_ft._train_step(state, data, _RNG, scheduler, config)
    gain_after = np.asarray(jax.tree_util.tree_leaves(new_state.params)[0])
    assert not np.array_equal(gain_before, gain_after)  # optimizer moved the param
    loss = float(metrics["scalar"]["learning/loss"])
    grad_norm = float(metrics["scalar"]["learning/grad_norm"])
    max_abs_grad = float(metrics["scalar"]["learning/max_abs_grad"])
    assert np.isfinite(loss) and np.isfinite(grad_norm) and np.isfinite(max_abs_grad)
    assert grad_norm > 0.0  # a real gradient flowed to the transformer Param


# (9) f32 discipline: even with a bf16 weights_dtype config, the interpolation/pin
# math runs in float32; the z_t handed to the transformer is bf16(f32_reference),
# NOT the value bf16 arithmetic on bf16-cast inputs would produce.


def test_interpolation_stays_float32_under_bf16_weights_dtype():
    _, state, data, config, scheduler = _make_fixture(
        weights_dtype="bfloat16", activations_dtype="bfloat16", data_dtype=jnp.bfloat16
    )
    ref = _reference(state, data, _RNG, config, scheduler)
    full_ft._denoising_loss(state.params, state, data, _RNG, config, scheduler)
    hidden_states = _STUB_CALLS[0]["hidden_states"]
    assert hidden_states.dtype == jnp.bfloat16
    expected = ref.z_t_f32.astype(jnp.bfloat16)  # f32 math, single cast at the boundary
    np.testing.assert_array_equal(np.asarray(hidden_states), np.asarray(expected))

    # Guard against a vacuous test: bf16 arithmetic on bf16-cast inputs must give a
    # different (non-frame-0) result, so the equality above genuinely proves f32.
    zb = data["z_video"][: ref.b].astype(jnp.bfloat16)
    eb = ref.eps.astype(jnp.bfloat16)
    sb = ref.sigma_t.astype(jnp.bfloat16).reshape((ref.b, 1, 1, 1, 1))
    bf16_math = ((1.0 - sb) * zb + sb * eb)[:, :, 1:]
    assert not np.array_equal(np.asarray(expected[:, :, 1:]), np.asarray(bf16_math))
