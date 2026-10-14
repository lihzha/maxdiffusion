"""Tests for the per-view temporal-frequency loss weighting (``temporal_freq_loss``).

Covers the Q matrices against their FFT definition, the stacked-view application
in both models' layouts, the config reader, and one real train step per model:
unit weights must reproduce the plain loss and other weights must change it.

Run: JAX_PLATFORMS=cpu python -m unittest maxdiffusion.tests.temporal_freq_loss_test
"""

import dataclasses
import functools
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from maxdiffusion import temporal_freq_loss as tfl

_LOW, _HIGH = (2.0, 0.5, 1.0), (0.5, 2.0, 1.0)


def _padded(e, n_lead):
  u = np.concatenate([np.zeros(n_lead), e])
  return np.concatenate([u, u[::-1]])


class BandWeightMatrixTest(unittest.TestCase):

  def test_unit_weights_are_identity(self):
    for n_hist, n_fut, zero_history in [(7, 5, True), (7, 5, False), (0, 5, True), (3, 1, True)]:
      q = tfl.band_weight_matrices(n_hist, n_fut, [1, 1, 1], [1, 1, 1], 0.2, zero_history)
      np.testing.assert_allclose(q, np.broadcast_to(np.eye(n_fut), q.shape), atol=1e-6)

  def test_matches_fft_definition(self):
    rng = np.random.default_rng(0)
    low, high, sigma = [2.0, 0.5, 1.0], [0.3, 1.5, 0.0], 0.15
    for zero_history in (True, False):
      q = tfl.band_weight_matrices(7, 5, low, high, sigma, zero_history)
      e = rng.normal(size=5)
      s = _padded(e, 7 if zero_history else 0)
      n = len(s)
      w_low = np.exp(-np.fft.fftfreq(n) ** 2 / (2 * sigma ** 2))
      power = np.abs(np.fft.fft(s)) ** 2
      for v in range(3):
        w = high[v] + (low[v] - high[v]) * w_low
        np.testing.assert_allclose(e @ q[v] @ e, (w * power).sum() / n / 2, rtol=1e-5)

  def test_symmetric_psd(self):
    q = tfl.band_weight_matrices(7, 5, [2, 0, 1], [0, 3, 1], 0.2).astype(np.float64)
    np.testing.assert_allclose(q, np.swapaxes(q, 1, 2), atol=1e-6)
    self.assertGreaterEqual(np.linalg.eigvalsh(q).min(), -1e-5)

  def test_bands_separate_slow_and_fast_error(self):
    const, alt = np.ones(5), np.array([1.0, -1, 1, -1, 1])
    low_only = tfl.band_weight_matrices(7, 5, [1], [0], 0.2, zero_history=False)[0]
    high_only = tfl.band_weight_matrices(7, 5, [0], [1], 0.2, zero_history=False)[0]
    # The mirrored future alone: a constant error is pure DC.
    self.assertAlmostEqual(float(const @ high_only @ const), 0.0, places=5)
    self.assertGreater(alt @ high_only @ alt, 2 * (alt @ low_only @ alt))
    # Behind zero-error history the same constant error is a step at the seam,
    # which the high band sees.
    high_seam = tfl.band_weight_matrices(7, 5, [0], [1], 0.2, zero_history=True)[0]
    self.assertGreater(const @ high_seam @ const, 0.1)

  def test_rejects_bad_settings(self):
    with self.assertRaises(ValueError):
      tfl.band_weight_matrices(7, 5, [1, 1], [1, 1, 1], 0.2)
    with self.assertRaises(ValueError):
      tfl.band_weight_matrices(7, 5, [-1.0], [1.0], 0.2)
    with self.assertRaises(ValueError):
      tfl.band_weight_matrices(7, 5, [1.0], [1.0], 0.0)


class ApplyTest(unittest.TestCase):

  def _check_layout(self, shape, frame_axis, view_axis):
    rng = np.random.default_rng(1)
    q = tfl.band_weight_matrices(2, shape[frame_axis], _LOW, _HIGH, 0.2)
    diff = rng.normal(size=shape).astype(np.float32)
    out = np.asarray(tfl.apply_band_weights(jnp.asarray(diff), q, frame_axis, view_axis))
    x = np.moveaxis(diff, (frame_axis, view_axis), (0, 1))
    hv = x.shape[1] // 3
    ref = np.empty_like(x)
    for v in range(3):
      ref[:, v * hv:(v + 1) * hv] = np.tensordot(q[v], x[:, v * hv:(v + 1) * hv], axes=(1, 0))
    np.testing.assert_allclose(out, np.moveaxis(ref, (0, 1), (frame_axis, view_axis)), rtol=1e-5, atol=1e-5)

  def test_wan_layout(self):  # (B, C, F, H, W)
    self._check_layout((2, 3, 5, 6, 4), frame_axis=2, view_axis=3)

  def test_ctrl_world_layout(self):  # (B, F, C, H, W)
    self._check_layout((2, 5, 4, 6, 4), frame_axis=1, view_axis=3)

  def test_identity_is_plain_square_and_gradient_matches(self):
    diff = jax.random.normal(jax.random.key(0), (2, 3, 5, 6, 4))
    eye = tfl.band_weight_matrices(2, 5, [1, 1, 1], [1, 1, 1], 0.2)
    np.testing.assert_allclose(diff * tfl.apply_band_weights(diff, eye, 2, 3), diff ** 2, rtol=1e-5, atol=1e-6)
    q = tfl.band_weight_matrices(2, 5, _LOW, _HIGH, 0.2)
    grad = jax.grad(lambda d: jnp.sum(d * tfl.apply_band_weights(d, q, 2, 3)))(diff)
    np.testing.assert_allclose(grad, 2 * tfl.apply_band_weights(diff, q, 2, 3), rtol=1e-4, atol=1e-5)

  def test_rejects_indivisible_view_axis(self):
    q = tfl.band_weight_matrices(2, 5, [1, 1, 1], [1, 1, 1], 0.2)
    with self.assertRaises(ValueError):
      tfl.apply_band_weights(jnp.zeros((1, 1, 5, 7, 2)), q, 2, 3)


class SettingsTest(unittest.TestCase):

  def test_disabled_and_defaults(self):
    self.assertIsNone(tfl.settings_from_config(types.SimpleNamespace(), 3))
    self.assertIsNone(tfl.settings_from_config(types.SimpleNamespace(freq_loss_weighting=False), 3))
    s = tfl.settings_from_config(types.SimpleNamespace(freq_loss_weighting=True), 3)
    self.assertEqual(s, tfl.FreqLossSettings((1.0,) * 3, (1.0,) * 3, 0.2, True))

  def test_wrong_view_count(self):
    cfg = types.SimpleNamespace(freq_loss_weighting=True, freq_loss_low_weights=[1, 2])
    with self.assertRaises(ValueError):
      tfl.settings_from_config(cfg, 3)


class TrainStepTest(unittest.TestCase):
  """Unit weights reproduce each model's plain loss; other weights change it."""

  def test_wan_train_step(self):
    from maxdiffusion.schedulers import FlaxFlowMatchScheduler
    from maxdiffusion.tests import cam_action_test as cat
    from maxdiffusion.trainers.wan_ctrl_world_trainer import _train_step

    mesh = cat._mesh()
    stats = (-np.ones((3, 3), np.float32), np.ones((3, 3), np.float32))
    sched = FlaxFlowMatchScheduler(dtype=jnp.float32)
    sched_state = sched.set_timesteps(sched.create_state(), num_inference_steps=1000, training=True)
    helper = cat.WanTest()
    config, _, state = helper._tiny_setup("cam_action_adaln", mesh)
    batch = helper._tiny_batch(np.random.default_rng(5))

    def loss(**freq):
      cfg = types.SimpleNamespace(**{**vars(config), **freq})
      with mesh, cat.nn_partitioning.axis_rules(cat._RULES):
        step = jax.jit(functools.partial(_train_step, scheduler=sched, config=cfg, cam_stats=stats))
        _, _, metrics, _ = step(state, batch, jax.random.key(0), sched_state)
      return float(metrics["scalar"]["learning/loss"])

    plain = loss()
    unit = loss(freq_loss_weighting=True)
    weighted = loss(freq_loss_weighting=True, freq_loss_low_weights=list(_LOW), freq_loss_high_weights=list(_HIGH))
    self.assertTrue(np.isfinite(weighted))
    np.testing.assert_allclose(unit, plain, rtol=1e-5)
    self.assertGreater(abs(weighted - plain), 1e-3 * abs(plain))

  def test_ctrl_world_train_step(self):
    from maxdiffusion.models.svd.camera_action_encoder_flax import build_cam_action_modules
    from maxdiffusion.models.svd.ctrl_world_flax import CtrlWorldTrainConfig, action_world_train_step
    from maxdiffusion.models.svd.video_unet_flax import FlaxVideoUNet
    from maxdiffusion.tests import cam_action_test as cat

    b, n_hist, n_fut, h, w = 1, 2, 3, 24, 8
    t = n_hist + n_fut
    unet = FlaxVideoUNet(
        block_out_channels=(32, 32, 32, 32), num_attention_heads=(1, 1, 1, 1),
        cross_attention_dim=32, layers_per_block=1,
    )
    mesh = cat._mesh()
    with mesh:
      unet_params = unet.init(
          {"params": jax.random.key(0), "dropout": jax.random.key(1)},
          jnp.zeros((b * t, 8, h, w)), jnp.ones((b * t,)), jnp.zeros((b * t, 1, 32)),
          {"adm_vector": jnp.zeros((b, 768))}, None, num_frames=t,
      )["params"]
    rng = np.random.default_rng(6)
    cart, poses, _ = cat._episode_geometry(rng, t)
    batch = {
        "latent": jnp.asarray(rng.normal(size=(b, t, 4, h, w)), jnp.float32),
        "action": jnp.asarray(rng.uniform(-1, 1, (b, t, 7)), jnp.float32),
        "cam_pose": jnp.asarray(np.stack([p.reshape(t, 12) for p in poses], axis=1)[None]),
        "ee_cartesian": jnp.asarray(cart[None]),
    }
    mode = "cam_action_adaln"
    extras = build_cam_action_modules(
        mode, action_dim=cat.cfa.CAM_ACTION_DIM, hidden_size=32, num_views=3,
        time_embed_dim=128, model_channels=32, seed=0, dtype=jnp.float32, weights_dtype=jnp.float32,
    )
    params = {"unet": unet_params, **{k: p for k, (_m, p) in extras.items()}}
    apply_fns = {"unet": unet.apply, **{k: m.apply for k, (m, _p) in extras.items()}}
    cfg = CtrlWorldTrainConfig(
        num_history=n_hist, num_frames=n_fut, hidden_size=32, text_embed_dim=None,
        action_cond_mode=mode, time_embed_dim=128, model_channels=32,
        cam_action_xyz_lo=-np.ones((3, 3), np.float32), cam_action_xyz_hi=np.ones((3, 3), np.float32),
        use_task_instructions=False, cfg_drop_prob=0.0,
    )

    def loss(freq, train=True):
      with mesh:
        return float(action_world_train_step(
            jax.random.key(3), params, apply_fns, batch, dataclasses.replace(cfg, freq_loss=freq),
            0.18215, train=train,
        ))

    unit = tfl.FreqLossSettings((1.0,) * 3, (1.0,) * 3, 0.2, True)
    weighted = tfl.FreqLossSettings(_LOW, _HIGH, 0.2, True)
    plain = loss(None)
    np.testing.assert_allclose(loss(unit), plain, rtol=1e-5)
    self.assertGreater(abs(loss(weighted) - plain), 1e-3 * abs(plain))
    # Eval keeps the plain MSE.
    np.testing.assert_allclose(loss(weighted, train=False), loss(None, train=False), rtol=1e-6)


if __name__ == "__main__":
  absltest.main()
