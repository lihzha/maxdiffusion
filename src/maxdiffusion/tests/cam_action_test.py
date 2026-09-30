"""Tests for the camera-frame action conditioning (``cam_action*`` modes).

Covers the geometry, both datasets' slot alignment, view/camera locking in both
models, and one real train step per mode on tiny WAN and SVD models.

Run: JAX_PLATFORMS=cpu python -m unittest maxdiffusion.tests.cam_action_test
"""

import functools
import json
import os
import tempfile
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl.testing import absltest
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh

from maxdiffusion.input_pipeline.robot import camera_frame_actions as cfa

# The datasets hide accelerators from TF, which is only legal before TF's first
# op; the synthetic-record helpers below run TF ops before any dataset exists.
tf.config.set_visible_devices([], "GPU")

_RULES = [
    ["batch", ["data", "fsdp"]],
    ["activation_batch", ["data", "fsdp"]],
    ["activation_self_attn_heads", ["context", "tensor"]],
    ["activation_cross_attn_q_length", ["context", "tensor"]],
    ["activation_length", "context"],
    ["activation_heads", "tensor"],
    ["mlp", "tensor"],
    ["embed", ["context", "fsdp"]],
    ["heads", "tensor"],
    ["norm", "tensor"],
    ["conv_batch", ["data", "context", "fsdp"]],
    ["out_channels", "tensor"],
    ["conv_out", "context"],
]


def _mesh():
  return Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1, 1), ("data", "fsdp", "context", "tensor"))


def _rand_pose(rng):
  q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
  q *= np.sign(np.linalg.det(q))
  pose = np.eye(4)
  pose[:3, :3] = q
  pose[:3, 3] = rng.normal(size=3)
  return pose


def _episode_geometry(rng, t):
  """``(cartesian (t, 6), [P_wrist, P_ext1, P_ext2] each (t, 3, 4), extrinsics)``."""
  cart = np.concatenate(
      [rng.uniform(0.3, 0.7, (t, 3)), rng.uniform(-np.pi, np.pi, (t, 3))], axis=1
  ).astype(np.float32)
  hand_eye = _rand_pose(rng)
  e_wrist = np.linalg.inv(cfa.cartesian_to_pose_np(cart) @ hand_eye)
  e_ext = [_rand_pose(rng), _rand_pose(rng)]
  poses = [cfa.ee_pose_in_camera(cart, e) for e in (e_wrist, *e_ext)]
  return cart, poses, (e_wrist, *e_ext)


def _bytes(x):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()]))


def _int(x):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x)]))


class GeometryTest(unittest.TestCase):

  def test_euler_matches_droid_convention(self):
    try:
      from scipy.spatial.transform import Rotation
    except ImportError:
      self.skipTest("scipy not installed")
    rpy = np.random.default_rng(0).uniform(-np.pi, np.pi, (64, 3))
    ref = Rotation.from_euler("xyz", rpy).as_matrix()
    np.testing.assert_allclose(cfa.euler_xyz_to_matrix_np(rpy), ref, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(cfa.euler_xyz_to_matrix(jnp.asarray(rpy, jnp.float32))), ref, atol=1e-5
    )

  def test_anchoring_static_and_wrist(self):
    rng = np.random.default_rng(1)
    t = 16
    cart, poses, (e_wrist, e1, e2) = _episode_geometry(rng, t)
    # Rigid mount: the wrist's own per-frame pose is constant.
    np.testing.assert_allclose(poses[0], np.broadcast_to(poses[0][:1], poses[0].shape), atol=1e-5)
    cam_pose = np.stack([p.reshape(t, 12) for p in poses], axis=1)[None]
    t_ee = cfa.cartesian_to_pose_np(cart)
    for anchor in (0, 5, 15):
      got = np.asarray(cfa.anchored_camera_poses(
          jnp.asarray(cam_pose), jnp.asarray(cart[None]), jnp.asarray([anchor])
      ))[0]
      want = np.stack(
          [(e_wrist[anchor] @ t_ee)[:, :3], (e1 @ t_ee)[:, :3], (e2 @ t_ee)[:, :3]], axis=1
      )
      np.testing.assert_allclose(got, want, atol=1e-4)

  def test_features_shape_and_mask(self):
    rng = np.random.default_rng(2)
    cart, poses, _ = _episode_geometry(rng, 8)
    cam_pose = np.stack([p.reshape(8, 12) for p in poses], axis=1)[None]
    valid = np.ones((1, 8), bool)
    valid[0, 3] = False
    feats = cfa.camera_action_features(
        jnp.asarray(cam_pose), jnp.asarray(cart[None]), jnp.full((1, 8), 0.5), jnp.asarray([2]),
        -np.ones((3, 3), np.float32), np.ones((3, 3), np.float32), valid=jnp.asarray(valid),
    )
    self.assertEqual(feats.shape, (1, 8, cfa.NUM_VIEWS, cfa.CAM_ACTION_DIM))
    self.assertTrue(bool(jnp.all(feats[0, 3] == 0)))
    self.assertTrue(bool(jnp.all(feats[0, 2, :, -1] == 0.5)))

  def test_wan_anchor_and_valid(self):
    anchor, valid = cfa.wan_anchor_and_valid(jnp.asarray([[0, 0, 0, 1, 2], [4, 5, 6, 7, 8]]), n_hist=3)
    np.testing.assert_array_equal(np.asarray(anchor), [8, 11])
    expect = np.ones((2, 20), bool)
    expect[0, [1, 2, 3, 5, 6, 7, 9, 10, 11]] = False
    np.testing.assert_array_equal(np.asarray(valid), expect)


class DatasetTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.mkdtemp()
    with open(os.path.join(self.tmp, "stats.json"), "w") as f:
      json.dump({"state_01": [0.0] * 3 + [-np.pi] * 3 + [0.0], "state_99": [1.0] * 3 + [np.pi] * 3 + [1.0]}, f)

  def _wan_example(self, rng, episode_id, f_lat):
    """One WAN episode: ``4*f_lat - 3`` raw frames (frame 0 encodes one)."""
    t_raw = 4 * f_lat - 3
    cart, poses, _ = _episode_geometry(rng, t_raw)
    action = np.concatenate([cart, rng.uniform(0, 1, (t_raw, 1))], 1).astype(np.float32)
    feats = {
        **{f"latent_cam{i}": _bytes(np.zeros((f_lat, 4, 2, 2), np.float16)) for i in range(3)},
        **{k: _bytes(p) for k, p in zip(cfa.EE_POSE_KEYS, poses)},
        "action": _bytes(action),
        "text_embed": _bytes(np.zeros((2, 8), np.float16)),
        "episode_id": _int(episode_id),
        "traj_len": _int(f_lat),
    }
    return tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString(), cart, poses

  def test_wan_slots_align_with_actions(self):
    from maxdiffusion.input_pipeline.robot.wan_ctrl_world_dataset import WanCtrlWorldDroidDataset

    record, cart, poses = self._wan_example(np.random.default_rng(3), 7, f_lat=4)
    with tf.io.TFRecordWriter(os.path.join(self.tmp, "shard-0000.tfrecord")) as w:
      w.write(record)
    ds = WanCtrlWorldDroidDataset(
        data_dir=self.tmp, stats_path=os.path.join(self.tmp, "stats.json"), n_hist=2,
        max_latent_frames=4, batch_size=1, split="val", shuffle=False,
        shard_for_training=False, load_cam_pose=True, first_window_only=True, repeat=False,
    )
    batch = next(iter(ds))
    # frame_now=1, n_hist=2 -> frames [0, 0, 1, 2] -> raw slots:
    raw = [0, -1, -1, -1, 0, -1, -1, -1, 1, 2, 3, 4, 5, 6, 7, 8]
    for slot, r in enumerate(raw):
      if r < 0:
        self.assertTrue(np.all(batch["cam_pose"][0, slot] == 0))
        self.assertTrue(np.all(batch["ee_cartesian"][0, slot] == 0))
      else:
        np.testing.assert_allclose(batch["ee_cartesian"][0, slot], cart[r])
        for v in range(3):
          np.testing.assert_allclose(batch["cam_pose"][0, slot, v], poses[v][r].reshape(12))
    np.testing.assert_array_equal(batch["frame_positions"][0], [0, 0, 1, 2])

  def test_wan_rejects_misaligned_pose_rows(self):
    from maxdiffusion.input_pipeline.robot.wan_ctrl_world_dataset import WanCtrlWorldDroidDataset

    rng = np.random.default_rng(8)
    example = tf.train.Example()
    example.ParseFromString(self._wan_example(rng, 1, f_lat=4)[0])
    # A 15 Hz pose track in a 5 Hz shard: three pose rows per action row.
    _, poses, _ = _episode_geometry(rng, 3 * 13)
    for k, pose in zip(cfa.EE_POSE_KEYS, poses):
      example.features.feature[k].CopyFrom(_bytes(pose))
    with tf.io.TFRecordWriter(os.path.join(self.tmp, "shard-0000.tfrecord")) as w:
      w.write(example.SerializeToString())
    ds = WanCtrlWorldDroidDataset(
        data_dir=self.tmp, stats_path=os.path.join(self.tmp, "stats.json"), n_hist=2,
        max_latent_frames=4, batch_size=1, split="val", shuffle=False,
        shard_for_training=False, load_cam_pose=True, first_window_only=True, repeat=False,
    )
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "one row per action row"):
      next(iter(ds))

  def test_stats_script_on_consistent_shards(self):
    from maxdiffusion import compute_cam_action_stats as stats_script

    rng = np.random.default_rng(7)
    with tf.io.TFRecordWriter(os.path.join(self.tmp, "shard-0000.tfrecord")) as w:
      for ep in range(3):
        w.write(self._wan_example(rng, ep, f_lat=8)[0])
    config = types.SimpleNamespace(
        model_name="wan2.2", train_data_dir=self.tmp,
        action_stats_path=os.path.join(self.tmp, "stats.json"),
        num_history_latent_frames=2, num_predicted_latents=2, action_dim=7, seed=0,
    )
    summary = stats_script.summarize_windows(stats_script._windows(config), n_windows=16)
    self.assertEqual(summary["num_windows"], 16)
    self.assertEqual(summary["xyz_01"].shape, (3, 3))
    self.assertTrue(np.all(summary["xyz_01"] <= summary["xyz_99"]))
    # Writer and reader agree by construction here, so every invariant holds.
    self.assertLess(float(summary["ext_drift"].max()), 1e-4)
    self.assertLess(float(summary["wrist_drift"].max()), 1e-4)
    self.assertLess(summary["ortho"], 1e-4)

  def _svd_record(self, rng, t5, t15):
    cart, poses, _ = _episode_geometry(rng, t15)
    feats = {
        **{f"latent_cam{i}": _bytes(np.zeros((t5, 4, 2, 2), np.float16)) for i in range(3)},
        **{k: _bytes(p) for k, p in zip(cfa.EE_POSE_KEYS, poses)},
        "cartesian": _bytes(cart),
        "gripper": _bytes(rng.uniform(0, 1, (t15, 1)).astype(np.float32)),
        "text_embed": _bytes(np.zeros((512,), np.float32)),
        "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"pick"])),
        "episode_id": _int(3),
        "traj_len_5hz": _int(t5),
        "traj_len_15hz": _int(t15),
        "success": _int(1),
    }
    with tf.io.TFRecordWriter(os.path.join(self.tmp, "shard-0000.tfrecord")) as w:
      w.write(tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString())
    return cart, poses

  def test_svd_gathers_at_state_ids(self):
    from maxdiffusion.input_pipeline.robot.ctrl_world_droid_dataset import (
        CtrlWorldDroidLatentDataset,
        CtrlWorldDroidRolloutDataset,
    )

    rng = np.random.default_rng(4)
    cart, poses = self._svd_record(rng, t5=30, t15=90)
    stats = os.path.join(self.tmp, "stats.json")
    roll = next(iter(CtrlWorldDroidRolloutDataset(
        data_dir=self.tmp, stats_path=stats, window_frames=5, load_cam_pose=True
    )))
    train = next(iter(CtrlWorldDroidLatentDataset(
        data_dir=self.tmp, stats_path=stats, num_history=2, num_frames=2, batch_size=1,
        split="val", shuffle=False, shard_for_training=False, load_cam_pose=True,
    )))
    # Rollout: frames 0..4 -> states 0, 3, ..., 12. Val window at frame_now=0 with
    # skip_his=4: frames clip([-8, -4, 0, 1]) -> states [0, 0, 0, 3].
    for batch, state_ids in ((roll, [0, 3, 6, 9, 12]), (train, [0, 0, 0, 3])):
      np.testing.assert_allclose(batch["ee_cartesian"][0], cart[state_ids])
      for v in range(3):
        np.testing.assert_allclose(batch["cam_pose"][0, :, v], poses[v][state_ids].reshape(-1, 12))


class WanTest(unittest.TestCase):

  def test_cross_attention_locks_frame_and_camera(self):
    from maxdiffusion.models.wan.transformers.transformer_wan import WanRotaryPosEmbed, WanTransformerBlock

    b, f, h_lat, w_lat, k, v, dim = 1, 2, 6, 4, 2, 3, 32
    length = f * (h_lat // 2) * (w_lat // 2)
    per_view = length // (f * v)
    mesh = _mesh()
    with mesh, nn_partitioning.axis_rules(_RULES):
      block = WanTransformerBlock(
          rngs=nnx.Rngs(0), dim=dim, ffn_dim=64, num_heads=2, cross_attn_norm=True,
          attention="dot_product", mesh=mesh,
      )
      rope = WanRotaryPosEmbed(16, (1, 2, 2), 64)(jnp.ones((b, f, h_lat, w_lat, 4)))
      hidden = jax.random.normal(jax.random.key(1), (b, length, dim))
      enc = jax.random.normal(jax.random.key(2), (b, f * v * k, dim))
      temb = 0.1 * jax.random.normal(jax.random.key(3), (b, 6, dim))
      call = functools.partial(block, frame_level_cond=True, cond_tokens_per_frame=k)
      base = call(hidden, enc, temb, rope)
      for frame, view in ((0, 2), (1, 1)):
        g = frame * v + view
        moved = call(hidden, enc.at[:, g * k:(g + 1) * k].add(3.0), temb, rope)
        changed = np.where(np.abs(np.asarray(moved - base)).max(-1)[0] > 1e-5)[0]
        np.testing.assert_array_equal(changed, np.arange(g * per_view, (g + 1) * per_view))

  def _tiny_setup(self, mode, mesh):
    from maxdiffusion.models.wan.transformers.transformer_wan import WanModel
    from maxdiffusion.trainers.wan_ctrl_world_trainer import (
        TrainState, WanCtrlWorldModel, _build_cam_action_modules,
    )

    config = types.SimpleNamespace(
        action_cond_mode=mode, action_tokens_per_latent_frame=2, wan_action_encoder_hidden_dim=32,
        wan_text_dim=32, seed=0, activations_dtype="float32", weights_dtype="float32",
        global_batch_size_to_train_on=2, num_history_latent_frames=2, grad_accum_steps=1,
        history_noise_max_timestep=0, use_task_instructions=False, ctrl_cfg_drop_prob=0.0,
        disable_training_weights=False, log_attn_activation_stats=False,
        grad_norm_skip_threshold=0.0, log_attn_param_stats=False, cam_action_embed_alpha=0.1,
    )
    with mesh, nn_partitioning.axis_rules(_RULES):
      transformer = WanModel(
          rngs=nnx.Rngs(0), model_type="ti2v", patch_size=(1, 2, 2), num_attention_heads=2,
          attention_head_dim=16, in_channels=4, out_channels=4, text_dim=32, freq_dim=16,
          ffn_dim=64, num_layers=2, attention="dot_product", mesh=mesh, scan_layers=True,
      )
      enc, adaln_proj, add_proj = _build_cam_action_modules(config, transformer.config)
      model = WanCtrlWorldModel(
          transformer, cam_action_encoder=enc, cam_action_adaln_proj=adaln_proj,
          cam_action_add_proj=add_proj,
      )
      graphdef, params, rest = nnx.split(model, nnx.Param, ...)
      state = TrainState.create(
          apply_fn=graphdef.apply, params=params, tx=optax.adam(1e-3),
          graphdef=graphdef, rest_of_state=rest,
      )
    return config, model, state

  def _tiny_batch(self, rng, b=2, f=4, h_lat=6, w_lat=4):
    s = 4 * f
    cam_pose, cart = [], []
    for _ in range(b):
      c, poses, _ = _episode_geometry(rng, s)
      cart.append(c)
      cam_pose.append(np.stack([p.reshape(s, 12) for p in poses], axis=1))
    return {
        "latent": jnp.asarray(rng.normal(size=(b, 4, f, h_lat, w_lat)), jnp.float32),
        "action": jnp.asarray(rng.uniform(-1, 1, (b, s, 7)), jnp.float32),
        "frame_positions": jnp.asarray([[0, 0, 1, 2], [3, 4, 5, 6]][:b], jnp.int32),
        "cam_pose": jnp.asarray(np.stack(cam_pose)),
        "ee_cartesian": jnp.asarray(np.stack(cart)),
        "episode_id": jnp.arange(b, dtype=jnp.int32),
    }

  def test_train_step_all_cam_modes(self):
    from maxdiffusion.schedulers import FlaxFlowMatchScheduler
    from maxdiffusion.trainers.wan_ctrl_world_trainer import (
        _cam_action_tokens, _placeholder_action_tokens, _route_cam_action_conditioning, _train_step,
    )

    mesh = _mesh()
    stats = (-np.ones((3, 3), np.float32), np.ones((3, 3), np.float32))
    sched = FlaxFlowMatchScheduler(dtype=jnp.float32)
    sched_state = sched.set_timesteps(sched.create_state(), num_inference_steps=1000, training=True)
    for mode in ("cam_action", "cam_action_adaln", "cam_action_cross_attn"):
      with self.subTest(mode=mode):
        config, model, state = self._tiny_setup(mode, mesh)
        batch = self._tiny_batch(np.random.default_rng(5))
        with mesh, nn_partitioning.axis_rules(_RULES):
          # Step 0 is the unconditioned model: zero-init encoder output routes to
          # zero conditioning at every site.
          tokens = _cam_action_tokens(model.cam_action_encoder, batch, 2, 4, 2, stats)
          self.assertEqual(tokens.shape, (2, 4 * 3 * 2, 32))
          enc, adaln, add = _route_cam_action_conditioning(model, tokens, mode, 2, 4, 6, 4)
          ts = jnp.full((2, 4 * 6), 500.0)
          common = dict(hidden_states=batch["latent"], timestep=ts, frame_positions=batch["frame_positions"])
          out = model.transformer(
              encoder_hidden_states=enc, action_hidden_states=adaln, skeleton_hidden_states=add,
              frame_level_cond=mode == "cam_action_cross_attn", cond_tokens_per_frame=2, **common,
          )
          ref = model.transformer(
              encoder_hidden_states=_placeholder_action_tokens(2, 4, 2, 32, jnp.float32), **common
          )
          np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)

          step = jax.jit(functools.partial(_train_step, scheduler=sched, config=config, cam_stats=stats))
          new_state, _, metrics, _ = step(state, batch, jax.random.key(0), sched_state)
        self.assertTrue(np.isfinite(float(metrics["scalar"]["learning/loss"])))
        # linear_3 starts at zero; one Adam step moves it only if gradient reached it.
        k3 = new_state.params["cam_action_encoder"]["linear_3"]["kernel"].value
        self.assertGreater(float(jnp.abs(k3).max()), 0.0)


class SvdTest(unittest.TestCase):

  def test_view_locked_cross_attention(self):
    from maxdiffusion.models.attention_flax import FlaxBasicTransformerBlock

    block = FlaxBasicTransformerBlock(dim=16, n_heads=2, d_head=8)
    hidden = jax.random.normal(jax.random.key(0), (2, 12, 16))
    ctx = jax.random.normal(jax.random.key(1), (2, 6, 16))
    kwargs = {"cross_attn_views": 3}
    with _mesh():
      params = block.init(jax.random.key(2), hidden, ctx)
      base = block.apply(params, hidden, ctx, cross_attention_kwargs=kwargs)
      moved = block.apply(params, hidden, ctx.at[:, 2:4].add(3.0), cross_attention_kwargs=kwargs)
      plain = block.apply(params, hidden, ctx)
      empty_kwargs = block.apply(params, hidden, ctx, cross_attention_kwargs={})
    changed = np.where(np.abs(np.asarray(moved - base)).max(-1).max(0) > 1e-5)[0]
    np.testing.assert_array_equal(changed, np.arange(4, 8))
    # Unset -> the ordinary block, bit for bit.
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(empty_kwargs))

  def test_temporal_context_is_view_locked(self):
    from maxdiffusion.models.svd.video_attention_flax import FlaxSpatialVideoTransformer

    b, t, v, h, w = 2, 3, 3, 6, 2
    ctx = jax.random.normal(jax.random.key(0), (b * t, v * 2, 4))
    module = FlaxSpatialVideoTransformer(in_channels=8, n_heads=1, d_head=8, context_dim=4)
    out = module.apply({}, ctx, t, h, w, v, method=FlaxSpatialVideoTransformer._build_time_context)
    # Each view's (h // v) * w positions get the mean of that view's frame-0 keys only.
    frame0 = np.asarray(ctx).reshape(b, t, v, 2, 4)[:, 0].mean(axis=2)
    expect = np.repeat(frame0.reshape(b * v, 1, 4), (h // v) * w, axis=0)
    np.testing.assert_allclose(np.asarray(out), expect, atol=1e-6)

  def test_batch_cast_keeps_pose_geometry_float32(self):
    from maxdiffusion.trainers.ctrl_world_trainer import _cast_batch

    batch = {k: jnp.ones((1, 2), jnp.float32) for k in ("latent", "cam_pose", "ee_cartesian")}
    batch["n"] = jnp.ones((1,), jnp.int32)
    cast = _cast_batch(batch, jnp.bfloat16)
    self.assertEqual(cast["latent"].dtype, jnp.bfloat16)
    self.assertEqual(cast["cam_pose"].dtype, jnp.float32)
    self.assertEqual(cast["ee_cartesian"].dtype, jnp.float32)
    self.assertEqual(cast["n"].dtype, jnp.int32)

  def test_train_step_all_cam_modes(self):
    from maxdiffusion.models.svd.camera_action_encoder_flax import build_cam_action_modules
    from maxdiffusion.models.svd.ctrl_world_flax import CtrlWorldTrainConfig, action_world_train_step
    from maxdiffusion.models.svd.video_unet_flax import FlaxVideoUNet

    b, n_hist, n_fut, h, w = 1, 2, 2, 24, 8
    t = n_hist + n_fut
    unet = FlaxVideoUNet(
        block_out_channels=(32, 32, 32, 32), num_attention_heads=(1, 1, 1, 1),
        cross_attention_dim=32, layers_per_block=1,
    )
    mesh = _mesh()
    with mesh:
      unet_params = unet.init(
          {"params": jax.random.key(0), "dropout": jax.random.key(1)},
          jnp.zeros((b * t, 8, h, w)), jnp.ones((b * t,)), jnp.zeros((b * t, 1, 32)),
          {"adm_vector": jnp.zeros((b, 768))}, None, num_frames=t,
      )["params"]
    rng = np.random.default_rng(6)
    cart, poses, _ = _episode_geometry(rng, t)
    batch = {
        "latent": jnp.asarray(rng.normal(size=(b, t, 4, h, w)), jnp.float32),
        "action": jnp.asarray(rng.uniform(-1, 1, (b, t, 7)), jnp.float32),
        "cam_pose": jnp.asarray(np.stack([p.reshape(t, 12) for p in poses], axis=1)[None]),
        "ee_cartesian": jnp.asarray(cart[None]),
    }
    for mode in ("cam_action", "cam_action_adaln", "cam_action_cross_attn"):
      with self.subTest(mode=mode):
        extras = build_cam_action_modules(
            mode, action_dim=cfa.CAM_ACTION_DIM, hidden_size=32, num_views=3,
            time_embed_dim=128, model_channels=32, seed=0,
            dtype=jnp.float32, weights_dtype=jnp.float32,
        )
        params = {"unet": unet_params, **{k: p for k, (_m, p) in extras.items()}}
        apply_fns = {"unet": unet.apply, **{k: m.apply for k, (m, _p) in extras.items()}}
        cfg = CtrlWorldTrainConfig(
            num_history=n_hist, num_frames=n_fut, hidden_size=32, text_embed_dim=None,
            action_cond_mode=mode, time_embed_dim=128, model_channels=32,
            cam_action_xyz_lo=-np.ones((3, 3), np.float32), cam_action_xyz_hi=np.ones((3, 3), np.float32),
            use_task_instructions=False, cfg_drop_prob=0.0,
        )
        with mesh:
          loss, grads = jax.value_and_grad(
              lambda p: action_world_train_step(jax.random.key(3), p, apply_fns, batch, cfg, 0.18215)
          )(params)
        self.assertTrue(np.isfinite(float(loss)))
        g3 = grads["cam_action_encoder"]["linear_3"]["kernel"]
        self.assertGreater(float(jnp.abs(g3).max()), 0.0)


if __name__ == "__main__":
  absltest.main()
