"""TFRecord dataset for action-conditioned SVD (Ctrl-World) training on TPU.

Each TFRecord example carries one DROID trajectory in pre-encoded form
(3 cam latents at 5 Hz + 15 Hz cartesian/gripper state + a single CLIP text
embedding). See docs/ctrl_world_data_format.md for the full schema. At
training time we ``flat_map`` each trajectory into many ``(num_history +
num_frames)``-length windows that match Ctrl-World's GPU loader
(``Ctrl-World/dataset/dataset_droid_exp33.py``).

Yielded sample (batched along axis 0 by ``.batch(batch_size)``):
    latent:      [num_history+num_frames, 4, H_lat_stacked, W_lat]  float32
    action:      [num_history+num_frames, action_dim]               float32 in [-1, 1]
    text_embeds: [text_embed_dim]                                   float32
"""

from __future__ import annotations

import contextlib
import json
import os

import jax
import numpy as np
import psutil
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


_FEATURE_DESCRIPTION = {
    "latent_cam0":    tf.io.FixedLenFeature([], tf.string),
    "latent_cam1":    tf.io.FixedLenFeature([], tf.string),
    "latent_cam2":    tf.io.FixedLenFeature([], tf.string),
    "cartesian":      tf.io.FixedLenFeature([], tf.string),
    "gripper":        tf.io.FixedLenFeature([], tf.string),
    "text_embed":     tf.io.FixedLenFeature([], tf.string),
    "text":           tf.io.FixedLenFeature([], tf.string),
    "episode_id":     tf.io.FixedLenFeature([], tf.int64),
    "traj_len_5hz":   tf.io.FixedLenFeature([], tf.int64),
    "traj_len_15hz":  tf.io.FixedLenFeature([], tf.int64),
    "success":        tf.io.FixedLenFeature([], tf.int64),
}


def _configure_tf_for_jax() -> None:
    tf.config.set_visible_devices([], "GPU")
    with contextlib.suppress(Exception):
        tf.config.set_visible_devices([], "TPU")


def _tf_data_options(deterministic: bool = False) -> tf.data.Options:
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    opts.autotune.enabled = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_fusion = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_warm_start = True
    opts.experimental_threading.private_threadpool_size = int(max(16, psutil.cpu_count(logical=True)))
    return opts


def _load_state_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load `state_01` / `state_99` percentile arrays."""
    with tf.io.gfile.GFile(stats_path, "r") as f:
        stat = json.load(f)
    p01 = np.asarray(stat["state_01"], dtype=np.float32)
    p99 = np.asarray(stat["state_99"], dtype=np.float32)
    return p01, p99


class CtrlWorldDroidLatentDataset:
    """Yields windows of pre-encoded latents + actions for Ctrl-World training.

    Args:
      data_dir:         Directory containing ``shard-*.tfrecord`` files. Either
                        a local path or a ``gs://`` URI.
      stats_path:       Path to ``stats.json`` produced by the data agent.
      num_history:      Number of history slots per window (Ctrl-World default 6).
      num_frames:       Number of future slots per window (Ctrl-World default 5).
                        Total window length = ``num_history + num_frames``.
      action_dim:       Width of the per-frame action (default 7 = cartesian + gripper).
      text_embed_dim:   Width of the CLIP text embedding (default 512).
      batch_size:       Per-host batch size.
      split:            ``"train"`` or ``"val"``. Affects skip / shuffle behaviour.
      seed:             Seed for stateless tf.random calls.
      down_sample:      Stride between the 5 Hz frame index and the 15 Hz state
                        index. Should match the rgb_skip used at pre-encode time.
      max_skip:         Max value of the random ``skip`` for future frames
                        (Ctrl-World draws ``skip in {1, 2}`` → max_skip=2).
      max_skip_his:     ``skip_his = 4 * skip`` in Ctrl-World, so max=8.
      skip_his_zero_prob: With this probability, force ``skip_his = 0`` (collapses
                        the history to repeats of ``frame_now``). Matches the
                        ``p < 0.15`` branch in ``dataset_droid_exp33.py``.
      shuffle:          Whether to shuffle file ordering and post-window samples.
      shuffle_buffer:   Post-window shuffle buffer in samples.
      shard_for_training: If True, shard files across JAX processes.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        stats_path: str,
        num_history: int = 6,
        num_frames: int = 5,
        action_dim: int = 7,
        text_embed_dim: int = 512,
        batch_size: int,
        split: str = "train",
        seed: int = 0,
        down_sample: int = 3,
        max_skip: int = 2,
        max_skip_his: int = 8,
        skip_his_zero_prob: float = 0.15,
        shuffle: bool = True,
        shuffle_buffer: int = 512,
        shard_for_training: bool = True,
    ):
        _configure_tf_for_jax()
        tf.random.set_seed(seed)

        self.num_history = num_history
        self.num_frames = num_frames
        self.action_dim = action_dim
        self.text_embed_dim = text_embed_dim
        self.down_sample = down_sample
        self.max_skip = max_skip
        self.max_skip_his = max_skip_his
        self.skip_his_zero_prob = skip_his_zero_prob
        self.is_train = split == "train"

        p01, p99 = _load_state_stats(stats_path)
        if p01.shape != (action_dim,) or p99.shape != (action_dim,):
            raise ValueError(
                f"stats.json must contain {action_dim}-dim percentile arrays, "
                f"got state_01={p01.shape}, state_99={p99.shape}."
            )
        self._p01 = tf.constant(p01, dtype=tf.float32)
        self._p99 = tf.constant(p99, dtype=tf.float32)

        # Filter trajectories shorter than the worst-case window so we don't waste
        # samples on all-clamped indices.
        self._min_traj_len_5hz = num_history * max_skip_his + num_frames * max_skip + 1
        self._seed = seed

        files = tf.io.gfile.glob(os.path.join(data_dir, "shard-*.tfrecord"))
        if not files:
            raise FileNotFoundError(
                f"No TFRecord shards matched {data_dir}/shard-*.tfrecord. "
                "See docs/ctrl_world_data_format.md for the expected layout."
            )

        ds = tf.data.Dataset.from_tensor_slices(files)
        if shuffle and self.is_train:
            ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
        if shard_for_training:
            ds = ds.shard(jax.process_count(), jax.process_index())
        ds = ds.interleave(
            lambda fname: tf.data.TFRecordDataset(fname),
            cycle_length=32,
            num_parallel_calls=AUTOTUNE,
            deterministic=not self.is_train,
        )
        ds = ds.with_options(_tf_data_options(deterministic=not self.is_train))

        ds = ds.map(self._parse, num_parallel_calls=AUTOTUNE)
        ds = ds.filter(
            lambda traj: tf.greater_equal(traj["traj_len_5hz"], self._min_traj_len_5hz)
        )
        ds = ds.flat_map(self._traj_to_windows)
        if self.is_train:
            ds = ds.repeat()
        if shuffle and self.is_train:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        self.dataset = ds

    # ── Trajectory parser ──────────────────────────────────────────────────────

    def _parse(self, example: tf.Tensor) -> dict:
        f = tf.io.parse_single_example(example, _FEATURE_DESCRIPTION)
        cam0 = tf.io.parse_tensor(f["latent_cam0"], out_type=tf.float16)
        cam1 = tf.io.parse_tensor(f["latent_cam1"], out_type=tf.float16)
        cam2 = tf.io.parse_tensor(f["latent_cam2"], out_type=tf.float16)
        # (T_5hz, 4, H_per_cam, W) → stacked vertically along the H axis.
        latent_stacked = tf.cast(tf.concat([cam0, cam1, cam2], axis=-2), tf.float32)

        cart = tf.io.parse_tensor(f["cartesian"], out_type=tf.float32)
        grip = tf.io.parse_tensor(f["gripper"], out_type=tf.float32)
        state = tf.concat([cart, grip], axis=-1)  # (T_15hz, 7)

        text_embed = tf.io.parse_tensor(f["text_embed"], out_type=tf.float32)
        text_embed = tf.reshape(text_embed, [self.text_embed_dim])

        return {
            "latent_stacked": latent_stacked,
            "state":          state,
            "text_embed":     text_embed,
            "traj_len_5hz":   f["traj_len_5hz"],
            "traj_len_15hz":  f["traj_len_15hz"],
            "episode_id":     f["episode_id"],
        }

    # ── Trajectory → windows ───────────────────────────────────────────────────

    def _traj_to_windows(self, traj: dict) -> tf.data.Dataset:
        T5 = tf.cast(traj["traj_len_5hz"], tf.int32)
        # frame_now must allow at least one valid future frame (frame_now + 0).
        # Index clipping handles edges, but we still need at least num_history
        # leading frames to exist before clipping kicks in for non-zero skip_his.
        # Use the full range [0, T5) and let clipping handle out-of-bounds.
        valid_starts = tf.range(0, T5)
        return tf.data.Dataset.from_tensor_slices(valid_starts).map(
            lambda s: self._build_window(traj, s),
            num_parallel_calls=AUTOTUNE,
            deterministic=not self.is_train,
        )

    def _build_window(self, traj: dict, frame_now: tf.Tensor) -> dict:
        num_history = self.num_history
        num_frames = self.num_frames

        # Stateless RNG keyed on (episode_id, frame_now) so each window has a
        # reproducible draw — essential for val-loss stability and easy debug.
        episode = tf.cast(traj["episode_id"], tf.int64)
        seed_a = tf.stack([
            tf.cast(self._seed, tf.int32) + tf.cast(episode % 2147483647, tf.int32),
            tf.cast(frame_now, tf.int32),
        ])
        seed_b = tf.stack([
            tf.cast(self._seed, tf.int32) + tf.cast(episode % 2147483647, tf.int32),
            tf.cast(frame_now, tf.int32) + 7919,
        ])
        seed_c = tf.stack([
            tf.cast(self._seed, tf.int32) + tf.cast(episode % 2147483647, tf.int32),
            tf.cast(frame_now, tf.int32) + 6271,
        ])

        if self.is_train:
            skip = tf.random.stateless_uniform(
                [], seed=seed_a, minval=1, maxval=self.max_skip + 1, dtype=tf.int32
            )
            zero_skip = tf.random.stateless_uniform([], seed=seed_b) < self.skip_his_zero_prob
            skip_his = tf.where(zero_skip, 0, (self.max_skip_his // self.max_skip) * skip)
        else:
            skip = tf.constant(1, dtype=tf.int32)
            skip_his = tf.constant(self.max_skip_his // self.max_skip, dtype=tf.int32)

        # History indices: frame_now - num_history*skip_his, ..., frame_now - skip_his
        hist_offsets = tf.range(num_history, 0, -1) * skip_his              # (num_history,)
        # Future indices: frame_now, frame_now + skip, ..., frame_now + (num_frames-1)*skip
        fut_offsets  = tf.range(0, num_frames) * skip                       # (num_frames,)

        rgb_id = tf.concat([
            tf.cast(frame_now, tf.int32) - hist_offsets,
            tf.cast(frame_now, tf.int32) + fut_offsets,
        ], axis=0)                                                          # (T,)
        T5 = tf.cast(traj["traj_len_5hz"], tf.int32)
        rgb_id = tf.clip_by_value(rgb_id, 0, T5 - 1)

        T15 = tf.cast(traj["traj_len_15hz"], tf.int32)
        state_id = tf.clip_by_value(rgb_id * self.down_sample, 0, T15 - 1)

        latent = tf.gather(traj["latent_stacked"], rgb_id, axis=0)          # (T,4,H,W)
        state = tf.gather(traj["state"], state_id, axis=0)                  # (T,7)

        # normalize_bound to [-1, 1] (matches Ctrl-World).
        action = 2.0 * (state - self._p01) / (self._p99 - self._p01 + 1e-8) - 1.0
        action = tf.clip_by_value(action, -1.0, 1.0)

        # Ensure shapes are static for downstream tracing.
        T_static = num_history + num_frames
        latent.set_shape([T_static, None, None, None])
        action.set_shape([T_static, self.action_dim])

        return {
            "latent":      latent,
            "action":      action,
            "text_embeds": traj["text_embed"],
        }

    # ── Iterator protocol ──────────────────────────────────────────────────────

    def __iter__(self):
        return self.dataset.as_numpy_iterator()
