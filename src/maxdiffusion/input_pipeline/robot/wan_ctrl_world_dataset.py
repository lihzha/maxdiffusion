"""TFRecord dataset for action-conditioned WAN (Ctrl-World style) training.

Each TFRecord example stores one full episode in pre-encoded form:

    latent_cam0: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 0
    latent_cam1: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 1
    latent_cam2: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 2
    action:      (T_ep, 7)                 float32 — raw (unnormalised) cartesian+gripper
    text_embed:  (512, 4096)               float16 — T5 text tokens
    episode_id:  int64
    traj_len:    int64                              — F_lat (number of latent frames)

At read time each episode is windowed into (n_hist + n_fut) latent frames. The
three camera latents are concatenated along H and transposed to channel-first,
producing a single ``latent: (C, n_hist+n_fut, H_lat*3, W_lat)`` tensor.

Actions are normalised to [-1, 1] using per-dimension percentile stats loaded
from ``stats_path`` (produced by ``make_wan_ctrl_world_tfrecords.py``).  Four
consecutive raw-frame actions are associated with each latent frame, so the
output action has shape ``(4*(n_hist+n_fut), 7)`` — the trainer groups them
into ``(n_hist+n_fut, 4, 7)`` before passing to the action encoder.

Each yielded batch contains:
    latent:      (B, C, n_hist+n_fut, H_lat*3, W_lat)  float32
    action:      (B, 4*(n_hist+n_fut), 7)               float32  in [-1, 1]
    text_embeds: (B, 512, 4096)                         float32
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
    "latent_cam0": tf.io.FixedLenFeature([], tf.string),
    "latent_cam1": tf.io.FixedLenFeature([], tf.string),
    "latent_cam2": tf.io.FixedLenFeature([], tf.string),
    "action":      tf.io.FixedLenFeature([], tf.string),
    "text_embed":  tf.io.FixedLenFeature([], tf.string),
    "episode_id":  tf.io.FixedLenFeature([], tf.int64),
    "traj_len":    tf.io.FixedLenFeature([], tf.int64),
}


def _configure_tf() -> None:
    tf.config.set_visible_devices([], "GPU")
    with contextlib.suppress(Exception):
        tf.config.set_visible_devices([], "TPU")


def _tf_options(deterministic: bool = False) -> tf.data.Options:
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    opts.autotune.enabled = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_fusion = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_warm_start = True
    opts.experimental_threading.private_threadpool_size = int(
        max(16, psutil.cpu_count(logical=True))
    )
    return opts


def _load_action_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    with tf.io.gfile.GFile(stats_path, "r") as f:
        stat = json.load(f)
    p01 = np.asarray(stat["state_01"], dtype=np.float32)
    p99 = np.asarray(stat["state_99"], dtype=np.float32)
    return p01, p99


class WanCtrlWorldDroidDataset:
    """Pre-encoded TFRecord dataset for action-conditioned WAN training.

    Expects TFRecord shards at ``data_dir/shard-*.tfrecord`` produced by
    ``make_wan_ctrl_world_tfrecords.py``.

    Each episode is windowed into (n_hist + n_fut) consecutive latent frames.
    Three camera latents are concatenated along H at read time, then transposed
    to channel-first, matching the trainer's expected layout.

    Args:
        data_dir:           Directory of ``shard-*.tfrecord`` files.
        stats_path:         Path to ``action_stats.json`` with ``state_01`` /
                            ``state_99`` arrays for per-dimension normalisation.
        n_hist:             Number of history latent frames per window.
        n_fut:              Number of future latent frames per window.
        action_dim:         Width of a single raw-frame action (default 7).
        batch_size:         Per-host batch size.
        split:              ``"train"`` or ``"val"``.
        seed:               Shuffle seed.
        shuffle:            Whether to shuffle.
        shuffle_buffer:     Post-window shuffle buffer size in windows.
        shard_for_training: Shard files across JAX processes.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        stats_path: str,
        n_hist: int = 1,
        n_fut: int = 1,
        action_dim: int = 7,
        batch_size: int,
        split: str = "train",
        seed: int = 0,
        shuffle: bool = True,
        shuffle_buffer: int = 512,
        shard_for_training: bool = True,
    ):
        _configure_tf()
        tf.random.set_seed(seed)

        self._is_train = split == "train"
        self.n_hist = n_hist
        self.n_fut = n_fut
        self.action_dim = action_dim
        self._window_size = n_hist + n_fut
        self._seed = seed

        p01, p99 = _load_action_stats(stats_path)
        if p01.shape != (action_dim,) or p99.shape != (action_dim,):
            raise ValueError(
                f"stats must contain {action_dim}-dim arrays, "
                f"got state_01={p01.shape}, state_99={p99.shape}."
            )
        self._p01 = tf.constant(p01, dtype=tf.float32)
        self._p99 = tf.constant(p99, dtype=tf.float32)

        files = tf.io.gfile.glob(os.path.join(data_dir, "shard-*.tfrecord"))
        if not files:
            raise FileNotFoundError(
                f"No TFRecord shards matched {data_dir}/shard-*.tfrecord."
            )

        ds = tf.data.Dataset.from_tensor_slices(files)
        if shuffle and self._is_train:
            ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
        if shard_for_training:
            ds = ds.shard(jax.process_count(), jax.process_index())

        ds = ds.interleave(
            tf.data.TFRecordDataset,
            cycle_length=AUTOTUNE,
            num_parallel_calls=AUTOTUNE,
            deterministic=not self._is_train,
        )
        ds = ds.with_options(_tf_options(deterministic=not self._is_train))
        ds = ds.map(self._parse, num_parallel_calls=AUTOTUNE)

        # Drop episodes shorter than one window.
        ds = ds.filter(lambda traj: tf.greater_equal(traj["traj_len"], self._window_size))

        ds = ds.flat_map(self._traj_to_windows)

        if self._is_train:
            ds = ds.repeat()
        if shuffle and self._is_train:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        self.dataset = ds

    # ── Trajectory parser ──────────────────────────────────────────────────────

    def _parse(self, serialised: tf.Tensor) -> dict:
        f = tf.io.parse_single_example(serialised, _FEATURE_DESCRIPTION)

        cam0 = tf.io.parse_tensor(f["latent_cam0"], out_type=tf.float16)
        cam1 = tf.io.parse_tensor(f["latent_cam1"], out_type=tf.float16)
        cam2 = tf.io.parse_tensor(f["latent_cam2"], out_type=tf.float16)
        # Each cam: (F_lat, C, H_lat, W_lat) time-first

        action = tf.io.parse_tensor(f["action"], out_type=tf.float32)
        # (T_ep, 7) raw unnormalised

        text_embed = tf.cast(
            tf.io.parse_tensor(f["text_embed"], out_type=tf.float16), tf.float32
        )  # (512, 4096)

        traj_len = tf.cast(f["traj_len"], tf.int32)

        return {
            "cam0":       cam0,
            "cam1":       cam1,
            "cam2":       cam2,
            "action_raw": action,
            "text_embed": text_embed,
            "traj_len":   traj_len,
            "episode_id": f["episode_id"],
        }

    # ── Trajectory → windows ───────────────────────────────────────────────────

    def _traj_to_windows(self, traj: dict) -> tf.data.Dataset:
        traj_len = traj["traj_len"]
        # Valid start indices: window [s, s+window_size) must fit within [0, traj_len).
        n_windows = traj_len - self._window_size + 1
        starts = tf.range(0, n_windows)
        return tf.data.Dataset.from_tensor_slices(starts).map(
            lambda s: self._build_window(traj, s),
            num_parallel_calls=AUTOTUNE,
            deterministic=not self._is_train,
        )

    def _build_window(self, traj: dict, start: tf.Tensor) -> dict:
        W = self._window_size
        lat_indices = tf.range(start, start + W)  # (W,) latent frame indices

        # Gather per-camera latent windows: (W, C, H_lat, W_lat)
        cam0_w = tf.gather(traj["cam0"], lat_indices, axis=0)
        cam1_w = tf.gather(traj["cam1"], lat_indices, axis=0)
        cam2_w = tf.gather(traj["cam2"], lat_indices, axis=0)

        # Concat cameras along H: (W, C, H_lat*3, W_lat)
        latent = tf.cast(tf.concat([cam0_w, cam1_w, cam2_w], axis=-2), tf.float32)
        # Transpose to channel-first: (C, W, H_lat*3, W_lat)
        latent = tf.transpose(latent, [1, 0, 2, 3])

        # Action window: 4 raw frames per latent frame.
        # Latent frame (start+j) → raw frames [4*(start+j) .. 4*(start+j)+3].
        # Equivalently, raw window starts at 4*start with length 4*W.
        T_ep = tf.shape(traj["action_raw"])[0]
        raw_start = start * 4
        raw_indices = tf.range(raw_start, raw_start + 4 * W)
        raw_indices = tf.clip_by_value(raw_indices, 0, T_ep - 1)
        action_raw = tf.gather(traj["action_raw"], raw_indices, axis=0)  # (4*W, 7)

        # Normalise to [-1, 1].
        action = 2.0 * (action_raw - self._p01) / (self._p99 - self._p01 + 1e-8) - 1.0
        action = tf.clip_by_value(action, -1.0, 1.0)

        # Set static shapes for downstream tracing.
        latent.set_shape([None, W, None, None])
        action.set_shape([4 * W, self.action_dim])

        return {
            "latent":      latent,
            "action":      action,
            "text_embeds": traj["text_embed"],
        }

    def __iter__(self):
        return self.dataset.as_numpy_iterator()
