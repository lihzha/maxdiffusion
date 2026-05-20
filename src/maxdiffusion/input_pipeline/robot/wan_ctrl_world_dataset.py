"""TFRecord dataset for action-conditioned WAN (Ctrl-World style) training.

Each TFRecord example stores one full episode in pre-encoded form:

    latent_cam0: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 0
    latent_cam1: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 1
    latent_cam2: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 2
    action:      (T_ep, 7)                 float32 — raw (unnormalised) cartesian+gripper
    text_embed:  (512, 4096)               float16 — T5 text tokens
    episode_id:  int64
    traj_len:    int64                              — F_lat (number of latent frames)

Trajectories are batched together; within each batch the window size W equals
``min(traj_len in batch)``, so ``n_fut = W - n_hist`` varies per batch. Three
camera latents are concatenated along H and transposed to channel-first.

Actions are normalised to [-1, 1] using per-dimension percentile stats loaded
from ``stats_path`` (produced by ``make_wan_ctrl_world_tfrecords.py``).  Four
consecutive raw-frame actions are associated with each latent frame, so the
output action has shape ``(4*W, 7)`` — the trainer groups them into
``(W, 4, 7)`` before passing to the action encoder.

Each yielded batch contains:
    latent:      (B, C, W, H_lat*3, W_lat)  float32  W varies per batch
    action:      (B, 4*W, 7)                float32  in [-1, 1]
    text_embeds: (B, 512, 4096)             float32
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

    Trajectories are batched together. Window size W is determined per batch:

    * ``max_latent_frames > 0``: W = ``max_latent_frames`` (static shapes, no
      JAX recompilation). Only trajectories of at least that length are kept.
    * ``max_latent_frames <= 0``: W = ``min(traj_len in batch)`` (dynamic; JAX
      recompiles on each unique W).

    Three camera latents are concatenated along H, then transposed to
    channel-first, matching the trainer's expected layout.

    Args:
        data_dir:           Directory of ``shard-*.tfrecord`` files.
        stats_path:         Path to ``action_stats.json`` with ``state_01`` /
                            ``state_99`` arrays for per-dimension normalisation.
        n_hist:             Number of history latent frames per window.
        max_latent_frames:  Fixed total window length (n_hist + n_fut). Pass
                            ``-1`` (default) to use per-batch dynamic sizing.
        action_dim:         Width of a single raw-frame action (default 7).
        batch_size:         Per-host batch size.
        split:              ``"train"`` or ``"val"``.
        seed:               Shuffle seed.
        shuffle:            Whether to shuffle.
        shuffle_buffer:     Trajectory-level shuffle buffer size.
        shard_for_training: Shard files across JAX processes.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        stats_path: str,
        n_hist: int = 1,
        max_latent_frames: int = -1,
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
        self.max_latent_frames = max_latent_frames
        self.action_dim = action_dim
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

        # Drop episodes shorter than the fixed window (or n_hist+1 if no fixed window).
        min_len = max_latent_frames if max_latent_frames > 0 else n_hist + 1
        ds = ds.filter(lambda traj: tf.greater_equal(traj["traj_len"], min_len))

        if self._is_train:
            ds = ds.repeat()
        if shuffle and self._is_train:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

        ds = ds.padded_batch(batch_size, padded_shapes=None, drop_remainder=True)
        ds = ds.map(self._build_batch, num_parallel_calls=AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        self.dataset = ds

    # ── Trajectory parser ──────────────────────────────────────────────────────

    def _parse(self, serialised: tf.Tensor) -> dict:
        f = tf.io.parse_single_example(serialised, _FEATURE_DESCRIPTION)

        cam0 = tf.io.parse_tensor(f["latent_cam0"], out_type=tf.float16)
        cam1 = tf.io.parse_tensor(f["latent_cam1"], out_type=tf.float16)
        cam2 = tf.io.parse_tensor(f["latent_cam2"], out_type=tf.float16)
        # Each cam: (F_lat, C, H_lat, W_lat) time-first
        cam0.set_shape([None, None, None, None])
        cam1.set_shape([None, None, None, None])
        cam2.set_shape([None, None, None, None])

        action = tf.io.parse_tensor(f["action"], out_type=tf.float32)
        # (T_ep, 7) raw unnormalised
        action.set_shape([None, None])

        text_embed = tf.cast(
            tf.io.parse_tensor(f["text_embed"], out_type=tf.float16), tf.float32
        )  # (512, 4096)
        text_embed.set_shape([None, None])

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

    # ── Batch builder ──────────────────────────────────────────────────────────

    def _build_batch(self, batch: dict) -> dict:
        """Build one training batch from a trajectory-level padded batch.

        Window size W = max_latent_frames if fixed, else min(traj_len in batch).
        A random start is sampled per trajectory so the window fits within its
        valid (non-padded) frames.
        """
        # batch["traj_len"]: (B,) int32
        # batch["cam*"]:     (B, T_max, C, H_lat, W_lat) float16  (padded)
        # batch["action_raw"]: (B, T_ep_max, 7) float32            (padded)

        # Use a fixed window so all batches have the same shape (avoids JAX recompilation).
        W = self.max_latent_frames if self.max_latent_frames > 0 else tf.reduce_min(batch["traj_len"])
        B = tf.shape(batch["traj_len"])[0]

        # Random start per trajectory in [0, traj_len - W].
        max_starts = tf.maximum(batch["traj_len"] - W, 0)  # (B,)
        rand = tf.random.uniform([B], 0.0, 1.0, dtype=tf.float32, seed=self._seed)
        starts = tf.cast(tf.cast(max_starts, tf.float32) * rand, tf.int32)  # (B,)

        # Latent frame indices: (B, W)
        frame_indices = tf.expand_dims(starts, 1) + tf.expand_dims(tf.range(W), 0)

        # Gather window from each cam: (B, W, C, H_lat, W_lat)
        cam0_w = tf.gather(batch["cam0"], frame_indices, batch_dims=1)
        cam1_w = tf.gather(batch["cam1"], frame_indices, batch_dims=1)
        cam2_w = tf.gather(batch["cam2"], frame_indices, batch_dims=1)

        # Concat cameras along H: (B, W, C, H_lat*3, W_lat)
        latent = tf.cast(tf.concat([cam0_w, cam1_w, cam2_w], axis=-2), tf.float32)
        # Transpose to channel-first: (B, C, W, H_lat*3, W_lat)
        latent = tf.transpose(latent, [0, 2, 1, 3, 4])

        # Action window: 4 raw frames per latent frame.
        T_ep_max = tf.shape(batch["action_raw"])[1]
        raw_starts = starts * 4  # (B,)
        raw_frame_indices = (
            tf.expand_dims(raw_starts, 1) + tf.expand_dims(tf.range(4 * W), 0)
        )  # (B, 4W)
        raw_frame_indices = tf.clip_by_value(raw_frame_indices, 0, T_ep_max - 1)
        action_raw = tf.gather(batch["action_raw"], raw_frame_indices, batch_dims=1)  # (B, 4W, 7)

        # Normalise to [-1, 1].
        action = 2.0 * (action_raw - self._p01) / (self._p99 - self._p01 + 1e-8) - 1.0
        action = tf.clip_by_value(action, -1.0, 1.0)

        return {
            "latent":      latent,
            "action":      action,
            "text_embeds": batch["text_embed"],
        }

    def __iter__(self):
        return self.dataset.as_numpy_iterator()
