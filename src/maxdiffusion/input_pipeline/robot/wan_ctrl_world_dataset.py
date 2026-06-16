"""TFRecord dataset for action-conditioned WAN (Ctrl-World style) training.

Each TFRecord example stores one full episode in pre-encoded form:

    latent_cam0: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 0
    latent_cam1: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 1
    latent_cam2: (F_lat, C, H_lat, W_lat)  float16 — WAN VAE latent sequence, camera 2
    action:      (T_ep, 7)                 float32 — raw (unnormalised) cartesian+gripper,
                                                     T_ep raw video frames (not pre-grouped)
    text_embed:  (512, 4096)               float16 — T5 text tokens
    episode_id:  int64
    traj_len:    int64                              — F_lat (number of latent frames)

Each trajectory is flat_mapped into windows. History frames are picked with a
random stride going backwards from ``frame_now`` (ctrl-world style); future frames
are contiguous from ``frame_now``. Three camera latents are concatenated along H
and transposed to channel-first.

Actions are normalised to [-1, 1] using per-dimension percentile stats. Latent
frame k uses raw actions at indices 4k..4k+3.

Each yielded batch contains:
    latent:      (B, C, W, H_lat*3, W_lat)  float32
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

    Each trajectory is flat_mapped into per-frame windows. History frames are
    sampled with a random stride ``skip_his`` going backwards from ``frame_now``;
    future frames are contiguous from ``frame_now``. This matches the Ctrl-World
    data-augmentation scheme.

    Args:
        data_dir:           Directory of ``shard-*.tfrecord`` files.
        stats_path:         Path to ``action_stats.json`` with ``state_01`` /
                            ``state_99`` arrays for per-dimension normalisation.
        n_hist:             Number of history latent frames per window.
        max_latent_frames:  Total window length (n_hist + n_fut). Must be > 0.
        action_dim:         Width of a single raw-frame action (default 7).
        batch_size:         Per-host batch size.
        split:              ``"train"`` or ``"val"``.
        seed:               Shuffle seed.
        max_skip_his:       Maximum stride for history frame sampling. At train
                            time ``skip_his`` is drawn uniformly from
                            ``[1, max_skip_his]`` (or forced to 0 with probability
                            ``skip_his_zero_prob``). At val time ``skip_his=1``.
        skip_his_zero_prob: Probability of collapsing all history to repeats of
                            ``frame_now`` (``skip_his=0``). Train only.
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
        max_latent_frames: int,
        action_dim: int = 7,
        batch_size: int,
        split: str = "train",
        seed: int = 0,
        max_skip_his: int = 1,
        skip_his_zero_prob: float = 0.0,
        shuffle: bool = True,
        shuffle_buffer: int = 512,
        shard_for_training: bool = True,
    ):
        if max_latent_frames <= 0:
            raise ValueError("max_latent_frames must be > 0")

        _configure_tf()
        tf.random.set_seed(seed)

        self._is_train = split == "train"
        self.n_hist = n_hist
        self.n_fut = max_latent_frames - n_hist
        self.max_latent_frames = max_latent_frames
        self.action_dim = action_dim
        self.max_skip_his = max_skip_his
        self.skip_his_zero_prob = skip_his_zero_prob
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

        # Keep only trajectories long enough to have at least one valid future window.
        ds = ds.filter(lambda traj: tf.greater_equal(traj["traj_len"], self.n_fut))

        if self._is_train:
            ds = ds.repeat()
        if shuffle and self._is_train:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

        ds = ds.flat_map(self._traj_to_windows)

        if shuffle and self._is_train:
            ds = ds.shuffle(shuffle_buffer * batch_size, seed=seed, reshuffle_each_iteration=True)

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
        cam0.set_shape([None, None, None, None])
        cam1.set_shape([None, None, None, None])
        cam2.set_shape([None, None, None, None])

        action = tf.io.parse_tensor(f["action"], out_type=tf.float32)
        # (4*F_lat, 7) raw unnormalised — 4 consecutive raw frames per latent frame
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

    # ── Trajectory → windows ───────────────────────────────────────────────────

    def _traj_to_windows(self, traj: dict) -> tf.data.Dataset:
        T = tf.cast(traj["traj_len"], tf.int32)
        # frame_now in [0, T - n_fut]: future window frame_now..frame_now+n_fut-1 stays in bounds.
        valid_starts = tf.range(0, T - self.n_fut + 1)
        return tf.data.Dataset.from_tensor_slices(valid_starts).map(
            lambda frame_now: self._build_window(traj, frame_now),
            num_parallel_calls=AUTOTUNE,
            deterministic=not self._is_train,
        )

    def _build_window(self, traj: dict, frame_now: tf.Tensor) -> dict:
        n_hist = self.n_hist
        n_fut = self.n_fut
        W = n_hist + n_fut
        T = tf.cast(traj["traj_len"], tf.int32)

        # Stateless RNG keyed on (episode_id, frame_now) for reproducibility.
        episode = tf.cast(traj["episode_id"], tf.int64)
        seed_a = tf.stack([
            tf.cast(self._seed, tf.int32) + tf.cast(episode % 2147483647, tf.int32),
            tf.cast(frame_now, tf.int32),
        ])
        seed_b = tf.stack([
            tf.cast(self._seed, tf.int32) + tf.cast(episode % 2147483647, tf.int32),
            tf.cast(frame_now, tf.int32) + 7919,
        ])

        if self._is_train:
            zero_skip = tf.random.stateless_uniform([], seed=seed_b) < self.skip_his_zero_prob
            skip_his = tf.where(
                zero_skip,
                tf.constant(0, tf.int32),
                tf.random.stateless_uniform(
                    [], seed=seed_a, minval=1, maxval=self.max_skip_his + 1, dtype=tf.int32
                ),
            )
        else:
            skip_his = tf.constant(1, dtype=tf.int32)

        # History: n_hist frames going back at stride skip_his from frame_now.
        hist_offsets = tf.range(n_hist, 0, -1) * skip_his          # (n_hist,)
        # Future: n_fut contiguous frames starting at frame_now.
        fut_offsets = tf.range(n_fut)                                # (n_fut,)

        rgb_id = tf.concat([
            tf.cast(frame_now, tf.int32) - hist_offsets,
            tf.cast(frame_now, tf.int32) + fut_offsets,
        ], axis=0)                                                   # (W,)
        rgb_id = tf.clip_by_value(rgb_id, 0, T - 1)

        # Gather latents: (W, C, H_lat, W_lat) per camera.
        cam0_w = tf.cast(tf.gather(traj["cam0"], rgb_id, axis=0), tf.float32)
        cam1_w = tf.cast(tf.gather(traj["cam1"], rgb_id, axis=0), tf.float32)
        cam2_w = tf.cast(tf.gather(traj["cam2"], rgb_id, axis=0), tf.float32)

        # Concat cameras along H: (W, C, H_lat*3, W_lat), then channel-first: (C, W, H_lat*3, W_lat).
        latent = tf.concat([cam0_w, cam1_w, cam2_w], axis=-2)
        latent = tf.transpose(latent, [1, 0, 2, 3])

        # Actions: WAN VAE has non-uniform temporal compression — latent 0 is a
        # single-frame anchor (raw frame 0); latent k>0 covers raw frames 4k-3..4k.
        # Formula: action_start = 4*rgb_id[k] - 3, clipped to 0.
        # This naturally repeats action[0] for the anchor ([-3,-2,-1,0] → all 0).
        T_raw = tf.shape(traj["action_raw"])[0]
        raw_indices = tf.clip_by_value(
            tf.reshape(rgb_id[:, None] * 4 - 3 + tf.range(4)[None, :], [-1]),
            0, T_raw - 1,
        )                                                             # (4*W,)
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
