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

Each trajectory yields one window per pass: frame_now is sampled uniformly from
valid positions. History frames are picked with a random stride going backwards
from frame_now (ctrl-world style); future frames are contiguous from frame_now.
Three camera latents are concatenated along H and transposed to channel-first.

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

    Each trajectory yields one window per pass: frame_now is sampled uniformly
    from valid positions. History is the ``n_hist`` frames immediately before
    ``frame_now`` (indices clipped to episode start, so frame 0 and its actions
    repeat when there isn't enough past); future frames are contiguous from
    ``frame_now``.

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
        shuffle:            Whether to shuffle.
        shuffle_buffer:     Trajectory-level shuffle buffer size.
        shard_for_training: Shard files across JAX processes.
        repeat:             Repeat the dataset indefinitely. Defaults to
                            ``split == "train"``. Training needs it; in-training
                            eval also passes ``True`` so every host yields the
                            same number of batches forever (a finite, per-host
                            sharded val set drains mid-run and hosts then
                            exhaust on different steps, diverging on the
                            collective eval step). Leave ``False`` for offline
                            consumers that iterate val to completion.
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
        shuffle: bool = True,
        shuffle_buffer: int = 512,
        shard_for_training: bool = True,
        first_window_only: bool = False,
        repeat: bool | None = None,
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
            cycle_length=32,
            num_parallel_calls=AUTOTUNE,
            deterministic=not self._is_train,
        )
        ds = ds.with_options(_tf_options(deterministic=not self._is_train))
        ds = ds.map(self._parse, num_parallel_calls=AUTOTUNE)

        # first_window_only requires the full window starting at frame 0 to fit;
        # otherwise the future portion plus the anchor frame (frame_now >= 1,
        # so the anchor latent is never predicted) needs to fit.
        min_traj_len = self.max_latent_frames if first_window_only else self.n_fut + 1
        ds = ds.filter(lambda traj: tf.greater_equal(traj["traj_len"], min_traj_len))

        if self._is_train if repeat is None else repeat:
            ds = ds.repeat()
        if shuffle and self._is_train:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

        if first_window_only:
            # One window per episode: history = episode frame 0, actions indexed from 0.
            ds = ds.map(self._first_window, num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(self._random_window, num_parallel_calls=AUTOTUNE)

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

    def _first_window(self, traj: dict) -> dict:
        """Return the single window anchored at the start of the episode.

        frame_now = 1 so that all history frames clip to episode frame 0 —
        simulating a real rollout start where only one initial observation is
        available. For n_hist=2: hist ids = clip([-1, 0], 0, T-1) = [0, 0]
        (frame 0 repeated).
        Future frames are [1, ..., n_fut], giving a 0-based action tensor whose
        row groups align with the AR slice offsets in run_ar_denoising.
        """
        frame_now = tf.constant(1, dtype=tf.int32)
        return self._build_window(traj, frame_now)

    def _random_window(self, traj: dict) -> dict:
        """Return one window with frame_now sampled uniformly from valid range.

        minval=1 keeps the episode anchor latent (frame 0: encodes 1 raw frame,
        zero-padded actions) out of the predicted window — it can only appear
        as history, so every predicted latent is a full 4-raw-frame chunk.
        """
        T = tf.cast(traj["traj_len"], tf.int32)
        frame_now = tf.random.uniform([], minval=1, maxval=T - self.n_fut + 1, dtype=tf.int32)
        return self._build_window(traj, frame_now)

    def _build_window(self, traj: dict, frame_now: tf.Tensor) -> dict:
        n_hist = self.n_hist
        n_fut = self.n_fut
        W = n_hist + n_fut
        T = tf.cast(traj["traj_len"], tf.int32)

        # History: the n_hist frames immediately before frame_now; clipping to 0
        # repeats frame 0 (and its actions) when there isn't enough past.
        # Future: n_fut contiguous frames starting at frame_now.
        rgb_id = tf.concat([
            tf.cast(frame_now, tf.int32) - tf.range(n_hist, 0, -1),
            tf.cast(frame_now, tf.int32) + tf.range(n_fut),
        ], axis=0)                                                   # (W,)
        rgb_id = tf.clip_by_value(rgb_id, 0, T - 1)

        # Gather latents: (W, C, H_lat, W_lat) per camera.
        cam0_w = tf.cast(tf.gather(traj["cam0"], rgb_id, axis=0), tf.float32)
        cam1_w = tf.cast(tf.gather(traj["cam1"], rgb_id, axis=0), tf.float32)
        cam2_w = tf.cast(tf.gather(traj["cam2"], rgb_id, axis=0), tf.float32)

        # Concat cameras along H: (W, C, H_lat*3, W_lat), then channel-first: (C, W, H_lat*3, W_lat).
        latent = tf.concat([cam0_w, cam1_w, cam2_w], axis=-2)
        latent = tf.transpose(latent, [1, 0, 2, 3])

        # Normalise to [-1, 1], then insert 3 zero-padded slots after action[0]:
        #   [a[0], 0, 0, 0, a[1], ..., a[T-1]]  — shape (T_raw+3, action_dim)
        # Latent frame k indexes padded positions 4k..4k+3, giving:
        #   k=0 → [a[0], 0, 0, 0];  k>0 → [a[4k-3], a[4k-2], a[4k-1], a[4k]]
        action_norm = 2.0 * (traj["action_raw"] - self._p01) / (self._p99 - self._p01 + 1e-8) - 1.0
        action_norm = tf.clip_by_value(action_norm, -1.0, 1.0)

        action_padded = tf.concat([
            action_norm[0:1],
            tf.zeros([3, self.action_dim], dtype=tf.float32),
            action_norm[1:],
        ], axis=0)                                                    # (T_raw+3, action_dim)

        padded_indices = tf.reshape(
            rgb_id[:, None] * 4 + tf.range(4)[None, :], [-1]
        )                                                             # (4*W,)
        action = tf.gather(action_padded, padded_indices, axis=0)    # (4*W, action_dim)

        # Set static shapes for downstream tracing.
        latent.set_shape([None, W, None, None])
        action.set_shape([4 * W, self.action_dim])
        rgb_id.set_shape([W])

        return {
            "latent":          latent,
            "action":          action,
            "text_embeds":     traj["text_embed"],
            "frame_positions": rgb_id,
            # Sequential episode index from preprocessing (int32-safe); threaded
            # through the batch so grad-spike steps can be attributed to episodes.
            "episode_id":      tf.cast(traj["episode_id"], tf.int32),
        }

    def __iter__(self):
        return self.dataset.as_numpy_iterator()
