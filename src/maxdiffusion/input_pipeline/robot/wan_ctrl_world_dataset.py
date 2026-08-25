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

Skeleton-conditioned datasets (``load_skeleton=True``) carry three more
features with exactly the same shape and dtype as their ``latent_cam*``
counterparts:

    skeleton_cam0/1/2: (F_lat, C, H_lat, W_lat) float16

These are the VAE latents of a rendered 2D-kinematic-skeleton video — robot
proprioception turned into an image via URDF forward kinematics and camera
projection — encoded with the *same* WAN VAE as the RGB video. Because the
shapes and the 3-camera H-concat match token for token, they can be injected
additively in the transformer's token space (see ``NNXWanSkeletonPatchEmbed``).
Only datasets built with the skeleton pass have them, so the feature spec is
selected by the flag rather than always requested: ``FixedLenFeature`` on a
missing key fails at parse time for every record.

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
and, with ``load_skeleton=True``:
    skeleton:    (B, C, W, H_lat*3, W_lat)  float32  — same window, same layout
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

# Only present in datasets built with the skeleton-rendering pass.
_SKELETON_FEATURE_DESCRIPTION = {
    "skeleton_cam0": tf.io.FixedLenFeature([], tf.string),
    "skeleton_cam1": tf.io.FixedLenFeature([], tf.string),
    "skeleton_cam2": tf.io.FixedLenFeature([], tf.string),
}

_CAM_KEYS = ("cam0", "cam1", "cam2")
_SKEL_KEYS = ("skel0", "skel1", "skel2")


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
        load_skeleton:      Also read the ``skeleton_cam*`` features and emit a
                            ``skeleton`` tensor alongside ``latent``, windowed
                            identically. Required by
                            ``action_cond_mode='skeleton'``; only valid on a
                            dataset built with the skeleton pass (e.g.
                            ``droid_wan_skeletal_192_320``).
        pad_short_episodes: Keep episodes shorter than ``max_latent_frames``
                            instead of filtering them out, padding the window by
                            repeating the last real action and latent frame while
                            the RoPE positions keep advancing. Requires
                            ``first_window_only``. Each batch reports
                            ``n_real_frames`` so consumers know where the
                            ground-truth portion of the window ends.
        min_latent_frames:  With ``pad_short_episodes``, the minimum episode
                            length to keep (floored at 2 so at least one real
                            frame is predicted). 0 uses that floor.
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
        load_skeleton: bool = False,
        first_window_only: bool = False,
        pad_short_episodes: bool = False,
        min_latent_frames: int = 0,
        repeat: bool | None = None,
    ):
        if max_latent_frames <= 0:
            raise ValueError("max_latent_frames must be > 0")
        if pad_short_episodes and not first_window_only:
            # _random_window samples frame_now from [1, T - n_fut + 1), which is
            # empty once T <= n_fut, so the random-window path genuinely needs the
            # length filter.
            raise ValueError("pad_short_episodes requires first_window_only=True")

        _configure_tf()
        tf.random.set_seed(seed)

        self._is_train = split == "train"
        self._emit_n_real_frames = pad_short_episodes
        self._load_skeleton = load_skeleton
        self._feature_description = dict(_FEATURE_DESCRIPTION)
        if load_skeleton:
            self._feature_description.update(_SKELETON_FEATURE_DESCRIPTION)
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

        # Only the future portion plus the anchor frame has to fit: frame_now >= 1
        # in both window modes, so the highest index touched is
        # frame_now + n_fut - 1 <= T - 1, i.e. T >= n_fut + 1. This holds for
        # first_window_only too — it pins frame_now = 1 and clips every history
        # frame to 0, so history needs no episode length of its own. Requiring the
        # whole n_hist + n_fut window to fit (as this did) dropped episodes that
        # covered the window fine: at n_hist=7, n_fut=50 it wanted 57 frames where
        # 51 suffice, losing 2 of 10 eligible val episodes.
        if pad_short_episodes:
            # Short episodes are kept and padded by _build_window's clamped gather
            # (last action and last latent repeat). Still require >= 2 frames so
            # there is at least one real predicted frame beyond the frame-0 anchor.
            min_traj_len = max(min_latent_frames, 2)
        else:
            min_traj_len = self.n_fut + 1
        self.min_traj_len = min_traj_len
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
        f = tf.io.parse_single_example(serialised, self._feature_description)

        def _cams(prefix: str) -> list:
            """Parse the three per-camera latent tensors under ``prefix``."""
            out = []
            for i in range(3):
                cam = tf.io.parse_tensor(f[f"{prefix}_cam{i}"], out_type=tf.float16)
                # (F_lat, C, H_lat, W_lat) time-first
                cam.set_shape([None, None, None, None])
                out.append(cam)
            return out

        cam0, cam1, cam2 = _cams("latent")

        action = tf.io.parse_tensor(f["action"], out_type=tf.float32)
        # (4*F_lat, 7) raw unnormalised — 4 consecutive raw frames per latent frame
        action.set_shape([None, None])

        text_embed = tf.cast(
            tf.io.parse_tensor(f["text_embed"], out_type=tf.float16), tf.float32
        )  # (512, 4096)
        text_embed.set_shape([None, None])

        traj_len = tf.cast(f["traj_len"], tf.int32)

        out = {
            "cam0":       cam0,
            "cam1":       cam1,
            "cam2":       cam2,
            "action_raw": action,
            "text_embed": text_embed,
            "traj_len":   traj_len,
            "episode_id": f["episode_id"],
        }
        if self._load_skeleton:
            for key, cam in zip(_SKEL_KEYS, _cams("skeleton")):
                out[key] = cam
        return out

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
        raw_id = tf.concat([
            tf.cast(frame_now, tf.int32) - tf.range(n_hist, 0, -1),
            tf.cast(frame_now, tf.int32) + tf.range(n_fut),
        ], axis=0)                                                   # (W,)
        # Index used to *gather* latents and actions: clamped to the episode, so a
        # window running off either end repeats the first/last real frame and its
        # actions rather than reading out of bounds.
        rgb_id = tf.clip_by_value(raw_id, 0, T - 1)
        # Index used as the temporal RoPE position: clamped below at 0 (episode
        # start, same as the gather index) but *not* above. When a window runs past
        # the end of a short episode the actions repeat while time keeps advancing,
        # so the model never sees several latent frames claiming one position —
        # which training, where windows always fit, never produces. For any window
        # that does fit, raw_id <= T-1 already and this is identical to rgb_id.
        pos_id = tf.maximum(raw_id, 0)

        def _window_cams(keys) -> tf.Tensor:
            """Gather the window from three per-camera latent stacks and lay them
            out the way the transformer wants: concat the cameras along H, then
            move channels first — (C, W, H_lat*3, W_lat)."""
            per_cam = [
                tf.cast(tf.gather(traj[k], rgb_id, axis=0), tf.float32)  # (W, C, H_lat, W_lat)
                for k in keys
            ]
            stacked = tf.concat(per_cam, axis=-2)                       # (W, C, H_lat*3, W_lat)
            return tf.transpose(stacked, [1, 0, 2, 3])

        latent = _window_cams(_CAM_KEYS)
        # Skeleton latents share the episode's frame indexing exactly (same VAE,
        # same F_lat), so the identical gather keeps them frame-aligned with the
        # video window — including the clamped repeats at either end.
        skeleton = _window_cams(_SKEL_KEYS) if self._load_skeleton else None

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
        if skeleton is not None:
            skeleton.set_shape([None, W, None, None])
        action.set_shape([4 * W, self.action_dim])
        rgb_id.set_shape([W])
        pos_id.set_shape([W])

        out = {
            "latent":          latent,
            "action":          action,
            "text_embeds":     traj["text_embed"],
            "frame_positions": pos_id,
            # Sequential episode index from preprocessing (int32-safe); threaded
            # through the batch so grad-spike steps can be attributed to episodes.
            "episode_id":      tf.cast(traj["episode_id"], tf.int32),
        }
        if skeleton is not None:
            out["skeleton"] = skeleton
        if self._emit_n_real_frames:
            # Window frames backed by real episode data; the remaining W - n_real
            # are repeat-the-last-action padding, so consumers can tell where
            # ground truth stops being meaningful. Emitted only when padding is on:
            # the training step's jit in_shardings pin an exact key set, and an
            # extra key is a pytree-prefix mismatch there.
            out["n_real_frames"] = (
                tf.minimum(T - tf.cast(frame_now, tf.int32), n_fut) + n_hist
            )
        return out

    def __iter__(self):
        return self.dataset.as_numpy_iterator()
