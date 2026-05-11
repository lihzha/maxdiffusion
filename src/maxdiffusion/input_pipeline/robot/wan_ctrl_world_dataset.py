"""TFRecord dataset for action-conditioned WAN (Ctrl-World style) training.

Each TFRecord example stores **one pre-encoded clip** produced by the offline
data-preparation script:

    latent:     (F_lat, 16, H_lat, W_lat)  float16 — WAN VAE latent
    action:     (T,     action_dim)         float32 — normalised to [-1, 1]
    text_embed: (512,   4096)               float16 — T5 text tokens

where T = num_history + num_frames (e.g. 9 with the default 4+5 split) and
F_lat = (T - 1) // 4 + 1 (WAN temporal compression factor 4; T must satisfy
(T-1) % 4 == 0, e.g. 9, 13, 17 …).

At training time no windowing is needed — each example is already one clip.
The dataset just parses, batches, and prefetches.

Data preparation (offline)
--------------------------
See the docstring of ``WanCtrlWorldDroidDataset`` for the expected offline
data-prep command. Briefly, for each DROID trajectory:

1. Extract T consecutive frames with their cartesian/gripper states.
2. Encode the T-frame video clip with the WAN VAE (temporal factor 4).
3. Encode the episode language instruction with the T5 encoder.
4. Normalise actions to [-1, 1] using dataset-level percentile stats.
5. Serialise as TFRecord.
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
    "latent":     tf.io.FixedLenFeature([], tf.string),
    "action":     tf.io.FixedLenFeature([], tf.string),
    "text_embed": tf.io.FixedLenFeature([], tf.string),
    "episode_id": tf.io.FixedLenFeature([], tf.int64),
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


class WanCtrlWorldDroidDataset:
    """Pre-encoded TFRecord dataset for action-conditioned WAN training.

    Expects TFRecord shards at ``data_dir/shard-*.tfrecord``.  Each shard is
    produced by the offline data-preparation pipeline described in the module
    docstring.

    Each yielded batch contains:
        latent:      ``(B, F_lat, 16, H_lat, W_lat)``  float32
        action:      ``(B, T, action_dim)``              float32  in [-1, 1]
        text_embeds: ``(B, 512, 4096)``                 float32

    The trainer reshapes ``latent`` to ``(B, 16, F_lat, H_lat, W_lat)``
    before passing it to the WAN transformer (C-first convention).

    Args:
        data_dir:            Directory of ``shard-*.tfrecord`` files.
        batch_size:          Per-host batch size.
        split:               ``"train"`` or ``"val"``.
        seed:                Shuffle seed.
        shuffle:             Whether to shuffle.
        shuffle_buffer:      Post-parse shuffle buffer size in examples.
        shard_for_training:  Shard files across JAX processes.
    """

    def __init__(
        self,
        *,
        data_dir: str,
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

        files = tf.io.gfile.glob(os.path.join(data_dir, "shard-*.tfrecord"))
        if not files:
            raise FileNotFoundError(
                f"No TFRecord shards matched {data_dir}/shard-*.tfrecord. "
                "See docs/wan_ctrl_world_data_format.md for the expected layout."
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

        if self._is_train:
            ds = ds.repeat()
        if shuffle and self._is_train:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTOTUNE)
        self.dataset = ds

    def _parse(self, serialised: tf.Tensor) -> dict:
        f = tf.io.parse_single_example(serialised, _FEATURE_DESCRIPTION)

        # Latent: (F_lat, 16, H_lat, W_lat) float16 → cast to float32
        latent = tf.cast(tf.io.parse_tensor(f["latent"], out_type=tf.float16), tf.float32)

        # Action: (T, action_dim) float32
        action = tf.io.parse_tensor(f["action"], out_type=tf.float32)

        # T5 text tokens: (512, 4096) float16 → cast to float32
        text_embed = tf.cast(
            tf.io.parse_tensor(f["text_embed"], out_type=tf.float16), tf.float32
        )

        return {
            "latent":      latent,
            "action":      action,
            "text_embeds": text_embed,
        }

    def __iter__(self):
        return self.dataset.as_numpy_iterator()
