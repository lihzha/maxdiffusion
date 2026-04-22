"""Load video clips from existing DROID TFDS records for WAN I2V training.

Stripped-down copy of DroidDataset from language-action-pretraining with all
action/state processing removed. Replaces per-frame flatten with flat_map-based
sliding-window extraction to yield video clips of fixed length.

Output per sample:
    frames:               [clip_length, height, width, 3]  float32 in [0, 1]
    language_instruction: []                               string
"""

from __future__ import annotations

import contextlib

import jax
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds


# Raw DROID TFDS feature keys (before any standardization)
_EXTERIOR_IMAGE_KEYS = ("exterior_image_1_left", "exterior_image_2_left")
_WRIST_IMAGE_KEY = "wrist_image_left"
_LANGUAGE_KEYS = ("language_instruction", "language_instruction_2", "language_instruction_3")

# DROID control frequency (Hz). Matches WAN's 16 fps closely enough that
# sampling every frame gives the right temporal density.
DROID_CONTROL_FREQUENCY = 15


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
    opts.experimental_optimization.map_and_filter_fusion = True
    opts.experimental_optimization.inject_prefetch = False
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_warm_start = True
    opts.experimental_threading.private_threadpool_size = int(max(16, psutil.cpu_count(logical=True)))
    return opts


def _episode_to_traj(episode: dict) -> tf.data.Dataset:
    """Convert an RLDS episode (with nested steps Dataset) to a single-element
    Dataset containing a trajectory dict with [T, ...] shaped tensors.

    Replicates what dlimp.DLataset.from_rlds does: stacks per-step fields and
    re-wraps episode-level metadata under traj_metadata so the rest of the
    pipeline can use [...][0] indexing on file_path.
    """
    file_path = episode["episode_metadata"]["file_path"]  # scalar string
    steps = episode["steps"]

    # Batch all steps into [T, ...] tensors. DROID trajectories are at most a
    # few hundred frames; 1_000_000 is a safe upper bound and does not
    # pre-allocate memory.
    batched = steps.batch(1_000_000)

    def _build_traj(batch: dict) -> dict:
        return {
            "observation": batch["observation"],
            "language_instruction": batch["language_instruction"],
            "language_instruction_2": batch["language_instruction_2"],
            "language_instruction_3": batch["language_instruction_3"],
            "traj_metadata": {
                "episode_metadata": {
                    # Wrap in a length-1 tensor so callers can use [...][0].
                    "file_path": tf.expand_dims(file_path, 0),
                }
            },
        }

    return batched.map(_build_traj)  # single-element Dataset per trajectory


class DroidVideoDataset:
    """Iterate DROID TFDS records and yield fixed-length video clips.

    Copies the RLDS loading / filtering / image-selection logic from
    language-action-pretraining's DroidDataset but drops all action/state
    processing. Windowing (flat_map over start indices) replaces the
    per-frame flatten step.

    Args:
        data_dir:              Path to the directory that contains the DROID
                               TFDS records (the parent of the ``droid/``
                               subdirectory, i.e. the ``data_dir`` argument
                               passed to ``tfds.builder``).
        clip_length:           Number of consecutive frames per clip.
                               Must satisfy ``(clip_length - 1) % 4 == 0``
                               for WAN VAE compatibility (e.g. 17, 49, 81).
        height:                Decoded frame height in pixels.
        width:                 Decoded frame width in pixels.
        stride:                Step between clip start indices. stride=1 gives
                               maximum overlap; stride=clip_length gives
                               non-overlapping clips.
        split:                 ``"train"`` or ``"val"``. Val uses a 2 % hash
                               split derived from the trajectory ID.
        val_fraction:          Fraction of trajectories reserved for val.
        seed:                  Random seed for instruction / camera selection
                               and train/val split.
        batch_size:            Clips per batch.
        shuffle:               Whether to shuffle at file level and with a
                               post-window buffer.
        shuffle_buffer:        Size of the post-window shuffle buffer (clips).
        num_parallel_reads:    Passed to tfds.ReadConfig as interleave_cycle_length.
        num_parallel_calls:    Used for tf.data map operations.
        tfds_name:             Override the TFDS dataset name. Defaults to
                               ``"droid"``.
        filter_success:        If True, keep only trajectories whose file_path
                               contains the string ``"success"``.
        shard_for_training:    If True, shard across JAX processes (standard
                               for multi-host training). Disable for
                               single-host debugging.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        clip_length: int,
        height: int,
        width: int,
        stride: int = 1,
        split: str = "train",
        val_fraction: float = 0.02,
        seed: int = 0,
        batch_size: int = 1,
        shuffle: bool = True,
        shuffle_buffer: int = 1000,
        num_parallel_reads: int = tf.data.AUTOTUNE,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        tfds_name: str = "droid",
        filter_success: bool = True,
        shard_for_training: bool = True,
    ):
        _configure_tf_for_jax()
        tf.random.set_seed(seed)

        self.clip_length = clip_length
        self.height = height
        self.width = width
        self.stride = stride
        self.seed = seed
        self._num_parallel_calls = num_parallel_calls
        want_val = split == "val"
        deterministic = want_val

        builder = tfds.builder(tfds_name, data_dir=data_dir)

        read_config = tfds.ReadConfig(
            interleave_cycle_length=num_parallel_reads,
            shuffle_seed=seed if (shuffle and not deterministic) else None,
        )
        episodes = builder.as_dataset(
            split="all",
            shuffle_files=shuffle and not deterministic,
            read_config=read_config,
        )

        if shard_for_training and not want_val:
            episodes = episodes.shard(jax.process_count(), jax.process_index())

        episodes = episodes.with_options(_tf_data_options(deterministic))

        # ── Convert RLDS episodes → trajectory dicts ─────────────────────────
        dataset = episodes.flat_map(_episode_to_traj)

        # ── Trajectory-level filters ────────────────────────────────────────
        if filter_success:
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0],
                    r".*success.*",
                )
            )

        # Filter out trajectories shorter than one full clip
        dataset = dataset.filter(
            lambda traj: tf.greater_equal(
                tf.shape(traj["observation"][_EXTERIOR_IMAGE_KEYS[0]])[0],
                clip_length,
            )
        )

        # ── Train / val split (hash on trajectory ID) ───────────────────────
        def _traj_id(traj) -> tf.Tensor:
            return traj["traj_metadata"]["episode_metadata"]["file_path"][0]

        def _split_filter(traj):
            key = tf.strings.join([tf.strings.as_string(seed), _traj_id(traj)])
            bucket = tf.strings.to_hash_bucket_fast(key, 1000)
            threshold = tf.cast(int(val_fraction * 1000), tf.int64)
            is_val = bucket < threshold
            return is_val if want_val else tf.logical_not(is_val)

        dataset = dataset.filter(_split_filter)

        # ── Per-trajectory image / instruction selection ─────────────────────
        dataset = dataset.map(self._select_camera_and_instruction, num_parallel_calls=num_parallel_calls)

        # ── Windowing: trajectory → clips ────────────────────────────────────
        dataset = dataset.flat_map(self._traj_to_clips)

        # ── Per-clip image decoding & resizing ───────────────────────────────
        dataset = dataset.map(self._decode_clip, num_parallel_calls=num_parallel_calls)

        # ── Batching & prefetch ──────────────────────────────────────────────
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer, seed=seed)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.dataset = dataset

    # ── Trajectory-level transform ───────────────────────────────────────────

    def _select_camera_and_instruction(self, traj: dict) -> dict:
        """Pick one exterior camera and one language instruction per trajectory."""
        traj_len = tf.shape(traj["observation"][_EXTERIOR_IMAGE_KEYS[0]])[0]

        # Deterministic-per-trajectory random selection using file path as seed.
        path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        path_hash = tf.strings.to_hash_bucket_fast(path, 2147483647)
        seed_pair = [tf.cast(self.seed, tf.int64), tf.cast(path_hash, tf.int64)]

        # Select one of two exterior cameras.
        cam_idx = tf.cast(
            tf.random.stateless_uniform([], seed=seed_pair, minval=0, maxval=2, dtype=tf.int32),
            tf.int32,
        )
        images = tf.cond(
            tf.equal(cam_idx, 0),
            lambda: traj["observation"][_EXTERIOR_IMAGE_KEYS[0]],
            lambda: traj["observation"][_EXTERIOR_IMAGE_KEYS[1]],
        )  # [T] of JPEG-encoded bytes

        # Select one of three language instructions.
        lang_candidates = tf.stack([traj[k] for k in _LANGUAGE_KEYS], axis=0)  # [3, T]
        lang_idx = tf.cast(
            tf.random.stateless_uniform([], seed=[seed_pair[0] + 1, seed_pair[1]], minval=0, maxval=3, dtype=tf.int32),
            tf.int32,
        )
        instructions = lang_candidates[lang_idx]  # [T]

        return {
            "images": images,  # [T] bytes
            "instructions": instructions,  # [T] string
            "traj_len": traj_len,
        }

    # ── Windowing ────────────────────────────────────────────────────────────

    def _traj_to_clips(self, traj: dict) -> tf.data.Dataset:
        """Convert one trajectory dict into a dataset of fixed-length clips."""
        images = traj["images"]  # [T] bytes
        instructions = traj["instructions"]  # [T] string
        traj_len = tf.shape(images)[0]

        num_clips = (traj_len - self.clip_length) // self.stride + 1
        start_indices = tf.range(num_clips) * self.stride  # [num_clips]

        clip_length = self.clip_length  # capture for lambda

        def extract_clip(start: tf.Tensor) -> dict:
            frames = images[start : start + clip_length]  # [clip_length, H, W, C] uint8
            instruction = instructions[start]
            return {"frames": frames, "language_instruction": instruction}

        return tf.data.Dataset.from_tensor_slices(start_indices).map(
            extract_clip, num_parallel_calls=self._num_parallel_calls
        )

    # ── Per-clip decoding ────────────────────────────────────────────────────

    def _decode_clip(self, sample: dict) -> dict:
        """Cast uint8 pixel frames to float32, resize to (height, width)."""
        height, width = self.height, self.width
        frames = tf.cast(sample["frames"], tf.float32) / 255.0  # [clip_length, H, W, 3]
        frames = tf.image.resize(frames, [height, width], method="bilinear")
        return {
            "frames": frames,
            "language_instruction": sample["language_instruction"],
        }

    # ── Iterator protocol ────────────────────────────────────────────────────

    def __iter__(self):
        return self.dataset.as_numpy_iterator()
