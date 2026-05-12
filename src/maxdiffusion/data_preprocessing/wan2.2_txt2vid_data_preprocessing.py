"""Convert DROID Ctrl-World episodes to TFRecord shards for WAN Ctrl-World training.

Source layout:
  data_root/annotation/{train_split}/<episode_id>.json   — metadata, states, actions
  data_root/videos/{train_split}/<episode_id>/<cam>.mp4  — raw video (cameras 0/1/2)

Output layout:
  tfrecords_dir/{train_split}/shard-NNNN.tfrecord

Each TFRecord example stores one full episode as a latent sequence:
  latent_cam0: float16  (F_lat, C, H_lat, W_lat)  — WAN VAE latent sequence, camera 0
  latent_cam1: float16  (F_lat, C, H_lat, W_lat)  — WAN VAE latent sequence, camera 1
  latent_cam2: float16  (F_lat, C, H_lat, W_lat)  — WAN VAE latent sequence, camera 2
  action:      float32  (T_ep, 7)                  — raw (unnormalised) cartesian+gripper
  text_embed:  float16  (512, 4096)                — T5 text tokens
  episode_id:  int64
  traj_len:    int64                                — F_lat (number of latent frames)

Latents are time-first (F_lat on axis 0) for easy windowing at read time.
Actions are stored raw; normalisation is applied at read time by the dataset.

Usage:
  python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
    data_root=/n/fs/iromdata/droid_ctrl_world \
    tfrecords_dir=droid_wan_tfrecords_test \
    train_split=train \
    no_records_per_shard=50 \
    action_stats_path=action_stats_test.json \
    max_frames=300 \
    max_episodes=200
"""

from __future__ import annotations

import functools
import json
import os
import sys
from typing import Sequence

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))

from maxdiffusion import pyconfig
from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2

NUM_CAMERAS = 3
T5_SEQ_LEN = 512
ACTION_DIM = 7  # cartesian (6) + gripper (1)


# ── TFRecord helpers ──────────────────────────────────────────────────────────


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _serialize_example(
    latents_per_cam: list[np.ndarray],  # 3 × (F_lat, C, H_lat, W_lat) float16
    action: np.ndarray,                 # (T_ep, 7) float32  — raw
    text_embed: np.ndarray,             # (512, 4096) float16
    episode_id: int,
    traj_len: int,
) -> bytes:
    feature = {
        "latent_cam0": _bytes_feature(tf.io.serialize_tensor(tf.constant(latents_per_cam[0])).numpy()),
        "latent_cam1": _bytes_feature(tf.io.serialize_tensor(tf.constant(latents_per_cam[1])).numpy()),
        "latent_cam2": _bytes_feature(tf.io.serialize_tensor(tf.constant(latents_per_cam[2])).numpy()),
        "action":      _bytes_feature(tf.io.serialize_tensor(tf.constant(action)).numpy()),
        "text_embed":  _bytes_feature(tf.io.serialize_tensor(tf.constant(text_embed)).numpy()),
        "episode_id":  _int64_feature(episode_id),
        "traj_len":    _int64_feature(traj_len),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


# ── Action stats ──────────────────────────────────────────────────────────────


def compute_action_stats(annotation_dir: str, max_episodes: int = -1) -> tuple[np.ndarray, np.ndarray]:
    print("Computing action stats …", flush=True)
    all_states: list[np.ndarray] = []
    json_files = sorted(f for f in os.listdir(annotation_dir) if f.endswith(".json"))
    if max_episodes > 0:
        json_files = json_files[:max_episodes]
    for fname in json_files:
        with open(os.path.join(annotation_dir, fname)) as f:
            meta = json.load(f)
        all_states.append(np.array(meta["states"], dtype=np.float32))
    all_np = np.concatenate(all_states, axis=0)
    p01 = np.percentile(all_np, 1, axis=0).astype(np.float32)
    p99 = np.percentile(all_np, 99, axis=0).astype(np.float32)
    print(f"  p01 = {p01.tolist()}\n  p99 = {p99.tolist()}", flush=True)
    return p01, p99


# ── Video loading ─────────────────────────────────────────────────────────────


def load_video_frames(video_path: str, height: int, width: int) -> np.ndarray | None:
    """Load all frames, resize to (height, width). Returns (T, H, W, 3) float32 in [-1,1]."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    return np.stack(frames).astype(np.float32) / 127.5 - 1.0


# ── VAE encoding ──────────────────────────────────────────────────────────────


def vae_encode(video: jax.Array, rng: jax.Array, vae, vae_cache) -> jax.Array:
    """Pure JAX encode — JIT-compiled via functools.partial.

    Args:
        video: (N, T, H, W, 3) float32 in [-1, 1].

    Returns:
        (N, C, F_lat, H_lat, W_lat) float32, channel-first.
    """
    output = vae.encode(video, feat_cache=vae_cache)
    latent = output.latent_dist.sample(rng)           # (N, F_lat, H_lat, W_lat, C)
    latent = jnp.transpose(latent, (0, 4, 1, 2, 3))  # (N, C, F_lat, H_lat, W_lat)
    lat_mean = jnp.array(vae.latents_mean).reshape(1, vae.z_dim, 1, 1, 1)
    lat_std  = jnp.array(vae.latents_std).reshape(1, vae.z_dim, 1, 1, 1)
    return (latent - lat_mean) / lat_std


def encode_episode(
    frames_per_cam: list[np.ndarray],  # 3 × (T_ep, H, W, 3) float32
    p_vae_encode,
) -> list[np.ndarray]:
    """Encode a full episode for all cameras, returning time-first float16 latents."""
    x = jnp.array(np.stack(frames_per_cam, axis=0))  # (3, T_ep, H, W, 3)
    latent = p_vae_encode(video=x, rng=jax.random.key(0))  # (3, C, F_lat, H_lat, W_lat)
    latent = jnp.transpose(latent, (0, 2, 1, 3, 4))        # (3, F_lat, C, H_lat, W_lat)
    arr = np.array(latent, dtype=np.float16)
    return [arr[i] for i in range(NUM_CAMERAS)]


# ── Main ──────────────────────────────────────────────────────────────────────


def run(config) -> None:
    split    = config.train_split
    ann_dir  = os.path.join(config.data_root, "annotation", split)
    out_dir  = os.path.join(config.tfrecords_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    # ── Action stats ──────────────────────────────────────────────────────────
    stats_path = config.action_stats_path
    if stats_path and os.path.exists(stats_path):
        print(f"Loaded action stats from {stats_path}", flush=True)
    else:
        train_ann_dir = os.path.join(config.data_root, "annotation", "train")
        p01, p99 = compute_action_stats(train_ann_dir, max_episodes=config.max_episodes)
        stats_out = stats_path or os.path.join(config.tfrecords_dir, "action_stats.json")
        os.makedirs(os.path.dirname(os.path.abspath(stats_out)), exist_ok=True)
        with open(stats_out, "w") as f:
            json.dump({"state_01": p01.tolist(), "state_99": p99.tolist()}, f, indent=2)
        print(f"Saved action stats → {stats_out}", flush=True)

    # ── Load pipeline (VAE + text encoder, no transformer) ────────────────────
    print("Loading pipeline …", flush=True)
    pipeline = WanPipelineTI2V_2_2.from_pretrained(config, load_transformer=False)
    p_vae_encode = functools.partial(vae_encode, vae=pipeline.vae, vae_cache=pipeline.vae_cache)

    # ── Collect episode list ──────────────────────────────────────────────────
    json_files = sorted(f for f in os.listdir(ann_dir) if f.endswith(".json"))
    if config.max_episodes > 0:
        json_files = json_files[: config.max_episodes]
    print(f"Processing {len(json_files)} episodes ({split} split) …", flush=True)

    # ── Write TFRecords ───────────────────────────────────────────────────────
    shard_idx        = 0
    shard_count      = 0
    total_episodes   = 0
    skipped_episodes = 0

    writer = tf.io.TFRecordWriter(os.path.join(out_dir, f"shard-{shard_idx:04d}.tfrecord"))

    for ep_idx, fname in enumerate(json_files):
        ep_id = int(os.path.splitext(fname)[0])

        with open(os.path.join(ann_dir, fname)) as f:
            meta = json.load(f)

        text   = (meta["texts"] or [""])[0]
        states = np.array(meta["states"], dtype=np.float32)  # (T_ep, 7) raw

        cam_list = meta.get("videos", [])
        if len(cam_list) < NUM_CAMERAS:
            skipped_episodes += 1
            continue

        video_paths = [
            os.path.join(config.data_root, cam_list[i].get("video_path", ""))
            for i in range(NUM_CAMERAS)
        ]
        if not all(os.path.exists(p) for p in video_paths):
            skipped_episodes += 1
            continue

        # ── Load frames ───────────────────────────────────────────────────────
        all_cam_frames = [load_video_frames(p, config.height, config.width) for p in video_paths]
        if any(f is None for f in all_cam_frames):
            skipped_episodes += 1
            continue

        T_vid = min(len(f) for f in all_cam_frames)
        T_ep  = min(len(states), T_vid)
        if config.max_frames > 0:
            T_ep = min(T_ep, config.max_frames)
        if T_ep < 1:
            skipped_episodes += 1
            continue

        frames_per_cam = [f[:T_ep] for f in all_cam_frames]
        raw_action     = states[:T_ep]  # (T_ep, 7)

        # ── Encode text ───────────────────────────────────────────────────────
        text_embed_pt = pipeline._get_t5_prompt_embeds(text, max_sequence_length=T5_SEQ_LEN)
        text_embed = text_embed_pt[0].detach().float().numpy().astype(np.float16)  # (512, 4096)

        # ── Encode VAE (full episode, all cameras) ────────────────────────────
        latents_per_cam = encode_episode(frames_per_cam, p_vae_encode)
        # 3 × (F_lat, C, H_lat, W_lat) float16, time-first
        F_lat = latents_per_cam[0].shape[0]

        writer.write(_serialize_example(latents_per_cam, raw_action, text_embed, ep_id, F_lat))
        shard_count    += 1
        total_episodes += 1

        if shard_count >= config.no_records_per_shard:
            writer.close()
            shard_idx  += 1
            shard_count = 0
            writer = tf.io.TFRecordWriter(os.path.join(out_dir, f"shard-{shard_idx:04d}.tfrecord"))

        if (ep_idx + 1) % 100 == 0:
            print(
                f"  [{ep_idx + 1}/{len(json_files)}] {total_episodes} written, "
                f"{skipped_episodes} skipped",
                flush=True,
            )

    writer.close()
    print(
        f"\nDone. {total_episodes} episodes → {shard_idx + 1} shards in {out_dir}\n"
        f"Skipped {skipped_episodes} episodes.",
        flush=True,
    )


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)


if __name__ == "__main__":
    app.run(main)
