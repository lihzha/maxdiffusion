"""Convert DROID Ctrl-World episodes to TFRecord shards for WAN Ctrl-World training.

Source layout (all from raw_data_root):
  raw_data_root/{LAB}/{outcome}/{date}/{timestamp}/
    metadata_<uuid>.json                    — camera serial → uuid mapping
    trajectory.h5                           — robot states / actions
    recordings/MP4/<cam_serial>.mp4         — raw video (wrist, ext1, ext2)
  raw_data_root/aggregated-annotations-*.json — language instructions keyed by uuid

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

Episode ordering
----------------
The script first builds (or loads) a stable id→video-path index from raw_data_root.
Episodes are sorted by UUID so that every run — regardless of start_episode — sees
the same canonical ordering.  The index is saved to video_index_path and reused on
subsequent calls, which guarantees that start_episode=0..N-1 and start_episode=N..
are always disjoint and cover the same episodes.

Usage:
  python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
    raw_data_root=/n/fs/iromdata/droid_raw/1.0.1 \
    data_root=droid_wan_tfrecords_test \
    video_index_path=droid_wan_tfrecords_test/video_index.json \
    no_records_per_shard=50 \
    action_stats_path=action_stats_test.json \
    val_fraction=0.05 \
    max_frames=300 \
    max_episodes=200 \
    start_episode=0
"""

from __future__ import annotations

import functools
import json
import os
import queue
import random
import sys
import threading
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import cv2
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # TF is CPU-only (TFRecord I/O); leave GPU for JAX
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


# ── Raw DROID helpers ─────────────────────────────────────────────────────────


def load_aggregated_annotations(raw_root: str) -> dict[str, str]:
    """Load aggregated-annotations-*.json, returning {uuid: text}."""
    matches = [
        f for f in os.listdir(raw_root)
        if f.startswith("aggregated-annotations") and f.endswith(".json")
    ]
    if not matches:
        return {}
    with open(os.path.join(raw_root, sorted(matches)[-1])) as fh:
        data = json.load(fh)
    return {uuid: entry.get("language_instruction1", "") for uuid, entry in data.items()}


def load_states_from_h5(h5_path: str) -> np.ndarray | None:
    """Read (T, 7) float32 states from a raw DROID trajectory H5 file."""
    try:
        with h5py.File(h5_path, "r") as f:
            cart  = f["observation/robot_state/cartesian_position"][:]   # (T, 6)
            grip  = f["observation/robot_state/gripper_position"][:]     # (T,)
        return np.concatenate([cart, grip[:, None]], axis=1).astype(np.float32)
    except Exception:
        return None


# ── Action stats ──────────────────────────────────────────────────────────────


def compute_action_stats(episode_index: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    print("Computing action stats …", flush=True)
    h5_paths = [e["h5_path"] for e in episode_index]
    nworkers = min(16, len(h5_paths))
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(nworkers) as pool:
        results = []
        for i, r in enumerate(pool.imap(load_states_from_h5, h5_paths), 1):
            results.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(h5_paths)} files read", flush=True)
    all_states = [s for s in results if s is not None]
    all_np = np.concatenate(all_states, axis=0)
    p01 = np.percentile(all_np, 1, axis=0).astype(np.float32)
    p99 = np.percentile(all_np, 99, axis=0).astype(np.float32)
    print(f"  p01 = {p01.tolist()}\n  p99 = {p99.tolist()}", flush=True)
    return p01, p99


# ── Episode index ─────────────────────────────────────────────────────────────
def build_video_index(raw_root: str, index_path: str) -> list[dict]:
    """Walk raw_root to build a stable id→video-path index sorted by UUID.

    Each raw DROID episode lives at:
      raw_root/{LAB}/{outcome}/{date}/{timestamp}/
        metadata_{uuid}.json          — has wrist/ext1/ext2 camera serials
        recordings/MP4/{serial}.mp4   — one file per camera

    Returns a list of dicts (sorted by uuid, integer id = list index):
      {"id": int, "uuid": str, "h5_path": str, "videos": [wrist_path, ext1_path, ext2_path]}

    Saves the list to index_path as JSON.
    """
    entries: list[dict] = []
    skipped = 0

    for lab_entry in sorted(os.scandir(raw_root), key=lambda e: e.name):
        if not lab_entry.is_dir():
            continue
        for outcome_entry in sorted(os.scandir(lab_entry.path), key=lambda e: e.name):
            if not outcome_entry.is_dir():
                continue
            for date_entry in sorted(os.scandir(outcome_entry.path), key=lambda e: e.name):
                if not date_entry.is_dir():
                    continue
                for ts_entry in sorted(os.scandir(date_entry.path), key=lambda e: e.name):
                    if not ts_entry.is_dir():
                        continue
                    p = ts_entry.path

                    mp4_dir = os.path.join(p, "recordings", "MP4")
                    if not os.path.isdir(mp4_dir):
                        print(f"  skip {p}: no recordings/MP4 dir", flush=True)
                        skipped += 1
                        continue

                    # Find the metadata JSON to get camera serials and UUID.
                    meta_files = [
                        f for f in os.listdir(p)
                        if f.startswith("metadata_") and f.endswith(".json")
                    ]
                    if not meta_files:
                        print(f"  skip {p}: no metadata_*.json", flush=True)
                        skipped += 1
                        continue

                    try:
                        with open(os.path.join(p, meta_files[0]), encoding="utf-8", errors="replace") as fh:
                            meta = json.load(fh)
                    except Exception as exc:
                        print(f"  skip {p}: failed to parse {meta_files[0]}: {exc}", flush=True)
                        skipped += 1
                        continue

                    uuid = meta.get(
                        "uuid",
                        meta_files[0].removeprefix("metadata_").removesuffix(".json"),
                    )
                    wrist = meta.get("wrist_cam_serial", "")
                    ext1  = meta.get("ext1_cam_serial", "")
                    ext2  = meta.get("ext2_cam_serial", "")

                    video_paths = [
                        os.path.join(mp4_dir, f"{serial}.mp4")
                        for serial in (wrist, ext1, ext2)
                    ]
                    missing = [vp for vp in video_paths if not os.path.exists(vp)]
                    if missing:
                        print(f"  skip {p}: missing video(s): {missing}", flush=True)
                        skipped += 1
                        continue

                    h5_path = os.path.join(p, "trajectory.h5")
                    if not os.path.exists(h5_path):
                        print(f"  skip {p}: no trajectory.h5", flush=True)
                        skipped += 1
                        continue

                    entries.append({"uuid": uuid, "h5_path": h5_path, "videos": video_paths})

    # Sort by UUID first for a deterministic base order, then shuffle so that
    # any contiguous slice (train prefix / val suffix) is a random split.
    entries.sort(key=lambda e: e["uuid"])
    random.Random(42).shuffle(entries)
    for i, e in enumerate(entries):
        e["id"] = i

    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    with open(index_path, "w") as fh:
        json.dump(entries, fh)

    print(
        f"Built video index: {len(entries)} episodes → {index_path}"
        + (f" ({skipped} dirs skipped)" if skipped else ""),
        flush=True,
    )
    return entries


def load_video_index(index_path: str) -> list[dict]:
    with open(index_path) as fh:
        entries = json.load(fh)
    print(f"Loaded video index: {len(entries)} episodes from {index_path}", flush=True)
    return entries


# ── Video loading ─────────────────────────────────────────────────────────────


def load_video_frames(video_path: str, *, height: int, width: int, max_frames: int = 0) -> np.ndarray | None:
    """Load frames, resize to (height, width). Returns (T, H, W, 3) uint8.

    Stops early after max_frames if set, avoiding decoding unused tail frames.
    """
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
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        return None
    return np.stack(frames)  # uint8 (T, H, W, 3) — normalized to float32 at encode time


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
    frames_per_cam: list[np.ndarray],  # 3 × (T_ep, H, W, 3) uint8
    p_vae_encode,
) -> list[np.ndarray]:
    """Encode each camera separately to keep peak GPU memory at 1× episode size."""
    results = []
    for cam_frames in frames_per_cam:
        x = jnp.array(cam_frames.astype(np.float32) / 127.5 - 1.0)[None]  # (1, T_ep, H, W, 3)
        latent = p_vae_encode(video=x, rng=jax.random.key(0))              # (1, C, F_lat, H_lat, W_lat)
        latent = jnp.transpose(latent, (0, 2, 1, 3, 4))                    # (1, F_lat, C, H_lat, W_lat)
        results.append(np.array(latent[0], dtype=np.float16))
        del x, latent  # release GPU buffers before next camera
    return results


# ── Prefetch pipeline ─────────────────────────────────────────────────────────

_SENTINEL    = object()
_WORK_DONE   = object()


def _load_one_episode(index_entry, annotations, config, pipeline, t5_lock, t5_cache):
    """Load, decode, and T5-encode one episode.

    index_entry:  dict with 'id', 'uuid', 'h5_path', 'videos'
    annotations:  {uuid: text} from aggregated-annotations-*.json
    Returns (ep_id, text_embed, raw_action, frames_per_cam) or None.
    """
    ep_id       = index_entry["id"]
    video_paths = index_entry["videos"]

    states = load_states_from_h5(index_entry["h5_path"])
    if states is None:
        return None

    text = annotations.get(index_entry["uuid"], "")

    load_fn = functools.partial(load_video_frames, height=config.height, width=config.width, max_frames=config.max_frames)
    with ThreadPoolExecutor(max_workers=NUM_CAMERAS) as ex:
        all_cam_frames = list(ex.map(load_fn, video_paths))

    if any(f is None for f in all_cam_frames):
        return None

    T_vid = min(len(f) for f in all_cam_frames)
    T_ep  = min(len(states), T_vid)
    if config.max_frames > 0:
        T_ep = min(T_ep, config.max_frames)
    if T_ep < 1:
        return None

    with t5_lock:
        if text not in t5_cache:
            text_embed_pt = pipeline._get_t5_prompt_embeds(text, max_sequence_length=T5_SEQ_LEN)
            t5_cache[text] = text_embed_pt[0].detach().float().numpy().astype(np.float16)
    text_embed = t5_cache[text]  # (512, 4096)

    return (ep_id, text_embed, states[:T_ep], [f[:T_ep] for f in all_cam_frames])


def _loader_worker(work_queue, out_queue, annotations, config, pipeline, t5_lock, t5_cache):
    """Pull index entries from work_queue, load episodes, push results to out_queue."""
    while True:
        entry = work_queue.get()
        if entry is _WORK_DONE:
            break
        out_queue.put(_load_one_episode(entry, annotations, config, pipeline, t5_lock, t5_cache))


# ── Main ──────────────────────────────────────────────────────────────────────


def _write_split(
    split: str,
    index_slice: list[dict],
    out_dir: str,
    shard_start: int,
    p_vae_encode,
    annotations: dict,
    config,
    pipeline,
) -> None:
    """Encode and write one split (train or val) to TFRecord shards."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing {len(index_slice)} episodes ({split} split) …", flush=True)

    shard_idx        = shard_start
    shard_count      = 0
    total_episodes   = 0
    skipped_episodes = 0
    writer           = None

    n_workers      = config.prefetch_workers
    work_queue     = queue.Queue()
    prefetch_queue = queue.Queue(maxsize=2)
    t5_lock        = threading.Lock()
    t5_cache: dict[str, np.ndarray] = {}

    for entry in index_slice:
        work_queue.put(entry)
    for _ in range(n_workers):
        work_queue.put(_WORK_DONE)

    workers = [
        threading.Thread(
            target=_loader_worker,
            args=(work_queue, prefetch_queue, annotations, config, pipeline, t5_lock, t5_cache),
            daemon=True,
        )
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()

    def _coordinator():
        for w in workers:
            w.join()
        prefetch_queue.put(_SENTINEL)

    threading.Thread(target=_coordinator, daemon=True).start()

    ep_idx       = 0
    t_encode_sum = 0.0
    t_window     = 0
    while True:
        item = prefetch_queue.get()
        if item is _SENTINEL:
            break
        if item is None:
            skipped_episodes += 1
            ep_idx += 1
            continue

        ep_id, text_embed, raw_action, frames_per_cam = item

        t0 = time.perf_counter()
        latents_per_cam = encode_episode(frames_per_cam, p_vae_encode)
        jax.effects_barrier()
        t_encode_sum += time.perf_counter() - t0
        t_window     += 1

        F_lat = latents_per_cam[0].shape[0]

        if writer is None:
            writer = tf.io.TFRecordWriter(os.path.join(out_dir, f"shard-{shard_idx:04d}.tfrecord"))
        writer.write(_serialize_example(latents_per_cam, raw_action, text_embed, ep_id, F_lat))
        shard_count    += 1
        total_episodes += 1
        ep_idx         += 1

        if shard_count >= config.no_records_per_shard:
            writer.close()
            writer = None
            shard_idx  += 1
            shard_count = 0

        if ep_idx % 100 == 0:
            avg_s = t_encode_sum / t_window if t_window else 0.0
            print(
                f"  [{ep_idx}/{len(index_slice)}] {total_episodes} written, "
                f"{skipped_episodes} skipped | "
                f"vae encode {avg_s:.2f}s/traj (last {t_window})",
                flush=True,
            )
            t_encode_sum = 0.0
            t_window     = 0

    if writer is not None:
        writer.close()
    print(
        f"Done ({split}). {total_episodes} episodes → {shard_idx - shard_start + (1 if total_episodes else 0)} shards"
        f" in {out_dir}  (skipped {skipped_episodes})",
        flush=True,
    )


def run(config) -> None:
    # ── Build or load stable episode index (needed for stats too) ────────────
    index_path = config.video_index_path
    if os.path.exists(index_path):
        episode_index = load_video_index(index_path)
    else:
        episode_index = build_video_index(config.raw_data_root, index_path)

    # ── Language annotations ──────────────────────────────────────────────────
    annotations = load_aggregated_annotations(config.raw_data_root)
    print(f"Loaded {len(annotations)} language annotations", flush=True)

    # ── Slice index into train and val ────────────────────────────────────────
    # Snap to shard boundary so shard-N always holds the same episode range.
    sps   = config.no_records_per_shard
    start = (max(config.start_episode, 0) // sps) * sps
    tail  = episode_index[start:]
    if config.max_episodes > 0:
        tail = tail[: config.max_episodes]

    # ── Global train/val split via split_metadata.json ───────────────────────
    # The file covers the full dataset so the split is identical across all
    # parallel jobs regardless of chunk boundaries.  Job 0 creates it
    # atomically; other jobs just read it.
    os.makedirs(config.data_root, exist_ok=True)
    meta_out = os.path.join(config.data_root, "split_metadata.json")
    if not os.path.exists(meta_out):
        val_frac  = max(0.0, min(1.0, config.val_fraction))
        n_val_g   = round(len(episode_index) * val_frac)
        n_train_g = len(episode_index) - n_val_g
        tmp = meta_out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "train": [{"id": e["id"], "uuid": e["uuid"], "h5_path": e["h5_path"], "videos": e["videos"]} for e in episode_index[:n_train_g]],
                    "val":   [{"id": e["id"], "uuid": e["uuid"], "h5_path": e["h5_path"], "videos": e["videos"]} for e in episode_index[n_train_g:]],
                },
                f, indent=2,
            )
        os.rename(tmp, meta_out)  # atomic: last writer wins but all produce same content
        print(f"Split metadata → {meta_out}  (train={n_train_g}, val={n_val_g})", flush=True)

    with open(meta_out) as f:
        meta = json.load(f)
    train_ids = {e["id"] for e in meta["train"]}
    n_train_global = len(train_ids)

    train_slice = [e for e in tail if e["id"] in train_ids]
    val_slice   = [e for e in tail if e["id"] not in train_ids]
    print(f"This job: {len(train_slice)} train, {len(val_slice)} val episodes", flush=True)

    # ── Action stats (full global train split) ───────────────────────────────
    stats_path = config.action_stats_path
    if stats_path and os.path.exists(stats_path):
        print(f"Loaded action stats from {stats_path}", flush=True)
    else:
        # Use all global train episodes so stats are not biased by chunk boundaries.
        global_train = [e for e in episode_index if e["id"] in train_ids]
        p01, p99 = compute_action_stats(global_train)
        stats_out = stats_path or os.path.join(config.data_root, "action_stats.json")
        os.makedirs(os.path.dirname(os.path.abspath(stats_out)), exist_ok=True)
        with open(stats_out, "w") as f:
            json.dump({"state_01": p01.tolist(), "state_99": p99.tolist()}, f, indent=2)
        print(f"Saved action stats → {stats_out}", flush=True)

    if config.setup_only or config.action_stats_only:
        return

    # ── Load pipeline (VAE + text encoder, no transformer) ────────────────────
    print("Loading pipeline …", flush=True)
    pipeline = WanPipelineTI2V_2_2.from_pretrained(config, load_transformer=False)
    p_vae_encode = functools.partial(vae_encode, vae=pipeline.vae, vae_cache=pipeline.vae_cache)

    # ── Write splits ──────────────────────────────────────────────────────────
    train_shard_start = start // config.no_records_per_shard
    # Val episodes are the tail of the global index (id >= n_train_global), so
    # val episodes written before this job = max(0, start - n_train_global).
    val_eps_before  = max(0, start - n_train_global)
    val_shard_start = val_eps_before // config.no_records_per_shard

    _write_split(
        "train", train_slice,
        os.path.join(config.data_root, "train"),
        train_shard_start,
        p_vae_encode, annotations, config, pipeline,
    )
    if val_slice:
        _write_split(
            "val", val_slice,
            os.path.join(config.data_root, "val"),
            val_shard_start,
            p_vae_encode, annotations, config, pipeline,
        )


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)
    os._exit(0)  # bypass JAX/XLA CUDA cleanup hang


if __name__ == "__main__":
    app.run(main)
