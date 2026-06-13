"""Convert Wan2.2 cached DROID latent windows to side-adapter TFRecords.

Input cache window directory:
  <cache_root>/<name>/z_I0.pt
  <cache_root>/<name>/z_video.pt
  <cache_root>/<name>/actions.npy
  <cache_root>/<name>/meta.json

Output TFRecord fields:
  z_i0    float16 serialized tensor [48, 1, 12, 20]
  z_video float16 serialized tensor [48, 9, 12, 20]
  actions float32 serialized tensor [32, 7]

The script is deliberately conservative: it estimates output size before
writing, refuses to exceed --max-output-gb, and can check local free space when
writing to /lustre temporary locations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
import torch


EXPECTED_Z_I0_SHAPE = (48, 1, 12, 20)
EXPECTED_Z_VIDEO_SHAPE = (48, 9, 12, 20)
EXPECTED_ACTIONS_SHAPE = (32, 7)


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _torch_load_tensor(path: Path) -> np.ndarray:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _load_window(cache_root: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, bytes]:
    sample_dir = cache_root / name
    z_i0 = _torch_load_tensor(sample_dir / "z_I0.pt")
    z_video = _torch_load_tensor(sample_dir / "z_video.pt")
    actions = np.load(sample_dir / "actions.npy")
    meta_path = sample_dir / "meta.json"
    meta_bytes = meta_path.read_bytes() if meta_path.exists() else b"{}"

    if z_i0.shape == (48, 9, 12, 20):
        z_i0 = z_i0[:, :1]
    if z_i0.shape != EXPECTED_Z_I0_SHAPE:
        raise ValueError(f"{name}: z_I0 shape {z_i0.shape} != {EXPECTED_Z_I0_SHAPE}")
    if z_video.shape != EXPECTED_Z_VIDEO_SHAPE:
        raise ValueError(f"{name}: z_video shape {z_video.shape} != {EXPECTED_Z_VIDEO_SHAPE}")
    if actions.shape != EXPECTED_ACTIONS_SHAPE:
        raise ValueError(f"{name}: actions shape {actions.shape} != {EXPECTED_ACTIONS_SHAPE}")

    return (
        z_i0.astype(np.float16, copy=False),
        z_video.astype(np.float16, copy=False),
        actions.astype(np.float32, copy=False),
        meta_bytes,
    )


def _make_example(cache_root: Path, name: str, ordinal: int) -> bytes:
    z_i0, z_video, actions, meta_bytes = _load_window(cache_root, name)
    features = {
        "name": _bytes_feature(name.encode("utf-8")),
        "ordinal": _int64_feature(ordinal),
        "z_i0": _bytes_feature(tf.io.serialize_tensor(tf.convert_to_tensor(z_i0)).numpy()),
        "z_video": _bytes_feature(tf.io.serialize_tensor(tf.convert_to_tensor(z_video)).numpy()),
        "actions": _bytes_feature(tf.io.serialize_tensor(tf.convert_to_tensor(actions)).numpy()),
        "meta_json": _bytes_feature(meta_bytes),
    }
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


def _iter_manifest_names(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield item["name"]


def _list_cache_names(cache_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in cache_root.iterdir()
        if p.is_dir() and (p / "z_I0.pt").exists() and (p / "z_video.pt").exists() and (p / "actions.npy").exists()
    )


def _select_names(args) -> list[str]:
    if args.manifest_jsonl:
        names = list(_iter_manifest_names(Path(args.manifest_jsonl)))
    else:
        names = _list_cache_names(Path(args.cache_root))
    if args.start_index:
        names = names[args.start_index :]
    if args.end_index is not None:
        names = names[: max(0, args.end_index - args.start_index)]
    if args.max_examples > 0:
        names = names[: args.max_examples]
    return names


def _estimate_bytes(cache_root: Path, names: list[str], estimate_samples: int) -> tuple[int, int]:
    if not names:
        return 0, 0
    n = min(len(names), max(1, estimate_samples))
    sizes = []
    for i in range(n):
        serialized = _make_example(cache_root, names[i], i)
        sizes.append(len(serialized))
    bytes_per_sample = int(math.ceil(float(np.mean(sizes))))
    return bytes_per_sample, bytes_per_sample * len(names)


def _check_local_storage(args, estimated_output_bytes: int):
    if args.output_dir.startswith("gs://") or not args.local_storage_root:
        return
    root = Path(args.local_storage_root)
    if not root.exists():
        raise FileNotFoundError(f"--local-storage-root does not exist: {root}")
    output_path = Path(args.output_dir).resolve()
    root_path = root.resolve()
    try:
        output_path.relative_to(root_path)
    except ValueError:
        return
    usage = shutil.disk_usage(root_path)
    free_after = usage.free - estimated_output_bytes
    required = int(args.min_free_tb * (1024**4))
    if free_after < required:
        raise RuntimeError(
            f"Refusing local write under {root_path}: estimated output "
            f"{estimated_output_bytes / 1e9:.2f} GB would leave "
            f"{free_after / (1024**4):.2f} TiB free, below required "
            f"{args.min_free_tb:.2f} TiB."
        )


def _maybe_upload_and_delete(local_path: str, output_dir: str, delete_local: bool):
    if not output_dir.startswith("gs://"):
        return local_path
    dest = output_dir.rstrip("/") + "/" + os.path.basename(local_path)
    subprocess.run(["gsutil", "cp", local_path, dest], check=True)
    if delete_local:
        os.remove(local_path)
    return dest


def convert(args) -> dict:
    cache_root = Path(args.cache_root)
    names = _select_names(args)
    if args.require_manifest and not args.manifest_jsonl:
        raise ValueError("--require-manifest was set but --manifest-jsonl is empty")
    if not names:
        raise ValueError("No cache window names selected")

    bytes_per_sample, estimated_output_bytes = _estimate_bytes(cache_root, names, args.estimate_samples)
    estimated_gb = estimated_output_bytes / 1e9
    if args.max_output_gb > 0 and estimated_gb > args.max_output_gb:
        raise RuntimeError(
            f"Refusing conversion: estimated output {estimated_gb:.2f} GB exceeds "
            f"--max-output-gb={args.max_output_gb:.2f}. Use a smaller manifest/chunk or raise the cap deliberately."
        )
    _check_local_storage(args, estimated_output_bytes)

    summary = {
        "cache_root": str(cache_root),
        "manifest_jsonl": args.manifest_jsonl,
        "output_dir": args.output_dir,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_examples": args.max_examples,
        "selected_examples": len(names),
        "bytes_per_sample_estimate": bytes_per_sample,
        "estimated_output_bytes": estimated_output_bytes,
        "estimated_output_gb": estimated_gb,
        "shard_size": args.shard_size,
        "dry_run": args.dry_run,
        "written_examples": 0,
        "written_shards": [],
        "failures": [],
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return summary

    output_dir = args.output_dir.rstrip("/")
    direct_gcs = output_dir.startswith("gs://") and not args.local_staging_dir
    writer_output_dir = output_dir if direct_gcs else args.local_staging_dir or output_dir
    tf.io.gfile.makedirs(writer_output_dir)
    if output_dir.startswith("gs://") and not direct_gcs:
        tf.io.gfile.makedirs(output_dir)

    num_shards = int(math.ceil(len(names) / args.shard_size))
    started = time.time()
    for shard_idx in range(num_shards):
        shard_names = names[shard_idx * args.shard_size : (shard_idx + 1) * args.shard_size]
        shard_name = f"{args.shard_prefix}-{args.shard_offset + shard_idx:05d}-of-{args.shard_offset + num_shards:05d}.tfrecord"
        shard_path = writer_output_dir.rstrip("/") + "/" + shard_name
        final_path = output_dir.rstrip("/") + "/" + shard_name
        if args.skip_existing and tf.io.gfile.exists(final_path):
            print(f"[skip] {final_path}")
            summary["written_shards"].append({"path": final_path, "examples": 0, "skipped": True})
            continue
        written = 0
        with tf.io.TFRecordWriter(shard_path) as writer:
            for local_idx, name in enumerate(shard_names):
                ordinal = args.start_index + shard_idx * args.shard_size + local_idx
                try:
                    writer.write(_make_example(cache_root, name, ordinal))
                    written += 1
                except Exception as exc:  # noqa: BLE001
                    if args.fail_fast:
                        raise
                    summary["failures"].append({"name": name, "error": repr(exc)})
        if output_dir.startswith("gs://") and not direct_gcs:
            final_path = _maybe_upload_and_delete(shard_path, output_dir, args.delete_local_after_upload)
        summary["written_examples"] += written
        summary["written_shards"].append({"path": final_path, "examples": written, "skipped": False})
        elapsed = max(1e-6, time.time() - started)
        print(
            f"[write] shard={shard_idx + 1}/{num_shards} examples={written} "
            f"total={summary['written_examples']} rate={summary['written_examples'] / elapsed:.2f}/s"
        )

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest-jsonl", default="")
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-staging-dir", default="")
    parser.add_argument("--delete-local-after-upload", action="store_true")
    parser.add_argument("--shard-prefix", default="shard")
    parser.add_argument("--shard-offset", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--estimate-samples", type=int, default=16)
    parser.add_argument("--max-output-gb", type=float, default=120.0)
    parser.add_argument("--local-storage-root", default="")
    parser.add_argument("--min-free-tb", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = convert(args)
    if args.summary_path:
        summary_path = args.summary_path
        parent = os.path.dirname(summary_path)
        if parent:
            tf.io.gfile.makedirs(parent)
        with tf.io.gfile.GFile(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
