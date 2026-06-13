"""Convert Wan2.2 cached DROID latent windows to side-adapter TFRecords.

Input cache window directory:
  <cache_root>/<name>/z_I0.pt
  <cache_root>/<name>/z_video.pt
  <cache_root>/<name>/actions.npy
  <cache_root>/<name>/meta.json

Output TFRecord fields:
  z_i0    raw float16 bytes [48, 1, 12, 20]
  z_video raw float16 bytes [48, 9, 12, 20]
  actions raw float32 bytes [32, 7]

The script is deliberately conservative: it estimates output size before
writing, refuses to exceed --max-output-gb, and can check local free space when
writing to /lustre temporary locations.

It intentionally does not import TensorFlow. Della's existing Wan2.2 venv has
Torch/NumPy but not TensorFlow, and installing TensorFlow on a nearly-full GPFS
just to write TFRecords is unnecessary. The writer below emits standard
TFRecord framing around standard tf.train.Example protobuf bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


EXPECTED_Z_I0_SHAPE = (48, 1, 12, 20)
EXPECTED_Z_VIDEO_SHAPE = (48, 9, 12, 20)
EXPECTED_ACTIONS_SHAPE = (32, 7)


_CRC32C_TABLE: list[int] | None = None


def _crc32c_table() -> list[int]:
    global _CRC32C_TABLE
    if _CRC32C_TABLE is None:
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0x82F63B78
                else:
                    crc >>= 1
            table.append(crc & 0xFFFFFFFF)
        _CRC32C_TABLE = table
    return _CRC32C_TABLE


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    table = _crc32c_table()
    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def _masked_crc32c(data: bytes) -> int:
    crc = _crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _field_key(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _len_field(field_number: int, payload: bytes) -> bytes:
    return _field_key(field_number, 2) + _varint(len(payload)) + payload


def _varint_field(field_number: int, value: int) -> bytes:
    return _field_key(field_number, 0) + _varint(value)


def _bytes_feature(value: bytes) -> bytes:
    bytes_list = _len_field(1, value)
    return _len_field(1, bytes_list)


def _int64_feature(value: int) -> bytes:
    int64_list = _varint_field(1, int(value))
    return _len_field(3, int64_list)


def _example_proto(features: dict[str, bytes]) -> bytes:
    feature_entries = []
    for name, feature in sorted(features.items()):
        entry = _len_field(1, name.encode("utf-8")) + _len_field(2, feature)
        feature_entries.append(_len_field(1, entry))
    features_msg = b"".join(feature_entries)
    return _len_field(1, features_msg)


class _TFRecordWriter:
    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "wb")

    def write(self, data: bytes):
        length = struct.pack("<Q", len(data))
        self._f.write(length)
        self._f.write(struct.pack("<I", _masked_crc32c(length)))
        self._f.write(data)
        self._f.write(struct.pack("<I", _masked_crc32c(data)))

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


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
        "z_i0": _bytes_feature(np.ascontiguousarray(z_i0).tobytes()),
        "z_video": _bytes_feature(np.ascontiguousarray(z_video).tobytes()),
        "actions": _bytes_feature(np.ascontiguousarray(actions).tobytes()),
        "meta_json": _bytes_feature(meta_bytes),
    }
    return _example_proto(features)


def _iter_manifest_names(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            yield item["name"]


def _valid_cache_dir(cache_root: Path, name: str) -> bool:
    p = cache_root / name
    return p.is_dir() and (p / "z_I0.pt").exists() and (p / "z_video.pt").exists() and (p / "actions.npy").exists()


def _list_cache_names(cache_root: Path, limit: int = 0, unsorted_listing: bool = False) -> list[str]:
    if unsorted_listing or limit > 0:
        names = []
        with os.scandir(cache_root) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                if _valid_cache_dir(cache_root, entry.name):
                    names.append(entry.name)
                    if limit > 0 and len(names) >= limit:
                        break
        return names
    return sorted(
        p.name
        for p in cache_root.iterdir()
        if p.is_dir() and (p / "z_I0.pt").exists() and (p / "z_video.pt").exists() and (p / "actions.npy").exists()
    )


def _select_names(args) -> list[str]:
    if args.manifest_jsonl:
        names = []
        for idx, name in enumerate(_iter_manifest_names(Path(args.manifest_jsonl))):
            if idx < args.start_index:
                continue
            if args.end_index is not None and idx >= args.end_index:
                break
            names.append(name)
            if args.max_examples > 0 and len(names) >= args.max_examples:
                break
    else:
        listing_limit = 0
        can_limit_listing = args.start_index == 0 and args.end_index is None and args.max_examples > 0
        if can_limit_listing:
            listing_limit = args.max_examples
        names = _list_cache_names(Path(args.cache_root), limit=listing_limit, unsorted_listing=args.unsorted_listing)
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


def _local_write_target(args) -> str:
    if args.local_staging_dir:
        return args.local_staging_dir
    if not args.output_dir.startswith("gs://"):
        return args.output_dir
    return ""


def _check_local_storage(args, estimated_output_bytes: int, bytes_per_sample: int, selected_count: int):
    target = _local_write_target(args)
    if not target or not args.local_storage_root:
        return
    root = Path(args.local_storage_root)
    if not root.exists():
        raise FileNotFoundError(f"--local-storage-root does not exist: {root}")
    output_path = Path(target).resolve()
    root_path = root.resolve()
    try:
        output_path.relative_to(root_path)
    except ValueError:
        return
    required_write_bytes = estimated_output_bytes
    if args.output_dir.startswith("gs://") and args.delete_local_after_upload:
        required_write_bytes = bytes_per_sample * min(args.shard_size, selected_count)
    usage = shutil.disk_usage(root_path)
    free_after = usage.free - required_write_bytes
    required = int(args.min_free_tb * (1024**4))
    if free_after < required:
        raise RuntimeError(
            f"Refusing local write under {root_path}: estimated output "
            f"{required_write_bytes / 1e9:.2f} GB would leave "
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


def _exists(path: str) -> bool:
    if not path.startswith("gs://"):
        return os.path.exists(path)
    return subprocess.run(["gsutil", "-q", "stat", path], check=False).returncode == 0


def _makedirs(path: str):
    if path and not path.startswith("gs://"):
        os.makedirs(path, exist_ok=True)


def _write_json(path: str, value: dict):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.startswith("gs://"):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            f.write(payload)
            tmp = f.name
        try:
            subprocess.run(["gsutil", "cp", tmp, path], check=True)
        finally:
            os.remove(tmp)
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)


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
    _check_local_storage(args, estimated_output_bytes, bytes_per_sample, len(names))

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
    if output_dir.startswith("gs://") and not args.local_staging_dir:
        raise ValueError(
            "Pure-Python TFRecord writing requires --local-staging-dir for gs:// output. "
            "Use --delete-local-after-upload to keep only one shard staged locally."
        )
    writer_output_dir = args.local_staging_dir or output_dir
    _makedirs(writer_output_dir)

    num_shards = int(math.ceil(len(names) / args.shard_size))
    started = time.time()
    for shard_idx in range(num_shards):
        shard_names = names[shard_idx * args.shard_size : (shard_idx + 1) * args.shard_size]
        shard_name = f"{args.shard_prefix}-{args.shard_offset + shard_idx:05d}-of-{args.shard_offset + num_shards:05d}.tfrecord"
        shard_path = writer_output_dir.rstrip("/") + "/" + shard_name
        final_path = output_dir.rstrip("/") + "/" + shard_name
        if args.skip_existing and _exists(final_path):
            print(f"[skip] {final_path}")
            summary["written_shards"].append({"path": final_path, "examples": 0, "skipped": True})
            continue
        written = 0
        with _TFRecordWriter(shard_path) as writer:
            for local_idx, name in enumerate(shard_names):
                ordinal = args.start_index + shard_idx * args.shard_size + local_idx
                try:
                    writer.write(_make_example(cache_root, name, ordinal))
                    written += 1
                except Exception as exc:  # noqa: BLE001
                    if args.fail_fast:
                        raise
                    summary["failures"].append({"name": name, "error": repr(exc)})
        if output_dir.startswith("gs://"):
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
    parser.add_argument(
        "--unsorted-listing",
        action="store_true",
        help="Do not sort cache directories when no manifest is supplied. Useful for smoke tests and full one-pass conversion.",
    )
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = convert(args)
    if args.summary_path:
        _write_json(args.summary_path, summary)


if __name__ == "__main__":
    main()
