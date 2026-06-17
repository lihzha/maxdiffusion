#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}}"
HF_PREFETCH_ATTEMPTS="${HF_PREFETCH_ATTEMPTS:-${HF_PREFETCH_MAX_ATTEMPTS:-6}}"
HF_PREFETCH_SLEEP_SECS="${HF_PREFETCH_SLEEP_SECS:-${HF_PREFETCH_RETRY_SLEEP_SECS:-30}}"
HF_PREFETCH_WORKERS="${HF_PREFETCH_WORKERS:-2}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "maxdiffusion_venv/bin/activate" ]; then
  source maxdiffusion_venv/bin/activate
fi

if [ -d "$MODEL_DIR" ]; then
  echo "[prefetch_hf_snapshot] local model directory exists: $MODEL_DIR"
  exit 0
fi

case "$MODEL_DIR" in
  gs://*)
    echo "[prefetch_hf_snapshot] skipping non-Hugging Face model path: $MODEL_DIR"
    exit 0
    ;;
esac

echo "HF prefetch MODEL_DIR=${MODEL_DIR}"
echo "HF_PREFETCH_ATTEMPTS=${HF_PREFETCH_ATTEMPTS}"
echo "HF_PREFETCH_SLEEP_SECS=${HF_PREFETCH_SLEEP_SECS}"
echo "HF_PREFETCH_WORKERS=${HF_PREFETCH_WORKERS}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"

python - "$MODEL_DIR" "$HF_PREFETCH_ATTEMPTS" "$HF_PREFETCH_SLEEP_SECS" "$HF_PREFETCH_WORKERS" <<'PY'
import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download


repo_id = sys.argv[1]
max_attempts = int(sys.argv[2])
retry_sleep_secs = int(sys.argv[3])
max_workers = int(sys.argv[4])

allow_patterns = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "vae/*",
    "transformer/*",
]


def verify_snapshot(snapshot_path: str) -> None:
    root = Path(snapshot_path)
    required = [
        root / "model_index.json",
        root / "vae" / "config.json",
        root / "vae" / "diffusion_pytorch_model.safetensors",
        root / "transformer" / "config.json",
        root / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required snapshot files: {missing}")

    index_path = root / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    with index_path.open() as f:
        index = json.load(f)
    shard_names = sorted(set(index["weight_map"].values()))
    if not shard_names:
        raise ValueError("Transformer index contains no shard filenames")
    for shard_name in shard_names:
        shard = root / "transformer" / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"Missing transformer shard: transformer/{shard_name}")
        if shard.stat().st_size <= 0:
            raise OSError(f"Transformer shard is empty: transformer/{shard_name}")


last_error: Exception | None = None
success = False
for attempt in range(1, max_attempts + 1):
    try:
        print(f"[prefetch_hf_snapshot] attempt {attempt}/{max_attempts}: {repo_id}", flush=True)
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            max_workers=max_workers,
        )
        verify_snapshot(snapshot_path)
        print(f"[prefetch_hf_snapshot] verified snapshot: {snapshot_path}", flush=True)
        success = True
        break
    except Exception as exc:  # keep retrying transient hub/network failures
        last_error = exc
        print(f"[prefetch_hf_snapshot] attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
        if attempt < max_attempts:
            time.sleep(retry_sleep_secs)

if not success:
    raise SystemExit(f"[prefetch_hf_snapshot] failed after {max_attempts} attempts: {last_error!r}")
PY
