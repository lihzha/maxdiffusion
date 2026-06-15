#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}}"
HF_PREFETCH_ATTEMPTS="${HF_PREFETCH_ATTEMPTS:-6}"
HF_PREFETCH_SLEEP_SECS="${HF_PREFETCH_SLEEP_SECS:-30}"
HF_PREFETCH_WORKERS="${HF_PREFETCH_WORKERS:-2}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

if [ -d "$MODEL_DIR" ]; then
  echo "HF prefetch skipped for local MODEL_DIR=${MODEL_DIR}"
  exit 0
fi

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "maxdiffusion_venv/bin/activate" ]; then
  source maxdiffusion_venv/bin/activate
fi

echo "HF prefetch MODEL_DIR=${MODEL_DIR}"
echo "HF_PREFETCH_ATTEMPTS=${HF_PREFETCH_ATTEMPTS}"
echo "HF_PREFETCH_WORKERS=${HF_PREFETCH_WORKERS}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"

attempt=1
while true; do
  echo "HF prefetch attempt ${attempt}/${HF_PREFETCH_ATTEMPTS}"
  if python - "$MODEL_DIR" "$HF_PREFETCH_WORKERS" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
max_workers = int(sys.argv[2])
snapshot_download(repo_id=model_id, max_workers=max_workers)
print(f"HF prefetch complete: {model_id}")
PY
  then
    exit 0
  fi

  if (( attempt >= HF_PREFETCH_ATTEMPTS )); then
    echo "HF prefetch failed after ${HF_PREFETCH_ATTEMPTS} attempts" >&2
    exit 1
  fi

  sleep_for=$((HF_PREFETCH_SLEEP_SECS * attempt))
  echo "HF prefetch retrying in ${sleep_for}s"
  sleep "$sleep_for"
  attempt=$((attempt + 1))
done
