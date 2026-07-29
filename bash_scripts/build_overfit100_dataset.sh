#!/usr/bin/env bash
set -euo pipefail

# --- exp_02 overfit100 dataset build (plan v4 §2 D4/D6, cycle B) ---
# Runs data_preprocessing/build_overfit100_dataset.py on a TPU host: manifest preflight (incl.
# the mandatory VAE pin) -> pinned MP4 download -> ffmpeg frames -> pipeline-parity Wan-VAE
# encode -> gates V1-V4 -> staged schema-v2 TFRecords -> physical readback -> promotion +
# _SUCCESS, for train100 + train10. PROBE=1 restricts the run to the first 2 manifest episodes,
# writes ONLY <OUT_ROOT>/probe2/, and prints the extrapolated full-build cost
# (validation-ladder rung 4). Any gate failure exits non-zero.
#
# Only the VAE files are prefetched, at the manifest's PINNED revision (cycle-B review B1):
# this job never loads the transformer or the text encoder. The prefetch runs BEFORE python
# starts (distributed launches used to die on HF 408 timeouts mid-JAX-init) and the build then
# resolves that revision from the LOCAL cache only.
#
# Run bash_scripts/setup.sh once on the TPU before this script:
#   bash bash_scripts/setup.sh MODE=stable DEVICE=tpu

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

if [ -f "$HOME/.config/irom-tpu/secrets.env" ]; then
  set +x
  source "$HOME/.config/irom-tpu/secrets.env"
  set -x
fi

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "maxdiffusion_venv/bin/activate" ]; then
  source maxdiffusion_venv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

MANIFEST_PATH="${MANIFEST_PATH:-docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json}"
OUT_ROOT="${OUT_ROOT:-gs://v6_east1d/datasets/exp02_overfit100}"
CONFIG_PATH="${CONFIG_PATH:-src/maxdiffusion/configs/base_wan_5b_full_ft.yml}"
TMP_DIR="${TMP_DIR:-}"
PROBE="${PROBE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SHARD_SIZE="${SHARD_SIZE:-0}"
# pyconfig key=value pairs forwarded to the VAE-only pipeline, space separated.
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-hardware=tpu}"
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

# B1: the VAE repo/revision come from the MANIFEST, not from the launcher -- the manifest pin
# is the contract, and the build re-verifies it against the resolved snapshot.
VAE_REPO="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['hf_repo'])" "${MANIFEST_PATH}")"
VAE_REVISION="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['revision'])" "${MANIFEST_PATH}")"

echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "VAE_REPO=${VAE_REPO}"
echo "VAE_REVISION=${VAE_REVISION}"
echo "CONFIG_OVERRIDES=${CONFIG_OVERRIDES}"
echo "PROBE=${PROBE}"
echo "DRY_RUN=${DRY_RUN}"
echo "SHARD_SIZE=${SHARD_SIZE:-<default>}"
echo "TMP_DIR=${TMP_DIR:-<mktemp>}"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

# VAE-only, pinned prefetch. Nothing from transformer/ or text_encoder/ is requested.
HF_PREFETCH_REVISION="${VAE_REVISION}" \
  bash bash_scripts/prefetch_hf_snapshot.sh "${VAE_REPO}" "model_index.json vae/*"

ARGS=(
  --manifest "${MANIFEST_PATH}"
  --out-root "${OUT_ROOT}"
  --config "${CONFIG_PATH}"
)
for override in ${CONFIG_OVERRIDES}; do
  ARGS+=(--config-override "${override}")
done
if [ -n "${TMP_DIR}" ]; then
  ARGS+=(--tmp-dir "${TMP_DIR}")
fi
if [ "${SHARD_SIZE}" != "0" ]; then
  ARGS+=(--shard-size "${SHARD_SIZE}")
fi
if [ "${PROBE}" = "1" ]; then
  ARGS+=(--probe)
fi
if [ "${DRY_RUN}" = "1" ]; then
  ARGS+=(--dry-run)
fi

python -m maxdiffusion.data_preprocessing.build_overfit100_dataset "${ARGS[@]}"
