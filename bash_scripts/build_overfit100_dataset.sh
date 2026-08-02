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
# >>> launch-commit guard  (executed verbatim by test_overfit100_gates.py -- keep the sentinels)
# The queue deploys an uploaded TARBALL with no .git, so the builder's clean-commit guard runs
# in deployed-code mode and relays THIS value (probe failure 20260729-062523). It must be
# EXPORTED -- python reads the process environment -- and it must be the sha the launcher
# verified clean: `tpu create --env COMMIT=$(git rev-parse HEAD)`. Checked here so a bad launch
# fails in seconds instead of after the HF prefetch.
#
# T1: the worktree probe strips git's repository-selection variables, so an ambient GIT_DIR
# cannot make a real checkout look deployed. T3: COMMIT must be EXACTLY one 40-hex token --
# `grep -Eq '^[0-9a-f]{40}$'` accepts a multi-line value if ANY line matches.
GIT_ISOLATED=(env -u GIT_DIR -u GIT_WORK_TREE -u GIT_COMMON_DIR -u GIT_INDEX_FILE
  -u GIT_INDEX_VERSION -u GIT_OBJECT_DIRECTORY -u GIT_ALTERNATE_OBJECT_DIRECTORIES
  -u GIT_CEILING_DIRECTORIES -u GIT_DISCOVERY_ACROSS_FILESYSTEM -u GIT_NAMESPACE
  -u GIT_PREFIX -u GIT_TOPLEVEL git)
COMMIT="${COMMIT:-$("${GIT_ISOLATED[@]}" rev-parse HEAD 2>/dev/null || echo unknown)}"
export COMMIT
if ! "${GIT_ISOLATED[@]}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ ${#COMMIT} -ne 40 || ! ${COMMIT} =~ ^[0-9a-f]{40}$ || ${COMMIT} == *[[:space:]]* ]]; then
    echo "[build] FATAL: deployed-code mode (no git worktree) requires COMMIT=<exactly one 40-hex sha> in the job env; got '${COMMIT}'" >&2
    exit 1
  fi
  echo "[build] deployed-code mode: relaying launch-time COMMIT=${COMMIT}"
fi
# <<< launch-commit guard

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

# >>> ffmpeg ensure
# Probe attempt 2 (job 20260729-172443-23bcb17a) loaded the pinned VAE and passed all three V1
# windows, then died in the V3 precheck with `FileNotFoundError: 'ffmpeg'`: the TPU worker image
# has no ffmpeg. Install it HERE -- before the multi-minute HF prefetch and the JAX init -- so
# the failure mode is a 30-second apt error rather than 20 minutes of wasted TPU time.
#
# apt options mirror setup.sh's ephemeral-worker hardening: a single bounded wall-clock budget
# (`apt_deadline_run`, never -1), `-o DPkg::Lock::Timeout=60` per invocation, and a LOUD exit on
# any failure. setup.sh has already stopped/disabled the apt-daily timers under
# EPHEMERAL_WORKER=1, so contention is expected to be gone by the time this runs.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "[build] ffmpeg/ffprobe not on PATH; installing (the TPU worker image ships without them)"
  FFMPEG_APT_BUDGET="${FFMPEG_APT_BUDGET:-420}"
  APT_SECTION_START=$SECONDS
  APT_SUDO=""
  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then APT_SUDO="sudo"; fi
  apt_deadline_run() {
    rem=$((FFMPEG_APT_BUDGET - (SECONDS - APT_SECTION_START)))
    if [ "$rem" -le 0 ]; then
      echo "[build] FATAL: ffmpeg install exceeded its ${FFMPEG_APT_BUDGET}s budget" >&2
      exit 1
    fi
    timeout "$rem" $APT_SUDO "$@"
  }
  apt_deadline_run apt-get -o DPkg::Lock::Timeout=60 update -y &&
    apt_deadline_run apt-get -o DPkg::Lock::Timeout=60 install -y --no-install-recommends ffmpeg ||
    {
      echo "[build] FATAL: could not install ffmpeg (budget ${FFMPEG_APT_BUDGET}s, 60s dpkg-lock bound)" >&2
      exit 1
    }
fi
for tool in ffmpeg ffprobe; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "[build] FATAL: ${tool} is still not on PATH after the install attempt" >&2
    exit 1
  }
done
ffmpeg -version | head -1
ffprobe -version | head -1
# <<< ffmpeg ensure

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
