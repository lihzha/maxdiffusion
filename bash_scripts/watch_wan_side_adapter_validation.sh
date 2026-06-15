#!/usr/bin/env bash
set -euo pipefail

# Local-side watcher. It polls a training run's GCS checkpoint directory and
# launches one-shot visual validation jobs on a separate TPU slice.

if [ -f "$HOME/.config/irom-tpu/secrets.env" ]; then
  set +x
  source "$HOME/.config/irom-tpu/secrets.env"
fi

export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
export TPU_PROJECT="${TPU_PROJECT:-mae-irom-lab-guided-data}"
export TPU_ZONE_v4="${TPU_ZONE_v4:-us-central2-b}"
export TPU_ZONE_v5="${TPU_ZONE_v5:-us-central1-a}"
export TPU_ZONE_v6="${TPU_ZONE_v6:-us-east1-d}"
export TPU_BUCKET_v4="${TPU_BUCKET_v4:-gs://pi0-cot}"
export TPU_BUCKET_v5="${TPU_BUCKET_v5:-gs://v5_central1_a}"
export TPU_BUCKET_v6="${TPU_BUCKET_v6:-gs://v6_east1d}"
export GH_OWNER="${GH_OWNER:-lihzha}"
export GH_REPO_NAME="${GH_REPO_NAME:-maxdiffusion}"
export SLEEP_SECS="${SLEEP_SECS:-30}"
export TPU_WATCH_HEALTH_CHECK_SECS="${TPU_WATCH_HEALTH_CHECK_SECS:-30}"

RUN_NAME="${RUN_NAME:?RUN_NAME must match the training run}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/validation}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
VALIDATION_TPU_NAME="${VALIDATION_TPU_NAME:-v6-8-wan-val-lzha}"
VALIDATION_TPU_CHIPS="${VALIDATION_TPU_CHIPS:-8}"
VALIDATION_EVERY="${VALIDATION_EVERY:-1000}"
VALIDATION_FIRST_STEP="${VALIDATION_FIRST_STEP:-100}"
VALIDATION_MIN_STEP="${VALIDATION_MIN_STEP:-100}"
NUM_EVAL_VIDEOS="${NUM_EVAL_VIDEOS:-4}"
VALIDATION_START_INDEX="${VALIDATION_START_INDEX:-0}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
POLL_SECS="${POLL_SECS:-120}"
VALIDATION_MAX_WAIT_SECS="${VALIDATION_MAX_WAIT_SECS:-21600}"
COMMIT="${COMMIT:-$(git rev-parse HEAD)}"
STATE_FILE="${STATE_FILE:-logs/wan_side_adapter_validation_seen_${RUN_NAME}.txt}"

mkdir -p "$(dirname "$STATE_FILE")" logs
touch "$STATE_FILE"

echo "RUN_NAME=${RUN_NAME}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR}"
echo "VALIDATION_TPU_NAME=${VALIDATION_TPU_NAME}"
echo "VALIDATION_TPU_CHIPS=${VALIDATION_TPU_CHIPS}"
echo "VALIDATION_FIRST_STEP=${VALIDATION_FIRST_STEP}"
echo "VALIDATION_EVERY=${VALIDATION_EVERY}"
echo "VALIDATION_MAX_WAIT_SECS=${VALIDATION_MAX_WAIT_SECS}"
echo "COMMIT=${COMMIT}"

_step_is_target() {
  local step="$1"
  if (( step < VALIDATION_MIN_STEP )); then
    return 1
  fi
  if (( step == VALIDATION_FIRST_STEP )); then
    return 0
  fi
  if (( VALIDATION_EVERY > 0 && step % VALIDATION_EVERY == 0 )); then
    return 0
  fi
  return 1
}

_checkpoint_steps() {
  gsutil ls -d "${CHECKPOINT_DIR%/}/"* 2>/dev/null \
    | sed -n 's#.*/\([0-9][0-9]*\)/$#\1#p' \
    | sort -n
}

_summary_path() {
  local step="$1"
  printf "%s/step_%06d/summary.json" "$VALIDATION_OUTPUT_DIR" "$step"
}

_remote_proc_running() {
  gcloud alpha compute tpus tpu-vm ssh "$VALIDATION_TPU_NAME" \
    --project "$TPU_PROJECT" \
    --zone "$TPU_ZONE_v6" \
    --worker=0 \
    --command='bash -lc '"'"'ps -u "$USER" -o cmd | grep -F "generate_wan_side_adapter.py" | grep -v grep >/dev/null'"'"'' \
    >/dev/null 2>&1
}

_wait_for_validation() {
  local step="$1"
  local summary
  local waited=0
  summary="$(_summary_path "$step")"
  while true; do
    if gsutil -q stat "$summary"; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) validation step ${step} complete: ${summary}"
      echo "$step" >> "$STATE_FILE"
      return 0
    fi
    if (( waited >= VALIDATION_MAX_WAIT_SECS )); then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) validation step ${step} timed out after ${waited}s without ${summary}"
      return 1
    fi
    if ! _remote_proc_running; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) validation process not running yet/anymore for step ${step}; waiting for summary or relaunch signal"
    fi
    sleep "$POLL_SECS"
    waited=$((waited + POLL_SECS))
  done
}

_launch_validation() {
  local step="$1"
  local run_tag="wan-side-adapter-validation-${RUN_NAME}-step-${step}"
  local setup_cmd
  setup_cmd="git fetch origin codex/wan-ti2v-side-adapter-20260613-073227 && git checkout --detach ${COMMIT} && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu"

  export TPU_NAME="$VALIDATION_TPU_NAME"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launching validation step ${step} on ${VALIDATION_TPU_NAME}"
  tpu watch v6 -n "$VALIDATION_TPU_CHIPS" --force \
    --setup-cmd "$setup_cmd" \
    "$COMMIT" \
    RUN_NAME="$RUN_NAME" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    CHECKPOINT_STEP="$step" \
    VALIDATION_OUTPUT_DIR="$VALIDATION_OUTPUT_DIR" \
    EVAL_DATA_DIR="$EVAL_DATA_DIR" \
    MODEL_DIR="$MODEL_DIR" \
    NUM_EVAL_VIDEOS="$NUM_EVAL_VIDEOS" \
    VALIDATION_START_INDEX="$VALIDATION_START_INDEX" \
    VALIDATION_SEED="$VALIDATION_SEED" \
    bash bash_scripts/validate_wan_side_adapter.sh \
    2>&1 | tee -a "logs/tpu_watch_${run_tag}.log"
  if ! _wait_for_validation "$step"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) validation step ${step} did not finish; it remains eligible for retry"
  fi
}

while true; do
  for step in $(_checkpoint_steps); do
    if ! _step_is_target "$step"; then
      continue
    fi
    if grep -qx "$step" "$STATE_FILE"; then
      continue
    fi
    if gsutil -q stat "$(_summary_path "$step")"; then
      echo "$step" >> "$STATE_FILE"
      continue
    fi
    _launch_validation "$step"
  done
  sleep "$POLL_SECS"
done
