#!/usr/bin/env bash
# Submit a one-shot Wan2.2 TI2V 5B visual-validation job to the IROM TPU queue.
#
# This is the queue-model replacement for the old v6e-8 watcher: it provisions a
# small v6e slice, restores one adapter checkpoint of an existing training run,
# runs a 25-step rollout on the val TFRecords, and writes comparison videos +
# latent/pixel/SSIM metrics under <OUTPUT_DIR>/<RUN_NAME>/validation/step_XXXXXX/.
#
# How the queue works (same as launch_wan_train.sh):
#   - This script only SUBMITS a job (uploads the repo as a code tarball + spec).
#     A central scheduler provisions the TPU, runs --setup-cmd on the worker, and
#     runs the validation command.
#   - Code comes from --code-dir (tracked + untracked, non-ignored files), NOT a
#     git branch. Commit/save what you want included.
#
# Usage:
#   RUN_NAME=wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110 \
#     bash bash_scripts/launch_wan_validate.sh
#
# Optional overrides (env vars):
#   WAN_EXPERIMENT=pre_context|side_adapter   (default pre_context; sets adapter + OUTPUT_DIR)
#   CHECKPOINT_STEP=-1                        (-1 = latest; or a specific step, e.g. 30000)
#   NUM_EVAL_VIDEOS=4                         (number of val samples to render)
#   EVAL_DATA_DIR=gs://.../val                (point at .../train to probe overfit)
#   VALIDATION_START_INDEX=0                  (which val sample to start from)
#   TPU_CHIPS=8                               (v6e chip count for the validation slice)
#   NAME=<job name>                           (default wan-<type>-val-yixun)

set -euo pipefail

export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAN_EXPERIMENT="${WAN_EXPERIMENT:-pre_context}"
TPU_CHIPS="${TPU_CHIPS:-8}"

RUN_NAME="${RUN_NAME:?RUN_NAME must be the existing training run to validate}"
MODEL_DIR="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"

# ---- Choose exactly one experiment (must match how the run was trained) ----
case "$WAN_EXPERIMENT" in
  pre_context)
    ACTION_ADAPTER_TYPE="pre_context"
    OUTPUT_DIR="gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter"
    ;;
  side_adapter)
    ACTION_ADAPTER_TYPE="side_adapter"
    OUTPUT_DIR="gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter"
    ;;
  *)
    echo "WAN_EXPERIMENT must be pre_context or side_adapter" >&2
    exit 1
    ;;
esac

# ---- Validation knobs ----
CHECKPOINT_STEP="${CHECKPOINT_STEP:--1}"
NUM_EVAL_VIDEOS="${NUM_EVAL_VIDEOS:-4}"
VALIDATION_START_INDEX="${VALIDATION_START_INDEX:-0}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"

NAME="${NAME:-wan-${ACTION_ADAPTER_TYPE}-val-yixun}"

# Record the exact code commit for provenance. The worker has no .git, so we
# capture it here and pass it via --env (validate_wan_side_adapter.sh reads it).
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Runs on the worker before validation: build the venv and prefetch the model.
SETUP_CMD="bash bash_scripts/setup.sh MODE=stable DEVICE=tpu && bash bash_scripts/prefetch_hf_snapshot.sh ${MODEL_DIR}"

echo "[submit] validating RUN_NAME=${RUN_NAME}"
echo "[submit]   WAN_EXPERIMENT=${WAN_EXPERIMENT} ACTION_ADAPTER_TYPE=${ACTION_ADAPTER_TYPE}"
echo "[submit]   OUTPUT_DIR=${OUTPUT_DIR}"
echo "[submit]   CHECKPOINT_STEP=${CHECKPOINT_STEP} NUM_EVAL_VIDEOS=${NUM_EVAL_VIDEOS}"
echo "[submit]   EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "[submit]   videos + metrics -> ${OUTPUT_DIR%/}/${RUN_NAME}/validation/step_*/"

# ---- Submit the job to the queue ----
tpu create v6 -n "$TPU_CHIPS" \
  --name "$NAME" \
  --code-dir "$REPO_ROOT" \
  --setup-cmd "$SETUP_CMD" \
  --env RUN_NAME="$RUN_NAME" \
  --env COMMIT="$COMMIT" \
  --env ACTION_ADAPTER_TYPE="$ACTION_ADAPTER_TYPE" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env EVAL_DATA_DIR="$EVAL_DATA_DIR" \
  --env MODEL_DIR="$MODEL_DIR" \
  --env CHECKPOINT_STEP="$CHECKPOINT_STEP" \
  --env NUM_EVAL_VIDEOS="$NUM_EVAL_VIDEOS" \
  --env VALIDATION_START_INDEX="$VALIDATION_START_INDEX" \
  --env VALIDATION_SEED="$VALIDATION_SEED" \
  --env PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
  --env PRE_CONTEXT_TOKENS="8" \
  --env PRE_CONTEXT_HEADS="8" \
  --env HF_HUB_DISABLE_XET="1" \
  --env HF_HUB_ENABLE_HF_TRANSFER="0" \
  -- bash bash_scripts/validate_wan_side_adapter.sh
