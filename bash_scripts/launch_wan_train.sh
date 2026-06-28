#!/usr/bin/env bash
# Submit a Wan2.2 TI2V 5B training job to the IROM TPU queue via `tpu create`.
#
# How the queue works (vs the old `tpu watch`):
#   - This script only SUBMITS a job (uploads the repo as a code tarball + a
#     spec to gs://.../tpu-job-queue). A central scheduler then provisions the
#     v6e TPU, runs --setup-cmd on every worker, and runs the training command.
#   - Code comes from --code-dir (this repo's tracked + untracked, non-ignored
#     files), NOT a git branch. Commit/save what you want included.
#   - W&B key is read on the worker from Secret Manager secret `wandb-api-key`
#     (key is in ~/.zshrc)
#
# Usage:
#   SMOKE=1 bash bash_scripts/launch_wan_train.sh     # 1-step smoke test
#   bash bash_scripts/launch_wan_train.sh             # full run
#
# Optional overrides (env vars):
#   WAN_EXPERIMENT=pre_context|side_adapter   (default pre_context)
#   TPU_CHIPS=64                              (v6e chip count)
#   NAME=<job name>                           (default wan-<type>-yixun)

set -euo pipefail

export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAN_EXPERIMENT="${WAN_EXPERIMENT:-pre_context}"
TPU_CHIPS="${TPU_CHIPS:-64}"

MODEL_DIR="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
TRAIN_DATA_DIR="gs://v6_east1d/datasets/droid_wan_side_adapter/train"
EVAL_DATA_DIR="gs://v6_east1d/datasets/droid_wan_side_adapter/val"

# ---- Choose exactly one experiment ----
case "$WAN_EXPERIMENT" in
  pre_context)
    ACTION_ADAPTER_TYPE="pre_context"
    OUTPUT_DIR="gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter"
    WANDB_PROJECT="maxdiffusion-wan-pre-context-adapter"
    MAX_TRAIN_STEPS="30000"
    ;;
  side_adapter)
    ACTION_ADAPTER_TYPE="side_adapter"
    OUTPUT_DIR="gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter"
    WANDB_PROJECT="maxdiffusion-wan-side-adapter"
    MAX_TRAIN_STEPS="10000"
    ;;
  *)
    echo "WAN_EXPERIMENT must be pre_context or side_adapter" >&2
    exit 1
    ;;
esac

# ---- Training hyperparameters (full-run defaults; match README) ----
CHECKPOINT_EVERY="100"
CHECKPOINT_KEEP_PERIOD="1000"
EVAL_EVERY="1000"
EVAL_MAX_BATCHES="4"
LOG_PERIOD="10"
SAVE_FINAL_CHECKPOINT="True"
PER_DEVICE_BATCH_SIZE="8"
GLOBAL_BATCH_SIZE_TO_TRAIN_ON="512"
GLOBAL_BATCH_SIZE_TO_LOAD="512"
TFRECORD_SHUFFLE_BUFFER_SIZE="1024"
RUN_NAME="wan-${ACTION_ADAPTER_TYPE}-v6e${TPU_CHIPS}-full-gbs512-fresh-$(date -u +%Y%m%d-%H%M%S)"

# ---- Optional one-step smoke (SMOKE=1) ----
if [ "${SMOKE:-0}" = "1" ]; then
  RUN_NAME="smoke-${ACTION_ADAPTER_TYPE}-$(date -u +%Y%m%d-%H%M%S)"
  MAX_TRAIN_STEPS="1"
  CHECKPOINT_EVERY="0"
  CHECKPOINT_KEEP_PERIOD=""
  EVAL_EVERY="0"
  EVAL_MAX_BATCHES=""
  SAVE_FINAL_CHECKPOINT="False"
  echo "[submit] SMOKE mode: 1 step, no checkpoint/eval."
fi

NAME="${NAME:-wan-${ACTION_ADAPTER_TYPE}-yixun}"

# Record the exact code commit for provenance. The worker has no .git, so we
# capture it here and pass it via --env. Note: the queue uploads the
# working tree, so this is only accurate when your tree is committed/clean.
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

# Runs on every worker before training: build the venv and prefetch the model.
# (An empty setup_cmd is a no-op on the queue, and train_wan_side_adapter.sh
# needs the .venv that setup.sh creates.)
SETUP_CMD="bash bash_scripts/setup.sh MODE=stable DEVICE=tpu && bash bash_scripts/prefetch_hf_snapshot.sh ${MODEL_DIR}"

# ---- Submit the job to the queue ----
tpu create v6 -n "$TPU_CHIPS" \
  --name "$NAME" \
  --code-dir "$REPO_ROOT" \
  --setup-cmd "$SETUP_CMD" \
  --secret WANDB_API_KEY=wandb-api-key \
  --env RUN_NAME="$RUN_NAME" \
  --env COMMIT="$COMMIT" \
  --env ACTION_ADAPTER_TYPE="$ACTION_ADAPTER_TYPE" \
  --env TRAIN_DATA_DIR="$TRAIN_DATA_DIR" \
  --env EVAL_DATA_DIR="$EVAL_DATA_DIR" \
  --env OUTPUT_DIR="$OUTPUT_DIR" \
  --env MODEL_DIR="$MODEL_DIR" \
  --env PRE_CONTEXT_TOKENS="8" \
  --env PRE_CONTEXT_HEADS="8" \
  --env SIDE_ADAPTER_NOISE_MODE="fresh" \
  --env MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
  --env CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  --env CHECKPOINT_KEEP_PERIOD="$CHECKPOINT_KEEP_PERIOD" \
  --env EVAL_EVERY="$EVAL_EVERY" \
  --env EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" \
  --env LOG_PERIOD="$LOG_PERIOD" \
  --env SAVE_FINAL_CHECKPOINT="$SAVE_FINAL_CHECKPOINT" \
  --env PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
  --env GLOBAL_BATCH_SIZE_TO_TRAIN_ON="$GLOBAL_BATCH_SIZE_TO_TRAIN_ON" \
  --env GLOBAL_BATCH_SIZE_TO_LOAD="$GLOBAL_BATCH_SIZE_TO_LOAD" \
  --env TFRECORD_SHUFFLE_BUFFER_SIZE="$TFRECORD_SHUFFLE_BUFFER_SIZE" \
  --env WANDB_PROJECT="$WANDB_PROJECT" \
  --env HF_HUB_DISABLE_XET="1" \
  --env HF_HUB_ENABLE_HF_TRANSFER="0" \
  -- bash bash_scripts/train_wan_side_adapter.sh

# Note: visual validation (the old v6e-8 watcher) is a separate concern in the
# queue model. Submit it as its own `tpu create v6 -n 8 ... -- <validation cmd>`
# job once training is producing checkpoints.
