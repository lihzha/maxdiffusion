#!/usr/bin/env bash
# Submit a Wan2.2 TI2V 5B training job to the IROM TPU queue via `tpu create`.
#
# How the queue works (vs the old `tpu watch`):
#   - This script only SUBMITS a job (uploads the repo as a code tarball + a
#     spec to gs://.../tpu-job-queue). A central scheduler then provisions the
#     v6e TPU, runs --setup-cmd on every worker, and runs the training command.
#   - Code comes from --code-dir (this repo's tracked + untracked, non-ignored
#     files), NOT a git branch. Commit/save what you want included.
#   - W&B: the key is forwarded from the submitting shell (see the W&B block
#     below); export WANDB_API_KEY (and optionally WANDB_ENTITY) before running.
#
# Usage:
#   SMOKE=1 bash bash_scripts/launch_wan_train.sh     # 1-step smoke test
#   bash bash_scripts/launch_wan_train.sh             # full run
#   DRY_RUN=1 bash bash_scripts/launch_wan_train.sh   # print the submit command, submit nothing
#
# Optional overrides (env vars):
#   WAN_EXPERIMENT=pre_context|side_adapter|full_ft|overfit100   (default pre_context)
#   TPU_CHIPS=64                              (v6e chip count)
#   NAME=<job name>                           (default wan-<type>-yixun)
#   RUN_NAME=<run name>                       (default: the fresh-run template below; export it
#                                              explicitly to RESUME an existing run dir — Orbax
#                                              restores the latest checkpoint under the same name)
#   PER_DEVICE_BATCH_SIZE=<n>                 (AUTHORITATIVE batch knob: pyconfig derives
#                                              GBS = num_devices x this on the worker.
#                                              Arm-dependent defaults: adapters 8;
#                                              full_ft 4 -- per-device 8 OOMs v6e-64 for
#                                              full-FT, fit probe #4 / Query 7; overfit100 4.0
#                                              (the exp_02 campaign recipe). Small
#                                              topologies must lower it further, e.g. 1)
#   GLOBAL_BATCH_SIZE_TO_TRAIN_ON=<n>         (inert to training — pyconfig recomputes
#   GLOBAL_BATCH_SIZE_TO_LOAD=<n>              both from per-device; defaults follow the
#                                              arm: adapters 512, full_ft/overfit100 256.
#                                              Override alongside PER_DEVICE_BATCH_SIZE so the
#                                              worker-log echoes stay honest, 1/8/8 on
#                                              a v6e-8 smoke)
#   overfit100-only knobs (exp_02; forwarded only for that arm, all env-overridable):
#     MAX_TRAIN_STEPS (2500)                  DATA_DIR (gs://.../exp02_overfit100/train100)
#     EXPECTED_WINDOWS (1629)                 NUM_TEXT_SLOTS (100)
#     CHECKPOINT_STEPS ([250,500,1000,1750,2500])   LEARNING_RATE (1e-5)
#     WARMUP_STEPS (250)                      TEXT_ENCODE_BATCH (8)
#     MANIFEST_PATH (docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json)
#     (train10 smoke ladder: DATA_DIR=.../train10 EXPECTED_WINDOWS=167 NUM_TEXT_SLOTS=10)

set -euo pipefail

export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAN_EXPERIMENT="${WAN_EXPERIMENT:-pre_context}"
TPU_CHIPS="${TPU_CHIPS:-64}"

MODEL_DIR="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
TRAIN_DATA_DIR="gs://v6_east1d/datasets/droid_wan_side_adapter/train"
EVAL_DATA_DIR="gs://v6_east1d/datasets/droid_wan_side_adapter/val"

# The queue runs this training wrapper on each worker. The full_ft / overfit100 arms override
# it to their own wrappers; pre_context / side_adapter keep the side-adapter wrapper.
TRAIN_SCRIPT="bash_scripts/train_wan_side_adapter.sh"

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
  full_ft)
    # Diagnostic full-finetune overfit probe (docs/worklogs_yixun/
    # exp_01_full_ft_overfit_claude/plan_full_ft_overfit.md). No adapter exists, so
    # ACTION_ADAPTER_TYPE is a run-name / provenance tag ONLY here -- train_wan_full_ft.sh
    # never reads it (model_type: FULL_FT_TI2V comes from the yml). The 2500 checkpoint
    # cadence + train-split eval are re-applied AFTER the common defaults below so they win.
    ACTION_ADAPTER_TYPE="full-ft"
    OUTPUT_DIR="gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft"
    WANDB_PROJECT="maxdiffusion-wan-full-ft"
    MAX_TRAIN_STEPS="20000"
    TRAIN_SCRIPT="bash_scripts/train_wan_full_ft.sh"
    ;;
  overfit100)
    # exp_02 memorization probe (docs/worklogs_yixun/exp_02_overfit100_claude/
    # plan_overfit100.md). As with full_ft, ACTION_ADAPTER_TYPE is a run-name / provenance
    # tag ONLY -- train_wan_overfit100.sh never reads it (model_type: OVERFIT100_TI2V comes
    # from base_wan_5b_overfit100.yml). The campaign's dataset/cadence/LR knobs are applied
    # AFTER the common defaults below so they win; MAX_TRAIN_STEPS is env-respecting because
    # the exp_02 ladder drives it per segment (2,500-step segments, resumed extensions).
    ACTION_ADAPTER_TYPE="overfit100"
    OUTPUT_DIR="gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100"
    WANDB_PROJECT="maxdiffusion-wan-overfit100"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2500}"
    TRAIN_SCRIPT="bash_scripts/train_wan_overfit100.sh"
    ;;
  *)
    echo "WAN_EXPERIMENT must be pre_context, side_adapter, full_ft, or overfit100" >&2
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
# Batch trio: env-overridable (mini-cycle 6: the v6e-8 smoke OOM'd at a hard-set
# per-device 8) with ARM-DEPENDENT defaults (mini-cycle 8): the full_ft override block
# below swaps in the amended full-FT recipe, and the env-respecting collapse happens
# ONCE after it -- env unset => adapter arms submit 8/512/512 and full_ft submits
# 4/256/256; an explicitly exported value wins for every arm. Per-device is the
# authoritative knob (pyconfig recomputes both GBS keys from it; round-5 F1 test); the
# globals ride along so submitted env + worker echoes stay coherent — NOT bash-derived
# from TPU_CHIPS because per-device may legitimately be fractional (e.g. 1.0) and
# $((...)) would choke, duplicating pyconfig's formula for no gain.
PER_DEVICE_BATCH_SIZE_DEFAULT="8"
GLOBAL_BATCH_SIZE_TO_TRAIN_ON_DEFAULT="512"
GLOBAL_BATCH_SIZE_TO_LOAD_DEFAULT="512"
TFRECORD_SHUFFLE_BUFFER_SIZE="1024"

# ---- Full-FT overrides ----
# Applied AFTER the shared defaults above so the common CHECKPOINT_EVERY=100 /
# CHECKPOINT_KEEP_PERIOD=1000 / val-split EVAL_DATA_DIR do NOT clobber the probe's 2500
# cohort cadence + train-split evaluation (plan §2.2/§2.3, finding G1). Placed BEFORE the
# SMOKE block so a full_ft smoke still disables checkpointing.
if [ "$WAN_EXPERIMENT" = "full_ft" ]; then
  CHECKPOINT_EVERY="2500"
  CHECKPOINT_KEEP_PERIOD="2500"
  EVAL_DATA_DIR="$TRAIN_DATA_DIR"
  # Amended primary recipe (plan §2.2 v3.1, Query 7, 2026-07-19): fit probe #4 proved
  # per-device 8 OOMs v6e-64 by ~37MB (FSDP collective buffers that full-FT pays and the
  # frozen-adapter runs did not), so full_ft defaults to per-device 4 => GBS 256; the
  # arm's MAX_TRAIN_STEPS=20000 keeps the 3.55-pass sample budget unchanged.
  PER_DEVICE_BATCH_SIZE_DEFAULT="4"
  GLOBAL_BATCH_SIZE_TO_TRAIN_ON_DEFAULT="256"
  GLOBAL_BATCH_SIZE_TO_LOAD_DEFAULT="256"
fi

# ---- overfit100 overrides (exp_02) ----
# Same placement rationale as the full_ft block. train_wan_overfit100.sh drives checkpoint
# cadence with an EXPLICIT step list (CHECKPOINT_STEPS) + CHECKPOINT_EVERY=0 and runs no
# in-loop eval, so the common every-100/every-1000 cadence is zeroed here rather than
# forwarded (the campaign's Jobs 9-51 all ran this shape). SAVE_FINAL_CHECKPOINT=False
# matches the wrapper's own default: the segment end is always IN the step list.
if [ "$WAN_EXPERIMENT" = "overfit100" ]; then
  DATA_DIR="${DATA_DIR:-gs://v6_east1d/datasets/exp02_overfit100/train100}"
  EVAL_DATA_DIR="$DATA_DIR"
  EXPECTED_WINDOWS="${EXPECTED_WINDOWS:-1629}"
  NUM_TEXT_SLOTS="${NUM_TEXT_SLOTS:-100}"
  TEXT_ENCODE_BATCH="${TEXT_ENCODE_BATCH:-8}"
  MANIFEST_PATH="${MANIFEST_PATH:-docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json}"
  LEARNING_RATE="${LEARNING_RATE:-1e-5}"
  WARMUP_STEPS="${WARMUP_STEPS:-250}"
  CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-[250,500,1000,1750,2500]}"
  CHECKPOINT_EVERY="0"
  CHECKPOINT_KEEP_PERIOD=""
  EVAL_EVERY="0"
  EVAL_MAX_BATCHES=""
  SAVE_FINAL_CHECKPOINT="False"
  # Campaign recipe (S3 on v6e-64 and the v6e-8 smokes both ran the wrapper default 4.0).
  PER_DEVICE_BATCH_SIZE_DEFAULT="4.0"
  GLOBAL_BATCH_SIZE_TO_TRAIN_ON_DEFAULT="256"
  GLOBAL_BATCH_SIZE_TO_LOAD_DEFAULT="256"
fi

# Env-respecting collapse: an explicitly exported batch value beats the arm default
# (small-topology smokes pass e.g. PER_DEVICE_BATCH_SIZE=1 GLOBAL_*=8).
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-$PER_DEVICE_BATCH_SIZE_DEFAULT}"
GLOBAL_BATCH_SIZE_TO_TRAIN_ON="${GLOBAL_BATCH_SIZE_TO_TRAIN_ON:-$GLOBAL_BATCH_SIZE_TO_TRAIN_ON_DEFAULT}"
GLOBAL_BATCH_SIZE_TO_LOAD="${GLOBAL_BATCH_SIZE_TO_LOAD:-$GLOBAL_BATCH_SIZE_TO_LOAD_DEFAULT}"

# Non-smoke run name, built AFTER the batch collapse so it interpolates the RESOLVED
# global batch (cycle-8 change order: the old hard-coded `gbs512` lied for full_ft's
# gbs256 recipe; adapter arms still resolve 512, keeping their names byte-identical).
# Env-respecting (2026-08-13, overfit100 arm): an exported RUN_NAME wins, which is how
# the exp_02 resume/extension flow re-enters an existing run dir. The SMOKE block below
# overwrites the whole name with its own template, as before.
RUN_NAME="${RUN_NAME:-wan-${ACTION_ADAPTER_TYPE}-v6e${TPU_CHIPS}-full-gbs${GLOBAL_BATCH_SIZE_TO_TRAIN_ON}-fresh-$(date -u +%Y%m%d-%H%M%S)}"

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
# (An empty setup_cmd is a no-op on the queue, and the train wrapper
# needs the .venv that setup.sh creates.)
# EPHEMERAL_WORKER=1: queue workers are single-job throwaway VMs -- root setup.sh uses
# the flag to gate its PERSISTENT auto-update disable (persistent GPU/dev hosts running
# setup.sh manually keep their security-update posture; they get current-boot stops only).
SETUP_CMD="EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu && bash bash_scripts/prefetch_hf_snapshot.sh ${MODEL_DIR}"

# W&B key is forwarded from your shell env (export WANDB_API_KEY in ~/.zshrc).
# NOTE: this writes the key into the job spec on GCS, readable by anyone with
# bucket access. If the key is absent, disable W&B so the run doesn't crash.
# WANDB_ENTITY (optional): forwarded only when exported, so the run lands in a named
# entity instead of the authenticated key's default one (see CLAUDE.md "W&B entity
# facts", 2026-08-13: with entity unset, runs go wherever the worker's key points).
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[submit] WARNING: WANDB_API_KEY not set in your shell; disabling W&B for this run." >&2
  WANDB_PROJECT=""
fi

# Arm-specific extra env, appended to the submit command. Expanded with the
# bash-3.2-safe idiom (macOS /bin/bash chokes on empty arrays under set -u).
EXTRA_ENV=()
if [ -n "${WANDB_ENTITY:-}" ]; then
  EXTRA_ENV+=( --env WANDB_ENTITY="$WANDB_ENTITY" )
fi
if [ "$WAN_EXPERIMENT" = "overfit100" ]; then
  EXTRA_ENV+=(
    --env DATA_DIR="$DATA_DIR"
    --env EXPECTED_WINDOWS="$EXPECTED_WINDOWS"
    --env NUM_TEXT_SLOTS="$NUM_TEXT_SLOTS"
    --env TEXT_ENCODE_BATCH="$TEXT_ENCODE_BATCH"
    --env MANIFEST_PATH="$MANIFEST_PATH"
    --env LEARNING_RATE="$LEARNING_RATE"
    --env WARMUP_STEPS="$WARMUP_STEPS"
    --env CHECKPOINT_STEPS="$CHECKPOINT_STEPS"
  )
fi

# ---- Submit the job to the queue ----
SUBMIT=( tpu create v6 -n "$TPU_CHIPS"
  --name "$NAME"
  --code-dir "$REPO_ROOT"
  --setup-cmd "$SETUP_CMD"
  --env WANDB_API_KEY="${WANDB_API_KEY:-}"
  --env RUN_NAME="$RUN_NAME"
  --env COMMIT="$COMMIT"
  --env ACTION_ADAPTER_TYPE="$ACTION_ADAPTER_TYPE"
  --env TRAIN_DATA_DIR="$TRAIN_DATA_DIR"
  --env EVAL_DATA_DIR="$EVAL_DATA_DIR"
  --env OUTPUT_DIR="$OUTPUT_DIR"
  --env MODEL_DIR="$MODEL_DIR"
  --env PRE_CONTEXT_TOKENS="8"
  --env PRE_CONTEXT_HEADS="8"
  --env SIDE_ADAPTER_NOISE_MODE="fresh"
  --env MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS"
  --env CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
  --env CHECKPOINT_KEEP_PERIOD="$CHECKPOINT_KEEP_PERIOD"
  --env EVAL_EVERY="$EVAL_EVERY"
  --env EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES"
  --env LOG_PERIOD="$LOG_PERIOD"
  --env SAVE_FINAL_CHECKPOINT="$SAVE_FINAL_CHECKPOINT"
  --env PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE"
  --env GLOBAL_BATCH_SIZE_TO_TRAIN_ON="$GLOBAL_BATCH_SIZE_TO_TRAIN_ON"
  --env GLOBAL_BATCH_SIZE_TO_LOAD="$GLOBAL_BATCH_SIZE_TO_LOAD"
  --env TFRECORD_SHUFFLE_BUFFER_SIZE="$TFRECORD_SHUFFLE_BUFFER_SIZE"
  --env WANDB_PROJECT="$WANDB_PROJECT"
  --env HF_HUB_DISABLE_XET="1"
  --env HF_HUB_ENABLE_HF_TRANSFER="0"
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}
  -- bash "$TRAIN_SCRIPT" )

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[submit] DRY_RUN=1 — the command below was NOT submitted:"
  printf '%q ' "${SUBMIT[@]}"
  printf '\n'
  exit 0
fi

"${SUBMIT[@]}"

# Note: visual validation (the old v6e-8 watcher) is a separate concern in the
# queue model. Submit it as its own `tpu create v6 -n 8 ... -- <validation cmd>`
# job once training is producing checkpoints.
