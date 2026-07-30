#!/usr/bin/env bash
set -euo pipefail

# Raise the open-file limit. The queue worker's login shell defaults to 1024, which the 5B model
# + optimizer state + tensorstore checkpoint restore exhaust (Errno 24).
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

# Run bash_scripts/setup.sh once on the TPU before this script:
#   bash bash_scripts/setup.sh MODE=stable DEVICE=tpu

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

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
--xla_tpu_megacore_fusion_allow_ags=false \
--xla_enable_async_collective_permute=true \
--xla_tpu_enable_ag_backward_pipelining=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_enable_async_all_gather=true \
--xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_async_all_to_all=true \
--xla_tpu_enable_all_experimental_scheduler_features=true \
--xla_tpu_enable_scheduler_memory_pressure_tracking=true \
--xla_tpu_host_transfer_overlap_limit=24 \
--xla_tpu_aggressive_opt_barrier_removal=ENABLED \
--xla_lhs_prioritize_async_depth_over_stall=ENABLED \
--xla_should_allow_loop_variant_parameter_in_chain=ENABLED \
--xla_should_add_loop_invariant_op_in_chain=ENABLED \
--xla_max_concurrent_host_send_recv=100 \
--xla_tpu_scheduler_percent_shared_memory_limit=100 \
--xla_latency_hiding_scheduler_rerun=2 \
--xla_tpu_use_minor_sharding_for_major_trivial_input=true \
--xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1 \
--xla_tpu_assign_all_reduce_scatter_layout=true}"

# --- exp_02 overfit100 ONE-STEP loss sweep (plan v4 D11 "one-step instrument", cycle D) ---
# Runs eval_wan_full_ft_val_loss.py in OVERFIT100 mode against base_wan_5b_overfit100.yml: for
# each checkpoint it computes the EXACT training objective (velocity MSE, frame-0 masked, with
# the SAME per-episode gathered text context) over ALL windows of the set, with each window's
# (t, eps) fixed by (episode_id, window_start) -- NOT by record order -- so the loss curve is a
# pure model effect and is invariant to dataset ordering. Writes val_loss.{json,csv} +
# val_loss_plot.png plus the exp_02-only val_loss_per_window.{json,csv} (per-window loss with
# episode_id / window_start).
#
# Deltas vs eval_wan_full_ft_val_loss.sh: (1) the exp_02 yml; (2) EVAL_DATA_DIR is a TRAIN set
# (memorization is measured ON the training windows); (3) the manifest-pinned model (C1) and the
# exp_02 set knobs (EXPECTED_WINDOWS / NUM_TEXT_SLOTS) travel with it; (4) EXPECTED_COUNT is the
# built window count of the set being evaluated (train100 -> 1629, train10 -> 167).
#
# SMOKE gate: SMOKE_LIMIT=<N> evaluates only the first N batches of only the first checkpoint
# into an isolated validation_loss_smoke/ directory (the n==expected assertion is skipped).

RUN_NAME="${RUN_NAME:?RUN_NAME must match the training run to evaluate}"
MANIFEST_PATH="${MANIFEST_PATH:-docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100}"
DATA_DIR="${DATA_DIR:-gs://v6_east1d/datasets/exp02_overfit100/train100}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR}"
EXPECTED_WINDOWS="${EXPECTED_WINDOWS:-1629}"
NUM_TEXT_SLOTS="${NUM_TEXT_SLOTS:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250,500,1000,1750,2500}"
EXPECTED_COUNT="${EXPECTED_COUNT:-$EXPECTED_WINDOWS}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
VALIDATION_LOSS_OUTPUT_DIR="${VALIDATION_LOSS_OUTPUT_DIR:-}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"
SKIP_HF_PREFETCH="${SKIP_HF_PREFETCH:-0}"
# The training run's SHA is MANDATORY provenance for every queue-launched job (exp_01 F2): fail
# HERE, before python.
TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT must be set to the recorded training-run commit SHA (stamped into every val_loss row)}"
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

# The evaluator reads these three from the ENVIRONMENT (not pyconfig overrides).
export SMOKE_LIMIT
export TRAIN_COMMIT
export COMMIT

# The model repo + revision come from the MANIFEST (C1), never from this launcher.
if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "[overfit100-loss] FATAL: manifest not found at MANIFEST_PATH=${MANIFEST_PATH}" >&2
  exit 1
fi
MODEL_REPO="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['hf_repo'])" "${MANIFEST_PATH}")"
MODEL_REVISION="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['revision'])" "${MANIFEST_PATH}")"
if ! printf '%s' "${MODEL_REVISION}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "[overfit100-loss] FATAL: manifest revision '${MODEL_REVISION}' is not a 40-hex commit sha" >&2
  exit 1
fi

echo "RUN_NAME=${RUN_NAME}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "MODEL_REPO=${MODEL_REPO}"
echo "MODEL_REVISION=${MODEL_REVISION}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "EXPECTED_WINDOWS=${EXPECTED_WINDOWS}"
echo "NUM_TEXT_SLOTS=${NUM_TEXT_SLOTS}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
echo "EXPECTED_COUNT=${EXPECTED_COUNT}"
echo "VALIDATION_SEED=${VALIDATION_SEED}"
echo "VALIDATION_LOSS_OUTPUT_DIR=${VALIDATION_LOSS_OUTPUT_DIR:-config_default}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "SMOKE_LIMIT=${SMOKE_LIMIT:-<full run>}"
echo "TRAIN_COMMIT=${TRAIN_COMMIT}"
echo "SKIP_HF_PREFETCH=${SKIP_HF_PREFETCH}"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

# FULL-repo prefetch AT THE PINNED REVISION: the loss path needs the transformer and the T5 (the
# context table is built before the VAE/text modules are freed).
if [ "${SKIP_HF_PREFETCH}" != "1" ]; then
  HF_PREFETCH_REVISION="${MODEL_REVISION}" \
    bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_REPO}"
fi

if [ -z "${MODEL_DIR:-}" ]; then
  MODEL_DIR="$(python - "${MODEL_REPO}" "${MODEL_REVISION}" <<'PY'
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True))
PY
  )"
fi
echo "MODEL_DIR=${MODEL_DIR}"
case "${MODEL_DIR}" in
  *"${MODEL_REVISION}"*) ;;
  *)
    echo "[overfit100-loss] FATAL: resolved MODEL_DIR='${MODEL_DIR}' does not carry the pinned revision ${MODEL_REVISION}" >&2
    exit 1
    ;;
esac

python src/maxdiffusion/eval_wan_full_ft_val_loss.py \
  src/maxdiffusion/configs/base_wan_5b_overfit100.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  model_manifest_path="${MANIFEST_PATH}" \
  expected_model_revision="${MODEL_REVISION}" \
  train_data_dir="${DATA_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  expected_windows="${EXPECTED_WINDOWS}" \
  num_text_slots="${NUM_TEXT_SLOTS}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  validation_checkpoint_steps="${CHECKPOINT_STEPS}" \
  validation_expected_count="${EXPECTED_COUNT}" \
  validation_loss_output_dir="${VALIDATION_LOSS_OUTPUT_DIR}" \
  validation_seed="${VALIDATION_SEED}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
