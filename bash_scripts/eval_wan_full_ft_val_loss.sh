#!/usr/bin/env bash
set -euo pipefail

# Raise the open-file limit. The queue worker's login shell defaults to 1024,
# which the 5B model + optimizer state + tensorstore checkpoint restore exhaust
# (same Errno 24 / "Too many open files" that aborted the train smoke).
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

# --- Full-finetune full-VALIDATION one-step-loss sweep (docs/worklogs_yixun/
#     exp_01_full_ft_overfit_claude/plan_full_ft_overfit.md Part II, T1) ---
# Runs eval_wan_full_ft_val_loss.py against the FULL-FT yml (model_type: FULL_FT_TI2V). For each
# checkpoint step it computes the EXACT training objective (velocity MSE, frame-0 masked) over ALL
# held-out windows with per-example (t, eps) held fixed across checkpoints, then writes
# val_loss.{json,csv} + val_loss_plot.png under {output_dir}/{run_name}/validation_loss. Deltas vs
# validate_wan_full_ft.sh: (1) invokes the evaluator, not the rollout script; (2) EVAL_DATA_DIR
# defaults to the VAL split; (3) passes validation_checkpoint_steps / validation_expected_count.
#
# SMOKE gate (plan F5): set SMOKE_LIMIT=<N> to evaluate ONLY the first N batches of ONLY the first
# checkpoint (isolated validation_loss_smoke/ output; the n==expected assertion is skipped) as the
# storage-light fit probe before the full 8-checkpoint pass.

RUN_NAME="${RUN_NAME:?RUN_NAME must match the training run to evaluate}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-2500,5000,7500,10000,12500,15000,17500,20000}"
EXPECTED_COUNT="${EXPECTED_COUNT:-14636}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
VALIDATION_LOSS_OUTPUT_DIR="${VALIDATION_LOSS_OUTPUT_DIR:-}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
SMOKE_LIMIT="${SMOKE_LIMIT:-}"
# F2 (cycle-B review): the training run's SHA is MANDATORY provenance for every queue-launched
# job -- fail HERE, before python, rather than let the evaluator start. (The module additionally
# refuses FULL runs without it; only the module-level SMOKE path is exempt, by design.)
TRAIN_COMMIT="${TRAIN_COMMIT:?TRAIN_COMMIT must be set to the recorded training-run commit SHA (stamped into every val_loss row; plan D5/F6)}"
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"

# The evaluator reads these three from the ENVIRONMENT (not pyconfig overrides): SMOKE_LIMIT gates
# the smoke subset, TRAIN_COMMIT stamps the run's training SHA, COMMIT stamps the eval-code SHA.
export SMOKE_LIMIT
export TRAIN_COMMIT
export COMMIT

echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
echo "EXPECTED_COUNT=${EXPECTED_COUNT}"
echo "VALIDATION_SEED=${VALIDATION_SEED}"
echo "VALIDATION_LOSS_OUTPUT_DIR=${VALIDATION_LOSS_OUTPUT_DIR:-config_default}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "SMOKE_LIMIT=${SMOKE_LIMIT:-<full run>}"
echo "TRAIN_COMMIT=${TRAIN_COMMIT}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

python src/maxdiffusion/eval_wan_full_ft_val_loss.py \
  src/maxdiffusion/configs/base_wan_5b_full_ft.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  validation_checkpoint_steps="${CHECKPOINT_STEPS}" \
  validation_expected_count="${EXPECTED_COUNT}" \
  validation_loss_output_dir="${VALIDATION_LOSS_OUTPUT_DIR}" \
  validation_seed="${VALIDATION_SEED}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
