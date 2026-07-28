#!/usr/bin/env bash
set -euo pipefail

# Raise the open-file limit. The queue worker's login shell defaults to 1024,
# which the 5B model + adapter + data pipeline + tensorstore checkpoint restore
# exhaust (same Errno 24 / "Too many open files" that aborted the train smoke).
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

# --- Full-finetune cohort validation (docs/worklogs_yixun/exp_01_full_ft_overfit_claude/
#     plan_full_ft_overfit.md §2.3) ---
# Rolls out the memorization cohort on generate_wan_side_adapter.py against the FULL-FT yml
# (model_type: FULL_FT_TI2V -> plain-transformer rollout, no adapter). Deltas vs
# validate_wan_side_adapter.sh: (1) points at base_wan_5b_full_ft.yml; (2) EVAL_DATA_DIR
# defaults to the TRAIN split (the cohort is training windows); (3) passes VALIDATION_ORDINALS
# (the fixed noncontiguous cohort; empty -> contiguous fallback); (4) drops the inert adapter
# knobs. checkpoint_step selects the baseline/checkpoint (0 = pretrained; 2500/5000/7500/10000
# = retained cohort checkpoints; -1 = latest).

RUN_NAME="${RUN_NAME:?RUN_NAME must match the training run to validate}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/train}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:--1}"
NUM_EVAL_VIDEOS="${NUM_EVAL_VIDEOS:-4}"
VALIDATION_ORDINALS="${VALIDATION_ORDINALS:-}"
VALIDATION_START_INDEX="${VALIDATION_START_INDEX:-0}"
VALIDATION_SEED="${VALIDATION_SEED:-0}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/validation}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"

echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "CHECKPOINT_STEP=${CHECKPOINT_STEP}"
echo "NUM_EVAL_VIDEOS=${NUM_EVAL_VIDEOS}"
echo "VALIDATION_ORDINALS=${VALIDATION_ORDINALS:-config_default}"
echo "VALIDATION_START_INDEX=${VALIDATION_START_INDEX}"
echo "VALIDATION_SEED=${VALIDATION_SEED}"
echo "VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"
echo "COMMIT=${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

python src/maxdiffusion/generate_wan_side_adapter.py \
  src/maxdiffusion/configs/base_wan_5b_full_ft.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  checkpoint_step="${CHECKPOINT_STEP}" \
  num_eval_videos="${NUM_EVAL_VIDEOS}" \
  validation_ordinals="${VALIDATION_ORDINALS}" \
  validation_start_index="${VALIDATION_START_INDEX}" \
  validation_seed="${VALIDATION_SEED}" \
  validation_output_dir="${VALIDATION_OUTPUT_DIR}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
