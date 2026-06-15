#!/usr/bin/env bash
set -euo pipefail

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

RUN_NAME="${RUN_NAME:?RUN_NAME must match the training run to validate}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:--1}"
NUM_EVAL_VIDEOS="${NUM_EVAL_VIDEOS:-4}"
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
echo "VALIDATION_START_INDEX=${VALIDATION_START_INDEX}"
echo "VALIDATION_SEED=${VALIDATION_SEED}"
echo "VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "COMMIT=$(git rev-parse HEAD)"
git status --short --branch

python src/maxdiffusion/generate_wan_side_adapter.py \
  src/maxdiffusion/configs/base_wan_5b_side_adapter.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  checkpoint_step="${CHECKPOINT_STEP}" \
  num_eval_videos="${NUM_EVAL_VIDEOS}" \
  validation_start_index="${VALIDATION_START_INDEX}" \
  validation_seed="${VALIDATION_SEED}" \
  validation_output_dir="${VALIDATION_OUTPUT_DIR}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
