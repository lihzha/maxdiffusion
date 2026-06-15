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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

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

RUN_NAME="${RUN_NAME:-wan-side-adapter-smoke}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/train}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-}"
LOG_PERIOD="${LOG_PERIOD:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-True}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"
GLOBAL_BATCH_SIZE_TO_TRAIN_ON="${GLOBAL_BATCH_SIZE_TO_TRAIN_ON:-1}"
GLOBAL_BATCH_SIZE_TO_LOAD="${GLOBAL_BATCH_SIZE_TO_LOAD:-$GLOBAL_BATCH_SIZE_TO_TRAIN_ON}"
TFRECORD_SHUFFLE_BUFFER_SIZE="${TFRECORD_SHUFFLE_BUFFER_SIZE:-}"
WANDB_PROJECT="${WANDB_PROJECT-}"

echo "RUN_NAME=${RUN_NAME}"
echo "TRAIN_DATA_DIR=${TRAIN_DATA_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-config_default}"
echo "LOG_PERIOD=${LOG_PERIOD}"
echo "SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT}"
echo "GLOBAL_BATCH_SIZE_TO_TRAIN_ON=${GLOBAL_BATCH_SIZE_TO_TRAIN_ON}"
echo "GLOBAL_BATCH_SIZE_TO_LOAD=${GLOBAL_BATCH_SIZE_TO_LOAD}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "TFRECORD_SHUFFLE_BUFFER_SIZE=${TFRECORD_SHUFFLE_BUFFER_SIZE:-config_default}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "COMMIT=$(git rev-parse HEAD)"
git status --short --branch

python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_5b_side_adapter.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  train_data_dir="${TRAIN_DATA_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  max_train_steps="${MAX_TRAIN_STEPS}" \
  checkpoint_every="${CHECKPOINT_EVERY}" \
  eval_every="${EVAL_EVERY}" \
  ${EVAL_MAX_BATCHES:+eval_max_batches="${EVAL_MAX_BATCHES}"} \
  log_period="${LOG_PERIOD}" \
  save_final_checkpoint="${SAVE_FINAL_CHECKPOINT}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  global_batch_size_to_train_on="${GLOBAL_BATCH_SIZE_TO_TRAIN_ON}" \
  global_batch_size_to_load="${GLOBAL_BATCH_SIZE_TO_LOAD}" \
  ${TFRECORD_SHUFFLE_BUFFER_SIZE:+tfrecord_shuffle_buffer_size="${TFRECORD_SHUFFLE_BUFFER_SIZE}"} \
  wandb_project="${WANDB_PROJECT}" \
  hardware=tpu
