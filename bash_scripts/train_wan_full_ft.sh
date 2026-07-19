#!/usr/bin/env bash
set -euo pipefail

# Raise the open-file limit. The queue worker's login shell defaults to 1024,
# which the model + data pipeline exhaust (process 0 hit Errno 24 at import wandb).
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

# --- Full-finetune overfit probe (docs/worklogs_yixun/exp_01_full_ft_overfit_claude/
#     plan_full_ft_overfit.md) ---
# Trains the FULL Wan2.2 TI2V 5B backbone (NO adapter) via train_wan.py ->
# WanTI2VFullFTTrainer, dispatched by model_type: FULL_FT_TI2V in the yml below.
# Deltas vs train_wan_side_adapter.sh: (1) points at base_wan_5b_full_ft.yml; (2) defaults
# SIDE_ADAPTER_NOISE_MODE=fresh -- the adapter template's `fixed` default is the documented
# foot-gun that yields good-looking train loss but broken generations, and the full-FT
# trainer ASSERTS fresh; (3) defaults EVAL_DATA_DIR to the TRAIN split (in-training eval on
# train, plan §2.2/F7); (4) exposes LEARNING_RATE and drops the inert adapter knobs.
#
# Batch contract (round-5 F1; amended per plan §2.2 v3.1 / Query 7): the yml itself
# defaults to the PRIMARY-RUN recipe (per_device_batch_size 4.0 -> GBS 256 on v6e-64 --
# per-device 8 OOMs there for full-FT, fit probe #4; pyconfig recomputes the global batch
# from per_device x device_count), so the plain `python train_wan.py <yml> run_name=...`
# command satisfies plan §6 with no overrides. THIS WRAPPER's batch defaults stay
# SMOKE-SCALED (per_device 1.0 / GBS 1) because a bare wrapper invocation is a dev smoke;
# it always passes them as explicit CLI overrides (never silently inheriting the yml's
# 256-recipe), and the launcher full_ft arm exports PER_DEVICE_BATCH_SIZE=4 /
# GLOBAL_BATCH_SIZE_TO_TRAIN_ON=256 / GLOBAL_BATCH_SIZE_TO_LOAD=256 for real queue runs.
# Same for W&B: the yml defaults wandb_project=maxdiffusion-wan-full-ft (plain command
# logs), the launcher exports the identical project for queue runs, and a bare wrapper
# run passes empty unless WANDB_PROJECT is exported (quiet dev smoke).

RUN_NAME="${RUN_NAME:-wan-full-ft-smoke}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/train}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$TRAIN_DATA_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
SIDE_ADAPTER_NOISE_MODE="${SIDE_ADAPTER_NOISE_MODE:-fresh}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"
CHECKPOINT_KEEP_PERIOD="${CHECKPOINT_KEEP_PERIOD:-2500}"
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
echo "SIDE_ADAPTER_NOISE_MODE=${SIDE_ADAPTER_NOISE_MODE}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "CHECKPOINT_KEEP_PERIOD=${CHECKPOINT_KEEP_PERIOD}"
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
echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"
echo "COMMIT=${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_5b_full_ft.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  train_data_dir="${TRAIN_DATA_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  side_adapter_noise_mode="${SIDE_ADAPTER_NOISE_MODE}" \
  learning_rate="${LEARNING_RATE}" \
  max_train_steps="${MAX_TRAIN_STEPS}" \
  checkpoint_every="${CHECKPOINT_EVERY}" \
  checkpoint_keep_period="${CHECKPOINT_KEEP_PERIOD}" \
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
