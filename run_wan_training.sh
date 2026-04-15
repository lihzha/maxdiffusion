#!/bin/bash
# WAN 2.1 14B Training Script
# TPU v4-4 (4 chips, ~30.7GB HBM each)
#
# Usage:
#   bash run_wan_training.sh
#
# NOTE ON DATASET:
#   The tfrecords in wan_tfr_dataset_pusa_v1 were encoded at 1280x720x81 frames
#   (~75K sequence length), which requires ~46GB HBM — too large for 4xv4.
#   To train on real data you need tfrecords re-prepared at a smaller resolution,
#   e.g. 480x480x9 frames. Re-run wan_pusav1_to_tfrecords.py at that resolution
#   and point DATASET_DIR at the output.
#
#   To test with synthetic data at the working resolution, set:
#     DATASET_TYPE=synthetic
#   (default below)

set -e

# --- Environment ---
source ~/maxdiffusion_venv/bin/activate

# --- Paths ---
BUCKET_NAME=pi0-cot
RUN_NAME=jfacevedo-wan-v5p-8-${RANDOM}
OUTPUT_DIR=gs://$BUCKET_NAME/wan/
TFRECORDS_DATASET_DIR=wan_tfr_dataset_pusa_v1
DATASET_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/train/
EVAL_DATA_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/eval_timesteps/
SAVE_DATASET_DIR=gs://$BUCKET_NAME/${TFRECORDS_DATASET_DIR##*/}/save/

# Set to 'tfrecord' once you have tfrecords prepared at height x width x num_frames below
DATASET_TYPE=${DATASET_TYPE:-synthetic}

echo "RUN_NAME:         ${RUN_NAME}"
echo "OUTPUT_DIR:       ${OUTPUT_DIR}"
echo "DATASET_DIR:      ${DATASET_DIR} (used when DATASET_TYPE=tfrecord)"
echo "DATASET_TYPE:     ${DATASET_TYPE}"
echo ""

# --- XLA / TPU compiler flags ---
export LIBTPU_INIT_ARGS='--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
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
--xla_tpu_assign_all_reduce_scatter_layout=true'

# --- Training ---
python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_14b.yml \
  attention='flash' \
  weights_dtype=bfloat16 \
  activations_dtype=bfloat16 \
  guidance_scale=5.0 \
  flow_shift=5.0 \
  fps=16 \
  skip_jax_distributed_system=False \
  run_name="${RUN_NAME}" \
  output_dir="${OUTPUT_DIR}" \
  train_data_dir="${DATASET_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  dataset_save_location="${SAVE_DATASET_DIR}" \
  load_tfrecord_cached=True \
  dataset_type="${DATASET_TYPE}" \
  height=1280 \
  width=720 \
  num_frames=81 \
  num_inference_steps=50 \
  jax_cache_dir="${OUTPUT_DIR}/jax_cache/" \
  max_train_steps=1000 \
  checkpoint_every=100 \
  remat_policy='HIDDEN_STATE_WITH_OFFLOAD' \
  flash_min_seq_length=0 \
  seed=$RANDOM \
  per_device_batch_size=1 \
  ici_data_parallelism=1 \
  ici_fsdp_parallelism=8 \
  ici_tensor_parallelism=1
