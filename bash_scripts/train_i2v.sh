# --- 1. Activate the training env ---
source ~/.zshrc
source ~/maxdiffusion_venv/bin/activate
cd ~/maxdiffusion

# --- 2. Bucket mount ---
export GCS_BUCKET=v6_east1d
export GCS_MOUNT=/home/irom-lab/gcs-mount

if ! command -v gcsfuse >/dev/null; then
  export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install -y gcsfuse
fi

mkdir -p "$GCS_MOUNT" /dev/shm/gcsfuse-cache
if ! mountpoint -q "$GCS_MOUNT"; then
  gcsfuse \
    --implicit-dirs \
    --file-cache-max-size-mb=-1 \
    --cache-dir=/dev/shm/gcsfuse-cache \
    "$GCS_BUCKET" "$GCS_MOUNT"
fi

# --- 3. Resolve the I2V model snapshot dir ---
export WAN_MODEL_DIR="$(ls -d $GCS_MOUNT/wan/wan-diffusers/snapshots/*/* 2>/dev/null | head -1)"
if [ -z "$WAN_MODEL_DIR" ]; then
  echo "ERROR: could not find WAN model snapshot under $GCS_MOUNT/wan/wan-diffusers/snapshots/"
  exit 1
fi
echo "Using WAN_MODEL_DIR=$WAN_MODEL_DIR"

# --- 4. XLA flags ---
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

# --- 5. Launch I2V training ---
#
# dataset_type options:
#   synthetic  — no data needed; smoke-test that the train step compiles
#   tfrecord   — pre-encoded latents/condition; set train_data_dir to TFRecord GCS path
#   droid      — on-the-fly encoding from DROID TFDS records; set train_data_dir to TFDS parent dir
#
# Uncomment and set the right dataset_type / train_data_dir for your run.

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_wan.py \
    src/maxdiffusion/configs/base_wan_i2v_14b.yml \
    run_name=i2v-test-run-1 \
    output_dir=gs://v6_east1d/i2v-test-run-1 \
    pretrained_model_name_or_path=$WAN_MODEL_DIR \
    dataset_type=synthetic \
    attention=flash \
    weights_dtype=bfloat16 \
    activations_dtype=bfloat16 \
    remat_policy=FULL \
    ici_fsdp_parallelism=2 \
    ici_data_parallelism=1 \
    ici_tensor_parallelism=1 \
    ici_context_parallelism=4 \
    scan_layers=True \
    max_train_steps=1000 \
    per_device_batch_size=0.25 \
    height=720 \
    width=1280 \
    num_frames=81 \
    flash_min_seq_length=0

# --- TFRecord path (uncomment to use) ---
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
# python src/maxdiffusion/train_wan.py \
#     src/maxdiffusion/configs/base_wan_i2v_14b.yml \
#     run_name=i2v-tfrecord-run-1 \
#     output_dir=gs://v6_east1d/i2v-tfrecord-run-1 \
#     pretrained_model_name_or_path=$WAN_MODEL_DIR \
#     dataset_type=tfrecord \
#     train_data_dir=gs://v6_east1d/wan_i2v_tfrecords/train \
#     cache_latents_text_encoder_outputs=True \
#     attention=flash \
#     weights_dtype=bfloat16 \
#     activations_dtype=bfloat16 \
#     remat_policy=FULL \
#     ici_fsdp_parallelism=2 \
#     ici_data_parallelism=1 \
#     ici_tensor_parallelism=1 \
#     ici_context_parallelism=4 \
#     scan_layers=True \
#     max_train_steps=1000 \
#     per_device_batch_size=0.25 \
#     height=720 \
#     width=1280 \
#     num_frames=81 \
#     flash_min_seq_length=0

# --- DROID path (uncomment to use) ---
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
# python src/maxdiffusion/train_wan.py \
#     src/maxdiffusion/configs/base_wan_i2v_14b.yml \
#     run_name=i2v-droid-run-1 \
#     output_dir=gs://v6_east1d/i2v-droid-run-1 \
#     pretrained_model_name_or_path=$WAN_MODEL_DIR \
#     dataset_type=droid \
#     train_data_dir=/path/to/tfds_parent/ \
#     droid_clip_stride=8 \
#     attention=flash \
#     weights_dtype=bfloat16 \
#     activations_dtype=bfloat16 \
#     remat_policy=FULL \
#     ici_fsdp_parallelism=2 \
#     ici_data_parallelism=1 \
#     ici_tensor_parallelism=1 \
#     ici_context_parallelism=4 \
#     scan_layers=True \
#     max_train_steps=1000 \
#     per_device_batch_size=0.25 \
#     height=480 \
#     width=832 \
#     num_frames=49 \
#     flash_min_seq_length=0

# --- 6. Unmount ---
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"
