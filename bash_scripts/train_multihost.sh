# --- 5. Activate the training env ---
source ~/.zshrc
source ~/maxdiffusion_venv/bin/activate
cd ~/maxdiffusion

# --- 1. Bucket mount (paths and bucket name) ---
export GCS_BUCKET=v6_east1d
export GCS_MOUNT=/home/irom-lab/gcs-mount

# --- 2. Install gcsfuse (first time only; re-running is a no-op) ---
if ! command -v gcsfuse >/dev/null; then
  export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install -y gcsfuse
fi

# --- 3. Mount the bucket (cache on /dev/shm because / is full) ---
mkdir -p "$GCS_MOUNT" /dev/shm/gcsfuse-cache
# Always unmount first to clear any stale handle from a previous killed run.
fusermount -uz "$GCS_MOUNT" 2>/dev/null || true
gcsfuse \
  --implicit-dirs \
  --file-cache-max-size-mb=-1 \
  --cache-dir=/dev/shm/gcsfuse-cache \
  "$GCS_BUCKET" "$GCS_MOUNT"

# --- 4. Resolve the actual snapshot dir containing config.json ---
export WAN_MODEL_DIR="$(ls -d $GCS_MOUNT/wan/wan-diffusers/snapshots/*/* | head -1)"
echo "Using WAN_MODEL_DIR=$WAN_MODEL_DIR"


# --- 6. Launch training (reads weights from the mount, not /tmp) ---
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

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_wan.py \
    src/maxdiffusion/configs/base_wan_14b.yml \
    run_name=test-run-64 \
    output_dir=gs://v6_east1d/test-run-64 \
    pretrained_model_name_or_path=$WAN_MODEL_DIR \
    train_data_dir=gs://v6_east1d/wan_tfr_dataset_pusa_v1_tenny/train \
    attention=flash \
    weights_dtype=bfloat16 \
    activations_dtype=bfloat16 \
    remat_policy=FULL \
    ici_fsdp_parallelism=16 \
    ici_data_parallelism=1 \
    ici_tensor_parallelism=1 \
    ici_context_parallelism=4 \
    scan_layers=True \
    max_train_steps=1000 \
    per_device_batch_size=1 \
    global_batch_size=256 \
    height=1280 \
    width=720 \
    num_frames=81 \
    flash_min_seq_length=0 \
    allow_split_physical_axes=True

# unmount
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"