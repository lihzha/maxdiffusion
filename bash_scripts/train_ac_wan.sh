# --- 1. Activate the training env ---
set -e
echo "[$(hostname)] Script started at $(date)"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # add uv to PATH

uv venv --python 3.12 ./maxdiffusion_venv --seed
source ./maxdiffusion_venv/bin/activate
bash setup.sh MODE=stable DEVICE=tpu

# --- 2. Bucket mount ---
# source ~/.zshrc
# : "${WANDB_API_KEY:?WANDB_API_KEY is not set. Run 'wandb login' or export WANDB_API_KEY=<your-key>.}"
export WANDB_API_KEY=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
export GCS_BUCKET=v6_east1d
export GCS_MOUNT=/home/zheng/gcs-mount

if ! command -v gcsfuse >/dev/null; then
  DISTRO=$(lsb_release -cs 2>/dev/null)
  [ -z "$DISTRO" ] && DISTRO=$(. /etc/os-release && echo "$VERSION_CODENAME")
  export GCSFUSE_REPO="gcsfuse-${DISTRO}"
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get -o DPkg::Lock::Timeout=-1 update
  sudo apt-get -o DPkg::Lock::Timeout=-1 install -y gcsfuse || { echo "gcsfuse install failed on $(hostname)"; exit 1; }
fi

mkdir -p "$GCS_MOUNT" /dev/shm/gcsfuse-cache
if ! mountpoint -q "$GCS_MOUNT"; then
  gcsfuse \
    --implicit-dirs \
    --file-cache-max-size-mb=-1 \
    --cache-dir=/dev/shm/gcsfuse-cache \
    "$GCS_BUCKET" "$GCS_MOUNT"
fi

# --- 3. TI2V model path ---
export WAN_TI2V_MODEL_DIR="$GCS_MOUNT/wan/Wan2.2-TI2V-5B-Diffusers"

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

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
# python src/maxdiffusion/train_wan.py \
#     src/maxdiffusion/configs/base_wan_i2v_14b.yml \
#     run_name=i2v-test-run-1 \
#     output_dir=gs://v6_east1d/i2v-test-run-1 \
#     pretrained_model_name_or_path=$WAN_TI2V_MODEL_DIR \
#     dataset_type=synthetic \
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

source ./maxdiffusion_venv/bin/activate
ulimit -n 65536

# --- TFRecord path (uncomment to use) ---
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_wan.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    run_name=ac_wan_droid_history_fewer_frames_downsampled_zero_init \
    output_dir=gs://v6_east1d/checkpoints/wan-ac \
    pretrained_model_name_or_path=$WAN_TI2V_MODEL_DIR \
    dataset_type=tfrecord \
    train_data_dir=gs://v6_east1d/wan2.2_tfr_dataset_lowres_downsampled/train \
    eval_data_dir=gs://v6_east1d/wan2.2_tfr_dataset_lowres_downsampled/val \
    action_stats_path=gs://v6_east1d/wan2.2_tfr_dataset_lowres_downsampled/stats.json \
    cache_latents_text_encoder_outputs=True \
    attention=tokamax_flash \
    weights_dtype=float32 \
    activations_dtype=bfloat16 \
    remat_policy=HIDDEN_STATE_WITH_OFFLOAD \
    ici_data_parallelism=1 \
    ici_fsdp_parallelism=-1 \
    ici_tensor_parallelism=1 \
    ici_context_parallelism=1 \
    dcn_data_parallelism=1 \
    dcn_fsdp_parallelism=1 \
    dcn_tensor_parallelism=1 \
    dcn_context_parallelism=1 \
    per_device_batch_size=1.0 \
    grad_accum_steps=1 \
    allow_split_physical_axes=True \
    scan_layers=True \
    learning_rate=1e-5 \
    warmup_steps_fraction=0.05 \
    learning_rate_schedule_type=cosine \
    learning_rate_end_ratio=0.0 \
    max_train_steps=101000 \
    checkpoint_every=100 \
    checkpoint_keep_period=10000 \
    eval_every=1000 \
    height=480 \
    width=832 \
    num_predicted_latents=10 \
    num_history_latent_frames=10 \
    history_noise_max_timestep=200 \
    flash_min_seq_length=128 \
    hardware='tpu' \
    wandb_project='wan-ac-history-fewer-frames-no-text-downsampled-zero-init' \
    wandb_video_every=1000 \
    wandb_video_samples=1 \
    wandb_video_inference_steps=20

# --- DROID path (uncomment to use) ---
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
# python src/maxdiffusion/train_wan.py \
#     src/maxdiffusion/configs/base_wan_i2v_14b.yml \
#     run_name=i2v-droid-run-1 \
#     output_dir=gs://v6_east1d/i2v-droid-run-1 \
#     pretrained_model_name_or_path=$WAN_TI2V_MODEL_DIR \
#     dataset_type=droid \
#     train_data_dir=gs://v6_east1d/OXE \
#     droid_clip_stride=8 \
#     attention=flash \
#     weights_dtype=bfloat16 \
#     activations_dtype=bfloat16 \
#     remat_policy=FULL \
#     ici_fsdp_parallelism=8 \
#     ici_data_parallelism=1 \
#     ici_tensor_parallelism=1 \
#     ici_context_parallelism=1 \
#     scan_layers=True \
#     max_train_steps=1000 \
#     per_device_batch_size=0.25 \
#     height=480 \
#     width=832 \
#     num_frames=49 \
#     flash_min_seq_length=0

# --- 6. Unmount ---
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"

# tpu create v6 --name train_ac_wan -n 64 --setup-cmd "" --priority 0 --max-attempts 100 -- bash bash_scripts/train_ac_wan.sh