# --- 1. Activate the training env ---
curl -LsSf https://astral.sh/uv/install.sh | sh
git checkout origin/catherine-dev

uv venv --python 3.12 ./maxdiffusion_venv --seed
source ./maxdiffusion_venv/bin/activate
bash setup.sh MODE=stable DEVICE=tpu

# --- 2. Bucket mount ---
source ~/.zshrc
: "${WANDB_API_KEY:?WANDB_API_KEY is not set. Run 'wandb login' or export WANDB_API_KEY=<your-key>.}"
export GCS_BUCKET=v6_east1d
export GCS_MOUNT=/home/zheng/gcs-mount

if ! command -v gcsfuse >/dev/null; then
  DISTRO=$(lsb_release -cs 2>/dev/null)
  [ -z "$DISTRO" ] && DISTRO=$(. /etc/os-release && echo "$VERSION_CODENAME")
  export GCSFUSE_REPO="gcsfuse-${DISTRO}"
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get -o DPkg::Lock::Timeout=-1 update
  sudo apt-get -o DPkg::Lock::Timeout=-1 install -y gcsfuse
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

# --- 5. Launch TI2V training ---
#
# dataset_type options:
#   synthetic  — no data needed; smoke-test that the train step compiles
#   tfrecord   — pre-encoded latents; set train_data_dir to TFRecord GCS path
#                TFRecords must contain: latents (C,F,H,W), encoder_hidden_states (512,4096)
#                Clips should be longer than the training window (1 + num_frames//4 latent frames)
#                to benefit from random temporal windowing.
#
# Uncomment the desired block below.

# --- Synthetic path (smoke-test) ---
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
# python src/maxdiffusion/train_wan.py \
#     src/maxdiffusion/configs/base_wan_5b.yml \
#     run_name=ti2v-synthetic-test \
#     output_dir=gs://v6_east1d/checkpoints/wan-ti2v-synthetic \
#     pretrained_model_name_or_path=$WAN_TI2V_MODEL_DIR \
#     dataset_type=synthetic \
#     attention=flash \
#     weights_dtype=bfloat16 \
#     activations_dtype=bfloat16 \
#     remat_policy=FULL \
#     ici_data_parallelism=2 \
#     ici_fsdp_parallelism=4 \
#     ici_tensor_parallelism=1 \
#     ici_context_parallelism=4 \
#     allow_split_physical_axes=True \
#     scan_layers=True \
#     max_train_steps=100 \
#     per_device_batch_size=0.25 \
#     height=720 \
#     width=1280 \
#     num_frames=80 \
#     flash_min_seq_length=128 \
#     hardware='tpu'

# --- TFRecord path (DROID data converted via wan_convert.sh) ---
# Cameras are stacked along H: effective latent height = 704/8 * 3 = 264.
# Eval samples timesteps uniformly (no timesteps field in DROID records).
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_wan.py \
    src/maxdiffusion/configs/base_wan_5b.yml \
    run_name=ti2v_wan_droid_single_view \
    output_dir=gs://v6_east1d/checkpoints/wan-ti2v-finetune \
    pretrained_model_name_or_path=$WAN_TI2V_MODEL_DIR \
    dataset_type=tfrecord \
    train_data_dir=gs://v6_east1d/wan2.2_tfr_dataset_lowres/train \
    eval_data_dir=gs://v6_east1d/wan2.2_tfr_dataset_lowres/val \
    cache_latents_text_encoder_outputs=True \
    eval_every=1000 \
    attention=flash \
    weights_dtype=bfloat16 \
    activations_dtype=bfloat16 \
    remat_policy=FULL \
    ici_data_parallelism=1 \
    ici_fsdp_parallelism=-1 \
    ici_tensor_parallelism=1 \
    ici_context_parallelism=1 \
    dcn_data_parallelism=1 \
    dcn_fsdp_parallelism=1 \
    dcn_tensor_parallelism=1 \
    dcn_context_parallelism=1 \
    allow_split_physical_axes=True \
    scan_layers=True \
    max_train_steps=100100 \
    checkpoint_every=100 \
    checkpoint_keep_period=10000 \
    per_device_batch_size=1.0 \
    height=480 \
    width=832 \
    num_frames=80 \
    flash_min_seq_length=128 \
    num_privileged_frames=0 \
    hardware='tpu' \
    ema_decay=0.0 \
    distill=False \
    wandb_project='wan-ti2v-finetune_single_view_2' \
    single_camera=True

# --- 6. Unmount ---
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"


# 1.6 seconds/sec
# remat_policy=MATMUL_WITHOUT_BATCH \
# ici_data_parallelism=1 \
# ici_fsdp_parallelism=-1 \
# ici_tensor_parallelism=1 \
# ici_context_parallelism=1 \
# per_device_batch_size=1.0 \

# tpu create v6 --name wan_ti2v -n 64 --setup-cmd "export WANDB_API_KEY=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 && bash bash_scripts/train_ti2v_wan_single.sh"
# tpu tmux v6-64-02-catherine -- 'cd maxdiffusion && git checkout origin/catherine-dev && git pull origin catherine-dev && bash bash_scripts/train_ti2v_wan_single.sh' Enter