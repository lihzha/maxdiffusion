# Launch action-conditioned SVD (Ctrl-World) training on TPU with CAMERA-FRAME
# action conditioning (action_cond_mode=cam_action_cross_attn). The 7-dim action is
# re-expressed in each camera's frame (EEF position + 6D rotation + gripper),
# anchored at the conditioning frame (frame_now) so the wrist view gets its
# ego-motion, and encoded once per camera. Site for THIS run:
#   cam_action_cross_attn -> the SPATIAL cross-attention K/V, locked per camera: each
#                            view's rows attend only to [text (zeros here), that
#                            view's action] (the 'skeleton_cross_attn' site)
#
# Hyperparameters are identical to train_ctrl_world_skeleton_adaln_no_text.sh so
# the camera-frame and skeleton runs differ only in the conditioning.
#
# NOT checkpoint-compatible with any other action_cond_mode: this tree carries a
# cam_action_encoder subtree (+ its projection) and NO action_encoder. Start a
# fresh RUN_TAG (section 3b).
#
# Pre-requisites (one-time):
#   1. Shards carrying ee_pose_cam0/1/2 beside latent_cam0/1/2 (see
#      camera_frame_actions.py for the contract), uploaded under $CAM_DATA.
#   2. $CAM_DATA/cam_action_stats.json, written once with the same window:
#        python src/maxdiffusion/compute_cam_action_stats.py \
#            src/maxdiffusion/configs/base_ctrl_world.yml \
#            train_data_dir=$CAM_DATA/train stats_path=$CAM_DATA/stats.json \
#            cam_action_stats_path=$CAM_DATA/cam_action_stats.json \
#            num_history=7 num_frames=5
#
# Resuming: re-run this script verbatim; the trainer picks up the latest
# checkpoint under $output_dir/checkpoints. Starting fresh: bump RUN_TAG.

# --- 1. Activate the training env ---
set -e
echo "[$(hostname)] Script started at $(date)"
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"   # add uv to PATH

uv venv --python 3.12 ./maxdiffusion_venv --seed
source ./maxdiffusion_venv/bin/activate
bash setup.sh MODE=stable DEVICE=tpu

# --- 2. Bucket mount ---
export WANDB_API_KEY=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
export GCS_BUCKET=v6_east1d
export GCS_MOUNT=/home/zheng/gcs-mount

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

# --- 3. Model paths ---
# Default: cold-start UNet/VAE from the upstream SVD repo on disk. Point this at
# the directory produced by scripts/convert_ctrl_world_ckpt.py to warm-start
# from a Ctrl-World torch checkpoint.
export SVD_MODEL_DIR="$GCS_MOUNT/svd/svd_checkpoint"
if [ ! -f "$SVD_MODEL_DIR/unet/config.json" ]; then
  echo "ERROR: SVD model not found at $SVD_MODEL_DIR/unet/config.json — download it first."
  echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-video-diffusion-img2vid', local_dir='/tmp/svd')\""
  echo "  gsutil -m cp -r /tmp/svd gs://$GCS_BUCKET/svd/stable-video-diffusion-img2vid"
  exit 1
fi
echo "Using SVD_MODEL_DIR=$SVD_MODEL_DIR"
echo "Camera-action encoder: cold start (zero-init output projection)."

# --- 3b. Run identity ---
# A genuinely fresh run needs its own tag; RUN_TAG also names the W&B run.
export RUN_TAG="${RUN_TAG:-cam-action-cross-attn-no-text}"
export OUTPUT_DIR="gs://$GCS_BUCKET/checkpoints/svd_ac"
echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# --- 4. Data paths (camera-pose shards: ee_pose_cam0/1/2) ---
# Point CAM_DATA at wherever the camera-pose build is uploaded.
export CAM_DATA="gs://$GCS_BUCKET/datasets/droid_wan_2.2_skeleton_192_320_cam_action"
export TRAIN_DATA_DIR="$CAM_DATA/train"
export EVAL_DATA_DIR="$CAM_DATA/val"
export STATS_PATH="$CAM_DATA/stats.json"
export CAM_STATS_PATH="$CAM_DATA/cam_action_stats.json"

# --- 5. XLA flags ---
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

# --- 6. Launch training ---
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_ctrl_world.py \
    src/maxdiffusion/configs/base_ctrl_world.yml \
    action_cond_mode='cam_action_cross_attn' \
    cam_action_stats_path=$CAM_STATS_PATH \
    run_name=$RUN_TAG \
    output_dir=$OUTPUT_DIR \
    pretrained_model_name_or_path=$SVD_MODEL_DIR \
    action_encoder_init_path='' \
    dataset_type=ctrl_world \
    train_data_dir=$TRAIN_DATA_DIR \
    eval_data_dir=$EVAL_DATA_DIR \
    stats_path=$STATS_PATH \
    attention=flash \
    weights_dtype=float32 \
    activations_dtype=float32 \
    remat_policy=MATMUL_WITHOUT_BATCH \
    ici_fsdp_parallelism=-1 \
    ici_data_parallelism=1 \
    ici_tensor_parallelism=1 \
    ici_context_parallelism=1 \
    scan_layers=True \
    max_train_steps=100000 \
    learning_rate=1e-5 \
    per_device_batch_size=1.0 \
    num_history=7 \
    num_frames=5 \
    action_dim=7 \
    text_embed_dim=512 \
    checkpoint_every=1000 \
    eval_every=1000 \
    eval_max_batches=50 \
    save_optimizer=True \
    checkpoint_max_to_keep=3 \
    reshuffle_data_on_restart=True \
    wandb_project='svd-ac-cam-action-cross-attn-no-text' \
    wandb_video_every=2000 \
    use_task_instructions=False 

# --- 7. Unmount ---
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"

# tpu create v6 --name train_ac_svd_cam_action_cross_attn_no_text -n 32 --setup-cmd "" --priority 0 --max-attempts 40 -- bash bash_scripts/train_ctrl_world_cam_action_cross_attn_no_text.sh
