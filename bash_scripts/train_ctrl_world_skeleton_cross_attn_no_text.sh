# Launch action-conditioned SVD (Ctrl-World) training on TPU with AdaLN action
# conditioning (action_cond_mode=adaln): action tokens are projected per frame
# and summed into the UNet's timestep embedding, the per-frame cross-attention
# route is dropped, and cross-attention carries the text embedding on its own.
# For the original per-frame cross-attention conditioning use
# bash_scripts/train_ctrl_world.sh.
#
# action_cond_mode=adaln ALSO switches every spatial and temporal resnet from
# the default additive time-embedding injection (``norm2(h + shift)``, whose
# shift GroupNorm largely normalises away) to AdaGN
# (``norm2(h) * (1 + scale) + shift``). That is deliberate: t_emb is the action
# pathway in this mode, and AdaGN is the UNet analogue of the multiplicative
# AdaLN modulation the WAN arm uses, so the two arms are comparable. It costs
# +51.6M params (one zero-init Dense per resnet, +3.4%) and ~0.6 GB of extra
# fp32 AdamW state. cross_attn mode is unaffected and stays bit-identical to
# pretrained SVD.
#
# NOT checkpoint-compatible with adaln runs from before AdaGN landed: the tree
# gains 88 adagn_scale_proj leaves, so an old checkpoint fails the orbax restore.
# Start a fresh RUN_TAG (section 3b).
#
# Pre-requisites (one-time):
#   1. Pre-encoded data uploaded to gs://<bucket>/ctrl_world_droid/{train,val}/
#      and gs://<bucket>/ctrl_world_droid/stats.json. See
#      docs/ctrl_world_data_format.md for the schema.
#   2. The action encoder is ALWAYS cold-started here; adaln has no warm-start
#      path (see the note above the launch command). scripts/convert_ctrl_world_ckpt.py
#      and its action_encoder.safetensors apply to the cross_attn script only.
#
# Resuming: re-run this script verbatim. Checkpoints live in
# $output_dir/checkpoints (unless checkpoint_dir is set) and the trainer always
# picks up the latest step, restoring params, optimizer state, the step counter,
# and the training RNG. Each restart also bumps a saved restart counter that
# reseeds the input pipeline, so the resumed run sees a freshly shuffled data
# stream instead of replaying the windows it already trained on. Any warm-start
# paths below are ignored once a checkpoint exists.
#
# Starting fresh: bump RUN_TAG (section 3b). Reusing a tag whose checkpoints dir
# is non-empty resumes that run instead, silently discarding the action-encoder
# init. The trainer logs which path it took ("no checkpoint found; starting from
# step 0" vs "restoring checkpoint at step N") — check that line on startup.

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

# Action encoder: always cold start in adaln mode — there is no warm-start path.
# linear_3 is zero-init, so the encoder emits no action signal at step 0 and the
# pretrained UNet starts undisturbed; that zero is also what makes the adaln
# projector's normal-init kernel a no-op at step 0, so the run begins exactly at
# the pretrained operating point. (The converted Ctrl-World weights were trained
# to drive cross-attention and their linear_3 is non-zero, which is why
# CtrlWorldTrainer rejects warm-start + adaln outright. Use
# bash_scripts/train_ctrl_world.sh if you want to warm-start.)
echo "Action encoder: cold start (zero-init output projection); adaln projector fresh too."

# --- 3b. Run identity ---
# The trainer always resumes from $output_dir/checkpoints if anything is there,
# which would restore an old action encoder and discard the fresh init above.
# So a genuinely fresh run needs its own tag; bump RUN_TAG (never reuse one).
# RUN_TAG also names the W&B run (it is passed through as run_name).
export RUN_TAG="${RUN_TAG:-skeleton-cross-attn-no-text}"
export OUTPUT_DIR="gs://$GCS_BUCKET/checkpoints/svd_ac"
echo "RUN_TAG=$RUN_TAG"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# --- 4. Data paths (pre-encoded TFRecords; see docs/ctrl_world_data_format.md) ---
export TRAIN_DATA_DIR="gs://$GCS_BUCKET/datasets/droid_ctrl_world_skeleton/train"
export EVAL_DATA_DIR="gs://$GCS_BUCKET/datasets/droid_ctrl_world_skeleton/val"
export STATS_PATH="gs://$GCS_BUCKET/datasets/droid_ctrl_world_skeleton/stats.json"

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
# Skeleton conditioning for the SVD arm. The 7-dim vector actions are NOT used:
# conditioning is a rendered 2D-kinematic-skeleton video pushed through the SAME
# VAE as the RGB video, so the two latent grids are element-for-element aligned.
# No action encoder is built, so no dead weights land in the checkpoint.
#
# Requires a dataset carrying skeleton_cam0/1/2 beside latent_cam0/1/2, encoded
# with the SVD VAE (4-channel, 8x) — NOT the WAN skeleton datasets, whose latents
# are 48-channel/16x and are not interchangeable.
#
# The in-training W&B video preview is OFF: FlaxCtrlWorldPipeline has no skeleton
# route yet, so the trainer skips it. Train/eval loss are unaffected.
#
# THIS RUN: cross-attention site. The skeleton becomes a per-frame spatial K/V
# grid for the SPATIAL cross-attention — stride 4 over a 72x40 latent gives 180
# keys per frame, matching the WAN cross-attn route`s per-frame key count.
#
# Frame locking is free here: SVD`s cross-attention context is already
# (B*T, S, C), so frame k can only ever see frame k`s keys. WAN needed an
# explicit frame_level_cond reshape for the same property.
#
# ALIGNMENT differs from WAN by necessity. Softmax over keys is
# permutation-invariant, so a bare key says nothing about which grid cell it came
# from. WAN restores that with the transformer`s existing 3D RoPE at zero
# parameter cost; SVD has NO rotary embeddings anywhere, so alignment is instead a
# learned zero-init positional embedding on the keys. Also note the queries are a
# different grid size at each of the three cross-attention stages while these keys
# are one fixed grid, so at two of the three the model must learn a resampling
# rather than an identity. That is the honest cost of not having RoPE.
#
# The temporal transformer blocks mean-pool this context to a single key
# (see _build_time_context): they only ever see latent frame 0 anyway, so 180
# scrambled keys there would be cost without information.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_ctrl_world.py \
    src/maxdiffusion/configs/base_ctrl_world.yml \
    action_cond_mode='skeleton_cross_attn' \
    skeleton_cross_attn_stride=4 \
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
    wandb_project='svd-ac-skeleton-cross-attn-no-text' \
    wandb_video_every=0 \
    use_task_instructions=False 

# --- 7. Unmount ---
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"

# tpu create v6 --name train_ac_svd_skeleton_cross_attn_no_text -n 32 --setup-cmd "" --priority 0 --max-attempts 40 -- bash bash_scripts/train_ctrl_world_skeleton_cross_attn_no_text.sh
