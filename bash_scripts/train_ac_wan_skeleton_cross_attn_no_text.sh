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

# --- 5. Launch training ---
#
# Same skeleton conditioning signal as train_ac_wan_skeleton.sh and
# train_ac_wan_skeleton_adaln_no_text.sh, injected at the third site:
# action_cond_mode=skeleton_cross_attn patch-embeds the SAME rendered
# 2D-kinematic-skeleton latents into the CROSS-ATTENTION K/V.
#
# Conditioning here factors into an action REPRESENTATION (vector actions or
# rendered-skeleton latents) crossed with a SITE (cross-attn K/V, AdaLN
# modulation, additive in video-token space). This run fills the last skeleton
# cell, so all three skeleton sites are now trainable and differ only in where
# the identical signal lands:
#   skeleton              -> added onto the video tokens, once, after patching
#   skeleton_adaln        -> summed into the per-token timestep embedding
#   skeleton_cross_attn   -> the cross-attention K/V   (this run)
# The site is deliberately the SAME one train_ac_wan_cross_attn.sh uses for
# vector actions, so "cross_attn" names one site regardless of representation.
#
# THE ALIGNMENT PROBLEM, and why this mode needs machinery the other two do not.
# The additive and adaln routes are elementwise, so video token (f,h,w) meets
# skeleton token (f,h,w) by construction. Attention has no such guarantee:
# softmax over keys is permutation-invariant, so a bare key says nothing about
# which grid cell it came from. Two mechanisms restore the alignment
# STRUCTURALLY, and neither learns a single parameter:
#
#   1. Frame locking. cond_tokens_per_frame is set to the spatial patch count
#      (H_lat//2)*(W_lat//2) — 180 at 192x320 with 3 cameras stacked — so the
#      block folds F into the batch and latent frame k's 180 video tokens attend
#      ONLY to latent frame k's 180 skeleton tokens. Same reshape the vector
#      cross_attn route already uses, with the per-frame key count raised from a
#      handful of action tokens to a full grid. Also 12x cheaper than attending
#      over all 2160 skeleton tokens: 12*180^2 vs 2160^2, ~8% of a self-attention.
#   2. Cross-attention RoPE. attn2 receives the same 3D rotary embedding attn1
#      already uses, sliced to this frame, so Q and K carry identical phases at
#      identical grid cells and the logit peaks on the diagonal. Because Q and K
#      share the frame's temporal phase it cancels in the relative rotation,
#      leaving a purely spatial offset. This is the one mode that enables it —
#      the vector route's action tokens sit on no grid, so there is no position
#      to encode.
#
# WHAT THIS SITE BUYS over the two elementwise routes: the query CHOOSES what to
# read instead of being handed a fixed partner. That matters exactly when the
# alignment is imperfect — a skeleton render is a projection of a kinematic
# model, so camera-calibration or URDF drift displaces it from the true arm by a
# few pixels, and an additive route can only bake that error in as a systematic
# wrong offset. Attention can shift where it reads. Likewise a token where the
# arm is occluded can down-weight the skeleton rather than have it added anyway.
#
# WHAT IT COSTS. (a) ~8% of a self-attention per block, plus flash-kernel block
# padding: 180 keys pad up to the kernel's block size, so the realised cost is
# higher than the FLOP count suggests. (b) The pretrained to_q/to_k in attn2 were
# trained on 512-token T5 TEXT; a spatial key grid is off-distribution for them
# in a way the other two sites never expose. Note this is a different failure
# mode from the K=1 vector route, where those weights are simply inert (softmax
# over one key is constant, so they get exactly zero gradient) — here they are
# active and mismatched. Watch the cross-attention logit statistics early.
#
# NO skeleton_embed_alpha here, deliberately — same reasoning as skeleton_adaln.
# The conv is zero-init, so step 0 is exactly the no-skeleton baseline without
# it; all alpha would still do is throttle this path's effective learning rate by
# alpha**2. Passing it would be silently ignored.
#
# One consequence of that zero init worth knowing: with every key identical (all
# zero) the softmax is uniform AND every value is the same vector, so to_q/to_k
# get exactly zero gradient on step 0. This is NOT the permanent starvation the
# action_adaln_proj docstring guards against — the conv itself gets gradient
# immediately through to_v's normally-initialised kernel, so the keys become
# distinct after one update and Q/K train from step 2. A one-step delay, not a
# deadlock.
#
# The conv emits wan_text_dim (4096), NOT inner_dim, because cross-attention
# context enters WanModel before condition_embedder.text_embedder projects it
# down — so the skeleton reaches attn2 through the identical path the action
# tokens take. That is also why this module's shape differs from the other two
# skeleton embeds, which emit inner_dim.
#
# Requires the same skeleton dataset as train_ac_wan_skeleton.sh:
# droid_wan_2.2_skeleton_192_320, carrying skeleton_cam0/1/2 alongside
# latent_cam0/1/2. Pointing this at a non-skeletal dataset fails at TFRecord
# parse time on the missing feature.
#
# This is the text-OFF arm (use_task_instructions=False): the rendered skeleton
# video is the ONLY conditioning signal, and here it IS the cross-attention K/V.
# Note that with text ON this mode cannot hand cross-attention the full 512-token
# T5 sequence the way skeleton/skeleton_adaln can — the K/V is frame-locked, so
# there is no room for a second sequence and the instruction is POOLED onto the
# skeleton tokens, exactly as the vector cross_attn route pools it onto the
# action tokens.
#
# Not checkpoint-compatible with ANY other mode, skeleton included: this trains a
# skeleton_cross_attn_embed and no skeleton_embed, skeleton_adaln_embed or
# action_encoder. Unlike the other two skeleton modules, this one's tensor SHAPE
# also differs (48->4096 rather than 48->inner_dim), so the incompatibility is
# not merely the param-tree path.
# Kept as its own script so the runs' checkpoints and wandb projects never collide.

source ./maxdiffusion_venv/bin/activate
ulimit -n 65536

# --- TFRecord path ---
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_wan.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    run_name=ac_wan_droid_skeleton_cross_attn_no_text \
    output_dir=gs://v6_east1d/checkpoints/wan-ac \
    pretrained_model_name_or_path=$WAN_TI2V_MODEL_DIR \
    dataset_type=tfrecord \
    train_data_dir=gs://v6_east1d/datasets/droid_wan_2.2_skeleton_192_320/train \
    eval_data_dir=gs://v6_east1d/datasets/droid_wan_2.2_skeleton_192_320/val \
    action_stats_path=gs://v6_east1d/datasets/droid_wan_2.2_skeleton_192_320/stats.json \
    action_cond_mode=skeleton_cross_attn \
    cache_latents_text_encoder_outputs=True \
    attention=tokamax_flash \
    weights_dtype=float32 \
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
    per_device_batch_size=1.0 \
    grad_accum_steps=1 \
    allow_split_physical_axes=True \
    scan_layers=True \
    learning_rate=1e-5 \
    warmup_steps_fraction=0.05 \
    learning_rate_schedule_type=cosine \
    learning_rate_end_ratio=0.0 \
    max_train_steps=100100 \
    checkpoint_every=500 \
    checkpoint_keep_period=10000 \
    eval_every=1000 \
    height=192 \
    width=320 \
    num_predicted_latents=5 \
    num_history_latent_frames=7 \
    history_noise_max_timestep=200 \
    flash_min_seq_length=128 \
    hardware='tpu' \
    log_attn_param_stats=False \
    log_attn_activation_stats=False \
    wandb_project='wan-ac-skeleton-cross-attn-no-text' \
    wandb_video_every=1000 \
    wandb_video_samples=1 \
    wandb_video_inference_steps=20 \
    use_task_instructions=False

# --- 6. Unmount ---
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"

# tpu create v6 --name train_ac_wan_skeleton_cross_attn_no_text -n 32 --setup-cmd "" --priority 0 --max-attempts 40 -- bash bash_scripts/train_ac_wan_skeleton_cross_attn_no_text.sh
