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
# Same skeleton conditioning signal as train_ac_wan_skeleton.sh, injected at a
# different site: action_cond_mode=skeleton_adaln patch-embeds the SAME rendered
# 2D-kinematic-skeleton latents into per-token AdaLN conditioning instead of
# adding them onto the video tokens.
#
# The additive route injects once, right after the patch embedding, and leaves
# the residual stream to carry the signal through the remaining blocks. This one
# is summed into the per-token timestep embedding before time_proj expands it
# into the 6-way modulation vector, so it re-modulates the activations in EVERY
# block. That is the whole variable under test — the adapter is deliberately the
# same shape in both (a single conv from the 192-dim patch), so a
# skeleton-vs-skeleton_adaln comparison varies the injection site alone.
#
# The site is deliberately the SAME one train_ac_wan_adaln.sh uses for vector
# actions (WanModel's action_hidden_states). Conditioning here factors into an
# action representation (vector actions / rendered-skeleton latents) crossed with
# a site (cross-attn K/V, AdaLN, additive in token space), and 'adaln' has to
# mean one site in both cells or a cross-representation comparison confounds the
# representation with the wiring. Sharing the argument makes that structural.
#
# No projection and no spatial repeat is needed. The vector route has both only
# because a 7-dim EEF action has no spatial extent, so it collapses to one vector
# per latent frame and repeats it across that frame's patches. A skeleton's token
# grid is already per-token and already inner_dim wide, and the transformer's
# per-token AdaLN (built for WAN 2.2 TI2V's per-token timesteps) is already
# spatially varying, so it lines up token for token — spatial and camera
# alignment stay structural, with no positional or camera embedding involved.
#
# Known cost of this site — SHARED with train_ac_wan_adaln.sh, not specific to
# the skeleton: the signal lives in the same representation as the diffusion
# timestep and reaches all six modulation components including scale and gate,
# which act multiplicatively in every block. That adaln run already trains here,
# which is the best evidence available that it is tolerable.
#
# Only the perturbation's shape differs. The per-token timestep varies across
# frames but is constant across the patches within one, so the vector route's
# per-frame-repeated offset sits in the same subspace a timestep change occupies
# and is maximally confusable with it; a spatial skeleton offset is separable
# from the time signal but off-distribution for time_proj's pretrained kernel.
# Two different risks, not a ranking. Watch the modulation stats either way.
#
# If this destabilises, the escape hatch is a separate projection into modulation
# space (post-time_proj, shift slots only) at the price of the two routes no
# longer sharing a site.
#
# NO skeleton_embed_alpha here, deliberately. The conv is zero-init, so step 0 is
# exactly the no-skeleton baseline without it; all alpha would still do is
# throttle this path's effective learning rate by alpha**2. Passing it would be
# silently ignored.
#
# Requires the same skeleton dataset as train_ac_wan_skeleton.sh:
# droid_wan_2.2_skeleton_192_320, carrying skeleton_cam0/1/2 alongside
# latent_cam0/1/2. Pointing this at a non-skeletal dataset fails at TFRecord
# parse time on the missing feature.
#
# This is the text-ON arm (use_task_instructions=True). Like adaln and skeleton,
# this mode leaves cross-attention free, so the T5 instruction is the full
# 512-token cross-attention sequence rather than a pooled bias.
#
# Not checkpoint-compatible with ANY other mode, skeleton included: this trains a
# skeleton_adaln_embed and no skeleton_embed or action_encoder. Note the two
# skeleton modules now have the IDENTICAL shape (one 48->inner_dim conv), so the
# incompatibility is the param-tree path, not the tensor — and warm-starting one
# from the other by renaming the key would be meaningless anyway: the additive
# route's conv maps into video-token space (with alpha folded into its effective
# scale), this one into the time-embedding space.
# Kept as its own script so the runs' checkpoints and wandb projects never collide.

source ./maxdiffusion_venv/bin/activate
ulimit -n 65536

# --- TFRecord path ---
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python src/maxdiffusion/train_wan.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    run_name=ac_wan_droid_skeleton_adaln \
    output_dir=gs://v6_east1d/checkpoints/wan-ac \
    pretrained_model_name_or_path=$WAN_TI2V_MODEL_DIR \
    dataset_type=tfrecord \
    train_data_dir=gs://v6_east1d/datasets/droid_wan_2.2_skeleton_192_320/train \
    eval_data_dir=gs://v6_east1d/datasets/droid_wan_2.2_skeleton_192_320/val \
    action_stats_path=gs://v6_east1d/datasets/droid_wan_2.2_skeleton_192_320/stats.json \
    action_cond_mode=skeleton_adaln \
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
    checkpoint_every=1000 \
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
    wandb_project='wan-ac-skeleton-adaln' \
    wandb_video_every=1000 \
    wandb_video_samples=1 \
    wandb_video_inference_steps=20 \
    use_task_instructions=True

# --- 6. Unmount ---
fusermount -u "$GCS_MOUNT" || fusermount -uz "$GCS_MOUNT"

# tpu create v6 --name train_ac_wan_skeleton_adaln -n 32 --setup-cmd "" --priority 0 --max-attempts 40 -- bash bash_scripts/train_ac_wan_skeleton_adaln.sh
