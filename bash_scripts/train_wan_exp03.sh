#!/usr/bin/env bash
set -euo pipefail

# --- exp_03 `rollout_objective` TRAINING launcher (plan v3.1 §3) ---
#
# bash_scripts/train_wan_overfit100.sh with the exp03_* passthroughs and the exp_03 config; the
# manifest pin, the full HF prefetch at the pinned revision, the local-only snapshot resolution,
# the COMMIT export and the smoke-scaled defaults are all inherited unchanged, because an arm must
# differ from exp_02's control in the objective and nothing else.
#
# No ffmpeg-ensure block, by the same reasoning as the one-step loss arm: this is a TRAINING arm.
# It decodes no MP4 and writes no video -- the videos are produced later by the eval launcher,
# which has the ffmpeg block. Adding one here would be cargo cult.

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

# --- exp_02 overfit100 text-conditioned memorization run (plan v4 §3, cycle C) ---
# Trains the FULL Wan2.2 TI2V 5B backbone with PER-EPISODE language conditioning via
# train_wan.py -> WanTI2VOverfit100Trainer (model_type: OVERFIT100_TI2V in the yml below).
#
# Deltas vs train_wan_full_ft.sh:
#  (1) points at base_wan_5b_overfit100.yml;
#  (2) DATA_DIR / EXPECTED_WINDOWS / NUM_TEXT_SLOTS travel together -- they select and
#      then VERIFY one built set. train100 -> 1629 windows / 100 slots; train10 -> 167 / 10.
#      A mismatch is a hard startup failure (the trainer refuses to train), which is
#      exactly what stops an accidental "train10 config on train100 data" run;
#  (3) CHECKPOINT_STEPS is an explicit non-uniform LIST (H2). It is passed as a single
#      quoted pyconfig token, e.g. [250,500,1000,1750,2500] -- NO SPACES, since pyconfig
#      splits argv on whitespace and parses the value with ast.literal_eval. The trainer
#      keeps every listed checkpoint (max_to_keep=None);
#  (4) the HF prefetch requests the FULL repo (no allow-pattern argument): this trainer
#      loads the transformer AND the T5 text encoder (for the 100-prompt context table)
#      AND the VAE config -- unlike the cycle-B dataset build, which was VAE-only.
#      It runs BEFORE python starts (distributed launches used to die on HF 408 timeouts
#      mid-JAX-init) and is a no-op when the snapshot is already cached;
#  (5) THE MODEL PIN (cycle-C review C1), mirroring build_overfit100_dataset.sh: the repo and
#      revision come from the MANIFEST, not from this launcher. We prefetch that EXACT
#      revision, resolve it to a local snapshot directory from the LOCAL CACHE ONLY, and pass
#      that directory as pretrained_model_name_or_path. The trainer then re-asserts that the
#      path carries the pinned revision and that the revision matches the committed manifest,
#      so a run cannot silently train against a mutated hub default.
#
# Batch contract: per_device_batch_size is the only authoritative knob (pyconfig recomputes
# the global batch from it x device_count). The yml defaults to 4.0 -> GBS 256 on the
# v6e-64 S3 target and GBS 32 on the v6e-8 S1/S2 target; this wrapper's defaults stay
# SMOKE-SCALED and are always passed as explicit CLI overrides, so a bare wrapper
# invocation is a dev smoke rather than a silent full-recipe run.

RUN_NAME="${RUN_NAME:-wan-exp03-smoke}"
MANIFEST_PATH="${MANIFEST_PATH:-docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json}"
DATA_DIR="${DATA_DIR:-gs://v6_east1d/datasets/exp02_overfit100/train10}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR}"
EXPECTED_WINDOWS="${EXPECTED_WINDOWS:-167}"
NUM_TEXT_SLOTS="${NUM_TEXT_SLOTS:-10}"
TEXT_ENCODE_BATCH="${TEXT_ENCODE_BATCH:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-exp03}"
SIDE_ADAPTER_NOISE_MODE="${SIDE_ADAPTER_NOISE_MODE:-fresh}"

# C1: the model repo + revision come from the MANIFEST (same pattern as
# build_overfit100_dataset.sh). Fails in seconds if the manifest is missing or unpinned.
if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "[exp03] FATAL: manifest not found at MANIFEST_PATH=${MANIFEST_PATH}" >&2
  exit 1
fi
MODEL_REPO="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['hf_repo'])" "${MANIFEST_PATH}")"
MODEL_REVISION="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['revision'])" "${MANIFEST_PATH}")"
if ! printf '%s' "${MODEL_REVISION}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "[exp03] FATAL: manifest revision '${MODEL_REVISION}' is not a 40-hex commit sha" >&2
  exit 1
fi

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-250}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-[250,500,1000,2500]}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
EVAL_EVERY="${EVAL_EVERY:-0}"
LOG_PERIOD="${LOG_PERIOD:-1}"
SAVE_FINAL_CHECKPOINT="${SAVE_FINAL_CHECKPOINT:-False}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4.0}"
TFRECORD_SHUFFLE_BUFFER_SIZE="${TFRECORD_SHUFFLE_BUFFER_SIZE:-}"
WANDB_PROJECT="${WANDB_PROJECT-}"
SKIP_HF_PREFETCH="${SKIP_HF_PREFETCH:-0}"

# exp_03: the objective and its knobs (plan v3.1 §1). EXP03_OBJECTIVE is the experiment variable;
# the rest are the trials' hyperparameters, passed always so a run's log records what it ran with.
# Non-control objectives raise NotImplementedError until round 3, i.e. a mistyped arm fails at
# startup instead of quietly training the control.
EXP03_OBJECTIVE="${EXP03_OBJECTIVE:-control}"
EXP03_K_A="${EXP03_K_A:-2}"
EXP03_K_B="${EXP03_K_B:-2}"
EXP03_LAMBDA="${EXP03_LAMBDA:-0.5}"
EXP03_P_SS_MAX="${EXP03_P_SS_MAX:-0.5}"
EXP03_P_SS_RAMP_STEPS="${EXP03_P_SS_RAMP_STEPS:-500}"
EXP03_RAMP_ORIGIN="${EXP03_RAMP_ORIGIN:-0}"

echo "RUN_NAME=${RUN_NAME}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "MODEL_REPO=${MODEL_REPO}"
echo "MODEL_REVISION=${MODEL_REVISION}"
echo "DATA_DIR=${DATA_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "EXPECTED_WINDOWS=${EXPECTED_WINDOWS}"
echo "NUM_TEXT_SLOTS=${NUM_TEXT_SLOTS}"
echo "TEXT_ENCODE_BATCH=${TEXT_ENCODE_BATCH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SIDE_ADAPTER_NOISE_MODE=${SIDE_ADAPTER_NOISE_MODE}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "CHECKPOINT_STEPS=${CHECKPOINT_STEPS}"
echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "EVAL_EVERY=${EVAL_EVERY}"
echo "LOG_PERIOD=${LOG_PERIOD}"
echo "SAVE_FINAL_CHECKPOINT=${SAVE_FINAL_CHECKPOINT}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "TFRECORD_SHUFFLE_BUFFER_SIZE=${TFRECORD_SHUFFLE_BUFFER_SIZE:-config_default}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "SKIP_HF_PREFETCH=${SKIP_HF_PREFETCH}"
echo "EXP03_OBJECTIVE=${EXP03_OBJECTIVE}"
echo "EXP03_K_A=${EXP03_K_A}"
echo "EXP03_K_B=${EXP03_K_B}"
echo "EXP03_LAMBDA=${EXP03_LAMBDA}"
echo "EXP03_P_SS_MAX=${EXP03_P_SS_MAX}"
echo "EXP03_P_SS_RAMP_STEPS=${EXP03_P_SS_RAMP_STEPS}"
echo "EXP03_RAMP_ORIGIN=${EXP03_RAMP_ORIGIN}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT}"
echo "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT}"

# The queue deploys an uploaded TARBALL with no .git, so COMMIT must be relayed from the
# launch env (`tpu create --env COMMIT=$(git rev-parse HEAD)`) and EXPORTED -- python and
# the logs read it from the process environment.
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export COMMIT
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

# FULL-repo prefetch AT THE MANIFEST'S PINNED REVISION: transformer + T5 text encoder + VAE
# are all needed (the context table is built from the instructions before training starts), so
# NO allow-pattern argument is passed -- unlike the VAE-only cycle-B build.
if [ "${SKIP_HF_PREFETCH}" != "1" ]; then
  HF_PREFETCH_REVISION="${MODEL_REVISION}" \
    bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_REPO}"
fi

# Resolve the pinned revision to ONE local snapshot directory, from the LOCAL CACHE ONLY
# (local_files_only=True) so this can never silently fetch a different revision. MODEL_DIR may
# be pre-set to a staged directory; either way the trainer re-asserts that the path carries
# ${MODEL_REVISION} before loading anything.
if [ -z "${MODEL_DIR:-}" ]; then
  MODEL_DIR="$(python - "${MODEL_REPO}" "${MODEL_REVISION}" <<'PY'
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_files_only=True))
PY
  )"
fi
echo "MODEL_DIR=${MODEL_DIR}"
case "${MODEL_DIR}" in
  *"${MODEL_REVISION}"*) ;;
  *)
    echo "[exp03] FATAL: resolved MODEL_DIR='${MODEL_DIR}' does not carry the pinned revision ${MODEL_REVISION}" >&2
    exit 1
    ;;
esac

python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_5b_exp03.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  model_manifest_path="${MANIFEST_PATH}" \
  expected_model_revision="${MODEL_REVISION}" \
  train_data_dir="${DATA_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  expected_windows="${EXPECTED_WINDOWS}" \
  num_text_slots="${NUM_TEXT_SLOTS}" \
  text_encode_batch="${TEXT_ENCODE_BATCH}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  side_adapter_noise_mode="${SIDE_ADAPTER_NOISE_MODE}" \
  learning_rate="${LEARNING_RATE}" \
  warmup_steps="${WARMUP_STEPS}" \
  max_train_steps="${MAX_TRAIN_STEPS}" \
  checkpoint_steps="${CHECKPOINT_STEPS}" \
  checkpoint_every="${CHECKPOINT_EVERY}" \
  eval_every="${EVAL_EVERY}" \
  log_period="${LOG_PERIOD}" \
  save_final_checkpoint="${SAVE_FINAL_CHECKPOINT}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  ${TFRECORD_SHUFFLE_BUFFER_SIZE:+tfrecord_shuffle_buffer_size="${TFRECORD_SHUFFLE_BUFFER_SIZE}"} \
  wandb_project="${WANDB_PROJECT}" \
  exp03_objective="${EXP03_OBJECTIVE}" \
  exp03_k_a="${EXP03_K_A}" \
  exp03_k_b="${EXP03_K_B}" \
  exp03_lambda="${EXP03_LAMBDA}" \
  exp03_p_ss_max="${EXP03_P_SS_MAX}" \
  exp03_p_ss_ramp_steps="${EXP03_P_SS_RAMP_STEPS}" \
  exp03_ramp_origin="${EXP03_RAMP_ORIGIN}" \
  hardware=tpu
