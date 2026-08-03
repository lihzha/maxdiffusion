#!/usr/bin/env bash
set -euo pipefail

# Raise the open-file limit. The queue worker's login shell defaults to 1024, which the 5B
# model + data pipeline + tensorstore checkpoint restore exhaust (Errno 24).
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
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"

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

# --- exp_03 S1.5 NO-UPDATE discriminator probe (plan v3.2 §4) ---
# Runs NO optimizer updates. At BOTH states the experiment starts from -- the exp_02 step-10,000
# checkpoint (Tier 1) and the pinned pretrained init (Tier 2) -- it reports per-objective losses,
# gradient norms and cosines against the plain objective, A's label isolation, the p_ss=0 parity
# identity, the support-gradient variance decomposition, and the mechanism-B sigma traces. The segment-final pass
# ran the plan's 25-step sampler and reported mean m_corr = 0.8133 with 0/100 windows at 0.95, so
# before drawing conclusions about the weights we measure the sampler's contribution.
#
# VERDICT-ISOLATED BY CONSTRUCTION. This runs probe_overfit100_sampling_steps.py, which touches no
# role validation, no aggregation artifact, no staging and no publication marker, and writes a plain
# diagnostic JSON to <output_dir>/<run_name>/validation_probe_sampling/. It cannot produce an
# admissible artifact, so it cannot disturb the segment-final evidence or the success statistic.
# The rollout's real step knob (side_adapter_sampling_steps, fixed at 25 by the D9 recipe AND by the
# role contract) is overridden PER ARM inside the script through a read-only config view -- never
# globally, and never for an eval pass.
#
# The 25-step arm is included as an in-probe validity control: with the same cohort, the same
# correct-mode context and the same per-window rng as the eval path, its rows should reproduce the
# segment-final rows for those windows.
#
# NO ffmpeg-ensure block here, deliberately -- mirroring the one-step loss arm. The probe scores only
# against the VAE decode of the stored z_video (latent MSE, pixel MSE, SSIM); it never pulls or
# decodes a source MP4 and never requests the auxiliary RGB path (eval_aux_rgb), which is the only
# thing in the eval arm that needs ffmpeg.

RUN_NAME="${RUN_NAME:?RUN_NAME must match the training run to probe}"
MANIFEST_PATH="${MANIFEST_PATH:-docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100}"
DATA_DIR="${DATA_DIR:-gs://v6_east1d/datasets/exp02_overfit100/train100}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR}"
EXPECTED_WINDOWS="${EXPECTED_WINDOWS:-1629}"
NUM_TEXT_SLOTS="${NUM_TEXT_SLOTS:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-10000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?CHECKPOINT_DIR must name the exp_02 step-10,000 checkpoint (Tier-1 state)}"
S1_5_NUM_BATCHES="${S1_5_NUM_BATCHES:-8}"
S1_5_SUPPORT_DRAWS="${S1_5_SUPPORT_DRAWS:-4}"
EXP03_RAMP_ORIGIN="${EXP03_RAMP_ORIGIN:-10000}"
PROBE_NUM_WINDOWS="${PROBE_NUM_WINDOWS:-30}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"
SKIP_HF_PREFETCH="${SKIP_HF_PREFETCH:-0}"

# The model repo + revision come from the MANIFEST (C1), never from this launcher.
if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "[overfit100-probe] FATAL: manifest not found at MANIFEST_PATH=${MANIFEST_PATH}" >&2
  exit 1
fi
MODEL_REPO="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['hf_repo'])" "${MANIFEST_PATH}")"
MODEL_REVISION="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['revision'])" "${MANIFEST_PATH}")"
if ! printf '%s' "${MODEL_REVISION}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "[overfit100-probe] FATAL: manifest revision '${MODEL_REVISION}' is not a 40-hex commit sha" >&2
  exit 1
fi

echo "RUN_NAME=${RUN_NAME}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "MODEL_REPO=${MODEL_REPO}"
echo "MODEL_REVISION=${MODEL_REVISION}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "EXPECTED_WINDOWS=${EXPECTED_WINDOWS}"
echo "NUM_TEXT_SLOTS=${NUM_TEXT_SLOTS}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "CHECKPOINT_STEP=${CHECKPOINT_STEP}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "S1_5_NUM_BATCHES=${S1_5_NUM_BATCHES}"
echo "S1_5_SUPPORT_DRAWS=${S1_5_SUPPORT_DRAWS}"
echo "EXP03_RAMP_ORIGIN=${EXP03_RAMP_ORIGIN}"
echo "PROBE_NUM_WINDOWS=${PROBE_NUM_WINDOWS}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "SKIP_HF_PREFETCH=${SKIP_HF_PREFETCH}"

# The queue deploys an uploaded TARBALL with no .git, so COMMIT must be relayed from the launch
# env (`tpu create --env COMMIT=$(git rev-parse HEAD)`) and EXPORTED -- the aggregation artifact
# stamps it as the eval-code provenance.
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export COMMIT
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

# NO ffmpeg-ensure block here, deliberately -- mirroring the one-step loss arm: this probe decodes
# latents with the VAE and never touches a source MP4, so it has no ffmpeg/ffprobe dependency.

# FULL-repo prefetch AT THE MANIFEST'S PINNED REVISION: the probe loads the transformer, the T5
# (context table + null embedding) AND the VAE (latent -> video decode), so NO allow-pattern
# argument is passed.
if [ "${SKIP_HF_PREFETCH}" != "1" ]; then
  HF_PREFETCH_REVISION="${MODEL_REVISION}" \
    bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_REPO}"
fi

# Resolve the pinned revision to ONE local snapshot dir, from the LOCAL CACHE ONLY, so this can
# never silently fetch a different revision.
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
    echo "[overfit100-probe] FATAL: resolved MODEL_DIR='${MODEL_DIR}' does not carry the pinned revision ${MODEL_REVISION}" >&2
    exit 1
    ;;
esac

python src/maxdiffusion/probe_exp03_s1_5.py \
  src/maxdiffusion/configs/base_wan_5b_exp03.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  model_manifest_path="${MANIFEST_PATH}" \
  expected_model_revision="${MODEL_REVISION}" \
  train_data_dir="${DATA_DIR}" \
  eval_data_dir="${EVAL_DATA_DIR}" \
  expected_windows="${EXPECTED_WINDOWS}" \
  num_text_slots="${NUM_TEXT_SLOTS}" \
  output_dir="${OUTPUT_DIR}" \
  base_output_directory="${OUTPUT_DIR}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  checkpoint_step="${CHECKPOINT_STEP}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  s1_5_num_batches="${S1_5_NUM_BATCHES}" \
  s1_5_support_draws="${S1_5_SUPPORT_DRAWS}" \
  exp03_ramp_origin="${EXP03_RAMP_ORIGIN}" \
  probe_num_windows="${PROBE_NUM_WINDOWS}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
