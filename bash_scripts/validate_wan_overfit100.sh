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

# --- exp_02 overfit100 ROLLOUT evaluation (plan v4 D11, cycle D) ---
# Rolls out the selected windows on generate_wan_side_adapter.py's OVERFIT100_TI2V branch and
# writes the machine-readable aggregation artifact the success statistic consumes
# (<VALIDATION_OUTPUT_DIR>/step_XXXXXX/aggregation.json + summary.csv/json, plus mp4s when
# WRITE_VIDEOS=True). Deltas vs validate_wan_full_ft.sh:
#  (1) points at base_wan_5b_overfit100.yml (model_type OVERFIT100_TI2V);
#  (2) EVAL_DATA_DIR is a TRAIN set -- exp_02 measures memorization OF the training windows;
#  (3) selection is EVAL_WINDOWS ('canonical' = the median window of every episode in the set,
#      or an explicit comma-separated list of ep<ID>_v0_s<START> names), not
#      VALIDATION_ORDINALS/NUM_EVAL_VIDEOS (ignored in this mode);
#  (4) ROLLOUT_SEEDS x CONTEXT_MODES is the D11 coverage cell: 3 seeds and correct/null/shuffled
#      at segment-final checkpoints; 1 seed, correct only, at intermediate ones;
#  (5) THE MODEL PIN (cycle-C review C1), identical to train_wan_overfit100.sh: the repo and
#      revision come from the MANIFEST, are prefetched at that exact revision, resolved from the
#      LOCAL CACHE ONLY, and re-asserted by the trainer's snapshot check. The manifest is also
#      what supplies each episode's n_windows (hence the canonical window) and authenticates
#      episodes.json, so it is a REQUIRED input here, not an optional pin.
#
# CHECKPOINT_STEP selects what to roll out: 0 = the pretrained baseline (no Orbax read), a
# positive step must exist in CHECKPOINT_DIR, -1 = latest.

RUN_NAME="${RUN_NAME:?RUN_NAME must match the training run to validate}"
MANIFEST_PATH="${MANIFEST_PATH:-docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100}"
DATA_DIR="${DATA_DIR:-gs://v6_east1d/datasets/exp02_overfit100/train100}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$DATA_DIR}"
EXPECTED_WINDOWS="${EXPECTED_WINDOWS:-1629}"
NUM_TEXT_SLOTS="${NUM_TEXT_SLOTS:-100}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/checkpoints}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:--1}"
EVAL_PASS_ROLE="${EVAL_PASS_ROLE:?EVAL_PASS_ROLE must be one of s2_gate|s3_intermediate|s3_segment_final|s3_full_set (D11 coverage cell; the pass is refused if its seeds/modes/windows do not match)}"
EVAL_WINDOWS="${EVAL_WINDOWS:-canonical}"
ROLLOUT_SEEDS="${ROLLOUT_SEEDS:-0,1,2}"
CONTEXT_MODES="${CONTEXT_MODES:-correct}"
CONTEXT_SHUFFLE_SEED="${CONTEXT_SHUFFLE_SEED:-0}"
WRITE_VIDEOS="${WRITE_VIDEOS:-False}"
EVAL_AUX_RGB="${EVAL_AUX_RGB:-True}"
FLAGGED_WINDOWS="${FLAGGED_WINDOWS:-}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-${OUTPUT_DIR%/}/${RUN_NAME}/validation}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"
SKIP_HF_PREFETCH="${SKIP_HF_PREFETCH:-0}"

# The model repo + revision come from the MANIFEST (C1), never from this launcher.
if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "[overfit100-val] FATAL: manifest not found at MANIFEST_PATH=${MANIFEST_PATH}" >&2
  exit 1
fi
MODEL_REPO="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['hf_repo'])" "${MANIFEST_PATH}")"
MODEL_REVISION="$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['vae_fingerprint']['revision'])" "${MANIFEST_PATH}")"
if ! printf '%s' "${MODEL_REVISION}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "[overfit100-val] FATAL: manifest revision '${MODEL_REVISION}' is not a 40-hex commit sha" >&2
  exit 1
fi

echo "RUN_NAME=${RUN_NAME}"
echo "MANIFEST_PATH=${MANIFEST_PATH}"
echo "MODEL_REPO=${MODEL_REPO}"
echo "MODEL_REVISION=${MODEL_REVISION}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "EVAL_DATA_DIR=${EVAL_DATA_DIR}"
echo "EXPECTED_WINDOWS=${EXPECTED_WINDOWS}"
echo "NUM_TEXT_SLOTS=${NUM_TEXT_SLOTS}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "CHECKPOINT_STEP=${CHECKPOINT_STEP}"
echo "EVAL_PASS_ROLE=${EVAL_PASS_ROLE}"
echo "EVAL_WINDOWS=${EVAL_WINDOWS}"
echo "ROLLOUT_SEEDS=${ROLLOUT_SEEDS}"
echo "CONTEXT_MODES=${CONTEXT_MODES}"
echo "CONTEXT_SHUFFLE_SEED=${CONTEXT_SHUFFLE_SEED}"
echo "WRITE_VIDEOS=${WRITE_VIDEOS}"
echo "EVAL_AUX_RGB=${EVAL_AUX_RGB}"
echo "FLAGGED_WINDOWS=${FLAGGED_WINDOWS:-<none>}"
echo "VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "SKIP_HF_PREFETCH=${SKIP_HF_PREFETCH}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"

# The queue deploys an uploaded TARBALL with no .git, so COMMIT must be relayed from the launch
# env (`tpu create --env COMMIT=$(git rev-parse HEAD)`) and EXPORTED -- the aggregation artifact
# stamps it as the eval-code provenance.
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
export COMMIT
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

# FULL-repo prefetch AT THE MANIFEST'S PINNED REVISION: this branch loads the transformer, the
# T5 (the per-episode context table + the null embedding) AND the VAE (latent -> video decode),
# so NO allow-pattern argument is passed.
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
    echo "[overfit100-val] FATAL: resolved MODEL_DIR='${MODEL_DIR}' does not carry the pinned revision ${MODEL_REVISION}" >&2
    exit 1
    ;;
esac

python src/maxdiffusion/generate_wan_side_adapter.py \
  src/maxdiffusion/configs/base_wan_5b_overfit100.yml \
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
  eval_pass_role="${EVAL_PASS_ROLE}" \
  eval_windows="${EVAL_WINDOWS}" \
  rollout_seeds="${ROLLOUT_SEEDS}" \
  context_modes="${CONTEXT_MODES}" \
  context_shuffle_seed="${CONTEXT_SHUFFLE_SEED}" \
  write_videos="${WRITE_VIDEOS}" \
  eval_aux_rgb="${EVAL_AUX_RGB}" \
  flagged_windows="${FLAGGED_WINDOWS}" \
  validation_output_dir="${VALIDATION_OUTPUT_DIR}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
