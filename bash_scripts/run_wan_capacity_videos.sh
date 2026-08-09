#!/usr/bin/env bash
set -euo pipefail

# Post-STOP capacity-videos runbook (2026-08-09): render the never-decoded arms of exp_04 and
# exp_05's published capacity artifacts as GT-vs-prediction comparison mp4s, one v6e-8 job for both
# experiments. Read-only against the capacity roots; writes only mp4s + one videos_report.json per
# attempt-scoped out root, all through overwrite-idempotent small-artifact writers -- so a queue
# auto-retry from phase 1 simply re-renders (issue #13's shard-immutability trap does not apply).
#
# SIBLING of bash_scripts/run_wan_pos_inversion.sh: same env, prefetch, provenance and preflight
# machinery, this runbook's config interface.
#
# Run `bash bash_scripts/setup.sh MODE=stable DEVICE=tpu` once on the host before this script.

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

LOG_DIR="${LOG_DIR:-docs/worklogs_yixun/exp_05_pos_context_claude}"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y-%m-%d_%H:%M:%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/capacity_videos_${STAMP}.log}"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "LOG_FILE=${LOG_FILE}"

if [ -f "$HOME/.config/irom-tpu/secrets.env" ]; then
  __shell_flags="$-"
  set +x
  source "$HOME/.config/irom-tpu/secrets.env"
  case "${__shell_flags}" in *x*) set -x ;; esac
  unset __shell_flags
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

COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo '')}"
if ! printf '%s' "${COMMIT}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "FATAL: COMMIT is '${COMMIT}' -- the videos report carries a 40-hex code_sha or nothing runs."
  echo "       Export COMMIT explicitly when running from an uploaded tarball with no git checkout."
  exit 1
fi
export COMMIT

# Same 5B transformer, same mesh, same flags as the capacity launchers.
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

RUN_NAME="${RUN_NAME:-wan-capacity-videos}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
ATTEMPT="${ATTEMPT:-$(date -u +%m%d-%H%M%S)}"

# The published capacity roots (read-only here). No ``:-`` on the out roots: set-but-empty means a
# wrapper bug and must refuse, not silently retarget.
VIDEOS_NULL_ROOT="${VIDEOS_NULL_ROOT-gs://v6_east1d/datasets/droid_wan_null_adapter/j1r2/capacity_att-0806-164625}"
VIDEOS_POS_ROOT="${VIDEOS_POS_ROOT-gs://v6_east1d/datasets/droid_wan_pos_context/k1/capacity}"
VIDEOS_NULL_OUT="${VIDEOS_NULL_OUT-gs://v6_east1d/datasets/droid_wan_null_adapter/j1r2/videos_att-${ATTEMPT}}"
VIDEOS_POS_OUT="${VIDEOS_POS_OUT-gs://v6_east1d/datasets/droid_wan_pos_context/k1/videos_att-${ATTEMPT}}"
VIDEOS_SUBSET="${VIDEOS_SUBSET:-8}"
VIDEOS_PROBE_K="${VIDEOS_PROBE_K:-0}"
VIDEOS_DECODE_BATCH="${VIDEOS_DECODE_BATCH:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"

if [ -z "${VIDEOS_NULL_ROOT}" ] && [ -z "${VIDEOS_POS_ROOT}" ]; then
  echo "FATAL: both capacity roots are empty -- nothing to render."
  exit 1
fi

# --- issue #4: prefetch the HF snapshot before JAX starts, with retries ------------------
bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_DIR}"

echo "RUN_NAME=${RUN_NAME} ATTEMPT=${ATTEMPT}"
echo "VIDEOS_NULL_ROOT=${VIDEOS_NULL_ROOT} -> VIDEOS_NULL_OUT=${VIDEOS_NULL_OUT}"
echo "VIDEOS_POS_ROOT=${VIDEOS_POS_ROOT} -> VIDEOS_POS_OUT=${VIDEOS_POS_OUT}"
echo "VIDEOS_SUBSET=${VIDEOS_SUBSET} VIDEOS_PROBE_K=${VIDEOS_PROBE_K} VIDEOS_DECODE_BATCH=${VIDEOS_DECODE_BATCH}"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

python src/maxdiffusion/run_wan_capacity_videos.py \
  src/maxdiffusion/configs/base_wan_5b_pos_inversion.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  videos_null_capacity_root="${VIDEOS_NULL_ROOT}" \
  videos_null_out="${VIDEOS_NULL_OUT}" \
  videos_pos_capacity_root="${VIDEOS_POS_ROOT}" \
  videos_pos_out="${VIDEOS_POS_OUT}" \
  videos_subset="${VIDEOS_SUBSET}" \
  videos_probe_k="${VIDEOS_PROBE_K}" \
  null_decode_batch_size="${VIDEOS_DECODE_BATCH}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu

echo "[capacity-videos] launcher done"
