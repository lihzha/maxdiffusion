#!/usr/bin/env bash
set -euo pipefail

# exp_05 K1: positive-context (8-token) inversion -- B-arm capacity study and adequacy probe -- on a
# v6e TPU. This is the SIBLING of bash_scripts/run_wan_null_inversion.sh, not an edit of it: exp_04's
# launcher is settled and exp_05 never edits exp_04's settled files (plan §6/F6). Same safety
# machinery, the positive slot's config and env interface.
#
# Why a second launcher at all: argv[1] decides the slot. The null YAML does not declare
# ``embedding_slot``, and pyconfig only coerces overrides for keys the YAML already declares, so
# ``embedding_slot=positive`` cannot ride exp_04's command line -- by design.
#
# Run
#   bash bash_scripts/setup.sh MODE=stable DEVICE=tpu
# once on the host before this script.

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

LOG_DIR="${LOG_DIR:-docs/worklogs_yixun/exp_05_pos_context_claude}"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y-%m-%d_%H:%M:%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/pos_context_${STAMP}.log}"

# --- the log is ALL terminal output, so tee comes before everything ----------------------
# Inherited from exp_04's R10 review (finding 11): a run that died in prefetch left a log file that
# did not mention prefetch. The HF prefetch, the preflight, the config echo and the git state are
# exactly the parts a post-mortem needs.
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "LOG_FILE=${LOG_FILE}"

if [ -f "$HOME/.config/irom-tpu/secrets.env" ]; then
  # Hide the secrets from an xtrace-enabled shell, then restore whatever xtrace state we were called
  # with -- unconditionally turning tracing back ON would leak every later expansion into this log.
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

# --- provenance: every published record is stamped with this, so it must be real -------------
# ``resolved_code_sha()`` refuses anything that is not 40 hex, and a positive-slot record carries the
# same stamp as a null one, so this fails here rather than after the arms have run.
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo '')}"
if ! printf '%s' "${COMMIT}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "FATAL: COMMIT is '${COMMIT}' -- exp_05 records carry a 40-hex code_sha or they are not published."
  echo "       Export COMMIT explicitly when running from an uploaded tarball with no git checkout."
  exit 1
fi
export COMMIT

# Copied verbatim from train_wan_side_adapter.sh (and identical to the null launcher's): the same 5B
# transformer, the same mesh.
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

RUN_NAME="${RUN_NAME:-wan-pos-inversion-smoke}"
# No ``:-``, for the same reason as the roots below: POS_MODE set-but-empty is a wrapper bug, and
# substituting the default there silently starts the multi-hour capacity job nobody asked for.
POS_MODE="${POS_MODE-capacity}"
POS_COHORT="${POS_COHORT:-dev64}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
POS_DATA_DIR="${POS_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"
# exp_05 consumes exp_04's published J0 manifests VERBATIM (plan §4 cohorts), so this default is the
# same ratified J0 mirror the null launcher names.
POS_MANIFEST_DIR="${POS_MANIFEST_DIR:-gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/}"
# Slot-isolated roots, under exp_05's own artifact tree. No ``:-`` on these two: an explicitly empty
# root is a wrapper bug (an unset variable interpolated into the launch env), and substituting the
# default there would publish K1 somewhere nobody asked for. Empty stays empty and is refused below.
POS_ARTIFACT_DIR="${POS_ARTIFACT_DIR-gs://v6_east1d/artifacts/exp05/${RUN_NAME}}"
POS_STAGING_DIR="${POS_STAGING_DIR-gs://v6_east1d/artifacts/exp05/${RUN_NAME}/_staging}"
POS_SELECTION_URI="${POS_SELECTION_URI:-}"
POS_ADEQUACY_URI="${POS_ADEQUACY_URI:-}"
POS_BATCH_SIZE="${POS_BATCH_SIZE:-8}"
POS_DECODE_BATCH_SIZE="${POS_DECODE_BATCH_SIZE:-8}"
POS_SMOKE_EXAMPLES="${POS_SMOKE_EXAMPLES:-0}"
POS_INNER_ITERS="${POS_INNER_ITERS:-10}"
POS_LR="${POS_LR:-0.01}"
POS_GUIDE_SCALE="${POS_GUIDE_SCALE:-5.0}"
POS_L="${POS_L:-8}"
POS_ABLATION_L="${POS_ABLATION_L:-1,8}"
POS_NOISE_CONVENTION="${POS_NOISE_CONVENTION:-keyed}"
POS_LATENT_DTYPE="${POS_LATENT_DTYPE:-fp16}"
POS_PIXEL_CONVENTION="${POS_PIXEL_CONVENTION:-unit}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"
POS_WATCHDOG_SECONDS="${POS_WATCHDOG_SECONDS:-0}"

# --- the cheap gates: everything decidable before a single process starts -----------------
# ``pos_execute`` wires exactly these two modes and raises on the rest -- but it raises after the
# manifests are read and the 5B model is on device. The same refusal costs nothing here.
case "${POS_MODE}" in
  capacity|adequacy_probe) ;;
  *)
    echo "FATAL: POS_MODE='${POS_MODE}' is not a mode the positive slot wires."
    echo "       The positive slot runs capacity|adequacy_probe; every other mode belongs to a later"
    echo "       round and pos_execute refuses it rather than falling through to the null slot."
    exit 1
    ;;
esac

# The authoritative slot isolation is the entrypoint's ``positive_roots`` (it normalizes and refuses
# the null slot's roots). This is the cheap net in front of it: an empty root cannot even be
# free-space checked, and an exp_04 root would put two experiments' selection.json in one directory.
check_positive_root() {
  if [ -z "$2" ]; then
    echo "FATAL: $1 is empty -- a positive-slot run publishes into its OWN root, and an empty root"
    echo "       cannot be free-space checked. Set $1 or leave it unset for the exp05 default."
    exit 1
  fi
  case "$2" in
    *exp04*)
      echo "FATAL: $1='$2' points into exp_04's artifact tree."
      echo "       K1 publishes under exp_05's own roots; sharing a tree with the null slot puts two"
      echo "       experiments' selection.json in one directory."
      exit 1
      ;;
  esac
}
check_positive_root POS_ARTIFACT_DIR "${POS_ARTIFACT_DIR}"
check_positive_root POS_STAGING_DIR "${POS_STAGING_DIR}"

# --- issue #4: prefetch the HF snapshot before JAX starts, with retries ------------------
bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_DIR}"

# --- issue #8 precedent: declared != installed. Preflight the imports, the ffmpeg binary, the
# --- pipeline class this driver actually loads, and the positive slot's own module.
PYTHONPATH="${PYTHONPATH:-src}" python - <<'PREFLIGHT'
import os
import sys

missing = []
for module in ("tensorflow", "skimage", "imageio", "imageio_ffmpeg", "jax", "numpy", "huggingface_hub"):
    try:
        __import__(module)
    except Exception as error:  # noqa: BLE001 -- any import failure is fatal here
        missing.append(f"{module}: {type(error).__name__}: {error}")

# The class the driver loads -- not the base WanPipeline, which has no from_pretrained. Importing it
# here turns "wrong class" into a preflight failure instead of a crash after the model is on device.
try:
    from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2

    if not callable(getattr(WanPipelineTI2V_2_2, "from_pretrained", None)):
        missing.append("WanPipelineTI2V_2_2 has no callable from_pretrained")
except Exception as error:  # noqa: BLE001
    missing.append(f"WanPipelineTI2V_2_2 import: {type(error).__name__}: {error}")

# The positive slot's own dispatch: main() imports these lazily, i.e. only once the slot resolves, so
# a broken positive module would otherwise surface after the config, manifests and model are loaded.
try:
    from maxdiffusion.pos_context_modes import pos_execute, positive_plan, positive_roots  # noqa: F401
except Exception as error:  # noqa: BLE001
    missing.append(f"maxdiffusion.pos_context_modes import: {type(error).__name__}: {error}")

ffmpeg = ""
try:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # os.access(X_OK), not "the path exists": a non-executable file passes an existence check and
    # then fails inside the writer, after the arms have already run.
    if not os.access(ffmpeg, os.X_OK):
        missing.append(f"imageio_ffmpeg reported {ffmpeg!r}, which is not executable")
except Exception as error:  # noqa: BLE001
    missing.append(f"imageio_ffmpeg.get_ffmpeg_exe(): {type(error).__name__}: {error}")

if missing:
    print("FATAL: exp_05 host preflight failed -- the positive slot cannot run here:")
    for line in missing:
        print(f"  - {line}")
    print("Install with: pip install scikit-image 'imageio[ffmpeg]' imageio-ffmpeg")
    sys.exit(1)
print(f"[preflight] ok: skimage/imageio/tensorflow/TI2V pipeline/pos modes importable, ffmpeg at {ffmpeg}")
PREFLIGHT

# --- the R1 noise golden, ON THIS DEVICE, before a single arm runs ------------------------
# The positive slot draws its B2/probe noise from exp_04's keyed_noise BY IMPORT (parity audit, item
# 11), so K1's artifacts are keyed to exactly this draw. If this backend's threefry differs from the
# one R1 pinned, every artifact this job publishes is keyed to noise nobody else can reproduce.
PYTHONPATH="${PYTHONPATH:-src}" python - <<'GOLDEN'
import sys

import numpy as np

from maxdiffusion.models.wan.null_inversion_wan import LATENT_SHAPE, NOISE_DOMAIN, keyed_noise

# Transcribed from test_null_adapter_noise.py's _GOLDEN_HEADS -- the R1 fingerprint, carried here so
# the same numbers are asserted on the TPU host that will produce the artifacts.
GOLDEN = {
    ("droid_ep_000001/w0", 0): (
        1.392072319984436,
        0.18953724205493927,
        -0.06578119099140167,
        -0.0243215449154377,
        0.2619726359844208,
        -0.3992597460746765,
        1.1612269878387451,
        0.13727153837680817,
    ),
    ("droid_ep_000001/w0", 1): (
        -0.26154080033302307,
        0.8063388466835022,
        1.614790439605713,
        -0.22073400020599365,
        1.955387830734253,
        -0.3821457326412201,
        0.6670055985450745,
        0.15746262669563293,
    ),
}

failures = []
if NOISE_DOMAIN != 0x4E4F4953 or LATENT_SHAPE != (48, 9, 12, 20):
    failures.append(f"noise constants drifted: NOISE_DOMAIN={NOISE_DOMAIN:#x} LATENT_SHAPE={LATENT_SHAPE}")
for (name, k), expected in sorted(GOLDEN.items()):
    head = np.asarray(keyed_noise(name, k)).reshape(-1)[:8]
    if not np.array_equal(head, np.asarray(expected, dtype=np.float32)):
        failures.append(f"keyed_noise({name!r}, {k}) head {head.tolist()} != golden {list(expected)}")

if failures:
    print("FATAL: the R1 noise golden does not hold on this device -- every artifact would be keyed")
    print("       to a draw nothing else can reproduce:")
    for line in failures:
        print(f"  - {line}")
    sys.exit(1)
print("[preflight] ok: R1 keyed-noise golden reproduces on this device")
GOLDEN

# --- the hard stop, opt-in -----------------------------------------------------------------
# A synchronous XLA compilation cannot be cancelled from inside the process blocked in it, so any real
# ceiling is an OUTER timeout. The positive slot wires no direct-optimization mode (capacity and
# adequacy_probe only), so nothing in K1's recipe implies a per-phase budget to derive one from:
# POS_WATCHDOG_SECONDS is off by default and job-level protection is the TPU queue's own timeout.
# When it is set, it is armed here -- and if `timeout` is unavailable the run says so rather than
# silently continuing without the stop the operator asked for.
# Unquoted on purpose below (word splitting is the point); every word here is a flag or an integer.
WATCHDOG_CMD=""
if [ "${POS_WATCHDOG_SECONDS}" != "0" ]; then
  if command -v timeout >/dev/null 2>&1; then
    WATCHDOG_CMD="timeout --signal=TERM --kill-after=60 ${POS_WATCHDOG_SECONDS}"
    echo "watchdog: ${POS_WATCHDOG_SECONDS}s external timeout armed (mode=${POS_MODE})"
  else
    echo "WARNING: POS_WATCHDOG_SECONDS=${POS_WATCHDOG_SECONDS} but 'timeout' is unavailable; no hard stop is armed."
  fi
else
  echo "watchdog: not armed (POS_WATCHDOG_SECONDS=0); job-level protection is the queue timeout."
fi

echo "RUN_NAME=${RUN_NAME}"
echo "POS_MODE=${POS_MODE}"
echo "POS_COHORT=${POS_COHORT}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "POS_DATA_DIR=${POS_DATA_DIR}"
echo "POS_MANIFEST_DIR=${POS_MANIFEST_DIR}"
echo "POS_ARTIFACT_DIR=${POS_ARTIFACT_DIR}"
echo "POS_STAGING_DIR=${POS_STAGING_DIR}"
echo "POS_SELECTION_URI=${POS_SELECTION_URI}"
echo "POS_ADEQUACY_URI=${POS_ADEQUACY_URI}"
echo "POS_BATCH_SIZE=${POS_BATCH_SIZE} POS_DECODE_BATCH_SIZE=${POS_DECODE_BATCH_SIZE}"
echo "POS_SMOKE_EXAMPLES=${POS_SMOKE_EXAMPLES}"
echo "POS_INNER_ITERS=${POS_INNER_ITERS} POS_LR=${POS_LR} POS_GUIDE_SCALE=${POS_GUIDE_SCALE}"
echo "POS_L=${POS_L} POS_ABLATION_L=${POS_ABLATION_L}"
echo "POS_NOISE_CONVENTION=${POS_NOISE_CONVENTION} POS_LATENT_DTYPE=${POS_LATENT_DTYPE}"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

# shellcheck disable=SC2086 -- WATCHDOG_CMD is a deliberate word-split list, empty when not armed.
${WATCHDOG_CMD} python src/maxdiffusion/run_wan_null_inversion.py \
  src/maxdiffusion/configs/base_wan_5b_pos_inversion.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  embedding_slot=positive \
  null_mode="${POS_MODE}" \
  null_cohort="${POS_COHORT}" \
  null_data_dir="${POS_DATA_DIR}" \
  null_manifest_dir="${POS_MANIFEST_DIR}" \
  pos_artifact_dir="${POS_ARTIFACT_DIR}" \
  pos_staging_dir="${POS_STAGING_DIR}" \
  pos_selection_uri="${POS_SELECTION_URI}" \
  pos_adequacy_uri="${POS_ADEQUACY_URI}" \
  null_batch_size="${POS_BATCH_SIZE}" \
  null_decode_batch_size="${POS_DECODE_BATCH_SIZE}" \
  null_smoke_examples="${POS_SMOKE_EXAMPLES}" \
  null_inner_iters="${POS_INNER_ITERS}" \
  null_lr="${POS_LR}" \
  null_guide_scale="${POS_GUIDE_SCALE}" \
  pos_L="${POS_L}" \
  pos_ablation_L="${POS_ABLATION_L}" \
  null_noise_convention="${POS_NOISE_CONVENTION}" \
  null_latent_dtype="${POS_LATENT_DTYPE}" \
  null_pixel_convention="${POS_PIXEL_CONVENTION}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
