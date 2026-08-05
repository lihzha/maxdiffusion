#!/usr/bin/env bash
set -euo pipefail

# exp_04 J1: null-text inversion capacity study / target caching on a v6e TPU.
# Env-configured in the style of bash_scripts/train_wan_side_adapter.sh; run
#   bash bash_scripts/setup.sh MODE=stable DEVICE=tpu
# once on the host before this script.

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

LOG_DIR="${LOG_DIR:-docs/worklogs_yixun/exp_04_null_adapter_claude}"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y-%m-%d_%H:%M:%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/null_adapter_${STAMP}.log}"

# --- the log is ALL terminal output, so tee comes before everything ----------------------
# The HF prefetch, the preflight, the config echo and the git state are exactly the parts a
# post-mortem needs, and R10 emitted every one of them before the pipe was open: a run that died in
# prefetch left a log file that did not mention prefetch (R10 review, finding 11).
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "LOG_FILE=${LOG_FILE}"

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

# --- provenance: every published record is stamped with this, so it must be real -------------
# R10 printed COMMIT and never exported it, so the driver's os.environ.get("COMMIT", "unknown")
# would have stamped every record in the P2 cache with code_sha="unknown" (review, finding 7).
# resolved_code_sha() refuses anything that is not 40 hex, so this fails here rather than there.
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo '')}"
if ! printf '%s' "${COMMIT}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "FATAL: COMMIT is '${COMMIT}' -- exp_04 records carry a 40-hex code_sha or they are not published."
  echo "       Export COMMIT explicitly when running from an uploaded tarball with no git checkout."
  exit 1
fi
export COMMIT

# Copied verbatim from train_wan_side_adapter.sh: the same 5B transformer, the same mesh.
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

RUN_NAME="${RUN_NAME:-wan-null-inversion-smoke}"
NULL_MODE="${NULL_MODE:-capacity}"
NULL_COHORT="${NULL_COHORT:-dev64}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
NULL_DATA_DIR="${NULL_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/val}"
# The ratified J0 mirror (worklog, J0-1 acceptance criterion 7).
NULL_MANIFEST_DIR="${NULL_MANIFEST_DIR:-gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/}"
NULL_ARTIFACT_DIR="${NULL_ARTIFACT_DIR:-gs://v6_east1d/artifacts/exp04/${RUN_NAME}}"
NULL_STAGING_DIR="${NULL_STAGING_DIR:-gs://v6_east1d/artifacts/exp04/${RUN_NAME}/_staging}"
NULL_SELECTION_URI="${NULL_SELECTION_URI:-}"
NULL_BATCH_SIZE="${NULL_BATCH_SIZE:-8}"
NULL_DECODE_BATCH_SIZE="${NULL_DECODE_BATCH_SIZE:-8}"
NULL_SMOKE_EXAMPLES="${NULL_SMOKE_EXAMPLES:-0}"
NULL_INNER_ITERS="${NULL_INNER_ITERS:-10}"
NULL_LR="${NULL_LR:-0.01}"
NULL_GUIDE_SCALE="${NULL_GUIDE_SCALE:-5.0}"
NULL_L="${NULL_L:-16}"
NULL_NOISE_CONVENTION="${NULL_NOISE_CONVENTION:-keyed}"
NULL_LATENT_DTYPE="${NULL_LATENT_DTYPE:-fp16}"
NULL_PIXEL_CONVENTION="${NULL_PIXEL_CONVENTION:-unit}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"

# --- issue #4: prefetch the HF snapshot before JAX starts, with retries ------------------
bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_DIR}"

# --- issue #8 precedent: declared != installed. Preflight the imports, the ffmpeg binary, the
# --- pipeline class this driver actually loads, and the R1 noise golden on this device.
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

ffmpeg = ""
try:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # os.access(X_OK), not "the path exists": a non-executable file passes an existence check and
    # then fails inside the writer, after the arms have already run (review, finding 11).
    if not os.access(ffmpeg, os.X_OK):
        missing.append(f"imageio_ffmpeg reported {ffmpeg!r}, which is not executable")
except Exception as error:  # noqa: BLE001
    missing.append(f"imageio_ffmpeg.get_ffmpeg_exe(): {type(error).__name__}: {error}")

if missing:
    print("FATAL: exp_04 host preflight failed -- comparison videos and SSIM cannot be produced:")
    for line in missing:
        print(f"  - {line}")
    print("Install with: pip install scikit-image 'imageio[ffmpeg]' imageio-ffmpeg")
    sys.exit(1)
print(f"[preflight] ok: skimage/imageio/tensorflow/TI2V pipeline importable, ffmpeg at {ffmpeg}")
PREFLIGHT

# --- the R1 noise golden, ON THIS DEVICE, before a single arm runs ------------------------
# Every cached target is bound to a noise convention; if this backend's threefry draw differs from
# the one R1 pinned on CPU, every artifact this job publishes is keyed to noise nobody else can
# reproduce. Cheap, and it fails before any compute is spent (review, finding 11).
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

echo "RUN_NAME=${RUN_NAME}"
echo "NULL_MODE=${NULL_MODE}"
echo "NULL_COHORT=${NULL_COHORT}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "NULL_DATA_DIR=${NULL_DATA_DIR}"
echo "NULL_MANIFEST_DIR=${NULL_MANIFEST_DIR}"
echo "NULL_ARTIFACT_DIR=${NULL_ARTIFACT_DIR}"
echo "NULL_SELECTION_URI=${NULL_SELECTION_URI}"
echo "NULL_BATCH_SIZE=${NULL_BATCH_SIZE} NULL_DECODE_BATCH_SIZE=${NULL_DECODE_BATCH_SIZE}"
echo "NULL_SMOKE_EXAMPLES=${NULL_SMOKE_EXAMPLES}"
echo "NULL_INNER_ITERS=${NULL_INNER_ITERS} NULL_LR=${NULL_LR} NULL_GUIDE_SCALE=${NULL_GUIDE_SCALE}"
echo "NULL_NOISE_CONVENTION=${NULL_NOISE_CONVENTION} NULL_LATENT_DTYPE=${NULL_LATENT_DTYPE}"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

python src/maxdiffusion/run_wan_null_inversion.py \
  src/maxdiffusion/configs/base_wan_5b_null_inversion.yml \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  null_mode="${NULL_MODE}" \
  null_cohort="${NULL_COHORT}" \
  null_data_dir="${NULL_DATA_DIR}" \
  null_manifest_dir="${NULL_MANIFEST_DIR}" \
  null_artifact_dir="${NULL_ARTIFACT_DIR}" \
  null_staging_dir="${NULL_STAGING_DIR}" \
  null_selection_uri="${NULL_SELECTION_URI}" \
  null_batch_size="${NULL_BATCH_SIZE}" \
  null_decode_batch_size="${NULL_DECODE_BATCH_SIZE}" \
  null_smoke_examples="${NULL_SMOKE_EXAMPLES}" \
  null_inner_iters="${NULL_INNER_ITERS}" \
  null_lr="${NULL_LR}" \
  null_guide_scale="${NULL_GUIDE_SCALE}" \
  null_L="${NULL_L}" \
  null_noise_convention="${NULL_NOISE_CONVENTION}" \
  null_latent_dtype="${NULL_LATENT_DTYPE}" \
  null_pixel_convention="${NULL_PIXEL_CONVENTION}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware=tpu
