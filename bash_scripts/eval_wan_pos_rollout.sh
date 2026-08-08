#!/usr/bin/env bash
set -euo pipefail

# exp_06 M4: the evaluation launcher -- anchor protocol, benchmark-row freeze, the two gates, and the
# TEST confirmation. Sibling of bash_scripts/train_wan_pos_rollout.sh; exp_05's and exp_04's settled
# launchers are read-only here, and this file deliberately is NOT named generate_wan_null_adapter.sh
# (exp_05's S9 carries a tripwire asserting that file does not exist -- plan §5-5).
#
# THE ORDER IS THE POINT (plan §3c). The anchor runs before anything new is measured, the benchmark
# row is frozen before any arm is scored against it, and TEST is confirmed only from a DEV
# certificate. The phases are separately invocable so the queue can retry one without re-running the
# rest, and each publishes into its own attempt-scoped root (issue #13).
#
# REVIEW PASS 3 (T6-2): the order was a comment, not a contract. `confirm` was accepted directly and
# the evaluator was handed no anchor certificate, no frozen benchmark row and no DEV certificate --
# and with separate attempt roots, a later phase could not even LOCATE its prerequisites. So each
# phase now DECLARES its prior-phase inputs, they are refused BEFORE the prefetch when absent, and a
# pre-prefetch check verifies each one parses, carries an issuing marker, and was issued by THIS
# code_sha. What those artifacts CONTAIN is the evaluator's contract (pos_rollout_gates,
# eval_wan_pos_rollout); this launcher transports them and refuses the shapes it can see.
#
# THE ROOTS ARE DERIVED, NEVER SUPPLIED (T6-3). A caller-supplied ARTIFACT_ROOT used to remove the
# phase/attempt scoping entirely. The customizable input is the storage PARENT; everything else is
# derived from parent + run + arm + phase + attempt.

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

LOG_DIR="${LOG_DIR:-docs/worklogs_yixun/exp_06_rollout_adapter_claude}"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y-%m-%d_%H:%M:%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/rollout_adapter_eval_${STAMP}.log}"

# --- tee before everything: a phase that dies in prefetch must still say so in the log ---------
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "LOG_FILE=${LOG_FILE}"

if [ -f "$HOME/.config/irom-tpu/secrets.env" ]; then
  # issue #12: preserve the CALLER's xtrace state; never force it back on.
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
  echo "FATAL: COMMIT is '${COMMIT}' -- every certificate exp_06 publishes carries a 40-hex code_sha."
  exit 1
fi
export COMMIT

# No ``:-``: an explicitly empty phase or arm is a wrapper bug, and defaulting it would run the
# anchor when the operator meant the gates, or score R-B's tree as matched-C0's.
POS_EVAL_PHASE="${POS_EVAL_PHASE-anchor}"
POS_ROLLOUT_ARM="${POS_ROLLOUT_ARM-rollout}"

RUN_NAME="${RUN_NAME:-wan-pos-rollout-smoke}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pos-rollout}"
POS_DEV_MANIFEST="${POS_DEV_MANIFEST:-docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests/dev64.json}"
POS_TEST_MANIFEST="${POS_TEST_MANIFEST:-docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests/test64.json}"
SAMPLING_STEPS="${SAMPLING_STEPS:-25}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
POS_ROLLOUT_K="${POS_ROLLOUT_K:-2}"
SEED="${SEED:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1.0}"
HARDWARE="${HARDWARE:-tpu}"
# The trained attempt being evaluated. A single derived path SEGMENT, never a root: the launcher
# still builds the namespace, so no caller can point the evaluator outside the run it names.
POS_CHECKPOINT_ATTEMPT="${POS_CHECKPOINT_ATTEMPT:-}"
# The prior-phase artifacts. Each phase declares which of these it requires; see the table below.
POS_ANCHOR_CERTIFICATE="${POS_ANCHOR_CERTIFICATE:-}"
POS_BENCHMARK_ROW="${POS_BENCHMARK_ROW:-}"
POS_DEV_CERTIFICATE="${POS_DEV_CERTIFICATE:-}"

case "${POS_EVAL_PHASE}" in
  anchor|benchmark|gates|confirm) ;;
  *)
    echo "FATAL: POS_EVAL_PHASE='${POS_EVAL_PHASE}' is not a phase this evaluator wires."
    echo "       The order is the protocol (plan §3c): anchor -> benchmark -> gates -> confirm."
    echo "       'confirm' scores TEST and runs only from a DEV certificate the gates phase issued."
    exit 1
    ;;
esac

case "${POS_ROLLOUT_ARM}" in
  rollout|one_step) ;;
  *)
    echo "FATAL: POS_ROLLOUT_ARM='${POS_ROLLOUT_ARM}' is not an arm exp_06 declares."
    exit 1
    ;;
esac

# --- THE PHASE PREREQUISITE TABLE (T6-2) -------------------------------------------------------
# Each phase names the prior-phase artifact it cannot run without. Refused HERE, before the prefetch.
REQUIRED_INPUTS=""
case "${POS_EVAL_PHASE}" in
  anchor) REQUIRED_INPUTS="" ;;
  benchmark) REQUIRED_INPUTS="POS_ANCHOR_CERTIFICATE" ;;
  gates) REQUIRED_INPUTS="POS_ANCHOR_CERTIFICATE POS_BENCHMARK_ROW" ;;
  confirm) REQUIRED_INPUTS="POS_ANCHOR_CERTIFICATE POS_BENCHMARK_ROW POS_DEV_CERTIFICATE" ;;
esac
for required in ${REQUIRED_INPUTS}; do
  eval "value=\${${required}}"
  if [ -z "${value}" ]; then
    echo "FATAL: phase '${POS_EVAL_PHASE}' requires ${required}, and it is empty."
    echo "       The protocol's order is a dependency, not a convention: the anchor certifies the"
    echo "       wiring, the benchmark row freezes the baseline every table carries, and TEST is"
    echo "       scored only from the PASSING DEV certificate the gates phase issued (plan §3c)."
    echo "       Each phase publishes into its own attempt-scoped root, so pass the path it printed."
    exit 1
  fi
done

ATTEMPT="${ATTEMPT:-att-$(date -u +%Y%m%dT%H%M%SZ)}"
if ! printf '%s' "${ATTEMPT}" | grep -Eq '^att-[A-Za-z0-9._-]+$'; then
  echo "FATAL: ATTEMPT='${ATTEMPT}' is not an attempt id (att-<token>)."
  exit 1
fi
if [ -n "${POS_CHECKPOINT_ATTEMPT}" ] && ! printf '%s' "${POS_CHECKPOINT_ATTEMPT}" | grep -Eq '^att-[A-Za-z0-9._-]+$'; then
  echo "FATAL: POS_CHECKPOINT_ATTEMPT='${POS_CHECKPOINT_ATTEMPT}' is not an attempt id (att-<token>);"
  echo "       it is a path SEGMENT inside this run's namespace, never a root."
  exit 1
fi
if [ -z "${OUTPUT_DIR}" ] || [ -z "${RUN_NAME}" ]; then
  echo "FATAL: OUTPUT_DIR and RUN_NAME must be non-empty -- every root is derived from them."
  exit 1
fi

TRAIN_NAMESPACE="${OUTPUT_DIR}/${RUN_NAME}/${POS_ROLLOUT_ARM}/train"
RESUME_PARENT="${TRAIN_NAMESPACE}/attempts"
if [ -n "${POS_CHECKPOINT_ATTEMPT}" ]; then
  CHECKPOINT_DIR="${RESUME_PARENT}/${POS_CHECKPOINT_ATTEMPT}/checkpoints"
else
  # Unset means "the latest COMPLETE publication for this arm at this SHA"; the evaluator resolves it
  # from RESUME_PARENT with the same selector the trainer uses, so an incomplete or foreign-arm
  # attempt is never adopted.
  CHECKPOINT_DIR="${RESUME_PARENT}"
fi
ARTIFACT_ROOT="${OUTPUT_DIR}/${RUN_NAME}/${POS_ROLLOUT_ARM}/eval/${POS_EVAL_PHASE}/attempts/${ATTEMPT}"
RUN_REPORT="${RUN_REPORT:-${TRAIN_NAMESPACE}/run_report.json}"

CONFIG="src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml"
ENTRYPOINT="src/maxdiffusion/eval_wan_pos_rollout.py"
for required in "${ENTRYPOINT}" "${CONFIG}"; do
  if [ ! -f "${required}" ]; then
    echo "FATAL: ${required} is not in this working tree -- the tarball is incomplete."
    exit 1
  fi
done

cat <<'RECONCILE'
[note] A checkpoint root whose _selection sibling cannot be reconciled with its history FAILS CLOSED
       by design; the evaluator restores the SELECTION artifact and never the resume tree's latest.
       "there is no selection artifact" means the run published none, not that this job crashed.
RECONCILE

# --- prior-phase artifacts, verified BEFORE the prefetch (T6-2) --------------------------------
# Shape only: it must exist, parse as an object, carry an issuing marker, and -- where it records one
# -- have been issued by THIS code_sha. A `confirm` that presents a FAILING DEV certificate is
# refused here as well as in the evaluator. What the artifacts CONTAIN is the evaluator's contract.
PYTHONPATH="${PYTHONPATH:-src}" POS_EVAL_PHASE="${POS_EVAL_PHASE}" \
  POS_REQUIRED_INPUTS="${REQUIRED_INPUTS}" POS_ANCHOR_CERTIFICATE="${POS_ANCHOR_CERTIFICATE}" \
  POS_BENCHMARK_ROW="${POS_BENCHMARK_ROW}" POS_DEV_CERTIFICATE="${POS_DEV_CERTIFICATE}" python - <<'PREREQ'
import json
import os
import sys

required = os.environ.get("POS_REQUIRED_INPUTS", "").split()
if not required:
    print(f"[prereq] {os.environ['POS_EVAL_PHASE']}: the first phase of the protocol; nothing precedes it")
    sys.exit(0)

from maxdiffusion.pos_rollout_support import storage_exists, storage_read_bytes

failures = []
for name in required:
    path = os.environ.get(name, "")
    if not storage_exists(path):
        failures.append(f"{name}={path}: no such artifact")
        continue
    try:
        payload = json.loads(storage_read_bytes(path).decode("utf-8"))
    except Exception as error:  # noqa: BLE001
        failures.append(f"{name}={path}: not readable JSON ({type(error).__name__}: {error})")
        continue
    body = payload.get("payload") if isinstance(payload, dict) and "payload" in payload else payload
    if not isinstance(body, dict):
        failures.append(f"{name}={path}: not an issued artifact (a JSON object was expected)")
        continue
    if not (body.get("protocol") or body.get("certificate")):
        failures.append(f"{name}={path}: carries no issuing marker, so nothing issued it")
    recorded = str(body.get("code_sha", "")) if "code_sha" in body else ""
    if recorded and recorded != os.environ.get("COMMIT", ""):
        failures.append(f"{name}={path}: issued at {recorded}, this job runs {os.environ.get('COMMIT')}")
    if name == "POS_DEV_CERTIFICATE" and "passed" in body and not body.get("passed"):
        failures.append(f"{name}={path}: the DEV primary gate did NOT pass, so TEST may not be scored")

if failures:
    print("FATAL: this phase's prerequisites are missing or inconsistent:")
    for line in failures:
        print(f"  - {line}")
    sys.exit(1)
print(f"[prereq] {os.environ['POS_EVAL_PHASE']}: {', '.join(required)} present, issued and SHA-consistent")
PREREQ

bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_DIR}"

PYTHONPATH="${PYTHONPATH:-src}" python - <<'PREFLIGHT'
import sys

missing = []
for module in ("tensorflow", "skimage", "jax", "numpy", "orbax.checkpoint"):
    try:
        __import__(module)
    except Exception as error:  # noqa: BLE001
        missing.append(f"{module}: {type(error).__name__}: {error}")

# The evaluation path's own modules, plus the two constants a mis-wired job would otherwise discover
# only after the 5B model was loaded and the rollout had run.
try:
    from maxdiffusion.eval_wan_pos_rollout import DEPLOYED_SAMPLING_STEPS, HISTORICAL_ANCHOR
    from maxdiffusion.pos_rollout_gates import GATE_CERTIFICATE, PRIMARY_MARGIN  # noqa: F401
except Exception as error:  # noqa: BLE001
    missing.append(f"maxdiffusion exp_06 eval modules: {type(error).__name__}: {error}")
else:
    if float(HISTORICAL_ANCHOR.mean_ssim) != 0.2946 or int(DEPLOYED_SAMPLING_STEPS) != 25:
        missing.append(
            f"anchor/grid drifted: ssim={HISTORICAL_ANCHOR.mean_ssim} steps={DEPLOYED_SAMPLING_STEPS}"
        )
    if float(PRIMARY_MARGIN) != 0.05:
        missing.append(f"the primary gate's margin drifted to {PRIMARY_MARGIN}")

if missing:
    print("FATAL: preflight failed -- these would have surfaced after the 5B model was on device:")
    for line in missing:
        print(f"  - {line}")
    sys.exit(1)
print("[preflight] ok: imports, the published anchor, the deployed grid and the gate margin")
PREFLIGHT

echo "RUN_NAME=${RUN_NAME}"
echo "POS_EVAL_PHASE=${POS_EVAL_PHASE}   (order: anchor -> benchmark -> gates -> confirm)"
echo "POS_ROLLOUT_ARM=${POS_ROLLOUT_ARM}"
echo "ATTEMPT=${ATTEMPT}"
echo "ARTIFACT_ROOT=${ARTIFACT_ROOT}   (phase- and attempt-scoped -- issue #13)"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "RESUME_PARENT=${RESUME_PARENT}   (latest COMPLETE publication for this arm at this SHA)"
echo "RUN_REPORT=${RUN_REPORT}"
echo "POS_ANCHOR_CERTIFICATE=${POS_ANCHOR_CERTIFICATE:-<not required by this phase>}"
echo "POS_BENCHMARK_ROW=${POS_BENCHMARK_ROW:-<not required by this phase>}"
echo "POS_DEV_CERTIFICATE=${POS_DEV_CERTIFICATE:-<not required by this phase>}"
echo "POS_DEV_MANIFEST=${POS_DEV_MANIFEST}"
echo "POS_TEST_MANIFEST=${POS_TEST_MANIFEST}"
echo "SAMPLING_STEPS=${SAMPLING_STEPS} GUIDE_SCALE=${GUIDE_SCALE} POS_ROLLOUT_K=${POS_ROLLOUT_K}"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

python "${ENTRYPOINT}" "${CONFIG}" \
  run_name="${RUN_NAME}" \
  pretrained_model_name_or_path="${MODEL_DIR}" \
  output_dir="${OUTPUT_DIR}" \
  pos_eval_phase="${POS_EVAL_PHASE}" \
  pos_rollout_arm="${POS_ROLLOUT_ARM}" \
  base_output_directory="${ARTIFACT_ROOT}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  pos_resume_parent="${RESUME_PARENT}" \
  pos_run_report="${RUN_REPORT}" \
  pos_anchor_certificate="${POS_ANCHOR_CERTIFICATE}" \
  pos_benchmark_row="${POS_BENCHMARK_ROW}" \
  pos_dev_certificate="${POS_DEV_CERTIFICATE}" \
  pos_dev_manifest="${POS_DEV_MANIFEST}" \
  pos_test_manifest="${POS_TEST_MANIFEST}" \
  side_adapter_sampling_steps="${SAMPLING_STEPS}" \
  side_adapter_guide_scale="${GUIDE_SCALE}" \
  pos_rollout_k="${POS_ROLLOUT_K}" \
  seed="${SEED}" \
  per_device_batch_size="${PER_DEVICE_BATCH_SIZE}" \
  hardware="${HARDWARE}"
