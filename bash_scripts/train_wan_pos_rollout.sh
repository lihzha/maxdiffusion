#!/usr/bin/env bash
set -euo pipefail

# exp_06 M1/M2/M3: rollout-objective training of the pre_context adapter on the frozen Wan2.2 TI2V 5B.
# A SIBLING of bash_scripts/train_wan_side_adapter.sh and of exp_05's run_wan_pos_inversion.sh, not an
# edit of either: settled launchers stay settled (plan §5-7, §6/F6).
#
# THE COMPARABILITY CONTRACT (Yixun's decision 1): R-B and matched-C0 differ by ONE variable, the
# objective. Review pass 3 found that the first version of this file did the opposite of that: both
# arms got the same run name AND the same checkpoint root, so running R-B and then matched-C0 could
# make C0 RESTORE R-B's parameters, optimizer state, step and history -- and two concurrent arms
# collided outright. Three structural changes, in this file and in the trainer it launches:
#
#   1. ONE COMMON RECIPE ARRAY.  Every override that is not the arm or a destination is built once,
#      in ${COMMON_RECIPE}, and both arms are emitted from it. The arm reaches exactly one override
#      and the three destinations derived from it; a test executes both arms and pins that the two
#      command lines differ in nothing else.
#   2. ARM-SCOPED NAMESPACES.  Checkpoints, artifacts and the resume search root all carry the arm,
#      so the two arms cannot share state even if an operator tries.
#   3. THE RECIPE LOCK.  Two queue submissions a day apart could still differ in a seed, a learning
#      rate or a data directory, and no shell can see that. So the first arm of a run PUBLISHES its
#      normalized recipe and the second ADOPTS it or refuses to start, naming the keys that differ
#      (trainers/wan_pos_rollout_trainer.py: publish_recipe_lock). Divergence is not detected after
#      the fact; it cannot start.
#
# THE ROOTS ARE DERIVED, NEVER SUPPLIED (issue #13, review pass 3 T6-3). A caller used to be able to
# pass a complete ARTIFACT_ROOT or CHECKPOINT_DIR and remove phase/attempt/arm scoping entirely. Now
# the customizable input is the storage PARENT; the launcher derives
#   <parent>/<run>/<arm>/<mode>/attempts/<attempt>/{checkpoints,artifacts}
# and the immutable resume INPUT is the attempts root, distinct from this attempt's fresh OUTPUT. A
# retry adopts the latest COMPLETE publication whose recorded SHA is the code now running.
#
# TWO JOB MODES. POS_JOB_MODE=fit_probe runs M1 (plan §4-P1: the microbatch x k ladder, per-arm, with
# the 90% headroom rule) and PUBLISHES the authorization; POS_JOB_MODE=train runs M2/M3 and must
# PRESENT one. The plan requires an M1 mode on this launcher and the first version had none.
#
# Run
#   bash bash_scripts/setup.sh MODE=stable DEVICE=tpu
# once on the host before this script.

ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

LOG_DIR="${LOG_DIR:-docs/worklogs_yixun/exp_06_rollout_adapter_claude}"
mkdir -p "${LOG_DIR}"
STAMP="$(date -u +%Y-%m-%d_%H:%M:%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/rollout_adapter_${STAMP}.log}"

# --- the log is ALL terminal output, so tee comes before everything ----------------------
# exp_04's R10 review, finding 11: a run that died in prefetch left a log that never mentioned
# prefetch. Nothing that can fail may run before this line.
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "LOG_FILE=${LOG_FILE}"

if [ -f "$HOME/.config/irom-tpu/secrets.env" ]; then
  # issue #12: hide the secrets from an xtrace-enabled shell, then restore the caller's OWN xtrace
  # state. exp_04's launcher force-enables `set -x` here, which sprays every later expansion into a
  # teed log; that is the defect this line exists not to reproduce.
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

# Unbuffered stdout, so per-cell progress reaches the teed log as it happens rather than at exit.
# F3 note: this was ALREADY here during the three blind M1 attempts, so buffering was not why those
# logs were empty -- the probe had no statement to print until a cell FINISHED, and no cell ever did
# (it was stuck compiling a module carrying 10.18 GB of captured constants). The fix for the silence
# is `[M1] entering ...` in pos_rollout_fit_probe._measure_under_mesh, not a flag here. Adding `-u`
# to the interpreter would be redundant with this and would shift argv, which three launcher contract
# tests read.
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

# --- provenance: a run whose SHA is not real cannot be adopted or reproduced ------------------
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo '')}"
if ! printf '%s' "${COMMIT}" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "FATAL: COMMIT is '${COMMIT}' -- exp_06 checkpoints and certificates carry a 40-hex code_sha."
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

# =================================================================================================
# THE LAUNCHER INTERFACE. Every variable below maps to exactly one config key, and every default
# equals the checked-in YAML's value -- except the run-identity, derived-storage and derived-topology
# keys, which are exempt BY NAME in the interface table the test file carries. A launcher that
# recipes differently from the config everyone reads is exactly the drift the SOP's parity audit
# exists for. `per_device_batch_size` is the third category and the only one added since: it is
# DERIVED below from POS_DEVICE_COUNT, because no YAML default can be right on both v6e-8 and v6e-64.
# =================================================================================================

# No ``:-``: an explicitly empty mode or arm is a wrapper bug, and substituting the default would
# silently run the wrong job -- the one substitution that invalidates the pilot.
POS_JOB_MODE="${POS_JOB_MODE-train}"
POS_ROLLOUT_ARM="${POS_ROLLOUT_ARM-rollout}"

RUN_NAME="${RUN_NAME:-wan-pos-rollout-smoke}"
MODEL_DIR="${MODEL_DIR:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-gs://v6_east1d/datasets/droid_wan_side_adapter/train}"
POS_DEV_MANIFEST="${POS_DEV_MANIFEST:-docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests/dev64.json}"
POS_DEV_MANIFEST_SHA256="${POS_DEV_MANIFEST_SHA256:-3c59d023f3b782542ecae443b8d83008e7d8dfd801347f41adfab75218340836}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
POS_LOGICAL_BATCH="${POS_LOGICAL_BATCH:-256}"
POS_MICROBATCH="${POS_MICROBATCH:-32}"
POS_ROLLOUT_K="${POS_ROLLOUT_K:-2}"
POS_SUPPORT_SALT="${POS_SUPPORT_SALT:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-25}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
NOISE_MODE="${NOISE_MODE:-fresh}"
DROPOUT="${DROPOUT:-0.0}"
LEARNING_RATE="${LEARNING_RATE:-5e-05}"
SEED="${SEED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
HARDWARE="${HARDWARE:-tpu}"
# The customizable STORAGE PARENT. This is the only path input; every root below is derived from it,
# so no caller can flatten the phase/attempt/arm scoping the queue's auto-retry depends on.
OUTPUT_DIR="${OUTPUT_DIR:-gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pos-rollout}"
# M1's published authorization. In `train` mode it is a REQUIRED input -- an operator who has not run
# M1 is refused by name, before the prefetch, rather than five minutes into a pipeline load. In
# `fit_probe` mode it is DERIVED, because that mode is the thing that publishes it.
POS_FIT_AUTHORIZATION="${POS_FIT_AUTHORIZATION:-}"
# F5: where the fit probe LOOKS for cells a prior attempt of this same job already published. Left
# unset in `fit_probe` mode it defaults to this job's attempts root, which is correct whenever
# OUTPUT_DIR is stable across attempts. The submit wrapper scopes OUTPUT_DIR PER ATTEMPT
# (`$M1ROOT/$ATT`), which puts each attempt's tree in its own OUTPUT_DIR and makes the attempts root
# empty every time -- so that wrapper must pass POS_FIT_ADOPTION_ROOT="$M1ROOT" for adoption to see
# anything. Inert in `train` mode: nothing there adopts cells.
POS_FIT_ADOPTION_ROOT="${POS_FIT_ADOPTION_ROOT:-}"
# F6: cells this run DECLARES unreachable ('arm:mb:k', comma separated) and the required reason.
# EMPTY by default: populating it publishes a table authorizing less than the ladder was designed to
# measure, which is a plan deviation and needs Yixun's decision, not an operator's.
POS_FIT_EXCLUDED_CELLS="${POS_FIT_EXCLUDED_CELLS:-}"
POS_FIT_EXCLUSION_REASON="${POS_FIT_EXCLUSION_REASON:-}"

case "${POS_JOB_MODE}" in
  train|fit_probe) ;;
  *)
    echo "FATAL: POS_JOB_MODE='${POS_JOB_MODE}' is not a job this launcher runs."
    echo "       fit_probe = M1 (walk the ladder, publish the authorization); train = M2/M3."
    exit 1
    ;;
esac

case "${POS_ROLLOUT_ARM}" in
  rollout|one_step) ;;
  *)
    echo "FATAL: POS_ROLLOUT_ARM='${POS_ROLLOUT_ARM}' is not an arm exp_06 declares."
    echo "       The pilot runs rollout (R-B) and one_step (matched-C0); every other arm is a"
    echo "       conditional admission that needs a recorded decision first (plan §3)."
    exit 1
    ;;
esac

# --- W2b: the per-device batch is DERIVED from the topology, never typed -----------------------
# `pyconfig` computes global_batch_size_to_load = device_count x per_device_batch_size, and the
# trainer refuses to start unless that equals `pos_logical_batch`: accumulation PRESERVES the logical
# batch and never adapts to what the loader handed it (plan §3b). So the correct per-device value is
# a function of the TOPOLOGY -- 32.0 on v6e-8, 4.0 on v6e-64 at the pilot's GBS 256 -- and no single
# YAML default can be right for both. This launcher's previous PER_DEVICE_BATCH_SIZE=1.0 made every
# M2/M3 submission fail at startup, and no operator-facing value fixes that: a documented number is
# one an operator can still get wrong, so the number is not an input any more.
#
# The operator declares the ONE thing they necessarily know -- they chose the accelerator -- and the
# arithmetic happens here. No `:-`: an empty declaration is a wrapper bug, and defaulting it would
# run a different global batch in silence (S10a's no-colon lesson).
#
# DECLARED -> DERIVED -> VERIFIED, and the verification is deliberately not here. A preflight cannot
# check the declaration: before the distributed system is initialized `jax.device_count()` is the
# LOCAL count, so a preflight on a v6e-64 job would see 8 chips, derive 256/8 = 32, and train at a
# global batch of 2048 on every host. A check that is right on one topology and silently wrong on the
# other is worse than no check. The authoritative number exists inside the real process, and that is
# where a wrong declaration dies: `assert_loader_yields_the_logical_batch` names the mismatch and the
# knob, in seconds, before the pipeline load.
if ! printf '%s' "${POS_LOGICAL_BATCH}" | grep -Eq '^[1-9][0-9]*$'; then
  echo "FATAL: POS_LOGICAL_BATCH='${POS_LOGICAL_BATCH}' is not a positive whole number of examples."
  exit 1
fi
POS_DEVICE_COUNT="${POS_DEVICE_COUNT-}"
if ! printf '%s' "${POS_DEVICE_COUNT}" | grep -Eq '^[1-9][0-9]*$'; then
  echo "FATAL: POS_DEVICE_COUNT='${POS_DEVICE_COUNT}' is not a chip count."
  echo "       Declare the topology this job is submitted to and nothing else: POS_DEVICE_COUNT=8"
  echo "       for v6e-8 (M1/M2), POS_DEVICE_COUNT=64 for v6e-64 (M1'/M3). The per-device batch is"
  echo "       derived from it, so there is no second number to keep in step by hand."
  exit 1
fi
if [ "$(( POS_LOGICAL_BATCH % POS_DEVICE_COUNT ))" -ne 0 ]; then
  echo "FATAL: POS_LOGICAL_BATCH=${POS_LOGICAL_BATCH} is not divisible by POS_DEVICE_COUNT=${POS_DEVICE_COUNT}."
  echo "       No per-device batch loads exactly ${POS_LOGICAL_BATCH} examples per step on"
  echo "       ${POS_DEVICE_COUNT} chips, and accumulation preserves the logical batch rather than"
  echo "       rounding it (plan §3b)."
  exit 1
fi
# Emitted as a float because that is the YAML key's type; pyconfig coerces to the declared type.
DERIVED_PER_DEVICE_BATCH="$(( POS_LOGICAL_BATCH / POS_DEVICE_COUNT )).0"

# --- issue #13: derived roots, attempt-scoped OUTPUT, immutable resume INPUT --------------------
# The queue's auto-retry re-runs a job from the top, and the storage layer correctly refuses to
# rewrite a published artifact -- so every immutable destination is attempt-scoped. The RESUME input
# is the attempts ROOT, not this attempt's own tree: the trainer adopts the latest COMPLETE
# publication under it whose recorded code_sha is the code now running, and writes here.
ATTEMPT="${ATTEMPT:-att-$(date -u +%Y%m%dT%H%M%SZ)}"
if ! printf '%s' "${ATTEMPT}" | grep -Eq '^att-[A-Za-z0-9._-]+$'; then
  echo "FATAL: ATTEMPT='${ATTEMPT}' is not an attempt id (att-<token>); attempt roots are derived,"
  echo "       and a caller-shaped path segment would let a job escape its own namespace."
  exit 1
fi
if [ -z "${OUTPUT_DIR}" ] || [ -z "${RUN_NAME}" ]; then
  echo "FATAL: OUTPUT_DIR and RUN_NAME must be non-empty -- every root is derived from them."
  exit 1
fi

# M1 measures BOTH arms in one job (the ladder is arm x microbatch x k), so its namespace is
# run-level; training is per-arm, and the arm is IN the path so the two arms cannot share state.
if [ "${POS_JOB_MODE}" = "fit_probe" ]; then
  ARM_NAMESPACE="${OUTPUT_DIR}/${RUN_NAME}/fit_probe"
else
  ARM_NAMESPACE="${OUTPUT_DIR}/${RUN_NAME}/${POS_ROLLOUT_ARM}/train"
fi
RESUME_PARENT="${ARM_NAMESPACE}/attempts"
ATTEMPT_ROOT="${RESUME_PARENT}/${ATTEMPT}"
CHECKPOINT_DIR="${ATTEMPT_ROOT}/checkpoints"
ARTIFACT_ROOT="${ATTEMPT_ROOT}/artifacts"
# The recipe lock is RUN-level, deliberately NOT arm-scoped: it is the artifact the two arms share,
# and sharing it is what makes a divergent second arm refuse to start.
RECIPE_LOCK="${OUTPUT_DIR}/${RUN_NAME}/recipe_lock.json"

if [ "${POS_JOB_MODE}" = "fit_probe" ]; then
  # M1 PUBLISHES the authorization, so its path is derived and attempt-scoped: a retried probe
  # publishes into its own root instead of dying on the storage layer's refusal to rewrite.
  POS_FIT_AUTHORIZATION="${ATTEMPT_ROOT}/fit_authorization.json"
  # F5: the probe banks each finished cell under ${ATTEMPT_ROOT}/cells and adopts verified cells from
  # prior attempts under this root. Defaulting to the attempts root is the issue-#13 layout's own
  # answer; an operator whose wrapper scopes OUTPUT_DIR per attempt widens it explicitly.
  POS_FIT_ADOPTION_ROOT="${POS_FIT_ADOPTION_ROOT:-${RESUME_PARENT}}"
  ENTRYPOINT="src/maxdiffusion/pos_rollout_fit_probe.py"
elif [ -z "${POS_FIT_AUTHORIZATION}" ]; then
  echo "FATAL: POS_FIT_AUTHORIZATION is empty and POS_JOB_MODE=train."
  echo "       A training run executes only an (arm, microbatch, k) cell M1 measured and found to"
  echo "       fit (plan §4-P1). Run POS_JOB_MODE=fit_probe first and pass the path it publishes."
  exit 1
else
  ENTRYPOINT="src/maxdiffusion/train_wan.py"
fi
CONFIG="src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml"

for required in "${ENTRYPOINT}" "${CONFIG}"; do
  if [ ! -f "${required}" ]; then
    echo "FATAL: ${required} is not in this working tree -- the tarball is incomplete and no amount of"
    echo "       TPU time will fix it. Run this launcher from the repository root."
    exit 1
  fi
done

# --- the operator note the blocker round earned (T3b-4 B-2) ----------------------------------
# A checkpoint root whose selection sibling cannot be reconciled with the restored history FAILS
# CLOSED at startup, before a single optimizer step. That is by design: the alternative is shipping a
# checkpoint the run never selected. It reads like a crash in a worker log, so it is named here.
cat <<'RECONCILE'
[note] If this run stops immediately with "the selection artifact cannot be reconciled", that is a
       DESIGNED refusal, not a crash: the resume tree and its _selection sibling disagree about which
       checkpoint is best, and continuing would ship an artifact the stop rule never selected.
       Repair the root (or point OUTPUT_DIR at a clean one) rather than retrying.
[note] "arm X asks to join run Y with a DIFFERENT recipe" is also a DESIGNED refusal: matched-C0 is
       a control only if the two arms share every value outside the arm and its destinations.
RECONCILE

# --- prerequisites, BEFORE the prefetch (review pass 3, T6-2) ---------------------------------
# A job whose inputs are missing or inconsistent must be refused before it downloads 10GB of model.
PYTHONPATH="${PYTHONPATH:-src}" POS_JOB_MODE="${POS_JOB_MODE}" \
  POS_FIT_AUTHORIZATION="${POS_FIT_AUTHORIZATION}" python - <<'PREREQ'
import os
import sys

mode = os.environ["POS_JOB_MODE"]
path = os.environ.get("POS_FIT_AUTHORIZATION", "")
if mode != "train":
    print(f"[prereq] {mode}: no prior-phase artifact is required; this mode publishes one")
    sys.exit(0)

from maxdiffusion.pos_rollout_fit_probe import load_authorization
from maxdiffusion.pos_rollout_support import storage_exists

if not storage_exists(path):
    print(f"FATAL: pos_fit_authorization {path} does not exist -- M1 has not published for this run.")
    sys.exit(1)
try:
    authorization = load_authorization(path)
except Exception as error:  # noqa: BLE001 -- an unreadable authorization authorizes nothing
    print(f"FATAL: {path} is not a usable fit authorization: {type(error).__name__}: {error}")
    sys.exit(1)
measured_sha = str((authorization.get("context") or {}).get("code_sha", ""))
if measured_sha != os.environ.get("COMMIT", ""):
    print(f"FATAL: {path} was measured on {measured_sha}, this job runs {os.environ.get('COMMIT')}.")
    print("       An HBM peak is a measurement OF A PROGRAM; re-run M1 at this SHA.")
    sys.exit(1)
print(f"[prereq] M1 authorization {path}: {authorization['authorized_cells']}")
for entry in authorization.get("excluded_cells") or []:
    print(f"[prereq]   EXCLUDED {entry['arm']} microbatch={entry['microbatch']} k={entry['k_b']}: {entry['reason']}")
PREREQ

# --- issue #4: prefetch the HF snapshot before JAX starts, with retries ------------------
bash bash_scripts/prefetch_hf_snapshot.sh "${MODEL_DIR}"

# --- issue #8 precedent: declared != installed. Preflight before the 5B model is loaded. ------
PYTHONPATH="${PYTHONPATH:-src}" POS_ROLLOUT_ARM="${POS_ROLLOUT_ARM}" \
  POS_RESUME_PARENT="${RESUME_PARENT}" python - <<'PREFLIGHT'
import os
import sys

missing = []
for module in ("tensorflow", "jax", "numpy", "orbax.checkpoint", "huggingface_hub"):
    try:
        __import__(module)
    except Exception as error:  # noqa: BLE001 -- any import failure is fatal here
        missing.append(f"{module}: {type(error).__name__}: {error}")

# exp_06's own dispatch: train_wan.train imports the trainer lazily, so a broken exp_06 module would
# otherwise surface only after the config, the data pipeline and the 5B model are loaded.
try:
    from maxdiffusion.trainers.wan_pos_rollout_trainer import WanPosRolloutTrainer  # noqa: F401
    from maxdiffusion.trainers.wan_pos_rollout_trainer import describe_resume_candidates
    from maxdiffusion.pos_rollout_arms import ARMS
    from maxdiffusion.pos_rollout_dev_instrument import J0_DEV64_SHA256  # noqa: F401
    from maxdiffusion.pos_rollout_fit_probe import LADDER_ARMS
    from maxdiffusion.pos_rollout_loop import run_loop  # noqa: F401
except Exception as error:  # noqa: BLE001
    missing.append(f"maxdiffusion exp_06 modules: {type(error).__name__}: {error}")
else:
    if tuple(ARMS) != ("rollout", "one_step") or tuple(LADDER_ARMS) != tuple(ARMS):
        missing.append(f"declared arms drifted: {ARMS} vs ladder {LADDER_ARMS}")

if missing:
    print("FATAL: preflight failed -- these would have surfaced after the 5B model was on device:")
    for line in missing:
        print(f"  - {line}")
    sys.exit(1)

# What this attempt MIGHT adopt. Review F5c, BLOCKER 2: adoption is decided by the full derived
# context -- code manifest, model revision, device topology, geometry, recipe -- and it is not
# derivable here. This preflight runs AFTER the HF prefetch above, but BEFORE distributed
# initialization and the model load: so `jax.devices()` would report this host's LOCAL chip count,
# which on a v6e-64 job is wrong by a factor of eight, and the config has not been through `pyconfig`
# here, so the recipe fingerprint is not available either. (F5d corrects an earlier version of this
# comment that claimed the prefetch had not run yet -- it has; the topology argument is what stands.)
# So this lists CANDIDATES and
# says plainly that it is not the decision; the decision is `resume_source` inside the real process,
# with the context that process actually derived. Reporting a prediction here would be the same class
# of error as the selector matching a commit label: a confident claim from an identity nobody checked.
candidates = describe_resume_candidates(
    os.environ["POS_RESUME_PARENT"], code_sha=os.environ["COMMIT"], arm=os.environ["POS_ROLLOUT_ARM"]
)
if not candidates:
    print("[preflight] resume: no COMPLETE publication for this arm at this SHA -- this attempt starts at step 0")
else:
    print(f"[preflight] resume: {len(candidates)} candidate publication(s) for this arm at this SHA:")
    for entry in candidates:
        print(f"[preflight]   {entry['attempt']} step {entry['step']} context {entry['context_digest'][:16]}")
    print("[preflight]   ADOPTION IS NOT DECIDED HERE: the running process matches the FULL derived context")
    print("[preflight]   (manifest, model, topology, geometry, recipe), so a candidate above may be refused.")
print("[preflight] ok: imports, exp_06 dispatch, the declared arms and the resume candidates")
PREFLIGHT

# =================================================================================================
# THE COMMON RECIPE. Built once; both arms are emitted from it. Nothing arm-dependent may appear
# here, and nothing that is not a destination may appear in ${ARM_SCOPED}.
# =================================================================================================
COMMON_RECIPE=(
  run_name="${RUN_NAME}"
  pretrained_model_name_or_path="${MODEL_DIR}"
  train_data_dir="${TRAIN_DATA_DIR}"
  output_dir="${OUTPUT_DIR}"
  pos_recipe_lock="${RECIPE_LOCK}"
  pos_fit_authorization="${POS_FIT_AUTHORIZATION}"
  pos_fit_adoption_root="${POS_FIT_ADOPTION_ROOT}"
  pos_fit_excluded_cells="${POS_FIT_EXCLUDED_CELLS}"
  pos_fit_exclusion_reason="${POS_FIT_EXCLUSION_REASON}"
  pos_dev_manifest="${POS_DEV_MANIFEST}"
  pos_dev_manifest_sha256="${POS_DEV_MANIFEST_SHA256}"
  pos_rollout_k="${POS_ROLLOUT_K}"
  pos_logical_batch="${POS_LOGICAL_BATCH}"
  pos_microbatch="${POS_MICROBATCH}"
  pos_rollout_support_salt="${POS_SUPPORT_SALT}"
  max_train_steps="${MAX_TRAIN_STEPS}"
  eval_every="${EVAL_EVERY}"
  checkpoint_every="${CHECKPOINT_EVERY}"
  side_adapter_sampling_steps="${SAMPLING_STEPS}"
  side_adapter_guide_scale="${GUIDE_SCALE}"
  side_adapter_noise_mode="${NOISE_MODE}"
  dropout="${DROPOUT}"
  learning_rate="${LEARNING_RATE}"
  seed="${SEED}"
  per_device_batch_size="${DERIVED_PER_DEVICE_BATCH}"
  wandb_project="${WANDB_PROJECT}"
  hardware="${HARDWARE}"
)
# The permitted differences, and the only ones: the arm, and the destinations derived FROM the arm.
ARM_SCOPED=(
  pos_rollout_arm="${POS_ROLLOUT_ARM}"
  checkpoint_dir="${CHECKPOINT_DIR}"
  base_output_directory="${ARTIFACT_ROOT}"
  pos_resume_parent="${RESUME_PARENT}"
)

echo "RUN_NAME=${RUN_NAME}"
echo "POS_JOB_MODE=${POS_JOB_MODE}   (fit_probe = M1 publishes; train = M2/M3 present)"
echo "POS_ROLLOUT_ARM=${POS_ROLLOUT_ARM}"
echo "ATTEMPT=${ATTEMPT}"
echo "ARM_NAMESPACE=${ARM_NAMESPACE}   (the arm is IN the namespace -- the arms cannot share state)"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}   (this attempt's fresh OUTPUT -- issue #13)"
echo "RESUME_PARENT=${RESUME_PARENT}   (the immutable resume INPUT: latest COMPLETE at this SHA)"
echo "ARTIFACT_ROOT=${ARTIFACT_ROOT}   (attempt-scoped, so a retry cannot collide -- issue #13)"
echo "RECIPE_LOCK=${RECIPE_LOCK}   (run-level: the second arm adopts it or refuses to start)"
echo "MODEL_DIR=${MODEL_DIR}"
echo "TRAIN_DATA_DIR=${TRAIN_DATA_DIR}"
echo "POS_DEV_MANIFEST=${POS_DEV_MANIFEST}"
echo "POS_FIT_AUTHORIZATION=${POS_FIT_AUTHORIZATION}"
echo "POS_FIT_ADOPTION_ROOT=${POS_FIT_ADOPTION_ROOT}   (F5: verified cells under this root are adopted, \
not re-measured; empty = measure everything)"
echo "POS_FIT_EXCLUDED_CELLS=${POS_FIT_EXCLUDED_CELLS}   (F6: DECLARED unreachable -- never built, never \
measured, recorded EXCLUDED in the table; empty = the full ladder runs)"
if [ -n "${POS_FIT_EXCLUDED_CELLS}" ]; then
  echo "[note] This run publishes a table that authorizes LESS than the full ladder, by declaration:"
  echo "[note]   ${POS_FIT_EXCLUSION_REASON}"
  echo "[note] That is a plan deviation. It must carry Yixun's recorded decision, not just this flag."
fi
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS} EVAL_EVERY=${EVAL_EVERY} CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "POS_LOGICAL_BATCH=${POS_LOGICAL_BATCH} POS_MICROBATCH=${POS_MICROBATCH} POS_ROLLOUT_K=${POS_ROLLOUT_K}"
echo "SAMPLING_STEPS=${SAMPLING_STEPS} GUIDE_SCALE=${GUIDE_SCALE} NOISE_MODE=${NOISE_MODE} DROPOUT=${DROPOUT}"
echo "LEARNING_RATE=${LEARNING_RATE} SEED=${SEED}"
echo "POS_DEVICE_COUNT=${POS_DEVICE_COUNT}   PER_DEVICE_BATCH=${DERIVED_PER_DEVICE_BATCH}   (global batch \
${POS_LOGICAL_BATCH} = pos_logical_batch; the trainer refuses the job if this topology is not the real one)"
echo "COMMIT=${COMMIT}"
git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"

python "${ENTRYPOINT}" "${CONFIG}" "${COMMON_RECIPE[@]}" "${ARM_SCOPED[@]}"
