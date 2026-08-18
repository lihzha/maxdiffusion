#!/bin/bash
# exp_06 M2 — learnability probe PAIR (plan §4-P2, Yixun-approved 2026-08-17):
#   job 1: R-B  (POS_ROLLOUT_ARM=rollout,  mb=16, k=2)  — authorized cell, 15.63 s/step
#   job 2: C0   (POS_ROLLOUT_ARM=one_step, mb=16, k=2)  — authorized cell,  3.11 s/step
# Both: 2,000 steps, GBS 256, seed 0, eval+checkpoint every 1,000, v6e-8.
# RECIPE NOTE (deviation, Yixun-acknowledged before firing): the plan's "32 examples" was never
# wired as a knob, and pos_logical_batch/train_data_dir are BOUND into M1-10's authorization
# (recipe fingerprint) — so M2 runs the AUTHORIZED recipe: GBS 256 on the full train stream.
# PAIR RULE (plan v2.8): if EITHER arm is preempted mid-run, the pair is NON-QUOTABLE —
# do NOT resume one alone; BOTH arms restart from fresh attempts.
set -euo pipefail
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter
TIP=$(git rev-parse HEAD)
RULED="f872a42c82c720408a658c2cab361948b72e49ed"  # F10f — the APPROVE tip M1-10 measured under
if ! git diff --quiet "$RULED" HEAD -- src bash_scripts; then echo "FATAL: executable tree differs from $RULED — re-review first"; exit 9; fi
if [ -n "$(git status --porcelain)" ]; then echo "FATAL: dirty tree — never launch it"; exit 9; fi
AUTH="gs://v6_east1d/datasets/droid_wan_pos_rollout/m1/att-0817-015756/exp06-m1-fitprobe/fit_probe/attempts/att-20260817T015756Z/fit_authorization.json"

for ARM in rollout one_step; do
  [ "$ARM" = rollout ] && TAG=rb || TAG=c0
  tpu create v6 -n 8 --worker0-only --name "exp06-m2-${TAG}-yixun" \
    --code-dir . \
    --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
    --env COMMIT="$TIP" \
    --env RUN_NAME="exp06-m2-${TAG}-20260817" \
    --env POS_JOB_MODE=train \
    --env POS_ROLLOUT_ARM="$ARM" \
    --env POS_MICROBATCH=16 \
    --env POS_ROLLOUT_K=2 \
    --env POS_DEVICE_COUNT=8 \
    --env MAX_TRAIN_STEPS=2000 \
    --env EVAL_EVERY=1000 \
    --env CHECKPOINT_EVERY=1000 \
    --env POS_FIT_AUTHORIZATION="$AUTH" \
    --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    -- bash bash_scripts/train_wan_pos_rollout.sh
done
