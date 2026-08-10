#!/bin/bash
# exp_06 M1-2 — fit probe relaunch after the F3 captured-constants arc (M1-1: 12 dead attempts, CANCELED).
# v6e-8; POS_DEVICE_COUNT=8 (derives per-device batch 32 at GBS 256); ~1 h projected.
# Attempt-scoped roots per issue #13; the probe authorizes ONLY its measured (arm, microbatch, k) cells.
set -euo pipefail
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter
TIP=$(git rev-parse HEAD)
RULED="7b3f10cc61e2c41865bd3813bac5cff81832483c"  # F3-arc ceremony tip (captured-constants fix, 3 review passes)
if ! git diff --quiet "$RULED" HEAD -- src bash_scripts; then echo "FATAL: the EXECUTABLE tree (src/, bash_scripts/) differs from the READY-ruled state $RULED — re-review before launching"; exit 9; fi
if [ -n "$(git status --porcelain)" ]; then echo "FATAL: dirty tree — never launch it"; exit 9; fi
M1ROOT="gs://v6_east1d/datasets/droid_wan_pos_rollout/m1"

tpu create v6 -n 8 --worker0-only --name "exp06-m1b-fitprobe-yixun" \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env COMMIT="$TIP" \
  --env RUN_NAME="exp06-m1-$(date -u +%Y%m%d)" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env M1ROOT="$M1ROOT" \
  -- bash -c '
set -euo pipefail
ATT="att-$(date -u +%m%d-%H%M%S)"
echo "=== exp_06 M1: FIT PROBE (16-cell ladder x 2 arms, attempt root $ATT) ==="
POS_JOB_MODE=fit_probe POS_DEVICE_COUNT=8 \
  OUTPUT_DIR="$M1ROOT/$ATT" \
  bash bash_scripts/train_wan_pos_rollout.sh
echo "=== M1 COMPLETE (authoritative root: $M1ROOT/$ATT) ==="
'
