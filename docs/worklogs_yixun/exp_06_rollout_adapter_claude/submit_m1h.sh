#!/bin/bash
# exp_06 M1-10 — fit probe under the F10 authorization amendment (plan v2.9, Yixun Option A):
# authorization bound = compiled memory analysis (<= 90% capacity, exact integer arithmetic);
# runtime watermark recorded as cross-check (watermark > analysis => inconsistent, refused).
# FULL RE-MEASURE (~2.5-3.5h): the F10/F10b/F10c edits moved the deployed manifest, so the
# M1-9 bank (att-0816-*) is correctly non-adoptable; attempts at THIS SHA adopt each other.
# Expected from M1-9's numbers re-derived under the new rule: 10/12 authorized (both arms
# mb=8/16; rollout mb=16/32/64), rollout mb=8 refused on true headroom (96.6%), 4 excluded (#18).
# v6e-8; POS_DEVICE_COUNT=8; attempt-scoped roots per issue #13.
set -euo pipefail
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter
TIP=$(git rev-parse HEAD)
RULED="f872a42c82c720408a658c2cab361948b72e49ed"  # F10f — final APPROVE of the F10 authorization-amendment series
if ! git diff --quiet "$RULED" HEAD -- src bash_scripts; then echo "FATAL: the EXECUTABLE tree (src/, bash_scripts/) differs from the READY-ruled state $RULED — re-review before launching"; exit 9; fi
if [ -n "$(git status --porcelain)" ]; then echo "FATAL: dirty tree — never launch it"; exit 9; fi
M1ROOT="gs://v6_east1d/datasets/droid_wan_pos_rollout/m1"

tpu create v6 -n 8 --worker0-only --name "exp06-m1h-fitprobe-yixun" \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env COMMIT="$TIP" \
  --env RUN_NAME="exp06-m1-fitprobe" \
  --env POS_FIT_ADOPTION_ROOT="$M1ROOT" \
  --env POS_FIT_EXCLUDED_CELLS="one_step:32:2,one_step:32:4,one_step:64:2,one_step:64:4" \
  --env POS_FIT_EXCLUSION_REASON="deterministic bad_smem_address chip fault at one_step microbatch>=32 (issue #18), reproduced 2/2 on distinct VMs" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env M1ROOT="$M1ROOT" \
  -- bash -c '
set -euo pipefail
ATT="att-$(date -u +%m%d-%H%M%S)"
echo "=== exp_06 M1-10: FIT PROBE under F10 authorization amendment (attempt root $ATT) ==="
POS_JOB_MODE=fit_probe POS_DEVICE_COUNT=8 \
  OUTPUT_DIR="$M1ROOT/$ATT" \
  bash bash_scripts/train_wan_pos_rollout.sh
echo "=== M1-10 COMPLETE (authoritative root: $M1ROOT/$ATT) ==="
'
