#!/bin/bash
# exp_04 J1-5 — the CLEAN-GATE RERUN at the ADOPTED recipe (issue #15 fixed; Yixun-approved).
# Decides the plan-compliant target selection (the J=10 STOP was ruled indeterminate).
# Capacity dev64 + trainfit16, NULL_ADEQUACY_URI wired, attempt-scoped roots. ~1.4 h.
set -euo pipefail
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter
TIP=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain -- src bash_scripts)" ]; then echo "FATAL: dirty EXECUTABLE tree (src/, bash_scripts/)"; exit 9; fi  # docs may be mid-edit by the report reviser
J1ROOT="gs://v6_east1d/datasets/droid_wan_null_adapter/j1r2"

tpu create v6 -n 8 --worker0-only --name "exp04-j1-5-cleangate-yixun" \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env COMMIT="$TIP" \
  --env RUN_NAME="exp04-j1-5-$(date -u +%Y%m%d)" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env J1ROOT="$J1ROOT" \
  -- bash -c '
set -euo pipefail
ATT="att-$(date -u +%m%d-%H%M%S)"
echo "=== J1-5 CLEAN-GATE RERUN (adopted recipe; attempt root suffix $ATT) ==="
ADOPT=$(gsutil ls "$J1ROOT/adequacy/**.json" | grep -i -m1 "adequacy\|adoption" || true)
if [ -z "$ADOPT" ]; then echo "FATAL: no published adequacy artifact"; exit 9; fi
echo "ADOPTING: $ADOPT"
echo "=== PHASE B: CAPACITY dev64 AT THE ADOPTED RECIPE ==="
NULL_MODE=capacity NULL_ADEQUACY_URI="$ADOPT" \
  NULL_ARTIFACT_DIR="$J1ROOT/cleangate_$ATT" NULL_STAGING_DIR="$J1ROOT/cleangate_$ATT/_staging" \
  bash bash_scripts/run_wan_null_inversion.sh
echo "=== PHASE C: CAPACITY trainfit16 AT THE ADOPTED RECIPE ==="
NULL_MODE=capacity NULL_COHORT=trainfit16 NULL_ADEQUACY_URI="$ADOPT" \
  NULL_ARTIFACT_DIR="$J1ROOT/cleangate_trainfit_$ATT" NULL_STAGING_DIR="$J1ROOT/cleangate_trainfit_$ATT/_staging" \
  bash bash_scripts/run_wan_null_inversion.sh
echo "=== J1-5 COMPLETE (authoritative roots: cleangate_$ATT + cleangate_trainfit_$ATT) ==="
'
