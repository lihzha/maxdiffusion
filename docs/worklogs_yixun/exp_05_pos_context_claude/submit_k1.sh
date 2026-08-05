#!/bin/bash
# exp_05 K1 submission — DRAFT pending S10a strengthen commit (verify TIP first).
# FOUR-phase runbook: smoke capacity -> adequacy probe -> capacity dev64 -> capacity trainfit16
# (cohorts are one-per-invocation; both capacity phases adopt the one DEV adequacy artifact —
# the S10a strengthen fixed adoption to validate DEV provenance independent of the running cohort).
# Positive slot wires {capacity, adequacy_probe} only (pos_execute) — no verify phase in K1 by design.
set -euo pipefail
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context
TIP=$(git rev-parse HEAD)
K1ROOT="gs://v6_east1d/datasets/droid_wan_pos_context/k1"

tpu create v6 -n 8 --worker0-only --name "exp05-k1-pos-yixun" \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env COMMIT="$TIP" \
  --env RUN_NAME="exp05-k1-$(date -u +%Y%m%d)" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  --env K1ROOT="$K1ROOT" \
  -- bash -c '
set -euo pipefail
echo "=== K1 PHASE 1: SMOKE (2 examples, production settings) ==="
POS_MODE=capacity POS_SMOKE_EXAMPLES=2 \
  POS_ARTIFACT_DIR="$K1ROOT/smoke" POS_STAGING_DIR="$K1ROOT/smoke/_staging" \
  bash bash_scripts/run_wan_pos_inversion.sh
echo "=== K1 PHASE 2: ADEQUACY PROBE (first-8 DEV, approved grid) ==="
POS_MODE=adequacy_probe \
  POS_ARTIFACT_DIR="$K1ROOT/adequacy" POS_STAGING_DIR="$K1ROOT/adequacy/_staging" \
  bash bash_scripts/run_wan_pos_inversion.sh
echo "=== K1 PHASE 3: FULL CAPACITY on DEV-64 (B-arms + L_pos ablation, DEV adequacy adopted) ==="
ADOPT=$(gsutil ls "$K1ROOT/adequacy/**.json" | grep -i -m1 "adequacy\|adoption" || true)
POS_MODE=capacity POS_COHORT=dev64 \
  POS_ADEQUACY_URI="$ADOPT" \
  POS_ARTIFACT_DIR="$K1ROOT/capacity" POS_STAGING_DIR="$K1ROOT/capacity/_staging" \
  bash bash_scripts/run_wan_pos_inversion.sh
echo "=== K1 PHASE 4: FULL CAPACITY on TRAINFIT-16 (same DEV adequacy adopted) ==="
POS_MODE=capacity POS_COHORT=trainfit16 \
  POS_ADEQUACY_URI="$ADOPT" \
  POS_ARTIFACT_DIR="$K1ROOT/capacity_trainfit" POS_STAGING_DIR="$K1ROOT/capacity_trainfit/_staging" \
  bash bash_scripts/run_wan_pos_inversion.sh
echo "=== K1 COMPLETE ==="
'
