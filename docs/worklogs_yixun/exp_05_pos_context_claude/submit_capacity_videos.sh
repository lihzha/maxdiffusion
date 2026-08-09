#!/bin/bash
# Post-STOP capacity-videos job (2026-08-09, approved scope: ~1 v6e-8 hour).
# ONE phase: render the never-decoded capacity arms of BOTH experiments as GT-vs-pred mp4s.
#   exp_04: A2 (decode-only from stored expected_final_latent) + A1-probe k=0 (true replay)
#   exp_05: B1 + B2 (decode-only) + B1-probe k=0 (true replay)
# Read-only against the published capacity roots; writes mp4s + one videos_report.json per
# attempt-scoped out root. ATTEMPT is pinned HERE so a queue auto-retry re-renders the SAME
# roots (all writes are overwrite-idempotent) instead of forking new ones.
set -euo pipefail
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context
TIP=$(git rev-parse HEAD)  # expected: 6f3146acfb0eab585e5599fca23cf1affc3513c8

tpu create v6 -n 8 --worker0-only --name "capvideos-yixun" \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env COMMIT="$TIP" \
  --env RUN_NAME="cap-videos-$(date -u +%Y%m%d)" \
  --env ATTEMPT="$(date -u +%m%d-%H%M%S)" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash -c '
set -euo pipefail
echo "=== CAPACITY VIDEOS: exp_04 A2 + A1-probe(k0); exp_05 B1 + B2 + B1-probe(k0) ==="
bash bash_scripts/run_wan_capacity_videos.sh
echo "=== CAPACITY VIDEOS COMPLETE ==="
'
