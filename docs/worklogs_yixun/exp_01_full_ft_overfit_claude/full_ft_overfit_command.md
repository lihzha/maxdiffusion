# exp_01 `full_ft_overfit` — command log

Every launch appended AT LAUNCH TIME (SOP artifact 7). Failed/superseded runs stay, marked. `_worklog.md` records why; this file records how to reproduce.

## 1. Smoke run (rung 5) — 2026-07-18T20:39:58Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `07eb5b2` on `claude-exp_01_full_ft_overfit-20260715` (pushed; worker verifies via COMMIT env)
- **Approval:** Yixun, "Approve smoke", 2026-07-18 (announcement 02)
- **Command (from the exp worktree):**
```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit
WAN_EXPERIMENT=full_ft SMOKE=1 TPU_CHIPS=8 NAME=wan-full-ft-smoke-yixun \
  bash bash_scripts/launch_wan_train.sh
```
- **Effective config:** v6e-8 us-east1-d; smoke run-name `smoke-full-ft-<utc-ts>`; MAX_TRAIN_STEPS=1; CHECKPOINT_EVERY=0; SAVE_FINAL_CHECKPOINT=False; EVAL_EVERY=0; per-device batch 8 (GBS 64 on 8 chips); fresh noise; EVAL_DATA_DIR=train split (full_ft override block); yml `base_wan_5b_full_ft.yml` via `train_wan_full_ft.sh`.
- **Job id:** `20260718-204019-6aad21e8-wan-full-ft-smoke-yixun` (queue: `tpu status 20260718-204019-6aad21e8-wan-full-ft-smoke-yixun`; authoritative state: `gs://v6_east1d/tpu-job-queue/jobs/20260718-204019-6aad21e8-wan-full-ft-smoke-yixun/status.json`)
