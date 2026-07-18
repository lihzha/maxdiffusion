# exp_01 `full_ft_overfit` — command log

Every launch appended AT LAUNCH TIME (SOP artifact 7). Failed/superseded runs stay, marked. `_worklog.md` records why; this file records how to reproduce.

## 1. Smoke run (rung 5) — 2026-07-18T20:39:58Z

- **Status:** FAILED (attempt 1, APPLICATION_ERROR) — v6e-8 CompileTimeHbmOom, over by 44.29MB at per-device batch 8. Kept for the record; retry (config-changed: per-device 1) requires fresh approval — entry 2 below when approved.
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

## 2. Smoke attempt 2 (rung 5, post-fix) — 2026-07-18T21:22:09Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `0405a30` (cycle-6: env-overridable batch trio; reviewed APPROVE/no-findings)
- **Approval:** Yixun, "Approve smoke 2 + fit probe conditional on pass", 2026-07-18
- **Command (exp worktree):**
```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit
WAN_EXPERIMENT=full_ft SMOKE=1 PER_DEVICE_BATCH_SIZE=1 \
  GLOBAL_BATCH_SIZE_TO_TRAIN_ON=8 GLOBAL_BATCH_SIZE_TO_LOAD=8 \
  TPU_CHIPS=8 NAME=wan-full-ft-smoke2-yixun \
  bash bash_scripts/launch_wan_train.sh
```
- **Effective config:** v6e-8; 1 step; checkpoints/eval/final-save off; per-device 1 (GBS 8); fresh noise; train-split eval dir; yml `base_wan_5b_full_ft.yml`.
- **Job id:** `20260718-212209-ff679a2a-wan-full-ft-smoke2-yixun` (state: `gs://v6_east1d/tpu-job-queue/jobs/20260718-212209-ff679a2a-wan-full-ft-smoke2-yixun/status.json`)

- **Outcome (appended):** SUCCEEDED attempt 3 (2 infra preemptions, queue-retried). Log-verified PASS; loss-value display gap noted in worklog. wandb: `wewfe1kx`.

## 3. Fit probe (rung 6) — 2026-07-18T22:07:20Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `2da034d` tip (code identical to smoke's `0405a30`; only docs advanced)
- **Approval:** Yixun Query 4 — "fit probe conditional on pass"; smoke-2 PASS verified above.
- **Command (exp worktree):**
```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit
WAN_EXPERIMENT=full_ft SMOKE=1 TPU_CHIPS=64 NAME=wan-full-ft-fitprobe-yixun \
  bash bash_scripts/launch_wan_train.sh
```
- **Effective config:** v6e-64; 1 step; checkpoints/eval/final-save off; per-device 8 (GBS 512 — launcher defaults, the exact full-run memory shape); fresh noise; train-split eval dir.
- **Job id:** `20260718-220720-f19db2ab-wan-full-ft-fitprobe-yixun`
