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

- **Outcome (appended to entry 3):** FAILED — INFRASTRUCTURE (worker-10 apt/unattended-upgrades lock hang blocked setup; 15 healthy hosts died at JAX-init DEADLINE_EXCEEDED). Queue barrier under-counted (8/16). No code change; resubmitted as entry 4.

## 4. Fit probe resubmit (rung 6, unchanged) — 2026-07-18T22:32:23Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `fd65eb2` tip (code identical to entry 3)
- **Approval:** covered by Query 4/5 (same approved job; infra resubmit per announcement 02)
- **Command:** identical to entry 3.
- **Job id:** `20260718-223223-ef0a159d-wan-full-ft-fitprobe-yixun`

- **Outcome (appended to entry 4):** FAILED — same infra class: workers 9+11 apt-locked (unattended-upgrades), coordination deadline killed the rest. Root cause isolated to setup.sh's unbounded apt lock wait; hardening in mini-cycle 7.

## 5. Fit probe resubmit #3 (rung 6, unchanged) — 2026-07-18T22:57:54Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `6185cd8` tip (code identical to entries 3–4)
- **Approval:** pre-authorized infra resubmit (announcement 02; Query 4/5 conditionals still armed)
- **Command:** identical to entry 3.
- **Job id:** `20260718-225754-87cbd078-wan-full-ft-fitprobe-yixun`

- **Outcome (appended to entry 5):** FAILED — same coordination-deadline class (worker-12 first abort; stuck host in unsampled ranks). 3/3 pre-fix failures → launch freeze until the cycle-7 post-fix commit; probe #4 will be entry 6.

## 6. Fit probe #4 (rung 6, POST-FIX) — 2026-07-18T23:38:00Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `0ffd950` (cycle-7 setup hardening; reviewed APPROVE)
- **Approval:** Query 6 (post-fix launches blessed)
- **Command (exp worktree):** identical to entry 3 (`WAN_EXPERIMENT=full_ft SMOKE=1 TPU_CHIPS=64 NAME=wan-full-ft-fitprobe-yixun bash bash_scripts/launch_wan_train.sh`)
- **Job id:** `20260718-233800-5d773c8b-wan-full-ft-fitprobe-yixun`

- **Outcome (appended to entry 6):** FAILED — but the setup hardening PASSED (no stalls, 16/16 hosts compiled). Real finding: CompileTimeHbmOom 31.28/31.25G (+36.92M) at per-device 8 on v6e-64 ⇒ full-FT per-device 8 does not fit any topology (FSDP collective buffers). Remedy proposal: per-device 4 (entry 7 pending approval).

## 7. Fit probe #5 (rung 6, amended batch) — 2026-07-19T16:27:02Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `c01722c` tip (code = post-hardening `0ffd950`)
- **Approval:** Yixun Query 7 ("Approve amended run")
- **Command (exp worktree):**
```bash
WAN_EXPERIMENT=full_ft SMOKE=1 TPU_CHIPS=64 PER_DEVICE_BATCH_SIZE=4 \
  GLOBAL_BATCH_SIZE_TO_TRAIN_ON=256 GLOBAL_BATCH_SIZE_TO_LOAD=256 \
  NAME=wan-full-ft-fitprobe-yixun bash bash_scripts/launch_wan_train.sh
```
- **Job id:** `20260719-162702-4d29b151-wan-full-ft-fitprobe-yixun`

## 8. PRIMARY FULL RUN (rung 7, amended recipe) — 2026-07-19T16:52:22Z

- **Status:** LAUNCHED (job id below)
- **Commit:** `031228e` (cycle-8 amended recipe; setup hardening included)
- **Approval:** Queries 5 ("full run conditional on fit probe pass") + 6 (post-fix commit) + 7 (amended GBS 256 × 20k); probe #5 PASS log-verified above.
- **Command (exp worktree):**
```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit
WAN_EXPERIMENT=full_ft TPU_CHIPS=64 NAME=wan-full-ft-yixun \
  bash bash_scripts/launch_wan_train.sh
```
- **Effective config:** v6e-64; per-device 4 ⇒ GBS 256; 20000 steps (≈3.55 passes); LR 1e-5; fresh noise; guide 1.0; ckpt every 2500 keep-period 2500; eval-in-training every 1000 on TRAIN shards; wandb `maxdiffusion-wan-full-ft`; run name `wan-full-ft-v6e64-full-gbs256-fresh-<ts>`; yml `base_wan_5b_full_ft.yml`.
- **Job id:** `20260719-165222-62b5c10e-wan-full-ft-yixun`

## 9. Cohort validation ×5 (plan §2.3) — 2026-07-20T14:21:48Z

- **Status:** LAUNCHED (5 jobs, ids below)
- **Commit:** `c562856` tip (code = `031228e`)
- **Approval:** Yixun, "Approve cohort validation", 2026-07-20
- **Command template (exp worktree; documented queue pattern for validation jobs):**
```bash
export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
for STEP in 0 5000 10000 15000 20000; do
tpu create v6 -n 8 --name "wan-full-ft-cohort-s${STEP}-yixun" \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu && bash bash_scripts/prefetch_hf_snapshot.sh Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
  --env RUN_NAME="wan-full-ft-v6e64-full-gbs256-fresh-20260719-165222" \
  --env CHECKPOINT_STEP="${STEP}" \
  --env NUM_EVAL_VIDEOS="16" \
  --env VALIDATION_ORDINALS="0,96037,192074,288111,384147,480184,576221,672258,768295,864332,960369,1056406,1152442,1248479,1344516,1440553" \
  --env VALIDATION_SEED="0" \
  --env COMMIT="c562856" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/validate_wan_full_ft.sh
done
```
- **Job ids:**
  - step 0: `20260720-142149-03b03447-wan-full-ft-cohort-s0-yixun`
  - step 5000: `20260720-142155-14dfc21a-wan-full-ft-cohort-s5000-yixun`
  - step 10000: `20260720-142201-8da2f201-wan-full-ft-cohort-s10000-yixun`
  - step 15000: `20260720-142207-d91d6159-wan-full-ft-cohort-s15000-yixun`
  - step 20000: `20260720-142212-0cfc8455-wan-full-ft-cohort-s20000-yixun`
