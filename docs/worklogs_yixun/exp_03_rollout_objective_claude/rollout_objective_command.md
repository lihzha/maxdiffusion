# exp_03 commands

Every launch recorded here at launch time per the SOP. (Bookkeeping note: this file was briefly committed
EMPTY at `0c08b70` — a 2-minute shell timeout killed the heredoc that was writing it, after the package text
had already been reviewed into the plan; the content below was restored minutes after the launches, from the
same text. The gap is disclosed rather than backdated.)

## Jobs 1–4 — v6e-8 S1 SMOKE: control / corrective_ss / rollout_loss / combined — launched 2026-08-03T02:45Z

**Approved by Yixun (Query 3, conditional grant: "approve S1 smoke when the package is ready"; conditions
met — round 3 CLOSED with APPROVE, package below).** Rounds 1–3 all CLOSED (extraction APPROVE; trainer
APPROVE; losses APPROVE). Suite 1,399 + 2. `COMMIT=0c08b70` (tip at submission).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective
for obj in control corrective_ss rollout_loss combined; do
  short=$(echo $obj | tr -d '_' | cut -c1-8)
  tpu create v6 -n 8 --worker0-only --name "exp03-s1-${short}-yixun" \
    --code-dir . \
    --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
    --env RUN_NAME="exp03-s1-${short}-20260803" \
    --env EXP03_OBJECTIVE="$obj" \
    --env EXP03_RAMP_ORIGIN=0 --env EXP03_P_SS_RAMP_STEPS=10 \
    --env MAX_TRAIN_STEPS=30 --env SAVE_FINAL_CHECKPOINT=False \
    --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train100" \
    --env EXPECTED_WINDOWS=1629 --env NUM_TEXT_SLOTS=100 \
    --env COMMIT="<tip>" \
    --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    -- bash bash_scripts/train_wan_exp03.sh
done
```

**What:** 4 × v6e-8, from-init (Tier-2-style — no checkpoint seed), 30 steps each, tiny footprint,
`SAVE_FINAL_CHECKPOINT=False`. `EXP03_RAMP_ORIGIN=0` and `EXP03_P_SS_RAMP_STEPS=10` for the smoke ONLY, so
A/C's self-generation path is genuinely exercised within 30 steps (p_ss reaches 0.5 by step 10). Data =
train100 (unchanged pins).

**Gates (predeclared):**
1. All losses finite at every step, all four arms; grad norms same order as control.
2. Hook parity on hardware: control's first-step loss within fp tolerance of the exp_02 trainer's at the
   same seed (the smoke-scale proxy for the ctrl0 AND-gate).
3. **STOP budgets on measured step-time ratio vs the control smoke (same hardware):** A ≤ 1.6×, B ≤ 2.5×,
   C ≤ 3.2×. Exceeding a budget is a STOP for that arm (report to Yixun), not a silent accept.
   Reference (hardware-independent jaxpr census): fwd-call/eqn ratios 1.00 / 1.79 / 1.38 / 3.24.
4. B/C compile cleanly with scan+remat on-device (the real S1.6 mesh-fit at GBS 256 on v6e-64 is a
   separate gated launch).

- **Job ids (submitted 2026-08-03T02:45–02:46Z):** control → `20260803-024504-7175ecdb-exp03-s1-control-yixun`;
  corrective_ss → `20260803-024531-b4f93a1a-exp03-s1-correcti-yixun`; rollout_loss →
  `20260803-024558-3534905b-exp03-s1-rolloutl-yixun`; combined → `20260803-024622-0206cf9b-exp03-s1-combined-yixun`.

## S1 outcome (Jobs 1–4) — 2026-08-03T~04:30Z

All four SUCCEEDED as queue jobs (after one v6e-8 maintenance sweep + suspensions; attempts 1–2 each).
Gates evaluated from step logs (steady-state steps/s = mean over steps 10–29):

| arm | finiteness | steps/s | ratio | budget | gate |
| --- | --- | --- | --- | --- | --- |
| control | all finite | 1.786 | 1.00× | — | PASS |
| corrective_ss | all finite | 1.219 | 1.47× | ≤1.6× | **PASS** |
| rollout_loss | all finite | 0.698 | 2.56× | ≤2.5× | **STOP** (marginal, +2.4%) |
| combined | **NaN from step 8** | 0.422 | 4.23× | ≤3.2× | **STOP** (×2) |

- Gate 2 (hook parity on hardware) is NOT evaluable against exp_02's history at smoke scale (different
  batch/hardware); it is carried by the suite's exact JIT-parity certificate now and by ctrl0's AND-gate at
  S2b. Recorded as deferred-by-design, not passed.
- B's overrun is small and plausibly small-batch/remat overhead; decision deferred to S1.6's at-scale
  measurement (no code change).
- C's NaN at LR≈1e-6 is a numerical edge in the loss computation for a specific draw — deterministically
  reproducible from (seed, step 8, purposes). Diagnosis round dispatched; C re-smoke will need a fresh
  launch approval after the fix + review.

## Jobs 5–7 — v6e-8 RE-SMOKE COHORT (control timing / A timing / C replay) — launched 2026-08-03

**Under Yixun's Query-4 grant** (C re-smoke conditional on the S1-fix review passing — passed at `fdadb5f`,
reviewer GO with the cohort spec confirmed). One contemporaneous cohort, identical seed/data/ramp/compiler
to S1: 30 steps, `EXP03_RAMP_ORIGIN=0`, `EXP03_P_SS_RAMP_STEPS=10`, `LOG_PERIOD=1` (full 16-field line every
step), strict STOP gates control-relative (A ≤ 1.6× — note A now always runs 2 advances, was mean 1.5, so
this re-measures its real cost; C ≤ 3.2×). C additionally arms `EXP03_SNAPSHOT_BEFORE_STEP=7` (single host —
gate open): pre-failure params/opt/rng/batch land in the run dir before global_step 7 executes. If C goes
non-finite: the forced NON-FINITE line names the term, every host raises, and the frozen-state A/B/C
discriminator (`exp03_frozen_replay`) runs from the snapshot as a follow-up job.
- **Job ids (submitted 2026-08-03T16:45–16:46Z):** control → `20260803-164526-06c4fa27-exp03-rs-control2-yixun`; A → `20260803-164552-452e8a59-exp03-rs-corrss2-yixun`; C replay (snapshot armed) → `20260803-164618-cf2ca830-exp03-rs-combined2-yixun`. COMMIT=86c7408.

### Job 7 correction (2026-08-03T~17:15Z)

Job 7 (combined2) FAILED at launch-plumbing, not in training: an unquoted $extra in the zsh launch loop
passed `--env EXP03_SNAPSHOT_BEFORE_STEP=7` as ONE malformed argument, scrambling the CLI's arg order so
the worker ran `bash -- …` (invalid option, exit 2). No trainer code executed; no NaN evidence either way.
Relaunched as **Job 7b** `20260803-171003-8ce1d02c-exp03-rs-combined3-yixun` with the env inline. (Second zsh word-splitting incident this session —
noted: always inline or array-expand extra args.)

