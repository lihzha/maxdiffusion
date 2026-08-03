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

### Re-smoke cohort outcome so far (2026-08-03T~19:40Z)

- **Timing pair complete:** control2 1.801 steps/s; corrss2 1.533 → **A ratio 1.17× (PASS, was 1.47×)** —
  the unroll compiles better than the dynamic loop it replaced. Both fully finite.
- **C replay (combined3): the NaN did NOT recur.** Attempt 1 ran 19 finite steps before preemption —
  past the original failing step AND through the hardest draw class (global_step 15: s_a=0, σ_hi=1.0,
  k_a=2, self-generated) with every *_finite flag green and all 16 diagnostic fields flowing. C cost
  3.96× vs 3.2× budget (better than 4.23×, still over — S1.6 item).
- **combined3 terminal:** attempt 4 hit borderline HBM OOM at program load (26.47G vs 23.70G free) — the
  same binary ran on attempt 1, so allocator-layout flakiness at the memory edge; exit 1 stopped queue
  retries. **C now carries TWO S1.6 flags: step-time (3.96×) and HBM headroom.** One resubmit under the
  infra-flake reading: **Job 7c** `20260803-193617-1214ad57-exp03-rs-combined4-yixun` for the formal 30/30; the substantive verdict stands either way.

### S1 CLOSED (2026-08-03T~20:45Z)

**combined4 attempt 1 completed the formal 30/30 — every step finite** (zero `finite=0` occurrences; final
line step 30/30, loss 0.974, all per-term flags green). The post-completion VM preemption made the queue
requeue needlessly; the redundant attempt was cancelled — the evidence is complete.

**S1 final gate table:**
| arm | finiteness | steady ratio | budget | status |
| --- | --- | --- | --- | --- |
| control | 30/30 finite | 1.00× | — | PASS |
| corrective_ss (A) | 30/30 finite | **1.17×** (improved from 1.47× by the unroll) | ≤1.6× | **PASS** |
| rollout_loss (B) | 30/30 finite | 2.56× | ≤2.5× | finite; budget → S1.6 |
| combined (C) | **30/30 finite — NaN resolved** | 3.96× | ≤3.2× | finite; budget + HBM headroom (26.5G vs 23.7G) → S1.6 |

The S1 NaN's proximate cause stands as the compiler-shape hypothesis (traced-bound loop), now supported by:
corrected-draw replay through the self-generated branch, the hazard-class pass (σ_hi=1.0, k=2,
self-generated), and two independent full-30 finite runs. Next: **S1.5** under the Query-4 grant.


## Job 8 — v6e-8 S1.5 DUAL-STATE DISCRIMINATOR PROBE — launched 2026-08-03

**Under Yixun's Query-4 grant** (S1.5 conditional on the re-smoke being clean — S1 CLOSED — and the probe
passing review — APPROVE at `3ffb8f9`, reviewer GO with spec). One v6e-8 job: no-update diagnostics at BOTH
states (exp_02 step-10,000 checkpoint, exact-step-pinned restore; pretrained init via the production
empty-restore path), K=8 batches × M=4 salted support draws; per-objective losses/grads/cosines; A's label
isolation; conditional fixed-support parity; forced-p_ss=1 diagnostics; support-variance decomposition
(law of total variance, finite-M honesty note); per-state sigma traces; branch outcomes. Artifacts: two
immutable state JSONs + two trace JSONs under `validation_probe_sampling/`.
- **Job id:** `20260803-220241-064d6ea5-exp03-s15-probe-yixun` (COMMIT=e933b48).

### Job 8 outcome + Job 8b (2026-08-04T~00:20Z)

Job 8 FAILED at startup on hardware: `gen.load_next_batch` never existed (the e2e test had stubbed the
batch pull — the inspect-vs-execute class, 4th instance). Fix `e2249e1` (APPROVE): probe uses the trainer's
own `train_utils.load_next_batch` with matching iterator seeding; a new AST attribute-resolution guard
closes the wrong-attribute class for probe + both trainers. Relaunch **Job 8b**: `20260803-230447-b14dcde5-exp03-s15-probe2-yixun` (COMMIT=7dcc1f82b78eec9a6bda1e404cc056c36286c3ed).

