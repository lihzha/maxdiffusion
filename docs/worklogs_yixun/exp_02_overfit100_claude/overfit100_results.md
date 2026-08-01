# exp_02 `overfit100` — Results (appended as runs finish)

## S2 gate run (train10: 10 episodes / 167 windows; v6e-8; 2,500 steps, GBS 32, LR 1e-5)

**Training** (job `20260730-173902`, 36 min): loss **0.533 → 0.061**, still decreasing at 2,500 steps (0.170 @250 → 0.154 @500 → 0.138 @1000 → 0.061 @2500); grad norm 4.3 → 0.15; 1.82 steps/s. Reference: exp_01 (full DROID, no text) plateaued at 0.176 — this regime is 3× below that floor and unconverged.

**Gate evals** (jobs `20260730-1842xx`, role `s2_gate`, 10 canonical windows × seeds {0,1,2}, 25-step rollouts; artifacts in `overfit100_s2_gate_artifacts/`, role validation OK at every checkpoint):

| checkpoint | mean m_corr | min | max |
|---|---|---|---|
| 250 | 0.7665 | 0.7062 | 0.8929 |
| 500 | 0.7993 | 0.7444 | 0.9008 |
| 1000 | 0.8347 | 0.7827 | 0.9183 |
| 2500 | **0.8896** | 0.8179 | **0.9426** |

Every window improved monotonically across all four checkpoints (10/10).

**Predeclared gate rule:** (i) mean m(w,2500) ≥ 0.70 → **0.8896 PASS** (margin +0.19); (ii) growth 250→2500 ≥ 0.15 → **+0.1230 MISS** (by 0.027); (iii) max ≥ 0.85 → **0.9426 PASS**. **Formal verdict: STOP & ANALYZE** (plan §7 R2 → escalation requires Yixun's approval).

**Text-ablation at 2500** (mean SSIM over 10 windows × 3 seeds): correct **0.8895** / null 0.8398 / shuffled 0.8342. The correct-text advantage is +0.050 vs null and +0.055 vs shuffled, and wrong text is *worse than no text* — the model genuinely uses language, with a modest effect size consistent with the duplicate-audit finding (58/100 taxonomy-label instructions) and first-frame fingerprints carrying much of the identification.

**Known defect found by the run:** the auxiliary RGB/VAE-ceiling path failed on all rows (`AttributeError: 'str' object has no attribute 'parent'` — a str-vs-Path bug in the aux fetch). The D5 machinery reported it exactly as designed (aux_status per row; run completed). Primary metrics unaffected. Fix queued before any S3 eval.

## S3 step-2500 segment-final (s3_segment_final, 900 rollouts) — landed 2026-07-31T23:39Z

Job `20260731-160907-6359b989-exp02-o100-s3ev-final2500-yixun`, SUCCEEDED on attempt 9 (8 preemption kills first). Artifact: `.../validation/step_002500_s3_segment_final/aggregation.json` (900 rows, role_validation.ok=true, manifest c02a67be…). Committed copy + formal verdict in `overfit100_s3_artifacts/`.

**Formal verdict (verdict CLI, segment-final only — full-set tier not yet evaluable):**

| Claim | Established | Detail |
| --- | --- | --- |
| Canonical-window memorization (≥90% of 100 at m_corr ≥ 0.95) | **NO** | fraction **0.0** (0/100); best window 0.9461; mean m_corr **0.8133**; fixed denominator 100, coverage complete |
| Full-set memorization | not evaluable | awaiting s3_full_set pass |

**m_corr distribution (median over seeds 0,1,2, correct mode):** mean 0.8133, median 0.8133, min 0.5870, max 0.9461; ≥0.90: 8/100; ≥0.85: 28/100; ≥0.80: 55/100.

**Ablations (mean SSIM over 300 rows each):** correct **0.8133** > null 0.7992 (gap 0.0141) > shuffled 0.7824 (gap 0.0310) — ordering correct at 100-episode scale; text conditioning contributes, margin modest (consistent with S2's 0.8895/0.8398/0.8342 at 10-episode scale).

**Checkpoint trajectory (canonical-100 mean, correct mode):** 0.7580 (250) → 0.7707 (500) → 0.7892 (1000) → 0.8020 (1750) → **0.8133 (2500)** — monotone, ~+0.011 per 750 steps at the tail, not saturated; train loss at 2500 was 0.145 and still falling. Read: memorization is progressing but far from the 0.95 bar at ~390 epochs — the D10 extension question (continue past 2,500 steps) is now the live decision.

**Aux:** all 900 rows `FileNotFoundError: ffmpeg` (expected — pre-fix tarball, issue #8); VAE ceilings recoverable via the checkpoint-independent backfill; primary metrics unaffected.

## S3 step-2500 full-set (s3_full_set, 1,629 rollouts) + COMPLETE TWO-TIER VERDICT — landed 2026-08-01T01:25Z-ish (attempt 9)

Job `20260731-160912-184642ed-exp02-o100-s3ev-fullset-yixun`, SUCCEEDED attempt 9 (8 preemption kills; the same calm window carried both big passes). Artifact: `.../validation/step_002500_s3_full_set/aggregation.json` (1,629 rows, role ok, covered 1,629/1,629). Committed copy + complete verdict in `overfit100_s3_artifacts/`.

**COMPLETE TWO-TIER VERDICT (verdict CLI, both artifacts admitted):**

| Tier | Established | Detail |
| --- | --- | --- |
| Canonical-window memorization (≥90% of 100 at m_corr ≥ 0.95) | **NO** | fraction 0.0 (0/100), mean m_corr 0.8133 |
| Full-set memorization (≥90% of 1,629 at seed-0 SSIM ≥ 0.90, c*=2500) | **NO** | fraction **0.0645** (105/1,629) |

**Full-set distribution (seed 0, correct mode):** mean 0.7984, median 0.8111, min 0.2589, max 0.9480; ≥0.90: 105 (6.4%); ≥0.85: 435 (26.7%); ≥0.80: 891 (54.7%); ≥0.70: 1,413 (86.7%). Consistent with the canonical cohort (mean 0.8133) — the full set adds a harder tail (min 0.259 vs canonical min 0.587).

**Bottom line at 2,500 steps (~390 epochs):** memorization is real, text-conditioned, monotonically improving, and uniformly incomplete — no window in either cohort reaches 0.95. Train loss 0.145 still falling. The D10 extension question (continue S3 beyond 2,500 steps) is the live decision; the alternative reading (rollout-error / objective-mismatch ceiling) is examined in `_analysis.md`.

**Aux:** all rows ffmpeg-missing as in segment-final (issue #8; pre-fix tarball); primary metrics unaffected; ceilings recoverable via the checkpoint-independent backfill.
