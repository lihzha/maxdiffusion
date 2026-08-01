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

## Sampling-steps probe (H1/H2 discriminator) @ ckpt 2500 — landed 2026-08-01T05:15Z-ish

Job 24 `20260801-042558-e41be63a-exp02-o100-probe-steps-yixun`, attempt 1. 30 seeded canonical windows × arms {25, 50, 100}, seed 0, correct context. Artifact committed: `overfit100_s3_artifacts/probe_steps_ckpt2500.json`.

**Validity: PASSED perfectly** — 25-arm reproduces the landed segment-final rows bitwise (max |Δssim| = 0.0; mean exactly 0.8100125855 = the reviewer's independently-computed anchor).

| arm | mean | median | min | max |
| --- | --- | --- | --- | --- |
| 25 | 0.8100 | 0.8059 | 0.6956 | 0.9440 |
| 50 | 0.8026 | 0.7977 | 0.6899 | 0.9385 |
| 100 | 0.7979 | 0.7937 | 0.6854 | 0.9359 |

Paired deltas: 50−25 mean **−0.0074** (0/30 improved); 100−25 mean **−0.0121** (0/30 improved).

**Read:** sampling-side H2 EXCLUDED. More integration steps strictly degrade reconstruction — velocity-field error compounding dominates discretization error already at 25 steps. No sampling knob recovers the gap to 0.95; the remaining hypotheses are H1 (more training — being tested by the 10k extension) and the one-step-objective ceiling (recipe change territory). Open question noted for the record: the monotone degradation suggests FEWER steps might do marginally better; untested (arms approval-pinned), not decision-relevant now.

## Extension intermediates i5000 + i7500 (s3_intermediate, seed-0 canonical) — landed 2026-08-01T~15:20–15:45Z

Jobs 25/26, both SUCCEEDED attempt 1, role ok, **aux 100/100 populated for the first time** (ffmpeg fix live; VAE ceiling mean 0.9493, min 0.8812 — context only, does not bound the decoded-vs-decoded metric). Artifacts committed alongside.

| step | mean | median | max | ≥0.95 | ≥0.90 | ≥0.85 | gain/250 steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2500 | 0.8133 | 0.8133 | 0.9461 | 0/100 | 8 | 28 | 0.0038 |
| 5000 | 0.8320 | 0.8319 | 0.9472 | 0/100 | 12 | 41 | 0.0019 |
| 7500 | 0.8377 | 0.8372 | 0.9475 | 0/100 | 17 | 43 | 0.0006 |

**Read: saturation ≈0.84.** Per-250-step gain fell 0.0038 → 0.0019 → 0.0006 (6× decay over 5k steps); the best window is frozen at ~0.947; no window has crossed 0.95 at any checkpoint. Tracks the loss flattening (0.145→0.132→0.127) and the §8 projection (0.82–0.84 at 10k). Combined with the probe (sampling axis closed), the picture entering the 10k verdict: **the one-step-denoising→25-step-rollout recipe saturates near mean 0.84 / max ~0.95 on this cohort — the D11 bar is not reachable by more training at this rate.**

## S3 step-10000 segment-final (s3_segment_final, 900 rollouts) — landed 2026-08-01T~16:25Z

Job 27 `20260801-142914-9d6458a0-…-final10k`, SUCCEEDED. **Resume staging proved out**: the pass survived an infra kill mid-run and finished from staged rows instead of restarting (657/900 banked at the time of the last check). Role ok, 900 rows, aux **900/900 populated**.

**Canonical tier at 10,000 steps: still NOT established.**

| statistic | step 2500 | step 10000 |
| --- | --- | --- |
| m_corr mean | 0.8133 | **0.8414** |
| m_corr median | 0.8133 | 0.8430 |
| best window | 0.9461 | 0.9484 |
| ≥0.95 | 0/100 | **0/100** |
| ≥0.90 | 8 | 17 |
| ≥0.85 | 28 | 46 |
| ≥0.80 | 55 | 78 |

Verdict CLI (10k generation, c*=10000, denominator 100, coverage complete): headline fraction **0.0**, established **False**.

**Full trajectory + saturation:** 0.7580 (250) → 0.7707 → 0.7892 → 0.8020 → 0.8133 (2500) → 0.8320 (5000) → 0.8377 (7500) → **0.8414 (10000)**. Gains per 2,500 steps after 2500: +0.0187, +0.0057, +0.0037 — a 5× decay, asymptote clearly below 0.90 and nowhere near 0.95.

**New finding — the text signal STRENGTHENS with training.** Ablation gaps vs correct: at 2500, null −0.0141 / shuffled −0.0310; at 10000, null **−0.0393** / shuffled **−0.0530**. The model leans on the instruction *more* as it memorizes more (~3× wider gap), which is the cleanest evidence yet that the conditioning path is genuinely used and not a nuisance variable.

**Process note (fail-closed guard fired correctly):** the verdict CLI **refused** to build one verdict from the 2500 artifacts (eval commit `e27fdc3`) mixed with the 10k artifacts (`46c5f41`) — "one verdict must be built from one run's passes". Correct and intended: exp_02 therefore reports **two internally-consistent verdicts**, one per eval-code generation, rather than one silently-mixed claim. The 2500 verdict (both tiers) stands as committed; the 10k verdict comprises segment-final@10000 + full-set@10000 (pending).

## S3 step-10000 full-set (s3_full_set, 1,629 rollouts) + FINAL VERDICT — landed 2026-08-01T~18:05Z

Job 28 `20260801-142941-95e2e0a1-…-fullset10k`, SUCCEEDED on attempt 2. **Resume carried it through a
preemption at 533/1,629** — it continued from staged rows rather than restarting, which is the difference
between finishing and never finishing under the day's spot weather. Role ok, 1,629 rows, aux 1,629/1,629.

**FINAL VERDICT — step-10,000 generation (eval `46c5f41`), c\*=10000, both tiers evaluable, coverage complete:**

| Tier | Rule | Measured | Verdict |
| --- | --- | --- | --- |
| Canonical | ≥90% of 100 at m_corr ≥ 0.95 | 0/100 (0.0%); mean 0.8414 | **not established** |
| Full-set | ≥90% of 1,629 at seed-0 ≥ 0.90 | **229/1,629 (14.1%)**; mean 0.8322 | **not established** |

**Full-set distribution at 10k** (seed 0, correct): mean 0.8322, median 0.8402, min 0.4445, **max 0.9509**;
≥0.95: **2** (0.1%); ≥0.90: 229 (14.1%); ≥0.85: 714 (43.8%); ≥0.80: 1,216 (74.6%); ≥0.70: 1,571 (96.4%).

**Correction to an earlier claim.** At 2,500 steps no window anywhere reached 0.95, and I had written that no
window did "at any training budget". That is now **false in the strict sense**: at 10,000 steps **2 of 1,629
full-set windows crossed 0.95** (max 0.9509) — the first crossings observed in the experiment. The verdict is
unaffected (0.1% against a required 90%), and the canonical cohort still has none, but the corrected statement
is: *crossings exist and are vanishingly rare*. The analysis-review finding that forced me to scope this claim
to measured cohorts (review item 7) is what made the correction visible rather than embarrassing.

**Progress across the extension (full-set tier):** ≥0.90 rose 105 → 229 windows (6.4% → 14.1%) and the mean
0.7984 → 0.8322 between steps 2,500 and 10,000 — real movement, ~6× short of the bar.

## Video passes (Jobs 29–30) — 600 comparison videos at ckpts 2500 + 10000 — landed 2026-08-01T~19:20Z

Both SUCCEEDED (10k needed attempt 1 after a setup-phase preemption). Fresh `s3_intermediate` role dirs at
each checkpoint; 100 windows × 3 videos = 300 mp4 each.

**Acceptance PASSED, exactly:** both passes reproduced their landed seed-0 SSIM **per window, bitwise**
(max |Δssim| = 0.00e+00, n=100 each; means 0.8139 @2500 and 0.8416 @10000 matching the verdict passes). The
rollouts are deterministic across independent jobs, code generations and machines — the strongest end-to-end
reproducibility evidence in the experiment.

**The 5 delivered windows** (representative spread by 10k m_corr; same windows at both checkpoints):

| role | window | ssim @2500 | ssim @10000 | Δ |
| --- | --- | --- | --- | --- |
| worst | `ep30738_v0_s00132` | 0.6020 | 0.6903 | **+0.0883** |
| 25th | `ep4358_v0_s00040` | 0.7585 | 0.8068 | +0.0483 |
| median | `ep4015_v0_s00000` | 0.8190 | 0.8398 | +0.0208 |
| 75th | `ep50125_v0_s00028` | 0.8508 | 0.8805 | +0.0297 |
| best | `ep36295_v0_s00020` | 0.9440 | 0.9483 | **+0.0043** |

**Finding — the extension helped the *worst* windows most, and the best hardly at all.** The gain is
monotonically ordered against starting quality: +0.088 at the bottom vs +0.004 at the top (a 20× spread). The
cohort mean's modest +0.028 is therefore a blend of real recovery in the tail and a genuinely saturated head.
This sharpens §4/§5 of the analysis: whatever caps the good windows near ~0.95 is *not* relieved by training,
while the poor windows were partly a dose problem. Consistent with the ceiling being a property of the
recipe rather than of optimization, and it also means "mean SSIM" understates how differently the cohort
behaves at its two ends.

All 600 videos remain on GCS at
`…/validation/step_{002500,010000}_s3_intermediate/mode_correct/seed_0/<window>/`.

## Diagnostics D1 + D3 (Yixun-approved) — 2026-08-01

Both run **locally on existing artifacts, zero TPU**. Script committed at `diagnostics/d1_per_frame_ssim.py`.

### D3 — episode weighting (H6): **REFUTED**

Training weights each episode by its 1–99 overlapping windows; the canonical statistic weights each episode
equally. If that mismatch under-trained sparse episodes, few-window episodes would score worse. They do not:

| quartile by window count | windows/ep | mean episode SSIM |
| --- | --- | --- |
| Q1 fewest | 1–5 | 0.8319 |
| Q2 | 5–12 | 0.8434 |
| Q3 | 12–20 | 0.8367 |
| Q4 most | 21–99 | 0.8364 |

Pearson r(n_windows, mean episode SSIM) = **−0.092** — flat, and the sign is *opposite* to the hypothesis.
H6 is eliminated.

### D1 — per-frame SSIM (H7): **the free frame is negligible; the compounding is the whole story**

Method: decode the rendered GT/pred mp4s and recompute SSIM per frame with an exact numpy replication of
`skimage.structural_similarity(channel_axis=-1, data_range=1.0, win_size=7)`. **Self-validating** — the
mean-over-frames must track the SSIM the eval recorded for the same window: measured **mean |diff| 0.0024,
max 0.0083** (n=14), so codec noise is negligible and the frame-index trend is trustworthy.

**Answer to H7 — the pinned frame is NOT inflating the score.** Frame 0 scores 0.9721, but it is 1 of 33
frames, lifting the reported mean by only **+0.0060**. The reported ~0.84 is genuine prediction.

**The real finding — SSIM decays monotonically along the rollout:** frame 0 (pinned) 0.9721 → frame 1 0.9142
→ … → frame 32 0.7106. A **−0.204 slide** from the first predicted frame to the last.

**Every window starts in the same place and fans out.** Across the 5 spread windows at step 10000:

| window | reported | frame 0 | frame 1 | last | decay (1→last) |
| --- | --- | --- | --- | --- | --- |
| worst | 0.690 | 0.9659 | 0.8955 | 0.5962 | **−0.2993** |
| 25th | 0.807 | 0.9733 | 0.8750 | 0.7632 | −0.1118 |
| median | 0.840 | 0.9825 | 0.9784 | 0.7427 | −0.2356 |
| 75th | 0.881 | 0.9778 | 0.9400 | 0.8430 | −0.0970 |
| best | 0.948 | 0.9802 | 0.9774 | 0.9366 | **−0.0407** |

Frame-0 fidelity is **uniformly 0.966–0.983 regardless of final score**; the difference between a 0.95 window
and a 0.69 window is almost entirely *how fast it degrades along the trajectory*. Tercile analysis agrees:
decay 1→last is −0.263 (bottom), −0.207 (middle), −0.161 (top) — worse windows decay faster, and the best
window barely decays at all.

**Why this matters for the verdict.** Every window's first predicted frame is already at or near the D11 bar.
**If the rollout preserved frame-0 fidelity across the trajectory, most windows would clear 0.95.** The
memorization is largely *there*; it is being spent on trajectory divergence. This is the most direct evidence
yet for H2b (objective/rollout mismatch) — previously supported only indirectly by the sampling probe — and
it converts "the recipe is the leading explanation" into a mechanism visible frame by frame.
