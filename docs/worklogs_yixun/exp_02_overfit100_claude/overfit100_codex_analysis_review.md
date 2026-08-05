# Codex analysis review — `overfit100_analysis.md` v2

- **Date:** 2026-08-01 | **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only
- **Verdict:** SOUND-WITH-REVISIONS — 3 MAJOR (causal conclusion overstated; alternative hypotheses not eliminated; adapter-program bound unsupported), 4 MODERATE (asymptote overstated + trajectory-table mislabel; text finding needs narrowing; reliability wording; VAE argument used incorrectly), 2 MINOR.
- **Reviewer's own measurements (independently computed from the artifacts, adopted into v3):** null gap rose in 287/300 matched pairs, shuffled in 277/300; correct improved 0.028 vs null 0.0028 / shuffled 0.0059; Pearson r≈0.68 between canonical m_corr and per-window VAE-vs-RGB ceiling; seed-0 latent MSE 0.1592→0.1247→0.1146→0.1079 (2500→10000); at step 2500 the three seeds gave 7/8/9 windows ≥0.90 and zero ≥0.95; a power law constrained to asymptote 0.90 fits the tail at ~0.0013 RMSE.

## Reviewer output (verbatim)

```
SOUND-WITH-REVISIONS

1. **MAJOR — The central elimination is not conclusive.** Claim: “**the recipe is the ceiling … independent of training dose and sampler settings**” and “**capacity/interference: excluded**.” The evidence supports diminishing returns under the tested configuration, but not this causal conclusion:

   - The 4× extension still improved mean SSIM by 0.028 and training loss from 0.145 to ≈0.12; it makes further identical training low-value, but does not exclude dose under a different LR, optimizer schedule, or convergence regime.
   - The probe excludes only *increasing this sampler from 25 to 50/100 steps* at checkpoint 2500. Fewer steps, another solver, or different timestep spacing remain untested.
   - S2’s lower loss and 0.890 SSIM on 10 episodes cannot exclude scale-dependent interference on 100 episodes.
   
   Correction: replace the causal conclusion with: “Under LR 1e-5, the present optimizer, and 25-step rollout, gains diminish sharply through 10k; one-step/rollout mismatch is the leading explanation, not an identified binding cause.” See [analysis §5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:91).

2. **MAJOR — Important alternative hypotheses remain insufficiently tested.** Claim: “**now resolved by elimination**” and “**the training objective itself: the surviving explanation**.” At least three live alternatives remain:

   - Optimization floor: LR was user-locked at 1e-5, and the reported losses are noisy training logs rather than the fixed-RNG, all-window one-step instrument required by D11. That planned instrument is not reported.
   - Metric/content effects: the primary SSIM includes the frame-0-pinned portion. My audit of the 10k aggregation also finds Pearson \(r≈0.68\) between canonical `m_corr` and the per-window VAE-vs-RGB ceiling, indicating substantial content/texture difficulty entanglement even though the VAE is not a mathematical bound.
   - Window/data structure: training weights episodes by their 1–99 overlapping windows, while the canonical statistic weights each episode equally.
   
   Correction: retain these as alternatives and run future-only/per-frame SSIM, the already-planned fixed one-step loss curve, and a one-episode or extended-S2 optimization bound before declaring the objective causal. This also follows the approved plan’s escalation order in [R2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/plan_overfit100.md:86).

3. **MODERATE — Diminishing gains are real; the claimed asymptote is overstated.** Claim: “**the asymptote sits below 0.90**” and “**the asymptote is measured**.” The +0.0187/+0.0057/+0.0037 increments do establish a pronounced slowdown, and “5× decay” is honest: the first increment is 5.1× the last. But four tail points, with no replicated training runs, do not statistically identify an asymptote.

   Independent fits illustrate the model dependence: an exponential tail levels near 0.8425 and an inverse-power tail near 0.8508, both below 0.90, while a slow log-like continuation remains compatible with later improvement. A power-law constrained to asymptote at 0.90 fits with only about 0.0013 RMSE. Thus “clearly below 0.90” is too certain.

   The table is also mislabeled “seed-0”: steps 2500/10000 use mean `m_corr`, whereas 5000/7500 are seed-0 intermediates, though the numerical discrepancy is small. Correction: say “simple tail fits suggest a practical plateau around 0.84–0.85, but the true asymptote is not identifiable; another identical 10k extension has low expected value.” See [trajectory and extrapolation](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:74).

4. **MODERATE — The text effect is real, but its interpretation should be narrower.** Claim: “**The model leans on the instruction more as it memorizes more**.” The aggregate finding is robust:

   - Null gap relative to correct rose from 1.73% to 4.67%.
   - Shuffled gap rose from 3.81% to 6.30%.
   - Recomputing all matched window-seed pairs, the null gap increased in 287/300 pairs and the shuffled gap in 277/300.
   - Correct improved by ≈0.028, while null and shuffled improved only ≈0.0028 and ≈0.0059.

   This demonstrates increasing dependence on the correct text embedding. It does not distinguish semantic instruction use from prompt-as-identifier behavior or increasing out-of-distribution sensitivity to null/wrong context. Nor is “the first frame determines most” measured. Also, ablations were not run “at every checkpoint,” only the listed ablation checkpoints.

   Correction: “Sensitivity to the correct training context increased strongly between 2500 and 10000; this is consistent with greater reliance on text conditioning, without establishing semantic use.” See [conditioning table](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:119).

5. **MODERATE — Parts of the reliability argument overstate what was validated.** Claims: “**verified bitwise**,” “**three independent runs trace one coherent loss→rollout-SSIM relation**,” and “**seed noise is not in play**.”

   - The probe exactly reproduced the 30 selected SSIM scalars at 25 steps (`max |Δssim|=0`), which is excellent, but it did not establish bitwise equality of complete rollout tensors or entire artifact rows.
   - The cross-experiment loss comparison is confounded: exp_01 used differently aggregated train/validation losses and different cohorts; S2/S3 use noisy training logs on different dataset sizes. Even the exp_01 analysis warns these are not calibrated, like-for-like quantities. Consistency is supportive, but does not “strongly” exclude a shared pipeline artifact.
   - The negative verdict is robust to seeds: at step 2500, the three canonical seeds produced only 7/8/9 windows at ≥0.90 and zero at ≥0.95. But the 1,629-window fraction was measured only at seed 0, so uncertainty in the exact 6.4% remains unmeasured.
   
   Correction: distinguish exact SSIM-scalar reproduction, qualitative cross-run consistency, and the narrower conclusion that seed variation cannot plausibly reverse the D11 verdict.

6. **MODERATE — The VAE argument is internally qualified but still used incorrectly.** Claim: “**the bar was set at roughly the fidelity of the VAE’s own reconstruction**” and recommendation to “**re-set the success metric**.” D11’s primary statistic compares decoded prediction against decoded target. Therefore perfect latent reconstruction yields primary SSIM 1.0 regardless of the VAE’s ≈0.949 SSIM against raw RGB. The two 0.95-scale numbers concern different reference targets.

   Correction: describe 0.95 as an empirically stringent SSIM threshold, not a codec ceiling. Report latent MSE and future-only SSIM alongside D11; any replacement success threshold should be predeclared in a new experiment, not used to reinterpret exp_02 post hoc. The committed artifacts already show seed-0 latent MSE declining 0.1592 → 0.1247 → 0.1146 → 0.1079 from 2500–10000, so recommendation §7.3 can be completed now.

7. **MINOR — The two-verdict structure is the correct scientific choice.** Claim: “**two internally-consistent verdicts, one per eval-code generation**.” The refusal to mix `e27fdc3` and `46c5f41` correctly preserves provenance; transparent reporting does not create material cherry-picking risk. The 10k canonical verdict should be foregrounded as the final-budget result, with the complete 2500 verdict retained historically. If a single formal cross-checkpoint D11 verdict is required, step 2500 must be rerun under the 10k eval generation.

   The pending 10k full-set pass affects only the stronger full-set tier. None of the 10k canonical failure, observed slowdown, sampling probe, or conditioning result depends on it. However, “**No window in either cohort … at any training budget**” must be restricted: the 10k full set has not been measured. See [two verdicts and pending row](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:23).

8. **MAJOR — The adapter-program “bound” is unsupported.** Claim: “**Frozen-backbone adapters should not be expected to beat [0.84] … an adapter result near 0.8 is at the recipe’s ceiling**.” A particular full-FT optimization run is not an upper bound on a different parameterization or optimization regime; adapters can regularize optimization, add different computation paths, or reach different minima. Historical adapter runs also differ in conditioning, split, objective, and evaluation.

   Correction: call 0.84 an empirical full-FT reference under this exact exp_02 protocol. Direct adapter implications require a matched experiment. This correction is already explicitly supported by [exp_01’s final analysis](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_01_full_ft_overfit_claude/full_ft_overfit_analysis.md:57).

9. **MINOR — The actual user question is answered early, but later causal wording exceeds D11.** Claim: “**not established**.” That is the correct plain answer: at 10,000 steps, the model did not meet the predeclared canonical memorization rule—0/100 windows reached median SSIM ≥0.95. “Not established” is appropriate rather than “impossible.” Later statements that the threshold “is not reachable” or that the recipe is a proven ceiling go beyond what D11 licenses.

   Correction: lead with: “At the tested 10k budget and recipe: no—the predeclared canonical memorization criterion was not met; this experiment does not prove that the model can never memorize the set.”

A1 — **DOES NOT HOLD:** the tested alternatives were narrowed, not eliminated; objective mismatch remains a leading hypothesis.

A2 — **PARTLY HOLDS:** diminishing returns and the descriptive 5× decay are sound; the below-0.90 asymptote is not statistically established.

A3 — **HOLDS WITH HEDGE:** increasing dependence on correct text is robust; semantic instruction use is not identified.

A4 — **PARTLY HOLDS:** artifact integrity is strong, but bitwise, cross-experiment, and seed claims need narrower wording.

A5 — **HOLDS:** separate verdicts are conservative and correct; the pending 10k full set affects only that stronger tier.

A6 — **PARTLY HOLDS:** stopping another identical extension is rational, but objective-first exp_03 and the adapter bound are overclaimed.

A7 — **PARTLY HOLDS:** the formal negative answer is plain and correct; the later causal ceiling language exceeds D11.

The single most important thing the analysis gets right: the predeclared canonical memorization claim reliably failed at 10k, with fixed denominators, complete three-seed coverage, and fail-closed provenance.

The single most important thing it gets wrong or misses: it converts a strong negative result under tested settings into causal proof that the one-step objective is the binding ceiling, without excluding optimization-, scale-, and metric-related alternatives.
tokens used
176,371
SOUND-WITH-REVISIONS

1. **MAJOR — The central elimination is not conclusive.** Claim: “**the recipe is the ceiling … independent of training dose and sampler settings**” and “**capacity/interference: excluded**.” The evidence supports diminishing returns under the tested configuration, but not this causal conclusion:

   - The 4× extension still improved mean SSIM by 0.028 and training loss from 0.145 to ≈0.12; it makes further identical training low-value, but does not exclude dose under a different LR, optimizer schedule, or convergence regime.
   - The probe excludes only *increasing this sampler from 25 to 50/100 steps* at checkpoint 2500. Fewer steps, another solver, or different timestep spacing remain untested.
   - S2’s lower loss and 0.890 SSIM on 10 episodes cannot exclude scale-dependent interference on 100 episodes.
   
   Correction: replace the causal conclusion with: “Under LR 1e-5, the present optimizer, and 25-step rollout, gains diminish sharply through 10k; one-step/rollout mismatch is the leading explanation, not an identified binding cause.” See [analysis §5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:91).

2. **MAJOR — Important alternative hypotheses remain insufficiently tested.** Claim: “**now resolved by elimination**” and “**the training objective itself: the surviving explanation**.” At least three live alternatives remain:

   - Optimization floor: LR was user-locked at 1e-5, and the reported losses are noisy training logs rather than the fixed-RNG, all-window one-step instrument required by D11. That planned instrument is not reported.
   - Metric/content effects: the primary SSIM includes the frame-0-pinned portion. My audit of the 10k aggregation also finds Pearson \(r≈0.68\) between canonical `m_corr` and the per-window VAE-vs-RGB ceiling, indicating substantial content/texture difficulty entanglement even though the VAE is not a mathematical bound.
   - Window/data structure: training weights episodes by their 1–99 overlapping windows, while the canonical statistic weights each episode equally.
   
   Correction: retain these as alternatives and run future-only/per-frame SSIM, the already-planned fixed one-step loss curve, and a one-episode or extended-S2 optimization bound before declaring the objective causal. This also follows the approved plan’s escalation order in [R2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/plan_overfit100.md:86).

3. **MODERATE — Diminishing gains are real; the claimed asymptote is overstated.** Claim: “**the asymptote sits below 0.90**” and “**the asymptote is measured**.” The +0.0187/+0.0057/+0.0037 increments do establish a pronounced slowdown, and “5× decay” is honest: the first increment is 5.1× the last. But four tail points, with no replicated training runs, do not statistically identify an asymptote.

   Independent fits illustrate the model dependence: an exponential tail levels near 0.8425 and an inverse-power tail near 0.8508, both below 0.90, while a slow log-like continuation remains compatible with later improvement. A power-law constrained to asymptote at 0.90 fits with only about 0.0013 RMSE. Thus “clearly below 0.90” is too certain.

   The table is also mislabeled “seed-0”: steps 2500/10000 use mean `m_corr`, whereas 5000/7500 are seed-0 intermediates, though the numerical discrepancy is small. Correction: say “simple tail fits suggest a practical plateau around 0.84–0.85, but the true asymptote is not identifiable; another identical 10k extension has low expected value.” See [trajectory and extrapolation](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:74).

4. **MODERATE — The text effect is real, but its interpretation should be narrower.** Claim: “**The model leans on the instruction more as it memorizes more**.” The aggregate finding is robust:

   - Null gap relative to correct rose from 1.73% to 4.67%.
   - Shuffled gap rose from 3.81% to 6.30%.
   - Recomputing all matched window-seed pairs, the null gap increased in 287/300 pairs and the shuffled gap in 277/300.
   - Correct improved by ≈0.028, while null and shuffled improved only ≈0.0028 and ≈0.0059.

   This demonstrates increasing dependence on the correct text embedding. It does not distinguish semantic instruction use from prompt-as-identifier behavior or increasing out-of-distribution sensitivity to null/wrong context. Nor is “the first frame determines most” measured. Also, ablations were not run “at every checkpoint,” only the listed ablation checkpoints.

   Correction: “Sensitivity to the correct training context increased strongly between 2500 and 10000; this is consistent with greater reliance on text conditioning, without establishing semantic use.” See [conditioning table](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:119).

5. **MODERATE — Parts of the reliability argument overstate what was validated.** Claims: “**verified bitwise**,” “**three independent runs trace one coherent loss→rollout-SSIM relation**,” and “**seed noise is not in play**.”

   - The probe exactly reproduced the 30 selected SSIM scalars at 25 steps (`max |Δssim|=0`), which is excellent, but it did not establish bitwise equality of complete rollout tensors or entire artifact rows.
   - The cross-experiment loss comparison is confounded: exp_01 used differently aggregated train/validation losses and different cohorts; S2/S3 use noisy training logs on different dataset sizes. Even the exp_01 analysis warns these are not calibrated, like-for-like quantities. Consistency is supportive, but does not “strongly” exclude a shared pipeline artifact.
   - The negative verdict is robust to seeds: at step 2500, the three canonical seeds produced only 7/8/9 windows at ≥0.90 and zero at ≥0.95. But the 1,629-window fraction was measured only at seed 0, so uncertainty in the exact 6.4% remains unmeasured.
   
   Correction: distinguish exact SSIM-scalar reproduction, qualitative cross-run consistency, and the narrower conclusion that seed variation cannot plausibly reverse the D11 verdict.

6. **MODERATE — The VAE argument is internally qualified but still used incorrectly.** Claim: “**the bar was set at roughly the fidelity of the VAE’s own reconstruction**” and recommendation to “**re-set the success metric**.” D11’s primary statistic compares decoded prediction against decoded target. Therefore perfect latent reconstruction yields primary SSIM 1.0 regardless of the VAE’s ≈0.949 SSIM against raw RGB. The two 0.95-scale numbers concern different reference targets.

   Correction: describe 0.95 as an empirically stringent SSIM threshold, not a codec ceiling. Report latent MSE and future-only SSIM alongside D11; any replacement success threshold should be predeclared in a new experiment, not used to reinterpret exp_02 post hoc. The committed artifacts already show seed-0 latent MSE declining 0.1592 → 0.1247 → 0.1146 → 0.1079 from 2500–10000, so recommendation §7.3 can be completed now.

7. **MINOR — The two-verdict structure is the correct scientific choice.** Claim: “**two internally-consistent verdicts, one per eval-code generation**.” The refusal to mix `e27fdc3` and `46c5f41` correctly preserves provenance; transparent reporting does not create material cherry-picking risk. The 10k canonical verdict should be foregrounded as the final-budget result, with the complete 2500 verdict retained historically. If a single formal cross-checkpoint D11 verdict is required, step 2500 must be rerun under the 10k eval generation.

   The pending 10k full-set pass affects only the stronger full-set tier. None of the 10k canonical failure, observed slowdown, sampling probe, or conditioning result depends on it. However, “**No window in either cohort … at any training budget**” must be restricted: the 10k full set has not been measured. See [two verdicts and pending row](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md:23).

8. **MAJOR — The adapter-program “bound” is unsupported.** Claim: “**Frozen-backbone adapters should not be expected to beat [0.84] … an adapter result near 0.8 is at the recipe’s ceiling**.” A particular full-FT optimization run is not an upper bound on a different parameterization or optimization regime; adapters can regularize optimization, add different computation paths, or reach different minima. Historical adapter runs also differ in conditioning, split, objective, and evaluation.

   Correction: call 0.84 an empirical full-FT reference under this exact exp_02 protocol. Direct adapter implications require a matched experiment. This correction is already explicitly supported by [exp_01’s final analysis](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_01_full_ft_overfit_claude/full_ft_overfit_analysis.md:57).

9. **MINOR — The actual user question is answered early, but later causal wording exceeds D11.** Claim: “**not established**.” That is the correct plain answer: at 10,000 steps, the model did not meet the predeclared canonical memorization rule—0/100 windows reached median SSIM ≥0.95. “Not established” is appropriate rather than “impossible.” Later statements that the threshold “is not reachable” or that the recipe is a proven ceiling go beyond what D11 licenses.

   Correction: lead with: “At the tested 10k budget and recipe: no—the predeclared canonical memorization criterion was not met; this experiment does not prove that the model can never memorize the set.”

A1 — **DOES NOT HOLD:** the tested alternatives were narrowed, not eliminated; objective mismatch remains a leading hypothesis.

A2 — **PARTLY HOLDS:** diminishing returns and the descriptive 5× decay are sound; the below-0.90 asymptote is not statistically established.

A3 — **HOLDS WITH HEDGE:** increasing dependence on correct text is robust; semantic instruction use is not identified.

A4 — **PARTLY HOLDS:** artifact integrity is strong, but bitwise, cross-experiment, and seed claims need narrower wording.

A5 — **HOLDS:** separate verdicts are conservative and correct; the pending 10k full set affects only that stronger tier.

A6 — **PARTLY HOLDS:** stopping another identical extension is rational, but objective-first exp_03 and the adapter bound are overclaimed.

A7 — **PARTLY HOLDS:** the formal negative answer is plain and correct; the later causal ceiling language exceeds D11.

The single most important thing the analysis gets right: the predeclared canonical memorization claim reliably failed at 10k, with fixed denominators, complete three-seed coverage, and fail-closed provenance.

The single most important thing it gets wrong or misses: it converts a strong negative result under tested settings into causal proof that the one-step objective is the binding ceiling, without excluding optimization-, scale-, and metric-related alternatives.
```

## Resolution record (Planner, 2026-08-01) — analysis v3

Every finding accepted; **none rejected**. All reviewer measurements independently re-derived from the
artifacts before adoption and all reproduce exactly (latent MSE 0.1592/0.1247/0.1146/0.1079; r=0.683;
seeds 7/8/9 at ≥0.90 with zero ≥0.95; m_corr-vs-seed0 label discrepancy ≤0.0006).

| # | Sev | Finding | Resolution in v3 |
| --- | --- | --- | --- |
| 1 | MAJOR | Causal conclusion ("the recipe is the ceiling … independent of dose and sampler") unwarranted | §5 rewritten as **narrowed, not eliminated**. H1 → "further identical training is low-value, dose not excluded" (LR locked at 1e-5; other regimes untested). H2a → "increasing step count excluded; fewer steps / other solvers / spacing untested". H2b → "leading explanation, not a proven cause". §1 and §7 reworded to match. |
| 2 | MAJOR | Live alternatives untested (optimization floor + missing one-step instrument; metric/content entanglement; window-weighting) | Added as **H5 (optimization floor / instrument gap)**, **H6 (weighting mismatch)**, **H7 (metric scope)**. The missing fixed-RNG one-step instrument is now called out in §3 as an explicit instrument gap, and all three become the §7 cheap-diagnostics list, ordered ahead of any exp_03 commitment. |
| 3 | MODERATE | Asymptote overstated; trajectory table mislabeled | §4 now reports "practical plateau ≈0.84–0.85, asymptote not identifiable", quoting the reviewer's competing fits incl. the 0.90-constrained power law at ~0.0013 RMSE. Table relabeled to a **single consistent seed-0 statistic** at every checkpoint, with the m_corr values stated separately and the v2 error disclosed inline. |
| 4 | MODERATE | Text finding over-interpreted as semantic use | §6 narrowed to "sensitivity to the correct training context increased"; explicitly does **not** distinguish semantic use from prompt-as-identifier or OOD sensitivity. Reviewer's matched-pair counts (287/300, 277/300) adopted. v2's unmeasured "first frame determines most" claim dropped; "ablations at every checkpoint" corrected. |
| 5 | MODERATE | Reliability wording overstated ("bitwise", cross-experiment, seed noise) | §3 rewritten: "exact SSIM-scalar reproduction" (not bitwise tensors); cross-run consistency "supportive, not calibrated"; seed robustness scoped to **the verdict**, with the 6.4% fraction's own uncertainty stated as unquantified. |
| 6 | MODERATE | VAE argument used incorrectly | v2's "the bar was set at the VAE's own fidelity" **withdrawn** as incoherent (D11 is decoded-vs-decoded; perfect latent reconstruction gives 1.0 regardless). Replaced with the measured, weaker r=0.683 content-difficulty entanglement. Latent-MSE trajectory added to §4 — this **completes** v2's deferred §7.3 item and surfaces that latent MSE falls 32% while SSIM rises 3.4%. |
| 7 | MINOR | Two-verdict structure correct; "no window in either cohort" over-scoped | Structure retained with the 10k verdict **foregrounded as final** and 2500 marked historical; the no-window claim is now scoped to the canonical cohort + the 2500 full set, since the 10k full set is unmeasured. |
| 8 | MAJOR | Adapter-program "bound" unsupported | **Withdrawn.** §7 now says 0.84 is "an empirical full-FT reference under exp_02's exact protocol", explicitly not an upper bound on a different parameterization, and notes a matched experiment is required — consistent with exp_01's own analysis. |
| 9 | MINOR | Plain answer buried; "not reachable" exceeds D11 | New §1 "The question, and the plain answer" leads with: not met at this budget/recipe, and explicitly **not** a proof that the model can never memorize the set. "Not reachable" language removed throughout. |

**Reviewer's "most important thing right"** — the fail-closed provenance discipline and the two-verdict
structure. **"Most important thing wrong"** — asserting an identified cause where the evidence supports a
leading hypothesis. v3 fixes the latter without weakening the former.

## v5 review (2026-08-05, xhigh, account #3) — SOUND-WITH-REVISIONS → v5.1

Reviewer independently reproduced the formal verdict (seed cells 0.953635/67, 0.953701/69,
0.953574/69; median-of-3 69/100 at 0.95, 98/100 at 0.90; full-set 1,618/1,629 = 99.3247%; ablation
gaps exact; seed span 0.000128). Five findings, ALL ACCEPTED and applied as v5.1:
1. **MAJOR — "LR axis closed" overclaim** → "tested fixed-LR escalations through 1e-4 fail to
   restore pace" (no contemporaneous 5e-5 control at 17.5k→20k; schedules/optimizers untested);
   pace factors corrected (2.89×, then 4.74× vs predecessor; 1e-4 recovered 34.7%).
2. **MODERATE — two §9 SSIMs were line-predictions printed as measurements** (0.9174/0.9427 →
   measured 0.9159/0.9451); residuals recomputed (+0.00019/+0.00380/+0.00404/+0.00501); the
   propagated values corrected in `_results.md` (with a correction note) and the tracker.
3. **MAJOR — §10 intercept/slope causal-extrapolative overreach** → in-range MEAN forecaster; the
   observed 20k point (not the intercept) shows the mean can exceed 0.95; a mean law is silent on
   the per-window 90/100 criterion; "six independent checkpoints" → five held-out evaluations,
   sequential checkpoints not independent.
4. **MODERATE — "actively follows the instruction"** violated §6's preserved non-license →
   dependence strengthened; semantic-vs-identifier still indistinguishable.
5. **MAJOR — §12 "fully resolved" / "dynamical, not representational"** → hypothesis space
   *updated*; compounding = LEADING MECHANISM, not an exclusion; the tail (31/100 windows) not
   identified as non-representational; capacity at the margin unexcluded.
Citation guidance recorded: cite `partial`, 69/100, 99.3247% confidently; describe LR as "tested
through 1e-4 without pace restoration"; treat the law as an in-range aggregate predictor.

