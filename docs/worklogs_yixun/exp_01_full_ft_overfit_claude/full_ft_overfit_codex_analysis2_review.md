# Analysis review: exp_01 full_ft_overfit — Part II addendum
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-27

## Context loaded

- Queries 8–9: held-out evaluation specification and conditional TPU authorization.
- Results: complete T1/T2 tables and Part-I train-cohort cross-references.
- Analysis: reviewed Part I v2 and the proposed Part-II addendum.
- Worklog: Part-II reviews, dataset scan, smoke PASS, and full-job launches.

## Verdict

REQUEST-REVISION — The held-out evidence is valuable and the cited experimental numbers reconcile, but the addendum materially overstates what it establishes about absence of overfitting, domain learning, and an exp_02 upper bound. These are wording and inference corrections; the underlying T1/T2 results remain sound.

## Findings

1. **MAJOR — “No overfitting, anywhere” is not supported.** T1 establishes that aggregate held-out loss did not deteriorate at the eight sampled checkpoints: 0.184468 → 0.178854 (−3.0%). It cannot exclude every form of overfitting, especially when T2 itself has a train–val rollout difference. Moreover, 0.1763 is a 10-step windowed training log with fresh stochastic draws, whereas 0.178854 is an exact full-val mean with fixed per-position draws; their ≈1.5% difference is not a calibrated generalization gap. **Concrete edit:** replace finding 1 with: “At the eight evaluated checkpoints, no aggregate held-out-loss degradation is detected; val loss decreases from 0.184468 to 0.178854. The final train and val losses are broadly consistent, but their differently aggregated protocols preclude treating the difference as a formal train/val gap.” Delete “generalizes essentially everything it learns.”

2. **MAJOR — The plateau reconciliation is plausible, but its causal and saturation wording is too strong.** The lower-noise full-val curve shows a slow one-step-loss tail that the train windows did not visibly resolve. It does not prove that windowed noise was the sole reason for the apparent plateau. Likewise, four 16-clip train-cohort means near 0.787 support “no detected rollout improvement after 5k,” not “genuinely saturated” or a general claim that later loss gains cannot convert to rollout gains. **Concrete edit:** write: “Part I’s apparent train-loss plateau contains a small continued improvement under the full-val instrument. The evaluated 16-clip train rollout means show no detectable gain after step 5000 under this single-seed protocol.”

3. **MAJOR — T2 demonstrates transfer, but does not resolve domain learning versus memorization quantitatively.** Held-out SSIM 0.7269 and latent MSE 0.3356 are strong evidence that improvement transfers to unseen clips, ruling out a purely train-clip-only explanation. However, the 16-train and 6-val cohorts are unequal, unpaired, and potentially different in difficulty; the val SSIM range is wide, and there is no step-0 val comparator. Thus the 0.0606 SSIM difference cannot be labeled “mild specialization,” nor can the relative contributions of generalization and memorization be identified. The 92% ratio is also misleading: SSIM is not a ratio-scale quantity, and dividing two means from different cohorts does not measure “quality retained.” **Concrete edit:** report the raw means and say: “The held-out result establishes substantial domain transfer, while the experiment remains unable to quantify how much additional train-clip memorization occurred.” Remove the 92% framing and the hypothetical “a memorizing model would show train ≫ val.”

4. **MAJOR — “Upper-bound,” “ceiling,” and “like-for-like” are unjustified for exp_02.** This single full-FT run supplies an empirical reference at its particular budget, optimizer, and hyperparameters—not an upper bound. A lower-dimensional or regularized method can outperform a particular full-FT optimization run. More importantly, Part I explicitly records objective, conditioning, and split confounds for the historical adapter runs; adding a held-out full-FT cohort fixes the train-proxy issue but not those other confounds. **Concrete edit:** call 0.1789 and 0.7269 “full-FT reference targets under this run’s protocol.” State that historical adapter SSIMs remain contextual and become direct comparators only in the proposed identifying design with conditioning, guide scale, data, dose, and evaluation held fixed.

5. **MINOR — Statistical and measurement caveats need tightening.** “≈8× stderr” compares the endpoint change with a marginal per-checkpoint stderr, not the stderr of the paired per-example checkpoint differences; it should not imply an 8σ change. The reported stderr is per window and does not address possible correlation among windows. Absolute T1 values also use one validation RNG seed, while T2 uses six clips and one rollout seed. Finally, rollout metrics compare predictions with VAE decodes of cached `z_video`, not original DROID RGB, and no human qualitative review is recorded. **Concrete edit:** retain the exact −0.005614/−3.0% change, drop “8× stderr” unless a paired-difference stderr is computed, and add these limitations to Reliability.

6. **MINOR — Numerical reconciliation passes, but the final evidence trail is incomplete.** All T1/T2, train-cohort, baseline, range, and loss values cited in the addendum match `full_ft_overfit_results.md` within stated rounding. The adapter SSIM values 0.30 and 0.62–0.66 appear in Part I but not in the results cross-reference, and the worklog currently records the smoke PASS plus jobs 12–13 launches without their final log-verified acceptance. **Concrete edit:** source the adapter figures explicitly or remove them from this addendum, and append T1-full/T2 completion entries with job IDs and acceptance checks before marking the human-facing record final.

---

# Resolutions (Planner, addendum v2 — 2026-07-27)

All 6 findings **accepted**, every concrete edit applied verbatim-or-stronger in `_analysis.md` (Part-II addendum v2) and the report-02 HTML verdict:

- **F1** — "no overfitting, anywhere" → the ordered narrow statement; "generalizes essentially everything" deleted; train/val difference explicitly de-calibrated (different estimators).
- **F2** — plateau wording → "apparent plateau contains a small continued improvement under the full-val instrument"; rollout claim → "no detectable gain after 5000 under this single-seed protocol".
- **F3** — memorization framing → "substantial domain transfer established; memorization share not quantified"; 92% ratio and the train≫val hypothetical removed; cohort asymmetry + no step-0-val comparator stated.
- **F4** — "upper bound/ceiling/like-for-like" → "full-FT reference targets under this run's protocol"; adapter numbers contextual until the identifying design.
- **F5** — "≈8× stderr" dropped (exact −0.005614/−3.0% retained); marginal-vs-paired stderr, window correlation, single-seed T1, 6-clip/1-seed T2, VAE-decode GT, and no-human-review caveats added to Reliability.
- **F6** — adapter figures sourced explicitly in the worklog note; T1-full/T2 completion entries with job ids + acceptance checks appended to `_command.md`/`_worklog.md` before finalization.
