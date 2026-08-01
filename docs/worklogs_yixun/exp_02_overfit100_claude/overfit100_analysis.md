# exp_02 `overfit100` — Analysis (Planner)

**v4, written 2026-08-01** — adds §5b with the three Yixun-approved diagnostics (D1/D2/D3), which resolve
H5, H6 and H7 and replace the qualitative "leading explanation" with a quantitative mechanism. v3's text is
otherwise preserved; where the diagnostics change a claim, §5b says so explicitly.

**v3**, revising v2 against the Codex analysis review
(`overfit100_codex_analysis_review.md`, verdict SOUND-WITH-REVISIONS: 3 MAJOR / 4 MODERATE / 2 MINOR).
Every finding is addressed here; the resolution record is appended to the review file. v2's central error was
**overclaiming causation from an elimination that was narrower than I described** — corrected throughout.
I independently re-derived the reviewer's measurements from the artifacts and they all reproduce exactly.

> **Complete as of 2026-08-01T18:05Z** — the step-10,000 full-set pass landed (1,629/1,629, role ok) and is
> incorporated below. It changed no conclusion, but it did correct one factual claim: see §2.

## 1. The question, and the plain answer

**Question (Query 1).** Can full-FT Wan2.2 TI2V 5B memorize the 1,629 windows of 100 successful DROID
trajectories, conditioned on a per-trajectory language instruction plus the first-frame latent?

**Answer.** **No — at the tested budget and recipe, the predeclared criterion was not met.** At 10,000 steps,
0 of 100 canonical windows reached m_corr ≥ 0.95 (cohort mean 0.8414, best window 0.9484), and the full-set
tier reached 14.1% against a required 90%. This experiment does **not** show that the model can never
memorize this set — it shows that this recipe, at this budget, does not.

## 2. Verdicts

exp_02 reports **two internally-consistent verdicts, one per eval-code generation**, because the success
statistic *refused* to build a single verdict mixing step-2500 artifacts (eval commit `e27fdc3`) with
10k-generation artifacts (`46c5f41`): "one verdict must be built from one run's passes." The guard was right
and I did not route around it. The 10k canonical verdict is the **final-budget result**; the 2500 verdict is
retained as the historical complete-tier record. (A single formal cross-checkpoint verdict would require
re-running step 2500 under the 10k eval generation — not worth the compute.)

**Verdict B — step 10,000 (`46c5f41`), c\*=10000, denominator 100, coverage complete — FINAL:**

| Tier | Rule | Measured | Verdict |
| --- | --- | --- | --- |
| Canonical | ≥90% of 100 at m_corr ≥ 0.95 | **0/100 (0.0%)**; mean 0.8414, max 0.9484 | **not established** |
| Full-set | ≥90% of 1,629 at seed-0 ≥ 0.90 | **229/1,629 (14.1%)**; mean 0.8322, max 0.9509 | **not established** |

**Verdict A — step 2,500 (`e27fdc3`), both tiers — historical:**

| Tier | Measured | Verdict |
| --- | --- | --- |
| Canonical | 0/100 (0.0%); mean 0.8133, max 0.9461 | not established |
| Full-set | 105/1,629 (6.4%); mean 0.7984, max 0.9480 | not established |

**Correction, now that the last cohort is measured.** No *canonical* window reached 0.95 at any checkpoint.
But in the 1,629-window full set at 10,000 steps, **2 windows crossed 0.95** (max 0.9509) — the only crossings
observed in the experiment. v2's "no window in either cohort at any training budget" was therefore wrong in
the strict sense; the accurate statement is that crossings exist and are vanishingly rare (0.1%). The verdict
is untouched. This is precisely the claim the review (item 7) required me to scope to measured cohorts, and
scoping it is what made the correction routine instead of a retraction.

**Full-set movement across the extension:** ≥0.90 rose 105 → 229 windows (6.4% → 14.1%), mean 0.7984 → 0.8322
— real progress, roughly 6× short of the bar.

## 3. Reliability audit — precisely what was validated

- **Fail-closed chain held and demonstrably bites.** Manifest-derived fixed cohorts (never pass-derived),
  role grids validated on exact (window, checkpoint, seed, mode) tuples, `role_validation.ok=true` and complete
  coverage on every admitted pass, manifest binding `c02a67be…`, immutable role-keyed paths. The
  cross-generation refusal in §2 is the guard catching a real mistake of mine.
- **Reproducibility: exact SSIM-scalar reproduction, not "bitwise everything."** The probe's 25-step arm
  reproduced the landed segment-final SSIM scalars for its 30 windows exactly (max |Δssim| = 0, mean
  0.8100125855 matching the reviewer's independent computation). That validates determinism of the metric
  path end-to-end; it does not claim equality of full rollout tensors or whole artifact rows.
- **Seed robustness — of the verdict, not of every number.** At step 2500 the three canonical seeds
  independently gave 7 / 8 / 9 windows ≥ 0.90 and **zero** ≥ 0.95. Seed variation cannot plausibly reverse the
  D11 verdict. The 1,629-window fraction was measured at seed 0 only, so the uncertainty on "6.4%" itself is
  unquantified.
- **Cross-run consistency is supportive, not calibrated.** exp_01 (val loss, different cohort/aggregation),
  S2 (10 episodes) and S3 (100 episodes) trace a *qualitatively* consistent loss→rollout-SSIM relation. These
  are not like-for-like quantities — exp_01's own analysis warns as much — so this argues against, but does
  not exclude, a shared pipeline artifact.
- **Data integrity.** `_SUCCESS`-published dataset with byte-verify at preflight; V1 encode-parity r≈0.995;
  V4 frame-0 future-invariance bitwise-exact; V3 decode ceilings 0.902–0.966.
- **Instrument gap (worth stating plainly).** The reported losses are **noisy training logs**, not the
  fixed-RNG, all-window one-step loss instrument the plan contemplated. That instrument was never run, and its
  absence is why the "optimization floor" hypothesis in §5 cannot be closed here.

## 4. What the numbers say

**Canonical-cohort trajectory (seed-0, correct instruction — one consistent statistic at every checkpoint):**

| step | 250 | 500 | 1000 | 1750 | 2500 | 5000 | 7500 | 10000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean SSIM | 0.7580 | 0.7707 | 0.7892 | 0.8020 | 0.8139 | 0.8320 | 0.8377 | 0.8416 |
| mean latent MSE | — | — | — | — | 0.1592 | 0.1247 | 0.1146 | 0.1079 |

*(v2 mislabeled this row: it mixed m_corr means at 2500/10000 with seed-0 means at 5000/7500. The two differ
by ≤0.0006, but the label is now correct and consistent. m_corr means — the D11 statistic — are 0.8133 at
2500 and 0.8414 at 10000.)*

**Diminishing returns are real; the asymptote is not identified.** Gains per 2,500 steps after step 2500:
+0.0181, +0.0057, +0.0039 — the first increment is ~5× the last. But four tail points from a single
unreplicated run do not pin an asymptote: an exponential tail levels near 0.843 and an inverse-power tail near
0.851, while a power law *constrained* to asymptote at 0.90 still fits at ≈0.0013 RMSE. The defensible
statement is: **a practical plateau around 0.84–0.85, with another identical 10k extension having low expected
value** — not "the asymptote is measured."

**Latent MSE falls much faster than SSIM rises** — 0.1592 → 0.1079, a **32% reduction**, while seed-0 SSIM
moves only +3.4% (0.8139 → 0.8416) over the same span. The model's latent reconstruction is improving
substantially more than the pixel metric reports. This is a genuine caution about reading D11's SSIM as a
direct measure of memorization, and it completes the "latent trajectory" item v2 listed as future work.

**Content difficulty is entangled with the score.** Across the 100 canonical windows at 10k, per-window m_corr
correlates with the per-window VAE-vs-RGB ceiling at **Pearson r = 0.683**. Windows that are hard for the VAE
are also the windows the model scores worst on — so some of the spread is scene/texture difficulty, not
memorization depth.

**The per-window ceiling barely moved:** best window 0.9461 → 0.9472 → 0.9475 → 0.9484 across the extension.

## 5. Why 0.95 is not reached — narrowed, not eliminated

v2 claimed the hypotheses were "resolved by elimination." That was too strong. What the two discriminators
actually establish:

- **H1 — dose: further identical training is low-value, but dose is not excluded.** The 4× extension improved
  the mean by 0.028 and loss 0.145 → ≈0.12 while the gain rate decayed ~5×. This makes *more of the same*
  training a poor investment. It does not exclude a different LR (locked at 1e-5 by user decision), optimizer
  schedule, or convergence regime.
- **H2a — sampling: increasing the step count is excluded; the sampler is not.** The probe tested 25 → 50 →
  100 steps at checkpoint 2500: strictly worse, 0/30 windows improved at either arm. **Fewer** steps, a
  different solver, and different timestep spacing remain untested. Separately confirmed: this rollout has
  **no CFG branch at all** and training used none, so the suspected guidance mismatch does not exist.
- **H2b — objective/rollout mismatch: the leading explanation, not a proven cause.** Training optimizes
  single-step denoising; evaluation integrates 25 steps with frame-0 pinning, and nothing in the objective
  penalizes compounding error. The probe's *direction* (more integration → worse) is consistent with
  compounding dominating. The sharpest supporting anchor is S2: 10 episodes at loss 0.061 — far deeper
  memorization than S3 reached — still scored only 0.890.
- **H3 — capacity/interference: unlikely, not excluded.** 5B parameters on 100 trajectories, and the
  10-episode run fell short at a *lower* loss. But an S2 result cannot rule out scale-dependent interference
  at 100 episodes.
- **H4 — VAE ceiling: not a bound, and v2 misused it.** D11 compares decoded prediction against decoded
  target, so perfect latent reconstruction yields SSIM 1.0 **regardless** of the VAE's ≈0.949 fidelity against
  raw RGB. Those two 0.95-scale numbers have different reference targets and v2's "the bar was set at the
  VAE's own fidelity" was incoherent — withdrawn. What survives is the weaker, measured point above: content
  difficulty correlates with score at r = 0.683.
- **H5 (new, from the review) — optimization floor / instrument gap.** LR was locked at 1e-5 and the
  D11-contemplated fixed-RNG one-step loss instrument was never run. Without it, "the objective saturated"
  and "this optimization run saturated" are not separated.
- **H6 (new) — weighting mismatch.** Training weights each episode by its window count (1–99 overlapping
  windows); the canonical statistic weights each *episode* equally. Episodes contributing few windows are
  under-trained relative to how the verdict counts them.
- **H7 (new) — metric scope.** Primary SSIM includes the frame-0-pinned region, which is free. A future-only
  or per-frame SSIM would measure the predicted part alone.

**Honest summary: the objective/rollout mismatch is the leading explanation, and further identical training
is a poor investment — but exp_02 has narrowed the field, not identified a binding cause.**

## 5b. Diagnostics D1–D3 (Yixun-approved) — H5, H6, H7 resolved; the mechanism quantified

Three cheap measurements were run on existing checkpoints/artifacts. Full numbers in `_results.md`; script and
artifact in `diagnostics/`. They resolve every alternative §5 had left open, and together they replace the
qualitative story with a quantitative one.

**D3 — episode weighting (H6): REFUTED.** Episodes contributing 1–5 windows score 0.8319; those contributing
21–99 score 0.8364. Flat across quartiles, r = −0.092, sign opposite to the hypothesis. Eliminated.

**D1 — per-frame SSIM (H7): the free frame is negligible; compounding is visible directly.** The pinned frame
0 scores 0.9721 but is 1 of 33 frames, lifting the reported mean by only **+0.0060** — so the ~0.84 is genuine
prediction, and H7's inflation concern is answered in the negative. The real finding is the shape: SSIM decays
monotonically from 0.9142 (frame 1) to 0.7106 (frame 32), **−0.204 across the rollout**. Critically, *every*
window starts in the same place — frame-0 fidelity is 0.966–0.983 regardless of final score — and they fan out
along the trajectory (decay −0.041 for the best window, −0.299 for the worst). **If the rollout preserved
frame-0 fidelity, most windows would clear the 0.95 bar.** Method self-validates: mp4-decoded mean-over-frames
reproduces the eval's recorded SSIM to mean |diff| 0.0024.

**D2 — the fixed-RNG one-step loss instrument (H5): RESOLVED, and it corrects the framing.** This is the
instrument §3 flagged as missing. Three results:

1. **The optimization has not saturated.** The paired final leg (7,500→10,000) still reduces one-step loss at
   **42σ**, 1,609/1,629 windows improving; 2500→10000 is 48.6σ with 1,627/1,629 improving. Both "the objective
   saturated" and "this run saturated" are **false** — training is still descending, at 0.0035 per 2,500 steps
   and decelerating.
2. **The loss→SSIM mapping is linear and tight**: SSIM ≈ 0.9885 − 1.201 × loss, **r = −0.9994** across all
   eight checkpoints, per-leg ratio 0.99–1.50 with no trend. This *revises* v3's framing: the rollout metric
   never stopped responding to the objective. **The SSIM plateau is a consequence of the loss plateau, not a
   failure of transfer.**
3. **The bar's price:** SSIM 0.95 sits at one-step loss ≈ **0.032**, a further **74% reduction** from 0.1223 —
   unreachable at the observed, decelerating rate. *(Local linear fit over 0.19–0.12; need not hold at 0.03.)*

**The synthesis — two levers, now named precisely.** The fitted line's intercept implies a *perfect* one-step
denoiser would score ≈0.989, above the bar; its slope of 1.2 is the price the rollout charges for each unit of
one-step error — which is exactly the compounding D1 shows frame by frame. So:

- **Lower the loss floor** (optimization, capacity, LR): measured as decelerating and 74% short. Poor odds.
- **Lower the slope** (train on the trajectory the eval actually runs, so one-step error stops compounding):
  D1 shows the headroom is real, since every window's frame 0 already clears the bar.

This is the sharpest statement exp_02 supports, and it strengthens rather than replaces §5's conclusion: the
objective/rollout relationship is the binding constraint, and it is now measured (slope 1.2, intercept 0.989)
rather than inferred. What remains genuinely untested is whether a *different optimization regime* (LR was
locked at 1e-5) could reach a materially lower loss floor.

## 6. Text conditioning: dependence on the correct context grows with training

| | correct | null (empty) | shuffled (deranged) |
| --- | --- | --- | --- |
| S2 (10 ep, step 2500) | 0.8895 | 0.8398 (−0.050) | 0.8342 (−0.055) |
| S3 step 2500 | 0.8133 | 0.7992 (−0.014) | 0.7824 (−0.031) |
| S3 step 10000 | 0.8413 | 0.8020 (−0.039) | 0.7883 (−0.053) |

The effect is robust at the matched-pair level (reviewer-computed, reproduced here): the null gap widened in
**287/300** matched window-seed pairs and the shuffled gap in **277/300**; over the extension the correct mode
gained ≈0.028 while null gained only ≈0.0028 and shuffled ≈0.0059.

**What this licenses:** sensitivity to the *correct training context* increased strongly between 2,500 and
10,000 steps — the conditioning path is genuinely load-bearing. **What it does not license:** that the model
uses the instruction *semantically*. Prompt-as-identifier behaviour (the embedding acting as an episode key)
and growing out-of-distribution sensitivity to null/wrong context predict the same pattern. v2's claim that
"the first-frame latent already determines most of the frame" is also unmeasured — dropped. Note the ablations
ran at the ablation checkpoints only (2500 and 10000), not at every checkpoint.

## 7. Recommendations

**Do not run another identical extension.** That is the one strong operational conclusion: the measured gain
rate makes another 10k steps worth perhaps ~0.005 mean SSIM.

**The three cheap diagnostics have now been run** (§5b) and they point one way: the defect is the ~1.2×
amplification of one-step error into rollout error, not representation quality, not episode weighting, not a
free pinned frame, and not a saturated optimizer.

**exp_03 should attack the compounding directly** — train on
the trajectory the eval actually runs (multi-step / scheduled sampling, or a short-horizon rollout loss atop
the denoising objective). Also worth predeclaring: a success metric appropriate to the recipe (latent-space
reconstruction, or a perceptual metric with a calibrated bar), chosen **in advance** rather than used to
reinterpret exp_02 after the fact.

**One thing still worth testing cheaply before exp_03:** an LR/optimizer sweep at fixed short budget. D2
shows the loss is still descending but decelerating, and LR was locked at 1e-5 throughout — the one lever §5b
could not rule out. A few short runs at 2e-5 / 5e-5 would bound how much of the remaining 74% is reachable by
optimization alone.

**On the adapter program — a reference point, not a bound.** v2 claimed frozen adapters "should not be
expected to beat" 0.84; that is unsupported and is withdrawn. A particular full-FT optimization run does not
upper-bound a different parameterization: adapters can regularize optimization or reach different minima, and
the historical adapter runs differ in conditioning, split, objective and evaluation anyway. The correct
statement: **0.84 is an empirical full-FT reference under exp_02's exact protocol**, useful for calibrating
expectations; any direct adapter comparison needs a matched experiment.

## 8. Standing state

- Code closed and reviewed: suite **1,236 passed / 2 skipped**, ~25 Codex passes, every finding resolved on
  record. The eval-resume series (5 passes to APPROVE) proved itself in production — the 10k canonical pass
  survived an infra kill and finished from staged rows, and the full-set pass is doing the same now.
- Artifacts committed under `overfit100_s3_artifacts/`; HTML report at
  `overfit100_01_memorization_trajectory_results.html` (regenerates from the artifacts).
- Checkpoints {250, 500, 1000, 1750, 2500, 5000, 7500, 10000} retained on GCS.
- **All planned passes are complete.** Both tiers measured at both generations; no eval work outstanding.
- Open decisions for Yixun: **merge or leave unmerged** (SOP: merge only on confirmed success; the formal
  answer is *not established*, so the default is to leave code on the branch with docs on `yixun-dev`, matching
  the exp_01 precedent), and the **exp_03 direction** per §7 — including whether to run the three cheap
  diagnostics first.
