# exp_02 `overfit100` — Analysis (Planner)

**v2, written 2026-08-01** after the step-10,000 canonical verdict landed and the sampling probe closed
the H1/H2 discrimination. Author: Planner. Supersedes v1 (step-2500 only); v1's §§1–7 are preserved
below in substance, with §5's open question now **resolved**. Inputs: all committed code, configs,
`_results.md`, the verdict CLI outputs and aggregations in `overfit100_s3_artifacts/`, the S2 gate
artifacts, the probe artifact, and the training logs recorded in `_worklog.md` / `_command.md`.

> Pending at time of writing: the step-10,000 **full-set** pass (1,629 windows) is still running, resuming
> across preemptions from staged rows. It confirms the second tier at the 10k generation; it cannot change
> the conclusion, which rests on the canonical cohort, the saturation curve and the probe. §2 is marked
> where its number lands.

## 1. Question and design

Can full-FT Wan2.2 TI2V 5B **memorize** the 1,629 windows of 100 successful DROID trajectories, conditioned
on a per-trajectory language instruction (T5) plus the first-frame latent — reaching near-perfect
reconstruction? Framed as a finite-set memorization test with text-ablation controls (plan-review F1), not
a "text determines the future" claim. Success rule D11 is two-tier: **canonical** ≥90% of the 100-window
cohort at m_corr ≥ 0.95 (median over seeds 0–2, correct instruction); **full-set** ≥90% of all 1,629
windows at seed-0 SSIM ≥ 0.90 at c\*.

## 2. Verdict — not established, at either training budget

exp_02 reports **two internally-consistent verdicts, one per eval-code generation.** This is not a
bookkeeping quirk: the verdict CLI *refused* to build a single verdict mixing the step-2500 artifacts (eval
commit `e27fdc3`) with the 10k-generation artifacts (`46c5f41`) — "one verdict must be built from one run's
passes (same manifest, dataset, and eval commit)". The guard is correct and I did not work around it.

**Verdict A — step 2,500 (eval `e27fdc3`), both tiers evaluable:**

| Tier | Rule | Measured | Verdict |
| --- | --- | --- | --- |
| Canonical | ≥90% of 100 at m_corr ≥ 0.95 | 0/100 (0.0%); mean 0.8133, max 0.9461 | **not established** |
| Full-set | ≥90% of 1,629 at seed-0 ≥ 0.90 | 105/1,629 (6.4%); mean 0.7984, max 0.9480 | **not established** |

**Verdict B — step 10,000 (eval `46c5f41`), c\*=10000, denominator 100, coverage complete:**

| Tier | Rule | Measured | Verdict |
| --- | --- | --- | --- |
| Canonical | ≥90% of 100 at m_corr ≥ 0.95 | **0/100 (0.0%)**; mean 0.8414, max 0.9484 | **not established** |
| Full-set | ≥90% of 1,629 at seed-0 ≥ 0.90 | *(pass in flight)* | pending |

**No window in either cohort has reached 0.95 at any checkpoint, at any training budget.** The shortfall is
*uniform* — a broad distribution short of the bar, not a memorized subset plus stragglers.

## 3. Reliability audit — do I believe these numbers?

Yes, and with more confidence than v1 had. Grounds:

- **Fail-closed chain held end-to-end, and demonstrably bites.** Manifest-derived fixed cohorts (denominator
  100 / 1,629 — never pass-derived), role grids validated on exact (window, checkpoint, seed, mode) tuples
  (`role_validation.ok=true`, coverage complete on every admitted pass), manifest binding `c02a67be…`, immutable
  role-keyed paths. The cross-generation refusal in §2 is the guard catching a real mistake I was about to make.
- **Determinism.** Per-rollout RNG is `window_fold_key(seed, episode_id, window_start)` — order-independent,
  which the probe verified *bitwise*: its 25-step arm reproduced the landed segment-final rows exactly
  (mean 0.8100125855, matching the reviewer's independently computed anchor to 10 decimal places). This is an
  unusually strong end-to-end reproducibility check and it passed perfectly.
- **Data integrity.** Dataset published under `_SUCCESS` with byte-verify at training preflight; V1
  encode-parity r≈0.995; V4 frame-0 future-invariance bitwise-exact; V3 decode ceilings 0.902–0.966.
- **Ceilings now measured (v1's open caveat closed).** All 10k-generation passes carry aux 100%: VAE
  round-trip ceiling mean **0.9493** (min 0.8812). Note what this implies — the encode/decode round trip
  *itself* sits at ≈0.95 against raw RGB. It does not bound our decoded-vs-decoded metric mathematically,
  but it does mean the D11 bar was set at roughly the fidelity of the VAE's own reconstruction.
- **Cross-scale and cross-experiment consistency.** exp_01 (full-corpus FT, different code path) at val loss
  ≈0.176–0.184 gave rollout SSIM 0.727/0.787; exp_02's S2 (10 episodes) at loss 0.061 gave 0.890; exp_02's S3
  (100 episodes) at loss ≈0.12 gives 0.841. Three independent runs trace one coherent loss→rollout-SSIM
  relation, which argues strongly against a pipeline artifact.
- **Seed noise is not in play.** The canonical tier medians three seeds; the full-set fraction (6.4% vs a
  required 90%) is orders of magnitude outside any plausible seed effect.

## 4. What the numbers say

**Canonical-cohort trajectory (seed-0, correct instruction):**

| step | 250 | 500 | 1000 | 1750 | 2500 | 5000 | 7500 | 10000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean SSIM | 0.7580 | 0.7707 | 0.7892 | 0.8020 | 0.8133 | 0.8320 | 0.8377 | 0.8414 |

Gains per 2,500 steps after step 2500: **+0.0187 → +0.0057 → +0.0037** (a 5× decay). Train loss over the
same span: 0.145 → 0.132 → 0.127 → ≈0.12. The 4× training extension bought ~0.028 SSIM and ~0.025 loss.
Extrapolating the decaying gain, the asymptote sits **below 0.90** — the 0.95 bar is not reachable by more
of this training, and per-window ≥90%-at-0.95 is stricter still.

**The per-window ceiling is frozen.** Best window: 0.9461 (2500) → 0.9472 (5000) → 0.9475 (7500) → 0.9484
(10000). The easiest window in the cohort gained 0.002 across a 4× extension.

**Distribution at 10k:** ≥0.95: 0; ≥0.90: 17; ≥0.85: 46; ≥0.80: 78 of 100. The mass moved up but the top
never crossed.

## 5. Why 0.95 is not reached — now resolved by elimination

v1 listed four hypotheses and could not separate H1 from H2. Both discriminators have now run.

- **H1 — undertrained (dose): tested and insufficient.** The Yixun-approved extension to 10k is the direct
  test. Training 4× longer produced a 5× *decay* in the improvement rate and left the best window static.
  Dose is not the binding constraint.
- **H2a — sampling-side mismatch: EXCLUDED.** The approved probe re-rolled 30 canonical windows at 25/50/100
  integration steps on the step-2500 checkpoint. More steps were **strictly worse**: 50 steps −0.0074 mean
  with **0/30 windows improved**; 100 steps −0.0121, again 0/30. Discretization error is not the limiter at
  25 steps; velocity-field error compounding dominates. A corollary discovered while building the probe: the
  OVERFIT100 rollout has **no CFG branch at all** and training used none, so the suspected train/eval
  guidance mismatch does not exist — that hypothesis was void, not merely untested.
- **H2b — the training objective itself: the surviving explanation.** Training optimizes single-step
  denoising at randomly sampled timesteps; evaluation integrates 25 steps from noise with frame-0 pinning.
  Nothing in the objective penalizes compounding error along a trajectory, and the probe shows that
  compounding is exactly what caps the result. The S2 anchor is the sharpest form of this: at 10 episodes
  and loss 0.061 — memorization far deeper than S3 ever reached — rollout SSIM was still only 0.890.
- **H3 — capacity/interference: excluded.** 5B parameters on 100 trajectories, and the 10-episode run fell
  short at a *lower* loss. Capacity is not the limiter.
- **H4 — VAE ceiling: not a mathematical bound**, since SSIM compares decoded-pred against decoded-GT through
  the same VAE. But see §3: the measured round-trip ceiling of ≈0.949 means the bar was set at about the
  VAE's own fidelity, which makes 0.95 an unusually demanding target in practice.

**Conclusion of the discrimination: the recipe is the ceiling.** One-step denoising → 25-step rollout
saturates near mean 0.84 with a per-window ceiling around 0.95 on this cohort, independent of training dose
and sampler settings.

## 6. A positive finding: text conditioning strengthens with training

The ablation ordering held at every scale and checkpoint (correct > null > shuffled), and the **gap widens
as memorization deepens**:

| | correct | null (empty text) | shuffled (deranged) |
| --- | --- | --- | --- |
| S2 (10 ep, step 2500) | 0.8895 | 0.8398 (−0.050) | 0.8342 (−0.055) |
| S3 step 2500 | 0.8133 | 0.7992 (−0.014) | 0.7824 (−0.031) |
| S3 step 10000 | 0.8413 | 0.8020 (**−0.039**) | 0.7883 (**−0.053**) |

At 100 episodes the text gap roughly **tripled** between 2,500 and 10,000 steps. The model leans on the
instruction *more* as it memorizes more. Two things follow: the conditioning path is genuinely load-bearing
(not a nuisance variable), and the modest absolute gap is explained by the first-frame latent already
determining most of each frame — text disambiguates the future, it does not paint the picture.

## 7. Recommendation

**Do not extend training further.** The extension already bought its information: the asymptote is measured
and it is below the bar. Another 10k steps would cost real compute to move the mean by perhaps 0.005.

**exp_03 should change the recipe, the metric, or the claim — not the dose.** In rough order of expected
value:

1. **Attack the compounding directly.** Train on the rollout the eval actually runs — multi-step / scheduled
   sampling, or a short-horizon rollout loss on top of the denoising objective. This targets the one surviving
   hypothesis, and the probe result (more steps = worse) is direct evidence that trajectory error, not
   per-step error, is what is unbounded.
2. **Re-set the success metric to something the recipe can express.** SSIM ≥ 0.95 on decoded video is close to
   the VAE's own round-trip fidelity (0.949 measured). A latent-space reconstruction criterion, or a
   perceptual metric with a calibrated bar, would measure memorization rather than the codec.
3. **Cheap and worth having:** a *latent*-MSE trajectory across the same checkpoints from the artifacts already
   on GCS, to confirm the saturation is in the model rather than in the pixel metric.

**For the adapter program specifically**, the useful bound is this: a *fully unfrozen* 5B backbone, given 100
trajectories and 390+ epochs, reconstructs held-in video to ≈0.84 mean rollout SSIM under this recipe. Frozen-
backbone adapters should not be expected to beat that under the same objective — so an adapter result near
0.8 is at the recipe's ceiling, not evidence of a weak adapter. That reframes how every prior adapter number
in this repo should be read.

## 8. Standing state

- All exp_02 code closed and reviewed: suite **1,236 passed / 2 skipped**, ~24 Codex review passes, every
  finding fixed or explicitly ruled on the record. Notable: the eval-resume series took 5 passes to APPROVE
  and proved itself in production (the 10k canonical pass survived an infra kill and finished from staged
  rows).
- Artifacts: all aggregations, both verdict JSONs, the probe artifact, and the HTML report generator are
  committed under `overfit100_s3_artifacts/` and `overfit100_01_memorization_trajectory_results.html`.
- Checkpoints {250, 500, 1000, 1750, 2500, 5000, 7500, 10000} retained on GCS.
- Open, optional: the S2 ceiling backfill (aux only, checkpoint-independent); a latent-MSE trajectory (§7.3).
- Open decisions for Yixun: **merge or leave unmerged** (SOP: merge only on confirmed success — the formal
  answer here is *not established*, so the default is to leave the code on its branch with docs on
  `yixun-dev`), and the **exp_03 direction** per §7.
