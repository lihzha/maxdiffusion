# Codex plan review — exp_03 `rollout_objective` plan v1

- **Date:** 2026-08-02 | **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only
- **Verdict:** REQUEST-REVISION — 2 BLOCKER (P2 estimand mislabeled compute-matched + C's alternation ≠ the literal λ-loss; P3 metric/role mismatch: m_corr needs 3 seeds, s3_intermediate has 1), 5 MAJOR (P1 B's un-normalized horizon weighting [A's corrective-target math VERIFIED CORRECT]; P4 subclass binding + mesh-fit gap; P5 sampler extraction touches proven eval code; P6 A's label confound + C's fidelity; P7 missing no-update discriminator probe + sigma-trajectory metric).

## Reviewer output (verbatim)

```
REQUEST-REVISION

Context loaded: SOP; Query 1; plan v1; exp_02 analysis v4, results D1–D3, and Codex review; overfit100 trainer; rollout implementation; FlaxFlowMatch scheduler.

1. **P1 — MAJOR —** Plan text at issue: “`v* = (z_{σ_lo} − z_gt)/σ_lo`” and B’s raw endpoint MSE. A’s target is mathematically correct for the actual Euler rule `z_next=z+(σ_next−σ)v`: it yields `z_next−z_gt=(σ_next/σ)(z−z_gt)` and reduces exactly to `ε−z_gt` on-path; flow shifting only changes the sigma grid. Explicitly exclude only terminal `σ_lo=0`, use the shifted sigma in FP32, and test every valid unequal interval—the smallest positive configured sigma is ≈0.1724, so no clamp is needed. B’s same-ε target is also the exact ideal trajectory and gives zero loss for `v=ε−z_gt`; however, raw endpoint MSE weights starts by `(σ_hi−σ_lo)^2`, severely distorting loss/gradient scale across the nonuniform grid. Normalize by the squared horizon or predeclare equivalent weighting, and test zero-at-optimum, pin masking, and deterministic eval-dtype rounding.

2. **P2 — BLOCKER —** Plan text at issue: “at matched additional compute” and “expectation-equivalent.” Equal +2,500 optimizer steps matches examples/updates, not compute: A adds stop-grad forwards, B differentiates two forwards, and C gives only about 1,250 updates to each component. Retain this as an update-matched estimand but stop calling it compute-matched; report forward/backward counts, TPU-hours, and a compute-normalized comparison. Predeclare identical checkpoint bytes, optimizer state, data order, and purpose-folded RNGs keyed by global step so common ε/start indices remain matched and schedules survive resume. For C, use exact balanced counts and acknowledge that alternation is only an unbiased stochastic-gradient estimator—not an exact weighted-loss trajectory under Adam.

3. **P3 — BLOCKER —** Plan text at issue: “canonical-100 mean `m_corr`” evaluated with `s3_intermediate`, plus “D1 decay slope … shrinks by ≥25%.” `s3_intermediate` contains seed-0 correct-mode SSIM; `m_corr` requires seeds `{0,1,2}`. Either rename the primary metric to canonical seed-0 mean SSIM or run `s3_segment_final` and compute actual `m_corr`. The +0.02 bar is defensible as a practical-effect gate—roughly five times the expected ≈0.0042 control gain—but is not calibrated to training-seed noise; add paired episode-bootstrap intervals and remove “far outside noise.” Define D1 over all 100 canonical windows as per-window OLS on frames 1–32, specify the reduction formula and CI, and ensure videos are emitted. Predeclare the existing all-1,629-window fixed-RNG plain-loss reading, including reproduction of the 10k anchor 0.12227 and deviation from the old loss→SSIM line.

4. **P4 — MAJOR —** Plan text at issue: “subclass of the exp_02 trainer,” “S1 … tiny batch,” and unspecified `p_ss` ramp. The parent binds module-level loss/train functions, so the new class must explicitly replace `start_training` or introduce a tested binding hook; merely subclassing cannot select the new objectives. Keep the new `EXP03_TI2V` dispatch for provenance, reuse the identical `Overfit100TrainState` and Orbax item shapes, and require an actual 10k-checkpoint restore/optimizer roundtrip. Specify `p_ss_max`, ramp length, k distribution, valid start-index distribution, and segment/resume semantics before approval. A v6e-8 tiny smoke does not certify scan/remat compilation and GBS 256 on the v6e-64 mesh; add a user-gated one-step target-mesh fit probe before S2.

5. **P5 — MAJOR —** Plan text at issue: “eval stack runs … unchanged” while extracting the sampler step from `generate_wan_side_adapter.py`. New RUN_NAMEs are accepted without role changes, and the existing launcher can restore the unchanged checkpoint layout using its `OVERFIT100_TI2V` eval config; however, sampler extraction is an eval-code change. Characterize the old operator before extraction, reproduce the landed 30-window seed-0 SSIM scalars exactly afterward, and rerun the shared control under the same post-extraction eval commit in a fresh immutable output root. Keep every verdict at 25 steps and emit D1 videos on the first pass; no trial-specific two-step evaluation should enter comparison claims.

6. **P6 — MAJOR —** Plan text at issue: A’s “corrective target” and C’s “implemented as stochastic batch alternation.” B faithfully operationalizes a short-horizon deterministic rollout loss, but A combines scheduled exposure with a new corrective label, so an A win would not isolate scheduled sampling; add a same-ε-label scheduled-sampling discriminator or explicitly rename A as “corrective scheduled sampling.” C is not literally `λL_A+(1−λ)L_B` per update and finite-run Adam breaks expectation equivalence; implement the true weighted same-batch gradient, or obtain approval for deterministic balanced alternation and narrow the interpretation—especially “C wins ⇒ complementary.”

7. **P7 — MAJOR —** Plan text at issue: proceeding from generic S1 smoke directly to all three S2 runs. Before v6e-64, add a no-update v6e-8 discriminator at the 10k checkpoint measuring raw/normalized A/B losses, gradient norms and cosine versus the plain objective, A with same-ε versus corrective targets, and `p_ss=0` parity through the new trainer. Also record fixed-ε latent error against the ideal interpolant at every sigma step for control and trials; D1 measures temporal video-frame decay, not denoising-sigma trajectory divergence, so it cannot alone establish that sampler compounding was reduced.

Single biggest risk to interpretability: the experiment could claim reduced denoising-step compounding from a temporal-frame D1 improvement while A simultaneously changes both exposure and supervision and C does not implement the literal combined loss.
tokens used
169,276
REQUEST-REVISION

Context loaded: SOP; Query 1; plan v1; exp_02 analysis v4, results D1–D3, and Codex review; overfit100 trainer; rollout implementation; FlaxFlowMatch scheduler.

1. **P1 — MAJOR —** Plan text at issue: “`v* = (z_{σ_lo} − z_gt)/σ_lo`” and B’s raw endpoint MSE. A’s target is mathematically correct for the actual Euler rule `z_next=z+(σ_next−σ)v`: it yields `z_next−z_gt=(σ_next/σ)(z−z_gt)` and reduces exactly to `ε−z_gt` on-path; flow shifting only changes the sigma grid. Explicitly exclude only terminal `σ_lo=0`, use the shifted sigma in FP32, and test every valid unequal interval—the smallest positive configured sigma is ≈0.1724, so no clamp is needed. B’s same-ε target is also the exact ideal trajectory and gives zero loss for `v=ε−z_gt`; however, raw endpoint MSE weights starts by `(σ_hi−σ_lo)^2`, severely distorting loss/gradient scale across the nonuniform grid. Normalize by the squared horizon or predeclare equivalent weighting, and test zero-at-optimum, pin masking, and deterministic eval-dtype rounding.

2. **P2 — BLOCKER —** Plan text at issue: “at matched additional compute” and “expectation-equivalent.” Equal +2,500 optimizer steps matches examples/updates, not compute: A adds stop-grad forwards, B differentiates two forwards, and C gives only about 1,250 updates to each component. Retain this as an update-matched estimand but stop calling it compute-matched; report forward/backward counts, TPU-hours, and a compute-normalized comparison. Predeclare identical checkpoint bytes, optimizer state, data order, and purpose-folded RNGs keyed by global step so common ε/start indices remain matched and schedules survive resume. For C, use exact balanced counts and acknowledge that alternation is only an unbiased stochastic-gradient estimator—not an exact weighted-loss trajectory under Adam.

3. **P3 — BLOCKER —** Plan text at issue: “canonical-100 mean `m_corr`” evaluated with `s3_intermediate`, plus “D1 decay slope … shrinks by ≥25%.” `s3_intermediate` contains seed-0 correct-mode SSIM; `m_corr` requires seeds `{0,1,2}`. Either rename the primary metric to canonical seed-0 mean SSIM or run `s3_segment_final` and compute actual `m_corr`. The +0.02 bar is defensible as a practical-effect gate—roughly five times the expected ≈0.0042 control gain—but is not calibrated to training-seed noise; add paired episode-bootstrap intervals and remove “far outside noise.” Define D1 over all 100 canonical windows as per-window OLS on frames 1–32, specify the reduction formula and CI, and ensure videos are emitted. Predeclare the existing all-1,629-window fixed-RNG plain-loss reading, including reproduction of the 10k anchor 0.12227 and deviation from the old loss→SSIM line.

4. **P4 — MAJOR —** Plan text at issue: “subclass of the exp_02 trainer,” “S1 … tiny batch,” and unspecified `p_ss` ramp. The parent binds module-level loss/train functions, so the new class must explicitly replace `start_training` or introduce a tested binding hook; merely subclassing cannot select the new objectives. Keep the new `EXP03_TI2V` dispatch for provenance, reuse the identical `Overfit100TrainState` and Orbax item shapes, and require an actual 10k-checkpoint restore/optimizer roundtrip. Specify `p_ss_max`, ramp length, k distribution, valid start-index distribution, and segment/resume semantics before approval. A v6e-8 tiny smoke does not certify scan/remat compilation and GBS 256 on the v6e-64 mesh; add a user-gated one-step target-mesh fit probe before S2.

5. **P5 — MAJOR —** Plan text at issue: “eval stack runs … unchanged” while extracting the sampler step from `generate_wan_side_adapter.py`. New RUN_NAMEs are accepted without role changes, and the existing launcher can restore the unchanged checkpoint layout using its `OVERFIT100_TI2V` eval config; however, sampler extraction is an eval-code change. Characterize the old operator before extraction, reproduce the landed 30-window seed-0 SSIM scalars exactly afterward, and rerun the shared control under the same post-extraction eval commit in a fresh immutable output root. Keep every verdict at 25 steps and emit D1 videos on the first pass; no trial-specific two-step evaluation should enter comparison claims.

6. **P6 — MAJOR —** Plan text at issue: A’s “corrective target” and C’s “implemented as stochastic batch alternation.” B faithfully operationalizes a short-horizon deterministic rollout loss, but A combines scheduled exposure with a new corrective label, so an A win would not isolate scheduled sampling; add a same-ε-label scheduled-sampling discriminator or explicitly rename A as “corrective scheduled sampling.” C is not literally `λL_A+(1−λ)L_B` per update and finite-run Adam breaks expectation equivalence; implement the true weighted same-batch gradient, or obtain approval for deterministic balanced alternation and narrow the interpretation—especially “C wins ⇒ complementary.”

7. **P7 — MAJOR —** Plan text at issue: proceeding from generic S1 smoke directly to all three S2 runs. Before v6e-64, add a no-update v6e-8 discriminator at the 10k checkpoint measuring raw/normalized A/B losses, gradient norms and cosine versus the plain objective, A with same-ε versus corrective targets, and `p_ss=0` parity through the new trainer. Also record fixed-ε latent error against the ideal interpolant at every sigma step for control and trials; D1 measures temporal video-frame decay, not denoising-sigma trajectory divergence, so it cannot alone establish that sampler compounding was reduced.

Single biggest risk to interpretability: the experiment could claim reduced denoising-step compounding from a temporal-frame D1 improvement while A simultaneously changes both exposure and supervision and C does not implement the literal combined loss.
```

## Resolution record (Planner, 2026-08-02) — plan v2

All 7 findings **adopted**; none rejected. Plan v2 changes:

| # | Resolution |
| --- | --- |
| P1 | A's target kept (reviewer verified exact under the Euler rule; terminal-σ excluded; FP32 σ; min grid σ ≈0.1724 ⇒ no clamp). B's endpoint MSE now **horizon-normalized** by (σ_hi−σ_lo)²; zero-at-optimum, masking parity, dtype tests predeclared. |
| P2 | Estimand renamed **update-matched**; forward/backward counts + TPU-hours recorded; compute-normalized reading reported alongside. Identical checkpoint bytes / opt state / data order predeclared; **purpose-folded RNGs keyed by global step** (resume-stable, cross-arm aligned). |
| P3 | Primary metric renamed **canonical seed-0 mean SSIM** (s3_intermediate); m_corr deferred to S3 s3_segment_final. +0.02 gate kept as practical-effect gate with **paired per-episode bootstrap CI**; "far outside noise" removed. D1 redefined: all 100 windows, per-window OLS frames 1→32, reduction formula + CI, WRITE_VIDEOS=True on trial evals and a control re-eval. Loss-instrument reading per trial predeclared incl. the 0.12227 anchor and deviation-from-line. |
| P4 | Explicit tested **binding hook** (parent refactored to route through `_loss_and_step_fns()`; p_ss=0 parity test ≡ exp_02 step). Same TrainState/Orbax shapes + restore-roundtrip test. p_ss_max=0.5, ramp 500 steps keyed to global step −10,000, k_A~U{1,2}, non-terminal start-index distribution — all specified. **S1.6 one-step v6e-64 mesh-fit probe** added before S2. |
| P5 | Extraction gated **bitwise**: landed 30-window seed-0 scalars must reproduce exactly post-extraction; control re-evaled at the post-extraction commit into a fresh root; **one-generation rule** for every comparison; verdicts stay at 25 steps. |
| P6 | A renamed **corrective scheduled sampling**; label confound quantified in S1.5 (same-ε vs corrective label on identical states); optional pure-SS arm A′ predeclared if A wins. **C reimplemented as the literal weighted same-batch loss** λ·L_A+(1−λ)·L_B (single Adam update; ~2.5–3× step time; memory = B's peak) — no alternation, no expectation caveat. |
| P7 | **S1.5 no-update discriminator probe** added (losses, grad norms, grad cosine vs plain, label isolation, p_ss=0 parity). **Sigma-trajectory metric** added as Mechanism B (fixed-ε latent error vs ideal interpolant per sigma step) — D1 alone cannot attribute reduced sampler compounding. |

Reviewer's headline interpretability risk (temporal-D1-only attribution + A's dual change + C's non-literal loss) is addressed by exactly these three: the sigma-trajectory metric, S1.5 label isolation, and C-literal.

## Re-review (plan v2) — verdict and v2.1 closures

**Verdict: REQUEST-REVISION** — P1/P5/P6 RESOLVED (+ P2's core: update-matched estimand and literal C);
four residuals, each closed in **plan v2.1**:

| Residual | v2.1 closure |
| --- | --- |
| P2: resume-stable data order (iterator reseeded with `seed+start_step`; preempted arms could diverge) | **Stability by construction:** no intermediate saves in (10000, 12500) ⇒ any preemption restarts the whole segment from 10,000 with the same iterator seed ⇒ identical order across arms and retries. Predeclared; 39-min re-run cost accepted. |
| P3: D1 script computes aggregates, not the declared per-window OLS/CI | New predeclared script `diagnostics/d1_per_frame_slopes.py` (OLS frames 1→32, reduction formula, 10k-resample paired bootstrap, 95% CI, seed 0, synthetic-slope unit tests). |
| P4: index supports underspecified | Exact supports: A draws `k_A~U{1,2}` then `hi~U{i: k_A≤i≤24}`, `lo=hi−k_A`, teacher-forced branch shares the draw; B: `hi~U{2..24}`, consecutive path; C draws both supports per batch. |
| P7: sigma-trajectory metric not operationalized | New probe module `diagnostics/sigma_trajectory_trace.py` on the extracted step fn; 30-window probe cohort; eval's own ε keying; immutable per-checkpoint JSON under the probe root; oracle-zero unit test. Eval rollout untouched. |

**Approval-time considerations (reviewer's, to be surfaced to Yixun verbatim):** (1) C costs ~2.5–3×
baseline per update — S1/S1.6 measure the real budget before S2 approval; (2) the paired episode bootstrap
covers cohort uncertainty, not training-seed variability; (3) A is a compound exposure+label intervention —
scheduled sampling alone requires A′; (4) gain-per-TPU-hour is descriptive efficiency, not a compute-matched
causal comparison.

## Closing pass (plan v2.1) and v2.2 fix

Closures (1) data-order-by-construction, (2) D1 slope script, (4) sigma trace: **SOUND**. Closure (3)
**UNSOUND** — index-direction inversion: the σ grid descends (index 0 = highest σ; the eval advances
`i → i+1`), so v2.1's `lo = hi − k_A` walked the sampler backward toward HIGHER σ. **v2.2** rewrites the
supports in the eval's direction: A: `k_A~U{1,2}`, start `s~U{0..24−k_A}`, end `e = s+k_A`; B: `s~U{0..22}`,
path `s→s+1→s+2`; σ_hi = σ[s] > σ_lo = σ[e], terminal never reached since e ≤ 24.

## Final verdict (plan v2.2): **APPROVE-PLAN**

"The supports correctly follow the descending 25-point positive-sigma grid in the eval's `i → i+1`
direction, with A ending at `e ≤ 24`, B ending at `24` at most, and neither reaching terminal index `25`."

Plan cycle closed: v1 → review (2 BLOCKER + 5 MAJOR) → v2 (all adopted) → re-review (4 residuals) → v2.1 →
closing pass (1 UNSOUND: index direction) → v2.2 → **APPROVE-PLAN**. Surfaced to Yixun for approval with
this full record.

## Delta review (Tier 2, plan v3) — REQUEST-REVISION → v3.1 adoptions

All 5 findings adopted verbatim: (1) Tier 2 reframed as an early from-init screen (no optimum claims; null
scoped to the budget; training-loss vs instrument attribution fixed); (2) ctrl0 guard operationalized —
exp_02's exact RNG stream preserved (aux folded keys only for new-objective draws) + hard AND-gate
(|Δloss| ≤ 1e-4, |ΔSSIM_mean| ≤ 5e-4, max_window |ΔSSIM| ≤ 1e-3) at every checkpoint against full-precision
anchors + bitwise-certificate bridge (else re-evaluate exp_02 references); (3) confounds handled —
"training-history package" wording, restart-from-init preemption rule, S1.5/S1.6 from both states, one eval
generation; (4) ctrl0-first strengthened to training+instrument+SSIM+gates before A0/B0/C0, scope = Tier 2
only; (5) cost corrected to 5.3 h v6e-64 (+6.9 h as contingency label), eval count 6+4.

## Final verdict (plan v3.1): **APPROVE-PLAN**

All five Tier-2 adoptions verified (early-screen framing; RNG-stream preservation + AND-gate guard;
continuity controls; strong ctrl0-first; corrected costs). Full cycle: v1 → 7 findings → v2 → 4 residuals →
v2.1 → index fix → v2.2 APPROVE → Query 2 → v3 → 5 delta findings → v3.1 → **APPROVE-PLAN**. Surfaced to
Yixun for the approval that gates all implementation and launches.
