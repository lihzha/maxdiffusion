# exp_03 `rollout_objective` — Analysis (Planner)

> **CLOSED 2026-08-08 by Yixun (Query 8).** His verdict: from scratch, C (λ=0.5) works better than
> the A and B methods. The +0.02 practical-effect hard gate was NOT met (his adjudication, per
> announcement 03). Unmerged per SOP; recommendation ladder parked.

**v1.1, written 2026-08-08** — v1 revised against the Codex analysis review
(SOUND-WITH-REVISIONS, 4 MAJOR / 2 MODERATE, all accepted; record in
`rollout_objective_codex_analysis_review.md`). The review caught an arm-mixing arithmetic error in
the headline growth claim, recomputed every quoted comparison (all others exact), and re-fit the
law from the raw exp_02 means (0.98895405 − 1.20004257·loss — the published rounded coefficients
0.9885/1.201 shift fifth-decimal residuals; e.g. control@17,500 is −0.00055 under the exact fit,
not +0.00001). Five-decimal residual claims are hereby downgraded to the exact-fit basis.

**v1, written 2026-08-08** — synthesizing Results 1–9 (`rollout_objective_results.md`; every number
artifact-backed, provenance in `_command.md`). Instrument pedigree: ctrl0's AND-gate replication of
exp_02 at ~1e-11; the loss→SSIM law (exact fit 0.98895405 − 1.20004257·loss) at **9 held-out evaluations** (NOT
statistically independent — checkpoints share ancestry and data, per exp_02 v5.1's identical
correction; an in-range mean predictor, no causal reading of the slope); exact fixed-RNG anchors reproduced at every checkpoint handoff.

## 1. The question, and the plain answer

**Question (Yixun's opening directive):** do trial losses that train on the trajectory the
evaluation runs — A (corrective scheduled sampling), B (short-horizon rollout), C (λ·A+(1−λ)·B) —
beat one-step denoising, from a trained state (Tier 1) and from scratch (Tier 2)?

**Answer (one training seed, one rollout seed, canonical-100 — every statistic below is
conditional on that): yes, decisively within-seed; not yet at the predeclared practical bar.**
The t-tests measure per-episode consistency conditional on one trained model; they say nothing
about training-seed variability. (Deviation note: the plan predeclared paired-bootstrap CIs; t
statistics were used instead — flagged, not hidden.) Every trial
arm beats its control somewhere; the best single results are **A at the trained state** (+0.0079
SSIM at 17,500, t=+30.8, 100/100 windows, and it wins the one-step metric simultaneously) and
**C₀.₅ from init** (+0.0063, t=+14.2, 95/100). No arm reaches the +0.02 practical-effect gate at
tested budgets (best ≈ 40% of it, widening).

## 2. The mechanism, quantified

The law provides a DESCRIPTIVE decomposition (a reference counterfactual, not a mechanism):

1. **Off-law gain** — rollout SSIM above what one-step loss predicts. Every trial arm has it
   (+0.0046…+0.0144); no one-step-trained checkpoint ever did (9 law holds within ±0.005, control
   residuals −0.0002 and +0.00001 in this experiment). It rises monotonically with the A-share in
   the λ-sweep: A's corrective label is the stronger rollout signal per unit weight.
2. **One-step-loss cost** — what the trial objective surrenders on the shared metric. This is
   where the regimes differ, and it is U-shaped in λ (min at 0.5: +0.0039 vs ~+0.01 for either
   pure objective) — a nonlinear mixture benefit CONSISTENT WITH error cancellation — hypothesis, not established:
   the U-shape is five endpoint cells, the cosines are instantaneous (and cosine already FAILED to
   predict endpoint cost, Results 3 correction); the predeclared post-training D1/sigma-trace
   comparisons that would test the story have not been run.

Net advantage = (1) − law-priced (2). This bookkeeping is CONSISTENT with every measured ordering (it is near-algebraic once both
residuals are included — consistency, not independent explanatory evidence):
- **From init:** costs are large (~+0.01), so cancellation dominates → C₀.₅ wins (+0.0063), pure
  arms ≈ 0 (A0 +0.0011, B0 −0.0015).
- **From the trained state:** costs vanish (A's is NEGATIVE — it beats the control on the
  control's own objective, −0.00364 vs −0.00224), so there is nothing to cancel → pure A wins
  (+0.0074 → +0.0079) and blending only dilutes it (C +0.0056 ≈ B +0.0059).

## 3. What is established vs. open

**Established (this design, one seed, canonical-100):** the objective/rollout decoupling exp_02
hypothesized is real and reproducible across two starting states and five λ values; A improves
both metrics simultaneously at the trained state and its advantage PERSISTED through 17,500
(+0.0074 → +0.0079 — a +0.0005 widening over 5,000 updates, too small and too few points to call a
trend; the earlier "grows" phrasing arm-mixed B's 12,500 number and is corrected); the trained-state
regime — precisely where exp_02 stalled — is where these objectives are cheapest and most profitable.

**Open / not established:** the +0.02 gate (untested beyond 17,500; linear extrapolation says
~15k more updates, but everything in this system saturates); seed robustness (n=1 seed
everywhere); horizon/k sensitivity (k∈{1,2} for A, k=2 for B only); the state-dependent-λ
recipe (§2 implies blend-early-anneal-to-A; untested); transfer beyond OVERFIT100 memorization
to generalization — the setting the adapter program actually cares about.

## 4. Recommendation

The natural next increments, in order of information-per-chip-hour: (1) an **independent-training-seed replication** of Tier-1 A and C₀.₅-from-init WITH matched
controls (confirmatory — n=2 is not robust seed inference, but it is the single highest-value
check); (2) **A continued to ~25,000** against the extended control, framed as a SATURATION PROBE (the
measured widening is +0.0005/5k — no extrapolation to the gate is defensible); (3) the **λ-schedule arm** (blend→anneal) only if (1)–(2) hold up AND the cancellation story
survives a post-training D1/sigma-trace test — until then it is an optimization hypothesis. exp_03's machinery (certified controls, law, anchors, watcher) makes each of
these a routine launch.
