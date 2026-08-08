# exp_03 `rollout_objective` — Analysis (Planner)

**v1, written 2026-08-08** — synthesizing Results 1–9 (`rollout_objective_results.md`; every number
artifact-backed, provenance in `_command.md`). Instrument pedigree: ctrl0's AND-gate replication of
exp_02 at ~1e-11; the loss→SSIM law (SSIM ≈ 0.9885 − 1.201·loss) at **9 independent holds**, the
last with residual +0.00001; exact fixed-RNG anchors reproduced at every checkpoint handoff.

## 1. The question, and the plain answer

**Question (Yixun's opening directive):** do trial losses that train on the trajectory the
evaluation runs — A (corrective scheduled sampling), B (short-horizon rollout), C (λ·A+(1−λ)·B) —
beat one-step denoising, from a trained state (Tier 1) and from scratch (Tier 2)?

**Answer: yes, decisively in statistics; not yet at the predeclared practical bar.** Every trial
arm beats its control somewhere; the best single results are **A at the trained state** (+0.0079
SSIM at 17,500, t=+30.8, 100/100 windows, and it wins the one-step metric simultaneously) and
**C₀.₅ from init** (+0.0063, t=+14.2, 95/100). No arm reaches the +0.02 practical-effect gate at
tested budgets (best ≈ 40% of it, widening).

## 2. The mechanism, quantified

The law separates every result into two orthogonal components:

1. **Off-law gain** — rollout SSIM above what one-step loss predicts. Every trial arm has it
   (+0.0046…+0.0144); no one-step-trained checkpoint ever did (9 law holds within ±0.005, control
   residuals −0.0002 and +0.00001 in this experiment). It rises monotonically with the A-share in
   the λ-sweep: A's corrective label is the stronger rollout signal per unit weight.
2. **One-step-loss cost** — what the trial objective surrenders on the shared metric. This is
   where the regimes differ, and it is U-shaped in λ (min at 0.5: +0.0039 vs ~+0.01 for either
   pure objective) — direct evidence that the two objectives' one-step errors partially cancel
   when mixed (gradient near-orthogonality: S1.5 measured cos(A,B)=0.459 at init, 0.165 at 10k).

Net advantage = (1) − law-priced (2). This single accounting reproduces every measured ordering:
- **From init:** costs are large (~+0.01), so cancellation dominates → C₀.₅ wins (+0.0063), pure
  arms ≈ 0 (A0 +0.0011, B0 −0.0015).
- **From the trained state:** costs vanish (A's is NEGATIVE — it beats the control on the
  control's own objective, −0.00364 vs −0.00224), so there is nothing to cancel → pure A wins
  (+0.0074 → +0.0079) and blending only dilutes it (C +0.0056 ≈ B +0.0059).

## 3. What is established vs. open

**Established (this design, one seed, canonical-100):** the objective/rollout decoupling exp_02
hypothesized is real and reproducible across two starting states and five λ values; A improves
both metrics simultaneously at the trained state and its advantage grows over 7,500 updates
(+0.0059 → +0.0079, off-law residual growing while the control stays law-bound); the trained-state
regime — precisely where exp_02 stalled — is where these objectives are cheapest and most profitable.

**Open / not established:** the +0.02 gate (untested beyond 17,500; linear extrapolation says
~15k more updates, but everything in this system saturates); seed robustness (n=1 seed
everywhere); horizon/k sensitivity (k∈{1,2} for A, k=2 for B only); the state-dependent-λ
recipe (§2 implies blend-early-anneal-to-A; untested); transfer beyond OVERFIT100 memorization
to generalization — the setting the adapter program actually cares about.

## 4. Recommendation

The natural next increments, in order of information-per-chip-hour: (1) a **second seed** of
Tier-1 A and C₀.₅-from-init (the two headline claims; kills the n=1 caveat); (2) **A continued to
~25,000** against the extended control (does the gap keep widening or saturate — the gate question
settled empirically instead of by extrapolation); (3) the **λ-schedule arm** (blend→anneal) only
if (1)–(2) hold up. exp_03's machinery (certified controls, law, anchors, watcher) makes each of
these a routine launch.
