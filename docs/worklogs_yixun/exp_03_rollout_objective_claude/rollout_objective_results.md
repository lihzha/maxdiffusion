# exp_03 `rollout_objective` — Results (Planner)

All numbers from committed GCS artifacts; every arm's provenance is in `_command.md`.

## RESULT 1 — Tier 2 (from init, 2,500 updates): B0 vs ctrl0 — 2026-08-07

Instrument = fixed-RNG one-step loss (n=1,629); SSIM = canonical seed-0 correct, 25-step rollout.
The control is CERTIFIED: ctrl0 reproduces exp_02 to ~1e-11 (AND-gate).

| | one-step loss @2,500 | SSIM @2,500 | law prediction | residual vs law |
| --- | --- | --- | --- | --- |
| **ctrl0** (one-step objective) | 0.1459820 | **0.813901** | 0.813176 | **+0.0007** |
| **B0** (short-horizon rollout) | 0.1549584 (+6.15%) | **0.812425** | 0.802395 | **+0.0100** |

Loss trajectory (B0 − ctrl0): +0.00169 @250, +0.00281 @1000, +0.00898 @2500 — B0 steadily gives up
one-step loss, as expected for an arm not trained on that objective.

**Reading — the mechanism is REAL but nets to zero at this budget.**
1. **B0 − ctrl0 SSIM = −0.0015.** The +0.02 practical-effect gate is **NOT met**; on the primary
   metric the two are a tie (0/100 windows ≥0.95 either way at 2,500 updates from init).
2. **But B0 breaks off the loss→SSIM law in the predicted direction.** exp_02's law (SSIM ≈ 0.9885 −
   1.201·loss) governed every one-step-trained checkpoint to ±0.005. ctrl0 sits on it (+0.0007).
   B0 sits **+0.0100 above it** — a 14× larger residual. The rollout objective *does* buy rollout
   quality that one-step loss cannot predict; that is exactly the hypothesis exp_03 was built to test.
3. **The two effects cancel.** B0 buys +0.0100 of off-law SSIM and pays 0.00898 of one-step loss,
   which the law prices at 1.201 × 0.00898 = **0.0108**. Net −0.0008 — within noise of the observed
   −0.0015. At 2,500 updates from init, the trade is break-even.
4. **What this does NOT settle:** whether the trade turns positive with more updates, from a trained
   state (Tier 1 asks exactly this), at a longer horizon (k=2 here), or for A/C. A single tied arm
   at one budget is not a verdict on the objective.

## RESULT 2 (partial) — Tier 1 (from 10k) B: seed validity + loss — 2026-08-07

| | loss @10,000 (anchor) | loss @12,500 | segment gain |
| --- | --- | --- | --- |
| control lr1e5c (one-step) | 0.1222672 | 0.1200277 | −0.00224 |
| **B (rollout)** | **0.1222672 — reproduces exp_02's anchor EXACTLY** | 0.1219373 | −0.00033 |

Seed validity CONFIRMED (identical 10,000 readings ⇒ B genuinely continued exp_02's step-10,000
state). B moves one-step loss 6.8× less than the control — again expected. **SSIM @12,500 pending.**

### GAP FOUND: the Tier-1 control has NO SSIM at 12,500

The Query-6 package asserted the lr1e5c control needed "no new launch" because it was already
instrumented — true for LOSS, false for SSIM: `wan-overfit100-s3ext-lr1e5c-20260802/` has
`validation_loss/` but no `validation/` pass. exp_02 only ever ran SSIM evals on the 2e-5 and 5e-5
arms. **Tier 1's primary metric therefore has no comparator yet**; closing it needs one v6e-8
canonical seed-0 SSIM eval at the control's step-12,500 checkpoint (which exists). That launch was
NOT in the approved package — it goes to Yixun.

## RESULT 2 (complete) — Tier 1 (from 10k, +2,500 updates): B — 2026-08-07

| | one-step loss @12,500 | SSIM @12,500 | law prediction | residual vs law |
| --- | --- | --- | --- | --- |
| start (exp_02 @10,000) | 0.1222672 | 0.8416 | 0.8417 | +0.0000 |
| control lr1e5c (one-step) | 0.1200277 | *0.8443 predicted, ±0.005* (measured PENDING) | 0.8443 | — |
| **B (rollout loss)** | 0.1219373 | **0.850115** | 0.8421 | **+0.0081** |

- **B gained +0.0085 SSIM over the segment while moving one-step loss only −0.00033** (the control
  moved it −0.00224, 6.8× more). B bought SSIM almost entirely OFF the law.
- **B beats the control's law-predicted SSIM by +0.0058** — larger than the law's ±0.005 residual
  band across six one-step checkpoints, so this is not obviously noise; it is NOT yet the
  predeclared comparison (measured control pending) and NOT the +0.02 gate.
- Windows ≥0.95: 1/100 (from 0/100 at the 10,000 start).

**Reading — Tier 1 points the opposite way from Tier 2, and that is the informative part.**
From init (Tier 2) the rollout objective's off-law gain (+0.0100) was exactly cancelled by the
one-step loss it surrendered (0.0108 law-priced) → net tie. **From a trained state (Tier 1) it
surrenders almost nothing (−0.00033 vs −0.00224) and keeps the off-law gain (+0.0081) → net
positive.** Mechanistically consistent with S1.5: at the trained state the trial gradients are far
from the control's (cos 0.228 for B) and support-draw noise is low (44% vs 77% at init), so the
rollout signal is both distinct and well-estimated exactly where it pays. **Caveat:** one arm, one
seed, one 2,500-update segment, and the control's SSIM is still a law prediction rather than a
measurement — the +0.02 practical-effect gate is NOT met either way.

## RESULT 3 — TIER 1 PRIMARY COMPARISON, MEASURED (B vs control @12,500) — 2026-08-07

| | one-step loss | **SSIM (measured)** | law prediction | residual | ≥0.95 |
| --- | --- | --- | --- | --- | --- |
| control lr1e5c (one-step) | 0.1200277 | **0.844169** | 0.8443 | −0.0002 | 0/100 |
| **B (short-horizon rollout)** | 0.1219373 | **0.850115** | 0.8421 | **+0.0081** | 1/100 |

**B − control = +0.005947.** Paired per-window: mean +0.00595, sd 0.0028, **se 0.00028, t = +21.0,
99/100 windows improved.**

1. **The effect is real and essentially uniform** — a 21-sigma paired difference with 99 of 100
   windows improving is not sampling noise, and its tightness (sd 0.0028 on a +0.0059 mean) says the
   rollout objective helps nearly every window a little rather than a few windows a lot.
2. **It does NOT meet the predeclared +0.02 practical-effect gate.** By the plan's own rule, B at
   2,500 Tier-1 updates is a **statistically decisive but practically small** improvement. The gate
   stands as written; this is a real effect of about a third the size deemed practically meaningful.
3. **The law survives its 7th test and explains the mechanism.** It predicted the control at 0.8443
   from loss alone; measured 0.844169 (residual −0.0002 — the tightest hold yet). The control is ON
   the line; B is +0.0081 ABOVE it. **All of B's advantage is off-law**: B is worse on one-step loss
   (0.12194 vs 0.12003) yet better on rollout SSIM, which is precisely the decoupling exp_03 set out
   to produce. Training on the trajectory the eval runs buys rollout quality that one-step loss
   cannot see.
4. **Tier 1 vs Tier 2 remains the sharpest contrast:** same off-law gain (+0.0081 vs +0.0100), but
   from a trained state B surrenders almost no one-step loss (−0.0003 vs the control's −0.0022) and
   the gain survives (+0.0059 net); from init it surrenders 0.0090 and the gain cancels (−0.0015 net).

A0's Tier-2 loss trajectory vs ctrl0: **+0.00189 @250, +0.00444 @1000, +0.01046 @2500**
(B0 was +0.00169 / +0.00281 / +0.00898).

**CORRECTION (same commit-day, self-caught):** my first write-up of this line said A0 gives up "far
less" one-step loss than B0. That is BACKWARDS — **A0 surrenders MORE** (+0.01046 vs B0's +0.00898
at 2,500). The claim is corrected here rather than silently edited.

That also **refutes a prediction I had drawn from S1.5**: A's gradient sits closer to the control's
(cosine 0.465 vs B's 0.228), from which I expected A to deviate less on the one-step metric. It
deviates more. So gradient-cosine-to-control does NOT predict how much one-step loss an objective
surrenders — a useful negative, and a reminder that the S1.5 diagnostics describe the *instantaneous*
gradient, not the trajectory 2,500 updates of following it produces. Whether A0 also buys more
off-law SSIM (which would keep the trade-ratio story intact) is exactly what its pending SSIM
answers.

## RESULT 4 — TIER 2 COMPLETE (A0, B0, ctrl0 @2,500 from init) — 2026-08-08

| arm | one-step loss | SSIM | law | **off-law** | vs ctrl0 | loss paid | law-priced cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ctrl0 (one-step) | 0.14598 | 0.813901 | 0.8132 | +0.0007 | — | — | — |
| B0 (rollout) | 0.15496 | 0.812425 | 0.8024 | **+0.0100** | −0.00148 (t=−2.9, 44/100) | +0.00898 | −0.01078 |
| **A0 (corrective SS)** | 0.15644 | **0.815026** | 0.8006 | **+0.0144** | **+0.00113** (t=+1.6, 67/100) | +0.01046 | −0.01256 |

**The trade-ratio story survives its own refuted prediction.** A0 surrenders MORE one-step loss than
B0 (+0.01046 vs +0.00898 — the reverse of what I predicted from S1.5's gradient cosines) but buys
proportionally MORE off-law SSIM (+0.0144 vs +0.0100). Off-law gain per unit of law-priced loss:
**A0 1.15, B0 0.93, ctrl0 n/a** — i.e. A0's exchange rate is favourable (>1) and B0's is not, which
is exactly why A0 nets positive from init (+0.0011) where B0 nets negative (−0.0015). Neither
approaches the +0.02 gate; A0's advantage is not significant at n=100 seed-0 (t=+1.6).

**What Tier 2 establishes:** from init, both trial objectives decouple from the law (4×–20× the
control's residual), so the mechanism is not an artifact of starting from a trained state. But at
2,500 updates the decoupling merely pays for the one-step loss it costs — A0 slightly ahead, B0
slightly behind, both practically a wash. **Tier 1 is where the trade turns clearly profitable**
(B +0.0059, t=+21), because there the objective surrenders almost no one-step loss.

**A Tier-1 instrument** (this hour): anchor 0.1222672 (exp_02's 0.12227 — VALID),
12,500 = 0.1186258 (segment -0.00364; B was −0.00033, control −0.00224).
A's Tier-1 SSIM pending — the last piece of the A-vs-B comparison at the trained state.

## RESULT 5 — TIER 1 A vs B vs control @12,500 (C pending) — 2026-08-08

| arm | one-step loss | Δloss over segment | SSIM | law | off-law | **vs control** | t | improved | ≥0.95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (one-step) | 0.12003 | −0.00224 | 0.844169 | 0.8443 | −0.0002 | — | — | — | 0 |
| B (rollout) | 0.12194 | −0.00033 | 0.850115 | 0.8421 | +0.0081 | **+0.00595** | +21.0 | 99/100 | 1 |
| **A (corrective SS)** | **0.11863** | **−0.00364** | **0.851570** | 0.8460 | +0.0055 | **+0.00740** | **+22.8** | **100/100** | 1 |

**A wins the trained state, and wins it differently than B does.**
- **A beats the control on BOTH axes simultaneously** — better one-step loss (0.11863 vs 0.12003;
  it optimizes the *shared* metric 1.6× faster than the control does) AND better rollout SSIM
  (+0.00740, t=+22.8, **100/100 windows improved**). No trade at all: A is a strict improvement.
- **B wins only off-law**: worse one-step loss, better SSIM. Two genuinely different mechanisms
  reaching the same place.
- **A > B directly**: +0.00146 paired, t=+5.8, A better in 73/100 windows.
- **Neither meets the +0.02 practical-effect gate.** By the predeclared rule both are
  statistically overwhelming, practically small (a third of the bar).
- The law now has **8 holds**: the control's residual −0.0002 is its tightest ever; A and B sit
  +0.0055 and +0.0081 above it, both far outside the ±0.005 band that governed every one-step arm.

**Synthesis across tiers (C pending):**
| | off-law gain | one-step loss surrendered | net vs control |
| --- | --- | --- | --- |
| Tier 2 B0 (init) | +0.0100 | +0.00898 | −0.0015 |
| Tier 2 A0 (init) | +0.0144 | +0.01046 | +0.0011 |
| Tier 1 B (10k) | +0.0081 | +0.00191 | +0.0059 |
| **Tier 1 A (10k)** | **+0.0055** | **−0.00161 (IMPROVED)** | **+0.0074** |

The pattern is monotone in one quantity: **how much one-step loss the objective gives up.** From
init both arms pay ~0.01 and net ≈0; from the trained state B pays 0.002 and nets +0.006; A pays
nothing (it gains) and nets +0.007. exp_03's mechanism is real, and it is cheapest — therefore most
profitable — exactly where exp_02 stalled: at a trained checkpoint whose one-step loss has already
flattened.

## RESULT 6 — TIER 2 FINAL: C0 breaks the interpolation — 2026-08-08

| arm | loss | Δloss vs ctrl0 | SSIM | law | off-law | **vs ctrl0** | ratio* | t | improved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctrl0 (one-step) | 0.14598 | — | 0.813901 | 0.8132 | +0.0007 | — | — | — | — |
| B0 (rollout) | 0.15496 | +0.00898 | 0.812425 | 0.8024 | +0.0100 | −0.00148 | 0.93 | −2.9 | 44/100 |
| A0 (corrective SS) | 0.15644 | +0.01046 | 0.815026 | 0.8006 | +0.0144 | +0.00113 | 1.15 | +1.6 | 67/100 |
| **C0 (λ·A + (1−λ)·B, λ=0.5)** | **0.14983** | **+0.00385** | **0.820169** | 0.8086 | **+0.0116** | **+0.00627** | **2.51** | **+14.2** | **95/100** |

\*ratio = off-law gain ÷ law-priced cost of the surrendered one-step loss. >1 profits.

**C0 is not an interpolation of its components — it is better than both, on both axes.**
- Its one-step loss cost is **+0.00385 — LESS THAN HALF** of either parent (A0 +0.01046, B0 +0.00898)
  and far outside the [B0, A0] range a 50/50 blend would predict (+0.00972). Same at every
  checkpoint: 250 (+0.00106 vs blend +0.00179), 1000 (+0.00102 vs +0.00362).
- It keeps an off-law gain of +0.0116, between its parents' +0.0100 and +0.0144.
- Net vs ctrl0 **+0.00627 (t=+14.2, 95/100 windows)** — **5.5× A0's advantage and the only Tier-2
  arm with a decisive win.** Its exchange ratio 2.51 more than doubles A0's 1.15.

**Interpretation (hypothesis, not established):** the two objectives' gradients are near-orthogonal
at init (S1.5 measured cos(A,B) = 0.459 at init, 0.165 at 10k) and their one-step-loss *costs* appear
to partially cancel while their off-law *gains* do not. Averaging two differently-wrong descent
directions lands closer to the shared metric than either alone, without giving up the rollout signal.
A cleaner test would be a λ-sweep (0.25/0.75) — not yet run, not yet proposed.

**Tier 2 final ordering: C0 (+0.0063) ≫ A0 (+0.0011) > ctrl0 > B0 (−0.0015).** The combined
objective — the arm that needed the 34MB HBM fight and the accumulation redesign to run at all — is
the one that wins from scratch. Tier-1 C (still queued) now carries the experiment's most
interesting open question: whether the same super-additivity appears at the trained state, where A
already beats the control on both axes.


## RESULT 7 (partial) — the λ-sweep confirms the U-shape — 2026-08-08

Query 7's predeclared prediction: if C0's super-additivity comes from gradient cancellation, the
one-step-loss cost should be U-shaped in λ. Measured (Tier 2, @2,500, cost vs ctrl0):

| λ (A-share) | 0 (B0) | 0.25 | **0.5 (C0)** | 0.75 | 1 (A0) |
| --- | --- | --- | --- | --- | --- |
| loss cost | +0.00898 | +0.00467 | **+0.00385** | +0.00599 | +0.01046 |

**U-shape CONFIRMED, minimum at (or near) λ=0.5**, consistent at every checkpoint (250/1000/2500).
The two objectives' one-step-loss errors partially cancel when mixed. Off-law SSIM gains for
λ=0.25/0.75 pending (their SSIMs queued) — those decide whether NET advantage also peaks at 0.5.

**Also this hour:** the law's **9th hold is its most extreme** — control @17,500 measured 0.848196
vs predicted 0.8482 (residual +0.00001). And Tier-1 C's instrument: anchor exact, segment Δloss
−0.00294 — C beats the control on the one-step axis at the trained state too (A −0.00364 >
C −0.00294 > control −0.00224 > B −0.00033; the same A>C>ctrl>B ordering as Tier 2's cost ranking,
inverted — the blend inherits A's one-step-friendliness).

## RESULT 8 — λ-sweep COMPLETE + Tier-1 C: the full mechanism map — 2026-08-08

**Tier-2 net curve (vs ctrl0, @2,500 from init):**

| λ (A-share) | loss cost | SSIM | off-law | **NET** | t | improved |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (B0) | +0.00898 | 0.812425 | +0.0100 | −0.00148 | −2.9 | 44/100 |
| 0.25 | +0.00467 | 0.818092 | +0.0105 | +0.00419 | +10.1 | 90/100 |
| **0.5 (C0)** | **+0.00385** | **0.820169** | +0.0116 | **+0.00627** | +14.2 | 95/100 |
| 0.75 | +0.00599 | 0.819832 | +0.0139 | +0.00593 | +11.6 | 93/100 |
| 1 (A0) | +0.01046 | 0.815026 | **+0.0144** | +0.00113 | +1.6 | 67/100 |

The decomposition is textbook: **off-law gain rises monotonically with the A-share** (+0.0100 →
+0.0144 — A's corrective label is the stronger rollout signal), while **loss cost is U-shaped with
its minimum at 0.5** (gradient cancellation). Net = their difference, peaking at λ=0.5 (0.75 within
noise of it). Both new arms are decisive (t=+10.1/+11.6). λ=0.5 was, by luck or by symmetry, already
optimal among the five tested.

**Tier 1 FINAL @12,500 (all four arms, measured control):**

| arm | Δloss | SSIM | off-law | vs control | t | improved |
| --- | --- | --- | --- | --- | --- | --- |
| control | −0.00224 | 0.844169 | −0.0002 | — | — | — |
| B | −0.00033 | 0.850115 | +0.0081 | +0.0059 | +21.0 | 99/100 |
| **A** | **−0.00364** | **0.851570** | +0.0055 | **+0.0074** | +22.8 | 100/100 |
| C | −0.00294 | 0.849752 | +0.0046 | +0.0056 | +21.9 | 100/100 |

**C is NOT super-additive at the trained state** — A > B ≈ C. And that asymmetry CONFIRMS the
cancellation theory rather than undermining it: cancellation pays only where the parents' one-step
costs are large (init: ~+0.01 each → C0 halves them → big win). At the trained state the costs are
small or negative (A GAINS one-step loss), so there is nothing to cancel — the blend just averages
its parents' off-law gains and A's solo advantage survives. **Recipe implication: λ should be
state-dependent — blend early (from init), anneal toward pure A as training matures.** Not tested;
would need a schedule arm.

**Standing scoreboard:** trained-state winner **A** (+0.0074, both axes, 100/100); from-init winner
**C₀.₅** (+0.0063, 95/100). The law: 9 holds, last at +0.00001. Gate status: nothing yet at +0.02;
best single number so far is A@12,500. A-ext@17,500 SSIM pending — the compounding verdict.

## RESULT 9 — A @17,500: the advantage GROWS — 2026-08-08

| @17,500 | loss | SSIM | law | off-law | vs control |
| --- | --- | --- | --- | --- | --- |
| control | 0.1168342 | 0.848196 | 0.8482 | +0.0000 | — |
| **A** | **0.1153891** | **0.856099** | 0.8499 | **+0.0062** | **+0.007904 (t=+30.8, 100/100)** |

- **CORRECTED (analysis review, finding 1 — my arm-mixing error, disclosed):** the original text
  compared B@12,500 (+0.0059) to A@17,500 (+0.0079). A's own trajectory is **+0.007401 @12,500 →
  +0.007903 @17,500 — a widening of +0.0005 over 5,000 updates**, with no
  difference-in-differences uncertainty computed. Supported claim: the advantage PERSISTED and was
  marginally larger. NOT supported: "compounds"; the "~15k updates to the gate" extrapolation is
  WITHDRAWN. A's off-law residual +0.0055 → +0.0062; the control stayed on the law.
- Per-7,500-update accounting stands: A gained +0.0104 SSIM over 10,000→17,500 vs the control's
  +0.0064 at identical LR and updates.
- **Gate arithmetic WITHDRAWN** (rested on the arm-mixed widening). Honest statement: A is
  decisively better within this seed and its lead persisted through 17,500; whether it grows,
  saturates, or shrinks is not established from two closely-spaced points.

**This closes the Query-7 overnight program.** All 9 results stand on: certified controls (1e-11
AND-gate; 9-hold law), exact seed anchors at every handoff, one seed, canonical-100 windows.
