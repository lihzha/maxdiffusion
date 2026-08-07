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
