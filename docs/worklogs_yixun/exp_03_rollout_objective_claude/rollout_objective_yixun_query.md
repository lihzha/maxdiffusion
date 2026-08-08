# exp_03 `rollout_objective` — Yixun's driving queries

## Query 1 — 2026-08-02 (verbatim)

> And please open the exp_03 parallely: three trials for exp_03: A. multi-step / scheduled sampling, B. short-horizon rollout loss and C. \lambda (loss of A) and (1-\lambda) (loss of B) which combines multi-step /scheduled sampling and short-horizon rollout loss.

**Summary.** Open exp_03 in parallel with the exp_02 LR sweep. Three trials attacking the compounding
mechanism exp_02 identified: (A) multi-step / scheduled sampling; (B) a short-horizon rollout loss;
(C) a convex combination λ·L_A + (1−λ)·L_B.

**Context / why this experiment runs.** exp_02's verdict: full-FT memorization of 100 DROID trajectories
is NOT established (0/100 canonical at 0.95; plateau ≈0.84). Its diagnostics located the defect precisely:
the one-step denoising objective transfers to 25-step rollout SSIM through a stable linear map
(SSIM ≈ 0.9885 − 1.201·loss, r = −0.9994) whose slope is the ~1.2× price of error compounding; per-frame
SSIM starts at ~0.97 for EVERY window and decays monotonically along the rollout (−0.04 best window,
−0.30 worst). The intercept says a perfect one-step denoiser would clear the bar; the slope says one-step
error is amplified by the trajectory. exp_03 attacks the slope — train on (or toward) the trajectory the
eval actually runs — rather than the loss floor, which the LR sweep is bounding in parallel.

**User's hypothesis (implicit).** Objective-side changes (scheduled sampling and/or rollout loss) reduce
the compounding amplification and lift rollout SSIM at matched compute, where more one-step training
provably plateaus.

## Query 2 — 2026-08-02 (verbatim)

> Keep the continue-from-10k A/B/C arms. Also add a from-scratch tier: same Wan init, retrain from step 0 with the new losses (A/B/C + a one-step control); you pick the budget in the plan. Update the plan for my approval first — no launch until I approve.

**Summary.** Two-tier design: Tier 1 = continue-from-10k A/B/C (kept as planned); Tier 2 = from-scratch
A0/B0/C0 + fresh one-step control (ctrl0) from the same Wan init. Budget is the Planner's choice, made in
plan v3 (2,500 steps/arm). Plan returns for approval before any launch.

## Query 3 — 2026-08-03 (verbatim)

> approve S1 smoke when the package is ready

**Summary.** Conditional pre-approval: the S1 smoke launch (v6e-8) is approved contingent on (a) round 3
closing with the reviewer's APPROVE, and (b) the package being assembled per plan v3.2 — explicit
`EXP03_RAMP_ORIGIN` per tier, the declared A/B/C step-time STOP budgets (≤1.6× / ≤2.5× / ≤3.2× baseline;
exceeding one is a STOP, not a silent accept), and the standard `_command.md` entry at launch time. No
further ask before the S1 launch once both conditions hold. (Mirrors exp_02's recorded
conditional-approval pattern.)

## Query 4 — 2026-08-03 (verbatim)

> What is your status for exp_03, please do the lr exp_02 and exp_03 parallelly

Plus the AskUserQuestion grant: **"Yes, grant all three"** — pre-approval for the exp_03 S1 follow-up
launches, each conditional on its preceding gate: (1) **C re-smoke** when the S1-fix review of `76ff476`
passes (incl. any strengthening the reviewer requires first); (2) **S1.5** no-update discriminator probe
(v6e-8) when the re-smoke is clean; (3) **S1.6** one-step mesh-fit at GBS 256 (v6e-64, ~minutes) after
S1.5. S2a/S2b training arms remain separately approval-gated. exp_02's lr thread and exp_03 proceed in
parallel.

## Query 5 — S1.6 budget decisions (2026-08-06)

Presented: B at 2.713× vs the ≤2.5× budget (accept vs trim); C misses the GBS-256 fit by 34.32MB
(per-device batch 2 + gradient accumulation / deeper remat / drop from Tier 1).

**Yixun: "B accept and C batch-2"** — verbatim. Decisions bound into the S2 package:
- **B: ACCEPTED at 2.713×** (update-matched design unchanged; wall-clock cost ~8.5% over the
  planning budget, knowingly carried).
- **C: per-device batch 2 with 2× gradient accumulation** (GBS stays 256, updates stay matched;
  the accumulation step is a config/trainer-capability check for the package — if the trainer
  lacks native accumulation, that small addition goes through the usual code round + review
  before the S2 launches).

The S2a/S2b launches themselves remain approval-gated (announcement 02): the package goes to
Yixun with these settings baked in.

## Query 6 — THE S2a/S2b TRAINING-ARMS LAUNCH PACKAGE (for approval) — 2026-08-06

All gates passed: S1 CLOSED, S1.5 banked+admissible, S1.6 CLOSED; Query-5 decisions bound
(B accepted at 2.713×; C at per-device batch 2 + 2× accumulation).

**S2a — Tier 1 (from-10k), READY NOW (no code change):**
- **Arms A + B**: v6e-64, GBS 256, +2,500 updates each from a per-arm server-side copy of
   (sweep pattern, byte-verified);
   explicit (standing requirement); LR 1e-5 (update-matched vs control);
  COMMIT = tip (code = APPROVED 7da7a66 lineage).
- **Control = the EXISTING exp_02 lr1e5c arm** (10,000→12,500 at 1e-5) — already run, already
  instrumented (loss 0.1919…-anchored); no new launch.
- **Cost:** A ≈ 46 min, B ≈ 1.8 h (2.713×, accepted) on v6e-64 + the standard measurement pair
  (instrument + seed-0 canonical SSIM) per arm on v6e-8 afterward under the same approval.
- **Acceptance per arm:** restore start_step=10000; ramp origin logged 10000; all steps finite;
  checkpoint 12,500 saved; then primary metric = canonical seed-0 mean SSIM at 12,500 vs lr1e5c's
  (+0.02 practical-effect gate per plan v3.2).

**S2b — Tier 2 (from-scratch), ctrl0 FIRST:**
- **ctrl0**: v6e-64, 2,500 updates from Wan init, exp_02's exact RNG stream; **AND-gate** vs the
  exp_02 full-precision anchors (|Δloss|≤1e-4, |ΔSSIM|≤5e-4, max-window ≤1e-3 at the pinned
  checkpoints). A0/B0 launch only after ctrl0's gate PASSES (separate go).
- **C and C0**: BLOCKED on the gradient-accumulation capability (confirmed absent from the
  trainer) — one small code round + Codex review, then C launches under this same approval.

**Ask:** approve (1) A+B now, (2) ctrl0 now, (3) the accumulation code round now (C/C0 launch
after its review passes, no separate ask). Tip at packaging: 31f00c474.


## Query 7 — overnight blanket grant (2026-08-08, ~10 h window)

**Yixun, verbatim: "Currently I will go to bed, so I will approve everythin after your
recommendation util I wake up, potential 10h from now. Please go ahead"** — a time-boxed approval
umbrella for Planner-recommended actions. Under it, launched now (each individually recorded in
`_command.md`):
1. **λ-sweep (Tier 2):** C-objective arms at λ=0.25 and λ=0.75, from init, 2,500 updates, N=2 —
   the direct test of RESULT 6's super-additivity hypothesis (prediction if gradient-cancellation
   is right: one-step-loss cost is U-shaped in λ with the minimum near where the components'
   errors cancel; off-law gain roughly monotone in the A-share).
2. **Trained-state extensions 12,500 → 17,500:** the A arm and the lr1e5c control, +5,000 updates
   each, update-matched — does A's both-axes advantage (+0.0074) compound toward the +0.02 gate?
3. Tier-1 C continues (already queued); its eval pair on landing.
4. **Predeclared rule for a C extension:** if Tier-1 C's net vs control exceeds A's +0.0074, extend
   C 12,500→17,500 as well (~4.6 h at 5.92×; fits the window if it lands early enough); otherwise not.
All follow-on eval pairs (instrument with known anchors + canonical seed-0 SSIM) launch
automatically as runs land. Nothing outside this list launches under the umbrella.
