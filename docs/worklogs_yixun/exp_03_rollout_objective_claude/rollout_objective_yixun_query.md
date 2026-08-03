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
