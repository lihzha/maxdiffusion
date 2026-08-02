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
