# exp_02 `overfit100` — Analysis (Planner)

Written 2026-08-01 after both step-2500 verdict-bearing artifacts landed. Author: Planner (Claude Fable 5). Inputs: all committed code (`fc9ac52` tip), configs, `_results.md`, the verdict CLI outputs in `overfit100_s3_artifacts/`, S2 gate artifacts, and the training logs recorded in `_worklog.md` / `_command.md`.

## 1. Question and design (recap)

Can full-FT Wan2.2 TI2V 5B **memorize** 1,629 windows of 100 successful DROID trajectories, conditioned on per-trajectory language instruction (T5) + first-frame latent — reaching near-perfect reconstruction (D11 two-tier rule)? This is a finite-set memorization test with text-ablation controls, not a "text determines the future" claim.

## 2. Verdict

**Both tiers formally NOT established at step 2,500 (c\*=2500, the sole segment-final checkpoint):**

| Tier | Rule | Measured | Verdict |
| --- | --- | --- | --- |
| Canonical | ≥90% of 100 windows at m_corr ≥ 0.95 (median over seeds 0–2, correct mode) | 0/100 (fraction 0.0); mean m_corr 0.8133, max 0.9461 | **not established** |
| Full-set | ≥90% of 1,629 windows at seed-0 SSIM ≥ 0.90 at c\* | 105/1,629 (fraction 0.0645); mean 0.7984, max 0.9480 | **not established** |

No window in either cohort reaches 0.95. The result is *uniform incompleteness*, not a bimodal memorized/unmemorized split.

## 3. Reliability audit — do I believe these numbers?

Yes, with one bounded caveat (aux). Grounds:

- **Fail-closed chain held end-to-end.** Both artifacts were admitted by the untouched `overfit100_success_statistic` (unchanged from the deployed eval SHA `e27fdc3` through `fc9ac52`, re-verified by Codex in every resume-round pass): manifest-derived fixed cohorts (denominator 100 / 1,629 — never pass-derived), role grids validated on exact (window, checkpoint, seed, mode) tuples (`role_validation.ok=true`, coverage complete both passes), manifest binding `c02a67be…` intact, immutable role-keyed paths.
- **Deployed code provenance:** both eval jobs ran the reviewed `e27fdc3` tarball (pre-resume; the resume series `78819dc…fc9ac52` landed after these jobs were queued and touches only staging/publication control — Codex pass-4 X6 explicitly confirmed it does not change how these artifacts are read).
- **Data integrity:** dataset published under `_SUCCESS` with byte-verify at training preflight; V1 encode-parity r≈0.995; V4 frame-0 future-invariance bitwise-exact; V3 decode ceilings 0.902–0.966 measured on probe windows.
- **Determinism/noise:** per-rollout RNG is `window_fold_key(seed, episode_id, window_start)` — order-independent. Canonical tier uses 3 seeds (median); seed spread was small in S2 and the intermediate checkpoints. Full-set is seed-0 single-sample per window by design (tier definition), so individual full-set window values carry seed noise; the *fraction* (0.0645 vs required 0.90) is far outside any plausible seed effect.
- **Caveat (bounded):** all aux/VAE-ceiling columns are missing (`ffmpeg` absent on the pre-fix tarball — issue #8). This affects **no** primary metric; it only removes the per-window ceiling context. Recoverable later via the checkpoint-independent backfill; the fix is merged on the branch (`9c26070`+`f4da3eb`).
- **Cross-experiment consistency check:** exp_01 (full-corpus FT) at val loss ≈0.176–0.184 produced rollout SSIM 0.727 (val) / 0.787 (train cohort). exp_02 at train loss 0.145 produces 0.798–0.813. The loss→rollout-SSIM mapping is consistent across two independent runs and codebases-paths, which argues against a pipeline artifact.

## 4. What the numbers say

**Trajectory (canonical-100 mean, correct mode):**

| step | 250 | 500 | 1000 | 1750 | 2500 |
| --- | --- | --- | --- | --- | --- |
| mean SSIM | 0.7580 | 0.7707 | 0.7892 | 0.8020 | 0.8133 |
| gain /250 steps | — | 0.0064 | 0.0093 | 0.0043 | 0.0038 |

Monotone, unsaturated, but **decelerating**: the per-250-step gain has fallen ~40% over the last 1,500 steps. Train loss 0.586→0.145 and still falling (~390 epochs at GBS 256).

**Cross-scale anchor (S2, 10 episodes, same recipe, same 2,500 steps):** loss reached 0.061 (≈2.4× lower than S3's 0.145) and m_corr reached 0.8896 — *also* far from 0.95, also still rising. Ablations ordered identically at both scales (S3: correct 0.8133 > null 0.7992 > shuffled 0.7824; S2: 0.8895 > 0.8398 > 0.8342).

**Full-set vs canonical:** mean 0.7984 vs 0.8133; the full set adds a harder tail (min 0.2589 vs 0.5870). 86.7% of all windows clear 0.70; only 6.4% clear 0.90.

## 5. Why 0.95 is not reached — hypotheses and discrimination

- **H1 — undertrained (dose).** Supported by: unsaturated curves, falling loss. Against: naive extrapolation at the current decelerating rate puts the *mean* at 0.95 only after ~10k+ further steps, and per-window 90%-at-0.95 is stricter than the mean; deceleration may reflect an approaching asymptote below the bar.
- **H2 — train/eval objective mismatch ceiling.** Training is one-step denoising; evaluation is a 25-step CFG rollout with frame-0 pinning. Error accumulates across steps, plausibly capping achievable rollout SSIM below ~0.9–0.95 *regardless of memorization quality*. The strongest evidence is S2: at loss 0.061 — deep memorization of 10 episodes — rollout m_corr was still only 0.890. If near-zero denoising loss maps to ≈0.89 rollout SSIM, the 0.95 bar may be unreachable under this sampling recipe.
- **H3 — capacity/interference.** Not credible as the binding constraint at 100 episodes on 5B params; the 10-episode run's shortfall at *lower* loss argues directly against capacity being the limiter.
- **H4 — VAE ceiling.** Not applicable to this metric: SSIM compares decoded-pred vs decoded-GT through the same VAE; the V3 ceilings (vs raw RGB) do not bound it.

**The data cannot yet separate H1 from H2.** Two cheap discriminators exist:

1. **Sampling-recipe probe on the existing step-2500 checkpoint** (one v6e-8 job, no training): sweep rollout steps (25 → 50/100) and CFG scale (incl. guidance off — training had CFG in the loss path, so match it) on the 100-canonical cohort. If SSIM moves materially, H2 has a large sampling-side component and extension alone won't reach the bar. *(Note: rollout steps / guidance are signature-bound in the new resume code — each variant is its own artifact; roles for probe passes would use a non-verdict role or a fresh output root, keeping D11 artifacts untouched.)*
2. **S3 extension (D10)** with checkpoints at 5,000 / 7,500 / 10,000 + intermediate evals: directly measures whether the curve saturates below the bar. Training itself is cheap (2,500 steps ≈ 39 min on v6e-64); the evals are the cost (~17.5 min per 100-window pass; segment-final-style passes only if a new c\* is to be claimed).

## 6. Recommendation (Planner)

Run **both discriminators, probe first** — it is a single cheap job on an existing checkpoint and can reorder everything else: if the sampling sweep already lifts windows to ≳0.95, the memorization is substantially *already there* and the story becomes "objective/sampling mismatch," changing what an extension should even optimize. Then decide the extension with that information. Both require Yixun's approval (new launches; the extension additionally per D10).

If neither discriminator moves the ceiling materially, the honest conclusion of exp_02 is: *full-FT Wan2.2 TI2V 5B memorizes 100 DROID trajectories only to ≈0.81 mean rollout SSIM at 390 epochs under the one-step-denoising → 25-step-rollout recipe; text conditioning contributes measurably (ablations ordered at both scales); near-perfect reconstruction is not attained and appears sampling-limited rather than capacity-limited* — a genuinely informative negative for the adapter program, since it bounds what the frozen-backbone adapters can be expected to reproduce.

## 7. Standing state for whoever picks this up

- All exp_02 code closed and reviewed (suite 1,197+2; ~23 Codex passes; eval-resume series APPROVED at `fc9ac52` — preemption-tolerant staging available for any future long pass).
- Both D11 artifacts + complete verdict committed under `overfit100_s3_artifacts/`; checkpoints {250,500,1000,1750,2500} retained on GCS.
- Open cheap follow-ups: aux/ceiling backfill job (checkpoint-independent); S2 ceiling backfill; both optional for the record, not for the verdict.
- Open decisions (Yixun): sampling probe approval; S3 extension approval (D10); merge decision (SOP: only on confirmed success — as of step 2,500 the formal answer is "not established").

## 8. Post-verdict updates (2026-08-01)

**Sampling probe (Yixun-approved) — sampling-side H2 EXCLUDED.** 30 canonical windows at ckpt 2500, arms {25, 50, 100} rollout steps: validity bitwise-perfect (25-arm reproduces the landed rows exactly), and more steps are strictly WORSE — 50: −0.0074 mean, 0/30 windows improved; 100: −0.0121, 0/30. Velocity-field error compounding dominates discretization error at 25 steps already. Corollary: the CFG arm was void by discovery — the OVERFIT100 rollout has no guidance branch and training used none, so no train/eval CFG mismatch exists. §5-H2's sampling-side component is closed; what remains of H2 is the one-step training objective itself.

**S3 extension to 10k (Yixun: "extend to 10k") — training PASSED, loss flattening hard.** Resume verified (step 2501 = 0.139, continuous with 0.145); loss 0.145 (2500) → 0.1318 (5000) → 0.1270 (7500) → ≈0.12 (10000, noisy 0.111–0.137). Tripling the compute bought ~0.025 of loss. Read together with the probe: if the 5000→7500→10000 SSIM trajectory flattens correspondingly below the bar, the conclusion sharpens to "the one-step-denoising objective is the ceiling under this recipe" — H1's remaining room is small.

**Revised expectation for the 10k verdict:** loss-to-SSIM has been roughly linear-ish across runs; a ~0.145→0.12 loss move projects only a modest SSIM gain (order +0.01–0.03 on the canonical mean, i.e. ≈0.82–0.84 at 10k). The 0.95 bar would remain far out of reach unless the mapping breaks upward. Awaiting the four extension evals for the formal answer.
