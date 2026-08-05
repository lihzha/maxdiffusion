# exp_04 `null_adapter` — Yixun's driving queries

## Query 1 — 2026-08-04 (session start)

**Verbatim:**

> I need you first load the context form @CLAUDE.md and @docs/worklogs_yixun/ to understand this project. And I need you to follow the @docs/worklogs_yixun/experiment_SOP.md to design, record and report, and run all the experiment. And what what I want you to run experiment is like this. You can read @inverse_DDIM_pdf.md as reference to DDIM, and currently we want to train the adapter for the maxdiffusion project. I want you to follow the code of @third_party/Wan2.2/scripts/embedding_search.py and @third_party/Wan2.2/scripts/verify_reconstruction_from_null.py and @third_party/Wan2.2/scripts/embedding_search_smoke.py files to implement the null embedding computation and use the side adapter dataset to train the adapter and visualize how good this method will do for the Wan2.2

**Summary:** Port the null-text-inversion machinery (Mokady et al. 2022, `inverse_DDIM_pdf.md`; reference implementation in `third_party/Wan2.2/scripts/embedding_search.py` `--mode null_inversion`, verified by `verify_reconstruction_from_null.py`, gradient-flow smoke pattern in `embedding_search_smoke.py`) into the maxdiffusion JAX codebase: (1) implement the **null embedding computation** — DDIM-style reverse-Euler inversion of a target video latent at w=1, then per-sampling-step optimization of the unconditional (null) T5 embedding at deployment CFG so the guided sampler tracks the inversion pivot trajectory; (2) use the **side-adapter dataset** (cached DROID latent windows: `z_i0` [48,1,12,20], `z_video` [48,9,12,20], `actions` [32,7] on `gs://v6_east1d/datasets/droid_wan_side_adapter/`) to **train the adapter** on this method; (3) **visualize how good the method is** for Wan2.2 (reconstruction quality vs bounds, rollout metrics, comparison videos).

**User's assumption / hypothesis:** Per-timestep null embeddings recovered by inversion can reconstruct DROID videos through the frozen Wan2.2 TI2V 5B backbone far more accurately than the current one-step-denoising-trained adapters roll out (exp-observed rollout SSIM ≈ 0.29 for pre_context at 30k steps). If an action-conditioned adapter can be trained on/toward these null embeddings, it inherits that reconstruction quality — attacking the same objective/rollout mismatch exp_03 attacks, but from the "what conditioning signal would the sampler need" direction instead of the "what loss should training use" direction.

**Why the experiment needs to run:** The existing adapters optimize a one-step denoising loss whose relationship to 25-step rollout quality is indirect (exp_02's loss→SSIM law; exp_03's objective work). Null-text inversion gives a per-example *constructive* answer — an explicit per-step conditioning sequence that provably reconstructs the target through the frozen backbone — establishing (a) a capacity upper bound for conditioning-only control of Wan2.2 on DROID, and (b) a concrete regression target for an amortized action-conditioned adapter.

**Scope notes recorded at scaffold time (Planner reading, to be confirmed in the plan):**
- "null embedding computation" = the `null_inversion` mode (inversion trajectory + per-step null-embedding optimization), operating on the already-cached latents (no VAE encode needed for inversion itself; VAE decode needed only for pixel metrics/videos).
- "train the adapter" = an action-conditioned module that predicts the per-step null embeddings from (z_i0, actions) — exact architecture/target format to be fixed in the plan after auditing the existing PyTorch adaptor line in `third_party/Wan2.2/scripts/`.
- "visualize" = HTML report(s) per SOP artifact 12: bounds ladder (VAE-ceiling / null-text reconstruction / DDIM-inv-only), per-step loss curves, rollout videos + SSIM/MSE vs the existing side_adapter / pre_context baselines.

## Query 2 — 2026-08-04T15:18Z (plan approval + exp_05 directive)

**Verbatim:**

> Yes to all, but what I actually want is to get text token from inverse DDIM, and use pre_context structure to learn the text token (the positive text embedding and use loss function to constrain adapter to do that). You can treat what I am proposing here as exp_05, please design the plan for exp_05 and parallel run exp_04 and exp_05.

**Grants recorded for exp_04 (per the approval package's decision points 1–5):**
1. **Plan v5 APPROVED.** **J0 approved** (manifest build, host-only, capped). **J1 approved conditional on**: P0 tests complete + parity audit clean — per the package wording "once P0 tests + the parity audit are done"; no re-ask needed when those conditions are met (announcement 02 conditional-grant rule). J1b–J5 remain gated and will be asked separately.
2. Cohorts DEV-64 / TEST-64 / TRAINFIT-16 / TRAIN-2000 approved.
3. L_null = 16 approved (ablation diagnostic-only).
4. A2 `noise=global` fallback deployment convention approved.
5. Pilot scope acknowledged.

**Scope note:** the second sentence defines a NEW experiment (exp_05 `pos_context`): per-step POSITIVE text embeddings from DDIM inversion + a pre_context-structure adapter trained with a regression loss to those embeddings, run in parallel with exp_04. Recorded in `exp_05_pos_context_claude/pos_context_yixun_query.md`; exp_04's scope is unchanged.

## Grant (2026-08-05T20:26Z) — J1-2b supplement pre-approved

Context: status report recording the J1 runbook's missing TRAINFIT-16 capacity half (worklog 2026-08-05T20:40Z entry) and the prepared `submit_j1b_trainfit.sh` remediation, gated on J1-2 completing (its adequacy artifact is J1-2b's input).

**Yixun, verbatim:** "Yes to all, continue with both experiments"

Planner reading: J1-2b launches via Yixun's `!` once J1-2 is terminal-success; if J1-2 fails on infra, auto-resubmit first per the standing policy.
