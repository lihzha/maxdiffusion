# Master Experiment Tracker

Cross-experiment status index — one section per experiment/run, newest first. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`. Per-experiment detail lives in `exp_<NN>_<name>_claude/` folders per `experiment_SOP.md`; this file is the at-a-glance map.

Last updated: 2026-08-01

## exp_02 `overfit100` — text-conditioned 100-trajectory memorization test (ACTIVE — 10k verdict passes in flight)

- **What:** Can full-FT Wan2.2 TI2V 5B **memorize** the windows of **100 successful DROID trajectories** conditioned on (per-trajectory **language instruction** via T5 + first-frame latent)? Lihan/Yixun's corrected overfit design after exp_01 (which was full-corpus and unconditioned). Framed as a finite-set memorization test, NOT "text determines the future" (plan-review F1).
- **Status (2026-08-01T16:15Z): ALL training done; step-2500 verdict COMPLETE (both tiers NOT established); extension to 10k trained; 2 of 4 extension evals done; the two 10k verdict passes are running with resume staging.**
  - **Headline science:** memorization is real, text-conditioned, and **saturating well below the D11 bar**. Canonical-100 mean SSIM by checkpoint: 0.7580 (250) → 0.7707 (500) → 0.7892 (1000) → 0.8020 (1750) → **0.8133 (2500)** → **0.8320 (5000)** → **0.8377 (7500)**. Per-250-step gain decayed 0.0038 → 0.0019 → 0.0006. **No window has ever reached 0.95** at any checkpoint (best ≈0.947, frozen since 2500). Train loss 0.586 → 0.145 (2500) → 0.132 (5000) → 0.127 (7500) → ≈0.12 (10000).
  - **Formal verdict at step 2500 (complete, committed):** canonical tier 0/100 at m_corr ≥ 0.95 (mean 0.8133) — NOT established; full-set tier 105/1,629 at ≥0.90 (6.4%, needs 90%; mean 0.7984, min 0.2589, max 0.9480) — NOT established. Verdict JSONs + both aggregations in `overfit100_s3_artifacts/`.
  - **Sampling probe (Yixun-approved, Job 24, DONE):** sampling axis **CLOSED**. Validity bitwise-perfect (25-step arm reproduces landed rows exactly, mean 0.8100125855). 50 steps: −0.0074 mean, **0/30 windows improved**; 100 steps: −0.0121, 0/30. More integration steps are strictly worse — velocity-field error compounding dominates discretization error at 25 steps. Also discovered: the OVERFIT100 rollout has **no CFG branch at all** and training used none, so no train/eval guidance mismatch exists (that probe arm was void).
  - **Extension to 10k (Yixun: "extend to 10k", Job 23, training PASSED attempt 1):** resumed from 2500 seamlessly (step 2501 loss 0.139), 1.9 steps/s, checkpoints {5000, 7500, 10000} saved. Evals: i5000 (0.8320) and i7500 (0.8377) DONE with **aux/VAE ceilings populated for the first time** (mean 0.9493, min 0.8812) — the ffmpeg fix works.
  - **In flight:** `20260801-142914-9d6458a0-…-final10k` (900 rollouts, 3×3 canonical) and `20260801-142941-95e2e0a1-…-fullset10k` (1,629 windows). Both on attempt 1 after infra kills; **resume staging is working** — final10k had 440/900 rows banked across the preemption instead of restarting from zero.
  - **Code:** all closed and reviewed. Suite **1,236 passed / 2 skipped**. The eval-resume series (`78819dc`→`99ee724`→`2b0fd30`→`9c12a1f`→`fc9ac52`) took 5 Codex passes to APPROVE (fail-closed run-signature envelope, strict-type admission, D4-first ordering, write-suppressed published dirs, authenticated marker + compare-only publication); then ffmpeg fix, probe tooling + hardening (2 passes, APPROVE). `overfit100_success_statistic.py` and the aggregation schema were re-verified untouched in every pass.
- **Locked decisions:** view 0 only; LR 1e-5 (absolute 250-step warmup then constant — extension-safe by design); data path A′; selection = seeded, `success==1`, non-empty-instruction filter, 1-of-3 pick per episode; two-tier D11 success rule (canonical ≥90% at m_corr ≥ 0.95; full-set ≥90% at seed-0 ≥ 0.90 at c*); text ablations correct/null/shuffled (value-derangement). Fail-closed verdict pipeline throughout.
- **Branch / worktree:** `claude-exp_02_overfit100-20260728` (tip ≈`46c5f41`+); worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100`. Docs auto-sync to `yixun-dev` via post-commit hook.
- **Docs:** `docs/worklogs_yixun/exp_02_overfit100_claude/` — queries 1–7, plan v4, manifest, worklog, `_command.md` (Jobs 1–28), `_results.md` (S2 + all S3 + probe), `_analysis.md` (v1 + §8 updates), `overfit100_s3_artifacts/`, 4 plan reviews + ~22 code-review records with strengthening records.
- **Next:** 10k verdict passes land → verdict CLI over ALL admitted artifacts (c* by fraction tie-break) → finalize `_analysis.md` → Codex analysis review → HTML reports → Yixun: merge decision + exp_03 direction. **Expected conclusion:** the one-step-denoising → 25-step-rollout recipe saturates near mean 0.84 / max ~0.95 on this cohort; the bar is unreachable by more training or sampling tweaks, so exp_03 should change the recipe/objective or the claim — not the dose.

## exp_01 `full_ft_overfit` — full-FT overfit diagnostic

- **What:** Plain Wan TI2V **full finetune** (backbone unfrozen, **no adapter**) overfit sanity check on DROID. Diagnostic to separate **(A)** "data/loss/pipeline broken" from **(B)** "frozen-backbone + adapter optimization too hard". Explicitly **NOT** the long-term method.
- **Status:** PARTS I+II COMPLETE (2026-07-27) — Part I: train-cohort SSIM 0.197→0.787 (official). Part II (held-out): val loss monotone 0.18447→0.17885 across ckpts 2500–20000 (n=14,636/pt, deterministic per-position (t,ε)); val-clip rollout SSIM 0.7269 vs train 0.7875 — domain transfer established, memorization share unquantified. Analyses Codex-reviewed (Part I: 6 findings; Part II: 6 findings — all applied). Reports 01/02/03/04 (03 = val gallery, 04 = train-cohort gallery; both with in-repo videos incl. |pred−GT| residuals ×4). **Merge decision made 2026-07-27: NOT merging** (Query 10) — code stays on the experiment branch; docs are on `yixun-dev` via the hook.
- **Branch / worktree:** `claude-exp_01_full_ft_overfit-20260715` off `yixun-dev` @ `8258965`; worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit`. Exp-docs auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Docs:** `docs/worklogs_yixun/exp_01_full_ft_overfit_claude/` — queries 1–10, plan (Parts I+II), 2 plan reviews + 6 code reviews + 2 analysis reviews, worklog, `_command.md` (13 launches), params, results, analysis, reports 01–04 + assets.
- **Decision rule:** overfits fast → pipeline OK, continue adapter work; can't overfit after substantial steps → debug data / loss / noise / CFG / latent alignment first.
- **Next:** none open — exp_01 closed. Optional leftovers: qualitative video review by Yixun/Lihan; a matched held-out cohort job to quantify memorization; §2.4 controls (deferred, required only before any capacity/dose-ceiling claim). exp_02 direction (trainable-leverage variants under the identifying design in `_analysis.md` §4) awaits Lihan.

## Run: wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110

- **What:** `pre_context` adapter (~128M trainable) on frozen Wan2.2 TI2V 5B; DROID first-frame + 32×7 actions → video latents. v6e-64, global batch 512, pure FSDP, `side_adapter_noise_mode=fresh`, 30k steps.
- **Status:** COMPLETE (training + step-30000 validation).
- **Training health:** wandb train/val curves healthy, no overfit (checked 2026-07-11).
- **Validation @ step_030000** (25-step rollout, 4 samples, v6e-8 job — succeeded on attempt 6 after 5 infra preemptions):
  - mean_latent_mse **1.496**, mean_pixel_mse **0.0983**, mean_ssim **0.2946**
  - Artifacts: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter/wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110/validation/step_030000/` — per-sample `ground_truth.mp4` / `sample.mp4` / `comparison_gt_top_pred_bottom.mp4` / `metrics.json`, plus `summary.json`/`summary.csv`.
  - Interpretation: low SSIM does not contradict the healthy one-step denoising loss — the 25-step rollout accumulates error. User has the gsutil pull commands; awaiting their read on the videos.
- **Open options:** comparison validation at checkpoint 29000 (offered, not requested); next-experiment direction pending user's video review.

## Prior work (pre-tracker)

Side-adapter (~240M) and earlier pre-context history: see `docs/side_adaptor.md`, `docs/wan_ti2v_pre_context_adapter_methodology_results.md`, and Lihan's worklogs under `docs/worklogs/wan-ti2v-side-adapter/`.

## Process / infra state (for handoff)

- SOP: `docs/worklogs_yixun/experiment_SOP.md` — three-role separation (Planner Fable 5 / Coder Opus 5 max / Reviewer Codex `gpt-5.6-sol` xhigh), closed write→review→strengthen cycles, TDD tests in `src/maxdiffusion/tests/worklogs_yixun/`, every launch logged in `_command.md` at launch time.
- Codex reviewer: CLI 0.144.1 (`codex exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh`), MCP server `codex` wired at user scope; verified working 2026-07-12.
- Submodule: `third_party/Wan2.2` = lihzha/Wan2.2 @ `f370228`.
- Handoff automation: `.claude/settings.local.json` + `.claude/hooks/handoff_snapshot.sh` (both gitignored, local-only) fire on `ConfigChange` / `PreCompact` / `SessionEnd` — they append a git-state breadcrumb to `docs/worklogs_yixun/_handoff_events.log` and nudge to refresh the handoff docs. No hook exists for a model exhausting its quota, so that trigger is proactive-only. Full protocol in `CLAUDE.md`. To re-create on a fresh clone: recreate those two files (gitignored) and open `/hooks` once.
- Branch: `yixun-dev` (integration). Experiments so far: `exp_01_full_ft_overfit` (COMPLETE Parts I+II, unmerged by decision, on branch `claude-exp_01_full_ft_overfit-20260715`); `exp_02_overfit100` (ACTIVE, on branch `claude-exp_02_overfit100-20260728`). Next experiment number is `exp_03`.
