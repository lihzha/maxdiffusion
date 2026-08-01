# Master Experiment Tracker

Cross-experiment status index — one section per experiment/run, newest first. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`. Per-experiment detail lives in `exp_<NN>_<name>_claude/` folders per `experiment_SOP.md`; this file is the at-a-glance map.

Last updated: 2026-08-01

## exp_02 `overfit100` — text-conditioned 100-trajectory memorization test (COMPLETE — verdict: NOT ESTABLISHED)

- **What:** Can full-FT Wan2.2 TI2V 5B **memorize** the 1,629 windows of **100 successful DROID trajectories** conditioned on a per-trajectory **language instruction** (T5) + first-frame latent? Yixun's corrected overfit design after exp_01. A finite-set memorization test with text-ablation controls.
- **ANSWER: No — at the tested budget and recipe the predeclared D11 criterion was not met.** All planned work is complete; no eval outstanding.
  - **Final verdict (step 10,000, eval `46c5f41`, both tiers, coverage complete):** canonical **0/100** at m_corr ≥ 0.95 (mean 0.8414, max 0.9484); full-set **229/1,629 = 14.1%** at ≥ 0.90 (needs 90%; mean 0.8322, max 0.9509). Historical verdict at step 2,500 (eval `e27fdc3`): 0/100 and 105/1,629 = 6.4%. Two verdicts, not one — the success statistic **refused** to mix eval commits, correctly.
  - **Trajectory (canonical, seed-0 correct):** 0.7580 (250) → 0.8139 (2500) → 0.8320 (5000) → 0.8377 (7500) → **0.8416 (10000)**. Gains per 2,500 steps after 2500: +0.0181 → +0.0057 → +0.0039 (~5× decay). Practical plateau ≈0.84–0.85; the true asymptote is **not** statistically identified (a 0.90-constrained power law still fits at ~0.0013 RMSE).
  - **Only 2 of 1,629 windows ever crossed 0.95** (both at 10k, max 0.9509); the canonical cohort never did.
  - **Sampling axis closed (Job 24 probe):** 50 and 100 rollout steps were strictly worse than 25 — 0/30 windows improved at either. Validity check bitwise-exact on the 25-step arm. Also discovered: this rollout has **no CFG branch** and training used none, so the suspected guidance mismatch was void.
  - **Text conditioning is load-bearing and grows with training:** correct-vs-shuffled gap 0.031 (2500) → 0.053 (10000); null gap widened in 287/300 matched pairs. Narrowed claim: dependence on the *correct context* increased — this does not establish *semantic* use.
  - **Latent MSE fell 32%** (0.1592 → 0.1079) while SSIM rose only 3.4% — caution against reading D11's SSIM as a direct memorization measure. Per-window score correlates with the VAE's own difficulty at **r = 0.683**.
  - **Why it stops (narrowed, not eliminated):** objective/rollout mismatch (one-step denoising trained, 25-step rollout evaluated) is the **leading explanation**, not an identified cause. Dose is low-value but not excluded (LR was locked at 1e-5); the sampler is only excluded *upward*; capacity unlikely but not ruled out at 100-episode scale. Open alternatives H5–H7: optimization floor + the never-run fixed-RNG one-step instrument, episode-vs-window weighting mismatch, and metric scope (frame-0 pinning included in SSIM).
- **Analysis reviewed:** Codex verdict SOUND-WITH-REVISIONS (3 MAJOR / 4 MODERATE / 2 MINOR) — **all 9 accepted**, none rejected; v2's causal overclaim and its "adapter bound" were withdrawn. `_analysis.md` is at v3 with a full resolution record in `overfit100_codex_analysis_review.md`.
- **Code:** suite **1,236 passed / 2 skipped**, ~25 Codex review passes. The eval-resume series (5 passes to APPROVE: fail-closed run-signature envelope, strict-type admission, D4-first ordering, write-suppressed published dirs, authenticated marker + compare-only publication) **proved itself in production** — both 10k verdict passes survived preemptions and finished from staged rows. Reusable beyond exp_02.
- **Branch / worktree:** `claude-exp_02_overfit100-20260728`; worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100`. Docs auto-sync to `yixun-dev`.
- **Docs:** `exp_02_overfit100_claude/` — queries 1–7, plan v4, manifest, worklog, `_command.md` (Jobs 1–28), `_results.md`, `_analysis.md` v3, `overfit100_01_memorization_trajectory_results.html` (+ generator), `overfit100_s3_artifacts/` (all aggregations, both verdicts, probe), 4 plan reviews + ~25 review records.
- **Open decisions (Yixun):** (1) **merge or leave unmerged** — SOP says merge only on confirmed success; formal answer is *not established*, so the default is to leave code on the branch (exp_01 precedent); (2) whether to run the **three cheap diagnostics** on existing checkpoints (future-only SSIM; the fixed-RNG one-step instrument; per-episode-vs-window-count breakdown) before committing to exp_03; (3) **exp_03 direction** — recommended: train on the trajectory the eval runs (multi-step / scheduled sampling or short-horizon rollout loss) with a predeclared metric suited to the recipe.

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
