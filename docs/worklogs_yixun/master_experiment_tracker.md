# Master Experiment Tracker

Cross-experiment status index — one section per experiment/run, newest first. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`. Per-experiment detail lives in `exp_<NN>_<name>_claude/` folders per `experiment_SOP.md`; this file is the at-a-glance map.

Last updated: 2026-07-31

## exp_02 `overfit100` — text-conditioned 100-trajectory memorization test (ACTIVE — awaiting final S3 evals)

- **What:** Can full-FT Wan2.2 TI2V 5B **memorize** the windows of **100 successful DROID trajectories** conditioned on (per-trajectory **language instruction** via T5 + first-frame latent)? Lihan/Yixun's corrected overfit design after exp_01 (which was full-corpus and unconditioned). Framed as a finite-set memorization test, NOT "text determines the future" (plan-review F1).
- **Status (2026-07-31): all code + S1 + S2 + S3 training + S3 intermediate evals DONE; two verdict-bearing evals in flight.**
  - **Code:** cycles A–D closed (~18 Codex review rounds, every finding resolved on record); suite **1,021 passed / 2 skipped**. Latest commit `9c26070` (ffmpeg-ensure in eval launcher + aux-degradation warning) — focused Codex review pending.
  - **Dataset:** published & `_SUCCESS`-verified at `gs://v6_east1d/datasets/exp02_overfit100/` — train100 (1,629 windows / 7 shards) + train10 (167/1), build_commit 319ed93; probe PASSED (V1 r≈0.995, V3 ceilings 0.902–0.966, V4 bitwise-exact); V2 envelope recalibrated (std ∈ [0.25,1.25], |mean| ≤ 0.30) after ep3905's legit high-contrast tail; full build 642.9 s.
  - **S1 smoke:** PASSED (1.82 steps/s, v6e-8).
  - **S2 10-episode gate** (run `wan-overfit100-s2gate-20260730`): loss 0.533→**0.061** over 2.5k steps (well below exp_01 full-corpus floor 0.176). Gate evals: m_corr mean **0.7665→0.8896** monotone across ckpts, all 10 windows; formal gate (ii) growth 0.123 < 0.15 — missed by 0.027, analyzed as rule miscalibration (start already at 0.7665); ablations ordered correctly (correct 0.8895 > null 0.8398 > shuffled 0.8342 ⇒ real text use). Yixun **Query 7 = "A"**: proceed to S3 despite formal gate miss (approves S3 training + all its D11 evals incl. full-set pass).
  - **S3 100-episode** (run `wan-overfit100-s3-20260730`, v6e-64 GBS 256): training PASSED, loss 0.586→**0.145** over 2.5k steps (~390 epochs); 5 checkpoints {250, 500, 1000, 1750, 2500} at `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100/wan-overfit100-s3-20260730/checkpoints/`. Intermediate evals (seed-0, canonical-100): mean SSIM **0.758→0.771→0.789→0.802** (steps 250→1750), monotone, all role_ok.
  - **In flight:** step-2500 **segment-final** (3 seeds × 3 modes × 100 windows = 900 rollouts; job `20260731-160907-6359b989-exp02-o100-s3ev-final2500-yixun`) and **full-set** (1,629 windows seed-0 correct; job `20260731-160912-184642ed-exp02-o100-s3ev-fullset-yixun`) — both RUNNING on attempt 8 amid heavy spot preemption (attempts 1–7 all infra kills, 5–35 min each); queue auto-retries; artifacts immutable so retries are safe.
  - **Known gap (non-blocking):** aux/VAE-ceiling rows failed on S3 intermediates (`FileNotFoundError: 'ffmpeg'`) — ffmpeg-ensure existed only in the *build* launcher, not `validate_wan_overfit100.sh`. Fix committed `9c26070` (review pending). Verdict needs no aux data; ceilings recoverable later via one cheap checkpoint-independent job.
- **Locked decisions:** view 0 only; LR 1e-5; data path A′ (re-encode aligned view-0 MP4s with Wan VAE); selection = seeded, `success==1`, non-empty-instruction filter, 1-of-3 pick per episode; staged compute S1→S2→S3; success rule = two-tier D11: headline ≥90% of 100 canonical windows at m_corr (median over seeds 0–2, correct mode) ≥ 0.95; full-set tier = all 1,629 windows seed-0 pass at c*; text-ablation controls correct/null/shuffled (shuffled = value-derangement). Fail-closed verdict pipeline: manifest-derived cohorts, role-validated grids on exact (window, checkpoint, seed, mode) tuples, whole-artifact admission, `_SUCCESS`-authoritative publication, manifest_sha256 binding chain.
- **Branch / worktree:** `claude-exp_02_overfit100-20260728` @ `9c26070` (exp_01's branch merged in at start; `yixun-dev` stays clean of exp code); worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100`. Docs auto-sync to `yixun-dev` via post-commit hook.
- **Docs:** `docs/worklogs_yixun/exp_02_overfit100_claude/` — queries 1–7, plan v4, manifest (100 eps / 1,629 windows, sha256 c02a67be…), worklog, `_command.md` (Jobs 1–22+), `_results.md` (S2), `overfit100_s2_gate_artifacts/`, 4 plan reviews + ~14 code-review records with strengthening records.
- **Next:** (1) focused Codex review of `9c26070`; (2) both big evals land → pull `step_002500_s3_segment_final/aggregation.json` + `step_002500_s3_full_set/aggregation.json` → run verdict CLI (`overfit100_success_statistic`) → two-tier claim; (3) `_analysis.md` + Codex analysis review + HTML reports; (4) optional ceilings backfill job; (5) Yixun decisions: extend S3 beyond 2.5k steps (loss still falling; needs new approval per D10) and merge-or-not.

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
