# Master Experiment Tracker

Cross-experiment status index — one section per experiment/run, newest first. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`. Per-experiment detail lives in `exp_<NN>_<name>_claude/` folders per `experiment_SOP.md`; this file is the at-a-glance map.

Last updated: 2026-07-28

## exp_02 `overfit100` — text-conditioned 100-trajectory memorization test (ACTIVE)

- **What:** Can full-FT Wan2.2 TI2V 5B **memorize** the windows of **100 successful DROID trajectories** conditioned on (per-trajectory **language instruction** via T5 + first-frame latent)? Lihan/Yixun's corrected overfit design after exp_01 (which was full-corpus and unconditioned). Framed as a finite-set memorization test, NOT "text determines the future" (plan-review F1).
- **Status:** PLAN v4 APPROVED (2026-07-28, Query 4) after 4 Codex review rounds (F1–F7, G1–G5, H1–H2, I1 — all resolved; final verdict APPROVE-WITH-CHANGES, changes applied). Dataset-build + S1-smoke TPU jobs pre-approved conditional on dual sign-off; S2/S3 separate approvals. Cycle A (fixture extractor + manifest builder, local CPU) in progress. v1 review = REQUEST-REVISION (F1–F7: determinacy framing, eval-tooling gap, coverage/threshold, latent-layout proof, text-table memory, manifest reproducibility, staged compute) — all resolved in v2. Key probes: cached `z_i0 == z_video[:,0]` bit-identical; aligned `latent_videos/*.pt` are Ctrl-World-space `(T,4,24,40)` (unusable); aligned MP4s already 320×192@5fps matching annotations ⇒ data path **A′** (user choice): re-encode view-0 MP4s with the Wan VAE.
- **Locked decisions:** view 0 only; LR 1e-5; A′; selection = seeded, `success==1`, non-empty-instruction filter, 1-of-3 pick per episode; staged compute S1 smoke (v6e-8) → S2 10-episode gate (v6e-8, 2.5k steps) → S3 100-episode (v6e-64, GBS 256, 2.5k-step first segment); success rule ≥90% of canonical windows at SSIM ≥0.95 with text-ablation controls (correct/null/shuffled).
- **Branch / worktree:** `claude-exp_02_overfit100-20260728` off `yixun-dev` @ `1bc0030`, **plus exp_01's branch merged in** (needs its trainer/tooling; `yixun-dev` stays clean of exp_01 code per the no-merge decision); worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100`.
- **Docs:** `docs/worklogs_yixun/exp_02_overfit100_claude/` — queries 1–3, plan v2, plan review + resolutions, worklog.
- **Next:** re-review verdict → Yixun approves plan → cycle A (manifest builder; manifest committed) → B (dataset build, TPU-gated) → C (trainer) → D (eval tooling) → E (staged launches, all approval-gated).

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
