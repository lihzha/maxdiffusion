# Master Experiment Tracker

Cross-experiment status index — one section per experiment/run, newest first. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`. Per-experiment detail lives in `exp_<NN>_<name>_claude/` folders per `experiment_SOP.md`; this file is the at-a-glance map.

Last updated: 2026-07-27

## exp_01 `full_ft_overfit` — full-FT overfit diagnostic

- **What:** Plain Wan TI2V **full finetune** (backbone unfrozen, **no adapter**) overfit sanity check on DROID. Diagnostic to separate **(A)** "data/loss/pipeline broken" from **(B)** "frozen-backbone + adapter optimization too hard". Explicitly **NOT** the long-term method.
- **Status:** PARTS I+II COMPLETE (2026-07-27) — Part I: train-cohort SSIM 0.197→0.787 (official). Part II (held-out): val loss monotone 0.18447→0.17885 across ckpts 2500–20000 (n=14,636/pt, deterministic per-position (t,ε)); val-clip rollout SSIM 0.7269 vs train 0.7875 — domain transfer established, memorization share unquantified. Analyses Codex-reviewed (Part I: 6 findings; Part II: 6 findings — all applied). Reports 01/02/03 (03 = val gallery with in-repo videos). **Awaiting Yixun: merge decision.**
- **Branch / worktree:** `claude-exp_01_full_ft_overfit-20260715` off `yixun-dev` @ `8258965`; worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit`. Exp-docs auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Docs:** `docs/worklogs_yixun/exp_01_full_ft_overfit_claude/` — `_yixun_query.md`, `_worklog.md`.
- **Decision rule:** overfits fast → pipeline OK, continue adapter work; can't overfit after substantial steps → debug data / loss / noise / CFG / latent alignment first.
- **Next:** Yixun approves → v6e-8 smoke (≈20 steps, storage-light) → v6e-64 fit probe → full run (10k steps, GBS 512, LR 1e-5, fresh noise, guide 1.0) → cohort validation at steps 0/2500/5000/7500/10000 (16 fixed train ordinals). Escalation on negative: §2.4 (30k resume → LR 2e-5 → fp32 `weights_dtype` control).

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
- Branch: `yixun-dev` (integration). Experiments so far: `exp_01_full_ft_overfit` (scaffolded, on branch `claude-exp_01_full_ft_overfit-20260715`). Next experiment number is `exp_02`.
