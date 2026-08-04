# exp_04 `null_adapter` — worklog (append-only lab notebook)

Experiment: port null-text inversion (Mokady et al. 2022) to maxdiffusion JAX against the frozen Wan2.2 TI2V 5B backbone, compute per-step null embeddings for side-adapter-dataset examples, train an action-conditioned adapter on them, and visualize reconstruction/rollout quality.

Worktree: `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter`
Branch: `claude-exp_04_null_adapter-20260803` (off `yixun-dev` @ `744094a`)
Primary agent: claude (Planner: Claude Fable 5 max; Coder: Opus 5 max subagent; Reviewer: Codex `gpt-5.6-sol` xhigh)

## 2026-08-04T02:48:06Z — Scaffold exp_04

- **Goal** — Reserve experiment number 04, create branch + worktree + docs folder + query doc per SOP, before any planning.
- **Change** — New folder `docs/worklogs_yixun/exp_04_null_adapter_claude/` with `null_adapter_yixun_query.md` (Query 1 verbatim + summary/hypothesis/why) and this worklog. No source code touched.
- **Version Control** — branch `claude-exp_04_null_adapter-20260803`, base_commit `744094a` (= `yixun-dev` tip, pushed to origin before branching). Worktree added at `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter`.
- **Command / Validation** — `git worktree add /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter -b claude-exp_04_null_adapter-20260803 yixun-dev`. Docs will auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Result** — `launched` (scaffold committed with this entry's commit).
- **Analysis** — Tracker confirmed next number is exp_04; no competing worktree holds it. exp_03 (S1.5 probe) remains active in its own worktree — untouched by this experiment.
- **Next** — Two Explore agents (launched from the main session) map (a) the maxdiffusion side-adapter/rollout/CFG/data path and (b) the existing PyTorch inversion→adaptor pipeline in `third_party/Wan2.2/scripts/`. Then Planner writes `plan_null_adapter.md`; Codex plan review; resolutions; surface to Yixun for approval. No TPU launch without explicit approval (announcement 02).

## 2026-08-04T03:35:00Z — Context exploration complete; plan v1 written

- **Goal** — Load all context needed to plan: reference scripts (read in full), the Mokady paper, the maxdiffusion side-adapter/rollout path, and the PyTorch fork's prior inversion/adaptor line.
- **Result** — `passed`. Two Explore-agent reports landed (main session). Load-bearing facts:
  1. maxdiffusion side-adapter path uses **T5("") as BOTH CFG branches** (`wan_ti2v_side_adapter_trainer.py:313-325`); `side_adapter_guide_scale: 5.0`; CFG amplifies the adapter delta. No text conditioning exists.
  2. All sampler machinery reusable on `yixun-dev`: `build_rollout_sigmas` / `apply_first_frame_pin` / `_build_per_token_timestep` (`side_adapter_wan.py:814/528/518`), Euler update + JAX VAE decode + SSIM/video helpers in `generate_wan_side_adapter.py`.
  3. **No inversion/null-text code exists anywhere in this repo (any branch)** — from-zero build.
  4. PyTorch fork: DROID inversion line used **positive_inversion only**; `null_inversion` never run on DROID; no network ever trained to predict null embeddings. Central negative finding = **noise-basin problem** (own-z_init latent MSE 0.015–0.022 vs fresh-noise 2.0–3.3; `Wan2.2/docs/adaptor_design.md` §3.11–3.14, `WORKLOG.md:943-949`); also CFG-consistency failures (train w=1 vs eval w=5) and `sample≈null` underfitting of the positive adaptor. Sigma grids: PyTorch starts σ_max=0.999 vs maxdiffusion 1.0 — small documented deviation, acceptable (no cross-repo artifacts).
- **Change** — `plan_null_adapter.md` v1 written (Planner): 4-phase design — P0 TDD infra (tiny-model smoke port), P1 capacity study + basin probe (arms A0/A1/A1-probe/A2 fixed-noise-nulls/A3 direct-opt), P2 target caching, P3 action-conditioned null-embedding adapter (DiT-free regression) + same-cohort eval vs pre_context baseline, P4 reports. Gates G1/G3 predeclared; 5 decision points for Yixun.
- **Version Control** — branch `claude-exp_04_null_adapter-20260803`; plan committed with this entry.
- **Next** — Codex plan review (briefed: SOP, query doc, plan, this worklog, prior exp analyses, reference scripts), then Planner resolutions, re-review if materially revised, then surface to Yixun.
