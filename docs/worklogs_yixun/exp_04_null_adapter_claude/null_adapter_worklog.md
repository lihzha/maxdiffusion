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
