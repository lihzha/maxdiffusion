# exp_05 `pos_context` — worklog (append-only lab notebook)

Experiment: per-step positive text embeddings from DDIM inversion (positive_inversion port to JAX) + the existing pre_context adapter structure trained by regression onto those embeddings; parallel sibling of exp_04 (null slot), sharing its reviewed infrastructure.

Worktree: `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context`
Branch: `claude-exp_05_pos_context-20260804` (off `yixun-dev` @ `695d410`)
Primary agent: claude (Planner: Claude Fable 5 max; Coder: Opus 5 max subagent; Reviewer: Codex `gpt-5.6-sol` xhigh)

## 2026-08-04T15:25:00Z — Scaffold exp_05

- **Goal** — Reserve experiment number 05 per SOP (tracker said next is exp_04; exp_04 took it earlier today, so exp_05 is next), create branch/worktree/docs before planning.
- **Change** — `docs/worklogs_yixun/exp_05_pos_context_claude/` with `pos_context_yixun_query.md` (Query 1 verbatim + summary/hypothesis/why) and this worklog.
- **Version Control** — branch `claude-exp_05_pos_context-20260804`, base_commit `695d410` (`yixun-dev` tip incl. all exp_04 plan docs). Worktree added.
- **Command / Validation** — `git worktree add /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context -b claude-exp_05_pos_context-20260804 yixun-dev`.
- **Result** — `launched` (committed with this entry).
- **Analysis** — exp_05 deliberately shares exp_04's reviewed contracts (manifests, gates, noise conventions, artifact integrity, evaluator) — the plan will bind to `exp_04_null_adapter_claude/plan_null_adapter.md` v5 by reference and specify only deltas + exp_05-specific parts. Code reuse via merging exp_04's branch at its shared-core boundary (precedent: exp_03 merged exp_02's branch).
- **Next** — Planner writes `plan_pos_context.md`; Codex plan review; resolutions; surface to Yixun. exp_04 implementation (Coder R1) proceeds in parallel in its own worktree.
