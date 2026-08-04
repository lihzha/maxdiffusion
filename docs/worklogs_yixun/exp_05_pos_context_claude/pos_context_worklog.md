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

## 2026-08-04T15:55:00Z — Plan v1 written (delta-based on exp_04 v5)

- **Goal** — Planner plan for the positive-slot experiment, binding shared contracts to exp_04's five-pass-approved plan by reference; only deltas specified.
- **Change** — `plan_pos_context.md` v1: positive optimization (branch swap of exp_04's method, per `optimize_positive_embeddings`), L_pos=8 aligned to `pre_context_tokens`, per-step state caching for teacher-forced regression, B-arms mirroring exp_04's with identical gate forms (H1/H2/H3 ≡ G1/G2/G3), the EXISTING `NNXWanSideAdapterStack` pre_context configuration as the model (Q3' isolates training signal vs architecture), regression trainer touching only block-0 of the frozen transformer, closed-loop eval through the existing pre_context rollout structure via exp_04's evaluator, jobs K1–K4 (+shared J5), S1–S4 Coder rounds gated on exp_04's shared core (branch merge at the R8 boundary).
- **Version Control** — committed with this entry; Codex plan review dispatched (reviewer instructed that exp_04 v5 contracts are settled — review the deltas and their consistency).
- **Next** — Review cycle → resolutions → surface to Yixun. In parallel: exp_04 Coder R1 running in its own worktree.
