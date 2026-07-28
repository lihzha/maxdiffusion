# exp_02 `overfit100` — Worklog

Append-only lab notebook. Entry template in `experiment_SOP.md`.

## 2026-07-28T00:10:00Z — Scaffold exp_02: reserve number, branch + worktree

- **Goal** — Reserve `exp_02` and stand up bookkeeping for the text-conditioned 100-trajectory overfit experiment (Lihan/Yixun's corrected design after exp_01).
- **Change** — Created `docs/worklogs_yixun/exp_02_overfit100_claude/` with `overfit100_yixun_query.md` (Query 1 verbatim + accepted critique of exp_01 + spec + open questions) and this worklog.
- **Version Control** — Integration branch `yixun-dev` @ `1bc0030` (base_commit). Experiment branch **`claude-exp_02_overfit100-20260728`**; worktree **`/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100`**. Docs auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Command / Validation** — `git worktree add … -b claude-exp_02_overfit100-20260728 yixun-dev`. Verified `exp_02` unused (only `exp_01_full_ft_overfit_claude` exists; no exp_02 branch) — SOP never-reuse rule satisfied.
- **Result** — `scaffolded`.
- **Analysis** — SOP-role note: the Planner tier is specified as Fable 5 (max); this session is **Opus 5 (max effort)** by the user's explicit `/model` choice. Recorded as a deliberate deviation, not a silent substitution.
- **Next** — Data investigation (does `meta_json` carry the DROID language instruction? where is the raw source? how is text embedded?), then `plan_overfit100.md` → Codex plan review → user approval → cycles.
