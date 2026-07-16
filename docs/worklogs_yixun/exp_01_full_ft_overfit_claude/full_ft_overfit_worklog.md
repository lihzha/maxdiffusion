# exp_01 `full_ft_overfit` — Worklog

Append-only lab notebook (one entry per action) for the plain-Wan-TI2V full-finetune overfit diagnostic. Newest entries at the bottom. Entry template in `experiment_SOP.md`.

## 2026-07-16T02:15:00Z — Scaffold exp_01: reserve number, create branch + worktree

- **Goal** — Reserve `exp_01` and stand up the experiment bookkeeping per the SOP, so planning can begin from a committed scaffold.
- **Change** — Created `docs/worklogs_yixun/exp_01_full_ft_overfit_claude/` with `full_ft_overfit_yixun_query.md` (driving intent from Lihan, relayed by Yixun) and this worklog.
- **Version Control** — Integration branch `yixun-dev` @ `8258965` (`base_commit`). Experiment branch **`claude-exp_01_full_ft_overfit-20260715`** created off `yixun-dev`. Worktree absolute path (SOP isolation rule 1): **`/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit`**. All experiment commits land on the experiment branch; the `exp_01_full_ft_overfit_claude/` docs auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Command / Validation** — `git worktree add /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit -b claude-exp_01_full_ft_overfit-20260715 yixun-dev`. Verified before reserving: no `exp_*` folder on `yixun-dev`, no `exp` branch, single worktree — `exp_01` never used (SOP: never reuse a number).
- **Result** — `scaffolded` — folder + query + worklog created; number reserved.
- **Analysis** — n/a (bookkeeping; no experiment signal yet).
- **Next** — Planner writes `plan_full_ft_overfit.md` resolving the four open design questions in the query doc (trainer path, conditioning, overfit subset/steps/LR, success metric). Then Codex `gpt-5.6-sol` xhigh reviews it → `full_ft_overfit_codex_plan_review.md` → Planner resolves findings → user approval. **NOTE:** per the SOP the Planner is Fable 5 (max); this session is currently Opus 4.8 — switch to Fable 5 for the plan, or confirm proceeding on Opus.

## 2026-07-16T02:50:00Z — Design locked (Query 2) + plan drafted

- **Goal** — Turn Lihan's intent + Yixun's design decisions into the reviewable plan.
- **Change** — Appended Query 2 (verbatim decisions: Planner=Fable 5; no adapter, full backbone unfreeze; conditioning = first-frame + video only, no actions; full DROID) to `_yixun_query.md`. Wrote `plan_full_ft_overfit.md` (draft for review).
- **Version Control** — exp branch `claude-exp_01_full_ft_overfit-20260715`; worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit`; base `2dd8cab`.
- **Command / Validation** — Code reading (Planner, no code written): side-adapter trainer, config yml, launch scripts, generate script, `max_utils.create_optimizer`, `disable_training_weights` grep. Key findings baked into the plan: (1) loss already pins frame 0 + uses null-text — probe = drop adapter/action path, plain transformer forward; (2) **CFG trap**: with no adapter, guide_scale=5.0 degenerates to a silent ×5 gradient scale → probe must assert guide_scale=1.0 (plan §2.1); (3) side-adapter `_shard_state` replicates params/opt_state (fine for 128M adapters, fatal for 5B) → full-FT trainer keeps FSDP shardings; (4) `disable_training_weights` only read by the BaseWanTrainer family — inert for self-contained trainers.
- **Result** — `in_progress` — plan drafted, Codex review being invoked next.
- **Analysis** — Subclassing the side-adapter trainer maximizes parity (same iterator/sigmas/noise/mask by imported code, not re-implementation), which is the whole point of a diagnostic.
- **Next** — Briefed Codex `gpt-5.6-sol` xhigh plan review → `full_ft_overfit_codex_plan_review.md` → resolve findings → re-review if material → surface to Yixun for approval.
