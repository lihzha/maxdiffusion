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

## 2026-08-04T16:15:00Z — Plan review dispatch FAILED: Codex usage limit (issue #9 recurrence)

- **Goal** — Run the exp_05 plan review.
- **Result** — `partial`: `codex exec` returned "You've hit your usage limit … try again at Aug 7th, 2026 11:35 PM" before producing any review. The previous entry's "review dispatched" is corrected: dispatched but FAILED at the provider.
- **Analysis** — Infrastructure (reviewer quota), not a bug. exp_04's five plan passes consumed the purchased credits. Per the SOP's no-silent-substitution rule the review gate is blocked pending Yixun's choice: buy credits / wait for Aug 7 reset / explicitly approve a substitute reviewer (would be recorded in every affected review file). Logged as issue #9 recurrence in `issue_report.md` (committed on yixun-dev @ 167d046).
- **Next** — Await Yixun's decision. exp_04 Coder R1 write phase continues in parallel (its review gate is equally blocked once the write lands).

## 2026-08-04T18:05:00Z — Plan review v1: REQUEST-REVISION (7 MAJOR + 2 MINOR) → plan v2

- **Goal** — Independent review of plan v1 (dispatched after the credit refill).
- **Result** — `passed` (review obtained): **REQUEST-REVISION**. The decisive finding (F1): v1 optimized text tokens inside a 512-row padded context, but the deployed `wan_pre_context_adapter_forward` passes the head's [B,8,4096] output DIRECTLY as the whole context (seq len 8, no padding-mask on this path) — targets and deployment were different representations. Also: decidable overfit/closed-loop rules (F2), trainer feasibility contract (F3), commit-pinned exp_04 dependency matrix (F4), fold-option removal (F5), no-mutation-of-exp_04 + S1–S10 round split (F6), z_bar_states fidelity gate + corrected ~17.1 GiB storage (F7), achieved-quality framing (F8), per-token-steering claim dropped (F9).
- **Analysis** — All 9 accepted. F1 would have invalidated P3a and every oracle comparison — exactly the class of error the review exists to catch. Consequences absorbed: pivots not shared with exp_04, CFG active at init, K1 fully self-contained.
- **Change** — `plan_pos_context.md` rewritten as v2; resolutions appended to `pos_context_codex_plan_review.md`.
- **Version Control** — committed with this entry; re-review pass 2 dispatched.
- **Next** — Pass-2 verdict → surface to Yixun (decision points now: L_pos=8; pure-regression primary; K1 conditional approval; pilot scope).

## 2026-08-04T19:15:00Z — Re-review pass 2: near-converged (F1–F9: 8 RESOLVED + F4 partial → G1) → plan v3

- **Goal** — Verify v2 closures; screen the delta.
- **Result** — `passed`: F1–F3, F5–F9 RESOLVED (incl. the deployment-matching context convention verified against both implementations); F4 PARTIALLY-RESOLVED via **G1 (MAJOR)**: the shared-file merge rule ("exp_04's side verbatim") could erase exp_05's `train_wan.py` dispatch at merge-2 — internally inconsistent with the combined-green requirement. Storage note: reviewer confirms 17.1 GiB is conservative (raw ≈ 14.8 GiB).
- **Change** — plan v3: three-class conflict policy with additive-union rule for enumerated dual-touch files, reviewed merge commits, both-dispatch-routes post-merge test requirement.
- **Version Control** — committed with this entry; pass-3 (delta-only) dispatched.
- **Next** — Pass-3 verdict (expected APPROVE-PLAN) → surface to Yixun.

## 2026-08-04T19:45:00Z — Re-review pass 3: APPROVE-PLAN — exp_05 plan cycle closed; surfaced to Yixun

- **Goal** — Verify G1 closure.
- **Result** — `passed`: **APPROVE-PLAN** ("G1 is fully closed, and the v2→v3 delta is clean"). Three-pass trail complete; all findings accepted and implemented.
- **Next** — Surface the approval package to Yixun: plan v3, review trail, decision points (L_pos=8; pure-regression primary; K1 conditional approval; pilot scope). Implementation S1 waits on exp_04's R9 boundary (merge-1) per the dependency matrix — exp_04 is at R2 in flight.
