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

## 2026-08-05T05:05:00Z — merge-1 executed (exp_04 R9 boundary); S1 dispatched

- **Goal** — Bring exp_04's shared core (R1–R9, 610 tests) onto the exp_05 branch per the plan §6 dependency matrix.
- **Version Control** — merge commit `5f4e487`: one-way `claude-exp_04_null_adapter-20260803` @ `6fd18fc` → this branch. Conflicts: two add/add in exp_04-OWNED docs (worklog, plan — both branches carried copies via the yixun-dev sync history) — resolved class-(a), exp_04's side verbatim; NO dual-touch (class-c) files involved ⇒ no merge review required per the policy. Post-merge acceptance: **full combined suite 610 passed in 39.9s** in this worktree.
- **Result** — `passed`. exp_05 implementation unblocked; S1 `truncate-pad-parity` dispatched to the Coder.
- **Analysis** — K1's remaining conditions: P0' green (S-rounds), exp_05 parity audit, J0 published (currently blocked on an ADC reauth on the exp_04 side — infra, issue #6 class).
- **Next** — S1 write → review → strengthen → commit.

## 2026-08-05T06:10:00Z — S1 `truncate-pad-parity` write phase complete; review dispatched

- **Goal** — S1: the deployment-matching 8-token context construction + the two parity tests.
- **Change** — `pos_context_inversion_wan.py` (32 exec LOC construction layer; 223 total with the parity harness) + `test_pos_context_truncate_pad.py` (33 tests). exp_04's modules untouched (verified).
- **Command / Validation** — red evidenced; **643 passed** (610 inherited + 33); ruff/black/py_compile clean. **12 mutants, 0 survivors** (M6 equal-length fp32-laundering was a real gap, closed; M11 kills removal of the deployed dtype cast; M12 kills a forced re-projection).
- **Result** — `passed` (write). **Three load-bearing findings:** (1) the two call idioms are literally the same call at inference values — bitwise parity confirmed and documented as the guarantee; (2) **the deployed path casts C to the activation dtype (`side_adapter_wan.py:767`) — at bf16 this is NOT a no-op (max |Δ| 3.1e-2 measured); S3's replay operator MUST cast C to the transformer activation dtype — pinned by test, carried as an S3 contract**; (3) `frame_positions` is dropped from the deployed final re-run (`:768-774` vs the feature path) — harmless at frame_positions=None (exp_05's entire scope) but added to the deviations register; any fix edits a shared deployed file ⇒ a decision, never silent. Bonus: cross-attention row-permutation invariance measured (9.5e-7 vs 3.4 for re-projection), empirically corroborating plan F9. **Planner acceptances:** l_pos kwarg on the constructor (ablation uses one constructor); LOC overage (parity harness); rank-3-unit squeeze; two extra fail-closed checks.
- **Next** — S1 review (exp_05's first code review) → strengthen → commit → S2 `optimize-positives`.

## 2026-08-05T08:20:00Z — S1 cycle CLOSED: review (1 MAJOR + 1 MINOR; three findings verified) → strengthen (649 green, 0/16) → commit

- **Goal** — Close S1.
- **Command / Validation** — Review + strengthening in `pos_context_codex_code_truncate-pad-parity_review.md`; all three S1 findings independently verified by the reviewer with measurements; the bf16-cast S3 contract and the seam-vs-operator scope note both pinned in-code for S3's Coder. **649 passed**; 16 mutants, 0 survivors. (Process note: three transient API drops during strengthening — infra, work resumed each time with no loss.)
- **Result** — `passed`. S1 committed with this entry.
- **Next** — S2 `optimize-positives`: the branch-swap optimizer (cached v_uncond, grads through the 8-token v_cond, per-step state recording).

## 2026-08-05T09:00:00Z — Coder handoff: fresh agent for S2 after repeated API drops

- **Goal** — Continue S2 after the S1 Coder agent dropped on API connection errors three times consecutively with no S2 work landed (transcript ~180k tokens; the drops followed this agent while other sessions ran normally).
- **Analysis** — Infrastructure. Worktree verified: only S1's committed state, no S2 files. Mitigation per the R6-handoff precedent: retire the agent; fresh Opus Coder with a self-contained S2 brief (conventions live in the committed code + review files).
- **Result** — `fix_ready`.
- **Next** — S2 write → review → strengthen → commit.

## 2026-08-05T12:10:00Z — S2 `optimize-positives` write phase complete; review dispatched

- **Goal** — S2: the branch-swap per-step positive-context optimizer.
- **Change** — `optimize_positive_embeddings` (+99 exec LOC in `pos_context_inversion_wan.py`) + `test_pos_context_optimize.py` (39 tests). exp_04 modules untouched (private-import reuse of `_checked_velocity`/`_validate_sigmas` — one owner for the guards).
- **Command / Validation** — red evidenced; **688 passed** (649 inherited + 39, re-confirmed post-battery on diff-identical source); ruff/py_compile/diff-check clean. **11 mutants, 0 survivors** — incl. C-in-uncond-slot (killed by the w=1 nonzero-grad property test: the positive slot's distinguishing physics vs the null slot) and the cast-seam violation.
- **Result** — `passed` (write). **THE CAST-SEAM DECISION (Planner-endorsed): `velocity_fn` owns the activation-dtype cast**, the operator hands it bit-exact fp32 C — one owner across optimize/replay/deployment (deployment's cast lives at side_adapter_wan.py:767; the reference's lives in `_dit_velocity`'s autocast); **S1's "S3 must cast" obligation formally moves to the S3/S4 runner-built velocity_fn**, stated in the implementation docstring as the wiring contract. **Planner acceptances:** `pos_init=None` kwarg (required by the composition test; single entry point for the L_pos ablation; mirrors both references); relative dim validation with production geometry at the runner boundary (exp_04 precedent); test-side LOC overage (the literal reference, S1 precedent). Confirmed on the real backbone: nonzero C-grads at both w=5 AND w=1 (tiny-WanModel smoke). Process: multiple API drops + one killed battery process (mutant M5 detected on disk, restored from backup, battery re-run) — all infra, all recovered.
- **Next** — S2 review → strengthen → commit → S3 `replay-state-capture`.

## 2026-08-05T15:10:00Z — CORRECTION (append-only): the S1 cast obligation, restated

- The S1 entries said "S3's replay operator MUST cast C to the transformer activation dtype." Per the S2 cast-seam decision (reviewer-ratified as architecturally sound): **the single rule is that the optimizer and the replay operator pass fp32 context unchanged, and the runner-built real-backbone `velocity_fn` performs the cast immediately before the transformer** (matching the deployed path's cast at `side_adapter_wan.py:767` and the reference's autocast placement). The substance of S1's finding is unchanged — no fp32 C may reach the frozen transformer — but the obligation's owner is the S3/S4 runner wiring, not `replay_with_positive`. Carried contracts: S3 re-runs the parity fixture through the actual replay operator; S4 adds a bf16 both-branches closure test.

## 2026-08-05T15:45:00Z — S2 cycle CLOSED: review (2 MAJOR; core empirically confirmed) → strengthen (689 green, 0/12) → commit

- **Goal** — Close S2.
- **Command / Validation** — Trail in `pos_context_codex_code_optimize-positives_review.md`. **689 passed**; 12 mutants, 0 survivors. The K2-facing state schema is now explicit (`z_bar_states [N]` + `z_final`), and the cast rule is single-sourced with MUST contracts for S3/S4.
- **Result** — `passed`. S2 committed with this entry.
- **Next** — S3 `replay-state-capture`.
