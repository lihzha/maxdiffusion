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

## 2026-08-05T15:55:00Z — S3 `replay-state-capture` write phase complete; review dispatched

- **Goal** — S3: the positive-slot replay operator + the first MUST contract.
- **Change** — `replay_with_positive` (+58 exec LOC) + `test_pos_context_replay.py` (26 tests).
- **Command / Validation** — red evidenced; **715 passed** (689 + 26); ruff/py_compile/diff-check clean. **7 mutants, 0 survivors** — incl. N6 (operator-casts violation) killed three ways, and N7 (hoisted v_unc) pinning the round's freshness-symmetry claim.
- **Result** — `passed` (write). **MUST-1 DISCHARGED:** S1's parity fixture re-run through the ACTUAL operator (bitwise at fp32; operator+casting-velocity_fn ≡ deployment bitwise at bf16; non-casting closure demonstrably fails). **B0 semantics pinned as a deliberate inversion of exp_04's A0 test:** frozen-C ACTIVE-CFG control, w=1-vs-w=5 structural separation measured at 7.9e+1. Planner acceptances: test-side LOC overage (precedent); [N,1,L,D]-at-B>1 rejected not broadcast (R4a lesson); the extra N7 mutant. S4's MUST (runner closure casts both branches at bf16) remains open, recorded.
- **Next** — S3 review → strengthen → commit → S4 `runner-slot-arms`.

## 2026-08-05T16:40:00Z — S3 cycle CLOSED (715 green, 0/8) → commit; SEQUENCING AMENDMENT: S5 before S4

- **Goal** — Close S3; set the next round.
- **Command / Validation** — Trail in `pos_context_codex_code_replay-state-capture_review.md`; MUST-1 discharged and independently verified; H1 interpretation note added to the plan (scope-neutral). **715 passed**; 8 mutants, 0 survivors.
- **Result** — `passed`. S3 committed with this entry.
- **Analysis** — **Sequencing amendment (scope-neutral, Planner):** the plan's dependency matrix placed S4 `runner-slot-arms` in the merge-1-only block, but the runner file (`run_wan_null_inversion.py`) was CREATED by exp_04's R10 — which is in its strengthen cycle right now. S4 therefore factually depends on exp_04-R10 closing + an interim one-way merge (same policy as merge-1/2). **S5 `schema-states-fidelity` proceeds first** (own pos-records module importing exp_04 primitives; no runner dependency); S4 follows the merge. Matrix corrected in spirit; plan text untouched beyond this record.
- **Next** — S5 write → review → strengthen → commit; S4 after exp_04-R10 + merge.

## 2026-08-05T19:40:00Z — S5 write phase complete: THE F7 PREMISE IS FALSE (measured); plan expectation amended; review dispatched

- **Goal** — S5: the positive-slot record schema + states fidelity policy.
- **Change** — `pos_context_records.py` (258 exec LOC, numpy-only, `_PosGeometry` rebuilt locally per F6 with two extra invariants; schema-independent primitives imported from the sibling) + `test_pos_context_records.py` (34 tests). **Plan §4-P2' amended (scope-neutral):** the S5 measurement disproved the fp16→bf16 value-preservation premise — 6.2% of latent-like elements diverge by exactly 1 bf16 ulp via double rounding (witness constructed), so the feature-tolerance path is load-bearing and the gate conservatively selects fp32 absent feature deltas; storage corrected to exact 7.12 MiB/record / 14.80 GiB fp16.
- **Command / Validation** — red evidenced; **749 passed** (715 + 34); ruff/py_compile clean. **8 mutants: 7 killed, 1 proven equivalent** (P7 — geometry enforced at two layers; investigated with a direct probe, the stronger layer kept, the misleading comment corrected).
- **Result** — `passed` (write). **Open item flagged for S4:** whether the runner reuses exp_04's ProvenanceHeader (carrying l_null) or needs an l_pos variant — a runner-level decision. `ml_dtypes` used (numpy-family, not jax) with a bitwise pin against jnp.bfloat16.
- **Next** — S5 review → strengthen → commit. S4 still gated on exp_04-R10 + merge.

## 2026-08-05T22:10:00Z — S5 cycle CLOSED (769 green, 0/15) → commit; merge-interim (exp_04 R10 boundary) next

- **Goal** — Close S5; unblock S4.
- **Command / Validation** — Trail in `pos_context_codex_code_schema-states-fidelity_review.md`; the F7 science is now stated at verified precision in plan + module + tests. **769 passed**; 15 mutants, 0 survivors.
- **Result** — `passed`. S5 committed with this entry. **The K2-facing storage layer is done**: schema, integrity, fidelity policy, exact storage arithmetic.
- **Next** — merge-interim: one-way exp_04 (R10 boundary, 2ebaaad+) → exp_05 per the amended matrix; then S4 `runner-slot-arms`.

## 2026-08-05T22:40:00Z — merge-interim executed (exp_04 R10 boundary); S4 dispatched

- **Version Control** — merge commit `8695fac`: one-way exp_04 @ `cdd4653` → this branch. One content conflict (exp_04's own commit ledger — class (a), exp_04's side). Post-merge acceptance: **full combined suite 1003 passed in 90s** (exp_04's 844 ⊕ exp_05's additions).
- **Result** — `passed`. S4 unblocked: the runner/modes files exp_05 extends now exist here. S4's edits to `run_wan_null_inversion.py`/`null_adapter_modes.py` are class-(c) dual-touch — they MUST be additive (the merge-2 additive-union rule depends on it), and the ProvenanceHeader l_pos decision lands in S4 per the S5 flag.
- **Next** — S4 write → review → strengthen → commit.

## 2026-08-06T00:10:00Z — S4 `runner-slot-arms` write phase complete; review dispatched

- **Goal** — S4: the runner's positive-slot extension under strict additivity.
- **Change** — new `pos_context_modes.py` (312 exec LOC: B-arm batch runner, pos-record builder, positive capacity body) + `PosProvenanceHeader` in `pos_context_records.py` (+86) + **exactly 24 insertions / 0 deletions** in the dual-touch `run_wan_null_inversion.py` (`null_adapter_modes.py` untouched — the positive path is a sibling module, shrinking the merge-2 surface to one file) + 11 tests.
- **Command / Validation** — red evidenced; **1014 passed** (1003 + 11); ruff clean. **8 mutants, 0 survivors** — incl. R2 (inversion at the 512 context — the pivots-differ pin) and R3a/R3b (cast dropped / one-branch-only — the MUST closure test).
- **Result** — `passed` (write). **THE l_pos DECISION (Planner-endorsed):** pos-specific `PosProvenanceHeader` with `l_pos` + explicit `embedding_slot: "positive"` — no l_null reuse, no rename; every other field matches exp_04's for diffability. **S4 MUST DISCHARGED** — the cast rule is closed end-to-end (S2/S3: operators pass fp32; S4: the wiring casts both branches, proven at bf16 on a real tiny model). **Additivity proven** by diff-numstat + the field-by-field null-path characterization (mutant-backed). **Open item for the Planner:** no positive-slot shard WRITER exists (exp_04's write_shard hardcodes the null codec; F6 forbids editing it) — pos_default_sinks fails closed on record publishing; a `pos_write_shard` mirroring the R8 storage discipline on the S5 codec is required before K1 publishes records → scheduled as round S4b. Comparison videos deferred (K1's decisions ride on tables/selection/report). pos_execute wires capacity only (others raise, pinned).
- **Next** — S4 review → strengthen → commit → S4b `pos-shard-writer`.

## 2026-08-06T01:20:00Z — S4 strengthen sitting 1 complete (launch-safety core); sitting 2 dispatched

- **Result** — `partial` (as instructed — honest split): findings 1 (semantic additivity — real-HyperParameters resolution + the new `base_wan_5b_pos_inversion.yml` + AST-extracted-class characterization), 5 (S4b folded in — `pos_write_shard` on the S5 codec with the full R8 discipline; run-JSONs publish only after all shards; a failing writer leaves ZERO artifacts), and 6 (cast wiring main-observed) FIXED; finding 4 partially (l_pos threaded to all four consumers; adequacy/ablation runner pending); finding 3 partially (slot roots isolated + probed; selection payload pending); finding 2 (record-provenance preflight) pending. **1028 passed**; 8/8 mutants. A latent import bug caught by the new real-filesystem writer test (invisible to fakes). Battery incident: a timeout left one mutant on disk mid-run — detected, removed, re-verified; scripts now write disk backups first.
- **Next** — Sitting 2: findings 2, 3-residue, 4-residue → follow-up review → commit.
