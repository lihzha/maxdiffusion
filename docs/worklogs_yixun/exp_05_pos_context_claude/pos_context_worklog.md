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

## 2026-08-06T05:30:00Z — S4 cycle CLOSED (1063 green, 28/28 mutants) → commit — ALL K1-PATH CODE COMPLETE

- **Result** — `passed`. S4 committed with this entry (pos modes incl. the folded S4b writer, PosProvenanceHeader, the pos YAML, the slot dispatch). exp_05 suite contribution: 219 tests over S1–S5; combined 1063.
- **Analysis** — K1's code path is structurally complete: slot-isolated storage, provenance-bound records/selection/adoption, the deployment-matching 8-token convention pinned end-to-end (cast rule closed), the B-arms + adequacy + diagnostic ablation wired. Remaining K1 conditions: exp_05 parity audit (next) — merge-1 ✓, P0' green ✓ (the S-suite), J0 published ✓.
- **Next** — exp_05 PARITY AUDIT → the K1 package under the standing grant. S6+ (trainer path) proceeds after K1 is in flight.

## 2026-08-05T19:45:00Z — Timestamp correction (append-only)

- Entries above stamped `2026-08-06T*` (merge-interim through S4 close) are future-dated by one day — actual dates are 2026-08-05. Same drift as exp_04's (corrected there); content unaffected.

## 2026-08-05T20:00:00Z — PARITY AUDIT (plan §8 = exp_04 §8 inherited + §3 positive-slot deltas) — CLEAN; recorded before K1 per the SOP

Scope: the inherited exp_04 audit (recorded CLEAN 2026-08-05 in `exp_04_null_adapter_claude/null_adapter_worklog.md`) covers every shared primitive exp_05 imports rather than reimplements — `invert_trajectory`, `_checked_velocity`, `_validate_sigmas`, the Adam constants, σ grid, pin discipline, per-token timestep, noise conventions, manifests/cohorts. This audit covers the positive-slot deltas line-by-line against `third_party/Wan2.2/scripts/embedding_search.py` @ pinned `f370228`, each with its Planner-verified reference reading (done today, direct reads of :681-788, :822-853, :1150-1199) and where it is pinned:

1. **L_pos forcing** (`truncate_or_pad_context` ≡ :1181-1195): leading-row truncation kept contiguous, zero-row append in the input's dtype, bit-exact passthrough at equal length. Pinned: S1 `test_pos_context_truncate_pad.py` (incl. batch/rank rules the reference never needed); S1 review verified.
2. **Warm start** `C_init = truncate_or_pad(T5(""), 8)`: the reference takes `truncate(T5(heuristic))` (:1171, :1181-1195) — **deviation (a), ratified in the plan review (F1)**: DROID has no captions, so T5("") is the only available seed and the deployed adapter's own unconditional base. Pinned: `test_the_warm_start_is_the_truncated_t5_context_and_l_pos_is_settable`.
3. **Inversion at w=1 with the 8-token conditional context** ≡ `run_positive_inversion` (forcing at :1181-1195, then `compute_inversion_trajectory(context_list=[context_pos])` — verified at :1197+46-53): recurrence itself is exp_04's audited `invert_trajectory` **by import**; only the context changes. **Consequence (deviation d): pivots are NOT shared with exp_04** (8-token vs 512-row inversion contexts ⇒ different trajectories; K1 computes its own, never reads exp_04 artifacts). Pinned: S4 mutant R2 (inversion switched to the 512 context ⇒ the pivots-differ test kills it).
4. **Per-step positive optimization** (`optimize_positive_embeddings` ≡ :681-788), clause by clause: **(i)** the branch asymmetry — v_unc computed ONCE per outer step under no-grad (:736-741) ≡ `lax.stop_gradient` outside the inner scan (module :282); gradients flow through v_cond. Slot physics pinned by `test_guide_scale_one_keeps_the_positive_branch_in_the_graph` (C-grads NONZERO at w=1 — the exact inversion of exp_04's zero-∅-grad pin) and `test_gradients_flow_through_a_real_tiny_wan_transformer` (both guide scales, real model). **(ii)** fresh Adam per outer step (:743 inside the i-loop) ≡ `optimizer.init(pos)` inside `outer` (:300); recipe = exp_04's audited constants + `eps_root=0.0` spelled out; pinned by `test_next_step_warm_starts_from_the_locked_context_with_fresh_adam_state`. **(iii)** inner loop — fresh v_cond per iteration, CFG combine, Euler step, first-frame pin, MSE to `traj[i+1]` (:747-758) ≡ module :284-297. **(iv)** lock-and-advance — exactly one extra forward with the LOCKED context and the cached v_unc (:767-776) ≡ `z_next = euler_step(locked)` (:302). **(v)** warm-start carry — one tensor across outer steps (:703, :722) ≡ the outer-scan carry. **(vi)** `losses[i,j]` = pre-update value, the reference's logging convention (:755).
5. **Batched execution (deviation e, inherited class):** objective = SUM of per-example means, so each example's gradient is the gradient of its own loss — batch composition cannot change any example's result. Pinned: `test_the_objective_is_the_batch_sum_not_the_batch_mean`, `test_example_zero_is_independent_of_the_rest_of_the_batch`, with grad-norm assertions on UNNORMALIZED norms (exp_04 R11's Adam-scale-invariance lesson applied at write time).
6. **z̄ state capture:** reference returns `z_bar` list[N+1] (:788); the port returns the aligned `[N]` pre-step states + the terminal `z_final` split off BY NAME (S2 review finding 1 — no implicit `[:-1]` left to K2/S4 readers). Same information, safer shape; this is the K2 cache schema.
7. **Replay** (`replay_with_positive` ≡ `regenerate_with_positive_embeds` :822-853): both branches FRESH at the current latent every step (:844-849; neither hoistable — replay moves z), v_cond-first order preserved, unconditional branch frozen at full-length T5(""), CFG combine + Euler + pin (:850-852). Pinned: S3 bitwise oracle — the deployed forward IS the operator's own v_cond (fp32 bitwise; bf16 bitwise for operator + casting closure, and UNEQUAL without the cast).
8. **Deployment-matching context + the cast seam (the load-bearing §3 delta):** the conditional context is `[B, 8, 4096]` passed DIRECTLY as the whole `encoder_hidden_states` ≡ `wan_pre_context_adapter_forward` (side_adapter_wan.py:768-774); the bf16 activation cast (:767, non-no-op) is owned by exactly one component — the runner-built `casting_velocity_fn`, BOTH branches — matching deployment's placement and the reference's autocast placement (`_dit_velocity` :503-513). Pinned end-to-end: S2/S3 operators pass fp32 bit-unchanged (tests + the S3 unequal-without-cast pin); S4 closure test on the wiring; mutants R3a/R3b (cast dropped / one-branch-only) killed.
9. **B0 control physics (deviation c):** B0 (frozen `C_init` every step) is an ACTIVE-CFG control — POS_L vs S sequence lengths make v_cond ≠ v_unc, so B0 genuinely depends on w, unlike exp_04's A0 collapse-to-identity. Stated in plan §4, pinned in `test_pos_context_replay.py`.
10. **Records and fidelity:** fp16 storage with FEATURE-TOLERANCE `states_fidelity_check` (fp16→bf16 double-rounding is NOT value-preserving — ~6.25% of N(0,1) elements diverge; stated at verified precision after the S5 review's F7 correction); write-time hashes; slot-isolated roots; `PosProvenanceHeader` carries `l_pos` + `embedding_slot="positive"`; `pos_write_shard` restates every exp_04 writer rule (canonical bijection, size ceiling, staging ownership, immutable destination, data-first-marker-LAST) over the S5 codec.
11. **Deviations register (all ratified in reviews):** (a) empty warm start — T5("") for T5(heuristic); (b) 8-token DIRECT context, no 512-row splice — deployment-matching, reference-faithful via the reference's own L_pos forcing; (c) active CFG at init / B0 as matched control; (d) pivots not shared with exp_04; (e) inherited from exp_04's register verbatim: σ₀ = 1.0 vs 0.999, batched execution with per-example independence, optax-vs-torch Adam (recipe-pinned), JAX threefry noise (golden-pinned).

**Numeric-recipe defaults cross-check (SOP):** the pos YAML reuses the slot-neutral recipe keys at the audited values — J=10 (`null_inner_iters`), lr=1e-2 (`null_lr`), w=5.0 (`null_guide_scale`), inversion w=1.0, Adam (0.9, 0.999, 1e-8, eps_root 0), 25 steps, `embedding_slot: positive`, slot-isolated `pos_*` roots and URIs — declared in `base_wan_5b_pos_inversion.yml` and pinned by the S4 config tests. **Data parity:** exp_04's published J0 manifests are consumed unchanged (K1-gate provenance check against the J0 manifest hashes per plan §7-table).

**Audit corrections applied with this entry (doc-only, no executable change):** (i) `pos_context_inversion_wan.py` module docstring still carried the S2-era "S4 MUST … STILL OPEN" note — S4 discharged it (closure test + R3a/R3b mutants); rewritten to record the discharge. (ii) `pos_execute`'s docstring said "Only ``capacity`` (K1) is wired in S4" while `adequacy_probe` is also wired (its error message already names both); corrected.

**Planner finding (plan sequencing, K1-blocking, no code defect):** plan §9 places K1 after S1–S5, but the launch script `run_wan_pos_inversion.sh` is an S10 deliverable — K1 has no launch path without it (the null launcher hardcodes the null YAML, and `embedding_slot` cannot ride the null config's CLI by design). **Amendment: round S10a `pos-launcher` — `run_wan_pos_inversion.sh` + its tests, pulled forward from S10; `train_wan_pos_context.sh` stays in S10** (genuinely post-K2). K1's gate set becomes: merge-1 ✓ + P0' green ✓ + J0 published ✓ + parity audit ✓ (this entry) + **S10a committed**.

**Verdict: PARITY AUDIT CLEAN** — no numerical or structural deviation beyond the ratified register; two stale docstrings corrected in place. K1's remaining precondition is S10a.

## 2026-08-05T20:55:00Z — S10a `pos-launcher` write phase complete (Coder); review dispatched

- **Goal** — The pulled-forward K1 launch script (parity-audit amendment): `run_wan_pos_inversion.sh` + tests, mirroring the settled null launcher without touching it.
- **Change** — 2 NEW files: the launcher (331 lines; pos YAML as argv[1], `embedding_slot=positive` as a CLI literal, POS_* env surface with YAML-equal defaults, mode gate {capacity, adequacy_probe} pre-python, tee-first/preflight/prefetch/golden machinery mirrored) + `test_pos_launcher.py` (60 cases). Settled files byte-untouched.
- **Command / Validation** — red evidenced (24F/35E before the launcher existed); **1123 passed** (1063 + 60); black/ruff clean; **15 mutants, 0 survivors** with sha256-verified restore.
- **Result** — `passed` (write). **Planner positions on the Coder's judgment calls:** (1) executed-under-bash sandbox technique — NEW to the repo, endorsed (my round brief wrongly claimed exp_04 had PATH-shim launcher tests; that technique lived in exp_02's eval-launcher tests — second recorded instance of an unverified repo-state premise in a brief; lesson re-logged); (2) opt-in `POS_WATCHDOG_SECONDS` (default off) instead of importing exp_04 R11's unmerged watchdog — endorsed (no direct_opt in the positive slot; R11 is past the merge-interim boundary); (3) no-colon `${VAR-default}` for MODE/ARTIFACT_DIR/STAGING_DIR so set-but-empty fails loudly — endorsed; (4) launcher-side refusal of roots containing `exp04` — endorsed as defense-in-depth beyond `positive_roots`; (5) test file renames to `test_pos_context_launcher.py` in strengthen (round naming convention); (6) slot-neutral keys at YAML defaults exactly as exp_04's launcher — endorsed.
- **Coder flags on settled/other-branch files (recorded, not fixed):** (i) `run_wan_null_inversion.py:660-675` `mode_kwargs` reads `null_artifact_dir`/`null_selection_uri` for cache/verify_replay — harmless today (positive slot wires neither) but whoever wires positive cache/verify MUST redirect those three reads or a positive run reads null-slot shards; carried as a standing obligation for that future round (also: `POS_SELECTION_URI` is currently a forward-looking knob no wired mode consumes). (ii) The settled null launcher's `set +x; source secrets.env; set -x` force-enables xtrace on real TPU hosts, spraying later expansions into teed logs — exp_04-owned, logged as issue #12. (iii) exp_04 R11's watchdog array expansion is fatal under macOS bash 3.2 (`set -u` + empty array) — Linux TPU hosts unaffected; noted for exp_04.
- **Next** — S10a review → strengthen (incl. the rename) → commit → the K1 package (all other gate conditions met).

## 2026-08-05T20:35:00Z — S10a cycle CLOSED (1139 green, 30/30 mutants, follow-up APPROVE) → commit — K1 GATE FULLY MET

- **Result** — `passed`. Strengthen closed all three findings: **BLOCKER** — pos adoption validates DEV provenance and binds to the DEV manifest digest independent of the capacity cohort (the K1-enabling fix; A1–A7 mutants); **LOW** — heredoc-AST + real-config-parser tests (M13–M20); **MINOR** — rename to `test_pos_context_launcher.py`. Dual-touch delta stays additive/positive-branch-only (reviewer-confirmed). Committed with this entry. Process note: the Coder died on an API session limit after completing the battery; the Planner assembled the closing record from disk evidence (battery log + independent suite re-run), recorded honestly rather than reconstructed from memory.
- **Ratified reading (reviewer, follow-up §3):** cohorts are one-per-invocation ⇒ **K1 = four launcher invocations** (smoke capacity → adequacy_probe → capacity dev64 → capacity trainfit16, both capacity phases consuming the one DEV `POS_ADEQUACY_URI`), DEV and TRAINFIT under DISTINCT artifact roots so TRAINFIT cannot overwrite the DEV-authoritative selection/tables. The same reading exposed exp_04 J1's missing TRAINFIT half — recorded in exp_04's worklog with its J1-2b remediation.
- **K1 package (predeclared acceptance criteria, plan §9 + SOP):** (1) worker reports this round's tip; (2) v6e-8; (3) SMOKE publishes ≥1 pos shard (S5 codec, `PosProvenanceHeader` l_pos=8, slot=positive) + the R1 golden asserted; (4) ADEQUACY publishes the positive adoption artifact (first-8 DEV, approved grid, full evidence); (5) CAPACITY dev64: all six B-arms + L_pos∈{1,8} ablation, zero unexplained quarantines, full-cohort decode, gates tables + selection.json + records provenance-bound under `…/k1/capacity`; (6) CAPACITY trainfit16: same arms under `…/k1/capacity_trainfit`, SAME DEV adequacy adopted, records bound to the trainfit digest; (7) no OOM/NaN (per-example divergence ⇒ quarantine); (8) gates per G1/G2 + target selection — any outcome is acceptance. Failure triage per SOP. Est. ~3–6 h. Submission via `submit_k1.sh` (archived at launch) — awaiting Yixun's `!` (issue #10); the `_command.md` K1-1 entry is written at actual launch.
- **Next** — K1 launch (Yixun) → S6–S10 trainer path in parallel with K1's run.

## 2026-08-05T21:00:00Z — S6 `regression-gather-loss` write phase complete (Coder); review dispatched

- **Goal** — The trainer path's data/objective layer: gather (K2-schema records + sampled step indices → training tuple), fp32 MSE + per-example normalized MSE, per-step TRAIN-cache variance table.
- **Change** — 2 NEW files: `pos_context_regression.py` (155 exec LOC; POS_STEPS, RegressionBatch, sample_step_indices, gather_training_tuple, regression losses, target_variance_table via streaming Chan in float64, normalized_regression_loss) + `test_pos_context_gather_loss.py` (44 cases). No tracked file modified; timestep/σ primitives imported from exp_04.
- **Command / Validation** — red evidenced (ModuleNotFoundError first; one oracle fixed for a real bf16 round-to-even subtlety at 1+2⁻⁸ and renamed to what it pins); **1183 passed** (1139 + 44); black/ruff clean. **30 mutants: 28 killed, 2 ratified** as ONE equivalence class (redundant fp32 casts — either alone is byte-unobservable; dropping BOTH is killed by the finite-fp32 output test; class boundary proven with probe mutants).
- **Result** — `passed` (write). **Planner positions, all endorsed:** (1) population variance per step over all cache elements, single streaming pass (float64 Chan — the cache is ~15 GiB, never resident; Σx²−mean² cancels exactly on the low-variance steps the metric exists for); (2) normalized metric is PER-EXAMPLE by its own step's variance (batch-MSE ÷ mean variance is uninterpretable for the S7 stop rule; oracle distinguishes them); (3) G12 equivalence class kept as defence-in-depth (exp_04 precedent — explicit output-dtype declaration is worth two ratified survivors); (4) jittable loss with trace-safe guards; (5) sigmas seam defaulting to canonical_sigmas, value-pinned; (6) duck-typed records (S7 may stream lighter views); (7) z_i0 excluded — the objective doesn't use it; S7 asks if it wants it.
- **Coder flag (deferred by Planner):** repo-wide `black` would reformat settled S4/S5 files (committed un-black-formatted) — a deliberate one-time formatting commit AFTER the K-jobs settle, never inside a feature round.
- **Next** — S6 review → strengthen → commit → S7 `trainer-state-checkpoint`.

## 2026-08-05T21:25:00Z — S6 cycle CLOSED (APPROVE; LOW closed in-cycle; 1184 green) → commit

- **Result** — `passed`. Review: APPROVE, zero production findings, all five Planner positions ratified. The one LOW (float64 accumulation not strictly pinned) closed in-cycle: tolerance tightened to the principled half-fp32-ulp bound (rtol 1e-7 — admits every correct float64 implementation at 5.96e-8, excludes the reviewer's float32-Chan probe at 1.528e-7, and the test asserts the exclusion so the bound cannot go slack), plus a high-offset (2²⁰) low-spread fixture where float32 accumulation is off by >1% — documenting WHY float64. New V6 float32-accumulator mutant killed by both (it survived pre-tightening — the reviewer's exact point). Focused 45; **full suite 1184**; module byte-identical to what was reviewed.
- **Commit hygiene (append-only correction, Coder-caught):** the S6 source files were accidentally included in the docs commit `c702206` (an `add -A` sweep while untracked) instead of a feat commit of their own. History is pushed and stays unrewritten; the ledger below maps S6 to BOTH commits explicitly. Rule reaffirmed for the Planner: stage docs commits by explicit path, never `add -A`, in worktrees carrying uncommitted rounds.
- **Next** — S7 `trainer-state-checkpoint` (dispatched with this entry).

## 2026-08-05T21:55:00Z — S7 `trainer-state-checkpoint` write phase complete (Coder); review dispatched

- **Goal** — The regression trainer core: train step over cached batches, adapter-only optimizer, gradient accumulation, the F2 stop rule, Orbax checkpoint/resume.
- **Change** — 2 NEW files: `trainers/wan_pos_context_regression_trainer.py` (249 exec LOC — over the 200 budget under Planner pre-authorization; justification recorded: five coupled contracts, 29 LOC already trimmed) + `test_pos_context_trainer.py` (45 cases). No tracked file touched.
- **Command / Validation** — red evidenced (3 first-run failures, all real: a wrong streak expectation rewritten to the property that matters; an Orbax-internals poke rewritten to on-disk items; a fixture whose train MSE wasn't falling fixed AND the conjunct now asserted); **1229 passed** (1184 + 45); black/ruff clean. **31/31 mutants** (first-pass survivor C4 — saves without their metric — closed by test strengthening: manager.best_step() == report.retained_step).
- **Result** — `passed` (write). **Planner positions, all endorsed:** (1) 249 LOC accepted; (2) the CLOSURE-SEAM freeze — frozen params captured, not passed, structurally unreachable by grads, verified on a real tiny WanModel + real adapter stack — endorsed strongly (structural impossibility over convention); (3) cheap-half forward (patchify → block-0 → head; no 40-block run — the teacher-forced objective needs only Ĉ_t; fp32 cast stays in S6's loss per the single-cast rule); (4) stop-rule edge semantics pinned and now CANONICAL: running best strictly before current, "falling" vs previous eval, tie keeps earliest, first eval never triggers; (5) windowed (non-overlapping) train MSE; (6) Orbax best_fn/min retention; (7) `optional_config_value` never getattr — the issue-#11 trap institutionalized as mutant D5.
- **S8 obligation recorded:** declare `pos_logical_batch`/`pos_microbatch` in the S8 YAML/dispatch — the trainer reads them optionally with plan defaults; K3 cannot use the accumulation fallback until declared.
- **Next** — S7 review → strengthen → commit → S8 `dispatch-config`.

## 2026-08-06T01:15:00Z — S7 cycle CLOSED (1243 green, 45/45 mutants) → commit

- **Result** — `passed`. The heaviest cycle this experiment: write (45 tests) → review (4 BLOCKER + 2 MAJOR, all accepted) → strengthen (pytree/JIT with executing oracle; logical-GBS enforcement; the CHECKPOINT REDESIGN — recency-only resume tree, immutable earliest-best selection artifact on a sibling path, F2 decision state atomic inside the step item; integrated interrupted≡uninterrupted oracle through `run`; adam-based freeze proof; pooled DEV weighting) → follow-up (5 closed + 1 MAJOR) → final fix (terminal-verdict honor guard, reviewer-prescribed verbatim, closed without a third pass per the recorded judgment in the review file). Committed with this entry.
- **Incidents recorded in the review file:** the mid-battery timeout that left C7 on disk (caught by fixed-string audit; C7 then EXPOSED the real Orbax nested-manager temp-dir defect — the sibling `_selection` layout is the fix, docstring carries the mechanism); the concurrent stray battery (killed; foreground chunks + sha-verified restores now the rule). The battery's anchor-uniqueness refusal on G1 worked as designed.
- **Design facts now load-bearing for S8/S9/K3:** selection artifact at `<ckpt_dir>_selection` is what K4 consumes; history rides in the step JSON item (Orbax metrics() needs best_fn, forbidden on the resume tree); `pos_logical_batch`/`pos_microbatch` MUST be declared in S8's YAML.
- **Next** — merge-interim-2 (exp_04 fix `27efcd1` → this branch; the K1-1 remediation) → K1-2 relaunch → S8 `dispatch-config`.

## 2026-08-06T02:05:00Z — merge-interim-2 EXECUTED (exp_04 fix `27efcd1` → this branch) — the K1-1 remediation; K1-2 ready

- **Version Control** — merge of exp_04 @ `27efcd13` (brings R11 `direct_opt` + the `hyperparameters-config-access` fix round). Two content conflicts in the dual-touch entrypoint, both additive-union (Planner-resolved): NULL_MODES gains `direct_opt` alongside EMBEDDING_SLOTS; the adoption expression keeps the slot conditional with the null branch adopting the fix's direct `config.null_adequacy_uri` read. Coder reconciliation: `Sinks` gained R11's `write_arrays` (transactional npz for A3 — `pos_default_sinks` inherited it via `dataclasses.replace`; one test fake mirrored); `optional_config_value` UNIFIED on exp_04's reviewed helper verbatim (duplicate deleted at exp_05's position; exp_05's S4-era None-coalescing DROPPED — unreachable on the real class, not load-bearing at any call site, ruling recorded: not re-proposed); the pos YAML mirrors R11's `null_a3_measure`/`null_a3_iters` (copy-contract pin); `pos_execute` gains a fail-closed `a3_measure` guard (+9 — refuses an explicitly-requested A3 on the positive slot rather than silently dropping it; Planner-endorsed). Black-churn on settled files avoided (HEAD-restore + re-apply, second occurrence of the S10a trap).
- **Command / Validation** — full combined suite: **1353 passed, 0 failed** (from 18F/1335P mid-merge). The fix-round AST guard (`test_no_three_argument_getattr_on_config_survives`) now enforces issue #11 in THIS branch.
- **Result** — `passed`. K1-1's crash site (`mode_kwargs` code_sha) is fixed here by the same reviewed code exp_04 launched J1-3 on. **K1-2 relaunch ready at this tip** — same four-phase runbook (no selection-consuming phase ⇒ unaffected by J1-2's flaw class), `…/k1` root clean (K1-1 published nothing).
- **Next** — K1-2 launch (Yixun) → S8 `dispatch-config`.

## 2026-08-06T02:50:00Z — S8 `dispatch-config` write phase complete (Coder); review dispatched

- **Goal** — The K3 launch wiring: `train_wan.py` dispatch arm + the training YAML + config-only trainer construction.
- **Change** — train_wan.py **+3/−0** (one `elif`, dual-touch minimal); NEW `base_wan_5b_pos_context_train.yml` (282 lines, generated FROM the side-adapter config — superset structural, pinned; +5 pos keys incl. the S7 obligation `pos_logical_batch: 256`/`pos_microbatch: 256`); NEW `test_pos_context_dispatch.py` (25 cases); trainer +30/−4 (optional seams; `start_training` S9 boundary).
- **Command / Validation** — red evidenced (10F/8P/7E — the 8 passes were the pre-existing arms' characterizations, correctly green-first); **1378 passed** (1353 + 25); black/ruff clean (train_wan.py deliberately NOT black-formatted — upstream 2-space style, dual-touch diff hygiene). **21/21 mutants** (D3 SIDE_ADAPTER_TI2V reroute = the merge hazard; Y10/Y11 leak guards; C1–C4 construction).
- **Result** — `passed` (write). **Planner positions, all endorsed:** (1) config-only construction with optional seams (train_wan does `Trainer(config)`); (2) `start_training` raises a NAMED S9 boundary — honest stop, not a fake run; (3) house `checkpoint_dir` reused, no pos_ prefix; (4) `pos_microbatch == pos_logical_batch` as the no-accumulation sentinel (int-stable; 64×4==256 pinned); (5) train_wan.py formatting untouched; (6) the AST-executed dispatch test — built from scratch; my brief wrongly implied an existing dispatch test to mirror (THIRD unverified-premise instance, logged: verify test-inventory claims against the tree before writing briefs).
- **Next** — S8 review → strengthen → commit → S9 `evaluator-regressed-restore-rollout` (the last round before the K3 package).

## 2026-08-06T03:10:00Z — S8 cycle CLOSED first-pass (APPROVE, zero findings; 1378 green) → commit

- **Result** — `passed`. The reviewer independently confirmed: dispatch exactly additive, all 186 baseline YAML keys retained + exactly five pos keys, F3 sharding identical to the side-adapter config, the trainer's six config reads all declared, zero leak, empty-checkpoint_dir safe. Committed with this entry.
- **Next** — S9 `evaluator-regressed-restore-rollout` (the last code round before the K3/K4 packages).

## 2026-08-06T04:05:00Z — S9 cycle CLOSED as a HELD-OPEN PARTIAL round (1417 green, 26/26 mutants) → commit

- **Result** — `passed` (the exp_05-owned two-thirds). Shipped: `pos_context_eval.py` (134 exec LOC — metadata-verified `restore_selected_adapter` consuming ONLY the S7 selection artifact, refusal matrix incl. both step directions and required DEV metric; `pre_k4_dev_gate` = exp_04's G3 form by import, DEV-only, issuing a STAMPED v1 certificate bound to the checkpoint identity; `k4_comparison_row` refusing TEST without a verified certificate; the R14/R15 stall boundary with a repo-rooted tripwire test) + 39 tests + the exp_05-owned trainer delta (l_pos stamped into checkpoint metadata). Review trail: REQUEST-REVISION (2 blockers + 1 major oracle gap, all accepted) → strengthen (sharper-probe closures of four first-pass battery survivors) → closed per the recorded no-third-pass judgment.
- **THE STALL (Planner ruling, reviewer-ratified):** exp_04's evaluator (`generate_wan_null_adapter.py`, rounds R14/R15) does not exist in any ref — exp_04's plan gates it on the P1 outcome J1-3 is computing now. Per plan §6's own rule, S9's rollout-wiring third STALLS AT THE MATRIX rather than authoring exp_04's deliverable (dual-touch delta this round: ZERO lines). The tripwire test fails loudly when exp_04 ships the file. **S9 reopens at merge-2/R15-boundary.**
- **exp_05 code state:** S1–S8 + S10a complete; S9 held open (external dependency); S10's remaining launcher (`train_wan_pos_context.sh`) rides with the K3 package. Everything up to K3's trainer launch is BUILT AND REVIEWED; K2/K3 packages await K1-2's outcome; K4 awaits R14/R15 + the pre-K4 DEV gate.
- **Next** — results-driven: K1-2 terminal → gates/selection reading → K2 package; J1-3 terminal → the shared P1 reading + exp_04's R12+ unlocking.

## 2026-08-06T16:35:00Z — K1 RESULT READING (P1' primary outcome; Planner) — H1 PASS ~0.92, H2 FAIL ~0.16, TARGET = STOP on both cohorts

**The headline, robust across DEV-64 and TRAINFIT-16 (full coverage, zero invalid pairs, 10k-resample CIs, seed 20260804):**

| Gate / arm | DEV-64 | TRAINFIT-16 |
|---|---|---|
| **H1** — B1 (optimize+replay, own basin) vs B0 | **PASS**: mean SSIM **0.9227** [0.9129, 0.9314], frac_improved **1.00**, median MSE-ratio 28.6× | **PASS**: **0.9095** [0.8839, 0.9270], 1.00, 26.0× |
| **H2** — B2 (optimize+replay from fresh ε₀) vs B2-0 | **FAIL**: mean SSIM **0.1610** [0.1358, 0.1868], frac_improved **0.00**, median ratio 0.254 (worse than control) | **FAIL**: **0.1570** [0.1076, 0.2082], 0.00, 0.196 |
| **B1-probe** (B1's locked contexts under keyed{0,1,2}) | 0.5254 abs; **0.569 relative** (floors: 0.70 abs, 0.7× rel) | 0.4885 abs; 0.537 relative |
| **Selection (predeclared rule)** | **STOP** | **STOP** |

**Scientific statement:** (1) The deployed 8-token conditioning channel has ENORMOUS in-basin capacity — per-clip optimized contexts drive the FROZEN 5B backbone to ~0.92 SSIM reconstruction, ~3.6× the frozen-context control (~0.25) and ~3.1× the trained adapter's deployed rollout (0.2946). The adapter's bottleneck was never channel capacity. (2) That signal is strictly NOISE-BASIN-BOUND: locked contexts under foreign keyed noise collapse to ~0.5; and — the decisive arm — B2, which RE-OPTIMIZES per-step from fresh ε₀ with the same adopted recipe, lands at 0.16, WORSE than doing nothing. Even per-clip, per-step optimization cannot steer a fresh-noise trajectory onto the clip via the 8-token context. This replicates the PyTorch fork's basin problem (own-z_init 0.015 vs fresh 2.7 MSE) at n=64 with CIs, at the deployment-matched representation.
**Consequence per the predeclared target-selection rule:** no arm qualifies → **K2 target caching does NOT proceed** on these targets. The design nuance for Yixun's decision (recorded, not decided here): K3's teacher-forced regression conditions on cached z̄_t states (a state-conditioned emitter, not a fixed context), which H2/probe do not directly measure — but B2's in-fresh-basin failure substantially weakens that bet: if direct optimization from ε₀ fails, a learned emitter of the same representation is unlikely to succeed. Options land in the status report; exp_04's J1-4 (running) answers the same transfer question for the null slot, whose CFG physics differ.
