# pos_context — Codex code review: round S3 `replay-state-capture`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_05 worktree). HEAD `db1384b`. MUST-1 ruled GENUINELY DISCHARGED (independent probes: fp32 bitwise; bf16 composition bitwise; omitting the cast changed 156 elements max 0.039). B0 active-CFG semantics confirmed (w=1-vs-w=5: 95 elements, max 4.19 on a tiny bf16 model); H1 valid within-exp_05, cross-experiment note required.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md`, both standing announcements, and `pos_context_yixun_query.md`.
- `plan_pos_context.md` v3, especially §3 replay and §4-P1′ B-arm semantics; exp_04’s inherited A0/G1 definitions.
- `pos_context_worklog.md` through S3, including the cast-rule correction and MUST-1 claim.
- Complete S1 and S2 reviews with strengthening records.
- exp_04 R4a replay review, including batch guards, A0 semantics, call structure, and trajectory-level composition standard.
- HEAD `db1384b`; the complete uncommitted diff, full `pos_context_inversion_wan.py`, and all 442 lines/26 tests of `test_pos_context_replay.py`.
- Shared pin, timestep, velocity-check, and sigma-validation helpers.
- MAIN Wan2.2 checkout `f370228`: `_dit_velocity`, `optimize_positive_embeddings`, `regenerate_with_positive_embeds:822-853`, and L_pos forcing.

1. **MINOR — test batch-composition assertions stop short of the R4a/S1 standard.** [test_pos_context_replay.py:353](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_replay.py:353), [test_pos_context_replay.py:365](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_replay.py:365)  
   `test_example_zero_is_independent_of_the_rest_of_the_batch` checks only example 0’s final latent; a collapse-to-row-0 mutation passes that specific test. The shared `[N,L,D]` comparison likewise checks only the endpoint. The literal oracle currently catches material cross-wiring, and my stronger probe passed bitwise, so this is not an operator defect. Change both tests to request trajectories, compare every batched example against its singleton trajectory, and compare shared versus explicitly repeated contexts over the full trajectory.

2. **MINOR — the module-level handoff contradicts S3’s completed state.** [pos_context_inversion_wan.py:3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/models/wan/pos_context_inversion_wan.py:3), [pos_context_inversion_wan.py:36](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/models/wan/pos_context_inversion_wan.py:36)  
   It still says the S3 operator “lands later” and that neither MUST is dischargeable, while the replay docstring correctly declares S3 discharged. Update the lifecycle text to mark MUST-1 discharged by the actual-operator tests and leave only S4’s real runner-closure/both-branches bf16 obligation open.

3. **MINOR — §4-P1′ warrants an explicit H1 interpretation sentence.** [plan_pos_context.md:41](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md:41)  
   “Forms verbatim” is mathematically correct, but H1 and G1 do not have identical estimands. Add that H1 measures B1’s optimization/repair lift over a frozen short-context **active-CFG** B0—including the w=1-pivot to w=5-replay mismatch—and that its ratio must not be interpreted as directly comparable to exp_04’s A1/A0 gate, where A0 collapses CFG.

MUST-1 verification verdict: **GENUINELY DISCHARGED.** The fixture invokes the actual replay operator and captures its own conditional forward at its actual latent/timestep. Operator output equals deployment bitwise at fp32; operator plus the casting closure equals deployment bitwise at bf16. Omitting that cast changed 156 elements with max `|Δ| = 0.0390625`, so the bf16 half is load-bearing. S4’s actual runner closure for both branches remains correctly open.

Operator fidelity otherwise passes: contexts and timesteps are correct at every step, both velocities are fresh and ordered conditional→unconditional, CFG and negative `dsigma` match `f370228`, initial/per-step pins and trajectories are correct, and fp32 arithmetic is preserved. The 26-test file passed; independent four-step probes also established bitwise return-trajectory invariance, shared/explicit equivalence, and every-example singleton composition.

B0 is correctly an active-CFG frozen-C control. A tiny bf16 WanModel probe found w=1 versus w=5 changed 95 elements, max `|Δ| = 4.1918`. H1 remains a valid within-exp_05 B1-vs-B0 gate because both arms replay at w=5; only cross-experiment interpretation against G1/A0 needs the note above.

Final verdict: **REQUEST-REVISION — the replay operator and MUST-1 discharge are correct, but the focused test/documentation strengthening should land before S3 commits.**

Status:

- S3 review and all local probes are complete; no reviewer subprocesses remain.
- J0 data job: user-reported running and untouched; ETA unknown.
- Earliest steering point: now; interruption is safe.

---

## Strengthening record (Coder, same round; items 1–2 closed; item 3 was the Planner's plan amendment, landed at 58c1676)

1. **MINOR (trajectory-level batch assertions) — FIXED.** Every-example full-trajectory-vs-singleton comparison (bitwise) + shared-vs-repeated full-trajectory equivalence; the new N8 collapse-to-row-0 mutant measured directly: old assertion PASSES blind (Δ=0 on example 0), new one FAILS (Δ=9.57 on example 1); killed by 5 tests.
2. **MINOR (lifecycle text) — FIXED.** MUST-1 marked DISCHARGED naming both tests and what each proves; S4's obligation marked the only open half of the cast rule, with the reason stated.

Suite **715 passed** (strengthened in place, no count change); cumulative S3 record **8 mutants, 0 survivors**. Round S3 closed; committed with this record.
