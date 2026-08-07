# rollout_adapter — Codex code review: T3b-1 `step-stream`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: REQUEST-REVISION — **1 BLOCKER + 2 MAJOR, all accepted.**

- **(a) PASS** — the additive purpose: old ids unchanged under name-hashing, the re-derivation oracle genuinely independent, T1's 40 tests untouched. **(e) ACCEPT** — U08's message-matching is specificity, not over-fitting.
- **BLOCKER — the stream omits `t_idx`.** The settled C0 objective samples per-example timesteps from `step_rng` (`wan_ti2v_side_adapter_trainer.py:133`); outside `StepDraws` that draw can be accumulation- and resume-dependent, which would silently confound the R-B-vs-C0 comparison. Fix: an additive purpose-keyed `t_idx` draw at logical-batch shape inside `StepDraws`, plus the dropout key derived there or dropout structurally pinned inert.
- **MAJOR — the accumulation-invariance proof is vacuous AND the API permits real divergence.** The test deletes `accumulation_steps` and repeats an identical call (it cannot fail); meanwhile the arbitrary-shape API means a per-microbatch draw concatenated gives epsilon EXACTLY unequal to the factor-1 draw at factors 2/4/8 — the reviewer demonstrated it. Required: a draw-once-at-checked-logical-shape-then-split seam, with factors 1/2/4/8 compared by reconstruction on epsilon, `t_idx` and support.
- **MAJOR — the restored-`state.step` obligation is NOT discharged.** Module-local guards hold, but any caller may still pass `state.step` as `global_step`, and the demonstration's opening-stream equality is tautological (it exercises neither a state object nor the loader/loop). It stays OPEN for T3b-4, which must carry a production-callsite AST pin plus interrupted-vs-uninterrupted execution through the REAL restore path. Verdict (d) is therefore downgraded to "primitive statelessness only" — S7's loop-level resume failure class remains live.

**Planner note:** the BLOCKER was relayed to the Coder MID-FLIGHT because it changes what T3b-2's C0 arm consumes (`t_idx` must come from the stream, not be drawn in the arm) — folding it in beats reworking it.

## Reviewer output (verbatim)

124,657
## Findings

1. **BLOCKER — the stream omits a random quantity consumed by matched-C0.** [pos_rollout_stream.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_stream.py:60) defines “everything one optimizer step draws” as support plus epsilon, but the settled C0 objective also samples per-example `t_idx` from `step_rng` ([wan_ti2v_side_adapter_trainer.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py:133)). If that draw remains outside `StepDraws`, it can be accumulation- and resume-dependent. Add an additive, purpose-keyed C0 timestep draw at logical-batch shape and carry it in `StepDraws`. Also either derive the dropout key there or structurally pin dropout as inert.

2. **MAJOR — accumulation invariance is not proven by value, and the API permits divergent epsilon.** [test_pos_rollout_stream.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_stream.py:248) deletes `accumulation_steps` and repeats the identical call. Because [draw_step_stream](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_stream.py:84) accepts an arbitrary shape, drawing once per microbatch and concatenating produced exact inequality against the factor-1 epsilon for factors 2/4/8. Provide an orchestration seam that draws once at the checked logical-batch shape before splitting, then test factors 1/2/4/8 by reconstructing and exactly comparing epsilon, `t_idx`, and support.

3. **MAJOR — the restored-`state.step` obligation is not discharged.** [test_pos_rollout_stream.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_stream.py:174) guards only this module and selected identifier spellings; any caller can still pass `state.step` as `global_step`. The demonstration at line 192 correctly illustrates replay arithmetic, but its opening-stream equality is tautological and it exercises neither a state nor the loader/loop. Keep the obligation open for T3b-4 and require a production-callsite AST pin plus interrupted-versus-uninterrupted execution through the real restore path.

## Verdicts

- **(a) PASS.** The purpose is appended, old names retain their SHA-256-derived IDs, and the helper arithmetic is unchanged. The independent oracle hardcodes the offset, recomputes SHA-256, and reimplements both folds. The untouched T1 suite passes.

- **(b) FAIL as a discharge claim.** Local state-blindness holds; caller-level loop-step provenance does not. The counterfactual demonstrates the right failure concept but not actual wiring.

- **(c) FAIL.** Current arm-blindness is only local/structural, with no two-arm consumption path. Accumulation’s claimed value proof is vacuous and a microbatch-shaped call demonstrably changes epsilon.

- **(d) PASS only for primitive statelessness.** Cold/walk/resume fresh namespaces reproduce step 400 exactly, but this is insufficient for the loop-level resume contract; S7’s failure class remains possible.

- **(e) ACCEPT.** Matching semantic substrings such as “larger than” versus “does not divide” makes the more specific diagnostic observable without pinning exact punctuation or formatting; this is not over-fitting.

Requested battery: **84 passed**.

**REQUEST-REVISION — the additive RNG primitive is sound, but the stream is incomplete and its accumulation and loop-step invariance claims are not yet established.**
