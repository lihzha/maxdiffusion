# rollout_adapter — Codex code review: T3b-1 strengthen + T3b-2 `arm-losses` (combined)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: **BOTH REQUEST-REVISION** — 2 BLOCKER + 1 MAJOR + 4 MINOR, all accepted by the Planner.

**PASSED independent verification:** B1 — matched-C0 is tied to the settled `_denoising_loss` by two contiguous AST spans, with bitwise value equality and adapter-gradient difference exactly 0.0 across both timestep fixtures (a valid matched control within the stated fresh-noise/uniform-timestep policy). B2 — the inertness proof holds: the no-stop-gradient construction has exactly the same adapter gradient because `z_t` and the frozen unconditional branch carry no adapter-parameter path, and together with T3a's measured rollout contrast, *"inert here, load-bearing there" is established*. A3 — the restored-`state.step` re-labelling is honest and the obligation is properly carried open.

**BLOCKER (T3b-2) — R-B ignores the streamed support and redraws it.** `rollout_arm_loss` reads only `draws.epsilon`, passing independent seed/global_step/support_salt into T2's kernel, which calls `rollout_support` again — so epsilon can come from one `StepDraws` while support comes from a different step or salt. The arms-share-one-stream property is semantically false for R-B, and the composition oracle reproduces the same redraw so it cannot catch it. Fix: T2's kernel takes explicit `support_start`/`support_end`; the arm passes `draws.support_start/end`; support-derivation inputs leave the arm.

**BLOCKER (T3b-2) — dropout is outside the stream and not proven inert.** C0 accepts `dropout_rng` and runs `deterministic=False` while R-B discards it, so "C0 samples nothing" is false semantically. Must be discharged **before M1 or any training execution**; T3b-4 is the last acceptable scheduling point and does not close the all-randomness contract until done.

**MAJOR (T3b-1) — the unsafe accumulation path is still public.** `draw_step_stream` accepts an arbitrary shape and the raw helpers are exported; `draw_logical_step` proves a safe path exists, not that callers must use it. Fix: privatize the raw helpers; the public seam takes the logical batch, checks it, derives the shape, draws once, splits.

**MINORs:** arm-blindness guard does not ban `rollout_arm_loss`/`one_step_denoising_loss`/importing `pos_rollout_arms`; purpose count stale (six at review, seven live after T3b-3's `dev_instrument`) with two contradictory docstrings; and a test comment still carries the exact overclaim the inertness test disproved, plus a malformed residual sentence in the arm module docstring.

**Planner rulings on the dropout blocker:** first MEASURE the adapter's production dropout rate. If 0 — take the prove-it path rather than changing deployment semantics: structurally pin production dropout to zero and prove that changing the supplied key leaves value AND gradient bitwise unchanged (a stronger, cheaper claim than plumbing an unneeded key). If nonzero — it becomes an accumulation-safe stream-derived draw like the others. On the R-B blocker: T2 is committed but it is exp_06's own module, so changing its signature is allowed; **preserve T2's extraction-equivalence claim** by feeding its equivalence tests exactly the values the old derivation produced — the same move the plan already made for seed/k/salt (explicit arguments, arithmetic unchanged).

**Sandbox caveat recorded:** the reviewer's environment blocked writable-tmp probing (92 passed / 9 environment failures); after pre-initializing the tmp setting it measured **101 passed**. It did not rerun the full 1,594-test suite or the mutation batteries — the Coder must re-run both and report real numbers.

## Reviewer output (verbatim)

163,605
Both deliverables remain **REQUEST-REVISION**. The strengthened tests are substantially better, but two stream-boundary violations remain commit-blocking: callers can still draw at microbatch width, and R-B ignores the support in `StepDraws`. Dropout is also still unresolved.

## T3b-1 findings

1. **MAJOR — the unsafe accumulation path remains public and reachable.** [pos_rollout_stream.py:43](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_stream.py:43) exports `draw_step_stream`, `rollout_epsilon`, and `one_step_timestep_indices`; [draw_step_stream:108](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_stream.py:108) still accepts an arbitrary `shape`. The failure-mode test itself demonstrates that a caller can draw at microbatch width. `draw_logical_step` proves a safe path exists, not that callers must use it.

   Concrete change: make the raw draw helpers private/non-exported and have the public seam accept the actual logical batch, run `checked_logical_batch`, derive its shape, draw once, and split. Add a production-callsite pin in T3b-4.

2. **MINOR — the replacement arm-blindness guard does not actually ban both current arm losses.** [test_pos_rollout_stream.py:183](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_stream.py:183) bans `rollout_endpoint_loss` and `_denoising_loss`, but not `rollout_arm_loss`, `one_step_denoising_loss`, or importing `pos_rollout_arms`.

   Concrete change: use AST checks over imports and calls for those exact symbols/module. Removing the blanket `"one_step"` substring ban was correct.

3. **MINOR — purpose-count and documentation claims are stale.** The reviewed T3b-1 snapshot had **six**, not five, purposes: four inherited plus `rollout_epsilon` and `one_step_index`. [rollout_adapter_worklog.md:107](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_worklog.md:107) says five; [pos_rollout_support.py:21](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_support.py:21) still says there is no epsilon purpose; and [StepDraws:72](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_stream.py:72) says “noise, and nothing else” despite carrying `t_idx`.

   Concrete change: update these statements to the current additive history. A concurrent T3b-3 edit has since added `dev_instrument`, making the live total seven; that later round was not reviewed here.

## T3b-2 findings

1. **BLOCKER — R-B ignores the streamed support and draws it again.** [rollout_arm_loss:216](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_arms.py:216) reads only `draws.epsilon`; it passes independent `seed`, `global_step`, and `support_salt` into T2’s kernel. That kernel calls `rollout_support` again at [pos_rollout_losses.py:263](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_losses.py:263). Thus epsilon can come from one `StepDraws` while support comes from a different step or salt. The composition oracle reproduces the same redraw, so it cannot catch this.

   Concrete change: let the T2 kernel accept explicit `support_start/support_end`, pass `draws.support_start/end` from R-B, and remove support derivation inputs from the arm. Add a test replacing only the support fields and proving R-B consumes them.

2. **BLOCKER — dropout is outside the stream and not proven inert.** C0 accepts `dropout_rng` at [pos_rollout_arms.py:159](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_arms.py:159) and runs `deterministic=False` at [pos_rollout_arms.py:184](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_arms.py:184), while R-B discards that argument. Therefore “C0 samples nothing” is false semantically even though this module contains no direct `jax.random` spelling.

   Concrete change: either provide an accumulation-safe stream-derived dropout policy, or structurally pin production dropout to zero and prove changing the supplied key leaves value and gradient bitwise unchanged. This must be resolved before any training job. Implementing it in T3b-4 is acceptable as scheduling, but not as grounds to close the present all-randomness contract.

3. **MINOR — corrected stop-gradient commentary is not completely consistent.** [test_pos_rollout_arms.py:343](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_arms.py:343) still says gradient equivalence proves the stop-gradients are present, precisely the overclaim the new inertness test disproved. The arm module docstring also contains a malformed residual sentence at [pos_rollout_arms.py:41](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_arms.py:41).

   Concrete change: state that numerical equivalence proves inertness, while the AST assertion alone pins the deployed syntax.

## Requested verdicts

- **A1 — PARTIAL.** `one_step_index` is genuinely additive and name-hashed; the four inherited keys remain unchanged, and T1’s generic hash test covers every declared purpose untouched. C0 consumes `draws.t_idx` and performs no timestep/noise draw. But the total was six, not five, and C0 may still consume dropout randomness.

- **A2 — FAIL.** The reconstruction oracle is now meaningful and exact across 8/4/2/1, and the naive counterexample is excellent. The seam is not exclusive; arbitrary microbatch-shaped draws remain public.

- **A3 — PASS.** The module and demonstration test now plainly limit the claim to primitive statelessness and carry the real restore obligation to T3b-4. The relabelling is honest.

- **Arm-blindness loosening — correct idea, incomplete guard.** Dropping the substring ban was right. No arm parameters or arm-dependent branches exist, but exact current arm-loss imports/calls are not all prohibited.

- **B1 — PASS.** The copied objective is tied to the settled `_denoising_loss` by two contiguous AST spans. The value comparison is bitwise and the adapter-gradient difference is exactly `0.0` across both timestep fixtures. This is a valid matched-C0 control within the stated fresh-noise/uniform-timestep policy.

- **B2 — PASS, within the current parameterization.** The no-stop-gradient construction has exactly the same adapter gradient because `z_t` and the frozen unconditional branch have no adapter-parameter path. Together with T3a’s measured rollout contrast, “inert here, load-bearing there” is established. It does not imply either individual stop-gradient is numerically detectable in C0; only the structural check pins them.

- **B3 — PARTIAL.** Parameter trees and batch-field sets match, and reverting from multiset to set equality was right. V07’s “C0 stops reading actions” mutant is materially better than duplicate-read bookkeeping. Semantic stream isolation nevertheless fails because R-B redraws support and C0 receives external dropout randomness.

- **B4 — PASS for composition equivalence, subject to the streamed-support blocker.** Exact value/gradient equivalence to T2-over-T3a is appropriate. V04 is a strong S16-class mutant when exercised on the rollout path; it does not numerically pin C0’s inert stop-gradients. V09 is a realistic and valuable accumulation mutant.

- **B5 — FAIL/open.** Deferral to T3b-4 is the last acceptable scheduling point, but dropout must be discharged before M1 or any other training execution.

## Validation

The literal requested command reached **92 passed / 9 failed**, with all nine failures caused by the managed read-only environment preventing Python/Abseil from probing a writable temporary directory—not test assertions. Running the same test selection after pre-initializing Python’s temporary-directory setting produced:

```text
101 passed in 34.16s
```

The full 1,594-test suite and the two mutation batteries were not independently rerun here.

- **T3b-1 `step-stream`: REQUEST-REVISION**
- **T3b-2 `arm-losses`: REQUEST-REVISION**
