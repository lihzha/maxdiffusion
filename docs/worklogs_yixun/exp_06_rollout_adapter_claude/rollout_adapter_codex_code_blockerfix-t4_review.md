# rollout_adapter — Codex code review: blocker-fix round + T4 `dispatch-config` (backlog pass 1 of 3)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **T4 `dispatch-config`: APPROVE** (no commit-blocking production defect). **Blocker-fix round: REQUEST-REVISION — 2 BLOCKERs + 1 MINOR.**

- **A1–A5 all PASS**, and **all four Coder judgment calls ACCEPTED**: removing `expected_size` (the pinned digest already fixes exact bytes, so size/content follow; the remaining cohort/split/size checks are defense-in-depth for a future pin change and test-only monkeypatching is appropriate); rejecting the closure-hidden token (a closure relocates the secret without eliminating it — verified construction as the only constructor is sounder); reconciling before the terminal guard (a crash during the final evaluation's write window is exactly when terminal reopening must repair the artifact); failing closed on an unreconcilable sibling (continuing would knowingly produce a run report naming an unavailable checkpoint). A3's post-resume empty window is also ACCEPTED with a reason the Coder had not stated: resume checkpoints are written only at evaluation boundaries AFTER the completed window is recorded, so a mid-window crash restores the preceding boundary and recomputes the whole window.
- **BLOCKER — A6 FAIL: `DevBatchReader`'s public decoder/binder injection reproduces arbitrary-content scoring under genuine DEV provenance.** The reviewer executed it: a `reader` echoing every genuine DEV name and declared ordinal while returning tensors filled with `999`, plus a `binder` echoing the manifest's generation and size, yielded `metric 999.0 / cohort dev64 / sha 3c59d02… / examples 64`. Its verdict: *"this is the previous unrestricted callback one layer lower."* Fix: remove `reader`/`binder` from the production constructor and `batch_reader` from `score_dev_cohort`; scoring constructs and uses the canonical reader internally; tests monkeypatch the underlying decoder/binding modules.
- **BLOCKER — NEW: non-finite DEV metrics poison the historical-best and sibling-selection contract.** `stop_verdict([NaN, 0.1])` returns `best_step=1000, best_value=nan`; because comparisons against `NaN` are false no later finite value can replace that history-best, while `preserve_selection` can still replace the sibling ⇒ report and shipped artifact disagree and the next reconciliation fails closed. Fix: reject non-finite values at every entry point (per-example scores, aggregate DEV metrics, train-window means, restored history) before either tree is written; add NaN and both signed infinities to the decision-level tests.
- **MINOR — a vacuous assertion:** `test_pos_rollout_dispatch.py:192`'s predicate ends with `and False`, so the indentation-preservation check can never fire.
- **T4 verdicts:** B1 PASS (186 inherited + exactly 7 additions, only `model_type` and `eval_data_dir` changed; all pilot defaults match plan v2.5 and Yixun's decisions — the reviewer correctly excluded the four later T6/T7 additions now bringing the file to 197 keys). B2 PASS (`eval_data_dir` empty, manifest path DEV-64, bytes match the pinned digest).

**PLANNER SELF-ASSESSMENT:** the A6 hole was flagged honestly by the Coder as a residual ("the decode seam stays injectable — exp_04's standard, no better") and **the Planner ACCEPTED it**. The reviewer refused and executed the attack. **Standing correction: "matching a prior experiment's tolerance" is not a justification when the hole is reachable** — this is the third round in which the structural-vs-checked pattern went one layer deeper than the previous fix, and each time the residual was described accurately and closed insufficiently.

## Reviewer output (verbatim)

251,495
# Review verdict

**T4 `dispatch-config`: APPROVE.**

**Blocker-fix round: REQUEST-REVISION.** A1–A5 are closed, but A6 remains exploitable through a renamed callback seam. I also found a separate non-finite-metric selection blocker.

## Part A findings

1. **BLOCKER — A-B2 remains reachable through `DevBatchReader`’s public decoder injection.**  
   [pos_rollout_dev_instrument.py:271](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:271) publicly accepts arbitrary `reader` and `binder` callbacks. Class-level dispatch at [line 366](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:366) prevents overriding `read`, but canonical `read` still invokes the caller-created `_read_batch` closure.

   I supplied a reader that echoed every genuine DEV name and declared ordinal while returning tensors filled with `999`, plus a binder echoing the manifest’s generation and size. The result was:

   ```text
   metric 999.0
   cohort dev64
   sha 3c59d023f3b782542ecae443b8d83008e7d8dfd801347f41adfab75218340836
   examples 64
   ```

   A reader can equivalently load TEST tensors, relabel them with the requested DEV identity, and receive genuine DEV provenance. This is the previous unrestricted callback one layer lower.

   **Concrete change:** remove `reader`/`binder` from the production constructor and remove `batch_reader` from `score_dev_cohort`; scoring should construct and use the canonical production reader internally. Tests can monkeypatch the underlying decoder/binding modules rather than passing a callback through the scoring API. Add the executed foreign-tensor/echoed-identity attack.

2. **BLOCKER — non-finite DEV metrics poison the historical-best and sibling-selection contract.**  
   [pos_rollout_dev_instrument.py:367](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:367) accepts `NaN`/`Inf` loss values, while [pos_rollout_loop.py:169](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_loop.py:169) treats the first metric—even `NaN`—as the historical best. Executing `stop_verdict([NaN, 0.1])` returned `best_step=1000, best_value=nan`.

   Because comparisons against `NaN` are false, later finite values cannot replace that history best, while `preserve_selection` can nevertheless replace the sibling. The report and shipped artifact can therefore disagree, and the next reconciliation fails closed.

   **Concrete change:** reject non-finite per-example scores, aggregate DEV metrics, train-window means, and restored history before writing either checkpoint tree. Add `NaN` and both signed infinities to the decision-level tests.

## Part B findings

1. **MINOR — the indentation-preservation assertion is vacuous.**  
   [test_pos_rollout_dispatch.py:192](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_dispatch.py:192) ends its predicate with `and False`, so it can never detect reformatting.

   **Concrete change:** replace it with explicit `col_offset` assertions for the `train` body and nested dispatch bodies, or compare the expected three-line additive diff.

No commit-blocking T4 production defect was found.

## A1–A6 verdicts

- **A1 — PASS.** The terminal guard precedes iterator construction, returns `steps_run=0`, logs loudly, and leaves both trees untouched. Z03—the guard moved after iterator construction—is killed by the raising factory test.

- **A2 — PASS.** Reconciliation runs before both the terminal guard and training. Latest-strict-best repair, exact earlier-best acceptance, and unrecoverable-root refusal are correctly separated and tested with both managers.

- **A3 — PASS.** The recorded values are disjoint window means, reset after evaluation, and the decision-level oracle distinguishes them from last-minibatch values. Restarting with an empty window is acceptable: resume checkpoints are written only at evaluation boundaries after the completed window is recorded; a mid-window crash restores the preceding boundary and recomputes the entire window.

- **A4 — PASS.** The replacement is production-callsite strong. The resumed factory receives `schedule.seed + start_step`, and seed-dependent cursors concatenate to the uninterrupted sequence.

- **A5 — PASS.** Construction performs the digest verification; no token, caller rows, digest override, or size override remains.

- **A6 — FAIL.** Direct callback and subclass attacks are refused, but the public decoder/binder injection reproduces arbitrary-content scoring under genuine DEV provenance.

## Four judgment calls

1. **Remove `expected_size`: ACCEPT.** The pinned digest already fixes exact bytes and therefore size/content. Cohort, split, and size checks remain useful defense-in-depth for a future pin change; exercising them by test-only constant monkeypatching is appropriate.

2. **Reject a closure-hidden issue token: ACCEPT.** A closure relocates the secret without eliminating it. Making verified construction the only constructor is the sounder design.

3. **Reconcile before the terminal guard: ACCEPT.** A crash during the final evaluation’s write window is precisely when terminal reopening must repair the artifact before returning.

4. **Fail closed on an unreconcilable sibling: ACCEPT.** This operational behavior change is necessary. Continuing would knowingly produce a run report naming an unavailable checkpoint.

## B1–B4 verdicts

- **B1 — PASS.** At the T4 boundary, the config is exactly 186 inherited keys plus seven additions, with only `model_type` and `eval_data_dir` changed. All pilot defaults match plan v2.5 and Yixun’s decisions. The current working file’s four later T6/T7 additions, producing 197 keys, were excluded.

- **B2 — PASS.** `eval_data_dir` is empty, the manifest path is DEV-64, and its bytes match the pinned digest.

- **B3 — PASS.** The exp_05 contract is preserved. The prefix check retains all five pre-existing conditions in order, while exp_05’s executed dispatch tests still verify each route’s trainer. Appending exp_06 after its arm neither reroutes nor reorders anything ahead of it.

- **B4 — NEW COMMIT-BLOCKER IN PART A.** The non-finite selection defect above is newly identified. No commit-blocking T4 defect was found.

## Deliverables

- **T3b-3 `dev-instrument`: REQUEST-REVISION**
- **T3b-4 `loop-and-selection`: REQUEST-REVISION**
- **T4 `dispatch-config`: APPROVE**

Validation: the exact requested command produced **68 passed, 18 setup errors, 0 assertion failures**; the 18 disk-backed tests could not create temporary directories because this review sandbox is read-only. T4 alone produced **23 passed**. The preserved harness likewise stopped at its first tempfile creation, so I reran its read-only attacks individually. `git diff --check` passed.
