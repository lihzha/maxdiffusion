# pos_context — Codex code review: S7 `trainer-state-checkpoint`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: REQUEST-REVISION — 4 BLOCKER (JIT-callability; logical-GBS unenforced; best-N retention breaks latest-resume; resume loses F2 decision state / continuation never tested through run) + 2 MAJOR (vacuous SGD opt-tree assertion; DEV eval batch-mean vs per-example). All findings ACCEPTED by the Planner; strengthen dispatched same cycle; record follows below.

## Reviewer output (verbatim)

## Findings

1. **BLOCKER** — [wan_pos_context_regression_trainer.py:90](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:90), [wan_pos_context_regression_trainer.py:177](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:177): `train_step` is not JIT-callable. `RegressionTrainState` and `RegressionBatch` are unregistered dataclasses, and converting traced metrics with `float(...)` would subsequently raise concretization errors. A direct probe fails on `state` before tracing. Register/replace both as pytrees, keep metrics as JAX scalars until the host loop, and add an executing `jax.jit(train_step)` oracle.

2. **BLOCKER** — [wan_pos_context_regression_trainer.py:136](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:136), [wan_pos_context_regression_trainer.py:341](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:341): configured logical GBS is never checked against the iterator batch. A 128-example batch under `logical=256, microbatch=64` is silently accepted as four 32-example microbatches. Validate `B == schedule.logical_batch` and microbatch width, with a wrong-sized iterator test.

3. **BLOCKER** — [wan_pos_context_regression_trainer.py:249](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:249), [wan_pos_context_regression_trainer.py:273](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:273): Orbax `best_fn + max_to_keep` retains the best-N checkpoints instead of the latest-N. Consequently, `latest_step()` can be an old “best” state after the actual newest state was deleted, so resume cannot reproduce uninterrupted training. It also does not guarantee the earliest checkpoint survives metric ties. Preserve latest resume state separately from the immutable earliest-best selection.

4. **BLOCKER** — [wan_pos_context_regression_trainer.py:335](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:335), [test_pos_context_trainer.py:523](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_trainer.py:523): the claimed continuation test manually restores and applies explicitly selected batches; it never exercises `WanPosContextRegressionTrainer.run`, the resumed iterator, manager retention, or stop-state continuation. `run` also resets history/window, losing running best, streak, and previous train MSE. Add an integrated interrupted-vs-uninterrupted oracle and restore the minimal F2 decision state, potentially through checkpoint metrics while retaining the three-item payload.

5. **MAJOR** — [test_pos_context_trainer.py:294](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_trainer.py:294): the real-model test does prove frozen Wan leaves remain bit-unchanged and that adapter leaves move, but its optimizer-tree assertion is vacuous because plain SGD has stateless `EmptyState`. Use Adam or momentum and assert its parameter-shaped state matches only the adapter tree.

6. **MAJOR** — [wan_pos_context_regression_trainer.py:193](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:193): DEV evaluation averages batch means equally, not per-example normalized losses. Unequal final batches therefore change the selection metric. Accumulate normalized-loss sums and example counts, or enforce equal batch sizes.

## Positions

- **(1) Ratified for `stop_verdict`:** prior-only running best, previous-eval train comparison, strict `>`, earliest tie, and first-eval behavior are correct. **Contested end-to-end** because Orbax tie retention and resumed history do not preserve them.
- **(2) Ratified for uninterrupted runs:** train windows are non-overlapping. **Contested across resume** because history/window reset.
- **(3) Ratified:** patchify → block-0 → head is the appropriate cheap half; fp32 conversion remains in S6’s loss.
- **(4) Contested as the complete mechanism:** `best_fn/min` preserves good checkpoints but breaks latest-state resume and earliest-tie retention.
- **(5) Ratified:** optional config reads and defaults are sound; S8 must declare both keys before K3 accumulation.
- Accumulation equivalence is **tolerance-based**, not bitwise (`atol=1e-6`, `rtol=0` for parameters). Exact name/target partitioning is pinned, but the configured logical batch is not.
- `seed = config.seed + start_step` is pinned both as arithmetic and at the iterator call, but full resumed-continuation equivalence is not.

The requested pytest run and collect-only fallback were both environment-blocked because no writable temporary directory is available; collection reached TensorFlow/absl before failing.

## Verdict

**REQUEST-REVISION** — JIT execution, logical-batch enforcement, and checkpoint/resume semantics currently violate central S7 contracts.

## Follow-up reviewer output (verbatim)

## Per-finding verdicts

1. **Closed.** Both dataclasses are registered pytrees, with `RegressionBatch.names` static, and metrics remain JAX scalars ([batch](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_regression.py:46), [state/step](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:91), [oracle](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_trainer.py:266)). The original executing JIT/eager probe passed directly.

2. **Closed.** `checked_training_batch` enforces the logical batch and configured microbatch width at consumption, including the named 128-under-256/64 failure and loop-level guard ([implementation](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:155), [tests](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_trainer.py:301)). Direct probes passed.

3. **Closed.** Resume retention is recency-only with no `best_fn`; selection uses the disjoint `<ckpt_dir>_selection` sibling, `max_to_keep=1`, and a strict metric comparison ([managers](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:266), [selection](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:292)). Tests evict the best-DEV resume step, preserve the earliest tie, and independently restore selection.

4. **Not fully closed.** History is correctly atomic with the three-item checkpoint, restored by default, and the integrated mid-streak interruption tests are substantive ([checkpoint JSON](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:311), [integrated tests](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_trainer.py:922)). The installed Orbax version only writes manager metrics when `best_fn` enables best tracking, so using the step JSON is the correct design. However, a restored history whose verdict is already terminal is ignored before entering the loop.

5. **Closed.** The real freeze oracle now uses Adam and requires its parameter-shaped leaves to equal exactly two copies of the adapter tree while frozen leaves remain bit-identical ([test](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_trainer.py:404)). It passed directly.

6. **Closed.** DEV evaluation pools normalized per-example sums/counts, and the 4+1 oracle distinguishes this from mean-of-means ([implementation](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:209), [oracle](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_trainer.py:839)). Direct probe passed.

## Incident and residue

The sibling layout eliminates the nested-manager temporary-directory collision, and its docstring accurately records the failure mechanism. The trainer SHA-256 is exactly the recorded baseline, `fe13f5ea…47d0df`, byte-identical to the mutation backup. C7’s nested-root signature is absent; the mutation script contains 43 signatures, and the strengthened A9 and C11 tests are present. The S6 diff is exactly `+9/−3`, limited to pytree registration, static `names`, and explanatory text.

Tolerance-based accumulation equivalence is accepted: fp32 reductions differ by association, while `1e-6` is safely separated from the approximately `1e-1` semantic mutants. Bitwise equality would be inappropriate.

The requested full pytest command remained blocked before collection because the read-only sandbox has no writable temporary directory. The JIT, logical-batch, accumulation, pooled-DEV, and real Adam-freeze oracles passed when invoked directly.

## Finding

**MAJOR** — [run:443](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:443): `run` computes the restored `StopVerdict` but enters training regardless. A direct probe restored a terminal step-5 history with `verdict.stop=True`; the trainer nevertheless executed step 6. At production cadence, a retry after a terminal checkpoint can perform another 1,000 optimizer steps and advance the resume state. Return immediately when restored history is terminal, with a reopen-after-stop regression test.

## Verdict

**REQUEST-REVISION — the six requested strengthens are otherwise sound, but a restored terminal F2 checkpoint is not treated as terminal.**

## Closing record

The follow-up closed findings 1-3, 5, 6 and the incident/residue assessment; its one remaining MAJOR (restored terminal verdict not honored at run:443) was fixed exactly as prescribed: the guard returns before the input pipeline is built (an iterator-factory AssertionError proves it is never called), report shape preserved, steps_run=0, disk state pinned unchanged. Red-first; guard-removed and guard-falls-through mutants killed. Focused 59; full suite 1243; battery total 45/45; trainer sha matches the battery baseline. Closed WITHOUT a third Codex pass — the fix is the reviewer's own prescription with zero deviation, recorded here per the review-budget discipline (issue #9); any challenge lands in the next round's review.
