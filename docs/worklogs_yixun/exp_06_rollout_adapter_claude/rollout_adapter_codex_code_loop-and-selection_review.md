# rollout_adapter — Codex code review: T3b-4 `loop-and-selection` + T3b-3/A2 strengthens (combined)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **T3b-1 APPROVE · T3b-2 APPROVE** (doc cleanup only) · **T3b-3 REQUEST-REVISION · T3b-4 REQUEST-REVISION** — 5 BLOCKERs + 1 MAJOR, **every one EXECUTED by the reviewer rather than hypothesized**, all accepted by the Planner.

**PASSED:** A-B3 (`eval_index` reaches provenance only; the key depends on fixed seed, name, field and whitelisted replicate — no step/cadence path remains). A-A2 (the CLI override becomes nonzero `nnx.Dropout` nodes and the constructed-module scan catches them even when the caller omits `dropout_rate`; the declared-rate path is closed too). B1 — **the standing randomness obligation is DISCHARGED**: production uses the loop counter, never reads `state.step`, and the real Orbax interrupted/uninterrupted draw comparison is appropriate.

**BLOCKERs, with the reviewer's executed evidence:**
- **B-1 — a restored TERMINAL verdict runs another optimizer step.** `run_loop` computes the verdict then enters the loop unconditionally; restoring a terminal step-4 history produced `[('iterator', 4), ('update', 5), ('save', 5)]`. **This is the identical defect exp_05's S7 hit and fixed.**
- **B-2 — the sibling selection artifact is not interruption-safe.** A crash between the resume save and the selection update leaves the best state only in the resume tree and startup never reconciles: simulated `history_best 2` with `selection_replay_calls [(4, 0.9)]` — an empty selection tree ships the worse step 4; a populated one goes stale.
- **B-3 — the stop rule reads ONE minibatch, not the evaluation-window mean.** S7 averaged the disjoint window; this loop overwrites `train_metric` per update and records the last. For updates `[100, 0, 50, 0]` it records `[0, 0]` where the window means are `[50, 25]` — flipping "train metric still falling" and therefore the stop decision.
- **A-B1 — TEST still reachable, two executed forgeries.** `_ISSUE_TOKEN` is a module attribute (passing it built a cohort containing the TEST name and produced a draw); and, needing no private access, the **public `expected_sha256` override** let the reviewer build a 64-row DEV-labelled manifest whose first row is the genuine TEST row, supply its computed digest, load it and draw.
- **A-B2 — the Planner's structural ruling was not carried far enough.** The reviewer supplied a `batch_loader` ignoring every validated row and returning TEST content; it scored `1.0` and was stamped with genuine DEV cohort and digest. *"TEST tensors under DEV keys has no expression" is false: the expression is `lambda row: test_batch`.* An unrestricted callback preserves the original trust hole in different syntax — an instrument-owned canonical row reader (opening `shard_path`/`ordinal`, verifying decoded identity) is required.
- **MAJOR — X15's replacement is not production-callsite strong:** the fixture discards the seed and the helper is tested in isolation, so `run_loop` could switch to `batches(schedule.seed)` with both tests still green.

**Correction accepted:** W12's *survival* is not itself the structural proof — the step-free call graph and signatures are; pinning the exact key signature strengthens the argument. The Planner had endorsed the weaker framing.

**PLANNER SELF-ASSESSMENT (recorded):** B-1 is a Planner miss. The T3b-4 brief named S7's retention design and its interrupted-vs-uninterrupted oracle but **never named the terminal-resume guard**, which the Planner personally adjudicated in exp_05's S7 follow-up. A cross-experiment lesson failed to transfer because the Planner did not carry it into the brief. Standing correction: **when a round generalizes a design from a prior experiment, the brief must enumerate that design's KNOWN DEFECTS, not just its structure.**

**Sandbox caveat:** 51 passed / 6 environment setup errors (the read-only sandbox could not create its pytest tmp dir); no assertion failures. The executable in-memory attacks all ran successfully.

## Reviewer output (verbatim)

210,891
# Review verdict

**REQUEST-REVISION.** A-B3 and A-A2 are closed, and the narrow restored-`state.step` obligation is discharged. A-B1/A-B2 remain exploitable, and T3b-4 has three commit-blocking interruption/selection defects.

## Part A findings

1. **BLOCKER — A-B1 remains reachable through two executable forgeries.**  
   [pos_rollout_dev_instrument.py:97](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:97) leaves `_ISSUE_TOKEN` accessible on the module. Supplying `instrument._ISSUE_TOKEN` constructed a cohort containing `ep61399_v0_s00000` and produced a draw.

   More importantly, the public `expected_sha256` override at [line 183](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:183) recreates the original attack without private access: I built a 64-row DEV-labelled manifest whose first row was the genuine TEST row, supplied its computed digest, loaded it successfully, and drew for the TEST name.

   Concrete change: remove the caller-controlled digest override and always compare against `J0_DEV64_SHA256`. Also stop exposing the issue secret as a module attribute, and add the forged-file attack—not only three guessed constructor tokens—to the tests.

2. **BLOCKER — A-B2’s residual is not acceptable.**  
   [score_dev_cohort:287](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:287) calls an arbitrary `batch_loader(row)` but cannot require it to use the row. I supplied a loader that ignored every validated row and returned TEST-labelled content; the result was scored as `1.0` and stamped with the genuine DEV cohort and digest.

   Thus “TEST tensors under DEV keys has no expression” is false: the expression is simply `lambda row: test_batch`.

   Concrete change: use an instrument-owned canonical row reader that opens the row’s `shard_path`/`ordinal` and verifies the decoded example identity before scoring. An unrestricted callback preserves the original trust hole in a different syntax.

3. **MINOR — the prior malformed documentation remains.**  
   [pos_rollout_arms.py:47](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_arms.py:47) still says “`accumulation behaviour. is an ARGUMENT`”.

## Part B findings

1. **BLOCKER — a restored terminal verdict executes another optimizer step.**  
   [run_loop:331](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_loop.py:331) computes `stop_verdict(history)` but enters the loop unconditionally. I restored a terminal step-4 history; `run_loop` built the iterator, executed step 5, and saved it:

   ```text
   terminal_resume_attack 1 5 [('iterator', 4), ('update', 5), ('save', 5)]
   ```

   This is the exact terminal-resume defect previously fixed in exp_05.

   Concrete change: return a zero-step `RunReport` before constructing `batches` whenever the restored verdict is terminal. Test with an iterator factory that raises if called.

2. **BLOCKER — the sibling selection artifact is not interruption-safe.**  
   The resume checkpoint is saved at [line 365](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_loop.py:365), then selection is updated separately at line 374. A crash between them leaves the best state only in the resume tree. Startup never reconciles selection from restored history.

   I simulated a crash after saving best step 2. After restart, history still selected step 2, but the only selection call was the worse step 4:

   ```text
   history_best 2  selection_replay_calls [(4, 0.9)]
   ```

   With an empty selection tree, step 4 becomes the shipped artifact; with an older incumbent, the artifact remains stale.

   Concrete change: reconcile selection before training. If the restored latest evaluation is the strict historical best, repair the sibling from the restored state; otherwise require the sibling to match the recorded best or fail closed. Add a real crash-window restart test including both managers.

3. **BLOCKER — the generalized S7 stop rule uses one minibatch, not the evaluation-window train metric.**  
   [run_loop:349](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_loop.py:349) overwrites `train_metric` each update and records only that last value at line 362. S7 averaged the disjoint window since the preceding evaluation. For update metrics `[100, 0, 50, 0]`, this loop records `[0, 0]`; the window means are `[50, 25]`. That changes whether “train metric still falling” is true and therefore changes the stop decision.

   Concrete change: accumulate metrics between evaluations, record their mean, reset after each evaluation, and add the inherited disjoint-window-versus-last-step/cumulative-average oracle.

4. **MAJOR — X15’s replacement test is not production-callsite strong.**  
   [test_pos_rollout_loop.py:56](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_loop.py:56) discards the seed, while [the X15 replacement](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_loop.py:263) tests only `resume_seed` in isolation. `run_loop` could change to `batches(schedule.seed)` and both the integrated restart test and helper test would remain green.

   Concrete change: make the iterator factory record its argument or produce seed-dependent batches, then assert the resumed production call receives `schedule.seed + start_step` and continues the uninterrupted data sequence.

## Requested verdicts

- **A-B1 — FAIL.** The guessed-token test passes, but the actual module token and the public arbitrary-digest loader both reproduce the TEST attack.
- **A-B2 — FAIL.** Single-read hashing and retained rows are correct, but caller-selected tensors remain expressible and receive DEV provenance.
- **A-B3 — PASS.** `eval_index` reaches provenance only. The key depends on the fixed seed, name, field, and whitelisted replicate; no step/cadence path remains.
- **A-A2 — PASS.** The CLI override becomes nonzero `nnx.Dropout` nodes in `WanModel`, and the constructed-module scan catches those even if the caller omits `dropout_rate`. The declared-rate path is also closed.

- **B1 — PASS, narrowly.** The standing randomness obligation is discharged: current production code uses the loop counter, never reads `state.step`, and the real Orbax interrupted/uninterrupted draw comparison is appropriate. The broader interruption contract still fails for the reasons above.
- **B2 — FAIL.** Normal-path retention, strict tie handling, pure stop-rule edges, and metadata are correct; terminal resume, train-window semantics, and cross-manager crash recovery are not.
- **B3 — PARTIAL/FAIL overall.** X04’s `all_steps() == (latest,)` replacement is strong. X15’s helper-only test is insufficient.
- **B4 — PASS with correction.** Rejecting W12 as inert and replacing it with a faithful mutation was correct. Its survival is not itself structural proof; the step-free call graph and signatures are the proof. Pinning the exact key signature would make that argument stronger.
- **B5 — FAIL.** Three new commit-blocking code defects and one production-path test gap were found.

## Deliverables

- **T3b-1 `step-stream`: APPROVE**
- **T3b-2 `arm-losses`: APPROVE** — nonblocking doc cleanup remains.
- **T3b-3 `dev-instrument`: REQUEST-REVISION**
- **T3b-4 `loop-and-selection`: REQUEST-REVISION**

Validation: the available focused tests produced **51 passed and 6 environment setup errors** because this read-only sandbox could not create `/private/tmp/pytest-of-yixunhu`; none were assertion failures. The executable in-memory attacks above ran successfully. `git diff --check` passed.
