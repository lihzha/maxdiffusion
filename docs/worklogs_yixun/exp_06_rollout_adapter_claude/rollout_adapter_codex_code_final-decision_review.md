# rollout_adapter — Codex FINAL decision review: W2/W2b/F3a-fix + M1 ruling

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **RULING: NOT-READY.** F3a-fix CLOSED with no finding; W1's A3/A5 spot-verified still standing; the W2b width derivation APPROVED. One new BLOCKER + 3 MAJOR + 1 LOW + 1 evidence gap.

**THE BLOCKER, and it is material to M1's purpose: shared factories ≠ shared PROGRAM.** The trainer replicates adapter parameters and compiles under `logical_axis_rules`; M1 returns an unsharded tree and measures under a bare `with mesh`. The model translates logical axes during the forward (`side_adapter_wan.py:634`), so **M1 can compile different activation shardings and report a different per-device HBM peak — the very quantity it authorizes.** M1 also closes over a ZERO null context and privately constructs its grid, where training uses the settled loader's REAL null context and scheduler. Fix: share the entire program-finalization boundary (context, shardings, mesh, axis-rule scope, jitted update), with a multi-device oracle comparing leaf values, shardings and LOWERED input/output shardings on both paths.

**MAJORs:** the advertised value comparison compares SHAPES only; the freeze oracle observes THREE different transformer objects (identity-disjointness automatic ⇒ vacuous); **interrupted≡uninterrupted is FALSE for the deployed reseeding loader** — the reviewer's own alternative adopted as plan v2.8: *a preempted training pair is NON-QUOTABLE; both arms restart from a fresh attempt* (resuming one arm breaks the same-batch causal comparison; checkpointing a data cursor would change the settled loader). **The recipe lock is check-then-write over an overwriting storage helper** — concurrent divergent arms can both observe absence; needs atomic create-if-absent + the lock digest persisted in each attempt publication (M2/M3-gating, with the already-recorded `publish_attempt` path binding). **LOW:** `assert_cell_authorized` is before pipeline-load but not literally first. **Evidence gap:** the harness source/logs live only in the session scratchpad, invisible to the review sandbox — to be preserved IN-REPO as review-package artifacts.

## Reviewer output (verbatim)

263,896
# Final ruling: NOT-READY

The prior peak-source and manifest-framing blockers are closed, and F3a-fix is sound. However, M1 still does not compile the same sharded program as the live trainer.

## Findings

### Part A — W2

1. **BLOCKER — M1 and training share factories, but not the resulting deployed program.**

   The trainer replicates adapter parameters and runs compilation/execution under logical axis rules at [wan_pos_rollout_trainer.py:564](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:564). M1 performs neither operation: it returns an unsharded parameter tree from [pos_rollout_fit_probe.py:1366](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1366) and measures under `with mesh` only at [pos_rollout_fit_probe.py:1465](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1465).

   This is material because the model translates logical axes during the forward at [side_adapter_wan.py:634](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/models/wan/side_adapter_wan.py:634). M1 can therefore compile different activation shardings and report a different per-device HBM peak.

   M1 also closes over a zero null context and privately constructs its grid at [pos_rollout_fit_probe.py:1317](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1317), whereas training uses the settled loader’s real null context and scheduler at [wan_pos_rollout_trainer.py:511](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:511).

   **Concrete change:** share the entire program-finalization boundary—context, replicated parameter/optimizer shardings, mesh, axis-rule scope and jitted update—not merely the three factories. Add a multi-device oracle comparing leaf values, shardings and lowered input/output shardings on both paths.

2. **MAJOR — the advertised “value comparison” compares no values.**

   [test_pos_rollout_trainer_wiring.py:557](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:557) compares parameter shapes/dtypes and optimizer tree structure only. It performs no `array_equal`/`allclose` comparison and cannot catch the sharding divergence above.

   **Concrete change:** compare all adapter and optimizer leaves by value and sharding.

3. **MAJOR — the freeze end-to-end proof observes different transformers.**

   [test_pos_rollout_trainer_wiring.py:643](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:643) constructs one `_tiny_transformer()` before training and another afterward. The actual backbone supplied to `start_training()` is a third object. Identity-disjointness is consequently automatic, and equal initialization does not prove the trained closure’s backbone remained unchanged.

   **Concrete change:** retain the exact transformer returned by the loader stub and compare that object’s parameter leaves before and after execution.

4. **MAJOR — “interrupted ≡ uninterrupted” is false for the deployed loader.**

   The positive resume oracle explicitly substitutes a cursor-aware loader at [test_pos_rollout_trainer_wiring.py:738](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:738). The deployed-behavior test then proves resumed metrics differ because `seed + start_step` reshuffles rather than continues at [test_pos_rollout_trainer_wiring.py:778](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:778).

   **Concrete change:** either checkpoint a deterministic data position or make the runbook declare any preempted training pair non-quotable and restart both arms from step zero. Resuming only one arm breaks the same-batch causal comparison.

5. **LOW — `assert_cell_authorized` is not literally first.**

   `start_training()` loads the DEV manifest before deriving and checking authorization at [wan_pos_rollout_trainer.py:742](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:742). It remains before pipeline loading, so this is not an M1 blocker, but the stated ordering and test claim should be corrected or reordered.

6. **Previously recorded M2/M3 blocker remains: resume markers are not path-bound.**

   `publish_attempt` accepts arbitrary `checkpoint_dir`, while `load_publication` checks only that it is nonempty at [wan_pos_rollout_trainer.py:214](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:214). Selection never derives `<parent>/<attempt>/checkpoints`.

   **Concrete change:** derive and validate checkpoint paths from the marker location and require payload attempt, parent and checkpoint existence to agree.

### Part B — W2b and F3a-fix

1. **MAJOR — W2b width arithmetic is correct, but the cross-topology recipe-lock claim is not absolute.**

   Removing `PER_DEVICE_BATCH_SIZE`, requiring `POS_DEVICE_COUNT`, deriving the width at [train_wan_pos_rollout.sh:177](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:177), and verifying after distributed initialization are all correct. The derived value is fingerprinted and locked.

   However, `publish_recipe_lock` remains check-then-write at [wan_pos_rollout_trainer.py:172](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:172), and the storage helper overwrites using `"wb"` at [pos_rollout_support.py:245](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_support.py:245). Concurrent divergent arms can both observe absence and overwrite; deletion or a different run/root also resets the comparison.

   **Concrete change:** use stable pair identity plus atomic create-if-absent/CAS semantics and persist the adopted lock digest in each attempt publication.

2. **F3a-fix — no finding.**

   The loader now obtains the scheduler through the settled trainer, uses its sigma bounds and training horizon, then immediately validates the grid at [eval_wan_pos_rollout.py:1429](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1429). The real-`HyperParameters` repro still raises on `config.num_train_timesteps`, confirming the original defect. The bf16 tiny-backend execution and disagreeing-scheduler refusals are substantive.

3. **EVIDENCE GAP — B3 cannot be independently verified.**

   The stated `scratchpad/attacks_after_w2*.log` files and harness source are absent from this worktree. Only the summary in [rollout_adapter_worklog.md:371](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_worklog.md:371) remains.

   **Concrete change:** preserve the 75-probe harness source and final log in the review package. I cannot confirm that no other probe silently measures nothing without them.

## Item verdicts

| Item | Verdict | Assessment |
|---|---|---|
| A1 | **PASS with low ordering correction** | Settled-trainer provider choice, scheduler authority, loop composition and once-after-return publication are sound. |
| A2 | **FAIL** | Shared factory identity does not establish identical axis rules, shardings, context or compiled program. |
| A3 | **FAIL** | Most end-to-end properties pass, but the freeze oracle is vacuous and deployed resume is not trajectory-equivalent. |
| A4 | **FAIL / mixed** | Once-only publication issues are runbook items; deployed reseeding and current unmeasured sharding are training blockers. |
| B1 | **FAIL** | Width derivation is approved; the claimed recipe-lock enforcement remains racy/routable. |
| B2 | **PASS** | C3 is closed. |
| B3 | **FAIL — evidence unavailable** | Harness/logs are missing from the supplied tree. |

W1 spot checks: **A3 peak-source fix PASS** and **A5 length-framed manifest fix PASS**.

## Part C — NOT-READY

Minimal M1 blocking set:

1. Make M1 compile and measure the live trainer’s exact sharded program: same pipeline mesh, real scheduler/null context, replicated adapter and optimizer state, logical-axis-rule scope, and jitted update. Add a two-sided value-and-sharding oracle.

After that passes rereview, commit the complete reviewed tree before launching; the current dirty tree must never launch.

The recipe-lock, resume-path binding and deployed-loader cursor findings do not block the fit-probe job itself, but they must be closed—or converted into explicit restart-both-arms policy—before M2/M3.

## Round verdicts

- **W2: REQUEST-REVISION**
- **W2b: REQUEST-REVISION**
- **F3a-fix: APPROVE**

Validation performed: HEAD `c0b8245` confirmed; `git diff --check`, both launcher `bash -n` checks, Python AST parsing and YAML parsing passed. This container lacks `pytest`; the supplied **2111 passed, 0 failed** remains the authoritative suite result.
