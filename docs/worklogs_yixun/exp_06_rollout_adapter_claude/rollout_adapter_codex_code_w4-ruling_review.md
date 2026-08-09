# rollout_adapter — Codex ruling: W4 verification + M1

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **W4: REQUEST-REVISION (verification only) · M1: NOT-READY (minimal set = repair the oracle) — BUT: production PASS on every item, no launch-blocking production defect found, the epsilon question formally RULED (data placement ACCEPTED as the recorded decision), and the FINAL launch-caveat list supplied.** The remaining blocker is that the oracle still lowers with shared params/opt_state/draws (only batch differs), compares compiled shardings relatively, and never lowers the scorers — verification debt, not code debt.

**The recorded epsilon ruling:** *"Epsilon is per-example and batch-major, so placing it on the loader split is the natural contract. Replication would consume ~106 MB/chip at GBS 256 without correctness benefit. 'Match old XLA choice' is not pin-able: the prior choice was implicit, compiler-dependent, and not recorded as a stable contract. M1 should measure and authorize the newly explicit data-sharded choice."*

**The final launch-caveat list (verbatim, to appear in the M1 package):** commit/push the complete tree first (never the dirty tree); `POS_DEVICE_COUNT=8` on v6e-8 (derives per-device 32 at GBS 256); exact pushed SHA + command + acceptance criteria + log path + monitoring entry recorded; sweep leftover `checkpoint_dir/_m1_probe/**` after best-effort cleanup; authorization requires peak ≤90%, zero reservation failures, `peak_source` ∈ {runtime-reset, runtime-raised} — compiled-analysis floors and refused-allocation capacity may REFUSE, never authorize; epsilon data-sharded — M1 authorizes that explicit choice; M1 authorizes ONLY its exact measured arm/microbatch/k cells; k=4 exploratory.

## Reviewer output (verbatim)

164,568
## 1. Findings

1. **BLOCKER — the required own-operands/absolute-lowering oracle is not actually present.**

   The production fix is correct: `place_step_inputs` places every update operand, and both `step()` and `.lower()` enter it ([pos_rollout_update.py:330](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:330), [pos_rollout_update.py:415](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:415)).

   But the oracle lowers both programs with:

   ```python
   program.step.lower(left.params, left.opt_state, batch, right.draws)
   ```

   ([test_pos_rollout_trainer_wiring.py:1244](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1244)). Only `batch` differs. M1 does not lower with `right.params/right.opt_state`; production does not lower with independently produced trainer draws. The assertion named `lowered_with_own_inputs` checks only the two batch objects ([test_pos_rollout_trainer_wiring.py:1302](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1302)).

   Absolute helper-output checks are strong and use exact sharding-object equality ([test_pos_rollout_trainer_wiring.py:1261](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1261)), but the compiled executable’s input/output shardings are still compared only by relative string equality ([test_pos_rollout_trainer_wiring.py:1298](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1298)).

   The scorer is now correct in production, but its oracle remains weak: `score_is_shared` means only that both fields are non-`None`, and neither scorer is lowered and compared. M1’s score arity is evaluated using `left.params`, not M1’s own params ([test_pos_rollout_trainer_wiring.py:1344](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1344)).

   **Minimal change:** lower production with a complete left-owned operand tuple and M1 with a complete right-owned tuple; compare tree structures; assert the lowered input shardings against independently constructed absolute production specs, not only each other; and lower both shared scorers on their own scoring operands.

2. **LOW — the stale-probe response improved the spelling but did not eliminate spelling dependence.**

   `W3-1` still searches for exact source substrings such as `jax.device_put(adapter_params, replicated)` ([reviewer_attacks.py:1837](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:1837)). `W4-1` is similarly source-spelling based. An equivalent refactor can therefore produce another false `SUCCEEDED`. The README caution is correct, but the probe is not genuinely behavioral.

   Two legacy oracle fields also still recognize `spec=P()` by substring, although the exact `_contract` checks make those particular fields redundant.

## 2. Verdicts on 1–5

1. **Blocker: production PASS; mandated oracle FAIL.** No update operand reaches the compiled update unplaced. The separate scorer remains outside `place_step_inputs`; both paths use the shared scorer construction, but its lowering equivalence is not proved. The oracle still shares params/optimizer/draws and asserts compiled shardings relatively.

2. **Epsilon placement: ACCEPT DATA PLACEMENT.** Epsilon is per-example and batch-major, so placing it on the loader split is the natural contract. Replication would consume approximately 106 MB per chip at GBS 256 without a correctness benefit. “Match old XLA choice” is not a pin-able alternative: the prior choice was implicit, compiler-dependent, and not recorded as a stable contract. M1 should measure and authorize the newly explicit data-sharded choice.

3. **Optimizer/scorer/authorization:**
   
   - Optimizer placement: **PASS** — every leaf is explicitly replicated ([pos_rollout_update.py:404](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:404)).
   - Shared scorer runtime path: **PASS** — M1 times `program.score` with `(loss, aux)` intact ([pos_rollout_fit_probe.py:1505](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1505)).
   - Scorer oracle: **FAIL** as described above.
   - `authorized_context()` literally at `body[0]`: **PASS**, independently asserted in three test locations.

4. **Three self-reported failures:**
   
   - Discarding the contaminated battery and rerunning serially with restore digests: **correct response**.
   - Exact sharding equality: **correct for the main `_contract`**, with only redundant substring assertions remaining.
   - Stale W3-1 probe: **partially corrected**; the README rule is good, but W3-1/W4-1 remain spelling-sensitive.
   
   Static reconciliation confirms 80 `_report` calls, 80 `REFUSED` log lines, and zero `SUCCEEDED`.

5. **New W4 defects:** no launch-blocking production defect found. One launch-blocking verification defect remains: the oracle does not satisfy W3’s explicit own-operands/absolute-lowering requirement.

## 3. THE RULING

**NOT-READY.**

Minimal set: repair and rerun the oracle so both update and scorer lower with each path’s complete own operand tuple, and assert the lowered input contract against independent absolute production shardings.

Once that passes, the carried launch caveats are:

- Commit and push the complete production tree first; current HEAD `004a7dc` contains documentation while the reviewed code remains dirty/untracked.
- Launch M1 on v6e-8 with `POS_DEVICE_COUNT=8`; at GBS 256 this derives per-device batch 32.
- Record the exact pushed SHA, command, acceptance criteria, log path, and monitoring entry.
- Sweep and remove any leftover `checkpoint_dir/_m1_probe/**` paths after the best-effort cleanup.
- Authorization requires peak ≤90%, zero reservation failures, and `peak_source` equal to runtime-reset or runtime-raised. Compiled-analysis floors and refused-allocation capacity may refuse, never authorize.
- Epsilon is data-sharded; M1 authorizes that explicit choice.
- M1 authorizes only its exact measured arm/microbatch/k cells; k=4 remains exploratory.
- Under v2.7, v6e-8 M1 can authorize M2 only. M3 requires separate v6e-64 M1′.
- Under v2.8, a preempted M2/M3 pair is non-quotable and both arms restart fresh; never resume only one arm.

I could not execute pytest/JAX because this reviewer environment lacks both. `git diff --check`, both launcher syntax checks, and AST parsing of all 19 changed Python files passed.

## 4. W4

**REQUEST-REVISION.**
