# pos_context — Codex code review: S8 `dispatch-config`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: APPROVE first-pass — zero findings; all six Planner positions ratified; the reviewer independently verified the 186-key baseline retention, the exact five-key addition, and the F3 sharding identity. Strengthening record: N/A.

## Reviewer output (verbatim)

### 1. Findings

No commit-blocking findings.

- Dispatch is exactly additive at [train_wan.py:42](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/train_wan.py:42); all prior branches and fallback are unchanged.
- The characterization executes all five named routes plus four fallback cases and calls `start_training`.
- The YAML retains all 186 baseline keys, adds exactly five POS training keys, and changes only the four intentional experiment values.
- All F3 mesh, logical-axis, data-sharding, and ICI/DCN parallelism values are identical to the side-adapter YAML.
- Trainer reads are exactly `checkpoint_dir`, `eval_every`, `max_train_steps`, `pos_logical_batch`, `pos_microbatch`, and `seed`; all are declared correctly. S6 has no config reads.
- No `embedding_slot`, inversion `pos_artifact*`, or `null_*` key leaks.
- Empty `checkpoint_dir` safely produces no managers at [wan_pos_context_regression_trainer.py:429](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:429).

The requested pytest run was blocked before collection because the read-only sandbox has no writable temporary directory. Direct AST/YAML checks passed, including `git diff --check`.

### 2. Planner positions

All six ratified:

1. Config-only construction with optional S9 seams.
2. Named `NotImplementedError("…S9…")` boundary.
3. Reuse of `checkpoint_dir`.
4. `256 == 256` as the no-accumulation sentinel; `64 × 4 == 256` remains valid.
5. Preserve `train_wan.py`’s upstream two-space formatting.
6. AST-executed dispatch characterization; repository search confirms there was no prior dispatch test to mirror.

### 3. Verdict

**APPROVE** — the S8 dispatch/config wiring satisfies the stated contract without a commit-blocking defect.
