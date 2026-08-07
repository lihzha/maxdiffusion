# rollout_adapter — Codex code review: T1 `exp03-imports`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: **APPROVE first-pass** — zero findings, zero code changes required. The reviewer INDEPENDENTLY verified: the 4,940-byte sampler body equals the pinned blob exactly (sha256 `7e91fbd7…`, git blob `8104edf4…`); all four `side_adapter_wan` dependency segments byte-identical across both trees; all five extracted source-segment hashes match the pinned trainer AND the executable ASTs of `_purpose_id`/`exp03_aux_key`/`rollout_support` are identical; the arithmetic oracle is genuinely independent (hardcodes the offset, recomputes the purpose id, re-derives the fold order). M15 ratification ACCEPTED. All six Coder observations confirmed correctly non-blocking/future-round. Strengthening record: N/A.

**New obligation recorded for T3b (reviewer):** the trainer must pass the LOOP's global step to the aux-key derivation, not a restored `state.step`.

## Reviewer output (verbatim)

## Findings

No commit-blocking findings. No code changes required.

Review-state note: HEAD advanced during review from `c07dc3f` to `26794ce`; the intervening commit changes documentation only. The worktree now contains exactly the three untracked T1 files and no tracked diff.

## Verdicts

- **(a) Import soundness — PASS.** The 4,940-byte sampler body exactly equals `git show 2ef9b8a:.../overfit100_sampling.py`; SHA-256 is `7e91fbd7…` and Git blob ID is `8104edf4…`. All four dependency source segments—`_build_per_token_timestep`, `apply_first_frame_pin`, `build_rollout_sigmas`, and `rollout_timesteps_from_sigmas`—are byte-identical at the pin and current tree.

- **(b) Extraction fidelity — PASS.** All five recorded source-segment hashes match the pinned trainer, constants match exactly, and the executable ASTs of `_purpose_id`, `exp03_aux_key`, and `rollout_support` are identical. The arithmetic oracle in [test_exp03_imports.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_exp03_imports.py:503) is genuinely independent: it hardcodes the offset and independently computes the SHA-256 purpose ID and fold order.

- **(c) Characterization strength — PASS.** The deployed-loop parity covers 25 chained steps, fp32/bf16, and CFG scales 1/5 with exact equality; its AST tripwire binds the copied loop body to the evaluator. The fresh-namespace cold/walk/resume tests adequately prove helper-level restart stability. Ensuring the future trainer passes the loop’s global step—not restored `state.step`—remains correctly assigned to T3b.

- **(d) M15 — ACCEPT.** The hashes are useful re-pin audit records; requiring tests to access non-hermetic Git history would be counterproductive. Independent behavioral derivation plus the verified pin and mutation coverage provide the appropriate gate.

All six reported observations are correctly non-blocking for T1: indexing is equivalent; evaluator duplication is handled by parity; `masked_velocity_mse` and forbidden config access belong to T2; CFG stop-gradient handling belongs to T3a; and the additive ε purpose belongs to T3b.

Focused validation: **40 passed in 5.48s**. Pytest capture required `-s` because the read-only sandbox provides no writable temporary directory.

**APPROVE — the pin is faithful and the characterization is load-bearing for T1’s import boundary.**
