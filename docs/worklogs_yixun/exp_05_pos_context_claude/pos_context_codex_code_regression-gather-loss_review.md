# pos_context — Codex code review: S6 `regression-gather-loss`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: APPROVE — zero production findings; all five Planner positions ratified; one LOW test-strength note (float64 accumulation not strictly pinned) closed in-cycle by the strengthen recorded below.

## Reviewer output (verbatim)

### Findings

- **LOW (non-blocking)** — [test_pos_context_gather_loss.py:303](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_gather_loss.py:303): population semantics and axes are pinned, but float64 accumulation is not. A float32-Chan probe differs by `1.55e-7` yet passes the current `rtol=1e-6`. Consider a strict fp32 comparison against the float64 oracle or a high-offset/low-spread case. No production-code defect found.

The gather oracles genuinely pin independent target/state rows and timestep indexing. Focused suite: **44 passed**.

### Positions

1. **Ratified:** population variance per step across all TRAIN-cache target elements, streaming Chan in float64.
2. **Ratified:** normalize each example by its selected step’s variance, then average; the `0.75` oracle rejects batch-MSE ÷ mean variance.
3. **Ratified:** retain G12’s redundant fp32 casts as defense-in-depth within the valid K2-record domain.
4. **Ratified:** exclude `z_i0`; cached pre-step `z̄_t` is already pinned and is the block-0 input.
5. **Ratified:** four-field duck typing is appropriate for validated lightweight K2 views.

### Verdict

**APPROVE — the gather alignment, fp32 losses, variance table, and normalized metric are plan-faithful and sound.**
