# null_adapter — Codex code review: round R2 `invert-trajectory`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Code under review: working-tree diff of `null_inversion_wan.py` + new `test_null_adapter_invert_trajectory.py` at HEAD `c93c973`. Reviewer ran the suite (21 focused / 66 total passed) and verified the N9/N11/N13 mutation-fix claims directly.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md` — TDD, review, strengthening, and validation requirements.
- `plan_null_adapter.md` v5 — §§3, 5 items 1/12, 6, and 8; also the retained v2 §3 recurrence expanded by v5.
- `null_adapter_worklog.md` — final R1 closure and R2 write-phase entries, including all six Planner positions.
- `null_adapter_codex_code_sigma-embed-noise_review.md` — R1 conventions and strengthening record.
- Working-tree diff for `null_inversion_wan.py` and the complete new inversion test file; HEAD `c93c973`.
- Pinned Wan reference `embedding_search.py:488-500,522-572` from the main checkout at `f370228`.
- `side_adapter_wan.py` timestep, pin, sigma-grid, and timestep-scaling helpers.
- Supplemental SOP context: `null_adapter_yixun_query.md` and exp_02 analysis.
- Validation: focused R2 tests **21 passed**; full worklog suite **66 passed**; Ruff passed.

Findings:

1. **MAJOR** — [null_inversion_wan.py:204](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:204) — `velocity_fn` output shape is not validated. Scalar and batchless `[C,F,H,W]` outputs currently broadcast silently across `[B,C,F,H,W]`, producing plausible but corrupted batched pivots instead of failing closed. After conversion to fp32, require `velocity.shape == current.shape` and raise a targeted `ValueError`; add scalar, missing-batch, and singleton-batch rejection tests matched on that message.

2. **MINOR** — [null_inversion_wan.py:138](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:138), [test_null_adapter_invert_trajectory.py:218](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_invert_trajectory.py:218) — sigma validation accepts `[inf, 1.0, 0.0]` as strictly descending, after which inversion emits invalid arithmetic. Add an explicit `np.all(np.isfinite(values))` guard before monotonicity checking and test both infinities and NaNs with a matched validation message.

The recurrence itself is faithful: indices, sign, evaluation point, clean-pivot/per-step pins, reversed scan ordering, stacking, fp32 arithmetic, and production-geometry `temp_ts` parity all match the reference. The `rtol=1e-6` FMA allowance is acceptable and does not mask the independently tested structural mutations. The narrated N9/N11/N13 survivor fixes are present rather than merely recorded: the horizon expectation is independently hardcoded, and the rejection tests match guard-specific messages. Eager host validation is documented and works when the concrete grid is closed over by a jitted runner.

Ruling on `_f32_bits` duplication:

Keep it duplicated for now. Two self-contained test files do not justify shared test infrastructure; extract it only when a third consumer appears.

Final verdict: **REQUEST-REVISION** — the recurrence and its principal oracles are sound, but two fail-closed validation gaps can silently admit invalid sigma grids or broadcast malformed velocity outputs.

---

## Strengthening record (Coder, same round; both findings fixed, none rejected)

1. **MAJOR (velocity shape) — FIXED.** Trace-time shape-equality guard after the fp32 conversion inside the scan body (static shapes ⇒ fires at trace time, zero per-step cost, jit/scan-safe); seam docstring updated. Tests: three matched rejections (scalar, batchless, singleton-batch at B=2) + an acceptance complement so the guard can't pass by rejecting everything. Direct demonstration recorded: with the guard removed, a batchless velocity yields a correctly-shaped trajectory in which both examples silently share one velocity — plausible output, wrong meaning, no error.
2. **MINOR (non-finite σ) — FIXED.** `np.all(np.isfinite(...))` guard placed before monotonicity/tail checks with its own message; the NaN and −inf cases are matched on the finiteness message specifically, proving guard ordering (they would otherwise be caught by different guards).

Mutation evidence: guard-removal mutants fail 3 tests each against the strengthened suite and **pass the entire pre-strengthen suite 21/21** — both findings substantiated as real gaps, not style; weakened-guard mutants (rank-only shape check; head-only finiteness) also killed. Suite **73 passed in 5.03s** (+7); ruff/py_compile/diff-check clean; +24 executable LOC. `_f32_bits` left duplicated per the reviewer's ruling. Behavior changed beyond the findings: none (the guards ARE the findings) ⇒ no follow-up pass per SOP. Round R2 closed; committed with this record.
