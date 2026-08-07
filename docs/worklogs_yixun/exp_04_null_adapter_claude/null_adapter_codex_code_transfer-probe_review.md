# null_adapter — Codex code review: R12-lite `transfer-probe` (J1c enabler)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: APPROVE first-pass — zero findings; row/ε₀ identity independently verified; all three Planner positions ratified. Strengthening record: N/A.

## Reviewer output (verbatim)

### Findings

No launch-blocking defects found.

- Row identity is preserved: producer returns `[N,B,L,D]`, writer stores unchanged, and replay consumes `nulls[i]` as `[B,L,D]` without transposition or row selection.
- J1b and J1c both derive the own basin from the identical imported `global_noise(0)` helper ([null_adapter_modes.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:674), [null_transfer_probe.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_transfer_probe.py:100)).
- Refusals validate `[25,8,16,4096]` through runtime σ-grid, cohort names, plan `l_null`, and `base_context` width; cohort mismatch is rejected, never truncated.
- Provenance hashes the fetched NPZ bytes and records its URI, J1c code SHA, manifest hash, model revision, names, and recipe.
- Focused tests: **34/34 passed**; ruff and `git diff --check` passed. The broad run confirmed **989 collected** (`955 + 34`); its remaining failures were caused by the read-only environment lacking writable temp/OpenMP storage.

### Positions

1. **Approve `read_bytes` as a module seam.** It is single-input object-store glue; adding it to shared output-oriented `Sinks` would unnecessarily widen established interfaces and merge surface.
2. **Approve four settings.** `global(0)` is the sole own-basin control; keyed `{0,1,2}` is the established foreign-noise estimand. Additional global draws would measure a different, redundant convention.
3. **Approve refusal over truncation.** Exact example-axis equality protects row identity and prevents a full DEV cohort from silently consuming J1b’s eight rows.

**APPROVE — the implementation preserves J1b row and ε₀ identity, fails closed on incompatible artifacts, and is sound for TPU launch.**
