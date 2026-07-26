# Code review: exp_01 full_ft_overfit — round val-loss-core

Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-26

## Context loaded

- `experiment_SOP.md` — TDD, focused review, validation, and experiment-record requirements.
- `full_ft_overfit_yixun_query.md` — Query 8’s held-out evaluation and launch contracts.
- `plan_full_ft_overfit.md` — Part II v2 D1–D3, cycle-A scope, and F1/F2/F3 resolutions.
- `full_ft_overfit_codex_plan2_review.md` — prior findings and the original XLA reduction-order caution.
- `full_ft_overfit_worklog.md` — Query 8 and Part-II plan-cycle entries.
- Cycle-A diff — three intended files only; evaluator main/config/wrapper/gallery are correctly absent.
- Verification — new core tests 19/19 passed; expected 119 collected, with 103 passing and 16 pre-existing `tmp_path` tests blocked at setup by the read-only sandbox; no assertion failures.
- Scalar-helper audit — current and `HEAD` function hashes match exactly; diff is additive and `git diff --check` is clean.

## Adjudications

(a) ACCEPTED — tight closeness is the meaningful F3 contract because XLA does not guarantee bitwise equality; loop delegation would make characterization tautological and discard the independent vectorized mask implementation.

(b) ACCEPTED — changing the mask’s shape source is an equivalent mutant after the shape guard; mutant (i′), the frame-0 invariance test, and the mismatch test exercise the substantive behavior.

(c) ACCEPTED — keep both forms; list-of-batches is cycle B’s canonical path, while a concatenated array is an unambiguous, harmless convenience normalized to the same ordered stream.

(d) ACCEPTED — Python lists fit host-side record indexing and batch planning; cycle B can convert validity to an array only at the device boundary.

(e) ACCEPTED — `n < 2` is unreachable for the required positive count of 14,636, and the explicit NaN guard is conventional defensive behavior rather than a load-bearing contract.

## Verdict

APPROVE-WITH-CHANGES. The implementation satisfies the substantive F1–F3 contracts, uses sound float64 host reductions for mean and sample stderr, and the tests catch fold-in removal, validity-mask removal, and `ddof=0`. One minor RNG test strengthening should land before commit.

## Findings

1. **F1 — MINOR.** The production-25 test checks `0 <= t < 25`, but the non-25 call at `test_full_ft_overfit_val_loss_core.py:152` checks only shape and dtype; a mutant hard-coding `maxval=25` therefore survives despite violating the generic `num_steps` contract. Add a deterministic non-25 bounds assertion or monkeypatch `jax.random.randint` to pin `shape=()`, `minval=0`, `maxval=num_steps`, and `dtype=jnp.int32`.

## Notes for cycle B

- Enumerate dataset position independently of stored `ordinal`; use position for `per_example_rng`.
- Drain the reader to EOF and assert the source count before restoring or evaluating any checkpoint.
- Preserve identical `(t, ε)` across reorder, rebatching, and checkpoint restores; the integration test must use deliberately unrelated stored ordinals.
- Feed `aggregate` ordered per-batch `[B]` loss arrays with matching validity masks; invalid padded rows must affect neither mean nor stderr.
- Require `n == 14,636` and report the float64 host aggregate with sample stderr using `ddof=1`.

---

# Strengthening record (Coder, same cycle — 2026-07-26)

- **F1 (MINOR) — FIXED (deterministic option):** new `test_rng_generic_num_steps_bounds_and_coverage` — `num_steps=3`, seed 0, positions 0–63: every `t ∈ [0,3)` AND `set(draws) == {0,1,2}` (full-support coverage). Kills the hard-coded `maxval=25` mutant (~88% of draws out of range → red) AND the off-by-one `maxval=num_steps-1` mutant (`{0,1} != {0,1,2}` → red); sha-verified restores. Implementation file unchanged (sha-identical) — test-side fix only, per the finding. Deterministic option chosen over the monkeypatch because it pins the behavioral contract (bounds + support), not the call signature.
- **Adjudications a–e:** all ACCEPTED as flagged (closeness contract, mutant (i′) teeth, dual aggregate input, list returns, NaN guard untested).

**Cycle A closed:** write (19 tests, 3+1 mutants) → review (APPROVE-WITH-CHANGES, 1 MINOR) → strengthen (FIXED, 2 more mutants) → commit. Final: 120/120 green.
