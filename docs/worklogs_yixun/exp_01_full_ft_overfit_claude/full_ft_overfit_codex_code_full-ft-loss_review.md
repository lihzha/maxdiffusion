# Code review: exp_01 full_ft_overfit — round full-ft-loss
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-17

## Context loaded

- Read `experiment_SOP.md`, the driving query, approved plan, full worklog through round 2, and the strengthened round-1 review.
- Inspected `git status --short`, the complete uncommitted diff, all three new files, and the edited round-1 test region.
- Compared the new `_denoising_loss`, state, and step functions line-by-line with the post-round-1 side-adapter trainer.
- Inspected the shared objective helpers, WAN transformer dtype wiring, pipeline construction, and reference trainer startup path.
- Ran the worklog suite without capture/cache after the exact command hit the read-only temporary-directory restriction; all 23 tests passed, including all 14 round-1 tests.
- Ran an in-memory `n_hist=0` mutant against the nine new tests; all nine still passed.

## Adjudications

- (a) **ACCEPTED** — `activations_dtype` is the appropriate null-context compute boundary and preserves the planned float32-parameter/bfloat16-activation control, although its distinguishing case needs F2’s test.
- (b) **ACCEPTED** — the production call is deliberately keyword-based, and any positional call would fail the stub rather than evade its exact-key/no-actions assertion.
- (c) **ACCEPTED** — the future-frame comparison demonstrably differs from bfloat16 arithmetic, making the float32 interpolation assertion non-vacuous.
- (d) **ACCEPTED** — the conftest change is an exact hoist of Darwin setup; the round-1 assertions are unchanged and all 14 pass.
- (e) **ACCEPTED** — the inline sigma-grid transcription is independent and exactly matches the helper; the separate per-token timestep coverage gap is F1.

## Verdict

REQUEST-REVISION. The production implementation has equation-level parity with the reference and correctly removes only adapters, actions, and CFG. The integration suite nevertheless admits a corrupt `n_hist` implementation, so its advertised strictness must be strengthened before closing the round.

## Findings

1. **F1 — MAJOR:** `test_sigma_and_timestep_match_build_rollout_sigmas_indexing` validates only `sigma_mean` and the mean of per-example `step_t`; it never compares the transformer’s recorded `timestep` with `ref.timestep_2d`. An in-memory mutant forcing `_build_per_token_timestep(..., n_hist=0)` passed all nine new tests, meaning frame-0 tokens could receive the noisy timestep without detection. Assert bitwise equality of `_STUB_CALLS[0]["timestep"]` and `ref.timestep_2d`, plus explicit zero history tokens and per-example `step_t` on future tokens.

2. **F2 — MINOR:** The intentional null-context dtype deviation is tested only where `weights_dtype == activations_dtype`—float32 by default and bfloat16 in the precision test—so reverting the production cast to `weights_dtype` would remain green. Run the null-context test with `weights_dtype="float32", activations_dtype="bfloat16"` and assert both the recorded dtype and exact broadcast value.

## Notes for round 3

After F1–F2 are strengthened, construct `FullFTTrainState` from the transformer split and bind these module-level train/eval steps directly. Preserve the activation-dtype null context, retain computed FSDP shardings for params and optimizer state, and add only the planned trainer assertions, logging, sharding audit, and dispatch.

---

# Strengthening record (Coder, same round — 2026-07-17)

- **F1 (MAJOR) — FIXED.** Sigma/timestep test now asserts bitwise equality of the stub-recorded `timestep` kwarg vs transcribed `ref.timestep_2d`, PLUS helper-independent structural assertions (shape `(b, seq)`; frame-0 history block exactly zero; every future position = its example's own `step_t`; non-vacuity guard `step_t[0] != step_t[1]`). **Mutant-validated:** the reviewer's exact `n_hist=0` mutant — previously green on all nine tests — now fails with `Mismatched elements: 12/48 (25%)` (= the b=2 × 6-token history block). Trainer restored, sha256-verified (`2dd193dd…6042e` before == after).
- **F2 (MINOR) — FIXED.** New `test_null_context_cast_follows_activations_dtype_not_weights_dtype` with `weights_dtype=float32, activations_dtype=bfloat16`: state stores f32; recorded `encoder_hidden_states` is bf16 and bitwise equals `broadcast(null_context.astype(bf16))`. Revert-to-`weights_dtype` mutant fails on the dtype assert; trainer restored, sha256-verified.
- **Behavior beyond findings:** none — zero production changes this phase (trainer byte-identical to the reviewed version); +37 test lines total. No follow-up review required.

**Cycle 2 closed:** write → review (REQUEST-REVISION, F1–F2; adjudications a–e all ACCEPTED) → strengthen (both FIXED, mutant-validated) → commit. Final: 24/24 tests green in 11.3s.
