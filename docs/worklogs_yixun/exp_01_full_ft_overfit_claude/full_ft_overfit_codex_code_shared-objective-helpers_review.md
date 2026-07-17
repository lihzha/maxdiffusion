# Code review: exp_01 full_ft_overfit — round shared-objective-helpers
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-16

## Context loaded
- Read `experiment_SOP.md`, the driving query, approved plan, and full worklog with `sed`/`cat`.
- Skimmed `docs/side_adaptor.md` for objective provenance and fixed-versus-fresh-noise history with `rg`/`sed`.
- Inspected `git status --short`, `git diff`, `git diff --cached`, and `git diff --check`.
- Read the complete new test file, both touched source regions, and the pre-refactor `_denoising_loss` from `HEAD` with numbered lines.
- Searched the existing test suite for other coverage of `_denoising_loss`; none exists.
- Ran the focused tests without capture/cache because the exact command could not create a temporary file in this read-only sandbox; all 7 tests passed.
- Probed the changed mask-shape behavior with broadcastable unequal shapes; the new helper returned `0.0` where the old equation returned `1.0`.

## Verdict
REQUEST-REVISION. The production-path equations are identical for the expected equal-shaped float32 tensors, and the auxiliary dictionary remains correctly wired. However, the approved trainer-level characterization is missing, and the changed mask-shape source can silently hide a malformed model output.

## Findings
1. **F1 — MAJOR:** `test_full_ft_overfit_shared_objective.py` never imports or executes the refactored trainer’s `_denoising_loss`; it only tests the two helpers independently. The suite would remain green if the trainer passed the wrong target or batch size, stopped using either helper, or changed the production guide-scale path, so it does not satisfy plan §3’s fixed-RNG equality of the *refactored side-adapter loss* against the pre-refactor formula. Add a tiny stub-based `_denoising_loss` characterization using deterministic noise, timestep selection, transformer outputs, and state merges; compare loss and relevant auxiliary values against the transcribed old equations, including `velocity_mse == loss` and pinned-`z_t` diagnostics. Exercise the production `guide_scale=5.0` path at minimum.

2. **F2 — MINOR:** `masked_velocity_mse` builds its mask from `v_pred.shape[1:]`, while the old code built it from `z_video`—equivalently, the new helper’s `v_target`. Equal production shapes preserve results, but a broadcastable malformed prediction can now produce a silently wrong normalization or zero loss; a one-frame prediction against a four-frame target returns `0.0` instead of the old equation’s `1.0`. Build the mask from `v_target.shape[1:]` to preserve the old source, and preferably reject `v_pred.shape != v_target.shape` explicitly; add a regression test for a broadcastable mismatch.

3. **F3 — MINOR:** The tests do not establish the newly claimed defensive dtype and explicit-batch contracts. `test_helpers_return_float32` supplies only float32 inputs, and every loss test passes `batch_size == v_pred.shape[0]`; implementations that omit the casts, use a low-precision mask, or ignore the argument in favor of the tensor batch would pass. Add bf16 inputs with a float32 reference and a normalization case whose explicit `batch_size` deliberately differs from the tensor’s leading dimension.

## Notes for round 2
Do not build the full-FT loss on these helpers until F1–F2 are strengthened. Round 2 should pass the actual sliced batch size, retain float32 interpolation/target math through pinning, and keep its planned stub integration test strict about one plain-transformer call, null context, and absence of actions.

---

# Strengthening record (Coder, same round — 2026-07-17)

- **F1 (MAJOR) — FIXED.** Stub-based characterization of the refactored trainer `_denoising_loss` itself added (real trainer module imported; stub adapter/transformer through the real `TrainState` fields; fixed rng; data-batch 3 vs `global_batch_size_to_train_on=2` so slicing is pinned; real `FlaxFlowMatchScheduler`). `_reference_denoising_loss` transcribes the complete pre-refactor body verbatim. Parametrized over guide_scale **[1.0, 5.0]** (production CFG stop-grad branch exercised; value-identical, stop_gradient is gradient-only); asserts bitwise equality of loss + all 8 aux entries, `velocity_mse == loss`, `z_init_anchor_mse == 0.0`. Helper-call spies pin that each shared helper is invoked exactly once. **Mutant-validated:** corrupted target formula and de-pinned z_t both fail the suite (trainer restored, sha256-verified).
- **F2 (MINOR) — FIXED.** `masked_velocity_mse` now raises `ValueError` on `v_pred.shape != v_target.shape` (trace-time, jit-safe) and builds the mask from `v_target.shape[1:]` (the pre-refactor source). Regression test written RED-first on the review's exact one-frame-vs-four case (`DID NOT RAISE` → raise). Equal-shape results bit-identical (round-1 atol=0 characterization still green).
- **F3 (MINOR) — FIXED.** (a) bf16-input test: helpers must bitwise-reproduce the f32-upcast reference and return f32 (dropped internal casts change rounding → fail). (b) `batch_size=4` vs leading-dim-2 test: loss exactly `sum_sq/(count·4)` — the explicit argument governs `n_valid`.
- **Deviation (documented):** `import tensorflow, grain.python` segfaults (exit 139) on macOS arm64 (repro in test-file header; grain 0.2.16–18, both orders). Darwin-only, `sys.modules`-guarded stub of `grain`/`grain.python` (two class bases) installed before trainer import; linux/TPU loads real grain. Reviewer may prefer a conftest hoist — flagged for round 2.
- **Behavior beyond findings:** none (F2's mismatch-raise is the finding's own scope) → no follow-up review required.

**Cycle 1 closed:** write → review (REQUEST-REVISION, 3 findings) → strengthen (3 FIXED) → commit. Final: 14/14 tests green in 9.7s; +461/−9 (source +46/−9, tests +411; >200 LOC solely from review-mandated tests).
