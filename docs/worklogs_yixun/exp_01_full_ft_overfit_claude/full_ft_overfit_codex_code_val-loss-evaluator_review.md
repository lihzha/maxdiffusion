# Code review: exp_01 full_ft_overfit — round val-loss-evaluator
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-26

## Context loaded
- `experiment_SOP.md` — establishes the closed-cycle TDD, parity, validation, provenance, and launch requirements.
- `full_ft_overfit_yixun_query.md` Query 8 — fixes full 14,636-window coverage, position-keyed RNG, objective parity, artifacts, and reporting.
- `plan_full_ft_overfit.md` Part II v2 — defines D1–D5, II.3, smoke isolation, sequential restore, and the nine-column schema.
- `full_ft_overfit_codex_plan2_review.md` — establishes the F1/F2/F4/F5/F6 implementation contracts.
- `full_ft_overfit_codex_code_val-loss-core_review.md` — supplies the five cycle-B notes checked below.
- `full_ft_overfit_worklog.md` — records cycle-A closure and rung-3 PASS with exactly 14,636 contiguous, unique records.
- Current uncommitted diff — contains the evaluator, restore kwarg, three YAML keys, wrapper, and 791-line evaluator test file.
- `wan_ti2v_full_ft_trainer.py` — provides the line-by-line training-objective reference.
- Verification — 151 tests collected; 129 passed with no assertion failures, while 22 `tmp_path` setups were blocked solely by the read-only sandbox, corresponding mechanically to the expected additional 20 passes plus 2 matplotlib skips; YAML parse, `bash -n`, and `git diff --check` passed.

## Cycle-A notes check
- Position independent of stored `ordinal`: **HONORED** — records are enumerated by reader position and the unrelated-ordinal fixture proves RNG lookup uses position.
- EOF drain and count assertion before build/restore: **HONORED** — `load_all_records` drains completely and `evaluate` invokes it before `_build_and_free_state`.
- Identical RNG across reorder, rebatching, and restores: **HONORED** — arbitrary-order, rebatching, and two-checkpoint tests establish position-keyed `(t, eps)` identity.
- Ordered losses with matching validity masks: **HONORED** — per-batch losses and masks remain paired, and padded rows affect neither mean nor stderr.
- Exact count plus float64 sample-stderr aggregation: **HONORED** — the full path requires the configured 14,636 count and uses float64 reductions with `ddof=1`.

## Adjudications
- (a) **ACCEPTED** — `deterministic=True` without a dropout RNG is semantically correct evaluation and numerically equivalent here because dropout is exactly `0.0`.
- (b) **ACCEPTED** — freeing rollout modules after the shared builder preserves T2’s VAE requirement and removes them before T1 evaluation.
- (c) **ACCEPTED** — removing `__all__` changes only wildcard-export metadata, not any function contract.
- (d) **ACCEPTED** — removing the unused mesh parameter is behavior-neutral cleanup.
- (e) **CHANGE-ORDERED** — process environment is acceptable job-visible provenance, but optional `TRAIN_COMMIT` permits a contract-violating `"unknown"` artifact.
- (f) **ACCEPTED** — guarded worker plotting plus mandatory recorded `plot-only` regeneration matches F6, provided acceptance remains blocked until the production PNG exists.

## Verdict
**REQUEST-REVISION.** The numeric objective, count gate, position-keyed RNG, restore assertions, schema, and smoke isolation are substantively correct. The target multi-host v6e-8 run cannot safely materialize the globally sharded per-example losses without a process gather, so this cycle must be strengthened before commit or launch.

## Findings
1. **F1 — BLOCKER.** `eval_wan_full_ft_val_loss.py:604-608` shards the `[B]` result over the global mesh, but line 392 converts each non-fully-addressable global `jax.Array` directly with `np.asarray`. The experiment’s v6e topology is four chips per host, making v6e-8 a two-host job; direct host conversion cannot recover all 32 losses and normally raises for non-addressable devices. Gather every batch collectively on all processes before host conversion, using `multihost_utils.process_allgather(loss, tiled=True)` followed by `jax.device_get`; add a tested gather-to-host seam and require the v6e-8 smoke to exercise it.

2. **F2 — MAJOR.** `eval_wan_full_ft_val_loss.sh:83` defaults `TRAIN_COMMIT` to empty and `_resolve_commits` converts that to `"unknown"`, despite Query 8 and F6 requiring the training commit SHA in every result row. Make `TRAIN_COMMIT` mandatory in the wrapper and reject empty/`unknown` provenance before a full evaluation writes artifacts; test both rejection and exact propagation.

3. **F3 — MAJOR.** The current objective is correct, and the tests catch a target-sign flip and any nine-column rename, but a sigma-grid argument drift in `evaluate` would survive: tests pass preconstructed sigmas directly and never pin the evaluator’s call to `build_rollout_sigmas(num_steps, flow_shift, scheduler.sigma_min, scheduler.sigma_max)`. Add a wiring test or pure sigma-builder seam that pins all four arguments and `num_train_timesteps`; also assert uniform timestep sampling and add static wrapper tests covering every env-to-config mapping plus `bash -n`.

## Notes for cycle C
Do not open cycle C until F1–F3 are strengthened and cycle B is re-run in a writable environment to obtain 149 passed + 2 matplotlib skips. The pre-launch package must require an explicit training SHA, exercise the multi-host gather in smoke, keep smoke output isolated, and withhold T1 acceptance until `validation_loss/val_loss_plot.png` exists.

---

# Strengthening record (Coder, same cycle — 2026-07-26)

- **F1 (BLOCKER) — FIXED.** `_loss_to_host` seam: fully-addressable → direct; global-mesh sharded → `multihost_utils.process_allgather(tiled=True)` + `jax.device_get`, invoked for EVERY batch (also supplies the block-before-restore fence; explicit block_until_ready removed as redundant). Seam spy (6 calls = 3 batches × 2 ckpts), fake-sharded routing, tiled=True pinned. Mutant α (bypass) red; sha restore verified. Real two-host path exercised by the smoke job (acceptance criterion).
- **F2 (MAJOR) — FIXED (both layers).** Wrapper `${TRAIN_COMMIT:?…}` hard-fails pre-python; module `_require_train_commit` raises in FULL mode before drain/build/write (smoke exempt — isolated `validation_loss_smoke/` output, never a T1 artifact; rationale in docstring). Rejection (both layers, incl. subprocess proof python never ran), smoke exemption, and exact propagation into every JSON row + CSV cells pinned. Mutant β red at both module layers. Incidental find: bash `${:?}` apostrophe-swallows-brace bug — message rephrased, bash -n now a pytest case.
- **F3 (MAJOR) — FIXED.** `_build_sigma_grid(config, scheduler)` = the only sigma source in evaluate(); seam + evaluate-level behavioral wiring tests pin all four args (distinct non-default values 13/3.0/0.25/0.75) + num_steps pass-through + grid-reaches-loop; 28 parametrized wrapper env→override needles + fake-python argv/env capture + bash -n. Mutant γ (flow_shift hardcode) red at both levels.
- Suite 149→192 passed (+43) + 2 matplotlib skips; orchestrator independently reproduced 192+2 in the writable env (reviewer's re-run requirement).

# Follow-up review: exp_01 full_ft_overfit — val-loss-evaluator (strengthened)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-26

## Focus-finding check

F1: CLOSED — The production global-mesh loss has process-consistent addressability, and `_loss_to_host` collectively gathers and host-materializes every batch before any subsequent checkpoint restore.

F2: CLOSED — Both launcher and module reject missing full-run provenance before dataset/model work, while positive smoke runs write only to the isolated `_smoke` output path and cannot become T1 artifacts.

F3: CLOSED — The behavioral `evaluate()` wiring test pins sampling steps, flow shift, scheduler sigma minimum, and scheduler sigma maximum at the production call site and verifies the resulting grid reaches the loop.

## Adjacencies

block_until_ready removal: findings-scoped — per-batch host materialization now provides the required synchronization fence.

evaluate-head reorder: findings-scoped — it places the cheap F2 guard before dataset draining and state construction.

env-hermeticity fixture: findings-scoped — test-only isolation with no production risk.

## New issues

None

## Verdict

APPROVE — All prior findings are closed with no new launch-blocking risk; this is the Codex half of the T1 launch sign-off. Static, shell-syntax, and diff checks passed; pytest rerunning was prevented solely by the read-only reviewer sandbox lacking a writable temporary directory.

**Cycle B closed:** write → review (REQUEST-REVISION: F1 BLOCKER multihost gather, F2, F3; cycle-A notes 5/5 HONORED; adjudications 5 ACCEPTED + 1 → F2) → strengthen (3 FIXED, 3 mutants) → follow-up review **APPROVE = Codex half of the T1 launch sign-off**.
