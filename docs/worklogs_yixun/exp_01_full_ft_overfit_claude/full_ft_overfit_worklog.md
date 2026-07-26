# exp_01 `full_ft_overfit` — Worklog

Append-only lab notebook (one entry per action) for the plain-Wan-TI2V full-finetune overfit diagnostic. Newest entries at the bottom. Entry template in `experiment_SOP.md`.

## 2026-07-16T02:15:00Z — Scaffold exp_01: reserve number, create branch + worktree

- **Goal** — Reserve `exp_01` and stand up the experiment bookkeeping per the SOP, so planning can begin from a committed scaffold.
- **Change** — Created `docs/worklogs_yixun/exp_01_full_ft_overfit_claude/` with `full_ft_overfit_yixun_query.md` (driving intent from Lihan, relayed by Yixun) and this worklog.
- **Version Control** — Integration branch `yixun-dev` @ `8258965` (`base_commit`). Experiment branch **`claude-exp_01_full_ft_overfit-20260715`** created off `yixun-dev`. Worktree absolute path (SOP isolation rule 1): **`/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit`**. All experiment commits land on the experiment branch; the `exp_01_full_ft_overfit_claude/` docs auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Command / Validation** — `git worktree add /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit -b claude-exp_01_full_ft_overfit-20260715 yixun-dev`. Verified before reserving: no `exp_*` folder on `yixun-dev`, no `exp` branch, single worktree — `exp_01` never used (SOP: never reuse a number).
- **Result** — `scaffolded` — folder + query + worklog created; number reserved.
- **Analysis** — n/a (bookkeeping; no experiment signal yet).
- **Next** — Planner writes `plan_full_ft_overfit.md` resolving the four open design questions in the query doc (trainer path, conditioning, overfit subset/steps/LR, success metric). Then Codex `gpt-5.6-sol` xhigh reviews it → `full_ft_overfit_codex_plan_review.md` → Planner resolves findings → user approval. **NOTE:** per the SOP the Planner is Fable 5 (max); this session is currently Opus 4.8 — switch to Fable 5 for the plan, or confirm proceeding on Opus.

## 2026-07-16T02:50:00Z — Design locked (Query 2) + plan drafted

- **Goal** — Turn Lihan's intent + Yixun's design decisions into the reviewable plan.
- **Change** — Appended Query 2 (verbatim decisions: Planner=Fable 5; no adapter, full backbone unfreeze; conditioning = first-frame + video only, no actions; full DROID) to `_yixun_query.md`. Wrote `plan_full_ft_overfit.md` (draft for review).
- **Version Control** — exp branch `claude-exp_01_full_ft_overfit-20260715`; worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit`; base `2dd8cab`.
- **Command / Validation** — Code reading (Planner, no code written): side-adapter trainer, config yml, launch scripts, generate script, `max_utils.create_optimizer`, `disable_training_weights` grep. Key findings baked into the plan: (1) loss already pins frame 0 + uses null-text — probe = drop adapter/action path, plain transformer forward; (2) **CFG trap**: with no adapter, guide_scale=5.0 degenerates to a silent ×5 gradient scale → probe must assert guide_scale=1.0 (plan §2.1); (3) side-adapter `_shard_state` replicates params/opt_state (fine for 128M adapters, fatal for 5B) → full-FT trainer keeps FSDP shardings; (4) `disable_training_weights` only read by the BaseWanTrainer family — inert for self-contained trainers.
- **Result** — `in_progress` — plan drafted, Codex review being invoked next.
- **Analysis** — Subclassing the side-adapter trainer maximizes parity (same iterator/sigmas/noise/mask by imported code, not re-implementation), which is the whole point of a diagnostic.
- **Next** — Briefed Codex `gpt-5.6-sol` xhigh plan review → `full_ft_overfit_codex_plan_review.md` → resolve findings → re-review if material → surface to Yixun for approval.

## 2026-07-16T03:35:00Z — Codex plan review round 1: REQUEST-REVISION → plan v2

- **Goal** — Close the plan-review cycle: verdict, resolutions, revision.
- **Change** — Saved `full_ft_overfit_codex_plan_review.md` (Codex gpt-5.6-sol xhigh; REQUEST-REVISION; 1 BLOCKER / 7 MAJOR / 2 MINOR). Accepted all 10 findings; rewrote `plan_full_ft_overfit.md` to v2; appended per-finding resolutions to the review file.
- **Version Control** — exp branch `claude-exp_01_full_ft_overfit-20260715`; prior tip `ce5fc4f`.
- **Command / Validation** — Reviewer's factual claims spot-verified against `docs/side_adaptor.md`: train set = 1,440,554 windows (10k@512 = 3.55 passes, NOT "many epochs"); fresh side-adapter val SSIM 0.615–0.664 (my 0.29 "threshold" was the pre-context number and cross-split — both my errors, both real.
- **Result** — `fix_ready` — v2 keys: fresh-noise enforced at wrapper+trainer (BLOCKER F1); CFG wording corrected (5× pre-optimizer gradient, not 5× LR); shared objective helpers so both trainers run the same code; cohort-based memorization metric vs pretrained step-0 baseline; §2.4 escalation (30k → LR control → fp32-opt-state control) gates any negative verdict; 5 focused Coder rounds.
- **Analysis** — Review caught one launch-path landmine (wrapper fixed-noise default) that the config-level parity audit would likely have missed until a broken run: cross-model review paying for itself before any code exists.
- **Next** — Re-review of plan v2 (material revision), then surface plan + review + resolutions to Yixun for approval.

## 2026-07-16T04:05:00Z — Conditional user approval recorded

- **Goal** — Record the approval gate state before the re-review verdict lands.
- **Change** — Query 3 appended to `_yixun_query.md`: plan v2 pre-approved conditional on the Codex re-review verdict being APPROVE; Coder round 1 (shared-objective-helpers) to start immediately on that verdict. Any other verdict returns to Yixun.
- **Result** — `in_progress` — re-review (Codex gpt-5.6-sol xhigh) still running.
- **Next** — On APPROVE: append re-review to the review file, commit, launch Coder round 1 (Opus 4.8 max subagent, test-first, in the exp worktree, no commits — commit happens at cycle close after review+strengthen).

## 2026-07-17T02:45:00Z — Re-review relaunched (infra: codex stdin hang)

- **Goal** — Unblock the plan-v2 re-review gate.
- **Result** — `fix_ready→launched` — first re-review invocation never started: `codex exec` in a background shell blocked on open stdin (`Reading additional input from stdin...`, ~0 CPU, no version header; stuck ~10 h). Killed PID 4692; relaunched identically **with `< /dev/null`**; header printed within seconds this time.
- **Analysis** — **Infrastructure**, not a bug: round-1 review had worked because its stdin happened to close, masking the hazard. Standing workaround recorded in `issue_report.md` (#5): every non-interactive `codex exec` gets `< /dev/null`.
- **Next** — On APPROVE verdict: append to review file, commit, launch Coder round 1 per Query 3's conditional approval (still armed).

## 2026-07-17T03:05:00Z — Re-review: REQUEST-REVISION (converging) → plan v3

- **Goal** — Close re-review round: 6/10 RESOLVED, 4 partial (F4/F5/F6/F9), 1 new MAJOR (G1: keep_period 5000 would evict ckpt 2500 needed by the cohort protocol).
- **Change** — Plan → v3: §2.4 controls fully specified (resume-vs-fresh semantics, durations, eval checkpoints; fp32 mechanism = `weights_dtype: float32` override); §2.3 execution path (`validation_ordinals` key + `checkpoint_step: 0` restore-bypass); §3/§6 large-leaf sharding audit + `checkpoint_keep_period: 2500`. Resolutions appended to review file.
- **Result** — `fix_ready` — re-review #2 (focused on the 5 items) launched with the `< /dev/null` fix.
- **Next** — On APPROVE: Coder round 1 auto-starts per Query 3 conditional (extended by Yixun's momentum intent to this verdict); otherwise back to Yixun.

## 2026-07-17T03:40:00Z — Re-review #2: APPROVE-WITH-CHANGES → plan APPROVED, Coder round 1 starts

- **Goal** — Close the plan gate.
- **Result** — `passed` — F4/F5/F6/F9/G1 all CLOSED (reviewer verified keep-period OR-policy in Orbax v0.11.33 source and mu/nu dtype behavior in Optax v0.2.8 source). One MINOR H1 (aggregate retention ≈760 GB) resolved: budgeted + post-eval pruning rule in R4. Per Query-3 conditional approval, plan v3 is **APPROVED**.
- **Next** — Coder round 1 `shared-objective-helpers` (Opus 4.8 max, test-first, in this worktree, no commits): extract `build_noisy_pinned_latents` + `masked_velocity_mse` into `side_adapter_wan.py`, refactor reference `_denoising_loss` to call them, characterization tests in `src/maxdiffusion/tests/worklogs_yixun/test_full_ft_overfit_shared_objective.py`.

## 2026-07-17T04:15:00Z — TPU launch gate added (announcement 02)

- **Goal** — Record Yixun's standing directive: explicit approval required before ANY TPU run.
- **Change** — `announcement/02_tpu_run_requires_approval.md` + SOP Running-discipline bullet. For this experiment: validation-ladder rungs 5 (v6e-8 smoke), 6 (v6e-64 fit probe), 7 (full run), all escalation controls, and all cohort-validation jobs each require Yixun's go-ahead, requested with the pre-launch package. Rungs 1–4 (local CPU) unaffected; Coder rounds proceed.
- **Result** — `passed` (bookkeeping).
- **Next** — Coder round 1 in progress; first approval ask will be the v6e-8 smoke run after rounds 1–5 + local ladder rungs pass.

## 2026-07-17T04:45:00Z — Coder round 1 (shared-objective-helpers): write phase complete

- **Goal** — Round 1 of 5: extract the shared objective math (parity-by-shared-code, plan §3).
- **Change** — Coder (Opus 4.8 max): `build_noisy_pinned_latents` + `masked_velocity_mse` added to `models/wan/side_adapter_wan.py` (+42); side-adapter `_denoising_loss` refactored to call them (+4/−9, behavior-preserving); new `tests/worklogs_yixun/test_full_ft_overfit_shared_objective.py` (+143, 7 tests). Total +189/−9 (<200 LOC). Uncommitted by design (commit at cycle close).
- **Command / Validation** — TDD: RED = ImportError on the two new names (right reason) → GREEN = 7/7 passed in 2.10s (CPU venv, jax 0.11.0). Characterization tests assert atol=0 equality against verbatim-transcribed pre-refactor equations. `py_compile` OK ×3; new lines black-119/ruff clean.
- **Result** — `passed` (write phase). Notable deltas for review: trainer's `apply_first_frame_pin` import removed (moved into helper; ruff F401), mask shape now derived from `v_pred.shape[1:]` (≡ `z_video` shape), defensive f32 casts in helpers (no-op on f32 inputs, proven bit-identical).
- **Next** — Review phase: briefed Codex `gpt-5.6-sol` xhigh on this diff → `full_ft_overfit_codex_code_shared-objective-helpers_review.md` → strengthen → commit closes the cycle.

## 2026-07-17T05:30:00Z — Coder round 1: strengthen complete → cycle 1 CLOSED

- **Goal** — Resolve review findings F1/F2/F3 and close the round.
- **Change** — Trainer-level fixed-RNG characterization (guide 1.0 + 5.0, all aux, helper-call spies, mutant-validated); `masked_velocity_mse` mask source → `v_target` + shape-mismatch `ValueError` (RED-first regression); bf16 + explicit-batch-size contract tests. Source +46/−9, tests +411; 14/14 green. Darwin grain-segfault stub documented in test header (linux loads real grain).
- **Version Control** — cycle-close commit on `claude-exp_01_full_ft_overfit-20260715` (this commit); >200 LOC justified: review-mandated tests only.
- **Result** — `passed` — strengthening record appended to `full_ft_overfit_codex_code_shared-objective-helpers_review.md`; no behavior change beyond findings → no follow-up review.
- **Next** — Coder round 2 `full-ft-loss`: `FullFTTrainState` + full-FT `_denoising_loss`/`_train_step`/`_eval_step` + fixed-RNG integration test, honoring the reviewer's Notes for round 2.

## 2026-07-17T16:20:00Z — Coder round 2 (full-ft-loss): write phase complete

- **Goal** — Round 2 of 5: `FullFTTrainState` + full-FT `_denoising_loss` + `_train_step`/`_eval_step` on the shared helpers.
- **Change** — New `trainers/wan_ti2v_full_ft_trainer.py` (+176; state/loss/steps only, trainer class is round 3; no CFG branch per plan §2.1; `_sample_step_indices`/`_build_noise` imported, zero objective-math re-implementation). New `test_full_ft_overfit_denoising_loss.py` (+325, 9 strict integration tests: one-transformer-call/no-actions, null-context identity, fresh per-example noise, σ/t indexing vs transcribed grid, target=eps−z_video, frame-0 pin, sliced batch size, train-step updates params, f32-under-bf16). Grain stub hoisted to shared `conftest.py` (+29; round-1 file −19/+3, its 14 tests re-verified green post-hoist).
- **Command / Validation** — RED: ModuleNotFoundError on the new trainer module (right reason) → GREEN: 23/23 in 10.4s; py_compile OK; black-119 + ruff clean. Coder-flagged deviation for review: null-context cast uses `activations_dtype` (reference text: `weights_dtype`; value-identical in all shipped configs).
- **Result** — `passed` (write phase). Interruption note: Coder agent died once mid-wrap-up on an API connection drop (infrastructure) and was resumed; work verified complete.
- **Next** — Briefed Codex review (marker `full-ft-loss`) → strengthen → cycle-2 commit.

## 2026-07-17T16:25:00Z — Infra: false yixun-dev divergence diagnosed; origin switched to HTTPS

- **Goal** — Explain Yixun's `git status` showing yixun-dev "diverged 15/9" from origin.
- **Result** — `passed` — NO real divergence: fresh fetch proved remote == local (`1edc16f`) exactly. Cause: the manual `refs/remotes/origin/yixun-dev` update (part of the old SSH-broken push workaround) once ran from the exp-worktree cwd and recorded the exp-branch tip `bf36900` as the tracking ref. Fix: `origin` set to HTTPS; full `git fetch --prune` rebuilt all tracking refs from truth (`+ bf36900...1edc16f (forced update)` confirmed); plain `git push origin` now standard. `issue_report.md` #1 updated (workaround retired).
- **Analysis** — Infrastructure/bookkeeping, no repo damage; classic hazard of hand-maintained refs — eliminated by removing the need for them.

## 2026-07-17T17:40:00Z — Coder round 2: strengthen complete → cycle 2 CLOSED

- **Goal** — Resolve round-2 findings F1/F2 and close the cycle.
- **Change** — Test-only: timestep_2d bitwise + structural (helper-independent) assertions killing the reviewer's `n_hist=0` mutant; split-dtype null-context test (f32 weights / bf16 activations) killing the revert mutant. Production trainer sha256-identical to the reviewed artifact. 24/24 green; style clean.
- **Version Control** — cycle-close commit on `claude-exp_01_full_ft_overfit-20260715` (this commit).
- **Result** — `passed` — strengthening record appended to `full_ft_overfit_codex_code_full-ft-loss_review.md`.
- **Next** — Round 3 `trainer-wiring`: `WanTI2VFullFTTrainer` class (start_training with guide-scale + fresh-noise asserts, dtype/dir logging), `_shard_state` override + large-leaf audit, `FULL_FT_TI2V` dispatch, wiring tests — honoring the reviewer's Notes for round 3.

## 2026-07-17T18:55:00Z — Coder round 3 (trainer-wiring): write phase complete

- **Goal** — Round 3 of 5: the `WanTI2VFullFTTrainer` class + guards + sharding + dispatch.
- **Change** — `wan_ti2v_full_ft_trainer.py` +384: `_validate_probe_config` (guide-scale §2.1 + fresh-noise F1 asserts, run before any load), `start_training` mirroring the parent shell (121/146 lines byte-identical; 9 enumerated deviations, each constraint-commented), `_shard_state` override keeping computed FSDP shardings (+ retained `_apply_actual_sharding_for_tpu`), `audit_large_leaves` (top-8 + full-tree >100MB-replicated raise), startup dtype logging (params/mu/nu/activations). `train_wan.py` +3 (FULL_FT_TI2V dispatch). New `test_full_ft_overfit_trainer_wiring.py` (213 lines, 14 tests incl. real-optimizer moments-follow-param-dtype).
- **Command / Validation** — RED: AttributeError on all missing names → GREEN: 38/38 in 13.5s; py_compile OK; ruff clean; black deviations limited to one parent-byte-identical 120-char line + pre-existing train_wan style (documented).
- **Result** — `passed` (write phase). Coder-flagged for review: source LOC over target (mandated parity mirror + audit block), by-magnitude opt-twin association in the audit top-k (raise scans ALL leaves), `_full_ft_state_shardings` written as explicit no-op `.replace` for line-parity with the parent.
- **Next** — Briefed Codex review (marker `trainer-wiring`) → strengthen → cycle-3 commit.

## 2026-07-17T20:10:00Z — Coder round 3: strengthen complete → cycle 3 CLOSED

- **Goal** — Resolve round-3 findings F1–F5, close the cycle.
- **Change** — Per-dtype param/mu/nu startup summaries (primary-vs-fp32-control now machine-distinguishable — the §2.4/§6 precondition); per-host PHYSICAL byte accounting (replicas counted); path-matched labeled opt twins; four behavioral wiring tests (validate-before-load, shard-dataflow object identity, audit-raise propagation, executed dispatch); NaN-proof exact guide-scale guard. Strengthen delta ≈ +69 trainer lines + test growth to 26 wiring tests.
- **Command / Validation** — 50/50 green (Coder run + orchestrator's independent rerun, 11.6s); 7/7 revert-mutants caught, sha256-verified restorations; ruff/py_compile clean. Safety-classifier-unavailable note on the subagent run → orchestrator manually verified touched-file scope + suite before commit.
- **Result** — `passed` — strengthening record appended; cycle-3 commit is this commit.
- **Next** — Round 4 `ckpt-generation` per plan §3 + the round-3 review's gate note: behavioral proof of params/opt_state/step restore into `FullFTTrainState`, `checkpoint_step: 0` bypass, `validation_ordinals` reader, plain-transformer rollout branch. No round-5 config/launcher work.

## 2026-07-17T21:25:00Z — Coder round 4 (ckpt-generation): write phase complete

- **Goal** — Round 4 of 5: validation-side plumbing — restore into `FullFTTrainState`, step-0 baseline bypass, `validation_ordinals` cohort reader, plain-transformer rollout branch.
- **Change** — `generate_wan_side_adapter.py` +212/−29 (net +183): `_restore_validation_state` → thin dispatcher over `_build_side_adapter_validation_state` (original body moved verbatim) / `_build_full_ft_validation_state`; shared `_restore_checkpoint_state` with full-FT-only (`cohort_mode`) step-0 bypass + actionable missing-step error; `_iter_parsed_records`/`_select_eval_records` seam honoring `validation_ordinals` (0-based dataset POSITIONS, listed order, duplicates preserved, out-of-range ValueError, empty → contiguous fallback byte-identical); `_rollout_sample` full-FT branch (no adapter fields, plain transformer). 15 new tests across 2 files.
- **Command / Validation** — RED: missing names/branches (14 failed, 1 guard green) → GREEN: 65/65 in ~13s; py_compile/black/ruff clean; 5 revert-mutants (restore→init, step-0-no-bypass, ordinals-sorted, contiguous off-by-one, forward-branch-off) all caught, sha256-verified restores.
- **Result** — `passed` (write phase). Coder-flagged for review: step-0/missing-N gating to full-FT-only (protects adapter semantics); +183 LOC vs plan's ~60 (relocation + docstrings + testability seams); `validation_ordinals` NOT added to the shared `config.json` artifact (would touch adapter output; cohort recorded in `_command.md` per §2.3).
- **Next** — Briefed Codex review (marker `ckpt-generation`) → strengthen → cycle-4 commit.

## 2026-07-17T23:05:00Z — Coder round 4: strengthen complete → cycle 4 CLOSED

- **Goal** — Resolve round-4 findings F1–F4; close the cycle.
- **Change** — Restored step written into `FullFTTrainState` (cohort mode); production dispatcher/builder behavioral tests (cohort_mode=False and builder-swap production mutants killed); honest full-FT artifact provenance via `_validation_config_artifact` (model_type + resolved ordinals; adapter artifact byte-identical); correctly-labeled cohort error message. 8 new tests → 73/73 green (orchestrator-verified).
- **Version Control** — cycle-close commit (this commit). Review attempt 1 died on OpenAI capacity (infra, logged); retry produced the full verdict + the exhaustive round-5 requirements list.
- **Result** — `passed` — strengthening record appended to `full_ft_overfit_codex_code_ckpt-generation_review.md`.
- **Next** — Round 5 `configs-launchers` (final): `base_wan_5b_full_ft.yml`, `train_wan_full_ft.sh`, `launch_wan_train.sh` full_ft arm, full-FT validation wrapper — built exactly to the reviewer's Notes-for-round-5 list (incl. the launcher common-override pitfalls). Then local ladder rungs 1–4 → smoke-run approval ask to Yixun.

## 2026-07-18T00:35:00Z — Coder round 5 (configs-launchers): write phase complete

- **Goal** — Final round: launchable configuration per the round-4 reviewer's requirements list.
- **Change** — `base_wan_5b_full_ft.yml` (288 lines; cp-of-base + 7 deltas + `validation_ordinals` + rationale header; 34 retained keys byte-identical by construction); `train_wan_full_ft.sh` (fresh-noise default, EVAL_DATA_DIR→TRAIN_DATA_DIR, LR/steps/ckpt knobs); `launch_wan_train.sh` full_ft arm (common-defaults block byte-unchanged; post-common override block sets 2500/2500/train-split AFTER commons, BEFORE smoke; TRAIN_SCRIPT dispatch variable); `validate_wan_full_ft.sh` (cohort passthroughs). 21 new tests (executed launcher goldens, not text greps) → 94/94.
- **Command / Validation** — RED 17-failed (4 harness-sanity greens) → GREEN 94/94 in 13.9s; bash -n ×3, yaml parses (float-LR trap avoided: `1.e-5`), black/ruff clean. 4 mutants (noise→fixed, dispatch→side-adapter script, keep_period drop, EVAL_DATA_DIR clobber) caught, restores sha-verified.
- **Result** — `passed` (write phase). Deviations for review: train wrapper omits `validation_ordinals` (training never reads it — generate-script concept); `ACTION_ADAPTER_TYPE="full-ft"` in the arm as inert run-name tag; inert adapter CLI overrides dropped from wrappers (yml keeps keys).
- **Next** — Briefed Codex review (marker `configs-launchers`; includes end-of-implementation check vs plan §6) → strengthen → cycle-5 commit → ladder rungs 1–4 → smoke approval ask.

## 2026-07-18T01:40:00Z — Coder round 5: strengthen complete → cycle 5 CLOSED → IMPLEMENTATION COMPLETE

- **Goal** — Resolve the final finding; close the last cycle.
- **Change** — yml ships the primary recipe standalone (per_device 8.0 → GBS 512 on v6e-64; wandb project = launcher's, single source of truth behaviorally tested); batch-derivation semantics executed against real pyconfig and test-bound; wrapper smoke-contract documented+tested. 95/95 green.
- **Result** — `passed`. Five closed cycles: shared-objective-helpers → full-ft-loss → trainer-wiring → ckpt-generation → configs-launchers. 22 findings total across 5 reviews + 2 plan reviews; every finding fixed or rejected-with-reason on the record; 21+ mutants killed.
- **Next** — Validation ladder rungs 1–4 locally, parity-audit worklog entry, then the v6e-8 smoke-run approval package to Yixun (announcement 02 gate).

## 2026-07-18T02:20:00Z — Validation ladder rungs 1–4 + parity audit: PASS

- **Goal** — Clear the pre-TPU gates (plan §5, SOP ladder + parity audit).
- **Command / Validation** —
  - **Rung 1 (static+unit):** 95/95 pytest; py_compile ×5 touched modules; yaml.safe_load; bash -n ×3; git diff --check. All clean on committed tree `d7bfd49`.
  - **Rung 2 (tiny synthetic forward):** satisfied by the stub-transformer integration tests (fixed-RNG loss-path, train-step-updates-params, rollout tests) per plan §5.2.
  - **Rung 3 (real-data readback):** parsed records 0–1 of `train-00000-of-00704.tfrecord` (3MB partial fetch): z_i0 f16 [48,1,12,20] 23040 B, z_video f16 [48,9,12,20] 207360 B, actions f32 [32,7] 896 B — byte-exact vs schema; finite; stats plausible (latent std ≈0.60/0.91, actions ±0.65). Dataset `summary.json` independently confirms 1,440,554 examples / 704 shards / 2048 per shard (matches review-F4 numbers).
  - **Rung 4:** n/a (no dataset build).
- **Parity audit (plan "Parity audit before scaling"):**
  - **Numeric:** objective math shared BY CODE (round-1 helpers, atol=0 characterization vs pre-refactor equations; trainer-level fixed-RNG characterization incl. guide-5.0 CFG path); sigmas/t-sampling/noise imported not copied; AdamW b1/b2/eps/wd + clipping + warmup retained-key-tested equal to the reference yml; LR 1e-5 is the one deliberate numeric departure (recorded §2.2).
  - **Structural:** trainable set = whole transformer (wiring tests + startup param-count log); no adapter modules constructible in the path (booby-trapped in tests); CFG bypassed with NaN-proof assert; frame-0 pin verified bitwise into the transformer call; fresh-noise enforced at trainer AND both launch surfaces (mutants killed).
  - **Data:** same TFRecord loader code path (inherited `_load_dataset`), rung-3 byte-level readback above.
- **Result** — `passed` — all pre-TPU gates green. Implementation totals: 5 closed cycles, 2 plan reviews + 5 code reviews (Codex gpt-5.6-sol xhigh), 23 findings all resolved on the record, 25+ mutants killed, 95 tests.
- **Next** — Smoke-run approval package to Yixun (announcement 02). No TPU action until approval.

## 2026-07-18T20:39:58Z — SMOKE LAUNCH (rung 5): Yixun approved; acceptance criteria

- **Goal** — First TPU contact for the full-FT probe: prove the 5B-unfrozen step compiles, fits, and logs correctly on v6e-8. Approved by Yixun ("Approve smoke", 2026-07-18; announcement 02 satisfied).
- **Hypothesis** — The full-FT train step fits v6e-8 at per-device batch 8 with remat FULL and FSDP-sharded params/opt-state, and all round-3 startup instrumentation behaves on real hardware.
- **Command / Validation** — `WAN_EXPERIMENT=full_ft SMOKE=1 TPU_CHIPS=8 NAME=wan-full-ft-smoke-yixun bash bash_scripts/launch_wan_train.sh` from the exp worktree (cwd rule); exact entry + job id in `full_ft_overfit_command.md`. Launcher-canonical smoke deltas vs the approval package, accepted: **1 step** (not ~20 — SMOKE hard-sets it; satisfies the ≥1-step criterion) and **per-device 8 ⇒ GBS 64 on 8 chips** (not GBS 8 — the true per-device load; a stricter memory probe).
- **Acceptance criteria** (judge against these, not vibes):
  1. Worker reports COMMIT=07eb5b2; 8 devices.
  2. `_validate_probe_config` passes silently (guide 1.0, fresh noise) — startup log shows `guide scale: 1.0`, `noise mode: fresh`.
  3. `trainable transformer params: ~5.0B` logged; NO adapter-param line.
  4. Per-dtype lines for params/mu/nu present (expect mixed bf16+f32 per the loader's f32 norms).
  5. Large-leaf audit table (8 params + path-matched mu/nu twins) logged; >100MB-replicated assert passes.
  6. Resolved eval_data_dir ends `/train`.
  7. Reaches step 1 with finite loss; no OOM/NaN; steps/s recorded (throughput datum for R5).
  8. NO checkpoint objects written under the run dir.
  W&B optional for smoke (key forwarded only if present in the submitting shell).
- **Result** — `launched` — job `20260718-204019-6aad21e8-wan-full-ft-smoke-yixun` (v6e-8, submitted 20:40:19Z; teed submission log `full_ft_overfit_2026-07-18_20:40:19.log`). W&B key present → wandb active for the smoke.
- **Analysis** — pending run.
- **Next** — On PASS: request v6e-64 fit-probe approval. On failure: infra-vs-bug triage per SOP.

## 2026-07-18T21:10:00Z — Smoke attempt 1 FAILED: v6e-8 HBM OOM by 44MB — triage: config (smoke-topology), not pipeline bug

- **Goal** — Triage the smoke failure (status.json: APPLICATION_ERROR, worker-0 exit 1, attempt 1, non-retryable).
- **Result** — `fix_ready` — worker-0 log (`logs/attempt-1/worker-0.log`, pulled): `CompileTimeHbmOom: Used 31.29G of 31.25G hbm. Exceeded by 44.29M`. Startup up to compile was healthy.
- **Analysis** — **Application/config, NOT infrastructure and NOT a pipeline bug.** v6e-8 carries the full 5B state sharded over 8 chips (≈5 GB/chip params+grads+moments) vs ≈0.6 GB/chip on the target v6e-64; at per-device batch 8 that misses fitting by 0.14%. Plan §5.5 specified the v6e-8 smoke at **GBS 8 (per-device 1)**; the launcher's SMOKE block kept the full-run per-device 8 — that deviation (accepted at launch as "stricter probe") is precisely what OOM'd. Target-topology implication: v6e-64 at per-device 8 has ≈4.4 GB/chip MORE headroom than this failed configuration — the plan's fit expectation stands, to be confirmed by the fit probe (rung 6). Per announcement 02, the retry is a config-changed launch → requires fresh Yixun approval.
- **Next** — Mini-cycle 6: make the launcher's `PER_DEVICE_BATCH_SIZE` env-overridable (1-line; full-run default 8 unchanged; worker-side GBS keys are inert per pyconfig — per-device is authoritative, proven in round 5), quick Codex review, commit. Then request approval for smoke attempt 2 at per-device 1 (plan-spec GBS 8).

## 2026-07-18T22:05:00Z — Mini-cycle 6 (smoke-batch-override): APPROVE, zero findings → cycle CLOSED

- **Goal** — Close the OOM-fix cycle.
- **Change** — Launcher batch trio env-overridable (defaults 8/512/512 unchanged; globals overridable for worker-log echo honesty — inert to training per round-5 pyconfig proof); 2 new executed-golden tests incl. the exact smoke-attempt-2 recipe; default-submission regressions on all three arms. 97/97 green.
- **Result** — `passed` — reviewer confirmed triage, 1/8/8 recipe, and the provisional v6e-64 headroom argument; stray-env sensitivity accepted as documented interface.
- **Next** — Awaiting Yixun's smoke-attempt-2 approval (announcement 02; config-changed relaunch). On approval: launch with `SMOKE=1 PER_DEVICE_BATCH_SIZE=1 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=8 GLOBAL_BATCH_SIZE_TO_LOAD=8 TPU_CHIPS=8`.

## 2026-07-18T21:22:09Z — SMOKE attempt 2 LAUNCH (approved) + acceptance criteria

- **Goal** — Rung-5 smoke at the plan-§5.5 recipe (per-device 1, GBS 8) from post-fix commit `0405a30`.
- **Command / Validation** — `WAN_EXPERIMENT=full_ft SMOKE=1 PER_DEVICE_BATCH_SIZE=1 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=8 GLOBAL_BATCH_SIZE_TO_LOAD=8 TPU_CHIPS=8 NAME=wan-full-ft-smoke2-yixun bash bash_scripts/launch_wan_train.sh` (worktree cwd; matches the cycle-6 test golden). Entry 2 in `_command.md`.
- **Acceptance criteria** — as attempt 1's, re-scoped: COMMIT=0405a30; 8 devices; guide-1.0 + fresh asserts pass; ~5.0B trainable, no adapter line; per-dtype param/mu/nu lines (mixed bf16+f32); large-leaf audit + no-replicated assert pass; eval dir ends /train; **echoed batch 1/8/8**; reaches step 1, finite loss, no OOM/NaN; zero checkpoints. (Per-device-8 memory claim moved to the fit probe.)
- **Result** — `launched` — job `20260718-212209-ff679a2a-wan-full-ft-smoke2-yixun` (v6e-8, 21:22:09Z; teed log `full_ft_overfit_2026-07-18_21:22:09.log`).
- **Next** — On log-verified PASS: auto-launch the v6e-64 fit probe per Query 4's conditional (1 step, per-device 8/GBS 512, storage-light, NAME=wan-full-ft-fitprobe-yixun). On failure: triage; any config change returns for approval.

## 2026-07-18T22:07:20Z — Smoke 2 PASS (log-verified) → fit probe launches per Query-4 conditional

- **Goal** — Verdict smoke attempt 2 against its acceptance criteria; trigger the pre-approved fit probe.
- **Result** — `passed` — job `20260718-212209-ff679a2a` SUCCEEDED on attempt 3 (attempts 1–2: TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE, INFRASTRUCTURE_PREEMPTION, queue-retried — infra, no action). Worker-0 log verification: COMMIT=0405a30 ✓; 8 devices ✓; guide 1.0 + fresh ✓; **trainable 5.00B**, no adapter line ✓; per-dtype lines: params/mu/nu each bf16(604 leaves, 9.82GB)+f32(221, 0.36GB) — the predicted mixed tree ✓; byte totals params 9.48GB / opt 18.96GB global ✓; **audit table: 8 largest params + path-matched mu/nu twins, all FSDP-sharded P(('context','fsdp')), no replicated large leaf** ✓; eval dir …/train ✓; batch echo 1/8/8 ✓; zero checkpoint step-dirs (empty init prefix only) ✓; JOB_EXIT=0, wandb synced (run wewfe1kx) ✓.
- **Analysis** — One criterion partially evidenced: loss VALUE unobserved because LOG_PERIOD=10 > max_steps=1 (the log gate never fires on a 1-step smoke) — display artifact, not a training gap; the step completed through block_until_ready. First loss observation lands at full-run step 10. Not worth a launcher change.
- **Next** — Fit probe (v6e-64, per-device 8 ⇒ GBS 512, 1 step, storage-light) — pre-approved by Query 4, launching now.

## 2026-07-18T22:10:28Z — Query 5 recorded: full run pre-approved conditional on fit-probe PASS; cohort predeclared

- **Goal** — Arm the full-run launch; satisfy §2.3's predeclare-before-launch requirement.
- **Change** — Query 5 in `_yixun_query.md`; `full_ft_overfit_params_set_up.md` written (full config + 16 deterministic cohort ordinals + seed protocol + escalation controls + abort record).
- **Result** — `in_progress` — fit probe running; on log-verified PASS the full run launches with acceptance criteria per plan §6.
- **Next** — Fit-probe verdict → full-run launch (`WAN_EXPERIMENT=full_ft TPU_CHIPS=64`, no SMOKE) → monitor with wandb curve checks at 500-step marks.

## 2026-07-18T22:32:23Z — Fit probe attempt 1 FAILED: INFRA (worker-10 apt-lock hang → JAX init deadline) → auto-resubmit

- **Goal** — Triage fit-probe job `20260718-220720-f19db2ab` (queue label APPLICATION_ERROR, worker-0 exit 134).
- **Result** — `fix_ready` — **infrastructure, not a bug**: worker-10's log is an endless `Waiting for cache lock: /var/lib/dpkg/lock-frontend … held by unattended-upgr` (host-level Ubuntu auto-update); it never completed setup. All other 15 hosts entered train_wan.py, blocked at JAX distributed init, and died on `DEADLINE_EXCEEDED` → SIGABRT (134). Secondary quirk: the queue's setup barrier reported "All **8** workers completed setup" on a **16-host** v6e-64 slice — the barrier under-counted, letting the command start despite the stuck host (queue-infra behavior, noted for the runbook).
- **Analysis** — No code path of ours executed past config echo on the healthy workers; nothing to fix in-repo. Per the standing infra-resubmit policy (announcement 02: same approved job, unchanged config), resubmitting.
- **Next** — Resubmit fit probe unchanged; Query-5 full-run conditional stays armed on the resubmission's log-verified PASS.

## 2026-07-18T22:57:54Z — Fit probe attempt 2 FAILED: same infra class, root cause isolated to setup.sh apt wait

- **Goal** — Triage `20260718-223223-ef0a159d` (worker-5 exit 134 during jax.distributed.initialize).
- **Result** — `fix_ready` — workers 9+11 stuck in `unattended-upgrades` dpkg-lock wait (~700 wait lines each, never reached the command); 14 healthy hosts died at the coordination deadline. **Root cause is in-repo:** root `setup.sh:76-83` uses `apt-get -o DPkg::Lock::Timeout=-1` (wait FOREVER) — on fresh VMs unattended-upgrades holds the lock for minutes; combined with the queue's barrier under-count (8/16, seen both attempts), the job starts and times out. Recurrence: 3 stuck hosts across 2 attempts ⇒ P(clean 16-host slice) ≈ 25–45%.
- **Analysis** — Infra-at-provisioning triggered by an in-repo unbounded wait: classify infra, but with an in-repo hardening available (stop/disable unattended-upgrades on the ephemeral worker before apt; bound the lock timeout so a stuck host fails LOUD in setup instead of silently starving JAX init). Queue barrier under-count reported to the runbook (queue-side, not ours).
- **Next** — (a) Resubmit #3 unchanged now (pre-authorized infra retry — a clean slice passes as-is); (b) mini-cycle 7 in parallel: setup.sh hardening, reviewed; (c) full run should launch from the post-fix commit — Yixun asked to bless that (announcement 02: code-changed launch).

## 2026-07-19T00:20:00Z — Fit probe attempt 3 FAILED (same class, 3/3) → hold all launches for post-fix commit

- **Goal** — Verdict probe #3 (`20260718-225754-87cbd078`).
- **Result** — `fix_ready` — same signature: worker-12 first to abort on DEADLINE_EXCEEDED during jax.distributed init; sampled healthy workers show zero apt-lock lines (stuck host among unsampled ranks). Three consecutive 16-host provisions killed by the same setup-starvation class ⇒ per-attempt clean odds far worse than estimated; blind resubmits are waste.
- **Analysis** — Infrastructure (provisioning), with the in-repo hardening (cycle 7, strengthening in progress) as the mitigation. Decision: NO further launches from pre-fix SHAs. Per Query 6, both the fit-probe resubmit and the full run go from the post-fix commit once cycle 7 closes.
- **Next** — Cycle-7 strengthen → follow-up review → commit → fit probe #4 from post-fix SHA → on PASS, full run.

## 2026-07-19T01:55:00Z — Cycle 7 CLOSED (final APPROVE) → launch freeze lifted

- **Goal** — Close the hardening cycle; resume launches from the post-fix commit.
- **Change** — setup.sh: jammy-safe timer/service stops, bounded PID-exact escalation with exit-after-KILL, global 420s budget over apt+curl execution, EPHEMERAL_WORKER-gated persistence (launcher sets it); tests comment-insensitive, zero skips. 100/100 green.
- **Result** — `passed` — reviewer re-derived the timing arithmetic and approved with zero launch-blocking issues.
- **Next** — Fit probe #4 from this commit (entry 6); on PASS → full run (Queries 5+6).

## 2026-07-19T02:40:00Z — Fit probe #4: setup hardening WORKED; probe FAILED on real HBM limit — per-device 8 does not fit v6e-64

- **Goal** — Verdict probe #4 (`20260718-233800-5d773c8b`, post-fix commit `0ffd950`).
- **Result** — `partial` — **the cycle-7 hardening succeeded**: zero setup stalls, all 16 hosts reached compile (3/3 prior failures never got here). The probe then failed on its actual question: `CompileTimeHbmOom: 31.28G/31.25G, over by 36.92M` at per-device batch 8.
- **Analysis** — **Real capacity finding, not infra.** v6e-8 overflowed by 44MB, v6e-64 by 37MB — near-identical despite ~4.4GB/chip less resident state on 64 chips ⇒ the headroom is consumed by FSDP collective buffers (per-layer weight all-gathers, grad reduce-scatters, XLA prefetch) that full-FT pays and the frozen-backbone adapter runs did not (no weight grads/moments, no grad reduce-scatter). My provisional headroom argument (worklog 2026-07-18T21:10) was wrong; the reviewer's fit-is-provisional-until-probe gate (round-3 F9) was the correct epistemic stance — on the record. Remat is already FULL; a 37MB miss might yield to XLA-flag tuning but that is fragile ground for a diagnostic. Robust remedy: **per-device 4**.
- **Next** — Propose to Yixun: probe #5 at per-device 4 (GBS 256) → on PASS, full run amended to **GBS 256 × 20000 steps** (same 3.55 passes, same total compute, same 2500-multiple checkpoint/eval structure; cohort protocol unchanged). Plan §2.2 amendment on approval. NO launch until approved (config change).

## 2026-07-19T16:27:02Z — Query 7: amended run approved → probe #5 launch (per-device 4)

- **Goal** — Rung-6 fit probe at the amended batch; acceptance criteria below.
- **Command / Validation** — `WAN_EXPERIMENT=full_ft SMOKE=1 TPU_CHIPS=64 PER_DEVICE_BATCH_SIZE=4 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=256 GLOBAL_BATCH_SIZE_TO_LOAD=256 NAME=wan-full-ft-fitprobe-yixun bash bash_scripts/launch_wan_train.sh` (entry 7 in `_command.md`).
- **Acceptance criteria** — COMMIT=c01722c; 64 devices; asserts pass; 5.00B trainable; per-dtype + audit lines; batch echo 4/256/256; **compiles and completes step 1, no HBM OOM**; zero checkpoints. Setup phase: no apt stalls (hardening in effect).
- **Result** — `launched` (job id below on submission).
- **Next** — PASS → amended full run from the post-cycle-8 commit (recipe amendment under review in parallel). Plan §2.2 amendment noted as v3.1 (Query 7 provenance).

## 2026-07-19T17:35:00Z — Cycle 8 CLOSED: amended recipe (GBS 256 × 20k) committed

- **Goal** — Encode Query 7's amendment into the committed recipe, reviewed.
- **Change** — yml `per_device_batch_size: 4.0` / documentary 256/256 / `max_train_steps: 20000`; launcher full_ft arm 20000 + arm-dependent batch defaults (4/256/256) via `*_DEFAULT` restructure (set-vs-unset semantics preserved); RUN_NAME interpolates the resolved GBS (review-ordered — names no longer lie); wrapper comment accuracy. 100/100 tests; goldens prove adapter arms byte-identical.
- **Result** — `passed` — review APPROVE-WITH-CHANGES, single ordered change fixed with exact-split mutant proof.
- **Next** — Probe #5 verdict → on PASS, launch the amended full run FROM THIS COMMIT (Queries 6+7).

## 2026-07-19T16:52:22Z — Probe #5 PASS (log-verified) → FULL RUN LAUNCH

- **Goal** — Launch the amended primary run (rung 7). All gates green: probe #5 SUCCEEDED first-attempt (job `20260719-162702-4d29b151`; worker-8 = JAX process 0: 5.00B trainable, mixed-dtype lines, audit table on fsdp:64 specs, per-host physical 0.62+1.24 GB, batch echo 4/256, JOB_EXIT=0, no OOM — note for the runbook: queue worker index ≠ JAX process index on multi-host slices; host-0 logs can live in any worker-N.log). Cycle 8 closed at `031228e`. Approvals: Queries 5+6+7.
- **Acceptance criteria (plan §6, amended):** worker reports COMMIT=031228e; 64 devices; GBS **256** (echo 4/256/256); guide 1.0 + fresh asserts pass; 5.00B trainable, no adapter line; per-dtype + audit lines; eval dir …/train; RUN_NAME contains **gbs256**; ≥1 step, finite loss by step 10 log line; wandb live (`train/loss` descending by step 500); checkpoints appear at 2500-multiples (~30 GB each), keep-period retains all.
- **Command / Validation** — `WAN_EXPERIMENT=full_ft TPU_CHIPS=64 NAME=wan-full-ft-yixun bash bash_scripts/launch_wan_train.sh` (entry 8). Runtime estimate: refreshed from the first sps log lines.
- **Result** — `launched` (job id below).
- **Next** — Startup verification once RUNNING; then loss-curve checks at 500/1000-step marks; cohort validation asks after checkpoint 5000 exists.

## 2026-07-19T22:30:00Z — PRIMARY RUN COMPLETE: all acceptance criteria pass; loss plateaus ~0.176

- **Goal** — Verify and record the primary run.
- **Result** — `passed` (as a run) — 20000/20000 steps, attempt 1, no preemptions, 4h40m, 1.90 steps/s; all §6 criteria log-verified (process-0 logs in worker-6 — worker≠process indexing again); 8/8 checkpoints retained. Loss: 0.60→0.19 by step 500, then hard plateau 0.176–0.183 through 20k; eval-on-train mirrors. wandb context: pre-context adapter plateaued 0.57–0.60 under its guide-5 objective (not directly comparable; noted as context only).
- **Analysis** — Pipeline trains cleanly unfrozen (favors "optimizable"); no memorization signature in one-step loss at 3.55 passes (inconclusive per plan §2.4 asymmetry — the designed metric is the cohort rollout). Runtime estimate lesson: actual 4.7h vs my 1.5–3-day guess — short latent sequences; recorded for future planning.
- **Next** — Yixun approval for cohort validation: 5 v6e-8 jobs (checkpoint_step 0/5000/10000/15000/20000, validation_ordinals per `_params_set_up.md`, seed 0). Then `_analysis.md` + HTML report.

## 2026-07-20T14:21:48Z — Cohort validation launched (5 × v6e-8, approved)

- **Goal** — The designed memorization metric: rollout the 16 predeclared train clips at step-0 baseline + 4 checkpoints.
- **Acceptance criteria (per job):** worker reports COMMIT=c562856; restores exactly its CHECKPOINT_STEP (step-0 = manager-free bypass, log shows no Orbax restore); 16 samples processed in listed-ordinal order, seed 0; `summary.json` + per-sample metrics/videos under `wan-full-ft-v6e64-full-gbs256-fresh-20260719-165222/validation/step_NNNNNN/`; JOB_EXIT=0. Cross-job: identical cohort/seeds (only the checkpoint varies).
- **Result** — `launched` — 5 jobs, ids in `_command.md` entry 9.
- **Next** — All terminal → pull 5 summaries → within-cohort delta table → `_analysis.md` + HTML report + Codex analysis review → final report to Yixun/Lihan.

## 2026-07-21T18:50:00Z — Analysis review (6 MAJOR) → analysis v2; HTML verdict aligned

- **Goal** — Close the analysis-review cycle.
- **Change** — Codex analysis review: REQUEST-REVISION, 6 MAJOR findings (verdict overreach, memorization claim unsupported, provisional-pending-s0, §2.4 rationale, exp_02 speculation labeling, reliability caveats). ALL accepted; `_analysis.md` rewritten to v2 (provisional verdict, narrow claims, identifying-design spec for exp_02, full caveat set); resolutions appended to the review file; HTML verdict box reworded to match F1.
- **Result** — `passed` — the record now claims exactly what the data supports, no more.
- **Next** — Yixun: video review at leisure + merge decision. s0b lands → official row swaps in. Optional held-out cohort job (analysis §4.4c) if Lihan wants the memorization question answered.

## 2026-07-22T13:40:00Z — Official s0 landed → analysis FINAL

- **Goal** — Complete the predeclared comparator protocol.
- **Result** — `passed` — s0b SUCCEEDED (att 2): official 16/16 pretrained baseline SSIM **0.1966** / latent 3.479 / pixel 0.199 — confirms the preliminary 14/16 (0.20/3.51/0.199). `_results.md` row finalized, HTML updated (all points now official), `_analysis.md` status PROVISIONAL → **FINAL** (verdict wording unchanged — the review-approved form). exp_01 evidence collection is closed.
- **Next** — Yixun: merge decision (SOP rule 4); optional video review; optional held-out cohort job (analysis §4.4c).

## 2026-07-26T15:10:00Z — Query 8 recorded; Part-II plan drafted (val-set evaluation)

- **Goal** — Continue exp_01: full val loss per checkpoint (T1) + step-20k val visualization/gallery (T2), offline-only.
- **Change** — Query 8 appended (verbatim + 3 interpretation notes: position-as-ordinal, plan-review-still-runs, merge-still-open); `plan_full_ft_overfit.md` gains Part II (D1–D7 design, per-file code plan, 3 Coder cycles, launch gate).
- **Command / Validation** — Grounding attempted: val `summary.json` unreadable and shard listing empty → **gcloud reauth required** (blocks rung 3 + launches only; local cycles proceed). 14,636 count taken from Query 8 pending rung-3 verification + the evaluator's own hard assertion.
- **Result** — `in_progress` — Codex plan review (Part II scope) launching next.
- **Next** — Plan review → resolve → cycles A/B/C → rungs 1–3 → pre-launch package to Yixun (launch reserved).

## 2026-07-26T16:05:00Z — Part-II plan review: APPROVE-WITH-CHANGES → plan v2, cycles begin

- **Goal** — Close the Part-II plan cycle.
- **Result** — `passed` — 7 findings (6 MAJOR, 1 MINOR), all accepted: RNG contracts + position-indexing test, EOF-drain/full-scan/golden-aggregate, allclose-not-bitwise for the vector mean, `requested_step` kwarg (pyconfig immutability trap caught pre-code), T1 smoke gate + VAE deletion, 9-column schema + mandatory plot, gallery position-join semantics. Plan v2 in place; resolutions appended to `full_ft_overfit_codex_plan2_review.md`.
- **Next** — Coder cycle A `val-loss-core`.

## 2026-07-26T17:20:00Z — Cycle A (val-loss-core) CLOSED

- **Goal** — Part-II cycle 1 of 3: per-example loss helper + deterministic RNG + batching/aggregation pure functions.
- **Change** — `masked_velocity_mse_per_example` (+32, additive; scalar training helper hash-verified untouched); `eval_wan_full_ft_val_loss.py` pure part (`per_example_rng`/`plan_batches`/`aggregate`, float64 host reductions, ddof=1); 20 tests. Review APPROVE-WITH-CHANGES (adjudications 5/5 ACCEPTED — incl. the bitwise→1-ULP-closeness deviation, empirically forced by CPU XLA and consistent with the reviewer's own plan-F3 wording); 1 MINOR fixed test-side (generic num_steps bounds+support coverage). 6 mutants killed total.
- **Result** — `passed` — 120/120 green.
- **Next** — Cycle B `val-loss-evaluator` (Coder launching); rung-3 scan re-downloading cleanly (gsutil pileup from colliding parallel downloads killed — infra note: two concurrent gsutil -m runs on one destination thrash .gstmp slices; sequential re-fetch in flight).

## 2026-07-26T18:35:00Z — Rung 3 (full val-split scan): PASS

- **Goal** — F2-upgraded dataset verification before any launch.
- **Command / Validation** — All 8 shards (3.16 GiB) downloaded (sequential, post gsutil-collision cleanup) and every record parsed locally: per-record byte schema (z_i0 23040 / z_video 207360 / actions 896) asserted on all 14,636 records.
- **Result** — `passed` — TOTAL=14,636 exactly; stored ordinals 0..14,635 globally contiguous (⇒ stored ordinal ≡ dataset position on this split — evaluator still keys RNG by position per contract, the two coincide); 0 duplicate names; T2 positions 0/2927/5854/8781/11708/14635 all valid (last = final record).
- **Analysis** — Dataset integrity fully established; the evaluator's runtime EOF-drain + n==14,636 assertion remains as the launch-time guard.
- **Next** — Cycle B in progress; local scan artifacts deleted (3.2 GB freed).
