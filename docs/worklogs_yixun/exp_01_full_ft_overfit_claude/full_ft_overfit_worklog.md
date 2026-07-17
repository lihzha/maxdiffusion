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
