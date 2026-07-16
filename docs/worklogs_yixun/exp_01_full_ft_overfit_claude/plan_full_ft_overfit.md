# plan_full_ft_overfit — Wan TI2V full-finetune overfit diagnostic

Planner: Claude Fable 5 (max effort). **v2 — revised per Codex plan review** (`full_ft_overfit_codex_plan_review.md`, verdict REQUEST-REVISION; all 10 findings accepted, resolutions appended there). Status: awaiting re-review, then user approval. Base: exp branch `claude-exp_01_full_ft_overfit-20260715` @ `ce5fc4f`.

## 1. Objective

Measure how fast a **fully-trainable** Wan2.2 TI2V 5B backbone (NO adapter) can overfit/memorize the full cached-DROID train split (**1,440,554 windows**, 704 shards), holding the data pipeline, objective, noise schedule, and conditioning identical to the adapter runs. This isolates: *is the pipeline learnable at all (A), or is frozen-backbone + adapter optimization the bottleneck (B)?*

Decision rule (from `_yixun_query.md`), with the evidential asymmetry made explicit (review F4):
- **Positive** (fast overfit) is valid at any point it appears — pipeline exonerated, continue adapter work.
- **Negative at 10k steps (3.55 train-set passes) is inconclusive**, not "pipeline suspect". The escalation protocol in §2.4 must run before any pipeline-suspect verdict.

## 2. Design (parity choices explicit)

**Approach: a new self-contained trainer that is the side-adapter trainer minus the adapter.** Not `WanTI2VTrainer` (different data path — would confound). Parity is enforced by **extracting the objective math into shared helpers that BOTH trainers call** (review F3), not by re-implementation and not by claim.

| Aspect | Adapter run (reference) | This probe | Rationale |
|---|---|---|---|
| Trainable params | adapters only (~128M/240M), backbone frozen | **all transformer params (~5B)**, no adapters exist | the experiment variable |
| Data | DROID TFRecords `gs://v6_east1d/datasets/droid_wan_side_adapter/train` (1,440,554 windows) | **identical** (same loader code) | parity |
| Conditioning | z_i0 pin + null-text + action tokens | **z_i0 pin + null-text only** (actions parsed but unused) | user decision |
| Objective | one-step flow-matching velocity MSE, frame-0 masked | **same code via shared helpers** | parity by shared code |
| σ/t sampling | `build_rollout_sigmas(25, shift=5.0)`, uniform t | **identical** | parity |
| Noise | `fresh` (config) — but the reference *shell wrapper* defaults `fixed` | **`fresh`, enforced**: trainer asserts `side_adapter_noise_mode == "fresh"`; new wrapper defaults `fresh` | review F1 (BLOCKER); known fixed-noise failure |
| CFG in loss | `guide_scale=5.0`, frozen uncond branch, stop-grad | **bypassed (≡ guide_scale=1.0), asserted** | §2.1 |
| Optimizer | AdamW b1=.9 b2=.999 eps=1e-8 wd=1e-2, global-norm clip 1.0 | **identical family**, LR 1e-5; param/moment dtypes logged at startup | §2.2, review F6 |
| Precision | weights/activations bf16, remat FULL | **identical for the primary run**; fp32-optimizer-state control predeclared in §2.4 | review F6 |
| Parallelism | v6e-64, pure FSDP, GBS 512 | **identical** | parity + memory |

### 2.1 Why the CFG branch is bypassed (corrected per review F2)

With no adapter, both CFG branches are the same trainable network on the same inputs: `v_pred ≡ v_cond` numerically, and `stop_gradient(v_uncond)` makes the raw derivative `∂v_pred/∂θ = s·∂v_cond/∂θ` — a **5× pre-optimizer gradient multiplier plus a wasted second forward pass** (dropout is 0.0, so the branches are exactly identical). It is *not* a 5× effective learning rate: global-norm clipping runs before AdamW and Adam's moment normalization largely absorbs a constant scale; weight decay is unscaled. The bypass is still mandatory — it removes an uncontrolled interaction with the clip threshold and halves compute. The config ships `side_adapter_guide_scale: 1.0` and the trainer **asserts** it.

### 2.2 Hyperparameters (probe recipe)

- `learning_rate: 1e-5` (AdamW, warmup fraction 0.05, then the reference schedule shape — warmup-to-constant per `max_utils.create_learning_rate_schedule`). Reference parity holds for the Adam coefficients, clipping, and wd 1e-2; the LR itself is a full-FT choice, not a parity value (review F6 wording).
- `max_train_steps: 10000` @ GBS 512 ⇒ 5.12M samples ⇒ **≈3.55 passes** over the 1,440,554-window train set (review F4). Enough for a *positive* memorization signal; explicitly **not** enough to conclude failure (§2.4).
- `checkpoint_every: 2500`, keep 3 — bf16 full-FT checkpoint ≈ 10 GB params + ≈ 20 GB Adam moments.
- In-training eval on TRAIN shards: config yml sets **`eval_data_dir: gs://v6_east1d/datasets/droid_wan_side_adapter/train`** and the `full_ft` launcher arm sets `EVAL_DATA_DIR="$TRAIN_DATA_DIR"` (review F7 — the current queue launcher would otherwise force the val dir). The startup log records the resolved eval path; acceptance criteria check it.
- `side_adapter_noise_mode: fresh` — asserted in the trainer; wrapper default `fresh` (review F1).

### 2.3 Memorization validation (review F5 — cohort protocol)

- **Fixed train cohort:** N=16 training windows at predeclared, diverse shard/record ordinals + fixed rollout seeds, recorded in `_params_set_up.md` before the full run starts.
- **Evaluate on the same cohort:** the **pretrained backbone (step 0)** and checkpoints **2500 / 5000 / 7500 / 10000** — 25-step rollout conditioned on frame 0; latent MSE, pixel MSE, SSIM + comparison videos, via the extended no-adapter `generate_wan_side_adapter.py`.
- **Success metric is within-cohort:** the memorization *delta* over the pretrained baseline (step-0 rollouts on the identical cohort/seeds). No cross-split, cross-method numeric threshold. Context only (never thresholds): fresh side-adapter r20 reached val-clip SSIM ≈0.664 @ 2k / ≈0.615 @ 10k; pre-context reached ≈0.30 (review F5 corrected my earlier 0.29-as-threshold).
- Validation command records its dataset path + cohort spec in `_command.md`.

### 2.4 Escalation protocol before any negative verdict (reviews F4 + F6)

A "pipeline suspect" conclusion requires ALL of, in order, each logged in `_command.md`/`_worklog.md`:
1. **Extended budget:** continue to 30k steps (≈10.7 passes) if 10k shows no memorization trend.
2. **LR control:** one run/segment at `learning_rate: 2e-5` (R2).
3. **Optimizer-precision control:** one run with **fp32 optimizer state** (`optax.adamw(mu_dtype=fp32)` + fp32 nu if needed, or fp32 master params if optax's accumulator default proves insufficient — optax is pinned only `>=0.2.8`, so accumulator dtype is an uncontrolled default; the trainer logs actual param/moment dtypes at startup so the primary run's precision is on record). Memory delta ≈ +40 GB sharded over 64 chips — fits; checkpoints grow to ≈70 GB.
Only if memorization fails across 1–3 does the experiment conclude "debug data/loss/noise/CFG/latent alignment first."

## 3. Planned code, per file

**Edit `src/maxdiffusion/models/wan/side_adapter_wan.py`** (+~30 LOC): add shared objective helpers (review F3) —
- `build_noisy_pinned_latents(z_video_f32, z_i0_f32, eps, sigma_t) -> z_t_f32`: `(1−σ)·z + σ·ε`, then `apply_first_frame_pin`. (`eps` is an explicit argument — F3 signature fix.)
- `masked_velocity_mse(v_pred, v_target, batch_size) -> loss`: frame-0-masked MSE, `n_valid = sum(mask)·b`, exactly the reference lines.

**Edit `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`** (−~10/+~6 LOC): `_denoising_loss` calls the two shared helpers instead of its inline copies. **Behavior-preserving refactor** → characterization test first (fixed-RNG loss value identical before/after, per SOP red-rule).

**New file `src/maxdiffusion/trainers/wan_ti2v_full_ft_trainer.py`** (~180 LOC), subclassing `WanTI2VSideAdapterTrainer` (inherits scheduler, pipeline load, null-context, dataset loader, optimizer factory, checkpoint manager, loop shell):
- `FullFTTrainState(train_state.TrainState)`: `graphdef`, `rest_of_state` (transformer's), `null_context`; `params` = transformer params.
- `_denoising_loss`: same skeleton via shared helpers + imported `build_rollout_sigmas`/`_sample_step_indices`/`_build_noise`/`_build_per_token_timestep`; forward = **one** plain `transformer(...)` call; no adapter, no actions, no CFG.
- `_train_step`/`_eval_step` over `FullFTTrainState`.
- `WanTI2VFullFTTrainer`: `start_training` override — **asserts** `guide_scale == 1.0` (§2.1) and `side_adapter_noise_mode == "fresh"` (F1); logs trainable-param count (≈5B expected), **param dtype and Adam moment dtypes** (F6), and resolved train/eval data dirs (F7).
- `_shard_state` override: keep computed FSDP shardings for `params`/`opt_state` (no replicate override), **retain `_apply_actual_sharding_for_tpu`**, and log global + per-host addressable byte totals for params and opt_state (F9). Fit treated as provisional until the v6e-64 probe passes.

**Edit `src/maxdiffusion/train_wan.py`** (+3 lines): `FULL_FT_TI2V` dispatch.

**New file `src/maxdiffusion/configs/base_wan_5b_full_ft.yml`**: copy of the side-adapter yml with deltas — `model_type: FULL_FT_TI2V`, `side_adapter_guide_scale: 1.0`, `learning_rate: 1.e-5`, `max_train_steps: 10000`, `checkpoint_every: 2500`, **`eval_data_dir: …/train`** (F7), `output_dir: gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft`, header comment on purpose + inert adapter keys.

**Edit `src/maxdiffusion/generate_wan_side_adapter.py`** (~40 LOC, review F8): explicit full-FT branches in BOTH `_restore_validation_state` (build `FullFTTrainState`, restore transformer params/opt_state/step from the full-FT checkpoint) and `_rollout_sample` (state merge without adapter fields; body calls plain `transformer(...)`). Checkpoint symmetry proven by test, not asserted (below).

**New file `bash_scripts/train_wan_full_ft.sh`**: copy of the side-adapter wrapper pointing at the new yml; **`SIDE_ADAPTER_NOISE_MODE` defaults to `fresh`** (F1 — the reference wrapper's `fixed` default is the documented foot-gun); `EVAL_DATA_DIR` defaults to `$TRAIN_DATA_DIR` (F7); env knobs `LEARNING_RATE`, `MAX_TRAIN_STEPS`, batch sizes.

**Edit `bash_scripts/launch_wan_train.sh`**: add `full_ft` arm — runs `train_wan_full_ft.sh`, sets `EVAL_DATA_DIR="$TRAIN_DATA_DIR"` (F7), passes fresh noise explicitly.

**Tests in `src/maxdiffusion/tests/worklogs_yixun/`** (CPU-only, no 5B weights):
- `test_full_ft_overfit_shared_objective.py` — helpers: exact values, pin correctness, mask excludes frame 0, normalization; **characterization**: fixed-RNG equality of refactored side-adapter loss vs pre-refactor formula (transcribed reference equations).
- `test_full_ft_overfit_denoising_loss.py` — fixed-RNG integration on a stub transformer (F3): fresh per-example noise (row-distinct), t/σ selection matches `build_rollout_sigmas` indexing, target `eps − z_video`, frame-0 pin present, null context used, **exactly one** transformer call (call-counting stub), actions absent from the call, and one optimizer step changes transformer params.
- `test_full_ft_overfit_trainer_wiring.py` — guide-scale assert fires at 5.0 / passes at 1.0; noise-mode assert fires at `fixed`; shard-selection keeps computed (non-replicated) specs on fake trees; dispatch maps `FULL_FT_TI2V`.
- `test_full_ft_overfit_ckpt_roundtrip.py` — tiny CPU Orbax round trip (F8): save deliberately-modified params/opt_state/step → reconstruct validation state → restore → assert rollout path consumes the restored (not initial) params.
- `test_full_ft_overfit_generate_forward.py` — forward-selection returns plain-transformer callable for `FULL_FT_TI2V`; adapter path untouched otherwise.

## 4. Coder rounds (closed cycles; restructured per review F10)

1. **shared-objective-helpers** — extract helpers + refactor reference `_denoising_loss` + characterization tests. *(touches the production adapter trainer; smallest, most scrutinized round)*
2. **full-ft-loss** — `FullFTTrainState`, `_denoising_loss`, `_train_step`/`_eval_step` + fixed-RNG integration test.
3. **trainer-wiring** — trainer class, asserts, dtype/byte logging, shard override, dispatch + wiring tests.
4. **ckpt-generation** — generate restore + rollout branches + Orbax round-trip + forward-selection tests.
5. **configs-launchers** — yml + both bash scripts (+ `bash -n`, yaml parse, noise-mode/eval-dir default greps as tests).

Each round: write (Opus 4.8 max, test-first) → briefed Codex `gpt-5.6-sol` xhigh review (`full_ft_overfit_codex_code_<marker>_review.md`) → strengthen (resolutions appended) → commit (<200 LOC).

## 5. Validation ladder mapping

1. `py_compile` touched files; `yaml.safe_load` new yml; `bash -n` both scripts; the pytest suite above.
2. Tiny synthetic forward: CPU stub-transformer through the full-FT `_denoising_loss` (shapes/dtypes/mask), already covered by the integration test.
3. Real-data readback: 2 records from the train shards → shapes `[48,1,12,20]`/`[48,9,12,20]`/`[32,7]`, dtypes f16/f16/f32, finite stats.
4. n/a (no dataset build).
5. Smoke on v6e-8: GBS 8, ~20 steps, checkpoints disabled; expects the two asserts to pass, dtype + byte-total log lines present, loss finite.
6. Fit probe on v6e-64: GBS 512 with FSDP + remat FULL; byte-total logs vs HBM; provisional until this passes (F9).
7. Full run only after 1–6 + parity audit, from a pushed, audited commit.

## 6. Launch acceptance criteria (copied into `_worklog.md` at launch)

Worker reports the exp-branch SHA; 64 devices; GBS 512; startup log shows: `trainable transformer params: ~5.0B` and **no** adapter-param line; `guide scale: 1.0`; `noise mode: fresh` (assert passed); **resolved eval_data_dir ends in `/train`**; **param dtype + Adam moment dtypes logged**; params/opt-state byte totals logged; ≥1 optimizer step, loss finite, no OOM/NaN; wandb live with `train/loss` descending over the first 500 steps.

## 7. Risks / knobs

- **R1 (silent ×5 pre-optimizer gradients)** — hard assert on guide scale (§2.1).
- **R2 (LR wrong)** — 1e-5 primary; 2e-5 is escalation control #2 (§2.4).
- **R3 (bf16 Adam accumulators)** — uncontrolled optax default (F6): dtypes logged at startup; fp32-state control predeclared as escalation #3; never conclude "pipeline suspect" from a bf16-only stall.
- **R4 (checkpoint size)** — ~30 GB/save bf16, ~70 GB under the fp32 control; `checkpoint_every: 2500`, keep 3.
- **R5 (throughput)** — full-FT step est. 1.3–1.8× the adapter step; measured at smoke; budget re-estimated then.
- **R6 (fixed-noise foot-gun)** — reference wrapper defaults `SIDE_ADAPTER_NOISE_MODE=fixed` (F1); new wrapper defaults fresh AND the trainer rejects non-fresh, so neither launch path can regress.

## 8. What success/failure looks like

- **Pipeline OK:** train loss falls decisively below its early plateau within ≤10k steps **and** cohort rollouts show a large memorization delta over the step-0 pretrained baseline (visually reconstructing the exact train clips). → Continue adapter/embedding-supervision work.
- **Inconclusive:** no memorization by 10k → run §2.4 escalation (30k steps → LR control → fp32-state control).
- **Pipeline suspect:** memorization fails through all §2.4 controls with healthy optimizer signals. → Debug data/loss/noise/CFG/latent alignment; adapter structure exonerated for now.

## 9. Answers to the Reviewer's questions

1. **Comparator:** the primary comparator is the **pretrained backbone at step 0 on the identical train cohort + seeds** (within-cohort memorization delta). Fresh side-adapter r20 (val SSIM ≈0.664 @ 2k) and pre-context (≈0.30) are reported as context, never as thresholds.
2. **Step budget:** 10k is not a hard limit — it is the positive-signal milestone; negative evidence follows the §2.4 escalation (30k + controls) before any conclusion.
3. **fp32-control storage:** accepted — ≈70 GB checkpoints on `gs://v6_east1d` for the control run are within budget.
