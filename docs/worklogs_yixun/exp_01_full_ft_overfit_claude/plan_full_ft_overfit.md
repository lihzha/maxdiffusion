# plan_full_ft_overfit — Wan TI2V full-finetune overfit diagnostic

Planner: Claude Fable 5 (max effort). Status: **draft for Codex review** (per SOP: plan → plan review → resolve findings → re-review if material → user approval). Base commit: exp branch `claude-exp_01_full_ft_overfit-20260715` @ `2dd8cab` (== `yixun-dev` @ `8258965`).

## 1. Objective

Measure how fast a **fully-trainable** Wan2.2 TI2V 5B backbone (NO adapter) can overfit/memorize the full cached-DROID train split, holding the data pipeline, objective, noise schedule, and conditioning **identical** to the adapter runs. This isolates the question: *is the pipeline learnable at all (A), or is frozen-backbone + adapter optimization the bottleneck (B)?*

Decision rule (from `_yixun_query.md`): fast overfit → pipeline OK, continue adapter work; no overfit after substantial steps → debug data/loss/noise/CFG/latent alignment before touching adapter structure.

## 2. Design (all parity choices explicit)

**Approach: a new self-contained trainer that is the side-adapter trainer minus the adapter.** Not `WanTI2VTrainer` (different data path — would confound the diagnostic). We reuse the side-adapter trainer's exact dataset iterator, sigma construction, timestep sampling, noise builder, frame-0 pinning, loss mask, and normalization by importing the same helper functions — parity by construction, not by re-implementation.

| Aspect | Adapter run (reference) | This probe | Rationale |
|---|---|---|---|
| Trainable params | adapters only (~128M/240M), backbone frozen | **all transformer params (~5B)**, no adapters exist | the experiment variable |
| Data | DROID TFRecords `gs://v6_east1d/datasets/droid_wan_side_adapter/train` | **identical** (same loader code) | parity |
| Conditioning | z_i0 pin + null-text + action tokens | **z_i0 pin + null-text only** (actions parsed but unused) | user decision: first-frame + video only |
| Objective | one-step flow-matching velocity MSE, frame-0 masked out | **identical math** | parity |
| σ/t sampling | `build_rollout_sigmas(25, shift=5.0)`, uniform t index | **identical** | parity |
| Noise | `fresh` | **`fresh`** | parity + known `fixed`-mode bug |
| CFG in loss | `guide_scale=5.0`, frozen uncond branch, stop-grad | **removed (≡ guide_scale=1.0)** | see §2.1 — with no adapter CFG degenerates and silently scales gradients ×5 |
| Optimizer | AdamW b1=0.9 b2=0.999 eps=1e-8 wd=1e-2, global-norm clip 1.0 | **identical family**, LR lowered (§2.2) | 5e-5 is an adapter LR; full-FT 5B needs ~1e-5 |
| Precision | weights/activations bf16, remat FULL | **identical** | parity; §7 risk R3 notes bf16 Adam moments |
| Parallelism | v6e-64, pure FSDP, GBS 512 | **identical** | parity + memory need |

### 2.1 Why CFG must be dropped (not just "left at 5.0")

In the adapter loss, `v_cond` (adapter-conditioned) ≠ `v_uncond` (frozen plain backbone), and `v_pred = v_uncond + s·(v_cond − v_uncond)` with stop-grad on `v_uncond` trains the adapter through an amplified difference. With no adapter, both branches are *the same trainable network on the same inputs*: `v_pred ≡ v_cond` numerically, but the stop-grad mixing makes `∂v_pred/∂θ = s·∂v_cond/∂θ` — a silent **5× gradient scale** (equivalent to 5× LR) plus a wasted second forward. The probe therefore hard-bypasses the CFG branch; the config ships `side_adapter_guide_scale: 1.0` and the trainer asserts it.

### 2.2 Hyperparameters (probe recipe)

- `learning_rate: 1e-5` (AdamW; standard full-FT range for ~5B diffusion; adapter runs' 5e-5 would risk instability on the full backbone). Warmup `warmup_steps_fraction: 0.05`, same schedule shape as the reference.
- `max_train_steps: 10000` @ GBS 512 ⇒ 5.12M samples seen — many epochs over the DROID window set; enough to see a memorization trend clearly (the adapter run's loss curve over the same first 10k steps is the comparison).
- `checkpoint_every: 2500`, `max_to_keep=3` (hardcoded in the manager) — full-FT checkpoints are ~10 GB params + ~20 GB Adam moments (bf16), so ~30 GB/save; 4 saves + final ≈ manageable on `gs://v6_east1d`.
- Eval during training: `eval_every: 1000` with **`eval_data_dir` pointed at the TRAIN shards** — for a memorization probe, the interesting "eval" is one-step loss on training data with fresh noise/timesteps (val-split loss is secondary; we log it in the final validation instead).

### 2.3 Memorization validation (after/at end of training)

Extend `generate_wan_side_adapter.py` to a no-adapter mode and run it with `eval_data_dir` = **train** shards: 25-step rollout from noise conditioned on frame 0, reconstructing *training* clips; report latent/pixel MSE + SSIM + comparison videos. Success looks like SSIM on train clips ≫ the 0.29 the adapter run scored on val clips.

## 3. Planned code, per file

**New file `src/maxdiffusion/trainers/wan_ti2v_full_ft_trainer.py`** (~200 LOC), self-contained, subclassing `WanTI2VSideAdapterTrainer` to inherit `_create_scheduler`, `_load_wan_pipeline`, `_compute_null_context`, `_load_dataset`, `_build_optimizer`, checkpoint manager/save/restore, and the train-loop shell. Contents:
- `FullFTTrainState(train_state.TrainState)`: fields `graphdef`, `rest_of_state` (the transformer's), `null_context`. `params` = transformer params (trainable). No adapter fields.
- `full_ft_build_noisy_latents(z_video_f32, z_i0_f32, sigma_t) -> z_t_f32` — pure: `(1−σ)·z + σ·ε` then `apply_first_frame_pin` (ε passed in). *(TDD unit A)*
- `full_ft_velocity_loss(v_pred, v_target, batch) -> loss` — pure: frame-0-masked MSE with `n_valid = sum(mask)·b` normalization, exactly the reference lines. *(TDD unit B)*
- `_denoising_loss(params, state, data, rng, config, scheduler)` — same skeleton as the reference: reuses imported `build_rollout_sigmas`, `_sample_step_indices`, `_build_noise`, `_build_per_token_timestep`, `apply_first_frame_pin`; forward is **plain** `transformer(hidden_states=z_t, timestep=timestep_2d, encoder_hidden_states=null_context, deterministic=False)`; **no adapter, no actions, no CFG branch**; same aux metrics minus adapter-specific ones.
- `_train_step` / `_eval_step` — as reference, over `FullFTTrainState`.
- `WanTI2VFullFTTrainer(WanTI2VSideAdapterTrainer)`:
  - `start_training` override: split transformer → `params` trainable; **assert `abs(config.side_adapter_guide_scale − 1.0) < 1e-6`** with a message referencing §2.1; log "trainable transformer params: X.XXB" (acceptance criterion ≈ 5B); no `_build_adapters` call.
  - `_shard_state` override: keep the **computed FSDP shardings** for `params`/`opt_state` (the reference's replicate-override is for tiny adapter trees and would OOM instantly on 5B). *(TDD unit C: the shard-tree selection logic, testable with fakes)*
- Checkpoint layout unchanged (`params`/`opt_state`/`step` Composite) — `params` is now the full transformer; the validation script restores it symmetrically.

**Edit `src/maxdiffusion/train_wan.py`** (+3 lines): `elif config.model_type == "FULL_FT_TI2V": from …wan_ti2v_full_ft_trainer import WanTI2VFullFTTrainer`. *(folded into TDD unit C's commit)*

**New file `src/maxdiffusion/configs/base_wan_5b_full_ft.yml`**: copy of `base_wan_5b_side_adapter.yml` with deltas — `model_type: 'FULL_FT_TI2V'`, `side_adapter_guide_scale: 1.0`, `learning_rate: 1.e-5`, `max_train_steps: 10000`, `checkpoint_every: 2500`, `output_dir: gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft`, comment header stating the probe's purpose; adapter keys retained but inert (documented as such) so the shared loader keeps working.

**Edit `src/maxdiffusion/generate_wan_side_adapter.py`** (~25 LOC): in `_restore_validation_state`, branch on `config.model_type == "FULL_FT_TI2V"` → build `FullFTTrainState` via the new trainer (no adapters) and restore the transformer params; in `_rollout_sample`'s `_body`, when full-FT call `transformer(...)` directly (the plain call already exists as the `v_uncond` branch — reuse that expression, skip `wan_action_adapter_forward`). *(TDD unit D: forward-selection helper + restore-state branch)*

**New file `bash_scripts/train_wan_full_ft.sh`**: copy of `train_wan_side_adapter.sh` pointing at the new yml; env knobs `LEARNING_RATE`, `MAX_TRAIN_STEPS`, batch sizes; drops adapter-specific envs; keeps the `SIDE_ADAPTER_NOISE_MODE` explicit-pass convention (still a real knob for the loss).

**Edit `bash_scripts/launch_wan_train.sh`**: add `full_ft` arm to the `WAN_EXPERIMENT` case (mirrors the two existing arms; runs `train_wan_full_ft.sh`).

**New tests in `src/maxdiffusion/tests/worklogs_yixun/`** (CPU-only, no 5B weights): `test_full_ft_overfit_build_noisy_latents.py` (unit A: exact values, pin correctness), `test_full_ft_overfit_velocity_loss.py` (unit B: mask excludes frame 0; normalization; invariance to frame-0 perturbations), `test_full_ft_overfit_trainer_wiring.py` (unit C: shard-selection logic with fake trees; dispatch mapping; guide-scale assertion fires), `test_full_ft_overfit_generate_forward.py` (unit D: forward-selection returns plain-transformer callable for FULL_FT_TI2V).

## 4. Coder rounds (closed cycles, in order)

1. **unit-a-noisy-latents** — test + `full_ft_build_noisy_latents`.
2. **unit-b-velocity-loss** — test + `full_ft_velocity_loss`.
3. **unit-c-trainer-wiring** — tests + `FullFTTrainState`, `_denoising_loss`/`_train_step`/`_eval_step`, trainer class, dispatch, guide-scale assert, shard override.
4. **unit-d-generate-and-launch** — tests + generate no-adapter branch, new yml, both bash scripts.

Each round: write (Opus 4.8 max, test-first) → briefed Codex `gpt-5.6-sol` xhigh review (`full_ft_overfit_codex_code_<marker>_review.md`) → strengthen (resolutions appended) → commit (<200 LOC).

## 5. Validation ladder mapping

1. `py_compile` all touched files; `yaml.safe_load` the new yml; `bash -n` both scripts; pytest suite above.
2. Tiny synthetic forward: CPU, stub transformer callable through `_denoising_loss` path (shapes/dtypes/mask).
3. Real-data readback: parse 2 records from the train shards; assert `z_i0 [48,1,12,20] f16`, `z_video [48,9,12,20] f16`, `actions [32,7] f32` + finite stats.
4. n/a (no dataset build).
5. Smoke on **v6e-8**: GBS 8, ~20 steps, checkpoints/final-save disabled; confirms 5B full-FT step compiles, fits, loss finite, and logs "trainable transformer params ≈ 5B".
6. Fit probe on v6e-64: confirm GBS 512 fits with FSDP + remat FULL (adapter run's activation memory is the dominant term and is unchanged; new memory = grads + Adam moments, ~30 GB sharded over 64 chips — expected to fit comfortably).
7. Full run: v6e-64, GBS 512, 10k steps, from the audited, pushed commit.

## 6. Launch acceptance criteria (to be copied into `_worklog.md` at launch)

Worker reports the exp-branch SHA; 64 devices; per-device batch 8 (GBS 512); log line `trainable transformer params: ~5.0B` present and **no** adapter-param line; `guide scale: 1.0`, `noise mode: fresh`, `t sampling: uniform` in the startup log; ≥1 optimizer step, loss finite, no OOM/NaN; wandb run live with `train/loss` descending over the first 500 steps.

## 7. Risks / knobs

- **R1 (silent ×5 gradients)** — mitigated by hard assert on guide scale (§2.1).
- **R2 (LR wrong for full FT)** — 1e-5 chosen; if loss plateaus early *and* grad-norm is tiny, retry 2e-5 (one knob, one rerun; recorded in `_command.md`).
- **R3 (bf16 Adam moments)** — parity with the reference keeps params + moments bf16; for a memorization probe this is acceptable, but if loss stalls anomalously with healthy grad norms, suspect moment precision before blaming the pipeline (would itself be a finding).
- **R4 (checkpoint size)** — 30 GB/save × keep-3 on GCS; keep `checkpoint_every: 2500`.
- **R5 (throughput)** — full-FT step adds weight-grad + optimizer work over the adapter run (rough expectation 1.3–1.8× slower); measured at smoke, budget adjusted then if needed.

## 8. What success/failure looks like (ties to decision rule)

- **Pipeline OK:** train loss drops decisively below the adapter run's plateau within ≤10k steps, and train-clip rollouts reconstruct (SSIM ≫ 0.29, visually matching motion). → Continue adapter/embedding-supervision work.
- **Pipeline suspect:** loss tracks the adapter run's plateau (or NaNs/stalls with healthy optimizer signals), train-clip rollouts stay generic. → Debug data/loss/noise/CFG/latent alignment; adapter structure exonerated for now.
