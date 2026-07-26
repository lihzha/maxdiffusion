# plan_full_ft_overfit — Wan TI2V full-finetune overfit diagnostic

Planner: Claude Fable 5 (max effort). **v3 — revised per Codex re-review** (`full_ft_overfit_codex_plan_review.md`: round 1 REQUEST-REVISION → v2 resolved F1–F10; re-review REQUEST-REVISION → v3 closes the partials F4/F5/F6/F9 and new finding G1). Status: awaiting focused re-review #2; Yixun pre-approval (Query 3) armed on APPROVE. Base: exp branch `claude-exp_01_full_ft_overfit-20260715` @ `03e55ac`.

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
- `checkpoint_every: 2500` **and `checkpoint_keep_period: 2500`** (finding G1): Orbax keeps every step that is a multiple of `keep_period` regardless of `max_to_keep=3`, so 2500/5000/7500/10000 all survive for the cohort evaluation (the copied yml's `5000` would have evicted step 2500). bf16 checkpoint ≈ 10 GB params + ≈ 20 GB Adam moments; 4 kept ≈ 120 GB on GCS — acceptable.
- In-training eval on TRAIN shards: config yml sets **`eval_data_dir: gs://v6_east1d/datasets/droid_wan_side_adapter/train`** and the `full_ft` launcher arm sets `EVAL_DATA_DIR="$TRAIN_DATA_DIR"` (review F7 — the current queue launcher would otherwise force the val dir). The startup log records the resolved eval path; acceptance criteria check it.
- `side_adapter_noise_mode: fresh` — asserted in the trainer; wrapper default `fresh` (review F1).

### 2.3 Memorization validation (review F5 — cohort protocol, execution path specified)

- **Fixed train cohort:** N=16 training windows at predeclared, diverse, **noncontiguous** dataset ordinals + fixed rollout seeds, recorded in `_params_set_up.md` before the full run starts.
- **Cohort selection mechanism (code):** new config key **`validation_ordinals`** (comma-separated dataset ordinals, e.g. `"0,90000,180000,…"`); the generate-script sample reader skips/selects to exactly those records (replacing the contiguous `validation_start_index` read for this mode). CPU-testable.
- **Step-0 baseline mechanism (code):** **`checkpoint_step: 0` bypasses Orbax restore entirely** and rolls out the freshly-loaded pretrained weights — a first-class, tested path (no checkpoint required for the baseline).
- **Evaluate on the same cohort:** pretrained step-0 baseline and checkpoints **2500 / 5000 / 7500 / 10000** — 25-step rollout conditioned on frame 0; latent MSE, pixel MSE, SSIM + comparison videos, via the extended no-adapter `generate_wan_side_adapter.py`. Checkpoint retention is guaranteed by `checkpoint_keep_period: 2500` (§2.2, finding G1).
- **Success metric is within-cohort:** the memorization *delta* over the step-0 baseline on identical cohort/seeds. No cross-split, cross-method thresholds. Context only: fresh side-adapter r20 val SSIM ≈0.664 @ 2k / ≈0.615 @ 10k; pre-context ≈0.30.
- Validation command records its dataset path + `validation_ordinals` + seeds in `_command.md`.

### 2.4 Escalation protocol before any negative verdict (reviews F4 + F6, fully specified)

A "pipeline suspect" conclusion requires ALL of, in order, each with its command in `_command.md` and its acceptance criteria in `_worklog.md`:
1. **Extended budget — RESUME, 20k more steps.** Resume the primary run in place (same `run_name`; Orbax restores params/opt_state/step; input iterator reseeded `seed + start_step` per the repo's resume convention) to `max_train_steps: 30000` (≈10.7 passes). Cohort evaluation at **20000 and 30000** (both multiples of `keep_period=2500`, so retained).
2. **LR control — FRESH run, 10k steps.** New `run_name` suffix `-lr2e5`, from pretrained (never resumed from the primary — mixing optimizer states would confound), `learning_rate: 2e-5`, all else identical. Cohort evaluation at 2500/5000/7500/10000.
3. **Optimizer-precision control — FRESH run, 10k steps.** New `run_name` suffix `-fp32state`, from pretrained, **mechanism: `weights_dtype: float32` override** — the loader casts transformer params to `weights_dtype`, and optax's Adam moments are created with the param dtype, so params + both moments become fp32 while activations stay bf16 via `activations_dtype` (one config knob, no new optimizer code). The startup dtype log line must show `params=float32, mu=float32, nu=float32` — that log is the control's precondition, and the wiring test asserts moments follow param dtype. State ≈ 60 GB sharded over 64 chips — fits; checkpoints ≈ 70 GB. Cohort evaluation at 2500/5000/7500/10000.

Optax note (context for #3): the dependency is pinned only `>=0.2.8`, so accumulator dtype is an uncontrolled default in the primary bf16 run — which is exactly why the primary run's startup dtype log line is an acceptance criterion (§6), putting the actual precision on record.

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
- `_shard_state` override: keep computed FSDP shardings for `params`/`opt_state` (no replicate override), **retain `_apply_actual_sharding_for_tpu`**, and log global + per-host addressable byte totals for params and opt_state (F9).
- **Large-leaf sharding audit (F9, on target hardware):** at startup, log the **8 largest param leaves and their opt-state twins** — path, shape, dtype, global bytes, addressable bytes, and resolved `PartitionSpec` — and **assert no leaf > 100 MB global resolves to a fully-replicated spec** on a multi-device mesh. The selection + assertion logic is a pure function over (tree, specs) → CPU-testable with fake trees; the real-hardware log lines are acceptance criteria at smoke (§6). Fit treated as provisional until the v6e-64 probe passes.

**Edit `src/maxdiffusion/train_wan.py`** (+3 lines): `FULL_FT_TI2V` dispatch.

**New file `src/maxdiffusion/configs/base_wan_5b_full_ft.yml`**: copy of the side-adapter yml with deltas — `model_type: FULL_FT_TI2V`, `side_adapter_guide_scale: 1.0`, `learning_rate: 1.e-5`, `max_train_steps: 10000`, `checkpoint_every: 2500`, **`checkpoint_keep_period: 2500`** (G1), **`eval_data_dir: …/train`** (F7), **`validation_ordinals: ''`** (new key so the cohort spec is CLI-overridable), `output_dir: gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft`, header comment on purpose + inert adapter keys.

**Edit `src/maxdiffusion/generate_wan_side_adapter.py`** (~60 LOC, reviews F5 + F8): (a) explicit full-FT branches in BOTH `_restore_validation_state` (build `FullFTTrainState`, restore transformer params/opt_state/step) and `_rollout_sample` (state merge without adapter fields; body calls plain `transformer(...)`); (b) **cohort reader honors `validation_ordinals`** — selects exactly the listed noncontiguous dataset ordinals (falls back to the contiguous `validation_start_index` read when empty); (c) **`checkpoint_step: 0` skips Orbax restore** and rolls out the loaded pretrained weights (step-0 baseline). Checkpoint symmetry proven by test, not asserted (below).

**New file `bash_scripts/train_wan_full_ft.sh`**: copy of the side-adapter wrapper pointing at the new yml; **`SIDE_ADAPTER_NOISE_MODE` defaults to `fresh`** (F1 — the reference wrapper's `fixed` default is the documented foot-gun); `EVAL_DATA_DIR` defaults to `$TRAIN_DATA_DIR` (F7); env knobs `LEARNING_RATE`, `MAX_TRAIN_STEPS`, batch sizes.

**Edit `bash_scripts/launch_wan_train.sh`**: add `full_ft` arm — runs `train_wan_full_ft.sh`, sets `EVAL_DATA_DIR="$TRAIN_DATA_DIR"` (F7), passes fresh noise explicitly.

**Tests in `src/maxdiffusion/tests/worklogs_yixun/`** (CPU-only, no 5B weights):
- `test_full_ft_overfit_shared_objective.py` — helpers: exact values, pin correctness, mask excludes frame 0, normalization; **characterization**: fixed-RNG equality of refactored side-adapter loss vs pre-refactor formula (transcribed reference equations).
- `test_full_ft_overfit_denoising_loss.py` — fixed-RNG integration on a stub transformer (F3): fresh per-example noise (row-distinct), t/σ selection matches `build_rollout_sigmas` indexing, target `eps − z_video`, frame-0 pin present, null context used, **exactly one** transformer call (call-counting stub), actions absent from the call, and one optimizer step changes transformer params.
- `test_full_ft_overfit_trainer_wiring.py` — guide-scale assert fires at 5.0 / passes at 1.0; noise-mode assert fires at `fixed`; shard-selection keeps computed (non-replicated) specs on fake trees; **large-leaf audit: selection of 8 largest leaves + the >100 MB-replicated assertion on fake trees** (F9); moments-follow-param-dtype check (fp32 control mechanism, F6); dispatch maps `FULL_FT_TI2V`.
- `test_full_ft_overfit_ckpt_roundtrip.py` — tiny CPU Orbax round trip (F8): save deliberately-modified params/opt_state/step → reconstruct validation state → restore → assert rollout path consumes the restored (not initial) params; **`checkpoint_step: 0` path bypasses restore and uses the initial (pretrained-stand-in) params** (F5).
- `test_full_ft_overfit_generate_forward.py` — forward-selection returns plain-transformer callable for `FULL_FT_TI2V`; adapter path untouched otherwise; **`validation_ordinals` reader selects exactly the listed noncontiguous ordinals** (F5).

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

Worker reports the exp-branch SHA; 64 devices; GBS 512; startup log shows: `trainable transformer params: ~5.0B` and **no** adapter-param line; `guide scale: 1.0`; `noise mode: fresh` (assert passed); **resolved eval_data_dir ends in `/train`**; **param dtype + Adam moment dtypes logged** (primary run expected bf16 — on record per §2.4); **large-leaf audit lines present (8 leaves + opt twins, spec + bytes) and the no-replicated->100 MB assertion passed**; `checkpoint_keep_period=2500` in the resolved config; ≥1 optimizer step, loss finite, no OOM/NaN; wandb live with `train/loss` descending over the first 500 steps.

## 7. Risks / knobs

- **R1 (silent ×5 pre-optimizer gradients)** — hard assert on guide scale (§2.1).
- **R2 (LR wrong)** — 1e-5 primary; 2e-5 is escalation control #2 (§2.4).
- **R3 (bf16 Adam accumulators)** — uncontrolled optax default (F6): dtypes logged at startup; fp32-state control predeclared as escalation #3; never conclude "pipeline suspect" from a bf16-only stall.
- **R4 (checkpoint size / aggregate retention, re-review-2 H1)** — ~30 GB/save bf16, ~70 GB fp32-control; `keep_period=2500` retains every save ⇒ worst case across primary-to-30k + both controls ≈ **760 GB** (360+120+280). Budgeted on `gs://v6_east1d`; **prune each segment's non-terminal checkpoints once its cohort evals are recorded in `_results.md`** (keep segment finals), pruning commands logged in `_command.md`.
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

---

# Part II — Validation-set evaluation (Query 8) — plan v2

Planner: Claude Fable 5 (max effort). Status: reviewed (APPROVE-WITH-CHANGES, `full_ft_overfit_codex_plan2_review.md`); all 7 findings applied below (marked ←F#). Scope: offline evaluation only — zero changes to training behavior, trainers' train paths, or checkpoints. Base: exp branch @ `3ff55ab`.

## II.1 Objective

(T1) Per-checkpoint full-validation one-step loss: the exact training objective (velocity MSE, frame-0 masked) over ALL 14,636 val windows for checkpoints 2500…20000, with per-example (t, ε) held fixed across checkpoints so the curve is purely a model effect. (T2) Qualitative val-set rollout at step 20000 on six fixed positions + an HTML gallery. Together these add the held-out instrument the Part-I analysis explicitly lacked (its F2/F6 caveats).

## II.2 Design decisions (the reviewable core)

**D1 — Deterministic per-example RNG (T1).** For val dataset position `p` (0-based, the round-4 coordinate; stored `ordinal` field NOT used as the index): `k = fold_in(key(validation_seed), p)`; `(k_t, k_eps) = split(k)`; `t_idx = randint(k_t, [], 0, side_adapter_sampling_steps)` over the SAME 25-point `build_rollout_sigmas(25, flow_shift=5.0, …)` grid as training's uniform sampling; `ε = normal(k_eps, z_video.shape, float32)`. Contracts (←F1): `randint(k_t, (), 0, num_steps, dtype=jnp.int32)` with validated `num_steps > 0`; ε generated at the UNBATCHED example shape then stacked; this reproduces training's marginal law of independent (t, ε) — distribution parity, deliberately NOT replay of training's stateful draws (held-out examples have none). Independent of checkpoint/batch/order/host by construction; ε recomputed per batch per checkpoint. The integration test wires stored `ordinal` fields deliberately UNRELATED to positions and verifies captured (t, ε) against POSITIONS after reordering, rebatching, and two checkpoint restores — pinning that the evaluator indexes by position, never the stored field.

**D2 — Exactly-once coverage + padding exclusion (T1).** Reader = the tested `_iter_parsed_records` seam (file-ordered, no shuffle/repeat) consuming the val shards once; each record tagged with its position. Batches of B=32 (per-device 4 × 8 chips) assembled in position order; the final partial batch (14,636 = 457×32 + 12) is padded by repeating the last record with a **validity mask**; per-example losses come back to host and aggregation counts ONLY valid entries. Hard assertions (←F2): (a) the reader is DRAINED TO EOF and the total record count asserted == `validation_expected_count` BEFORE any checkpoint evaluation (catches more-than-expected, not just fewer); (b) positions enumerated 0…N−1; (c) aggregation counts only mask-valid entries, and a unit test compares the padded-tail mean/stderr against an unpadded golden aggregate (proves the duplicate's loss is excluded, not merely uncounted). Rung 3 performs a FULL-dataset scan (every shard): stored-ordinal contiguity + record-name uniqueness — dataset-level duplicate guard. Tests cover fewer-than-expected AND more-than-expected → hard failure.

**D3 — Objective parity (T1).** Loss math via the SAME shared code as training: `build_noisy_pinned_latents` (pin) → one plain transformer call (null context broadcast, activations dtype; NO actions/adapter/CFG) → target `ε − z_video` → frame-0-masked MSE. New pure helper `masked_velocity_mse_per_example(v_pred, v_target) -> [B]` (needed for stderr), **characterized against `masked_velocity_mse`** (←F3): each vector element == the scalar helper on its B=1 slice (bitwise — same reduction tree); the vector MEAN vs the batch scalar uses tight `allclose` (rtol 1e-6) since XLA reduction-tree equality is not guaranteed across the two paths. The scalar training helper is NOT touched. stderr = sample std of the 14,636 per-example losses / √N.

**D4 — Checkpoint loop in ONE job (T1).** Sequential over the 8 steps inside a single v6e-8 process: build state once (reusing the tested `_build_full_ft_validation_state` + `_restore_checkpoint_state(cohort_mode=True)`), stream the 8 passes from a host-RAM record cache (~3.4 GB, loaded once). Restore-step mechanism (←F4): `_restore_checkpoint_state` gains an optional, backward-compatible `requested_step: int | None = None` kwarg (None → existing config-driven behavior byte-identical; tests pin both paths) — pyconfig objects are immutable, so per-iteration `config.checkpoint_step` mutation is impossible. Loop discipline: block on the final batch's result before the next restore; assert BOTH the returned step and `state.step` equal the requested checkpoint. Resource note: each restore materializes fresh params+opt_state (~28 GB global, ~3.5 GB/chip transient on 8 chips) before releasing the old — counted in the fit estimate. T1 deletes the pipeline VAE/vae_cache/text-encoder/tokenizer right after state+null-context construction (no rollouts in T1; ←F5) — pinned by a structure test. Sequential-vs-parallel: 8 parallel jobs would pay 8× (queue slot + setup + model load ≈ 30 min) to save ~40 min of eval — strictly worse under the queue contention we've measured; sequential also guarantees identical data/RNG trivially.

**D5 — Outputs (T1).** `{output_dir}/{run_name}/validation_loss/`: `val_loss.json` (rows: step, mean_loss, stderr, n, seed, dataset path, commit, eval code SHA, timestamp), `val_loss.csv` (same rows), `val_loss_plot.png` (single-series checkpoint-vs-loss). Column schema, identical in JSON and CSV (←F6): `checkpoint_step, mean_loss, stderr (ddof=1), n, validation_seed, dataset_path, checkpoint_path, train_commit, eval_commit`. Plot: attempted on-worker (matplotlib import guard); if absent, a MANDATORY recorded post-step (in `_command.md`) regenerates the PNG locally via the module's `plot-only` mode and uploads it to the declared path — T1 is NOT accepted until the PNG exists at `validation_loss/val_loss_plot.png`.

**D6 — T2 reuses Part-I machinery unchanged.** `validate_wan_full_ft.sh` with `EVAL_DATA_DIR=<val split>`, `VALIDATION_ORDINALS="0,2927,5854,8781,11708,14635"`, `CHECKPOINT_STEP=20000`, `NUM_EVAL_VIDEOS=6`, `VALIDATION_SEED=0`, and `VALIDATION_OUTPUT_DIR={output_dir}/{run_name}/validation_valset` (separate root — prevents any collision with the Part-I train-cohort outputs at `validation/step_020000`). Zero evaluator-code changes for T2.

**D7 — Gallery (T2).** New CPU-only `src/maxdiffusion/make_wan_val_gallery.py`: input = a locally pulled `step_020000` directory; reads each sample's `metrics.json` + the three MP4s; emits `gallery.html` (self-contained styling, relative video paths, per-sample ordinal + latent/pixel/SSIM, run/checkpoint/commit header, and the REQUIRED provenance statement: ground truth is the VAE decode of cached `z_video`, not the original DROID RGB). Actionable errors on missing samples/files; sample order = the config.json `validation_ordinals` order.

## II.3 Planned code, per file

- **`src/maxdiffusion/models/wan/side_adapter_wan.py`** (+~15): `masked_velocity_mse_per_example` beside the existing helpers (same mask/normalization source), shape-mismatch raise matching the scalar helper.
- **`src/maxdiffusion/eval_wan_full_ft_val_loss.py`** (NEW, ~220): pure functions `per_example_rng(seed, position, num_steps, shape)` (D1), `plan_batches(n, batch)` (positions + validity), `aggregate(losses, validity, expected)` (mean/stderr/count + assertions), `write_outputs(...)` (JSON/CSV/plot fn); jitted eval step (stub-testable); main = config → state build → per-checkpoint restore loop → outputs. `FULL_FT_TI2V`-only guard; asserts guide 1.0 + fresh noise keys as the trainer does.
- **`src/maxdiffusion/configs/base_wan_5b_full_ft.yml`** (+3 keys): `validation_checkpoint_steps: ''`, `validation_expected_count: 0`, `validation_loss_output_dir: ''` (all CLI-overridable; evaluator requires the first two non-empty/positive).
- **`bash_scripts/eval_wan_full_ft_val_loss.sh`** (NEW, from the validate wrapper template): env knobs RUN_NAME (required), CHECKPOINT_STEPS, EXPECTED_COUNT, EVAL_DATA_DIR (default val split), VALIDATION_SEED, PER_DEVICE_BATCH_SIZE (default 4).
- **`src/maxdiffusion/make_wan_val_gallery.py`** (NEW, ~140): D7 with the F7 contracts — joins `sample_index` to the ORDERED `config.json["validation_ordinals"]` list and labels that value **dataset position** (stored `ordinal` shown separately); pins the exact artifact keys (`latent_mse`, `pixel_mse`, `ssim_avg`; summary `num_samples`); per-missing-file actionable errors (each MP4 + metrics.json individually); output = `gallery.html` inside the pulled step directory (relative refs only), copied under the exp folder's `_results_assets/` for the final report per SOP artifact 12.
- **`src/maxdiffusion/generate_wan_side_adapter.py`** (+~6, ←F4): the `requested_step` kwarg on `_restore_checkpoint_state` (default-None path byte-identical; adapter behavior untouched).
- **Tests** (`src/maxdiffusion/tests/worklogs_yixun/`): `test_full_ft_overfit_val_loss_core.py` (per-example helper characterization incl. bitwise mean-equality + mismatch raise; RNG determinism/independence: same (seed,p) → identical (t,ε) regardless of call order/batch; distinct p → distinct draws; t within [0,25)); `test_full_ft_overfit_val_loss_evaluator.py` (stub-transformer end-to-end on a fake 37-record/„expected 37" dataset via the `_iter_parsed_records` monkeypatch seam: exactly-once coverage, padded-tail exclusion (n stays 37 with B=8), count-mismatch hard failure, cross-"checkpoint" (t,ε) identity via two stub restores, JSON/CSV/plot writer contents incl. commit field); `test_full_ft_overfit_val_gallery.py` (fake step dir → 6 entries, ordinals, metrics, provenance sentence present; missing metrics → actionable error).

## II.4 Coder rounds (closed cycles)

A `val-loss-core` — per-example helper + RNG/batching/aggregation pure functions + core tests. B `val-loss-evaluator` — evaluator script + yml keys + wrapper + integration tests. C `val-gallery` — gallery generator + tests. Each: write (test-first) → briefed Codex review (`…_codex_code_<marker>_review.md`) → strengthen → commit.

## II.5 Validation ladder + launch gate

Rung 1: suite + py_compile + yaml + bash -n. Rung 2: stub-transformer integration test. Rung 3 (←F2): FULL val-split scan — every shard, schema + stored-ordinal contiguity + record-name uniqueness + TOTAL COUNT == 14,636 (needs gcloud reauth). Rungs 5–6 (←F5): a dedicated storage-light **T1 smoke/fit job** precedes the full pass — SMOKE env on the wrapper limits to ONE checkpoint and ~4 real batches at the production B=32, isolated output subdir `validation_loss_smoke/`; passing it is the fit evidence for the 8-checkpoint job. Rung 7 = the two production jobs. All jobs in the pre-launch package; **no launch without Yixun's approval** (announcement 02; Query 8 explicitly reserves it).

## II.6 Acceptance criteria (draft — finalized in the package)

T1 job: COMMIT match; 8/8 checkpoints restored at their exact steps (log line each); `n == 14636` assertion passes per checkpoint; JSON/CSV/PNG present under `validation_loss/`; monotonicity NOT assumed — whatever the curve is, it's the result. T2 job: COMMIT match; step-20000 restore; 6/6 samples at the exact positions in listed order; videos + metrics per sample; summary.json n=6. Gallery: opens from disk, 6 entries, provenance statement, no broken video refs.
