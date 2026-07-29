# plan_overfit100 — exp_02: text-conditioned 100-trajectory memorization test (v4)

Planner: Claude Fable 5 (xhigh; v1 under Opus 5 max — deviation logged). Status: **v4 after third Codex review** (v1 → F1–F7, v2 → G1–G5, v3 → H1–H2 + G2/G4 residue; reviews + resolutions in `overfit100_codex_plan_review.md` / `…plan2_review.md` / `…plan3_review.md`). Data path **A′** (Query 3). Base: `claude-exp_02_overfit100-20260728` (exp_01 code merged in; `yixun-dev` untouched). v1 @ `cb5d73f`, v2 @ `092eb91`, v3 @ `42c9057`.

## 1. Objective — a finite-set memorization test (F1: RESOLVED)

**Question (Yixun/Lihan):** find a case where the model can (near-)perfectly reconstruct its training samples. exp_01 could not answer this: full-corpus scale, no-text/no-action conditioning left the future under-determined.

**exp_02 asks:** can fully-unfrozen Wan2.2 TI2V 5B, trained on the windows of **100 successful DROID trajectories** conditioned on (per-trajectory **language instruction** + **first-frame latent**), memorize that finite set — reproduce the recorded futures under 25-step rollout on the training windows themselves?

**Not claimed:** text + first frame does not determine the recorded future causally; success = the weights recite a finite mapping whose inputs are (audited-)unique, possibly leaning on first-frame fingerprints more than language; failure ≠ broken pipeline (→ §7); exp_01's 0.787 is not a "conditioning ceiling."

**Text-ablation controls:** at gate checkpoints the eval cohort rolls out under **correct** / **null** (empty prompt) / **shuffled** (seeded derangement) context. Gaps reported always; no pass/fail attached.

**Duplicate-condition audit:** build report lists exact-duplicate instructions and min pairwise `z_i0` L2 distance among window pairs with different targets. **The success-rule denominator is fixed at build time** — collided windows are flagged in the pre-launch report and stay in the denominator; they may be discussed in analysis but never removed post hoc (G4).

## 2. Data — path A′ (F4/G2, F6/G5)

**Source (all GCS):** `gs://v6_east1d/datasets/droid_ctrl_world_aligned/` — `annotation/train/<ep>.json` (69,723 eps, 0–69722; `texts[1..3]`, `success`), `videos/train/<ep>/0.mp4` (probed: 320×192 @ 5 fps, yuv420p; frame counts match annotations — no resize). `latent_videos/*.pt` are Ctrl-World-space `(T,4,24,40)` — **not used**.

**D1 — Selection:** seeded draw (`selection_seed=0`) without replacement over 0–69722; accept iff `success==1` ∧ ≥1 non-empty `texts` ∧ view-0 MP4 exists ∧ `nb_frames ≥ 33`. Stop at 100.

**D2 — Instruction:** among non-empty texts, pick by `fold_in(selection_seed, episode_id)`; empties filtered before the pick (Query 1). Pick + all candidates recorded.

**D3 — View:** index 0 only (Query 2).

**D4 — Windows & encoding (G2: exact contract).** Windows = 33 consecutive frames at starts `s = 0, 4, 8, …`, `s+33 ≤ nb_frames`; `1+(33−1)/4 = 9` latent frames. **Encode contract, locked:** replicate `wan_pipeline.py`'s video-encode path verbatim — frames decoded via ffmpeg to RGB, normalized and laid out exactly as that path's preprocessing (`[B, C, T, H, W]`, the pipeline's value range), `vae.encode(x, vae_cache)[0]`**`.mode()`** (deterministic posterior mode — **no sampling, no RNG**), then the pipeline's `latents_mean/latents_std` normalization; stored as f16. `z_i0 := z_video[:, 0:1]` (cache contract proven bit-identical). **VAE fingerprint** (HF snapshot revision + VAE config hash) recorded in the build summary; the build asserts it matches the manifest's pin. Per-window independent encode (matches rollout-time decode; no trajectory-encode-then-slice).

**Encoder-validation gates (G2: exact, fixed sample sets, all thresholds final):**
- **V1 — round-trip vs cache** on three cached reference windows (`ep0_v0_s00000/s00004/s00008`): decode cached `z_video` → re-encode via our path → require relative L2 `‖ẑ−z‖₂/‖z‖₂ ≤ 0.25` **and** Pearson r ≥ 0.97 per window. Catches normalization/layout errors (a missed mean/std shows as ~16× std blowup) while tolerating VAE round-trip loss. **Fixture materialization (H1):** cycle A extracts these three records from `droid_wan_side_adapter/train/train-00000-of-00704.tfrecord` (name-verified), saves `z_i0`/`z_video` + names/shapes/dtypes as one `.npz`, uploads to `gs://v6_east1d/datasets/exp02_overfit100/fixtures/v1_cache_windows.npz`, and records its **GCS generation/md5/size in the committed manifest**; the build job's **preflight reads the fixture and verifies md5 + names before any encoding** — V1 is not executable until this preflight passes.
- **V2 — stats envelope**, every built window: finite; std ∈ [0.35, 0.95]; |mean| ≤ 0.15 (cache-observed: std 0.60–0.70).
- **V3 — decode-vs-RGB**: SSIM(decode(our `z_video`), source frames) ≥ 0.80 on the fixed set {first window of manifest episodes at index 0, 10, 20, …, 90}. These numbers double as the per-window **VAE ceiling** at eval.
- **V4 — frame-0 future-invariance** (reviewer's ask): for one fixed window (manifest ep index 0, s=0), encode frames `[0,33)` and `[0,17)` (→5 latent frames); latent frame 0 must agree within rtol 1e-3 — proves no future leakage into `z_i0` through the causal VAE.
- **2-episode probe** (rung 4) records peak HBM and windows/sec; full-build cost extrapolated and logged in `_worklog.md` **before** the full build runs. Any gate failing ⇒ build aborts loudly.

**D5 — Manifest (F6/G5: complete provenance).** `build_overfit100_manifest.py` runs once, locally → committed `overfit100_manifest.json`: per accepted episode — `episode_id`, chosen text + index, all texts, **GCS generation/md5/size for BOTH the annotation JSON and the MP4**, ffprobe geometry; plus the **ordered draw log** (every drawn id, accept/reject + reason), the builder commit, and tool versions (ffprobe/ffmpeg, gsutil, python). The build job consumes the manifest only, verifies every fingerprint before use, **fails on drift**. Seed-only reproducibility claim withdrawn.

**D6 — Built artifacts (G3): TWO fingerprinted sets from the same manifest+build job:** `gs://v6_east1d/datasets/exp02_overfit100/train100/` (all 100 episodes) and `…/train10/` (manifest episode_index 0–9), each with own TFRecords + `summary.json` (incl. expected window count). Schema v2 per record: `name`, `episode_id`, `episode_index`, `window_start`, `z_i0` f16, `z_video` f16, `instruction` bytes — **no `actions`**. The trainer asserts `expected_windows` (config) against records seen. Sidecars: `episodes.json`, duplicate-audit report.

## 3. Model / training

**D7 — Trainer architecture (G1: matches the real seams).** The full-FT trainer's step functions are **module-level** and jit-bound inside `start_training` — subclass *methods* cannot replace them. Plan accordingly: new module `wan_ti2v_overfit100_trainer.py` containing —
- `Overfit100TrainState(train_state.TrainState)` with `context_table: jax.Array` `[100, L, 4096]` (replaces `null_context`);
- module-level `_denoising_loss` — identical math to full-FT's (shared helpers, frame-0 pin, `ε − z_video`, fresh noise, no actions/adapter/CFG) except context = `state.context_table[data["episode_index"]]` (batched gather) instead of null-broadcast;
- module-level `_train_step` / `_eval_step` binding that loss;
- `WanTI2VOverfit100Trainer(WanTI2VFullFTTrainer)` overriding the **genuine seams**: `_load_dataset()` (schema-v2 parse — the parse fn is nested there, so the override owns it), `_data_shardings()` (adds `episode_index`), and **`start_training()` rewritten** (structured copy of the full-FT one) to build the text table, construct `Overfit100TrainState`, and jit-bind **this module's** step functions.
- **Checkpoint-schedule contract (H2):** the full-FT loop saves only at one periodic cadence and the inherited manager hard-codes `max_to_keep=3` — neither can produce/retain D10's lists. The rewritten `start_training()` saves **when `(step+1) ∈ checkpoint_steps`** (explicit list in the config: S2 `[250,500,1000,2500]`, S3 `[250,500,1000,1750,2500]` + later segment finals) and constructs its **own CheckpointManager with `max_to_keep=None`** so every listed checkpoint is retained, including across resumes (segment-final checkpoints are never garbage-collected). Storage budget, predeclared: full `FullFTTrainState` ≈ **30 GB/checkpoint** (bf16 params + both Adam moments) → S2 ≈ 120 GB, S3 first segment ≈ 150 GB on GCS — acceptable, monitored in `_worklog.md`. **Test:** a fake-loop unit test asserts the exact emitted **and retained** step set for both configs.
- **Test (G1):** with a stub transformer, a batch containing two different `episode_index` values must receive **row-distinct** context tensors (fixture where index ≠ episode_id), plus an objective-parity test vs the full-FT loss when the table row equals the null embedding.

**D8 — Text table (F5: resolved contract kept).** Encode the 100 positive prompts only, bounded loop (`text_encode_batch: 8`), replicating `encode_prompt`'s positive branch (tokenize→512, mask through UMT5, truncate, zero-pad); parity + order-invariance tests; **no new attention mask** (`encoder_attention_mask=None` preserved); T5 freed after. Table 400 MiB bf16 replicated; bytes in the startup memory audit; fit probe authoritative; pre-planned fallback = host-side per-batch gather.

**D9 — Recipe:** LR **1e-5** (locked), AdamW/clip as exp_01, **warmup 250** (logged dose-schedule delta), bf16, remat FULL, fresh noise, guide 1.0.

**D10 — Staged compute.**
- **S1 — smoke** (v6e-8, GBS 32 = pd4, ~20 steps on `train10`, storage-light): schema/table/gather asserts + fit check.
- **S2 — 10-episode gate** (v6e-8, GBS 32, `train10`, 2,500 steps ≈ 320–530 epochs; checkpoints 250/500/1000/2500). Eval per D11 on the 10 canonical windows, 3 seeds each checkpoint (cheap at this scale), text-ablation at 2500. **Numerical gate (G3), all predeclared:** let `m(w,c)` = median-over-3-seeds SSIM. **PROCEED to S3 iff** (i) cohort mean of `m(w, 2500)` ≥ **0.70**, (ii) cohort mean rises from c=250 to c=2500 by ≥ **0.15**, (iii) `max_w m(w, 2500)` ≥ **0.85**. Any miss ⇒ stop & analyze (§7) — no S3 launch.
- **S3 — 100-episode run** (v6e-64, GBS 256 = pd4, `train100`): first segment 2,500 steps, checkpoints 250/500/1000/1750/2500 (keep all); extension toward 10k only on evidence, as separately-approved resumable segments.

**D11 — Evaluation (F3/G4: executable statistic).**
- **Coverage (one matrix, I1):** canonical windows = median window per episode (`start = 4·⌊(n_w−1)/2⌋`). **S2** (10-episode cohort, gate-only): 3 fixed seeds, **correct mode**, at every checkpoint {250,500,1000,2500}; **null/shuffled only at 2500**. **S3** (100-window cohort): intermediate checkpoints → 1 seed, correct mode; segment-final checkpoints → 3 fixed seeds × 3 modes. Final candidate checkpoint → **all** built windows, 1 seed, correct mode.
- **Metrics:** primary = SSIM / latent MSE / pixel MSE vs **VAE-decode of stored `z_video`**; auxiliary = vs true RGB frames, always paired with the per-window VAE ceiling (V3).
- **Success rule (exact; G4 residue closed — S3-only, correct-mode-only):** define **`C₃¹⁰⁰` = the S3 run's segment-final checkpoints** (2500; +5000/7500/10000 if extended) — S2 checkpoints never enter this statistic. For `c ∈ C₃¹⁰⁰` and canonical window `w` (denominator = the 100 canonical windows, fixed at build; collisions flagged, never dropped): **`m_corr(w,c) = median_{seed∈{0,1,2}} SSIM(w,c,seed | context_mode=correct)`** — ablation modes (null/shuffled) are reported context, never inputs to the statistic. **Headline claim — "canonical-window memorization": established iff `max_{c∈C₃¹⁰⁰} fraction{w : m_corr(w,c) ≥ 0.95} ≥ 0.90`**; partial at the 0.90 threshold. Best checkpoint `c*` = argmax of that fraction, ties broken by higher mean `m_corr(·,c)`, then by **earlier** step (deterministic). **Stronger claim — "full-set memorization": additionally requires `fraction{SSIM(w,c*,seed 0 | correct) ≥ 0.90} ≥ 0.90` over ALL built windows.** Without it, only the canonical-window claim is ever made (the v3 "0.75 guard" is replaced by this two-tier claim structure — the headline claim is always scoped to what was measured). **The evaluator writes the aggregation artifact** (per-window/per-seed/per-mode SSIM, `m_corr`, fractions, `c*`, verdict inputs) as JSON — no hand computation.
- **One-step instrument:** the val-loss evaluator core retargeted (schema v2, per-example context) over all windows at every kept checkpoint.

## 4. Planned code (per file)

- **NEW** `data_preprocessing/build_overfit100_manifest.py` (~180 LOC, local): D1/D2 + dual fingerprints + ordered draw log; `--dry-run`.
- **NEW** `data_preprocessing/extract_v1_fixture.py` (~80 LOC, local): pulls the three named cache records → `.npz` → GCS fixture; its generation/md5 lands in the manifest (H1).
- **NEW** `data_preprocessing/build_overfit100_dataset.py` (~300 LOC, v6e-8): manifest verify → ffmpeg frames → pipeline-parity encode (D4) → gates V1–V4 → `train100` + `train10` TFRecords + sidecars + audit.
- **NEW** `trainers/wan_ti2v_overfit100_trainer.py` (~320 LOC): D7 as specified (own state/loss/steps + rewritten `start_training` + seam overrides).
- **EDIT** `train_wan.py` (+3): dispatch `OVERFIT100_TI2V`.
- **NEW** `configs/base_wan_5b_overfit100.yml`: model_type, dataset dirs (train100/train10 switch), steps/warmup/ckpt knobs, `num_text_slots`, `text_encode_batch`, `expected_windows`.
- **EDIT** `generate_wan_side_adapter.py`: OVERFIT100 branch — schema-v2 reader, per-example context, `context_mode ∈ {correct,null,shuffled}` (seeded derangement), full-FT restore reuse, **aggregation-artifact writer** (G4), metrics incl. VAE ceiling.
- **EDIT** `eval_wan_full_ft_val_loss.py`: OVERFIT100 mode (schema v2, per-example context, model-type gate extended); aggregation core unchanged.
- **NEW** bash arms: build / train / validate + launcher arm.
- **Tests:** selection determinism + empty-filter + draw-log; manifest fingerprint-drift fail; **V1-fixture extraction + preflight md5/name verification (H1)**; window math (ep0→14, ep1→24, edges); schema round-trip; **train10 filter + count assert (G3)**; text-table parity/order/no-negatives/bytes; `_data_shardings` tree-match; **row-distinct gather with index≠id fixture (G1)**; objective parity vs full-FT on null-equal table row; context modes (null ≡ exp_01 path, shuffled = seeded derangement); gate V1–V4 threshold logic on fixtures; **checkpoint-schedule emit/retain set for both configs (H2)**; **success-statistic function (`m_corr`, fractions, `c*` tie-break, two-tier rule) as a pure tested unit (G4)**.

## 5. Cycles

**A** manifest builder + tests → run → **commit manifest**. **B** dataset builder + gate tests → rung-4 2-episode probe build (cost extrapolation logged) → full build (TPU, approval-gated). **C** trainer/config/dispatch/table. **D** eval tooling + success-statistic unit. **E** S1 → S2 (gate) → S3, each with acceptance criteria + `_command.md` entries at launch.

## 6. Validation ladder mapping

1 suite/static; 2 synthetic encode + stub-transformer gather forward; 3 readback (TFRecord ↔ manifest ↔ decoded frames); 4 two-episode bounded build + V1–V4; 5 S1; 6 fit checks (S1 on v6e-8; v6e-64 pd4+table at S3 start); 7 S2 → S3.

## 7. Risks / escalation

**R1** V1–V4 gate failure → stop, reconcile VAE config; no training. **R2** S2 gate miss → analyze (loss curve, one-step eval, ablations); separately-approved escalations: extend S2 to 10k; 1-episode bound; only then question the pipeline. LR stays 1e-5 unless Yixun changes it. **R3** table+pd4 OOM → host-gather fallback. **R4** source drift → fingerprints fail loudly. **R5** condition collisions → flagged pre-launch, denominator unchanged. **R6** encode cost surprises → bounded by the rung-4 extrapolation gate before the full build.
