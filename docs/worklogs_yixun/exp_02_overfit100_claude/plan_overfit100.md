# plan_overfit100 — exp_02: text-conditioned 100-trajectory memorization test (v2)

Planner: Claude Fable 5 (v1 drafted under Opus 5 max — session-model deviation recorded in `_worklog.md`; v2 by Fable 5 xhigh). Status: **v2 after Codex plan review** (`overfit100_codex_plan_review.md`, REQUEST-REVISION, F1–F7 — resolutions appended there). Data path **A′** per Query 3. Base: `claude-exp_02_overfit100-20260728` (includes exp_01's reviewed code by branch merge; `yixun-dev` untouched per exp_01 Query 10). v1 is in git history (`cb5d73f`).

## 1. Objective — a finite-set memorization test (F1)

**Question (Yixun/Lihan):** find a case where the model can (near-)perfectly reconstruct its training samples. exp_01 could not answer this: full-corpus scale, and its no-text/no-action conditioning left the future under-determined.

**exp_02 asks, precisely:** can the fully-unfrozen Wan2.2 TI2V 5B, trained on the windows of **100 successful DROID trajectories** conditioned on (per-trajectory **language instruction** + **first-frame latent**), memorize that finite set — i.e. reproduce the recorded futures under 25-step rollout on the training windows themselves?

**What is NOT claimed (review F1):** text + first frame does *not* determine the recorded future in any causal sense — an instruction fixes the goal, not motion timing or path. Success here means the weights can recite a finite mapping whose inputs are (in practice) unique; it may lean on first-frame fingerprints more than language. Failure does **not** imply a broken pipeline — it triggers §7 escalation, not a verdict. exp_01's 0.787 is not treated as an established "conditioning ceiling."

**Text-ablation controls (F1):** at gate checkpoints, the eval cohort is rolled out under three context modes — **correct** instruction, **null** (empty prompt, exp_01's context), and **shuffled** (seeded derangement of instruction→episode). The correct-vs-shuffled/null gaps measure how much the model actually uses language. Reported always; no pass/fail threshold attached.

**Duplicate-condition audit (F1):** the build report lists (a) exact-duplicate instructions among the 100, (b) the minimum pairwise `z_i0` L2 distance across all window pairs with different targets (same- and cross-episode), so "inputs are effectively unique" is checked, not assumed.

## 2. Data — path A′ (F4, F6; Queries 2–3)

**Source (all GCS):** `gs://v6_east1d/datasets/droid_ctrl_world_aligned/` — `annotation/train/<ep>.json` (69,723 eps, ids 0–69722; `texts[1..3]`, `success`, `video_length`) and `videos/train/<ep>/0.mp4` (**probed: already 320×192 @ 5 fps, yuv420p; frame counts match annotations** — no resize/crop step). The `latent_videos/*.pt` are Ctrl-World-space `(T,4,24,40)` — **not used** (probe-verified wrong latent space).

**D1 — Selection (unchanged from v1):** seeded draw (`selection_seed=0`) without replacement over 0–69722; accept iff `success==1` ∧ ≥1 non-empty `texts` ∧ `videos/train/<ep>/0.mp4` exists ∧ `nb_frames ≥ 33`. Stop at 100. Rejections tallied by reason.

**D2 — Instruction (unchanged):** among **non-empty** texts, pick one by `fold_in(selection_seed, episode_id)` — stable, order-independent; empties filtered before the pick (Query 1). All candidates + the pick recorded.

**D3 — View:** index 0 only (Query 2).

**D4 — Windows & encoding (replaces v1's dead .pt path):** per episode, windows = 33 **consecutive** frames starting at `s = 0, 4, 8, …` while `s+33 ≤ nb_frames` (start-stride 4, matching the cache's `candidate_stride`). Each window is **VAE-encoded independently** with maxdiffusion's own Flax Wan VAE (the same weights snapshot the pipeline decodes with) → `z_video [48, 9, 12, 20]` f16; **`z_i0 := z_video[:, 0:1]`** — this slice contract is now *empirically proven*: in the exp_01 cache `z_i0 == z_video[:,0]` bit-identically (max|diff| 0.0000). Per-window encode = exactly what rollout-time decode assumes; no trajectory-level encode-then-slice ambiguity. ep0: 14 windows; ep1: 24; expected total ≈ **1.0–2.0k windows** (exact count fixed at build, asserted at train).

**Encoder validation (F4), part of the build job:** (i) stats gate per window (std/min/max within the cache's observed envelope, e.g. std 0.4–0.9); (ii) round-trip: decode a cached exp_01 `z_video` → re-encode with our encoder → agreement within a documented tolerance (validates our encode against the cache's encoder up to VAE round-trip error); (iii) decode(our `z_video`) vs source frames SSIM ≥ 0.8 per sampled window — and this decode-vs-RGB number doubles as the **per-window VAE ceiling** used in evaluation (F3). Any gate failing ⇒ build aborts loudly.

**D5 — Manifest is the reproducibility root (F6):** selection runs once, locally, producing `overfit100_manifest.json` — per episode: `episode_id`, chosen text + index, all texts, and the MP4's **GCS generation, size, and md5** (from `gsutil stat`), plus ffprobe geometry. The manifest is **committed before any encoding**; the build job consumes the manifest only (never re-runs selection), verifies each object's generation/md5 before use, and **fails on any fingerprint drift** (the corpus was observed mid-upload on 2026-07-28). Seed alone is explicitly *not* the reproducibility claim.

**D6 — Built artifact:** `gs://v6_east1d/datasets/exp02_overfit100/train/` TFRecords — `name` (`ep<ID>_v0_s<START>`), `episode_id`, `episode_index` (0–99), `window_start`, `z_i0` f16, `z_video` f16, `instruction` bytes. **No `actions` field** — schema v2; every reader is updated accordingly (F2). Sidecars: `episodes.json` (index→id, texts, window count), `summary.json` (counts, seeds, rejection tally, build commit, encoder-validation numbers), duplicate-audit report. ~1–2 GB total.

## 3. Model / training (F5, F7)

**D7 — Trainer:** new `model_type: OVERFIT100_TI2V` → `WanTI2VOverfit100Trainer(WanTI2VFullFTTrainer)`. Overrides, exhaustively: (a) TFRecord parse for schema v2; (b) **`_data_shardings()` extended for the new batch keys** (`episode_index` — the review's JIT-tree-mismatch catch); (c) startup text-table build; (d) `_denoising_loss` context = table gathered by `episode_index`. Objective math untouched — shared `build_noisy_pinned_latents`/`masked_velocity_mse`, frame-0 pin, target `ε − z_video`, fresh noise, no actions/adapter/CFG, guide-scale assert.

**D8 — Text table (F5):** encode the 100 **positive** prompts only, in a **bounded minibatch loop** (`text_encode_batch: 8`), replicating `encode_prompt`'s positive branch (tokenize→512, attention mask through UMT5, truncate to true length, zero-pad) — with a **parity test** vs `encode_prompt` on a small fixture, and a loop-vs-batch/order-invariance test. **No new attention mask** — the TI2V forward passes `encoder_attention_mask=None`; the zero-padded contract is preserved as-is. T5 freed after, as in exp_01. Table `[100, 512, 4096]` bf16 = **400 MiB, device-replicated** — v1's "HBM story unchanged" was false and is retracted; the table's physical bytes go into the startup memory audit and **the fit probe is authoritative**. Fallback if pd4 no longer fits: host-side per-batch gather (context enters as a `[B,512,4096]` batch tensor), pre-planned, not improvised.

**D9 — Recipe:** LR **1e-5** (Query 2, locked), AdamW/clip as exp_01; **warmup 250 steps** (deliberate delta from exp_01's 1000 — at 10-episode scale 1000 warmup steps would be ~40% of the first segment; logged as a dose-schedule change, objective untouched). bf16 weights/activations, remat FULL, fresh noise, guide 1.0.

**D10 — Staged compute (F7):** no blind 10k run.
- **S1 — smoke** (v6e-8, GBS 32 = pd 4, ~20 steps, storage-light): schema/table/step-time asserts; also the pd4-with-table fit check on 8 chips.
- **S2 — 10-episode sanity gate** (v6e-8, GBS 32): train on the manifest's first 10 episodes (~150–250 windows), 2,500 steps ≈ 320–530 epochs, checkpoints at 250/500/1000/2500; rollout eval (D11) at those checkpoints on 1 canonical window/episode ×3 noise seeds + text-ablation at step 2500. **Gate:** clear monotone reconstruction improvement with best-window SSIM approaching ≥0.9 ⇒ proceed; flat/broken ⇒ stop and analyze — the cheap negative.
- **S3 — 100-episode run** (v6e-64, GBS 256 = pd 4, ≈5–13 steps/epoch): **first segment 2,500 steps** with checkpoints at 250/500/1000/1750/2500 (keep all), eval per D11; **resume toward 10k only on evidence**, as separately-approved segments (Orbax resume = exp_01's proven path).

**D11 — Evaluation (F2, F3):** all rollouts 25-step, seed-controlled, on **training windows** (this is the point).
- **Coverage:** intermediate checkpoints → 1 canonical window per episode (median window: `start = 4·⌊(n_windows−1)/2⌋`) = 100 windows (10 in S2), 1 seed at intermediates, **3 seeds + the 3 text modes at gate/final checkpoints**; final candidate checkpoint → **ALL built windows**, 1 seed, correct text.
- **Metrics:** primary = SSIM and latent/pixel MSE of prediction vs **VAE-decode of the stored target `z_video`**; auxiliary = prediction vs the true RGB frames `[s, s+33)` of the source MP4, reported **alongside the per-window VAE ceiling** (decode(target) vs RGB) so the auxiliary number is interpretable (F3).
- **Predeclared aggregates:** mean, median, P10, min, and fraction of windows with SSIM ≥ 0.90 and ≥ 0.95. **Success rule:** memorization established iff **≥ 90% of canonical windows reach SSIM ≥ 0.95** (correct text, best checkpoint, median over the 3 seeds); **partial** if ≥ 90% reach ≥ 0.90; otherwise → §7. Latent-MSE has no pass/fail role (v1's "≤ ~0.05" dropped as arbitrary); it is reported with the same aggregates.
- **One-step instrument:** exp_01's val-loss evaluator core, retargeted (schema v2 + per-example context) over **all** windows at every kept checkpoint — the low-noise loss curve behind the rollout numbers.

## 4. Planned code (per file — F2's tooling gap closed explicitly)

- **NEW** `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py` (~150 LOC, local CPU): D1/D2 selection + `gsutil stat` fingerprints → `overfit100_manifest.json`; `--dry-run` prints selection.
- **NEW** `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py` (~250 LOC, runs on v6e-8): manifest → fingerprint verify → frames → per-window Wan-VAE encode → encoder-validation gates → TFRecords + sidecars + duplicate audit.
- **NEW** `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py` (~200 LOC): D7 overrides incl. `_data_shardings`.
- **EDIT** `src/maxdiffusion/train_wan.py` (+3): dispatch.
- **NEW** `src/maxdiffusion/configs/base_wan_5b_overfit100.yml`: from full-FT yml; `model_type`, dataset dirs, steps/warmup/checkpoint knobs, `num_text_slots: 100`, `text_encode_batch: 8`.
- **EDIT** `src/maxdiffusion/generate_wan_side_adapter.py`: OVERFIT100 branch — schema-v2 reader (no `actions`), per-example context with `context_mode ∈ {correct, null, shuffled}` (seeded derangement), full-FT restore path reuse, checkpoint round-trip covered by test; metrics vs decode-of-target + RGB-auxiliary + VAE ceiling.
- **EDIT** `src/maxdiffusion/eval_wan_full_ft_val_loss.py`: OVERFIT100 mode (schema v2, per-example context, model-type gate extended) — aggregation/RNG core reused unchanged.
- **NEW** bash arms: `build_overfit100_dataset.sh`, `train_wan_overfit100.sh`, `validate_wan_overfit100.sh` + launcher arm.
- **Tests** (`src/maxdiffusion/tests/worklogs_yixun/`, one file per unit): selection/instruction determinism + empty-filter + rejection accounting; manifest fingerprint-drift failure; window math (ep0→14, ep1→24, boundary cases); schema round-trip; text-table loop-vs-batch parity + gather-with-index≠id fixture + no-negative-encode + bytes audit; `_data_shardings` tree-match; context-mode unit tests (null ≡ exp_01 null path, shuffled = seeded derangement, correct = identity); objective parity vs exp_01 helpers on identical inputs.

## 5. Cycles (closed write→review→strengthen each)

**A** manifest builder + selection tests → run it → **commit the manifest itself**. **B** dataset builder + encoder-validation tests (rung-2 synthetic encode; rung-4 bounded build of 2 episodes + readback; then full build — TPU job, approval-gated). **C** trainer/config/dispatch/text-table. **D** eval tooling (generator modes + loss evaluator). **E** launches S1→S2→S3, each with acceptance criteria in `_worklog.md` + `_command.md` entries at launch time.

## 6. Validation ladder mapping

1 = suite + static; 2 = synthetic VAE-encode + stub-transformer forward with gathered context; 3 = real readback (TFRecord ↔ manifest ↔ decoded frames); 4 = bounded 2-episode build; 5 = S1 smoke; 6 = fit check (S1 doubles as v6e-8 probe; v6e-64 pd4+table confirmed at S3 start); 7 = S2 then S3.

## 7. Risks / escalation

**R1** encoder-validation gate fails (our encode ≁ cache encoder) → stop; reconcile VAE config before any training. **R2** S2 gate flat at 2,500 steps → analyze first (loss curve, one-step eval, ablations); candidate escalations, each separately approved: extend S2 to 10k; drop to 1 episode (the cleanest possible memorization bound); only then question the pipeline. LR stays 1e-5 unless Yixun changes it. **R3** table+pd4 OOM → D8 fallback (host gather). **R4** corpus changes under us → D5 fingerprints fail loudly. **R5** duplicate/near-duplicate conditions found by the audit → report before launch; if two windows share near-identical (text, z_i0) with different targets, perfect reconstruction of both is impossible and the success rule's denominator notes them.
