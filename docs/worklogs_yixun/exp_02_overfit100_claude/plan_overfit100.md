# plan_overfit100 — exp_02: text-conditioned 100-trajectory overfit

Planner: Claude Opus 5 (max effort — SOP names Fable 5; deviation recorded in `_worklog.md`, user-set session model). Status: **draft for Codex plan review**. Base: `claude-exp_02_overfit100-20260728` after merging exp_01's reviewed code (yixun-dev untouched, per Query 10).

## 1. Objective and why exp_01 doesn't answer it

Lihan/Yixun's critique (Query 1), accepted: exp_01 trained on the FULL DROID split (1.44M windows, 3.55 passes) with **no text and no actions**, so (a) the task was generalization-scale, not memorization-scale, and (b) generation was **under-determined** — a single first frame cannot say *why* the scene evolves, so many futures are consistent with the input and exact reconstruction is impossible in principle. exp_01's 0.787 train-clip SSIM is therefore plausibly near that setup's ceiling, and says little about pipeline capacity.

**exp_02 makes the task determined and small:** 100 successful DROID trajectories, each conditioned on **its own language instruction** (T5 context) **plus the first-frame latent**, full-weight Wan2.2-TI2V-5B fine-tune, evaluated **on the training samples themselves**. Success = (near-)perfect reconstruction of those samples, which is the evidence Yixun asked for ("find a case where the model can perfectly reconstruct the training samples").

## 2. Data (all GCS-resident; verified 2026-07-28 — no cluster dependency)

Source: `gs://v6_east1d/datasets/droid_ctrl_world_aligned/` — `annotation/train/<episode_id>.json` (69,723 episodes, ids 0–69722; fields `texts[1..3]`, `success 0/1`, `episode_id`, `video_length`, `videos[]`, `latent_videos[]`), `latent_videos/train/<ep>/<view>.pt` (~1.35 MB), `videos/train/<ep>/<view>.mp4`.

**D1 — Episode selection (100).** Seeded RNG (`selection_seed=0`) draws candidate episode ids from 0–69722 without replacement; each candidate is **accepted iff** `success == 1` **AND** it has ≥1 non-empty `texts` entry **AND** `latent_videos/train/<ep>/0.pt` exists **AND** its latent length yields ≥1 window. Draw until exactly **100** accepted. Rejections are counted by reason and recorded. Selection is reproducible from the seed alone; the accepted list is frozen into `overfit100_params_set_up.md` **before training**.

**D2 — Instruction per episode.** Among the episode's **non-empty** `texts` (the DROID 1-of-3 annotations), pick one with a per-episode deterministic key (`fold_in(selection_seed, episode_id)`), so the choice is stable and independent of draw order. Empty/whitespace-only strings are filtered *before* the pick (Query 1's explicit requirement). Chosen text + all candidates are recorded per episode.

**D3 — View.** **View index 0 only** (Query 2) — `latent_videos/train/<ep>/0.pt`.

**D4 — Windows.** Rung-3 readback first establishes the `.pt` layout; the plan assumes it decodes to `[48, T_lat, 12, 20]` fp16 latents matching the cached geometry (`z_video_shape [48,9,12,20]`, `z_I0_shape [48,1,12,20]`, 33 raw frames → 9 latent frames, raw stride 4 = 1 latent frame). Windows: all starts `0 … T_lat-9` at latent-stride 1; `z_video` = frames `[s, s+9)`, `z_i0` = frame `s` (first frame of that window, matching exp_01's pin semantics). Expected ≈ 10–25 windows/episode ⇒ **≈1–2.5k windows total** (exact count fixed at build time and asserted thereafter). If the layout differs, the build fails loudly and the plan is revised — no silent reinterpretation.

**D5 — Built artifact.** `gs://v6_east1d/datasets/exp02_overfit100/` : TFRecords with `z_i0` (f16), `z_video` (f16), `episode_id` (int64), `episode_index` (int64 0–99), `window_start` (int64), `name` (bytes `ep<ID>_v0_s<START>`), `instruction` (bytes — self-describing) + `episodes.json` (index → episode_id, instruction, all candidates, window count) + `summary.json` (counts, seed, rejection tally, build commit). Small enough (~1–3 GB) to rebuild at will.

## 3. Model / training

**D6 — Per-example text conditioning (the core change from exp_01).** exp_01 computed ONE null-prompt T5 embedding and broadcast it. exp_02: at startup, encode the ≤100 **unique** instructions with the pipeline's T5 (already loaded for the null prompt) into a table `[100, L, 4096]` (L = `wan_max_sequence_length` 512), then **gather per example by `episode_index`** to form `encoder_hidden_states` — after which the text encoder is freed exactly as exp_01 frees it. Everything else in the objective stays byte-identical to exp_01's reviewed path (shared `build_noisy_pinned_latents` / `masked_velocity_mse`, frame-0 pin, target `ε − z_video`, fresh noise, **no adapter, no actions, no CFG**, guide-scale assert).

**D7 — Recipe.** v6e-64, pure FSDP, per-device 4 ⇒ **GBS 256** (exp_01-proven fit; the HBM story is unchanged since only the context source differs), **LR 1e-5** (Query 2), AdamW/clip/warmup identical to exp_01, bf16 weights+activations, remat FULL, fresh noise, guide 1.0. **Steps: 10,000** ⇒ ≈1,250 epochs over ~2k windows (vs exp_01's 3.55 passes) — the actual memorization regime. Checkpoints every 2500 (keep-period 2500) at 2500/5000/7500/10000.

**D8 — Evaluation = the training samples.** Cohort: **16 fixed training windows** spanning 16 distinct episodes (deterministic spread over the 100, predeclared with seeds). At step-0 (pretrained baseline) and each checkpoint: 25-step rollout conditioned on (first-frame latent + that episode's instruction), reporting latent MSE / pixel MSE / SSIM, plus comparison and **residual** videos via exp_01's reviewed tooling. Additionally the full one-step training-set loss (exp_01's evaluator, retargeted at this dataset) at each checkpoint. **True RGB ground truth is available** here (`videos/train/<ep>/0.mp4`), so galleries can show real frames — no VAE-decode caveat.

**D9 — Success criterion (predeclared).** Primary: cohort **SSIM ≥ 0.95** with latent MSE ≤ ~0.05 at any checkpoint = "essentially perfect reconstruction" ⇒ the pipeline + conditioning can fit when the task is determined. Secondary/partial: SSIM ≥ 0.90. Below 0.85 after 10k steps = the interesting negative — escalate per §6 rather than concluding.

## 4. Planned code (per file)

- **NEW `src/maxdiffusion/data_preprocessing/build_exp02_overfit100.py`** (~260 LOC, CPU): annotation scan + filtering (D1/D2), latent fetch + window build (D4), TFRecord + sidecar writer (D5); pure functions for selection/instruction-pick/windowing; `--dry-run` prints the selection without downloading.
- **NEW `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py`** (~200 LOC): subclasses exp_01's `WanTI2VFullFTTrainer`; overrides only (a) dataset parsing (new fields incl. `episode_index`), (b) startup text-table construction, (c) `_denoising_loss` context = gathered per-example embeddings. Objective math untouched (shared helpers).
- **NEW `src/maxdiffusion/configs/base_wan_5b_overfit100.yml`**: from `base_wan_5b_full_ft.yml`; deltas `model_type: OVERFIT100_TI2V`, dataset dirs, `max_train_steps: 10000`, LR 1e-5, GBS 256 keys, `num_text_slots: 100`.
- **EDIT `train_wan.py`** (+3): dispatch `OVERFIT100_TI2V`.
- **NEW `bash_scripts/train_wan_overfit100.sh`** + `full_ft`-style launcher arm.
- **Tests** (`src/maxdiffusion/tests/worklogs_yixun/`): `test_overfit100_selection.py` (success/empty-text filters, seeded reproducibility, per-episode instruction stability, rejection accounting); `test_overfit100_windows.py` (window enumeration/geometry, z_i0 = window's first frame, count math); `test_overfit100_text_conditioning.py` (unique-table build, **gather-by-episode_index correctness with a fixture where index ≠ episode_id**, context dtype/shape, exactly-one transformer call, no actions/adapter/CFG, objective parity vs exp_01 on identical inputs).

## 5. Cycles & ladder

Cycles: **A** selection+windowing pure functions → **B** dataset builder + build the real dataset (CPU/local, no TPU) → **C** trainer + config + launcher. Each: write (test-first) → briefed Codex review → strengthen → commit. Ladder: rung 1 suite/static; rung 2 stub-transformer forward with per-example context; **rung 3 readback of a real `.pt` + the built TFRecords** (geometry, dtypes, instruction round-trip); rung 5 v6e-8 smoke (asserts + text-table + one step, storage-light); rung 6 v6e-64 fit probe (GBS 256); rung 7 full run.

## 6. Risks / escalation

R1 latent `.pt` layout differs from assumption → build fails loudly (D4), plan revised. R2 memorization still incomplete at 10k → escalate: 30k steps, then LR 2e-5, then fewer episodes (e.g. 10) to bound the question — each a separate approved launch. R3 text table memory: 100×512×4096 bf16 ≈ 400 MB replicated — acceptable; asserted at startup. R4 dataset upload still in progress (files timestamped today) → per-episode existence check at selection time, and the frozen list makes any later change detectable. R5 one view per episode reduces diversity — accepted per Query 2.
