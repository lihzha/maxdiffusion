# exp_02 `overfit100` — Worklog

Append-only lab notebook. Entry template in `experiment_SOP.md`.

## 2026-07-28T00:10:00Z — Scaffold exp_02: reserve number, branch + worktree

- **Goal** — Reserve `exp_02` and stand up bookkeeping for the text-conditioned 100-trajectory overfit experiment (Lihan/Yixun's corrected design after exp_01).
- **Change** — Created `docs/worklogs_yixun/exp_02_overfit100_claude/` with `overfit100_yixun_query.md` (Query 1 verbatim + accepted critique of exp_01 + spec + open questions) and this worklog.
- **Version Control** — Integration branch `yixun-dev` @ `1bc0030` (base_commit). Experiment branch **`claude-exp_02_overfit100-20260728`**; worktree **`/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100`**. Docs auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Command / Validation** — `git worktree add … -b claude-exp_02_overfit100-20260728 yixun-dev`. Verified `exp_02` unused (only `exp_01_full_ft_overfit_claude` exists; no exp_02 branch) — SOP never-reuse rule satisfied.
- **Result** — `scaffolded`.
- **Analysis** — SOP-role note: the Planner tier is specified as Fable 5 (max); this session is **Opus 5 (max effort)** by the user's explicit `/model` choice. Recorded as a deliberate deviation, not a silent substitution.
- **Next** — Data investigation (does `meta_json` carry the DROID language instruction? where is the raw source? how is text embedded?), then `plan_overfit100.md` → Codex plan review → user approval → cycles.

## 2026-07-28T01:05:00Z — Data investigation: what exists, what's missing (blocking fact identified)

- **Goal** — Determine the cheapest correct data path for a 100-trajectory, text-conditioned overfit set.
- **Findings (repo evidence):**
  1. **Trajectory grouping is FREE from the existing GCS cache.** Cached window `name` is `ep<EPISODE>_v<VIEW>_s<START>` (e.g. `ep10099_v0_s00000`) — all windows of one trajectory share the `ep<N>` prefix, so "100 trajectories" can be selected and their windows enumerated from `gs://v6_east1d/datasets/droid_wan_side_adapter/{train,val}` with no cluster access.
  2. **Text is NOT in the cached TFRecords.** Fields are exactly `name, ordinal, z_i0, z_video, actions, meta_json` (converter `src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py:179-186`). `meta_json` is a **verbatim passthrough** of `<cache_root>/<name>/meta.json` from Lihan's Della cache (`:150-158`) — contents unknown from the repo; docs (`side_adaptor.md`) never mention instructions or the meta schema.
  3. **The instruction machinery already exists for RAW DROID.** `input_pipeline/robot/droid_video_dataset.py` implements exactly the user's spec: `filter_success=True` (keeps `*success*` file_paths, :180-186), `_LANGUAGE_KEYS = (language_instruction, language_instruction_2, language_instruction_3)` with **seeded random 1-of-3 selection** (:249-255), seeded camera choice, windowing. **Gap vs the user's ask:** it does NOT filter EMPTY instructions.
  4. **Raw DROID is on cluster filesystems, not GCS** — `base_wan_ctrl_world.yml:288 raw_data_root: /n/fs/iromdata/droid_raw/1.0.1`; the Della converter reads `/scratch/gpfs/AM43/lz3952/Wan2.2/data/droid_cache_windows_v0/train`. TPU workers see only GCS.
  5. **Per-example T5 conditioning is new work either way** — every existing Wan trainer computes ONE null-prompt embedding and broadcasts it (`wan_ti2v_side_adapter_trainer._compute_null_context`); nothing caches or batches per-example text embeddings.
- **The one blocking fact:** does `meta.json` carry the DROID language instruction (or a joinable episode/file_path key)? GCS probe blocked — **gcloud reauth required** (`ReauthUnattendedError`).
  - **If YES** → exp_02 is buildable end-to-end from GCS: select 100 `ep*` trajectories → read instructions from `meta_json` → precompute T5 embeddings once (100 strings) → write a small text-conditioned TFRecord set → full-FT with per-example context. No cluster dependency.
  - **If NO** → we need either (a) Lihan/Della to re-cache 100 trajectories including instructions, or (b) a raw-DROID→instruction join on a machine that can see `/n/fs/iromdata/droid_raw/1.0.1` (IROM) to produce an `ep<N> → instruction` map, which is then uploaded to GCS (tiny file) and joined to the cached windows.
- **Result** — `in_progress` (investigation), blocked on reauth for the decisive probe.
- **Next** — Probe `meta_json` the moment auth returns; then draft `plan_overfit100.md` on the confirmed branch (A or B).

## 2026-07-28T02:15:00Z — Data question RESOLVED: everything needed is on GCS (no cluster dependency)

- **Goal** — Settle the blocking fact (instruction source) and fix the data path.
- **Command / Validation** — Probed a cached window + the bucket:
  - `meta_json` schema (verbatim from Lihan's cache): `episode_id, video_path, view, start_frame, frames(33), usable_length, video_length, rank_in_episode, action_score, candidate_stride(4), z_I0_shape[48,1,12,20], z_video_shape[48,9,12,20], source="droid_ctrl_world", split`. **No instruction text** — but `episode_id` is a clean join key.
  - **`gs://v6_east1d/datasets/droid_ctrl_world_aligned/`** exists with `annotation/train/<episode_id>.json` (**69,723 episodes**, ids 0–69722), `latent_videos/train/<ep>/<view>.pt` (~1.35 MB each, 3 views), `videos/train/<ep>/<view>.mp4` (~0.3–0.9 MB).
  - Annotation schema (sampled 10): **`texts`: 1–3 instructions** (the DROID triple), **`success`: 0/1**, `episode_id`, `video_length`, `videos[]`, `latent_videos[]`, `states[]`. Examples: ep1 success=1, 3 texts ("Put the banana plushy in the box"); ep5000 success=0; ep45678 success=1, 1 text.
  - Upload timestamps are TODAY (latents 09:37Z, videos 15:13Z) — dataset may still be filling; **per-episode availability must be verified at selection time**.
- **Result** — `passed` — every ingredient for the user's spec is GCS-resident and TPU-readable: success filter, the 1-of-3 instruction choice with empty-filtering, and latents. **No Della/IROM access required.**
- **Analysis** — Two candidate data paths: **(A)** build the 100-episode window set directly from `latent_videos` (100×~1.35 MB per view = trivial download, full control over windowing, and the MP4s give true RGB ground truth for galleries) vs **(B)** locate the same episodes' windows inside the existing 334 GB cached train shards (needs a one-time full-shard index scan). **A is strictly cheaper and more flexible** — adopted, subject to a rung-3 readback confirming the .pt layout reproduces the cache geometry ([48, T_lat, 12, 20], z_i0 = first latent frame).
- **Next** — `plan_overfit100.md`, then Codex plan review.

## 2026-07-28T04:10:00Z — Plan review (4 BLOCKERs) + empirical checks that INVALIDATE the planned data path

- **Goal** — Close the plan-review cycle; verify F4's latent-layout demand before locking any build contract.
- **Review result** — REQUEST-REVISION, `overfit100_codex_plan_review.md`: **F1** determinacy claim overstated (text+first frame does NOT determine the recorded future; and exp_01 could itself have memorized via unique first-frame fingerprints, so its 0.787 is not an established conditioning ceiling) → reframe as a **finite-set memorization test** and add correct/null/shuffled-text rollout controls; **F2** the eval tooling literally cannot run a new `model_type` (generate script requires an `actions` feature, hard-codes FULL_FT_TI2V, broadcasts null context; the val-loss evaluator rejects other model types; inherited `_data_shardings()` lacks `episode_index` ⇒ JIT tree mismatch); **F3** 16 windows ≠ "reconstruct the training samples" (need per-episode coverage, all windows at the final checkpoint, multiple seeds, and an exact threshold rule; primary SSIM must compare against the VAE decode of target latents, raw MP4 only as an aligned auxiliary); **F4** prove the `.pt` layout and `z_i0` semantics before building; **F5** text-table memory is 400 MiB replicated per chip (my "HBM story unchanged" was false), encode in a bounded loop, keep Wan's zero-padded no-mask contract; **F6** frozen-list reproducibility needs a committed manifest with source fingerprints (the dataset was uploading during our probes); **F7** stage the compute — 2,500-step resumable first segment with observations at 250/500/1000/2500, a v6e-8 GBS-32 probe, and a 1-clip / 10-episode sanity gate BEFORE any 10k run.
- **Empirical checks (this is why F4 mattered):**
  1. **`z_i0 == z_video[:, 0]` bit-identically** in the exp_01 cache (max|diff| 0.0000) ⇒ first-frame conditioning IS a slice; no separate image-conditioning encode. Slicing is parity-correct.
  2. **The aligned latents are a DIFFERENT latent space:** `latent_videos/train/0/0.pt` is a bare tensor **(88, 4, 24, 40) float32** — 4 channels, 24×40, one latent per raw frame. Wan2.2 TI2V needs **[48, T, 12, 20]** with 4× temporal compression. These are Ctrl-World/SD-style per-frame latents ⇒ **plan D4's build path is dead as written.**
  3. **The Wan-latent cache is not globally episode-sorted** (shard 0→ep0, 100→ep22422, 300→ep46937, 500→ep71256, 703→ep9944) and **its meta_json schema varies** (some records lack `episode_id`; `name` always encodes it). Cached episode ids exceed the train-annotation range (71256 > 69722) and cached `video_length` (56) disagrees with the annotation's (88) for ep0 ⇒ the two corpora are related but **not frame-aligned**; joining is safe only at episode granularity, and even that needs verification.
- **Result** — `fix_ready` — plan v2 required; two viable data paths (below), both costed.
- **Next** — Yixun's steer on the data path, then plan v2 addressing F1–F7.

## 2026-07-28T05:30:00Z — Query 3 (A′) + plan v2 resolving F1–F7

- **Goal** — Close the plan cycle: adopt the user's A′ data-path decision, resolve all review findings, produce plan v2, request re-review.
- **Change** — `plan_overfit100.md` rewritten (v2); resolutions appended to `overfit100_codex_plan_review.md`; Query 3 recorded. Session model now Fable 5 (xhigh) — Planner tier restored (v1 was drafted under Opus 5 max while Fable was unavailable).
- **Command / Validation** — New probes behind v2: aligned MP4s are **320×192 @ 5fps** (exact cache geometry; ep0 88 frames, ep1 128 — match annotations); `gsutil stat` exposes generation/md5/size (manifest fingerprints, F6). Earlier probes: `z_i0 == z_video[:,0]` bitwise in cache; aligned `.pt` = Ctrl-World `(T,4,24,40)`.
- **Version Control** — branch `claude-exp_02_overfit100-20260728`; v1 @ `cb5d73f`; this commit = plan v2 + resolutions.
- **Result** — `fix_ready`; Codex re-review launched (background) on v2.
- **Next** — Re-review verdict → surface plan v2 + review + resolutions to Yixun for approval → cycle A (manifest builder).

## 2026-07-28T06:40:00Z — Plan v2 re-review (REQUEST-REVISION, G1–G5) → plan v3

- **Goal** — Close the second plan-review round.
- **Result** — Re-review: F1/F5 resolved; F2/F3/F4/F6/F7 partial; new G1–G5 (trainer-seam mismatch BLOCKER — step fns are module-level, jit-bound in `start_training`, confirmed in code; encode contract under-specified BLOCKER — `.mode()` + latents mean/std convention confirmed at `wan_pipeline.py:585/608`; S2 10-episode data path missing BLOCKER; success statistic inexact MAJOR; provenance gaps MAJOR). Plan v3 written resolving all five (own-module trainer with `context_table` state field + rewritten `start_training`; locked encode contract + gates V1–V4 with final thresholds; dual `train100`/`train10` artifacts + numerical S2 gate; exact success formula + guard + machine aggregation artifact; dual fingerprints + ordered draw log).
- **Version Control** — this commit; v2 @ `092eb91`. Third review launched (background).
- **Next** — v3 verdict → surface to Yixun for plan approval.

## 2026-07-28T07:40:00Z — Plan v3 review (REQUEST-REVISION: H1, H2) → plan v4

- **Goal** — Close the third plan-review round.
- **Result** — G1/G3/G5 RESOLVED on record; H1 (V1 fixtures not materialized — true: the three cache windows lived only as an unfingerprinted scratchpad probe) and H2 (checkpoint lists unexecutable: single periodic cadence at `wan_ti2v_full_ft_trainer.py:615-616`, `max_to_keep=3` at `wan_ti2v_side_adapter_trainer.py:392-396`) both verified and fixed in v4, plus the G4 residue (S2/S3 conflation in C₃, mode-unqualified m, overbroad claim → `C₃¹⁰⁰`/`m_corr`/tie-break/two-tier claim).
- **Version Control** — this commit; v3 @ `42c9057`. Fourth review launched (background).
- **Next** — v4 verdict → surface to Yixun for plan approval.

## 2026-07-28T08:30:00Z — Plan v4 review: APPROVE-WITH-CHANGES → plan cycle CLOSED

- **Goal** — Close the plan cycle (round 4).
- **Result** — `passed`: H1/H2/G4-residue all RESOLVED on record; reviewer confirms the new CheckpointManager stays compatible with exp_01's Composite `params`/`opt_state`/`step` restore path and that the context table is correctly rebuilt outside the checkpoint; explicitly "No launch blocker remains", no fifth review needed. Single MINOR I1 (D10/D11 S2-ablation contradiction) fixed in place — one coverage matrix, D10's cheaper contract adopted.
- **Analysis** — Plan cycle: 4 rounds, 14 findings (F1–F7, G1–G5, H1–H2) + 1 minor, all resolved or fixed. Two would-have-been-silent training bugs caught at plan time: subclass overrides that would have trained on null context (G1), and checkpoint retention that would have deleted the gate's step-250 checkpoint (H2).
- **Next** — Surface plan v4 + all four reviews + resolutions to Yixun for the SOP approval gate. On approval: cycle A (V1-fixture extractor + manifest builder — local CPU, no TPU).

## 2026-07-28T09:00:00Z — Plan approved (Query 4); cycle A opened

- **Goal** — Record the approval + conditional launch grant; start Coder round 1.
- **Result** — Plan v4 approved by Yixun; dataset-build + S1-smoke pre-approved conditional on dual sign-off; S2/S3 reserved. Cycle A (provenance tooling: `extract_v1_fixture.py` + `build_overfit100_manifest.py`, test-first) assigned to the Opus Coder. Cycle A is local-CPU only — no TPU involved.
- **Next** — Coder implements (red→green) → runs the extractor + manifest builder for real (fixture upload + `overfit100_manifest.json`) → Codex review → strengthen → commits A1/A2.
