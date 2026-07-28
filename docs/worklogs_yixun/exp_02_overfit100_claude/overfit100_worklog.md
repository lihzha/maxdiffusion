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
