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

## 2026-07-28T12:00:00Z — Cycle A Coder round complete; manifest built (100 eps / 1,629 windows); review launched

- **Goal** — Cycle A write phase (Opus Coder) + real artifact production.
- **Change** — NEW (uncommitted, pending review): `extract_v1_fixture.py` (352 LOC), `build_overfit100_manifest.py` (460 LOC), 3 test files (67 tests). Suite **320 passed + 2 skipped** (253+2 inherited); red-first evidence recorded for all three files; black/ruff clean.
- **Command / Validation** — A1 ran for real: fixture at `gs://v6_east1d/datasets/exp02_overfit100/fixtures/v1_cache_windows.npz` (gen 1785291278937267, md5 Szm9uNUI2AtjyRNTtpm9SA==, 693,246 B; `z_i0 == z_video[:,:1]` bitwise ×3; byte-reproducible re-run). A2 ran for real (~25 min): **100 accepted / 129 draws** (tally: not_success 23, too_short 6; zero missing/no-text), **1,629 windows** (min/max/median per ep = 1/99/12), `train10` = 167 windows, geometry uniform 320×192@5fps, chosen-index histogram {0:73, 1:15, 2:12}. `verify_manifest()` vs live GCS: 201 objects, **zero drift**, 100.7 s — the cycle-B preflight proven at scale.
- **Result** — `passed` (write phase). Codex review of the round launched (marker `provenance-tooling`).
- **Analysis** — Coder deviations logged for the reviewer (block-prefetch with order-invariance test; shared gsutil layer; LOC overage; schema additions; `nb_read_frames`; provisional-accept seam; verify scope). **Two data findings of experimental significance:** (1) **instruction coarseness** — 58/100 episodes carry a single DROID taxonomy-category label (e.g. "Fold, spread out, or clump object…"), not an episode-specific instruction; 6 duplicate-instruction groups cover 22 episodes (largest group 7) — the §1 duplicate-condition audit has real numbers pre-build, and the text channel is coarser than "per-trajectory instruction" implies for most of the set; (2) **window skew** — 1,629 total (not ~1,400), per-episode 1–99 (median 12): uneven training weight per episode; affects encode cost and the full-set claim's denominator. Both surfaced to Yixun. Bookkeeping fix landed: `.gitignore` now negates `docs/worklogs_yixun/**/*.log` on both branches (SOP artifact 9 had been silently blocked since exp_01 — zero committed .log repo-wide).
- **Next** — Review verdict → strengthen (same Coder agent) → commits A1/A2 (+ artifacts incl. the build log) → cycle B.

## 2026-07-28T14:00:00Z — Cycle A CLOSED (strengthened, committed, manifest reproducible)

- **Goal** — Close cycle A: strengthen A1–A4, commit, regenerate the production manifest from a clean commit.
- **Result** — `passed`. All four findings fixed (dirty-tree refusal + full tool versions; stat-first + generation-pinned downloads + md5/size/episode-id binding; exact absence classification + individual retries + deferred post-stop errors + lazy in-walk downloads; fail-closed structural validators with 33 mutation tests + strict fixture validation). One extra defect found by the Coder beyond the review (Resolved.error classmethod shadowing the field — every outcome carried a truthy bound method; renamed + regression-tested). Suite **399 passed + 2 skipped** (146 cycle-A tests), independently re-run by the Planner. Commits `7166436` (A1), `9a24518` (A2), `61ae51b` (artifacts).
- **Analysis** — The rebuilt manifest is bit-identical to the first in episodes/draw_log/tally/totals/fixture (only builder_commit/created_utc/tool_versions differ) — independent evidence of zero source drift across ~75 min and that the hardened IO path reaches identical decisions. `builder_commit` now names `9a24518`, which contains the builder — A1's reproducibility contract holds. Live verify: 201 objects, zero drift. Bookkeeping note: the pre-strengthen artifacts were accidentally swept into the review commit `dbd9734` by a folder-wide git add; superseded by `61ae51b` — recorded here for honesty of the trail. Plan clarified (from the review verdict): the cycle-D shuffled control deranges instruction VALUES, never returning any episode its own text.
- **Next** — Cycle B: `build_overfit100_dataset.py` (encode + gates V1–V4 + train100/train10 writers, test-first). Its v6e-8 build job (2-episode probe → full) is pre-approved conditional on dual sign-off (Query 4).

## 2026-07-29T00:20:00Z — Cycle B write phase complete (dataset builder); review launched

- **Goal** — Cycle B write phase: D4 encode contract + gates V1–V4 + D6 writers, test-first.
- **Change** — NEW (uncommitted): `build_overfit100_dataset.py` (1131 LOC — over plan estimate, flagged), `bash_scripts/build_overfit100_dataset.sh`, 5 test files (+109 tests). `build_overfit100_manifest.py`: `assert_implementation_committed` gains backward-compatible `paths=`.
- **Command / Validation** — Suite **508 passed + 2 skipped** (baseline 399+2). Red: 5 files at collection + 4 post-green mutation spot-checks (each caught: channels-last store, V2 min-std zeroed, f16-before-normalization, same-target-exclusion removed). Live read-only: full `--dry-run` preflight vs GCS clean (100 eps/1,629 windows, zero drift, fixture verified); real-MP4 smoke on ep 25189 — 133 frames == manifest, 26 windows, correct names, preprocess in [−1,1]. Encode path characterized **bit-exactly** vs `VideoProcessor.preprocess_video`; replica of `_encode_video_to_t2v_latents` (`.mode()` + mean/std + f16-last).
- **Result** — `passed` (write phase). Codex review launched (marker `dataset-builder`) with the Coder's decision list (V4_ATOL=0, missing manifest VAE pin, V1 uint8 round-trip, numpy-f32 decode postprocess, abort-leaves-shards, seams, LOC) and the explicit 9-item "not validatable until TPU probe" list for probe-gap judgment.
- **Analysis** — Observed VAE fingerprint recorded: revision `b8fff7315c768468a5333511427288870b2e9635`, config sha256 `d996c340fe9a…` (manifest predates the pin — reviewer to rule on amend-vs-warn).
- **Next** — Review → strengthen → commits → **dual sign-off pre-launch verification** → v6e-8 probe build (rung 4, pre-approved) → full build.

## 2026-07-29T06:10:00Z — Cycle B CLOSED; dual sign-off met; LAUNCH: v6e-8 probe build (rung 4)

- **Goal** — Close cycle B and launch the pre-approved 2-episode probe build.
- **Result (cycle close)** — All B1–B7 fixed (mandatory VAE pin amended into the manifest with provenance; V4 finiteness + abort policy; staging→promote→_SUCCESS; per-shard fingerprints + full physical readback + train10≡train100 byte-identity; probe forces 3-shard rollover + full-scale audit benchmark + per-device HBM required + separated timings + all-ten V3 precheck; guard covers 9 paths + manifest content-hash + probes guarded; V3 gated value bf16 pipeline-parity). Commits `d2faac5`/`2222d54`/`2b7623f`/`4783ed4`, pushed. Suite **577 passed + 2 skipped**, independently re-run by the Planner. Strengthening stayed within the reviewer's own concrete-change directives ⇒ no follow-up review required (SOP: behavior beyond findings would have required one).
- **Dual sign-off (Query 4)** — (a) Codex: cycle A closed; cycle B strengthening record appended (`4783ed4`), every finding resolved. (b) Planner: suite green in own run; rungs 1–3 on record (static+tests; dry-run preflight vs GCS clean incl. pin→snapshot `b8fff73…`; real-MP4 smoke ep 25189; bit-exact preprocessing characterization); commit pushed; this package written at launch time. **Grant conditions met for the probe.**
- **Launch topology note** — v6e-8 is two queue workers; the builder is single-process. Queue CLI provides `--worker0-only` — used, no code change (sign-off preserved). Known residual risk: JAX backend init on one worker of a 2-host slice may hang waiting for the pod; classified infra if it occurs (mitigation at relaunch: TPU single-host bounds env), and the probe is precisely the rung that tests this.
- **Acceptance criteria (probe)** — worker reports commit `4783ed4817a0f26da2b73e61d92340ed87a5e6eb`; preflight: manifest zero-drift (201 objects), fixture verified, VAE pin resolves to snapshot `b8fff7315c…`; V1 passes on all 3 fixture windows (rel-L2 ≤ 0.25 ∧ r ≥ 0.97); V4 passed ∧ finite; V3 ≥ 0.80 on ALL TEN precheck windows; V2 clean on the 41 probe windows (labeled sampled); `probe2/` gets exactly 3 shards (16/16/9) + episodes/window_stats/duplicate_audit/summary + `_SUCCESS`; **nothing** written under `train100/` or `train10/`; per-device `peak_bytes_in_use` present for every local device; audit benchmark (1,629×11,520 f64) time + peak RSS recorded; separated phase timings + extrapolated full-build cost in summary; exit 0, no OOM/NaN. Full build launches only after these are log-verified and the extrapolation is written here.
- **Next** — Launch (entry in `_command.md`), light monitoring of `status.json`.

## 2026-07-29T16:30:00Z — Probe attempt 1 FAILED: real bug (guard assumes git on the worker); fix mini-round dispatched

- **Goal** — Triage probe job `20260729-062523-1937c065-exp02-overfit100-probe-yixun` (FAILED, APPLICATION_ERROR, exit 1, attempt 1, non-retryable).
- **Result** — Worker-0 log: queue capacity wait 06:25→09:06Z; setup + VAE-only pinned prefetch SUCCEEDED (3 files @ pinned revision, snapshot verified — the B1 path works); then `run()` crashed at `assert_implementation_committed` → `git ls-tree HEAD` exit 128: **the queue deploys an uploaded tarball with no .git**. The B6 "probes run the guard too" strengthening assumed a git checkout that never exists on workers.
- **Analysis** — **Real bug** (environment-contract mismatch), not infra. The guard's cleanliness premise is verifiable only on the launching machine; worker-side it must fall back to the launch-verified `COMMIT` env (fail-closed if absent). Same landmine possibly in `assert_manifest_matches_committed` — sweep ordered. Monitoring lesson (separate, infra): 30 status polls read "PENDING" for 2.5 h because gsutil reauth had expired and the poll treated auth failure as absence — future polls must distinguish auth errors and alarm; reauth recurrence going into the issue report.
- **Next** — Coder mini-round (marker `tarball-guard`) → focused Codex review → strengthen → commit/push → re-sign-off (Query 4: code change voids the prior grant for this job; dual sign-off re-establishes it) → resubmit probe.

## 2026-07-29T19:30:00Z — Tarball-guard round CLOSED (T1–T3); dual sign-off re-established; probe RESUBMITTED

- **Goal** — Close the fix mini-round and resubmit the probe under the Query-4 grant.
- **Result** — T1 (sanitized git env for every git subprocess + fatal-on-ambiguous discovery + bash-side isolation; poisoned-env regressions incl. our linked-worktree layout), T2 (deployed-mode manifest must resolve beneath the deployed-code root; symlink-escape rejected), T3 (strict single-line 40-hex bash guard, executed verbatim by pytest). Suite **638 passed + 2 skipped**; Planner re-ran the suite AND independently reproduced the poisoned-GIT_DIR check (worktree still detected; guard returns real HEAD `53d69f5`, ignores planted COMMIT). Commits `49f4412` + `53d69f5`, pushed. Coder disclosed a mid-round `git checkout --` that transiently reverted uncommitted work — caught by git status, re-applied and re-verified pre-commit; committed tree verified correct.
- **Dual sign-off (re-established for the changed job)** — Codex: tarball-guard review closed, T1–T3 resolutions recorded (`overfit100_codex_code_tarball-guard_review.md`). Planner: suite + poisoned-env verification above; commit pushed; this entry + `_command.md` entry at launch time. Acceptance criteria: unchanged from the 06:10Z entry, plus worker must report `build_commit=53d69f53…` in deployed-code mode via the exported COMMIT env.
- **Next** — Resubmit probe at `53d69f5`; monitor with reauth-alarming polls (issue #6).
