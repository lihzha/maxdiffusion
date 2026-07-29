# Code review: exp_02 overfit100 — cycle B (dataset-builder)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-29

## Context loaded
- Experiment SOP and announcements — review separation, validation ladder, provenance, launch approval, and failure discipline.
- Approved plan v4 — D4 encode contract, gates V1–V4, D6 schema-v2 artifacts, and probe requirements.
- Queries 1–4, worklog, and committed manifest — user decisions, 100-episode identity, 1,629-window total, and conditional launch grant.
- Cycle-A review and strengthening record — source binding, fail-closed manifest validation, fixture integrity, and provenance continuity.
- exp_01 analysis/results — cached-latent semantics, rollout metric conventions, and the motivation for finite-set memorization.
- Cycle-B implementation and tests — builder, launcher, shared guard change, five test files, mutation evidence, and reference Wan encode/decode paths.

## Verdict
REQUEST-REVISION. The windowing, pipeline-parity encode math, schema-v2 serialization, same-byte train10 subset, and core V1–V3 logic are strong. The TPU launch remains blocked because the approved VAE pin is optional, V4 can pass non-finite output, canonical prefixes can retain partial/unfingerprinted data, and the proposed probe does not yet validate several scale and provenance claims it is meant to de-risk.

## Findings

1. **B1 — BLOCKER — The build can succeed without the approved VAE pin and does not bind weight loading to the fingerprinted revision.** The committed manifest contains no `vae_fingerprint`; `preflight()` explicitly converts that missing contract into a warning, while the later loader receives the mutable repository name rather than the observed revision (`docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json:1`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:674`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:786`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:805`). The launcher also invokes a general prefetch that requests transformer and text files for a VAE-only job (`bash_scripts/build_overfit100_dataset.sh:63`, `bash_scripts/prefetch_hf_snapshot.sh:54`). **Concrete change:** land the small cycle-A follow-up: add the exact revision and full VAE-config SHA-256 to the committed manifest, make the structural validator require both, resolve that revision to one local snapshot, and pass that exact snapshot to both fingerprinting and `from_checkpoint`; prefetch only the VAE files. Absence or mismatch must abort before model loading.

2. **B2 — MAJOR — V4 can pass vacuously on NaNs, and its numerical-escalation policy is not diagnostic enough.** `difference > tolerance` is false for NaN, so a non-finite short-graph output can produce `passed=True`; neither input is checked for finiteness (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:337`). **Concrete change:** fail V4 unless both tensors and every derived metric are finite, add a NaN/Inf regression test, and persist the two frame-zero tensors plus difference quantiles on failure. A probe trip must still abort; any tolerance revision should require same-shape future-replacement/repeated-encode controls, a new reviewed commit, and renewed launch sign-off—not an automatic continuation.

3. **B3 — MAJOR — Gate or audit failure can leave partial shards in the canonical production prefixes.** Shards are uploaded as buffers fill, but final gate completion, audit computation, and sidecar creation happen later; a late V2/V3 failure or audit exception therefore leaves trainer-globbable canonical files despite the module’s “never write a partial dataset” contract (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:711`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:938`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:1010`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:1018`). **Concrete change:** write under a build-specific staging prefix, complete gates, counts, audit, and readback there, then promote to an initially empty canonical prefix and write `_SUCCESS` last; on promotion failure, remove only the explicitly enumerated objects created by that attempt. Future readers must require `_SUCCESS`.

4. **B4 — MAJOR — The two advertised “fingerprinted sets” are not fingerprinted or physically read back.** Each shard record contains only a path and intended record count; the summary records no generation, MD5, or size, and `run()` never parses the uploaded bytes (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:739`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:1056`). A later overwrite can retain the expected count while changing training bytes. **Concrete change:** record local hash/size and remote generation/MD5/size for every shard, download or directly read each pinned uploaded object, parse all records, and assert exact ordered names, schema, byte lengths, `z_i0 == z_video[:, :1]`, per-set counts, and train10/train100 example-byte consistency before publishing `_SUCCESS`.

5. **B5 — MAJOR — The proposed probe does not substantiate its scale, rollover, or resource claims.** The first two manifest episodes contain only 41 windows, below the 256-record shard boundary, and the audit therefore runs at 41 rather than 1,629 vectors (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:105`, `docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json:45`, `docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json:75`). HBM reads only the first device and may silently be `None`; the extrapolation mixes fixed startup/JIT cost into per-window rate while excluding audit and sidecar time (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:703`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:899`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:1021`). **Concrete change:** make the probe force at least one shard rollover, run a full-geometry/full-count synthetic audit benchmark before the expensive encode, require peak-memory data for every local device or fail the probe, and report separate preflight/load, compilation, steady-state encode, upload, audit, and total timings.

6. **B6 — MAJOR — The provenance guard’s API change is sound, but the guarded input set is incomplete and probes bypass it.** `CYCLE_B_IMPLEMENTATION_PATHS` omits the modified shared guard itself, the consumed manifest, effective config, and invoked prefetch helper; additionally, every probe skips the guard and deliberately records `build_commit=null` (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:124`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:904`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:1035`). **Concrete change:** retain the backward-compatible `paths=` API, but guard or content-hash every effective code/config/manifest input, run the clean-commit check for probes as well as production, and record the actual pushed SHA in probe summaries.

7. **B7 — MINOR — V3’s stated VAE ceiling is not decode-parity with the later rollout evaluator.** The builder intentionally omits the pipeline’s bfloat16 postprocess, while `_decode_latents_to_video` performs that conversion; thus the authoritative V3 number is not exactly the ceiling used by generation-time metrics (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:251`, `src/maxdiffusion/pipelines/wan/wan_pipeline.py:663`). **Concrete change:** use pipeline-equivalent bfloat16 postprocessing for the gated/reported V3 ceiling; retain the float32 result only as a separately named diagnostic. Float32 decode followed by uint8 quantization remains acceptable for V1’s encode-path isolation.

## Decision judgments

a) **V4:** A trip must abort; no in-job re-thresholding—only a separately reviewed rerun after saved-array and same-shape controls establish numerical rather than causal failure.

b) **Manifest VAE pin:** Amend the committed manifest through a small cycle-A follow-up and make the pin mandatory; warning-and-observe does not satisfy the approved plan.

c) **V1 uint8 round-trip:** Accept—the quantization exercises the exact pixel door used by the build and is more representative than feeding decoder floats directly.

d) **Float32 decode postprocess:** Split the treatment—acceptable for V1 isolation, but V3’s authoritative eval ceiling must use pipeline-parity bfloat16 postprocessing.

e) **Abort remnants:** Require staging plus a final success marker and exact cleanup/promotion semantics; idempotent filenames alone do not make exposed partial prefixes safe.

f) **Driver seams:** Appropriate for core control-flow testing, but insufficient for launch; add mandatory-pin, late-abort-after-flush, uploaded-readback, and multi-shard integration coverage.

g) **LOC overrun:** Accept the current cohesive implementation; a pre-launch module split would add risk, but no further scope should accumulate here.

h) **Other choices:** Loud SSIM failure, removal of a production `--max-episodes`, float64 Gram arithmetic, and same-target exclusion are correct; retain `probe2/` isolation but replace null probe provenance with the actual guarded commit.

## Probe-gap judgments

- **Real gate values — NEEDS-PRE-PROBE-MITIGATION:** precheck all ten fixed V3 windows in the probe so a failure at index 90 cannot waste most of the full build; clearly label V2’s 41-window probe coverage as sampled.
- **V4 numerics — NEEDS-PRE-PROBE-MITIGATION:** add non-finite rejection, saved diagnostics, and the reviewed abort/recalibration policy from B2.
- **V1-vs-colorspace — COVERED-BY-PROBE:** real V1 executes decode→uint8→preprocess→encode, while real-MP4 V3 exercises the ffmpeg RGB side of the contract.
- **VAE-only pipeline load — COVERED-BY-PROBE:** the exact loader is invoked, provided the mandatory pinned-snapshot change lands first.
- **Throughput/RSS/HBM — NEEDS-PRE-PROBE-MITIGATION:** separate fixed/compile/steady-state timing and require per-device peak-HBM evidence rather than permitting `None`.
- **`gs://` writer branch — NEEDS-PRE-PROBE-MITIGATION:** add post-upload stat, pinned readback, full parse, and `_SUCCESS`; a successful `gsutil cp` alone is insufficient.
- **Multi-shard at scale — NEEDS-PRE-PROBE-MITIGATION:** 41 records create one shard; force a probe-only small shard boundary or add a target-GCS rollover microbuild.
- **ffmpeg on the TPU image — COVERED-BY-PROBE:** both real MP4 decoding and frame-count/geometry assertions execute on the target image.
- **Audit at scale — NEEDS-PRE-PROBE-MITIGATION:** benchmark the exact 1,629×11,520 float64 workload or redesign it before running it after all expensive production encodes.

---
## Strengthening record (Coder: Claude Opus 5, 2026-07-29)

Commits: `d2faac5` (cycle-A follow-up: manifest pin), `2222d54` (builder + bash arms),
`2b7623f` (tests). Suite after strengthening: **577 passed, 2 skipped** (399+2 at cycle-B start).
Every finding below was implemented; none was rejected.

1. **B1 — FIXED.** The pin is now part of the manifest contract and binds the weights.
   `build_overfit100_manifest` gained `vae_pin_errors` / `vae_config_sha256` /
   `resolve_vae_snapshot` / `amend_manifest_vae_pin`; `validate_manifest_structure` REQUIRES
   `{hf_repo, revision(40-hex), vae_config_sha256(64-hex)}` (one mutation test per malformation)
   and validates the new `amended` log. The committed manifest was amended in place through the
   tested `--amend-vae-pin` path — verified additive (only `vae_fingerprint` + `amended`; zero
   selection content changed) — pinning `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff7315c768468a5333511427288870b2e9635`,
   config sha256 `d996c340…ac11360`. `preflight()` aborts on absence/mismatch/malformation
   (no warn path), resolves the pinned revision to ONE local snapshot dir (`local_files_only`
   unless `--allow-hub-download`), fingerprints that dir, and passes it to `from_checkpoint`,
   so checked-VAE ≡ loaded-VAE by construction. `--model-dir` is gone; a pre-staged
   `--vae-snapshot-dir` is allowed only if its config sha256 matches. The launcher reads
   repo+revision from the manifest and prefetches `model_index.json vae/*` at that revision
   (`prefetch_hf_snapshot.sh` gained an optional PATTERNS arg + `HF_PREFETCH_REVISION`; its
   default set and exp_01 callers are unchanged, and transformer verification is skipped only
   when transformer patterns were not requested).
2. **B2 — FIXED.** `check_v4` now computes `finite` over both tensors and every derived metric
   and requires it for `passed` (NaN/Inf regression tests on either side, ×3 non-finite values).
   `v4_diagnostics` adds p50/p90/p99/p999/p100 difference quantiles; `persist_v4_diagnostics`
   writes `v4_frame0.npz` (both tensors) + `v4_diagnostics.json` to the output prefix on failure.
   `V4_FAILURE_POLICY` is stated in the module docstring, in the gate-failure message, and in
   `summary.gate_policy`: abort; no in-job re-thresholding; recovery only via a separately
   reviewed rerun with same-shape controls and renewed launch approval. `V4_ATOL` remains 0.0.
3. **B3 — FIXED.** All writes go to `<out_root>/<set>/_staging_<build_id>/` with
   `build_id = <12-char commit>-<UTC stamp>`. Gates, counts, audit, sidecars and the B4 readback
   complete against staging; `require_empty_canonical` runs both before the expensive work and
   again inside `promote()` (staging paths are excluded from that view); `promote()` copies
   enumerated objects and, on failure, removes only the ones it created; `_SUCCESS` is written
   LAST with build_id, commit, records, shards, summary sha256 and manifest sha256; staging is
   then cleaned. `summary.readers_must_require = "_SUCCESS"` — carried to the Planner for cycle C.
4. **B4 — FIXED.** `ShardWriter` records local sha256+size at write time and the remote
   generation/md5/size after upload. `readback_set` re-reads every staged shard (pinned
   generation for GCS), re-hashes it against the written sha256, parses ALL records, and asserts
   ordered names (`assert_readback_names`), schema/dtype/byte lengths, `z_i0 == z_video[:, :1]`
   bitwise, and per-shard/per-set counts; `assert_subset_is_byte_identical` proves every train10
   record is the same bytes as its train100 counterpart. Only then does promotion/_SUCCESS run.
   Integration tests cover late-abort-after-flush (zero canonical objects), a flipped byte, a
   truncated shard, and 5-shard rollover.
5. **B5 — FIXED.** Probe shard size is 16 (`shard_ranges(41, 16) == 3`, asserted); the
   full-geometry synthetic audit benchmark (1,629 × 11,520 float64) runs BEFORE the VAE loads;
   `require_device_memory_stats` fails the probe unless EVERY local device reports
   `peak_bytes_in_use`; `PhaseTimer` separates preflight / vae_load / v1_gate /
   first_window_compile / steady_state_encode / download_decode / upload / readback / audit /
   promote / total; the extrapolation uses the steady-state rate plus separately reported fixed
   and audit costs. The probe also prechecks all ten fixed V3 windows (downloading those MP4s)
   so a failure at index 90 surfaces before the full build, and labels V2 coverage
   `"sampled (41/1,629 windows)"`.
6. **B6 — FIXED.** `CYCLE_B_IMPLEMENTATION_PATHS` now includes
   `build_overfit100_manifest.py`, `bash_scripts/build_overfit100_dataset.sh` and
   `bash_scripts/prefetch_hf_snapshot.sh` (9 paths); `assert_manifest_matches_committed`
   content-hashes the consumed manifest against the committed artifact and aborts on any
   difference (sha256 recorded in the summary and in `_SUCCESS`); probes run the same
   clean-commit guard, so `build_commit` is never null. Verified on the committed tree:
   guard SHA `2b7623f`, manifest sha256 `c02a67be…95df`.
7. **B7 — FIXED.** `decode_latents_to_frames(..., postprocess="bfloat16"|"float32")`; the gated
   V3 value (`run_v3_window`) uses the bfloat16 branch, characterized bit-exactly against
   `VideoProcessor.postprocess_video` on a real torch bf16 tensor, and the float32 number is
   retained as `ssim_float32_diagnostic`. V1 keeps float32 + the uint8 door as accepted.

**Decision judgments:** (a) adopted verbatim as `V4_FAILURE_POLICY`. (b) adopted — pin amended
and mandatory. (c)/(g)/(h) adopted unchanged; probe provenance is now the real guarded SHA.
(d) split as directed. (e) adopted — staging + promotion + `_SUCCESS`. (f) adopted — the four
named integration cases were added on top of the existing seam tests.
