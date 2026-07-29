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

*(Strengthening record appended at round close.)*
