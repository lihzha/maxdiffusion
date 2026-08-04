# null_adapter — Codex code review: round R7 `runner-decode-videos`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `06e31aa`. Reviewer independently confirmed the latent→pixel frame mapping against the repo's causal-VAE conventions (latent frame 0 → pixel frame 0 only; future = 1–32), verified SSIM-default parity bit-exactly, fact-checked the dependency premise against the requirements manifests, and ran adversarial probes (NaN latent + input-independent decoder ⇒ valid perfect metrics). Item-6 ruling: KEEP the R6 cached-run reuse.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md` in full.
- `plan_null_adapter.md` v5, especially §§3, 4-P1 metrics/gates, 5, 6, and 8.
- `null_adapter_worklog.md` through the R7 entry and all six Planner positions.
- All R1–R6 Codex code reviews and strengthening records: R1, R2, R3, R4a, R4b, R4c, R5, and R6.
- Both R7 files in full: `null_adapter_pixels.py` and `test_null_adapter_pixels.py`.
- `null_adapter_runner_core.py` arm results, seed keys, and `emit_metric_tables`; `null_adapter_gates.py` parsing, observation validity, and G1.
- `generate_wan_side_adapter.py` import side effect, decode call, `_frame_ssim`, pixel MSE, and comparison-video layout.
- Wan geometry and decode conventions: DROID cache converter, side-adapter config, TI2V pipeline, base pipeline decode/postprocessing, and Wan 2.2 causal VAE temporal upsampling.
- Repository dependency manifests and the issue-#8 ffmpeg record.
- HEAD `06e31aa`; only the two R7 files are untracked.

1. **MAJOR — non-finite arm latents pass “pre-decode validation” and can become valid, even perfect metrics.** [null_adapter_pixels.py:149](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_pixels.py:149), [null_adapter_pixels.py:204](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_pixels.py:204)

   `_checked_latents` checks only shape. My probe placed one NaN in `a0` and used a decoder that returned finite black frames independently of its input; `decode_and_score` accepted it and emitted `future_ssim == 1.0`, after decoding GT and then the corrupt arm. Thus a malformed result can reach every SSIM gate as valid evidence, and the promised validation-before-any-VAE-pass is false.

   Concrete change: require `np.all(np.isfinite(value))` in `_checked_latents`. Add NaN/±inf cases for GT, a single-seed arm, and one probe seed, using `_forbidden_decode` to prove every arm is completely validated before the first decode.

2. **MINOR — the claimed line-by-line SSIM parity is not exact because R7 clips while the reference does not.** [null_adapter_pixels.py:79](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_pixels.py:79), [null_adapter_pixels.py:92](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_pixels.py:92), [generate_wan_side_adapter.py:255](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/generate_wan_side_adapter.py:255)

   The reference passes postprocessed arrays directly to `structural_similarity`; it does not clip inside `_frame_ssim`, nor before pixel MSE. R7 clips both metrics, while `_decoded` permits excursions up to `1e-3`. With values only `5e-4` outside the range—therefore accepted by the seam—I measured an SSIM difference of `1.11e-5`.

   Concrete change: for exact parity, enforce the `[0,1]` decode contract strictly and remove metric-layer clipping. Alternatively, document clipping as a deliberate deviation instead of claiming exact parity, but that weakens comparability unnecessarily because the real pipeline postprocessor already clamps.

3. **MINOR — the broad video fallback is silent and non-transactional.** [null_adapter_pixels.py:307](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_pixels.py:307), [null_adapter_pixels.py:315](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_pixels.py:315)

   Any backend exception is discarded without warning. A partially written MP4 remains, and an existing fallback directory is reused without removing stale `frame_*.png` files; rewriting a shorter clip can therefore return a directory containing frames from the previous clip.

   Concrete change: keeping the broad backend catch is reasonable, but warn with the original exception and publish either the MP4 or PNG sequence through a fresh staged path. Add a fake writer that fails after partial output and a same-path 3-frame→2-frame rewrite test asserting no partial MP4 or stale PNGs remain.

4. **NIT — the R7 dependency premise is factually wrong.** [null_adapter_worklog.md:214](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_worklog.md:214), [base requirements:14](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/dependencies/requirements/base_requirements/requirements.txt:14), [generated requirements:75](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/dependencies/requirements/generated_requirements/requirements.txt:75)

   `imageio-ffmpeg`, `imageio`, and `scikit-image` are already declared in both base and generated requirements. Append a worklog correction rather than editing the append-only entry. R10 should still preflight the actual TPU-host imports/backend executable—the issue-#8 lesson—but should not describe these Python packages as undeclared.

Confirmed:

- SSIM defaults match the reference: `gaussian_weights=False`, default `K1/K2`, `channel_axis=-1`, `data_range=1`, and adaptive odd `win_size`. My default comparison was bit-exact; Gaussian/K1/K2 mutations differed and the direct-skimage oracle catches them.
- Exactly `7×7` correctly uses `win_size=7`.
- The repository confirms temporal factor 4, `num_frames=32`, nine latent frames, and 33 decoded frames. The first latent chunk produces only pixel frame 0; future frames are exactly 1–32. Causal decoding means the anchor may influence later frames, but those later frames are not pinned.
- GT is decoded once; F=32, grayscale, dropped-batch, nonfinite output, and `[-1,1]` hazards are rejected.
- Float64 MSE accumulation, exact namespace matching, non-mutation, and all-four-metric injection are correct.
- Raise-instead-of-NaN and finiteness-only fill are sound divisions of labor: systemic environment/geometry faults should abort, while negative finite SSIM is a real measurement for gate policy to handle.
- `generate_wan_side_adapter` is not imported; clean import also leaves `skimage` and `imageio` unloaded.
- Validation: 37 non-I/O R7 tests passed. Full suite rerun with capture disabled reached 372 passes; the known tiny-Wan failure and two R7 `tmp_path` setup errors were solely the read-only sandbox’s lack of a writable temporary directory.

Item-6 ruling:

**APPROVE / KEEP.** The R6 test-module import is confined to one test-only end-to-end composition check, reuses a genuinely cached production run, and is not R7’s principal metric oracle—the independent synthetic tests already carry that role. Extract shared test support only if R8 or another round becomes a second cross-round consumer.

Final verdict: **REQUEST-REVISION — non-finite arm latents can currently be converted into valid gate evidence, and the clipping/fallback discrepancies should be tightened in the same round.**
