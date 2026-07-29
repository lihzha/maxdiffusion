# Plan re-review (v2): exp_02 overfit100
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-28

## Context loaded
- v1 review and appended resolutions — the original F1–F7 requirements and claimed closures.
- `plan_overfit100.md` v2 — the revised A′ experiment and implementation plan.
- exp_02 queries — including the user-selected aligned-MP4 re-encoding path.
- exp_02 worklog — latent-space, cache-slice, MP4-geometry, and object-fingerprint probes.
- experiment SOP — review, TDD, validation-ladder, reproducibility, and launch contracts.
- Repository code — Wan VAE encoding, prompt encoding, trainer seams, generator, and loss evaluator.

## Verdict
REQUEST-REVISION. The scientific reframing and prompt-table design are substantially improved; the Wan2.2 VAE has a usable encode path, 33 frames correctly yield 9 latent frames, and D8 accurately describes the positive prompt branch. However, D7 relies on subclass hooks that do not exist, while A′ still lacks an exact latent-encoding contract and executable validation thresholds. The 10-episode gate and final success rule also remain insufficiently specified for launch.

## Per-finding resolution check
- **F1: RESOLVED** — finite-set framing, non-claims, ablations, and collision audit address the determinacy objection.
- **F2: PARTIALLY** — generator/evaluator edit points are real, but the planned trainer context override is not.
- **F3: PARTIALLY** — coverage improved, but the canonical-only success rule and “best checkpoint, median over seeds” calculation remain ambiguous.
- **F4: PARTIALLY** — the wrong latent path was removed and VAE encoding is feasible, but A′ does not define the exact posterior/normalization or fixed validation gates.
- **F5: RESOLVED** — positive-only bounded encoding, Wan-compatible padding semantics, memory accounting, and fallback are adequate.
- **F6: PARTIALLY** — the committed manifest closes rebuild reproducibility, but annotation fingerprints and rejected-candidate provenance are missing.
- **F7: PARTIALLY** — staging is much better, but S2 lacks an implemented 10-episode data path and a measurable proceed gate.

## New findings
1. **G1 — BLOCKER — D7’s override strategy does not match the code.** `prepare_sample` is nested inside `_load_dataset()` in `wan_ti2v_side_adapter_trainer.py`; there is no parse-function hook. More importantly, `_denoising_loss`, `_train_step`, and `_eval_step` are module-level functions, and `WanTI2VFullFTTrainer.start_training()` directly binds those functions, so defining subclass methods cannot replace null context with the episode table. Only `_data_shardings()` is genuinely overrideable. **Concrete change:** either refactor the full-FT trainer to expose dataset/context/step hooks, or explicitly plan an overridden `_load_dataset()` plus new loss/train/eval functions and an overridden `start_training()` that binds them; add a test proving the subclass executes row-distinct gathered contexts.

2. **G2 — BLOCKER — A′ does not uniquely specify the stored targets or its validation gates.** `AutoencoderKLWan2p2.encode()` returns a Gaussian distribution; the existing preprocessing path samples it and normalizes by `latents_mean/std`, while the pipeline’s video helper uses `mode()` and the same normalization. V2 specifies neither choice, sampling seed, RGB normalization/layout, latent normalization, nor exact VAE revision. Its gates use “e.g.,” “documented tolerance,” and an undefined sampled cohort. The encode path itself is feasible and its causal arithmetic is correct: `1 + (33−1)//4 = 9`. **Concrete change:** lock mode versus seeded sampling, preprocessing, mean/std normalization, transpose, dtype, and VAE fingerprint; predeclare every gate’s metric, threshold, and deterministic sample set. Make the two-episode probe record peak HBM and windows/second, extrapolate the full per-window cost, and require a future-frame-invariance check for latent frame zero.

3. **G3 — BLOCKER — S2 cannot currently select the promised 10-episode training set.** D6 defines one 100-episode TFRecord set, while neither the trainer/config nor planned files define a pre-shuffle `episode_index < 10` filter or a separate 10-episode artifact. The proceed criterion—“clear monotone improvement with best-window SSIM approaching ≥0.9”—is also subjective and cherry-pickable. **Concrete change:** materialize a fingerprinted `train10` split or add a tested pre-shuffle subset filter with an exact count assertion; replace the gate with a numerical cohort aggregate and an explicit monotonicity tolerance.

4. **G4 — MAJOR — The success rule is not yet an exact executable statistic and overstates its scope.** “Best checkpoint, median over the 3 seeds” does not identify eligible checkpoints or state whether SSIM is first medianed per window and then thresholded. Intermediate checkpoints have only one seed, and the experiment can pass on 90 canonical windows even if most noncanonical training windows fail; the all-window evaluation has only one seed and no pass/fail role. No planned edit explicitly owns cross-seed aggregation. **Concrete change:** define the formula, for example `m(w,c)=median_seed SSIM(w,c,seed)` followed by the fraction of windows with `m≥0.95`, list eligible three-seed checkpoints, and add a machine-written aggregation artifact. Either add an all-window success guard or narrow the conclusion to “canonical-window memorization”; collision cases must not alter the denominator post hoc.

5. **G5 — MAJOR — Source provenance remains incomplete.** Selection depends on annotation JSON contents, but D5 fingerprints only the MP4 and records only rejection tallies, not rejected candidate IDs/reasons. **Concrete change:** record generation/md5/size for both annotation and MP4 objects, preserve the ordered candidate/rejection log, and stamp the manifest-builder commit and decoding-tool versions.

---

# Planner resolutions (plan v3, 2026-07-28 — Claude Fable 5 xhigh)

All five findings **accepted**; plan revised to v3 (v2 @ `092eb91`). Code claims in G1/G2 were independently verified against the tree before revising (module-level step binding confirmed at `wan_ti2v_full_ft_trainer.py:534-541`; `.mode()` + latents normalization convention confirmed at `wan_pipeline.py:585/608`).

- **G1 (BLOCKER, trainer seams) — FIXED.** D7 rewritten to match reality: new module owns `Overfit100TrainState` (with `context_table` replacing the `null_context` state field), its own module-level `_denoising_loss`/`_train_step`/`_eval_step`, and a **rewritten `start_training()`** that jit-binds this module's functions; subclass overrides only the genuine seams (`_load_dataset`, `_data_shardings`). Row-distinct-gather test (index ≠ id fixture) + objective-parity-on-null-row test added to §4.
- **G2 (BLOCKER, encode contract) — FIXED.** D4 locks: pipeline-verbatim preprocessing/layout, `.mode()` (deterministic, no RNG), pipeline `latents_mean/std` normalization, f16 storage, VAE snapshot-revision + config-hash pin. Gates V1–V4 now have final thresholds and fixed deterministic sample sets (V1 rel-L2 ≤ 0.25 ∧ r ≥ 0.97 on the 3 locally-held cached windows; V2 envelope from cache-observed stats; V3 SSIM ≥ 0.80 on manifest indices 0,10,…,90; V4 frame-0 future-invariance rtol 1e-3 via 33- vs 17-frame encode). Rung-4 two-episode probe records peak HBM + windows/sec and the extrapolated full-build cost before the full build.
- **G3 (BLOCKER, S2 data path + gate) — FIXED.** D6 now builds **two fingerprinted artifacts** (`train100`, `train10` = manifest indices 0–9) from the same job; config selects the dir; trainer asserts `expected_windows`; train10 filter + count test added. S2 gate is numerical and predeclared: proceed iff mean m(w,2500) ≥ 0.70 ∧ (mean at 2500 − mean at 250) ≥ 0.15 ∧ max_w m(w,2500) ≥ 0.85.
- **G4 (MAJOR, success statistic) — FIXED.** D11 defines `m(w,c) = median_seed SSIM` (median first, threshold second), `C₃` = segment-final checkpoints, success = `max_{c∈C₃} frac{m(w,c) ≥ 0.95} ≥ 0.90`; denominator fixed at build, collisions never dropped; all-window guard (< 0.75 at 0.90 ⇒ claim narrowed to canonical-window memorization); evaluator writes the machine aggregation artifact; the statistic is a pure tested unit.
- **G5 (MAJOR, provenance) — FIXED.** D5 fingerprints annotation JSON **and** MP4 (generation/md5/size), keeps the ordered draw log (every drawn id + accept/reject reason), and stamps builder commit + tool versions.

Material revision ⇒ third review requested (v3).
