# Plan review (v3): exp_02 overfit100
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-28

## Context loaded
- v2 review and v3 resolutions — G1–G5 requirements and claimed closures.
- v3 plan — data, trainer, staging, evaluation, and provenance contracts under review.
- round-1 review — F1–F7 continuity and prior resolutions.
- exp_02 query and worklog — user decisions and empirical latent/data probes.
- experiment SOP — TDD, reproducibility, validation, retention, and launch requirements.
- repository code — Wan VAE path, dataset iterator, trainer seams, checkpointing, generator, and evaluator.

## Verdict
REQUEST-REVISION. G1, G3, and G5 are genuinely closed, and D4’s numerical tolerances are generally sane, but G2 remains non-executable because V1’s reference fixtures are not materialized, while G4 still conflates runs/context modes and permits an overbroad claim. The nonuniform checkpoint schedule also has no corresponding save/retention implementation, so S2’s required checkpoint-250 comparison is not guaranteed to survive.

## Per-finding check
- **G1 / F2: RESOLVED** — D7 now matches the real seams: own state and module-level steps, overridden nested-parser owner and data shardings, and rewritten `start_training()` binding the new functions.
- **G2 / F4: PARTIALLY** — mode, normalization, layout, dtype, VAE pin, sample names, and thresholds are explicit; no “e.g.” or “documented tolerance” remains, and V1’s `rel-L2 ≤ 0.25 ∧ r ≥ 0.97` is a reasonable VAE-cycle preflight, but its three reference tensors are unavailable/unfingerprinted (H1).
- **G3: RESOLVED** — separate `train10`/`train100` artifacts plus physical-count assertion are sufficient; the S2 endpoint gate is deterministic, permissive but sensible, and avoids subjective monotonicity.
- **G4 / F3: PARTIALLY** — `C₃` contains both S2-step-2500 and S3-step-2500 although only S3 has the 100-window denominator, and `m(w,c)` omits `context_mode` despite three modes being emitted. Moreover, passing only 75% of all windows at SSIM 0.90 cannot support an unqualified finite-set-memorization claim; define `C₃¹⁰⁰` using run-qualified S3 checkpoints, `m_correct`, a deterministic best-checkpoint tie-break, and either always claim canonical-window memorization or add a separate all-window success rule.
- **G5 / F6: RESOLVED** — accepted annotation and MP4 fingerprints, complete ordered draw/rejection history, builder commit, and decoding-tool versions close provenance.
- **F7: PARTIALLY** — the staged decision logic is sound, but its checkpoint production and retention contract is reopened by H2.

## New findings
1. **H1 — BLOCKER — V1’s fixed reference cohort is named but not materialized.** No corresponding latent tensors exist in the worktree or under `/Users/yixunhu`; only exp_01 metadata/metrics for `s00000` are present, while the V1-containing builder is planned to run remotely on v6e-8. **Concrete change:** place the three exact cache tensors in a committed or fingerprinted GCS fixture, record paths/object hashes and tensor metadata, and add a preflight proving the build job can read all three.

2. **H2 — BLOCKER — D10’s checkpoint lists are not executable through the current trainer.** `wan_ti2v_full_ft_trainer.py:615-616` saves only at one periodic `checkpoint_every`, while `wan_ti2v_side_adapter_trainer.py:392-396` hard-codes `max_to_keep=3`; this cannot directly produce and retain S2 `{250,500,1000,2500}` or S3 `{250,500,1000,1750,2500}`. **Concrete change:** add an exact `checkpoint_steps` contract to the rewritten trainer/config, test the emitted and retained step set, protect segment-final checkpoints across resumes, and budget the roughly 30-GB full-state cost per retained checkpoint.

---

# Planner resolutions (plan v4, 2026-07-28 — Claude Fable 5 xhigh)

Both blockers and the G2/G4 residue **accepted**; plan revised to v4 (v3 @ `42c9057`).

- **H1 (BLOCKER, V1 fixtures) — FIXED.** D4/V1 now specifies materialization: new `extract_v1_fixture.py` (cycle A) pulls the three named cache records into one `.npz`, uploads to `gs://v6_east1d/datasets/exp02_overfit100/fixtures/v1_cache_windows.npz`, and its GCS generation/md5/size go into the committed manifest; the build job runs a preflight (read + md5 + name verification) before any encoding. Extraction + preflight are tested units.
- **H2 (BLOCKER, checkpoint schedule) — FIXED.** D7 gains an explicit contract: the rewritten `start_training()` saves when `(step+1) ∈ checkpoint_steps` (config lists for S2/S3) and builds its own CheckpointManager with `max_to_keep=None`; segment-final checkpoints protected across resumes; ~30 GB/checkpoint budget predeclared (S2 ≈ 120 GB, S3 seg-1 ≈ 150 GB); fake-loop test asserts the emitted AND retained step sets.
- **G2 residue — closed by H1** (the only open item was fixture availability).
- **G4 residue — FIXED.** D11 now: `C₃¹⁰⁰` = S3 segment-final checkpoints only (S2 excluded from the statistic); `m_corr` is context-mode-qualified (correct only; ablations reported, never in the statistic); deterministic `c*` tie-break (fraction → mean → earlier step); two-tier claim replaces the 0.75 guard — headline "canonical-window memorization" per the canonical statistic, stronger "full-set memorization" only with the separate all-window rule (`≥0.90` of ALL windows at SSIM ≥0.90, seed 0, correct, at `c*`).

Material revision ⇒ fourth review requested (v4).
