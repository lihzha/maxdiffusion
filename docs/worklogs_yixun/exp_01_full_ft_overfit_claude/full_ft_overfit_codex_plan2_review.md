# Plan review: exp_01 full_ft_overfit — Part II (val-set evaluation)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-26

## Context loaded

- `experiment_SOP.md` — establishes the TDD, review, validation-ladder, launch, and artifact requirements.
- `full_ft_overfit_yixun_query.md` — Query 8 fixes the full-val coverage, RNG, objective, rollout, reporting, and launch contracts.
- `full_ft_overfit_analysis.md` — identifies the missing held-out instrument and limits of the Part-I conclusions.
- `full_ft_overfit_worklog.md` — the last three entries close Part I and record Query 8 plus the blocked GCS readback.
- `plan_full_ft_overfit.md` — Part II sections II.1–II.6 are the artifact under review.
- `generate_wan_side_adapter.py` — establishes ordered TFRecord iteration, full-FT state construction, restore behavior, output paths, and rollout artifact schemas.
- `side_adapter_wan.py` — establishes pinning, sigma construction, and scalar masked-loss normalization.
- `wan_ti2v_full_ft_trainer.py` — establishes the exact full-FT objective, transformer call, and RNG split.
- `wan_ti2v_side_adapter_trainer.py` — establishes uniform timestep sampling and fresh Gaussian-noise generation.
- `base_wan_5b_full_ft.yml` — confirms the 25-step uniform schedule, fresh noise, guide 1.0, dtypes, and existing validation keys.
- `validate_wan_full_ft.sh` — confirms `VALIDATION_OUTPUT_DIR` is passed as an independent root.
- `pyconfig.py` — confirms runtime configuration objects cannot be mutated through attribute assignment.
- `wan_side_adapter_droid_cache_to_tfrecord.py` and its conversion wrapper — establish how stored ordinals and shard ordering were created.

## Verdict

APPROVE-WITH-CHANGES. The experimental design is sound: full held-out coverage, fixed counter-based per-example randomness, masked per-example aggregation, and sequential checkpoint evaluation are appropriate. D1 correctly needs to match the training distribution of `(t, ε)`, not replay nonexistent training draws for held-out examples, but several load-bearing implementation and test contracts need to be made explicit before coding. TPU launch remains separately gated by the requested pre-launch package.

## Findings

1. **F1 — MAJOR.** **Plan claim:** `fold_in(key(seed), position) → split → (t, ε)` is checkpoint-, order-, batch-, and host-invariant. **Evidence:** JAX’s stateless keys make this sound for scalar positions below the `fold_in` data range, and training samples `t ~ randint([0,25))` independently of fresh `ε ~ N(0,I)` using split keys. This reproduces the correct marginal law and independence, although intentionally not training’s exact stateful bit sequence. The planned tests do not prove that the evaluator wires dataset position rather than the parsed record’s stored `ordinal`. **Concrete change:** specify `randint(..., maxval=num_steps, dtype=jnp.int32)`, validate positive `num_steps`, generate noise with the unbatched example shape, and document distribution parity versus exact-draw replay. In the evaluator integration test, make stored ordinals deliberately unrelated to positions and verify captured `(t, ε)` against positions after reordering, rebatching, and two checkpoint restores.

2. **F2 — MAJOR.** **Plan claim:** enumeration plus a validity-masked padded tail guarantees exactly 14,636 unique examples, including when the source count is wrong. **Evidence:** enumerated positions are automatically contiguous and therefore do not themselves detect duplicated source records. A greater-than-expected dataset is caught only if the iterator is drained to EOF before the count assertion. Repeating the last record is safe for this batch-independent transformer and per-example loss, but `n == 14636` alone does not prove that the duplicate’s loss was excluded from the mean and stderr. **Concrete change:** require an EOF-draining count before checkpoint evaluation; perform a full rung-3 scan asserting stored-ordinal contiguity and unique record names, not merely first/last records; add both fewer-than-expected and more-than-expected tests; and compare padded mean/stderr against an unpadded golden aggregate.

3. **F3 — MINOR.** **Plan claim:** the mean of `masked_velocity_mse_per_example` must equal `masked_velocity_mse` bitwise. **Evidence:** the scalar implementation divides the global float32 sum by `sum(mask) * B`, while a vector mean normally uses a different floating-point reduction tree. They are exactly equal algebraically for equal shapes and `batch_size == B`, but bitwise equality is not guaranteed by JAX/XLA. **Concrete change:** compare each vector element against the scalar helper on its `B=1` slice, then use a tight `allclose` comparison between the vector mean and the existing batch scalar. Keep the existing scalar helper unchanged so training numerics are not altered.

4. **F4 — MAJOR.** **Plan claim:** the evaluator can loop over steps by reusing `_restore_checkpoint_state(..., cohort_mode=True)`. **Evidence:** this is correct, and writing the restored step into `state.step` is harmless because pure evaluation does not consume it. However, `_restore_checkpoint_state` reads `config.checkpoint_step`, while `pyconfig.HyperParameters.__setattr__` rejects mutation; the plan does not define how each loop iteration supplies its step. The helper also restores `params` and the large `opt_state` into newly returned arrays rather than overwriting fixed buffers. **Concrete change:** specify and test either an immutable checkpoint-step proxy passed to the unchanged helper or a reviewed explicit `requested_step` argument. Block on each checkpoint’s final batch before restoring the next, assert both returned step and `state.step`, and account for the params-plus-optimizer restore peak in the resource estimate.

5. **F5 — MAJOR.** **Plan claim:** “Rungs 5–7” are covered by the two production v6e-8 jobs. **Evidence:** those two jobs are T1 and T2 themselves; they do not constitute a storage-light smoke and batch-fit probe. This matters because `_build_full_ft_validation_state` constructs optimizer state and retains the rollout pipeline’s VAE, while prior worklogs show v6e-8 HBM sensitivity. **Concrete change:** include an approved conditional T1 smoke/fit sequence before the full pass—one checkpoint and bounded real batches, followed by the intended B=32 fit check with isolated outputs. Delete the unused VAE/VAE cache in T1 after state/null-context construction, or explicitly justify and fit-test their retention.

6. **F6 — MAJOR.** **Plan claim:** JSON, CSV, and PNG always satisfy the reporting contract. **Evidence:** the proposed fields `commit` and `eval code SHA` are ambiguous, and the matplotlib fallback permits a successful worker run without the required plot while II.6 still requires the PNG. **Concrete change:** define identical JSON/CSV columns explicitly: checkpoint step, mean loss, sample stderr with `ddof=1`, `n`, validation seed, dataset path, run/checkpoint path, training commit SHA, and evaluation commit SHA. Make local plot-only regeneration a mandatory recorded post-step when worker plotting fails, and place or upload the resulting PNG at the declared output path before marking T1 accepted.

7. **F7 — MAJOR.** **Plan claim:** unchanged T2 outputs can be consumed directly as the six requested dataset ordinals and metrics. **Evidence:** the collision-free root claim is correct: `VALIDATION_OUTPUT_DIR=.../validation_valset` becomes `.../validation_valset/step_020000`. However, `metrics.json["ordinal"]` is the stored TFRecord field, not the selector position; the requested positions live in `config.json["validation_ordinals"]` and must be joined by `sample_index`. The exact metric keys are `latent_mse`, `pixel_mse`, and `ssim_avg`, while `summary.json` uses `num_samples`, not `n`. **Concrete change:** have the gallery join `sample_index` to the ordered config positions and label that value “dataset position,” optionally showing stored ordinal separately. Pin the exact metric and summary keys in tests, use deliberately different stored ordinals, test every missing MP4/metrics failure, and specify the SOP-compliant final HTML/assets location with relative video references and the exact provenance sentence.

## Questions for the Planner

None

---

# Resolutions (Planner, Part-II plan v2 — 2026-07-26)

Verdict APPROVE-WITH-CHANGES; all 7 findings **accepted** and applied to the plan in place (marked ←F# there):

- **F1** — randint dtype/maxval + num_steps validation pinned; unbatched-shape ε; distribution-parity-vs-replay documented; integration test specified with stored-ordinals ≠ positions + reorder/rebatch/two-restore (t, ε) capture.
- **F2** — EOF-drain + count assertion BEFORE eval; rung 3 upgraded to full-shard scan (contiguity + name uniqueness); fewer/more-than-expected failure tests; padded-vs-unpadded golden aggregate test.
- **F3** — per-element B=1 bitwise + vector-mean allclose (rtol 1e-6); scalar training helper untouched.
- **F4** — explicit reviewed `requested_step` kwarg (immutability-safe, default byte-identical); block-before-restore; assert returned step AND state.step; restore transient counted; VAE/text-encoder deleted in T1 (structure-tested).
- **F5** — dedicated T1 smoke/fit job (1 checkpoint, ~4 real batches at production B, isolated output) added as rungs 5–6, in the pre-launch package.
- **F6** — exact 9-column schema fixed for JSON+CSV; plot mandatory (worker attempt → recorded local regeneration + upload; T1 acceptance gated on the PNG existing).
- **F7** — gallery joins sample_index→ordered config positions (labeled dataset position; stored ordinal shown separately); exact artifact keys pinned in tests; per-file missing errors; final assets location per SOP artifact 12.

No design change → per the cycle-8 precedent for APPROVE-WITH-CHANGES, no plan re-review; the cycle code reviews (briefed on this plan) verify the implementations.
