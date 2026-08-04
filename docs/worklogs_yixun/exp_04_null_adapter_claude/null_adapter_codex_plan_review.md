# null_adapter — Codex plan review (plan v1)

Reviewer: OpenAI Codex `gpt-5.6-sol`, reasoning effort xhigh (codex-cli 0.146.0, session 019fcab8-6960-7cc3-a0d4-60d38d6d4305, read-only sandbox, workdir = exp_04 worktree). Artifact under review: `plan_null_adapter.md` v1 @ commit `7583f6c`. Dispatched 2026-08-04 by the Planner with the SOP-mandated briefing (prompt archived in session scratchpad).

## Reviewer output (verbatim)

Context loaded:

- `docs/worklogs_yixun/experiment_SOP.md` — full file.
- `docs/worklogs_yixun/announcement/01_status_block_every_reply.md` and `02_tpu_run_requires_approval.md` — full files.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_yixun_query.md` — full file.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/plan_null_adapter.md` — full plan v1.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_worklog.md` — full file.
- `docs/worklogs_yixun/master_experiment_tracker.md` — full file.
- `docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_analysis.md` — full file.
- `docs/worklogs_yixun/exp_03_rollout_objective_claude/plan_rollout_objective.md` — full file.
- `inverse_DDIM_pdf.md` — null-text/DDIM definitions and Algorithm 1, lines 439–704.
- Pinned Wan2.2 commit `f370228` from the local submodule Git object store—the worktree itself is uninitialized:
  - `scripts/embedding_search.py` — CLI/context initialization plus requested functions/ranges: 91–305, 374–400, 488–678, 791–819, 856–1148.
  - `scripts/verify_reconstruction_from_null.py` — full file.
  - `scripts/embedding_search_smoke.py` — full file.
  - `docs/adaptor_design.md` — §§3.11–3.14 and the earlier fresh-noise basin evidence.
  - `wan/utils/fm_solvers_unipc.py` — sigma construction and `set_timesteps`, lines 80–215.
- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py` — full file.
- `src/maxdiffusion/models/wan/side_adapter_wan.py` — full file.
- `src/maxdiffusion/generate_wan_side_adapter.py` — full file.
- `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml` — full file.
- Supporting implementation context:
  - `src/maxdiffusion/pipelines/wan/wan_pipeline.py` — T5 padding/encoding path, lines 463–526.
  - `src/maxdiffusion/models/wan/transformers/transformer_wan.py` — model initialization, 2-D timestep path, remat/scan path, lines 469–760.
  - `src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py` — record schema/order/writer, lines 165–415.
  - `bash_scripts/della_convert_wan_side_adapter_droid.sh` — full inspected conversion workflow.
  - `docs/side_adaptor.md` — dataset counts/schema, lines 90–125.
  - `src/maxdiffusion/tests/wan_transformer_test.py` — relevant Wan smoke coverage, lines 240–325.

1. **MAJOR — §§1–2 and §4/P1: the plan overstates finite optimization as a capacity upper bound.** A1 with `L=16`, 10 Adam iterations, one LR, a greedy pivot-tracking objective, and bf16 forwards is an achieved oracle under one recipe—not an upper bound on conditioning capacity. A3 with 300 finite optimizer iterations is likewise not a “deployment-objective ceiling.” If these optimizers fail, the plan currently permits a misleading “capacity result only” conclusion.

   **Recommended change:** Rename these throughout as “per-example optimized oracle” or “achieved reconstruction bound.” Before interpreting G1 failure, require a small predeclared optimization-adequacy probe over several `inner_iters`/LR settings, with convergence and gradient diagnostics. Scope any negative conclusion to the tested token count, optimizer, schedule, and budget.

2. **MAJOR — §§3 and 8: A1 is not Mokady-faithful in the key conditioning setup.** The PyTorch reference rejects empty `--heuristic` for null inversion, keeps a distinct non-empty conditional prompt, and optimizes the natural-length `T5("")` result. Its `--L_opt=16` does not configure `run_null_inversion`. Plan v1 instead sets both branches initially to padded 512-row `T5("")`, replaces 16 rows, and calls the result “Mokady-faithful.” The recurrence and CFG algebra are faithful; the conditioning experiment is a substantial, necessary local variant.

   **Recommended change:** Name A1 “empty-positive null-branch inversion.” Document separately: loss of the source caption, 512-row zero padding versus natural-length reference contexts, and the added `L_null=16` capacity. Include a cheap natural-null-length versus `L=16` P1 ablation on the optimization subset, or justify rejecting it after measurement. Test equality of the two branches after the exact fp32→bf16 boundary, not merely before casting.

3. **MAJOR — §§2 and 4/P1: the “VAE ceiling” and cited SSIM values are not comparable.** The planned pixel metric compares `decode(z_pred)` with `decode(z_video)`, whose exact latent-reconstruction ceiling is SSIM 1.0. The cited PyTorch 0.87 versus 0.96 result compared reconstruction and VAE round-trip against original RGB. Calling both a VAE ceiling conflates different references; the asserted lower≤result≤upper ordering is also not guaranteed for a finite optimizer.

   **Recommended change:** Label the cached-data metric “decoded-latent GT reconstruction” with ceiling 1.0. If source RGB is recoverable from metadata, report a separate raw-RGB ladder. Make non-pinned latent MSE and future-frame-only SSIM primary, with full-frame SSIM secondary, and do not calibrate G1’s 0.85 threshold using the PyTorch 0.87 number.

4. **MAJOR — §§4/P1–P3: the cohort design leaks method selection into final evaluation and may be highly correlated.** P1 selects the arm using the first 64 validation windows, P2 caches validation targets, and P3 declares those same 64 as its final cohort. That evaluates on the data used to select A1 versus A2 and possibly `L_null`. “First windows” can also contain adjacent windows from very few episodes.

   **Recommended change:** Create immutable, name/ordinal-hashed, episode-stratified manifests before P1: a development cohort for arm/hyperparameter selection and a disjoint locked test cohort for G3. Prefer one window per episode or explicitly cap windows per episode. All G3 methods use the locked test cohort; its oracle targets may be computed in advance but must not be inspected during selection. If only a subset is affordable, call the result a pilot and predeclare the later full-val or broader confirmation needed for a DROID-wide claim.

5. **MAJOR — §4/P1, Gate G1: A2 has no matched control and “G1-equivalents” is undefined.** A0 starts from each clip’s inverted `traj[0]`; A2 starts from shared `ε₀`. Comparing A2 against A0 cannot isolate whether A2 null optimization improves generation from its deployment start. Two fresh-noise seeds and the relative threshold `0.7·A1` also permit caching targets with poor absolute deployment performance.

   **Recommended change:** Add A2-0: frozen-null rollout from the exact shared `ε₀`. Define a separate G2 using paired A2-versus-A2-0 improvement, an absolute deployment-quality floor, and a minimum fraction of examples improved. Evaluate fresh-noise transfer over a predeclared multi-seed set keyed stably by example; decode the entire gate cohort, since VAE decode is cheap relative to inversion.

6. **MAJOR — §4/P1 and P3, Gates G1/G3: the gates are not statistically or operationally well-posed.** G1 mixes median latent MSE over the cohort with mean SSIM over only eight designated examples. G3 uses undefined “decisively,” then defines success as any positive difference in mean SSIM. Neither gives uncertainty, seed aggregation, a practical-effect margin, or a failure rule for heterogeneous examples.

   **Recommended change:** Define exact primary metrics, aggregation unit, paired effect, bootstrap CI, seed set, practical margin, and missing/nonfinite-artifact handling. Gate on non-pinned/future-only metrics and report full-frame metrics. Require coverage and a predeclared fraction-improved condition so a small mean gain cannot conceal widespread regressions.

7. **MAJOR — §§4/P2 and 5.2/5.5: the cached training dataset is incomplete as specified.** Each P2 record contains the target nulls and name but not the adapter inputs `z_i0` and `actions`; the proposed trainer merely “reads P2 artifacts.” Joining two independently sharded datasets by iteration order is unsafe, while an unspecified name join is a substantial missing implementation.

   **Recommended change:** Make each P2 record self-contained with at least `name`, source `ordinal`, `z_i0`, `actions`, null target, arm/noise convention, and optional `z_video` for oracle evaluation. Otherwise specify a fail-closed keyed join with uniqueness and exact coverage tests. Bind the cache to source-manifest hash, source split, code/model revision, sigma vector, CFG scale, base-context fingerprint, and optimization config.

8. **MAJOR — §§4/P2 and 7: cache integrity and fp16 fidelity lack a gate.** The plan says “skip completed shards” but does not define atomic completion, checksums, duplicate/coverage validation, or behavior after a partial GCS write. It also compresses fp32 optimized nulls—and A1 `z_init`—to fp16 without verifying reconstruction parity. This can silently spend J3 on corrupt or degraded labels.

   **Recommended change:** Publish shards through verified staging plus completion metadata; skip only shards whose checksum, count, schema, and config fingerprint validate. Require unique names and exact manifest coverage. Before the train build, replay fp32 in-memory versus serialized-fp16 artifacts and gate maximum latent/SSIM degradation. Preserve fp32 if fp16 fails that gate.

9. **MAJOR — §§4/P3 and 5.4–5.5: adapter training is under-specified and lacks a real-target learnability gate.** “20–50k steps,” unspecified batch/LR/weight decay/schedule/checkpoint selection, and an architecture left partly to the Coder permit post-hoc recipe choice. Synthetic loss reduction does not establish that the network can fit real optimized nulls or that embedding MSE preserves rollout quality.

   **Recommended change:** Specify the image tokenization/positions, attention heads, FFN width, normalization, dropout, parameter dtype, optimizer, batch, fixed step budget or development-only stopping rule, initialization seeds, and checkpoint-selection rule. Add P3a: overfit a small real cached cohort, first to a predeclared embedding-error threshold and then to rollout performance close to the serialized oracle, before full J3. Acknowledge/test that a zero-initialized final head gives upstream layers zero gradient on the first update.

10. **MAJOR — §§4/P3 and 5.6: the baseline comparison is not yet noise- or evaluator-matched.** Current `generate_wan_side_adapter.py` samples fresh noise by sequential RNG splitting. A2 instead uses one shared `ε₀`. Re-evaluating pre_context on the same examples but different initial latents is not paired. Editing this evaluator also risks invalidating the stored four-sample 0.2946 anchor.

    **Recommended change:** Key initial noise by `(method-independent seed, stable name/ordinal)` and feed the identical `z_start` to null-only, oracle, proposed adapter, and pre_context; for A2, all baselines must receive the same shared `ε₀`. First reproduce the original four baseline samples within a declared tolerance at the post-change evaluator commit. Define G3 as an achieved-quality comparison, not a controlled causal attribution—the methods have different architectures, target construction, and training exposure.

11. **MAJOR — §§4/P1, 9, and 10: A3 feasibility and J1 cost are not credible enough for approval.** Exp_03 established a remat/scan pattern for a two-step differentiable rollout and recorded B/C cost and HBM problems; it did not prove a 25-step, 5B-backbone rollout. A3 costs roughly 25 differentiated model calls per iteration × 300 × 8 examples, with rematerialized backward work. Folding it into a 1.5–3-hour J1 estimate risks making A3 dominate or fail the entire capacity job.

    **Recommended change:** Move A3 to a separately approved conditional job. First compile and execute one optimizer update for one example, record compile time, step time, peak HBM, and sharding, then estimate the requested scale from measurement. Set hard stop budgets. Do not describe exp_03 as proving full-horizon feasibility.

12. **MAJOR — §§5.1 and 10: the JAX batching/sharding contract is too vague.** “Vmapped/jitted natively” leaves open `vmap(grad)` over the 5B model, accidental transformer replication, batch-averaged gradients that obscure failed examples, and unsafe donation. Planned losses `[N,J]` discard per-example convergence information.

    **Recommended change:** Specify a single batched transformer call with null parameters `[B,L,D]`, independent Adam moments per example, and per-example losses `[N,J,B]`; prohibit vmapping the full model gradient. Add B=1 versus B=2 independence/cross-talk tests. Declare transformer/null/latent shardings, assert no large replicated leaves, log HBM, and donate only evolving latent/optimizer buffers—not frozen model state or artifacts reused by another arm.

13. **MAJOR — §5.9/P0: two central proposed tests assert properties the algorithm does not guarantee.** Reverse Euler evaluates velocity at a different state from forward Euler, so a random tiny Wan inversion→replay is not an exact round trip. Adam loss also need not decrease monotonically at every iteration on a random model. Such tests will be flaky or encourage implementation changes that depart from the reference.

    **Recommended change:** Use a constant or analytically tractable velocity oracle to pin recurrence indices and signs, and compare the scan against a literal Python reference loop elementwise. Test optimization on a controlled convex toy by requiring final/applied loss below initial, not monotonicity. Add explicit tests for the final post-Adam forward, `[N,B]↔[B,N]` layout, A3 gradients/pinning, fixed-noise batch-size invariance, all pin points, strict sigma validation, guide-scale validation, and the actual tiny-Wan 2-D timestep route.

14. **MAJOR — §§5.2 and 8: the independent replay contract from `verify_reconstruction_from_null.py` is missing.** An I/O round trip does not prove that an artifact alone can regenerate its claimed output. The planned artifact omits several replay-defining values and no verifier is listed.

    **Recommended change:** Add a replay-only verifier that consumes the source `z_i0`/artifact but never `z_video` or the inversion trajectory. Store or fingerprint the exact `z_start`, nulls, sigmas, CFG, context, model revision, dtype policy, and expected final latent. Require numerical replay equivalence and fail closed on provenance/config mismatch.

15. **MAJOR — §§5.5–5.7: checkpoint and evaluator wiring are incomplete and untested.** The existing generator is hardwired to `WanTI2VSideAdapterTrainer`, its `TrainState`, and side-adapter checkpoint structure. A config mode alone cannot safely restore a different null-adapter graph. The planned tests omit a real checkpoint save/restore and wrong-checkpoint rejection.

    **Recommended change:** Either create a null-specific evaluator or implement typed restore dispatch. Save architecture, `model_type`, target-cache manifest hash, base-model revision, and noise convention with the checkpoint. Add save/restore equivalence, wrong-model/config rejection, and a characterization test proving existing side/pre_context rollout behavior remains unchanged.

16. **MINOR — §§4/P1 and 10: Gaussianity statistics include the deliberately pinned frame.** The full `z_init` cannot be expected to be standard Gaussian when one ninth of its temporal latent is replaced by `z_i0`.

    **Recommended change:** Report mean/std and tail diagnostics separately for future/unpinned elements; retain full-tensor values only for comparison with the reference’s logging convention.

17. **MINOR — §6: several planned rounds are not plausibly below the SOP’s 200-line guideline.** `runner-config-io+bash`, trainer dispatch, and evaluator wiring each combine multiple contracts; R5 also lacks a named TDD test in §5.9.

    **Recommended change:** Split runner parsing, artifact schema/resume, launch wrapper, trainer/checkpointing, and evaluator restore/rollout into separate closed review cycles, each with its own red test and marker.

Status:

- No subprocesses or jobs are running; this read-only review is complete.
- Earliest steer: now. The next process gate should be Planner revisions and a full re-review before user approval.
- **REQUEST-REVISION — The recurrence is mostly faithful, but unresolved cohort leakage, unmatched controls, incomplete cache/evaluator contracts, under-specified gates, and an unproven A3 cost path could invalidate conclusions or waste substantial TPU compute.**

---

## Planner resolutions (plan v1 → v2)

All 17 findings **accepted**; none rejected. Plan v2 (same file, committed with this record) implements them as follows:

1. **F1 accepted** — renamed throughout to "per-example optimized oracle / achieved reconstruction bound" (§2 Q1); optimization-adequacy probe (J∈{10,25,50} × lr∈{1e-2,3e-2}, 8 DEV examples, plateau rule) added inside J1 and made a precondition for interpreting G1 failure; negative conclusions scoped to the tested recipe.
2. **F2 accepted** — A1 renamed "empty-positive null-branch inversion (Mokady-style)"; deviations register in §8 (empty positive branch, padded-512 vs natural length, σ₀); L_null ∈ {L_nat, 16} ablation added to J1; bf16-boundary branch-equality test added (§3, §5.11).
3. **F3 accepted** — pixel metric relabeled "decoded-latent GT reconstruction" (ceiling 1.0 by construction); PyTorch 0.87/0.96 declared non-comparable and excluded from gate calibration; primary metrics switched to future-frame (non-pinned) latent MSE + future-frame SSIM, full-frame secondary (§4-P1 Metrics).
4. **F4 accepted** — episode-stratified immutable DEV-64/TEST-64/TRAINFIT-16 manifests built before P1 (one window per episode, sha256 ordering, committed + mirrored); selection confined to DEV; G3 on TEST only; pilot-scope statement + decision point 5 (§4 Cohort manifests).
5. **F5 accepted** — A2-0 matched control (frozen ∅ from the same ε₀) added; G2 defined as paired A2-vs-A2-0 with absolute floor; multi-seed probes keyed by (2026, sha256(name), k); entire gate cohort decoded (§4-P1).
6. **F6 accepted** — G1/G2/G3 fully specified: paired per-example ratios, median + fraction-improved conditions, 10k-resample paired bootstrap CIs, practical margins (+0.02 vs pre_context, +0.05 vs null-only), fail-closed nonfinite handling (§4).
7. **F7 accepted** — P2 records made self-contained (name, ordinal, split, z_i0, actions, z_video, nulls, z_start, arm/noise convention, losses) — no cross-dataset join; shard provenance header binds manifest hash, code SHA, model revision, σ vector, w, base-context fingerprint, opt config (§4-P2).
8. **F8 accepted** — staging-prefix publish with completion markers (count+sha256+fingerprint), validated-marker-only resume, unique-name + exact-coverage checks; fp16 fidelity gate (8-example serialized-vs-fp32 replay, ΔSSIM ≤ 0.01 / ΔMSE ≤ 5%, else fp32) before the train build (§4-P2).
9. **F9 accepted** — adapter architecture and training recipe fully pinned (tokenizations, positions, heads, FFN, norms, dropout, dtype, AdamW hyperparameters, batch 256, fixed 30k budget, checkpoint-selection rule); P3a real-target learnability gate (32-example overfit: embedding-MSE and rollout-vs-oracle thresholds); zero-init two-step gradient behavior acknowledged + tested (§4-P3).
10. **F10 accepted** — noise keyed by (2026, sha256(name), k), identical z_start across methods (ε₀ for all under the A2 convention); anchor preservation: unchanged-old-script 4-sample replication within |ΔSSIM| ≤ 0.01 + cross-evaluator check before J4/J5 count; G3 reframed as achieved-quality comparison (§4-P3 Eval).
11. **F11 accepted** — A3 moved to conditional J1b, separately approved, sized from an in-J1 single-update measurement (compile/step/HBM) with hard budget stops; exp_03 precedent explicitly scoped to 2-step unrolls (§4-P1b, §9).
12. **F12 accepted** — batching/sharding contract in §3: single batched call, Σ-over-batch of per-example means, per-example Adam moments, losses [N,J,B], no vmap(grad) over the model, FSDP shardings asserted, restricted donation; B=1-vs-B=2 cross-talk and layout tests.
13. **F13 accepted** — P0 tests redesigned: constant/analytic velocity oracle + scan≡python-loop elementwise for recurrence pinning; convex-toy final<initial (no monotonicity); added locked-∅ advance, layout, pin-point, σ/guide-scale validation, tiny-Wan 2-D timestep-route, batch-independence, A3-gradient tests (§4-P0, §5.11).
14. **F14 accepted** — verify_replay mode: consumes only the published artifact, replays from stored z_start with fingerprint-checked context, asserts stored-final-latent equivalence, fails closed on any provenance mismatch; never reads z_video/trajectory (§5).
15. **F15 accepted** — `generate_wan_side_adapter.py` left untouched; new `generate_wan_null_adapter.py` owns metadata-checked restore (wrong-model/config rejection), modes adapter/oracle/null_only/pre_context; checkpoint metadata.json carries model_type, arch, cache hash, model revision, noise convention; save/restore + rejection tests (§5.9, §5.11).
16. **F16 accepted** — Gaussianity stats on unpinned elements primary, full-tensor secondary for reference comparability (§4-P1 Metrics).
17. **F17 accepted** — rounds re-split into R1–R11, each a single contract with named tests, incl. a dedicated A3 round (§6).

Material revision ⇒ full re-review of plan v2 dispatched per SOP before user approval.

---

## Re-review pass 2 (plan v2 @ `58c14dd`)

Reviewer: Codex `gpt-5.6-sol` xhigh, session 019fcacb-6788-7cb2-9bcf-e6e3d0c525cd, read-only, exp_04 worktree.

### Reviewer output (verbatim)

Context loaded:

- `docs/worklogs_yixun/experiment_SOP.md` — full file.
- `docs/worklogs_yixun/announcement/01_status_block_every_reply.md` and `02_tpu_run_requires_approval.md` — full files.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_yixun_query.md` — full file.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/plan_null_adapter.md` — full plan v2 at commit `58c14dd`.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_plan_review.md` — full v1 review and Planner resolutions.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_worklog.md` — full file.
- Wan2.2 submodule commit `f370228` from its local Git object store:
  - `scripts/embedding_search.py` — CLI/validation, context construction, inversion/null-optimization recurrence, artifacts, replay, and metrics paths.
  - `scripts/verify_reconstruction_from_null.py` — full file.
  - `scripts/embedding_search_smoke.py` — full file.
  - `docs/adaptor_design.md` — §§3.11–3.14.
- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py` — null-context, sharding, checkpoint, restore, and training paths.
- `src/maxdiffusion/models/wan/side_adapter_wan.py` — action encoding, pre_context, pinning/timestep, dispatch, and sigma paths.
- `src/maxdiffusion/generate_wan_side_adapter.py` — sample reading, rollout, RNG, restore, metrics, and aggregation paths.
- `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml` — relevant model/data/mesh/optimizer/sampler keys.
- `docs/worklogs_yixun/master_experiment_tracker.md` — full file.
- Supporting verification context: T5 padding in `wan_pipeline.py`, TFRecord identity/schema in `wan_side_adapter_droid_cache_to_tfrecord.py`, and dataset counts/schema in `docs/side_adaptor.md`.
- Repository state: clean worktree; plan v2/review resolution commit `58c14dd`.

## F1–F17 verification

| Finding | Status | Verification |
|---|---|---|
| F1 | PARTIALLY-RESOLVED | Honest oracle naming, recipe scoping, and the J/LR probe are present, but the requested gradient diagnostics and rule to rerun/select an adequate primary recipe are absent. |
| F2 | RESOLVED | A1 is renamed, all three deviations are registered, the natural-length/16-token ablation is included, and equality is tested after bf16 casting. |
| F3 | RESOLVED | Metrics are correctly reframed around decoded latent GT; future-only metrics are primary and the PyTorch raw-RGB numbers are excluded from calibration. |
| F4 | RESOLVED | DEV and TEST are locked, episode-disjoint cohorts; selection is DEV-only and final G3 is TEST-only with pilot-scope language. |
| F5 | PARTIALLY-RESOLVED | A2-0, its absolute/fraction gate, and full-cohort decoding are present, but A2 fresh-noise transfer uses only `k=2`, not the claimed multi-seed probe. |
| F6 | PARTIALLY-RESOLVED | G1/G3 are substantially improved, but G2 has no CI condition, G3 does not fix its `k` set, and “worst rank” does not define numeric mean/CI handling for invalid pairs. |
| F7 | RESOLVED | P2 records contain the trainer inputs and targets, while the shard header binds the named provenance/configuration fields. |
| F8 | RESOLVED | Validated completion markers, checksum/schema/fingerprint resume, coverage checks, and fp16 fallback are all specified. |
| F9 | PARTIALLY-RESOLVED | The model/trainer are much more concrete and P3a exists, but initialization seeds, several block details, and P3a’s maximum budget/reset contract remain unspecified. |
| F10 | PARTIALLY-RESOLVED | Same-start evaluation and achieved-quality framing are intended, but the A2 noise convention conflicts with name-keyed noise and the cross-evaluator check deliberately uses different noise without a pass criterion. |
| F11 | PARTIALLY-RESOLVED | A3 is conditional and measurement-first, but “hard budget stop” has no numerical compile/time/HBM limits. |
| F12 | RESOLVED | The single batched call, per-example Adam state/losses, FSDP contract, donation restrictions, and cross-talk tests are specified. |
| F13 | RESOLVED | The invalid exact-roundtrip/monotonicity tests were replaced and the requested recurrence, layout, pinning, validation, batching, and A3-gradient cases are listed. |
| F14 | PARTIALLY-RESOLVED | The verifier contract is present, but P2’s declared record schema omits the `expected_final_latent` that the verifier requires. |
| F15 | PARTIALLY-RESOLVED | A separate typed evaluator, null-checkpoint metadata, and restore/rejection tests are planned, but legacy pre_context compatibility and same-input behavioral characterization remain unresolved. |
| F16 | RESOLVED | Gaussianity is primary on unpinned elements and secondary on the full tensor. |
| F17 | PARTIALLY-RESOLVED | Rounds are better split, but R5, R7, R10, and R11 still combine contracts unlikely to fit the stated `<200 LOC` cycle, and several lack dedicated runner/dispatch tests. |

## New findings

1. **N1 — MAJOR — §4 P1 basin rule:** A1 targets can be selected even when G1 fails. `A1-probe ≥ 0.7×A1` can pass when both have poor absolute quality.  
   **Recommended change:** Require `G1 pass AND A1 transfer pass`, with an absolute A1-probe floor, before selecting A1; otherwise try G2, then stop.

2. **N2 — MAJOR — §4 cohort manifests, P2, §9:** No immutable TRAIN-2000 manifest or deterministic within-episode selection rule is defined. Moreover, scanning 1,440,554 serialized training records reads over 330 GB before framing/meta overhead, yet no J0 cost or approval is listed.  
   **Recommended change:** Build or reuse a compact authoritative name/episode index; emit TRAIN-2000 explicitly; bind every manifest to source shard path, generation, size/checksum, split, name, and ordinal. If a remote scan remains necessary, make it an approved, costed J0.

3. **N3 — MAJOR — §§4 P1/P3:** A2 is alternately defined as one globally shared `ε₀` and as noise keyed by `sha256(name)`. Those are different experiments.  
   **Recommended change:** Define separate functions for global fixed noise and per-example fresh noise, fix every `k` set, store the convention in artifacts/checkpoints, and test order/batch-size invariance.

4. **N4 — MAJOR — §§4/5 revised gates and tests:** There is no planned pure gate/aggregation implementation or test. G2 lacks uncertainty, G3’s seed aggregation is unspecified, and invalid examples do not have executable mean/bootstrap semantics.  
   **Recommended change:** Add a host-only gate module and `test_null_adapter_gates.py`; predeclare bootstrap RNG seed, resampling unit, seed reduction, exact coverage, and automatic gate failure on any missing/nonfinite pair.

5. **N5 — MAJOR — §§4 P1/P3:** The `L_null` ablation has no outcome rule, while the cache is variable-`L_null` and the adapter is hardcoded to 16 queries.  
   **Recommended change:** Either declare the ablation diagnostic-only with `L=16` fixed, or predeclare selection and parameterize the cache, model, metadata, and restore path by the selected `L_null`.

6. **N6 — MAJOR — §4 P3:** P3a has no maximum steps, seed, batch/optimizer override, or statement that full training restarts from a clean initialization. The block definition also omits FFN activation, LayerNorm epsilon, initializers, and whether query self-attention exists—the stated ≈13M count depends on that distinction.  
   **Recommended change:** Pin exact module pseudocode/parameter count and all seeds; give P3a a fixed maximum budget and checkpoint rule; state whether J3 restarts or continues.

7. **N7 — MAJOR — §4 P3 Eval/§5:** The existing pre_context checkpoint contains only Orbax `params/opt_state/step`, not the proposed metadata. Additionally, a four-sample “distributional” comparison under different noise cannot validate the new evaluator.  
   **Recommended change:** Define a legacy restore contract using a pinned config plus parameter-tree fingerprint/sidecar. Compare old and new pre_context latent outputs under identical sample, RNG/`z_start`, sigmas, and params within a numerical tolerance. Store new null metadata atomically per checkpoint step.

8. **N8 — MAJOR — §4 P2/§5 replay:** The authoritative P2 schema lacks `expected_final_latent`; the fp16 gate does not state whether thresholds are maxima across all eight examples, and no fidelity-gate test is listed.  
   **Recommended change:** Add the expected latent and exact dtype/shape policy to the schema; gate on worst-example degradation over a predeclared DEV/train-only subset; add threshold/fallback tests.

9. **N9 — MINOR — §4 P3 Eval:** Under A1 fresh-noise deployment, replaying cached A1 nulls from a different name-keyed noise is not an “upper” oracle and may be outperformed by the adapter.  
   **Recommended change:** Label it “serialized-target replay,” or separately optimize nulls for each evaluation `z_start` before calling it an achieved oracle.

## Status

- Running subprocesses/jobs: none; the read-only re-review is complete.
- Earliest steer: now; safe to revise the plan before user approval.

REQUEST-REVISION — v2 closes many v1 defects, but unresolved target selection, noise semantics, gate execution, cache schema, trainer specification, and legacy-evaluator parity make it not yet ready for user approval.

### Planner resolutions (plan v2 → v3)

All pass-2 items **accepted**; plan v3 implements them:

- **F1 partial → closed:** adequacy probe now logs per-inner-iter grad-norm traces; explicit adoption rule (≥2× median improvement ⇒ adopt recipe, re-run A1/A2 DEV-wide before gating; gates evaluated once, on the adopted recipe only).
- **F5 partial → closed:** fixed k-sets — A1-probe keyed{0,1,2}, A2-probe keyed{0,1,2}, eval keyed/global{0,1,2}; seed reduction = per-example mean over the k-set, then paired aggregation.
- **F6 partial → closed:** G2 gains a CI condition (mean ≥ 0.75, CI-low ≥ 0.70); G3 k-set fixed {0,1,2} with declared seed reduction; invalid-pair semantics made executable (not-improved for fractions, excluded from mean/CI, auto-FAIL at >10% invalid) inside the new gates module.
- **F9 partial → closed (with N6):** init seeds, initializers, LayerNorm eps, GELU FFN, no query self-attention, ≈9M count derivation; P3a max 2,000 steps / batch 32 / lr 3e-4 declared; J3 restarts from clean init, P3a artifacts discarded.
- **F10 partial → closed (with N3):** named conventions `keyed(k)` / `global(k)` via fold_in(2026, sha256(name)|"GLOBAL", k); stored in every artifact/checkpoint; evaluator rejects mismatch; cross-evaluator parity now at identical z_start via RNG replication with a numeric pass criterion (final-latent max-abs ≤ 1e-2, |ΔSSIM| ≤ 0.005).
- **F11 partial → closed:** numerical stops — compile > 30 min, update > 120 s, OOM ⇒ abort measurement; J1b proposed only if projection ≤ 4 h v6e-8.
- **F14/F15 partial → closed (with N7, N8):** `expected_final_latent` (+sha256) added to the authoritative schema; legacy pre_context restore contract (pinned config + eval_shape structure match + step assertion + param-tree fingerprint sidecar); metadata written atomically per checkpoint step.
- **F17 partial → closed:** rounds re-split to R1–R15, each single-contract with named tests (gates module and noise helpers got their own rounds/tests).
- **N1 accepted:** target selection requires G1 pass AND A1-probe relative (≥0.7×) AND absolute floor (mean future-SSIM ≥ 0.70); else G2; else stop.
- **N2 accepted:** J0 manifest job added (full val scan; bounded train scan with 5k-distinct-episode early stop + hard shard cap); TRAIN-2000 manifest with ≤2/episode and deterministic within-episode rule (lowest ordinal); rows bind shard path + GCS generation id; header binds builder SHA + listing checksum.
- **N3 accepted:** see F10 closure; order/batch-size invariance tested (`test_null_adapter_noise.py`).
- **N4 accepted:** host-only `null_adapter_gates.py` + `test_null_adapter_gates.py`; bootstrap RNG seed 20260804, resampling unit = example, percentile CIs, coverage assertion, auto-fail semantics.
- **N5 accepted:** L_null ablation declared diagnostic-only; L=16 fixed for P2/P3; metadata records L_null; adapter/evaluator assert cache L_null == 16.
- **N6 accepted:** see F9 closure.
- **N7 accepted:** see F14/F15 closure; plus anchor re-run of the unchanged old script (|ΔSSIM| ≤ 0.01/sample) before J4/J5 count; §10 records the fallback (instrumented noise fingerprint from a 1-sample dry run) if RNG replication proves brittle.
- **N8 accepted:** fidelity gate thresholds are worst-example maxima over the first 8 DEV-manifest examples; threshold/fallback branches unit-tested.
- **N9 accepted:** relabeled "serialized-target replay (reference point)" under keyed deployment; achieved per-example oracle only under global(0) with A2 targets.

Material revision ⇒ re-review pass 3 dispatched.

---

## Re-review pass 3 (plan v3 @ `7bead68`)

Reviewer: Codex `gpt-5.6-sol` xhigh, session 019fcad9-8634-7640-8bb0-d0751856e055, read-only, exp_04 worktree.

### Reviewer output (verbatim)

Context loaded:

- `docs/worklogs_yixun/experiment_SOP.md` — full file.
- `docs/worklogs_yixun/announcement/01_status_block_every_reply.md` and `02_tpu_run_requires_approval.md` — full files.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_yixun_query.md` — full file.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/plan_null_adapter.md` — full v3 at `7bead68`; also the relevant v2 text at `58c14dd` and the v2→v3 diff.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_plan_review.md` — full file: both reviews and both resolution records.
- `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_worklog.md` — full file.
- `docs/worklogs_yixun/master_experiment_tracker.md`, `exp_02_overfit100_claude/overfit100_analysis.md`, and `exp_03_rollout_objective_claude/plan_rollout_objective.md` — full files.
- Wan2.2 submodule object `f370228`: targeted null-inversion/optimization/replay/artifact sections of `scripts/embedding_search.py`; full `scripts/verify_reconstruction_from_null.py`.
- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py` — relevant null-context, RNG, sharding, checkpoint, restore, and training sections.
- `src/maxdiffusion/models/wan/side_adapter_wan.py` — relevant adapter, pinning, timestep, forward-dispatch, and sigma sections.
- `src/maxdiffusion/generate_wan_side_adapter.py` — relevant sample-reading, rollout, restore, sequential-RNG, metrics, and aggregation sections.
- `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml` — full file.
- Repository state: clean worktree, HEAD `7bead6893d2d393d1f74cbd7677b215bd0a0c356`; Wan2.2 worktree uninitialized, reference read from its local Git object store.

Pass-2 item verification:

- **F1 — PARTIALLY-RESOLVED:** Gradient traces and full-DEV rerun were added, but the adoption rule does not deterministically select among multiple qualifying recipes or define the median’s aggregation.
- **F5 — RESOLVED:** A1/A2 transfer probes now use fixed keyed seeds `{0,1,2}` with declared per-example seed reduction.
- **F6 — RESOLVED:** G2 has a CI floor, G3 fixes its seeds/reduction, and invalid-pair calculations now have executable semantics.
- **F9 — RESOLVED:** Architecture, initialization, seeds, parameter count, P3a budget/override, and clean J3 restart are pinned.
- **F10 — PARTIALLY-RESOLVED:** Same-start parity is now numerical, but A2 is selected and trained at `global(0)` while deployment G3 averages `global(0..2)`.
- **F11 — RESOLVED:** Compile/update/OOM stops and the four-hour projected J1b limit provide an executable measurement gate.
- **F14 — RESOLVED:** `expected_final_latent` and its hash are authoritative artifact fields consumed by the independent verifier.
- **F15 — PARTIALLY-RESOLVED:** Typed/legacy restore and same-input parity are planned, but the exact effective pre_context restore configuration is not pinned.
- **F17 — PARTIALLY-RESOLVED:** Rounds are materially smaller, but R6/R7 and trainer dispatch still lack clearly mapped dedicated test names; this is non-blocking process polish.

- **N1 — RESOLVED:** A1 selection requires G1, the relative transfer threshold, and absolute SSIM ≥0.70.
- **N2 — PARTIALLY-RESOLVED:** J0 and TRAIN-2000 exist, but the claimed hard train-scan cap has no numeric value and exact DEV/TEST allocation was dropped.
- **N3 — PARTIALLY-RESOLVED:** Conventions are named and persisted, but the hash-to-key expression is not a valid canonical JAX contract and global evaluation conflicts with the fixed-ε₀ fallback.
- **N4 — PARTIALLY-RESOLVED:** The module/tests/statistics exist, but v3 permits up to 10% invalid pairs rather than implementing the accepted fail-on-any-invalid rule.
- **N5 — RESOLVED:** The ablation is diagnostic-only and P2/P3 are fixed to `L_null=16`.
- **N6 — RESOLVED:** Module details, seeds, P3a budget, and clean-restart behavior are pinned.
- **N7 — PARTIALLY-RESOLVED:** Structure/fingerprint/step/parity checks exist, but the legacy checkpoint URI and complete effective pre_context config remain unspecified.
- **N8 — PARTIALLY-RESOLVED:** Expected latent, worst-example thresholds, and tests were added, but the schema hardcodes fp16 while the failure branch says to store fp32 without defining which fields change dtype.
- **N9 — RESOLVED:** Keyed replay is correctly labeled a reference point; oracle language is limited to the matching `global(0)` A2 start.

New findings:

1. **M1 — MAJOR — §3 Noise conventions:** `int.from_bytes(sha256(name)[:8])` neither defines UTF-8/digest/byte order nor respects `jax.random.fold_in`’s 32-bit data contract. Define an exact mapping—e.g. UTF-8 SHA-256 digest split into two big-endian `uint32` words, each folded in—and pin golden key/noise fingerprints.

2. **M2 — MAJOR — §§3, 4/P1, 4/P3 Eval:** A2 targets and G2 are based only on `global(0)`, while G3 averages `global(0..2)` even though approval decision 4 defines the fallback as one canonical noise. Either evaluate A2 deployment only at `k={0}`, or gate global-seed transfer and cache/define targets for every evaluated `k`.

3. **M3 — MAJOR — §4/P1 Adequacy probe:** “If any recipe improves ≥2×, adopt it” is non-decidable when several recipes qualify and does not define whether the median pools steps, examples, or per-example summaries. Specify the statistic, deterministic argmin/tie-break, and worst-case J1 rerun budget.

4. **M4 — MAJOR — §4/J0:** “Never a full 330-GiB pass” is not enforced because the risk section’s hard shard-count cap has no value; v3 also omits v2’s first-64/next-64 DEV/TEST assignment. Pin a maximum shard/byte count with fail-before-next-shard behavior and the exact cohort slicing rules.

5. **M5 — MAJOR — §4/P2:** The fp16 fallback is internally contradictory: `z_start` and `expected_final_latent` are declared fp16 and `nulls` has no dtype, yet failure says “store fp32.” Add an artifact dtype field and declare exactly which replay-defining tensors switch to fp32, with reader/verifier rejection of inconsistent byte lengths.

6. **M6 — MAJOR — §4/P3 Legacy restore:** The cited base YAML currently defaults to `action_adapter_type: side_adapter` and `max_train_steps: 10000`, whereas the baseline is pre_context@30k. Pin the checkpoint URI plus the complete effective configuration/overrides—or an immutable saved-config hash—used to construct the legacy parameter tree.

7. **M7 — MAJOR — §3 Gates:** Excluding as many as six invalid TEST-64 pairs from mean/CI can let an absolute-quality gate pass after its hardest failures disappear. Implement the accepted fail-on-any-invalid-primary-metric rule, or conservatively impute the metric’s worst value in every mean/CI.

8. **M8 — NIT — header/worklog:** Pass 2 had nine partial F-items, not seven; correct the v3 header, resolution heading, and worklog count.

## Status

- No subprocesses or jobs are running; this read-only pass is complete.
- Earliest steer: now; revising these plan contracts is safe before any approval or implementation.

REQUEST-REVISION — v3 closes most prior gaps, but its key derivation, A2 deployment estimand, recipe adoption, bounded scan, artifact fallback, legacy restore, and invalid-pair rules still leave conclusions or execution materially underdetermined.


### Planner resolutions (plan v3 → v4)

All pass-3 items **accepted**; plan v4 implements them:

- **M1:** exact derivation pinned — UTF-8 sha256 digest, bytes [0:4]/[4:8] as big-endian uint32 words, folded in sequence with a domain constant and k; golden key/noise fingerprints hardcoded in `test_null_adapter_noise.py`.
- **M2:** A2 deployment estimand = the single canonical ε₀: under `global` convention J4/J5 evaluate exactly k={0} (targets exist only for ε₀); keyed convention keeps k∈{0,1,2}; A2-probe remains the reported transfer diagnostic. (Closes F10.)
- **M3:** adoption statistic = median over 8 examples of per-example mean-over-steps final tracking loss; adopt the lowest-scoring recipe among those ≤ 0.5× default, ties → lower J then lower lr; none qualifies → keep default; worst-case DEV rerun budget +2 h, exceeded → stop and surface. (Closes F1.)
- **M4:** J0 hard caps — max 200 shards AND 60 GiB, checked before each next shard, fail-closed with nothing written; DEV-64/TEST-64 slicing restored explicitly (first 64 / next 64 val episodes by ascending hex sha256(episode_id)); TRAIN-2000 = 1,000 episodes × ≤2 lowest-ordinal windows after TRAINFIT-16. (Closes N2.)
- **M5:** `latent_dtype` field added; fp32 fallback scoped to exactly {nulls, z_start, expected_final_latent}; z_i0/z_video stay source-fp16; readers/verifier derive byte lengths from the field and reject inconsistencies. (Closes N8.)
- **M6:** legacy checkpoint URI pinned (wan-pre_context-v6e64-full-gbs512-fresh-20260629-034110, step 30000); effective config = base YAML + pinned override set {action_adapter_type: pre_context, side_adapter_noise_mode: fresh} with the base-YAML defaults that must NOT apply called out; eval_shape structure guard fail-closed; fingerprint sidecar also hashes the effective-config JSON; override-set rejection test added. (Closes F15/N7.)
- **M7:** invalid pairs are worst-value imputed into every mean/CI (SSIM←0.0, MSE←max finite observed, ratio←1.0), counted not-improved, never excluded; >10% invalid still auto-fails; imputation branch unit-tested. (Closes N4.)
- **M8:** pass-2 partial count corrected to nine in the v4 header and in the worklog entry recording this pass.
- **F17 residue:** R6/R7/R13 now have named dedicated tests (`test_null_adapter_runner_capacity.py`, `test_null_adapter_runner_decode.py`, `test_null_adapter_dispatch.py`).

Material revision ⇒ re-review pass 4 dispatched.
