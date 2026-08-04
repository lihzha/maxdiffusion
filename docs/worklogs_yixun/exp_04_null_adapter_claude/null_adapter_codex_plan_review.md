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
