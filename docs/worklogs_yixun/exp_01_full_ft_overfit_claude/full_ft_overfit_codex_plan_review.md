# Plan review: exp_01 full_ft_overfit
Reviewer: OpenAI Codex gpt-5.6-sol (reasoning effort xhigh), 2026-07-16

## Context loaded

- `docs/worklogs_yixun/experiment_SOP.md` — requires reviewed, test-first implementation, cheapest-first validation, and component-level parity auditing.

- `docs/worklogs_yixun/exp_01_full_ft_overfit_claude/full_ft_overfit_yixun_query.md` — locks full-DROID training, no adapter/actions, and a fully trainable transformer conditioned by the first frame and null text.

- `docs/worklogs_yixun/exp_01_full_ft_overfit_claude/full_ft_overfit_worklog.md` — records the trainer-subclass strategy and the Planner’s CFG/FSDP concerns.

- `docs/side_adaptor.md` — establishes fresh per-example noise as mandatory and reports valid side-adapter validation SSIM up to 0.664.

- `docs/wan_ti2v_pre_context_adapter_methodology_results.md` — records the pre-context recipe, fixed-noise failure, fresh-noise training through 26,900 steps, and roughly 0.30 validation SSIM.

- `docs/worklogs_yixun/exp_01_full_ft_overfit_claude/plan_full_ft_overfit.md` — proposes the full-FT trainer, four TDD units, v6e validation ladder, and 10,000-step diagnostic.

- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py` — contains the authoritative loss, CFG stop-gradient placement, dataset loader, TrainState, Composite checkpointing, and adapter replication override.

- `src/maxdiffusion/models/wan/side_adapter_wan.py` — confirms the sigma grid, per-token timestep, first-frame pinning, and action-adapter forwarding behavior.

- `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml` — confirms fresh-noise config, guide scale 5, AdamW settings, BF16 weights, full rematerialization, and pure FSDP.

- `src/maxdiffusion/generate_wan_side_adapter.py` — confirms rollout math and the current adapter-specific state construction/restoration.

- `src/maxdiffusion/train_wan.py` — confirms model-type dispatch requires a new explicit full-FT branch.

- `bash_scripts/train_wan_side_adapter.sh` — confirms the dangerous shell default is still `SIDE_ADAPTER_NOISE_MODE=fixed`.

- `bash_scripts/launch_wan_train.sh` — confirms queue launches force fresh noise but currently force the validation data directory to the val split.

- `src/maxdiffusion/max_utils.py` — confirms warmup-to-constant scheduling, AdamW weight decay `1e-2`, and global-norm clipping before AdamW.

- `src/maxdiffusion/pyconfig.py` — confirms `checkpoint_dir` is derived symmetrically as `output_dir/run_name/checkpoints`.

- `src/maxdiffusion/pipelines/wan/wan_pipeline_ti2v_2p2.py` — confirms the TI2V pipeline supplies the transformer and mesh used by the trainer.

- `src/maxdiffusion/pipelines/wan/wan_pipeline.py` — confirms pretrained transformer parameters are cast to `weights_dtype` and loaded directly under logical mesh shardings.

- `src/maxdiffusion/models/wan/transformers/transformer_wan.py` — confirms BF16 parameter support, logical partition annotations, dropout zero, and per-layer rematerialization.

- `dependencies/requirements/base_requirements/requirements.txt` — declares Optax as the optimizer dependency.

- `dependencies/requirements/generated_requirements/requirements.txt` — constrains Optax only as `>=0.2.8`, without pinning accumulator behavior to one version.

## Verdict

REQUEST-REVISION. The basic direction—plain transformer, direct forward, fresh-noise one-step loss, and retaining FSDP shardings—is sound. The current plan nevertheless leaves a known fixed-noise failure path, overstates both parity and the evidentiary strength of a 10,000-step negative result, and lacks the integration/restore tests needed to trust this diagnostic. Those issues could produce either an invalid run or an unjustified “pipeline broken” conclusion.

## Findings

1. **F1 — BLOCKER.** Section 3 says the new wrapper will copy `train_wan_side_adapter.sh` and retain `SIDE_ADAPTER_NOISE_MODE` as a real knob. The actual reference wrapper defaults that variable to `fixed`, which overrides the YAML’s `fresh` value; this is precisely the previously diagnosed train/validation mismatch. Change the new wrapper’s default to `fresh`, make the full-FT trainer reject every mode except `fresh`, and test that both direct-wrapper and queue-launch configurations resolve to fresh noise.

2. **F2 — MAJOR.** Section 2.1 is correct that, under the current `dropout: 0.0` configuration, identical conditional/unconditional transformer calls plus `stop_gradient(v_uncond)` give the raw derivative `∂v_pred/∂θ = s·∂v_cond/∂θ`. It is not correct to call that “equivalent to 5× LR”: the code clips by global norm before AdamW, Adam normalizes using its moments, and weight decay is not scaled with the data gradient. Keep the direct single-forward path and guide-scale assertion, but describe this as a 5× pre-optimizer gradient multiplier plus wasted computation, not a 5× effective learning rate.

3. **F3 — MAJOR.** Sections 2 and 3 claim “parity by construction,” but the reference trainer keeps noisy-latent construction and masked velocity MSE inline, while the plan creates new, duplicated implementations. The proposed noisy-latent signature also omits `eps` even though the function requires it. Extract shared objective helpers and make both trainers call them, or add a differential characterization test against the exact reference equations; also add a fixed-RNG `_denoising_loss` integration test covering fresh per-example noise, timestep/sigma selection, target construction, frame pinning, null context, exactly one plain-transformer call, ignored actions, and an actual transformer-parameter update.

4. **F4 — MAJOR.** Section 2.2 says 5.12 million samples are “many epochs” over DROID. The documented train set contains 1,440,554 windows, so 10,000 steps at batch 512 are only about 3.55 passes. Keep 10,000 steps as a comparative milestone if desired, but either extend the negative-evidence budget or declare failure at 10,000 steps inconclusive; a “pipeline suspect” conclusion should require predefined LR/optimizer-precision controls and a sufficient number of train-set passes.

5. **F5 — MAJOR.** Sections 2.3 and 8 use “SSIM ≫ 0.29” as the success reference. Approximately 0.29 is the pre-context result, whereas the valid fresh-noise side-adapter run reached mean SSIM 0.664 at step 2,000 and 0.615 at step 10,000; moreover, those are val clips, not the proposed train clips. Define a fixed, diverse set of exact training ordinals and rollout seeds, evaluate the pretrained model and checkpoints 2,500/5,000/7,500/10,000 on that same cohort, and either evaluate the comparator on the same cohort or avoid a cross-split numeric threshold. The copied config’s default of one contiguous evaluation video is inadequate for this decision.

6. **F6 — MAJOR.** R3 treats BF16 Adam state as acceptable, while §8 permits a stalled run to implicate the pipeline. The loader explicitly casts transformer parameters to BF16, and `create_optimizer` does not explicitly request FP32 accumulators; accumulator precision is therefore an uncontrolled dependency/default, not a harmless parity detail for full-parameter training. Specify and log parameter/moment dtypes, predeclare an FP32-state or FP32-master-weight control before any negative verdict, and update memory/checkpoint estimates for that control. LR `1e-5`, Adam coefficients, clipping, and weight decay `1e-2` are otherwise reasonable, with only the latter settings actually established by reference parity.

7. **F7 — MAJOR.** Section 2.2 says in-training evaluation will read train shards, but the per-file YAML deltas omit `eval_data_dir`, the reference YAML points to val, and the current launcher sets and passes the val directory before experiment dispatch. Explicitly set `EVAL_DATA_DIR="$TRAIN_DATA_DIR"` in the `full_ft` launcher arm and the new wrapper/config, while requiring the final validation command to record its dataset path. Add this resolved path to the startup assertions.

8. **F8 — MAJOR.** Section 3 calls checkpoint restoration symmetric, but unit D tests only a forward-selection helper and a nominal restore branch. The Composite item names and `pyconfig` path derivation are compatible, but the current rollout merges adapter and frozen-transformer fields before entering its body; `FullFTTrainState` will not have those fields. Specify the pre-body state branch explicitly and add a tiny CPU Orbax round trip that saves deliberately changed transformer parameters, optimizer state, and step, reconstructs the validation state, restores them, and proves rollout uses the restored—not pretrained—parameters.

9. **F9 — MINOR.** The proposed full-FT `_shard_state` direction is correct: removing the reference replacement of `params` and `opt_state` with `P()` allows the pretrained transformer’s FSDP shardings to survive. A fake-tree selection test does not verify the important NNX/Optax case, however. Preserve `_apply_actual_sharding_for_tpu`, add target-hardware logging/assertions for large parameter and optimizer leaves plus global/addressable byte totals, and treat “fits comfortably” as provisional until the exact v6e-64 batch-512 fit probe passes.

10. **F10 — MINOR.** Unit C combines TrainState, complete loss/train/eval functions, trainer startup, sharding, dispatch, and tests in one nominal `<200 LOC` round. This is too broad for the SOP’s focused write-review-strengthen cycle and leaves failures hard to localize. Split it into at least denoising integration/train update, state/sharding/dispatch, and checkpoint/generation rounds, each with its own focused review.

## Questions for the Planner

- Which prior method is the intended comparator: fresh side-adapter r20, whose best reported SSIM is about 0.664, or pre-context, whose reported SSIM is about 0.30?

- Is 10,000 steps a hard compute limit, or may the negative-evidence protocol extend beyond 3.55 train-set passes?

- Is the increased checkpoint/storage budget for an explicit FP32 optimizer-state or master-weight control acceptable?

---

# Resolutions (Planner, plan v2 — 2026-07-16)

All 10 findings **accepted**; none rejected. Plan revised in place (`plan_full_ft_overfit.md` v2); section refs below are to v2.

- **F1 (BLOCKER, fixed-noise default)** — FIXED. New wrapper defaults `SIDE_ADAPTER_NOISE_MODE=fresh`; `full_ft` launcher arm passes fresh explicitly; trainer **asserts** `side_adapter_noise_mode == "fresh"`; wiring test asserts the assert; acceptance criteria require the fresh log line. (§2 table, §3, §6, R6.)
- **F2 (×5-LR mischaracterization)** — FIXED. §2.1 rewritten: 5× *pre-optimizer gradient multiplier* + wasted forward; clip-before-AdamW and Adam moment normalization noted; bypass + assert retained.
- **F3 (parity overstated / duplicated math / missing eps)** — FIXED. Objective math extracted into shared helpers in `side_adapter_wan.py` called by BOTH trainers; reference trainer refactor is behavior-preserving with a fixed-RNG characterization test (Coder round 1); `eps` is an explicit argument; fixed-RNG `_denoising_loss` integration test added covering fresh per-example noise, σ/t selection, target, pin, null context, exactly-one-forward (call-counting stub), actions-unused, and a real param update. (§3, §4, tests.)
- **F4 (epoch math / negative-evidence budget)** — FIXED. 10k @ 512 = **3.55 passes** stated; decision rule made asymmetric (positive valid, negative-at-10k inconclusive); §2.4 escalation protocol (30k steps → LR 2e-5 control → fp32-optimizer-state control) is now a precondition for any "pipeline suspect" verdict.
- **F5 (wrong comparator / cross-split threshold)** — FIXED. §2.3 cohort protocol: fixed N=16 train ordinals + fixed seeds predeclared in `_params_set_up.md`; pretrained step-0 baseline + ckpts 2500/5000/7500/10000 evaluated on the same cohort; success = within-cohort delta vs pretrained; side-adapter 0.664/0.615 and pre-context 0.30 cited as context only.
- **F6 (bf16 Adam accumulators uncontrolled)** — FIXED. Trainer logs param + Adam moment dtypes at startup (acceptance criterion); fp32-optimizer-state control predeclared as escalation #3 with memory/checkpoint estimates (≈70 GB); optax `>=0.2.8` unpinned-default risk recorded as R3/R6 context. LR/wd wording corrected (parity holds for Adam coefficients/clip/wd; LR is a full-FT choice).
- **F7 (eval_data_dir omitted / launcher forces val)** — FIXED. Yml delta sets `eval_data_dir: …/train`; wrapper defaults `EVAL_DATA_DIR="$TRAIN_DATA_DIR"`; `full_ft` launcher arm sets it too; startup log records resolved path; acceptance criterion added; validation command must record its dataset path in `_command.md`.
- **F8 (restore symmetry untested / rollout merges adapter fields)** — FIXED. Explicit full-FT branches planned in both `_restore_validation_state` and `_rollout_sample` (no adapter fields in the merge); new CPU Orbax round-trip test saves modified params/opt/step, reconstructs, restores, and proves the rollout consumes restored — not pretrained — params. (§3, round 4.)
- **F9 (shard test too weak / keep actual-sharding pass)** — FIXED. `_shard_state` override retains `_apply_actual_sharding_for_tpu`; logs global + per-host addressable byte totals for params/opt_state; "fits comfortably" downgraded to provisional-until-probe (ladder rung 6 gate).
- **F10 (unit C too broad)** — FIXED. Rounds restructured into 5 focused cycles: shared-objective-helpers / full-ft-loss / trainer-wiring / ckpt-generation / configs-launchers (§4).

Material revision → **re-review requested** (this file will gain the re-review verdict below).
