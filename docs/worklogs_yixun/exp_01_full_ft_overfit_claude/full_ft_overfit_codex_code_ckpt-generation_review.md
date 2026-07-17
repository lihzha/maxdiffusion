# Code review: exp_01 full_ft_overfit — round ckpt-generation
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-17

## Context loaded

- Read `experiment_SOP.md`, the driving query, approved plan §§2.3–4, and worklog through the round-4 write entry.
- Read all three prior code reviews and their completed strengthening records.
- Inspected `git status --short`, the complete uncommitted diff, full generator, and both new test files.
- Mechanically compared the side-adapter builder, restore, rollout, pin, and metrics blocks against `HEAD`; all are exact apart from dispatcher nesting and indentation.
- Inspected the full-FT trainer, checkpoint save format, rollout helpers, pyconfig path derivation, and installed Orbax 0.12.1 restore semantics.
- Confirmed no round-5 config or launcher changes are present.
- Ran `git diff --check`, Ruff, and in-memory compilation successfully.
- Attempted the exact 65-test pytest command, but the read-only sandbox had no writable temporary directory, preventing collection; the worklog records the Coder’s 65/65 run.

## Adjudications

(a) **ACCEPTED** — gate step-0 bypass and exact-step validation to full-FT; sharing them would silently change the established adapter meaning of `checkpoint_step=0` from latest checkpoint to pretrained weights.

(b) **ACCEPTED** — no split required; the 183-net-line production delta remains below the SOP’s general 200-line guideline and is one cohesive unit dominated by relocation and test seams.

(c) **CHANGE-ORDERED** — `_command.md` remains mandatory, but the full-FT `config.json` must also conditionally record `model_type` and resolved `validation_ordinals`; it currently records an ignored `validation_start_index` and misleading adapter metadata instead.

(d) **ACCEPTED** — 0-based physical dataset positions match the existing `validation_start_index` coordinate; listed order, duplicates, deterministic seed assignment, and max-position early stopping form a sound cohort contract.

(e) **ACCEPTED** — the single traced call is valid evidence of which forward the `fori_loop` body contains, though it is branch-selection proof rather than a literal runtime call-count assertion.

## Verdict

**REQUEST-REVISION.** The adapter hard rule passes and the full-FT rollout/Orbax target structures are substantively correct. The binding state-step restore and production dispatcher proofs are incomplete, so this round cannot close yet.

## Findings

1. **F1 — MAJOR:** `_restore_checkpoint_state` restores only `params` and `opt_state` into the state at lines 396–397; `FullFTTrainState.step` remains its initial value. The round-trip test’s “Step restored” assertion checks only the separately returned scalar, so it misses this violation of the binding round-3 note. In full-FT/cohort mode, replace `state.step` with the restored JSON step and assert both `restored.step == N` and the returned step equals `N`, while preserving adapter state behavior.

2. **F2 — MAJOR:** No test executes `_restore_validation_state` or `_build_full_ft_validation_state`. Consequently, changing line 410 to `cohort_mode=False` would make production step 0 restore latest while every new test stayed green, and a wrong production state builder could likewise evade the Orbax test’s manually constructed `FullFTTrainState`. Add behavioral tests that exercise the full-FT builder with a stub transformer and the top-level dispatcher, proving it produces/routes a `FullFTTrainState`, passes `cohort_mode=True`, never selects the adapter builder, and preserves the manager-free step-0 path.

3. **F3 — MINOR:** Full-FT output provenance omits the cohort selector. The artifact records `validation_start_index` even when ignored, omits `model_type` and `validation_ordinals`, and writes `action_adapter_type: side_adapter`. Add full-FT-only artifact fields for `model_type`, the resolved ordered positions, and preferably their seed-assignment order; leave the adapter artifact byte-identical.

4. **F4 — MINOR:** An empty full-FT checkpoint directory raises `No adapter checkpoints found`, and the test accepts that mislabeled message. Make the cohort-mode error say `No full-FT checkpoints found` while retaining the exact adapter message outside cohort mode, and tighten the test accordingly.

## Notes for round 5

- Create `base_wan_5b_full_ft.yml` by copying the complete side-adapter base so pipeline, mesh, dtype, TFRecord geometry, action parsing, optimizer, scheduler, decode, and validation compatibility fields remain available.
- Required deltas: `model_type: FULL_FT_TI2V`, `side_adapter_guide_scale: 1.0`, `side_adapter_noise_mode: fresh`, `learning_rate: 1.e-5`, `max_train_steps: 10000`, `checkpoint_every: 2500`, `checkpoint_keep_period: 2500`, train-split `eval_data_dir`, full-FT `output_dir`, and `validation_ordinals: ''`.
- Retain `weights_dtype: bfloat16`, `activations_dtype: bfloat16`, `scan_layers: false`, `side_adapter_sampling_steps: 25`, `flow_shift: 5.0`, `flow_sigma_min: 0.0`, `flow_sigma_max: 1.0`, `side_adapter_t_sampling: uniform`, all logical/data sharding rules, latent dimensions `48×9×12×20`, and action dimensions `32×7`.
- Retain checkpoint/evaluation keys: `run_name`, `checkpoint_dir`, `checkpoint_step`, `num_eval_videos`, `validation_start_index`, `validation_seed`, `validation_output_dir`, `seed`, and `fps`.
- Runtime defaults are: missing/empty `validation_ordinals` → contiguous selection; `checkpoint_step=0` → full-FT pretrained baseline; positive step → exact retained checkpoint; negative step → latest; empty `checkpoint_dir` → pyconfig derives `<output_dir>/<run_name>/checkpoints`.
- `train_wan_full_ft.sh` must invoke `train_wan.py` with the full-FT yml and expose `LEARNING_RATE`, `MAX_TRAIN_STEPS`, `CHECKPOINT_EVERY`, `CHECKPOINT_KEEP_PERIOD`, `PER_DEVICE_BATCH_SIZE`, `GLOBAL_BATCH_SIZE_TO_TRAIN_ON`, and `GLOBAL_BATCH_SIZE_TO_LOAD`.
- The wrapper defaults must include `SIDE_ADAPTER_NOISE_MODE=fresh`, `EVAL_DATA_DIR="${EVAL_DATA_DIR:-$TRAIN_DATA_DIR}"`, `LEARNING_RATE=1e-5`, `MAX_TRAIN_STEPS=10000`, `CHECKPOINT_EVERY=2500`, and `CHECKPOINT_KEEP_PERIOD=2500`.
- Add a `full_ft` arm to `launch_wan_train.sh` selecting the full-FT output root, W&B project, run-name tag, train script, fresh noise, train-split evaluation, and the 2500-step checkpoint cadence.
- Ensure the launcher’s current common `CHECKPOINT_EVERY=100`, `CHECKPOINT_KEEP_PERIOD=1000`, val-split `EVAL_DATA_DIR`, and hard-coded `train_wan_side_adapter.sh` do not overwrite the full-FT arm.
- Cohort validation must invoke `generate_wan_side_adapter.py` with the full-FT yml plus identical `eval_data_dir`, ordered `validation_ordinals`, `validation_seed`, and `validation_output_dir` for steps 0/2500/5000/7500/10000.
- The existing `validate_wan_side_adapter.sh` hard-codes the adapter yml and lacks `validation_ordinals`; use an exact direct command recorded in `_command.md` or add a separate full-FT validation wrapper without altering adapter defaults.

---

# Strengthening record (Coder, same round — 2026-07-17)

- **F1 (MAJOR) — FIXED.** Cohort-mode restore now writes the restored JSON step into the state (`state.replace(step=restored_step)`, adapter path untouched); test asserts pre-restore step 0 (non-vacuity), `restored.step == 7`, and returned scalar == 7. Step-drop mutant fails `assert 0 == 7`.
- **F2 (MAJOR) — FIXED.** Three production-path behavioral tests: full-FT builder (stub transformer behind the `_load_wan_pipeline` seam; `_build_adapters` booby-trapped) produces a genuine `FullFTTrainState` that drives the rollout; dispatcher full-FT arm selects the full-FT builder AND the manager-free step-0 bypass holds through the production call site (`ocp.CheckpointManager` construction booby-trapped); adapter arm still selects the adapter builder with `cohort_mode=False`. The exact production `cohort_mode=False` mutant and the builder-swap mutant both caught.
- **F3 (MINOR, change-ordered) — FIXED.** Artifact extracted to `_validation_config_artifact`: adapter modes byte-identical (tested as dicts AND through the same JSON serialization, incl. no-`model_type`-attr configs); FULL_FT_TI2V drops `action_adapter_type`, records `model_type` + resolved ordered `validation_ordinals` (list order == per-sample seed-assignment order, documented), omits `validation_start_index` when ordinals govern (kept for contiguous fallback). Field-drop mutant fails full-FT tests while adapter byte-identity stays green.
- **F4 (MINOR) — FIXED.** Cohort-mode empty-dir error: `No full-FT checkpoints found in {ckpt_dir}`; adapter message byte-identical outside cohort mode; both pinned by exact-string tests. Mislabel mutant caught.
- **Process disclosure:** first M7 revert attempt aborted (over-broad match string), briefly leaving M7 applied while M8–M10 ran (each still failed distinctively); baseline restored sha256-verified and M7–M10 re-run cleanly — reported table is from the clean run.
- **Beyond findings:** none. Orchestrator independently re-ran the suite: 73/73 green.

**Cycle 4 closed:** write → review (attempt 1 died on OpenAI capacity @199k tokens — infra, retried same reviewer; REQUEST-REVISION F1–F4; adjudications a/b/d/e ACCEPTED, c CHANGE-ORDERED) → strengthen (4 FIXED, 5/5 mutants incl. the required production cohort_mode mutant) → commit. Final: 73/73 green; net source +216 (findings-mandated growth from +183).
