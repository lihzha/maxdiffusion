# Code review: exp_01 full_ft_overfit — round configs-launchers
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-17

## Context loaded

- Read `experiment_SOP.md`, the driving query, approved plan §§2.2–3/6–7, and worklog through the round-5 write entry.
- Read the round-4 review and checked every Notes-for-round-5 requirement.
- Inspected `git status --short`, full tracked diff, and all four implementation files plus the new test file.
- Compared the YAML against `base_wan_5b_side_adapter.yml`; only the header, seven documented value deltas, and `validation_ordinals` differ.
- Compared the training wrapper against `train_wan_side_adapter.sh` and the launcher against `HEAD` hunk-by-hunk.
- Executed historical and current launchers with fixed provenance; `pre_context` and `side_adapter` submitted argv matched exactly.
- Verified full-FT launch and smoke environments, all five cohort checkpoint invocations, `bash -n`, YAML parsing, and `git diff --check`.
- The exact 94-test command was blocked by the read-only sandbox’s lack of a temporary directory; 16/16 non-temporary round-5 tests passed, launcher behavior was exercised manually, and the worklog records the Coder’s 94/94 run.

## Round-4 requirements checklist

- **SATISFIED** — `base_wan_5b_full_ft.yml` is a complete standalone copy preserving pipeline, optimizer, data, mesh, and compatibility fields.
- **SATISFIED** — all required model, guide-scale, noise, LR, step, checkpoint, train-eval, output, and `validation_ordinals` values are present.
- **SATISFIED** — required dtype, sampling, flow, sharding, latent, and action values remain equal to the side-adapter base.
- **SATISFIED** — all specified checkpoint and evaluation compatibility keys are retained.
- **SATISFIED** — contiguous fallback, step-0 baseline, exact positive checkpoint, latest negative checkpoint, and derived empty checkpoint directory semantics are implemented.
- **SATISFIED** — `train_wan_full_ft.sh` targets the full-FT YAML and exposes every listed training and batch override.
- **SATISFIED** — wrapper defaults are fresh noise, train-split evaluation, LR 1e-5, 10k steps, and 2500/2500 checkpoint cadence.
- **SATISFIED** — the launcher’s `full_ft` arm selects the correct output root, W&B project, run tag, wrapper, noise mode, evaluation split, and cadence.
- **SATISFIED** — full-FT overrides occur after untouched common defaults and before SMOKE; existing arms retain exact submitted behavior.
- **SATISFIED** — the validation wrapper passes identical cohort path, ordinals, seed, and output arguments for steps 0/2500/5000/7500/10000.
- **SATISFIED** — a separate full-FT validation wrapper was added without altering adapter validation defaults.

## Adjudications

- (a) **ACCEPTED** — training never consumes `validation_ordinals`; the validation wrapper owns and passes it.
- (b) **ACCEPTED** — `ACTION_ADAPTER_TYPE="full-ft"` is awkwardly named but harmless, explicitly documented, and ignored by the full-FT wrapper.
- (c) **ACCEPTED** — dropping inert adapter CLI overrides avoids false provenance while retaining config-shape compatibility in the YAML.
- (d) **ACCEPTED** — unconditional keep-period preserves the full-run cohort; smoke still saves nothing because cadence is zero and final saving is false.
- (e) **ACCEPTED** — the extra guard tests remain focused on launcher safety and regression protection.

## Verdict

**REQUEST-REVISION.** The canonical queue launcher and cohort-validation paths are correct, but the explicitly required plain-YAML training command does not meet the plan’s GBS-512 or W&B startup criteria. Fix the standalone defaults or document that the plain command is not an accepted launch path.

## Findings

1. **F1 — MAJOR:** `base_wan_5b_full_ft.yml` defaults to `per_device_batch_size: 1.0` and an empty `wandb_project`; pyconfig recomputes the target batch from per-device batch size, so the stated plain command produces GBS 64—not 512—on 64 devices and disables W&B. Set standalone defaults to per-device 8 and the full-FT W&B project, with regression coverage, or require the complete launcher/override command instead of claiming the plain command satisfies §6.

## End-of-implementation gaps

F1 only. All plan §3 artifacts were delivered; trainer guards match the YAML, generator arguments match the validation wrapper, and the pre-existing `learning_rate` and `weights_dtype` keys support the §2.4 direct CLI controls.

---

# Strengthening record (Coder, same round — 2026-07-18)

- **F1 (MAJOR) — FIXED.** `base_wan_5b_full_ft.yml` now ships the primary-run recipe standalone: `per_device_batch_size: 8.0` (→ GBS 512 on v6e-64), documentary `global_batch_size_to_*: 512` kept coherent with pyconfig's unconditional recompute (`int(num_devices * per_device)` in `user_init`; per-device is the only authoritative knob — proven by executing the real `calculate_global_batch_sizes` under a patched 64-device view, permanently test-bound), and `wandb_project: maxdiffusion-wan-full-ft` pinned test-side to the exact value the launcher full_ft arm submits. Wrapper batch defaults stay smoke-scaled by documented, tested contract (always explicit CLI overrides); launcher exports 8/512/512 for real runs; full_ft SMOKE still submits explicit small batch with periodic AND final checkpointing disabled. Mutants (per-device→1.0, wandb→'', launcher-project drift, documentary-global drift) all red; restores sha256-verified. Suite 94→95 green (orchestrator-verified).
- Residual noted for `_command.md` practice: a plain-yml generate invocation would inherit per-device 8.0; the canonical cohort path (validate_wan_full_ft.sh) pins 1.0 explicitly.

**Cycle 5 closed:** write (round-4 requirements checklist: 11/11 SATISFIED; adjudications a–e ACCEPTED) → review (REQUEST-REVISION, single F1) → strengthen (FIXED). **Implementation complete: 5/5 rounds, plan §3 fully delivered, end-of-implementation gaps: none.**
