# Codex code review — sampling-steps probe round (3c85608 + 66720dc)

- **Date:** 2026-08-01
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 1 HIGH (probe output-dir override unrestricted → could write inside a step_* verdict dir), 2 MEDIUM (approved fixed experiment not enforced: arms/checkpoint/baseline; REVERT 3c85608's dead knobs), 1 LOW (H1/H2 terminology).
- **Verified:** verdict-CLI inadmissibility of the payload, preemption/retry write semantics, _ArmConfig tracing soundness, eval-path parity (restore/context/RNG/metric), 30-window golden digest, ffmpeg omission, no settled-scope changes.
- **Scientific anchor supplied:** landed seed-0 correct reference for the 30 probe windows = mean SSIM 0.8100125855 / median 0.8059329625 — probe acceptance requires row-level 25-arm agreement.

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

1. HIGH — [probe_overfit100_sampling_steps.py:137](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:137), [probe_wan_overfit100_sampling.sh:91](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/probe_wan_overfit100_sampling.sh:91): `validation_probe_output_dir` is unrestricted, so the probe can write `probe_steps_ckpt2500.json` inside a `step_*` verdict directory; remove the override or reject any noncanonical root/path component beginning `step_`, and test hostile overrides/run names.
2. MEDIUM — [probe_overfit100_sampling_steps.py:210](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:210), [probe_overfit100_sampling_steps.py:237](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:237), [probe_wan_overfit100_sampling.sh:88](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/probe_wan_overfit100_sampling.sh:88): the approved fixed experiment is not enforced—arbitrary positive arms are accepted, the baseline silently becomes `min(arms)`, and checkpoints other than 2500 are allowed; require restored step 2500 and arms exactly `{25,50,100}`, with baseline fixed to 25.
3. MEDIUM — [validate_wan_overfit100.sh:98](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/validate_wan_overfit100.sh:98), [validate_wan_overfit100.sh:249](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/validate_wan_overfit100.sh:249): revert `3c85608`; these prominently logged controls are provably ignored by this launcher’s entry point, and tests preserving dead knobs do not prevent operational misuse.
4. LOW — [probe_overfit100_sampling_steps.py:1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:1), [probe_wan_overfit100_sampling.sh:63](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/probe_wan_overfit100_sampling.sh:63): “H1 diagnostic” reverses the analysis terminology; rename it “H1/H2 discriminator” or “H2 sampling-side diagnostic.”

Verified claims:

- P1: The payload cannot currently be admitted by the verdict CLI: inputs are explicit, and this schema lacks the required 25-step aggregation provenance/role fields. The unrestricted output override nevertheless breaks write-location isolation.
- P1: No JSON is written during rollouts. Preemption before final publication leaves no probe JSON; identical completed retries are accepted, changed bytes are refused. The writer is not concurrency-safe, so confirm the prior attempt is dead before retrying.
- P2: `_ArmConfig` is sound under tracing. Each arm creates a separate jitted function with a distinct sigma shape and loop bound; the driver test catches shared/pinned config wiring and behaviorally identical arms.
- P3: Restore, batch shape, latent dtype conversion, correct context, folded RNG, rollout, VAE decode, and SSIM match the eval path. The relevant rollout/reader/context/RNG/metric functions are unchanged from the landed segment-final code.
- P3: The selected 30-window digest matches the golden value. The landed seed-0/correct reference for those windows is mean SSIM `0.8100125855`, median `0.8059329625`; accepting the probe scientifically must require row-level 25-step agreement.
- P5: Omitting ffmpeg is justified: neither auxiliary RGB, MP4 fetch/decode, nor video export is reachable. Full pinned-snapshot prefetch, local-only resolution, manifest/dataset binding, exact checkpoint restore, venv handling, and COMMIT export mirror the established launcher.
- P6: Neither commit changes the settled rollout, restore implementation, verdict statistic, role validation, staging, or publication machinery.
- Validation: source compilation, both `bash -n` checks, and `git diff --check` passed. I could not independently rerun pytest because this review sandbox provides no writable temporary directory; `1222 passed / 2 skipped` remains commit-record evidence.

**Status / launch fitness**

- Probe job: **NOT FIT TO LAUNCH** until findings 1–3 are fixed, focused tests/mutations rerun, the resulting SHA is pushed (the current two commits are ahead of `origin`), and the launch command/acceptance criteria are recorded.
- After re-review: 1 × v6e-8, approximately 55 minutes, no resume is reasonable. Under poor spot availability, `PROBE_NUM_WINDOWS=15` is an acceptable predeclared fallback, provided the prior attempt is confirmed terminated and no completed JSON occupies the immutable path.
- Job 23: the worklog records the 10k extension as launched; live status was not queried or changed.
- Review subprocess: complete; Yixun can steer immediately.
tokens used
193,755
REQUEST-REVISION

1. HIGH — [probe_overfit100_sampling_steps.py:137](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:137), [probe_wan_overfit100_sampling.sh:91](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/probe_wan_overfit100_sampling.sh:91): `validation_probe_output_dir` is unrestricted, so the probe can write `probe_steps_ckpt2500.json` inside a `step_*` verdict directory; remove the override or reject any noncanonical root/path component beginning `step_`, and test hostile overrides/run names.
2. MEDIUM — [probe_overfit100_sampling_steps.py:210](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:210), [probe_overfit100_sampling_steps.py:237](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:237), [probe_wan_overfit100_sampling.sh:88](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/probe_wan_overfit100_sampling.sh:88): the approved fixed experiment is not enforced—arbitrary positive arms are accepted, the baseline silently becomes `min(arms)`, and checkpoints other than 2500 are allowed; require restored step 2500 and arms exactly `{25,50,100}`, with baseline fixed to 25.
3. MEDIUM — [validate_wan_overfit100.sh:98](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/validate_wan_overfit100.sh:98), [validate_wan_overfit100.sh:249](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/validate_wan_overfit100.sh:249): revert `3c85608`; these prominently logged controls are provably ignored by this launcher’s entry point, and tests preserving dead knobs do not prevent operational misuse.
4. LOW — [probe_overfit100_sampling_steps.py:1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/probe_overfit100_sampling_steps.py:1), [probe_wan_overfit100_sampling.sh:63](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/bash_scripts/probe_wan_overfit100_sampling.sh:63): “H1 diagnostic” reverses the analysis terminology; rename it “H1/H2 discriminator” or “H2 sampling-side diagnostic.”

Verified claims:

- P1: The payload cannot currently be admitted by the verdict CLI: inputs are explicit, and this schema lacks the required 25-step aggregation provenance/role fields. The unrestricted output override nevertheless breaks write-location isolation.
- P1: No JSON is written during rollouts. Preemption before final publication leaves no probe JSON; identical completed retries are accepted, changed bytes are refused. The writer is not concurrency-safe, so confirm the prior attempt is dead before retrying.
- P2: `_ArmConfig` is sound under tracing. Each arm creates a separate jitted function with a distinct sigma shape and loop bound; the driver test catches shared/pinned config wiring and behaviorally identical arms.
- P3: Restore, batch shape, latent dtype conversion, correct context, folded RNG, rollout, VAE decode, and SSIM match the eval path. The relevant rollout/reader/context/RNG/metric functions are unchanged from the landed segment-final code.
- P3: The selected 30-window digest matches the golden value. The landed seed-0/correct reference for those windows is mean SSIM `0.8100125855`, median `0.8059329625`; accepting the probe scientifically must require row-level 25-step agreement.
- P5: Omitting ffmpeg is justified: neither auxiliary RGB, MP4 fetch/decode, nor video export is reachable. Full pinned-snapshot prefetch, local-only resolution, manifest/dataset binding, exact checkpoint restore, venv handling, and COMMIT export mirror the established launcher.
- P6: Neither commit changes the settled rollout, restore implementation, verdict statistic, role validation, staging, or publication machinery.
- Validation: source compilation, both `bash -n` checks, and `git diff --check` passed. I could not independently rerun pytest because this review sandbox provides no writable temporary directory; `1222 passed / 2 skipped` remains commit-record evidence.

**Status / launch fitness**

- Probe job: **NOT FIT TO LAUNCH** until findings 1–3 are fixed, focused tests/mutations rerun, the resulting SHA is pushed (the current two commits are ahead of `origin`), and the launch command/acceptance criteria are recorded.
- After re-review: 1 × v6e-8, approximately 55 minutes, no resume is reasonable. Under poor spot availability, `PROBE_NUM_WINDOWS=15` is an acceptable predeclared fallback, provided the prior attempt is confirmed terminated and no completed JSON occupies the immutable path.
- Job 23: the worklog records the 10k extension as launched; live status was not queried or changed.
- Review subprocess: complete; Yixun can steer immediately.
```

## Strengthening record — coder response (round 14, marker: sampling-probe)

**Scope:** all four findings, closed. No verdict-machinery file was touched (`overfit100_success_statistic.py`
and the staging/publication code in `generate_wan_side_adapter.py` are byte-identical to `66720dc`).

### 1. HIGH — unrestricted output override → **removed entirely** (the reviewer's stated preference)

`validation_probe_output_dir` is gone from the module, the launcher (`PROBE_OUTPUT_DIR`) and the YAML.
`probe_output_path()` now builds the canonical path only, and still *checks the result*, because
`output_dir`/`run_name` remain operator-controlled: any path component beginning `step_` is refused with
an error that names the offender and says why (a diagnostic must never land in the verdict's evidence
tree), and the file must sit directly under `validation_probe_sampling/`. Tests cover hostile `run_name`s
(`step_002500_s3_segment_final`, `x/step_…_s3_full_set`, `../step_…`, `ok/../step_…_s3_intermediate`), a
hostile `output_dir`, source-level absence of the override in the module, and its absence from the
launcher *and* the config — an env knob or a live YAML key would be a way back in.

### 2. MEDIUM — the approved experiment is now pinned, not parameterised

New module constants `APPROVED_SAMPLING_ARMS = (25, 50, 100)`, `APPROVED_CHECKPOINT_STEP = 2500`,
`BASELINE_SAMPLING_STEPS = 25`, enforced by `assert_approved_design(steps=…, checkpoint_step=…)`:

* arms compared as an order-canonicalized **set** — subsets, supersets and other values all refused;
* **checkpoint checked twice** — once against the requested `config.checkpoint_step` *before* the ~5B
  load (cheap refusal), once against the step Orbax actually **restored** (the request is not proof of
  what came back);
* `probe_artifact` now passes `baseline=BASELINE_SAMPLING_STEPS`; `baseline=min(...)` is gone and a test
  pins its absence, so a future arm-set edit cannot silently move the baseline.

`probe_num_windows` stays free: 15 is the pre-declared bad-weather fallback.

### 3. MEDIUM — `3c85608` reverted

`bash_scripts/validate_wan_overfit100.sh` restored to its `fc9ac52` bytes
(`git diff fc9ac52..HEAD -- bash_scripts/validate_wan_overfit100.sh` is empty). **Declared deviation:**
the three *launcher* tests from `3c85608` were deleted, but two **module-fact** tests were kept, reworded
—`test_the_overfit100_rollout_ignores_the_pipeline_sampling_knobs` and
`test_the_signature_binds_the_knobs_that_actually_drive_the_rollout`. They assert facts about production
code, not about the reverted knobs: they are the executable record of *why* those knobs were dead
(the rollout reads `side_adapter_sampling_steps`, has no CFG branch, and the run signature binds the
parameters that do drive it). A new byte test, `test_the_eval_launcher_carries_no_sampling_passthroughs`,
pins the revert itself. If the reviewer wants the module-fact pair gone too, say so and it is a one-line
deletion.

### 4. LOW — terminology

"H1 diagnostic" → **"H1/H2 discriminator"** in the module docstring and the launcher header, with the
discrimination stated explicitly: H1 (sampler-limited) predicts a material SSIM lift from the longer
arms; H2 (weight-limited) predicts none; the paired per-window deltas are the read-out.

### Acceptance criteria recorded in the module docstring

The reviewer's anchor is now in the source: the 30 probe windows' landed 25-step reference is
**mean SSIM 0.8100125855, median 0.8059329625**. The docstring states that the row-level 25-arm
agreement check is performed **offline** against the published aggregation — deliberately, since reading
a verdict artifact here would destroy the isolation the reviewer verified — and that a 25-step arm which
does not reproduce those numbers invalidates the probe, not the checkpoint. **Confirmed present in the
JSON** for that offline join: every row carries `name`, `episode_id`, `episode_index`, `window_start`,
`sampling_steps`, `ssim`, `latent_mse`, `pixel_mse` (plus the cohort list and the 30-window digest), so
the join key and the per-arm metric are both there. No new fields were needed.

### Verification

* Full worklogs suite: **1236 passed, 2 skipped** (was 1222 + 2; +14 net after the launcher-test swap).
* Mutation spot-checks, all killed (probe + eval-selection files, 96 tests):
  1. `step_`-component rejection dropped → **5 failed**;
  2. arm-set pin dropped → **3 failed**;
  3. checkpoint pin dropped → **2 failed**;
  4. `baseline=BASELINE_SAMPLING_STEPS` → `min(arms)` → **1 failed**;
  5. `3c85608` partially resurrected (one `NUM_INFERENCE_STEPS` default re-added to the eval launcher) →
     **1 failed** (the byte test).
* `black --line-length 119`, `ruff check` clean; `bash -n` on both launchers; staged-tree `git diff --check`
  clean.

No push, no launch.

## Re-review (pass 2) — APPROVE

- **Date:** 2026-08-01 | **Commit:** `a921917` | Codex `gpt-5.6-sol` xhigh
- All four findings VERIFIED fixed; declared deviation (two module-fact tests kept) ACCEPTED; reviewer independently reproduced the 30-window anchor (mean 0.8100125855 / median 0.8059329625) and digest. FIT TO LAUNCH (1 × v6e-8, ~55 min, no resume; PROBE_NUM_WINDOWS=15 only as predeclared fallback; single writer; 25-arm mismatch invalidates the probe, not the checkpoint).

```
APPROVE

1. VERIFIED — Output override is removed from module, launcher, and YAML; canonical path enforcement rejects all five hostile `step_` cases and requires the file directly under `validation_probe_sampling/`.
2. VERIFIED — Arms are set-canonicalized to `{25,50,100}`; checkpoint 2500 is checked before model load and against Orbax’s restored step; baseline is fixed at 25; 15 windows remains valid.
3. VERIFIED — `validate_wan_overfit100.sh` is byte-identical to `fc9ac52`. **Declared deviation ACCEPTED:** the two retained tests assert relevant production invariants and do not preserve the reverted knobs. Minor record correction: the new regression test checks token absence, not whole-file byte identity.
4. VERIFIED — H1/H2 terminology and acceptance criteria are correct. Independent offline recomputation reproduced mean `0.8100125855`, median `0.8059329625`, and the exact 30-window digest; JSON contains every required join field.
5. VERIFIED — Mutation failure counts correspond to the relevant regression tests. Focused re-run: **21 passed**; syntax, `bash -n`, and `git diff --check` passed. The reported full **1236 passed / 2 skipped** remains Coder-run evidence because this read-only sandbox cannot provide pytest temporary storage.

Launch fitness — **FIT TO LAUNCH** as 1 × v6e-8 at `a921917` once pushed, approximately 55 minutes, with no resume. `PROBE_NUM_WINDOWS=15` is acceptable only as the predeclared fallback. Obtain explicit launch approval, record the exact command and acceptance criteria at launch, confirm any prior attempt is terminated and the immutable output path is unoccupied, and allow only one writer. A mismatching 25-step control invalidates the probe—not the checkpoint.

Status

- Review: complete.
- Job 23: last recorded as launched; live state was not queried or changed.
- Earliest steer: now.
tokens used
179,728
APPROVE

1. VERIFIED — Output override is removed from module, launcher, and YAML; canonical path enforcement rejects all five hostile `step_` cases and requires the file directly under `validation_probe_sampling/`.
2. VERIFIED — Arms are set-canonicalized to `{25,50,100}`; checkpoint 2500 is checked before model load and against Orbax’s restored step; baseline is fixed at 25; 15 windows remains valid.
3. VERIFIED — `validate_wan_overfit100.sh` is byte-identical to `fc9ac52`. **Declared deviation ACCEPTED:** the two retained tests assert relevant production invariants and do not preserve the reverted knobs. Minor record correction: the new regression test checks token absence, not whole-file byte identity.
4. VERIFIED — H1/H2 terminology and acceptance criteria are correct. Independent offline recomputation reproduced mean `0.8100125855`, median `0.8059329625`, and the exact 30-window digest; JSON contains every required join field.
5. VERIFIED — Mutation failure counts correspond to the relevant regression tests. Focused re-run: **21 passed**; syntax, `bash -n`, and `git diff --check` passed. The reported full **1236 passed / 2 skipped** remains Coder-run evidence because this read-only sandbox cannot provide pytest temporary storage.

Launch fitness — **FIT TO LAUNCH** as 1 × v6e-8 at `a921917` once pushed, approximately 55 minutes, with no resume. `PROBE_NUM_WINDOWS=15` is acceptable only as the predeclared fallback. Obtain explicit launch approval, record the exact command and acceptance criteria at launch, confirm any prior attempt is terminated and the immutable output path is unoccupied, and allow only one writer. A mismatching 25-step control invalidates the probe—not the checkpoint.

Status

- Review: complete.
- Job 23: last recorded as launched; live state was not queried or changed.
- Earliest steer: now.
```
