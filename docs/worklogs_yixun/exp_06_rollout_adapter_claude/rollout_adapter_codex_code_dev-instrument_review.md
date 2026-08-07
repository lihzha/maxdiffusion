# rollout_adapter — Codex code review: T3b-3 `dev-instrument` + T3b-1/T3b-2 strengthens (combined)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: **ALL THREE REQUEST-REVISION — 4 BLOCKERs**, all accepted by the Planner.

- **A1 PASS** — the support-signature change preserves T2's extraction equivalence (the tests compute the old mapping and pass identical endpoints; arithmetic unchanged from start/end onward) and the kernel cannot redraw.
- **A2 FAIL (BLOCKER)** — zero dropout is *demonstrated* but not *enforced*: CLI values override YAML (`pyconfig.py:124`) and production passes `config.dropout` into `WanModel` (`wan_pipeline.py:137`), so a launch with `dropout=0.1` reaches production while the inertness test stays green. Needs a fail-closed guard at arm construction.
- **A3 PASS with caveat** — unsafe helpers are private and out of `__all__` and the public seam validates the actual batch, but Python callers can still reach `_draw_step_stream`; the T3b-4 production-callsite pin is what actually closes it.
- **A4 PARTIAL** — AST guard, seven-purpose docs, StepDraws docstring and the inertness wording fixed; the historical purpose count and the malformed arm sentence remain.
- **B1 FAIL (BLOCKER) — TEST is reachable and the reviewer EXECUTED the attack.** `DevCohort` is publicly constructible and its `__post_init__` checks only that the caller passed the string `dev64` plus unique names — the allowlist was on the LABEL, not the CONTENT. With the real TEST name `ep61399_v0_s00000` the reviewer produced both a `StepDraws` and a bare key; `dev_draw_key` independently turns an unchecked bare name into a key.
- **B2 FAIL (BLOCKER)** — `expected_sha256` optional (any schema-valid file labelled `dev64` is accepted); the cohort keeps names but not row/shard identity; `score_dev_cohort` accepts arbitrary caller tensors checked only for DEV-name presence, so TEST tensors under DEV keys would be scored and stamped with DEV provenance; and the loader validates one read while hashing a second, permitting payload/digest disagreement and breaking `gfile` paths.
- **B3 FAIL (BLOCKER) — the "fixed-draw" estimand is not fixed.** `dev_draw_key` folds `eval_index` into the key, so checkpoints evaluated at different cadence steps differ in parameters AND evaluation randomness — the round's own test confirms it. Selection would carry evaluation noise. Needs a fixed instrument seed and the same predeclared replicate IDs for every checkpoint.

**Planner rulings.** (1) On B2's choice, take the STRUCTURAL option: load batches through the validated cohort rather than accepting caller-supplied mappings and verifying them — verification of caller tensors is checkable-but-forgeable, loading through the cohort makes the wrong thing unconstructible. (2) On A2, the earlier "measure then prove" ruling is superseded: proof-of-inertness-at-zero is not enforcement of zero, and the reviewer's standard is adopted. (3) **A pattern named after its third occurrence** — T2's rebinding lock, T3b-1's still-public unsafe path, and T3b-3's forgeable cohort are the same error: *a guard that checks a claim instead of making the wrong thing unconstructible.* Structural impossibility is the default; where it is unavailable, say so explicitly.

**Sandbox caveat:** 113 passed / 27 environment failures on the literal command (unwritable-temp probe during TensorFlow/Chex import); **140 passed** after preinitializing `tempfile.tempdir`. Concurrent T3b-4 files appeared mid-review and were excluded from the verdicts.

## Reviewer output (verbatim)

185,793
# Review verdict

The focused tests are green after the known temp-directory workaround, but the review is still **REQUEST-REVISION**. Part A retains the dropout blocker and two documentation minors. T3b-3 has three commit-blocking contract violations.

## Findings

### Part A

1. **BLOCKER — zero dropout is demonstrated, but not structurally enforced in production.**  
   [test_pos_rollout_arms.py:294](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_arms.py:294) pins text in the base YAML and proves inertness on a default-zero tiny model. However, command-line values override YAML at [pyconfig.py:124](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pyconfig.py:124), and production construction passes `config.dropout` directly into `WanModel` at [wan_pipeline.py:137](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pipelines/wan/wan_pipeline.py:137). Neither `PRODUCTION_DROPOUT_RATE` nor [build_arm](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_arms.py:255) validates the supplied transformer/config. A launch using `dropout=0.1` can therefore reach production while this test remains green.

   Concrete change: add a fail-closed production config/model guard requiring `dropout == PRODUCTION_DROPOUT_RATE == 0.0`, and test that a runtime/config override to nonzero is refused before model construction or training.

2. **MINOR — the purpose-count correction is incomplete.**  
   [rollout_adapter_worklog.md:107](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_worklog.md:107) still says five purposes. At that historical point there were six; there are seven now.

   Concrete change: correct the historical count to six while retaining the current seven-purpose statement.

3. **MINOR — the malformed arm docstring remains.**  
   [pos_rollout_arms.py:45](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_arms.py:45) still contains “`accumulation behaviour. is an ARGUMENT ...`”.

   Concrete change: replace the dangling residue with one coherent sentence.

### Part B

1. **BLOCKER — `DevCohort` is forgeable, and a bare-name draw API exists.**  
   [DevCohort](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:87) is publicly constructible; `__post_init__` only checks that the caller supplied the string `"dev64"` and unique names. A caller can wrap real TEST names in a forged `"dev64"` object. Independently, [dev_draw_key](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:76) directly turns an unchecked bare name into a key.

   I executed that attack with real TEST name `ep61399_v0_s00000`; it successfully produced both a `StepDraws` and a bare key.

   Concrete change: make the cohort an opaque loader-issued capability, remove the bare-name key entry point, and derive keys only after validated cohort membership. Add the exact adversarial test: construct a DEV-labelled cohort around an actual TEST name and require refusal.

2. **BLOCKER — scores are not actually bound to the approved manifest or its rows.**  
   [load_dev_cohort](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:113) makes `expected_sha256` optional, so any schema-valid file labelled `dev64` is accepted. The resulting object retains only names, not row/shard identity. [score_dev_cohort](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:212) accepts arbitrary caller-provided tensors and checks only that every DEV name is a mapping key. TEST tensors placed under DEV keys would be scored and stamped with DEV provenance.

   The loader also validates and hashes two separate reads at [lines 122 and 132](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:122), allowing payload/digest disagreement and breaking `gfile` paths.

   Concrete change: require the approved digest, validate and hash the same bytes once, retain manifest row bindings, and verify each scored batch’s identity against its row—or load batches through the validated cohort rather than accepting arbitrary mappings.

3. **BLOCKER — the “fixed” estimand changes between checkpoint evaluations.**  
   [dev_draw_key](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:83) folds `eval_index` into the key. The test explicitly confirms that a later evaluation gets different epsilon at [test_pos_rollout_dev_instrument.py:164](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_dev_instrument.py:164). Thus checkpoints evaluated at different cadence steps differ in parameters **and evaluation randomness**, contradicting the module’s stated selection property.

   The dedicated purpose does decouple draws from training-stream consumption and call order, but not from training seed/evaluation cadence.

   Concrete change: use a fixed instrument seed and the same predeclared replicate IDs for every checkpoint. Do not fold optimizer/global evaluation step into the selection key. Test identical checkpoint scores across different training/evaluation steps.

## Requested verdicts

- **A1 — PASS.** The signature change preserves extraction equivalence: the tests compute the old `rollout_support` mapping and pass the identical endpoints; arithmetic from `start/end` onward is unchanged. The kernel cannot redraw.
- **A2 — FAIL.** Value/gradient inertness at zero is proven, but nonzero dropout can still enter through runtime config override.
- **A3 — PASS for the prior public-API finding, with caveat.** The unsafe helpers are private and absent from `__all__`; the public seam validates the actual batch. They are merely inconvenient—not literally unreachable—because Python callers can still invoke `_draw_step_stream`. The T3b-4 production-callsite pin remains important.
- **A4 — PARTIAL.** AST guard, current seven-purpose documentation, StepDraws docstring, and inertness wording are fixed. The historical count and malformed arm sentence remain.

- **B1 — FAIL.** TEST is reachable through a forged DEV-labelled `DevCohort`, a bare-name key API, and unbound batch contents.
- **B2 — FAIL.** Provenance fields are emitted, but the approved digest is optional and the actual tensors are not bound to manifest rows.
- **B3 — FAIL.** Purpose separation is real, but checkpoint evaluations receive different draws through `eval_index`.
- **B4 — FAIL overall** because the preceding issues are commit-blocking. The `sys.modules` registration fix in both restart helpers is correct and should be kept.

## Deliverables

- **T3b-1 `step-stream`: REQUEST-REVISION**
- **T3b-2 `arm-losses`: REQUEST-REVISION**
- **T3b-3 `dev-instrument`: REQUEST-REVISION**

## Validation

The literal requested command produced **113 passed / 27 failed**; all 27 failures were the managed sandbox’s unwritable-temp probe during TensorFlow/Chex import. With `tempfile.tempdir` preinitialized, the identical selection completed:

```text
140 passed in 36.41s
```

`git diff --check` passed. Concurrent T3b-4 files appeared during the review and were excluded from these verdicts.
