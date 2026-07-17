# Code review: exp_01 full_ft_overfit — round trainer-wiring
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-17

## Context loaded

- Read `experiment_SOP.md`, the driving query, approved plan §§2.1/2.4/3/4/6, and worklog through round 3.
- Read both prior code reviews and their strengthening records; the binding round-3 notes were honored.
- Inspected `git status --short`, the complete diff, full trainer file, full new test file, and `train_wan.py`.
- Mechanically diffed both `start_training` methods; the 121/146 byte-identical-line claim is exact, with nine full-FT-required semantic deviations.
- Traced transformer loading, logical FSDP placement, actual-TPU sharding retention, restore targets, JIT shardings, donation, and inherited checkpoint/eval methods.
- Confirmed rounds 4–5 generation/config/launcher changes are correctly absent.
- Attempted the exact pytest command; read-only temporary-file restrictions prevented startup, then a non-writing capture/cache workaround completed all 38 tests successfully.
- Ran `git diff --check` and Ruff successfully; Black’s only source reformat is the inherited 120-character split line.

## Adjudications

- (a) **ACCEPTED** — the limit is normally valuable, but §4 deliberately bundles the 146-line parent mirror with the independently testable audit/logging machinery; record that the audit/introspection portion is about 135 lines, not the claimed ~55.
- (b) **CHANGE-ORDERED** — the all-leaf raise preserves OOM safety, but magnitude-sorted optimizer leaves are not the specified path-matched mu/nu twins and do not satisfy §6’s audit record.
- (c) **ACCEPTED** — the explicit identity replacement is a clearly documented parity seam and retains the computed objects exactly; the unused `state` argument is harmless.
- (d) **ACCEPTED** — preserving this single parent-identical line is a reasonable parity exception, and Ruff remains clean.
- (e) **CHANGE-ORDERED** — the source checks catch simple deletion but do not establish reachability, ordering, final sharding use, audit propagation, or actual dispatch behavior.

## Verdict

REQUEST-REVISION. The core training route is correct: transformer-derived `FullFTTrainState`, direct module-level steps, activation-dtype null context, parent-identical JIT shardings/donation, and final computed/actual FSDP specs are retained through placement; a fully replicated real WAN state would trip the >100 MiB audit. However, the dtype and per-host-memory logs cannot reliably serve §§2.4/6, and the critical integration wiring needs behavioral coverage before round 4.

## Findings

1. **F1 — MAJOR:** Startup dtype logging reports only the first leaf (`_first_leaf_dtype` and `_adam_moment_dtypes`, trainer lines 319–331 and 445–452). The production loader intentionally keeps norms, `condition_embedder`, and scale/shift tables in float32 while casting other weights to `weights_dtype` (`wan_pipeline.py:57–71`), so the primary tree is mixed. A real Optax mixed-tree probe reported `float32/float32/float32` while both bfloat16 and float32 existed in params and mu, making the primary run potentially indistinguishable from the fp32 control. Replace first-leaf sampling with deterministic per-dtype leaf/count/byte summaries for params, mu, and nu; add a mixed-dtype real-Optax test proving primary `{bfloat16,float32}` versus fp32-control `{float32}` and per-path moment agreement.

2. **F2 — MAJOR:** `_leaf_bytes` deduplicates `addressable_shards` by index (lines 247–254), so replicated per-device copies are excluded from the value labeled “per-host addressable” and used for HBM assessment. A two-replica 40-byte fake leaf reports 40 addressable bytes instead of 80 physical local bytes. Sum every addressable shard for physical per-host HBM; if unique logical bytes are also useful, log them under a separate label and test repeated shard indices.

3. **F3 — MINOR:** `audit_large_leaves` returns the globally largest eight optimizer leaves (lines 313–316), not the mu/nu twins of the eight selected parameter paths. With Adam’s equal-sized mu and nu, this commonly covers both moments for only four parameter leaves. Locate the real `ScaleByAdamState`, path-match both `mu` and `nu` for each selected parameter, label them explicitly, and add an extra/skewed optimizer-leaf test that defeats magnitude matching.

4. **F4 — MAJOR:** The critical structure tests at lines 92–100 and 193–213 inspect source tokens. They do not prove validation occurs before loading, that the helper’s returned shardings are the objects audited/placed/returned, that an integrated audit raise propagates, or that dispatch instantiates the full-FT trainer. Add behavioral tests with a failing pipeline-load spy for early validation, a fake context-manager mesh plus monkeypatched sharding/placement calls, a replicated-large-leaf integration raise, and an executed `train_wan.train` dispatch using a fake trainer.

5. **F5 — MINOR:** The guide-scale guard’s `abs(guide_scale - 1.0) > 1e-6` condition lets `NaN` pass and weakens the plan’s hard `== 1.0` invariant. Reject non-finite values explicitly and either compare exactly or document/test the tolerance; add a `NaN` regression case. The fresh-noise guard and both error messages are otherwise early and actionable.

## Notes for round 4

Do not begin `ckpt-generation` until F1–F5 are strengthened; afterward, round 4 should behaviorally prove the generic `params/opt_state/step` restore into `FullFTTrainState`, the step-0 bypass, and the plain-transformer rollout without introducing round-5 config or launcher work.

---

# Strengthening record (Coder, same round — 2026-07-17)

- **F1 (MAJOR) — FIXED.** First-leaf dtype sampling replaced by per-dtype summaries `{dtype: (leaf_count, total_bytes)}` for params/mu/nu (3 log lines + activations). Mixed real-optax test (f32-norm-first tree, the reviewer's probe shape) proves primary `{bfloat16,float32}` vs control `{float32}` logs differ; per-path moment-dtype agreement test added. First-leaf revert-mutant fails the mixed test. Before/after log evidence in the worklog/report.
- **F2 (MAJOR) — FIXED.** `_leaf_bytes` sums ALL addressable shards (no index dedupe) → true per-host physical HBM; label renamed "per-host physical". Repeated-index test: (40, **80**) not (40, 40); dedupe revert-mutant fails. Sub-item rejected with reason: no third "unique logical addressable" label — the entry's `global` column already carries unique-bytes meaning; a third number reintroduces the ambiguity the fix removes.
- **F3 (MINOR) — FIXED.** Opt twins now located via real `ScaleByAdamState` and path-matched (mu AND nu per selected param, labeled, with path-matched specs); all-leaf raise unchanged. Skewed-tree test defeats magnitude matching; magnitude revert-mutant fails.
- **F4 (MAJOR) — FIXED.** Four behavioral tests added (source-token checks retained as deletion guards): validate-fires-before-pipeline-load (raising load spy); shard-dataflow object-identity through audit→host-transfer→device_put→return; integrated audit-raise propagation out of `_shard_state` (137 MB replicated leaf, mesh 64, device_put booby-trapped); executed `train_wan.train` dispatch with recording fake. No production seams needed — module-attribute monkeypatching only; production code untouched by F4. Mutants (validate-after-load, replicate-in-helper, audit-swallowed, dispatch-dropped) all caught.
- **F5 (MINOR) — FIXED.** Guard: `not math.isfinite(gs) or gs != 1.0` (exact compare, documented); parametrized NaN/±inf/1.0000001 regressions; old tolerance revert-mutant fails on NaN and 1.0000001.
- **Beyond findings:** log-label/mu-nu-labeling cosmetics only; one test renamed for post-F3 accuracy; ruff C408 literal fix. 7/7 mutants caught; trainer sha256-verified restored after each. Orchestrator independently re-ran the suite: 50/50 green.

**Cycle 3 closed:** write → review (REQUEST-REVISION, F1–F5; adjudications: a/c/d ACCEPTED — a with corrected ~135-line audit accounting — b/e CHANGE-ORDERED→F3/F4) → strengthen (5 FIXED, 1 sub-item rejected with reason) → commit. Final: 50/50 green; `start_training` parity 121/147 byte-identical to parent.
