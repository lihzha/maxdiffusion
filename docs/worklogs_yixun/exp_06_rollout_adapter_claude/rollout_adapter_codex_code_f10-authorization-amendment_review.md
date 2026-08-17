# Codex code review — F10 authorization amendment (6aa021c..4e264dc)

Reviewer: codex gpt-5.6-sol, model_reasoning_effort=xhigh, 2026-08-16T~23:40Z. Raw session log retained in the Planner session scratchpad (f10_codex_review_raw.log). Sandbox note: the reviewer's sandbox rejects /dev/fd process substitution, so its full-suite run showed 157 launcher-test failures + 37 errors that are sandbox artifacts (the Coder's run on the real venv was 2313 passed / 0 failed); the four touched non-launcher test files passed 299/299 in-sandbox, and all F10-1..4 battery probes + 13 controls passed there.

## Findings

1. **MAJOR — A missing runtime watermark authorizes the cell.**  
   `cell_verdict` checks the watermark only when it is non-`None`, so an analysis-only record can authorize without satisfying `watermark <= analysis` ([pos_rollout_fit_probe.py:1321](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1321), [pos_rollout_fit_probe.py:1330](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1330)). This is reachable when telemetry returns capacity but no watermark ([pos_rollout_fit_probe.py:2643](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2643)). Worse, the test explicitly requires this authorization, so it pins the implementation rather than the stated contract ([test_pos_rollout_runtime_peak.py:139](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_runtime_peak.py:139), [test_pos_rollout_runtime_peak.py:153](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_runtime_peak.py:153)).  
   **Fix:** refuse `watermark_bytes is None` with a distinct reason such as `watermark_missing`, or fail the measurement before publication. Add an end-to-end F10 battery attack for an analysis-bounded record with no watermark.

2. **MAJOR — Independent trial maxima erase trial-local inconsistencies and can gate a launch.**  
   Aggregation independently takes `max(peak)`, `max(analysis)`, and `max(watermark)` ([pos_rollout_fit_probe.py:1407](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1407), [pos_rollout_fit_probe.py:1420](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1420), [pos_rollout_fit_probe.py:1421](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1421)). I executed these through publication, reload, and `assert_cell_authorized`:

   - Trial 1: `analysis=10, watermark=11`; trial 2: `analysis=20, watermark=5` → aggregate `analysis=20, watermark=11`, authorized.
   - Trial 1: `analysis=8, peak=31, watermark=1`; trial 2: `analysis=32, peak=32` → `peak_exceeds_analysis` is masked and the aggregate authorizes.
   - One trial with no analysis plus one with `analysis=20` also authorizes.

   Thus `peak_exceeds_analysis` closes the single-record space but not the repeated-trial space. The test’s claim that no authorizing aggregate can contain a contradicted trial is false ([test_pos_rollout_runtime_peak.py:1196](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_runtime_peak.py:1196)). This is not acceptable merely as a docstring residual: disagreement about the same compiled program is itself fail-closed evidence.  
   **Fix:** evaluate and preserve refusal conditions per raw trial before collapsing them. Any missing analysis, `watermark > analysis`, or `peak > analysis` in any trial must survive aggregation as a cell refusal; alternatively reject inconsistent analysis values outright. Add a two-trial end-to-end battery attack.

3. **MODERATE — Malformed byte counts are silently truncated, and the headroom comparison is not mathematically exact.**  
   `_optional_bytes` accepts floats by applying `int()` without checking whole-number equality ([pos_rollout_fit_probe.py:1209](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1209)); peak and capacity do likewise ([pos_rollout_fit_probe.py:1266](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1266)). A hand-built record with `analysis=watermark=peak=9.9`, `capacity=10` becomes `9/10` and authorizes. Even integral values can cross the boundary through floating division at sufficiently large magnitudes: an exact over-90% pair can round to `0.9` at [pos_rollout_fit_probe.py:1328](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1328).  
   **Fix:** reject booleans and non-integral byte/count values, and compare the pinned boundary exactly, e.g. `10 * analysis > 9 * capacity`. Add tests for exactly 90%, one byte above it, and malformed fractional counts.

## Other judgments

- The standing-lifetime watermark behavior is acceptable. It can falsely refuse a later smaller cell, but only in the fail-closed direction; the literal contract and unchanged F9 discipline do not require a cell-attributed alternative.
- Runtime sources remaining eligible is correct for the real tie case: the measurement path can select a runtime source when `watermark == analysis`, and equality must authorize.
- `authorized_bytes`, `authorized_fraction`, and `watermark_bytes` are appropriate derived v7 fields. Keeping `fit_cell.v2` is also correct because raw trial fields did not change.
- The twice-run subprocess test is sound: it isolates directories and derives the second run’s capacity from the first run’s actual analysis.
- v6/v7 handling is otherwise correct: loader, direct protocol check, and republication refuse v6; production always loads before gating ([wan_pos_rollout_trainer.py:441](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:441)).
- The F10 fixture repair is sound: `_fit` now starts from an authorizing analysis/watermark pair while preserving explicit `None` and explicit attack overrides ([reviewer_attacks.py:683](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:683)).
- The nine-file diff matches the declared surface. `BINDING_FIELDS`, `binding_digest`, `_adoption_refusal`, `deployed_manifest_digest`, exclusions, adoption, and phase bracketing were not weakened.

Verification: the four touched non-launcher test files passed, **299 passed**. The full 2,313-test command collected all tests but this sandbox rejects Bash `/dev/fd` process substitution, producing **2,119 passed / 157 failed / 37 errors**, all in launcher execution. The same restriction prevented four legacy battery launcher probes from running; all F10-1..4 probes and all 13 controls passed. The battery still needs the missing-watermark, cross-trial-masking, and exact-analysis-headroom cases above.

**REWORK**

---

# Focused re-review of F10b (4e264dc..1e5dda9) — same reviewer, 2026-08-16T~23:50Z

(First invocation of this re-review was killed by the OpenAI-side cyber-risk filter on the prompt's adversarial phrasing — false positive on our own module's defensive review; re-issued with neutral wording, identical obligations.)


1. **MODERATE — malformed count values can still be truncated and authorize a banked cell.**  
   [pos_rollout_fit_probe.py:1186](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1186), [pos_rollout_fit_probe.py:1238](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1238)

   Two executed residuals remain:

   - `_exact_count` tests integrality after `float(value)`. A non-integral `Fraction` or `Decimal` of `81064793292668928.5` rounds to an integral float, is truncated to `81064793292668928`, and authorizes exactly at 90%, even though the original value satisfies `10a > 9c`.
   - `CellMeasurement.from_payload` applies bare `int()` to `peak_bytes`, `capacity_bytes`, and `reservation_failures` before `_checked` sees them. Digest-valid bank artifacts containing `peak_bytes=9.9`, `peak_bytes=True`, or `reservation_failures=0.9` were loaded, adopted, republished, and accepted by `assert_cell_authorized`.
   - Digit strings are also accepted and authorize. Ordinary `int` subclasses are correctly accepted; every tested negative count was refused.

   **Concrete fix:** compare against the original numeric value without passing through `float`, explicitly reject strings/bytes and booleans, and use `_exact_count` directly in `from_payload` and serialization. For example, obtain `count = int(value)` and require `value == count`; this preserves legitimate integral subclasses while rejecting large fractional `Decimal`/`Fraction` values. Add an adopted-cell round-trip test, not only direct-dataclass tests.

2. **MODERATE — a banked analysis disagreement permanently poisons retries.**  
   [pos_rollout_fit_probe.py:1482](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1482), [pos_rollout_fit_probe.py:2255](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2255), [pos_rollout_fit_probe.py:2501](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2501)

   I published a cell artifact whose individually valid trials reported analyses 10 and 20. Two consecutive retries both adopted it successfully and then stopped at publication with `analysis_disagreement`. Thus the retry-robustness concern is concrete.

   **Concrete fix:** retain the table-wide raise, but validate `aggregate_trials(artifact.trials)` before banking and while loading/adopting a banked cell. An inconsistent cached artifact should be rejected and remeasured rather than adopted into another guaranteed publication failure.

## Original finding status

- **Finding 1: CLOSED.** `watermark_missing` is unconditional. A cross-product of valid analysis, peak, capacity, source, and failure shapes produced zero authorizations without a watermark. F10b-1 also refused as expected.
- **Finding 2: CLOSED for authorization correctness.** The first two original counterexamples raise `analysis_disagreement`; the missing-analysis pair aggregates to `analysis=None` and refuses. All-trial absence and a peak-only trial refuse with both missing-evidence reasons. Changing `watermark_before_bytes` never changed a verdict.
- **Finding 3: NOT-CLOSED.** Exact integer headroom arithmetic and ordinary `9.9`/boolean direct records are fixed, but Finding 1 above leaves both a large-number direct truncation and the banked-record deserialization path.

## Design ruling

I accept **raising on analysis disagreement** rather than converting it to `analysis_missing`: disagreement means the claimed single executable did not produce stable evidence, and silently replacing two recorded values with “missing” loses that distinction. However, the bank must quarantine that artifact before adoption as described in Finding 2; otherwise the fail-closed choice becomes a permanent retry outage.

## Protocol ruling

A v8 bump is **not required** under the stated facts. Refusal reasons are derived, not trusted enum inputs, and loading requires byte-identical reconstruction. An old/new reader encountering a verdict affected by `watermark_missing` or exact headroom will reject the table rather than mis-authorize it. With no published v7 table requiring compatibility, retaining v7 is sufficient.

## Verification

- Focused pytest suite: **280 passed**.
- Harness: all F10b attacks refused and **15/15 controls passed**. The four `SUCCEEDED` entries were exactly the known sandbox-blocked launcher probes; effective result is the expected 100 refused / 1 declared / 0 succeeded outside this sandbox.
- `watermark == analysis`, standing-lifetime behavior, and v7 handling remain covered and passing.
- Diff scope is seven declared files: production module, two tests, harness/README/log, and the prior review document. Binding, manifest, adoption-policy, trainer, and phase-bracketing code had no diff hunks.

**Final verdict: REWORK**

---

# Verification pass of F10c (1e5dda9..39f164e) — same reviewer, 2026-08-17T~00:05Z

codex
1. **MODERATE — exact-count handling remains asymmetric and still truncates identity/evidence fields.**  
   [pos_rollout_fit_probe.py:467](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:467), [pos_rollout_fit_probe.py:563](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:563), [pos_rollout_fit_probe.py:1576](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1576), [pos_rollout_fit_probe.py:2286](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2286)

   Executed counterexamples:

   - `FitCell("rollout", 8.5, 2.5)` published successfully and loaded as cell `8/2`; `__post_init__` and `as_payload` still use bare `int()`.
   - A digest-valid bank artifact with authoritative `context.device_count=8.5` loaded as `8` and was adopted by an eight-device context. `ProbeContext.from_payload` and `as_payload` still truncate a binding field.
   - `_positive_int(Fraction(162129586585337857, 2), ...)` and the equivalent `Decimal` returned `81064793292668928`.
   - `CellArtifact.from_payload` accepted `trial_count=1.9`, `"1"`, and `True` as one trial.

   **Concrete fix:** use `_exact_count` plus positivity checks in `FitCell.__post_init__/as_payload`, `ProbeContext.from_payload/as_payload`, `_positive_int`, and `CellArtifact.from_payload`’s `trial_count`. Validate raw ladder/trial/projection inputs before any `int()` normalization.

Residual rulings:

- **Malformed-count residual (a)+(b): NOT-CLOSED.** The specifically requested boundary `Fraction`/`Decimal`, digit-string, bytes, bytearray, bool, and overflow cases now fail correctly. The digest-valid `peak_bytes=9.9`, `peak_bytes=True`, and `reservation_failures=0.9` artifacts also fail parsing and are remeasured. However, the remaining coercion paths above still alter cell identity, topology binding, or evidence.
- **Retry-poisoning residual: CLOSED.** Attempt 2 quarantined the disagreement, remeasured, and published normally; attempt 3 adopted attempt 2’s clean bank without measurement. `_adoption_refusal` returned a refusal string for the poisoned artifact, and malformed cached inputs were caught without killing the ladder.

No regressions found in the previously closed behavior: unconditional `watermark_missing`, unanimous aggregation, disagreement raises, exact headroom, watermark equality, standing watermark lifetime, v7 enforcement, binding/manifest policy, trainer gating, and bracketing all remain intact. The production diff has no unexpected trainer or policy hunks.

4. **Publication ruling:** accept `publish_cell` raising. Analysis disagreement is an evidence invariant violation, not a storage condition. Early failure also avoids banking poison and potentially wasting the remainder of the ladder; skip-write-and-log is not required.

5. **Test-surface ruling:** accepted. Adoption round-trip tests belong naturally in `test_pos_rollout_cell_publication.py`, where the fixtures and retry machinery already live, and it is within the declared fit-probe test surface.

Verification:

- Focused pytest: **290 passed**.
- Battery: printed **98 REFUSED / 1 DECLARED / 4 SUCCEEDED**, with the four successes exactly the known launcher sandbox failures; effective result **102 REFUSED / 1 DECLARED / 0 SUCCEEDED**. All **16 controls passed**.

**REWORK**
