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
