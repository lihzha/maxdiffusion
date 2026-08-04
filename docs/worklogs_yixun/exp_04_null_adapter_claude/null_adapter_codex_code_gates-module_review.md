# null_adapter — Codex code review: round R5 `gates-module`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `0bc19e8`. Reviewer ran the focused suite (33 passed) and empirical probes (missing-SSIM tables passing G1; k={0}-PASS/k={0,1,2}-FAIL G3 tables; duplicate-JSON-key coverage bypass; seed-swap surviving all tests). Independently confirmed: G3 baseline imputation direction, percentile edge cases, strict invalidity boundary, E7/E9/E12 closures, no nanmean leakage.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md`, both standing announcements, exp_04 query, and relevant exp_02 analysis.
- `plan_null_adapter.md` v5 in full, including §3, §4-P1/P3, fixed k-sets, and the M7/P1 plan-review resolutions.
- `null_adapter_worklog.md` through R5 and all five Planner rulings.
- R1–R4c reviews and strengthening records: `sigma-embed-noise`, `invert-trajectory`, `optimize-nulls`, `replay-operator`, `record-schema-io`, and `verify-replay`.
- Both R5 files in full: `null_adapter_gates.py` and `test_null_adapter_gates.py`.
- Repository state: HEAD `0bc19e8`; only those two R5 files are untracked.

Validation: the requested pytest command hit the known read-only temporary-directory failure before collection. With capture disabled, the focused suite passed: **33 passed in 0.30s**. Standalone module loading imported no JAX modules.

1. **MAJOR — G1/G2 do not propagate SSIM invalidity into pair invalidity.** [null_adapter_gates.py:141](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:141), [test_null_adapter_gates.py:174](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_gates.py:174)

   `pair_valid` considers only the two MSEs; control SSIM is never read. This contradicts §3 and Planner ruling 3 that an MSE-or-SSIM-invalid observation invalidates the pair. Empirically, with 7/64 method SSIMs missing, G1 reports `invalid_pairs=0`, ratio `10`, improved fraction `1.0`, and **passes**; seven missing control SSIMs also pass untouched. Both should exceed the strict 10% invalidity ceiling, with ratio `1.0` and improved=false for those pairs.

   Concrete change: compute method/control validity from both primary metrics, derive pair validity from both observations, preserve measured method SSIM only for control-only invalidity, and add method/control × MSE/SSIM cases at the 6/64 and 7/64 boundary.

2. **MAJOR — fixed noise estimands and A1-probe reduction remain caller-controlled.** [null_adapter_gates.py:205](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:205), [null_adapter_gates.py:219](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:219), [null_adapter_gates.py:288](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:288)

   G2 and G3 accept arbitrary `k_set`, while target selection accepts already-aggregated probe scalars with no seed coverage, imputation, or range enforcement. A probe produced G3 **PASS** at `k={0}` with mean delta `+0.10`, but **FAIL** under the required keyed `{0,1,2}` reduction with `−0.0333`. An incomplete A1-probe can similarly choose A1 before this module sees the table.

   Concrete change: define and enforce the fixed keyed/global k-sets in the public APIs; make G3 derive its k-set from the selected convention; and make A1-probe selection consume its metric table and manifest, returning coverage/invalid/imputation counts. The relative conjunct itself is correctly implemented, but public range enforcement is needed so the deliberately out-of-range test case cannot become a P4 “transfer-only” explanation.

3. **MAJOR — the JSON boundary cannot uphold exact coverage and can emit non-standard JSON.** [null_adapter_gates.py:75](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:75), [null_adapter_gates.py:292](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:292), [null_adapter_gates.py:314](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:314)

   `json.load` silently overwrites duplicate object keys before coverage runs, so a table containing the same manifest name twice can pass “exactly once” coverage. On output, `a1_mean_ssim=0` produces `probe_relative=Infinity`; `verdicts_to_json` serializes that token, which strict JSON rejects.

   Concrete change: parse with duplicate-key rejection, expose missing/extra/duplicate coverage tallies, represent undefined ratios as `null`, serialize with `allow_nan=False`, and test raw duplicate JSON plus strict round-tripping and deterministic key ordering.

4. **MINOR — the tests do not pin the declared bootstrap seed to the computed CI.** [null_adapter_gates.py:110](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_gates.py:110), [test_null_adapter_gates.py:274](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_gates.py:274)

   The E5 unseeded-mutant closure is real, but replacing the actual RNG seed with fixed seed `20260805` while continuing to report `20260804` leaves all **33 tests passing**. Add an independently computed exact/golden CI for a nondegenerate fixture using literal seed `20260804` and 10,000 example-level resamples.

Confirmed independently: G3 imputes a missing baseline to `−1.0`; n=64 tied and degenerate percentile CIs match example-level resampling; 6/64 versus 7/64 gives the correct strict invalidity boundary when invalidity is counted; E7/E9/E12 are genuinely closed; and no `nanmean` path silently drops ordinary NaNs.

Final verdict: **REQUEST-REVISION — G1/G2 can pass despite contract-invalid SSIM coverage, while fixed seed estimands and exact JSON coverage are not enforced at the public boundary.**

**Status**

- R5 review: complete now; no subprocesses or background jobs are running.
- Earliest steer: now, before R5 strengthening and commit.

---

## Strengthening record (Coder, same round; all four findings closed, none rejected)

1. **MAJOR (SSIM invalidity) — FIXED.** Reviewer's 7/64 red reproduced first; validity now derives from both metrics of both observations with a single `_Observation.valid` definition (the initially-equivalent mutant F1 was made non-equivalent by refusing the dead clause); ruling-3 nuance preserved with a separate `imputed_method_ssim` tally; side×metric×boundary matrix (8 cases).
2. **MAJOR (estimands) — FIXED.** `NoiseConvention` enum with derived k-sets (public), private explicit-k seam; reviewer's k={0}-PASS/keyed-FAIL scenario is now a test; `select_target` consumes the probe table+manifest+convention through the full machinery with reported counts; out-of-range SSIM invalid at the boundary; transfer-failure-implies-floor-failure reporting asserted.
3. **MAJOR (JSON) — FIXED.** Duplicate-key-rejecting parse hook before coverage; missing/extra/duplicate tallies in verdicts; None/null for undefined ratios; `allow_nan=False, sort_keys=True`; strict round-trip tests.
4. **MINOR (seed) — FIXED.** Golden CI `[0.06953125, 0.159375]` under literal seed 20260804 at abs=5e-8; seed-swap mutant killed.

Suite **246 passed in 16.94s** (+26); mutants F1–F14 zero survivors; module numpy-only; call-site churn confined to R5's own tests. LOC 689 (module 329) — overage reported, driven by the findings. Behavior beyond findings: none. Round R5 closed; committed with this record.
