# Code review: exp_02 overfit100 — cycle D (eval-tooling)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-30

## Context loaded

- Experiment SOP — review, evaluation-integrity, provenance, and fail-closed requirements.
- Approved plan v4 — fixed denominator, D11 coverage matrix, exact statistic, tie-break, controls, and two-tier claim.
- Cycle-C review record — authenticated dataset/model/context preflights and ratified sparse-eval semantics.
- Worklog and committed manifest — 100 episodes, 1,629 windows, duplicate-text cohort, and current build state.
- Cycle-D implementation and 131 tests — rollout, context controls, aggregation, success statistic, loss evaluator, configs, and launchers.
- Validation evidence — reported 943 passed + 2 skipped and 6/6 mutations; independent `git diff --check` and `bash -n` passed, while pytest was unavailable in this reviewer environment.

## Verdict

REQUEST-REVISION. The median-then-threshold statistic, correct-mode isolation, tie-break, null control, per-window RNG, and value-derangement proof are sound. The machine verdict can nevertheless shrink the canonical denominator, admit non-`C₃¹⁰⁰` checkpoints or incomplete controls, and cannot operationally establish the full-set tier; those paths can silently change the experiment’s conclusion.

## Findings

1. **D1 — BLOCKER — The canonical denominator is pass-derived rather than fixed at the build’s 100 windows.** The evaluator defines `canonical_windows` as canonical windows selected in this pass, and the CLI unions only the keys found in supplied artifacts; a sparse ten-window artifact can therefore produce denominator 10 and an “established” verdict (`src/maxdiffusion/generate_wan_side_adapter.py:1381`, `src/maxdiffusion/generate_wan_side_adapter.py:1420`, `src/maxdiffusion/overfit100_success_statistic.py:421`, `src/maxdiffusion/overfit100_success_statistic.py:428`). **Concrete change:** derive the S3 canonical cohort from the authenticated manifest independently of selection, store all 100 keys plus separate covered/missing keys in every artifact, and make the verdict CLI reject any S3 denominator other than that exact cohort. Validate flags against the fixed cohort, not merely the current pass.

2. **D2 — BLOCKER — `C₃¹⁰⁰`, coverage role, and control validity remain operator-trusted.** With no explicit override, every artifact’s `checkpoint_step` becomes a segment-final candidate, so S2 or intermediate artifacts can enter `c*`; artifacts have no `s2_gate`/`s3_intermediate`/`s3_segment_final`/`s3_full_set` role, and the default rollout is correct-mode only despite segment finals requiring three modes (`src/maxdiffusion/overfit100_success_statistic.py:429`, `src/maxdiffusion/overfit100_success_statistic.py:431`, `src/maxdiffusion/generate_wan_side_adapter.py:1401`, `src/maxdiffusion/configs/base_wan_5b_overfit100.yml:345`). The aggregator also does not reject mixed run names, manifests, datasets, commits, or non-25-step artifacts. **Concrete change:** introduce and validate an explicit pass role/coverage specification; require exact D11 seeds, modes, cohort, 25 sampling steps, S3 identity, manifest hash, and run provenance for each role; derive `C₃¹⁰⁰` only from validated S3-segment-final passes.

3. **D3 — MAJOR — The shipped CLI cannot evaluate the stronger full-set claim.** It derives canonical windows, checkpoints, and flags but never derives or passes `full_set_windows`; `main` exposes no option to supply them, so even an all-window artifact yields `evaluable: false` (`src/maxdiffusion/overfit100_success_statistic.py:421`, `src/maxdiffusion/overfit100_success_statistic.py:439`, `src/maxdiffusion/overfit100_success_statistic.py:452`). Separately, `_full_set_claim` becomes evaluable after any matching measurement instead of requiring the prescribed all-window pass (`src/maxdiffusion/overfit100_success_statistic.py:373`, `src/maxdiffusion/overfit100_success_statistic.py:385`). **Concrete change:** derive all 1,629 window keys from the authenticated manifest, require complete seed-0/correct coverage at `c*`, and feed that fixed cohort into the CLI-generated verdict.

4. **D4 — MAJOR — Separate canonical and all-window passes at the same checkpoint overwrite each other by default.** Output paths are keyed only by checkpoint step, while aggregation and summaries use fixed filenames opened for replacement; both passes default to the same validation root (`src/maxdiffusion/generate_wan_side_adapter.py:1512`, `src/maxdiffusion/generate_wan_side_adapter.py:1593`, `src/maxdiffusion/generate_wan_side_adapter.py:522`, `bash_scripts/validate_wan_overfit100.sh:95`). **Concrete change:** include the validated pass role or immutable pass ID in the path, refuse existing artifacts unless byte-identical, and preserve canonical and full-set inputs as separate immutable artifacts.

5. **D5 — MINOR — Systematic auxiliary-RGB failure has no run-level visibility.** Per-row failures are swallowed as intended, but the summary reports ceilings only when at least one succeeds and the final log gives no auxiliary coverage warning (`src/maxdiffusion/generate_wan_side_adapter.py:1297`, `src/maxdiffusion/generate_wan_side_adapter.py:1440`, `src/maxdiffusion/generate_wan_side_adapter.py:1605`). **Concrete change:** record requested/ok/failed counts and coverage fraction, aggregate failure reasons, and emit a loud warning when coverage is incomplete—especially when zero.

6. **D6 — MINOR — The mirrored window-name contract is claimed tested but has no cross-module parity test.** The evaluator says parity is pinned, while the builder remains an independent implementation (`src/maxdiffusion/generate_wan_side_adapter.py:657`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:238`). **Concrete change:** add the admitted parity/round-trip test against the builder, including the committed cohort and a six-digit start; make the parser accept the builder’s “at least five digits” format.

## Scrutiny-list rulings

1. **CHANGE** — subset-derived denominators are unacceptable; every S3 artifact/verdict must carry or derive the fixed 100-key cohort with missing coverage explicit.
2. **ACCEPT** — `c*` remains the headline-threshold argmax even for a partial verdict; `c_star_partial` is diagnostic, and the full-set gate reads plan-literal `c*`.
3. **ACCEPT** — absolute tolerance `1e-9` is safely fail-loud; it cannot silently average divergent TPU measurements.
4. **ACCEPT** — the five-array positional extension is scoped to OVERFIT100, while the default four-array exp_01 path remains intact and characterized.
5. **CHANGE** — add summary-level auxiliary coverage and a loud incomplete/zero-coverage log warning.
6. **ACCEPT** — draining roughly 100 MB to detect duplicate names and guarantee sparse-selection completeness is a reasonable correctness cost.
7. **CHANGE** — add the one-line builder/evaluator parity test and parser round-trip coverage.
8. **SPLIT** — `assert_ssim_available` and `_overfit100_rollout_fn` are good; `assert_flagged_windows_selected` must validate membership in the fixed cohort rather than require coverage in this particular pass.
9. **Deviation 1 — ACCEPT** — `eval_windows` as a pyconfig key is the correct constraint-compatible interface.
10. **Deviation 2 — ACCEPT-WITH-CHANGES** — the thin D2 CLI is appropriate, but it must receive or derive authoritative cohort and pass-role inputs.
11. **Deviation 3 — ACCEPT** — separate per-window loss files preserve exp_01’s aggregate schema and are cleaner than widening it.
12. **Deviation 4 — ACCEPT** — collapse numerically identical repeats and refuse conflicts; never average duplicate measurements.

---

# Strengthening record (2026-07-30 — Claude Opus 5, Coder)

All six findings **accepted and fixed**, all twelve rulings actioned, tests written first.
Suite: **996 passed / 2 skipped** (up from 943+2; +53 tests, 48 of them the new
`test_overfit100_eval_contracts.py`). The unifying change: **the verdict is now derivable only
from authenticated, role-validated, fixed-cohort inputs.**

## D1 — BLOCKER, pass-derived denominator — **FIXED**

The canonical cohort is derived from the manifest, never from a selection:

- `overfit100_success_statistic.canonical_cohort_from_manifest` / `all_window_keys_from_manifest`
  are now the SINGLE definition of the cohort math (`canonical_window_start` moved there too);
  `generate_wan_side_adapter` imports them and adds only the file read
  (`manifest_canonical_cohort` / `manifest_all_window_keys`), with gen↔statistic equality pinned
  by a parity test on the committed manifest (100 canonical / 1,629 built keys, matching
  `totals`).
- Artifact **schema v2** (`overfit100_eval_aggregation_v2`; a v1 artifact is refused, not
  reinterpreted) carries `canonical_cohort` (ALL cohort keys), `cohort_size`, `covered_windows`,
  `covered_canonical_windows`, `missing_canonical_windows`, `manifest_sha256`, `eval_pass_role`
  and `role_validation`. A canonical window outside the cohort is a hard error.
- The verdict CLI **derives** the cohort from the manifest it is handed, **verifies** every
  artifact's recorded `manifest_sha256` against `sha256(manifest bytes)` — the same hash cycle C
  bound to the published dataset — and passes the derived cohort as `require_cohort`, which
  `evaluate_success` enforces. A sparse pass claiming segment-final status is refused twice over
  (role validation + denominator check), naming both sizes.
- Ruling 8's SPLIT: `assert_flagged_windows_selected` → **`assert_flagged_windows_in_cohort`**,
  membership in the FIXED cohort, not coverage in this pass.

## D2 — BLOCKER, operator-trusted role/coverage — **FIXED**

- New mandatory config key **`eval_pass_role`** ∈ {`s2_gate`, `s3_intermediate`,
  `s3_segment_final`, `s3_full_set`} (no default; the launcher requires it with `:?`).
- `role_requirements` / `pass_role_plan_reasons` encode each cell's D11 contract: exact seeds,
  exact modes, cohort scope, and `sampling_steps == 25`. The driver calls
  `assert_pass_role_plan` **before the 5B load** (booby-trapped-loader tests prove it), and
  `validate_artifact_role` re-derives the same verdict from the written artifact plus a
  **complete `(window, seed, mode)` row grid** — nothing the artifact says about itself is
  trusted.
- `C3_100` = `segment_final_checkpoints_from_artifacts`: only role-validated
  `s3_segment_final` artifacts; S2/intermediate/full-set and self-mislabeled passes are excluded,
  and an empty result raises naming each rejection. `assert_artifacts_consistent` refuses mixed
  `run_name` / `manifest_sha256` / `eval_data_dir` / `commit` and any non-25-step artifact.
- The CLI feeds the canonical statistic **segment-final rows only**
  (`rows_from_artifacts(..., roles=(SEGMENT_FINAL_ROLE,))`), so a 1-seed full-set row can never
  enter a 3-seed median and the two tiers never collide as "duplicate" measurements.
- The shipped default (`context_modes: 'correct'`) now **fails** a segment-final pass loudly
  instead of silently producing an incomplete control set.

## D3 — MAJOR, full-set tier not operable — **FIXED**

`full_set_windows` is gone; `evaluate_success(full_set={"windows", "rows"})` takes the tier's
input from `full_set_input_from_artifacts` — the role-validated `s3_full_set` artifacts only.
`_full_set_claim` now requires **complete** seed-0/correct coverage at `c*`: partial coverage is
`evaluable: False` with a reason, never a low fraction. `eval_windows` gained the **`all`** spec
(a 1,629-name list is not a usable CLI value), without which the full-set pass was not even
expressible. The CLI derives all 1,629 keys and establishes the stronger tier end to end (tested
with a 12-window synthetic manifest, and the incomplete case tested separately).

## D4 — MAJOR, colliding artifacts — **FIXED**

`overfit100_step_root` puts the validated role in the path (`step_002500_s3_segment_final`), so
the canonical and full-set passes at one checkpoint are separate artifacts. `_write_json_immutable`
/ `_write_text_immutable` (used for `aggregation.json`, `summary.json`, `summary.csv`) skip a
byte-identical rewrite — an infra retry is a no-op — and **raise** when the bytes differ, naming
the path. Driver-level test: rerunning the same pass is idempotent; a changed result at the same
path is refused and the original evidence is intact. Videos stay overwritable by design (mp4
muxing is not byte-stable) and are documented as such.

## D5 — MINOR, invisible auxiliary failure — **FIXED**

`_aux_coverage` adds `{requested, ok, failed, coverage_fraction, failure_reason_counts}` to
`summary.json`, counting only rows that actually requested the metric (`requested == 0` means
"not asked", not "all failed"). `aux_coverage_log_lines` emits a **WARNING** block (with per-reason
counts) when coverage < 1.0 and an **ERROR** block when zero, including the "check gsutil/ffmpeg
and the manifest video fingerprints" pointer; the driver logs them after the summary.

## D6 — MINOR, unpinned window-name parity — **FIXED**

Parity is now executable: the parser regex accepts the builder's "at least five digits"
(`\d{5,}`) and the name must **round-trip** through `overfit100_window_name`, so
`ep100_v0_s000004` is refused as an ambiguous second spelling. Tests compare against the
builder's own `window_name` across every committed episode (first/second/last window), for
six-digit starts (100000, 123456), and round-trip every committed canonical name.

## Rulings

| # | Ruling | Action |
|---|---|---|
| 1 | CHANGE (fixed cohort) | D1 |
| 2 | ACCEPT (`c*` semantics) | unchanged |
| 3 | ACCEPT (`1e-9` tolerance) | unchanged |
| 4 | ACCEPT (five-array extension) | unchanged |
| 5 | CHANGE (aux coverage) | D5 |
| 6 | ACCEPT (full drain) | unchanged |
| 7 | CHANGE (parity test) | D6 |
| 8 | SPLIT (flags vs cohort) | `assert_flagged_windows_in_cohort`; ssim gate + jit seam kept |
| 9 | ACCEPT (`eval_windows` key) | unchanged |
| 10 | ACCEPT-WITH-CHANGES (CLI inputs) | landed via D1–D3 (manifest + roles) |
| 11 | ACCEPT (per-window loss file) | unchanged |
| 12 | ACCEPT (duplicate policy) | unchanged |

## Verification

- **Red evidence.** 46 failed / 37 passed before implementation (the new contracts file plus the
  updated statistic tests); a second red round for the driver role/immutability tests.
- **Mutation spot-checks, 9/9 caught** (each applied to the real module, then restored
  byte-identically): artifact cohort ← pass coverage → 1 fail; `require_cohort` dropped in the CLI
  → 1; `C3_100` ← every artifact's step → 2; driver role validation removed → 7; partial full-set
  scored → 1; role dropped from the path → 5; immutability guard removed → 3; aux warning silenced
  → 1; parser round-trip check removed → 1. **Three of these initially survived** (artifact
  cohort, CLI `require_cohort`, `C3_100` wiring) and exposed genuine test gaps — three tests were
  added (subset-coverage artifact, an intermediate-artifact CLI case, and wiring deletion guards)
  before the record was written.
- **Static.** `black --line-length 119 --target-version py312`, `ruff check`, `py_compile`,
  `bash -n` on both arms, `git diff --check`, `yaml.safe_load` (205 keys) all clean.
- **Real pyconfig parse** with the launcher's full override set: role parses, cohorts derive
  100/1,629 from the committed manifest, `assert_pass_role_plan` validates the segment-final cell
  (100 of 100), the same selection is **refused** for `s3_full_set` (1,529 windows missing), and
  `step_root` carries the role.

## Cannot-validate-until-TPU (updated)

Unchanged from the cycle-D report, plus: (a) `_write_json_immutable`'s read-back on real GCS
objects (an interrupted multi-host write could leave a partial object that reads as "different"),
(b) `_sha256_of_file` on a `gs://` manifest, and (c) the S3 cost of the now-mandatory
segment-final coverage — 100 windows x 3 modes x 3 seeds x 25 steps per segment-final checkpoint
**plus** a separate 1,629-window full-set pass for the stronger tier; the Planner should
extrapolate that from S2 timings before approving either pass.

