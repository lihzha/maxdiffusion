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

*(Strengthening record appended at round close.)*
