# Codex code review — eval-resume pass 4 (on the finalization)

- **Date:** 2026-08-01
- **Commit under review:** `9c12a1f` (eval-resume finalization)
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 1 BLOCKER (raw `checkpoint_dir` signature field unnormalized, defeating the resolved-field slash normalization), 1 MINOR (published re-verification is effectively same-signature/same-commit only — stricter than documented and SAFE, but the non-binding claim is false as written; amend contract text).
- **Now passing:** X2 marker authentication (no preemption/foreign-writer/copied-marker sequence can publish an incomplete or mutated dir under the documented single-writer invariant), X3 compare-only published mode, X5 all prior protections + publication-resume walk, X6 the landed segment-final result is valid and must not be relaunched. X1: no further rollout-affecting input missing beyond the slash issue.
- **Amended single-job relaunch checklist recorded below (full-set only; segment-final preserved).**

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

## Findings

1. **BLOCKER — trailing-slash normalization is defeated by the raw signature field** — [`generate_wan_side_adapter.py:1885`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1885): although `resolved_checkpoint_dir` is normalized, `checkpoint_dir` still records the unnormalized string, and exact admission rejects any field mismatch. The test only compares `_resolved_checkpoint_dir` outputs at [`test_overfit100_context_modes.py:2498`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:2498), not the complete signatures. Fix: canonicalize/remove the raw field and add an end-to-end test proving slash variants produce equal signatures and admit each other’s staged rows.

2. **MINOR — the claimed `run_signature_sha256` non-binding is not effective** — [`generate_wan_side_adapter.py:2785`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2785): published mode regenerates and byte-compares `_PUBLISHED`, thereby requiring the current run-signature hash to equal the recorded one. Moreover, `aggregation.json` embeds current `COMMIT` at [`generate_wan_side_adapter.py:2357`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2357), so a newer commit fails aggregation comparison first. Fix: either implement genuine cross-commit comparison semantics, or amend the contract to state that published re-verification is effectively same-signature/same-commit only. This is stricter and safe, but claim 4 is false as written.

## Verified claims

- **X1 — Partial:** all nine new effective-runtime fields are correctly built and enforced; no further rollout-affecting input is missing in this eval path, but checkpoint slash equivalence remains broken.
- **X2 — Pass:** malformed, copied, mis-bound, hash-mismatched, or incomplete marker states fail without artifact writes; under the documented single-writer invariant, no preemption sequence can publish an incomplete or mutated directory.
- **X3 — Pass:** published mode suppresses videos, per-window metrics, and staging; all three final artifacts and `_PUBLISHED` use `compare_only=True`, making create-if-absent unreachable.
- **X4 — Fail as claimed:** omitting the hash from entry authentication is safe, but later exact marker and aggregation comparisons effectively rebind it and prevent the stated newer-commit re-verification.
- **X5 — Pass:** R-B ordering/parity, R-D single-process gating, R-E partial-window decode behavior, R-H schema/statistic invariants, pass-2/pass-3 protections, and the publication-resume walk remain intact. AST compilation and `git diff --check` passed; the recorded 1194/2 suite could not be independently rerun because this environment lacks pytest.
- **X6 — Pass:** `overfit100_success_statistic.py` is unchanged from `e27fdc3` through HEAD, and resume changes only add staging/publication control around the unchanged primary-metrics path. The [segment-final result](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_results.md:24) remains correctly read as 0/100 at 0.95, mean `m_corr=0.8133`; it needs no relaunch.

**Fitness ruling:** `9c12a1f` is not fit to carry the full-set relaunch until finding 1 is fixed and reviewed. The landed segment-final remains valid and must not be relaunched.

**Amended single-job relaunch checklist:**

- Land and review the corrected pushed SHA; record the exact command in `_command.md` and `_worklog.md`; obtain renewed approval.
- Cancel only the old **full-set** job family and confirm every attempt, worker, and TPU resource is gone. Do not cancel or relaunch segment-final.
- Preserve `step_002500_s3_segment_final`. Inspect and archive/move the entire old `step_002500_s3_full_set` role directory, then use a clean path—never selectively clear staging or final files.
- Set `COMMIT=<corrected 40-hex SHA>`, `OVERFIT100_EVAL_RESUME=1`, `RUN_NAME=wan-overfit100-s3-20260730`, both data directories to `train100`, `EXPECTED_WINDOWS=1629`, `NUM_TEXT_SLOTS=100`, exact checkpoint directory, `CHECKPOINT_STEP=2500`, and the pinned manifest/model revision.
- Full-set-only settings: `EVAL_PASS_ROLE=s3_full_set`, `EVAL_WINDOWS=all`, `ROLLOUT_SEEDS=0`, `CONTEXT_MODES=correct`, `CONTEXT_SHUFFLE_SEED=0`.
- Keep every signature-bound flag stable across retries; retain `EVAL_AUX_RGB=True`, `WRITE_VIDEOS=False`, `fps=16`, with ffmpeg and gsutil ready.

## Status

- No subprocesses, tests, background agents, or TPU jobs are running.
- Existing remote full-set jobs were not queried; treat them as potentially live until termination is confirmed.
- Earliest steer: now.
tokens used
352,367
REQUEST-REVISION

## Findings

1. **BLOCKER — trailing-slash normalization is defeated by the raw signature field** — [`generate_wan_side_adapter.py:1885`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1885): although `resolved_checkpoint_dir` is normalized, `checkpoint_dir` still records the unnormalized string, and exact admission rejects any field mismatch. The test only compares `_resolved_checkpoint_dir` outputs at [`test_overfit100_context_modes.py:2498`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:2498), not the complete signatures. Fix: canonicalize/remove the raw field and add an end-to-end test proving slash variants produce equal signatures and admit each other’s staged rows.

2. **MINOR — the claimed `run_signature_sha256` non-binding is not effective** — [`generate_wan_side_adapter.py:2785`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2785): published mode regenerates and byte-compares `_PUBLISHED`, thereby requiring the current run-signature hash to equal the recorded one. Moreover, `aggregation.json` embeds current `COMMIT` at [`generate_wan_side_adapter.py:2357`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2357), so a newer commit fails aggregation comparison first. Fix: either implement genuine cross-commit comparison semantics, or amend the contract to state that published re-verification is effectively same-signature/same-commit only. This is stricter and safe, but claim 4 is false as written.

## Verified claims

- **X1 — Partial:** all nine new effective-runtime fields are correctly built and enforced; no further rollout-affecting input is missing in this eval path, but checkpoint slash equivalence remains broken.
- **X2 — Pass:** malformed, copied, mis-bound, hash-mismatched, or incomplete marker states fail without artifact writes; under the documented single-writer invariant, no preemption sequence can publish an incomplete or mutated directory.
- **X3 — Pass:** published mode suppresses videos, per-window metrics, and staging; all three final artifacts and `_PUBLISHED` use `compare_only=True`, making create-if-absent unreachable.
- **X4 — Fail as claimed:** omitting the hash from entry authentication is safe, but later exact marker and aggregation comparisons effectively rebind it and prevent the stated newer-commit re-verification.
- **X5 — Pass:** R-B ordering/parity, R-D single-process gating, R-E partial-window decode behavior, R-H schema/statistic invariants, pass-2/pass-3 protections, and the publication-resume walk remain intact. AST compilation and `git diff --check` passed; the recorded 1194/2 suite could not be independently rerun because this environment lacks pytest.
- **X6 — Pass:** `overfit100_success_statistic.py` is unchanged from `e27fdc3` through HEAD, and resume changes only add staging/publication control around the unchanged primary-metrics path. The [segment-final result](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_results.md:24) remains correctly read as 0/100 at 0.95, mean `m_corr=0.8133`; it needs no relaunch.

**Fitness ruling:** `9c12a1f` is not fit to carry the full-set relaunch until finding 1 is fixed and reviewed. The landed segment-final remains valid and must not be relaunched.

**Amended single-job relaunch checklist:**

- Land and review the corrected pushed SHA; record the exact command in `_command.md` and `_worklog.md`; obtain renewed approval.
- Cancel only the old **full-set** job family and confirm every attempt, worker, and TPU resource is gone. Do not cancel or relaunch segment-final.
- Preserve `step_002500_s3_segment_final`. Inspect and archive/move the entire old `step_002500_s3_full_set` role directory, then use a clean path—never selectively clear staging or final files.
- Set `COMMIT=<corrected 40-hex SHA>`, `OVERFIT100_EVAL_RESUME=1`, `RUN_NAME=wan-overfit100-s3-20260730`, both data directories to `train100`, `EXPECTED_WINDOWS=1629`, `NUM_TEXT_SLOTS=100`, exact checkpoint directory, `CHECKPOINT_STEP=2500`, and the pinned manifest/model revision.
- Full-set-only settings: `EVAL_PASS_ROLE=s3_full_set`, `EVAL_WINDOWS=all`, `ROLLOUT_SEEDS=0`, `CONTEXT_MODES=correct`, `CONTEXT_SHUFFLE_SEED=0`.
- Keep every signature-bound flag stable across retries; retain `EVAL_AUX_RGB=True`, `WRITE_VIDEOS=False`, `fps=16`, with ffmpeg and gsutil ready.

## Status

- No subprocesses, tests, background agents, or TPU jobs are running.
- Existing remote full-set jobs were not queried; treat them as potentially live until termination is confirmed.
- Earliest steer: now.
```

---

## Strengthening record (2026-08-01 — Claude Opus 5, Coder)

Both residuals **accepted and fixed**, tests first (4 failures before implementation). Surgical by
design: two changes in `generate_wan_side_adapter.py` (one field dropped, contract text corrected)
plus tests and record edits. Suite: **1197 passed / 2 skipped** (from 1194+2; +3).

### 1 (BLOCKER) — the raw `checkpoint_dir` field is **dropped**, not canonicalized

Of the two options offered, I removed the field rather than normalizing it. The argument: the
checkpoint identity is the directory the restore actually reads, which is exactly
`resolved_checkpoint_dir` (already normalized, and already covering the empty-config fallback). The
raw config string is a *spelling* of that same fact, and carrying two representations of one fact is
precisely what produced this defect — normalizing both would leave the same duplication in place,
waiting for the next divergence. One canonical field cannot disagree with itself.

Nothing is lost: two raw spellings that resolve to the same directory ARE the same checkpoint
source, and any genuinely different directory still changes `resolved_checkpoint_dir`.

**Tests (end-to-end, whole-signature — the gap the reviewer named):**
`test_checkpoint_dir_slash_variants_produce_identical_signatures_end_to_end` asserts the full
signatures **and their hashes** are equal for `gs://b/ck` vs `gs://b/ck/`, then stages a row under
one and admits it under the other (both directions).
`test_the_driver_admits_a_slash_variant_retrys_staged_rows` proves the same through the real driver:
a retry whose `checkpoint_dir` gained a trailing slash **resumes** every row instead of recomputing.

### 2 (MINOR) — the re-verification contract now states what the code does

The claim that `run_signature_sha256` was non-binding was **false as written**, and the reviewer is
right about the mechanism: compare-only mode regenerates and byte-compares `_PUBLISHED` (rebinding
the hash), and `aggregation.json` embeds `COMMIT`, so a newer commit fails at the aggregation
comparison first. Per the ruling I did **not** implement cross-commit semantics — the stricter
behaviour is the safe one. The contract text now says so plainly in
`overfit100_published_marker`'s docstring: **published re-verification is same-signature and
same-commit only; a newer commit re-verifying a published directory is expected to refuse, and that
refusal is the fail-closed intent** (published evidence belongs to the run and the code that made
it; a different build publishes into a clean role directory). `overfit100_publication_state`'s
docstring explains why entry authentication deliberately omits the hash comparison — the later
artifact comparison enforces it with a better error, naming the artifact that differs. The pass-3
record's version of the claim is marked **retracted** with a pointer here.

**Tests:** `test_published_re_verification_is_same_commit_only` publishes, flips `COMMIT`, and
asserts the rerun **refuses** naming `aggregation.json` with the role directory byte-unchanged, then
restores the commit and shows the same-commit re-verification passing cleanly;
`test_the_contract_text_states_same_commit_re_verification` pins the wording so the retracted claim
cannot creep back.

### Verification

* **Red evidence:** 4 failed / 196 passed before implementation.
* **Mutation spot-checks, 3/3 caught** (each restored byte-identically): checkpoint-dir
  normalization dropped → 3 failures (the mutation the review asked for); the raw unnormalized
  `checkpoint_dir` field reinstated alongside the resolved one → 103; compare-only wiring removed
  → 1. The third initially survived — with entry authentication now rejecting a marker beside a
  missing artifact, the driver cannot reach publication with anything absent, so no behavioural test
  can distinguish `compare_only=True` from `False` there. It is kept as defence in depth and pinned
  with a source assertion, disclosed in the test body rather than left overclaiming.
* **Static:** black (`--line-length 119 --target-version py312`), ruff, `py_compile`, and
  `git diff --check` on the **staged** tree, all clean. `overfit100_success_statistic.py` untouched;
  aggregation schema unchanged.
