# Codex code review — eval-resume pass 3 (on the hardening)

- **Date:** 2026-07-31
- **Commit under review:** `2b0fd30` (eval-resume hardening)
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 2 BLOCKER (signature omits quantization / resolved weight source / scheduler num_train_timesteps + ckpt-dir slash normalization; `_PUBLISHED` marker unauthenticated + published mode not compare-only), 1 MINOR (trailing-slash marker size unchecked).
- **Converging:** pass-1 five findings → pass-2 six → pass-3 three, each narrower. Publication-resume state walk (W2) otherwise verified across all preemption points; R-B/R-D/R-E/R-H and pass-2 items preserved. Reviewer supplied the full operational relaunch checklist (recorded verbatim below) — includes cancel-and-confirm of old jobs and archiving old role dirs before relaunch into clean paths.

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

1. **BLOCKER — run identity remains under-bound** — [generate_wan_side_adapter.py:1570](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1570): the signature still omits effective transformer quantization (`use_qwix_quantization`, quantization mode/rules), although it can replace the rollout graph at [wan_pipeline.py:406](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/pipelines/wan/wan_pipeline.py:406); it also omits the resolved transformer-weight source and `scheduler.config.num_train_timesteps`, which directly scales rollout timesteps at [generate_wan_side_adapter.py:1249](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1249). Fix: bind a canonical effective runtime identity covering the scheduler’s complete rollout configuration, resolved pipeline/transformer sources, and effective quantizer configuration; mutation-test at least quantization enablement and scheduler timestep count. The checkpoint-directory field itself cannot be spoofed—exact field/hash comparison rejects aliases—but `_resolved_checkpoint_dir` does not normalize trailing slashes despite its comment claiming equivalent spellings agree.

2. **BLOCKER — marker existence neither authenticates the marker nor proves the artifact set is complete** — [generate_wan_side_adapter.py:1708](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1708): any `_PUBLISHED` object, including an empty/foreign marker, selects published mode without parsing it or requiring all three final artifacts. Published mode then calls ordinary immutable writers at [generate_wan_side_adapter.py:2628](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2628), which create missing files; therefore a marker-backed directory can mutate, and an exact copied marker can make an incomplete directory succeed from recomputation. An empty/different marker fails only at the final marker comparison, after missing artifacts may already have been written. Fix: marker presence must require the exact final-artifact set and strict marker schema/key/type/content validation; published-mode comparisons must be compare-only and refuse every missing artifact before any write. Add empty-marker, foreign-marker-at-entry, and marker-plus-missing-artifact snapshot/booby-trap tests.

3. **MINOR — the GCS directory-marker test does not establish zero-byte classification** — [generate_wan_side_adapter.py:1926](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1926): every filename ending in `/` is ignored without checking its size, so a non-empty foreign object with that name is admitted as a marker. The test at [test_overfit100_context_modes.py:2375](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:2375) mocks names but never sizes. Fix: tolerate trailing-slash objects only when `stat().length == 0`, fail closed on stat errors, and test zero- versus nonzero-byte objects.

Verified claims:

1. **Fail:** exact signature keys, exact types, field equality, self-hash, and independent expected-hash equality work, but the runtime identity is still incomplete.
2. **Partial:** a valid complete published directory suppresses video, per-window metric, and staging writes; malformed/incomplete marker-backed directories violate that guarantee.
3. **Partial:** before aggregation retries from staging; every legitimate between-artifact prefix and the all-three-before-marker state repairs from staging; after-marker retries recompute and compare; lost/corrupt staging during partial publication refuses—but invalid marker states remain unsafe.
4. **Partial:** listing errors fail closed and the private sentinel matches TensorFlow 2.21’s `walk` control flow; actual zero-byte marker classification is not implemented.
5. **Pass:** aux recovery now correctly requires a separate backfill artifact and never promises mutation of published evidence.
6. **Pass:** the prior trailing blank line is removed and `git diff --check 2b0fd30^ 2b0fd30` is clean.
7. **Qualified pass:** the two source guards honestly disclose redundancy and are acceptable as static defense-in-depth contracts, not behavioral mutation evidence; the recorded 1162/2 suite was not independently rerun because this sandbox has no writable temporary directory.
8. **Pass:** R-B row order/parity, R-D single-process staging gate, R-E partial-window GT decode, role validation placement, aggregation schema, and untouched success-statistic module are preserved.

Fitness ruling: `2b0fd30` is **not fit** to carry the segment-final or full-set relaunches. Do not launch this SHA.

Operational relaunch checklist:

- Land and review a corrected pushed SHA, record the exact commands in `_command.md`/`_worklog.md`, and obtain renewed approval because the code changed.
- Cancel both old segment-final and full-set job families; poll until no queued, provisioning, active, retrying, or suspending attempts remain; confirm their worker processes and TPU resources are gone.
- Inspect both `step_002500_s3_segment_final` and `step_002500_s3_full_set`. Because old staging binds older commits, archive/move old role directories and relaunch into clean paths. Never clear only selected files from a partial publication.
- Explicit common environment: `COMMIT=<corrected 40-hex SHA>`, `OVERFIT100_EVAL_RESUME=1`, `RUN_NAME=wan-overfit100-s3-20260730`, train/eval data `train100`, `EXPECTED_WINDOWS=1629`, `NUM_TEXT_SLOTS=100`, the exact checkpoint directory, `CHECKPOINT_STEP=2500`, pinned manifest/model revision, `EVAL_AUX_RGB=True`, `WRITE_VIDEOS=False` unless videos are intentionally required, and unchanged `fps=16`.
- Segment-final: `EVAL_PASS_ROLE=s3_segment_final`, `EVAL_WINDOWS=canonical`, `ROLLOUT_SEEDS=0,1,2`, `CONTEXT_MODES=correct,null,shuffled`, `CONTEXT_SHUFFLE_SEED=0`.
- Full-set: `EVAL_PASS_ROLE=s3_full_set`, `EVAL_WINDOWS=all`, `ROLLOUT_SEEDS=0`, `CONTEXT_MODES=correct`, `CONTEXT_SHUFFLE_SEED=0`.
- Keep every signature-bound flag byte-for-byte stable across retries and require ffmpeg/gsutil readiness when auxiliary coverage is expected.

Status

- Review complete; no local subprocesses or background agents are running.
- Existing remote queue jobs were not queried; treat them as potentially live until explicitly confirmed terminated.
- Earliest safe steer: now.
tokens used
283,566
REQUEST-REVISION

1. **BLOCKER — run identity remains under-bound** — [generate_wan_side_adapter.py:1570](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1570): the signature still omits effective transformer quantization (`use_qwix_quantization`, quantization mode/rules), although it can replace the rollout graph at [wan_pipeline.py:406](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/pipelines/wan/wan_pipeline.py:406); it also omits the resolved transformer-weight source and `scheduler.config.num_train_timesteps`, which directly scales rollout timesteps at [generate_wan_side_adapter.py:1249](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1249). Fix: bind a canonical effective runtime identity covering the scheduler’s complete rollout configuration, resolved pipeline/transformer sources, and effective quantizer configuration; mutation-test at least quantization enablement and scheduler timestep count. The checkpoint-directory field itself cannot be spoofed—exact field/hash comparison rejects aliases—but `_resolved_checkpoint_dir` does not normalize trailing slashes despite its comment claiming equivalent spellings agree.

2. **BLOCKER — marker existence neither authenticates the marker nor proves the artifact set is complete** — [generate_wan_side_adapter.py:1708](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1708): any `_PUBLISHED` object, including an empty/foreign marker, selects published mode without parsing it or requiring all three final artifacts. Published mode then calls ordinary immutable writers at [generate_wan_side_adapter.py:2628](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2628), which create missing files; therefore a marker-backed directory can mutate, and an exact copied marker can make an incomplete directory succeed from recomputation. An empty/different marker fails only at the final marker comparison, after missing artifacts may already have been written. Fix: marker presence must require the exact final-artifact set and strict marker schema/key/type/content validation; published-mode comparisons must be compare-only and refuse every missing artifact before any write. Add empty-marker, foreign-marker-at-entry, and marker-plus-missing-artifact snapshot/booby-trap tests.

3. **MINOR — the GCS directory-marker test does not establish zero-byte classification** — [generate_wan_side_adapter.py:1926](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1926): every filename ending in `/` is ignored without checking its size, so a non-empty foreign object with that name is admitted as a marker. The test at [test_overfit100_context_modes.py:2375](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:2375) mocks names but never sizes. Fix: tolerate trailing-slash objects only when `stat().length == 0`, fail closed on stat errors, and test zero- versus nonzero-byte objects.

Verified claims:

1. **Fail:** exact signature keys, exact types, field equality, self-hash, and independent expected-hash equality work, but the runtime identity is still incomplete.
2. **Partial:** a valid complete published directory suppresses video, per-window metric, and staging writes; malformed/incomplete marker-backed directories violate that guarantee.
3. **Partial:** before aggregation retries from staging; every legitimate between-artifact prefix and the all-three-before-marker state repairs from staging; after-marker retries recompute and compare; lost/corrupt staging during partial publication refuses—but invalid marker states remain unsafe.
4. **Partial:** listing errors fail closed and the private sentinel matches TensorFlow 2.21’s `walk` control flow; actual zero-byte marker classification is not implemented.
5. **Pass:** aux recovery now correctly requires a separate backfill artifact and never promises mutation of published evidence.
6. **Pass:** the prior trailing blank line is removed and `git diff --check 2b0fd30^ 2b0fd30` is clean.
7. **Qualified pass:** the two source guards honestly disclose redundancy and are acceptable as static defense-in-depth contracts, not behavioral mutation evidence; the recorded 1162/2 suite was not independently rerun because this sandbox has no writable temporary directory.
8. **Pass:** R-B row order/parity, R-D single-process staging gate, R-E partial-window GT decode, role validation placement, aggregation schema, and untouched success-statistic module are preserved.

Fitness ruling: `2b0fd30` is **not fit** to carry the segment-final or full-set relaunches. Do not launch this SHA.

Operational relaunch checklist:

- Land and review a corrected pushed SHA, record the exact commands in `_command.md`/`_worklog.md`, and obtain renewed approval because the code changed.
- Cancel both old segment-final and full-set job families; poll until no queued, provisioning, active, retrying, or suspending attempts remain; confirm their worker processes and TPU resources are gone.
- Inspect both `step_002500_s3_segment_final` and `step_002500_s3_full_set`. Because old staging binds older commits, archive/move old role directories and relaunch into clean paths. Never clear only selected files from a partial publication.
- Explicit common environment: `COMMIT=<corrected 40-hex SHA>`, `OVERFIT100_EVAL_RESUME=1`, `RUN_NAME=wan-overfit100-s3-20260730`, train/eval data `train100`, `EXPECTED_WINDOWS=1629`, `NUM_TEXT_SLOTS=100`, the exact checkpoint directory, `CHECKPOINT_STEP=2500`, pinned manifest/model revision, `EVAL_AUX_RGB=True`, `WRITE_VIDEOS=False` unless videos are intentionally required, and unchanged `fps=16`.
- Segment-final: `EVAL_PASS_ROLE=s3_segment_final`, `EVAL_WINDOWS=canonical`, `ROLLOUT_SEEDS=0,1,2`, `CONTEXT_MODES=correct,null,shuffled`, `CONTEXT_SHUFFLE_SEED=0`.
- Full-set: `EVAL_PASS_ROLE=s3_full_set`, `EVAL_WINDOWS=all`, `ROLLOUT_SEEDS=0`, `CONTEXT_MODES=correct`, `CONTEXT_SHUFFLE_SEED=0`.
- Keep every signature-bound flag byte-for-byte stable across retries and require ffmpeg/gsutil readiness when auxiliary coverage is expected.

Status

- Review complete; no local subprocesses or background agents are running.
- Existing remote queue jobs were not queried; treat them as potentially live until explicitly confirmed terminated.
- Earliest safe steer: now.
```

---

## Strengthening record (2026-07-31 — Claude Opus 5, Coder)

All three findings **accepted and fixed**, tests first (12 failures before implementation). Suite:
**1194 passed / 2 skipped** (from 1162+2; +32). `overfit100_success_statistic.py` untouched, the
aggregation schema unchanged, and the pass-2 behaviours the reviewer verified (publication-resume
state walk, aux-recovery text, R-B/R-D/R-E/R-H) are preserved and still covered.

### 1 (BLOCKER) — the effective runtime identity is now bound

Nine fields added to the signature: `num_train_timesteps` (read off the scheduler that scales every
rollout timestep), the effective quantizer configuration (`use_qwix_quantization`, `quantization`,
`qwix_module_path`, and the three calibration methods — quantization can replace the rollout graph
in `wan_pipeline.quantize_transformer`), and the resolved transformer weight source
(`transformer_weight_source`, taken from `wan_transformer_pretrained_model_name_or_path` which is
what the Wan loader actually reads, plus `from_pt`).

`_resolved_checkpoint_dir` now **normalizes trailing slashes**, so the comment's promise that
equivalent spellings agree is true rather than merely claimed — a stray slash in a launcher can no
longer force a full recompute. (The reviewer's point stands that aliases were never spoofable; this
was a comment/behaviour mismatch and retry brittleness.)

**Tests:** `test_signature_binds_the_effective_runtime_identity`,
`test_every_runtime_identity_field_is_enforced` (9 parametrized cases, including quantization
enablement and the timestep count as required), `test_resolved_checkpoint_dir_normalizes_trailing_slashes`.

### 2 (BLOCKER) — the marker is authenticated, and published mode is compare-only

`overfit100_publication_state(step_root, expected=...)` now selects `published` only when BOTH hold:

* the **complete** final-artifact set is present (a marker beside a missing `summary.csv` is a hard
  failure, not a licence to recreate it), and
* the marker **authenticates** the directory: exact schema, exact key set, exact types, agreement
  with this run on `eval_pass_role` / `checkpoint_step` / `manifest_sha256` / `n_rows`, and an
  `aggregation_sha256` equal to the sha256 of the `aggregation.json` sitting beside it — which is
  what stops a marker copied from another directory from validating.

Every failure raises **at entry, before any write**, with guidance naming the directory and telling
the operator to archive it whole rather than repair it file by file.

Published mode is now **compare-only**: `_write_json_immutable` / `_write_text_immutable` /
`_write_rows_csv` take `compare_only=True`, under which a missing artifact is refused instead of
created, so the create-if-absent path is unreachable there.

> **Retracted by pass 4 — see `overfit100_codex_code_eval-resume4_review.md` finding 2.** This
> record originally claimed that the marker's `run_signature_sha256` was deliberately non-binding
> and that a newer commit could still re-verify a published directory. That was **false as
> written**: compare-only mode regenerates and byte-compares the marker (rebinding the hash), and
> `aggregation.json` embeds `COMMIT`, so a newer commit fails the aggregation comparison first.
> The reviewer ruled the stricter behaviour SAFE and it is now documented as the real contract:
> published re-verification is **same-signature/same-commit only**, and a newer commit is expected
> to refuse.

**Tests:** `test_the_marker_records_what_it_authenticates`,
`test_an_unparseable_or_foreign_marker_hard_fails_at_entry` (6 shapes: empty, blank, corrupt, list,
foreign schema, no fields), `test_a_marker_valid_in_every_way_except_its_schema_tag_hard_fails`,
`test_a_marker_with_an_extra_key_hard_fails`,
`test_a_marker_that_does_not_bind_this_run_hard_fails_at_entry` (5 binding fields),
`test_a_marker_with_a_missing_final_artifact_hard_fails_before_any_write` (snapshot + write
booby-traps), `test_published_mode_is_compare_only_and_never_creates`,
`test_compare_only_writers_refuse_a_missing_artifact`, plus
`test_a_partial_publication_is_still_repairable_after_the_marker_rules` as a regression guard that
the new strictness did not break the pass-2 repair path.

### 3 (MINOR) — trailing-slash objects are markers only when zero bytes

A trailing-slash entry is now tolerated only when `stat().length == 0`; a non-empty one is an
offender, and a **stat failure is fatal** (it cannot be classified, so it cannot be skipped).

**Tests:** `test_a_zero_byte_trailing_slash_object_is_a_directory_marker`,
`test_a_nonzero_trailing_slash_object_is_refused`,
`test_a_stat_failure_on_a_candidate_marker_fails_closed` — all mocking **sizes**, not just names, and
the pass-2 marker test now stubs `stat` too.

### Verification

* **Red evidence:** 12 failed / 183 passed before implementation.
* **Mutation spot-checks, 12/12 caught** (each restored byte-identically): marker schema validation
  dropped → 1; marker key-set dropped → 1; marker/aggregation binding dropped → 1; marker run-binding
  dropped → 4; complete-artifact-set requirement dropped → 1; published-mode create-if-absent
  restored → 1; quantization field unbound → 99; timestep count unbound → 100; zero-byte size check
  dropped → 1; trailing-slash normalization dropped → 1. **Two initially survived** — the marker
  schema and key-set checks masked each other (a foreign-schema marker was rejected by the key-set
  check and vice versa) — so two isolating cases were added (a marker valid in every way except its
  tag; a marker with one extra key) before this record was written.
* **Static:** black (`--line-length 119 --target-version py312`), ruff, `py_compile`, and
  `git diff --check` **on the staged tree** (the lesson from pass 2's false claim) all clean.
