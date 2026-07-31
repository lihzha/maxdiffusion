# Codex code review — eval-resume follow-up (pass 2, on the strengthening)

- **Date:** 2026-07-31
- **Commit under review:** `99ee724` (eval-resume strengthening)
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 3 BLOCKER (signature still under-bound: flow sigmas/geometry/numerics/fps/resolved ckpt dir + exact key-set; completed role dir still mutable via videos/per-window metrics; publication non-transactional — any-one-artifact completion probe + aggregation-first write order can brick a role dir on mid-publication preemption), 3 MINOR (walk fails open on listing errors / GCS marker semantics; aux-recovery message promises a nonexistent path; strengthening record's git-diff-check claim false).
- **Also ruled:** deliberate non-fix of builder-level typing ACCEPTED (with follow-up suggestion); prior verified claims R-B/R-D/R-E/R-H re-confirmed; relaunch guidance — do NOT fly 99ee724; at relaunch time cancel and confirm termination of pre-resume jobs (single-writer).

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

1. **BLOCKER — run identity remains under-bound** — [generate_wan_side_adapter.py:1547](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1547): the signature omits `flow_sigma_min/max`, although they construct the scheduler at [wan_ti2v_side_adapter_trainer.py:261](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py:261) and directly enter rollout sigmas at [generate_wan_side_adapter.py:1243](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1243); latent geometry, `wan_max_sequence_length`, transformer numerical settings (`attention`/`precision`/flash configuration), video `fps`, and the resolved checkpoint directory are also unbound. Fix: derive the signature from resolved runtime objects/configuration, add these fields and mutation tests, and require the staged signature’s exact key set plus equality with the independently computed expected hash.

2. **BLOCKER — completed role directories can still be mutated** — [generate_wan_side_adapter.py:2291](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2291): completion disables staging, but the loop still overwrites videos and per-window `metrics.json` at [generate_wan_side_adapter.py:2418](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2418); `_save_video` explicitly uses `overwrite=True` at [generate_wan_side_adapter.py:549](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:549). Existing snapshot tests use `write_videos=False` at [test_overfit100_context_modes.py:947](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:947). Fix: when completed at entry, suppress every filesystem write until immutable final comparison, and add video-enabled booby-trap/snapshot tests.

3. **BLOCKER — final publication is neither transactional nor race-safe** — [generate_wan_side_adapter.py:1541](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1541): any one final artifact is treated as completion, while publication writes `aggregation.json` first at [generate_wan_side_adapter.py:2453](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2453). Preemption after that write disables staging on retry; nondeterministic aux can then fail immutable comparison before missing summaries are repaired. The one-time completion probe also races another writer. Fix: publish a commit marker last, treat only a complete marker-backed set as completed, resume partial publication from staged rows, and use generation-match/create-if-absent or equivalent single-writer protection.

4. **MINOR — recursive enumeration still fails open on listing errors and does not establish real GCS-marker behavior** — [generate_wan_side_adapter.py:1794](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1794): `tf.io.gfile.walk` is called without `onerror`; TensorFlow documents that listing errors are ignored by default, while Cloud Storage represents some folders using zero-byte objects. Fix: rethrow through `_staging_error`, avoid relying on prefix `exists`, explicitly define tolerated zero-byte marker forms, and test mocked GCS file/marker/error classifications. [TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf/io/gfile/walk), [Cloud Storage object documentation](https://docs.cloud.google.com/storage/docs/objects).

5. **MINOR — aux recovery documentation promises a path that does not exist** — [generate_wan_side_adapter.py:2329](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2329): clearing staging cannot recover a ceiling after final publication because completed-mode ignores staging and immutable aggregation rejects changed aux. Fix: document the actual new-output/backfill procedure or implement and test a ceiling-only backfill path.

6. **MINOR — the strengthening record’s clean-check claim is false** — [overfit100_codex_code_eval-resume_review.md:198](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_codex_code_eval-resume_review.md:198): `git diff --check 78819dc 99ee724` reports a new blank line at EOF. Fix: remove it and rerun the recorded check.

Claim verification:

1. **Partial:** the 40-hex COMMIT gate, exact known-field typing/comparison, and hash re-derivation work; signature completeness and exact key-set enforcement do not.
2. **Pass with stated trust boundary:** coercions, booleans-as-ints, domains, descriptor identity, context source, and duplicates are rejected; a type-valid physically inconsistent metric tuple is still admissible because no universal SSIM/MSE invariant or stored prediction exists.
3. **Fail:** completed-artifact detection precedes staging, but completed video-enabled directories remain mutable and partial final publication is unsafe.
4. **Partial:** both aux tests genuinely pin the documented admission/parity semantics; the advertised recovery guidance is inaccurate.
5. **Partial:** recursive shape enforcement, offender aggregation, root naming, and escape guidance work; listing failures and real GCS markers remain uncovered.
6. **Not independently reproduced:** the recorded 1131/2 suite and 11/11 mutations could not run because this read-only sandbox has no writable temporary directory; both changed Python files compiled successfully, but `git diff --check` failed.
7. **Pass:** `overfit100_success_statistic.py` is untouched, aggregation schema is unchanged, and role validation remains before rollout.
8. **Accepted:** builder-level belt-and-braces typing is not required for this commit because admitted staging rows are strictly typed and fresh rows are constructed in-process; reusing the admission validator would still be a worthwhile follow-up.
9. **Pass:** `_SUCCESS`, finite SSIM stubbing, and distinct video-manifest fixture corrections are present and appropriate.

Risk read: `99ee724` is not fit to carry segment-final or full-set passes. Do not relaunch this SHA. When a corrected commit is approved and relaunch begins, cancel and confirm termination of all pre-resume queue jobs—do not leave them racing against the new evaluator.

Status

- Review complete; no subprocesses or background agents are running.
- ETA: complete. Earliest safe steer: now.
tokens used
300,739
REQUEST-REVISION

1. **BLOCKER — run identity remains under-bound** — [generate_wan_side_adapter.py:1547](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1547): the signature omits `flow_sigma_min/max`, although they construct the scheduler at [wan_ti2v_side_adapter_trainer.py:261](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py:261) and directly enter rollout sigmas at [generate_wan_side_adapter.py:1243](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1243); latent geometry, `wan_max_sequence_length`, transformer numerical settings (`attention`/`precision`/flash configuration), video `fps`, and the resolved checkpoint directory are also unbound. Fix: derive the signature from resolved runtime objects/configuration, add these fields and mutation tests, and require the staged signature’s exact key set plus equality with the independently computed expected hash.

2. **BLOCKER — completed role directories can still be mutated** — [generate_wan_side_adapter.py:2291](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2291): completion disables staging, but the loop still overwrites videos and per-window `metrics.json` at [generate_wan_side_adapter.py:2418](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2418); `_save_video` explicitly uses `overwrite=True` at [generate_wan_side_adapter.py:549](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:549). Existing snapshot tests use `write_videos=False` at [test_overfit100_context_modes.py:947](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:947). Fix: when completed at entry, suppress every filesystem write until immutable final comparison, and add video-enabled booby-trap/snapshot tests.

3. **BLOCKER — final publication is neither transactional nor race-safe** — [generate_wan_side_adapter.py:1541](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1541): any one final artifact is treated as completion, while publication writes `aggregation.json` first at [generate_wan_side_adapter.py:2453](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2453). Preemption after that write disables staging on retry; nondeterministic aux can then fail immutable comparison before missing summaries are repaired. The one-time completion probe also races another writer. Fix: publish a commit marker last, treat only a complete marker-backed set as completed, resume partial publication from staged rows, and use generation-match/create-if-absent or equivalent single-writer protection.

4. **MINOR — recursive enumeration still fails open on listing errors and does not establish real GCS-marker behavior** — [generate_wan_side_adapter.py:1794](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1794): `tf.io.gfile.walk` is called without `onerror`; TensorFlow documents that listing errors are ignored by default, while Cloud Storage represents some folders using zero-byte objects. Fix: rethrow through `_staging_error`, avoid relying on prefix `exists`, explicitly define tolerated zero-byte marker forms, and test mocked GCS file/marker/error classifications. [TensorFlow documentation](https://www.tensorflow.org/api_docs/python/tf/io/gfile/walk), [Cloud Storage object documentation](https://docs.cloud.google.com/storage/docs/objects).

5. **MINOR — aux recovery documentation promises a path that does not exist** — [generate_wan_side_adapter.py:2329](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2329): clearing staging cannot recover a ceiling after final publication because completed-mode ignores staging and immutable aggregation rejects changed aux. Fix: document the actual new-output/backfill procedure or implement and test a ceiling-only backfill path.

6. **MINOR — the strengthening record’s clean-check claim is false** — [overfit100_codex_code_eval-resume_review.md:198](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_codex_code_eval-resume_review.md:198): `git diff --check 78819dc 99ee724` reports a new blank line at EOF. Fix: remove it and rerun the recorded check.

Claim verification:

1. **Partial:** the 40-hex COMMIT gate, exact known-field typing/comparison, and hash re-derivation work; signature completeness and exact key-set enforcement do not.
2. **Pass with stated trust boundary:** coercions, booleans-as-ints, domains, descriptor identity, context source, and duplicates are rejected; a type-valid physically inconsistent metric tuple is still admissible because no universal SSIM/MSE invariant or stored prediction exists.
3. **Fail:** completed-artifact detection precedes staging, but completed video-enabled directories remain mutable and partial final publication is unsafe.
4. **Partial:** both aux tests genuinely pin the documented admission/parity semantics; the advertised recovery guidance is inaccurate.
5. **Partial:** recursive shape enforcement, offender aggregation, root naming, and escape guidance work; listing failures and real GCS markers remain uncovered.
6. **Not independently reproduced:** the recorded 1131/2 suite and 11/11 mutations could not run because this read-only sandbox has no writable temporary directory; both changed Python files compiled successfully, but `git diff --check` failed.
7. **Pass:** `overfit100_success_statistic.py` is untouched, aggregation schema is unchanged, and role validation remains before rollout.
8. **Accepted:** builder-level belt-and-braces typing is not required for this commit because admitted staging rows are strictly typed and fresh rows are constructed in-process; reusing the admission validator would still be a worthwhile follow-up.
9. **Pass:** `_SUCCESS`, finite SSIM stubbing, and distinct video-manifest fixture corrections are present and appropriate.

Risk read: `99ee724` is not fit to carry segment-final or full-set passes. Do not relaunch this SHA. When a corrected commit is approved and relaunch begins, cancel and confirm termination of all pre-resume queue jobs—do not leave them racing against the new evaluator.

Status

- Review complete; no subprocesses or background agents are running.
- ETA: complete. Earliest safe steer: now.
```

---

## Strengthening record (2026-07-31 — Claude Opus 5, Coder)

All six findings **accepted and fixed**, tests first (29 failures before implementation). Suite:
**1162 passed / 2 skipped** (from 1131+2; +31). `overfit100_success_statistic.py` remains untouched
and the aggregation schema is unchanged (R-H holds).

### 1 (BLOCKER) — signature completeness + exact key set

`OVERFIT100_RUN_SIGNATURE_TYPES` gains 14 fields, all derived from RESOLVED runtime objects where
they exist: `scheduler_sigma_min` / `scheduler_sigma_max` (read off the scheduler the rollout
actually uses, not the raw config), `resolved_checkpoint_dir` (via the new `_resolved_checkpoint_dir`
helper, now shared with `_restore_overfit100_validation_state` so the bound value is literally the
directory the restore reads), the latent geometry (`latent_channels/frames/height/width`),
`wan_max_sequence_length`, the transformer numerics (`attention`, `precision`,
`flash_min_seq_length`, `flash_block_sizes` as canonical JSON text, `split_head_dim`) and `fps`.
Admission now also requires the staged signature's **exact key set** (an unknown field is a hard
fail with the offending names) and **equality with the independently recomputed expected hash**.

**Tests:** `test_signature_binds_every_rollout_affecting_input`,
`test_signature_uses_the_resolved_checkpoint_dir_not_the_config_string`,
`test_every_new_signature_field_is_enforced` (14 parametrized cases, one per new field),
`test_an_extra_signature_key_is_refused`,
`test_signature_hash_equality_is_required_against_the_expected_signature`.

> **Honest note on the hash check.** With the exact key set enforced and every typed field compared
> by value, hash equality is currently unreachable as the *sole* failure — no input passes those and
> still hashes differently. It is kept because the review required it and because it keeps the
> guarantee whole-object rather than field-enumerated; its presence is asserted against the source
> as well as exercised behaviourally, so it cannot be deleted silently.

### 2 (BLOCKER) — a completed role directory now takes no writes at all

A `writes_suppressed` flag (set when the published marker is present at entry) gates the video
block, the per-window `metrics.json`, and the staging write. The previous snapshot tests all ran
`write_videos=False`, which is exactly why `_save_video`'s `overwrite=True` escaped.

**Tests:** `test_a_completed_dir_writes_nothing_even_with_videos_enabled` runs a **video-enabled**
pass, snapshots the whole role directory, then booby-traps both `_save_video` and `_write_json`
before the rerun; `test_a_completed_dir_suppresses_staging_writes_too` covers the staging half
(behaviourally plus a source assertion, because with resume already off in published mode the guard
is unreachable-by-construction — stated in the test rather than papered over).

### 3 (BLOCKER) — marker-last transactional publication

`_PUBLISHED` is written **after** `aggregation.json`, `summary.csv` and `summary.json`, and
completion is the **marker**, not any one artifact. `overfit100_publication_state` returns
`fresh` / `partial_publication` / `published`, probed once at entry before any staging interaction:

* **published** → recompute in memory, write nothing, let the immutable writers compare;
* **partial_publication** → PUBLICATION-RESUME: rebuild the rows from staging, let the immutable
  writers verify whatever already exists, write the missing artifacts, then the marker;
* **fresh** → the normal (possibly staging-resumed) pass.

**Aux semantics, as asked.** Staged rows are treated as **authoritative** for the repair. The
argument: publication only begins once the grid is complete, so staging holds exactly the rows that
produced the partial publication — aux included — and rebuilding from them is byte-consistent with
whatever was already written by construction; the immutable writers then *verify* that rather than
assume it. Recomputing instead would risk publishing a different auxiliary block over evidence that
is already partly published. The fail-closed complement: if staging cannot rebuild the full grid the
pass **refuses**, naming the directory and telling the operator to move it aside and re-run into a
clean role directory (`test_publication_resume_refuses_when_staging_cannot_rebuild_the_grid`).

**Single writer** is enforced **operationally** — at relaunch the previous jobs are cancelled and
confirmed dead — and that is stated in the module docstring beside the marker logic, because
`tf.io.gfile` offers no generation-precondition write. Marker-last plus completion-by-marker keeps
the race window small, and the cheap detectable case is pinned:
`test_a_foreign_writer_publishing_different_bytes_is_refused` makes a marker appear mid-run with
different content and shows the immutable writer refusing it.

**Tests:** `test_the_published_marker_is_written_last_and_defines_completion` (write-order captured),
`test_an_artifact_without_the_marker_is_not_completion`,
`test_a_mid_publication_preemption_is_repaired_from_staging` (crash after `aggregation.json`, then a
rerun that re-runs **zero** rollouts, leaves `aggregation.json` byte-identical, writes the missing
artifacts and the marker), plus the refusal and foreign-writer cases and
`test_publication_state_is_probed_before_any_staging_interaction`.

### 4 (MINOR) — enumeration fails closed

`tf.io.gfile.walk` now takes an `onerror` that rethrows through `_staging_error`; an absent root
raises a private `_StagingRootAbsent` sentinel instead (TF's `walk` cannot continue after a
swallowed listing error, so returning from `onerror` corrupts its internal state — found while
implementing). The prefix `exists()` probe is gone, because on Cloud Storage a prefix is not an
object. Tolerated directory markers are defined explicitly: an empty name, or a zero-byte object
whose name ends in `/`; any other stray object is an offender.

**Tests:** `test_a_listing_error_during_enumeration_is_fatal`,
`test_gcs_zero_byte_directory_markers_are_tolerated_but_other_zero_byte_objects_are_not` (mocked
GCS-shaped walk), `test_enumeration_does_not_require_the_prefix_to_exist`.

### 5 (MINOR) — aux-recovery guidance corrected

The parity note no longer says an operator "clears the staging root and re-runs" to recover a
ceiling — impossible after publication, since completed-mode ignores staging and the immutable
aggregation refuses a changed aux block. It now states the real procedure: the VAE ceiling is
checkpoint- and rollout-independent, so it is recovered by the separate **backfill artifact**
(decode the stored `z_video`, score against the source frames, write to its OWN output), never by
mutating a published role directory. The staging error text likewise no longer implies recovery,
only that disabling resume recomputes rollouts without altering published artifacts.

**Tests:** `test_staging_error_text_does_not_promise_ceiling_recovery_by_clearing_staging`,
`test_the_aux_recovery_note_points_at_the_backfill_not_at_staging`.

### 6 (MINOR) — the false clean-check claim

The trailing blank line at EOF of `overfit100_codex_code_eval-resume_review.md` (introduced by my
own append in the previous round) is removed. **Re-run and recorded honestly:**
`git diff --check 78819dc 99ee724` did report `:198: new blank line at EOF`; after the fix
`git diff --check` is clean for this commit's staged tree. The previous record's claim was wrong
because I ran `git diff --check` on the working tree *before* staging the doc, so the appended file
was untracked and unchecked.

### Verification

* **Red evidence:** 29 failed / 136 passed before implementation.
* **Mutation spot-checks, 10/10 caught** (each restored byte-identically): marker write dropped → 7;
  completion probe back to any-one-artifact → 3; publication-resume repairing an incomplete grid → 1;
  video writes unsuppressed → 1; staging-write suppression removed → 1; `scheduler_sigma_min`
  unbound → 88; exact key set unenforced → 1; expected-hash equality dropped → 1; `walk` without
  `onerror` → 2; GCS trailing-slash markers rejected → 1. Two mutants initially survived
  (staging-write suppression, expected-hash equality) because both guards are unreachable-by-
  construction today; both tests were made honest — behavioural assertion plus a source guard —
  rather than left claiming more than they check.
* **Static:** black (`--line-length 119 --target-version py312`), ruff, `py_compile`,
  `git diff --check` clean on the staged tree. Statistic module and aggregation schema untouched.
