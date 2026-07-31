# Codex code review — eval-ffmpeg round (cycle D fix 2)

- **Date:** 2026-07-31
- **Commit under review:** `9c26070` (ffmpeg-ensure in eval launcher + aux-degradation warning)
- **Reviewer:** Codex `gpt-5.6-sol`, reasoning effort xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION (1 MAJOR + 2 MINOR — all three are test-strength findings; production code verified on all focus points F-A/F-B/F-C(qualified)/F-E/F-F)
- **F-C qualification (recorded, no code change):** an ffmpeg-install FATAL exits 1, which the queue treats as a non-retryable application error — recovery is manual/approved resubmission, not auto-retry. Acceptable: fail-loud is the intent.

## Reviewer output (verbatim)

```
I’ll inspect the SOP, the exact commit diff, and only the surrounding launcher/module/tests needed for F-A through F-G. I’ll keep the review read-only and won’t reopen prior settled issues unless this commit regresses them.
exec
/bin/zsh -lc 'git status --short --branch && git show --stat --oneline --decorate --no-renames 9c26070 && git show --format=fuller --no-ext-diff --no-renames 9c26070 -- docs/worklogs_yixun/experiment_SOP.md bash_scripts/validate_wan_overfit100.sh bash_scripts/eval_wan_overfit100_val_loss.sh src/maxdiffusion/generate_wan_side_adapter.py tests' in /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
 succeeded in 0ms:
## claude-exp_02_overfit100-20260728
9c26070 (HEAD -> claude-exp_02_overfit100-20260728) exp_02(overfit100): cycle D fix 2 — ffmpeg-ensure in eval launcher + loud aux-degradation warning (S3 intermediate finding)
 bash_scripts/eval_wan_overfit100_val_loss.sh       |   5 +
 bash_scripts/validate_wan_overfit100.sh            |  46 ++++++++
 src/maxdiffusion/generate_wan_side_adapter.py      |  37 ++++++
 .../test_overfit100_eval_contracts.py              |  65 +++++++++++
 .../test_overfit100_eval_selection.py              | 126 +++++++++++++++++++++
 5 files changed, 279 insertions(+)
commit 9c2607072a96d1056bf0e7aa52fa6a659d2c35b8
Author:     Yixun-Hu <yh4742@princeton.edu>
AuthorDate: Fri Jul 31 17:48:06 2026 -0400
Commit:     Yixun-Hu <yh4742@princeton.edu>
CommitDate: Fri Jul 31 17:48:06 2026 -0400

    exp_02(overfit100): cycle D fix 2 — ffmpeg-ensure in eval launcher + loud aux-degradation warning (S3 intermediate finding)
    
    The S3 intermediate evals ran clean on primary metrics, and the str-vs-Path fix worked, but every
    auxiliary row carried "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'": the
    bounded apt ffmpeg-ensure block added to bash_scripts/build_overfit100_dataset.sh in the
    ffmpeg-ensure round was never added to the eval launcher, and the TPU worker image ships without
    ffmpeg.
    
    * validate_wan_overfit100.sh now carries that block VERBATIM (inserted programmatically from the
      build script), placed after venv activation and before the HF prefetch, so a worker without
      ffmpeg fails in ~30 apt-seconds instead of completing a full eval with no VAE ceilings. Its log
      lines stay tagged [build] because byte-equality between the two launchers is asserted by a test
      -- the blocks can never drift.
    * eval_wan_overfit100_val_loss.sh deliberately does NOT get the block: that arm reads TFRecords,
      runs the transformer in latent space, frees the VAE right after state construction, and writes
      json/csv/png. The omission is documented in the script and pinned by a test asserting no decoder
      call (decode_mp4_frames / overfit100_aux_rgb / _save_video / export_to_video / ffprobe) is
      reachable from the module.
    * generate_wan_side_adapter gains `aux_prerequisite_warning`: when aux is requested and ffmpeg or
      gsutil is absent, the driver logs ONE loud line before any rollout naming the missing binary and
      stating that all aux metrics will be null. D5's contract is unchanged -- aux still never fails
      the run and every row still records its own aux_status -- but the degradation is now visible in
      the job log instead of only in the artifact.
    
    Tests: the ffmpeg block is extracted from the shipped scripts and EXECUTED under bash with a PATH
    shim (no-op when the tools exist, installs when they do not, FATAL when the install does not
    provide them), parametrized over both launchers; placement bracketing venv -> prefetch -> python;
    the loss-arm omission; and the warning's content, silence conditions and driver wiring. Mutation
    spot-checks 4/4 caught (block removed, block drifted, driver warning dropped, warning silenced).
    
    Suite: 1021 passed / 2 skipped.
    
    Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>

diff --git a/bash_scripts/eval_wan_overfit100_val_loss.sh b/bash_scripts/eval_wan_overfit100_val_loss.sh
index 5bf861a..033b137 100755
--- a/bash_scripts/eval_wan_overfit100_val_loss.sh
+++ b/bash_scripts/eval_wan_overfit100_val_loss.sh
@@ -69,6 +69,11 @@ export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_async_collective_f
 # exp_02 set knobs (EXPECTED_WINDOWS / NUM_TEXT_SLOTS) travel with it; (4) EXPECTED_COUNT is the
 # built window count of the set being evaluated (train100 -> 1629, train10 -> 167).
 #
+# NO ffmpeg-ensure block here, deliberately: this arm never decodes video. It reads TFRecords,
+# runs the transformer in latent space, frees the VAE right after state construction, and writes
+# json/csv/png -- so unlike validate_wan_overfit100.sh it has no ffmpeg/ffprobe dependency to
+# install. A test pins that claim against the module (no decoder call is reachable from it).
+#
 # SMOKE gate: SMOKE_LIMIT=<N> evaluates only the first N batches of only the first checkpoint
 # into an isolated validation_loss_smoke/ directory (the n==expected assertion is skipped).
 
diff --git a/bash_scripts/validate_wan_overfit100.sh b/bash_scripts/validate_wan_overfit100.sh
index 7ded88a..da25d8c 100755
--- a/bash_scripts/validate_wan_overfit100.sh
+++ b/bash_scripts/validate_wan_overfit100.sh
@@ -141,6 +141,52 @@ export COMMIT
 echo "COMMIT=${COMMIT}"
 git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"
 
+# The auxiliary RGB / VAE-ceiling metrics decode the source MP4 with ffmpeg, and the TPU worker
+# image ships without it: the S3 intermediate evals completed with EVERY aux row carrying
+# "FileNotFoundError: ... 'ffmpeg'" (contained by design, but no VAE ceilings). The block below
+# is copied VERBATIM from build_overfit100_dataset.sh -- byte-equality is asserted by a test, so
+# the two launchers can never drift -- which is why its log lines are tagged [build].
+# >>> ffmpeg ensure
+# Probe attempt 2 (job 20260729-172443-23bcb17a) loaded the pinned VAE and passed all three V1
+# windows, then died in the V3 precheck with `FileNotFoundError: 'ffmpeg'`: the TPU worker image
+# has no ffmpeg. Install it HERE -- before the multi-minute HF prefetch and the JAX init -- so
+# the failure mode is a 30-second apt error rather than 20 minutes of wasted TPU time.
+#
+# apt options mirror setup.sh's ephemeral-worker hardening: a single bounded wall-clock budget
+# (`apt_deadline_run`, never -1), `-o DPkg::Lock::Timeout=60` per invocation, and a LOUD exit on
+# any failure. setup.sh has already stopped/disabled the apt-daily timers under
+# EPHEMERAL_WORKER=1, so contention is expected to be gone by the time this runs.
+if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
+  echo "[build] ffmpeg/ffprobe not on PATH; installing (the TPU worker image ships without them)"
+  FFMPEG_APT_BUDGET="${FFMPEG_APT_BUDGET:-420}"
+  APT_SECTION_START=$SECONDS
+  APT_SUDO=""
+  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then APT_SUDO="sudo"; fi
+  apt_deadline_run() {
+    rem=$((FFMPEG_APT_BUDGET - (SECONDS - APT_SECTION_START)))
+    if [ "$rem" -le 0 ]; then
```

---

## Strengthening record (2026-07-31 — Claude Opus 5, Coder)

All three findings **accepted and fixed**, test-first where a red state was meaningful. No
production code changed: every finding was test strength, and the reviewer had verified the
shipped behavior on F-A/F-B/F-C/F-E/F-F. Suite: **1028 passed / 2 skipped** (from 1021+2; **+7**).

### Finding 1 (MAJOR) — the no-decoder claim was only a source-token scan

The reviewer's preferred fix was feasible, so the fallback (weakening the in-script claim) was not
used. A probe first confirmed the REAL `evaluate()` loop drives on one CPU device with a
`jax.tree.map`-built sharding tree, so no production seam was needed.

**New:** `test_overfit100_loss_evaluation_completes_with_every_codec_entry_point_booby_trapped`
(`test_overfit100_eval_loss.py`) runs the real OVERFIT100 loss evaluation — record load, state
build, per-checkpoint restore, jitted batches, aggregation, artifact writes — with

* `gen._save_video`, `gen.overfit100_aux_rgb`, `gen/utils.export_to_video`,
  `builder.decode_mp4_frames` and `builder.fetch_pinned` booby-trapped to raise, **and**
* `subprocess.run` / `subprocess.Popen` wrapped to raise on any `ffmpeg`/`ffprobe` argv (non-codec
  spawns such as `git rev-parse` pass through) — the inline-argv case a token scan cannot see, and
* `ffmpeg`/`ffprobe` stripped from `PATH` and from `shutil.which`.

It asserts zero trap hits, one aggregate row with `n == 4`, all four artifacts written
(`val_loss.{json,csv}` + `val_loss_per_window.{json,csv}`) with the four expected window names and
finite losses, and that the VAE was released before the loop.

**Non-vacuity:** `test_the_codec_traps_are_not_vacuous` proves the traps fire for a direct decoder
call and for both inline `["ffmpeg", …]` and `["/usr/local/bin/ffprobe", …]` argvs, while
`["echo", …]` still passes through. The source scan is retained as supplementary evidence
(`test_loss_eval_source_scan_remains_as_supplementary_evidence`).

**Red/mutation:** two mutants of the production module, each invisible to the old scan, are caught
by the new test — an inline `subprocess.run(["ffmpeg", "-version"])` inside `evaluate()` (1 failure)
and a transitive `gen.overfit100_aux_rgb(...)` call (2 failures).

### Finding 2 (MINOR) — the "warning logged once before restore" test asserted nothing

**New:** `test_driver_logs_exactly_one_aux_warning_before_the_restore` and
`test_driver_is_silent_when_there_is_nothing_to_warn_about[tools present|aux disabled]`
(`test_overfit100_context_modes.py`) drive the real `run_overfit100` through the existing synthetic
driver fixture, recording ordered `("log", msg)` / `("restore",)` events. They assert **exactly
one** warning, that its index precedes the restore, that only the missing binary is named (`ffmpeg`
present, `gsutil` absent from the text), and silence in both no-warning cases — with the run still
reaching the restore.

The old contracts-file test was rewritten as
`test_missing_ffmpeg_is_contained_per_row_and_wired_into_the_driver`: it keeps the D5 containment
check (status recorded, never raised) and the wiring guard, and no longer patches a logger it never
inspects. It points at the behavioral test by name.

**Mutation:** deleting the production `max_logging.log(aux_warning)` call now fails (1 failure) —
it passed the old test — and logging it twice also fails (1 failure).

### Finding 3 (MINOR) — the bash harness appended `/usr/bin:/bin` to the fake PATH

`_run_ffmpeg_block` now builds its child environment through `_hermetic_env(binroot, tmp_path)`,
which is **only** the shim directory, and invokes bash by absolute path. Shims were added for every
external command the block actually runs — `id` (the `id -u` sudo branch), `head` (from
`ffmpeg -version | head -1`), `sudo`, `timeout`, `apt-get` — with `chmod` resolved absolutely inside
the `apt-get` stub.

**New:** `test_the_ffmpeg_block_harness_is_hermetic` puts a decoy `ffmpeg` on the PARENT's PATH,
asserts the parent can see it, then asserts the harness's OWN returned env has `PATH == fakebin`,
that the block still took the install branch, and that the decoy never appeared in the output.

**Mutation:** restoring `/usr/bin:/bin` in `_hermetic_env` fails (1 failure); dropping `env=` so the
child inherits the parent environment fails (7 failures). The first mutation of this finding
initially survived because the self-test built its own env — that gap was closed before this record
was written.

### F-C qualification — accepted, no change

An ffmpeg-install FATAL exits 1 and the queue treats that as a non-retryable application error, so
recovery is manual/approved resubmission rather than auto-retry. That is the intended fail-loud
behavior (the alternative — proceeding without ffmpeg — is exactly the silent-degradation mode this
round exists to remove), and it is already recorded in this file's header.

### Verification

* Full suite **1028 passed / 2 skipped**; black (`--line-length 119 --target-version py312`), ruff,
  `py_compile`, `git diff --check` clean on the four touched test files.
* **Mutation spot-checks 6/6 caught**, each restored byte-identically: F1 inline-subprocess codec,
  F1b transitive codec, F2 warning deleted, F2b warning duplicated, F3 non-hermetic PATH, F3b
  inherited parent env.
* No production file was modified in this round; `bash_scripts/*` and
  `src/maxdiffusion/generate_wan_side_adapter.py` are byte-identical to `9c26070`.

