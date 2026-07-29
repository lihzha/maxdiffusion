# Code review: exp_02 overfit100 — cycle A (provenance-tooling)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-28

## Context loaded
- Experiment SOP — TDD, focused cross-model review, provenance, and validation contracts.
- Approved plan v4 — D5 manifest-root and H1 fixture-materialization requirements for cycle A.
- exp_02 queries and worklog — user decisions, empirical probes, approval, and Coder evidence.
- Four prior plan reviews — F1–F7, G1–G5, H1–H2, and I1 continuity.
- exp_01 analysis and results — prior cache semantics and the motivation for a finite-set memorization test.
- Cycle-A code, 67 tests, command/build log, fixture fingerprint, and the real 100-episode manifest.

## Verdict
REQUEST-REVISION. The pure selection/window logic and fixture extraction are strong, and the real manifest has the correct draw prefix, non-empty instruction picks, window arithmetic, totals, and geometry. However, its recorded commit cannot reproduce the uncommitted builder, and the IO/preflight paths can silently bind decisions to the wrong bytes or accept malformed provenance; regenerate from a clean code commit after strengthening. Cycle D should also derange instruction values—not merely episode indices—so duplicate taxonomy labels cannot remain unchanged in the shuffled-text control.

## Findings

1. **A1 — BLOCKER — The real manifest’s builder provenance is not reproducible.** The artifact records `b88bac1…`, but that commit contains none of the five untracked cycle-A source/test files; `repo_commit()` simply stamps the current HEAD without rejecting a dirty implementation. The required `ffmpeg` version is also absent (`docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json:3`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:392`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:403`, `docs/worklogs_yixun/exp_02_overfit100_claude/plan_overfit100.md:36`). **Concrete change:** commit the strengthened extractor, builder, and tests first; make production manifest generation reject relevant uncommitted changes; record ffmpeg plus the selection-library versions; then rebuild and reverify the manifest so `builder_commit` names the actual implementation.

2. **A2 — MAJOR — Selection content is not cryptographically bound to the recorded source generation.** Annotation bytes are downloaded and parsed, deleted, and only later statted; videos are statted before a separate unpinned download. An overwrite between those operations can therefore leave chosen text/success or ffprobe geometry from one generation paired with another generation’s fingerprint, while the subsequent live-stat verification passes (`src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:175`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:198`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:201`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:216`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:286`). The annotation’s embedded `episode_id` is also never checked against the requested ID (`src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:241`). **Concrete change:** stat first and download the exact generation, or hash/size-check every downloaded annotation and MP4 against its recorded stat before using it; assert embedded `episode_id == candidate_id`; abort on any mismatch and add race/mismatch tests.

3. **A3 — MAJOR — Block prefetch is output-equivalent only on an error-free fake and can silently change selection under real IO failures.** A mixed `gsutil stat` failure is ignored whenever any output parsed; annotation failures are declared benign by substring matching; invalid JSON and every download/ffprobe failure become absence, and a stat-success/probe-failure becomes `missing_video` (`src/maxdiffusion/data_preprocessing/extract_v1_fixture.py:121`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:181`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:193`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:220`, `src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:303`). The invariance test compares only final JSON and does not assert fetched/statted candidates or error behavior, even though the final block prefetches beyond the stopping acceptance (`src/maxdiffusion/tests/worklogs_yixun/test_overfit100_manifest.py:213`). **Concrete change:** distinguish confirmed per-object absence from transient/tool/source errors; retry unresolved batch members individually and abort on unclassified errors; defer errors for prefetched candidates until they are actually consumed so post-stop failures cannot affect the walk; test failures before and after the nth acceptance.

4. **A4 — MAJOR — The advertised preflight fails open on malformed manifest and fixture structure.** `verify_manifest()` silently skips absent fixture or episode fingerprints and checks only live generation/md5/size; it does not validate accepted-ID correspondence, unique contiguous indices, draw/tally/totals reconciliation, non-empty chosen index, expected URIs, or `n_windows` against counted frames (`src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py:348`). Likewise, `verify_fixture()` checks only that fingerprint-listed names exist, not exact ordered-name equality or the stored `z_i0 == z_video[:, :1]` contract (`src/maxdiffusion/data_preprocessing/extract_v1_fixture.py:265`). Existing verifier tests mutate remote-stat values but do not exercise these fail-open cases (`src/maxdiffusion/tests/worklogs_yixun/test_overfit100_manifest.py:284`, `src/maxdiffusion/tests/worklogs_yixun/test_overfit100_fixture.py:227`). **Concrete change:** add a fail-closed structural validator invoked before remote stats and before cycle-B encoding, plus mutation tests for every invariant; require the exact three fixture names, shapes/dtypes for every window, and bitwise first-frame equality.

## Seam judgments

a. **CHANGE** — block sizes preserve successful JSON output, but they perform extra tail reads and do not preserve failure behavior; strengthen IO isolation and tests.

b. **OK** — shared gsutil primitives in the extractor are dependency-safe, and lazy TensorFlow import keeps the manifest path lightweight.

c. **OK** — the LOC overage is justified by shared IO, CLIs, validation, and documentation; avoid expanding this module into the cycle-B IO layer.

d. **OK** — `provisional`, fingerprint `uri`, and accepted-inclusive tally are coherent and reconciled by tests.

e. **OK** — counted `nb_read_frames` is the correct quantity for materializable windows.

f. **CHANGE** — stat-success/probe-failure must abort or receive an explicit source-error classification, not masquerade as `missing_video`.

g. **CHANGE** — replace substring-based benign detection with exact per-object classification and retry/abort semantics.

h. **OK** — provisional acceptance is confined to dry-run, marked explicitly, and the CLI suppresses artifact writes.

i. **CHANGE** — add fail-closed manifest and fixture internal-consistency validation before cycle B consumes either artifact.

---

## Strengthening record (Coder: Claude Opus 5, 2026-07-29)

All four findings **FIXED**; all four CHANGE seams (a, f, g, i) **FIXED**. Nothing rejected.
Commits: A1 `7166436`, A2 `9a24518`, artifacts in the following commit. Suite: **399 passed,
2 skipped** (146 cycle-A tests, up from 67); `black -l 119` clean, `ruff check` clean.

**A1 — BLOCKER — builder provenance not reproducible. FIXED.**
`assert_implementation_committed()` cross-checks `git ls-tree HEAD` and `git status --porcelain`
for all five implementation paths and raises `DirtyImplementationError` before any network call,
so a production manifest can only be built from a commit that actually contains the builder
(`--dry-run` is exempt — it writes nothing — and stamps `builder_commit: "dry-run"`, which the
structural gate rejects as a non-SHA). `collect_tool_versions()` now also records **ffmpeg**
plus the two libraries that determine the selection itself: **numpy** (`candidate_order`) and
**jax** (`pick_instruction_index`). Red evidence: run against the exact pre-commit tree, the
builder refused, listed all five files, and wrote nothing. The manifest was then rebuilt from
the clean commit and carries `builder_commit 9a24518cbe82f35386e607169410f751fb1b7af7`.

**A2 — MAJOR — content not bound to the recorded generation. FIXED.**
Every object is now **statted first**, then downloaded at that exact generation
(`gsutil cp gs://…#<generation>`; verified to work for both batched and single-file copies),
then re-hashed: `verify_payload_binding()` compares base64-md5 **and** size against the stat
before the bytes are parsed or probed, and any mismatch aborts with `SourceError`.
`verify_annotation_binding()` additionally requires the annotation's embedded `episode_id` to
equal the drawn candidate (plus `success` present, `texts` a list). Tests: md5-mismatch abort,
episode-id-mismatch abort, size mismatch, and `pinned_uri` refusing a fingerprint with no
generation. This closes the window in which a decision could be made from one generation's
bytes while a different generation's fingerprint was recorded.

**A3 — MAJOR — absence conflated with failure; invariance only on an error-free fake. FIXED.**
Added an explicit per-object outcome type (`Resolved`: `found` / `absent` / `error`).
`classify_stat_batch()` derives absence **only** from gsutil's exact `No URLs matched: <uri>`
lines — the substring heuristic (`"404"`, `"NotFoundException"`) is gone, and a transient error
whose text merely contains `404` is now classified as an error (regression test included).
Unresolved batch members are retried **once individually** before being reported. Absence is
established solely by the stat, so a stat-present object that fails to download, fails its md5,
or yields invalid JSON is an **error, never a rejection reason**; likewise stat-success /
probe-failure now aborts (seam f) instead of masquerading as `missing_video`. Errors are
**deferred** — they raise only when the walk *consumes* that candidate — so a failure on a
candidate prefetched past the stopping acceptance cannot change the result; tested both ways
(error before the nth acceptance aborts, error after it is invisible). MP4 downloads also moved
inside the ordered walk, so no video is fetched for a candidate that is never consumed; only
cheap metadata (stats) is prefetched. The six recorded rejection reasons are unchanged and a
test pins that vocabulary.

**A4 — MAJOR — verifiers fail open. FIXED.**
`validate_manifest_structure()` is a new pure, fail-closed gate covering: exact required key
sets (top level / episode / fingerprint / ffprobe), `provisional is False`, 40-hex
`builder_commit`, all six tool versions non-empty, fixture block complete with the exact three
window names, contiguous `episode_index` 0..n-1, unique `episode_id`, `chosen_text_index`
landing on a **non-empty** text with `chosen_text_raw`/`used_text` consistent, expected
annotation/MP4 URI patterns, pinned corpus geometry (320x192 @ 5 fps yuv420p),
`nb_frames >= 33`, `n_windows == 1 + (nb_frames-33)//4`, the draw-log reason vocabulary, no
duplicate draws, a draw log ending on the accepting draw, accepted ids == episodes in order,
tally reconciliation, and both totals. It runs **before any remote stat** in `verify_manifest`
(asserted: a structurally broken manifest produces zero stat calls) and again on the freshly
built manifest before it is written. 33 mutation tests, one per invariant, each required to be
caught. `verify_fixture()` gained `validate_fixture_structure()`: exact ordered name set,
shapes/dtypes for **every** array of every window, and the `z_i0 == z_video[:, :1]` bitwise
contract — mutation-tested, including a structurally broken but correctly-hashed fixture.
`verify_manifest` now also distinguishes a stat **error** from an **absent** object instead of
reporting both as missing.

**Seam judgments.** a / f / g / i fixed as described above. b, c, d, e, h were judged OK and are
unchanged; per (c) the shared IO layer was **not** expanded — cycle B should own its own.

**Additional defect found during strengthening (not in the review).** Ruff F811 flagged that a
`Resolved.error` *classmethod* shadows the `error` *field*'s default, leaving every
`absent`/`found` outcome with a truthy bound method in `.error` (confirmed at runtime before the
fix). The constructor was renamed `Resolved.failed()` and a regression test asserts
`Resolved.absent().error is None`. Had this shipped, every "is this an error?" check on a
non-error outcome would have read as true.

**Rebuild result (review step 3).** Manifest regenerated from the clean commit `9a24518`:
100 episodes / 1,629 windows / 129 draws, tally `{accepted: 100, not_success: 23, too_short: 6}`.
Diffed against the superseded artifact: `episodes`, `draw_log`, `rejection_tally`, `totals`,
`fixture`, `selection_seed` and `provisional` are **all identical**; only `builder_commit`,
`created_utc` and the (now larger) `tool_versions` differ, with no unexpected key differences.
That identity is independent evidence both that no source drift occurred between the two builds
and that the strengthened stat-first / pinned-download / md5-bound IO path reaches exactly the
same decisions. The published fixture was re-verified from GCS with the strengthened
`verify_fixture()` (clean) before the rebuild; the rebuilt manifest passed the offline
structural gate (`expected_episodes=100`) and a live `verify_manifest()` over all 201 objects
(clean, 52.7 s).

**Planner note for cycle D (from the verdict paragraph):** the shuffled-text control must derange instruction **values**, not episode indices — with 6 duplicate-instruction groups (22 episodes), an index derangement could map an episode to a different episode carrying the *same text*, silently weakening the control. Carried into cycle D's requirements.
