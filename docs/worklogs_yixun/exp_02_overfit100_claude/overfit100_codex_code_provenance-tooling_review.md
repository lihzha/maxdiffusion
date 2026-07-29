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

*(Strengthening record to be appended at the close of this round — per SOP, the round is not finished until every finding's resolution is recorded below.)*

**Planner note for cycle D (from the verdict paragraph):** the shuffled-text control must derange instruction **values**, not episode indices — with 6 duplicate-instruction groups (22 episodes), an index derangement could map an episode to a different episode carrying the *same text*, silently weakening the control. Carried into cycle D's requirements.
