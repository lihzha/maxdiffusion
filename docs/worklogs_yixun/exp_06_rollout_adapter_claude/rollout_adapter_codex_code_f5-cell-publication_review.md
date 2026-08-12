# exp_06 F5 `cell-publication` — Codex code review

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-12. Verdict: **REQUEST-REVISION — 2 BLOCKER + 1 MAJOR.** B1: cell artifacts prove consistency, not measurement legality (digest = tamper check, not legality check — the campaign's own pattern turned on this round; the committed battery probe scored its forged-cell attack REFUSED while the forgery was in fact ADOPTED). B2: code_sha reads git HEAD only — does not identify running bytes (dirty tree / caller-supplied COMMIT). MAJOR: two concurrent publishers can permanently tear the content/sidecar pair (content-B + digest-A) and the cell becomes forever unadoptable. Hunt items C/D/F/G all passed, incl. the ruling that RUN_NAME pinning across resubmissions is sound once artifacts are authenticated.

# REQUEST-REVISION

## BLOCKER — Cell artifacts prove consistency, not measurement legality

[load_cell_artifact](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1391) recomputes the unkeyed SHA-256 correctly, but every hashed value—including `code_sha`, context, run identity, peak, and peak source—is supplied by the artifact writer. [_adoption_refusal](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1451) compares those public claims to equally public local values; nothing establishes that this measurement was produced by M1.

A writer can copy the current context/run name, set both trials to `peak_bytes=1` with an authorizing peak source, recompute the payload digest and sidecar, and adoption at [run_fit_probe:2088](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2088) skips the measurer. Aggregation then authorizes the fabricated cell.

The committed attack nearly demonstrates this itself: [reviewer_attacks.py:2187](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:2187) rewrites a trial, recomputes both digests, observes it being adopted, and labels the probe “REFUSED” merely because the final-table digest changed. That verifies propagation, not legality. The reported 86/0 therefore does not cover hunt A.

Require an authenticated publication authority—such as a workload-identity/KMS signature or an enforced trusted, immutable GCS writer boundary—and add an attack where rehashed favorable peak data must cause remeasurement.

## BLOCKER — `code_sha` does not identify the running bytes

[derive_code_sha](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:427) reads only `git rev-parse HEAD`; the launcher does the same at [train_wan_pos_rollout.sh:88](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:88). Neither refuses a dirty checkout, and a tarball without Git trusts the caller-supplied `COMMIT`.

The present F5 code is concrete evidence: its running bytes are uncommitted while its derived SHA remains `a3ba5c0…`. A dirty measurement-code change can therefore adopt cells measured before that change. Enforce a clean checkout before packaging and verify a bundle/content manifest on the worker, or bind context to such a manifest rather than HEAD alone.

## MAJOR — Concurrent publishers can permanently tear the content/sidecar pair

[publish_cell](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1359) performs a racy existence check, then independently overwrites content and sidecar at [line 1372](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1372). Two writers can finish as `content-B + digest-A`. Production artifacts are not byte-identical because measured timings vary.

GCS makes each individual object upload/overwrite atomic, so readers should not see a partial JSON object, but it does not make this two-object transaction atomic. [Google recommends generation preconditions for race prevention](https://docs.cloud.google.com/storage/docs/consistency). Worse, a later publication sees both paths and returns without verifying or repairing the pair, leaving that cell path permanently unadoptable.

Use immutable content-addressed objects plus one conditional commit marker, or generation-match/create-if-absent semantics. Add an interleaving test with distinct payloads.

Other hunt items pass:

- M2 imports the current loader and rejects any protocol other than v3; genuine v2 tables cannot start training.
- The resumed-vs-fresh test compares canonical full tables after removing provenance and its necessarily changed top-level digest.
- Every adoption enters through the sidecar-requiring loader; the final table uses those verified trials.
- Adoption bypasses the measurer/program builder; F3/F4 tracing tests remain on the real builder.
- With authenticated artifacts, pinning `RUN_NAME` across resubmissions is sound: code, model, device, geometry, and recipe still have to match.

Read-only verification: `bash -n`, `git diff --check`, and AST parsing passed. Pytest is unavailable in this reviewer environment; I relied on the supplied 2201/0 result, but it cannot close the source-demonstrable blockers above.

---

# Strengthening record — round F5b (Coder, 2026-08-12)

Every finding resolved below; none rejected. Full narrative in `rollout_adapter_worklog.md`
(2026-08-12T03:40Z). Scope for BLOCKER 1 was set by a Planner ruling: bind identity to the running
bytes, and **declare** the authentication boundary rather than fake it.

## BLOCKER 1 — cell artifacts prove consistency, not legality — **FIXED, with a declared residual**

- **The probe that covered it up is corrected first.** `reviewer_attacks.py:2187` (`F5-5 smuggle past
  the digest`) watched the run-level digest move and called that a refusal while the forgery was
  adopted. Rewritten as `F5-5 fabricate a cheap cell`, asserting **remeasurement**. On the unfixed
  code it reported `SUCCEEDED: a fabricated cheap cell was adopted without being measured`, with the
  fabricated 1-byte peak reaching a `11.77h at 0.0% of capacity` projection. That is this round's red.
- **Fixed** by binding `deployed_manifest_digest()` — sha256 over the running `.py` bytes of the
  deployed package — into `ProbeContext`, which adoption already compares byte-for-byte. A forger who
  is not running the deployed bytes is refused, and the refusal names `manifest_digest`.
- **Declared, not faked:** these artifacts are integrity-checked and program-bound, **not
  authenticated**. A writer holding the deployed source tree *and* bucket write access can reproduce
  the manifest and fabricate a measurement. The trust anchor is the **bucket ACL**, the same anchor
  the final authorization table and every published artifact in this campaign already rest on. KMS /
  workload-identity signing is infrastructure and is **escalated to Yixun as a policy decision in the
  M1 pre-launch package**. Stated in the module docstring, in `publish_cell`, in
  `adopt_published_cell`, in the corrected probe, and in the worklog; kept honest by
  `test_the_module_declares_what_adoption_does_not_prove`.
- Tests: `test_a_cell_measured_by_other_running_bytes_is_re_measured`,
  `test_the_context_carries_the_manifest_and_adoption_compares_it`,
  `test_editing_a_deployed_module_moves_the_manifest`. Probe: `F5-5`.

## BLOCKER 2 — `code_sha` does not identify the running bytes — **FIXED**

- The manifest above is now the identity; `code_sha` survives as the label the launcher's prerequisite
  compares. Two honesty rules added to `derive_code_sha`: a process that **declares** a commit
  (`COMMIT` set) from a tree with uncommitted *measurement* code is refused loudly (scoped to the
  manifest's files, so a dirty test file does not block red-first work); a **git-less deployment must
  bind a manifest**, because `COMMIT` alone is an environment variable anybody can set.
- The reviewer's own evidence — F5's running bytes uncommitted under SHA `a3ba5c0` — is precisely what
  the first rule now refuses.
- Tests: `test_a_process_declaring_a_commit_refuses_to_publish_from_a_dirty_tree`,
  `test_a_deployment_without_git_is_identified_by_content_not_by_a_declared_commit`, and
  `test_the_code_sha_is_derived_and_a_disagreement_is_fatal` updated (dirty state pinned by
  monkeypatch so it reads the same before and after a ceremony commit).

## MAJOR — concurrent publishers can permanently tear the pair — **FIXED**

- Content objects are **named by their own digest** (`cells/<cell>.<digest12>.json`); the marker
  (`cells/<cell>.json.digest`) is the single mutable name holding one digest. Two writers write two
  objects; the marker commits one whole object; last-writer-wins on the marker only, never a mixed
  pair. `publish_cell` additionally **verifies and repairs** a marker naming a missing/mismatched
  object instead of returning early — the window the earlier orphan fix did not close.
- Tests, red-first, with **distinct payloads** (25.347 s / 25.356 s):
  `test_two_interleaved_publishers_never_leave_a_torn_pair`,
  `test_a_marker_pointing_at_a_missing_content_object_is_repaired_not_returned_from`,
  `test_a_marker_naming_content_that_does_not_hash_to_it_is_refused`, and
  `test_the_retired_two_object_scheme_really_did_tear`, which re-derives the retired scheme in-test
  and drives the same interleaving through it to `content-B + digest-A`. Probe: `F5-7`.

## Verification

Canonical suite **2213 passed / 0 failed**; battery **87 probes, 87 REFUSED, 0 SUCCEEDED**
(`harness/attacks_f5b_20260812.log`, sha256 `ac9b419ba26e75f8…`); `black` / `ruff` /
`git diff --check` clean. Manifest cost measured at **35 ms** over 300 files / 4.16 MB, cached per
process. Nothing committed.

**Note on format change:** F5-era artifacts (fixed-name content + sidecar) are not readable by the
F5b loader and fail closed to a re-measure. No such artifact exists — every attempt root's
`fit_probe/` is empty — so the change costs nothing.


---

# F5b re-review (appended 2026-08-12)

Verdict: **REQUEST-REVISION — 2 BLOCKER + 1 MINOR.** B1: F5-5 still a false refusal one level up — the probe tests a FOREIGN-manifest artifact, but the real attack copies the current context (the manifest digest is PUBLIC in the payload), swaps 1-byte peaks, rehashes, and is ADOPTED (reviewer executed it through the real functions; measurer skipped). Prescription accepted per Planner ruling: an explicit DECLARED verdict class for the in-boundary forgery + honest residual wording (any bucket overwriter who can READ one artifact can fabricate — no source tree needed). B2 (new scope): trainer resume selection matches code_sha only, dropping context_digest — executed: same-SHA publications with different contexts cross-resume. MINOR: content names truncate the digest to 48 bits. Verified passes: manifest determinism/coverage (300 files), dirty-refusal scope (state it precisely — .py non-test; YAML bound via recipe_fingerprint), GCS marker last-writer-wins harmless at full-digest naming, F5-7 credible, final-table/protocol-v3/F3-F4/RUN_NAME no regression. Closed as round F5c.

# REQUEST-REVISION — 2 BLOCKER, 1 MINOR

## BLOCKER — F5-5 still reports refusal for the wrong attack

[F5-5:2208](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:2208>) changes the artifact’s manifest to `0 * 64`. That tests a foreign-manifest artifact, not a forged favorable cell carrying the current public manifest.

The current manifest is recomputed locally at [pos_rollout_fit_probe.py:715](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:715>), but adoption only compares equality at [pos_rollout_fit_probe.py:1680](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1680>). Therefore a forger can copy the current context—including its public manifest digest—replace both trials with one-byte peaks, rehash, and pass. No deployed source tree is required.

I executed that construction through the real publication, loader, and adoption functions using in-memory storage. Result:

```text
[M1] adopting rollout ... (2 trials, peak 1 bytes)
local_manifest    4af0e0f2...
artifact_manifest 4af0e0f2...
adopted_peaks     [1, 1]
```

`run_fit_probe` then skips the measurer at [pos_rollout_fit_probe.py:2317](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2317>).

This can be an explicitly accepted ACL boundary, but the probe must report the same-manifest forgery as DECLARED/SUCCEEDED—not REFUSED—or authentication must be added. The residual statement at [pos_rollout_fit_probe.py:73](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:73>) must also say any authorized bucket overwriter who can read a current artifact can fabricate.

## BLOCKER — Checkpoint resume discards the manifest-bearing identity

Attempt publications already record `context_digest` at [wan_pos_rollout_trainer.py:225](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:225>), but:

- `load_publication` does not require it at [line 258](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:258>).
- `select_resume_publication` accepts only `code_sha` and `arm`, and filters only those at [line 266](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:266>).
- `resume_source` derives the complete current context and then drops everything except `code_sha` at [line 427](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:427>).

I executed the exact selector body with two same-SHA publications carrying different context digests; it selected the higher-step foreign context.

Thus two git-less deployments with the same `COMMIT` label but different manifests can resume each other’s optimizer/parameter state. Require and match `context_digest` throughout resume selection, including the launcher preflight.

## MINOR — Content names use only 48 bits of the digest

[_content_for_marker:1411](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1411>) uses `digest[:12]`. Consequently, content objects are not strictly named by their own digest: two full digests sharing that prefix contend for one object name and can re-express `marker-A → content-B`. Use the full 64-hex digest.

## Verified passes

- The manifest list is deterministic, length-framed, and covers all 300 non-test Python files, including fit probe, support, update, step, arms, loop, and trainer.
- Dirty refusal covers `.py` files under `maxdiffusion`, excluding tests. Dirty YAML does not refuse, but its loaded values are bound by `recipe_fingerprint`; that exact scope is safe but should not be described as whole-checkout cleanliness.
- GCS marker writes are last-writer-wins, not generation-conditional: `gfile.rename(..., overwrite=True)` supplies no generation precondition. Individual object replacement is atomic, so marker→A while object B also exists is harmless; readers follow the marker. Repair races likewise end at a complete object, subject to the truncated-name finding. [Cloud Storage consistency](https://docs.cloud.google.com/storage/docs/consistency), [request preconditions](https://docs.cloud.google.com/storage/docs/request-preconditions).
- F5-7 observes the real committed marker and loaded payload and is credible.
- Final-table re-decision, protocol v3 failure-closed behavior, F3/F4 measurement-path tracing, and `RUN_NAME` cell scoping show no F5b regression.
- `git diff --check`, shell syntax, and AST parsing passed. The supplied battery log hash matches and contains 87 `REFUSED` results, but F5-5 is one of the false refusals.
- I could not rerun pytest because this environment’s active Python lacks pytest.

---

# Strengthening record — round F5c (Coder, 2026-08-12)

All three findings resolved; none rejected. Narrative in `rollout_adapter_worklog.md`
(2026-08-12T05:30Z). Planner ruling carried forward: bucket-ACL boundary accepted, honesty mandatory,
no new cryptography.

## BLOCKER — F5-5 still reports refusal for the wrong attack — **FIXED (disclosure, not a patch)**

- **The finding is accepted in full.** The manifest digest is public in the payload, so the forger
  copies the current context verbatim; the local recomputation is only ever equality-compared against
  a value the forger controls. No deployed source tree is required.
- **`DECLARED` verdict class added** to the battery, with `REFUSED / DECLARED / SUCCEEDED / UNPARSED`
  counted separately and a `SUMMARY:` line. Defined in the harness README: *the attack succeeds by
  design inside an explicitly accepted, written-down trust boundary; it is not a refusal.*
- **`F5-8 forge w/ CURRENT manifest`** reports the in-boundary forgery as **DECLARED**, and is a
  tripwire: if a publication authority is added it flips to REFUSED, and its return value says the
  docstring, worklog and probe must be updated together.
- **`F5-5` retained** as `forge w/ FOREIGN manifest` — genuinely refused, a different class.
- **Residual statement corrected** at the module docstring: **any writer with bucket write access who
  can READ one current artifact can fabricate a measurement; the deployed source tree is NOT
  required.** F5b's "both the tree and write access" was wrong and flattering. What the manifest buys
  is the accident case (dirty tree, stale tarball, hand-edit, cross-code adoption); against a
  deliberate writer it buys nothing.
- **`test_the_accepted_residual_a_bucket_writer_can_forge_a_cell`** asserts the weakness deliberately,
  so it is measured rather than claimed, and so adding authentication breaks a test.

## BLOCKER — Checkpoint resume discards the manifest-bearing identity — **FIXED**

- `load_publication` now **requires** `context_digest` (fail-closed on absent).
- `select_resume_publication` takes it as a **required keyword** and filters on it — required, because
  the identity was lost by being droppable.
- `resume_source` passes the **whole derived context digest**, the same one cell adoption compares.
- **Launcher preflight:** it cannot derive the context (runs AFTER the HF prefetch but before
  distributed initialization and the model load, where `jax.devices()` reports the host's local count
  — wrong by 8x on v6e-64 — and `pyconfig` has not run, so there is no recipe fingerprint either;
  corrected in F5d, where the F5c wording had the prefetch ordering backwards), so
  it now calls `describe_resume_candidates`, which **reports and never decides**, and prints that
  adoption is settled in-process against the full context. A preflight predicting adoption from a
  commit label would be the same error as the selector matching one.
- Red-first with the reviewer's two-publication construction, **executing the real selector and the
  real `resume_source`** per the F3c liveness lesson: before the fix, `resume_source` returned the
  foreign `att-2`. Tests: `test_two_same_sha_publications_with_different_contexts_do_not_resume_each_other`,
  `test_the_selector_cannot_be_called_without_a_context_to_match`,
  `test_a_publication_without_a_context_digest_is_not_adoptable`,
  `test_resume_source_matches_the_whole_derived_context_not_just_its_sha`.

## MINOR — Content names use only 48 bits — **FIXED**

`_content_for_marker` uses the **full 64-hex digest**, so an object is named by itself rather than by
48 bits of itself and the truncation cannot re-express `marker-A -> content-B`.

## Verified-passes wording fix — **APPLIED**

The dirty-tree refusal is no longer described as whole-checkout cleanliness. Stated scope: `.py` files
under `maxdiffusion` outside `tests/`; a dirty **YAML** does not refuse, and its loaded values are
bound by `recipe_fingerprint` instead — the binding that actually decides the footprint.

## Verification

Battery **88 probes — 87 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED**
(`harness/attacks_f5c_20260812.log`, sha256 `2149fedeeee66c72…`). Canonical suite **2218 passed / 0
failed** (604 s). `black` / `ruff` / `bash -n` / `git diff --check` clean. Nothing committed.

**Note on the two UNPARSED verdicts the strict classifier surfaced:** pre-existing probes `B-1`/`B-2`
returned diagnostics before their verdict word. Their strings were reordered (content unchanged)
rather than loosening the classifier — a classifier that guesses is how this class of error survives.


---

# F5c re-review (appended 2026-08-12)

Verdict: **REQUEST-REVISION — 1 BLOCKER + 1 MAJOR + 1 MINOR, all harness/docs-layer; every production disposition PASSED** (resume context binding end-to-end with one shared derivation; report-only preflight ratified as the right design; 64-hex names; the deliberate residual test; B-1/B-2 classifier integrity). Findings: DECLARED lacks a call-site allowlist (any probe returning the word classifies as accepted-residual — executed); P3-5 lost its attack to the required-keyword change (TypeError scored as REFUSED — the third false-verdict incident this week); preflight comment misstates launcher order. Closed as round F5d (harness accounting fixes; Planner spot-check in lieu of a fifth pass — production surface unchanged).

Verdict: **REQUEST-REVISION — 1 BLOCKER, 1 MAJOR, 1 MINOR.**

1. **BLOCKER — `DECLARED` is not restricted to explicitly declared probes.**  
   [_report](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:714) accepts any return beginning with `DECLARED`; [_summarize](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:734) then labels every such result an accepted residual. The F5-8 call site is not marked specially. I executed the helper with a refusal-worded probe returning `DECLARED: accidental drift`; it was classified as `DECLARED`.  
   Require an explicit call-site declaration/allowlist—only F5-8 may currently produce `DECLARED`; an unexpected `DECLARED` must become `UNPARSED`/failure. The current tree contains only one literal `DECLARED:` return, so today’s 1 DECLARED is F5-8, but future drift is silent.

2. **MAJOR — one required-keyword caller was missed and a standing probe no longer executes its attack.**  
   [P3-5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:585) still omits `context_digest`. `_report` catches the resulting `TypeError` as REFUSED. The supplied [battery log](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/attacks_f5c_20260812.log:30) confirms P3-5 never reached selection. Pass the published digest and rerun; this is a regression to previously standing coverage.

3. **MINOR — the preflight rationale misstates launcher order.**  
   The launcher performs HF prefetch at [line 337](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:337), then starts preflight at [line 341](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:341), while comments/docstrings claim preflight precedes prefetch. The decisive topology argument remains correct; update the wording to “after prefetch, but before distributed initialization/model load.”

Everything else in scope passes:

- F5-8 honestly demonstrates current-manifest forgery and F5-5 remains the foreign-manifest refusal.
- The deliberate residual test correctly breaks if authentication is introduced.
- `load_publication` requires `context_digest`; the selector requires and matches it.
- `start_training` derives one context, uses it for authorization, then passes that exact object to `resume_source`.
- Report-only preflight is the right design: it preserves diagnostics without making an under-informed adoption decision.
- Full 64-hex names are used. An old short-named content object produces a clean `ValueError` and therefore remeasurement, not a crash.
- B-1/B-2 were correctly reordered; the classifier was not loosened.
- The log hash and counts match: 87 REFUSED / 1 DECLARED / 0 SUCCEEDED / 0 UNPARSED.
- AST parsing, `git diff --check`, and `bash -n` passed. This environment lacks pytest/JAX, so I could not independently rerun the canonical suite.

Residual wording for Yixun: **any bucket writer with write access who can read one current artifact can fabricate a measurement; the manifest protects accidental drift only and provides nothing against a deliberate writer.**
Verdict: **REQUEST-REVISION — 1 BLOCKER, 1 MAJOR, 1 MINOR.**

1. **BLOCKER — `DECLARED` is not restricted to explicitly declared probes.**  
   [_report](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:714) accepts any return beginning with `DECLARED`; [_summarize](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:734) then labels every such result an accepted residual. The F5-8 call site is not marked specially. I executed the helper with a refusal-worded probe returning `DECLARED: accidental drift`; it was classified as `DECLARED`.  
   Require an explicit call-site declaration/allowlist—only F5-8 may currently produce `DECLARED`; an unexpected `DECLARED` must become `UNPARSED`/failure. The current tree contains only one literal `DECLARED:` return, so today’s 1 DECLARED is F5-8, but future drift is silent.

2. **MAJOR — one required-keyword caller was missed and a standing probe no longer executes its attack.**  
   [P3-5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:585) still omits `context_digest`. `_report` catches the resulting `TypeError` as REFUSED. The supplied [battery log](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/attacks_f5c_20260812.log:30) confirms P3-5 never reached selection. Pass the published digest and rerun; this is a regression to previously standing coverage.

3. **MINOR — the preflight rationale misstates launcher order.**  
   The launcher performs HF prefetch at [line 337](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:337), then starts preflight at [line 341](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:341), while comments/docstrings claim preflight precedes prefetch. The decisive topology argument remains correct; update the wording to “after prefetch, but before distributed initialization/model load.”

Everything else in scope passes:

- F5-8 honestly demonstrates current-manifest forgery and F5-5 remains the foreign-manifest refusal.
- The deliberate residual test correctly breaks if authentication is introduced.
- `load_publication` requires `context_digest`; the selector requires and matches it.
- `start_training` derives one context, uses it for authorization, then passes that exact object to `resume_source`.
- Report-only preflight is the right design: it preserves diagnostics without making an under-informed adoption decision.
- Full 64-hex names are used. An old short-named content object produces a clean `ValueError` and therefore remeasurement, not a crash.
- B-1/B-2 were correctly reordered; the classifier was not loosened.
- The log hash and counts match: 87 REFUSED / 1 DECLARED / 0 SUCCEEDED / 0 UNPARSED.
- AST parsing, `git diff --check`, and `bash -n` passed. This environment lacks pytest/JAX, so I could not independently rerun the canonical suite.

Residual wording for Yixun: **any bucket writer with write access who can read one current artifact can fabricate a measurement; the manifest protects accidental drift only and provides nothing against a deliberate writer.**

---

# Strengthening record — round F5d, the closer (Coder, 2026-08-12)

All three findings resolved; none rejected. Narrative in `rollout_adapter_worklog.md`
(2026-08-12T07:10Z). Every production disposition from F5c stood; these are harness/docs fixes.

## BLOCKER — DECLARED accepted from any probe — **FIXED**

Explicit call-site allowlist `_MAY_DECLARE = {"F5-8"}`. A `DECLARED` from any other probe prints
`HARNESS FAILURE`, shows the original verdict for diagnosis, and counts **UNPARSED**. `_summarize`
returns pass/fail and the runner **exits non-zero** on any SUCCEEDED or UNPARSED. Both directions
tested (`test_only_an_allowlisted_probe_may_report_an_accepted_residual`,
`test_the_summary_fails_the_run_when_a_verdict_is_unclassifiable`,
`test_the_declared_allowlist_names_exactly_the_probes_that_earned_it`), and the gate demonstrated
load-bearing: with the foreign probe allowlisted the drift classifies `DECLARED`, with the gate
`UNPARSED`.

## MAJOR — P3-5 no longer executed — **FIXED**

The call passes `context_digest`, so the attack runs again; its REFUSED now comes from selection logic
and it additionally exercises F5c's fourth filter. **It did not succeed — no real resume hole.** The
probe also catches selector `TypeError` and reports `SUCCEEDED: THE PROBE DID NOT RUN`, so this
disappearance cannot recur silently. Sweep: line 585 was the only omitting call site across `src/`,
`bash_scripts/` and the harness. The pattern — *a probe that dies on a signature change scores as a
refusal* — is recorded in the harness README as the fifth caution, with the note that the exception
type cannot distinguish attack-refused from attack-never-ran (several genuine refusals here ARE
`TypeError`s), so the distinction belongs in the probe.

## MINOR — preflight ordering claim — **FIXED**

Corrected in the launcher comment, `describe_resume_candidates`' docstring, and the F5c record above:
the preflight runs **after** the HF prefetch (`:337`) but **before** distributed initialization and the
model load, so `jax.devices()` reports the host's local count (wrong by 8x on v6e-64) and `pyconfig`
has not run, so there is no recipe fingerprint. Conclusion unchanged: it reports candidates and never
decides.

## Verification

Battery **88 probes — 87 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED**, exit 0
(`harness/attacks_f5d_20260812.log`, sha256 `5675593ce02f0e99…`). Canonical suite **2221 passed / 0
failed** (598 s). `black` / `ruff` / `bash -n` / `git diff --check` clean. Nothing committed.
