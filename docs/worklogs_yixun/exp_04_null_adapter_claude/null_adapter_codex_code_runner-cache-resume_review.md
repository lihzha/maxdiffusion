# null_adapter — Codex code review: round R8 `runner-cache-resume`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree; new account after the issue-#9 recurrence). Working tree at HEAD `7937b4f`. Reviewer exercised adversarial IO with an in-memory gfile implementation + the real codec (path-escape probe, cross-header resume, marker type-confusion, cherry-picked fidelity subsets, concurrent-writer interleavings). Ratifications: quarantine CONDITIONALLY RATIFIED; batch-only-failure re-raise RATIFIED; fully-quarantined shard NOT RATIFIED as implemented; layout+marker-last CONDITIONALLY RATIFIED (single-writer, immutable only).

## Reviewer output (verbatim)

Context loaded:

- `docs/worklogs_yixun/experiment_SOP.md`, including Running & failure discipline, safe-boundary resume, and storage guardrail.
- `plan_null_adapter.md` v5, especially §3 and §4-P2: coverage, markers, fp16 thresholds, and M5 dtype scope.
- `null_adapter_worklog.md`, including R6 closure and R8 design/ratification entries.
- All R1–R7 reviews and strengthening records: `sigma-embed-noise`, `invert-trajectory`, `optimize-nulls`, R4a `replay-operator`, R4b `record-schema-io`, R4c `verify-replay`, `gates-module`, `runner-capacity-core`, and `runner-decode-videos`.
- Both R8 files in full: `null_adapter_shards.py` and `test_null_adapter_shards.py`.
- Composed surfaces: `null_adapter_records.py`, `null_adapter_gates.py`, and `null_adapter_runner_core.py`.
- Repository state: live HEAD is `7937b4f`, one worklog-only commit after requested `cd36899`; only the two R8 files are untracked.
- Validation: AST parsing passed. The requested pytest invocation lacked a writable temporary directory; with capture disabled, 18 non-`tmp_path` tests passed and 33 I/O tests failed during fixture setup, not execution. Adversarial I/O behavior was exercised with an in-memory gfile implementation and the real production codec.

1. **MAJOR — Marker record paths are not confined to the published shard.** [null_adapter_shards.py:280](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:280)

   `marker.files[name]` is joined directly onto `shard_path`. My probe changed it to `../../staging/other_shard.staging/record_00000.npz`, provided the correct record and SHA there, and `validate_shard` returned `valid=True`. Thus a marker can depend on staging leftovers that were never published into its shard, defeating marker-last and safe-boundary semantics.

   Concrete change: require the exact canonical bijection `sorted names[i] → record_{i:05d}.npz`, reject absolute paths, separators, `.`/`..`, duplicate filenames, and any normalized path outside `shard_path`. Add the different-shard-staging probe.

2. **MAJOR — Validated resume is not provenance- or cohort-closed.** [null_adapter_shards.py:169](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:169), [null_adapter_shards.py:306](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:306)

   Empirically:

   - `resume_plan` accepted two shards with different header fingerprints and returned complete coverage because `expected_header_fingerprint` defaults to `None`.
   - `_validate_write` accepted an fp32 record under an fp16 header and accepted A1/keyed and A2/global records in one shard.
   - A fully quarantined shard listing a foreign name was accepted; only covered strangers are checked against the manifest.

   This reopens the R4c pair-provenance and R6 header findings.

   Concrete change: make the expected header fingerprint mandatory for production resume; require expected arm and noise convention; compare each decoded record’s dtype/arm/convention with its header/current run; enforce homogeneity; and apply manifest outsider/duplication checks to quarantined names as well as covered names.

3. **MAJOR — Marker parsing is not fail-closed or total.** [null_adapter_shards.py:122](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:122)

   The parser checks only top-level keys. It accepted `schema_version=true`, negative counts, list-shaped quarantine data, and duplicate JSON keys. A marker with `quarantined="x"` then caused `validate_shard` itself to raise while constructing `dict(marker.quarantined)`, contradicting `ShardReport`’s “invalid is a value” contract.

   Concrete change: use duplicate-key-rejecting JSON parsing and one strict shared marker validator covering exact integer types, nonnegative count, unique sorted string names, dict types, canonical filenames, 64-hex digests, string quarantine reasons, and names/quarantine disjointness. Artifact parsing/read errors must become invalid reports rather than escape.

4. **MAJOR — The fidelity estimand remains caller-selectable.** [null_adapter_shards.py:354](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:354), [test_null_adapter_shards.py:578](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_shards.py:578)

   `expected_names` is optional, so a caller can gate a cherry-picked subset rather than the predeclared first eight DEV examples. Empty tables reach `max()` and raise an incidental error; negative MSE values are accepted as metrics.

   Concrete change: take the DEV manifest and derive its first eight internally, or require a locked eight-name subset artifact; reject missing, duplicate, or empty subsets and require fp32 MSE `> 0` and serialized MSE `>= 0`.

   The worst-example maxima, inclusive `1e-9` boundary handling, relative-MSE formula, exact `0.01`/`0.05` thresholds, and M5 dtype outcome are otherwise correct. The docstring’s limitation is properly scoped, but lines 34/367 should attribute writer-order non-discrimination to R6, not R7, and state explicitly that the function proves only the supplied metric-table deltas.

5. **MAJOR — Quarantine conflates pathological examples with systemic failures.** [null_adapter_shards.py:414](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:414), [null_adapter_shards.py:225](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:225)

   Every `Exception` becomes quarantineable. Configuration errors, backend failures, OOMs, and infrastructure failures can therefore become data gaps. If every singleton fails, the code permits a zero-record marker with no header and substitutes `sha256(b"")` as its provenance fingerprint. Such a shard cannot validate against the real expected fingerprint.

   Concrete change: introduce a dedicated example-divergence exception and re-raise all other failures; rerun the survivor batch to ensure the composition failure is gone; re-raise when everything fails. A quarantine-only diagnostic marker must still carry the real header, or be disallowed.

6. **MAJOR — Marker-last is unsafe under overwrite or concurrent writers.** [null_adapter_shards.py:154](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:154), [null_adapter_shards.py:229](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:229)

   Staging paths are deterministic by shard basename and `_fresh_directory` deletes them; destination publishing uses `overwrite=True`. Two writers therefore delete/overwrite one another’s staging and records. Rewriting an already complete shard also leaves its old marker visible while new data replaces the files, destroying the last validated boundary.

   Concrete change: publish to immutable, writer-unique attempt/shard paths, or enforce a real single-writer lease/conditional create; never overwrite a completed shard. Add an interleaving double-writer test and a rewrite-of-complete-shard test.

7. **MINOR — The explicit storage guardrail is absent.** [null_adapter_shards.py:212](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:212)

   `write_shard` copies all pairs and materializes every serialized blob simultaneously, with no shard-size/byte cap or free-space check. Sequential staging is otherwise bounded, and same-shard retry removes its stale staging directory.

   Concrete change: stream each record into staging while retaining only filename/digest metadata, enforce a declared maximum shard size, and have the runner enforce/log the SOP’s free-space floor and clean abandoned staging prefixes.

Ratification rulings:

1. **Quarantine policy — CONDITIONALLY RATIFIED.** Per-example retry, marker disclosure, not-covered resume, and gate-coverage honesty are the correct recovery semantics. A missing DEV/TEST name causes a hard coverage failure even below 10%, which is correctly stricter than the `>10%` invalid-observation rule. A quarantined TRAIN name must remain incomplete J2 work; it must not silently authorize P3 on a reduced TRAIN-2000 cohort without explicit acceptance.

2. **Batch-only-failure re-raise — RATIFIED.** If no singleton fails, the original batched failure is evidence of a composition/capacity bug and must propagate. Rechecking the surviving batch should strengthen the mixed-failure case.

3. **Fully quarantined shard — NOT RATIFIED AS IMPLEMENTED.** It may exist only as a zero-coverage diagnostic attempt carrying the real provenance header; otherwise all-singleton failure must be re-raised. It is never a completed safe boundary.

4. **One-file-per-record plus marker-last on `gs://` — CONDITIONALLY RATIFIED.** The layout and copy/delete-aware publication order are sound for a fresh immutable shard under one writer. They are not sound with the current overwrite and shared-staging concurrency semantics.

Final verdict: **REQUEST-REVISION — marker path escape, optional provenance, non-total marker validation, and unsafe quarantine/concurrent publication prevent R8 from providing a fail-closed resume boundary.**

---

## Strengthening record + follow-up pass (all seven findings closed; ratification conditions implemented)

Strengthening (Coder): module split into `null_adapter_shards.py` (storage, 321 exec LOC) + `null_adapter_cache_policy.py` (fidelity gate + quarantine, 103); canonical-bijection paths recomputed on read; mandatory resume provenance (fingerprint/arm/convention, signature-pinned); total fail-closed marker validation; manifest-derived fidelity subset; `ExampleDivergenceError`-only quarantine with survivor-batch rerun and real-header diagnostic shards (explicit-header `write_shard` signature); attempt-unique staging, no completed-shard overwrite, `discard_incomplete_shard`; streamed writes under `MAX_SHARD_BYTES`. Suite **523 passed in 35.9s** (pre-R8 392 unchanged); **47 mutants, 1 documented survivor** (S2 containment check, redundant-by-construction defence-in-depth).

### Follow-up reviewer verdict (verbatim)

Context loaded:

- HEAD `4f99ea9`; four expected untracked files only.
- R8 review briefing, plan §4-P2, R6–R8 worklog contracts, and SOP safe-boundary/storage rules.
- Both implementation modules, both test files, record codec, and repository-wide call sites.
- Validation: AST parse 4/4; 67 non-filesystem tests passed. The full pytest command could not initialize due the read-only temp directory; all requested adversarial I/O probes passed using in-memory `gfile` with the production codec.

Finding verification:

1. **RESOLVED** — canonical mapping is recomputed and confined; both the staging path escape and shuffled-but-consistent bijection probes are refused.
2. **RESOLVED** — fingerprint/arm/convention are mandatory keyword-only expectations; record dtype, arm, convention, homogeneity, and quarantined manifest membership are enforced; cross-header resume excludes the foreign shard.
3. **RESOLVED** — duplicate-key rejection and strict marker validation landed; the parser battery passes and `quarantined="x"` produces an invalid report without escaping.
4. **RESOLVED** — `fidelity_gate` derives the first eight DEV names internally, requires exact table coverage and valid MSEs, rejects the cherry-picked subset, and correctly attributes the writer-order limitation to R6.
5. **RESOLVED** — only `ExampleDivergenceError` is quarantined; generic failures propagate, survivor-batch results are returned, batch-only/all-example failures re-raise, and diagnostic shards carry the real header.
6. **RESOLVED** — staging is attempt-unique, completed/incomplete destinations are protected, publication uses `overwrite=False`, and the synchronized double-writer probe produced one loser plus one valid immutable shard.
7. **RESOLVED for the R8 storage unit** — record blobs are serialized/written one at a time, `MAX_SHARD_BYTES` is enforced, and attempt staging is cleaned in `finally`; the runner still owns the SOP free-space-floor and stale-attempt sweeping obligations.

S2 ruling:

**KEEP** — the containment check is deliberately redundant with the filename grammar but directly preserves the security invariant if that grammar changes; this is justified defense-in-depth consistent with R3 M11 and R4c C10.

Split/signature check: **CLEAN** — the policy module is pure and loses no behavior, storage has no reverse policy dependency, repository-wide search found no stale imports, and mandatory explicit `header` on `write_shard` correctly fixes diagnostic-shard provenance.

Any NEW findings:

- None at invalidate-conclusions or unsafe-resume severity.

Final verdict: **APPROVE — all six MAJOR findings, the MINOR storage finding within this unit, and the ratification conditions are correctly strengthened without a new blocking defect.**
