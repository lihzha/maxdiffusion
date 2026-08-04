# null_adapter — Codex code review: round R4b `record-schema-io`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `c610580`. Reviewer ran the suite and empirical probes (stale-hash serialization, dtype/endian/type confusion, npz key injection, header asymmetry); confirmed M5 exactness, mechanical import swaps, byte-determinism mechanism, and R4c-sufficiency of the record+header pair.

## Reviewer output (verbatim)

Context loaded:

- `docs/worklogs_yixun/experiment_SOP.md`.
- `plan_null_adapter.md` v5, especially §4-P2, §5 item 5, and §6’s R4a/R4b/R4c amendment.
- `null_adapter_worklog.md` through the R4b entry and all four Planner decisions.
- R1–R4a code reviews and strengthening records; relevant M5/F14/N8 plan-review history.
- All three new files in full: `null_adapter_records.py`, `test_null_adapter_records.py`, and `bit_test_helpers.py`.
- The complete four-file import-swap diff.
- Main-checkout `verify_reconstruction_from_null.py` at pinned Wan SHA `f37022874c588817d4ed77d463e3d27745053df4`.
- Supplemental SOP briefing: exp_04 query, both announcements, and related exp_02 analysis.
- Repository state: uncommitted R4b changes at HEAD `c610580`.

Findings:

1. **MAJOR — the writer can emit a record whose stored integrity hash is already invalid.** [null_adapter_records.py:205](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:205)

   `record_to_bytes()` calls `_validate()`, but `_validate()` never recomputes `expected_final_latent_sha256`. Empirically, both `dataclasses.replace(record, expected_final_latent_sha256="0"*64)` and in-place mutation of the supposedly frozen record’s NumPy array serialized successfully; only the subsequent reader rejected them.

   Concrete change: recompute and compare the expected-latent digest during writer validation, rejecting stale hashes. Add tests for a replaced hash and post-`make_record` array mutation; optionally mark stored arrays read-only.

2. **MAJOR — the public artifact boundary does not enforce the authoritative production schema or finite data.** [null_adapter_records.py:109](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:109), [null_adapter_records.py:48](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:48)

   The reader accepts `actions [1,1]`, `nulls [4,1,1]`, and NaN `expected_final_latent`, provided the few relational checks hold. This conflicts with §4-P2’s exact `[25,16,4096]` null schema and R3’s ruling that hard dimensions belong at the production runner/**artifact boundary**. Deferring checks to R6/R8 protects one future writer, not `record_from_bytes`, R4c, or trainer consumers.

   Concrete change: make the public production codec enforce `z_video/z_start/expected=(48,9,12,20)`, `z_i0=(48,1,12,20)`, `actions=(32,7)`, `nulls=(25,16,4096)`, and losses `(25,)`; reject nonfinite tensors, losses, and `final_future_mse`. Keep any tiny-shape codec seam private to tests.

3. **MAJOR — provenance deserialization does not enforce the provenance writer’s own contract.** [null_adapter_records.py:246](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:246), [null_adapter_records.py:257](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:257)

   `header_to_json()` rejects nested optimization config and non-1-D sigmas, but `header_from_json()` accepts them. It also accepts NaN sigmas. Both header and record readers accept `schema_version=true`, because `True == 1`. This is not fail-closed provenance for R4c.

   Concrete change: introduce one shared header validator called on both write and read; require an exact integer schema version, finite one-dimensional production sigma vector, finite guide scale/config floats, valid dtype policy, integral `l_null`, and the flat optimization-config contract. Add asymmetric-reader and type-confusion tests.

4. **MINOR — record metadata/archive namespaces are not explicitly closed.** [null_adapter_records.py:214](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:214)

   Missing fields and injected NPZ arrays happen to fail through `KeyError`/`TypeError`, rather than an explicit schema check, while an injected key under `meta["shapes"]` is silently accepted.

   Concrete change: validate the exact archive member set, exact metadata key set, and exact shapes key set before loading/constructing arrays; reject duplicate ZIP members. Add deletion, extra-key, duplicate-key, and extra-shape-key tests with matched `ValueError`s.

5. **MINOR — the public hashing helper does not implement its documented little-endian canonicalization.** [null_adapter_records.py:94](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:94)

   `np.ascontiguousarray()` preserves byte order. Equal numeric `<f4` and `>f4` arrays produced different digests in the probe. Current expected-latent records happen to be normalized before hashing, but the helper’s contract and planned reuse are unsafe cross-endian.

   Concrete change: convert to the input dtype’s explicitly little-endian equivalent before hashing and add a big-endian-input equality test.

6. **MINOR — “record alone” is inaccurate; replay requires the record-plus-header pair.** [null_adapter_records.py:3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py:3), [test_null_adapter_records.py:3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_records.py:3)

   Sigmas, guide scale, model revision, and base-context fingerprint live only in `ProvenanceHeader`.

   Concrete change: say “record plus its shard provenance header” and make that pairing explicit in R4c’s API.

Validation and rulings:

- Focused R4b suite: **23 passed**.
- Full worklog suite: **137 passed, 1 failed**, with only the acknowledged tiny-Wan tmpdir failure.
- Dtype confusion, truncated arrays, big-endian archive members, and expected-latent tampering were correctly rejected.
- Header field reordering was correctly accepted; missing header keys were correctly rejected.
- The fixed-epoch ZIP mechanism is sound: stable member order, sorted metadata JSON, `ZIP_STORED`, and fixed timestamps make repeated output byte-identical under the same Python/NumPy serialization format.
- M5 is implemented exactly: fp32 fallback changes only `nulls`, `z_start`, and `expected_final_latent`; `z_i0` and `z_video` remain fp16.
- The four import swaps are genuinely mechanical and characterization-safe.
- The module itself is NumPy-only: direct standalone loading imported zero JAX modules. A normal `maxdiffusion.null_adapter_records` import loaded JAX through the existing package initializer, so this proves module independence—not a JAX-free process. That is sufficient for R4c/R6/R8, which already run in the MaxDiffusion stack, but not for a strictly JAX-free external CLI.

Schema R4c-sufficiency: **Yes, as a `NullAdapterRecord` plus `ProvenanceHeader` pair.** Every replay input from the reference is represented: `z_i0`, `z_start`, nulls, sigmas, guide scale, model revision, reproducible base-context fingerprint, dtype policy, and expected final latent plus hash. `actions` and `z_video` are intentionally extra for training and must remain unused by R4c. No field needs adding, but the boundary validation above should be fixed before R4c depends on it.

Final verdict: **REQUEST-REVISION — the field set and serialization mechanics are fundamentally sound, but stale-hash serialization and non-fail-closed production/provenance validation must be corrected before R4c builds on this boundary.**

Status:

- exp_04 R4b review is complete; no subprocesses are running.
- Earliest steer: now; it is safe to strengthen R4b before commit or R4c begins.

---

## Strengthening record (Coder, same round; all six findings closed, none rejected)

1. **MAJOR (stale-hash writer) — FIXED.** `_validate` recomputes the expected-latent digest on every validation (write and read); all stored arrays frozen read-only by both writer and reader; replaced-hash, post-make_record-mutation, and read-only-enforcement tests added.
2. **MAJOR (production boundary) — FIXED; the ruling supersedes the Planner's earlier structural-validation acceptance.** Module-private `_Geometry` table defaulted to production; public codec = thin wrappers pinning `PRODUCTION_GEOMETRY`; 100%-shared validation path; nonfinite tensors/losses/mse rejected; public path exercised at real geometry incl. an explicit tiny-arrays-refused test. Mutant S5 (geometry from caller — the old behavior) killed by 6 tests.
3. **MAJOR (header asymmetry) — FIXED.** One `_validate_header` on both paths; exact-integer schema_version (bool rejected), finite len-26 sigma vector, finite floats, integral l_null, flat config; 11 cases asserted on writer AND hand-built-JSON reader in the same test body.
4. **MINOR (namespaces) — FIXED.** Duplicate-ZIP-member rejection + exact member/meta/shapes key sets via `_check_namespace`, matched ValueErrors, 6 tests.
5. **MINOR (endianness) — FIXED.** `newbyteorder("<")` canonicalization; >f4-vs-<f4 single-digest test.
6. **MINOR (wording) — FIXED.** "Record plus its shard provenance header"; R4c takes the pair.

Suite **166 passed in 14.64s** (+28); mutants S1–S10 killed; module still provably numpy-only standalone. `ordinal` integrality added (same type-confusion class, flagged). Behavior changes beyond the findings: none. Round R4b closed; committed with this record.
