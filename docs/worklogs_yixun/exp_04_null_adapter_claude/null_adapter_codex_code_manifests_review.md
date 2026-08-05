# null_adapter — Codex code review: round R9 `manifests`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `5500833`. Reviewer independently verified the dual-source identity rule against the producer chain (upstream format, meta provenance, no legitimate disagreement case; canonical-preimage digests re-computed) and ran adversarial probes (mid-shard overshoot, reauth-poisoned gsutil, reversed listings, corrupt-artifact loading, malformed meta). M5 ruling: KEEP (documented equivalent mutant).

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md`; exp_04 query; both standing announcements; relevant exp_02 analysis and issue #6 reauthentication policy.
- `plan_null_adapter.md` v5 §3 and the complete §4-J0 contract.
- `null_adapter_worklog.md` through R9, including the dual-source derivation, Planner positions, and mutation-M5 survivor record.
- All R1–R8 review and strengthening files: R1–R3, R4a/R4b/R4c, R5–R8.
- Both R9 files in full: [builder](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py) and [41-test suite](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_manifests.py).
- Main checkout’s Wan2.2 submodule at `f37022874c588817d4ed77d463e3d27745053df4`: `make_droid_window_plan.py`, `precompute_features_droid_plan.py`, plus the complete TFRecord producer’s name/meta path.
- Validation: both files AST-parse. The exact pytest invocation could not initialize because the read-only sandbox has no usable temp directory; with capture disabled, 28 tests passed and 13 `tmp_path` tests failed during fixture setup only.
- Independent probes covered mid-shard stopping, cap call counts, reauth-poisoned `gsutil`, malformed metadata, `v1`, leading zeros, Unicode prefixes, 63 validation episodes, listing-order provenance, and corrupt artifact loading.

1. **MAJOR — There is no executable J0 builder that pins the approved scan contract.**  
   [build_null_adapter_manifests.py:153](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:153), [build_null_adapter_manifests.py:233](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:233), [build_null_adapter_manifests.py:310](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:310)

   R9 provides generic primitives whose target and caps default to `None`, but no `main`/top-level build function pins 14,636 VAL records, 5,000 TRAIN episodes, 200 shards, or 60 GiB. It also never requires a train scan to have reached 5,000 episodes: the tests routinely produce valid manifests from only 1,016 train episodes. No production path performs deterministic listing → both scans → target assertions → selection/binding → all-or-nothing publication.

   Concrete change: add one J0 entry point with fixed constants, deterministic shard listing, exact VAL-count validation, mandatory `train_scan.episodes >= 5000`, the fixed caps, and publication only after every scan/selection/binding check passes. Test listing exhaustion below 5,000 and every cap branch with a forbidden writer.

2. **MAJOR — Early stop is shard-granular, contrary to “as soon as 5,000 distinct episodes are seen.”**  
   [build_null_adapter_manifests.py:179](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:179), [build_null_adapter_manifests.py:193](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:193), [test_null_adapter_manifests.py:249](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_manifests.py:249)

   A one-shard probe containing ten distinct episodes with target six returned all ten records and episodes. Those overshoot episodes enter the hash ordering and can silently change TRAINFIT/TRAIN-2000. The existing test places the target exactly at a shard boundary and misses this. Also, at `max_shards=2`, the code still calls `sizer` on shard three before raising.

   Concrete change: stop record iteration immediately when the newly observed episode reaches the target; assert exact reader-yield and next-shard/sizer call counts. Check the shard-count cap before statting/sizing the next shard, then check its size against the byte cap.

3. **MAJOR — The shard-listing checksum is unbound caller input, and listing order is discarded.**  
   [build_null_adapter_manifests.py:281](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:281), [build_null_adapter_manifests.py:315](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:315)

   `Manifests` retains only a sorted set of selected shard paths; it loses each scan’s full ordered listing. `write_manifests` then accepts any checksum string. My reversed-listing probe produced identical `Manifests`, so a checksum over sorted paths—or simply `"b"*64`—can be written even though early-stop selection depends on ordering.

   Concrete change: preserve the complete ordered VAL and TRAIN listings, compute a domain-separated checksum internally over split, ordinal position, and exact path, store the ordered listings alongside the digest, and remove the free-form checksum argument from the production builder.

4. **MAJOR — GCS/local bindings do not fail closed or establish the bytes actually scanned.**  
   [build_null_adapter_manifests.py:287](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:287)

   Two probes succeeded incorrectly:

   - Valid-looking stdout plus `stderr="ReauthUnattendedError: Reauthentication required"` was accepted.
   - `Generation: Reauthentication required` was stored literally as the generation.

   This violates standing issue #6. Parsing also permits duplicates/unrecognized output, and binding happens independently of scanning, leaving a stat/read replacement race. Locally, `mtime_ns + size` does not identify exact bytes despite the docstring’s claim.

   Concrete change: reject any nonempty/unclassified stderr and any reauth marker in either stream; require exactly one decimal generation and one nonnegative decimal size; reject duplicate fields. Bind before and after each scan—or read a generation-pinned object—and require equality with the size used for caps. For local shards, stream a content SHA-256 rather than treating mtime/size as exact identity.

5. **MAJOR — Artifact validation accepts materially corrupt cohort evidence, and publication is not race-safe.**  
   [build_null_adapter_manifests.py:347](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:347), [build_null_adapter_manifests.py:359](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:359), [build_null_adapter_manifests.py:387](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:387)

   The loader correctly checks fixed sizes, within-file duplicate names, and DEV/TEST overlap, but my in-memory probe also accepted all of these simultaneously:

   - `schema_version: true` and a header containing no other provenance;
   - the same name in DEV and TEST;
   - duplicate episodes within DEV;
   - TRAINFIT/TRAIN-2000 episode overlap and three TRAIN windows from one episode;
   - string ordinal, negative shard size, and object-valued generation.

   It additionally does not check hash ordering, row/header binding consistency, header cohort sizes, or exact header schema. The check-then-`GFile(..., "w")` publication can overwrite in a concurrent-writer race, reopening R8’s immutability ruling.

   Concrete change: use duplicate-key-rejecting strict JSON; validate exact integer types, canonical episode strings, names, ordinals, bindings, header fields and fixed sizes; enforce cross-cohort name uniqueness, DEV/TEST and TRAINFIT/TRAIN-2000 episode disjointness, ≤2 TRAIN windows per episode, and prescribed hash/ordinal ordering. Cross-check every row against the header binding. Publish through exclusive/conditional creation or an attempt-unique staged directory with a final completion signal.

6. **MAJOR — Malformed present metadata silently bypasses the dual-source identity check.**  
   [build_null_adapter_manifests.py:99](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/build_null_adapter_manifests.py:99), [test_null_adapter_manifests.py:101](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_manifests.py:101)

   `episode_id_for("ep7_v1_s00004", b"not json")` returned `"7"`. A nonempty corrupt second source is treated like an absent source, allowing a wrong but syntactically valid name to evade the intended agreement check. Invalid JSON is not a legitimate producer case; missing metadata is specifically represented by `b"{}"`.

   Concrete change: reject malformed/non-object nonempty JSON and duplicate keys; accept the name-only path only for valid metadata lacking `episode_id` (including `{}`). Also reject negative episode integers. Add the valid-name/malformed-meta regression.

Identity-rule verification verdict:

- The upstream format is exactly `ep{ep}_v{view}_s{start:05d}`; `v1` is valid because `view` is variable. The TFRecord producer preserves the supplied name byte-for-byte, including a possible `split/` prefix.
- `meta.json["episode_id"]` is written as `int(ep)` and copied verbatim; missing `meta.json` becomes `b"{}"`.
- Both values originate from the same `ep`, pass through the same cache directory, and are never independently transformed. I found no legitimate real-cache disagreement. Refusal on disagreement is correct; only manual renaming/cross-copying or corruption creates it.
- Leading-zero name IDs normalize correctly against integer metadata; Unicode prefixes do not alter basename extraction or stored-name fidelity.
- Decimal canonical preimages are correct. Independent digests are:
  - `"0"` → `5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9`
  - `"7"` → `7902699be42c8a8e46fbbb4501726517e86b22c56a189f7625a6da49081b2451`
  - `"30738"` → `b51d6398dac7426234ec52dbe930937892cec74d244ea263a0b86705b06a8ed4`
- The pure cohort selector’s hash ordering, 64/64 slicing, 16-episode exclusion, ≤2-window fill, exact-2,000 stop, underfill refusal, and 63-episode VAL refusal are correct.
- Mutation **M5: KEEP**. The `len(available)` operand is presently redundant with slicing but harmlessly restates `min(2, available)` and protects that rule if slicing is refactored. It is a documented equivalent mutant, not missing observable coverage.

Final verdict: **REQUEST-REVISION — the selector itself is sound, but missing production-level scan pins, mid-shard overshoot, poisonable provenance, and permissive artifact validation can silently redefine or misbind every experimental cohort.**

Status: No subprocesses are running; review is complete, and it is safe to steer now.
