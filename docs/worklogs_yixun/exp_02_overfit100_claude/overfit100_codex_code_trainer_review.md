# Code review: exp_02 overfit100 — cycle C (trainer)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-30

## Context loaded
- Experiment SOP and announcements — review, validation, reproducibility, and TPU-approval requirements.
- Approved plan v4, Queries 1–5, and worklog — D7–D9, H2, dataset publication, and current V2 recalibration context.
- Prior plan reviews and cycle-B strengthening record — trainer seams, checkpoint retention, and fingerprinted schema-v2 publication.
- exp_01 analysis/results — objective precedent, checkpoint layout, recipe, and known limitations.
- Implementation, parent trainers, Wan pipeline, configuration, launcher, and four new test files — focused parity and failure-path audit.
- Reported 737-passed suite and mutation record — independently supplemented by successful compile, YAML, `bash -n`, and `git diff --check`; pytest could not be rerun in the read-only sandbox.

## Verdict

REQUEST-REVISION. The loss is genuine full-FT parity with the intended gathered-context delta, and the explicit checkpoint manager/restore contract is sound. Before TPU launch, however, the trainer must bind the pretrained snapshot and published dataset bytes fail-closed, and it must prevent a different evaluation set from silently using incompatible context-table rows.

## Findings

1. **C1 — MAJOR — Training is not bound to the model revision that produced the dataset.** The manifest pins revision `b8fff731…`, but the training config leaves `revision` empty and the launcher prefetches/loads the mutable repository default; the Wan loader also receives only the repository/path, without a revision (`docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json:3656`, `src/maxdiffusion/configs/base_wan_5b_overfit100.yml:62`, `src/maxdiffusion/configs/base_wan_5b_overfit100.yml:67`, `bash_scripts/train_wan_overfit100.sh:91`, `bash_scripts/train_wan_overfit100.sh:143`, `src/maxdiffusion/pipelines/wan/wan_pipeline.py:119`, `src/maxdiffusion/pipelines/wan/wan_pipeline.py:271`). A future repository update could silently change the transformer or T5 context table while the run still claims the reviewed recipe. **Concrete change:** make the manifest revision a required launcher input, prefetch that exact revision, pass the resolved local snapshot directory to training, log/assert its commit and relevant config hashes, and test that an unpinned or mismatched snapshot is rejected before pipeline loading.

2. **C2 — MAJOR — Dataset readiness is both late and weaker than cycle B’s fingerprint contract.** `assert_dataset_ready` checks marker existence and a metadata count only; it does not verify `_SUCCESS.summary_sha256`, the summary’s shard hashes/sizes, or the exact canonical shard set, and broad exception handling can silently replace an unreadable marker count with `summary.json` (`src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:258`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:267`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:284`). Moreover, it runs only when `_load_dataset` is called after the 5B pipeline, optimizer, and state have already been built (`src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:657`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:676`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:707`). Cycle B already publishes the necessary hashes (`src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:1630`, `src/maxdiffusion/data_preprocessing/build_overfit100_dataset.py:1697`). **Concrete change:** add a lightweight preflight before model loading—and preferably before HF prefetch—that requires a readable, structurally valid `_SUCCESS`, verifies the raw summary hash, exact shard names/counts, and canonical shard fingerprints, then add same-count byte-mutation and “pipeline loader must not run” regressions.

3. **C3 — MAJOR — A distinct evaluation directory can silently bind examples to the training set’s context mapping.** The context table is always constructed from `train_data_dir`, while a different `eval_data_dir` is accepted after checking only `_SUCCESS`; no equality/subset check is made between the two `episodes.json` mappings, and parsed `episode_index` has no range assertion before the JIT gather (`src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:155`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:558`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:565`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:590`). A valid-but-different index mapping therefore evaluates with the wrong text without an error. **Concrete change:** either require `eval_data_dir == train_data_dir`, or verify every evaluation `(episode_index, episode_id, used_text)` against the training mapping and assert `0 ≤ episode_index < num_text_slots` in the parser; test incompatible same-sized mappings and out-of-range records.

4. **C4 — MINOR — Combining `checkpoint_steps` and `checkpoint_every` violates H2’s exact-list contract.** `CheckpointScheduler` emits their union, so an accidental nonzero cadence can create unplanned retained checkpoints at roughly 30 GB each (`src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:443`, `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:464`). The shipped config avoids this with zero, but the approved contract says the explicit list governs. **Concrete change:** make a non-empty list suppress cadence or reject configurations setting both; replace the union test with the selected precedence/mutual-exclusion contract.

## Deviation judgments

1. **ACCEPT** — the fifth `_build_optimizer` seam is the correct expression of D9: it uses the shared schedule/optimizer factory while fixing warmup at an absolute 250 optimizer counts across resumed extensions.

2. **ACCEPT** — module-level parse helpers are cleaner to test and the override still exclusively owns their use; nesting was a description of the parent seam, not a required structure.

3. **ACCEPT-WITH-HARDENING** — `_SUCCESS.records` primary plus `summary.json` fallback is reasonable only when `_SUCCESS` was read and validated successfully; it must not mask marker read or integrity failures.

4. **CHANGE** — `expected_windows` may remain training-set-specific, but a different evaluation set needs explicit episode/context-map compatibility validation, not `_SUCCESS` alone.

5. **ACCEPT** — the scheduler drives the actual loop, same-run final saves are deduplicated, and resume at an already-complete step safely reaches Orbax’s existing-step no-op.

6. **CHANGE** — a non-empty explicit list should suppress cadence, or the configuration should reject both knobs; union semantics weaken the exact retained-set and storage-budget contract.

7. **DEFER TO CYCLE D** — `episode_id` is unnecessary for the training objective; cycle D must parse and shard `name`, `episode_id`, `episode_index`, and `window_start` for aggregation.

8. **CHANGE** — narrow `_read_json` handling: transient/permission/read/JSON failures for `_SUCCESS` must fail loudly; fallback should occur only for a successfully validated marker that explicitly lacks the optional count field.

9. **ACCEPT** — for finite, in-bounds values, gather-then-cast and cast-then-gather have identical elementwise bf16 rounding; index bounds and mapping compatibility are the real uncovered edge addressed by C3.

10. **OUT OF SCOPE, FILE SEPARATELY** — the parent’s `checkpoint_keep_period=-1` behavior is a shared-trainer bug worth fixing, but it does not affect this trainer because its manager explicitly uses `max_to_keep=None` and omits `keep_period`.

---

# Strengthening record (2026-07-30 — Claude Opus 5, Coder)

All four findings **accepted and fixed**; all ten deviation judgments actioned. Tests were
written first for every new behavior (red evidence below), then implemented to green. Suite:
**787 passed / 2 skipped** (140 of those are this round's five overfit100 trainer files; the
delta over the pre-strengthen 737 includes the parallel V2-envelope round's additions to
`test_overfit100_gates.py`, which this round did not touch).

## C1 — MAJOR, training not bound to the pinned model revision — **FIXED**

Mirrors `build_overfit100_dataset.sh`'s pattern exactly: the **manifest is the source of the
repo + revision**, not the launcher.

- `bash_scripts/train_wan_overfit100.sh`: new `MANIFEST_PATH` env (default the committed
  manifest); reads `vae_fingerprint.{hf_repo,revision}` from it and refuses to start unless the
  revision is 40-hex; prefetches **that exact revision** with `HF_PREFETCH_REVISION` and **no
  allow-pattern argument** (full repo — this trainer needs transformer + T5 + VAE, unlike the
  VAE-only build); resolves the revision to one local snapshot dir with
  `snapshot_download(..., local_files_only=True)` so resolution can never silently fetch a
  different revision; hard-fails if the resolved `MODEL_DIR` does not carry the revision; then
  passes `pretrained_model_name_or_path=<snapshot dir>`, `expected_model_revision`, and
  `model_manifest_path` to training.
- `configs/base_wan_5b_overfit100.yml`: new `model_manifest_path` (defaulted to the committed
  manifest) and `expected_model_revision` (launcher-filled). The bare repo id left in
  `pretrained_model_name_or_path` is now documented as a placeholder that the trainer rejects.
- `WanTI2VOverfit100Trainer._validate_pinned_snapshot(config)`: runs as the **first act of
  `start_training`, before `_load_wan_pipeline`**. The manifest is the authority — a launcher
  claiming a revision the manifest does not pin is refused; a missing/non-40-hex revision is
  refused; and **both** `pretrained_model_name_or_path` and
  `wan_transformer_pretrained_model_name_or_path` (the path `wan_pipeline.py:271` actually
  loads) must carry the revision as a path component, which is exactly what rejects the mutable
  bare repo id. Logs the pinned revision and the resolved snapshot.

Tests (`test_overfit100_preflight.py`): pinned path accepted; **bare repo id rejected**;
different-revision snapshot rejected; explicit unpinned transformer path rejected; empty and
non-sha revisions rejected; manifest supplies the revision when the config is silent;
launcher/manifest disagreement rejected; unreadable manifest path rejected (not ignored); the
committed manifest still pins `b8fff731…`; and `start_training` rejects an unpinned snapshot
**with the pipeline loader booby-trapped**.

## C2 — MAJOR, dataset readiness late and weaker than cycle B's contract — **FIXED**

New `verify_dataset_integrity(data_dir, expected_windows, *, verify_bytes=True)` runs in
`_preflight_dataset()` as `start_training`'s **first I/O, before the pipeline, optimizer, and
state exist**. Fail-closed chain: structurally valid `_SUCCESS` → `summary.json`'s **raw bytes**
hash to `_SUCCESS.summary_sha256` (one check that binds every downstream fingerprint and count)
→ the canonical `*.tfrecord` set is **exactly** what the summary lists (a stray shard would
otherwise silently join the globbed training stream) → per-shard `size` via
`tf.io.gfile.stat` → per-shard `sha256` **and** `md5` recomputed from one streaming read →
record counts agree marker↔summary↔shards↔`expected_windows`. Returns a report that the startup
log prints (records, source, shards, `bytes_verified`, build commit, summary hash).

Two deliberate design decisions, both documented in the code:

- **`generation` is NOT verified.** Cycle B stats the **staging** object
  (`ShardWriter.flush` → `stat_object(staging_uri)`), then `promote()` **copies** staging →
  canonical, minting a new generation. Verifying the recorded generation against the canonical
  object would therefore *always* fail. `md5` is content-derived and survives the copy, so it is
  checked instead — and re-hashing the bytes we are about to read is strictly stronger than
  trusting any remote metadata field, needs no `gsutil`/`google-cloud-storage`, and behaves
  identically on a local path and on GCS (`tf.io.gfile` only).
- **`dataset_verify_bytes`** (new yml key, default `True`, launcher-exposed) gates only the
  streaming hash (~375 MB train100 / ~38 MB train10, read once before the 5B load). Every
  metadata binding stays in force when disabled, and the log records `bytes_verified=False`.

Judgment 8 honoured: `_read_json_strict` replaces the broad-except. Missing / unreadable /
non-JSON / non-object / structurally-invalid `_SUCCESS` all fail **loudly**; the `summary.json`
count fallback fires **only** for a marker that read and validated fine and merely omits the
optional `records` field (reported as `records_source`).

Tests: valid publication verifies and reports; missing marker; unparseable marker (no
fallback); each of the five required marker fields dropped (parametrized); sanctioned
`records`-absent fallback; summary byte mutation breaks the binding; missing summary; wrong set
name; extra shard on disk; listed-but-absent shard; summary listing no shards; shard size
mismatch; **well-formed same-count/same-size content mutation caught**; md5 mismatch; stale
generation tolerated; `expected_windows` mismatch; marker count vs shard records disagreement;
`verify_bytes=False` behaviour; and **"the pipeline loader must not run"** with the loader
booby-trapped.

> **Test-design correction found while writing this.** The first version of the byte-mutation
> regression flipped one bit and asserted the record count was unchanged — it is not: TFRecord's
> per-record CRC32C makes `TFRecordDataset` raise `DataLossError`. That is a *mid-run* defence
> (after the 5B load, when the reader reaches the record), and it misses the mutation that
> matters: a **re-serialized** shard with correct CRCs, identical length, identical count, and
> different latent values. The regression now performs exactly that mutation (only the content
> hash catches it), and a second test documents the bit-flip case being caught at preflight
> rather than mid-training.

## C3 — MAJOR, eval dir can silently mis-map context — **FIXED** (judgment 4)

- `read_episode_mapping(data_dir)` → `episode_index → {episode_id, used_text}`;
  `read_episode_texts` now delegates to it (contiguity/count/empty-text checks unchanged).
- `assert_context_map_compatible(train_dir, eval_dir)`: every eval index must exist in the
  training mapping and agree on the **full `(episode_index, episode_id, used_text)` triple** —
  the training set is what builds the table, so a same-sized-but-different mapping would score
  against the wrong instruction. Called from `_preflight_dataset` whenever
  `eval_data_dir != train_data_dir` (which is also integrity-verified in its own right).
- **Range assert in the parse path**: `_schema_v2_prepare_sample` wraps `episode_index` in
  `tf.debugging.assert_greater_equal(0)` + `assert_less(num_text_slots)` under
  `tf.control_dependencies`. The tf.data graph is the cheap correct place — a scalar compare per
  record, failing loudly, whereas a `jnp` gather **clamps** out-of-range indices silently, so
  index 99 against a 10-row table would have trained on row 9's text with no error anywhere.

Tests: identical mapping compatible; same-sized different `used_text` refused; different
`episode_id` refused; eval index outside the training mapping refused; index ≥
`num_text_slots` and negative index both refused in the parse path; boundary indices `0` and
`N-1` accepted; and `start_training` refuses an incompatible eval mapping with the pipeline
loader booby-trapped.

## C4 — MINOR, union violates H2's exact-list contract — **FIXED** (judgment 6)

`CheckpointScheduler.should_save` is now **precedence, not union**: a non-empty
`checkpoint_steps` suppresses `checkpoint_every` entirely; the cadence remains the fallback only
when no list is configured. `precedence_note()` returns one log line naming the ignored cadence
when both are set non-trivially, and `start_training` logs it beside the planned step list. The
union test is replaced by the precedence contract (`[250,1750]` + `every=1000` at
`max_train_steps=2000` → `[250, 1750, 2000]`, with `1000` explicitly absent), plus
note-present / note-absent tests.

## Deviation judgments — actions taken

| # | Judgment | Action |
|---|---|---|
| 1 | ACCEPT (`_build_optimizer` absolute warmup) | kept unchanged |
| 2 | ACCEPT (module-level parse helpers) | kept unchanged |
| 3 | ACCEPT-WITH-HARDENING (`records` fallback) | hardened per C2/judgment 8 — fallback only for a validated marker missing the optional field |
| 4 | CHANGE (eval-set compatibility) | implemented as C3 |
| 5 | ACCEPT (scheduler drives the loop) | kept unchanged |
| 6 | CHANGE (list suppresses cadence) | implemented as C4 |
| 7 | DEFER TO CYCLE D (`episode_id` in the batch) | left out of the batch; an explicit NOTE in `_schema_v2_prepare_sample` tells cycle D to extend **both** it and `_data_shardings` with `name`/`episode_id`/`window_start` |
| 8 | CHANGE (narrow `_read_json` handling) | `_read_json_strict`; broad-except removed |
| 9 | ACCEPT (gather-then-cast bf16) | kept unchanged; the bounds edge it flagged is closed by C3 |
| 10 | OUT OF SCOPE, FILE SEPARATELY (parent `keep_period=-1`) | not touched here; already pinned by two tests (eviction with `keep_period=0`, accidental-retention with `-1`). **Flagged to the Planner to file as a separate shared-trainer fix.** |

## Verification

- **Red evidence.** 46 failures across the new/changed tests before implementation
  (`test_overfit100_preflight.py` 40 — including collection-level absence of
  `verify_dataset_integrity` / `assert_context_map_compatible` / `_validate_pinned_snapshot` —
  plus 4 in `test_overfit100_trainer_data.py` and 3 in `test_overfit100_checkpoint_schedule.py`).
- **Mutation spot-checks** (each applied to the real module, then restored byte-identical):
  revision-in-path check dropped → 4 failures; `summary_sha256` binding dropped → 1;
  `_preflight_dataset()` moved after `_load_wan_pipeline()` → 3; eval context-map check dropped
  → 1; parse-path range assert dropped → 2; precedence reverted to union → 1. **All caught.**
- **Static.** `py_compile`, `yaml.safe_load` (198 keys), `bash -n`, `git diff --check` clean;
  `black --line-length 119` + `ruff check` clean on this round's files only (the parallel
  V2-envelope round's `build_overfit100_dataset.py` / `test_overfit100_gates.py` / archived
  failed-gates JSON were not touched).
- **Real pyconfig parse** of the yml with the launcher's full override set (resolved snapshot
  path, `model_manifest_path`, `expected_model_revision`, `dataset_verify_bytes`,
  `checkpoint_steps=[250,500,1000,2500]`): every key types correctly, pyconfig copies the
  snapshot path into the transformer key, and all three validators pass.

## Still unverifiable until the S1 smoke on TPU

1. `snapshot_download(local_files_only=True)` resolving the pinned revision on a real worker
   after the full-repo prefetch (the launcher's `MODEL_DIR` resolution), and that the pinned
   snapshot dir loads through `WanPipelineTI2V_2_2.from_pretrained` unchanged.
2. Wall-clock and memory cost of the streaming shard hash against real GCS objects
   (train100 ≈ 375 MB, per host), and `tf.io.gfile.stat().length` on `gs://`.
3. That the published `train100` / `train10` `_SUCCESS` + `summary.json` really carry
   `summary_sha256` and per-shard `sha256`/`md5`/`size` in the shapes assumed here — the fixture
   mirrors the builder's code, but the V2-envelope recalibration means no final publication
   exists yet. **This is the single highest-value S1 check.**
4. Whether the published `episodes.json` is index-contiguous with `used_text` populated for the
   final episode set, and that `expected_windows` matches the recalibrated build's counts
   (1629/167 may change if the cohort changes).
5. Real T5 encoding of the instructions (table content/timing), HBM cost of the replicated
   400 MiB table at per-device 4, `nnx.get_named_sharding` for `context_table` on a real
   multi-device mesh, and the jit compile of the gather inside the 5B graph.
6. `tf.debugging` assert behaviour inside the real multi-host `MultiHostDataLoadIterator`
   (it fires in the tf.data graph; the failure surface on a TPU worker is untested).
7. End-to-end resume with a rebuilt context table on TPU and the real ~30 GB/checkpoint write.
# Follow-up review: exp_02 overfit100 — cycle C strengthening
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-30

## Verdict
REQUEST-REVISION. The core C1–C4 resolutions and generation deviation are directionally sound, but several fail-closed edges remain that could admit wrong model, dataset, or context bytes before an expensive TPU run.

## Per-item
**a.** Manifest-authoritative pinned prefetch and local-only resolution are correct and precede model work; the two earlier pure config guards are harmless. Snapshot validation itself is not fully fail-closed—see F1.

**b.** The integrity chain is sound when byte verification is enabled. Accept omitting `generation`: promotion necessarily changes it, while hashing canonical bytes provides the stronger behaviorally relevant guarantee.

**c.** Full shared-index `(episode_index, episode_id, used_text)` compatibility is correct. Graph assertions are appropriate, but narrowing to `int32` before asserting leaves an overflow bypass; the context sidecar is also not content-bound.

**d.** Non-empty `checkpoint_steps` correctly suppresses cadence, and the log explicitly identifies the ignored value. This matches H2 and preserves final-save behavior already accepted in C1–C4.

**e.** `dataset_verify_bytes` should be unconditional for S1/S2/S3: disabling it admits the exact same-size mutation the regression demonstrates. The CRC-aware reserialization test is well designed and materially stronger than a raw bit flip.

**f.** The 6/6 mutation record and reported 787-passed/2-skipped suite are coherent; the five files contain 125 test functions plus parametrized cases. Pytest was unavailable for an independent rerun; F1–F4 remain budget-protection holes.

## Findings
1. **F1 — MAJOR — Snapshot paths are validated by optional substring matching.** At `wan_ti2v_overfit100_trainer.py:814-818`, an empty path is skipped and any path merely containing the SHA passes, contradicting the claimed exact path-component check for both fields. Require both paths to be non-empty and the normalized local snapshot path to contain an exact component equal to the revision; add empty-field and embedded-SHA spoof regressions.

2. **F2 — MAJOR — Canonical-byte integrity can be disabled.** At `wan_ti2v_overfit100_trainer.py:394-407` and `:839-842`, `dataset_verify_bytes=False` skips both SHA-256 and MD5, allowing a same-length replacement through preflight. Remove the knob or reject `False` for this trainer; the known ≤375 MB verification cost is negligible relative to S1/S2/S3.

3. **F3 — MAJOR — The context-table input is not fingerprint-bound.** `episodes.json` directly supplies training text at `wan_ti2v_overfit100_trainer.py:476-489` and `:925-928`, but cycle B records no hash for it and the preflight hashes only summary/shards; default `eval==train` performs no second-map comparison. Publish and verify an `episodes_sha256`, or bind its full mapping to the committed manifest after verifying `_SUCCESS.manifest_sha256`.

4. **F4 — MINOR — The range assertion occurs after lossy narrowing.** At `wan_ti2v_overfit100_trainer.py:617-634`, an `int64` value such as `2**32` casts to `int32(0)` and can pass both assertions. Assert bounds on the original `int64`, then cast under the control dependency; add an overflow regression.

---

# Strengthening record 2 (2026-07-30 — Claude Opus 5, Coder)

Follow-up review verdict **REQUEST-REVISION (F1–F4)**; all four **accepted and fixed**, tests
first. Suite: **812 passed / 2 skipped** (up from 787+2; +25 tests, all in the two files this
round touched).

## F1 — MAJOR, snapshot paths validated by optional substring matching — **FIXED**

Two holes, both closed in `_validate_pinned_snapshot`:

- **Empty fields were skipped.** `if not path: continue` meant
  `pretrained_model_name_or_path=''` sailed through the pin check entirely. Both fields are now
  **required to be non-empty** (pyconfig copies the pipeline path into the transformer key, so
  neither is legitimately empty at load time).
- **Substring matching.** `expected not in path` accepted `/x/prefix<sha>suffix/y`,
  `/cache/<sha>-backup`, `/cache/<sha>.old/model`. The check is now an **exact path component**:
  `expected in pathlib.PurePath(os.path.normpath(path)).parts`.

`normpath` first, so `.`/`..`/duplicate/trailing separators normalise away. What the launcher
actually passes is `snapshot_download()`'s return — the snapshot **root**
(`…/snapshots/<sha>`) — which passes; so do a trailing slash, `…/snapshots/<sha>/transformer`,
and a relative `cache/snapshots/<sha>`, all pinned by tests.

Tests: empty field refused (both keys, parametrized); five embedded-SHA spoofs refused; four
real layouts accepted; relative path accepted; one-pinned-one-spoofed refused naming the
transformer key.

## F2 — MAJOR, canonical-byte integrity could be disabled — **FIXED (knob deleted)**

`dataset_verify_bytes` is **gone** — from the yml, the launcher, and the function signature.
`verify_dataset_integrity(data_dir, expected_windows)` no longer takes a `verify_bytes`
parameter, so **no code path can skip** the sha256+md5 verification; the report's
`bytes_verified` field is now the constant `True` (kept so the startup log line is unchanged).
The yml keeps an explanatory NOTE in place of the key, stating why there is deliberately no
opt-out (~375 MB train100 / ~38 MB train10 read once is negligible against the S1/S2/S3 budget,
and the opt-out admitted exactly the same-length replacement the regression demonstrates).

Tests: `inspect.signature` has no `verify_bytes`; the yml contains no such key; the launcher
contains neither `DATASET_VERIFY_BYTES` nor `dataset_verify_bytes`; a mutated shard is refused
with no way to wave it through; the report always says `bytes_verified=True`.

## F3 — MAJOR, the context-table input was not fingerprint-bound — **FIXED (manifest binding)**

Took the stronger of the two offered options, and the one that needs **no new cycle-B field**
(publishing `episodes_sha256` would require rebuilding the dataset):

1. `_SUCCESS.manifest_sha256` is the sha256 of the exact manifest bytes cycle B consumed
   (`assert_manifest_matches_committed`). The trainer verifies it against the manifest **this
   run was handed** — refusing a non-64-hex value, so a `--dry-run` build's
   `"unverified (--dry-run)"` can never gate a real run.
2. With the manifest thereby authenticated, its `episodes[]` becomes the reference for the
   table's text: every `episodes.json` entry must match the manifest's entry at the same index
   on **both** `episode_id` and `used_text`, every index must exist in the manifest, and the
   table-source set must be index-contiguous `0..N-1`.

`model_manifest_path` is now **mandatory** (`_validate_overfit100_config`): the manifest is
load-bearing for the *text*, not only the revision. New helpers: `read_manifest_episodes`,
`assert_episodes_bound_to_manifest(..., require_contiguous=True)`, wired into
`_preflight_dataset` for the train dir and (with `require_contiguous=False`) for a distinct
eval dir.

> **On the eval == train skip.** The reviewer noted the C3 map check is skipped when the dirs
> match. After F3 that is safe *and* the skip is now the uninteresting case: the training
> mapping itself is authenticated against the committed manifest, so there is no unverified
> mapping left to compare against. Two sets bound to the *same* manifest cannot disagree with
> each other — which my own test caught: the "incompatible eval mapping" case now fails on the
> **manifest** binding (a strictly stronger statement: that set was not built from the reviewed
> manifest at all) rather than on the pairwise comparison. `assert_context_map_compatible`
> stays as cheap defence-in-depth and keeps its direct unit tests.

> **One over-strictness found by my own test.** Requiring index-contiguity of *every* set is
> wrong: a separate eval set may legitimately be a **sparse** subset of the training episodes
> (e.g. one canonical window per selected episode). `require_contiguous` therefore applies to
> the table-source set only; a sparse eval set is still manifest-bound per index and still
> range-checked against `num_text_slots` in the parse path. Both directions are tested.

Tests: manifest-bound publication verifies; manifest-sha mismatch refused; **tampered
`used_text` refused** (naming the index); tampered `episode_id` refused; index gap refused;
index absent from the manifest refused; `train10` as a 10-of-100 contiguous prefix accepted;
sparse eval subset accepted only with `require_contiguous=False`; non-sha marker hash refused;
missing manifest path refused; `model_manifest_path` mandatory; `start_training` refuses
tampered context text with the pipeline loader booby-trapped; the committed manifest still has
the fields the binding needs (100 episodes, contiguous, all three fields).

## F4 — MINOR, range assertion after lossy narrowing — **FIXED**

The bounds are now asserted on the **raw int64 feature**, and the `int32` narrowing happens
*inside* the `control_dependencies` block. There is no longer any `tf.cast(features["episode_
index"] …)` before the asserts; the only pre-assert cast path is the `num_slots <= 0`
branch (unreachable in an exp_02 run, since `num_text_slots > 0` is a config gate) and it is
documented as such.

Tests: `2**32`, `2**32 + 5` and `2**31` all refused with a non-vacuity assertion that each
really does narrow into the accepted range; `-(2**32)` refused; a structural test pinning that
both assertions consume `raw_episode_index`, that both bounds are `tf.int64` constants, and
that no assert reads a narrowed value.

## Verification

- **Red evidence.** 19 failures before implementation (15 in `test_overfit100_preflight.py`,
  4 in `test_overfit100_trainer_data.py`).
- **Mutation spot-checks** (each applied to the real module, then restored byte-identical):
  substring match restored → 2 failures; empty-path skip restored → 1; `verify_bytes` opt-out
  reintroduced → 1; `manifest_sha256` check dropped → 1; the `(episode_id, used_text)` triple
  check dropped → 3; assert moved back after the `int32` narrowing → 4. **All caught.**
- **Static.** `py_compile`, `yaml.safe_load` (197 keys, `dataset_verify_bytes` absent),
  `bash -n`, `git diff --check` clean; `black --line-length 119` + `ruff check` clean on the
  three files touched.
- **Real pyconfig parse** with the launcher's override set: no `dataset_verify_bytes` key
  exists, the mandatory manifest path resolves, all three validators pass, and
  `read_manifest_episodes` reads all 100 episodes off the committed manifest.

## Unchanged from strengthening record 1

The cannot-validate-until-S1 list stands, with item 3 now **strictly stronger**: the S1 smoke
must confirm the published `_SUCCESS` carries a real 64-hex `manifest_sha256` equal to
`sha256(overfit100_manifest.json)` **and** that the published `episodes.json` matches the
manifest triple-for-triple — if the recalibrated build consumes an amended manifest, the
trainer will (correctly) refuse until `model_manifest_path` points at that same amended file.
Also still open for the Planner: the parent trainer's `checkpoint_keep_period=-1` quirk
(judgment 10) as a separate shared-trainer fix.
# Third-pass review: exp_02 overfit100 — cycle C close-out
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-30

## Verdict

APPROVE. Commit a08051e resolves F1–F4 as prescribed; both scoped behavior deltas are sound for cycle D, with no launch-blocking issue remaining.

## Per-finding

F1: RESOLVED — both snapshot paths must be non-empty, and normalized exact-component matching rejects embedded-SHA spoofs.

F2: RESOLVED — the knob is absent from the YAML, launcher, and function signature; shard-byte verification is unconditional.

F3: RESOLVED — the marker hash must be 64-hex and equal the supplied manifest’s SHA-256; every sidecar triple is manifest-bound, and `model_manifest_path` is mandatory.

F4: RESOLVED — bounds are asserted on raw `int64` values before the dependent `int32` cast; `2^32` and related overflow cases are refused.

## Delta judgments

D-a: AGREE — contiguity belongs to the table-source set; sparse eval subsets remain safe through manifest binding, training-map membership, and parse-path range checks.

D-b: AGREE — `eval==train` uses the authenticated table-source mapping itself, so a pairwise comparison adds nothing; the retained distinct-set check remains useful for subset membership and defence-in-depth.

## Findings

none
