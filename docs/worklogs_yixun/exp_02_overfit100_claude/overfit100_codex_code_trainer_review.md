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

*(Strengthening record appended at round close.)*
