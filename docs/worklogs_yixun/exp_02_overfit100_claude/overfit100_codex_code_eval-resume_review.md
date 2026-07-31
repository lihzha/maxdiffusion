# Codex code review — eval per-window resume round

- **Date:** 2026-07-31
- **Commit under review:** `78819dc` (eval per-window resume — preemption-tolerant rollout staging)
- **Reviewer:** Codex `gpt-5.6-sol`, reasoning effort xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 2 BLOCKER (envelope under-binds run identity; admission type-coercion holes), 2 MAJOR (D4 ordering vs staging; conditional byte-parity overstated), 1 MINOR (staging enumeration/error message)
- **Reviewer's operational advice (adopted):** keep pre-resume queue retries alive; strengthen first; cancel-and-relaunch only at the corrected SHA.

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

1. **BLOCKER — The envelope does not identify the exact row-producing run.** It binds only checkpoint step, role, manifest hash, and eval commit ([generate_wan_side_adapter.py:1540](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1540)); it omits `checkpoint_dir`/checkpoint identity, train/eval dataset identity, guide scale, shuffle seed/derangement, `eval_aux_rgb`, and `write_videos`. A retry at the same SHA/step/role can therefore admit rows from another checkpoint or configuration and then report the current configuration in the final artifact ([generate_wan_side_adapter.py:1833](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1833), [generate_wan_side_adapter.py:2090](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2090)); switching `write_videos=False→True` also skips videos for resumed rows. The `COMMIT="unknown"` fallback further allows cross-commit admission when provenance relay is missing ([generate_wan_side_adapter.py:1549](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1549)). **Fix:** bind a canonical, strictly typed run signature covering checkpoint path/identity, dataset fingerprints, sampler/guidance, derangement, aux/video settings and model provenance; require a valid 40-hex commit whenever staging is enabled; mutation-test every bound input.

2. **BLOCKER — Admission uses coercion and incomplete row validation, so malformed rows can reach the verdict artifact.** Envelope checkpoint `"2500"` is accepted as `2500`; row seed `0.9`, `False`, or `"0"` is accepted as seed 0; and `episode_index`, `canonical`, `context_source_episode_index`, metric types, and metric finiteness are never checked against the selected window/current mode ([generate_wan_side_adapter.py:1679](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1679), [generate_wan_side_adapter.py:1691](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1691), [generate_wan_side_adapter.py:1701](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1701)). The aggregation builder only checks required-key presence, so even a string SSIM can be emitted and later numerically coerced by the statistic ([generate_wan_side_adapter.py:1805](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1805)). **Fix:** require exact JSON types—using `type(x) is int` to exclude booleans—and exact envelope values; compare every identity field against the full window descriptor and expected context source; validate all metric/aux value domains before admission; add string/float/bool mutation cases.

3. **MAJOR — D4 runs after staging interaction and permits partial mutation of a completed role directory.** Existing staging is read at [generate_wan_side_adapter.py:2054](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2054), changed rows are written at [generate_wan_side_adapter.py:2135](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2135), and only then can the aggregation guard refuse replacement at [generate_wan_side_adapter.py:2168](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2168). The new changed-rerun test asserts only that `aggregation.json` stayed unchanged; it does not detect that staging was repopulated with the changed rows, and its “new commit” comment is inaccurate because it never changes `COMMIT` ([test_overfit100_context_modes.py:1500](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:1500)). **Fix:** when any completed artifact exists, disable all staging reads/writes and use the original recompute-in-memory/immutable-compare D4 path; snapshot or booby-trap the entire role directory in the test.

4. **MAJOR — Whole-artifact byte parity is conditional, but the test presents it as unconditional.** Production defaults `eval_aux_rgb=True`, while auxiliary values and `aux_status` legitimately vary with network/tool failures and may embed attempt-specific temporary paths ([generate_wan_side_adapter.py:1401](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1401), [generate_wan_side_adapter.py:1444](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1444)); the parity fixture disables aux ([test_overfit100_context_modes.py:917](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:917)). Thus a resumed artifact can differ from a fresh straight-through attempt even with identical primary rollout values. **Fix:** either narrow the documented guarantee to deterministic primary/statistic fields, or make the auxiliary contract deterministic enough for whole-file parity; in both cases test aux-enabled success and failure/resume paths.

5. **MINOR — “Every file under staging_rows” is not validated.** The single three-level `*.json` glob ignores corrupt JSON at another depth, unexpected extensions, and possible temporary objects ([generate_wan_side_adapter.py:1647](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1647)). The error also names only the first bad file and says “clear it,” which is tedious when every row is foreign after a code upgrade ([generate_wan_side_adapter.py:1627](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1627)). **Fix:** recursively enumerate files, allow directory markers but reject every nonconforming object, and name the exact `<step_root>/staging_rows` root the operator should deliberately clear.

Verified claims:

- **R-A partial:** zero-byte/truncated JSON at the expected path hard-fails; unchanged W-at-V placement, out-of-coverage rows, and duplicate tuple aliases are refused ([generate_wan_side_adapter.py:1667](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1667), [generate_wan_side_adapter.py:1691](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1691), [generate_wan_side_adapter.py:1712](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1712)).

- **R-B:** for valid deterministic rows, final order remains exactly windows→modes→seeds in both whole-window and partial resume paths ([generate_wan_side_adapter.py:2070](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2070), [generate_wan_side_adapter.py:2084](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2084)); Python JSON round-trip preserves the original binary64 float, so reserialization itself does not change float text.

- **R-C partial:** `aggregation.json`, `summary.csv`, and `summary.json` still use the prior immutable writers; only the new staging interaction precedes those guards ([generate_wan_side_adapter.py:2168](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2168)).

- **R-D:** `process_count()!=1` sets `resume_on=False`; both the only read and only write are guarded by that value, while all hosts retain the unchanged rollout/decode sequence ([generate_wan_side_adapter.py:1577](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1577), [generate_wan_side_adapter.py:2054](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2054), [generate_wan_side_adapter.py:2135](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2135)).

- **R-E:** a fully staged window exits before GT decode; a partially staged window performs one GT decode before the mode/seed loops, and every missing tuple uses that window’s `gt0` ([generate_wan_side_adapter.py:2072](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2072), [generate_wan_side_adapter.py:2081](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2081)).

- **R-F:** empty pre-resume role directories/markers yield no matched staged files and safely restart from zero; with no environment override, the committed config enables resume. A newer SHA hard-fails only when `COMMIT` is correctly relayed, subject to finding 1.

- **R-G:** the YAML parses to 206 keys with `overfit100_eval_resume=True`; pyconfig makes the key CLI-overridable, and the environment accepts exact `"0"`/`"1"` with environment precedence ([base_wan_5b_overfit100.yml:356](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/configs/base_wan_5b_overfit100.yml:356), [generate_wan_side_adapter.py:1568](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1568), [pyconfig.py:124](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/pyconfig.py:124)).

- **R-H:** commit `78819dc` does not modify `overfit100_success_statistic.py`; the shared aggregation schema remains v2, role validation remains before rollout, and no aggregation fields were added ([overfit100_success_statistic.py:79](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/overfit100_success_statistic.py:79), [generate_wan_side_adapter.py:1829](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1829), [generate_wan_side_adapter.py:1992](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1992)).

Risk read: do **not** cancel the existing jobs in favor of relaunching at `78819dc`. The current empty role directories are operationally safe, and an unchanged, correctly stamped relaunch would probably work, but the SHA does not meet the requested fail-closed BLOCKER bar. Keep the pre-resume retries alive while producing a small strengthening commit, then cancel-and-relaunch at the corrected SHA rather than spending further retries on jobs that need 85–144 uninterrupted minutes.
tokens used
170,468
REQUEST-REVISION

1. **BLOCKER — The envelope does not identify the exact row-producing run.** It binds only checkpoint step, role, manifest hash, and eval commit ([generate_wan_side_adapter.py:1540](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1540)); it omits `checkpoint_dir`/checkpoint identity, train/eval dataset identity, guide scale, shuffle seed/derangement, `eval_aux_rgb`, and `write_videos`. A retry at the same SHA/step/role can therefore admit rows from another checkpoint or configuration and then report the current configuration in the final artifact ([generate_wan_side_adapter.py:1833](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1833), [generate_wan_side_adapter.py:2090](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2090)); switching `write_videos=False→True` also skips videos for resumed rows. The `COMMIT="unknown"` fallback further allows cross-commit admission when provenance relay is missing ([generate_wan_side_adapter.py:1549](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1549)). **Fix:** bind a canonical, strictly typed run signature covering checkpoint path/identity, dataset fingerprints, sampler/guidance, derangement, aux/video settings and model provenance; require a valid 40-hex commit whenever staging is enabled; mutation-test every bound input.

2. **BLOCKER — Admission uses coercion and incomplete row validation, so malformed rows can reach the verdict artifact.** Envelope checkpoint `"2500"` is accepted as `2500`; row seed `0.9`, `False`, or `"0"` is accepted as seed 0; and `episode_index`, `canonical`, `context_source_episode_index`, metric types, and metric finiteness are never checked against the selected window/current mode ([generate_wan_side_adapter.py:1679](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1679), [generate_wan_side_adapter.py:1691](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1691), [generate_wan_side_adapter.py:1701](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1701)). The aggregation builder only checks required-key presence, so even a string SSIM can be emitted and later numerically coerced by the statistic ([generate_wan_side_adapter.py:1805](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1805)). **Fix:** require exact JSON types—using `type(x) is int` to exclude booleans—and exact envelope values; compare every identity field against the full window descriptor and expected context source; validate all metric/aux value domains before admission; add string/float/bool mutation cases.

3. **MAJOR — D4 runs after staging interaction and permits partial mutation of a completed role directory.** Existing staging is read at [generate_wan_side_adapter.py:2054](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2054), changed rows are written at [generate_wan_side_adapter.py:2135](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2135), and only then can the aggregation guard refuse replacement at [generate_wan_side_adapter.py:2168](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2168). The new changed-rerun test asserts only that `aggregation.json` stayed unchanged; it does not detect that staging was repopulated with the changed rows, and its “new commit” comment is inaccurate because it never changes `COMMIT` ([test_overfit100_context_modes.py:1500](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:1500)). **Fix:** when any completed artifact exists, disable all staging reads/writes and use the original recompute-in-memory/immutable-compare D4 path; snapshot or booby-trap the entire role directory in the test.

4. **MAJOR — Whole-artifact byte parity is conditional, but the test presents it as unconditional.** Production defaults `eval_aux_rgb=True`, while auxiliary values and `aux_status` legitimately vary with network/tool failures and may embed attempt-specific temporary paths ([generate_wan_side_adapter.py:1401](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1401), [generate_wan_side_adapter.py:1444](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1444)); the parity fixture disables aux ([test_overfit100_context_modes.py:917](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/tests/worklogs_yixun/test_overfit100_context_modes.py:917)). Thus a resumed artifact can differ from a fresh straight-through attempt even with identical primary rollout values. **Fix:** either narrow the documented guarantee to deterministic primary/statistic fields, or make the auxiliary contract deterministic enough for whole-file parity; in both cases test aux-enabled success and failure/resume paths.

5. **MINOR — “Every file under staging_rows” is not validated.** The single three-level `*.json` glob ignores corrupt JSON at another depth, unexpected extensions, and possible temporary objects ([generate_wan_side_adapter.py:1647](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1647)). The error also names only the first bad file and says “clear it,” which is tedious when every row is foreign after a code upgrade ([generate_wan_side_adapter.py:1627](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1627)). **Fix:** recursively enumerate files, allow directory markers but reject every nonconforming object, and name the exact `<step_root>/staging_rows` root the operator should deliberately clear.

Verified claims:

- **R-A partial:** zero-byte/truncated JSON at the expected path hard-fails; unchanged W-at-V placement, out-of-coverage rows, and duplicate tuple aliases are refused ([generate_wan_side_adapter.py:1667](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1667), [generate_wan_side_adapter.py:1691](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1691), [generate_wan_side_adapter.py:1712](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1712)).

- **R-B:** for valid deterministic rows, final order remains exactly windows→modes→seeds in both whole-window and partial resume paths ([generate_wan_side_adapter.py:2070](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2070), [generate_wan_side_adapter.py:2084](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2084)); Python JSON round-trip preserves the original binary64 float, so reserialization itself does not change float text.

- **R-C partial:** `aggregation.json`, `summary.csv`, and `summary.json` still use the prior immutable writers; only the new staging interaction precedes those guards ([generate_wan_side_adapter.py:2168](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2168)).

- **R-D:** `process_count()!=1` sets `resume_on=False`; both the only read and only write are guarded by that value, while all hosts retain the unchanged rollout/decode sequence ([generate_wan_side_adapter.py:1577](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1577), [generate_wan_side_adapter.py:2054](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2054), [generate_wan_side_adapter.py:2135](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2135)).

- **R-E:** a fully staged window exits before GT decode; a partially staged window performs one GT decode before the mode/seed loops, and every missing tuple uses that window’s `gt0` ([generate_wan_side_adapter.py:2072](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2072), [generate_wan_side_adapter.py:2081](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:2081)).

- **R-F:** empty pre-resume role directories/markers yield no matched staged files and safely restart from zero; with no environment override, the committed config enables resume. A newer SHA hard-fails only when `COMMIT` is correctly relayed, subject to finding 1.

- **R-G:** the YAML parses to 206 keys with `overfit100_eval_resume=True`; pyconfig makes the key CLI-overridable, and the environment accepts exact `"0"`/`"1"` with environment precedence ([base_wan_5b_overfit100.yml:356](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/configs/base_wan_5b_overfit100.yml:356), [generate_wan_side_adapter.py:1568](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1568), [pyconfig.py:124](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/pyconfig.py:124)).

- **R-H:** commit `78819dc` does not modify `overfit100_success_statistic.py`; the shared aggregation schema remains v2, role validation remains before rollout, and no aggregation fields were added ([overfit100_success_statistic.py:79](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/overfit100_success_statistic.py:79), [generate_wan_side_adapter.py:1829](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1829), [generate_wan_side_adapter.py:1992](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100/src/maxdiffusion/generate_wan_side_adapter.py:1992)).

Risk read: do **not** cancel the existing jobs in favor of relaunching at `78819dc`. The current empty role directories are operationally safe, and an unchanged, correctly stamped relaunch would probably work, but the SHA does not meet the requested fail-closed BLOCKER bar. Keep the pre-resume retries alive while producing a small strengthening commit, then cancel-and-relaunch at the corrected SHA rather than spending further retries on jobs that need 85–144 uninterrupted minutes.
```

---

## Strengthening record (2026-07-31 — Claude Opus 5, Coder)

All five findings **accepted and fixed**, tests first (92 failures before implementation). Suite:
**1131 passed / 2 skipped** (from 1057+2; +74 tests). The verified claims R-B/R-D/R-E/R-H are
preserved by construction and re-pinned: row order is still windows→modes→seeds in both resume
paths, the multi-host gate still guards the only read and the only write,
`overfit100_success_statistic.py` is untouched, and the aggregation schema is unchanged.

### 1 (BLOCKER) — the envelope did not identify the row-producing run

The four-field binding is replaced by a canonical, strictly typed **run signature**
(`OVERFIT100_RUN_SIGNATURE_SCHEMA`, built by `overfit100_run_signature`) covering all 25 inputs the
review named: checkpoint identity (`checkpoint_step`, `checkpoint_dir`), pass role, dataset identity
(`train_data_dir`/`eval_data_dir` **plus** each set's `_SUCCESS.summary_sha256`, the fingerprint the
cycle-C preflight already verifies), model provenance (`model_snapshot`, `model_revision`,
`manifest_sha256`), sampler/guidance (`sampling_steps`, `guide_scale`, `flow_shift`, both dtypes),
context identity (`context_shuffle_seed`, `context_derangement_sha256`, `num_text_slots`), coverage
(`eval_windows_spec`, `num_windows`, `rollout_seeds`, `context_modes`), the `eval_aux_rgb` /
`write_videos` behaviour flags, and `code_commit`. The envelope stores the signature and its
`run_signature_sha256`; admission compares **field by field** (so the error names the field) and
re-derives the hash, so a tampered dict/hash pair is refused.

`resume_state` now **requires a 40-hex COMMIT**: an `"unknown"` provenance hard-disables staging,
reads *and* writes, with a logged reason naming the relay command. Staging is therefore never
written under, nor admitted under, an unidentifiable code state.

**Tests:** `test_run_signature_is_built_from_the_run_and_strictly_typed`;
`test_every_bound_signature_field_is_enforced` — **one parametrized case per bound field (25)**;
`test_a_tampered_signature_hash_is_refused`;
`test_staging_requires_a_real_commit_and_hard_disables_on_unknown`;
`test_driver_disables_staging_entirely_without_a_valid_commit`.

### 2 (BLOCKER) — admission coercion holes

Exact JSON typing everywhere: signature fields are checked with `type(value) is kind` (list members
too), and rows against a per-field type table where `bool` is its own type, so a bool can never
satisfy an int. Envelope values are compared **as-is** — `"2500"` is not `2500`, `2500.0` is not
`2500`. Row identity is validated against the **full window descriptor** (`episode_id`,
`episode_index`, `window_start`, `canonical`) and `context_source_episode_index` against the value
this `(mode, derangement)` implies, so a shuffled row claiming its own context row is refused.
Metric domains are checked before admission: every metric finite, SSIM-like fields within
`[-1, 1]`, squared errors and standard deviations non-negative.

**Tests:** `test_signature_values_are_type_exact_not_coerced` (8 cases),
`test_row_fields_are_type_exact_not_coerced` (15), `test_metric_domains_are_validated_before_admission`
(10), `test_row_identity_is_checked_against_the_full_window_descriptor` (4),
`test_context_source_index_must_match_the_mode_and_derangement`,
`test_a_bool_cannot_masquerade_as_an_int_in_the_signature`.

> **Scope note.** The review also observed that `overfit100_aggregation_artifact` checks only key
> presence. That builder is settled shared code (R-H), and rollout-produced rows are constructed
> in-process; the only untrusted row source is staging, which is now strictly validated at
> admission. Left unchanged deliberately to keep the diff surgical — flagged here rather than
> silently skipped.

### 3 (MAJOR) — D4 now runs first

`overfit100_completed_artifacts(step_root)` is consulted **before any staging interaction**: if
`aggregation.json`, `summary.json` or `summary.csv` exists, resume is switched off for the whole
pass (no reads, no writes) with a logged reason, and the run takes the original
recompute-in-memory + immutable-compare path. A completed role directory can therefore never be
partially mutated by a rerun.

**Tests:** `test_a_completed_pass_reruns_idempotently_through_staging` and
`test_the_immutability_guard_still_blocks_a_changed_rerun_without_touching_the_role_dir` both
snapshot the **entire role directory including `staging_rows`** (`_dir_snapshot`) and assert
byte-equality after the rerun; the inaccurate "new commit" comment is gone (the test never changed
`COMMIT`, and no longer pretends to). `test_a_partial_role_dir_still_resumes` pins that a *preempted*
directory — staging present, no aggregation.json — still resumes, which is the case resume exists for.

### 4 (MAJOR) — parity claim narrowed and aux-tested

The production comment and the test-section banner now state the guarantee precisely: byte parity
covers the **deterministic primary/statistic** fields; the auxiliary block depends on gsutil/ffmpeg
and the network and can legitimately differ between attempts. Admission is by **tuple identity, not
content**, so a row staged while aux was failing keeps its recorded failure — deliberate, because
aux never feeds the success statistic, `aux_status` + `aux_coverage` make the gap explicit, and the
VAE ceiling is recoverable independently (the S2-ceiling-backfill path).

**Tests:** `test_parity_holds_with_aux_enabled_when_both_attempts_succeed` (whole-file parity with
aux ON) and `test_a_staged_aux_failure_is_admitted_as_is_even_when_a_fresh_attempt_would_succeed`
(the staged failure persists, the recomputed tuple gets its ceiling, and `aux_coverage` reports
1 ok / 1 failed).

### 5 (MINOR) — staging enumeration

`_enumerate_staging_files` walks the staging root recursively (`tf.io.gfile.walk`), tolerates
directories and directory markers, and treats **every** nonconforming object as fatal — wrong depth,
wrong extension, temp files, malformed `seed_<n>`. Offenders are collected and reported **together**
(count + first five + "(+N more)"), and every staging error names the exact
`<step_root>/staging_rows` root to clear, plus the `OVERFIT100_EVAL_RESUME=0` escape.

**Tests:** `test_any_nonconforming_object_under_staging_is_refused` (6 shapes),
`test_enumeration_error_lists_every_offender_with_a_count`,
`test_empty_directories_under_staging_are_tolerated`.

### Test-fixture corrections found while doing this

Two driver-fixture faults surfaced and were fixed, both making the fixture more faithful to a real
run: the synthetic eval dir now carries a `_SUCCESS` marker (a real set always does — the preflight
requires it, and the signature binds its `summary_sha256`), and `_frame_ssim` is stubbed to a finite
in-range value because this venv has no scikit-image and would otherwise stage `NaN` SSIM — which
admission now correctly refuses. The with-video manifest also moved to its own filename; previously a
second `_driver_env` call overwrote it, which would have made the aux parity test pass for the wrong
reason.

### Verification

* **Red evidence:** 92 failed / 41 passed before implementation.
* **Mutation spot-checks, 11/11 caught** (each applied to the real module, then restored
  byte-identically): signature field ignored → 1; COMMIT gate removed → 4; signature hash unverified
  → 1; signature typing relaxed to `isinstance` → 1; row typing relaxed → 1; SSIM domain unchecked
  → 6; identity-vs-descriptor unchecked → 4; context-source unchecked → 1; D4-first removed → 2;
  enumeration back to the 3-level glob → 7. The `isinstance` mutant initially **survived** — every
  existing case was still caught by value — so `test_a_bool_cannot_masquerade_as_an_int_in_the_signature`
  was added (`True == 1`, the only case where exact typing is load-bearing) before this record.
* **Static:** black (`--line-length 119 --target-version py312`), ruff, `py_compile`,
  `git diff --check` clean. `overfit100_success_statistic.py` untouched; no aggregation-schema change.
