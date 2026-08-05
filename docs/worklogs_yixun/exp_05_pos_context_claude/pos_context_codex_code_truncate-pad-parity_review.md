# pos_context — Codex code review: round S1 `truncate-pad-parity`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_05 worktree). Working tree at HEAD `177ad95`. Reviewer independently verified all three S1 findings with its own measurements (idiom identity with scope note; bf16 cast — 134 elements changed, max 0.03125; frame_positions drop — 0.37477) and ran probes incl. a B=2 distinct-context parity check that exposed the fixture's batch-collapse blindness.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md`, including briefing, TDD/review/strengthen, validation, and announcement requirements.
- `pos_context_yixun_query.md`.
- `plan_pos_context.md` v3, especially §§3, 5.1, 6, and the deviations register.
- `pos_context_worklog.md` through S1 and its Planner acceptances.
- MAIN checkout `embedding_search.py:1150-1195` at submodule pin `f370228`.
- `side_adapter_wan.py:537-774` and `transformer_wan.py:621-646`.
- exp_04 R1 `embed_null_tokens`, its validation tests, shared bit helpers, review/strengthening record, and mutation-battery conventions.
- Both uncommitted S1 files at HEAD `177ad95`.

1. **MAJOR** — [test_pos_context_truncate_pad.py:217](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_truncate_pad.py:217), [test_pos_context_truncate_pad.py:297](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_truncate_pad.py:297) — The load-bearing velocity-parity fixture uses `B=1` and broadcasts one identical context. It therefore cannot detect collapsing, broadcasting, or permuting per-example contexts. An independent `B=2` probe with distinct contexts preserved correct bitwise parity, while a `broadcast(C[:1])` mutant changed example 1 by `1.86618`; that mutant survives the present fixture. Change the parity/cast tests to use at least two distinct per-example contexts and include the batch-collapse mutant guard.

2. **MINOR** — [pos_context_inversion_wan.py:64](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/models/wan/pos_context_inversion_wan.py:64), [test_pos_context_truncate_pad.py:151](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_truncate_pad.py:151) — `truncate_or_pad_context` documents rank-3 inputs as either unit-leading T5 or genuinely batched `B>1`, but accepts `B=0` for truncate, pad, and equal-length paths. Reject an empty leading batch axis and add the corresponding malformed-shape case.

Verification verdicts on the three S1 findings:

1. **Identical call idioms — VERIFIED, with scope.** At `frame_positions=None`, deployed final rerun and the raw replay velocity seam are the same call: omitted `deterministic`/`rngs` resolve to `True`/`None`. The focused suite passed all 33 tests. Because S1 characterizes the seam rather than the not-yet-written replay operator, S3 must still test the actual replay implementation against this fixture.

2. **bf16 activation cast — VERIFIED.** Deployed output was bitwise equal to the explicitly bf16-cast seam; fp32 context changed 134 output elements with max `|Δ| = 0.03125`. The requirement appears in the worklog, production-module briefing, and discriminating test, which is sufficient for S3 discoverability; S3 must bind it to the real replay code.

3. **`frame_positions` drop — VERIFIED.** With `(1,0)`, deployed output was bitwise identical to a direct call using `frame_positions=None`; forwarding `(1,0)` changed all 192 tested elements, max `|Δ| = 0.37477`. It is harmless for the declared exp_05 scope, whose call sites use `None`.

Reference construction parity otherwise passed: non-contiguous truncate/pad fixtures matched PyTorch bitwise, outputs materialized contiguously where the reference calls `.contiguous()`, arbitrary `D` worked, bf16 slice bits were preserved, typed zero-padding was exact, and equal-length JAX input returned the same array object. The Planner-accepted deviations are reasonable. Ruff, AST compilation, and whitespace checks passed; Black’s only disagreement was the already-ratified exp_04 import-spacing convention.

Final verdict: **REQUEST-REVISION** — construction semantics are reference-faithful, but the deployment-parity guarantee needs distinct batched contexts and the extra batched API should fail closed on `B=0`.

Status:

- No subprocesses are running; S1 review is complete.
- Earliest steering point: now, before strengthening and committing S1.

---

## Strengthening record (Coder, same round; both findings closed, none rejected)

1. **MAJOR (B=1 parity blindness) — FIXED.** Parity fixtures at B=2 with distinct per-example contexts (degeneracy-guarded); example-i-vs-singleton bitwise batch-collapse test; the corrupted-context guard parametrized over {reproject, collapse, permute} each requiring >1e-3 movement; the S3 scope note (re-run this fixture against the real replay operator's v_cond) restated in the file and test docstrings with the review citation.
2. **MINOR (B=0) — FIXED.** Empty leading batch axis rejected on all three paths, matched message, three cases.

Suite **649 passed** (610 inherited + 39); **16 mutants, 0 survivors** — M13 (the reviewer's broadcast(C[:1]) mutant, previously surviving) now killed; M16 (module-level collapse analogue) added by the Coder. Behavior beyond findings: none. Round S1 closed; committed with this record.
