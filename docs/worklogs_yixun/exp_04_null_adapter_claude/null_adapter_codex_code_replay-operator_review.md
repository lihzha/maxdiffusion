# null_adapter — Codex code review: round R4a `replay-operator`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `37504a8`. Reviewer ran the R4a tests (16/16) and independent probes (zero-timestep/zero-latent/bad-splice mutations killed; bitwise batched-vs-singleton replay; A0 through w=50).

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md` — reviewer briefing, TDD, review/strengthen cycle, and validation ladder.
- `plan_null_adapter.md` v5 — §3 replay/A0/batching contracts; §5 items 1/12; §6 including the R4a/R4b/R4c amendment; §8 parity audit.
- `null_adapter_worklog.md` — R4a entry and all four Planner decisions.
- R1–R3 Codex reviews and strengthening records, including R3’s oracle-blindness findings.
- HEAD `37504a8`; complete implementation diff and full `test_null_adapter_replay.py`.
- Pinned Wan2.2 reference `f370228`: `embedding_search.py:791-819` and `verify_reconstruction_from_null.py`.
- Shared timestep/pinning/sigma helpers in `side_adapter_wan.py`.
- Validation: R4a **16/16 passed**. Full suite: **111 passed**, with only the existing tiny-Wan smoke failing because the read-only sandbox cannot create a temporary directory.
- Independent probes: zero-timestep, zero-latent, and bad-splice mutations were killed; batched and singleton replay were bitwise equal; A0 remained within `rtol=1e-4` through `w=50`.

Findings:

1. **MINOR — malformed-null tests do not pin the batch-axis guard.** [test_null_adapter_replay.py:238](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_replay.py:238), [null_inversion_wan.py:443](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:443)  
   The implementation correctly rejects rank-4 nulls whose batch differs from `z_start`, but no test exercises this guard. I removed only `nulls.shape[1] != batch` in memory and all 16 tests still passed. A `[N,1,L,D]` tensor at `B=2` can then broadcast through a batch-independent velocity and silently apply one example’s nulls to both examples. Add a matched rejection case such as `nulls[:, :1]` at `B=2`; preferably also cover an oversized `L`.

2. **MINOR — replay’s batch-composition invariant is not directly asserted.** [test_null_adapter_replay.py:227](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_replay.py:227)  
   The literal-loop oracle strongly tests layout within one `B=2` call, but every test still executes replay at the same batch size. Add a batched-versus-per-example-singleton comparison for the full trajectory using distinct `[N,B,L,D]` nulls and a tight numerical tolerance. This pins the `B=1` artifact-verification path as well as composition independence.

No operator defect found: fresh per-step `v_cond`, CFG ordering, timestep construction, negative `dsigma`, start/per-step pins, scan stacking, and fp32 arithmetic match the reference. The A0 `rtol=1e-4` assertion is justified and does not mask guidance being ignored because the separate analytic CFG test pins the formula and the `>1.0` contrast pins input sensitivity. Accepting `[N,L,D]` is sound for shared/per-record nulls and is explicitly distinguished from `[N,B,L,D]`.

B-independence ruling: **required**. The current implementation passes my bitwise batched-versus-singleton probe, so this is test strengthening, not a code fix.

Final verdict: **REQUEST-REVISION** — the replay operator is faithful, but its batch guard and batch-composition contract need the two focused regression tests above before R4a closes.

---

## Strengthening record (Coder, same round; both findings closed, none rejected)

1. **MINOR (unpinned batch guard) — FIXED.** The clause was hidden in an `or`; split into three independently-pinned guards with distinct matched messages (batch mismatch / oversized L / feature-dim), with a comment recording the silent-broadcast hazard. Reviewer's exact probe (guard deletion) re-run: previously 16-passed, now killed.
2. **MINOR + ruling (B-independence) — ADDED.** Batched-vs-per-example-singleton composition test, distinct per-example nulls, full trajectories, **bitwise equality** on final latent and every step (no tolerance needed, matching the reviewer's probe); docstring records this as the exact B=1 path R4c's artifact verification takes. Effect-mutant S3 (per-example nulls indexed at example 0) killed by this test and the literal loop.

Suite **115 passed in 14.68s** (+3); mutants S1–S3 killed; ruff/py_compile/diff-check clean. Behavior changes beyond the findings: none. Round R4a closed; committed with this record.
