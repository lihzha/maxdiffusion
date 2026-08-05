# null_adapter — Codex code review: round R11 `a3-direct-opt`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). HEAD `b027f22`. Reviewer independently confirmed direct_rollout ≡ replay_with_nulls bitwise on current code, then demonstrated the reversed-dsigma mutant surviving all 40 tests (the FD test differentiates the same wrong forward); probed the zero-operand jit (85 embedded constants), the NaN-authorizes verdict, the OOM substring laundering, and the compile-excluded projection. Rulings: remat jaxpr pin KEEP with a primitive-walker allowlist, fail closed on drift; six runbook caveats for J1/J1b timing recorded.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md` — reviewer briefing, TDD/review/strengthen discipline, validation ladder, separate TPU approval.
- `plan_null_adapter.md` v5 — §3 batching contract, §4-P1(iii), §4-P1b, §5, §8, §9. Because v5 says P1b is unchanged, I also loaded its expanded v2 text from commit `58c14dd`.
- `null_adapter_worklog.md` through the R11 entry, including the Adam-scaling, finite-difference, and remat test-design notes.
- R3 `optimize-nulls`, R4a `replay-operator`, and R10 `launchers-config` reviews, including all strengthening/follow-up records.
- `null_adapter_yixun_query.md`, announcements 01/02, and exp_03’s two-step remat/scan precedent.
- Both R11 files in full: [null_direct_opt_wan.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py) and [test_null_adapter_direct_opt.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_direct_opt.py).
- The composed J1 caller path: runner, modes, YAML, and launcher.
- HEAD `b027f22`; only the two R11 files are untracked. `git diff --check` passed.

## Findings

1. **BLOCKER — R11 is unreachable from J1, and no J1b launch path exists.**  
   [run_wan_null_inversion.py:65](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:65), [run_wan_null_inversion.py:686](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:686), [base_wan_5b_null_inversion.yml:268](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/configs/base_wan_5b_null_inversion.yml:268)

   Nothing outside the R11 tests imports `measure_single_update` or `direct_optimize_nulls`. `NULL_MODES` still contains only `capacity`, `cache`, `verify_replay`, and `adequacy_probe`; consequently the approved J1 capacity run never performs its required A3 measurement, and a separately approved J1b cannot be dispatched. There is likewise no production boundary enforcing ε₀=`global_noise(0)`, first-eight DEV ordering, `N=25`, `L=16`, `D=4096`, B=8, or final-result persistence.

   Concrete change: wire a one-example A3 measurement stage into J1 and persist its provenance-bound report. Add a distinct, separately approved J1b mode/launcher that reads the first eight DEV examples, constructs canonical ε₀, executes the single B=8/300-iteration optimizer, evaluates the final post-update endpoint, and writes its nulls/losses/report. Test both composed paths through `main`.

2. **MAJOR — the timing helper executes two updates and compiles a zero-argument, constant-specialized program.**  
   [null_direct_opt_wan.py:402](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:402), [null_direct_opt_wan.py:417](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:417)

   `timed()` invokes `update()` twice: once for “compile+run” and once for step timing. That is two executions, contrary to “compile + execute exactly one update.” Moreover, `jax.jit(lambda: ...)` has no operands; my lowered toy StableHLO had `@main()` with no arguments and 85 embedded constants. On CPU this reported 73 ms first-call time versus 17 μs for the second call. With the 5B model, closed-over model/data constants can materially distort compilation, executable size, sharding, and runtime relative to an actual dynamic optimizer step.

   First-minus-second is only a rough estimate: on TPU, the first call also includes tracing, executable loading, transfers and possible first-run initialization; the second runs warm. The checks also occur only after compilation/update returns, so they do not actually abort a compile at 1,800 seconds or a step at 120 seconds.

   Concrete change: build a jitted single-update kernel with dynamic array/model-state operands; time `lower()`/`compile()` separately without executing, then invoke the compiled executable exactly once and synchronize. Enforce the hard wall stops with an outer process/watchdog, since a synchronous TPU compilation cannot be cancelled by the post-return comparisons.

3. **MAJOR — the ≤4-hour wall projection excludes recompilation and can approve a run exceeding four hours.**  
   [null_direct_opt_wan.py:381](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:381), [null_direct_opt_wan.py:397](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:397)

   The projection is only `iters × examples × step_seconds`. J1b is a separate job, with a different B=8 executable and no configured persistent compilation cache, so its compile time is part of wall time. My probe supplied a realistic 600-second compile plus exactly 14,400 seconds of updates; the report returned `fits_budget=True` although projected wall was 15,000 seconds.

   `examples` is also unvalidated: `examples=0` produces a zero projection and `fits_budget=True`; `examples=-1` produces −300 seconds and also fits.

   Concrete change: validate positive integral `examples` and finite non-negative budgets; include projected compilation/setup in the wall calculation. State that B=1×8 is a preliminary compute estimate, not B=8 HBM certification, and make the separately approved J1b begin with a B=8 single-update fit probe before committing to all 300 iterations.

4. **MAJOR — the tests do not pin recurrence fidelity; reversing every `dsigma` survives all runnable tests.**  
   [test_null_adapter_direct_opt.py:118](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_direct_opt.py:118), [null_direct_opt_wan.py:216](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:216)

   Current production is correct: my independent comparison found `direct_rollout` bitwise identical to R4a’s `replay_with_nulls` for both endpoint and complete trajectory. But no R11 test performs that comparison or uses an independent recurrence oracle. I reversed `sigmas[1:]−sigmas[:-1]` to the wrong positive sign; all 40 runnable R11 tests still passed. The finite-difference test remains self-consistent under this mutant because both numerical and analytic derivatives differentiate the same wrong forward recurrence.

   Concrete change: add a coupled-oracle, distinct-null bitwise comparison against reviewed R4a replay for endpoint and every trajectory element. This should also pin fresh per-step `v_cond`, timestep contents, CFG algebra and `dsigma`; retain the finite-difference test for invisible gradient-only mutations.

5. **MAJOR — invalid numerical execution can authorize J1b, while a non-OOM error can be laundered as OOM.**  
   [null_direct_opt_wan.py:308](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:308), [null_direct_opt_wan.py:451](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:451)

   A velocity returning NaNs produced `verdict="ok"`, `loss=nan`, and `fits_budget=True`. Separately, the bare `"OOM"` substring classified `RuntimeError("BOOM: model kernel bug")` as an OOM verdict, contradicting the stated rule that non-allocation bugs propagate.

   Concrete change: after synchronization, require finite losses, gradients and updated nulls; raise a typed numerical failure or return a non-authorizing structured verdict. Detect OOM through exception/status type plus allocation-memory wording or at least word-boundary matching—never a bare substring.

6. **MINOR — peak-HBM reporting can understate or mislabel memory.**  
   [null_direct_opt_wan.py:70](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:70), [null_direct_opt_wan.py:312](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:312), [null_direct_opt_wan.py:331](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_direct_opt_wan.py:331)

   The `None`-rather-than-zero semantics are correct. However, `bytes_in_use` is current allocation, not peak allocation, yet is returned as `peak_hbm_bytes`; and only device 0 is queried on a v6e-8.

   Concrete change: report per-device memory and its source key, take the maximum actual peak across addressable devices, and keep current allocation separate. If TPU exposes no peak counter, preserve `None` and treat it as unavailable evidence—not as a fit certificate.

## Remat-pin ruling and platform caveats

The mathematical core is sound:

- R11 rollout and R4a replay were bitwise identical.
- Pins, fresh per-step `v_cond`, negative `dsigma`, timestep routing, and remat placement around both model forwards are correct.
- Nothing is inadvertently stopped.
- The finite-difference test is sound. At ε=`1e-3`, production errors were approximately `8.6e-4–9.8e-4`; the stopped-`v_cond` mutant missed first-step derivatives by `0.258–0.918` and clearly fails the 2% tolerance. ε=`1e-2` and `1e-4` gave the same ruling.
- The Σ-objective fix is complete inside this optimizer: a mean-objective mutant made the B=2 grad norm exactly half the singleton norm and was killed. Per-example Adam state remains independent.
- The persistent-Adam reference is sound for its intended state-carry comparison. It is an imperative Optax reference rather than literally hand-rolled Adam, but R3 already independently pins the declared recipe.

Remat ruling: **KEEP the jaxpr structural pin and fail closed on dependency upgrades.** On JAX 0.10.2 the recursive primitive is `remat2` inside `scan`; the current substring test sees it. Prefer a recursive primitive walker with an explicit reviewed allowlist over raw string matching. Do not automatically tolerate an unknown renamed primitive—inspect the new lowered backward graph before extending the allowlist.

For the J1 runbook:

- CPU timing is only a logic smoke and predicts neither TPU compilation nor step time.
- Run TPU measurement in a fresh process with dynamic operands and synchronized execution.
- Use an external 1,800/120-second watchdog for real hard stops.
- Record all-device memory; `None` means unknown.
- Treat B=1 timing as preliminary only. B=8 has a different compile, execution, sharding and HBM profile.
- J1b remains separately approved; its B=8 fit probe and full 300-iteration continuation must be covered by that separate approval.

Focused validation: the exact command failed before collection because the read-only sandbox has no usable temporary directory. With pytest capture disabled, 40 tests passed; the tiny-Wan import alone failed for the same temporary-directory limitation. No code assertion failed.

Final verdict: **REQUEST-REVISION — R11’s differentiable optimizer is mathematically faithful, but the missing runner integration, unsound measurement methodology, incomplete wall projection, and demonstrated recurrence-test gap prevent it from safely authorizing J1b.**

### Status

No subprocesses, tests, agents, or TPU jobs are running. Review is complete; Yixun can steer immediately.
