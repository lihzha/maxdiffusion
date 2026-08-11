# exp_06 F4 `scan-accumulation` — Codex code review

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-11. Verdict: **REQUEST-REVISION — MINOR only** (a worklog wording fix; "no BLOCKER or MAJOR"), with the explicit closing ruling: "After correcting the sweep wording and receiving the green suite result, I see no remaining reason to withhold the v6e-8 relaunch." The suite gate returned **2159 passed / 0 failed** while this review was in flight. The one_step 2-ulp departure and the repaired trace-count test were both independently ratified.

Verdict: **REQUEST-REVISION — MINOR only.** No BLOCKER or MAJOR found.

- **MINOR — the sweep claim is overbroad.** The worklog says this was the only unrolled accumulation in `src/`, based on a non-recursive `src/maxdiffusion/*.py` grep ([worklog:604](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_worklog.md:604)). A recursive sweep finds another Python-unrolled microbatch gradient accumulation in the separate POS-context regression trainer ([wan_pos_context_regression_trainer.py:194](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:194)). Narrow the statement to the exp_06 rollout/M1 path. This does not affect M1-3.

Everything execution-relevant checks out:

- The scan preserves batch/draw ordering, gradient and fp32 loss accumulation, final averaging, and the single optimizer update.
- The frozen tree remains an explicit outer-JIT argument and a loop-invariant scan operand; there is no per-iteration placement, donation, or unsafe aliasing.
- Trainer and fit probe both still obtain the same scanned step from `build_training_program`; M1 measures what M3 runs.
- The F3 captured-constant guard traces `program.step.trace`, i.e. the real production builder after the rewrite.
- Fit-probe trial loops are host orchestration, and the scorer remains batch-one without microbatch accumulation.
- The one-step numerical departure is acceptable. Global-norm clipping does spread last-bit differences downstream, as recorded, but both experimental arms now use the same builder; comparison against the retired implementation is not the estimand. The existence assertion is an acceptable environment-change tripwire, albeit intentionally churn-sensitive.
- The repaired trace-count test is sound: `==1` pins the single staged gradient block, while numerical parity tests preserve the actual all-microbatch contract.
- The supervision-bypass process finding is recorded at [worklog:677](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_worklog.md:677).
- Accepted evidence: battery 80 REFUSED / 0 SUCCEEDED. The 2159-test canonical suite remains pending and must be green before ceremony or M1-3 relaunch.

After correcting the sweep wording and receiving the green suite result, I see no remaining reason to withhold the v6e-8 relaunch. The production graph now contains one scanned gradient block rather than the prior ~118k-equation unroll.
