# rollout_adapter — Codex code review: T3a `cfg-rollout-step`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: REQUEST-REVISION — 1 MAJOR + 2 MINOR. **The reviewer REJECTED the Planner-accepted two-oracle deviation and supplied the construction that makes it unnecessary**, running it on the real fixture: isolate the second-step unconditional term `(1−w)·(σ_{s+2}−σ_{s+1})·v_unc(z_1(θ))` and FD *that* rather than the whole loss ⇒ **13.3–107× discrimination at h_rel=3e-3, 109–1004× at 1e-3**. It also refuted the Planner's float64 premise: `jax.experimental.enable_x64` is absent but **`jax.enable_x64()` EXISTS in JAX 0.10.2** as a scoped restoring context manager. Everything else PASSED independently: probe-direction correction ACCEPTED (not circular — `d` frozen before perturbation, FD evaluates the true directional derivative, Cauchy–Schwarz makes it maximally discriminating); carry-cut PASS; all three §3a clauses PASS (39 adapter leaves, frozen leaves unchanged, no production stop-gradient, block-0 stop-grad preserved, trainer tripwire); bitwise 25-step both-branch parity PASS with the evaluator unrewired; CFG exact-form PASS.

**PLANNER ERROR, recorded:** the Planner ACCEPTED the deviation (plan v2.3) on the Coder's fp32 evidence. The evidence was sound for the *whole-loss* FD but the conclusion was wrong — branch isolation was never tried. The reviewer's rejection is upheld; **plan v2.3's two-oracle clarification is SUPERSEDED by v2.4** (direct isolated-unconditional FD restored as the §3a discharge; exact contrast RETAINED for the conditional branch and for obfuscated-mutant coverage, where it remains the only killer). Lesson: when a measurement says "X cannot be done", check whether it measured X or a harder proxy for X.

## Reviewer output (verbatim)

## Findings

- **MAJOR — missing autodiff-independent unconditional attribution.** [test_pos_rollout_step.py:576](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_step.py:576), [pos_rollout_step.py:30](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_step.py:30), [plan_rollout_adapter.md:23](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:23): the composite FD plus exact contrast does not discharge the original unconditional-branch FD obligation, and a viable fp32 construction exists. Isolate the second-step term
  \[
  (1-w)(\sigma_{s+2}-\sigma_{s+1})\,v_\text{unc}(z_1(\theta))
  \]
  and compare its central FD to \(\langle\nabla L_\text{full}-\nabla L_{\text{cut-uncond}},d\rangle\). On the existing real fixture and five seeds, this produced 13.3–107× discrimination at `h_rel=3e-3` and 109–1,004× at `1e-3`. Add this oracle; retain exact contrast for the conditional branch and obfuscated-mutant coverage.

- **MINOR — reported headroom is stale/internally inconsistent.** [test_pos_rollout_step.py:22](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_step.py:22), [test_pos_rollout_step.py:79](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_step.py:79): `5.03 / 3.0` is 1.68×, not ≥2.3×. My clean run measured FD ratios 4.97–21.89, giving 1.66× minimum FD headroom. Correct the narrative after settling the replacement oracle.

- **MINOR — S05’s consolidated battery log still records the syntax-error kill.** [battery_t3a.log:21](</private/tmp/claude-501/-Users-yixunhu-Home-maxdiffusion/800fed95-7c3f-418d-b779-9914ed8480b4/scratchpad/battery_t3a.log:21>): replace or append the valid rerun result. I independently compiled and injected the corrected S05; collection succeeded and the exact-contrast test genuinely failed with `relative["uncond"] == 0.0`.

## Verdicts

- **(a) Deviation: REJECT; direction correction: ACCEPT.** Choosing \(d=\mathrm{normalize}(g_\text{full}-g_\text{truncated})\) is not circular: \(d\) is frozen before perturbation, FD evaluates the true directional derivative, and Cauchy–Schwarz makes this the maximally discriminating direction. The FD measurement is independent, although its direction selection is AD-informed. Also, while the two `jax.experimental` names are absent, JAX 0.10.2 has the scoped, restoring `jax.enable_x64()` context manager; the “fallback unavailable” inference is false. More importantly, branch isolation succeeds in fp32.

- **(b) Numerics:** `h_rel=3e-3` is defensible for the current composite fixture, but start/configuration and thresholds were selected from the same measured seeds: they are fitted regression sensitivities, not independent validation margins. That is acceptable as guardrail coverage once the direct unconditional FD is added and honestly labeled.

- **(c) Carry cut: PASS.** Cutting each step’s input preserves the final step’s direct parameter derivative while removing inter-step carry; cutting the output would make the comparison trivial.

- **(d) §3a clauses: PASS.** Independently confirmed 39 adapter leaves, structural parameter-only differentiation, unchanged frozen leaves, no production stop-gradient, preserved block-0 stop-gradient, and the one-step trainer tripwire.

- **(e) Parity: PASS.** Both 25-step fp32 branches pass bitwise against the drift-pinned deployed loop body; the evaluator remains unrewired.

- **(f) CFG arithmetic: PASS.** Deployment’s exact expression is preserved, and S08 is a genuine numerical/parity kill.

Validation: focused suite **18 passed in 43.49s**; lint passed. Actual reviewed HEAD was `7a09d29`—the T3a documentation commit atop the stated `649cff4`.

**REQUEST-REVISION — add the branch-isolated unconditional central-FD oracle before committing T3a.**
