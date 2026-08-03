# Codex code review — exp_03 S1.5 probe (210e7b1)

- **Date:** 2026-08-03
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 5 BLOCKER (label isolation and parity never executed + a wrong
  parity definition; the variance loop mixes support with branch and schedule; the "init" sigma
  trace is a second checkpoint trace; state fidelity and the ramp mapping are wrong; the
  implementation is not executable at 5B), 3 MAJOR (frozen-replay semantics changed and a vacuous
  assertion; a collision-prone fingerprint; partial launcher/output safeguards).

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

1. **BLOCKER — V2 and parity are not executed.** `state_report()` only runs frozen replay and the variance loop; it never calls `label_isolation()` or `parity_report()`. Consequently neither `label_isolation` nor `p_ss_zero_parity` can appear in an artifact—the logger’s conditional branches are unreachable. The replay also discards B’s `raw_endpoint_mse` and `horizon_sq`, so the raw/normalized-loss claim is incomplete. [probe_exp03_s1_5.py:292](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/probe_exp03_s1_5.py:292) [wan_ti2v_exp03_trainer.py:574](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py:574)

   The parity definition also needs correction: production A at `p_ss=0` uses one auxiliary scalar end-support per batch, while the plain objective samples per-example timesteps from the shared stream. Those end-to-end gradients are not mathematically identical. Require conditional parity against a plain fixed-support comparator using the identical sigma, ε, dropout key, batch, and state. Report the production-control difference separately rather than demanding it be zero.

2. **BLOCKER — V1 does not isolate sigma-support variance for A/C.** The inner loop changes `global_step`; that changes A/B supports, but also A’s `p_ss_coin` key and `exp03_p_ss()` itself. Therefore A/C’s within-batch term mixes sigma-support variation with branch-selection and schedule variation. B alone is clean. [probe_exp03_s1_5.py:315](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/probe_exp03_s1_5.py:315) [wan_ti2v_exp03_trainer.py:92](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py:92)

   Add a support-draw salt consumed only by `k_a_draw`, `index_support`, and `index_support_rollout`; keep logical global step, `p_ss`, coin, batch, ε, and dropout fixed across M. Also call the between term “batch+shared-RNG variance,” not pure data variance. With M=4, the variance of batch means still contains support Monte Carlo error; either report the empirical decomposition honestly or add the usual finite-M ANOVA correction.

3. **BLOCKER — the init sigma trace is actually another checkpoint trace.** `trace.run_trace(config)` reloads state from the unchanged configuration in both iterations. With launcher `checkpoint_step=10000`, both calls restore and write `sigma_trace_ckpt10000.json`; the already-built init state is never traced. [probe_exp03_s1_5.py:447](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/probe_exp03_s1_5.py:447) [sigma_trajectory_trace.py:324](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/diagnostics_exp03/sigma_trajectory_trace.py:324)

   Trace the state already in memory, or pass a state-specific config with checkpoint step 0 for init. Reusing the loaded state is preferable because loading a second 5B model while the first remains live is itself an OOM risk.

4. **BLOCKER — V3/V5 state fidelity and ramp handling are wrong.** The checkpoint probe restores `manager.latest_step()` rather than the required step 10000; the passed `CHECKPOINT_STEP` does not constrain this path. Init bypasses the checkpoint-manager/empty-restore path entirely. The checkpoint data iterator also uses `seed`, whereas production continuation uses `seed + start_step`. [probe_exp03_s1_5.py:258](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/probe_exp03_s1_5.py:258) [wan_ti2v_overfit100_trainer.py:1445](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:1445)

   `state_report()` then evaluates both states at global steps `0..7`, while the launcher supplies one ramp origin, `10000`. Thus both states see `p_ss=0`. The correct canonical mapping is:

   - checkpoint: global steps `10000..10007`, ramp origin `10000`;
   - init: global steps `0..7`, ramp origin `0`.

   Yes, init must use origin `0`. Although the probe performs no updates, the ramp changes A/C’s branch and therefore their losses and gradients. Add separate forced-`p_ss=1` A/C diagnostics for the approved self-generated-state reading.

5. **BLOCKER — the variance implementation is not executable at 5B scale.** It retains 32 full gradient trees per objective, then converts every leaf to a concatenated float64 vector. A 5B-element float64 gradient is about 40 GB; 32 vectors are about 1.28 TB host memory, aside from the retained device gradients. `_flat_gradient()` creates the same class of large host copies for every cosine. [probe_exp03_s1_5.py:159](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/probe_exp03_s1_5.py:159) [wan_ti2v_exp03_trainer.py:543](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py:543)

   Compute dots, norms, and Welford variance reductions leafwise/on-device, retaining only O(1) gradient trees.

6. **MAJOR — V4’s prior frozen-replay semantics are not preserved or regression-certified.** The default now performs an extra control forward/backward, and legacy `grad_cosine_ab` changed from its prior float32 tree reduction to host float64 flattening. The new test is source inspection; its signature assertion ends with `or True`, making it unconditional. [test_exp03_s1_5_probe.py:326](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_s1_5_probe.py:326)

   Preserve the old default (`include_control=False`) and opt in from S1.5, or prove legacy outputs against the parent commit—including `grad_cosine_ab`—with a real behavioral regression test.

7. **MAJOR — the “EXACT” no-update proof is not exact.** Summing bytes per leaf is collision-prone; swapping two parameter values leaves the fingerprint unchanged. A one-ULP test proves detection of that mutation only, not bit identity. Use direct bitwise leaf comparison against retained immutable pre-probe arrays, or a documented cryptographic digest. [probe_exp03_s1_5.py:123](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/probe_exp03_s1_5.py:123)

8. **MAJOR — launcher/output safeguards are only partial.** `CHECKPOINT_DIR` is not actually required because it is defaulted before the `:?` check. The claimed bidirectional drift test only checks keys missing from the new launcher, not unexpected additions. The path refusal, immutable writer, lack of verdict fields, and no-ffmpeg decision are sound, but artifacts should reject non-finite JSON values and record per-batch rows, state-specific ramp/global-step metadata, exact restored identity, support salts, and branch counts. [probe_exp03_s1_5.sh:90](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/bash_scripts/probe_exp03_s1_5.sh:90) [test_exp03_s1_5_probe.py:372](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_s1_5_probe.py:372)

### Required S1.5 launch spec

Use a fresh `RUN_NAME` and require/print:

- `CHECKPOINT_DIR=<exact exp_02 checkpoint root>`
- `CHECKPOINT_STEP=10000`, with exact-step restore and refusal if absent
- `INIT_CHECKPOINT_DIR=<reviewed empty root>` or a shared production initializer helper that provably bypasses restoration
- checkpoint ramp origin `10000`; init ramp origin `0`—two pinned settings, not one shared `EXP03_RAMP_ORIGIN`
- checkpoint base step `10000`; init base step `0`
- `S1_5_NUM_BATCHES=8`
- `S1_5_SUPPORT_DRAWS=4`
- `EXP03_P_SS_MAX=0.5`, `EXP03_P_SS_RAMP_STEPS=500`
- forced-`p_ss=1` label-isolation/A/C diagnostic
- `PROBE_NUM_WINDOWS=30`, sampling steps `25`, seed `0`
- manifest-derived model revision/local snapshot and a required, non-`unknown` `COMMIT`
- checkpoint iterator seed `seed+10000`; init iterator seed `seed`

Expected immutable outputs under `validation_probe_sampling/`:

- `s1_5_checkpoint_ckpt10000.json`
- `s1_5_init_ckpt0.json`
- `sigma_trace_ckpt10000.json`
- `sigma_trace_ckpt0.json`

Each state artifact must contain per-batch objective diagnostics, raw and normalized components, conditional parity with loss and gradient gaps ≤`1e-5`, forced-self-generated label isolation on demonstrably identical stop-gradient states, support-only M=4 variance with fixed coin/logical step/shared stream, bounded-memory noise-scale reductions, strict-finite JSON, and an exact no-update certificate. No S1.5 launch should use `210e7b1` as written.
```

## Strengthening record — coder response (S1.5 strengthening)

### BLOCKER 1 — the diagnostics are wired, and the parity definition is corrected

`state_report` now calls `label_isolation` and `parity_report`, and returns `label_isolation`,
`p_ss_zero_parity`, `forced_p_ss_one` and `per_batch` — so the artifact contains them and the
logger's branches are reachable. B's `raw_endpoint_mse` and `horizon_sq` (and both sigma endpoints,
supports, `p_ss`, `k_a`) are surfaced from the replay's aux, so the raw/normalized pair is complete.

The parity definition was wrong and is replaced. Production A at `p_ss=0` scores ONE scalar sigma
per batch while the control samples a timestep per example, so their gradients are not the same
quantity and demanding equality demanded the wrong thing. The gate is now **conditional parity**
against `plain_fixed_support_loss` — a comparator holding sigma, epsilon, dropout key, batch and
state identical to A's and differing only in the label — and the production-control difference is
**reported separately**, explicitly not gated.

### BLOCKER 2 — a support SALT, so the variance isolates the support

`exp03_aux_key` takes a `salt` consumed only by `k_a_draw` / `index_support` /
`index_support_rollout`, and folded **only when non-zero**, so every pre-existing draw is bit
identical (pinned by the S1 failing-draw test). The M draws now vary the salt alone: batch,
epsilon, dropout key, logical global step, `p_ss` and A's coin are all held fixed. The between term
is renamed **`batch_shared_rng_variance`**, and both population and unbiased (`M-1`, `K-1`)
estimates are reported with a `finite_m_note` stating that the between term still carries support
Monte-Carlo error at M=4 — reported as measured, not corrected away.

### BLOCKER 3 — the sigma trace uses the state in memory

`trace.run_trace(config)` is gone (it re-restored, so both traces were checkpoint traces and two 5B
models would have been live at once). `trace_in_memory_state` merges the already-built state and
traces it with the shared step function.

### BLOCKER 4 — state fidelity and the ramp

Both states go through the checkpoint manager: the checkpoint state must come back at the
**required** step 10000 (`latest_step()` is not a pin) and the init state takes the production
empty-restore path. The iterator is seeded `seed + start_step`, production's continuation
semantics. `S1_5_STATE_PLAN` gives the canonical per-state mapping — checkpoint at global steps
10000..10007 with ramp origin 10000, init at 0..7 with origin **0** — so the launcher no longer
carries a single `EXP03_RAMP_ORIGIN` that would give both states the same ramp. Forced `p_ss=1`
A/C diagnostics are added at both states.

### BLOCKER 5 — executable at 5B

All reductions are leafwise/on-device: `tree_dot`, `tree_sq_norm`, `grad_cosine` (float32 tree
reduction), a `_relative_gradient_gap` that never flattens, and a `_TreeWelford` streaming
accumulator retaining **one** mean tree and a scalar. The previous shape — 32 retained gradient
trees flattened to host float64, ~1.28 TB at 5B — is gone, and a test forbids `np.float64`,
`np.concatenate` and `_flat_gradient` in both modules.

### MAJOR 6/7/8

`include_control` defaults to **False** again (S1.5 opts in), `grad_cosine` is back to the float32
tree reduction, and the vacuous `or True` is deleted — with a meta-test that fails if one
reappears. The fingerprint is a **sha256 digest per leaf** (a byte sum collides on any permutation;
a test now swaps two values and requires detection). `CHECKPOINT_DIR` is genuinely required (the
default that preceded the `:?` check is removed, and the check is verified to fire). Artifacts
record per-batch rows, per-state ramp/global-step metadata, the required and restored steps, the
iterator seed, and the support salts — and `assert_finite_payload` refuses to write a non-finite
value, naming the key.

### Verification

* Full worklogs suite: **1470 passed, 2 skipped** (+8).
* Mutations — 7, all killed:
  1. the diagnostics unreachable again -> **2F**;
  2. the salt no longer isolating (support view dropped) -> **1F**;
  3. the salt folded even when zero -> **2F** (caught by the pinned S1 draw tests, as designed);
  4. the restore falling back to `latest_step()` -> **1F**;
  5. the float64 host flatten reintroduced -> **1F**;
  6. the in-memory trace reverting to `run_trace` -> **1F**;
  7. an `or True` reintroduced -> **1F**.
* `black`, `ruff`, `bash -n`, staged-tree `git diff --check` clean.

No push, no launch.
