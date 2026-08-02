# Codex code review — exp_03 cycle A round 1, sampler-step extraction (8ccaf3a + fc9a07d)

- **Date:** 2026-08-02
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 2 HIGH (D1 fails open; the trace's approved design is not pinned),
  3 MEDIUM (bf16 oracle floor; parity certificate narrower than claimed; AST guard evadable). All
  hardening: **the motion itself was verified inert at both call sites.**
- **Verified:** no arithmetic reordering at either call site (grid, broadcast, timestep construction,
  latent-dtype delta cast, Euler update, pin ordering, `deterministic=True`, adapter/CFG formula all
  unchanged); the `_rollout_sample` scope extension ruled **sound** under the one-sampler rule; both
  test substitutions preserve their original claims; no launcher/config/trainer work leaked in.

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

1. **HIGH — [d1_per_frame_slopes.py:87](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/diagnostics_exp03/d1_per_frame_slopes.py:87), [d1_per_frame_slopes.py:198](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/diagnostics_exp03/d1_per_frame_slopes.py:198), [d1_per_frame_slopes.py:231](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/diagnostics_exp03/d1_per_frame_slopes.py:231)** — D1 silently truncates unequal videos, fits every available frame after frame 0 rather than exactly frames 1–32, skips missing video pairs, and analyzes the intersection rather than requiring all 100 canonical episodes. This can bias the predeclared mechanism gate. **Fix:** require equal 33-frame videos, fit exactly indices 1–32, require identical 100-name cohorts with unique/matching episode IDs, and fail on missing/extra windows; make both aggregation/self-validation inputs part of the required run contract.

2. **HIGH — [sigma_trajectory_trace.py:223](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/diagnostics_exp03/sigma_trajectory_trace.py:223)** — the approved trace design is not pinned: `config.seed`, `config.probe_num_windows`, and `config.side_adapter_sampling_steps` can silently select a non-seed-0 cohort, fewer/more than 30 windows, or a non-25-step trace. Constants and unit examples do not constrain `run_trace`. **Fix:** add a cheap pre-restore design assertion requiring seed 0, exactly 30 windows, and exactly 25 steps; use the constants directly, verify the resulting cohort/grid lengths, and mutation-test hostile overrides.

3. **MEDIUM — [sigma_trajectory_trace.py:70](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/diagnostics_exp03/sigma_trajectory_trace.py:70), [sigma_trajectory_trace.py:267](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/diagnostics_exp03/sigma_trajectory_trace.py:267), [test_exp03_diagnostics.py:174](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_diagnostics.py:174)** — oracle-zero is tested only in fp32 and only to `<1e-10`, while the real configuration casts `z_gt`, ε, velocity, and Euler delta to bf16. Incremental bf16 Euler rounding is not identical to directly evaluating the interpolant, so the claim that all growth is model error “and nothing else” is not established for the actual run. **Fix:** test the oracle at the production dtype, compute the reference with an explicitly documented dtype, record the per-index finite-precision oracle floor, and predeclare whether the reported metric is raw error or excess over that floor.

4. **MEDIUM — [test_exp03_sampler_extraction.py:110](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_sampler_extraction.py:110), [test_exp03_sampler_extraction.py:151](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_sampler_extraction.py:151)** — the parity certificate is narrower than claimed: single-step parity covers fp32 and bf16, but the chained 25-step test is fp32 only. The reference also covers the plain-transformer velocity, not `_rollout_sample`’s adapter/CFG path. **Fix:** parameterize the chained test over fp32/bf16 and add exact pre-extraction rollout references for adapter guide scales 1 and 5.

5. **MEDIUM — [test_exp03_sampler_extraction.py:260](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_sampler_extraction.py:260)** — the AST guard confirms the current source is clean but does not strongly bind future implementations. A rollout may call the shared function and then add a manual update using renamed indices or `jnp.subtract`; the exact source-string patterns would miss it. **Fix:** assert that each nested `_body` consists solely of returning one shared-step call, that the grid assignment comes solely from the shared-grid call and its outputs feed that step, and add evasive negative mutants.

Claim verification:

- **Claim 1 — verified:** static parent/current comparison found no arithmetic reordering at either call site; grid, broadcast, timestep construction, latent-dtype delta cast, Euler update, pin ordering, `deterministic=True`, and adapter/CFG formula are unchanged. Extending extraction to `_rollout_sample` was sound under the one-sampler rule.
- **Claim 2 — partial:** exact grid and per-index fp32/bf16 equality are present, but full-chain bf16 coverage and a robust no-reimplementation guard are not.
- **Claim 3 — partial:** both diagnostics are substantive implementations; formulas, pairing operation, seed-0 ε, immutable canonical path, `step_` refusal, shared step import, and probe-cohort reuse are present, subject to findings 1–3.
- **Claim 4 — record-consistent:** the new tests collect as +35 cases and the worklog records 1271 passed/2 skipped plus eight killed mutants; I could not independently rerun pytest because this read-only environment lacks the JAX test environment. AST parsing and `git diff --check` passed.
- **Claim 5 — verified:** both substitutions preserve their original claims. The schedule test observes the relocated grid seam and lengths; the knob test now checks that `num_inference_steps` is never read from config while retaining proof that rollout steps come from `side_adapter_sampling_steps`.
- **Claim 6 — verified:** no launcher, exp_03 config, trainer, or binding-hook work entered these commits; the hardware gate remains deferred.

### Risk read / status

Do not stack round 2 on this commit until the five findings are strengthened and re-reviewed. The sampler motion itself is low risk: I found no plausible arithmetic reason for exp_02’s 30-window bitwise gate to fail, though failure remains possible because the real 5B bf16/XLA path has not run and the bf16 chain certificate is incomplete. That S1.5 gate must still pass before the post-extraction evaluator can inherit exp_02’s **PROVEN** status.

- Sub-processes: none running; review complete.
- Earliest steer: now.
```

## Strengthening record — coder response (round 1 strengthening)

All five findings closed. The sampler module's *arithmetic* is untouched by this round — the review
verified it inert and nothing here re-opens it; the changes are contracts, pins and certificates.

### 1. HIGH — D1 is now fail-closed at every input

`per_frame_ssim` refuses unequal-length videos (the exp_02 `min(len(a), len(b))` is gone) and refuses
anything but exactly 33 frames; `window_slope` fits exactly frames `1..32` and refuses another window
length (unit tests that want a different length now say so explicitly via `expected_frames`);
`per_window_slopes` raises on a window directory missing either MP4 instead of skipping it, and
requires exactly the 100-window cohort; the new `assert_paired_cohorts` requires the trial and control
to present the **identical** cohort — no intersection — with unique, matching episode ids, naming the
offending windows on both sides; `compare` now takes **both aggregations as required positional
arguments** and runs the exp_02 self-validation for each arm as part of the contract, failing if any
rendered window has no seed-0 correct-mode row. `main` refuses anything but its four arguments.
Constants added: `D1_LAST_FRAME=32`, `D1_FRAME_COUNT=33`, `D1_COHORT_SIZE=100`.

### 2. HIGH — the trace's approved design is pinned before the model loads

`assert_approved_trace_design(seed, num_windows, sampling_steps)` requires seed 0, exactly 30 windows
and exactly 25 steps, using `TRACE_SEED` / `TRACE_NUM_WINDOWS` / the new `TRACE_SAMPLING_STEPS`
**directly**; `run_trace` calls it *before* `_restore_overfit100_validation_state` (a ~5B load that
then throws has already burned the slot — tested), then passes the constants to the cohort selection
and the grid builder rather than the config values, and verifies the returned cohort length and the
grid's `(26, 25)` shapes. Hostile seeds / window counts / step counts are parameterized refusal tests.

### 3. MEDIUM — the bf16 floor is measured, not assumed

`TRACE_REFERENCE_DTYPE = float32` is now an explicit, documented constant: the reference interpolant
is evaluated in float32 whatever the latents' dtype, so it contributes no rounding of its own. The new
`oracle_floor` re-runs each window with the exact velocity `eps - z_gt` **in the latents' own dtype**,
and `trace_rows` attaches the result as `floor` **inside the same entry** as the `error` it bounds
(one list, so they cannot be misaligned); `mean_trace` averages it; the artifact records
`reference_dtype`, `latent_dtype` and a `reported_metric` string. **Predeclared:** the reported metric
is the RAW error with the floor alongside — subtraction is an analysis decision to be stated when the
numbers are read, never applied silently. Tests show the fp32 floor is < 1e-10 while the bf16 floor is
strictly larger (the finding, made visible) yet still two orders of magnitude below a wrong model's
reading, so Mechanism B can be read through it.

### 4. MEDIUM — the parity certificate now covers what was claimed

The 25-step chain test is parameterized over **fp32 and bf16** (dtype asserted on the result). For the
adapter path, a **verbatim pre-extraction copy of the whole `_rollout_sample`** now lives in the test
file, and the REAL `gen._rollout_sample` is compared against it with `array_equal` on the latents and
on every metric, at guide scale **1.0 (no-CFG shortcut) and 5.0 (two-forward combination)**, in fp32
and bf16 — plus the `FULL_FT_TI2V` branch. `nnx.merge` and the adapter forward are stubbed so the
arithmetic is exercised, not the 5B model.

### 5. MEDIUM — the AST guard binds structure, not strings

`test_each_rollout_body_is_nothing_but_one_shared_step_call` asserts each rollout defines exactly one
`_body`, whose body is a single `return` of one `overfit100_sampler_step` call with exactly the five
expected keywords; that `sigmas`/`timesteps` are the names bound by **one** `overfit100_sampler_grid`
assignment and are never rebound; and that the single `fori_loop` runs that `_body`. The reviewer's
two named evasions and a private second grid are now mutants that fail it.

### Verification

* Full worklogs suite: **1296 passed, 2 skipped** (round-1 commit: 1271 + 2; +25).
* Mutations — 10 killed, 1 proven equivalent:
  1. D1 min-truncates unequal videos again -> 1F;
  2. D1 analyses `set(trial) & set(control)` again -> 2F;
  3. D1 skips windows missing an MP4 -> 1F;
  4. trace design pin removed from `run_trace` -> 1F;
  5. `floor` dropped from the trace rows -> 1F;
  6. **EVASIVE** — body calls the shared step, then adds a manual update with *renamed* indices -> 1F;
  7. **EVASIVE** — the same via `jnp.subtract`/`jnp.add`, so no forbidden substring appears -> 1F;
  8. **EVASIVE** — a second, private grid rebinds `sigmas`/`timesteps` -> 1F;
  9. adapter CFG formula drifts (`v_cond` base instead of `v_uncond`) -> 2F;
  10. latent-dtype cast dropped from the Euler delta -> **4F** (was 1F before this round: the bf16
      chain coverage is what widened it);
  11. *equivalent mutant, reported not chased* — computing the oracle velocity in fp32 and rounding to
      bf16 survives, because bf16 subtraction **is** exactly "subtract, then round to bf16". Verified
      directly: over 4,096 random bf16 pairs, `a - b` and `(f32(a) - f32(b)).astype(bf16)` are
      bit-identical. There is no observable difference to test.
* `black --line-length 119`, `ruff check` clean; staged-tree `git diff --check` clean.

No push, no launch. The S1.5 on-hardware bitwise gate against the landed 30-window scalars remains
where the plan put it, and the post-extraction evaluator does not inherit exp_02's PROVEN status
until it passes.

## Re-review (e4a11a4) — APPROVE, round 1 CLOSED

```
APPROVE

1. CLOSED — D1 now enforces equal 33-frame videos, frames 1–32, complete 100-window identical cohorts, matching unique episode IDs, required aggregations, and per-arm self-validation.
2. CLOSED — Seed 0, 30 windows, and 25 steps are asserted before restore; constants drive cohort/grid construction and returned lengths are validated.
3. CLOSED — The per-window oracle floor uses latent dtype, accompanies raw error in each JSON entry, and the fp32 reference dtype/reporting policy is explicit.
4. CLOSED — Exact parity covers 25-step fp32/bf16 chains, adapter guide scales 1.0/5.0 in both dtypes with all metrics, and the FULL_FT branch.
5. CLOSED — The AST guard binds one shared-grid assignment, one single-return shared-step body with exact keywords, and one `fori_loop`; all three evasive mutants are covered.
6. EQUIVALENT — Accepted for finite production bf16 latents: when subtraction can affect a bf16 rounding boundary, its exact result fits within fp32 precision; larger exponent gaps round identically. The 4,096-pair check corroborates this argument.

Round-2 (trainer + hook): **GO**. The recorded 1296 passed / 2 skipped result is consistent with the changes; this read-only runtime lacks pytest/JAX, so I could not independently rerun it.
tokens used
93,756
APPROVE

1. CLOSED — D1 now enforces equal 33-frame videos, frames 1–32, complete 100-window identical cohorts, matching unique episode IDs, required aggregations, and per-arm self-validation.
2. CLOSED — Seed 0, 30 windows, and 25 steps are asserted before restore; constants drive cohort/grid construction and returned lengths are validated.
3. CLOSED — The per-window oracle floor uses latent dtype, accompanies raw error in each JSON entry, and the fp32 reference dtype/reporting policy is explicit.
4. CLOSED — Exact parity covers 25-step fp32/bf16 chains, adapter guide scales 1.0/5.0 in both dtypes with all metrics, and the FULL_FT branch.
5. CLOSED — The AST guard binds one shared-grid assignment, one single-return shared-step body with exact keywords, and one `fori_loop`; all three evasive mutants are covered.
6. EQUIVALENT — Accepted for finite production bf16 latents: when subtraction can affect a bf16 rounding boundary, its exact result fits within fp32 precision; larger exponent gaps round identically. The 4,096-pair check corroborates this argument.

Round-2 (trainer + hook): **GO**. The recorded 1296 passed / 2 skipped result is consistent with the changes; this read-only runtime lacks pytest/JAX, so I could not independently rerun it.
```
