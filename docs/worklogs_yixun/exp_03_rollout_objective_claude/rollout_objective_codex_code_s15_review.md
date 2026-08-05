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

## Re-review (3218d5f) — findings 1/6/7 CLOSED, four remain

```
APPROVE, so do not launch `3218d5f`/current `HEAD`.

After strengthening, use v6e-8 with:

```text
RUN_NAME=exp03-s1_5-<fresh-UTC-stamp>-yixun
CHECKPOINT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100/wan-overfit100-s3-20260730/checkpoints
CHECKPOINT_STEP=10000
MANIFEST_PATH=docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json
OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100
DATA_DIR=gs://v6_east1d/datasets/exp02_overfit100/train100
EVAL_DATA_DIR=gs://v6_east1d/datasets/exp02_overfit100/train100
EXPECTED_WINDOWS=1629
NUM_TEXT_SLOTS=100
S1_5_NUM_BATCHES=8
S1_5_SUPPORT_DRAWS=4
PROBE_NUM_WINDOWS=30
PER_DEVICE_BATCH_SIZE=1.0
SKIP_HF_PREFETCH=0
COMMIT=<post-fix deployed 40-hex SHA>
HF_HUB_DISABLE_XET=1
HF_HUB_ENABLE_HF_TRANSFER=0
```

The deployed config must retain seed `0`, 25 sampling steps, `p_ss_max=0.5`, and a 500-step ramp. No `EXP03_RAMP_ORIGIN` environment variable: origins remain internally pinned to checkpoint `10000` and init `0`. The derived init checkpoint directory must be verified empty before submission.

Expected immutable outputs:

```text
validation_probe_sampling/s1_5_checkpoint_ckpt10000.json
validation_probe_sampling/s1_5_init_ckpt0.json
validation_probe_sampling/sigma_trace_ckpt10000.json
validation_probe_sampling/sigma_trace_ckpt0.json
```

As written, the current code would write the checkpoint S1.5 artifact, crash in `log_summary`, and produce neither the init artifact nor the standalone trace artifacts.
```

## Strengthening record — coder response (S1.5 closing)

### BLOCKER 1 — the renamed key, and a test that EXECUTES both states

`log_summary` still read `data_variance`, so it raised `KeyError` after the checkpoint artifact and
**the init state never ran**. Renamed to `batch_shared_rng_variance`.

The important part is the test. This wiring-bug class has now bitten three times (an unreachable
branch, then a dead conditional, then a renamed key), and every time it was because a test
*inspected* the control flow instead of *running* it. `test_both_states_run_the_full_report_and_log_path`
is parameterized over both states and drives the real path — `state_report` -> `s1_5_artifact` ->
`log_summary` -> `write_s1_5_artifact` — on a tiny real `Overfit100TrainState` with real batches,
asserting the summary lines actually appear and the artifact actually writes.

### BLOCKER 2 — the first state is freed before the next is built

The per-state work moved into `_run_one_state`, whose locals die when it returns, and which ends
with `release(state, batches, report, pipeline, iterator)`; the driver calls `release()` again
between states. Collectability is asserted with a `weakref` rather than by inspection. The
standalone per-state sigma-trace JSONs are now written too, through the trace module's own
`trace_artifact` / `trace_output_path` / `write_trace_artifact`, so mechanism B has the files its
later comparisons expect (`sigma_trace_ckpt10000.json` and `sigma_trace_ckpt0.json`).

### BLOCKER 3 — the required step is SELECTED

`restore_exact_step` calls `manager.restore(10000, ...)` after checking `all_steps()`; the
required-step comparison downstream is now only a backstop. The test uses a manager holding
`{10000, 12500}` — the real shape of the exp_02 run directory — and asserts 10000 is the step
requested, that a missing required step is refused, and that `required=0` (init) both returns
untouched state and refuses a non-empty directory.

### BLOCKER 4 — true streaming, and a test that can see it

`variance_decomposition` consumes a generator of generators, `del`s each gradient as it goes, and
accumulates through `_TreeWelford`; the probe passes generator expressions. Peak retention is
asserted with `weakref.finalize` on the leaf arrays (peak <= 3, not 32).

Worth flagging: the first version of this check was a substring assertion, and the "materialize into
lists" mutant **survived** it — `[list(_draws(...)) for ...]` still contains the generator
function's name. The check is now structural (AST): the single argument must be a `GeneratorExp`
whose element is a call, `list(` must not appear in it, and `_draws` must itself contain a `yield`.
That kills both the list-of-lists mutant and the subtler list-of-generators one.

### MAJOR residuals

Artifacts record `branch_outcomes` (self-generated vs teacher-forced counts, plus the per-batch coin
and `p_ss` values, surfaced from the replay's aux). `assert_commit_is_pinned` refuses anything but a
40-hex COMMIT before either state loads, per the exp_02 resume precedent. The drift test now rejects
unexpected **additions** as well as omissions.

### Verification

* Full worklogs suite: **1477 passed, 2 skipped** (+7).
* Mutations — 6, all killed:
  1. the rename half-reverted -> **2F**;
  2. the first state retained across the next build -> **1F**;
  3. the restore falling back to `latest_step()` -> **2F**;
  4. the gradients materialized into lists -> **1F** (this SURVIVED the substring assertion, which
     is why the check is now AST-based);
  4b. a list comprehension of generators -> **1F**;
  5. an unknown COMMIT accepted -> **1F**.
* `black`, `ruff`, `bash -n`, staged-tree `git diff --check` clean.

No push, no launch.

## Final verdict (3ffb8f9): **APPROVE — S1.5 GO**

All four BLOCKERs + MAJOR residuals closed and verified (real-path execution both states; scoped state
lifetime; exact-step 10000 from {10000,12500}; genuine streaming; canonical per-state traces; branch
outcomes; pre-load commit pin; bidirectional drift). S1.5 series: 210e7b1 → 3218d5f → 3ffb8f9, three passes.

## Fix #2 series — config views empty on the pyconfig proxy (Job 8b failure)

- **`2b9177d` review (xhigh): REQUEST-REVISION** — items 1/2/4/5 PASS (helper + guards + unit
  closures discriminating, spot-run 5/5; no other production `vars(config)`; proxy getattr-with-default
  raises ValueError but no replay-path soft read targets a key absent from the YAML; no new hardware-path
  defect). **Item 3 BLOCKER:** the e2e converted the 225-key production key set into a SimpleNamespace,
  so the reviewer's executed reversion mutation (helper back to `dict(vars(config))`) still passed both
  e2e cases — the e2e retained the exact shape-blindness that shipped Jobs 8/8b. Relaunch: NO-GO.
- **Strengthening `994af9b`:** `_proxy_config` test double with pyconfig's real contract (empty instance
  `__dict__` asserted, `__getattr__` over a keys closure, `get_keys()`, missing key → ValueError,
  assignment refused); e2e parameterized [proxy, namespace] × [checkpoint, init]. Reversion mutation now
  fails 4 tests — both proxy e2e cases die AT VIEW CONSTRUCTION via the empty-keys guard
  (state_report:533 → state_view:369 → guard:352) — plus the 2 unit closures; M2/M3/M4 still 1 kill
  each; baseline 58; suite 1,489 passed / 2 skipped.
- **`994af9b` verify (xhigh): APPROVE — "Relaunch: GO — the blocker is closed."** Reviewer re-executed
  the reversion in-memory: both proxy cases fail at the guard with zero replay calls; all four baseline
  e2e cases pass; namespace cases still pass under reversion (fallback branch stays covered); the commit
  touches only tests and documentation.

## Fix #3 series — frozen-replay HBM OOM at 5B (Job 8c failure)

- **`46b43c5` review (xhigh): REQUEST-REVISION** — 2 BLOCKERs / 2 MAJORs / 1 MINOR; Welford donation
  VERIFIED; replay-side jit caching ruled sound (objective/config/scheduler identities separate
  closures; global_step traced; K batches reuse 4 compilations/state).
  1. **BLOCKER (correctness, silent):** `_PROBE_GRAD_CACHE` keyed by `tag` only while cached functions
     closure-capture state-specific views — the init state would reuse checkpoint closures, corrupting
     A/C variance + forced_p_ss_one via the wrong ramp origin.
  2. VERIFIED — `_TreeWelford` donation safe; equations match reference.
  3. **MAJOR:** `ResidentTrees` is manual acquire/release bookkeeping, not lifetime measurement —
     hidden references would not change the artifact; auditability claim false as stated.
  4. **MAJOR:** `test_no_eager_whole_tree_gradient_remains...` is inspect-only (AST-matches `jax.grad`
     spellings) — 6th instance of the inspect-vs-execute class.
  5. MINOR: empty-tree max_abs silently changed −1 → 0, untested.
  6. **BLOCKER:** probe-wide residency not capped — parity path retains trial/comparator/production
     grads while forced grads are created (real 4-tree peak outside the replay counter); Welford holds
     two fp32 mean trees in later rows; ~54 reverse specializations probe-wide (8 in the replay).
  **Relaunch: NO-GO** — "fix the cross-state cache collision and release the parity gradients before
  forced diagnostics; then measure the complete probe's residency, including Welford."
- Revision round dispatched: varying quantities as traced/static args instead of closure captures
  (structural collision removal + fewer specializations), parity grads released pre-forced,
  `jax.live_arrays()`-based high-water gauge recorded into the artifact, executable two-state
  compilation-count + residency test housing the collision regression pin, empty-tree semantics pinned.

- **`7241030` re-review (xhigh): REQUEST-REVISION** — design accepted (tracer-safe ramp_origin,
  static_key, cache shape sound; parity trio released; empty-tree max_abs pinned) but the EVIDENCE
  layer failed under executed mutations: collision pin watched replay outputs (never collision-prone —
  forced-origin mutation passed it); gauge samples after peaks unwind (`_draws` holds the previous
  gradient through the next reverse pass; replay sampled post-unwind; baseline dropped);
  `PROBE_COMPILATIONS` counts wrapper constructions, not JAX compilations; stale isolation-loop `grad`
  → real four-tree parity peak; Welford alias-dance rested on a false theory (aliases don't decline
  donation) and added an exception-path bug (count incremented, mean=None). **Relaunch: NO-GO.**
- Round-3 dispatched: stale-loop-variable sweep + dels; gauge sampling inside peak windows and inside
  the replay; grad-shaped buffer COUNT from `jax.live_arrays()` (tree-multiples assertable at toy AND
  5B scale); per-wrapper `._cache_size()` compile assertions incl. helpers; collision pin re-pointed
  at cache-served rows with the forced-origin mutation required to kill it; transactional Welford
  update; per-tag first-call wall-time log for v6e-8 compile observability.

- **`99b5529` re-review (xhigh): REQUEST-REVISION** (third NO-GO on the fix-#3 series; review passes on
  this series now: 3 — over the ~2–3/round budget note in issue #9, justified by executed-mutation
  catches each pass). Item 4 (Welford transactional/donation) **CLOSED**; stale-`_draws` and
  eager-fallback mutants confirmed killed. Open: (1+3) the collision pin still watches a PARALLEL
  `_jit_observed_p_ss` recomputation — the reviewer's surgical mutation (forcing origin only inside
  cache-served calls) survived; (2, executed) the gauge miscounts at production shape — AdamW moment
  trees (param-shaped) read as baseline 2.0 phantom grad trees, while fp32 Welford buffers for bf16
  params are missed by the shape+dtype filter; (5) no production specialization total
  asserted/reported; probe tags untimed; COMPILE_TIMINGS merges the init-state replay compile.
  **Relaunch: NO-GO.**
- Round-4 dispatched: cached fns EMIT their ramp-dependent quantities in aux (pin re-pointed at the
  real path; reviewer's surgical mutation as acceptance); opt_state DROPPED after restore (no-update
  probe — moments are dead weight; frees ~40 GB HBM, the headroom that killed 8c, and cleans the
  gauge baseline at the source); identity-based gauge (baseline buffer-identity snapshot,
  tree-equivalents by bytes); production specialization total + per-(tag,state) compile timing into
  the artifact.

- **`ce6717d` (round 4, fresh Coder after two stalls) re-review (xhigh): REQUEST-REVISION** — finding
  1 (collision evidence) **CLOSED**: "loss and aux share one traced view; RampWitness is cache-served,
  and the surgical-origin test faithfully mutates only that path" (`_jit_observed_p_ss` deleted; the
  surgical mutation is itself a committed test; the fresh Coder also fixed a latent bug in the
  inherited partial work — aux had been computed on a DIFFERENT view than the loss). Moment-drop
  ruled safe (restore contract + no-reader proof by full double-execution diff); census mechanism
  ruled real. Remaining: **(BLOCKER, executed on an 8-device mesh)** the gauge double-counts sharded
  allocations — `jax.live_arrays()` returns global arrays PLUS per-shard arrays, one 256-byte
  gradient read as 512 B / 2.0 trees; production bf16 census is 39 not 37 (grad_stats + welford_first
  + welford_update each specialize twice); 4 salt-specific variance wrappers share one timing key
  (12/16 variance compiles untimed per state). **Relaunch: NO-GO.**
- Round-5 dispatched (mechanical): physical (device, buffer-pointer) dedupe for both identity and
  bytes; executed 8-device inversion test; census 39; per-(tag, salt, state) timing with a
  coverage==specializations assertion.

- **`2ef9b8a` (round 5) closing review (xhigh): APPROVE — "Relaunch: GO".** All three items CLOSED:
  physical-allocation aggregation exclusive (8-device execution: 256 B / 1.0 tree; old object-sum
  reproduces 9 objects / 512 B / 2.0 trees as the asserted must-fail); helper caches split 2/2/2/1/1
  → 8 helpers, production total 39; 16 distinct variance tags; missing/collapsed timings fail the
  23-tag / 27-per-state assertions; zero-addressable fallback conservative + flagged and irrelevant
  at v6e-8 (single-VM, one process). **Fix-#3 series CLOSED after 5 rounds** (46b43c5 → 7241030 →
  99b5529 → ce6717d → 2ef9b8a; review passes: 5). Suite 1,511 / 2.

## Round 6 (`4d2a79e`, Job-8d program-OOM fix) — REQUEST-REVISION

- Accepted: stale pre-restore 5B transformer retention analysis + free (no surviving alias);
  encode-time frees downstream-safe; umt5-xxl correction (PyTorch, host RAM). Factual fix: training
  donates RESTORED arrays — its immunity comes from no-restore aliasing (from-scratch) and v6e-64
  sharding, not from donating the stale tree.
- **BLOCKER:** except-path `largest_live_arrays()` unguarded (an inspection failure would REPLACE
  the original OOM) and sums shards across all 8 devices with global/shard double-counting.
- **Key insight (finding 4):** remaining deficit (~5.61G = 15.11 wanted − 9.50 free, vs ~2.6G
  visible reclaim) is plausibly LOADED-EXECUTABLE residency — invisible to `jax.live_arrays()`.
  "Instrument or release the cached control executable." **Relaunch: NO-GO.**
- Round-7 dispatched: best-effort failure diagnostics; per-chip deduplicated top-10;
  `unattributed = in_use − arrays_bytes` per ledger point + `post_control_replay` point;
  **per-objective-outer replay walk with per-objective executable release** (census snapshots
  before each release; walk-order equivalence pinned by a both-orders toy test); per-device-batch
  fallback lever to be REPORTED, not implemented.

- **`ec87d8d` (round 7) review (xhigh): APPROVE — "Relaunch: GO".** All round-6 findings closed
  (best-effort failure diagnostics; per-chip dedup sound; release/census mechanics correct — the
  budget catches released-then-reused at 3). Reviewer CONCURS with rejecting per-objective-outer
  (~9–10 GiB/chip of control grads, violates the 3-tree cap; batch-outer stays). Relaunch plan
  ratified: **one bounded v6e-8 attempt (8e) at XLA_PYTHON_CLIENT_MEM_FRACTION=0.95** (+0.9 GiB) to
  harvest unattributed-bytes attribution — improved odds only from the untried pre-replay frees +
  mem fraction; **stop-rule: if the control→A OOM repeats, do NOT spend another v6e-8 attempt** —
  take the ledger to Yixun for v6e-64 approval. Suite 1,523 / 2.

- **`6dab9b1` (round 8, out_shardings pin) review (xhigh, account #3 after the quota block):
  APPROVE — "Relaunch: GO".** Pins complete (4 replay + 23 probe producers; aux scalar-only; hook
  produces no arrays); Welford donation+pinning compatible, structural sharding-tree cache key
  clean, wrapped-function cache-keying handled in census/reset; context table concurred as an
  intentional replicated input (single physical allocation). 8f prediction "quantitatively
  credible" — retain 0.95 mem fraction, acceptance = `post_replay_control ≈ 2.9G`; v6e-64 only if
  the ledger contradicts sharded residency. Suite 1,525 / 2.

## Rounds 9–10 (`c052c91` →) — the batch-1 sigma-trace fix (Job 8f blocker)

- **8f hardware verdict first:** the entire checkpoint-state report COMPLETED at 5B — the
  out_shardings fix verified (post_replay_control 3.09G vs predicted 2.9G; peak 8.81G/31.25G;
  all releases clean). Death was in the last per-state diagnostic: the sigma trace's batch-1
  rollout vs the FSDP divisibility constraint.
- **`c052c91` (round 9) review (xhigh): REQUEST-REVISION** — tiling + row-0 readback PASS (bitwise
  on the 8-device toy; per-row independence audited in the transformer source and CONFIRMED,
  flash/collective paths unreachable at 540 tokens); sweep PASS (the standalone `run_trace`
  carried the same latent defect, fixed via one shared helper). **FAIL:** the trace runs ~1,500
  synchronized EAGER forwards through 40 nnx.remat blocks — tens of minutes to low hours,
  unacceptable retry uncertainty → cache a jitted tiled forward, slice row 0 outside the compiled
  boundary. Strengthen the hand-rolling guard structurally (string match evadable). Design (a)
  ruled NOT required. **Relaunch: NO-GO** pending the jit wrap. Round-10 dispatched.

- **`f049f16` (round 10, jitted trace forward) verify (xhigh): REQUEST-REVISION** — jit core PASS
  (params traced, merge-inside, cache/timing/snapshot-before-drop sound); tiling RATIFIED with a
  sharper reason (untiled jit = silent partial replication, a different regime from production's
  batch-partitioned layout); GSPMD size-1 measurement accepted. Integration defects: stale local
  `forward`/`velocity_fn` refs make the post_trace_release ledger line false; persisted census
  captured pre-trace (trace section missing from the artifact); the AST guard remains evadable
  (positional/kwargs/partial; tautological identity assertions). **Relaunch: NO-GO.** Round-11
  dispatched (del-before-release; census refresh post-trace + trace_total in the budget;
  callee-identity or runtime-seam guard with extended bite evidence).

- **`adfdd3b` (round 11) verify (xhigh): REQUEST-REVISION** — findings 1 (release-ledger truth) and
  2 (post-trace census + trace_total budget) **CLOSED**; no new production defect. Finding 3 (the
  hand-rolling regression guard) still OPEN: reviewer executed evasions (positional, **kwargs,
  partial, renamed top-level helper, indirect attribute) against `_nested_callables`, and the
  runtime seam test executes neither entry point. **Relaunch: NO-GO** on the guard alone.
  Guard micro-round dispatched to a fresh Coder (the incumbent stalls each turn at ~680k transcript
  tokens): execution-based frame-inspection guard through BOTH entry points, test-only change.
  (Round 11 itself was committed by the Planner after the Coder finished the work but could not
  complete its turn — disclosed in the commit message.)

- **`7da7a66` (guard micro-round, fresh Coder, test-only) micro-verify (xhigh): APPROVE —
  "Relaunch: GO — finding 3 is closed."** All four items CLOSED: the class-level frame seam
  intercepts direct merge, repeated merge and graphdef.apply while thread/generator/exec routes
  cannot forge the required blessed + entry-point frames; both entry-point mutants execute their
  real bodies (≥25 recorded calls, anti-vacuity held); blessed set = exactly `_forward` with the
  secondary assertion preventing silent growth; no production defect. Probe file 111 passed.
  **The S1.5 fix campaign is CLOSED end-to-end** (rounds/commits: e2249e1 → 2b9177d/994af9b →
  46b43c5/7241030/99b5529/ce6717d/2ef9b8a → 6dab9b1 → c052c91/f049f16/adfdd3b/7da7a66; suite
  1,007 → 1,534+... probe-file-count basis 111).

