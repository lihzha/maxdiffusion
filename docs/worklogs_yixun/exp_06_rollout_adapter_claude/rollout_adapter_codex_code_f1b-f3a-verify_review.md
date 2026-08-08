# rollout_adapter — Codex review: F1b + F3a verification + M1 re-ruling

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **F1b REQUEST-REVISION (A2/A3/A5 FAIL; A1/A4/A6 PASS) · F3a REQUEST-REVISION (C1/C2 PASS; C3 FAIL on a loader defect) · M1: NOT-READY, minimal set of two.**

**The reframing finding (A2):** accumulation, optimizer behaviour, batch-one×64 timing and checkpoint payload are all individually correct — but "the probe measures the trainer's step" is FALSE because **the production trainer is still unwired** (`start_training` raises `NotImplementedError`; only the probe imports `pos_rollout_update`). The deferred model/data wiring round is therefore the actual M1 blocker: wire `WanPosRolloutTrainer.start_training()` through the SHARED factories (model/adapter with production dtype args, `build_optimizer`, `build_logical_update`) adapted to `run_loop`'s state signature, with a call-graph test proving M1 and the live trainer call the SAME factories; M1 runs at that final SHA.

**A3 FAIL:** the compiled-memory-analysis fallback is a FLOOR that can authorize (the test institutionalized a 30-GiB standing mark + 7-GiB analysis reporting 7 GiB); for a ≤90% rule a lower bound is non-conservative ⇒ persist `peak_source`, reject analysis-only evidence at `cell_verdict` (or subprocess isolation). **A5 FAIL:** the manifest hash lacks length framing (`a=Xb\0Y` vs `a=X,b=Y` collide as chosen serializations) ⇒ length-framed records or `(path, file_sha256)`. **A1 PASS with a whole-YAML sweep finding no further plan-§3 contradiction.** A4/A6 PASS on re-executed probes.

**C3 FAIL — a genuine production-path bug of the issue-#11 family:** `load_device_backend` reads UNDECLARED `config.num_train_timesteps` (raises on the real HyperParameters), where deployment reads `scheduler.config.num_train_timesteps` ⇒ the real loader dies before any anchor execution or grid enforcement. Fix: derive from the loaded pipeline scheduler as deployment does, then immediately `assert_deployed_grid`; add a loader-composition/config-contract test. C1 (bf16 boundary incl. the self-found asymmetry, verified against deployment) and C2 (per-layer finiteness isolation) PASS.

**Launch-package adjudications:** the probe's scoped checkpoint write = acceptable caveat (runbook must verify/remove leftover `_m1_probe` prefixes); batch-one×64 arithmetic = fine as-is; **the peak floor = an M1 acceptance BLOCKER, not a caveat**; commit/push the final reviewed code before M1 — never launch the dirty tree.

## Reviewer output (verbatim)

387,010
# Review outcome

**M1: NOT-READY**

- **F1b: REQUEST-REVISION**
- **F3a: REQUEST-REVISION**

The supplied 2053/0 suite and mutation/harness results are accepted, but they do not exercise three production call-path defects below.

## Part A — Round F1b

### Findings

1. **BLOCKER — A2: the measured update still is not the trainer’s update.**

   [`build_logical_update`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:69) correctly averages every equal-width microbatch before one optimizer application, matching the stream split and T3b-2 accumulation contract. The optimizer also correctly comes through `max_utils`.

   However, the live trainer explicitly says model/state/optimizer wiring is absent at [`wan_pos_rollout_trainer.py:31`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:31) and still raises `NotImplementedError` at [`:398`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:398). The loop accepts an arbitrary injected `update_fn` at [`pos_rollout_loop.py:389`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_loop.py:389). Repository-wide, only the fit probe and its tests import `pos_rollout_update`.

   The recording monkeypatch at [`test_pos_rollout_fit_probe.py:1863`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_fit_probe.py:1863) therefore proves that the **probe** calls the builder, not that production does.

   The probe also manually reconstructs the adapter at [`pos_rollout_fit_probe.py:1369`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1369), omitting the explicit dtype/weights-dtype/precision arguments used by production at [`wan_ti2v_side_adapter_trainer.py:289`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py:289). Current pinned defaults happen to agree, but this is not an exact shared construction.

   **Concrete change:** wire `WanPosRolloutTrainer.start_training()` through the shared model/adaptor factory, `build_optimizer`, and `build_logical_update`, adapting it to `run_loop`’s state signature. Add an integration/call-graph test proving both M1 and the live trainer call those same factories. M1 must then run at that final SHA.

2. **BLOCKER — A3: compiled-memory analysis remains an authorization-capable floor.**

   [`end_steady_state`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1153) accepts `memory_analysis()` alone whenever a no-reset backend’s cell does not raise the standing HWM. But [`_program_bytes`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1521) correctly admits this number is a floor that excludes closed-over weights.

   The test at [`test_pos_rollout_fit_probe.py:1616`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_fit_probe.py:1616) institutionalizes the unsafe case: a 30-GiB standing mark plus a 7-GiB program analysis reports 7 GiB. [`cell_verdict`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:611) can authorize that value, and [`CellMeasurement`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:522) does not retain the source.

   For a “peak ≤90%” rule, a lower bound is non-conservative. Reset + attribution is sufficient only when an attributable runtime peak exists; the analysis-only path is not.

   **Concrete change:** either isolate cells in fresh subprocesses, or refuse authorization unless every device reset successfully or the region raised its prior mark. Persist `peak_source` in measurements/artifacts and reject `compiled memory analysis`-only evidence at `cell_verdict`.

3. **MAJOR — A5: the local byte hash has ambiguous record framing.**

   [`snapshot_manifest_digest`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:430) hashes `path + NUL + contents` without a file count or content length. Consequently:

   - one file `a` containing `Xb\0Y`
   - two files `a=X`, `b=Y`

   feed the identical byte stream `a\0Xb\0Y` into SHA-256. This is a chosen serialization collision, not a SHA-256 break.

   The same-size byte-swap behavior itself is fixed, and the 2-GiB fail-closed ceiling is reasonable because the production launcher uses an immutable remote snapshot.

   **Concrete change:** hash canonical length-framed records—file count plus path length/path/content length/content—or hash canonical `(path, file_sha256)` entries.

### A1–A6 verdicts

| Item | Verdict | Assessment |
|---|---|---|
| **A1** | **PASS** | `pre_context` is locked in YAML at [`:148`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml:148), as the fourth substitution at [`test_pos_rollout_dispatch.py:84`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_dispatch.py:84), first plan-critical default at [`:132`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_dispatch.py:132), and fingerprint input at [`test_pos_rollout_fit_probe.py:1529`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_fit_probe.py:1529). Whole-YAML comparison found no further plan-§3 model-default contradiction. |
| **A2** | **FAIL** | Accumulation, optimizer behavior, batch-one ×64 timing, and checkpoint payload/path are individually correct; exact trainer-step identity is false because production is unwired. |
| **A3** | **FAIL** | Reset/raised-HWM paths are attributable; analysis-only fallback can falsely authorize. |
| **A4** | **PASS** | `"boom"` and `"No room left on device"` return false; storage `OSError("OOM…")` returns false; device-allocation `RuntimeError("OOM…")` returns true. |
| **A5** | **FAIL** | Same-size byte mutation is detected and the ceiling is acceptable, but record framing permits local-tree identity collisions. |
| **A6** | **PASS** | Canonical serialized-byte comparison at [`pos_rollout_fit_probe.py:1007`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1007) rejects `2.0 → 2` retyping. |

**Round F1b verdict: REQUEST-REVISION.**

## Part B — M1 launch-readiness

**NOT-READY**

Minimal blocking set:

1. Bind M1’s logical update/model construction to the live production trainer, which must no longer terminate at `NotImplementedError`.
2. Prevent compiled-memory-analysis floors from authorizing cells; require attributable runtime peaks and retain their source.

Launch-package adjudication:

- The real checkpoint write under `checkpoint_dir/_m1_probe` is an acceptable operational caveat. Cleanup is best-effort, so the runbook should verify/remove leftover probe prefixes.
- Batch-one evaluation timing multiplied by 64 is the correct DEV-64 arithmetic and need not become a separate acceptance criterion.
- The peak-floor issue is **not** merely a caveat; it is an M1 acceptance blocker.
- The present worktree is uncommitted while provenance records only `git HEAD`. Commit/push the final reviewed code before M1; never launch this dirty tree.

## Part C — Round F3a

### Finding

1. **BLOCKER — C3: the production evaluator loader reads an undeclared configuration key.**

   [`load_device_backend`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1426) constructs the live grid with `config.num_train_timesteps` at line 1431. No rollout YAML declares that key, and [`HyperParameters.__getattr__`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pyconfig.py:316) raises for undeclared keys.

   Deployment obtains it from `scheduler.config.num_train_timesteps` at [`generate_wan_side_adapter.py:121`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/generate_wan_side_adapter.py:121). Therefore the default real `load_device_backend` path fails before any anchor execution or grid-digest enforcement.

   **Concrete change:** derive timestep count and sigma limits from the loaded pipeline scheduler, as deployment does, then immediately run `assert_deployed_grid`; alternatively use `deployed_grid()` after explicitly verifying the scheduler matches its pinned constants. Add a loader-composition/config-contract test.

### C1–C3 verdicts

| Item | Verdict | Assessment |
|---|---|---|
| **C1** | **PASS** | [`DeviceBackend.score`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1356) casts all four inputs before noise. `_dtype(config.weights_dtype)` is used at line 1450. Phase-level bf16 parity is non-vacuous. Deployment asymmetry is reproduced: latent MSE uses cast `z_video`, while pixel/SSIM uses original float32 ground truth, matching [`generate_wan_side_adapter.py:116`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/generate_wan_side_adapter.py:116) and [`:347`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/generate_wan_side_adapter.py:347). |
| **C2** | **PASS** | Per-sample, aggregate, anchor-mean, and deviation finiteness are independently guarded. The finite-value aggregate overflow and `1.7e308` deviation tests genuinely bypass earlier layers; the swapped still-terminal-zero grid isolates monotonicity. |
| **C3** | **FAIL** | Grid-by-value construction, digest, execution→row→Measurement→certificate propagation, summary/anchor rechecks, and the flow-shift-3 killer mutant are correct for injected backends. The live loader defect prevents the production path from reaching them. |

The declared downstream boundary is confirmed: [`SCORE_ROW_FIELDS`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1026) omits `grid_sha256`, and [`pos_rollout_gates.py:527`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:527) drops the execution grid when building rows. That remains F3b/F3c scope, not an additional F3a charge.

**Round F3a verdict: REQUEST-REVISION.**

Static integrity checks passed: `git diff --check` and launcher `bash -n`. The exact pytest reruns could not create temporary paths in this read-only review sandbox; the supplied authoritative 2053/0 result remains the suite result of record.
