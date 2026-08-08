# rollout_adapter — Codex re-review: launch-surface rework verification

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: **REQUEST-REVISION — 6 BLOCKERs + 4 MAJORs**, all accepted. Progress acknowledged per-item (arm-scoped roots, root-override removal, worst-of aggregation, negative-cost refusal all landed) but every verdict FAIL on residual depth. **M1-critical subset:** the production M1 entrypoint is **still guaranteed to fail before measuring its first cell**; authorization loading does not revalidate recorded measurements; the 24-key fingerprint omits graph/HBM-bearing configuration; projection arithmetic does not match the run being projected; unresolved/local model provenance must fail closed (the `@no-local-snapshot` fallback REJECTED). **M2/M3-gating subset:** the recipe lock is evadable/non-atomic/deletable; resume publication stamps the arm but does not bind the checkpoint path to it; `publish_attempt` must be wired before any training launch. **M4-gating:** confirm's pre-prefetch check misses SHA/pass fields. Rulings: invented-number measurer acceptable only after a real production adapter exists; dirty-checkout SHA deferral acceptable ONLY because launches require commit/push first — **"the current uncommitted worktree must not launch."**

## Reviewer output (verbatim)

302,527
## Findings

1. **BLOCKER — the recipe lock is evadable and non-atomic.**  
   [train_wan_pos_rollout.sh:205](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:205) locates the lock using the very `OUTPUT_DIR` and `RUN_NAME` values it is meant to compare. A divergent second arm can choose another parent/name and publish a new lock. [publish_recipe_lock:155](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:155) is also check-then-overwrite, so concurrent divergent arms both succeed; deleting the lock likewise lets the second arm republish. I executed all three cases successfully. Runtime-affecting inputs such as `LIBTPU_INIT_ARGS` are outside the normalized config too.  
   **Change:** identify the pair with a stable lock URI independent of bound recipe values, validate `RUN_NAME` as a segment, include relevant runtime inputs, and use atomic create-if-absent/CAS storage semantics. Persist the adopted lock digest in each arm publication so deletion cannot reset the pair.

2. **BLOCKER — `confirm` still reaches prefetch without a demonstrably passing, SHA-bound DEV certificate.**  
   [eval_wan_pos_rollout.sh:209](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/eval_wan_pos_rollout.sh:209) checks SHA only when `code_sha` happens to exist, and [line 212](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/eval_wan_pos_rollout.sh:212) rejects `passed: false` but accepts a DEV artifact with no `passed` field. Thus `{"protocol":"anything"}` satisfies the launcher’s shape check.  
   **Change:** require the exact expected artifact type, mandatory provenance binding, and `passed is True` before prefetch. Coordinate a mandatory SHA/digest-bearing envelope if the evaluator-owned payload currently lacks it.

3. **BLOCKER — resume publication stamps the arm but does not bind the checkpoint path to it.**  
   [publish_attempt:195](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:195) accepts an arbitrary `checkpoint_dir`; [load_publication:237](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:237) checks only that it is nonempty. I placed a digest-valid `arm="rollout"` publication under the rollout parent that pointed to a `one_step/.../checkpoints` tree, and [select_resume_publication:265](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:265) adopted it.  
   **Change:** derive—not accept—`<parent>/<attempt>/checkpoints`, require the payload attempt to equal its directory, verify containment and checkpoint existence/completeness, then select it.

4. **BLOCKER — the 24-key recipe fingerprint still omits graph- and HBM-bearing configuration.**  
   [FOOTPRINT_KEYS:127](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:127) omits at least `action_tokens`, `action_dim`, `action_len`, `pre_context_tokens`, `flash_block_sizes`, sharding rules, `latent_frames`, and `latent_channels`. These feed the actual adapter construction at [wan_ti2v_side_adapter_trainer.py:297](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py:297). I verified that changing `action_tokens`, `pre_context_tokens`, `flash_block_sizes`, or `latent_frames` leaves the current context digest unchanged.  
   **Change:** fingerprint a canonical complete graph/shape/sharding recipe, with a reviewed allowlist of exclusions limited to the cell and non-semantic destinations.

5. **BLOCKER — authorization loading does not revalidate the recorded measurements.**  
   [load_authorization:743](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:743) validates only the three cell lists. It ignores the `measurements`, their context digests and verdicts, projections, and headroom constant. I changed an authorized cell’s recorded measurement to capacity-level peak plus a reservation failure, recomputed the unkeyed digest, and both loading and [assert_cell_authorized:789](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:789) accepted it.  
   **Change:** deserialize and validate every measurement, require one aggregated record per measured cell, recompute every verdict/list and projection, and enforce the pinned headroom value.

6. **BLOCKER — the production M1 entrypoint is still guaranteed to fail.**  
   [measure_cell_on_device:825](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:825) unconditionally raises `NotImplementedError`, while it is the default measurer used by [run_fit_probe:837](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:837) and `main()`. Hardware-specific code may be untestable on the host; it cannot be absent at the last gate before launching that exact job.  
   **Change:** implement the real compile/steady-state/memory/capacity/overhead/reservation telemetry adapter and execute the actual M1 entrypoint through a controlled backend integration test.

7. **MAJOR — the “real-Python” launcher test still does not execute the real entrypoint or real config parser.**  
   The shim exits successfully for every non-heredoc invocation at [test_pos_rollout_launcher.py:80](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:80). The claimed pyconfig check at [line 1068](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:1068) is a hand-written type conversion, not `pyconfig.initialize`. Consequently the integration test passes while the real M1 entrypoint necessarily raises finding 6.  
   **Change:** run real `pyconfig.initialize` and real `main`/dispatch, replacing only the actual device backend.

8. **MAJOR — projection arithmetic does not match the run being projected.**  
   [project_wall_clock:566](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:566) floors cadence counts. The loop evaluates at cadence **and at the final step** at [pos_rollout_loop.py:154](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_loop.py:154): for 1,001 steps and cadence 1,000, projection says one evaluation while production performs two. Moreover, `checkpoint_every` is not consumed anywhere by `LoopSchedule`; checkpoints are currently written on evaluations at [line 480](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_loop.py:480).  
   **Change:** count final events correctly and either wire the independent checkpoint cadence into the loop or stop projecting/exposing a cadence production ignores.

9. **MAJOR — the interface audit remains incomplete for operational inputs and derived destinations.**  
   [TRAIN_INTERFACE:232](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:232) covers emitted overrides, but not `POS_JOB_MODE`, `ATTEMPT`, `COMMIT`, `POS_CHECKPOINT_ATTEMPT`, `LIBTPU_INIT_ARGS`, or other operational inputs that select code, provenance, roots, or runtime behavior. Also, shared `output_dir`/`run_name` cause pyconfig to derive identical TensorBoard and metrics destinations for both arms at [pyconfig.py:204](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pyconfig.py:204).  
   **Change:** add a complete external-input/derived-output table and make every arm/attempt-writing destination distinct.

10. **MAJOR — unresolved/local model provenance is not conservatively bound.**  
    [derive_model_revision:361](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:361) identifies any local directory only as `@local-dir`, irrespective of contents, and [line 368](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:368) turns resolution failure into `@no-local-snapshot:<exception>`. Two unresolved or mutated models can therefore compare equal.  
    **Change:** require an immutable resolved revision; hash/manifest local snapshots and fail closed when a remote snapshot cannot be resolved.

## Verification notes

- The worktree is at `f4575df`, not stated `e3656c9`; `e3656c9` is its parent. The intervening commit changes only the worklog, but shared uncommitted files include the later evaluator work.
- The exact pytest command ran, but this managed session has no writable temporary directory: **42 passed, 151 setup errors, 0 assertion failures**. The stored prior run reports 1,951 passing tests, but I could not independently reproduce it.
- The supplied harness currently contains **47**, not 32, attacks. `_report` does wrap individual attack invocations, but the outer `TemporaryDirectory` is unwrapped, so it crashed there in this environment. There is no preserved source baseline from which to prove byte-for-byte that the original attack bodies were unchanged.
- `bash -n`, Python AST parsing, YAML parsing, and `git diff --check` passed.

## Verdicts on requested items

1. **T6-1: FAIL.** Arm checkpoint roots are separated, but the cross-submission lock is routable around, deletable, and racy.
2. **T6-2: FAIL.** Required paths are transported, but missing SHA/pass fields survive the pre-prefetch check.
3. **T6-3: FAIL.** Root overrides and attempt traversal are fixed; checkpoint-path binding is not. An uncalled `publish_attempt` is safe for M1 only because it produces no false resumable state, but resume is not operational for M2/M3.
4. **MAJOR-a: FAIL.** YAML `output_dir` and emitted override mappings are fixed; operational inputs and derived destinations remain outside the audit.
5. **MAJOR-b: FAIL.** Real heredocs run, but the actual config parser and entrypoint still do not.
6. **T7-1: FAIL.** Production now compares contexts, but unresolved/local model identity is not immutable, and the production evidence generator does not exist.
7. **T7-2: FAIL.** Arm identity is fixed; the global fingerprint does not cover the remaining footprint-bearing recipe.
8. **T7-3: FAIL.** Worst-of aggregation and list uniqueness/disjointness pass, but loading does not recompute verdicts from the recorded measurements.
9. **T7-4: FAIL.** Host orchestration is substantially improved, but the launched default path still raises before measuring its first cell.
10. **MAJOR-c: FAIL.** Negative/nonfinite costs and separate arguments are fixed; event counts and production checkpoint behavior remain inconsistent.

Residual rulings:

- Invented-number measurer: acceptable only as a host-test seam after a real, fixed production adapter exists; not acceptable in the present launchable state.
- Dirty checkout SHA: acceptable as a procedural deferral only because the standing grant explicitly requires commit/push first. The current uncommitted worktree must not launch.
- `derive_model_revision` fallback: not acceptable; it must fail closed or produce a content-bound revision.
- Uncalled `publish_attempt`: safe ordering for M1, but must be wired and path-bound before any training launch.

# REQUEST-REVISION
