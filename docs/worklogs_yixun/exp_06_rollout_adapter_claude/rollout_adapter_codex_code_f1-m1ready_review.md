# rollout_adapter — Codex review: Round F1 verification + M1 launch-readiness

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdicts: **Round F1 REQUEST-REVISION; Part B: NOT-READY-FOR-M1** — minimal blocking set of three. A2 (evidence re-decision) / A3 (inverted fingerprint, schedule exclusions explicitly endorsed) / A4 (cadence counts) / A6 (subprocess layer, stub set ruled minimal and honest) all PASSED.

**THE HEADLINE FINDING — the campaign's most consequential catch: `base_wan_5b_pos_rollout.yml` declares `action_adapter_type: side_adapter`. M1 would have measured the WRONG ADAPTER (~240M side_adapter) while the experiment trains the unchanged ~128M pre_context.** Lineage: the config was generated from the side-adapter YAML and inherited its default; exp_05's S8 set `pre_context` explicitly in its own YAML; **the Planner's T4 brief listed nine pilot-critical defaults and omitted this one**, so the generation test faithfully preserved the wrong value and T4's review — checking the listed defaults — passed it. The same brief-omission failure mode as the terminal-resume miss, now recorded twice.

Also blocking: the timed "step" is one MICROBATCH retained from a full product (`[0]`), so projections understate 4–32× at GBS 256 and accumulation-state HBM is never resident (plus a bare `optax.adamw` substituted for the configured optimizer; eval timed at the wrong unit vs the batch-one DEV instrument; checkpoint timed to a local tmpdir); and peaks are LIFETIME high-water marks never reset between load/compile/warmup/32 trials — later cells inherit earlier peaks. MAJORs: substring `"OOM"` classification (the reviewer's `"boom in program build"` and `"No room left on device"` both classified as HBM refusals); local model identity metadata-bound (`(relpath,size)` misses in-place byte changes). MINOR: "byte-identical reconstruction" is mapping equality (JSON `2.0` == `2`).

**Part B's clean rulings, for the record:** F2's open findings are NOT on the probe path (fit-mode resume selection is print-only); the evaluator modules do not execute on it (importing the DEV instrument is not importing the evaluator); the trainer gate and decode/NaN fixes import cleanly and are compatible. After revision: commit-first still required; v2.7's M1′ topology rule stands.

## Reviewer output (verbatim)

261,648
## Findings

1. **BLOCKER — M1 measures the wrong adapter architecture.**  
   [base_wan_5b_pos_rollout.yml:143](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml:143) declares `action_adapter_type: side_adapter`, while the approved experiment requires the unchanged `pre_context` adapter. `ProductionModelSource` consumes this value directly. The generation test at [test_pos_rollout_dispatch.py:78](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_dispatch.py:78) incorrectly preserves it.  
   **Change:** set `action_adapter_type: pre_context`; update the generated-config substitutions and add it as a plan-critical pinned default.

2. **BLOCKER — the timed “step” is one microbatch, not one logical optimizer update.**  
   [pos_rollout_fit_probe.py:1177](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1177) correctly produces all microbatches, but lines 1220–1221 retain only `[0]`. Consequently `step_seconds` times one microbatch while `project_wall_clock` multiplies it by logical updates. At GBS 256 this understates computation by factors of 4–32 and omits accumulation-state HBM. The probe also substitutes a bare `optax.adamw`, ignoring the configured schedule, clipping, and optimizer options.

   Evaluation and checkpoint costs are likewise non-production units: evaluation times a microbatch and scales by `ceil(64/microbatch)`, although [pos_rollout_dev_instrument.py:284](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_dev_instrument.py:284) scores DEV batch-one; checkpoint timing writes to a local temporary directory rather than the configured GCS destination.  
   **Change:** share one production logical-update primitive between trainer and probe, accumulating every microbatch before one optimizer update; measure batch-one DEV-64 execution and the production checkpoint payload/storage class.

3. **BLOCKER — reported peaks are neither steady-state nor cell-local.**  
   [pos_rollout_fit_probe.py:1084](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1084) reads the runtime lifetime high-water mark. Nothing resets it between model loading, compilation, warmup, or the 32 sequential cell trials. Later cells therefore inherit earlier peaks, and transient pipeline-loading/compilation peaks are reported as steady-state.  
   **Change:** isolate each trial or reset telemetry immediately before the steady-state region using a supported backend facility; fail closed if a per-cell steady-state peak cannot be obtained.

4. **MAJOR — unrelated failures are misclassified as HBM exhaustion.**  
   [pos_rollout_fit_probe.py:1105](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1105) uses substring `"OOM"`. Fresh probes showed both `"boom in program build"` and `"No room left on device"` classified as resource exhaustion. Such model or disk failures become apparently measured HBM refusals and M1 can exit successfully.  
   **Change:** prefer structured backend status codes; otherwise require bounded exact phrases such as `RESOURCE_EXHAUSTED`, `OUT OF MEMORY`, or word-boundary `OOM`.

5. **MAJOR — local model identity is metadata-bound, not content-bound.**  
   [pos_rollout_fit_probe.py:424](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:424) hashes only relative paths and sizes. An in-place byte change, or replacement with another same-shaped checkpoint, leaves authorization unchanged. This can include same-length transformer configuration changes that alter the graph.  
   **Change:** hash file bytes once, use an immutable model-provided digest, or disallow arbitrary local directories and require a validated snapshot commit. The remote snapshot branch is acceptable.

6. **MINOR — “byte-identical reconstruction” is not literal.**  
   [pos_rollout_fit_probe.py:986](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:986) uses Python mapping equality. I changed `step_seconds` from JSON `2.0` to `2`, rehashed, and loading accepted it. This does not change the verdict, but contradicts the claim.  
   **Change:** compare canonical serialized bytes with strict field types, or describe the guarantee as semantic field/value equality.

## A1–A6 verdicts

- **A1 — FAIL.** Real parsing, orchestration, JIT, Orbax, and telemetry calls exist, but they measure the wrong adapter and the wrong train/eval/checkpoint units; peak telemetry is contaminated, and non-OOM failures can become refusals.
- **A2 — PASS, with finding 6.** `ProbeEvidence` has exactly the three claimed fields. I reproduced the edited-and-rehashed peak/reservation attack; it is refused. Headroom, projections, verdicts, lists, context binding, and one-record-per-cell are recomputed.
- **A3 — PASS.** Default inclusion works; the previously invisible graph/HBM keys and fresh nested keys move the digest. Canonical nested rendering works. Cell, destination, and schedule exclusions are appropriate—especially `max_train_steps`, which must not prevent M2-length evidence from transferring to the M3-length recipe.
- **A4 — PASS.** The six event-count cases match `should_evaluate`, including the final step. Refusing unequal checkpoint/evaluation cadences is the correct F1-scoped decision.
- **A5 — FAIL.** Remote resolution closes correctly; local `(relpath,size)` does not meet the claimed immutable/content-bound identity.
- **A6 — PASS.** The subprocess test reaches the real parser and entrypoint. The four import-only dependency shims and the two explicit hardware/weights seams are minimal and honestly described. They do not, however, cure A1’s production-fidelity defects.

## Part B — M1 launch readiness

**NOT-READY.**

Minimal blocking set:

1. Correct the YAML to build the approved `pre_context` adapter.
2. Measure a real logical accumulated update, the actual DEV batch-one evaluation unit, and the production checkpoint unit.
3. Obtain an isolated steady-state peak and stop treating arbitrary `"OOM"` substrings as allocation refusals.

The open F2 findings are not otherwise executed by probe mode: recipe-lock publication, attempt publication/adoption, and confirm-certificate logic are training/evaluation concerns. Fit-mode resume selection in the shell is print-only. Likewise, none of the M4 evaluator modules executes on this path; importing the DEV instrument is not importing the evaluator. The trainer gate and the decode/NaN fixes import cleanly and are compatible with the authorization format, but are not invoked by M1.

After revision, the launch still requires committing and pushing the presently uncommitted worktree. The v2.7 topology rule also remains: v6e-8 M1 authorizes M2 only; M3 needs the separate v6e-64 M1′.

The named scratchpad logs were absent from this checkout, and this sandbox lacks the test environment, so I could not independently reproduce `2010/24/53`. `bash -n` and `git diff --check` passed; the focused in-memory authorization, fingerprint, cadence, and OOM-classification probes ran as described.

## Overall

**REQUEST-REVISION for Round F1.**
