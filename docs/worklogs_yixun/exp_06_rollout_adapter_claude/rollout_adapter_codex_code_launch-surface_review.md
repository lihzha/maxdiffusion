# rollout_adapter — Codex code review: T6 `launchers` + T7 `fit-probe-mode` (backlog pass 3 of 3)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **BOTH REQUEST-REVISION — 7 BLOCKERs + 3 MAJORs**, all accepted. The harshest pass of the campaign, and it lands on the LAUNCH SURFACE — the last code between a reviewed repo and a TPU job.

**TWO FINDINGS ARE SCIENTIFICALLY CRITICAL, not merely engineering defects:**
1. **T6-1 — the two arms share checkpoint state by default.** Both arms get the same run name and a stable checkpoint root that does not include the arm, so **running R-B then C0 can make C0 restore R-B's parameters, optimizer, step and history**; concurrent arms collide. The existing comparison test proves only that identical supplied environments expand identically — it does not make divergence unconstructible. This would silently destroy the causal comparison Yixun's decision-1 matched control exists to support.
2. **T7-2 — the authorization cell omits the ARM.** `FitCell` is only `(microbatch, k)`, so **a C0 measurement authorizes R-B** and vice versa despite different forward/backward graphs; dtype, remat, logical batch/accumulation, sampling geometry and model revision are likewise unchecked except as caller-stamped globals. An HBM measurement would be applied to a program it never measured.

**The reviewer applied its own pass-2 generalization (STAMPED ≠ BOUND) to the fit probe and found the same flaw class: T7-1** — `publish_authorization` receives measurements separately from caller-supplied SHA/revision/device/geometry, so the digest proves only that those *claims* were not edited afterward; and `assert_cell_authorized` is called **without the current SHA**, revision, device, geometry or recipe. The reviewer executed an authorization carrying a wrong SHA/model/device and production-style assertion accepted it.

**T7-4 — M1 CANNOT BE RUN FROM THIS LAUNCH SURFACE.** `run_fit_probe` does not walk the ladder, invoke a measurement seam, aggregate trials, project or publish — it hides missing ORCHESTRATION, not a device primitive (the same distinction that made the evaluator's `main()` a pass-2 blocker) — **and the launcher has no M1 probe mode at all**, despite the approved plan requiring one.

**T7-3** — contradictory duplicate measurements authorize a refused cell: publication appends each result independently while assertion returns on any authorized occurrence; the reviewer supplied the same cell once fitting and once at 96.9% HBM with a reservation failure, and it was authorized. **T6-2** — `confirm` has no DEV-certificate dependency at the launcher boundary and no phase transports its prerequisites (with separate attempt roots, later phases cannot even locate them). **T6-3** — the resume/attempt-root rules are implemented OPPOSITELY: one mutable checkpoint tree, no SHA-bound adoption of the latest COMPLETE publication into a fresh attempt root, and a caller-supplied `ARTIFACT_ROOT` can remove phase/attempt scoping entirely in both launchers.

**MAJORs:** the "exhaustive" env/default pin is incomplete and **already misses real drift** (`OUTPUT_DIR` defaults to the rollout root in the launcher while the checked-in YAML still defaults to the side-adapter root); the executed-under-bash sandbox is faithful for shell expansion and argv recording but **not for "working launcher"** (the `python` shim never parses the preflight heredoc, pyconfig or entrypoint and returns success even though `src/` and the config are absent) — keep it for argv assertions and add a real-Python integration layer; and `project_wall_clock` assumes checkpoint cadence equals evaluation cadence while accepting negative/nonfinite overheads — the reviewer produced **a finite 6.55-hour projection from negative costs**.

**PASSED:** T6-5 — both worker logs clearly identify selection reconciliation as a designed fail-closed refusal rather than a crash.

## Reviewer output (verbatim)

161,013
## T6 `launchers` findings

1. **BLOCKER — the two arms share checkpoint state by default and are not structurally paired.**  
   [train_wan_pos_rollout.sh:94](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:94) gives both arms the same run name, while [line 132](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:132) derives the same stable checkpoint root without the arm. Running default R-B followed by C0 can therefore make C0 restore R-B’s parameters, optimizer, step, and history; concurrent arms can collide. Moreover, separate invocations remain free to vary every env field. The comparison test at [test_pos_rollout_launcher.py:206](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:206) proves only that identical supplied environments expand identically—it does not make divergence unconstructible.  
   **Change:** create both arm commands from one common argument array/recipe digest, with independently derived arm-specific checkpoint/artifact namespaces. Compare normalized recipes while treating only arm and destination identity as permitted differences.

2. **BLOCKER — `confirm` has no DEV-certificate dependency at the launcher boundary.**  
   [eval_wan_pos_rollout.sh:77](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/eval_wan_pos_rollout.sh:77) accepts `confirm` directly, but the evaluator invocation at [line 145](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/eval_wan_pos_rollout.sh:145) passes no anchor certificate, frozen benchmark row, gate certificate, or prior-phase root. With separate attempt roots, later phases cannot even locate their prerequisites. This is independent of the already-open pass-2 evaluator orchestration blocker.  
   **Change:** require phase-specific, digest/SHA-bound inputs: anchor certificate for benchmark, benchmark row for gates, and the issued passing DEV certificate for confirm. Refuse before prefetch when absent or inconsistent.

3. **BLOCKER — the standing resume and attempt-root rules are implemented oppositely.**  
   [train_wan_pos_rollout.sh:124](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:124) deliberately reuses one mutable checkpoint tree; it neither finds the latest COMPLETE publication nor verifies its SHA nor adopts it into a fresh attempt root. The test at [test_pos_rollout_launcher.py:241](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:241) pins that contradiction. In both launchers, caller-supplied `ARTIFACT_ROOT` can also remove phase/attempt scoping entirely ([train:133](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:133), [eval:75](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/eval_wan_pos_rollout.sh:75)).  
   **Change:** distinguish immutable resume input from fresh attempt output; select only the latest COMPLETE checkpoint whose recorded SHA matches derived running code. Derive final roots internally from a customizable parent, phase, arm, and attempt—do not accept a complete root override.

4. **MAJOR — the claimed exhaustive env/default pin is incomplete and already misses real drift.**  
   `OUTPUT_DIR` defaults to the rollout root at [train_wan_pos_rollout.sh:118](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:118), while the checked-in YAML still defaults to the side-adapter root at [base_wan_5b_pos_rollout.yml:172](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml:172). The “every default” test omits `output_dir`, `run_name`, model, authorization, W&B, and operational keys; the env-mapping table at [test_pos_rollout_launcher.py:133](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:133) also omits several training mappings and nearly all evaluation mappings. D20 is killed specifically, but adjacent dropped mappings survive.  
   **Change:** define an exhaustive launcher interface table and test every emitted override’s env mapping and declared default, with explicit documented exemptions only for derived run identity/storage keys.

5. **MAJOR — the sandbox is faithful for shell expansion/argv recording, not for “working launcher.”**  
   The `python` shim at [test_pos_rollout_launcher.py:44](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:44) never parses the preflight heredoc, pyconfig, or entrypoint and returns success even though `src/` and the config are absent from the sandbox. The setup at [line 57](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_launcher.py:57) also bypasses real git provenance, secrets, activation, and prefetch.  
   **Change:** retain this layer for exact argv assertions, but add a real-Python integration layer that parses the config/overrides and executes the preflight with controlled module stubs; execute both secret/xtrace states and verify real entrypoint/config existence.

## T7 `fit-probe-mode` findings

1. **BLOCKER — authorization provenance is stamped, not bound, and production does not check it.**  
   `publish_authorization` receives measurements separately from caller-supplied SHA/revision/device/geometry at [pos_rollout_fit_probe.py:194](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:194). The digest proves only that those claims were not edited afterward. Worse, [wan_pos_rollout_trainer.py:95](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:95) calls `assert_cell_authorized` without the current SHA—or revision, device, geometry, or recipe. I executed an authorization carrying a wrong SHA/model/device and production-style assertion accepted it. The existing trainer test itself publishes under `"c"*40` and lets the current HEAD reach the model boundary.  
   **Change:** device execution must issue evidence containing provenance derived from the running program, resolved model snapshot, actual devices, tensors/config geometry, and telemetry. Training must derive its current context independently and require exact compatible binding. Validate the whole authorization schema and measured/authorized/refused consistency on load.

2. **BLOCKER — the authorization cell omits the arm and the rest of the footprint-bearing recipe.**  
   [FitCell at pos_rollout_fit_probe.py:73](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:73) is only `(microbatch, k)`. A C0 measurement therefore authorizes R-B with the same pair, or vice versa, despite different forward/backward graphs. Changes to dtype, remat, logical batch/accumulation, sampling geometry, and model revision are likewise unchecked except for caller-stamped global fields.  
   **Change:** authorize `(arm, microbatch, k, recipe_fingerprint)` or bind a complete immutable common-recipe fingerprint globally and arm in each cell. Trainer comparison must include it.

3. **BLOCKER — contradictory duplicate measurements authorize a refused cell.**  
   Publication appends each result independently at [pos_rollout_fit_probe.py:211](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:211), while assertion returns immediately on any authorized occurrence at [line 274](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:274). I supplied the same cell once fitting and once with 96.9% HBM plus a reservation failure; it appeared in both lists and was authorized.  
   **Change:** reject duplicate cells or conservatively aggregate repeated trials—maximum peak and total reservation failures—with any refusal making the cell refused. Require disjoint, unique lists.

4. **BLOCKER — `run_fit_probe` hides missing orchestration, not merely a device primitive.**  
   [pos_rollout_fit_probe.py:295](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:295) does not walk the ladder, invoke a measurement seam, aggregate trials, project, or publish. The launcher also has no M1 probe mode despite the approved plan explicitly requiring one. M1 cannot be run from this launch surface.  
   **Change:** implement host-testable orchestration around a narrow TPU measurer: derive provenance, loop over cells/arms, compile, collect steady-state telemetry, aggregate, project, and publish. Only the actual TPU telemetry adapter may remain device-specific.

5. **MAJOR — wall-clock projection can understate or fabricate cost.**  
   [project_wall_clock:152](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:152) assumes checkpoint cadence equals evaluation cadence even though the launcher exposes independent `CHECKPOINT_EVERY`. It also accepts negative/nonfinite overheads and invalid step/cadence values; I produced a finite 6.55-hour projection using negative eval/checkpoint costs.  
   **Change:** require `checkpoint_every`, calculate counts independently, and validate positive integer steps/cadences plus finite nonnegative overhead measurements.

## Clause verdicts

- **T6-1: FAIL.** Correct argv[1] and arm substitution, but defaults drift, independent envs are unconstrained, and the default arms share checkpoint state.
- **T6-2: FAIL.** Default phase roots are distinct, but overrides can collapse them and no phase prerequisite/certificate is transported.
- **T6-3: FAIL.** Xtrace preservation, declared reads, logging, and the operator note are good; SHA-bound COMPLETE adoption into a fresh attempt root is absent.
- **T6-4: PARTIAL.** Faithful for Bash expansion and recorded argv only; insufficient to imply an operational launcher.
- **T6-5: PASS.** Both worker logs clearly identify selection reconciliation as a designed fail-closed refusal.

- **T7-1: FAIL.** Clean artifacts separate the three cell states and messages, but provenance and measurement are caller-stamped; production accepts another SHA.
- **T7-2: FAIL.** The single-record 90% rule refuses correctly and required projection arguments are good; duplicate trials and projection arithmetic reopen unsafe cells/costs.
- **T7-3: FAIL.** The prose is honest, but `run_fit_probe` omits orchestration, not only TPU-specific implementation.
- **T7-4 rulings:**
  - **90% headroom: ACCEPT.** It is already verbatim in approved plan §4-P1.
  - **Microbatch ladder 8/16/32/64: ACCEPT**, with a recorded rationale: powers-of-two divisors of GBS 256 spanning the default 32. It must be arm/recipe-bound.
  - **`capacity_bytes`: ACCEPT only as device-derived same-run evidence; REJECT as the current caller-supplied publication input.**
  - **Gate order DEV then M1: ACCEPT.** Both are cheap pre-device gates; preserving T4 behavior is reasonable.
- **T7-5: PASS narrowly.** D20 is genuinely killed and the launcher carries `pos_fit_authorization`; the surrounding mapping table is not exhaustive.

## Round verdicts

- **T6 `launchers`: REQUEST-REVISION**
- **T7 `fit-probe-mode`: REQUEST-REVISION**

The prescribed pytest command was attempted, but this managed reviewer sandbox is read-only and pytest could not create `tmp_path`: 12 non-temporary cases passed and 53 cases errored during setup, with no product assertion failure. Bash syntax and Python AST checks passed.
