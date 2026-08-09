# rollout_adapter — Codex re-ruling: W3 verification + M1

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **W3 REQUEST-REVISION · M1 NOT-READY** — minimal set: *"make the shared program own the complete compiled-input contract — production batch sharding, explicit replicated optimizer state, and the actual DEV scorer — then rerun an oracle that lowers each path using ITS OWN inputs and asserts ABSOLUTE expected shardings."*

- **BLOCKER:** M1's synthetic batch arrays are single-device while production's loader returns `NamedSharding(mesh, P('data','fsdp','context','tensor'))` (`multihost_dataloading.py:39`); the shared jit declares no `in_shardings`, so each path specializes on its actual arguments — **and the oracle lowered BOTH sides with M1's inputs, proving equal lowering on identical probe inputs rather than equality with the live trainer's input program.**
- **MAJOR:** optimizer state replication is equality-only (params are placed on `P()`; `optimizer.init` output is not, and scalar/count leaves are not guaranteed to inherit) — every leaf needs explicit placement AND an absolute assertion. **MAJOR:** M1 discards the shared DEV scorer for a private scalar-only jit whose `[0]` lets XLA PRUNE the aux computation, understating the eval component of the wall projection — time the shared scorer itself and compare its lowering too. **LOW:** `authorized_cell` still not literally `body[0]` and the AST test checks relative order only.
- MAJOR-2/MAJOR-3 from the prior pass verified PASS; the harness evidence gap CLOSED (78 unique probes statically reconciled, zero SUCCEEDED lines). W1's peak-floor ruling stands closed. The reviewer could not execute the oracle (no JAX in its image) and says so; the finding is source-demonstrable regardless.

## Reviewer output (verbatim)

207,548
# Findings

1. **BLOCKER — M1 still finalizes against a different batch sharding.**  
   M1 creates ordinary single-device arrays at [pos_rollout_fit_probe.py:1312](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1312), whereas production’s loader returns `NamedSharding(mesh, P('data','fsdp','context','tensor'))` arrays via [multihost_dataloading.py:39](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/multihost_dataloading.py:39). The shared JIT declares no `in_shardings` at [pos_rollout_update.py:323](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:323), so each path specializes on its actual arguments.

   The oracle misses this because both sides lower using M1’s `right.batch/right.draws` at [test_pos_rollout_trainer_wiring.py:1215](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1215). It proves equal lowering on identical probe inputs, not equality with the live trainer’s input program.

   **Change:** make the shared program own/enforce the production batch-sharding contract; place M1’s synthetic inputs accordingly. Lower each side with its own path-produced inputs and assert the expected production input specs.

2. **MAJOR — replicated optimizer state is neither fully enforced nor absolutely asserted.**  
   Parameters are explicitly placed on `P()` at [pos_rollout_update.py:319](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:319), but `optimizer.init()` follows with no tree-wide placement at line 322. Scalar/count state is not guaranteed to inherit parameter sharding. The oracle’s absolute `replicated` check covers only parameters at [test_pos_rollout_trainer_wiring.py:1231](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1231); optimizer sharding is only compared side-to-side.

   **Change:** explicitly place every optimizer-state leaf on `NamedSharding(mesh, P())` and assert that absolute property for every leaf.

3. **MAJOR — M1 discards the shared DEV scorer and compiles a private one.**  
   The shared builder creates a scorer at [pos_rollout_update.py:324](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:324), but M1 instead defines and jits a private scalar-only scorer at [pos_rollout_fit_probe.py:1329](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1329). Training’s scorer returns `(loss, aux)`; M1’s `[0]` permits XLA to prune the auxiliary norms/statistics, understating the evaluation component of the wall projection.

   **Change:** expose and time the shared scorer itself; compare its lowering as well as the update lowering.

4. **LOW — `authorized_cell` is still not literally first, and the AST test does not pin that claim.**  
   `start_training()` begins with `context = self.running_context()` at [wan_pos_rollout_trainer.py:601](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:601), followed by `authorized_cell`. The test at [test_pos_rollout_trainer_wiring.py:965](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:965) checks relative ordering, not index zero.

   **Change:** make the authorization call the first executable statement and assert `body[0]` accordingly.

# Verdicts on items 1–4

1. **Blocker: FAIL.** The model/context/factory boundary is substantially improved, but path-specific batch inputs still finalize different programs; the committed oracle substitutes M1 inputs on both sides. Optimizer state and the scorer also escape the claimed complete boundary.

2. **Prior findings:**

   - MAJOR 2: **PASS** — every adapter and optimizer leaf is compared with `np.array_equal`; null context and sigmas are compared; the context is asserted non-zero.
   - MAJOR 3: **PASS** — the oracle retains `seam.transformer`, asserts `backbone.transformer is seam.transformer`, and compares that object’s leaves across training.
   - LOW 4: **FAIL** — see Finding 4.

3. **Coder limitation: INSUFFICIENT.** Z04’s built-inside-scope assertion and the scheduler-source/value tests are useful, and parameter replication has an absolute check. But optimizer-state replication remains equality-only, production batch sharding is absent from the oracle, and the private scorer is not compared. Absolute correctness is therefore still partly oracle-only—or entirely omitted.

4. **New W3 defects: FAIL.** Growing the tiny meshes from two axes to the configuration’s four axes is genuine evidence that axis rules were previously absent. It did not expose the remaining unsharded-world dependency: the test loader/probe arrays are unsharded, and the multi-device oracle lowers both sides with those probe arrays.

The prior harness availability gap is closed: the repository contains the source, README, and log; static reconciliation found 78 unique `_report` calls, all 78 labels logged, and zero `SUCCEEDED` lines.

# THE RULING

**NOT-READY-FOR-M1.**

Minimal blocking set: make the shared program own the complete compiled-input contract—production batch sharding, explicit replicated optimizer state, and the actual DEV scorer—then rerun an oracle that lowers each path using its own inputs and asserts absolute expected shardings.

The W1 peak-floor ruling remains closed. M2/M3’s recipe-lock/publication work and v2.8’s restart-both-arms rule remain outside this M1 blocker.

I attempted the committed oracle, but this reviewer image has neither `pytest` nor JAX/project dependencies, so it could not execute here. Static checks passed: `git diff --check`, both launcher `bash -n` checks, and Python AST parsing. The supplied 2112/0 and 14/14 results do not cover the source-demonstrable oracle hole above.

# W3

**REQUEST-REVISION.**
