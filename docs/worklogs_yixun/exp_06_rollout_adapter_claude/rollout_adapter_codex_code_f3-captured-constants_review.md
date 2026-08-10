# exp_06 F3 `captured-constants` — Codex code review

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-10 (~8.5 h pass; executed its attacks against the tiny Wan stack, reproduced the guard's red-side byte counts independently, and traced the production-shaped evaluator form). Verdict: **REQUEST-REVISION** — the M1 trainer/scorer capture is FIXED (no BLOCKER); 3 MAJOR + 1 MINOR on the freeze claim, the evaluator seam, and guard blind spots.

## BLOCKER

None found in the trainer/M1 argument-threading path itself. The 10.18 GB training/scorer capture appears removed.

## MAJOR

1. The backbone is still differentiable; keyword-only does not make differentiation “unspellable.”

   [pos_rollout_step.py:167](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_step.py:167) accepts `frozen_state` keyword-only, but a wrapper can trivially make it the differentiated positional argument:

   ```python
   jax.grad(
       lambda state: make_velocity_fn(
           params, frozen_state=state, actions=actions, guide_scale=5.0
       )(hidden, timestep, context).sum()
   )(frozen.state)
   ```

   I ran this against the repository’s tiny Wan stack: it produced 42 frozen gradient leaves with a nonzero aggregate norm of approximately 2209.46.

   The production update currently remains safe because [pos_rollout_update.py:113](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:113) differentiates default `argnums=0`. However, an outer `grad`, direct builder caller, or `nnx.grad` wrapper can allocate a full backbone gradient tree. The optimizer assertion at [pos_rollout_update.py:468](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:468) runs only during program construction and cannot observe such a wrapper.

   The signature-only test at [test_pos_rollout_captured_constants.py:482](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_captured_constants.py:482) therefore proves syntax, not nondifferentiability. Apply `stop_gradient` leafwise to `frozen_state` at the builder/loss boundary and replace this with an adversarial gradient test.

2. The evaluator guard tests a safer calling convention than production.

   Production’s [velocity_for closure](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1427) closes over `frozen.state`. The new test instead gives `frozen_state` explicitly to its test-local outer JIT at [test_pos_rollout_captured_constants.py:388](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_captured_constants.py:388).

   I traced the production-shaped form—`velocity_for` closes over `frozen.state`, then an outer JIT calls `cfg_rollout`. It captured **4,373,412 bytes** for a marked backbone containing **4,372,352 bytes**. Thus the new test is green while the actual evaluator-shaped compiled form is red; on the real model this recreates the approximately 10 GB failure.

   The explanation at [eval_wan_pos_rollout.py:1417](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1417) is also incorrect: `DeviceBackend.score` reaches [cfg_rollout’s `lax.fori_loop`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_step.py:219), whose body is staged and compiled. The evaluator is not an entirely op-by-op, non-lowering path.

3. The fake-model budget can miss production-scale partial captures.

   The guard uses a one-layer, 16-wide model and a 1 MB allowance at [test_pos_rollout_captured_constants.py:65](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_captured_constants.py:65). Only the FFN is enlarged as the mark. Production is 40 layers and 5120-wide at [base_wan_5b_pos_rollout.yml:125](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/configs/base_wan_5b_pos_rollout.yml:125).

   A 16×16 fp32 projection is only 1 KB in the fake model, while its 5120×5120 bf16 production counterpart is 50 MiB. Capturing attention subsets across 40 layers can therefore remain under the fake 1 MB budget while producing multi-GB production literals.

   The closure detector has another blind spot at [test_pos_rollout_captured_constants.py:130](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_captured_constants.py:130): `FrozenBackbone` is an unregistered dataclass, so `jax.tree.leaves(frozen)` treats the object as opaque and reports zero even though `frozen.state` contains all weights.

## MINOR

1. Frozen-state sharding is implemented but not actually pinned.

   The compiled-step oracle skips input index 1—the frozen state—at [test_pos_rollout_trainer_wiring.py:1299](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1299). The scorer oracle then assumes every input is replicated at [test_pos_rollout_trainer_wiring.py:1371](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1371), which is not the load-time contract for sharded backbone leaves.

   With an eight-device fake whose transformer state was sharded before building the programs, the step oracle remained green because it omitted index 1, while both `score_absolute_*` checks became false. A future 10 GB reshard/copy regression would therefore not be reliably diagnosed by the claimed absolute oracle.

## Verified

- Update and scorer math, dtype paths, and graphdef/state pairing are unchanged; each merge uses both halves from the same split.
- Trainer and M1 call the same `build_training_program`. Training jobs build one arm, and M1 reloads the backbone per cell, so I found no new cross-arm frozen-state alias.
- `frozen.state` is passed raw through `_placed`; it bypasses `place_step_inputs`.
- Underlying frozen array buffers retain identity; no donation is configured, and the two-step deletion test is meaningful.
- The actual training update and DEV scorer traces exercise the production builder and no longer contain the marked full backbone.
- The largest remaining production literal is approximately 4 MiB: the bf16 `1×512×4096` null context. The dominant compile object is now the computation graph, especially the 32 Python-unrolled microbatch gradient blocks in the first M1 cell (`256 / 8`), not weight constants. Removing 10.18 GB of literals makes compilation substantially more plausible, but the unrolled graph remains the primary health-window risk.
- No other training-side JIT capture site was found in the trainer, loop, stream, gates, or support modules. The evaluator closure is the remaining unsafe compiled form.
- `[M1] entering …` is printed with flushing before program build/compile.
- The launcher change is comments only; `PYTHONUNBUFFERED=1` remains as at the baseline and no `python -u` change remains.
- Recipe-fingerprint machinery has zero changed lines.
- The new captured-constant file passes all 18 tests, but the adversarial checks above expose the guard gaps.

## Verdict

**REQUEST-REVISION**

The immediate M1 training capture is fixed, but the claimed structural freeze and evaluator regression guard are demonstrably false. Given the cost of another hardware attempt, I would require the adversarial frozen-gradient test, a production-shaped evaluator trace, and explicit frozen-input sharding assertions before ruling this round ready.
