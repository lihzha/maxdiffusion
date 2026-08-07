# rollout_adapter — Codex plan review

## Pass 1 (2026-08-07, vs plan v1): REQUEST-REVISION — 3 BLOCKER + 9 MAJOR, ALL ACCEPTED; plan rewritten as v2 (changelog in the plan header maps findings→changes)

### Reviewer output (verbatim)

1. **BLOCKER — The S7 trainer is not an objective-swappable production trainer.**  
   Its train step is hard-wired to `regression_loss(..., target_context)`, its DEV evaluation requires cached context targets plus a variance table, and its dispatched `start_training()` deliberately raises `NotImplementedError` because model/data wiring never landed ([trainer](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:178), [dispatch boundary](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py:444)). It predicts only the cheap pre-context head output, not a Wan rollout.  
   **Fix:** Base the rollout trainer on `WanTI2VSideAdapterTrainer`’s real pipeline, TFRecord loader, sharded state, optimizer, CFG forward, and checkpoint restore. Import/generalize only S7’s pure checkpoint/selection/stop utilities. Pass frozen transformer state as an explicit non-differentiated, correctly sharded argument; differentiate only adapter parameters.

2. **BLOCKER — T1’s exp_03 merge strategy is not realistic, and R-B-only does not remove the dependency.**  
   The exp_03 losses are private functions inside the 935-line full-FT `wan_ti2v_exp03_trainer.py`, coupled to its context table and train state; there are no standalone “loss modules” to wrap thinly. A read-only merge analysis from the common base shows roughly 44k source insertions and real conflicts, including `train_wan.py`, setup/docs add/add conflicts—not “class-(a) only.” R-B-only still needs this integration.  
   **Fix:** Pin a specific reviewed SHA now—not “whatever SHA exists at T1”—and have exp_03 export a minimal generic dependency commit containing the sampler, support/RNG helpers, and state-agnostic loss kernels. Alternatively import exact blobs with recorded hashes and equivalence tests. If a full branch merge is retained, enumerate its conflict matrix, inherited scope, and dedicated merge review honestly.

3. **BLOCKER — Nothing tests the central ACTION-CONDITIONING claim.**  
   Pure reconstruction SSIM can improve while the adapter ignores actions and exploits the first frame or dataset regularities. Neither C0 nor R-B has a shuffled-, zero-, time-permuted-, or wrong-action comparison.  
   **Fix:** Add paired true-action versus deterministically shuffled-action and zero-action evaluations under identical examples and noise. Predeclare a DEV action-use gate with CI-low above zero and repeat it on TEST. Include an adapter-disabled/no-adapter frozen-backbone row as a diagnostic. A shuffled-action training arm is optional; the evaluation intervention is mandatory.

4. **MAJOR — E1–E3 are directionally motivating but materially overstate what was demonstrated.**  
   Exp_05 supports an eight-token *representation-channel* oracle: own-basin SSIM 0.9227, but fresh-noise re-optimization falls to 0.161 and triggered STOP ([exp_05 reading](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_worklog.md:263)). Exp_04 J1b/J1c covers only eight clips, the null/CFG slot, full-rollout per-clip optimization, and achieved 0.651 own-basin versus about 0.47 foreign-basin SSIM: 72% relative retention, but below the 0.70 absolute floor ([J1c](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_worklog.md:419)). That does not establish that the positive, action-conditioned 128M amortizer has adequate capacity or that short-horizon training will reproduce J1b.  
   The exp_02 correlation motivates rollout compounding, but it does not itself prove rollout loss will outperform further one-step fitting.  
   **Fix:** Recast E3 explicitly as a cross-slot, cross-horizon hypothesis. Say the architecture is held fixed for isolation, not because E1 proves amortizer adequacy; a negative result will not by itself falsify the rollout-loss family.

5. **MAJOR — §11 describes the historical control honestly locally, but contradicts the plan’s “objective-only” claim globally.**  
   A 30k/GBS-512 historical checkpoint versus a 10k/GBS-256 pilot is an achieved-quality screen, not an objective-only ablation. M2 nevertheless demands “C0-at-equal-updates,” which the recommended reuse policy does not supply. Selection cadence and data order are also unmatched.  
   **Fix:** Keep the historical checkpoint as the deployment benchmark, but train C0 alongside M2 and M3 from the same initialization with identical data order, seed stream, GBS, updates, optimizer, evaluation cadence, and checkpoint-selection rule. If that cost is declined, remove causal “objective-only” language. A compute/forward-matched multi-draw one-step control should also be considered before attributing gains specifically to trajectory differentiation.

6. **MAJOR — The +0.05 gate is not yet well-posed as written or implemented.**  
   It should be `mean(paired SSIM delta) ≥ 0.05` and the paired-delta bootstrap CI-low `> 0`, not an arm-only CI compared with a baseline point. Existing code does not provide this exact named gate: `gate_g3_vs_baseline` uses +0.02 and a 60%-improved condition, while the numerically closer +0.05 function is semantically “vs null-only” ([gates](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/null_adapter_gates.py:368)).  
   **Fix:** Add a dedicated exp_06 gate-as-code with pinned manifest, keyed seeds, coverage/imputation, paired bootstrap, and TEST confirmation rule. First reproduce the historical four-sample 0.2946 anchor within tolerance, then evaluate and freeze the old checkpoint’s DEV-64 table under the new paired protocol before scoring new arms.

7. **MAJOR — “Through the sampler” needs an exact deployed CFG gradient contract.**  
   Exp_03’s operator has no CFG branch, whereas deployment uses conditional action-adapter output plus frozen unconditional output at guide scale 5 ([current evaluator](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/generate_wan_side_adapter.py:111)). Copying the one-step trainer’s `stop_gradient(z_t)` on the unconditional branch would silently truncate inter-step gradients; it is harmless only when the noisy input is parameter-independent. Conversely, the architecture’s existing stop-gradient on first-block features must remain because architecture is held fixed.  
   **Fix:** Specify the conditional/unconditional derivative boundaries. Require fixed-input full-rollout output parity with the deployed evaluator and a two-step finite-difference oracle proving inter-step adapter gradients—including the state dependence of the unconditional CFG branch—while confirming the gradient tree contains adapter parameters only.

8. **MAJOR — Noise/support and horizon selection are underspecified.**  
   “Fresh grid position” does not state whether support is per batch or per example, the allowed start range, exclusion of terminal σ=0, or resume/accumulation-stable PRNG derivation. Exp_03 uses one support per batch. R-A also omits its scheduled-sampling probability and ramp. Fit alone is not a scientific rule for choosing k=2 versus k=4.  
   **Fix:** Make k=2 the predeclared primary because it is the validated construction. Treat k=4 as a separate exploratory arm or require a predeclared matched learnability comparison. Pin support ranges, support granularity, epsilon/support key derivation, resume behavior, and C0/R-B paired randomness; specify R-A’s `p_ss` schedule if retained.

9. **MAJOR — Checkpoint selection and the S7 stop rule cannot be reused “verbatim.”**  
   S7 selects on cached-target `dev_normalized_mse`; those targets and their variance table do not exist here. The inherited YAML points `eval_data_dir` at the whole validation TFRecord collection ([YAML](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/configs/base_wan_5b_pos_context_train.yml:111)), which risks touching TEST-64 during training-time selection if reused naively.  
   **Fix:** Define a deterministic fixed-draw DEV-64 rollout-loss estimand for every-1k selection, manifest-bound and explicitly excluding TEST-64. Generalize checkpoint metadata and the stop rule to that metric, and apply the identical rule to matched C0. Give M2’s four-example spot-check an executable numerical continuation rule rather than “improves.”

10. **MAJOR — The evaluator overlap claim omits required settled-file edits.**  
    Exp_04 never shipped R14/R15 after its STOP. Exp_05 S9 remains deliberately partial: `regressed_pre_context_eval` raises, and its tripwire asserts that `generate_wan_null_adapter.py` does not exist ([boundary](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_context_eval.py:195), [tripwire](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_eval.py:462)). Landing that file will break the inherited suite.  
    **Fix:** Either withdraw the claim that exp_06 un-stalls exp_05, or explicitly enumerate the dual-touch edits to `pos_context_eval.py` and its tripwire tests, with a focused settled-module review and support for both historical and selected-checkpoint metadata formats.

11. **MAJOR — Cost bounds are too confident for k=4 and the adapter backward.**  
    The latest closed exp_03 v6e-64 measurement is B at 2.713×, while C missed fit at 31.28G/31.25G and the first accumulation attempt worsened this to 31.98G ([exp_03 measurements](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/docs/worklogs_yixun/exp_03_rollout_objective_claude/rollout_objective_command.md:382)). k=4 is unmeasured and can exceed 4×. Freezing weights avoids weight gradients but still requires VJPs through the 5B network to the context and preceding rollout state.  
    **Fix:** State cost/HBM as unknown beyond the k=2 reference. Define the 10% headroom statistic precisely, measure compile and steady-step peaks plus reservation failures, and include evaluation/checkpoint overhead in wall projections. M1 should authorize only the exact horizon/microbatch cell it measured.

12. **MAJOR — SOP and runbook compliance needs revision.**  
    Plan approval cannot approve M1+M2: no implementation SHA, acceptance criteria, or pushed artifact exists yet, and M2 depends on M1’s verdict. T3 and T5 also bundle several contracts and are unlikely to satisfy the <200-LOC round rule. T1 should use characterization tests rather than an artificial red. Finally, the runbook paragraph covers issues #10 and #13 but omits #11’s `HyperParameters` access/fix-propagation rule and #12’s xtrace-secret rule.  
    **Fix:** Request each remote launch only after the exact pushed SHA, command-ledger entry, worklog acceptance criteria, topology/batch declaration, and log path exist; request M2 only after M1 passes. Split trainer and evaluator rounds into smaller TDD units. Require YAML-declared direct config reads or the reviewed optional helper, preserve the caller’s xtrace state around secrets, and define SHA-bound adoption of the latest complete checkpoint into a fresh attempt-scoped root.

**Status:** No subprocesses or jobs are running; this pass is complete, and the earliest safe steering point is now—the planner can resolve these findings before pass 2.

**REQUEST-REVISION** — The experiment is promising, but the current plan cannot yet isolate the rollout objective, execute through the claimed trainer/dependency seams, or establish that any SSIM gain is action-conditioned.

## Pass 2 (2026-08-07, vs plan v2): REQUEST-REVISION — 10 RESOLVED, 2 PARTIAL (F3 shuffle executability, F9 cadence) + 1 new MAJOR (batch-1 derangement contract); both pins applied as v2.1

### Reviewer output (verbatim)

Pass 2 result: 10 RESOLVED, 2 PARTIALLY-RESOLVED.

| Finding | Status | Evidence |
|---|---|---|
| F1 | RESOLVED | [v2 §3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:19) bases the sibling on `WanTI2VSideAdapterTrainer`, excludes S7’s regression path, and generalizes only utilities; the tree confirms real pipeline, TFRecord, sharding, optimizer, restore, and adapter-only-gradient seams in [the production trainer](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py:451). |
| F2 | RESOLVED | [T1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:45) pins `2ef9b8a`, imports blobs with hashes, extracts kernels, and requires equivalence tests; the reachable pin contains `overfit100_sampling.py`, `exp03_aux_key`, `rollout_support`, and `_rollout_loss`, and the sampler blob matches current exp_03 head. |
| F3 | PARTIALLY-RESOLVED | [§3e](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:31) adds the required true/shuffled/zero paired intervention, DEV CI gate, TEST repetition, and no-adapter diagnostic, but its “derangement within batch” is undefined for the deployed batch-one evaluation path. |
| F4 | RESOLVED | [§2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:13) explicitly labels E3 a cross-slot/cross-horizon/cross-regime hypothesis, holds architecture fixed for isolation, and scopes a negative result to this cell. |
| F5 | RESOLVED | [P3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:38) requires matched C0 with identical initialization, data/seed stream, GBS, updates, optimizer, cadence, and selection; the historical checkpoint is only the benchmark row. |
| F6 | RESOLVED | [§3c](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:27) defines the exact paired mean-ΔSSIM `≥+0.05` and paired-CI-low `>0` gate, manifest binding, imputation, TEST confirmation, and anchor-first protocol. |
| F7 | RESOLVED | [§3a](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:23) explicitly differentiates both CFG branches through rollout state, preserves the architectural block-0 stop-gradient, restricts gradients to adapter parameters, and requires parity plus a two-step finite-difference oracle. |
| F8 | RESOLVED | [§3b](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:25) pins per-batch support, legal grid range, fresh ε, purpose-folded resume/accumulation-stable PRNGs, and paired streams; [the arm policy](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:21) makes k=2 primary and k=4 exploratory. |
| F9 | PARTIALLY-RESOLVED | [§3d](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:29) supplies the fixed-draw DEV-64 estimand, manifest isolation, TEST exclusion, matched-C0 stop rule, and metadata, but v2 no longer states the required every-1k selection cadence—only that the arms share an unspecified cadence. |
| F10 | RESOLVED | [planned file 5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:49) withdraws the exp_05-unstall claim and uses `eval_wan_pos_rollout.py`; the inherited tripwire checks only `generate_wan_null_adapter.py` in [the settled test](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_eval.py:462), so the proposed filename is safe. |
| F11 | RESOLVED | [§10](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:72) declares costs beyond k=2 unknown, retains the measured 2.713×/HBM facts, acknowledges frozen-backbone VJPs, and makes M1 cell-specific with defined peak/headroom accounting. |
| F12 | RESOLVED | [§4](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:33), [runbook rules](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:41), and [§6](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:55) enforce per-SHA sequential approval, split rounds, characterization imports, HyperParameters-safe reads, xtrace preservation, and SHA-bound resume adoption. |

New finding:

- MAJOR — The action derangement is not executable under the inherited evaluator shape. The deployed evaluator constructs `[1,…]` batches in [`_as_batch`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/generate_wan_side_adapter.py:282), and the inherited config defaults to one evaluation video. A within-batch derangement therefore has no valid permutation. Fix: define a manifest/cohort-level seeded derangement independently for DEV-64 and TEST-64, require no fixed points or byte-identical replacement actions, persist its permutation/hash, then feed each mapped wrong-action pair through the batch-one rollout.

For F9, restore an explicit `eval/selection every 1,000 updates` requirement—or another numerically pinned cadence—identically for C0 and R-B.

REQUEST-REVISION — v2 resolves the substantive design direction, but the action-shuffle execution contract and selection cadence must be pinned before implementation.

## Pass 3 (2026-08-07, vs plan v2.1): **APPROVE-PLAN** — both pass-2 pins RESOLVED, zero regressions, zero new findings

### Reviewer output (verbatim)

- §3e — **RESOLVED**: cohort-independent seeded derangements, collision handling, persisted permutation/hash, and batch-1 execution are explicitly pinned ([plan:31](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:31)).
- §3d — **RESOLVED**: evaluation and checkpoint selection occur every 1,000 updates for both arms through a YAML-declared, drift-pinned key ([plan:29](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/plan_rollout_adapter.md:29)).
- Regression check — **NONE**: `c3ba9c7..3360f89` changes only the header, §3d, and §3e.
- New plan-blocking findings — **NONE**.

**APPROVE-PLAN** — v2.1 closes both remaining pass-2 issues without regressing the previously resolved design.
