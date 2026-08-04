# pos_context — Codex plan review (plan v1)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (codex-cli 0.146.0, read-only, exp_05 worktree). Artifact under review: `plan_pos_context.md` v1 @ `cd2e59b`. Briefed per SOP incl. the exp_04 v5 inherited-contract base and its five-pass review history; scope calibrated to exp_05 deltas only (prompt archived in session scratchpad).

## Reviewer output (verbatim)

Context loaded:

- Full experiment SOP and announcements `01_status_block_every_reply.md` / `02_tpu_run_requires_approval.md`.
- Full exp_05 query, plan v1 at `cd2e59b`, and worklog.
- Full exp_04 plan v5 at `e8205e1` and complete five-pass Codex review/resolution history.
- Pinned Wan2.2 commit `f370228`: requested positive optimization/replay/driver ranges, `L_pos` handling, DROID `L_pos=1` launch path, and `adaptor_design.md` §§1.2, 3.6, 3.11–3.14.
- Requested maxdiffusion pre_context implementation, trainer freeze/optimizer split, base YAML keys, plus the existing rollout’s conditional/unconditional context handling.

The branch-swap recurrence itself is faithful: `v_uncond` is cached and C-independent, gradients pass through `v_cond`, Adam is reset per outer step, and the locked post-update C advances the trajectory. I found no issue in the inherited exp_04 contracts themselves.

1. **MAJOR — §3 Method; §4 P3/P3a/K4: inversion targets and deployed pre_context use different context representations.**

   The plan defines `embed_pos(C)` as a 512-row `T5("")` context with eight rows replaced. The PyTorch reference instead forces the positive context itself to `L_pos` rows and passes that tensor directly to `v_cond`. More importantly, `wan_pre_context_adapter_forward` passes the head’s `[B,8,4096]` output directly to Wan; it does not restore the other 504 rows. Wan has no text padding mask on this path, so the extra rows and sequence length affect cross-attention.

   Consequently, serialized-target replay and the deployed adapter are not realizing the same C*, and P3a’s rollout gap would conflate regression error with a 512-token→8-token representation shift.

   **Recommended change:** Match the existing pre_context path: optimize and replay an exact `[B,8,4096]` conditional context, constructed by truncating/padding `T5("")` to eight rows, while retaining the baseline’s normal frozen unconditional context. Remove the branch-equality premise and replace it with reference context-construction and conditional-velocity parity tests. This also invalidates automatic B0/A0 and B2-0/A2-0 reuse—and identical-pivot claims—unless numerical characterization proves equivalence. Pin the `{1,8}` ablation to the same direct-length convention, arm, and adopted optimization recipe.

2. **MAJOR — §4 P3/Risks: overfit and closed-loop-shift handling is not decidable.**

   At batch 256, 30k steps expose 7.68M sampled pairs—about 154 passes over the nominal 50k correlated state/target pairs. “If the DEV/train gap explodes” supplies no threshold, while best DEV embedding MSE does not certify closed-loop behavior. P3a evaluates rollout only on its 32 fitted examples; K4 is currently the first unseen closed-loop check.

   **Recommended change:** Define normalized train/DEV target MSE, evaluation cadence, patience, and a quantitative stop rule—for example, stop and retain the prior best checkpoint when DEV normalized MSE exceeds 2× its best for three consecutive 1k-step evaluations while train MSE continues falling. Before unlocking TEST/K4, run one noise-matched closed-loop DEV evaluation of the selected checkpoint against null-only and serialized-target replay with inherited coverage/imputation rules. Predeclare that any later shift mitigation is selected on DEV and cannot reuse TEST as a tuning set.

3. **MAJOR — §4 P3; §9 K3: the v6e-8 batching, sharding, and cost contract is missing.**

   Exp_04’s 9M adapter trainer does not establish feasibility for this trainer, which executes Wan patch/time embedding plus block-0 self-attention and backpropagates through the roughly 128M pre_context stack. The reused trainer pattern replicates adapter parameters and optimizer state. Logical batch 256 on eight devices implies 32 examples/device, but no microbatching or accumulation rule is given. The “≈1/40” statement does not justify the 1–3-hour estimate.

   **Recommended change:** Pin adapter/optimizer/data shardings and a logical-GBS-preserving gradient-accumulation fallback. During P3a/fit probe, record compile time, step time, peak HBM, actual parameter count, and projected 30k wall time; require HBM headroom and a predeclared wall-time ceiling before continuing. If batch 256 cannot fit, do not silently change the logical batch—accumulate, request different hardware, or reopen the recipe.

4. **MAJOR — §§6, 9–10: the exp_04 merge dependency is incomplete.**

   Merging exp_04 only through R8 supplies the inversion runner, but K1 also requires R9/J0’s immutable manifests. S4 edits `generate_wan_null_adapter.py` and relies on legacy restore/parity machinery implemented only in exp_04 R14–R15. No later merge is specified.

   **Recommended change:** Add a commit-pinned dependency matrix: S1/S2 after R8; K1 only after R9 and successful J0 manifest publication; S4/K4 only after R15 and J5 anchor/parity availability. Specify a second one-way exp_04→exp_05 merge, its exact boundary SHA, conflict policy, and full shared-suite validation—or wait for R15 and perform one merge.

5. **MAJOR — §§9/11: folding K1 into J1 is not currently executable under branch isolation.**

   Positive-mode code exists only on exp_05 after exp_04→exp_05 merging, whereas J1 belongs to the exp_04 branch. The plan does not identify the authoritative pushed SHA, command/worklog ownership, artifact destinations, or how both experiments bind provenance. After fixing Finding 1, the pivots are also no longer automatically shareable because the positive inversion uses an eight-token conditional context.

   **Recommended change:** Make separate J1/K1 jobs the default and remove the fold option. Retain it only if the plan defines an SOP-compliant authoritative commit and dual-artifact provenance, and first proves the pivot computations numerically identical.

6. **MAJOR — §§5–6: the planned code mutates a settled shared path and the four Coder rounds cannot plausibly remain under 200 LOC.**

   Retrofitting null-state recording “if ever needed” silently changes exp_04’s settled core without an exp_04 review. Likewise, leaving separate replay versus `branch=` to the Coder is an unresolved shared-interface decision. S2 combines arm execution, config, schema, artifact reuse, and provenance; S3 combines a full trainer, dataset path, checkpointing, optimizer and dispatch; S4 combines restore, evaluation, tables, tests, and launchers.

   **Recommended change:** Keep null behavior untouched and pin a separate `replay_with_positive` API. Split rounds into focused units: positive optimization; positive replay/state capture; runner slot/arms; schema/reuse; regression gather/loss; trainer state/optimizer/checkpoint; dispatch/config; typed restore; rollout/gate integration; launchers. Give each a dedicated red test and review marker.

7. **MAJOR — §4 P2: the new `z_bar_states` field lacks its own fidelity gate, and storage is understated.**

   The inherited fidelity replay validates serialized embeddings/start/final latent, not the newly introduced teacher inputs. A cache can therefore pass replay fidelity while fp16 `z_bar_states` alter the block-0 features used throughout K3. The stated size is also incorrect: the states alone are about 5.93 MiB/record; the full record is about 8.23 MiB and 2,128 records are roughly 17.1 GiB before overhead. The declared fp32 fallback approaches 34 GiB.

   **Recommended change:** Extend byte-length validation and completion fingerprints explicitly to `z_bar_states`. On the predeclared DEV fidelity subset, require the serialized states to produce the same bf16 model inputs—or a pinned block-0 feature tolerance—as the in-memory fp32 states; otherwise select fp32 before TRAIN-2000 is built. Correct the storage/free-space estimates and restrict PCA/cosine inspection to TRAIN/DEV, leaving TEST uninspected until K4.

8. **MINOR — §2 Q3': the comparison does not isolate training signal causally.**

   Architecture is held fixed, but the historical denoising baseline used different data exposure and optimizer/batch settings. This is still the correct same-architecture comparator requested, but it is an achieved-quality comparison rather than a controlled loss-only ablation.

   **Recommended change:** Adjust the interpretation language accordingly; do not add an expensive matched denoising retrain unless separately approved.

9. **MINOR — §2 Q1': “which slot steers more per token” has no defined statistic.**

   Positive and null experiments use different token counts (`8` versus `16`) and different CFG coefficients, so raw reconstruction gains do not establish per-token steering.

   **Recommended change:** Either remove the science-bonus claim or predefine a diagnostic normalization, clearly labeled non-causal.

**Status**

- Running subprocesses/jobs: none; this read-only review is complete.
- Earliest steer: now; revision is safe before implementation or any TPU approval.

**REQUEST-REVISION — The branch-swap math is sound, but the 512-token inversion target does not match the eight-token deployed pre_context path, and the training, dependency, artifact, and round contracts remain materially underdetermined.**

---

## Planner resolutions (plan v1 → v2)

All 9 findings **accepted**; plan v2 implements them:

1. **F1 (the load-bearing catch):** context convention rebuilt around the deployed representation — C ∈ R^{8×4096} passed directly as the entire encoder_hidden_states (seq len 8), warm start via the reference's truncate/pad L_pos forcing; uncond stays the baseline 512-row T5(""); CFG active at init (branch-equality premise removed, replaced by context-construction parity + conditional-velocity parity tests); pivots no longer shared with exp_04 (8-token inversion context, reference-faithful); B0/B2-0 recomputed as frozen-C controls; L_pos ablation pinned to the same convention/arm/recipe.
2. **F2:** normalized-MSE metric, 1k eval cadence, 3-consecutive-evals ×2 divergence stop rule retaining prior best, pre-K4 noise-matched closed-loop DEV gate vs null-only + serialized replay, and a predeclared TEST-is-never-a-tuning-set rule.
3. **F3:** sharding pinned (adapter/opt replicated, data batch-sharded, frozen block-0 FSDP), logical-GBS-preserving gradient accumulation, P3a records compile/step/HBM/param-count/projected wall with HBM-headroom >10% and wall ≤6 h continuation gates; no silent logical-batch change.
4. **F4:** commit-pinned dependency matrix (S1–S5 after exp_04 R9 via merge-1; K1 after J0 publication; S9/K4 after R15 + J5 via merge-2), one-way merges only, conflict policy (shared files take exp_04's side), full-suite-green post-merge, both merges SHA-logged.
5. **F5:** fold-K1-into-J1 removed; separate jobs are the default and only option; the pivot-identity premise it rested on is gone with F1 anyway.
6. **F6:** exp_04's settled module is never modified (no null-state retrofit; shared primitives imported); `replay_with_positive` is its own API in a new `pos_context_inversion_wan.py`; rounds split S1–S10, each single-contract with a named red test; runner edit additive with a null-slot characterization test.
7. **F7:** `z_bar_states` covered by byte-length validation + completion fingerprints; fidelity gate extended (bit-identical bf16 model inputs, else pinned block-0 feature tolerance, else fp32); storage corrected (≈8.2 MiB/record, ≈17.1 GiB fp16, ≈34 GiB fp32 fallback) with a pre-build free-space check; structure diagnostic restricted to TRAIN+DEV, TEST targets uninspected until K4.
8. **F8:** Q3' reframed as achieved-quality (data exposure and optimizer settings differ from the historical baseline); no matched retrain added.
9. **F9:** per-token-steering claim removed; side-by-side tables descriptive only.

Material revision ⇒ re-review pass 2 dispatched.

---

## Re-review pass 2 (plan v2 @ `bac718b`)

Reviewer: Codex `gpt-5.6-sol` xhigh, read-only, exp_05 worktree.

### Reviewer output (verbatim)

Context loaded:

- Full [experiment SOP](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/experiment_SOP.md) and both standing announcements.
- Full exp_05 [query](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_yixun_query.md), [plan v2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md) at `bac718b`, [pass-1 review/resolutions](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_codex_plan_review.md), and [worklog](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_worklog.md).
- Full inherited [exp_04 plan v5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_04_null_adapter_claude/plan_null_adapter.md).
- Exact `git diff cd2e59b..bac718b -- docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md`.
- Pinned Wan2.2 `f370228` positive optimization/replay and `run_positive_inversion`, including `L_pos` forcing.
- MaxDiffusion pre-context forward, 2-D timestep/text handling, prompt padding, and side-adapter sharding paths.

F1–F9 verification:

- F1 — RESOLVED: Direct 8-token conditional context, 512-token unconditional context, active-CFG controls, 8-token inversion, parity tests, and non-shared pivots are consistent with both implementations.
- F2 — RESOLVED: Normalized metric, cadence, quantitative stop/patience rule, best-checkpoint retention, DEV closed-loop gate, and TEST isolation are specified.
- F3 — RESOLVED: Sharding, logical-GBS-preserving accumulation, and measured HBM/wall-time continuation gates make feasibility fail-closed.
- F4 — PARTIALLY-RESOLVED: The dependency matrix and two merge boundaries are present, but the shared-file conflict rule is internally inconsistent; see G1.
- F5 — RESOLVED: The unsupported J1/K1 folding option and pivot-sharing premise are removed.
- F6 — RESOLVED: Positive logic is isolated, null behavior is characterized, APIs are pinned, and implementation is split into reviewed TDD rounds.
- F7 — RESOLVED: State integrity, fidelity fallback, and TEST non-inspection are explicit; the 17.1-GiB estimate is conservative—the declared raw payload is nearer 14.8 GiB before overhead—so storage is not under-budgeted.
- F8 — RESOLVED: Q3′ is correctly framed as an achieved-quality comparison.
- F9 — RESOLVED: The unsupported per-token-steering claim is removed.

New findings:

1. **G1 — MAJOR — §6 dependency matrix / merge policy.** S8 edits shared `train_wan.py` to add `POS_CONTEXT_TI2V`, while exp_04 R13 independently edits the same dispatch chain for `NULL_EMBED_TI2V`. Merge-2 occurs afterward, yet conflicts in shared files must take exp_04 verbatim; that can erase S8 and makes the required combined-green suite unattainable. Make merge-2 a prerequisite to S8, or explicitly require a reviewed additive resolution preserving both dispatches and test both routes post-merge.

Status:

- Running subprocesses/jobs: none; this read-only review is complete.
- Earliest steer: now; revision is safe before S8/merge-2.

**REQUEST-REVISION — v2 closes eight findings and most of F4, but its second-merge policy can discard the exp_05 training dispatch and is not executable as written.**


### Planner resolutions (plan v2 → v3)

- **G1 accepted (closes F4):** merge policy rewritten with three file classes; the enumerated dual-touch files (`train_wan.py`, `run_wan_null_inversion.py`, `generate_wan_null_adapter.py`) resolve to the additive union preserving both experiments' behavior; any merge commit containing such a resolution gets its own focused Codex review (`pos_context_codex_code_merge-<n>_review.md`) before the next round; post-merge acceptance requires the combined suite green with BOTH dispatch-route tests exercised. S8's ordering unchanged; merge-2 can no longer erase it.

Delta-only re-review pass 3 dispatched.
