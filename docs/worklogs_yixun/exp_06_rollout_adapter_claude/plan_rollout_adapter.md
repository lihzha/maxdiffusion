# plan_rollout_adapter — exp_06: Rollout-objective training of the frozen-backbone pre_context adapter

v1 (2026-08-07). Planner: Claude Fable 5. Status: DRAFT — pending Codex plan review, then Yixun approval (§11).

## 1. What Yixun asked for

The project goal, restated 2026-08-07: **a good action-conditioned world model for robotics WITHOUT fine-tuning the Wan2.2 backbone.** The Planner's synthesis of exp_01–exp_05 recommended training the existing pre_context adapter with rollout-based objectives; Yixun: "go ahead for exp_06".

## 2. The evidence this plan stands on (all recorded in the corpus; not re-derived here)

- **E1 (capacity, exp_05 H1 / exp_04 A1):** 8 optimized conditioning tokens per step drive the FROZEN 5B to 0.92 SSIM DROID-future reconstruction (null slot: 0.85). The backbone and the channel both suffice; the deployed adapter's 0.2946 sits at ~32% of the demonstrated ceiling.
- **E2 (objective mismatch, exp_02 law / deployed baseline / exp_04-05 greedy-vs-joint):** one-step-trained systems are bottlenecked by rollout compounding, not one-step fit (r = −0.9994); per-clip greedy optimization is basin-locked (probes 0.17), while THROUGH-THE-SAMPLER endpoint optimization lifts capacity and transfers at 72% relative retention (J1b/J1c).
- **E3 (the amortization bet, exp_04/05 STOP nuance):** static per-clip targets are dead (predeclared STOPs), but a state-conditioned adapter trained through the rollout across thousands of clips × fresh basins is the amortized form of the thing J1b showed works. exp_06 tests E3 directly.

## 3. Method

**Model:** the existing `NNXWanSideAdapterStack` pre_context configuration, UNCHANGED (~128M trainable; frozen 5B; head emits the 8-token context consumed as the whole `encoder_hidden_states`; bf16 activation-cast contract as audited in exp_05 §8). No architecture changes in exp_06 — the objective is the only manipulated variable (E1 justifies this).

**Objectives (the arms):**
- **R-B (primary): short-horizon differentiable rollout.** From a GT latent state noised to grid position σ_hi (fresh ε per draw), unroll k sampler steps THROUGH the extracted sampler with the adapter in the loop (`lax.scan` + remat, exp_03's construction; exp_04's A3 proved the frozen-5B remat'd unroll on TPU), endpoint MSE against the GT-consistent target ÷ (σ_hi−σ_lo)². k ∈ {2, 4} (k=2 is exp_03's validated cell; k=4 is the extension the adapter's cheaper backward may afford — fit probe decides).
- **R-A (secondary): corrective scheduled sampling.** k_A∼U{1,2} stop-grad advances along the descending grid, corrective target v* = (z_lo − z_gt)/σ_lo (exact under the Euler rule) — exp_03's A.
- **R-C (conditional): the λ-mix** (0.5·L_A + 0.5·L_B, single update) — included ONLY if exp_03's S1.5/S1.6 verdicts favor it (its cost 3.96× and HBM flags are exp_03-context; re-measured here by the fit probe).
- **C0 (control): one-step denoising** — the deployed objective, policy per §11 decision 1 (reuse the existing 30k/512 checkpoint vs retrain update-matched).

**Noise/exposure policy:** `side_adapter_noise_mode=fresh` everywhere (house rule); every rollout draw samples fresh ε and a fresh grid position — multi-basin exposure by construction (E3).

**Data:** the cached DROID side-adapter TFRecords (`gs://v6_east1d/datasets/droid_wan_side_adapter/{train,val}`), identical to the baseline run. Eval cohorts: **exp_04's published J0 manifests reused verbatim** (DEV-64 for selection/gates; TEST-64 untouched until the final claim; the leakage-proof episode-stratified construction carries over).

**Primary metric and gate:** canonical 25-step rollout mean future-SSIM on DEV-64 (the exp_04/05 decode + SSIM machinery, noise-matched, paired bootstrap 10k/seed 20260804, coverage/imputation rules inherited). **Success gate: selected-arm SSIM ≥ baseline + 0.05 with CI-low > baseline** (the baseline column measured under the identical protocol — the 0.2946 was a 4-sample validation, not comparable raw). TEST-64 confirmation only after the DEV gate passes. The exp_02 law provides the expectation frame: loss must fall ~0.04 per +0.05 SSIM if the law's slope transfers to the adapter setting — a predeclared diagnostic, not a gate.

## 4. Phases and jobs (every job separately approved per announcement 02)

- **P0 — TDD (no TPU):** rounds §6; suite green + mutation batteries per SOP.
- **P1 — fit probe (job M1, v6e-8):** compile/step-time/peak-HBM for R-B k∈{2,4} (and R-A; R-C if in) at the adapter setting, microbatch ladder; hard budgets (compile ≤30 min, projection rule ≤ declared wall); the A3-measurement machinery pattern (measure-then-project, structured verdicts). Continuation requires >10% HBM headroom (exp_03's S1.6-class lesson).
- **P2 — learnability probe (job M2, v6e-8):** 32 examples, ≤2k steps, R-B only: loss falls + a 4-example rollout SSIM spot-check improves over C0-at-equal-updates. Cheap kill switch before any big spend.
- **P3 — training arms (job M3+, v6e-64):** per §11 decisions 2–3; GBS 256 with the S7 accumulation fallback (logical batch preserved); eval-every-1k + the S7 stop rule verbatim; checkpoint/selection discipline = the S7 redesign (recency resume + immutable earliest-best selection + F2 state in-checkpoint).
- **P4 — eval + report:** DEV gates → (conditional) TEST confirmation → analysis + HTML report per SOP; comparison videos from the eval machinery.

**Runbook discipline (issues #10–#13, standing):** attempt-scoped artifact roots for every multi-phase job; adopt-published-artifacts on retry; no fixed immutable destinations inside a queue-retried command; submissions via Yixun where the classifier blocks.

## 5. Planned code, per file (reuse-first; F6-style: settled modules from other experiments are imported or merged at pinned SHAs, never edited)

1. **(M) merge-T1:** one-way exp_03 → this branch at a PINNED SHA (post-S1.6 preferred): `models/wan/overfit100_sampling.py` (the bitwise-inert extracted sampler) + exp_03's loss modules. Class rules as exp_04→exp_05 merges; conflicts expected class-(a) only (this branch already carries exp_04+exp_05).
2. **(N) `src/maxdiffusion/pos_context_rollout_losses.py`:** the R-A/R-B(/R-C) losses adapted to the adapter's frozen-split forward (the S7 closure seam: frozen transformer captured, adapter params the only grad path) — thin adapters over exp_03's loss cores, NOT reimplementations.
3. **(E) `trainers/wan_pos_context_regression_trainer.py` → (N) subclass or sibling `wan_pos_context_rollout_trainer.py`:** swap the objective layer (S6's gather is cache-specific and NOT used here — this trainer consumes the TFRecord latents directly like the side-adapter trainer); keep S7's schedule/accumulation/stop-rule/checkpoint machinery by import.
4. **(N) config `base_wan_5b_pos_context_rollout.yml`** (from the S8 YAML; + k, arm, rollout-noise keys) + **(E) `train_wan.py`** additive arm `POS_CONTEXT_ROLLOUT_TI2V`.
5. **(N) eval path:** the exp_04/05 decode+SSIM+gates machinery over J0 cohorts driven by adapter checkpoints (the S9 restore/certificate discipline reused; the R14/R15-shaped evaluator work lands HERE, which also un-stalls exp_05's S9 tripwire obligations if exp_05 ever resumes).
6. **(N) launchers** `train_wan_pos_rollout.sh`, `eval_wan_pos_rollout.sh` — executed-under-bash tests per the S10a technique; attempt-scoped-root support built in.

## 6. Coder rounds (single-contract, <200 exec LOC each unless pre-authorized)

**T1** `merge-exp03-modules` (the pinned-SHA merge + combined suite). **T2** `rollout-losses` (analytic oracles per exp_03's forms; the frozen-split grad-path pins). **T3** `rollout-trainer` (objective swap; S7 machinery byte-reused; jit oracle; accumulation equivalence re-proven at the new loss). **T4** `dispatch-config`. **T5** `evaluator` (restore→rollout→gates over J0 cohorts). **T6** `launchers`. **T7** `fit-probe mode` (M1). Each: red → green → Codex review → strengthen → commit, one review file per round.

## 7/8. Validation ladder and parity

Inherited wholesale: the exp_04/05 noise conventions (golden-pinned), sigma schedules, decode/SSIM parity, gates-as-code, provenance-bound artifacts, fail-closed manifests. New parity obligation: **T2's losses vs exp_03's originals** — fixed-input equivalence tests against the merged exp_03 modules at the pinned SHA (bitwise where the reduction order permits, else principled tolerances per the S6 half-ulp precedent). The exp_03 dependency is COORDINATED, not assumed: the merge pins whatever SHA exp_03's session has committed at T1 time, and the plan does not block on exp_03's S2 arms.

## 9. Launch plan (indicative; v6e-8 probes, v6e-64 training; all pending per-job approval)

| Job | What | Est. |
|---|---|---|
| M1 | fit probe (k ladder, arm set) | ~1 h |
| M2 | learnability probe (32 ex, ≤2k steps) | ~2–3 h |
| M3 | first full arm (R-B at the §11-3 budget) + C0 policy per §11-1 | ~1–3 d wall depending on budget class |
| M4 | eval sweep (selected checkpoints, DEV-64) | ~2–4 h |

## 10. Risks

- **Cost:** 2.5–4× per-step over one-step training (exp_03 smokes); mitigated by the pilot budget class and the M1/M2 kill switches.
- **HBM:** k-unroll backward at the adapter setting is NEW territory (A3's backward was over the null tensor, not adapter params; exp_03's was full-FT) — M1 exists precisely for this; no training job before its verdict.
- **exp_03 coupling:** its S1.5/S1.6 verdicts inform the arm set but its branch is owned by the parallel session — T1 pins a committed SHA and never edits exp_03 files; if its rounds slip, exp_06 launches R-B-only (the arm exp_03 already validated through S1).
- **The law may not transfer** to the adapter setting (slope was measured under full-FT); it is a diagnostic frame, not a gate.
- **NaN history:** exp_03's C-arm step-8 NaN was resolved by the fixed-k unroll; the same construction is used here, and the emit-before-raise instrumentation comes with the merge.

## 11. Decision points for Yixun (at plan approval)

1. **Control policy:** reuse the existing 30k/GBS-512 checkpoint as baseline (cheap; different batch/updates than the arms) vs retrain an update-matched C0 (rigorous; ~1 extra arm of cost). Planner recommends: **reuse for the pilot, retrain only if the pilot gate passes.**
2. **Arm set at M3:** R-B only vs R-B + R-A (+R-C per exp_03's verdicts). Planner recommends: **R-B only for the pilot.**
3. **Budget class for M3:** pilot 10k steps @ GBS 256 vs baseline-matched 30k. Planner recommends: **pilot first**; the stop rule + eval cadence make an early success/failure visible by ~5k.
4. Approve M1+M2 together at plan approval (both v6e-8, ~4 h combined) with M3+ gated separately?
