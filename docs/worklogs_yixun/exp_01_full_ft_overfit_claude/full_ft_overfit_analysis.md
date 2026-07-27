# exp_01 `full_ft_overfit` — Analysis (Planner: Claude Fable 5; v2 after Codex analysis review)

**Status: FINAL** — the official 16/16 step-0 validation landed 2026-07-22 (job `s0b`): SSIM **0.1966**, latent MSE 3.479, pixel MSE 0.199 — confirming the preliminary 14/16 aggregate (0.20/3.51/0.199). The predeclared comparator protocol is complete. This v2 incorporates all six findings of `full_ft_overfit_codex_analysis_review.md` (REQUEST-REVISION → resolutions appended there).

**Question (Lihan):** can the Wan TI2V training pipeline be overfit at all when nothing is frozen — i.e. is the adapters' difficulty caused by (A) a broken data/loss/pipeline, or (B) the difficulty of optimizing through a frozen 5B backbone?

**Verdict:** the strong "pipeline cannot fit" form of **(A) is disfavored**; the results are consistent with — and, under the predeclared decision rule, **favor (B)** — but this experiment does **not causally isolate** frozen-backbone optimization as the adapters' binding constraint. The adapter comparisons differ in split (val vs train clips), objective (guide-5 vs guide-1), and conditioning (actions vs none); shared helpers establish core loss-code parity, not complete objective/conditioning parity.

## 1. Evidence

**Training dynamics (official, 20k steps @ GBS 256, 3.55 passes):** loss 0.60 → 0.19 within ~500 steps, hard plateau 0.176–0.183 thereafter (−6% over the remaining 19k steps); gradients steady (norm 0.09–0.10), zero instability, 1.90 steps/s, no preemptions. Every §6 acceptance criterion passed on real-hardware logs.

**Cohort reconstruction (official for trained checkpoints; 16 fixed train clips, identical seeds):** SSIM ≈ **0.787**, latent MSE ≈ **0.25** at ALL of steps 5000/10000/15000/20000 — flat to three decimals. The pretrained comparator is **preliminary**: 14/16 samples salvaged from the original s0 job, which was **queue-wedged after 8 preempted attempts** (not merely "a preempted attempt"); official job re-queued. Preliminary aggregate: SSIM **0.20** (range 0.13–0.31), latent MSE **3.51**. The official 16/16 comparator (SSIM 0.1966) confirms it; the predeclared protocol is complete. **Qualitative video review has not yet been performed** (pull commands in the HTML report); no claim is made here that clips were visually verified as reconstructed.

**What the run demonstrates, stated narrowly:** the no-action, guide-1, full-FT path achieves a substantial training-cohort reconstruction improvement, with no further measured improvement after step 5000. **This experiment cannot separate domain learning from per-clip memorization** — all rollout evaluation is on training clips; a matched held-out cohort (same protocol, unseen clips) would be required for that distinction, and the unknown irreducible component of the flow-matching loss means the 0.176 plateau by itself proves neither.

**Adapter-run context (different splits/objectives/conditioning — context, NOT comparators):** pre-context 0.295 SSIM (val clips); fresh side-adapter 0.62–0.66 (val clips). Direct cross-method claims (e.g. speed-to-quality ratios) are not licensed by this design and are not made.

## 2. Reliability judgment

**High for the narrow implementation/run-health claim; limited for causal or population-level conclusions.**

Grounds for the narrow claim: (i) core objective/data loss-code parity enforced by shared helpers + characterization tests; (ii) launches log-verified against predeclared acceptance criteria (the wedged-s0 path being the exception — its salvage is preliminary by definition); (iii) cohort ordinals + seeds predeclared before the run; (iv) 9 closed review cycles (30 findings resolved) behind the code; (v) training loss and cohort metrics — **two correlated measures from the same run and distribution, not independent instruments** — consistently show early saturation.

Limitations, explicitly: one training seed/run; one rollout-seed protocol; a 16-clip deterministically spaced cohort with no demonstrated representativeness and no uncertainty intervals; train-only evaluation (no matched held-out instrument); Adam moments in bf16 (the uncontrolled optax default, on record via dtype logs); review cycles establish implementation assurance, not empirical replication.

## 3. Interpretation against the decision rule

- The predeclared positive condition — decisive reconstruction improvement on training clips over the step-0 baseline — is **met on the official comparator** (0.1966 → 0.787). Under the decision rule this routes effort back to the trainable-path line rather than pipeline debugging.
- The negative condition ("cannot overfit → debug pipeline first") does not apply.
- What is NOT established: that frozen weights are the binding cause of the adapter results (confounds above); whether the model did or didn't memorize clips (no held-out instrument); whether the plateau reflects a capacity/dose ceiling (untested — see §4).

## 4. Recommended next steps

1. **Return effort to the trainable-path line** (adapter / embedding-supervision / trainable-leverage variants) — pipeline debugging is de-prioritized by this result.
2. **Candidate hypotheses for exp_02 (hypothesis-generating, not expected outcomes):** LoRA on attention/FFN, or last-N-block unfreezing, as intermediate trainable-leverage points between adapters and full FT. Note: LoRA requires introducing low-rank parameters, not merely an optimizer mask. **For an identifying design:** hold the no-action conditioning, guide scale, cohort, and evaluation protocol fixed while varying ONLY trainable leverage; restore action conditioning as a separate (or factorial) arm — otherwise the confound structure of §1 recurs.
3. **§2.4 controls: deferred, not discarded.** The plan's letter requires them only before a negative "pipeline suspect" verdict, which is not being issued — so they are unnecessary for the narrow learnability fork. They remain **required evidence before anyone interprets the 0.176 plateau as a capacity/dose ceiling or concludes full-FT cannot memorize** (30k-resume for dose; LR 2e-5 for step size; fp32 moments for the bf16-accumulator question).
4. **Cheap completions of this experiment's own record:** (a) ~~official s0~~ done (2026-07-22); (b) qualitative review of the comparison videos by Yixun/Lihan; (c) optionally, a matched held-out 16-clip cohort evaluated at step 20000 only — one v6e-8 job — which would convert the domain-vs-memorization question from unanswerable to answered.

## 5. Experiment status

Complete. Remaining: Yixun/Lihan video review (optional); merge decision (SOP isolation rule 4). Code recommendation unchanged and separable from the science: **merge** — every line went through closed review cycles, and the launcher/setup hardening benefits all future queue jobs regardless of exp_01's conclusions.

---

# Part II addendum — held-out evaluation (Planner: Claude Fable 5; v2 after Codex analysis review — all 6 findings applied)

**What Part II adds:** the two instruments Part I explicitly lacked — a full-coverage held-out loss curve and held-out rollout reconstruction.

**Findings (stated within what the instruments measure):**

1. **No aggregate held-out-loss degradation detected at the eight evaluated checkpoints.** Val loss decreases from 0.184468 (step 2500) to 0.178854 (step 20000), −0.005614 (−3.0%), decreasing at every interval. The final train (≈0.1763) and val (0.178854) losses are broadly consistent, but the two are differently aggregated estimators (10-step windowed training log with fresh stochastic draws vs exact full-val mean with fixed per-position draws), so their difference is not a calibrated train/val generalization gap.
2. **Part I's apparent train-loss plateau contains a small continued improvement under the full-val instrument.** The lower-noise full-val curve resolves a slow one-step-loss tail the train windows did not visibly show. The evaluated 16-clip train rollout means show no detectable gain after step 5000 under that single-seed protocol.
3. **T2 establishes substantial domain transfer; it does not quantify memorization.** Held-out rollout at step 20000: SSIM 0.7269 / latent MSE 0.3356 (6 fixed val clips, seed 0) vs 0.7875 / 0.2495 on the 16 train clips (same protocol/seeds). This rules out a purely train-clip-only explanation of Part I's improvement. The two cohorts are unequal, unpaired, and possibly different in difficulty; per-clip val SSIM spans 0.535–0.862; there is no step-0 val comparator — so the 0.0606 SSIM difference cannot be attributed between generalization and train-clip memorization, and the experiment remains unable to quantify how much additional memorization occurred.
4. **Per-clip spread is large** (val SSIM 0.535–0.862; position 11708 hardest). Future per-method comparisons should report per-clip numbers, not just means.

**Implication for exp_02:** 0.1789 (val loss) and 0.7269 (val-clip SSIM) are **full-FT reference targets under this run's protocol** — its budget, optimizer, and hyperparameters — not upper bounds; a lower-dimensional or regularized method can outperform a particular full-FT optimization run. Historical adapter results remain contextual (objective/conditioning/split confounds recorded in Part I) and become direct comparators only under the proposed identifying design (conditioning, guide scale, data, dose, and evaluation held fixed).

**Reliability & measurement caveats:** the −0.005614 endpoint change is stated exactly; no σ-multiple is claimed (the per-checkpoint stderr is marginal, not the stderr of paired per-example differences, and per-window losses may be correlated). T1 absolute values use one validation RNG seed; T2 uses six clips and one rollout seed. Rollout metrics compare predictions against VAE decodes of cached `z_video`, not original DROID RGB; no human qualitative video review is recorded yet. Deterministic per-position (t, ε) makes the T1 curve checkpoint-comparable by construction; n=14,636 asserted per row; both SHAs stamped in every artifact row.
