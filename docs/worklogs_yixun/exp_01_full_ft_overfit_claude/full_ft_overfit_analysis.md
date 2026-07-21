# exp_01 `full_ft_overfit` — Analysis (Planner: Claude Fable 5; v2 after Codex analysis review)

**Status: PROVISIONAL** — primary training complete and log-verified; final analysis pending the official 16/16 step-0 validation (job `s0b`, queued). This v2 incorporates all six findings of `full_ft_overfit_codex_analysis_review.md` (REQUEST-REVISION → resolutions appended there).

**Question (Lihan):** can the Wan TI2V training pipeline be overfit at all when nothing is frozen — i.e. is the adapters' difficulty caused by (A) a broken data/loss/pipeline, or (B) the difficulty of optimizing through a frozen 5B backbone?

**Verdict (provisional):** the strong "pipeline cannot fit" form of **(A) is disfavored**; the results are consistent with — and, under the predeclared decision rule, **favor (B)** — but this experiment does **not causally isolate** frozen-backbone optimization as the adapters' binding constraint. The adapter comparisons differ in split (val vs train clips), objective (guide-5 vs guide-1), and conditioning (actions vs none); shared helpers establish core loss-code parity, not complete objective/conditioning parity.

## 1. Evidence

**Training dynamics (official, 20k steps @ GBS 256, 3.55 passes):** loss 0.60 → 0.19 within ~500 steps, hard plateau 0.176–0.183 thereafter (−6% over the remaining 19k steps); gradients steady (norm 0.09–0.10), zero instability, 1.90 steps/s, no preemptions. Every §6 acceptance criterion passed on real-hardware logs.

**Cohort reconstruction (official for trained checkpoints; 16 fixed train clips, identical seeds):** SSIM ≈ **0.787**, latent MSE ≈ **0.25** at ALL of steps 5000/10000/15000/20000 — flat to three decimals. The pretrained comparator is **preliminary**: 14/16 samples salvaged from the original s0 job, which was **queue-wedged after 8 preempted attempts** (not merely "a preempted attempt"); official job re-queued. Preliminary aggregate: SSIM **0.20** (range 0.13–0.31), latent MSE **3.51**. Quantitatively the gap is robust — two additional samples cannot raise a ~0.20 mean near 0.79 — but the predeclared protocol designates the complete 16/16 result as the primary comparator, so the verdict stays provisional until it lands. **Qualitative video review has not yet been performed** (pull commands in the HTML report); no claim is made here that clips were visually verified as reconstructed.

**What the run demonstrates, stated narrowly:** the no-action, guide-1, full-FT path achieves a substantial training-cohort reconstruction improvement, with no further measured improvement after step 5000. **This experiment cannot separate domain learning from per-clip memorization** — all rollout evaluation is on training clips; a matched held-out cohort (same protocol, unseen clips) would be required for that distinction, and the unknown irreducible component of the flow-matching loss means the 0.176 plateau by itself proves neither.

**Adapter-run context (different splits/objectives/conditioning — context, NOT comparators):** pre-context 0.295 SSIM (val clips); fresh side-adapter 0.62–0.66 (val clips). Direct cross-method claims (e.g. speed-to-quality ratios) are not licensed by this design and are not made.

## 2. Reliability judgment

**High for the narrow implementation/run-health claim; limited for causal or population-level conclusions.**

Grounds for the narrow claim: (i) core objective/data loss-code parity enforced by shared helpers + characterization tests; (ii) launches log-verified against predeclared acceptance criteria (the wedged-s0 path being the exception — its salvage is preliminary by definition); (iii) cohort ordinals + seeds predeclared before the run; (iv) 9 closed review cycles (30 findings resolved) behind the code; (v) training loss and cohort metrics — **two correlated measures from the same run and distribution, not independent instruments** — consistently show early saturation.

Limitations, explicitly: one training seed/run; one rollout-seed protocol; a 16-clip deterministically spaced cohort with no demonstrated representativeness and no uncertainty intervals; train-only evaluation (no matched held-out instrument); step-0 provenance incomplete pending `s0b`; Adam moments in bf16 (the uncontrolled optax default, on record via dtype logs); review cycles establish implementation assurance, not empirical replication.

## 3. Interpretation against the decision rule

- The predeclared positive condition — decisive reconstruction improvement on training clips over the step-0 baseline — is **met on the preliminary comparator** and extremely unlikely to reverse with the official one. Under the decision rule this routes effort back to the trainable-path line rather than pipeline debugging.
- The negative condition ("cannot overfit → debug pipeline first") does not apply.
- What is NOT established: that frozen weights are the binding cause of the adapter results (confounds above); whether the model did or didn't memorize clips (no held-out instrument); whether the plateau reflects a capacity/dose ceiling (untested — see §4).

## 4. Recommended next steps

1. **Return effort to the trainable-path line** (adapter / embedding-supervision / trainable-leverage variants) — pipeline debugging is de-prioritized by this result.
2. **Candidate hypotheses for exp_02 (hypothesis-generating, not expected outcomes):** LoRA on attention/FFN, or last-N-block unfreezing, as intermediate trainable-leverage points between adapters and full FT. Note: LoRA requires introducing low-rank parameters, not merely an optimizer mask. **For an identifying design:** hold the no-action conditioning, guide scale, cohort, and evaluation protocol fixed while varying ONLY trainable leverage; restore action conditioning as a separate (or factorial) arm — otherwise the confound structure of §1 recurs.
3. **§2.4 controls: deferred, not discarded.** The plan's letter requires them only before a negative "pipeline suspect" verdict, which is not being issued — so they are unnecessary for the narrow learnability fork. They remain **required evidence before anyone interprets the 0.176 plateau as a capacity/dose ceiling or concludes full-FT cannot memorize** (30k-resume for dose; LR 2e-5 for step size; fp32 moments for the bf16-accumulator question).
4. **Cheap completions of this experiment's own record:** (a) official s0 (queued — updates `_results.md`/HTML when it lands); (b) qualitative review of the comparison videos by Yixun/Lihan; (c) optionally, a matched held-out 16-clip cohort evaluated at step 20000 only — one v6e-8 job — which would convert the domain-vs-memorization question from unanswerable to answered.

## 5. Experiment status

Primary training complete; final analysis pending official step-0 validation. Remaining: s0b lands → table/HTML update; Yixun/Lihan video review; merge decision (SOP isolation rule 4). Code recommendation unchanged and separable from the science: **merge** — every line went through closed review cycles, and the launcher/setup hardening benefits all future queue jobs regardless of exp_01's conclusions.
