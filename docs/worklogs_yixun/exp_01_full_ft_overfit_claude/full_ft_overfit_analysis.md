# exp_01 `full_ft_overfit` — Analysis (Planner: Claude Fable 5)

**Question (Lihan):** can the Wan TI2V training pipeline be overfit at all when nothing is frozen — i.e. is the adapters' difficulty caused by (A) a broken data/loss/pipeline, or (B) the difficulty of optimizing through a frozen 5B backbone?

**Verdict: (A) is strongly disfavored; (B) is supported.** The unfrozen pipeline trains fast, stably, and to dramatically better reconstruction than any adapter run produced. One nuance matters: what the model does is rapid *domain fitting*, not per-clip *memorization* — and that distinction shapes the recommendation.

## 1. Evidence

**Training dynamics (official, 20k steps @ GBS 256, 3.55 passes):** loss 0.60 → 0.19 within ~500 steps, hard plateau 0.176–0.183 thereafter (−6% over the remaining 19k steps); gradients steady (norm 0.09–0.10), zero instability, 1.90 steps/s, no preemptions. Every §6 acceptance criterion passed on real-hardware logs.

**Cohort reconstruction (official for trained checkpoints; 16 fixed train clips, identical seeds):** SSIM ≈ **0.787** and latent MSE ≈ **0.25** at ALL of steps 5000/10000/15000/20000 — flat to three decimals. The pretrained baseline is preliminary (14/16 samples salvaged from a preempted attempt; official job queued): SSIM **0.20** (range 0.13–0.31), latent MSE **3.51**, pixel MSE **0.199**. The two missing samples cannot move a 0.20 mean anywhere near 0.79; the qualitative gap is not in doubt.

**Contrast with adapter runs (context, not thresholds — different splits/objectives):** pre-context reached 0.29 SSIM (val clips, 30k steps); fresh side-adapter 0.62–0.66 (val clips). Full-FT on train clips: 0.79, reached ≥4× faster in steps.

## 2. Reliability judgment

I consider the result reliable. Grounds: (i) objective/data parity with the adapter runs is enforced by shared code (round-1 helpers + characterization tests), not by claim; (ii) all launches log-verified against predeclared acceptance criteria; (iii) the cohort protocol was predeclared (ordinals + seeds in `_params_set_up.md`) before the run existed; (iv) 9 closed review cycles (30 findings, all resolved on record) stand behind the code; (v) the flat checkpoint curve is internally consistent with the loss plateau — two independent instruments agreeing.

**Caveats, explicitly:** (a) step-0 baseline is preliminary until `s0b` lands (results table withholds it accordingly); (b) Adam moments ran in bf16 (the uncontrolled optax default — on record via the dtype logs; the fp32 control exists but see §4 for why I do not recommend spending on it); (c) cohort is train-clips-only by design — this experiment says nothing about generalization, deliberately; (d) the loss plateau's absolute level (0.176) has an unknown irreducible component (fresh-noise/timestep entropy at low σ); we did not attempt to decompose it.

## 3. Interpretation against the decision rule

- **"Fast overfitting → pipeline OK":** satisfied in the sense that matters. The pipeline, fed exactly the adapter runs' data and objective, converts 4.7 hours of unfrozen training into a 0.20→0.79 SSIM reconstruction jump on training clips. Nothing about the data, loss, noise handling, CFG bypass, or latent alignment prevents fitting.
- **"Cannot overfit → debug pipeline first":** does NOT apply — but note the model also does not *memorize*: reconstruction and loss both saturate by ~step 5000 and stay flat for 2.7 more epochs. A 5B model at LR 1e-5 over 1.44M windows in 3.55 passes fits the DROID domain hard and then stops improving. True per-clip memorization (loss→0) would need a far higher dose (many more epochs, higher LR, or a tiny subset) and is not needed to answer Lihan's question.
- **Therefore:** the adapters' hard-to-overfit behavior is an optimization/capacity phenomenon of the frozen-backbone setup — hypothesis **(B)** — not a broken pipeline. The backbone's weights need to move (or be low-rank-adapted much closer to the weights) for this domain; injecting residual streams/context tokens against frozen weights has been the binding constraint.

## 4. Recommended next steps (Planner's ordering)

1. **Return to the adapter/embedding-supervision line with this evidence in hand** — the pipeline is exonerated; design effort should go to giving the trainable path more leverage over the backbone (e.g. LoRA on attention/FFN weights, last-N-block unfreezing, or higher-capacity adapters), not to further pipeline debugging.
2. **A cheap, high-value follow-up suggested by this run:** full-FT reached 0.79-SSIM domain fit in <5 h of v6e-64. A LoRA or partial-unfreeze variant under the SAME harness (this experiment's trainer needs only an optimizer-mask change) would map the curve between "adapter: 0.3–0.66" and "full-FT: 0.79" and likely find a deployable point. That would be exp_02.
3. **Escalation controls (§2.4): NOT recommended.** They exist to shore up a negative result; the result is positive. Spending v6e-64 time on 30k-resume/LR-2e-5/fp32-moment controls buys rigor against a conclusion nobody is doubting.
4. **Action conditioning remains untested here by design** (first-frame + video only). exp_02 should restore action conditioning in whatever trainable-path variant is chosen — that is the actual product target.

## 5. Experiment status

Operationally complete pending: official s0 summary (queued; will update `_results.md` table row), HTML report (artifact 12), and Yixun's merge decision (SOP isolation rule 4). Code recommendation: **merge** — every line went through closed review cycles, the shared-helper refactor is characterization-tested, and the launcher/setup hardening benefits all future queue jobs regardless of exp_01's science.
