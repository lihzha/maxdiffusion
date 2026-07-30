# exp_02 `overfit100` — Results (appended as runs finish)

## S2 gate run (train10: 10 episodes / 167 windows; v6e-8; 2,500 steps, GBS 32, LR 1e-5)

**Training** (job `20260730-173902`, 36 min): loss **0.533 → 0.061**, still decreasing at 2,500 steps (0.170 @250 → 0.154 @500 → 0.138 @1000 → 0.061 @2500); grad norm 4.3 → 0.15; 1.82 steps/s. Reference: exp_01 (full DROID, no text) plateaued at 0.176 — this regime is 3× below that floor and unconverged.

**Gate evals** (jobs `20260730-1842xx`, role `s2_gate`, 10 canonical windows × seeds {0,1,2}, 25-step rollouts; artifacts in `overfit100_s2_gate_artifacts/`, role validation OK at every checkpoint):

| checkpoint | mean m_corr | min | max |
|---|---|---|---|
| 250 | 0.7665 | 0.7062 | 0.8929 |
| 500 | 0.7993 | 0.7444 | 0.9008 |
| 1000 | 0.8347 | 0.7827 | 0.9183 |
| 2500 | **0.8896** | 0.8179 | **0.9426** |

Every window improved monotonically across all four checkpoints (10/10).

**Predeclared gate rule:** (i) mean m(w,2500) ≥ 0.70 → **0.8896 PASS** (margin +0.19); (ii) growth 250→2500 ≥ 0.15 → **+0.1230 MISS** (by 0.027); (iii) max ≥ 0.85 → **0.9426 PASS**. **Formal verdict: STOP & ANALYZE** (plan §7 R2 → escalation requires Yixun's approval).

**Text-ablation at 2500** (mean SSIM over 10 windows × 3 seeds): correct **0.8895** / null 0.8398 / shuffled 0.8342. The correct-text advantage is +0.050 vs null and +0.055 vs shuffled, and wrong text is *worse than no text* — the model genuinely uses language, with a modest effect size consistent with the duplicate-audit finding (58/100 taxonomy-label instructions) and first-frame fingerprints carrying much of the identification.

**Known defect found by the run:** the auxiliary RGB/VAE-ceiling path failed on all rows (`AttributeError: 'str' object has no attribute 'parent'` — a str-vs-Path bug in the aux fetch). The D5 machinery reported it exactly as designed (aux_status per row; run completed). Primary metrics unaffected. Fix queued before any S3 eval.
