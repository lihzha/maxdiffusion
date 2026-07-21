# exp_01 `full_ft_overfit` — Results (appended as runs finish)

## Primary run: `wan-full-ft-v6e64-full-gbs256-fresh-20260719-165222` (COMPLETE 2026-07-19)

- Job `20260719-165222-62b5c10e`; commit `031228e`; v6e-64; GBS 256 (per-device 4); 20,000/20,000 steps (≈3.55 passes over 1,440,554 windows); attempt 1, zero preemptions; wall 4h40m (setup→final save); steady **1.90 steps/s** (~487 samples/s).
- **Train loss (windowed log_period=10):** 0.601 @ step 10 → 0.194 @ 500 → 0.187 @ 1000 → **plateau ~0.176–0.183 from step ~1000 through 20000** (0.1763 @ 20000; min windowed 0.1722). Grad-norm steady 0.09–0.10, no NaN/spike, LR ramped to 1e-5 by ~step 1000 (warmup fraction 0.05).
- **Eval-on-train (fresh noise/t, 4×256 batches, every 1000):** 0.1905 @ 1000 → 0.1784 @ 20000 — mirrors train loss; total post-plateau improvement ≈ −6% over 19k steps / 3.4 passes.
- **Checkpoints:** all eight 2500-multiples retained (2500…20000), ~30 GB each, `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft/wan-full-ft-v6e64-full-gbs256-fresh-20260719-165222/checkpoints/`.
- **Startup acceptance:** ALL PASS (COMMIT match, 64 devices, guide 1.0 + fresh asserts, 5.00B trainable / no adapter line, mixed-dtype param/mu/nu lines, FSDP:64 audit, eval dir = train split, honest `gbs256` run name).

## Reference context (NOT a threshold — objectives differ)

- Pre-context adapter run `wan-pre_context-…-034110` (guide-5.0 objective, GBS 512): wandb longest attempt plateaus ≈ **0.57–0.60** at 30k steps (2.25 @ 1000 → 0.5732 @ 30000). Full-FT's guide-1.0 loss is not directly comparable (CFG-amplified objective); the designed cross-condition metric is the cohort rollout below.

## Interpretation so far (full analysis in `_analysis.md` after cohort validation)

- Full-FT **optimizes healthily** (fast initial fit, clean gradients, no instability) — the training pipeline is trainable end-to-end with the backbone unfrozen.
- **No strong memorization signal in one-step loss at 3.55 passes:** a perfectly memorizing model could in principle drive this loss toward ~0 (frame-0 pin identifies the clip; v is then computable exactly), and we sit at a hard plateau ~0.176. Per plan §1/§2.4 this negative-at-budget is **inconclusive by design** — 1.44M windows in 3.55 passes is a weak memorization dose.
- **Pending decisive evidence:** cohort rollout (16 fixed train clips, step-0 pretrained baseline vs checkpoints 5000/10000/15000/20000) — reconstruction delta is the designed metric.

## Cohort validation (plan §2.3; 16 predeclared train clips, seed 0, 25-step rollout)

| Checkpoint | latent MSE | pixel MSE | SSIM | provenance |
|---|---|---|---|---|
| 0 (pretrained) | — | — | — | **pending official** (job `s0b` queued; preliminary 14/16-sample aggregate cited in `_analysis.md` only) |
| 5000  | 0.2536 | 0.01912 | 0.7873 | official `summary.json`, 16/16 |
| 10000 | 0.2573 | 0.01946 | 0.7851 | official, 16/16 |
| 15000 | 0.2537 | 0.01926 | 0.7876 | official, 16/16 |
| 20000 | 0.2495 | 0.01896 | 0.7875 | official, 16/16 |

Artifacts per checkpoint: per-sample `ground_truth.mp4` / `sample.mp4` / `comparison_gt_top_pred_bottom.mp4` / `metrics.json` + `summary.{json,csv}` under `…/validation/step_NNNNNN/`. Jobs: `_command.md` entries 9–10.
