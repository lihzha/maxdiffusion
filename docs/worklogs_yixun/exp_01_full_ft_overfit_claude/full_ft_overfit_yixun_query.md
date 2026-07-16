# exp_01 `full_ft_overfit` — Driving queries

Source user: Yixun (relaying experiment intent from Lihan). Append each new query verbatim as it arrives.

## Query 1 — 2026-07-16 (from Lihan, relayed by Yixun)

**Verbatim intent:**

> Run a plain Wan TI2V overfitting sanity check: NO adapter, FULL finetuning of the Wan backbone (unfreeze transformer), on the current DROID dataset. Goal: measure how quickly the model can overfit / memorize. This is NOT the long-term method — it diagnoses whether hard-to-overfit adapter runs fail because (A) data/loss/pipeline is broken, or (B) frozen-backbone + adapter optimization is too hard.
>
> How to read results:
> - Fast overfitting (train loss down, can reconstruct training clips) → pipeline/objective roughly OK; bottleneck likely frozen backbone + adapter; continue adapter / embedding-supervision work.
> - Cannot overfit after substantial steps → debug data, loss, noise, CFG, latent alignment first; do not blame adapter structure yet.

**Summary.** A diagnostic full-finetune of the Wan2.2 TI2V 5B backbone (transformer **unfrozen**, **no adapter**) on the existing DROID latent-window dataset, run purely to measure how fast the model can memorize / overfit the training clips.

**Assumption / hypothesis being tested.** The adapter experiments so far have been hard to overfit. Two competing explanations:
- **(A) Pipeline broken** — the data / loss / noise / CFG / latent-alignment path is subtly wrong, so *nothing* can fit it.
- **(B) Optimization hard** — the pipeline is fine, and the difficulty is optimizing a small adapter on top of a *frozen* 5B backbone.

A full finetune removes the adapter + frozen-backbone variable. If a fully-trainable backbone overfits quickly → favors **(B)**. If even a full finetune cannot overfit → favors **(A)**.

**Why it needs to run.** Cheap, decisive fork in the road: the result routes the next round of effort — keep pushing adapter / embedding-supervision methods (pipeline validated), or stop and debug data/objective (pipeline suspect). Running it *before* more adapter iterations avoids burning compute on a possibly-broken pipeline.

**Decision rule (carried into `full_ft_overfit_analysis.md`):**
- **Overfits fast** (train loss drops; training clips reconstruct) → pipeline + objective roughly OK → bottleneck is frozen-backbone + adapter optimization → continue adapter / embedding-supervision work.
- **Cannot overfit after substantial steps** → debug data / loss / noise / CFG / latent alignment first → do **not** blame adapter structure yet.

**Scope note.** Diagnostic only — **NOT** the long-term method. Full-finetuning the 5B backbone is not the deployment plan; this run exists solely to isolate the pipeline-vs-adapter question.

**Open design questions for the plan phase (`plan_full_ft_overfit.md`):**
1. **Trainer path** — `model_type=TI2V` → `WanTI2VTrainer` with the full backbone trainable, vs. reusing the self-contained side-adapter trainer with the adapter disabled + backbone unfrozen. Which yields a clean "no adapter, full FT" config with the least new code?
2. **Conditioning** — the DROID cache carries `actions [32,7]`, but plain Wan TI2V does not consume actions. Does the probe condition on first-frame latent only (`z_i0`), first-frame + text, or also actions? (For pure memorization, first-frame + video target likely suffices; adding actions changes what "overfit" means.)
3. **Overfit protocol** — tiny fixed subset (e.g. 1–8 clips) vs. full DROID; step budget; LR / schedule / optimizer for full-FT of a 5B model; batch size that fits on the target TPU (and whether FSDP is still needed with the whole backbone trainable).
4. **Success metric** — train-loss curve **plus** reconstruction of the *exact training clips* (latent MSE / pixel MSE / SSIM on the memorized set) — this is a memorization probe, so held-out generalization is explicitly not the target.
