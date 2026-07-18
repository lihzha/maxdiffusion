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

## Query 2 — 2026-07-16 (design decisions, from Yixun)

**Verbatim:**

> Planner use Fable as the planner, the same architecture as I have mentioned in @docs/worklogs_yixun/experiment_SOP.md . for design, make this no adapter, purely unfreeze backbone of wan (fin tune), wan transformer trainable, and contition the overfit probe on just first-frame + video. Overfit on full DROID.

**Decisions locked:**
- **Planner** = Claude Fable 5 (max effort), per the SOP roles table (session switched to Fable 5 for planning).
- **Architecture** = NO adapter; the Wan2.2 TI2V 5B **transformer backbone is fully unfrozen and trainable** (full finetune). Answers open question 1's direction: plain backbone, no adapter modules at all.
- **Conditioning** = first-frame latent (`z_i0` pinning) + video target only. **No action conditioning.** (Text stays the fixed null-prompt embedding the pipeline already uses — it is a constant, not a per-sample condition.) Answers open question 2.
- **Data** = overfit on **full DROID** train split (not a tiny subset). Answers the subset half of open question 3; step budget / LR / batch resolved in the plan.

## Query 3 — 2026-07-16 (pre-approval, from Yixun)

**Verbatim:**

> Pre-approve conditional on APPROVE verdict, start coder round 1 immediately

**Interpretation:** plan v2 is approved by Yixun **conditional on the pending Codex re-review returning APPROVE**. On APPROVE: start Coder round 1 (shared-objective-helpers) immediately, no further ask. On any other verdict (APPROVE-WITH-CHANGES / REQUEST-REVISION): the conditional does NOT trigger — resolve findings and return to Yixun.

## Query 4 — 2026-07-18 (launch approvals, from Yixun)

**Verbatim:** "Approve smoke" → attempt 1 (per-device 8) OOM'd on v6e-8 by 44MB; then, after the reviewed launcher fix: "Approve smoke 2 + fit probe conditional on pass".

**Scope:** smoke attempt 2 (v6e-8, per-device 1 / GBS 8, 1 step, storage-light) approved unconditionally; the v6e-64 fit probe (per-device 8 / GBS 512, 1 step, storage-light) is pre-approved **conditional on smoke 2 passing its worklog acceptance criteria** (log-verified, not merely queue-SUCCEEDED). The 10k full run remains a separate future approval.

## Query 5 — 2026-07-18 (full-run pre-approval, from Yixun)

**Verbatim:** "Approve full run conditional on fit probe pass"

**Scope:** the primary 10k-step full run (v6e-64, GBS 512, LR 1e-5, fresh noise, guide 1.0, checkpoints every 2500 with keep-period 2500) is pre-approved, **conditional on the fit probe passing its acceptance criteria log-verified** (not merely queue-SUCCEEDED). Escalation-protocol runs (§2.4: 30k resume, LR control, fp32 control) and cohort-validation jobs remain separate future approvals.

## Query 6 — 2026-07-19 (post-fix full-run blessing, from Yixun)

**Verbatim:** "Approve full run from post-fix commit"

**Scope:** supersedes Query 5's implicit SHA — the 10k full run launches from the commit containing the reviewed setup.sh apt-hardening (mini-cycle 7), once BOTH hold: fit probe passed (any attempt) AND cycle 7 closed. All other full-run parameters unchanged from `_params_set_up.md`.
