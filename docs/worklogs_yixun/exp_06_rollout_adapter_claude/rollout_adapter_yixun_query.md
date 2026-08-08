# rollout_adapter — Yixun queries and grants (SOP artifact)

## Grant (2026-08-07, session close of the exp_04/exp_05 campaign) — exp_06 initiated

Context: the Planner's strategic synthesis of exp_01–exp_05 (recorded in the 2026-08-07 status report and the master tracker's session-close addendum), recommending exp_06 = rollout-objective training of the existing pre_context adapter on the frozen Wan2.2 5B, against the 0.2946 deployed baseline, assembling exp_03's loss family + exp_04's through-the-sampler rollout machinery + exp_05's reviewed trainer skeleton.

**Yixun, verbatim:** "go ahead for exp_06"

Planner reading: exp_06 is initiated per the SOP — Planner drafts the plan, Codex reviews it, the plan then comes back to Yixun for approval with its decision points (arm set, control policy, budgets). NO TPU work is authorized by this grant; every job in the plan carries its own approval gate per announcement 02.

## Open decision points (to be answered at plan approval — full framing in plan §11)

1. Control policy: reuse the existing 30k/GBS-512 denoising-trained checkpoint (0.2946) as the sole baseline (cheap), or retrain an update-matched one-step control alongside the arms (rigorous)?
2. Arm set at launch: B only (short-horizon differentiable rollout), or B + A (corrective scheduled sampling) + C (mix) — informed by exp_03's S1.5/S1.6 verdicts at the launch gate?
3. Budget class for the first full arm (pilot 10k steps vs baseline-matched 30k), given the ~2.5–4× per-step cost measured by exp_03's smokes?

## Grant (2026-08-07T~05:20Z) — PLAN v2.1 APPROVED; §11 decisions answered

Context: plan v2.1 presented after the Codex review cycle converged (pass 1 REQUEST-REVISION 3 BLOCKER + 9 MAJOR all accepted → v2; pass 2 two pins → v2.1; pass 3 **APPROVE-PLAN**, zero regressions/new findings), with the four §11 decisions and the Planner's recommendations.

**Yixun, verbatim:** "1–3 yes as recommended；4 defer"

Planner reading — the plan of record is **v2.1** with:
1. **Matched-C0 REQUIRED** (trained alongside R-B from the same initialization, identical data order/seed stream/GBS/updates/optimizer/eval cadence/selection rule) — accepted with its ≈1-arm cost; the historical 30k/GBS-512 checkpoint stays a benchmark row only.
2. **Arm set = R-B k=2 + matched-C0 for the pilot.** k=4 exploratory only inside an M1-measured cell, never the headline; R-A/R-C admission deferred to a future recorded decision after exp_03's S1.5/S1.6 verdicts.
3. **M3 budget = pilot 10k steps @ GBS 256** (not baseline-matched 30k).
4. **Compute-matched multi-draw one-step control: DEFERRED** — revisited only if the pilot gate passes.

**This grant authorizes NO TPU job** (announcement 02 unchanged): it opens the T1–T7 code rounds. M1 (fit probe) is requested separately at its own pushed SHA with the full pre-launch package.

## Grant (2026-08-08T~04:05Z / bedtime) — sleep-window standing approval, ~10 h

**Yixun, verbatim:** "Currently I will go to bed, so I will approve everythin after your recommendation util I wake up, potential 10h from now. Please go ahead"

Planner reading (recorded BEFORE acting on it):
1. The grant covers, for ~10 h, every approval the Planner would otherwise seek AND recommend — it is a conditional grant riding on the Planner's own recommendations, not a carte blanche beyond them.
2. **Job ladder, in order, each rung only on its predecessor's PREDECLARED pass:** M1 fit probe (v6e-8, ~1 h) may launch once both rework tracks re-review clean and commit; M2 learnability probe (v6e-8, 32 ex, ≤2k steps + matched-C0) may launch on M1's authorization verdict; M3 (the Yixun-approved pilot budget class: 10k steps @ GBS 256, v6e-64) may launch ONLY if M2 passes its predeclared numerical continuation rule — realistically beyond this window. Every launch gets its `_command.md` entry, acceptance criteria, and ScheduleWakeup monitoring at launch time.
3. Submissions are attempted directly under this grant (the classifier permitted J1b/J1c under explicit in-conversation grants); if the classifier blocks (issue #10), the exact `! bash` command is queued for wake-up and everything else continues.
4. Non-TPU work needing no grant proceeds regardless: rework completion, re-reviews, strengthens, commits, and the exp_04 P4 / exp_05 P4' REPORT DRAFTS (Coder-written; their Codex analysis reviews are DEFERRED past wake-up to preserve reviewer quota for exp_06's critical path).
5. **Failure policy overnight:** infra failures auto-resubmit per the standing policy; real bugs get fix cycles per SOP; gsutil/gcloud reauth expiry (issue #6, ~4-hourly) is an ALARM the Planner cannot clear — anything auth-blocked is queued for wake-up with a clear morning summary.
