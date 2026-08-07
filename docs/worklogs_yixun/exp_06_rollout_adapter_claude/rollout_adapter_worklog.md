# rollout_adapter — worklog (SOP artifact)

## 2026-08-07T03:20:00Z — exp_06 initiated; plan v1 drafted; plan review dispatched

- **Provenance:** Yixun's "go ahead for exp_06" (query doc, verbatim) following the exp_01–exp_05 strategic synthesis. Branch `claude-exp_06_rollout_adapter-20260807` from exp_05's tip `0f505d3` (carries exp_04 through the fix/R11 merge + exp_05 S1–S10a — the reviewed trainer/launcher/eval substrate). exp_03's sampler/loss modules arrive at round T1 via a pinned-SHA one-way merge.
- **Plan v1:** the objective-swap experiment — rollout-based losses (exp_03's family) on the UNCHANGED pre_context adapter over the frozen 5B, gated at DEV-64 +0.05 SSIM over the re-measured baseline; probes-first job ladder (fit → learnability → pilot arm); the campaign's runbook rules (issues #10–#13) baked in as standing discipline.
- **Next** — Codex plan review (pass 1) → revisions → Yixun approval with §11's four decisions.
