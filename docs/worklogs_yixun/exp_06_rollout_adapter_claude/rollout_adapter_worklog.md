# rollout_adapter — worklog (SOP artifact)

## 2026-08-07T03:20:00Z — exp_06 initiated; plan v1 drafted; plan review dispatched

- **Provenance:** Yixun's "go ahead for exp_06" (query doc, verbatim) following the exp_01–exp_05 strategic synthesis. Branch `claude-exp_06_rollout_adapter-20260807` from exp_05's tip `0f505d3` (carries exp_04 through the fix/R11 merge + exp_05 S1–S10a — the reviewed trainer/launcher/eval substrate). exp_03's sampler/loss modules arrive at round T1 via a pinned-SHA one-way merge.
- **Plan v1:** the objective-swap experiment — rollout-based losses (exp_03's family) on the UNCHANGED pre_context adapter over the frozen 5B, gated at DEV-64 +0.05 SSIM over the re-measured baseline; probes-first job ladder (fit → learnability → pilot arm); the campaign's runbook rules (issues #10–#13) baked in as standing discipline.
- **Next** — Codex plan review (pass 1) → revisions → Yixun approval with §11's four decisions.

## 2026-08-07T04:20:00Z — Plan review pass 1: REQUEST-REVISION (3 BLOCKER + 9 MAJOR, all accepted) → plan v2; pass 2 dispatched

- Headline corrections: trainer base is the side-adapter trainer (S7 contributes generalized utilities only); exp_03 dependency = pinned-SHA blob import + kernel extraction with equivalence tests, NOT a branch merge (the reviewer measured ~44k insertions for a merge); the ACTION-USE GATE is now mandatory (true-vs-shuffled/zero actions, paired, CI-gated) — the reviewer confirmed the Planner-planted concern and supplied the design; matched-C0 required for causal objective-only claims; paired-delta +0.05 gate with anchor-reproduction protocol; explicit CFG gradient contract (no z-stop-grad across steps, FD oracle); the exp_05-unstall claim withdrawn (evaluator = exp_06-owned filename, tripwire untouched); k=2 primary; per-job approvals strictly at pushed SHAs.

## 2026-08-07T05:10:00Z — Plan review CONVERGED: APPROVE-PLAN at v2.1 (3 passes: 12 findings → 2 pins → clean) → to Yixun for approval

- The plan now goes to Yixun with §11's four decisions (matched-C0 cost; arm set; M3 budget class; optional compute-matched control). Plan approval authorizes NO job — T1–T7 code rounds begin on approval; M1 is the first launch request, at its own pushed SHA.

## 2026-08-07T05:25:00Z — PLAN v2.1 APPROVED by Yixun (§11: 1–3 as recommended, 4 deferred) → T1 dispatched

- **Plan of record:** v2.1. Decisions locked: matched-C0 required; pilot = R-B k=2 + matched-C0; M3 budget = 10k steps @ GBS 256; compute-matched control deferred. Grant verbatim in the query doc. No TPU job authorized by the approval.
- **Coder:** a FRESH Opus subagent takes exp_06 (the exp_04/05 Coder is retired to that campaign — one persistent Coder per experiment, and its context is ~790k tokens deep).
- **Next** — T1 `exp03-imports` (pinned-SHA blob import + characterization), then T2 `loss-kernels`.
