# Plan review (v4): exp_02 overfit100
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-28

## Context loaded
- v3 review and v4 resolutions — H1, H2, and G4 closure claims.
- v4 plan — the complete artifact under review.
- prior reviews — F1–F7 and G1–G5 continuity.
- exp_02 query and worklog — user decisions and empirical probes.
- experiment SOP and announcements — review, launch, and reproducibility requirements.
- exp_01 analysis/results — checkpoint size and evaluation precedent.
- repository code — Wan encoding, trainer seams, Orbax save/restore, generator, and evaluator.

## Verdict
APPROVE-WITH-CHANGES. H1, H2, and the G4 residue are genuinely closed; the new manager remains compatible with exp_01’s Composite `params`/`opt_state`/`step` restore path, while the deterministic context table is rebuilt outside the checkpoint. No launch blocker remains, but one minor S2 evaluation-coverage contradiction should be reconciled in place before approval; no fifth review is needed.

## Per-finding check
- **H1: RESOLVED** — exact source records, fingerprinted GCS fixture, committed manifest metadata, pre-encoding preflight, and extraction/preflight tests form an executable contract.
- **H2: RESOLVED** — explicit step-list emission, `max_to_keep=None`, inherited save/restore compatibility, resume retention, exact-set testing, and the approximately 30-GB-per-checkpoint budget cover the requirement.
- **G4-residue: RESOLVED** — `C₃¹⁰⁰` is S3-only, `m_corr` is correct-mode-only, `c*` has a total deterministic tie-break, and canonical/full-set claims have separate executable rules.

## New findings
1. **I1 — MINOR — S2 text-ablation coverage conflicts between D10 and D11.** D10 limits text ablation to checkpoint 2500, whereas D11 assigns all three context modes to every S2 checkpoint `{250,500,1000,2500}`, materially changing the rollout workload without affecting the gate statistic. **Concrete change:** make both sections specify one matrix; preferably retain D10’s cheaper contract—three correct-mode seeds at every S2 checkpoint, with null/shuffled modes only at step 2500.

---

# Planner resolution (2026-07-28 — Claude Fable 5 xhigh)

- **I1 (MINOR, S2 ablation-coverage contradiction) — FIXED in place.** D11's coverage rewritten as one matrix adopting D10's cheaper contract: S2 = 3 correct-mode seeds at every checkpoint {250,500,1000,2500}, null/shuffled only at 2500; S3 unchanged (1 seed correct at intermediates; 3 seeds × 3 modes at segment finals; all windows at the final candidate). Gate statistic unaffected. Per the reviewer, no fifth review needed — the plan cycle is closed with verdict **APPROVE-WITH-CHANGES → changes applied**.
