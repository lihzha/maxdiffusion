
## K1-1 — P1' positive reconstruction study + basin probe (2026-08-05T20:33Z, LAUNCHED — submitted by Yixun via `!`, issue #10; grant recorded verbatim in the query doc)

- Queue job: `20260805-203303-d3a02688-exp05-k1-pos-yixun` (v6e-8, worker0-only); authoritative record confirmed PENDING at `gs://v6_east1d/tpu-job-queue/jobs/20260805-203303-d3a02688-exp05-k1-pos-yixun/status.json` (created 2026-08-05T20:33:17Z).
- Submitted tip: `9d326e8b222c6f9b8c53cbc8790bf642684a256c` (= the S10a ledger commit; `src/` + `bash_scripts/` identical to the reviewed S10a round `102ae84`; COMMIT env carries the full SHA).
- Submission: `submit_k1.sh` (archived alongside this file; reproduced in the job record above) — FOUR sequential phases in one job via the S10a launcher `run_wan_pos_inversion.sh`: (1) SMOKE capacity POS_SMOKE_EXAMPLES=2 → `gs://v6_east1d/datasets/droid_wan_pos_context/k1/smoke`; (2) adequacy_probe (first-8 DEV, approved grid) → `…/k1/adequacy`; (3) full capacity POS_COHORT=dev64, six B-arms + L_pos∈{1,8} ablation, DEV adequacy adopted → `…/k1/capacity`; (4) full capacity POS_COHORT=trainfit16, SAME DEV adequacy adopted → `…/k1/capacity_trainfit` (distinct root per the follow-up reviewer's operational note). Remaining knobs at launcher defaults = the audited plan values.
- Acceptance criteria: predeclared in the worklog's K1-package entry (2026-08-05T20:35Z, criteria 1–8; criterion 1's tip = this SHA). Failure triage per SOP: infra ⇒ auto-resubmit unchanged (standing policy); real bug ⇒ fix cycle.
- Monitoring: status.json poll every 10 min from the session (reauth-in-stderr = ALARM); `tpu status/logs 20260805-203303-d3a02688-exp05-k1-pos-yixun`.

## K1-1 outcome (2026-08-05T22:50Z, **FAILED — REAL BUG, inherited stale copy** (attempt 2 APPLICATION_ERROR after 23s of command; attempt 1 infra preemption))

- **Crash:** `ValueError: Requested key code_sha, not in config` at `mode_kwargs` (run_wan_null_inversion.py:654) — the EXACT J1-1 bug class (issue #11: three-arg getattr never falls back on HyperParameters). exp_05's copy of the entrypoint predates exp_04's fix round `925ee17`: the merge-interim took exp_04 at the R10 boundary, the fix landed after, and no propagation step existed. The positive-slot smoke got as far as backend load + revision resolution (the S10a launcher, preflights, prefetch, and config all worked), then died at the first shared-code code_sha read.
- **Nothing published** (fail-closed before any artifact) — `…/k1` root is clean; K1-2 can reuse it.
- **Remediation:** merge-interim-2 — one-way exp_04 @ `27efcd1` (the fix + docs) → this branch, reconciling the dual-touch entrypoint (fix sites + exp_05's additive positive branch), full combined suite, commit; then K1-2 relaunch at the new tip (same four-phase runbook — unaffected by the flaw class J1-2 hit: K1 has no selection-consuming phase). Sequenced AFTER S7's in-flight final fix commits (clean-worktree requirement for the merge).

## K1-2 — relaunch at the merge-2 tip (2026-08-06T01:24Z, LAUNCHED — submitted by Yixun via `!`)

- Queue job: `20260806-012445-20a744fe-exp05-k1-pos-yixun` (v6e-8, worker0-only); authoritative record confirmed at `gs://v6_east1d/tpu-job-queue/jobs/20260806-012445-20a744fe-exp05-k1-pos-yixun/status.json`.
- Submitted tip: `bb845eaf43de2f05abbf99dfd4344a51bc2930bc` (= the merge-2 ledger commit; carries the K1-1 fix — exp_04's reviewed `optional_config_value` discipline now enforced in this branch by the AST guard; suite 1353).
- Submission: `submit_k1.sh` VERBATIM from K1-1 (four phases: smoke → adequacy → capacity dev64 → capacity trainfit16; `…/k1` root reused — K1-1 published nothing). Acceptance criteria: the K1-package entry's 1–8 with criterion 1's tip = this SHA.
- Monitoring: 10-min status.json poll (reauth = ALARM); `tpu status/logs 20260806-012445-20a744fe-exp05-k1-pos-yixun`.

## K1-2 outcome (2026-08-06T~08:00Z, **SUCCEEDED** — all four phases; attempt 1 after one infra preemption)

- Artifacts (all provenance-bound, run-level JSONs present): `…/k1/{smoke, adequacy, capacity, capacity_trainfit}` — gate_tables.json + selection.json + run_report.json + b1/b2 record shards under both capacity roots. Adequacy: `adequacy_report.json` (recipe plateau: recipe-limited; 1822s).
- Acceptance criteria 1–8: MET (worker at the merge-2 tip; smoke published; adequacy full-evidence; both capacity cohorts complete, zero unexplained quarantines; gates evaluated; the gates' verdict IS the acceptance — outcome recorded in the worklog result entry).
