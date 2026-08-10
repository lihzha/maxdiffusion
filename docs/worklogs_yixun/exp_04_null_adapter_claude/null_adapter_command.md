# null_adapter_command.md — exact reproduction commands (SOP artifact 7)

## J0-1 — cohort-manifest build (2026-08-05T04:50Z, **FAILED — INFRASTRUCTURE**: TF/ADC reauth `invalid_rapt`; fail-closed, nothing written; see worklog. Re-run pending ADC refresh.)

- Commit: `7199feb99514d5c4e460e84629b133566f6624d7` (branch `claude-exp_04_null_adapter-20260803`, clean worktree)
- Host: local macOS (darwin), scratchpad venv python 3.11 (tensorflow 2.21, numpy), gsutil authenticated (yh4742@princeton.edu)
- Log: `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_2026-08-05_04:48:27.log`

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter && \
PYTHONPATH=src <venv>/bin/python <scratchpad>/j0_driver.py 2>&1 | tee docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_2026-08-05_04:48:27.log
# where j0_driver.py (archived alongside this file as j0_driver.py after the run) calls:
# build_j0_manifests('gs://v6_east1d/datasets/droid_wan_side_adapter/val',
#                    'gs://v6_east1d/datasets/droid_wan_side_adapter/train',
#                    'docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests',
#                    builder_sha='7199feb99514d5c4e460e84629b133566f6624d7')
# <venv> = /private/tmp/claude-501/-Users-yixunhu-Home-maxdiffusion/800fed95-7c3f-418d-b779-9914ed8480b4/scratchpad/venv
```

## J0-2 — cohort-manifest build re-run (2026-08-05_14:59:41Z, LAUNCHED at 2026-08-05_15:27:35Z — both credentials verified live (gsutil listing + Generation stat; ADC refreshed earlier). Same driver as J0-1; log: null_adapter_2026-08-05_15:27:35.log)

- Commit: `7199feb99514d5c4e460e84629b133566f6624d7` (J0 runs the committed builder; working tree carries only uncommitted R10 files not touched by J0)
- Same command as J0-1; log: `null_adapter_2026-08-05_14:59:41.log`


## J1-1 — P1 capacity study + basin probe (2026-08-06, **FAILED — REAL BUG**: HyperParameters getattr at run_wan_null_inversion.py:611; smoke-phase crash after a successful pipeline load; fix cycle + J1-2 relaunch per worklog)

- Queue job: `20260805-181744-61377ea2-exp04-j1-null-yixun` (v6e-8, worker0-only); authoritative record confirmed PENDING at `gs://v6_east1d/tpu-job-queue/jobs/20260805-181744-61377ea2-exp04-j1-null-yixun/status.json` (created 2026-08-05T18:17:48Z). The CLI's initial "not found" was a CLI-side quirk; the GCS record rules (issue-#2 discipline).
- Submitted tip: `3616a94` (docs-only delta over the parity-audited `f06dfc1` — `src/` + `bash_scripts/` byte-identical; COMMIT env carries the full submitted SHA).
- Submission: `tpu create v6 -n 8 --worker0-only --name exp04-j1-null-yixun --code-dir . --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" --env COMMIT=<tip> ... -- bash -c '<four-phase runbook>'` — four sequential phases in one job: (1) SMOKE capacity NULL_SMOKE_EXAMPLES=2 → `gs://v6_east1d/datasets/droid_wan_null_adapter/j1/smoke`; (2) verify_replay over the smoke shards; (3) adequacy_probe → `…/j1/adequacy`; (4) full capacity, six arms, NULL_A3_MEASURE=True, adoption via discovered URI → `…/j1/capacity`. Remaining knobs at launcher defaults (= plan values; `null_adapter_params_set_up.md`). Full script preserved verbatim in the session scratchpad (`submit_j1.sh`) and reproduced by this entry.
- Monitoring: `tpu status/logs 20260805-181744-61377ea2-exp04-j1-null-yixun`; reauth-in-stderr = ALARM (issue #6); auto-resubmit on infra failure per the standing grant.

## J1-2 — P1 capacity study + basin probe RELAUNCH (2026-08-05T19:43Z, LAUNCHED — submitted by Yixun via `!` after the session's classifier block, issue #10; running the reviewed J1-crash fix)

- Queue job: `20260805-194305-c6536960-exp04-j1-null-yixun` (v6e-8, worker0-only); authoritative record confirmed PENDING at `gs://v6_east1d/tpu-job-queue/jobs/20260805-194305-c6536960-exp04-j1-null-yixun/status.json` (created 2026-08-05T19:43:19Z).
- Submitted tip: `27efcd13a960a541807d098c3b8aa6a0bc4fc75f` (docs-only delta over the fix commit `925ee17` — `src/` + `bash_scripts/` byte-identical to the reviewed/committed fix round; `COMMIT` env carries the full SHA). Grant reading recorded in the worklog: the conditional J1 grant re-evaluated at this SHA (P0 954-green + parity audit untouched by the config-access delta); Yixun's own submission is the sign-off.
- Submission: VERBATIM J1-1 four-phase runbook via the archived `submit_j1.sh` (reproduced in the J1-1 entry above): (1) SMOKE capacity NULL_SMOKE_EXAMPLES=2 → `…/j1/smoke`; (2) verify_replay over the smoke shards; (3) adequacy_probe → `…/j1/adequacy`; (4) full capacity, six arms, NULL_A3_MEASURE=True, adoption via discovered URI → `…/j1/capacity`. Artifact root confirmed EMPTY pre-launch (J1-1 failed closed). Acceptance criteria: unchanged from the J1-1 entry (worklog 2026-08-05T18:16Z, criteria 1–7) except criterion 1's commit = this tip.
- Monitoring: status.json poll every 10 min from the session (reauth-in-stderr = ALARM, issue #6); `tpu status/logs 20260805-194305-c6536960-exp04-j1-null-yixun`. Failure triage per SOP: infra ⇒ auto-resubmit unchanged (standing grant); real bug ⇒ fix cycle.

## J1-2 outcome (2026-08-05T23:18Z, **FAILED — RUNBOOK DESIGN FLAW** (attempt 4 APPLICATION_ERROR; attempts 1–3 infra preemptions, auto-retried by the queue). THE PIPELINE ITSELF PASSED.)

- **What actually happened:** PHASE 1 (smoke capacity, n=2) COMPLETED — pipeline load, inversion, per-step null optimization, replay, decode, SSIM, gates, provenance-bound publication all ran on TPU at production settings. The fix from J1-1 held (no config-access crash). The n=2 gates then did their fail-closed job — G1 (ssim_ci_low), A1-probe below the 0.70 floor and 0.7× transfer ratio, G2 (mean_ssim, ssim_ci_low) ⇒ selection verdict STOP — CI-based gates on two examples fail near-tautologically; NOT a scientific result. PHASE 2 (verify_replay) then read that selection and correctly refused: `selected_arm` raised "no selected arm to cache or verify" (null_adapter_modes.py:312) ⇒ exit 1. Gates and verify each behaved exactly as reviewed; the RUNBOOK wired smoke→verify through a selection that cannot name an arm at smoke scale. Classified: Planner runbook design flaw (same class as the TRAINFIT omission — runbook-level, zero code defects).
- **Partial artifacts:** `…/j1/smoke` holds the published smoke shards/tables/selection (immutable; archived as-is — they are the proof of phase 1).
- **Remediation — J1-3 (`submit_j1_v3.sh`, archived in scratchpad + reproduced here):** (a) phase 2 GUARDED — verify runs only if the smoke selection names an arm; STOP ⇒ loud skip (read-back verification discharged by J2 at full scale — acceptance criterion 3 amended accordingly, flagged to Yixun); (b) NEW phase 5 = capacity on TRAINFIT-16 with the same DEV adequacy, distinct root (folds in J1-2b — criterion 5 now complete in one job); (c) fresh artifact root `…/j1r2` (immutable-destination collision avoidance). No repo code changes ⇒ same tip `27efcd1`; relaunch under the standing grant via Yixun's `!`.

## J1-3 — corrected five-phase runbook (2026-08-06T00:55Z, LAUNCHED — submitted by Yixun via `!`)

- Queue job: `20260806-005532-0f751b8b-exp04-j1-null-yixun` (v6e-8, worker0-only); authoritative record confirmed at `gs://v6_east1d/tpu-job-queue/jobs/20260806-005532-0f751b8b-exp04-j1-null-yixun/status.json`.
- Submitted tip: `27efcd13a960a541807d098c3b8aa6a0bc4fc75f` — UNCHANGED from J1-2 (the runbook flaw was script-level; the pipeline itself passed J1-2's phase 1).
- Submission: `submit_j1_v3.sh` (reproduced in the J1-2 outcome entry): five phases under fresh root `…/j1r2` — smoke → GUARDED verify (skip on STOP with the loud message) → adequacy → capacity dev64 (A3 measured) → capacity trainfit16 (criterion 5 complete in-job). Acceptance criteria: as J1-1's entry with criterion 3 amended (verify conditional on a GO smoke selection; read-back proof discharged by J2 otherwise) and criterion 5 now dischargeable.
- Monitoring: 10-min status.json poll (reauth = ALARM); `tpu status/logs 20260806-005532-0f751b8b-exp04-j1-null-yixun`.

## J1-3 outcome (2026-08-06T05:23Z, **FAILED — RUNBOOK IDEMPOTENCY FLAW under queue auto-retry** (attempt 3 APPLICATION_ERROR; attempts 1-2 infra preemptions). PHASES 1–3 COMPLETED AND PUBLISHED on attempt 1.)

- **What happened:** attempt 1 ran ~3.5 h — smoke COMPLETE, guarded phase-2 skip worked as designed, adequacy COMPLETE (published under `…/j1r2/{smoke,adequacy}`), capacity dev64 IN PROGRESS (partial a1 shards published) — then TPU_VM_PREEMPTED. Attempt 2 preempted at 8 min. Attempt 3 re-ran the runbook from phase 1 and `write_shard` correctly REFUSED to rewrite the published smoke shard (`FileExistsError: …/j1r2/smoke/a1/shard_00000 is already published: a completed shard is never rewritten`, null_adapter_shards.py:314) ⇒ exit 1. The immutability discipline behaved exactly as reviewed; the RUNBOOK is not idempotent under the queue's restart-from-scratch retry. Classified: Planner runbook design flaw (third runbook-class lesson: phase-2 wiring, TRAINFIT coverage, now idempotency); zero code defects.
- **Standing artifacts (immutable, reusable at this SHA):** `…/j1r2/smoke` (complete), `…/j1r2/adequacy` (complete — the adoption artifact for all further capacity runs), `…/j1r2/capacity` (PARTIAL — attempt-1 shards without run-level JSON; superseded by J1-4's attempt-scoped roots, retained for the record).
- **Remediation — J1-4 (`submit_j1_v4.sh`):** capacity-only (phases 1-3 not re-run — their published artifacts stand), HARD-FAILS if the adequacy artifact is absent, and **attempt-scoped roots** (`capacity_att-<ts>`) so every queue retry writes a fresh subtree — preemption-proof with zero code changes. Authoritative attempt = the roots carrying the run-level JSONs (published last by design). Same tip `27efcd1`.

## J1-4 — capacity-only, attempt-scoped roots (2026-08-06T16:18Z, LAUNCHED — submitted by the session under Yixun's explicit in-conversation permission)

- Queue job: `20260806-161812-3aeaf86f-exp04-j1-null-yixun` (v6e-8); record confirmed PENDING (created 16:18:22Z). Tip `27efcd1` (unchanged — zero code deltas across J1-2/3/4; runbook-only evolution).
- Submission: `submit_j1_v4.sh` (reproduced in the J1-3 outcome entry): adopts the STANDING adequacy artifact (`…/j1r2/adequacy/adequacy_report.json`, verified present pre-launch; hard-fail if absent) → capacity dev64 (A3 measured) + capacity trainfit16, each under attempt-scoped roots `capacity[_trainfit]_att-<ts>` per issue #13. Authoritative attempt = roots carrying run-level JSONs.
- Monitoring: 10-min status.json poll (reauth = ALARM; credentials refreshed by Yixun 2026-08-06 ~16:15Z).

## J1-4 outcome (2026-08-06T~20:00Z, **SUCCEEDED** — attempt 0, zero preemptions; the attempt-scoped design untested by churn but the runbook completed clean)

- Authoritative roots: `…/j1r2/capacity_att-0806-164625` (dev64: selection.json, gate_tables.json, run_report.json, **a3_measurement.json**, a1/ a2/ record shards, videos/) + `…/j1r2/capacity_trainfit_att-0806-164625` (trainfit16, same set minus A3). Bare `…/j1r2/capacity/` = J1-3's superseded partial (retained for the record). Adequacy adopted from the standing `…/j1r2/adequacy` artifact as designed.
- Acceptance criteria: MET through criterion 7; criterion 8's gate outcome = **TARGET STOP on both cohorts** (the gates decide — recorded as the P1 result in the worklog).

## J1b-1 — A3 direct optimization (2026-08-06T20:15Z, LAUNCHED — Yixun's GO recorded verbatim in the query doc)

- Queue job: `20260806-201501-3edbb77d-exp04-j1b-a3-yixun` (v6e-8, worker0-only); record confirmed PENDING (created 20:15Z). Tip `db8c3dc4052f8482b5d8403a601da4c0a56fafeb` (docs-only deltas over the J1-4 code — `src/`+`bash_scripts/` unchanged since `27efcd1`'s executable tree).
- Submission: `submit_j1b.sh` (archived in scratchpad; reproduced by this entry): NULL_MODE=direct_opt, first-8 DEV, NULL_A3_ITERS=300 (the measured recipe: projection ≈ 47 min ≪ the 4 h budget; measurement verdict ok), attempt-scoped root `…/j1r2/j1b_att-<ts>` (issue #13), launcher's phase-aware A3 watchdog armed.
- Acceptance (plan §4-P1b): the direct_opt run publishes its optimization evidence (nulls / losses / grad-norms / final endpoint via the R11 write_arrays sink) + endpoint future-SSIM comparable against A2's 0.4973 — the greedy-vs-joint mechanism answer. Any outcome is acceptance. Failure triage per SOP.
- Monitoring: ScheduleWakeup polls (~25 min cadence; background shells are being killed — issue noted in-session).

## J1b-1 outcome (2026-08-06T~22:15Z, **SUCCEEDED** — attempt 2; two spot preemptions absorbed cleanly by the attempt-scoped design)

- Authoritative root: `…/j1r2/j1b_att-0806-211405/` — `a3_direct_opt.json` (per-example losses/endpoints/grad-norms, embedded re-run fit_probe, full provenance at code_sha `db8c3dc`) + `a3_nulls.npz` (the optimized joint null tensors — the J1c input if wanted). 300 iters, 3237 s wall, fits_budget confirmed in-run (projection 0.74 h).
- Result recorded in the worklog reading below; any outcome is acceptance.

## J1c-1 — transfer probe (2026-08-07T01:28Z, LAUNCHED under Yixun's "GO for J1c")

- Queue job: `20260807-012812-b06e590a-exp04-j1c-transfer-yixun` (v6e-8); record confirmed PENDING. Tip `921358521f8d7769f0723b0d2cd67f20e3f25911` (R12 committed at `5ad79fb`).
- Submission: `submit_j1c.sh` (archived; reproduced by this entry): NULL_MODE=transfer_probe, first-8 DEV, NULL_TRANSFER_NULLS_URI = J1b's `a3_nulls.npz` (sha256 recorded in the output table by design), settings global(0) own-ε₀ + keyed{0,1,2}, attempt-scoped root `…/j1r2/j1c_att-<ts>`.
- Acceptance: the probe publishes the per-example × per-noise future-SSIM/MSE table, provenance-bound to J1b. Any outcome is acceptance — this table IS the revival-vs-close decision input.

## J1c-1 outcome (2026-08-07T~02:20Z, **SUCCEEDED-BY-ARTIFACT** — attempt 0 published the complete `transfer_probe.json` before a teardown-window preemption; the queue's redundant retry is being cancelled; authoritative root `…/j1r2/j1c_att-0807-020621/`)

- Table provenance verified: npz sha256 `677502c5…eb68` = J1b's `a3_nulls.npz`; code_sha = the R12 tip; l_null 16; w 5.0. The own-basin column reproduces J1b's per-example endpoint MSEs as a multiset (loss-vs-replay deltas ≤ ~0.03) — cross-job consistency check PASSED.

## J1-5 — the CLEAN-GATE RERUN at the adopted recipe (2026-08-09T17:56Z, LAUNCHED by the session under Yixun's explicit approval "exp_04 clean_gate rerun")

- Queue job: `20260809-175610-fc3a3414-exp04-j1-5-cleangate-yixun` (v6e-8); record confirmed at the authoritative status.json. Tip `a5aa6bce119b400df31d67dabcdbf7659c1c2036` (the adequacy-wiring fix `a520e9d` + ledger; APPROVE first-pass). Submit script archived beside this file (guard = clean EXECUTABLE tree; a docs file was legitimately mid-edit by the report reviser).
- Phases: capacity dev64 + capacity trainfit16, both adopting the standing `…/j1r2/adequacy` artifact (J=50/lr=0.01 — now actually reaching the runner via the fixed launcher), attempt-scoped roots `cleangate[_trainfit]_att-<ts>`. ~1.4 h projected.
- **Stakes (per the P4 analysis review): this run DECIDES the plan-compliant target selection** — the J=10 STOP was ruled indeterminate. Acceptance: gates evaluated at the adopted recipe on both cohorts; ANY outcome is acceptance (STOP retained ⇒ the predeclared verdict stands clean; an arm selected ⇒ the P2 question reopens as a Yixun decision). Provenance headers must show J=50. Triage per SOP.


---

## 2026-08-09 — J1-5 clean-gate rerun: DEV-64 capacity cohort verdict — **STOP retained at the adopted J=50 recipe**

**Job:** `20260809-175610-fc3a3414-exp04-j1-5-cleangate-yixun` (v6e-8, attempt 1). Artifact root:
`gs://v6_east1d/datasets/droid_wan_null_adapter/j1r2/cleangate_att-0809-180640/` (capacity/DEV-64 phase published ~20:45Z; trainfit-16 phase still running at write time).

**Provenance (the point of the rerun) — verified bound in the artifacts, not just the log:** every shard `header.json` carries `"optimization_config": {"inner_iters": 50, "lr": 0.01}`, `code_sha a5aa6bc` (the adequacy-wiring-fix tip), `manifest_hash 433f8691…` matching `selection.json`. The launcher log independently shows `adopted recipe from the adequacy artifact: {'adopted': True, 'inner_iters': 50, 'lr': 0.01}`. Issue #15's fix is confirmed live end-to-end.

**Formal target selection (plan §gates, DEV-64, J=50): `target = STOP`.**

| arm | J=10 (J1-4, informal) | J=50 (J1-5, formal) | movement |
|---|---|---|---|
| A0 (per-clip control, locked basin) | 0.6665 | 0.6665 | identical (deterministic control — exact cross-run comparability) |
| A1 (optimized null, locked basin) | 0.8523 | **0.8868** (CI 0.8711–0.9010) | +0.035 — more iters buys more in-basin capacity |
| A1-probe (fresh noise) | 0.1729 | **0.1666** (rel. 0.188) | flat — transfer does NOT improve with J |
| A2 (fresh-noise-trained) | 0.4973 | **0.6638** (CI 0.6312–0.6949) | +0.166 — converges to the control, not past it |
| A2-0 (zero-null baseline) | 0.1423 | 0.1423 | identical (deterministic) |

(All numbers mean future-SSIM over the 64-clip DEV cohort, k=0; probe means over k∈{0,1,2} per the selection rule.)

**Gate clauses (measured vs bar):**
- **G1** — median MSE ratio **4.681 vs ≥ 5** (FAIL, the only clause missed); improved 100% vs ≥80% (pass); mean 0.8868 ≥ 0.80 with CI-low 0.8711 ≥ 0.75 (pass).
- **A1 selection clauses** — probe 0.1666 vs the 0.70 absolute floor (FAIL ×4.2); probe/A1 = 0.188 vs ≥ 0.7 (FAIL). ⇒ even a G1 pass could not have selected A1.
- **G2** — mean 0.6638 vs ≥ 0.75 (FAIL) and CI-low 0.6312 vs ≥ 0.70 (FAIL); its ratio clause 17.8 ≥ 5 and improved 100% pass.

**Robustness note:** the STOP does not hinge on G1's near-miss (4.681 vs 5). A1's selection was independently barred by the transfer probe missing its floor by >4x, and G2 by both SSIM clauses. Every path to a non-STOP outcome fails on at least one clause that is nowhere near its bar.

**Scientific reading:** J=50 *strengthens* exp_04's conclusion rather than merely re-confirming it. The fresh-noise arm A2 improved from "worse than doing nothing" (0.4973 at J=10) to *statistically indistinguishable from doing nothing* (0.6638 vs control 0.6665) — 5x the optimization budget moved it exactly to the control, not past it. Meanwhile locked-basin capacity rose (0.8868) and fresh-noise transfer stayed at 0.17 — the basin-specificity of greedy per-clip nulls is budget-independent.

**Formal consequence (per the analysis review's indeterminacy ruling):** the plan-compliant target selection, previously INDETERMINATE because the gates had only ever run at un-adopted J=10, is now **measured at the adopted recipe: STOP**. The predeclared verdict stands clean. Adjudication of closure is Yixun's (announcement 03) — reported to him this session.

**Pending in this run:** trainfit-16 cohort (G2' fit-check) still executing; report drop-in rows (results §1/§4) will be folded in one commit when it lands.


---

## 2026-08-10 ~02:20Z — J1-5 COMPLETE (job SUCCEEDED): reproduction exact, trainfit-16 same signature — exp_04 data collection CLOSED

**Job trail:** attempt 1 killed 21:08Z by `TPU_VM_HEALTH_TIMEOUT` (issue #16) after publishing capacity; attempt 2 re-ran the full runbook into fresh roots and SUCCEEDED. Authoritative roots (run-level JSON): `cleangate_att-0809-211639/` (capacity, `seconds: 10582`) + `cleangate_trainfit_att-0809-211639/`.

**Reproduction check (attempt 2 vs attempt 1 capacity, DEV-64):** every gate number IDENTICAL to 4+ decimals — G1 ratio 4.6812, A1 0.8868, probe 0.1666, A2 0.6638 CI [0.6312, 0.6949], target `stop`. The pipeline is deterministic end-to-end; the 2026-08-09 DEV-64 ledger entry stands unchanged as the verdict of record.

**TRAINFIT-16 (G2' fit check):** same failure signature as DEV-64 — G1 fails only `median_ratio` (4.064 vs >=5; in-basin mean SSIM 0.8847, CI [0.8596, 0.9075]); G2 fails `mean_ssim`/`ssim_ci_low` (0.6379, CI-low 0.5644); all four selection reasons identical. **The STOP is not a dev-set artifact**: the training cohort's own basin shows the same capacity-without-transfer shape (trainfit in-basin 0.8847 vs dev 0.8868 — no memorization gap either).

**Formal state:** the plan-compliant target selection is now measured, reproduced, and cohort-consistent: **STOP**. Report fold-in dispatched (results section 1/section 4 + analysis supersede notes). Closure adjudication remains Yixun's.
