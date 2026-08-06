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
