# rollout_adapter — exact reproduction commands (SOP artifact 7)

## M1-1 — the fit probe (2026-08-09T17:41Z, LAUNCHED — submitted by Yixun via `!` after the READY-FOR-M1 ruling)

- Queue job: `20260809-174108-67fe9b39-exp06-m1-fitprobe-yixun` (v6e-8, worker0-only); authoritative record confirmed at `gs://v6_east1d/tpu-job-queue/jobs/20260809-174108-67fe9b39-exp06-m1-fitprobe-yixun/status.json`.
- Submitted tip: `e6cddf8ba18faf4ffda2df62a4d5cf5e7ff2f805` (docs-only descendant of the READY-ruled state `a266fe6` — the submit script verified the EXECUTABLE tree identical and the worktree clean before submitting; `COMMIT` env carries the full SHA).
- Submission: `submit_m1.sh` (archived beside this file): `POS_JOB_MODE=fit_probe POS_DEVICE_COUNT=8` (derives per-device batch 32 at GBS 256) under attempt-scoped root `gs://v6_east1d/datasets/droid_wan_pos_rollout/m1/att-<ts>` per issue #13.
- Acceptance criteria: the pre-launch package in `rollout_adapter_yixun_query.md` (authorization digest-verified; measured/authorized/refused disjoint; `peak_source` ∈ {runtime-reset, runtime-raised} — floors refuse, never authorize; cells scoped (arm, microbatch, k); epsilon data-sharded per the recorded ruling; ANY measured outcome is acceptance). First-minutes watch item: `load_backbone`'s real 5B path — the one deliberately untestable seam.
- Runbook: after the job, verify/remove leftover `<checkpoint_dir>/_m1_probe/**`; a wrong `POS_DEVICE_COUNT` fails closed in seconds inside the real process. Monitoring: ScheduleWakeup polls; reauth = ALARM. Triage per SOP: infra ⇒ auto-resubmit unchanged (standing policy); real bug ⇒ fix cycle.
