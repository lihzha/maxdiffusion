# rollout_adapter — exact reproduction commands (SOP artifact 7)

## M1-1 — the fit probe (2026-08-09T17:41Z, LAUNCHED — submitted by Yixun via `!` after the READY-FOR-M1 ruling)

- Queue job: `20260809-174108-67fe9b39-exp06-m1-fitprobe-yixun` (v6e-8, worker0-only); authoritative record confirmed at `gs://v6_east1d/tpu-job-queue/jobs/20260809-174108-67fe9b39-exp06-m1-fitprobe-yixun/status.json`.
- Submitted tip: `e6cddf8ba18faf4ffda2df62a4d5cf5e7ff2f805` (docs-only descendant of the READY-ruled state `a266fe6` — the submit script verified the EXECUTABLE tree identical and the worktree clean before submitting; `COMMIT` env carries the full SHA).
- Submission: `submit_m1.sh` (archived beside this file): `POS_JOB_MODE=fit_probe POS_DEVICE_COUNT=8` (derives per-device batch 32 at GBS 256) under attempt-scoped root `gs://v6_east1d/datasets/droid_wan_pos_rollout/m1/att-<ts>` per issue #13.
- Acceptance criteria: the pre-launch package in `rollout_adapter_yixun_query.md` (authorization digest-verified; measured/authorized/refused disjoint; `peak_source` ∈ {runtime-reset, runtime-raised} — floors refuse, never authorize; cells scoped (arm, microbatch, k); epsilon data-sharded per the recorded ruling; ANY measured outcome is acceptance). First-minutes watch item: `load_backbone`'s real 5B path — the one deliberately untestable seam.
- Runbook: after the job, verify/remove leftover `<checkpoint_dir>/_m1_probe/**`; a wrong `POS_DEVICE_COUNT` fails closed in seconds inside the real process. Monitoring: ScheduleWakeup polls; reauth = ALARM. Triage per SOP: infra ⇒ auto-resubmit unchanged (standing policy); real bug ⇒ fix cycle.


---

## 2026-08-10 ~15:25Z — M1-1 CANCELLED by Yixun's order (12 dead attempts)

Yixun ordered the cancel in-conversation ("cancel the old M1 job"); executed via `tpu cancel 20260809-174108-67fe9b39-exp06-m1-fitprobe-yixun` (scheduler confirmed deletion of the queued resource + VM). Final tally: 12 attempts, all infrastructure-classified deaths (1 pre-start preemption + 11 health timeouts/unhealthy at ~2h each), zero cells measured, no artifacts published. Root cause was OURS, not the zone's: the frozen 5B was baked into the XLA program as 10.18GB of literal constants (pos_rollout_update closure design), so the first compile never finished inside the queue health window — diagnosed 2026-08-10, fixed as round F3 (frozen state as jit argument), hardened as F3b per the Codex review (evaluator seam + freeze semantics + guard blind spots). M1-2 relaunch package goes to Yixun at the post-F3b SHA. Issue #16's "zone flakiness" reading is RETRACTED for this job's kills (J1-5's single kill remains plausibly zone).


---

## 2026-08-10 ~17:46Z — M1-2 LAUNCHED (Yixun-approved via `!` submission)

**Job:** `20260810-174626-d126336b-exp06-m1b-fitprobe-yixun` (v6e-8, worker0-only). Tip `7b3f10c` (ceremony commit `88dd491` + docs) — the F3–F3d captured-constants arc; guard verified the executable tree identical to RULED before submit. Root: `gs://v6_east1d/datasets/droid_wan_pos_rollout/m1/att-<ts>` (attempt-scoped, issue #13).

**What it must prove beyond M1-1's mandate:** the compile now completes inside the queue health window. First-minutes watch item: the `[M1] entering <cell>` line (printed unbuffered BEFORE each compile). If no cell line within ~30–40 min of jax init and the process dies at ~2 h again — the 32 Python-unrolled microbatch grad blocks (the reviewer's flagged residual) are the next suspect, and the remedy is a design change (scan over microbatches), not a retry.

**Acceptance unchanged** (plan v2.8 §4-P1): digest-verified authorization table over the measured (arm, microbatch, k) cells; peak_source ∈ {runtime-reset, runtime-raised} per cell; step-time table + M2/M3 projections; M2 quotes only cells this probe authorizes.


---

## 2026-08-10 ~20:40Z — M1-2 diagnosis: cell-1 compile kills the VM — 4/4 same-phase deaths; F4 design round opened

**Evidence:** attempts 3, 4, 6, 8 each passed backbone load (F3 threading PROVEN on real hardware — M1-1 never got here), printed `[M1] entering rollout microbatch=8 k=2: building and compiling`, then died 2–10 min later, all `TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE`, on four different VMs (attempts 1/2/5 were setup-phase preemptions; 7 SUSPENDED). Same-phase death across VMs = workload signature (the issue-#16 lesson applied in real time). The 10.18 GB constants are gone (F3 worked); the killer is now the **graph itself** — cell 1 unrolls 32 microbatch gradient blocks of the 5B forward+backward in Python, and XLA's compile of that graph exhausts the host. This was the F3c reviewer's flagged residual risk, now measured.

**Action:** recommended Yixun cancel `20260810-174626-d126336b` (12 attempts remain on the cap; each burns ~20 min). **F4 design round dispatched:** replace the Python-unrolled microbatch accumulation with `lax.scan` over microbatch chunks — one gradient block compiled once, graph size O(1) in microbatch count instead of O(32). Parity of scanned vs unrolled gradients pinned by test; a graph-size guard asserts eqn count stays flat as microbatches grow. Review + battery + fresh package to follow; no relaunch without Yixun's approval.
