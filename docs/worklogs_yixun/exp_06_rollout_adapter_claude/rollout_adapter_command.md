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


---

## 2026-08-11 ~03:47Z — M1-2 CANCELED by Yixun (14 dead attempts); M1-3 LAUNCHED

**M1-2 final tally:** 14 attempts, zero cells measured — 3 setup preemptions, the rest killed 2–10 min into cell-1's compile (the unrolled-graph killer F4 removed). Canceled by Yixun in-conversation.

**M1-3:** `20260811-034649-b29039ef-exp06-m1c-fitprobe-yixun` (v6e-8, worker0-only), tip `5ca0fe5` — F3 (constants-as-arguments) + F4 (scan accumulation, graph 4,805 eqns flat) both aboard; submit guard verified the executable tree identical to RULED. Root: `gs://v6_east1d/datasets/droid_wan_pos_rollout/m1/att-<ts>`.

**Proof point this run owns:** the first cell RESULT line within ~15–30 min of its `[M1] entering` line proves both compile fixes end-to-end; M1-2 already proved backbone load + the entering line. Acceptance unchanged (plan v2.8 §4-P1): digest-verified authorization table over measured (arm, microbatch, k) cells, peak_source per cell, step-time table + M2/M3 projections.


---

## 2026-08-11 ~16:15Z — M1-3 attempt 2: 24/32 cells measured, then a chip-level fatal; RETRIED under the auto-resubmit rule

**The good:** the F3+F4 fixes are fully validated. Attempt 2 measured the ENTIRE rollout arm (mb 8/16/32/64 x k 2/4, two trials each — peaks 32.4/18.4/12.9/19.4 GB, k=2 steps 25.35/15.63/14.22/11.05 s; mb=32 is the efficiency sweet spot) plus one_step through mb=16 (peak 10.75 GB flat, ~3.12 s — k-invariant as expected). Trial-to-trial deltas <=0.01 s; compile minutes per cell.

**The failure:** at `one_step microbatch=32 k=2`, during the per-cell backbone reload, the TPU runtime dumped `bad_smem_address` (tc_scalar_program_errors — chip-level scalar-memory fault) and the worker exited 1 at 07:39Z. No Python traceback; the queue classified it program-failure and went terminal. **Planner ruling: infra** (hardware-fault family; the same arm's smaller cells measured clean seconds before; fatal fired in a load path exercised 11 times prior). Retried via `tpu retry` (same spec, same SHA `5ca0fe5`, zero changes) under announcement 02's auto-resubmit rule; a same-cell repeat on a fresh VM would overturn the ruling to deterministic ⇒ stop + fix round.

**Formal state:** `fit_probe/` is EMPTY — the authorization table publishes at completion, so NO cells are authorized; the 24 measurements are log evidence only. (Design note for a possible F5, not this round: publish-at-end means a 3.5 h ladder that dies at cell 12/16 authorizes nothing — per-cell incremental publication with a terminal digest would fit the issue-#13 lesson.)


---

## 2026-08-12 ~00:00Z — Yixun chose (c): hybrid — F5 `cell-publication` round OPENED while the churn continues

Zone tally for M1-3: 7 dead attempts today (health events at 30min–2h lifetimes vs the 3.5h ladder); every re-measured cell byte-identical across attempts — the waste is structural (publish-at-end). Yixun approved option (c). F5 Coder dispatched: per-cell incremental publication (digest-bound cell JSONs to the attempt root as each cell completes) + adopt-if-published on restart (digest + recipe-fingerprint + job-identity verified; refusal paths re-measure), final authorization table unchanged in semantics with measured/adopted provenance recorded. The queue keeps churning attempt 8+ in the background; whichever lands first wins — a lucky completed churn attempt moots F5 for M1 but the machinery stays for M1' (v6e-64) and any future ladder.


---

## 2026-08-12 ~04:27Z — M1-3 CANCELED (13 dead attempts, all zone infra post-proof); M1-4 LAUNCHED with F5 banking

**M1-4:** `20260812-042726-a43aeec0-exp06-m1d-fitprobe-yixun` (v6e-8), tip `6eda654` — F3 (constants-as-arguments) + F4 (scan accumulation) + F5 (per-cell banking + manifest-bound adoption) all aboard. RUN_NAME pinned `exp06-m1-fitprobe` (constant across resubmissions so banked cells adopt cross-job); `POS_FIT_ADOPTION_ROOT=$M1ROOT`. Submit guard verified executable tree == RULED.

**What changes operationally:** every completed cell now banks immediately; any restart adopts verified banked cells and measures only the remainder — the ladder accumulates across zone kills instead of restarting. Note: M1-3's log measurements (24 unique cells) are NOT adoptable (pre-F5 code, no banked artifacts; and the manifest binding would rightly refuse them) — M1-4 re-measures from zero, banking as it goes.

**Trust decision riding with Yixun (non-blocking):** accept the bucket-ACL anchor (recommended for M1/M2) vs commission artifact signing before M3; the battery carries the residual as its 1 DECLARED verdict.


---

## 2026-08-12 ~13:50Z — M1-4 attempt 1: F5 banking PROVEN in production; the one_step mb=32 chip fault is DETERMINISTIC (2/2); F6 opened; infra ruling overturned

**F5 worked:** 12/16 cells banked (`att-0812-053153/.../cells/`, full-digest names + sidecars) — the complete rollout arm (mb 8/16/32/64 × k 2/4) plus one_step mb=8/16 × k 2/4. Any future attempt adopts these in minutes.

**The killer is deterministic:** attempt 1 died at `one_step microbatch=32 k=2` with `bad_smem_address` (tc_scalar_program_errors) — the IDENTICAL cell and fault as M1-3 attempt 2, on different VMs on different days. 2/2 = the 2026-08-11 "infra" ruling is OVERTURNED (append-only correction): this is a workload-triggered XLA codegen fault. Signature: the one_step loss under F4's scan at chunk width 32 (256/32 = 8 iterations) on v6e-8; the same width works on the rollout arm, and one_step works at widths 8/16. Recorded as issue #18 material.

**Consequence:** one_step mb∈{32,64} (4 cells) have never measured on hardware and cannot be reached — the killer cell blocks the ladder tail and the table publication. Scientifically they are not needed: M2/M3 quote authorized cells only; the one_step control is cheap at mb=8/16 (3.1 s steps, 10.75 GB), and the memory-critical arm (rollout) measured completely.

**F6 `cell-exclusion` round dispatched (build now; relaunch needs Yixun):** a config-declared cell exclusion list — excluded cells recorded in the authorization table as EXCLUDED with the declared reason (never silently absent, never authorized); fail-loud if any downstream consumer quotes an excluded cell. With exclusions set to the 4 unreachable cells, M1-5 adopts the 12 banked cells and publishes the table in ~30 min.

**Process note (honest):** Planner monitoring lapsed 05:45Z→13:47Z (a turn ended without re-arming the wakeup); the failure sat unread for ~5 h. The banked cells made the lapse cost zero, but the rule stands: no turn ends without an armed wakeup while a job runs.


---

## 2026-08-12 ~15:45Z — F6 review: my "~30-min adoption" claim was WRONG (append-only correction)

The F6 reviewer reproduced both manifests and proved the point: F6 changes manifest-covered files, so M1-5's context cannot match the F5-era banked cells — adoption will correctly REFUSE all 12. That is the manifest binding working as designed; a migration rule would reopen the different-code-adoption hole F5b/F5c closed, and is rejected. **Corrected M1-5 profile: re-measures the 12 reachable cells (~2–2.5 h per attempt), banking as it goes; attempts at the SAME new SHA adopt each other's cells, so zone churn converges instead of restarting.** The 2026-08-12 ~13:50Z entry's "adopts the 12 banked cells and publishes the table in ~30 min" is retracted.


---

## 2026-08-12 ~16:02Z — M1-5 LAUNCHED (Yixun approved the 4-cell exclusion deviation + the launch in one submission)

**Job:** `20260812-160205-4a3dc8a8-exp06-m1e-fitprobe-yixun` (v6e-8), tip `a8a0b78` (F3+F4+F5+F6). Exclusion ARMED in the job env: `one_step:32:2, one_step:32:4, one_step:64:2, one_step:64:4`, reason bound (issue #18). RUN_NAME `exp06-m1-fitprobe`; adoption root `$M1ROOT`.

**Expected:** re-measures the 12 reachable cells (~2–2.5 h/attempt; F5-era banked cells correctly refuse cross-build), banking as it goes; same-SHA attempts adopt each other; publishes the v4 table with 12 authorized/refused + 4 EXCLUDED. On success → acceptance verification (peak_source per cell) → M2 package.


---

## 2026-08-12 ~19:00Z — F5 CROSS-ATTEMPT ADOPTION CONFIRMED IN PRODUCTION; M1-5 retried after a setup network flake

**The debut:** attempt 2's log: `[M1] adopting rollout microbatch=8 k=2 from .../att-20260812T161545Z/cells/... (2 trials, peak 32405364416 bytes)` — attempt 1's banked cell adopted at the same SHA, F5-era cells simultaneously REFUSED by name (different program). The mechanism works end-to-end on hardware.

**The churn:** att 1 suspended (banked m8_k2), att 2 preempted (banked m8_k4 progress unknown), att 3 preempted pre-start, att 4 died in SETUP on a transient GitHub qwix-archive fetch (http2 refused stream after 4.5s of retries) — queue classified terminal. Ruled infra; `tpu retry` issued (same spec, zero changes, auto-resubmit rule).
