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


---

## 2026-08-13 ~16:06Z — M1-6 submitted (M1-5 dead at its 20-attempt cap; retry never took)

**M1-5 post-mortem:** 20 attempts, zero survivals, every death zone infra (preemptions/health events; the deterministic one_step-mb=32 killer never reached — excluded). The 04:49Z `tpu retry` was accepted by the CLI but never re-armed the capped job. Bank at death: 3 verified cells (m8_k2, m8_k4, m16_k2).

**M1-6:** `20260813-160603-9d6387bb-exp06-m1e-fitprobe-yixun`, identical script/RULED `a8a0b78` (auto-resubmit rule; classifier blocked the session submit — Yixun ran it). Adopts the 3 banked cells on start; 9 cells (~75 min measurement) remain. IROM reserved-slot option surfaced to Yixun as a non-blocking parallel path.


---

## 2026-08-13 ~21:55Z — M1-6 COMPLETED: the ladder finished and authorization refused ALL 12 cells on `peak_source` — the acceptance system worked; the probe has a measurement gap (F9 opened); M1-7 submission WITHDRAWN

**The run:** attempt 4 (root `att-0813-194519`) completed the full ladder — 12 measured (7 adopted from earlier attempts: F5 cross-attempt convergence fully proven), 4 excluded with the issue-#18 reason, protocol v4, digest verified (`3af075eb…`). Compile ~5-6 min/cell (F4 confirmed at scale); step times identical to every prior observation.

**The verdict: `authorized_cells: []` — every cell refused on `peak_source`** (mb=8 additionally on `headroom`: peak 32.41 GB = 96.6% of 33.55 GB capacity > the 90% floor — a genuine, correct refusal). Every measurement records `peak_source: "compiled memory analysis"`; the plan (v2.8 §4-P1) requires runtime-derived evidence (`runtime-reset` / `runtime-raised`), and the authorization floor refused exactly as reviewed ("analysis floors refuse, never authorize"). The probe's measurement path never implemented the runtime capture — the CPU fakes could not exercise TPU `memory_stats`, the same structural blind-spot family as F3's constants.

**Consequences:** (1) M1-7 at the F7 tip would reproduce the same all-refused table — the pending submission ask to Yixun is WITHDRAWN. (2) F9 opened: capture runtime peaks per cell (device `memory_stats` peak watermark; `runtime-reset` where resettable, else `runtime-raised`; fallback NEVER fakes — records analysis and lets the floor refuse). (3) One real sizing fact already stands: rollout mb=8 is over the headroom floor and will stay refused; mb∈{16,32,64} are the viable rollout cells pending runtime evidence.


---

## 2026-08-15 ~20:45Z — M1-8 SUBMITTED (Yixun): the first run that can AUTHORIZE

**Job:** `20260815-204530-3e2bd050-exp06-m1g-fitprobe-yixun` (v6e-8), tip `7f23ac3` — F3 (constants as jit
arguments) + F4 (scan accumulation) + F5 (per-cell banking) + F6 (declared exclusions) + F7 (identity =
running bytes + runtime policy) + F9 (runtime HBM peaks, declared ladder order) all aboard, plus F8's
honest battery. `submit_m1g.sh` archived; RUN_NAME pinned `exp06-m1-fitprobe`; adoption root `$M1ROOT`.

**Why this one differs from M1-6 (which authorized ZERO):** M1-6's every cell carried
`peak_source: "compiled memory analysis"` and the authorization floor refuses analysis-only evidence.
F9 makes the probe capture a real runtime watermark bracketing load → steps → eval → checkpoint, and F9b/c
run the sole above-floor cell (`rollout mb=8`) LAST so the monotone mark cannot taint the others' bounds.

**Expected table:** 10 authorized (rollout mb 16/32/64 + one_step mb 8/16, × k∈{2,4}) with runtime-sourced
peaks; `rollout mb=8` refused on genuine headroom (30.18 GiB = 96.6% > the 90% floor); 4 EXCLUDED (issue #18).
The bank starts empty at this tip (F7 changed manifest-covered files — the last such reset) and accumulates
across attempts.

**Blocked at submission:** the lab queue's central scheduler has been down since 2026-08-13T21:41Z
(issue #19) — this job is `PENDING` and will not start until it is restarted. Three jobs now wait on it.

**OUTCOME — M1-8 FAILED at the setup barrier, zero cells measured (2026-08-16T00:57Z).** Worker 0 finished
setup, then waited 1800 s for a second worker that never booted (`Setup barrier progress: 1/2 workers`;
only `worker-0.log` and `setup-ready/worker-0` exist) and exited `SETUP_ERROR` / non-retryable. Cause is
lab-side: `v6-8` is now declared **2 workers** but provisions one — issue #22; the identical failure hit
exp_02's Job 53 in the same minute, while v6-64 jobs cleared setup the same day. **The F9 runtime-peak code
was never exercised** — this says nothing about the fix. Resubmit unchanged (`submit_m1g.sh`, RULED
`7f23ac3`) once the resource config is repaired; the empty bank means nothing was lost.

## M1-9 — fit-probe resubmit after the v6-8 repair — launched 2026-08-16T16:32Z

**Submitted by the exp_02 session on Yixun's instruction** ("Currently the v6e-8 is fixed, could you
please turn on the exp06?"). Unchanged resubmit of `submit_m1g.sh` per M1-8's outcome note — no code,
config or cell-list change.

- **Job id:** `20260816-163233-09339cf7-exp06-m1g-fitprobe-yixun`; `COMMIT=768eeb9` (docs tip; the
  executable tree is **byte-identical to the RULED `7f23ac3`** — `git diff 7f23ac3 HEAD -- src bash_scripts`
  is empty, verified pre-submission), worktree clean at submission.
- **Issue #22 verified repaired before submitting:** `tpu admin resources` now lists
  **`v6-8  v6e-8  8 chips  1 WORKERS`** (was 2), and the uploaded `spec.json` for this job records
  `workers: 1` — the setup barrier that deadlocked M1-8 cannot recur for it.
- **Unchanged from M1-8:** `RUN_NAME=exp06-m1-fitprobe`, `M1ROOT=gs://v6_east1d/datasets/droid_wan_pos_rollout/m1`,
  attempt-scoped root `$M1ROOT/att-<ts>`, `POS_FIT_EXCLUDED_CELLS=one_step:32:{2,4},one_step:64:{2,4}`
  (issue #18). Bank starts empty, as M1-8 measured zero cells.
- **Acceptance (from the M1-8 package):** 10 authorized cells with runtime-sourced peaks
  (rollout mb 16/32/64 + one_step mb 8/16, × k∈{2,4}); `rollout mb=8` expected to refuse on genuine
  headroom; 4 EXCLUDED. **The F9 runtime-peak code has never been exercised on hardware** — M1-8 died
  before reaching it, so this run is its first real test.

**Duplicate-submission note (2026-08-16T16:35Z).** Two identical fit-probe jobs briefly existed: the exp_02
session submitted M1-9 (`20260816-163233-09339cf7`, COMMIT `768eeb9`) at 16:32:33Z on Yixun's instruction,
and this session submitted `20260816-163416-c29777f8` (COMMIT `8aa86cb`) 103 s later, having independently
verified the v6-8 repair on hardware. The two specs are equivalent — same RUN_NAME `exp06-m1-fitprobe`,
same adoption root, same exclusion list; the COMMIT labels differ only by docs commits, and under F7 the
binding is the running bytes, so both would adopt each other's banked cells. **This session cancelled its
own later duplicate**; M1-9 is the canonical run. Cross-session lesson: with one experiment lane reachable
from two sessions, check `gsutil ls gs://v6_east1d/tpu-job-queue/jobs/ | grep <name>` before submitting.

### M1-9 outcome (2026-08-16T20:23Z) — job SUCCEEDED, ladder fully measured, **AUTHORIZATION EMPTY (all cells REFUSED on `peak_source`)**

Authoritative root: `gs://v6_east1d/datasets/droid_wan_pos_rollout/m1/att-0816-172718` —
`fit_authorization.json` sha256 `79253ea5…3fe826`, protocol `exp06.fit_authorization.v6`, exit 0.
One spot preemption (17:23Z) absorbed by the F5 cell bank: attempt 2 adopted `one_step m8 k2/k4`
from attempt 1's root (`att-0816-164423`) and reproduced `one_step m16 k2` bit-identically
(10,754,255,744 B; 3.109 s vs 3.123 s). Aug-12/13 banked cells correctly refused (`fit_cell.v1` ≠ `v2`).
10 cells measured this attempt + 2 adopted + 4 EXCLUDED (issue #18), 0 skipped.

**Measured ladder** (peak = `analysis_bytes`; capacity 33,550,233,600 B; 2 trials each, exact agreement):

| arm | mb | k | peak (B / GiB / %cap) | step s | note |
|---|---|---|---|---|---|
| one_step | 8 | 2/4 | 15,991,929,696 / 14.89 / 47.7% | 4.88 / 4.90–5.05 | |
| one_step | 16 | 2/4 | 10,754,255,744 / 10.02 / 32.1% | 3.11 / 3.12 | |
| rollout | 32 | 2/4 | 12,932,769,696 · 12,941,453,376 / 12.04–12.05 / 38.6% | 14.22 / 28.33 | k=4 ≈ 2× time, ≈ flat memory |
| rollout | 16 | 2/4 | 18,417,237,024 · 18,421,529,760 / 17.15–17.16 / 54.9% | 15.64 / 31.13 | |
| rollout | 64 | 2/4 | 19,389,143,680 · 19,406,838,368 / 18.06–18.07 / 57.8% | 11.09 / 22.10 | |
| rollout | 8 | 2/4 | 32,405,364,416 · 32,408,018,592 / 30.18 / **96.6%** | 25.35 / 50.47 | REFUSED also on `headroom` (>90% floor) — as predicted |
| one_step | ≥32 | 2/4 | — | — | 4 × EXCLUDED (issue #18) |

**Why zero authorizations:** every measurement carries `peak_source: "compiled memory analysis"`,
`peak_attribution: "none"`, and a runtime watermark (4.23→4.87 GiB across the ladder) that plainly does
not see the step's scratch memory — the allocation-watermark instrument cannot observe XLA's
preallocated arena on TPU, so no runtime-sourced peak ever materialized and the record fell back to the
compile-time analysis. The authorization gate then refused all 12 measured cells on `peak_source`
(the m8 rollout pair additionally on genuine `headroom`), exactly per the fail-closed contract
("analysis floors refuse, never authorize" — the M1-6 lesson). **The floor worked; the F9 runtime-peak
instrument did not produce runtime evidence on hardware.** M2 cannot be authorized from this run.

**Triage: real instrumentation gap, not infra.** Next step belongs to the owning session: an F10-class
fix that sources a true runtime peak (per-device memory stats read inside the step bracket, not an
allocation watermark), then a resubmit — the bank will adopt all 12 measured cells' timings, so only
re-measurement of peaks under the new instrument is at stake. Recorded by the exp_02 session, which
monitored this job.

### M1-9 adjudication escalated (owning session, 2026-08-16T~21:40Z) — PLAN-AMENDMENT DECISION → Yixun

The exp_02 session's outcome entry above is adopted as the measured record. Owning-session diagnosis
goes one step further than "instrument gap": the authorization design is **unsatisfiable in the
common case** — three individually-reviewed rules (peak = max(watermark, analysis); peak_source
names the origin of the max; the floor authorizes only runtime-sourced peaks) jointly imply
authorization can occur ONLY when the compile analysis UNDERestimates true usage. On hardware the
PJRT watermark (4.2–4.9 GiB) never sees XLA's temp arena while analysis (10–30 GiB) is an upper
bound, so analysis wins the max everywhere and every cell refuses by construction. CPU fakes were
built with watermark > analysis, so no test could see it; coder + 3 Codex passes missed the joint
implication. Evidence consistent with analysis-as-upper-bound: rollout m8 (analysis 30.18 GiB,
96.6% cap) ran with reservation_failures=0 at watermark 4.87 GiB.

Because plan v2.8 §4-P1 predeclared *runtime* evidence as the authorization requirement, changing
that requirement is a plan amendment — **Yixun's call (announcement 03), not a fix round**. Options
delivered: **(A, recommended)** authorize on compiled-memory-analysis as conservative upper bound
(headroom errs safe; watermark recorded + cross-check watermark ≤ analysis, violation = refuse);
under A the banked table authorizes 10/16 cells (both arms at mb=16 included — M2 unblocked) and
refuses rollout mb=8 on genuine headroom (96.6% > 90%); F10 = classify/floor change + tests + one
Codex pass + M1-10 resubmit adopting all 12 banked cells (~cheap, re-derivation only).
**(B)** investigate libtpu-level metrics that see the temp arena (unknown cost/feasibility on
jax 0.10.2). **(C)** A now + B as background hardening. Awaiting Yixun.

### F10 SERIES COMPLETE + APPROVED (2026-08-17T~02:20Z) — the plan-v2.9 authorization amendment is implemented; M1-10 package to Yixun

Six implementation rounds (F10 `4e264dc` → F10b `1e5dda9` → F10c `39f164e` → F10d `9830264` →
F10e `623107a` → F10f `f872a42`) against six review passes (initial REWORK + four verification
REWORKs + final **APPROVE, findings NONE**), all by codex gpt-5.6-sol xhigh; full trail with
appended verdicts in `rollout_adapter_codex_code_f10-authorization-amendment_review.md`.

**What the amendment now is, as built:** authorization bound = compiled memory analysis
(≤ 90% capacity in EXACT integer arithmetic, 10·a ≤ 9·c); runtime watermark recorded and
cross-checked (watermark > analysis, or missing watermark, or missing/partial/unparseable
analysis ⇒ refuse); per-trial refusal survival + unanimous analysis across trials (disagreement
raises at publication; a poisoned banked artifact is quarantined at adoption and re-measured —
proven self-healing); exact-count parsing of every payload/identity/binding/evidence number in
both directions, with a 20-site AST-verified survivor enumeration written into the module;
protocol v6→v7 (fit_cell.v2 unchanged). Suite 2308 → **2345 passed / 0 failed**; battery 91 →
**106 probes (105 REFUSED / 1 DECLARED F5-8 / 0 SUCCEEDED), 18/18 honest controls, exit 0**.
Four review-found production defects beyond the amendment itself were fixed on the way
(missing-watermark bypass, trial-max masking, coercion truncations, partial-analysis
under-bound) — the reviewer executed a live counterexample for every one.

**M1-10:** `submit_m1h.sh` ready at RULED=`f872a42`; full re-measure (~2.5–3.5 h v6e-8) because
the F10 edits moved the deployed manifest and the M1-9 bank is correctly non-adoptable
(adoption discipline held exactly as designed). Expected from M1-9's numbers re-derived under
the amended rule: **10/12 authorized** (one_step mb=8/16, rollout mb=16/32/64, both k),
rollout mb=8 refused on true headroom (96.6%), 4 cells declared-excluded (issue #18).
Awaiting Yixun's launch approval per announcement 02.

### M1-10 SUBMITTED (2026-08-17T01:37Z, by Yixun via `!` after package approval)

**Job id:** `20260817-013752-c700b0fd-exp06-m1h-fitprobe-yixun` (v6e-8, submit_m1h.sh at RULED=f872a42,
guards passed). Full re-measure, ~2.5–3.5 h + queue weather. Expected: 10/12 authorized, rollout mb=8
headroom-refused, 4 declared-excluded. On SUCCEEDED: read the newest att root's fit_authorization.json
(protocol v7), verify against expectation, then M2 proposal.

### M1-10 SUCCEEDED — M1 PHASE COMPLETE (2026-08-17T~05:20Z)

Single attempt, ~3.3 h, exit 0. Authoritative root `gs://v6_east1d/datasets/droid_wan_pos_rollout/m1/att-0817-015756`;
`fit_authorization.json` sha256 `ed6262a1…539a25`, protocol **exp06.fit_authorization.v7**. Table verified
EXACTLY as predicted from M1-9's numbers re-derived under plan v2.9: **10 authorized** (one_step mb=8/16,
rollout mb=16/32/64, both k), **2 refused** (rollout mb=8 k=2/4, reasons `['headroom']` only — no
peak_source refusals remain), **4 declared-excluded** (#18), **0 watermark cross-check firings**, all 12
measurements analysis-sourced with watermark fields recorded. Every re-measured value byte-identical to
M1-9 (e.g. rollout m16 18,417,237,024 B; one_step m16 step 3.113 s). Old banks refused adoption for the
right reasons in the log (manifest_digest differ; fit_cell v1≠v2). The F10 amendment did on hardware
precisely what it was ruled to do. **M1 (v6e-8) is COMPLETE; M2 is unblocked pending Yixun's approval;
M1′ (v6e-64 topology re-run) remains required before M3 per plan v2.7.**

Step-time basis for M2 costing (GBS-256 cells): rollout mb=16 k=2 = 15.634 s/step; one_step mb=16 = 3.113 s/step.

### M2 PAIR SUBMITTED (2026-08-18T02:51Z, Yixun) — after a flax-drift replace

First submission (024447/024457) was deleted PRE-RUN: flax 0.12.8 released unpinned between Aug 17
and Aug 18 and fresh venvs crash at import (`jax.experimental.hijax.MutableHiType` missing — proven
by exp_03 Job 26b attempt 8). Resubmitted with `flax==0.12.6` pinned in the setup-cmd (matches the
venv M1-10 actually measured under; no reviewed-code change; issue #27).
- **M2 R-B:** `20260818-025132-5caec9a8-exp06-m2-rb-yixun` (rollout, mb=16, k=2, 2k steps, ~8.7 h)
- **M2 C0:** `20260818-025138-f26ddecb-exp06-m2-c0-yixun` (one_step, mb=16, k=2, 2k steps, ~1.7 h)
Both v6e-8, GBS 256, seed 0, eval/ckpt every 1,000, POS_FIT_AUTHORIZATION = att-0817-015756 table.
**Recipe deviation (Yixun-acknowledged by submission):** plan §4-P2's "32 examples" was never wired
and pos_logical_batch/train_data_dir are authorization-bound ⇒ M2 runs the AUTHORIZED recipe
(GBS 256, full train stream). Continuation rule unchanged. Pair rule: either arm preempted ⇒ both restart.
Watch: pin visible in setup logs; assert_cell_authorized passes (first production test of the F7b
binding gate); finite losses; step times ≈ M1-10's cell numbers.

### Issue #27 in full: three env drifts, three failed batches, quartet pin (2026-08-18T13:24Z)

Batches 2 (02:51Z) and 3 (04:47Z) of the M2 pair both died pre-training on supply-chain drift, each
with a DIFFERENT root cause, each proven from final-install log lines:
- Aug-17 working stack (M1-10, 26b's 12 h): jax 0.11.0 / jaxlib 0.11.0 / libtpu 0.0.41 / flax 0.12.8.
- Overnight jax 0.11.1 broke flax 0.12.8 (`hijax.MutableHiType` removed) → 26b att-8.
- Planner pin #1 (flax==0.12.6 only) inverted the mismatch (`jax.core.Effect`) → batch-2 trio died.
- Planner pin #2 (jax[tpu]==0.11.0 + flax==0.12.8) restored jax but the `[tpu]` extra re-resolved
  libtpu to the just-released **0.0.44.1**, which REFUSES our `LIBTPU_INIT_ARGS` AllGather
  continuation-fusion flag on the v6e ("ghostlite") platform at backend init → batch-3 trio died.
**Batch 4 (this one): full quartet pin** `pip install jax==0.11.0 jaxlib==0.11.0 libtpu==0.0.41
flax==0.12.8` (no extras, nothing left to a resolver) — byte-matches the Aug-17 stack.
- M2 R-B: `20260818-132349-d81db93f-exp06-m2-rb-yixun`; M2 C0: `20260818-132417-bfdffa77-exp06-m2-c0-yixun`.
Lesson (standing): TPU-stack pins must close over the QUARTET jax/jaxlib/libtpu/flax; an extra or
`-f` index re-resolution is a hole. Long-term: lock the quartet in setup.sh (unified-launcher lane).

### RETRACTION (2026-08-18T19:05Z): the R-B "hang" was a Planner monitoring artifact

For 5 hours the Planner read attempt-1/worker-0.log (frozen at 13:41 — that attempt died ~13:41 and
the queue silently started attempt 2 at 13:54) and inferred a hang from its silence, escalating to a
hang-determination window. REALITY: attempt 2 trained on schedule the whole time and saved its
step-1000 checkpoint at 18:27:25Z (att-20260818T135027Z), within minutes of the 15.6 s/step
prediction. No hang existed; no action was taken on the false alarm (the predeclared
evidence-before-action window did its job). **Standing monitoring rule (issue #14 variant): a log
that stops moving while status.json says RUNNING means you may be reading a DEAD attempt — list
logs/attempt-*/ and match against current_qr_name BEFORE inferring anything from silence.**
C0 meanwhile burns attempts (9 by 19:00Z, all infra-preempted before completing; pair integrity is
structurally safe — train mode adopts only COMPLETE publications, so every attempt restarts at 0 and
no mid-run resume can taint the pair). R-B ETA 2k ~22:40Z if att-2 survives.
