# null_adapter_results — exp_04 P4 (SOP artifact 10)

**Status: DRAFT for review.** Written by the Coder; the Codex analysis review is deliberately deferred, so every
judgment call is flagged inline with **[JUDGMENT]** and every deviation from plan v5 with **[DEVIATION]**.

This document is the **factual record only**: what was predeclared, what ran (including every failed attempt and
its classification), the numbers exactly as the gate module computed them, the gate verdicts, and the artifact
URIs with their recorded digests. Interpretation lives in `null_adapter_analysis.md`.

**Verification basis.** Every number below was re-derived from the published artifacts pulled from GCS on
2026-08-08 (cached locally for the write-up), not from the worklog readings. Where a worklog figure and the
artifact disagree, the artifact wins and the discrepancy is recorded in §8. Gate metric = **`future_ssim` /
`future_mse`** — the non-pinned latent frames (latent frames 1–8 ⇒ pixel frames 1–32); frame 0 is the pinned
first-frame condition and is excluded from every primary statistic.

---

## 1. Scope and terminal status

| Item | Value |
|---|---|
| Experiment | exp_04 `null_adapter` — null-text inversion (Mokady et al. 2022) ported to JAX against the frozen Wan2.2 TI2V 5B |
| Plan of record | `plan_null_adapter.md` **v5** (5 Codex review passes, all findings accepted, APPROVE-PLAN 2026-08-04) |
| Phases executed | J0 (cohorts) → **P1** (J1 capacity + basin probe) → **P1b** (J1b joint direct optimization) → **P1c** (J1c transfer probe) |
| Phases NOT executed | **P2** (target caching), **P3** (null-embedding adapter), J2–J5 — stopped by the predeclared target-selection rule |
| Terminal verdict | **TARGET = STOP on both cohorts.** Every predeclared gate fired; no arm qualified for caching |
| Branch / tip | `claude-exp_04_null_adapter-20260803` @ `0b780df` (clean) |
| Code state | R1–R11 + the `hyperparameters-config-access` fix + R12-lite `transfer-probe`; suite **989** tests |

---

## 2. What was predeclared (plan v5 §4-P1, verbatim thresholds)

### 2.1 Arms

| Arm | Definition | Role |
|---|---|---|
| **A0** | frozen base ∅ (T5("")), replay from `traj[0]` (the inversion endpoint) | control for A1. **CFG collapses to identity** here — base-row nulls make both CFG branches identical |
| **A1** | ∅ optimized per step from `traj[0]`, replayed from `traj[0]` | own-basin capacity |
| **A1-probe** | A1's optimized ∅ locked, replayed from `keyed(k)`, k∈{0,1,2} | transfer |
| **A2-0** | frozen base ∅, replay from ε₀ = `global(0)` | control for A2 |
| **A2** | ∅ optimized per step from ε₀, replayed from ε₀ | fresh-noise capacity |
| **A2-probe** | A2's ∅ locked, replayed from `keyed(k)`, k∈{0,1,2} | diagnostic |
| **A3** (P1b) | ∅ optimized **jointly through the differentiable 25-step rollout** against the endpoint, from ε₀ | conditional, separately approved |

### 2.2 Gates (`null_adapter_gates.py`; paired unit = example; 10,000 bootstrap resamples, seed 20260804, percentile CIs)

- **G1** (A1 vs A0, DEV-64): median ratio `future_MSE(A0)/future_MSE(A1)` **≥ 5** AND **≥ 80 %** improved AND mean `future_SSIM(A1)` **≥ 0.80** with 95 % CI-low **≥ 0.75**.
- **G2** (A2 vs A2-0, from ε₀): median ratio **≥ 5** AND **≥ 80 %** improved AND mean `future_SSIM(A2)` **≥ 0.75** with 95 % CI-low **≥ 0.70**.
- **Target-selection rule:** select A1 iff **G1 passes AND** A1-probe ≥ 0.7 × mean SSIM(A1) **and** A1-probe ≥ **0.70 absolute**. Else select A2 iff **G2 passes**. Else **stop after P1**.
- Coverage is asserted (every manifest name present exactly once, else the gate FAILS); invalid pairs impute in the claim-penalizing direction; the gate auto-FAILS above 10 % invalid pairs.

> **Standing note carried from the R5 review (recorded in the worklog 2026-08-05T06:20Z, binding on this
> document):** on SSIM ∈ [0,1] the absolute floor (≥ 0.70) **strictly implies** the relative transfer test
> (≥ 0.7 × A1, since 0.7 × A1 ≤ 0.7). No result statement may attribute a probe rejection to the relative
> conjunct alone — **the absolute floor is always the binding constraint.** Honored throughout §4.

### 2.3 Adequacy probe and adoption (predeclared, plan v5 §4-P1)

8 DEV examples × J ∈ {10, 25, 50} × lr ∈ {1e-2, 3e-2}. Adoption statistic `s_e` = mean over the 25 steps of the
final post-inner-loop tracking loss; recipe score = median of `s_e` over the 8. **Adoption rule:** among recipes
scoring ≤ 0.5 × score(J=10, lr=1e-2), adopt the lowest; ties → lower J, then lower lr; else keep the default.
**If a recipe is adopted, re-run A1/A2 on the full DEV cohort under it BEFORE gating; G1/G2 are evaluated once,
on the adopted recipe only.** Plateau rule: < 10 % improvement of the adopted lr's score from J=25→50 ⇒
"reconstruction-limited", else "recipe-limited".

---

## 3. Job trail — every attempt, including failures

No failure is omitted or reclassified. Classifications are as recorded at the time in `null_adapter_command.md`
and the worklog.

| Job | Date (UTC) | Tip | Outcome | Classification |
|---|---|---|---|---|
| **J0-1** | 2026-08-05T04:50Z | `7199feb` | **FAILED** — TensorFlow/ADC reauth `invalid_rapt` + anonymous-caller 401 at the first listing; aborted before any scan, **nothing written** | **Infrastructure** (issue #6, new sub-variant: gsutil credentials live but ADC stale — two distinct logins) |
| **J0-2** | 2026-08-05T15:27Z | `7199feb` | **SUCCEEDED** — VAL exactly **14,636** records / 8 shards; TRAIN **5,000** distinct episodes in 40 shards (caps 200 shards / 60 GiB untouched); staged publication complete; `load_manifests` re-validation passed | — |
| **J1-1** | 2026-08-05T18:17Z | `3616a94` | **FAILED** — `getattr(config, "code_sha", "")` at `run_wan_null_inversion.py:611`; `HyperParameters.__getattr__` raises `ValueError`, so 3-arg `getattr` never falls back (`pyconfig.py:316-319`). Crashed ~8 min in, **after a successful full pipeline load** (model shards + revision `b8fff7315c…` resolved — the R10 backend fix held) | **REAL BUG** → fix round `hyperparameters-config-access`, commit `925ee17`, 10-site audit, AST regression pins, APPROVE first-pass with zero findings. Now issue #11 |
| **J1-2** | 2026-08-05T19:43Z | `27efcd1` | **FAILED** at attempt 4 (attempts 1–3 = infra preemptions, queue auto-retried). **Phase 1 (smoke, n=2) COMPLETED on TPU** — inversion, per-step optimization, replay, decode, SSIM, gates, provenance-bound publication all ran at production settings. The n=2 gates then correctly returned STOP (CI gates on 2 examples fail near-tautologically — *not* a scientific result), and phase 2 (`verify_replay`) correctly refused: `selected_arm` → "no selected arm to cache or verify" ⇒ exit 1 | **Planner runbook design flaw** (zero code defects — gates and verify each behaved exactly as reviewed; the *runbook* wired smoke→verify through a selection that cannot name an arm at smoke scale) |
| **J1-3** | 2026-08-06T00:55Z | `27efcd1` | **FAILED** at attempt 3 (attempts 1–2 = infra preemptions). Attempt 1 ran ~3.5 h: smoke COMPLETE, guarded phase-2 skip worked, **adequacy COMPLETE**, capacity dev64 in progress → `TPU_VM_PREEMPTED`. Attempt 3 re-ran from phase 1 and `write_shard` correctly refused to rewrite the published smoke shard (`FileExistsError: … is already published: a completed shard is never rewritten`, `null_adapter_shards.py:314`) ⇒ exit 1 | **Planner runbook design flaw** — the runbook is not idempotent under the queue's restart-from-scratch retry. The immutability discipline behaved exactly as reviewed. Now **issue #13**; remedy = **attempt-scoped roots** |
| **J1-4** | 2026-08-06T16:18Z | `3bdbd2a` (see §8.1) | **SUCCEEDED** — attempt 0, zero preemptions. Capacity-only (phases 1–3 adopted from the standing artifacts), attempt-scoped roots | — |
| **J1b-1** | 2026-08-06T20:15Z | `db8c3dc` | **SUCCEEDED** — attempt 2; two spot preemptions absorbed cleanly by the attempt-scoped design. 300 iters, 3,237 s wall | — |
| **J1c-1** | 2026-08-07T01:28Z | `9213585` | **SUCCEEDED-BY-ARTIFACT** — attempt 0 published the complete `transfer_probe.json` before a teardown-window preemption; the queue's redundant retry was cancelled | — |

**Three runbook-class lessons in one campaign** (phase-2 wiring, missing TRAINFIT cohort half, idempotency under
auto-retry) — all Planner-authored orchestration errors, **zero code defects across J1-2/3/4** (identical
executable tree throughout).

**Cohort-coverage gap and its remediation.** Acceptance criterion 5 said "all six arms on DEV-64 **+ TRAINFIT-16**",
but cohorts are one-per-invocation (`plan_run` reads `config.null_cohort`) and the J1-1/J1-2 runbook contained no
TRAINFIT invocation. Found 2026-08-05T20:40Z via exp_05's S10a review chain, reviewer-ratified; folded into J1-3
as a fifth phase under a distinct artifact root, and delivered by J1-4.

---

## 4. J1-4 — P1 capacity study and basin probe (the primary result)

Authoritative roots (both published 2026-08-06):
`gs://v6_east1d/datasets/droid_wan_null_adapter/j1r2/capacity_att-0806-164625` (DEV-64) and
`…/capacity_trainfit_att-0806-164625` (TRAINFIT-16). Full coverage, **zero quarantines, zero invalid pairs,
`coverage_ok: true`** on every gate in both cohorts.

### 4.1 Per-arm means (gate metric `future_ssim`; secondary columns re-derived from `gate_tables.json`)

**DEV-64** (n = 64; probe arms average over k ∈ {0,1,2} ⇒ 192 observations)

| Arm | mean future-SSIM | mean full-SSIM | mean future-MSE | mean future-pixel-MSE |
|---|---|---|---|---|
| A0 (control) | **0.6665** | 0.6766 | 0.3354 | 0.02085 |
| **A1** | **0.8523** | 0.8567 | 0.1127 | 0.00648 |
| A1-probe | **0.1729** | 0.1980 | 4.3589 | 0.18647 |
| A2-0 (control) | **0.1423** | 0.1683 | 4.7431 | 0.21958 |
| **A2** | **0.4973** | 0.5126 | 0.4901 | 0.03321 |
| A2-probe | 0.2958 | 0.3172 | 3.2459 | 0.10646 |

**TRAINFIT-16** (n = 16)

| Arm | mean future-SSIM | mean full-SSIM | mean future-MSE |
|---|---|---|---|
| A0 | 0.7335 | 0.7416 | 0.2391 |
| **A1** | **0.8722** | 0.8761 | 0.0778 |
| A1-probe | **0.1738** | 0.1988 | 4.2236 |
| A2-0 | 0.1355 | 0.1617 | 4.6438 |
| **A2** | **0.4563** | 0.4728 | 0.4850 |
| A2-probe | 0.2540 | 0.2766 | 3.5268 |

### 4.2 Gate verdicts as computed (`selection.json`)

| Gate | Cohort | median ratio (bar ≥ 5) | frac improved (≥ 0.80) | mean SSIM | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| **G1** (A1 vs A0) | DEV-64 | **3.6052** ✗ | 0.9531 ✓ | 0.8523 ✓ (≥0.80) | **[0.8327, 0.8710]** ✓ (low ≥0.75) | **FAIL** — `median_ratio` only |
| **G1** | TRAINFIT-16 | **3.0134** ✗ | 1.0000 ✓ | 0.8722 ✓ | [0.8549, 0.8891] ✓ | **FAIL** — `median_ratio` only |
| **G2** (A2 vs A2-0) | DEV-64 | 10.2156 ✓ | 1.0000 ✓ | **0.4973** ✗ (<0.75) | **[0.4697, 0.5264]** ✗ (low <0.70) | **FAIL** — `mean_ssim`, `ssim_ci_low` |
| **G2** | TRAINFIT-16 | 9.5375 ✓ | 1.0000 ✓ | **0.4563** ✗ | [0.4165, 0.4973] ✗ | **FAIL** — `mean_ssim`, `ssim_ci_low` |

**Transfer (target-selection conjuncts):**

| Cohort | A1-probe abs (floor 0.70) | A1-probe relative (0.7 × A1) | verdict |
|---|---|---|---|
| DEV-64 | **0.1729** ✗ | 0.2029 ✗ | fails; **the absolute floor is the binding constraint** |
| TRAINFIT-16 | **0.1738** ✗ | 0.1993 ✗ | same |

**Selection: `target = "stop"` on both cohorts**, with the reasons recorded verbatim in `selection.json`:

```
"G1 failed (median_ratio)"
"A1-probe below 0.7x the A1 mean (transfer)"
"A1-probe below the 0.70 absolute floor"
"G2 failed (mean_ssim, ssim_ci_low)"
```

**One-line gate readings (no interpretation beyond the gate):**
- G1 failed on the MSE-ratio conjunct alone; A1's absolute SSIM and its CI both cleared their bars.
- G2 failed on both absolute-SSIM conjuncts; its ratio and improvement conjuncts both cleared.
- A1-probe sits at **0.1729** against a 0.70 floor — a factor of four short.
- **[DEVIATION — RECORD]** The nearest measured "do nothing under a non-own basin" reference is A2-0 (base nulls from ε₀) at **0.1423**, so A1-probe is **+0.031 ABOVE it**, not below. The worklog's J1 reading states "locked wrong-basin nulls are actively destructive, WORSE than doing nothing" — **that is not supported by these artifacts** and appears to have been transposed from exp_05's B2 result, where the same phrase *is* gate-verified (B2 0.1610 vs B2-0 0.2814, 0/64 improved). The supported exp_04 statement is: **locked wrong-basin nulls buy essentially nothing over doing nothing (+0.03 SSIM) while costing a full optimization pass.** Caveat: the comparison is approximate — A1-probe replays from `keyed(k)`, A2-0 from `global(0)`; a "base nulls under `keyed(k)`" arm was never run.
- The selection verdict is **over-determined**: it holds on the probe floors alone even if G1 had passed.

### 4.3 Adequacy probe (`…/j1r2/adequacy/adequacy_report.json`, published by J1-3 attempt 1)

Grid `[[10,0.01],[25,0.01],[50,0.01],[10,0.03],[25,0.03],[50,0.03]]`; 8 DEV examples; 1,805 s.

| Cell | score (median `s_e`) | wall (s) |
|---|---|---|
| J=10, lr=0.01 (**default**) | 0.00213840 | 169.6 |
| J=25, lr=0.01 | 0.00127321 | 226.0 |
| **J=50, lr=0.01 (ADOPTED)** | **0.00083133** | 321.6 |
| J=10, lr=0.03 | 0.00256496 | 172.3 |
| J=25, lr=0.03 | 0.00155268 | 226.1 |
| J=50, lr=0.03 | 0.00115508 | 316.8 |

- Adoption threshold = 0.5 × default = **0.00106920**; J=50/lr=0.01 qualifies at 0.00083133 ⇒ `adopted: true`.
- `projection_seconds_per_example` = **40.195** s.
- Plateau: improvement J=25→50 at the adopted lr = **34.71 %** ≥ 10 % ⇒ **`plateau: "recipe-limited"`** — i.e. the probe's own verdict is that **more inner iterations were still buying accuracy**; the recipe, not the reconstruction, was the limit at the grid's edge.
- `reasons`: `["adopted J=50, lr=0.01 at score 0.000831327", "plateau: recipe-limited"]`.

### 4.4 [DEVIATION — MATERIAL] The adopted recipe never reached the capacity run

**Finding.** The adequacy probe adopted **J=50, lr=0.01**. The capacity run that produced G1/G2 ran at
**J=10, lr=0.01** — the default. This is verified three independent ways:

1. `capacity_att-0806-164625/run_report.json` → `"recipe": {"inner_iters": 10, "lr": 0.01}`;
2. the published shard provenance header
   `…/capacity_att-0806-164625/a1/shard_00000/header.json` → `"optimization_config": {"inner_iters": 10, "lr": 0.01}`
   (and `build_capacity_records` refuses records that advertise a recipe they were not produced at, so this is the recipe that actually ran);
3. the same for the TRAINFIT root.

**Root cause (traced, not inferred).** `bash_scripts/run_wan_null_inversion.sh` **never passes
`null_adequacy_uri`** — the key is declared in `base_wan_5b_null_inversion.yml` (line 295, default `''`) and
consumed correctly by `load_adoption` / `apply_adopted_recipe`, but no launcher path sets it. With an empty URI
`load_adoption` returns `None` (`if not uri or not exists(uri): return None`), `apply_adopted_recipe` is never
called, and capacity silently uses the launcher defaults `NULL_INNER_ITERS=10 / NULL_LR=0.01`. The fail-closed
guard that the R10 follow-up review added (finding 3) covers *"an artifact exists at the configured URI but
cannot be understood"* — it does **not** cover *"the URI is empty"*. Its own docstring names this exact failure:

> "An adequacy artifact sitting at the configured URI while capacity quietly runs at (J=10, lr=1e-2) is the
> failure that produces a perfectly plausible set of gate verdicts for an experiment nobody chose."

**Contrast that isolates it:** the sibling experiment's launcher `bash_scripts/run_wan_pos_inversion.sh` **does**
wire `POS_ADEQUACY_URI → pos_adequacy_uri` (line 318), and `submit_k1.sh` discovers and passes it — so exp_05's
capacity ran at its adopted **J=50** (verified in `…/pos_context/k1/capacity/b1/shard_00000/header.json`).
Same adoption logic, same adopted cell, opposite outcome, because of one missing launcher line.

**Consequences (stated factually; weighed in the analysis doc §4):**
- Plan v5 §4-P1's clause "*G1/G2 are evaluated once, on the adopted recipe only*" was **not satisfied**.
- The projection would have permitted the re-run: 40.195 s × 64 examples × `RERUN_ARMS=2` = **5,145 s ≈ 1.43 h**, under the plan's ≤ 2 h re-run budget (`RERUN_BUDGET_SECONDS = 7200`). The deviation is *not* a budget stop; it is a plumbing gap.
- **Direction is not uniform.** For G2 (bars 0.75 / 0.70 vs measured 0.4973 / 0.4697) the deviation is conservative — a stronger optimizer could only have raised A2. For **G1 it is not**: G1 failed on `median_ratio` alone at **3.605 against a bar of 5**, and the adopted recipe cut the tracking loss **2.57×**; whether J=50 would have crossed 5 is **unmeasured**.
- The **selection verdict is unaffected**, because it is over-determined by the A1-probe floors (0.1729 vs 0.70), which fail by a factor of four and were measured on locked nulls whose transfer is a property of the optimization *procedure*, not its iteration count. [JUDGMENT — this is the load-bearing claim of §4.4 and the first thing a reviewer should attack.]
- **Cross-experiment comparability is compromised:** exp_04's A1 (0.8523) and exp_05's B1 (0.9227) were optimized at **different budgets** (J=10 vs J=50). See analysis §4.2.
- **Not recorded anywhere before this document** — not in the worklog, `_command.md`, the tracker, or CLAUDE.md. `null_adapter_params_set_up.md` states "FULL CAPACITY … at the production recipe (adoption consumed via `NULL_ADEQUACY_URI`)", which was never achievable through the shipped launcher.

### 4.5 L_null ablation (diagnostic-only, plan v5 §4-P1 N5)

Same 8 DEV examples, recipe [10, 0.01], `diagnostic_only: true`:

| L_null | score | note |
|---|---|---|
| L_nat = **1** | 0.00302201 | the natural (non-padding) length of the T5("") context is **1 row** |
| L_null = **16** | 0.00213840 | 1.41× better; **L=16 stays fixed for P2/P3 regardless** (plan N5) |

### 4.6 A3 single-update measurement (`a3_measurement.json`, plan v5 §4-P1 F11 stops)

| Quantity | Value | Predeclared stop | Verdict |
|---|---|---|---|
| compile | **412.33 s** | abort if > 30 min (1,800 s) | pass |
| one update (`step_seconds`) | **6.537 s** | abort if > 120 s | pass |
| 300 iters @ job batch 8 (`compute_seconds`) | **2,395.4 s** | — | — |
| projection | **0.7487 h** | J1b proposed only if ≤ 4 h | pass |
| peak HBM / device | **15,492,600,832 B ≈ 14.43 GiB** (`current` 12,880,283,136 B) | no OOM | pass |
| `verdict` / `fits_budget` | **`"ok"` / `true`**, `reasons: []`, `preliminary: true` | — | — |

> Note: `peak_hbm_bytes` is 15.49 **GB** decimal = 14.43 GiB. Prior summaries said "~15.5 GB" (decimal) and one
> said "13 GB/16 GB" for the J1b re-run's own probe (13,040,069,632 B = 12.14 GiB) — both refer to real numbers
> in different artifacts; stated here with units to stop the drift.

---

## 5. J1b — A3 joint endpoint optimization (P1b)

Root `…/j1r2/j1b_att-0806-211405/`; `a3_direct_opt.json` + `a3_nulls.npz`. First 8 DEV examples, 300 iterations,
3,237 s wall, `code_sha db8c3dc`, embedded re-run fit-probe `verdict: "ok"` (`projection_hours` 0.736).

**Two distinct reported quantities — do not conflate:**
- `final_loss` = the **last pre-update** objective value (the R3 logging convention, mirroring the reference). Mean **0.2709**.
- `final_endpoint` = the endpoint future-MSE **after** the last update — the quantity J1c re-measures by replay. Mean **0.2731**.

The worklog's J1b table reports `final_loss`; J1c's own-basin column reports `final_endpoint`. Both round to 0.27.

### 5.1 Per-clip, all four regimes on the same 8 examples (latent future-MSE, lower is better)

| Clip | A2-0 (do nothing, fresh ε₀) | A2 (greedy, fresh ε₀) | **A3 `final_loss`** | **A3 `final_endpoint`** | A1 (greedy, own basin) |
|---|---|---|---|---|---|
| ep12399 | 5.157 | 0.499 | 0.6807 | 0.7029 | 0.0527 |
| ep45499 | 6.300 | 0.037 | **0.0067** | **0.0067** | 0.0039 |
| ep26599 | 4.520 | 0.491 | 0.2425 | 0.2417 | 0.0392 |
| ep19599 | 4.768 | 0.340 | 0.3522 | 0.3505 | 0.1478 |
| ep42299 | 4.766 | 0.449 | 0.3668 | 0.3667 | 0.1282 |
| ep67499 | 4.743 | 0.701 | **0.1265** | **0.1258** | 0.0449 |
| ep28899 | 4.716 | 0.566 | 0.2908 | 0.2905 | 0.1161 |
| ep21099 | 4.603 | 0.352 | **0.1007** | **0.1001** | 0.0534 |
| **mean** | **4.946** | **0.4294** | **0.2709** | **0.2731** | **0.0733** |

### 5.2 Readings against the artifact

- **Cross-job consistency:** J1b's `initial_loss` mean **4.9477** reproduces A2-0's mean on the same 8 (**4.9465**) to 0.02 % — the fresh-noise starting point is consistent across independently launched jobs.
- **Joint beats greedy on 6/8 clips**, by up to **5.57×** (ep67499: A2 0.701 → A3 0.1258).
- **3/8 clips reach own-basin quality from fresh noise:** endpoints **0.0067 / 0.1001 / 0.1258** all fall inside A1's own range on these 8 clips, **[0.0039, 0.1478]**.
- **Convergence is uneven:** ep12399 stalls at 0.703 — worse than its *own-basin* A0 control (0.4341). Gradient norms fell 8.49–16.65 → 0.0029–0.308 (most examples converged inside the 300-iter budget; ep12399's last grad norm 0.308 is the largest).
- **J1b measures latent MSE only** — no decode, therefore **no SSIM** at this stage.

---

## 6. J1c — transfer probe (P1c)

Root `…/j1r2/j1c_att-0807-020621/transfer_probe.json`. Replays J1b's `a3_nulls.npz` (verified step-major
`[25, 8, 16, 4096]`; a batch-major misread would have transposed the study and was killed as R12 mutant X2) under
four noise settings, decodes, and reports SSIM/MSE. 468.9 s. Provenance binds `nulls_sha256`
`677502c58a65…eb768` = J1b's file, `code_sha 9213585`, `sigma_steps 25`, `l_null 16`, `guide_scale 5.0`.

### 6.1 Means over the 8 clips

| Setting | mean future-SSIM | mean full-SSIM | mean future-MSE |
|---|---|---|---|
| `global(0)` — own ε₀ | **0.6509** | 0.6615 | **0.2731** |
| `keyed(0)` foreign | **0.4709** | 0.4870 | 1.3148 |
| `keyed(1)` foreign | **0.4765** | 0.4923 | 1.4234 |
| `keyed(2)` foreign | **0.4784** | 0.4942 | 0.9891 |
| *(pooled keyed)* | *0.4753* | *0.4912* | *1.2424* |

Reference points on the same 8 clips: **A1-probe (greedy nulls, foreign) = 0.1633**; A2-0 (do nothing, fresh) MSE
**4.946**. Cohort-wide A1-probe = 0.1729.

### 6.2 Relative retention — [JUDGMENT] the estimator matters, so all three are given

| Estimator | Value |
|---|---|
| `keyed(0)` mean ÷ own mean (the worklog's figure, 0.471/0.651) | **0.7235** |
| pooled keyed mean ÷ own mean | **0.7301** |
| mean of the 24 **per-example paired** ratios | **0.7007** (range 0.428–0.855) |

The worklog and tracker state "**~72 %**", which corresponds to the first estimator. All three exceed the
plan's 0.7× relative conjunct; **none of them changes the verdict**, because the **absolute floor of 0.70 is the
binding constraint** (per §2.2's standing note) and foreign SSIM ~0.47 misses it everywhere. The
"2.8× the greedy probes" figure resolves to **2.75×** against the cohort-wide A1-probe (0.4753/0.1729) or
**2.91×** against the matched 8-clip A1-probe (0.4753/0.1633).

### 6.3 Per-clip transfer

| Clip | own `global(0)` SSIM | `keyed(0)` | `keyed(1)` | `keyed(2)` | mean retention |
|---|---|---|---|---|---|
| ep45499 | 0.991 | 0.729 | 0.878 | 0.880 | 0.837 |
| ep21099 | 0.834 | 0.691 | 0.747 | 0.701 | 0.855 |
| ep67499 | 0.829 | 0.566 | 0.591 | 0.583 | 0.700 |
| ep28899 | 0.692 | 0.607 | 0.421 | 0.673 | 0.819 |
| ep26599 | 0.590 | 0.277 | 0.253 | 0.227 | 0.428 |
| ep42299 | 0.585 | 0.446 | 0.495 | 0.403 | 0.766 |
| ep19599 | 0.416 | 0.269 | 0.254 | 0.225 | 0.600 |
| ep12399 | 0.271 | 0.183 | 0.172 | 0.134 | 0.600 |

**Quantified (n = 8, no CIs — see §9):** own-basin SSIM predicts **absolute** foreign SSIM very strongly
(Pearson **+0.945**, Spearman **+0.976**) but predicts **relative retention** only weakly (Pearson **+0.623**,
Spearman **+0.714**). [JUDGMENT] The defensible statement is therefore *"better-optimized clips transfer to
better absolute foreign quality"*, **not** *"better-optimized clips retain a larger fraction"* — the worklog's
"transfer quality tracks optimization quality" is true of the absolute reading and only weakly of the relative one.

- Own-basin `global(0)` MSE column reproduces J1b's `final_endpoint` as an exact multiset — the cross-job consistency check passed.
- **Absolute floors are unmet everywhere:** own-basin 0.651 and foreign ~0.47 both sit below the 0.70 floor. At 300 iterations the joint optimum is itself not deployment-grade.

---

## 7. Coverage, integrity, and validation state

| Property | Evidence |
|---|---|
| Cohort coverage | `coverage_ok: true`, `missing_names`/`extra_names`/`duplicate_names` all empty on every gate, both cohorts |
| Invalid pairs | `invalid_pairs: 0`, `invalid_fraction: 0.0`, `imputed_method_ssim: 0` — **no imputation was exercised in this experiment** |
| Quarantines | `"quarantined": {}` in both `run_report.json` files |
| k-sets | G1/G2 evaluated at `k_set: [0]`; probes at `k_set: [0,1,2]` — as predeclared |
| Bootstrap | 10,000 resamples, seed 20260804, every gate |
| Parity audit | Recorded CLEAN 2026-08-06T03:00Z (worklog), 11 components + numeric-defaults cross-check + data parity |
| Test suite | 989 passing at the R12 tip; ~470+ mutants killed cumulatively across R1–R11; 6 ratified defence-in-depth survivors |
| Wall times | capacity DEV 8,667.9 s; capacity TRAINFIT 2,323.5 s; adequacy 1,805.4 s; J1b 3,237.1 s; J1c 468.9 s |

---

## 8. Artifact index and record corrections

### 8.1 [DEVIATION — RECORD] J1-4's submitted tip

`null_adapter_command.md` records J1-4 at tip `27efcd1`. The published artifacts record
`code_sha 3bdbd2a141484ad9c8e1efebe897c9a562c8f712`. Verified: `3bdbd2a` ("docs(exp_04): J1-3 triage — runbook
idempotency under auto-retry; J1-4 prepared") **is a descendant of `27efcd1`** with
`git diff --stat 27efcd1 3bdbd2a -- src bash_scripts` **empty**. The ledger's substantive claim (executable tree
unchanged across J1-2/3/4) is **correct and now verified**; only the literal SHA in the J1-4 entry is wrong.
Corrected here rather than by editing the append-only ledger.

### 8.2 Published artifacts

| Artifact | URI | Digest / key provenance |
|---|---|---|
| J0 cohort manifests | `gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/` (+ committed under `j0_manifests/`) | listing checksum `5827f4da…0d14` |
| DEV-64 capacity | `…/j1r2/capacity_att-0806-164625/` — `selection.json`, `gate_tables.json`, `run_report.json`, `a3_measurement.json`, `a1/`, `a2/`, `videos/` | `manifest_hash 433f8691…3f76fa` |
| TRAINFIT-16 capacity | `…/j1r2/capacity_trainfit_att-0806-164625/` (same set minus A3) | `manifest_hash b4e62fc9…45abb66` |
| Adequacy | `…/j1r2/adequacy/adequacy_report.json` | published by J1-3 attempt 1; adopted by J1-4 |
| J1b | `…/j1r2/j1b_att-0806-211405/` — `a3_direct_opt.json`, `a3_nulls.npz` | `code_sha db8c3dc`; npz `sha256 677502c5…eb768` |
| J1c | `…/j1r2/j1c_att-0807-020621/transfer_probe.json` | `code_sha 9213585`; binds the npz sha256 above |
| Superseded | `…/j1r2/capacity/` (J1-3 attempt-1 partial, no run-level JSON); `…/j1r2/smoke/`; `…/j1/` (J1-1/J1-2 era) | retained for the record |
| Common | model revision `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff7315c768468a5333511427288870b2e9635`; `base_context_fingerprint 6c79000e…373b`; `guide_scale 5.0`; `l_null 16`; 25-step σ grid, shift 5.0, σ₀ = 1.0 | |
| Comparison videos | 8 per cohort under `capacity*/videos/` — **not yet pulled or reviewed**; the HTML report (SOP artifact 12) is outstanding | |

### 8.3 Numbers corrected against the artifacts

| Where | Said | Artifact says |
|---|---|---|
| Worklog J1b table, tracker | J1b mean 0.270 | `final_loss` **0.2709** ✓; but `final_endpoint` (the replayed quantity) is **0.2731** — §5 |
| Worklog J1c, tracker | "~72 % retention", "2.8×" | estimator-dependent: **0.7235 / 0.7301 / 0.7007**; **2.75× / 2.91×** — §6.2 |
| Worklog J1 reading | A3 "peak HBM ~15.5 GB" | **15,492,600,832 B = 15.49 GB = 14.43 GiB** — §4.6 |
| Worklog J1 reading, tracker | A1-probe "actively destructive, WORSE than doing nothing" | A1-probe **0.1729** vs the A2-0 do-nothing proxy **0.1423** — the probe is **+0.031 above** it. The claim holds for exp_05's B2, not for exp_04's probe — §4.2 |
| `params_set_up.md` | capacity ran "at the production recipe (adoption consumed)" | ran at the **default J=10** — §4.4 |
| `_command.md` J1-4 | tip `27efcd1` | artifacts record `3bdbd2a` (docs-only descendant, identical `src/`) — §8.1 |

All other figures in the worklog's J1-4 / J1b / J1c readings reproduce **exactly**.

---

## 9. Caveats that bound every number in this document

1. **The 0.2946 anchor is not a quality baseline.** The deployed `pre_context` adapter's rollout SSIM 0.2946 comes from a **4-sample** validation at step 30,000 — and those four samples are **four correlated windows of a single episode**. It is a **wiring/sanity check**, never a statistically meaningful quality baseline, and no comparison in this document or the analysis treats it as one.
2. **A1/A2/A3 are ORACLES, not deployable systems.** Every arm optimizes conditioning **per clip against that clip's own ground-truth latents**. They bound what the conditioning channel can express; they say nothing directly about what an amortized, action-conditioned predictor can achieve.
3. **J1b/J1c are n = 8**, first-8-DEV, **no confidence intervals**, no multiplicity control. J1b is **latent-MSE only**; SSIM appears only in J1c. The n=8 correlations in §6.3 are descriptive.
4. **The gates ran at a recipe the experiment's own adequacy probe rejected** (§4.4) — the single largest threat to the quantitative claims here.
5. **Exp_05's method pivots were never shared with exp_04.** exp_05 inverts with an **8-token** context; exp_04 with the **512-row padded** context with 16 rows replaced. The trajectories, controls, and pivots are different objects; cross-experiment tables are descriptive only (exp_05 plan §4's H1 interpretation note, and its `A0` vs `B0` control asymmetry — A0 collapses CFG to identity, B0 does not).
6. **One cohort, one dataset, one backbone, one guidance scale (w = 5).** DEV-64/TRAINFIT-16 of DROID at 192×320×32f through Wan2.2 TI2V 5B. TEST-64 was never touched — correctly, since selection stopped at P1.
7. **σ₀ = 1.0** here vs **0.999** in the PyTorch reference (ratified deviation, plan §8 register): no cross-repo artifacts were exchanged, so prior-art numbers are directional context, not matched comparisons.
8. **The `L_nat` ablation cell used L = 1**, the natural length of T5(""), so the ablation contrasts 1 vs 16 rows — not "a shorter padded context" vs 16.
