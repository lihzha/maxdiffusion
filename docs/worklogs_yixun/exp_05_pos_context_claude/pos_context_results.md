# pos_context_results — exp_05 P4′ (SOP artifact 10)

**Status: FINAL — revision 3.** Revision 2 answered the first Codex analysis review; **revision 3 applies the
closing review** (`pos_context_codex_closing_review.md`, REQUEST-REVISION — *"the exp_05 STOP and gate record
remain valid"*; two scoping leaks, the exp_04-J=50 fold-in, and artifact-index staleness; no new compute). All
four items are applied. **This document is submitted for closure adjudication.** Judgment calls are flagged
**[JUDGMENT]**, deviations from plan v3 **[DEVIATION]**, and edits are marked **[REV2]** / **[REV3]**. The reviewer independently reproduced the entire gate
record — including the corrected B0 = 0.32147 and B1/B0 = 2.87022× — so the numbers below stand; three of the
*analysis* document's conclusions did not.

This document is the **factual record only**: what was predeclared, what ran (including the failed attempt and
its classification), the numbers exactly as the gate module computed them, the gate verdicts, and the artifact
URIs with their recorded digests. Interpretation lives in `pos_context_analysis.md`.

**Verification basis.** Every number below was re-derived from the published artifacts pulled from GCS on
2026-08-08 (cached locally for the write-up), not from the worklog reading. Where the worklog and the artifact
disagree, the artifact wins and the discrepancy is recorded in §8 — **one such correction is material and is
called out in §4.1.** Gate metric = **`future_ssim` / `future_mse`** — the non-pinned latent frames (latent
frames 1–8 ⇒ pixel frames 1–32); frame 0 is the pinned first-frame condition and is excluded throughout.

---

## 1. Scope and terminal status

| Item | Value |
|---|---|
| Experiment | exp_05 `pos_context` — per-step **positive** text embeddings by DDIM inversion + the existing `pre_context` adapter trained by teacher-forced regression |
| Plan of record | `plan_pos_context.md` **v3** (3 Codex review passes, all findings accepted, APPROVE-PLAN 2026-08-04); binds exp_04 plan v5 by reference for manifests / noise / gates / integrity / verifier contracts |
| Phases executed | **P1′** (job K1 — positive reconstruction study + basin probe) |
| Phases NOT executed | **P2′** (K2 target caching), **P3′** (K3 regression training, K4 eval) — stopped by the predeclared target-selection rule; **STOP honored by Yixun 2026-08-06** ("Honor the STOP for K2") |
| Terminal verdict | **TARGET = STOP on both cohorts** — H1 PASS but the transfer floors fail; **and** H2 FAIL on all four conjuncts. **[REV2]** Both are required: the selection rule rejects B1 on the probe floors *and* rejects the B2 fallback because H2 failed. This verdict was produced at the **adopted** recipe (§4.4), so it is the predeclared verdict. **[REV3]** *(Revision 2 added "unlike exp_04"; exp_04's J1-5 clean-gate rerun has since produced its verdict at the adopted recipe too, so the contrast is removed.)* |
| Tips **[REV2, updated REV3]** | STOP-decision tip `0f505d3` · report revision 1 `2e4bc12` · revision 2 `d9a306e` · capacity-videos code + HTML `6f3146a`. **Evidence-producing tip: K1-2 ran `bb845ea`**; the capacity-videos job ran the same. A branch tip plus "clean" is mutable and is not quoted as provenance |
| Code state | S1–S8 + S10a complete and committed; **S9 held open** at the R14/R15 matrix stall (tripwired); S10's trainer launcher parked. Suite **1,417** tests |

**The trainer stack that will not be used (yet).** S6 (gather/loss), S7 (regression trainer + checkpoint
redesign), S8 (dispatch/config) and two-thirds of S9 (eval restore + the stamped pre-K4 DEV-gate certificate)
were built, reviewed and committed **before** K1's verdict landed. They remain on the branch, green, should a
future direction revive them. Recording this so the STOP is not misread as "nothing was built".

---

## 2. What was predeclared (plan v3 §4-P1′)

### 2.1 Arms

Gate forms are **exp_04's G1/G2 verbatim** (thresholds, imputation, k-sets); arm letters are B-for-positive.

| Arm | Definition | Role |
|---|---|---|
| **B0** | frozen `C_init` = `truncate_or_pad(T5(""), 8)` in the conditional slot, CFG replay from `traj[0]` | control for B1. **Deliberately unlike exp_04's A0**: because the conditional context is 8 tokens and the unconditional is the full 512-row T5(""), `v_cond ≠ v_uncond`, so **B0 is an ACTIVE-CFG control** and genuinely depends on w (structural separation measured at 7.9e+1 in S3) |
| **B1** | C optimized per step from `traj[0]`, replayed from `traj[0]` | own-basin capacity |
| **B1-probe** | B1's optimized C locked, replayed from `keyed(k)`, k∈{0,1,2} | transfer |
| **B2-0** | frozen `C_init`, replay from ε₀ = `global(0)` | control for B2 |
| **B2** | C optimized per step from ε₀, replayed from ε₀ | fresh-noise capacity |
| **B2-probe** | B2's C locked, replayed from `keyed(k)`, k∈{0,1,2} | diagnostic |

**Representation (plan review finding F1 — the decisive design correction).** C ∈ ℝ^{8×4096} is passed
**directly as the entire `encoder_hidden_states`** (sequence length 8), exactly what
`wan_pre_context_adapter_forward` feeds the transformer — **not** spliced into a 512-row padded context.
Plan v1 had it the other way; the reviewer caught that targets and deployment would have been different
representations, which would have invalidated P3a′ and every oracle comparison.

### 2.2 Gates (`null_adapter_gates.py`, imported unchanged; paired unit = example; 10,000 resamples, seed 20260804, percentile CIs)

- **H1** (B1 vs B0): median ratio `future_MSE(B0)/future_MSE(B1)` **≥ 5** AND **≥ 80 %** improved AND mean `future_SSIM(B1)` **≥ 0.80** with 95 % CI-low **≥ 0.75**.
- **H2** (B2 vs B2-0, from ε₀): median ratio **≥ 5** AND **≥ 80 %** improved AND mean `future_SSIM(B2)` **≥ 0.75** with 95 % CI-low **≥ 0.70**.
- **Target-selection rule:** select B1 iff **H1 passes AND** B1-probe ≥ 0.7 × mean SSIM(B1) **and** B1-probe ≥ **0.70 absolute**. Else select B2 iff **H2 passes**. Else **stop after P1′**.

> **Two standing interpretation notes, both predeclared and both binding on this document:**
> 1. **(exp_04 R5 review)** On SSIM ∈ [0,1] the absolute floor (≥ 0.70) **strictly implies** the relative test (≥ 0.7 × B1). No result statement may attribute a probe rejection to the relative conjunct alone — **the absolute floor is always the binding constraint.**
> 2. **(exp_05 plan §4, S3-review amendment)** H1's estimand is B1's lift over a frozen-context **active-CFG** control. It is **NOT directly comparable to exp_04's G1**, where A0 collapses CFG entirely. Any cross-experiment table must carry this caveat. Honored in §4.3 and in the analysis doc.

### 2.3 Adequacy probe and adoption

Identical statistic, grid, adoption rule, ±2 h re-run budget and plateau rule as exp_04 (plan v5 §4-P1,
inherited by reference).

---

## 3. Job trail — every attempt, including the failure

| Job | Date (UTC) | Tip | Outcome | Classification |
|---|---|---|---|---|
| **K1-1** | 2026-08-05T20:33Z | `9d326e8` | **FAILED** at attempt 2 (attempt 1 = infra preemption), 23 s into the command: `ValueError: Requested key code_sha, not in config` at `mode_kwargs` (`run_wan_null_inversion.py:654`). The positive-slot smoke reached backend load + revision resolution — the S10a launcher, preflights, HF prefetch and config **all worked** — then died at the first *shared-code* `code_sha` read. **Nothing published**; the `…/k1` root stayed clean | **REAL BUG — inherited stale copy.** Exactly the J1-1 defect class (issue #11: 3-arg `getattr` never falls back on `HyperParameters`). exp_05's copy of the shared entrypoint predated exp_04's fix `925ee17`: merge-interim took exp_04 at the R10 boundary, the fix landed *after*, and **no propagation step existed** |
| — | **2026-08-06T01:22:54Z** **[REV2]** | `0d1f4a5` | **Remediation:** merge-interim-2, one-way exp_04 @ `27efcd1` → this branch (brings the fix + R11). Two dual-touch conflicts resolved additive-union; `optional_config_value` unified on exp_04's reviewed helper verbatim; combined suite **1,353** green; exp_04's AST guard `test_no_three_argument_getattr_on_config_survives` now enforces issue #11 on this branch | — |
| **K1-2** | 2026-08-06T01:24Z | `bb845ea` | **SUCCEEDED** — all four phases; attempt 1 after one infra preemption. Same `submit_k1.sh` verbatim (the `…/k1` root was reusable because K1-1 published nothing) | — |

`bb845eaf43de2f05abbf99dfd4344a51bc2930bc` is confirmed as the tip that actually ran: it is the `code_sha`
recorded in the published shard headers.

> **[REV2] Chronology correction — the remediation timestamp.** Revision 1 dated merge-interim-2 at
> **02:05Z**, taken from the exp_05 worklog's entry heading. That is **impossible as written**: K1-2 launched at
> **01:24:45Z** (per its queue job id `20260806-012445-…`), so the merge that enabled it cannot postdate it.
> **`git log` gives commit `0d1f4a5` at `2026-08-05T21:22:54-04:00` = `2026-08-06T01:22:54Z`** — 111 seconds
> before the launch, which is coherent. The worklog's `2026-08-06T02:05:00Z` heading is **erroneous**; because
> the worklog is append-only, it has been annotated there with a correction entry rather than edited. The
> ordering of events was correct throughout; only the stamp was wrong.

**K1-2's four phases** (cohorts are one-per-invocation, so K1 is four launcher invocations — the reading
ratified in the S10a follow-up review): smoke capacity (n=2) → adequacy_probe (first-8 DEV) → capacity dev64 →
capacity trainfit16, both capacity phases consuming the **one DEV** adequacy artifact, under **distinct artifact
roots** so TRAINFIT cannot overwrite the DEV-authoritative selection/tables. *(The same review reading exposed
exp_04's missing TRAINFIT half — remediated there as J1-2b/J1-3 phase 5.)*

**K1-1's crash also produced a standing process rule** (now CLAUDE.md / issue #11): when a fix lands in a file a
sibling experiment carries a copy of, name that branch in the fix round's closing entry and either propagate
immediately or record the deferred merge as a launch blocker there.

---

## 4. K1-2 — P1′ capacity study and basin probe (the primary result)

Authoritative roots: `gs://v6_east1d/datasets/droid_wan_pos_context/k1/capacity` (DEV-64) and
`…/k1/capacity_trainfit` (TRAINFIT-16). Full coverage, **zero quarantines, zero invalid pairs,
`coverage_ok: true`** on every gate in both cohorts. Recipe actually executed: **`{inner_iters: 50, lr: 0.01}`**
— the adopted recipe (see §4.4). `l_pos: 8`, `embedding_slot: "positive"`.

### 4.1 Per-arm means (gate metric `future_ssim`)

**DEV-64** (n = 64; probe arms average over k ∈ {0,1,2} ⇒ 192 observations)

| Arm | mean future-SSIM | mean full-SSIM | mean future-MSE | mean future-pixel-MSE |
|---|---|---|---|---|
| **B0 (control)** | **0.3215** | 0.3420 | 0.8897 | 0.05164 |
| **B1** | **0.9227** | 0.9250 | 0.0369 | 0.00197 |
| B1-probe | **0.5254** | 0.5398 | 0.6387 | 0.03543 |
| B2-0 (control) | **0.2814** | 0.3032 | 1.4178 | 0.07526 |
| **B2** | **0.1610** | 0.1865 | 5.6250 | 0.31332 |
| B2-probe | 0.2118 | 0.2357 | 5.9509 | 0.36205 |

**TRAINFIT-16** (n = 16)

| Arm | mean future-SSIM | mean full-SSIM | mean future-MSE |
|---|---|---|---|
| B0 | 0.3115 | 0.3324 | 0.8820 |
| **B1** | **0.9095** | 0.9122 | 0.0397 |
| B1-probe | **0.4885** | 0.5040 | 0.8140 |
| B2-0 | 0.2803 | 0.3021 | 1.3944 |
| **B2** | **0.1570** | 0.1825 | 5.8810 |
| B2-probe | 0.2121 | 0.2359 | 5.4751 |

> ### [DEVIATION — RECORD, MATERIAL] B0 is **0.3215**, not "≈ 0.25"
> The worklog's K1 reading (2026-08-06T16:35Z), the master tracker and CLAUDE.md all state the frozen-context
> control at **"~0.25"** and derive **"~3.6× the frozen-context control"**. The published `gate_tables.json`
> gives **B0 mean future-SSIM = 0.32147** (median 0.30456, **min 0.01763, max 0.59362** **[REV2]**; full-SSIM
> 0.34203); TRAINFIT B0 = 0.31153. The control pairing is confirmed by exact reproduction of the gate's own
> statistic: median of `future_MSE(B0)/future_MSE(B1)` per example = **28.6320**, identical to `selection.json`'s
> `median_ratio` to five decimals, so B0 *is* H1's control and 0.3215 *is* its mean. The reviewer independently
> reproduced both 0.32147 and 2.87022×.
> **Corrected multiples: B1 is 2.87× B0** (not 3.6×) and **3.13× the 0.2946 anchor** (that second figure was
> right).
> **[REV2] What is and is not a gate input — revision 1 got this half wrong.** B0 **IS** a gate input: its
> **future-MSE** (mean 0.8897) is the numerator of H1's `median_ratio` statistic. What is **not** a gate input is
> **B0's future-SSIM** — the quantity corrected here — which appears only in prose multiples. That is why **no
> gate verdict changes**: the corrected number never entered a conjunct, and the MSE that did enter was never in
> dispute. Revision 1's phrasing ("B0 is not a gate input, only a ratio denominator") conflated the two.
> **[JUDGMENT]** I could not reconstruct where 0.25 came from; the nearest published numbers are B2-probe 0.2118
> and B2-0 0.2814. The reviewer did not identify a source either.

### 4.2 Gate verdicts as computed (`selection.json`)

| Gate | Cohort | median ratio (≥ 5) | frac improved (≥ 0.80) | mean SSIM | 95 % CI | verdict |
|---|---|---|---|---|---|---|
| **H1** (B1 vs B0) | DEV-64 | **28.6320** ✓ | **1.0000** ✓ | **0.9227** ✓ (≥0.80) | **[0.9129, 0.9314]** ✓ (low ≥0.75) | **PASS** (`reasons: []`) |
| **H1** | TRAINFIT-16 | **26.0421** ✓ | 1.0000 ✓ | **0.9095** ✓ | [0.8839, 0.9270] ✓ | **PASS** |
| **H2** (B2 vs B2-0) | DEV-64 | **0.2540** ✗ | **0.0000** ✗ | **0.1610** ✗ | **[0.1358, 0.1868]** ✗ | **FAIL — all four conjuncts** |
| **H2** | TRAINFIT-16 | **0.1962** ✗ | 0.0000 ✗ | **0.1570** ✗ | [0.1076, 0.2082] ✗ | **FAIL — all four conjuncts** |

**Transfer (target-selection conjuncts):**

**[REV2]** Column 3 is the **ratio** `B1-probe / B1`, tested against 0.7 — revision 1 headed it "0.7 × B1",
which named the threshold rather than the statistic.

| Cohort | B1-probe abs (floor 0.70) | `B1-probe / B1` (≥ 0.7) | verdict |
|---|---|---|---|
| DEV-64 | **0.5254** ✗ | **0.5695** ✗ | fails; **the absolute floor is the binding constraint** |
| TRAINFIT-16 | **0.4885** ✗ | **0.5371** ✗ | same |

**Selection: `target = "stop"` on both cohorts**, reasons verbatim from `selection.json`:

```
"A1-probe below 0.7x the A1 mean (transfer)"
"A1-probe below the 0.70 absolute floor"
"G2 failed (median_ratio, fraction_improved, mean_ssim, ssim_ci_low)"
```

*(The reason strings say "A1"/"G2" because the gate module is exp_04's, imported unchanged — a cosmetic label
carry-over, not a wrong statistic. The `selection.json` keys are `h1`/`h2`, the `run_report.json` keys are
`g1`/`g2`, and the numbers under them are exp_05's B-arms. Noted so no reader mistakes it for cross-experiment
contamination.)*

**One-line gate readings (no interpretation beyond the gate):**
- **H1 passes on every conjunct, with margin** — the median MSE ratio is 28.6× against a bar of 5, and 64/64 examples improved.
- **H2 fails on every conjunct**, and its median ratio of **0.254 is below 1**: optimizing from fresh noise made the reconstruction **worse than the frozen-context control** on **0 of 64** examples improved.
- The probe sits at 0.5254 against a 0.70 floor — a partial, not catastrophic, transfer loss.
- **[REV2]** The verdict requires **both** legs of the selection rule: B1 is rejected on the transfer floors, **and** the B2 fallback is rejected because H2 failed. Revision 1 said the verdict rests "entirely on the transfer floors — unlike exp_04, exp_05's capacity gate H1 *passed*". **Both halves are removed**: the first is logically incomplete (the rule's B2 branch must also fail for STOP to be reached), and the second invites precisely the cross-experiment comparison §4.3 forbids.

### 4.3 Cross-experiment framing constraint (predeclared)

H1's PASS and exp_04's G1 FAIL are **not** a like-for-like slot comparison:
(a) exp_04's A0 collapses CFG to identity while exp_05's B0 is an active-CFG control (§2.1 / §2.2 note 2);
(b) exp_04 replaced 16 rows inside a 512-row padded context, exp_05 passes 8 tokens as the whole context;
(c) **the pivots are different objects** — exp_05 inverts at w=1 with the 8-token `C_init`, exp_04 with the
512-row context, so the two experiments' inversion trajectories, controls and targets are not shared (plan §3,
pinned by S4 mutant R2); and
**[REV3] (d) budgets — RESOLVED, no longer a mismatch.** Historically exp_05 ran at the **adopted J=50** while
exp_04's J1-4 ran the **unadopted J=10** (its launcher never passed `null_adequacy_uri`; exp_04
`null_adapter_results.md` §4.4, issue #15), which left exp_04's formal selection INDETERMINATE. **exp_04's J1-5
clean-gate rerun fixed the launcher, re-ran both cohorts at J=50, and retained the predeclared STOP** — so both
experiments' headline numbers are now at the **same adopted recipe**, and exp_04's authoritative figures are
A1 **0.8868**, A2 **0.6638**, A1-probe **0.1666**, G1 ratio **4.681×**.

**Three mismatches remain and they are sufficient:** (a) controls, (b) representations, (c) pivots.
**Descriptive side-by-side tables only.**

### 4.4 Adequacy probe — adopted **and honored**

`…/k1/adequacy/adequacy_report.json`; 8 DEV examples; 1,821.8 s; `manifest_hash 433f8691…3f76fa`.

| Cell | score (median `s_e`) | wall (s) |
|---|---|---|
| J=10, lr=0.01 (**default**) | 0.00563540 | 135.3 |
| J=25, lr=0.01 | 0.00362877 | 184.5 |
| **J=50, lr=0.01 (ADOPTED)** | **0.00238379** | 273.0 |
| J=10, lr=0.03 | 0.00746982 | 130.7 |
| J=25, lr=0.03 | 0.00334528 | 187.8 |
| J=50, lr=0.03 | 0.00273655 | 271.8 |

- Threshold = 0.5 × default = **0.00281770**; J=50/lr=0.01 qualifies ⇒ `adopted: true`; `projection_seconds_per_example` = **34.127** s.
- Plateau: J=25→50 improvement **34.31 %** ≥ 10 % ⇒ **`plateau: "recipe-limited"`**.
- **Adoption was actually applied**, unlike exp_04's: `run_report.json` → `"recipe": {"inner_iters": 50, "lr": 0.01}`, and the shard provenance header `…/k1/capacity/b1/shard_00000/header.json` → `"optimization_config": {"inner_iters": 50, "lr": 0.01}`. The mechanism is `run_wan_pos_inversion.sh:318` (`pos_adequacy_uri="${POS_ADEQUACY_URI}"`) plus `submit_k1.sh`'s discovery step — **[REV3]** the wiring exp_04's launcher **lacked at J1-4** and **gained in the `adequacy-wiring` fix (`a520e9d`) that J1-5 ran on**. Plan v3's inherited clause "*G1/G2 are evaluated once, on the adopted recipe only*" is **satisfied here**, and — as of J1-5 — in exp_04 as well.

### 4.5 L_pos ablation (diagnostic-only)

Arm b1, adopted recipe {50, 0.01}, `diagnostic_only: true`, `published_l_pos: 8`:

| L_pos | final tracking loss | note |
|---|---|---|
| **1** | 0.01316775 | matches the PyTorch fork's L_pos=1 prior art |
| **8** | 0.00238379 | **5.52× better**; L=8 fixed for K2/K3 regardless (plan) |

### 4.6 Smoke phase (n = 2, recorded for completeness — **not a scientific result**)

`…/k1/smoke`, recipe {10, 0.01}, 1,421.3 s, `declared: 2`. Arms: B0 0.1270, B1 0.9358, B1-probe 0.3340,
B2-0 0.1273, B2 0.5643, B2-probe 0.4428; `target: "stop"`. CI-based gates at n=2 fail near-tautologically; this
phase exists to prove the pipeline publishes, and it did.

---

## 5. Coverage, integrity, and validation state

| Property | Evidence |
|---|---|
| Cohort coverage | `coverage_ok: true`, `missing_names`/`extra_names`/`duplicate_names` all empty on every gate, both cohorts |
| Invalid pairs | `invalid_pairs: 0`, `invalid_fraction: 0.0`, `imputed_method_ssim: 0` — **no imputation was exercised** |
| Quarantines | `"quarantined": {}` in both `run_report.json` files |
| k-sets | H1/H2 at `k_set: [0]`; probes at `[0,1,2]` — as predeclared |
| Bootstrap | 10,000 resamples, seed 20260804, every gate |
| Cohort identity with exp_04 | DEV-64 `manifest_hash` **`433f8691…3f76fa`** and TRAINFIT-16 **`b4e62fc9…45abb66`** are **byte-identical to exp_04's** — the two experiments genuinely ran on the same clips |
| Parity audit | Recorded CLEAN 2026-08-05T20:00Z: exp_04's audit inherited for every imported primitive, plus 11 positive-slot deltas audited line-by-line against `embedding_search.py` @ `f370228`, plus the numeric-defaults and data-parity cross-checks |
| Test suite | 1,417 passing at the S9 tip (219 exp_05-specific over S1–S5, growing through S10a/S6–S9) |
| Wall times | capacity DEV 10,467.0 s; capacity TRAINFIT 2,765.5 s; adequacy 1,821.8 s; smoke 1,421.3 s |

**Three load-bearing implementation contracts established and pinned during S1–S4** (each would have silently
corrupted the result if missed):
1. **The bf16 activation cast is not a no-op** (`side_adapter_wan.py:767`, max |Δ| 3.1e-2 measured). Single-owner rule: the runner-built `casting_velocity_fn` casts **both** CFG branches; operators pass fp32 unchanged. Proven by an S3 bitwise oracle (equal with the cast, **unequal without**) and mutants R3a/R3b.
2. **fp16 → bf16 double-rounding is NOT value-preserving** — the plan's F7 premise was measured **false**: ≈ 6.25–6.27 % of latent-like N(0,1) elements diverge by one bf16 ulp (larger for fp16-subnormals). The fidelity gate therefore uses **feature tolerances**, and selects fp32 conservatively absent feature deltas. Storage restated exactly: 7,468,516 payload bytes/record fp16 ⇒ **14.80 GiB** cohort.
3. **Positive-slot physics differ from the null slot's** — C-gradients are **nonzero at w=1** (the exact inversion of exp_04's zero-∅-gradient pin), verified on a real tiny WanModel at both guidance scales.

---

## 6. What the STOP did and did not stop

| | Status |
|---|---|
| K2 target caching (TRAIN-2000 + DEV-64 + TEST-64, ~14.80 GiB fp16) | **STOPPED** — Yixun, verbatim: "Honor the STOP for K2, and GO for J1b" (2026-08-06 ~20:30Z) |
| K3 regression training / K4 eval | **STOPPED** (downstream of K2's cache) |
| TEST-64 | **NEVER TOUCHED** — correct: selection stopped at P1′ and TEST is not a tuning set (predeclared) |
| S1–S8 + S10a code | Built, reviewed, committed, green |
| S9 | Held open at the R14/R15 matrix stall — exp_04's `generate_wan_null_adapter.py` was gated on exp_04's own P1 outcome and never authored. A repo-rooted **tripwire test fails loudly** if that file ever appears |

---

## 7. Artifact index

| Artifact | URI | Digest / key provenance |
|---|---|---|
| DEV-64 capacity | `gs://v6_east1d/datasets/droid_wan_pos_context/k1/capacity/` — `selection.json`, `gate_tables.json`, `run_report.json`, `b1/`, `b2/` | `manifest_hash 433f8691…3f76fa` |
| TRAINFIT-16 capacity | `…/k1/capacity_trainfit/` (same set) | `manifest_hash b4e62fc9…45abb66` |
| Adequacy | `…/k1/adequacy/adequacy_report.json` | consumed by **both** capacity phases |
| Smoke | `…/k1/smoke/` | n=2, retained for the record |
| Common | `code_sha bb845eaf43de2f05abbf99dfd4344a51bc2930bc`; model revision `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff7315c768468a5333511427288870b2e9635`; `base_context_fingerprint 6c79000e…373b`; `guide_scale 5.0`; `l_pos 8`; `embedding_slot "positive"`; `dtype_policy fp16`; 25-step σ grid, shift 5.0, σ₀ = 1.0 | |
| **[REV3]** Comparison videos | `gs://v6_east1d/datasets/droid_wan_pos_context/k1/videos_att-0809-173808/` — **24 mp4s** (B1, B2, B1-probe k=0 over 8 clips) + `videos_report.json`, from the post-STOP capacity-videos job `20260809-173808-13c3cadc-capvideos-yixun` (v6e-8, SUCCEEDED attempt 2). K1 itself published none — deferred by design, ruled not K1-blocking. **Cross-check CLEAN:** recomputed per-clip future-SSIM matches the published `gate_tables.json` within fp16-storage tolerance (max \|Δ\| 0.052 on b2; means +0.002–0.006); 8-clip means **b1 0.923, b2 0.178, b1_probe_k0 0.439**. Provenance binds `bb845ea` (K1-2) |
| **[REV3] [DEVIATION — RECORD]** rendered subset | the **8 lowest-`ordinal` DEV clips**, not a first-8-by-manifest-row subset — `subset_records` sorts by the record's `ordinal`, a global dataset ordinal rather than row position. All 8 are valid DEV-64 clips and the three arms are same-clip comparable with each other. A row-order re-render is a cheap optional follow-up |
| **[REV3]** HTML reports (SOP artifact 12) | `pos_context_01-capacity-gates_results.html` and `pos_context_02-video-gallery_results.html` (+ `…_02…_assets/`, 24 mp4s) — **committed in-repo** |

---

## 8. Numbers corrected against the artifacts

| Where | Said | Artifact says |
|---|---|---|
| Worklog K1 reading, tracker, CLAUDE.md | frozen-context control "~0.25"; B1 "~3.6×" it | **B0 = 0.3215**; **B1 = 2.87× B0** — §4.1 |
| Worklog K1 reading | probe "collapse to ~0.5" | 0.5254 (DEV) / 0.4885 (TRAINFIT) ✓ — but see the analysis on "collapse" as a word for a 57 % retention |
| Worklog K1 reading | adequacy "1822s" | 1,821.792 s ✓ |

Every other figure in the worklog's K1 reading — H1 0.9227 [0.9129, 0.9314] and 0.9095 [0.8839, 0.9270];
H2 0.1610 [0.1358, 0.1868] and 0.1570 [0.1076, 0.2082]; median ratios 28.6 / 26.0 / 0.254 / 0.196;
`frac_improved` 1.00 / 1.00 / 0.00 / 0.00; probes 0.5254 (rel 0.569) and 0.4885 (rel 0.537); the 3.13× multiple
over 0.2946 — reproduces **exactly**.

---

## 9. Caveats that bound every number in this document

1. **The 0.2946 anchor is not a quality baseline.** The deployed `pre_context` adapter's rollout SSIM 0.2946 comes from a **4-sample** validation at step 30,000, and those four samples are **four correlated windows of a single episode**. It is a wiring/sanity check. The "3.13×" multiple in §4.1 inherits that weakness entirely and must never be quoted as a controlled comparison.
2. **B1 is an ORACLE, not a deployable system.** It optimizes a per-clip context against that clip's own ground-truth latents. It bounds what the 8-token channel can express; it says nothing directly about what an amortized, action-conditioned emitter can achieve.
3. **The pivots are exp_05's own.** No B0/A0 or B2-0/A2-0 artifact is shared with exp_04 (8-token vs 512-row inversion contexts). Cross-experiment tables are descriptive, under the four constraints in §4.3.
4. **H1's control is active-CFG** (B0), exp_04's is CFG-collapsed (A0) — H1's median **MSE** ratio of 28.6× and **[REV3] exp_04's authoritative J=50 G1 median MSE ratio of 4.681×** measure different things. *(exp_04's historical J=10 G1 ratio was 3.605×. Neither is related to the "~3.6×" SSIM multiple corrected in §4.1, which was wrong and is now 2.87× — those figures collided numerically by coincidence.)*
5. **One cohort, one dataset, one backbone, one guidance scale (w = 5), one L_pos.** DEV-64/TRAINFIT-16 of DROID at 192×320×32f through Wan2.2 TI2V 5B. TEST-64 untouched.
6. **[REV3 — largely resolved] Videos and HTML reports now exist.** The capacity-videos job rendered B1/B2/B1-probe k=0 (24 mp4s) with a **CLEAN** numeric cross-check, and both HTML pages are committed. **Residual caveats:** the rendered subset is the **8 lowest-`ordinal`** DEV clips (a recorded deviation), and **no systematic human qualitative review has been recorded** — so the risk that a systematic visual artifact hides behind these means is reduced, not eliminated.
7. **σ₀ = 1.0** here vs **0.999** in the PyTorch reference (ratified deviation): prior-art numbers are directional context, not matched comparisons.
8. **The K1-1 failure means the smoke rung's first pass never ran on the K1-2 code path at n>2 before the full job** — in practice K1-2 re-ran all four phases from scratch, so this is recorded for completeness rather than as an open risk.
