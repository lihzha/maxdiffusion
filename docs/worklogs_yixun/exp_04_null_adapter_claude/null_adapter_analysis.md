# null_adapter_analysis — exp_04 P4 (SOP artifact 11)

**Status: DRAFT — revision 3.** Revision 2 answered the Codex analysis review
(`null_adapter_codex_analysis_review.md`, REQUEST-REVISION); **revision 3 folds in the completed J1-5 clean-gate
re-run**. Every interpretive step is labelled **[FINDING]** (artifact-backed), **[INFERENCE]** (a reading the
artifacts support but do not compel), or **[HYPOTHESIS]** (a mechanism story *not* tested here); **[REV2]** /
**[REV3]** mark edits by the round that made them. Numbers are not restated except where the argument turns on
them — the record is `null_adapter_results.md`.

> **STATE OF THE TWO OVERTURNED CONCLUSIONS FROM REVISION 1:**
> **(1) RESOLVED BY MEASUREMENT [REV3].** Revision 1's over-determination argument was struck and the selection
> ruled INDETERMINATE. **J1-5 re-ran both cohorts at the adopted J=50 and the predeclared STOP was retained**
> (results §4.5) — so the formal verdict is now *measured*, not argued. The withdrawn argument stays on the
> record in §4.1; it was wrong about the logic, and the conclusion it wrongly defended turned out to hold.
> **(2) STILL OPEN.** The objective-shape attribution **remains compute-confounded**; a budget-matched greedy
> probe is required for causal attribution (§3.2). J1-5 supplies a *partially* relevant new data point — flagged
> in §3.2 for the next review, **without changing that section's hedged conclusion**.

---

## 1. The verdict in one paragraph **[REV3 — updated for the measured outcome]**

exp_04 asked whether per-step null embeddings recovered by inversion can drive the frozen Wan2.2 TI2V 5B to
reconstruct DROID futures, and whether those embeddings are stable enough to become regression targets for an
amortized adapter. **The answers, now measured at the adopted J=50 recipe, are a qualified yes and a clear no.**
Optimized nulls reach **0.8868** own-basin SSIM — clearing G1's absolute conditions — while G1's *ratio* conjunct
against a strong CFG-collapsed control failed (**4.681** vs 5); fresh-noise optimization reached **0.6638**
against a 0.75 bar with CI-low 0.6312 against 0.70; and locked nulls under foreign noise reached **0.1666**
against a 0.70 floor, short by a factor of 4.2. **`target = STOP`, on DEV-64 and TRAINFIT-16 alike.**

**This is the predeclared verdict, and it is clean.** Revision 2 had to record the selection as INDETERMINATE:
plan v5 required the gates to run at the adopted J=50, and a launcher gap (issue #15) meant J1-4 ran them at
J=10, leaving both branches of the selection rule unmeasured at the right recipe. **J1-5 closed that** — both
branches were recomputed at J=50, both fail, and the two attempts of that job reproduced every gate number
**byte-for-byte**. The STOP does not hinge on G1's near-miss: A1 is independently barred by its probe missing
the absolute floor by 4.2×, and G2 fails both SSIM clauses. TRAINFIT-16 shows the same signature, so this is
not a dev-set artifact and there is no memorization gap (in-basin 0.8847 vs 0.8868).

**Five-fold more optimization did not change the verdict — it sharpened the picture.** In-basin capacity rose
(0.8523 → 0.8868); fresh-noise capacity rose a lot (0.4973 → 0.6638) but landed *level with the do-nothing
locked-basin control* rather than past it (§2.4); and **transfer did not move at all** (0.1729 → 0.1666). The
basin-specificity of greedy per-clip nulls is **budget-independent** over this range — which is the most
durable single fact exp_04 produced about the target family it set out to build.

Two conditional follow-ons produced the more interesting result. Replacing per-step tracking with joint
optimization through the differentiable 25-step rollout — **at roughly 19× the per-example compute** — cut
fresh-noise endpoint MSE from 0.429 to 0.273 and lifted foreign-basin SSIM from 0.163 to 0.475, about 2.9×.
**Joint endpoint optimization at substantially greater compute produced better n=8 capacity and transfer;
objective shape is a promising explanation, but a budget-matched greedy probe is required for causal
attribution.** Every aggregate setting mean still missed the 0.70 floor. **The experiment's durable contribution
is a well-supported research direction — through-the-sampler objectives — not a null-embedding adapter and not
yet a demonstrated mechanism.**

---

## 2. What each gate outcome actually means

### 2.1 G1's absolute conditions passed; only its ratio against a strong control failed [FINDING → INFERENCE] **[REV2 — retitled and rescoped]**

**[REV2]** Revision 1 titled this "G1 failed on its control", which implies a defective control. **A0 behaved
exactly as designed.** The accurate statement is:

> **G1's absolute A1 conditions passed; only its ratio against the strong, CFG-collapsed A0 control failed.**

Concretely, **[REV3] at the adopted J=50 recipe**: median `MSE(A0)/MSE(A1)` = **4.681** against a bar of 5,
while A1's absolute conjuncts both passed (**0.8868** ≥ 0.80; CI-low **0.8711** ≥ 0.75). *(At J=10 the same
clauses read 3.605 / 0.8523 / 0.8327 — issue #15's numbers, results §4.1.)* **[REV2]** Revision 1 also claimed
the strong control left "only ~3× of headroom" — **withdrawn**: MSE ratios have no fixed headroom, and the
arithmetic does not bound what a better-optimized A1 could achieve. **[REV3]** That withdrawal was well taken:
J=50 moved the ratio from 3.605 to 4.681, a 30 % gain no headroom argument would have predicted — and **still a
FAIL**.

Why is A0 strong? [INFERENCE, mechanically grounded in the plan's own parity register] With **base** null rows in
the 512-token context the two CFG branches are *identical*, so `v = v_unc + w(v_cond − v_unc)` degenerates to
`v = v_unc`: A0's dynamics become independent of `w` and match the single-context inversion dynamics — and the
trajectory it replays was itself computed at w = 1. "Near-self-consistent replay" is a reasonable inference, and
the reviewer confirmed the algebra. The consequence to carry forward is interpretive, not critical of the design:
**a ratio-form gate measured against a control this strong is a different and harder test than the same ratio
measured against a weak control** — which is exactly why exp_04's **4.681×** and exp_05's 28.6× cannot be
compared (§5.2), **[REV3]** even now that the two are budget-matched. Reading "G1 FAIL" as "the null channel is
weak" remains a misreading.

### 2.2 G2 is the honest measurement of the null channel's fresh-noise reach [FINDING]

G2 compares like with like: both arms start from the same ε₀ and both replay at w = 5. Here the null channel does
real work — **[REV3] at J=50, a 17.8× median MSE reduction, 64/64 examples improved, 4.743 → 0.303 MSE** — and
still lands at **0.6638** SSIM against a 0.75 bar, CI-low **0.6312** against 0.70. *(At J=10: 10.2×, 4.743 →
0.490, 0.4973.)* **Direction without magnitude** remains the accurate summary, and J=50 sharpened it rather than
overturning it: five-fold more optimization closed roughly two-thirds of the gap to the bar and stopped —
**the channel reliably steers a fresh basin toward the target and reliably fails to arrive.** Where it stops is
the subject of §2.4.

### 2.3 The transfer result is what drove the discretionary decision [FINDING] **[REV2 — retitled]**

A1-probe = 0.1729 against a 0.70 floor — short by a factor of four, i.e. **very low absolute quality**. Per the
standing R5 note this is a **failure of the absolute floor**; the relative conjunct is logically redundant on
[0,1] SSIM and is never the binding constraint.

**[REV2]** Revision 1 said the selection was "over-determined" by this floor. **That is withdrawn** — A1-probe
replays A1's own nulls, so it is not budget-independent, and the A2 fallback branch is unmeasured (§4.1). The
probe is what *motivated* the discretionary decision not to advance; it does not *formally settle* the selection.

**[REV3] Measured at J=50, the probe barely moved: 0.1729 → 0.1666** (TRAINFIT 0.1677), against the same 0.70
floor. The reviewer was right that the number was *not guaranteed* to be budget-independent — that was a
logical point about what the J=10 artifacts could support — and the measurement has now shown that in fact it
**is** budget-flat over a 5× range. **[FINDING]** Greedy per-clip nulls do not become more transferable with
more iterations; if anything they drift very slightly the other way. That converts revision 1's assumption into
an observation, which is the right order.

**[REV2] The comparison against doing nothing, stated at the precision the artifacts allow.** The nearest
reference is A2-0 (base nulls, fresh ε₀) at **0.1423**, so A1-probe is **+0.031 above** it. But A1-probe replays
from `keyed(k)` while A2-0 replays from `global(0)`, and **no base-null `keyed(k)` control arm exists**. So:

> A1-probe has very low absolute quality and is +0.031 above the nearest **unmatched** do-nothing proxy; its
> **matched incremental benefit or harm was not measured.**

Revision 1 called the tensor "very nearly *inert*" outside its basin. **That is too categorical** and is
withdrawn: inertness is a claim about a matched incremental effect, and the matched control was never run.

**[DEVIATION — RECORD]** The worklog and tracker say instead that these nulls are "actively destructive, WORSE
than doing nothing". **The artifacts do not support that** (0.1729 > 0.1423); the phrase belongs to exp_05's B2
arm, where it is gate-verified (0.1610 vs 0.2814, 0/64 improved), and appears to have been transposed across the
joint reading. The distinction matters for §5.1: *destructive* and *low-quality-but-not-measurably-harmful* are
different geometries, and the opposite-geometries argument depends on keeping them apart.

### 2.4 **[REV3]** A2 at J=50 converged **to** doing-nothing, not past it [FINDING → INFERENCE]

The largest movement anywhere in the study is A2's: **0.4973 → 0.6638** when the optimization budget went 5×.
Where it landed is the interesting part. **A0 — the locked-basin control that does no optimization at all —
scores 0.6665.** A2 arrived at 0.6638, i.e. **0.0027 below it.**

Two things must be kept apart here, and revision 3 keeps them apart deliberately:

- **The gate comparison is matched and A2 wins it decisively.** A2's predeclared control is **A2-0** (0.1423, same fresh ε₀). Against it A2 passes both relative clauses — median MSE ratio **17.8**, **64/64** improved. Fresh-noise null optimization unambiguously does a great deal of work.
- **The A0 comparison is cross-arm and unmatched**, and is descriptive only: A0 replays from its own inversion endpoint with CFG collapsed, A2 from fresh ε₀. It is not a gate statistic and no verdict depends on it (results §4.5.3).

With that caveat stated, the coincidence is worth recording. **[INFERENCE]** Five-fold more greedy optimization
from a fresh basin bought a large improvement that terminated almost exactly at the quality obtainable by not
optimizing at all in the *right* basin — and still **0.086 short of the 0.75 G2 bar**, with in-basin A1 a
further 0.22 above it. Combined with the flat probe (§2.3), the shape of the J=10 → J=50 movement is:
**greedy per-clip optimization converts budget into in-basin quality, and converts none of it into transfer.**

**[HYPOTHESIS — untested]** If the greedy per-step objective's reachable set from a foreign basin is bounded by
something like "recover the pivot trajectory's own dynamics", then A0 is exactly that bound, and more iterations
approach it asymptotically without crossing it. Nothing in this experiment tests that; it would need a budget
sweep (J = 100, 200) to see whether A2 plateaus at A0 or eventually passes it.

---

## 3. Joint endpoint optimization outperformed greedy — at ~19× the compute **[REV2 — retitled]**

**[REV2]** Revision 1 titled this "The real result: greedy is the artifact". That attributes the effect to
objective shape, which the design does not isolate. The supported headline is the one below.

### 3.1 What J1b and J1c measured [FINDING]

J1b and J1c hold the channel (16 null rows), the backbone, the guidance scale, the clips and the noise fixed,
and change the objective — from *per-step tracking of the inversion pivots* to *joint optimization of the rollout
endpoint through the differentiable sampler* — **and, unavoidably, the compute budget with it** (§3.2):

| Axis | greedy per-step (J=10) | joint endpoint (300 iters) | change |
|---|---|---|---|
| Fresh-noise capacity (mean endpoint MSE, same 8 clips) | 0.4294 (A2) | **0.2731** (A3) | 1.57×; **0/8 beat their paired A1**, 3/8 fall inside the pooled A1 range |
| Foreign-basin transfer (mean SSIM, same 8 clips) | 0.1633 (A1-probe) | **0.4753** (J1c keyed) | **2.91×** |

> **[REV2] The supported statement, verbatim from the review's prescribed wording:**
> *Joint endpoint optimization at substantially greater compute produced better n=8 capacity and transfer;
> objective shape is a promising explanation, but a budget-matched greedy probe is required for causal
> attribution.*

Both of the greedy arm's headline weaknesses — "can't get there from fresh noise" and "doesn't survive a
different noise draw" — moved substantially. **What moved them is not established.**

### 3.2 The compute confound, which is NOT eliminated on either axis **[REV2 — conclusion reversed]**

**J1b changed the budget as well as the objective.** Greedy A2 ran 25 steps × J=10 = 250 cheap single-step inner
iterations at ~21.2 s/example (adequacy J=10 cell). A3 ran 300 iterations of a *full 25-step remat'd rollout* at
~405 s/example (3,237 s ÷ 8) — roughly **19× the per-example optimization compute** of the arm it beat, and about
10× the adopted J=50 recipe. The adequacy probe's plateau verdict was **"recipe-limited"**: more greedy
iterations were still buying accuracy at the grid's edge. **The capacity half of §3.1 is therefore confounded**,
and no arm in this experiment separates budget from objective.

**[REV2] Revision 1 then argued the transfer half escapes the confound, on the reasoning that a
better-optimized greedy null fits its own pivots more tightly and should therefore transfer *worse*, so the
confound runs against the observed effect. The reviewer rejected that, and I accept the rejection.** The
counter-mechanism is straightforward and I did not consider it: **more greedy iterations might first learn
transferable, shared corrections before beginning to overfit basin-specific detail.** Under that trajectory,
extra greedy compute would *improve* transfer over the J=10 measurement, and the observed 0.163 → 0.475 gap
would shrink for reasons that have nothing to do with objective shape. Nothing in the artifacts distinguishes
the two trajectories, because greedy transfer was only ever measured at one budget.

**Standing conclusion:** J1c is *cleaner* evidence about transfer than J1b is about capacity — it at least
compares locked tensors under identical replay conditions — but **it is not causal either.** The proposed sign
of the confound is **plausible, not established**. Any future write-up should lead with transfer *and* state the
confound in the same breath.

> **[REV3] NEW DATA BEARING ON THIS SECTION — recorded, conclusion deliberately NOT changed.**
> J1-5 measured both greedy probes at **J=50**, which is a partial instance of the budget-matched greedy probe
> this section asks for. Both went **down**, not up:
> **A2-probe** (greedy from ε₀, replayed foreign — the arm *matched in starting basin* to J1c's joint nulls)
> **0.2958 → 0.2510**; **A1-probe** 0.1729 → 0.1666. J1c's joint nulls from the same ε₀ score **0.4753**.
> On its face this runs **against** the reviewer's counter-mechanism (that extra greedy compute would first buy
> transferable shared corrections), and it narrows the compute gap from ~19× to ~10×.
> **It is not a clean resolution and I am not treating it as one:** J=50 greedy (~40.2 s/example) is still
> ~10× cheaper than A3 (~405 s/example), so the probe is *closer to* but not *at* budget parity; and two points
> (J=10, J=50) cannot establish that greedy transfer is monotone in budget.
> **The hedged conclusion above stands unchanged.** This is flagged for the next analysis review as the first
> item in §8, together with the question of whether a J=200-greedy probe would close the gap outright.

**The experiment that would settle it, and was never run:** the greedy arm at the adopted J=50 (or higher) on the
same 8 clips, re-probed for transfer — a **budget-matched greedy probe**. **[JUDGMENT]** ~1 v6e-8-hour, and it is
the difference between a research direction and a demonstrated mechanism. It is now priority 2 in §7, and the
approved J=50 clean-gate re-run (§4.1) is a natural vehicle for it.

### 3.3 Transfer tracks *absolute* quality, not *retention* [FINDING]

Across the 8 clips, own-basin SSIM predicts absolute foreign-basin SSIM almost perfectly (Pearson +0.945,
Spearman +0.976) but predicts the *retained fraction* only weakly (+0.623 / +0.714). The defensible statement is
"**better-optimized clips end up better everywhere**", not "better-optimized clips lose proportionally less".
The worklog's phrasing ("transfer quality tracks optimization quality") is true of the first reading and
overstated for the second, and the retention figure itself is estimator-dependent (0.700 / 0.724 / 0.730
depending on how you average — results §6.2). **[REV2]** These correlations are n = 8 with no CIs and are
descriptive only. **None of this changes the picture**: at 0.475 foreign and 0.651 own-basin, **every aggregate
setting mean missed the 0.70 floor**, so the joint optimum is not deployment-grade at 300 iterations either.

---

## 4. Threats to validity — stated against ourselves

### 4.1 The gates ran at a recipe the experiment's own probe had rejected [THE BIGGEST ONE]

Results §4.4: the adequacy probe adopted J=50/lr=0.01 (2.57× better tracking loss than the default), the plan
requires G1/G2 to be evaluated *only* on the adopted recipe, and the capacity run executed at the **default
J=10** because `bash_scripts/run_wan_null_inversion.sh` never passes `null_adequacy_uri`. The re-run was
affordable (1.43 h against a 2 h budget); this was a plumbing gap, not a budget stop.

**[REV2] How much does it hurt? — revision 1's answer was wrong on the decisive point.**

Revision 1 concluded "the selection verdict is unaffected, because it is over-determined by the A1-probe floor",
and flagged it as its own most attackable claim. **The reviewer attacked it and it does not survive.** Three
independent reasons, all of which I accept:

1. **A1-probe is not budget-independent.** It replays *A1's* nulls. Recompute A1 at J=50 and you get different nulls, hence a different probe. 0.1729 characterises the J=10 nulls, not the arm.
2. **The A2 fallback is unmeasured.** The selection rule's second branch asks whether G2 passes. There is no J=50 G2 measurement, so that branch is simply open.
3. **No monotonicity is available.** Revision 1's supporting claim — "a stronger optimizer could only have raised A2" — **is withdrawn.** Tracking loss and decoded SSIM are not guaranteed to move together, in either direction, so I cannot even sign the effect of the missing re-run.

**Consequence, and the formal state of the experiment *as it stood at revision 2*:**

> The artifacts genuinely say `target="stop"` at J=10, but the **plan-compliant target selection is
> INDETERMINATE.** The decision not to advance to P2/P3 was **discretionary**, taken on the observed J=10 result.

### **[REV3] RESOLVED — how the re-run answered each open point**

J1-5 re-ran both cohorts at the adopted J=50 with the launcher fix live. **`target = STOP` on both**
(results §4.5). Point by point against what revision 2 said was unmeasurable:

| Revision 2's open point | What J1-5 measured | Outcome |
|---|---|---|
| A1-probe is budget-dependent; 0.1729 characterises the J=10 nulls only | A J=50 A1 (0.8868) with its own probe | **0.1666** — barely moved; still 4.2× short of the 0.70 floor. A1 remains independently barred |
| The A2 fallback branch is unmeasured at J=50 | G2 at J=50 | **0.6638**, CI-low **0.6312** — fails both SSIM clauses (bars 0.75 / 0.70). Branch closed |
| No monotonicity available, so the effect of the missing re-run cannot even be signed | Every arm at both budgets | Movements were **large and of mixed sign** (A1 +0.035, A2 +0.166, A1-probe −0.006, A2-probe −0.045) — vindicating the refusal to sign them in advance |
| G1's FAIL is "materially in doubt" (3.605 vs 5) | G1 at J=50 | **4.681 vs 5** — closer, but **still a FAIL**. Revision 2 was right that this was in doubt, and the doubt resolved against a pass |

**[REV3] The withdrawn argument stays withdrawn, and this is the important nuance.** Revision 1 claimed the
selection was over-determined by the probe floor; the reviewer struck it as special pleading; **J1-5 has now
shown the *conclusion* it defended was correct while confirming the *reasoning* was not.** The probe did turn
out to be budget-flat — but that was an empirical fact nobody had measured, not something the J=10 artifacts
entitled anyone to assert. **Being right for unavailable reasons is still being wrong about the evidence**, and
the record keeps both halves.

**[REV3] What the re-run also strengthened, beyond closing the deviation:**
- **Robustness:** the STOP does not rest on G1's near-miss. A1 is barred by a probe missing its floor by **4.2×**, and G2 fails **both** SSIM clauses. Every non-STOP path fails a clause that is nowhere near its bar.
- **Reproducibility:** two attempts of J1-5 produced **byte-equal** `g1`/`g2`/`selection` number blocks, and the deterministic controls A0/A2-0 are **per-example bitwise identical** to J1-4's — exact cross-run comparability, three days and two code tips apart.
- **Cohort generality:** TRAINFIT-16 reproduces the failure signature exactly, with in-basin capacity 0.8847 vs DEV's 0.8868 — **no memorization gap**, so the STOP is not a dev-set artifact.
- **Cross-experiment comparability [REV3]:** exp_04's A1 is now **0.8868 @ J=50** against exp_05's B1 **0.9227 @ J=50** — the budget mismatch of §5.2(b) is **gone**, though the control, representation and pivot mismatches all remain, so the comparison stays descriptive.

**Process reading.** The R10 follow-up review installed a fail-closed guard against precisely this
failure — its docstring names it verbatim — but the guard covers "URI set and unparseable", not "URI empty". The
sibling experiment's launcher wires the URI and honored its adoption. **This is the CLAUDE.md fix-propagation
lesson running in reverse**: exp_05 built the capability, exp_04 never received it, and nothing in either
experiment's closing record noticed. A one-line launcher addition plus a "capacity refuses to start when an
adequacy artifact exists at the conventional path and no URI was passed" assertion would close it.

### 4.2 Sample sizes and metric coverage

J1b and J1c are **n = 8** — the first eight DEV clips, no CIs, no multiplicity control, and the §3.3 correlations
are descriptive. J1b reports **latent MSE only**; SSIM enters only at J1c.

**[REV2]** Revision 1's "3/8 clips reach own-basin quality" **overstated the comparison** and is corrected in
results §5.2: **none of the eight A3 endpoints matches or beats its *paired* A1 MSE (0/8)**. Three fall inside
the **pooled cross-clip** A1 range, which measures a clip's joint endpoint against the best *other* clips'
greedy own-basin results — a much weaker claim. One of those three (ep45499) is also an outlier the greedy arm
nearly solved (A2 = 0.037). The capacity story is therefore weaker than revision 1 presented it, independently
of the compute confound in §3.2.

### 4.3 Oracles are not systems

A1/A2/A3 all optimize conditioning **per clip against that clip's own ground-truth latents**. They bound what
the channel can express. They are *not* evidence that any amortized predictor can find those tensors from
`(z_i0, actions)`. The planned P3 predictor (~9M params) was never built or trained, so exp_04 has **zero**
evidence on the amortization question it was designed to reach.

### 4.4 The 0.2946 anchor cannot carry the weight often placed on it

The deployed `pre_context` adapter's 0.2946 rollout SSIM is a **4-sample** validation, and those four samples are
**four correlated windows of one episode**. Statements of the form "the adapter sits at ⅓ of the demonstrated
oracle" divide a 64-clip oracle by a 4-window single-episode wiring check. The *qualitative* claim (the deployed
adapter is far below what the channel can express) survives; the *ratio* should not be quoted as a measurement.

### 4.5 Narrowness

One dataset, one backbone, one resolution, one guidance scale (w = 5), one L_null (16), one σ grid (σ₀ = 1.0 vs
the reference's 0.999 — a ratified deviation, so prior-art numbers are directional only). TEST-64 was never
touched, which is correct but means every number here is DEV/TRAINFIT.

### 4.6 What went right, so the reader can calibrate

Coverage was complete, invalid pairs were zero (the imputation machinery was never exercised — it is untested in
production), quarantines were zero, the parity audit was clean across 11 components, the gate statistics were
frozen as code before any data existed, and every gate verdict was computed by that code rather than argued.
Three separate runbook failures cost wall-clock but **corrupted nothing**, because the storage layer refused
every unsafe write. The failure modes this campaign hit were orchestration failures, not evidence failures.

---

## 5. The joint reading with exp_05

### 5.1 The two experiments' fresh-noise arms moved in opposite directions [FINDING — observation only] **[REV2 — retitled and de-causalised]**

From a **fresh** noise basin, on the **same 64 clips**, with the **same manifest hash** — **and, since J1-5, at
the same adopted recipe**, but still in different conditioning representations, against differently-constructed
controls, and on non-shared pivots (mismatches (a), (c) and (d) of §5.2):

**[REV3] Now recipe-matched.** Revision 2 had to note that the two rows ran at different budgets. J1-5 supplies
the null row at **J=50**, so both rows are now at the **same adopted recipe** — mismatch (b) of §5.2 is gone.
The J=10 row is retained beneath for continuity.

| | recipe | do-nothing control (MSE) | after per-step greedy optimization | effect | clips improved |
|---|---|---|---|---|---|
| **Null slot** (exp_04, A2-0 → A2) **[REV3]** | **J=50** (adopted) | 4.743 | **0.303** | **15.7× better** | 64/64 |
| **Positive slot** (exp_05, B2-0 → B2) | **J=50** (adopted) | 1.418 | 5.625 | **4.0× worse** | 0/64 |
| *(null slot at J=10, superseded)* | J=10 | 4.743 | 0.490 | 9.7× better | 64/64 |

Each row is internally matched — same cohort, same ε₀, each arm against its own control — so **each row is a
sound finding on its own.** The positive slot's do-nothing control is 3.3× better than the null slot's, and
greedy optimization inverts the ordering. **[REV3]** At the matched J=50 recipe the contrast is *sharper* than
revision 2 recorded: the null slot improves **15.7×** while the positive slot degrades **4.0×**. In-basin the
picture inverts once more: positive **0.9227**, null **0.8868** — a gap that narrowed from 0.070 to **0.036**
once the budgets were matched.

**[REV2]** Revision 1 concluded from this that "the positive channel is the more powerful and the more dangerous;
the null channel is weaker and better behaved." **That conclusion is withdrawn.** It attributes a
between-row difference to the *slot*, when the rows also differ in recipe (J=10 vs J=50), context construction
(16 rows inside 512 vs 8 tokens as the whole context), pivots (not shared), and control design. **Comparing the
two rows is descriptive, not a controlled slot contrast.** What survives is the pair of within-row observations
— and they are genuinely opposite in direction, which is what makes the pair worth recording at all.

### 5.2 A mechanism story for that asymmetry [HYPOTHESIS — not tested by either experiment]

Under `v = v_unc + w(v_cond − v_unc)` at w = 5, the conditional branch carries coefficient **+5** and the
unconditional **−4**, and in exp_05 the conditional is an 8-token context against a 512-row unconditional (so it
is also structurally the *lighter* tensor with the *larger* lever). A greedy per-step objective asks the channel
to jump to a pivot that, from a foreign basin, is far away — which demands a large velocity correction. **More
per-step authority applied greedily from the wrong basin produces more damage, not less.** That would explain
why the same procedure improves the low-authority null slot 9.7× and degrades the high-authority positive slot
4.0×. It is consistent with every number in both experiments, and it is **untested**: no arm in either experiment
varies w, varies L, or measures per-step step magnitude. I would not put it in a paper without the ablation.
**[REV2]** Note also that it is a story about the *slot*, and §5.1 no longer licenses a slot attribution — so
this hypothesis now rests on even less than when it was written, and would need the recipe held constant before
it could be tested at all.

### 5.3 Why exp_04's G1 FAIL and exp_05's H1 PASS are not a slot comparison [FINDING]

Their controls measure different things. exp_04's A0 collapses CFG to identity and therefore replays at the same
w = 1 the pivots were computed at (§2.1) — a near-self-consistent, strong control (MSE 0.335). exp_05's B0 keeps
CFG active (8-token conditional ≠ 512-row unconditional) and so replays w = 5 dynamics against w = 1 pivots — a
mismatched, weak control (MSE 0.890). Ratios of 3.6× and 28.6× against those two controls are simply not the
same statistic. **This was predeclared** in exp_05's plan §4 H1 interpretation note; it is repeated here because
the two numbers are the ones most likely to be quoted side by side.

**[REV2] The full mismatch list is four items, not two**, and it applies to §5.1 as well as to the H1/G1 pair:
(a) **controls** — CFG-collapsed A0 vs active-CFG B0; (b) **recipe** — exp_04 ran the **unadopted J=10 because
of its launcher deviation** (§4.1), exp_05 honored the adopted **J=50**; (c) **representation** — 16 rows inside
a 512-row context vs 8 tokens as the entire context; (d) **pivots** — not shared, so the two experiments'
inversion trajectories and targets are different objects. **Each row of §5.1's table is internally matched;
comparing the two rows remains descriptive, not a controlled slot contrast.**

**[REV3] Mismatch (b) is now REMOVED — the other three stand.** J1-5 measured exp_04 at the adopted J=50, so
A1 **0.8868** and B1 **0.9227**, and the two G1/H1 ratios, are now budget-matched. **This does not make the
comparison controlled**: (a) the controls still differ in kind — a CFG-collapsed A0 (MSE 0.335) versus an
active-CFG B0 (MSE 0.890) — which is on its own enough to make the 4.681× and 28.6× ratios different
statistics; and (c) and (d) are untouched. **The comparison remains descriptive.** Recording the narrowing
because it is real, not because it licenses anything new.

### 5.4 The joint conclusion [INFERENCE] **[REV2 — narrowed]**

Revision 1 said "static per-clip conditioning targets are dead as a training signal." **Too universal.** The
evidence rejects **the specific family that was tested**:

> The predeclared **single-basin cached-target family** — one per-clip, per-timestep context sequence produced by
> **greedy pivot tracking** — is **rejected for P2/P3 and K2/K3 under these conditions.**

What remains **untested**, and must not be swept in: **multi-noise / robust joint optimization of a static
tensor** (optimizing one tensor against several noise draws at once was never attempted in either experiment);
and a **state-conditioned emitter** that re-reads `z_t` (exp_05's K3 never ran). Both fresh-noise arms did fail
their predeclared gates, and both experiments' locked contexts degrade across basins — but that indicts *greedy,
single-basin* target construction, not every conceivable static formulation. This is the finding that stopped
exp_04's P2/P3 and exp_05's K2/K3, and it should keep them stopped; it is not a proof that no static target can
work.

---

## 6. What these results do and do not license

### 6.1 Licensed

1. **Capacity-first changes are not indicated by this evidence.** **[REV2]** More *conditioning tokens* or *deeper injection* attack a constraint that has been measured not to bind on the **output channel**: per-clip optimized conditioning drives the frozen backbone to **0.8868** (null) and **0.9227** (positive) SSIM — **[REV3]** both now measured at the same adopted J=50 recipe. **[REV2]** Revision 1 added "the deployed adapter does not come close" — **removed**, because that compares a 64-clip oracle against the 4-window single-episode 0.2946 anchor this document elsewhere disowns (§4.4). No proposal should assume output-channel capacity binds without new evidence.
2. **Through-the-sampler objectives are the most promising direction.** **[REV2]** Downgraded from "are the direction": the manipulation that improved both axes also carried ~19× the compute, so this is a well-motivated research direction, not a demonstrated mechanism (§3.2).
3. **[REV2] The oracle demonstrates backbone expressivity on these evaluated clips.** Revision 1 said "the frozen backbone is exonerated", which overreaches: reaching **0.89–0.92** SSIM *when handed per-clip optimized conditioning* **[REV3]** shows the frozen backbone can express these reconstructions on DEV/TRAINFIT clips. It does **not** establish deployable predictive sufficiency, and it is evidence against unfreezing only in the weak sense that no observed failure is attributable to the backbone.
4. **Gate-as-code and predeclaration earned their cost.** Every verdict was computed, not argued. **[REV2]** And the one place the process failed (§4.1) failed *silently in the plumbing* — which is precisely the class of error predeclared gates cannot catch, provenance-bound artifacts can surface after the fact, and a launcher-level assertion should have caught before the run.

### 6.2 NOT licensed — and one distinction that is load-bearing

1. **No claim that a null-embedding adapter would work.** P3 was never built. exp_04 has no amortization evidence.
2. **No deployment claim.** Every arm is a per-clip oracle (§4.3); every aggregate setting mean misses the 0.70 floor.
3. **No claim that "the basin problem is solved".** J1c retained ~0.70–0.73 of a 0.651 own-basin SSIM. A fraction of not-good-enough is still not good enough.
4. **[REV3] The selection verdict IS now settled: STOP, measured at the adopted recipe on both cohorts** (§4.1, results §4.5). Revision 2's "INDETERMINATE" is discharged. What is still *not* licensed is any claim that the null channel is intrinsically incapable — G1 failed on its ratio against a strong control (4.681 vs 5), with A1's absolute conditions passing.
5. **[REV2] No causal claim that objective shape is what produced J1b/J1c's gains** — a budget-matched greedy probe is required (§3.2).
6. **THE DISTINCTION THAT MATTERS.** The campaign's basin findings refute **the tested greedy single-basin cached-target family**. They do **NOT** refute the **rollout-loss family**. These are different objects, and conflating them would cancel the campaign's most promising direction:

   | | measured and refuted | **untested** — not refuted, not validated |
   |---|---|---|
   | **Target form** | a *fixed per-clip, per-timestep tensor sequence* from **greedy pivot tracking**, cached and regressed onto | a *state-conditioned emitter* re-reading `z_t` each step; **also** robust/multi-noise joint optimization of a static tensor — neither was attempted |
   | **Objective form** | *per-step greedy tracking* of inversion pivots, with no endpoint term | *through-the-sampler* optimization of the rollout endpoint — better on both axes at n=8, **at ~19× compute**, mechanism unattributed |
   | **Training signal** | teacher-forced regression onto a cache | a differentiable rollout loss computed live, no cache |

   **[REV2] The honest status line, replacing revision 1's implication that the right column is "alive":**
   rollout-loss training is **not refuted, and has an n=8 motivating oracle result** — J1c at 0.475 foreign SSIM,
   still **below the 0.70 floor**, with J1b's capacity half compute-confounded. It is **not validated** and
   carries real risk. **[JUDGMENT]** This remains the paragraph most at risk of being flattened, in either
   direction: into "the adapter line failed" or into "rollout losses work".

---

## 7. Recommended next steps

| Priority | Action | Cost | Why |
|---|---|---|---|
| **✅ DONE** **[REV3]** | ~~The approved J=50 clean-gate re-run~~ — **J1-5 ran 2026-08-09/10; STOP retained on both cohorts, reproduced byte-exactly across two attempts** | 10,582 s v6e-8 | §4.1, results §4.5. The formal outcome is measured |
| **✅ DONE** **[REV3]** | ~~The launcher fix~~ — `null_adequacy_uri` wired; adoption confirmed live end-to-end in J1-5's shard headers (`{50, 0.01}`) | — | Issue #15 closed |
| **1** **[REV3, was 2]** | A **fully budget-matched greedy transfer probe** — greedy from ε₀ at a budget matched to A3's ~405 s/example (J≈200+), replayed under `keyed{0,1,2}` | ~1–2 h v6e-8 | §3.2. J1-5's J=50 probes moved *down* (A2-probe 0.2958 → 0.2510), which points against the reviewer's counter-mechanism but does **not** settle it at ~10× remaining compute gap. Still the one measurement that would make the objective-shape claim causal |
| **2** **[REV3]** | A **budget sweep on A2** (J = 100, 200) to test whether fresh-noise greedy optimization plateaus at A0's 0.6665 or eventually passes it | ~1 h v6e-8 | §2.4 — the new hypothesis J1-5 generated. Cheap, and it would characterise the greedy objective's reachable set |
| **3** | Add a **base-null `keyed(k)` control arm** if the probe comparison is ever load-bearing | small | §2.3 — the missing matched control that leaves the probe's incremental effect unmeasurable |
| **4** | Pull the published `videos/` and write the P4 HTML report (SOP artifact 12) | host-only | Nobody has looked at what a 0.89 vs 0.66 vs 0.17 reconstruction *looks like*; the qualitative half of the evidence is unexamined |
| **—** | **exp_06 is not blocked** by any of the above — but its motivating claim (E2) should be stated as "not refuted, n=8 motivating oracle, compute-confounded", not as a demonstrated mechanism | — | §6.2 |
| **—** | **Do NOT** revive P2/P3 as planned | ~225+ v6e-8-h for A3-caching TRAIN-2000 alone | §5.4. Any revival is an **exp_07-scale new proposal** with its own gates |

---

## 8. Open questions — **answered by the review; recorded with their answers** **[REV2]**

Revision 1's five questions were all answered in `null_adapter_codex_analysis_review.md`. Recorded here so the
resolutions travel with the document:

| # | Question (revision 1) | Reviewer's answer | Where applied |
|---|---|---|---|
| 1 | Does the over-determination argument hold? | **No — "special pleading as written; formal selection remains unmeasured."** | §1, §2.3, §4.1, results §1/§4.2/§4.4 |
| 2 | Is the asymmetric-confound argument sound? | **No — "plausible hypothesis, not a valid elimination of the compute confound. J1c is cleaner but not causal."** | §3.1, §3.2 |
| 3 | Is the A0 CFG-collapse account correct? | **Algebraically correct; but "G1 failed on its control" is overstated** — A0 behaved as designed | §2.1 |
| 4 | Should headline numbers carry a J=10 annotation everywhere? | **Yes — every headline, table and terminal-status statement** | results §1, §4.1, §4.4, §9; analysis §5.1 |
| 5 | Is the static-vs-rollout distinction strong enough? | **Visually prominent but scientifically too binary** — scope "failed" to the tested greedy targets; list robust multi-noise static targets among the untested | §5.4, §6.2 |

### Open questions for the NEXT review (raised at revision 3)

1. **[REV3 — the priority question] Does J1-5's greedy-probe movement bear on §3.2's confound, and how much?** Both greedy probes fell at 5× budget (A2-probe 0.2958 → 0.2510; A1-probe 0.1729 → 0.1666) — the *opposite* of the reviewer's counter-mechanism, which posited that extra greedy compute would first buy transferable shared corrections. I deliberately did **not** touch §3.2's hedged conclusion on this basis: A3 still has ~10× more compute than J=50 greedy, and two budget points cannot establish monotonicity. **Is the hedge still right, should it be softened, or does it need the J≈200 probe first?**
2. **[REV3]** §2.4 records A2 landing 0.0027 below A0 as a cross-arm, unmatched coincidence, and offers a reachable-set hypothesis. Is even that framing too suggestive for a comparison with no matched control?
3. **[REV3]** Mismatch (b) of §5.2 is now removed (both slots at J=50), leaving (a), (c), (d). Does removing one of four mismatches change what §5.1 may say, or is the controls mismatch alone still disqualifying for any slot attribution?
4. Does §5.1 still overreach by keeping the two rows adjacent — is adjacency itself the problem, as the exp_05 reviewer judged it for H1/G1?
5. Is §6.2's "not refuted, n=8 motivating oracle, not validated" still the right status line for exp_06's motivation now that the null slot's own verdict is measured and clean?
6. **[REV3]** Should the mechanism sections (§3, §5, §6) carry an explicit banner that they rest on **J=10-era** J1b/J1c baselines, since neither follow-on was re-run at J=50?
