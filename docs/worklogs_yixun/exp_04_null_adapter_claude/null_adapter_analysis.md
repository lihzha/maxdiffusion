# null_adapter_analysis — exp_04 P4 (SOP artifact 11)

**Status: DRAFT for review.** The Codex analysis review is deferred; this document is written to be attacked.
Every interpretive step is labelled **[FINDING]** (artifact-backed), **[INFERENCE]** (a reading the artifacts
support but do not compel), or **[HYPOTHESIS]** (a mechanism story that is *not* tested by this experiment).
Numbers are not restated except where the argument turns on them — the record is `null_adapter_results.md`.

---

## 1. The verdict in one paragraph

exp_04 asked whether per-step null embeddings recovered by inversion can drive the frozen Wan2.2 TI2V 5B to
reconstruct DROID futures, and whether those embeddings are stable enough to become regression targets for an
amortized adapter. **The answer to the first question is a qualified yes and to the second an unambiguous no —
for the specific target family the plan proposed.** Optimized nulls reach 0.8523 own-basin SSIM, but the
predeclared G1 ratio bar failed (3.605 vs 5), fresh-noise optimization reached only 0.4973 against a 0.75 bar,
and locked nulls transferred to foreign noise at 0.1729 against a 0.70 floor — a mere **+0.03 SSIM above the
0.1423 do-nothing reference**, i.e. very nearly inert outside their own basin. The predeclared rule stopped the
experiment before any caching spend.
Two conditional follow-ons then found the more interesting result: **the ceiling and the basin-boundness were
both substantially artifacts of the greedy per-step objective, not of the conditioning channel.** Replacing
per-step tracking with joint optimization through the differentiable 25-step rollout cut fresh-noise endpoint
MSE from 0.429 to 0.273 (own-basin quality on 3 of 8 clips) and lifted foreign-basin SSIM from 0.163 to 0.475 —
about 2.9×. Absolute deployment floors remained unmet everywhere. **The experiment's durable contribution is not
a null-embedding adapter; it is the demonstration that the objective's shape, not the channel's capacity, is
what binds.**

---

## 2. What each gate outcome actually means

### 2.1 G1 failed on its control, not on its method [FINDING → INFERENCE]

G1 failed a single conjunct: median `MSE(A0)/MSE(A1)` = 3.605 against a bar of 5. A1's *absolute* conjuncts both
passed comfortably (0.8523 ≥ 0.80; CI-low 0.8327 ≥ 0.75). The ratio bar was missed because **A0 is an unusually
strong control** — 0.6665 SSIM / 0.3354 MSE — leaving only ~3× of headroom before the bar is arithmetically
reachable.

Why is A0 so strong? [INFERENCE, mechanically grounded in the plan's own parity register] With **base** null rows
in the 512-token context, the two CFG branches are *identical*, so the combine
`v = v_unc + w(v_cond − v_unc)` degenerates to `v = v_unc`: A0 replays at **effective w = 1**. The inversion
trajectory it replays was itself computed at **w = 1**. A0 is therefore close to a self-consistent replay of its
own pivots, and it should be good. **This is a property of the experiment's control design, not evidence about
the null channel.** The practical consequence: a ratio-form gate calibrated against a near-self-consistent
control was always going to be hard to pass, and reading "G1 FAIL" as "the null channel is weak" is a
misreading the results doc's one-line gate readings deliberately avoid.

### 2.2 G2 is the honest measurement of the null channel's fresh-noise reach [FINDING]

G2 compares like with like: both arms start from the same ε₀ and both replay at w = 5. Here the null channel
does real work — 10.2× median MSE reduction, **64/64 examples improved**, 4.743 → 0.490 MSE — and still lands at
0.4973 SSIM against a 0.75 bar. **Direction without magnitude** is the accurate summary: the channel reliably
steers a fresh basin toward the target and reliably fails to arrive.

### 2.3 The transfer collapse is the finding that actually stopped the experiment [FINDING]

A1-probe = 0.1729 against a 0.70 floor — short by a factor of four. The selection verdict is
**over-determined**: even had G1 passed, the probe floors alone would have stopped it. Per the standing R5 note
this is a **failure of the absolute floor**; the relative conjunct is logically redundant on [0,1] SSIM and is
never the binding constraint.

The sharpest available framing is a comparison against doing nothing. The nearest measured reference is A2-0
(base nulls, fresh ε₀) at **0.1423**, so the locked nulls are **+0.031 above** it. **[FINDING]** An entire
per-step optimization pass, transplanted to a different noise draw, is worth about three SSIM points over not
optimizing at all — the optimized tensor is very nearly *inert* outside the basin it was fitted in.

**[DEVIATION — RECORD]** The worklog and tracker say instead that these nulls are "actively destructive, WORSE
than doing nothing". **The artifacts do not support that** (0.1729 > 0.1423); the phrase belongs to exp_05's B2
arm, where it is gate-verified (0.1610 vs 0.2814, 0/64 improved), and appears to have been transposed across the
joint reading. The distinction matters for §5.1: *inert* and *harmful* are different geometries, and the whole
opposite-geometries argument depends on keeping them apart. Caveat both ways: A1-probe replays from `keyed(k)`
and A2-0 from `global(0)`, and a "base nulls under `keyed(k)`" arm was never run, so this is a proxy comparison.

---

## 3. The real result: greedy is the artifact

### 3.1 What J1b and J1c changed, and what they showed [FINDING]

J1b and J1c hold the channel (16 null rows), the backbone, the guidance scale, the clips and the noise fixed,
and change **only the objective**: from *per-step tracking of the inversion pivots* to *joint optimization of
the rollout endpoint through the differentiable sampler*. Both failure modes move:

| Axis | greedy per-step | joint endpoint | change |
|---|---|---|---|
| Fresh-noise capacity (mean endpoint MSE, same 8 clips) | 0.4294 (A2) | **0.2731** (A3) | 1.57×; 3/8 clips land inside A1's own-basin range |
| Foreign-basin transfer (mean SSIM, same 8 clips) | 0.1633 (A1-probe) | **0.4753** (J1c keyed) | **2.91×** |

The greedy arm's two headline weaknesses — "can't get there from fresh noise" and "doesn't survive a different
noise draw" — **both** substantially dissolve under a through-the-sampler objective. That is a strong, coherent
signal, and it is the single most transferable thing exp_04 produced.

### 3.2 The confound I cannot dismiss, and the one place it does not bite [INFERENCE — read this before quoting §3.1]

**J1b changed the budget as well as the objective.** Greedy A2 ran 25 steps × J=10 = 250 cheap single-step inner
iterations at ~21.2 s/example (adequacy J=10 cell). A3 ran 300 iterations of a *full 25-step remat'd rollout* at
~405 s/example (3,237 s ÷ 8). That is roughly **19× the per-example optimization compute** of the arm it beat —
and about 10× the adopted J=50 recipe. Worse, the adequacy probe's own plateau verdict was **"recipe-limited"**:
more greedy iterations were still buying accuracy at the grid's edge. **So the capacity half of §3.1 is
confounded: some unknown fraction of 0.429 → 0.273 is budget, not objective shape.** No arm in this experiment
separates them.

**The transfer half is much cleaner, and here is why.** A *better-optimized* greedy null is one that fits its own
per-step pivots more tightly. Tighter fitting to a basin-specific trajectory should, if anything, make a locked
null **less** transferable, not more. So the direction of the budget confound on the transfer axis runs
*against* the observed effect: J1c's 0.163 → 0.475 cannot be explained by "A3 simply had more compute" without
also positing that extra greedy compute would have improved greedy transfer, which the mechanism argues against.
**[INFERENCE]** I therefore treat J1c, not J1b, as the load-bearing evidence for the objective-shape claim, and I
would recommend any future write-up lead with the transfer number rather than the capacity number.

A cheap experiment would settle it: run the greedy arm at the *adopted* J=50 (or higher) on the same 8 clips and
re-probe transfer. It was never run. **[JUDGMENT]** This is the highest-value ~1 v6e-8-hour follow-up exp_04
could still buy, and I flag it as the reviewer's most likely "why didn't you do this".

### 3.3 Transfer tracks *absolute* quality, not *retention* [FINDING]

Across the 8 clips, own-basin SSIM predicts absolute foreign-basin SSIM almost perfectly (Pearson +0.945,
Spearman +0.976) but predicts the *retained fraction* only weakly (+0.623 / +0.714). The defensible statement is
"**better-optimized clips end up better everywhere**", not "better-optimized clips lose proportionally less".
The worklog's phrasing ("transfer quality tracks optimization quality") is true of the first reading and
overstated for the second, and the retention figure itself is estimator-dependent (0.700 / 0.724 / 0.730
depending on how you average — results §6.2). **None of this changes the verdict**: at 0.475 foreign and 0.651
own-basin, the joint optimum is not deployment-grade at 300 iterations either.

---

## 4. Threats to validity — stated against ourselves

### 4.1 The gates ran at a recipe the experiment's own probe had rejected [THE BIGGEST ONE]

Results §4.4: the adequacy probe adopted J=50/lr=0.01 (2.57× better tracking loss than the default), the plan
requires G1/G2 to be evaluated *only* on the adopted recipe, and the capacity run executed at the **default
J=10** because `bash_scripts/run_wan_null_inversion.sh` never passes `null_adequacy_uri`. The re-run was
affordable (1.43 h against a 2 h budget); this was a plumbing gap, not a budget stop.

**How much does it hurt?**

- **G2: not much.** 0.4973 against a 0.75 bar, with a CI-low of 0.4697 against 0.70. A 2.57× tracking-loss improvement closing a 0.25 SSIM gap is implausible on its face, and J1b — which threw ~19× the compute at the *same* fresh-noise problem with a *better-shaped* objective — still only reached 0.651 own-basin SSIM. **[INFERENCE]** G2's FAIL is robust.
- **G1: materially.** G1 failed on `median_ratio` **3.605 against a bar of 5** — a 1.39× gap, against a recipe change worth 2.57× on the optimization's own tracking metric. **I cannot rule out that J=50 would have passed G1.** Anyone who reads "G1 FAILED" as a settled fact about the null channel is reading more than the evidence supports.
- **The selection verdict: not at all.** Selection is over-determined by the A1-probe floor (0.1729 vs 0.70 — a factor of four), and transfer is a property of the *procedure* (greedy per-step, basin-specific pivots), which more inner iterations do not change in the helpful direction (§3.2). **[JUDGMENT] This is the load-bearing claim that rescues the experiment's conclusion from its own deviation, and it is the first thing a reviewer should attack.**
- **Cross-experiment comparability: badly.** exp_04's A1 (0.8523, J=10) and exp_05's B1 (0.9227, J=50) were optimized at *different budgets*. §5.1 handles this.

**Process reading.** The R10 follow-up review installed a fail-closed guard against precisely this
failure — its docstring names it verbatim — but the guard covers "URI set and unparseable", not "URI empty". The
sibling experiment's launcher wires the URI and honored its adoption. **This is the CLAUDE.md fix-propagation
lesson running in reverse**: exp_05 built the capability, exp_04 never received it, and nothing in either
experiment's closing record noticed. A one-line launcher addition plus a "capacity refuses to start when an
adequacy artifact exists at the conventional path and no URI was passed" assertion would close it.

### 4.2 Sample sizes and metric coverage

J1b and J1c are **n = 8** — the first eight DEV clips, no CIs, no multiplicity control, and the §3.3 correlations
are descriptive. J1b reports **latent MSE only**; SSIM enters only at J1c. The headline "3/8 clips reach
own-basin quality" is a count over eight, and one of them (ep45499) is an outlier the greedy arm also nearly
solved (A2 = 0.037). Strip that clip and the story weakens but survives (2/7, and the transfer ratio is
essentially unchanged).

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

### 5.1 The two slots fail with opposite geometries [FINDING]

From a **fresh** noise basin, on the **same 64 clips**, with the **same manifest hash**:

| | do-nothing control (MSE) | after per-step greedy optimization | effect |
|---|---|---|---|
| **Null slot** (exp_04, A2-0 → A2) | 4.743 | 0.490 | **9.7× better** |
| **Positive slot** (exp_05, B2-0 → B2) | 1.418 | 5.625 | **4.0× worse** |

The positive slot's *do-nothing* control is already 3.3× better than the null slot's, and greedy optimization
**inverts the ordering**: 64/64 improved in the null slot, **0/64 improved** in the positive slot. In-basin the
picture flips again — the positive slot reaches 0.9227 and the null slot 0.8523. So: **the positive channel is
the more powerful one and the more dangerous one; the null channel is weaker and better behaved.**

### 5.2 A mechanism story for that asymmetry [HYPOTHESIS — not tested by either experiment]

Under `v = v_unc + w(v_cond − v_unc)` at w = 5, the conditional branch carries coefficient **+5** and the
unconditional **−4**, and in exp_05 the conditional is an 8-token context against a 512-row unconditional (so it
is also structurally the *lighter* tensor with the *larger* lever). A greedy per-step objective asks the channel
to jump to a pivot that, from a foreign basin, is far away — which demands a large velocity correction. **More
per-step authority applied greedily from the wrong basin produces more damage, not less.** That would explain
why the same procedure improves the low-authority null slot 9.7× and degrades the high-authority positive slot
4.0×. It is consistent with every number in both experiments, and it is **untested**: no arm in either experiment
varies w, varies L, or measures per-step step magnitude. I would not put it in a paper without the ablation.

### 5.3 Why exp_04's G1 FAIL and exp_05's H1 PASS are not a slot comparison [FINDING]

Their controls measure different things. exp_04's A0 collapses CFG to identity and therefore replays at the same
w = 1 the pivots were computed at (§2.1) — a near-self-consistent, strong control (MSE 0.335). exp_05's B0 keeps
CFG active (8-token conditional ≠ 512-row unconditional) and so replays w = 5 dynamics against w = 1 pivots — a
mismatched, weak control (MSE 0.890). Ratios of 3.6× and 28.6× against those two controls are simply not the
same statistic. **This was predeclared** in exp_05's plan §4 H1 interpretation note; it is repeated here because
the two numbers are the ones most likely to be quoted side by side. Add the §4.1 budget difference (J=10 vs
J=50) and the pivot difference (512-row vs 8-token inversion contexts), and the only defensible cross-experiment
statements are the ones in §5.1, which are matched by construction (same cohort, same fresh noise, each slot
against its own control).

### 5.4 The joint conclusion [INFERENCE]

Across both slots, **static per-clip conditioning targets are dead as a training signal.** Both experiments'
fresh-noise arms failed their predeclared gates (null 0.4973 vs a 0.75 bar; positive 0.1610, worse than doing
nothing), and both experiments' locked contexts degrade badly across noise basins (null 0.173; positive 0.525).
A cached tensor per clip cannot be the supervision, because the tensor that works is a function of the noise
draw, and at training time the noise draw is fresh every step. **This is the finding that killed exp_04's P2/P3
and exp_05's K2/K3, and it should stay killed.**

---

## 6. What these results do and do not license

### 6.1 Licensed

1. **Stop proposing capacity-first changes.** More conditioning tokens, a bigger adapter, deeper injection points — none of these are indicated. The channel drives the frozen backbone to 0.85 (null) and 0.92 (positive) SSIM; the deployed adapter does not come close. **Capacity is not the binding constraint, and no proposal should assume it is without new evidence.**
2. **Through-the-sampler objectives are the direction.** The one manipulation that improved both the capacity and the transfer axes was changing the objective's shape (§3.1), with the transfer half surviving the budget confound (§3.2).
3. **The frozen backbone is exonerated.** It reconstructs DROID futures to 0.85–0.92 SSIM when conditioned correctly. Unfreezing it is not indicated by anything here.
4. **Gate-as-code and predeclaration earned their cost.** Every verdict in this experiment was computed, not argued, and the one place the process failed (§4.1) failed *silently in the plumbing*, which is exactly the class of error predeclared gates cannot catch and provenance-bound artifacts can.

### 6.2 NOT licensed — and one distinction that is load-bearing

1. **No claim that a null-embedding adapter would work.** P3 was never built. exp_04 has no amortization evidence.
2. **No deployment claim.** Every arm is a per-clip oracle (§4.3), and even the best of them misses the 0.70 absolute floor.
3. **No claim that "the basin problem is solved".** J1c retained ~72 % of a 0.651 own-basin SSIM. 72 % of not-good-enough is still not good enough.
4. **No settled claim that the null channel fails G1** (§4.1) — that verdict is contaminated by the recipe deviation.
5. **THE DISTINCTION THAT MATTERS.** exp_05's basin finding kills **static per-clip conditioning targets**. It does **NOT** kill the **rollout-loss family**. These are different objects, and conflating them would have cancelled the campaign's most promising direction:

   | | what was measured and failed | what remains untested |
   |---|---|---|
   | **Target form** | a *fixed tensor per clip*, cached and regressed onto | a *state-conditioned emitter* that reads `z_t` and re-emits every step — not a fixed tensor, so "the tensor doesn't transfer across basins" does not apply to it |
   | **Objective form** | *per-step greedy tracking* of inversion pivots | *through-the-sampler* optimization of the rollout endpoint — which J1b/J1c showed behaves differently on **both** the capacity and the transfer axes |
   | **Training signal** | teacher-forced regression onto cached targets | a differentiable rollout loss computed live, with no cache at all |

   Everything exp_04 and exp_05 refuted lives in the left column. exp_06 lives in the right one, and J1b/J1c are
   its positive evidence. **[JUDGMENT]** I consider this the single most important paragraph in either analysis
   document, and the one most at risk of being flattened into "the adapter line failed" by a summary.

---

## 7. Recommended next steps

| Priority | Action | Cost | Why |
|---|---|---|---|
| **1** | Nothing here blocks **exp_06**; it is the correct continuation and its motivation (E2) rests on J1b/J1c | — | The mechanism arc is complete; every predeclared gate has fired |
| **2** | Add the one-line `null_adequacy_uri` wiring to `run_wan_null_inversion.sh` **plus** a fail-closed assertion when an adequacy artifact exists at the conventional path and no URI was passed | minutes | §4.1; the guard exists, the plumbing does not |
| **3** | If exp_04's G1 verdict is ever cited as a fact, first re-run A1/A2 on DEV-64 at the **adopted J=50** and re-gate | ~1.5 h v6e-8 | §4.1 — closes the deviation properly rather than arguing around it |
| **4** | Re-probe **greedy transfer at J=50** on the same 8 clips | ~1 h v6e-8 | §3.2 — the cheapest way to disentangle objective-shape from budget in the campaign's headline lesson |
| **5** | Pull the published `videos/` and write the P4 HTML report (SOP artifact 12) | host-only | Nobody has looked at what a 0.85 vs 0.50 vs 0.17 reconstruction *looks like*; the qualitative half of the evidence is unexamined |
| **—** | **Do NOT** revive P2/P3 as planned | ~225+ v6e-8-h for A3-caching TRAIN-2000 alone | §5.4. Any revival is an **exp_07-scale new proposal** with its own gates, not a continuation under exp_04's fired ones |

---

## 8. Open questions I would put to the reviewer first

1. Does the over-determination argument in §4.1 actually hold — is A1-probe's transfer failure genuinely independent of the optimization budget, or is that special pleading for a run that missed its own predeclared recipe?
2. Is §3.2's asymmetry argument (budget confounds capacity but runs *against* the transfer effect) sound, or is there a mechanism by which more greedy iterations improve transfer that I have not considered?
3. Is §2.1's account of A0's strength (CFG collapse ⇒ effective w = 1 ⇒ near-self-consistent replay of w = 1 pivots) correct as stated? The whole "G1 failed on its control" reading depends on it.
4. Should the results doc's headline numbers carry an explicit "measured at J=10, not the adopted J=50" annotation everywhere they appear, rather than only in §4.4?
5. Is the §6.2 static-targets-vs-rollout-family distinction stated strongly enough to survive being summarised by someone who has not read the arms?
