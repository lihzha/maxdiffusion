# pos_context_analysis — exp_05 P4′ (SOP artifact 11)

**Status: FINAL — revision 3.** Revision 2 answered the first Codex analysis review; **revision 3 applies the
closing review** (`pos_context_codex_closing_review.md`, REQUEST-REVISION — *"the exp_05 STOP and gate record
remain valid"*; two scoping leaks, the exp_04-J=50 fold-in, and artifact-index staleness; no new compute). All
four items are applied. **This document is submitted for closure adjudication.** Every interpretive step is labelled **[FINDING]** (artifact-backed), **[INFERENCE]**
(supported but not compelled), or **[HYPOTHESIS]** (a mechanism story this experiment does *not* test);
**[REV2]** / **[REV3]** mark edits by the round that made them. Numbers are not restated except where the argument
turns on them — the record is `pos_context_results.md`.

> **⚠ THREE CONCLUSIONS FROM REVISION 1 WERE NARROWED OR WITHDRAWN.** (1) B2-probe > B2 does **not** rule out
> optimizer failure — the signature is mixed, and §3.2 is hedged accordingly. (2) The static-target conclusion
> narrows to the **predeclared single-basin greedy-pivot family**; rollout-loss training is *not refuted*, which
> is not the same as *alive*. (3) §4.1 now leads with **retention**; its absolute-quality comparison is
> **inconclusive**. **The gate record itself was independently reproduced by the reviewer and stands unchanged.**

---

## 1. The verdict in one paragraph

Yixun's question was sharp: *is the `pre_context` adapter's poor rollout quality a limit of its capacity or of
its training signal?* exp_05 answers it, at a scope worth stating precisely. **[REV3] Output-channel
expressivity is not limiting on these clips; head function capacity remains untested.** Per-clip optimized
8-token contexts —
the exact tensor the deployed head emits, passed exactly as deployment passes it — drive the **frozen** Wan2.2
5B to **0.9227 mean SSIM**, with 64/64 clips improved and a 28.6× median MSE reduction over the frozen-context
control. The deployed channel, unchanged, is capable of near-reconstruction. **But that capability is bound to
the noise basin it was optimized in, and bound hard.** Re-optimizing per-step from a fresh ε₀ does not merely
underperform — it lands at 0.1610, **worse than doing nothing** (0.2814), with **0 of 64** clips improved and a
median ratio of 0.254. Locked contexts under foreign noise fall to 0.5254 against a 0.70 floor. The predeclared
rule stopped the experiment before any of the ~14.80 GiB cache was written, and Yixun honored the STOP.
**[REV3] The finding is not "the positive slot is weak" — it is that the tested single-basin greedy-pivot
cached-target family was rejected for K2/K3**, because the tensor that works is a function of the noise draw and
the training-time noise draw is fresh every step. **Robust/multi-noise static targets and state-conditioned
emitters remain untested.** *(Revision 2 said "a per-clip conditioning tensor is the wrong object to regress
onto" — broader than what was tested, and narrowed here.)*

---

## 2. H1: the campaign's strongest positive fact

### 2.1 What it establishes [FINDING]

H1 passed every conjunct with margin (0.9227, CI [0.9129, 0.9314], 64/64 improved, ratio 28.6× against a bar
of 5), and replicated on TRAINFIT-16 (0.9095). Three design choices make it unusually load-bearing:

1. **The representation is deployment-exact.** C ∈ ℝ^{8×4096} is passed as the *entire* `encoder_hidden_states`, which is literally what `wan_pre_context_adapter_forward` does. Plan v1 had it spliced into a 512-row padded context; the reviewer's F1 caught that this would have made the targets a different object from the deployment path — the single most consequential catch in either experiment's plan cycle. Without it, H1 would have measured a channel nobody ships.
2. **The conditioning path is bit-verified.** S3's oracle shows the deployed forward *is* the replay operator's `v_cond` — bitwise at fp32, and bitwise at bf16 once the single-owner activation cast is applied (and demonstrably **unequal** without it). So "the optimized context reconstructs the video" and "the deployed head emitting that context would reconstruct the video" are the same statement, not two hopefully-similar ones.
3. **The backbone is frozen and the architecture is untouched.** No new parameters, no new injection points.

**[REV2] Therefore, stated at the scope the evidence supports [INFERENCE]:** H1 establishes **output-channel
expressivity** — the 8-token conditioning representation the deployed head emits *can* carry a solution that
drives the frozen backbone to ~0.92 SSIM on these clips. It does **not** establish that the ~128M
state/action→context head has sufficient **function capacity** to compute such a context from `(z_i0, actions)`;
that is a different quantity and K3 never ran.

So: a proposal to add **conditioning tokens** or **injection depth**, or to **unfreeze the backbone**, attacks a
constraint measured not to bind and should say why it now does. A proposal to **widen or deepen the head** does
**not** fall under that objection — revision 1 wrongly included it, and nothing here bounds the head's function
capacity.

### 2.2 What it does not establish

H1 is a **per-clip oracle**: it optimizes C against that clip's own ground-truth latents. It proves the channel
can *express* a solution; it says nothing about whether any function of `(z_i0, actions)` can *find* it. And
because H1's own-basin quality is what a **cached target** would have preserved, H1's magnitude is exactly the
thing H2 and the probe then take away.

---

## 3. H2: not a shortfall, an inversion

### 3.1 The detail that matters [FINDING]

The usual reading of a failed gate is "it didn't get far enough". H2 is worse than that: **0/64 improved**, and
a median ratio of **0.254** means the typical clip's endpoint MSE got roughly **four times worse** than leaving
the frozen warm-start context alone. Per-step optimization from a fresh basin is not weak conditioning; it is
**harmful** conditioning.

### 3.2 B2-probe vs B2: a mixed signature, consistent with objective mismatch but not discriminating **[REV2 — claim hedged]**

Revision 1 presented this as a "clean fingerprint" that "rules out" optimizer failure. **The reviewer showed the
signature is mixed, and the exclusionary claim is withdrawn.** The full picture from the artifacts:

| Comparison (B2-probe vs B2) | Value | Direction |
|---|---|---|
| Mean future-SSIM, all 64 | 0.2118 vs 0.1610 | probe **better** |
| **Mean future-MSE, all 64** | **5.9509 vs 5.6250** | **probe WORSE** — the other endpoint metric disagrees |
| Examples where probe SSIM > B2 (3 seeds averaged), all 64 | **35 / 64** | barely a majority |
| Median paired ΔSSIM, all 64 | **+0.00435** | negligible |
| Examples where probe > B2, matched 8 | **3 / 8** | minority |
| Median paired ΔSSIM, matched 8 | **−0.00797** | **negative** |

So "the contexts perform better on noise they were not optimized for" holds only for the **mean SSIM over all
64**, and reverses on MSE, on the median, and on the matched eight. **[REV2]** Moreover the inference itself was
unsound: **optimizer failure or overfitting can also produce basin-specific destructive contexts** — a failed
optimizer need not leave a context that behaves identically everywhere, which was revision 1's unstated premise.

**Standing statement:** the B2-probe/B2 relationship is **consistent with objective mismatch, but it is not
discriminating** between mismatch and optimizer failure/overfitting.

### 3.3 The mechanism story [HYPOTHESIS — untested here]

The per-step objective is *match pivot `i+1`*; the quantity anyone cares about is *the endpoint*. In-basin these
coincide, because the pivots were produced by inverting from the target — following them arrives at the target.
From a fresh ε₀ they diverge: the sampler is somewhere the pivot sequence never passes through, so tracking
demands a large velocity correction at every step, and the greedy objective contains **no term that penalizes
endpoint error**. Twenty-five large, endpoint-blind corrections compound. Add that the conditional branch carries
coefficient **+5** in `v = v_unc + w(v_cond − v_unc)` at w = 5, and the 8-token context has a large lever on the
combined velocity: **more per-step authority applied greedily from a wrong basin produces more damage.** That
would explain why the same procedure *helps* exp_04's lower-authority null slot 9.7× and *hurts* exp_05's
positive slot 4.0× (§5.1).

**[REV2] The check that would actually discriminate, on already-published data, host-only.** If B2's per-step
**tracking** losses fell normally while its endpoint degraded, that is **surrogate/endpoint mismatch**; if they
did not fall, it is optimizer failure. The data are already written into the published records:
`…/k1/capacity/b2/shard_*/record_*.npz` → `per_step_final_losses [25]`, alongside `final_future_mse`.

Revision 1 said "reading one shard settles it". **That is corrected**: the diagnostic requires **all eight B2
shards** — the full DEV-64 cohort — and the comparison of interest is **B2's final tracking-loss distribution
against B1's**, since B1 is the arm whose surrogate *and* endpoint both succeeded. A single shard is eight
examples and could not carry a distributional claim.

Even then, the **compounding mechanism above stays a HYPOTHESIS**: low surrogate loss with a bad endpoint would
establish that the surrogate and the endpoint came apart, but not *that twenty-five large corrections compounding
under a +5 CFG coefficient* is why. **[JUDGMENT]** I did not run it in this docs-only round; it is §8 priority 1.

---

## 4. The probe: "collapse" is the wrong word

B1-probe = 0.5254 (DEV), i.e. **57 % of B1's own-basin 0.9227 retained** under foreign noise — and *the highest
absolute foreign-basin SSIM recorded anywhere in the campaign on the full 64-clip cohort*. The worklog's
"collapse to ~0.5" undersells it. This matters for two reasons:

1. **It is a partial, structured failure, not a null result.** Something in an optimized 8-token context is basin-*independent* and worth about 0.53 SSIM. What that component is was never characterized (the plan's cross-example cosine/PCA diagnostic was scheduled for K2 and never ran).
2. **It failed the gate anyway, correctly.** 0.5254 < 0.70. Per the standing R5 note, the **absolute floor is the binding constraint**; the relative conjunct (0.569 < 0.7) is logically redundant on [0,1] SSIM and should never be quoted as an independent reason.

### 4.1 Matched-eight **retention** favours joint endpoint optimization; **absolute quality is inconclusive** **[REV2 — retitled and downgraded]**

Restricting every arm to the **same 8 clips** J1b/J1c used (mean future-SSIM), **retention first**:

| Conditioning, replayed under **foreign** noise | **retention** | absolute | own-basin |
|---|---|---|---|
| exp_04 A1-probe — **greedy null**, locked · **J=10-era** | 0.183 | 0.1633 | 0.8911 |
| exp_05 B1-probe — **greedy positive**, locked · J=50 | 0.489 | 0.4586 | 0.9379 |
| **exp_04 J1c — JOINT null, locked** · **J=10-era** | **0.730** | 0.4753 | 0.6509 |

**[REV3] The two exp_04 rows are J=10-era J1b/J1c evidence.** Neither follow-on was re-run at J=50, so — unlike
§5.1's clean-gate comparison, which is now recipe-matched — this table's exp_04 rows remain at the J=10 budget
and the J=10-vs-J=50 caveat still applies to them. They are preserved rather than overwritten because the
joint-null result has no J=50 counterpart.

**The retention ordering is the informative half:** jointly-optimized tensors keep **0.730** of their own-basin
quality against **0.489** for greedily-optimized ones — about 1.5× — and they do so from a channel that is
*weaker in-basin by a wide margin* (0.651 vs 0.938). Note that exp_04's joint-retention estimate itself ranges
**0.7007–0.7301** depending on the averaging convention (exp_04 results §6.2), so the 1.5× is approximate.

**[REV2] The absolute-quality comparison is inconclusive and no longer carries a conclusion.** 0.4753 vs 0.4586
is a gap of 0.017 at **n = 8 with no CIs**, on a subset **unfavourable to the positive slot** (B1-probe scores
0.4586 here against 0.5254 on all 64), across **different slots**, **different optimization budgets** (J=10 vs
J=50), and different pivots. **[REV3]** No 64-clip joint-null counterpart exists, so no full-cohort ordering can
be inferred either way. *(Revision 2 speculated the positive probe "would likely lead" on the full cohort; that
extrapolation is unsupported and is deleted.)*

**[REV2] Revision 1 concluded "on the transfer axis, which objective you optimize matters more than which slot
you optimize" and called it the campaign's central lesson. That is withdrawn.** The supported statement is:

> Matched-eight retention is **suggestive evidence for objective form** mattering on the transfer axis. It is
> not proof that objective matters more than slot — the arms differ in slot, budget and pivots simultaneously.

---

## 5. The joint reading with exp_04

### 5.1 The two experiments' fresh-noise arms moved in opposite directions [FINDING — observation only] **[REV2 — retitled and de-causalised]**

From a **fresh** basin, on the **same 64 clips** (identical `manifest_hash 433f8691…`) — **[REV3] and, since
exp_04's J1-5 clean-gate rerun, at the same adopted J=50 recipe** — but still in different representations,
against differently-constructed controls, and on non-shared pivots (§5.2):

| | recipe | do-nothing control (MSE) | after per-step greedy optimization | effect | clips improved |
|---|---|---|---|---|---|
| **Null slot** (exp_04, A2-0 → A2) **[REV3]** | **J=50** (adopted) | 4.743 | **0.303** | **15.7× better** | 64/64 |
| **Positive slot** (exp_05, B2-0 → B2) | **J=50** (adopted) | 1.418 | 5.625 | **4.0× worse** | **0/64** |
| *(null slot at J=10 — superseded by J1-5, retained for continuity)* | J=10 | 4.743 | 0.490 | 9.7× better | 64/64 |

The positive slot's *do-nothing* control is 3.3× better than the null slot's — the frozen 8-token context is
already doing useful work, whereas base nulls collapse CFG to identity and do none. Greedy optimization then
inverts the ordering. **[REV3]** In-basin the picture inverts once more: positive **0.9227** vs null **0.8868**
— a **0.036** gap, now measured at the same recipe (it was 0.070 against exp_04's J=10 A1 of 0.8523).

**[REV2] Revision 1 concluded "the positive channel is the more powerful and the more dangerous; the null
channel is weaker and better behaved." That is withdrawn** — it attributes a between-row difference to the
*slot* when the rows differ in representation, pivots and control design. **[REV3]** (Recipe was a fourth
mismatch at revision 2; exp_04's J1-5 removed it. The remaining three still prohibit slot attribution.) **Each row is internally
matched** (same cohort, same ε₀, each arm against its own control) **and is a sound finding on its own;
comparing the two rows remains descriptive, not a controlled slot contrast.** What survives is that the two
directions are genuinely opposite, which is why the pair is worth recording.

### 5.2 What must NOT be compared [FINDING — predeclared]

exp_05's H1 PASS and exp_04's G1 FAIL are **not** a slot comparison. **[REV3]** Revision 2 listed four reasons;
exp_04's J1-5 clean-gate rerun removed the budget one, so **three remain — and they are sufficient**:
(a) **controls differ in kind** — exp_04's A0 collapses CFG (effective w = 1, replaying pivots computed at w = 1:
a near-self-consistent, strong control at MSE 0.335), exp_05's B0 keeps CFG active and replays w = 5 dynamics
against w = 1 pivots (a mismatched, weak control at MSE 0.890). **[REV3]** Ratios of **4.681×** (exp_04's
authoritative J=50 G1; the historical J=10 figure was 3.605×) and 28.6× against those two controls are
different statistics. This was predeclared in plan §4's H1 interpretation note;
(b) **representations differ** — 8 tokens as the whole context vs 16 rows inside a 512-row padded one;
(c) **pivots are not shared** — exp_05 inverts with `C_init` at 8 tokens, so its trajectories, controls and
targets are different objects (pinned by S4 mutant R2); and
**[REV3] (d) budgets — RESOLVED, no longer a mismatch.** Historically exp_05 honored the **adopted J=50**
while **exp_04's J1-4 ran the *unadopted* J=10** because its launcher never passed `null_adequacy_uri`
(exp_04 `null_adapter_results.md` §4.4, issue #15), leaving exp_04's formal selection INDETERMINATE.
**exp_04's J1-5 clean-gate rerun fixed the launcher and re-ran both cohorts at J=50, retaining the predeclared
STOP** — so both experiments' verdicts are now predeclared verdicts at the adopted recipe, and the budget
mismatch is gone. Retained here because it explains why earlier revisions of these reports treated exp_04 as
J=10-only evidence; the sentence that followed it — that comparing a predeclared verdict against an
unadopted-recipe observation is not a slot comparison — no longer applies, because both are now predeclared
verdicts.

**[REV2]** And the same discipline applies one section up: **each row of §5.1 is internally matched; comparing
the two rows remains descriptive, not a controlled slot contrast.** Revision 1 said §5.1's comparisons were
"matched by construction", which was true of each row and false of the pair.

### 5.3 The joint conclusion [INFERENCE]

**[REV2]** Revision 1 said "static per-clip conditioning targets are dead as a training signal". **Too broad.**
The supported conclusion, at the scope of what was actually tested:

> The predeclared **single-basin cached-target family** — one per-clip, per-timestep context sequence produced by
> **greedy pivot tracking** — is **rejected for K2/K3 under these conditions.**

The supporting evidence is real: both fresh-noise arms failed their predeclared gates (positive 0.1610, worse
than its control; **[REV3]** null **0.6638** against a 0.75 bar, CI-low 0.6312 against 0.70 — exp_04's
authoritative J=50 figures) and both experiments' locked contexts degrade across basins (positive 0.525, null
**0.1666**).

**What this does NOT reject, and must not be swept in [REV2]:**
- **Multi-noise / robust joint optimization of a static tensor.** Neither experiment ever optimized one tensor against several noise draws simultaneously. "A tensor fitted to one basin doesn't transfer" is not "no static tensor can transfer".
- **A state-conditioned emitter.** K2's target is a 25-step *sequence*, and K3's proposed emitter reads `z_t` rather than replaying a frozen tensor. **K3 never ran.**

This is why exp_05's K2/K3 and exp_04's P2/P3 stopped, and they should stay stopped. It is not a proof that no
static formulation can work.

---

## 6. Threats to validity — stated against ourselves

1. **The control was misreported until now.** B0 is **0.3215**, not the "~0.25" carried in the worklog, tracker and CLAUDE.md; the derived multiple is **2.87×**, not 3.6× (results §4.1). **[REV2] Precisely why no verdict changes:** B0's **future-MSE** *is* a gate input — it is the numerator of H1's `median_ratio` — but B0's **future-SSIM**, the quantity that was wrong, enters no conjunct at all; it appears only in prose multiples. Revision 1 said "B0 is a ratio denominator, not a gate input", which conflated the two quantities and was half wrong. The error was live in three handoff documents and **[JUDGMENT]** I could not reconstruct its origin; the reviewer could not either. That is a provenance failure in the reporting layer, and it is why this document re-derives every number from the artifacts rather than the worklog.
2. **B1 is an oracle, not a system** (§2.2). exp_05 has **zero** evidence about amortization: K3 never ran, so the question Yixun actually asked — can the head *learn* to emit these? — is unanswered by measurement. Only the *precondition* (the channel can express a solution) is established.
3. **The state-conditioned caveat was flagged at the time and remains live.** K3's regression conditions on cached `z̄_t` states — a *state-conditioned emitter*, not a fixed context — which H2 and the probe do not directly measure. The worklog's own reading called this out and then argued it is "substantially weakened" by B2's failure. **I think that argument is right but weaker than it sounds**, and §7.2 explains why the distinction it points at is load-bearing rather than a technicality.
4. **The 0.2946 anchor cannot carry the "3.13×" claim.** That anchor is a **4-sample** validation over **four correlated windows of one episode** — a wiring check. The qualitative claim (the deployed adapter sits far below what the channel expresses) survives; the ratio should not be quoted as a measurement.
5. **[REV3 — largely resolved] The videos now exist and have been cross-checked.** K1 itself published none (ruled not K1-blocking), but the post-STOP capacity-videos job (`20260809-173808-13c3cadc-capvideos-yixun`) rendered **B1, B2 and B1-probe k=0** over 8 clips — **24 mp4s** at `gs://v6_east1d/datasets/droid_wan_pos_context/k1/videos_att-0809-173808/` — and both HTML pages are committed. The numeric cross-check is **CLEAN** (recomputed per-clip future-SSIM matches `gate_tables.json` within fp16-storage tolerance, max |Δ| 0.052 on b2; 8-clip means b1 0.923, b2 0.178, b1_probe_k0 0.439). **Residual caveat:** the rendered subset is the **8 lowest-`ordinal` DEV clips**, and no systematic qualitative review by a human has been recorded — so a systematic artifact hiding behind the means is *less* likely than at revision 2, not excluded.
6. **Narrowness.** One dataset, one backbone, one resolution, one guidance scale, one L_pos, DEV/TRAINFIT only (TEST-64 correctly untouched). σ₀ = 1.0 vs the reference's 0.999 (ratified deviation) means prior-art numbers are directional context, not matched comparisons.
7. **What went right, for calibration.** Coverage complete, invalid pairs zero (so the imputation machinery is production-untested), quarantines zero, parity audit clean across 11 positive-slot deltas plus the inherited core, **the adopted recipe actually applied** (the thing exp_04's launcher silently failed to do), and the two measurements that could have silently corrupted everything — the non-no-op bf16 cast and the false fp16→bf16 value-preservation premise — were both caught by measurement rather than assumption. The one real bug (K1-1) was an **inherited stale copy**, an integration failure, not a defect in exp_05's own code.

---

## 7. What these results do and do not license

### 7.1 Licensed

1. **[REV2] OUTPUT-CHANNEL-capacity proposals are not indicated.** 0.9227 SSIM through the frozen backbone via the unchanged deployed channel means **more conditioning tokens**, **deeper injection**, or **unfreezing the backbone** all attack a constraint measured not to bind. **"Bigger head" is NOT in this list** — revision 1 wrongly included it; the head's *function capacity* (can it compute such a context from `(z_i0, actions)`?) was never measured, because K3 never ran (§2.1).
2. **[REV2] The evidence favors training/objective limitations over output-channel capacity** — which is the direction Yixun's Query 1 proposed. Revision 1 said "the training signal *was* the bottleneck"; that asserts more than a single oracle arm can show, since the head's function capacity was never measured (§2.1). exp_05 supports the premise's *direction* while refuting the specific remedy proposed for it.
3. **Do not cache the tested single-basin greedy-pivot targets** (§5.3). The ~14.80 GiB K2 build and everything downstream were correctly not spent.
4. **The deployment-exact representation discipline should be standard.** The F1 catch is why H1 measures something real; any future oracle study must be pinned to the deployed forward the same way.

### 7.2 NOT licensed — and the distinction that motivated exp_06

1. **No claim that the `pre_context` adapter cannot be trained well.** exp_05 refuted one *training signal*, not the architecture — whose **output channel** it vindicated (§2.1; its function capacity is untested).
2. **No deployment claim.** Every arm is a per-clip oracle; the best transfer number in the experiment (0.5254) misses the 0.70 floor.
3. **No mechanism claim about *why* B2 inverts** — §3.3 is a hypothesis, and **[REV2]** §3.2 is no longer even discriminating evidence that a mismatch (rather than optimizer failure) is what happened.
4. **THE LOAD-BEARING DISTINCTION.** The campaign refutes **the tested greedy single-basin cached-target family**. It does **NOT** refute the **rollout-loss family**. Flattening these together would cancel the campaign's most promising direction, so it is worth a table:

   | | measured and refuted | **untested** — neither refuted nor validated |
   |---|---|---|
   | **Target form** | a *fixed per-clip, per-timestep tensor sequence* from **greedy pivot tracking**, cached and regressed onto | a *state-conditioned emitter* reading `z_t` and re-emitting every step; **also** robust/multi-noise joint optimization of a static tensor — **[REV2]** neither was ever attempted |
   | **Objective form** | *per-step greedy tracking* of inversion pivots, with **no endpoint term** | *through-the-sampler* optimization of the rollout endpoint — better on both axes at n=8, **[REV2]** at ~19× the compute, so the mechanism is unattributed. **[REV3]** This is **J=10-era J1b/J1c evidence**; neither was re-run at J=50 |
   | **Training signal** | teacher-forced regression onto a cache | a differentiable rollout loss computed live, no cache at all |
   | **Evidence status** | gates fired: 0.1610 / 0/64 improved / probes below floor | §4.1: joint nulls retain **0.730** vs greedy's 0.489 — suggestive on the transfer axis; absolute quality inconclusive |

   **[REV2] The status line, replacing revision 1's implication that the right column is "alive":**

   > **Rollout-loss training is NOT REFUTED, and has an n=8 motivating oracle result. It is not validated and
   > not "alive" in any deployment sense** — J1c sits at 0.4753 foreign SSIM, **below the 0.70 floor**, and
   > J1b's capacity half is compute-confounded.

   **[JUDGMENT]** This paragraph remains the one most at risk of being flattened — now in *either* direction:
   into "the adapter line failed", or into "rollout losses work".

5. **A specific warning about over-reading §5.1.** "Greedy optimization of the positive slot is harmful from fresh noise" is **not** "the positive slot is harmful from fresh noise". The frozen-context control B2-0 reaches 0.2814 doing nothing at all, and B1-probe reaches 0.5254 with a context optimized elsewhere. The channel is not the problem in either case.

---

## 8. Recommended next steps

| Priority | Action | Cost | Why |
|---|---|---|---|
| **1** **[REV2]** | Read `per_step_final_losses` from **all eight** published `b2` shards (the full DEV-64 cohort) and compare **B2's final tracking-loss distribution against B1's** | host-only | §3.3. Low surrogate loss with a bad endpoint would establish surrogate/endpoint mismatch — the thing §3.2 can no longer show. Revision 1 said "one shard"; one shard is 8 examples and cannot carry a distributional claim |
| **2** **[REV2]** | Propagate **both** corrections into the tracker and CLAUDE.md: B0 = 0.3215 / 2.87×, **and** the narrowed conclusion — *"the planned single-basin greedy cache target was rejected; state-conditioned live rollout training remains untested and risky, but was not refuted"* | minutes | §6.1, §7.2. A wrong number and an over-broad conclusion are both live in the handoff documents |
| **✅ DONE** **[REV3]** | ~~Comparison videos + the P4′ HTML report~~ — job `20260809-173808-13c3cadc` rendered B1/B2/B1-probe k=0 (24 mp4s, cross-check CLEAN); `pos_context_01-capacity-gates_results.html` and `pos_context_02-video-gallery_results.html` (+ assets) are committed | ~1 v6e-8-h + host | §6.5. Optional follow-up: a **row-order** re-render (the subset is the 8 lowest-`ordinal` clips, a recorded deviation), and a human qualitative pass |
| **4** | Run the cross-example cosine/PCA structure diagnostic on B1's optimized contexts | host-only | §4 — characterizes the ~0.53-SSIM basin-independent component. Was scheduled for K2 and never ran |
| **5** | Keep S1–S9 on the branch, unmerged | — | Per SOP, a stopped experiment's code stays on its branch. If exp_06 succeeds, S7's trainer + S9's certificate machinery are the natural starting point for a rollout-trained variant |
| **—** | **Do NOT** revive K2/K3 as planned | ~14.80 GiB cache + training | §5.3, and Yixun's STOP is honored. Any revival is a new proposal with its own gates |

---

## 9. Open questions — **answered by the review; recorded with their answers** **[REV2]**

| # | Question (revision 1) | Reviewer's answer | Where applied |
|---|---|---|---|
| 1 | Is B2-probe > B2 as strong as I claim? | **No — mixed signature** (probe worse on MSE; 35/64; median +0.004; 3/8 on the matched eight). Optimizer failure can also produce basin-specific destructive contexts. **Hedge to "consistent with, not discriminating"** | §3.2, §7.2 item 3 |
| 2 | Is the matched-8 comparison fair? | **Lead with retention; the absolute comparison is inconclusive** (0.4753 vs 0.4586, n=8, no CIs, unfavourable subset, different slots and budgets) | §4.1 |
| 3 | Does the four-reason list protect the H1/G1 contrast? | **Nearly sufficient** — keep the list, but state explicitly that exp_04 ran the **unadopted J=10 because of its launcher deviation**; and delete the "unlike exp_04" sentence from results §4.2 | §5.2(d), results §4.2 |
| 4 | Is the static-vs-rollout distinction strong enough? | **Overbroad in the other direction** — narrow "failed" to the predeclared single-basin greedy-pivot family; rollout-loss is "not refuted with an n=8 motivating oracle", **not "alive"** | §5.3, §7.2 |

Also applied from the review: H1 proves **output-channel expressivity**, not head function capacity (§2.1);
"bigger head" removed from the refuted proposals (§7.1); "the training signal was the bottleneck" softened
(§7.1); the **B0 gate-input distinction** (§6.1, results §4.1); B0 extrema **0.01763 / 0.59362**; the transfer
table heading; the **01:22:54Z** remediation chronology with an append-only worklog annotation; the tip
references; and the "3.6×" label disambiguation.

### Open questions for the NEXT review (revision 2)

1. §5.1 now refuses a slot attribution but still places the two rows adjacent. Is adjacency itself the problem, as the reviewer judged it to be for H1/G1 in results §4.2?
2. Is the §7.2 status line ("not refuted, n=8 motivating oracle, not validated, risky") the right wording for the tracker to carry verbatim into exp_06's motivation?
3. §3.2 is now non-discriminating. Does that weaken §7.2's case for the rollout-loss family — which rested partly on the objective-mismatch reading — or is exp_04's J1c evidence independent enough to carry it alone?
4. Should §4 still describe B1-probe's 0.5254 as "the highest absolute foreign-basin SSIM in the campaign on the full cohort", given that the only comparator at 64 clips is exp_04's greedy probe and no joint-null 64-clip arm exists?
5. The worklog argued B2's failure "substantially weakens" the state-conditioned-emitter bet. §7.2 argues the two are meaningfully different objects. Which framing should the tracker carry — and does the difference change anything about exp_06's risk?
