# pos_context_analysis — exp_05 P4′ (SOP artifact 11)

**Status: DRAFT for review.** The Codex analysis review is deferred; this document is written to be attacked.
Every interpretive step is labelled **[FINDING]** (artifact-backed), **[INFERENCE]** (supported but not compelled
by the artifacts), or **[HYPOTHESIS]** (a mechanism story this experiment does *not* test). Numbers are not
restated except where the argument turns on them — the record is `pos_context_results.md`.

---

## 1. The verdict in one paragraph

Yixun's question was sharp: *is the `pre_context` adapter's poor rollout quality a limit of its capacity or of
its training signal?* exp_05 answers it. **Capacity is not the limit.** Per-clip optimized 8-token contexts —
the exact tensor the deployed head emits, passed exactly as deployment passes it — drive the **frozen** Wan2.2
5B to **0.9227 mean SSIM**, with 64/64 clips improved and a 28.6× median MSE reduction over the frozen-context
control. The deployed channel, unchanged, is capable of near-reconstruction. **But that capability is bound to
the noise basin it was optimized in, and bound hard.** Re-optimizing per-step from a fresh ε₀ does not merely
underperform — it lands at 0.1610, **worse than doing nothing** (0.2814), with **0 of 64** clips improved and a
median ratio of 0.254. Locked contexts under foreign noise fall to 0.5254 against a 0.70 floor. The predeclared
rule stopped the experiment before any of the ~14.80 GiB cache was written, and Yixun honored the STOP.
**The finding is not "the positive slot is weak" — it is that a per-clip conditioning tensor is the wrong
object to regress onto, because the tensor that works is a function of the noise draw and the training-time noise
draw is fresh every step.**

---

## 2. H1: the campaign's strongest positive fact

### 2.1 What it establishes [FINDING]

H1 passed every conjunct with margin (0.9227, CI [0.9129, 0.9314], 64/64 improved, ratio 28.6× against a bar
of 5), and replicated on TRAINFIT-16 (0.9095). Three design choices make it unusually load-bearing:

1. **The representation is deployment-exact.** C ∈ ℝ^{8×4096} is passed as the *entire* `encoder_hidden_states`, which is literally what `wan_pre_context_adapter_forward` does. Plan v1 had it spliced into a 512-row padded context; the reviewer's F1 caught that this would have made the targets a different object from the deployment path — the single most consequential catch in either experiment's plan cycle. Without it, H1 would have measured a channel nobody ships.
2. **The conditioning path is bit-verified.** S3's oracle shows the deployed forward *is* the replay operator's `v_cond` — bitwise at fp32, and bitwise at bf16 once the single-owner activation cast is applied (and demonstrably **unequal** without it). So "the optimized context reconstructs the video" and "the deployed head emitting that context would reconstruct the video" are the same statement, not two hopefully-similar ones.
3. **The backbone is frozen and the architecture is untouched.** No new parameters, no new injection points.

**Therefore [INFERENCE]:** the ~128M `pre_context` adapter's architecture and output representation are
*sufficient*. Any future proposal that attacks this problem by adding tokens, widening the head, adding
injection depth, or unfreezing the backbone is attacking a constraint that has been measured not to bind, and
should be required to say why it now does.

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

### 3.2 A second, sharper signature of the same thing [FINDING — not previously recorded anywhere]

B2-probe — B2's fresh-noise-optimized contexts, **locked and replayed under foreign noise** — scores **0.2118**
(all 64) and **0.1652** (matched 8). B2 itself scores **0.1610** (all 64) and **0.1204** (matched 8).
**The contexts perform better on noise they were *not* optimized for than on the noise they *were*.**

That is a strong statement. It rules out the benign reading "B2 simply failed to find anything useful": an
optimizer that found nothing would leave a context that behaves the same everywhere. Instead B2 acquires
something **specifically destructive in its own basin** — it is actively fitting a per-step correction that
makes *that* trajectory worse. **[INFERENCE]** This is the clean fingerprint of an **objective mismatch**, not
an optimization failure.

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

**A cheap check that would test this directly, on already-published data, host-only.** If B2's per-step
*tracking* losses fell normally while its endpoint degraded, that is direct evidence for the mismatch reading
rather than for "the optimizer diverged". Those losses are already written into the published records:
`…/k1/capacity/b2/shard_*/record_*.npz` → `per_step_final_losses [25]`, alongside `final_future_mse`. Reading one
shard settles it. **[JUDGMENT]** I deliberately did not compute it in this docs-only round; I flag it as the
single highest-value follow-up in §7, because it converts §3.3 from a hypothesis into a finding for near-zero cost.

---

## 4. The probe: "collapse" is the wrong word

B1-probe = 0.5254 (DEV), i.e. **57 % of B1's own-basin 0.9227 retained** under foreign noise — and *the highest
absolute foreign-basin SSIM recorded anywhere in the campaign on the full 64-clip cohort*. The worklog's
"collapse to ~0.5" undersells it. This matters for two reasons:

1. **It is a partial, structured failure, not a null result.** Something in an optimized 8-token context is basin-*independent* and worth about 0.53 SSIM. What that component is was never characterized (the plan's cross-example cosine/PCA diagnostic was scheduled for K2 and never ran).
2. **It failed the gate anyway, correctly.** 0.5254 < 0.70. Per the standing R5 note, the **absolute floor is the binding constraint**; the relative conjunct (0.569 < 0.7) is logically redundant on [0,1] SSIM and should never be quoted as an independent reason.

### 4.1 On matched clips, the objective beats the slot [FINDING — the campaign's cleanest single comparison]

Restricting every arm to the **same 8 clips** J1b/J1c used (mean future-SSIM):

| Conditioning, replayed under **foreign** noise | absolute | own-basin | retention |
|---|---|---|---|
| exp_04 A1-probe — **greedy null**, locked | 0.1633 | 0.8911 | 0.183 |
| exp_05 B1-probe — **greedy positive**, locked | 0.4586 | 0.9379 | 0.489 |
| **exp_04 J1c — JOINT null, locked** | **0.4753** | 0.6509 | **0.730** |

The jointly-optimized **null** tensors — from the channel that is *weaker in-basin by a wide margin*
(0.651 vs 0.938) — beat the greedily-optimized **positive** tensors on absolute foreign quality **and** by 1.5×
on retention. **[INFERENCE] On the transfer axis, which objective you optimize matters more than which slot you
optimize.** That is the campaign's central lesson stated in its cleanest form, and it is exp_06's premise.

*Honest caveat:* B1-probe scores 0.4586 on these 8 but 0.5254 on all 64, so the matched-8 subset is unfavourable
to the positive slot; on the full cohort the positive probe would likely lead on absolute (no 64-clip joint-null
counterpart exists). The **retention** ordering (0.730 vs 0.489) is the more robust half of the claim. n = 8, no CIs.

---

## 5. The joint reading with exp_04

### 5.1 Opposite geometries [FINDING]

From a **fresh** basin, on the **same 64 clips** (identical `manifest_hash 433f8691…`):

| | do-nothing control (MSE) | after per-step greedy optimization | effect | clips improved |
|---|---|---|---|---|
| **Null slot** (exp_04, A2-0 → A2) | 4.743 | 0.490 | **9.7× better** | 64/64 |
| **Positive slot** (exp_05, B2-0 → B2) | 1.418 | 5.625 | **4.0× worse** | **0/64** |

The positive slot's *do-nothing* control is 3.3× better than the null slot's — the frozen 8-token context is
already doing useful work, whereas base nulls collapse CFG to identity and do none. Greedy optimization then
**inverts the ordering**. In-basin the picture flips once more: positive 0.9227 vs null 0.8523. The compact
statement: **the positive channel is the more powerful and the more dangerous; the null channel is weaker and
better behaved.**

### 5.2 What must NOT be compared [FINDING — predeclared]

exp_05's H1 PASS and exp_04's G1 FAIL are **not** a slot comparison, for four independent reasons:
(a) **controls differ in kind** — exp_04's A0 collapses CFG (effective w = 1, replaying pivots computed at w = 1:
a near-self-consistent, strong control at MSE 0.335), exp_05's B0 keeps CFG active and replays w = 5 dynamics
against w = 1 pivots (a mismatched, weak control at MSE 0.890). Ratios of 3.6× and 28.6× against those two
controls are different statistics. This was predeclared in plan §4's H1 interpretation note;
(b) **representations differ** — 8 tokens as the whole context vs 16 rows inside a 512-row padded one;
(c) **pivots are not shared** — exp_05 inverts with `C_init` at 8 tokens, so its trajectories, controls and
targets are different objects (pinned by S4 mutant R2); and
(d) **budgets differ** — exp_05 ran at the **adopted J=50**, exp_04 at the **default J=10** (see exp_04
`null_adapter_results.md` §4.4). Only §5.1's within-slot, against-own-control comparisons are matched by construction.

### 5.3 The joint conclusion [INFERENCE]

**Static per-clip conditioning targets are dead as a training signal.** Both slots' fresh-noise arms failed their
predeclared gates (positive 0.1610, worse than nothing; null 0.4973 against a 0.75 bar) and both slots' locked
contexts degrade across basins (positive 0.525, null 0.173). A cached tensor per clip cannot be the supervision,
because the tensor that works is a function of the noise draw. This killed exp_05's K2/K3 and exp_04's P2/P3,
and it should stay killed.

---

## 6. Threats to validity — stated against ourselves

1. **The control was misreported until now.** B0 is **0.3215**, not the "~0.25" carried in the worklog, tracker and CLAUDE.md; the derived multiple is **2.87×**, not 3.6× (results §4.1). No verdict changes — B0 is a ratio denominator, not a gate input, and the ratio conjunct passed 28.6× either way. But it was wrong in three places and propagated, and **[JUDGMENT]** I could not reconstruct its origin. That is a provenance failure in the reporting layer, and it is the reason this document re-derived every number from the artifacts rather than from the worklog.
2. **B1 is an oracle, not a system** (§2.2). exp_05 has **zero** evidence about amortization: K3 never ran, so the question Yixun actually asked — can the head *learn* to emit these? — is unanswered by measurement. Only the *precondition* (the channel can express a solution) is established.
3. **The state-conditioned caveat was flagged at the time and remains live.** K3's regression conditions on cached `z̄_t` states — a *state-conditioned emitter*, not a fixed context — which H2 and the probe do not directly measure. The worklog's own reading called this out and then argued it is "substantially weakened" by B2's failure. **I think that argument is right but weaker than it sounds**, and §7.2 explains why the distinction it points at is load-bearing rather than a technicality.
4. **The 0.2946 anchor cannot carry the "3.13×" claim.** That anchor is a **4-sample** validation over **four correlated windows of one episode** — a wiring check. The qualitative claim (the deployed adapter sits far below what the channel expresses) survives; the ratio should not be quoted as a measurement.
5. **Nobody has looked at the videos.** K1 deliberately published none (ruled not K1-blocking). No human or model has seen what a 0.92-SSIM reconstruction or a 0.16-SSIM fresh-noise failure looks like. Numeric SSIM is a weak proxy for video quality, and a systematic artifact could hide behind these means undetected.
6. **Narrowness.** One dataset, one backbone, one resolution, one guidance scale, one L_pos, DEV/TRAINFIT only (TEST-64 correctly untouched). σ₀ = 1.0 vs the reference's 0.999 (ratified deviation) means prior-art numbers are directional context, not matched comparisons.
7. **What went right, for calibration.** Coverage complete, invalid pairs zero (so the imputation machinery is production-untested), quarantines zero, parity audit clean across 11 positive-slot deltas plus the inherited core, **the adopted recipe actually applied** (the thing exp_04's launcher silently failed to do), and the two measurements that could have silently corrupted everything — the non-no-op bf16 cast and the false fp16→bf16 value-preservation premise — were both caught by measurement rather than assumption. The one real bug (K1-1) was an **inherited stale copy**, an integration failure, not a defect in exp_05's own code.

---

## 7. What these results do and do not license

### 7.1 Licensed

1. **Capacity-first proposals are refuted for this architecture.** 0.9227 SSIM through the frozen backbone via the unchanged deployed channel. More tokens / bigger head / deeper injection / unfreezing all attack a non-binding constraint.
2. **The training signal was the bottleneck** — which is exactly the hypothesis Yixun's Query 1 proposed. exp_05 confirms the premise while refuting the specific remedy it proposed for it.
3. **Do not cache per-clip conditioning targets** (§5.3). The ~14.80 GiB K2 build and everything downstream were correctly not spent.
4. **The deployment-exact representation discipline should be standard.** The F1 catch is why H1 measures something real; any future oracle study must be pinned to the deployed forward the same way.

### 7.2 NOT licensed — and the distinction that motivated exp_06

1. **No claim that the `pre_context` adapter cannot be trained well.** exp_05 refuted one *training signal*, not the architecture — which it in fact vindicated.
2. **No deployment claim.** Every arm is a per-clip oracle; the best transfer number in the experiment (0.5254) misses the 0.70 floor.
3. **No mechanism claim about *why* B2 inverts** — §3.3 is a hypothesis with a cheap unrun test.
4. **THE LOAD-BEARING DISTINCTION.** exp_05's basin finding kills **static per-clip conditioning targets**. It does **NOT** kill the **rollout-loss family**. Flattening these together would have cancelled the campaign's most promising direction, so it is worth stating as a table:

   | | measured and refuted (exp_04 + exp_05) | untested — and where exp_06 lives |
   |---|---|---|
   | **Target form** | a *fixed tensor per clip*, cached and regressed onto | a *state-conditioned emitter* reading `z_t` and re-emitting every step. "The tensor does not transfer across basins" is not a statement about a function that re-reads the state in the new basin |
   | **Objective form** | *per-step greedy tracking* of inversion pivots, with **no endpoint term** | *through-the-sampler* optimization of the rollout endpoint — which exp_04's J1b/J1c showed behaves differently on **both** the capacity and the transfer axes |
   | **Training signal** | teacher-forced regression onto a cache | a differentiable rollout loss computed live, no cache at all |
   | **Evidence status** | gates fired: 0.1610 / 0/64 improved / probes below floor | §4.1: joint nulls reach 0.4753 foreign at **0.730 retention** — 1.5× the retention of greedy positives, from a *weaker* channel |

   Everything both experiments refuted lives in the left column; exp_06 lives in the right one. **[JUDGMENT]**
   This is the most important paragraph in this document and the one most at risk of being flattened into
   "the adapter line failed" by a summary that has not read the arms.

5. **A specific warning about over-reading §5.1.** "Greedy optimization of the positive slot is harmful from fresh noise" is **not** "the positive slot is harmful from fresh noise". The frozen-context control B2-0 reaches 0.2814 doing nothing at all, and B1-probe reaches 0.5254 with a context optimized elsewhere. The channel is not the problem in either case.

---

## 8. Recommended next steps

| Priority | Action | Cost | Why |
|---|---|---|---|
| **1** | Read `per_step_final_losses` from one published `b2` record shard | host-only, minutes | §3.3 — converts the objective-mismatch mechanism from hypothesis to finding on already-paid-for data. Highest value per unit cost in either experiment |
| **2** | Propagate the B0 = 0.3215 correction into the tracker and CLAUDE.md | minutes | §6.1 — a wrong number is live in three handoff documents |
| **3** | Generate comparison videos for a handful of DEV clips across B0 / B1 / B2 / B1-probe from the published records, then write the P4′ HTML report (SOP artifact 12) | ~1 v6e-8-h + host | §6.5 — the qualitative half of the evidence is entirely unexamined; the records are self-contained and replayable |
| **4** | Run the cross-example cosine/PCA structure diagnostic on B1's optimized contexts | host-only | §4 — characterizes the ~0.53-SSIM basin-independent component. Was scheduled for K2 and never ran |
| **5** | Keep S1–S9 on the branch, unmerged | — | Per SOP, a stopped experiment's code stays on its branch. If exp_06 succeeds, S7's trainer + S9's certificate machinery are the natural starting point for a rollout-trained variant |
| **—** | **Do NOT** revive K2/K3 as planned | ~14.80 GiB cache + training | §5.3, and Yixun's STOP is honored. Any revival is a new proposal with its own gates |

---

## 9. Open questions I would put to the reviewer first

1. Is §3.2 (B2-probe > B2 — contexts do better on noise they were *not* optimized for) as strong an argument for objective-mismatch-over-optimization-failure as I claim, or is there a benign explanation?
2. Is §4.1's matched-8 comparison fair, given that the positive probe scores below its 64-clip average on that subset? Should the retention ordering be reported without the absolute ordering?
3. Does §5.2's four-reason non-comparability list adequately protect the H1-PASS-vs-G1-FAIL contrast from being quoted as a slot comparison, or should the results doc refuse to place the two numbers on one page at all?
4. Is the §7.2 static-targets-vs-rollout-family distinction stated strongly enough to survive summarisation?
5. The worklog argued B2's failure "substantially weakens" the state-conditioned-emitter bet. §7.2 argues the two are meaningfully different objects. Which framing should the tracker carry — and does the difference change anything about exp_06's risk?
