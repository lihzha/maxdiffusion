# exp_03 `rollout_objective` — Plan v1 (Planner)

**Written 2026-08-02.** Answers Query 1: three trials — **A** multi-step / scheduled sampling, **B**
short-horizon rollout loss, **C** the convex combination λ·L_A + (1−λ)·L_B. Surfaced for Codex plan review,
then Yixun approval, before any implementation. All launches approval-gated per announcement 02.

## 0. Question and success frame

**Question.** Does training on (or toward) the trajectory the eval actually runs lift rollout reconstruction
above exp_02's measured plateau, at matched additional compute?

**Inherited facts this plan is built on (exp_02, all measured):**
- One-step loss → rollout SSIM is linear and tight: `SSIM ≈ 0.9885 − 1.201 · loss` (r = −0.9994). The
  intercept clears the 0.95 bar; the ~1.2 slope is the compounding price. exp_03's target is the **slope**.
- Per-frame SSIM starts ≈0.97 for every window and decays monotonically along the 25-step rollout
  (D1). The representation is largely present; the trajectory spends it.
- More identical training is measured to plateau (0.0035 loss / 2,500 steps and decelerating at step 10k);
  more sampler steps make things strictly worse; CFG does not exist in this path; capacity and episode
  weighting are excluded. The parallel LR sweep is bounding the optimization-floor lever.
- The eval contract is settled and reusable verbatim: canonical-100 cohort, m_corr statistic, 25-step
  role-locked rollout, fail-closed admission, the per-frame D1 script, and the fixed-RNG loss instrument.

**Primary success criterion (predeclared).** At matched budget (+2,500 steps from the exp_02 step-10,000
checkpoint, GBS 256, LR 1e-5), a trial **beats the 1e-5 control arm's canonical-100 mean m_corr by ≥ +0.02**
(≈6× the control's expected gain, far outside seed/eval noise) **at the settled 25-step eval**. Secondary
(mechanism) criterion: the D1 per-frame decay slope (frames 1→32) shrinks by ≥ 25% vs control. Stretch:
any window ≥ 0.95 in the canonical cohort (exp_02 never produced one). Formal D11 two-tier claims are NOT
re-run per trial; they return only if a winner is extended (S3).

**Comparator discipline.** Every trial resumes from the SAME exp_02 step-10,000 checkpoint with the SAME
LR/optimizer/data as the LR-sweep control arm (`wan-overfit100-s3ext-lr1e5c-20260802`, +2,500 steps of the
plain objective). One shared control anchors both experiments; no trial gets its own bespoke baseline.

## 1. The three trials — operational definitions

Shared notation: flow-matching interpolant `z_σ = (1−σ)·z_gt + σ·ε`; the network predicts velocity
`v ≈ ε − z_gt`; frame-0 latent is pinned to `z_i0` everywhere (training and rollout, as in exp_02); the
sampler step is the SAME code the eval rollout uses (see §3, "one sampler" rule).

### Trial A — scheduled sampling (input-side; gradients through ONE forward)

Teacher forcing trains the model only on inputs lying exactly on the GT interpolant path; at eval it sees
its own slightly-off-path states, and D1 shows the error compounding. A trains on **self-generated inputs**:

1. Sample `σ_hi > σ_lo` (two points on the eval sigma grid; `σ_hi` from the training σ-distribution,
   `σ_lo` the next grid point(s) down).
2. Form `z_{σ_hi}` from GT + fresh ε (teacher-forced start).
3. With probability `p_ss` (scheduled: 0 → `p_ss_max` over the segment), advance `k_A ∈ {1, 2}` sampler
   steps `σ_hi → σ_lo` using the model itself, **stop-gradient** on these steps; else set
   `z_{σ_lo}` from the GT interpolant (plain objective).
4. One-step velocity loss at `σ_lo` on the (possibly self-generated) input, target `ε_eff − z_gt` — where
   for self-generated inputs the effective target is the velocity that points the CURRENT state back to the
   GT path: `v* = (z_{σ_lo} − z_gt) / σ_lo` (the flow toward `z_gt` from wherever the model actually is;
   reduces exactly to `ε − z_gt` on-path). This "corrective target" is the load-bearing design choice — it
   teaches the model to steer back to the memorized clip from its own drifted states.

Cost: `k_A` extra forwards WITHOUT gradient per A-batch (~+`k_A`·0.5× step time at p_ss=1). Memory: same as
baseline (one differentiated forward). Risk: distribution of self-generated `z_{σ_lo}` early in training is
noisy — mitigated by the p_ss ramp.

### Trial B — short-horizon rollout loss (output-side; gradients through k forwards)

Directly penalize accumulated error over a short window of the real sampler:

1. Sample a start index on the eval sigma grid; form `z_{σ_hi}` from GT + fresh ε.
2. Run `k_B = 2` sampler steps `σ_hi → σ_mid → σ_lo` **with gradients through both forwards**
   (`jax.remat` per step; the two-step unroll is a `lax.scan` of the shared step fn).
3. Loss: `MSE(z_{σ_lo}, ẑ_{σ_lo})` against the SAME-ε GT interpolant point
   `ẑ_{σ_lo} = (1−σ_lo)·z_gt + σ_lo·ε` — i.e., "after k real sampler steps, be where the GT path is",
   with per-element masking of the pinned frame exactly as exp_02's loss masks it.

Cost: ~2× baseline step time, ~2× activation traffic under remat (S1 smoke gate verifies fit at GBS 256 on
v6e-64; fallback `per_device_batch_size` halving + 2× steps is predeclared as NOT a comparability break for
S1 only). Risk: memory; and short-horizon (k=2 of 25) may under-train long-range compounding — that is
exactly what the D1 slope metric will show.

### Trial C — the combination (λ = 0.5 v1)

`L_C = λ·L_A + (1−λ)·L_B`, **implemented as stochastic batch alternation**: each step is an A-step with
probability λ else a B-step. Expectation-equivalent to the weighted sum, and avoids paying A's extra
forwards AND B's unroll memory in one step. λ = 0.5 fixed for v1; a λ sweep only if C wins and is extended.

## 2. Trial-vs-mechanism predictions (what each outcome means)

| Outcome | Reading |
| --- | --- |
| A wins | Exposure bias (off-path inputs) is the binding term; corrective targets suffice without unroll cost |
| B wins | Credit assignment through the sampler matters; scheduled inputs without trajectory gradients are not enough |
| C wins | The terms are complementary (input-distribution + trajectory-gradient) |
| None beat control | Short-horizon objective changes cannot fix 25-step compounding at this scale/budget — points to inference-time or architectural remedies, and exp_02's plateau stands as the recipe family's ceiling |

## 3. Implementation plan (per file)

**One-sampler rule (binding):** the k-step advances in A and B call the *same* step function the eval
rollout uses (extracted, not duplicated), so the trained trajectory operator and the evaluated one are the
same code object. A unit test pins that A/B's step ≡ eval's step on fixed inputs.

- `src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py` — **no behavioral change** (control keeps
  running through the settled path).
- **NEW** `src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py` — subclass of the exp_02 trainer;
  dispatched by `model_type: EXP03_TI2V`; reads `exp03_objective ∈ {scheduled_sampling, rollout_loss,
  combined}`, `exp03_k_a`, `exp03_k_b`, `exp03_lambda`, `exp03_p_ss_max`, `exp03_p_ss_ramp_steps`. Owns the
  three loss builders; A's corrective target and B's same-ε target as above; frame-0 pinning inside every
  advanced step; stop-grad boundaries explicit.
- **NEW** shared step extraction: the single-step sampler fn moves from `generate_wan_side_adapter.py`'s
  rollout into a small module both import (`models/wan/overfit100_sampling.py`), byte-behavior pinned by
  tests on both call sites (the eval path must remain bit-identical — its bitwise reproducibility is a
  proven asset we must not lose).
- `src/maxdiffusion/configs/base_wan_5b_exp03.yml` — clone of the overfit100 config + the exp03 keys;
  same data/manifest/eval keys so the entire exp_02 eval stack runs on exp_03 runs unchanged.
- `bash_scripts/train_wan_exp03.sh` — clone of the overfit100 training launcher + `EXP03_*` envs.
- Tests (TDD, `src/maxdiffusion/tests/worklogs_yixun/test_exp03_*`): corrective-target math (on-path
  reduction to the plain target; off-path direction checks), same-ε target correctness, stop-grad boundaries
  (A's advance carries no grad; B's carries grad through exactly k forwards — checked via `jax.grad`
  structure), p_ss ramp schedule, λ alternation statistics, one-sampler equivalence, config plumbing,
  masking parity with exp_02's loss, memory-shape smoke (CPU).

**Reused unchanged:** train100 dataset + manifest, eval launcher + D11 machinery, loss instrument,
per-frame D1 script, resume staging. Evals run at role `s3_intermediate` on each trial's RUN_NAME.

## 4. Staged compute (every launch Yixun-gated)

- **S1 smoke (v6e-8):** 30 steps per trial, tiny batch. Gates: losses finite; B fits memory at target batch
  (else predeclared fallback); A's self-generated states in sane stats envelope; grad norms comparable to
  baseline; step-time overhead measured (predeclared budget: A ≤ 1.6×, B ≤ 2.5× baseline step time).
- **S2 trials (v6e-64):** three runs, each +2,500 steps from the exp_02 step-10,000 checkpoint (seeded the
  LR-sweep way), LR 1e-5, GBS 256. Then per trial: `s3_intermediate` eval (seed-0, canonical-100) + loss
  instrument + D1 per-frame script. Compare against the shared lr1e5c control.
- **S3 (conditional):** extend the winner (if the ≥+0.02 criterion is met) and only then re-run the formal
  D11 two-tier verdict on the extended run.

Budget note: S2 ≈ 3 × (39 min × overhead) v6e-64 + 3 × ~30 min v6e-8 evals; well under one exp_02 day.

## 5. Risks / open questions (for the reviewer)

1. **A's corrective target divides by σ_lo** — near-zero σ_lo inflates the target; mitigation: exclude the
   lowest grid point from σ_lo candidates (predeclared) or clamp. Reviewer: check the formulation.
2. **B's same-ε target** assumes the sampler should track the *stochastic* interpolant point rather than the
   conditional mean; for memorization of a single (z_gt, text) pair this is the intended fixed point, but
   the reviewer should audit the flow-matching math.
3. **Off-path velocity semantics (A):** `v* = (z − z_gt)/σ` implies a σ-parameterized contraction field;
   confirm consistency with the scheduler's update rule so the composed step is a contraction toward z_gt.
4. **Eval comparability:** trials change the objective, not the eval; the 25-step contract stays locked.
   Any trial-specific eval temptation (e.g., "B looks better at 2 steps") is out of scope for the verdict.
5. **The control already exists** (LR-sweep lr1e5c). If its result arrives first, S2 trials are judged
   against the measured number; no re-run.

## 6. Artifacts & bookkeeping

Standard SOP set under `docs/worklogs_yixun/exp_03_rollout_objective_claude/` (this plan, reviews with
strengthening records, `_command.md` at launch time, worklog, params, results, analysis, HTML report).
Coder = Opus subagent per round; reviewer = Codex `gpt-5.6-sol` xhigh; closed cycles with mutation
spot-checks as in exp_02.
