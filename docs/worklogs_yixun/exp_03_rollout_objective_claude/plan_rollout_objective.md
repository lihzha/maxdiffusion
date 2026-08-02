# exp_03 `rollout_objective` — Plan v2.2 (Planner)

**v2.1, 2026-08-02** — v2 plus the re-review's four residual closures (data-order stability by
save-schedule construction; D1 slope analysis as a new predeclared script; exact index supports for A/B;
the sigma-trajectory trace operationalized per file). v2 revised v1 against the Codex plan review (REQUEST-REVISION: 2 BLOCKER + 5 MAJOR; all
findings adopted, none rejected — resolutions appended to the review file). The two central upgrades:
**C now implements Yixun's literal `λ·L_A + (1−λ)·L_B` as a true weighted same-batch loss** (v1's stochastic
alternation was not the asked-for objective under finite-run Adam), and the estimand is honestly named
**update-matched, not compute-matched**, with compute reported. The reviewer verified A's corrective-target
math against the actual Euler step rule (exact on-path reduction; min positive σ ≈ 0.1724 so no clamp needed).

## 0. Question and success frame

**Question.** Does training on (or toward) the trajectory the eval actually runs lift rollout reconstruction
above exp_02's measured plateau, at a matched **update** budget?

**Inherited measured facts:** one-step loss → rollout SSIM is linear (`SSIM ≈ 0.9885 − 1.201·loss`,
r = −0.9994) — intercept above the bar, slope = the compounding price; per-frame SSIM starts ≈0.97 for every
window and decays along the rollout (D1); more identical training plateaus (D2, 42σ but 0.0035/2,500 steps
and decelerating); more sampler steps strictly hurt; no CFG exists in this path; capacity and episode
weighting excluded. The LR sweep (running) bounds the optimization-floor lever in parallel.

**Estimand (renamed per review P2).** All trials are **update-matched**: +2,500 optimizer steps from the
exp_02 step-10,000 checkpoint, GBS 256, LR 1e-5, against the LR-sweep `lr1e5c` control arm. This matches
examples and Adam updates, NOT FLOPs: A adds `k_A` stop-grad forwards; B differentiates 2 forwards; C pays
both. Per-trial forward/backward counts and TPU-hours are recorded in `_results.md` and a compute-normalized
reading (gain per TPU-hour) is reported alongside the primary update-matched one.

**Primary metric (corrected per review P3).** **Canonical seed-0 mean SSIM** over the 100-window cohort at
the settled 25-step eval (role `s3_intermediate`) — the exact statistic of exp_02's intermediate passes.
(`m_corr` needs 3 seeds; it returns only at S3 on a winner, via `s3_segment_final`.)

**Success gates (predeclared):**
- **Primary (practical-effect gate):** trial − control ≥ **+0.02** canonical seed-0 mean SSIM. This is ~5×
  the control's expected gain (≈0.0042 via the linear map × the measured 0.0035 loss trend). It is a
  practical-effect threshold, NOT a statistical one; a **paired per-episode bootstrap CI** (10k resamples of
  the 100 episodes, trial−control paired per window) is reported with it, and a gate pass with a CI
  overlapping zero is reported as "gate met, not significant".
- **Mechanism A (temporal):** D1 per-frame decay slope — per-window OLS on frames 1→32 over **all 100
  canonical windows** (trial evals and the control re-eval run `WRITE_VIDEOS=True`), reduction =
  `1 − mean_slope_trial / mean_slope_control`, paired per-episode bootstrap (10,000 resamples, **95% CI,
  bootstrap seed 0**); predeclared threshold ≥ 25%. **Implemented as a NEW committed script**
  `diagnostics/d1_per_frame_slopes.py` (the exp_02 `d1_per_frame_ssim.py` computes aggregate means and
  endpoint drops only — it is NOT reused for this metric); unit tests on synthetic decay curves with known
  slopes; the exp_02 self-validation check (mean-over-frames vs recorded SSIM) is retained.
- **Mechanism B (sigma-trajectory, new per review P7):** fixed-ε latent error vs the ideal interpolant at
  every sigma step of the 25-step rollout, control vs trials (D1 measures temporal decay of decoded frames;
  this measures denoising-trajectory divergence directly — the thing the objectives actually target).
  **Operationalized (re-review P7 residual)** as a NEW probe module `diagnostics/sigma_trajectory_trace.py`:
  imports the extracted step fn (§3), runs the 25-step rollout over the SAME 30-window seeded canonical
  subsample the exp_02 sampling probe used, ε keyed by `window_fold_key(0, episode_id, window_start)` (the
  eval's own keying), records `‖z_{σ_i} − ((1−σ_i)z_gt + σ_i ε)‖² / numel` at every i, writes one immutable
  JSON per checkpoint under the run's `validation_probe_sampling/` root (canonical path rules from the
  exp_02 probe review apply). The existing eval rollout is NOT modified — the trace is its own harness on
  the shared step fn. Unit tests: trace of a perfect velocity oracle is identically zero; trace length/keys;
  path canonicality.
- **Instrument:** the fixed-RNG one-step plain loss on all 1,629 windows per trial checkpoint, including
  exact reproduction of the 10k anchor **0.12227** pre-training and the trial's deviation from the exp_02
  loss→SSIM line (a trial that beats the line demonstrates slope reduction; one that rides it merely moved
  along it).
- **Stretch:** any canonical window ≥ 0.95.

## 1. The three trials (v2 definitions)

Shared: interpolant `z_σ = (1−σ)·z_gt + σ·ε`; network predicts velocity; frame-0 pinned everywhere; sampler
step = the SAME extracted function the eval uses (§3). σ grid = the eval's 25-step shifted grid, indices
`0..24` from highest σ to the last positive σ, with the terminal σ=0 boundary excluded from every draw; all
σ arithmetic in FP32 (min positive grid σ ≈ 0.1724 — reviewer-verified, no clamp needed).

**Exact index supports (re-review P4 residual; direction corrected in v2.2 — the grid is DESCENDING in σ,
index 0 = highest σ, and the eval advances `i → i+1` toward lower σ; start/end indices below follow the
eval's direction):**
- Trial A: draw `k_A ~ Uniform{1, 2}` FIRST, then start `s ~ Uniform{0 .. 24 − k_A}`, end `e = s + k_A`;
  σ_hi = σ[s] (larger), σ_lo = σ[e] (smaller, never terminal since e ≤ 24). The teacher-forced branch
  (prob 1−p_ss) uses the SAME `(s, e)` draw and forms `z_{σ[e]}` from the interpolant — the loss-point
  distribution over σ_lo is identical between branches by construction.
- Trial B: start `s ~ Uniform{0 .. 22}`, path `s → s+1 → s+2 = e` (consecutive grid steps in the eval's own
  advancing direction); σ_hi = σ[s], σ_lo = σ[e].
- Trial C: each batch draws BOTH supports independently (A's and B's), computes both losses on the same
  examples, and combines.

### Trial A — corrective scheduled sampling (renamed per review P6)

1. Draw grid indices `hi > lo` (lo never terminal); form `z_{σ_hi} = (1−σ_hi)z_gt + σ_hi ε`.
2. With probability `p_ss` (linear 0 → **p_ss_max = 0.5** over the first **500** segment steps, then
   constant; schedule keyed to **global step − 10,000** so it survives resume): advance `k_A ~ Uniform{1,2}`
   sampler steps with **stop-gradient**; else stay teacher-forced.
3. One differentiated forward at `σ_lo`; target `v* = (z_{σ_lo} − z_gt)/σ_lo` — the corrective velocity,
   exact under the Euler rule `z_next = z + (σ_next − σ)·v` (contraction `z_next − z_gt = (σ_next/σ)(z − z_gt)`),
   reducing exactly to `ε − z_gt` on-path.

**Honest naming (review P6):** A changes BOTH input exposure and the label. A win supports "corrective
scheduled sampling", not scheduled sampling per se. The S1.5 probe (§4) measures the label's isolated effect
(same-ε label vs corrective label on identical self-generated states) so the confound is quantified before
S2; a pure-scheduled-sampling arm A′ is predeclared as a follow-up ONLY if A wins and isolation matters.

### Trial B — short-horizon rollout loss

1. Draw a grid start (non-terminal); form `z_{σ_hi}` teacher-forced.
2. `k_B = 2` sampler steps with gradients through both forwards (`lax.scan` over the shared step fn,
   `jax.remat` per step).
3. Loss (horizon-normalized per review P1): `MSE(z_{σ_lo}, ẑ_{σ_lo}) / (σ_hi − σ_lo)²` with
   `ẑ_{σ_lo} = (1−σ_lo)z_gt + σ_lo ε` (same ε) — zero at the optimum `v ≡ ε − z_gt` (reviewer-verified);
   without the normalizer, start-index draws re-weight the loss by the nonuniform grid spacing squared.
   Pin masking identical to exp_02's loss; unit test zero-at-optimum, masking parity, and dtype rounding.

### Trial C — the literal combination (changed per review P2/P6)

**True weighted same-batch loss:** every step computes BOTH `L_A` and `L_B` on the same batch and optimizes
`λ·L_A + (1−λ)·L_B`, λ = 0.5, single Adam update. This is exactly Yixun's Query-1 objective — no alternation,
no expectation argument needed. Cost: A's stop-grad forwards + B's differentiated unroll every step (≈2.5–3×
baseline step time; memory peak = B's, since A adds no differentiated activations — S1 verifies). "C wins ⇒
complementary" is licensed only under this literal implementation.

## 2. Outcome readings

| Outcome | Reading |
| --- | --- |
| A wins | Off-path exposure + corrective supervision is the binding fix (label isolation via S1.5 / optional A′) |
| B wins | Gradient credit assignment through the sampler is required |
| C wins | The terms are complementary (literal λ-combination) |
| None beat control | Short-horizon objective changes don't fix 25-step compounding at this scale/budget; exp_02's plateau stands as the recipe family's ceiling; remedies move to inference-time/architecture |

## 3. Implementation plan (per file)

**One-sampler rule with a bitwise gate (review P5).** The single-step sampler fn is extracted from
`generate_wan_side_adapter.py` into `models/wan/overfit100_sampling.py`; trainers and eval both import it.
Because this touches proven eval code, the extraction commit must pass: (i) the existing suite's rollout
tests, (ii) a **bitwise reproduction** of the landed 30-window seed-0 SSIM scalars (probe harness), and the
control arm's eval is RE-RUN at the post-extraction commit into a fresh immutable root — **every exp_03
comparison uses that single post-extraction eval commit** (one-generation rule, learned from exp_02).

- **NEW** `src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py` — `EXP03_TI2V` dispatch (provenance).
  **Binding hook (review P4):** the exp_02 parent binds module-level `_train_step`/`_denoising_loss`; the
  subclass overrides an explicit, tested `_loss_and_step_fns()` hook (parent refactored to route through the
  hook with byte-identical behavior — pinned by a parity test `p_ss=0 ∧ plain objective ≡ exp_02 step`).
  Same `Overfit100TrainState`, same Orbax item shapes (restore-roundtrip test against a real checkpoint
  structure); config keys `exp03_objective ∈ {corrective_ss, rollout_loss, combined}`, `exp03_k_a=2max`,
  `exp03_k_b=2`, `exp03_lambda=0.5`, `exp03_p_ss_max=0.5`, `exp03_p_ss_ramp_steps=500`.
- **RNG discipline (review P2):** purpose-folded keys from `(global_step, purpose)` so ε draws, grid draws,
  p_ss coin flips, and k_A draws are (a) identical across arms at the same step where purposes align, and
  (b) resume-stable.
- **Data-order stability by construction (re-review P2 residual):** the segment saves NO intermediate
  checkpoints — `checkpoint_steps` contains nothing in (10000, 12500) — so ANY preemption restarts the whole
  segment from the step-10,000 checkpoint with the same iterator seed (`seed + 10000`). Data order is
  therefore identical across arms AND across queue retries by construction, not by comparison; the
  first-batch log check remains only as a sanity print. (Cost: a preemption re-runs up to 2,500 steps ≈ 39
  min — acceptable; predeclared.)
- `src/maxdiffusion/configs/base_wan_5b_exp03.yml` — overfit100 config + exp03 keys.
- `bash_scripts/train_wan_exp03.sh` — overfit100 launcher + `EXP03_*` envs.
- Tests (TDD): corrective-target exactness on-path & off-path (vs the Euler contraction, per the reviewer's
  identity), every valid unequal grid interval, terminal-σ exclusion, B zero-at-optimum & horizon
  normalization & masking parity, stop-grad boundaries (grad flows through exactly the intended forwards —
  checked structurally), C's same-batch weighted gradient equals λ∇L_A+(1−λ)∇L_B on a toy model, p_ss ramp
  keyed to global step (resume test), RNG purpose-folding, hook-parity (p_ss=0 ≡ exp_02), config plumbing,
  restore roundtrip.

**Reused unchanged:** train100 dataset + manifest, eval launcher/roles, loss instrument, D1 script,
resume staging, seeding-by-checkpoint-copy procedure (LR-sweep pattern).

## 4. Staged compute (each launch Yixun-gated)

- **S1 — CPU/tiny smoke (v6e-8):** losses finite; hook parity; overhead measured per trial (budgets: A ≤ 1.6×,
  B ≤ 2.5×, C ≤ 3.2× baseline step time — exceeding a budget is a STOP, not a silent accept).
- **S1.5 — no-update discriminator probe (v6e-8, NEW per review P7):** at the 10k checkpoint, no optimizer
  updates: raw + normalized A/B/C losses; grad norms and grad **cosine vs the plain objective**; A's same-ε
  label vs corrective label on identical self-generated states (the P6 confound, quantified); `p_ss=0`
  parity through the new trainer (loss equals exp_02's to fp tolerance); per-sigma-step latent error vs the
  ideal interpolant for the plain model (mechanism-B baseline trace).
- **S1.6 — target-mesh fit probe (v6e-64, one step, NEW per review P4):** compile + one update of B and C at
  GBS 256 on the real mesh (remat/scan certified where it will run). Cheap (minutes).
- **S2 — trials (v6e-64):** A, B, C; +2,500 updates each from the copied step-10,000 checkpoint; then per
  trial: `s3_intermediate` eval with `WRITE_VIDEOS=True` + loss instrument + D1 + sigma-trajectory trace;
  control re-eval at the post-extraction commit (fresh root, videos on) if not already run.
- **S3 (conditional):** extend the winner; only then the formal D11 `s3_segment_final`/full-set machinery.

## 5. Risks (updated)

1. **Interpretability** (the reviewer's headline risk): addressed by C-literal, A-renaming + S1.5 label
   isolation, and the sigma-trajectory metric (temporal D1 alone cannot attribute reduced sampler
   compounding).
2. **Eval-code motion:** gated bitwise (§3); the one-generation rule prevents any cross-commit verdict mixing.
3. **B/C memory & step-time:** S1 budgets + S1.6 mesh probe before any v6e-64 training commitment.
4. **p_ss ramp × short segment:** ramp completes in 500 of 2,500 steps; A trains at full p_ss for 80% of the
   segment. If A shows instability in S1.5 grad diagnostics, p_ss_max drops to 0.25 (predeclared fallback).
5. **Shared control:** `lr1e5c` (LR sweep) is the comparator; its eval is re-run at the post-extraction
   commit so all comparisons share one eval generation.

## 6. Artifacts & bookkeeping

Standard SOP set under `exp_03_rollout_objective_claude/`. Coder = Opus subagent; Reviewer = Codex
`gpt-5.6-sol` xhigh; closed cycles with mutation spot-checks. `_command.md` entries at launch time; every
run's forward/backward counts and TPU-hours recorded for the compute-normalized reading.
