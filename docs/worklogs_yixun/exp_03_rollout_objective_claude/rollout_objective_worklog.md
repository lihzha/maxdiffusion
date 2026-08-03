# exp_03 `rollout_objective` — worklog (append-only lab notebook)

## 2026-08-02T17:10:00Z — Experiment scaffolded

- **Goal** — Reserve exp_03; record Query 1; establish branch/worktree per SOP.
- **Version Control** — branch `claude-exp_03_rollout_objective-20260802` off `yixun-dev` @ `0263fa7`,
  with `claude-exp_02_overfit100-20260728` merged in (needs its trainer/eval/dataset machinery; `yixun-dev`
  stays clean of experiment code — exp_01 precedent). Worktree
  `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective`.
- **Result** — scaffolded; plan v1 next, then Codex plan review, then Yixun approval before any code.
- **Next** — Planner writes `plan_rollout_objective.md`.

## 2026-08-02T18:55:00Z — Plan v3.1 APPROVED by Yixun ("approve exp_03 plan")

- **Goal** — Record the approval that gates implementation; open cycle A.
- **Result** — Plan cycle closed at v3.1 (six reviewer passes, final APPROVE-PLAN); Yixun approval on
  record. Implementation order per plan §3: (1) sampler-step extraction w/ bitwise gate, (2) exp03 trainer
  + binding hook, (3) losses A/B/C + tests, (4) config + launcher, (5) diagnostics scripts (D1 slopes,
  sigma trace). TPU stages S1/S1.5/S1.6/S2a/S2b each separately Yixun-gated at launch time.
- **Next** — Cycle A round 1: extraction (Coder), then focused Codex review.

## 2026-08-02T21:40:00Z — Cycle A round 1: sampler-step extraction (bitwise-inert) + diagnostics skeletons

- **Goal** — Plan §3's one-sampler rule, and the two predeclared diagnostics, before any objective code.
- **What landed**
  - **NEW `src/maxdiffusion/models/wan/overfit100_sampling.py`** — `overfit100_sampler_grid`,
    `overfit100_step_timestep`, `overfit100_euler_update`, `overfit100_sampler_step`. Pure motion out of
    `generate_wan_side_adapter.py`. The velocity model enters as a `velocity_fn` callable, so the eval
    closes over the frozen transformer (`deterministic=True`), the adapter path closes over its CFG
    branch, and an exp_03 trainer will close over the differentiated params — one operator, three callers.
    No Python side effects inside, explicit args in / state out: safe to `scan` and `remat`.
  - **BOTH eval rollouts rewired** — `_rollout_overfit100_sample` AND its exp_01 sibling `_rollout_sample`.
    Converting only the overfit100 one would have left a second copy of the step in the same file, which
    is exactly what the rule forbids; the adapter/CFG velocity branch did not move, only the arithmetic
    around it.
  - **NEW `src/maxdiffusion/diagnostics_exp03/`** (package): `d1_per_frame_slopes.py` (Mechanism A) and
    `sigma_trajectory_trace.py` (Mechanism B). Placed in the package, not under `docs/`, because the trace
    imports the extracted step, runs on the pod, and must ship in the code tarball; the exp_02 D1 script
    could live under `docs/` only because it is a laptop-side CSV/MP4 reader.
- **Bitwise discipline** — the extraction is pinned three ways: (1) exact-equality parity against a
  VERBATIM copy of the pre-extraction step (every grid index, float32 and bfloat16, single step and the
  full 25-step chain — `array_equal`, not `allclose`); (2) an AST test that each rollout calls the
  extracted step and that no duplicate step/grid construction survives anywhere in the evaluator;
  (3) the untouched synthetic-driver tests that execute the real loop. The on-hardware gate against the
  landed 30-window scalars stays where the plan put it — S1.5.
- **Predeclared, before any trial ran** — D1: OLS frames 1→32 (frame 0 excluded as the free pinned
  condition), reduction `1 - mean_slope_trial/mean_slope_control`, paired per-episode bootstrap
  10,000 resamples / 95% CI / seed 0, threshold 0.25, exp_02 self-validation retained. Trace: fixed-ε
  interpolant error per sigma step, 30-window probe cohort reused, ε = `window_fold_key(0, ep, ws)`,
  immutable JSON under the canonical `validation_probe_sampling/` root with the probe review's
  `step_`-component refusal.
- **Result** — suite 1271 passed / 2 skipped (baseline 1236; +35). Eight mutants killed, incl. pin-order
  drift, dtype-cast drift (bf16-only, as designed), a duplicate step left in the evaluator, oracle-zero
  broken two ways, slope sign, frame-0 leak and an ignored bootstrap seed.
- **Two existing tests were rewired** (declared, not silent): the probe's sigma-schedule test referenced
  `gen.build_rollout_sigmas`, which moved; and the exp_02 "rollout ignores the pipeline sampling knobs"
  test byte-matched `num_inference_steps`, which is now the extracted grid builder's PARAMETER name. Both
  now assert the same claim at the new seam — the pipeline knob is still never read.
- **Next** — focused Codex review of this round; then cycle A round 2 (exp03 trainer + binding hook).

## 2026-08-02T23:05:00Z — Cycle A round 1 STRENGTHENING (Codex REQUEST-REVISION on 8ccaf3a)

- **Verdict** — 2 HIGH + 3 MEDIUM, all hardening; the motion itself VERIFIED inert at both call sites
  and the `_rollout_sample` scope extension ruled sound. Record + strengthening response in
  `rollout_objective_codex_code_extraction_review.md`.
- **Closed** — (1) D1 is fail-closed everywhere: equal 33-frame videos, exact 1-32 fit, no skipped
  windows, identical 100-window cohorts with unique matching episode ids, both aggregations required;
  (2) the trace's design is approval-pinned (seed 0 / 30 windows / 25 steps) *before* the ~5B load,
  with cohort and grid lengths verified after; (3) the bf16 finite-precision floor is measured by an
  exact-velocity re-run and travels in the same JSON entry as the error it bounds — reported metric
  predeclared as RAW, reference dtype float32 and documented; (4) chain parity now fp32 AND bf16, plus
  a verbatim whole-function pre-extraction reference for the adapter path at guide scales 1 and 5 and
  for the FULL_FT branch; (5) the AST guard binds structure — one `_body`, one returned shared-step
  call, one grid binding, one `fori_loop` — so the reviewer's named evasions fail it.
- **Result** — suite 1296 passed / 2 skipped (+25). Ten mutants killed incl. all three evasive AST
  ones; one mutant proven **equivalent** (fp32-then-round == bf16 subtract, verified bit-identical over
  4,096 random pairs) and reported rather than chased.
- **Next** — re-review of this SHA; round 2 (exp03 trainer + binding hook) does not stack until it
  passes.

## 2026-08-03T01:20:00Z — Cycle A round 2: exp_03 trainer + binding hook (ctrl0-capable)

- **Goal** — make ctrl0 runnable end-to-end through a trainer of its own, with the A/B/C objectives'
  seats built but empty (round 3 fills them).
- **Parent refactor (byte-identical)** — `_train_step` is now `_make_train_step(loss_fn)`, and
  `WanTI2VOverfit100Trainer._loss_and_step_fns()` is the hook `start_training` routes through. The
  module-level `_train_step` stayed a *thin dispatch* into the factory rather than a closure bound at
  import, so this module's `_denoising_loss` remains LATE-bound — exp_02's suite pins that property
  (it patches the name and asserts the spy fires) and it is settled behaviour. exp_02's suite passes
  untouched.
- **NEW `trainers/wan_ti2v_exp03_trainer.py`** — `Exp03Trainer`, dispatched by `model_type:
  EXP03_TI2V`. `exp03_objective: control` returns the parent's `(_denoising_loss, _train_step)` **by
  identity**, which is what makes ctrl0 a replication guard rather than a second implementation; the
  three trials raise `NotImplementedError` at startup (before the ~5B load) so a mistyped arm cannot
  quietly train the control under a trial's run name.
- **RNG discipline (delta-review 2)** — `exp03_aux_key(seed, global_step, purpose)` derives auxiliary
  keys from `key(seed + 1_000_003)` folded with the step and a sha256-of-the-NAME purpose id. Derived,
  never split off the training stream; keyed on the global step, so resume-stable; same key in every
  arm at the same (step, purpose). Tested: the exp_03 control run's shared draws equal the
  pre-refactor trainer's exactly, over a multi-step run and across a resume boundary, and interleaving
  aux draws changes nothing.
- **Compatibility** — same `Overfit100TrainState`, same Orbax items (`params`/`opt_state`/`step`), same
  preflights, all asserted by identity of the inherited methods; real save→corrupt→restore roundtrip
  through the production `_save_checkpoint`/`_maybe_restore`, plus the empty-dir (Tier-2 from-init)
  path returning step 0.
- **Config + launcher** — `configs/base_wan_5b_exp03.yml` is exp_02's config plus the six `exp03_*`
  keys, with `model_type` the ONLY changed value (asserted key-by-key); `bash_scripts/train_wan_exp03.sh`
  clones the overfit100 training launcher with `EXP03_*` passthroughs, keeping the manifest pin, HF
  prefetch, local-only snapshot resolution and COMMIT export. No ffmpeg block — training arm, loss-arm
  precedent, justified in a comment.
- **Result** — suite 1331 passed / 2 skipped (+35). Six mutants killed: hook bypassed; aux key split
  off the training root; control loss drifted; control arm returning a copy instead of the parent's
  functions; dispatch dropped; unimplemented objective silently falling back to the control.
- **Next** — focused Codex review of round 2; then round 3 (the A/B/C losses).

## 2026-08-02T03:40:00Z — Cycle A round 2 STRENGTHENING (Codex REQUEST-REVISION on c0aaaa2)

- **Verdict** — 1 BLOCKER + 2 MAJOR, round 3 NO-GO until re-reviewed. Record + response in
  `rollout_objective_codex_code_trainer_review.md`.
- **Closed** — (1) the global step now crosses the jit boundary as a dynamic scalar (4-arg compiled
  adapter, `jnp.asarray(step, int32)` from the LOOP, never `state.step`), and `exp03_aux_key` folds a
  traced uint32 with no `int()` coercion; resume-safety is proven end-to-end through the production
  save/restore path. (2) A JIT parity certificate: both steps compiled with AdamW, four cached calls,
  params + full optimizer state + `state.step` + every metric + rng exact, one trace each, matching
  jaxprs. (3) The launcher drift test parses default and override maps and compares them
  bidirectionally under a tight allowlist.
- **Compatibility** — when no step is threaded the loss keeps exp_02's exact six-argument call shape,
  so the exp_02 spy test passes untouched; both shapes pinned.
- **Result** — suite 1340 passed / 2 skipped (+9). Six mutants killed incl. both named launcher
  examples; a meta-mutation confirms the allowlist's tightness is load-bearing.
- **Next** — re-review; round 3 (A/B/C losses) still gated on it.

## 2026-08-02T05:05:00Z — Round-2 residual: drift parser covers export/multiline defaults

- **Trigger** — re-review of `a072ae2`: BLOCKER + JIT-parity MAJOR closed, conditional `global_step`
  approved; one residual on the launcher drift parser kept round 3 NO-GO.
- **Fix** — `_logical_lines` joins backslash continuations, `_ASSIGNMENT` accepts an `export` prefix
  and uses `re.DOTALL`, so `export`-prefixed defaults and the ~25-line `LIBTPU_INIT_ARGS` are now
  compared verbatim as single entries. Bidirectional comparison + tight allowlist unchanged; parser
  non-vacuity asserted (>10 `--xla` and an embedded newline inside the joined value).
- **Mutations** — the reviewer's two examples both now FAIL as required (JAX_PLATFORMS default
  drifted; one XLA flag inside LIBTPU_INIT_ARGS drifted), plus a control (exported literal drifted).
- **Result** — suite 1340 passed / 2 skipped (unchanged; the round widened an existing test).
- **Next** — closing micro-pass by the coordinator; then round 3 (A/B/C losses).

## 2026-08-02T07:30:00Z — Cycle A round 3: the A/B/C objectives

- **Goal** — the scientific heart: corrective scheduled sampling (A), short-horizon rollout loss (B),
  and the literal weighted combination (C), replacing round 2's `NotImplementedError` stubs.
- **A (`corrective_ss`)** — `k_A ~ U{1..exp03_k_a}` drawn FIRST, then `s ~ U{0..24-k_A}`, `e = s+k_A`
  (plan v2.2's direction-corrected supports; the terminal index 25 is unreachable by construction).
  Teacher-forced `z_{σ[s]}` from the SHARED-stream ε; with probability `p_ss` the state is advanced
  `k_A` steps of the EXTRACTED sampler under `stop_gradient`, else the interpolant at `σ[e]` — the
  same `(s, e)` draw either way, so the loss point's distribution is identical between branches. One
  differentiated forward at `σ[e]`, target `v* = (z_lo − z_gt)/σ_lo`, exp_02's exact pin masking.
- **B (`rollout_loss`)** — teacher-forced start, `k_B = 2` extracted-sampler steps under `lax.scan`
  with `jax.remat` per step, gradients through both forwards, endpoint MSE against the same-ε ideal
  interpolant divided by `(σ_hi − σ_lo)²`.
- **C (`combined`)** — both losses on the same batch with independently drawn supports (different
  aux purposes ⇒ structural independence), `λ·L_A + (1−λ)·L_B`, one Adam update.
- **RNG** — new purposes `k_a_draw` and `index_support_rollout` beside the existing `p_ss_coin` /
  `index_support`; the shared stream is split in exp_02's exact order so an arm's ε and dropout key at
  a step are the control's (`step_rng` is deliberately unused). `self_gen_noise` stays declared but
  unused — A's self-generated state comes from the sampler on the SAME ε, which is what keeps the two
  branches comparable.
- **Result** — suite 1371 passed / 2 skipped (+31). Six mutants killed: corrective denominator σ_hi;
  stop-grad dropped; horizon normalization dropped; C's weights swapped; support off-by-one reaching
  the terminal σ; ramp ignoring the origin. Control parity still bites (a drifted control loss fails).
- **Flagged for review** — (a) `exp03_ramp_origin` (new config key, default 0) generalizes the plan's
  "global step − 10,000" so Tier 1 and Tier 2 share one expression; (b) supports are drawn PER BATCH,
  not per example (the plan's own wording for C, and per-example indices would break the extracted
  step's Euler broadcast); (c) differentiated forwards use exp_02's training convention
  (`deterministic=False` + dropout), stop-gradient state-producing forwards use the eval convention.
- **Next** — focused Codex review of round 3.

## 2026-08-02T09:15:00Z — Round-3 STRENGTHENING (Codex REQUEST-REVISION on 01c6362)

- **Verdict** — 2 MAJOR + 1 MINOR + 1 LOW; core math verified correct (A entirely; B apart from the
  convention; C literal). D1 BLESSED (Tier-1 launches must pass `EXP03_RAMP_ORIGIN=10000`
  explicitly), D2 BLESSED with an S1.5 caveat (quantify support-gradient variance), D3: A sound,
  B amend. Record + response in `rollout_objective_codex_code_losses_review.md`.
- **Closed** — (1) B and C's B-term now differentiate the DETERMINISTIC eval sampler; (2) B's tests
  detect rather than reproduce the convention — an independently written unroll, a stub that observes
  `deterministic`, fp32 AND production bf16, loss exact in both and gradients exact in fp32 with the
  bf16 scan-accumulation-order difference stated and bounded; (3) exact support certificates against
  independently constructed `randint` draws from the named keys, and the corrective identity over
  every positive sigma 0..24; (4) `self_gen_noise` removed, stale "unimplemented" text scrubbed,
  `EXP03_IMPLEMENTED_OBJECTIVES` derived from the dispatch table, explicit `NotImplementedError` and
  a refusal test that exercises the real path. The tautological ramp assertion is now a
  counterfactual.
- **Result** — suite 1399 passed / 2 skipped (+28). Five new mutants killed incl. "guard the guard";
  round-3's own mutants re-verified after the test rewrite.
- **Next** — focused re-review; then the S1 smoke package (explicit ramp origins + the A/B/C overhead
  STOP budgets).

## 2026-08-02T10:40:00Z — Round-3 residuals: bf16 parameter fixture, comment corrections

- **Trigger** — re-review of `371816c` (new reviewer account, 63 tests independently rerun): the
  bf16 >50x substitution ACCEPTED (measured 171x), two residuals left.
- **Closed** — (1) the bf16 case is now end-to-end: a bfloat16 PARAMETER, with dtype assertions on
  parameters and all three gradient trees, and the expectation derived from the config through
  production's `_dtype` converter (the first attempt, keyed to the fixture's own variable, let a
  revert-to-fp32 mutant survive); (2) the two stale comments corrected — epsilon comes from the
  shared stream, and only A's final supervised forward uses training mode.
- **Measured** — with bf16 parameters the eval-reference gradient gap is 0.0 (bf16 rounds the
  scan-vs-unroll accumulation difference away) against 0.0562 for the training convention.
- **Result** — suite 1399 passed / 2 skipped (unchanged). Three bf16/D3 mutants killed.
- **S1 inputs** — trace-time forward counts and graph sizes recorded in the review doc as overhead
  EXPECTATIONS (control 1 fwd / 80 eqns; A 2 traced fwd / 1.79x; B 1 traced fwd / 1.38x; C 3 / 3.24x),
  with the caveat that loop bodies trace once and that CPU wall-clock on a one-tanh stub predicts
  nothing. Real budgets (A 1.6x, B 2.5x, C 3.2x; exceeding = STOP) get measured in S1.
- **Next** — closing micro-review, then the S1 package (ramp origins Tier 1 = 10000 / Tier 2 = 0,
  STOP budgets, D2's support-variance requirement for S1.5) for Yixun's approval.

## 2026-08-03T00:30:00Z — S1 fallout: combined-arm NaN diagnosis (step-8 draw) — PARTIAL root cause

- **S1 results** — control PASS (1.786 steps/s); corrective_ss PASS (1.47x); rollout_loss finite but
  2.56x vs the 2.5x budget (STOP, marginal — S1.6 re-measures at scale, no code change); **combined
  loss=nan from step 8 of 30** (step 7 finite at 1.972) AND 4.23x vs 3.2x.
- **First question, settled** — C's support purposes are **identical** to the standalone arms'
  (`k_a_draw`, `index_support`, `index_support_rollout`; one draw site each, asserted by test), so at
  global step 8 C drew exactly what A's arm and B's arm drew at their own step 8 — and **both were
  finite there** (A 2.158888, B 0.503659). The NaN is therefore an **interaction of the two terms in
  one trace, not an unlucky draw**.
- **Exact step-8 draw, reconstructed and pinned** — `k_A=2, s_A=0, e_A=2` (σ_hi = **1.0**, the top of
  the grid; the FIRST of the 30 steps to start there), `s_B=16, e_B=18`, coin 0.4463 ≥ p_ss 0.40 ⇒
  **teacher-forced**. The learning rate at step 8 was 2.8e-7 with grad_norm 8.277 at step 7, so a
  gradient-driven parameter divergence is arithmetically impossible: the step-8 computation itself
  produced the NaN.
- **Bisection at production shapes/dtypes** — `[B,48,9,12,20]` bf16: A's advance at the step-8
  support is finite for velocity scales 1..1e3; the corrective target is bounded (σ_lo ≥ 0.1724, max
  |v*| ~10); B's normalizer is finite at every one of the 23 starts, computed in fp32, peaking at
  **3422x at s=0** (in bf16 the same subtraction would land on 0.015625 and inflate it to 4096x —
  which is why it must stay fp32). **No structurally singular support exists.**
- **What changed** — the riskiest construct is gone: A's advance was `fori_loop` with **traced
  bounds** (lowering to dynamic control flow inside the differentiated trace, with a graph whose
  shape depends on the step's draw). It is now a fixed-length `k_max`-step unroll with a select,
  proven to pick exactly the k-th state (exact) and to match the old loop to ≤1 ULP. Plus per-term
  metrics (`loss_a`, `loss_b`, `p_ss`, `k_a`, `sigma_hi`, `horizon_sq`) so a recurrence localises
  itself in the log instead of needing another run.
- **Honest limitation** — the NaN was **not reproduced on CPU**: it needs the real 5B forward, and
  every arithmetic path enumerated here is finite. The fix removes the one construct that plausibly
  differs between C and the arms; the per-term metrics make the next smoke decisive either way.
- **Result** — suite 1407 passed / 2 skipped (+8). Three mutants killed (dynamic-bound loop restored;
  per-term metrics dropped; support formula shifted).
- **Next** — focused review; the C re-smoke needs a fresh launch approval.

## 2026-08-03T02:10:00Z — S1-fix STRENGTHENING (Codex 2 BLOCKER + 3 MAJOR on 76ff476)

- **BLOCKER 1, off-by-one** — the loop logs `step + 1` while passing the zero-based `global_step`;
  the LR pins it (7 x 4e-8 = 2.8e-7 on the "step 8/30" line). The failing step is **global_step 7**.
  Re-derived: `k_A=2, s_A=1, e_A=3` (sigma_lo 0.97345), `s_B=10, e_B=12`, coin **0.2878 < p_ss 0.35**
  ⇒ the failing step took the **SELF-GENERATED** branch. The first reconstruction (step 8,
  teacher-forced, top-of-grid) was wrong.
- **BLOCKER 2, logging** — whitelist removed (every aux key is forwarded), all sixteen promised
  metrics printed and sent to W&B, fail-fast `NonFiniteStepError` before the next batch, and a
  separate no-update `exp03_frozen_replay` with per-term grad norms / max-abs / finite-leaf counts /
  A-B gradient cosine.
- **MAJORs** — "interaction, not draw" withdrawn as a conclusion (different histories at the same
  step); sweep now enumerates all 2,162 legal combinations with scope stated (forward-only, toy
  model); A's cost rise from mean 1.5 to 2 advances recorded (may reach the 1.6x STOP); the ULP
  wording replaced by the tolerance actually asserted.
- **Result** — suite 1416 passed / 2 skipped (+9); five mutants killed.
- **Next** — re-review, then the re-smoke package under Yixun's standing grant.

## 2026-08-03T04:00:00Z — S1-fix CLOSING round (Codex: 2 residuals + 1 conditional on a73cd53)

- **Emit before raise** — the finiteness check ran before the log block, so a NaN aborted before
  printing the line naming it. Split into `step_finite_failures` + a single `report_step` seam the
  loop delegates to: a non-finite step forces the full diagnostic line (prefixed `NON-FINITE `,
  ignoring `log_period`) and the W&B entry, THEN raises. Tested with a fake logger + fake W&B.
- **True cross-product** — L_C evaluated over all 2,162 legal triples, count derived from the
  collected values (an arithmetic-only assertion let the skip mutant survive first time round). The
  INTERACTION conclusion is deleted from the test.
- **Pre-step snapshot** — `exp03_snapshot_before_step` (default -1) writes params/opt_state + rng +
  batch + manifest immediately before the named zero-based step; config key and launcher passthrough
  added and pinned.
- **Result** — suite 1420 passed / 2 skipped (+4); five mutants killed.
- **Next** — re-review; on APPROVE, ONE contemporaneous v6e-8 cohort (control + A timing companion +
  faithful C replay, LOG_PERIOD=1, strict 1.6x/3.2x gates, snapshot armed at step 7).

## 2026-08-03T05:30:00Z — S1-fix FINAL micro-round (closing re-review residuals)

- **Host-scoped logging** — non-primary hosts received `log_period=0`, which the emitter turned into
  every step. `report_step` now takes `is_primary`: process 0 alone writes lines (periodic or
  forced) and a zero period means never; every host still RAISES, because the flags are replicated
  and a mesh where only process 0 stopped would hang.
- **Stale sentence** deleted (the "interaction" wording); frozen-replay-only phrasing stands.
- **Snapshot correctness** — the Orbax save is collective and now runs on every host (only the
  rng/batch/manifest extras are process-0 work), and it is AWAITED via
  `wait=ckpt_mgr.wait_until_finished` before the armed step executes. The test drives the real
  checkpoint path on a local tmp dir and asserts the order save -> wait -> step, plus a genuine
  restore of the saved params.
- **Result** — suite 1424 passed / 2 skipped (+4); four mutants killed (await dropped; nonzero-host
  logging; collective gated to p0; armed block gated to p0).
- **Next** — final re-review; on APPROVE the re-smoke cohort launches per the confirmed spec.

## 2026-08-03T06:40:00Z — S1-fix LAST residuals (zero-period loop guard, addressable-only extras)

- **Zero period, end to end** — the loop computed `% config.log_period` before the emitter saw it,
  so a non-primary host's period of 0 raised ZeroDivisionError. One `is_log_due` helper now answers
  it in both places; 0 means never. Tested on the helper, on the loop source, and behaviourally at
  `log_period=0` for both host roles.
- **Addressable-only snapshot extras** — a production batch is a GLOBAL array of per-host shards, so
  `np.asarray` raises; it was also running before the `is_primary` gate, which would have crashed
  every host before the collective save. Materialization is now inside the primary branch and reads
  only `addressable_shards`, saving `<name>__shard<i>` with each shard's global index in the
  manifest. Chose shard+manifest over `process_allgather` deliberately: a gather is a collective, and
  this runs one step before an expected failure.
- **Result** — suite 1427 passed / 2 skipped (+3); four mutants killed.
- **Next** — re-review; then the re-smoke cohort per the confirmed spec.

## 2026-08-03T07:30:00Z — Snapshot gated to single-host (eval-resume precedent)

- **Finding** — primary-only shards make multi-host reconstruction impossible (other hosts' shards
  never saved, no reassembler, no multi-device round-trip test).
- **Resolution (Planner's call, precedented)** — the snapshot is active only when
  `jax.process_count() == 1`; a multi-host run logs a reason naming `process_count`, skips the
  snapshot and CONTINUES. Rationale predeclared in the module docstring: on one host the primary's
  addressable shards are the whole batch; on many hosts a file holding a fraction of a batch is
  worse than no file. Complete multi-host capture is its own reviewed round if an S2-scale NaN ever
  makes it necessary. The v6e-8 C re-smoke is single-host, so the gate costs the diagnosis nothing.
- **Result** — suite 1431 passed / 2 skipped (+4); three mutants killed (gate removed; gate always
  allowing; reason not logged).
- **Next** — micro-review, then the re-smoke cohort per the confirmed spec.

## 2026-08-03T09:00:00Z — S1.5 probe driver + launcher (dual-state, no-update)

- **Built** — `src/maxdiffusion/probe_exp03_s1_5.py` + `bash_scripts/probe_exp03_s1_5.sh`. The probe
  applies NO optimizer updates (bit-level fingerprint before/after, asserted per state) and runs at
  BOTH states: the exp_02 step-10,000 checkpoint (restore path; refuses to fall back to init if the
  directory is empty) and the pretrained init (empty-checkpoint path through the same code).
- **Measures, per state over K=8 batches** — (1) per-objective losses / grad norms / max-abs /
  finite-leaf counts / **cosine vs the plain objective**, via `exp03_frozen_replay` EXTENDED (control
  arm + cosines added there) rather than duplicated; (2) A's label isolation, corrective vs same-ε on
  identical states; (3) `p_ss=0` parity in loss AND gradient at 1e-5 relative; (4) support-gradient
  variance by the law of total variance — each batch replayed under M=4 support draws, within-batch =
  support term, between-batch = data term, plus a per-objective gradient-noise scale; (5) the
  mechanism-B sigma traces at both states under their own canonical path rules.
- **Pinned** — K, M and the two state labels are approval constants with hostile-override refusal;
  output is one immutable JSON per state under `validation_probe_sampling/` with the `step_`-component
  refusal; the launcher requires `CHECKPOINT_DIR`, defaults `EXP03_RAMP_ORIGIN=10000` (Tier 1 per D1),
  keeps the manifest pin / prefetch / local-only resolution / COMMIT export, and carries no ffmpeg
  block (it scores gradients, not pixels — loss-arm precedent).
- **Result** — suite 1462 passed / 2 skipped (+31). Six mutants killed.
- **Next** — focused review, then S1.5 launches under the standing grant.

## 2026-08-03T12:00:00Z — S1.5 STRENGTHENING (Codex 5 BLOCKER + 3 MAJOR on 210e7b1)

- **Wired** — label isolation, conditional parity (against a fixed-support comparator, with the
  production-control difference reported not gated), forced p_ss=1 A/C diagnostics, per-batch rows,
  and B's raw endpoint MSE + fp32 horizon now all reach the artifact.
- **Salted variance** — `exp03_aux_key(salt=...)` consumed only by the support purposes and folded
  only when non-zero (existing draws bit-identical); the M draws vary the salt alone, so the
  within-batch term is the sigma support and nothing else. Between term renamed
  `batch_shared_rng_variance`, population + unbiased estimates both reported with a finite-M note.
- **In-memory traces** — `trace.run_trace` is gone; the already-built state is traced, so the init
  trace is the init and only one 5B model is ever live.
- **State pins** — required step 10000 (not `latest_step()`), init through the production
  empty-restore path, iterator seed `seed + start_step`, per-state ramp origins (checkpoint 10000 /
  init 0) in a `S1_5_STATE_PLAN` rather than one launcher env.
- **5B-executable** — leafwise on-device dots/norms and a streaming `_TreeWelford` (one tree +
  scalar); the ~1.28 TB retained-and-flattened shape is gone and forbidden by test.
- **Also** — `include_control` default restored to False, fp32 tree cosine restored, `or True`
  deleted (with a meta-test), sha256-per-leaf fingerprint (permutation now detected), CHECKPOINT_DIR
  genuinely required, non-finite JSON refused.
- **Result** — suite 1470 passed / 2 skipped (+8); seven mutants killed.
- **Next** — re-review, then S1.5 under the standing grant.
