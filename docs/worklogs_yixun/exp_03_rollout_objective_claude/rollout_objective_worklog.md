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
