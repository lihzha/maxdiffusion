# null_adapter — Codex code review: round R6 `runner-capacity-core`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `c35f9dc`. Reviewer ran both suites and empirical mutants (false-header records verifying; same-names/different-tensors records verifying; NaN-ed diagnostics leaving reports unchanged). Rulings: A1 noise_convention = deployment interpretation APPROVED; A2 traj.at[0].set(ε₀) reading APPROVED; table-matching convention sound for J1; fp32 writer-order test discriminating, fp16 note correctly scoped to R8; no-tiny-seam choice accepted.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md` in full.
- `plan_null_adapter.md` v5, including §3, §4-P1/P2, plus v2’s inherited detailed metric contract.
- `null_adapter_worklog.md` through the R6 handoff/write entry and all 11 Planner positions.
- R1–R5 reviews and strengthening records: `sigma-embed-noise`, `invert-trajectory`, `optimize-nulls`, `replay-operator`, `record-schema-io`, `verify-replay`, and `gates-module`.
- Both R6 files in full: `null_adapter_runner_core.py` and `test_null_adapter_runner_core.py`.
- Composed surfaces: `null_inversion_wan.py`, `null_adapter_records.py`, `null_adapter_verify.py`, and `null_adapter_gates.py`.
- Pinned reference `embedding_search.py:880-1148` from main checkout submodule SHA `f37022874c588817d4ed77d463e3d27745053df4`.
- Repository state: HEAD `c35f9dc`; only the two R6 files are untracked.
- Validation: R6 focused suite **61 passed in 10.07s**. Full suite with capture disabled: **306 passed, 1 known tiny-Wan tmpdir failure** in 22.67s.

## Findings

1. **MAJOR — Records are not provenance-bound to the run that produced their nulls and metrics.** [null_adapter_runner_core.py:125](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_runner_core.py:125), [null_adapter_runner_core.py:334](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_runner_core.py:334)

   `ArmResults` retains neither `CapacityParams` nor an input/context fingerprint. `build_capacity_records` checks only example names, then recomputes the expected endpoint under the supplied header. Empirical mutants showed that results produced with `J=1, lr=.01, w=5` can be written under a header claiming `J=50, lr=.03, w=7`, and all records pass `verify_replay`. A same-name batch with entirely different `z_i0/z_video` also emitted verifying records while retaining the original stale `final_future_mse`.

   Concrete change: bind exact parameters, base-context fingerprint, and canonical batch-content fingerprint into `ArmResults`; reject any mismatch before replay. Require the header’s guide scale, `l_null`, and optimization recipe to match those bound values. Add false-header, changed-context, and same-names/different-tensors mutants using a forbidden velocity callback.

2. **MAJOR — Required tracking curves and adequacy grad-norm traces are discarded.** [null_adapter_runner_core.py:262](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_runner_core.py:262), [null_adapter_runner_core.py:423](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_runner_core.py:423)

   Both optimizer calls throw away the returned `[N,J,B]` losses and grad norms. `RecipeScore` retains only the derived score and per-example aggregate. This violates §4-P1’s requirement to log per-step final losses and per-inner-iteration grad-norm traces, as well as the inherited P1 tracking-curve metric. Replacing every returned loss and grad norm with NaN left the adequacy report finite and unchanged.

   Concrete change: retain host copies of `[N,J,B]` tracking losses and grad norms for A1/A2 and every adequacy recipe, alongside the distinct post-loop `[B,N]` final losses. Add shape/layout and corrupted-diagnostic mutants.

3. **MINOR — The exact 10% plateau boundary is classified incorrectly.** [null_adapter_runner_core.py:430](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_runner_core.py:430)

   For scores `J25=1.0`, `J50=0.9`, floating-point evaluation produces `0.09999999999999998`, yielding `reconstruction-limited`. The plan says only improvement **below** 10% gets that label; exactly 10% belongs to `recipe-limited`.

   Concrete change: use a declared boundary-aware comparison and test exact, immediately-below, and immediately-above 10% cases.

4. **MINOR — Invalid recipes reach expensive model computation before rejection.** [null_adapter_runner_core.py:173](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_runner_core.py:173), [null_adapter_runner_core.py:252](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_runner_core.py:252)

   `_validate` omits `inner_iters` and `lr`; `_validate_grid` omits finite/nonnegative LR checks. An `inner_iters=0` probe reached the inversion velocity callback before failing inside R3.

   Concrete change: validate integral `J≥1` and finite `lr≥0` before inversion in both paths, with matched-message tests using the forbidden callback.

## Rulings

- **Item 3 — A1 `noise_convention`: APPROVE the deployment interpretation.** An A1 record should say `keyed`: `z_start` explicitly records its optimization/verification origin, while `noise_convention` identifies the selected P2/P3 deployment convention and k-set.

- **A2 same-pivots reading: APPROVE.** `traj.at[0].set(ε₀)` is exactly right for this API: index 0 initializes `z̄₀`, while every optimization target remains the original `traj[1:]`.

- The table-matching convention is numerically sound for J1: call G1/G2 with `GLOBAL` solely to select singleton key `{0}`, and reduce A1-probe with `KEYED`; that singleton reducer label must not replace A1’s deployment metadata.

- The fp32 writer-order test is genuinely discriminating: correct records verify at `atol=1e-5`, while replay-before-cast records fail. The fp16 non-discrimination note is correctly scoped to R8’s fidelity/tolerance policy.

- The no-tiny-seam choice is acceptable: production geometry caught drift at a bounded cost—61 focused tests in about 10 seconds—with shared cached fixtures limiting runtime creep.

Final verdict: **REQUEST-REVISION** — the arm wiring is correct, but false provenance and missing adequacy diagnostics can invalidate J1/P2 evidence.

---

## Strengthening record (Coder, same round; all four findings closed, none rejected)

1. **MAJOR (provenance) — FIXED.** ArmResults binds CapacityParams + base-context fingerprint + canonical batch-content fingerprint (name-sorted, NUL-terminated, codec byte discipline, order-free — pinned by mutants N6/N7); build_capacity_records refuses misdeclaring headers, foreign contexts, and changed-tensor batches BEFORE any forward (forbidden-callback tests); both reviewer mutants killed. Header optimization_config must now declare exactly {inner_iters, lr} — an R8/R10 writer contract.
2. **MAJOR (diagnostics) — FIXED.** [N,J,B] tracking losses + grad norms retained for A1/A2 and every adequacy recipe (plus [B,N] final losses); `_checked_trace` makes non-finite traces a hard failure at construction; retained curves pinned bitwise against an independent optimizer call.
3. **MINOR (plateau boundary) — FIXED.** Declared PLATEAU_BOUNDARY_ATOL=1e-9 comparison; exactly-10% ⇒ recipe-limited; three-point boundary tests.
4. **MINOR (early validation) — FIXED.** `_validate_recipe` (integral J≥1, finite lr≥0) before inversion in both paths, incl. bad-second-grid-cell; forbidden-callback proof.

Suite **336 passed in 29.3s** (246 inherited unchanged, R6 file 90 tests); **38 mutants, 0 survivors** (one mid-battery survivor — N2 masked by the nulls-width conjunct — closed with a dedicated disagreement test rather than weakening either check). Module 365 exec LOC (+68, all mandated) — overage recorded; a measured 92-LOC adequacy seam stands ready if R7/R8 pressure warrants a split (Planner call deferred). Behavior beyond findings: none. Round R6 closed; committed with this record.
