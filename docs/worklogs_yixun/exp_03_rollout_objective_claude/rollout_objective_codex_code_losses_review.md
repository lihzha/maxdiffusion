# Codex code review — exp_03 cycle A round 3, the A/B/C objectives (01c6362)

- **Date:** 2026-08-02
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 2 MAJOR (B differentiates the training forward instead of the
  deterministic eval sampler; B's tests reproduce that convention instead of detecting it, and omit
  the production-bf16 certificate), 1 MINOR (support tests check frequencies, not the exact keyed
  draws), 1 LOW (dead `self_gen_noise`, stale "unimplemented" text, a refusal test that accepts an
  incidental `KeyError`).
- **Core verified:** A entirely correct — supports, direction, terminal exclusion, the same `(s,e)`
  across branches, the same epsilon, the stop-gradient boundary, the pin mask, and
  `v* = (z_lo − z_gt)/σ_lo` with the positive floor ≈0.17241379 (no clamp needed). B's scan/remat
  structure, same-ε endpoint, pin masking and `(σ_hi − σ_lo)²` normalization correct apart from the
  convention. C is the literal same-batch combination with one optimizer update. Round-2's AST
  narrowing sound; the ramp is resume-stable.
- **Decision rulings:** **D1 BLESSED** (`exp03_ramp_origin`) with the requirement that every Tier-1
  launch package pass `EXP03_RAMP_ORIGIN=10000` explicitly. **D2 BLESSED** (per-batch supports) with
  the caveat that S1.5 must quantify support-gradient variance, since the comparison estimates the
  objective-plus-support-estimator package rather than the loss formula alone. **D3:** A sound,
  **B amend**.

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

## Findings

1. **MAJOR — B differentiates the training forward, not the deterministic eval sampler.** [wan_ti2v_exp03_trainer.py:404](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py:404)

   `_rollout_loss` uses `_training_velocity_fn`, meaning `deterministic=False` with an RNG. B was approved as a deterministic short-horizon rollout loss through the operator evaluation runs. Differentiability does not require training mode. Use `_sampling_velocity_fn(ctx)` for B; gradients still flow through both forwards. C inherits this defect.

   The configured dropout is currently `0.0`, so the present Wan model is numerically insensitive to the flag. Nevertheless, the implementation is not literally the approved operator, validation does not enforce zero dropout, and a future nonzero dropout would train through a different trajectory.

2. **MAJOR — The tests lock B to the wrong convention and omit the required production-bf16 certificate.** [test_exp03_objectives.py:385](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_objectives.py:385)

   The explicit unroll reuses `_training_velocity_fn`, so it reproduces rather than detects the D3 error. All fixtures use float32 although production uses bf16, and the claimed “exact equality” is actually `np.allclose` on gradients only. Fix by:

   - Comparing B against an independently written deterministic two-step unroll.
   - Making the stub observably distinguish `deterministic=True` from `False`.
   - Testing both float32 and production bf16/eval-dtype rounding.
   - Checking endpoint/loss and gradients, with exact equality where claimed.

3. **MINOR — The support tests do not pin the claimed exact uniform distributions.** [test_exp03_objectives.py:192](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_objectives.py:192)

   The implementation itself is correct: scalar `jax.random.randint` uses the exact bounds for A and B. But A’s test checks only a loose 40–60% frequency for `k_A` plus coverage; B checks coverage only. A biased keyed mapping could pass. Compare outputs exactly against independently constructed `randint` draws from the named keys/bounds. Also extend the corrective identity across every valid positive sigma, including index 24, as required by P1.

4. **LOW — Remove dead/stale declarations.** [wan_ti2v_exp03_trainer.py:73](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py:73)

   `self_gen_noise` is deliberately unused and should be removed; retaining it misstates the same-ε design. The module/config/launcher also still say the trials are unimplemented. The “future objective refusal” test merely accepts an incidental `KeyError` and mutates an implementation set the dispatch never reads.

## Decision rulings

- **D1 — BLESS, with the plan amendment.** `exp03_ramp_origin` cleanly supports Tier 2 at `0` and Tier 1 at `10000`. The schedule is a pure function of threaded global step and origin, so it is resume-stable. Every Tier-1 launch package must explicitly pass `EXP03_RAMP_ORIGIN=10000`; the launcher default is intentionally Tier-2-oriented.

- **D2 — BLESS per-batch supports, with an interpretation caveat.** This is the literal approved wording, preserves the scalar extracted sampler, and keeps the stochastic gradient unbiased for the intended support-marginal objective. It does not invalidate the term “update-matched”: examples, GBS, Adam updates, and data remain matched.

  It does change gradient covariance substantially versus ctrl0’s per-example timesteps. Adam therefore need not follow the same expected trajectory, so the comparison estimates the complete objective-plus-support-estimator package, not the loss formula alone. A null result could partly reflect higher support-gradient variance. Quantifying that variance in S1.5 would strengthen the reading.

- **D3 — A sound; B amend.** A correctly uses deterministic eval forwards for its detached sampler advance and the training convention for its final supervised forward. B must use deterministic eval convention for both differentiated rollout forwards. Current B uses training convention.

## Core verification

- A’s supports, direction, terminal exclusion, same `(s,e)` branch discipline, same ε, stop-gradient boundary, pin mask, and `v*=(z_lo−z_gt)/σ_lo` are correct. The smallest denominator is the verified positive floor `≈0.17241379`; no clamp is needed.
- B’s two-step scan/remat structure, same-ε endpoint, pin masking, and `(σ_hi−σ_lo)²` normalization are correct apart from D3.
- C is the literal same-batch `λL_A+(1−λ)L_B` with independent auxiliary support purposes and one optimizer update.
- The round-2 AST narrowing is sound: it proves `exp03_aux_key` itself does not split the stream, while separate identity/parity tests preserve ctrl0 replication. The trial prologue splits the shared stream in exp_02’s order.
- Ramp implementation is resume-stable, although the test’s final “resume” assertion is tautological.

Static checks passed (`git diff --check`, Python AST parsing). I could not independently rerun the reported 1371/2 suite because this read-only environment lacks pytest/JAX and the available `uv` cache is sandbox-inaccessible.

## S1 readiness

Round 3 is not closed. Before the S1 smoke package goes to Yixun:

1. Switch B/C rollout forwards to deterministic sampler convention.
2. Add the convention-sensitive and bf16 explicit-unroll tests; strengthen the exact-support certificates.
3. Remove stale/dead RNG and “unimplemented” text.
4. Rerun the full suite and D3/bf16 mutations, then obtain the focused re-review.
5. Package S1 with explicit ramp origins and the declared A/B/C overhead STOP budgets.
```

## Strengthening record — coder response (round-3 strengthening)

All four findings closed, plus the tautological ramp assertion the reviewer flagged in passing.

### 1. MAJOR — B (and C's B-term) now differentiate the DETERMINISTIC eval sampler

``_rollout_loss`` uses ``_sampling_velocity_fn`` for both differentiated forwards. Gradients flow
through a deterministic forward exactly as well, and the approved operator is the sampler the
evaluation runs. **Currently numerically invisible** — the configured dropout is 0.0, so today's Wan
model is insensitive to the flag — but the implementation was not literally the approved operator,
nothing validates that dropout stays zero, and a future nonzero dropout would differentiate a
different trajectory. A's split convention is unchanged (ruled sound): its detached advance uses the
eval convention, its final supervised forward the training one.

### 2. MAJOR — B's tests now DETECT the convention

* ``_independent_rollout_loss`` is written from primitives — it reconstructs exp_02's 3-way split and
  epsilon, draws B's support from the named auxiliary key, builds the interpolant inline, and uses
  its OWN velocity closure whose ``deterministic`` flag is an argument. It calls none of
  ``_training_velocity_fn`` / ``_sampling_velocity_fn`` / ``_interpolant_at``, so it cannot
  reproduce the trainer's choice. (The Euler step itself is the extracted
  ``overfit100_sampler_step`` — the one-sampler rule, verified in round 1.)
* The stub transformer now **observes** the flag (a constant is added in training mode), so the two
  conventions are numerically distinguishable at all.
* Parameterized over **float32 and production bfloat16**, both dtypes end to end (weights and
  activations).
* Both the loss and the gradient are checked. The **loss is exact** (``array_equal``) in both
  dtypes. The **gradient is exact in fp32**; in bf16 it is not, and the record says why rather than
  quietly loosening: the reverse pass of ``lax.scan`` accumulates the parameter cotangent in a
  different ORDER than an unrolled loop, worth ~3e-4 relative in bf16. What is asserted there is the
  operator claim — the eval-convention reference must be **more than 50x closer** than the
  training-convention one, which it is by orders of magnitude.
* "Exactly two differentiated forwards" is now pinned by one-step and three-step references that
  must both differ, and C's B-term is compared against the same independent unroll.

### 3. MINOR — exact support certificates

``corrective_support`` and ``rollout_support`` are compared **bit-for-bit** against independently
constructed ``jax.random.randint`` draws from the named keys with the plan's bounds, at five global
steps each, plus assertions that the three purposes are genuinely distinct keys. The frequency and
coverage tests are retained as a second, weaker net. The corrective identity now runs over **every
valid positive sigma, indices 0..24 inclusive** (both the on-path reduction and the off-path
contraction), so the smallest positive grid sigma is covered as P1 requires.

### 4. LOW — dead and stale text removed

``self_gen_noise`` is gone from ``EXP03_AUX_PURPOSES`` (a test now asserts every declared purpose is
actually drawn, so this cannot recur), with a comment stating the design it misdescribed. The
"unimplemented until round 3" text is scrubbed from the module docstring, the config and the
launcher; ``EXP03_IMPLEMENTED_OBJECTIVES`` is now **derived** from the dispatch table rather than
hand-maintained. The dispatch raises an explicit ``NotImplementedError`` naming the arm (not a bare
``KeyError``), and the refusal test exercises the real path: it extends ``EXP03_OBJECTIVES`` (which
``validate_exp03_config`` genuinely reads), asserts the config surface accepts the new name, and
then asserts the dispatch refuses it.

**Launcher (D1 requirement):** the launcher header now states that **Tier-1 arms must pass
``EXP03_RAMP_ORIGIN=10000`` explicitly**, and why — the default is Tier-2-oriented, and a Tier-1 arm
resuming from step 10,000 with origin 0 would start at full ``p_ss`` instead of ramping.

**Ramp test:** the tautological final assertion is replaced by a counterfactual — at global step
10,250 the ramp must read 0.25 (half of a 500-step ramp from origin 10,000) and must NOT read the
0.15 a segment-local counter would give after a resume at 10,100.

### Verification

* Full worklogs suite: **1399 passed, 2 skipped** (round-3 commit: 1371 + 2; +28).
* Mutations — 5 new, all killed:
  1. B back on the training convention (the D3 defect) -> **3F**;
  2. *guard the guard* — the independent unroll rewired to the trainer's velocity fn -> **2F**;
  3. the stub stops observing ``deterministic`` -> **2F** (the detection dies, and the tests say so);
  4. A's start index drawn from B's key -> **3F**;
  5. ``k_A`` drawn from the start-index key -> **3F**.
* Regression-checked that round 3's own mutants still bite after the test rewrite: horizon
  normalization dropped -> 4F; corrective denominator ``sigma_hi`` -> 2F; C's weights swapped -> 1F.
* ``black --line-length 119``, ``ruff check`` clean; ``bash -n`` clean; staged-tree
  ``git diff --check`` clean.

No push, no launch.
