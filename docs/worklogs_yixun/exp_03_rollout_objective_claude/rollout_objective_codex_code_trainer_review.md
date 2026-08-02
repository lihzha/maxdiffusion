# Codex code review — exp_03 cycle A round 2, trainer + binding hook (c0aaaa2)

- **Date:** 2026-08-02
- **Reviewer:** Codex `gpt-5.6-sol`, xhigh, read-only sandbox
- **Verdict:** REQUEST-REVISION — 1 BLOCKER (the step-keyed RNG API is not usable from inside the
  compiled step, and `state.step` is not resume-safe), 2 MAJOR (parity certified only eagerly;
  the launcher drift test binds neither shared defaults nor the optional override).
  **Round 3 NO-GO until fixed and re-reviewed.**
- **Verified:** identity-return clean (no per-step validation leakage into the compiled step); the
  late-binding choice endorsed ("reasonable, no retrace hazard"); the auxiliary root is
  collision-free against exp_02's streams and the dataloader's separate TF rng; the purpose-id
  scheme is sound and future collisions are tested; the resume test faithfully models exp_02's
  fresh `key(seed+1)` restart; preflights preserved; YAML parity binds in both directions.

## Reviewer output (verbatim, final verdict block)

```
REQUEST-REVISION

1. **BLOCKER — RNG cannot yet consume the production global step under JIT** — [wan_ti2v_exp03_trainer.py:77](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_exp03_trainer.py:77): `int(global_step)` will raise on a JAX tracer, while the compiled call receives only `(state, data, rng)` at [wan_ti2v_overfit100_trainer.py:1216](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/trainers/wan_ti2v_overfit100_trainer.py:1216). Using `state.step` is not resume-safe because restore replaces only params/opt-state, leaving its freshly initialized step behind. Fix: pass an explicit dynamic global-step scalar into the objective/compiled step, make auxiliary folding tracer-safe, and test uninterrupted versus real save/restore keys at the same global step.

2. **MAJOR — exact parity is tested only eagerly, not on the production JIT boundary** — [test_exp03_trainer.py:158](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_trainer.py:158): the wrapper is trace-time-only and appears semantically correct, but no test compiles the pre-refactor copy and dispatch with `jax.jit`, checks repeated cached calls, or compares full AdamW state. Fix: add multi-call JIT parity over params, optimizer state, state step, all metrics and RNG, plus a trace-count/Jaxpr assertion. Keep the late-binding test; preserving exp_02’s observable patch contract is reasonable and introduces no per-step closure/retrace hazard.

3. **MAJOR — launcher drift test does not bind shared defaults or every override** — [test_exp03_trainer.py:568](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective/src/maxdiffusion/tests/worklogs_yixun/test_exp03_trainer.py:568): its suffix filter excludes default assignments and the optional `tfrecord_shuffle_buffer_size` override, so changing `LEARNING_RATE`’s default or dropping that override still passes. Fix: compare parsed default-assignment and CLI-override maps bidirectionally after an explicit allowlist of EXP03/run/output deltas; mutation-test both examples.

- **Claim 1:** Arithmetic and trace-time binding look equivalent, and retaining late binding is the right compatibility choice, but production-JIT equivalence is not yet pinned.
- **Claim 2:** Pass: inherited `__init__` only stores config; validation occurs on the host before loading and again during startup, never inside the compiled step.
- **Claim 3:** Partial fail: the offset root is distinct from exp_02’s same-seed roots; the dataloader is a separate TensorFlow RNG; the three 32-bit purpose IDs are collision-free and future collisions are tested—but the global-step API is not JIT/resume-usable yet. The shared-stream resume test accurately models exp_02’s actual fresh `key(seed+1)` restart.
- **Claim 4:** Pass in current code: inherited save/restore and `super().start_training()` preserve preflights, including pinned-snapshot checks from an empty directory. The roundtrip is faithful to Orbax item structure, though local/unsharded rather than a GCS multi-host integration.
- **Claim 5:** YAML parity binds strongly in both directions; EXP03 passthrough presence binds, but shared launcher default/optional-override drift does not.
- **Claim 6:** No concrete ctrl0 arithmetic divergence is visible, but the missing JIT certificate and launcher-default binding leave avoidable routes to a spurious AND-gate failure.

Static checks (`git diff --check`, launcher `bash -n`) pass; pytest was unavailable in this read-only environment.

**Round 3: NO-GO until findings 1–3 are fixed and re-reviewed.**
```

## Strengthening record — coder response (round-2 strengthening)

All three findings closed. The control arm's arithmetic is unchanged; what changed is that the
global step now crosses the jit boundary, and that the parity and drift claims are certified where
production actually runs.

### 1. BLOCKER — the global step is threaded as a dynamic scalar, and the RNG is tracer-safe

* ``_denoising_loss`` and the factory's step take a keyword-only ``global_step``; the plain
  objective ``del``s it (it is stepwise-stationary), so the control path computes exactly what it
  computed before.
* ``start_training`` compiles a 4-argument adapter — ``(state, data, rng, global_step)`` — with
  ``in_shardings=(state_shardings, data_shardings, None, None)`` and calls it with
  ``jnp.asarray(step, jnp.int32)``: **the loop's step**, never ``state.step``. A test asserts
  structurally (AST, so the explanatory comment is not mistaken for the thing it warns about) that
  ``state.step`` is never read in the loop.
* ``exp03_aux_key`` folds ``jnp.asarray(global_step, jnp.uint32)`` — no ``int()`` on the step. The
  non-negative check now applies only to concrete Python integers, since under tracing there is no
  value to check and inventing one would be worse than none. An AST test pins that no
  ``int(global_step)`` survives outside that guard.
* **Compatibility kept:** when no step is threaded the loss is called with exp_02's exact
  six-argument shape, so the exp_02 test that spies on ``_denoising_loss`` with that signature keeps
  passing untouched. Both shapes are pinned by a test.
* Tests added: (a) aux keys under ``jit`` equal eager at the same step, for four step values, keys
  and draws; (b) **end-to-end resume equality through the production restore path** — an
  uninterrupted 6-step run versus a run preempted at 3, saved with ``_save_checkpoint``, restored
  with ``_maybe_restore`` into a *freshly built* state, and resumed at the restored step, produce
  identical step-keyed draws at every global step (with a non-vacuity check that the six draws are
  all distinct), plus the counterfactual ``int(restored.step) == 0 != start_step`` that motivates
  threading the loop's step; (c) the control's compiled step is unchanged by the extra argument and
  gives identical results at global steps 0 and 10,000.

### 2. MAJOR — the JIT certificate

Both the verbatim pre-refactor copy and the dispatched control step are compiled behind the same
4-argument production boundary with **AdamW** and run for four cached calls in sequence. Compared at
every call: params, the **full optimizer state** (mu, nu, count — all leaves), ``state.step``, every
metric, and the returned rng — all exact. Plus a **trace-count assertion** (each compiles exactly
once across the four calls, so neither the closure nor the dynamic step specializes the cache) and a
**jaxpr comparison** (identical primitive multiset and equation count). A separate test shows a
round-3-style step-keyed objective compiles once and still gets different draws per step.

### 3. MAJOR — the launcher drift test now binds defaults and overrides, bidirectionally

Two parsers: ``NAME="${NAME:-value}"`` default assignments (accepting the ``${VAR-}`` form too), and
every ``key=value`` passed to ``train_wan.py``, with conditional overrides such as
``${TFRECORD_SHUFFLE_BUFFER_SIZE:+...}`` captured verbatim as their own entry. Both maps are
compared **in both directions** — same keys *and* same values — outside a deliberately tight
allowlist: ``RUN_NAME``, ``OUTPUT_DIR`` and the six ``EXP03_*`` defaults; the six ``exp03_*``
overrides. Parser non-vacuity is asserted (``LEARNING_RATE``/``MAX_TRAIN_STEPS`` present; the
conditional found).

### Verification

* Full worklogs suite: **1340 passed, 2 skipped** (round-2 commit: 1331 + 2; +9).
* Mutations — 6 killed, plus one meta-check:
  1. ``int()`` coercion reintroduced in the fold -> **3F** (the jit tests);
  2. step threaded but the aux key folds a constant 0 -> **4F** (incl. the resume-equality test);
  3. the loop keys on ``state.step`` instead of the loop step -> 1F;
  4. the compiled adapter drops the threaded argument -> 1F;
  5. **named example** — a shared launcher DEFAULT drifts (``LEARNING_RATE`` 1e-5 -> 2e-5) -> 1F;
  6. **named example** — the optional ``tfrecord_shuffle_buffer_size`` override is dropped -> 1F.
  7. *meta* — with the allowlist broadened by one entry (``LEARNING_RATE``), mutant 5 **survives**,
     which is the evidence that the allowlist's tightness is what carries the claim. Reverted.
* ``black --line-length 119``, ``ruff check`` clean; staged-tree ``git diff --check`` clean.

No push, no launch.
