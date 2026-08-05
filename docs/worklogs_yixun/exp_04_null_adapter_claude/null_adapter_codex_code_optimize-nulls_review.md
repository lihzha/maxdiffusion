# null_adapter — Codex code review: round R3 `optimize-nulls`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Code under review: working-tree diff at HEAD `cf4f28f`. Reviewer ran the suite (89 passed incl. the real tiny-WanModel smoke) and performed its own empirical mutations (zeroed timesteps; zeroed latent inputs; ADAM_EPS 1e-4) to test oracle blindness.

## Reviewer output (verbatim)

Context loaded:

- [experiment_SOP.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/experiment_SOP.md) — reviewer briefing, TDD, review/strengthen cycle, validation ladder.
- [plan_null_adapter.md v5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/plan_null_adapter.md:24) — §3 optimization/batching contract, §5 items 1/12, §8 parity register.
- [null_adapter_worklog.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_worklog.md:110) through the R3 entry and five Planner positions.
- R1 and R2 review conventions: [sigma-embed-noise](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_code_sigma-embed-noise_review.md), [invert-trajectory](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_code_invert-trajectory_review.md).
- HEAD `cf4f28f`, complete implementation diff in [null_inversion_wan.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:237), and complete new [R3 test file](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_optimize_nulls.py).
- Pinned upstream `f37022874c588817d4ed77d463e3d27745053df4`: [embedding_search.py:488](/Users/yixunhu/Home/maxdiffusion/third_party/Wan2.2/scripts/embedding_search.py:488) and [optimize_null_embeddings:575](/Users/yixunhu/Home/maxdiffusion/third_party/Wan2.2/scripts/embedding_search.py:575).
- Supplemental briefing: query, both announcements, exp_02 analysis, shared timestep/pin helpers, and `WanModel`’s per-token timestep branch.
- Validation: **89 passed**, including the real tiny-`WanModel` smoke; Ruff `--no-cache` and `git diff --check` passed. The read-only sandbox required disabling pytest capture/cache.  

Findings:

1. **MAJOR** — [test_null_adapter_optimize_nulls.py:69](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_optimize_nulls.py:69), [test_null_adapter_optimize_nulls.py:307](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_optimize_nulls.py:307) — The strongest oracle’s synthetic velocity ignores both `z` and `timestep_2d`; the one-step smoke only checks nonzero gradients. Consequently, incorrect model evaluation points and completely wrong timestep contents are invisible. I replaced every timestep with an all-zero 2-D tensor and, separately, forced every velocity forward to receive `zeros_like(latents)`; **all 16 focused R3 tests passed in both cases**, including the smoke. Make the reference velocity affine in context but also nontrivially dependent on `z` and timestep, and add an explicit two-step check for frame-0 zeros and future-token values `sigma_i * 1000`.

2. **MINOR** — [test_null_adapter_optimize_nulls.py:143](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_optimize_nulls.py:143), [null_inversion_wan.py:323](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:323) — The `rtol=1e-3` trajectory allowance is justified—I reproduced the stated `7.62e-5` maximum relative `z_bar` gap—but it does not pin the claimed exact Adam recipe. Changing `ADAM_EPS` from `1e-8` to `1e-4` left all 16 tests green. Add an epsilon-sensitive optimizer fixture or literal recipe assertion, pass `eps_root=0.0` explicitly, and reserve the broad tolerance for `z_bar`. An independent actual Torch-vs-Optax fixed-gradient check otherwise showed close parity: max absolute difference `7.15e-7`.

3. **MINOR** — [test_null_adapter_optimize_nulls.py:34](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_optimize_nulls.py:34), [test_null_adapter_optimize_nulls.py:101](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_optimize_nulls.py:101) — The “hand-rolled reference” is not fully independent: it imports and uses production `embed_null_tokens` for both inner forwards and the advance. R1 tests mitigate this, but a shared splice regression cancels in R3’s principal comparison. Restate the small broadcast-and-splice operation locally in the literal reference and remove that production-helper import.

4. **MINOR** — [null_inversion_wan.py:289](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:289), [null_inversion_wan.py:323](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:323) — Argument validation silently truncates non-integral `inner_iters` (`1.5 → 1`) and accepts negative/nonfinite learning rates, yielding ascent or NaNs rather than Torch-style failure. Require an integer `inner_iters` and finite `lr >= 0`; retain `lr=0` for the frozen-null test.

The composition test genuinely pins both warm start and fresh Adam state. The disable-JIT call-count tests are sound for the current structural placement; tracer classification does not leak into production. The implementation itself correctly realizes cached `v_cond`, summed per-example means, batched moments, locked-null advance, pins, negative `dsigma`, and CFG mixing.

Rulings:

1. **M11 `stop_gradient`: KEEP.** It documents and enforces the upstream `torch.no_grad()` boundary for cached conditional velocity, particularly if this routine is ever enclosed by a higher-order differentiation transform.

2. **Context dimensions: KEEP generic relational validation.** Hard `512×4096` belongs at the production runner/artifact boundary; imposing it here would unnecessarily defeat the tiny-model seam.

3. **LOC overage: ACCEPT.** I found no genuinely dead test block; the literal loop, composition test, call-count tests, and real-model smoke cover distinct risks. `_f32_bits` duplication is too small to justify shared test infrastructure.

Final verdict: **REQUEST-REVISION** — the implementation is algorithmically faithful, but the principal R3 oracle demonstrably permits corrupted latent/timestep model inputs and does not fully pin the declared Adam recipe.

Status:

- No subprocesses running; R3 review is complete.
- Earliest steer: now, before strengthening and committing R3.

---

## Strengthening record (Coder, same round; all four findings addressed, none rejected)

1. **MAJOR (oracle blindness) — FIXED.** Oracle rebuilt as `_coupled_velocity` = affine-in-context + c_z·z + c_t·mean(timestep) (convexity and w=1 cancellation preserved); the hand-rolled reference computes its own timestep/z̄/splice/Adam. New explicit content test asserts the call sequence [cond, inner×J, advance] at two steps with frame-0 zeros and σ_i·1000 on every call. **Reviewer's probes re-run: B1 all-zero timesteps — was 16-passed, now 3 FAILED; B2 zeroed latents — was 16-passed, now 2 FAILED.**
2. **MINOR (Adam recipe) — FIXED.** `eps_root=0.0` explicit; epsilon-sensitive fixture in the ~1e-6-gradient regime matches hand-rolled eps=1e-8 (rtol 1e-5) AND asserts the eps=1e-4 reference differs (provably discriminating). B3 (eps→1e-4): was green, now 2 FAILED. B4 (b1→0.8) and B5 (eps_root→1e-8) killed.
3. **MINOR (reference independence) — FIXED.** Local `_splice` replaces the production import; mutant B6 (off-by-one splice in production `embed_null_tokens`) now caught by 6 tests including R3's principal reference comparison — the cancellation is gone.
4. **MINOR (argument validation) — FIXED.** Integral `inner_iters` (bool rejected), finite `lr ≥ 0` (0 legal), matched messages + rejection tests; guard-removal mutants B7/B8 killed.

Suite **96 passed in 13.05s** (+7); original R3 battery re-run against the coupled oracle — all 10 still killed, most with wider blast radius; M11 survives per the KEEP ruling. Behavior changes beyond the findings: none (guards + explicit eps_root are the findings) ⇒ no follow-up pass. Round R3 closed; committed with this record.
