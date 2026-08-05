# pos_context — Codex code review: round S2 `optimize-positives`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_05 worktree). Working tree at HEAD `45bf8cb`. Reviewer verified the algorithmic core empirically (zero-latent/zero-timestep oracle probes; w=1 physics measured: positive min-grad 0.91495 vs null max exactly 0; batch permutation exact) and ruled: cast-seam move architecturally sound but the handoff docs contradictory; pos_init ACCEPT.

## Reviewer output (verbatim)

Context loaded:

- [experiment_SOP.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/experiment_SOP.md), both standing announcements, and the experiment query.
- [plan_pos_context.md v3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md:22), including §3, the inherited exp_04 batching contract, and the `[25,...]` state schema.
- [pos_context_worklog.md through S2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_worklog.md:86).
- The complete S1 review and exp_04 R3 `optimize-nulls` review/strengthening record.
- The uncommitted implementation diff, complete [test_pos_context_optimize.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_optimize.py), and exp_04’s `optimize_null_embeddings`.
- Upstream `f370228`: `_dit_velocity:503-513` and `optimize_positive_embeddings:681-788`.
- Shared timestep, pin, velocity-check, sigma-validation helpers, and the deployed cast at `side_adapter_wan.py:767`.

1. **MAJOR** — [pos_context_inversion_wan.py:199](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/models/wan/pos_context_inversion_wan.py:199), [pos_context_inversion_wan.py:280](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/models/wan/pos_context_inversion_wan.py:280), [test_pos_context_optimize.py:364](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_optimize.py:364) — The “first-class training states” output contains `N+1` entries, including terminal `z̄_N`, while plan §3 and the cache schema require the `N` pre-step states `z̄_0…z̄_{N-1}` aligned one-to-one with `C_0…C_{N-1}`. A production 25-step probe produced `pos=(25,...)` but `z_bar=(26,...)`; the test explicitly entrenches the terminal state as a recorded training state. Return an explicit `[N,...]` `z_bar_states` output, exposing `z_final` separately if needed, or explicitly distinguish a full trajectory from `z_bar_states=trajectory[:-1]` and test the schema-facing conversion. Do not leave an implicit slice for S4/K2.

2. **MAJOR** — [pos_context_inversion_wan.py:19](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/models/wan/pos_context_inversion_wan.py:19), [test_pos_context_truncate_pad.py:19](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_truncate_pad.py:19), [test_pos_context_truncate_pad.py:365](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_truncate_pad.py:365) — The formal cast-owner move is contradicted by the committed S1 handoff: it repeatedly says the S3 replay operator must perform the cast, while the new S2 contract says `velocity_fn` owns it. [The S2 test](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_optimize.py:388) only proves that this optimizer does not cast; it cannot ensure that absent S3/S4 wiring does. Rewrite every stale handoff to one rule: optimizer and replay pass fp32 context unchanged, and the runner-built real-backbone `velocity_fn` casts immediately before the transformer. Require S3 parity through the actual replay operator and an S4 closure test covering both branches at bf16.

The algorithmic core otherwise checks out: cached `v_uncond`, `N·(J+1)` conditional forwards, direct variable-length positive context, locked advance, fresh Adam state, pins, fp32 arithmetic, summed per-example objective, and batch independence all match the contract. The hand-written oracle is independent and consumes context, latent, and timestep; zero-latent and zero-timestep probes changed outputs materially, and batch permutation was exact. At `w=1`, the measured positive minimum gradient norm was `0.91495`, while exp_04’s null maximum was exactly `0`, confirming that the test pins the distinguishing physics and that the contrast claim is accurate. The call-count test is sound under `jax.disable_jit()`.

Cast-seam ruling: moving the cast into the real-backbone `velocity_fn` is architecturally sound and matches upstream autocast placement, but the present contradictory handoff is not strong enough to prevent S3/S4 from missing or misplacing it.

`pos_init` ruling: **ACCEPT**. It mirrors the reference’s explicit initialization input, enables the approved `L_pos` ablation and composition test, and remains narrowly validated; it is a sound API extension, not scope creep. Relative dimension validation and the test LOC exception are also reasonable.

Final verdict: **REQUEST-REVISION** — correct branch-swap optimization is undermined by the state-cardinality mismatch and an internally contradictory cast-seam handoff.

Status:

- No subprocesses are running; the S2 review is complete.
- Earliest steering point: now, before strengthening and committing S2; interruption is safe.

---

## Strengthening record (Coder, same round; both findings closed, none rejected)

1. **MAJOR (state cardinality) — FIXED.** 5-tuple return: `z_bar_states [N,...]` (the K2 schema, 1:1 with contexts) + `z_final` split by name at the return site; the 1:1 consumer-zip test verifies pairing semantics (each state advances to the next under its paired context via the CFG Euler step); schema mutant M12 killed by 6 tests.
2. **MAJOR (cast-seam handoff) — FIXED.** One rule in the module docstring (operators pass fp32 unchanged; the runner-built velocity_fn casts both branches immediately before the transformer), referenced elsewhere; S1's phrasing explicitly retired in the S1 test file's docs; MUST carry-forwards pinned for S3 (parity through the actual replay operator) and S4 (bf16 both-branch closure test); repo-wide grep confirms only superseding quotes remain. Worklog correction was the Planner's (96c5379).

Suite **689 passed** (688 + 1); **12 mutants, 0 survivors** re-run in full. Behavior beyond findings: none. Round S2 closed; committed with this record.
