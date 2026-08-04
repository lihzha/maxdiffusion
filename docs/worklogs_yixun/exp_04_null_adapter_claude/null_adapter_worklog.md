# exp_04 `null_adapter` — worklog (append-only lab notebook)

Experiment: port null-text inversion (Mokady et al. 2022) to maxdiffusion JAX against the frozen Wan2.2 TI2V 5B backbone, compute per-step null embeddings for side-adapter-dataset examples, train an action-conditioned adapter on them, and visualize reconstruction/rollout quality.

Worktree: `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter`
Branch: `claude-exp_04_null_adapter-20260803` (off `yixun-dev` @ `744094a`)
Primary agent: claude (Planner: Claude Fable 5 max; Coder: Opus 5 max subagent; Reviewer: Codex `gpt-5.6-sol` xhigh)

## 2026-08-04T02:48:06Z — Scaffold exp_04

- **Goal** — Reserve experiment number 04, create branch + worktree + docs folder + query doc per SOP, before any planning.
- **Change** — New folder `docs/worklogs_yixun/exp_04_null_adapter_claude/` with `null_adapter_yixun_query.md` (Query 1 verbatim + summary/hypothesis/why) and this worklog. No source code touched.
- **Version Control** — branch `claude-exp_04_null_adapter-20260803`, base_commit `744094a` (= `yixun-dev` tip, pushed to origin before branching). Worktree added at `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter`.
- **Command / Validation** — `git worktree add /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter -b claude-exp_04_null_adapter-20260803 yixun-dev`. Docs will auto-sync to `yixun-dev` via `.githooks/post-commit`.
- **Result** — `launched` (scaffold committed with this entry's commit).
- **Analysis** — Tracker confirmed next number is exp_04; no competing worktree holds it. exp_03 (S1.5 probe) remains active in its own worktree — untouched by this experiment.
- **Next** — Two Explore agents (launched from the main session) map (a) the maxdiffusion side-adapter/rollout/CFG/data path and (b) the existing PyTorch inversion→adaptor pipeline in `third_party/Wan2.2/scripts/`. Then Planner writes `plan_null_adapter.md`; Codex plan review; resolutions; surface to Yixun for approval. No TPU launch without explicit approval (announcement 02).

## 2026-08-04T03:35:00Z — Context exploration complete; plan v1 written

- **Goal** — Load all context needed to plan: reference scripts (read in full), the Mokady paper, the maxdiffusion side-adapter/rollout path, and the PyTorch fork's prior inversion/adaptor line.
- **Result** — `passed`. Two Explore-agent reports landed (main session). Load-bearing facts:
  1. maxdiffusion side-adapter path uses **T5("") as BOTH CFG branches** (`wan_ti2v_side_adapter_trainer.py:313-325`); `side_adapter_guide_scale: 5.0`; CFG amplifies the adapter delta. No text conditioning exists.
  2. All sampler machinery reusable on `yixun-dev`: `build_rollout_sigmas` / `apply_first_frame_pin` / `_build_per_token_timestep` (`side_adapter_wan.py:814/528/518`), Euler update + JAX VAE decode + SSIM/video helpers in `generate_wan_side_adapter.py`.
  3. **No inversion/null-text code exists anywhere in this repo (any branch)** — from-zero build.
  4. PyTorch fork: DROID inversion line used **positive_inversion only**; `null_inversion` never run on DROID; no network ever trained to predict null embeddings. Central negative finding = **noise-basin problem** (own-z_init latent MSE 0.015–0.022 vs fresh-noise 2.0–3.3; `Wan2.2/docs/adaptor_design.md` §3.11–3.14, `WORKLOG.md:943-949`); also CFG-consistency failures (train w=1 vs eval w=5) and `sample≈null` underfitting of the positive adaptor. Sigma grids: PyTorch starts σ_max=0.999 vs maxdiffusion 1.0 — small documented deviation, acceptable (no cross-repo artifacts).
- **Change** — `plan_null_adapter.md` v1 written (Planner): 4-phase design — P0 TDD infra (tiny-model smoke port), P1 capacity study + basin probe (arms A0/A1/A1-probe/A2 fixed-noise-nulls/A3 direct-opt), P2 target caching, P3 action-conditioned null-embedding adapter (DiT-free regression) + same-cohort eval vs pre_context baseline, P4 reports. Gates G1/G3 predeclared; 5 decision points for Yixun.
- **Version Control** — branch `claude-exp_04_null_adapter-20260803`; plan committed with this entry.
- **Next** — Codex plan review (briefed: SOP, query doc, plan, this worklog, prior exp analyses, reference scripts), then Planner resolutions, re-review if materially revised, then surface to Yixun.

## 2026-08-04T04:40:00Z — Codex plan review v1: REQUEST-REVISION (17 findings) → plan v2 + resolutions

- **Goal** — Independent review of plan v1 before user approval (SOP artifact 3).
- **Command / Validation** — `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh "$(cat prompt)" < /dev/null` from the exp_04 worktree (session `019fcab8-6960-7cc3-a0d4-60d38d6d4305`, ~206k tokens, ~35 min). Reviewer confirmed full briefing loaded (SOP, announcements, query, plan, worklog, tracker, exp_02 analysis, exp_03 plan, Mokady paper, all three reference scripts + `fm_solvers_unipc.py` from the submodule object store, side-adapter trainer/model/generator/config, transformer, TFRecord producer).
- **Result** — `passed` (review obtained): **REQUEST-REVISION**, 15 MAJOR + 2 MINOR. Headline findings: capacity-bound overstatement (F1), Mokady-faithfulness naming (F2), VAE-ceiling conflation (F3), **cohort selection leaking into final eval** (F4), missing A2 matched control (F5), statistically under-specified gates (F6), non-self-contained cache (F7), missing cache integrity/fp16 gates (F8), under-specified adapter + no learnability gate (F9), unmatched baseline noise + anchor risk (F10), unproven A3 cost (F11), vague batching contract (F12), P0 tests asserting unguaranteed properties (F13), missing replay-verifier contract (F14), evaluator/checkpoint wiring gaps (F15), pinned-frame Gaussianity stats (F16), oversized rounds (F17). Saved verbatim to `null_adapter_codex_plan_review.md`.
- **Analysis** — All 17 accepted (no rejections); the review caught two errors that would have invalidated conclusions (F4 leakage, F5 missing control) and several that would have wasted TPU spend (F8, F11). Not an infra event.
- **Change** — `plan_null_adapter.md` rewritten as v2 (changelog in header); resolutions appended to the review file.
- **Version Control** — committed with this entry on `claude-exp_04_null_adapter-20260803`.
- **Next** — Dispatch full re-review of plan v2 (material revision), then surface to Yixun with review + resolutions + re-review verdict.

## 2026-08-04T05:35:00Z — Re-review pass 2: REQUEST-REVISION (7 partials + N1–N9) → plan v3

- **Goal** — Verify v2 resolutions and screen new content (SOP re-review after material revision).
- **Command / Validation** — same reviewer invocation; session `019fcacb-6788-7cb2-9bcf-e6e3d0c525cd` (~35 min). Full briefing confirmed; verified v2 at commit `58c14dd`.
- **Result** — `passed` (review obtained): **REQUEST-REVISION**. F-verification: 10 RESOLVED, 7 PARTIALLY-RESOLVED (F1, F5, F6, F9, F10, F11, F14, F15, F17); 8 new MAJOR + 1 MINOR (N1 target-selection floor, N2 TRAIN-manifest cost/immutability, N3 ε₀-vs-keyed noise conflict, N4 executable gate module, N5 L_null ablation outcome rule, N6 P3a/arch pinning, N7 legacy restore + evaluator parity, N8 schema/fidelity-gate gaps, N9 oracle labeling). Saved verbatim to the review file.
- **Analysis** — All accepted. The noise-convention conflict (N3) was a real internal inconsistency in v2 (two different experiments described as one); the rest are executable-precision demands that prevent post-hoc flexibility. Not infra.
- **Change** — `plan_null_adapter.md` rewritten as v3 (changelog in header): named noise conventions, J0 manifest job, gates module `null_adapter_gates.py`, target-selection floors, pinned adapter/P3a budgets, legacy restore contract + RNG-replicated parity, schema + fidelity-gate fixes, rounds R1–R15. Resolutions appended to the review file.
- **Version Control** — committed with this entry; pass-3 re-review dispatched.
- **Next** — Pass-3 verdict → surface to Yixun (target: APPROVE-PLAN; if further findings are minor, resolve and surface with the full review trail).
