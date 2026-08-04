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

## 2026-08-04T06:25:00Z — Re-review pass 3: REQUEST-REVISION (M1–M8) → plan v4

- **Goal** — Verify v3 closures; screen new v3 content.
- **Command / Validation** — same reviewer invocation; session `019fcad9-8634-7640-8bb0-d0751856e055`. Verified v3 at `7bead68`.
- **Result** — `passed` (review obtained): **REQUEST-REVISION**, now narrow: 7 MAJOR + 1 NIT, all executable-precision pins — M1 exact fold_in key derivation; M2 A2 deployment estimand (global k={0} only); M3 decidable adoption rule; M4 numeric J0 caps + explicit DEV/TEST slicing; M5 artifact `latent_dtype` fallback semantics; M6 legacy checkpoint URI + effective-config override set + eval_shape guard; M7 worst-value imputation for invalid pairs; M8 count correction (pass-2 partials were **nine** — F1, F5, F6, F9, F10, F11, F14, F15, F17 — not seven as the v3 header and the previous worklog entry stated; corrected here, previous entry stands as written per append-only rule).
- **Analysis** — All accepted; every item is a determinism/estimand pin that prevents post-hoc flexibility or execution ambiguity. Not infra.
- **Change** — `plan_null_adapter.md` → v4 (targeted edits; changelog in header). Resolutions appended to the review file.
- **Version Control** — committed with this entry; pass-4 re-review dispatched.
- **Next** — Pass-4 verdict → surface to Yixun with the full four-pass review trail.

## 2026-08-04T07:05:00Z — Re-review pass 4: near-converged (M1–M6, M8 RESOLVED; P1/P2) → plan v5

- **Goal** — Verify M-item closures; screen the v4 delta.
- **Result** — `passed` (review obtained): M1–M6 + M8 + F17-residue RESOLVED; M7 PARTIALLY-RESOLVED sharpened into **P1 (MAJOR)**: `SSIM←0.0` imputation is ambiguous/anti-conservative for *paired* differences (a missing baseline could inflate the adapter's advantage) — fix = aggregate-level, claim-penalizing imputation per gate with ΔSSIM←−1.0 for invalid G3 pairs; **P2 (MINOR)**: TRAIN-2000 assumed ≥2 windows/episode.
- **Analysis** — Both accepted; P1 is a genuine conservativeness hole in the gate contract. Not infra.
- **Change** — plan v5 (targeted edits: §3 gates imputation contract, §4 J0 TRAIN-2000 fill rule, header/changelog). Resolutions appended to the review file.
- **Version Control** — committed with this entry; delta-only pass-5 re-review dispatched.
- **Next** — Pass-5 verdict (expected APPROVE-PLAN) → surface to Yixun with the full trail.

## 2026-08-04T07:45:00Z — Re-review pass 5: APPROVE-PLAN — plan cycle closed; surfaced to Yixun

- **Goal** — Verify P1/P2 closures and the v5 delta.
- **Result** — `passed`: **APPROVE-PLAN** ("P1 and P2 are fully closed, and the narrowly scoped v5 delta introduces no blocking defect"). Five-pass trail complete; all findings across all passes accepted and implemented, none rejected.
- **Analysis** — The plan cycle (SOP artifact 2+3) is closed: plan → review → resolutions → re-reviews → APPROVE. Total reviewer spend: 5 xhigh passes (~200k tokens each) — recorded for budget sizing.
- **Next** — Surface the approval package to Yixun: plan v5, review trail, §11 decision points (cohorts; L_null=16; J0+J1 approval; ε₀ fallback convention; pilot scope). Implementation (R1–R15) begins only on Yixun's approval; every TPU job additionally needs its own pre-launch approval per announcement 02.

## 2026-08-04T16:45:00Z — R1 `sigma-embed-noise` write phase complete (Opus Coder); review gate BLOCKED (issue #9)

- **Goal** — Round R1 per plan §6: noise-key derivation + golden fingerprints, `embed_null_tokens` + bf16 branch equality, σ-grid characterization, `base_context_fingerprint`.
- **Change** — 4 new files, uncommitted in the working tree pending review: `src/maxdiffusion/models/wan/null_inversion_wan.py` (45 exec LOC) + `test_null_adapter_sigma_grid.py` / `test_null_adapter_embed_tokens.py` / `test_null_adapter_noise.py` (195 exec LOC total, within the <200 budget; golden constants excluded).
- **Command / Validation** — TDD red evidenced (ModuleNotFoundError collection errors before implementation); **34 passed in 2.56s** (`PYTHONPATH=src pytest src/maxdiffusion/tests/worklogs_yixun/ -q -W error::DeprecationWarning`, scratchpad venv: Python 3.11.13, jax 0.10.2 CPU/threefry); ruff clean, py_compile clean, `git diff --check` clean. **Mutation evidence:** 8 targeted mutants (σ_max 0.999, fold_in order, dropped domain constant, seed change, tail-write, concat, forced bf16, missing fp32 upcast) — 7 killed immediately, the 8th (fingerprint upcast) survived and was killed by a new test. Golden fingerprints generated from an independent transcription of plan §3 (spec-pins-module, not module-pins-itself); ε₀ first values recorded.
- **Result** — `passed` (write phase). **Planner acceptances of the Coder's 5 flagged items:** (1) `make fixup` inoperable in this fork (missing `utils/`) — direct black+ruff equivalents accepted; (2) black-26.5.1 blank-line disagreement — repo convention (2 lines, ruff `lines-after-imports=2`) kept, accepted; (3) three in-scope test additions (fingerprint tests incl. upcast pin, fail-closed shape rejection in `embed_null_tokens`, σ-literal transcription guard) — accepted, they are TDD hygiene within R1's module; (4) `embed_null_tokens` output dtype = `promote_types(null, base)` — **accepted as the contract** (callers own dtype; the model consumes post-bf16-cast; branch-equality holds without hidden round-trips); (5) goldens are CPU-threefry — **added to the J1 smoke checklist: assert one golden on-device before the arms run**.
- **Analysis** — Round quality is high (mutation-tested pins). The round is NOT closed: the SOP requires the Codex review + strengthening before commit, and the reviewer is quota-blocked (issue #9 recurrence, reset Aug 7 ~23:35). No further Coder rounds start until this cycle closes.
- **Next** — On Yixun's issue-#9 decision (credits / wait / approved substitute): dispatch the R1 review, strengthen, commit, open R2 (`invert-trajectory`). exp_05 plan review queues behind the same decision.

## 2026-08-04T18:40:00Z — R1 cycle CLOSED: review (1 MAJOR + 3 MINOR) → strengthen (45 green) → commit

- **Goal** — Close R1 per the SOP cycle.
- **Command / Validation** — Codex review saved to `null_adapter_codex_code_sigma-embed-noise_review.md` (verdict REQUEST-REVISION; reviewer independently re-derived all six goldens and ran the suite). Strengthen: all 4 findings fixed (shape-arg removal + canonical-draw tests; non-ASCII golden; true bitwise bf16 comparison; little-endian fingerprint canonicalization), none rejected; mutants M9–M13 killed; **45 passed in 2.74s**; goldens and fingerprint digest unchanged. No behavior change beyond the findings ⇒ no follow-up review pass (SOP rule).
- **Result** — `passed`. R1 committed with this entry (module + 3 test files + review file with strengthening record).
- **Analysis** — The MAJOR was a real API hazard (batch-shaped global draws would have silently broken the ε₀ convention downstream). LOC overage (262 vs ~200) is entirely review-mandated tests — accepted.
- **Next** — Open R2 `invert-trajectory` (same Coder agent).

## 2026-08-04T20:20:00Z — R2 `invert-trajectory` write phase complete; review dispatched

- **Goal** — R2 per plan §6: the reverse-Euler pivot recurrence.
- **Change** — `invert_trajectory` + `_validate_sigmas` + constants in `null_inversion_wan.py` (+49 exec LOC; R1 code untouched; reuses `rollout_timesteps_from_sigmas` so inversion and replay share one timestep definition); new `test_null_adapter_invert_trajectory.py` (140 exec LOC; 21 tests). R2 total 189 exec LOC — within budget.
- **Command / Validation** — red evidenced (ImportError); **66 passed in 5.50s** (R1's 45 intact); ruff/py_compile/diff-check clean. **14 mutants all killed** (index shift, sign flip, evaluation point, delta shift, missing pins, un-flipped scan, n_hist=0, constant=999, no fp32 upcast, removed/weakened validation, removed guards). Coder self-caught three test weaknesses before reporting: circular NUM_TRAIN_TIMESTEPS expectation (now literal-pinned), and two `pytest.raises` satisfied by incidental downstream errors (now `match=`-pinned per guard).
- **Result** — `passed` (write phase). **Planner positions on the 6 flagged items:** (1) FMA tolerance (rtol 1e-6) in the literal-loop test — accepted, pre-authorized, ~50× the ≤2-ULP observed gap and orders below any structural error; frame-0 stays bitwise. (2) bitwise B-independence — noted, stronger than required. (3) eager host-side sigma validation ⇒ no traced grids — accepted contract; runner note recorded for R5. (4) private `_build_per_token_timestep` import — accepted (single source of truth). (5) `_f32_bits` duplicated in the new test file — deferred to the reviewer; default keep self-contained, hoist when a third consumer appears. (6) reference read from the main checkout at the pinned submodule SHA `f370228` (worktree submodule not initialized) — read-only, accepted.
- **Next** — Briefed Codex review of R2 (marker `invert-trajectory`) → strengthen → commit → R3 `optimize-nulls`.
