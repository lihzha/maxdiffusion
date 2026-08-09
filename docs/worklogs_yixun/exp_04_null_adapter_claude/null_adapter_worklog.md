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

## 2026-08-04T21:30:00Z — R2 cycle CLOSED: review (1 MAJOR + 1 MINOR) → strengthen (73 green) → commit

- **Goal** — Close R2 per the SOP cycle.
- **Command / Validation** — Review saved to `null_adapter_codex_code_invert-trajectory_review.md` (recurrence/pins/timestep-parity/scan structure all confirmed faithful; FMA tolerance accepted; N9/N11/N13 fixes verified real by the reviewer; `_f32_bits` ruling: keep duplicated). Strengthen: velocity-shape trace-time guard + σ-finiteness-first guard, 6 matched rejection tests + acceptance complement; guard-removal mutants pass the pre-strengthen suite 21/21 (gaps proven real) and fail 3 each now; **73 passed in 5.03s**.
- **Result** — `passed`. R2 committed with this entry.
- **Analysis** — The MAJOR was the silent-corruption class the plan's batching contract exists to prevent (batchless velocity ⇒ shared velocity across examples, shape-correct output). Fail-closed discipline holding.
- **Next** — R3 `optimize-nulls` (same Coder): the per-step Adam core — cached v_cond, inner-loop optax Adam on ∅, locked-∅ advance, warm start, [N,J,B] diagnostics, CFG algebra tests (w=1 ⇒ zero ∅ grad), tiny-WanModel gradient-flow smoke port.

## 2026-08-04T22:15:00Z — R3 `optimize-nulls` write phase complete; review dispatched

- **Goal** — R3 per plan §6: the per-step null-embedding Adam optimization (Mokady inner loop, batched).
- **Change** — `optimize_null_embeddings` + Adam constants + `_checked_velocity` (R2's shape guard extracted, shared, same message) in `null_inversion_wan.py` (+79 exec LOC); new `test_null_adapter_optimize_nulls.py` (239 exec LOC, 16 tests incl. the hand-rolled literal reference implementation and the tiny-WanModel smoke — real WanModel, 2-D timestep route, nonzero finite ∅ grads, no skip).
- **Command / Validation** — red evidenced; **89 passed in 13.31s** (73 prior intact); ruff/py_compile/diff-check clean. **Mutants:** 10 behavioral all killed (sign flip, per-inner-iter v_cond via call-count under disable_jit, pre-loop-∅ advance, mean-vs-sum objective, dropped pins ×2, carried Adam state, dropped warm start, unpinned pivot, global grad-norm); M11 (dropped stop_gradient on v_cond) survives as a semantic no-op — flagged, not hidden. Test-design notes: warm-start pinned by a composition test (tail re-run reproduces the full run's tail — kills warm-start AND carried-Adam mutants); z_bar reference tolerance rtol 1e-3 justified by CFG-amplified FMA differences (measured 7.6e-5 max rel, ~13× headroom).
- **Result** — `passed` (write phase). **Planner positions on the flagged items:** (1) generic relational context-dim validation — accepted (matches R1's embed contract; production passes 512×4096); (2) M11 stop_gradient kept as documentation-of-intent mirroring the reference's no_grad — reviewer to ratify or delete; (3) LOC overage (318 vs ~200) — accepted; the literal reference kills 9/10 mutants and is the round's strongest oracle; (4) losses[i,j] = pre-update loss, reference logging convention — accepted; (5) full-tensor loss incl. pinned frame 0 — accepted, already in the plan §8 parity register.
- **Next** — Briefed Codex review of R3 → strengthen → commit → R4 `replay-verifier-schema`.

## 2026-08-04T23:30:00Z — R3 cycle CLOSED: review (1 MAJOR + 3 MINOR, empirically-probed) → strengthen (96 green) → commit

- **Goal** — Close R3.
- **Command / Validation** — Review saved (`null_adapter_codex_code_optimize-nulls_review.md`): implementation confirmed algorithmically faithful on every §3 contract point; findings all test-strength, discovered by the reviewer's own adversarial mutations (zeroed timesteps / zeroed latents / eps perturbation). Rulings: keep stop_gradient; keep relational dims; LOC overage accepted. Strengthen: coupled oracle + timestep-content test (both reviewer probes now FAIL), pinned Adam recipe (eps_root explicit + discriminating fixture), fully independent literal reference (splice-bug cross-mutant caught), integral/finite argument guards. **96 passed in 13.05s**; mutants B1–B8 killed, original 10 re-verified.
- **Result** — `passed`. R3 committed with this entry.
- **Analysis** — The optimizer core — the function every arm and every cached target depends on — is now pinned by an independent reference sensitive to all of its inputs. This is the strongest-reviewed unit in the experiment so far, appropriately.
- **Next** — R4 `replay-verifier-schema`: `replay_with_nulls` (A0≡w=1 identity), the self-contained artifact record schema (latent_dtype, provenance, expected_final_latent+hash), and the replay verifier (consumes ONLY the record; provenance fail-closed).

## 2026-08-05T00:20:00Z — R4 split R4a/R4b/R4c (Planner-approved); R4a write phase complete; review dispatched

- **Goal** — R4 per plan §6; the Coder measured full scope at ~435 exec LOC and stopped at the ceiling per instruction, proposing a three-way split.
- **Result** — `passed` (R4a write). **Planner decisions:** (1) split APPROVED — R4a `replay-operator` (delivered), R4b `record-schema-io` (+ `_f32_bits` extraction, trigger fired: 4 consumers), R4c `verify-replay`; scope-neutral amendment noted in plan §6. (2) A0-identity tolerance rtol 1e-4 accepted — mechanism measured (ULP-level XLA scheduling difference between branches, amplified linearly by w: 6.3e-6@w5 → 7.7e-5@w50, vs real guidance at 59.9 absolute; contrast assertion retained); bitwise would force the operator to serve the test. (3) `nulls [N,L,D]` broadcast acceptance — accepted, mirrors R3. (4) replay B-independence test — deferred to the reviewer's ruling.
- **Change** — `replay_with_nulls` (+59 exec LOC); `test_null_adapter_replay.py` (170 exec LOC, 16 tests). **112 passed in 14.96s**; 8 mutants killed incl. the R3-vs-R4 asymmetry mutant (v_cond hoisted out of the scan — correct in R3, a bug here — caught by call-structure + A0 + literal-loop).
- **Next** — R4a review → strengthen → commit → R4b.

## 2026-08-05T01:20:00Z — R4a cycle CLOSED: review (2 test-strength MINORs) → strengthen (115 green) → commit

- **Goal** — Close R4a.
- **Command / Validation** — Review saved (`null_adapter_codex_code_replay-operator_review.md`): no operator defect; A0 tolerance/broadcast/split all ratified; reviewer's independent probes incl. bitwise batched-vs-singleton and A0-through-w=50. Strengthen: guard split with matched messages (reviewer's deletion probe now killed), bitwise B-composition test pinning R4c's B=1 verification path. **115 passed in 14.68s**; S1–S3 killed.
- **Result** — `passed`. R4a committed with this entry.
- **Next** — R4b `record-schema-io`: `null_adapter_records.py` (record schema per plan §4-P2, npz serialization, latent_dtype-derived byte-length validation, expected-latent hash, provenance header) + the `_f32_bits`/`_bf16_bits` test-helper extraction (4 consumers; R2 ruling's trigger fired).

## 2026-08-05T02:20:00Z — R4b `record-schema-io` write phase complete; LOC overage accepted; review dispatched

- **Goal** — R4b per the split: record/header schema + serialization + integrity, plus the bit-helper extraction.
- **Change** — `null_adapter_records.py` (197 exec LOC, provably numpy-only when loaded standalone), `test_null_adapter_records.py` (187, 23 cases), `bit_test_helpers.py` + mechanical import swap in 4 committed test files (−16 net, green-to-green under the 115-test characterization net per the SOP refactor rule).
- **Command / Validation** — red evidenced; **138 passed in 14.70s**; ruff (2 real findings fixed, not suppressed), py_compile, diff-check clean. **8 mutants killed** incl. B5 (hash over caller input instead of stored bytes) and B8 (dtype policy flipping z_i0/z_video — the plan-M5 violation); B6 (zip wall-clock timestamp) initially survived a luck-based determinism test — re-pinned to assert the fixed epoch mechanism.
- **Result** — `passed` (write). **Planner decisions:** (1) 394-LOC overage ACCEPTED — irreducible schema width; the only split seam (header vs record IO) is artificial mid-contract; recorded here per the SOP's "generally <200" guideline. (2) structural/relational shape validation with `PRODUCTION_*` constants + one real-geometry round-trip — accepted, follows the R3 ruling ("hard sizes belong at the runner/artifact boundary"); the runner (R6/R8) MUST assert the production constants — noted as an R6/R8 contract item. (3) local `sha256_of_array` restatement for jax-freedom — accepted, same byte discipline documented. (4) byte-deterministic zip (fixed epoch) — accepted, load-bearing for R8 markers. 
- **Next** — R4b review → strengthen → commit → R4c `verify-replay`.

## 2026-08-05T03:30:00Z — R4b cycle CLOSED: review (3 MAJOR + 3 MINOR) → strengthen (166 green) → commit

- **Goal** — Close R4b.
- **Command / Validation** — Review saved (`null_adapter_codex_code_record-schema-io_review.md`): field set/serialization/M5/determinism/import-swaps confirmed; schema ruled **R4c-sufficient** as a record+header pair; empirical probes caught the stale-hash writer, non-fail-closed boundaries, bool type-confusion, and a no-op endian canonicalization. Strengthen: all six closed; **166 passed in 14.64s**; mutants S1–S10 killed.
- **Result** — `passed`. R4b committed with this entry.
- **Analysis** — **Planner-decision reversal recorded:** my write-phase acceptance of structural/relational validation (deferring hard sizes to the runner) was overridden by finding 2 and the reversal is correct — the codec IS the production artifact boundary that R3's ruling pointed at; the runner assertion I'd planned would have protected one writer, not the readers R4c and the trainer depend on. The private `_Geometry` seam preserves tiny-shape testability without weakening the public contract.
- **Next** — R4c `verify-replay`: `verify_replay(record, header, velocity_fn, base_context, *, atol)` — pair-consuming, fingerprint-checked, provenance fail-closed, never reads z_video/actions, tamper-detection via atol.

## 2026-08-05T04:30:00Z — R4c `verify-replay` write phase complete; review dispatched

- **Goal** — R4c: the pair-consuming replay verifier (F14 contract).
- **Change** — new `null_adapter_verify.py` (60 exec LOC; separate module so `null_adapter_records.py` stays provably numpy-only — verified) + `test_null_adapter_verify.py` (251 exec LOC, 17 tests incl. the tampered-nulls-with-recomputed-hash security test, garbage-z_video pass-through, restricted-proxy structural test, and a production-geometry smoke that runs in-suite).
- **Command / Validation** — red evidenced; **183 passed in 16.08s**; ruff/py_compile/diff-check clean. Mutants: 11 — C1–C9, C11 killed (C6 latent_dtype-lie was a genuine test gap, closed); C10 (redundant defensive upcast) survives — kept per the R3 stop_gradient precedent, reviewer to ratify.
- **Result** — `passed` (write). **Two real discoveries:** (a) **writer contract** — expected_final_latent must be computed from the STORED (cast) tensors: cast → replay → record; fp16 round-trip on z_i0 alone shifts the endpoint 4.6e-4, so a pre-cast-replay writer emits unverifiable records. Documented in the verifier docstring; **binding on R6/R8** (recorded as a contract item). (b) guide_scale had no in-function second source ⇒ `expected_guide_scale` required keyword added (mirrors expected_model_revision). **Planner acceptances:** expected_guide_scale API extension; 311-LOC (test-fixture width; module 60); C10 retention pending ruling; cross-module private `_validate_header` import (single implementation, R2 precedent); shift-vs-CFG-weight constant separation.
- **Next** — R4c review → strengthen → commit → R5 `gates-module`.

## 2026-08-05T05:20:00Z — R4c cycle CLOSED: review (2 MAJOR + 1 MINOR) → strengthen (187 green) → commit — artifact contract complete

- **Goal** — Close R4c; with it, the R4 artifact contract (replay / store / verify) is complete end-to-end.
- **Command / Validation** — Review saved (`null_adapter_codex_code_verify-replay_review.md`); rulings: C10 KEEP, writer-side pin deferred to R6/R8 (with a writer-order mutation test there — **carried as an R6/R8 contract item together with the cast→replay→record order**), tampered-nulls test confirmed beyond reader protection. Strengthen: pair-level provenance (dtype-policy pair, expected arm/convention), hostile proxy (defeats getattr-default/vars), raising-callback ordering pins. **187 passed in 16.04s**; D1–D7 killed incl. both reviewer escape probes.
- **Result** — `passed`. R4c committed with this entry.
- **Next** — R5 `gates-module` (host-only statistics per plan §3), then R6–R8 runner (carrying the two writer contracts), then R9 manifests → parity audit → J0/J1 + exp_05 merge-1.

## 2026-08-05T06:20:00Z — R5 `gates-module` write phase complete; review dispatched

- **Goal** — R5: the host-only gate/statistics module (plan §3's authoritative contract).
- **Change** — `null_adapter_gates.py` (229 exec LOC, numpy-only, no prints/IO beyond `load_table`) + `test_null_adapter_gates.py` (252 exec LOC, 33 tests).
- **Command / Validation** — red evidenced; **220 passed in 16.12s**; ruff (2 real findings fixed)/py_compile/black/diff-check clean. **12 mutants, 0 survivors** — 4 closed real test gaps first: imputed-ratio value unpinned (E7), no-op boundary mutant (E9), CI-touching-zero unasserted (E12), and the unseeded-bootstrap mutant (E5) surviving twice behind degenerate resample statistics (constant data, then too-few distinct values) — fixed with a 16-distinct-value fixture and the reason documented in-test.
- **Result** — `passed` (write). **Planner rulings on flagged items:** (1) **Redundant-conjunct finding ACCEPTED as an analysis-integrity note:** on [0,1] SSIM the absolute floor (≥0.70) strictly implies the relative transfer test (≥0.7×A1) — rule kept exactly as written (belt-and-braces; the relative conjunct binds only if future variants relax the floor), BUT recorded: no analysis/report may ever attribute an A1 rejection to the relative test alone — the floor is always the binding constraint. Carried to `_analysis.md`/P4 as a standing note. (2) dict-shaped coverage semantics (three real failure modes) — accepted. (3) inclusive method-invalidity reading (MSE-or-SSIM invalid ⇒ impute) — accepted, it is the claim-penalizing direction. (4) zero-denominator MSE = invalid pair — accepted (an infinite ratio is an unbounded gift to the claim). (5) `verdicts_to_json` — accepted (determinism for the write-up).
- **Next** — R5 review → strengthen → commit → R6 `runner-capacity-core` (carrying the cast→replay→record writer contract + PRODUCTION geometry assertions + writer-order mutation test).

## 2026-08-05T07:20:00Z — R5 cycle CLOSED: review (3 MAJOR + 1 MINOR) → strengthen (246 green, 0 survivors) → commit

- **Goal** — Close R5; the gate/statistics layer is done.
- **Command / Validation** — Review saved (`null_adapter_codex_code_gates-module_review.md`); reviewer's empirical probes (missing-SSIM pass, k-set estimand flip, duplicate-JSON bypass, seed swap) all now killed; G3 imputation/percentiles/boundary/nanmean confirmed independently. **246 passed in 16.94s**; F1–F14 zero survivors.
- **Result** — `passed`. R5 committed with this entry. Gates API surface: convention-derived k-sets, table-consuming select_target, strict-JSON verdicts — the runner (R6) consumes exactly this.
- **Next** — R6 `runner-capacity-core`: pure orchestration of arms A0/A1/A1-probe/A2/A2-0/A2-probe + adequacy probe + gates-table emission + record building under the two carried writer contracts (cast→replay→record with its mutation test; PRODUCTION geometry assertions). Entrypoint/pyconfig wiring deferred to R10.

## 2026-08-05T08:10:00Z — Coder handoff: fresh Opus agent for R6 after two infra stalls

- **Goal** — Continue R6 after the persistent Coder agent stalled twice (stream-watchdog 600s timeouts).
- **Analysis** — **Infrastructure, not a bug**: the single Coder agent had accumulated ~680k transcript tokens across R1–R5; the stalls began immediately after the R6 handover, coinciding with a transient model-availability error in the harness. Worktree verified clean (no partial R6 files). Mitigation: retire that agent; hand R6 to a FRESH Opus Coder with a self-contained brief pointing at the committed code + review files for conventions (per the SOP's handoff-recording rule).
- **Result** — `fix_ready` (handoff recorded; fresh agent dispatched with the unchanged R6 scope).
- **Next** — R6 write → review → strengthen → commit, as planned.

## 2026-08-05T09:10:00Z — R6 `runner-capacity-core` write phase complete (fresh Coder); review dispatched

- **Goal** — R6: pure orchestration of the P1 arms, tables, records, adequacy probe.
- **Change** — `null_adapter_runner_core.py` (297 exec LOC) + `test_null_adapter_runner_core.py` (503 exec LOC, 61 tests, ALL at production geometry — the Coder dropped the tiny-geometry seam entirely since a full six-arm toy run costs ~1 s; shapes flow from the codec's PRODUCTION_GEOMETRY and cannot be weakened).
- **Command / Validation** — red evidenced; **307 passed in 26.4s** (246 inherited intact — no committed file touched); ruff/black/py_compile/diff-check clean. **24 mutants, 0 survivors** (writer-order ×2, frame-0 metric, probe seeding, pivot-recompute, arm cross-wiring, ε₀ assembly, k-set truncation, adoption rule ×4, plateau, coverage, sigma-header, nonfinite-score pool…). **R4c writer pin DISCHARGED**: cast→replay→record enforced in `build_capacity_records`; wrong-order records fail verify_replay (fp32 policy: correct = exactly 0.0 Δ, mutant = 1.3e-3 vs atol 1e-5).
- **Result** — `passed` (write). **Planner positions:** (1) no-tiny-seam design ACCEPTED (strictly stronger; +8 s suite). (2) header-derived, cross-checked record parameters ACCEPTED (verifiable by construction). (3) A1's `noise_convention="keyed"` describing the DEPLOYMENT convention (z_start is an inversion endpoint, not a draw) — flagged for the reviewer's ruling. (4) A2 = same targets traj[1:], new start ε₀ — ACCEPTED (plan §4-P1's meaning). (5–8, 10) fp32 final_future_mse, full_mse secondary in tables, seed-key/convention semantics, final_latents for R7, private _validate_header import — ACCEPTED. (9) **fp16 policy cannot discriminate writer order (measured)** — carried to R8 as a fidelity-gate/atol contract note; the fp32 writer-order test is the code-level pin. (11) adequacy-split seam noted for R7/R8 if LOC pressure arises.
- **Next** — R6 review → strengthen → commit → R7 `runner-decode-videos`.

## 2026-08-05T10:40:00Z — R6 cycle CLOSED: review (2 MAJOR + 2 MINOR) → strengthen (336 green, 0/38 survivors) → commit

- **Goal** — Close R6; the capacity orchestration core is done.
- **Command / Validation** — Review saved (`null_adapter_codex_code_runner-capacity-core_review.md`); arm wiring confirmed correct; rulings: A1 noise_convention=deployment APPROVED, A2 pivot reading APPROVED, table-convention semantics sound, writer-order test discriminating, no-tiny-seam accepted. Strengthen: provenance binding (params/context/batch fingerprints, pre-replay refusal), retained [N,J,B] diagnostics with hard finiteness, plateau boundary epsilon, pre-inversion recipe validation. **336 passed in 29.3s**; 38 mutants, 0 survivors.
- **Result** — `passed`. R6 committed with this entry.
- **Analysis** — Two carried items for later rounds: (i) **J1-robustness decision at R8**: `_checked_trace` makes one diverged example a hard run failure — correct-first, but R8's cache mode likely needs a per-example quarantine policy (mark-invalid + continue) to avoid a single pathological example killing a multi-hour job; to be designed WITH the reviewer at R8, never silently. (ii) R8/R10 writers must populate optimization_config = {inner_iters, lr} exactly. (iii) Adequacy split seam (92 LOC) ready if needed.
- **Next** — R7 `runner-decode-videos`.

## 2026-08-05T11:40:00Z — R7 `runner-decode-videos` write phase complete; review dispatched

- **Goal** — R7: pixel metrics + video helpers completing the gates tables.
- **Change** — `null_adapter_pixels.py` (187 exec LOC; numpy-only imports, skimage lazy, `generate_wan_side_adapter` NOT imported — its module-import side effect avoided) + `test_null_adapter_pixels.py` (39 tests).
- **Command / Validation** — red evidenced; **375 passed in 29.1s** (336 inherited unchanged); ruff/black/py_compile clean. **23 mutants, 0 survivors** (frame-0 inclusion, stride, win_size parity ×2, data_range, channel_axis, clipping, GT-decode-once, range/finiteness of decoders, fill mismatches ×3, caller-mutation, stacked order, fallback honesty…). End-to-end contract test: real R6 run → decode → fill → real gate_g1 with invalid_pairs == 0.
- **Result** — `passed` (write). **Planner positions:** (1) **scikit-image + imageio + imageio-ffmpeg are NOT declared anywhere in the repo** — carried as a HARD R10 launcher contract (ensure-install block + startup check, mirroring the issue-#8 ffmpeg lesson); (2) raise-instead-of-NaN on missing skimage/degenerate windows — ACCEPTED (an environment fault must not masquerade as a claim-penalized measurement); (3) fill enforces finiteness only, [0,1] semantics stay in the gates — ACCEPTED (division of labor; negative SSIM is a measurement); (4) constants restated locally with a cross-module equality test — ACCEPTED (R4b numpy-only precedent); (5) float64 MSE accumulation, all-four-metrics injection, pre-decode arm validation, fallback path honesty — ACCEPTED; (6) the end-to-end test importing R6's test module for its cached run — flagged for the reviewer (fixture-reuse vs test-coupling trade).
- **Next** — R7 review → strengthen → commit → R8 `runner-cache-resume` (+ the quarantine-policy design with the reviewer).

## 2026-08-05T12:30:00Z — CORRECTION to the R7 write-phase entry (append-only)

- The R7 entry's Planner position (1) stated scikit-image/imageio/imageio-ffmpeg are "NOT declared anywhere in the repo". **That was factually wrong** — the R7 reviewer verified all three are declared in `dependencies/requirements/base_requirements/requirements.txt` and the generated requirements. The R10 launcher contract is revised accordingly: preflight the actual TPU-host imports + the ffmpeg backend executable (the issue-#8 lesson — declared ≠ installed), but do not describe the packages as undeclared. The error originated in the Coder's environment report and was not independently checked by the Planner before recording — noted as a process reminder: verify repo-state claims against the repo.

## 2026-08-05T13:30:00Z — R7 cycle CLOSED: review (1 MAJOR + 2 MINOR + 1 NIT) → strengthen (392 green, 0/32) → commit

- **Goal** — Close R7; pixel metrics + videos done, gates tables completable.
- **Command / Validation** — Review saved; frame mapping independently confirmed (future = pixel frames 1–32); SSIM parity bit-exact; the dependency-premise NIT corrected in the appended worklog entry. Strengthen: pre-decode finiteness on everything, strict [0,1] decode contract with metric clipping removed (exact reference parity), transactional video publishing. **392 passed in 28.6s**; 32 mutants, 0 survivors.
- **Result** — `passed`. R7 committed with this entry.
- **Next** — R8 `runner-cache-resume`: shard writer (staging + completion markers, validated resume, coverage), the fp16 fidelity gate, the R6 writer contracts (optimization_config keys; cast→replay→record already pinned), and the QUARANTINE POLICY design for per-example divergence in cache mode — a plan-affecting decision to be ratified by the reviewer, never silent.

## 2026-08-05T14:40:00Z — R8 `runner-cache-resume` write phase complete; review dispatched (quarantine ratification requested)

- **Goal** — R8: the P2 shard writer, resume, fidelity gate, and the quarantine policy.
- **Change** — `null_adapter_shards.py` (295 exec LOC; TF lazy-imported in IO functions only) + `test_null_adapter_shards.py` (51 tests).
- **Command / Validation** — red evidenced; **443 passed in 30.8s** (392 inherited unchanged); ruff/black clean. **32 mutants, 0 survivors** — five closed mid-battery: two circular-expectation traps (thresholds tested via the constant), an absolute-vs-relative MSE coincidence at baseline 1.0, a bytes-perfect record-name swap, and an invalid-shard-with-parsing-marker resume path.
- **Result** — `passed` (write). **Quarantine policy implemented as designed + one addition flagged for reviewer ratification:** per-example retry-and-quarantine with marker listing, not-covered resume semantics, gates-coverage honesty (all three proofs tested) — PLUS batch-only failures (no example fails alone) RE-RAISE rather than quarantine-nothing, since per-example independence is a plan §3 contract and a composition bug must not be laundered. **Planner positions:** layout (one file per record + marker; object-store rationale) ACCEPTED; gfile rename copy+delete semantics documented, order-not-atomicity as the publish signal ACCEPTED; validate-returns/resume-raises division ACCEPTED; FIDELITY_BOUNDARY_ATOL declared ACCEPTED; one-header-per-shard provenance unit ACCEPTED; fully-quarantined-shard legality ACCEPTED; TF added to the scratch venv (declared repo dep) noted.
- **Next** — R8 review (quarantine ratification explicit) → strengthen → commit → R9 `manifests`.

## 2026-08-05T15:15:00Z — R8 review dispatch FAILED: Codex usage limit (issue #9, 4th recurrence)

- **Goal** — Run the R8 review (incl. the quarantine ratification).
- **Result** — `partial`: codex returned the usage-limit error before producing anything. R8's write phase (443 tests, 0/32 mutants) sits safely uncommitted in the worktree; R1–R7 are committed and pushed.
- **Analysis** — Infrastructure (reviewer quota), not a bug. Per the SOP's no-silent-substitution rule the cycle is blocked pending Yixun: credits / wait for the Aug 7 reset / approved substitute. Logged in `issue_report.md` with per-refill budget math (~12 xhigh passes per refill at this cadence).
- **Next** — On Yixun's decision: R8 review → R9 → parity audit → J0/J1 + exp_05 merge-1.

## 2026-08-05T17:20:00Z — R8 cycle CLOSED: review (6 MAJOR + 1 MINOR, mixed ratifications) → strengthen (523 green, 1 ratified survivor) → follow-up APPROVE → commit

- **Goal** — Close R8; the P2 storage/resume boundary is done.
- **Command / Validation** — Review + strengthening + follow-up all in `null_adapter_codex_code_runner-cache-resume_review.md`. Follow-up re-ran the original probes (path escape, shuffled bijection, cross-header resume, type confusion, cherry-picked subset, generic-exception quarantine, double-writer): all refused. Rulings landed: quarantine conditionally ratified → conditions implemented (typed divergence error, survivor rerun, real-header diagnostics); S2 KEEP (defence-in-depth precedent line: R3 M11, R4c C10, R8 S2). **523 passed in 35.9s**.
- **Result** — `passed`. R8 committed with this entry (module split recorded: shards=storage, cache_policy=judgments).
- **Analysis** — Runner obligations carried forward to R10/J-launch docs: free-space floor + stale-attempt sweeping; the runner raises ExampleDivergenceError for per-example pathology (trace non-finiteness wrap).
- **Next** — R9 `manifests` (the last code round before the parity audit → J0/J1 + exp_05 merge-1).

## 2026-08-05T18:30:00Z — R9 `manifests` write phase complete; review dispatched

- **Goal** — R9: the J0 cohort-manifest builder.
- **Change** — `build_null_adapter_manifests.py` (280 exec LOC) + `test_null_adapter_manifests.py` (41 tests, incl. scans over REAL producer-format TFRecords written with the production module's own helpers).
- **Command / Validation** — red evidenced; **564 passed in 35.5s** (523 inherited unchanged); ruff/black clean. **29 mutants, 1 documented no-op survivor** (M5, the literal min(2,·) clause — kept as refactor insurance, precedent line R3-M11/R4c-C10/R8-S2); one unreachable assertion deleted rather than retained (sizes validated at the artifact boundary where they can actually be wrong).
- **Result** — `passed` (write). **The episode-identity rule (the round's load-bearing derivation):** dual-source — name format `ep{ep}_v{view}_s{start:05d}` (minted at `make_droid_window_plan.py:162`; real examples cited from exp_02's manifest) AND `meta_json["episode_id"]` (minted at `precompute_features_droid_plan.py:156`, copied verbatim by the TFRecord producer); parse both, require agreement, use either alone, refuse records with neither; canonical preimage = the decimal string of the integer id, pinned by hand-computed sha256 digests. **Planner positions:** LOC +12% accepted (boundary validators); `sizer`/`split` injection accepted (cap-before-open requires it); uniform binding shape accepted; write-time binding merge (selection stays pure) accepted; disjoint fixture namespaces accepted.
- **Next** — R9 review → strengthen → commit → **J0 execution** (approved) + exp_05 merge-1; then R10/R11 → parity audit → J1.

## 2026-08-05T21:10:00Z — R9 cycle CLOSED: review (6 MAJOR) → strengthen → follow-up (2 residues + 1 new) → final fixes → commit

- **Goal** — Close R9; the cohort foundation is done. R1–R9 complete: the entire P1/P2 computational + evidence stack is built and reviewed.
- **Command / Validation** — Full trail in `null_adapter_codex_code_manifests_review.md` (review → strengthening → follow-up verdict → final fixes). Identity rule independently verified by the reviewer; N8/M5 survivors ratified; gsutil field-level parsing ratified. **610 passed in 39.6s**; 56 mutants; suite from 246 (R5 close) → 610.
- **Result** — `passed`. R9 committed with this entry.
- **Next** — (1) **J0 execution** (approved; acceptance criteria + `_command.md` entry at launch); (2) **exp_05 merge-1 + S1** (R9 boundary reached); (3) R10 `launchers-config` → R11 → parity audit → J1 package.

## 2026-08-05T04:50:00Z — J0 LAUNCH: cohort-manifest build (approved; conditional grant, Query 2)

*(Timestamp correction note: several preceding entries this session carry UTC stamps drifted a few hours forward of real time; ordering is correct throughout. Real UTC resumes here.)*

- **Goal** — Execute J0: build the immutable DEV-64/TEST-64/TRAINFIT-16/TRAIN-2000 manifests from the published dataset.
- **Acceptance criteria (predeclared):** (1) runs at commit `7199feb99514d5c4e460e84629b133566f6624d7`, clean worktree, host-only; (2) VAL scan = exactly 14,636 records over the 8 val shards; (3) TRAIN scan reaches ≥5,000 distinct episodes within the 200-shard/60-GiB caps; (4) staged publication completes (data first, `_COMPLETE` last) into `docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests/`; (5) `load_manifests` re-validates the published artifacts (64/64/16/2000, disjointness, ordering, bindings); (6) zero reauth-poisoned bindings — any binding failure aborts with nothing written (issue-#6 rule); (7) afterwards: mirror to `gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/` and commit the manifests.
- **Command / Validation** — `null_adapter_command.md` entry J0-1; log `null_adapter_2026-08-05_04:48:27.log`.
- **Result** — `launched`.
- **Analysis** — Auth verified live immediately before launch. Triage rule: reauth/network ⇒ infra (re-run); count/target mismatch ⇒ surface to Yixun (plan-constant vs dataset disagreement is a plan question, not a retry).
- **Next** — On success: mirror + commit manifests; exp_05 merge-1/S1 in parallel; R10 opens.

## 2026-08-05T05:15:00Z — J0 attempt 1 FAILED (infrastructure: ADC reauth) — fail-closed, nothing written

- **Goal** — J0 run per entry J0-1.
- **Result** — `partial`: TensorFlow's GCS layer (Application Default Credentials) failed with `invalid_rapt` reauth + anonymous-caller 401 at the very first listing; `build_j0_manifests` aborted before any scan; **nothing was written** (fail-closed as designed). Log: `null_adapter_2026-08-05_04:48:27.log`.
- **Analysis** — **Infrastructure** (issue #6 class, new sub-variant): gsutil's credential store was live (verified minutes earlier) but TF uses ADC, a separate token that had gone stale. Workaround: Yixun runs `gcloud auth application-default login` (distinct from `gcloud auth login`). No code change; the run re-executes unchanged per the auto-resubmit-on-infra rule once ADC is refreshed.
- **Next** — Await ADC refresh → re-run J0-1 verbatim (same commit, same command; will be recorded as J0-2).

## 2026-08-05T07:40:00Z — R10 `launchers-config` write phase complete (incl. mode bodies); review dispatched

- **Goal** — R10: config + entrypoint + launcher + the four execution-mode bodies (extended into the round by Planner ruling — J1's smoke exercises exactly that path).
- **Change** — `base_wan_5b_null_inversion.yml` (208 keys), `run_wan_null_inversion.py` (212 exec LOC; pure decisions + thin glue), **`null_adapter_modes.py` (new, 321 exec LOC — capacity/adequacy_probe/cache/verify_replay behind Backend+Sinks injectable seams; Coder split ratified, R8/R9 precedent)**, `bash_scripts/run_wan_null_inversion.sh` (preflight: imports + ffmpeg executable, install-or-die; HF prefetch first), 2 test files (80 tests).
- **Command / Validation** — reds evidenced; **690 passed in 52s** (676 → 690; pre-R10 610 unchanged); ruff/black/bash -n/diff-check clean. **38 mutants across R10, 0 survivors.** **Decode-range finding:** the pipeline emits [0,1] clamped float32 (traced: wan_pipeline.py:663-671 → video_processor:99-113 → image_processor denormalize+clamp :185-189) — R7's strict contract holds; the wrapper DECLARES the convention (`null_pixel_convention`) rather than sniffing.
- **Result** — `passed` (write). Planner ratifications: modes-module split; declared-convention wrapper; `null_verify_atol=1e-2` and 100-GiB floor as defaults; `# pragma: no cover` confined to `_load_backend`'s literal pipeline calls + one glob (the smoke rung's residue).
- **Next** — R10 review → strengthen → commit → R11 `a3-direct-opt` → parity audit → J0 re-run (pending ADC) → J1 package.

## 2026-08-05T10:30:00Z — Coder handoff #2: fresh agent for the R10 revision (context exhaustion, self-declared)

- **Goal** — Execute the R10 strengthen (BLOCKER + 8 MAJOR + 2 MINOR).
- **Analysis** — The R6-onward Coder (10 rounds, ~830k transcript tokens) HALTED before starting rather than half-delivering: "half-written code and an unrun battery in a round whose entire subject is fail-closed integrity is the one failure mode this experiment cannot afford." Verified clean state handed over (six untracked write-phase files byte-identical to the 690-green state; HEAD `1e64eb9`) plus a five-item trap list (modes↔entrypoint import cycle; the load-bearing divergence-inside-quarantine seam; findings 2+5 must be done together — publication resequencing changes shard numbering; keep ffmpeg local + gfile copy; black-vs-mutant-pattern interactions). Classified: infra/agent-lifecycle, exemplary honesty — the SOP's report-outcomes-faithfully norm working as intended.
- **Result** — `fix_ready` (fresh Coder dispatched with the review + trap list embedded).
- **Next** — R10 revision (findings 1–6 priority, 2+5 together) → battery → follow-up review → commit.

## 2026-08-05T17:20:00Z — R10 strengthen complete (11/11 findings); follow-up review dispatched

- **Goal** — Verify-and-close the R10 revision.
- **Change** — All 11 findings implemented per the review's concrete changes: real TI2V backend under axis_rules + manifest-bound read_batch (BLOCKER); capacity resequenced (nothing immutable before decode+gating; smoke limiter); adequacy preflight + full evidence persistence + adopted-recipe threading with the +2h stop; cache requires the selection artifact + fidelity-before-caching + selected-arm-only; verify fail-closed with exit semantics; 40-hex provenance + manifest digest + corrected default URI; transactional video publish; pragmas minimized; unit-range zero-tolerance; launcher tee/X_OK/preflight/on-device golden.
- **Command / Validation** — **812 passed** (690 → 812); ruff/bash -n/diff-check clean; 66 mutants (3 late survivors closed, all re-verified incl. the review's 4 named probes; M54 reproduced finding 8's local-`gs:`-directory bug exactly).
- **Result** — `passed` (strengthen). **Flagged for the follow-up reviewer (mandatory pass):** (1) additive edits to two SETTLED modules — `runner_core.arms=` kwarg (finding 4's requirement) and `shards.header_fingerprint/next_shard_index/supersede` (finding 5's requirement) — settled-module deltas need explicit review; (2) R8's ratified quarantine-then-published-raises behavior OVERTURNED by finding 5's concrete change — needs explicit re-ratification; (3) the BLOCKER's class fix pinned statically via ast (transformers absent from the venv) — first executed proof is J1's smoke; (4) `natural_context_length` zero-pad assumption (diagnostic-only) unverified against the real encoder — smoke item; (5) fidelity gate = real new J2 front-cost (plan-required, sized at 8 examples); (6) venv black ≠ repo black — formatting to confirm at commit.
- **Next** — Follow-up review → commit → R11 `a3-direct-opt` → parity audit; J0-2 running in parallel.

## 2026-08-05T18:50:00Z — J0-2 SUCCEEDED — cohorts are immutable and published

- **Goal** — J0 per entry J0-2.
- **Result** — `passed`, every predeclared criterion met: (2) VAL = **exactly 14,636 records** over 8 shards; (3) TRAIN target = **5,000 distinct episodes reached in 40 shards** (caps 200/60 GiB untouched); (4) staged publication complete (`_COMPLETE.json` last); (5) `load_manifests` re-validation PASSED (sizes 64/64/16/2000, disjointness, ordering, bindings — fail-closed loader); (6) zero binding failures; (7) mirrored to `gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/` and committed with this entry. Listing checksum `5827f4da…0d14`. Log `null_adapter_2026-08-05_15:27:35.log` (reconstituted from the task capture — the on-disk tee entry was unlinked mid-run by an unidentified actor while tee held the inode; content preserved via the duplicate capture stream; benign, noted).
- **Analysis** — DEV-64/TEST-64/TRAINFIT-16/TRAIN-2000 are now frozen artifacts. exp_05's K1 condition "J0 published" is MET. Wall time ~2h20m (network-bound; ~19 GiB read).
- **Next** — exp_04: R10 follow-up verdict → commit → R11 → parity audit → J1 package. exp_05: S5 next round; K1 conditions remaining = P0' green + parity audit.

## 2026-08-05T21:00:00Z — R10 cycle CLOSED (844 green, 79/79 mutants) → commit

- **Goal** — Close R10 after the deepest cycle of the experiment (review 1 BLOCKER + 8 MAJOR + 2 MINOR → strengthen 11/11 → follow-up 8/11+rulings → residues 4/4).
- **Result** — `passed`. R10 committed with this entry: config (208 keys), entrypoint (620 exec LOC), modes module (878), launcher (242), 2 test files; additive deltas to runner_core (`arms=`) and shards (`header_fingerprint`/`next_shard_index`/supersede — reviewer-ratified, incl. the R8-semantics reversal re-ratification). Suite 690 → **844**.
- **Analysis** — J1 is now launchable in structure: config → backend (TI2V class, statically pinned; first executed proof = smoke) → capacity with gates and provenance-bound artifacts. Remaining before the J1 package: R11 (A3 measurement module) + the parity audit.
- **Next** — R11 `a3-direct-opt`; exp_05 merge-2-interim (R10 boundary) unblocks S4.

## 2026-08-05T23:20:00Z — R11 `a3-direct-opt` write phase complete (the final exp_04 code round); review dispatched

- **Goal** — R11: the A3 joint optimizer + the J1b measurement helper.
- **Change** — `null_direct_opt_wan.py` (184 exec LOC: differentiable remat'd rollout, endpoint future-MSE, single-Adam joint optimization, `measure_single_update` with structured budget verdicts and the ≤4h projection rule) + 41 tests.
- **Command / Validation** — red evidenced; **885 passed** (844 + 41); ruff/diff-check clean. **28 mutants, 0 survivors** (4 first-pass survivors were test gaps, closed).
- **Result** — `passed` (write). **Two load-bearing test-design findings recorded for the experiment's method notes:** (1) Adam scale-invariance masks Σ-vs-mean objective mutants at the parameter level — the batching contract must be asserted on UNNORMALIZED grad norms; (2) a stop_gradient'd v_cond is forward-bit-identical and survives all self-referential comparisons — the CENTRAL-FINITE-DIFFERENCE test is the actual proof of end-to-end differentiation. Remat: numerically unobservable (bit-identical grads, measured) but structurally pinned via jaxpr inspection — tested, not documented away. verdict-vs-fits_budget separation ("measurement worked" ≠ "job affordable") accepted.
- **Next** — R11 review (the last exp_04 code review) → commit → **PARITY AUDIT** → the J1 pre-launch package.

## 2026-08-06T02:40:00Z — R11 cycle CLOSED (936 green, 73/73 mutants) → commit — ALL exp_04 CODE ROUNDS COMPLETE

- **Result** — `passed`. R11 committed with this entry (the A3 module + the J1/J1b wiring across modes/entrypoint/config/launcher). Suite: 246 (R5 era) → **936**. Eleven rounds + three splits, every commit through closed review cycles; ~470 mutants killed cumulatively; 6 ratified defence-in-depth survivors.
- **Next** — **THE PARITY AUDIT** (Planner, plan §8, recorded here before J1), then the J1 pre-launch package (params_set_up + command entry + acceptance criteria + pushed SHA) under Yixun's standing conditional grant.

## 2026-08-06T03:00:00Z — PARITY AUDIT (plan §8) — CLEAN; recorded before J1 per the SOP

Component-by-component against `third_party/Wan2.2/scripts/embedding_search.py` @ pinned `f370228`, citing where each is pinned (test + independent reviewer verification):

1. **Inversion recurrence** (:522-572 — indices, signs, evaluation point, pin points incl. the pinned clean pivot): R2's constant/analytic-oracle + scan≡literal-loop tests; R2 review independently verified; the reversed-dsigma class killed AGAIN at R11 via the bitwise oracle against the reviewed replay.
2. **Per-step null optimization** (:575-678 — fresh Adam per step, v_cond cached once, locked-∅ advance with one extra forward, warm start): R3's call-count (N cond / N·(J+1) unc), composition (tail-rerun), and locked-advance tests; R3 review's empirical probes; the Adam recipe literal-pinned (eps-sensitive fixture; eps_root=0.0 explicit; the reviewer's own torch-vs-optax fixed-gradient check at 7.15e-7).
3. **CFG combine + w=5**: R3/R4a analytic tests (w=1 zero-∅-grad; w=5 algebra); the A0 guide-scale-invariance contract (R4a) with the measured ULP-mechanism provenance.
4. **Pin discipline** (init / each candidate / each step / advance): mutants across R2/R3/R4a/R11 (dropped-pin variants all killed).
5. **Per-token timestep ≡ temp_ts** (:488-500): R2's captured-timestep content tests (frame-0 zeros, σ·1000 elsewhere, n_hist=1); R3's every-forward test; R7's independently-confirmed latent→pixel frame mapping.
6. **Dtype boundaries** (bf16 model fwd; fp32 latents/∅/Adam): R1's bitwise-bf16 branch equality at the exact cast; R3 fp32 pins; R7's strict [0,1] decode contract with the R10-verified pipeline trace.
7. **σ grid**: R1's hardcoded-value characterization incl. the 0.1724 tail; **documented deviation** σ₀=1.0 vs the PyTorch 0.999 (no cross-repo artifact exchange; every in-repo baseline uses ours).
8. **Optimization loss** = full-tensor MSE (pinned frame inert, matching `F.mse_loss`) with future-frame reporting split: R3's convention + R6/R7's metric separation.
9. **Replay ≡ regenerate_with_null_embeds** (:791-819): R4a review's line-by-line confirmation; R11's endpoint+trajectory bitwise oracle at four guide scales.
10. **Verifier ≡ verify_reconstruction_from_null spirit** (loads no GT, no trajectory): R4c's hostile-proxy must-not-read enforcement + pair-level provenance; tamper detection beyond reader hashes.
11. **Deviations register (all ratified in reviews):** empty positive branch (no captions exist); ∅ optimized as 16 rows inside the padded-512 context ({L_nat,16} ablation is diagnostic, in-J1); σ₀ above; batched execution with per-example independence (bitwise B-tests throughout); optax-vs-torch Adam (recipe-pinned); JAX threefry noise (golden-pinned incl. non-ASCII, one on-device golden asserted by the launcher before arms).

**Numeric-recipe defaults cross-check (SOP):** J=10, lr=1e-2, w=5.0, inversion w=1.0, Adam (0.9, 0.999, 1e-8, eps_root 0), σ-shift 5.0, 25 steps — all as the reference/plan; config values pinned by the R10 config-drift tests. **Data parity:** R9's dual-source episode identity independently re-derived by its reviewer; real producer-TFRecord fixtures in the manifest tests; J0's published cohorts re-validated by the fail-closed loader.

**Verdict: PARITY AUDIT CLEAN.** Launch precondition (P0 + audit) for the conditionally-granted J1 is now MET on the code side; the J1 package (params, command, acceptance criteria, smoke-first runbook) follows at the launch action.

## 2026-08-05T18:16Z — J1 LAUNCH (conditionally granted, Query 2; conditions met: P0 936-green + parity audit CLEAN)

- **Acceptance criteria (predeclared, plan §9 + SOP):** (1) worker reports commit `f06dfc1`; (2) v6e-8, 8 devices, 1 host; (3) SMOKE completes: ≥1 published smoke shard whose verify_replay passes on-device + the R1 golden asserted; (4) ADEQUACY publishes the adoption artifact with full [N,J,B] evidence; (5) CAPACITY completes all six arms on DEV-64+TRAINFIT-16 with zero unexplained quarantines, full-cohort decode, gates tables + selection.json + A3 measurement published provenance-bound; (6) no OOM/NaN (trace-finiteness hard-fails count as real bugs unless per-example divergence, which quarantines); (7) gates evaluated per G1/G2 + the target-selection rule — ANY outcome is acceptance (the gates decide, not vibes). Failure triage per the SOP: infra (preemption/download/auth) ⇒ auto-resubmit unchanged; real bug ⇒ fix cycle.
- **Command / Validation** — `null_adapter_command.md` entry J1-1 at launch time; queue job name `exp04-j1-null-yixun`.
- **Result** — `launched`.

## 2026-08-06T05:50:00Z — J1-1 attempt 1 FAILED: REAL BUG (HyperParameters getattr) — fix cycle opened

- **Result** — `partial`: job `20260805-181744-61377ea2` FAILED (APPLICATION_ERROR, worker exit 1, ~8 min into phase 1). **The smoke did its job**: the pipeline load succeeded end-to-end on first real contact (model shards loaded; revision `b8fff7315c…` resolved — the R10 BLOCKER fix held); the crash is `run_wan_null_inversion.py:611` — `getattr(config, "code_sha", "")` on `HyperParameters`, which raises `ValueError` for missing keys instead of honoring the default (`pyconfig.py:318`).
- **Analysis** — **REAL BUG**, not infra. The exact defect class exp_05's S4 review caught in its own dual-touch code (its finding 1); exp_04's occurrence was invisible because the entrypoint tests use fake config objects. Scope: audit EVERY config-getattr-with-default in the exp_04 entry/modes/launcher-adjacent files, not just :611.
- **Next** — Fix round (marker `hyperparameters-config-access`): a HyperParameters-safe optional-value resolution (independently implemented on this branch), full-site audit, real-HyperParameters regression tests (the AST-extraction pattern), review, commit → **J1-2 relaunch**. Grant reading recorded: Yixun's conditional grant re-evaluates at the fixed SHA (P0 green + audit intact ⇒ conditions re-met); flagged to Yixun in the status report with veto opportunity.

## 2026-08-06T06:40:00Z — Fix round CLOSED first-pass (APPROVE, zero findings) → commit → J1-2 relaunch

- **Result** — `passed`: **954 tests**; 11/11 mutants; the reviewer independently confirmed the repro, the 10-site audit, and the AST pins. Committed with this entry. J1-2 relaunches at this SHA (grant conditions re-met: suite green incl. the new real-class regressions; parity audit unchanged — the fix touches config access, not any audited component; Yixun's veto window was offered and stands open until launch).

## 2026-08-05T19:05:00Z — Timestamp correction (append-only) + J1-2 submission handed to Yixun

- **Correction:** the previous worklog entry is stamped `2026-08-06T06:40:00Z` and the J1-1 command-ledger header says `2026-08-06` — both wrong by one day. Actual times: fix-round close/commit ≈ **2026-08-05T18:55Z**; J1-1 submission **2026-08-05T18:17:48Z** (as the queue job id `20260805-181744-…` itself records). Content unaffected.
- **J1-2 status:** artifact root `gs://…/droid_wan_null_adapter/j1` confirmed EMPTY (J1-1 failed closed — nothing written), so the verbatim relaunch has no collision surface. The session's attempt to run `submit_j1.sh` was denied by the auto-mode permission classifier (issue #10, new) — submission handed to Yixun to run via `!`; the J1-2 command entry will be written at actual launch. Grant reading unchanged (conditions re-met at `9338c7b`; veto window stands).

## 2026-08-05T20:40:00Z — J1 runbook gap found and remediated: the TRAINFIT-16 capacity half was never in the runbook (Planner authoring error)

- **Finding (surfaced by exp_05's S10a review chain, reviewer-ratified):** cohorts are one-per-invocation (`plan_run` takes `config.null_cohort`; nothing in the production runner references `trainfit16`), so J1's phase 4 — capacity at the launcher default `NULL_COHORT=dev64` — covers only the DEV half of acceptance criterion 5 ("all six arms on DEV-64+TRAINFIT-16"). The J1-1/J1-2 runbook I authored never contained a TRAINFIT invocation; no review caught it (the R10 J1-readiness walk checked the smoke chain, not cohort coverage). Recorded as a Planner error, not a code defect: the runner supports `NULL_COHORT=trainfit16` as-is, and null adoption is cohort-unbound (`load_adoption` takes no manifest binding), so the DEV adequacy artifact is consumable by a TRAINFIT run today.
- **Remediation — J1-2b (supplemental, single invocation):** after J1-2 completes, run capacity with `NULL_COHORT=trainfit16`, `NULL_ADEQUACY_URI` = J1-2's published adequacy artifact, artifact root `…/j1/capacity_trainfit` (DISTINCT from the DEV root so the DEV-authoritative selection.json/tables cannot be overwritten — reviewer's operational note). Script archived as `submit_j1b_trainfit.sh` (scratchpad; reproduced in the command entry at launch). Grant reading: criterion 5's TRAINFIT half was inside the approved J1 acceptance criteria, so J1-2b completes the approved scope rather than extending it; Yixun executes the submission either way (issue #10), which is the sign-off.
- **J1-2 impact:** none — everything J1-2 is running remains required and its artifacts stay authoritative for DEV; criterion 5 is simply not fully dischargeable until J1-2b lands.

## 2026-08-06T20:15:00Z — J1 RESULT READING (P1 primary outcome; Planner) — G1/G2 FAIL, probes catastrophic, TARGET = STOP both cohorts; A3 measurement OK (J1b affordable)

**Per-arm mean future-SSIM (DEV-64 / TRAINFIT-16; full coverage, zero invalid pairs, 10k-resample CIs):**

| Arm | DEV-64 | TRAINFIT-16 | Role |
|---|---|---|---|
| A0 (base nulls, inversion-endpoint replay; CFG collapses) | **0.6665** | — | control |
| A1 (optimized ∅, own basin) | **0.8523** [0.8327, 0.8710], frac_impr 0.95, med ratio 3.605 | 0.8722 [0.8549, 0.8891], 1.00, 3.013 | **G1 FAIL (median_ratio below bar)** |
| A1-probe (locked ∅, keyed{0,1,2}) | **0.1729** (rel 0.203) | 0.1738 (rel 0.199) | transfer FAIL, both floors |
| A2-0 (base nulls, fresh ε₀) | 0.1423 | — | control |
| A2 (optimized ∅, fresh ε₀) | **0.4973** [0.4697, 0.5264], frac_impr 1.00, med ratio 10.2 | 0.4563 [0.4165, 0.4973], 1.00, 9.5 | **G2 FAIL (mean_ssim, ssim_ci_low)** |
| A2-probe | 0.2958 | — | diagnostic |

**Selection: STOP** (G1 median_ratio; probe below 0.7× and 0.70 abs; G2 mean/CI) — P2 target caching does NOT proceed per the predeclared rule. **A3 measurement: verdict `ok`, fits_budget TRUE** (compile 412s; 300 iters at job batch 8 in 2395s; peak HBM ~15.5 GB/device) — the conditional J1b direct-opt run is affordable within its ≤4 h projection budget.
**Scientific statement:** the null slot's own-basin optimization works (0.85 from a 0.67 CFG-collapsed control — real but modest, ratio 3.6 < the G1 bar) and its FRESH-NOISE optimization shows direction-without-magnitude: 100% of examples improve, 10× median MSE ratio over the 0.14 control, but the 0.50 absolute lands far below the 0.70 floor. Locked-null transfer is catastrophic (0.17 — locked wrong-basin nulls are actively destructive, WORSE than doing nothing). Joint reading with exp_05's K1 recorded in the master tracker + status report: the noise-basin problem is slot-universal at this recipe; the slots fail with opposite geometries (positive: huge in-basin ceiling 0.92, active harm from fresh noise; null: modest in-basin gain, genuine-but-insufficient fresh-noise steering).
**Open decisions (Yixun):** J1b GO/NO-GO (affordable; the one remaining mechanism question — whether endpoint-objective joint optimization beats the per-step-greedy A2 from fresh noise); both experiments' caching stages honor their STOPs; P4/P4' reports.

## 2026-08-06T22:30:00Z — J1b RESULT READING (P1b; Planner) — the fresh-noise ceiling was SUBSTANTIALLY A GREEDY ARTIFACT

**Same 8 DEV examples, same metric (latent future-MSE), all four regimes:**

| Regime | mean | per-example |
|---|---|---|
| A2-0: base nulls, fresh ε₀ | 4.95 | 5.16, 6.30, 4.52, 4.77, 4.77, 4.74, 4.72, 4.60 |
| A2: per-step greedy, fresh ε₀ | 0.429 | 0.50, 0.037, 0.49, 0.34, 0.45, 0.70, 0.57, 0.35 |
| **A3: joint endpoint, fresh ε₀ (J1b)** | **0.270** | 0.68, **0.0067**, 0.24, 0.35, 0.37, **0.13**, 0.29, **0.10** |
| A1: per-step greedy, own basin | 0.073 | 0.05, 0.004, 0.04, 0.15, 0.13, 0.05, 0.12, 0.05 |

(J1b's initial_loss reproduces A2-0's per-example values to ~1% — the fresh-noise starting point is consistent across jobs. Grad norms 9–17 → 0.03–0.31: most examples converged within the 300-iter budget.)

**Statement:** joint endpoint optimization through the full differentiable rollout beats the per-step greedy recipe on 6/8 examples (up to 5.6×) and — the headline — **reaches own-basin-quality reconstruction from fresh noise on 3/8 examples** (0.0067 / 0.10 / 0.13, inside A1's own range). The G2 ceiling (A2's 0.50 SSIM) was substantially an optimization-procedure artifact, not a fundamental basin wall: the null channel CAN steer a fresh basin most of the way to the clip. Convergence is uneven (ex-1 stalls at 0.68, above even A0's own-basin 0.43) and J1b measures latent MSE only (no decode ⇒ no SSIM).
**What this does and does not revive:** it reopens the null-target program's mechanism, but two gates stand between this and any P2b/P3 revival: (1) TRANSFER — A3's nulls are optimized per-ε₀; whether they survive foreign noise (the probe question that killed A1/A2) is UNMEASURED — answerable by a tiny J1c (replay `a3_nulls.npz` under keyed{0,1,2} + decode/SSIM, ~minutes of TPU); (2) COST — at the measured recipe, A3-caching TRAIN-2000 ≈ 225 v6e-8-hours (batch scaling could cut this ~2–4×; peak HBM 13 GB/16 GB leaves room). Decision to Yixun with the P4 plan.

## 2026-08-06T23:30:00Z — R12-lite `transfer-probe` cycle CLOSED first-pass (APPROVE; 989 green, 22/22 mutants) → commit → J1c launch

- **Result** — `passed`. The J1c mode: replay a3_nulls.npz (VERIFIED step-major [25,8,16,4096] — the batch-major misread the brief assumed would have transposed the study; killed as mutant X2) under global(0) own-ε₀ + keyed{0,1,2}, decode, future-SSIM/MSE, provenance-bound incl. the npz bytes' sha256. Reviewer independently verified J1b/J1c ε₀ identity (same imported global_noise(0)). Wiring +19/−1 across four exp_04-owned files. One first-pass battery survivor (X7, default-seam swap invisible to stubs) closed with a real-replay frame-0-pin test. Baseline note: pre-existing suite measures 955 vs the recorded 954 — unreconciled-benign (all green), flagged not chased.
- **Planner rulings:** four noise settings suffice; read_bytes as module seam; refusal over truncation — all reviewer-ratified.

## 2026-08-07T02:30:00Z — J1c RESULT READING (the transfer answer; Planner) — JOINT NULLS RETAIN ~72% RELATIVE EFFECT ACROSS BASINS; absolute floors still unmet

**8 examples × 4 settings (future-SSIM / future-MSE means):**

| Setting | mean SSIM | mean MSE | reference points |
|---|---|---|---|
| global(0) — own ε₀ | **0.651** | 0.273 | reproduces J1b (multiset match) |
| keyed(0) foreign | 0.471 | 1.315 | A1-probe (greedy nulls, foreign): **0.173** |
| keyed(1) foreign | 0.477 | 1.423 | A2 (greedy, fresh): 0.497 |
| keyed(2) foreign | 0.478 | 0.989 | base nulls, fresh: MSE ~4.95 |

**Statement:** (1) Jointly-optimized nulls TRANSFER — foreign-basin SSIM ~0.47, stable across all three seeds, 2.8× the greedy nulls' 0.17 collapse, and MSE ~1.0–1.4 vs the do-nothing 4.95. **Relative retention 0.47/0.65 = 0.72 — above the 0.7× relative bar.** The basin-boundness that killed A1/A2 was substantially a GREEDY-OPTIMIZATION artifact on BOTH axes (capacity and transfer). (2) The absolute floor (0.70) is still unmet everywhere: own-basin 0.65, foreign 0.47 — at 300 iters the joint optimum itself is not yet deployment-grade, and per-example the pattern is consistent: well-converged examples (own SSIM 0.83–0.99) retain 0.58–0.88 under foreign noise; poorly-converged ones are weak everywhere. Transfer quality tracks optimization quality.
**Mechanism chain now COMPLETE (the P1/P1b/P1c arc):** greedy per-step nulls = basin-locked AND capacity-limited → joint endpoint optimization lifts both (own-basin to near-A1 on converged examples; 72% relative transfer) → the remaining gap is optimization quality/budget, not a basin wall. **Every predeclared exp_04 gate has fired (STOP / below-floor): the experiment's question set is fully answered.** A revival (longer/better joint optimization → caching at ~225+ v6e-8-h → P3) is an exp_06-scale NEW proposal, not a continuation under these gates.

## 2026-08-08T07:30:00Z — P4 audit corrections (append-only; found by the report Coder against primary artifacts, Planner-verified readings)

Four corrections to previously recorded statements. The drafted `null_adapter_results.md`/`null_adapter_analysis.md` carry the full evidence; review of both is deferred (Codex quota triage).

1. **MATERIAL — J1-4's capacity gates ran at a recipe the adequacy probe had REJECTED.** The probe adopted J=50/lr=0.01; capacity ran at the default **J=10** — confirmed in `run_report.json` AND the shard provenance headers. Cause: `bash_scripts/run_wan_null_inversion.sh` **never passes `null_adequacy_uri`** (no env mapping), so the YAML default `''` made `load_adoption` return None and the runbook's `NULL_ADEQUACY_URI` env was silently ignored. exp_05's launcher DOES wire it (line 318) and K1 honored its adoption — the two experiments' in-basin numbers are therefore not recipe-matched. Consequences: **G1's FAIL margin (median_ratio 3.605 vs bar 5) is contaminated** — it was measured against a recipe 2.57× weaker than adopted; **the STOP itself survives** because it is over-determined by the A1-probe floors (0.1729 vs 0.70 abs / 0.7× rel), which no recipe change could plausibly bridge; `null_adapter_params_set_up.md`'s contrary assertion is WRONG. The affordable clean re-run (~1.4 h v6e-8, needs a launcher fix cycle first) is a DECISION FOR YIXUN, queued in the morning summary — not launched under the sleep grant because it requires editing a settled exp_04 file and is off exp_06's critical path.
2. **The J1-4 reading's "locked wrong-basin nulls are actively destructive, WORSE than doing nothing" is UNSUPPORTED and retracted.** A1-probe 0.1729 is +0.031 ABOVE the A2-0 do-nothing reference (0.1423). The supported claim: locked nulls are nearly **INERT** under foreign noise. ("Actively destructive" is true of exp_05's B2 — 0.1610 vs its 0.2814 reference, 0/64 improved — and was transposed.) The inert-vs-harmful distinction matters: it is exactly what the opposite-geometries argument turns on.
3. **The J1-4 command entry records tip `27efcd1`; the artifacts record `3bdbd2a`.** Verified a docs-only descendant (`git diff -- src bash_scripts` empty), so every substantive claim stands; the ledger figure is corrected by this note.
4. (exp_05-side, recorded there: B0 = **0.3215**, not "≈0.25"; B1/B0 = **2.87×**, not 3.6×.)

## 2026-08-09T (local) — `adequacy-wiring` fix CLOSED first-pass (APPROVE, zero findings; 991 green, 6/6) → commit → CLEAN-GATE RERUN (J1-5) launches under Yixun's approval

- The 3-line surgical fix (default/echo/override, each beside its NULL_SELECTION_URI sibling) + static-source tests incl. the uniqueness regex that closes the mistyped-expansion hole — the silent failure mode that cost four jobs. Issue #12's xtrace defect deliberately untouched.
- **Stakes upgraded by the P4 analysis review:** the J=10 STOP was ruled formally INDETERMINATE (A1-probe is budget-dependent; A2's J=50 G2 unmeasured), so this rerun — capacity dev64 + trainfit16 at the ADOPTED J=50/lr=0.01, attempt-scoped roots — now DECIDES exp_04's plan-compliant target selection: either the predeclared STOP is retained on clean measurements, or an arm is selected and the P2 question reopens. Yixun's approval: "exp_04 clean_gate rerun" (mid-turn, 2026-08-09).
