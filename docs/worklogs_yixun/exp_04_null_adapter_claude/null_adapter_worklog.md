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
