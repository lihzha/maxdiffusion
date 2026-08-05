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
