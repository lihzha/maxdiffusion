# plan_null_adapter — exp_04: Null-text inversion port + action-conditioned null-embedding adapter

Planner: Claude Fable 5 (max effort). Status: **v3 — revised per Codex re-review of v2 (7 partials closed, N1–N9 addressed); awaiting re-review pass 3, then Yixun approval.**
Branch `claude-exp_04_null_adapter-20260803` off `yixun-dev` @ `744094a`. Worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter`.

v3 changelog vs v2 (see `null_adapter_codex_plan_review.md` for both review passes): named noise conventions replacing the ε₀/keyed conflict (N3, F10); target-selection rule requires G1 + absolute probe floor (N1); J0 manifest job with bounded scan + TRAIN-2000 manifest (N2); host-only gates module with executable statistics + test (N4, F6); L_null ablation declared diagnostic-only, L=16 fixed (N5); adapter/P3a fully pinned incl. budgets, seeds, init, ~9M count, clean-restart rule (N6, F9); legacy pre_context restore contract + RNG-replicated old-vs-new evaluator parity at identical z_start (N7, F15); `expected_final_latent` added to the artifact schema + worst-example fp16 gate + tests (N8, F14); oracle-rollout relabeling per convention (N9); adequacy-probe gradient diagnostics + recipe-adoption rule (F1); multi-seed probes with fixed k-sets (F5); G2 CI + G3 seed reduction + executable invalid-pair semantics (F6); A3 numerical budget stops (F11); rounds re-split again with per-round tests (F17).

References: `inverse_DDIM_pdf.md` (Mokady et al. 2022); `third_party/Wan2.2/scripts/embedding_search.py`, `verify_reconstruction_from_null.py`, `embedding_search_smoke.py` (submodule pin `f370228`).

---

## 1. Background and motivation

Unchanged from v2: pre_context adapter reaches mean rollout SSIM 0.2946 (step 30k, 4-sample validation); the PyTorch fork's positive-embedding inversion line underfit and established the noise-basin problem (own-`z_init` latent MSE ≈ 0.015–0.022 vs fresh-noise ≈ 2.0–3.3; `Wan2.2/docs/adaptor_design.md` §3.11–3.14). Null-text inversion proper was never run on DROID; no network anywhere predicts null embeddings; maxdiffusion has no inversion code. Structural fit: the side-adapter path uses T5("") in both CFG branches (`wan_ti2v_side_adapter_trainer.py:313-325`, w=5.0); deployment here needs no modules inside the DiT.

## 2. Questions

- **Q1 (achieved reconstruction bound):** per-example optimized oracle under a declared recipe — not a capacity upper bound; G1 failure interpretable only after the adequacy probe (§4-P1).
- **Q2 (basin/transfer):** do optimized ∅_t transfer to fresh noise? Measured (multi-seed) before any adapter spend; A2 (global-noise convention) with matched control A2-0 is the fallback.
- **Q3 (amortization):** does a small action-conditioned ∅̂_t predictor beat pre_context on the locked TEST cohort under noise-matched, achieved-quality comparison?

## 3. Method

Core math unchanged from v2 (§3 v2): normalized latents `[B,48,9,12,20]` fp32; `apply_first_frame_pin`; `build_rollout_sigmas(25, 5.0, 0.0, 1.0)`; per-token timestep (`n_hist=1`); bf16 model forward, fp32 latent/optimizer arithmetic; `embed(∅)` = T5("") `[512,4096]` with rows `[0:L_null]` replaced, `∅_init = T5("")[0:L_null]`, bf16-boundary branch-equality test; inversion at w=1; per-step null optimization at w=5 (fresh Adam per step, cached `v_cond`, locked-∅ advance, warm start; J=10, lr=1e-2 defaults); CFG Euler replay. A1 is named **"empty-positive null-branch inversion (Mokady-style)"**; deviations register in §8.

**Noise conventions (N3; single source of truth, stored in every artifact/checkpoint, evaluator rejects mismatch):**
- `noise=keyed(k)`: per-example — `jax.random.fold_in(fold_in(PRNGKey(2026), int.from_bytes(sha256(name)[:8])), k)` → standard normal of latent shape. Order/batch-size invariant by construction (tested).
- `noise=global(k)`: one draw shared by all examples — same formula with the literal name `"GLOBAL"`. ε₀ ≜ global(0).
- Fixed k-sets (F5, F6): A1-probe k∈{0,1,2} keyed; A2-probe k∈{0,1,2} keyed; A2/A2-0 use global(0); J4/J5 deployment eval k∈{0,1,2} of the convention selected at the P1 gate. Seed reduction everywhere: per-example mean over the k-set, then paired aggregation over examples.

**Batching contract** unchanged from v2 (single batched call, Σ-over-batch of per-example means, per-example Adam moments, losses `[N,J,B]`, no `vmap(grad)` over the model, FSDP shardings asserted, restricted donation, B-independence + layout tests).

**Gate statistics are code, not prose (N4):** host-only module `null_adapter_gates.py` — inputs: per-example per-seed metric tables (JSON); outputs: gate verdicts. Semantics: paired unit = example; bootstrap = 10k resamples, RNG seed 20260804, percentile CIs; coverage assertion — every manifest name present exactly once, else the gate FAILS; invalid pair = any nonfinite/missing metric for either method → counted as not-improved in fraction conditions, excluded from mean/CI, and the gate auto-FAILS if > 10% of pairs are invalid. Unit-tested (`test_null_adapter_gates.py`) with synthetic tables incl. the invalidity and coverage branches.

## 4. Phases, arms, and gates

### J0 — Cohort manifests (before P1; N2, F4)
`build_null_adapter_manifests.py` (host-only; no TPU; network-bounded; runs locally or on a TPU-host prelude — listed in §9 and approved like any job). VAL: full scan (14,636 records, ≈3.3 GiB) → parse `name`/`ordinal`, group by episode. TRAIN: **bounded early-stop scan** — stream shards in deterministic listing order, stop once ≥ 5,000 distinct episodes seen (margin over the 2,000×(≤2/episode) need); never a full 330-GiB pass. Emits immutable manifests (committed to the exp folder + mirrored to GCS): **DEV-64**, **TEST-64** (val; episode-disjoint; one window per episode — the window with the lowest `ordinal`; episodes ordered by sha256(episode_id)), **TRAINFIT-16** (train), **TRAIN-2000** (train; ≤2 windows/episode, same deterministic rules). Each manifest row binds: split, name, episode, ordinal, source shard path + GCS generation id; the manifest header binds builder code SHA + shard-listing checksum. Selection uses DEV only; G3 uses TEST only; pilot-scope statement stands.

### P0 — TDD infrastructure (no TPU)
As v2: constant/analytic velocity oracles pin recurrence indices/signs (scan ≡ literal Python loop elementwise); convex-toy optimization (final < initial); tiny-random-WanModel covers ∅-gradient flow (smoke port), the 2-D timestep route, pin/shape invariants; plus the gates module tests (§3) and noise-convention invariance tests.

### P1 — Reconstruction study + basin probe (job J1, v6e-8; **needs approval**)
On DEV-64 + TRAINFIT-16 (arms as v2): A0 (frozen-∅ replay from traj[0]); A1 (optimize from traj[0], replay from traj[0]); A1-probe (A1 nulls from keyed(0..2)); A2 (optimize from ε₀=global(0), replay from ε₀); **A2-0** (frozen ∅ from ε₀); A2-probe (A2 nulls from keyed(0..2)). Inside J1:
- **Adequacy probe (F1):** 8 DEV examples × J∈{10,25,50} × lr∈{1e-2,3e-2}, logging per-step final losses **and per-inner-iter grad-norm traces**. **Adoption rule:** if any probe recipe improves median final tracking loss ≥ 2× over (J=10, lr=1e-2), adopt it and re-run A1/A2 on the full DEV cohort under the adopted recipe **before** gating; G1/G2 are evaluated once, on the adopted recipe only (no post-hoc shopping). Plateau rule for interpreting failure: < 10% median improvement J=25→50 ⇒ "reconstruction-limited", else "recipe-limited".
- **L_null ablation (N5): diagnostic-only.** L_null ∈ {L_nat(T5("")), 16} on the same 8 examples, reported in `_results.md`; **L=16 is fixed for P2/P3 regardless** (revisiting it would be a Yixun decision that reopens the plan). Cache/checkpoint metadata still record L_null; adapter/evaluator assert cache L_null == 16.
- **A3 single-update measurement (F11), numerical stops:** abort the measurement if compile > 30 min, one update > 120 s, or OOM; J1b is only proposed if the projection fits ≤ 4 h wall on a v6e-8.

**Metrics** as v2 (future-frame non-pinned latent MSE primary; decoded-latent GT reconstruction SSIM, future-frame primary; full-cohort decode; unpinned Gaussianity primary). Every arm's per-example per-seed metrics land in the JSON tables the gates module consumes.

**Gates (evaluated by `null_adapter_gates.py`):**
- **G1 (A1 vs A0, DEV-64, paired):** median ratio future-MSE(A0)/future-MSE(A1) ≥ 5 AND ≥ 80% improved AND mean future-SSIM(A1) ≥ 0.80 with 95% CI lower bound ≥ 0.75.
- **G2 (A2 vs A2-0, from ε₀, paired):** median ratio ≥ 5 AND ≥ 80% improved AND mean future-SSIM(A2) ≥ 0.75 with 95% CI lower bound ≥ 0.70 (F6: CI added).
- **Target-selection rule (N1):** select A1 (deployment `noise=keyed`) iff **G1 passes AND** A1-probe passes both: mean future-SSIM(A1-probe) ≥ 0.7 × mean future-SSIM(A1) **and absolute floor mean future-SSIM(A1-probe) ≥ 0.70**. Else select A2 (deployment `noise=global`) iff **G2 passes**. Else stop after P1: report Q1/Q2 with adequacy scoping; J1b decision to Yixun.

### P1b — A3 (job J1b, conditional, separately approved) — unchanged from v2, now with the §4-P1 numerical stops and the ≤ 4 h sizing rule.

### P2 — Target caching (job J2, gated; **needs approval**)
Chosen arm over TRAIN-2000 + DEV-64 + TEST-64. **Record schema (F7, N8):** `name, ordinal, split, episode, z_i0 (fp16), actions (fp32), z_video (fp16), nulls [25,16,4096], z_start (fp16), noise_convention, arm, per_step_final_losses [25], final_future_mse, expected_final_latent (fp16) + sha256(expected_final_latent bytes)`. Shard provenance header as v2 (manifest hash, code SHA, model revision, σ vector, w, base-context fingerprint, optimization config, dtype policy, L_null). Integrity as v2 (staging + completion markers with count/sha256/fingerprint; validated-marker-only resume; unique names; exact manifest coverage). **fp16 fidelity gate (N8):** predeclared subset = first 8 DEV-manifest examples; thresholds are **worst-example maxima**: max ΔSSIM ≤ 0.01 and max future-MSE increase ≤ 5% between in-memory fp32 replay and serialized-fp16 replay, else store fp32; enforced by `test_null_adapter_artifact_io.py` (threshold + fallback branches). Cross-example ∅* cosine/PCA structure diagnostic reported before J3.

### P3 — Adapter training + eval (jobs J3/J4/J5; **needs approval**)
**Model (N6, fully pinned):** `NNXNullEmbedAdapter`. Action path: `a − a₀` → `Linear(7→512)` + learned temporal pos (32×512) → 32 tokens. Image path: `z_i0` → 240 spatial tokens → `Linear(48→512)` + fixed 2-D sin-cos pos. Memory = concat (272 tokens). Queries: learned `[25·16, 512]` embeddings + sinusoidal step embedding (128→512 projection, added); **no query self-attention** (queries attend to memory only). Two pre-norm cross-attn blocks: LayerNorm(eps 1e-6) → MHA(d=512, 8 heads) → residual → LayerNorm → FFN(512→2048→512, GELU) → residual; dropout 0.0. Final LayerNorm; head `Linear(512→4096)` zero-init; output `∅̂_t = T5("")[0:16] + Δ̂_t`. Initializers: Xavier-uniform linears, N(0, 0.02) learned embeddings, zeros head; init RNG seed = `config.seed` (0). **≈ 9M params** (queries 0.20M + pos/proj 0.10M + input linears 0.03M + 2 blocks ≈ 6.3M + head 2.1M + biases/norms), fp32. Zero-init two-step gradient behavior tested.
**Trainer:** as v2 (self-contained P2 records; adamw lr 1e-4, wd 0.01, betas (0.9,0.95), 1k warmup, cosine to 0.1×, global batch 256, fixed 30k steps; DEV-target eval every 1k; selection = best DEV embedding MSE; Orbax + `metadata.json` {model_type, arch, cache-manifest hash, model revision, noise_convention, L_null, code SHA}, written atomically with each step's checkpoint (N7)). Data shuffle seed = `config.seed + 1`.
**P3a learnability gate (N6):** 32 real cached examples; **max 2,000 steps**, batch 32, lr 3e-4 (declared override), same arch/seed; targets: embedding MSE ≤ 10% of the ∅*-variance baseline AND rollout ΔSSIM ≤ 0.02 vs serialized-target replay on those examples. **J3's full run restarts from a clean initialization; P3a artifacts are diagnostic-only and discarded.** P3a failure ⇒ stop, report.
**Eval (J4/J5, evaluator `generate_wan_null_adapter.py`; N7, N9):** methods on TEST-64 — adapter ∅̂; **serialized-target replay** of cached ∅* (labeled "reference point", not an upper bound, under `noise=keyed`; under `noise=global` from ε₀ it is the achieved per-example oracle for that ε₀); null-only; pre_context@30k. Noise: the selected convention, k∈{0,1,2}, identical `z_start` per (example, k) across all methods. **Legacy restore contract (N7):** pre_context checkpoint (Orbax `params/opt_state/step` only) restored against the pinned `base_wan_5b_side_adapter.yml` config with an `nnx.eval_shape` param-tree structure match, step==30000 assertion, and a param-tree sha256 fingerprint sidecar written on first restore and asserted afterwards. **Anchor + parity (N7):** (a) re-run the unchanged `generate_wan_side_adapter.py` 4-sample validation at the current commit; require |ΔSSIM| ≤ 0.01 per sample vs the stored step-30000 summary; (b) new evaluator's `pre_context` mode **replicates the old script's RNG derivation** (same `jax.random.split` chain from the same config seed) so both produce identical `z_start` on those 4 samples with the same restored params, same σ grid; require final-latent max-abs diff ≤ 1e-2 (bf16 rollout) and |ΔSSIM| ≤ 0.005; only then do J4/J5 numbers count.
**Gate G3 (TEST-64, gates module):** adapter vs null-only: paired mean future-SSIM ≥ +0.05 with 95% CI excluding 0; success vs pre_context: paired same-noise mean future-SSIM ≥ +0.02, 95% CI excluding 0, ≥ 60% examples improved. Seed reduction and invalid-pair semantics per §3. Achieved-quality framing.

### P4 — Results, analysis, HTML reports — unchanged from v2.

## 5. Planned code, per file

1. **(N) `src/maxdiffusion/models/wan/null_inversion_wan.py`** — `embed_null_tokens`, `base_context_fingerprint`, `invert_trajectory`, `optimize_null_embeddings` (batched; per-example losses `[N,J,B]`; grad-norm traces), `replay_with_nulls`, noise-convention helpers (`keyed_noise`, `global_noise`).
2. **(N) `src/maxdiffusion/models/wan/null_direct_opt_wan.py`** — A3 + single-update measurement helper with numerical stops.
3. **(N) `src/maxdiffusion/data_preprocessing/build_null_adapter_manifests.py`** — J0 builder per §4 (bounded scan, generation-id binding).
4. **(N) `src/maxdiffusion/null_adapter_gates.py`** — host-only gate/statistics module per §3.
5. **(N) `src/maxdiffusion/run_wan_null_inversion.py`** — driver; modes `capacity | cache | verify_replay | adequacy_probe`; artifact schema incl. `expected_final_latent`; staging/marker publish; validated resume; metric-table JSON emission; decode/videos.
6. **(N) `src/maxdiffusion/configs/base_wan_5b_null_inversion.yml`** — keys as v2 + `noise_convention`, manifest paths, gate-table paths.
7. **(N) `src/maxdiffusion/models/wan/null_embed_adapter.py`** — per §4-P3.
8. **(N) `src/maxdiffusion/trainers/wan_null_embed_adapter_trainer.py`** — per §4-P3.
9. **(E) `src/maxdiffusion/train_wan.py`** — dispatch `NULL_EMBED_TI2V`.
10. **(N) `src/maxdiffusion/generate_wan_null_adapter.py`** — evaluator: metadata-checked restore + legacy pre_context restore contract; modes `adapter | target_replay | null_only | pre_context | pre_context_legacy_parity`; convention-checked noise; imports decode/metric/video helpers.
11. **(N) bash launchers** — `run_wan_null_inversion.sh`, `train_wan_null_adapter.sh`, `validate_wan_null_adapter.sh`.
12. **(N) tests** (`src/maxdiffusion/tests/worklogs_yixun/`): as v2 §5.11 plus `test_null_adapter_gates.py` (semantics incl. coverage/invalidity), `test_null_adapter_noise.py` (convention determinism, order/batch invariance, k-set separation), fidelity-gate threshold/fallback tests, evaluator RNG-replication parity test (against a recorded old-script noise fingerprint on a tiny config), legacy restore structure/fingerprint rejection tests.

Replay verifier as v2, now consuming the schema's `expected_final_latent` (+hash) — never `z_video`/trajectory; provenance mismatch ⇒ hard error.

## 6. Coder rounds (each < 200 LOC with its own named tests; F17)

R1 `sigma-embed-noise` (grid/embed/noise helpers + tests). R2 `invert-trajectory`. R3 `optimize-nulls`. R4 `replay-verifier-schema`. R5 `gates-module`. R6 `runner-capacity-core` (arm execution + metric tables). R7 `runner-decode-videos`. R8 `runner-cache-resume` (staging/markers/fidelity mode). R9 `manifests` (J0 builder). R10 `launchers-config`. R11 `a3-direct-opt`. R12 `adapter-module`. R13 `trainer-dispatch`. R14 `evaluator-restore` (typed + legacy contract). R15 `evaluator-modes-noise` (incl. parity mode). R12–R15 start only after the P1 gate outcome. Each round: Opus-Coder test-first → briefed Codex review → strengthening record → commit.

## 7. Validation ladder

As v2, with: rung 3 additionally validates J0 manifests against a hand-checked sample; rung 4 = DEV/TEST target slice + worst-example fp16 gate before TRAIN-2000; P3a before J3's 30k run; anchor+parity checks before J4/J5 count.

## 8. Parity audit — as v2 (recurrence, inner-loop structure, CFG, pins, timestep ≡ `temp_ts`, dtypes, full-tensor optimization loss with future-frame reporting), plus deviations register: empty positive branch; padded-512 context (L ablation, diagnostic); σ₀ = 1.0 vs 0.999; batched execution; optax-vs-torch Adam (same hyperparameters, eps 1e-8, convex-toy verified).

## 9. Launch plan (all pending explicit approval)

| Job | What | Est. |
|---|---|---|
| J0 | Manifest build: full val scan (~3.3 GiB) + bounded train scan (early-stop at 5k episodes) — host-only, no TPU | ~0.5–1 h network |
| J1 | P1: smoke → fit probe → A0/A1/A1-probe/A2/A2-0/A2-probe on DEV-64+TRAINFIT-16, adequacy probe (+ possible one adopted-recipe re-run), L ablation, A3 measurement, full-cohort decode | ~3–6 h v6e-8 |
| J1b | A3 on 8 DEV examples — conditional, separately approved, sized from J1 measurement (≤ 4 h rule) | measured-first |
| J2 | P2 cache (TRAIN-2000 + DEV + TEST) + fidelity gate + structure diagnostic | ~2–5 h |
| J3 | P3a (≤ 2k steps) → clean-restart 30k adapter training | ~1–2 h |
| J4 | TEST-64: adapter / target-replay / null-only | ~1–2 h |
| J5 | pre_context: old-script 4-sample anchor → parity mode → TEST-64 re-eval | ~1–2 h |

Per-job pre-launch package per SOP; `_command.md` entry at launch time; auto-resubmit on infra failure only.

## 10. Risks — as v2, plus: J0 train-shard ordering may not be episode-diverse (mitigation: early-stop threshold at 5k distinct episodes with a hard shard-count cap and a coverage report; if the cap hits first, surface to Yixun before proceeding); legacy-parity tolerance may be unachievable if old-script noise derivation proves non-replicable outside its exact jit structure (mitigation: fingerprint the actual noise tensor from an instrumented dry run on 1 sample — still without editing the old script — and feed it explicitly; documented if used).

## 11. Decision points for Yixun (at plan approval)

1. Cohorts: DEV-64 / TEST-64 / TRAINFIT-16 / TRAIN-2000 (≤2 per episode). OK or resize?
2. `L_null = 16` fixed; in-J1 {L_nat, 16} ablation is diagnostic-only. OK?
3. Approve J0 + J1 now (after P0 + parity audit complete)? J1b–J5 asked separately at their gates.
4. A2's `noise=global` deployment convention acceptable as the fallback deliverable (one canonical noise for all generations)?
5. Pilot scope acknowledged (64-episode test cohort); DROID-wide claims deferred to a follow-up confirmation run.
