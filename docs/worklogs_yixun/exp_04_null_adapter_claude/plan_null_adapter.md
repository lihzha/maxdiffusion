# plan_null_adapter — exp_04: Null-text inversion port + action-conditioned null-embedding adapter

Planner: Claude Fable 5 (max effort). Status: **v2 — revised per Codex plan review of v1 (17/17 findings accepted); awaiting re-review, then Yixun approval.**
Branch `claude-exp_04_null_adapter-20260803` off `yixun-dev` @ `744094a`. Worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter`.

v2 changelog vs v1 (finding numbers from `null_adapter_codex_plan_review.md`): honest renaming of bounds/faithfulness claims (F1–F3); episode-stratified dev/test cohort manifests, leakage removed (F4); A2-0 control + gate G2 (F5); statistically specified gates (F6); self-contained, provenance-bound cache with integrity + fp16-fidelity gates (F7–F8); fully specified adapter/trainer + P3a learnability gate (F9); noise-matched, anchor-preserving baseline protocol via a new evaluator, `generate_wan_side_adapter.py` untouched (F10, F15); A3 moved to a separately approved conditional job sized from an in-J1 measurement (F11); explicit batching/sharding contract (F12); redesigned P0 tests (F13); independent replay verifier ported from `verify_reconstruction_from_null.py` (F14); unpinned-only Gaussianity stats (F16); re-split Coder rounds (F17).

References: `inverse_DDIM_pdf.md` (Mokady et al. 2022); `third_party/Wan2.2/scripts/embedding_search.py`, `verify_reconstruction_from_null.py`, `embedding_search_smoke.py` (submodule pin `f370228`).

---

## 1. Background and motivation

Unchanged from v1 in substance: the deployed `pre_context` adapter reaches mean rollout SSIM 0.2946 (step 30k, 4-sample validation); exp_02/exp_03 quantify and attack the one-step-loss↔rollout gap. The PyTorch fork ran per-step **positive**-embedding inversion on DROID (underfit; `sample ≈ null`) and established the **noise-basin problem**: per-step embeddings optimized around an inverted `z_init` reconstruct from that `z_init` (latent MSE ≈ 0.015–0.022) but fail from fresh noise (≈ 2.0–3.3) — `Wan2.2/docs/adaptor_design.md` §3.11–3.14. Null-text inversion proper was never run on DROID; no network anywhere predicts null embeddings; maxdiffusion has no inversion code.

Structural fit: maxdiffusion's side-adapter path uses T5("") as **both** CFG branches (`wan_ti2v_side_adapter_trainer.py:313-325`, w=5.0). Optimizing the unconditional branch's embedding per step drops in cleanly, and deployment needs **no modules inside the DiT**: v_cond = frozen backbone + T5(""), v_uncond = frozen backbone + predicted ∅̂_t.

## 2. Questions and hypotheses

- **Q1 (achieved reconstruction bound):** How closely can the 25-step Euler CFG rollout reconstruct DROID val clips when a per-example optimizer chooses ∅_t? This is a **per-example optimized oracle under a declared recipe** (L_null, inner iters, LR, bf16 forwards) — *not* a capacity upper bound; any negative result is scoped to the tested recipe (F1), and G1 failure may only be interpreted after the optimization-adequacy probe (§4-P1).
- **Q2 (basin/transfer):** Do optimized ∅_t transfer to fresh Gaussian noise, unlike the positive embeddings? Measured (A1-probe) before any adapter spend; A2 (fixed shared noise) with its matched control A2-0 is the designed fallback.
- **Q3 (amortization):** Can a small action-conditioned predictor of ∅_t beat the pre_context baseline on a **locked, disjoint test cohort** under **noise-matched** evaluation? Framed as an achieved-quality comparison between differently-constructed methods, not a controlled causal attribution (F10).

## 3. Method (exact math, maxdiffusion conventions)

Unchanged from v1: normalized VAE latents `z ∈ [B,48,9,12,20]` fp32; `apply_first_frame_pin` (`side_adapter_wan.py:528`); σ grid `build_rollout_sigmas(25, 5.0, 0.0, 1.0)` (`:814`); per-token timestep `_build_per_token_timestep(σ_i·1000, 9,12,20, n_hist=1)` (`:518`); velocity = frozen transformer, bf16 forward, fp32 latent arithmetic.

Null parameterization: `∅ ∈ R^{L_null×4096}`; `embed(∅)` = T5("") `[512,4096]` with rows `[0:L_null]` replaced. Init `∅ ← T5("")[0:L_null]`. **P0 test (F2):** `bf16(embed(∅_init)) == bf16(T5(""))` bitwise, i.e. branch equality holds after the exact cast the model consumes, so CFG is provably inert at init.

1. **Inversion (w=1):** `traj[25] = pin(z_video)`; for `i = 24..0`: `traj[i] = pin(traj[i+1] + (σ_i − σ_{i+1})·v_θ(traj[i+1], σ_i, C))`, C = T5("").
2. **Per-step null optimization (w=5):** `z̄_0 = z_start`; for `i = 0..24`: cache `v_cond = v_θ(z̄_i, σ_i, C)`; fresh Adam(lr) on ∅_i for J inner iters minimizing `‖pin(z̄_i + (σ_{i+1}−σ_i)·[v_unc + w(v_cond − v_unc)]) − traj[i+1]‖²`, `v_unc = v_θ(z̄_i, σ_i, embed(∅_i))`; lock ∅_i; advance z̄ with one more forward **using the locked ∅_i** (tested, F13); warm-start ∅_{i+1} ← ∅_i. Defaults J=10, lr=1e-2.
3. **Replay:** from any `z_start`: `z_{i+1} = pin(z_i + (σ_{i+1}−σ_i)·[v_unc(∅_i) + w(v_cond − v_unc(∅_i))])`.

**Naming (F2):** the A1 arm is **"empty-positive null-branch inversion (Mokady-style)"** — the recurrence, cached-v_cond inner loop, warm start, and CFG algebra follow Mokady/`embedding_search.py`, but: (a) the reference requires a non-empty source caption; here the conditional branch is T5("") because the dataset has no text — the caption's role collapses onto the null init; (b) the reference optimizes natural-length T5(""); we optimize L_null rows inside the 512-row padded context; (c) σ grid deviation (ours starts at 1.0, PyTorch at 0.999). All three are documented deviations in §8; (b) is additionally probed by the L_null ablation (§4-P1).

**Batching contract (F12):** one batched transformer call per velocity evaluation — nulls `[B,L_null,4096]` embedded into contexts `[B,512,4096]`; loss = Σ_b mean_elements(example b) so per-example gradients are independent; Adam moments live on the batched `[B,L_null,4096]` tensor (elementwise ⇒ per-example independence); per-example losses recorded `[N,J,B]`. **No `vmap(grad)` over the 5B model.** Transformer stays FSDP-sharded exactly as the trainer shards it; ∅/latents batch-replicated arrays (batch ≤ 64, tiny); assert no unexpected large replicated leaves; donate only evolving latent/opt buffers, never frozen model state or arm-shared tensors. P0 tests: B=1 vs B=2 cross-talk (example 0's outputs identical within tolerance when example 1 changes), layout `[N,B]↔[B,N]`.

## 4. Phases, arms, and gates

### Cohort manifests (before P1; F4)
Host-side builder scans the val + train TFRecord shards (deterministic order, reads `name`/`ordinal`), parses episode ids, and emits **immutable episode-stratified manifests** (JSON, committed to the exp folder + mirrored to GCS): **DEV-64** (val; one window per episode, episodes ordered by sha256(episode_id), first 64), **TEST-64** (val; next 64 episodes — disjoint from DEV by episode), **TRAINFIT-16** (train; same rule). P1/P2 arm & hyperparameter selection uses DEV-64 (+TRAINFIT-16) only; G3 runs on TEST-64, whose oracle targets may be precomputed but are not inspected before P3 evaluation. Scope statement: cohort results are a **pilot**; any DROID-wide claim requires a predeclared larger confirmation run (separate approval).

### P0 — TDD infrastructure (no TPU)
Core module + tests; test design per F13: recurrence indices/signs pinned with a **constant / analytically tractable velocity oracle** and an elementwise comparison of the `lax.scan` implementation against a literal Python loop; optimization tested on a convex toy (final loss < initial; no monotonicity assertion); tiny-random-WanModel tests cover only what it can guarantee — nonzero ∅ gradient through the frozen transformer (smoke port), the 2-D per-token-timestep route, and shape/pin invariants. Additional tests listed in §5.

### P1 — Reconstruction study + basin probe (job J1, v6e-8; **needs approval**)
On DEV-64 + TRAINFIT-16:

| Arm | z̄_0 / replay start | What it answers |
|---|---|---|
| A0 | traj[0] / traj[0], frozen ∅ (≡ w=1 replay) | matched control for A1 |
| A1 | traj[0] / traj[0], optimized ∅_t | Q1 headline (per-example oracle) |
| A1-probe | — / fresh noise, seeds k=0,1 keyed `(2026, sha256(name), k)` | Q2 basin |
| A2 | ε₀ / ε₀ (one shared seed-0 noise), optimized ∅_t tracking the same pivot | deployment-consistent fallback |
| A2-0 | — / ε₀, frozen ∅ | **matched control for A2 (F5)** |
| A2-probe | — / fresh per-example noise (same keying, k=2) | A2 generalization beyond ε₀ |

Plus, inside J1: **(i) optimization-adequacy probe (F1)** on 8 DEV examples — J ∈ {10,25,50} × lr ∈ {1e-2,3e-2}; G1 failure may only be called "recipe-limited vs reconstruction-limited" after this probe (plateau rule: <10% median tracking-loss reduction from J=25→50). **(ii) L_null ablation (F2)** on the same 8 examples — L_null ∈ {L_nat(T5("")), 16}. **(iii) A3 feasibility measurement (F11)**: compile + execute exactly one A3 optimizer update (one example): record compile time, step time, peak HBM — sizes the conditional J1b; hard budget stop.

**Metrics (F3, F16):** primary = per-example **future-frame (non-pinned) latent MSE** vs `z_video`; secondary = full-tensor latent MSE, per-step tracking curves `[N,J,B]`, and **decoded-latent GT reconstruction** pixel metrics — SSIM/MSE of decode(z) vs decode(z_video), whose ceiling is 1.0 by construction (the PyTorch 0.87/0.96 numbers compared against raw RGB and are **not** comparable; not used to calibrate gates). Future-frame-only SSIM primary among pixel metrics; full-frame secondary. **The entire gate cohort is decoded** (VAE decode is cheap relative to inversion; F5). `z_init` Gaussianity reported on **unpinned elements** primary, full-tensor secondary (F16).

**Gate G1 (A1, DEV-64, predeclared; F6):** pass iff (a) paired per-example ratio future-MSE(A0)/future-MSE(A1): median ≥ 5 and ≥ 80% of examples improved; (b) mean future-frame SSIM(A1) ≥ 0.80 with 95% paired-bootstrap (10k resamples over examples) CI lower bound ≥ 0.75. Nonfinite/failed example ⇒ counted as not-improved and assigned worst rank (fail-closed).
**Gate G2 (A2 vs A2-0, same ε₀, paired):** median ratio ≥ 5, ≥ 80% improved, mean future-SSIM(A2) ≥ 0.75. A2-probe reported (not gated).
**Basin rule (P3 target choice):** A1-probe passes iff mean future-SSIM(A1-probe) ≥ 0.7 × mean future-SSIM(A1) (paired per-example ratios also reported). Target choice: A1-probe pass → A1 targets, fresh-noise deployment; else G2 pass → A2 targets, fixed-ε₀ deployment convention (fresh-noise generalization still reported); else → stop after P1, report Q1/Q2 with the adequacy-probe scoping, bring A3/J1b decision to Yixun.

### P1b — A3 direct joint optimization (job J1b, **separately approved, conditional**; F11)
Joint opt of all `[25,L_null,4096]` through the remat+scan differentiable 25-step rollout from ε₀, endpoint future-frame MSE, ~300 Adam iters, 8 DEV examples — sized from J1's measured single-update numbers (exp_03's remat/scan precedent covered a 2-step unroll, **not** a 25-step one; treated as unproven until measured). Role: deployment-objective **achieved** bound informing the "what would it take" conversation.

### P2 — Target caching (job J2, gated on G1/G2 outcome; **needs approval**)
Chosen arm over **2,000 train windows (≤2 per episode) + DEV-64 + TEST-64** targets. **Self-contained records (F7):** `name, ordinal, split, z_i0 (fp16), actions (fp32), z_video (fp16), nulls [25,L_null,4096], z_start used, arm + noise convention, final future-MSE, per-step final losses`. Shard-level **provenance header**: source-manifest hash, code SHA, model revision (HF snapshot hash), σ vector, w, base-context fingerprint (sha256 of T5("") bytes), optimization config, dtype policy. **Integrity (F8):** shards written to a staging prefix then published with a completion marker containing count + sha256 + config fingerprint; resume skips only shards whose marker validates (count, checksum, schema, fingerprint); global checks: unique names, exact manifest coverage. **fp16 fidelity gate (F8):** before the train build, for 8 examples replay serialized-fp16 nulls vs in-memory fp32; require SSIM degradation ≤ 0.01 and future-MSE increase ≤ 5%, else store fp32. **Structure diagnostic (F9-adjacent):** cross-example cosine/PCA of ∅* (per step and pooled) reported before P3 trains — the regression-target-consistency check.

### P3 — Adapter training + eval (jobs J3/J4/J5; **needs approval**)
**Model (F9, fully specified):** `NNXNullEmbedAdapter` — inputs `z_i0 [B,48,1,12,20]`, `actions [B,32,7]`. Action path: delta encoding (`a − a₀`), `Linear(7→512)` + learned temporal pos → 32 tokens. Image path: `z_i0` → `[240,48]` spatial tokens (12×20), `Linear(48→512)` + 2-D sin-cos pos. Memory = concat (272 tokens). Queries: learned `[25,16,512]` token embeddings + sinusoidal step embedding (128-d, projected 512, added). Two pre-norm cross-attn+FFN blocks (d=512, 8 heads, FFN 2048, LayerNorm, dropout 0.0). Final LayerNorm; head `Linear(512→4096)` **zero-init**; output `∅̂_t = T5("")[0:16] + Δ̂_t`. ≈ 13M params, fp32. Zero-init note (F9): at step 0 upstream layers receive zero gradient through the head; the head's own weights receive nonzero gradient, so upstream flow begins at step 2 — standard additive zero-init (ControlNet-style), pinned by a two-step-training test.
**Trainer (F9):** reads self-contained P2 records (no cross-dataset join; F7); optax.adamw lr 1e-4, wd 0.01, betas (0.9, 0.95), 1k warmup, cosine to 0.1×, global batch 256, **fixed 30k-step budget**; eval on DEV-64 targets every 1k steps; checkpoint selection = best DEV embedding MSE (declared before J3); Orbax `params/opt_state/step` + `metadata.json` {model_type NULL_EMBED_TI2V, arch config, cache-manifest hash, base-model revision, noise convention, code SHA}.
**P3a learnability gate (F9, inside J3 before the full run):** overfit 32 real cached examples to embedding MSE ≤ 10% of the ∅*-variance baseline AND rollout of the fitted ∅̂ within ΔSSIM ≤ 0.02 of the serialized-oracle rollout on those examples. Fails ⇒ stop, report.
**Eval (J4 + J5, noise-matched; F10):** new evaluator `generate_wan_null_adapter.py` (existing `generate_wan_side_adapter.py` **untouched**; F15). Noise keyed by `(eval_seed=2026, sha256(name), k)` — method-independent; identical `z_start` fed to every method per example (under the A2 convention, all methods get ε₀). Methods on TEST-64: adapter ∅̂, oracle ∅* (upper), null-only (∅̂ ≡ T5("")), and **pre_context@30k via the new evaluator's pre_context mode** (restores the side-adapter checkpoint, reuses `wan_action_adapter_forward`). **Anchor preservation (F10):** before J4/J5 numbers count, (a) re-run the original 4-sample validation with the *unchanged* old script at the current commit and match the stored step-30000 summary within tolerance (|ΔSSIM| ≤ 0.01 per sample), (b) cross-check new-evaluator pre_context vs old-evaluator on those 4 samples (same restored params; noise differs by construction — compare distributionally, documented). Checkpoint restore is metadata-checked, wrong-model/config rejected (F15).
**Gate G3 (TEST-64, predeclared; F6, F10):** adapter beats null-only by mean future-SSIM ≥ +0.05 (paired, CI excluding 0); success claim vs pre_context requires paired same-noise mean future-SSIM difference ≥ +0.02 with 95% bootstrap CI excluding 0 AND ≥ 60% of examples improved. Fail-closed nonfinite handling as G1. Achieved-quality framing.

### P4 — Results, analysis, HTML reports (no TPU)
As v1: `_results.md`, `_analysis.md`, `null_adapter_01-capacity_results.html`, `null_adapter_02-adapter_results.html` + `_results_assets/`.

## 5. Planned code, per file

1. **(N) `src/maxdiffusion/models/wan/null_inversion_wan.py`** — `embed_null_tokens`, `base_context_fingerprint`, `invert_trajectory` (scan, fp32, pin each step), `optimize_null_embeddings` (scan over steps; inner `fori`; fresh per-step Adam; batched per §3; returns nulls `[N,B,L,4096]`, z̄ traj, losses `[N,J,B]`), `replay_with_nulls`.
2. **(N) `src/maxdiffusion/models/wan/null_direct_opt_wan.py`** — A3 (`remat`+`scan` rollout, joint opt) + single-update measurement helper.
3. **(N) `src/maxdiffusion/data_preprocessing/build_null_adapter_manifests.py`** — host-only cohort-manifest builder (§4).
4. **(N) `src/maxdiffusion/run_wan_null_inversion.py`** — driver; modes `capacity | cache | verify_replay | adequacy_probe`; artifact schema + staging/marker publish + validated resume (§4-P2); decode + metrics + videos; manifest-driven.
5. **(N) `src/maxdiffusion/configs/base_wan_5b_null_inversion.yml`** — side-adapter config + keys: `null_mode`, `null_L` (16), `null_inner_iters` (10), `null_lr` (1e-2), `null_guide_scale` (5.0), `inversion_guide_scale` (1.0), manifest paths, arm toggles, `null_artifact_dir`, eval seed.
6. **(N) `src/maxdiffusion/models/wan/null_embed_adapter.py`** — `NNXNullEmbedAdapter` per §4-P3.
7. **(N) `src/maxdiffusion/trainers/wan_null_embed_adapter_trainer.py`** — regression trainer per §4-P3 (checkpoint metadata incl. cache hash).
8. **(E) `src/maxdiffusion/train_wan.py`** — dispatch `NULL_EMBED_TI2V`.
9. **(N) `src/maxdiffusion/generate_wan_null_adapter.py`** — evaluator: metadata-checked restore; modes `adapter | oracle | null_only | pre_context`; name-keyed noise; reuses decode/metric/video helpers by import.
10. **(N) `bash_scripts/run_wan_null_inversion.sh`, `train_wan_null_adapter.sh`, `validate_wan_null_adapter.sh`** — per repo conventions (env config, HF prefetch, teed logs).
11. **(N) tests** (`src/maxdiffusion/tests/worklogs_yixun/`, named per round §6): `test_null_adapter_sigma_grid.py`, `test_null_adapter_embed_tokens.py` (incl. bf16 branch equality), `test_null_adapter_invert_replay_oracle.py` (constant/analytic velocity: indices, signs, scan≡python-loop elementwise, pin points, A0≡w=1 identity, guide-scale/σ validation), `test_null_adapter_optimize_nulls.py` (convex toy final<initial; locked-∅ advance; warm start; `[N,J,B]` shapes; B-independence; tiny-Wan grad + 2-D timestep route), `test_null_adapter_artifact_io.py` (schema, staging/marker, resume validation, provenance fail-closed), `test_null_adapter_verify_replay.py` (verifier never reads z_video/traj; equivalence; mismatch rejection), `test_null_adapter_manifests.py` (episode stratification, DEV/TEST disjointness, immutability hash), `test_null_adapter_a3.py` (grad reaches all steps' nulls; pin; budget stop), `test_null_adapter_adapter_module.py` (identity at init; shapes; delta invariance; two-step upstream-gradient test), `test_null_adapter_trainer_step.py` (loss decreases on synthetic; only adapter params in opt state; save/restore equivalence; wrong-metadata rejection), `test_null_adapter_evaluator.py` (noise keying determinism; mode dispatch; pre_context restore path; wrong-checkpoint rejection).

The **replay verifier (F14)**: `run_wan_null_inversion.py --null_mode=verify_replay` consumes only a published artifact record (+ nothing else): reconstructs `embed(∅_t)` from stored nulls + fingerprint-checked base context, replays from the stored `z_start`, and asserts the final latent matches the stored expected final latent (fp16 tolerance) — never touching `z_video` or the inversion trajectory; any provenance mismatch (σ vector, w, model revision, fingerprint) is a hard error.

## 6. Coder rounds (closed write→review→strengthen cycles, <200 LOC each; F17)

R1 `sigma-embed-replay` (files 1-partial, tests: sigma_grid, embed_tokens, replay parts of invert_replay_oracle). R2 `invert-trajectory`. R3 `optimize-nulls`. R4 `artifact-verifier` (schema + verify_replay + provenance). R5 `runner-capacity` (arms A0–A2 wiring + metrics/decode). R6 `runner-cache-resume` (staging/markers/validated resume + fidelity-gate mode). R7 `manifests-launchers` (file 3 + bash + config). R8 `a3-direct-opt` (before J1 so the measurement helper ships in J1; J1b itself stays conditional). R9 `adapter-module`. R10 `trainer-dispatch`. R11 `evaluator`. R9–R11 start only after the P1 gate outcome. Each round: Opus-Coder test-first → briefed Codex review → strengthening record → commit.

## 7. Validation ladder mapping

1. Static + pytest per round. 2. Tiny-model/CPU synthetic (P0). 3. Real-data readback: 4 val records host-side (shapes/dtypes/stats), T5("") structure + fingerprint, manifest build on real shard listing. 4. Bounded build: P2 DEV/TEST target slice + fp16 fidelity gate before the 2k train build. 5. Smoke inside J1: 2 examples × reduced grid (N=4, J=2), arms A0/A1, one full artifact publish + verify_replay pass. 6. Fit probe inside J1: max batch for the null-opt jit. 7. Full arms. P3: P3a learnability gate before the 30k run.

## 8. Parity audit (before J1 launch, recorded in worklog)

As v1 (recurrence indices/signs vs `embedding_search.py:522-678`; fresh-Adam-per-step, cached v_cond, warm start; CFG formula; pin points incl. candidate steps; timestep ≡ `temp_ts`; dtype boundaries; loss = full-tensor MSE for optimization exactly as reference — pinned frame inert — with future-frame MSE as the *reporting* primary), plus the **documented deviations register (F2, F3):** empty positive branch (no caption exists); padded-512 context vs natural-length (L_null ablation probes it); σ₀ = 1.0 vs 0.999; batched execution; JAX Adam (optax) vs torch.optim.Adam — same hyperparameters, eps 1e-8, verified in the convex-toy test.

## 9. Launch plan (all pending explicit approval; v6e-8 estimates)

| Job | What | Est. wall |
|---|---|---|
| J1 | P1: smoke → fit probe → arms A0/A1/A1-probe/A2/A2-0/A2-probe on DEV-64+TRAINFIT-16, adequacy probe, L_null ablation, A3 single-update measurement, full-cohort decode | ~3–5 h |
| J1b | A3 on 8 DEV examples — **conditional, separately approved, sized from J1 measurement** | measured-first |
| J2 | P2 cache (2k train + DEV + TEST targets, chosen arm) + fidelity gate + structure diagnostic | ~2–5 h |
| J3 | P3a learnability gate → 30k adapter training | ~1–2 h |
| J4 | TEST-64 eval: adapter / oracle / null-only via new evaluator | ~1–2 h |
| J5 | pre_context@30k: old-script 4-sample anchor replication → new-evaluator TEST-64 re-eval | ~1–2 h |

Pre-launch package per job (SOP): SHA, device/host count, batch, acceptance criteria, `_command.md` entry at launch time. Auto-resubmit on infra failure only.

## 10. Risks

Basin failure (measured first; A2/A3 fallbacks; Q1 standalone value); regression-target inconsistency (P2 structure diagnostic before J3; P3a gate); compile blowup of scanned 25×J optimization (tiny-model first, smoke rung, fallback = outer Python loop over steps with one jitted inner step); HBM (fit probe); A3 cost (measurement-gated separate job); evaluator drift vs the 0.2946 anchor (anchor replication step; old script untouched); GCS/gsutil reauth (issue #6 monitoring rules apply to all artifact polling).

## 11. Decision points for Yixun (at plan approval)

1. Cohorts: DEV-64 / TEST-64 (episode-stratified, one window per episode) + TRAINFIT-16; P2 = 2,000 train (≤2/episode). OK or resize?
2. `L_null = 16` default with in-J1 {L_nat, 16} ablation. OK?
3. Approve J1 now (after P0 + parity audit complete)? J1b–J5 will be asked separately at their gates.
4. A2's fixed-ε₀ deployment convention is acceptable as a fallback deliverable if A1-probe fails? (It changes the deployment story: one canonical noise for all generations.)
5. Pilot scope acknowledged (64-episode test cohort): DROID-wide claims deferred to a follow-up confirmation run.
