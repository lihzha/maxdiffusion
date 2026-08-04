# plan_null_adapter — exp_04: Null-text inversion port + action-conditioned null-embedding adapter

Planner: Claude Fable 5 (max effort). Status: **v1 — awaiting Codex plan review, then Yixun approval.**
Branch `claude-exp_04_null_adapter-20260803` off `yixun-dev` @ `744094a`. Worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter`.

References: `inverse_DDIM_pdf.md` (Mokady et al. 2022, arXiv:2211.09794); `third_party/Wan2.2/scripts/embedding_search.py` (`--mode null_inversion`), `verify_reconstruction_from_null.py`, `embedding_search_smoke.py`. Submodule pin `f370228`.

---

## 1. Background and motivation

The active research trains adapters on a frozen Wan2.2 TI2V 5B to predict DROID video from a first frame + 32×7 actions. Known state:

- The deployed `pre_context` adapter (~128M) reaches only **mean rollout SSIM 0.2946** at step 30k (25-step rollout, v6e-8 validation), despite healthy one-step denoising loss — exp_02 quantified the loss→SSIM gap; exp_03 attacks it via training objectives.
- The PyTorch fork (`third_party/Wan2.2`, lihzha) ran an extensive **positive-context inversion** line on DROID (per-step positive embeddings `C_t`, `L_pos=1`, factorization μ+rank-1, `TrajectoryAdaptor`): it **underfit** (`sample ≈ null`) and its central negative finding is the **noise-basin problem** — per-step embeddings optimized around an inverted `z_init` reconstruct that clip nearly perfectly from its own `z_init` (latent MSE ≈ 0.015–0.022) but **fail from fresh noise** (latent MSE ≈ 2.0–3.3; decodes blow out). See `third_party/Wan2.2/docs/adaptor_design.md` §3.11–3.14, `WORKLOG.md:943-949`.
- **Null-text inversion proper (optimize the unconditional embedding ∅_t, Mokady-faithful) was never run on DROID** and no network anywhere was trained to predict null embeddings. The maxdiffusion side has **no inversion code at all** (verified across all branches).

Structural fit: in maxdiffusion's side-adapter path the text context is **T5("") in both CFG branches** (`wan_ti2v_side_adapter_trainer.py:313-325`); CFG (w=5.0) amplifies an *adapter* delta, not a text delta. Null-text inversion drops into this cleanly: keep the conditional branch at frozen T5(""), optimize the **unconditional** branch's embedding ∅_t per sampling step. Deployment then needs **no modules inside the DiT at all** — an action-conditioned predictor of ∅_t is the whole adapter, and the frozen backbone is called exactly as today.

## 2. Questions and hypotheses

- **Q1 (capacity, the headline "how good can this method do"):** Through the frozen backbone with per-example optimized ∅_t, how closely can the 25-step Euler CFG rollout reconstruct DROID val clips? Bounds ladder per reference: DDIM-inv-only replay (lower) ≤ null-text reconstruction ≤ VAE-space GT.
  *Hypothesis:* reconstruction latent MSE ≪ DDIM-inv-only and pixel SSIM ≥ 0.9 (the positive-inversion analog reached 0.87 SSIM vs a 0.96 VAE ceiling at the same resolution; nulls at w=5 have 4× amplified leverage).
- **Q2 (basin/transfer, the de-risking question):** Do optimized ∅_t transfer to fresh Gaussian noise (the deployment condition), unlike positive embeddings (which did not)?
  *Hypothesis (weak prior):* partial transfer at best; measured before any adapter compute is spent. A fixed-shared-noise variant (arm A2 below) is the planned fallback.
- **Q3 (amortization):** Can a small action-conditioned network predict ∅_t from (z_i0, actions) well enough that its rollout beats the pre_context baseline (0.2946 SSIM) on the same protocol?

## 3. Method (exact math, maxdiffusion conventions)

Latent space: normalized VAE latents, `z ∈ [B,48,9,12,20]` fp32; frame-0 pin `apply_first_frame_pin` (`side_adapter_wan.py:528`). σ grid: `build_rollout_sigmas(25, 5.0, 0.0, 1.0)` (`side_adapter_wan.py:814`) — σ₀=1.0 > … > σ₂₅=0; per-token timestep `_build_per_token_timestep(σ_i·1000, 9,12,20, n_hist=1)` (`side_adapter_wan.py:518`; frame-0 tokens get t=0). Velocity `v_θ(z, σ_i, ctx) = transformer(hidden_states=z, timestep=2D, encoder_hidden_states=ctx, deterministic=True)` — bf16 forward, fp32 latent arithmetic.

Context convention: `C = T5("")` full `[512,4096]` (as computed by `_compute_null_context`). Learnable null: `∅ ∈ R^{L_null×4096}`, embedded as `embed(∅)` = context whose rows `[0:L_null]` are ∅ and rows `[L_null:512]` are the frozen T5("") rows. Init `∅ ← T5("")[0:L_null]`, so at init `embed(∅) ≡ T5("")` exactly ⇒ v_cond ≡ v_uncond ⇒ CFG is inert until ∅ moves (clean signal attribution). Default **L_null = 16** (T5("") is ~zero beyond its first rows; 16 gives headroom while keeping P3's prediction target small).

1. **Inversion (w=1)** — reference `compute_inversion_trajectory` (`embedding_search.py:522-572`): `traj[25] = pin(z_video)`; for `i = 24..0`: `traj[i] = pin(traj[i+1] + (σ_i − σ_{i+1})·v_θ(traj[i+1], σ_i, C))` (DDIM-inversion small-step approximation: v evaluated at traj[i+1] with σ_i). No CFG (w=1).
2. **Per-step null optimization (w=5)** — reference `optimize_null_embeddings` (`embedding_search.py:575-678`), Mokady Alg. 1: `z̄_0 = traj[0]`; for `i = 0..24`: cache `v_cond = v_θ(z̄_i, σ_i, C)`; fresh Adam(lr=1e-2) on ∅_i for J=10 inner iters minimizing `‖pin(z̄_i + (σ_{i+1}−σ_i)·[v_unc + w(v_cond − v_unc)]) − traj[i+1]‖²` where `v_unc = v_θ(z̄_i, σ_i, embed(∅_i))`; then lock ∅_i, advance `z̄_{i+1}` with it, warm-start `∅_{i+1} ← ∅_i`.
3. **Replay** — reference `regenerate_with_null_embeds` (`embedding_search.py:791-819`): from `z_init` (or any noise), `z_{i+1} = pin(z_i + (σ_{i+1}−σ_i)·[v_unc(∅_i) + w(v_cond − v_unc(∅_i))])`.

Deviations from the reference, accepted and documented (parity audit §8): (a) σ grid is maxdiffusion's (`linspace(1.0,0,26)` + shift) vs PyTorch's (`σ_max=0.999`) — no cross-repo artifact exchange, and every baseline in this repo uses ours; (b) positive context is T5("") (there is no caption; the reference's heuristic-caption role collapses onto the null init — structurally identical to Mokady with a degenerate caption); (c) batch dimension B is vmapped/jitted natively.

## 4. Phases, arms, and gates

### P0 — TDD infrastructure (no TPU)
Core module + tests with a **tiny randomly-initialized WanModel** (small dims, same class), porting `embedding_search_smoke.py`'s exit criterion to JAX: nonzero gradient reaches ∅ through the frozen transformer; inversion→replay round-trip consistency on the tiny model.

### P1 — Capacity study + basin probe (one v6e-8 job, ~1–2 h; **needs approval**)
Canonical cohort: **first 64 val windows + first 16 train windows** in deterministic TFRecord order (manifest with `name` fields written as an artifact). Arms per example:

| Arm | What | Cost/example | Answers |
|---|---|---|---|
| A0 | DDIM-inv-only: replay from `traj[0]`, frozen ∅ (≡ w=1 replay since embed(∅_init)=C) | 50 fwd | lower bound |
| A1 | Mokady-faithful: null-opt from `z̄_0 = traj[0]`, replay from `traj[0]` | ~325 fwd+bwd | Q1 headline |
| A1-probe | A1's {∅_t} replayed from fresh noise, seeds {0,1} | 100 fwd | Q2 basin |
| A2 | "fixed-noise nulls": same pivot traj, but `z̄_0 = ε₀` (one shared seed-0 noise); replay from ε₀ | ~325 fwd+bwd | deployment-consistent fallback |
| A2-probe | A2's {∅_t} from a second fresh seed | 50 fwd | does A2 generalize past ε₀ |
| A3 (8 examples only) | joint direct opt of all {∅_t} through the differentiable 25-step rollout from ε₀, endpoint loss `‖z_final − z_video‖²` (remat+scan, exp_03-proven pattern; ~300 Adam iters) | ~15k fwd-equiv | deployment-objective ceiling |

Per-arm metrics: full + non-pinned-frame latent MSE vs `z_video`, per-step tracking-loss curves, `z_init` Gaussianity stats (mean/std; reference expects ≈(0,1)); pixel SSIM/MSE via JAX VAE decode vs decode(`z_video`) + comparison videos for 8 designated examples per arm (reusing `generate_wan_side_adapter.py` decode/metric/video helpers).

**Gate G1 (predeclared):** A1 succeeds if median A1 latent MSE ≤ A0/5 **and** mean SSIM(8-example decode set) ≥ 0.85. **Target-choice rule for P3:** if A1-probe fresh-noise SSIM ≥ 0.7·A1 SSIM → P3 targets = A1 nulls, fresh-noise deployment; elif A2 passes G1-equivalents from ε₀ → P3 targets = A2 nulls, fixed-ε₀ deployment convention (fresh-noise generalization still reported); else → stop, report capacity result only, take A3's answer to Yixun as the "what would it take" datapoint.

### P2 — Target caching (gated on G1; one v6e-8 job; **needs approval**)
Run the chosen arm over **2,000 train windows + 200 val windows** (proposal — Yixun may resize). Artifacts per example to GCS (`gs://v6_east1d/datasets/droid_wan_null_adapter/<split>/`): `name`, `∅_{0..24}` `[25,16,4096]` fp16 (≈3.3 MB), `z_init` fp16 (when arm A1), final recon latent MSE, per-step final losses. Sharded npz/TFRecord + manifest; resumable at shard granularity.

### P3 — Adapter training + eval (v6e-8; **needs approval**)
**Model** (`NNXNullEmbedAdapter`, new; ~10–15M params): inputs `z_i0 [B,48,1,12,20]`, `actions [B,32,7]`. Delta-action encoding (`a − a₀`, per PyTorch finding that raw actions collapse cross-video diversity); action tokens + attention-pooled z_i0 tokens as memory; learned per-(step,token) queries `[25,16,d=512]` + sinusoidal step embedding; 2 pre-norm cross-attn+FFN blocks; head `Linear(512→4096)` **zero-init**, output `∅̂_t = T5("")[0:16] + Δ̂_t` (identity-at-init, repo convention).
**Loss:** MSE(∅̂, ∅*) over `[25,16,4096]` (DiT-free — training is cheap); per-step MSE logged. Optional second stage (only if regression eval disappoints and Yixun approves): fine-tune through the per-step tracking loss with cached pivots.
**Eval (canonical protocol):** 25-step rollout at w=5, v_cond = frozen backbone + T5(""), v_unc = frozen backbone + `embed(∅̂_t)`, from the P1-gated noise convention, on the 64-window val cohort; metrics + videos exactly as `generate_wan_side_adapter.py`. Baselines on the **same cohort**: (i) null-only rollout (∅̂ ≡ T5("") — no action information), (ii) **pre_context step-30k checkpoint re-validated on this cohort** (its published 0.2946 was 4 samples only; fair comparison requires same-cohort re-eval — separate small TPU job, needs approval), (iii) oracle ∅* rollout (P1/P2 numbers, upper bound).
**Gate G3:** adapter beats baseline (i) decisively and is compared honestly against (ii); success claim requires mean SSIM > pre_context's same-cohort number.

### P4 — Results, analysis, HTML report (no TPU)
`_results.md`, `_analysis.md`, then `null_adapter_01-capacity_results.html` (bounds ladder, per-step loss curves, basin-probe table, videos) and `null_adapter_02-adapter_results.html` (adapter vs baselines, galleries). Assets in `_results_assets/` folders, relative paths, opens from disk.

## 5. Planned code, per file

New files (N), edits (E). All Python under `PYTHONPATH=src`; style per repo (black, 119 cols).

1. **(N) `src/maxdiffusion/models/wan/null_inversion_wan.py`** — pure functions, velocity-fn seam (callable `(hidden_states, timestep, encoder_hidden_states) → v`, matching exp_03's pattern):
   - `embed_null_tokens(null_tokens [B,L,4096], base_context [512,4096]) → [B,512,4096]` (rows 0:L replaced).
   - `invert_trajectory(velocity_fn, z_video, z_i0, sigmas) → traj [N+1,B,...]` — `lax.scan` over reversed steps, pin each step, fp32.
   - `optimize_null_embeddings(velocity_fn, traj, z_i0, sigmas, null_init, base_context, inner_iters, lr, guide_scale) → (nulls [N,B,L,4096], z_bar_traj, per_step_losses [N,J])` — `lax.scan` over steps carrying `(z̄, ∅, )`; body: cache v_cond once; `lax.fori_loop`/scan over J inner Adam iters (fresh `optax.adam(lr)` state per step, matching the reference); `jax.grad` w.r.t. ∅ only.
   - `replay_with_nulls(velocity_fn, z_start, z_i0, sigmas, nulls, base_context, guide_scale) → z_final` — scan; also used for A0 by passing `nulls = tile(null_init)`.
   - `direct_optimize_nulls(...)` (A3): joint opt of `[N,L,4096]` through `remat`-wrapped scan rollout, endpoint MSE.
2. **(N) `src/maxdiffusion/run_wan_null_inversion.py`** — P1/P2 driver (pyconfig entrypoint like `generate_wan_side_adapter.py`): load pipeline (frozen transformer, VAE for decode subset, T5 only to compute T5("") then freed); deterministic TFRecord read incl. `name`; jitted batched arms; artifact writer (sharded npz + manifest.json to GCS via `tf.io.gfile`); decode+video subset; resumable (skip completed shards).
3. **(N) `src/maxdiffusion/configs/base_wan_5b_null_inversion.yml`** — copy of side-adapter config + new keys: `null_inversion_mode` (capacity|cache), `null_L` (16), `null_inner_iters` (10), `null_lr` (1e-2), `null_guide_scale` (5.0), `inversion_guide_scale` (1.0), cohort sizes/paths, arm toggles, `null_artifact_dir`.
4. **(N) `src/maxdiffusion/models/wan/null_embed_adapter.py`** — `NNXNullEmbedAdapter` per §4-P3.
5. **(N) `src/maxdiffusion/trainers/wan_null_embed_adapter_trainer.py`** — regression trainer (reads P2 artifacts; optax adamw; Orbax params/opt_state/step exactly like the side-adapter trainer's manager; no DiT in the train step).
6. **(E) `src/maxdiffusion/generate_wan_side_adapter.py`** — add a rollout mode where `v_uncond` uses `embed(∅̂_t)` from a restored `NNXNullEmbedAdapter` (or from cached oracle ∅*), guarded by a config key; no behavior change for existing modes.
7. **(E) `src/maxdiffusion/train_wan.py`** — dispatch `model_type: NULL_EMBED_TI2V → WanNullEmbedAdapterTrainer`.
8. **(N) `bash_scripts/run_wan_null_inversion.sh`**, **(N) `bash_scripts/train_wan_null_adapter.sh`** — following `train_wan_side_adapter.sh` conventions (env-var config, HF prefetch, teed logs).
9. **(N) tests** in `src/maxdiffusion/tests/worklogs_yixun/`: `test_null_adapter_embed_tokens.py`, `test_null_adapter_sigma_grid.py` (grid values vs hardcoded expected incl. the 0.1724 tail; documents the 0.999-vs-1.0 deviation), `test_null_adapter_invert_trajectory.py` (tiny model: forward-Euler of a replay then inversion recovers start within tolerance; pin invariants), `test_null_adapter_optimize_nulls.py` (smoke port: nonzero ∅ grad; per-step loss decreases on tiny model; warm-start plumbing), `test_null_adapter_replay.py` (A0 ≡ w=1 replay identity at ∅_init; CFG algebra), `test_null_adapter_runner_io.py` (artifact round-trip, manifest, resume-skip), `test_null_adapter_adapter.py` (identity at init: ∅̂ ≡ T5("")[0:16]; shapes; delta-action invariance), `test_null_adapter_trainer_step.py` (loss decreases on synthetic targets; only adapter params in opt state).

## 6. Coder rounds (closed write→review→strengthen cycles, <200 LOC each)

R1 `sigma-embed-replay`: items 9(grid/embed/replay parts) + module skeleton with `embed_null_tokens`, `replay_with_nulls`. R2 `invert-trajectory`. R3 `optimize-nulls` (+ smoke-port test). R4 `runner-config-io` (+ bash). R5 `direct-opt-a3`. R6 `adapter-module`. R7 `trainer-dispatch`. R8 `eval-wiring`. Rounds R6–R8 start only after G1. Each round: Opus-Coder test-first → Codex review (briefed per SOP) → strengthening record → commit.

## 7. Validation ladder mapping

1. Static + pytest suite (every round). 2. Tiny-model synthetic forwards (P0 tests, CPU). 3. Real-data readback: parse 4 val records on host, assert shapes/dtypes/stats vs schema; T5("") shape/zero-pad structure check. 4. Bounded build: P2 val-split slice (200) before the train build. 5. Smoke: P1 job first runs a 2-example, 2-step-grid pass end-to-end (arms A0/A1 only, no checkpointing) to produce one completed optimization + artifact write. 6. Fit probe: find max vmapped batch for the null-opt jit on v6e-8. 7. Full runs P1→P2→P3. (5–7 all inside approved jobs.)

## 8. Parity audit (before P1 launch, recorded in worklog)

Component-by-component vs `embedding_search.py`: inversion recurrence indices/signs (`:543-571`); inner-loop structure — fresh Adam per step, v_cond cached, warm start (`:620-676`); CFG formula and w=5; pin applied at exactly the same points (init/each step/each candidate); timestep construction (frame-0 tokens t=0 — ours via `_build_per_token_timestep` ≡ reference `temp_ts` `:488-500`); dtype boundaries (bf16 model fwd, fp32 latents/Adam — reference used fp32 latent arithmetic under autocast); σ-grid numbers (documented deviation); loss = plain MSE over all latent elements (reference `F.mse_loss` includes pinned frame 0, which is loss-inert since both sides are pinned — replicate, and log non-pinned MSE separately for reporting).

## 9. Launch plan (all pending explicit approval; costs are v6e-8 estimates)

| Job | What | Est. wall |
|---|---|---|
| J1 | P1 capacity study (80 examples × arms A0–A2 + A3×8, decode subset) | ~1.5–3 h incl. compile |
| J2 | P2 cache (2,200 examples, chosen arm) | ~2–4 h |
| J3 | P3 adapter training (DiT-free regression, 20–50k steps) | ~1–2 h |
| J4 | P3 rollout eval (adapter + null-only + oracle, 64-val cohort) | ~1–2 h |
| J5 | pre_context step-30k same-cohort re-validation (fair baseline) | ~1–2 h |

Pre-launch package per job (SOP): SHA, device/host count, batch, acceptance criteria ("≥1 completed unit, no OOM/NaN", plus per-job gates above), `_command.md` entry at launch time.

## 10. Risks

- **Basin failure (main risk):** measured at P1 (A1-probe) before any P2/P3 spend; A2/A3 are the designed fallbacks; worst case the experiment still delivers the capacity number (Q1) — valuable on its own.
- **Target inconsistency for regression:** per-example ∅* may be a non-smooth function of (z_i0, actions) (the positive-line seed-sensitivity finding, trajectory cosine 0.515 across seeds). Mitigations: deterministic optimization (no stochasticity in A1/A2 given the fixed ε₀), and P2-time diagnostic: cosine/PCA structure across examples' ∅* (port of `analyze_embeddings.py` intent) reported before P3 trains.
- **Compile-time blowup** of the scanned 25×10 optimization: bounded by tiny-model tests first (rung 2) and the 2-example smoke (rung 5); fallback is outer python loop over steps with one jitted inner step.
- **HBM pressure** from vmapped fwd+bwd: fit probe (rung 6) before the full P1 batch size is fixed.
- **w=5 CFG at ∅≡C is inert at init:** intended (clean attribution) but means A0 lower bound equals w=1 replay; documented so the bounds table is read correctly.

## 11. Decision points for Yixun (at plan approval)

1. Cohort sizes: P1 = 64 val + 16 train; P2 = 2,000 train + 200 val; P3 eval = the 64-val cohort. OK or resize?
2. Include arm A3 (differentiable joint opt, 8 examples) in J1? (Planner recommends yes — it's the deployment-objective ceiling and reuses exp_03's proven remat/scan pattern.)
3. `L_null = 16` default (ablation {1, 64} deferred unless P1 says otherwise). OK?
4. J5 baseline re-validation of pre_context@30k on the new cohort — approve as part of the package?
5. Approve J1 after P0 + parity audit complete (J2–J4 gated on G1/G3 and asked separately)?
