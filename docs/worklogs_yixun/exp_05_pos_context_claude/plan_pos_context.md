# plan_pos_context — exp_05: Positive text-token inversion + pre_context-structure regression adapter

Planner: Claude Fable 5 (max effort). Status: **v1 — awaiting Codex plan review, then Yixun approval.**
Branch `claude-exp_05_pos_context-20260804` off `yixun-dev` @ `695d410`. Worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context`.

**Contract inheritance.** exp_05 is the positive-slot sibling of exp_04. It **binds by reference to `docs/worklogs_yixun/exp_04_null_adapter_claude/plan_null_adapter.md` v5** (APPROVE-PLAN after five review passes) for every shared contract: cohort manifests (J0's DEV-64 / TEST-64 / TRAINFIT-16 / TRAIN-2000 — the identical manifest files), noise conventions and exact key derivation, the gates module and its statistics/imputation semantics, artifact integrity (staging/markers/`latent_dtype`/fidelity gate), the batching/sharding contract, the replay-verifier contract, the legacy pre_context restore contract + anchor/parity protocol, the validation ladder, and the parity-audit style. This plan specifies **only the deltas and exp_05-specific parts**; where it is silent, exp_04 v5 governs. Reviewed changes to shared contracts happen in exp_04's plan, never silently here.

References: `third_party/Wan2.2/scripts/embedding_search.py` — `optimize_positive_embeddings` (:681-788), `regenerate_with_positive_embeds` (:822-853), `run_positive_inversion` (:1150+); prior-art negative results in `third_party/Wan2.2/docs/adaptor_design.md` §§3.11–3.14; maxdiffusion `wan_pre_context_adapter_forward` (`side_adapter_wan.py:737-774`), `NNXPreContextFeatureContextHead` (`side_adapter_wan.py:376-421`).

---

## 1. What Yixun asked for (Query 1)

Get **text tokens from inverse DDIM** — per-step **positive** text embeddings C*_t — and train the **pre_context structure** to predict them, "use loss function to constrain adapter to do that" = a regression loss onto the inversion-derived tokens. Run in parallel with exp_04.

## 2. Questions

- **Q1' (achieved reconstruction bound, positive slot):** per-example optimized oracle for C_t at w=5 — directly comparable to exp_04's null-slot bound **on the same cohorts, same pivots, same gates**. Science bonus: which slot steers more per token (cond weight +w vs uncond −(w−1)).
- **Q2' (basin/transfer):** the PyTorch fork measured catastrophic fresh-noise failure for positive embeddings (latent MSE ≈ 2.7 vs 0.015; `adaptor_design.md` §3.6/`WORKLOG.md:943-949`) — at 480²/other settings. Re-measured here under our exact recipe with matched controls.
- **Q3' (amortization via the existing structure):** does the **existing pre_context architecture**, trained by **teacher-forced regression onto C*_t** instead of one-step denoising, beat its own denoising-trained baseline (0.2946 SSIM checkpoint) on the locked TEST cohort? This isolates "training signal" from "architecture capacity" for the deployed adapter — the head, injection point, and rollout path are byte-identical to the baseline's.

## 3. Method (deltas from exp_04 §3)

Same latents, σ grid, pin, per-token timestep, bf16/fp32 boundaries, batching contract. Same `embed(·)` helper: `embed_pos(C)` = T5("") `[512,4096]` with rows `[0:L_pos]` replaced; **L_pos = 8** (aligned with `pre_context_tokens: 8` so the target lives exactly in the head's output space); warm start `C_init = T5("")[0:8]` ⇒ branch equality at init (same bf16 bitwise test).

1. **Inversion (w=1):** identical to exp_04 — same function, same context T5(""), **identical pivot trajectories** (a cross-experiment determinism check: exp_05's recomputed traj-derived B0 metrics must equal exp_04's A0 metrics on shared examples when both jobs have run; provenance-linked, reported).
2. **Per-step positive optimization (w=5)** — reference `optimize_positive_embeddings`: `z̄_0 = z_start`; for `i = 0..24`: **cache `v_unc = v_θ(z̄_i, σ_i, T5(""))`** (the role swap: uncond is C-independent); fresh Adam(lr) on C_i for J inner iters minimizing `‖pin(z̄_i + (σ_{i+1}−σ_i)·[v_unc + w(v_cond(C_i) − v_unc)]) − traj[i+1]‖²` with `v_cond = v_θ(z̄_i, σ_i, embed_pos(C_i))`, **gradients through v_cond**; lock C_i, advance with it, warm-start C_{i+1} ← C_i. Defaults J=10, lr=1e-2 (own adequacy probe, §4).
3. **Replay** — reference `regenerate_with_positive_embeds`: per-step C_i in the cond slot, frozen T5("") uncond, w=5.
4. **Per-step states are first-class artifacts:** the optimization records `z̄_0..z̄_24` (post-pin, the per-step inputs) — the pre_context head consumes the current latent, so teacher-forced training needs them.

Naming discipline (inherited): the arm is "empty-warm-start positive-context inversion" — the reference warm-starts from T5(caption); no captions exist, so warm start is T5("")[0:8]; deviations register extends exp_04 §8 with this line. PyTorch DROID precedent used L_pos=1; our L_pos=8 is a declared choice probed by a diagnostic-only ablation {1, 8}.

## 4. Phases, arms, gates

### Cohorts — **reuse exp_04's J0 manifests verbatim** (same files, provenance-checked). No new manifest job.

### P0' — TDD (no TPU)
exp_05-specific tests only (shared machinery is tested in exp_04's suite): positive-optimization contract on the tiny model + convex toy (grads reach C, cached-v_unc asymmetry, locked-C advance, warm start), `embed_pos` branch equality, state-recording shapes/pin invariants, and the S-round tests in §6.

### P1' — Positive reconstruction study + basin probe (job K1, v6e-8; **needs approval**)
Arms on DEV-64 + TRAINFIT-16, mirroring exp_04's table with C in place of ∅: **B0** (frozen replay from traj[0] — bitwise the same computation as exp_04's A0; if J1's provenance-verified artifacts exist, reuse instead of recompute, else compute and cross-check later), **B1** (optimize from traj[0], replay from traj[0]), **B1-probe** (keyed{0,1,2}), **B2** (from ε₀=global(0)), **B2-0** (≡ A2-0, same reuse rule), **B2-probe** (keyed{0,1,2}). Own adequacy probe (8 DEV examples, J∈{10,25,50} × lr∈{1e-2,3e-2}, same adoption statistic/rule/±2 h budget as exp_04). **L_pos ablation {1, 8}** on the same 8 examples — diagnostic-only, L_pos=8 fixed for K2/K3.

**Gates (same forms, thresholds, imputation, and module as exp_04):** **H1** (B1 vs B0) ≡ G1's conditions; **H2** (B2 vs B2-0) ≡ G2's; **target-selection rule** ≡ exp_04's (keyed deployment iff H1 + probe relative ≥0.7× + absolute ≥0.70; else global iff H2; else stop and report). Metrics identical (future-frame primary, full-cohort decode).

### P2' — Target caching (job K2, gated; **needs approval**)
Chosen arm over TRAIN-2000 + DEV-64 + TEST-64. **Schema = exp_04's P2 schema with:** `nulls` → `pos_embeds [25,8,4096]`, plus **`z_bar_states [25,48,9,12,20]`** (post-pin per-step inputs; `latent_dtype`-governed alongside z_start/expected_final_latent/pos_embeds; ≈ 2.3 MB fp16 → record ≈ 6 MB, cache ≈ 13 GiB). Same staging/markers/coverage/fidelity gate (fidelity replay uses the serialized pos_embeds; worst-example thresholds identical). Same cross-example structure diagnostic (cosine/PCA of C*_t) before K3.

### P3' — Regression training of the pre_context structure + eval (jobs K3/K4; **needs approval**)
**Model:** the **existing** `NNXWanSideAdapterStack` in `pre_context` configuration, unchanged (`action_adapter_type: pre_context`, `pre_context_tokens: 8` — same class, same init, same freeze split; ~128M trainable). No new architecture: that is the point of Q3'.
**Training objective (teacher-forced regression):** per example, sample `t ~ U{0..24}` (per-example independent); inputs: cached `z̄_t`, per-token timestep for σ_t, actions; forward ONLY the pre-context path — `_patchify_and_time_embed` → `_first_block_self_attention_features` (stop-grad, frozen) → `predict_pre_context(features, actions)` → `Ĉ_t [B,8,4096]`; **loss = MSE(Ĉ_t, C*_t)** (fp32). The frozen transformer contributes block-0 only per step (≈ 1/40 of a forward — cheap; the full-transformer re-run of the deployed forward is NOT executed during training). Optimizer/schedule/budget/checkpoint-selection: exactly exp_04-P3's recipe (adamw 1e-4, wd 0.01, betas (0.9,0.95), 1k warmup, cosine 0.1×, batch 256, fixed 30k steps, best-DEV-embedding-MSE selection, atomic metadata incl. `model_type: POS_CONTEXT_TI2V`, cache hash, noise convention, L_pos). Dispatch key `POS_CONTEXT_TI2V` in `train_wan.py`.
**P3a' learnability gate:** exp_04's rule verbatim (32 examples, ≤2k steps, batch 32, lr 3e-4; embedding-MSE ≤ 10% of target variance AND rollout ΔSSIM ≤ 0.02 vs serialized-target replay; clean restart for the full run).
**Overfit risk (declared):** 128M params vs 50k (state, target) pairs (2,000 examples × 25 steps) — regularization = the recipe's wd + early selection on DEV embedding MSE; if the DEV/train gap explodes in K3, stop and bring a bigger-cache decision to Yixun rather than tuning ad hoc.
**Closed-loop eval (K4):** deployment is the **existing pre_context rollout structure** — v_cond = full transformer re-run with `encoder_hidden_states = predicted context` from the head at each step's own z_t (`wan_pre_context_adapter_forward`), v_uncond = frozen transformer + T5(""), w=5 — via exp_04's evaluator `pre_context` mode pointed at the regressed checkpoint. Noise-matched protocol, k-sets, and anchor/parity preconditions exactly as exp_04-P3. Methods on TEST-64: **regressed pre_context** (ours), **denoising-trained pre_context@30k** (the baseline — same architecture, different training signal), **null-only**, **serialized-target replay** (reference point / achieved oracle per convention). Cross-experiment row (informational, not gated): exp_04's null adapter on the same table.
**Gate H3 (≡ G3 forms):** regressed vs null-only ≥ +0.05 (CI excl. 0); success vs the denoising-trained baseline ≥ +0.02 mean future-SSIM, 95% CI excl. 0, ≥ 60% improved; same imputation.
**Teacher-forced → closed-loop shift (declared risk):** at rollout the head sees its own states, not z̄_t. Mitigations are follow-ups needing approval, not silent additions: (a) input-noise augmentation on z̄_t; (b) a short closed-loop fine-tune; (c) exp_03-style corrective objectives. K4's serialized-target replay vs regressed-adapter gap isolates how much quality the shift costs.

### P4' — Results, analysis, HTML reports — as exp_04, incl. a side-by-side null-vs-positive page (`pos_context_02-slot-comparison_results.html`).

## 5. Planned code, per file (deltas only; shared core comes from exp_04's branch)

1. **(E) `src/maxdiffusion/models/wan/null_inversion_wan.py`** — add `optimize_positive_embeddings(...)` mirroring `optimize_null_embeddings`'s signature/batching with the branch swap (cache v_unc; grads through v_cond) and per-step state recording (also retrofitted to the null path behind a flag, so exp_04 caches states too if ever needed); `replay_with_positive(...)` (or a `branch=` parameter on the existing replay — Coder's choice, reviewed).
2. **(E) `src/maxdiffusion/run_wan_null_inversion.py`** — `embedding_slot: null|positive` config key; B-arm wiring; `z_bar_states` in the schema; A0/A2-0 artifact-reuse path (provenance-verified) + cross-check report.
3. **(N) `src/maxdiffusion/configs/base_wan_5b_pos_context_inversion.yml`** — exp_05 keys (`embedding_slot: positive`, `pos_L: 8`, cache paths).
4. **(N) `src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py`** — teacher-forced regression trainer per §4-P3' (reuses the side-adapter trainer's pipeline/checkpoint/mesh patterns; only adapter params in opt state — pinned by test).
5. **(E) `src/maxdiffusion/train_wan.py`** — `POS_CONTEXT_TI2V` dispatch.
6. **(E) `src/maxdiffusion/generate_wan_null_adapter.py`** — accept the regressed checkpoint in `pre_context` mode (metadata-checked; rejects wrong model_type) + the cross-experiment comparison table.
7. **(N) bash launchers** — `run_wan_pos_inversion.sh`, `train_wan_pos_context.sh`.
8. **(N) tests** (`src/maxdiffusion/tests/worklogs_yixun/`): `test_pos_context_optimize.py` (branch-swap contract: v_unc cached/C-independent, grads reach C only, locked-C advance, warm start, `[N,J,B]` losses, state recording + pin), `test_pos_context_embed.py` (embed_pos + bf16 branch equality at L=8), `test_pos_context_runner.py` (slot dispatch, z_bar_states schema/bytes, reuse-path provenance rejection), `test_pos_context_trainer.py` (teacher-forced step: loss decreases on synthetic targets; frozen-transformer params absent from opt state; t-sampling per-example independence; save/restore + metadata rejection), `test_pos_context_evaluator.py` (regressed-checkpoint acceptance, wrong-type rejection).

## 6. Coder rounds (dependency: start after exp_04's shared core R1–R8 is committed; merge exp_04's branch into this branch at that boundary — recorded in the worklog like exp_03's merge of exp_02)

S1 `optimize-positives` (item 1 + tests). S2 `runner-positive-mode` (items 2–3 + tests). S3 `regression-trainer` (items 4–5 + tests). S4 `evaluator-regressed` (item 6 + tests + launchers). S3–S4 start only after K1's gate outcome. Same closed write→review→strengthen cycles; each < 200 LOC.

## 7. Validation ladder / 8. Parity audit — inherited from exp_04 (§7/§8) with the §3 deltas audited against `optimize_positive_embeddings`/`regenerate_with_positive_embeds` line-by-line; deviations register adds: empty warm start (no captions), L_pos=8 vs reference natural-length/L_pos=1 precedent.

## 9. Launch plan (all pending explicit approval; v6e-8)

| Job | What | Est. |
|---|---|---|
| K1 | P1': B-arms + adequacy probe + L_pos ablation on DEV-64+TRAINFIT-16 (reuses J0 manifests; reuses J1's A0/A2-0 artifacts when available) | ~3–6 h |
| K2 | P2' cache (TRAIN-2000 + DEV + TEST, +states) | ~2–5 h |
| K3 | P3a' gate → 30k regression training | ~1–3 h |
| K4 | TEST-64 eval: regressed / baseline pre_context / null-only / target-replay (+ exp_04 cross-row) | ~1–2 h |
| (J5) | pre_context baseline anchor + re-eval — **shared with exp_04, one job serves both**; asked once at its gate | — |

Option (Yixun decision 2): fold K1 into exp_04's J1 as one TPU submission (both arm-sets share the per-example pivot trajectories — saves the duplicated inversion pass and one queue slot; the runner supports both slots). Default if not chosen: separate jobs.

## 10. Risks

Prior-art negative results for the positive slot (underfit; basin MSE ≈ 2.7 from fresh noise; seed-sensitivity cosine 0.515) — differences here are declared (timestep-aware z_t-conditioned head, L=8, teacher-forced supervised targets, matched controls, fixed-noise fallback) and the basin is measured before any training spend; 128M-vs-50k-pairs overfit (K3 stop rule); teacher-forced/closed-loop shift (measured by the replay-vs-adapter gap; mitigations gated); shared-contract drift (bound to exp_04 v5 — changes only via exp_04's reviewed plan); merge risk at the R8 boundary (single-direction merge exp_04→exp_05, recorded, suite must stay green).

## 11. Decision points for Yixun (at plan approval)

1. **L_pos = 8** (aligned with the pre_context head; {1, 8} ablation diagnostic-only)?
2. **Fold K1 into J1** as one TPU submission (shared pivots, one queue slot), or keep separate jobs?
3. Primary training = **pure teacher-forced regression**; a combined regression+denoising-loss arm is optional/deferred unless you want it in K3 now?
4. Approve **K1** now, conditional on (exp_04 shared core committed + exp_05 parity audit clean + P0' green) — or wait for J1's null-slot results first?
5. Same pilot scope as exp_04 (TEST-64; DROID-wide claims deferred)?
