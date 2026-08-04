# plan_pos_context — exp_05: Positive text-token inversion + pre_context-structure regression adapter

Planner: Claude Fable 5 (max effort). Status: **v3 — revised per re-review pass 2 (G1: additive-union merge rule for dual-touch files, merge commits reviewed, both dispatch routes tested post-merge); awaiting pass 3, then Yixun approval.**
Branch `claude-exp_05_pos_context-20260804` off `yixun-dev` @ `695d410`. Worktree `/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context`.

v2 changelog vs v1 (findings in `pos_context_codex_plan_review.md`): **direct 8-token conditional-context convention replacing the 512-row embed — targets now match the deployed pre_context representation exactly** (F1); decidable training/overfit/closed-loop rules + pre-K4 DEV closed-loop gate (F2); trainer batching/sharding/cost contract with P3a measurement gates (F3); commit-pinned dependency matrix on exp_04 rounds with two one-way merges (F4); fold-K1-into-J1 option REMOVED, separate jobs default (F5); no mutation of exp_04's settled module — new `pos_context_inversion_wan.py`; rounds split S1–S10 (F6); `z_bar_states` fidelity gate + corrected storage math + TEST-uninspected rule (F7); Q3' reframed achieved-quality (F8); per-token-steering claim removed (F9).

**Contract inheritance** (unchanged): binds by reference to `exp_04_null_adapter_claude/plan_null_adapter.md` v5 for manifests, noise conventions + key derivation, gates module + imputation, artifact integrity, batching, replay verifier, legacy restore + anchor/parity, ladder. Where silent, exp_04 v5 governs; shared-contract changes happen only via exp_04's reviewed plan. **exp_05 never edits exp_04's settled modules** — positive-slot code lives in its own module; runner/evaluator extensions are additive and reviewed in exp_05 rounds on the exp_05 branch only (F6).

References: `third_party/Wan2.2/scripts/embedding_search.py` — `optimize_positive_embeddings` (:681-788), `regenerate_with_positive_embeds` (:822-853), `run_positive_inversion` (:1150-1389), **L_pos forcing (:1181-1195)**; `third_party/Wan2.2/docs/adaptor_design.md` §§3.6, 3.11–3.14; maxdiffusion `wan_pre_context_adapter_forward` (`side_adapter_wan.py:737-774` — **the head's `[B,8,4096]` output is passed directly as the entire context; no 512-row restoration**), `NNXPreContextFeatureContextHead` (:376-421), `_first_block_self_attention_features` (:611-649).

---

## 1. What Yixun asked for — unchanged (Query 1: positive text tokens from inverse DDIM; pre_context structure; regression-loss constraint; parallel with exp_04).

## 2. Questions

- **Q1' (achieved reconstruction bound, positive slot):** per-example optimized oracle for the **8-token conditional context** at w=5, under the deployed representation. Comparable to exp_04 descriptively (same cohorts, gates); raw side-by-side tables only — no per-token-steering claim (F9: token counts and CFG coefficients differ; any such statistic would be non-causal and is dropped).
- **Q2' (basin/transfer):** as v1 (multi-seed probes, matched controls).
- **Q3' (amortization, achieved-quality framing — F8):** does the existing pre_context architecture, trained by teacher-forced regression onto C*_t, beat the historical denoising-trained checkpoint on TEST-64? Architecture is held fixed, but data exposure and optimizer settings differ from the 30k baseline run — this is the requested same-architecture comparison, **not** a controlled loss-only ablation; interpretation language follows.

## 3. Method (deltas from exp_04 §3)

Same latents, σ grid, pin, per-token timestep, bf16/fp32 boundaries, batching-contract structure, noise conventions.

**Context convention (F1 — matches deployment exactly):**
- Conditional context: `C ∈ R^{8×4096}`, **passed directly as the entire `encoder_hidden_states` (sequence length 8)** — exactly what `wan_pre_context_adapter_forward` feeds the transformer. Construction of the warm start mirrors the reference's L_pos forcing (:1181-1195): `C_init = truncate_or_pad(T5(""), 8)` (T5("") has ≥ 8 rows post-padding, so truncation applies; the rule is implemented and tested for both cases).
- Unconditional context: the baseline's normal frozen `T5("")` `[1,512,4096]`, untouched.
- **CFG is active at init** (v_cond at 8 tokens ≠ v_uncond at 512 — different sequence lengths are separate forwards, as in the reference). The v1 branch-equality premise is removed; replaced by two parity tests: **context-construction parity** (our truncate/pad ≡ reference :1181-1195 semantics on fixtures) and **conditional-velocity parity** (the deployed forward invoked with a head emitting constant C equals the replay operator's v_cond with the same C, elementwise — the guarantee that serialized-target replay and the deployed adapter realize the same conditioning).
- **Pivots are NOT shared with exp_04** (F1/F5 consequence): inversion at w=1 uses `C_init` (8 tokens, reference-faithful — the reference inverts with the truncated positive context), so exp_05's trajectories differ from exp_04's 512-token-context trajectories. No B0/A0 or B2-0/A2-0 artifact reuse; K1 computes everything itself (~25 extra forwards/example — negligible).

**Optimization (w=5)** — as v1's branch swap, now on the 8-token context: cache `v_unc = v_θ(z̄_i, σ_i, T5("")[512])`; Adam on C_i through `v_cond = v_θ(z̄_i, σ_i, C_i)`; lock, advance, warm-start. J=10, lr=1e-2 defaults; own adequacy probe (exp_04's statistic/adoption/±2 h rules). **Replay** — per-step C_i cond (8 tokens), frozen T5("") uncond, w=5. **Per-step states z̄_0..z̄_24 recorded** (post-pin) by the positive path only.

## 4. Phases, arms, gates

### Cohorts — reuse exp_04's J0 manifests verbatim (dependency: J0 published; §6 matrix).

### P0' — TDD (no TPU): positive-optimization contract tests (tiny model + convex toy: cached-v_unc C-independence, grads through C only, locked-C advance, warm start, state recording + pin), context-construction parity, conditional-velocity parity, plus per-round tests (§6).

### P1' — Positive reconstruction study + basin probe (job K1, v6e-8; **needs approval**)
Arms on DEV-64 + TRAINFIT-16: **B0** frozen-C CFG replay from traj[0] (C_init in cond slot — the matched control under the active-CFG convention), **B1** optimize from traj[0]/replay from traj[0], **B1-probe** keyed{0,1,2}, **B2** from ε₀=global(0), **B2-0** frozen-C replay from ε₀, **B2-probe** keyed{0,1,2}. Adequacy probe (8 DEV examples). **L_pos ablation {1, 8}** — diagnostic-only, same arm and adopted recipe as the main run (F1 residue pinned), L_pos=8 fixed for K2/K3. Gates **H1/H2** and the target-selection rule ≡ exp_04's G1/G2 forms verbatim (thresholds, imputation, k-sets). Metrics identical.

### P2' — Target caching (job K2, gated; **needs approval**)
Chosen arm over TRAIN-2000 + DEV-64 + TEST-64. Schema = exp_04's with `pos_embeds [25,8,4096]` and **`z_bar_states [25,48,9,12,20]`** (both `latent_dtype`-governed). **Storage (F7, corrected):** states ≈ 5.93 MiB/record fp16; full record ≈ 8.2 MiB; 2,128 records ≈ **17.1 GiB** (fp32 fallback ≈ 34 GiB) — free-space check against the bucket before the build. **Integrity (F7):** byte-length validation and completion-marker fingerprints explicitly cover `z_bar_states`. **Fidelity gate extension (F7):** on the predeclared 8-DEV subset, serialized-fp16 states must produce **bit-identical bf16 model inputs** to in-memory fp32 states (the model consumes bf16; if bf16(fp16(x)) ≡ bf16(x) holds — expected, fp16→bf16 is value-preserving at bf16 precision for our range — the gate passes trivially and is documented; if not, a pinned block-0 feature tolerance max|Δ| ≤ 1e-2 applies, else fp32). Cross-example structure diagnostic (cosine/PCA of C*) computed on **TRAIN + DEV only; TEST targets are written but uninspected until K4** (F7).

### P3' — Regression training + eval (jobs K3/K4; **needs approval**)
**Model:** the existing `NNXWanSideAdapterStack` pre_context configuration, unchanged (~128M trainable; exact param count recorded at P3a). Head output `[B,8,4096]` **is** the deployed conditioning — no representation shim (F1).
**Objective:** teacher-forced regression as v1 (sample t per example; cached z̄_t + per-token timestep + actions → block-0 features (stop-grad) → head → MSE(Ĉ_t, C*_t) fp32).
**Decidable training contract (F2):** metric = **normalized MSE** (MSE ÷ per-step target variance computed once on the TRAIN cache); eval on DEV targets every 1k steps; **stop rule:** if DEV normalized MSE > 2× its running best for 3 consecutive evals while train MSE is still falling, stop and retain the prior best checkpoint; otherwise run the fixed 30k budget; selection = best DEV normalized MSE. **Pre-K4 closed-loop DEV gate (F2):** before TEST is touched, run one noise-matched closed-loop DEV-64 evaluation of the selected checkpoint vs null-only and serialized-target replay under inherited coverage/imputation rules; proceed to K4 only if the adapter beats null-only on DEV (same +0.05 form). **Any shift mitigation is selected on DEV; TEST is never a tuning set** (predeclared).
**Feasibility/sharding contract (F3):** adapter params + opt state replicated (side-adapter trainer convention), data batch-sharded over the mesh, frozen embedder/block-0 params FSDP-sharded as in the trainer; logical global batch 256 with a **gradient-accumulation fallback preserving logical GBS** (microbatch per device × accumulation steps; never a silent logical-batch change). **P3a additionally records:** compile time, step time, peak HBM, actual trainable param count, projected 30k wall time; continuing requires HBM headroom > 10% and projected wall ≤ 6 h, else stop and surface (hardware/recipe reopened with Yixun).
**P3a' learnability gate:** as v1 (32 examples, ≤2k steps, thresholds, clean restart), now with the F3 measurements.
**Eval (K4):** as v1 (exp_04 evaluator's pre_context mode: regressed checkpoint vs denoising-trained baseline vs null-only vs serialized-target replay; noise-matched; anchor/parity preconditions; **H3** ≡ G3 forms). The serialized-replay-vs-adapter gap is the closed-loop-shift measurement; conditional-velocity parity (§3) guarantees the replay and deployed paths share the conditioning representation.

### P4' — Reports as v1.

## 5. Planned code, per file (F6: no exp_04-module edits)

1. **(N) `src/maxdiffusion/models/wan/pos_context_inversion_wan.py`** — `truncate_or_pad_context`, `optimize_positive_embeddings` (batched; per-example losses `[N,J,B]`; grad-norm traces; state recording), `replay_with_positive` (own API; no `branch=` overload of exp_04's function), B-arm helpers. exp_04's `null_inversion_wan.py` is imported for shared primitives (noise, fingerprint) and **never modified**.
2. **(E) `src/maxdiffusion/run_wan_null_inversion.py`** — additive `embedding_slot: positive` dispatch to the new module; `z_bar_states` schema extension; no change to null-slot behavior (characterization test).
3. **(N) `src/maxdiffusion/configs/base_wan_5b_pos_context_inversion.yml`**.
4. **(N) `src/maxdiffusion/trainers/wan_pos_context_regression_trainer.py`** + **(E) `train_wan.py`** dispatch `POS_CONTEXT_TI2V`.
5. **(E) `src/maxdiffusion/generate_wan_null_adapter.py`** — accept the regressed checkpoint in pre_context mode (metadata-checked) + comparison table row; additive only.
6. **(N) launchers** `run_wan_pos_inversion.sh`, `train_wan_pos_context.sh`.
7. **(N) tests**, one file per round (names in §6): construction/velocity parity, optimization contract, replay/state capture, runner slot + null-characterization, schema/states fidelity, gather/loss, trainer state/checkpoint/stop-rule, dispatch/config, restore, rollout/gates, launchers.

## 6. Coder rounds (F6: single-contract, <200 LOC each) and dependency matrix (F4)

Rounds: **S1** `truncate-pad-parity` (construction parity + velocity-parity test fixtures). **S2** `optimize-positives`. **S3** `replay-state-capture`. **S4** `runner-slot-arms` (+ null characterization). **S5** `schema-states-fidelity`. **S6** `regression-gather-loss`. **S7** `trainer-state-checkpoint` (incl. stop rule + accumulation). **S8** `dispatch-config`. **S9** `evaluator-regressed-restore-rollout`. **S10** `launchers`. Each: red test → green → Codex review (`pos_context_codex_code_<marker>_review.md`) → strengthen → commit.

**Dependency matrix (commit-pinned at execution time, recorded in the worklog):**
| exp_05 item | requires exp_04 | mechanism |
|---|---|---|
| S1–S5 (P0'/K1 code) | R1–R9 committed (shared core + manifests round) | **merge-1**: one-way `exp_04-branch → exp_05-branch` at the R9 boundary SHA |
| K1 launch | J0 manifests published + K1 approval + parity audit | provenance check against J0 manifest hashes |
| S6–S8 (K2/K3 code) | nothing beyond merge-1 | — |
| S9/K4 | R14–R15 committed (evaluator) + J5 anchor/parity artifacts | **merge-2**: same one-way direction at the R15 boundary SHA |

Merge policy (G1-revised): one-way exp_04→exp_05 only, plain `git merge` (no cherry-picks). Conflict resolution by file class: (a) exp_04-settled-core files exp_05 never edits → exp_04's side verbatim; (b) exp_05-owned files → exp_05's side; (c) **enumerated dual-touch files — `train_wan.py`, `run_wan_null_inversion.py`, `generate_wan_null_adapter.py` — → the additive union preserving BOTH experiments' behavior** (both dispatch keys, both slots, both evaluator modes). Any merge commit containing a class-(c) resolution gets its own small focused Codex review (`pos_context_codex_code_merge-<n>_review.md`) before the next round opens. Post-merge acceptance: the full combined suite green, explicitly including exp_04's dispatch-route test (`test_null_adapter_dispatch.py`) AND exp_05's dispatch test — both routes exercised. Both merges logged in `commits_pos_context.md` + worklog with SHAs. S8's ordering is unchanged (after merge-1); merge-2 applies the same class-(c) rule, so it cannot erase S8. If exp_04's rounds slip, exp_05 stalls at the matrix rather than duplicating code.

## 7. Validation ladder / 8. Parity audit — inherited, with §3's positive-slot deltas audited line-by-line against `optimize_positive_embeddings` / `regenerate_with_positive_embeds` / the L_pos forcing; deviations register: empty warm start; **8-token direct context (deployment-matching; reference-faithful via L_pos forcing)**; active CFG at init; no pivot sharing with exp_04.

## 9. Launch plan (all pending explicit approval; v6e-8; **K1 is its own job — the fold option is removed (F5)**)

| Job | What | Est. |
|---|---|---|
| K1 | P1': B-arms + adequacy probe + L_pos ablation on DEV-64+TRAINFIT-16 (reuses J0 manifests; computes own pivots) | ~3–6 h |
| K2 | P2' cache (TRAIN-2000 + DEV + TEST, + states; ≈ 17.1 GiB fp16) | ~2–5 h |
| K3 | P3a' (+F3 measurements) → 30k regression training with the stop rule | measured at P3a; ceiling 6 h |
| K4 | pre-K4 DEV closed-loop gate → TEST-64 eval (regressed / baseline / null-only / target-replay) | ~1–2 h |
| (J5) | shared with exp_04 (anchor + baseline re-eval) — asked once at its gate | — |

## 10. Risks — as v1 (prior-art positive-slot failures; overfit — now with the F2 stop rule; closed-loop shift — now with the pre-K4 DEV gate; merge risk — now with the F4 matrix/policy), plus: 8-token inversion pivots are new territory (no exp_04 cross-check possible — mitigated by B0's matched control and the same gate discipline); storage floor check before K2.

## 11. Decision points for Yixun (at plan approval)

1. **L_pos = 8** (deployment-aligned; {1, 8} ablation diagnostic-only)?
2. Primary training = pure teacher-forced regression; combined regression+denoising arm deferred unless you want it in K3?
3. Approve **K1** conditional on (merge-1 done + P0' green + exp_05 parity audit clean + J0 published)?
4. Same pilot scope as exp_04?
