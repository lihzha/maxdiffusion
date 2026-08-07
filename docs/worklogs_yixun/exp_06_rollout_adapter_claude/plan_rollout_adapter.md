# plan_rollout_adapter — exp_06: Rollout-objective training of the frozen-backbone pre_context adapter

v2 (2026-08-07). Planner: Claude Fable 5. Status: DRAFT — pending plan-review pass 2, then Yixun approval (§11).

v2 changelog vs v1 (all 12 pass-1 findings accepted; trail in `rollout_adapter_codex_plan_review.md`): trainer base corrected to `WanTI2VSideAdapterTrainer` with S7 contributing generalized pure utilities only (F1); exp_03 dependency recast from branch-merge to pinned-SHA blob import + kernel extraction with equivalence tests (F2); **mandatory action-use gate added** (F3); E1–E3 recast as motivation + explicit hypothesis (F4); matched-C0 control required for causal claims, historical checkpoint demoted to benchmark row (F5); paired-delta gate + anchor-reproduction protocol (F6); explicit CFG gradient contract with finite-difference oracle (F7); support/PRNG/horizon policy pinned, k=2 primary (F8); DEV-64 fixed-draw selection estimand replacing S7's cached-target metric, TEST-64 exposure closed (F9); the exp_05-unstall claim WITHDRAWN — evaluator is exp_06-owned under a non-tripwired filename (F10); cost honesty (F11); per-job approval discipline, round splits, issues #11/#12 added to runbook rules (F12).

## 1. What Yixun asked for

The project goal, restated 2026-08-07: **a good action-conditioned world model for robotics WITHOUT fine-tuning the Wan2.2 backbone.** The Planner's synthesis of exp_01–exp_05 recommended rollout-objective adapter training; Yixun: "go ahead for exp_06".

## 2. Motivation and the hypothesis (recast per F4 — evidence motivates; it does not pre-prove)

Recorded evidence (citations in the exp_02/04/05 worklogs): **E1** — per-clip optimized conditioning drives the frozen 5B to 0.92 SSIM (positive slot, own basin; null slot 0.85): the backbone + a small conditioning channel can render DROID futures; the deployed adapter's 0.2946 sits far below that oracle. **E2** — one-step-trained systems are rollout-compounding-bottlenecked (exp_02's law, r=−0.9994) and per-step-greedy per-clip optimization is basin-locked, while through-the-sampler endpoint optimization lifted capacity and transferred at 72% relative retention (J1b/J1c — 8 clips, null slot, full-horizon). **E3 (THE HYPOTHESIS, not a result):** a state-conditioned ~128M adapter trained through short rollout horizons across thousands of clips × fresh basins amortizes what per-clip full-horizon optimization demonstrated. This is a cross-slot (null→positive), cross-horizon (25→k), cross-regime (per-clip→amortized) extrapolation. **A negative exp_06 outcome does not by itself falsify the rollout-loss family** — it bounds this adapter/horizon/budget cell. The architecture is held fixed FOR ISOLATION of the objective variable, not because E1 proves amortizer adequacy.

## 3. Method

**Model:** the existing `NNXWanSideAdapterStack` pre_context configuration, unchanged (~128M trainable; frozen 5B; bf16 activation-cast contract per the exp_05 §8 audit). The objective is the only manipulated variable.

**Trainer base (F1):** `WanTI2VSideAdapterTrainer` — the production pipeline load (axis_rules), TFRecord loader, sharded state (adapter replicated, data batch-sharded, frozen FSDP), adapter-only optimizer, CFG forward, Orbax restore. exp_06's trainer is a sibling built on those seams. From S7, by GENERALIZATION rounds (not verbatim reuse): the stop-rule pure function (re-targeted per §3d), the accumulation plan + equivalence oracle, the recency-resume/immutable-selection checkpoint discipline, `optional_config_value`. S7's regression train step, cached-target DEV eval, and variance table are NOT used (they have no referent here).

**Primary arm R-B: short-horizon differentiable rollout, k=2 (F8).** k=2 is the predeclared primary (exp_03's validated construction: fixed-k unroll, `lax.scan` + remat, endpoint MSE ÷ (σ_hi−σ_lo)²). k=4 is a separate EXPLORATORY arm, runnable only in an M1-measured cell and never the headline. R-A (corrective scheduled sampling) and R-C (mix) are CONDITIONAL arms admitted only by a recorded Yixun decision after exp_03's S1.5/S1.6 verdicts land, with R-A's p_ss schedule pinned at admission.

**§3a — The CFG gradient contract (F7).** Deployment computes v = v_unc(frozen, T5("")) + w·(v_cond(adapter) − v_unc), w=5. In the unrolled loss: (i) the gradient tree contains ADAPTER PARAMETERS ONLY (the frozen split is structural, S7's closure-seam lesson); (ii) BOTH branches' dependence on the current rollout state z_i is differentiated — no `stop_gradient` on z_i entering either branch (the one-step trainer's stop-grad pattern is NOT copied; it would truncate inter-step gradients); (iii) the architecture's internal stop-grad on block-0 features remains (architecture held fixed). Obligations as code: fixed-input full-rollout output parity with the deployed evaluator path (bitwise at fp32), and a TWO-STEP CENTRAL-FINITE-DIFFERENCE oracle proving inter-step adapter gradients including the unconditional branch's state dependence (the R11 lesson: the FD test is the actual proof of differentiation).

**§3b — Noise/support policy (F8), pinned:** one support draw PER BATCH (exp_03's convention); start position σ_hi uniform over grid indices {1..N−k} (terminal σ=0 excluded as an endpoint, never a start); fresh ε per draw; PRNG derivation resume-stable and accumulation-stable: fold_in(seed, global_step) then per-purpose fold (the exp_03 `exp03_aux_key` pattern, merged at T1); C0 and R-B consume THE SAME batch stream (identical data order at matched seeds) so the comparison is paired at the data level.

**§3c — Primary gate (F6), as code:** a NEW `pos_rollout_gates.py`: mean PAIRED per-example ΔSSIM(R-B − C0) on DEV-64 ≥ **+0.05** AND paired-bootstrap (10k, seed 20260804) CI-low **> 0**; coverage/imputation rules inherited (claim-penalizing); manifest-hash-bound. TEST-64 confirmation (same paired form) only after the DEV gate passes. **Anchor protocol:** before any new-arm scoring, (1) reproduce the historical checkpoint's recorded 4-sample validation (0.2946) within a predeclared tolerance — the wiring proof; (2) freeze the historical checkpoint's DEV-64 table under the new protocol — the benchmark row every table carries.

**§3d — Selection/stop metric (F9):** a deterministic FIXED-DRAW DEV-64 rollout-loss estimand — per (example, eval) a pinned (support, ε) pair (the exp_02 D2 instrument pattern), manifest-bound to DEV-64, TEST-64 structurally excluded (the config's eval source points at the DEV manifest, not the val directory; a test pins that no TEST name is ever read during training). S7's stop rule generalized to this metric; the identical rule runs on matched-C0. Checkpoint metadata carries the metric + arm + k.

**§3e — THE ACTION-USE GATE (F3, mandatory).** The project goal is ACTION-conditioned prediction; SSIM alone cannot certify it. For the selected checkpoint AND matched-C0: paired evaluations on identical examples and identical noise under {true actions, deterministically shuffled actions (derangement within batch, pinned seed), zero actions}; diagnostic row: adapter-disabled frozen backbone. **Gate: mean paired ΔSSIM(true − shuffled) CI-low > 0 on DEV-64**, repeated on TEST-64 at confirmation. Reported (not gated): Δ(true − zero), and the C0 row (does one-step training use actions more or less than rollout training — a finding either way).

## 4. Phases and jobs (each job separately approved AT ITS PUSHED SHA per announcement 02 and F12 — plan approval authorizes NO job)

- **P0 — TDD:** rounds §6.
- **P1 — M1 fit probe (v6e-8):** compile/steady-step/peak-HBM (per-device peak_bytes vs capacity; reservation failures counted; eval+checkpoint overhead included in wall projections) over the microbatch × k ∈ {2, 4} ladder. **M1 authorizes only the exact cells it measured.** Headroom rule: steady-state peak ≤ 90% capacity.
- **P2 — M2 learnability probe (v6e-8):** 32 examples, ≤2k steps, R-B k=2 + matched-C0-at-equal-updates side by side. **Numerical continuation rule:** R-B train loss at step 2k ≤ 70% of its step-200 running mean AND 4-example fixed-draw paired ΔSSIM(R-B − C0) > 0 in mean. Fail ⇒ stop and reopen with Yixun.
- **P3 — M3 training (v6e-64):** R-B k=2 + matched-C0 (F5: same init, identical data order/seed stream/GBS/updates/optimizer/eval cadence/selection rule), budget per §11-3, S7-generalized stop rule live.
- **P4 — M4 eval + gates → analysis + report:** anchor protocol → paired primary gate → action-use gate → (conditional) TEST confirmation → HTML report + videos.

**Runbook rules (standing; issues #10–#13 ALL of them, F12):** attempt-scoped roots for every queue-retried job; SHA-bound adoption of the latest COMPLETE published checkpoint into a fresh attempt root on resume; `optional_config_value`/direct-declared reads only (never three-arg getattr on HyperParameters); launchers preserve the caller's xtrace state around secrets; submissions via Yixun where the classifier blocks; every launch gets its `_command.md` entry at launch time with acceptance criteria and the exact SHA.

## 5. Planned code, per file (F2/F10 corrected)

1. **T1 (M) exp_03 imports, pinned NOW:** from exp_03 @ **`2ef9b8a`** (its last APPROVE+GO reviewed commit; coordinated via the tracker with the parallel session — if a newer reviewed SHA exists at T1 execution, the pin may be advanced BY RECORDED DECISION, never silently): blob-import `models/wan/overfit100_sampling.py` (the bitwise-inert extracted sampler) + the RNG/support helpers, with recorded source blob hashes; EXTRACT the R-B loss kernel (and R-A's if admitted) from `wan_ti2v_exp03_trainer.py` into a NEW exp_06-owned module with **fixed-input equivalence tests against the imported source construction** (bitwise where reduction order permits; else principled tolerances per the S6 half-ulp precedent). NO branch merge; NO exp_03 file edited; characterization tests, not artificial red (F12).
2. **(N) `src/maxdiffusion/pos_rollout_losses.py`** — the extracted kernels behind the §3a gradient contract.
3. **(N) `trainers/wan_pos_rollout_trainer.py`** — sibling of the side-adapter trainer (its pipeline/loader/sharding/optimizer seams), S7 utilities imported-generalized.
4. **(N) `configs/base_wan_5b_pos_rollout.yml`** (from the side-adapter YAML; + k/arm/support/gate keys; eval source = DEV manifest per §3d) + **(E) `train_wan.py`** additive arm.
5. **(N) `src/maxdiffusion/eval_wan_pos_rollout.py`** — exp_06-OWNED evaluator (restore→rollout→decode→SSIM→gates incl. action-use, over J0 cohorts, S9's certificate discipline reused by import). **Filename deliberately NOT `generate_wan_null_adapter.py`** — exp_05's tripwire is not touched; the v1 claim that exp_06 un-stalls exp_05 S9 is WITHDRAWN (F10). Zero dual-touch edits to exp_05's settled files.
6. **(N) `src/maxdiffusion/pos_rollout_gates.py`** — §3c/§3e gates as code.
7. **(N) launchers** `train_wan_pos_rollout.sh`, `eval_wan_pos_rollout.sh` (S10a technique; attempt-scoped roots; M1 probe mode env).

## 6. Coder rounds (F12 splits; each <200 exec LOC unless pre-authorized in-round)

**T1** `exp03-imports` (blobs + hashes + characterization). **T2** `loss-kernels` (extraction + equivalence + analytic oracles). **T3a** `cfg-rollout-step` (the §3a contract: parity + FD oracles). **T3b** `trainer-loop` (loop + schedule + generalized stop/selection on the §3d metric). **T4** `dispatch-config`. **T5a** `eval-anchor` (restore + anchor-reproduction + benchmark-row freeze). **T5b** `eval-gates` (paired primary + action-use gates). **T6** `launchers`. **T7** `fit-probe-mode` (M1). Each: red (or characterization where importing) → green → focused Codex review → strengthen → commit.

## 7/8. Validation ladder and parity

Inherited: noise conventions (golden-pinned), σ schedules, decode/SSIM parity, provenance-bound artifacts, fail-closed manifests, gates-as-code discipline. New parity obligations: T2 equivalence vs the pinned exp_03 construction; §3a parity vs the deployed evaluator; the anchor protocol (§3c). Deviations register starts empty and accretes per review.

## 9. Launch plan (indicative; NO job approved by plan approval)

| Job | What | Requested when |
|---|---|---|
| M1 | fit probe (k/microbatch ladder) | after T7 commits, at its SHA, with the full pre-launch package |
| M2 | learnability probe + matched-C0 | after M1's verdict, cells M1 authorized |
| M3 | R-B k=2 + matched-C0, budget per §11-3 | after M2 passes its numerical rule |
| M4 | eval + gates | after M3 checkpoints exist |

## 10. Risks (F11 honesty)

- **Cost/HBM beyond the k=2 reference is UNKNOWN.** Known: exp_03's B at 2.713× (v6e-64, full-FT context); C missed fit at 31.28G/31.25G. The adapter backward still runs VJPs through the 5B to context and rollout state; k=4 may exceed 4×. M1 gates everything; no projection is quoted before it.
- **The amortization hypothesis may fail** (E3 is an extrapolation across slot/horizon/regime) — M2 is the cheap kill switch; a negative is a bounded, publishable answer.
- **exp_03 coupling:** pin-based, no branch merge; if 2ef9b8a's constructions prove insufficient for extraction, T1 stalls at its matrix and reports rather than improvising.
- **The law's slope may not transfer** to the adapter setting — diagnostic frame only.
- **NaN history:** exp_03's resolved C-arm NaN class; the fixed-k unroll + emit-before-raise instrumentation come with T1/T2.

## 11. Decision points for Yixun (at plan approval — none authorize a launch)

1. **Matched-C0 is now REQUIRED for the causal claim (F5)** — accept its cost (≈1× arm) as part of M2/M3, with the historical checkpoint kept as a benchmark row? (Planner: yes; without it exp_06 can only claim achieved-quality, not objective-causality.)
2. **Arm set:** R-B k=2 + matched-C0 only for the pilot (Planner recommendation); k=4 exploratory only if M1's cell fits; R-A/R-C admission deferred to a recorded decision after exp_03's verdicts.
3. **M3 budget class:** pilot 10k steps @ GBS 256 (Planner recommendation; the stop rule + fixed-draw eval make failure visible early) vs baseline-matched 30k.
4. **Optional compute-matched control** (multi-draw one-step at equal forward passes, F5's last clause): include in M3 (+~1 arm cost) or defer to a follow-up if the pilot gate passes? (Planner: defer.)
