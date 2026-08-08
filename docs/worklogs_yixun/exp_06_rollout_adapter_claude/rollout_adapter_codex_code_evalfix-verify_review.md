# rollout_adapter — Codex re-review: evaluator rework verification

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: **REQUEST-REVISION — 4 BLOCKERs + 6 MAJORs**, all accepted. All 15 original pass-2 attacks now REFUSED (verified by direct execution); seven NEW probes succeeded. Item-2 (gs:// storage) PASS; item-6 (plan consumption, receiver-keyed noise) PASS narrowly. **The +1,070-LOC waiver was REJECTED**: "this spans separable device, artifact-schema, orchestration, derangement, and gate surfaces, and the shared battery missed defects in each" — adopted; subsequent rounds are split per surface.

Headline findings: the real anchor phase misses deployment's **bf16 input boundary** (float32-fed rollout ⇒ a completely different noise draw ⇒ the anchor can fail on wiring, not model quality — M4-critical); **nonfinite latent/pixel MSE certifies as reproduced** (`abs(NaN) > tol` is false); `require_anchor` accepts a **re-signed four-field forgery** (no embedded measurement); `load_dev_certificate` re-decides numbers but **not DEV identity**; frozen dataclasses' **nested payload dicts stay mutable after hashing** (digest goes stale silently); grid "validation" checks lengths only (an all-ones grid passes as the deployed 25-step execution); the paired gate does not verify **paired draw identities** across arm/control rows; a **legal-but-different rotation with a recomputed fingerprint** passes as "the" seeded derangement (fingerprint = tamper check ≠ legality check, now needing seed re-derivation); the battery accepts **semantically aliased tables** (true aliased into zero/adapter-disabled); the **benchmark prerequisite is orphaned** (gates never load the row; ordering not self-enforced).

## Reviewer output (verbatim)

278,133
## Findings

1. **BLOCKER — the real anchor phase does not reproduce deployment’s bf16 input boundary.** [eval_wan_pos_rollout.py:1425](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1425), [eval_wan_pos_rollout.py:1454](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1454), [eval_wan_pos_rollout.py:1200](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1200)

   The reader converts latents/actions to float32 and the phase passes them unchanged into `initial_latents`. The historical evaluator explicitly casts all three to `weights_dtype`—bf16 in the pinned configuration—before drawing noise. The native bf16 draw differs completely from a float32 draw, so the real anchor can fail for evaluator wiring rather than model quality.

   Concrete change: make `DeviceBackend` carry the configured evaluation dtype and cast `z_i0`, `z_video`, actions, and context before noise/rollout. Add a phase-level bf16 bitwise parity test, not merely a direct `initial_latents(bf16)` test.

2. **BLOCKER — nonfinite latent or pixel MSE certifies as reproduced.** [eval_wan_pos_rollout.py:789](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:789), [eval_wan_pos_rollout.py:804](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:804), [eval_wan_pos_rollout.py:283](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:283)

   Only SSIM is checked for finiteness. My `latent_mse=NaN` measurement returned `reproduced=True`, because `abs(NaN) > tolerance` is false.

   Concrete change: require every per-sample latent MSE, pixel MSE, and SSIM—and every aggregate/deviation—to be finite before constructing or deciding a `Measurement`.

3. **BLOCKER — `require_anchor` accepts a re-signed four-field forgery.** [eval_wan_pos_rollout.py:1337](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1337)

   This self-consistently published payload unlocked the protocol:

   ```python
   {
       "protocol": ANCHOR_PROTOCOL,
       "reproduced": True,
       "checkpoint": {"run_name": HISTORICAL_RUN, "step": 30000},
       "num_steps": 25,
   }
   ```

   It carries no measurement, sample names, means, tolerance, recorded values, or measurement digest.

   Concrete change: add a strict `load_anchor_certificate` that reconstructs and digest-checks the embedded measurement, recomputes `reproduce_anchor`, requires the exact historical names/order/run/step/grid, and verifies every duplicated field agrees.

4. **BLOCKER — `load_dev_certificate` re-decides numbers but not DEV identity.** [pos_rollout_gates.py:655](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:655), [pos_rollout_gates.py:680](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:680)

   A re-signed passing certificate with `cohort="not-dev"`, an arbitrary manifest digest, `example_count=1`, and arbitrary table hashes passed loading and could unlock TEST. The loader does not require the pinned DEV-64 cohort/digest/count or validate the checkpoint fields.

   Concrete change: enforce an exact schema, pinned DEV-64 identity/digest/count, finite `invalid_fraction`, checkpoint and digest fields, and ideally load the referenced score artifacts and rerun `primary_gate`.

5. **MAJOR — “frozen” measurements and tables remain mutable after hashing.** [eval_wan_pos_rollout.py:724](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:724), [eval_wan_pos_rollout.py:887](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:887)

   I constructed a failing measurement, mutated `measurement.payload["mean_ssim"]` to `0.2946`, and obtained `reproduced=True` while its recorded digest remained stale. `frozen=True` does not freeze nested dictionaries.

   Concrete change: store deeply immutable data or defensively copy it and recompute/verify the digest at every consumer. Apply the same rule to `ScoreTable`.

6. **MAJOR — grid “validation” checks only array lengths.** [eval_wan_pos_rollout.py:624](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:624)

   An all-ones 26-sigma grid with terminal sigma `1.0` and 25 zero timesteps was accepted and labeled a deployed 25-step execution.

   Concrete change: bind execution to the canonical sigma/timestep values or a pinned grid digest, including exact lengths, terminal zero, monotonicity, and timestep correspondence; publish that grid identity downstream.

7. **MAJOR — the primary paired gate does not enforce paired noise.** [pos_rollout_gates.py:571](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:571), [pos_rollout_gates.py:606](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:606)

   The gate passed with mean delta `0.10` when every rollout row carried draw digest `aaaa…` and every matched-C0 row carried `bbbb…`.

   Concrete change: `_agree` must verify each row uses the canonical receiver-keyed draw and that paired arm/control rows have identical draw identities, alongside expected arm/checkpoint identities.

8. **MAJOR — the permutation is legal but not bound to its pinned seed.** [pos_rollout_gates.py:364](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:364)

   Re-signed fixed-point and byte-identical forgeries are correctly refused. However, I replaced the generated permutation with a different legal rotation, recomputed its fingerprint, retained the pinned seed, and `action_use_plan` accepted it. That permits donor selection after observing results.

   Concrete change: recompute the canonical seeded assignment during validation and compare it exactly; also revalidate action digests against the cohort’s records at an artifact-consumption boundary.

9. **MAJOR — the full action-use/C0 battery is checked by mapping keys, not table semantics.** [pos_rollout_gates.py:799](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:799), [pos_rollout_gates.py:811](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:811)

   I aliased the true table into `zero` and `adapter_disabled`, and aliased matched-C0 true into its wrong/zero slots. The report accepted the battery and published the resulting deltas.

   Concrete change: validate every table’s condition, arm, checkpoint, cohort/digest, horizon, draw, action source, and derangement identity before reporting.

10. **MAJOR — the benchmark prerequisite is orphaned.** [eval_wan_pos_rollout.py:1512](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1512), [eval_wan_pos_rollout.py:1634](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1634)

    `run_evaluation` requires only the anchor for gates; `run_gates_phase` never loads the benchmark row, and later tables do not carry its digest despite the documented “benchmark row every table carries” contract. Thus the evaluator does not itself enforce `anchor → benchmark → gates → confirm`.

    Concrete change: require and semantically load the benchmark before gates, bind its digest into later tables/reports, and require the issued DEV certificate before confirm using the configured artifact paths.

## Verdicts 1–10

| Item | Verdict | Result |
|---|---|---|
| 1 | **FAIL** | Dispatch exists and there is no `NotImplementedError`, but the bf16 production boundary is missing and full phase ordering is not enforced. |
| 2 | **PASS** | No production `pathlib` appears in either scoped module. Remote URIs route through storage, which refuses local fallback. |
| 3 | **FAIL** | All three original foreign-name/checkpoint/short-horizon attacks are refused, but mutable measurements and arbitrary same-length grids break the claimed derivation chain. |
| 4 | **PARTIAL** | Exact names/order/run/step are correctly enforced by `reproduce_anchor` and wiring-only language is clear. Loaded anchor certificates do not preserve that enforcement. |
| 5 | **FAIL** | Cohort/digests/fingerprint and independent legality checks exist, but the seeded permutation itself is not re-derived. |
| 6 | **PASS narrowly** | `score_condition_table` consumes the plan, keys wrong-action noise on the receiver, and the original donor-keyed attack is refused. |
| 7 | **FAIL** | Marker-only forgery, KeyError coverage, and missing C0-zero are fixed. A richer foreign DEV certificate and semantically aliased battery still pass. |
| 8 | **FAIL as a launch precondition** | It has the structure I requested, and the stub-device second half does not undermine the orchestration test. The real half’s strongest oracle is only “does not reproduce,” which also passes when dtype/grid wiring is wrong; it never executes `load_device_backend`. |
| 9 | **Mixed** | Do not defer `load_device_backend` now—the dtype defect is already concrete. Reject mutable payloads as a harmless residual. Accept only the absolute Python `object.__new__` bound, not public constructors as provenance. Derangement read cost may defer to M4 instrumentation. The three missing launch values fail closed with naming messages today. |
| 10 | **FAIL** | Four blocker-class and six major defects remain. The +1,070 LOC waiver is not sustained: this spans separable device, artifact-schema, orchestration, derangement, and gate surfaces, and the shared battery missed defects in each. |

## Execution

The exact harness and pytest commands were attempted. This sandbox cannot create temporary files under `/tmp`, so the harness stopped after three attacks and pytest produced 50 passes before temp/TensorFlow setup fallout; those are environmental failures, not product failures.

I separately executed all 15 original pass-2 evaluator attacks directly: all were refused. Seven additional probes above succeeded. I accept the Planner’s reported serial `1960 passed, 0 failed` as the clean-suite result, but it does not cover these paths.

**REQUEST-REVISION.**
