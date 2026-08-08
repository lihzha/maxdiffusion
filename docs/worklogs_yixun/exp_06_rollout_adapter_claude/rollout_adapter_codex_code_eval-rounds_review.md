# rollout_adapter — Codex code review: T5a `eval-anchor` + T5b `eval-gates` (backlog pass 2 of 3)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **BOTH REQUEST-REVISION — 4 BLOCKERs + 6 MAJORs**, all accepted.

**PASSED:** A4 (composition — no private sampler loop is constructible), A5 (the A13 replacement correctly pins the native-bf16 draw bitwise against the deployed construction and rejects fp32-then-cast), **B1 — the Coder's flagged shape adapter is CORRECT** (`ssim → future_ssim`, `mse → future_mse` at key `"0"`) with exp_04's imported computation and constants properly pinned, B4 narrowly (the three self-found first-draft fixes landed), B5 (a malformed entry becomes missing coverage; missing/nonfinite measurements are claim-penalized, never imputed as perfect SSIM).

**THE GENERALIZATION UNDERNEATH NEARLY EVERY FINDING — "STAMPED ≠ BOUND".** Provenance fields are CARRIED rather than DERIVED from the artifact that produced them, so a certificate merely repeats what a caller told it. Executed instances: the anchor accepted the recorded means with **four unrelated sample names** (`reproduced=True`) and issued a certificate for checkpoint `{"run_name": "some-other-run", "step": 1}`; `rollout_prediction(num_steps=1)` executes and can later be certified as `num_steps=25`; `{"certificate": GATE_CERTIFICATE, "passed": True}` unlocked TEST, and `dev_certificate(GateVerdict(True, (), {}), …)` issues a pass with `mean_delta=NaN` and empty CI; any name permutation is accepted as "the derangement" (a TEST-seeded mapping passed for DEV; byte-identical donors passed because nothing receives action bytes or digests); and the identical-noise contract lives in `action_use_plan`, **which nothing consumes**. **This is the structural-vs-checked rule moved up a level: it is no longer enough to make the wrong thing unconstructible — the EVIDENCE must be derived from the MEASUREMENT.**

**Two ordinary production bugs that would fail on the first real run:** `gs://` artifacts handled with `pathlib` (`Path("gs://…")` silently becomes local `gs:/…`) across run-report loading, certificate publication and benchmark loading; and the evaluator entry point cannot run at all — `main()` always reaches `run_anchor_validation()`, which always raises, so no phase can restore, score, reproduce the anchor or freeze the benchmark. **Planner note: "name the boundary in the error" was accepted for the DEVICE work; it does not extend to the ORCHESTRATION.**

**T5b's two BLOCKERs:** `confirm_on_test` runs only the primary gate, but plan §3e requires the action-use confirmation on TEST with an independently derived TEST derangement; and the DEV certificate is forgeable by its marker.

**ADOPTED AS A REQUIREMENT (the review's most valuable recommendation):** the tests manufacture ideal scalar tables and never exercise table production, artifact round-tripping, GCS I/O, real certificate consumption, VAE decode layout, Orbax restore templates, or bf16/sharding. **A small end-to-end fake-model artifact test — restore → rollout → decode → summarize → certificate → gate → TEST door, on a tiny model with a fake gfile — is now a precondition for the first real checkpoint smoke run.**

## Reviewer output (verbatim)

165,487
Both rounds are **REQUEST-REVISION**. The focused unit assertions are strong, but several trust boundaries remain caller-asserted, and the advertised evaluation path cannot execute.

## T5a `eval-anchor` findings

1. **BLOCKER — the evaluator entry point is intentionally nonfunctional.** [eval_wan_pos_rollout.py:629](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:629)

   `main()` always reaches `run_anchor_validation()`, which always raises `NotImplementedError`. Thus no phase can restore, score, reproduce the anchor, or freeze the benchmark.

   Concrete change: implement phase dispatch and the real restore → rollout → decode → summarize → certificate flow, with the anchor completing before any new-arm scoring.

2. **BLOCKER — production `gs://` artifacts are handled with `pathlib`.** [eval_wan_pos_rollout.py:277](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:277), [eval_wan_pos_rollout.py:524](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:524), [eval_wan_pos_rollout.py:605](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:605)

   The configured run report and evaluation roots are GCS paths, but `load_run_report`, certificate publication, and benchmark loading use `Path.read_text/write_text`. `Path("gs://…")` becomes a local `gs:/…` path.

   Concrete change: use `tf.io.gfile` or the repository’s storage abstraction for all artifact reads/writes, hashing the same bytes subsequently parsed. Add a fake-gfile GCS round-trip test.

3. **MAJOR — the anchor is not bound to its four historical samples or checkpoint.** [eval_wan_pos_rollout.py:168](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:168), [eval_wan_pos_rollout.py:194](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:194)

   I passed the recorded means with four unrelated names; `reproduce_anchor` returned `reproduced=True`, and `anchor_certificate` accepted checkpoint `{"run_name": "some-other-run", "step": 1}`. The current test deliberately uses `s0…s3`, so it misses this.

   Concrete change: pin the exact four names/order in an anchor manifest, require them in reproduction, validate the historical run and step, and derive the certificate from the issued measurement/restored identity rather than free mappings.

4. **MAJOR — deployed horizon and provenance are stamped, but not bound to the measurement.** [eval_wan_pos_rollout.py:391](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:391), [eval_wan_pos_rollout.py:552](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:552)

   `rollout_prediction(..., num_steps=1)` executes successfully; its result can later be certified by separately passing `num_steps=25`. Benchmark tables likewise carry caller-supplied checkpoint/code/model/horizon metadata unrelated to their producer. `load_benchmark_row` verifies the digest but not even `BENCHMARK_PROTOCOL` or semantic consistency.

   Concrete change: enforce/remove the horizon argument at the scoring boundary, produce a digest-bound measurement object carrying checkpoint and execution provenance, and derive certificates/benchmark rows from it. Validate protocol, schema, count, mean, and deployed horizon on load.

### A1–A5 rulings

- **A1 — PARTIAL/FAIL.** Selection-root derivation, AST exclusion of `build_checkpoint_manager`, and step/metric/arm/k checks are correct. `restore_from_report` exists and verifies its digest, but there is no production caller, and its configured GCS report cannot be read.
- **A2 — FAIL.** All three metrics, fixed 2% tolerance, and the wiring-only warning are correct. The actual four-sample identity is unenforced.
- **A3 — FAIL.** Publish-once/adopt-identical/refuse-different and surface provenance fields are present. Measurement identity/horizon are unbound caller assertions, and loading lacks semantic validation.
- **A4 — PASS.** `rollout_prediction` composes T3a’s `cfg_rollout`; no private sampler loop exists in this module.
- **A5 — PASS.** The A13 replacement is correct. It checks the unpinned native-bf16 draw bitwise against the deployed construction and explicitly rejects fp32-then-cast, while checking frame 0 separately.

**T5a verdict: REQUEST-REVISION.**

## T5b `eval-gates` findings

1. **BLOCKER — TEST confirmation omits the mandatory action-use confirmation.** [pos_rollout_gates.py:333](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:333)

   `confirm_on_test` accepts only primary `rollout`/`control` tables and calls only `gate_g3_vs_null_only`. Plan §3e requires true-vs-wrong action use to be repeated on TEST with an independently derived TEST derangement.

   Concrete change: keep one TEST door, but make it run and publish both primary and action-use confirmation, including TEST true/wrong/zero and matched-C0 controls.

2. **BLOCKER — a DEV certificate is forgeable by its marker.** [pos_rollout_gates.py:126](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:126), [pos_rollout_gates.py:342](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:342)

   `dev_certificate(GateVerdict(True, (), {}), …)` issues a passing certificate with `mean_delta=NaN` and empty CI. More directly, `{"certificate": GATE_CERTIFICATE, "passed": True}` unlocked TEST in a probe. The test rejects a missing/wrong marker but not the exact-marker forgery.

   Concrete change: have certificate issuance compute the primary gate internally from bound score artifacts; publish/load a digest-verified strict schema; and validate protocol, cohort/digest, horizon, constants, coverage, finite mean/CI, and pass conditions before TEST.

3. **MAJOR — the gate does not enforce that its mapping is the cohort’s seeded, byte-legal derangement.** [pos_rollout_gates.py:165](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:165), [pos_rollout_gates.py:225](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:225)

   The swap repair itself is bijective and fail-closed. However, `action_use_gate` accepts any name permutation: a TEST-seeded mapping passed for DEV, and both the plan and gate accepted a mapping whose actual donor actions were byte-identical because neither receives action bytes or their digests.

   Concrete change: return a derangement artifact carrying cohort, seed, permutation, action-sequence digests, and fingerprint; require and fully validate that artifact in planning, scoring, and gating.

4. **MAJOR — identical noise is structural only in an unused plan dictionary.** [pos_rollout_gates.py:225](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:225), [pos_rollout_gates.py:248](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:248)

   Nothing consumes `action_use_plan`; the gate accepts naked scalar tables. A future scorer can key wrong-action noise on the donor and still pass every scoped test.

   Concrete change: implement one table producer that consumes the plan and emits receiver/donor/draw-key identities, checkpoint, horizon, and condition provenance. Gates should accept only loaded artifacts and verify identical draw identities across conditions.

5. **MAJOR — coverage failure loses required provenance and breaks reporting.** [pos_rollout_gates.py:262](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:262), [pos_rollout_gates.py:309](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:309)

   The early return correctly avoids indexing a nonexistent `ci`, but omits permutation/hash/cohort/manifest. `action_use_report` then crashes on incomplete coverage rather than returning the failing verdict. I reproduced the missing provenance and `KeyError`.

   Concrete change: enrich both return paths from one helper and avoid computing reported deltas when coverage is invalid.

6. **MAJOR — matched-C0 does not receive its full required battery.** [pos_rollout_gates.py:294](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_gates.py:294)

   Plan §3e requires true/wrong/zero for both the selected checkpoint and matched-C0. `control_tables` requires exactly `{"true", "wrong"}` and never reports C0 true-minus-zero.

   Concrete change: require `{"true", "wrong", "zero"}`, validate exact finite cohort coverage, and report both C0 deltas without adding a pass field.

### B1–B5 rulings

- **B1 — PASS.** The imported exp_04 computation/constants are correctly pinned. The flagged adapter mapping is correct: `ssim → future_ssim`, `mse → future_mse`, at key `"0"`.
- **B2 — FAIL overall.** The bijective-swap algorithm is correct. Re-stating the no-margin CI decision is also correct in principle, including an early coverage return. Its coverage artifact, derangement binding, identical-noise enforcement, and C0 battery are incomplete.
- **B3 — FAIL.** One manifest-reading TEST door, digest pinning, and certificate hash stamping exist, but the door accepts forged certificates and confirms only the primary gate.
- **B4 — PASS narrowly.** The three first-draft fixes landed: explicit no-margin decision, keyword-only arm/control APIs, and a provenance-bearing TEST verdict. They do not close B3’s certificate flaw.
- **B5 — PASS.** A malformed/non-mapping entry becomes missing coverage; missing/nonfinite measurements are claim-penalized rather than imputed as perfect SSIM.

**T5b verdict: REQUEST-REVISION.**

## Execution note

I attempted the exact requested pytest command. This read-only review sandbox cannot create any temporary file, so pytest/TensorFlow failed with “No usable temporary directory”; those failures are environmental. With temp discovery bypassed, the non-writing subset passed: **23 anchor + 20 gate tests**. I also reproduced the defects above with read-only direct probes.

The unexecuted residual is therefore material: the tests manufacture ideal scalar tables and never exercise table production, artifact round-tripping, GCS I/O, real certificate consumption, VAE decode layout, Orbax restore templates, or TPU bf16/sharding. A small end-to-end fake-model artifact test should precede the first real checkpoint smoke run.
