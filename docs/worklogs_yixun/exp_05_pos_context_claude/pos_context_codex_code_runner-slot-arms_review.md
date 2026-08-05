# pos_context — Codex code review: round S4 `runner-slot-arms`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_05 worktree). HEAD `fd7d823`. Rulings: additivity syntactic PASS / semantic FAIL (HyperParameters raises ValueError on missing keys — the legacy null launch crashes); l_pos schema APPROVED, integration not; S4b K1-BLOCKING; no-videos NOT blocking.

## Reviewer output (verbatim)

Context loaded:

- [experiment_SOP.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/experiment_SOP.md), in full.
- [plan_pos_context.md v3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md:40): P1′ B-arms/H1 note, §5 items 2–3, §6 additive-union policy, and S5-amended P2′.
- [pos_context_worklog.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_worklog.md:136), through the S4 `l_pos`, MUST, and S4b decisions.
- exp_04 merged boundary `cdd4653` via merge `8695fac`: `run_wan_null_inversion.py` and [null_adapter_modes.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/null_adapter_modes.py), in full.
- exp_04 review history: [R10 launchers/config](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_code_launchers-config_review.md), including follow-up/residue closure, plus the supporting [capacity](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_code_runner-capacity-core_review.md), [decode/video](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_code_runner-decode-videos_review.md), and [cache/resume](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_codex_code_runner-cache-resume_review.md) contracts.
- All S4 code: [pos_context_modes.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py), [pos_context_records.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_records.py:131), [run_wan_null_inversion.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:95), and [test_pos_context_runner.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_runner.py), including the exact dual-touch diff.
- Validation: `git diff --numstat` confirms exactly `24/0` on the runner and `null_adapter_modes.py` is untouched. The exact pytest command could not initialize pytest capture without a writable temporary directory; capture-disabled execution with `tempfile.tempdir` preseeded passed all **11 tests in 16.36s**.

## Findings

1. **BLOCKER — semantic additivity fails on the real MaxDiffusion config object.** [run_wan_null_inversion.py:104](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:104), [pyconfig.py:316](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pyconfig.py:316)

   `getattr(config, "embedding_slot", "")` does not use its default with `HyperParameters`: the class raises `ValueError`, not `AttributeError`, for a missing key. The unchanged exp_04 YAML has no `embedding_slot`, so an existing null launch now fails before `plan_run`. I reproduced this using the exact `HyperParameters` class definition: `ValueError: Requested key embedding_slot, not in config`.

   The characterization test misses this because it uses `SimpleNamespace`; it also compares only seven report fields, not the complete report or emitted artifacts.

   Concrete change: resolve optional keys through `config.get_keys().get("embedding_slot", "null")` for `HyperParameters`, with a safe mapping/generic-object fallback; add a test using the real missing-key behavior and compare the complete null sink trace. A launchable positive YAML must also declare the key because CLI overrides for undeclared YAML keys are rejected.

2. **MAJOR — positive records can be published under false provenance, reopening exp_04 R6’s central defect.** [pos_context_modes.py:233](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:233)

   `build_pos_capacity_records` ignores the result’s bound `params` and `batch_fingerprint`. It does not compare the current base context, names/tensors, canonical sigmas, guide scale, optimization recipe, or `l_pos` against what produced the embeddings. Empirical probes accepted:

   - embeddings made at `w=5, J=1, lr=.01` under a header claiming `w=7, J=50, lr=.03`;
   - a header claiming `l_pos=1` while the record stored eight rows;
   - the same arm results paired with changed `z_i0` and `z_video`.

   Concrete change: port exp_04’s strengthened writer preflight: validate the positive header, canonical sigmas, current/result context fingerprints, exact names and batch fingerprint, guide scale, J/lr, exact example-field namespace, and `header.l_pos == stored embedding rows == produced l_pos`, all before replay.

3. **MAJOR — positive artifacts and adequacy state are not slot-isolated, and selected B-arms are mislabeled as A-arms.** [run_wan_null_inversion.py:681](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:681), [run_wan_null_inversion.py:702](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:702), [pos_context_modes.py:365](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:365)

   The positive route continues to use `null_artifact_dir`, `null_staging_dir`, and `null_adequacy_uri`. My positive-run probe wrote B1/B2 paths beneath `gs://bucket/artifacts`, the configured null root, alongside the same `gate_tables.json`, `selection.json`, and `run_report.json` filenames. It can also consume a null-slot adequacy artifact.

   `selection_payload` is null-specific: a passing positive B1 selection serializes the inconsistent tuple `target="A1/keyed", arm="b1", label="B1"`; its nested gate selection also says `A1/keyed`.

   Concrete change: use a dedicated positive root/adequacy URI or a mandatory slot namespace, bind every JSON artifact to the slot, and build a positive selection payload that serializes B1/B2 and H1/H2 consistently. Add a probe proving no positive write or read touches a null path.

4. **BLOCKER — K1’s positive adequacy probe and `L_pos ∈ {1,8}` ablation are not wired.** [run_wan_null_inversion.py:354](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:354), [pos_context_modes.py:179](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:179), [pos_context_modes.py:423](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:423)

   `plan_run` emits only `l_null`; the B-arm runner calls `pos_context_from_t5` and the optimizer at their fixed eight-row defaults; `pos_execute` explicitly refuses `adequacy_probe`. Empirically, a config requesting `l_pos=1` produced a plan containing `l_null=16` and a positive header with `l_pos=8`.

   This omits plan P1′’s adequacy/adoption evidence and diagnostic ablation, so K1 cannot satisfy its approved method even after the shard writer lands.

   Concrete change: add positive-specific plan parameters, implement the first-eight-DEV adequacy/adoption path, and thread `l_pos` into inversion, warm start, frozen replay, and optimizer initialization. Keep shard publication fixed at eight rows; the one-row arm is diagnostic-only.

5. **MAJOR — the S4b interim refuses too late to be fail-closed at the run-artifact level.** [pos_context_modes.py:369](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:369), [pos_context_modes.py:379](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:379), [pos_context_modes.py:435](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:435)

   With the default refusing writer, the run completes all expensive arms and gates, publishes `gate_tables.json` and `selection.json`, then raises on the first shard and never publishes `run_report.json`. The probe observed exactly those two partial artifacts.

   No false completed shard is created, so marker-based record resume remains safe; however, the artifact root now contains an apparently authoritative selection from a failed run. That is not a sound fail-closed interim.

   Concrete change: land S4b before K1. Until then, refuse during preflight before model computation or artifact writes. With the writer present, stage the run-level JSON and make the authoritative selection/report visible only as part of a successful publication sequence.

6. **MAJOR — the cast arithmetic passes, but the claimed end-to-end MUST test exercises an unused wrapper.** [pos_context_modes.py:102](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:102), [run_wan_null_inversion.py:694](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:694), [test_pos_context_runner.py:493](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_runner.py:493)

   The real tiny-Wan test passes for both branches at bf16, and double wrapping is bit-idempotent for both context shapes. The merged exp_04 production closure also correctly casts every `encoder_hidden_states` at [run_wan_null_inversion.py:555](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:555).

   But `casting_velocity_fn` has no call site: `main` passes `backend["velocity_fn"]` directly. Thus the current production backend is correct by inheritance, while the claim that S4’s wrapper closes and tests the actual runner wiring is false; an injected or future positive backend bypasses it.

   Concrete change: either wire the wrapper into the positive dispatch and test `main` observing bf16 for both branches, or refactor/test the exact production velocity factory used by `_load_backend`; retain an explicit double-cast idempotence assertion.

## Rulings

- **Additivity:** syntactic claim **PASS** (`+24/−0`, sibling modes untouched); semantic claim **FAIL**. The real legacy config crashes, so the class-(c) additive-union protection is not established. Report-field comparison plus plan-key pinning is insufficient.
- **`l_pos` design:** **APPROVE at the schema-design level**. A positive-specific `l_pos` plus explicit `embedding_slot="positive"` correctly avoids misleading `l_null` semantics. Reusing the canonical sorted-JSON SHA-256 fingerprint rule is sound. The current record-builder integration is not approved because the header is not bound to what ran.
- **S4b shard writer:** **K1-blocking**. Deferral to an immediate reviewed S4b is acceptable only if K1 approval explicitly waits for it; the present late refusal is not launchable or wholly fail-closed.
- **No-videos deferral:** **not K1-blocking**. The gates depend on full-cohort decoded metric tables, not comparison MP4s; videos may remain diagnostic/P4 work, provided the successful slot-isolated report and tables exist.

Final verdict: **REQUEST-REVISION — semantic additivity is broken, and the positive route is not K1-safe until provenance binding, slot isolation, adequacy/`L_pos`, cast wiring evidence, and S4b publication are corrected.**

---

## Follow-up pass (S4 two-sitting strengthen @ working tree on `41d2e20`)

### Reviewer output (verbatim)

Context loaded:

- S4 review at [pos_context_codex_code_runner-slot-arms_review.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_codex_code_runner-slot-arms_review.md).
- Plan P1′, gate definitions, L_pos ablation, and H1 interpretation note in [plan_pos_context.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md:40).
- Worklog through the banked sitting-1 entry in [pos_context_worklog.md](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_worklog.md:150); sitting-2 claims came from the briefing because no second entry is present on disk.
- HEAD `41d2e20`, complete uncommitted delta, new positive YAML, and inherited exp_04 runner/shard/gate implementations.
- Focused 45-test suite. Exact command: 43 passed, with one failure and one error caused solely by the read-only sandbox having no usable temporary directory. With `tempfile.tempdir` preseeded and the real-filesystem writer case deselected: 44 passed, 1 deselected. The bf16 real-model test separately passed; the seven named non-filesystem probes passed together.

Finding verification:

1. **Finding 1 — CLOSED.** Real `HyperParameters` behavior reproduces the exact missing-key `ValueError`; `get_keys()` now resolves the legacy null slot correctly. The positive YAML is the null YAML plus only declared positive keys, and the null YAML remains unchanged.

2. **Finding 2 — CLOSED.** All three previously accepted false-provenance cases are rejected before replay: recipe/guide mismatch, `l_pos` mismatch, and same names with different batch tensors. Canonical sigma, current/result context fingerprints, names, batch fingerprint, and exact example-field namespace are also checked.

3. **Finding 3 — NOT CLOSED.** Published shards/JSON use the positive root and the top-level selection tuple is B-labeled with an enforcing consumer. However:

   - Positive `main` still checks `null_artifact_dir` and sweeps `null_staging_dir` at [run_wan_null_inversion.py:707](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:707). The checked-in positive YAML therefore raises `FileNotFoundError` on `os.statvfs("")` before loading manifests.
   - A populated positive run was observed checking the null artifact root and sweeping the null staging root, never the positive staging root.
   - Root comparison is unnormalized: `pos_artifact_dir="gs://bucket/artifacts/"` is accepted beside `null_artifact_dir="gs://bucket/artifacts"`.
   - A payload carrying `embedding_slot="null"` at `pos_adequacy_uri` was accepted and changed the positive recipe; URI separation is not consuming-side provenance validation.

4. **Finding 4 — NOT CLOSED.** The `l_pos=1` probe now yields one row throughout inversion, optimizer/frozen positive contexts, results, and header. Adequacy invalid requests preflight before data/forwards, and the ablation is publication-guarded. But the ablation does not use the adopted recipe required by P1′: forcing adoption of `(50, 0.03)` produced ablation calls `(1, 0.01)` for both L=1 and L=8 at [pos_context_modes.py:505](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:505). It is also fixed to the trajectory/B1-style calculation rather than the ultimately selected B arm.

5. **Finding 5 — ORIGINAL DEFECT CLOSED, CLAIM NARROWED.** The positive writer follows staging, immutable destination, cleanup, and marker-last publication; run JSONs appear only after all shards succeed. A first-writer failure leaves no artifacts in the test. “Zero artifacts on any failure” is false, however: when B1 succeeded and B2 failed, one complete B1 shard remained and no JSONs were published. That shard is safe but makes same-root retry collide with immutable publication.

6. **Finding 6 — CLOSED.** `main` wraps the actual positive-dispatch velocity function. Main-observed contexts were bf16, and the real tiny-Wan test passed for both conditional and unconditional branches.

The two rulings:

1. **A-worded gate reasons:** acceptable under plan §4-P1′. H1/H2 deliberately reuse `gate_g1/g2` verbatim, and the H1 interpretation note adequately scopes their meaning; relabeling reason strings is not required. The nested `gates.selection.target` also remains A-labeled, which is structural rather than a reason string, but the authoritative top-level B tuple and consumer are consistent; I do not rate that residue K1-blocking.

2. **K1 readiness:** not structurally K1-launchable yet. Even before the parity audit, P0′, and K1 smoke, the checked-in positive config cannot enter `main`, positive execution touches the null storage roots, adequacy adoption is not slot-bound, and an adopted recipe is not applied to the required L_pos ablation.

NEW findings at K1-blocking severity:

- **K1-BLOCKER — positive storage preflight is still null-wired.** Resolve and validate normalized positive roots before storage operations; require/check `pos_artifact_dir`, require/sweep `pos_staging_dir`, and never inspect or mutate null staging during a positive run.
- **K1-BLOCKER — adequacy adoption lacks positive-slot binding.** Validate at least `embedding_slot`, mode/cohort, `l_pos`, guide scale, and relevant manifest binding before applying an adoption.
- **K1-BLOCKER — the L_pos diagnostic ignores the adopted recipe/selected arm.** Run it with the recipe and arm that the main K1 study actually adopts/selects, and persist those bindings in its evidence.

Final verdict: **REQUEST-REVISION — four fixes are substantively closed, but slot-safe production preflight/adoption and the plan-mandated adopted-recipe L_pos ablation still block K1.**

---

## Final-fix record (the follow-up's 3 residual K1-blockers) + Planner closure

1. **(storage preflight) — FIXED.** `positive_roots` (required, normalized, collision-rejected incl. trailing-slash both directions); positive-root free-space + positive-staging sweep; null roots never touched on a positive run (root-observation probe); the checked-in pos YAML now enters main. Two new injectable seams (sweep/free_space) per the file's design.
2. **(adoption binding) — FIXED.** `pos_adoption` validates slot/mode/cohort/l_pos/guide_scale/manifest binding before applying; the null-slot-payload probe is a rejection; `run_pos_adequacy` records `manifest_hash`.
3. **(ablation binding) — FIXED.** Ablation runs at the ADOPTED recipe and the mirrored arm, both persisted in evidence; the forced-(50,0.03) probe is the regression.
Finding-5 claim narrowed to the exact semantics (first-shard failure ⇒ empty root; later-shard ⇒ immutable partials + resume-not-republish per R8; run-JSONs never describe incomplete record sets).

Suite **1063 passed**; cumulative S4 battery **28 mutants, 0 survivors** (V+W+X series). Dual-touch delta +106/−6 — the six deletions are slot-relocated call sites with exp_04's expressions preserved verbatim on the null branch; the null-path characterization (real-HyperParameters, complete-report) pins behavioral identity — recorded as within the class-(c) additive-union spirit. Round S4 closed (two sittings + follow-up + finals); committed with this record.
