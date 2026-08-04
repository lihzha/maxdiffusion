# null_adapter — Codex code review: round R4c `verify-replay`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `90007e8`. Reviewer ran the suite and empirical probes (fp32-record/fp16-header acceptance; arm/convention swap; optional-lookup bypass of the proxy). Rulings: C10 KEEP; writer-contract records-side pin NO (pin via writer-order mutation test when R6/R8 introduces the writer); tampered-nulls test confirmed genuinely beyond reader protection.

## Reviewer output (verbatim)

Context loaded:

- `docs/worklogs_yixun/experiment_SOP.md`.
- `plan_null_adapter.md` v5: §4-P2, §5 item 5/F14, §6 amendment, and relevant §3 provenance/noise contracts.
- `null_adapter_worklog.md` through the R4c entry, including the writer-order discovery and Planner acceptances.
- R1–R4b reviews and strengthening records: `sigma-embed-noise`, `invert-trajectory`, `optimize-nulls`, `replay-operator`, and `record-schema-io`; also the plan-review F14 history.
- [null_adapter_verify.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_verify.py) and [test_null_adapter_verify.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_verify.py) in full.
- Relevant [null_adapter_records.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_records.py), `replay_with_nulls`, sigma construction, timestep/pinning, and context fingerprinting.
- Main-checkout `verify_reconstruction_from_null.py` at pinned Wan SHA `f37022874c588817d4ed77d463e3d27745053df4`.
- Repository state: HEAD `90007e8`; only the two R4c files are untracked.
- Validation: R4c 17/17 passed; full worklog suite 182 passed with only the acknowledged tiny-Wan tmpdir failure; Ruff and `git diff --check` passed.

1. **MAJOR — provenance does not fail closed across the artifact pair.** [null_adapter_verify.py:101](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_verify.py:101)

   `header.dtype_policy` is validated as a legal value but never compared with `record.latent_dtype`. The verifier also never reads or checks `record.noise_convention` and `record.arm`, even though J1/R8 know which arm and convention they are verifying.

   Empirical probe: an fp32 record paired with a header declaring fp16 was accepted and reached the replay; changing metadata from `A1/keyed` to `A2/global` was also accepted.

   Concrete change:

   - Require `header.dtype_policy == record.latent_dtype`.
   - Add required `expected_noise_convention` and `expected_arm` caller declarations, mirroring `expected_model_revision` and `expected_guide_scale`, and compare them before replay.
   - Expand `ALLOWED_RECORD_FIELDS` only for those two provenance fields—not any answer-bearing payload.
   - Add mismatch tests whose velocity callback fails if invoked.

   The existing model-revision/guide-scale declaration pattern is correct for J1 and the future runner; it should cover arm/convention too. Header schema validation is present, record schema is already enforced during decoding, and exact float32 recomputation of the sigma grid is sound.

2. **MAJOR — the restricted proxy does not make the must-not-read regression test airtight.** [test_null_adapter_verify.py:251](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_verify.py:251)

   A missing `SimpleNamespace` attribute catches `record.z_video`, but not an optional lookup such as `getattr(record, "z_video", None)` or `vars(record).get("z_video")`. The garbage-fields test also remains green if the forbidden value is merely inspected without affecting the result.

   I wrapped the current verifier with an optional `z_video` lookup; both current test scenarios still passed, despite the forbidden read occurring on the full record.

   Concrete change: replace `SimpleNamespace` with a hostile proxy whose `__getattribute__` permits only the declared fields and raises `AssertionError`—not `AttributeError`—for every other lookup, including `__dict__`. Keep an external access log for the final assertion. This makes `getattr`, `hasattr`, and namespace inspection fail loudly.

3. **MINOR — “before any replay” is stated but not tested.** [test_null_adapter_verify.py:162](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_verify.py:162), [test_null_adapter_verify.py:179](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_verify.py:179)

   The implementation currently performs provenance checks before line 110, but the tests use an ordinary velocity function and only match the eventual error. Moving a provenance check after 50 expensive model forwards would leave them green.

   Concrete change: run every provenance/config mismatch with a velocity callback that raises immediately if called, or assert a zero call count. Include schema, sigma, model, guide scale, context, dtype policy, `l_null`, arm, and noise convention.

C10 ruling: **KEEP.** Verifier-side fp32 normalization owns the numerical boundary even though `replay_with_nulls` currently repeats the conversion. Like R3’s `stop_gradient`, it protects intent against future refactoring and is harmless redundancy.

Writer-contract pin now: **NO.** Cast → replay → record is correct and the verifier is the right place to document the consumer requirement. A records-side test now would only simulate a writer that does not yet exist and would weaken the records module’s NumPy-only separation. It must be pinned by an actual writer-order mutation test when R6/R8 introduces that writer.

Additional rulings:

- The tampered-nulls test is genuinely beyond reader protection: the stored hash authenticates `expected_final_latent`, not the nulls’ semantic relationship to that target. Prefer verifying the object returned by `_record_from_bytes`, but that tightening is non-blocking.
- The cross-module `_validate_header` import is acceptable: one package-internal validator is safer than restating provenance checks. A public alias would be naming polish, not a correctness requirement.
- Separating sigma shift from CFG weight is correct.
- Current code genuinely avoids `z_video`, actions, losses, and trajectory state; the defect is regression-proofing, not present-day access.

Final verdict: **REQUEST-REVISION** — the replay mathematics are sound, but pair-level provenance and the central must-not-read test are not yet fail-closed enough for the final artifact contract.

---

## Strengthening record (Coder, same round; all three findings closed, none rejected)

1. **MAJOR (pair-level provenance) — FIXED.** dtype-policy/latent_dtype pair equality; required `expected_noise_convention` + `expected_arm` declarations checked before replay; allowed-fields set widened by exactly those two labels (answer-bearing fields asserted absent); all mismatch tests use a raising velocity callback.
2. **MAJOR (hostile proxy) — FIXED.** `_RestrictedRecord` with `__getattribute__` allowlist raising AssertionError (incl. `__dict__`), access log asserting both subset-of-allowlist AND that the four replay inputs were actually read; the reviewer's `getattr(record, "z_video", None)` probe (D7) now dies — under the old proxy it returned None and passed.
3. **MINOR (ordering) — FIXED.** Every provenance/config mismatch test uses the raising callback; mutants D4/D5 (checks moved after the replay) are killed — they would have passed under the old tests.

Suite **187 passed in 16.04s**; mutants D1–D7 killed; both files fully black-clean; records module still numpy-only standalone. Rulings carried: C10 KEEP; writer-order pin deferred to R6/R8 with its own mutation test. Behavior beyond findings: none (the new kwargs/checks are the findings). Round R4c closed; committed with this record.
