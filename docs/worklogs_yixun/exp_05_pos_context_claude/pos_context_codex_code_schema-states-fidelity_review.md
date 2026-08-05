# pos_context — Codex code review: round S5 `schema-states-fidelity`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_05 worktree). HEAD `6d411b0`. Independent verification: 6-seed 6M-value probe CONFIRMED the 6.27% divergence frequency and the witness; REFUTED the always-1-ulp absolute (subnormal-region divergences of 2–35 bits) and corrected the overflow/underflow thresholds; storage arithmetic and ml_dtypes/jnp bitwise identity independently confirmed; P7 equivalence verified.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md`, both standing announcements, and `pos_context_yixun_query.md`.
- `plan_pos_context.md` v3 in full, including §4-P2′’s struck expectation and S5 amendment.
- `pos_context_worklog.md` through S5.
- Related exp_01/exp_02 analyses.
- exp_04’s `null_adapter_records.py`, `null_adapter_cache_policy.py`, and the complete R4b review/strengthening history.
- S5’s `pos_context_records.py` and all 34 tests, in full.
- Repository state: HEAD `6d411b0`; both S5 files uncommitted.

Measurement-verification verdict: REFUTED.

My independent six-seed probe over 6,000,000 `N(0,1)` fp32 values found 376,013 divergences, or 6.266883%, confirming the headline frequency. The witness is also correct:

- `x = 1.00439453125`
- fp16 intermediate: `1.00390625`
- direct bf16: `1.0078125` (`0x3f81`)
- via fp16: `1.0` (`0x3f80`)

However, 13 divergences were not one bf16 ULP; observed bit distances were 2, 3, 4, 6, and 35, all from fp16-subnormal-region inputs. Thus “always exactly one ulp” is false for an unqualified `N(0,1)` distribution.

Range destruction is correct in substance but imprecisely described: fp16’s largest finite value is 65,504, but round-to-nearest produces infinity starting at 65,520—not for every value above 65,504. Its minimum positive subnormal is `2^-24 ≈ 5.9605e-8`; values round to zero at the halfway boundary `2^-25 ≈ 2.9802e-8`, not throughout everything below `~6e-8`.

The conservative default is correct: after a bit-identity failure, absence of feature deltas must select fp32.

Findings:

1. BLOCKER — The load-bearing measurement amendment contains a false absolute claim. [pos_context_records.py:24](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_records.py:24), [test_pos_context_records.py:316](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_records.py:316), [plan_pos_context.md:46](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md:46)

   Concrete change: revert the S5 amendment as required, then replace it with a corrected, independently reproducible statement: approximately 6.25% divergence; one ULP for fp16-normal inputs, with rare larger divergences in the fp16-subnormal region. Correct the overflow/underflow thresholds and add a deterministic multi-ULP subnormal witness.

2. MAJOR — The public fidelity gate does not enforce the predeclared eight-example production cohort and has fail-open inputs. [pos_context_records.py:302](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_records.py:302), [test_pos_context_records.py:354](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_records.py:354)

   It has no manifest argument, accepts a single caller-chosen feature delta, and accepts arbitrary or empty shapes. Empirically, empty arrays return fp16/“bit-identical.” A finite `70000.0` state becomes fp16 `inf`, yet `feature_deltas={"only-one": 0.0}` returns fp16 despite `max_abs_delta=inf`.

   Concrete change: derive the first eight names from the DEV manifest, require exact evidence coverage and production state geometry for those eight, reject empty/nonfinite fp32 inputs, and select fp32 immediately when fp16 serialization becomes nonfinite. Preserve the existing fp32 default when feature evidence is absent.

3. MINOR — Closed-namespace behavior is implemented but not fully independently tested on this schema. [test_pos_context_records.py:249](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_records.py:249), [test_pos_context_records.py:283](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_records.py:283)

   The “unknown and missing” archive test only adds an unknown member. There are no missing-member, missing-metadata-key, or extra/missing shape-key tests, so removal of the shape-namespace check would survive.

   Concrete change: add the four explicit R4b-style tampering cases with matched `ValueError`s.

4. MINOR — The launch table still carries the superseded 17.1 GiB estimate. [plan_pos_context.md:89](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/plan_pos_context.md:89)

   Concrete change: update K2 to 7,468,516 payload bytes/record and 14.80 GiB for 2,128 records, while retaining a separate free-space safety margin.

Verified without additional findings:

- Focused suite: 34 passed in 0.49s with capture disabled; the exact command was initially blocked only because the read-only sandbox provided no pytest capture temp directory.
- Stored-byte hashing, writer/reader array freezing, deterministic bytes, latent-dtype scope including `z_bar_states`, and production geometry are sound.
- `_PosGeometry`’s local rebuild is justified by F6; both added invariants are correct.
- P7 is genuinely equivalent: the early byte check rejects truncation, and the shared `_validate` geometry check independently rejects it after the mutated first layer.
- Storage arithmetic is correct: 7,468,516 payload bytes/record; 15,893,002,048 bytes = 14.8015 GiB/cohort. Actual serialized overhead was 2,647 bytes/record, yielding 14.8068 GiB.
- `ml-dtypes>=0.5.4` is present in generated project requirements. Against JAX 0.10.2, 1,000,000 random plus 11 targeted values had zero bit mismatches, and `jax.dtypes.bfloat16 is ml_dtypes.bfloat16`. This pins CPU JAX conversion semantics, though not target-TPU execution.
- The `ProvenanceHeader.l_pos` decision may remain an S4 open item, but S4 must not reuse `l_null` with misleading semantics.

Final verdict: REQUEST-REVISION — the load-bearing measurement is false as written, and the public gate does not enforce the plan’s predeclared production cohort.

Status:

- No subprocesses are running; review is complete.
- Earliest steer: now; it is safe to begin S5 strengthening.
