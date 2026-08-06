# pos_context — Codex code review: S9 `evaluator-regressed-restore-rollout` (PARTIAL round — stall at the §6 matrix ratified)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: REQUEST-REVISION — 2 commit-blocking (optional DEV metadata; unverified TEST-gate certificate) + 1 MAJOR test-oracle gap. Stall boundary + both design positions RATIFIED. Strengthen dispatched same cycle; record follows.

## Reviewer output (verbatim)

## 1. Findings

- **Commit-blocking — DEV metadata is optional.** [`restore_selected_adapter`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_eval.py:54) defaults `expected_dev_metric=None`, causing the check at [line 74](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_eval.py:74) to be skipped. A direct probe restored a manifest containing only `step` and `l_pos`, violating the stated “metadata-absent ⇒ unevaluable” contract. Make the reported DEV metric required and always reject absent, nonfinite, or mismatching values.

- **Commit-blocking — TEST authorization is not a verified DEV-gate certificate.** [`k4_comparison_row`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_eval.py:124) trusts any mapping with truthy `passed`; it neither verifies `gate["cohort"] == "dev64"` nor binds the verdict to the selected manifest. A direct probe using `{"passed": True, "cohort": "test64"}` unlocked a TEST row. Require DEV provenance, and preferably selected-step/metric/`l_pos` identity.

- **Major test-contract gap — the signature test does not pin no-fallback.** [`test_restore_never_reads_a_resume_tree`](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_context_eval.py:169) only excludes the literal name `"manager"`. Adding `resume_manager`, `checkpoint_manager`, `resume_tree`, or `**kwargs` would pass. The implementation currently has no fallback path, but the oracle should pin the exact signature or otherwise exclude every secondary source.

The corrected coverage assertion matches exp_04 exactly: `numbers["missing_names"]["method"]`, alongside the sibling `baseline` payload. The specified suite was blocked before collection by the read-only sandbox’s lack of a writable temporary directory; with capture disabled, all 10 filesystem-free selected tests passed.

## 2. Positions

1. **Stall boundary: ratified.** No `generate_wan_null_adapter.py` exists in any reachable ref; exp_04 is at R11/J1-3, while R14/R15 are P1-gated. The exception names R14/R15 and the §6 matrix. The tripwire fires under the prescribed repository-root invocation, though resolving from `__file__` would remove its cwd dependence.

2. **Config-derived `l_pos`: ratified.** The real head is constructed from the same `pre_context_tokens` config, while template-based Orbax restore provides the structural shape check. The change is exp_05-owned, leaves the three S7 checkpoint items unchanged, and does not alter resume/selection retention semantics.

3. **Shift-never-decides: ratified.** Overall `passed` comes solely from imported `gate_g3_vs_null_only`; serialized-replay shift is returned separately and never participates in the boolean decision.

## 3. Verdict

**REQUEST-REVISION — require DEV metadata, harden the no-fallback oracle, and make TEST rows consume a verified DEV-gate certificate before committing this partial round.**

## Strengthening record + close

All three findings + the nit closed: (1) expected_dev_metric REQUIRED keyword-only (TypeError on the reviewer's probe; inspect-level pin against quiet re-optionalization); (2) the gate result is a stamped v1 CERTIFICATE with a required selection binding — k4_comparison_row requires stamp + dev64 cohort + identity triple equal to the restore-verified manifest (both reviewer probes refused, plus a genuine-certificate-wrong-checkpoint probe); (3) the signature oracle pins the exact parameter list with kinds, no VAR_* catch-all; (4) the tripwire resolves from __file__ with a repo-root sanity assert. Battery 26/26 — four first-pass survivors closed by SHARPER PROBES, not code churn: correctly-bound forgeries isolating the stamp/cohort checks; per-field cannot-identify probes isolating the identity binding. Focused 39; full suite 1417. Closed WITHOUT a third Codex pass: the fix is the review's own prescription including its 'preferably' clause verbatim — judgment recorded per the S7 precedent; any challenge lands in the next review.
