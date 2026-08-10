# exp_04 closing analysis review (post-J1-5, revision 3)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-10. Verdict: **REQUEST-REVISION** — "The formal STOP at J=50 is valid and closure-grade"; remaining items are wording/scoping/staleness only; no new compute required.

## Verdict: REQUEST-REVISION

The formal **STOP at J=50 is valid and closure-grade**. The new gate numbers agree with the artifact-derived run record, and every substantive requirement from my previous review was applied. However, the reports are **not yet publication-ready for closure adjudication** because several revision-3 claims and stale revision-2 passages need correction.

1. **Narrow the reproduction claim to DEV-64.** Attempt 1 published DEV-64 only; TRAINFIT-16 completed only in attempt 2. Therefore “both cohorts … reproduced across two independent attempts” is too broad in [results §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_results.md:8), [analysis §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:31), and the completion table. Use: “STOP measured on both cohorts; DEV-64 gate/selection blocks reproduced byte-for-byte across two attempts.” Also update the results code-state row to include `adequacy-wiring` and **991 tests**, rather than the pre-fix 989-test state.

2. **Keep the compute-confound hedge, but integrate J1-5 more precisely.** The new probe movement disfavors the simple monotone counter-mechanism over J=10→50, but cannot eliminate a later or non-monotone improvement at A3-equivalent compute. Thus the causal hedge remains correct. In [analysis §3.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:183):

   - Replace the now-false “only ever measured at one budget” and “J=50 … was never run” statements.
   - Replace “budget-independent over this range” with “neither cohort-wide probe improved between the measured J=10 and J=50 endpoints.”
   - Do not directly compare the **DEV-64** A2-probe mean 0.2510 with the **n=8** J1c mean 0.4753. Derive the J=50 first-eight A2-probe mean, or explicitly label the aggregates cohort-unmatched.
   - Define the outstanding experiment as a compute-matched **J≈200+** probe, not J=50.

3. **The A2-vs-A0 flag is adequate, but the convergence wording is not.** Section 4.5.3 correctly calls the 0.0027 difference cross-arm, unmatched, descriptive, and non-gating. Keep that. In [analysis §2.4](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:126), change “converged to” and “terminated” to “landed near at J=50”; no plateau or convergence was measured. The hypothesis may remain explicitly untested. The command record’s “statistically indistinguishable” claim likewise has no cross-arm test; append a correction if the ledger is immutable.

4. **Remove the memorization/generalization overclaim.** Similar A1 means—0.8847 versus 0.8868—show no observed cohort-mean separation, but do not establish “no memorization gap,” particularly for per-clip oracles and TRAINFIT n=16. In [results §4.5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_results.md:297) and analysis, use: “the same gate-failure signature occurred in both sampled cohorts; the observed A1 means were nearly equal.” “Not unique to DEV-64” is safer than “not a dev-set artifact.”

5. **The recipe-matched cross-experiment framing is adequate, but stale contradictions remain.** Adjacency is acceptable because each row is internally matched and the comparison is repeatedly marked descriptive; mismatches (a), (c), and (d) remain sufficient to prohibit slot attribution. Update:

   - Results §4.4’s statement that exp_04/05 still differ in budget.
   - Analysis §5.1’s claim that the current rows “also differ in recipe.”
   - §5.2’s **9.7×** to the authoritative **15.7×**, and remove “would need the recipe held constant”—that condition is now met, while the other mismatches remain.
   - §5.3’s current G1 comparison from **3.6×** to **4.681×**, retaining 3.6 only as explicitly historical.

6. **Clean the closure/publication state.** J1-4 is still titled “the primary result,” the correction table still says “J=50 re-run pending,” and the reports claim the videos/HTML are outstanding even though tracked HTML reports and assets exist. Those HTML reports still advertise “INDETERMINATE pending J1-5.” Update them for J1-5 or mark them explicitly superseded, record this review’s adjudications in §8, and change the Markdown reports from `DRAFT` only after these corrections.

No new compute is required. The [analysis §4.1 resolution table](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:252) and the principal J=50 gate numbers themselves require no correction.
## Verdict: REQUEST-REVISION

The formal **STOP at J=50 is valid and closure-grade**. The new gate numbers agree with the artifact-derived run record, and every substantive requirement from my previous review was applied. However, the reports are **not yet publication-ready for closure adjudication** because several revision-3 claims and stale revision-2 passages need correction.

1. **Narrow the reproduction claim to DEV-64.** Attempt 1 published DEV-64 only; TRAINFIT-16 completed only in attempt 2. Therefore “both cohorts … reproduced across two independent attempts” is too broad in [results §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_results.md:8), [analysis §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:31), and the completion table. Use: “STOP measured on both cohorts; DEV-64 gate/selection blocks reproduced byte-for-byte across two attempts.” Also update the results code-state row to include `adequacy-wiring` and **991 tests**, rather than the pre-fix 989-test state.

2. **Keep the compute-confound hedge, but integrate J1-5 more precisely.** The new probe movement disfavors the simple monotone counter-mechanism over J=10→50, but cannot eliminate a later or non-monotone improvement at A3-equivalent compute. Thus the causal hedge remains correct. In [analysis §3.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:183):

   - Replace the now-false “only ever measured at one budget” and “J=50 … was never run” statements.
   - Replace “budget-independent over this range” with “neither cohort-wide probe improved between the measured J=10 and J=50 endpoints.”
   - Do not directly compare the **DEV-64** A2-probe mean 0.2510 with the **n=8** J1c mean 0.4753. Derive the J=50 first-eight A2-probe mean, or explicitly label the aggregates cohort-unmatched.
   - Define the outstanding experiment as a compute-matched **J≈200+** probe, not J=50.

3. **The A2-vs-A0 flag is adequate, but the convergence wording is not.** Section 4.5.3 correctly calls the 0.0027 difference cross-arm, unmatched, descriptive, and non-gating. Keep that. In [analysis §2.4](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:126), change “converged to” and “terminated” to “landed near at J=50”; no plateau or convergence was measured. The hypothesis may remain explicitly untested. The command record’s “statistically indistinguishable” claim likewise has no cross-arm test; append a correction if the ledger is immutable.

4. **Remove the memorization/generalization overclaim.** Similar A1 means—0.8847 versus 0.8868—show no observed cohort-mean separation, but do not establish “no memorization gap,” particularly for per-clip oracles and TRAINFIT n=16. In [results §4.5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_results.md:297) and analysis, use: “the same gate-failure signature occurred in both sampled cohorts; the observed A1 means were nearly equal.” “Not unique to DEV-64” is safer than “not a dev-set artifact.”

5. **The recipe-matched cross-experiment framing is adequate, but stale contradictions remain.** Adjacency is acceptable because each row is internally matched and the comparison is repeatedly marked descriptive; mismatches (a), (c), and (d) remain sufficient to prohibit slot attribution. Update:

   - Results §4.4’s statement that exp_04/05 still differ in budget.
   - Analysis §5.1’s claim that the current rows “also differ in recipe.”
   - §5.2’s **9.7×** to the authoritative **15.7×**, and remove “would need the recipe held constant”—that condition is now met, while the other mismatches remain.
   - §5.3’s current G1 comparison from **3.6×** to **4.681×**, retaining 3.6 only as explicitly historical.

6. **Clean the closure/publication state.** J1-4 is still titled “the primary result,” the correction table still says “J=50 re-run pending,” and the reports claim the videos/HTML are outstanding even though tracked HTML reports and assets exist. Those HTML reports still advertise “INDETERMINATE pending J1-5.” Update them for J1-5 or mark them explicitly superseded, record this review’s adjudications in §8, and change the Markdown reports from `DRAFT` only after these corrections.

No new compute is required. The [analysis §4.1 resolution table](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:252) and the principal J=50 gate numbers themselves require no correction.
