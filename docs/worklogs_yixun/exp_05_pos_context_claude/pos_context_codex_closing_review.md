# exp_05 closing analysis review (revision 2)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-10. Verdict: **REQUEST-REVISION** — "The exp_05 STOP and gate record remain valid"; two scoping leaks + the exp_04-J=50 fold-in + artifact-index staleness; no new compute required.

Verdict: **REQUEST-REVISION**. The exp_05 STOP and gate record remain valid, but the reports are **not yet publication-ready for closure adjudication**.

The prior review was largely resolved: §3.2 now has the complete mixed-signature table and hedge; the proposed diagnostic covers all eight B2 shards; §4.1 is retention-led; rollout loss is correctly “not refuted, n=8 motivating oracle”; and all five factual fixes are present. Two scoping leaks and newer evidence still require edits:

1. Close the remaining overclaims in the analysis summary.

   - [Analysis §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:20) still says “Capacity is not the limit.” Narrow this to **output-channel expressivity is not limiting on these clips; head function capacity remains untested**.
   - [Lines 28–30](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:28) still conclude that “a per-clip conditioning tensor is the wrong object to regress onto.” Replace with the adopted scope: **the tested single-basin greedy-pivot cached-target family was rejected for K2/K3**. Robust/multi-noise static targets and state-conditioned emitters remain untested.
   - Delete [“On the full cohort the positive probe would likely lead”](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:151); no 64-clip joint-null arm exists, so that extrapolation is unsupported.

2. Fold in exp_04’s terminal J=50 result everywhere exp_04 is treated as J=10-only evidence.

   - Remove the now-false “Unlike exp_04” statement in [results §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:29).
   - Rewrite [results §4.3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:198) and [analysis §5.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:188): the comparison is now recipe-matched at J=50. Remove the budget mismatch and “INDETERMINATE/pending” language; retain the three real mismatches—control, representation, and pivots.
   - Update the [fresh-noise table](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:164) to exp_04 `4.743 → 0.303`, **15.7× better**, versus exp_05 **4.0× worse**. Update the in-basin comparison to **0.8868 versus 0.9227, gap 0.036**.
   - Replace exp_04’s historical 3.6× G1 ratio with the J=50 **4.681×** in [analysis §5.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:191) and [results caveat 4](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:313).
   - Update [analysis §5.3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:215) from null `0.4973/0.173` to the J=50 `A2=0.6638` and `A1-probe=0.1666`.
   - Historicalize the launcher wording in [results §4.4](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:227) and [analysis §6](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:236): the original J1-4 launcher failed adoption, but J1-5 fixed it and completed the clean rerun.

3. Preserve—but explicitly label—the legitimate J=10-era J1b/J1c evidence.

   The matched-eight `0.730/0.4753` joint-null result was not rerun at J=50, so it should not be overwritten. Label the exp_04 rows in [analysis §4.1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:133) and the `~19×` statement in [§7.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:259) as **J=10-era J1b/J1c evidence**. The existing J=10-versus-J=50 caveat remains correct for J1c versus exp_05, even though the clean-gate comparison is now J=50-matched.

4. Update the reports for post-revision-2 artifacts.

   [Results artifact index](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:289), [results caveat 6](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:315), [analysis threat 5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:234), and [next step 3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:282) say no videos or HTML report exist. The capacity-video job succeeded and both HTML pages are committed, as recorded in the [worklog](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_worklog.md:304). Record the outputs and lowest-ordinal-subset deviation, update the provenance/tip row, and change the `DRAFT` status when the closing revision is committed.

No new experiment or gate computation is required for these edits.
Verdict: **REQUEST-REVISION**. The exp_05 STOP and gate record remain valid, but the reports are **not yet publication-ready for closure adjudication**.

The prior review was largely resolved: §3.2 now has the complete mixed-signature table and hedge; the proposed diagnostic covers all eight B2 shards; §4.1 is retention-led; rollout loss is correctly “not refuted, n=8 motivating oracle”; and all five factual fixes are present. Two scoping leaks and newer evidence still require edits:

1. Close the remaining overclaims in the analysis summary.

   - [Analysis §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:20) still says “Capacity is not the limit.” Narrow this to **output-channel expressivity is not limiting on these clips; head function capacity remains untested**.
   - [Lines 28–30](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:28) still conclude that “a per-clip conditioning tensor is the wrong object to regress onto.” Replace with the adopted scope: **the tested single-basin greedy-pivot cached-target family was rejected for K2/K3**. Robust/multi-noise static targets and state-conditioned emitters remain untested.
   - Delete [“On the full cohort the positive probe would likely lead”](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:151); no 64-clip joint-null arm exists, so that extrapolation is unsupported.

2. Fold in exp_04’s terminal J=50 result everywhere exp_04 is treated as J=10-only evidence.

   - Remove the now-false “Unlike exp_04” statement in [results §1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:29).
   - Rewrite [results §4.3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:198) and [analysis §5.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:188): the comparison is now recipe-matched at J=50. Remove the budget mismatch and “INDETERMINATE/pending” language; retain the three real mismatches—control, representation, and pivots.
   - Update the [fresh-noise table](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:164) to exp_04 `4.743 → 0.303`, **15.7× better**, versus exp_05 **4.0× worse**. Update the in-basin comparison to **0.8868 versus 0.9227, gap 0.036**.
   - Replace exp_04’s historical 3.6× G1 ratio with the J=50 **4.681×** in [analysis §5.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:191) and [results caveat 4](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:313).
   - Update [analysis §5.3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:215) from null `0.4973/0.173` to the J=50 `A2=0.6638` and `A1-probe=0.1666`.
   - Historicalize the launcher wording in [results §4.4](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:227) and [analysis §6](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:236): the original J1-4 launcher failed adoption, but J1-5 fixed it and completed the clean rerun.

3. Preserve—but explicitly label—the legitimate J=10-era J1b/J1c evidence.

   The matched-eight `0.730/0.4753` joint-null result was not rerun at J=50, so it should not be overwritten. Label the exp_04 rows in [analysis §4.1](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:133) and the `~19×` statement in [§7.2](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:259) as **J=10-era J1b/J1c evidence**. The existing J=10-versus-J=50 caveat remains correct for J1c versus exp_05, even though the clean-gate comparison is now J=50-matched.

4. Update the reports for post-revision-2 artifacts.

   [Results artifact index](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:289), [results caveat 6](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_results.md:315), [analysis threat 5](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:234), and [next step 3](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_analysis.md:282) say no videos or HTML report exist. The capacity-video job succeeded and both HTML pages are committed, as recorded in the [worklog](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/docs/worklogs_yixun/exp_05_pos_context_claude/pos_context_worklog.md:304). Record the outputs and lowest-ordinal-subset deviation, update the provenance/tip row, and change the `DRAFT` status when the closing revision is committed.

No new experiment or gate computation is required for these edits.
