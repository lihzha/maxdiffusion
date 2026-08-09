# null_adapter — Codex ANALYSIS review of the P4 reports

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: **REQUEST-REVISION** — measurements largely verified; the closing CONCLUSIONS exceed the evidence in four places.

**THE OVERTURN THAT CHANGES THE EXPERIMENT'S FORMAL STATE: the over-determination argument does NOT rescue the STOP.** A1-probe's 0.1729 is not independent of the optimization budget (a J=50 A1 produces different locked nulls, hence a different probe), and even if A1 still failed, A2's J=50 G2 result is UNMEASURED ⇒ *"the artifact genuinely says target=stop at J=10, but the plan-compliant target selection is INDETERMINATE."* Two acceptable resolutions: report "observed J=10 STOP; formal selection unmeasured; discretionary decision not to advance," **or re-run at J=50 to retain the predeclared verdict** — the second is exactly the clean-gate rerun Yixun has already approved, so the rerun now DECIDES the formal outcome rather than merely cleaning the record. Also struck: "a stronger optimizer could only have raised A2" (no monotonicity guarantee).

Other required edits: the J1b/J1c objective-shape attribution stays COMPUTE-CONFOUNDED (~19× budget; "objective shape is a promising explanation, but a budget-matched greedy probe is required for causal attribution"); "G1 failed on its control" → "G1's absolute A1 conditions passed; only its ratio against the strong, CFG-collapsed A0 control failed" (A0 behaved as designed); and "inert" is TOO CATEGORICAL for A1-probe (keyed vs global conventions differ, no matched base-null keyed control exists — "+0.031 above the nearest UNMATCHED do-nothing proxy; its matched incremental effect was not measured").

## Reviewer output (verbatim)

198,824
## Verdict: REQUEST-REVISION

The artifacts support most reported measurements, but the closing conclusion and several causal claims exceed the evidence.

### Blocking findings

1. **The formal STOP is not rescued by over-determination.**

Plan §4 required A1/A2 and their gates to run at adopted J=50. They ran at J=10. The current argument fails because:

- A1-probe would itself change when A1 is recomputed at J=50; its 0.1729 is not independent of optimization budget.
- Even if A1 still failed its probe, A2 remained the fallback. A J=50 G2 result is unmeasured.
- More iterations are not guaranteed to improve—or worsen—decoded SSIM monotonically.

Thus the artifact genuinely says `target="stop"` at J=10, but the **plan-compliant target selection is indeterminate**. Revise [results §1/§4.2/§4.4](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_results.md:18>) and [analysis §1/§2.3/§4.1](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:10>) accordingly.

Two acceptable resolutions:

- Report “observed J=10 STOP; formal plan-compliant selection unmeasured; discretionary decision was not to advance,” or
- Re-run at J=50 before retaining “predeclared TARGET=STOP.”

Also delete “a stronger optimizer could only have raised A2”; tracking loss and endpoint SSIM need not move monotonically.

2. **The objective-shape attribution remains compute-confounded.**

J1c is cleaner evidence about transfer than J1b, but it does not isolate the objective: A3 had roughly 19× J10 compute. The proposed sign of the confound is plausible, not established. More greedy iterations might first learn transferable shared corrections before overfitting basin-specific details.

Replace claims such as “greedy is the artifact,” “the transfer half survives the budget confound,” and “demonstration that objective shape binds” with:

> Joint endpoint optimization at substantially greater compute produced better n=8 capacity and transfer; objective shape is a promising explanation, but a budget-matched greedy probe is required for causal attribution.

This affects [analysis §1 and §3](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:20>).

3. **The A0 CFG account is mechanically sound, but its headline is not.**

Frozen base nulls reproduce the base context, so `v_unc == v_cond`; CFG becomes independent of `w` and matches the single-context inversion dynamics. “Near-self-consistent replay” is a reasonable inference.

However, “G1 failed on its control” implies a defective control. A0 behaved exactly as designed, and MSE ratios have no fixed “~3× headroom.” Use:

> G1’s absolute A1 conditions passed; only its ratio against the strong, CFG-collapsed A0 control failed.

Revise [analysis §2.1](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:32>).

4. **Retracting “actively destructive” is correct; replacing it with “inert” is too categorical.**

A1-probe 0.1729 is above A2-0 0.1423, but these use different noise conventions—keyed versus global—and no matched base-null keyed control exists. The supported wording is:

> A1-probe has very low absolute quality and is +0.031 above the nearest unmatched do-nothing proxy; its matched incremental benefit or harm was not measured.

Revise [results §4.2](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_results.md:152>) and [analysis §2.3](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:55>).

5. **Additional artifact-level corrections are required.**

- The DEV “future-pixel-MSE” column is mislabeled. Except for A0, its values are `full_pixel_mse`; A0’s 0.02085 matches neither mean. Either rename it and set A0 to **0.0212166**, or retain “future” and use:

  `0.0218796, 0.0066857, 0.1922965, 0.2264450, 0.0342444, 0.1097904` for A0, A1, A1-probe, A2-0, A2, A2-probe.

- J1b first gradient-norm range is **6.5797–16.6542**, not 8.49–16.65.
- The 0.7007 retention statistic is equivalently the mean of 24 ratios or eight per-clip mean ratios, but **0.428–0.855 is the range of the eight clip means**. The 24 individual ratios range **0.3856–0.9732**.
- “All other figures reproduce exactly” must be removed.
- Replace the mutable “branch/tip … clean” entry. The report was committed at `6aefa6c`; record evidence-producing tips separately.

6. **Several n=8 and cross-experiment statements need narrower scope.**

- None of the eight A3 endpoints matches or beats its **paired** A1 MSE. Three merely fall inside the pooled cross-clip A1 range. Replace “3/8 reach own-basin quality” with that exact statement.
- “Floors unmet everywhere” is false at the observation level: some clips/seeds exceed 0.70. Say “every aggregate setting mean missed the 0.70 floor.”
- Annotate headline J1 values everywhere as **J=10, not adopted J=50**. A caveat only in §4.4 is insufficient.
- The exp_05 comparison must show J10 versus J50 in the table itself. “Positive channel is more powerful/dangerous” is not licensed by recipe-, context-, pivot-, or control-mismatched arms.
- The 0.2946 caveat is good, but [analysis §6.1](</Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_analysis.md:240>) contradicts it by saying the deployed adapter “does not come close.” Remove that comparison.
- “Static per-clip targets are dead” is too universal. The evidence rejects the **tested greedy cached targets**. Multi-noise joint optimization of a static tensor remains untested, as does a state-conditioned emitter.
- “Frozen backbone exonerated” should become “the oracle demonstrates backbone expressivity on these evaluated clips”; it does not establish deployable predictive sufficiency.

### Answers to the five open questions

1. **Over-determination:** special pleading as written; formal selection remains unmeasured.
2. **Asymmetric confound:** plausible hypothesis, not a valid elimination of the compute confound. J1c is cleaner but not causal.
3. **A0 CFG collapse:** algebraically correct; “G1 failed on its control” is overstated.
4. **J=10 annotation:** yes—put it in every headline/table and terminal-status statement.
5. **Static-target versus rollout distinction:** visually prominent, but scientifically too binary. Scope “failed” to the tested greedy targets and list robust multi-noise static targets among the untested alternatives.

Most core numbers did verify: arm means, gate statistics/CIs, adequacy scores, J1b losses/endpoints, J1c means and correlations, provenance, and the complete failed-job trail. The cached primary evidence survived under the supplied `p4_evidence` path. Direct GCS re-fetch was unavailable because the read-only environment prevented `gcloud` from opening its credential database.
