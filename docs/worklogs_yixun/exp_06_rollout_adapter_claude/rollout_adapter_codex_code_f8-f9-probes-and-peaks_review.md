# exp_06 F8+F9+F9b `probes-and-peaks` — combined Codex code review

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-14. Verdict: **REQUEST-REVISION — 3 MAJOR** (no blocker): (1) the runtime-watermark window closes before the eval/checkpoint phases it claims to bound (docstring asserts otherwise); (2) F8's honest-input reachability controls are not part of the executable battery — an always-refusing production regression would still print nine green lines; (3) the declared ladder order is bypassable via the public cells= seam. Ratified: the standing-bound soundness core ("previous-cell buffers being freed cannot invalidate the monotone-watermark bound"), per-cell analysis domination, no-upgrade aggregation, the F1b-2/W1-3 import coupling (loud DID-NOT-RUN failure), protocol chain fail-closed. MINORs: source-naming wording too absolute (downgrade-only conservatism); F1-5 stale string. Closed as rounds F9c (production) + F8b (harness), split along file ownership.

## Verdict: REQUEST-REVISION

1. **MAJOR — the attribution window does not cover the whole claimed cell requirement.** The window opens before load/build at [pos_rollout_fit_probe.py:2848](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2848), and optimizer initialization is included at [pos_rollout_update.py:480](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:480). Previous-cell buffers being freed cannot invalidate the monotone-watermark bound. However, the closing read occurs at [pos_rollout_fit_probe.py:2882](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2882), before evaluation and checkpoint work at [pos_rollout_fit_probe.py:2888](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2888) and [pos_rollout_fit_probe.py:2894](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2894). Therefore `W_after` is not guaranteed to bound those phases. A 95%-capacity evaluation could succeed once without a reservation failure while the cell authorizes on an earlier sub-90% mark.

   Either close the watermark after evaluation/checkpoint, or explicitly narrow `R` and the authorization contract to update-step steady state and state that evaluation/checkpoint receive no 10% headroom guarantee. The current documentation incorrectly says the peak is read after those operations at [pos_rollout_fit_probe.py:2780](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2780).

2. **MAJOR — F8’s reachability defense is not part of the executable battery.** The worklog/README says all nine honest controls were “run separately” at [README.md:73](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/README.md:73), but the battery invokes only the attacks at [reviewer_attacks.py:3119](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:3119). For example, T5a-3 contains only the TEST input at [reviewer_attacks.py:237](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:237), and T5b-3 only the forged derangement at [reviewer_attacks.py:552](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:552). An always-refusing production regression would still produce both green lines. Given the four prior false-verdict incidents, each honest control needs to execute in the recurring battery and fail the battery if it refuses.

3. **MAJOR — the load-bearing ladder order is bypassable through the public `cells=` seam.** Default CLI execution is correctly ordered, and exclusions/adoption preserve that order. But [pos_rollout_fit_probe.py:3043](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:3043) accepts an explicit sequence verbatim. Thus `cells=[rollout-mb8, one_step-mb8]` executes the poisoning cell first. The restart test at [test_pos_rollout_runtime_peak.py:718](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_runtime_peak.py:718) merely filters an already ordered list and does not exercise production with reversed cells or with mb=8 adopted while smaller cells remain unbanked. Apply the declared rank to every requested sequence or reject unordered production overrides.

Additional adjudications:

- Analysis domination is correctly per-cell; missing/zero analysis plus no rise fails closed.
- `classify_peak` cannot label an analysis-winning maximum as runtime. Mixed-trial aggregation can conservatively do the reverse—label a runtime numeric maximum as analysis at [pos_rollout_fit_probe.py:1282](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1282)—so the absolute “source names origin” wording should be narrowed, but it cannot upgrade authorization.
- Weakest attribution does not upgrade raised/standing mixtures.
- F1b-2/W1-3’s private test-helper imports are acceptable because failure is loudly DID-NOT-RUN.
- F1-5’s “reached the real model load” text at [reviewer_attacks.py:1803](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:1803) is stale MINOR wording.
- v6/v2 protocol checks and the training consumer’s `load_authorization → assert_cell_authorized` chain fail closed.

`git diff --check` is clean and the supplied log hashes match the worklog. I could not independently rerun the battery because this review sandbox provides no writable temporary directory; that limitation is not the basis of the verdict.
## Verdict: REQUEST-REVISION

1. **MAJOR — the attribution window does not cover the whole claimed cell requirement.** The window opens before load/build at [pos_rollout_fit_probe.py:2848](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2848), and optimizer initialization is included at [pos_rollout_update.py:480](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_update.py:480). Previous-cell buffers being freed cannot invalidate the monotone-watermark bound. However, the closing read occurs at [pos_rollout_fit_probe.py:2882](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2882), before evaluation and checkpoint work at [pos_rollout_fit_probe.py:2888](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2888) and [pos_rollout_fit_probe.py:2894](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2894). Therefore `W_after` is not guaranteed to bound those phases. A 95%-capacity evaluation could succeed once without a reservation failure while the cell authorizes on an earlier sub-90% mark.

   Either close the watermark after evaluation/checkpoint, or explicitly narrow `R` and the authorization contract to update-step steady state and state that evaluation/checkpoint receive no 10% headroom guarantee. The current documentation incorrectly says the peak is read after those operations at [pos_rollout_fit_probe.py:2780](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2780).

2. **MAJOR — F8’s reachability defense is not part of the executable battery.** The worklog/README says all nine honest controls were “run separately” at [README.md:73](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/README.md:73), but the battery invokes only the attacks at [reviewer_attacks.py:3119](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:3119). For example, T5a-3 contains only the TEST input at [reviewer_attacks.py:237](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:237), and T5b-3 only the forged derangement at [reviewer_attacks.py:552](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:552). An always-refusing production regression would still produce both green lines. Given the four prior false-verdict incidents, each honest control needs to execute in the recurring battery and fail the battery if it refuses.

3. **MAJOR — the load-bearing ladder order is bypassable through the public `cells=` seam.** Default CLI execution is correctly ordered, and exclusions/adoption preserve that order. But [pos_rollout_fit_probe.py:3043](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:3043) accepts an explicit sequence verbatim. Thus `cells=[rollout-mb8, one_step-mb8]` executes the poisoning cell first. The restart test at [test_pos_rollout_runtime_peak.py:718](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_runtime_peak.py:718) merely filters an already ordered list and does not exercise production with reversed cells or with mb=8 adopted while smaller cells remain unbanked. Apply the declared rank to every requested sequence or reject unordered production overrides.

Additional adjudications:

- Analysis domination is correctly per-cell; missing/zero analysis plus no rise fails closed.
- `classify_peak` cannot label an analysis-winning maximum as runtime. Mixed-trial aggregation can conservatively do the reverse—label a runtime numeric maximum as analysis at [pos_rollout_fit_probe.py:1282](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1282)—so the absolute “source names origin” wording should be narrowed, but it cannot upgrade authorization.
- Weakest attribution does not upgrade raised/standing mixtures.
- F1b-2/W1-3’s private test-helper imports are acceptable because failure is loudly DID-NOT-RUN.
- F1-5’s “reached the real model load” text at [reviewer_attacks.py:1803](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:1803) is stale MINOR wording.
- v6/v2 protocol checks and the training consumer’s `load_authorization → assert_cell_authorized` chain fail closed.

`git diff --check` is clean and the supplied log hashes match the worklog. I could not independently rerun the battery because this review sandbox provides no writable temporary directory; that limitation is not the basis of the verdict.

---

## Strengthening record — round F8b (harness findings only; F9c owns the production ones)

Addressed by the Coder for the two findings that fall inside the harness file. The three production
findings (runtime-watermark window, ladder-order `cells=` seam, source-naming wording) belong to
round **F9c** and are deliberately untouched here — `src/` was off limits for this round and all four
evaluator/gates modules are sha-verified identical to `HEAD`.

**MAJOR 2 — "F8's honest-input reachability controls are not part of the executable battery" — FIXED.**
Accepted without reservation; the finding is correct and the framing ("an always-refusing production
regression would still print nine green lines") is the right one. The controls now run **inside** the
recurring battery.

Of the two shapes offered, the **companion-entry** shape was chosen. Reason: a failing control is not
"the attack succeeded" — nothing got through, production has stopped accepting *legitimate* work,
which is a different defect with a different fix. Folding it into `SUCCEEDED` would reproduce exactly
the defect this campaign has paid for twice (`F5-5`, `T5a-2`): a probe whose verdict word contradicts
what it observed. So controls speak `CONTROL-PASSED` / `CONTROL-REFUSED` through a `_control`
reporter, `_summarize` counts them on a second SUMMARY line, and the runner exits non-zero on
`CONTROL-REFUSED` or an unparsed control. `_report` itself is **untouched** and remains byte-identical
to its F7d original. `_control` inherits F7d's discriminator verbatim, so a control cannot go silent
the way the nine probes did — and the control mechanism's own failure modes were mutation-tested
(raising control → `CONTROL-REFUSED (DID NOT RUN)`; legitimate case refused → `CONTROL-REFUSED`;
attack vocabulary → `UNPARSED`; `_summarize()` returned `False` in all three).

**On "extend cheaply to other families, but do not manufacture":** eleven controls total — the nine
revived probes plus the sigma grid and anchor reproduction, both of which were already trivially in
reach. The **T7/P3/F5/F6/F7** authorization and publication families are deliberately left
**attack-only**: their legitimate case is a full multi-phase publish/adopt cycle against the
in-memory bucket, which is a fixture with its own failure modes rather than a cheap witness, and
several already assert a positive outcome internally (`F5-6` requires two cells banked and
re-loadable, `F7-1` requires the launch authorized). The source-shape probes (`G3-13`, `W2-1`,
`W2-2`) assert presence rather than refusal, so a control would restate the probe. This inventory is
written into the harness above the controls, not only here.

**The finding paid for itself on its first execution.** Writing the anchor family's control exposed
that `_rows` never set `grid_sha256`. `summarize_samples` checks the grid before the horizon and long
before `reproduce_anchor` sees a name, so `G3-1 foreign names`, `G3-2 wrong order`, `G3-3 foreign
checkpoint` and `G3-4 short rollout` were all being refused with `grids ['']` — four probes green,
none testing the rule in its own name. All four now refuse on their own rule. That is the fourth
caution (`F5-5`) in four further places, and it was invisible to re-reading the attacks.

**MINOR — F1-5's stale "reached the real model load" — FIXED, and fixed at the cause.** The real
refusal post-F9 is `this backend reports no memory statistics for ['cpu:0']`. Rather than substitute
a new hardcoded description of where production stops — a second thing to keep in step with
production, which is precisely how the string went stale — the probe now **quotes production's own
refusal**, so it cannot go stale again.

**Not addressed here, by ownership:** MAJOR 1 (runtime-watermark window), MAJOR 3 (ladder order
bypassable via `cells=`), MINOR (source-naming wording). All three are `src/` and belong to F9c.

**Validation:** battery ×2, both `92 probes — 91 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED` and
`11 honest controls — 11 CONTROL-PASSED, 0 CONTROL-REFUSED, 0 UNPARSED`, exit 0, identical
verdict-word sequences → `harness/attacks_f8b_20260814.log`, sha256
`a178dfd73afe0e2ca6ae21a7952beccde2be509d94e77c08a53dc0928af0ddb7`. Nothing committed.


---

# F9c+F8b final verification (appended 2026-08-14)

Verdict: **APPROVE — "No blocking findings. Proceed with ceremony and M1-8."** The reviewer executed the eval-window scenario directly (30.5 GiB/95.3% -> refused on headroom), ruled _program_bytes measurement-instrumentation outside the contract boundary (correctly), verified order_cells on every path, the 11 controls + mutation-tested control mechanism, the G3 repairs, and both provenance directions.

## APPROVE

No blocking findings. Proceed with ceremony and M1-8.

- Window order is build/load → steps → eval → checkpoint → watermark read → `_program_bytes`. My direct fake-device run recorded 30.5 GiB/32 GiB (95.3125%) and refused on `headroom`.
- `_program_bytes` is measurement-only analysis instrumentation, not deployed cell work; excluding its recompilation allocation is the correct contract boundary.
- Every multi-cell path reaches `order_cells`; reversed requests sort and log, unknown ladder-order pairs refuse, and adoption preserves the restricted declared order.
- Harness has 11 controls. `_report` is byte-identical to HEAD/F7d. All three control-mechanism mutations made `_summarize()` fail.
- Direct production-path spot check: the honest anchor reproduced; G3-1 refused specifically on foreign sample names.
- Source provenance behaves in both directions: exact locally, conservative downgrade during mixed aggregation, never an authorization upgrade. F1-5 quotes the production refusal.
- `git diff --check` and AST parsing passed. Five focused no-temp tests passed. Supplied F9c/F8b hashes and summaries match the artifacts.

The full 2308-test suite and complete battery could not be independently rerun because this read-only sandbox has no writable temporary directory.
