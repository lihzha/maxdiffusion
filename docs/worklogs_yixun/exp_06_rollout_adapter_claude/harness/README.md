# exp_06 adversarial harness — the review package's executable half

**Why this folder exists.** The final decision review recorded an EVIDENCE GAP: the harness and its
logs lived only in the session scratchpad, which the review sandbox cannot see, so the "75 probes, all
refused" claim was unverifiable and the reviewer could not confirm that no probe was silently
measuring nothing. Standing rule from W3 on: the harness source and the final all-probes log are
review-package artifacts and are kept current in this folder.

## Files

- `reviewer_attacks.py` — every probe the campaign has accumulated, each one an attack that must be
  REFUSED. Rounds are additive: a probe is never removed, and the final run of a round executes all
  of them. Run from the repository root with `PYTHONPATH=src`.
- `attacks_f10e_20260816.log` — **current**: round F10e `analysis-components`, **106 probes — 105 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED** *and* **18 honest controls — 18 CONTROL-PASSED, 0 CONTROL-REFUSED**, exit 0. The last `or 0` in the module: `_program_bytes` summed the compiled analysis's components through `_exact_count(value or 0, ...)`, so `False` and `""` became a legal-looking zero and a NEGATIVE component was accepted and SUBTRACTED from the genuine ones — the reviewer's `(100, -90, 0, 0)` summed to a 10-byte bound against a 5-byte watermark on a 100-byte device and the cell AUTHORIZED. `F10e-1 shrink the bound` executes all five shapes end to end; `ctrl a real analysis sums` holds the other side (zero components included), because a fix that discarded good analyses too would re-create the M1-6 outage under a new reason. **The round's harness lesson is `F10d-2`:** the reviewer noticed it computed the artifact's derived digests from the RAW `8.5`, so a regression to bare `int()` would have rebuilt `8`, mismatched three internal digests and left the probe printing REFUSED while blind to the very truncation it exists to catch (the fourth caution, in its subtlest form yet). `_f5_edit` now derives from the value production WOULD PARSE — and the fix was verified by simulating the regression: with a truncating reader restored, `F10d-2` reports SUCCEEDED.
- `attacks_f10d_20260816.log` — round F10d `coercion-invariant`, **105 probes — 104 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED** *and* **17 honest controls — 17 CONTROL-PASSED, 0 CONTROL-REFUSED**, exit 0. Three rounds of patching the truncation class one field at a time (bytes → the deserialization boundary → this) ended with the invariant written down in the module instead: every number crossing a payload, config or device boundary — or defining an identity, a binding, a count or evidence — goes through one of four parsers, in both directions, and the module names the three situations where a bare `int()`/`float()` may still appear. `F10d-1` executes the reviewer's four counterexamples in one probe (`FitCell("rollout", 8.5, 2.5)` published and loaded as cell 8/2; `device_count: 8.5` binding to an eight-device context; `Fraction`/`Decimal` halves projecting as whole cadences; `trial_count` of `1.9`/`"1"`/`True`), `F10d-2` banks the fractional binding and requires re-measurement. `ctrl ordinary numbers parse` is the control an exactness sweep most needs — a too-strict parser fails silently, in production, at 3.5 hours in, and every attack above would still be green — so it exercises the deployed ladder, a payload round trip, the binding digest, a real projection and a real adoption. Writing it caught the harness's own forger: `_f5_edit` re-synced through `ProbeContext.from_payload`, which now refuses the very payload the attack needs to write, and the probe scored `THE PROBE DID NOT RUN` until the resync was computed from the payload (which is all a real forger has anyway).
- `attacks_f10c_20260816.log` — round F10c `exact-counts-and-bank-quarantine`, **103 probes — 102 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED** *and* **16 honest controls — 16 CONTROL-PASSED, 0 CONTROL-REFUSED**, exit 0. The F10b review's two executed residuals. `F10c-1 poison the bank forever` is the one worth reading: F10b's refusal to publish a table whose trials disagree was correct, and the reviewer then put that artifact in the CACHE the retry reads — two consecutive attempts adopted it and died at publication, which is issue #10's permanent wedge in its purest form. The table-wide raise stays and the artifact is now quarantined at adoption (and refused at banking), so the probe asserts the full arc: re-measured once, then the NEXT attempt adopts the clean cell the retry banked. `F10c-2 bank a malformed count` banks digest-valid artifacts carrying `peak_bytes: 9.9`, `true`, `reservation_failures: 0.9` and a digit string — all of which used to be `int()`-coerced at the deserialization boundary, before any check saw them, and travelled load → adopt → republish → gate. `ctrl consistent bank adopts` is the control the quarantine needed: if the new validation were wrong, every restart would silently re-measure the whole ladder and F5's reason for existing would be gone behind two green refusals.
- `attacks_f10b_20260816.log` — round F10b `authorization-evidence-fixes`, **101 probes — 100 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED** *and* **15 honest controls — 15 CONTROL-PASSED, 0 CONTROL-REFUSED**, exit 0. Codex's F10 review found three fail-closed gaps IN the new gate and **executed all of them through publication, reload and `assert_cell_authorized`** — the battery had not, which is the lesson of the round: F10's own probes attacked the single-record space and never the repeated-trial space, and one F10 *test* actively pinned the missing-watermark authorization as correct. Five new probes: `F10b-1` a bounded cell with no watermark at all (the cross-check is retained, so absent is not a pass), `F10b-2` four shapes of a friendly trial cancelling a contradicted one, `F10b-3` two different bounds for one executable, `F10b-4` fractional and boolean byte counts (`9.9` of a 10-byte device parsed as `9/10` and authorized), `F10b-5` a cell truly over 90% whose float division lands exactly on the boundary. Two controls guard the other direction, because both fixes could have moved a line instead of sharpening it: `ctrl two agreeing trials` (**the ladder measures every cell twice** — if the per-trial rules over-refused, M1-10 would authorize nothing, which is the outage F10 existed to end) and `ctrl exactly 90% authorizes`. `P3-8 contradictory duplicates` was updated to accept either refusal path (its two trials also disagree about the analysis, which production now refuses at aggregation) rather than pinning the mechanism.
- `attacks_f10_20260816.log` — round F10 `authorization-evidence`, **96 probes — 95 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED** *and* **13 honest controls — 13 CONTROL-PASSED, 0 CONTROL-REFUSED**, exit 0. Yixun's Option A (plan v2.9 §4-P1) makes the compiled memory analysis the authorization bound and the runtime watermark a recorded cross-check that can only refuse. Four new probes — `F10-1` a mark above the claimed bound, `F10-2` no bound at all, `F10-3` a reported peak above a quiet bound, `F10-4` a v6 table put in front of every v7 reader (loader, training gate, republication) — with two controls: `ctrl mark AT the bound` (equality is not an excess; without it the cross-check could be off by one in the safe-looking direction and every refusal above would still be green) and `ctrl analysis-bounded cell` (**the shape M1-9 actually measured must AUTHORIZE** — the round exists because the rule it replaced authorized zero of twelve measured cells, so a battery that only showed F10 refusing things would witness nothing). `W1-1 authorize on a floor` was RE-EXPRESSED rather than deleted (the seventh caution): its premise — an analysis can never authorize — is exactly what the contract inverted, so it now attacks the same cell one layer down, a record whose watermark and analysis contradict each other. The `_fit` fixture moved to the F10-authorizing shape in the same commit; leaving it would have turned probes into false refusals against a production refusing their SETUP rather than their attack, which is the fifth caution and which duly fired on `F1-3`, `F6-2` and `F7-1` the first time this battery was run against the new code.
- `attacks_f8b_20260814.log` — round F8b `controls-in-battery`, **92 probes — 91 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED** *and* **11 honest controls — 11 CONTROL-PASSED, 0 CONTROL-REFUSED**, exit 0. The controls are the headline: F8's reachability checks were run by hand beside the battery, so a production regression that refused everything would have left nine green REFUSED lines standing. They now run in the same battery, with their own verdict words and their own SUMMARY line. Writing them immediately caught four probes (`G3-1`..`G3-4`) that had been refusing on a missing grid digest instead of on their own rules.
- `attacks_f8_20260813.log` — round F8 `probe-revival`, **92 probes — 91 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED**, exit 0. The nine probes F7d exposed as dead were revived one at a time against the real current APIs. **Every one of them refused: the four-day hole was in COVERAGE, not in production's rules.** This is the first honestly green battery since the universal guard was installed — and the first whose number is worth quoting, because it is the first that is not counting probes that never ran.
- `attacks_f7d_20260813.log` — round F7d: **92 probes — 82 REFUSED, 1 DECLARED, 9 SUCCEEDED, 0 UNPARSED**, exit 1. **Honestly red, and deliberately kept.** Turning on the universal guard took the battery from a fake 92/0 to this; the nine `SUCCEEDED: THE PROBE DID NOT RUN` lines are the inventory F8 worked from. Read it beside the F8 log: the pair is the whole lesson.
- `attacks_f6_20260812.log` — round F6 `cell-exclusion`, **89 probes — 88 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED**. Adds `F6-1 quote an excluded cell`: a cell DECLARED unreachable must be as unconstructible as a refused one, and refused AS an exclusion rather than as one merely never measured. **Its headline overstates coverage** — like every log from `attacks_after_w5b` through `attacks_f7c`, it counted probes that were not executing.
- `attacks_f5d_20260812.log` — round F5d: **88 probes — 87 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED**. Added the `DECLARED` allowlist, the non-zero exit, and restored `P3-5` to actually executing.
- `attacks_f5c_20260812.log` — round F5c: **88 probes — 87/1/0/0**. Introduced the three-way accounting, `F5-8` (the in-boundary forgery, DECLARED), and split `F5-5` to the foreign-manifest case it actually tests. **Its `P3-5` line is a false refusal** — the probe could not run (see the fifth caution).
- `attacks_f5b_20260812.log` — round F5b: **87 probes, 87 refused**. **Superseded and overstated**: its `F5-5` tested a foreign-manifest artifact while reporting as though it covered forgery generally; the in-boundary forgery it did not test is adopted. Kept as record.
- `attacks_f5_20260812.log` — round F5: **86 probes, 86 refused**. **Superseded and partly WRONG**: its `F5-5` scored REFUSED on an attack that in fact succeeded. Kept because a log that overstated its coverage is part of the record.
- `attacks_after_w5b.log` — round W5b: **80 probes, 80 refused, 0 succeeded**. `W3-1` and `W4-1` are BEHAVIOURAL: they execute the placement contract and the scope, and observe shardings and rules, rather than matching source text. `W3-1`'s "the measurement enters the scope" half is an AST `with`-item check, which survives an equivalent refactor and fails a real removal.

## Reproducing

```
PYTHONPATH=src JAX_PLATFORMS=cpu python docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py
```

Every line is `<probe>: <VERDICT>: <why>`, and the run ends with **two** `SUMMARY:` lines — one for
the attacks, one for the honest controls — each counting its verdicts separately. The battery runs
**two kinds of thing**, and summing them would be the very category error the verdict split exists
to prevent.

**The attacks have three verdicts, and the third was added in F5c because a single "N refused"
headline hid a false refusal for two rounds:**

- **REFUSED** — production stopped the attack. What you want to see.
- **DECLARED** — *the attack SUCCEEDS, by design, inside a trust boundary this campaign has explicitly
  accepted and written down.* It is **not** a refusal and must never be counted as one. **The probes
  permitted to say it are an explicit allowlist** (`_MAY_DECLARE`, F5d): today exactly one, `F5-8` — an
  authorized bucket writer who can read one current artifact fabricating a fit cell. A DECLARED from
  any other probe is a **harness failure**, printed as one and counted UNPARSED, because an accepted
  residual is a decision somebody made and wrote down, not a word a probe can reach for. Adding an
  entry to that allowlist is a decision that has to appear in the diff a reviewer reads. The banked-cell artifacts are integrity-checked and program-bound but **not authenticated**;
  the anchor is the bucket ACL, and signing is escalated to Yixun as a policy decision. A DECLARED
  probe is a standing reminder of a known hole and a tripwire on the claim: **if the boundary ever
  changes — if a publication authority is added — the probe flips and forces the module docstring, the
  worklog and the probe text to be updated together.** `F5-8` says so in its own return value.
- **SUCCEEDED** — a defect in production, not in the probe.

**The honest controls have two verdicts** (F8b — see the eighth caution, which is the standing rule
here). A control points production at the LEGITIMATE case and requires it to be accepted, because a
refusal proves nothing on its own:

- **CONTROL-PASSED** — production still accepts the legitimate case, so the refusals this control
  witnesses are worth something.
- **CONTROL-REFUSED** — production has stopped accepting legitimate work. **This is a defect too, and
  a different one from SUCCEEDED**: nothing got through, but every refusal beside it is now
  unwitnessed. Also the verdict a control gets when it cannot run at all.

An `UNPARSED` line means a probe returned a verdict the summary could not classify, reached for
`DECLARED` without being on the allowlist, or (for a control) answered in the attacks' vocabulary;
the classifier is deliberately strict (`startswith`), so fix the probe rather than loosening it.
**The run exits non-zero** if any probe is SUCCEEDED or UNPARSED, or any control is CONTROL-REFUSED
or UNPARSED — so the battery is a gate and not just a report.

## A caution the harness earned in W2b

`_launch` used to return an empty override map when a launcher exited before its entrypoint, and a
probe comparing two empty maps read as SUCCEEDED (`both arms write and restore None`) — the harness
lying about production. It now REFUSES to return when a launch produced nothing, and a probe that
deliberately expects a non-launch opts out with `_expect_failure`. When adding a probe that inspects
launcher argv, make sure it fails loudly if the launcher did not run.

## The eighth caution, earned in F8b — a refusal proves nothing unless the same battery shows an acceptance

**This is the standing defense, and it resolves the seventh caution rather than sitting beside it.**
F8's rule was "every revival needs a reachability check". F8 obeyed it — and ran the nine checks *by
hand*, beside the battery, recording them in the worklog. The reviewer's objection is the one this
harness keeps re-learning: **unexecuted evidence is not evidence.** The recurring run invoked only
the attacks, so a production regression that refused everything would have gone on printing nine
green `REFUSED` lines, and the worklog paragraph asserting otherwise would have aged into a lie.

**The rule: a probe's honest control runs in the SAME battery as its attack, or the probe's refusal
does not count.** Concretely:

- Controls are invoked through `_control`, not `_report`, and speak a different vocabulary:
  **`CONTROL-PASSED` / `CONTROL-REFUSED`**. A control is *not* an attack — a failing control means
  production stopped accepting legitimate work, which is a different defect with a different fix,
  and folding it into `SUCCEEDED` would be the same category error the three-way split exists to
  stop. `_summarize` prints them on their own line and counts them separately.
- **The runner exits non-zero on `CONTROL-REFUSED` or an unparsed control**, so the battery cannot be
  green while its refusals are unwitnessed.
- `_control` inherits the F7d discriminator verbatim: a verdict is a RETURNED string, and anything
  escaping the body is the control's own failure. A control cannot go quiet the way the nine did.
- **Do not manufacture controls.** One is worth having only where the legitimate case is genuinely
  cheap to reach; a fabricated one is invented evidence. The section header above the controls in
  `reviewer_attacks.py` lists what is covered and states which families are deliberately
  attack-only (the T7/P3/F5/F6/F7 authorization and publication families, whose "legitimate case" is
  a whole multi-phase publish/adopt cycle, and the source-shape probes, where a control would merely
  restate the probe).

**It paid for itself on the first run.** Writing the anchor family's control exposed `_rows`
omitting `grid_sha256`: `summarize_samples` checks the grid before anything else, so `G3-1 foreign
names`, `G3-2 wrong order`, `G3-3 foreign checkpoint` and `G3-4 short rollout` had all been refused
with `grids ['']` — four probes green, and none of them testing the rule in its own name. That is
the fourth caution (`F5-5`, watching the wrong observable) in four more places, and no amount of
re-reading the attacks would have found it. Asking "does the honest case still pass?" found it
immediately.

## The seventh caution, earned in F7d/F8 — reviving a dead probe is not the same as making it green

F7d moved the guard into `_report` and the battery went from a reported 92/0 to **82 REFUSED /
1 DECLARED / 9 SUCCEEDED**. Nine probes had been dead since `76117df` (2026-08-09) and had been
counted as coverage in five consecutive green batteries. F8 revived them one at a time; all nine
refused, so **production's rules had been intact the whole time and the loss was pure coverage**.
Three rules came out of doing it:

1. **Re-express the attack's SPIRIT, never its dead letter.** A probe whose API moved is not repaired
   by making the call compile. `W1-3` read the source of a method W3 deleted; it is now a behavioural
   check that instruments the shared factory and builds M1's program at two dtypes. `F3a-5` lost the
   `velocity_for` seam F3c removed on purpose — the attack was re-expressed at
   `build_rollout_kernel`, where injection moved, rather than deleted as obsolete.
2. **Every revival needs a reachability check.** Point the probe at the legitimate case and confirm
   it does NOT refuse. A revived probe that refuses for a reason unrelated to its name is the F5b
   defect wearing a fresh timestamp — and the cheapest way to find one is to check that the honest
   input passes. All nine revivals in F8 got one, run separately and recorded.
3. **Prefer a scoped fake to `_fake_environment` when your probe runs early.** `_fake_environment`
   installs an in-memory `gs://` filesystem process-wide and never removes it, so a probe that calls
   it changes the ground twenty later probes stand on. F8's `_cohort_records` patches only the two
   seams a derangement needs and restores both.

**A dead probe is evidence of nothing in either direction.** That is why the nine had to be executed
rather than reasoned about, and why the honest red log is kept next to the green one.

## The sixth caution, earned in F7c — the fifth caution was not enough, and the rule had a hole

`P3-5` did it again. F5d repaired it, wrote the standing rule below (*grep the harness for call sites
in the same commit as any production signature change*), and guarded **one call**. F7 then changed
`publish_attempt`'s signature, the rule was not followed, and the F7b log shipped with:

```
P3-5  adopt incomplete/foreign:: REFUSED (TypeError): publish_attempt() missing 1 required
                                 keyword-only argument: 'binding_digest'
```

**Fourth false verdict of the campaign, second from this exact cause.** Two things were wrong. The
rule depended on a human remembering it at exactly the moment they were busy doing something else;
and the guard covered the one call that had broken last time, which is never the one that breaks
next.

**The sharpened rule: the did-not-run guard wraps the WHOLE probe body, not one call.**
F7c's version of this was `_must_execute`, a decorator applied to every probe that called a
production API whose signature had churned. **F7d deleted it**, because a hand-applied decorator is
another list to keep in step and it caught only `TypeError`: the guard now lives in `_report`, which
every probe is invoked through, and the discriminator is structural — *a verdict is a RETURNED
string; anything that escapes the probe body is the probe's own failure*. See the seventh caution.
The runner exits non-zero on it, so the battery cannot be green while a probe is silently absent.
The grep rule stands as well — a guard that turns a silent hole into a loud one is a safety net, not
a substitute for updating the call.

## A fifth caution, earned in F5d — a probe that CANNOT RUN scores as a refusal

`_report` turns any exception into a `REFUSED (...)` line. That is what makes the runner survive a
probe whose attack production has hardened out of existence — and it is also how a probe that can no
longer execute at all disappears into the pass column. F5c gave `select_resume_publication` a required
`context_digest` keyword and did not update `P3-5`'s call site; the `TypeError` was caught and the log
recorded:

```
P3-5  adopt incomplete/foreign:: REFUSED (TypeError): select_resume_publication() missing 1 required
                                 keyword-only argument: 'context_digest'
```

A standing attack had not executed for an entire round, and the battery counted it as coverage.

**The pattern: a probe that dies on a signature change scores as a refusal unless something
distinguishes *attack-refused* from *attack-never-ran*.** The exception type cannot make that
distinction — several genuine refusals in this battery ARE `TypeError`s, production declining a call
shape it does not have. So the distinction has to be made *in the probe*: where a probe calls a
production function whose signature could drift, catch that failure and report it as a **harness
failure** rather than a refusal, the way `P3-5` now does. When a production signature changes, grep
the harness for its call sites in the same commit.

**Three false verdicts in one week** — a stale fake (`_Gfile`), a probe watching the wrong observable
(`F5-5`), and now a probe that never ran (`P3-5`). All three were green. The battery's headline number
is the least trustworthy thing in the review package; read the reasons.

## A fourth caution, earned in F5b — a probe can watch the wrong thing and call it a refusal

`F5-5` was committed as `smuggle past the digest`. It rewrote a banked trial, recomputed both
digests, **watched the forged artifact be adopted**, and reported **REFUSED** — because the run-level
digest changed. That is *propagation* (adopted content lands inside the digest); the probe's name and
verdict claimed *legality* (adopted content is legitimate). Codex found it while reviewing F5: "the
committed attack nearly demonstrates this itself." The 86/0 that probe contributed to did not cover
the hunt it was named for.

It is now `F5-5 fabricate a cheap cell` and asserts what the reviewer demanded: an artifact carrying
favourable peaks and a correctly recomputed digest must cause **REMEASUREMENT**. Rewritten first, it
reported `SUCCEEDED: a fabricated cheap cell was adopted without being measured` — the blocker, live —
and went green only once the manifest binding landed.

**The rule this yields, alongside the `_Gfile` caution below.** The harness produced two false
REFUSALs in one week: one from a stale fake, one from a probe watching the wrong observable. When you
add a probe, write down *what the SUCCEEDED branch would have to observe*, and check that the probe
can actually observe it. A verdict is only as good as the assertion underneath it — and a probe whose
SUCCEEDED branch is unreachable is a probe that will never fire.

## A third caution, earned in F5 — the fakes go stale too

The first F5 run reported all six new probes as `REFUSED (AttributeError): '_Gfile' object has no
attribute 'rename'`. Six green-looking lines, and not one of them had executed an adoption: F5
publishes by staging bytes and renaming them onto the destination, and the harness's in-memory
`_Gfile` (installed by `_fake_environment`) knew only `exists` / `makedirs` / `GFile`. The fix was to
teach the fake `rename`, `listdir` and `remove`, and to make its `exists` prefix-aware the way a
bucket is — **not** to reword the probe. `_report` turns any exception into a REFUSED line, which is
what makes the runner robust and also what lets a stale fake masquerade as a refusal: **read the
REASON on a REFUSED line, not just the verdict.**

## A second caution, earned in W4

Probe `W3-1` reported SUCCEEDED after W4 refactored the replication behind `replicated_sharding` —
the probe was checking one SPELLING of the property, not the property. It was updated to name the
property, not deleted. **A probe that goes stale reports a defect it cannot see; treat every
SUCCEEDED as production-guilty until you have read the probe.**
