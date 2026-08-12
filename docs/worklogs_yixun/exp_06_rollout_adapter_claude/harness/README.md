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
- `attacks_f6_20260812.log` — **current**: round F6 `cell-exclusion`, **89 probes — 88 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED**. Adds `F6-1 quote an excluded cell`: a cell DECLARED unreachable must be as unconstructible as a refused one, and refused AS an exclusion rather than as one merely never measured.
- `attacks_f5d_20260812.log` — round F5d: **88 probes — 87 REFUSED, 1 DECLARED, 0 SUCCEEDED, 0 UNPARSED**. Added the `DECLARED` allowlist, the non-zero exit, and restored `P3-5` to actually executing.
- `attacks_f5c_20260812.log` — round F5c: **88 probes — 87/1/0/0**. Introduced the three-way accounting, `F5-8` (the in-boundary forgery, DECLARED), and split `F5-5` to the foreign-manifest case it actually tests. **Its `P3-5` line is a false refusal** — the probe could not run (see the fifth caution).
- `attacks_f5b_20260812.log` — round F5b: **87 probes, 87 refused**. **Superseded and overstated**: its `F5-5` tested a foreign-manifest artifact while reporting as though it covered forgery generally; the in-boundary forgery it did not test is adopted. Kept as record.
- `attacks_f5_20260812.log` — round F5: **86 probes, 86 refused**. **Superseded and partly WRONG**: its `F5-5` scored REFUSED on an attack that in fact succeeded. Kept because a log that overstated its coverage is part of the record.
- `attacks_after_w5b.log` — round W5b: **80 probes, 80 refused, 0 succeeded**. `W3-1` and `W4-1` are BEHAVIOURAL: they execute the placement contract and the scope, and observe shardings and rules, rather than matching source text. `W3-1`'s "the measurement enters the scope" half is an AST `with`-item check, which survives an equivalent refactor and fails a real removal.

## Reproducing

```
PYTHONPATH=src JAX_PLATFORMS=cpu python docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py
```

Every line is `<probe>: <VERDICT>: <why>`, and the run ends with a `SUMMARY:` counting each verdict
separately. **There are three verdicts, and the third was added in F5c because a single "N refused"
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

An `UNPARSED` line means a probe returned a verdict the summary could not classify, or reached for
`DECLARED` without being on the allowlist; the classifier is deliberately strict (`startswith`), so fix
the probe rather than loosening it. **The run exits non-zero** if any probe is SUCCEEDED or UNPARSED,
so the battery is a gate and not just a report.

## A caution the harness earned in W2b

`_launch` used to return an empty override map when a launcher exited before its entrypoint, and a
probe comparing two empty maps read as SUCCEEDED (`both arms write and restore None`) — the harness
lying about production. It now REFUSES to return when a launch produced nothing, and a probe that
deliberately expects a non-launch opts out with `_expect_failure`. When adding a probe that inspects
launcher argv, make sure it fails loudly if the launcher did not run.

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
