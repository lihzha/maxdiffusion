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
- `attacks_after_w5.log` — the final all-probes run of round W5: **80 probes, 80 refused, 0 succeeded**, with `W3-1` and `W4-1` rewritten as BEHAVIOURAL probes (they execute the placement contract and observe shardings, rather than matching source text).

## Reproducing

```
PYTHONPATH=src JAX_PLATFORMS=cpu python docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py
```

Every line is `<probe>: REFUSED: <why>` or `<probe>: SUCCEEDED: <what got through>`. A SUCCEEDED line
is a defect in production, not in the probe.

## A caution the harness earned in W2b

`_launch` used to return an empty override map when a launcher exited before its entrypoint, and a
probe comparing two empty maps read as SUCCEEDED (`both arms write and restore None`) — the harness
lying about production. It now REFUSES to return when a launch produced nothing, and a probe that
deliberately expects a non-launch opts out with `_expect_failure`. When adding a probe that inspects
launcher argv, make sure it fails loudly if the launcher did not run.

## A second caution, earned in W4

Probe `W3-1` reported SUCCEEDED after W4 refactored the replication behind `replicated_sharding` —
the probe was checking one SPELLING of the property, not the property. It was updated to name the
property, not deleted. **A probe that goes stale reports a defect it cannot see; treat every
SUCCEEDED as production-guilty until you have read the probe.**
