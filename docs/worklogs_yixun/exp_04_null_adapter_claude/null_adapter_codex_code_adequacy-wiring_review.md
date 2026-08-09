# null_adapter — Codex code review: fix round `adequacy-wiring` (issue #15)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: **APPROVE first-pass, zero findings** — the three-place mapping, the end-to-end path claim, the mistyped-expansion closure and the scope discipline all independently verified.

## Reviewer output (verbatim)

57,921
## Findings

No findings.

- Verified the three-place mapping: unset-safe default before use, logged value, and correctly spelled override inside the actual Python argument list.
- Verified the end-to-end path: YAML default → `config.null_adequacy_uri` → `load_adoption`; launcher wiring was the sole missing leg.
- The negative test’s exact invocation assertion plus uniqueness regex catches missing, displaced, and mistyped expansions.
- Scope is disciplined: only three launcher additions; issue #12’s xtrace behavior is untouched. `bash -n` and `git diff --check` pass.
- Requested suite reached 156 passed, then one unrelated `tmp_path` setup error because this review sandbox cannot write `/tmp`; the three affected launcher tests pass separately (3/3).

## Verdict

APPROVE — the surgical adequacy wiring is correct, complete, and has no launch-blocking defect.
