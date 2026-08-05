# pos_context — Codex code review: S10a `pos-launcher`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: REQUEST-REVISION — 1 BLOCKER (DEV adequacy adoption rejected by TRAINFIT-16 capacity; an S4-scope defect exposed by the K1-readiness walk) + 1 LOW (shim can mask heredoc/parser failures) + 1 MINOR (directed rename). Strengthen dispatched same cycle; record follows below the verbatim output.

## Reviewer output (verbatim)

1. Context loaded

HEAD is `2149833`; only the two stated files are untracked. Settled-file diff is empty. The launcher is executable, passes `bash -n`, and both embedded Python blocks parse successfully.

The requested pytest run could not start because this read-only review sandbox has no writable temporary directory. Read-only collection succeeded: 60 cases collected.

2. Findings

- **BLOCKER — K1 cannot apply the DEV adequacy result to the required TRAINFIT-16 capacity run.** [pos_context_modes.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/pos_context_modes.py:733) rejects the DEV-only probe when the capacity plan uses `trainfit16`; [run_wan_null_inversion.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/run_wan_null_inversion.py:771) also compares its DEV manifest hash against the active TRAINFIT hash. I reproduced the rejection directly. Smoke, DEV adequacy, and full DEV capacity work at the interface level, but K1 requires arms on DEV-64 plus TRAINFIT-16. Change adoption validation to require `ADEQUACY_COHORT`/DEV provenance and always validate against the DEV manifest digest, independent of the capacity cohort. Add a regression proving one DEV adequacy artifact is accepted by both DEV and TRAINFIT capacity while foreign slot/hash artifacts remain rejected.

- **LOW — the bash shim can mask real launcher failures.** [test_pos_launcher.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_launcher.py:54) makes every Python invocation succeed, so malformed heredoc Python, failed preflight imports, or pyconfig rejection could pass this suite. The current heredocs parse, every override exists in the actual checked-in YAML, and no present launcher defect was found. Strengthen with extracted-heredoc AST tests and one subprocess using the real config parser on the recorded argv.

- **MINOR — directed naming cleanup.** Rename [test_pos_launcher.py](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_05_pos_context/src/maxdiffusion/tests/worklogs_yixun/test_pos_launcher.py:1) to `test_pos_context_launcher.py`.

No environment/default route selects the null YAML or passes null-root keys: the positive config and slot are literals, `NULL_*` variables are ignored, and positive execution receives only `pos_*` roots. The `exp04` check is a narrow, case-sensitive lexical defense rather than general root identity protection.

3. Positions ratified/contested

1. **Contested in novelty, endorsed in substance.** Fake-Python argv capture under executed bash already existed in repository history at `d27d32f`, with hermetic PATH-shim precedent at `f4da3eb`. This implementation is broader, but the technique is not repository-new.
2. **Ratified:** opt-in, default-off watchdog is appropriate.
3. **Ratified:** no-colon defaults correctly make explicit empty mode/root values fail.
4. **Ratified with the lexical limitation above:** refusing literal `exp04` roots is useful defense-in-depth.
5. **Ratified:** perform the directed test-file rename.

4. Verdict

**REQUEST-REVISION — the launcher files are structurally sound, but K1 is not end-to-end launchable until DEV adequacy adoption works for the required TRAINFIT-16 capacity run.**
