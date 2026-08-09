# rollout_adapter — Codex FINAL verification: W5b + THE M1 RULING

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. **W5b: APPROVE. THE RULING: READY-FOR-M1.** No new defect or launch-blocking verification gap. Production digests re-verified against the ruled-clean snapshot; T01 re-executed in memory (the previously-missed rewiring now killed); the scorer grounding independently MEASURED (45 input leaves, all replicated, the pruned draw leaf ruled legitimate); and the decisive experiment — the reviewer INJECTED a non-replicated scorer sharding into both compiled views: semantic side-to-side equality stayed TRUE while both absolute checks went FALSE, proving the new gate catches exactly what the old one missed. The W4 launch-caveat list stands unchanged.

## Reviewer output (verbatim)

191,413
## 1. Findings

No new defect or launch-blocking verification gap found.

The production SHA-256 digests exactly match the ruled-clean snapshot:

- `pos_rollout_update.py`: `e6d0afdec6b5febc…`
- `pos_rollout_fit_probe.py`: `67eb06ea7b3693a5…`
- `wan_pos_rollout_trainer.py`: `894b867555226b53…`

The current oracle ran green. `git diff --check` and AST parsing also passed.

## 2. Verdicts on 1–4

1. **PASS.** All eight operand identities—including both batches—are pinned at [test_pos_rollout_trainer_wiring.py:1392](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1392), and the difference assertion reads the actual lowered operands. I re-executed T01 in memory: `operands_are_own=false` and `operands_differ_before_placement=false`, so the oracle kills the exact rewiring previously missed.

2. **PASS.** The compiled step/scorer string gates are gone. Scorer inputs are checked against independently constructed `NamedSharding(m, P())` using `is_equivalent_to`.

   I independently measured both the trainer’s batch-one host-input lowering and M1’s lowering: each compiled scorer had `39 + 3 + 3 = 45` input leaves, and every leaf was replicated. The fourth draw leaf is legitimately compiler-pruned. The grounding is accepted.

   I also injected the same non-replicated scorer sharding into both compiled views: semantic side-to-side equality stayed true, while both absolute checks became false. Thus the new absolute gate catches exactly what the old relative gate missed.

3. **PASS.** The behavioral check observed axis rules change from `()` outside to `(("batch", "data"),)` inside and restore afterward. The AST check recognized the current scope and rejected the removal mutant; no source-substring gate remains.

4. **PASS.** T02–T08 cover both ownership-collapse directions, draw collapse, symmetric scorer-input drift, placement regressions, private scorer restoration, and scope removal. The recorded battery is 8/8 with named killers and verified restoration. The in-repo harness reconciles to 80 reports, 80 `REFUSED`, zero `SUCCEEDED` or silent results.

## 3. THE RULING

**READY-FOR-M1.**

The W4 launch caveat list is unchanged and therefore not restated.

## 4. W5b

**APPROVE.**
