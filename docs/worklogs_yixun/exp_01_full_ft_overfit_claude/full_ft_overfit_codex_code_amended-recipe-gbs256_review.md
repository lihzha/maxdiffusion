# Code review: exp_01 full_ft_overfit — round amended-recipe-gbs256
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-19

## Context loaded

- Query 7 approves GBS 256 × 20,000 steps with cohort evaluations at 5k/10k/15k/20k.
- The 02:40 worklog records the 36.92 MB HBM overflow at per-device 8 from FSDP collective buffers.
- The focused pytest suite could not collect because the read-only sandbox provides no writable temporary directory; shell syntax and no-write launcher executions passed.

## Adjudication: RUN_NAME

CHANGE-ORDERED — `RUN_NAME` is a durable W&B and GCS namespace, so `gbs512` materially mislabels the GBS-256 experiment; separate env/log/config provenance does not repair that indexing surface. Move its construction after the batch-default collapse and interpolate `${GLOBAL_BATCH_SIZE_TO_TRAIN_ON}`, preserving adapter `gbs512` names while producing full-FT `gbs256`.

## Verdict

APPROVE-WITH-CHANGES — The amended 4.0/256/256/20000 recipe is coherent, preserves the 5.12M-sample budget, and retains every evaluation checkpoint. The default-variable restructure correctly preserves set-vs-unset semantics, and normalized adapter submissions are byte-identical to HEAD.

## Findings

- **Launch-blocking provenance:** [launch_wan_train.sh](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit/bash_scripts/launch_wan_train.sh:103) hard-codes `gbs512` in every non-smoke full-FT run name despite submitting 4/256/256. Relocate that assignment after the collapse at line 127, substitute `gbs${GLOBAL_BATCH_SIZE_TO_TRAIN_ON}`, and pin `gbs256` in the full-FT launcher test.

---

# Strengthening record (Coder, cycle 8 — 2026-07-19)

- **CHANGE-ORDERED (RUN_NAME gbs512 mislabel) — FIXED.** RUN_NAME construction relocated to after the env-respecting batch collapse, interpolating `gbs${GLOBAL_BATCH_SIZE_TO_TRAIN_ON}`: full_ft non-smoke names now `…-gbs256-…` (pinned; `gbs512` absent), adapter arms byte-identical `…-gbs512-…` (pinned), SMOKE template still overwrites wholesale (pinned). Mutant (literal-gbs512 revert): full_ft golden red; side_adapter, pre_context, and full_ft-smoke goldens green — exactly the ordered detection. Suite 100/100; launcher sha-verified restored.
- Bonus honesty property: an explicit GBS env override now surfaces truthfully in the name (e.g. `gbs8`) — names never lie about the resolved batch.

**Cycle 8 closed** (write → review APPROVE-WITH-CHANGES → strengthen FIXED). Amended recipe committed: yml 4.0/256/256/20000; launcher full_ft arm 20000 + arm-default 4/256/256 + honest RUN_NAME.
