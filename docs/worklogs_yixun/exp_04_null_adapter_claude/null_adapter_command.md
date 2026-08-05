# null_adapter_command.md — exact reproduction commands (SOP artifact 7)

## J0-1 — cohort-manifest build (2026-08-05T04:50Z, **FAILED — INFRASTRUCTURE**: TF/ADC reauth `invalid_rapt`; fail-closed, nothing written; see worklog. Re-run pending ADC refresh.)

- Commit: `7199feb99514d5c4e460e84629b133566f6624d7` (branch `claude-exp_04_null_adapter-20260803`, clean worktree)
- Host: local macOS (darwin), scratchpad venv python 3.11 (tensorflow 2.21, numpy), gsutil authenticated (yh4742@princeton.edu)
- Log: `docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_2026-08-05_04:48:27.log`

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter && \
PYTHONPATH=src <venv>/bin/python <scratchpad>/j0_driver.py 2>&1 | tee docs/worklogs_yixun/exp_04_null_adapter_claude/null_adapter_2026-08-05_04:48:27.log
# where j0_driver.py (archived alongside this file as j0_driver.py after the run) calls:
# build_j0_manifests('gs://v6_east1d/datasets/droid_wan_side_adapter/val',
#                    'gs://v6_east1d/datasets/droid_wan_side_adapter/train',
#                    'docs/worklogs_yixun/exp_04_null_adapter_claude/j0_manifests',
#                    builder_sha='7199feb99514d5c4e460e84629b133566f6624d7')
# <venv> = /private/tmp/claude-501/-Users-yixunhu-Home-maxdiffusion/800fed95-7c3f-418d-b779-9914ed8480b4/scratchpad/venv
```

## J0-2 — cohort-manifest build re-run (2026-08-05_14:59:41Z, LAUNCHED at 2026-08-05_15:27:35Z — both credentials verified live (gsutil listing + Generation stat; ADC refreshed earlier). Same driver as J0-1; log: null_adapter_2026-08-05_15:27:35.log)

- Commit: `7199feb99514d5c4e460e84629b133566f6624d7` (J0 runs the committed builder; working tree carries only uncommitted R10 files not touched by J0)
- Same command as J0-1; log: `null_adapter_2026-08-05_14:59:41.log`
