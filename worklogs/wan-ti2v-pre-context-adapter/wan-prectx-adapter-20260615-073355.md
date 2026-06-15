## 2026-06-15T07:33:55Z - pre-context action adapter implementation

Goal:
- Add a MaxDiffusion WAN TI2V adapter mode that mimics the `../Wan2.2` `pre_context` action-conditioning path and prepare it for v6e-64 DROID training.

Hypothesis:
- A context head that reads context-free first-block WAN features plus action tokens can produce T5-space context tokens for the frozen WAN forward, matching the non-circular `pre_context` design while keeping only adapter parameters trainable.

Change:
- Added `action_adapter_type` routing with existing `side_adapter` as the default.
- Added an NNX pre-context context head, context-free patch/time/first-self-attention feature extraction, and `wan_action_adapter_forward`.
- Routed train/eval/validation rollout through the generic adapter forward helper.
- Added config and shell wrapper keys for `ACTION_ADAPTER_TYPE`, `PRE_CONTEXT_TOKENS`, and `PRE_CONTEXT_HEADS`.

Version Control:
- agent_id: wan-prectx-adapter-20260615-073355
- worktree: /home/lzha/code/maxdiffusion-worktrees/wan-prectx-adapter-20260615-073355
- worklog: worklogs/wan-ti2v-pre-context-adapter/wan-prectx-adapter-20260615-073355.md
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- base_commit: 7ee701de743169e6888a77dac1f3d31d24e408e1
- implementation_commit: 60781c70405dece76409c1c38554b0ecfd43f5f1
- push/pull: pending
- changed_files: src/maxdiffusion/models/wan/side_adapter_wan.py; src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py; src/maxdiffusion/generate_wan_side_adapter.py; src/maxdiffusion/configs/base_wan_5b_side_adapter.yml; bash_scripts/train_wan_side_adapter.sh; bash_scripts/validate_wan_side_adapter.sh; bash_scripts/watch_wan_side_adapter_validation.sh; worklogs/wan-ti2v-pre-context-adapter/wan-prectx-adapter-20260615-073355.md
- remote_commit/status: pending TPU deployment

Command / Job:
- command: `python3 -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py src/maxdiffusion/generate_wan_side_adapter.py`
- command: `bash -n bash_scripts/train_wan_side_adapter.sh && bash -n bash_scripts/validate_wan_side_adapter.sh && bash -n bash_scripts/watch_wan_side_adapter_validation.sh`
- command: `git diff --check`
- job_id: n/a
- run_dir: n/a
- logs: n/a
- artifacts: n/a

Result:
- status: passed local syntax checks
- metrics/artifacts: Python bytecode compile passed; shell syntax checks passed; whitespace check passed.
- key evidence: Local `python3` lacks JAX, so module instantiation/runtime smoke must run after TPU setup.

Analysis:
- The new pre-context path ignores input text/null context during feature extraction, stops gradients through the first-block features, predicts context tokens from features/actions, then runs the frozen WAN transformer with predicted context. The existing side-adapter default is preserved.

Next:
- Commit and push the implementation, run TPU setup/import smoke from the exact commit, then launch v6e-64 training with `ACTION_ADAPTER_TYPE=pre_context`.
