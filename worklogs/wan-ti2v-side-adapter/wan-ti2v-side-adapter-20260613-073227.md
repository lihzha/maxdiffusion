## 2026-06-13T07:32:27Z - workspace setup

Goal:
- Isolate implementation work for Wan2.2 TI2V 5B side-adapter training in MaxDiffusion.

Hypothesis:
- Starting from origin/catherine-dev is the safest base because it already contains Wan2.2 TI2V 5B model, training, and checkpoint plumbing.

Change:
- Created a dedicated local worktree and branch.

Version Control:
- agent_id: wan-ti2v-side-adapter-20260613-073227
- worktree: /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227
- worklog: worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md
- branch: codex/wan-ti2v-side-adapter-20260613-073227
- base_commit: 2d6f8e0d54697661df33d3e1a32e7e0e9b994d97
- implementation_commit: 8743a7b7a6643a0d5c062e07d7240a4c7252ed3f
- push/pull: n/a
- changed_files: worklog only
- remote_commit/status: n/a

Command / Job:
- command: `git fetch origin catherine-dev && git worktree add -b codex/wan-ti2v-side-adapter-20260613-073227 /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227 origin/catherine-dev`
- job_id: n/a
- run_dir: n/a
- logs: n/a
- artifacts: n/a

Result:
- status: passed
- metrics/artifacts: Worktree created from origin/catherine-dev at 2d6f8e0d54697661df33d3e1a32e7e0e9b994d97.
- key evidence: `git status --short --branch` shows the owned branch tracking origin/catherine-dev.

Analysis:
- The current main checkout remains untouched on tenny-dev, avoiding conflicts with other users or jobs.

Next:
- Inspect existing Wan2.2 TI2V and AC_TI2V MaxDiffusion code paths, then implement the minimal side-adapter model/trainer surface.

## 2026-06-13T07:45:00Z - side adapter module

Goal:
- Port the Wan2.2 side-adapter model component into a standalone NNX module.

Hypothesis:
- Keeping the frozen WAN transformer outside the trainable adapter module makes adapter-only optimization explicit and avoids relying on path-based NNX parameter filters.

Change:
- Added `src/maxdiffusion/models/wan/side_adapter_wan.py` with action token encoder, zero-initialized side adapters, WAN forward wrapper, rollout sigma helpers, and first-frame pinning.
- The first correct implementation requires `scan_layers=False` so residual injection after exact layer indices is unambiguous.

Version Control:
- agent_id: wan-ti2v-side-adapter-20260613-073227
- worktree: /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227
- worklog: worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md
- branch: codex/wan-ti2v-side-adapter-20260613-073227
- base_commit: 2d6f8e0d54697661df33d3e1a32e7e0e9b994d97
- implementation_commit: 8743a7b7a6643a0d5c062e07d7240a4c7252ed3f
- push/pull: n/a
- changed_files: src/maxdiffusion/models/wan/side_adapter_wan.py, worklog
- remote_commit/status: n/a

Command / Job:
- command: `python3 -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py`
- job_id: n/a
- run_dir: n/a
- logs: n/a
- artifacts: n/a

Result:
- status: passed
- metrics/artifacts: Syntax check completed with no errors.
- key evidence: `py_compile` exit code 0.

Analysis:
- This validates only syntax. Runtime validation still needs an environment with Flax/JAX and a tiny synthetic adapter instantiation.

Next:
- Add trainer/config plumbing and then validate all new Python files together.

## 2026-06-13T08:02:00Z - trainer and config

Goal:
- Add MaxDiffusion training plumbing for frozen-backbone Wan2.2 TI2V 5B side-adapter training.

Hypothesis:
- A self-contained trainer with frozen transformer params stored outside `state.params` prevents accidental backbone optimization and avoids per-example text embeddings by computing one null T5 context at startup.

Change:
- Added `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`.
- Routed `model_type: SIDE_ADAPTER_TI2V` from `src/maxdiffusion/train_wan.py`.
- Added `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml` with 48-channel Wan2.2 2.2 latent geometry, v6 GCS defaults, side-adapter hyperparameters matching `../Wan2.2`, and `scan_layers: False`.

Version Control:
- agent_id: wan-ti2v-side-adapter-20260613-073227
- worktree: /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227
- worklog: worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md
- branch: codex/wan-ti2v-side-adapter-20260613-073227
- base_commit: 2d6f8e0d54697661df33d3e1a32e7e0e9b994d97
- implementation_commit: 8743a7b7a6643a0d5c062e07d7240a4c7252ed3f
- push/pull: n/a
- changed_files: src/maxdiffusion/models/wan/side_adapter_wan.py, src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py, src/maxdiffusion/train_wan.py, src/maxdiffusion/configs/base_wan_5b_side_adapter.yml, worklog
- remote_commit/status: n/a

Command / Job:
- command: `python3 -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py src/maxdiffusion/train_wan.py`
- command: `python3 - <<'PY' ... yaml.safe_load('src/maxdiffusion/configs/base_wan_5b_side_adapter.yml') ... PY`
- job_id: n/a
- run_dir: n/a
- logs: n/a
- artifacts: n/a

Result:
- status: passed
- metrics/artifacts: Python syntax checks passed; YAML parsed with 174 top-level keys.
- key evidence: Config reports `model_type=SIDE_ADAPTER_TI2V`, `scan_layers=False`, `latent_channels=48`, `latent_frames=9`, and v6 GCS paths.

Analysis:
- This still needs runtime validation on a TPU/JAX environment. Local project `uv run` currently fails dependency resolution due a torch marker issue, so runtime smoke should use `bash_scripts/setup.sh` on an interactive TPU as requested by the user.

Next:
- Add a bounded TFRecord converter that refuses unsafe output sizes and supports direct `gs://v6_east1d` output.

## 2026-06-13T07:49:56Z - converter, wrapper, and local validation

Goal:
- Add a storage-safe DROID latent-cache TFRecord converter and a TPU launch wrapper for staged smoke, batch-size probing, and final v6 training.

Hypothesis:
- The cached Wan2.2 DROID windows already contain exactly the tensors needed by the rollout loss, so TFRecords can store only `z_i0`, `z_video`, and `actions` plus metadata. Keeping W&B opt-in prevents first runtime smoke from failing on credentials before exercising the model path.

Change:
- Added `src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py`.
- Added `bash_scripts/train_wan_side_adapter.sh`.
- Updated the side-adapter trainer checkpoint directory creation to use `tf.io.gfile.makedirs()` for `gs://v6_east1d` output paths.
- Simplified `_shard_state` by removing an unnecessary pre-sharding tree conversion.
- Made W&B opt-in in the new side-adapter config/wrapper.

Version Control:
- agent_id: wan-ti2v-side-adapter-20260613-073227
- worktree: /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227
- worklog: worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md
- branch: codex/wan-ti2v-side-adapter-20260613-073227
- base_commit: 2d6f8e0d54697661df33d3e1a32e7e0e9b994d97
- implementation_commit: 8743a7b7a6643a0d5c062e07d7240a4c7252ed3f
- push/pull: n/a
- changed_files: src/maxdiffusion/models/wan/side_adapter_wan.py, src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py, src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py, src/maxdiffusion/configs/base_wan_5b_side_adapter.yml, src/maxdiffusion/train_wan.py, bash_scripts/train_wan_side_adapter.sh, worklog
- remote_commit/status: n/a

Command / Job:
- command: `python3 -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py src/maxdiffusion/train_wan.py`
- command: `python3 - <<'PY' ... yaml.safe_load('src/maxdiffusion/configs/base_wan_5b_side_adapter.yml') ... PY`
- command: `bash -n bash_scripts/train_wan_side_adapter.sh`
- job_id: n/a
- run_dir: n/a
- logs: n/a
- artifacts: n/a

Result:
- status: passed
- metrics/artifacts: Syntax checks passed, YAML side-adapter invariants passed, wrapper shell syntax passed.
- key evidence: Config invariants show `model_type=SIDE_ADAPTER_TI2V`, `scan_layers=False`, `latent_channels=48`, `side_adapter_sampling_steps=25`, and dataset paths under `gs://v6_east1d`.

Analysis:
- This is a local static validation only. Runtime validation still needs the requested TPU environment set up through `bash_scripts/setup.sh`.
- The converter is intentionally capped by default (`--max-output-gb=120`) and supports direct GCS writes or local staging with deletion after upload.

Next:
- Commit and push the exact implementation, then run remote smoke validation before any full conversion or v6 launch.

## 2026-06-13T07:56:00Z - v4 setup and sigma schedule correction

Goal:
- Validate the new side-adapter model path in a TPU environment before dataset conversion or final v6 launch.

Hypothesis:
- A tiny WAN/adapter forward on v4 is enough to catch NNX graph, mesh, per-token timestep, and shape errors without loading the full 5B checkpoint.

Change:
- Ran `bash_scripts/setup.sh MODE=stable DEVICE=tpu` on `v4-4-01-interactive` after checking out `origin/codex/wan-ti2v-side-adapter-20260613-073227`.
- Fixed `build_rollout_sigmas()` to match the PyTorch WAN reference schedule: `linspace(sigma_max, sigma_min, N+1)[:-1]`, then append terminal zero.
- Added explicit `flow_sigma_min: 0.0` and `flow_sigma_max: 1.0` to the side-adapter config, and passed those values into `FlaxFlowMatchScheduler`.

Version Control:
- agent_id: wan-ti2v-side-adapter-20260613-073227
- worktree: /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227
- worklog: worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md
- branch: codex/wan-ti2v-side-adapter-20260613-073227
- base_commit: 2d6f8e0d54697661df33d3e1a32e7e0e9b994d97
- implementation_commit: 8743a7b7a6643a0d5c062e07d7240a4c7252ed3f
- push/pull: branch already pushed before this correction; correction pending commit/push
- changed_files: src/maxdiffusion/models/wan/side_adapter_wan.py, src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py, src/maxdiffusion/configs/base_wan_5b_side_adapter.yml, worklog
- remote_commit/status: v4 checkout initially at 12e974edd63746ff936ac48bb0659cc21b7dc884

Command / Job:
- command: `gcloud alpha compute tpus tpu-vm ssh v4-4-01-interactive --project=mae-irom-lab-guided-data --zone=us-central2-b --worker=0 --command='... bash bash_scripts/setup.sh MODE=stable DEVICE=tpu'`
- command: `python3 -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py src/maxdiffusion/train_wan.py`
- command: `python3 - <<'PY' ... assert cfg['flow_sigma_min'] == 0.0 ... PY`
- job_id: n/a
- run_dir: /home/lzha/maxdiffusion on v4-4-01-interactive worker 0
- logs: terminal output in current Codex session
- artifacts: TPU venv at `/home/lzha/maxdiffusion/.venv`

Result:
- status: partial
- metrics/artifacts: v4 setup completed. Tiny forward smoke passed before the sigma correction; rerun after push is still pending.
- key evidence: Forward smoke printed `SMOKE_OK` with output shape `(1, 4, 3, 4, 4)`. The sigma correction changes the test schedule for `N=4, shift=5` from `[1.0, 0.9091, 0.7143, 0.0, 0.0]` to the WAN-reference convention.

Analysis:
- The initial failed smoke attempts were harness errors: first missing mesh context, then a context length not divisible by the test mesh context axis.
- The sigma mismatch was a real implementation issue and was fixed before advancing.

Next:
- Commit and push the sigma correction, pull it to the v4 checkout, rerun the TPU forward smoke, then validate TFRecord conversion on a tiny cached sample.
