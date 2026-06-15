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

## 2026-06-13T11:26:44Z - implementation validation and dataset conversion pivot

Goal:
- Record the completed implementation validation and the current safe path for full DROID TFRecord conversion.

Hypothesis:
- The correct stage ordering is model/trainer validation first, then small real TFRecord readback, then full val conversion, then bounded train conversion to GCS with temporary storage cleanup after each batch.

Change:
- Pushed implementation branch through commit `dc5dc3f` with the side-adapter model, frozen-backbone trainer, config, TPU wrapper, pure-Python TFRecord converter, Della wrapper, and a1001 streaming converter.
- Validated the side-adapter path on a v4 TPU with the corrected WAN sigma schedule.
- Converted and validated the full val split to `gs://v6_east1d/datasets/droid_wan_side_adapter/val`.
- Started full train conversion to `gs://v6_east1d/datasets/droid_wan_side_adapter/train`; pivoted from Della staging to a1001 streaming after Della GPFS filled.

Version Control:
- agent_id: wan-ti2v-side-adapter-20260613-073227
- worktree: /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227
- worklog: worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md
- branch: codex/wan-ti2v-side-adapter-20260613-073227
- base_commit: 2d6f8e0d54697661df33d3e1a32e7e0e9b994d97
- implementation_commit: dc5dc3f
- push/pull: pushed to origin/codex/wan-ti2v-side-adapter-20260613-073227; a1001 repo uses the pushed converter; v6 workers need a final pull before training launch
- changed_files: side-adapter model/trainer/config/wrappers/converter plus this worklog
- remote_commit/status: a1001 repo at dc5dc3f for conversion; v6 worker check/pull still pending before final training

Command / Job:
- command: `python3 -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py src/maxdiffusion/train_wan.py`
- command: `bash -n bash_scripts/train_wan_side_adapter.sh bash_scripts/della_convert_wan_side_adapter_droid.sh bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh`
- command: `bash_scripts/setup.sh MODE=stable DEVICE=tpu` on v4, followed by tiny Wan side-adapter forward smoke and pure TFRecord round-trip
- command: Della val conversion job `9617958`
- command: local a1001 orchestrator `START_SHARD=116 END_SHARD=703 BATCH_SHARDS=16 CONCURRENCY=8 UPLOAD_CONCURRENCY=4 bash bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh`
- job_id: Della val `9617958`; Della failed train continuation `9618891`; a1001 train batches include `29037532` and `29037885`
- run_dir: a1001 `/lustre/fsw/portfolios/nvr/users/lzha/wan_side_adapter_a1001`
- logs: local `/tmp/wan_a1001_remaining_116_703.log`; a1001 `/lustre/fsw/portfolios/nvr/users/lzha/wan_side_adapter_a1001/logs`
- artifacts: val and train TFRecords under `gs://v6_east1d/datasets/droid_wan_side_adapter`

Result:
- status: in_progress
- metrics/artifacts: v4 tiny forward smoke passed with output shape `(1, 4, 3, 4, 4)`, adapter params `10626`, first-frame pinning OK, and sigma schedule `[1.0, 0.9375, 0.833333, 0.625, 0.0]`.
- metrics/artifacts: val split converted to 8 shards; GCS readback parsed shard 0 with 2048 records and shard 7 with 300 records; feature byte lengths match `z_i0`, `z_video`, and `actions` schema.
- metrics/artifacts: train split has 704 expected shards for 1,440,554 examples; Della produced shards 00000-00095 before GPFS became full; a1001 streaming conversion uploaded through shard 00147 and advanced to batch 148-163.
- key evidence: `gsutil ls gs://v6_east1d/datasets/droid_wan_side_adapter/train/train-*.tfrecord | wc -l` returned `148`; a1001 storage remained at 26T free after cleanup, and Della `/scratch/gpfs` showed only 160M free, so Della remains read-only.

Analysis:
- The model implementation is validated enough to justify dataset conversion and target TPU smoke, but the full 5B training path still needs a small v6 run before batch-size scaling.
- Della cannot safely stage train outputs because the filesystem is full. The active a1001 pipeline streams source files from Della to Lustre, converts on CPU Slurm, uploads directly to GCS with short-lived local OAuth tokens, and deletes the batch cache/output before proceeding.
- Cleanup on Lustre is slower than conversion for some batches because each batch contains many small source files. It is still the correct storage-safe behavior; temporary occupancy is about 13G per 16-shard batch, far below the 2T free-space guard.

Next:
- Continue monitoring the a1001 train conversion until all 704 train shards exist in GCS.
- Write and upload the train `summary.json`, validate representative shards including the final partial shard, and remove a1001 staging data.
- Inspect `v6-64-01-lihan` worker processes before using the READY TPU resource; do not interrupt unrelated jobs without user approval.
- Pull the final branch on all v6 workers, run a short v6 smoke train, then scale batch size on v6e-64 for adapter-only training.

## 2026-06-13T13:21:24Z - a1001 resume hardening after cleanup-shell failure

Goal:
- Recover safely from a local orchestrator exit after uploading train shards 180-195, without duplicating or losing GCS shards.

Hypothesis:
- The data conversion/upload path succeeded because GCS contains contiguous train shards through 00195. The exit code 127 happened after upload, likely in the remote cleanup command running through a1001's default shell, so forcing cleanup commands through `bash -c` should remove that fragility.

Change:
- Added a `remote_bash()` helper to `bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh`.
- Switched the stage-start cleanup and post-upload cleanup calls to `remote_bash()`.
- Manually cleaned `/lustre/fsw/portfolios/nvr/users/lzha/wan_side_adapter_a1001/cache_chunk` and `tfrecord_stage` before resume.

Version Control:
- agent_id: wan-ti2v-side-adapter-20260613-073227
- worktree: /home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227
- worklog: worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md
- branch: codex/wan-ti2v-side-adapter-20260613-073227
- base_commit: d50ac01
- implementation_commit: 44104be
- push/pull: pushed to origin/codex/wan-ti2v-side-adapter-20260613-073227; local script was used for resume
- changed_files: bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh, worklog
- remote_commit/status: a1001 converter repo unchanged; this patch affects the local orchestrator only

Command / Job:
- command: `bash -n bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh`
- command: `cmd='set -euo pipefail; echo remote_bash_ok'; ssh a1001 "bash -c $(printf '%q' "$cmd")"`
- command: v4 TensorFlow readback of `gs://v6_east1d/datasets/droid_wan_side_adapter/train/train-00163-of-00704.tfrecord`
- command: manual a1001 cleanup via `bash -c 'set -euo pipefail; rm -rf ...; mkdir -p ...; du -sh ...; df -h ...'`
- job_id: a1001 batch 180-195 was `29039615`
- run_dir: a1001 `/lustre/fsw/portfolios/nvr/users/lzha/wan_side_adapter_a1001`
- logs: `/tmp/wan_a1001_remaining_116_703.log`
- artifacts: `gs://v6_east1d/datasets/droid_wan_side_adapter/train/train-00180-of-00704.tfrecord` through `train-00195-of-00704.tfrecord`

Result:
- status: passed
- metrics/artifacts: GCS train shard count reached 196; tail shards are contiguous through 00195.
- metrics/artifacts: v4 TFRecordDataset parsed train shard 00163 with 2048 records and byte lengths `z_i0=23040`, `z_video=207360`, `actions=896`.
- metrics/artifacts: manual cleanup returned staging to 40M and Lustre remained at 25T free.
- key evidence: `remote_bash_ok` probe succeeded; `bash -n` exit code 0.

Analysis:
- The correct resume point is shard 196. Because upload completed before the orchestrator exit, resuming from 196 avoids duplicate GCS objects and avoids reusing stale staging.
- The current conversion remains storage-safe; the only issue was shell robustness around cleanup on a1001.

Next:
- Commit and push the hardening patch.
- Resume local a1001 orchestrator with `START_SHARD=196 END_SHARD=703`.

## 2026-06-13T15:57:00Z - train conversion progress checkpoint

Goal:
- Record current train conversion progress after the a1001 resume hardening patch was exercised successfully.

Result:
- status: in_progress
- metrics/artifacts: GCS train shard count reached `308/704`; completed train shards are contiguous through `train-00307-of-00704.tfrecord`.
- metrics/artifacts: The resumed local orchestrator `/tmp/wan_a1001_remaining_196_703.log` successfully completed and cleaned batches 196-211, 212-227, 228-243, 244-259, 260-275, 276-291, and 292-307.
- metrics/artifacts: a1001 staging repeatedly returned to `40M` after cleanup and Lustre remained around `25T` free.

Analysis:
- The `remote_bash()` cleanup patch is validated by multiple post-upload cleanup cycles.
- Continue the current single-stream conversion; Della remains read-only and v6 remains occupied by an unrelated ego-lap training process.

Next:
- Continue monitoring from batch 308-323 until all 704 train shards are in `gs://v6_east1d`.

## 2026-06-13T18:48:00Z - train conversion progress checkpoint

Goal:
- Record current train conversion progress and storage behavior after additional a1001 streaming batches.

Result:
- status: in_progress
- metrics/artifacts: GCS train shard count reached `340/704`; completed train shards are contiguous through `train-00339-of-00704.tfrecord`.
- metrics/artifacts: a1001 batches 308-323 and 324-339 converted successfully on Slurm (`29047467`, `29047912`), uploaded to `gs://v6_east1d/datasets/droid_wan_side_adapter/train`, and entered cleanup.
- metrics/artifacts: Conversion jobs completed with exit `0:0`, per-worker write rates around 18-22 examples/s, and MaxRSS around 1.2G.
- metrics/artifacts: Lustre remained around `25T` free; each 16-shard batch staged about `7.4G` of source cache plus about `7.2G` of TFRecords before upload cleanup.

Analysis:
- The data path remains correct and storage-safe, but Lustre metadata operations are slower than conversion for these many-small-file batches. Staging, post-stage `find`/`du`, and cleanup are the dominant delays.
- The active orchestrator script should not be edited while it is running. If it fails at a safe boundary, a follow-up patch should replace recursive post-stage diagnostics with a cheaper top-level sanity check and consider tar extraction options that reduce metadata writes.

Next:
- Continue monitoring from batch 340-355 onward until all 704 train shards are in `gs://v6_east1d`.
- After final upload, write the train summary, validate representative shards, and delete remaining a1001 temporary data.

## 2026-06-13T19:25:00Z - train conversion progress and v6 launch permission

Goal:
- Record the current a1001 conversion state and the updated user instruction for the TPU launch stage.

Result:
- status: in_progress
- metrics/artifacts: GCS train shard count reached `356/704`; completed train shards are contiguous through `train-00355-of-00704.tfrecord`.
- metrics/artifacts: a1001 batch 340-355 converted successfully on Slurm job `29048355`, exit `0:0`, elapsed `00:03:31`, MaxRSS `1218248K`.
- metrics/artifacts: All 16 shards from batch 340-355 reached final local size around `453M`, uploaded to `gs://v6_east1d/datasets/droid_wan_side_adapter/train`, and cleanup is in progress.
- metrics/artifacts: Lustre remains around `24T` free, well above the user's 2T free-space guard.
- key evidence: GCS shard count returned `356`; `sacct -j 29048355` showed `COMPLETED|0:0`.

Analysis:
- The conversion data path is still correct and storage-safe. The remaining cost is Lustre metadata cleanup after each batch.
- The user explicitly said no additional confirmation is needed for subsequent steps once the current work is finished, and authorized using `tpu watch` to create new TPU v6 slices.

Next:
- Let the active cleanup finish before the wrapper stages batch 356-371.
- Complete all 704 train shards, upload `summary.json`, validate representative GCS shards including the final partial shard, and delete a1001 staging data.
- If the existing v6-64 is unavailable, use `tpu watch v6 -n 64` to provision a new v6 slice for the smoke and full adapter-only training run.

## 2026-06-13T21:13:16Z - train conversion progress checkpoint

Goal:
- Record the current train conversion progress after additional a1001 batches and confirm the storage guard remains intact.

Result:
- status: in_progress
- metrics/artifacts: GCS train shard count reached `404/704`; completed train shards are contiguous through `train-00403-of-00704.tfrecord`.
- metrics/artifacts: a1001 batches 356-371, 372-387, and 388-403 converted successfully on Slurm jobs `29048798`, `29049354`, and `29049802`, then uploaded to `gs://v6_east1d/datasets/droid_wan_side_adapter/train`.
- metrics/artifacts: Batch 372-387 and 388-403 each staged `32768` source files, about `7.4G` of cache, and uploaded all 16 TFRecord shards before cleanup.
- metrics/artifacts: Lustre remained around `24T` free, still well above the user's 2T free-space guard.

Analysis:
- The a1001 streaming conversion remains correct: batches continue to finish with contiguous GCS shard coverage and no duplicate/missing shard evidence.
- Runtime is dominated by source staging and cleanup of many small files; conversion itself remains short and low-memory.

Next:
- Continue monitoring the active orchestrator from batch 404 onward until all 704 train shards exist in GCS.
- After final upload, write the train summary, validate representative shards including the final partial shard, and delete remaining a1001 temporary data before launching TPU training.

## 2026-06-14T00:52:53Z - train conversion progress checkpoint

Goal:
- Record train conversion progress after the a1001 stream passed the 500-shard mark.

Result:
- status: in_progress
- metrics/artifacts: GCS train shard count reached `500/704`; completed train shards are contiguous through `train-00499-of-00704.tfrecord`.
- metrics/artifacts: a1001 batches 404-419, 420-435, 436-451, 452-467, 468-483, and 484-499 completed successfully and uploaded all expected TFRecord shards.
- metrics/artifacts: Representative Slurm job ids in this interval were `29050231`, `29050680`, `29051847`, `29052683`, `29053129`, and `29053559`.
- metrics/artifacts: Each full 16-shard batch continued to stage `32768` source files and about `7.4G` of source cache before conversion.
- metrics/artifacts: Lustre free space stayed around `23T`, still well above the requested 2T free-space guard.

Analysis:
- GCS shard coverage remains contiguous, which is the key correctness check for the current conversion stage.
- The bottleneck remains Lustre metadata during tar extraction, diagnostic `du`, and cleanup. Conversion and upload continue to finish cleanly once staging completes.

Next:
- Continue the active single-stream a1001 conversion from batch 500 onward until all `704/704` train shards exist in GCS.
- After final upload, write the train summary, validate representative shards including the final partial shard, delete remaining a1001 temporary data, and then move to TPU v6 smoke/full training.

## 2026-06-14T02:51:48Z - train conversion progress checkpoint

Goal:
- Record train conversion progress after the a1001 stream passed the 548-shard mark and confirm storage remains within the user's guardrail.

Result:
- status: in_progress
- metrics/artifacts: GCS train shard count reached `548/704`; completed train shards are contiguous through `train-00547-of-00704.tfrecord`.
- metrics/artifacts: a1001 batches 500-515, 516-531, and 532-547 completed successfully and uploaded all expected TFRecord shards.
- metrics/artifacts: Slurm jobs `29053938`, `29054391`, and `29054785` each completed cleanly for the three most recent batches.
- metrics/artifacts: Batch 548-563 has started staging on a1001; no TFRecords from that batch should appear in GCS until staging, Slurm conversion, and upload complete.
- metrics/artifacts: Lustre free space remains around `23T`, well above the requested 2T free-space guard.

Analysis:
- Conversion correctness still hinges on contiguous GCS shard coverage; the latest verified contiguous range is `0..547`.
- The active bottleneck remains Lustre metadata work during staging diagnostics and cleanup, but temporary storage footprint remains bounded to one batch.

Next:
- Continue monitoring the active a1001 orchestrator through the remaining train shards.
- After final upload, write the train summary, validate representative shards including final shard `703`, clean a1001 staging, and proceed to TPU v6 smoke/full training without asking for another confirmation.

## 2026-06-14T05:04:15Z - train conversion progress checkpoint

Goal:
- Record train conversion progress after passing 600 verified train shards in GCS.

Result:
- status: in_progress
- metrics/artifacts: GCS train shard count reached `612/704`; completed train shards are contiguous through `train-00611-of-00704.tfrecord`.
- metrics/artifacts: a1001 batches 548-563, 564-579, 580-595, and 596-611 completed successfully and uploaded all expected TFRecord shards.
- metrics/artifacts: Slurm jobs `29055104`, `29055514`, `29055994`, and `29056335` completed cleanly for these batches.
- metrics/artifacts: Conversion write rates remained around 18-23 examples/s per shard writer; GCS upload coverage stayed contiguous after every batch.
- metrics/artifacts: Lustre free space remains around `23T`, still well above the requested 2T free-space guard.

Analysis:
- The staged data and TFRecord conversion path remain correct; the strongest evidence remains contiguous GCS shard coverage through shard 611.
- Remaining work is six train batches: 612-627, 628-643, 644-659, 660-675, 676-691, and 692-703.

Next:
- Continue monitoring the active a1001 orchestrator until all `704/704` shards are present in GCS.
- Then upload the train summary, validate representative shards including the final partial shard, clean a1001 staging, and launch TPU v6 smoke/full training.

## 2026-06-14T07:05:18Z - train conversion sacct race recovery

Goal:
- Recover the a1001 train conversion after the local orchestrator exited during batch 644-659.

Change:
- Hardened `bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh` so `wait_for_job` waits for terminal `sacct` states after a job disappears from `squeue`, instead of failing on a transient `RUNNING` report.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- base_commit: `eb5829585410cdbfb4c207da77fb29885793e353`
- implementation_commit: `c62bc460f4c65bc8acd2e8926925e07ae3305c36`
- changed_files: `bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh`, `worklogs/wan-ti2v-side-adapter/wan-ti2v-side-adapter-20260613-073227.md`

Command / Job:
- command: manual upload of staged shards 644-659 using the existing local-token/remote-curl path; resume planned with `START_SHARD=660 END_SHARD=703`
- job_id: `29057788`
- logs: `/tmp/wan_a1001_remaining_196_703.log`
- artifacts: `gs://v6_east1d/datasets/droid_wan_side_adapter/train/train-00644-of-00704.tfrecord` through `train-00659-of-00704.tfrecord`

Result:
- status: in_progress
- metrics/artifacts: Slurm job `29057788` completed cleanly with exit `0:0` and produced all 16 staged TFRecords.
- metrics/artifacts: Manual upload moved shards 644-659 to GCS; GCS train shard count reached `660/704`.
- metrics/artifacts: a1001 Lustre remains around `22T` free, above the requested 2T guardrail.

Analysis:
- The failure was in orchestration, not conversion. `sacct` briefly reported `RUNNING` after `squeue` no longer showed the job; the job completed normally shortly afterward.
- Resuming from shard 660 avoids recomputing or reuploading the already-valid 644-659 batch.

Next:
- Let the manual cleanup of batch 644-659 finish, then resume the patched orchestrator from shard 660 through 703.
- After all shards are present, upload the train summary, validate representative shards including final shard 703, delete remaining a1001 staging data, and proceed to TPU v6 training.

## 2026-06-14T08:54:55Z - train TFRecord conversion complete

Goal:
- Finish the full DROID train TFRecord conversion and validate the GCS dataset before TPU training.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `c62bc460f4c65bc8acd2e8926925e07ae3305c36`
- changed_files: worklog only

Command / Job:
- command: `START_SHARD=660 END_SHARD=703 BATCH_SHARDS=16 CONCURRENCY=8 UPLOAD_CONCURRENCY=4 bash bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh`
- job_ids: `29058797` for shards 660-675, `29059376` for shards 676-691, `29059808` for shards 692-703
- logs: `/tmp/wan_a1001_remaining_660_703.log`
- artifacts: `gs://v6_east1d/datasets/droid_wan_side_adapter/train/train-00000-of-00704.tfrecord` through `train-00703-of-00704.tfrecord`, plus `gs://v6_east1d/datasets/droid_wan_side_adapter/train/summary.json`

Result:
- status: passed
- metrics/artifacts: GCS train shard count is `704/704`; shard indices are contiguous `0..703`.
- metrics/artifacts: Final shard `train-00703-of-00704.tfrecord` contains `810` records with ordinals `1439744..1440553`.
- metrics/artifacts: Pure Python TFRecord/protobuf validation passed for shards `0, 163, 307, 339, 371, 499, 611, 659, 675, 691, 703`.
- metrics/artifacts: Validated byte lengths per record: `z_i0=23040`, `z_video=207360`, `actions=896`; full shards contained `2048` records.
- metrics/artifacts: a1001 temp root returned to `41M`; Lustre remained around `22T` free.

Analysis:
- The train dataset is now in the expected MaxDiffusion-compatible TFRecord format under `gs://v6_east1d`, with representative parsing evidence including the final partial shard.
- The a1001 conversion path stayed within the storage guardrail by holding only one batch of source cache and TFRecords at a time.

Next:
- Prepare TPU v6 training: update/pull the pushed branch on a v6-64 slice, run a short smoke training job, inspect logs/loss/checkpoints, then scale batch size for the full adapter-only run.

## 2026-06-14T08:58:10Z - v6e-64 smoke launch

Goal:
- Queue an isolated v6e-64 TPU slice for a short Wan2.2 TI2V 5B side-adapter training smoke.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `5f5343bcc7673eab67faf5a868504083aba693e9`

Command / Job:
- tpu_name: `v6-64-02-lzha`
- accelerator: `v6e-64`
- setup: `git fetch origin && git checkout codex/wan-ti2v-side-adapter-20260613-073227 && git pull origin codex/wan-ti2v-side-adapter-20260613-073227 && cd ~ && bash maxdiffusion/bash_scripts/setup.sh MODE=stable DEVICE=tpu`
- command: `WANDB_DISABLED=true RUN_NAME=wan-side-adapter-v6e64-smoke-bs1-20260614-085810 MAX_TRAIN_STEPS=3 CHECKPOINT_EVERY=3 EVAL_EVERY=0 LOG_PERIOD=1 PER_DEVICE_BATCH_SIZE=0.015625 bash bash_scripts/train_wan_side_adapter.sh`
- train_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
- val_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- output_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-smoke-bs1-20260614-085810`

Result:
- status: launched
- metrics/artifacts: pending allocation/setup/worker log inspection.

Analysis:
- Existing `v6-64-01-lihan` is healthy but already running a Python training process, so this task should use a separate queued v6e-64 slice.
- Smoke uses `per_device_batch_size=0.015625`, giving global train batch size 1 on 64 devices, before scaling toward the reference DDP global batch size.

Next:
- Monitor `tpu watch` until allocation/setup/launch completes, then inspect worker logs for device count, dataset paths, loss, gradients, and checkpoint artifacts.

## 2026-06-14T09:44:39Z - v6e-64 smoke backend fix

Goal:
- Diagnose the first v6e-64 smoke failure and prepare a corrected relaunch on the same TPU slice.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `6d381756389d871b89859c2c2ae14d8a14cd7226`
- changed_files: `bash_scripts/train_wan_side_adapter.sh`, worklog

Command / Job:
- tpu_name: `v6-64-02-lzha`
- failed_run_name: `wan-side-adapter-v6e64-smoke-bs1-20260614-085810`
- worker0_log: `~/maxdiffusion/logs/tpu_20260614-093043.log`
- validation: `bash -n bash_scripts/train_wan_side_adapter.sh`

Result:
- status: fix_ready
- metrics/artifacts: The smoke reached `WanPipelineTI2V_2_2.from_pretrained` and failed before the first batch with `RuntimeError: Unknown backend cpu. Available backends are ['tpu']`.
- metrics/artifacts: The traceback entered `load_wan_vae(..., "cpu")`, which calls `jax.devices("cpu")`.
- metrics/artifacts: Updated the launcher default from `JAX_PLATFORMS=tpu` to `JAX_PLATFORMS=tpu,cpu` so the TPU backend remains primary while the Wan loader can access CPU devices for VAE/model setup.

Analysis:
- This was a launch environment issue, not a dataset or side-adapter implementation failure. The training wrapper hid the CPU backend, while the upstream Wan pipeline expects CPU device access during model construction.
- The v6e-64 TPU slice remains `READY` and can be reused for the corrected smoke after the stale watcher is stopped and the branch is pushed.

Next:
- Commit and push the launch fix, stop the stale `tpu watch` process, then relaunch the short smoke on `v6-64-02-lzha` and inspect worker logs for model load, dataset iteration, train metrics, and checkpoint output.

## 2026-06-14T10:04:23Z - v6e-64 smoke config upload fix

Goal:
- Diagnose the corrected smoke relaunch and fix the next multihost startup failure.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `51dd25d3bb30659c99a5a8af2d9e3648e5c3fa6e`
- changed_files: `bash_scripts/train_wan_side_adapter.sh`, `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`, worklog

Command / Job:
- failed_run_name: `wan-side-adapter-v6e64-smoke-bs1-20260614-095800`
- tpu_name: `v6-64-02-lzha`
- worker0_log: `~/maxdiffusion/logs/tpu_20260614-095608.log`
- worker15_log: `~/maxdiffusion/logs/tpu_20260614-095608.log`
- validation: `bash -n bash_scripts/train_wan_side_adapter.sh`

Result:
- status: fix_ready
- metrics/artifacts: The smoke passed the previous blockers: it found all `64` TPU devices and successfully loaded the Wan VAE on `cpu:0`.
- metrics/artifacts: Worker15 crashed first while writing `config.yml` before multihost initialization completed, attempting to use a nonexistent bucket named `wan-side-adapter-v6e64-smoke-bs1-20260614-095800`.
- metrics/artifacts: Worker0 and other workers then aborted in the JAX distributed shutdown barrier, with stack traces around `_compute_null_context`; this was secondary to the early worker15 config-upload failure.
- metrics/artifacts: Verified all 16 workers had zero stale `python -`, `train_wan.py`, or `tmux` processes after cleanup.

Analysis:
- `pyconfig.initialize()` calls `write_config_raw_keys_for_gcs()` before `ensure_machinelearning_job_runs()` initializes multihost JAX, so `jax.process_index()` is not a reliable multiworker guard at that point. Most Wan configs avoid this path with `save_config_to_gcs: False`; the side-adapter config should do the same.
- The wrapper also incorrectly defaulted `output_dir` to include `RUN_NAME`, while `pyconfig` appends `run_name` to `output_dir` for checkpoint, metrics, and tensorboard paths. The default now points at the base run directory and passes `base_output_directory` explicitly.

Next:
- Commit and push the config/output-dir fix, relaunch the 3-step smoke on the same v6e-64 slice, and inspect logs until it reaches dataset iteration, train steps, and checkpoint output or exposes the next implementation bug.

## 2026-06-14T10:12:17Z - side-adapter state sharding fix

Goal:
- Diagnose the next smoke failure after the config/output fix and correct adapter parameter sharding.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `ba3416d062f9e2c2dd27b061d96d71d1f83a5a31`
- changed_files: `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`, worklog

Command / Job:
- failed_run_name: `wan-side-adapter-v6e64-smoke-bs1-20260614-100700`
- tpu_name: `v6-64-02-lzha`
- worker0_log: `~/maxdiffusion/logs/tpu_20260614-100625.log`
- validation: `python3 -m py_compile src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`

Result:
- status: fix_ready
- metrics/artifacts: The smoke passed config initialization, found all `64` TPU devices, loaded the VAE on CPU, and loaded the transformer far enough to create training state.
- metrics/artifacts: It failed in `_shard_state()` with `IndivisibleError`: a leaf with shape `(7, 512)` was assigned a sharding that partitions axis 0 over the 64-way `context` mesh.
- metrics/artifacts: Verified all 16 TPU workers had zero stale smoke/probe processes after the failed run.

Analysis:
- The side-adapter module used standard WAN logical axis annotations such as `embed`, `mlp`, and `heads`. Those rules are appropriate for the large frozen transformer, but not for small adapter/action parameters such as action-length embeddings.
- The trainer now keeps the frozen transformer on its actual TPU sharding, but overrides the trainable adapter `params` and optimizer `opt_state` to replicated `NamedSharding(mesh, P())`.

Next:
- Commit and push the sharding fix, relaunch the short smoke, and inspect for dataset iteration, train metrics, and checkpoint output.

## 2026-06-14T10:32:26Z - v6e-64 smoke first-step monitor

Goal:
- Monitor the corrected 3-step smoke after the adapter train-state sharding fix.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `0a2301adb613b6352beeb830cb1e9e9d85aaedff`
- changed_files: worklog

Command / Job:
- run_name: `wan-side-adapter-v6e64-smoke-bs1-20260614-101400`
- tpu_name: `v6-64-02-lzha`
- accelerator: `v6e-64`
- command: `WANDB_DISABLED=true RUN_NAME=wan-side-adapter-v6e64-smoke-bs1-20260614-101400 MAX_TRAIN_STEPS=3 CHECKPOINT_EVERY=3 EVAL_EVERY=0 LOG_PERIOD=1 PER_DEVICE_BATCH_SIZE=0.015625 bash bash_scripts/train_wan_side_adapter.sh`
- train_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
- val_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-smoke-bs1-20260614-101400/checkpoints`
- primary_log: worker15 `~/maxdiffusion/logs/tpu_20260614-101416.log`

Result:
- status: running
- metrics/artifacts: All 16 workers are alive in `train_wan.py` with no traceback or resource-exhausted error.
- metrics/artifacts: The run passed model load, state creation, adapter-state replication, and Orbax checkpoint-manager creation.
- metrics/artifacts: Worker/process mapping is shuffled; worker15 is JAX `process=0` and has logged `***** Running WAN TI2V side-adapter training *****`.
- metrics/artifacts: No `step`, loss, or checkpoint has been logged yet. GCS currently contains only the run/checkpoints prefix.

Analysis:
- Worker thread sampling shows active `xla_tpu_thread` and `llvm-worker-*` threads, consistent with first-batch/first-step JIT compilation rather than a crashed process.
- The config log reports `global_batch_size_to_load=64` while the wrapper passes `global_batch_size_to_load=1`; this is expected from `pyconfig.calculate_global_batch_sizes()` for fractional per-device batches, which loads one example per device and trains on `global_batch_size_to_train_on=1`.

Next:
- Keep polling worker15 and GCS until the smoke emits train metrics/checkpoint output or fails with a new runtime error.

## 2026-06-14T10:43:12Z - v6e-64 smoke compile pressure fix

Goal:
- Avoid an oversized first-step compile while preserving the 25-step side-adapter rollout semantics.

Hypothesis:
- The Python `for` loop over `side_adapter_sampling_steps=25` is being unrolled inside the jitted train step, causing the compiler to trace a very large graph with repeated 5B-transformer calls.

Change:
- Replaced the Python-unrolled rollout in `_rollout_loss()` with `jax.lax.fori_loop`, keeping the same timestep, guidance, first-frame pinning, and update equations.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- base_commit: `0a2301adb613b6352beeb830cb1e9e9d85aaedff`
- implementation_commit: pending
- changed_files: `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`, worklog
- validation: `python3 -m py_compile src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`

Command / Job:
- affected_run_name: `wan-side-adapter-v6e64-smoke-bs1-20260614-101400`
- worker15_log: `~/maxdiffusion/logs/tpu_20260614-101416.log`

Result:
- status: fix_ready
- metrics/artifacts: At `2026-06-14T10:43:12Z`, the smoke had spent about 29 minutes after `***** Running WAN TI2V side-adapter training *****` with no step/loss/checkpoint log.
- metrics/artifacts: Worker15 process RSS reached about `305 GB`; the hottest thread remained `xla_tpu_thread`.
- metrics/artifacts: No traceback, resource-exhausted error, or checkpoint was produced before deciding to stop and relaunch with the loop patch.

Analysis:
- This is not enough evidence to call the implementation numerically correct. It is enough evidence that the current compile shape is unhealthy for iteration and likely caused by loop unrolling.
- An XLA loop is the minimal semantic-preserving change before rerunning the same 3-step smoke.

Next:
- Stop only the `wan-side-adapter-v6e64-smoke-bs1-20260614-101400` train processes, commit and push the loop patch, then relaunch a 3-step smoke on `v6-64-02-lzha`.

## 2026-06-14T11:02:38Z - loop smoke passed and probe checkpoint guard

Goal:
- Validate the loop-form rollout on v6e-64 and prepare storage-light batch-size probes.

Change:
- Added `SAVE_FINAL_CHECKPOINT` to `bash_scripts/train_wan_side_adapter.sh` so probe runs can disable final checkpoint writes while full training keeps checkpointing enabled.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `29384d5297072da2fd222aa23561b63ee8282144`
- changed_files: `bash_scripts/train_wan_side_adapter.sh`, worklog

Command / Job:
- run_name: `wan-side-adapter-v6e64-smoke-bs1-loop-20260614-104600`
- tpu_name: `v6-64-02-lzha`
- worker15_log: `~/maxdiffusion/logs/tpu_20260614-104702.log`
- command: `WANDB_DISABLED=true RUN_NAME=wan-side-adapter-v6e64-smoke-bs1-loop-20260614-104600 MAX_TRAIN_STEPS=3 CHECKPOINT_EVERY=3 EVAL_EVERY=0 LOG_PERIOD=1 PER_DEVICE_BATCH_SIZE=0.015625 bash bash_scripts/train_wan_side_adapter.sh`

Result:
- status: passed
- metrics/artifacts: The run used commit `29384d5297072da2fd222aa23561b63ee8282144` on all workers.
- metrics/artifacts: Worker15 was JAX `process=0` and logged `step 1/3 loss=2.578125 grad_norm=512.890`, `step 2/3 loss=4.781250 grad_norm=172.694`, and `step 3/3 loss=2.906250 grad_norm=142.403`.
- metrics/artifacts: Checkpoint step `3` finalized at `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-smoke-bs1-loop-20260614-104600/checkpoints/3`.
- metrics/artifacts: GCS contains `params`, `opt_state`, and `step` checkpoint trees with per-process array metadata and `commit_success.txt`.
- metrics/artifacts: A strict post-run check found zero remaining `train_wan.py`, `python`, or side-adapter launcher wrapper processes on all 16 workers.

Analysis:
- The XLA loop patch fixed the compile-pressure issue. The loop smoke completed the full 25-step side-adapter sampling path with nonzero gradients and a finalized multihost checkpoint.
- Batch probing should now start with no final checkpoint saves to keep GCS usage controlled, then re-enable checkpointing for the selected full run.

Next:
- Commit and push the probe checkpoint guard, then run storage-light batch-size probes on the same v6e-64 slice.

## 2026-06-14T11:52:00Z - v6e-64 batch-size probes through gbs12

Goal:
- Find the largest storage-light global batch size that compiles and completes real side-adapter optimizer steps on v6e-64 before launching the full frozen-backbone adapter-only run.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `d1427d4ea164dc9916f43385c90c95ae96c6ca28`
- changed_files: worklog

Command / Job:
- tpu_name: `v6-64-02-lzha`
- accelerator: `v6e-64`
- probe_template: `WANDB_DISABLED=true RUN_NAME=<run> MAX_TRAIN_STEPS=2 CHECKPOINT_EVERY=0 SAVE_FINAL_CHECKPOINT=False EVAL_EVERY=0 LOG_PERIOD=1 PER_DEVICE_BATCH_SIZE=<gbs/64> bash bash_scripts/train_wan_side_adapter.sh`
- train_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
- val_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`

Result:
- status: partial_pass_then_preempted_after_probe
- metrics/artifacts: `gbs8` (`PER_DEVICE_BATCH_SIZE=0.125`) completed two train steps: `step 1/2 loss=3.343750 grad_norm=537.249`, `step 2/2 loss=2.343750 grad_norm=106417.656`.
- metrics/artifacts: `gbs16` (`PER_DEVICE_BATCH_SIZE=0.25`) failed during compile with `CompileTimeHbmOom`, using `31.42G` of `31.25G` HBM and exceeding capacity by `182.73M`.
- metrics/artifacts: `gbs12` (`PER_DEVICE_BATCH_SIZE=0.1875`, run `wan-side-adapter-v6e64-probe-gbs12-20260614-113600`) completed two train steps: `step 1/2 loss=3.265625 grad_norm=355.902`, `step 2/2 loss=1.093750 grad_norm=422.942`.
- metrics/artifacts: Storage-light probe runs created no checkpoint payloads beyond the base GCS prefix, as intended.
- metrics/artifacts: During post-gbs12 cleanup, `v6-64-02-lzha` entered terminal state `PREEMPTED`; workers 14 and 15 could no longer accept SSH.

Analysis:
- Global batch 12 is a validated feasible batch for the 25-step rollout on v6e-64. Global batch 16 is just over the compile-time HBM limit, so the next useful probe is `gbs14`.
- The high `gbs8` step-2 grad norm is notable but did not produce NaNs or a runtime failure; `gbs12` gradients were much more ordinary over two steps.
- The preemption happened after gbs12 produced both metrics, so it invalidates only TPU reuse, not the gbs12 result.

Next:
- Reacquire or recreate a v6e-64 slice with `tpu watch`, using `bash_scripts/setup.sh` for environment setup, then run a storage-light `gbs14` probe.

## 2026-06-14T13:43:00Z - v6e-64 max-batch selection

Goal:
- Finish the last useful storage-light batch probes and select the full-run batch size.

Change:
- Added optional `EVAL_MAX_BATCHES` forwarding to `bash_scripts/train_wan_side_adapter.sh` so the full run can keep validation small, close to the previous Wan2.2 `MAX_VAL_SAMPLES=8` behavior.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- base_commit: `296263642b8fe01a94efde9987d32f00e51d5bab`
- implementation_commit: pending
- changed_files: `bash_scripts/train_wan_side_adapter.sh`, worklog
- validation: `bash -n bash_scripts/train_wan_side_adapter.sh`

Command / Job:
- tpu_name: `v6-64-02-lzha`
- accelerator: `v6e-64`
- gbs14_run: `wan-side-adapter-v6e64-probe-gbs14-20260614-115500`
- gbs15_run: `wan-side-adapter-v6e64-probe-gbs15-20260614-132700`
- probe_template: `WANDB_DISABLED=true RUN_NAME=<run> MAX_TRAIN_STEPS=2 CHECKPOINT_EVERY=0 SAVE_FINAL_CHECKPOINT=False EVAL_EVERY=0 LOG_PERIOD=1 PER_DEVICE_BATCH_SIZE=<gbs/64> bash bash_scripts/train_wan_side_adapter.sh`

Result:
- status: passed
- metrics/artifacts: `gbs14` (`PER_DEVICE_BATCH_SIZE=0.21875`) completed two train steps: `step 1/2 loss=3.453125 grad_norm=318.013`, `step 2/2 loss=1.617188 grad_norm=1164.384`.
- metrics/artifacts: `gbs15` (`PER_DEVICE_BATCH_SIZE=0.234375`) completed two train steps: `step 1/2 loss=3.453125 grad_norm=410.899`, `step 2/2 loss=1.593750 grad_norm=3072.043`.
- metrics/artifacts: `gbs16` remains the first failing batch, with prior `CompileTimeHbmOom` using `31.42G` of `31.25G` HBM.
- metrics/artifacts: Storage-light probes used `SAVE_FINAL_CHECKPOINT=False` and did not write checkpoint payloads.
- metrics/artifacts: After gbs15 cleanup, all workers reported no `tpu` tmux session and no side-adapter training processes.

Analysis:
- Global batch `15` is the largest validated v6e-64 batch for this implementation and dataset. It is close to the memory boundary, but it compiles and completes real optimizer steps.
- The full run should use `PER_DEVICE_BATCH_SIZE=0.234375`, `MAX_TRAIN_STEPS=10000`, `CHECKPOINT_EVERY=1000`, `EVAL_EVERY=1000`, `LOG_PERIOD=20`, `SAVE_FINAL_CHECKPOINT=True`, and `EVAL_MAX_BATCHES=1`.

Next:
- Commit and push the wrapper/worklog update, then launch the full 10k-step frozen-backbone side-adapter run on the ready v6e-64 slice.

## 2026-06-14T13:50:00Z - full v6e-64 gbs15 launch

Goal:
- Launch the full frozen-backbone Wan2.2 TI2V 5B side-adapter training run on v6e-64 using the largest validated batch size.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `36edeb1884222e0580e3f95cdc16455e0f195c37`
- push/pull: pushed to origin; TPU workers fast-forwarded from `2962636` to `36edeb1`
- changed_files: worklog
- remote_commit/status: worker logs report `COMMIT=36edeb1884222e0580e3f95cdc16455e0f195c37`

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-20260614-134500`
- tpu_name: `v6-64-02-lzha`
- accelerator: `v6e-64`
- launch_command: `tpu watch v6 --force -n 64 --setup-cmd "git fetch origin && git checkout codex/wan-ti2v-side-adapter-20260613-073227 && git pull origin codex/wan-ti2v-side-adapter-20260613-073227" codex/wan-ti2v-side-adapter-20260613-073227 WANDB_PROJECT=maxdiffusion-wan-side-adapter RUN_NAME=wan-side-adapter-v6e64-full-gbs15-20260614-134500 MAX_TRAIN_STEPS=10000 CHECKPOINT_EVERY=1000 SAVE_FINAL_CHECKPOINT=True EVAL_EVERY=1000 EVAL_MAX_BATCHES=1 LOG_PERIOD=20 PER_DEVICE_BATCH_SIZE=0.234375 bash bash_scripts/train_wan_side_adapter.sh`
- output_base: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs15-20260614-134500/checkpoints`
- primary_worker_log: worker5 `~/maxdiffusion/logs/tpu_20260614-134038.log`
- wandb_project: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter`
- wandb_run: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/omt2ym2z`

Result:
- status: running
- metrics/artifacts: Launch completed at `2026-06-14 06:40:51` local time.
- metrics/artifacts: Worker logs verify `global_batch_size_to_train_on=15`, `per_device_batch_size=0.234375`, `max_train_steps=10000`, `checkpoint_every=1000`, `eval_every=1000`, `eval_max_batches=1`, and `wandb_project=maxdiffusion-wan-side-adapter`.
- metrics/artifacts: Worker5 is JAX process 0 and W&B initialized online as run id `omt2ym2z`.
- metrics/artifacts: No train metric yet because full run uses `LOG_PERIOD=20`; first expected loss line is step 20.

Analysis:
- The full run matches the previous Wan2.2 DROID side-adapter recipe: fresh noise, 25-step rollout, lr `5e-5`, 500-step warmup via `warmup_steps_fraction=0.05`, 10k steps, checkpoint/eval every 1000, and frozen backbone with trainable side adapter only.
- `tpu create --force` could not be used for background recovery because the helper rejects suffix `lzha`; this is an irom-tool validation mismatch with the existing TPU name. Manual monitoring/relaunch remains required.

Next:
- Monitor worker5/process 0 until the first step-20 train metric appears or a failure/preemption occurs; also keep checking TPU health because no background watcher owns recovery for this run.

## 2026-06-14T14:00:00Z - full run first metric

Goal:
- Verify that the full 10k-step gbs15 run is genuinely training, not just launched.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-20260614-134500`
- primary_worker_log: worker5 `~/maxdiffusion/logs/tpu_20260614-134038.log`
- wandb_run: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/omt2ym2z`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Worker5 process `22227` is alive; elapsed `18:22` at `2026-06-14T13:59:01Z`.
- metrics/artifacts: First train metric logged at step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`.
- metrics/artifacts: No checkpoint is expected until step `1000`; GCS currently has only the base prefix.

Analysis:
- The low LR is expected during the 500-step warmup (`warmup_steps_fraction=0.05` over 10k steps).
- The run is now past compile/startup and advancing real training steps. No NaNs, OOMs, tracebacks, or TPU health failures are visible.

Next:
- Continue monitoring through additional train logs and the first checkpoint/eval at step `1000`.

## 2026-06-14T14:15:00Z - full run warmup metrics and maintenance warning

Goal:
- Inspect early full-run loss/gradient trend and TPU health after the first metric.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-20260614-134500`
- primary_worker_log: worker5 `~/maxdiffusion/logs/tpu_20260614-134038.log`

Result:
- status: running_with_maintenance_warning
- metrics/artifacts: TPU state is still `READY`, but health reports `UNHEALTHY_MAINTENANCE`.
- metrics/artifacts: Worker5 process `22227` is alive; elapsed `34:07` at `2026-06-14T14:14:46Z`.
- metrics/artifacts: Train logs:
  - step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`
  - step `40/10000`: `loss=1.478320`, `grad_norm=389378.840`, `lr=3.90e-06`, `steps/s=0.069`
  - step `60/10000`: `loss=1.130664`, `grad_norm=3744.459`, `lr=5.90e-06`, `steps/s=0.069`
  - step `80/10000`: `loss=0.612598`, `grad_norm=749.494`, `lr=7.90e-06`, `steps/s=0.069`
- metrics/artifacts: No checkpoint yet; first checkpoint/eval remains scheduled for step `1000`.

Analysis:
- The loss is decreasing during warmup and no NaNs/OOMs/tracebacks are visible. Step `40` has a large grad-norm spike, but subsequent grad norms recover and training continues.
- `UNHEALTHY_MAINTENANCE` means the spot slice may be reclaimed before the step-1000 checkpoint. If that happens before a checkpoint, relaunch from scratch on the next v6e-64 allocation.

Next:
- Tighten monitoring cadence until health recovers, preemption occurs, or the first checkpoint is reached.

## 2026-06-14T14:21:00Z - full run preempted before first checkpoint

Goal:
- Determine whether the maintenance warning resolved or preempted the active full run.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-20260614-134500`
- tpu_name: `v6-64-02-lzha`

Result:
- status: preempted
- metrics/artifacts: At `2026-06-14T14:20:25Z`, `gcloud compute tpus tpu-vm describe` reported `state: PREEMPTED`.
- metrics/artifacts: SSH failed with `This TPU has terminal state "PREEMPTED", so it cannot be used anymore.`
- metrics/artifacts: No checkpoint was written because the run had not reached step `1000`. Last durable train logs were through step `80`.

Analysis:
- The preemption followed the prior `UNHEALTHY_MAINTENANCE` warning. This is an infrastructure interruption, not a training-code failure.
- Since there is no checkpoint to restore, the correct continuation is a fresh full run with the same validated gbs15 settings and a new run name/W&B run.

Next:
- Requeue/recreate v6e-64 with `tpu watch`, run setup through the repo wrapper path as needed, and relaunch full training from scratch.

## 2026-06-14T14:48:00Z - full run restart1 first metric

Goal:
- Verify that the fresh restart after preemption is training with the intended gbs15 recipe.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart1-20260614-142200`
- tpu_name: `v6-64-02-lzha`
- tpu_type: `v6e-64`
- primary_worker_log: worker5 `~/maxdiffusion/logs/tpu_20260614-143107.log`
- wandb_run: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/wmv8j3iy`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: All 16 worker processes are alive on commit `049727d7ed4135de0ca606007f6e99f370a6f842`.
- metrics/artifacts: Resolved config is `global_batch_size_to_load=64`, `global_batch_size_to_train_on=15`, `per_device_batch_size=0.234375`, `eval_every=1000`, `eval_max_batches=1`, and `checkpoint_every=1000`.
- metrics/artifacts: First train metric logged at step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`.
- metrics/artifacts: No checkpoint is expected until step `1000`; GCS currently has only the run prefix.

Analysis:
- Restart1 is past setup/model load/first JIT and is executing real training. The step-20 metric exactly matches the previous preempted full run, which is expected for the same seed and fresh-from-step-0 launch.
- There are no NaNs, OOMs, tracebacks, or TPU health warnings in the current logs.

Next:
- Continue monitoring warmup metrics and TPU health, with first critical artifact validation at checkpoint/eval step `1000`.

## 2026-06-14T14:58:00Z - restart1 warmup through step 60

Goal:
- Confirm that restart1 reproduces the healthy early loss/gradient trend after the first visible metric.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart1-20260614-142200`
- primary_worker_log: worker5 `~/maxdiffusion/logs/tpu_20260614-143107.log`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Train logs:
  - step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`
  - step `40/10000`: `loss=1.478320`, `grad_norm=389378.840`, `lr=3.90e-06`, `steps/s=0.068`
  - step `60/10000`: `loss=1.130664`, `grad_norm=3744.459`, `lr=5.90e-06`, `steps/s=0.069`
- metrics/artifacts: No checkpoint is expected until step `1000`; GCS currently has only the run prefix.

Analysis:
- The warmup trajectory exactly reproduces the previous full-run trace through step `60`. The step-40 grad-norm spike is transient and has recovered by step `60`.
- No NaNs, OOMs, tracebacks, or TPU maintenance warnings are visible.

Next:
- Continue monitoring through step `80` and then toward checkpoint/eval step `1000`.

## 2026-06-14T15:03:00Z - restart1 passes previous preemption point

Goal:
- Check whether restart1 reaches the last metric from the previous preempted run while TPU health remains good.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart1-20260614-142200`
- primary_worker_log: worker5 `~/maxdiffusion/logs/tpu_20260614-143107.log`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Train logs:
  - step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`
  - step `40/10000`: `loss=1.478320`, `grad_norm=389378.840`, `lr=3.90e-06`, `steps/s=0.068`
  - step `60/10000`: `loss=1.130664`, `grad_norm=3744.459`, `lr=5.90e-06`, `steps/s=0.069`
  - step `80/10000`: `loss=0.612598`, `grad_norm=749.494`, `lr=7.90e-06`, `steps/s=0.068`
- metrics/artifacts: No checkpoint is expected until step `1000`.

Analysis:
- Restart1 has reached the previous run's last durable metric while retaining `HEALTHY` TPU status. The earlier `UNHEALTHY_MAINTENANCE` condition has not recurred.
- Loss and grad-norm behavior match the known-good preemption-interrupted trajectory.

Next:
- Continue lower-frequency monitoring toward step `1000`, where eval and the first adapter checkpoint should be written and inspected.

## 2026-06-14T15:05:00Z - restart1 maintenance warning

Goal:
- Track TPU health after restart1 passed the previous run's last durable metric.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart1-20260614-142200`
- primary_worker_log: worker5 `~/maxdiffusion/logs/tpu_20260614-143107.log`

Result:
- status: running_with_maintenance_warning
- metrics/artifacts: At `2026-06-14T15:05:23Z`, the TPU was still `READY`, but health reported `UNHEALTHY_MAINTENANCE`.
- metrics/artifacts: Worker5 process remained alive; latest train metric was still step `80/10000`.
- metrics/artifacts: No checkpoint has been written because the run has not reached step `1000`.

Analysis:
- This is an infrastructure maintenance signal, not a training-code failure. It matches the warning pattern that preceded the earlier preemption.

Next:
- Tighten monitoring. If the TPU preempts before step `1000`, relaunch a fresh run from scratch because no checkpoint exists yet.

## 2026-06-14T15:11:00Z - restart1 preempted before first checkpoint

Goal:
- Determine whether the maintenance warning resolved or terminated restart1.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart1-20260614-142200`
- tpu_name: `v6-64-02-lzha`

Result:
- status: preempted
- metrics/artifacts: At `2026-06-14T15:09:21Z`, SSH reported terminal TPU state `PREEMPTED`.
- metrics/artifacts: At `2026-06-14T15:10:56Z`, `gcloud compute tpus tpu-vm describe` reported `state: PREEMPTED`.
- metrics/artifacts: Restart1 reached step `80/10000`; no checkpoint was written because the first checkpoint/eval is scheduled at step `1000`.
- metrics/artifacts: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs15-restart1-20260614-142200/` contains only the run prefix.

Analysis:
- This is a second infrastructure maintenance/preemption event before step `1000`, not a training-code failure. The loss curve was healthy through the last durable metric.
- With no checkpoint available, the next run must start from step `0`.

Next:
- Relaunch the same validated gbs15 full run as `restart2` on a fresh v6e-64 slice.

## 2026-06-14T15:14:00Z - full run restart2 queued

Goal:
- Relaunch after restart1 preempted before the first checkpoint.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart2-20260614-151300`
- tpu_name: `v6-64-02-lzha`
- command: `tpu watch v6 --force -n 64 ... PER_DEVICE_BATCH_SIZE=0.234375 LOG_PERIOD=20 CHECKPOINT_EVERY=1000 EVAL_EVERY=1000 EVAL_MAX_BATCHES=1 bash bash_scripts/train_wan_side_adapter.sh`
- local_watcher_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs15-restart2-20260614-151300.log`

Result:
- status: queued
- metrics/artifacts: The watcher deleted the preempted TPU and stale queued resource, then submitted a fresh queued resource at `2026-06-14T15:13:44Z`.
- metrics/artifacts: Latest observed queued-resource state is `WAITING_FOR_RESOURCES`; TPU node is `NOT_FOUND` until allocation.

Analysis:
- Relaunch is blocked only on v6e-64 capacity. No code/config changes were made for restart2.

Next:
- Keep the watcher attached through provisioning, setup, launch, and first metric validation.

## 2026-06-14T15:40:00Z - restart2 first metric

Goal:
- Verify restart2 training after fresh v6e-64 allocation and setup.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart2-20260614-151300`
- primary_worker_log: worker8 `~/maxdiffusion/logs/tpu_20260614-152324.log`
- wandb_run: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/r9kxcubu`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Training launched on commit `4acb4b9c7fc3976a08ef0ef2863babb59d8cd1dd`.
- metrics/artifacts: Resolved config is `global_batch_size_to_load=64`, `global_batch_size_to_train_on=15`, `per_device_batch_size=0.234375`, `eval_every=1000`, `eval_max_batches=1`, and `checkpoint_every=1000`.
- metrics/artifacts: First train metric logged at step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`.
- metrics/artifacts: No checkpoint is expected until step `1000`.

Analysis:
- Restart2 is past setup/model-load/first JIT and reproduces the expected deterministic step-20 metric. No NaNs, OOMs, tracebacks, or TPU maintenance warnings are visible.

Next:
- Continue monitoring warmup metrics and TPU health toward step `1000`.

## 2026-06-14T15:51:00Z - restart2 warmup through step 60

Goal:
- Confirm restart2 reproduces the healthy early warmup curve after the first metric.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart2-20260614-151300`
- primary_worker_log: worker8 `~/maxdiffusion/logs/tpu_20260614-152324.log`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Train logs:
  - step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`
  - step `40/10000`: `loss=1.478320`, `grad_norm=389378.840`, `lr=3.90e-06`, `steps/s=0.068`
  - step `60/10000`: `loss=1.130664`, `grad_norm=3744.459`, `lr=5.90e-06`, `steps/s=0.069`
- metrics/artifacts: No checkpoint is expected until step `1000`.

Analysis:
- Restart2 matches the earlier deterministic trace through step `60`; the step-40 gradient spike is transient and recovers by step `60`.
- No NaNs, OOMs, tracebacks, or TPU maintenance warnings are visible.

Next:
- Continue monitoring through step `80`, then reduce cadence while watching for step `1000` checkpoint/eval.

## 2026-06-14T15:56:00Z - restart2 passes previous preemption point

Goal:
- Confirm restart2 reaches step `80` while TPU health remains stable.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart2-20260614-151300`
- primary_worker_log: worker8 `~/maxdiffusion/logs/tpu_20260614-152324.log`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Train logs:
  - step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`
  - step `40/10000`: `loss=1.478320`, `grad_norm=389378.840`, `lr=3.90e-06`, `steps/s=0.068`
  - step `60/10000`: `loss=1.130664`, `grad_norm=3744.459`, `lr=5.90e-06`, `steps/s=0.069`
  - step `80/10000`: `loss=0.612598`, `grad_norm=749.494`, `lr=7.90e-06`, `steps/s=0.069`
- metrics/artifacts: No checkpoint is expected until step `1000`.

Analysis:
- Restart2 has passed restart1's last durable metric without the maintenance warning recurring. Training behavior still matches the deterministic warmup trace.

Next:
- Continue lower-frequency monitoring toward checkpoint/eval step `1000`, with quick health checks for another maintenance event.

## 2026-06-14T16:00:00Z - restart2 reaches step 100

Goal:
- Verify restart2 advances beyond both previous preempted traces while TPU health remains good.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart2-20260614-151300`
- primary_worker_log: worker8 `~/maxdiffusion/logs/tpu_20260614-152324.log`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Latest train metric at step `100/10000`: `loss=0.432422`, `grad_norm=453.071`, `lr=9.90e-06`, `steps/s=0.069`.
- metrics/artifacts: No checkpoint is expected until step `1000`.

Analysis:
- Restart2 has advanced beyond the last durable metrics from both earlier full-run attempts. Loss continues downward and grad norm remains settled after the step-40 spike.

Next:
- Continue lower-frequency monitoring toward checkpoint/eval step `1000`; expected time from step `100` is roughly 3.5-4 hours at the current throughput if the TPU remains healthy.

## 2026-06-14T16:27:00Z - restart2 preempted before first checkpoint

Goal:
- Determine whether restart2 survived long enough to write a durable checkpoint.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart2-20260614-151300`
- tpu_name: `v6-64-02-lzha`

Result:
- status: preempted
- metrics/artifacts: TPU health changed to `UNHEALTHY_MAINTENANCE` at `2026-06-14T16:25:55Z`, then `gcloud compute tpus tpu-vm describe` reported `state: PREEMPTED`.
- metrics/artifacts: Latest durable train metric was step `180/10000`: `loss=0.546191`, `grad_norm=6198.543`, `lr=1.79e-05`, `steps/s=0.068`.
- metrics/artifacts: Restart2 GCS run prefix contains no checkpoint because the configured first checkpoint was step `1000`.
- metrics/artifacts: A representative adapter-only smoke checkpoint is `1.04 GiB`; with Orbax `max_to_keep=3`, checkpointing every `100` steps should remain storage-safe.

Analysis:
- This is the third maintenance/preemption before the step-1000 checkpoint target. Training behavior was valid before preemption, but no progress is durable under the current checkpoint cadence.
- To make progress under current v6 maintenance churn without changing the model, data, optimizer, batch size, eval cadence, or training objective, restart3 should use `CHECKPOINT_EVERY=100` and keep `EVAL_EVERY=1000`. This writes only adapter params/optimizer/step and keeps at most three recent checkpoints, so it should not materially threaten storage.

Next:
- Relaunch as restart3 with the same gbs15 recipe, `EVAL_EVERY=1000`, `EVAL_MAX_BATCHES=1`, and `CHECKPOINT_EVERY=100`.

## 2026-06-14T16:32:00Z - restart3 queued with checkpoint interval 100

Goal:
- Relaunch with earlier adapter checkpointing so preemptions after step `100` leave durable progress.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- tpu_name: `v6-64-02-lzha`
- command: `tpu watch v6 --force -n 64 ... CHECKPOINT_EVERY=100 EVAL_EVERY=1000 EVAL_MAX_BATCHES=1 PER_DEVICE_BATCH_SIZE=0.234375 bash bash_scripts/train_wan_side_adapter.sh`
- local_watcher_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000.log`

Result:
- status: queued
- metrics/artifacts: The watcher deleted the preempted restart2 TPU and suspended queued resource, then submitted a fresh queued resource at `2026-06-14T16:32:24Z`.

Analysis:
- Only durability cadence changed. Training objective, batch size, optimizer/lr, data, eval cadence, and adapter-only frozen-backbone setup are unchanged.

Next:
- Monitor queue/provisioning, verify training launch/config, and confirm checkpoint creation at step `100`.

## 2026-06-14T17:01:44Z - restart3 stale provisioning cleanup

Goal:
- Determine whether the restart3 queued resource progressed after the initial provisioning stall.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- tpu_name: `v6-64-02-lzha`
- queued_resource: `v6-64-02-lzha-qr`
- local_watcher_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000.log`

Result:
- status: stalled
- metrics/artifacts: The queued resource remained in `PROVISIONING` from `2026-06-14T16:32:24Z` through `2026-06-14T17:01:00Z`.
- metrics/artifacts: `gcloud compute tpus tpu-vm describe v6-64-02-lzha` still reported `NOT_FOUND`; no TPU workers or training logs exist for this attempt.
- metrics/artifacts: The local watcher process was still polling the stale QR.

Analysis:
- This is a TPU allocation/provisioning stall before setup or training launch. It does not provide any new evidence about the model, data, optimizer, or checkpoint cadence.

Next:
- Stop only the restart3 watcher, delete the stale queued resource, and requeue the same checkpoint-100 recipe under a fresh run name.

## 2026-06-14T17:05:00Z - restart3 watcher reattached

Goal:
- Avoid launching a duplicate `v6e-64` slice while the existing queued resource is still owned by GCP.

Command / Job:
- queued_resource: `v6-64-02-lzha-qr`
- attempted cleanup: `gcloud compute tpus queued-resources delete v6-64-02-lzha-qr --quiet`
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`

Result:
- status: waiting
- metrics/artifacts: The delete operation failed with `DeleteQueuedResource is not supported when state is PROVISIONING`.
- metrics/artifacts: The stale QR is still `PROVISIONING`; `v6-64-02-lzha` still has no node.

Analysis:
- Since GCP will not cancel the QR in this state, queuing a second `v6e-64` request could produce two live slices if both eventually allocate. Reattaching the watcher to the existing QR is safer.

Next:
- Restart `tpu watch` on the same QR/run name, continue monitoring, and delete or use the resource as soon as it transitions.

## 2026-06-14T17:08:45Z - restart3 QR recreated

Goal:
- Recover from the stale restart3 provisioning attempt without leaving an orphaned TPU request.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- tpu_name: `v6-64-02-lzha`
- queued_resource: `v6-64-02-lzha-qr`
- local_watcher_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000.log`

Result:
- status: queued
- metrics/artifacts: The stuck QR transitioned to `SUSPENDING, SERVICE`, then `FAILED`.
- metrics/artifacts: The watcher deleted the failed QR successfully and submitted a fresh `v6-64-02-lzha-qr` at local `10:08`.
- metrics/artifacts: No TPU node or training log existed before recreation, so no training progress was lost.

Analysis:
- The previous restart3 attempt failed inside TPU provisioning before setup. The watcher recovery path worked once GCP moved the QR to a deletable state.

Next:
- Monitor the fresh queued resource, verify setup/training launch when allocated, then confirm step-100 checkpoint creation.

## 2026-06-14T17:49:16Z - restart3 launched on v6e-64

Goal:
- Verify the fresh restart3 queued resource allocates a healthy `v6e-64` and launches the checkpoint-100 training recipe.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- tpu_name: `v6-64-02-lzha`
- tpu_type: `v6e-64`
- launch: `tpu watch v6 --force -n 64 ... CHECKPOINT_EVERY=100 EVAL_EVERY=1000 EVAL_MAX_BATCHES=1 PER_DEVICE_BATCH_SIZE=0.234375 bash bash_scripts/train_wan_side_adapter.sh`
- primary_worker: worker `13`
- primary_log: `~/maxdiffusion/logs/tpu_20260614-173628.log`
- wandb: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/gz8jzz7h`

Result:
- status: running
- metrics/artifacts: The fresh QR allocated a `READY/HEALTHY` `v6e-64`; setup completed on all 16 workers and `tpu watch` launched training successfully at local `10:36:40`.
- metrics/artifacts: Remote commit is `7ee947725f0d527f8bfb5a946b7cab4278a67af5`.
- metrics/artifacts: The resolved config uses `checkpoint_every=100`, `eval_every=1000`, `eval_max_batches=1`, `global_batch_size_to_load=64`, `global_batch_size_to_train_on=15`, and `per_device_batch_size=0.234375`.
- metrics/artifacts: Primary log confirms `trainable adapter params: 239.5M` and `frozen transformer params: 5.00B`.
- metrics/artifacts: W&B run `gz8jzz7h` is online. No train step or checkpoint has been written yet.

Analysis:
- The TPU allocation/setup path is now healthy, and the v6 run is using the intended adapter-only frozen-backbone training setup. The run is still in startup/JIT/data path before the first metric.

Next:
- Monitor for the first logged train metric and then for the step-100 adapter checkpoint in GCS.

## 2026-06-14T17:58:04Z - restart3 first metrics

Goal:
- Confirm the v6 restart3 run is actually training and reproduces the expected early curve.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- primary_worker: worker `13`
- primary_log: `~/maxdiffusion/logs/tpu_20260614-173628.log`
- wandb: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/gz8jzz7h`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Step `20/10000`: `loss=2.838281`, `grad_norm=6681.592`, `lr=1.90e-06`, `steps/s=0.025`.
- metrics/artifacts: Step `40/10000`: `loss=1.478320`, `grad_norm=389378.840`, `lr=3.90e-06`, `steps/s=0.068`.
- metrics/artifacts: No checkpoint is expected until step `100` with the restart3 cadence.

Analysis:
- The first metrics exactly match the earlier deterministic healthy runs. The step-40 gradient spike is the same transient spike already observed and recovered from in prior attempts.

Next:
- Continue monitoring through steps `60`, `80`, and especially the step-`100` adapter checkpoint.

## 2026-06-14T18:16:25Z - restart3 first durable checkpoint

Goal:
- Verify the checkpoint-100 restart cadence produces a usable adapter checkpoint before another possible v6 maintenance event.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- primary_worker: worker `13`
- primary_log: `~/maxdiffusion/logs/tpu_20260614-173628.log`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000/checkpoints`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Step `60/10000`: `loss=1.130664`, `grad_norm=3744.459`, `lr=5.90e-06`, `steps/s=0.069`.
- metrics/artifacts: Step `80/10000`: `loss=0.612598`, `grad_norm=749.494`, `lr=7.90e-06`, `steps/s=0.069`.
- metrics/artifacts: Step `100/10000`: `loss=0.432422`, `grad_norm=453.071`, `lr=9.90e-06`, `steps/s=0.068`.
- metrics/artifacts: Orbax finalized `checkpoints/100` at `2026-06-14T18:12:42Z`; `gsutil du -sh` reports `1.01 GiB` for the run checkpoint directory.

Analysis:
- The run now has durable adapter-only progress. The checkpoint size matches the storage expectation from smoke testing, so checkpointing every `100` steps is safe under the current `max_to_keep=3` policy.

Next:
- Continue monitoring post-checkpoint training, verify the run advances past step `100`, and watch for TPU maintenance/preemption or abnormal loss/gradient behavior.

## 2026-06-14T18:41:44Z - restart3 maintenance after checkpoint 200

Goal:
- Verify the run has durable progress before handling another v6 maintenance event.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- primary_worker: worker `13`
- primary_log: `~/maxdiffusion/logs/tpu_20260614-173628.log`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000/checkpoints`

Result:
- status: maintenance
- metrics/artifacts: Step `120/10000`: `loss=0.295703`, `grad_norm=24.024`, `lr=1.19e-05`, `steps/s=0.067`.
- metrics/artifacts: Step `140/10000`: `loss=0.265234`, `grad_norm=12.836`, `lr=1.39e-05`, `steps/s=0.068`.
- metrics/artifacts: Step `160/10000`: `loss=0.254932`, `grad_norm=11.610`, `lr=1.59e-05`, `steps/s=0.067`.
- metrics/artifacts: Step `180/10000`: `loss=0.546191`, `grad_norm=6198.543`, `lr=1.79e-05`, `steps/s=0.069`.
- metrics/artifacts: Step `200/10000`: `loss=0.472656`, `grad_norm=2298.336`, `lr=1.99e-05`, `steps/s=0.068`.
- metrics/artifacts: Orbax finalized checkpoint `200` at `2026-06-14T18:37:15Z`; GCS contains checkpoints `100/` and `200/`, total `2.02 GiB`.
- metrics/artifacts: TPU changed to `READY` / `UNHEALTHY_MAINTENANCE`; SSH to worker `13` began returning `Connection refused`.

Analysis:
- Restart3 solved the prior no-durable-progress issue: maintenance happened again, but checkpoint `200` is finalized and small enough for the storage budget.
- The next launch should reuse the same run name/checkpoint prefix so it restores from step `200`; using a new run name would start from scratch unless explicit restore wiring is added.

Next:
- Relaunch the same run name through `tpu watch --force`, verify restore from checkpoint `200`, and continue monitoring to the next checkpoint.

## 2026-06-14T18:44:43Z - restart3 requeued from checkpoint 200

Goal:
- Recover from the v6 maintenance/preemption after preserving checkpoint `200`.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- tpu_name: `v6-64-02-lzha`
- queued_resource: `v6-64-02-lzha-qr`
- launch: `tpu watch v6 --force -n 64 ... RUN_NAME=wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000 ...`

Result:
- status: queued
- metrics/artifacts: The TPU transitioned to `PREEMPTED` after reporting `UNHEALTHY_MAINTENANCE`.
- metrics/artifacts: The watcher deleted the preempted node and stale active queued resource.
- metrics/artifacts: A fresh `v6-64-02-lzha-qr` was submitted and is `WAITING_FOR_RESOURCES`.

Analysis:
- Reusing the same run name is intentional so the next launch uses the same checkpoint prefix and restores from step `200`.

Next:
- Wait for the replacement v6e-64, verify restore from checkpoint `200`, and confirm training resumes past step `200`.

## 2026-06-14T19:06:00Z - audit low batch-size ceiling

Goal:
- Explain why the v6e-64 side-adapter run only fit global batch `15` and correct the implementation before any further full training launch.

Change:
- Stopped/abandoned the failed post-restore `gbs15` path after it restored checkpoint `200` but failed the first resumed `jit__train_step`.
- Changed the side-adapter TPU mesh defaults from context-parallel to FSDP-parallel: `dcn_fsdp_parallelism=-1`, `dcn_context_parallelism=1`, `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`.
- Matched `../Wan2.2` CFG semantics by stopping gradients through the unconditional branch and its latent input.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- base_commit: `3d5b99536b25a022335575c18e5fda097f2fb0a8`
- implementation_commit: pending
- changed_files: `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`, `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`, worklog

Command / Job:
- failed_resume_run: `wan-side-adapter-v6e64-full-gbs15-restart3-ckpt100-20260614-163000`
- failed_wandb: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/ikvx89sc`
- failed_log: `~/maxdiffusion/logs/tpu_20260614-185447.log`

Result:
- status: failed as implemented
- metrics/artifacts: The resumed run restored adapter checkpoint `200` successfully, then failed with `RESOURCE_EXHAUSTED: RuntimeProgramAllocationFailure`, trying to reserve `29.33G` with `29.30G` reservable.
- key evidence: Worker log showed `ici_fsdp_parallelism=1`, `ici_context_parallelism=-1`, so the v6e-64 mesh was all context-parallel. Logical batch axes map to `data/fsdp`, both size `1`, meaning the model path was not using the intended batch/FSDP sharding.
- key evidence: The JAX rollout let gradients flow through `v_uncond`; the PyTorch `../Wan2.2` trainer computes that branch under `torch.no_grad()` with detached `z`.

Analysis:
- User was right to challenge the batch-size result. The previous probes did not establish a meaningful maximum batch size because the side-adapter config was not using the FSDP mesh expected for TPU training.
- The current implementation also over-retained the 25-step CFG graph relative to the PyTorch reference, increasing memory and changing the gradient path.

Next:
- Commit/push the mesh and CFG fixes, run storage-light v6e-64 probes starting at global batch `256`, and only relaunch full training after the corrected implementation compiles and completes real optimizer steps.

## 2026-06-14T19:15:36Z - corrected FSDP batch-256 compile failure

Goal:
- Test whether the corrected v6e-64 FSDP mesh can compile and run a storage-light global batch `256` side-adapter probe.

Hypothesis:
- With FSDP enabled across the v6e-64 mesh, global batch `256` should be the first serious target. If it still fails, the failure should identify a remaining memory bug rather than the previous all-context mesh bug.

Change:
- Launched a two-step probe from commit `bd7202eaab070d729f4b90905f900df4751615a6`, which contains the FSDP mesh fix and CFG stop-gradient fix.
- Used `PER_DEVICE_BATCH_SIZE=4`, `MAX_TRAIN_STEPS=2`, `CHECKPOINT_EVERY=0`, `SAVE_FINAL_CHECKPOINT=False`, `EVAL_EVERY=0`, and `WANDB_DISABLED=true` to avoid checkpoint or W&B storage while probing.
- Added a local follow-up patch that rematerializes the outer 25-step rollout loop with `jax.checkpoint(_rollout_body)`.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- base_commit: `bd7202eaab070d729f4b90905f900df4751615a6`
- implementation_commit: pending
- changed_files: `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`, worklog

Command / Job:
- run_name: `wan-side-adapter-v6e64-probe-gbs256-fsdp-20260614-190700`
- tpu_name: `v6-64-02-lzha`
- tpu_type: `v6e-64`
- launch: `tpu watch v6 --force -n 64 ... PER_DEVICE_BATCH_SIZE=4 MAX_TRAIN_STEPS=2 CHECKPOINT_EVERY=0 SAVE_FINAL_CHECKPOINT=False EVAL_EVERY=0 WANDB_DISABLED=true bash bash_scripts/train_wan_side_adapter.sh`
- log: `~/maxdiffusion/logs/tpu_20260614-190725.log`

Result:
- status: failed
- metrics/artifacts: Worker log confirms remote `HEAD=bd7202e`, `global_batch_size_to_train_on=256`, `per_device_batch_size=4.0`, `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `dcn_fsdp_parallelism=-1`, and `dcn_context_parallelism=1`.
- metrics/artifacts: Adapter/backbone split remained correct at `239.5M` trainable adapter params and `5.00B` frozen transformer params.
- key evidence: Compile failed with `CompileTimeHbmOom`: `Used 144.33G of 31.25G hbm`, exceeded by `113.08G`.
- key evidence: Largest allocations included repeated `f32[25,4,540,3072]` tensors, showing reverse-mode was retaining per-denoising-step residuals across the full rollout.
- cleanup: No active training processes remained on the workers after the failure check.

Analysis:
- FSDP is now wired correctly; the corrected config reached the intended batch-256 logical shape and the mesh values are no longer the old all-context values.
- The remaining batch-256 failure points at outer rollout activation retention. That is distinct from model-weight sharding and should be addressed by rematerializing the 25-step rollout body, matching the PyTorch reference's checkpointed differentiable sampling loop.

Next:
- Commit/push the rollout rematerialization patch, relaunch the storage-light batch-256 probe from that exact commit, and verify it reaches real optimizer steps before considering larger batch sizes or full training.

## 2026-06-14T19:29:41Z - remat FSDP batch-256 probe passed

Goal:
- Verify global batch `256` fits with the corrected FSDP mesh and rollout rematerialization.

Hypothesis:
- The previous corrected-FSDP OOM was caused by retained 25-step rollout residuals. Rematerializing the outer rollout body should allow global batch `256` to compile and execute real optimizer steps.

Change:
- Committed and pushed rollout rematerialization in `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `30347d4c934e00b718200ed0888808706d24fc9a`
- push/pull: pushed to origin; `tpu watch` pulled the commit on all workers
- changed_files: `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`, worklog
- remote_commit/status: worker 0 reported `HEAD=30347d4c934e00b718200ed0888808706d24fc9a` and a clean branch status

Command / Job:
- run_name: `wan-side-adapter-v6e64-probe-gbs256-fsdp-remat-20260614-191642`
- tpu_name: `v6-64-02-lzha`
- tpu_type: `v6e-64`
- command: `PER_DEVICE_BATCH_SIZE=4 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=256 GLOBAL_BATCH_SIZE_TO_LOAD=256 MAX_TRAIN_STEPS=2 CHECKPOINT_EVERY=0 SAVE_FINAL_CHECKPOINT=False EVAL_EVERY=0 WANDB_DISABLED=true bash bash_scripts/train_wan_side_adapter.sh`
- log: `~/maxdiffusion/logs/tpu_20260614-191720.log`

Result:
- status: passed
- metrics/artifacts: Config confirmed `global_batch_size_to_train_on=256`, `global_batch_size_to_load=256`, `per_device_batch_size=4.0`, `total_train_batch_size=256.0`, `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `dcn_fsdp_parallelism=-1`, and `dcn_context_parallelism=1`.
- metrics/artifacts: Adapter/backbone split remained `239.5M` trainable adapter params and `5.00B` frozen transformer params.
- metrics/artifacts: Step `1/2`: `loss=3.296875`, `grad_norm=10573147136.000`, `lr=5.00e-05`, `steps/s=0.004`.
- metrics/artifacts: Step `2/2`: `loss=3.390625`, `grad_norm=90701496.000`, `lr=5.00e-05`, `steps/s=0.004`.
- cleanup: Checkpointing and final checkpoint saving were disabled; all workers had no remaining training process after completion; TPU remained `READY/HEALTHY`.

Analysis:
- User's expected lower bound is now satisfied: global batch `256` does fit on v6e-64 when FSDP and rollout rematerialization are both correct.
- The earlier batch ceiling was not a hardware limit. It was the combination of all-context mesh configuration and retained rollout residuals.
- Throughput is slow at this tiny two-step probe because it includes first compile and short-run overhead; a longer run is needed to estimate steady-state speed.

Next:
- Probe larger global batches with the same storage-light settings, starting at global batch `512`, to find the largest fitting batch before launching full training.

## 2026-06-14T19:36:00Z - batch-512 probe interrupted by maintenance

Goal:
- Test whether global batch `512` fits with corrected FSDP and rollout rematerialization.

Change:
- Launched a one-step storage-light global batch `512` probe with `PER_DEVICE_BATCH_SIZE=8`, `GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512`, `GLOBAL_BATCH_SIZE_TO_LOAD=512`, `MAX_TRAIN_STEPS=1`, `CHECKPOINT_EVERY=0`, `SAVE_FINAL_CHECKPOINT=False`, `EVAL_EVERY=0`, and `WANDB_DISABLED=true`.
- First attempted `tpu watch --force`; it reported successful handoff but produced no remote trainer log or active process.
- Relaunched directly on all 16 workers with explicit `nohup` and per-worker log path `logs/tpu_direct_wan-side-adapter-v6e64-probe-gbs512-direct-20260614-193200.log`.

Version Control:
- implementation_commit: `30347d4c934e00b718200ed0888808706d24fc9a`
- remote_commit/status: all direct-launch workers printed `head=30347d4c934e00b718200ed0888808706d24fc9a`

Command / Job:
- run_name: `wan-side-adapter-v6e64-probe-gbs512-direct-20260614-193200`
- tpu_name: `v6-64-02-lzha`
- tpu_type: `v6e-64`
- command: `PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=1 CHECKPOINT_EVERY=0 SAVE_FINAL_CHECKPOINT=False EVAL_EVERY=0 WANDB_DISABLED=true bash bash_scripts/train_wan_side_adapter.sh`
- log: `~/maxdiffusion/logs/tpu_direct_wan-side-adapter-v6e64-probe-gbs512-direct-20260614-193200.log`

Result:
- status: interrupted
- metrics/artifacts: Worker 0 confirmed `global_batch_size_to_train_on=512`, `global_batch_size_to_load=512`, `per_device_batch_size=8.0`, `total_train_batch_size=512.0`, `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `dcn_fsdp_parallelism=-1`, and `dcn_context_parallelism=1`.
- metrics/artifacts: Adapter/backbone split remained `239.5M` trainable adapter params and `5.00B` frozen transformer params.
- metrics/artifacts: Before a train step or compile OOM was logged, the TPU reported maintenance and then became `PREEMPTED`; subsequent SSH failed with `This TPU has terminal state "PREEMPTED"`.

Analysis:
- This does not establish whether batch `512` fits. The interruption happened during the first-step window and was caused by v6 maintenance/preemption, not a logged XLA or runtime memory failure.

Next:
- Requeue a fresh `v6e-64` through `tpu watch --force` and retry the global batch `512` probe from the same commit.

## 2026-06-14T20:43:40Z - batch-size audit cleanup

Goal:
- Finish the Wan v6 cleanup after the global batch `512` retry was blocked by v6 maintenance/provisioning instability.

Change:
- Stopped the local `tpu watch` retry for `v6-64-04-lzha` after it remained in provisioning without starting a trainer.
- Polled the queued resource until it moved out of `PROVISIONING`, then deleted it when the API allowed deletion.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `30347d4c934e00b718200ed0888808706d24fc9a`
- changed_files: worklog only

Command / Job:
- cleanup_target: `v6-64-04-lzha`
- queued_resource: `v6-64-04-lzha-qr`
- cleanup: `gcloud alpha compute tpus queued-resources delete v6-64-04-lzha-qr --zone us-east1-d --project mae-irom-lab-guided-data --quiet`

Result:
- status: cleaned
- metrics/artifacts: `v6-64-04-lzha` node is `NOT_FOUND`.
- metrics/artifacts: `v6-64-04-lzha-qr` queued resource is `NOT_FOUND`.
- metrics/artifacts: No local Wan `tpu watch` processes remain; the only observed local watcher was unrelated `ego-lap` work on `v6-64-01-lihan`.

Analysis:
- The corrected implementation uses the FSDP mesh settings needed for v6e-64: `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `dcn_fsdp_parallelism=-1`, and `dcn_context_parallelism=1`.
- The global batch `256` probe is the current confirmed fitting point. It compiled and completed two optimizer steps with checkpointing disabled, so it did not consume GCS storage.
- Global batch `512` remains inconclusive. The direct launch confirmed the intended config and parameter split, but v6 maintenance/preemption happened before a train step or memory failure could be logged.

Next:
- Use global batch `256` for the next full training launch if we need to proceed immediately, or retry storage-light probes above `256` when v6 provisioning is stable enough to establish the true maximum fitting batch.

## 2026-06-14T22:05:52Z - side-adapter parity audit and recipe fixes

Goal:
- Re-audit the MaxDiffusion side-adapter implementation against `../Wan2.2` before retrying global batch `512`.

Change:
- Changed side-adapter `FP32LayerNorm` eps from `1e-6` to `1e-5` to match PyTorch `nn.LayerNorm` defaults in `models/action_conditioned_wan.py`.
- Changed `adam_weight_decay` from `0.0` to `1.e-2` to match the PyTorch DROID distributed script default.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- base_commit: `3b126904f27b93456d63262c88af66801f3ca376`
- implementation_commit: pending
- changed_files: `src/maxdiffusion/models/wan/side_adapter_wan.py`, `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`, worklog

Validation:
- `python3 -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py src/maxdiffusion/train_wan.py`
- `bash -n bash_scripts/train_wan_side_adapter.sh bash_scripts/della_convert_wan_side_adapter_droid.sh bash_scripts/local_a1001_continue_wan_side_adapter_tfrecords.sh`
- Static config checks confirmed `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `dcn_fsdp_parallelism=-1`, `dcn_context_parallelism=1`, and v6 GCS train/val roots.

Result:
- status: patch pending commit
- model parity: action delta encoding, action token pooling, side residual addition after selected blocks, zero-initialized residual output, CFG stop-gradient through unconditional branch, shifted sigma grid, per-token frame-0 timestep zeroing, and first-frame pinning match the PyTorch side-adapter rollout.
- parameter ownership: the train state optimizer `params` tree contains only adapter params. Frozen WAN transformer params are stored separately as `transformer_params` and are not an argument to `nnx.value_and_grad`; checkpoint saves only adapter params and optimizer state.
- sharding: v6 mesh config is FSDP-dominant. Frozen transformer weights are loaded with logical partition specs onto the mesh via `create_sharded_logical_transformer` and `device_put_replicated(..., sharding)`. Adapter params/optimizer state are intentionally `P()` replicated because the adapter is small and has action-length axes that should not inherit WAN context sharding.
- data parity: source Della cache sample `ep6434_v0_s00048` has `z_I0`/`z_video` dtype `torch.float16` and `actions` dtype `float32`; GCS TFRecord train shard `443` record `508` for ordinal `907772` has identical byte counts and matching min/max/std for `z_i0`, `z_video`, and `actions`.
- dataset completeness: train summary reports `1,440,554` examples across `704` shards; val summary reports `14,636` examples across `8` shards.

Analysis:
- The implementation structure is aligned with the original PyTorch side-adapter code. The two mismatches found were recipe-level numeric defaults, not shape or data-loader failures.
- The next global batch `512` probe should run from the patched commit so the memory test reflects the corrected recipe.

Next:
- Commit and push the audit fixes, then run a storage-light global batch `512` v6e-64 probe with checkpointing and final save disabled.

## 2026-06-14T22:08:13Z - launch v6e-64 global batch 512 recipe-fixed probe

Goal:
- Prove whether the corrected side-adapter training stack fits and runs at global batch `512` on a v6e-64 TPU slice.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `83c1a9ce54e7e2f8906205e68f145dfea68d79e0`
- push/pull: pushed to `origin/codex/wan-ti2v-side-adapter-20260613-073227`

Command / Job:
- run_name: `wan-side-adapter-v6e64-probe-gbs512-recipefix-20260614-220813`
- tpu_name: `v6-64-05-lzha`
- accelerator: `v6e-64`
- command: `tpu watch v6 -n 64 --setup-cmd "... bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" 83c1a9ce54e7e2f8906205e68f145dfea68d79e0 RUN_NAME=wan-side-adapter-v6e64-probe-gbs512-recipefix-20260614-220813 WANDB_DISABLED=true PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=2 CHECKPOINT_EVERY=0 SAVE_FINAL_CHECKPOINT=False EVAL_EVERY=0 LOG_PERIOD=1 bash bash_scripts/train_wan_side_adapter.sh`
- local_log: `logs/tpu_watch_wan-side-adapter-v6e64-probe-gbs512-recipefix-20260614-220813.log`
- expected artifacts: worker logs only; checkpointing and final save intentionally disabled for the storage-light fit probe.

Acceptance Criteria:
- Worker log confirms commit `83c1a9ce54e7e2f8906205e68f145dfea68d79e0`, v6e-64 with 64 devices, per-device batch `8`, total/global batch `512`, FSDP `-1`, context parallelism `1`, adapter-only trainable params, and frozen WAN params.
- Training reaches at least one completed optimizer step without OOM, `RESOURCE_EXHAUSTED`, NaN, data parse failure, or pre-step TPU maintenance interruption.

Result:
- status: pending launch

## 2026-06-15T02:02:00Z - relaunch full training after setup checkout failure

Goal:
- Relaunch the global batch `512` full training run on the already allocated `v6-64-06-lzha` slice after the first setup attempt failed before training.

Result:
- status: first full launch failed before setup completed
- key evidence: the TPU helper cloned `origin/main` before running setup; `origin/main` does not have the branch-only `bash_scripts/setup.sh`, so all workers failed with `bash: maxdiffusion/bash_scripts/setup.sh: No such file or directory`.
- remote worker 0 had no `train_wan`/Python training process after the failed setup. The TPU remained `READY` and task-owned.

Analysis:
- This was a launch-order issue, not a model, data, memory, or batch-size failure. No training process started and no checkpoint artifacts were written.
- The relaunch should make the setup command explicitly fetch and checkout the target commit before invoking `bash_scripts/setup.sh`.

Next:
- Relaunch with `tpu watch --force` on `v6-64-06-lzha` using setup command `git fetch origin codex/wan-ti2v-side-adapter-20260613-073227 && git checkout --detach b4e26d4edf2492701dff9cb89dd983c7c00bbf89 && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu`.

## 2026-06-14T22:15:49Z - relaunch batch 512 after setup cwd failure

Goal:
- Relaunch the same global batch `512` v6e-64 fit probe after a setup-command path failure.

Result:
- status: first launch failed before training
- key evidence: `tpu watch` cloned `maxdiffusion` on all workers, then ran the setup command from `$HOME`; every worker failed with `bash: bash_scripts/setup.sh: No such file or directory` and `Setup failed (rc=127)`.

Analysis:
- This is a launcher current-working-directory issue, not a model/data/batch-size result. No training process started and no checkpoints were written.

Next:
- Stop the local watcher and relaunch against the existing `READY` TPU with setup command `cd maxdiffusion && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu`.

## 2026-06-14T22:17:26Z - relaunch batch 512 with absolute setup path

Goal:
- Run the recipe-fixed global batch `512` v6e-64 fit probe after setup-command cwd failures.

Change:
- Relaunched on the same task-owned v6e-64 slice with setup invoked from `$HOME` via `bash maxdiffusion/bash_scripts/setup.sh MODE=stable DEVICE=tpu`, avoiding the setup script's internal `cd maxdiffusion` conflict.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- local_head_at_record: `db24b7d5934f0e868a775e94af36141c5e01b4bd`
- remote_training_commit: `83c1a9ce54e7e2f8906205e68f145dfea68d79e0`
- push/pull: training commit was already pushed and checked out detached on TPU workers

Command / Job:
- run_name: `wan-side-adapter-v6e64-probe-gbs512-recipefix-abssetup-20260614-221726`
- tpu_name: `v6-64-05-lzha`
- queued_resource: `v6-64-05-lzha-qr`
- accelerator: `v6e-64`
- remote_log: `/home/lzha/maxdiffusion/logs/tpu_20260614-222052.log`
- archived_log: `/home/lzha/code/shared_artifacts/wan-ti2v-side-adapter/wan-side-adapter-v6e64-probe-gbs512-recipefix-abssetup-20260614-221726/worker14_tpu_20260614-222052.log`
- command: `python src/maxdiffusion/train_wan.py src/maxdiffusion/configs/base_wan_5b_side_adapter.yml run_name=wan-side-adapter-v6e64-probe-gbs512-recipefix-abssetup-20260614-221726 pretrained_model_name_or_path=Wan-AI/Wan2.2-TI2V-5B-Diffusers train_data_dir=gs://v6_east1d/datasets/droid_wan_side_adapter/train eval_data_dir=gs://v6_east1d/datasets/droid_wan_side_adapter/val output_dir=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter base_output_directory=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter max_train_steps=2 checkpoint_every=0 eval_every=0 log_period=1 save_final_checkpoint=False per_device_batch_size=8 global_batch_size_to_train_on=512 global_batch_size_to_load=512 hardware=tpu`

Result:
- status: passed
- metrics/artifacts: Worker 14 confirmed `HEAD=83c1a9c`, `Found 64 devices`, `global_batch_size_to_train_on=512`, `global_batch_size_to_load=512`, `per_device_batch_size=8.0`, and `total_train_batch_size=512.0`.
- metrics/artifacts: Parallelism config was `mesh_axes=['data', 'fsdp', 'context', 'tensor']`, `data_sharding=(('data', 'fsdp', 'context', 'tensor'),)`, `ici_data_parallelism=1`, `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `ici_tensor_parallelism=1`, `dcn_data_parallelism=1`, `dcn_fsdp_parallelism=-1`, `dcn_context_parallelism=1`, and `dcn_tensor_parallelism=1`.
- metrics/artifacts: Adapter/backbone split remained `239.5M` trainable adapter params and `5.00B` frozen transformer params.
- metrics/artifacts: Step `1/2`: `loss=3.250000`, `grad_norm=3877746688.000`, `lr=5.00e-05`, `steps/s=0.003`.
- metrics/artifacts: Step `2/2`: `loss=2.687500`, `grad_norm=590872576.000`, `lr=5.00e-05`, `steps/s=0.003`.
- metrics/artifacts: No `Traceback`, `RESOURCE_EXHAUSTED`, OOM, or NaN was found in the archived worker-14 train log. The CUDA init warning is expected backend probing noise on TPU.
- storage: GCS run root contained only empty placeholder directories because checkpointing and final save were disabled for this storage-light fit probe.
- cleanup: All 16 TPU workers reported no remaining `train_wan`/Python process after completion. Deleted TPU node `v6-64-05-lzha`; deleted queued resource `v6-64-05-lzha-qr`; both verify as `NOT_FOUND`.

Analysis:
- Global batch `512` fits and completes optimizer updates on a v6e-64 slice with the recipe-fixed side-adapter implementation.
- The current parallelism is FSDP-only across ICI/DCN with no context or tensor parallelism: FSDP axes are auto-sized by `-1`, while `data`, `context`, and `tensor` are all size `1` in the config. Data batches are sharded over the full `('data', 'fsdp', 'context', 'tensor')` mesh.
- The storage-light probe produced no checkpoint artifacts by design; the archived log is the durable evidence for this fit test.

Next:
- Use these settings for the next full training launch unless a larger-batch probe is needed: `per_device_batch_size=8`, `global_batch_size_to_train_on=512`, `global_batch_size_to_load=512`, `ici_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `dcn_fsdp_parallelism=-1`, `dcn_context_parallelism=1`.

## 2026-06-15T01:52:34Z - launch v6e-64 global batch 512 full training

Goal:
- Start the real Wan2.2 TI2V 5B DROID side-adapter training run on a task-owned v6e-64 TPU slice using the batch size proven by the storage-light fit probe.

Hypothesis:
- The recipe-fixed MaxDiffusion implementation can run the full DROID side-adapter training with `global_batch_size_to_train_on=512` on pure ICI FSDP and produce stable early train metrics plus bounded adapter checkpoints.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `0a91485ce04485229e396cdc62841875a582579f`
- push/pull: local branch is clean and pushed to `origin/codex/wan-ti2v-side-adapter-20260613-073227`
- changed_files: worklog entry only for this launch

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-20260615-015234`
- tpu_name: `v6-64-06-lzha`
- accelerator: `v6e-64`
- command: `tpu watch v6 -n 64 --setup-cmd "bash maxdiffusion/bash_scripts/setup.sh MODE=stable DEVICE=tpu" 0a91485ce04485229e396cdc62841875a582579f RUN_NAME=wan-side-adapter-v6e64-full-gbs512-20260615-015234 WANDB_PROJECT=maxdiffusion-wan-side-adapter PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=10000 CHECKPOINT_EVERY=1000 EVAL_EVERY=1000 EVAL_MAX_BATCHES=4 LOG_PERIOD=10 SAVE_FINAL_CHECKPOINT=False bash bash_scripts/train_wan_side_adapter.sh`
- train_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
- eval_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- output_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter`
- local_watch_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs512-20260615-015234.log`

Acceptance Criteria:
- Worker log confirms commit `0a91485ce04485229e396cdc62841875a582579f`, v6e-64 with 64 devices, global batch `512`, pure FSDP mesh, adapter-only trainable params, frozen WAN params, and GCS train/val roots.
- Training reaches stable early optimizer steps without OOM, `RESOURCE_EXHAUSTED`, NaN, data parse failure, or TPU maintenance before first metrics.
- Checkpoint writes are bounded by Orbax `max_to_keep=3` plus `checkpoint_keep_period=5000`; `SAVE_FINAL_CHECKPOINT=False` avoids a duplicate step-10000 save because periodic checkpointing already saves that step.

Result:
- status: pending launch

## 2026-06-15T02:17:30Z - retry1 batch 512 full run startup verified

Goal:
- Verify that the retry1 full training launch is running the intended immutable commit, data roots, batch size, sharding recipe, and adapter-only frozen-backbone parameter split before waiting for the first train metric.

Hypothesis:
- The full run is in normal first-batch/JAX compile startup; the primary log has confirmed setup and W&B, but step metrics are not expected until after compilation plus `LOG_PERIOD=10` completed optimizer steps.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- local_head: `0ce8313df3a29b1d22ba37aa69057457efb16c81`
- remote_training_commit: `0ce8313df3a29b1d22ba37aa69057457efb16c81`
- push/pull: branch was pushed; TPU workers checked out the commit detached before running `bash_scripts/setup.sh`
- changed_files: worklog entry only

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-retry1-20260615-020300`
- tpu_name: `v6-64-06-lzha`
- queued_resource: `v6-64-06-lzha-qr`
- accelerator: `v6e-64`
- local_watch_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs512-retry1-20260615-020300.log`
- primary_worker: worker `13` (`t1v-n-1305de00-w-13`, JAX process `0`)
- primary_log: `~/maxdiffusion/logs/tpu_20260615-020922.log`
- wandb: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/pvlzlovx`
- launch: `tpu watch --force v6 -n 64 --setup-cmd "git fetch origin codex/wan-ti2v-side-adapter-20260613-073227 && git checkout --detach 0ce8313df3a29b1d22ba37aa69057457efb16c81 && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" 0ce8313df3a29b1d22ba37aa69057457efb16c81 RUN_NAME=wan-side-adapter-v6e64-full-gbs512-retry1-20260615-020300 WANDB_PROJECT=maxdiffusion-wan-side-adapter PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=10000 CHECKPOINT_EVERY=1000 EVAL_EVERY=1000 EVAL_MAX_BATCHES=4 LOG_PERIOD=10 SAVE_FINAL_CHECKPOINT=False bash bash_scripts/train_wan_side_adapter.sh`
- train_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
- eval_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- output_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: All 16 workers have active `train_wan.py` processes with about `65-70 GiB` RSS each and host memory headroom around `596 GiB`.
- metrics/artifacts: Config confirms `global_batch_size_to_train_on=512`, `global_batch_size_to_load=512`, `per_device_batch_size=8.0`, `total_train_batch_size=512.0`, `ici_fsdp_parallelism=-1`, `dcn_fsdp_parallelism=-1`, `ici_context_parallelism=1`, `dcn_context_parallelism=1`, `ici_tensor_parallelism=1`, and `dcn_tensor_parallelism=1`.
- metrics/artifacts: Primary log confirms `trainable adapter params: 239.5M` and `frozen transformer params: 5.00B`.
- metrics/artifacts: Primary log confirms W&B run `pvlzlovx` is online for `wan-side-adapter-v6e64-full-gbs512-retry1-20260615-020300`.
- metrics/artifacts: No train metric or checkpoint has been written yet; GCS run prefix contains only the run directory and empty `checkpoints/` directory.
- setup note: the retry setup encountered an Ubuntu unattended-upgrade dpkg lock on worker 2; after verifying it blocked only setup, the unattended-upgrade PIDs were killed and setup completed. Training then launched on all workers.

Analysis:
- This verifies the full run is using the intended adapter-only/frozen-backbone implementation and data path. The lack of step logs at this point is not itself abnormal because the primary process has entered the train loop after W&B setup and the first JAX compilation plus ten large-batch optimizer steps can take substantially longer than the two-step probe.
- The current checkpoint cadence (`1000`) is storage-light but less preemption-resilient than the previous gbs15 checkpoint-100 run. This was intentionally kept for the user-requested batch-512 full run and to limit storage writes.

Next:
- Continue monitoring worker `13` for the first `step 10/10000` metric, TPU health, and GCS checkpoint artifacts. If the process exits, logs an OOM/`RESOURCE_EXHAUSTED`/NaN/traceback, or stalls with no CPU activity, diagnose before relaunching.

## 2026-06-15T02:38:30Z - relaunch batch 512 with checkpoint interval 100

Goal:
- Keep the validated global batch `512` training run, but make progress durable under v6 maintenance risk before the first checkpoint.

Hypothesis:
- Step-10 throughput from the retry1 run is slow enough that `CHECKPOINT_EVERY=1000` would leave roughly a day of work uncheckpointed, while `CHECKPOINT_EVERY=100` should remain storage-safe because adapter checkpoints were about `1.0 GiB` and Orbax keeps only a small bounded set.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `cb8e4e498a00d34b6888b16a632af750e9a948a5`
- push/pull: pushed to `origin/codex/wan-ti2v-side-adapter-20260613-073227`; TPU workers checked out the commit detached
- changed_files: worklog entries only since training commit `0ce8313`; model/data/training code unchanged

Command / Job:
- stopped_run_name: `wan-side-adapter-v6e64-full-gbs512-retry1-20260615-020300`
- stopped_run_primary_log: `~/maxdiffusion/logs/tpu_20260615-020922.log`
- stopped_run_wandb: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/pvlzlovx`
- new_run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800`
- tpu_name: `v6-64-06-lzha`
- queued_resource: `v6-64-06-lzha-qr`
- accelerator: `v6e-64`
- local_watch_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800.log`
- launch: `tpu watch --force v6 -n 64 --setup-cmd "git fetch origin codex/wan-ti2v-side-adapter-20260613-073227 && git checkout --detach cb8e4e498a00d34b6888b16a632af750e9a948a5 && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" cb8e4e498a00d34b6888b16a632af750e9a948a5 RUN_NAME=wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800 WANDB_PROJECT=maxdiffusion-wan-side-adapter PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=10000 CHECKPOINT_EVERY=100 EVAL_EVERY=1000 EVAL_MAX_BATCHES=4 LOG_PERIOD=10 SAVE_FINAL_CHECKPOINT=False bash bash_scripts/train_wan_side_adapter.sh`
- train_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
- eval_data_dir: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- output_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter`

Result:
- status: running
- metrics/artifacts: Retry1 reached `step 10/10000` with `loss=3.273438`, `grad_norm=7666088704.000`, `lr=9.00e-07`, and `steps/s=0.010`.
- metrics/artifacts: Retry1 had no `Traceback`, `RESOURCE_EXHAUSTED`, OOM, or NaN; TPU state was `READY`, health `HEALTHY`.
- metrics/artifacts: Retry1 GCS run prefix was still `0 B` because checkpointing was set to step `1000`.
- metrics/artifacts: The retry1 training was stopped intentionally by matching the run-specific Python entrypoint; a first stop attempt matched its own remote shell and did not stop training, then a non-self-matching `[t]rain_wan.py` pattern left only defunct zero-RSS Python zombies before relaunch.
- metrics/artifacts: New ckpt100 launch succeeded at `2026-06-15T02:38:21Z`.
- metrics/artifacts: Direct worker verification confirms commit `cb8e4e4`, run name `wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800`, `checkpoint_every=100`, `global_batch_size_to_train_on=512`, and `per_device_batch_size=8.0`.
- metrics/artifacts: All 16 workers have active new `train_wan.py` processes; TPU state remains `READY`, health `HEALTHY`.
- storage: New GCS run prefix currently contains no checkpoint payload, as expected before step `100`.

Analysis:
- The batch-512 training path is valid and has now produced real optimizer metrics. The gradient norm is very large, but this matches the earlier batch-512 probe scale and is not accompanied by NaN/OOM; continue watching trend at steps `20`, `30`, etc.
- At `0.010 steps/s`, checkpoint step `1000` would be too far away for current v6 maintenance behavior. Relaunching at checkpoint interval `100` sacrifices only the first ten uncheckpointed steps and keeps expected checkpoint storage bounded to a few GiB.

Next:
- Monitor the ckpt100 run for adapter/frozen-param confirmation, W&B run id, first `step 10/10000` metric, and then the step-`100` checkpoint in GCS.

## 2026-06-15T03:00:00Z - checkpoint-100 run first metric

Goal:
- Verify the checkpoint-100 relaunch is training with the same batch-512 behavior as the validated retry1 run.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800`
- tpu_name: `v6-64-06-lzha`
- primary_worker: worker `13` (`t1v-n-1305de00-w-13`, JAX process `0`)
- primary_log: `~/maxdiffusion/logs/tpu_20260615-023808.log`
- wandb: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/6u6pl1hn`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800/checkpoints`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Primary log confirms adapter/backbone split: `239.5M` trainable adapter params and `5.00B` frozen transformer params.
- metrics/artifacts: Step `10/10000`: `loss=3.273438`, `grad_norm=7666088704.000`, `lr=9.00e-07`, `steps/s=0.010`.
- metrics/artifacts: No `Traceback`, `ERROR`, `RESOURCE_EXHAUSTED`, OOM, killed process, or NaN signature was visible in the primary log.
- metrics/artifacts: GCS run prefix is still `0 B`; first checkpoint is expected at step `100`.

Analysis:
- Step-10 metrics exactly match retry1, so the checkpoint-100 relaunch preserved the batch-512 training behavior. Continue monitoring trend; the high early grad norm is consistent with the previous batch-512 run/probe and not yet a failure signal.
- At current throughput, step `100` should be on the order of a few hours from launch, which is acceptable for v6 maintenance risk and storage.

Next:
- Continue monitoring step `20`, subsequent loss/gradient trend, TPU health, and step-`100` checkpoint finalization.

## 2026-06-15T03:15:30Z - checkpoint-100 run early trend

Goal:
- Inspect the early loss/gradient trend before the first checkpoint.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800`
- primary_worker: worker `13`
- primary_log: `~/maxdiffusion/logs/tpu_20260615-023808.log`
- wandb: `https://wandb.ai/lihanzha/maxdiffusion-wan-side-adapter/runs/6u6pl1hn`

Result:
- status: running
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Step `10/10000`: `loss=3.273438`, `grad_norm=7666088704.000`, `lr=9.00e-07`, `steps/s=0.010`.
- metrics/artifacts: Step `20/10000`: `loss=3.171875`, `grad_norm=41297142732.800`, `lr=1.90e-06`, `steps/s=0.028`.
- metrics/artifacts: Step `30/10000`: `loss=2.476562`, `grad_norm=66094817894.400`, `lr=2.90e-06`, `steps/s=0.028`.
- metrics/artifacts: Step `40/10000`: `loss=2.003906`, `grad_norm=7660980044.800`, `lr=3.90e-06`, `steps/s=0.028`.
- metrics/artifacts: GCS run prefix is still `0 B`; first checkpoint is expected at step `100`.

Analysis:
- Loss is decreasing through step `40`, and the high gradient norms at steps `20-30` have already dropped back by step `40`. This resembles the earlier transient-gradient behavior observed in lower-batch full runs and does not show NaN/divergence.
- Post-startup throughput is about `0.028 steps/s`, so the step-`100` checkpoint is expected roughly 35-40 minutes after this check if the TPU remains healthy.

Next:
- Continue monitoring toward step `100`; verify checkpoint finalization and storage size once written.

## 2026-06-15T03:36:30Z - checkpoint-100 run preempted before first checkpoint

Goal:
- Determine whether the checkpoint-100 relaunch produced durable progress before v6 preemption.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800`
- tpu_name: `v6-64-06-lzha`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs512-ckpt100-20260615-023800/checkpoints`

Result:
- status: preempted
- metrics/artifacts: Latest observed metrics before preemption were through step `40/10000`: `loss=2.003906`, `grad_norm=7660980044.800`, `lr=3.90e-06`, `steps/s=0.028`.
- metrics/artifacts: TPU state changed to `PREEMPTED` before step `100`.
- metrics/artifacts: GCS run prefix remained `0 B`; no checkpoint payload was written.
- metrics/artifacts: After terminal preemption, direct TPU SSH was unavailable, so no later worker log could be fetched.

Analysis:
- The run was training correctly but lost progress before the first durable checkpoint due to v6 preemption. This is infrastructure churn, not a model/data failure.
- The step-100 cadence is still preferable to step-1000; under current v6 instability, however, the next durable target remains step `100`.

Next:
- Requeue the same global-batch-512 checkpoint-100 recipe under a fresh run name, since there is no checkpoint to resume.

## 2026-06-15T03:36:45Z - requeue checkpoint-100 batch 512 after preemption

Goal:
- Restart batch-512 side-adapter training after the preemption, preserving checkpoint interval `100`.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `93296c879dd7c9a31d93b8b19602a0576f0b42b9`
- push/pull: branch is pushed; queued run will checkout the commit detached
- changed_files: worklog entries only since the validated code commit

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-r1-20260615-033000`
- tpu_name: `v6-64-06-lzha`
- queued_resource: `v6-64-06-lzha-qr`
- accelerator: `v6e-64`
- local_watch_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs512-ckpt100-r1-20260615-033000.log`
- launch: `tpu watch v6 -n 64 --setup-cmd "git fetch origin codex/wan-ti2v-side-adapter-20260613-073227 && git checkout --detach 93296c879dd7c9a31d93b8b19602a0576f0b42b9 && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" 93296c879dd7c9a31d93b8b19602a0576f0b42b9 RUN_NAME=wan-side-adapter-v6e64-full-gbs512-ckpt100-r1-20260615-033000 WANDB_PROJECT=maxdiffusion-wan-side-adapter PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=10000 CHECKPOINT_EVERY=100 EVAL_EVERY=1000 EVAL_MAX_BATCHES=4 LOG_PERIOD=10 SAVE_FINAL_CHECKPOINT=False bash bash_scripts/train_wan_side_adapter.sh`

Result:
- status: requeueing
- metrics/artifacts: `tpu watch` observed the existing TPU in terminal state `PREEMPTED` and started deleting it at local `2026-06-14 20:36:25`.

Analysis:
- No model or data changes were made. This is a straight restart because there is no durable checkpoint.

Next:
- Monitor deletion/requeue, verify setup/launch when a fresh v6e-64 slice is allocated, then repeat first-metric and first-checkpoint validation.

## 2026-06-15T03:56:00Z - corrected requeue after bad commit SHA

Goal:
- Stop the failed requeue watcher that was repeatedly trying to checkout a nonexistent commit, then relaunch the same batch-512 checkpoint-100 recipe at the actual pushed HEAD.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- bad_commit_attempted: `93296c879dd7c9a31d93b8b19602a0576f0b42b9`
- corrected_commit: `936aa68435904f22103c064cd456f7101bb83d94`
- push/pull: local branch is aligned with `origin/codex/wan-ti2v-side-adapter-20260613-073227`
- changed_files: worklog entry only

Command / Job:
- stopped_local_pids: `42559`, `42566`, `42567`
- tpu_name: `v6-64-06-lzha`
- queued_resource: `v6-64-06-lzha-qr`
- accelerator: `v6e-64`
- intended_run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-r2-20260615-034800`
- intended_launch: `tpu watch --force v6 -n 64 --setup-cmd "git fetch origin codex/wan-ti2v-side-adapter-20260613-073227 && git checkout --detach 936aa68435904f22103c064cd456f7101bb83d94 && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" 936aa68435904f22103c064cd456f7101bb83d94 RUN_NAME=wan-side-adapter-v6e64-full-gbs512-ckpt100-r2-20260615-034800 WANDB_PROJECT=maxdiffusion-wan-side-adapter PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=10000 CHECKPOINT_EVERY=100 EVAL_EVERY=1000 EVAL_MAX_BATCHES=4 LOG_PERIOD=10 SAVE_FINAL_CHECKPOINT=False bash bash_scripts/train_wan_side_adapter.sh`

Result:
- status: preparing corrected relaunch
- metrics/artifacts: The bad watcher was stopped locally. The unrelated `ego-lap` watcher on `v6-64-01-lihan` was left running.
- metrics/artifacts: TPU `v6-64-06-lzha` is `READY/HEALTHY`; queued resource `v6-64-06-lzha-qr` is `ACTIVE`.
- metrics/artifacts: All-worker process check found no active `train_wan.py` or `train_wan_side_adapter.sh` process.

Analysis:
- The previous failure was caused by recording/launching a bad full SHA for a worklog-only commit, not by the model implementation, data path, TPU memory, or batch-size configuration.
- Reusing the healthy v6e-64 slice with `--force` is safe after verifying there is no active Wan training process on the workers.

Next:
- Launch the corrected run at commit `936aa68435904f22103c064cd456f7101bb83d94`, then monitor setup, worker logs, W&B, early metrics, and the step-100 checkpoint.

## 2026-06-15T04:02:00Z - corrected batch-512 run launched

Goal:
- Confirm the corrected batch-512 checkpoint-100 run starts on the existing v6e-64 slice with the valid commit and expected command-line/data paths.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- launched_commit: `936aa68435904f22103c064cd456f7101bb83d94`
- local_worklog_status: worklog-only changes pending

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-r2-20260615-034800`
- tpu_name: `v6-64-06-lzha`
- accelerator: `v6e-64`
- launch_log: `logs/tpu_watch_wan-side-adapter-v6e64-full-gbs512-ckpt100-r2-20260615-034800.log`
- primary_log_observed: `~/maxdiffusion/logs/tpu_20260615-035832.log` on worker `0`
- launch: `tpu watch --force v6 -n 64 --setup-cmd "git fetch origin codex/wan-ti2v-side-adapter-20260613-073227 && git checkout --detach 936aa68435904f22103c064cd456f7101bb83d94 && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" 936aa68435904f22103c064cd456f7101bb83d94 RUN_NAME=wan-side-adapter-v6e64-full-gbs512-ckpt100-r2-20260615-034800 WANDB_PROJECT=maxdiffusion-wan-side-adapter PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 MAX_TRAIN_STEPS=10000 CHECKPOINT_EVERY=100 EVAL_EVERY=1000 EVAL_MAX_BATCHES=4 LOG_PERIOD=10 SAVE_FINAL_CHECKPOINT=False bash bash_scripts/train_wan_side_adapter.sh`

Result:
- status: launched, still in model initialization
- metrics/artifacts: Setup reached command launch at local `2026-06-14 20:58:44`; all 16 workers have active `train_wan.py` processes.
- metrics/artifacts: Worker `15` had a CPU-bound `unattended-upgrade` holding the dpkg lock for about eight minutes. It was terminated after verifying all other workers were clean; the intended setup `apt-get` then acquired the lock and completed.
- metrics/artifacts: TPU state is `READY`, health is `HEALTHY`.
- metrics/artifacts: Primary log confirms `dataset_type=tfrecord`, train path `gs://v6_east1d/datasets/droid_wan_side_adapter/train`, eval path `gs://v6_east1d/datasets/droid_wan_side_adapter/val`, `per_device_batch_size=8.0`, `total_train_batch_size=512.0`, `checkpoint_every=100`, `save_final_checkpoint=False`, `width=320`, `height=192`, `num_frames=32`, and side-adapter config `layers=0-29`, `hidden=512`, `heads=8`.
- metrics/artifacts: Primary log has loaded the VAE and is currently loading/porting the `Wan-AI/Wan2.2-TI2V-5B-Diffusers` transformer. No traceback, OOM, RESOURCE_EXHAUSTED, or NaN signature is visible.
- storage: GCS run prefix is still `0 B`, expected before the first checkpoint at step `100`.

Analysis:
- The corrected launch uses the exact code/data/batch/parallelism setup validated by the earlier batch-512 probe and ckpt100 attempt. The remaining uncertainty is normal startup time through 5B model load, sharding, first compile, and first batch.
- Because `tpu watch --force` exits after launch on an existing TPU, active monitoring is now being done through direct worker log/process checks and TPU health polls.

Next:
- Wait for model initialization to finish, then verify the adapter trainable/frozen parameter split, W&B run id, first `step 10/10000` metric, and step-`100` checkpoint.

## 2026-06-15T04:10:00Z - r2 preempted during initialization

Goal:
- Determine whether the corrected r2 run produced any durable progress before the v6 maintenance event.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-r2-20260615-034800`
- tpu_name: `v6-64-06-lzha`
- primary_log: `~/maxdiffusion/logs/tpu_20260615-035832.log`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs512-ckpt100-r2-20260615-034800/checkpoints`

Result:
- status: preempted
- metrics/artifacts: The run reached WAN TI2V model load, VAE load, transformer load, checkpoint-manager initialization, and all 16 ranks were active with the expected batch-512 command line.
- metrics/artifacts: The primary log had not yet emitted the adapter trainable/frozen parameter counts, W&B run link, or the first `step 10/10000` metric.
- metrics/artifacts: TPU health changed from `HEALTHY` to `UNHEALTHY_MAINTENANCE`, SSH began refusing connections, and the TPU then transitioned to terminal state `PREEMPTED`.
- storage: GCS run prefix contains only two empty placeholder objects (`0 B` total); no checkpoint payload was written.

Analysis:
- This is another infrastructure loss before the first durable checkpoint, not a model/data failure. The code path reached deeper initialization than prior failed launches but preempted before training metrics or checkpoint step `100`.
- The current r2 checkpoint prefix is safe to leave or delete later; it is zero bytes and does not threaten storage.

Next:
- Requeue the same batch-512 checkpoint-100 recipe with a fresh run name. Use the exact pushed HEAD from `git rev-parse HEAD` after committing this worklog entry.

## 2026-06-15T04:37:22Z - visual validation sidecar implementation

Goal:
- Add periodic visual validation for the live MaxDiffusion WAN TI2V side-adapter run, matching the intent of `../Wan2.2/eval_adaptor.py`: restore adapter checkpoints, replay the same CFG Euler-style latent rollout, decode samples, save ground-truth-vs-prediction videos, and record metrics.

Hypothesis:
- A sidecar validator is safer than decoding inside the batch-512 training loop because the training run deletes VAE/text components to preserve HBM. A separate v6 validation slice can restore adapter-only checkpoints and inspect videos without perturbing the training process.

Change:
- Added `src/maxdiffusion/generate_wan_side_adapter.py`, a TFRecord-based visual validator for side-adapter checkpoints.
- Added `bash_scripts/validate_wan_side_adapter.sh`, a one-shot TPU validator wrapper that uses `bash_scripts/setup.sh` beforehand and writes validation artifacts to GCS.
- Added `bash_scripts/watch_wan_side_adapter_validation.sh`, a local checkpoint watcher that launches validation on a separate v6 TPU for step `100` and every `1000` steps afterward by default.
- Added validation config compatibility fields to `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- worktree: `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`
- branch: `codex/wan-ti2v-side-adapter-20260613-073227`
- base_commit: `1fbc1609b4bd87f11224222e08c1cbe7d18e9ce7`
- implementation_commit: pending
- changed_files: `src/maxdiffusion/generate_wan_side_adapter.py`, `bash_scripts/validate_wan_side_adapter.sh`, `bash_scripts/watch_wan_side_adapter_validation.sh`, `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`, this worklog

Command / Job:
- validation_targets: `CHECKPOINT_STEP=100`, then `CHECKPOINT_STEP % 1000 == 0`
- validation_tpu_default: `v6-8-wan-val-lzha`, `VALIDATION_TPU_CHIPS=8`
- validation_output_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/<RUN_NAME>/validation`
- expected_artifacts: per-step `config.json`, `summary.csv`, `summary.json`; per-sample `ground_truth.mp4`, `sample.mp4`, `comparison_gt_top_pred_bottom.mp4`, `metrics.json`, optional `meta.json`
- local_checks: `python3 -m py_compile src/maxdiffusion/generate_wan_side_adapter.py`; `bash -n bash_scripts/validate_wan_side_adapter.sh`; `bash -n bash_scripts/watch_wan_side_adapter_validation.sh`

Result:
- status: implementation validated locally, not yet launched because no step-100 checkpoint exists.
- metrics/artifacts: TFRecord parser was checked against the converter and trainer. All use raw bytes for `z_i0` float16 `[48,1,12,20]`, `z_video` float16 `[48,9,12,20]`, and `actions` float32 `[32,7]`.
- metrics/artifacts: Validator compares generated videos to the VAE decode of cached `z_video`, because the MaxDiffusion TFRecords do not include raw source frames.
- metrics/artifacts: Watcher now bounds validation wait with `VALIDATION_MAX_WAIT_SECS=21600`; failed or missing summaries leave the checkpoint step eligible for retry instead of blocking future validations forever.

Analysis:
- The validation rollout reuses `wan_side_adapter_forward`, `build_rollout_sigmas`, `rollout_timesteps_from_sigmas`, `_build_per_token_timestep`, and `apply_first_frame_pin`, so it is sampling the same model path as training rather than a second hand-rolled path.
- The restored `TrainState.params` contains only adapter parameters and restores from the adapter checkpoint; the frozen transformer is loaded from the pretrained model and sharded by the same trainer helper used by training.
- The first validation may need a larger slice if v6e-8 cannot fit the 5B transformer plus VAE decode path, but batch-1 generation is expected to fit. If it does not, relaunch the watcher with a larger `VALIDATION_TPU_CHIPS` value and leave the training slice untouched.

Next:
- Commit and push this validation implementation, then start the validation watcher for run `wan-side-adapter-v6e64-full-gbs512-ckpt100-r3-20260615-041100` at the pushed commit.

## 2026-06-15T04:37:22Z - r3 batch-512 live status before validation watcher

Goal:
- Record the current state of the active batch-512 training run before attaching the periodic visual validation sidecar.

Command / Job:
- run_name: `wan-side-adapter-v6e64-full-gbs512-ckpt100-r3-20260615-041100`
- tpu_name: `v6-64-06-lzha`
- accelerator: `v6e-64`
- training_commit: `1fbc1609b4bd87f11224222e08c1cbe7d18e9ce7`
- primary_log: `~/maxdiffusion/logs/tpu_20260615-042314.log`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs512-ckpt100-r3-20260615-041100/checkpoints`

Result:
- status: active, no durable checkpoint yet.
- metrics/artifacts: TPU `v6-64-06-lzha` is `READY/HEALTHY`; queued resource is `ACTIVE`.
- metrics/artifacts: All 16 workers have active `python src/maxdiffusion/train_wan.py` processes with the expected batch-512 command line.
- metrics/artifacts: Worker logs have no traceback, `RESOURCE_EXHAUSTED`, OOM, NaN, or killed-process signature. One worker has emitted the expected `[wan_side_adapter] trainable adapter params: 239.5M`, `[wan_side_adapter] frozen transformer params: 5.00B`, and `***** Running WAN TI2V side-adapter training *****` lines.
- metrics/artifacts: No `step N/10000` metrics or step-`100` checkpoint have appeared yet; the GCS run prefix is still `0 B`.

Analysis:
- The current evidence is consistent with first-batch compilation/materialization on the 5B model at global batch `512`. It is not yet evidence of a model or data failure.
- Periodic visual validation cannot run until the first adapter checkpoint is saved at step `100`, so the watcher should be started now and allowed to idle on the GCS checkpoint prefix.

Next:
- Start the watcher after the validation implementation is pushed, then continue direct log/GCS monitoring until the first training metrics and validation videos are produced and inspected.

## 2026-06-15T04:41:00Z - validation watcher started

Goal:
- Attach the periodic visual validation sidecar to the active r3 training run without touching the v6e-64 training slice.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- implementation_commit: `79d74119425eef15aea83a21e2bd9e90c434df87`
- push/pull: pushed to `origin/codex/wan-ti2v-side-adapter-20260613-073227`
- validation_checkout: watcher will checkout `79d74119425eef15aea83a21e2bd9e90c434df87` detached on the validation TPU.

Command / Job:
- local_session_id: `96626`
- local_log: `logs/wan_side_adapter_validation_watch_wan-side-adapter-v6e64-full-gbs512-ckpt100-r3-20260615-041100.log`
- command: `RUN_NAME=wan-side-adapter-v6e64-full-gbs512-ckpt100-r3-20260615-041100 COMMIT=79d74119425eef15aea83a21e2bd9e90c434df87 VALIDATION_TPU_NAME=v6-8-wan-val-lzha VALIDATION_TPU_CHIPS=8 VALIDATION_FIRST_STEP=100 VALIDATION_EVERY=1000 NUM_EVAL_VIDEOS=4 POLL_SECS=120 bash bash_scripts/watch_wan_side_adapter_validation.sh`
- checkpoint_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs512-ckpt100-r3-20260615-041100/checkpoints`
- validation_output_dir: `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/wan-side-adapter-v6e64-full-gbs512-ckpt100-r3-20260615-041100/validation`

Result:
- status: watcher running and idle.
- metrics/artifacts: `tpu list v6` shows only the training TPU `v6-64-06-lzha`; no validation TPU has been created yet.
- metrics/artifacts: GCS checkpoint prefix is still `0 B` and contains no numeric checkpoint step, so the watcher correctly has not launched validation.

Analysis:
- The watcher is now ready to validate step `100` as soon as Orbax writes it. A failed validation will not mark the step as seen, so the step remains eligible for retry.

Next:
- Keep monitoring the r3 training logs and GCS checkpoint prefix. When step `100` appears, confirm the watcher creates the validation TPU, then inspect `summary.json`, `summary.csv`, and representative MP4s for frame count/resolution/nonblank content.

## 2026-06-15T04:45:00Z - configurable TFRecord shuffle buffer fallback

Goal:
- Prepare a conservative fallback if the active batch-512 run remains stuck before first TPU execution due to first-batch TFRecord shuffle fill.

Hypothesis:
- The current default training data iterator shuffles `global_batch_size * 10` examples per host after host sharding. At global batch `512`, this can force a large first-batch fill from GCS before the first train step. Keeping the default preserves current behavior, but an explicit smaller buffer can reduce startup latency on a relaunch if needed.

Change:
- Added `tfrecord_shuffle_buffer_size` to the generic TFRecord iterator with default `-1`, which retains the existing `global_batch_size * 10` behavior.
- Added the same config field to `base_wan_5b_side_adapter.yml`.
- Added optional `TFRECORD_SHUFFLE_BUFFER_SIZE` forwarding in `bash_scripts/train_wan_side_adapter.sh`.

Version Control:
- agent_id: `wan-ti2v-side-adapter-20260613-073227`
- base_commit: `9c52e11ccfe7441276743489d154866b4d6065c9`
- implementation_commit: pending
- changed_files: `src/maxdiffusion/input_pipeline/_tfds_data_processing.py`, `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`, `bash_scripts/train_wan_side_adapter.sh`, this worklog

Command / Job:
- local_checks: `python3 -m py_compile src/maxdiffusion/input_pipeline/_tfds_data_processing.py`; `bash -n bash_scripts/train_wan_side_adapter.sh`
- active_run: unchanged; no relaunch performed.

Result:
- status: implementation validated locally, pending commit/push.
- metrics/artifacts: The r3 checkpoint prefix still has no numeric checkpoint; watcher remains idle.

Analysis:
- This patch does not change any active process and does not alter default semantics. It only provides an explicit next-run lever, for example `TFRECORD_SHUFFLE_BUFFER_SIZE=512`, if the active run is lost or conclusively stuck before first step.

Next:
- Commit/push the fallback knob, continue monitoring r3, and only relaunch with the smaller shuffle buffer if the current run fails or remains pre-step long enough to justify replacing it.
