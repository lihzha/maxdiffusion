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
