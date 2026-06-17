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

## 2026-06-15T07:40:27Z - v6e-64 pre-context smoke launch

Goal:
- Run a one-step target-surface smoke on a fresh v6e-64 before launching full training.

Hypothesis:
- If the pre-context NNX path is wired correctly, TPU setup should complete and a one-step full-global-batch denoising train step should compile and log loss/grad metrics.

Change:
- No code changes since implementation commit.

Version Control:
- agent_id: wan-prectx-adapter-20260615-073355
- worktree: /home/lzha/code/maxdiffusion-worktrees/wan-prectx-adapter-20260615-073355
- worklog: worklogs/wan-ti2v-pre-context-adapter/wan-prectx-adapter-20260615-073355.md
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- base_commit: 7ee701de743169e6888a77dac1f3d31d24e408e1
- implementation_commit: 60781c70405dece76409c1c38554b0ecfd43f5f1
- push/pull: pushed branch; launch checkout commit 89d22a379b494dd87c1731a83f83c98f4550464c
- changed_files: n/a for launch
- remote_commit/status: pending, worker not allocated yet

Command / Job:
- command: `tpu watch v6 -n 64 --setup-cmd "export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 && git fetch origin codex/wan-ti2v-pre-context-adapter-v6e64 && git checkout --detach 89d22a379b494dd87c1731a83f83c98f4550464c && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" codex/wan-ti2v-pre-context-adapter-v6e64 RUN_NAME=wan-pre-context-v6e64-smoke-gbs512-r1-20260615-074027 ACTION_ADAPTER_TYPE=pre_context PRE_CONTEXT_TOKENS=8 PRE_CONTEXT_HEADS=40 TRAIN_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/train EVAL_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/val OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter MODEL_DIR=Wan-AI/Wan2.2-TI2V-5B-Diffusers MAX_TRAIN_STEPS=1 CHECKPOINT_EVERY=1000 EVAL_EVERY=0 LOG_PERIOD=1 SAVE_FINAL_CHECKPOINT=False PER_DEVICE_BATCH_SIZE=8 GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 GLOBAL_BATCH_SIZE_TO_LOAD=512 TFRECORD_SHUFFLE_BUFFER_SIZE=1024 WANDB_PROJECT=maxdiffusion-wan-pre-context-adapter HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 bash bash_scripts/train_wan_side_adapter.sh`
- job_id: local watcher session 86675; queued resource v6-64-08-lzha-qr; TPU name v6-64-08-lzha
- run_dir: remote maxdiffusion checkout pending allocation
- logs: logs/tpu_watch_wan-pre-context-v6e64-smoke-gbs512-r1-20260615-074027.log
- artifacts: W&B smoke metrics expected; no checkpoint expected because `SAVE_FINAL_CHECKPOINT=False`

Result:
- status: queued
- metrics/artifacts: none yet
- key evidence: queued resource remains `PROVISIONING`; `gcloud alpha compute tpus tpu-vm describe v6-64-08-lzha` returns `NOT_FOUND`.

Analysis:
- No code path has run yet. This is a TPU capacity/allocation wait, not a model/setup failure.

Next:
- Continue watching until the TPU reaches READY and setup begins, or record external infrastructure blockage if provisioning does not progress.

## 2026-06-15T08:24:00Z - v6e-64 launch blocked by quota/occupied slices

Goal:
- Find a usable v6e-64 slice for the pre-context smoke and subsequent full training.

Hypothesis:
- If `v6-64-08-lzha` is tied to a stale queued resource, either an existing READY lzha v6e-64 or a new unused TPU name can be used for the smoke.

Change:
- Stopped only the local pre-context smoke watcher for `v6-64-08-lzha`; left the unrelated older side-adapter watcher on `v6-64-08-lzha` untouched.
- Checked `v6-64-07-lzha` and found it occupied by an active `ego-lap` v6-64 training job.
- Tried a fresh unused name `v6-64-09-lzha`, then stopped the retry watcher after GCP returned quota exhaustion.

Version Control:
- agent_id: wan-prectx-adapter-20260615-073355
- worktree: /home/lzha/code/maxdiffusion-worktrees/wan-prectx-adapter-20260615-073355
- worklog: worklogs/wan-ti2v-pre-context-adapter/wan-prectx-adapter-20260615-073355.md
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- base_commit: 7ee701de743169e6888a77dac1f3d31d24e408e1
- implementation_commit: 60781c70405dece76409c1c38554b0ecfd43f5f1
- push/pull: branch pushed; final blockage entry pending commit/push
- changed_files: worklogs/wan-ti2v-pre-context-adapter/wan-prectx-adapter-20260615-073355.md
- remote_commit/status: no pre-context remote run launched

Command / Job:
- command: `gcloud alpha compute tpus tpu-vm ssh v6-64-07-lzha --worker=all --command='ps ...'`
- command: `tpu watch v6 -n 64 ... TPU_NAME=v6-64-09-lzha ... MAX_TRAIN_STEPS=1 ...`
- job_id: local watcher sessions 86675 and 19751 stopped; no active pre-context TPU job
- run_dir: n/a
- logs: logs/tpu_watch_wan-pre-context-v6e64-smoke-gbs512-r1-20260615-074027.log; logs/tpu_watch_wan-pre-context-v6e64-smoke-gbs512-r2-20260615-082200.log
- artifacts: n/a

Result:
- status: blocked
- metrics/artifacts: no training metrics; no checkpoint; no validation artifacts
- key evidence: `v6-64-07-lzha` has active `ego-lap` Python processes across workers; `v6-64-09-lzha` creation failed with `RESOURCE_EXHAUSTED` for quota `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI`, limit 512 in `us-east1-d`; queued resources show `v6-64-08-lzha-qr` still `PROVISIONING`.

Analysis:
- The implementation is ready and pushed, but v6e-64 execution is externally blocked. The only lzha READY v6e-64 is already running another training job, the requested fresh `v6-64-08-lzha` name is tied to an older provisioning watcher, and creating another v6e-64 exceeds project quota.

Next:
- Launch the smoke/full training when a v6e-64 slice becomes available, or explicitly free/reassign an existing lzha v6e-64 queued resource/job.

## 2026-06-15T20:18:00Z - resumed v6e-64 availability monitoring

Goal:
- Keep an active monitor on v6e-64 availability for the pre-context adapter smoke launch.

Hypothesis:
- If the old side-adapter watcher releases `v6-64-08-lzha`, or a fresh v6e-64 quota path becomes available, the pre-context smoke can be launched from the pushed branch without additional code changes.

Change:
- Started a local monitoring loop for TPU node state, queued resources, local `tpu watch` ownership, and worker process/log state.
- Did not stop or alter unrelated active watchers or TPU jobs.

Version Control:
- agent_id: wan-prectx-adapter-20260615-073355
- worktree: /home/lzha/code/maxdiffusion-worktrees/wan-prectx-adapter-20260615-073355
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- current_commit: aa70ec157574e20b6a842078b98489bc2656cbce
- changed_files: worklogs/wan-ti2v-pre-context-adapter/wan-prectx-adapter-20260615-073355.md

Command / Job:
- command: local monitor loop checking `gcloud compute tpus tpu-vm list`, `gcloud alpha compute tpus queued-resources list`, local `tpu watch` processes, and `v6-64-08-lzha` worker state when READY.
- monitor_log: logs/monitor_wan_pre_context_v6e64_availability_20260615-201801.log
- owned_pre_context_job: none currently active

Result:
- status: monitoring
- key evidence: `v6-64-08-lzha` is `CREATING/PROVISIONING` again and is still associated with the older side-adapter watcher; `v6-64-07-lzha` is also under a separate ego-lap watcher; no pre-context remote run has started.

Analysis:
- The active blocker remains resource ownership/capacity, not code readiness. The monitor should wait for a clean v6e-64 slot before launching the pre-context smoke.

Next:
- Continue polling; launch the one-step pre-context smoke only after a usable v6e-64 slice is available and not owned by another watcher.

## 2026-06-15T20:31:32Z - retry fresh v6e-64 pre-context smoke

Goal:
- Launch the one-step target-surface pre-context adapter smoke on a fresh v6e-64 now that quota appears to have freed.

Hypothesis:
- A new `v6-64-09-lzha` SPOT queued resource should now fit under the project v6e preemptible quota, avoiding the occupied `v6-64-08-lzha` side-adapter run.

Change:
- Stopped the local availability-only monitor.
- Preparing a fresh `tpu watch v6 -n 64` launch for `ACTION_ADAPTER_TYPE=pre_context`, full global batch 512, `MAX_TRAIN_STEPS=1`, and no final checkpoint.
- Will source local TPU secrets for W&B-enabled smoke logging without writing secrets to the repo or command text.

Version Control:
- agent_id: wan-prectx-adapter-20260615-073355
- worktree: /home/lzha/code/maxdiffusion-worktrees/wan-prectx-adapter-20260615-073355
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- current_commit_before_launch_entry: 780933643402d914bff379119097aa4754286ec0
- changed_files: worklogs/wan-ti2v-pre-context-adapter/wan-prectx-adapter-20260615-073355.md

Command / Job:
- target_tpu: v6-64-09-lzha
- run_name: wan-pre-context-v6e64-smoke-gbs512-r3-20260615-203132
- command: `tpu watch v6 -n 64 --setup-cmd "<checkout exact commit; setup; prefetch Wan-AI/Wan2.2-TI2V-5B-Diffusers>" codex/wan-ti2v-pre-context-adapter-v6e64 ACTION_ADAPTER_TYPE=pre_context PRE_CONTEXT_TOKENS=8 PRE_CONTEXT_HEADS=40 MAX_TRAIN_STEPS=1 SAVE_FINAL_CHECKPOINT=False ... bash bash_scripts/train_wan_side_adapter.sh`
- logs: logs/tpu_watch_wan-pre-context-v6e64-smoke-gbs512-r3-20260615-203132.log
- artifacts: W&B smoke metrics expected; no final checkpoint expected

Result:
- status: pending launch
- metrics/artifacts: none yet

Analysis:
- The old `v6-64-08-lzha` side-adapter run is active and should not be interrupted. A fresh TPU name is the cleaner path if quota now permits it.

Next:
- Commit/push this launch record, start `tpu watch`, then monitor allocation, remote setup, first train step, W&B/log metrics, and exit state.

## 2026-06-15T20:43:00Z - fresh v6e-64 retry remains quota-blocked

Goal:
- Keep the pre-context smoke launch retry loop alive until a fresh v6e-64 slot can be created.

Hypothesis:
- If enough preemptible v6e quota frees in `us-east1-d`, the existing `tpu watch` process for `v6-64-09-lzha` will create `v6-64-09-lzha-qr` and proceed to setup without needing a new command.

Change:
- Left the `tpu watch` process attached instead of stopping after the first quota error.
- Did not interrupt the active old side-adapter training on `v6-64-08-lzha`.

Version Control:
- agent_id: wan-prectx-adapter-20260615-073355
- worktree: /home/lzha/code/maxdiffusion-worktrees/wan-prectx-adapter-20260615-073355
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- launch_commit: 6277a16490fb88f7bb9e96cd7e155b5f56b57ddf

Command / Job:
- target_tpu: v6-64-09-lzha
- run_name: wan-pre-context-v6e64-smoke-gbs512-r3-20260615-203132
- local_watcher_pids: 772049, 772062, 772063
- logs: logs/tpu_watch_wan-pre-context-v6e64-smoke-gbs512-r3-20260615-203132.log
- queued_resource: not created yet

Result:
- status: active retry loop; no remote setup started
- key evidence: repeated `RESOURCE_EXHAUSTED` errors for quota `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI`, limit 512 in `us-east1-d`; `v6-64-09-lzha` remains `NOT_FOUND`.

Analysis:
- The implementation launch command is staged correctly, but current progress is blocked at queued-resource creation by project-level preemptible v6e quota. This is still an infrastructure wait.

Next:
- Continue monitoring the attached `tpu watch` process. If creation succeeds, verify the remote exact commit, setup, adapter mode, one train step, W&B/log metrics, and exit state.

## 2026-06-15T22:00:00Z - quota wait continuing

Goal:
- Continue the requested monitoring loop for the fresh pre-context v6e-64 smoke launch.

Hypothesis:
- The active watcher will proceed automatically once project preemptible v6e quota frees.

Change:
- No code or launch-command changes.
- Kept the existing `tpu watch` process attached.

Version Control:
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- current_worklog_commit_before_entry: cd173650e733bb2b82b4b5ec0cda71a62a955c9d
- launch_commit: 6277a16490fb88f7bb9e96cd7e155b5f56b57ddf

Command / Job:
- target_tpu: v6-64-09-lzha
- run_name: wan-pre-context-v6e64-smoke-gbs512-r3-20260615-203132
- local_watcher_pids: 772049, 772062, 772063
- logs: logs/tpu_watch_wan-pre-context-v6e64-smoke-gbs512-r3-20260615-203132.log

Result:
- status: still retrying queued-resource creation
- key evidence: from 20:32Z through 22:00Z, repeated create attempts returned `RESOURCE_EXHAUSTED` for `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI`, limit 512 in `us-east1-d`; `v6-64-09-lzha` remains `NOT_FOUND`.

Analysis:
- This remains an external quota/resource contention blocker. No remote setup, model import, JAX compile, or training step has started for the pre-context run.

Next:
- Keep the retry loop alive unless the user asks to stop, switch provisioning model, switch zone, or free an existing v6e resource.

## 2026-06-15T22:37:00Z - stopped unsafe pre-context watcher after remote checkout conflict

Goal:
- Prevent the pre-context smoke from launching against the wrong remote checkout after `v6-64-09-lzha` allocation succeeded.

Hypothesis:
- The smoke launch should be stopped and relaunched only from an isolated remote checkout, because the shared `~/maxdiffusion` tree was changed by another local process.

Change:
- Stopped the owned pre-context `tpu watch` process for `wan-pre-context-v6e64-smoke-gbs512-r3-20260615-203132`.
- Cleaned up an owned orphaned setup SSH process left after interrupting the watcher.
- Did not stop the separate local process targeting `v6-64-09-lzha` with `git checkout adaptor`.

Version Control:
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- latest_worklog_commit_before_entry: c390e7e9ca5185497148b8f40720e258e2dfab92
- launch_commit: 6277a16490fb88f7bb9e96cd7e155b5f56b57ddf

Command / Job:
- target_tpu: v6-64-09-lzha
- queued_resource: v6-64-09-lzha-qr
- queued_resource_state: ACTIVE
- tpu_state: READY, HEALTHY
- stopped_owned_pids: 772049, 772062, 772063; orphaned SSH child 898679
- remaining_conflicting_process: local pids 900854, 900863, 900946, 900948 running a separate setup command on `adaptor`

Result:
- status: pre-context smoke not running
- key evidence: worker 0 remote checkout changed to `adaptor` at `HEAD=f5ed765`; branch does not contain `bash_scripts/prefetch_hf_snapshot.sh`; the launch setup command called that missing helper and used the shared `~/maxdiffusion` tree.
- metrics/artifacts: no pre-context training metrics, checkpoints, or validation artifacts produced.

Analysis:
- The v6e-64 quota/resource blocker cleared and `v6-64-09-lzha` was allocated, but the launch path was unsafe because it used the shared remote checkout and a missing prefetch helper. A relaunch should avoid `bash_scripts/setup.sh`'s `cd maxdiffusion` wrapper and should not depend on `bash_scripts/prefetch_hf_snapshot.sh`.

Next:
- Wait for the separate `adaptor` setup process to finish or for the user to allow stopping it. Then relaunch using an isolated remote repo name such as `GH_REPO_NAME=maxdiffusion-prectx`, exact commit checkout, root `setup.sh MODE=stable DEVICE=tpu`, no prefetch helper, and direct remote verification before training.

## 2026-06-15T22:48:04Z - cleanup after maintenance and prepare isolated relaunch

Goal:
- Relaunch the pre-context smoke from a clean, isolated remote checkout after the first `v6-64-09-lzha` allocation was disrupted.

Hypothesis:
- Reusing the same TPU name after deleting the stale queued resource should be safe if the next launch uses `GH_REPO_NAME=maxdiffusion-prectx`, a robust setup command from the repo root, and no missing prefetch helper.

Change:
- Deleted stale `v6-64-09-lzha-qr` after the TPU went `UNHEALTHY_MAINTENANCE`, then `DELETING`, then `NOT_FOUND`.
- Observed the separate `adaptor` setup process exit before cleanup.
- Preparing a corrected smoke launch with a new run name.

Version Control:
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- current_commit_before_entry: 701d42ab3d72514c82bd9b2f3e5e80269f9794b2

Command / Job:
- cleanup_command: `gcloud alpha compute tpus queued-resources delete v6-64-09-lzha-qr --project mae-irom-lab-guided-data --zone us-east1-d --quiet`
- next_target_tpu: v6-64-09-lzha
- next_run_name: wan-pre-context-v6e64-smoke-gbs512-r4-20260615-224804
- next_remote_repo: `~/maxdiffusion-prectx`
- next_setup: checkout exact commit; `uv venv --python 3.12 .venv --seed`; source venv; `bash setup.sh MODE=stable DEVICE=tpu`; no `bash_scripts/prefetch_hf_snapshot.sh`

Result:
- status: stale queued resource deleted; ready to retry
- metrics/artifacts: none yet

Analysis:
- The previous allocation reached READY but was lost to maintenance before a valid pre-context setup/train could run. The next launch should avoid shared `~/maxdiffusion` interference and the missing prefetch script.

Next:
- Commit/push this cleanup record, then start the corrected `tpu watch` retry and monitor allocation/setup/train from the isolated repo.

## 2026-06-15T22:49:30Z - aborted invalid isolated repo-name launch

Goal:
- Avoid launching from a non-existent GitHub repository after discovering how `tpu watch` interprets `GH_REPO_NAME`.

Hypothesis:
- `GH_REPO_NAME` controls the GitHub repository name, not only the remote directory name, so setting it to `maxdiffusion-prectx` is invalid for this project.

Change:
- Started and immediately stopped `wan-pre-context-v6e64-smoke-gbs512-r4-20260615-224804` after `tpu watch` reported repo `lihzha/maxdiffusion-prectx`.
- Verified no `v6-64-09-lzha-qr` remained to delete.

Version Control:
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- launch_commit: 37009b292cf962a35ae8b7029abb51de4156ea08

Command / Job:
- stopped_run_name: wan-pre-context-v6e64-smoke-gbs512-r4-20260615-224804
- invalid_repo: lihzha/maxdiffusion-prectx
- target_tpu: v6-64-09-lzha

Result:
- status: aborted before setup; no TPU created and no training started
- metrics/artifacts: none

Analysis:
- The next relaunch should keep `GH_REPO_NAME=maxdiffusion` and rely on exact commit checkout plus direct verification rather than trying to change the remote clone directory through this variable.

Next:
- Relaunch with `GH_REPO_NAME=maxdiffusion`, exact commit checkout, root `setup.sh`, and no missing prefetch helper.

## 2026-06-15T23:12:05Z - r5 failed on Hugging Face download; add prefetch retry mitigation

Goal:
- Diagnose the first valid pre-context launch on `v6-64-09-lzha` and relaunch without repeating the same failure.

Change:
- Added `bash_scripts/prefetch_hf_snapshot.sh`, a retryable Hugging Face `snapshot_download` helper with Xet and `hf_transfer` disabled by default.
- Added `HF_HUB_DOWNLOAD_TIMEOUT=300` and `HF_HUB_ETAG_TIMEOUT=120` defaults to training and validation launchers.
- Updated the validation watcher setup path to prefetch the model before entering JAX.

Version Control:
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- failing_launch_commit: db3c3d89ce642241e345b6856713a1c4d4742955
- next_commit: pending

Command / Job:
- failed_run_name: wan-pre-context-v6e64-smoke-gbs512-r5-20260615-224804
- target_tpu: v6-64-09-lzha
- launch_state: TPU READY/HEALTHY; setup completed; training launched.

Result:
- status: failed before the first train step
- metrics/artifacts: no loss, checkpoint, or validation artifact produced.
- key evidence: worker 3 failed while loading `Wan-AI/Wan2.2-TI2V-5B-Diffusers` with `408 Client Error: Request Time-out` for `text_encoder/model-00002-of-00003.safetensors`; other workers then aborted through JAX coordination shutdown.

Analysis:
- This was a distributed launch hygiene failure, not an adapter-shape failure. Model downloads occurred after JAX distributed startup, so a transient Hugging Face timeout on one worker cascaded into a mesh-wide abort.
- The relaunch should prefetch the model in the setup phase, before JAX initializes, and then start training only after every worker has a warmed local cache.

Next:
- Commit/push the prefetch mitigation, stop the failed r5 watcher, relaunch as r6 with setup prefetch enabled, and monitor through first loss/checkpoint evidence.

## 2026-06-15T23:18:58Z - r7 reached adapter init; fix pre-context head count

Goal:
- Continue the r7 launch past setup/prefetch and diagnose the first adapter-code failure.

Change:
- Changed the pre-context attention head implementation to treat configured heads as a requested value and fall back to `gcd(feature_dim, heads)` when the actual feature dimension is not divisible by the request.
- Updated default `pre_context_heads` from 40 to 8 in the 5B side-adapter config and launch scripts.

Version Control:
- branch: codex/wan-ti2v-pre-context-adapter-v6e64
- failing_launch_commit: 0a3bb14315df680aed88fdc018af0dc955fe5d64
- next_commit: pending

Command / Job:
- failed_run_name: wan-pre-context-v6e64-smoke-gbs512-r7-20260615-231417
- target_tpu: v6-64-09-lzha
- setup_result: completed with model prefetch; training launched.

Result:
- status: failed before first train step at adapter construction
- metrics/artifacts: no loss, checkpoint, or validation artifact produced.
- key evidence: worker 0 raised `ValueError: feature_dim must be divisible by heads` in `NNXPreContextFeatureContextHead`.

Analysis:
- The Hugging Face prefetch mitigation worked: the previous worker-local 408 did not recur during JAX startup.
- The new pre-context head incorrectly assumed the requested head count would divide the model feature dimension. For this 5B path, 8 is a safer default and the code should defensively select a valid divisor.

Next:
- Commit/push the head-count fix, stop the failed r7 watcher, relaunch as r8 with `PRE_CONTEXT_HEADS=8`, and monitor through first loss/checkpoint evidence.

## 2026-06-17T18:25:40Z - stopped pre-context training and deleted TPU watcher resources

Goal:
- Stop all task-owned MaxDiffusion pre-context TPU jobs/watchers and delete the v6e-64 TPU resources after the user requested teardown.

Change:
- Confirmed no local `tpu watch` / validation watcher process remained for `wan-pre-context`, `v6-64-10-lzha`, or `v6-8-wan-val-lzha`.
- Attempted a scoped all-worker stop for `train_wan.py` / `train_wan_side_adapter.sh` matching run `wan-pre-context-v6e64-full-gbs512-fresh-scratch-ckpt100-east1d-20260616-191526`; SSH was unreliable after the processes detached, so cleanup proceeded by deleting the owned TPU slice.
- Deleted TPU VM `v6-64-10-lzha` and queued resource `v6-64-10-lzha-qr` in `us-east1-d`.

Command / Job:
- `gcloud alpha compute tpus tpu-vm delete v6-64-10-lzha --project=mae-irom-lab-guided-data --zone=us-east1-d --quiet`
- `gcloud alpha compute tpus queued-resources delete v6-64-10-lzha-qr --project=mae-irom-lab-guided-data --zone=us-east1-d --quiet`
- Verification: `gcloud alpha compute tpus tpu-vm describe v6-64-10-lzha ...` and `gcloud alpha compute tpus queued-resources describe v6-64-10-lzha-qr ...`

Result:
- status: stopped and deleted
- verification: both describe calls returned `NOT_FOUND`; `us-east1-d` lzha TPU/queued-resource listing showed no remaining lzha entries; local watcher process check was clean.
- latest durable numeric checkpoint observed in GCS: `26900`; no `27000` checkpoint metadata was present during cleanup.

Analysis:
- The active run was the fresh-noise pre-context training run on commit `7260778`, with `action_adapter_type=pre_context` and `side_adapter_noise_mode=fresh`. Deleting the TPU/queued resource terminates any worker process that survived the interrupted SSH kill attempt.

Next:
- No active MaxDiffusion TPU watcher, TPU VM, queued resource, or monitoring loop remains for this run.
