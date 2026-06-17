<!--
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# Overview

MaxDiffusion is a collection of reference implementations of various latent diffusion models written in pure Python/Jax that run on XLA devices including Cloud TPUs and GPUs. MaxDiffusion aims to be a launching off point for ambitious Diffusion projects both in research and production. We encourage you to start by experimenting with MaxDiffusion out of the box and then fork and modify MaxDiffusion to meet your needs.



## Wan2.2 TI2V 5B Action-Adapter Training on TPUs

This section covers the Wan2.2 TI2V 5B side-adapter and pre-context adapter TPU
training path. For implementation notes, experiment history, and known failure
modes, read [docs/wan_ti2v_side_adapter_handoff.md](docs/wan_ti2v_side_adapter_handoff.md)
and [docs/wan_ti2v_pre_context_adapter_methodology_results.md](docs/wan_ti2v_pre_context_adapter_methodology_results.md)
before launching a full run.

The same trainer and config are used for both adapter modes:

| Mode | Override |
| --- | --- |
| Side adapter residual injection | `ACTION_ADAPTER_TYPE=side_adapter` |
| Pre-context action adapter | `ACTION_ADAPTER_TYPE=pre_context` |

Keep `PRE_CONTEXT_TOKENS=8` and `PRE_CONTEXT_HEADS=8` for the pre-context
path. Set `SIDE_ADAPTER_NOISE_MODE=fresh` explicitly for current training runs;
some wrappers or older commands may otherwise reproduce fixed-noise diagnostic
runs.

### Required files and assets

The TPU training path depends on:

- `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`
- `src/maxdiffusion/train_wan.py`
- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`
- `src/maxdiffusion/models/wan/side_adapter_wan.py`
- `src/maxdiffusion/generate_wan_side_adapter.py` for visual validation
- `bash_scripts/setup.sh`
- `bash_scripts/train_wan_side_adapter.sh`
- `bash_scripts/prefetch_hf_snapshot.sh`
- `bash_scripts/validate_wan_side_adapter.sh` and
  `bash_scripts/watch_wan_side_adapter_validation.sh` for validation

External assets:

```bash
MODEL_DIR=Wan-AI/Wan2.2-TI2V-5B-Diffusers
TRAIN_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/train
EVAL_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/val
OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter
```

The converted DROID TFRecords contain one cached latent window per example:
`z_i0` as `float16 [48, 1, 12, 20]`, `z_video` as
`float16 [48, 9, 12, 20]`, and `actions` as `float32 [32, 7]`.

### TPU setup

Use TPU VMs with the repository checked out at the exact commit you intend to
run. For v6/v6e runs in this workflow, use `us-east1-d` and the `gs://v6_east1d`
bucket unless you intentionally change both the TPU zone and storage paths.

On TPU workers, set up the environment through the repository wrapper:

```bash
bash bash_scripts/setup.sh MODE=stable DEVICE=tpu
```

During multihost launch, prefetch the Hugging Face snapshot before JAX
distributed startup. This avoids one worker streaming model shards while other
workers wait at the distributed barrier:

```bash
bash bash_scripts/prefetch_hf_snapshot.sh Wan-AI/Wan2.2-TI2V-5B-Diffusers
```

`bash_scripts/setup.sh` is safe to call from either the repository root or from
`$HOME` when a `maxdiffusion` checkout exists. In the `tpu watch` flow below,
the setup command runs from the remote checkout, so do not add `cd maxdiffusion`
inside `SETUP_CMD`.

### Full v6e-64 launch

The established full-run target is v6e-64 with pure FSDP and global batch size
512. `tpu watch` expects a branch name, while the setup command should checkout
the exact local commit for reproducibility.

```bash
set +x
source "$HOME/.config/irom-tpu/secrets.env"
set -x

export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
export TPU_PROJECT=mae-irom-lab-guided-data
export TPU_ZONE_v6=us-east1-d
export TPU_BUCKET_v6=gs://v6_east1d
export GH_OWNER=lihzha
export GH_REPO_NAME=maxdiffusion
export TPU_NAME=<your-v6e-64-tpu-name>
export SLEEP_SECS=15
export TPU_WATCH_HEALTH_CHECK_SECS=15
unset WANDB_DISABLED

COMMIT="$(git rev-parse HEAD)"
WATCH_BRANCH=adaptor
RUN_NAME="wan-pre-context-v6e64-full-gbs512-fresh-$(date -u +%Y%m%d-%H%M%S)"
OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter
SETUP_CMD="export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=120 && git fetch origin ${WATCH_BRANCH} && git checkout --detach ${COMMIT} && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu && bash bash_scripts/prefetch_hf_snapshot.sh Wan-AI/Wan2.2-TI2V-5B-Diffusers"

tpu watch v6 -n 64 \
  --setup-cmd "$SETUP_CMD" \
  "$WATCH_BRANCH" \
  RUN_NAME="$RUN_NAME" \
  TRAIN_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/train \
  EVAL_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/val \
  OUTPUT_DIR="$OUTPUT_DIR" \
  MODEL_DIR=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  ACTION_ADAPTER_TYPE=pre_context \
  PRE_CONTEXT_TOKENS=8 \
  PRE_CONTEXT_HEADS=8 \
  SIDE_ADAPTER_NOISE_MODE=fresh \
  MAX_TRAIN_STEPS=30000 \
  CHECKPOINT_EVERY=100 \
  CHECKPOINT_KEEP_PERIOD=1000 \
  EVAL_EVERY=1000 \
  EVAL_MAX_BATCHES=4 \
  LOG_PERIOD=10 \
  SAVE_FINAL_CHECKPOINT=True \
  PER_DEVICE_BATCH_SIZE=8 \
  GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 \
  GLOBAL_BATCH_SIZE_TO_LOAD=512 \
  TFRECORD_SHUFFLE_BUFFER_SIZE=1024 \
  WANDB_PROJECT=maxdiffusion-wan-pre-context-adapter \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  bash bash_scripts/train_wan_side_adapter.sh \
  2>&1 | tee -a "logs/tpu_watch_${RUN_NAME}.log"
```

For a side-adapter run, change:

```bash
ACTION_ADAPTER_TYPE=side_adapter
OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter
WANDB_PROJECT=maxdiffusion-wan-side-adapter
```

Do not use `--force` unless you have verified the target TPU slice is idle and
old training processes are stopped on all workers. Drop `--force` when creating
a new queued TPU.

### Visual validation

Training loss alone is not enough for these experiments. Run periodic visual
validation on a separate TPU slice. The watcher copies each target checkpoint
to a temporary validation prefix so the training checkpoint manager cannot
prune it while validation is queued.

```bash
COMMIT="$(git rev-parse HEAD)"
RUN_NAME=<training-run-name>
OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter

WATCH_BRANCH=adaptor \
COMMIT="$COMMIT" \
RUN_NAME="$RUN_NAME" \
OUTPUT_DIR="$OUTPUT_DIR" \
EVAL_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/val \
ACTION_ADAPTER_TYPE=pre_context \
PRE_CONTEXT_TOKENS=8 \
PRE_CONTEXT_HEADS=8 \
VALIDATION_TPU_NAME=<your-v6e-8-validation-tpu-name> \
VALIDATION_TPU_CHIPS=8 \
VALIDATION_FIRST_STEP=100 \
VALIDATION_EVERY=1000 \
NUM_EVAL_VIDEOS=4 \
HF_HUB_DISABLE_XET=1 \
HF_HUB_ENABLE_HF_TRANSFER=0 \
nohup bash bash_scripts/watch_wan_side_adapter_validation.sh \
  > "logs/wan_side_adapter_validation_watch_${RUN_NAME}.log" 2>&1 &
```

Validation writes summaries and videos under:

```text
<OUTPUT_DIR>/<RUN_NAME>/validation/
```

Inspect `summary.json`, `summary.csv`, and generated comparison videos before
selecting a checkpoint. Scheduler success or checkpoint creation is not enough.

### WAN TPU training pitfalls

- The DROID action-adapter trainer uses one-step denoising flow-matching loss,
  not rollout MSE. Rollout is only for visual generation and validation.
- Use `SIDE_ADAPTER_NOISE_MODE=fresh` for current scratch training runs.
- Use `CHECKPOINT_KEEP_PERIOD=1000` with `CHECKPOINT_EVERY=100` if validation
  should run every 1000 steps. Otherwise intermediate targets may be pruned.
- Keep `HF_HUB_DISABLE_XET=1` and `HF_HUB_ENABLE_HF_TRANSFER=0` unless every
  TPU worker has proven those transfer backends reliable.
- Resume restores adapter params, optimizer state, and integer step. The
  TFRecord iterator cursor is rebuilt after restore.
- Do not duplicate launches into the same checkpoint directory.
