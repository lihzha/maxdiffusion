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



## Wan2.2 TI2V 5B TPU Sample Commands

Run from the repo root on the machine where you use `tpu watch`. The training
command is the `tpu watch v6 -n 64 ... bash bash_scripts/train_wan_side_adapter.sh`
block. The worker setup and model prefetch happen inside `SETUP_CMD`.

```bash
# Common settings.
mkdir -p logs
export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
if [ -f "$HOME/.config/irom-tpu/secrets.env" ]; then
  set +x
  source "$HOME/.config/irom-tpu/secrets.env"
fi

export TPU_PROJECT=mae-irom-lab-guided-data
export TPU_ZONE_v6=us-east1-d
export TPU_BUCKET_v6=gs://v6_east1d
export GH_OWNER=lihzha
export GH_REPO_NAME=maxdiffusion
export WATCH_BRANCH=adaptor
export TPU_NAME=your-v6e-64-training-tpu-name
export VALIDATION_TPU_NAME=your-v6e-8-validation-tpu-name
export COMMIT="$(git rev-parse HEAD)"

export MODEL_DIR=Wan-AI/Wan2.2-TI2V-5B-Diffusers
export TRAIN_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/train
export EVAL_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/val
export PRE_CONTEXT_TOKENS=8
export PRE_CONTEXT_HEADS=8
export SIDE_ADAPTER_NOISE_MODE=fresh
export CHECKPOINT_EVERY=100
export CHECKPOINT_KEEP_PERIOD=1000
export EVAL_EVERY=1000
export EVAL_MAX_BATCHES=4
export LOG_PERIOD=10
export SAVE_FINAL_CHECKPOINT=True
export PER_DEVICE_BATCH_SIZE=8
export GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512
export GLOBAL_BATCH_SIZE_TO_LOAD=512
export TFRECORD_SHUFFLE_BUFFER_SIZE=1024
export SLEEP_SECS=15
export TPU_WATCH_HEALTH_CHECK_SECS=15
unset WANDB_DISABLED

# Choose exactly one experiment.
export WAN_EXPERIMENT=pre_context
# export WAN_EXPERIMENT=side_adapter

case "$WAN_EXPERIMENT" in
  pre_context)
    export ACTION_ADAPTER_TYPE=pre_context
    export OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter
    export WANDB_PROJECT=maxdiffusion-wan-pre-context-adapter
    export MAX_TRAIN_STEPS=30000
    export RUN_NAME="wan-pre-context-v6e64-full-gbs512-fresh-$(date -u +%Y%m%d-%H%M%S)"
    ;;
  side_adapter)
    export ACTION_ADAPTER_TYPE=side_adapter
    export OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter
    export WANDB_PROJECT=maxdiffusion-wan-side-adapter
    export MAX_TRAIN_STEPS=10000
    export RUN_NAME="wan-side-adapter-v6e64-full-gbs512-fresh-$(date -u +%Y%m%d-%H%M%S)"
    ;;
  *)
    echo "WAN_EXPERIMENT must be pre_context or side_adapter" >&2
    exit 1
    ;;
esac

# Optional one-step smoke. Leave this commented for a full run.
# export RUN_NAME="smoke-${ACTION_ADAPTER_TYPE}-$(date -u +%Y%m%d-%H%M%S)"
# export MAX_TRAIN_STEPS=1
# export CHECKPOINT_EVERY=0
# export CHECKPOINT_KEEP_PERIOD=
# export EVAL_EVERY=0
# export EVAL_MAX_BATCHES=
# export SAVE_FINAL_CHECKPOINT=False
# export WANDB_PROJECT=

# Launch training on v6e-64.
export SETUP_CMD="export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=120 HF_PREFETCH_ATTEMPTS=6 HF_PREFETCH_WORKERS=2 && git fetch origin ${WATCH_BRANCH} && git checkout --detach ${COMMIT} && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu && bash bash_scripts/prefetch_hf_snapshot.sh ${MODEL_DIR}"

nohup bash -lc '
tpu watch v6 -n 64 \
  --setup-cmd "$SETUP_CMD" \
  "$WATCH_BRANCH" \
  RUN_NAME="$RUN_NAME" \
  TRAIN_DATA_DIR="$TRAIN_DATA_DIR" \
  EVAL_DATA_DIR="$EVAL_DATA_DIR" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  MODEL_DIR="$MODEL_DIR" \
  ACTION_ADAPTER_TYPE="$ACTION_ADAPTER_TYPE" \
  PRE_CONTEXT_TOKENS="$PRE_CONTEXT_TOKENS" \
  PRE_CONTEXT_HEADS="$PRE_CONTEXT_HEADS" \
  SIDE_ADAPTER_NOISE_MODE="$SIDE_ADAPTER_NOISE_MODE" \
  MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
  CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
  CHECKPOINT_KEEP_PERIOD="$CHECKPOINT_KEEP_PERIOD" \
  EVAL_EVERY="$EVAL_EVERY" \
  EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" \
  LOG_PERIOD="$LOG_PERIOD" \
  SAVE_FINAL_CHECKPOINT="$SAVE_FINAL_CHECKPOINT" \
  PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" \
  GLOBAL_BATCH_SIZE_TO_TRAIN_ON="$GLOBAL_BATCH_SIZE_TO_TRAIN_ON" \
  GLOBAL_BATCH_SIZE_TO_LOAD="$GLOBAL_BATCH_SIZE_TO_LOAD" \
  TFRECORD_SHUFFLE_BUFFER_SIZE="$TFRECORD_SHUFFLE_BUFFER_SIZE" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  bash bash_scripts/train_wan_side_adapter.sh
' > "logs/tpu_watch_${RUN_NAME}.log" 2>&1 &
echo $! > "logs/tpu_watch_${RUN_NAME}.pid"

# Launch visual validation on v6e-8.
export VALIDATION_TPU_CHIPS=8
export VALIDATION_FIRST_STEP=100
export VALIDATION_MIN_STEP=100
export VALIDATION_EVERY=1000
export NUM_EVAL_VIDEOS=4
export VALIDATION_FORCE=0
export VALIDATION_CACHE_CHECKPOINTS=1
export VALIDATION_DELETE_CACHED_CHECKPOINT=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

nohup bash bash_scripts/watch_wan_side_adapter_validation.sh \
  > "logs/wan_side_adapter_validation_watch_${RUN_NAME}.log" 2>&1 &
echo $! > "logs/wan_side_adapter_validation_watch_${RUN_NAME}.pid"

# Output locations and logs.
echo "checkpoints: ${OUTPUT_DIR}/${RUN_NAME}/checkpoints/"
echo "validation:  ${OUTPUT_DIR}/${RUN_NAME}/validation/"
tail -f \
  "logs/tpu_watch_${RUN_NAME}.log" \
  "logs/wan_side_adapter_validation_watch_${RUN_NAME}.log"
```

Do not duplicate launches into the same `${OUTPUT_DIR}/${RUN_NAME}`. Keep
`SIDE_ADAPTER_NOISE_MODE=fresh`, `PRE_CONTEXT_TOKENS=8`,
`PRE_CONTEXT_HEADS=8`, and `CHECKPOINT_KEEP_PERIOD=1000` for these runs. Use
validation summaries and videos, not training loss alone, to choose checkpoints.
