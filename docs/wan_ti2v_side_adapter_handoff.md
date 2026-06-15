# Wan2.2 TI2V Side-Adapter Handoff

This document is the handoff for continuing Wan2.2 TI2V 5B side-adapter work
in MaxDiffusion, including implementing new adapter variants or launching the
DROID training run.

## Current State

- Main development branch before handoff: `codex/wan-ti2v-side-adapter-20260613-073227`.
- Handoff branch: `adaptor`.
- The side-adapter model/trainer is implemented for `MODEL_TYPE=SIDE_ADAPTER_TI2V`.
- The converted cached DROID dataset lives on the v6 bucket:
  - train: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
  - val: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- Use v6 TPUs for training. The established full run target is v6e-64 with
  global batch size 512.
- Run `bash_scripts/setup.sh MODE=stable DEVICE=tpu` on TPU workers during
  setup. Do not rely on local workstation package state for TPU validation.

## Important Correction

The production DROID side-adapter run in `../Wan2.2/run_train_droid.sh` uses
the default `--loss_type denoise`, not rollout loss. Training must sample one
diffusion step per example and optimize the flow-matching velocity target:

```text
eps ~ N(0, I) or fixed eps_0
t ~ Uniform({0, ..., N - 1})
sigma_t = shifted_sigma_grid[t]
z_t = (1 - sigma_t) * z_video + sigma_t * eps
z_t[:, :, 0] = z_i0[:, :, 0]
v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
v_target = eps - z_video
loss = mean((v_pred - v_target)^2 over non-frame-0 latent elements)
```

The previous r9 run used a full rollout MSE inside the training step. It was
stopped because that objective is wrong for this stage and is about 25x more
expensive per step. Rollout is still correct for visual generation/validation.

## Key Files

- `src/maxdiffusion/models/wan/side_adapter_wan.py`
  - NNX side-adapter modules and `wan_side_adapter_forward`.
  - Extension point for new adapter architectures.
  - `build_rollout_sigmas` builds the shifted sigma grid shared by training
    timestep sampling and generation.
- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`
  - Loads the frozen Wan2.2 TI2V 5B transformer.
  - Builds trainable adapter modules.
  - Keeps transformer parameters outside the optimizer state.
  - Implements the one-step denoising loss.
  - Reads cached TFRecords.
- `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`
  - Default training/data/model config.
  - Current side-adapter defaults match the original DROID run:
    `side_adapter_sampling_steps=25`, `side_adapter_noise_mode=fixed`,
    `side_adapter_t_sampling=uniform`, `side_adapter_guide_scale=5.0`,
    `flow_shift=5.0`.
- `bash_scripts/train_wan_side_adapter.sh`
  - TPU training wrapper.
  - Defaults Hugging Face Xet and hf_transfer off because r7-r9 showed those
    transfer backends can fail on multi-host TPU setup.
- `bash_scripts/watch_wan_side_adapter_validation.sh`
  - Local watcher that polls GCS checkpoints and launches visual validation
    jobs on a separate v6e-8 slice.
- `bash_scripts/validate_wan_side_adapter.sh`
  - One-shot validation wrapper.
- `src/maxdiffusion/generate_wan_side_adapter.py`
  - Restores adapter checkpoints and generates validation videos through the
    full rollout sampler.
- `src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py`
  - Converter from the cached full DROID latent dataset to MaxDiffusion
    TFRecords.

## Model Inputs And Outputs

Training TFRecords contain one cached window per example:

- `z_i0`: float16 bytes reshaped to `[48, 1, 12, 20]`
- `z_video`: float16 bytes reshaped to `[48, 9, 12, 20]`
- `actions`: float32 bytes reshaped to `[32, 7]`

At train time the global batch is:

- `z_i0`: `[B, 48, 1, 12, 20]`
- `z_video`: `[B, 48, 9, 12, 20]`
- `actions`: `[B, 32, 7]`

The transformer predicts a velocity tensor with shape `[B, 48, 9, 12, 20]`.
The loss target is `eps - z_video`. Frame 0 is pinned to `z_i0` in `z_t` and
excluded from the MSE.

## Parallelism And Sharding

Current full-run settings are pure ICI FSDP:

```yaml
mesh_axes: ['data', 'fsdp', 'context', 'tensor']
ici_data_parallelism: 1
ici_fsdp_parallelism: -1
ici_context_parallelism: 1
ici_tensor_parallelism: 1
dcn_data_parallelism: 1
dcn_fsdp_parallelism: -1
dcn_context_parallelism: 1
dcn_tensor_parallelism: 1
```

On v6e-64 this uses all 64 devices for FSDP. The launch tested so far used
`PER_DEVICE_BATCH_SIZE=8`, `GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512`, and
`GLOBAL_BATCH_SIZE_TO_LOAD=512`. Adapter parameters and optimizer state are
replicated intentionally in `_shard_state`; the frozen 5B transformer follows
the MaxDiffusion logical axis rules and is not part of the optimizer.

If implementing another adapter, re-check:

- only adapter parameters appear in `state.params`;
- frozen transformer params remain in `state.transformer_params`;
- `_shard_state` does not accidentally replicate the large frozen backbone;
- the optimizer state is only for trainable adapter leaves;
- data batch axes still shard across all devices.

## Launch Full Training

Use the `adaptor` branch for `tpu watch`. Use the exact commit in the setup
command if reproducibility matters; `tpu watch` itself expects a branch name.

Example launch on an existing healthy v6e-64 slice:

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
export TPU_NAME=v6-64-07-lzha
export SLEEP_SECS=15
export TPU_WATCH_HEALTH_CHECK_SECS=15
unset WANDB_DISABLED

COMMIT="$(git rev-parse HEAD)"
RUN_NAME="wan-side-adapter-v6e64-full-gbs512-denoise-ckpt100-r10-$(date -u +%Y%m%d-%H%M%S)"
SETUP_CMD="export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 && git fetch origin adaptor && git checkout --detach ${COMMIT} && bash bash_scripts/setup.sh MODE=stable DEVICE=tpu"

tpu watch v6 -n 64 --force \
  --setup-cmd "$SETUP_CMD" \
  adaptor \
  RUN_NAME="$RUN_NAME" \
  TRAIN_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/train \
  EVAL_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/val \
  OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter \
  MODEL_DIR=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  MAX_TRAIN_STEPS=10000 \
  CHECKPOINT_EVERY=100 \
  EVAL_EVERY=1000 \
  EVAL_MAX_BATCHES=4 \
  LOG_PERIOD=10 \
  SAVE_FINAL_CHECKPOINT=False \
  PER_DEVICE_BATCH_SIZE=8 \
  GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512 \
  GLOBAL_BATCH_SIZE_TO_LOAD=512 \
  TFRECORD_SHUFFLE_BUFFER_SIZE=1024 \
  WANDB_PROJECT=maxdiffusion-wan-side-adapter \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_ENABLE_HF_TRANSFER=0 \
  bash bash_scripts/train_wan_side_adapter.sh \
  2>&1 | tee -a "logs/tpu_watch_${RUN_NAME}.log"
```

Drop `--force` when creating a new queued TPU instead of reusing a known idle
slice. Do not use `--force` until old training processes are verified stopped
on all workers.

## Periodic Validation

Start the local watcher after training is launched. It will validate checkpoint
100, then every 1000 steps, by creating or reusing a v6e-8 validation slice.

```bash
COMMIT="$(git rev-parse HEAD)"
RUN_NAME="<training-run-name>"

WATCH_BRANCH=adaptor \
COMMIT="$COMMIT" \
RUN_NAME="$RUN_NAME" \
OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter \
EVAL_DATA_DIR=gs://v6_east1d/datasets/droid_wan_side_adapter/val \
VALIDATION_TPU_NAME=v6-8-wan-val-lzha \
VALIDATION_TPU_CHIPS=8 \
VALIDATION_FIRST_STEP=100 \
VALIDATION_EVERY=1000 \
NUM_EVAL_VIDEOS=4 \
HF_HUB_DISABLE_XET=1 \
HF_HUB_ENABLE_HF_TRANSFER=0 \
nohup bash bash_scripts/watch_wan_side_adapter_validation.sh \
  > "logs/wan_side_adapter_validation_watch_${RUN_NAME}.log" 2>&1 &
```

Validation output goes to:

```text
gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter/<RUN_NAME>/validation/
```

Inspect the `summary.json` and generated videos. A clean process exit alone is
not enough.

## Storage Notes

- Keep datasets and checkpoints on `gs://v6_east1d` for v6 training.
- The user has a 15T limit under `/luster/.../lzha` and asked to leave at least
  2T free. Do not do full conversion or staging there unless free space is
  explicitly checked first.
- Temporary conversion work may use `a1001`, but delete local scratch shards
  after GCS transfer and readback verification.

## Implementing A New Adapter Type

Recommended path:

1. Add the new NNX module in `src/maxdiffusion/models/wan/side_adapter_wan.py`
   or a sibling module if it is large.
2. Keep the frozen WAN transformer outside the trainable module, like the
   current trainer does with `transformer_graphdef`, `transformer_params`, and
   `transformer_rest`.
3. Route the adapter through `wan_side_adapter_forward` or a new forward helper.
   Preserve these invariants:
   - `hidden_states` in/out shape is `[B, C, F, H, W]`;
   - per-token timesteps are `[B, seq_len]`;
   - frame 0 receives timestep 0 and is pinned in latent space;
   - the unconditional CFG branch does not depend on adapter params.
4. Update `WanTI2VSideAdapterTrainer._build_adapters` and the config keys.
5. Re-run cheap checks before TPU launch:
   - `python -m py_compile src/maxdiffusion/models/wan/side_adapter_wan.py`
   - `python -m py_compile src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`
   - `bash -n bash_scripts/train_wan_side_adapter.sh`
   - `bash -n bash_scripts/watch_wan_side_adapter_validation.sh`
   - `bash -n bash_scripts/validate_wan_side_adapter.sh`
6. Run a TPU smoke through `bash_scripts/setup.sh` before a full v6e-64 launch.

## Known Pitfalls

- Do not train the adapter with rollout MSE for this DROID stage. It is slow and
  not what the original run did.
- `side_adapter_sampling_steps=25` means the discrete sigma grid size for
  timestep sampling during training, not 25 DiT forwards per training step.
- Keep `HF_HUB_DISABLE_XET=1` and `HF_HUB_ENABLE_HF_TRANSFER=0` unless you have
  evidence the Hugging Face transfer backend is fixed on all workers.
- `tpu watch` wants a branch name. Use `adaptor` as the watch branch and put the
  exact commit checkout inside `--setup-cmd`.
- W&B online training requires secrets from `~/.config/irom-tpu/secrets.env`.
  Never print or commit those values.
- If restarting from an existing run name and older W&B steps overlap, worker
  logs are the authority for repeated step numbers.
