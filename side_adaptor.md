# Wan2.2 TI2V 5B Side Adaptor

This document records the implementation methodology and experiment results for
the Wan2.2 TI2V 5B side-adaptor training in MaxDiffusion. The active branch for
this work is `adaptor`; the implementation was developed in the worktree
`/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`.

## Current Status

- The Wan2.2 TI2V 5B side-adaptor model, trainer, data path, validation path,
  and training launch path have been implemented.
- The converted full DROID latent dataset is on GCS:
  - Train: `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
  - Val: `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- The correct training objective is one-step denoising loss with fresh random
  noise per example. The earlier full-rollout objective and fixed-noise
  denoising run are retained below as invalid or diagnostic experiments.
- The best aggregate validation result on the fixed four-sample validation set
  came from the fresh-noise run at step 2000. The final step 10000 checkpoint is
  valid, but it is not the best checkpoint on that fixed subset.
- v6e-64 training with global batch size 512 fits using pure FSDP.

## Implementation Methodology

### Model And Trainer

The implementation ports the side-adaptor training used in `../Wan2.2` into the
MaxDiffusion Wan2.2 TI2V 5B stack.

Primary files:

- `src/maxdiffusion/models/wan/side_adapter_wan.py`
  - Adds the side adaptor around the Wan2.2 TI2V transformer.
  - Encodes low-dimensional robot actions into action tokens.
  - Injects side residual streams after selected transformer blocks.
  - Uses zero-initialized residual output projections so the initial model starts
    close to the frozen base model behavior.
  - Provides rollout and sigma helper utilities used by validation.
  - Pins the first latent video frame to the conditioning first-frame latent.
- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`
  - Loads and freezes the Wan2.2 TI2V 5B backbone.
  - Builds optimizer state only for adaptor parameters.
  - Computes the side-adaptor denoising objective.
  - Handles dataset batches with shape checks and normalization assumptions.
- `src/maxdiffusion/configs/base_wan_5b_side_adapter.yml`
  - Defines the default side-adaptor model, data, optimizer, parallelism, and
    validation settings.
- `bash_scripts/train_wan_side_adapter.sh`
  - Launch wrapper for TPU training.
  - Important: set `SIDE_ADAPTER_NOISE_MODE=fresh` explicitly for current
    training. The wrapper has historically defaulted this value to `fixed`, so
    relying on the wrapper default can reproduce the diagnostic fixed-noise run
    instead of the corrected run.

The trainable/frozen split was verified during smoke tests and training:

- Frozen Wan2.2 TI2V transformer: about 5.00B parameters.
- Trainable side adaptor: about 239.5M parameters.
- Optimizer state is created for adaptor parameters only.
- Backbone parameters are frozen and excluded from the train state update.

### Data Schema

The converted TFRecord examples contain cached Wan latent windows and robot
actions:

| Field | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `z_i0` | fp16 | `[48, 1, 12, 20]` | First-frame image latent condition |
| `z_video` | fp16 | `[48, 9, 12, 20]` | Target video latent sequence |
| `actions` | fp32 | `[32, 7]` | Robot action sequence |

Batch shapes used by training:

| Tensor | Shape |
| --- | --- |
| `z_i0` | `[B, 48, 1, 12, 20]` |
| `z_video` | `[B, 48, 9, 12, 20]` |
| `actions` | `[B, 32, 7]` |
| model velocity output | `[B, 48, 9, 12, 20]` |

The latent resolution corresponds to `height=192`, `width=320`, and
`num_frames=32`. The VAE latent grid is therefore `12 x 20` spatially, with 9
latent time frames for the video path. Validation videos are decoded as 33-frame
videos at 16 fps, with comparison videos rendered as ground truth on top and
prediction on bottom.

### Data Conversion

The DROID cached latent dataset was converted to MaxDiffusion-compatible
TFRecords with storage guardrails:

- The conversion streamed data rather than staging the full converted dataset on
  local storage.
- Temporary work used `a1001`; raw data was read from Della-side cached data.
- Completed shards were uploaded to `gs://v6_east1d` and local temporary files
  were deleted after transfer.
- The `/luster/.../lzha` storage budget was preserved by avoiding large local
  accumulation and cleaning temporary conversion outputs.

Final converted dataset:

| Split | GCS path | Shards | Examples |
| --- | --- | ---: | ---: |
| Train | `gs://v6_east1d/datasets/droid_wan_side_adapter/train` | 704 | 1,440,554 |
| Val | `gs://v6_east1d/datasets/droid_wan_side_adapter/val` | 8 | 14,636 |

Validation checks:

- Train shards are contiguous from `train-00000-of-00704.tfrecord` through
  `train-00703-of-00704.tfrecord`.
- Val shard 0 contains 2048 records; final val shard contains 300 records.
- Representative pure-Python TFRecord/protobuf reads passed for train shards
  `0`, `163`, `307`, `339`, `371`, `499`, `611`, `659`, `675`, `691`, and
  `703`.
- Final partial train shard was read successfully.
- Field byte lengths matched the schema above.

### Training Objective

The source PyTorch Wan2.2 side-adaptor run uses a conventional denoising
diffusion objective. The training step samples one diffusion timestep per
example, not a full rollout loss.

Correct objective:

```text
eps ~ N(0, I)
t ~ Uniform({0, ..., num_train_timesteps - 1})
sigma_t = shifted_sigma_grid[t]
z_t = (1 - sigma_t) * z_video + sigma_t * eps
z_t[:, :, 0] = z_i0[:, :, 0]

v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
target = eps - z_video
loss = mean((v_pred - target)^2 over non-frame-0 latent elements)
```

Important corrections made during development:

- The first full run used an incorrect 25-step rollout MSE inside the training
  step. That was stopped and marked invalid because it did not match the source
  Wan2.2 training recipe and was unnecessarily slow.
- A later denoising run used a fixed broadcast noise tensor during training.
  Visual inspection showed decoded ground truth videos were coherent but
  predictions were poor. The diagnosis was a train/validation mismatch:
  validation and generation require fresh noise, while training had used fixed
  noise.
- The corrected run uses fresh per-example Gaussian noise during training.

### Parallelism And Batch Size

Final v6e-64 training uses pure FSDP:

| Mesh axis | Value |
| --- | --- |
| `ici_fsdp_parallelism` | `-1` |
| `dcn_fsdp_parallelism` | `-1` |
| `ici_data_parallelism` | `1` |
| `dcn_data_parallelism` | `1` |
| `ici_context_parallelism` | `1` |
| `dcn_context_parallelism` | `1` |
| `ici_tensor_parallelism` | `1` |
| `dcn_tensor_parallelism` | `1` |

Batch and optimizer settings for the main corrected run:

| Setting | Value |
| --- | --- |
| TPU | v6e-64 |
| per-device batch size | 8 |
| global batch size | 512 |
| train/load batch size | 512 |
| optimizer | AdamW |
| learning rate | `5e-5` |
| warmup fraction | `0.05` |
| weight decay | `1e-2` |
| max train steps | 10000 |
| checkpoint cadence | 100 steps |
| eval cadence | 1000 steps |
| validation max batches | 4 |
| precision | bfloat16 weights and activations |

The initial batch-size ceiling was caused by the wrong mesh choice and retained
rollout residuals. After moving to pure FSDP, stopping gradients through the CFG
unconditional branch, and rematerializing rollout internals used for validation,
global batch size 512 fit on v6e-64.

### Validation Protocol

Validation is periodic and follows the generation path rather than the one-step
training objective:

- Copy a checkpoint to `validation_checkpoints/<step>` under the run output.
- Launch validation on a TPU worker.
- Run 25-step rollout sampling for a fixed validation subset.
- Decode four validation samples.
- Write per-sample and mean metrics:
  - latent MSE
  - decoded pixel MSE
  - decoded SSIM
- Write artifacts:
  - comparison MP4, ground truth top and prediction bottom, `320 x 384`
  - ground-truth MP4, `320 x 192`
  - prediction MP4, `320 x 192`
  - decoded frame grids
  - validation summary JSON
- Delete the temporary copied validation checkpoint after inspection.

Artifact inspection should use `viz-open`. This was added to the development
skill guidance after the validation inspection work.

## Experiment Results

### Conversion Results

| Stage | Result |
| --- | --- |
| Val conversion | Completed, 8 shards, 14,636 examples |
| Train conversion | Completed, 704 shards, 1,440,554 examples |
| GCS placement | Completed under `gs://v6_east1d/datasets/droid_wan_side_adapter` |
| Representative readback | Passed |
| Final shard readback | Passed |
| Local storage cleanup | Completed for temporary conversion artifacts |

### Smoke And Batch-Fit Results

| Experiment | Objective/noise | Batch | Result | Notes |
| --- | --- | ---: | --- | --- |
| v4 setup smoke | config/model checks | 1 | Passed | Verified environment, sigma schedule, and basic model path |
| v6e-64 loop smoke | early smoke | 1 | Passed | Losses: 2.578125, 4.781250, 2.906250 |
| early probe | wrong rollout objective | 8 | Passed | Losses: 3.343750, 2.343750 |
| early probe | wrong rollout objective | 12 | Passed | Losses: 3.265625, 1.093750 |
| early probe | wrong rollout objective | 14 | Passed | Losses: 3.453125, 1.617188 |
| early probe | wrong rollout objective | 15 | Passed | Losses: 3.453125, 1.593750 |
| early probe | wrong rollout objective | 16 | Failed | Compile-time HBM OOM before FSDP/remat fixes |
| first gbs256 probe | denoise path not fully rematted | 256 | Failed | Compile-time HBM OOM due retained rollout residuals |
| corrected gbs256 probe | denoise | 256 | Passed | Losses: 3.296875, 3.390625 |
| corrected gbs512 probe | denoise | 512 | Passed | Losses: 3.250000, 2.687500; no OOM/NaN |

### Main Training Runs

| Run | Status | Objective | Noise | Batch | Result |
| --- | --- | --- | --- | ---: | --- |
| `wan-side-adapter-v6e64-full-gbs512-ckpt100-r9-20260615-063300` | Stopped | 25-step rollout MSE | rollout noise | 512 | Invalid objective; stopped after confirming source uses denoising loss |
| `wan-side-adapter-v6e64-full-gbs512-denoise-ckpt100-r19-20260615-141659` | Completed | one-step denoising | fixed | 512 | Final train loss about 0.2859, but validation and visuals were poor |
| `wan-side-adapter-v6e64-full-gbs512-fresh-denoise-ckpt100-r20-20260616-065855` | Completed | one-step denoising | fresh | 512 | Final train loss 0.323785; much better validation than fixed-noise run |

The r9 run is not comparable because it optimized the wrong objective. The r19
run is useful as a diagnostic: it showed that training loss alone can look good
while generation quality remains poor if the training noise distribution is
wrong. The r20 run is the current valid training run.

### Fixed-Noise Denoising Run: r19

Run:
`wan-side-adapter-v6e64-full-gbs512-denoise-ckpt100-r19-20260615-141659`

Training:

- Global batch size: 512.
- Objective: one-step denoising loss.
- Noise: fixed broadcast noise.
- Final retry W&B run: `z0uuxuue`.
- Step 10000 train loss: `0.28593420684337617`.
- Checkpoint 10000 was written.

Validation metrics on the fixed four-sample subset:

| Step | Mean latent MSE | Mean pixel MSE | Mean SSIM |
| ---: | ---: | ---: | ---: |
| 5000 | 1.5004 | 0.1043 | 0.3416 |
| 6000 | 1.5599 | 0.1063 | 0.3326 |
| 7000 | 1.6222 | 0.1088 | 0.3206 |
| 8000 | 1.5957 | 0.1086 | 0.3262 |
| 9000 | 1.5810 | 0.1084 | 0.3201 |
| 10000 | 1.6042 | 0.1097 | 0.3212 |

Visual conclusion:

- Decoded ground-truth videos were coherent.
- Prediction videos were nonblank and correctly encoded, but contained severe
  artifacts and weak temporal/scene consistency.
- The failure was traced to fixed training noise. The model learned under a
  distribution that did not match validation/generation noise.

### Fresh-Noise Denoising Run: r20

Run:
`wan-side-adapter-v6e64-full-gbs512-fresh-denoise-ckpt100-r20-20260616-065855`

Training:

- TPU: v6e-64 slice `v6-64-12-lzha` in `us-east1-d`.
- GCS output root:
  `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter`
- Global batch size: 512.
- Per-device batch size: 8.
- Objective: one-step denoising loss.
- Noise: fresh random Gaussian noise per example.
- Checkpoint cadence: 100 steps.
- Validation cadence: 1000 steps.
- W&B run: `j17wnp4x`.
- Training resumed after maintenance and completed checkpoint 10000.

Selected training losses:

| Step | Train loss | Notes |
| ---: | ---: | --- |
| 10 | 0.610243 | Initial corrected fresh-noise training |
| 100 | 0.398474 | Early training |
| 200 | 0.412408 | Early training |
| 270 | 0.394279 | Before maintenance interruption |
| 610 | 0.352888 | After resume |
| 700 | 0.348239 | After resume |
| 760 | 0.354832 | After resume |
| 10000 | 0.323785 | Final checkpoint, grad norm 1.247, about 0.897 steps/s |

Validation metrics on the fixed four-sample subset:

| Step | Mean latent MSE | Mean pixel MSE | Mean SSIM |
| ---: | ---: | ---: | ---: |
| 1000 | 0.501075 | 0.035048 | 0.543436 |
| 2000 | 0.354568 | 0.025436 | 0.663717 |
| 3000 | 0.416834 | 0.029608 | 0.607507 |
| 4000 | 0.496580 | 0.035743 | 0.550084 |
| 5000 | 0.545641 | 0.036928 | 0.508979 |
| 6000 | 0.444165 | 0.030355 | 0.580033 |
| 7000 | 0.376921 | 0.025593 | 0.638788 |
| 8000 | 0.377103 | 0.026553 | 0.634340 |
| 9000 | 0.396358 | 0.025859 | 0.615188 |
| 10000 | 0.397113 | 0.026944 | 0.615260 |

Per-sample metrics for r20 step 10000:

| Sample | Latent MSE | Pixel MSE | SSIM | Visual note |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.281489 | 0.013569 | 0.6986 | Best final sample; coarse layout and motion preserved |
| 1 | 0.287343 | 0.022948 | 0.7242 | Good structure on final checkpoint |
| 2 | 0.557821 | 0.044859 | 0.4445 | Persistent failure; bright/yellow wrong-scene artifacts and smear |
| 3 | 0.461800 | 0.026403 | 0.5938 | Some structure, but ghosting and temporal artifacts |

Visual conclusion:

- Fresh noise substantially improved generation metrics and videos compared with
  fixed noise.
- Step 2000 was the best aggregate checkpoint on the fixed validation subset.
- Step 7000 was the best later checkpoint and had the cleanest sample 0.
- Step 10000 remained valid but did not improve over step 2000 on the fixed
  validation subset.
- Sample 2 remained a hard failure case across late checkpoints, suggesting the
  next debugging target should be broader validation, action/latent alignment
  checks on that example, and learning-rate or overtraining analysis.

## Artifacts

Important GCS roots:

| Artifact | Path |
| --- | --- |
| Train TFRecords | `gs://v6_east1d/datasets/droid_wan_side_adapter/train` |
| Val TFRecords | `gs://v6_east1d/datasets/droid_wan_side_adapter/val` |
| Checkpoints and validation outputs | `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-side-adapter` |

Local validation artifacts were staged under
`/home/lzha/code/maxdiffusion_artifacts/`. Use `viz-open` to inspect MP4s,
frame grids, and decoded comparisons.

## Reproduction Notes

Use `bash_scripts/setup.sh` to set the environment before TPU training or
validation.

For the corrected training recipe, launch on a v6e-64 TPU with pure FSDP and
set fresh noise explicitly. The essential launch settings are:

```bash
SIDE_ADAPTER_NOISE_MODE=fresh \
PER_DEVICE_BATCH_SIZE=8 \
GLOBAL_BATCH_SIZE=512 \
TRAIN_BATCH_SIZE=512 \
MAX_TRAIN_STEPS=10000 \
CHECKPOINT_EVERY=100 \
EVAL_EVERY=1000 \
EVAL_MAX_BATCHES=4 \
bash bash_scripts/train_wan_side_adapter.sh
```

Before starting a new run, confirm that the launch command resolves to:

- `model_type=SIDE_ADAPTER_TI2V`
- `side_adapter_noise_mode=fresh`
- train dataset path
  `gs://v6_east1d/datasets/droid_wan_side_adapter/train`
- validation dataset path
  `gs://v6_east1d/datasets/droid_wan_side_adapter/val`
- pure FSDP parallelism
- global batch size 512 on v6e-64
- optimizer state only for side-adaptor parameters

For validation, copy the target checkpoint into a temporary
`validation_checkpoints/<step>` directory, run the 25-step rollout validation
job, inspect artifacts with `viz-open`, then remove the temporary copied
checkpoint after validation finishes.

## Open Follow-Ups

- Evaluate r20 step 2000 and step 7000 on a larger validation subset, not only
  the fixed four-sample subset.
- Investigate the persistent sample 2 failure with direct checks of action
  sequence, source latent window, first-frame condition, and decoded ground
  truth.
- Test whether lower learning rate, EMA, or earlier stopping improves late-run
  degradation.
- Keep `SIDE_ADAPTER_NOISE_MODE=fresh` explicit in launch scripts until the
  wrapper default is audited and changed.
