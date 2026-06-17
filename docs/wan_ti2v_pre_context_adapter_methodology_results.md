# Wan2.2 TI2V Pre-Context Adapter Methodology And Results

Last updated: 2026-06-17T17:51:01Z

This document summarizes the MaxDiffusion Wan2.2 TI2V pre-context action
adapter work that was developed from the `adaptor` branch, merged back to
`adaptor`, and trained on v6e TPUs. It is a handoff-style methodology and
results record, not a replacement for the raw worklog or GCS/W&B artifacts.

## Source State

- Canonical branch after merge: `adaptor`
- Merge commit on `adaptor`: `f35000e2e8e45c772041c42a10a8a75602c7b4aa`
- Main implementation commit included by the merge:
  `7260778fe9202382d3bdc0deba4977445f648408`
- Development branch:
  `codex/wan-ti2v-pre-context-adapter-v6e64`
- Active training worktree:
  `/home/lzha/code/maxdiffusion-worktrees/wan-prectx-adapter-20260615-073355`
- Clean merged adaptor worktree:
  `/home/lzha/code/.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227`

Merge validation before push:

- `bash -n bash_scripts/prefetch_hf_snapshot.sh`
- `bash -n bash_scripts/watch_wan_side_adapter_validation.sh`
- `python3 -m py_compile` on:
  - `src/maxdiffusion/input_pipeline/_tfds_data_processing.py`
  - `src/maxdiffusion/generate_wan_side_adapter.py`
  - `src/maxdiffusion/models/wan/side_adapter_wan.py`
  - `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`
- `git diff --cached --check`

## Methodology

### Adapter Objective

The new adapter type is selected with:

```text
ACTION_ADAPTER_TYPE=pre_context
```

The implementation keeps the original `side_adapter` path available and routes
both modes through `wan_action_adapter_forward`. The pre-context mode is
intended to mimic the upstream `../Wan2.2` action-conditioning style: predict
additional T5-space context tokens from visual/action features, then pass those
tokens into the frozen Wan transformer instead of injecting a residual adapter
inside many transformer layers.

Key implementation files:

- `src/maxdiffusion/models/wan/side_adapter_wan.py`
  - `NNXPreContextFeatureContextHead`
  - `WanActionAdapters(... action_adapter_type="pre_context" ...)`
  - `wan_pre_context_adapter_forward`
  - `wan_action_adapter_forward`
- `src/maxdiffusion/trainers/wan_ti2v_side_adapter_trainer.py`
  - adapter construction
  - denoising loss
  - checkpoint save/restore
  - training/eval loop
- `src/maxdiffusion/generate_wan_side_adapter.py`
  - validation rollout and video generation with the same adapter forward path

### Pre-Context Forward Path

The pre-context path follows this high-level sequence:

1. Build noisy TI2V latents from cached `z_i0`, cached target video latents,
   sampled diffusion timestep, and sampled noise.
2. Run context-free Wan feature extraction through the initial patch/time and
   first self-attention feature path.
3. Stop gradients through those context-free backbone features.
4. Encode the 32-step, 7-DoF action sequence.
5. Use `NNXPreContextFeatureContextHead` to cross-attend learned context queries
   over frozen features and action tokens, producing `pre_context_tokens`
   predicted context vectors.
6. Run the frozen Wan transformer using the predicted context tokens.
7. Optimize only adapter parameters; keep the 5B Wan transformer frozen and
   outside the optimizer state.

Production pre-context settings:

```text
PRE_CONTEXT_TOKENS=8
PRE_CONTEXT_HEADS=8
```

The head count was originally tried with 40 heads and failed because the actual
feature dimension was not divisible by the requested head count. The code now
uses 8 as the default and defensively selects a valid divisor.

### Dataset And Batch

Dataset:

```text
train: gs://v6_east1d/datasets/droid_wan_side_adapter/train
val:   gs://v6_east1d/datasets/droid_wan_side_adapter/val
```

Each TFRecord example contains one cached DROID latent window:

```text
z_i0:    float16 [48, 1, 12, 20]
z_video: float16 [48, 9, 12, 20]
actions: float32 [32, 7]
```

Full v6e-64 training settings:

```text
PER_DEVICE_BATCH_SIZE=8
GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512
GLOBAL_BATCH_SIZE_TO_LOAD=512
TFRECORD_SHUFFLE_BUFFER_SIZE=1024
```

Parallelism is pure ICI FSDP on all 64 v6e devices:

```text
ici_data_parallelism=1
ici_fsdp_parallelism=-1
ici_context_parallelism=1
ici_tensor_parallelism=1
dcn_data_parallelism=1
dcn_fsdp_parallelism=-1
dcn_context_parallelism=1
dcn_tensor_parallelism=1
```

### Loss

Training uses one-step denoising/flow-matching, not rollout loss:

```text
eps ~ noise sampler
t ~ configured timestep sampler
sigma_t = shifted sigma grid[t]
z_t = (1 - sigma_t) * z_video + sigma_t * eps
z_t[:, :, 0] = z_i0[:, :, 0]
v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
v_target = eps - z_video
loss = MSE(v_pred, v_target) over non-frame-0 latent elements
```

The baseline r4 run used the inherited fixed-noise default. The requested
scratch run uses:

```text
SIDE_ADAPTER_NOISE_MODE=fresh
```

Fresh noise is sampled with `jax.random.normal` from the training step RNG. The
fixed-noise mode instead reuses a seed-derived noise tensor across examples.

### Checkpointing And Resume Behavior

Checkpoint settings for the production runs:

```text
CHECKPOINT_EVERY=100
EVAL_EVERY=1000
EVAL_MAX_BATCHES=4
checkpoint_keep_period=5000
max_to_keep=3 with periodic keeps
```

Resume behavior:

- There is no `restore_dataloader_state=False` flag in this trainer path.
- The trainer does not save or restore a serialized dataloader iterator.
- Orbax restores only:
  - adapter `params`
  - optimizer `opt_state`
  - integer `step`
- The train TFRecord iterator is rebuilt after restore with:

```text
seed = config.seed + start_step
```

Implication: after a TPU maintenance/preemption, model and optimizer state
resume from the latest committed checkpoint, but the exact input iterator
cursor is not restored. For the fresh-noise run, new noise continues to be
sampled from the step RNG after restore.

### TPU Workflow

All v6/v6e work should use:

```text
zone: us-east1-d
bucket: gs://v6_east1d
```

The TPU workflow skill was updated on 2026-06-16 to encode this rule after an
incorrect central-zone attempt. Central v6 zones should not be used unless the
user explicitly overrides that rule in the same turn.

Launch/recovery is managed with `tpu watch v6 -n 64`, which:

1. submits or reuses a queued resource,
2. waits for allocation,
3. runs setup on all workers,
4. launches the MaxDiffusion training command,
5. monitors TPU health,
6. deletes/requeues on `UNHEALTHY_MAINTENANCE`.

Setup includes robust Hugging Face prefetch:

```text
MODEL_DIR=Wan-AI/Wan2.2-TI2V-5B-Diffusers
HF_HUB_DISABLE_XET=1
HF_HUB_ENABLE_HF_TRANSFER=0
HF_PREFETCH_ATTEMPTS=6
HF_PREFETCH_WORKERS=2
```

This was added after a worker-local Hugging Face 408 timeout caused a
distributed launch abort.

## Experiment Ledger

### Implementation And Smoke Checks

| Date | Run / change | Result |
| --- | --- | --- |
| 2026-06-15 | Implemented `pre_context` adapter mode | Local syntax checks passed. TPU runtime smoke still required because local Python lacked the full JAX TPU stack. |
| 2026-06-15 | Initial v6e-64 smoke attempts | Blocked by occupied slices, stale queued resources, and v6e quota/capacity. No model code ran. |
| 2026-06-15 | r5 smoke on `v6-64-09-lzha` | Setup completed, but training failed before first step due to Hugging Face `408 Request Time-out` on one worker. |
| 2026-06-15 | Added `prefetch_hf_snapshot.sh` and setup prefetch | Prevented the previous distributed HF download failure from recurring during JAX startup. |
| 2026-06-15 | r7 smoke after prefetch | Reached adapter construction, then failed with `feature_dim must be divisible by heads`. |
| 2026-06-15 | Head-count fix and `PRE_CONTEXT_HEADS=8` | Fixed adapter initialization. |
| 2026-06-15 | r8 one-step v6e-64 smoke | Completed 1 train step. Loss `42.409554`, grad norm `158.307`, lr `5e-05`, no NaN/traceback. |

### Fixed-Noise Baseline Run

Run:

```text
wan-pre-context-v6e64-full-gbs512-denoise-ckpt100-r4-east1d-20260616-035932
```

Config:

```text
ACTION_ADAPTER_TYPE=pre_context
PRE_CONTEXT_TOKENS=8
PRE_CONTEXT_HEADS=8
SIDE_ADAPTER_NOISE_MODE=fixed
MAX_TRAIN_STEPS=10000
CHECKPOINT_EVERY=100
EVAL_EVERY=1000
```

Important events:

| Step / event | Result |
| --- | --- |
| First metrics | Step 10 loss `42.461531`, step 30 loss `41.569269`; adapter params `128.8M`, frozen transformer `5.00B`. |
| Step 100 | Checkpoint committed successfully. |
| Step 1000 | Training crashed during eval before saving checkpoint 1000 because the TFRecord loader tried to read `summary.json` as a TFRecord. Latest committed checkpoint was 900. |
| Fix | Commit `9aa51d206b29acafd1bf0f53f6dce74fbd746c08` made the loader prefer `*.tfrecord` and avoid metadata files. |
| Resume from 900 | Restored params/optimizer/step from checkpoint 900; no dataloader cursor restore. |
| Step 1000 after fix | Eval path passed and checkpoint 1000 committed. No `DataLossError`, no corrupted record, no traceback. |
| Steps 3000-5000 | Loss improved from about `1.33` to about `0.54`; checkpoints and eval boundaries committed. |
| Steps 6000-9000 | Loss settled around `0.52 -> 0.49`; eval/checkpoint boundaries continued to pass. |
| Step 10000 | Final baseline checkpoint committed; task-owned resources cleaned up. |

Representative fixed-noise metrics:

| Step | Train loss | Grad norm | Notes |
| --- | ---: | ---: | --- |
| 3000 | 1.337230 | 13.998 | After resume and eval fix. |
| 4000 | 0.691586 | 6.212 | Large improvement around this interval. |
| 5000 | 0.553775 | 3.337 | Stable checkpoint/eval. |
| 6000 | 0.522735 | 4.289 | Stable. |
| 7000 | 0.515786 | 3.515 | Stable. |
| 8000 | 0.499242 | 3.899 | Stable. |
| 9000 | 0.490071 | 3.602 | Stable. |
| 10000 | checkpoint committed | n/a | Final fixed-noise baseline complete. |

Visual validation at checkpoint 900:

```text
num_samples=4
mean_latent_mse=1.5644839406013489
mean_pixel_mse=0.08236705139279366
mean_ssim=0.2982816883278164
```

Qualitative read: videos were valid and nonblank, but predictions were still
noisy/unstable at this early checkpoint.

### Fresh-Noise Continuation Attempt

After the fixed-noise 10000-step baseline, a continuation was queued with:

```text
SIDE_ADAPTER_NOISE_MODE=fresh
MAX_TRAIN_STEPS=30000
expected restore step=10000
```

This continuation was superseded by the user's instruction to start over from
scratch with fresh noise. The continuation did not become the final training
trajectory.

### Fresh-Noise Scratch Run

Active run:

```text
wan-pre-context-v6e64-full-gbs512-fresh-scratch-ckpt100-east1d-20260616-191526
```

Config:

```text
ACTION_ADAPTER_TYPE=pre_context
PRE_CONTEXT_TOKENS=8
PRE_CONTEXT_HEADS=8
SIDE_ADAPTER_NOISE_MODE=fresh
MAX_TRAIN_STEPS=30000
CHECKPOINT_EVERY=100
EVAL_EVERY=1000
EVAL_MAX_BATCHES=4
PER_DEVICE_BATCH_SIZE=8
GLOBAL_BATCH_SIZE_TO_TRAIN_ON=512
GLOBAL_BATCH_SIZE_TO_LOAD=512
TFRECORD_SHUFFLE_BUFFER_SIZE=1024
```

GCS output:

```text
gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-pre-context-adapter/wan-pre-context-v6e64-full-gbs512-fresh-scratch-ckpt100-east1d-20260616-191526/
```

W&B:

```text
https://wandb.ai/lihanzha/maxdiffusion-wan-pre-context-adapter/runs/3dy1iylr
https://wandb.ai/lihanzha/maxdiffusion-wan-pre-context-adapter/runs/r1q1n5pe
https://wandb.ai/lihanzha/maxdiffusion-wan-pre-context-adapter/runs/5pdildlb
```

The different W&B run URLs correspond to different replacement TPU launches or
different metric-primary workers after preemption. Worker logs are authoritative
for overlapped/resumed steps.

Early fresh-noise scratch results:

| Step | Train loss | Grad norm | Checkpoint |
| --- | ---: | ---: | --- |
| 10 | 43.705788 | n/a | none yet |
| 50 | 43.004725 | n/a | none yet |
| 100 | 39.420489 | n/a | committed |
| 200 | 4.205924 | 71.350 | committed |
| 300 | 2.492955 | 74.275 | committed |
| 400 | 2.392539 | 247.657 | committed |
| 500 | 2.201019 | 134.609 | committed |
| 600 | 2.167699 | 141.401 | committed |
| 700 | 2.361574 | 334.474 | committed |
| 800 | 2.385976 | 302.552 | committed |
| 900 | 2.540737 | 179.948 | committed |
| 1000 | 2.330696 | 170.834 | eval loss `2.224861`; committed |
| 1100 | 2.024808 | 51.138 | committed |
| 2000 | 2.100327 | 33.327 | eval loss `2.087445`; committed |
| 2200 | 1.983413 | 18.294 | committed |
| 2400 | latest durable before first maintenance | n/a | committed |

Recovery after maintenance:

- TPU maintenance interrupted the run after checkpoint 2400.
- Replacement restored from checkpoint 2400.
- Post-restore evidence:
  - step 2410 loss `1.965673`, grad norm `23.252`
  - step 2500 loss `1.911206`, grad norm `16.638`; checkpoint committed
  - step 2600 loss `1.872538`, grad norm `15.884`; checkpoint committed

Later fresh-noise scratch results after additional maintenance/requeue:

| Step | Train loss | Grad norm | Checkpoint |
| --- | ---: | ---: | --- |
| 26410 | 0.569720 | 4.309 | restored from 26400 |
| 26500 | 0.561134 | 3.825 | committed |
| 26600 | 0.558787 | 4.164 | committed |
| 26700 | 0.578451 | 4.444 | committed |
| 26800 | 0.564451 | 4.399 | committed |
| 26900 | 0.584789 | 4.159 | committed |
| 27000 | 0.572305 | 4.011 | not committed; maintenance hit at eval/checkpoint boundary |

Latest safe checkpoint as of this document:

```text
26900
```

Checkpoint 27000 is not a safe resume point:

```text
_CHECKPOINT_METADATA=no
commit_success=no
```

Current active recovery state as of 2026-06-17T17:51:01Z:

- TPU/QR: `v6-64-10-lzha` / `v6-64-10-lzha-qr`
- Zone: `us-east1-d`
- QR state: `ACTIVE`
- TPU state: `READY`
- TPU health: `HEALTHY`
- Watcher state: setup/HF prefetch is running on the replacement TPU after SSH
  key propagation.
- Expected restore: checkpoint `26900`.
- Required next verification:
  - setup completion,
  - remote commit `7260778fe9202382d3bdc0deba4977445f648408`,
  - `ACTION_ADAPTER_TYPE=pre_context`,
  - `SIDE_ADAPTER_NOISE_MODE=fresh`,
  - restore from checkpoint `26900`,
  - first post-restore train metrics,
  - checkpoint `27000` or next committed checkpoint.

## Visual Validation Results

Committed artifact root:

```text
artifacts/
```

Validated fresh-noise scratch checkpoints:

| Checkpoint | num samples | mean latent MSE | mean pixel MSE | mean SSIM | Qualitative read |
| --- | ---: | ---: | ---: | ---: | --- |
| 13300 | 4 | 1.5255145579576492 | 0.09962983056902885 | 0.26878652969996136 | Nonblank but poor, noisy, distorted predictions. |
| 17200 | 4 | 1.4145757555961609 | 0.09378330409526825 | 0.298560576682741 | Improved over 13300, still blurry/distorted with color smearing. |

Useful local artifact paths:

```text
artifacts/wan_ti2v_pre_context_validation_step_013300/comparison_midframe_contact_sheet.png
artifacts/wan_ti2v_pre_context_validation_step_017200/comparison_midframe_contact_sheet.png
artifacts/wan_ti2v_pre_context_validation_step_017200/sample_0000_ep10099_v0_s00000/comparison_gt_top_pred_bottom.mp4
```

Artifact viewer URL shape from `/home/lzha/code`:

```text
http://localhost:8765/view?path=.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227/artifacts/wan_ti2v_pre_context_validation_step_017200/comparison_midframe_contact_sheet.png
http://localhost:8765/view?path=.codex-worktrees/maxdiffusion-wan-ti2v-side-adapter-20260613-073227/artifacts/wan_ti2v_pre_context_validation_step_017200/sample_0000_ep10099_v0_s00000/comparison_gt_top_pred_bottom.mp4
```

No fresh validation video has yet been generated from checkpoint 26900 or later.
That requires a separate validation sidecar TPU, preferably `v6e-8` in
`us-east1-d`, after the active training recovery is stable.

## Known Failures And Fixes

| Issue | Symptom | Fix / outcome |
| --- | --- | --- |
| v6e quota/capacity | `RESOURCE_EXHAUSTED` or long `PROVISIONING` with intermittent `CREATING -> NOT_FOUND` | Keep one east1-d QR active; do not duplicate launches into the same checkpoint dir. |
| Wrong zone attempt | Tried central v6/v6e zones while debugging quota | Corrected workflow: v6/v6e must use `us-east1-d` unless explicitly overridden. |
| Shared remote checkout conflict | A remote shared `~/maxdiffusion` checkout was touched by another setup path | Stopped unsafe launch and relaunched from exact commits with direct verification. |
| HF download timeout | Worker-local `408 Request Time-out` on Wan model files | Added retryable prefetch before JAX distributed startup. |
| Pre-context head mismatch | `feature_dim must be divisible by heads` | Changed default/config to 8 heads and added defensive divisor logic. |
| Eval TFRecord loader | Eval crashed by trying to parse `summary.json` as TFRecord | Loader now prefers `*.tfrecord`; step-1000 eval passed after fix. |
| TPU maintenance | Multiple `UNHEALTHY_MAINTENANCE` events before/during setup and at checkpoint/eval boundaries | `tpu watch` deletes/requeues; resume uses latest committed checkpoint. |
| SSH public-key propagation | New replacement workers rejected raw SSH with `Permission denied (publickey)` | Direct `gcloud alpha compute tpus tpu-vm ssh ... --zone=us-east1-d` propagated keys; watcher retries proceeded. |

## Current Interpretation

The pre-context adapter implementation is merged and has passed distributed
v6e-64 execution. The fixed-noise baseline completed 10000 steps with stable
checkpoints and decreasing loss. The fresh-noise scratch run has also trained
substantially longer, reaching committed checkpoint 26900 before maintenance
interrupted the 27000 boundary.

The main remaining limitation is sample quality: visual rollouts at checkpoints
13300 and 17200 are nonblank and somewhat improving by metrics, but still poor
for usable behavior cloning/video prediction quality. A fresh validation run
from checkpoint 26900 or later is the next evidence target once the current
replacement TPU restores and resumes cleanly.
