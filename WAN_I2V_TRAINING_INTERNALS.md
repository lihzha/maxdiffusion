# WAN I2V Training Internals

A reference document covering the architecture, data flow, and conditioning strategy of WAN Image-to-Video (I2V) training in `src/maxdiffusion/trainers/wan_i2v_trainer.py`. Read alongside `WAN_TRAINING_INTERNALS.md`.

---

## Table of Contents

1. [Entry Point & Class Hierarchy](#1-entry-point--class-hierarchy)
2. [System Diagram](#2-system-diagram)
3. [I2V vs T2V: The Core Difference](#3-i2v-vs-t2v-the-core-difference)
4. [The Condition Tensor](#4-the-condition-tensor)
5. [History Frames: How Multi-Frame Conditioning Works](#5-history-frames-how-multi-frame-conditioning-works)
6. [Wan 2.1 vs Wan 2.2 I2V](#6-wan-21-vs-wan-22-i2v)
7. [Dataset Paths](#7-dataset-paths)
8. [DROID Sliding-Window Dataset](#8-droid-sliding-window-dataset)
9. [Training Step](#9-training-step)
10. [Inference Pipelines](#10-inference-pipelines)
11. [Checkpointing](#11-checkpointing)

---

## 1. Entry Point & Class Hierarchy

```
train_wan.py
│
├── WanI2VTrainer                              trainers/wan_i2v_trainer.py
│   └── BaseWanTrainer (ABC)                  trainers/base_wan_trainer.py
│
├── WanCheckpointerI2V_2_1  (wan2.1)          checkpointing/wan_checkpointer_i2v_2p1.py
│   └── WanCheckpointer (ABC)
│
├── WanCheckpointerI2V_2_2  (wan2.2)          checkpointing/wan_checkpointer_i2v_2p2.py
│   └── WanCheckpointer (ABC)
│
├── WanPipelineI2V_2_1  ◄── SINGLE XFMR      pipelines/wan/wan_pipeline_i2v_2p1.py
│   └── WanPipeline (ABC)
│
├── WanPipelineI2V_2_2  ◄── DUAL XFMR        pipelines/wan/wan_pipeline_i2v_2p2.py
│   └── WanPipeline (ABC)
│
└── WanModel  ◄── SHARED TRANSFORMER          models/wan/transformers/transformer_wan.py
    (same class as T2V, but in_channels=33 from pretrained config)
```

**Key fact:** `WanModel` is not subclassed for I2V. It already supports I2V via its `in_channels` and `image_dim` constructor parameters, both of which come from the pretrained checkpoint's `config.json`. No architectural subclass is needed.

---

## 2. System Diagram

```
__main__ → app.run(main) → train(config) → WanI2VTrainer.start_training()
│
├─ WanCheckpointerI2V_2_1/2_2.load_checkpoint()
│     ├─ Orbax checkpoint found → WanPipelineI2V.from_checkpoint()
│     └─ No checkpoint          → WanPipelineI2V.from_pretrained()
│          └─ loads: WanModel (in_channels=33) + AutoencoderKLWan (VAE)
│                  + UMT5EncoderModel (text encoder)
│                  + FlaxCLIPVisionModel (image encoder, wan2.1 only)
│
├─ load_dataset()
│     ├─ synthetic:  random tensors via make_data_iterator()
│     ├─ droid:      DroidVideoDataset → raw frames [B, T, H, W, 3]
│     └─ tfrecord:   pre-encoded {latents, encoder_hidden_states, condition, [image_embeds]}
│
├─ [droid path only] VAE + T5 + CLIP remain loaded — do NOT call delete_vae()
│
├─ create_scheduler() → FlaxFlowMatchScheduler (1000 timesteps)
├─ _create_optimizer() → AdamW + LR schedule
│
└─ training_loop()
      └─ for step in range(max_train_steps):
            ├─ [droid path] preprocess_batch()
            │     ├─ T5 encode text → encoder_hidden_states [B, 512, 4096]
            │     ├─ VAE encode full video → latents [B, 16, T', H', W']
            │     ├─ VAE encode first-frame-only → latent_cond [B, 16, T', H', W']
            │     ├─ build mask (1.0 at t'=0) → [B, 1, T', H', W']
            │     ├─ condition = concat(mask, latent_cond) → [B, 17, T', H', W']
            │     └─ [wan2.1] CLIP encode first frame → image_embeds [B, 257, 1280]
            │
            └─ i2v_train_step(state, batch, rng, scheduler_state)
                  ├─ sample_timesteps(rng, bsz)
                  ├─ apply_flow_match(noise, latents, t) → noisy_latents, target, weight
                  ├─ latent_model_input = concat(noisy_latents, condition)  [B, 33, T', H', W']
                  ├─ model(latent_model_input, timestep, text_embeds, [image_embeds])
                  ├─ MSE loss × training_weight → scalar
                  └─ state.apply_gradients(grads) → new TrainState
```

---

## 3. I2V vs T2V: The Core Difference

The only change at the transformer level is the number of input channels:

```
T2V forward pass:
  hidden_states = noisy_latents                          [B, 16, T', H', W']

I2V forward pass:
  hidden_states = concat(noisy_latents, condition, axis=1)  [B, 33, T', H', W']
                         └── [B, 16, …]   └── [B, 17, …]
```

Everything else — the transformer blocks, attention, RoPE, AdaLN, loss function, scheduler, sharding — is identical to T2V. The wider patch embedding (`in_channels: 16→33`) is the only architectural delta, and it lives in the pretrained checkpoint rather than code.

---

## 4. The Condition Tensor

The 17-channel `condition` tensor carries the image conditioning signal into every forward pass.

### 4a. Structure

```
condition [B, 17, T', H', W']
  channel 0:      mask        [B,  1, T', H', W']
  channels 1–16:  latent_cond [B, 16, T', H', W']
```

### 4b. Construction (training, channels-first)

```python
# _encode_video_i2v  (wan_i2v_trainer.py:182)

video = frames * 2.0 - 1.0                             # [B, T, H, W, 3] → [-1, 1]
latents   = vae_encode(video)                           # [B, 16, T', H', W']

first_frame = video[:, 0:1, :, :, :]                   # [B, 1, H, W, 3]
video_cond  = concat([first_frame, zeros(T-1)], axis=1) # [B, T, H, W, 3]
latent_cond = vae_encode(video_cond)                    # [B, 16, T', H', W']
                                                        # non-zero only at T'=0

mask = zeros((B, 1, T', H', W'))
mask[:, :, 0, :, :] = 1.0                              # mark first latent frame

condition = concat([mask, latent_cond], axis=1)         # [B, 17, T', H', W']
```

### 4c. What the transformer sees

- At temporal position `t'=0`: channels 1–16 carry the real VAE encoding of the first frame, channel 0 = 1.0.
- At all other positions: channels 1–16 are zero (the VAE sees a black frame and encodes near-zero latents), channel 0 = 0.0.

The model learns to denoise `noisy_latents` to the full video while receiving this first-frame anchor in the other 17 channels of every forward pass.

### 4d. Channel-axis convention

| Context | Format | Concatenation axis |
|---|---|---|
| Training (`_encode_video_i2v`) | channels-first `[B, C, T, H, W]` | `axis=1` |
| Inference (`prepare_latents`) | channels-last `[B, T, H, W, C]` | `axis=-1` |

The condition tensor is transposed to channels-first before entering `i2v_train_step`.

---

## 5. History Frames: How Multi-Frame Conditioning Works

"History frames" refers to the mechanism for conditioning generation on **more than one anchor frame** — specifically, specifying both the first and last frame of a clip. This is supported at inference time via the `last_image` parameter in `prepare_latents_i2v_base`.

### 5a. Single-frame conditioning (default)

```
video_condition = [first_frame | zeros × (T-1)]

mask layout (T' latent frames):
  t'=0:  1.0  ← conditioned on first_frame
  t'=1…: 0.0  ← model generates freely
```

### 5b. Dual-frame conditioning (`last_image` provided)

```python
# wan_pipeline.py:568
video_condition = concat([first_frame,
                           zeros × (T-2),
                           last_image], axis=2)  # [B, C, T, H, W]
```

```
mask layout (T' latent frames):
  t'=0:     1.0  ← conditioned on first_frame
  t'=1…T'-2: 0.0  ← model generates freely
  t'=T'-1:  1.0  ← conditioned on last_image
```

Both anchor frames are VAE-encoded and placed into `latent_cond`; the mask is 1.0 at their latent positions. The model must generate coherent intermediate frames that bridge start → end. This is the **outpainting / video interpolation** use case.

### 5c. Why not more anchor frames?

The mask channel is a binary float tensor — any temporal position can in principle be set to 1.0. The current implementation sets exactly one or two positions, but the architecture imposes no constraint. Extending to N anchor frames would only require changing how `video_condition` and `mask` are built before calling `vae_encode`.

### 5d. History frames in training

Training currently uses **single-frame conditioning only** (`last_image=None` always). The `_encode_video_i2v` method sets `mask[:, :, 0, :, :] = 1.0` and nowhere else. No multi-frame conditioning is applied during training — the model learns the first-frame anchor only.

### 5e. Temporal relationship: VAE downsampling

The VAE compresses time by 4×:

```
T input frames → T' = (T - 1) / 4 + 1 latent frames

Examples:
  T=17  → T'=5
  T=49  → T'=13
  T=81  → T'=21
```

`mask[:, :, 0, :, :]` corresponds to input frame 0 (the first 4 input frames collapse into one latent frame via `WanCausalConv3d`). When conditioning on `last_image`, `mask[:, :, T'-1, :, :]` corresponds to input frame `T-1`.

---

## 6. Wan 2.1 vs Wan 2.2 I2V

| Aspect | Wan 2.1 (14B-720P) | Wan 2.2 (A14B / 27B) |
|---|---|---|
| Transformer count | 1 | 2 (low-noise + high-noise) |
| Image conditioning | VAE latents **+** CLIP embeddings | VAE latents only |
| `encoder_hidden_states_image` | `[B, 257, 1280]` | `None` |
| `in_channels` | 33 | 33 |
| Timestep routing | All steps → same model | `t/T > boundary_ratio` → high-noise model |
| Default `boundary_ratio` | — | 0.875 |
| Checkpointer | `WanCheckpointerI2V_2_1` | `WanCheckpointerI2V_2_2` |
| Checkpoint items saved | 1 transformer state | 2 transformer states |

### CLIP image embedding (Wan 2.1 only)

```python
# _encode_clip  (wan_i2v_trainer.py:234)
pixels  → resize to 224×224
        → normalize with ImageNet mean/std
        → FlaxCLIPVisionModel
        → hidden_states[-2]              # penultimate layer, not CLS-only
image_embeds: [B, 257, 1280]            # 1 CLS + 256 patch tokens
```

Injection in the transformer (prepended to text sequence before cross-attention):

```python
# transformer_wan.py:659
if encoder_hidden_states_image is not None:
    encoder_hidden_states = concat(
        [encoder_hidden_states_image, encoder_hidden_states], axis=1
    )
    # Cross-attention K/V: 257 image tokens + 512 text tokens = 769 total
```

### Dual-transformer routing (Wan 2.2)

```
for t in timesteps:
    if t / num_timesteps > boundary_ratio:   # early denoising, high noise
        pred = high_noise_transformer(...)
    else:                                     # late denoising, low noise
        pred = low_noise_transformer(...)
```

Training manages two `TrainState` objects with separate optimizer states. Both transformers receive the same `condition` tensor — the `in_channels=33` configuration applies to both.

---

## 7. Dataset Paths

Three paths are controlled by `config.dataset_type`:

```
dataset_type=synthetic   →  make_data_iterator() → random tensors (smoke test)
dataset_type=droid       →  DroidVideoDataset → raw frames → preprocess_batch() per step
dataset_type=tfrecord    →  pre-encoded TFRecords → direct field read
```

### Batch fields per path

| Field | synthetic | droid (after preprocess) | tfrecord |
|---|---|---|---|
| `latents` [B,16,T',H',W'] | ✓ random | ✓ VAE-encoded | ✓ stored |
| `encoder_hidden_states` [B,512,4096] | ✓ random | ✓ T5-encoded | ✓ stored |
| `condition` [B,17,T',H',W'] | ✓ random | ✓ built in `_encode_video_i2v` | ✓ stored |
| `encoder_hidden_states_image` [B,257,1280] | ✓ (wan2.1) | ✓ CLIP-encoded (wan2.1) | ✓ (wan2.1) |

### VAE lifecycle per path

| Path | VAE during training loop |
|---|---|
| `tfrecord` | Freed before training loop (`del pipeline.vae`) |
| `droid` | **Must stay loaded** — `_encode_video_i2v` runs every step |
| `synthetic` | Freed (no real data) |

---

## 8. DROID Sliding-Window Dataset

`input_pipeline/robot/droid_video_dataset.py`

### Pipeline

```
DROID TFDS (RLDS)
  ↓ filter: trajectory contains "success" in file path
  ↓ filter: trajectory length ≥ clip_length
  ↓ hash-based train/val split (2% val by default)
  ↓ _select_camera_and_instruction()   ← 1-of-2 exterior cameras, 1-of-3 instructions
  ↓ _traj_to_clips()                   ← sliding window → many clips per trajectory
  ↓ _decode_clip()                     ← JPEG decode + bilinear resize
  ↓ batch(batch_size, drop_remainder=True)
  ↓ prefetch(AUTOTUNE)
  → batch: {"frames": [B, T, H, W, 3], "language_instruction": [B]}
```

### Sliding window (clip extraction)

```python
# _traj_to_clips  (droid_video_dataset.py:235)
num_clips    = (traj_len - clip_length) // stride + 1
start_indices = range(num_clips) * stride

# stride=1   → maximum overlap, adjacent clips differ by 1 frame
# stride=clip_length → non-overlapping clips
```

Each clip's **first frame** becomes the conditioning image; the other frames are the generation target. There is no explicit cross-clip memory — the model is trained on independent clips. Long-horizon temporal continuity at inference time is achieved through chaining (using the last generated frame as `last_image` for the next call).

### VAE compatibility constraint

`clip_length` must satisfy `(clip_length - 1) % 4 == 0` for WAN VAE temporal compression:

```
clip_length=17 → T'=5   (recommended for fast iteration)
clip_length=49 → T'=13  (standard)
clip_length=81 → T'=21  (extended)
```

### DROID control frequency

DROID runs at 15 Hz, which is close to WAN's 16 fps. Sampling every frame gives approximately the right temporal density without additional subsampling.

---

## 9. Training Step

`i2v_train_step` / `_i2v_step_optimizer`  (wan_i2v_trainer.py:286)

```
Input batch:
  latents                    [B, 16, T', H', W']  — pre-encoded or VAE-encoded
  encoder_hidden_states      [B, 512, 4096]        — T5 text embeddings
  condition                  [B, 17, T', H', W']   — mask + first-frame latent
  encoder_hidden_states_image [B, 257, 1280]        — CLIP (wan2.1 only)

Step:
  1. sample_timesteps(rng, B)                   → timesteps [B]
  2. noise = jax.random.normal(latents.shape)
  3. apply_flow_match(noise, latents, timesteps) → noisy_latents, training_target, weight
  4. latent_model_input = concat([noisy_latents, condition], axis=1)  [B, 33, T', H', W']
  5. model(latent_model_input, timesteps, text_embeds, image_embeds)  → model_pred [B, 16, T', H', W']
  6. loss = mean((training_target - model_pred)² × training_weight)
  7. value_and_grad → grads
  8. state.apply_gradients(grads) → new TrainState
```

**Note:** `model_pred` has 16 channels (not 33) — the model predicts only the denoised video latent, not the conditioning channels. The conditioning channels are input-only.

Loss function is identical to T2V: Flow-Match MSE with `training_weight`.

---

## 10. Inference Pipelines

### WanPipelineI2V_2_1 — Single transformer

```python
pipeline(
    prompt,
    image,                  # PIL image or array: the first frame
    num_frames,
    last_image=None,        # optional: anchor the last frame too
    guidance_scale=5.0,
)
```

Condition preparation:

```
image [H, W, 3]
  → [B, C, 1, H, W]                   # add temporal dim
  → concat with zeros (or last_image)  # [B, C, T, H, W]
  → VAE encode → latent_cond           # [B, T', H', W', 16]  (channels-last in pipeline)
  → normalize with VAE stats
  → build mask (1.0 at t'=0 [and T'-1 if last_image])
  → condition = concat(mask, latent_cond, axis=-1)  # [B, T', H', W', 17]
```

Then at each denoising step:
```
latent_model_input = concat(noisy_latents, condition, axis=-1)  # [B, T', H', W', 33]
```

### WanPipelineI2V_2_2 — Dual transformer

Identical condition construction, but routes each denoising step to one of two transformers based on `boundary_ratio`:

```python
if current_timestep / total_timesteps > boundary_ratio:
    pred = high_noise_transformer(latent_model_input, ...)
else:
    pred = low_noise_transformer(latent_model_input, ...)
```

---

## 11. Checkpointing

### WanCheckpointerI2V_2_1

Saves/loads a single transformer state:
```python
items = {
    "transformer_state": train_states["transformer"],
}
```

### WanCheckpointerI2V_2_2

Saves/loads two transformer states:
```python
items = {
    "low_noise_transformer_state":  train_states["low_noise_transformer"],
    "high_noise_transformer_state": train_states["high_noise_transformer"],
}
```

Both extend `WanCheckpointer` and use Orbax `CheckpointManager` to write to GCS or local paths, identical to T2V.

---

## Appendix: File Map

| File | Purpose |
|---|---|
| `trainers/wan_i2v_trainer.py` | Trainer, data loading, `_encode_video_i2v`, train step |
| `trainers/base_wan_trainer.py` | Base class; `preprocess_batch` hook called every step |
| `pipelines/wan/wan_pipeline.py` | Base pipeline; `prepare_latents_i2v_base` (shared) |
| `pipelines/wan/wan_pipeline_i2v_2p1.py` | Wan 2.1 single-transformer inference |
| `pipelines/wan/wan_pipeline_i2v_2p2.py` | Wan 2.2 dual-transformer inference |
| `models/wan/transformers/transformer_wan.py` | Shared transformer; `in_channels=33`, `image_dim` for CLIP |
| `checkpointing/wan_checkpointer_i2v_2p1.py` | Single-transformer checkpoint save/load |
| `checkpointing/wan_checkpointer_i2v_2p2.py` | Dual-transformer checkpoint save/load |
| `configs/base_wan_i2v_14b.yml` | Wan 2.1 14B config |
| `configs/base_wan_i2v_27b.yml` | Wan 2.2 27B config |
| `input_pipeline/robot/droid_video_dataset.py` | DROID TFDS sliding-window loader |
