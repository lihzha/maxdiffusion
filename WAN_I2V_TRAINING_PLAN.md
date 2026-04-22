# WAN I2V Training Implementation

I2V **inference** is fully implemented in this repo. I2V **training** is now
implemented but not yet validated end-to-end. This document describes the
architecture, what was built, the two supported data paths, and what remains.

---

## Current State

| Component | T2V | I2V |
|---|---|---|
| Transformer (`WanModel`) | training + inference | inference only (model handles 33-ch input) |
| Pipeline (`WanPipeline*`) | 2.1 + 2.2 | 2.1 + 2.2 (inference only) |
| Checkpointer | 2.1 + 2.2 | 2.1 + 2.2 |
| Trainer | `WanTrainer` | `WanI2VTrainer` ✓ |
| Train step | `wan_trainer.py` | `wan_i2v_trainer.py` ✓ |
| Entry point routing | `train_wan.py` | `model_type: I2V` branch ✓ |
| Data path — TFRecord | `wan_txt2vid_data_preprocessing.py` | `wan_i2v_data_preprocessing.py` (not yet written) |
| Data path — DROID live | n/a | `DroidVideoDataset` + `preprocess_batch` ✓ |
| Config | `base_wan_14b.yml`, `base_wan_27b.yml` | `base_wan_i2v_14b.yml`, `base_wan_i2v_27b.yml` ✓ |
| End-to-end validation | ✓ | **not yet run** |

---

## How I2V Differs from T2V at the Transformer Level

**T2V transformer input:**
```
hidden_states:             [B, 16, T, H, W]   noisy latents
encoder_hidden_states:     [B, 512, 4096]     T5 text embeddings
```

**I2V transformer input:**
```
hidden_states:             [B, 33, T, H, W]   concat([noisy_latents, condition], axis=1)
  where condition:         [B, 17, T, H, W]   concat([mask, vae_latent_condition], axis=1)
    mask:                  [B,  1, T, H, W]   1.0 for frame 0, 0.0 elsewhere
    vae_latent_condition:  [B, 16, T, H, W]   VAE-encoded first frame, zeros for rest

encoder_hidden_states:     [B, 512, 4096]     T5 text embeddings  (same as T2V)
encoder_hidden_states_image: [B, 257, 1280]   CLIP image embeddings  (Wan 2.1 only)
                             None                                    (Wan 2.2)
```

`in_channels=33` is already set in the pretrained checkpoint's `config.json` — no
model code change. `encoder_hidden_states_image` already exists in `WanModel.__call__`
and `WanTimeTextImageEmbedding`.

**Loss is identical to T2V:** MSE between model prediction and flow-match target,
weighted by `training_weight`.

---

## The Main Footgun: Channel Axis Format

The inference pipeline works channels-last (`[B, T, H, W, C]`) and transposes
before the transformer. The training pipeline works channels-first (`[B, C, T, H, W]`).

- Inference: `condition = concat([mask, latent_condition], axis=-1)` → `[B, T, H, W, 17]`
- Training:  `condition = concat([mask, latent_condition], axis=1)`  → `[B, 17, T, H, W]`

Getting this wrong produces a shape that still compiles (channel count is still 33
after concatenation) but the values fed to the patch embedding are completely wrong.

**Validation check:** after any preprocessing or in `preprocess_batch`, assert
`condition[:, 0].min() == 1.0` and `condition[:, 0].max() == 1.0` — the mask
channel must be exactly 1.0 for all spatial positions of frame 0.

---

## Two Data Paths

Both paths produce the same encoded batch and use the same training step.

### Path A — DROID live (`dataset_type: "droid"`)

Load video clips directly from the existing DROID TFDS records. No separate
preprocessing step. VAE, T5, and CLIP encoders stay loaded throughout training.

```
DROID TFDS records
  → DroidVideoDataset  (tf.data pipeline, JPEG decode + resize)
  → MultiHostDataLoadIterator  →  {"frames": [B,T,H,W,3], "language_instruction": [B]}
  → WanI2VTrainer.preprocess_batch  (VAE + T5 + CLIP encoding, runs every step)
  → i2v_train_step
```

**Launch:**
```bash
python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_i2v_14b.yml \
  run_name=robot_i2v \
  dataset_type=droid \
  train_data_dir=/path/to/tfds_parent/ \
  height=480 width=832 num_frames=49 \
  droid_clip_stride=8
```

**Trade-offs vs Path B:**

| | Path A (droid) | Path B (tfrecord) |
|---|---|---|
| Preprocessing step | none | required upfront |
| Storage | uses existing TFDS records | 2× storage (new TFRecords) |
| Training throughput | lower (encode every step) | higher (read pre-encoded) |
| VAE / CLIP in memory | always | can be freed after load |

### Path B — Pre-encoded TFRecords (`dataset_type: "tfrecord"`)

Run `wan_i2v_data_preprocessing.py` (not yet written — see below) once to produce
TFRecords, then train with lower per-step cost.

```
DROID TFDS records
  → wan_i2v_data_preprocessing.py  (one-time, offline)
  → TFRecords: {latents, encoder_hidden_states, condition, encoder_hidden_states_image}
  → MultiHostDataLoadIterator
  → WanI2VTrainer (preprocess_batch is a no-op)
  → i2v_train_step
```

**Launch:**
```bash
python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_i2v_14b.yml \
  run_name=robot_i2v_tfrecord \
  dataset_type=tfrecord \
  train_data_dir=gs://your-bucket/wan-i2v-tfrecords/ \
  height=480 width=832 num_frames=49
```

---

## Implementation Details

### `DroidVideoDataset`

**File:** `src/maxdiffusion/input_pipeline/robot/droid_video_dataset.py`

Adapted from `DroidDataset` in `language-action-pretraining`, stripping all
action/state processing. Key design difference: replaces `flatten()` with
`flat_map`-based sliding-window extraction.

**Pipeline:**
1. `dl.DLataset.from_rlds(builder, ...)` — load RLDS trajectories via dlimp
2. `filter` — keep only `.*success.*` trajectories (configurable)
3. `filter` — drop trajectories shorter than `clip_length`
4. Hash-based train/val split on file path
5. `traj_map(_select_camera_and_instruction)` — per-trajectory: randomly pick one
   of two exterior cameras and one of three language instructions
6. `flat_map(_traj_to_clips)` — convert each trajectory into a dataset of clips
   using `tf.data.Dataset.from_tensor_slices(start_indices).map(extract_clip)`
7. `map(_decode_clip)` — `tf.io.decode_jpeg` + `tf.image.resize` → float32 [0, 1]
8. `shuffle` → `batch` → `prefetch`

**Output per batch:**
```
frames:               [B, clip_length, height, width, 3]  float32 [0, 1]
language_instruction: [B]                                  bytes
```

**Raw DROID TFDS feature keys used:**
```
observation/exterior_image_1_left   # JPEG bytes
observation/exterior_image_2_left   # JPEG bytes
language_instruction                # string (per-step, same value throughout episode)
language_instruction_2
language_instruction_3
traj_metadata/episode_metadata/file_path  # used for filtering and hashing
```

**`num_frames` must satisfy `(num_frames - 1) % 4 == 0`** (VAE temporal stride = 4):

| Purpose | `height` | `width` | `num_frames` | Duration |
|---|---|---|---|---|
| Smoke test | 256 | 448 | 17 | ~1 s |
| Standard | 480 | 832 | 49 | ~3 s |
| Full WAN | 720 | 1280 | 81 | ~5 s |

### `preprocess_batch` hook

**File:** `src/maxdiffusion/trainers/base_wan_trainer.py`

Added a `preprocess_batch(batch, pipeline)` method (default no-op) called in
`training_loop` between `load_next_batch` and `p_train_step`:

```python
example_batch = self.preprocess_batch(example_batch, pipeline)
state, scheduler_state, train_metric, rng = p_train_step(state, example_batch, ...)
```

Note: encoding runs on the critical path (not overlapped with the previous step's
compute). For throughput-sensitive runs, use Path B instead.

### `WanI2VTrainer`

**File:** `src/maxdiffusion/trainers/wan_i2v_trainer.py`

Extends `BaseWanTrainer` directly (not `WanTrainer`). Overrides:

- `_get_checkpointer` → `WanCheckpointerI2V_2_1` or `_2_2`
- `get_data_shardings` → adds `condition` and (for 2.1) `encoder_hidden_states_image`
- `load_dataset` → handles `"droid"`, `"tfrecord"`, and `"synthetic"` branches
- `get_train_step` / `get_eval_step` → JIT-compiled `i2v_train_step` / `i2v_eval_step`
- `preprocess_batch` → on-the-fly encoding for the `"droid"` path

**`preprocess_batch` encoding flow (DROID path):**
1. Decode text: `pipeline._get_t5_prompt_embeds(texts)` → `encoder_hidden_states [B, 512, 4096]`
2. Encode full video: `jit(vae_encode)(video * 2 - 1)` → `latents [B, 16, T', H', W']`
3. Encode first-frame-only video (zeros for remaining frames): → `latent_cond [B, 16, T', H', W']`
4. Build mask (1.0 at frame 0): `condition = concat([mask, latent_cond], axis=1)` → `[B, 17, T', H', W']`
5. CLIP encode first frame (Wan 2.1 only): `pipeline.image_encoder(first_frames).hidden_states[-2]` → `[B, 257, 1280]`

### TFRecord format (Path B)

Pre-encoded records must contain these serialised float32 tensors:

| Field | Shape | Notes |
|---|---|---|
| `latents` | `[16, T', H', W']` | normalised VAE output, channels-first |
| `encoder_hidden_states` | `[512, 4096]` | T5 embeddings |
| `condition` | `[17, T', H', W']` | `concat([mask, latent_cond])` — **channels-first** |
| `encoder_hidden_states_image` | `[257, 1280]` | CLIP embeddings (Wan 2.1 only) |

Where `T' = (T-1)//4 + 1`, `H' = H//8`, `W' = W//8`.

### Entry point routing

**File:** `src/maxdiffusion/train_wan.py`

```python
def train(config):
    if config.model_type == "I2V":
        from maxdiffusion.trainers.wan_i2v_trainer import WanI2VTrainer
        trainer = WanI2VTrainer(config)
    else:
        from maxdiffusion.trainers.wan_trainer import WanTrainer
        trainer = WanTrainer(config)
    trainer.start_training()
```

### Config additions (`base_wan_i2v_14b.yml`)

```yaml
# dataset_type options: 'tfrecord' | 'droid' | 'synthetic'
dataset_type: 'tfrecord'

# Stride between clip start indices (droid path only).
# stride=1: maximum overlap; stride=num_frames: non-overlapping clips.
droid_clip_stride: 1
```

---

## What Remains

### 1. `wan_i2v_data_preprocessing.py` (Path B only)

Produces pre-encoded TFRecords from DROID TFDS records. Reuse `vae_encode()` and
`text_encode()` from `wan_txt2vid_data_preprocessing.py`. Replace the HuggingFace
data loader with `DroidVideoDataset`. Add the condition and CLIP encoding steps.

### 2. End-to-end smoke test

```bash
# Synthetic data — verify train step compiles and gradients flow
python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_i2v_14b.yml \
  run_name=i2v_smoke \
  dataset_type=synthetic \
  height=256 width=448 num_frames=17 \
  max_train_steps=5

# DROID path — verify data loading and encoding
python src/maxdiffusion/train_wan.py \
  src/maxdiffusion/configs/base_wan_i2v_14b.yml \
  run_name=i2v_droid_smoke \
  dataset_type=droid \
  train_data_dir=/path/to/tfds_parent/ \
  height=256 width=448 num_frames=17 \
  max_train_steps=5
```

**Things to verify during smoke test:**
- `condition[:, 0].min() == 1.0` — mask channel is correct
- `condition[:, 1:].abs().sum(axis=(2,3,4))` is non-zero only at `t=0` — latent condition is correct
- Loss decreases over 5 steps (not NaN, not stuck at 0)
- Checkpoint saves and restores without error

### 3. `base_wan_i2v_27b.yml` training params

The 27B config exists for inference but needs the same training param additions
as the 14B config.

---

## File Map

| File | Status | Purpose |
|---|---|---|
| `src/maxdiffusion/train_wan.py` | modified | entry point routing |
| `src/maxdiffusion/trainers/wan_i2v_trainer.py` | **new** | trainer + train/eval steps |
| `src/maxdiffusion/trainers/base_wan_trainer.py` | modified | `preprocess_batch` hook |
| `src/maxdiffusion/input_pipeline/robot/droid_video_dataset.py` | **new** | DROID → video clips |
| `src/maxdiffusion/input_pipeline/input_pipeline_interface.py` | modified | `"droid"` branch |
| `src/maxdiffusion/configs/base_wan_i2v_14b.yml` | modified | training params added |
| `src/maxdiffusion/data_preprocessing/wan_i2v_data_preprocessing.py` | **not yet written** | Path B offline encoding |
