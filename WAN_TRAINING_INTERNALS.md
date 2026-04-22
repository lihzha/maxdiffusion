# WAN Training Internals

A reference document covering the architecture, data flow, and sharding strategy of `src/maxdiffusion/train_wan.py`. Written as a substitute for reading the full repo.

---

## Table of Contents

1. [Entry Point & Class Hierarchy](#1-entry-point--class-hierarchy)
2. [System Diagram](#2-system-diagram)
3. [Model Architecture: WanModel](#3-model-architecture-wanmodel)
4. [WanModel vs WanVACEModel](#4-wanmodel-vs-wanvacemodel)
5. [AutoencoderKLWan (VAE)](#5-autoencoderkLwan-vae)
6. [Scheduler](#6-scheduler)
7. [Device Mesh Creation](#7-device-mesh-creation)
8. [Sharding Strategy](#8-sharding-strategy)
9. [How FSDP Works Here](#9-how-fsdp-works-here)
10. [What with_sharding_constraint Actually Does](#10-what-with_sharding_constraint-actually-does)

---

## 1. Entry Point & Class Hierarchy

```
train_wan.py
│
├── WanTrainer                              trainers/wan_trainer.py
│   └── BaseWanTrainer (ABC)               trainers/base_wan_trainer.py
│       └── TrainState                     base_wan_trainer.py
│           └── flax.training.TrainState
│
├── WanCheckpointer2_1                     checkpointing/wan_checkpointer_2_1.py
│   └── WanCheckpointer (ABC)             checkpointing/wan_checkpointer.py
│
├── WanPipeline2_1                         pipelines/wan/wan_pipeline_2_1.py
│   └── WanPipeline (ABC)                 pipelines/wan/wan_pipeline.py
│
├── WanModel  ◄── THE TRANSFORMER         models/wan/transformers/transformer_wan.py
│   ├── WanRotaryPosEmbed
│   ├── WanTimeTextImageEmbedding
│   ├── WanTransformerBlock × N layers
│   │   ├── FP32LayerNorm                  models/normalization_flax.py
│   │   ├── FlaxWanAttention (self-attn)   models/attention_flax.py
│   │   ├── FP32LayerNorm
│   │   ├── FlaxWanAttention (cross-attn)  models/attention_flax.py
│   │   └── WanFeedForward
│   └── (output: FP32LayerNorm + Linear)
│
├── AutoencoderKLWan  ◄── THE VAE         models/wan/autoencoder_kl_wan.py
│   ├── WanEncoder3d
│   │   ├── WanCausalConv3d
│   │   ├── WanResidualBlock
│   │   ├── WanResample / WanAttentionBlock / WanMidBlock
│   ├── WanDecoder3d  (mirror of encoder)
│   └── AutoencoderKLWanCache
│
├── FlaxFlowMatchScheduler  ◄── TRAINING  schedulers/scheduling_flow_match_flax.py
│
└── FlaxUniPCMultistepScheduler  ◄── INFERENCE  schedulers/scheduling_unipc_multistep_flax.py
    └── UniPCMultistepSchedulerState

External (HuggingFace / PyTorch — frozen, not trained):
  UMT5EncoderModel   ← text encoder (google/umt5-xxl)
  AutoTokenizer      ← tokenizer
```

**Key fact:** `WanModel` is the **only trained component**. The VAE and text encoder are frozen.

Two schedulers are used for different purposes:
- `FlaxUniPCMultistepScheduler` — loaded at init, used for inference / video generation eval
- `FlaxFlowMatchScheduler` — created in `BaseWanTrainer.create_scheduler()`, used during the training loop for `apply_flow_match` and `sample_timesteps`

---

## 2. System Diagram

```
__main__ → app.run(main) → train(config) → WanTrainer.start_training()
│
├─ WanCheckpointer2_1.load_checkpoint()
│     ├─ Orbax checkpoint found → WanPipeline2_1.from_checkpoint()
│     └─ No checkpoint       → WanPipeline2_1.from_pretrained()
│          └─ loads: WanModel (transformer) + AutoencoderKLWan (VAE)
│                  + UMT5EncoderModel (text encoder, frozen)
│
├─ [opt] generate_sample()  ← pre-training SSIM baseline video
│
├─ del pipeline.vae / vae_cache   ← free HBM (VAE not needed for training)
│
├─ load_dataset()
│     ├─ TFRecords: pre-encoded {latents, encoder_hidden_states}
│     └─ Synthetic: generated on the fly via make_data_iterator()
│
├─ create_scheduler()  → FlaxFlowMatchScheduler (1000 timesteps)
│
├─ _create_optimizer() → AdamW + LR schedule
│
└─ training_loop()
      ├─ JAX mesh + state sharding setup
      ├─ TensorBoard writer (background thread)
      ├─ jax.jit compile: p_train_step, p_eval_step
      │
      └─ for step in range(max_train_steps):
            ├─ ThreadPoolExecutor: load_next_batch (async, overlaps with compute)
            │
            ├─ p_train_step(state, batch, rng, scheduler_state)
            │     ├─ sample_timesteps(rng, bsz)
            │     ├─ apply_flow_match(noise, latents, t) → noisy_latents, target, weight
            │     ├─ transformer forward pass (hidden_states, timestep, encoder_hidden_states)
            │     ├─ MSE loss × training_weight → scalar
            │     ├─ nnx.value_and_grad → grads + max_grad_norm
            │     └─ state.apply_gradients(grads) → new TrainState
            │
            ├─ record_scalar_metrics → TensorBoard (background thread)
            │
            ├─ [every eval_every steps] eval()
            │     ├─ deterministic forward pass (no grad)
            │     ├─ MSE loss bucketed by timestep
            │     └─ [opt] inference_generate_video() → decode latents via VAE → write to GCS
            │
            └─ [every checkpoint_every steps] checkpointer.save_checkpoint()
                  └─ Orbax CheckpointManager → GCS / local path
```

---

## 3. Model Architecture: WanModel

`models/wan/transformers/transformer_wan.py`

WanModel is a **3D video DiT** (Diffusion Transformer). It operates on video latents of shape `[B, C, T, H, W]`.

### Forward pass stages

```
Input: noisy_latents [B, 16, T, H/8, W/8]
       timestep      [B]
       encoder_hidden_states [B, 512, 4096]  ← T5 text embeddings

1. Patch embedding (Conv3D, stride=patch_size)
   → flatten spatial dims → [B, seq_len, inner_dim]

2. WanRotaryPosEmbed
   → rotary position embeddings for 3D (T, H, W)

3. WanTimeTextImageEmbedding
   → sinusoidal timestep embedding → MLP → 6×inner_dim (AdaLN params)
   → linear projection of T5 text embeddings

4. N × WanTransformerBlock
   ├─ AdaLN: shift/scale/gate from timestep embedding
   ├─ self-attention  (FlaxWanAttention, RoPE, RMS-norm on Q/K)
   ├─ cross-attention (FlaxWanAttention, text encoder as K/V)
   └─ FFN (GELU-approx, SwiGLU-style: gate × proj → output proj)

5. Output norm (FP32LayerNorm + AdaLN scale/shift)
   + Linear → [B, seq_len, out_channels × prod(patch_size)]

6. Unpatchify → [B, 16, T, H/8, W/8]   (same shape as input)
```

### Key design choices

- All LayerNorms are **FP32** regardless of `weights_dtype` (prevents instability)
- `condition_embedder` and `scale_shift_table` are kept in FP32 (see `cast_with_exclusion` in `wan_pipeline.py:56–75`)
- `scan_layers=True` folds all N blocks into a single `jax.lax.scan` for memory efficiency (removes N-way replication of activations in remat)
- Attention supports `dot_product`, `flash`, and `cudnn` kernels via `config.attention`

---

## 4. WanModel vs WanVACEModel

`models/wan/transformers/transformer_wan_vace.py`

`WanVACEModel` inherits from `WanModel` and adds a **parallel conditioning branch** (ControlNet-style) for video-to-video editing tasks.

### Structural additions

| Component | WanModel | WanVACEModel |
|---|---|---|
| Transformer blocks | N × `WanTransformerBlock` | N × `WanTransformerBlock` (unchanged) |
| Control blocks | — | M × `WanVACETransformerBlock` (one per `vace_layers` index) |
| Patch embeddings | 1 (`patch_embedding`, in=16) | 2 (+ `vace_patch_embedding`, in=`vace_in_channels`) |
| Extra input | — | `control_hidden_states` (conditioning video latents) |
| `scan_layers` | Supported | Not supported (raises `NotImplementedError`) |

### `WanVACETransformerBlock` internals

```
[opt] proj_in (Linear, only block 0)   ← fuses control + main hidden states
    ↓
norm1 + self-attn (on control_hidden_states only)
    ↓
norm2 + cross-attn (control queries → text encoder K/V)
    ↓
norm3 + FFN
    ↓
[opt] proj_out (Linear, always applied) ← produces conditioning_states
```

### Forward pass difference

```python
# Phase 1: run all VACE blocks, collect projected outputs
for i, vace_block in enumerate(vace_blocks):
    conditioning_states, control_hidden_states = vace_block(
        hidden_states, encoder_hidden_states, control_hidden_states, ...)
    control_hidden_states_list.append((conditioning_states, scale[i]))

control_hidden_states_list.reverse()

# Phase 2: run main DiT, inject at vace_layers indices
for i, block in enumerate(blocks):
    hidden_states = block(hidden_states, ...)
    if i in vace_layers:
        control_hint, scale = control_hidden_states_list.pop()
        hidden_states = hidden_states + control_hint * scale  # ← injection
```

### Training data difference

| Field | WanTrainer | WanVaceTrainer |
|---|---|---|
| `latents` | ✓ | ✓ |
| `encoder_hidden_states` | ✓ | ✓ |
| `conditioning_latents` | — | ✓ |

---

## 5. AutoencoderKLWan (VAE)

`models/wan/autoencoder_kl_wan.py`

A **3D causal VAE** with temporal convolutions (`WanCausalConv3d`): each frame only attends to past frames (causal in time).

```
Input video [B, C, T, H, W]
     ↓ WanEncoder3d
Latent [B, 16, T', H/8, W/8]   where T' = (T-1)/4 + 1
     ↓ WanDecoder3d
Reconstructed video [B, C, T, H, W]
```

Spatial downscale: **8×**. Temporal downscale: **4×**. Latent channels: **16**.

### Role during training

**The VAE is NOT used during the training loop.** Latents are pre-encoded offline (by `wan_txt2vid_data_preprocessing.py`) and stored in TFRecords. This is why the trainer immediately frees it:

```python
# base_wan_trainer.py:188-190
del pipeline.vae
del pipeline.vae_cache
```

It is only reloaded for:
1. **Pre/post SSIM video generation** (`config.enable_ssim=True`) — decode to pixel space for quality comparison
2. **Eval video generation** (`config.enable_generate_video_for_eval=True`) — decode denoised latents to a watchable video at eval checkpoints

In both cases: **decode only**, never encode during training.

---

## 6. Scheduler

Two schedulers, two purposes:

### FlaxUniPCMultistepScheduler (inference)
- Loaded from pretrained checkpoint in `WanPipeline.load_scheduler()`
- Used during `generate_wan.py` / eval video generation
- Multi-step ODE solver for fast inference

### FlaxFlowMatchScheduler (training)
- Created fresh in `BaseWanTrainer.create_scheduler()` with 1000 timesteps
- Replaces the pipeline scheduler for the training loop
- Two key methods called per train step:
  - `sample_timesteps(rng, bsz)` — sample random t ∈ [0, 1000] per sample
  - `apply_flow_match(noise, latents, t)` → `(noisy_latents, training_target, training_weight)`

---

## 7. Device Mesh Creation

`max_utils.py:352–405`, called from `wan_pipeline.py:612–613`

### Step 1 — Config declares parallelism degrees

```yaml
# bash_scripts/train_command.sh (example for 8 chips, single slice)
ici_data_parallelism=1
ici_fsdp_parallelism=2
ici_context_parallelism=4
ici_tensor_parallelism=1
# product must equal chips per slice: 1×2×4×1 = 8
```

**ICI** = Inter-Chip Interconnect (within one TPU pod slice, fast).  
**DCN** = Data Center Network (across pods, slower). Only relevant for multi-pod runs.

### Step 2 — Physical device array

```python
devices = jax.devices()                      # all visible TPU chips
num_slices = 1 + max(d.slice_index for d in devices)

if multi_slice:
    mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism, devices)
else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism, devices)
```

`mesh_utils` (from `jax.experimental.mesh_utils`) does **topology-aware placement** — reads the physical interconnect graph and permutes device order so the axis needing most bandwidth maps to the fastest physical links.

### Step 3 — Named mesh

```python
devices_array = max_utils.create_device_mesh(config)   # shape e.g. (1,2,4,1)
mesh = Mesh(devices_array, config.mesh_axes)
#                          ("data", "fsdp", "context", "tensor")
```

After this, every `PartitionSpec("data", "fsdp", None, "tensor")` refers to those named axes.

### Example: 8 chips, single slice

```
devices_array shape: (1, 2, 4, 1)
mesh axes:           (data, fsdp, context, tensor)

chip 0 → fsdp=0, context=0
chip 1 → fsdp=0, context=1
chip 2 → fsdp=0, context=2
chip 3 → fsdp=0, context=3
chip 4 → fsdp=1, context=0
chip 5 → fsdp=1, context=1
chip 6 → fsdp=1, context=2
chip 7 → fsdp=1, context=3
```

---

## 8. Sharding Strategy

The `logical_axis_rules` in `configs/base_wan_14b.yml:169–183` are the Rosetta Stone — they translate logical names to mesh axes:

```yaml
logical_axis_rules:
  ['batch',      ['data', 'fsdp']]          # batch dim split over data+fsdp
  ['embed',      ['context', 'fsdp']]       # weight row/col dims → fsdp+context
  ['mlp',        'tensor']                  # FFN inner dim → tensor-parallel
  ['heads',      'tensor']                  # attention heads → tensor-parallel
  ['norm',       'tensor']                  # norm scales → tensor
  ['activation_length',     'context']      # sequence length → context-parallel
  ['activation_heads',      'tensor']       # heads in activations → tensor
```

### 8a. Input Data

| File | Line | Code |
|---|---|---|
| `wan_trainer.py` | 35–38 | `NamedSharding(mesh, P(*config.data_sharding))` |
| `base_wan_14b.yml` | 197 | `data_sharding: [['data','fsdp','context','tensor']]` |
| `wan_trainer.py` | 117–122 | Passed as `in_shardings` to `jax.jit` |

All 4 mesh axes shard the batch dimension. JAX inserts collectives at the JIT boundary if the data arrives in the wrong layout.

### 8b. Model Parameters

Weight annotations at **declaration time** using `nnx.with_partitioning`:

| Layer | File:Line | Logical spec | Resolves to |
|---|---|---|---|
| Patch embedding | `transformer_wan.py:507` | `(None,None,None,None,"conv_out")` | out-channels → `context` |
| FFN gate proj | `transformer_wan.py:125` | `("embed","mlp")` | rows→`fsdp/context`, cols→`tensor` |
| FFN out proj | `transformer_wan.py:249` | `("mlp","embed")` | rows→`tensor`, cols→`fsdp/context` |
| Output linear | `transformer_wan.py:590` | `("embed", None)` | first dim→`fsdp/context` |
| Q/K/V kernels | `attention_flax.py:1194` | `("embed","heads")` | rows→`fsdp/context`, cols→`tensor` |
| O proj kernel | `attention_flax.py:1242` | `("heads","embed")` | rows→`tensor`, cols→`fsdp/context` |
| QK norm scale | `attention_flax.py:1262` | `("norm",)` | → `tensor` |

Resolved and placed onto devices at load time:
```python
# wan_pipeline.py:150–151
logical_state_spec     = nnx.get_partition_spec(state)
logical_state_sharding = nn.logical_to_mesh_sharding(
    logical_state_spec, mesh, config.logical_axis_rules)

# wan_pipeline.py:190
state[path].value = device_put_replicated(val, sharding)
```

### 8c. Optimization State (Adam m/v moments)

```python
# base_wan_trainer.py:271–273
state_spec      = nnx.get_partition_spec(state)
state           = jax.lax.with_sharding_constraint(state, state_spec)
state_shardings = nnx.get_named_sharding(state, mesh)
```

Adam `m` and `v` moments inherit the same `PartitionSpec` as their corresponding parameters. Each device owns both the parameter shard **and** its optimizer state shard — this is what makes it FSDP.

### 8d. Activations (hidden states)

| File:Line | Tensor | Constraint |
|---|---|---|
| `transformer_wan.py:617` | model entry `hidden_states` [B,C,T,H,W] | `("batch", None, None, None, None)` |
| `transformer_wan.py:399` | per-block residual `hidden_states` [B,seq,d] | `axis_names` from logical rules |
| `transformer_wan.py:402` | `encoder_hidden_states` [B,512,d] | same |
| `normalization_flax.py:46–87` | AdaLN shift/scale/gate (×6) | `("activation_batch","activation_embed")` |
| `transformer_wan_vace.py:202–209` | VACE `control_hidden_states` | `PartitionSpec("data","fsdp","tensor")` |

### 8e. Attention Q/K/V Tensors

| File:Line | When | Constraint |
|---|---|---|
| `attention_flax.py:715–717` | Before cuDNN kernel | `(BATCH, LENGTH, HEAD, D_KV)` |
| `attention_flax.py:1639–1641` | After Q/K/V projection (flash path) | `query_axis_names`, `key_axis_names`, `value_axis_names` |
| `attention_flax.py:1791–1793` | After Q/K/V projection (non-flash path) | same |
| `attention_flax.py:1798` | Attention output | `(BATCH, LENGTH, HEAD)` |
| `attention_flax.py:2077` | Final output | `hidden_state_axis_names` |
| `attention_flax.py:121–146` | Every head pack/unpack reshape | `(BATCH, LENGTH, HEAD)` |

---

## 9. How FSDP Works Here

**FSDP in JAX is not a separate library.** It emerges from a specific pattern of weight sharding annotations. No explicit all-gather or reduce-scatter calls are written — XLA inserts them automatically.

### The mechanism

1. Weights are annotated with `PartitionSpec` including the `fsdp` mesh axis → each device holds `1/fsdp_size` of each weight tensor.
2. When XLA compiles a matmul with a sharded weight, it sees the layout mismatch and inserts a collective. Two options:
   - **All-gather** the weight → full matmul → discard gathered copy
   - **Partial matmul** → **reduce-scatter** the output
3. XLA's cost model picks whichever minimizes communication volume.

### Per training step

```
Forward:
  all-gather weight across fsdp axis → full weight on each device
        ↓ matmul ↓
  discard gathered weight (memory freed)

Backward:
  all-gather weight again (needed for grad computation)
        ↓ compute ∂L/∂W ↓
  reduce-scatter gradient → each device holds 1/fsdp_size of grad
        ↓
  AdamW step on local shard only
```

### Why activation constraints are needed

Without constraints on activations, XLA might choose a layout for intermediate tensors that makes the weight all-gather "unnecessary" for that op but triggers a more expensive collective two layers later. The `with_sharding_constraint` calls act as checkpoints that lock in the expected layout at key points, steering XLA toward the intended FSDP communication pattern across all 40 layers.

---

## 10. What with_sharding_constraint Actually Does

Both `jax.lax.with_sharding_constraint` and `nn.with_logical_constraint` insert an **annotation into the XLA computation graph**, not a runtime data movement call.

```
nn.with_logical_constraint(tensor, logical_axes)
    │
    └─ translates logical names via logical_axis_rules
    └─ calls jax.lax.with_sharding_constraint(tensor, PartitionSpec(...))
           │
           └─ inserts a constraint node into the XLA HLO graph
```

**What XLA does with it:**

```
Tensor arrives in layout A
        ↓
XLA sees constraint: must be in layout B
        ↓
If A == B:   insert no-op (zero cost, common case)
If A != B:   insert the appropriate collective:
               Sharded→Replicated  →  all-gather
               Replicated→Sharded  →  reduce-scatter
               Sharded(X)→Sharded(Y) →  all-to-all
        ↓
Tensor leaves in layout B (guaranteed)
```

**The constraint applies to the value at that program point**, not to a "device". Every device participates — the `PartitionSpec` only describes which slice each device holds.

**Practical note:** In a well-designed sharding setup most constraints are no-ops because the natural output layout of each op already matches. Their value is in *preventing drift* — stopping XLA from propagating an unexpected layout through 40 layers before hitting an expensive mismatch.
