# Stable Video Diffusion Port Plan for maxdiffusion

**Target model:** Stable Video Diffusion base (`stabilityai/stable-video-diffusion-img2vid`) — 14 frames @ 576×1024, image-to-video only, checkpoint `svd.safetensors`.

**Reference:** `/home/irom-lab/projects/generative-models/SVD_JAX_PORT_NOTES.md` — authoritative notes on the PyTorch SGM source. Treat those facts as established (conditioning streams, UNet config, VideoDecoder structure, EDM sampler math, known gotchas).

**Scope:** Inference-only port. Training is out of scope for this pass.

---

## 1. Repo survey findings

### Module style

The SD 1.x/2.x/XL stack is **Flax Linen** with Huggingface Diffusers conventions: channels-last `(N,H,W,C)` internal layout with a transpose at UNet entry/exit (`unet_2d_condition_flax.py:469, 524`). `@flax_register_to_config` + `FlaxModelMixin` give each module a Diffusers-style `from_pretrained`/`from_config`. Newer models diverge: **WAN** uses **NNX** (`models/wan/autoencoder_kl_wan.py:70`, transformers under `models/wan/transformers/`), and **LTX-Video** runs a **PyTorch** graph under `torchax` (`models/ltx_video/autoencoders/causal_conv3d.py:23` is `torch.nn`). Flux uses a mix.

**Decision:** stay with **Flax Linen** and live under `models/svd/`, directly alongside `unet_2d_condition_flax.py`/`vae_flax.py`/`resnet_flax.py`/`attention_flax.py`/`embeddings_flax.py`. The blocks needed (`FlaxResnetBlock2D`, `FlaxBasicTransformerBlock`, `FlaxAttention`, `FlaxTransformer2DModel`, `FlaxTimesteps`, `FlaxTimestepEmbedding`, `FlaxEncoder`, `FlaxDiagonalGaussianDistribution`) all exist and are heavily reused. The NNX path would force duplication or a bridge.

### Directly reusable pieces

- **VAE encoder:** `models/vae_flax.py:483` `FlaxEncoder` is the exact SGM `Encoder` — reuse verbatim; SVD and SD share encoder weights.
- **Diagonal gaussian regularizer:** `models/vae_flax.py:725` `FlaxDiagonalGaussianDistribution`.
- **Spatial self/cross-attn blocks:** `attention_flax.py:1802` `FlaxBasicTransformerBlock`, with LayerNorm-norm1/2/3, FlaxGEGLU FF — exactly the SGM `BasicTransformerBlock` used by `SpatialVideoTransformer`.
- **Timestep embedding:** `embeddings_flax.py:157` `FlaxTimestepEmbedding` and `FlaxTimesteps`.
- **Resblock:** `resnet_flax.py:98` `FlaxResnetBlock2D` — two-conv, GroupNorm, SiLU, time-MLP. Matches SGM `ResBlock` with `up=False, down=False, use_conv=True` — the default path.
- **`added_cond_kwargs` plumbing for SDXL** (`unet_2d_condition_flax.py:253, 445-466`) — exact pattern for SVD's `vector` (ADM) stream.
- **PyTorch weight conversion:** `models/modeling_flax_pytorch_utils.py:89` `rename_key_and_reshape_tensor` already handles Conv2d (line 144) and Conv3d (line 150) transpositions, GroupNorm `weight→scale`, and linear `weight→kernel` transpositions. Entry point `convert_pytorch_state_dict_to_flax` (line 368).
- **Pipeline loading:** `checkpointing/base_stable_diffusion_checkpointer.py:189, 261` `load_diffusers_checkpoint` and `load_checkpoint` — `from_pt=True` converts on the fly; otherwise orbax.
- **CLIP image encoder:** `pipelines/wan/wan_pipeline.py:47, 286` already loads `FlaxCLIPVisionModel` from HF transformers + `CLIPImageProcessor`. For SVD we need `FlaxCLIPVisionModelWithProjection` on `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (1024-dim image embeds from the projection head).
- **Config system:** YAML under `src/maxdiffusion/configs/*.yml`, loaded via `pyconfig.initialize(argv)`. No `ml_collections`. `base14.yml` and `base_wan_i2v_14b.yml` are templates. `model_name` is validated at `pyconfig.py:42` against `_ALLOWED_MODEL_NAMES = {WAN2_1, WAN2_2, LTX2_VIDEO}` in `common_types.py:52-54` — need to add `SVD`.
- **Script layout:** `generate.py` / `generate_sdxl.py` / `generate_wan.py` — per-model top-level scripts. `generate_sdxl.py` is the closest reference.

### EDM sampler: absent

`schedulers/scheduling_euler_discrete_flax.py:51` is a vanilla beta-scheduled Euler; it supports `v_prediction` but uses linear/scaled-linear σ schedules, not Karras ρ=7 / σ_max≈700 EDM discretization. We must add `EDMDiscretization`, EDM pre/post scaling, and per-frame linear CFG.

### Attention kernels / sharding

`attention_flax.py` already wires SplashAttention (TPU), cuDNN flash (GPU), and tokamax variants behind `attention_kernel` config. Logical axis rules like `('activation_batch', ['data','fsdp'])` and `('conv_batch', ['data','context','fsdp'])` are already in the configs. For SVD, the effective batch for most of the UNet is `B·T` — existing rules carry through unchanged. The temporal transformer's "batch" is spatial `B·H·W` — same rule works in principle (see Risk §5).

### 3D conv infrastructure

No Flax-Linen 3D conv module exists (LTX's is torch; WAN's is NNX `nnx.Conv` via `WanCausalConv3d`). Need a small Linen 3D conv wrapper. `rename_key_and_reshape_tensor` already handles 5D conv transposition `(O,I,kT,kH,kW)→(kT,kH,kW,I,O)`.

### Fork vs. reuse decisions

| Component | Disposition |
|---|---|
| `FlaxEncoder`, `FlaxDiagonalGaussianDistribution`, `FlaxTimesteps`, `FlaxTimestepEmbedding`, `FlaxFeedForward`, `FlaxGEGLU`, `FlaxAttention`, `FlaxResnetBlock2D` | Reuse as-is |
| `FlaxBasicTransformerBlock` | Reuse verbatim for the spatial transformer inside `SpatialVideoTransformer` |
| `FlaxTransformer2DModel` | **Fork** → `FlaxSpatialVideoTransformer`: same 2D preamble (norm + proj_in) and postamble (proj_out), but block loop runs `[spatial, temporal, AlphaBlender]` per depth, threading `num_video_frames` and `image_only_indicator` |
| `FlaxUNet2DConditionModel` | **Fork** → `FlaxVideoUNet`: block loop passes `num_video_frames`/`image_only_indicator` through; Down/CrossAttnDown/Mid/Up/CrossAttnUp blocks accept these kwargs. Small `_apply_block(block, x, temb, context, T, ioi)` helper replaces SGM's implicit `TimestepEmbedSequential` |
| `FlaxDecoder` (in `vae_flax.py`) | **Fork** → `FlaxVideoDecoder`: inject temporal 3D convs between spatial layers |

---

## 2. Target file/directory layout

```
src/maxdiffusion/
├─ common_types.py                                     [edit: add SVD = "svd"]
├─ pyconfig.py                                         [edit: add SVD to _ALLOWED_MODEL_NAMES (inference only)]
├─ generate_svd.py                                     [new: CLI entry, derived from generate_sdxl.py + generate_wan.py]
├─ configs/
│  └─ base_svd.yml                                     [new: mirrors base_wan_i2v_14b.yml structure]
├─ models/
│  ├─ svd/
│  │  ├─ __init__.py                                   [new]
│  │  ├─ video_blocks_flax.py                          [new: FlaxAlphaBlender, FlaxConv3DTemporal, FlaxVideoTransformerBlock (temporal-only), FlaxTimeMixBlock]
│  │  ├─ video_attention_flax.py                       [new: FlaxSpatialVideoTransformer]
│  │  ├─ video_unet_blocks_flax.py                     [new: FlaxCrossAttnDownVideoBlock/..., forks of unet_2d_blocks_flax.py that thread T/ioi]
│  │  ├─ video_unet_flax.py                            [new: FlaxVideoUNet (FlaxModelMixin + ConfigMixin + flax_register_to_config)]
│  │  ├─ video_decoder_flax.py                         [new: FlaxVideoDecoder + FlaxVideoResnetBlock + FlaxAE3DConv + FlaxVideoAttnBlock]
│  │  ├─ video_autoencoder_flax.py                     [new: FlaxSVDAutoencoderKL]
│  │  ├─ edm_denoiser_flax.py                          [new: EDMDenoiser + VScalingWithEDMcNoise]
│  │  └─ svd_key_mapping.py                            [new: PyTorch→Flax key translator]
│  └─ embeddings_flax.py                               [edit: add FlaxConcatTimestepEmbedderND — 256-dim sinusoidal per micro-cond, concat to 768]
├─ schedulers/
│  └─ scheduling_edm_euler_flax.py                     [new: EDMDiscretization + EulerEDMSampler + LinearPredictionGuider]
├─ pipelines/
│  └─ svd/
│     ├─ __init__.py                                   [new]
│     ├─ pipeline_flax_svd.py                          [new: FlaxStableVideoDiffusionPipeline]
│     └─ pipeline_output.py                            [new]
└─ checkpointing/
   └─ svd_checkpointer.py                              [new: SVDCheckpointer]
```

Rationale:
- New modules live in `models/svd/` next to SD counterparts, matching the flat `models/wan/`, `models/ltx_video/`, `models/flux/` pattern.
- `schedulers/scheduling_edm_euler_flax.py` matches existing scheduler naming.
- `pipelines/svd/` matches `pipelines/wan/`, `pipelines/stable_diffusion/`.
- Checkpointer matches `wan_checkpointer*.py`.
- `configs/base_svd.yml` joins the top-level `configs/` folder.
- `generate_svd.py` sits at the top of `src/maxdiffusion/` next to `generate_wan.py`.

---

## 3. Porting phases

Reordered based on this repo's existing capabilities:
- **EDM sampler is absent** → pure-math, zero external dependencies → do it first; gives a harness to verify other components in isolation later.
- **VAE encoder is already present** → no work needed; we only port the VideoDecoder.
- **CLIP image encoder is already wired in WAN** → trivial integration phase rather than a port.

### Phase 0 — Plumbing (no math)

**Deliverable:** config + pyconfig + empty pipeline skeleton.

- `common_types.py`: add `SVD = "svd"`.
- `pyconfig.py`: add `SVD` to `_ALLOWED_MODEL_NAMES`.
- `configs/base_svd.yml`: copy `base_wan_i2v_14b.yml` and add SVD-specific fields:
  - `num_frames: 14`, `height: 576`, `width: 1024`
  - `fps_id: 6`, `motion_bucket_id: 127`, `cond_aug: 0.02`
  - `sigma_min: 0.002`, `sigma_max: 700.0`, `rho: 7.0`
  - `num_inference_steps: 25`
  - `min_guidance_scale: 1.0`, `max_guidance_scale: 2.5`
  - `image_url`
  - `pretrained_model_name_or_path: 'stabilityai/stable-video-diffusion-img2vid'`
  - Drop T5-related knobs. Keep mesh axes and flash kernel blocks. Drop `vae_logical_axis_rules`.
- `pipelines/svd/pipeline_flax_svd.py`: stub `FlaxStableVideoDiffusionPipeline(FlaxDiffusionPipeline)` with `__init__(vae, unet, image_encoder, feature_extractor, scheduler)` and a TODO `__call__`.
- `generate_svd.py`: stub that does `pyconfig.initialize` and constructs the pipeline from the stub.

**Verify:** stub pipeline constructs; `python -m maxdiffusion.generate_svd src/maxdiffusion/configs/base_svd.yml` imports without errors.

### Phase 1 — EDM sampler (pure math, no weights)

**Deliverable:** `schedulers/scheduling_edm_euler_flax.py` containing:

- `EDMDiscretization(sigma_min, sigma_max, rho, N)` → σ schedule, with `sigma_to_idx`, `append_sigma_zero`.
- `VScalingWithEDMcNoise` — pure function `σ → (c_skip, c_out, c_in, c_noise)` for v-prediction.
- `EulerEDMSampler` — one-step Euler update `x_{i+1} = x_i + (σ_{i+1} − σ_i) * d_i`, where `d_i = (x_i − D(x_i, σ_i)) / σ_i`.
- `LinearPredictionGuider(min_scale, max_scale, T)` — expands scalar CFG to per-frame scales `jnp.linspace(min_scale, max_scale, T)`, with `apply(x_cond, x_uncond)`; broadcasts across `(B*T, C, H, W)` by tiling the per-frame scale along T.

**Verify:** unit-test against PyTorch `sgm/modules/diffusionmodules/{sampling.py, discretizer.py, denoiser_scaling.py, guiders.py}` with seeded inputs. No weights required.

### Phase 2 — VideoDecoder + VAE round-trip

**Deliverable:** `models/svd/video_decoder_flax.py` and `models/svd/video_autoencoder_flax.py`.

- `FlaxConv3DTemporal(features, kernel_size=(3,1,1))` — Linen `nn.Conv` with explicit 3D kernel, SAME padding on time, computes on `(B, T, H, W, C)`. Since maxdiffusion is channels-last, spatial convs run on `(B*T, H, W, C)` then `rearrange → (B, T, H, W, C)` for the temporal conv.
- `FlaxAE3DConv(out_channels, video_kernel_size=[3,1,1])` — wraps spatial `nn.Conv(3,3)` then adds `time_mix_conv = FlaxConv3DTemporal(...)`. PT key maps: `conv.weight → spatial.kernel`, `time_mix_conv.weight → time_mix_conv.kernel`.
- `FlaxAlphaBlender(merge_strategy='learned_with_images', alpha_init=0.0)` — stores `mix_factor: nn.Param` (scalar); applies `alpha = sigmoid(mix_factor)` and returns `alpha*x_temporal + (1-alpha)*x_spatial`. When `image_only_indicator` is passed (shape `(B,T)`) use `jnp.where(image_only_indicator, 1.0, alpha)` to disable temporal mixing per-frame.
- `FlaxVideoResnetBlock` — spatial resblock + temporal time_stack resblock with `video_kernel_size=[3,1,1]` Conv3D, merged via `FlaxAlphaBlender`.
- `FlaxVideoAttnBlock` — per `time_mode: conv-only` in `svd.yaml`, a no-op wrapper around the existing spatial `FlaxAttentionBlock` (`vae_flax.py:211`). Leave a `time_mode='attn-and-conv'` path stubbed for SVDXT-VAE, but never exercised in base SVD.
- `FlaxVideoDecoder` — fork `FlaxDecoder` (`vae_flax.py:603`): replace `conv_in`/`conv_out` with `FlaxAE3DConv`, replace every `FlaxResnetBlock2D` inside `FlaxUNetMidBlock2D`/`FlaxUpDecoderBlock2D` with `FlaxVideoResnetBlock`, leave `FlaxAttentionBlock` unchanged. Thread `num_video_frames` through `__call__`.
- `FlaxSVDAutoencoderKL` — `@flax_register_to_config` module exposing `encode(x) → DiagonalGaussian` (reuses `FlaxEncoder`) and `decode(z, num_video_frames) → frames`. Config knob `scaling_factor=0.18215`.

**Verify:**
1. Structural init: `init_weights` succeeds; param shapes printed.
2. Weight-loading: convert `first_stage_model.*` keys from `svd.safetensors` using `convert_pytorch_state_dict_to_flax` + Phase 4's `svd_key_mapping.py`. Load and forward-decode a fixed random `z` of shape `(1, 14, 72, 128, 4)`. Compare to PyTorch pixel-wise; expect <1e-3 MSE in bf16.
3. Encode-decode round trip on a real video: project the 14 RGB frames to latent, tile noise-conditioned, decode, compare PSNR to PyTorch (>40 dB).

### Phase 3 — Conditioner: CLIP image + ADM vector

**Deliverable:** `pipelines/svd/pipeline_flax_svd.py` grows `encode_image` and `encode_micro_conds`.

- Copy WAN's CLIP loading pattern (`pipelines/wan/wan_pipeline.py:286`). Use `transformers.FlaxCLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")` + `CLIPImageProcessor.from_pretrained(...)`. The `.image_embeds` attribute is the 1024-dim projection, matching SGM's `FrozenOpenCLIPImageEmbedder`. Reshape to `(B, 1, 1024)` then tile to `(B*T, 1, 1024)` for the UNet `encoder_hidden_states`.
- `FlaxConcatTimestepEmbedderND` in `embeddings_flax.py`: Linen module using `FlaxTimesteps(outdim=256)` applied to each scalar in `(fps_id, motion_bucket_id, cond_aug)`; concat to 768. Yields SGM's `vector` ADM stream.
- `prepare_concat_stream`: encode the (noise-augmented) conditioning image through `FlaxEncoder` → `(B,1,H/8,W/8,4)`, tile along T → `(B,T,H/8,W/8,4)`, rearrange to `(B*T,H/8,W/8,4)`, channel-concat with the noised latent to get 8-channel UNet input.

**Verify:** produce the three streams and compare shapes/values to PyTorch on a fixed input image. Log: `crossattn (B*14,1,1024)`, `concat (B*14,72,128,4)` (channels-last), `vector (B,768)`.

### Phase 4 — VideoUNet (largest piece)

**Deliverable:** `models/svd/video_blocks_flax.py`, `models/svd/video_attention_flax.py`, `models/svd/video_unet_blocks_flax.py`, `models/svd/video_unet_flax.py`.

Bottom-up sub-order:

**4a. `FlaxVideoTransformerBlock`** (temporal-only): uses existing `FlaxAttention` + `FlaxFeedForward`. Input shape `(B*H*W, T, C)` — batch dim is `B*H*W`, sequence is T. Uses self-attn and (optionally) cross-attn with `spatial_context` spread to `(B*H*W, T, 1024)`. Per `svd.yaml`: `extra_ff_mix_layer: true` and `use_spatial_context: true`. Temporal block takes the same `crossattn` stream as spatial.

**4b. `FlaxSpatialVideoTransformer`** (fork of `FlaxTransformer2DModel`): same preamble/postamble, but each depth layer runs `spatial_block(x, context) → temporal_block(rearrange(x, 'b t h w c → (b h w) t c'), spatial_context) → AlphaBlender(spatial, temporal) → rearrange back`. The `transformer_blocks` list becomes `(spatial_block, temporal_block, alpha_blender)` triples. Keep flash-attention plumbing identical. Same `use_linear_projection=true` (per `svd.yaml`).

**4c. Video-aware UNet blocks** (`FlaxCrossAttnDownVideoBlock2D` / `FlaxCrossAttnUpVideoBlock2D` / `FlaxVideoMidBlock2DCrossAttn` / `FlaxDownVideoBlock2D` / `FlaxUpVideoBlock2D`): near-verbatim forks of `unet_2d_blocks_flax.py`. Changes: (a) use `FlaxSpatialVideoTransformer` in place of `FlaxTransformer2DModel`; (b) thread `num_video_frames` and `image_only_indicator` through `__call__`; (c) each `FlaxResnetBlock2D` becomes `FlaxVideoResBlockUNet` — spatial resblock + temporal resblock (time-mix conv) + `AlphaBlender`. Architecturally same shape as the decoder's temporal block but separate weights (different PT prefix).

**4d. `FlaxVideoUNet`** (fork of `FlaxUNet2DConditionModel`):
- `in_channels=8, out_channels=4`.
- `cross_attention_dim=1024`.
- `addition_embed_type='adm_vector'` — new path that takes a pre-computed 768-dim `vector` directly (SVD does the sincos outside in `ConcatTimestepEmbedderND`). Project via a new `add_embedding = FlaxTimestepEmbedding(time_embed_dim=1280)` to add to `t_emb`. Parallels SDXL's `text_time` branch (`unet_2d_condition_flax.py:253-259`).
- `block_out_channels=(320, 640, 1280, 1280)`, `layers_per_block=2`, `num_attention_heads=(5, 10, 20, 20)` (from `model_channels=320` / `num_head_channels=64`).
- Pre-transpose from `(B*T, 4+4, H, W) → (B*T, H, W, 8)` handled by existing entrance transpose. Track T as a python int on `FlaxVideoUNet.num_video_frames` or pass as kwarg.

**Verify (progressive):**
1. `init_weights(eval_only=True)` succeeds; print param tree.
2. Load `model.diffusion_model.*` from `svd.safetensors` via the key translator. All keys must resolve; missing keys fail loudly.
3. Activation-match one block at a time against PyTorch. Use identical seeded input `(x, t, context, image_only_indicator)` and compare after `conv_in`, after first downblock, after mid, after each upblock. <1e-3 in bf16.
4. Full UNet forward-pass match on a fixed `(x, t, cond dict)`.

### Phase 5 — End-to-end inference

**Deliverable:** `FlaxStableVideoDiffusionPipeline.__call__` and `generate_svd.py`.

Pipeline `__call__` steps:
1. Load conditioning image, CLIP-encode → crossattn `(B,1,1024)`.
2. VAE-encode conditioning image (with `cond_aug` noise) → `concat` latent; tile to T.
3. Build ADM vector `(fps_id, motion_bucket_id, cond_aug)` → 768-dim.
4. Sample noise `x_T ~ N(0, σ_max²)` of shape `(B, T, C, H/8, W/8)`; reshape to `(B*T, 4, H/8, W/8)`.
5. EDM Euler loop (`fori_loop`): at each σ, compute `x_in = c_in(σ) * x`, `t_in = c_noise(σ)`, channel-concat with `concat` stream → 8 channels; call UNet twice (cond + uncond) in a batched `(2*B*T, 8, H/8, W/8)` pass; apply `LinearPredictionGuider` with per-frame scales; undo v-scaling to get `D(x, σ)`; Euler step.
6. `x_0 = 1/0.18215 * x`; reshape to `(B, T, 4, H/8, W/8)`; `FlaxVideoDecoder` decode with `num_video_frames=T`.

`generate_svd.py` mirrors `generate_wan.py`: load checkpointer, init pipeline, jit the function, run, write frames/MP4.

**Verify:** full run against PyTorch reference with fixed seed, fps_id=6, motion_bucket_id=127, cond_aug=0.02, 25 Euler steps. Compare decoded frames pixel-wise; visually identical outputs, <2% MSE in bf16.

### Phase 6 — Checkpointing integration (optional polish)

**Deliverable:** `checkpointing/svd_checkpointer.py`.

- Derive from `BaseStableDiffusionCheckpointer`. Override `_get_pipeline_class` to return `FlaxStableVideoDiffusionPipeline`. Override `load_diffusers_checkpoint` to call `safetensors.flax.load_file("svd.safetensors")` then `svd_key_mapping.translate_svd_keys(pt_state_dict, flax_model)`.
- Optionally add an orbax-save path so re-runs skip the 9 GB safetensors load each time.

---

## 4. Checkpoint translation strategy

### `checkpoints/svd.safetensors` layout
- `model.diffusion_model.<...>` → VideoUNet (~1.5 B params)
- `first_stage_model.<...>` → VAE (~84 M encoder + ~300 M VideoDecoder)
- `conditioner.embedders.0.open_clip.model.<...>` → CLIP ViT-H/14 (frozen; use HF weights instead — don't translate)
- `conditioner.embedders.1.encoder.<...>` → VAE encoder duplicated (share with `first_stage_model.encoder`; don't translate twice)
- `conditioner.embedders.2.timestep.<...>` → sinusoidal embedder — parameter-free, nothing to translate.

### Key translator

`models/svd/svd_key_mapping.py` exposes `translate_svd_keys(pt_state_dict, flax_model) -> flax_params`. Wraps `convert_pytorch_state_dict_to_flax` (`modeling_flax_pytorch_utils.py:368`) with SVD-specific preprocessing:

1. **Prefix stripping.** Strip `model.diffusion_model.` for UNet, `first_stage_model.` for VAE; drop `conditioner.embedders.` (HF CLIP + on-the-fly sincos).

2. **UNet SGM→Diffusers name remapping.** SGM uses `input_blocks.{i}.{j}.`, `middle_block.{j}.`, `output_blocks.{i}.{j}.` with interleaved ResBlock/TransformerBlock/Downsample. Our Diffusers-style fork uses `down_blocks_{i}.resnets_{j}.`, `down_blocks_{i}.attentions_{j}.`, `down_blocks_{i}.downsamplers_0.`, etc. Deterministic mapping table needed. Options: (a) copy Diffusers' upstream `diffusers/loaders/single_file_utils.py` mapping; (b) write a small iterator over SGM's fixed `input_blocks`/`middle_block`/`output_blocks` indexing emitting Diffusers-style flat tuples. **Prefer (b)** — simpler, less brittle.

3. **Temporal module names (SVD-specific):**
   - `.time_stack.` (inside ResBlock) → `.temporal_resblock.` under the spatial resblock in `FlaxVideoResBlockUNet`.
   - `.time_mixer.mix_factor` → `.alpha_blender.mix_factor` (scalar param, preserve).
   - `.time_mix_blocks.{k}.` inside SpatialVideoTransformer → `.temporal_transformer_blocks_{k}.`.
   - `.time_pos_embed.` (sinusoidal per-frame pos) → `.time_pos_embedding.` Linen module.

4. **VAE VideoDecoder temporal names:**
   - `first_stage_model.decoder.up.{i}.block.{j}.time_stack.` → `decoder.up_blocks_{i}.resnets_{j}.temporal_resblock.`
   - `first_stage_model.decoder.up.{i}.block.{j}.time_mixer.mix_factor` → `decoder.up_blocks_{i}.resnets_{j}.alpha_blender.mix_factor`
   - `first_stage_model.decoder.conv_in.time_mix_conv.weight` → `decoder.conv_in.time_mix_conv.kernel`
   - `first_stage_model.decoder.conv_out.time_mix_conv.weight` → `decoder.conv_out.time_mix_conv.kernel`

5. **Layout transposes are handled by the existing generic path** (`modeling_flax_pytorch_utils.py:144-152`): Conv2d `(O,I,kH,kW)→(kH,kW,I,O)` and Conv3d `(O,I,kT,kH,kW)→(kT,kH,kW,I,O)`. Norm `weight→scale` and `bias→bias`. Linear `weight.T`. No manual work needed once key names line up.

6. **Validation:** After `translate_svd_keys`, run `validate_flax_state_dict(expected_pytree, new_pytree)` (`modeling_flax_pytorch_utils.py:33`) — logs missing and shape-mismatched keys. Successful port: zero missing keys, zero shape mismatches.

**Incremental approach:** first port the pure-SD parts (mapping table from SD weights, ignoring `time_*`). Then layer the 4 temporal name maps on top. Phase 2 (VAE decoder) exercises both at small scale (~300 M) before the 1.5 B UNet.

---

## 5. Risks and open questions

1. **NHWC vs NCHW in temporal rearranges.** maxdiffusion is internally NHWC (`unet_2d_condition_flax.py:469`), SGM is NCHW. Every `rearrange("(b t) c h w → b c t h w")` in PT becomes `rearrange("(b t) h w c → b t h w c")` in Flax. Affects every AlphaBlender, every time-mix conv, every temporal-transformer seq-dim rearrange. **High-risk for silent shape-correct-but-semantically-wrong bugs.** Mitigation: one helper `_bt_to_btxC(x, T)` used everywhere; unit-test against its PT counterpart before wiring to any block.

2. **`num_video_frames` must flow through jit.** In SDXL this was a pure function of `sample.shape[0]`. Here it's either (a) a constant baked into the jitted function (cleanest — base SVD is always 14 frames) or (b) a static dim from reshape. Recommend (a): `num_video_frames` is a `functools.partial`-bound int in `FlaxVideoUNet.__call__`, not a traced array. If we ever want variable T (XT=25), that's a retrace — acceptable, matches WAN.

3. **Sharding for the (B·T) batch layout.** Existing `activation_batch` / `conv_batch` rules FSDP-shard the flattened `B*T` batch along `fsdp`. For temporal blocks where the effective batch is `B·H·W`, no existing axis name covers that; naive rearrange sends the tensor through an unsharded op. Mitigation: inside `FlaxVideoTransformerBlock`, add `nn.with_logical_constraint(..., ('activation_batch', 'activation_length', 'embed'))` after rearrange. Validate on a 4-device mesh before declaring the port complete. At the deepest stage the UNet spatial resolution is tiny (H/64, W/64) — `B·H·W = 1*9*16 = 144`, smaller than some device counts; may need a gather rather than shard.

4. **`AlphaBlender` mix_factor init.** SGM inits `mix_factor=0.0` (sigmoid → 0.5 blending). Flax param init needs explicit `nn.initializers.constant(0.0)`. Trivial but easy to miss.

5. **`image_only_indicator` shape.** Base SVD sets `image_only_indicator = jnp.zeros((B, T))` (pure video mode). PT passes this as float with per-batch-and-frame granularity; AlphaBlender uses `where(image_only_indicator.bool(), 1.0, alpha)` per-element. Ensure JAX `.bool()` cast is handled (use `>0.5`, not `.astype(bool)` which dtype-promotes).

6. **CLIP ViT-H/14 weight compatibility.** We swap SGM's internal `open_clip` for HF's `FlaxCLIPVisionModelWithProjection`. HF's `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` has the same architecture but **different layer ordering in the projection head** vs OpenCLIP. Projection matrix weights should be bit-identical — not verified. Mitigation: Phase 3 forwards the same preprocessed image through both and compares 1024-dim output. If disagreement, port OpenCLIP's `visual.proj` on top of HF's vision tower.

7. **`num_heads` derivation.** SGM uses `num_heads = channels // num_head_channels` implicitly. Our `num_attention_heads: Tuple[int]` is explicit. UNet config has `num_head_channels: 64`; at `(320, 640, 1280, 1280)` that gives `(5, 10, 20, 20)`. Make sure `base_svd.yml` and `FlaxVideoUNet` config agree; off-by-one would be a silent slow-bake failure.

8. **`use_linear_projection: true`** per `svd.yaml` — supported in `FlaxTransformer2DModel` (`attention_flax.py:1991`), but PT key `proj_in.weight` is `(out, in)` (Linear), not `(out, in, 1, 1)` (Conv). `rename_key_and_reshape_tensor` routes 2D weights through the linear branch (transpose) when destination expects `kernel` — should work automatically. Verify on small tensor.

9. **Flash attention on the temporal axis — mostly a non-issue.** T=14 tokens is way below SplashAttention's `block_q=2048`, but the dispatcher in `attention_flax.py:750-761` (`_apply_attention`) already auto-falls-back to `dot_product` (pure XLA) whenever `seq_len < flash_min_seq_length`. Base WAN config sets `flash_min_seq_length: 4096` (`base_wan_i2v_14b.yml:64`). No manual routing needed. Action: just set `flash_min_seq_length` in `base_svd.yml`. Default 4096 works — it catches both the T=14 temporal path and the deepest spatial stages (H·W=144) and routes them to XLA; it keeps the shallow spatial stages (H·W up to 9216) on Splash where the perf actually matters. If we want Splash on the deeper spatial stages too, set `flash_min_seq_length: 128` — but this is a performance knob, not a correctness concern.

    Side note: WAN itself doesn't hit this because it uses **full 3D attention** over flattened (T·H·W) tokens (`transformer_wan.py:629` `jax.lax.collapse`). SVD's factored spatial+temporal design is a legacy of the SD 2.x UNet backbone; modern video DiTs (WAN, CogVideoX, OpenSora) went full-3D precisely to dodge this. Not a consideration for this port since we're matching base SVD weights, but worth noting if a future training-time architecture change is on the table.

10. **No training path in this port.** Requirements specify inference only per the reference notes. If training is needed later, the `StableDiffusionTrainer` pattern does not cleanly accommodate the 5D video tensor + EDM noise sampling — budget a separate phase.

11. **Memory footprint.** 14 frames × 576×1024 = 14 full SDXL-sized UNet forwards per step, batched. ~10 GB of activations per UNet call in bf16. Fits on v5p-8 but not on 24 GB consumer GPU. Remat mandatory: plumb `remat_policy` through `FlaxVideoUNet` (config field exists at `base_wan_i2v_14b.yml:235`).

---

## 6. Critical files for implementation

- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/models/unet_2d_condition_flax.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/models/unet_2d_blocks_flax.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/models/attention_flax.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/models/vae_flax.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/pipelines/wan/wan_pipeline.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/models/embeddings_flax.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/schedulers/scheduling_euler_discrete_flax.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/pyconfig.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/common_types.py`
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/configs/base_wan_i2v_14b.yml` (template)
- `/home/irom-lab/projects/maxdiffusion/src/maxdiffusion/generate_sdxl.py` (template)
- `/home/irom-lab/projects/generative-models/SVD_JAX_PORT_NOTES.md` (PT-side reference)
