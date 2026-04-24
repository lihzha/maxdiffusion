# WAN Pipelines — Reference Notes

Reference material for the WAN 2.1 and 2.2 pipelines in `src/maxdiffusion/pipelines/wan/`.
Covers T2V vs I2V, internal data flow, and the differences between 2.1 and 2.2.

---

## 1. WAN 2.1 — T2V vs I2V at a glance

Pipelines:
- T2V: `wan_pipeline_2_1.py::WanPipeline2_1`
- I2V: `wan_pipeline_i2v_2p1.py::WanPipelineI2V_2_1`

### Conditioning modalities

| | T2V | I2V |
|---|---|---|
| Text prompt (T5) | yes | yes |
| First-frame image | no | yes (required) |
| Last-frame image | no | yes (optional; enables FLF2V) |
| CLIP image encoder | no | yes (`FlaxCLIPVisionModel`, `hidden_states[-2]`) |

I2V's `_create_common_components` is called with `i2v=True` (see `wan_pipeline.py:611‑683`), which additionally loads an `image_processor` and `image_encoder` (CLIP vision). T2V does not.

### Model architecture differences (same `WanModel` class, different config)

- **Patch-embedding input channels**
  - T2V: `in_channels = vae.z_dim` (16). Patch conv eats pure noisy latents.
  - I2V: `in_channels ≈ 36` — latents are concatenated on channels with a mask and a latent image condition:
    `z_dim (16) + vae_scale_factor_temporal (4 mask) + z_dim (16)`
    See `wan_pipeline_i2v_2p1.py:120‑132, 325`.

- **CLIP image embedder (`condition_embedder.image_embedder`)**
  `transformer_wan.py:141‑152, 515‑524, 658‑662`. Only built when `image_embed_dim` is set (I2V uses 1280). At forward time:
  ```python
  encoder_hidden_states = concat([image_embeds, text_embeds], axis=seq)
  ```
  T2V passes `encoder_hidden_states_image=None` and skips that branch.

So the two architectural deltas are: (a) wider input convolution, (b) an image-embedding MLP + a cross-attention sequence that includes CLIP image tokens.

### `__call__` inputs

**T2V** (`wan_pipeline_2_1.py:80‑100`) — text only, plus:
`prompt`, `negative_prompt`, `height`, `width`, `num_frames`, `guidance_scale`, optional `latents`/`prompt_embeds`/`negative_prompt_embeds`, `use_cfg_cache`, `use_magcache`/`magcache_thresh`/`magcache_K`/`retention_ratio`.

**I2V** (`wan_pipeline_i2v_2p1.py:135‑158`) — adds:
- `image` (required) — first-frame RGB image.
- `last_image` (optional) — FLF2V anchor.
- `image_embeds` — optional pre-computed CLIP embeds (seq ~257, dim 1280).
- `output_type`, `rng`.
- **Missing**: `use_cfg_cache` (CFG cache is wired only for T2V in 2.1).

Default differences: T2V `magcache_thresh=0.12`, I2V `magcache_thresh=0.04`.

### Latent preparation

**T2V** — `_prepare_model_inputs` samples gaussian noise shape `(B, F_lat, H_lat, W_lat, z_dim)` and returns it unchanged.

**I2V** — `prepare_latents` (`wan_pipeline_i2v_2p1.py:81‑133`) builds three things:
1. `latents` — gaussian noise, same shape as T2V.
2. `latent_condition` — VAE-encode of a video where frame 0 is the image (and frame −1 is `last_image` if given), rest zeros. Normalized via `(x‑mean)/std`.
3. `mask_lat_size` — binary temporal mask packed into 4 channels (see §5).

Final: `condition = concat([mask_lat_size, latent_condition], axis=-1)`. At each step:
```python
latent_model_input = concat([noisy_latents, condition], axis=channel)   # 16 + 20 = 36 channels
```

### Inference-loop differences

| Feature | T2V `run_inference_2_1` | I2V `run_inference_2_1_i2v` |
|---|---|---|
| Full CFG | yes | yes |
| CFG-cache (FFT) | yes (FasterCache style) | no |
| MagCache | yes | yes |
| Image cross-attention | no | yes (`encoder_hidden_states_image=image_embeds_combined`) |
| Per-step concat of `condition` | no | yes |
| `(B,F,H,W,C)↔(B,C,F,H,W)` transpose | no | yes |
| `mag_ratios_base` source | `config.mag_ratios_base` | split by height: `..._720p` / `..._480p` |

T2V uses three separately-jitted forward variants (`transformer_forward_pass`, `…_full_cfg`, `…_cfg_cache`) so XLA sees static shapes across full vs cached CFG. I2V only uses `transformer_forward_pass`.

### One-line mental model

> **I2V = T2V + (channel-concat of mask + VAE(image) into `latents`) + (sequence-concat of CLIP(image) into text context).**

---

## 2. Why the CLIP image embedder is needed alongside VAE

VAE and CLIP encode the same input image at **different levels of abstraction** and feed **different streams** of the transformer.

### Two streams, joined only via cross-attention

```
 Spatial stream                                  Context stream
 latents (16ch) ─┐                               text_embeds ──┐
                 │ channel concat                              │
 condition (20ch)┘                               CLIP(image) ──┤ seq concat
                 │                                             │
                 ▼                                             ▼
           patch_embedding (3D conv, 36→D)       text_embedder + image_embedder MLPs
                 │                                             │
                 ▼                                             ▼
          hidden_states [B, L_vid, D]            encoder_hidden_states [B, L_img+L_txt, D]
                 │                                             │
                 └─────── cross-attention (Q=video, KV=context) ──────┘
                                    × num_layers blocks
```

### VAE latent — pixel-faithful, spatially aligned

- Shape `(F_lat, H_lat, W_lat, 16)`.
- Patch `(h, w)` in the latent ↔ region `(8h, 8w)` in pixels.
- Enters via **channel-concat at the patch embedding**. Participates in **self-attention** as spatial tokens.
- Answers: "where is the dog's ear in pixel space, what color are those leaves, how does frame 1 continue from this texture."

### CLIP embedding — semantic, unlocalized

- Token sequence of ~257 tokens, dim 1280, trained contrastively on image-text pairs.
- Enters via **sequence-concat with text tokens**. Used in **cross-attention** only.
- Answers: "what kind of scene is this, what object is that, what style should propagate across all 81 frames."

Both are needed. Without CLIP the model can copy frame-0 pixels but struggles to generalize the *concept* across 80 future frames. Without the VAE latent the model has a concept description but no anchor for faithful pixel continuation — the first frame wouldn't match the input.

This pattern is broadly used: SD img2img uses VAE; image-variation diffusion uses CLIP; WAN I2V uses both.

### Where in the transformer each input lands

| Interaction | Where | Effect |
|---|---|---|
| `latents ↔ condition` | `patch_embedding` first conv | Channel-wise fusion per patch; conv learns "copy observed latent where mask=1, synthesize from noise where mask=0." |
| `hidden_states ↔ hidden_states` | `self_attn` per block | Anchored tokens (mask=1) leak their pixel signal across frames → temporal consistency. |
| `hidden_states ↔ encoder_hidden_states` | `cross_attn` per block | Video tokens pull semantic guidance from CLIP image tokens + T5 text tokens jointly. |

After the patch embed there is no separate "latents" or "condition"; the 3D conv fuses all 36 channels into one `hidden_states` tensor.

### Why re-present the VAE image latent each step (not just as init)?

1. **Anchoring against drift.** Without constant re-exposure, by step 30 the generated frame 0 can wander off the reference pixels.
2. **Inpainting framing.** I2V is formally an inpainting problem: observed + mask + unknown re-fed every step is standard.
3. **Free re-use.** `condition` is computed once outside the loop; only `latents` and `t` change per step.

---

## 3. `latents` vs `condition` vs `encoder_hidden_states_image`

Per-step walk-through in `run_inference_2_1_i2v`:

1. **Spatial stream** — `latent_model_input = concat([latents, condition], axis=-1)` (36 channels); transpose `(B,F,H,W,C)→(B,C,F,H,W)`.
2. **Patch embed** — 3D conv `(p_t, p_h, p_w, 36, inner_dim)` collapses 36 channels into one token per patch. From here on, `latents` and `condition` are indistinguishable.
3. **Context stream** — T5 embed (dim 4096) and CLIP embed (dim 1280) projected to `inner_dim`. Then:
   ```python
   encoder_hidden_states = concat([encoder_hidden_states_image, encoder_hidden_states], axis=seq)
   ```
4. **~40 blocks** of `self_attn → cross_attn → mlp`. Self-attention works on the video stream only; cross-attention lets video tokens query the combined image+text context.

Per-input role:
- `latents` — "blank canvas for this step."
- `condition` — "notes taped to the canvas: frame 0 must look like *this*; mask says what's fixed."
- `encoder_hidden_states_image` — "description board off to the side: scene category/style/mood."

### T2V simplification

`condition` and `encoder_hidden_states_image` are both absent in T2V:
- `in_channels=16`, no channel concat.
- `encoder_hidden_states_image=None`, the concat branch is skipped, cross-attention sees text only.
- No per-step concat or transpose.

Everything downstream (self/cross/FFN blocks, AdaLN, scheduler) is identical.

---

## 4. Why `mask_lat_size` has 4 channels (and not 1)

The mask is defined at **pixel-frame resolution** (e.g., 81 frames) but must be fed to the transformer at **latent-frame resolution** (21 frames). The Wan VAE temporally compresses 4 pixel frames → 1 latent frame (`vae_scale_factor_temporal=4`).

### Construction (`wan_pipeline_i2v_2p1.py:120‑131`)

```python
mask = ones((B,1,num_frames,H_lat,W_lat))       # pixel-frame resolution
mask[:,:,1:,:,:] = 0                            # [1, 0, 0, ..., 0]
first = repeat(mask[:,:,0:1], 4, axis=2)        # repeat first frame 4x
mask = concat([first, mask[:,:,1:]], axis=2)    # total = 4·num_latent_frames
mask = mask.reshape(B, 1, num_latent_frames, 4, H, W)
mask = transpose(..., (0,2,4,5,3,1)).squeeze(-1)
# final shape: (B, F_lat, H_lat, W_lat, 4)
```

The 4 is **not a learned feature axis** — it's the 4 pixel frames that got compressed into each latent frame.

### When 1 dim would suffice

Plain first-frame I2V only:
```
latent 0 : [1,1,1,1]
latent 1 : [0,0,0,0]
...
latent 20: [0,0,0,0]
```
All 4-tuples are uniform → a single scalar would do.

### Why 4 is actually needed — FLF2V

With `last_image` supplied:
```
latent 0 : [1,1,1,1]
latent 1..19 : [0,0,0,0]
latent 20: [0,0,0,1]   ← only the final pixel frame is anchored
```
The last latent frame compresses pixel frames (77,78,79,80), and only frame 80 is observed. A scalar mask would conflate `[0,0,0,1]` with `[1,0,0,0]`, `[0,1,0,0]`, etc. — four qualitatively different conditioning patterns.

### General principle

The 4-dim mask is a **sub-frame positional encoding of the observation pattern**. It preserves the finer (pixel-frame) mask when down-mapping to the coarser (latent-frame) grid, by packing the extra temporal axis into channels.

The patch embed's 3D conv `(p_t, p_h, p_w, 36, D)` can then learn per-sub-slot gating — "if mask channels read `(0,0,0,1)`, trust the last sub-slot of the observed VAE latent."

---

## 5. WAN 2.2 vs WAN 2.1

Pipelines:
- T2V: `wan_pipeline_2_2.py::WanPipeline2_2`
- I2V: `wan_pipeline_i2v_2p2.py::WanPipelineI2V_2_2`

### 5.1 Dual transformer (MoE-style by noise level)

Both 2.2 pipelines hold two transformer instances with the same architecture but different weights:

```python
self.high_noise_transformer   # from subfolder "transformer"
self.low_noise_transformer    # from subfolder "transformer_2"
```

Selection per step:
```python
boundary_timestep = self.boundary_ratio * num_train_timesteps
step_uses_high[s] = (t_s >= boundary_timestep)
```

- **High-noise transformer** — early, noisy steps. Low-frequency structure and layout.
- **Low-noise transformer** — later, cleaner steps. High-frequency detail and texture.

Two guidance scales (defaults 4.0 high / 3.0 low): CFG strength tuned per phase.
`use_cfg_cache` requires both > 1.0. `is_boundary` detection forces a recompute at the switch.

### 5.2 WAN 2.2 I2V drops CLIP

`_prepare_model_inputs_i2v` (`wan_pipeline.py:721‑739`):
```python
if self.config.model_name == "wan2.1":
    image_embeds = self.encode_image(...)   # CLIP
else:
    image_embeds = None                     # WAN 2.2 I2V
```

Every transformer call in 2.2 I2V passes `encoder_hidden_states_image=None`. The context stream becomes text-only (same shape as T2V).

Preserved: the 36-channel `[latents ‖ mask ‖ VAE(image)]` channel-concat. VAE-latent anchoring alone is kept.

Rationale: VAE-latent anchoring is spatially precise; CLIP overlaps with text for semantics. Dropping CLIP saves a ~300M ViT and simplifies training.

WAN 2.2 T2V is unchanged from 2.1 T2V in terms of transformer inputs.

### 5.3 New cache strategy: SenCache

Mutually exclusive with CFG-cache:
```python
score = α_x · Σ‖Δx‖  +  α_t · Σ|Δt|
if score ≤ ε and reuse_count < max_reuse:
    noise_pred = ref_noise_pred   # skip full transformer call
else:
    # full recompute + reset
```

Unlike CFG-cache (which still calls transformer on cond), SenCache can skip transformer entirely on cached steps. Safeguards: no-cache first 30% (structure) / last 10% (refinement); `max_reuse=3`; force recompute at transformer boundary.

### 5.4 CFG-cache schedule is boundary-aware in 2.2

- 2.1: midpoint split drives FFT phase-weights `(w1, w2)`.
- 2.2: split tied to transformer boundary; **high-noise steps never cached**.

```python
# 2.2
first_low_step = first step using low-noise transformer
t0_step = first_low_step        # high-freq boost for all low-noise cache steps
# high-noise steps: step_is_cache[s] = False  (always full CFG)
```

### 5.5 MagCache is gone in 2.2

Only `use_cfg_cache` xor `use_sen_cache` in 2.2. Likely replaced by SenCache or didn't compose cleanly with the transformer boundary.

### 5.6 Summary table

| | WAN 2.1 | WAN 2.2 |
|---|---|---|
| Transformer(s) | single `transformer` | `low_noise_transformer` + `high_noise_transformer` |
| Guidance scale | `guidance_scale` | `guidance_scale_low` + `guidance_scale_high` |
| I2V image conditioning | VAE latent + mask + **CLIP** | VAE latent + mask only |
| Cache strategies | MagCache + (CFG-cache T2V only) | CFG-cache (boundary-aware) + SenCache |
| CFG-cache available in I2V | no | yes |
| Non-cache step branch | single | `if step_uses_high[s]` (T2V) / `jax.lax.cond` (I2V) |

### 5.7 Revised data-flow diagrams

**WAN 2.2 T2V — per step**
```
                ┌─ t ≥ boundary ─ high_noise_transformer ─┐
latents ────────┤                                          ├─ noise_pred
                └─ t < boundary ─ low_noise_transformer  ─┘
text embeds ─ cross-attn (unchanged from 2.1)
```

**WAN 2.2 I2V — per step**
```
                              ┌─ t ≥ boundary ─ high_noise_transformer ─┐
[latents ‖ mask ‖ VAE(img)] ──┤                                          ├─ noise_pred
                              └─ t < boundary ─ low_noise_transformer  ─┘
text embeds ─ cross-attn (no CLIP)
```

---

## 6. Neither pipeline supports history-frame conditioning

The I2V pipelines only support:
- 1 anchored frame (first), **or**
- 2 anchored frames (first + last, FLF2V).

No API for feeding a multi-frame history. The hardcoded pattern in `prepare_latents_i2v_base` (`wan_pipeline.py:547‑590`):

```python
video_condition = concat([image, zeros(num_frames-1)], axis=time)
# or [image, zeros(num_frames-2), last_image]
```

And the hardcoded mask patterns (`[1,0,0,…,0]` or `[1,0,…,0,1]`).

### Why this is a hard limitation

1. **`prepare_latents_i2v_base`** only splices 1–2 RGB frames into the video-condition tensor.
2. **`mask_lat_size` construction** has exactly two hardcoded patterns.

The transformer's 4-channel mask *could* represent arbitrary observation patterns (that's what the sub-frame packing affords), but the model was only **trained** on first-frame and first+last conditioning, so even if the pipeline were patched, the denoiser would likely produce poor results on a held-out history-mask distribution.

### Workarounds

- **Sliding window**: run I2V with the last frame of the previous clip as `image`. Cheap but semantic drift accumulates.
- **Model swap**: use a model trained with random-prefix observation (MAGI-1, CausVid, StreamingT2V) — outside this repo.

---

## Reference file paths

- Base pipeline: `src/maxdiffusion/pipelines/wan/wan_pipeline.py`
- 2.1 T2V: `src/maxdiffusion/pipelines/wan/wan_pipeline_2_1.py`
- 2.1 I2V: `src/maxdiffusion/pipelines/wan/wan_pipeline_i2v_2p1.py`
- 2.2 T2V: `src/maxdiffusion/pipelines/wan/wan_pipeline_2_2.py`
- 2.2 I2V: `src/maxdiffusion/pipelines/wan/wan_pipeline_i2v_2p2.py`
- Transformer: `src/maxdiffusion/models/wan/transformers/transformer_wan.py`
