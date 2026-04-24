# SVD port — debug post-mortem

Working copy of what was investigated, what was ruled in/out, and what is still
worth looking at. Pairs with `docs/svd_port_plan.md`.

Section 9 captures the 2026-04-24 pass focused on activation-level comparison
against the SGM reference in `../generative-models/`.

## 1. Symptoms that triggered this

1. **First frame of the generated video is not the conditioning image.** User
   reported frame 0 of the maxdiffusion output looked like a different dog —
   different pose, floppy ears, closed/squinty eyes — while the PyTorch
   `generative-models` reference at the same input produced frame 0 that
   closely matched the input.
2. **OOM at 512×512 in the JAX port** but not in the PyTorch reference; user
   had to downsize to get a run through at all.
3. Secondary quality concerns: the video overall looked worse than the PyTorch
   reference.

Tested primarily at 448×448, 14 frames, 25 EDM steps,
`motion_bucket_id=180`, seed 0, bf16, on `base_svd_gpu.yml`.

## 2. Confirmed bugs (fixed)

### 2.1 CLIP image preprocessing was wrong

- **Where:** `src/maxdiffusion/pipelines/svd/pipeline_flax_svd.py::encode_image_clip`.
- **Symptom:** non-square inputs (SVD is trained on 576×1024) had their sides
  cropped off before CLIP saw them, producing wrong cross-attention features.
- **Reference:** `diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py::_encode_image`
  stretch-resizes with antialiasing to 224×224 and bypasses the HF processor's
  resize/crop/rescale. The HF default is aspect-preserving resize + center-crop.
  The SVD `feature_extractor/preprocessor_config.json` has
  `{size: {shortest_edge: 224}, do_center_crop: true, do_resize: true, do_rescale: true}`,
  so the HF default on a 1024×576 input would crop the left/right sides out
  entirely before CLIP.
- **Fix:** mirror Diffusers exactly — `image * 2 - 1`, call Diffusers'
  `_resize_with_antialiasing` to 224×224, `(image + 1) / 2`, then feed the HF
  processor with `do_resize=False, do_center_crop=False, do_rescale=False`.

### 2.2 GroupNorm `eps` mismatch across the UNet and VAE

- **Where:** `src/maxdiffusion/models/resnet_flax.py::FlaxResnetBlock2D`,
  `src/maxdiffusion/models/svd/video_decoder_flax.py::FlaxTemporalResBlock3D`,
  `video_blocks_flax.py::FlaxVideoResBlockUNet`,
  `video_unet_blocks_flax.py::FlaxCrossAttnDownVideoBlock/...`,
  `video_attention_flax.py::FlaxSpatialVideoTransformer.norm`.
- **What Diffusers uses (verified by `grep -B1 "eps=1e-" .../unet_3d_blocks.py`):**

| Location | Diffusers | Maxdiffusion (pre-fix) |
|---|---|---|
| `CrossAttnDownBlockSpatioTemporal.resnets` spatial | 1e-6 | **1e-5** ❌ |
| `CrossAttnDownBlockSpatioTemporal.resnets` temporal | 1e-6 | **1e-6** ✓ |
| `DownBlockSpatioTemporal.resnets` spatial/temporal | 1e-5 | 1e-5 / **1e-6** ❌ |
| `UNetMidBlockSpatioTemporal.resnets` | 1e-5 | 1e-5 / **1e-6** ❌ |
| `TransformerSpatioTemporalModel.norm` (preamble) | 1e-6 | **1e-5** ❌ |
| VAE `SpatioTemporalResBlock` | spatial 1e-6, temporal 1e-5 | spatial 1e-6 ✓, temporal **1e-6** ❌ |

- **Fix:** added `norm_eps`/`resnet_eps`/`temporal_eps` parameters to the
  relevant Flax modules and plumbed per-block defaults matching Diffusers.
- **Impact before fix:** with `motion_bucket_id=180`, frame 0 had wrong pose
  (puppy lying down, eyes closed, wrong ears). After fix: correct pose,
  recognizable puppy; only high-motion smearing remains, which is by design.
  With `motion_bucket_id=127` frame 0 matches `image_0.png` essentially
  pixel-for-pixel (to the eye).

## 3. Ruled out (with evidence)

### 3.1 VAE encoder + video decoder (including temporal 3D convs)

- **Evidence:** added `_debug_save_vae_roundtrip` in the pipeline. Encodes the
  conditioning image with the 2D VAE encoder, tiles the 4-channel latent to 14
  frames, runs the full `FlaxVideoDecoder` (every `FlaxVideoResnetBlock`,
  `FlaxAlphaBlender`, `FlaxAE3DConv`-equivalent `conv_in`/`time_conv_out` path,
  and the mid-block spatial attention). Output frame 0 is visually identical
  to the input image at both tested resolutions (448×448 and 512×512).
- **Consequence:** VAE weight load, NHWC↔NCHW transposes inside the decoder,
  and temporal rearrangements inside the decoder are all correct.

### 3.2 UNet architecture / weight-loading correctness

- **Evidence:** activation-level diff in `/tmp/compare_unets_jax.py`. Ran one
  forward of Diffusers `UNetSpatioTemporalConditionModel` on CPU in fp32 and
  `FlaxVideoUNet` on GPU in bf16 with bit-identical inputs (seeded random
  sample, encoder hidden states, added_time_ids, timestep). At 32×32 latent
  × 14 frames:
  - cosine similarity: **0.999774**
  - abs diff mean: 0.00468 (Diffusers std ≈ 0.32 → ~1.5% of signal)
  - abs diff p99: 0.02450
  - abs diff max: 0.09632
  - per-frame diff means: 0.004–0.008, uniform across all 14 frames
- **Consequence:** the UNet forward is correct to within bf16↔fp32 precision
  noise. Uniformity across frames rules out reshape/permutation bugs in the
  temporal branch (those would spike a specific frame index).

### 3.3 EDM sigma schedule and `init_noise_sigma`

- **Evidence:** ran `FlaxEDMEulerScheduler.set_timesteps(25)` and Diffusers'
  `EulerDiscreteScheduler.from_pretrained(..., subfolder='scheduler')` +
  `set_timesteps(25)` and compared sigmas element-wise. Match to ~1e-6
  relative, including the trailing 0 and `init_noise_sigma` ≈ 700.00073
  (Diffusers applies `sqrt(sigma_max²+1)` under `timestep_spacing=leading`,
  maxdiffusion applies `sigma_max` directly; the difference is 0.0007 out of
  700, negligible).

### 3.4 ADM / micro-conditioning embedding

- **Evidence:** compared `svd_micro_cond_embed(fps=6, motion=127, aug=0.02)`
  against Diffusers' `Timesteps(256, True, 0)` on the same inputs. Max abs
  diff 7.6e-6 on a sum of 320.7. Matches.

### 3.5 CFG math equivalence

- Applying per-frame CFG to `pred_x0` (maxdiffusion) is mathematically
  identical to applying per-frame CFG to `v` (Diffusers) — both fold linearly
  through `c_skip * x + c_out * v` because `c_skip * x` is shared between
  cond/uncond. Verified algebraically.

### 3.6 "Frame 0 resolution ≠ image_0.png" observation

- Not a bug. `image_0.png` is 512×512; the user's config renders at 448×448
  because of the OOM-driven downsize (`base_svd_gpu.yml::width: 448`). The
  video is written at the config'd resolution. The `macro_block_size=16`
  default in `utils/export_utils.py::export_to_video` would only auto-scale
  if the resolution were not a multiple of 16 (448 is).

## 4. Not bugs — worth knowing

### 4.1 `motion_bucket_id` has a first-order effect on frame-0 fidelity

- `motion_bucket_id=180` in `base_svd_gpu.yml` (set to bias toward "puppy
  running" motion) produces noticeably softer/smearier frame 0 even with a
  correct UNet. This is how SVD is trained — higher bucket injects variance
  for larger inter-frame displacement, at the cost of anchoring to the input
  image.
- **Evidence:** rerunning identical seed/resolution with `motion_bucket_id=127`
  produces a frame 0 that matches the VAE roundtrip to the eye.

### 4.2 `from_pretrained` logs "unexpected config attrs"

- The Diffusers config has `{addition_time_embed_dim: 256, num_frames: 14,
  projection_class_embeddings_input_dim: 768, force_upcast: True}` which
  `FlaxVideoUNet` / `FlaxSVDAutoencoderKL` don't declare. These are bookkeeping
  in the Diffusers module, not weight-carrying. The strict validator
  (`svd_checkpointer.py::_strict_validate_state_dict`) confirms 1428 UNet keys
  and 374 VAE keys all load with matching shapes, which is what matters.

### 4.3 `fps_id = fps - 1` convention

- Diffusers subtracts 1 inside its pipeline (so `fps=7` → UNet sees 6).
  Maxdiffusion takes `fps_id` post-offset from the config (`base_svd_gpu.yml`
  has `fps_id: 6` with a comment calling out the -1). Correct for the default
  invocation; a caller who passes `fps_id` thinking it's raw FPS would end up
  off by one bucket. Not a correctness bug as-is; consider accepting `fps` and
  doing the `-1` internally for fewer footguns.

## 5. Known low-impact mismatches not yet addressed

### 5.1 `init_noise_sigma` is `sigma_max` vs `sqrt(sigma_max² + 1)`

- `src/maxdiffusion/schedulers/scheduling_edm_euler_flax.py::EDMEulerSchedulerState.init_noise_sigma = sigma_max`.
- Diffusers' SVD scheduler uses `sqrt(max(sigmas)² + 1)` because its
  `timestep_spacing` is `"leading"` (not in `["linspace", "trailing"]`).
- Numerical gap: 700.0007 vs 700.0 — ~1 ppm. Not visually consequential. Worth
  aligning for pedantic reproducibility.

### 5.2 Flax `FeedForward.net_2` (final linear) bias

- Diffusers' `FeedForward` ends with `nn.Linear(inner_dim, dim_out, bias=True)`
  by default. Maxdiffusion's `FlaxFeedForward.net_2 = nn.Dense(self.dim)` uses
  `nn.Dense`'s default `use_bias=True`. Matches.
- (Listed here only because this was repeatedly re-verified during the hunt —
  keep it in mind if anyone changes `nn.Dense` defaults.)

## 6. Not yet directly verified — worth checking if further divergence is found

### 6.1 Full-pipeline diff vs Diffusers at identical seed

- The UNet-only diff (§3.2) is a single forward pass. The 25-step sampling
  loop was only compared end-to-end *visually* and indirectly via the
  "motion_bucket=127 → matches input image" test. A step-by-step latent-space
  diff against Diffusers' CUDA pipeline at identical seed would be a stronger
  guarantee. Blocked locally by the venv's PyTorch being `2.10.0+cpu` — full
  pipeline on CPU is ~30–60 min per run.

### 6.2 `_resize_with_antialiasing` depends on a Diffusers internal

- The fix imports from
  `diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion`.
  That path is a private helper and could move across Diffusers versions. If
  Diffusers renames or removes it, CLIP preprocessing silently breaks again.
  A vendored copy with a unit test against the reference values would be safer.

### 6.3 `fori_loop` peak memory vs chunked eager loop — **fixed 2026-04-24**

- **Was:** `jax.lax.fori_loop` traced the 25-step denoising body once. XLA
  saw every iteration's intermediates as SSA values on the same graph and
  its buffer allocator could overlap their lifetimes across iterations,
  keeping near-worst-case live sets resident. PyTorch avoids this by
  Python-dispatching each step so refcounts drop between iterations. This
  was the best candidate for the 512×512 OOM that has no analog in the
  reference.
- **Now:** replaced with a Python `for` over a `@jax.jit`-compiled
  single-step function (`_sampler_step`) with `donate_argnums=(0,)` on `x`.
  One compiled program per step, so when the step call returns, XLA
  retires the whole step and only the new `x` survives to the next
  iteration. Donation lets the latent buffer be reused in place. Original
  `fori_loop` kept as a comment block in `pipeline_flax_svd.py` for
  reference.
- **Expected:** 512×512 should now fit on a 24 GB GPU. ~25 extra
  Python→XLA dispatches per generation, dominated by the UNet forward
  (<1% overhead). No numerical change.
- **Still OOM at 512×512?** The bottleneck is then inside a single step,
  not across steps. The prime suspect is the shallowest spatial attention:
  at 512×512 the latent is 64×64 → `seq_len=H·W=4096` with `T=14` frames
  and 5 heads, and `dot_product` materializes a
  `(B·T, 5, 4096, 4096)` bf16 scores tensor ≈ 2.3 GB per call (×2 for
  cond+uncond, × some transient multiplier during softmax/value). To dodge
  this, `attention_flax.py` already has a chunked-query path
  (`jax_memory_efficient_attention`) that keeps memory linear in seq_len,
  but it was not wired into the SVD stack. Plumbed 2026-04-24 via a new
  `use_memory_efficient_attention` flag threaded from
  `base_svd_gpu.yml` → `SVDCheckpointer` → `FlaxVideoUNet` →
  `FlaxCrossAttn{Down,Up}VideoBlock` / `FlaxVideoMidBlock2DCrossAttn` →
  `FlaxSpatialVideoTransformer` → `FlaxBasicTransformerBlock` / 
  `FlaxTemporalTransformerBlock` → `FlaxAttention`. Requires
  `split_head_dim=False` (the SVD port's default — the chunked path
  asserts 3D Q/K/V in `_apply_attention_dot`).
- **While wiring this up**, also found a pre-existing latent bug in
  `_apply_attention_dot`: the `use_memory_efficient_attention=True` branch
  returned `(B*heads, seq, dim_head)` instead of `(B, seq, heads*dim_head)`
  (it forgot to undo the heads→batch merge that
  `_reshape_heads_to_batch_dim` did at the top of the function). Caused a
  shape mismatch on the first residual add inside
  `FlaxBasicTransformerBlock`: `(B*heads, HW, inner) + (B, HW, inner)`.
  Branch was dead code — no caller had ever enabled it — so the bug was
  latent. Fixed by adding `_reshape_batch_dim_to_heads(..., heads)` after
  the transpose-back, mirroring the non-chunked path.

### 6.4 Flash attention path on GPU

- The config uses `attention: 'flash'` with `flash_min_seq_length: 4096`. At
  448×448, the max spatial seq len is H·W = 3136 < 4096, so the dispatcher
  falls back to `dot_product` for every stage. The flash path is never
  exercised in the runs that produced the reported symptoms — it's therefore
  not responsible for anything seen here, but also hasn't been validated
  numerically.

### 6.5 VAE `force_upcast` is ignored

- Diffusers upcasts the VAE to fp32 when the container dtype is fp16 and
  `force_upcast=True`. The HF config sets `force_upcast: true`; maxdiffusion
  ignores it and runs the VAE in bf16. bf16 has enough range so numerically
  fine, but if someone switches to fp16 weights (not currently wired up), the
  decoder could underflow.

### 6.6 `LinearPredictionGuider` per-frame scale direction

- `scales = linspace(min_scale, max_scale, T)` is applied in the order `[0..T-1]`
  along the `(B, T, ...)` axis after reshape. Double-check against Diffusers if
  a future change reorders frames — a reversed scale is visually subtle and
  would escape notice except as "frame 0 has more motion than expected."

## 7. How we arrived at each conclusion (artifacts that exist on disk)

- `/tmp/svd_debug/debug_vae_input.png`, `debug_vae_roundtrip_frame{00..13}.png`
  — from `_debug_save_vae_roundtrip`. Roundtrip = input ⇒ rules out VAE (§3.1).
- `/tmp/svd_debug/debug_final_latent_frame{00..13}.png` — from
  `_debug_save_pred_x0`. Decoded final-step latent; equals MP4 frame 0, so
  the last Euler step is not the culprit.
- `/tmp/svd_debug_mb127/debug_final_latent_frame00.png` — the same run with
  `motion_bucket_id=127`; sharp, matches input. Evidence for §4.1.
- `/tmp/diffusers_unet_{in_*,out}.npy` — Diffusers reference inputs/output for
  the UNet-diff harness (`/tmp/compare_unets.py`, `/tmp/compare_unets_jax.py`).
- `/tmp/compare_unets.log` — cosine 0.9998 confirmation (§3.2).

## 8. Reproduction

```bash
source .venv/bin/activate
JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
HF_HUB_ENABLE_HF_TRANSFER=1 \
  python -m maxdiffusion.generate_svd \
    src/maxdiffusion/configs/base_svd_gpu.yml \
    motion_bucket_id=127 \
    debug_dir=/tmp/svd_debug
```

Sanity check: `/tmp/svd_debug/debug_vae_roundtrip_frame00.png` should match
`image_0.png`; `/tmp/svd_debug/debug_final_latent_frame00.png` should also
match (modulo the ~1% bf16 smoothing). If either fails, §3.1 or §3.2 has
regressed.

## 9. 2026-04-24 SGM-anchored pass

Symptoms still reported after the §2 fixes: "first-frame blurriness, overall
quality worse than the SGM reference." Previous work anchored corrections to
Diffusers (`stabilityai/stable-video-diffusion-img2vid`), not SGM
(`Stability-AI/generative-models` → `svd.safetensors`). The goal of this pass
was to verify Diffusers and SGM actually agree and then check each stage
against SGM directly.

### 9.1 Diffusers ≈ SGM at UNet forward — cosine **0.99999958**

Script: `/tmp/compare_sgm_vs_diffusers.py` loads the raw `svd.safetensors`
checkpoint into SGM's `VideoUNet` and feeds it the exact inputs that the
existing `/tmp/diffusers_unet_in_*.npy` dump used. Result (fp32 GPU, same
seed):

| metric | value |
|---|---|
| cosine similarity | 0.99999958 |
| abs diff mean | 0.000158 |
| abs diff p99 | 0.000838 |
| abs diff max | 0.004742 |
| relative p99 / std | 0.26% |

**Implication:** Diffusers' `UNetSpatioTemporalConditionModel` and SGM's
`VideoUNet` produce numerically identical outputs from the same weights and
inputs. Combined with the previous §3.2 result (maxdiffusion ↔ Diffusers
cosine 0.9998 in bf16), this proves **the maxdiffusion UNet forward is
faithful to SGM**. No UNet-level bug remains. The bug must be in the
surrounding pipeline.

### 9.2 CLIP vision tower — HF = OpenCLIP exactly

Scripts: `/tmp/compare_clip_sgm.py` (OpenCLIP ViT-H/14, SGM venv) +
`/tmp/compare_clip_hf.py` (HF `CLIPVisionModelWithProjection`, maxdiffusion
venv). Fed both the same CLIP-normalized 224×224 input. Result: **cosine
1.00000000, max abs diff 5e-6, norm ratio 1.000000.** HF's projection head
is bit-identical to OpenCLIP's `visual.proj`. CLIP path is not the bug.

### 9.3 Diffusers' antialiased resize ≡ kornia's — also verified

`_resize_with_antialiasing(input, (224, 224))` (Diffusers helper, which the
port currently imports) produces **max abs diff 0.0** against
`kornia.geometry.resize(..., interpolation='bicubic', align_corners=True,
antialias=True)` on a 576×1024 image. SGM's `FrozenOpenCLIPImageEmbedder`
uses the kornia form, so the §2.1 preprocessing fix is consistent with SGM.

### 9.4 **The real bug — `adm_uncond` was zeroed**

SGM's `GeneralConditioner.get_unconditional_conditioning` only zeros the
embeddings named in `force_uc_zero_embeddings=["cond_frames",
"cond_frames_without_noise"]` (the VAE concat stream and CLIP features).
The **micro-cond scalars** (`fps_id`, `motion_bucket_id`, `cond_aug`) are
explicitly *not* in that list, so their 768-dim `vector` embedding is
**identical between cond and uncond**. Diffusers reproduces this exactly via
`pipeline_stable_video_diffusion.py::_get_add_time_ids →
torch.cat([add_time_ids, add_time_ids])`.

The port was doing the opposite:
```python
adm_uncond = jnp.zeros_like(adm_cond)   # WRONG
```
This fed a different time embedding into the uncond UNet call. Over 25 Euler
steps with linear per-frame CFG (`min_scale=1, max_scale=2.5`), the wrong
uncond propagates into `uncond + scale * (cond - uncond)` for every frame
except the first (where `scale=1` collapses the guider to `cond`). That
matches the observed symptom profile: frame 0 mostly fine, later frames
visibly degraded, overall worse than SGM.

**Fix** (already applied in `pipeline_flax_svd.py`):
```python
adm_uncond = adm_cond
```

Per the math: at frame 0 with `scale=1.0`, the guider returns `cond`, so this
does not move frame 0. It does correct frames 1–13, particularly at high
guidance where `uncond_wrong - uncond_right` is amplified by `(1 - scale)`.

### 9.5 GroupNorm eps — SGM differs from Diffusers (small residual effect)

SGM's `normalization()` returns `GroupNorm32(32, ch)` with the PyTorch
default `eps=1e-5`. It is used for **every** GroupNorm inside the UNet
(`openaimodel.py`, `video_model.py`). Diffusers by contrast hard-codes
mixed values: `CrossAttnDownBlockSpatioTemporal` and
`CrossAttnUpBlockSpatioTemporal` use `eps=1e-6` while `DownBlockSpatioTemporal`
/ `UNetMidBlockSpatioTemporal` / `conv_norm_out` use `eps=1e-5`. The only
preamble that's truly `1e-6` on both sides is
`TransformerSpatioTemporalModel.norm` in the `SpatialVideoTransformer` (SGM's
`sgm/modules/attention.py::Normalize` also uses `eps=1e-6` there).

The §2.2 fix aligned maxdiffusion to Diffusers. As a result:

| Block | current port | SGM ground truth |
|---|---|---|
| `FlaxCrossAttnDownVideoBlock.resnet_eps` | **1e-6** | 1e-5 |
| `FlaxCrossAttnUpVideoBlock.resnet_eps` | **1e-6** | 1e-5 |
| `FlaxDownVideoBlock.resnet_eps` | 1e-5 | 1e-5 ✓ |
| `FlaxUpVideoBlock.resnet_eps` | 1e-5 | 1e-5 ✓ (differs from Diffusers 1e-6) |
| `FlaxVideoMidBlock.resnet_eps` | 1e-5 | 1e-5 ✓ |
| `FlaxSpatialVideoTransformer.norm` | 1e-6 | 1e-6 ✓ |
| VAE spatial ResBlock / norm_out | 1e-6 | 1e-6 ✓ |
| VAE `time_stack` (temporal ResBlock) | 1e-5 | 1e-5 ✓ |

The CrossAttnDown/Up mismatch is numerically tiny (§9.1 already proved
Diffusers ↔ SGM cos 0.9999995 for a UNet forward with Diffusers' 1e-6 values).
Not a priority, but the principled SGM-matching values are `1e-5` in all five
UNet ResBlock families — revert if a future pass wants true parity.

### 9.6 What was *not* the bug (SGM-anchored)

- UNet forward (§9.1).
- CLIP vision projection (§9.2).
- `_resize_with_antialiasing` vs kornia (§9.3).
- ADM sinusoidal embedding (§3.4, still holds — max diff 7.6e-6 against
  `Timesteps(256, True, 0)` which matches SGM).
- EDM σ schedule (§3.3, still holds — matches to 1e-6 relative).
- VAE encode + video decode (§3.1, still holds — round-trip matches input).
- CFG math equivalence for `pred_x0` vs `v` (§3.5, still holds).
- `init_noise_sigma = sigma_max` vs `sqrt(sigma_max²+1)` (§5.1 — ~1 ppm).

### 9.7 Artifacts from this pass

- `/tmp/compare_sgm_vs_diffusers.py` + `/tmp/sgm_unet_out.npy` — UNet-level
  Diffusers ≡ SGM proof.
- `/tmp/compare_clip_sgm.py`, `/tmp/compare_clip_hf.py`,
  `/tmp/clip_ref_input.npy`, `/tmp/clip_ref_openclip_out.npy` — CLIP tower
  bit-parity proof.
- SGM venv: `/home/irom-lab/projects/generative-models/.pt2/bin/python`
  (torch 2.0.1+cu117, kornia 0.6.9, open_clip 3.3.0).
