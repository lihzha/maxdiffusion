# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from maxdiffusion.image_processor import PipelineImageInput
from maxdiffusion import max_logging
from .wan_pipeline import (
    WanPipeline,
    transformer_forward_pass,
    transformer_forward_pass_full_cfg,
    transformer_forward_pass_cfg_cache,
)
from ...models.wan.transformers.transformer_wan import WanModel
from typing import List, Union, Optional, Tuple
from ...pyconfig import HyperParameters
from functools import partial
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from ...schedulers.scheduling_unipc_multistep_flax import FlaxUniPCMultistepScheduler


class WanPipelineTI2V_2_2(WanPipeline):
  """Pipeline for WAN 2.2 Text+Image-to-Video (TI2V 5B, single transformer)."""

  def __init__(
      self,
      config: HyperParameters,
      transformer: Optional[WanModel],
      **kwargs,
  ):
    super().__init__(config=config, **kwargs)
    self.transformer = transformer

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint=None, vae_only=False, load_transformer=True):
    # i2v=True tells _create_common_components to skip CLIP for wan2.2
    common_components = cls._create_common_components(config, vae_only, i2v=True)
    transformer = None
    if not vae_only and load_transformer:
      transformer = super().load_transformer(
          devices_array=common_components["devices_array"],
          mesh=common_components["mesh"],
          rngs=common_components["rngs"],
          config=config,
          restored_checkpoint=restored_checkpoint,
          subfolder="transformer",
      )

    pipeline = cls(
        tokenizer=common_components["tokenizer"],
        text_encoder=common_components["text_encoder"],
        image_processor=common_components["image_processor"],
        image_encoder=common_components["image_encoder"],
        transformer=transformer,
        vae=common_components["vae"],
        vae_cache=common_components["vae_cache"],
        scheduler=common_components["scheduler"],
        scheduler_state=common_components["scheduler_state"],
        devices_array=common_components["devices_array"],
        mesh=common_components["mesh"],
        vae_mesh=common_components["vae_mesh"],
        vae_logical_axis_rules=common_components["vae_logical_axis_rules"],
        config=config,
    )
    return pipeline, transformer

  @classmethod
  def from_pretrained(cls, config: HyperParameters, vae_only=False, load_transformer=True):
    pipeline, transformer = cls._load_and_init(config, None, vae_only, load_transformer)
    pipeline.transformer = cls.quantize_transformer(config, transformer, pipeline, pipeline.mesh)
    return pipeline

  @classmethod
  def from_checkpoint(cls, config: HyperParameters, restored_checkpoint=None, vae_only=False, load_transformer=True):
    pipeline, _ = cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)
    return pipeline

  def _get_num_channel_latents(self) -> int:
    return self.transformer.config.in_channels

  def prepare_latents(
      self,
      image: Optional[jax.Array],
      batch_size: int,
      height: int,
      width: int,
      num_frames: int,
      dtype: jnp.dtype,
      rng: jax.Array,
      latents: Optional[jax.Array] = None,
      last_image: Optional[jax.Array] = None,
      num_videos_per_prompt: int = 1,
      oracle_latents: Optional[jax.Array] = None,
      n_priv: int = 0,
  ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
    num_channels_latents = self.vae.z_dim
    # num_frames = frames to generate (excludes the anchor frame).
    # Total output latent frames = anchor(1) + gen(num_frames // temporal_scale).
    num_latent_gen_frames = num_frames // self.vae_scale_factor_temporal
    num_latent_frames = 1 + num_latent_gen_frames
    latent_height = height // self.vae_scale_factor_spatial
    latent_width = width // self.vae_scale_factor_spatial

    shape = (batch_size, num_latent_frames, latent_height, latent_width, num_channels_latents)

    if latents is None:
        latents = jax.random.normal(rng, shape=shape, dtype=jnp.float32)
    else:
      latents = latents.astype(dtype)

    if oracle_latents is not None and n_priv > 0:
      frame_0 = oracle_latents[:, 0:1, :, :, :]
      oracle_future = oracle_latents[:, 1:1 + n_priv, :, :, :]       # (B, n_priv, H, W, C)
      noisy_gen = latents[:, 1:, :, :, :]                             # (B, num_latent_gen_frames, H, W, C)
      latents = jnp.concatenate([frame_0, oracle_future, noisy_gen], axis=1)
      clean_latent = jnp.concatenate([frame_0, oracle_future], axis=1)  # (B, 1+n_priv, H, W, C)
    elif oracle_latents is not None:
      # n_priv == 0 but pre-encoded latents provided: use oracle frame 0 as anchor.
      frame_0 = oracle_latents[:, 0:1, :, :, :]
      latents = latents.at[:, 0:1, :, :, :].set(frame_0)
      clean_latent = frame_0                                           # (B, 1, H, W, C)
    else:
      total_pixel_frames = 1 + num_frames
      if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
      image = jnp.array(image)
      if last_image is not None:
        if hasattr(last_image, "detach"):
          last_image = last_image.detach().cpu().numpy()
        last_image = jnp.array(last_image)

      latent_condition, _ = self.prepare_latents_i2v_base(image, total_pixel_frames, dtype, last_image)
      latents = latents.at[:, 0:1, :, :, :].set(latent_condition[:, 0:1, :, :, :])
      if last_image is not None:
        latents = latents.at[:, -1:, :, :, :].set(latent_condition[:, -1:, :, :, :])

      if last_image is not None:
        clean_latent = jnp.concatenate([latent_condition[:, 0:1], latent_condition[:, -1:]], axis=1)
      else:
        clean_latent = latent_condition[:, 0:1]  # (B, 1, H, W, C)

    return latents, clean_latent, None

  def __call__(
      self,
      prompt: Union[str, List[str]],
      image: Optional[PipelineImageInput] = None,
      negative_prompt: Optional[Union[str, List[str]]] = None,
      height: Optional[int] = None,
      width: Optional[int] = None,
      num_frames: Optional[int] = None,
      num_inference_steps: int = 50,
      guidance_scale: float = 5.0,
      num_videos_per_prompt: int = 1,
      max_sequence_length: int = 512,
      latents: Optional[jax.Array] = None,
      prompt_embeds: Optional[jax.Array] = None,
      negative_prompt_embeds: Optional[jax.Array] = None,
      last_image: Optional[PipelineImageInput] = None,
      output_type: Optional[str] = "np",
      rng: Optional[jax.Array] = None,
      use_cfg_cache: bool = False,
      use_sen_cache: bool = False,
      conditioning_video: Optional[jax.Array] = None,
      privileged: bool = False,
      preencoded_oracle_latents: Optional[jax.Array] = None,
  ):
    if use_cfg_cache and use_sen_cache:
      raise ValueError("use_cfg_cache and use_sen_cache are mutually exclusive. Enable only one.")
    if conditioning_video is None and image is None and preencoded_oracle_latents is None:
      raise ValueError("Provide either 'image', 'conditioning_video', or 'preencoded_oracle_latents'.")

    height = height or self.config.height
    width = width or self.config.width
    num_frames = num_frames or self.config.num_frames

    # num_frames = frames to generate; must be a multiple of the temporal scale factor.
    if num_frames % self.vae_scale_factor_temporal != 0:
      max_logging.log(
          f"`num_frames` has to be divisible by {self.vae_scale_factor_temporal}. "
          f"Rounding {num_frames} down to the nearest valid number."
      )
      num_frames = (num_frames // self.vae_scale_factor_temporal) * self.vae_scale_factor_temporal
      max_logging.log(f"Adjusted num_frames to: {num_frames}")
    num_frames = max(num_frames, self.vae_scale_factor_temporal)

    # image_embeds is always None for WAN 2.2 (VAE latent conditioning, no CLIP)
    prompt_embeds, negative_prompt_embeds, _, effective_batch_size = self._prepare_model_inputs_i2v(
        prompt,
        image,
        negative_prompt,
        num_videos_per_prompt,
        max_sequence_length,
        prompt_embeds,
        negative_prompt_embeds,
        None,
        last_image,
    )

    # Encode conditioning_video when provided. oracle_latents: [B, T', H', W', C_z].
    oracle_latents = None
    n_priv = 0
    if preencoded_oracle_latents is not None:
      oracle_latents = preencoded_oracle_latents.astype(prompt_embeds.dtype)
      privileged = True
    elif conditioning_video is not None:
      oracle_latents = self._encode_video_to_i2v_latents(
          conditioning_video, prompt_embeds.dtype, total_pixel_frames=1 + num_frames
      )
    if oracle_latents is not None and privileged:
      num_privileged_frames = getattr(self.config, "num_privileged_frames", -1)
      n_available = oracle_latents.shape[1] - 1  # T' - 1 (all frames except frame 0)
      n_priv = n_available if num_privileged_frames < 0 else min(num_privileged_frames, n_available)
      # Clip oracle frames to the generation length so the conditioning video
      # length never exceeds what we are actually generating.
      num_latent_gen_frames = num_frames // self.vae_scale_factor_temporal
      n_priv = min(n_priv, num_latent_gen_frames)

    def _process_image_input(img_input, height, width, batch_size):
      if img_input is None:
        return None
      tensor = self.video_processor.preprocess(img_input, height=height, width=width)
      jax_array = jnp.array(tensor.cpu().numpy())
      if jax_array.ndim == 3:
        jax_array = jax_array[None, ...]
      if batch_size > 1:
        jax_array = jnp.repeat(jax_array, batch_size, axis=0)
      return jax_array

    image_tensor = _process_image_input(image, height, width, effective_batch_size)
    last_image_tensor = _process_image_input(last_image, height, width, effective_batch_size)

    if rng is None:
      rng = jax.random.key(self.config.seed)
    latents_rng, _ = jax.random.split(rng)

    latents, clean_latent, _ = self.prepare_latents(
        image=image_tensor,
        batch_size=effective_batch_size,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=prompt_embeds.dtype,
        rng=latents_rng,
        latents=latents,
        last_image=last_image_tensor,
        oracle_latents=oracle_latents,
        n_priv=n_priv,
    )

    scheduler_state = self.scheduler.set_timesteps(
        self.scheduler_state, num_inference_steps=num_inference_steps, shape=latents.shape
    )

    graphdef, state, rest = nnx.split(self.transformer, nnx.Param, ...)
    data_sharding = NamedSharding(self.mesh, P())
    if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
      data_sharding = jax.sharding.NamedSharding(self.mesh, P(*self.config.data_sharding))

    latents = jax.device_put(latents, data_sharding)
    clean_latent = jax.device_put(clean_latent, data_sharding)
    prompt_embeds = jax.device_put(prompt_embeds, data_sharding)
    negative_prompt_embeds = jax.device_put(negative_prompt_embeds, data_sharding)

    frame_positions = None
    if n_priv > 0:
      # Layout: [frame_0, oracle_1..n_priv, noisy_gen_1..num_latent_gen_frames].
      # Oracle frames share temporal positions 1..n_priv with the first n_priv noisy
      # gen frames.  Noisy gen frames beyond n_priv get positions n_priv+1..num_latent_gen_frames
      # with no oracle counterpart (generated from motion prior alone).
      num_latent_gen_frames = num_frames // self.vae_scale_factor_temporal
      frame_positions = tuple(
          [0] + list(range(1, n_priv + 1)) + list(range(1, num_latent_gen_frames + 1))
      )

    p_run_inference = partial(
        run_inference_ti2v_2_2,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        scheduler=self.scheduler,
        use_cfg_cache=use_cfg_cache,
        use_sen_cache=use_sen_cache,
        height=height,
        has_last_image=last_image_tensor is not None and oracle_latents is None,
        n_priv=n_priv,
        frame_positions=frame_positions,
        oracle_noise_offset=None if getattr(self.config, "oracle_noise_offset", -1) < 0 else getattr(self.config, "oracle_noise_offset", -1),
    )

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      latents = p_run_inference(
          graphdef=graphdef,
          state=state,
          rest=rest,
          latents=latents,
          clean_latent=clean_latent,
          prompt_embeds=prompt_embeds,
          negative_prompt_embeds=negative_prompt_embeds,
          scheduler_state=scheduler_state,
      )
      if oracle_latents is not None:
        gen_start = 1 + n_priv  # index of first gen frame in unstripped latents
        # Gen frames share temporal positions 1..n_total_latent with oracle frames.
        # GT is at oracle_latents[:, 1:] regardless of n_priv.
        n_gen = min(latents.shape[1] - gen_start, oracle_latents.shape[1] - 1)
        if n_gen > 0:
          diff = latents[:, gen_start:gen_start + n_gen] - oracle_latents[:, 1:1 + n_gen]
          mse = float(jnp.mean(diff ** 2))
          mae = float(jnp.mean(jnp.abs(diff)))
          print(f"[conditioning_video] diffused vs GT latent (gen frames [{gen_start}:{gen_start+n_gen}]) — MSE: {mse:.6f}, MAE: {mae:.6f}")
      if n_priv > 0:
        # Strip clean oracle prefix; keep frame_0 + denoised gen frames
        latents = jnp.concatenate([latents[:, 0:1], latents[:, n_priv + 1:]], axis=1)
      latents = jnp.transpose(latents, (0, 4, 1, 2, 3))
      latents = self._denormalize_latents(latents)

    if output_type == "latent":
      return latents
    return self._decode_latents_to_video(latents)


def _build_per_token_timestep(
    latents: jnp.array, t, has_last_image: bool, n_priv: int = 0, oracle_noise_offset: Optional[int] = None
) -> jnp.array:
  """Build a per-token 2D timestep array for the WAN 2.2 Ti2V per-token conditioning scheme.

  Frame-0 token positions receive t=0 (clean); all other positions receive t.
  If n_priv > 0 and oracle_noise_offset is not None, oracle frames 1..n_priv receive
  max(0, t - oracle_noise_offset), matching the noise level used in _restore_clean_frames.
  If oracle_noise_offset is None, oracle frames receive t=0 (clean).
  If has_last_image, the last-frame token positions also receive t=0.

  latents shape: (B, num_latent_frames, latent_h, latent_w, C)  [BFHWC]
  Returns: (B, seq_len) where seq_len = num_latent_frames * (H//2) * (W//2)
  """
  bsz, num_latent_frames, latent_h, latent_w, _ = latents.shape
  tokens_per_frame = (latent_h // 2) * (latent_w // 2)  # spatial patch size is (2, 2)
  seq_len = num_latent_frames * tokens_per_frame
  timestep_2d = jnp.full((bsz, seq_len), t, dtype=jnp.int32)
  timestep_2d = timestep_2d.at[:, :tokens_per_frame].set(0)  # frame 0
  t_priv = jnp.maximum(t - oracle_noise_offset, 0) if oracle_noise_offset is not None else 0
  for i in range(1, 1 + n_priv):  # oracle future frames
    timestep_2d = timestep_2d.at[:, i * tokens_per_frame:(i + 1) * tokens_per_frame].set(t_priv)
  if has_last_image:
    timestep_2d = timestep_2d.at[:, -tokens_per_frame:].set(0)
  return timestep_2d


def run_inference_ti2v_2_2(
    graphdef,
    state,
    rest,
    latents: jnp.array,
    clean_latent: jnp.array,
    prompt_embeds: jnp.array,
    negative_prompt_embeds: jnp.array,
    guidance_scale: float,
    num_inference_steps: int,
    scheduler: FlaxUniPCMultistepScheduler,
    scheduler_state,
    use_cfg_cache: bool = False,
    use_sen_cache: bool = False,
    height: int = 480,
    has_last_image: bool = False,
    n_priv: int = 0,
    frame_positions: Optional[tuple] = None,
    oracle_noise_offset: Optional[int] = None,
):
  """Denoising loop for WAN 2.2 TI2V using per-token timestep conditioning.

  Image conditioning is embedded by initialising frame-0 (and optionally the
  last frame, or oracle future frames 1..n_priv) of `latents` with their clean
  VAE-encoded values, and assigning timestep=0 to those token positions via a
  2D per-token timestep array. After each scheduler step, those frames are
  restored to `clean_latent` to prevent the scheduler from corrupting them.

  n_priv > 0 enables oracle mode: future frames 1..n_priv from the conditioning
  video are also anchored as clean context throughout denoising.

  Supports two optional caching strategies:
    use_cfg_cache: FasterCache-style CFG caching with FFT frequency compensation.
    use_sen_cache: Sensitivity-Aware Caching (arXiv:2602.24208).
  """
  do_cfg = guidance_scale > 1.0
  bsz = latents.shape[0]

  def _restore_clean_frames(lat, t):
    lat = lat.at[:, 0:1, :, :, :].set(clean_latent[:, 0:1, :, :, :])
    t_val = int(t)
    for i in range(1, 1 + n_priv):
      if oracle_noise_offset is not None and t_val - oracle_noise_offset >= 0:
        t_priv = t_val - oracle_noise_offset
        t_norm = t_priv / scheduler.config.num_train_timesteps
        # Match training: FlowMatchScheduler defaults (sigma_min=0.003/1.002, sigma_max=1.0)
        sigma = (1.0 - t_norm) * (0.003 / 1.002) + t_norm * 1.0
        noise = jax.random.normal(
            jax.random.fold_in(jax.random.PRNGKey(t_val), i),
            shape=clean_latent[:, i:i + 1, :, :, :].shape,
        )
        noised = (1.0 - sigma) * clean_latent[:, i:i + 1, :, :, :] + sigma * noise
        lat = lat.at[:, i:i + 1, :, :, :].set(noised)
      else:
        lat = lat.at[:, i:i + 1, :, :, :].set(clean_latent[:, i:i + 1, :, :, :])
    if has_last_image:
      lat = lat.at[:, -1:, :, :, :].set(clean_latent[:, -1:, :, :, :])
    return lat

  # ── SenCache path ──
  if use_sen_cache and do_cfg:
    timesteps_np = np.array(scheduler_state.timesteps, dtype=np.int32)

    sen_epsilon = 0.1
    max_reuse = 3
    warmup_steps = 1
    nocache_start_ratio = 0.3
    nocache_end_ratio = 0.1
    alpha_x, alpha_t = 1.0, 1.0

    nocache_start = int(num_inference_steps * nocache_start_ratio)
    nocache_end_begin = int(num_inference_steps * (1.0 - nocache_end_ratio))
    num_train_timesteps = float(scheduler.config.num_train_timesteps)

    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

    ref_noise_pred = None
    ref_latent = None
    ref_timestep = 0.0
    accum_dx = 0.0
    accum_dt = 0.0
    reuse_count = 0
    cache_count = 0

    for step in range(num_inference_steps):
      t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
      latents = _restore_clean_frames(latents, t)
      t_float = float(timesteps_np[step]) / num_train_timesteps

      force_compute = (
          step < warmup_steps or step < nocache_start or step >= nocache_end_begin or ref_noise_pred is None
      )

      if force_compute:
        timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv, oracle_noise_offset)
        latents_doubled = jnp.concatenate([latents, latents], axis=0)
        latent_model_input = jnp.transpose(latents_doubled, (0, 4, 1, 2, 3))
        timestep_doubled = jnp.concatenate([timestep_2d, timestep_2d], axis=0)
        noise_pred, _, _ = transformer_forward_pass_full_cfg(
            graphdef,
            state,
            rest,
            latent_model_input,
            timestep_doubled,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
            encoder_hidden_states_image=None,
            frame_positions=frame_positions,
        )
        noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
        ref_noise_pred = noise_pred
        ref_latent = latents
        ref_timestep = t_float
        accum_dx = 0.0
        accum_dt = 0.0
        reuse_count = 0
        latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
        continue

      dx_norm = float(jnp.sqrt(jnp.mean((latents - ref_latent) ** 2)))
      dt = abs(t_float - ref_timestep)
      accum_dx += dx_norm
      accum_dt += dt
      score = alpha_x * accum_dx + alpha_t * accum_dt

      if score <= sen_epsilon and reuse_count < max_reuse:
        noise_pred = ref_noise_pred
        reuse_count += 1
        cache_count += 1
      else:
        timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv, oracle_noise_offset)
        latents_doubled = jnp.concatenate([latents, latents], axis=0)
        latent_model_input = jnp.transpose(latents_doubled, (0, 4, 1, 2, 3))
        timestep_doubled = jnp.concatenate([timestep_2d, timestep_2d], axis=0)
        noise_pred, _, _ = transformer_forward_pass_full_cfg(
            graphdef,
            state,
            rest,
            latent_model_input,
            timestep_doubled,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
            encoder_hidden_states_image=None,
            frame_positions=frame_positions,
        )
        noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
        ref_noise_pred = noise_pred
        ref_latent = latents
        ref_timestep = t_float
        accum_dx = 0.0
        accum_dt = 0.0
        reuse_count = 0

      latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()

    print(
        f"[SenCache] Cached {cache_count}/{num_inference_steps} steps "
        f"({100*cache_count/num_inference_steps:.1f}% cache ratio)"
    )
    return latents

  # ── CFG cache path ──
  if use_cfg_cache and do_cfg:
    if height >= 720:
      cfg_cache_interval = 5
      cfg_cache_start_step = int(num_inference_steps / 3)
      cfg_cache_end_step = int(num_inference_steps * 0.9)
      cfg_cache_alpha = 0.2
    else:
      cfg_cache_interval = 5
      cfg_cache_start_step = int(num_inference_steps / 3)
      cfg_cache_end_step = num_inference_steps - 1
      cfg_cache_alpha = 0.2

    prompt_cond_embeds = prompt_embeds
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)

    first_full_seen = False
    step_is_cache = []
    for s in range(num_inference_steps):
      is_cache = (
          first_full_seen
          and s >= cfg_cache_start_step
          and s < cfg_cache_end_step
          and (s - cfg_cache_start_step) % cfg_cache_interval != 0
      )
      step_is_cache.append(is_cache)
      if not is_cache:
        first_full_seen = True

    w1, w2 = 1.0, 1.0 + cfg_cache_alpha

    cached_noise_cond = None
    cached_noise_uncond = None

    for step in range(num_inference_steps):
      t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
      latents = _restore_clean_frames(latents, t)
      timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv, oracle_noise_offset)

      if step_is_cache[step]:
        latent_model_input = jnp.transpose(latents, (0, 4, 1, 2, 3))
        noise_pred, cached_noise_cond = transformer_forward_pass_cfg_cache(
            graphdef,
            state,
            rest,
            latent_model_input,
            timestep_2d,
            prompt_cond_embeds,
            cached_noise_cond,
            cached_noise_uncond,
            guidance_scale=guidance_scale,
            w1=jnp.float32(w1),
            w2=jnp.float32(w2),
            encoder_hidden_states_image=None,
            frame_positions=frame_positions,
        )
      else:
        latents_doubled = jnp.concatenate([latents, latents], axis=0)
        latent_model_input = jnp.transpose(latents_doubled, (0, 4, 1, 2, 3))
        timestep_doubled = jnp.concatenate([timestep_2d, timestep_2d], axis=0)
        noise_pred, cached_noise_cond, cached_noise_uncond = transformer_forward_pass_full_cfg(
            graphdef,
            state,
            rest,
            latent_model_input,
            timestep_doubled,
            prompt_embeds_combined,
            guidance_scale=guidance_scale,
            encoder_hidden_states_image=None,
            frame_positions=frame_positions,
        )

      noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
      latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    return latents

  # ── Basic path (no cache) ──
  if do_cfg:
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
  else:
    prompt_embeds_combined = prompt_embeds

  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    latents = _restore_clean_frames(latents, t)
    timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv, oracle_noise_offset)
    latents_input = jnp.concatenate([latents, latents], axis=0) if do_cfg else latents
    latent_model_input = jnp.transpose(latents_input, (0, 4, 1, 2, 3))
    timestep_for_transformer = jnp.concatenate([timestep_2d, timestep_2d], axis=0) if do_cfg else timestep_2d

    noise_pred, _ = transformer_forward_pass(
        graphdef,
        state,
        rest,
        latent_model_input,
        timestep_for_transformer,
        prompt_embeds_combined,
        do_cfg,
        guidance_scale,
        encoder_hidden_states_image=None,
        frame_positions=frame_positions,
    )
    noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
  return latents
