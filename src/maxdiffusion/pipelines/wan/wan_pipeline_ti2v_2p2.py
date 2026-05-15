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
    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    latent_height = height // self.vae_scale_factor_spatial
    latent_width = width // self.vae_scale_factor_spatial

    shape = (batch_size, num_latent_frames, latent_height, latent_width, num_channels_latents)

    if latents is None:
      latents = jax.random.normal(rng, shape=shape, dtype=jnp.float32)
    else:
      latents = latents.astype(dtype)

    if oracle_latents is not None:
      # Oracle conditioning: future frames from conditioning_video are prepended
      # before frame_0 in the latent tensor and treated exactly like frame_0 —
      # they receive t=0 in the per-token timestep and are restored after every
      # scheduler step, so the model sees them as clean context throughout.
      # Layout: [oracle_1..n_priv (t=0), frame_0 (t=0), noisy_gen_1..T'-1]
      # The oracle prefix is stripped from the output after the denoising loop.
      frame_0 = oracle_latents[:, 0:1, :, :, :]
      gen_shape = (batch_size, num_latent_frames - 1, latent_height, latent_width, num_channels_latents)
      gen_frames = jax.random.normal(rng, shape=gen_shape, dtype=jnp.float32)
      if n_priv > 0:
        oracle_future = oracle_latents[:, 1:1 + n_priv, :, :, :]
        latents = jnp.concatenate([oracle_future, frame_0, gen_frames], axis=1)
        clean_latent = jnp.concatenate([oracle_future, frame_0], axis=1)  # (B, n_priv+1, H, W, C)
      else:
        latents = jnp.concatenate([frame_0, gen_frames], axis=1)
        clean_latent = frame_0  # (B, 1, H, W, C)
    else:
      # Standard image path: encode the anchor image(s) via VAE.
      if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
      image = jnp.array(image)
      if last_image is not None:
        if hasattr(last_image, "detach"):
          last_image = last_image.detach().cpu().numpy()
        last_image = jnp.array(last_image)

      latent_condition, _ = self.prepare_latents_i2v_base(image, num_frames, dtype, last_image)
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
  ):
    if use_cfg_cache and use_sen_cache:
      raise ValueError("use_cfg_cache and use_sen_cache are mutually exclusive. Enable only one.")
    if conditioning_video is None and image is None:
      raise ValueError("Provide either 'image' or 'conditioning_video'.")

    height = height or self.config.height
    width = width or self.config.width
    num_frames = num_frames or self.config.num_frames

    if num_frames % self.vae_scale_factor_temporal != 1:
      max_logging.log(
          f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
          f"Rounding {num_frames} to the nearest valid number."
      )
      num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
      max_logging.log(f"Adjusted num_frames to: {num_frames}")
    num_frames = max(num_frames, 1)

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
    if conditioning_video is not None:
      oracle_latents = self._encode_video_to_i2v_latents(conditioning_video, prompt_embeds.dtype)
      if privileged:
        num_privileged_frames = getattr(self.config, "num_privileged_frames", -1)
        n_available = oracle_latents.shape[1] - 1  # T' - 1 (all frames except frame 0)
        n_priv = n_available if num_privileged_frames < 0 else min(num_privileged_frames, n_available)

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
      if n_priv > 0:
        latents = latents[:, n_priv:, :, :, :]  # strip oracle prefix; keep frame_0 + gen frames
      latents = jnp.transpose(latents, (0, 4, 1, 2, 3))
      latents = self._denormalize_latents(latents)

    if output_type == "latent":
      return latents
    return self._decode_latents_to_video(latents)


def _build_per_token_timestep(latents: jnp.array, t, has_last_image: bool, n_priv: int = 0) -> jnp.array:
  """Build a per-token 2D timestep array for the WAN 2.2 Ti2V per-token conditioning scheme.

  Frame-0 token positions receive t=0 (clean); all other positions receive t.
  If n_priv > 0, oracle frames 1..n_priv also receive t=0.
  If has_last_image, the last-frame token positions also receive t=0.

  latents shape: (B, num_latent_frames, latent_h, latent_w, C)  [BFHWC]
  Returns: (B, seq_len) where seq_len = num_latent_frames * (H//2) * (W//2)
  """
  bsz, num_latent_frames, latent_h, latent_w, _ = latents.shape
  tokens_per_frame = (latent_h // 2) * (latent_w // 2)  # spatial patch size is (2, 2)
  seq_len = num_latent_frames * tokens_per_frame
  timestep_2d = jnp.full((bsz, seq_len), t, dtype=jnp.int32)
  timestep_2d = timestep_2d.at[:, :tokens_per_frame].set(0)  # frame 0
  for i in range(1, 1 + n_priv):  # oracle future frames
    timestep_2d = timestep_2d.at[:, i * tokens_per_frame:(i + 1) * tokens_per_frame].set(0)
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

  def _restore_clean_frames(lat):
    lat = lat.at[:, 0:1, :, :, :].set(clean_latent[:, 0:1, :, :, :])
    for i in range(1, 1 + n_priv):
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
      t_float = float(timesteps_np[step]) / num_train_timesteps

      force_compute = (
          step < warmup_steps or step < nocache_start or step >= nocache_end_begin or ref_noise_pred is None
      )

      if force_compute:
        timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv)
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
        )
        noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
        ref_noise_pred = noise_pred
        ref_latent = latents
        ref_timestep = t_float
        accum_dx = 0.0
        accum_dt = 0.0
        reuse_count = 0
        latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
        latents = _restore_clean_frames(latents)
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
        timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv)
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
        )
        noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
        ref_noise_pred = noise_pred
        ref_latent = latents
        ref_timestep = t_float
        accum_dx = 0.0
        accum_dt = 0.0
        reuse_count = 0

      latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
      latents = _restore_clean_frames(latents)

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
      timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv)

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
        )

      noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
      latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
      latents = _restore_clean_frames(latents)
    return latents

  # ── Basic path (no cache) ──
  if do_cfg:
    prompt_embeds_combined = jnp.concatenate([prompt_embeds, negative_prompt_embeds], axis=0)
  else:
    prompt_embeds_combined = prompt_embeds

  for step in range(num_inference_steps):
    t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
    timestep_2d = _build_per_token_timestep(latents, t, has_last_image, n_priv)
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
    )
    noise_pred = jnp.transpose(noise_pred, (0, 2, 3, 4, 1))
    latents, scheduler_state = scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
    latents = _restore_clean_frames(latents)
  return latents
