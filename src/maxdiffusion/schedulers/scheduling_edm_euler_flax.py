# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""EDM Euler scheduler for Stable Video Diffusion (base).

Ports the four pieces used by SVD inference:

  - EDMDiscretization     (Karras et al. 2022 sigma schedule, rho spacing)
  - VScalingWithEDMcNoise (c_skip / c_out / c_in / c_noise for v-prediction)
  - EulerEDMSampler       (one-step Euler update over the EDM sigma grid)
  - LinearPredictionGuider (per-frame linear CFG for T-frame video)

References (sgm, Stability-AI/generative-models):
  sgm/modules/diffusionmodules/discretizer.py      EDMDiscretization
  sgm/modules/diffusionmodules/denoiser_scaling.py VScalingWithEDMcNoise
  sgm/modules/diffusionmodules/sampling.py         EulerEDMSampler
  sgm/modules/diffusionmodules/guiders.py          LinearPredictionGuider
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import flax
import jax.numpy as jnp

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import FlaxSchedulerMixin, FlaxSchedulerOutput


@flax.struct.dataclass
class EDMEulerSchedulerState:
  """Scheduler state: the sigma grid and step counter.

  sigmas has shape (N+1,): sigmas[0] = sigma_max, sigmas[N] = 0.
  timesteps has shape (N,) and mirrors sigmas[:-1] (for API parity with
  other FlaxSchedulers).
  """

  sigmas: jnp.ndarray
  timesteps: jnp.ndarray
  init_noise_sigma: jnp.ndarray
  num_inference_steps: Optional[int] = None


@dataclass
class FlaxEDMEulerSchedulerOutput(FlaxSchedulerOutput):
  state: EDMEulerSchedulerState


def edm_sigmas(
    num_inference_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 700.0,
    rho: float = 7.0,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
  """Karras rho-schedule sigmas, descending, with a trailing 0.

  sigma_i = (sigma_max^{1/rho} + i/(N-1) * (sigma_min^{1/rho} - sigma_max^{1/rho}))^rho
  Returns shape (N+1,): the N scheduled sigmas followed by 0.
  """
  ramp = jnp.linspace(0.0, 1.0, num_inference_steps, dtype=dtype)
  min_inv = sigma_min ** (1.0 / rho)
  max_inv = sigma_max ** (1.0 / rho)
  sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
  sigmas = jnp.concatenate([sigmas, jnp.zeros((1,), dtype=dtype)], axis=0)
  return sigmas


def v_scaling_edm(sigma: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """EDM preconditioning for v-prediction.

  Returns (c_skip, c_out, c_in, c_noise). Matches
  sgm/modules/diffusionmodules/denoiser_scaling.py:VScalingWithEDMcNoise.
  """
  sigma2 = sigma * sigma
  c_skip = 1.0 / (sigma2 + 1.0)
  c_out = -sigma / jnp.sqrt(sigma2 + 1.0)
  c_in = 1.0 / jnp.sqrt(sigma2 + 1.0)
  c_noise = 0.25 * jnp.log(sigma)
  return c_skip, c_out, c_in, c_noise


def apply_v_denoiser(
    raw_model_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    sample: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
  """Wrap raw_model_fn(x_in, t_in) -> model_output as an EDM v-prediction denoiser.

  Computes D(x, sigma) = c_skip * x + c_out * model(c_in * x, c_noise).
  sigma is a scalar; broadcast manually when the model sees a batched sigma.
  """
  c_skip, c_out, c_in, c_noise = v_scaling_edm(sigma)
  model_out = raw_model_fn(c_in * sample, c_noise)
  return c_skip * sample + c_out * model_out


class LinearPredictionGuider:
  """Per-frame linear classifier-free guidance used by SVD.

  Scale ramps linearly along the time axis from `min_scale` (first frame) to
  `max_scale` (last frame). Inputs are assumed to have leading batch dim of
  `B * T` with C/H/W (or H/W/C) trailing; the guider reshapes internally to
  `(B, T, ...)`, applies the per-frame scale, then flattens back.
  """

  def __init__(self, num_frames: int, min_scale: float = 1.0, max_scale: float = 2.5):
    self.num_frames = num_frames
    self.min_scale = min_scale
    self.max_scale = max_scale

  def scales(self, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Per-frame CFG scales, shape (T,)."""
    return jnp.linspace(self.min_scale, self.max_scale, self.num_frames, dtype=dtype)

  def __call__(self, x_cond: jnp.ndarray, x_uncond: jnp.ndarray) -> jnp.ndarray:
    """Apply per-frame CFG.

    x_cond and x_uncond have matching shape with leading dim `B*T`. The
    per-frame scale broadcasts over all trailing dims.
    """
    if x_cond.shape != x_uncond.shape:
      raise ValueError(
          f"LinearPredictionGuider shape mismatch: cond={x_cond.shape}, uncond={x_uncond.shape}"
      )
    bt = x_cond.shape[0]
    if bt % self.num_frames != 0:
      raise ValueError(
          f"LinearPredictionGuider batch {bt} not divisible by num_frames={self.num_frames}"
      )
    b = bt // self.num_frames
    trailing = x_cond.shape[1:]
    scales = self.scales(dtype=x_cond.dtype)  # (T,)
    # Broadcast to (1, T, 1, 1, ...) to match (B, T, *trailing).
    scale = scales.reshape((1, self.num_frames) + (1,) * len(trailing))
    cond = x_cond.reshape((b, self.num_frames) + trailing)
    uncond = x_uncond.reshape((b, self.num_frames) + trailing)
    out = uncond + scale * (cond - uncond)
    return out.reshape((bt,) + trailing)


class FlaxEDMEulerScheduler(FlaxSchedulerMixin, ConfigMixin):
  """Karras EDM sigma schedule + Euler step + v-prediction, in Flax.

  Usage (mirrors FlaxEulerDiscreteScheduler):

    sched = FlaxEDMEulerScheduler(sigma_min=0.002, sigma_max=700.0, rho=7.0)
    state = sched.create_state()
    state = sched.set_timesteps(state, num_inference_steps=25)

    # Initial noise has std sigma_max.
    x = jax.random.normal(key, shape) * state.init_noise_sigma

    for i in range(state.num_inference_steps):
      sigma = state.sigmas[i]
      x_in = sched.scale_model_input(state, x, i)   # c_in(sigma) * x
      t_in = sched.scale_noise_input(sigma)         # c_noise(sigma)
      model_output = unet(x_in, t_in, ...)
      pred_x0 = sched.combine_v_prediction(x, sigma, model_output)
      x = sched.step(state, pred_x0, sigma, x)
  """

  dtype: jnp.dtype

  @property
  def has_state(self) -> bool:
    return True

  @register_to_config
  def __init__(
      self,
      sigma_min: float = 0.002,
      sigma_max: float = 700.0,
      rho: float = 7.0,
      prediction_type: str = "v_prediction",
      num_train_timesteps: int = 1000,
      dtype: jnp.dtype = jnp.float32,
  ):
    if prediction_type != "v_prediction":
      raise ValueError(
          f"FlaxEDMEulerScheduler only supports prediction_type='v_prediction', got {prediction_type}"
      )
    self.dtype = dtype

  def create_state(self, num_inference_steps: int = 25) -> EDMEulerSchedulerState:
    sigmas = edm_sigmas(
        num_inference_steps=num_inference_steps,
        sigma_min=self.config.sigma_min,
        sigma_max=self.config.sigma_max,
        rho=self.config.rho,
        dtype=self.dtype,
    )
    timesteps = sigmas[:-1]
    return EDMEulerSchedulerState(
        sigmas=sigmas,
        timesteps=timesteps,
        init_noise_sigma=jnp.asarray(self.config.sigma_max, dtype=self.dtype),
        num_inference_steps=num_inference_steps,
    )

  def set_timesteps(
      self, state: EDMEulerSchedulerState, num_inference_steps: int, shape: Tuple = ()
  ) -> EDMEulerSchedulerState:
    sigmas = edm_sigmas(
        num_inference_steps=num_inference_steps,
        sigma_min=self.config.sigma_min,
        sigma_max=self.config.sigma_max,
        rho=self.config.rho,
        dtype=self.dtype,
    )
    return state.replace(
        sigmas=sigmas,
        timesteps=sigmas[:-1],
        init_noise_sigma=jnp.asarray(self.config.sigma_max, dtype=self.dtype),
        num_inference_steps=num_inference_steps,
    )

  def scale_model_input(
      self, state: EDMEulerSchedulerState, sample: jnp.ndarray, step_index: Union[int, jnp.ndarray]
  ) -> jnp.ndarray:
    """c_in(sigma) * sample. step_index is an integer in [0, N)."""
    sigma = state.sigmas[step_index]
    c_in = 1.0 / jnp.sqrt(sigma * sigma + 1.0)
    return sample * c_in

  def scale_noise_input(self, sigma: jnp.ndarray) -> jnp.ndarray:
    """c_noise(sigma) = 0.25 * log(sigma). Returned as a scalar."""
    return 0.25 * jnp.log(sigma)

  def combine_v_prediction(
      self, sample: jnp.ndarray, sigma: jnp.ndarray, model_output: jnp.ndarray
  ) -> jnp.ndarray:
    """Fold raw v-prediction model_output back into D(x, sigma)."""
    c_skip, c_out, _, _ = v_scaling_edm(sigma)
    return c_skip * sample + c_out * model_output

  def step(
      self,
      state: EDMEulerSchedulerState,
      pred_x0: jnp.ndarray,
      sigma: jnp.ndarray,
      sample: jnp.ndarray,
      step_index: Union[int, jnp.ndarray],
      return_dict: bool = True,
  ) -> Union[FlaxEDMEulerSchedulerOutput, Tuple]:
    """One Euler step: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * (x_i - D) / sigma_i.

    `pred_x0` is the denoiser output D(x, sigma), i.e. what
    `combine_v_prediction` returned. `sigma` is the current scheduler sigma,
    `step_index` is the integer index into state.sigmas.
    """
    derivative = (sample - pred_x0) / sigma
    dt = state.sigmas[step_index + 1] - sigma
    prev_sample = sample + derivative * dt

    if not return_dict:
      return (prev_sample, state)
    return FlaxEDMEulerSchedulerOutput(prev_sample=prev_sample, state=state)

  def __len__(self) -> int:
    return self.config.num_train_timesteps
