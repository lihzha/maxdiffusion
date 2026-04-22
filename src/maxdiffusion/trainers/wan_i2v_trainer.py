"""WAN I2V trainer.

Extends WanTrainer with:
  - I2V-aware data loading (TFRecord with condition + image_embeds fields)
  - I2V training step (33-channel hidden_states, CLIP image conditioning)
  - preprocess_batch hook for on-the-fly encoding when dataset_type="droid"

TFRecord path  (dataset_type="tfrecord"):
    Pre-encoded records must contain four fields:
      latents, encoder_hidden_states, condition, encoder_hidden_states_image
    Use wan_i2v_data_preprocessing.py to produce them.

DROID path  (dataset_type="droid"):
    Raw DROID TFDS records are loaded by DroidVideoDataset.
    preprocess_batch runs VAE / T5 / CLIP encoding on every batch.
    The VAE and text/image encoders remain loaded throughout training;
    do NOT call pipeline.delete_vae() before the training loop.
"""

import functools

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
import jaxopt
import tensorflow as tf

from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.trainers.base_wan_trainer import BaseWanTrainer


class WanI2VTrainer(BaseWanTrainer):

  def _get_checkpointer(self):
    if self.config.model_name == "wan2.1":
      return WanCheckpointerI2V_2_1(config=self.config)
    return WanCheckpointerI2V_2_2(config=self.config)

  # ── Data shardings ─────────────────────────────────────────────────────────

  def get_data_shardings(self, mesh):
    shard = NamedSharding(mesh, P(*self.config.data_sharding))
    shardings = {
        "latents": shard,
        "encoder_hidden_states": shard,
        "condition": shard,
    }
    if self.config.model_name == "wan2.1":
      shardings["encoder_hidden_states_image"] = shard
    return shardings

  def get_eval_data_shardings(self, mesh):
    shardings = self.get_data_shardings(mesh)
    shardings["timesteps"] = NamedSharding(mesh, P(*self.config.data_sharding))
    return shardings

  # ── Dataset loading ────────────────────────────────────────────────────────

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config

    if config.dataset_type == "synthetic":
      return make_data_iterator(
          config,
          jax.process_index(),
          jax.process_count(),
          mesh,
          config.global_batch_size_to_load,
          pipeline=pipeline,
          is_training=is_training,
      )

    if config.dataset_type == "droid":
      # Raw-pixel path: encoding is deferred to preprocess_batch.
      return make_data_iterator(
          config,
          jax.process_index(),
          jax.process_count(),
          mesh,
          config.global_batch_size_to_load,
          is_training=is_training,
      )

    # TFRecord path (pre-encoded latents).
    if config.dataset_type != "tfrecord" or not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "WanI2VTrainer requires dataset_type='tfrecord' with "
          "cache_latents_text_encoder_outputs=True, or dataset_type='droid'."
      )

    feature_description = {
        "latents": tf.io.FixedLenFeature([], tf.string),
        "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
        "condition": tf.io.FixedLenFeature([], tf.string),
    }
    if config.model_name == "wan2.1":
      feature_description["encoder_hidden_states_image"] = tf.io.FixedLenFeature([], tf.string)

    if not is_training:
      feature_description["timesteps"] = tf.io.FixedLenFeature([], tf.int64)

    def prepare_sample_train(features):
      latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
      encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
      condition = tf.io.parse_tensor(features["condition"], out_type=tf.float32)
      out = {
          "latents": latents,
          "encoder_hidden_states": encoder_hidden_states,
          "condition": condition,
      }
      if config.model_name == "wan2.1":
        out["encoder_hidden_states_image"] = tf.io.parse_tensor(
            features["encoder_hidden_states_image"], out_type=tf.float32
        )
      return out

    def prepare_sample_eval(features):
      out = prepare_sample_train(features)
      out["timesteps"] = features["timesteps"]
      return out

    return make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample_train if is_training else prepare_sample_eval,
        is_training=is_training,
    )

  # ── On-the-fly encoding (DROID path) ──────────────────────────────────────

  def preprocess_batch(self, batch, pipeline):
    """Encode raw video frames into the I2V training tensors.

    Called every step only when dataset_type="droid". For the TFRecord path
    the batch already contains latents/condition/embeddings, so this is a no-op.
    """
    if "frames" not in batch:
      return batch  # TFRecord or synthetic path — already encoded.

    import functools
    import numpy as np

    frames = batch["frames"]          # [B, T, H, W, 3]  float32 numpy [0, 1]
    texts = batch["language_instruction"]  # [B]  bytes

    # ── Text encoding (T5) ─────────────────────────────────────────────────
    texts_str = [t.decode("utf-8") if isinstance(t, bytes) else t for t in texts]
    encoder_hidden_states = pipeline._get_t5_prompt_embeds(texts_str)  # torch
    encoder_hidden_states = jnp.array(
        encoder_hidden_states.detach().float().numpy(), dtype=self.config.weights_dtype
    )  # [B, 512, 4096]

    # ── Video encoding (VAE) ───────────────────────────────────────────────
    # frames: [B, T, H, W, 3] → VAE expects [B, T, H, W, C] then transposes internally
    frames_jax = jnp.array(frames, dtype=self.config.weights_dtype)
    latents, condition = self._encode_video_i2v(frames_jax, pipeline)
    # latents:   [B, 16, T', H', W']
    # condition: [B, 17, T', H', W']

    out = {
        "latents": latents,
        "encoder_hidden_states": encoder_hidden_states,
        "condition": condition,
    }

    # ── CLIP image encoding (Wan 2.1 only) ────────────────────────────────
    if self.config.model_name == "wan2.1":
      first_frames = frames_jax[:, 0]  # [B, H, W, 3]
      image_embeds = self._encode_clip(first_frames, pipeline)  # [B, 257, 1280]
      out["encoder_hidden_states_image"] = image_embeds

    return out

  def _encode_video_i2v(self, frames_jax, pipeline):
    """VAE-encode a batch of video clips and build the I2V condition tensor.

    Args:
      frames_jax: [B, T, H, W, 3] float32 pixels in [0, 1].
      pipeline:   WanPipelineI2V with loaded VAE.

    Returns:
      latents:   [B, 16, T', H', W']  normalised video latents.
      condition: [B, 17, T', H', W']  concat([mask, latent_first_frame]).
    """
    import jax

    vae = pipeline.vae
    vae_cache = pipeline.vae_cache

    # Normalise to [-1, 1] as the VAE expects.
    video = frames_jax * 2.0 - 1.0  # [B, T, H, W, 3]

    rng = jax.random.key(self.config.seed)

    @functools.lru_cache(maxsize=1)
    def _jit_vae_encode():
      import functools as ft
      from maxdiffusion.data_preprocessing.wan_txt2vid_data_preprocessing import vae_encode
      return jax.jit(ft.partial(vae_encode, vae=vae, vae_cache=vae_cache))

    p_vae_encode = _jit_vae_encode()

    with pipeline.mesh:
      latents = p_vae_encode(video=video, rng=rng)
    # latents: [B, 16, T', H', W'] (channels-first, normalised)

    # ── Build condition tensor ─────────────────────────────────────────────
    # Encode only the first frame (with zeros for remaining frames).
    B, T, H, W, C = video.shape
    first_frame = video[:, 0:1, :, :, :]              # [B, 1, H, W, 3]
    video_cond = jnp.concatenate(
        [first_frame, jnp.zeros((B, T - 1, H, W, C), dtype=video.dtype)],
        axis=1,
    )  # [B, T, H, W, 3]
    with pipeline.mesh:
      latent_cond = p_vae_encode(video=video_cond, rng=rng)
    # latent_cond: [B, 16, T', H', W']

    _, _, T_lat, H_lat, W_lat = latents.shape
    mask = jnp.zeros((B, 1, T_lat, H_lat, W_lat), dtype=latents.dtype)
    mask = mask.at[:, :, 0, :, :].set(1.0)            # first latent frame = conditioning

    condition = jnp.concatenate([mask, latent_cond], axis=1)  # [B, 17, T', H', W']
    return latents, condition

  def _encode_clip(self, first_frames, pipeline):
    """CLIP-encode the first frame of each clip.

    Args:
      first_frames: [B, H, W, 3] float32 pixels in [0, 1].
      pipeline:     WanPipelineI2V_2_1 with loaded FlaxCLIPVisionModel image_encoder.

    Returns:
      image_embeds: [B, 257, 1280]
    """
    # Resize to CLIP input resolution (224×224) and normalise.
    pixels_np = np.array(first_frames)  # [B, H, W, 3], float32 [0, 1]
    pixels_resized = tf.image.resize(pixels_np, [224, 224]).numpy()  # [B, 224, 224, 3]
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    pixels_norm = (pixels_resized - mean) / std  # [B, 224, 224, 3]
    pixel_values = jnp.array(pixels_norm.transpose(0, 3, 1, 2))  # [B, 3, 224, 224]

    out = pipeline.image_encoder(pixel_values, output_hidden_states=True)
    image_embeds = out.hidden_states[-2]  # [B, 257, 1280]
    return jnp.array(image_embeds, dtype=self.config.weights_dtype)

  # ── Train / eval steps ─────────────────────────────────────────────────────

  def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
    return jax.jit(
        functools.partial(
            i2v_train_step,
            scheduler=pipeline.scheduler,
            config=self.config,
            model_name=self.config.model_name,
        ),
        in_shardings=(state_shardings, data_shardings, None, None),
        out_shardings=(state_shardings, None, None, None),
        donate_argnums=(0,),
    )

  def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
    return jax.jit(
        functools.partial(
            i2v_eval_step,
            scheduler=pipeline.scheduler,
            config=self.config,
            model_name=self.config.model_name,
        ),
        in_shardings=(state_shardings, eval_data_shardings, None, None),
        out_shardings=(None, None),
    )


# ── Training step ─────────────────────────────────────────────────────────────

def i2v_train_step(state, data, rng, scheduler_state, scheduler, config, model_name):
  return _i2v_step_optimizer(state, data, rng, scheduler_state, scheduler, config, model_name)


def _i2v_step_optimizer(state, data, rng, scheduler_state, scheduler, config, model_name):
  _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

  for k, v in data.items():
    if hasattr(v, "shape"):
      data[k] = v[: config.global_batch_size_to_train_on]

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)
    condition = data["condition"].astype(config.weights_dtype)

    bsz = latents.shape[0]
    timesteps = scheduler.sample_timesteps(timestep_rng, bsz)
    noise = jax.random.normal(key=new_rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = scheduler.apply_flow_match(
        noise, latents, timesteps
    )

    # I2V: concatenate image-conditioning along the channel axis (channels-first).
    latent_model_input = jnp.concatenate([noisy_latents, condition], axis=1)  # [B, 33, T', H', W']

    image_embeds = None
    if model_name == "wan2.1":
      image_embeds = data["encoder_hidden_states_image"].astype(config.weights_dtype)

    with jax.named_scope("forward_pass"):
      model_pred = model(
          hidden_states=latent_model_input,
          timestep=timesteps,
          encoder_hidden_states=encoder_hidden_states,
          encoder_hidden_states_image=image_embeds,
          deterministic=False,
          rngs=nnx.Rngs(dropout=dropout_rng),
      )

    with jax.named_scope("loss"):
      loss = (training_target - model_pred) ** 2
      if not config.disable_training_weights:
        training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
        loss = loss * training_weight
      loss = jnp.mean(loss)

    return loss

  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  max_grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
  max_abs_grad = jax.tree_util.tree_reduce(
      lambda m, arr: jnp.maximum(m, jnp.max(jnp.abs(arr))), grads, initializer=-1.0
  )

  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": max_grad_norm,
          "learning/max_abs_grad": max_abs_grad,
      },
      "scalars": {},
  }

  new_state = state.apply_gradients(grads=grads)
  return new_state, scheduler_state, metrics, new_rng


# ── Eval step ─────────────────────────────────────────────────────────────────

def i2v_eval_step(state, data, rng, scheduler_state, scheduler, config, model_name):
  def loss_fn(params, latents, encoder_hidden_states, condition, image_embeds, timesteps, rng):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    noise = jax.random.normal(key=rng, shape=latents.shape, dtype=latents.dtype)
    noisy_latents, training_target, training_weight = scheduler.apply_flow_match(
        noise, latents, timesteps
    )
    latent_model_input = jnp.concatenate([noisy_latents, condition], axis=1)
    model_pred = model(
        hidden_states=latent_model_input,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_image=image_embeds,
        deterministic=True,
    )
    loss = (training_target - model_pred) ** 2
    if not config.disable_training_weights:
      training_weight = jnp.expand_dims(training_weight, axis=(1, 2, 3, 4))
      loss = loss * training_weight
    return loss.reshape(loss.shape[0], -1).mean(axis=1)

  bs = len(data["latents"])
  single_batch_size = config.global_batch_size_to_train_on
  losses = jnp.zeros(bs)
  for i in range(0, bs, single_batch_size):
    end = min(i + single_batch_size, bs)
    latents = data["latents"][i:end].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"][i:end].astype(config.weights_dtype)
    condition = data["condition"][i:end].astype(config.weights_dtype)
    image_embeds = data.get("encoder_hidden_states_image")
    if image_embeds is not None:
      image_embeds = image_embeds[i:end].astype(config.weights_dtype)
    timesteps = data["timesteps"][i:end].astype("int64")
    _, new_rng = jax.random.split(rng, num=2)
    loss = loss_fn(state.params, latents, encoder_hidden_states, condition, image_embeds, timesteps, new_rng)
    losses = losses.at[i:end].set(loss)

  metrics = {"scalar": {"learning/eval_loss": losses}}
  return metrics, new_rng
