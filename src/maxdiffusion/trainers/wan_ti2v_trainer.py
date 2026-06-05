"""WAN TI2V trainer.

Extends BaseWanTrainer with the per-token timestep training objective used by
WAN 2.2 Ti2V (Wan-AI/Wan2.2-TI2V-5B-Diffusers):

  * History latent frames (first config.num_history_latent_frames) are kept
    clean (no noise added).  With the default num_history_latent_frames=1 this
    mirrors inference: frame 0 is the image-conditioning anchor, frames 1+ are
    the frames to generate.
  * Future latent frames receive flow-matching noise at a sampled global t.
  * A (B, seq_len) per-token timestep array is passed to the transformer:
    history frame tokens get t=0, future frame tokens get the sampled t.
    WAN patches spatially at 2x2, so tokens_per_frame = (H_lat//2)*(W_lat//2).
  * MSE loss is computed only on the future latent frames.

TFRecord path (dataset_type="tfrecord"):
    TFRecords produced by wan_convert.sh (wan2.2_txt2vid_data_preprocessing.py).
    Each record stores one full episode with:
      latent_cam0/1/2  float16  (F_lat, C, H_lat, W_lat)  time-first per camera
      text_embed       float16  (512, 4096)
      traj_len         int64
    Clips should be longer than window_size = 1 + num_frames // 4 latent frames
    so that random temporal windowing samples different sub-clips each epoch.
    During training one camera is picked at random per sample; cam0 (wrist) is
    used for eval.
    Eval samples timesteps uniformly (DROID records have no pre-sampled timesteps
    field), so per-timestep loss bucketing is skipped — only mean eval loss is
    logged.
"""

import datetime
import functools

import jax
import jax.numpy as jnp
import jaxopt
import tensorflow as tf
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging
from maxdiffusion.checkpointing.wan_checkpointer_ti2v_2p2 import WanCheckpointerTI2V_2_2
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.train_utils import load_next_batch
from maxdiffusion.trainers.base_wan_trainer import BaseWanTrainer


def _build_per_token_timestep(
    timesteps: jnp.ndarray,
    F_lat: int,
    H_lat: int,
    W_lat: int,
    n_hist: int,
) -> jnp.ndarray:
    """Build (B, seq_len) timestep array for per-token Ti2V training.

    History frame tokens (indices 0..n_hist-1) receive t=0 (treated as clean
    by the transformer's AdaLN); future frame tokens receive the sampled t.
    Matches the per-token scheme used in wan_pipeline_ti2v_2p2 inference.
    """
    b = timesteps.shape[0]
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    seq_len = F_lat * tokens_per_frame
    n_hist_tokens = n_hist * tokens_per_frame
    full = jnp.broadcast_to(timesteps[:, None], (b, seq_len))
    is_future = jnp.arange(seq_len)[None, :] >= n_hist_tokens
    return jnp.where(is_future, full, jnp.zeros_like(full))


class WanTI2VTrainer(BaseWanTrainer):

    def _get_checkpointer(self):
        return WanCheckpointerTI2V_2_2(config=self.config)

    # ── Data shardings ───────────────────────────────────────────────────────

    def get_data_shardings(self, mesh):
        shard = NamedSharding(mesh, P(*self.config.data_sharding))
        return {
            "latents": shard,
            "encoder_hidden_states": shard,
        }

    def get_eval_data_shardings(self, mesh):
        # No timesteps field — DROID records don't store pre-sampled timesteps.
        return self.get_data_shardings(mesh)

    # ── Dataset loading ──────────────────────────────────────────────────────

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

        if config.dataset_type != "tfrecord" or not config.cache_latents_text_encoder_outputs:
            raise ValueError(
                "WanTI2VTrainer requires dataset_type='tfrecord' with "
                "cache_latents_text_encoder_outputs=True."
            )

        # Schema from wan_convert.sh / wan2.2_txt2vid_data_preprocessing.py.
        # Latents are time-first (F_lat, C, H_lat, W_lat) float16.
        # text_embed is (512, 4096) float16.
        feature_description = {
            "latent_cam0": tf.io.FixedLenFeature([], tf.string),
            "latent_cam1": tf.io.FixedLenFeature([], tf.string),
            "latent_cam2": tf.io.FixedLenFeature([], tf.string),
            "text_embed":  tf.io.FixedLenFeature([], tf.string),
            "traj_len":    tf.io.FixedLenFeature([], tf.int64),
        }

        # WAN VAE: 1 anchor + num_frames // 4 generated latent frames.
        window_size = 1 + config.num_frames // 4

        def prepare_sample_train(features):
            cam0 = tf.cast(tf.io.parse_tensor(features["latent_cam0"], out_type=tf.float16), tf.float32)
            cam1 = tf.cast(tf.io.parse_tensor(features["latent_cam1"], out_type=tf.float16), tf.float32)
            cam2 = tf.cast(tf.io.parse_tensor(features["latent_cam2"], out_type=tf.float16), tf.float32)
            # Concat cameras along H (axis 2): (F_lat, C, H_lat*3, W_lat) — matches ctrl_world.
            latent = tf.concat([cam0, cam1, cam2], axis=2)

            # Random temporal window on axis 0 (time), then transpose to channels-first.
            f_total = tf.shape(latent)[0]
            max_start = tf.maximum(0, f_total - window_size)
            start = tf.random.uniform((), 0, max_start + 1, dtype=tf.int32)
            latent = latent[start : start + window_size]       # (window_size, C, H_lat*3, W_lat)
            latent = tf.transpose(latent, [1, 0, 2, 3])        # (C, window_size, H_lat*3, W_lat)

            encoder_hidden_states = tf.cast(
                tf.io.parse_tensor(features["text_embed"], out_type=tf.float16), tf.float32
            )  # (512, 4096)
            return {"latents": latent, "encoder_hidden_states": encoder_hidden_states}

        def prepare_sample_eval(features):
            cam0 = tf.cast(tf.io.parse_tensor(features["latent_cam0"], out_type=tf.float16), tf.float32)
            cam1 = tf.cast(tf.io.parse_tensor(features["latent_cam1"], out_type=tf.float16), tf.float32)
            cam2 = tf.cast(tf.io.parse_tensor(features["latent_cam2"], out_type=tf.float16), tf.float32)
            latent = tf.concat([cam0, cam1, cam2], axis=2)     # (F_lat, C, H_lat*3, W_lat)
            latent = latent[:window_size]                      # (window_size, C, H_lat*3, W_lat)
            latent = tf.transpose(latent, [1, 0, 2, 3])        # (C, window_size, H_lat*3, W_lat)
            encoder_hidden_states = tf.cast(
                tf.io.parse_tensor(features["text_embed"], out_type=tf.float16), tf.float32
            )  # (512, 4096)
            return {"latents": latent, "encoder_hidden_states": encoder_hidden_states}

        # Drop clips whose latent temporal dim is shorter than window_size.
        # Latents are channels-first (C, F_lat, H, W) after prepare_sample.
        def filter_short_clips(sample):
            return tf.shape(sample["latents"])[1] >= window_size

        return make_data_iterator(
            config,
            jax.process_index(),
            jax.process_count(),
            mesh,
            config.global_batch_size_to_load,
            feature_description=feature_description,
            prepare_sample_fn=prepare_sample_train if is_training else prepare_sample_eval,
            is_training=is_training,
            filter_fn=filter_short_clips,
        )

    # ── Train / eval steps ───────────────────────────────────────────────────

    def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
        n_hist = getattr(self.config, "num_history_latent_frames", 1)
        return jax.jit(
            functools.partial(
                ti2v_train_step,
                scheduler=pipeline.scheduler,
                config=self.config,
                n_hist=n_hist,
            ),
            in_shardings=(state_shardings, data_shardings, None, None),
            out_shardings=(state_shardings, None, None, None),
            donate_argnums=(0,),
        )

    def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
        n_hist = getattr(self.config, "num_history_latent_frames", 1)
        return jax.jit(
            functools.partial(
                ti2v_eval_step,
                scheduler=pipeline.scheduler,
                config=self.config,
                n_hist=n_hist,
            ),
            in_shardings=(state_shardings, eval_data_shardings, None, None),
            out_shardings=(None, None),
        )

    def eval(self, mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer):
        """Eval on a single batch, advancing through the val set across calls."""
        if not hasattr(self, "_eval_data_iterator"):
            self._eval_data_iterator = self.load_dataset(mesh, is_training=False)
        eval_rng = eval_rng_key

        eval_batch = load_next_batch(self._eval_data_iterator, None, self.config)
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            metrics, eval_rng = p_eval_step(state, eval_batch, eval_rng, scheduler_state)
            metrics["scalar"]["learning/eval_loss"].block_until_ready()
        losses = metrics["scalar"]["learning/eval_loss"]
        gathered = multihost_utils.process_allgather(losses, tiled=True)
        all_losses = jax.device_get(gathered).flatten().tolist()

        if all_losses and jax.process_index() == 0:
            final_eval_loss = float(jnp.mean(jnp.array(all_losses)))
            max_logging.log(f"Step {step}, Eval loss: {final_eval_loss:.4f}")
            if writer:
                writer.add_scalar("learning/eval_loss", final_eval_loss, step)
            if getattr(self, "_wandb_run", None) is not None:
                self._wandb_run.log({"eval/loss": final_eval_loss}, step=step)


# ── Training step ─────────────────────────────────────────────────────────────


def ti2v_train_step(state, data, rng, scheduler_state, scheduler, config, n_hist):
    _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

    for k, v in data.items():
        if hasattr(v, "shape"):
            data[k] = v[: config.global_batch_size_to_train_on]

    def loss_fn(params):
        model = nnx.merge(state.graphdef, params, state.rest_of_state)
        # latents: (B, C, F_lat, H_lat, W_lat) channels-first
        latents = data["latents"].astype(config.weights_dtype)
        encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)

        b, _, F_lat, H_lat, W_lat = latents.shape
        timesteps = scheduler.sample_timesteps(timestep_rng, b)

        # Noise only future frames; history frames stay clean.
        future_latents = latents[:, :, n_hist:]
        noise = jax.random.normal(new_rng, future_latents.shape, dtype=future_latents.dtype)
        noisy_future, target_future, training_weight = scheduler.apply_flow_match(
            noise, future_latents, timesteps
        )
        noisy_latents = jnp.concatenate([latents[:, :, :n_hist], noisy_future], axis=2)

        # Per-token timestep: history frame tokens → 0, future frame tokens → t.
        timestep_2d = _build_per_token_timestep(timesteps, F_lat, H_lat, W_lat, n_hist)

        with jax.named_scope("forward_pass"):
            model_pred = model(
                hidden_states=noisy_latents,
                timestep=timestep_2d,
                encoder_hidden_states=encoder_hidden_states,
                deterministic=False,
                rngs=nnx.Rngs(dropout=dropout_rng),
            )

        with jax.named_scope("loss"):
            diff = target_future - model_pred[:, :, n_hist:]
            loss = diff ** 2
            if not config.disable_training_weights:
                loss = loss * jnp.expand_dims(training_weight, (1, 2, 3, 4))
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


def ti2v_eval_step(state, data, rng, scheduler_state, scheduler, config, n_hist):

    def loss_fn(params, latents, encoder_hidden_states, timesteps, rng):
        model = nnx.merge(state.graphdef, params, state.rest_of_state)
        b, _, F_lat, H_lat, W_lat = latents.shape
        future_latents = latents[:, :, n_hist:]
        noise = jax.random.normal(rng, future_latents.shape, dtype=future_latents.dtype)
        noisy_future, target_future, training_weight = scheduler.apply_flow_match(
            noise, future_latents, timesteps
        )
        noisy_latents = jnp.concatenate([latents[:, :, :n_hist], noisy_future], axis=2)
        timestep_2d = _build_per_token_timestep(timesteps, F_lat, H_lat, W_lat, n_hist)
        model_pred = model(
            hidden_states=noisy_latents,
            timestep=timestep_2d,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=True,
        )
        diff = target_future - model_pred[:, :, n_hist:]
        loss = diff ** 2
        if not config.disable_training_weights:
            loss = loss * jnp.expand_dims(training_weight, (1, 2, 3, 4))
        return loss.reshape(loss.shape[0], -1).mean(axis=1)

    bs = config.global_batch_size_to_train_on
    single_batch_size = config.global_batch_size_to_train_on
    losses = jnp.zeros(bs)
    for i in range(0, bs, single_batch_size):
        end = min(i + single_batch_size, bs)
        latents = data["latents"][i:end].astype(config.weights_dtype)
        encoder_hidden_states = data["encoder_hidden_states"][i:end].astype(config.weights_dtype)
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        timesteps = scheduler.sample_timesteps(t_rng, end - i)
        loss = loss_fn(state.params, latents, encoder_hidden_states, timesteps, noise_rng)
        losses = losses.at[i:end].set(loss)

    metrics = {"scalar": {"learning/eval_loss": losses}}
    return metrics, rng
