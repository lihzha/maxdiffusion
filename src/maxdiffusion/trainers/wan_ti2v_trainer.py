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
    Pre-encoded records must contain:
      latents              float32  (C, F_lat, H_lat, W_lat)  channels-first
      encoder_hidden_states  float32  (512, 4096)
    For eval, also:
      timesteps            int64
"""

import functools

import jax
import jax.numpy as jnp
import jaxopt
import tensorflow as tf
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion.checkpointing.wan_checkpointer_ti2v_2p2 import WanCheckpointerTI2V_2_2
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
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
        shardings = self.get_data_shardings(mesh)
        shardings["timesteps"] = NamedSharding(mesh, P(*self.config.data_sharding))
        return shardings

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

        feature_description = {
            "latents": tf.io.FixedLenFeature([], tf.string),
            "encoder_hidden_states": tf.io.FixedLenFeature([], tf.string),
        }
        if not is_training:
            feature_description["timesteps"] = tf.io.FixedLenFeature([], tf.int64)

        # WAN VAE temporal compression: 4 raw frames → 1 latent frame, plus 1 anchor.
        # Matches wan_pipeline_ti2v_2p2: num_latent_frames = 1 + num_frames // 4.
        window_size = 1 + config.num_frames // 4

        def _random_window(latents):
            """Randomly slice a window_size window along the temporal axis (dim 1).

            TFRecords may store clips longer than window_size. Each training step
            draws a fresh random start so the model sees different sub-clips each
            epoch. Records shorter than window_size are not supported and will
            produce incorrect shapes at batch time.
            """
            f_total = tf.shape(latents)[1]
            max_start = f_total - window_size
            start = tf.random.uniform((), minval=0, maxval=max_start + 1, dtype=tf.int32)
            return latents[:, start : start + window_size, :, :]

        def prepare_sample_train(features):
            latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
            encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
            latents = _random_window(latents)
            return {"latents": latents, "encoder_hidden_states": encoder_hidden_states}

        def prepare_sample_eval(features):
            latents = tf.io.parse_tensor(features["latents"], out_type=tf.float32)
            encoder_hidden_states = tf.io.parse_tensor(features["encoder_hidden_states"], out_type=tf.float32)
            # Fixed window from the start for deterministic eval.
            latents = latents[:, :window_size, :, :]
            timesteps = features["timesteps"]
            return {"latents": latents, "encoder_hidden_states": encoder_hidden_states, "timesteps": timesteps}

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


# ── Training step ─────────────────────────────────────────────────────────────


def ti2v_train_step(state, data, rng, scheduler_state, scheduler, config, n_hist):
    _, new_rng, timestep_rng, dropout_rng = jax.random.split(rng, num=4)

    for k, v in data.items():
        if hasattr(v, "shape"):
            data[k] = v[: config.global_batch_size_to_train_on]

    def loss_fn(params):
        model = nnx.merge(state.graphdef, params, state.rest_of_state)
        # latents: (B, C, F_lat, H_lat, W_lat) channels-first, as stored in TFRecords
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

    bs = len(data["latents"])
    single_batch_size = config.global_batch_size_to_train_on
    losses = jnp.zeros(bs)
    for i in range(0, bs, single_batch_size):
        end = min(i + single_batch_size, bs)
        latents = data["latents"][i:end].astype(config.weights_dtype)
        encoder_hidden_states = data["encoder_hidden_states"][i:end].astype(config.weights_dtype)
        timesteps = data["timesteps"][i:end].astype("int64")
        _, new_rng = jax.random.split(rng, num=2)
        loss = loss_fn(state.params, latents, encoder_hidden_states, timesteps, new_rng)
        losses = losses.at[i:end].set(loss)

    metrics = {"scalar": {"learning/eval_loss": losses}}
    return metrics, new_rng
