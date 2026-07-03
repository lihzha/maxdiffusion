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
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    seq_len = F_lat * tokens_per_frame
    n_hist_tokens = n_hist * tokens_per_frame
    is_future = jnp.arange(seq_len)[None, :] >= n_hist_tokens
    return jnp.where(is_future, timesteps[:, None], 0)


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

    def load_dataset(self, mesh, pipeline=None, is_training=True, seed=None):
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

        single_camera = getattr(config, "single_camera", False)

        def prepare_sample_train(features):
            cam0 = tf.cast(tf.io.parse_tensor(features["latent_cam0"], out_type=tf.float16), tf.float32)
            cam1 = tf.cast(tf.io.parse_tensor(features["latent_cam1"], out_type=tf.float16), tf.float32)
            cam2 = tf.cast(tf.io.parse_tensor(features["latent_cam2"], out_type=tf.float16), tf.float32)
            if single_camera:
                cam_idx = tf.random.uniform((), 0, 3, dtype=tf.int32)
                latent = tf.switch_case(cam_idx, [lambda: cam0, lambda: cam1, lambda: cam2])
            else:
                # Concat cameras along H (axis 2): (F_lat, C, H_lat*3, W_lat) — matches ctrl_world.
                latent = tf.concat([cam0, cam1, cam2], axis=2)

            # Random temporal window on axis 0 (time), then transpose to channels-first.
            f_total = tf.shape(latent)[0]
            max_start = tf.maximum(0, f_total - window_size)
            start = tf.random.uniform((), 0, max_start + 1, dtype=tf.int32)
            latent = latent[start : start + window_size]
            latent = tf.transpose(latent, [1, 0, 2, 3])        # (C, window_size, H_lat[*3], W_lat)

            encoder_hidden_states = tf.cast(
                tf.io.parse_tensor(features["text_embed"], out_type=tf.float16), tf.float32
            )  # (512, 4096)
            return {"latents": latent, "encoder_hidden_states": encoder_hidden_states}

        def prepare_sample_eval(features):
            cam0 = tf.cast(tf.io.parse_tensor(features["latent_cam0"], out_type=tf.float16), tf.float32)
            cam1 = tf.cast(tf.io.parse_tensor(features["latent_cam1"], out_type=tf.float16), tf.float32)
            cam2 = tf.cast(tf.io.parse_tensor(features["latent_cam2"], out_type=tf.float16), tf.float32)
            if single_camera:
                latent = cam0  # use cam0 (wrist) for eval, consistent with existing convention
            else:
                latent = tf.concat([cam0, cam1, cam2], axis=2)     # (F_lat, C, H_lat*3, W_lat)
            latent = latent[:window_size]
            latent = tf.transpose(latent, [1, 0, 2, 3])        # (C, window_size, H_lat[*3], W_lat)
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
            seed=seed,
        )

    # ── Train / eval steps ───────────────────────────────────────────────────

    def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
        n_hist = getattr(self.config, "num_privileged_frames", 0) + 1
        distill = getattr(self.config, "distill", False)
        distill_bptt = getattr(self.config, "distill_bptt", False)
        oracle_noise_offset = getattr(self.config, "oracle_noise_offset", -1)

        # CFG training: encode the empty prompt once with UMT5 to use as the
        # null embedding for text dropout. Matches the negative_prompt=""
        # embedding CFG contrasts against at inference. 512 = TFRecord
        # text_embed sequence length.
        cfg_dropout_prob = float(getattr(self.config, "cfg_dropout_prob", 0.0))
        null_prompt_embeds = None
        if cfg_dropout_prob > 0.0:
            embeds = pipeline._get_t5_prompt_embeds(prompt="", max_sequence_length=512)
            null_prompt_embeds = jnp.array(embeds.detach().float().numpy()[0], dtype=jnp.float32)

        return jax.jit(
            functools.partial(
                ti2v_train_step,
                scheduler=pipeline.scheduler,
                config=self.config,
                n_hist=n_hist,
                distill=distill,
                distill_bptt=distill_bptt,
                oracle_noise_offset=oracle_noise_offset,
                cfg_dropout_prob=cfg_dropout_prob,
                null_prompt_embeds=null_prompt_embeds,
            ),
            in_shardings=(state_shardings, data_shardings, None, None),
            out_shardings=(state_shardings, None, None, None),
            donate_argnums=(0,),
        )

    def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
        n_hist = getattr(self.config, "num_privileged_frames", 0) + 1
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


def ti2v_train_step(
    state, data, rng, scheduler_state, scheduler, config, n_hist,
    distill=False, distill_bptt=False, oracle_noise_offset=-1, cfg_dropout_prob=0.0, null_prompt_embeds=None,
):
    _, new_rng, timestep_rng, dropout_rng, oracle_rng, gen_rng, gen_noise_rng, cfg_rng = jax.random.split(rng, num=8)

    for k, v in data.items():
        if hasattr(v, "shape"):
            data[k] = v[: config.global_batch_size_to_train_on]

    # latents: (B, C, F_lat, H_lat, W_lat) channels-first
    latents = data["latents"].astype(config.weights_dtype)
    encoder_hidden_states = data["encoder_hidden_states"].astype(config.weights_dtype)

    b, _, F_lat, H_lat, W_lat = latents.shape

    # CFG training: per-sample, replace the text embedding with the null
    # (empty-prompt) embedding. In the distill path this conditions both the
    # on-policy rollout and the loss forward pass, keeping them consistent.
    if cfg_dropout_prob > 0.0 and null_prompt_embeds is not None:
        drop = jax.random.bernoulli(cfg_rng, cfg_dropout_prob, (b,))
        encoder_hidden_states = jnp.where(
            drop[:, None, None],
            null_prompt_embeds[None].astype(encoder_hidden_states.dtype),
            encoder_hidden_states,
        )
    future_latents = latents[:, :, n_hist:]

    if distill:
        # ── On-policy path: Euler rollout collecting all intermediate states ──
        F_future = F_lat - n_hist
        num_train_t = scheduler.config.num_train_timesteps
        _cfg_gen_steps = config.num_online_gen_steps
        num_gen_steps = config.num_inference_steps if _cfg_gen_steps < 0 else _cfg_gen_steps

        # Discretized rollout schedule: T_max → 0 in num_gen_steps Euler steps,
        # linearly spaced in t so uniform k gives uniform training timestep.
        t_uniform = jnp.linspace(1.0, 0.0, num_gen_steps + 1)
        rollout_ts = (t_uniform * (num_train_t - 1)).astype(jnp.int32)

        def _sigma(t_int):
            t_n = t_int.astype(jnp.float32) / (num_train_t - 1)
            return (1.0 - t_n) * scheduler.config.sigma_min + t_n * scheduler.config.sigma_max

        gen_init = jax.random.normal(gen_noise_rng, future_latents.shape, dtype=future_latents.dtype)

        if not distill_bptt:
            # Forward-only rollout collecting intermediate states.  BPTT mode
            # skips this: its trajectory is generated inside the loss so it
            # can be differentiated through.
            #
            # Build roll_model once outside the loop: nnx.merge is a Python/trace-time
            # operation, but keeping it inside a scan body would re-run it on
            # every JAX re-trace.  Moving it out is cleaner and saves trace overhead.
            roll_model = nnx.merge(state.graphdef, state.params, state.rest_of_state)

            def rollout_scan_body(lat, step_idx):
                t_from = rollout_ts[step_idx]
                t_to   = rollout_ts[step_idx + 1]
                sig_from = _sigma(t_from)
                sig_to   = _sigma(t_to)

                roll_input = jnp.concatenate([latents[:, :, :n_hist], lat], axis=2)
                roll_ts_2d = _build_per_token_timestep(
                    jnp.broadcast_to(t_from, (b,)), F_lat, H_lat, W_lat, n_hist
                )
                with jax.named_scope("online_gen_rollout_step"):
                    v_pred = roll_model(
                        hidden_states=roll_input,
                        timestep=roll_ts_2d,
                        encoder_hidden_states=encoder_hidden_states,
                        deterministic=True,
                    )
                v_future = v_pred[:, :, n_hist:]

                # Euler step: x_{t_to} = x_{t_from} + (σ_{t_to} - σ_{t_from}) * v
                # Cast back to lat.dtype: _sigma computes in float32, which would
                # otherwise promote new_lat and break the carry type check.
                new_lat = (lat + (sig_to - sig_from) * v_future).astype(lat.dtype)
                # Output lat (state before this step) = on-policy latent at rollout_ts[step_idx].
                return new_lat, lat

            # Collect every intermediate latent state via scan:
            # all_gen_latents[k] = on-policy noisy latent at rollout_ts[k], k ∈ [0, num_gen_steps).
            _, all_gen_latents = jax.lax.scan(rollout_scan_body, gen_init, jnp.arange(num_gen_steps))
            # shape: (num_gen_steps, B, C, F_future, H_lat, W_lat)

    else:
        # ── Off-policy path: single-step noising ─────────────────────────────
        timesteps = scheduler.sample_timesteps(timestep_rng, b)
        noise = jax.random.normal(new_rng, future_latents.shape, dtype=future_latents.dtype)
        noisy_future, target_future, training_weight = scheduler.apply_flow_match(
            noise, future_latents, timesteps
        )
        noisy_latents = jnp.concatenate([latents[:, :, :n_hist], noisy_future], axis=2)

        # Per-token timestep: history → 0, future → t.
        timestep_2d = _build_per_token_timestep(timesteps, F_lat, H_lat, W_lat, n_hist)
        timestep_2d = jax.lax.with_sharding_constraint(timestep_2d, P(("data", "fsdp", "context"), None))

    if distill and distill_bptt:
        # ── Option B: exact BPTT through the on-policy Euler rollout ─────────
        # The rollout runs inside the differentiated function, so each step's
        # velocity loss backpropagates through every earlier Euler update
        # (pathwise credit assignment through the dynamics), not just its own
        # forward pass.  The velocity target (x_k - x0)/σ_k is NOT
        # stop-gradiented: it depends on the on-policy state, so its gradient
        # also steers earlier steps — this is the exact gradient of the same
        # objective the per-step path optimizes.
        #
        # Memory: jax.checkpoint on the step body rematerializes transformer
        # activations in the backward sweep, so only the (K, ...) latent
        # carry stack is stored across steps.  FLOPs match the per-step path
        # (K fwd on the forward sweep + K fwd + K bwd on the backward sweep).
        q_weights = jnp.arange(num_gen_steps, 0, -1, dtype=jnp.float32)
        q_weights = q_weights / q_weights.sum()

        def bptt_loss_fn(params):
            model = nnx.merge(state.graphdef, params, state.rest_of_state)

            def step_body(carry, ts_pair):
                lat, rng_c = carry
                t_from, t_to = ts_pair
                sig_from = _sigma(t_from)
                sig_to = _sigma(t_to)
                rng_c, step_rng = jax.random.split(rng_c)

                roll_input = jnp.concatenate([latents[:, :, :n_hist], lat], axis=2)
                ts_2d = _build_per_token_timestep(
                    jnp.broadcast_to(t_from, (b,)), F_lat, H_lat, W_lat, n_hist
                )
                ts_2d = jax.lax.with_sharding_constraint(ts_2d, P(("data", "fsdp", "context"), None))
                with jax.named_scope("bptt_rollout_step"):
                    v_pred = model(
                        hidden_states=roll_input,
                        timestep=ts_2d,
                        encoder_hidden_states=encoder_hidden_states,
                        deterministic=False,
                        rngs=nnx.Rngs(dropout=step_rng),
                    )
                v_future = v_pred[:, :, n_hist:]

                target_k = (lat - future_latents) / sig_from.astype(lat.dtype)
                loss_k = jnp.mean((target_k - v_future) ** 2).astype(jnp.float32)

                new_lat = (lat + (sig_to - sig_from) * v_future).astype(lat.dtype)
                return (new_lat, rng_c), loss_k

            # prevent_cse=False is safe under scan and avoids slowdown.
            step_body = jax.checkpoint(step_body, prevent_cse=False)
            ts_pairs = (rollout_ts[:num_gen_steps], rollout_ts[1 : num_gen_steps + 1])
            _, step_losses = jax.lax.scan(step_body, (gen_init, dropout_rng), ts_pairs)
            return jnp.sum(q_weights * step_losses)

        grad_fn = nnx.value_and_grad(bptt_loss_fn)
        loss, grads = grad_fn(state.params)
    elif distill:
        # ── Per-timestep grad accumulation over every rollout timestep ────────
        # all_gen_latents[k]: on-policy latent at timestep rollout_ts[k].
        # GT velocity: target = (x_t - x_0) / σ_t (flow-matching parameterisation).
        #
        # value_and_grad runs INSIDE the scan body so each step's forward+backward
        # completes (and frees its activations) within the iteration.  Scanning the
        # forward inside a single loss_fn instead would force XLA to stack the
        # layer-scan's offloaded hidden-state residuals across all K steps for the
        # outer backward — a (K, n_layers, B, seq, dim) host tensor whose bitcast
        # to offload memory fails on TPU.  Same FLOPs either way (K fwd + K bwd).
        def step_loss_fn(params, gen_t_k, ts_k_scalar, step_rng):
            model = nnx.merge(state.graphdef, params, state.rest_of_state)
            ts_k = jnp.broadcast_to(ts_k_scalar, (b,))
            sigma_k = _sigma(ts_k)[:, None, None, None, None].astype(gen_t_k.dtype)
            target_k = (gen_t_k - future_latents) / sigma_k
            noisy_k = jnp.concatenate([latents[:, :, :n_hist], gen_t_k], axis=2)
            ts_2d_k = _build_per_token_timestep(ts_k, F_lat, H_lat, W_lat, n_hist)
            ts_2d_k = jax.lax.with_sharding_constraint(ts_2d_k, P(("data", "fsdp", "context"), None))
            with jax.named_scope("forward_pass"):
                pred_k = model(
                    hidden_states=noisy_k,
                    timestep=ts_2d_k,
                    encoder_hidden_states=encoder_hidden_states,
                    deterministic=False,
                    rngs=nnx.Rngs(dropout=step_rng),
                )
            diff_k = target_k - pred_k[:, :, n_hist:]
            return jnp.mean(diff_k ** 2)

        # jax.value_and_grad, not nnx's: nnx's wrapper extracts graph nodes from
        # state.params, which fails inside the scan trace ("different trace level").
        # Plain jax treats the nnx.State as an ordinary pytree, which is all we need.
        step_grad_fn = jax.value_and_grad(step_loss_fn)

        # Weight step k by remaining future steps (K-k), normalised to sum to 1.
        # Mirrors Q = Σ_{j≥k} r_j: earlier (high-noise) steps carry more credit.
        q_weights = jnp.arange(num_gen_steps, 0, -1, dtype=jnp.float32)
        q_weights = q_weights / q_weights.sum()

        def grad_scan_body(carry, inputs):
            rng_c, loss_acc, grads_acc = carry
            gen_t_k, ts_k_scalar, w_k = inputs
            rng_c, step_rng = jax.random.split(rng_c)
            loss_k, grads_k = step_grad_fn(state.params, gen_t_k, ts_k_scalar, step_rng)
            loss_acc = loss_acc + w_k * loss_k.astype(jnp.float32)
            # Accumulate in float32: per-step weights are small (~1/K) and bf16
            # accumulation over K steps would lose precision.
            grads_acc = jax.tree_util.tree_map(
                lambda a, g: a + w_k * g.astype(jnp.float32), grads_acc, grads_k
            )
            return (rng_c, loss_acc, grads_acc), None

        zero_grads = jax.tree_util.tree_map(
            lambda p: jnp.zeros(p.shape, jnp.float32), state.params
        )
        (_, loss, grads_f32), _ = jax.lax.scan(
            grad_scan_body,
            (dropout_rng, jnp.zeros((), jnp.float32), zero_grads),
            (all_gen_latents, rollout_ts[:num_gen_steps], q_weights),
        )
        grads = jax.tree_util.tree_map(
            lambda g, p: g.astype(p.dtype), grads_f32, state.params
        )
    else:
        def loss_fn(params):
            model = nnx.merge(state.graphdef, params, state.rest_of_state)
            with jax.named_scope("forward_pass"):
                model_pred = model(
                    hidden_states=noisy_latents,
                    timestep=timestep_2d,
                    encoder_hidden_states=encoder_hidden_states,
                    deterministic=False,
                    rngs=nnx.Rngs(dropout=dropout_rng),
                )

            with jax.named_scope("loss"):
                # Standard flow-matching loss against ground-truth velocity target.
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
