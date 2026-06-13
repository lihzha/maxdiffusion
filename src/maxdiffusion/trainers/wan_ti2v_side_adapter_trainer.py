"""WAN TI2V side-adapter trainer.

This trainer matches the side-adapter experiment in ``../Wan2.2``:

* load the Wan2.2 TI2V 5B transformer from the pretrained checkpoint,
* keep all backbone transformer parameters frozen,
* train only the action token encoder and side-adapter residual blocks, and
* optimize a full latent rollout MSE with the first latent frame pinned.

The dataset intentionally stores only cached VAE latents and actions. A single
null prompt embedding is computed once from the loaded T5 encoder and reused for
all samples, avoiding per-example text embeddings in TFRecords.
"""

from __future__ import annotations

import datetime
import functools
import os
from typing import Any

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging, max_utils
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.wan.side_adapter_wan import (
    NNXWanSideAdapterStack,
    adapter_param_count,
    apply_first_frame_pin,
    build_rollout_sigmas,
    rollout_timesteps_from_sigmas,
    wan_side_adapter_forward,
    _build_per_token_timestep,
    _dtype,
)
from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import load_next_batch


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef
    rest_of_state: nnx.State
    transformer_graphdef: nnx.GraphDef
    transformer_params: nnx.State
    transformer_rest: nnx.State
    null_context: jax.Array


def _apply_actual_sharding_for_tpu(state, computed_shardings):
    """Prefer real TPU shardings for arrays already materialized on TPU."""

    def _use_actual(x, computed):
        if not isinstance(x, jax.Array):
            return computed
        if any(d.platform == "cpu" for d in x.devices()):
            return computed
        actual = getattr(x, "sharding", None)
        return actual if isinstance(actual, jax.sharding.NamedSharding) else computed

    return jax.tree.map(_use_actual, state, computed_shardings)


def _to_target_if_cpu(state, target_shardings):
    def _place(x, target):
        if not isinstance(x, jax.Array):
            return x
        if not any(d.platform == "cpu" for d in x.devices()):
            return x
        full = np.asarray(x.addressable_data(0))
        return jax.make_array_from_callback(x.shape, target, lambda idx: full[idx])

    return jax.tree.map(_place, state, target_shardings)


def _rollout_loss(
    params,
    state: TrainState,
    data: dict,
    rng: jax.Array,
    config,
    scheduler: FlaxFlowMatchScheduler,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    noise_rng, dropout_rng = jax.random.split(rng)
    weights_dtype = _dtype(config.weights_dtype)
    bsz = config.global_batch_size_to_train_on

    adapters = nnx.merge(state.graphdef, params, state.rest_of_state)
    transformer = nnx.merge(state.transformer_graphdef, state.transformer_params, state.transformer_rest)

    z_i0 = data["z_i0"][:bsz].astype(weights_dtype)
    z_video = data["z_video"][:bsz].astype(weights_dtype)
    actions = data["actions"][:bsz].astype(weights_dtype)

    b, _, f_lat, h_lat, w_lat = z_video.shape
    sigmas = build_rollout_sigmas(
        config.side_adapter_sampling_steps,
        config.flow_shift,
        scheduler.config.sigma_min,
        scheduler.config.sigma_max,
    )
    timesteps = rollout_timesteps_from_sigmas(sigmas, scheduler.config.num_train_timesteps)
    null_context = jnp.broadcast_to(
        state.null_context.astype(weights_dtype),
        (b, state.null_context.shape[1], state.null_context.shape[2]),
    )

    if config.side_adapter_noise_mode != "fresh":
        raise ValueError("Only side_adapter_noise_mode='fresh' is supported for TPU training")
    z = jax.random.normal(noise_rng, z_video.shape, dtype=z_video.dtype)
    z = apply_first_frame_pin(z, z_i0)

    for i in range(config.side_adapter_sampling_steps):
        step_t = jnp.broadcast_to(timesteps[i], (b,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
        v_cond = wan_side_adapter_forward(
            transformer,
            adapters,
            hidden_states=z,
            timestep=timestep_2d,
            encoder_hidden_states=null_context,
            actions=actions,
            deterministic=False,
            rngs=nnx.Rngs(dropout=dropout_rng),
        )
        if abs(config.side_adapter_guide_scale - 1.0) > 1e-6:
            v_uncond = transformer(
                hidden_states=z,
                timestep=timestep_2d,
                encoder_hidden_states=null_context,
                deterministic=True,
            )
            v = v_uncond + config.side_adapter_guide_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = apply_first_frame_pin(z + (sigmas[i + 1] - sigmas[i]).astype(z.dtype) * v, z_i0)

    diff = z - z_video
    loss = jnp.mean(diff**2)
    aux = {
        "latent_mse": loss,
        "z_pred_std": jnp.std(z.astype(jnp.float32)),
        "z_target_std": jnp.std(z_video.astype(jnp.float32)),
        "z_init_anchor_mse": jnp.mean((z[:, :, :1].astype(jnp.float32) - z_i0[:, :, :1].astype(jnp.float32)) ** 2),
    }
    return loss, aux


def _train_step(state: TrainState, data: dict, rng: jax.Array, scheduler, config):
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        return _rollout_loss(params, state, data, loss_rng, config, scheduler)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    max_abs_grad = jax.tree_util.tree_reduce(
        lambda m, arr: jnp.maximum(m, jnp.max(jnp.abs(arr))), grads, initializer=-1.0
    )
    state = state.apply_gradients(grads=grads)
    metrics = {
        "scalar": {
            "learning/loss": loss,
            "learning/latent_mse": aux["latent_mse"],
            "learning/grad_norm": grad_norm,
            "learning/max_abs_grad": max_abs_grad,
            "learning/z_pred_std": aux["z_pred_std"],
            "learning/z_target_std": aux["z_target_std"],
            "learning/z_init_anchor_mse": aux["z_init_anchor_mse"],
        },
        "scalars": {},
    }
    return state, metrics, rng


def _eval_step(state: TrainState, data: dict, rng: jax.Array, scheduler, config):
    losses = jnp.zeros((config.global_batch_size_to_train_on,), dtype=jnp.float32)
    loss, aux = _rollout_loss(state.params, state, data, rng, config, scheduler)
    losses = losses.at[:].set(loss)
    metrics = {
        "scalar": {
            "learning/eval_loss": losses,
            "learning/eval_z_pred_std": aux["z_pred_std"],
            "learning/eval_z_target_std": aux["z_target_std"],
        },
        "scalars": {},
    }
    return metrics, rng


class WanTI2VSideAdapterTrainer:
    """Self-contained side-adapter trainer."""

    def __init__(self, config):
        self.config = config

    def _create_scheduler(self):
        scheduler = FlaxFlowMatchScheduler(
            dtype=jnp.float32,
            shift=self.config.flow_shift,
            sigma_min=getattr(self.config, "flow_sigma_min", 0.0),
            sigma_max=getattr(self.config, "flow_sigma_max", 1.0),
        )
        state = scheduler.create_state()
        state = scheduler.set_timesteps(
            state,
            num_inference_steps=self.config.side_adapter_sampling_steps,
            training=False,
            shift=self.config.flow_shift,
        )
        return scheduler, state

    def _load_wan_pipeline(self) -> WanPipelineTI2V_2_2:
        if self.config.scan_layers:
            raise ValueError("WanTI2VSideAdapterTrainer requires scan_layers=False")
        max_logging.log("[wan_side_adapter] loading WAN TI2V pipeline from pretrained")
        with nn_partitioning.axis_rules(self.config.logical_axis_rules):
            return WanPipelineTI2V_2_2.from_pretrained(self.config)

    def _build_adapters(self, transformer) -> NNXWanSideAdapterStack:
        model_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        return NNXWanSideAdapterStack(
            rngs=nnx.Rngs(jax.random.key(self.config.seed)),
            num_layers=transformer.config.num_layers,
            model_dim=model_dim,
            action_dim=self.config.action_dim,
            action_len=self.config.action_len,
            action_repr=self.config.action_repr,
            action_tokens=self.config.action_tokens,
            action_hidden=self.config.action_hidden,
            action_heads=self.config.action_heads,
            side_adapter_layers=self.config.side_adapter_layers,
            side_adapter_hidden=self.config.side_adapter_hidden,
            side_adapter_heads=self.config.side_adapter_heads,
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
            precision=getattr(jax.lax.Precision, self.config.precision),
        )

    def _compute_null_context(self, pipeline, mesh):
        max_len = getattr(self.config, "wan_max_sequence_length", 512)
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt=[""],
            negative_prompt=[""],
            num_videos_per_prompt=1,
            max_sequence_length=max_len,
        )
        prompt_embeds = prompt_embeds.astype(_dtype(self.config.activations_dtype))
        null_context = jnp.asarray(prompt_embeds)
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            null_context = jax.device_put(null_context, NamedSharding(mesh, P()))
        return null_context

    def _load_dataset(self, mesh, is_training: bool, seed: int | None = None):
        config = self.config
        if config.dataset_type != "tfrecord" or not config.cache_latents_text_encoder_outputs:
            raise ValueError(
                "WanTI2VSideAdapterTrainer requires dataset_type='tfrecord' and "
                "cache_latents_text_encoder_outputs=True."
            )
        data_dir = config.train_data_dir if is_training else config.eval_data_dir
        if not data_dir:
            raise ValueError("train_data_dir/eval_data_dir must point to side-adapter TFRecord shards")

        feature_description = {
            "z_i0": tf.io.FixedLenFeature([], tf.string),
            "z_video": tf.io.FixedLenFeature([], tf.string),
            "actions": tf.io.FixedLenFeature([], tf.string),
        }
        c = int(config.latent_channels)
        f = int(config.latent_frames)
        h = int(config.latent_height)
        w = int(config.latent_width)
        action_len = int(config.action_len)
        action_dim = int(config.action_dim)

        def prepare_sample(features):
            z_i0 = tf.reshape(tf.io.decode_raw(features["z_i0"], tf.float16), [c, 1, h, w])
            z_video = tf.reshape(tf.io.decode_raw(features["z_video"], tf.float16), [c, f, h, w])
            actions = tf.reshape(tf.io.decode_raw(features["actions"], tf.float32), [action_len, action_dim])
            z_i0 = tf.cast(z_i0, tf.float32)
            z_video = tf.cast(z_video, tf.float32)
            actions = tf.cast(actions, tf.float32)
            return {"z_i0": z_i0, "z_video": z_video, "actions": actions}

        return make_data_iterator(
            config,
            jax.process_index(),
            jax.process_count(),
            mesh,
            config.global_batch_size_to_load,
            feature_description=feature_description,
            prepare_sample_fn=prepare_sample,
            is_training=is_training,
            seed=seed if seed is not None else config.seed,
        )

    def _build_optimizer(self, num_steps: int):
        schedule_steps = (
            self.config.learning_rate_schedule_steps
            if self.config.learning_rate_schedule_steps > 0
            else num_steps
        )
        lr_schedule = max_utils.create_learning_rate_schedule(
            self.config.learning_rate,
            schedule_steps,
            self.config.warmup_steps_fraction,
            num_steps,
        )
        tx = max_utils.create_optimizer(self.config, lr_schedule)
        return tx, lr_schedule

    def _build_checkpoint_manager(self, ckpt_dir: str) -> ocp.CheckpointManager:
        tf.io.gfile.makedirs(ckpt_dir)
        keep_period = getattr(self.config, "checkpoint_keep_period", -1) or None
        return ocp.CheckpointManager(
            ckpt_dir,
            item_names=("params", "opt_state", "step"),
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "step": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(
                create=True,
                max_to_keep=3,
                enable_async_checkpointing=True,
                keep_period=keep_period,
            ),
        )

    def _save_checkpoint(self, manager: ocp.CheckpointManager, step: int, state: TrainState):
        if jax.process_index() == 0:
            max_logging.log(f"[wan_side_adapter] saving adapter checkpoint at step {step}")
        manager.save(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(state.params),
                opt_state=ocp.args.StandardSave(state.opt_state),
                step=ocp.args.JsonSave({"step": int(step)}),
            ),
        )

    def _maybe_restore(self, manager: ocp.CheckpointManager, state: TrainState) -> tuple[TrainState, int]:
        latest = manager.latest_step()
        if latest is None:
            return state, 0
        max_logging.log(f"[wan_side_adapter] restoring adapter checkpoint at step {latest}")
        restored = manager.restore(
            latest,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(state.params),
                opt_state=ocp.args.StandardRestore(state.opt_state),
                step=ocp.args.JsonRestore(),
            ),
        )
        return state.replace(params=restored["params"], opt_state=restored["opt_state"]), int(restored["step"]["step"])

    def _shard_state(self, mesh, state: TrainState) -> tuple[TrainState, Any]:
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            state_spec = nnx.get_partition_spec(state)
            state_shardings = nnx.get_named_sharding(state, mesh)
            state_shardings = _apply_actual_sharding_for_tpu(state, state_shardings)
            state = _to_target_if_cpu(state, state_shardings)
            state = jax.lax.with_sharding_constraint(state, state_spec)
        return state, state_shardings

    def _data_shardings(self, mesh) -> dict:
        sharding = NamedSharding(mesh, P(*self.config.data_sharding))
        return {"z_i0": sharding, "z_video": sharding, "actions": sharding}

    def start_training(self):
        config = self.config
        pipeline = self._load_wan_pipeline()
        mesh = pipeline.mesh

        null_context = self._compute_null_context(pipeline, mesh)
        if hasattr(pipeline, "vae"):
            del pipeline.vae
        if hasattr(pipeline, "vae_cache"):
            del pipeline.vae_cache
        if hasattr(pipeline, "text_encoder"):
            del pipeline.text_encoder
        if hasattr(pipeline, "tokenizer"):
            del pipeline.tokenizer

        adapters = self._build_adapters(pipeline.transformer)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            adapter_graphdef, adapter_params, adapter_rest = nnx.split(adapters, nnx.Param, ...)
            transformer_graphdef, transformer_params, transformer_rest = nnx.split(pipeline.transformer, nnx.Param, ...)

        if jax.process_index() == 0:
            n_adapter = adapter_param_count(adapter_params)
            n_frozen = adapter_param_count(transformer_params)
            max_logging.log(f"[wan_side_adapter] trainable adapter params: {n_adapter / 1e6:.1f}M")
            max_logging.log(f"[wan_side_adapter] frozen transformer params: {n_frozen / 1e9:.2f}B")

        tx, lr_schedule = self._build_optimizer(config.max_train_steps)
        state = TrainState.create(
            apply_fn=adapter_graphdef.apply,
            params=adapter_params,
            tx=tx,
            graphdef=adapter_graphdef,
            rest_of_state=adapter_rest,
            transformer_graphdef=transformer_graphdef,
            transformer_params=transformer_params,
            transformer_rest=transformer_rest,
            null_context=null_context,
        )
        state, state_shardings = self._shard_state(mesh, state)
        data_shardings = self._data_shardings(mesh)

        scheduler, _ = self._create_scheduler()
        ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
        ckpt_mgr = self._build_checkpoint_manager(ckpt_dir)
        state, start_step = self._maybe_restore(ckpt_mgr, state)
        if start_step:
            max_logging.log(f"[wan_side_adapter] resumed at step {start_step}")

        train_iter = self._load_dataset(mesh, is_training=True, seed=config.seed + start_step)
        p_train_step = jax.jit(
            functools.partial(_train_step, scheduler=scheduler, config=config),
            in_shardings=(state_shardings, data_shardings, None),
            out_shardings=(state_shardings, None, None),
            donate_argnums=(0,),
        )
        p_eval_step = jax.jit(
            functools.partial(_eval_step, scheduler=scheduler, config=config),
            in_shardings=(state_shardings, data_shardings, None),
            out_shardings=(None, None),
        )

        if jax.process_index() == 0:
            max_logging.log("***** Running WAN TI2V side-adapter training *****")
            max_logging.log(f"  Per-device batch size: {config.per_device_batch_size}")
            max_logging.log(f"  Devices: {jax.device_count()}")
            max_logging.log(f"  Max train steps: {config.max_train_steps}")
            max_logging.log(f"  Output dir: {config.output_dir}")
            max_logging.log(f"  Sampling steps: {config.side_adapter_sampling_steps}")
            max_logging.log(f"  Guidance scale: {config.side_adapter_guide_scale}")

        wandb_run = None
        if jax.process_index() == 0 and getattr(config, "wandb_project", ""):
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=getattr(config, "wandb_entity", None) or None,
                name=config.run_name or None,
                settings=wandb.Settings(start_method="thread"),
            )

        rng = jax.random.key(config.seed + 1)
        recent_loss: list[float] = []
        recent_grad: list[float] = []
        last_log_time = datetime.datetime.now()
        batch = load_next_batch(train_iter, None, config)

        for step in range(start_step, config.max_train_steps):
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                state, metrics, rng = p_train_step(state, batch, rng)
                metrics["scalar"]["learning/loss"].block_until_ready()

            recent_loss.append(float(metrics["scalar"]["learning/loss"]))
            recent_grad.append(float(metrics["scalar"]["learning/grad_norm"]))

            if (step + 1) % config.log_period == 0 and jax.process_index() == 0:
                now = datetime.datetime.now()
                avg_loss = sum(recent_loss) / len(recent_loss)
                avg_grad = sum(recent_grad) / len(recent_grad)
                sps = len(recent_loss) / max(1e-6, (now - last_log_time).total_seconds())
                lr = float(lr_schedule(step))
                max_logging.log(
                    f"step {step + 1}/{config.max_train_steps} "
                    f"loss={avg_loss:.6f} grad_norm={avg_grad:.3f} "
                    f"lr={lr:.2e} steps/s={sps:.3f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/grad_norm": avg_grad,
                            "train/lr": lr,
                            "train/steps_per_sec": sps,
                        },
                        step=step + 1,
                    )
                recent_loss.clear()
                recent_grad.clear()
                last_log_time = now

            if config.eval_every > 0 and config.eval_data_dir and (step + 1) % config.eval_every == 0:
                self._run_eval(mesh, p_eval_step, state, data_shardings, step + 1, rng, wandb_run)

            if config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
                self._save_checkpoint(ckpt_mgr, step + 1, state)

            batch = load_next_batch(train_iter, batch, config)

        if config.save_final_checkpoint:
            self._save_checkpoint(ckpt_mgr, config.max_train_steps, state)
        ckpt_mgr.wait_until_finished()
        if wandb_run is not None:
            wandb_run.finish()

    def _run_eval(self, mesh, p_eval_step, state, data_shardings, step: int, rng: jax.Array, wandb_run):
        eval_iter = self._load_dataset(mesh, is_training=False)
        losses = []
        for _ in range(max(1, int(getattr(self.config, "eval_max_batches", 1)))):
            try:
                batch = load_next_batch(eval_iter, None, self.config)
            except StopIteration:
                break
            rng, sub = jax.random.split(rng)
            with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
                metrics, _ = p_eval_step(state, batch, sub)
                metrics["scalar"]["learning/eval_loss"].block_until_ready()
            losses.extend(np.asarray(jax.device_get(metrics["scalar"]["learning/eval_loss"])).reshape(-1).tolist())
        if losses and jax.process_index() == 0:
            mean_loss = float(np.mean(losses))
            max_logging.log(f"[wan_side_adapter] eval step={step} loss={mean_loss:.6f}")
            if wandb_run is not None:
                wandb_run.log({"eval/loss": mean_loss}, step=step)
