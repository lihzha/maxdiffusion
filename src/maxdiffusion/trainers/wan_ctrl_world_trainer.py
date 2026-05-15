"""Action-conditioned WAN (Ctrl-World style) trainer.

Initialises from the WAN 2.2 Ti2V 5B diffusers checkpoint, adds a small
NNX action encoder, and fine-tunes both jointly on robot trajectory data.

Architecture
------------
``WanCtrlWorldModel`` wraps the ``WanModel`` transformer together with
``NNXWanActionEncoder``. Action tokens ``(B, T_act, 4096)`` are prepended to
T5 text tokens before the WAN transformer's text projection — no changes to
the transformer itself.

Training objective (per-token timestep)
----------------------------------------
Uses the same per-token timestep scheme introduced by WAN Ti2V:

* History latent frames are kept **clean** (no noise added).
* Future latent frames receive flow-matching noise at a sampled global ``t``.
* A ``(B, seq_len)`` timestep array is passed to the transformer: history
  frame tokens get ``t=0``, future frame tokens get the sampled ``t``.
* The WAN model's AdaLN modulation therefore tells each block whether it is
  looking at a clean reference frame (history) or a frame being denoised
  (future), exactly as in Ti2V inference.
* MSE loss is computed **only on future latent frames**.

Conditioning
------------
Action tokens are the sole cross-attention conditioning. T5 text tokens are
mean-pooled to a single ``(B, 4096)`` vector and added inside the action
encoder (same additive fusion as SVD Ctrl-World). 5 % of samples have their
action tokens zeroed for classifier-free guidance.

Checkpointing
-------------
Combined params (transformer + action encoder) are saved with plain orbax.
Cold-start loads from the HF diffusers checkpoint via
``WanPipelineTI2V_2_2.from_pretrained``; warm restarts load the combined
orbax checkpoint.
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
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging, max_utils
from maxdiffusion.models.wan.action_encoder_wan import NNXWanActionEncoder
from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import load_next_batch


# ── Combined model ────────────────────────────────────────────────────────────


class WanCtrlWorldModel(nnx.Module):
    """WAN transformer + action encoder held together for nnx.split/merge."""

    def __init__(self, transformer, action_encoder: NNXWanActionEncoder):
        self.transformer = transformer
        self.action_encoder = action_encoder


# ── TrainState ────────────────────────────────────────────────────────────────


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef
    rest_of_state: nnx.State


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dtype(name: str | jnp.dtype) -> jnp.dtype:
    if isinstance(name, jnp.dtype):
        return name
    return {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[name]


def _apply_cfg_dropout(
    rng: jax.Array,
    action_tokens: jnp.ndarray,
    drop_prob: float,
) -> jnp.ndarray:
    """Zero action tokens for a Bernoulli-drop_prob fraction of samples."""
    b = action_tokens.shape[0]
    keep = (jax.random.uniform(rng, (b, 1, 1)) >= drop_prob).astype(action_tokens.dtype)
    return action_tokens * keep


def _build_per_token_timestep(
    timesteps: jnp.ndarray,
    F_lat: int,
    H_lat: int,
    W_lat: int,
    n_hist_lat: int,
) -> jnp.ndarray:
    """Build ``(B, seq_len)`` timestep array for per-token Ti2V training.

    History frame tokens receive ``t=0`` (treated as clean by AdaLN);
    future frame tokens receive the sampled global timestep ``t``.

    WAN patches spatially at 2×2, so tokens_per_frame = (H_lat//2)*(W_lat//2).
    """
    b = timesteps.shape[0]
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    seq_len = F_lat * tokens_per_frame
    n_hist_tokens = n_hist_lat * tokens_per_frame

    # Broadcast global t to all tokens, then zero the history prefix.
    full = jnp.broadcast_to(timesteps[:, None], (b, seq_len))
    is_future = jnp.arange(seq_len)[None, :] >= n_hist_tokens   # (1, seq_len)
    return jnp.where(is_future, full, jnp.zeros_like(full))


# ── Training step ─────────────────────────────────────────────────────────────


def _train_step(state: TrainState, data: dict, rng: jax.Array,
                scheduler_state, scheduler, config) -> tuple:
    _, noise_rng, timestep_rng, drop_rng, new_rng = jax.random.split(rng, 5)

    bsz = config.global_batch_size_to_train_on
    weights_dtype = _dtype(config.weights_dtype)
    n_hist = config.num_history_latent_frames

    def loss_fn(params):
        model: WanCtrlWorldModel = nnx.merge(state.graphdef, params, state.rest_of_state)

        latents = data["latent"][:bsz].astype(weights_dtype)        # (B,C,F_lat,H,W)
        actions = data["action"][:bsz].astype(weights_dtype)        # (B,T_raw,7)
        text_tokens = data["text_embeds"][:bsz].astype(weights_dtype)  # (B,512,4096)

        b, _, F_lat, H_lat, W_lat = latents.shape

        # Group 4 consecutive raw-frame actions per latent frame.
        # Latent frame f ← raw frames [4f, 4f+1, 4f+2, 4f+3], clamped to T_raw-1
        # so the last (partial) group is padded by repeating the final raw frame.
        act_idx = jnp.clip(
            jnp.arange(F_lat)[:, None] * 4 + jnp.arange(4)[None, :],
            0, actions.shape[1] - 1,
        )  # (F_lat, 4)
        actions_grouped = actions[:, act_idx, :]   # (B, F_lat, 4, 7)

        # ── Sample a global denoising timestep for the future frames ──────────
        timesteps = scheduler.sample_timesteps(timestep_rng, b)

        # ── Apply flow-matching noise to future frames only ───────────────────
        # History frames stay clean; future frames get noised at timestep t.
        future_latents = latents[:, :, n_hist:]
        noise = jax.random.normal(noise_rng, future_latents.shape, dtype=future_latents.dtype)
        noisy_future, target_future, training_weight = scheduler.apply_flow_match(
            noise, future_latents, timesteps
        )
        # Concatenate clean history + noisy future → full noisy input.
        noisy_latents = jnp.concatenate([latents[:, :, :n_hist], noisy_future], axis=2)

        # ── Per-token timestep: history → 0, future → t ───────────────────────
        timestep_2d = _build_per_token_timestep(timesteps, F_lat, H_lat, W_lat, n_hist)

        # ── Action conditioning ───────────────────────────────────────────────
        # The encoder aggregates 4 raw-frame actions into one token per latent
        # frame. The transformer's frame_level_cond cross-attention then lets
        # each latent frame's patches attend to its corresponding action token.
        text_pooled = text_tokens.mean(axis=1)                           # (B, 4096)
        action_tokens = model.action_encoder(actions_grouped, text_pooled)  # (B, F_lat, 4096)
        action_tokens = _apply_cfg_dropout(drop_rng, action_tokens, config.ctrl_cfg_drop_prob)

        # ── WAN transformer forward (per-token timestep triggers Ti2V path) ───
        model_pred = model.transformer(
            hidden_states=noisy_latents,
            timestep=timestep_2d,           # (B, seq_len) → per-token AdaLN
            encoder_hidden_states=action_tokens,
            deterministic=False,
            rngs=nnx.Rngs(dropout=drop_rng),
            frame_level_cond=True,
        )

        # ── Loss on future latent frames only ─────────────────────────────────
        diff = target_future - model_pred[:, :, n_hist:]
        loss = diff ** 2
        if not config.disable_training_weights:
            loss = loss * jnp.expand_dims(training_weight, (1, 2, 3, 4))
        return jnp.mean(loss)

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    new_state = state.apply_gradients(grads=grads)

    metrics = {
        "scalar": {
            "learning/loss": loss,
            "learning/grad_norm": grad_norm,
        },
        "scalars": {},
    }
    return new_state, scheduler_state, metrics, new_rng


# ── Trainer ───────────────────────────────────────────────────────────────────


class WanCtrlWorldTrainer:
    """Self-contained trainer for action-conditioned WAN video generation."""

    def __init__(self, config):
        self.config = config

    # ── Scheduler ─────────────────────────────────────────────────────────────

    def _create_scheduler(self):
        sched = FlaxFlowMatchScheduler(dtype=jnp.float32)
        state = sched.create_state()
        state = sched.set_timesteps(state, num_inference_steps=1000, training=True)
        return sched, state

    # ── Dataset ───────────────────────────────────────────────────────────────

    def _load_dataset(self, mesh, is_training: bool):
        from maxdiffusion.input_pipeline.robot.wan_ctrl_world_dataset import (
            WanCtrlWorldDroidDataset,
        )
        config = self.config
        split = "train" if is_training else "val"
        ds = WanCtrlWorldDroidDataset(
            data_dir=config.train_data_dir if is_training else config.eval_data_dir,
            stats_path=config.action_stats_path,
            n_hist=config.num_history_latent_frames,
            action_dim=config.action_dim,
            batch_size=max(1, int(jax.local_device_count() * config.per_device_batch_size)),
            split=split,
            seed=config.seed,
            shuffle=is_training,
            shard_for_training=jax.process_count() > 1,
        )
        return iter(ds)

    # ── Pipeline / model loading ───────────────────────────────────────────────

    def _load_wan_pipeline(self) -> WanPipelineTI2V_2_2:
        max_logging.log("[wan_ctrl_world] loading WAN Ti2V pipeline from pretrained")
        with nn_partitioning.axis_rules(self.config.logical_axis_rules):
            pipeline = WanPipelineTI2V_2_2.from_pretrained(self.config)
        return pipeline

    def _build_action_encoder(self) -> NNXWanActionEncoder:
        return NNXWanActionEncoder(
            rngs=nnx.Rngs(jax.random.key(self.config.seed)),
            action_dim=self.config.action_dim,
            num_actions=4,  # WAN 4× temporal compression → 4 raw frames per latent frame
            hidden_dim=self.config.wan_action_encoder_hidden_dim,
            out_dim=self.config.wan_text_dim,
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _build_checkpoint_manager(self, ckpt_dir: str) -> ocp.CheckpointManager:
        os.makedirs(ckpt_dir, exist_ok=True)
        options = ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=3,
            enable_async_checkpointing=True,
        )
        return ocp.CheckpointManager(
            ckpt_dir,
            item_names=("params", "step"),
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "step":   ocp.JsonCheckpointHandler(),
            },
            options=options,
        )

    def _save_checkpoint(self, mgr: ocp.CheckpointManager, step: int, state: TrainState):
        if jax.process_index() == 0:
            max_logging.log(f"[wan_ctrl_world] saving checkpoint at step {step}")
        mgr.save(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardSave(state.params),
                step=ocp.args.JsonSave({"step": int(step)}),
            ),
        )

    def _maybe_restore(
        self,
        mgr: ocp.CheckpointManager,
        state: TrainState,
    ) -> tuple[TrainState, int]:
        latest = mgr.latest_step()
        if latest is None:
            return state, 0
        max_logging.log(f"[wan_ctrl_world] restoring combined checkpoint at step {latest}")
        restored = mgr.restore(
            latest,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(state.params),
                step=ocp.args.JsonRestore(),
            ),
        )
        new_state = state.replace(params=restored["params"])
        return new_state, int(restored["step"]["step"])

    # ── Optimiser ─────────────────────────────────────────────────────────────

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

    # ── Sharding ──────────────────────────────────────────────────────────────

    def _shard_state(self, mesh, state: TrainState) -> tuple[TrainState, Any]:
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            state_spec = nnx.get_partition_spec(state)
            state_shardings = nnx.get_named_sharding(state, mesh)
            state = jax.lax.with_sharding_constraint(state, state_spec)
        return state, state_shardings

    def _data_shardings(self, mesh) -> dict:
        pspec = NamedSharding(mesh, P(*self.config.data_sharding))
        return {
            "latent":      pspec,
            "action":      pspec,
            "text_embeds": pspec,
        }

    # ── Main training entry point ─────────────────────────────────────────────

    def start_training(self):
        config = self.config

        # 1. Load WAN pipeline (transformer + mesh)
        pipeline = self._load_wan_pipeline()
        mesh = pipeline.mesh

        # Free VAE — we use pre-encoded latents
        if hasattr(pipeline, "vae"):
            del pipeline.vae
        if hasattr(pipeline, "vae_cache"):
            del pipeline.vae_cache

        # 2. Build combined model
        action_encoder = self._build_action_encoder()
        combined = WanCtrlWorldModel(pipeline.transformer, action_encoder)

        # 3. Split combined model into (graphdef, params, rest_of_state)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            graphdef, params, rest_of_state = nnx.split(combined, nnx.Param, ...)

        if jax.process_index() == 0:
            n_params = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))
            max_logging.log(f"[wan_ctrl_world] trainable params: {n_params / 1e6:.1f}M")

        # 4. Build optimizer and train state
        tx, lr_schedule = self._build_optimizer(config.max_train_steps)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            state = TrainState.create(
                apply_fn=graphdef.apply,
                params=params,
                tx=tx,
                graphdef=graphdef,
                rest_of_state=rest_of_state,
            )

        # Free the pre-shard param copy on TPU; GPU JAX needs it pinned.
        if config.hardware != "gpu":
            max_utils.delete_pytree(params)

        # 5. Shard state across devices
        state, state_shardings = self._shard_state(mesh, state)
        data_shardings = self._data_shardings(mesh)

        # 6. Scheduler
        scheduler, scheduler_state = self._create_scheduler()

        # 7. Checkpoint manager + possible restore
        ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
        ckpt_mgr = self._build_checkpoint_manager(ckpt_dir)
        state, start_step = self._maybe_restore(ckpt_mgr, state)
        if start_step:
            max_logging.log(f"[wan_ctrl_world] resumed at step {start_step}")

        # 8. Data iterator
        train_iter = self._load_dataset(mesh, is_training=True)

        # 9. Compile train step
        p_train_step = jax.jit(
            functools.partial(_train_step, scheduler=scheduler, config=config),
            in_shardings=(state_shardings, data_shardings, None, None),
            out_shardings=(state_shardings, None, None, None),
            donate_argnums=(0,),
        )

        # 10. Training loop
        if jax.process_index() == 0:
            max_logging.log("***** Running WAN Ctrl-World training *****")
            max_logging.log(f"  Per-device batch size: {config.per_device_batch_size}")
            max_logging.log(f"  Devices: {jax.device_count()}")
            max_logging.log(f"  Max train steps: {config.max_train_steps}")
            max_logging.log(f"  Output dir: {config.output_dir}")

        rng = jax.random.key(config.seed + 1)
        recent_loss: list[float] = []
        recent_grad: list[float] = []
        last_step_time = datetime.datetime.now()

        example_batch = next(train_iter)

        for step in range(start_step, config.max_train_steps):
            with (
                mesh,
                nn_partitioning.axis_rules(config.logical_axis_rules),
            ):
                state, scheduler_state, metrics, rng = p_train_step(
                    state, example_batch, rng, scheduler_state
                )
                metrics["scalar"]["learning/loss"].block_until_ready()

            recent_loss.append(float(metrics["scalar"]["learning/loss"]))
            recent_grad.append(float(metrics["scalar"]["learning/grad_norm"]))
            now = datetime.datetime.now()

            if (step + 1) % config.log_period == 0 and jax.process_index() == 0:
                lr = float(lr_schedule(step))
                avg_loss = sum(recent_loss) / len(recent_loss)
                avg_grad = sum(recent_grad) / len(recent_grad)
                sps = config.log_period / (now - last_step_time).total_seconds()
                max_logging.log(
                    f"step {step + 1}/{config.max_train_steps} "
                    f"loss={avg_loss:.4f} grad_norm={avg_grad:.3f} "
                    f"lr={lr:.2e} steps/s={sps:.2f}"
                )
                recent_loss.clear()
                recent_grad.clear()
                last_step_time = now

            if (
                config.eval_every > 0
                and (step + 1) % config.eval_every == 0
            ):
                self._run_eval(mesh, train_iter, state, state_shardings,
                               data_shardings, scheduler, scheduler_state, step + 1, rng)

            if (
                config.checkpoint_every > 0
                and (step + 1) % config.checkpoint_every == 0
            ):
                self._save_checkpoint(ckpt_mgr, step + 1, state)

            example_batch = next(train_iter)

        if config.save_final_checkpoint:
            self._save_checkpoint(ckpt_mgr, config.max_train_steps, state)
        ckpt_mgr.wait_until_finished()

    # ── Eval ──────────────────────────────────────────────────────────────────

    def _run_eval(
        self,
        mesh,
        train_iter,          # reuse training iterator for a quick eval sample
        state: TrainState,
        state_shardings,
        data_shardings,
        scheduler,
        scheduler_state,
        step: int,
        rng: jax.Array,
    ):
        config = self.config
        if not config.eval_data_dir:
            max_logging.log("[wan_ctrl_world] eval_every>0 but eval_data_dir not set; skipping")
            return

        eval_iter = self._load_dataset(mesh, is_training=False)

        p_eval_step = jax.jit(
            functools.partial(_eval_step, scheduler=scheduler, config=config),
            in_shardings=(state_shardings, data_shardings, None, None),
            out_shardings=None,
        )

        losses: list[float] = []
        for _ in range(max(1, int(getattr(config, "eval_max_batches", 50)))):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            rng, sub = jax.random.split(rng)
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                loss = p_eval_step(state, batch, sub, scheduler_state)
                loss.block_until_ready()
            losses.append(float(loss))

        if losses and jax.process_index() == 0:
            max_logging.log(
                f"[wan_ctrl_world] eval step={step} batches={len(losses)} "
                f"mean_loss={sum(losses)/len(losses):.4f}"
            )


def _eval_step(state: TrainState, data: dict, rng: jax.Array,
               scheduler_state, scheduler, config) -> jax.Array:
    """Eval-only forward pass — no gradient computation, same per-token scheme."""
    _, noise_rng, timestep_rng = jax.random.split(rng, 3)
    bsz = config.global_batch_size_to_train_on
    weights_dtype = _dtype(config.weights_dtype)
    n_hist = config.num_history_latent_frames

    model: WanCtrlWorldModel = nnx.merge(state.graphdef, state.params, state.rest_of_state)

    latents = data["latent"][:bsz].astype(weights_dtype)
    actions = data["action"][:bsz].astype(weights_dtype)
    text_tokens = data["text_embeds"][:bsz].astype(weights_dtype)

    b, _, F_lat, H_lat, W_lat = latents.shape

    act_idx = jnp.clip(
        jnp.arange(F_lat)[:, None] * 4 + jnp.arange(4)[None, :],
        0, actions.shape[1] - 1,
    )  # (F_lat, 4)
    actions_grouped = actions[:, act_idx, :]  # (B, F_lat, 4, 7)

    timesteps = scheduler.sample_timesteps(timestep_rng, b)

    # Future frames noised; history frames clean.
    future_latents = latents[:, :, n_hist:]
    noise = jax.random.normal(noise_rng, future_latents.shape, dtype=future_latents.dtype)
    noisy_future, target_future, training_weight = scheduler.apply_flow_match(
        noise, future_latents, timesteps
    )
    noisy_latents = jnp.concatenate([latents[:, :, :n_hist], noisy_future], axis=2)

    timestep_2d = _build_per_token_timestep(timesteps, F_lat, H_lat, W_lat, n_hist)

    text_pooled = text_tokens.mean(axis=1)
    action_tokens = model.action_encoder(actions_grouped, text_pooled)  # (B, F_lat, 4096)

    model_pred = model.transformer(
        hidden_states=noisy_latents,
        timestep=timestep_2d,
        encoder_hidden_states=action_tokens,
        deterministic=True,
        frame_level_cond=True,
    )

    diff = target_future - model_pred[:, :, n_hist:]
    loss = diff ** 2
    if not config.disable_training_weights:
        loss = loss * jnp.expand_dims(training_weight, (1, 2, 3, 4))
    return jnp.mean(loss)
