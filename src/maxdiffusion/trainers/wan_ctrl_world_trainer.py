"""Action-conditioned WAN (Ctrl-World style) trainer.

Initialises from the WAN 2.2 Ti2V 5B diffusers checkpoint, adds a small
NNX action encoder, and fine-tunes both jointly on robot trajectory data.

Architecture
------------
``WanCtrlWorldModel`` wraps the ``WanModel`` transformer together with
``NNXWanActionEncoder``. Action tokens ``(B, T_act, 4096)`` are fed through
the WAN transformer's text projection as the sole cross-attention sequence —
no changes to the transformer itself.

Training objective (per-token timestep)
----------------------------------------
Uses the same per-token timestep scheme introduced by WAN Ti2V:

* History latent frames are kept **clean** (no noise added) — unless history
  noise augmentation is enabled (``history_noise_max_timestep > 0``), in which
  case each history frame is corrupted at its own small ``t_hist`` so the
  model tolerates imperfect (AR-generated) history at inference. By default
  the corruption is *blind* (history tokens stay declared t=0); set
  ``history_noise_conditioned: True`` to expose ``t_hist`` to AdaLN instead.
* Future latent frames receive flow-matching noise at a sampled global ``t``.
* A ``(B, seq_len)`` timestep array is passed to the transformer: history
  frame tokens get ``t=0`` (or ``t_hist``), future frame tokens get the
  sampled ``t``.
* The WAN model's AdaLN modulation therefore tells each block whether it is
  looking at a clean reference frame (history) or a frame being denoised
  (future), exactly as in Ti2V inference.
* MSE loss is computed **only on future latent frames**.

Conditioning
------------
Action tokens are the sole conditioning signal. Text is not used: the action
encoder is called with ``text_embed=None``, matching the text-free inference
path. 5 % of samples have their action tokens zeroed for classifier-free
guidance.

``action_cond_mode`` (config) selects how the action tokens reach the
transformer:

* ``"cross_attn"`` (default): action tokens are used as the sole
  cross-attention K/V sequence, exactly as in the original design.
* ``"adaln"``: cross-attention instead receives all-zero tokens (the same
  no-op state used for the CFG-uncond branch) and the action tokens are
  projected (``NNXWanActionAdaLNProjector``) and summed into the per-token
  timestep embedding that drives AdaLN modulation. The two modes are
  mutually exclusive and not checkpoint-compatible with each other.

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
from maxdiffusion.models.wan.action_encoder_wan import NNXWanActionEncoder, NNXWanActionAdaLNProjector
from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import load_next_batch


# ── Action grouping ───────────────────────────────────────────────────────────


def _group_actions(actions: jnp.ndarray, F_lat: int) -> jnp.ndarray:
    """Map (B, 4*F_lat, 7) raw actions → (B, F_lat, 4, 7).

    The dataset pre-aligns actions so that index 4k..4k+3 corresponds to
    latent frame k (both history and future).
    """
    B = actions.shape[0]
    return actions.reshape(B, F_lat, 4, 7)


# ── Combined model ────────────────────────────────────────────────────────────


class WanCtrlWorldModel(nnx.Module):
    """WAN transformer + action encoder (+ optional AdaLN action projector),
    held together for nnx.split/merge.

    ``action_adaln_proj`` is only present (non-None) when
    ``config.action_cond_mode == "adaln"``; it is unused in the default
    ``"cross_attn"`` mode.
    """

    def __init__(
        self,
        transformer,
        action_encoder: NNXWanActionEncoder,
        action_adaln_proj: NNXWanActionAdaLNProjector | None = None,
    ):
        self.transformer = transformer
        self.action_encoder = action_encoder
        self.action_adaln_proj = action_adaln_proj if action_adaln_proj is not None else nnx.data(None)


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


def _route_action_conditioning(
    action_tokens: jnp.ndarray,
    action_adaln_proj: NNXWanActionAdaLNProjector | None,
    action_cond_mode: str,
    tokens_per_frame_k: int,
    H_lat: int,
    W_lat: int,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Route encoded action tokens to cross-attention or AdaLN conditioning.

    ``"cross_attn"`` (default): action tokens pass through unchanged as the
    transformer's cross-attention K/V; no AdaLN conditioning is added.
    ``"adaln"``: cross-attention gets all-zero tokens (a no-op — the same
    state used for CFG-uncond) and the action tokens are projected per latent
    frame, then repeated across each frame's spatial patch tokens to align
    with the per-token timestep embedding they get summed into.

    Returns ``(encoder_hidden_states, action_hidden_states)`` — the second
    element is ``None`` in cross-attention mode.
    """
    if action_cond_mode == "adaln":
        b, fk, d = action_tokens.shape
        f_lat = fk // tokens_per_frame_k
        grouped = action_tokens.reshape(b, f_lat, tokens_per_frame_k, d)
        action_temb = action_adaln_proj(grouped)                       # (B, F_lat, inner_dim)
        spatial_tokens_per_frame = (H_lat // 2) * (W_lat // 2)
        action_hidden_states = jnp.repeat(action_temb, spatial_tokens_per_frame, axis=1)
        return jnp.zeros_like(action_tokens), action_hidden_states
    return action_tokens, None


def _build_per_token_timestep(
    timesteps: jnp.ndarray,
    F_lat: int,
    H_lat: int,
    W_lat: int,
    n_hist_lat: int,
    hist_timesteps: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Build ``(B, seq_len)`` timestep array for per-token Ti2V training.

    History frame tokens receive ``t=0`` (treated as clean by AdaLN), or
    ``hist_timesteps`` — ``(B,)`` shared or ``(B, n_hist_lat)`` per-frame —
    when conditioned history noise augmentation is active; future frame
    tokens receive the sampled global timestep ``t``.

    WAN patches spatially at 2×2, so tokens_per_frame = (H_lat//2)*(W_lat//2).
    """
    tokens_per_frame = (H_lat // 2) * (W_lat // 2)
    b = timesteps.shape[0]
    n_fut = F_lat - n_hist_lat
    if hist_timesteps is None:
        hist_t = jnp.zeros((b, n_hist_lat), dtype=timesteps.dtype)
    else:
        hist_t = jnp.broadcast_to(hist_timesteps, (b, n_hist_lat)).astype(timesteps.dtype)
    fut_t = jnp.broadcast_to(timesteps[:, None], (b, n_fut))
    per_frame = jnp.concatenate([hist_t, fut_t], axis=1)          # (B, F_lat)
    return jnp.repeat(per_frame, tokens_per_frame, axis=1)        # (B, seq_len)


def _apply_history_noise(
    rng: jax.Array,
    hist_latents: jnp.ndarray,
    scheduler,
    config,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """History noise augmentation (GameNGen / diffusion-forcing style).

    Corrupts each history latent frame with flow-matching noise at its own
    independently sampled timestep t_hist ~ Uniform(0, history_noise_max_timestep).
    A history_noise_clean_prob fraction of samples keeps the entire history
    window exactly clean (t_hist=0), so the cold-start / ground-truth-history
    case stays in-distribution.

    Returns (noised_hist, hist_timesteps) with hist_timesteps shaped
    (B, F_hist) — one level per (sample, history frame).
    """
    b, _, f_hist = hist_latents.shape[:3]
    t_rng, clean_rng, noise_rng = jax.random.split(rng, 3)

    max_t = float(config.history_noise_max_timestep)
    t_hist = jax.random.uniform(t_rng, (b, f_hist), minval=0.0, maxval=max_t)
    clean = jax.random.bernoulli(
        clean_rng, float(getattr(config, "history_noise_clean_prob", 0.2)), (b, 1)
    )
    t_hist = jnp.where(clean, 0.0, t_hist)

    # Same formula as scheduler.apply_flow_match, but with an independent
    # noise level per history frame (its timesteps arg is per-sample only).
    t_norm = t_hist / float(scheduler.config.num_train_timesteps)
    sigma = (1.0 - t_norm) * scheduler.config.sigma_min + t_norm * scheduler.config.sigma_max
    sigma = sigma[:, None, :, None, None].astype(hist_latents.dtype)   # (B,1,F_hist,1,1)
    noise = jax.random.normal(noise_rng, hist_latents.shape, dtype=hist_latents.dtype)
    noised = (1.0 - sigma) * hist_latents + sigma * noise
    # The sigma formula has a sigma_min floor even at t=0; keep t_hist=0 frames
    # exactly clean instead.
    exact_clean = (t_hist == 0.0)[:, None, :, None, None]
    noised = jnp.where(exact_clean, hist_latents, noised).astype(hist_latents.dtype)
    return noised, t_hist


# ── Training step ─────────────────────────────────────────────────────────────


def _train_step(state: TrainState, data: dict, rng: jax.Array,
                scheduler_state, scheduler, config) -> tuple:
    """
    When grad_accum_steps == 1 (default): data leaves have shape [bsz, ...].
    When grad_accum_steps > 1: data leaves have shape [grad_accum_steps, bsz, ...];
    gradients are accumulated via jax.lax.scan before a single optimizer update.
    """
    _, noise_rng, timestep_rng, drop_rng, new_rng = jax.random.split(rng, 5)

    bsz = config.global_batch_size_to_train_on
    weights_dtype = _dtype(config.weights_dtype)
    n_hist = config.num_history_latent_frames
    grad_accum_steps = getattr(config, "grad_accum_steps", 1)

    def compute_loss(params, micro_data, n_rng, t_rng, d_rng):
        model: WanCtrlWorldModel = nnx.merge(state.graphdef, params, state.rest_of_state)

        latents = micro_data["latent"][:bsz].astype(weights_dtype)               # (B,C,F_lat,H,W)
        actions = micro_data["action"][:bsz].astype(weights_dtype)               # (B,4*F_lat,7)
        frame_positions = micro_data["frame_positions"][:bsz]                    # (B, W) int32

        b, _, F_lat, H_lat, W_lat = latents.shape

        actions_grouped = _group_actions(actions, F_lat)             # (B, F_lat, 4, 7)

        timesteps = scheduler.sample_timesteps(t_rng, b)

        future_latents = latents[:, :, n_hist:]
        n_rng, hist_rng = jax.random.split(n_rng)
        noise = jax.random.normal(n_rng, future_latents.shape, dtype=future_latents.dtype)
        noisy_future, target_future, training_weight = scheduler.apply_flow_match(
            noise, future_latents, timesteps
        )

        hist_latents = latents[:, :, :n_hist]
        hist_timesteps = None
        if getattr(config, "history_noise_max_timestep", 0) > 0:
            hist_latents, hist_t = _apply_history_noise(
                hist_rng, hist_latents, scheduler, config
            )
            # Conditioned: AdaLN is told each history frame's noise level.
            # Blind (default): history stays declared clean (t=0) — the model
            # learns to mildly distrust "clean" history, matching AR inference
            # where the true corruption level is unknown.
            if getattr(config, "history_noise_conditioned", False):
                hist_timesteps = hist_t
        noisy_latents = jnp.concatenate([hist_latents, noisy_future], axis=2)

        timestep_2d = _build_per_token_timestep(
            timesteps, F_lat, H_lat, W_lat, n_hist, hist_timesteps=hist_timesteps
        )
        timestep_2d = jax.lax.with_sharding_constraint(timestep_2d, P(("data", "fsdp", "context"), None))

        action_tokens = model.action_encoder(actions_grouped, None)     # (B, F_lat*K, 4096)
        cfg_rng, do_rng = jax.random.split(d_rng)
        action_tokens = _apply_cfg_dropout(cfg_rng, action_tokens, config.ctrl_cfg_drop_prob)

        action_cond_mode = getattr(config, "action_cond_mode", "cross_attn")
        cond_tokens_per_frame = getattr(config, "action_tokens_per_latent_frame", 1)
        enc_tokens, action_hidden_states = _route_action_conditioning(
            action_tokens, model.action_adaln_proj, action_cond_mode,
            cond_tokens_per_frame, H_lat, W_lat,
        )

        model_pred = model.transformer(
            hidden_states=noisy_latents,
            timestep=timestep_2d,           # (B, seq_len) → per-token AdaLN
            encoder_hidden_states=enc_tokens,
            action_hidden_states=action_hidden_states,
            deterministic=False,
            rngs=nnx.Rngs(dropout=do_rng),
            frame_level_cond=True,
            cond_tokens_per_frame=cond_tokens_per_frame,
            frame_positions=frame_positions,
        )

        diff = target_future - model_pred[:, :, n_hist:]
        loss = diff ** 2
        if not config.disable_training_weights:
            loss = loss * jnp.expand_dims(training_weight, (1, 2, 3, 4))
        return jnp.mean(loss)

    if grad_accum_steps == 1:
        def loss_fn(params):
            return compute_loss(params, data, noise_rng, timestep_rng, drop_rng)

        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        total_loss = loss
    else:
        # data leaves: [grad_accum_steps, bsz, ...]
        # Split each RNG into per-microbatch keys so every microbatch sees different noise.
        noise_rngs = jax.random.split(noise_rng, grad_accum_steps)
        timestep_rngs = jax.random.split(timestep_rng, grad_accum_steps)
        drop_rngs = jax.random.split(drop_rng, grad_accum_steps)

        # Python loop: unrolled at JIT trace time, keeping nnx.value_and_grad at
        # JIT trace level (avoids the cross-trace-level NNX graph inspection error
        # that occurs inside jax.lax.scan).
        acc_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        total_loss = jnp.zeros((), dtype=jnp.float32)

        for i in range(grad_accum_steps):
            micro_data = jax.tree_util.tree_map(lambda x, _i=i: x[_i], data)
            n_i, t_i, d_i = noise_rngs[i], timestep_rngs[i], drop_rngs[i]

            def loss_fn(params, _d=micro_data, _n=n_i, _t=t_i, _dr=d_i):
                return compute_loss(params, _d, _n, _t, _dr)

            grad_fn = nnx.value_and_grad(loss_fn)
            micro_loss, micro_grads = grad_fn(state.params)
            acc_grads = jax.tree_util.tree_map(
                lambda a, g: a + g / grad_accum_steps, acc_grads, micro_grads
            )
            total_loss = total_loss + micro_loss / grad_accum_steps

        grads = acc_grads

    grad_norm = jaxopt.tree_util.tree_l2_norm(grads)
    new_state = state.apply_gradients(grads=grads)

    # Bad-batch guard: if the raw (pre-clip) global grad norm is non-finite or
    # exceeds grad_norm_skip_threshold, revert the whole update (params,
    # opt_state, step) so one bad batch can't poison Adam's moments — even
    # clipped, a spike's direction lingers in m/v for ~1/(1-b2) steps.
    skip_threshold = float(getattr(config, "grad_norm_skip_threshold", 0.0) or 0.0)
    update_skipped = jnp.zeros((), dtype=jnp.float32)
    if skip_threshold > 0.0:
        is_bad = jnp.logical_or(jnp.logical_not(jnp.isfinite(grad_norm)), grad_norm > skip_threshold)
        keep = lambda new, old: jax.tree_util.tree_map(lambda n, o: jnp.where(is_bad, o, n), new, old)
        new_state = new_state.replace(
            step=jnp.where(is_bad, state.step, new_state.step),
            params=keep(new_state.params, state.params),
            opt_state=keep(new_state.opt_state, state.opt_state),
        )
        update_skipped = is_bad.astype(jnp.float32)

    # Data diagnostics: per-sample latent |max| and episode ids, replicated so
    # every host can attribute a grad spike to specific batch samples. Leading
    # dims ((accum,) B) are flattened.
    lat_abs = jnp.abs(data["latent"])
    latent_absmax_per_sample = jnp.max(lat_abs, axis=tuple(range(lat_abs.ndim - 4, lat_abs.ndim))).reshape(-1)
    latent_absmax_per_sample = jax.lax.with_sharding_constraint(latent_absmax_per_sample, P())
    episode_ids = jax.lax.with_sharding_constraint(data["episode_id"].reshape(-1), P())

    metrics = {
        "scalar": {
            "learning/loss": total_loss,
            "learning/grad_norm": grad_norm,
            "learning/latent_absmax": jnp.max(latent_absmax_per_sample),
            "learning/update_skipped": update_skipped,
        },
        "scalars": {
            "learning/latent_absmax_per_sample": latent_absmax_per_sample,
            "learning/episode_ids": episode_ids,
        },
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

    def _load_dataset(self, mesh, is_training: bool, seed: int = None):
        from maxdiffusion.input_pipeline.robot.wan_ctrl_world_dataset import (
            WanCtrlWorldDroidDataset,
        )
        from maxdiffusion.multihost_dataloading import MultiHostDataLoadIterator
        config = self.config
        split = "train" if is_training else "val"
        max_latent_frames = config.num_predicted_latents + config.num_history_latent_frames
        per_host_batch = max(1, config.global_batch_size_to_load // jax.process_count())
        ds = WanCtrlWorldDroidDataset(
            data_dir=config.train_data_dir if is_training else config.eval_data_dir,
            stats_path=config.action_stats_path,
            n_hist=config.num_history_latent_frames,
            max_latent_frames=max_latent_frames,
            action_dim=config.action_dim,
            batch_size=per_host_batch,
            split=split,
            seed=seed if seed is not None else config.seed,
            shuffle=is_training,
            shard_for_training=jax.process_count() > 1,
            # Eval windows are anchored at the episode start (history = frame 0
            # repeated), matching a deployment-style cold-start rollout.
            first_window_only=not is_training,
        )
        return MultiHostDataLoadIterator(ds.dataset, mesh)

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
            tokens_per_frame=getattr(self.config, "action_tokens_per_latent_frame", 1),
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    def _build_action_adaln_proj(self) -> NNXWanActionAdaLNProjector | None:
        """Only built when action_cond_mode == "adaln"; unused (None) otherwise."""
        if getattr(self.config, "action_cond_mode", "cross_attn") != "adaln":
            return None
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        return NNXWanActionAdaLNProjector(
            rngs=nnx.Rngs(jax.random.key(self.config.seed + 1)),
            tokens_per_frame=getattr(self.config, "action_tokens_per_latent_frame", 1),
            wan_text_dim=self.config.wan_text_dim,
            inner_dim=inner_dim,
            dtype=_dtype(self.config.activations_dtype),
            weights_dtype=_dtype(self.config.weights_dtype),
        )

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _build_checkpoint_manager(self, ckpt_dir: str) -> ocp.CheckpointManager:
        os.makedirs(ckpt_dir, exist_ok=True)
        keep_period = getattr(self.config, "checkpoint_keep_period", -1) or None
        options = ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=3,
            enable_async_checkpointing=True,
            keep_period=keep_period,
        )
        return ocp.CheckpointManager(
            ckpt_dir,
            item_names=("params", "opt_state", "step"),
            item_handlers={
                "params":    ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "step":      ocp.JsonCheckpointHandler(),
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
                opt_state=ocp.args.StandardSave(state.opt_state),
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
                opt_state=ocp.args.StandardRestore(state.opt_state),
                step=ocp.args.JsonRestore(),
            ),
        )
        restored_step = int(restored["step"]["step"])
        # step isn't consumed by the optimizer (bias correction and the LR
        # schedule run off counts inside opt_state, restored above) but keep it
        # consistent for anything that reads state.step.
        new_state = state.replace(
            params=restored["params"],
            opt_state=restored["opt_state"],
            step=jnp.asarray(restored_step, dtype=jnp.int32),
        )
        return new_state, restored_step

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def _build_optimizer(self, num_steps: int):
        schedule_steps = (
            self.config.learning_rate_schedule_steps
            if self.config.learning_rate_schedule_steps > 0
            else num_steps
        )
        schedule_type = getattr(self.config, "learning_rate_schedule_type", "constant")
        end_ratio = getattr(self.config, "learning_rate_end_ratio", 0.0)
        lr_schedule = max_utils.create_learning_rate_schedule(
            self.config.learning_rate,
            schedule_steps,
            self.config.warmup_steps_fraction,
            num_steps,
            schedule_type=schedule_type,
            end_value=self.config.learning_rate * end_ratio,
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

    def _data_shardings(self, mesh, for_eval: bool = False) -> dict:
        if not for_eval and getattr(self.config, "grad_accum_steps", 1) > 1:
            # Leading dim is the micro-batch index (local, not distributed).
            pspec = NamedSharding(mesh, P(None, *self.config.data_sharding))
        else:
            pspec = NamedSharding(mesh, P(*self.config.data_sharding))
        return {
            "latent":          pspec,
            "action":          pspec,
            "text_embeds":     pspec,
            "frame_positions": pspec,
            "episode_id":      pspec,
        }

    # ── Main training entry point ─────────────────────────────────────────────

    def start_training(self):
        config = self.config

        # 1. Load WAN pipeline (transformer + mesh)
        pipeline = self._load_wan_pipeline()
        mesh = pipeline.mesh

        # Free VAE — we use pre-encoded latents. When W&B video logging is
        # enabled, keep only the decoder weights (in bf16) for rollout decode.
        self._pipeline = pipeline
        if getattr(config, "wandb_video_every", 0) > 0:
            self._slim_vae_for_video_logging(pipeline)
        else:
            if hasattr(pipeline, "vae"):
                del pipeline.vae
            if hasattr(pipeline, "vae_cache"):
                del pipeline.vae_cache

        # 2. Build combined model
        action_encoder = self._build_action_encoder()
        action_adaln_proj = self._build_action_adaln_proj()
        combined = WanCtrlWorldModel(pipeline.transformer, action_encoder, action_adaln_proj)

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

        # 5. Shard state across devices
        state, state_shardings = self._shard_state(mesh, state)

        # Free the pre-shard param copy after sharding; GPU JAX needs it pinned until then.
        if config.hardware != "gpu":
            max_utils.delete_pytree(params)
        data_shardings = self._data_shardings(mesh)

        # 6. Scheduler
        scheduler, scheduler_state = self._create_scheduler()

        # 7. Checkpoint manager + possible restore
        ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
        ckpt_mgr = self._build_checkpoint_manager(ckpt_dir)
        state, start_step = self._maybe_restore(ckpt_mgr, state)
        if start_step:
            max_logging.log(f"[wan_ctrl_world] resumed at step {start_step}")

        # 8. Data iterator — offset seed by start_step so resume sees a fresh shuffle order.
        train_iter = self._load_dataset(mesh, is_training=True, seed=config.seed + start_step)

        grad_accum_steps = getattr(config, "grad_accum_steps", 1)

        def _next_batch(iterator):
            """Return one batch (grad_accum_steps == 1) or a stacked micro-batch tensor."""
            if grad_accum_steps == 1:
                return next(iterator)
            bufs = [next(iterator) for _ in range(grad_accum_steps)]
            return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *bufs)

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
            max_logging.log(f"  Grad accum steps: {grad_accum_steps}")
            max_logging.log(f"  Effective batch size: {config.global_batch_size_to_train_on * grad_accum_steps}")
            max_logging.log(f"  Max train steps: {config.max_train_steps}")
            max_logging.log(f"  Output dir: {config.output_dir}")

        self._wandb_run = None
        if jax.process_index() == 0 and getattr(config, "wandb_project", ""):
            import wandb
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                entity=getattr(config, "wandb_entity", None) or None,
                name=config.run_name or None,
                settings=wandb.Settings(start_method="thread"),
            )
        wandb_run = self._wandb_run

        rng = jax.random.key(config.seed + 1)
        recent_loss: list[float] = []
        recent_grad: list[float] = []
        recent_absmax: list[float] = []
        skipped_count = 0
        skip_threshold = float(getattr(config, "grad_norm_skip_threshold", 0.0) or 0.0)
        last_step_time = datetime.datetime.now()

        profiler = None
        first_profiling_step = config.skip_first_n_steps_for_profiler
        if max_utils.profiler_enabled(config) and first_profiling_step >= config.max_train_steps:
            raise ValueError("Profiling requested but initial profiling step set past training final step")
        last_profiling_step = np.clip(
            first_profiling_step + config.profiler_steps - 1,
            first_profiling_step,
            config.max_train_steps - 1,
        )

        example_batch = _next_batch(train_iter)

        for step in range(start_step, config.max_train_steps):
            if max_utils.profiler_enabled(config) and step == first_profiling_step:
                profiler = max_utils.Profiler(config)
                profiler.start()
            step_start = datetime.datetime.now()
            with (
                jax.profiler.StepTraceAnnotation("train", step_num=step),
                mesh,
                nn_partitioning.axis_rules(config.logical_axis_rules),
            ):
                state, scheduler_state, metrics, rng = p_train_step(
                    state, example_batch, rng, scheduler_state
                )

            # Measure compute time before prefetching the next batch.
            # _next_batch calls jax.device_put which allocates HBM buffers; on a
            # busy TPU this forces an implicit sync with the ongoing step and
            # inflates step_secs.  We measure first, then prefetch so that the
            # load overlaps with logging / eval / checkpoint logic below.
            metrics["scalar"]["learning/loss"].block_until_ready()
            now = datetime.datetime.now()
            step_secs = (now - step_start).total_seconds()

            if profiler is not None and step == last_profiling_step:
                profiler.stop()
                profiler = None

            example_batch = _next_batch(train_iter)

            recent_loss.append(float(metrics["scalar"]["learning/loss"]))
            recent_grad.append(float(metrics["scalar"]["learning/grad_norm"]))
            recent_absmax.append(float(metrics["scalar"]["learning/latent_absmax"]))
            skipped_count += int(float(metrics["scalar"]["learning/update_skipped"]))

            # Grad-spike attribution: on a skipped step, name the batch's
            # episodes sorted by latent |max| so bad data can be tracked down.
            grad_val = recent_grad[-1]
            if skip_threshold > 0.0 and (not np.isfinite(grad_val) or grad_val > skip_threshold):
                eids = np.asarray(metrics["scalars"]["learning/episode_ids"])
                absmax = np.asarray(metrics["scalars"]["learning/latent_absmax_per_sample"])
                if jax.process_index() == 0:
                    order = np.argsort(-absmax)
                    offenders = ", ".join(f"ep{int(eids[i])}:{absmax[i]:.3e}" for i in order)
                    max_logging.log(
                        f"[wan_ctrl_world] GRAD SPIKE step {step}: raw grad_norm={grad_val:.3e} "
                        f"(threshold {skip_threshold:g}) — update skipped. "
                        f"Batch episode_id:latent_absmax (desc): {offenders}"
                    )

            if jax.process_index() == 0 and (step < 5 or (step + 1) % config.log_period == 0):
                max_logging.log(f"step {step} s/step={step_secs:.2f}")

            if (step + 1) % config.log_period == 0 and jax.process_index() == 0:
                lr = float(lr_schedule(step))
                avg_loss = sum(recent_loss) / len(recent_loss)
                avg_grad = sum(recent_grad) / len(recent_grad)
                sps = config.log_period / (now - last_step_time).total_seconds()
                max_logging.log(
                    f"step {step + 1}/{config.max_train_steps} "
                    f"loss={avg_loss:.4f} grad_norm={avg_grad:.3f} "
                    f"lr={lr:.2e} steps/s={sps:.2f} s/step={1/sps:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log({"train/loss": avg_loss, "train/grad_norm": avg_grad,
                                   "train/grad_norm_max": max(recent_grad),
                                   "train/latent_absmax": max(recent_absmax),
                                   "train/updates_skipped": skipped_count,
                                   "train/lr": lr, "train/steps_per_sec": sps}, step=step + 1)
                recent_loss.clear()
                recent_grad.clear()
                recent_absmax.clear()
                skipped_count = 0
                last_step_time = now

            if (
                config.eval_every > 0
                and (step + 1) % config.eval_every == 0
            ):
                self._run_eval(mesh, state, state_shardings,
                               data_shardings, scheduler, scheduler_state, step + 1, rng)

            if (
                getattr(config, "wandb_video_every", 0) > 0
                and getattr(config, "wandb_project", "")
                and (step + 1) % config.wandb_video_every == 0
            ):
                self._run_video_log(mesh, state, state_shardings, scheduler, step + 1, rng)

            if (
                config.checkpoint_every > 0
                and (step + 1) % config.checkpoint_every == 0
            ):
                self._save_checkpoint(ckpt_mgr, step + 1, state)

        if config.save_final_checkpoint:
            self._save_checkpoint(ckpt_mgr, config.max_train_steps, state)
        ckpt_mgr.wait_until_finished()
        if wandb_run is not None:
            wandb_run.finish()

    # ── Eval ──────────────────────────────────────────────────────────────────

    def _run_eval(
        self,
        mesh,
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

        if not hasattr(self, "_eval_iter"):
            self._eval_iter = self._load_dataset(mesh, is_training=False)
        eval_iter = self._eval_iter

        if not hasattr(self, "_p_eval_step"):
            eval_data_shardings = self._data_shardings(mesh, for_eval=True)
            self._p_eval_step = jax.jit(
                functools.partial(_eval_step, scheduler=scheduler, config=config),
                in_shardings=(state_shardings, eval_data_shardings, None, None),
                out_shardings=None,
            )
        p_eval_step = self._p_eval_step

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
            mean_loss = sum(losses) / len(losses)
            max_logging.log(
                f"[wan_ctrl_world] eval step={step} batches={len(losses)} "
                f"mean_loss={mean_loss:.4f}"
            )
            if getattr(self, "_wandb_run", None) is not None:
                self._wandb_run.log({"eval/loss": mean_loss}, step=step)

    # ── W&B video logging ─────────────────────────────────────────────────────

    def _slim_vae_for_video_logging(self, pipeline):
        """Keep only the VAE weights video logging needs.

        Video logging only calls ``vae.decode``, which touches
        ``post_quant_conv`` and ``decoder`` — the encoder and ``quant_conv``
        weights (~150M params) are dropped, and the remaining ~555M decoder
        params are cast to bf16 (plenty for preview videos). Cuts the VAE
        footprint from ~2.8 GB fp32 to ~1.1 GB across the mesh.
        """
        vae = pipeline.vae
        vae.encoder = None
        vae.quant_conv = None
        graphdef, state = nnx.split(vae)
        # Cast inside jit: eager ops on multi-host globally-sharded arrays
        # are unsafe.
        state = jax.jit(
            functools.partial(
                jax.tree_util.tree_map,
                lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            )
        )(state)
        pipeline.vae = nnx.merge(graphdef, state)
        # The cache only uses its conv counts after construction; repoint its
        # module ref so the old fp32 VAE can be garbage-collected.
        pipeline.vae_cache.module = pipeline.vae
        max_logging.log("[wan_ctrl_world] VAE slimmed for video logging: encoder dropped, decoder cast to bf16")

    def _run_video_log(
        self,
        mesh,
        state: TrainState,
        state_shardings,
        scheduler,
        step: int,
        rng: jax.Array,
    ):
        """Rollout one eval batch, VAE-decode gen vs GT, log videos to W&B.

        Runs on every host (the rollout and decode are collective ops);
        only process 0 writes to W&B.
        """
        config = self.config
        pipeline = self._pipeline
        if not config.eval_data_dir:
            max_logging.log("[wan_ctrl_world] wandb_video_every>0 but eval_data_dir not set; skipping")
            return

        if not hasattr(self, "_eval_iter"):
            self._eval_iter = self._load_dataset(mesh, is_training=False)
        try:
            batch = next(self._eval_iter)
        except StopIteration:
            self._eval_iter = self._load_dataset(mesh, is_training=False)
            batch = next(self._eval_iter)

        if not hasattr(self, "_p_video_rollout"):
            lat_mean = jnp.array(pipeline.vae.latents_mean).reshape(1, -1, 1, 1, 1)
            lat_std = jnp.array(pipeline.vae.latents_std).reshape(1, -1, 1, 1, 1)
            self._p_video_rollout = jax.jit(
                functools.partial(
                    _video_rollout,
                    scheduler=scheduler,
                    config=config,
                    num_steps=int(getattr(config, "wandb_video_inference_steps", 20)),
                    guidance_scale=float(getattr(config, "wandb_video_guidance_scale", 1.0)),
                    num_samples=int(getattr(config, "wandb_video_samples", 1)),
                    lat_mean=lat_mean,
                    lat_std=lat_std,
                ),
                in_shardings=(state_shardings, self._data_shardings(mesh, for_eval=True), None),
                out_shardings=None,
            )

        rng, sub = jax.random.split(rng)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            gen_lat, gt_lat, future_mse = self._p_video_rollout(state, batch, sub)
            gen_lat.block_until_ready()

        # Spatially-sharded VAE decode + cross-host allgather → host numpy
        # (B*3cams, T, H, W, C) in [0, 1].
        gen_np = np.asarray(pipeline._decode_latents_to_video(gen_lat))
        gt_np = np.asarray(pipeline._decode_latents_to_video(gt_lat))

        if jax.process_index() != 0 or getattr(self, "_wandb_run", None) is None:
            return

        import wandb

        n = gen_np.shape[0] // 3
        t, h, w, c = gen_np.shape[1:]
        gen_np = gen_np.reshape(n, 3, t, h, w, c)
        gt_np = gt_np.reshape(n, 3, t, h, w, c)

        logs = {"eval/video_rollout_latent_mse": float(future_mse)}
        for i in range(n):
            gen_grid = np.concatenate(list(gen_np[i]), axis=1)              # cameras stacked on H
            gt_grid = np.concatenate(list(gt_np[i]), axis=1)
            side_by_side = np.concatenate([gen_grid, gt_grid], axis=2)      # gen | GT on W
            # wandb.Video expects uint8 (T, C, H, W)
            frames = (side_by_side * 255).clip(0, 255).astype(np.uint8).transpose(0, 3, 1, 2)
            logs[f"eval/video/sample_{i}"] = wandb.Video(
                frames, fps=config.output_video_fps, format="mp4"
            )
        self._wandb_run.log(logs, step=step)
        max_logging.log(f"[wan_ctrl_world] logged {n} rollout video(s) to W&B at step {step}")


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
    frame_positions = data["frame_positions"][:bsz]                  # (B, W) int32

    b, _, F_lat, H_lat, W_lat = latents.shape

    actions_grouped = _group_actions(actions, F_lat)                # (B, F_lat, 4, 7)

    timesteps = scheduler.sample_timesteps(timestep_rng, b)

    future_latents = latents[:, :, n_hist:]
    noise = jax.random.normal(noise_rng, future_latents.shape, dtype=future_latents.dtype)
    noisy_future, target_future, training_weight = scheduler.apply_flow_match(
        noise, future_latents, timesteps
    )
    noisy_latents = jnp.concatenate([latents[:, :, :n_hist], noisy_future], axis=2)

    timestep_2d = _build_per_token_timestep(timesteps, F_lat, H_lat, W_lat, n_hist)
    timestep_2d = jax.lax.with_sharding_constraint(timestep_2d, P(("data", "fsdp", "context"), None))

    action_tokens = model.action_encoder(actions_grouped, None)     # (B, F_lat*K, 4096)

    action_cond_mode = getattr(config, "action_cond_mode", "cross_attn")
    cond_tokens_per_frame = getattr(config, "action_tokens_per_latent_frame", 1)
    enc_tokens, action_hidden_states = _route_action_conditioning(
        action_tokens, model.action_adaln_proj, action_cond_mode,
        cond_tokens_per_frame, H_lat, W_lat,
    )

    model_pred = model.transformer(
        hidden_states=noisy_latents,
        timestep=timestep_2d,
        encoder_hidden_states=enc_tokens,
        action_hidden_states=action_hidden_states,
        deterministic=True,
        frame_level_cond=True,
        cond_tokens_per_frame=cond_tokens_per_frame,
        frame_positions=frame_positions,
    )

    diff = target_future - model_pred[:, :, n_hist:]
    loss = diff ** 2
    if not config.disable_training_weights:
        loss = loss * jnp.expand_dims(training_weight, (1, 2, 3, 4))
    return jnp.mean(loss)


def _video_rollout(state: TrainState, data: dict, rng: jax.Array,
                   scheduler, config, num_steps: int, guidance_scale: float,
                   num_samples: int, lat_mean: jnp.ndarray, lat_std: jnp.ndarray) -> tuple:
    """Euler rollout for W&B video logging.

    History latent frames stay clean; future frames are denoised from pure
    noise conditioned on action tokens (same per-token timestep scheme as
    training). Returns ``(gen, gt, future_mse)`` where gen/gt are
    denormalized VAE latents of shape (num_samples*3, C, F_lat, H_cam, W) —
    the 3 H-stacked cameras unstacked into the batch axis (sample-major) so
    they can be VAE-decoded in one call.
    """
    bsz = config.global_batch_size_to_train_on
    weights_dtype = _dtype(config.weights_dtype)
    n_hist = config.num_history_latent_frames

    model: WanCtrlWorldModel = nnx.merge(state.graphdef, state.params, state.rest_of_state)

    latents = data["latent"][:bsz].astype(weights_dtype)          # (B,C,F_lat,3H,W)
    actions = data["action"][:bsz].astype(weights_dtype)
    frame_positions = data["frame_positions"][:bsz]

    b, _, F_lat, H_lat, W_lat = latents.shape
    actions_grouped = _group_actions(actions, F_lat)
    action_tokens = model.action_encoder(actions_grouped, None)

    action_cond_mode = getattr(config, "action_cond_mode", "cross_attn")
    cond_tokens_per_frame = getattr(config, "action_tokens_per_latent_frame", 1)

    num_train_t = scheduler.config.num_train_timesteps
    # Shift-warped sigma schedule — the same warp FlaxFlowMatchScheduler.set_timesteps
    # applies at inference; official Wan2.2 TI2V-5B ships sample_shift=5.0.
    shift = float(getattr(config, "inference_sigma_shift", 5.0))
    t_uniform = jnp.linspace(1.0, 0.0, num_steps + 1)
    sigmas = scheduler.config.sigma_min + (scheduler.config.sigma_max - scheduler.config.sigma_min) * t_uniform
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    rollout_ts = (sigmas * num_train_t).astype(jnp.int32)

    history = latents[:, :, :n_hist]
    future_gt = latents[:, :, n_hist:]
    gen_init = jax.random.normal(rng, future_gt.shape, dtype=latents.dtype)

    def scan_body(lat, step_idx):
        t_from = rollout_ts[step_idx]
        sig_from = sigmas[step_idx]
        sig_to = sigmas[step_idx + 1]

        roll_input = jnp.concatenate([history, lat], axis=2)
        ts_2d = _build_per_token_timestep(
            jnp.broadcast_to(t_from, (b,)), F_lat, H_lat, W_lat, n_hist
        )
        ts_2d = jax.lax.with_sharding_constraint(ts_2d, P(("data", "fsdp", "context"), None))

        def _velocity(tokens):
            enc_tokens, action_hidden_states = _route_action_conditioning(
                tokens, model.action_adaln_proj, action_cond_mode,
                cond_tokens_per_frame, H_lat, W_lat,
            )
            return model.transformer(
                hidden_states=roll_input,
                timestep=ts_2d,
                encoder_hidden_states=enc_tokens,
                action_hidden_states=action_hidden_states,
                deterministic=True,
                frame_level_cond=True,
                cond_tokens_per_frame=cond_tokens_per_frame,
                frame_positions=frame_positions,
            )

        v_pred = _velocity(action_tokens)
        if guidance_scale > 1.0:
            v_uncond = _velocity(jnp.zeros_like(action_tokens))
            v_pred = v_uncond + guidance_scale * (v_pred - v_uncond)

        # Euler step: x_{t_to} = x_{t_from} + (σ_{t_to} - σ_{t_from}) * v
        v_future = v_pred[:, :, n_hist:]
        new_lat = (lat + (sig_to - sig_from) * v_future).astype(lat.dtype)
        return new_lat, None

    final_future, _ = jax.lax.scan(scan_body, gen_init, jnp.arange(num_steps))

    future_mse = jnp.mean(
        (final_future.astype(jnp.float32) - future_gt.astype(jnp.float32)) ** 2
    )
    gen = jnp.concatenate([history, final_future], axis=2)

    def _prep_for_decode(lat):
        # Denormalize with VAE stats, then unstack the 3 cameras from the H
        # axis into the batch axis. Done inside jit: eager ops on multi-host
        # globally-sharded arrays are unsafe.
        lat = lat[:num_samples].astype(jnp.float32) * lat_std + lat_mean
        n, c, f, h3, w = lat.shape
        lat = lat.reshape(n, c, f, 3, h3 // 3, w)
        lat = jnp.transpose(lat, (0, 3, 1, 2, 4, 5))          # (n, cam, C, F, H, W)
        return lat.reshape(n * 3, c, f, h3 // 3, w)

    return _prep_for_decode(gen), _prep_for_decode(latents), future_mse
