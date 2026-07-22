"""Inference script for WAN 2.2 Ctrl-World (action-conditioned Ti2V).

Loads a trained WanCtrlWorldModel checkpoint, runs the flow-matching denoising
loop on each eval episode, decodes with the WAN VAE, and writes GT-vs-predicted
comparison videos.

Usage:
  python src/maxdiffusion/generate_wan_ctrl_world.py \\
      src/maxdiffusion/configs/base_wan_ctrl_world.yml \\
      checkpoint_dir=wan-ctrl-world-output/checkpoints \\
      eval_data_dir=droid_wan_tfrecords_test/val \\
      action_stats_path=droid_wan_tfrecords_test/stats.json \\
      num_inference_steps=20 \\
      num_eval_videos=4 \\
      output_dir=inference_output
"""

from __future__ import annotations

import functools
import math
import os
from typing import Sequence

from maxdiffusion.utils import export_to_video
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.models.wan.action_encoder_wan import NNXWanActionEncoder, NNXWanActionAdaLNProjector
from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.trainers.wan_ctrl_world_trainer import (
    WanCtrlWorldModel,
    _build_per_token_timestep,
    _dtype,
    _group_actions,
    _route_action_conditioning,
)


# ── Inference step (single denoising iteration) ───────────────────────────────


def _denoise_step(
    params: nnx.State,
    graphdef: nnx.GraphDef,
    rest_of_state: nnx.State,
    latents: jnp.ndarray,
    clean_hist: jnp.ndarray,
    action_tokens: jnp.ndarray,
    uncond_action_tokens: jnp.ndarray,
    timestep: jnp.ndarray,
    frame_positions: jnp.ndarray,
    n_hist: int,
    guidance_scale: float,
    scheduler: FlaxFlowMatchScheduler,
    scheduler_state,
    cond_tokens_per_frame: int = 1,
    action_cond_mode: str = "cross_attn",
):
    """One Euler flow-matching step with optional classifier-free guidance.

    When guidance_scale > 1, runs a single double-batched forward pass with
    [uncond | cond] stacked along the batch axis, then blends:
        pred = uncond + guidance_scale * (cond - uncond)

    Returns (updated_latents, pred_std).
    """
    model: WanCtrlWorldModel = nnx.merge(graphdef, params, rest_of_state)
    b, _, F_lat, H_lat, W_lat = latents.shape
    t_batch = jnp.broadcast_to(timestep, (b,))
    timestep_2d = _build_per_token_timestep(t_batch, F_lat, H_lat, W_lat, n_hist)

    def _route(tokens):
        return _route_action_conditioning(
            tokens, model.action_adaln_proj, action_cond_mode,
            cond_tokens_per_frame, H_lat, W_lat,
        )

    if guidance_scale > 1.0:
        # Double-batch: [uncond, cond] in a single forward pass.
        enc_uncond, adaln_uncond = _route(uncond_action_tokens)
        enc_cond, adaln_cond = _route(action_tokens)
        latents_2x = jnp.concatenate([latents, latents], axis=0)
        tokens_2x  = jnp.concatenate([enc_uncond, enc_cond], axis=0)
        t_2d       = jnp.concatenate([timestep_2d, timestep_2d], axis=0)
        pos_2x     = jnp.concatenate([frame_positions, frame_positions], axis=0)
        action_hidden_states_2x = (
            jnp.concatenate([adaln_uncond, adaln_cond], axis=0) if adaln_cond is not None else None
        )
        pred_2x = model.transformer(
            hidden_states=latents_2x,
            timestep=t_2d,
            encoder_hidden_states=tokens_2x,
            action_hidden_states=action_hidden_states_2x,
            deterministic=True,
            frame_level_cond=True,
            cond_tokens_per_frame=cond_tokens_per_frame,
            frame_positions=pos_2x,
        )
        pred_uncond = pred_2x[:b]
        pred_cond   = pred_2x[b:]
        model_pred  = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    else:
        enc_tokens, action_hidden_states = _route(action_tokens)
        model_pred = model.transformer(
            hidden_states=latents,
            timestep=timestep_2d,
            encoder_hidden_states=enc_tokens,
            action_hidden_states=action_hidden_states,
            deterministic=True,
            frame_level_cond=True,
            cond_tokens_per_frame=cond_tokens_per_frame,
            frame_positions=frame_positions,
        )

    # Apply scheduler step to future frames only; restore clean history after.
    future_pred = model_pred[:, :, n_hist:]
    future_latents = latents[:, :, n_hist:]
    output = scheduler.step(scheduler_state, future_pred, timestep, future_latents)
    return jnp.concatenate([clean_hist, output.prev_sample], axis=2), jnp.std(future_pred)


# ── Action token encoding ─────────────────────────────────────────────────────


def _encode_actions(
    params: nnx.State,
    graphdef: nnx.GraphDef,
    rest_of_state: nnx.State,
    actions: jnp.ndarray,
    F_lat: int,
) -> jnp.ndarray:
    """Encode grouped actions into cross-attention tokens (no text conditioning)."""
    model: WanCtrlWorldModel = nnx.merge(graphdef, params, rest_of_state)
    actions_grouped = _group_actions(actions, F_lat)  # (B, F_lat, 4, 7)
    return model.action_encoder(actions_grouped, None) # (B, F_lat, 4096)


# ── Full denoising loop ───────────────────────────────────────────────────────


def run_ar_denoising(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    rest_of_state: nnx.State,
    initial_hist: jnp.ndarray,
    all_actions: jnp.ndarray,
    all_frame_positions: jnp.ndarray,
    p_encode,
    scheduler: FlaxFlowMatchScheduler,
    num_inference_steps: int,
    n_hist: int,
    ar_chunk_size: int,
    ar_num_chunks: int,
    rng: jax.Array,
    dtype,
    mesh,
    logical_axis_rules,
    guidance_scale: float = 1.0,
    cond_tokens_per_frame: int = 1,
    action_cond_mode: str = "cross_attn",
) -> jnp.ndarray:
    """Auto-regressive denoising: generate ar_num_chunks * ar_chunk_size future frames.

    At each step the last n_hist generated frames become the conditioning history
    for the next step, so errors can compound but temporal consistency is maintained.

    all_actions must cover the full trajectory window:
        shape (B, 4 * (n_hist + ar_num_chunks * ar_chunk_size), 7)
    all_frame_positions holds the temporal RoPE index of every latent frame in
    the full trajectory window, shape (B, n_hist + ar_num_chunks * ar_chunk_size)
    int32 — same tensor the dataset provides at training time. Each AR chunk's
    window covers latent frames [chunk_i*ar_chunk_size, chunk_i*ar_chunk_size +
    window_F_lat), so its positions are the matching contiguous slice: history
    frames (the last n_hist previously generated frames) keep the positions they
    were generated at, keeping RoPE consistent with training.

    Returns predicted future latents: (B, C, ar_num_chunks * ar_chunk_size, H, W).
    """
    window_F_lat = n_hist + ar_chunk_size
    current_hist = initial_hist
    generated_chunks = []

    for chunk_i in range(ar_num_chunks):
        max_logging.log(
            f"[wan_ctrl_world_infer] AR chunk {chunk_i + 1}/{ar_num_chunks}"
        )
        # Slice the actions that correspond to the current window.
        # Window covers latent frames [chunk_i*ar_chunk_size,
        #                              chunk_i*ar_chunk_size + window_F_lat).
        act_start = 4 * chunk_i * ar_chunk_size
        act_end   = 4 * (chunk_i * ar_chunk_size + window_F_lat)
        actions_chunk = all_actions[:, act_start:act_end, :]  # (B, 4*window_F_lat, 7)

        pos_start = chunk_i * ar_chunk_size
        positions_chunk = all_frame_positions[:, pos_start:pos_start + window_F_lat]

        action_tokens = p_encode(params, actions=actions_chunk, F_lat=window_F_lat)

        rng, step_rng = jax.random.split(rng)
        gen_chunk = run_denoising(
            graphdef, params, rest_of_state,
            current_hist, ar_chunk_size, action_tokens, positions_chunk,
            scheduler, num_inference_steps, n_hist,
            step_rng, dtype, mesh, logical_axis_rules,
            guidance_scale=guidance_scale,
            cond_tokens_per_frame=cond_tokens_per_frame,
            action_cond_mode=action_cond_mode,
        )  # (B, C, ar_chunk_size, H, W)

        generated_chunks.append(gen_chunk)

        # Roll history forward: take last n_hist frames from [hist | generated].
        full_window = jnp.concatenate([current_hist, gen_chunk], axis=2)
        current_hist = full_window[:, :, -n_hist:, :, :]

    return jnp.concatenate(generated_chunks, axis=2)  # (B, C, total_fut, H, W)


def run_denoising(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    rest_of_state: nnx.State,
    clean_hist: jnp.ndarray,
    n_fut: int,
    action_tokens: jnp.ndarray,
    frame_positions: jnp.ndarray,
    scheduler: FlaxFlowMatchScheduler,
    num_inference_steps: int,
    n_hist: int,
    rng: jax.Array,
    dtype,
    mesh,
    logical_axis_rules,
    guidance_scale: float = 1.0,
    cond_tokens_per_frame: int = 1,
    action_cond_mode: str = "cross_attn",
) -> jnp.ndarray:
    """Denoise future latent frames from random noise, conditioned on history + actions.

    frame_positions (B, n_hist + n_fut) int32 holds the per-frame temporal RoPE
    indices for the window, matching what the dataset fed the model at training
    time (absolute episode frame indices, history clipped/repeated at episode
    start).

    When guidance_scale > 1, uses classifier-free guidance: the model runs with
    zeroed action tokens (unconditioned) and real action tokens (conditioned) in
    a single double-batched forward pass per step, then blends the predictions.

    Returns the denoised future latents: (B, C, n_fut, H, W).
    """
    b, C, _, H, W = clean_hist.shape
    noisy_future = jax.random.normal(rng, (b, C, n_fut, H, W), dtype=dtype)
    latents = jnp.concatenate([clean_hist, noisy_future], axis=2)

    sched_state = scheduler.set_timesteps(
        scheduler.create_state(), num_inference_steps=num_inference_steps, training=False
    )

    uncond_action_tokens = jnp.zeros_like(action_tokens)

    p_step = jax.jit(
        functools.partial(
            _denoise_step,
            graphdef=graphdef,
            rest_of_state=rest_of_state,
            clean_hist=clean_hist,
            action_tokens=action_tokens,
            uncond_action_tokens=uncond_action_tokens,
            frame_positions=frame_positions,
            n_hist=n_hist,
            guidance_scale=guidance_scale,
            scheduler=scheduler,
            scheduler_state=sched_state,
            cond_tokens_per_frame=cond_tokens_per_frame,
            action_cond_mode=action_cond_mode,
        ),
    )

    timesteps_np = np.array(sched_state.timesteps)
    with mesh, nn_partitioning.axis_rules(logical_axis_rules):
      for step_i, t in enumerate(timesteps_np):
          latents, pred_std = p_step(params, latents=latents, timestep=jnp.array(t))
          if step_i == 0 or step_i == len(timesteps_np) - 1 or (step_i + 1) % 10 == 0:
              fut = latents[:, :, n_hist:]
              # std across the time axis → how much frames differ from each other
              temporal_std = float(jnp.std(fut, axis=2).mean())
              max_logging.log(
                  f"  denoise step {step_i + 1}/{len(timesteps_np)} "
                  f"t={t:.1f}  future_pred_std={float(pred_std):.4f}  "
                  f"future_lat_std={float(jnp.std(fut)):.4f}  "
                  f"temporal_std={temporal_std:.4f}"
              )

    return latents[:, :, n_hist:]   # (B, C, n_fut, H, W)


# ── Video helpers ─────────────────────────────────────────────────────────────


def _decode_latents(pipeline: WanPipelineTI2V_2_2, latents: jnp.ndarray) -> np.ndarray:
    """Decode (B, C, F, H*3, W) latents → (B, F, H*3, W, 3).

    The 3 cameras were encoded separately and stacked along H, so we split,
    decode each independently, then concatenate along H.

    Latents are stored normalized ((raw - mean) / std) by the data preprocessing
    script, so we denormalize before passing to the VAE decoder.
    """
    B, C, F, H3, W = latents.shape
    H = H3 // 3
    cam_videos = []
    for i in range(3):
        cam = latents[:, :, :, i * H:(i + 1) * H, :]        # (B, C, F, H, W)
        cam = pipeline._denormalize_latents(cam)
        cam_videos.append(pipeline._decode_latents_to_video(cam))     # (B, F, H, W, 3)
    return np.concatenate(cam_videos, axis=2)                 # (B, F, H*3, W, 3)


def _save_comparison_video(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    path: str,
    fps: int = 8,
) -> None:
    """Write GT (top half) / predicted (bottom half) stacked MP4."""
    gt = np.clip(gt_frames[0], 0.0, 1.0)    # (F, H, W, 3) float32 in [0, 1]
    pred = np.clip(pred_frames[0], 0.0, 1.0)
    F, H, W, _ = gt.shape
    h = H // 3
    # Split 3 camera views stacked in H, then arrange them width-wise.
    gt_wide   = np.concatenate(np.split(gt,   3, axis=1), axis=2)  # (F, h, W*3, 3)
    pred_wide = np.concatenate(np.split(pred, 3, axis=1), axis=2)  # (F, h, W*3, 3)
    combined  = np.concatenate([gt_wide, pred_wide], axis=1)        # (F, h*2, W*3, 3)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    export_to_video(list(combined), path, fps=fps)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config

    num_inference_steps = int(getattr(config, "num_inference_steps", 20))
    guidance_scale = float(getattr(config, "guidance_scale", 1.0))
    n_hist = config.num_history_latent_frames
    weights_dtype = _dtype(config.weights_dtype)

    autoregressive = bool(getattr(config, "autoregressive", False))
    # Number of future latent frames generated per AR step (defaults to the
    # single-pass window size, num_predicted_latents).
    n_fut_single = config.num_predicted_latents
    ar_chunk_size = int(getattr(config, "ar_chunk_size", 0))
    if ar_chunk_size <= 0:
        ar_chunk_size = n_fut_single
    ar_num_chunks = int(getattr(config, "ar_num_chunks", 1))
    # DROID is 15 Hz; WAN VAE has 4× temporal compression → ~4 fps for real-time playback.
    # config.fps is the WAN model's internet-video fps (16), not the data rate.
    fps = int(getattr(config, "output_video_fps", 16))

    max_logging.log(
        f"[wan_ctrl_world_infer] num_inference_steps={num_inference_steps}  "
        f"guidance_scale={guidance_scale}  "
        f"autoregressive={autoregressive}  "
        + (f"ar_chunk_size={ar_chunk_size}  ar_num_chunks={ar_num_chunks}" if autoregressive else "")
    )

    # ── Load WAN pipeline (keeps VAE for decoding) ────────────────────────────
    max_logging.log("[wan_ctrl_world_infer] loading WAN pipeline...")
    with nn_partitioning.axis_rules(config.logical_axis_rules):
        pipeline = WanPipelineTI2V_2_2.from_pretrained(config)

    # ── Build combined model (same architecture as training) ──────────────────
    action_tokens_per_frame = int(getattr(config, "action_tokens_per_latent_frame", 1))
    action_cond_mode = getattr(config, "action_cond_mode", "cross_attn")
    action_encoder = NNXWanActionEncoder(
        rngs=nnx.Rngs(jax.random.key(config.seed)),
        action_dim=config.action_dim,
        num_actions=4,
        hidden_dim=config.wan_action_encoder_hidden_dim,
        out_dim=config.wan_text_dim,
        tokens_per_frame=action_tokens_per_frame,
        dtype=weights_dtype,
        weights_dtype=weights_dtype,
    )
    action_adaln_proj = None
    if action_cond_mode == "adaln":
        # inner_dim must come from the loaded transformer's own registered
        # config, not config.num_attention_heads/attention_head_dim (those
        # top-level yaml fields are stale for this pipeline — the real
        # architecture is loaded from the pretrained checkpoint's config.json).
        inner_dim = pipeline.transformer.config.num_attention_heads * pipeline.transformer.config.attention_head_dim
        action_adaln_proj = NNXWanActionAdaLNProjector(
            rngs=nnx.Rngs(jax.random.key(config.seed + 1)),
            tokens_per_frame=action_tokens_per_frame,
            wan_text_dim=config.wan_text_dim,
            inner_dim=inner_dim,
            dtype=weights_dtype,
            weights_dtype=weights_dtype,
        )

    with pipeline.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        combined = WanCtrlWorldModel(pipeline.transformer, action_encoder, action_adaln_proj)
        graphdef, params, rest_of_state = nnx.split(combined, nnx.Param, ...)

    # ── Restore checkpoint ────────────────────────────────────────────────────
    ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
    ckpt_mgr = ocp.CheckpointManager(
        ckpt_dir,
        item_names=("params", "step"),
        item_handlers={
            "params": ocp.StandardCheckpointHandler(),
            "step":   ocp.JsonCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(max_to_keep=3),
    )
    checkpoint_step = getattr(config, "checkpoint_step", -1)
    latest = checkpoint_step if checkpoint_step > 0 else ckpt_mgr.latest_step()
    if latest is None:
        raise ValueError(f"No checkpoint found in {ckpt_dir}")

    restored = ckpt_mgr.restore(
        latest,
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(params),
            step=ocp.args.JsonRestore(),
        ),
    )
    params = restored["params"]
    max_logging.log(
        f"[wan_ctrl_world_infer] restored checkpoint step={restored['step']['step']}"
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Inference-only scheduler: shift shapes the sigma schedule in set_timesteps;
    # 5.0 matches the official Wan2.2 TI2V-5B sample_shift.
    scheduler = FlaxFlowMatchScheduler(
        shift=float(getattr(config, "inference_sigma_shift", 5.0)), dtype=jnp.float32
    )

    # ── Eval dataset ──────────────────────────────────────────────────────────
    from maxdiffusion.input_pipeline.robot.wan_ctrl_world_dataset import (
        WanCtrlWorldDroidDataset,
    )
    if autoregressive:
        max_latent_frames = n_hist + ar_num_chunks * ar_chunk_size
    else:
        max_latent_frames = config.num_predicted_latents + config.num_history_latent_frames
    dataset = WanCtrlWorldDroidDataset(
        data_dir=config.eval_data_dir,
        stats_path=config.action_stats_path,
        n_hist=n_hist,
        max_latent_frames=max_latent_frames,
        action_dim=config.action_dim,
        batch_size=1,
        split="val",
        seed=config.seed,
        shuffle=False,
        shard_for_training=False,
        first_window_only=autoregressive,
    )

    # ── Precompile action encoding ─────────────────────────────────────────────
    p_encode = jax.jit(
        functools.partial(
            _encode_actions,
            graphdef=graphdef,
            rest_of_state=rest_of_state,
        ),
        static_argnames=["F_lat"],
    )

    # ── Inference loop ────────────────────────────────────────────────────────
    output_dir = os.path.join(config.output_dir, "inference_videos")
    rng = jax.random.key(config.seed + 42)

    for vid_idx, batch in enumerate(dataset):
        max_logging.log(f"[wan_ctrl_world_infer] generating sample {vid_idx + 1}...")

        latent = jnp.array(batch["latent"]).astype(weights_dtype)       # (1, C, W, H, Wl)
        actions = jnp.array(batch["action"]).astype(weights_dtype)      # (1, 4*W, 7)
        # Temporal RoPE indices for every latent frame in the window — same
        # tensor training feeds the transformer (see wan_ctrl_world_trainer).
        frame_positions = jnp.array(batch["frame_positions"]).astype(jnp.int32)  # (1, W)

        _, _, F_lat, _, _ = latent.shape
        clean_hist = latent[:, :, :n_hist, :, :]                        # (1, C, n_hist, H, Wl)
        gt_future = latent[:, :, n_hist:, :, :]                         # (1, C, n_fut, H, Wl)
        n_fut = gt_future.shape[2]

        # Denoising — single-pass or auto-regressive.
        rng, step_rng = jax.random.split(rng)
        if autoregressive:
            pred_future = run_ar_denoising(
                graphdef, params, rest_of_state,
                clean_hist, actions, frame_positions,
                p_encode,
                scheduler, num_inference_steps,
                n_hist, ar_chunk_size, ar_num_chunks,
                step_rng, weights_dtype,
                mesh=pipeline.mesh,
                logical_axis_rules=config.logical_axis_rules,
                guidance_scale=guidance_scale,
                cond_tokens_per_frame=action_tokens_per_frame,
                action_cond_mode=action_cond_mode,
            )
        else:
            # Encode actions (constant across denoising steps).
            action_tokens = p_encode(
                params, actions=actions,
                F_lat=F_lat,
            )  # (1, F_lat*K, 4096)

            pred_future = run_denoising(
                graphdef, params, rest_of_state,
                clean_hist, n_fut, action_tokens, frame_positions,
                scheduler, num_inference_steps,
                n_hist, step_rng, weights_dtype,
                mesh=pipeline.mesh,
                logical_axis_rules=config.logical_axis_rules,
                guidance_scale=guidance_scale,
                cond_tokens_per_frame=action_tokens_per_frame,
                action_cond_mode=action_cond_mode,
            )

        # Decode GT and predicted full sequences.
        gt_full = jnp.concatenate([clean_hist, gt_future], axis=2)
        pred_full = jnp.concatenate([clean_hist, pred_future], axis=2)

        gt_temporal_std = float(jnp.std(gt_future, axis=2).mean())
        pred_temporal_std = float(jnp.std(pred_future, axis=2).mean())
        max_logging.log(
            f"[wan_ctrl_world_infer] vid {vid_idx}: "
            f"GT future temporal_std={gt_temporal_std:.4f}  "
            f"pred future temporal_std={pred_temporal_std:.4f}"
        )

        gt_video = _decode_latents(pipeline, gt_full.astype(jnp.float32))
        pred_video = _decode_latents(pipeline, pred_full.astype(jnp.float32))

        # Drop the decoded history context: the n_hist history latents decode to
        # 1 + 4*(n_hist - 1) leading frames (causal VAE: latent 0 → 1 frame,
        # each later latent → 4 frames). They were only prepended so the first
        # future latent decodes as a continuation rather than a first chunk.
        n_ctx_frames = 1 + 4 * (n_hist - 1)
        gt_video = gt_video[:, n_ctx_frames:]
        pred_video = pred_video[:, n_ctx_frames:]

        path = os.path.join(output_dir, f"video_{vid_idx:04d}.mp4")
        _save_comparison_video(gt_video, pred_video, path, fps=fps)
        max_logging.log(f"[wan_ctrl_world_infer] saved {path}")

    max_logging.log("[wan_ctrl_world_infer] done.")


if __name__ == "__main__":
    app.run(run)
