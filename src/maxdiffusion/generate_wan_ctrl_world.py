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
import time
from typing import Sequence

from maxdiffusion.utils import export_to_video
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.models.wan.action_encoder_wan import (
    NNXWanActionEncoder,
    NNXWanActionAdaLNProjector,
    NNXWanSkeletonPatchEmbed,
)
from maxdiffusion.pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2
from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
from maxdiffusion.trainers.wan_ctrl_world_trainer import (
    VALID_ACTION_COND_MODES,
    WanCtrlWorldModel,
    _build_per_token_timestep,
    _dtype,
    _encode_skeleton,
    _frame_level_cond,
    _group_actions,
    _placeholder_action_tokens,
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
    skeleton_tokens: jnp.ndarray | None = None,
):
    """One Euler flow-matching step with optional classifier-free guidance.

    When guidance_scale > 1, runs a single double-batched forward pass with
    [uncond | cond] stacked along the batch axis, then blends:
        pred = uncond + guidance_scale * (cond - uncond)

    ``skeleton_tokens`` is the already-patch-embedded, already-alpha-scaled
    skeleton bias for ``skeleton`` mode (``None`` in the action modes, which jit
    traces as an absent pytree so those modes are untouched). The uncond half of
    the guided batch gets zeros there, which is exactly the "skip the add" state
    training's CFG mask produces.

    Returns the updated latents (clean history re-attached).
    """
    model: WanCtrlWorldModel = nnx.merge(graphdef, params, rest_of_state)
    b, _, F_lat, H_lat, W_lat = latents.shape
    t_batch = jnp.broadcast_to(timestep, (b,))
    timestep_2d = _build_per_token_timestep(t_batch, F_lat, H_lat, W_lat, n_hist)

    def _route(tokens):
        # text_tokens=None: this script is action-only. run_wan_ctrl_world_inference
        # refuses to start on a use_task_instructions=True config rather than
        # silently evaluating a text-trained checkpoint without its instruction.
        return _route_action_conditioning(
            tokens, model.action_adaln_proj, action_cond_mode,
            cond_tokens_per_frame, H_lat, W_lat, text_tokens=None,
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
        skeleton_2x = (
            jnp.concatenate([jnp.zeros_like(skeleton_tokens), skeleton_tokens], axis=0)
            if skeleton_tokens is not None else None
        )
        pred_2x = model.transformer(
            hidden_states=latents_2x,
            timestep=t_2d,
            encoder_hidden_states=tokens_2x,
            action_hidden_states=action_hidden_states_2x,
            skeleton_hidden_states=skeleton_2x,
            deterministic=True,
            frame_level_cond=_frame_level_cond(action_cond_mode),
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
            skeleton_hidden_states=skeleton_tokens,
            deterministic=True,
            frame_level_cond=_frame_level_cond(action_cond_mode),
            cond_tokens_per_frame=cond_tokens_per_frame,
            frame_positions=frame_positions,
        )

    # Apply scheduler step to future frames only; restore clean history after.
    future_pred = model_pred[:, :, n_hist:]
    future_latents = latents[:, :, n_hist:]
    output = scheduler.step(scheduler_state, future_pred, timestep, future_latents)
    return jnp.concatenate([clean_hist, output.prev_sample], axis=2)


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


def _encode_skeleton_tokens(
    params: nnx.State,
    graphdef: nnx.GraphDef,
    rest_of_state: nnx.State,
    skeleton: jnp.ndarray,
) -> jnp.ndarray:
    """Patch-embed skeleton latents into the video-token bias (skeleton mode).

    ``(B, C, F_lat, H_lat, W_lat)`` → ``(B, seq_len, inner_dim)``, alpha applied.
    No CFG mask here: the uncond branch is formed inside ``_denoise_step``, which
    zeroes this tensor for the uncond half of the double batch.
    """
    model: WanCtrlWorldModel = nnx.merge(graphdef, params, rest_of_state)
    return _encode_skeleton(model.skeleton_embed, skeleton, None, 0.0, skeleton.dtype)


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
    all_skeleton: jnp.ndarray | None = None,
    p_encode_skeleton=None,
    wan_text_dim: int = 4096,
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

    In ``skeleton`` mode ``all_actions`` carries no information (the encoder does
    not exist) and ``all_skeleton`` — ``(B, C, n_hist + ar_num_chunks *
    ar_chunk_size, H, W)`` skeleton latents for the whole window — is the
    conditioning instead. It is sliced per chunk on the *frame* axis with the same
    ``[pos_start, pos_start + window_F_lat)`` window as ``all_frame_positions``,
    since skeleton latents are per-latent-frame just like the RoPE positions.

    Returns predicted future latents: (B, C, ar_num_chunks * ar_chunk_size, H, W).
    """
    if action_cond_mode == "skeleton" and (all_skeleton is None or p_encode_skeleton is None):
        raise ValueError(
            "action_cond_mode='skeleton' needs all_skeleton and p_encode_skeleton; "
            "without them the rollout would run entirely unconditioned."
        )
    window_F_lat = n_hist + ar_chunk_size
    current_hist = initial_hist
    generated_chunks = []

    global _FIRST_CHUNK
    for chunk_i in range(ar_num_chunks):
        t_chunk = time.perf_counter()
        # Slice the actions that correspond to the current window.
        # Window covers latent frames [chunk_i*ar_chunk_size,
        #                              chunk_i*ar_chunk_size + window_F_lat).
        act_start = 4 * chunk_i * ar_chunk_size
        act_end   = 4 * (chunk_i * ar_chunk_size + window_F_lat)
        actions_chunk = all_actions[:, act_start:act_end, :]  # (B, 4*window_F_lat, 7)

        pos_start = chunk_i * ar_chunk_size
        positions_chunk = all_frame_positions[:, pos_start:pos_start + window_F_lat]

        if action_cond_mode == "skeleton":
            # No action encoder in this mode — the zero placeholder just fills the
            # cross-attention K/V slot. The real conditioning is the skeleton
            # window, sliced on the frame axis exactly like positions_chunk.
            action_tokens = _placeholder_action_tokens(
                all_actions.shape[0], window_F_lat, cond_tokens_per_frame,
                wan_text_dim, dtype,
            )
            skeleton_chunk = all_skeleton[:, :, pos_start:pos_start + window_F_lat]
            skeleton_tokens = p_encode_skeleton(params, skeleton=skeleton_chunk)
        else:
            action_tokens = p_encode(params, actions=actions_chunk, F_lat=window_F_lat)
            skeleton_tokens = None

        rng, step_rng = jax.random.split(rng)
        gen_chunk = run_denoising(
            graphdef, params, rest_of_state,
            current_hist, ar_chunk_size, action_tokens, positions_chunk,
            scheduler, num_inference_steps, n_hist,
            step_rng, dtype, mesh, logical_axis_rules,
            guidance_scale=guidance_scale,
            cond_tokens_per_frame=cond_tokens_per_frame,
            action_cond_mode=action_cond_mode,
            skeleton_tokens=skeleton_tokens,
        )  # (B, C, ar_chunk_size, H, W)

        generated_chunks.append(gen_chunk)

        # Roll history forward: take last n_hist frames from [hist | generated].
        full_window = jnp.concatenate([current_hist, gen_chunk], axis=2)
        current_hist = full_window[:, :, -n_hist:, :, :]

        # Timed after the history roll so the number covers the whole chunk, and
        # block first — the denoise loop is async, so without this we would be
        # timing dispatch rather than execution.
        current_hist.block_until_ready()
        frame_lo = chunk_i * ar_chunk_size + 1
        max_logging.log(
            f"[wan_ctrl_world_infer]   AR chunk {chunk_i + 1}/{ar_num_chunks}: "
            f"latent frames {frame_lo}-{frame_lo + ar_chunk_size - 1} "
            f"({num_inference_steps} denoise steps) in "
            f"{time.perf_counter() - t_chunk:.1f}s"
            + ("  [includes one-time compile]" if _FIRST_CHUNK else "")
        )
        _FIRST_CHUNK = False

    return jnp.concatenate(generated_chunks, axis=2)  # (B, C, total_fut, H, W)


_DENOISE_STEP_CACHE: dict = {}

# Flipped after the first AR chunk finishes, so exactly one log line is marked as
# carrying the one-time XLA compile. Everything after it is steady state — the
# gap between the two is what tells you the sampler cache is working.
_FIRST_CHUNK = True


def _get_denoise_step(
    *,
    graphdef,
    rest_of_state,
    n_hist: int,
    guidance_scale: float,
    scheduler,
    cond_tokens_per_frame: int,
    action_cond_mode: str,
):
    """``jax.jit(_denoise_step)`` memoised on its static signature.

    This used to be built inside ``run_denoising``, which ``run_ar_denoising``
    calls once per AR chunk — so every chunk got a fresh ``jax.jit`` object with
    an empty compilation cache, and the whole transformer recompiled per chunk.
    The per-chunk arrays (``clean_hist``, ``action_tokens``, ``frame_positions``,
    ``scheduler_state``) were also bound into the partial, making them
    compile-time constants, so even a reused jit object would have recompiled
    whenever their values changed.

    Bound here: only values that must be Python-static (``guidance_scale`` gates
    a branch, ``n_hist`` drives slicing, ``graphdef``/``scheduler`` are objects)
    plus ``rest_of_state``, which is one fixed tree for the whole run. Keyed by
    ``id()`` for the unhashable objects — they outlive the process's use of this
    cache, so identity is a safe key.
    """
    key = (
        id(graphdef),
        id(rest_of_state),
        id(scheduler),
        int(n_hist),
        float(guidance_scale),
        int(cond_tokens_per_frame),
        action_cond_mode,
    )
    fn = _DENOISE_STEP_CACHE.get(key)
    if fn is None:
        fn = jax.jit(
            functools.partial(
                _denoise_step,
                graphdef=graphdef,
                rest_of_state=rest_of_state,
                n_hist=n_hist,
                guidance_scale=guidance_scale,
                scheduler=scheduler,
                cond_tokens_per_frame=cond_tokens_per_frame,
                action_cond_mode=action_cond_mode,
            )
        )
        _DENOISE_STEP_CACHE[key] = fn
    return fn


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
    skeleton_tokens: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Denoise future latent frames from random noise, conditioned on history + actions.

    frame_positions (B, n_hist + n_fut) int32 holds the per-frame temporal RoPE
    indices for the window, matching what the dataset fed the model at training
    time (absolute episode frame indices, history clipped/repeated at episode
    start).

    When guidance_scale > 1, uses classifier-free guidance: the model runs with
    the conditioning zeroed (unconditioned) and present (conditioned) in a single
    double-batched forward pass per step, then blends the predictions. In
    ``skeleton`` mode the conditioning being dropped is ``skeleton_tokens``; the
    action tokens are the zero placeholder in both halves.

    Returns the denoised future latents: (B, C, n_fut, H, W).
    """
    b, C, _, H, W = clean_hist.shape
    noisy_future = jax.random.normal(rng, (b, C, n_fut, H, W), dtype=dtype)
    latents = jnp.concatenate([clean_hist, noisy_future], axis=2)

    sched_state = scheduler.set_timesteps(
        scheduler.create_state(), num_inference_steps=num_inference_steps, training=False
    )

    uncond_action_tokens = jnp.zeros_like(action_tokens)

    # Only the genuinely static arguments are bound here; every per-chunk array
    # (clean_hist / action_tokens / frame_positions / scheduler_state) is passed
    # as a traced argument below. See _get_denoise_step for why.
    p_step = _get_denoise_step(
        graphdef=graphdef,
        rest_of_state=rest_of_state,
        n_hist=n_hist,
        guidance_scale=guidance_scale,
        scheduler=scheduler,
        cond_tokens_per_frame=cond_tokens_per_frame,
        action_cond_mode=action_cond_mode,
    )

    timesteps_np = np.array(sched_state.timesteps)
    with mesh, nn_partitioning.axis_rules(logical_axis_rules):
      for step_i, t in enumerate(timesteps_np):
          latents = p_step(
              params,
              latents=latents,
              timestep=jnp.array(t),
              clean_hist=clean_hist,
              action_tokens=action_tokens,
              uncond_action_tokens=uncond_action_tokens,
              frame_positions=frame_positions,
              scheduler_state=sched_state,
              skeleton_tokens=skeleton_tokens,
          )
          if step_i == 0 or step_i == len(timesteps_np) - 1 or (step_i + 1) % 10 == 0:
              max_logging.log(
                  f"  denoise step {step_i + 1}/{len(timesteps_np)} t={t:.1f}"
              )

    return latents[:, :, n_hist:]   # (B, C, n_fut, H, W)


# ── Video helpers ─────────────────────────────────────────────────────────────


def _decode_latents(
    pipeline: WanPipelineTI2V_2_2, latents: jnp.ndarray, num_views: int = 3
) -> list:
    """Decode (B, C, F, H*num_views, W) latents → list of (B, F, H, W, 3) per cam.

    The cameras were encoded separately and stacked along H, so we split and
    decode each independently. Returned per view rather than re-stacked: the
    callers want them apart (one MP4 per camera) and the tiled comparison is a
    cheap ``_tile_views`` away, whereas stacking here only to re-split downstream
    forced every consumer to know the packing.

    Latents are stored normalized ((raw - mean) / std) by the data preprocessing
    script, so we denormalize before passing to the VAE decoder.
    """
    H = latents.shape[3] // num_views
    cam_videos = []
    for i in range(num_views):
        cam = latents[:, :, :, i * H:(i + 1) * H, :]        # (B, C, F, H, W)
        cam = pipeline._denormalize_latents(cam)
        cam_videos.append(pipeline._decode_latents_to_video(cam))     # (B, F, H, W, 3)
    return cam_videos


def _tile_views(views: list) -> np.ndarray:
    """Per-view ``(B, F, H, W, 3)`` (B=1) → ``(F, H, num_views*W, 3)`` side by side."""
    return np.concatenate([np.clip(v[0], 0.0, 1.0) for v in views], axis=2)


def _save_comparison_video(
    gt_tile: np.ndarray,
    pred_tile: np.ndarray,
    path: str,
    fps: int = 8,
) -> None:
    """Write GT (top half) / predicted (bottom half) stacked MP4.

    Both inputs are already-tiled ``(F, H, num_views*W, 3)`` frames in [0, 1] —
    see ``_tile_views``.
    """
    combined = np.concatenate([gt_tile, pred_tile], axis=1)  # (F, H*2, num_views*W, 3)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    export_to_video(list(combined), path, fps=fps)


def _save_per_view_videos(
    pred_views: list,
    output_dir: str,
    name: str,
    fps: int = 8,
) -> list[str]:
    """Write the *predicted* video once per camera, into ``output_dir/cam{i}/``.

    ``pred_views`` is the per-camera list ``_decode_latents`` returns, already
    trimmed by the caller so these match the prediction half of the tiled
    comparison frame for frame. GT is deliberately not split out: it is already
    visible in the tiled video, and per-view copies of it would triple the file
    count for no extra information.

    One directory per camera (rather than a ``_cam{i}`` filename suffix) so a
    single camera's whole sweep is one glob, and ``name`` stays identical across
    directories for easy pairing.
    """
    paths = []
    for i, view in enumerate(pred_views):
        cam_dir = os.path.join(output_dir, f"cam{i}")
        os.makedirs(cam_dir, exist_ok=True)
        p = os.path.join(cam_dir, f"{name}.mp4")
        export_to_video(list(np.clip(view[0], 0.0, 1.0)), p, fps=fps)
        paths.append(p)
    return paths


# ── Main ──────────────────────────────────────────────────────────────────────


def run(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config

    if max_utils.config_get(config, "use_task_instructions", False):
        # This script never loads text_embeds from the dataset, so it can only
        # produce action-only rollouts. Running a checkpoint that was trained
        # with the instruction would put the model out of distribution and
        # quietly understate its quality, so refuse instead.
        raise NotImplementedError(
            "generate_wan_ctrl_world.py does not support use_task_instructions=True: "
            "it does not read text_embeds from the dataset, so the rollout would be "
            "action-only while the checkpoint expects an instruction. Either evaluate "
            "with use_task_instructions=False (matching an action-only checkpoint) or "
            "extend this script to thread text through _encode_actions and "
            "run_denoising, mirroring _text_routes/_add_text_bias in "
            "wan_ctrl_world_trainer.py."
        )

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
    if action_cond_mode not in VALID_ACTION_COND_MODES:
        raise ValueError(
            f"action_cond_mode={action_cond_mode!r} is not one of {VALID_ACTION_COND_MODES}."
        )
    skeleton_mode = action_cond_mode == "skeleton"
    # The module set must match what the training run checkpointed exactly, or the
    # orbax restore below hits a structure mismatch — skeleton-mode checkpoints
    # have skeleton_embed and no action_encoder at all.
    action_encoder = None if skeleton_mode else NNXWanActionEncoder(
        rngs=nnx.Rngs(jax.random.key(config.seed)),
        action_dim=config.action_dim,
        num_actions=4,
        hidden_dim=config.wan_action_encoder_hidden_dim,
        out_dim=config.wan_text_dim,
        tokens_per_frame=action_tokens_per_frame,
        dtype=weights_dtype,
        weights_dtype=weights_dtype,
    )
    skeleton_embed = None
    if skeleton_mode:
        skeleton_embed = NNXWanSkeletonPatchEmbed(
            rngs=nnx.Rngs(jax.random.key(config.seed + 2)),
            in_channels=pipeline.transformer.config.in_channels,
            inner_dim=(
                pipeline.transformer.config.num_attention_heads
                * pipeline.transformer.config.attention_head_dim
            ),
            patch_size=tuple(pipeline.transformer.config.patch_size),
            alpha=float(getattr(config, "skeleton_embed_alpha", 0.1)),
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
        combined = WanCtrlWorldModel(
            pipeline.transformer, action_encoder, action_adaln_proj, skeleton_embed
        )
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
    # Without padding, only episodes at least max_latent_frames long survive the
    # dataset's length filter — at a 57-frame window that is 8 of 110 val
    # episodes, since the median val episode is 18 latent frames. Padding keeps
    # the short ones and repeats their last action for the remainder of the
    # rollout; their GT track freezes at the last real frame.
    pad_short = bool(max_utils.config_get(config, "eval_pad_short_episodes", False))
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
        load_skeleton=skeleton_mode,
        first_window_only=autoregressive,
        pad_short_episodes=pad_short and autoregressive,
        min_latent_frames=int(max_utils.config_get(config, "eval_min_latent_frames", 0)),
    )
    if pad_short and autoregressive:
        max_logging.log(
            f"[wan_ctrl_world_infer] padding short episodes: keeping any episode "
            f">= {dataset.min_traj_len} latent frames, repeating the last action "
            f"out to the {max_latent_frames}-frame window"
        )

    # ── Precompile conditioning encoders ──────────────────────────────────────
    p_encode = None if skeleton_mode else jax.jit(
        functools.partial(
            _encode_actions,
            graphdef=graphdef,
            rest_of_state=rest_of_state,
        ),
        static_argnames=["F_lat"],
    )
    p_encode_skeleton = jax.jit(
        functools.partial(
            _encode_skeleton_tokens,
            graphdef=graphdef,
            rest_of_state=rest_of_state,
        ),
    ) if skeleton_mode else None

    # ── Inference loop ────────────────────────────────────────────────────────
    output_dir = os.path.join(config.output_dir, "inference_videos")
    rng = jax.random.key(config.seed + 42)
    t_sweep = time.perf_counter()

    for vid_idx, batch in enumerate(dataset):
        max_logging.log(f"[wan_ctrl_world_infer] ── sample {vid_idx + 1}...")
        t_vid = time.perf_counter()

        latent = jnp.array(batch["latent"]).astype(weights_dtype)       # (1, C, W, H, Wl)
        actions = jnp.array(batch["action"]).astype(weights_dtype)      # (1, 4*W, 7)
        # Temporal RoPE indices for every latent frame in the window — same
        # tensor training feeds the transformer (see wan_ctrl_world_trainer).
        frame_positions = jnp.array(batch["frame_positions"]).astype(jnp.int32)  # (1, W)
        # Skeleton latents for the whole window, same layout as `latent`.
        skeleton = (
            jnp.array(batch["skeleton"]).astype(weights_dtype) if skeleton_mode else None
        )

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
                all_skeleton=skeleton,
                p_encode_skeleton=p_encode_skeleton,
                wan_text_dim=config.wan_text_dim,
            )
        else:
            # Encode the conditioning once — constant across denoising steps.
            if skeleton_mode:
                action_tokens = _placeholder_action_tokens(
                    latent.shape[0], F_lat, action_tokens_per_frame,
                    config.wan_text_dim, weights_dtype,
                )
                skeleton_tokens = p_encode_skeleton(params, skeleton=skeleton)
            else:
                action_tokens = p_encode(
                    params, actions=actions,
                    F_lat=F_lat,
                )  # (1, F_lat*K, 4096)
                skeleton_tokens = None

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
                skeleton_tokens=skeleton_tokens,
            )

        rollout_s = time.perf_counter() - t_vid

        # Decode GT and predicted full sequences.
        gt_full = jnp.concatenate([clean_hist, gt_future], axis=2)
        pred_full = jnp.concatenate([clean_hist, pred_future], axis=2)

        # With padding on, the tail of the window is the last real latent/action
        # repeated — GT is frozen there and the prediction is rolling on filler.
        # n_real_fut is how many future latents are backed by real episode frames.
        n_real_fut = n_fut
        if "n_real_frames" in batch:
            n_real_fut = max(1, int(np.asarray(batch["n_real_frames"]).reshape(-1)[0]) - n_hist)
            n_real_fut = min(n_real_fut, n_fut)
        if n_real_fut < n_fut:
            max_logging.log(
                f"[wan_ctrl_world_infer] vid {vid_idx}: episode ends early — "
                f"keeping {n_real_fut}/{n_fut} future latents, trimming the "
                f"{n_fut - n_real_fut} padded ones from the video"
            )

        t_dec = time.perf_counter()
        # Per-camera lists of (B, F, H, W, 3); frames live on axis 1.
        gt_views = _decode_latents(pipeline, gt_full.astype(jnp.float32))
        pred_views = _decode_latents(pipeline, pred_full.astype(jnp.float32))
        decode_s = time.perf_counter() - t_dec

        # Drop the decoded history context: the n_hist history latents decode to
        # 1 + 4*(n_hist - 1) leading frames (causal VAE: latent 0 → 1 frame,
        # each later latent → 4 frames). They were only prepended so the first
        # future latent decodes as a continuation rather than a first chunk.
        n_ctx_frames = 1 + 4 * (n_hist - 1)
        gt_views = [v[:, n_ctx_frames:] for v in gt_views]
        pred_views = [v[:, n_ctx_frames:] for v in pred_views]

        # Trim padding back off: every future latent sits past latent 0, so each
        # decodes to 4 frames and the real span is the first 4*n_real_fut frames.
        # Without this a 2-frame episode padded out to the full window would be
        # written as a mostly-frozen GT against a prediction that keeps rolling.
        if n_real_fut < n_fut:
            keep_frames = min(4 * n_real_fut, gt_views[0].shape[1])
            gt_views = [v[:, :keep_frames] for v in gt_views]
            pred_views = [v[:, :keep_frames] for v in pred_views]

        gt_video = _tile_views(gt_views)
        pred_video = _tile_views(pred_views)

        # tiled/ holds the GT-vs-pred comparison; cam{i}/ holds the predicted
        # rollout for one camera. Same basename in every directory.
        name = f"video_{vid_idx:04d}"
        path = os.path.join(output_dir, "tiled", f"{name}.mp4")
        _save_comparison_video(gt_video, pred_video, path, fps=fps)
        view_paths = _save_per_view_videos(pred_views, output_dir, name, fps=fps)
        n_done = vid_idx + 1
        elapsed = time.perf_counter() - t_sweep
        max_logging.log(
            f"[wan_ctrl_world_infer]   saved {name}.mp4 -> tiled/ + "
            f"{'/'.join(os.path.basename(os.path.dirname(p)) for p in view_paths)}  "
            f"{pred_video.shape[0]} frames @ {fps}fps, "
            f"tile {pred_video.shape[2]}x{pred_video.shape[1] * 2}"
        )
        max_logging.log(
            f"[wan_ctrl_world_infer]   sample {vid_idx + 1} done in "
            f"{rollout_s + decode_s:.1f}s (rollout {rollout_s:.1f}s, "
            f"decode {decode_s:.1f}s)  |  {n_done} samples, "
            f"{elapsed / 60:.1f}min elapsed, {elapsed / n_done:.1f}s/sample avg"
        )

    max_logging.log("[wan_ctrl_world_infer] done.")


if __name__ == "__main__":
    app.run(run)
