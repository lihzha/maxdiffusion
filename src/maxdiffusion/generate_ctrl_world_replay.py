"""JAX / Flax trajectory-replay driver for the action-conditioned SVD world model.

Mirrors ``Ctrl-World/scripts/rollout_replay_traj.py`` so the two implementations
can be run on identical inputs and their outputs compared side by side.

High-level flow (one trajectory):
  1. Load annotation JSON from ``{val_dataset_dir}/annotation/val/{id}.json``
     and the 3 paired exterior videos.
  2. Slice ``pred_step * interact_num + 8`` frames starting at ``start_idx``;
     VAE-encode each view → per-view latents; stack the 3 views along the
     height axis to form the ``(T, 4, 72, 40)`` Ctrl-World latent.
  3. Seed the history buffer with the first stacked latent repeated
     ``num_history * 4`` times (Ctrl-World's warm-up hack — indices -8/-6/-4/-2
     all land on the same frame at step 0).
  4. For ``i in range(interact_num)`` run the Flax pipeline with:
        action   = [history_poses @ HISTORY_IDX, ground-truth actions from
                    ``eef_gt[start_id:end_id]``]               (1, 11, 7)
        image    = last stacked latent in the buffer             (1, 4, 72, 40)
        history  = cat(his_cond @ HISTORY_IDX)                   (1, 6, 4, 72, 40)
        text     = CLIP text embeddings of ``instruction``       (1, 512) if text_cond
     Decode predicted + ground-truth latents per view, stack gt-top / pred-bottom
     into a single MP4 per trajectory and write to ``save_dir/task_name/video/``.

The Flax pipeline (``FlaxCtrlWorldPipeline`` in pipelines/svd/pipeline_flax_ctrl_world.py)
takes ``image_latent`` pre-scaled by ``vae.scaling_factor`` and unscales
internally, so the buffer is maintained in scaled space throughout.

``text_cond``: the CLIP text tower is loaded as a torch CPU model (same shim
pattern as ``_TorchCLIPVisionWithProjection`` in the SVD checkpointer). The
ctrl-world training froze the text encoder, so stock
``openai/clip-vit-base-patch32`` weights are bit-identical to what the
torch checkpoint contains.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio
from jax.sharding import Mesh
from safetensors.torch import load_file as load_torch_safetensors

from maxdiffusion import max_logging
from maxdiffusion.models.svd.video_unet_flax import FlaxVideoUNet
from maxdiffusion.models.svd.video_autoencoder_flax import FlaxSVDAutoencoderKL
from maxdiffusion.models.svd.action_encoder_flax import FlaxActionEncoder
from maxdiffusion.schedulers.scheduling_edm_euler_flax import FlaxEDMEulerScheduler
from maxdiffusion.pipelines.svd.pipeline_flax_ctrl_world import FlaxCtrlWorldPipeline


# ─── constants (match Ctrl-World/config.py task_type='replay') ─────────────────

# Six-slot history selection relative to the rolling ``his_*`` buffers. The two
# leading 0s make the model see the current-step latent twice; -8/-6/-4/-2 reach
# back 2s–4s at 5 Hz. Matches rollout_replay_traj.py line 300.
_HISTORY_IDX = (0, 0, -8, -6, -4, -2)

# Stacked-view geometry: Ctrl-World concatenates 3 exterior views vertically in
# latent space. Each view is 192×320 px / 24×40 latent; 3 stacked → 576×320 / 72×40.
_NUM_VIEWS = 3

# Default replay trajectory set (config.py task_type='replay' branch).
_DEFAULT_VAL_IDS = ("899", "18599", "199")
_DEFAULT_START_IDX = (8, 14, 8)


# ─── helpers ───────────────────────────────────────────────────────────────────


def _normalize_bound(
    data: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Mirror of Ctrl-World ``agent.normalize_bound`` (per-dim min-max to [-1, 1])."""
    n = 2.0 * (data - data_min) / (data_max - data_min + eps) - 1.0
    return np.clip(n, clip_min, clip_max)


def _read_video_frames(path: str, frame_ids: np.ndarray) -> np.ndarray:
    """Return ``(T, H, W, 3)`` uint8 frames at the requested frame ids.

    ``imiter`` streams frames sequentially, which avoids the O(N) seeks that a
    per-frame ``imread`` would issue. We walk the video once and pick off the
    requested ids; ids are de-duplicated then restored to the original order.
    """
    wanted = set(int(i) for i in frame_ids)
    max_idx = int(frame_ids.max())
    picked: dict[int, np.ndarray] = {}
    for i, frame in enumerate(iio.imiter(path)):
        if i in wanted:
            picked[i] = np.asarray(frame)
            wanted.discard(i)
        if i >= max_idx and not wanted:
            break
    if wanted:
        raise RuntimeError(f"{path}: missing frames {sorted(wanted)} (video ended at {i})")
    return np.stack([picked[int(i)] for i in frame_ids], axis=0)


class _TorchCLIPTextShim:
    """Stock HF CLIPTextModelWithProjection on CPU, projecting to 512-dim text embeds.

    Ctrl-World froze its text encoder during training, so openai/clip-vit-base-patch32
    ``text_embeds`` are bit-identical to the ones encoded by the tensors baked into
    ``checkpoint-10000.pt`` (we just don't bother porting them to Flax — the tower
    runs once per trajectory).
    """

    def __init__(self, clip_path: str):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, use_fast=False)
        self.model = (
            CLIPTextModelWithProjection.from_pretrained(clip_path).eval().to("cpu")
        )

    def __call__(self, texts: Sequence[str]) -> jnp.ndarray:
        import torch

        with torch.no_grad():
            inputs = self.tokenizer(
                list(texts),
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            out = self.model(**{k: v for k, v in inputs.items()})
        return jnp.asarray(out.text_embeds.numpy())


def _load_action_encoder_params(path: str, dtype: jnp.dtype):
    """Load the 3-layer MLP weights from the converter's safetensors dump.

    PyTorch ``nn.Linear.weight`` is ``(out, in)``; Flax ``Dense.kernel`` is
    ``(in, out)``. The biases share layout. Naming map: ``action_encode.0`` →
    ``linear_1``, ``.2`` → ``linear_2``, ``.4`` → ``linear_3`` (the even indices
    are the torch ``nn.Sequential`` Linears; the odd indices were SiLU
    activations with no parameters).
    """
    pt = load_torch_safetensors(path)

    def _k(t):
        return jnp.asarray(t.numpy().T, dtype=dtype)

    def _b(t):
        return jnp.asarray(t.numpy(), dtype=dtype)

    return {
        "linear_1": {
            "kernel": _k(pt["action_encode.0.weight"]),
            "bias": _b(pt["action_encode.0.bias"]),
        },
        "linear_2": {
            "kernel": _k(pt["action_encode.2.weight"]),
            "bias": _b(pt["action_encode.2.bias"]),
        },
        "linear_3": {
            "kernel": _k(pt["action_encode.4.weight"]),
            "bias": _b(pt["action_encode.4.bias"]),
        },
    }


def _make_single_gpu_mesh() -> Mesh:
    """1×1×1×1 mesh matching maxdiffusion's 4-axis convention.

    Sharding constraints inside FlaxVideoUNet use logical axis names that resolve
    to these physical axes; with every axis of size 1, every constraint degenerates
    to ``P()`` (replicated) and we never actually shard on a single GPU.
    """
    devices = np.array(jax.devices()[:1]).reshape((1, 1, 1, 1))
    return Mesh(devices, ("data", "fsdp", "context", "tensor"))


def _build_jit_vae_encode(vae):
    """Return a jitted VAE encoder; compiles once per unique batch shape.

    Without the ``jit`` wrapper, every per-chunk encode pays tens of ms of
    JAX→XLA dispatch overhead per op; with ~hundreds of ops in the encoder,
    eager mode adds seconds of dead time per chunk that don't exist in torch.
    Closure-captures the scaling factor so we can return already-scaled latents.
    """
    scale = vae.config.scaling_factor

    @jax.jit
    def _enc(params, pixels_nchw):
        post = vae.apply(
            {"params": params},
            pixels_nchw,
            deterministic=True,
            method=vae.encode,
        )
        latent = post.latent_dist.mode()  # NHWC
        latent = jnp.transpose(latent, (0, 3, 1, 2))  # NHWC → NCHW
        return latent * scale

    return _enc


def _build_jit_vae_decode(vae):
    """Return a jitted VAE decoder; ``num_frames`` is static (static_argnames).

    Same motivation as ``_build_jit_vae_encode``; additionally ``num_frames`` is
    a Python int that controls the 3-D temporal decoder's frame count, so we
    declare it static — one compiled variant per distinct chunk length.
    """
    scale = vae.config.scaling_factor

    @functools.partial(jax.jit, static_argnames=("num_frames",))
    def _dec(params, latents_scaled, num_frames):
        unscaled = latents_scaled / scale
        dec = vae.apply(
            {"params": params},
            unscaled,
            num_frames=num_frames,
            deterministic=True,
            method=vae.decode,
        )
        frames = dec.sample  # NCHW in [-1, 1]
        return (frames / 2.0 + 0.5).clip(0.0, 1.0)

    return _dec


def _vae_encode_frames(
    jit_encode,
    vae_params,
    frames_uint8: np.ndarray,
    mesh: Mesh,
    dtype: jnp.dtype,
    chunk: int = 8,
) -> jnp.ndarray:
    """``(T, H, W, 3) uint8`` → ``(T, 4, H/8, W/8)`` scaled latent (NCHW).

    Pads the final chunk up to ``chunk`` frames so we only ever compile one
    encoder variant (otherwise the tail gives a second, one-shot shape).
    """
    pixels = (frames_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0
    pixels = np.transpose(pixels, (0, 3, 1, 2))  # NCHW for the Flax encoder API
    pixels = jnp.asarray(pixels, dtype=dtype)
    out = []
    for i in range(0, pixels.shape[0], chunk):
        batch = pixels[i : i + chunk]
        real = batch.shape[0]
        if real < chunk:
            # Pad with zeros then drop the pad outputs — keeps the jit cache warm.
            pad = jnp.zeros((chunk - real,) + batch.shape[1:], dtype=batch.dtype)
            batch = jnp.concatenate([batch, pad], axis=0)
        with mesh:
            latent = jit_encode(vae_params, batch)
        if real < chunk:
            latent = latent[:real]
        out.append(latent)
    return jnp.concatenate(out, axis=0)


def _vae_decode_chunks(
    jit_decode,
    vae_params,
    latents_scaled: jnp.ndarray,
    mesh: Mesh,
    chunk_size: int = 7,
) -> np.ndarray:
    """``(T, 4, H/8, W/8)`` scaled latent → ``(T, H, W, 3)`` uint8 frames.

    Matches the torch rollout's decode loop one-for-one: decode in
    ``chunk_size`` slices, pass ``num_frames=chunk.shape[0]`` per chunk.
    Each distinct chunk length costs one compile (the last tail chunk may
    differ) — static_argnames on ``num_frames`` caches them.
    """
    total = latents_scaled.shape[0]
    out = []
    for i in range(0, total, chunk_size):
        chunk = latents_scaled[i : i + chunk_size]
        with mesh:
            frames = jit_decode(vae_params, chunk, chunk.shape[0])
        out.append(frames)
    frames = jnp.concatenate(out, axis=0)
    frames = jnp.transpose(frames, (0, 2, 3, 1))  # NCHW → NHWC
    return (np.asarray(frames) * 255.0).astype(np.uint8)


# ─── main replay loop ──────────────────────────────────────────────────────────


def run_replay(args) -> None:
    dtype = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }[args.dtype]
    weights_dtype = dtype

    if not os.path.isdir(args.ctrl_world_dir):
        raise FileNotFoundError(
            f"--ctrl_world_dir {args.ctrl_world_dir} not found. Run "
            "scripts/convert_ctrl_world_ckpt.py first."
        )

    mesh = _make_single_gpu_mesh()

    max_logging.log(f"[mcw] loading UNet from {args.ctrl_world_dir}/unet ...")
    t_load = time.perf_counter()
    with mesh:
        unet, unet_params = FlaxVideoUNet.from_pretrained(
            args.ctrl_world_dir,
            subfolder="unet",
            dtype=dtype,
            weights_dtype=weights_dtype,
            from_pt=True,
            use_safetensors=True,
            attention_kernel="dot_product",
            temporal_attention_kernel="dot_product",
            use_memory_efficient_attention=True,
            flash_block_sizes={},
            flash_min_seq_length=4096,
            mesh=mesh,
            precision=jax.lax.Precision.DEFAULT,
            norm_num_groups=32,
        )
        max_logging.log(f"[mcw] loading VAE from {args.ctrl_world_dir}/vae ...")
        vae, vae_params = FlaxSVDAutoencoderKL.from_pretrained(
            args.ctrl_world_dir,
            subfolder="vae",
            dtype=dtype,
            weights_dtype=weights_dtype,
            from_pt=True,
            use_safetensors=True,
        )

    action_encoder = FlaxActionEncoder(
        action_dim=args.action_dim,
        hidden_size=1024,
        text_embed_dim=512 if args.text_cond else None,
        dtype=dtype,
        weights_dtype=weights_dtype,
    )
    action_encoder_params = _load_action_encoder_params(
        os.path.join(args.ctrl_world_dir, "action_encoder.safetensors"),
        weights_dtype,
    )

    scheduler = FlaxEDMEulerScheduler(
        sigma_min=0.002,
        sigma_max=700.0,
        rho=7.0,
        prediction_type="v_prediction",
        dtype=weights_dtype,
    )

    pipeline = FlaxCtrlWorldPipeline(
        vae=vae,
        unet=unet,
        action_encoder=action_encoder,
        scheduler=scheduler,
        image_encoder=None,
        feature_extractor=None,
        dtype=dtype,
    )

    accel = jax.devices()[0]

    def _to_accel(x):
        if isinstance(x, jax.Array):
            return jax.device_put(x.astype(weights_dtype), accel)
        return x

    params = jax.tree_util.tree_map(
        _to_accel,
        {"unet": unet_params, "vae": vae_params, "action_encoder": action_encoder_params},
    )
    max_logging.log(f"[mcw] all weights on device in {time.perf_counter() - t_load:.1f}s")

    # Jitted VAE wrappers: compile the encoder/decoder graphs once per shape.
    # The ctrl-world pipeline's denoising loop is already a ``lax.fori_loop``
    # (see pipeline_flax_ctrl_world.py), so no driver-side jit is needed around
    # ``pipeline(...)`` itself — the base SVD generate scripts call the pipeline
    # the same way.
    jit_encode = _build_jit_vae_encode(vae)
    jit_decode = _build_jit_vae_decode(vae)

    text_shim = _TorchCLIPTextShim(args.clip_path) if args.text_cond else None

    with open(args.data_stat_path) as f:
        stat = json.load(f)
    state_p01 = np.array(stat["state_01"])[None, :]
    state_p99 = np.array(stat["state_99"])[None, :]

    if args.val_ids:
        val_ids = list(args.val_ids)
        start_idxs = list(args.start_idx) if args.start_idx else [0] * len(val_ids)
        if len(start_idxs) == 1 and len(val_ids) > 1:
            start_idxs = start_idxs * len(val_ids)
    else:
        val_ids = list(_DEFAULT_VAL_IDS)
        start_idxs = list(_DEFAULT_START_IDX)

    save_root = Path(args.save_dir) / args.task_name / "video"
    save_root.mkdir(parents=True, exist_ok=True)

    for t_idx, (val_id, start_idx) in enumerate(zip(val_ids, start_idxs)):
        max_logging.log(
            f"[mcw] ({t_idx + 1}/{len(val_ids)}) replay trajectory id={val_id} start_idx={start_idx}"
        )
        ann_path = Path(args.val_dataset_dir) / "annotation" / "val" / f"{val_id}.json"
        ann = json.loads(ann_path.read_text())
        instruction = ann["texts"][0]
        try:
            length = len(ann["action"])
        except KeyError:
            length = ann["video_length"]

        num_total = int(args.pred_step * args.interact_num + 8)
        skip = args.skip_step
        frame_ids = np.arange(start_idx, start_idx + num_total * skip, skip)
        frame_ids = np.minimum(frame_ids, length - 1).astype(int)

        eef_gt = np.array(ann["states"])[frame_ids]  # (T, 7)
        joint_gt = np.array(ann["joints"])[frame_ids]  # (T, 8) — unused here but kept for parity

        per_view_gt: list[np.ndarray] = []
        per_view_lat: list[jnp.ndarray] = []
        for view_i, vid in enumerate(ann["videos"]):
            vpath = str(Path(args.val_dataset_dir) / vid["video_path"])
            t_io = time.perf_counter()
            true_vid = _read_video_frames(vpath, frame_ids)  # (T, Hv, Wv, 3) uint8
            per_view_gt.append(true_vid)
            t_enc = time.perf_counter()
            lat = _vae_encode_frames(jit_encode, params["vae"], true_vid, mesh, dtype)
            lat.block_until_ready()
            per_view_lat.append(lat)
            max_logging.log(
                f"[mcw]   view {view_i + 1}/{_NUM_VIEWS}: "
                f"read {t_enc - t_io:.1f}s, vae_encode {time.perf_counter() - t_enc:.1f}s"
                + (" (first call: compile included)" if t_idx == 0 and view_i == 0 else "")
            )

        if len(per_view_lat) != _NUM_VIEWS:
            raise RuntimeError(
                f"traj {val_id}: expected {_NUM_VIEWS} exterior videos, got {len(per_view_lat)}"
            )

        # stacked = (T, 4, 72, 40)
        stacked = jnp.concatenate(per_view_lat, axis=-2)
        his_cond = [stacked[0:1]] * (args.num_history * 4)  # each (1, 4, 72, 40)
        his_eef = [eef_gt[0:1]] * (args.num_history * 4)

        text_embeds = text_shim([instruction]) if text_shim is not None else None

        video_to_save: list[np.ndarray] = []

        for step in range(args.interact_num):
            start_id = step * (args.pred_step - 1)
            end_id = start_id + args.pred_step
            cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)

            his_pose = np.concatenate(
                [his_eef[idx] for idx in _HISTORY_IDX], axis=0
            )  # (6, 7)
            action_np = np.concatenate([his_pose, cartesian_pose], axis=0)  # (11, 7)
            action_norm = _normalize_bound(action_np, state_p01, state_p99)
            action = jnp.asarray(action_norm, dtype=dtype)[None, ...]  # (1, 11, 7)

            his_cond_input = jnp.concatenate(
                [his_cond[idx] for idx in _HISTORY_IDX], axis=0
            )[None, ...]  # (1, 6, 4, 72, 40)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)

            prng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1000 * t_idx + step)

            if t_idx == 0 and step == 0:
                max_logging.log(
                    "[mcw]   ── first pipeline call: tracing + compiling the denoising "
                    "loop body (fori_loop compiles the UNet body once, not 50x). "
                    "Expect ~1-3 min; every subsequent step runs in ~2-5 s."
                )
            t_step = time.perf_counter()
            with mesh:
                pred_latents = pipeline(
                    params=params,
                    prng_seed=prng,
                    action=action,
                    image_latent=current_latent,
                    history=his_cond_input,
                    text_embeds=text_embeds,
                    num_frames=args.pred_step,
                    num_history=args.num_history,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    min_guidance_scale=args.guidance_scale,
                    max_guidance_scale=args.guidance_scale,
                    fps_id=args.fps,
                    motion_bucket_id=args.motion_bucket_id,
                    cond_aug=args.cond_aug,
                    frame_level_cond=True,
                    his_cond_zero=False,
                    output_type="latent",
                )
            pred_latents.block_until_ready()  # force execution so the timing below is real
            elapsed = time.perf_counter() - t_step
            tag = "pipeline (compile+run)" if (t_idx == 0 and step == 0) else "pipeline"
            max_logging.log(
                f"[mcw]   traj {val_id} step {step + 1}/{args.interact_num} {tag}={elapsed:.1f}s"
            )
            # pred_latents: (1, pred_step, 4, 72, 40)
            pl = pred_latents[0]  # drop B
            # split 72-height latent back into 3 × (pred_step, 4, 24, 40) views
            pred_per_view = jnp.split(pl, _NUM_VIEWS, axis=-2)

            # decode predicted + gt per view
            t_dec = time.perf_counter()
            pred_videos = [
                _vae_decode_chunks(
                    jit_decode, params["vae"], v, mesh, args.decode_chunk_size
                )
                for v in pred_per_view
            ]
            gt_slice_per_view = [v[start_id:end_id] for v in per_view_lat]
            gt_videos = [
                _vae_decode_chunks(
                    jit_decode, params["vae"], v, mesh, args.decode_chunk_size
                )
                for v in gt_slice_per_view
            ]
            if t_idx == 0 and step == 0:
                max_logging.log(
                    f"[mcw]   vae_decode (compile+run) {time.perf_counter() - t_dec:.1f}s"
                )

            # Stack each per-view video on the H axis (match torch layout) then
            # place gt on top of pred: final frame shape (3*Hv, Wv, 3) → (6*Hv, Wv, 3).
            gt_stack = np.concatenate(gt_videos, axis=1)
            pred_stack = np.concatenate(pred_videos, axis=1)
            both = np.concatenate([gt_stack, pred_stack], axis=1)  # H: top=gt, bottom=pred
            # Match the torch trim: first pred_step-1 frames for all but the last step
            # (last frame of step i == first frame of step i+1 in replay).
            video_to_save.append(both if step == args.interact_num - 1 else both[: args.pred_step - 1])

            # Advance the rolling buffer.
            his_eef.append(cartesian_pose[args.pred_step - 1 : args.pred_step])
            last_pred = jnp.concatenate(
                [v[args.pred_step - 1] for v in pred_per_view], axis=-2
            )[None, ...]
            his_cond.append(last_pred)

        # Write MP4 matching torch naming.
        video = np.concatenate(video_to_save, axis=0)
        safe_text = (
            instruction.replace(" ", "_")
            .replace(",", "")
            .replace(".", "")
            .replace("'", "")
            .replace('"', "")
        )[:30]
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = (
            save_root
            / f"time_{uuid}_traj_{val_id}_{start_idx}_{args.pred_step}_{safe_text}.mp4"
        )
        iio.imwrite(str(fname), video, fps=4, codec="libx264")
        max_logging.log(f"[mcw] wrote {fname}")


def _parse_list(kind):
    def _parse(value: str):
        if not value:
            return []
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [kind(p) for p in parts]

    return _parse


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctrl_world_dir", required=True, help="Converted JAX-loadable dir (unet/, vae/, action_encoder.safetensors)")
    ap.add_argument("--clip_path", required=True, help="Local openai/clip-vit-base-patch32 dir")
    ap.add_argument("--val_dataset_dir", required=True, help="e.g. dataset_example/droid_subset")
    ap.add_argument("--data_stat_path", required=True, help="e.g. dataset_meta_info/droid/stat.json")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--task_name", default="Rollouts_replay")
    ap.add_argument("--val_ids", type=_parse_list(str), default=None, help="Comma-separated trajectory IDs (default: replay preset)")
    ap.add_argument("--start_idx", type=_parse_list(int), default=None, help="Comma-separated start indices (one per val_id, or single value broadcast)")
    ap.add_argument("--pred_step", type=int, default=5)
    ap.add_argument("--num_history", type=int, default=6)
    ap.add_argument("--interact_num", type=int, default=12)
    ap.add_argument("--action_dim", type=int, default=7)
    ap.add_argument("--skip_step", type=int, default=1)
    ap.add_argument("--height", type=int, default=576, help="Stacked-view latent height in pixels (192 per view × 3)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--motion_bucket_id", type=int, default=127)
    ap.add_argument("--cond_aug", type=float, default=0.02)
    ap.add_argument("--fps", type=int, default=7)
    ap.add_argument("--decode_chunk_size", type=int, default=7)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--text_cond", dest="text_cond", action="store_true", default=True)
    ap.add_argument("--no_text_cond", dest="text_cond", action="store_false")
    args = ap.parse_args(argv)
    run_replay(args)


if __name__ == "__main__":
    main()
