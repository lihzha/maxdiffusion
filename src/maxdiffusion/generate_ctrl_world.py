"""Val-split rollout inference for the action-conditioned SVD (Ctrl-World) model.

Counterpart to ``generate_wan_ctrl_world.py``, for the SVD-based world model
trained by ``trainers/ctrl_world_trainer.py``. Differences from the existing
``generate_ctrl_world_replay.py``, which this does not replace:

  * driven by ``pyconfig`` + ``configs/base_ctrl_world.yml`` (same config the
    run trained with) instead of argparse;
  * reads the **pre-encoded TFRecord val split** (``eval_data_dir``) rather than
    raw MP4s + upstream annotation JSON, so no VAE encode and no CLIP text tower
    are needed — text embeddings ship in the records;
  * loops the **entire** val split, one rollout per episode, instead of a
    hand-listed set of trajectory ids.

Every rollout starts from a single real observation (episode latent frame 0).
All six history slots collapse onto that frame for the first chunk, which is the
same condition training hits whenever ``frame_now`` is near an episode start (the
history gather clips at 0) and whenever the ``skip_his=0`` branch fires.

Rollout geometry — the first *future* slot is the current observation
(``fut_offsets`` starts at 0 and training's concat stream is
``latents[:, num_history]``), so a chunk of ``num_frames`` latents contributes
``num_frames - 1`` genuinely new frames and consecutive chunks overlap by one::

    horizon = 1 + ar_num_chunks * (num_frames - 1)

History for chunk ``k`` (anchor ``frame_now = k * (num_frames - 1)``) is taken
from the frames generated so far, at training's stride::

    stride   = ctrl_world_max_skip_his // ctrl_world_max_skip     # 4
    hist_ids = clip(frame_now - stride * [6,5,4,3,2,1], 0, frame_now)

Episodes shorter than the horizon are kept: the dataset repeats their last
latent/action out to the full window, and the video is trimmed back to
``n_real_frames`` so no padded frame is ever written.

Usage::

    python src/maxdiffusion/generate_ctrl_world.py \
        src/maxdiffusion/configs/base_ctrl_world.yml \
        pretrained_model_name_or_path=<svd dir with unet/ + vae/> \
        checkpoint_dir=<orbax dir written by CtrlWorldTrainer> \
        checkpoint_step=100000 \
        eval_data_dir=<.../val> \
        stats_path=<.../stats.json> \
        output_dir=./outputs_ctrl_world \
        max_guidance_scale=1.0 \
        hardware=gpu
"""

from __future__ import annotations

import functools
import os
import time
from typing import Sequence

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl import app
from flax.traverse_util import flatten_dict, unflatten_dict

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.input_pipeline.robot.ctrl_world_droid_dataset import (
    CtrlWorldDroidRolloutDataset,
)
from maxdiffusion.models.svd.action_encoder_flax import (
    FlaxActionAdaLNProjector,
    FlaxActionEncoder,
)
from maxdiffusion.models.svd.ctrl_world_flax import (
    _is_skeleton_mode,
    _skeleton_apply_key,
)
from maxdiffusion.models.svd.skeleton_encoder_flax import (
    FlaxSkeletonAdaLNProjector,
    FlaxSkeletonCrossAttnEmbed,
    FlaxSkeletonPatchEmbed,
)
from maxdiffusion.models.svd.video_autoencoder_flax import FlaxSVDAutoencoderKL
from maxdiffusion.models.svd.video_unet_flax import FlaxVideoUNet
from maxdiffusion.pipelines.svd.pipeline_flax_ctrl_world import FlaxCtrlWorldPipeline
from maxdiffusion.schedulers.scheduling_edm_euler_flax import FlaxEDMEulerScheduler


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dtype_from_str(name):
    return {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[
        jnp.dtype(name).name
    ]


def _build_mesh(config):
    devices = max_utils.create_device_mesh(config)
    return jax.sharding.Mesh(devices, config.mesh_axes)


# Must stay identical to ``ctrl_world_trainer._F32_PARAM_KEYWORDS``: the trainer
# keeps these leaves in float32 regardless of weights_dtype, so the checkpoint
# holds them in f32 and casting them to bf16 here would run the model at a lower
# precision than it was trained at, on exactly the layers the trainer singles out
# as precision-sensitive.
_F32_PARAM_KEYWORDS = ("norm", "time_embedding", "add_embedding")


def _cast_param(path, x, dtype):
    """Cast one param leaf to ``dtype``, excluding the f32-pinned ones."""
    if not (hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)):
        return x
    key = jax.tree_util.keystr(path).lower()
    if any(k in key for k in _F32_PARAM_KEYWORDS):
        return x.astype(jnp.float32)
    return x.astype(dtype)


def _build_jit_vae_decode(vae):
    """Jitted VAE decode; ``num_frames`` is static (one compile per chunk length).

    Takes latents in *scaled* space (as stored in the TFRecords and as produced
    by the pipeline) and returns frames in [0, 1], NCHW.
    """
    scale = vae.config.scaling_factor

    @functools.partial(jax.jit, static_argnames=("num_frames",))
    def _dec(params, latents_scaled, num_frames):
        dec = vae.apply(
            {"params": params},
            latents_scaled / scale,
            num_frames=num_frames,
            deterministic=True,
            method=vae.decode,
        )
        return (dec.sample / 2.0 + 0.5).clip(0.0, 1.0)

    return _dec


def _decode_views(jit_decode, vae_params, latents, mesh, num_views, chunk_size):
    """``(T, 4, num_views*h, w)`` scaled latents → list of ``(T, H, W, 3)`` uint8.

    The 3 cameras were encoded separately and stacked on the latent height axis,
    so they are split apart and decoded independently. Returned per view rather
    than pre-tiled so the caller can both tile them and write each one out on its
    own; use ``_tile_views`` for the side-by-side layout.
    """
    per_view = jnp.split(latents, num_views, axis=-2)
    decoded = []
    for view in per_view:
        chunks = []
        for i in range(0, view.shape[0], chunk_size):
            chunk = view[i : i + chunk_size]
            with mesh:
                chunks.append(jit_decode(vae_params, chunk, chunk.shape[0]))
        frames = jnp.concatenate(chunks, axis=0)             # (T, 3, h, w)
        frames = jnp.transpose(frames, (0, 2, 3, 1))         # (T, h, w, 3)
        decoded.append((np.asarray(frames) * 255.0).astype(np.uint8))
    return decoded


def _tile_views(views: list) -> np.ndarray:
    """Lay per-view frames side by side on the width axis (wan-script layout)."""
    return np.concatenate(views, axis=2)                     # (T, h, num_views*w, 3)


def _restore_params(ckpt_dir: str, step, template: dict):
    """Restore the ``params`` tree written by CtrlWorldTrainer.

    ``opt_state`` is declared so checkpoints that carry it still open, but it is
    never requested — inference needs params only (see the wan-ac equivalent:
    only ``params/`` and ``step/`` have to be downloaded).
    """
    mgr = ocp.CheckpointManager(
        ckpt_dir,
        item_names=("params", "step", "opt_state"),
        item_handlers={
            "params": ocp.StandardCheckpointHandler(),
            "step": ocp.JsonCheckpointHandler(),
            "opt_state": ocp.StandardCheckpointHandler(),
        },
    )
    target = int(step) if step and int(step) > 0 else mgr.latest_step()
    if target is None:
        raise FileNotFoundError(f"no checkpoint found under {ckpt_dir}")
    # Compare the action route against the checkpoint before handing the template
    # to orbax: a StandardRestore demands an exact structure match and reports a
    # mismatch as a wall of leaf values, which buries the one thing that matters
    # (this checkpoint was trained in the other action_cond_mode).
    on_disk = set(mgr.item_metadata(target).params.keys())
    wanted = set(template)
    if wanted != on_disk:
        raise KeyError(
            f"checkpoint {ckpt_dir} step {target} holds params {sorted(on_disk)} but "
            f"this config asks for {sorted(wanted)}. Every action_cond_mode writes a "
            "different subtree — cross_attn: action_encoder; adaln: + "
            "action_adaln_proj; skeleton / skeleton_adaln / skeleton_cross_attn: "
            "skeleton_embed / skeleton_adaln_proj / skeleton_cross_attn_embed and NO "
            "action_encoder at all — so pass the mode the checkpoint was trained with "
            "(or point checkpoint_dir at the matching run). Running one route's "
            "weights through another would silently drop the conditioning signal."
        )
    restored = mgr.restore(
        target, args=ocp.args.Composite(params=ocp.args.StandardRestore(template))
    )
    return restored["params"], target


def _load_modules(config, mesh, dtype, weights_dtype):
    """UNet + VAE + the conditioning modules this ``action_cond_mode`` trained.

    The UNet is built exactly as ``CtrlWorldTrainer._load_modules`` builds it, so
    the restored params tree lines up leaf for leaf. The VAE is never trained, so
    it always comes from ``pretrained_model_name_or_path``.

    Returns ``(unet, unet_params, vae, vae_params, action_encoder, ae_params,
    cond_extras)``, where ``cond_extras`` maps each extra params key to its
    ``(module, params)`` pair — the same shape the trainer uses, so the restore
    template is just ``{"unet": ..., **cond_extras}`` (plus ``action_encoder``
    outside the skeleton modes, which have no vector-action encoder at all).
    """
    with mesh:
        max_logging.log(
            f"[ctrl_world_infer] loading UNet from "
            f"{config.pretrained_model_name_or_path}/unet"
        )
        unet, unet_params = FlaxVideoUNet.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="unet",
            dtype=dtype,
            weights_dtype=weights_dtype,
            # Must mirror CtrlWorldTrainer._load_modules: adaln checkpoints were
            # trained with AdaGN resnets, so without this the restore template is
            # missing every adagn_scale_proj leaf and orbax fails on structure.
            adagn=config.action_cond_mode == "adaln",
            from_pt=config.from_pt,
            use_safetensors=True,
            attention_kernel=config.attention,
            temporal_attention_kernel=config.temporal_attention,
            use_memory_efficient_attention=config.use_memory_efficient_attention,
            flash_block_sizes=max_utils.get_flash_block_sizes(config) or {},
            flash_min_seq_length=config.flash_min_seq_length,
            mesh=mesh,
            precision=max_utils.get_precision(config),
            norm_num_groups=config.norm_num_groups,
        )
        max_logging.log(
            f"[ctrl_world_infer] loading VAE from "
            f"{config.pretrained_model_name_or_path}/vae"
        )
        vae, vae_params = FlaxSVDAutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="vae",
            dtype=dtype,
            weights_dtype=weights_dtype,
            from_pt=config.from_pt,
            use_safetensors=True,
        )

    skeleton_mode = _is_skeleton_mode(config.action_cond_mode)

    # No action encoder in the skeleton modes — the conditioning is the rendered
    # skeleton video and the vector actions are unused, so the trained checkpoint
    # carries no action_encoder subtree to restore into one. Must mirror
    # CtrlWorldTrainer._load_modules, which gates it off the same way.
    action_encoder, ae_params = None, None
    if not skeleton_mode:
        action_encoder = FlaxActionEncoder(
            action_dim=config.action_dim,
            hidden_size=config.hidden_size,
            text_embed_dim=config.text_embed_dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        # Structure only — every value is overwritten by the checkpoint below.
        ae_params = action_encoder.init_weights(
            jax.random.PRNGKey(config.seed), batch=1, num_frames=1
        )

    cond_extras = {}
    if skeleton_mode:
        mode = config.action_cond_mode
        if mode == "skeleton":
            skel_mod = FlaxSkeletonPatchEmbed(
                model_channels=unet.block_out_channels[0],
                alpha=float(max_utils.config_get(config, "skeleton_embed_alpha", 0.1)),
                dtype=dtype,
                weights_dtype=weights_dtype,
            )
        elif mode == "skeleton_adaln":
            skel_mod = FlaxSkeletonAdaLNProjector(
                time_embed_dim=unet.block_out_channels[0] * 4,
                dtype=dtype,
                weights_dtype=weights_dtype,
            )
        else:
            skel_mod = FlaxSkeletonCrossAttnEmbed(
                hidden_size=config.hidden_size,
                stride=int(
                    max_utils.config_get(config, "skeleton_cross_attn_stride", 4)
                ),
                latent_height=config.latent_height_per_cam * config.num_views,
                latent_width=config.width // 8,
                dtype=dtype,
                weights_dtype=weights_dtype,
            )
        # seed+2 matches the trainer, though every value is overwritten below —
        # only the tree structure has to line up.
        cond_extras[_skeleton_apply_key(mode)] = (
            skel_mod,
            skel_mod.init_weights(jax.random.PRNGKey(config.seed + 2)),
        )

    adaln = config.action_cond_mode == "adaln"
    if adaln:
        # from_pretrained returns only what the SVD checkpoint contains, so the
        # AdaGN scale projections are absent. Add them as zeros purely so the
        # template has the right *structure* — every value here is overwritten by
        # the trained checkpoint in _restore_params.
        with mesh:
            abstract = unet.init_weights(jax.random.PRNGKey(config.seed), eval_only=True)
        flat_a, flat_l = flatten_dict(abstract), flatten_dict(unet_params)
        for k in flat_a:
            if k not in flat_l and "adagn_scale_proj" in k:
                spec = flat_a[k]
                spec = spec.value if hasattr(spec, "value") else spec
                flat_l[k] = jnp.zeros(spec.shape, weights_dtype)
        unet_params = unflatten_dict(flat_l)

    if adaln:
        adaln_proj = FlaxActionAdaLNProjector(
            time_embed_dim=unet.block_out_channels[0] * 4,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        cond_extras["action_adaln_proj"] = (
            adaln_proj,
            adaln_proj.init_weights(
                jax.random.PRNGKey(config.seed + 1),
                batch=1,
                num_frames=1,
                hidden_size=action_encoder.hidden_size,
            ),
        )

    return (unet, unet_params, vae, vae_params, action_encoder, ae_params,
            cond_extras)


def _save_comparison_video(gt_frames, pred_frames, path: str, fps: int) -> None:
    """Write GT (top) / predicted (bottom) stacked MP4."""
    combined = np.concatenate([gt_frames, pred_frames], axis=1)  # stack on H
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    iio.imwrite(path, combined, fps=fps, codec="libx264")


def _save_per_view_videos(pred_views: list, output_dir: str, name: str, fps: int) -> list:
    """Write the *predicted* rollout once per camera, into ``output_dir/cam{i}/``.

    GT is deliberately not split out: it is already in the tiled comparison
    video, and per-view copies would triple the file count for no extra
    information. The frames are the same trimmed arrays that went into the tile,
    so these line up with it frame for frame.

    One directory per camera (rather than a ``_cam{i}`` filename suffix) so a
    single camera's whole sweep is one glob, and ``name`` stays identical across
    directories for easy pairing.
    """
    paths = []
    for i, view in enumerate(pred_views):
        cam_dir = os.path.join(output_dir, f"cam{i}")
        os.makedirs(cam_dir, exist_ok=True)
        p = os.path.join(cam_dir, f"{name}.mp4")
        iio.imwrite(p, view, fps=fps, codec="libx264")
        paths.append(p)
    return paths


def _fmt_dur(seconds: float) -> str:
    """``1234.5`` → ``"20m34s"``; keeps long sweeps readable in a slurm log."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ── Main ──────────────────────────────────────────────────────────────────────


def run(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config

    dtype = _dtype_from_str(config.activations_dtype)
    weights_dtype = _dtype_from_str(config.weights_dtype)

    num_history = config.num_history
    num_frames = config.num_frames
    # Frames added per chunk: slot 0 of the future is the current observation.
    per_chunk = num_frames - 1
    if per_chunk < 1:
        raise ValueError("num_frames must be >= 2 for an auto-regressive rollout")
    ar_num_chunks = int(max_utils.config_get(config, "ar_num_chunks", 12))
    horizon = 1 + ar_num_chunks * per_chunk

    # Training's history stride: (max_skip_his // max_skip) * skip, at skip=1 —
    # the same value the val branch of _build_window pins.
    hist_stride = config.ctrl_world_max_skip_his // config.ctrl_world_max_skip
    hist_offsets = hist_stride * np.arange(num_history, 0, -1)

    num_views = config.num_views
    fps = int(max_utils.config_get(config, "output_video_fps", 5))
    num_inference_steps = int(config.num_inference_steps)
    decode_chunk_size = max(1, int(config.decode_chunk_size))
    max_episodes = int(max_utils.config_get(config, "eval_max_episodes", -1))

    do_cfg = float(config.max_guidance_scale) > 1.0
    max_logging.log("***** Running Ctrl-World rollout inference *****")
    max_logging.log(f"  Devices:          {jax.device_count()} x {jax.devices()[0].device_kind}")
    max_logging.log(f"  Dtypes:           weights={config.weights_dtype} activations={config.activations_dtype}")
    max_logging.log(f"  Horizon:          {horizon} latent frames "
                    f"= 1 obs + {ar_num_chunks} chunks x {per_chunk} new frames")
    max_logging.log(f"  History:          {num_history} slots at stride {hist_stride} "
                    f"(offsets {hist_offsets.tolist()})")
    max_logging.log(f"  Denoise steps:    {num_inference_steps} per chunk "
                    f"({ar_num_chunks * num_inference_steps} UNet passes per episode"
                    f"{', double-batched for CFG' if do_cfg else ''})")
    max_logging.log(f"  Guidance:         [{config.min_guidance_scale}, {config.max_guidance_scale}]"
                    f"{'' if do_cfg else '  (CFG off)'}")
    max_logging.log(f"  Action cond mode: {config.action_cond_mode}")
    max_logging.log(f"  Attention:        {config.attention} / {config.temporal_attention} (temporal)")
    max_logging.log(f"  Val data:         {config.eval_data_dir}")
    max_logging.log(f"  Stats:            {config.stats_path}")
    max_logging.log(f"  Output dir:       {config.output_dir}")
    max_logging.log(f"  Max episodes:     {'all' if max_episodes < 0 else max_episodes}")

    # Optional op-level profile of ONE steady-state chunk (episode 0, chunk 2 —
    # chunk 1 is polluted by the one-time XLA compile). Cheap way to find where
    # a 0.7s/forward UNet is actually spending time; open the trace in
    # TensorBoard (`tensorboard --logdir <output_dir>/profile`) and read the
    # XLA op breakdown rather than guessing at kernels.
    enable_profiler = bool(max_utils.config_get(config, "enable_profiler", False))
    profile_dir = os.path.join(config.output_dir, "profile")
    if enable_profiler:
        max_logging.log(f"  Profiler:         ON -> {profile_dir} (episode 0, chunk 2 only)")

    mesh = _build_mesh(config)
    t_load = time.perf_counter()
    (unet, unet_params, vae, vae_params, action_encoder, ae_params,
     cond_extras) = _load_modules(config, mesh, dtype, weights_dtype)

    # ── Restore trained weights ───────────────────────────────────────────────
    adaln = config.action_cond_mode == "adaln"
    skeleton_mode = _is_skeleton_mode(config.action_cond_mode)
    skel_key = _skeleton_apply_key(config.action_cond_mode)
    ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
    template = {"unet": unet_params}
    if not skeleton_mode:
        template["action_encoder"] = ae_params
    for name, (_mod, mod_params) in cond_extras.items():
        template[name] = mod_params
    restored, restored_step = _restore_params(
        ckpt_dir, max_utils.config_get(config, "checkpoint_step", -1), template
    )
    # _restore_params rejects an action-route mismatch (adaln vs cross_attn) before
    # it ever reaches orbax, so restored is known to line up with the config here.
    max_logging.log(
        f"[ctrl_world_infer] restored params from {ckpt_dir} (step {restored_step})"
    )

    scheduler = FlaxEDMEulerScheduler(
        sigma_min=config.diffusion_scheduler_config["sigma_min"],
        sigma_max=config.diffusion_scheduler_config["sigma_max"],
        rho=config.diffusion_scheduler_config["rho"],
        prediction_type=config.diffusion_scheduler_config["prediction_type"],
        dtype=weights_dtype,
    )
    pipeline = FlaxCtrlWorldPipeline(
        vae=vae,
        unet=unet,
        action_encoder=action_encoder,
        action_adaln_proj=cond_extras.get("action_adaln_proj", (None, None))[0],
        skeleton_module=cond_extras.get(skel_key, (None, None))[0] if skel_key else None,
        skeleton_params_key=skel_key,
        # The skeleton modes have no action encoder, so the pipeline cannot read
        # the cross-attention / text widths off it — hand them over explicitly.
        cross_attn_dim=config.hidden_size,
        text_embed_dim=config.text_embed_dim,
        scheduler=scheduler,
        image_encoder=None,
        feature_extractor=None,
        dtype=dtype,
    )

    accel = jax.devices()[0]

    def _to_accel(path, x):
        if not isinstance(x, jax.Array):
            return x
        return jax.device_put(_cast_param(path, x, weights_dtype), accel)

    params = jax.tree_util.tree_map_with_path(
        _to_accel,
        {
            "unet": restored["unet"],
            "vae": vae_params,
            # Every key the template asked for came back; forwarding by key keeps
            # this in step with _load_modules instead of restating the routes.
            **{k: restored[k] for k in restored if k != "unet"},
        },
    )
    max_logging.log(
        f"[ctrl_world_infer] weights on device in {time.perf_counter() - t_load:.1f}s"
    )
    jit_decode = _build_jit_vae_decode(vae)

    # ── Val dataset: one rollout window per episode, anchored at frame 0 ──────
    if not config.eval_data_dir:
        raise ValueError("eval_data_dir must point at the val TFRecord shards")
    dataset = CtrlWorldDroidRolloutDataset(
        data_dir=config.eval_data_dir,
        stats_path=config.stats_path,
        window_frames=horizon,
        action_dim=config.action_dim,
        text_embed_dim=config.text_embed_dim,
        down_sample=config.ctrl_world_down_sample,
        batch_size=1,
        min_traj_len_5hz=int(max_utils.config_get(config, "eval_min_latent_frames", 2)),
        # The skeleton modes condition on the rendered-skeleton video, so the
        # records must carry skeleton_cam0/1/2 and the loader must read them.
        load_skeleton=skeleton_mode,
    )

    max_logging.log(
        f"[ctrl_world_infer] val split: {len(dataset.files)} shards under "
        f"{config.eval_data_dir} (episodes shorter than {horizon} frames are rolled "
        f"out and trimmed back to their real length)"
    )

    output_dir = os.path.join(config.output_dir, "inference_videos")
    text_embed_dim = config.text_embed_dim

    # Cross-episode tallies for the closing summary.
    t_sweep = time.perf_counter()
    n_written = 0
    n_trimmed = 0
    total_rollout_s = 0.0
    total_decode_s = 0.0

    # ── Rollout loop ──────────────────────────────────────────────────────────
    for ep_idx, batch in enumerate(dataset):
        if 0 <= max_episodes <= ep_idx:
            max_logging.log(
                f"[ctrl_world_infer] stopping at eval_max_episodes={max_episodes}"
            )
            break

        gt_latents = jnp.asarray(batch["latent"][0], dtype=weights_dtype)   # (horizon, 4, H, W)
        actions = jnp.asarray(batch["action"][0], dtype=weights_dtype)      # (horizon, 7)
        n_real = int(np.asarray(batch["n_real_frames"]).reshape(-1)[0])
        # (horizon, 4, H, W) — the rendered-skeleton latents for this episode,
        # indexed by absolute latent frame exactly like `actions`. Conditioning
        # only: it is never denoised and never written out.
        skeleton_ep = (
            jnp.asarray(batch["skeleton"][0], dtype=weights_dtype)
            if skeleton_mode
            else None
        )
        # use_task_instructions=False must reproduce the action-only training
        # setup, so drop the instruction here rather than inside the pipeline.
        text_embeds = (
            jnp.asarray(batch["text_embeds"], dtype=weights_dtype)
            if text_embed_dim and max_utils.config_get(config, "use_task_instructions", True)
            else None
        )                                                                  # (1, 512)

        max_logging.log(
            f"[ctrl_world_infer] ── episode {ep_idx}: rolling out {horizon} frames "
            f"from 1 observation ({n_real}/{horizon} frames have real GT)"
        )
        if ep_idx == 0:
            max_logging.log(
                "[ctrl_world_infer]    first chunk pays a one-time XLA compile of the "
                "denoising loop; the executable is cached for every later chunk, so "
                "chunk 1 runs long and the rest settle at a steady cost"
            )
        t_ep = time.perf_counter()

        # The rollout buffer starts with exactly one real frame; every later
        # entry is generated. Indexed by absolute latent frame number.
        generated = [gt_latents[0]]

        for chunk_i in range(ar_num_chunks):
            frame_now = chunk_i * per_chunk
            hist_ids = np.clip(frame_now - hist_offsets, 0, frame_now)
            history = jnp.stack([generated[i] for i in hist_ids], axis=0)[None]
            # (1, num_history, 4, H, W)

            fut_ids = np.arange(frame_now, frame_now + num_frames)
            action_window = jnp.concatenate(
                [actions[jnp.asarray(hist_ids)], actions[jnp.asarray(fut_ids)]], axis=0
            )[None]                                            # (1, num_history+num_frames, 7)
            # Same history/future gather as the actions, so the skeleton frames
            # line up slot for slot with the latents the UNet sees.
            skeleton_window = (
                jnp.concatenate(
                    [
                        skeleton_ep[jnp.asarray(hist_ids)],
                        skeleton_ep[jnp.asarray(fut_ids)],
                    ],
                    axis=0,
                )[None]                            # (1, num_history+num_frames, 4, H, W)
                if skeleton_mode
                else None
            )

            # Conditioning image = the current observation, i.e. the newest frame
            # in the buffer. Matches training, where the concat stream is built
            # from the first future slot (= frame_now).
            image_latent = generated[frame_now][None]           # (1, 4, H, W)

            prng = jax.random.fold_in(
                jax.random.PRNGKey(config.seed), 100_000 * ep_idx + chunk_i
            )
            profiling = enable_profiler and ep_idx == 0 and chunk_i == 1
            if profiling:
                os.makedirs(profile_dir, exist_ok=True)
                jax.profiler.start_trace(profile_dir)
            t_chunk = time.perf_counter()
            with mesh:
                pred = pipeline(
                    params=params,
                    prng_seed=prng,
                    action=action_window,
                    image_latent=image_latent,
                    history=history,
                    text_embeds=text_embeds,
                    skeleton=skeleton_window,
                    num_frames=num_frames,
                    num_history=num_history,
                    num_inference_steps=num_inference_steps,
                    min_guidance_scale=config.min_guidance_scale,
                    max_guidance_scale=config.max_guidance_scale,
                    # ADM micro-conditioning must match what training baked in
                    # (CtrlWorldTrainConfig takes these from the ctrl_* keys).
                    fps_id=config.ctrl_fps_id,
                    motion_bucket_id=config.ctrl_motion_bucket_id,
                    cond_aug=config.ctrl_noise_aug_strength,
                    frame_level_cond=True,
                    his_cond_zero=config.ctrl_his_cond_zero,
                    action_cond_mode=config.action_cond_mode,
                    output_type="latent",
                )
            pred.block_until_ready()
            if profiling:
                jax.profiler.stop_trace()
                max_logging.log(
                    f"[ctrl_world_infer]    profile of chunk 2 written to {profile_dir}"
                )
            # Slot 0 regenerates frame_now, which we already have — drop it so
            # chunks stitch without duplicating a frame.
            generated.extend(pred[0, 1:].astype(weights_dtype))
            # Per-chunk line is the liveness signal: the pipeline's denoising loop
            # is a fori_loop and cannot log from inside, so without this an episode
            # goes silent for ar_num_chunks * num_inference_steps UNet passes.
            max_logging.log(
                f"[ctrl_world_infer]    chunk {chunk_i + 1}/{ar_num_chunks}: "
                f"frames {frame_now + 1}-{frame_now + per_chunk} "
                f"(cond on frame {frame_now}, hist {hist_ids.tolist()}) "
                f"in {time.perf_counter() - t_chunk:.1f}s"
                + ("  [includes compile]" if ep_idx == 0 and chunk_i == 0 else "")
            )

        rollout_s = time.perf_counter() - t_ep
        total_rollout_s += rollout_s

        # Trim the padded tail: only frames backed by real GT get written.
        keep = min(n_real, len(generated))
        pred_seq = jnp.stack(generated[:keep], axis=0)
        gt_seq = gt_latents[:keep]
        if keep < horizon:
            n_trimmed += 1
            max_logging.log(
                f"[ctrl_world_infer]    episode ends early — writing {keep}/{horizon} "
                f"frames, trimming {horizon - keep} padded"
            )

        t_dec = time.perf_counter()
        gt_views = _decode_views(
            jit_decode, params["vae"], gt_seq, mesh, num_views, decode_chunk_size
        )
        pred_views = _decode_views(
            jit_decode, params["vae"], pred_seq, mesh, num_views, decode_chunk_size
        )
        gt_video = _tile_views(gt_views)
        pred_video = _tile_views(pred_views)
        decode_s = time.perf_counter() - t_dec
        total_decode_s += decode_s

        # tiled/ holds the GT-vs-pred comparison; cam{i}/ holds the predicted
        # rollout for one camera. Same basename in every directory.
        name = f"video_{ep_idx:04d}"
        path = os.path.join(output_dir, "tiled", f"{name}.mp4")
        _save_comparison_video(gt_video, pred_video, path, fps=fps)
        view_paths = _save_per_view_videos(pred_views, output_dir, name, fps=fps)
        n_written += 1
        n_done = ep_idx + 1
        elapsed = time.perf_counter() - t_sweep
        max_logging.log(
            f"[ctrl_world_infer]    saved {name}.mp4 -> tiled/ + "
            f"{'/'.join(os.path.basename(os.path.dirname(p)) for p in view_paths)}  "
            f"{keep} frames @ {fps}fps ({keep / fps:.1f}s), "
            f"tile {pred_video.shape[2]}x{pred_video.shape[1] * 2}"
        )
        max_logging.log(
            f"[ctrl_world_infer]    episode {ep_idx} done in "
            f"{rollout_s + decode_s:.1f}s (rollout {rollout_s:.1f}s, "
            f"decode {decode_s:.1f}s)  |  {n_done} episodes, "
            f"{_fmt_dur(elapsed)} elapsed, {_fmt_dur(elapsed / n_done)}/episode avg"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_sweep
    max_logging.log("***** Rollout inference complete *****")
    max_logging.log(
        f"  Videos written:   {n_written} per directory under {output_dir} "
        f"(tiled/ + cam0..cam{num_views - 1}/)"
    )
    max_logging.log(f"  Trimmed short:    {n_trimmed} of {n_written} "
                    f"(episode shorter than the {horizon}-frame horizon)")
    max_logging.log(f"  Wall clock:       {_fmt_dur(elapsed)}")
    if n_written:
        max_logging.log(
            f"  Per episode:      {_fmt_dur(elapsed / n_written)} avg "
            f"(rollout {total_rollout_s / n_written:.1f}s, "
            f"decode {total_decode_s / n_written:.1f}s)"
        )


if __name__ == "__main__":
    app.run(run)
