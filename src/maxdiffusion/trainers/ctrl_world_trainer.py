"""Action-conditioned SVD (Ctrl-World) trainer.

Loads UNet + VAE config from the SVD HF-Diffusers directory at
``config.pretrained_model_name_or_path`` (with ``from_pt=True`` so the same
loader handles both the upstream SVD repo and the directory produced by
``scripts/convert_ctrl_world_ckpt.py``). The action encoder is fresh-init by
default; pass ``action_encoder_init_path`` to warm-start from a Ctrl-World
checkpoint.

Latents are pre-encoded on disk (see docs/ctrl_world_data_format.md), so the
trainer never instantiates the VAE — it only needs the scaling factor (read
from ``config.vae_scaling_factor``) to unscale the channel-concat conditioning
stream inside ``action_world_train_step``.

State is FSDP-sharded across the ``fsdp`` mesh axis using the logical
partition annotations baked into ``FlaxVideoUNet`` (action-encoder params are
tiny and fall back to replicated). Data is sharded along all named mesh axes
(``[data, fsdp, context, tensor]``) along the batch axis — concretely on a
single-host 8-chip v6e mesh that becomes pure data-parallel data sharding.

Checkpoints are written with everything needed to resume mid-training: params,
optimizer state (so the LR schedule / Adam moments continue where they left
off), the step counter, the training RNG, and a restart counter. On restart the
restart counter is bumped and folded into the data-pipeline seed, so every
resumed run walks the dataset in a fresh shuffle order rather than replaying the
same window sequence from step 0 (see ``reshuffle_data_on_restart``).
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.linen import partitioning as nn_partitioning
from flax.linen.spmd import LogicallyPartitioned
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P
from safetensors.torch import load_file as load_torch_safetensors

from maxdiffusion import max_logging, max_utils
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.svd.skeleton_encoder_flax import (
    FlaxSkeletonPatchEmbed,
    FlaxSkeletonAdaLNProjector,
    FlaxSkeletonCrossAttnEmbed,
)
from maxdiffusion.models.svd.action_encoder_flax import (
    FlaxActionAdaLNProjector,
    FlaxActionEncoder,
)
from maxdiffusion.models.svd.ctrl_world_flax import (
    CtrlWorldTrainConfig,
    action_world_train_step,
    _is_skeleton_mode,
    _skeleton_apply_key,
)
from maxdiffusion.models.svd.video_unet_flax import FlaxVideoUNet


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_ctrl_world_train_config(config) -> CtrlWorldTrainConfig:
    return CtrlWorldTrainConfig(
        num_history=config.num_history,
        num_frames=config.num_frames,
        action_dim=config.action_dim,
        hidden_size=config.hidden_size,
        text_embed_dim=config.text_embed_dim,
        p_mean=config.ctrl_p_mean,
        p_std=config.ctrl_p_std,
        cond_aug_max=config.ctrl_cond_aug_max,
        history_sigma_std=config.ctrl_history_sigma_std,
        cfg_drop_prob=config.ctrl_cfg_drop_prob,
        fps_id=config.ctrl_fps_id,
        motion_bucket_id=config.ctrl_motion_bucket_id,
        noise_aug_strength=config.ctrl_noise_aug_strength,
        his_cond_zero=config.ctrl_his_cond_zero,
        action_cond_mode=max_utils.config_get(config, "action_cond_mode", "cross_attn"),
        time_embed_dim=_unet_time_embed_dim(config),
        # conv_in's output width — the additive skeleton route must match it.
        # Same source as time_embed_dim so the two can never disagree.
        model_channels=_unet_model_channels(config),
        skeleton_embed_alpha=float(
            max_utils.config_get(config, "skeleton_embed_alpha", 0.1)
        ),
        skeleton_cross_attn_stride=int(
            max_utils.config_get(config, "skeleton_cross_attn_stride", 4)
        ),
        use_task_instructions=max_utils.config_get(config, "use_task_instructions", True),
    )


def _unet_model_channels(config) -> int:
    """``block_out_channels[0]`` (320 for base SVD) — conv_in's output width.

    Read only by the additive skeleton route, whose embedded output is added onto
    conv_in's result and so has to be exactly this wide.
    """
    boc = max_utils.config_get(config, "block_out_channels", None)
    return boc[0] if boc else 320


def _unet_time_embed_dim(config) -> int:
    """SVD's timestep-embedding width: block_out_channels[0] * 4 (1280 for base SVD).

    Only used by the adaln action route, which must project into exactly this
    width to be summable with the UNet's t_emb.
    """
    boc = max_utils.config_get(config, "block_out_channels", None)
    return (boc[0] if boc else 320) * 4


VALID_ACTION_COND_MODES = (
    "cross_attn",
    "adaln",
    "skeleton",
    "skeleton_adaln",
    "skeleton_cross_attn",
)


def _load_action_encoder_params(path: str, dtype) -> Dict[str, Any]:
    """Load Ctrl-World's 3-layer MLP from a torch safetensors dump."""
    pt = load_torch_safetensors(path)
    return {
        "linear_1": {
            "kernel": jnp.asarray(pt["action_encode.0.weight"].numpy().T, dtype=dtype),
            "bias":   jnp.asarray(pt["action_encode.0.bias"].numpy(),    dtype=dtype),
        },
        "linear_2": {
            "kernel": jnp.asarray(pt["action_encode.2.weight"].numpy().T, dtype=dtype),
            "bias":   jnp.asarray(pt["action_encode.2.bias"].numpy(),    dtype=dtype),
        },
        "linear_3": {
            "kernel": jnp.asarray(pt["action_encode.4.weight"].numpy().T, dtype=dtype),
            "bias":   jnp.asarray(pt["action_encode.4.bias"].numpy(),    dtype=dtype),
        },
    }


def _dtype_from_str(name):
    # pyconfig.user_init already coerces these keys to jnp.dtype, but accept plain
    # strings too so the helper works on raw config values.
    return {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[jnp.dtype(name).name]


def _maybe_unbox(x):
    if isinstance(x, LogicallyPartitioned):
        return x.unbox()
    return x


def _unbox_tree(tree):
    return jax.tree_util.tree_map(
        _maybe_unbox, tree, is_leaf=lambda x: isinstance(x, LogicallyPartitioned)
    )


# Kept in float32 regardless of ``weights_dtype``: normalisation scales/biases
# and the two small conditioning embedders (4.7M params total, so the memory cost
# is negligible) are the parts most sensitive to reduced precision. Mirrors
# ``cast_with_exclusion`` in the wan/ltx2 pipelines, with names for this UNet.
_F32_PARAM_KEYWORDS = ("norm", "time_embedding", "add_embedding")


def _cast_weight(path, x, dtype):
    """Cast one loaded weight to ``dtype``, excluding precision-sensitive leaves."""
    if not (hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)):
        return x
    key = jax.tree_util.keystr(path).lower()
    if any(k in key for k in _F32_PARAM_KEYWORDS):
        return x.astype(jnp.float32)
    return x.astype(dtype)


def _fill_adagn_params(abstract, loaded, dtype):
    """Materialise the AdaGN scale projections no SVD checkpoint can supply.

    With ``adagn=True`` every spatial and temporal resnet gains an
    ``adagn_scale_proj`` Dense that the pretrained weights know nothing about.
    ``from_pretrained`` does no missing-key reconciliation here — it returns
    exactly what the torch converter produced — and ``_rebox_like``'s
    ``tree_map`` requires both trees to have the same structure, so the new
    leaves have to be created explicitly.

    Zeros are not a placeholder: they are what the module's own initialiser
    would produce (``kernel_init=zeros``), and scale=0 makes ``(1 + scale)``
    exactly 1, so the multiplicative path is inert until training moves it.

    Any *other* missing key is a real load failure and raises rather than being
    silently zero-filled.
    """
    flat_a = flatten_dict(abstract)
    flat_l = flatten_dict(loaded)

    unexpected = sorted(k for k in flat_l if k not in flat_a)
    if unexpected:
        raise ValueError(
            f"[ctrl_world] checkpoint has {len(unexpected)} params the UNet does not "
            f"declare, e.g. {unexpected[:3]}"
        )
    missing = sorted(k for k in flat_a if k not in flat_l)
    not_adagn = [k for k in missing if "adagn_scale_proj" not in k]
    if not_adagn:
        raise ValueError(
            f"[ctrl_world] {len(not_adagn)} non-AdaGN params missing from the "
            f"checkpoint, e.g. {not_adagn[:3]} — this is a load failure, not a "
            "new-parameter case."
        )
    for k in missing:
        a = flat_a[k]
        spec = a.value if isinstance(a, LogicallyPartitioned) else a
        flat_l[k] = jnp.zeros(spec.shape, dtype)
    return unflatten_dict(flat_l), len(missing)


def _rebox_like(abstract, loaded):
    """Re-attach the model's logical axis names to loaded checkpoint params.

    ``from_pretrained`` returns raw arrays — ``convert_pytorch_state_dict_to_flax``
    rebuilds the tree from scratch and drops every ``LogicallyPartitioned``
    wrapper the module definitions put there. Without the wrappers
    ``nn.get_partition_spec`` reports ``P()`` for every leaf, so the whole model
    ends up replicated on every device no matter what ``logical_axis_rules``
    says. ``abstract`` is an ``init_weights(eval_only=True)`` tree, which *is*
    boxed; copy its names onto the real arrays.
    """
    return jax.tree_util.tree_map(
        lambda a, x: a.replace(value=x) if isinstance(a, LogicallyPartitioned) else x,
        abstract,
        loaded,
        is_leaf=lambda x: isinstance(x, LogicallyPartitioned),
    )


def _place_on_sharding(x, sharding):
    """Multi-host-safe stand-in for ``jax.device_put(x, sharding)``.

    ``jax.device_put`` rejects any ``NamedSharding`` whose mesh includes devices
    this process cannot address, which is every mesh on a multi-host TPU slice.
    Each process here holds an identical host-local copy of every leaf (weights
    read from the same checkpoint, action encoder init'd from the same seed,
    optimizer state derived from those), so each can carve its own shards out of
    that copy — which is what ``max_utils.device_put_replicated`` does via
    ``jax.make_array_from_callback``.

    ``sharding`` is ``None`` for leaves that carry no shape (``state.step`` is a
    plain python int), and those need no placement.
    """
    if sharding is None or not hasattr(x, "shape"):
        return x
    if isinstance(x, jax.Array):
        spec = getattr(x.sharding, "spec", None)
        if spec is not None and any(axis is not None for axis in spec):
            # Already non-trivially sharded, so ``addressable_data(0)`` is one
            # shard rather than the whole tensor and the callback path would
            # silently corrupt it. A global array can be resharded directly.
            return jax.device_put(x, sharding)
        # Drop to numpy first: the loaded weights are CPU-resident jax arrays and
        # slicing those per shard would dispatch ~1400 leaves x mesh-size jitted
        # slices, where numpy slicing is a plain host memcpy.
        x = np.asarray(x.addressable_data(0))
    return max_utils.device_put_replicated(x, sharding)


def _count_nonfinite(params) -> int:
    """Total non-finite elements across a params pytree.

    One jitted reduction over the whole tree, so the cost is a scan of already-
    resident device memory rather than a host round-trip of 1.5B params.
    """
    @jax.jit
    def _count(tree):
        return sum(
            jnp.sum(~jnp.isfinite(x)).astype(jnp.int32)
            for x in jax.tree_util.tree_leaves(tree)
            if jnp.issubdtype(x.dtype, jnp.floating)
        )

    return int(_count(params))


def _assert_params_finite(params, what: str, hint: str = "") -> None:
    n_bad = _count_nonfinite(params)
    if n_bad:
        raise ValueError(
            f"[ctrl_world] {what} contains {n_bad} non-finite parameter values, so this "
            f"run would train (and log) NaN from its very first step.{' ' + hint if hint else ''}"
        )


def _is_typed_key(x) -> bool:
    prng_dtype = getattr(jax.dtypes, "prng_key", None)
    return prng_dtype is not None and jnp.issubdtype(x.dtype, prng_dtype)


def _rng_to_list(rng) -> list[int]:
    """Serialise a PRNG key to plain ints so it can ride along in the JSON item."""
    raw = jax.random.key_data(rng) if _is_typed_key(rng) else rng
    return [int(v) for v in np.asarray(raw).reshape(-1)]


def _rng_from_list(values, like):
    """Inverse of ``_rng_to_list``; ``like`` supplies dtype/shape/impl."""
    raw = np.asarray(values, dtype=np.uint32)
    if _is_typed_key(like):
        return jax.random.wrap_key_data(
            jnp.asarray(raw.reshape(np.asarray(jax.random.key_data(like)).shape))
        )
    return jnp.asarray(raw.reshape(like.shape), dtype=like.dtype)


# ── Trainer ──────────────────────────────────────────────────────────────────


class CtrlWorldTrainer:
    """Self-contained Linen trainer for action-conditioned SVD."""

    def __init__(self, config):
        self.config = config
        mode = getattr(config, "action_cond_mode", "cross_attn")
        if mode not in VALID_ACTION_COND_MODES:
            raise ValueError(
                f"action_cond_mode={mode!r} is not one of {VALID_ACTION_COND_MODES}."
            )
        if _is_skeleton_mode(mode) and int(
            max_utils.config_get(config, "wandb_video_every", 0)
        ) > 0:
            max_logging.log(
                f"[ctrl_world] WARNING: action_cond_mode={mode} — the in-training "
                "W&B video preview is skipped (FlaxCtrlWorldPipeline has no skeleton "
                "route yet). Training and eval loss are unaffected."
            )
        self.dtype = _dtype_from_str(config.activations_dtype)
        self.weights_dtype = _dtype_from_str(config.weights_dtype)
        self.train_cfg = _build_ctrl_world_train_config(config)
        # Overwritten in start_training once the checkpoint (if any) is read.
        self.restart_count = 0
        # Step at which loss/grad_norm first went non-finite, or None.
        self._first_nonfinite_step = None
        self.data_seed = int(config.seed)
        # Set up in start_training; stays None on non-zero hosts and when
        # wandb_project is empty, so every log site must guard on it.
        self._wandb_run = None
        # Lazily built by _log_wandb_videos, and only when video logging is on:
        # the rollout needs a VAE decoder, which training otherwise never loads
        # (latents are pre-encoded on disk).
        self._video_pipeline = None
        self._video_vae_params = None
        self._video_eval_iter = None
        self._video_modules = None

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _build_mesh(self):
        devices = max_utils.create_device_mesh(self.config)
        return jax.sharding.Mesh(devices, self.config.mesh_axes)

    def _load_modules(self, mesh):
        max_logging.log(
            f"[ctrl_world] loading UNet from {self.config.pretrained_model_name_or_path}/unet"
        )
        # AdaGN is tied to the conditioning mode, not exposed as its own knob:
        # in adaln mode the action rides t_emb, so the resnets' t_emb injection
        # IS the action pathway and it needs to be multiplicative to match the
        # WAN arm's AdaLN. In cross_attn mode the action never touches t_emb, so
        # the UNet stays exactly as pretrained.
        adagn = self.train_cfg.action_cond_mode == "adaln"
        with mesh:
            unet, unet_params = FlaxVideoUNet.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder="unet",
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                adagn=adagn,
                from_pt=self.config.from_pt,
                use_safetensors=True,
                attention_kernel=self.config.attention,
                temporal_attention_kernel=self.config.temporal_attention,
                use_memory_efficient_attention=self.config.use_memory_efficient_attention,
                flash_block_sizes=max_utils.get_flash_block_sizes(self.config) or {},
                flash_min_seq_length=self.config.flash_min_seq_length,
                mesh=mesh,
                precision=max_utils.get_precision(self.config),
                norm_num_groups=self.config.norm_num_groups,
            )
        # ``from_pretrained``'s ``dtype`` is the *computation* dtype and
        # ``weights_dtype`` only reaches ``param_dtype`` on the ``.init()`` path,
        # which loading never takes — the from_pt converter forces f32
        # (``v.float().numpy()``). So the returned params keep the checkpoint's
        # dtype and ``config.weights_dtype`` would be silently ignored. Cast here
        # so it is honoured. These arrays are still CPU-resident, so this costs
        # host RAM, not HBM.
        unet_params = jax.tree_util.tree_map_with_path(
            lambda path, x: _cast_weight(path, x, self.weights_dtype), unet_params
        )
        # ...and the same converter drops the logical axis annotations, so put
        # them back or nothing downstream can shard. See ``_rebox_like``.
        #
        # Deliberately NOT under nn_partitioning.axis_rules: the flash-attention
        # path calls nn.logical_to_mesh_axes without explicit rules, so an active
        # rule set resolves 'activation_batch' to ('data', 'fsdp') and shard_map
        # then demands the init batch (batch=1 x num_frames=2) divide by the fsdp
        # mesh size. With no rules in context the logical names resolve to None
        # and shard_map replicates, which is how from_pretrained's own internal
        # init_weights call already runs. The LogicallyPartitioned names this
        # produces are identical either way — the rules only matter later, in
        # _build_sharded_state, where they are passed to
        # nn.logical_to_mesh_sharding explicitly.
        with mesh:
            abstract_unet_params = unet.init_weights(
                jax.random.PRNGKey(self.config.seed), eval_only=True
            )
        unet_params, n_adagn = _fill_adagn_params(
            abstract_unet_params, unet_params, self.weights_dtype
        )
        if adagn:
            max_logging.log(
                f"[ctrl_world] AdaGN enabled (action_cond_mode=adaln): every resnet now "
                f"applies norm2(h)*(1+scale)+shift instead of norm2(h+shift); "
                f"{n_adagn} zero-init adagn_scale_proj leaves added. The pretrained "
                f"time_emb_proj is reused as the shift, but it now acts AFTER the norm, "
                f"so step 0 is close to — not identical to — pretrained SVD."
            )
        unet_params = _rebox_like(abstract_unet_params, unet_params)

        mode = self.train_cfg.action_cond_mode
        skeleton_mode = _is_skeleton_mode(mode)

        # No action encoder in the skeleton modes: the conditioning is the
        # rendered skeleton video, so an encoder here would receive zero gradient
        # forever — dead weights in the checkpoint and dead optimizer moments in
        # HBM. Mirrors WanCtrlWorldTrainer._build_action_encoder.
        action_encoder, ae_params = None, None
        if skeleton_mode:
            if self.config.action_encoder_init_path:
                raise ValueError(
                    f"action_cond_mode={mode!r} builds no action encoder, so "
                    "action_encoder_init_path cannot apply. Drop it, or switch to "
                    "a vector-action mode."
                )
            max_logging.log(
                f"[ctrl_world] action_cond_mode={mode}: no action encoder "
                "(conditioning is the rendered skeleton video; vector actions unused)"
            )
        else:
            action_encoder = FlaxActionEncoder(
                action_dim=self.config.action_dim,
                hidden_size=self.config.hidden_size,
                text_embed_dim=self.config.text_embed_dim,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
            )
        if action_encoder is not None and self.config.action_encoder_init_path:
            max_logging.log(
                f"[ctrl_world] loading action encoder from {self.config.action_encoder_init_path}"
            )
            ae_params = _load_action_encoder_params(
                self.config.action_encoder_init_path, self.weights_dtype
            )
        elif action_encoder is not None:
            max_logging.log("[ctrl_world] initialising action encoder from scratch")
            ae_params = action_encoder.init_weights(
                jax.random.PRNGKey(self.config.seed), batch=1, num_frames=1
            )

        # Exactly one optional conditioning module is ever live. Collect it in a
        # dict keyed by its PARAM NAME so every downstream consumer (params tree,
        # apply_fns, checkpoint) is a single loop rather than a tuple that grows
        # by two entries per mode.
        cond_extras: Dict[str, Any] = {}

        if skeleton_mode:
            key = _skeleton_apply_key(mode)
            if mode == "skeleton":
                skel_mod = FlaxSkeletonPatchEmbed(
                    model_channels=self.train_cfg.model_channels,
                    alpha=self.train_cfg.skeleton_embed_alpha,
                    dtype=self.dtype,
                    weights_dtype=self.weights_dtype,
                )
                max_logging.log(
                    f"[ctrl_world] action_cond_mode=skeleton: skeleton latents are "
                    f"patch-embedded (alpha={self.train_cfg.skeleton_embed_alpha}) and "
                    f"ADDED onto conv_in's output (width {self.train_cfg.model_channels}); "
                    "cross-attention carries the text embedding only"
                )
            elif mode == "skeleton_adaln":
                skel_mod = FlaxSkeletonAdaLNProjector(
                    time_embed_dim=self.train_cfg.time_embed_dim,
                    dtype=self.dtype,
                    weights_dtype=self.weights_dtype,
                )
                max_logging.log(
                    "[ctrl_world] action_cond_mode=skeleton_adaln: skeleton latents are "
                    f"POOLED to one vector per frame and summed into t_emb (width "
                    f"{self.train_cfg.time_embed_dim}). NOTE this site has no spatial "
                    "axis in SVD, so the pooling discards the skeleton's spatial "
                    "structure — expected to be the weakest of the three skeleton routes"
                )
            else:
                skel_mod = FlaxSkeletonCrossAttnEmbed(
                    hidden_size=self.config.hidden_size,
                    stride=self.train_cfg.skeleton_cross_attn_stride,
                    latent_height=self.config.latent_height_per_cam * self.config.num_views,
                    latent_width=self.config.width // 8,
                    dtype=self.dtype,
                    weights_dtype=self.weights_dtype,
                )
                max_logging.log(
                    f"[ctrl_world] action_cond_mode=skeleton_cross_attn: skeleton latents "
                    f"become {skel_mod.num_tokens} spatial K/V tokens per frame "
                    f"(stride {self.train_cfg.skeleton_cross_attn_stride}) with a learned "
                    "positional embedding standing in for the rotary embeddings SVD lacks"
                )
            cond_extras[key] = (
                skel_mod,
                skel_mod.init_weights(jax.random.PRNGKey(self.config.seed + 2)),
            )

        adaln = mode == "adaln"
        adaln_proj, adaln_params = None, None
        if adaln:
            if self.config.action_encoder_init_path:
                # The converted Ctrl-World encoder was trained to drive
                # cross-attention, and its linear_3 is non-zero, so adaln would
                # start far off the pretrained operating point instead of at it.
                raise ValueError(
                    "action_cond_mode='adaln' cannot be combined with "
                    "action_encoder_init_path: the upstream weights were trained "
                    "for the cross-attention route. Train adaln from scratch, or "
                    "switch to action_cond_mode='cross_attn'."
                )
            adaln_proj = FlaxActionAdaLNProjector(
                time_embed_dim=self.train_cfg.time_embed_dim,
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
            )
            adaln_params = adaln_proj.init_weights(
                jax.random.PRNGKey(self.config.seed + 1), batch=1, num_frames=1,
                hidden_size=self.config.hidden_size,
            )
            cond_extras["action_adaln_proj"] = (adaln_proj, adaln_params)
            max_logging.log(
                "[ctrl_world] action_cond_mode=adaln: action tokens are summed into "
                f"t_emb (width {self.train_cfg.time_embed_dim}); cross-attention "
                "carries the text embedding only and per-frame cross-attn is off"
            )

        if self.train_cfg.use_task_instructions:
            if mode == "skeleton_cross_attn":
                # SVD's context is inherently per-(sample, frame), so unlike the
                # WAN arm — which must POOL the instruction onto the action
                # tokens because its K/V is frame-locked by reshape — the full
                # text token simply rides alongside the skeleton grid.
                route = "one extra cross-attention key alongside the skeleton grid"
            elif adaln or skeleton_mode:
                route = "the cross-attention context"
            else:
                route = "tiled into the action tokens"
            max_logging.log(
                f"[ctrl_world] task instructions: ON — CLIP text is {route}; "
                "not CFG-dropped"
            )
        else:
            max_logging.log(
                "[ctrl_world] task instructions: OFF — action-only conditioning"
            )

        return unet, unet_params, action_encoder, ae_params, cond_extras

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

    # ── Sharding-aware state construction ──────────────────────────────────────

    def _build_sharded_state(self, mesh, unet, unet_params, action_encoder, ae_params, tx,
                             cond_extras=None):
        """Build a TrainState whose leaves are FSDP-sharded across the mesh.

        Path:
          1. Build a *boxed* TrainState (UNet params carry the
             ``LogicallyPartitioned`` wrappers re-attached by ``_rebox_like`` in
             ``_load_modules``; ``tx.init``'s tree_map propagates them through
             optimizer state, so mu/nu shard the same way the params do).
          2. Read partition specs from the boxed tree and translate logical →
             mesh shardings via ``nn.logical_to_mesh_sharding``.
          3. Unbox both trees to drop the wrappers, leaving raw arrays and a
             matching shardings tree.
          4. Place each leaf onto its sharding — this is where data actually
             leaves device 0 and gets sharded.
        """
        config = self.config

        # Step 1 — boxed state. Note that we feed unet_params unmodified;
        # tx.init's tree_map preserves the LogicallyPartitioned wrappers.
        params = {"unet": unet_params}
        # action_encoder is absent in the skeleton modes (no vector actions), so
        # it is only added when it exists — a None leaf would break tx.init.
        if ae_params is not None:
            params["action_encoder"] = ae_params
        for name, (_mod, mod_params) in (cond_extras or {}).items():
            params[name] = mod_params
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            state = train_state.TrainState.create(
                apply_fn=lambda *a, **k: None, params=params, tx=tx
            )
            # Step 2 — derive shardings from the boxed tree.
            state_logical_specs = nn.get_partition_spec(state)
            state_shardings = nn.logical_to_mesh_sharding(
                state_logical_specs, mesh, config.logical_axis_rules
            )
        # Step 3 — drop LogicallyPartitioned wrappers; downstream tx.update and
        # apply_fns expect raw arrays. The shardings tree was derived from the
        # *boxed* state, so it carries the same wrappers in its tree structure;
        # unbox those too so jit's in/out_shardings line up with the state and
        # so the two trees can be walked leaf-by-leaf below.
        state = max_utils.unbox_logicallypartioned_trainstate(state)
        state_shardings = max_utils.unbox_logicallypartioned_trainstate(state_shardings)
        # Step 4 — actually shard onto devices.
        state = jax.tree_util.tree_map(_place_on_sharding, state, state_shardings)
        return state, state_shardings

    # ── Steps ──────────────────────────────────────────────────────────────────

    def _build_train_step(self, apply_fns, state_shardings, data_shardings):
        """Jitted train step."""
        cfg = self.train_cfg
        vae_scaling_factor = float(self.config.vae_scaling_factor)
        weights_dtype = self.weights_dtype

        def loss_fn(params, batch, rng):
            return action_world_train_step(
                rng=rng, params=params, apply_fns=apply_fns, batch=batch,
                cfg=cfg, vae_scaling_factor=vae_scaling_factor, train=True,
            )

        grad_fn = jax.value_and_grad(loss_fn)

        def step_fn(state, batch, rng):
            batch = jax.tree_util.tree_map(
                lambda x: x.astype(weights_dtype) if x.dtype.kind == "f" else x, batch
            )
            loss, grads = grad_fn(state.params, batch, rng)
            grad_norm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2)
                                     for g in jax.tree_util.tree_leaves(grads)))
            new_state = state.apply_gradients(grads=grads)
            metrics = {"loss": loss, "grad_norm": grad_norm}
            return new_state, metrics

        return jax.jit(
            step_fn,
            in_shardings=(state_shardings, data_shardings, None),
            out_shardings=(state_shardings, None),
            donate_argnums=(0,),
        )

    def _build_eval_step(self, apply_fns, state_shardings, data_shardings):
        cfg = self.train_cfg
        vae_scaling_factor = float(self.config.vae_scaling_factor)
        weights_dtype = self.weights_dtype

        def eval_loss(params, batch, rng):
            batch = jax.tree_util.tree_map(
                lambda x: x.astype(weights_dtype) if x.dtype.kind == "f" else x, batch
            )
            return action_world_train_step(
                rng=rng, params=params, apply_fns=apply_fns, batch=batch,
                cfg=cfg, vae_scaling_factor=vae_scaling_factor, train=False,
            )

        # Wrap to take a TrainState (so we can pass the same state object) and
        # return only the loss.
        def step_fn(state, batch, rng):
            return eval_loss(state.params, batch, rng)

        return jax.jit(
            step_fn,
            in_shardings=(state_shardings, data_shardings, None),
            out_shardings=None,
        )

    # ── Training loop ──────────────────────────────────────────────────────────

    def start_training(self):
        config = self.config
        mesh = self._build_mesh()
        unet, unet_params, action_encoder, ae_params, cond_extras = (
            self._load_modules(mesh)
        )

        apply_fns = {"unet": unet.apply}
        if action_encoder is not None:
            apply_fns["action_encoder"] = action_encoder.apply
        for name, (mod, _p) in cond_extras.items():
            apply_fns[name] = mod.apply
        # Kept for the W&B video rollout, which needs the module objects (not just
        # their apply_fns) to build an inference pipeline.
        self._video_modules = (
            unet, action_encoder, cond_extras.get("action_adaln_proj", (None, None))[0]
        )
        self._cond_extra_names = tuple(cond_extras)
        tx, lr_schedule = self._build_optimizer(config.max_train_steps)

        state, state_shardings = self._build_sharded_state(
            mesh, unet, unet_params, action_encoder, ae_params, tx,
            cond_extras=cond_extras,
        )
        del unet_params, ae_params, cond_extras  # freed inside state

        if jax.process_index() == 0:
            num_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(state.params))
            max_logging.log(f"[ctrl_world] trainable params: {num_params / 1e6:.1f}M")

        # Data shardings — match the global array layout produced by
        # MultiHostDataLoadIterator (sharded along axis 0 over all named axes).
        batch_pspec = NamedSharding(mesh, P(*config.data_sharding))
        data_shardings = {
            "latent":      batch_pspec,
            "action":      batch_pspec,
            "text_embeds": batch_pspec,
        }
        # Must mirror the dataset's key set exactly — jit's in_shardings is a
        # pytree prefix match, so a key the batch carries but this dict omits
        # (or vice versa) is a trace-time structure mismatch, not a warning.
        if _is_skeleton_mode(self.train_cfg.action_cond_mode):
            data_shardings["skeleton"] = batch_pspec

        train_step_fn = self._build_train_step(apply_fns, state_shardings, data_shardings)
        eval_step_fn = self._build_eval_step(apply_fns, state_shardings, data_shardings)

        # ── Checkpointing / resume ────────────────────────────────────────────
        # Restore *before* building the data iterator: the restored restart
        # counter is what seeds the (re)shuffle for this run.
        ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
        ckpt_mgr = self._build_checkpoint_manager(ckpt_dir)
        state, start_step, ckpt_meta = self._maybe_restore(ckpt_mgr, state, state_shardings)

        # Restart counter is monotonic across launches and identical on every
        # host (it comes out of the checkpoint), so hosts stay in lockstep.
        self.restart_count = int(ckpt_meta.get("restart_count", -1)) + 1
        self.data_seed = self._data_seed(self.restart_count)

        rng = jax.random.PRNGKey(config.seed + 1)
        if "rng" in ckpt_meta:
            rng = _rng_from_list(ckpt_meta["rng"], rng)
        if start_step:
            max_logging.log(
                f"[ctrl_world] resumed at step {start_step} "
                f"(restart #{self.restart_count}, data_seed={self.data_seed}, "
                f"rng={'restored' if 'rng' in ckpt_meta else 'reseeded from config.seed'})"
            )

        train_iter = make_data_iterator(
            config,
            jax.process_index(),
            jax.process_count(),
            mesh,
            self._global_batch_size_to_load(),
            is_training=True,
            seed=self.data_seed,
        )

        if jax.process_index() == 0:
            max_logging.log("***** Running training *****")
            max_logging.log(f"  Per-host batch size: {self._global_batch_size_to_load() // jax.process_count()}")
            max_logging.log(f"  Global batch size:   {self._global_batch_size_to_load()}")
            max_logging.log(f"  Devices:             {jax.device_count()}")
            max_logging.log(f"  Max train steps:     {config.max_train_steps}")
            max_logging.log(f"  Start step:          {start_step}")
            max_logging.log(f"  Output dir:          {config.output_dir}")
            max_logging.log(f"  Checkpoint dir:      {ckpt_dir}")
            max_logging.log(f"  Save optimizer:      {config.save_optimizer}")
            max_logging.log(f"  Data seed:           {self.data_seed} (restart #{self.restart_count})")

        if jax.process_index() == 0 and getattr(config, "wandb_project", ""):
            import wandb
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                entity=getattr(config, "wandb_entity", None) or None,
                name=config.run_name or None,
            )
        wandb_run = self._wandb_run

        recent_loss: list[float] = []
        recent_grad: list[float] = []
        last_step_time = datetime.datetime.now()

        for step in range(start_step, config.max_train_steps):
            batch = next(train_iter)
            rng, step_rng = jax.random.split(rng)

            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                state, metrics = train_step_fn(state, batch, step_rng)
                metrics["loss"].block_until_ready()

            loss_v = float(metrics["loss"])
            grad_v = float(metrics["grad_norm"])
            # The periodic line below reports a mean over log_period steps, so a
            # single non-finite step poisons the window and every later one — with
            # log_period=100 that hides both *when* divergence started and whether
            # step 0 was already bad. Report the first occurrence the moment it
            # happens; it is the difference between "the forward pass or the data
            # is broken" and "training diverged after N steps".
            if self._first_nonfinite_step is None and not (
                np.isfinite(loss_v) and np.isfinite(grad_v)
            ):
                self._first_nonfinite_step = step
                max_logging.log(
                    f"[ctrl_world] FIRST non-finite metric at step {step}: "
                    f"loss={loss_v} grad_norm={grad_v}"
                )
            recent_loss.append(loss_v)
            recent_grad.append(grad_v)
            now = datetime.datetime.now()
            if (step + 1) % config.log_period == 0 and jax.process_index() == 0:
                lr = float(lr_schedule(step))
                # Average the finite steps only, so one NaN does not erase the
                # signal from the other 99.
                ok_loss = [x for x in recent_loss if np.isfinite(x)]
                ok_grad = [x for x in recent_grad if np.isfinite(x)]
                n_bad = len(recent_loss) - len(ok_loss)
                avg_loss = sum(ok_loss) / len(ok_loss) if ok_loss else float("nan")
                avg_grad = sum(ok_grad) / len(ok_grad) if ok_grad else float("nan")
                steps_per_sec = config.log_period / (now - last_step_time).total_seconds()
                max_logging.log(
                    f"step {step + 1}/{config.max_train_steps} "
                    f"loss={avg_loss:.4f} grad_norm={avg_grad:.3f} "
                    f"lr={lr:.2e} steps/s={steps_per_sec:.2f}"
                    + (
                        f" nonfinite={n_bad}/{len(recent_loss)}"
                        f" (first at step {self._first_nonfinite_step})"
                        if n_bad
                        else ""
                    )
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": avg_loss,
                            "train/loss_max": max(ok_loss) if ok_loss else float("nan"),
                            "train/grad_norm": avg_grad,
                            "train/grad_norm_max": max(ok_grad) if ok_grad else float("nan"),
                            "train/nonfinite_steps": n_bad,
                            "train/lr": lr,
                            "train/steps_per_sec": steps_per_sec,
                        },
                        step=step + 1,
                    )
                recent_loss.clear()
                recent_grad.clear()
                last_step_time = now

            if (
                config.eval_every > 0
                and (step + 1) % config.eval_every == 0
            ):
                self._run_eval(eval_step_fn, state, mesh, step + 1, rng)

            # Gated on the config, not on wandb_run: the rollout and VAE decode
            # are collective, so every host must enter. Only process 0 logs.
            # FlaxCtrlWorldPipeline has no skeleton route yet, so the in-training
            # video preview is unavailable in those modes. Gated here rather than
            # left to fail inside the rollout, and reported once at startup.
            if (
                int(max_utils.config_get(config, "wandb_video_every", 0)) > 0
                and getattr(config, "wandb_project", "")
                and not _is_skeleton_mode(self.train_cfg.action_cond_mode)
                and (step + 1) % int(config.wandb_video_every) == 0
            ):
                rng, video_rng = jax.random.split(rng)
                self._log_wandb_videos(state, mesh, step + 1, video_rng)

            if (
                config.checkpoint_every > 0
                and (step + 1) % config.checkpoint_every == 0
            ):
                self._save_checkpoint(ckpt_mgr, step + 1, state, rng)

        if config.save_final_checkpoint:
            self._save_checkpoint(ckpt_mgr, config.max_train_steps, state, rng)
        ckpt_mgr.wait_until_finished()
        ckpt_mgr.close()
        if wandb_run is not None:
            wandb_run.finish()

    # ── Eval ───────────────────────────────────────────────────────────────────

    def _run_eval(self, eval_step_fn, state, mesh, step: int, rng):
        config = self.config
        if not config.eval_data_dir:
            max_logging.log("[ctrl_world] eval_every>0 but eval_data_dir is empty; skipping eval")
            return
        max_logging.log(f"[ctrl_world] starting eval at step {step}")
        # Fixed seed (never the per-restart data seed) so eval loss stays
        # comparable across steps and across restarts.
        eval_iter = make_data_iterator(
            config,
            jax.process_index(),
            jax.process_count(),
            mesh,
            self._global_batch_size_to_load(),
            is_training=False,
            seed=config.seed,
        )
        max_batches = max(1, int(getattr(config, "eval_max_batches", 50)))
        losses: list[float] = []
        eval_start = datetime.datetime.now()
        for i in range(max_batches):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            rng, sub = jax.random.split(rng)
            with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                loss = eval_step_fn(state, batch, sub)
                loss.block_until_ready()
            losses.append(float(loss))
        if losses and jax.process_index() == 0:
            mean = sum(losses) / len(losses)
            elapsed = (datetime.datetime.now() - eval_start).total_seconds()
            max_logging.log(
                f"[ctrl_world] eval step={step} batches={len(losses)} "
                f"mean_loss={mean:.4f} elapsed={elapsed:.1f}s"
            )
            if self._wandb_run is not None:
                self._wandb_run.log(
                    {"eval/loss": mean, "eval/batches": len(losses)}, step=step
                )

    # ── W&B video rollout ──────────────────────────────────────────────────────

    def _build_video_pipeline(self, mesh):
        """Lazily build the inference pipeline used for W&B video rollouts.

        Training never touches the VAE (latents are pre-encoded on disk), so the
        decoder is loaded here and only here — costing ~2.7 GB of extra weights
        for the duration of the run. That is why video logging is opt-in.
        """
        from maxdiffusion.models.svd.video_autoencoder_flax import FlaxSVDAutoencoderKL
        from maxdiffusion.pipelines.svd.pipeline_flax_ctrl_world import (
            FlaxCtrlWorldPipeline,
        )
        from maxdiffusion.schedulers.scheduling_edm_euler_flax import (
            FlaxEDMEulerScheduler,
        )

        config = self.config
        max_logging.log(
            f"[ctrl_world] wandb video logging: loading VAE decoder from "
            f"{config.pretrained_model_name_or_path}/vae"
        )
        with mesh:
            vae, vae_params = FlaxSVDAutoencoderKL.from_pretrained(
                config.pretrained_model_name_or_path,
                subfolder="vae",
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
                from_pt=config.from_pt,
                use_safetensors=True,
            )
        sched = config.diffusion_scheduler_config
        unet, action_encoder, adaln_proj = self._video_modules
        self._video_pipeline = FlaxCtrlWorldPipeline(
            vae=vae,
            unet=unet,
            action_encoder=action_encoder,
            action_adaln_proj=adaln_proj,
            scheduler=FlaxEDMEulerScheduler(
                sigma_min=sched["sigma_min"],
                sigma_max=sched["sigma_max"],
                rho=sched["rho"],
                prediction_type=sched["prediction_type"],
                dtype=self.weights_dtype,
            ),
            image_encoder=None,
            feature_extractor=None,
            dtype=self.dtype,
        )
        # ``convert_pytorch_state_dict_to_flax`` pins every leaf to CPU device 0,
        # so decoding TPU-resident latents with these params raises
        # "Received incompatible devices for jitted computation". Replicate them
        # onto the mesh (the decoder is small and the activations it sees are
        # data-parallel, so P() is the right spec).
        repl = NamedSharding(mesh, P())
        self._video_vae_params = jax.tree_util.tree_map(
            lambda x: _place_on_sharding(x, repl), _unbox_tree(vae_params)
        )

    def _decode_views_for_log(self, latents):
        """``(B, T, 4, num_views*h, w)`` scaled latents → ``(B, num_views, T, H, W, 3)``.

        Cameras were encoded separately and stacked on the latent height axis, so
        they must be split before decoding — decoding the stack as one image would
        blend across camera seams.
        """
        pipeline = self._video_pipeline
        scale = pipeline.vae.config.scaling_factor
        num_views = int(self.config.num_views)
        b, t = latents.shape[:2]
        flat = latents.reshape((b * t,) + latents.shape[2:])   # (B*T, 4, H, W)
        out = []
        for view in jnp.split(flat, num_views, axis=-2):
            frames = pipeline.vae.apply(
                {"params": self._video_vae_params},
                view / scale,
                num_frames=view.shape[0],
                deterministic=True,
                method=pipeline.vae.decode,
            ).sample                                           # (B*T, 3, h, w)
            frames = (frames / 2.0 + 0.5).clip(0.0, 1.0)
            frames = jnp.transpose(frames, (0, 2, 3, 1))       # (B*T, h, w, 3)
            out.append(np.asarray(frames).reshape((b, t) + frames.shape[1:]))
        return np.stack(out, axis=1)                            # (B, V, T, h, w, 3)

    def _log_wandb_videos(self, state, mesh, step: int, rng):
        """Roll out one eval window, decode prediction vs GT, log to W&B.

        Single-chunk rollout: history and the conditioning frame come from the
        eval window exactly as in training, and the ``num_frames`` future slots
        are generated from noise. That keeps this directly comparable to the
        training objective (and one pipeline call) rather than compounding error
        over an auto-regressive sequence like the inference script does.

        The rollout and decode run on every host (they are collective); only
        process 0 writes to W&B.
        """
        config = self.config
        if not config.eval_data_dir:
            max_logging.log(
                "[ctrl_world] wandb_video_every>0 but eval_data_dir is empty; skipping"
            )
            return
        if self._video_pipeline is None:
            self._build_video_pipeline(mesh)
        if self._video_eval_iter is None:
            self._video_eval_iter = make_data_iterator(
                config, jax.process_index(), jax.process_count(), mesh,
                self._global_batch_size_to_load(), is_training=False,
                seed=config.seed,
            )
        try:
            batch = next(self._video_eval_iter)
        except StopIteration:
            self._video_eval_iter = None
            max_logging.log("[ctrl_world] eval split exhausted; skipping video log")
            return

        n = max(1, int(max_utils.config_get(config, "wandb_video_samples", 1)))
        n = min(n, batch["latent"].shape[0])
        t_hist = config.num_history
        latent = batch["latent"][:n].astype(self.weights_dtype)
        action = batch["action"][:n].astype(self.weights_dtype)
        text_embeds = (
            batch["text_embeds"][:n].astype(self.weights_dtype)
            if config.text_embed_dim else None
        )

        params = {
            "unet": state.params["unet"],
            "action_encoder": state.params["action_encoder"],
            "vae": self._video_vae_params,
        }
        if "action_adaln_proj" in state.params:
            params["action_adaln_proj"] = state.params["action_adaln_proj"]

        t_start = datetime.datetime.now()
        guidance = float(max_utils.config_get(config, "wandb_video_guidance_scale", 1.0))
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            pred_future = self._video_pipeline(
                params=params,
                prng_seed=rng,
                action=action,
                # Conditioning frame is the first future slot, matching
                # action_world_train_step's `latents[:, num_history]`.
                image_latent=latent[:, t_hist],
                history=latent[:, :t_hist],
                text_embeds=text_embeds,
                num_frames=config.num_frames,
                num_history=t_hist,
                num_inference_steps=int(
                    max_utils.config_get(config, "wandb_video_inference_steps", 25)
                ),
                min_guidance_scale=1.0,
                max_guidance_scale=guidance,
                fps_id=config.ctrl_fps_id,
                motion_bucket_id=config.ctrl_motion_bucket_id,
                cond_aug=config.ctrl_noise_aug_strength,
                frame_level_cond=True,
                his_cond_zero=config.ctrl_his_cond_zero,
                action_cond_mode=config.action_cond_mode,
                output_type="latent",
            )
            pred_future.block_until_ready()

        gt_future = latent[:, t_hist:]
        latent_mse = float(jnp.mean((pred_future.astype(jnp.float32)
                                     - gt_future.astype(jnp.float32)) ** 2))
        # Prepend the history so the clip shows the context it was conditioned on.
        with mesh:
            pred_np = self._decode_views_for_log(
                jnp.concatenate([latent[:, :t_hist], pred_future], axis=1)
            )
            gt_np = self._decode_views_for_log(latent)

        if jax.process_index() != 0 or self._wandb_run is None:
            return

        import wandb

        logs = {"eval/video_rollout_latent_mse": latent_mse}
        for i in range(pred_np.shape[0]):
            pred_grid = np.concatenate(list(pred_np[i]), axis=1)   # cams on H
            gt_grid = np.concatenate(list(gt_np[i]), axis=1)
            side_by_side = np.concatenate([gt_grid, pred_grid], axis=2)  # GT | pred on W
            frames = (side_by_side * 255).clip(0, 255).astype(np.uint8).transpose(0, 3, 1, 2)
            logs[f"eval/video/sample_{i}"] = wandb.Video(
                frames,
                fps=int(max_utils.config_get(config, "output_video_fps", 5)),
                format="mp4",
            )
        self._wandb_run.log(logs, step=step)
        elapsed = (datetime.datetime.now() - t_start).total_seconds()
        max_logging.log(
            f"[ctrl_world] logged {pred_np.shape[0]} rollout video(s) to W&B at "
            f"step {step} (latent_mse={latent_mse:.4f}, {elapsed:.1f}s)"
        )

    # ── Checkpoints ────────────────────────────────────────────────────────────

    def _build_checkpoint_manager(self, ckpt_dir: str) -> ocp.CheckpointManager:
        if not ckpt_dir.startswith("gs://"):
            os.makedirs(ckpt_dir, exist_ok=True)
        # "step" is the JSON sidecar holding *all* non-array resume metadata
        # (step counter, training RNG, restart counter). Kept under this name for
        # backwards compatibility with checkpoints written before those extra
        # fields existed — readers use .get() defaults.
        item_names = ("params", "step")
        item_handlers = {
            "params": ocp.StandardCheckpointHandler(),
            "step":   ocp.JsonCheckpointHandler(),
        }
        if self.config.save_optimizer:
            item_names = item_names + ("opt_state",)
            item_handlers["opt_state"] = ocp.StandardCheckpointHandler()
        options = ocp.CheckpointManagerOptions(
            create=True,
            max_to_keep=int(self.config.checkpoint_max_to_keep),
            enable_async_checkpointing=True,
        )
        return ocp.CheckpointManager(
            ckpt_dir,
            item_names=item_names,
            item_handlers=item_handlers,
            options=options,
        )

    def _save_checkpoint(self, mgr: ocp.CheckpointManager, step: int, state, rng):
        """Write a fully resumable checkpoint: params (+opt_state) + resume meta."""
        if step in set(mgr.all_steps()):
            max_logging.log(
                f"[ctrl_world] checkpoint for step {step} already exists; skipping save"
            )
            return
        # Never let a diverged state reach disk. Once params go non-finite they can
        # only be restored as non-finite, and a saved NaN turns one bad run into
        # every subsequent relaunch failing at "step 0" for no visible reason. The
        # older checkpoints kept by max_to_keep stay resumable this way.
        if _count_nonfinite(state.params) > 0:
            max_logging.log(
                f"[ctrl_world] REFUSING to save checkpoint at step {step}: params contain "
                "non-finite values. Training has diverged — the last good checkpoint in "
                "this directory is left intact so you can resume from it."
            )
            return
        if jax.process_index() == 0:
            max_logging.log(f"[ctrl_world] saving checkpoint at step {step}")
        meta = {
            "step": int(step),
            "rng": _rng_to_list(rng),
            "restart_count": int(self.restart_count),
            "data_seed": int(self.data_seed),
        }
        items = {
            "params": ocp.args.StandardSave(state.params),
            "step":   ocp.args.JsonSave(meta),
        }
        if self.config.save_optimizer:
            items["opt_state"] = ocp.args.StandardSave(state.opt_state)
        mgr.save(step, args=ocp.args.Composite(**items))

    def _checkpoint_items(self, mgr: ocp.CheckpointManager, step: int):
        """Item names present in an on-disk checkpoint, or None if unknown."""
        try:
            return set(mgr.item_metadata(step).keys())
        except Exception as e:  # older orbax / partial metadata — trust the config
            max_logging.log(f"[ctrl_world] could not read checkpoint item metadata: {e}")
            return None

    def _maybe_restore(self, mgr: ocp.CheckpointManager, state, state_shardings):
        """Returns (state, start_step, meta). state is unchanged if no ckpt exists."""
        del state_shardings  # StandardRestore reuses the input arrays' shardings
        latest = mgr.latest_step()
        if latest is None:
            max_logging.log("[ctrl_world] no checkpoint found; starting from step 0")
            return state, 0, {}
        max_logging.log(f"[ctrl_world] restoring checkpoint at step {latest}")
        available = self._checkpoint_items(mgr, latest)
        restore_args = {
            "params": ocp.args.StandardRestore(state.params),
            "step":   ocp.args.JsonRestore(),
        }
        want_opt = self.config.save_optimizer and (available is None or "opt_state" in available)
        if want_opt:
            restore_args["opt_state"] = ocp.args.StandardRestore(state.opt_state)
        elif self.config.save_optimizer:
            max_logging.log(
                "[ctrl_world] WARNING: checkpoint has no opt_state — Adam moments and the "
                "LR-schedule position restart from scratch."
            )
        restored = mgr.restore(latest, args=ocp.args.Composite(**restore_args))
        _assert_params_finite(
            restored["params"],
            f"checkpoint at step {latest} in {getattr(mgr, 'directory', '<ckpt dir>')}",
            hint=(
                "That checkpoint was written after the run had already diverged: "
                "opt_max_consecutive_nonfinite lets a non-finite update through once the "
                "skip budget runs out, which poisons params AND the Adam moments, and the "
                "next periodic save puts them on disk. Every relaunch then restores NaN "
                "and reports 'first non-finite at step 0', hiding the original cause. "
                "Start clean — bump run_name (checkpoint_dir is output_dir/run_name/"
                "checkpoints) or delete this directory — and rerun."
            ),
        )
        meta = dict(restored["step"])
        start_step = int(meta.get("step", latest))
        # Keep the step leaf's dtype identical to the freshly built state's so
        # jit's in_shardings/avals for the donated state argument still match.
        step_leaf = jnp.asarray(start_step, dtype=getattr(state.step, "dtype", jnp.int32))
        new_state = state.replace(params=restored["params"], step=step_leaf)
        if want_opt and restored.get("opt_state") is not None:
            new_state = new_state.replace(opt_state=restored["opt_state"])
        return new_state, start_step, meta

    # ── Misc ───────────────────────────────────────────────────────────────────

    def _data_seed(self, restart_count: int) -> int:
        """Seed for the input pipeline for this launch.

        With ``reshuffle_data_on_restart`` (default) the seed advances with the
        restart counter, so every resumed run draws a different file order,
        window shuffle, and skip/skip_his sequence instead of replaying the exact
        stream the previous launch already trained on. Set the flag to False for
        a bit-comparable rerun of the same data order.
        """
        base = int(self.config.seed)
        if not getattr(self.config, "reshuffle_data_on_restart", True):
            return base
        # Large odd stride so consecutive restarts land far apart in seed space.
        return (base + 1000003 * int(restart_count)) % (2**31 - 1)

    def _global_batch_size_to_load(self) -> int:
        if self.config.global_batch_size and self.config.global_batch_size > 0:
            return int(self.config.global_batch_size)
        per_device = self.config.per_device_batch_size
        gbs = max(1, int(jax.device_count() * per_device))
        if gbs % jax.process_count() != 0:
            gbs = (gbs // jax.process_count() + 1) * jax.process_count()
        return gbs
