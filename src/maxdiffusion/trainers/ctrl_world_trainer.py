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
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P
from safetensors.torch import load_file as load_torch_safetensors

from maxdiffusion import max_logging, max_utils
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.svd.action_encoder_flax import FlaxActionEncoder
from maxdiffusion.models.svd.ctrl_world_flax import (
    CtrlWorldTrainConfig,
    action_world_train_step,
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

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _build_mesh(self):
        devices = max_utils.create_device_mesh(self.config)
        return jax.sharding.Mesh(devices, self.config.mesh_axes)

    def _load_modules(self, mesh):
        max_logging.log(
            f"[ctrl_world] loading UNet from {self.config.pretrained_model_name_or_path}/unet"
        )
        with mesh:
            unet, unet_params = FlaxVideoUNet.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder="unet",
                dtype=self.dtype,
                weights_dtype=self.weights_dtype,
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
        unet_params = _rebox_like(abstract_unet_params, unet_params)

        action_encoder = FlaxActionEncoder(
            action_dim=self.config.action_dim,
            hidden_size=self.config.hidden_size,
            text_embed_dim=self.config.text_embed_dim,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
        )
        if self.config.action_encoder_init_path:
            max_logging.log(
                f"[ctrl_world] loading action encoder from {self.config.action_encoder_init_path}"
            )
            ae_params = _load_action_encoder_params(
                self.config.action_encoder_init_path, self.weights_dtype
            )
        else:
            max_logging.log("[ctrl_world] initialising action encoder from scratch")
            ae_params = action_encoder.init_weights(
                jax.random.PRNGKey(self.config.seed), batch=1, num_frames=1
            )

        return unet, unet_params, action_encoder, ae_params

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

    def _build_sharded_state(self, mesh, unet, unet_params, action_encoder, ae_params, tx):
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
        params = {"unet": unet_params, "action_encoder": ae_params}
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

    # ── NaN localisation ───────────────────────────────────────────────────────

    def _run_nan_probe(self, state, batch, rng, apply_fns, mesh, state_shardings,
                       data_shardings):
        """Run one forward+backward with per-stage finiteness checks and log where
        the first non-finite value appears.

        The periodic training log only says *that* loss went non-finite; this says
        *which stage*, which is the difference between a data/conditioning problem
        (an early stage), an overflow inside the UNet (19_unet_v_pred), and a
        preconditioning problem (07_sigma..12_loss_weight). Runs once, on the first
        batch, under the same mesh/shardings/dtypes as the real step so the numbers
        are the ones training actually sees.
        """
        cfg = self.train_cfg
        vae_scaling_factor = float(self.config.vae_scaling_factor)
        weights_dtype = self.weights_dtype
        names: dict[str, list[str]] = {"stages": [], "grads": []}

        def probe_step(params, batch, rng):
            batch = jax.tree_util.tree_map(
                lambda x: x.astype(weights_dtype) if x.dtype.kind == "f" else x, batch
            )
            def loss_fn(p):
                # probe_stack must run *inside* the differentiated function: the
                # recorded intermediates are tracers owned by this trace, and
                # touching them after value_and_grad returns is a tracer leak.
                probe: list = []
                loss = action_world_train_step(
                    rng=rng, params=p, apply_fns=apply_fns, batch=batch,
                    cfg=cfg, vae_scaling_factor=vae_scaling_factor, train=True,
                    probe=probe,
                )
                # Names are static python strings captured during tracing, so they
                # cannot be jit outputs — close over the dict instead.
                names["stages"] = [n for n, _ in probe]
                return loss, max_utils.probe_stack(probe)

            (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            # Gradients are summarised here, inside the jit; the full tree is ~1.5B
            # params and must never be pulled to the host.
            grad_names, grad_stats = max_utils.grad_probe(grads)
            names["grads"] = grad_names
            return loss, stats, grad_stats

        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            loss, stats, grad_stats = jax.jit(
                probe_step, in_shardings=(state_shardings.params, data_shardings, None),
                out_shardings=(None, None, None),
            )(state.params, batch, rng)

        max_logging.log(f"[nan-probe] loss={float(loss)}")
        first = max_utils.log_probe_report(names["stages"], stats, label="svd ctrl_world forward")
        if first is None:
            max_logging.log(
                "[nan-probe] forward is finite — the non-finite value is created in "
                "the backward pass; see the gradient breakdown below"
            )
        max_utils.log_probe_summary_table(
            names["grads"], grad_stats,
            label="svd ctrl_world gradients by parameter", top_n=25,
        )
        return first

    # ── Steps ──────────────────────────────────────────────────────────────────

    def _build_train_step(self, apply_fns, state_shardings, data_shardings):
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
        unet, unet_params, action_encoder, ae_params = self._load_modules(mesh)

        apply_fns = {"unet": unet.apply, "action_encoder": action_encoder.apply}
        tx, lr_schedule = self._build_optimizer(config.max_train_steps)

        state, state_shardings = self._build_sharded_state(
            mesh, unet, unet_params, action_encoder, ae_params, tx
        )
        del unet_params, ae_params  # freed inside state

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
                settings=wandb.Settings(start_method="thread"),
            )
        wandb_run = self._wandb_run

        recent_loss: list[float] = []
        recent_grad: list[float] = []
        last_step_time = datetime.datetime.now()

        probe_pending = bool(getattr(config, "debug_nan_probe", False))

        for step in range(start_step, config.max_train_steps):
            batch = next(train_iter)
            rng, step_rng = jax.random.split(rng)

            if probe_pending:
                probe_pending = False
                self._run_nan_probe(
                    state, batch, step_rng, apply_fns, mesh,
                    state_shardings, data_shardings,
                )

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
