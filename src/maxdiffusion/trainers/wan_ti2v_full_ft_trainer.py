"""WAN TI2V full-finetune overfit diagnostic -- objective, state, and steps.

This is the side-adapter trainer *minus the adapter* (plan_full_ft_overfit §2):
the Wan2.2 TI2V 5B transformer is fully trainable (all its params go in the
optimizer, no adapter exists) while the data path, noise schedule, one-step
flow-matching objective, and conditioning stay identical to the adapter runs.
The purpose is to measure how fast a fully-trainable backbone can memorize the
cached-DROID train split, isolating whether the pipeline is learnable at all
versus whether frozen-backbone + adapter optimization is the bottleneck.

Parity with the side-adapter run is enforced *by shared code*, not by
re-implementation: the noisy-latent construction and masked velocity MSE come
from ``side_adapter_wan`` (``build_noisy_pinned_latents`` / ``masked_velocity_mse``),
and the noise / step-index samplers are imported from the side-adapter trainer.
The only differences here are the experiment variable itself: the whole
transformer is the trainable ``params``, there is exactly one plain
``transformer(...)`` forward (no adapter, no actions), and the CFG branch is
bypassed (``guide_scale`` is asserted to 1.0 in the round-3 trainer class).

Round 2 provided ``FullFTTrainState``, ``_denoising_loss``, and
``_train_step`` / ``_eval_step``. Round 3 adds ``WanTI2VFullFTTrainer`` --
subclassing the side-adapter trainer, reusing its scheduler / pipeline-load /
null-context / dataset / optimizer / checkpoint machinery -- plus the probe-config
asserts (``guide_scale == 1.0`` §2.1 and ``noise_mode == "fresh"`` F1), the
param/moment dtype + byte-total startup logging, the shard-selection override that
KEEPS the computed FSDP shardings for params and opt_state (replicating the ~5B
transformer would OOM), and the large-leaf sharding audit (F9). ``train_wan.py``
dispatches ``FULL_FT_TI2V`` here.
"""

from __future__ import annotations

import datetime
import functools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import jaxopt
import optax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state

from maxdiffusion import max_logging
from maxdiffusion.models.wan.side_adapter_wan import (
    adapter_param_count,
    build_noisy_pinned_latents,
    build_rollout_sigmas,
    masked_velocity_mse,
    _build_per_token_timestep,
    _dtype,
)
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import load_next_batch
from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import (
    WanTI2VSideAdapterTrainer,
    _apply_actual_sharding_for_tpu,
    _build_noise,
    _sample_step_indices,
    _to_target_if_cpu,
)


class FullFTTrainState(train_state.TrainState):
    """Train state whose trainable ``params`` are the full WAN transformer.

    Unlike the side-adapter ``TrainState`` (which carries a frozen transformer
    alongside trainable adapter params), here the transformer *is* the trainable
    module: ``params`` holds its ``nnx.Param`` leaves and gets optimizer state,
    ``graphdef`` / ``rest_of_state`` are its static graph and non-Param state, and
    ``null_context`` is the reused null text embedding.
    """

    graphdef: nnx.GraphDef
    rest_of_state: nnx.State
    null_context: jax.Array


def _denoising_loss(
    params,
    state: FullFTTrainState,
    data: dict,
    rng: jax.Array,
    config,
    scheduler: FlaxFlowMatchScheduler,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """One-step flow-matching velocity MSE for the fully-trainable transformer.

    Mirrors ``wan_ti2v_side_adapter_trainer._denoising_loss`` but with no adapter,
    no actions, and no CFG branch: a single plain ``transformer(...)`` forward on
    the frame-0-pinned noisy latents against the null text context. The
    interpolation/pin math stays float32; only the transformer input is cast to
    ``weights_dtype``.
    """
    noise_rng, step_rng, dropout_rng = jax.random.split(rng, 3)
    weights_dtype = _dtype(config.weights_dtype)
    activations_dtype = _dtype(config.activations_dtype)
    bsz = config.global_batch_size_to_train_on

    transformer = nnx.merge(state.graphdef, params, state.rest_of_state)

    z_i0_f32 = data["z_i0"][:bsz].astype(jnp.float32)
    z_video_f32 = data["z_video"][:bsz].astype(jnp.float32)

    b, _, f_lat, h_lat, w_lat = z_video_f32.shape
    num_steps = int(config.side_adapter_sampling_steps)
    sigmas = build_rollout_sigmas(
        num_steps,
        config.flow_shift,
        scheduler.config.sigma_min,
        scheduler.config.sigma_max,
    )
    t_idx = _sample_step_indices(step_rng, b, num_steps, sigmas, config)
    sigma_t = sigmas[t_idx]
    step_t = sigma_t * jnp.asarray(scheduler.config.num_train_timesteps, dtype=jnp.float32)
    timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
    null_context = jnp.broadcast_to(
        state.null_context.astype(activations_dtype),
        (b, state.null_context.shape[1], state.null_context.shape[2]),
    )

    eps = _build_noise(noise_rng, z_video_f32.shape, jnp.float32, config)
    z_t_f32 = build_noisy_pinned_latents(z_video_f32, z_i0_f32, eps, sigma_t)

    v_pred = transformer(
        hidden_states=z_t_f32.astype(weights_dtype),
        timestep=timestep_2d,
        encoder_hidden_states=null_context,
        deterministic=False,
        rngs=nnx.Rngs(dropout=dropout_rng),
    )

    v_target = eps - z_video_f32
    loss = masked_velocity_mse(v_pred, v_target, b)
    aux = {
        "velocity_mse": loss,
        "sigma_mean": jnp.mean(sigma_t.astype(jnp.float32)),
        "timestep_mean": jnp.mean(step_t.astype(jnp.float32)),
        "v_pred_l2": jnp.linalg.norm(v_pred.astype(jnp.float32)),
        "v_target_l2": jnp.linalg.norm(v_target.astype(jnp.float32)),
        "z_noisy_std": jnp.std(z_t_f32),
        "z_target_std": jnp.std(z_video_f32),
        "z_init_anchor_mse": jnp.mean((z_t_f32[:, :, :1] - z_i0_f32[:, :, :1]) ** 2),
    }
    return loss, aux


def _train_step(state: FullFTTrainState, data: dict, rng: jax.Array, scheduler, config):
    rng, loss_rng = jax.random.split(rng)

    def loss_fn(params):
        return _denoising_loss(params, state, data, loss_rng, config, scheduler)

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
            "learning/velocity_mse": aux["velocity_mse"],
            "learning/grad_norm": grad_norm,
            "learning/max_abs_grad": max_abs_grad,
            "learning/sigma_mean": aux["sigma_mean"],
            "learning/timestep_mean": aux["timestep_mean"],
            "learning/v_pred_l2": aux["v_pred_l2"],
            "learning/v_target_l2": aux["v_target_l2"],
            "learning/z_noisy_std": aux["z_noisy_std"],
            "learning/z_target_std": aux["z_target_std"],
            "learning/z_init_anchor_mse": aux["z_init_anchor_mse"],
        },
        "scalars": {},
    }
    return state, metrics, rng


def _eval_step(state: FullFTTrainState, data: dict, rng: jax.Array, scheduler, config):
    losses = jnp.zeros((config.global_batch_size_to_train_on,), dtype=jnp.float32)
    loss, aux = _denoising_loss(state.params, state, data, rng, config, scheduler)
    losses = losses.at[:].set(loss)
    metrics = {
        "scalar": {
            "learning/eval_loss": losses,
            "learning/eval_sigma_mean": aux["sigma_mean"],
            "learning/eval_v_pred_l2": aux["v_pred_l2"],
            "learning/eval_v_target_l2": aux["v_target_l2"],
            "learning/eval_z_noisy_std": aux["z_noisy_std"],
            "learning/eval_z_target_std": aux["z_target_std"],
        },
        "scalars": {},
    }
    return metrics, rng


# --------------------------------------------------------------------------------------
# Round 3: shard-selection, large-leaf sharding audit, and dtype introspection helpers.
# All pure so the CPU wiring tests exercise them with fake trees (no weights/mesh).
# --------------------------------------------------------------------------------------


def _full_ft_state_shardings(computed, state):
    """Full-FT KEEPS the computed FSDP shardings for ``params`` and ``opt_state``.

    The side-adapter parent replaces both with a fully-replicated spec (its adapters
    are tiny). Replicating the ~5B transformer params + their Adam moments would OOM
    (plan §3), so the full-FT trainer keeps exactly what ``nnx.get_named_sharding`` +
    ``_apply_actual_sharding_for_tpu`` computed. Pure over (computed, state); the
    explicit ``.replace`` mirrors the parent's line-for-line so the deviation is just
    the RHS (keep computed vs. replicate).
    """
    del state  # signature parity with the parent's _shard_state(mesh, state)
    return computed.replace(params=computed.params, opt_state=computed.opt_state)


def _is_sharding_leaf(x) -> bool:
    return isinstance(x, (jax.sharding.NamedSharding, jax.sharding.PartitionSpec))


def _is_fully_replicated(spec) -> bool:
    """True iff ``spec`` places nothing on any mesh axis (a fully-replicated leaf)."""
    if isinstance(spec, jax.sharding.NamedSharding):
        spec = spec.spec
    if spec is None:
        return True

    def _axis_replicated(axis) -> bool:
        if axis is None:
            return True
        if isinstance(axis, (tuple, list)):
            return all(a is None for a in axis)
        return False

    return all(_axis_replicated(axis) for axis in spec)


def _leaf_bytes(leaf) -> tuple[int, int]:
    """(global logical bytes, per-host PHYSICAL bytes) for an array-like leaf.

    Physical bytes sum over ALL addressable shards with NO shard-index dedupe (F2):
    replicated per-device copies each occupy their own HBM, so a leaf replicated
    across two local devices costs twice its logical size on this host. Leaves
    without shards (host numpy / ShapeDtypeStruct) fall back to logical bytes.
    """
    itemsize = jnp.dtype(leaf.dtype).itemsize
    global_bytes = int(math.prod(leaf.shape)) * itemsize
    shards = getattr(leaf, "addressable_shards", None)
    if not shards:
        return global_bytes, global_bytes
    return global_bytes, sum(int(math.prod(s.data.shape)) * itemsize for s in shards)


def _leaf_entries(tree, spec_tree) -> list[dict]:
    """Flatten ``tree`` into per-leaf audit records, zipping in the parallel specs."""
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    if spec_tree is None:
        specs: list = [None] * len(leaves)
    else:
        specs = jax.tree_util.tree_leaves(spec_tree, is_leaf=_is_sharding_leaf)
    entries = []
    for (path, leaf), spec in zip(leaves, specs):
        if not hasattr(leaf, "shape"):
            continue
        global_bytes, addressable_bytes = _leaf_bytes(leaf)
        entries.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": tuple(int(d) for d in leaf.shape),
                "dtype": jnp.dtype(leaf.dtype).name,
                "bytes": global_bytes,
                "addressable_bytes": addressable_bytes,
                "spec": spec,
            }
        )
    return entries


def _adam_moment_trees(opt_state) -> tuple:
    """Locate the optax ``ScaleByAdamState`` in ``opt_state`` and return its (mu, nu).

    Works on the value tree AND on a structurally-mirrored sharding-spec tree (a
    ``jax.tree.map`` over the state preserves the ScaleByAdamState namedtuple nodes),
    which is what the path-matched twin audit needs (F3). (None, None) when absent.
    """
    for node in jax.tree_util.tree_leaves(opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState)):
        if isinstance(node, optax.ScaleByAdamState):
            return node.mu, node.nu
    return None, None


def _leaves_by_path(tree, is_leaf=None) -> dict:
    if tree is None:
        return {}
    return {jax.tree_util.keystr(p): leaf for p, leaf in jax.tree_util.tree_leaves_with_path(tree, is_leaf=is_leaf)}


def audit_large_leaves(params_tree, opt_tree, specs_tree, mesh_size, threshold_bytes=100 * 2**20, top_k=8):
    """Report the largest param leaves + their Adam twins; assert none big-and-replicated (F9).

    Pure over the trees, their resolved shardings (``specs_tree`` is a dict with
    ``"params"`` / ``"opt_state"`` sub-trees), and ``mesh_size``. Returns the ``top_k``
    largest param leaves plus the PATH-MATCHED Adam ``mu``/``nu`` twins of exactly
    those params (located inside the real ``ScaleByAdamState`` and labeled with the
    param path + ``moment`` -- F3: NOT the globally-largest opt leaves) as dicts
    (path, [moment,] shape, dtype, global bytes, addressable bytes, spec). RAISES if
    ANY param or opt leaf (top-k or not) larger than ``threshold_bytes`` resolves to
    a fully-replicated spec on a multi-device mesh (``mesh_size > 1``) -- replicating
    a 5B-scale leaf across FSDP would OOM (plan §3).
    """
    param_entries = _leaf_entries(params_tree, specs_tree.get("params"))
    opt_entries = _leaf_entries(opt_tree, specs_tree.get("opt_state"))

    if mesh_size > 1:
        offenders = [
            e
            for e in (param_entries + opt_entries)
            if e["bytes"] > threshold_bytes and _is_fully_replicated(e["spec"])
        ]
        if offenders:
            worst = max(offenders, key=lambda e: e["bytes"])
            raise ValueError(
                f"full-FT sharding audit: {len(offenders)} leaf(es) > {threshold_bytes / 2**20:.0f} MB are fully "
                f"replicated on a {mesh_size}-device mesh (would OOM, plan §3 / F9). Largest: {worst['path']} "
                f"shape={worst['shape']} {worst['bytes'] / 2**20:.0f} MB spec={worst['spec']}"
            )

    top_params = sorted(param_entries, key=lambda e: e["bytes"], reverse=True)[:top_k]

    # F3: the twins are the mu/nu OF the selected params, path-matched inside the
    # located ScaleByAdamState (magnitude-sorting opt leaves would typically return
    # both moments of only half as many params, or unrelated transform state).
    mu_tree, nu_tree = _adam_moment_trees(opt_tree)
    mu_specs, nu_specs = _adam_moment_trees(specs_tree.get("opt_state"))
    moment_leaves = {"mu": _leaves_by_path(mu_tree), "nu": _leaves_by_path(nu_tree)}
    moment_specs = {
        "mu": _leaves_by_path(mu_specs, is_leaf=_is_sharding_leaf),
        "nu": _leaves_by_path(nu_specs, is_leaf=_is_sharding_leaf),
    }
    opt_twins = []
    for e in top_params:
        for moment in ("mu", "nu"):
            leaf = moment_leaves[moment].get(e["path"])
            if leaf is None or not hasattr(leaf, "shape"):
                continue
            global_bytes, addressable_bytes = _leaf_bytes(leaf)
            opt_twins.append(
                {
                    "path": e["path"],
                    "moment": moment,
                    "shape": tuple(int(d) for d in leaf.shape),
                    "dtype": jnp.dtype(leaf.dtype).name,
                    "bytes": global_bytes,
                    "addressable_bytes": addressable_bytes,
                    "spec": moment_specs[moment].get(e["path"]),
                }
            )

    return {"params": top_params, "opt_twins": opt_twins}


def _dtype_summary(tree) -> dict[str, tuple[int, int]]:
    """Per-dtype ``{name: (leaf_count, total_bytes)}`` over ``tree``, sorted by name.

    One sampled leaf is NOT representative (F1): the WAN loader deliberately keeps
    norm / condition_embedder / scale_shift_table params float32 while casting the
    bulk to ``weights_dtype`` (``wan_pipeline.cast_with_exclusion``), so the real 5B
    tree is dtype-MIXED and the fp32-state control (plan §2.4) is distinguishable
    from the primary run only by the full per-dtype breakdown.
    """
    counts: dict[str, list[int]] = {}
    for leaf in jax.tree_util.tree_leaves(tree):
        if not (hasattr(leaf, "dtype") and hasattr(leaf, "shape")):
            continue
        name = jnp.dtype(leaf.dtype).name
        entry = counts.setdefault(name, [0, 0])
        entry[0] += 1
        entry[1] += int(math.prod(leaf.shape)) * jnp.dtype(leaf.dtype).itemsize
    return {name: (n, b) for name, (n, b) in sorted(counts.items())}


def _format_dtype_summary(summary: dict[str, tuple[int, int]]) -> str:
    if not summary:
        return "<empty>"
    return ", ".join(f"{name}(leaves={n}, bytes={b})" for name, (n, b) in summary.items())


def _tree_byte_totals(tree) -> tuple[int, int]:
    global_bytes = addressable_bytes = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if not hasattr(leaf, "shape"):
            continue
        g, a = _leaf_bytes(leaf)
        global_bytes += g
        addressable_bytes += a
    return global_bytes, addressable_bytes


class WanTI2VFullFTTrainer(WanTI2VSideAdapterTrainer):
    """Full-finetune overfit probe: the side-adapter trainer minus the adapter.

    Inherits the scheduler, pipeline load, null-context, dataset loader, optimizer
    factory, checkpoint manager/restore, data shardings, and eval loop from
    ``WanTI2VSideAdapterTrainer``. Here the transformer itself is the trainable
    module: ``start_training`` builds a ``FullFTTrainState`` from the transformer
    split (no adapter) and ``_shard_state`` keeps the computed FSDP shardings for
    params and opt_state instead of replicating them.
    """

    @staticmethod
    def _validate_probe_config(config) -> None:
        """Enforce the two probe invariants before any weights load (CPU-testable)."""
        guide_scale = float(config.side_adapter_guide_scale)
        # F5: exact comparison -- 1.0 is exactly float-representable and the plan's
        # invariant is hard equality; non-finite values are rejected explicitly rather
        # than slipping through a tolerance check's NaN comparison semantics.
        if not math.isfinite(guide_scale) or guide_scale != 1.0:
            raise ValueError(
                "full-FT overfit probe requires side_adapter_guide_scale == 1.0 exactly (CFG bypassed, plan §2.1); "
                f"got {guide_scale}. With no adapter both CFG branches are the same trainable network, so "
                "guide_scale != 1 is a silent pre-optimizer gradient multiplier plus a wasted forward pass."
            )
        noise_mode = getattr(config, "side_adapter_noise_mode", "fresh")
        if noise_mode != "fresh":
            raise ValueError(
                "full-FT overfit probe requires side_adapter_noise_mode == 'fresh' (F1 BLOCKER); "
                f"got {noise_mode!r}. 'fixed' noise creates a train/validation mismatch that looks like "
                "good train loss but produces broken generations."
            )

    def _shard_state(self, mesh, state: FullFTTrainState) -> tuple[FullFTTrainState, Any]:
        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            state_shardings = nnx.get_named_sharding(state, mesh)
            state_shardings = _apply_actual_sharding_for_tpu(state, state_shardings)
            # full-FT: KEEP the computed FSDP shardings for params AND opt_state.
            # The parent replicates both (adapters are tiny); replicating the ~5B
            # transformer params + Adam moments would OOM (plan §3).
            state_shardings = _full_ft_state_shardings(state_shardings, state)
            self._log_and_audit_shardings(state, state_shardings, mesh)
            state = _to_target_if_cpu(state, state_shardings)
            state = jax.device_put(state, state_shardings)
        return state, state_shardings

    def _log_and_audit_shardings(self, state, state_shardings, mesh) -> None:
        # The audit RAISES on any >100 MB leaf that resolves to a replicated spec on a
        # multi-device mesh (would OOM, plan §3 / F9); it runs on every host.
        audit = audit_large_leaves(
            state.params,
            state.opt_state,
            {"params": state_shardings.params, "opt_state": state_shardings.opt_state},
            mesh.size,
        )
        if jax.process_index() != 0:
            return
        for name, tree in (("params", state.params), ("opt_state", state.opt_state)):
            g, a = _tree_byte_totals(tree)
            # F2: "physical" = sum over ALL addressable shards (replicas each counted).
            max_logging.log(
                f"[wan_full_ft] {name} bytes: global={g / 2**30:.2f} GB per-host physical={a / 2**30:.2f} GB"
            )
        for e in audit["params"]:
            max_logging.log(
                f"[wan_full_ft]   param {e['path']} {e['shape']} {e['dtype']} "
                f"global={e['bytes'] / 2**20:.1f}MB addr={e['addressable_bytes'] / 2**20:.1f}MB spec={e['spec']}"
            )
        for e in audit["opt_twins"]:
            # F3: twins are labeled by their param's path + which Adam moment they are.
            max_logging.log(
                f"[wan_full_ft]   {e['moment']:5s} {e['path']} {e['shape']} {e['dtype']} "
                f"global={e['bytes'] / 2**20:.1f}MB addr={e['addressable_bytes'] / 2**20:.1f}MB spec={e['spec']}"
            )

    def start_training(self):
        config = self.config
        # full-FT: enforce guide_scale == 1.0 (§2.1) and fresh noise (F1) before any load.
        self._validate_probe_config(config)
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

        # full-FT: the transformer IS the trainable module -- no adapter is built.
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            transformer_graphdef, transformer_params, transformer_rest = nnx.split(pipeline.transformer, nnx.Param, ...)

        if jax.process_index() == 0:
            n_trainable = adapter_param_count(transformer_params)
            max_logging.log(f"[wan_full_ft] trainable transformer params: {n_trainable / 1e9:.2f}B")

        tx, lr_schedule = self._build_optimizer(config.max_train_steps)
        state = FullFTTrainState.create(
            apply_fn=transformer_graphdef.apply,
            params=transformer_params,
            tx=tx,
            graphdef=transformer_graphdef,
            rest_of_state=transformer_rest,
            null_context=null_context,
        )

        if jax.process_index() == 0:
            # full-FT: fp32-optimizer-state control precondition (plan §2.4). Per-dtype
            # summaries, not one sampled leaf: the loader keeps norm/condition_embedder
            # tables float32 beside the weights_dtype bulk, so the tree is MIXED (F1).
            mu_tree, nu_tree = _adam_moment_trees(state.opt_state)
            max_logging.log(f"[wan_full_ft] param dtypes: {_format_dtype_summary(_dtype_summary(state.params))}")
            max_logging.log(f"[wan_full_ft] adam mu dtypes: {_format_dtype_summary(_dtype_summary(mu_tree))}")
            max_logging.log(f"[wan_full_ft] adam nu dtypes: {_format_dtype_summary(_dtype_summary(nu_tree))}")
            max_logging.log(f"[wan_full_ft] activations dtype: {config.activations_dtype}")

        state, state_shardings = self._shard_state(mesh, state)
        data_shardings = self._data_shardings(mesh)

        scheduler, _ = self._create_scheduler()
        ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, "checkpoints")
        ckpt_mgr = self._build_checkpoint_manager(ckpt_dir)
        state, start_step = self._maybe_restore(ckpt_mgr, state)
        if start_step:
            max_logging.log(f"[wan_full_ft] resumed at step {start_step}")

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
            max_logging.log("***** Running WAN TI2V full-finetune overfit probe *****")
            max_logging.log(f"  Per-device batch size: {config.per_device_batch_size}")
            max_logging.log(f"  Devices: {jax.device_count()}")
            max_logging.log(f"  Max train steps: {config.max_train_steps}")
            max_logging.log(f"  Output dir: {config.output_dir}")
            # full-FT: resolved data dirs on record (F7 -- eval must read the train split).
            max_logging.log(f"  Train data dir: {config.train_data_dir}")
            max_logging.log(f"  Eval data dir: {config.eval_data_dir}")
            max_logging.log(f"  Denoising sigma steps: {config.side_adapter_sampling_steps}")
            max_logging.log(f"  Timestep sampling: {getattr(config, 'side_adapter_t_sampling', 'uniform')}")
            max_logging.log(f"  Noise mode: {getattr(config, 'side_adapter_noise_mode', 'fresh')}")
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

        with ThreadPoolExecutor(max_workers=1) as executor:
            for step in range(start_step, config.max_train_steps):
                next_batch_future = executor.submit(load_next_batch, train_iter, batch, config)
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

                batch = next_batch_future.result()

        if config.save_final_checkpoint:
            self._save_checkpoint(ckpt_mgr, config.max_train_steps, state)
        ckpt_mgr.wait_until_finished()
        if wandb_run is not None:
            wandb_run.finish()
