"""Visual validation for WAN TI2V side-adapter checkpoints.

This is the MaxDiffusion equivalent of ``../Wan2.2/eval_adaptor.py`` for the
cached DROID TFRecords.  The converted records contain cached latents and
actions, but not raw source frames, so validation compares generated videos to
the VAE decode of ``z_video`` and logs latent/pixel metrics.

**exp_02 OVERFIT100_TI2V branch** (cycle D; plan
``docs/worklogs_yixun/exp_02_overfit100_claude/plan_overfit100.md`` D11 / §4). The exp_01
full-FT restore path, sampler, metrics and video conventions are reused verbatim; what is new
is the *what* and the *conditioning*: schema-v2 records (no ``actions``), window SELECTION
(canonical median window per episode, or an explicit name list), three context modes
(``correct`` / ``null`` / ``shuffled``), several rollout seeds, and a machine-written
**aggregation artifact** that the pure success statistic
(``maxdiffusion.overfit100_success_statistic``) consumes -- no hand computation anywhere.

Aggregation artifact schema (``<step_root>/aggregation.json``, tag
``overfit100_eval_aggregation_v1``)::

    {
      "schema": "overfit100_eval_aggregation_v1",
      "checkpoint_step": int,            # the restored step this pass evaluated
      "run_name", "model_type", "commit",
      "eval_data_dir", "train_data_dir", "model_manifest_path",
      "eval_windows_spec": str,         # the --eval-windows spec as given
      "rollout_seeds": [int, ...],      # the seed list, in the order rolled out
      "context_modes": ["correct", ...],
      "context_shuffle_seed": int,
      "context_derangement": [int, ...],  # sigma over context-table rows (shuffled mode)
      "sampling_steps": int, "guide_scale": float,
      "num_windows": int,
      "windows": [{name, episode_id, episode_index, window_start, canonical, used_text}, ...],
      "canonical_windows": [[episode_id, window_start], ...],   # the FIXED denominator
      "flagged_windows": [[episode_id, window_start], ...],     # collisions: recorded, never dropped
      "rows": [ per-(window, seed, mode) row, keys == OVERFIT100_ROW_FIELDS ]
    }

Each row carries the window identity (``name`` / ``episode_id`` / ``episode_index`` /
``window_start`` / ``canonical``), the coordinates (``checkpoint_step`` / ``seed`` /
``context_mode`` / ``context_source_episode_index``), the PRIMARY metrics against the VAE
decode of the stored ``z_video`` (``ssim`` / ``latent_mse`` / ``pixel_mse``), and the
AUXILIARY metrics against the true DROID RGB frames paired with that window's VAE ceiling
(``ssim_vs_rgb`` / ``pixel_mse_vs_rgb`` / ``vae_ceiling_ssim`` / ``aux_status``). The auxiliary
block is ``None`` with a status string whenever the source MP4 cannot be pulled -- it never
fails the run.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
from absl import app
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from maxdiffusion import max_logging, pyconfig
from maxdiffusion.models.wan.side_adapter_wan import (
    _build_per_token_timestep,
    _dtype,
    apply_first_frame_pin,
    build_rollout_sigmas,
    rollout_timesteps_from_sigmas,
    wan_action_adapter_forward,
)

# Cycle-D strengthening (D1/D2): the cohort math, the artifact schema tag, the pass roles and the
# role contract live in the PURE statistic module -- ONE definition shared by the evaluator that
# writes the artifacts and the verdict that consumes them, so they can never disagree about what
# the cohort or a role is (parity pinned by test_overfit100_eval_contracts.py).
from maxdiffusion.overfit100_success_statistic import (
    AGGREGATION_SCHEMA as OVERFIT100_AGGREGATION_SCHEMA,
    PASS_ROLES as OVERFIT100_PASS_ROLES,
    all_window_keys_from_manifest,
    canonical_cohort_from_manifest,
    canonical_window_start,
    pass_role_plan_reasons,
    role_requirements,
)
from maxdiffusion.trainers.wan_ti2v_full_ft_trainer import (
    FullFTTrainState,
    WanTI2VFullFTTrainer,
)
from maxdiffusion.trainers.wan_ti2v_overfit100_trainer import (
    Overfit100TrainState,
    WanTI2VOverfit100Trainer,
    _read_json_strict,
    read_episode_mapping,
    read_episode_texts,
    read_manifest_episodes,
)
from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import (
    TrainState,
    WanTI2VSideAdapterTrainer,
)
from maxdiffusion.utils import export_to_video

jax.config.update("jax_use_shardy_partitioner", True)


@dataclass
class EvalSample:
    name: str
    ordinal: int
    z_i0: np.ndarray
    z_video: np.ndarray
    actions: np.ndarray
    meta: dict


def _tfrecord_files(data_dir: str) -> list[str]:
    files = sorted(tf.io.gfile.glob(data_dir.rstrip("/") + "/*.tfrecord"))
    if not files:
        files = sorted(tf.io.gfile.glob(data_dir.rstrip("/") + "/*.tfrecord-*"))
    if not files:
        raise FileNotFoundError(f"No TFRecord shards found under {data_dir}")
    return files


def _parse_ordinals(text) -> list[int]:
    """Parse the comma-separated ``validation_ordinals`` config value into positions.

    Empty / whitespace-only -> ``[]`` (the reader then uses the contiguous
    ``validation_start_index`` read). Non-integer or negative tokens are rejected with
    an actionable error.
    """
    text = (text or "").strip()
    if not text:
        return []
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"validation_ordinals must be comma-separated integers; got token {token!r}") from exc
        if value < 0:
            raise ValueError(f"validation_ordinals entries must be non-negative dataset positions; got {value}")
        out.append(value)
    return out


def _iter_parsed_records(config):
    """Yield parsed (numpy-decoded) records from the eval TFRecord shards, in order.

    The single TFRecord iteration seam: tests monkeypatch this to feed a fake record
    sequence without constructing any tf.data graph.
    """
    feature_description = {
        "name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "ordinal": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "z_i0": tf.io.FixedLenFeature([], tf.string),
        "z_video": tf.io.FixedLenFeature([], tf.string),
        "actions": tf.io.FixedLenFeature([], tf.string),
        "meta_json": tf.io.FixedLenFeature([], tf.string, default_value=b"{}"),
    }
    ds = tf.data.TFRecordDataset(_tfrecord_files(config.eval_data_dir))
    ds = ds.map(lambda raw: tf.io.parse_single_example(raw, feature_description))
    return ds.as_numpy_iterator()


def _select_eval_records(record_iter, *, ordinals, start_index, count, source=""):
    """Select ``(position, raw)`` pairs from a forward iterator of parsed records.

    CONTRACT (plan §2.3 cohort protocol):
      * ``ordinals`` non-empty: the records at exactly those 0-based dataset POSITIONS
        (the same coordinate as ``validation_start_index`` -- NOT the stored ``ordinal``
        field), returned IN THE LISTED ORDER (duplicates preserved), so the per-sample
        rollout seed that ``run()`` derives in iteration order is user-controlled. The
        iterator is consumed only up to the largest requested position (early stop). A
        position past the end raises an actionable error naming the offending ordinal(s)
        and the number of records seen.
      * ``ordinals`` empty: the contiguous ``skip(start_index).take(count)`` read --
        byte-identical to the pre-cohort behavior -- also early-stopping at
        ``start_index + count``.
    """
    if ordinals:
        wanted = set(ordinals)
        max_ordinal = max(ordinals)
        found: dict[int, object] = {}
        seen = 0
        for position, raw in enumerate(record_iter):
            seen = position + 1
            if position in wanted:
                found[position] = raw
            if position >= max_ordinal:
                break
        missing = [o for o in ordinals if o not in found]
        if missing:
            where = f" in {source}" if source else ""
            raise ValueError(
                f"validation_ordinals out of range{where}: {missing} not found; the dataset yielded only "
                f"{seen} records, so every ordinal must be in [0, {seen})."
            )
        return [(o, found[o]) for o in ordinals]

    selected: list[tuple[int, object]] = []
    stop = start_index + count
    for position, raw in enumerate(record_iter):
        if position < start_index:
            continue
        if position >= stop:
            break
        selected.append((position, raw))
    return selected


def _read_eval_samples(config, count: int, start_index: int) -> list[EvalSample]:
    c = int(config.latent_channels)
    f = int(config.latent_frames)
    h = int(config.latent_height)
    w = int(config.latent_width)
    action_len = int(config.action_len)
    action_dim = int(config.action_dim)

    ordinals = _parse_ordinals(getattr(config, "validation_ordinals", ""))
    selected = _select_eval_records(
        _iter_parsed_records(config),
        ordinals=ordinals,
        start_index=max(0, int(start_index)),
        count=max(1, int(count)),
        source=config.eval_data_dir,
    )

    samples: list[EvalSample] = []
    for position, raw in selected:
        name = raw["name"].decode("utf-8") or f"sample_{position:06d}"
        meta_bytes = raw["meta_json"] or b"{}"
        try:
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception:
            meta = {"raw_meta_json": meta_bytes.decode("utf-8", errors="replace")}
        samples.append(
            EvalSample(
                name=name,
                ordinal=int(raw["ordinal"]),
                z_i0=np.frombuffer(raw["z_i0"], dtype=np.float16).reshape(c, 1, h, w).astype(np.float32),
                z_video=np.frombuffer(raw["z_video"], dtype=np.float16).reshape(c, f, h, w).astype(np.float32),
                actions=np.frombuffer(raw["actions"], dtype=np.float32).reshape(action_len, action_dim),
                meta=meta,
            )
        )
    if not samples:
        raise ValueError(f"No eval samples read from {config.eval_data_dir}")
    return samples


def _rollout_sample(state, data: dict, rng: jax.Array, scheduler, config):
    weights_dtype = _dtype(config.weights_dtype)
    is_full_ft = getattr(config, "model_type", "") == "FULL_FT_TI2V"
    if is_full_ft:
        # full-FT: the transformer IS the trainable module (``params``/``rest_of_state``
        # are its own); no adapter is merged and the body issues one plain forward.
        transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
        adapters = None
    else:
        adapters = nnx.merge(state.graphdef, state.params, state.rest_of_state)
        transformer = nnx.merge(state.transformer_graphdef, state.transformer_params, state.transformer_rest)

    z_i0 = data["z_i0"].astype(weights_dtype)
    z_video = data["z_video"].astype(weights_dtype)
    actions = data["actions"].astype(weights_dtype)
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

    z = jax.random.normal(rng, z_video.shape, dtype=z_video.dtype)
    z = apply_first_frame_pin(z, z_i0)

    def _body(i, current):
        step_t = jnp.broadcast_to(timesteps[i], (b,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
        if is_full_ft:
            # one plain transformer forward: no adapter, no actions, no CFG (plan §3).
            # Structurally the same call as the adapter path's frozen v_uncond branch
            # below, but here it is the trainable prediction itself.
            v = transformer(
                hidden_states=current,
                timestep=timestep_2d,
                encoder_hidden_states=null_context,
                deterministic=True,
            )
        else:
            v_cond = wan_action_adapter_forward(
                transformer,
                adapters,
                hidden_states=current,
                timestep=timestep_2d,
                encoder_hidden_states=null_context,
                actions=actions,
                deterministic=True,
            )
            if abs(config.side_adapter_guide_scale - 1.0) > 1e-6:
                v_uncond = transformer(
                    hidden_states=current,
                    timestep=timestep_2d,
                    encoder_hidden_states=null_context,
                    deterministic=True,
                )
                v = v_uncond + config.side_adapter_guide_scale * (v_cond - v_uncond)
            else:
                v = v_cond
        return apply_first_frame_pin(current + (sigmas[i + 1] - sigmas[i]).astype(current.dtype) * v, z_i0)

    z_pred = jax.lax.fori_loop(0, int(config.side_adapter_sampling_steps), _body, z)
    diff = z_pred.astype(jnp.float32) - z_video.astype(jnp.float32)
    metrics = {
        "latent_mse": jnp.mean(diff**2),
        "latent_mae": jnp.mean(jnp.abs(diff)),
        "z_pred_std": jnp.std(z_pred.astype(jnp.float32)),
        "z_target_std": jnp.std(z_video.astype(jnp.float32)),
        "z_init_anchor_mse": jnp.mean((z_pred[:, :, :1].astype(jnp.float32) - z_i0.astype(jnp.float32)) ** 2),
    }
    return z_pred, metrics


def _build_side_adapter_validation_state(config):
    trainer = WanTI2VSideAdapterTrainer(config)
    pipeline = trainer._load_wan_pipeline()
    mesh = pipeline.mesh
    null_context = trainer._compute_null_context(pipeline, mesh)
    if hasattr(pipeline, "text_encoder"):
        del pipeline.text_encoder
    if hasattr(pipeline, "tokenizer"):
        del pipeline.tokenizer

    adapters = trainer._build_adapters(pipeline.transformer)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        adapter_graphdef, adapter_params, adapter_rest = nnx.split(adapters, nnx.Param, ...)
        transformer_graphdef, transformer_params, transformer_rest = nnx.split(pipeline.transformer, nnx.Param, ...)

    tx, _ = trainer._build_optimizer(config.max_train_steps)
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
    del pipeline.transformer
    state, state_shardings = trainer._shard_state(mesh, state)
    return trainer, pipeline, mesh, state, state_shardings


def _build_full_ft_validation_state(config):
    """Build the full-FT validation state: the transformer IS the trainable module.

    Mirrors the side-adapter builder but splits the transformer as the trainable
    ``params`` (no adapter is built), reusing the full-FT trainer's pipeline load,
    null-context, optimizer, and (keep-computed-FSDP) ``_shard_state`` -- so the state
    layout matches what training checkpointed.
    """
    trainer = WanTI2VFullFTTrainer(config)
    pipeline = trainer._load_wan_pipeline()
    mesh = pipeline.mesh
    null_context = trainer._compute_null_context(pipeline, mesh)
    if hasattr(pipeline, "text_encoder"):
        del pipeline.text_encoder
    if hasattr(pipeline, "tokenizer"):
        del pipeline.tokenizer

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        transformer_graphdef, transformer_params, transformer_rest = nnx.split(pipeline.transformer, nnx.Param, ...)

    tx, _ = trainer._build_optimizer(config.max_train_steps)
    state = FullFTTrainState.create(
        apply_fn=transformer_graphdef.apply,
        params=transformer_params,
        tx=tx,
        graphdef=transformer_graphdef,
        rest_of_state=transformer_rest,
        null_context=null_context,
    )
    del pipeline.transformer
    state, state_shardings = trainer._shard_state(mesh, state)
    return trainer, pipeline, mesh, state, state_shardings


def _restore_checkpoint_state(config, state, ckpt_dir, *, cohort_mode=False, requested_step=None):
    """Restore params/opt_state/step from the Orbax Composite checkpoint into ``state``.

    Shared by the side-adapter and full-FT validation paths: the Composite layout
    (params + opt_state via ``StandardCheckpointHandler``, step via JSON, read-only
    manager) matches the trainers' ``_save_checkpoint``. Returns ``(restored_state, step)``.

    ``cohort_mode`` (full-FT cohort evaluation, plan §2.3) enables two behaviors:
      * ``checkpoint_step == 0`` -> roll out the freshly-loaded pretrained weights: Orbax
        is NOT consulted (no manager built) and ``state`` is returned unchanged with
        step 0 (the within-cohort baseline needs no checkpoint).
      * ``checkpoint_step == N > 0`` must name an existing checkpoint; a missing N raises
        an actionable error listing the available steps (the cohort evaluates specific
        steps 2500/5000/7500/10000, so a bare Orbax failure would be unhelpful).
    In cohort mode the restored JSON step is also written into ``state.step`` (F1: the
    generic restore covers params/opt_state/STEP per the round-3 binding note), and the
    empty-directory error names the full-FT mode (F4). With ``cohort_mode=False`` the
    step resolution, state fields, and error message are byte-identical to the original
    side-adapter path (``requested if requested > 0 else latest_step()``).

    ``requested_step`` (Part II / Query 8 F4): when ``None`` (default) the requested step
    is read from ``config.checkpoint_step`` -- byte-identical to the pre-Part-II behavior,
    so the adapter path and the single-step T2 rollout are untouched. When an int is
    supplied it OVERRIDES the config-derived step, which is how the T1 per-checkpoint loop
    evaluates each cohort step in turn (pyconfig objects are immutable, so per-iteration
    ``config.checkpoint_step`` mutation is impossible).
    """
    requested = int(getattr(config, "checkpoint_step", -1)) if requested_step is None else int(requested_step)
    if cohort_mode and requested == 0:
        max_logging.log("[wan_side_adapter_val] checkpoint_step=0: pretrained baseline, skipping Orbax restore")
        return state, 0
    manager = ocp.CheckpointManager(
        ckpt_dir,
        item_names=("params", "opt_state", "step"),
        item_handlers={
            "params": ocp.StandardCheckpointHandler(),
            "opt_state": ocp.StandardCheckpointHandler(),
            "step": ocp.JsonCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    if cohort_mode and requested > 0:
        available = list(manager.all_steps())
        if requested not in available:
            raise ValueError(
                f"checkpoint_step={requested} not found in {ckpt_dir}; available steps: {sorted(available)}"
            )
        step = requested
    else:
        step = requested if requested > 0 else manager.latest_step()
        if step is None:
            # F4: name the actual mode -- outside cohort mode this stays byte-identical
            # to the historical adapter message.
            kind = "full-FT" if cohort_mode else "adapter"
            raise ValueError(f"No {kind} checkpoints found in {ckpt_dir}")
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(state.params),
            opt_state=ocp.args.StandardRestore(state.opt_state),
            step=ocp.args.JsonRestore(),
        ),
    )
    state = state.replace(params=restored["params"], opt_state=restored["opt_state"])
    restored_step = int(restored["step"]["step"])
    if cohort_mode:
        # F1: the generic restore covers params/opt_state/STEP -- the restored step goes
        # into the state itself, not only the returned scalar (round-3 binding note).
        # Full-FT-only: the adapter path's state.step stays exactly as before.
        state = state.replace(step=restored_step)
    max_logging.log(f"[wan_side_adapter_val] restored step={restored_step} from {ckpt_dir}")
    return state, restored_step


def _restore_validation_state(config):
    if str(getattr(config, "model_type", "")) == "OVERFIT100_TI2V":
        # exp_02 has its own builder (context table + null embedding + a 7-tuple return);
        # routing it through the exp_01 dispatcher would roll out with the wrong conditioning.
        raise ValueError(
            "model_type OVERFIT100_TI2V must use _restore_overfit100_validation_state / run_overfit100, not the "
            "exp_01 dispatcher (its state carries a per-episode context table, not a single null embedding)"
        )
    is_full_ft = getattr(config, "model_type", "") == "FULL_FT_TI2V"
    if is_full_ft:
        trainer, pipeline, mesh, state, state_shardings = _build_full_ft_validation_state(config)
    else:
        trainer, pipeline, mesh, state, state_shardings = _build_side_adapter_validation_state(config)

    ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, config.run_name, "checkpoints")
    state, restored_step = _restore_checkpoint_state(config, state, ckpt_dir, cohort_mode=is_full_ft)
    return trainer, pipeline, mesh, state, state_shardings, restored_step


def _validation_config_artifact(config, checkpoint_step: int, num_samples: int, seed: int) -> dict:
    """The provenance dict ``run()`` writes to ``<step_root>/config.json``.

    Adapter modes: byte-identical to the historical artifact. FULL_FT_TI2V (F3):
    ``action_adapter_type`` is dropped (no adapter exists in a full-FT run),
    ``model_type`` and the RESOLVED ``validation_ordinals`` (ordered dataset positions,
    plan §2.3 cohort) are recorded -- that list order IS the per-sample seed-assignment
    order, because ``run()`` splits the rollout rng sequentially over samples in exactly
    this order. When ordinals select the cohort, the contiguous
    ``validation_start_index`` is ignored and therefore omitted; with no ordinals the
    contiguous selector is in effect and the index stays on record.
    """
    artifact = {
        "run_name": config.run_name,
        "checkpoint_dir": config.checkpoint_dir,
        "checkpoint_step": checkpoint_step,
        "eval_data_dir": config.eval_data_dir,
        "num_eval_videos": num_samples,
        "validation_start_index": int(getattr(config, "validation_start_index", 0)),
        "seed": seed,
        "side_adapter_sampling_steps": int(config.side_adapter_sampling_steps),
        "side_adapter_guide_scale": float(config.side_adapter_guide_scale),
        "action_adapter_type": getattr(config, "action_adapter_type", "side_adapter"),
    }
    if getattr(config, "model_type", "") == "FULL_FT_TI2V":
        del artifact["action_adapter_type"]
        ordinals = _parse_ordinals(getattr(config, "validation_ordinals", ""))
        artifact["model_type"] = config.model_type
        artifact["validation_ordinals"] = ordinals
        if ordinals:
            del artifact["validation_start_index"]
    return artifact


def _write_json(path: str, value: dict):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    parent = os.path.dirname(path)
    if parent:
        tf.io.gfile.makedirs(parent)
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(payload)


def _copy_local_to_output(local_path: str, output_path: str):
    parent = os.path.dirname(output_path)
    if parent:
        tf.io.gfile.makedirs(parent)
    tf.io.gfile.copy(local_path, output_path, overwrite=True)


def _save_video(frames: np.ndarray, output_path: str, fps: int):
    frames = np.clip(frames, 0.0, 1.0)
    with tempfile.TemporaryDirectory(prefix="wan_side_adapter_val_") as tmpdir:
        local_path = os.path.join(tmpdir, os.path.basename(output_path))
        export_to_video(list(frames), local_path, fps=fps)
        _copy_local_to_output(local_path, output_path)


def _frame_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")
    vals = []
    for p, t in zip(pred, target):
        h, w = p.shape[:2]
        win = min(7, h, w)
        if win % 2 == 0:
            win -= 1
        if win < 3:
            return float("nan")
        vals.append(
            float(
                structural_similarity(
                    t,
                    p,
                    channel_axis=-1,
                    data_range=1.0,
                    win_size=win,
                )
            )
        )
    return float(np.mean(vals))


def _as_batch(sample: EvalSample) -> dict:
    return {
        "z_i0": jnp.asarray(sample.z_i0[None]),
        "z_video": jnp.asarray(sample.z_video[None]),
        "actions": jnp.asarray(sample.actions[None]),
    }


# ======================================================================================
# exp_02 OVERFIT100_TI2V: selection, context modes, rollout, metrics, aggregation.
# ======================================================================================

OVERFIT100_MODEL_TYPE = "OVERFIT100_TI2V"
OVERFIT100_CONTEXT_MODES = ("correct", "null", "shuffled")
OVERFIT100_VIEW_INDEX = 0
OVERFIT100_WINDOW_STRIDE = 4

# Every aggregation row carries exactly these keys, in this order (the CSV column order too).
OVERFIT100_ROW_FIELDS = (
    "name",
    "episode_id",
    "episode_index",
    "window_start",
    "canonical",
    "checkpoint_step",
    "seed",
    "context_mode",
    "context_source_episode_index",
    "ssim",
    "latent_mse",
    "latent_mae",
    "pixel_mse",
    "pixel_mae",
    "z_pred_std",
    "z_target_std",
    "z_init_anchor_mse",
    "ssim_vs_rgb",
    "pixel_mse_vs_rgb",
    "vae_ceiling_ssim",
    "aux_status",
)

# The columns the aggregation artifact REQUIRES of every row: the window identity, the
# measurement coordinates, and the primary metrics -- i.e. everything the success statistic
# and the D11 coverage matrix read. The remaining OVERFIT100_ROW_FIELDS entries are diagnostic
# (auxiliary RGB block, latent/pixel norms) and may legitimately be absent from a
# hand-assembled row; ``overfit100_metric_row`` always emits all of them.
OVERFIT100_ROW_REQUIRED = (
    "name",
    "episode_id",
    "episode_index",
    "window_start",
    "canonical",
    "checkpoint_step",
    "seed",
    "context_mode",
    "ssim",
    "latent_mse",
    "pixel_mse",
)

# The builder zero-pads the start to AT LEAST five digits (``:05d``), so a start >= 100000 is six
# digits (D6). Matching only five would silently reject a legitimate long-episode window name.
_WINDOW_NAME_RE = re.compile(r"^ep(\d+)_v(\d+)_s(\d{5,})$")


@dataclass
class Overfit100EvalSample:
    """One selected schema-v2 window, decoded to host arrays."""

    name: str
    episode_id: int
    episode_index: int
    window_start: int
    canonical: bool
    position: int
    z_i0: np.ndarray
    z_video: np.ndarray
    instruction: str


def _is_overfit100(config) -> bool:
    return str(getattr(config, "model_type", "")) == OVERFIT100_MODEL_TYPE


def overfit100_window_name(episode_id: int, window_start: int, view: int = OVERFIT100_VIEW_INDEX) -> str:
    """The schema-v2 record name: ``ep<ID>_v<VIEW>_s<START zero-padded to 5>``.

    Mirrors ``build_overfit100_dataset.window_name`` (pinned equal by a test) rather than
    importing it, so the eval path never pulls in the builder's ffmpeg/gsutil-oriented module
    at import time -- the same reason the trainer mirrors the sidecar constants.
    """
    return f"ep{int(episode_id)}_v{int(view)}_s{int(window_start):05d}"


def parse_overfit100_window_name(name: str) -> tuple[int, int, int]:
    """``ep<ID>_v<VIEW>_s<START>`` -> ``(episode_id, view, window_start)``.

    Accepts the builder's "at least five digits" padding (D6) but requires the name to ROUND-TRIP
    through :func:`overfit100_window_name`: ``ep100_v0_s000004`` is not a name the builder writes,
    and admitting it would let two spellings of one window into a cohort.
    """
    text = str(name).strip()
    match = _WINDOW_NAME_RE.match(text)
    if not match:
        raise ValueError(
            f"{name!r} is not a schema-v2 window name; expected ep<EPISODE_ID>_v<VIEW>_s<START zero-padded to at "
            f"least 5 digits>, e.g. ep25189_v0_s00048"
        )
    episode_id, view, start = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if overfit100_window_name(episode_id, start, view) != text:
        raise ValueError(
            f"{name!r} is not the canonical spelling of that window (the builder writes "
            f"{overfit100_window_name(episode_id, start, view)!r}); refusing an ambiguous window name"
        )
    return episode_id, view, start


def manifest_canonical_cohort(manifest_path: str, *, episode_indices=None) -> tuple[tuple[int, int], ...]:
    """The FIXED canonical cohort (D1), read from the manifest FILE through the strict reader.

    Delegates the math to the pure statistic module so the evaluator and the verdict share ONE
    definition of the cohort; only the file read differs (pinned equal by a parity test).
    """
    return canonical_cohort_from_manifest(
        _read_json_strict(manifest_path, "model manifest"), episode_indices=episode_indices
    )


def manifest_all_window_keys(manifest_path: str, *, episode_indices=None) -> tuple[tuple[int, int], ...]:
    """Every BUILT window key of the manifest (the full-set cohort, D3), from the manifest FILE."""
    return all_window_keys_from_manifest(
        _read_json_strict(manifest_path, "model manifest"), episode_indices=episode_indices
    )


def manifest_episode_windows(manifest_path: str) -> dict[int, dict]:
    """``episode_index -> {episode_id, used_text, n_windows, canonical_start, canonical_name}``.

    The COMMITTED manifest is the authenticated source of ``n_windows`` (the dataset's
    ``episodes.json`` says only WHICH episodes a set contains), and cycle C's preflight binds
    that manifest to the published bytes via ``_SUCCESS.manifest_sha256``. Selection therefore
    inherits the same provenance chain as training's context table.
    """
    base = read_manifest_episodes(manifest_path)  # validates indices / ids / texts
    manifest = _read_json_strict(manifest_path, "model manifest")
    counts: dict[int, int] = {}
    for entry in manifest.get("episodes") or []:
        index = int(entry["episode_index"])
        if "n_windows" not in entry:
            raise ValueError(
                f"manifest {manifest_path} episode_index {index} carries no n_windows; the canonical-window "
                f"selection is defined as 4 * floor((n_windows - 1) / 2) and cannot be computed without it."
            )
        counts[index] = int(entry["n_windows"])
    out: dict[int, dict] = {}
    for index, info in base.items():
        start = canonical_window_start(counts[index])
        out[index] = {
            "episode_index": int(index),
            "episode_id": int(info["episode_id"]),
            "used_text": str(info["used_text"]),
            "n_windows": int(counts[index]),
            "canonical_start": start,
            "canonical_name": overfit100_window_name(info["episode_id"], start),
        }
    return out


def parse_eval_pass_role(config) -> str:
    """The MANDATORY pass role (D2): which D11 coverage cell this pass is.

    There is no default: an unlabeled pass cannot be validated, and an unvalidated pass must never
    reach the verdict. ``eval_pass_role`` is therefore required, closed over
    :data:`OVERFIT100_PASS_ROLES`, and case-sensitive (the role is a machine key that lands in the
    artifact and in the output path).
    """
    role = str(getattr(config, "eval_pass_role", "") or "").strip()
    if role not in OVERFIT100_PASS_ROLES:
        raise ValueError(
            f"eval_pass_role must be one of {list(OVERFIT100_PASS_ROLES)} (D11's coverage matrix); got {role!r}. "
            f"Every exp_02 eval pass declares its role so the verdict can validate -- never assume -- what it covers."
        )
    return role


def assert_pass_role_plan(
    role: str,
    *,
    seeds,
    modes,
    sampling_steps,
    windows,
    cohort,
    all_window_keys,
) -> dict:
    """Refuse a mislabeled pass in SECONDS, before the 5B load (D2).

    The same pure contract the aggregator re-checks on the written artifact
    (``pass_role_plan_reasons``), applied to the plan: this pass's seeds, modes, sampling steps and
    selected windows against the manifest-derived cohorts. Returns the ``role_validation`` block
    the artifact records.
    """
    covered = tuple(_window_key_tuple(window) for window in windows)
    covered_canonical = tuple(_window_key_tuple(window) for window in windows if window.get("canonical"))
    reasons = pass_role_plan_reasons(
        role,
        seeds=seeds,
        modes=modes,
        sampling_steps=sampling_steps,
        covered_canonical=covered_canonical,
        covered_all=covered,
        cohort=cohort,
        all_window_keys=all_window_keys,
    )
    if reasons:
        raise ValueError(
            f"this pass does not satisfy its declared role {role!r} (D11 coverage matrix):\n  - "
            + "\n  - ".join(reasons)
            + f"\nEither fix the pass configuration or declare the role it really is "
            f"({list(OVERFIT100_PASS_ROLES)})."
        )
    return {
        "role": role,
        "ok": True,
        "scope": role_requirements(role)["scope"],
        "cohort_size": len(cohort),
        "all_windows_size": len(all_window_keys),
        "covered": len(covered),
        "covered_canonical": len(covered_canonical),
        "sampling_steps": int(sampling_steps),
        "seeds": [int(s) for s in seeds],
        "modes": [str(m) for m in modes],
    }


def parse_eval_windows_spec(text) -> tuple[str, tuple[str, ...]]:
    """The ``--eval-windows`` spec (config key ``eval_windows``).

    ``"canonical"`` selects one canonical window per episode IN THE SET; ``"all"`` selects EVERY
    built window of the set (the ``s3_full_set`` pass's scope -- a 1,629-name list is not a usable
    CLI value); anything else is a comma-separated list of window NAMES kept in the listed order
    (that order is the rollout order, and each window's rollout rng is keyed on its own identity,
    so the order never changes a number). Empty specs and duplicate names are refused -- a
    duplicate would double-count a window in the fixed denominator.
    """
    text = str(text or "").strip()
    if not text:
        raise ValueError(
            "eval_windows must be 'canonical' (one canonical window per episode in the set), 'all' (every built "
            "window of the set), or a comma-separated list of schema-v2 window names; got an empty value"
        )
    if text.lower() == "canonical":
        return ("canonical", ())
    if text.lower() == "all":
        return ("all", ())
    names = [token.strip() for token in text.split(",") if token.strip()]
    if not names:
        raise ValueError(f"eval_windows={text!r} parsed to no window names")
    for name in names:
        parse_overfit100_window_name(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"eval_windows lists duplicate window names {duplicates}; each window may appear only once")
    return ("names", tuple(names))


def _window_descriptor(info: dict, window_start: int) -> dict:
    return {
        "name": overfit100_window_name(info["episode_id"], window_start),
        "episode_id": int(info["episode_id"]),
        "episode_index": int(info["episode_index"]),
        "window_start": int(window_start),
        "canonical": int(window_start) == int(info["canonical_start"]),
        "used_text": str(info["used_text"]),
    }


def select_eval_windows(spec: tuple[str, tuple[str, ...]], episodes: dict[int, dict]) -> list[dict]:
    """Resolve a parsed spec against the set's episodes -> ordered window descriptors.

    ``canonical``: one window per episode, ordered by ``episode_index`` (a SPARSE episode set
    is fine -- cycle C's ratified semantics -- and nothing here assumes contiguity).
    Explicit names: kept in the listed order, each validated to be a real window of a real
    episode (on the 4-frame grid and within ``n_windows``).
    """
    kind, names = spec
    if not episodes:
        raise ValueError("no episodes to select from: the eval set's episodes.json listed none")
    by_id: dict[int, dict] = {}
    for info in episodes.values():
        episode_id = int(info["episode_id"])
        if episode_id in by_id:
            raise ValueError(f"two episodes in the eval set share episode_id {episode_id}")
        by_id[episode_id] = info
    if kind == "canonical":
        return [_window_descriptor(episodes[index], episodes[index]["canonical_start"]) for index in sorted(episodes)]
    if kind == "all":
        return [
            _window_descriptor(episodes[index], OVERFIT100_WINDOW_STRIDE * slot)
            for index in sorted(episodes)
            for slot in range(int(episodes[index]["n_windows"]))
        ]
    out = []
    for name in names:
        episode_id, view, start = parse_overfit100_window_name(name)
        if view != OVERFIT100_VIEW_INDEX:
            raise ValueError(f"eval window {name!r} names view {view}; the built dataset has view 0 only")
        info = by_id.get(episode_id)
        if info is None:
            raise ValueError(
                f"eval window {name!r} names episode_id {episode_id}, which the eval set does not contain "
                f"({len(by_id)} episodes: {sorted(by_id)[:5]}...)"
            )
        last = OVERFIT100_WINDOW_STRIDE * (int(info["n_windows"]) - 1)
        if start % OVERFIT100_WINDOW_STRIDE or start > last:
            raise ValueError(
                f"eval window {name!r} is not a built window of episode {episode_id}: starts are "
                f"0, {OVERFIT100_WINDOW_STRIDE}, ... {last} ({info['n_windows']} windows)"
            )
        out.append(_window_descriptor(info, start))
    return out


def resolve_eval_windows(config, *, data_dir=None, manifest_path=None) -> list[dict]:
    """The production selection: manifest window counts x the SET's own episode list x the spec.

    Every episode the set claims must exist in the committed manifest and agree with it on
    ``(episode_id, used_text)`` -- the same triple check cycle C's preflight makes -- so a
    selection can never be computed against an episode the experiment did not review.
    """
    data_dir = str(data_dir or getattr(config, "eval_data_dir", "") or "")
    if not data_dir:
        raise ValueError("eval_data_dir must point at an exp_02 overfit100 schema-v2 TFRecord set")
    manifest_path = str(manifest_path or getattr(config, "model_manifest_path", "") or "").strip()
    if not manifest_path:
        raise ValueError(
            "overfit100 eval requires model_manifest_path: the committed manifest supplies each episode's "
            "n_windows (hence the canonical window) and authenticates episodes.json"
        )
    manifest_map = manifest_episode_windows(manifest_path)
    set_map = read_episode_mapping(data_dir)
    episodes: dict[int, dict] = {}
    for index in sorted(set_map):
        if index not in manifest_map:
            raise ValueError(
                f"eval set {data_dir} claims episode_index {index}, which the committed manifest {manifest_path} "
                f"does not select (it defines 0..{max(manifest_map)})"
            )
        for field in ("episode_id", "used_text"):
            if set_map[index][field] != manifest_map[index][field]:
                raise ValueError(
                    f"overfit100 eval selection binding broken at episode_index {index}: episodes.json in "
                    f"{data_dir} has {field}={set_map[index][field]!r} but the committed manifest {manifest_path} "
                    f"has {field}={manifest_map[index][field]!r}; refusing to evaluate an unreviewed episode."
                )
        episodes[index] = manifest_map[index]
    return select_eval_windows(parse_eval_windows_spec(getattr(config, "eval_windows", "")), episodes)


# ---------------------------------------------------------------------------- schema-v2 reader


def _overfit100_feature_description() -> dict:
    """Schema v2 (plan D6). No ``actions``; ``name``/``episode_id``/``window_start`` included
    because the aggregation artifact is keyed on the window identity (cycle-C judgment 7)."""
    return {
        "name": tf.io.FixedLenFeature([], tf.string),
        "episode_id": tf.io.FixedLenFeature([], tf.int64),
        "episode_index": tf.io.FixedLenFeature([], tf.int64),
        "window_start": tf.io.FixedLenFeature([], tf.int64),
        "z_i0": tf.io.FixedLenFeature([], tf.string),
        "z_video": tf.io.FixedLenFeature([], tf.string),
        "instruction": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }


def _iter_overfit100_records(config):
    """Yield parsed schema-v2 records in shard order. The single TFRecord seam (tests patch it)."""
    ds = tf.data.TFRecordDataset(_tfrecord_files(config.eval_data_dir))
    ds = ds.map(lambda raw: tf.io.parse_single_example(raw, _overfit100_feature_description()))
    return ds.as_numpy_iterator()


def read_overfit100_samples(config, windows) -> list[Overfit100EvalSample]:
    """Read exactly the selected windows, returned IN THE REQUESTED ORDER.

    The reader is drained to EOF: the requested windows are scattered through the shards (a
    sparse subset read, cycle C's ratified semantics), and the full drain is also what catches
    a duplicated window name. Every matched record must agree with the selection on
    ``episode_index`` / ``episode_id`` / ``window_start`` -- the context gather uses
    ``episode_index``, so a contradiction would score against the wrong instruction.
    """
    c = int(config.latent_channels)
    f = int(config.latent_frames)
    h = int(config.latent_height)
    w = int(config.latent_width)
    wanted = {str(window["name"]): window for window in windows}
    if len(wanted) != len(windows):
        raise ValueError("the selected window list contains duplicate names")

    found: dict[str, tuple[int, dict]] = {}
    seen = 0
    for position, raw in enumerate(_iter_overfit100_records(config)):
        seen = position + 1
        name = raw["name"].decode("utf-8") if isinstance(raw["name"], bytes) else str(raw["name"])
        if name not in wanted:
            continue
        if name in found:
            raise ValueError(
                f"overfit100 eval set {config.eval_data_dir} contains a duplicate record named {name!r} "
                f"(positions {found[name][0]} and {position}); a window must be unique."
            )
        found[name] = (position, raw)

    missing = [str(window["name"]) for window in windows if str(window["name"]) not in found]
    if missing:
        raise ValueError(
            f"selected windows not present in {config.eval_data_dir}: {missing[:8]}"
            f"{' (+%d more)' % (len(missing) - 8) if len(missing) > 8 else ''}; the reader was drained to EOF and "
            f"saw {seen} records."
        )

    samples: list[Overfit100EvalSample] = []
    for window in windows:
        position, raw = found[str(window["name"])]
        for field in ("episode_index", "episode_id", "window_start"):
            stored = int(raw[field])
            if stored != int(window[field]):
                raise ValueError(
                    f"record {window['name']!r} at position {position} stores {field}={stored} but the selection "
                    f"says {int(window[field])}; the eval set does not match the manifest-derived selection."
                )
        instruction = raw.get("instruction", b"") or b""
        samples.append(
            Overfit100EvalSample(
                name=str(window["name"]),
                episode_id=int(window["episode_id"]),
                episode_index=int(window["episode_index"]),
                window_start=int(window["window_start"]),
                canonical=bool(window["canonical"]),
                position=position,
                z_i0=np.frombuffer(raw["z_i0"], dtype=np.float16).reshape(c, 1, h, w).astype(np.float32),
                z_video=np.frombuffer(raw["z_video"], dtype=np.float16).reshape(c, f, h, w).astype(np.float32),
                instruction=instruction.decode("utf-8") if isinstance(instruction, bytes) else str(instruction),
            )
        )
    return samples


# ---------------------------------------------------------------------------- seeds / context


def window_fold_key(seed: int, episode_id: int, window_start: int) -> jax.Array:
    """The per-window rng key: ``fold_in(fold_in(key(seed), episode_id), window_start)``.

    Keyed on the window IDENTITY, never on its position in the dataset: the same window rolls
    out with the same noise no matter which selection, shard order, or batch it arrived in, so
    two eval passes are comparable. Shared with the one-step val-loss evaluator, which folds
    the SAME key (see ``eval_wan_full_ft_val_loss.per_window_rng``).
    """
    if int(episode_id) < 0 or int(window_start) < 0:
        raise ValueError(
            f"window_fold_key needs non-negative (episode_id, window_start); got ({episode_id}, {window_start})"
        )
    key = jax.random.fold_in(jax.random.key(int(seed)), int(episode_id))
    return jax.random.fold_in(key, int(window_start))


def parse_rollout_seeds(text) -> tuple[int, ...]:
    """``rollout_seeds`` -> a non-empty, duplicate-free tuple of non-negative ints, in order."""
    text = str(text or "").strip()
    if not text:
        raise ValueError("rollout_seeds must be a non-empty comma-separated list of non-negative integers")
    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"rollout_seeds must be comma-separated integers; got token {token!r}") from exc
        if value < 0:
            raise ValueError(f"rollout_seeds entries must be non-negative; got {value}")
        seeds.append(value)
    if not seeds:
        raise ValueError("rollout_seeds parsed to empty; provide at least one seed")
    duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
    if duplicates:
        raise ValueError(f"rollout_seeds repeats {duplicates}; a repeated seed would duplicate a measurement")
    return tuple(seeds)


def parse_context_modes(text) -> tuple[str, ...]:
    """``context_modes`` -> an ordered, duplicate-free tuple drawn from ``correct/null/shuffled``."""
    text = str(text or "").strip()
    if not text:
        raise ValueError(
            f"context_modes must be a non-empty comma-separated subset of {list(OVERFIT100_CONTEXT_MODES)}"
        )
    modes = [token.strip() for token in text.split(",") if token.strip()]
    if not modes:
        raise ValueError(f"context_modes parsed to empty; valid modes are {list(OVERFIT100_CONTEXT_MODES)}")
    for mode in modes:
        if mode not in OVERFIT100_CONTEXT_MODES:
            raise ValueError(f"unknown context mode {mode!r}; valid modes are {list(OVERFIT100_CONTEXT_MODES)}")
    duplicates = sorted({m for m in modes if modes.count(m) > 1})
    if duplicates:
        raise ValueError(f"context_modes contains duplicate mode(s) {duplicates}; each mode is rolled out once")
    return tuple(modes)


def parse_flagged_windows(text) -> tuple[str, ...]:
    """``flagged_windows`` -> the collision-flagged window names (may be empty).

    Flagging is REPORTING only: the plan fixes the denominator at build time, so a flagged
    window is recorded in the aggregation artifact and stays in the statistic (G4). There is
    no code path that removes one.
    """
    text = str(text or "").strip()
    if not text:
        return ()
    names = [token.strip() for token in text.split(",") if token.strip()]
    for name in names:
        parse_overfit100_window_name(name)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"flagged_windows repeats {duplicates}")
    return tuple(names)


def _seeded_order(items, seed: int, salt: str = ""):
    """A deterministic, platform- and version-independent seeded ordering.

    A sha256 keyed sort rather than an RNG: reproducibility must not depend on any library's
    bit-generator stream staying stable across versions, and the tests assert exact tuples.
    """

    def _key(item):
        token = item[0] if isinstance(item, tuple) else item
        digest = hashlib.sha256(f"{int(seed)}|{salt}|{token!r}".encode("utf-8")).hexdigest()
        return (digest, repr(token))

    return sorted(items, key=_key)


def value_derangement(texts: Sequence[str], seed: int) -> tuple[int, ...]:
    """A seeded derangement of instruction VALUES: ``sigma`` with ``texts[sigma[i]] != texts[i]``.

    Plan §1: an index-level derangement is too weak, because 6 duplicate-instruction groups
    cover 22 of the 100 episodes -- a fixed-point-free permutation can still hand an episode a
    string equal to its own. The construction is therefore duplicate-aware:

    1. group the episode indices by their instruction (compared after ``strip()``);
    2. lay the groups out CONTIGUOUSLY in one list ``L`` (both the group order and the order
       within each group are the seeded permutation, so the assignment varies with the seed);
    3. rotate by ``s = n_max``, the largest group size: ``sigma[L[j]] = L[(j + s) mod N]``.

    Why that is always a value derangement when ``2 * n_max <= N``: a group occupies a
    contiguous block of length ``n_g`` in ``L``, so a same-group collision needs
    ``s ≡ d (mod N)`` for some ``|d| < n_g``, i.e. ``s in {0..n_g-1} ∪ {N-n_g+1..N-1}``. With
    ``s = n_max``: ``s >= n_g`` excludes the low set, and ``s = n_max <= N - n_max <= N - n_g``
    excludes the high one. Hence every episode is sent to a DIFFERENT instruction group.
    ``2 * n_max > N`` makes a value derangement impossible (the majority instruction would have
    to receive itself), and is refused loudly. The property is re-verified on the result before
    returning, so a future change to the layout cannot silently break it.
    """
    normalized = [str(text).strip() for text in texts]
    n = len(normalized)
    if n < 2:
        raise ValueError(f"a value derangement needs at least 2 episodes; got {n}")
    groups: dict[str, list[int]] = {}
    for index, text in enumerate(normalized):
        groups.setdefault(text, []).append(index)
    n_max = max(len(members) for members in groups.values())
    if 2 * n_max > n:
        biggest = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[0][0]
        raise ValueError(
            f"no value derangement exists: the largest duplicate-instruction group has {n_max} of {n} episodes "
            f"({biggest!r}), and a value-level derangement requires 2 * n_max <= N. Reducing the cohort's duplicate "
            f"instructions (or dropping the shuffled ablation) is the only fix -- refusing to emit a shuffled "
            f"assignment that hands an episode its own instruction."
        )
    layout: list[int] = []
    for text, members in _seeded_order(list(groups.items()), seed):
        layout.extend(_seeded_order(list(members), seed, salt=text))
    shift = n_max
    sigma = [0] * n
    for position, index in enumerate(layout):
        sigma[index] = layout[(position + shift) % n]
    for index in range(n):
        if normalized[sigma[index]] == normalized[index]:
            raise RuntimeError(
                f"value_derangement produced a self-assignment at episode index {index}; the group-rotation "
                f"invariant is broken (this is a bug, not a data condition)"
            )
    return tuple(sigma)


def context_source_index(mode: str, episode_index: int, derangement) -> int | None:
    """Which context-table ROW a mode reads for one episode (``None`` = the null embedding)."""
    if mode not in OVERFIT100_CONTEXT_MODES:
        raise ValueError(f"unknown context mode {mode!r}; valid modes are {list(OVERFIT100_CONTEXT_MODES)}")
    index = int(episode_index)
    if index < 0:
        raise ValueError(f"episode_index must be non-negative; got {episode_index}")
    if mode == "null":
        return None
    if mode == "correct":
        return index
    if derangement is None:
        raise ValueError("context mode 'shuffled' needs a derangement (built from the context table's texts)")
    if index >= len(derangement):
        raise ValueError(
            f"episode_index {index} is outside the derangement over {len(derangement)} context-table rows; the "
            f"shuffled ablation is defined over the table the run was built with."
        )
    return int(derangement[index])


def overfit100_context_for_mode(context_table, null_context, mode: str, episode_index: int, derangement):
    """The ``[1, L, D]`` context this (window, mode) rolls out with."""
    index = context_source_index(mode, episode_index, derangement)
    if index is None:
        return jnp.asarray(null_context)
    table = jnp.asarray(context_table)
    if index >= table.shape[0]:
        raise ValueError(f"context row {index} does not exist in a {table.shape[0]}-row context table")
    return table[index][None]


# ---------------------------------------------------------------------------- rollout + state


def _rollout_overfit100_sample(state: Overfit100TrainState, data: dict, rng: jax.Array, scheduler, config):
    """exp_01's full-FT rollout, with the per-window text context supplied in the batch.

    Line-for-line the ``_rollout_sample`` full-FT branch -- same ``build_rollout_sigmas`` grid,
    same ``rollout_timesteps_from_sigmas``, same per-token timestep, same frame-0 pin before and
    inside the loop, same Euler update, one plain transformer forward per step with no actions,
    no adapter and no CFG -- except that ``encoder_hidden_states`` is ``data["context"]`` (the
    mode's row of ``state.context_table``, or the null embedding) instead of
    ``state.null_context``. Pinned bitwise-equal to the exp_01 path by a parity test.
    """
    weights_dtype = _dtype(config.weights_dtype)
    transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)

    z_i0 = data["z_i0"].astype(weights_dtype)
    z_video = data["z_video"].astype(weights_dtype)
    b, _, f_lat, h_lat, w_lat = z_video.shape

    sigmas = build_rollout_sigmas(
        config.side_adapter_sampling_steps,
        config.flow_shift,
        scheduler.config.sigma_min,
        scheduler.config.sigma_max,
    )
    timesteps = rollout_timesteps_from_sigmas(sigmas, scheduler.config.num_train_timesteps)
    context = data["context"].astype(weights_dtype)
    context = jnp.broadcast_to(context, (b, context.shape[1], context.shape[2]))

    z = jax.random.normal(rng, z_video.shape, dtype=z_video.dtype)
    z = apply_first_frame_pin(z, z_i0)

    def _body(i, current):
        step_t = jnp.broadcast_to(timesteps[i], (b,))
        timestep_2d = _build_per_token_timestep(step_t, f_lat, h_lat, w_lat, n_hist=1)
        v = transformer(
            hidden_states=current,
            timestep=timestep_2d,
            encoder_hidden_states=context,
            deterministic=True,
        )
        return apply_first_frame_pin(current + (sigmas[i + 1] - sigmas[i]).astype(current.dtype) * v, z_i0)

    z_pred = jax.lax.fori_loop(0, int(config.side_adapter_sampling_steps), _body, z)
    diff = z_pred.astype(jnp.float32) - z_video.astype(jnp.float32)
    metrics = {
        "latent_mse": jnp.mean(diff**2),
        "latent_mae": jnp.mean(jnp.abs(diff)),
        "z_pred_std": jnp.std(z_pred.astype(jnp.float32)),
        "z_target_std": jnp.std(z_video.astype(jnp.float32)),
        "z_init_anchor_mse": jnp.mean((z_pred[:, :, :1].astype(jnp.float32) - z_i0.astype(jnp.float32)) ** 2),
    }
    return z_pred, metrics


def _build_overfit100_validation_state(config):
    """Build the exp_02 eval state: full-FT transformer params + the per-episode context table.

    Order is fail-closed and matches the trainer's ``start_training``: config gates, then the
    pinned-snapshot check, then the dataset/manifest preflight -- all BEFORE the ~5B pipeline
    loads. The context table is built with the trainer's OWN builder (so eval conditions on
    exactly what training conditioned on) and the exp_01 null embedding is computed alongside
    it for the ``null`` ablation, both before the text encoder is freed. The VAE is KEPT: this
    path decodes latents to video.

    Returns ``(trainer, pipeline, mesh, state, state_shardings, null_context)``.
    """
    trainer = WanTI2VOverfit100Trainer(config)
    trainer._validate_probe_config(config)
    trainer._validate_overfit100_config(config)
    trainer._validate_pinned_snapshot(config)
    trainer._preflight_dataset()

    pipeline = trainer._load_wan_pipeline()
    mesh = pipeline.mesh
    context_table = trainer._build_context_table(pipeline, mesh)
    null_context = trainer._compute_null_context(pipeline, mesh)
    for attr in ("text_encoder", "tokenizer"):
        if hasattr(pipeline, attr):
            delattr(pipeline, attr)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        transformer_graphdef, transformer_params, transformer_rest = nnx.split(pipeline.transformer, nnx.Param, ...)

    tx, _ = trainer._build_optimizer(config.max_train_steps)
    state = Overfit100TrainState.create(
        apply_fn=transformer_graphdef.apply,
        params=transformer_params,
        tx=tx,
        graphdef=transformer_graphdef,
        rest_of_state=transformer_rest,
        context_table=context_table,
    )
    del pipeline.transformer
    state, state_shardings = trainer._shard_state(mesh, state)
    return trainer, pipeline, mesh, state, state_shardings, null_context


def _restore_overfit100_validation_state(config):
    """Build + restore (params/opt_state/step) the exp_02 eval state.

    Reuses exp_01's ``_restore_checkpoint_state`` with ``cohort_mode=True``: ``checkpoint_step=0``
    rolls out the pretrained weights without consulting Orbax, a positive step must exist (the
    error lists the available steps), and the restored step lands in ``state.step`` too.
    Returns ``(trainer, pipeline, mesh, state, state_shardings, null_context, restored_step)``.
    """
    trainer, pipeline, mesh, state, state_shardings, null_context = _build_overfit100_validation_state(config)
    ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, config.run_name, "checkpoints")
    state, restored_step = _restore_checkpoint_state(config, state, ckpt_dir, cohort_mode=True)
    return trainer, pipeline, mesh, state, state_shardings, null_context, restored_step


# ---------------------------------------------------------------------------- metrics + rows


# The external binaries the auxiliary RGB path shells out to: ``gsutil`` pulls the pinned MP4 and
# ``ffmpeg`` decodes it. Neither is on the stock TPU worker image.
OVERFIT100_AUX_BINARIES = ("ffmpeg", "gsutil")


def aux_prerequisite_warning(want_aux: bool, *, which=None) -> str | None:
    """One loud startup line when the auxiliary metrics cannot possibly succeed (S3 finding).

    D5's contract is unchanged -- a missing binary must never fail the run, and every row still
    records its own ``aux_status``. But the S3 intermediate evals showed the gap that leaves: the
    job log looked clean and the degradation was only discoverable by reading 90 null rows out of
    the artifact afterwards. So when auxiliary metrics are REQUESTED and a required binary is
    absent, the driver says so once, up front, before any rollout.

    Returns ``None`` when nothing is wrong (or when aux was not requested); ``which`` is injectable
    for tests.
    """
    if not want_aux:
        return None
    probe = which or shutil.which
    missing = [binary for binary in OVERFIT100_AUX_BINARIES if not probe(binary)]
    if not missing:
        return None
    return (
        f"[wan_overfit100_val] WARNING: aux RGB requested but {', '.join(missing)} missing — ALL aux metrics will "
        f"be null (ssim_vs_rgb / pixel_mse_vs_rgb / vae_ceiling_ssim), and every row's aux_status will record the "
        f"failure. The rollouts and the PRIMARY metrics are unaffected. Install the missing binary on the worker "
        f"(bash_scripts/validate_wan_overfit100.sh ensures ffmpeg) or set eval_aux_rgb=False to stop requesting it."
    )


def assert_ssim_available() -> None:
    """Refuse to start an exp_02 pass that cannot compute SSIM (D11's ONLY statistic input).

    ``_frame_ssim`` returns NaN when scikit-image is missing. exp_01 could live with that -- SSIM
    was one diagnostic among several -- but exp_02's success rule IS
    ``median_seed SSIM(correct)``, and the statistic (correctly) refuses non-finite input. Without
    this gate a whole eval pass would run to completion, write a full artifact, and only then be
    discovered unusable, so the check runs before the 5B load.
    """
    try:
        from skimage.metrics import structural_similarity  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- any import failure is the same fatal condition
        raise ValueError(
            "overfit100 eval requires scikit-image: SSIM is the success statistic's only input "
            "(plan D11: m_corr = median-over-seeds SSIM), and without it every row would carry NaN and "
            f"the statistic would refuse the whole pass. Install scikit-image on the worker ({exc})."
        ) from exc


def overfit100_aux_rgb(manifest_path, episode_id, window_start, pred_frames, gt_frames, cache=None) -> dict:
    """AUXILIARY metrics against the true DROID RGB frames + that window's VAE ceiling.

    The primary metrics compare against the VAE decode of the stored ``z_video``; this pairs
    them with the source pixels, always alongside ``vae_ceiling_ssim`` = SSIM(decode(target),
    RGB) -- the ceiling any generation could reach through this VAE (plan D11 / gate V3, which
    computes the same number with the same bfloat16 decode postprocess).

    The MP4 is pulled at the manifest's exact recorded generation and bound to its md5
    (``build_overfit100_dataset.fetch_pinned``). Everything here is best-effort: any failure
    (no gsutil, no ffmpeg, network, drifted object) returns ``None`` metrics plus a status
    string and NEVER fails the eval run. ``cache`` is an optional single-episode frame cache.
    """
    result = {"ssim_vs_rgb": None, "pixel_mse_vs_rgb": None, "vae_ceiling_ssim": None, "aux_status": "unavailable"}
    try:
        frames = None
        if cache is not None and cache.get("episode_id") == int(episode_id):
            frames = cache.get("frames")
        if frames is None:
            from maxdiffusion.data_preprocessing.build_overfit100_dataset import decode_mp4_frames, fetch_pinned

            manifest = _read_json_strict(manifest_path, "model manifest")
            entry = next(
                (e for e in manifest.get("episodes") or [] if int(e["episode_id"]) == int(episode_id)),
                None,
            )
            if entry is None:
                result["aux_status"] = f"episode {int(episode_id)} not in the manifest"
                return result
            fingerprint = entry["video_fingerprint"]
            with tempfile.TemporaryDirectory(prefix="overfit100_aux_") as tmpdir:
                # ``fetch_pinned`` is path-like-typed (its first act is ``destination.parent.mkdir``).
                # The S2 gate evals passed ``os.path.join(...)`` -- a str -- so every auxiliary row
                # failed with "AttributeError: 'str' object has no attribute 'parent'" BEFORE any
                # download, losing the VAE ceiling for the whole run. Normalize at the boundary; the
                # helper now also normalizes defensively.
                local = fetch_pinned(fingerprint["uri"], fingerprint, Path(tmpdir) / "0.mp4")
                frames = decode_mp4_frames(local)
            if cache is not None:
                cache["episode_id"] = int(episode_id)
                cache["frames"] = frames
        start = int(window_start)
        window = np.asarray(frames[start : start + len(pred_frames)], dtype=np.float32) / 255.0
        if len(window) != len(pred_frames):
            result["aux_status"] = (
                f"source clip has {len(frames)} frames; window [{start}, {start + len(pred_frames)})"
            )
            return result
        result.update(
            {
                "ssim_vs_rgb": _frame_ssim(np.asarray(pred_frames, dtype=np.float32), window),
                "pixel_mse_vs_rgb": float(np.mean((np.asarray(pred_frames, dtype=np.float32) - window) ** 2)),
                "vae_ceiling_ssim": _frame_ssim(np.asarray(gt_frames, dtype=np.float32), window),
                "aux_status": "ok",
            }
        )
    except Exception as exc:  # noqa: BLE001 -- the auxiliary metric is never allowed to abort a run
        result["aux_status"] = f"{type(exc).__name__}: {exc}"[:200]
    return result


def overfit100_metric_row(
    sample: Overfit100EvalSample,
    *,
    checkpoint_step: int,
    seed: int,
    context_mode: str,
    context_source_episode_index,
    rollout_metrics: dict,
    pred_frames: np.ndarray,
    gt_frames: np.ndarray,
    aux: dict | None = None,
) -> dict:
    """One aggregation row, with exactly ``OVERFIT100_ROW_FIELDS`` in that order."""
    pred = np.asarray(pred_frames, dtype=np.float32)
    gt = np.asarray(gt_frames, dtype=np.float32)
    aux = dict(aux or {})
    row = {
        "name": sample.name,
        "episode_id": int(sample.episode_id),
        "episode_index": int(sample.episode_index),
        "window_start": int(sample.window_start),
        "canonical": bool(sample.canonical),
        "checkpoint_step": int(checkpoint_step),
        "seed": int(seed),
        "context_mode": str(context_mode),
        "context_source_episode_index": (
            None if context_source_episode_index is None else int(context_source_episode_index)
        ),
        "ssim": _frame_ssim(pred, gt),
        "latent_mse": float(rollout_metrics["latent_mse"]),
        "latent_mae": float(rollout_metrics["latent_mae"]),
        "pixel_mse": float(np.mean((pred - gt) ** 2)),
        "pixel_mae": float(np.mean(np.abs(pred - gt))),
        "z_pred_std": float(rollout_metrics["z_pred_std"]),
        "z_target_std": float(rollout_metrics["z_target_std"]),
        "z_init_anchor_mse": float(rollout_metrics["z_init_anchor_mse"]),
        "ssim_vs_rgb": aux.get("ssim_vs_rgb"),
        "pixel_mse_vs_rgb": aux.get("pixel_mse_vs_rgb"),
        "vae_ceiling_ssim": aux.get("vae_ceiling_ssim"),
        "aux_status": aux.get("aux_status", "not_requested"),
    }
    return {field: row[field] for field in OVERFIT100_ROW_FIELDS}


def _window_key_from_name(name: str) -> list[int]:
    episode_id, _, start = parse_overfit100_window_name(name)
    return [episode_id, start]


def _window_key_tuple(window: dict) -> tuple[int, int]:
    return (int(window["episode_id"]), int(window["window_start"]))


def assert_flagged_windows_in_cohort(flagged_windows, cohort) -> None:
    """Every flagged window must belong to the FIXED cohort -- checked BEFORE any rollout.

    Scrutiny ruling 8 (SPLIT): membership is checked against the manifest-derived cohort, NOT
    against what this particular pass covered. A collision flagged at build time is a property of
    the cohort, so an intermediate pass covering ten windows must still be able to carry the flag;
    what stays refused is a flag pointing outside the denominator, which would be a bookkeeping
    error the verdict cannot reconcile.
    """
    keys = {tuple(int(v) for v in key) for key in cohort}
    for name in flagged_windows or ():
        if tuple(_window_key_from_name(str(name))) not in keys:
            raise ValueError(
                f"flagged window {name!r} is not in the manifest-derived cohort ({len(keys)} windows); a flag records "
                f"a collision INSIDE the fixed denominator, so it cannot point outside it."
            )


# --------------------------------------------------------------------------------------
# Per-(window, mode, seed) staging: preemption tolerance for the long step-2500 passes.
#
# The segment-final pass is 900 rollouts (~85 min) and the full-set pass 1,629 (~2.4 h), while
# spot uptime has been as short as 34 minutes. Holding every row in memory and writing
# aggregation.json only at the end meant each preemption restarted from zero. Each completed
# rollout is now staged as one small JSON file; a restart admits staged rows ONLY when their
# envelope binds them to this exact run.
#
# The final aggregation path is deliberately untouched: same whole-grid role validation, same
# artifact schema, same immutable writers. Staging is an input-side optimization, never a new
# claim about coverage.
# --------------------------------------------------------------------------------------

OVERFIT100_STAGED_ROW_SCHEMA = "overfit100_eval_staged_row_v1"
OVERFIT100_STAGING_DIR = "staging_rows"
OVERFIT100_RESUME_ENV = "OVERFIT100_EVAL_RESUME"
# The envelope fields that BIND a staged row to one specific run. All four must match exactly or
# the pass fails closed -- a foreign staging directory is an operator problem, not something to
# silently recompute around.
OVERFIT100_STAGING_BINDING = ("checkpoint_step", "pass_role", "manifest_sha256", "code_commit")


def _eval_code_commit() -> str:
    """The eval code's commit, from the env the launcher exports (ONE provenance source).

    Shared with :func:`overfit100_aggregation_artifact`'s ``commit`` field so a staged row and the
    artifact it lands in can never disagree about which code produced them.
    """
    return str(os.environ.get("COMMIT", "") or "unknown")


def resume_state(config) -> tuple[bool, str]:
    """Is per-row staging/resume active for this pass, and why? Returns ``(enabled, reason)``.

    Precedence: the ``OVERFIT100_EVAL_RESUME`` env var (``1``/``0``) overrides the
    ``overfit100_eval_resume`` config key, mirroring how the loss evaluator takes ``SMOKE_LIMIT``
    and ``TRAIN_COMMIT`` from the environment. Default ON.

    **Single-process only, deliberately.** Every host must execute the SAME rollout set: the decode
    inside the loop runs a ``process_allgather``, so if one host skipped a rollout another host
    performed, the collectives would desynchronize and the job would hang. Staging is therefore
    gated on ``jax.process_count() == 1`` (what today's v6e-8 eval jobs are) rather than assuming
    it. The considered alternative -- process 0 writing a ``resume_manifest.json`` snapshot that
    every host then reads -- still needs a barrier between that write and the reads to be race-free,
    which buys nothing for a single-host job while adding a new way to hang a multi-host one.
    Multi-host runs therefore disable staging entirely (no reads AND no writes) and log the reason.
    """
    raw = str(os.environ.get(OVERFIT100_RESUME_ENV, "")).strip()
    if raw:
        if raw not in ("0", "1"):
            raise ValueError(f"{OVERFIT100_RESUME_ENV} must be '1' (resume) or '0' (disable); got {raw!r}")
        want, source = raw == "1", f"env {OVERFIT100_RESUME_ENV}={raw}"
    else:
        want, source = bool(getattr(config, "overfit100_eval_resume", True)), "config overfit100_eval_resume"
    if not want:
        return False, f"row staging disabled by {source}"
    processes = int(jax.process_count())
    if processes != 1:
        return False, (
            f"row staging disabled: multi-host job (process_count={processes}). Every host must roll out the same "
            f"set -- the decode's process_allgather would desynchronize if one host skipped a staged row -- so "
            f"resume is single-process only."
        )
    return True, f"row staging enabled by {source} (single-process job)"


def staged_row_path(step_root: str, context_mode: str, seed: int, name: str) -> str:
    """``<step_root>/staging_rows/<context_mode>/seed_<seed>/<window_name>.json``."""
    return f"{str(step_root).rstrip('/')}/{OVERFIT100_STAGING_DIR}/{context_mode}/seed_{int(seed)}/{name}.json"


def staged_row_envelope(
    row: dict, *, checkpoint_step: int, pass_role: str, manifest_sha256: str, code_commit: str
) -> dict:
    """One staged row plus the envelope binding it to this run (schema tag + the 4 binding fields)."""
    return {
        "schema": OVERFIT100_STAGED_ROW_SCHEMA,
        "checkpoint_step": int(checkpoint_step),
        "pass_role": str(pass_role),
        "manifest_sha256": str(manifest_sha256),
        "code_commit": str(code_commit),
        "row": dict(row),
    }


def write_staged_row(step_root: str, row: dict, *, checkpoint_step, pass_role, manifest_sha256, code_commit) -> str:
    """Persist one completed rollout row.

    Written by process 0 AFTER that row's videos, so an admitted row implies its artifacts are
    already on disk. Not immutability-guarded: re-staging an identical row is a no-op in effect,
    and a crash mid-write is caught by the strict reader on the next start.
    """
    path = staged_row_path(step_root, str(row["context_mode"]), int(row["seed"]), str(row["name"]))
    _write_json(
        path,
        staged_row_envelope(
            row,
            checkpoint_step=checkpoint_step,
            pass_role=pass_role,
            manifest_sha256=manifest_sha256,
            code_commit=code_commit,
        ),
    )
    return path


def _staged_row_error(path: str, detail: str) -> ValueError:
    return ValueError(
        f"overfit100 eval staging: refusing to resume from {path}: {detail}. Staged rows are admitted only when "
        f"their envelope matches this run EXACTLY; a mismatch means the directory holds another run's evidence, so "
        f"clear it deliberately (or set {OVERFIT100_RESUME_ENV}=0) rather than silently recomputing around it."
    )


def read_staged_rows(
    step_root: str, *, checkpoint_step, pass_role, manifest_sha256, code_commit, windows, seeds, modes
) -> dict:
    """Admit staged rows for this run -> ``{(name, mode, seed): row}``; ANY anomaly hard-fails.

    Every file under the staging root is validated: the schema tag, all four binding fields, the
    exact ``OVERFIT100_ROW_FIELDS`` column set, and agreement between the row's identity
    (``name`` / ``context_mode`` / ``seed``, plus the ``episode_id`` / ``window_start`` implied by
    the window name) and the path it was found at. A row outside this pass's coverage is refused
    too: it could not be reconciled with the artifact this pass will write.
    """
    root = f"{str(step_root).rstrip('/')}/{OVERFIT100_STAGING_DIR}"
    files = sorted(tf.io.gfile.glob(f"{root}/*/*/*.json"))
    if not files:
        return {}
    wanted_names = {str(window["name"]) for window in windows}
    wanted_seeds = {int(seed) for seed in seeds}
    wanted_modes = {str(mode) for mode in modes}
    expected = {
        "checkpoint_step": int(checkpoint_step),
        "pass_role": str(pass_role),
        "manifest_sha256": str(manifest_sha256),
        "code_commit": str(code_commit),
    }

    admitted: dict[tuple, dict] = {}
    for path in files:
        _, mode_dir, seed_dir, filename = path.rsplit("/", 3)
        name_from_path = filename[: -len(".json")]
        if not seed_dir.startswith("seed_") or not seed_dir[len("seed_") :].isdigit():
            raise _staged_row_error(path, f"path segment {seed_dir!r} is not seed_<n>")
        seed_from_path = int(seed_dir[len("seed_") :])
        try:
            with tf.io.gfile.GFile(path, "r") as handle:
                payload = json.load(handle)
        except Exception as exc:  # noqa: BLE001 -- unreadable/corrupt staging must fail loudly
            raise _staged_row_error(path, f"unreadable or corrupt JSON ({type(exc).__name__}: {exc})") from exc
        if not isinstance(payload, dict):
            raise _staged_row_error(path, "the staged file is not a JSON object")
        if payload.get("schema") != OVERFIT100_STAGED_ROW_SCHEMA:
            raise _staged_row_error(path, f"schema {payload.get('schema')!r} is not {OVERFIT100_STAGED_ROW_SCHEMA!r}")
        for field, value in expected.items():
            if field not in payload:
                raise _staged_row_error(path, f"envelope field {field} is missing")
            staged = int(payload[field]) if field == "checkpoint_step" else str(payload[field])
            if staged != value:
                raise _staged_row_error(path, f"envelope field {field}={staged!r} does not match this run's {value!r}")
        row = payload.get("row")
        if not isinstance(row, dict):
            raise _staged_row_error(path, "the envelope carries no 'row' object")
        missing = [field for field in OVERFIT100_ROW_FIELDS if field not in row]
        extra = [field for field in row if field not in OVERFIT100_ROW_FIELDS]
        if missing or extra:
            raise _staged_row_error(
                path, f"row columns do not match OVERFIT100_ROW_FIELDS (missing {missing}, extra {extra})"
            )
        if (
            str(row["name"]) != name_from_path
            or str(row["context_mode"]) != mode_dir
            or int(row["seed"]) != seed_from_path
        ):
            raise _staged_row_error(
                path,
                f"row identity (name={row['name']!r}, mode={row['context_mode']!r}, seed={row['seed']}) disagrees "
                f"with its path (name={name_from_path!r}, mode={mode_dir!r}, seed={seed_from_path})",
            )
        episode_id, _, window_start = parse_overfit100_window_name(name_from_path)
        if int(row["episode_id"]) != episode_id or int(row["window_start"]) != window_start:
            raise _staged_row_error(path, "row episode_id / window_start disagree with its window name")
        if int(row["checkpoint_step"]) != int(checkpoint_step):
            raise _staged_row_error(
                path, f"row checkpoint_step={row['checkpoint_step']} is not this run's {int(checkpoint_step)}"
            )
        if name_from_path not in wanted_names or mode_dir not in wanted_modes or seed_from_path not in wanted_seeds:
            raise _staged_row_error(
                path, f"({name_from_path}, {mode_dir}, seed {seed_from_path}) is not part of this pass's coverage"
            )
        key = (name_from_path, mode_dir, seed_from_path)
        if key in admitted:
            raise _staged_row_error(path, f"duplicate staged row for {key}")
        admitted[key] = row
    return admitted


def _sha256_of_file(path: str) -> str:
    """sha256 of a file's exact bytes -- the manifest hash the verdict re-verifies (D1).

    Uses ``tf.io.gfile`` so a ``gs://`` manifest works identically to a local one.
    """
    with tf.io.gfile.GFile(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def overfit100_step_root(output_root: str, checkpoint_step: int, pass_role: str) -> str:
    """The artifact directory for one pass: ``.../step_<step>_<role>`` (D4).

    Keying only on the checkpoint step let the canonical and full-set passes at the SAME step
    overwrite each other's ``aggregation.json``; the role is part of the path so the two tiers'
    inputs are separate, immutable artifacts.
    """
    return f"{str(output_root).rstrip('/')}/step_{int(checkpoint_step):06d}_{str(pass_role)}"


def _refuse_artifact_replacement(path: str, payload: bytes) -> bool:
    """True if ``path`` already holds exactly ``payload`` (skip the write); raise if it differs.

    D4: an eval artifact is immutable evidence. Re-running the same pass (an infra retry) must be
    a no-op, while a DIFFERENT result at the same path means two passes disagree -- that must be
    surfaced, never silently replaced.
    """
    if not tf.io.gfile.exists(path):
        return False
    with tf.io.gfile.GFile(path, "rb") as handle:
        existing = handle.read()
    if existing == payload:
        max_logging.log(f"[wan_overfit100_val] {path} already exists with identical bytes; keeping it")
        return True
    raise ValueError(
        f"refusing to replace the existing eval artifact {path}: its bytes differ from what this pass produced. "
        f"Eval artifacts are immutable evidence -- write to a fresh run_name / validation_output_dir, or delete the "
        f"stale artifact deliberately after recording why."
    )


def _write_json_immutable(path: str, value: dict) -> None:
    """:func:`_write_json` with the D4 immutability guard."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if _refuse_artifact_replacement(path, payload):
        return
    _write_json(path, value)


def _write_text_immutable(path: str, text: str) -> None:
    """Write a text artifact (the CSV) with the D4 immutability guard."""
    payload = text.encode("utf-8")
    if _refuse_artifact_replacement(path, payload):
        return
    parent = os.path.dirname(path)
    if parent:
        tf.io.gfile.makedirs(parent)
    with tf.io.gfile.GFile(path, "w") as handle:
        handle.write(text)


def overfit100_aggregation_artifact(
    config,
    *,
    checkpoint_step: int,
    windows,
    rows,
    seeds,
    modes,
    derangement,
    flagged_windows,
    pass_role: str,
    canonical_cohort,
    all_window_keys,
    manifest_sha256: str,
    role_validation: dict,
) -> dict:
    """The G4 machine-written aggregation artifact, schema v2 (schema in the module docstring).

    Cycle-D strengthening (D1/D2): the artifact carries the pass ROLE, the manifest HASH that
    authenticates it, the FULL manifest-derived ``canonical_cohort`` (100 keys -- never just what
    this pass selected) and the explicit ``covered_canonical_windows`` / ``missing_canonical_windows``
    / ``covered_windows`` sets. The verdict re-derives the cohort from the same manifest and refuses
    any denominator that is not it, so a sparse pass can report partial coverage but can never
    shrink the statistic. ``flagged_windows`` records collision flags against that FIXED cohort;
    they never leave the denominator.
    """
    for index, row in enumerate(rows):
        missing = [field for field in OVERFIT100_ROW_REQUIRED if field not in row]
        if missing:
            raise ValueError(f"aggregation row {index} is missing required column(s) {missing}")
    cohort = tuple(tuple(int(v) for v in key) for key in canonical_cohort)
    all_keys = tuple(tuple(int(v) for v in key) for key in all_window_keys)
    covered = [_window_key_tuple(window) for window in windows]
    covered_canonical = [key for key, window in zip(covered, windows) if window.get("canonical")]
    outside = [list(key) for key in covered_canonical if key not in set(cohort)]
    if outside:
        raise ValueError(
            f"canonical windows {outside[:3]} are not in the manifest-derived cohort ({len(cohort)} windows); the "
            f"selection and the cohort disagree, so the artifact would misreport coverage"
        )
    flagged_keys = []
    for name in flagged_windows or ():
        key = _window_key_from_name(str(name))
        if tuple(key) not in set(cohort):
            raise ValueError(
                f"flagged window {name!r} is not in the manifest-derived cohort ({len(cohort)} windows); a flag "
                f"records a collision inside the fixed denominator and cannot point outside it"
            )
        flagged_keys.append(key)
    return {
        "schema": OVERFIT100_AGGREGATION_SCHEMA,
        "eval_pass_role": str(pass_role),
        "role_validation": dict(role_validation),
        "checkpoint_step": int(checkpoint_step),
        "run_name": str(getattr(config, "run_name", "")),
        "model_type": str(getattr(config, "model_type", "")),
        "commit": _eval_code_commit(),
        "eval_data_dir": str(getattr(config, "eval_data_dir", "")),
        "train_data_dir": str(getattr(config, "train_data_dir", "")),
        "model_manifest_path": str(getattr(config, "model_manifest_path", "")),
        "manifest_sha256": str(manifest_sha256),
        "checkpoint_dir": str(getattr(config, "checkpoint_dir", "")),
        "eval_windows_spec": str(getattr(config, "eval_windows", "")),
        "rollout_seeds": [int(seed) for seed in seeds],
        "context_modes": [str(mode) for mode in modes],
        "context_shuffle_seed": int(getattr(config, "context_shuffle_seed", 0)),
        "context_derangement": ([int(v) for v in derangement] if derangement is not None else None),
        "sampling_steps": int(getattr(config, "side_adapter_sampling_steps", 0)),
        "guide_scale": float(getattr(config, "side_adapter_guide_scale", 1.0)),
        "num_text_slots": int(getattr(config, "num_text_slots", 0) or 0),
        "num_windows": len(list(windows)),
        "windows": [dict(window) for window in windows],
        # D1: the FIXED, manifest-derived denominator plus what this pass actually covered.
        "canonical_cohort": [list(key) for key in cohort],
        "cohort_size": len(cohort),
        "all_windows_size": len(all_keys),
        "covered_windows": [list(key) for key in covered],
        "covered_canonical_windows": [list(key) for key in covered_canonical],
        "missing_canonical_windows": [list(key) for key in cohort if key not in set(covered_canonical)],
        "flagged_windows": flagged_keys,
        "rows": list(rows),
    }


def _overfit100_summary(rows) -> dict:
    """Per-mode means plus AUXILIARY-RGB COVERAGE (D5); the statistic reads ``rows``, not this."""
    summary: dict[str, dict] = {}
    for mode in OVERFIT100_CONTEXT_MODES:
        subset = [row for row in rows if row["context_mode"] == mode]
        if not subset:
            continue
        finite = [row["ssim"] for row in subset if math.isfinite(row["ssim"])]
        summary[mode] = {
            "n_rows": len(subset),
            "mean_ssim": float(np.mean(finite)) if finite else float("nan"),
            "mean_latent_mse": float(np.mean([row["latent_mse"] for row in subset])),
            "mean_pixel_mse": float(np.mean([row["pixel_mse"] for row in subset])),
        }
    ceilings = [row["vae_ceiling_ssim"] for row in rows if row.get("vae_ceiling_ssim") is not None]
    if ceilings:
        summary["vae_ceiling"] = {"n": len(ceilings), "mean_ssim": float(np.mean(ceilings))}
    summary["aux_coverage"] = _aux_coverage(rows)
    return summary


def _aux_coverage(rows) -> dict:
    """Run-level auxiliary-RGB coverage (D5): a systematic failure must be VISIBLE.

    Per-row aux failures are swallowed by design (the auxiliary metric may never abort a rollout),
    which previously meant a worker with no gsutil/ffmpeg produced a complete-looking artifact with
    no VAE ceiling anywhere. Rows that never requested the metric (``eval_aux_rgb=False``) are not
    counted, so ``requested == 0`` is "not asked for", not "all failed".
    """
    requested = [row for row in rows if str(row.get("aux_status", "not_requested")) != "not_requested"]
    ok = [row for row in requested if str(row.get("aux_status")) == "ok"]
    failures: dict[str, int] = {}
    for row in requested:
        status = str(row.get("aux_status"))
        if status != "ok":
            failures[status] = failures.get(status, 0) + 1
    return {
        "requested": len(requested),
        "ok": len(ok),
        "failed": len(requested) - len(ok),
        "coverage_fraction": (len(ok) / len(requested)) if requested else None,
        "failure_reason_counts": dict(sorted(failures.items(), key=lambda item: (-item[1], item[0]))),
    }


def aux_coverage_log_lines(coverage: dict) -> list[str]:
    """The loud lines (D5): WARNING on incomplete auxiliary coverage, ERROR when it is zero.

    Empty when coverage is complete or the metric was never requested -- a silent success and a
    silent "not asked" are both fine; a silent PARTIAL is what this exists to prevent, because the
    auxiliary numbers carry the per-window VAE ceiling the primary SSIM is read against.
    """
    fraction = coverage.get("coverage_fraction")
    if fraction is None or fraction >= 1.0:
        return []
    level = "ERROR" if coverage.get("ok", 0) == 0 else "WARNING"
    lines = [
        f"[wan_overfit100_val] {level}: auxiliary RGB coverage {fraction:.1%} "
        f"({coverage.get('ok', 0)}/{coverage.get('requested', 0)} rows); the per-window VAE ceiling is MISSING for "
        f"{coverage.get('failed', 0)} row(s), so the auxiliary comparison against the true DROID frames is incomplete."
    ]
    for reason, count in (coverage.get("failure_reason_counts") or {}).items():
        lines.append(f"[wan_overfit100_val] {level}:   {count} x {reason}")
    if level == "ERROR":
        lines.append(
            "[wan_overfit100_val] ERROR: NO auxiliary row succeeded -- check gsutil/ffmpeg on the worker and the "
            "manifest's video fingerprints before treating the ceiling as unavailable."
        )
    return lines


def _write_rows_csv(rows, path: str, columns, *, immutable: bool = True) -> None:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    if immutable:
        _write_text_immutable(path, buf.getvalue())
        return
    parent = os.path.dirname(path)
    if parent:
        tf.io.gfile.makedirs(parent)
    with tf.io.gfile.GFile(path, "w") as handle:
        handle.write(buf.getvalue())


def _overfit100_rollout_fn(state_shardings, data_shardings, replicated, scheduler, config):
    """The jit-bound rollout :func:`run_overfit100` drives -- kept as an explicit SEAM.

    Binding jit here rather than inline lets the CPU tests exercise the whole production driver
    with the rollout unjitted: a jit call needs a real sharding tree for the ~5B state, which no
    CPU test can build, and without this seam the driver's wiring (selection -> per-mode context
    -> per-window rng -> decode -> rows -> artifacts) would stay unexercised until TPU time.
    """
    return jax.jit(
        functools.partial(_rollout_overfit100_sample, scheduler=scheduler, config=config),
        in_shardings=(state_shardings, data_shardings, None),
        out_shardings=(replicated, None),
    )


def run_overfit100(config) -> None:
    """The exp_02 evaluation pass: selected windows x context modes x rollout seeds.

    Everything that can be refused cheaply is refused BEFORE the 5B load (D2/D4): SSIM
    availability, the declared pass role against its D11 contract, the flags against the
    manifest-derived cohort, and -- via the immutable writer -- a colliding artifact path.
    """
    assert_ssim_available()  # the statistic's only input -- fail before the 5B load, not after
    pass_role = parse_eval_pass_role(config)
    windows = resolve_eval_windows(config)
    modes = parse_context_modes(getattr(config, "context_modes", "correct"))
    seeds = parse_rollout_seeds(getattr(config, "rollout_seeds", "0"))
    flagged = parse_flagged_windows(getattr(config, "flagged_windows", ""))
    write_videos = bool(getattr(config, "write_videos", False))
    want_aux = bool(getattr(config, "eval_aux_rgb", True))
    manifest_path = str(getattr(config, "model_manifest_path", ""))
    # Aux degradation is CONTAINED (D5) but must not be silent: say it once, in the job log,
    # before the rollouts -- not 90 null rows later when someone reads the artifact.
    aux_warning = aux_prerequisite_warning(want_aux)
    if aux_warning:
        max_logging.log(aux_warning)

    # D1: the cohorts come from the MANIFEST (the set's own episodes for an S2 pass), never from
    # this pass's selection, and the pass must satisfy the role it claims.
    episode_indices = sorted(read_episode_mapping(config.eval_data_dir))
    canonical_cohort = manifest_canonical_cohort(manifest_path, episode_indices=episode_indices)
    all_window_keys = manifest_all_window_keys(manifest_path, episode_indices=episode_indices)
    assert_flagged_windows_in_cohort(flagged, canonical_cohort)  # ruling 8: the FIXED cohort
    role_validation = assert_pass_role_plan(
        pass_role,
        seeds=seeds,
        modes=modes,
        sampling_steps=int(getattr(config, "side_adapter_sampling_steps", 0)),
        windows=windows,
        cohort=canonical_cohort,
        all_window_keys=all_window_keys,
    )
    manifest_sha256 = _sha256_of_file(manifest_path)
    code_commit = _eval_code_commit()
    resume_on, resume_reason = resume_state(config)

    (
        trainer,
        pipeline,
        mesh,
        state,
        state_shardings,
        null_context,
        checkpoint_step,
    ) = _restore_overfit100_validation_state(config)
    scheduler, _ = trainer._create_scheduler()
    replicated = NamedSharding(mesh, P())
    data_shardings = {"z_i0": replicated, "z_video": replicated, "context": replicated}
    p_rollout = _overfit100_rollout_fn(state_shardings, data_shardings, replicated, scheduler, config)

    derangement = None
    if "shuffled" in modes:
        # The ablation's texts ARE the context table's rows, so the derangement is over the
        # table built from the TRAIN set (the same rows the correct mode gathers).
        derangement = value_derangement(
            read_episode_texts(config.train_data_dir, int(config.num_text_slots)),
            seed=int(getattr(config, "context_shuffle_seed", 0)),
        )

    samples = read_overfit100_samples(config, windows)
    output_root = (
        getattr(config, "validation_output_dir", "") or os.path.join(config.output_dir, config.run_name, "validation")
    ).rstrip("/")
    # D4: the ROLE is part of the path, so the canonical and full-set passes at one checkpoint are
    # separate immutable artifacts instead of overwriting each other.
    step_root = overfit100_step_root(output_root, checkpoint_step, pass_role)
    fps = int(getattr(config, "fps", 16))
    if jax.process_index() == 0:
        tf.io.gfile.makedirs(step_root)
        max_logging.log(
            f"[wan_overfit100_val] step={checkpoint_step} role={pass_role} windows={len(samples)} "
            f"modes={list(modes)} seeds={list(seeds)} write_videos={write_videos} aux_rgb={want_aux}"
        )
        max_logging.log(
            f"[wan_overfit100_val] cohort={len(canonical_cohort)} canonical windows (manifest-derived), "
            f"all_windows={len(all_window_keys)}, manifest_sha256={manifest_sha256[:12]}"
        )
        max_logging.log(f"[wan_overfit100_val] {resume_reason}")

    # RESUME (preemption tolerance). The staged rows admitted here are byte-for-byte what an
    # uninterrupted pass would have produced for those tuples: every rollout's rng is
    # window_fold_key(seed, episode_id, window_start), which does not depend on visit order, so
    # skipping a tuple cannot change any OTHER tuple's numbers. Rows are appended in the same
    # loop order whether they were resumed or recomputed, so the artifact is identical too.
    staged_rows: dict[tuple, dict] = {}
    if resume_on:
        staged_rows = read_staged_rows(
            step_root,
            checkpoint_step=checkpoint_step,
            pass_role=pass_role,
            manifest_sha256=manifest_sha256,
            code_commit=code_commit,
            windows=windows,
            seeds=seeds,
            modes=modes,
        )

    rows: list[dict] = []
    aux_cache: dict = {}
    resumed_count = 0
    recomputed_count = 0
    for sample in samples:
        planned = [(mode, seed) for mode in modes for seed in seeds]
        if staged_rows and all((sample.name, mode, seed) in staged_rows for mode, seed in planned):
            # Whole window already done: skip its VAE decode as well as its rollouts.
            rows.extend(staged_rows[(sample.name, mode, seed)] for mode, seed in planned)
            resumed_count += len(planned)
            continue
        batch_base = {
            "z_i0": jnp.asarray(sample.z_i0[None]),
            "z_video": jnp.asarray(sample.z_video[None]),
        }
        gt_latents = batch_base["z_video"].astype(jnp.float32)
        gt_video = pipeline._decode_latents_to_video(pipeline._denormalize_latents(gt_latents))
        gt0 = np.asarray(gt_video[0], dtype=np.float32)
        for mode in modes:
            source_index = context_source_index(mode, sample.episode_index, derangement)
            context = overfit100_context_for_mode(
                state.context_table, null_context, mode, sample.episode_index, derangement
            )
            for seed in seeds:
                staged = staged_rows.get((sample.name, mode, seed))
                if staged is not None:
                    rows.append(staged)
                    resumed_count += 1
                    continue
                batch = {**batch_base, "context": context}
                batch = jax.tree.map(lambda x: jax.device_put(x, replicated), batch)
                rng = window_fold_key(seed, sample.episode_id, sample.window_start)
                with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
                    z_pred, metrics = p_rollout(state, batch, rng)
                    z_pred.block_until_ready()
                pred_video = pipeline._decode_latents_to_video(
                    pipeline._denormalize_latents(z_pred.astype(jnp.float32))
                )
                if jax.process_index() != 0:
                    continue
                pred0 = np.asarray(pred_video[0], dtype=np.float32)
                aux = (
                    overfit100_aux_rgb(manifest_path, sample.episode_id, sample.window_start, pred0, gt0, aux_cache)
                    if want_aux
                    else None
                )
                row = overfit100_metric_row(
                    sample,
                    checkpoint_step=checkpoint_step,
                    seed=seed,
                    context_mode=mode,
                    context_source_episode_index=source_index,
                    rollout_metrics={k: float(jax.device_get(v)) for k, v in metrics.items()},
                    pred_frames=pred0,
                    gt_frames=gt0,
                    aux=aux,
                )
                rows.append(row)
                recomputed_count += 1
                if write_videos:
                    window_dir = f"{step_root}/mode_{mode}/seed_{seed}/{sample.name}"
                    _save_video(gt0, f"{window_dir}/ground_truth.mp4", fps=fps)
                    _save_video(pred0, f"{window_dir}/sample.mp4", fps=fps)
                    _save_video(
                        np.concatenate([gt0, pred0], axis=1),
                        f"{window_dir}/comparison_gt_top_pred_bottom.mp4",
                        fps=fps,
                    )
                    _write_json(f"{window_dir}/metrics.json", row)
                if resume_on:
                    # AFTER the videos: an admitted row implies its artifacts are already on disk.
                    write_staged_row(
                        step_root,
                        row,
                        checkpoint_step=checkpoint_step,
                        pass_role=pass_role,
                        manifest_sha256=manifest_sha256,
                        code_commit=code_commit,
                    )
                max_logging.log(
                    f"[wan_overfit100_val] step={checkpoint_step} {sample.name} mode={mode} seed={seed} "
                    f"ssim={row['ssim']:.4f} latent_mse={row['latent_mse']:.6f} pixel_mse={row['pixel_mse']:.6f}"
                )

    if jax.process_index() != 0:
        return
    artifact = overfit100_aggregation_artifact(
        config,
        checkpoint_step=checkpoint_step,
        windows=windows,
        rows=rows,
        seeds=seeds,
        modes=modes,
        derangement=derangement,
        flagged_windows=flagged,
        pass_role=pass_role,
        canonical_cohort=canonical_cohort,
        all_window_keys=all_window_keys,
        manifest_sha256=manifest_sha256,
        role_validation=role_validation,
    )
    summary = _overfit100_summary(rows)
    _write_json_immutable(f"{step_root}/aggregation.json", artifact)
    _write_rows_csv(rows, f"{step_root}/summary.csv", OVERFIT100_ROW_FIELDS)
    _write_json_immutable(
        f"{step_root}/summary.json",
        {
            "checkpoint_step": int(checkpoint_step),
            "eval_pass_role": pass_role,
            "num_windows": len(samples),
            "num_rows": len(rows),
            "cohort_size": len(canonical_cohort),
            "missing_canonical_windows": artifact["missing_canonical_windows"],
            "per_mode": summary,
            "aggregation_json": f"{step_root}/aggregation.json",
        },
    )
    max_logging.log(f"[wan_overfit100_val] resumed n_rows={resumed_count} recomputed={recomputed_count}")
    for line in aux_coverage_log_lines(summary["aux_coverage"]):  # D5: loud, not silent
        max_logging.log(line)
    max_logging.log(
        f"[wan_overfit100_val] wrote {step_root}/aggregation.json ({len(rows)} rows, role={pass_role}, "
        f"{len(artifact['missing_canonical_windows'])} canonical window(s) not covered by this pass)"
    )


def run(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    if _is_overfit100(config):
        run_overfit100(config)
        return

    trainer, pipeline, mesh, state, state_shardings, checkpoint_step = _restore_validation_state(config)
    scheduler, _ = trainer._create_scheduler()
    replicated = NamedSharding(mesh, P())
    data_shardings = {"z_i0": replicated, "z_video": replicated, "actions": replicated}
    p_rollout = jax.jit(
        functools.partial(_rollout_sample, scheduler=scheduler, config=config),
        in_shardings=(state_shardings, data_shardings, None),
        out_shardings=(replicated, None),
    )

    samples = _read_eval_samples(
        config,
        count=int(getattr(config, "num_eval_videos", 1)),
        start_index=int(getattr(config, "validation_start_index", 0)),
    )
    output_root = getattr(config, "validation_output_dir", "") or os.path.join(
        config.output_dir,
        config.run_name,
        "validation",
    )
    output_root = output_root.rstrip("/")
    step_root = f"{output_root}/step_{checkpoint_step:06d}"
    fps = int(getattr(config, "fps", 16))
    seed = int(getattr(config, "validation_seed", getattr(config, "seed", 0)))

    if jax.process_index() == 0:
        tf.io.gfile.makedirs(step_root)
        _write_json(
            f"{step_root}/config.json", _validation_config_artifact(config, checkpoint_step, len(samples), seed)
        )

    rows = []
    rng = jax.random.key(seed)
    for idx, sample in enumerate(samples):
        rng, sample_rng = jax.random.split(rng)
        batch = _as_batch(sample)
        batch = jax.tree.map(lambda x: jax.device_put(x, replicated), batch)
        with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
            z_pred, metrics = p_rollout(state, batch, sample_rng)
            z_pred.block_until_ready()

        gt_latents = batch["z_video"].astype(jnp.float32)
        pred_video = pipeline._decode_latents_to_video(pipeline._denormalize_latents(z_pred.astype(jnp.float32)))
        gt_video = pipeline._decode_latents_to_video(pipeline._denormalize_latents(gt_latents))

        if jax.process_index() != 0:
            continue

        pred0 = np.asarray(pred_video[0], dtype=np.float32)
        gt0 = np.asarray(gt_video[0], dtype=np.float32)
        sample_dir = f"{step_root}/sample_{idx:04d}_{sample.name.replace('/', '_')}"
        comparison = np.concatenate([gt0, pred0], axis=1)
        _save_video(gt0, f"{sample_dir}/ground_truth.mp4", fps=fps)
        _save_video(pred0, f"{sample_dir}/sample.mp4", fps=fps)
        _save_video(comparison, f"{sample_dir}/comparison_gt_top_pred_bottom.mp4", fps=fps)

        metric_values = {k: float(jax.device_get(v)) for k, v in metrics.items()}
        metric_values.update(
            {
                "checkpoint_step": int(checkpoint_step),
                "sample_index": int(idx),
                "name": sample.name,
                "ordinal": int(sample.ordinal),
                "pixel_mse": float(np.mean((pred0 - gt0) ** 2)),
                "pixel_mae": float(np.mean(np.abs(pred0 - gt0))),
                "ssim_avg": _frame_ssim(pred0, gt0),
                "height": int(pred0.shape[1]),
                "width": int(pred0.shape[2]),
                "frames": int(pred0.shape[0]),
            }
        )
        _write_json(f"{sample_dir}/metrics.json", metric_values)
        if sample.meta:
            _write_json(f"{sample_dir}/meta.json", sample.meta)
        rows.append(metric_values)
        max_logging.log(
            "[wan_side_adapter_val] "
            f"step={checkpoint_step} sample={idx} name={sample.name} "
            f"latent_mse={metric_values['latent_mse']:.6f} "
            f"pixel_mse={metric_values['pixel_mse']:.6f} "
            f"ssim={metric_values['ssim_avg']:.4f}"
        )

    if jax.process_index() == 0 and rows:
        summary_path = f"{step_root}/summary.csv"
        keys = list(rows[0].keys())
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            tmp = f.name
        try:
            _copy_local_to_output(tmp, summary_path)
        finally:
            os.remove(tmp)
        mean_latent = float(np.mean([r["latent_mse"] for r in rows]))
        mean_pixel = float(np.mean([r["pixel_mse"] for r in rows]))
        finite_ssim = [r["ssim_avg"] for r in rows if math.isfinite(r["ssim_avg"])]
        aggregate = {
            "checkpoint_step": int(checkpoint_step),
            "num_samples": len(rows),
            "mean_latent_mse": mean_latent,
            "mean_pixel_mse": mean_pixel,
            "mean_ssim": float(np.mean(finite_ssim)) if finite_ssim else float("nan"),
            "summary_csv": summary_path,
        }
        _write_json(f"{step_root}/summary.json", aggregate)
        max_logging.log(f"[wan_side_adapter_val] wrote {summary_path}")
        max_logging.log(f"[wan_side_adapter_val] aggregate: {aggregate}")


def main(argv: Sequence[str]) -> None:
    run(argv)


if __name__ == "__main__":
    app.run(main)
