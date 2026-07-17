"""Visual validation for WAN TI2V side-adapter checkpoints.

This is the MaxDiffusion equivalent of ``../Wan2.2/eval_adaptor.py`` for the
cached DROID TFRecords.  The converted records contain cached latents and
actions, but not raw source frames, so validation compares generated videos to
the VAE decode of ``z_video`` and logs latent/pixel metrics.
"""

from __future__ import annotations

import csv
import functools
import json
import math
import os
import tempfile
from dataclasses import dataclass
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
from maxdiffusion.trainers.wan_ti2v_full_ft_trainer import (
    FullFTTrainState,
    WanTI2VFullFTTrainer,
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


def _restore_checkpoint_state(config, state, ckpt_dir, *, cohort_mode=False):
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
    """
    requested = int(getattr(config, "checkpoint_step", -1))
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


def run(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config

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
