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


def _read_eval_samples(config, count: int, start_index: int) -> list[EvalSample]:
    feature_description = {
        "name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "ordinal": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "z_i0": tf.io.FixedLenFeature([], tf.string),
        "z_video": tf.io.FixedLenFeature([], tf.string),
        "actions": tf.io.FixedLenFeature([], tf.string),
        "meta_json": tf.io.FixedLenFeature([], tf.string, default_value=b"{}"),
    }
    c = int(config.latent_channels)
    f = int(config.latent_frames)
    h = int(config.latent_height)
    w = int(config.latent_width)
    action_len = int(config.action_len)
    action_dim = int(config.action_dim)

    ds = tf.data.TFRecordDataset(_tfrecord_files(config.eval_data_dir))
    ds = ds.map(lambda raw: tf.io.parse_single_example(raw, feature_description))
    ds = ds.skip(max(0, int(start_index))).take(max(1, int(count)))

    samples: list[EvalSample] = []
    for fallback_idx, raw in enumerate(ds.as_numpy_iterator()):
        name = raw["name"].decode("utf-8") or f"sample_{start_index + fallback_idx:06d}"
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


def _rollout_sample(state: TrainState, data: dict, rng: jax.Array, scheduler, config):
    weights_dtype = _dtype(config.weights_dtype)
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


def _restore_validation_state(config):
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

    ckpt_dir = config.checkpoint_dir or os.path.join(config.output_dir, config.run_name, "checkpoints")
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
    requested = int(getattr(config, "checkpoint_step", -1))
    step = requested if requested > 0 else manager.latest_step()
    if step is None:
        raise ValueError(f"No adapter checkpoints found in {ckpt_dir}")
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
    max_logging.log(f"[wan_side_adapter_val] restored step={restored_step} from {ckpt_dir}")
    return trainer, pipeline, mesh, state, state_shardings, restored_step


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
            f"{step_root}/config.json",
            {
                "run_name": config.run_name,
                "checkpoint_dir": config.checkpoint_dir,
                "checkpoint_step": checkpoint_step,
                "eval_data_dir": config.eval_data_dir,
                "num_eval_videos": len(samples),
                "validation_start_index": int(getattr(config, "validation_start_index", 0)),
                "seed": seed,
                "side_adapter_sampling_steps": int(config.side_adapter_sampling_steps),
                "side_adapter_guide_scale": float(config.side_adapter_guide_scale),
                "action_adapter_type": getattr(config, "action_adapter_type", "side_adapter"),
            },
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
