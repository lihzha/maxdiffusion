"""Normalisation stats + a data sanity check for the ``cam_action*`` modes.

Samples training windows exactly as the trainer does (same dataset class, same
windowing, ``load_cam_pose=True``), re-anchors them with the same
``camera_frame_actions`` code, and writes the per-camera p01/p99 of the anchored
EEF position to ``cam_action_stats_path``. Run once per dataset, with the same
yml and window overrides as the training run (the anchored distribution depends
on the history length).

It also checks the ``ee_pose_cam*`` features against the geometry they are
supposed to encode, which is the quickest way to validate a new shard build:

* exterior cameras are static, so the implied base-to-camera transform
  ``P(t) @ inv(T_ee(t))`` must be the same at every slot of a window;
* the wrist camera is rigidly mounted, so its per-frame pose ``P(t)`` must be
  constant within a window;
* every stored rotation must be orthonormal.

A large deviation means the writer used a different Euler convention or
cartesian rows than the ones stored in the shard (see ``camera_frame_actions``).

Usage (WAN)::

    JAX_PLATFORMS=cpu python src/maxdiffusion/compute_cam_action_stats.py \\
        src/maxdiffusion/configs/base_wan_ctrl_world.yml \\
        train_data_dir=<.../train> action_stats_path=<.../stats.json> \\
        cam_action_stats_path=<.../cam_action_stats.json> \\
        num_history_latent_frames=7 num_predicted_latents=5

Usage (SVD)::

    JAX_PLATFORMS=cpu python src/maxdiffusion/compute_cam_action_stats.py \\
        src/maxdiffusion/configs/base_ctrl_world.yml \\
        train_data_dir=<.../train> stats_path=<.../stats.json> \\
        cam_action_stats_path=<.../cam_action_stats.json> num_history=7 num_frames=5
"""

from __future__ import annotations

import json
from typing import Sequence

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.input_pipeline.robot.camera_frame_actions import (
    NUM_VIEWS,
    _compose,
    _invert,
    anchored_camera_poses,
    pose_from_cartesian,
    wan_anchor_and_valid,
)

_BATCH = 16


def _windows(config):
    """Yield ``(cam_pose, ee_cartesian, anchor, valid)`` numpy batches of training windows."""
    if config.model_name == "ctrl_world":
        from maxdiffusion.input_pipeline.robot.ctrl_world_droid_dataset import (
            CtrlWorldDroidLatentDataset,
        )
        ds = CtrlWorldDroidLatentDataset(
            data_dir=config.train_data_dir,
            stats_path=config.stats_path,
            num_history=config.num_history,
            num_frames=config.num_frames,
            action_dim=config.action_dim,
            text_embed_dim=config.text_embed_dim,
            batch_size=_BATCH,
            split="train",
            seed=config.seed,
            down_sample=config.ctrl_world_down_sample,
            max_skip=config.ctrl_world_max_skip,
            max_skip_his=config.ctrl_world_max_skip_his,
            skip_his_zero_prob=config.ctrl_world_skip_his_zero_prob,
            shuffle=True,
            shuffle_buffer=config.ctrl_world_shuffle_buffer,
            shard_for_training=False,
            load_cam_pose=True,
        )
        for batch in ds:
            b, t = batch["ee_cartesian"].shape[:2]
            yield (batch["cam_pose"], batch["ee_cartesian"],
                   np.full((b,), config.num_history, np.int32), np.ones((b, t), bool))
    else:
        from maxdiffusion.input_pipeline.robot.wan_ctrl_world_dataset import (
            WanCtrlWorldDroidDataset,
        )
        n_hist = config.num_history_latent_frames
        ds = WanCtrlWorldDroidDataset(
            data_dir=config.train_data_dir,
            stats_path=config.action_stats_path,
            n_hist=n_hist,
            max_latent_frames=n_hist + config.num_predicted_latents,
            action_dim=config.action_dim,
            batch_size=_BATCH,
            split="train",
            seed=config.seed,
            shuffle=True,
            shard_for_training=False,
            load_cam_pose=True,
            repeat=True,
        )
        for batch in ds:
            anchor, valid = wan_anchor_and_valid(jnp.asarray(batch["frame_positions"]), n_hist)
            yield batch["cam_pose"], batch["ee_cartesian"], np.asarray(anchor), np.asarray(valid)


def _consistency(cam_pose, ee_cartesian, anchor, valid) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-window max deviation of the invariants the data must satisfy.

    Returns ``(ext_dev (B, V-1), wrist_dev (B,), ortho_err)``.
    """
    b, s, v, _ = cam_pose.shape
    pose = jnp.asarray(cam_pose, jnp.float32).reshape(b, s, v, 3, 4)
    t_ee = pose_from_cartesian(jnp.asarray(ee_cartesian, jnp.float32))
    extrinsic = _compose(pose, _invert(t_ee)[:, :, None])               # (B, S, V, 3, 4)
    idx = jnp.asarray(anchor)[:, None, None, None, None]
    ext_a = jnp.take_along_axis(extrinsic, idx, axis=1)
    pose_a = jnp.take_along_axis(pose, idx, axis=1)
    mask = jnp.asarray(valid)[:, :, None]
    ext_dev = jnp.max(jnp.abs(extrinsic - ext_a).max(axis=(-1, -2)) * mask, axis=1)   # (B, V)
    wrist_dev = jnp.max(jnp.abs(pose - pose_a)[:, :, 0].max(axis=(-1, -2)) * mask[..., 0], axis=1)
    rot = pose[..., :3]
    gram = jnp.einsum("...ji,...jk->...ik", rot, rot) - jnp.eye(3)
    ortho = jnp.max(jnp.abs(gram).max(axis=(-1, -2)) * mask)
    return np.asarray(ext_dev[:, 1:]), np.asarray(wrist_dev), float(ortho)


def summarize_windows(windows, n_windows: int) -> dict:
    """Percentiles + consistency checks over the first ``n_windows`` windows.

    ``windows`` yields what ``_windows`` does. Returns ``xyz_01`` / ``xyz_99``
    ``(NUM_VIEWS, 3)``, ``num_windows``, and the per-window drifts
    ``ext_drift`` ``(N, NUM_VIEWS - 1)``, ``wrist_drift`` ``(N,)`` plus the max
    rotation ``ortho`` error.
    """
    xyz = [[] for _ in range(NUM_VIEWS)]
    ext_devs, wrist_devs, ortho = [], [], 0.0
    seen = 0
    for cam_pose, ee_cartesian, anchor, valid in windows:
        anchored = np.asarray(anchored_camera_poses(
            jnp.asarray(cam_pose), jnp.asarray(ee_cartesian), jnp.asarray(anchor)
        ))                                                              # (B, S, V, 3, 4)
        keep = np.asarray(valid, bool)
        for v in range(NUM_VIEWS):
            xyz[v].append(anchored[:, :, v, :, 3][keep])
        e, w, o = _consistency(cam_pose, ee_cartesian, anchor, valid)
        ext_devs.append(e)
        wrist_devs.append(w)
        ortho = max(ortho, o)
        seen += anchored.shape[0]
        if seen >= n_windows:
            break
    if not seen:
        raise ValueError("no training windows found")
    return {
        "xyz_01": np.stack([np.percentile(np.concatenate(x), 1, axis=0) for x in xyz]).astype(np.float32),
        "xyz_99": np.stack([np.percentile(np.concatenate(x), 99, axis=0) for x in xyz]).astype(np.float32),
        "num_windows": int(seen),
        "ext_drift": np.concatenate(ext_devs),
        "wrist_drift": np.concatenate(wrist_devs),
        "ortho": ortho,
    }


def run(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    out_path = max_utils.config_get(config, "cam_action_stats_path", "")
    if not out_path:
        raise ValueError("set cam_action_stats_path=<output json>")
    summary = summarize_windows(
        _windows(config), int(max_utils.config_get(config, "cam_action_stats_windows", 4000))
    )
    lo, hi, seen = summary["xyz_01"], summary["xyz_99"], summary["num_windows"]
    ext_devs, wrist_devs, ortho = summary["ext_drift"], summary["wrist_drift"], summary["ortho"]

    max_logging.log(f"[cam_action_stats] {seen} windows from {config.train_data_dir}")
    for v, name in enumerate(("wrist", "ext1", "ext2")):
        max_logging.log(
            f"  cam{v} ({name}) anchored xyz p01={np.round(lo[v], 3).tolist()} "
            f"p99={np.round(hi[v], 3).tolist()} (m)"
        )
    max_logging.log(
        "  consistency (should all be ~1e-5; large values mean the ee_pose_cam* writer "
        "disagrees with the stored cartesian):"
    )
    for v in range(NUM_VIEWS - 1):
        max_logging.log(
            f"    cam{v + 1} implied extrinsic drift within a window: "
            f"median={np.median(ext_devs[:, v]):.2e} max={ext_devs[:, v].max():.2e}"
        )
    max_logging.log(
        f"    cam0 (wrist) per-frame pose drift within a window: "
        f"median={np.median(wrist_devs):.2e} max={wrist_devs.max():.2e}"
    )
    max_logging.log(f"    rotation orthonormality error: max={ortho:.2e}")

    with tf.io.gfile.GFile(out_path, "w") as f:
        json.dump(
            {
                "xyz_01": lo.tolist(),
                "xyz_99": hi.tolist(),
                "num_windows": int(seen),
                "source": config.train_data_dir,
            },
            f,
            indent=2,
        )
    max_logging.log(f"[cam_action_stats] wrote {out_path}")


if __name__ == "__main__":
    app.run(run)
