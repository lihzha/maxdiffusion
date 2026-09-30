"""Camera-frame robot actions, shared by the WAN and SVD ``cam_action*`` modes.

The vector-action modes condition on the raw DROID state — end-effector pose in
the ROBOT BASE frame plus gripper. DROID's exterior cameras sit somewhere
different in every scene, so the same base-frame motion lands on very different
image-space motion from episode to episode, and the model has to infer the
camera pose from pixels before an action means anything. The ``cam_action``
modes instead express the action in each camera's own frame, once per camera,
and hand camera ``c``'s action only to camera ``c``'s region of the 3-view
latent grid (see ``per-camera broadcast`` in the trainers).

Representation
--------------
For raw slot ``t`` and camera ``c``, with ``t0`` the window's ANCHOR slot::

    A_c(t; t0) = E_c(t0) @ T_ee(t)

where ``T_ee(t)`` is the EEF pose in the base frame and ``E_c(t0)`` the
base-to-camera transform of camera ``c`` at the anchor. The anchor is the most
recent frame the model sees clean: the last real slot of the last history
latent frame (WAN), or the conditioning frame ``frame_now`` (SVD).

* Exterior cameras are static, so ``E_c(t0) = E_c`` and this is simply the EEF
  pose in that camera's frame.
* The wrist camera is rigidly mounted on the hand, so its own per-frame frame
  would make the pose a constant (only the gripper would vary). Anchoring
  instead gives the gripper trajectory relative to where the wrist camera is
  *now*: the camera's ego-motion over the window.

Per (slot, camera) the model sees ``CAM_ACTION_DIM = 10`` numbers::

    [x, y, z]            anchored position, per-camera percentile-normalised
    [r_00, r_10, r_20,
     r_01, r_11, r_21]   first two columns of the anchored rotation (6D
                         representation; continuous, unlike DROID's Euler roll,
                         which wraps at +-pi)
    [gripper]            the existing normalised gripper value

Data contract (writer side, run where raw DROID lives)
------------------------------------------------------
Each TFRecord example gains three features, one per camera, in the SAME camera
order as ``latent_cam0/1/2`` (0 = wrist, 1 = ext1, 2 = ext2)::

    ee_pose_cam{i}: tf.io.serialize_tensor(float32 (T, 3, 4))

Row ``t`` is the top three rows of ``E_i(t) @ T_ee(t)`` — the EEF pose in camera
``i``'s frame AT THAT SAME FRAME ``t`` — produced by :func:`ee_pose_in_camera`:

* ``E_i(t)``: 4x4 base-to-camera (world-to-camera) transform, OpenCV axes
  (+x right, +y down, +z forward): exactly the ``extrinsic`` OSCAR's
  ``project()`` takes in the skeleton pass, after any
  ``skeleton_extrinsics_path`` override.
* ``T_ee(t)``: built from the SAME ``observation/robot_state/
  cartesian_position`` rows that are stored in ``action`` (WAN shards, after
  ``[::rgb_skip]``) or ``cartesian`` (SVD shards, 15 Hz). ``T`` must equal that
  feature's row count. Use :func:`ee_pose_in_camera` rather than re-deriving
  ``T_ee``: the reader re-anchors with this module's Euler convention, and the
  anchor term ``P(t0) @ inv(T_ee(t0))`` only cancels if both sides agree.

Why per-frame poses and not anchored ones
-----------------------------------------
Training windows start at a random ``frame_now`` and an autoregressive rollout
re-anchors at every chunk, so no single anchor can be baked in at write time.
The per-frame pose is anchor-free, and together with the base-frame cartesian
the shards already carry it determines the anchored pose for ANY anchor::

    E_c(t0)  = P_c(t0) @ inv(T_ee(t0))          (implied extrinsic at the anchor)
    A_c(t)   = E_c(t0) @ T_ee(t)

That composition runs in :func:`camera_action_features`, in JAX, so training,
in-training eval and rollout inference all share one implementation.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

NUM_VIEWS = 3
CAM_ACTION_DIM = 10
EE_POSE_KEYS = tuple(f"ee_pose_cam{i}" for i in range(NUM_VIEWS))
# Raw slots per WAN latent frame (4x temporal VAE compression).
WAN_ACTIONS_PER_LATENT = 4


# ── Writer side (numpy) ───────────────────────────────────────────────────────


def euler_xyz_to_matrix_np(rpy: np.ndarray) -> np.ndarray:
    """DROID's Euler convention: ``scipy Rotation.from_euler("xyz", rpy)``.

    Extrinsic x-y-z, i.e. ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``. ``(..., 3)`` ->
    ``(..., 3, 3)``. Written out so the writer needs no scipy and the formula is
    visibly the one :func:`euler_xyz_to_matrix` uses.
    """
    rpy = np.asarray(rpy, dtype=np.float64)
    cx, sx = np.cos(rpy[..., 0]), np.sin(rpy[..., 0])
    cy, sy = np.cos(rpy[..., 1]), np.sin(rpy[..., 1])
    cz, sz = np.cos(rpy[..., 2]), np.sin(rpy[..., 2])
    rows = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def cartesian_to_pose_np(cartesian: np.ndarray) -> np.ndarray:
    """``(T, 6)`` DROID ``cartesian_position`` -> ``(T, 4, 4)`` base-frame EEF pose."""
    cartesian = np.asarray(cartesian, dtype=np.float64)
    pose = np.zeros(cartesian.shape[:-1] + (4, 4), dtype=np.float64)
    pose[..., :3, :3] = euler_xyz_to_matrix_np(cartesian[..., 3:6])
    pose[..., :3, 3] = cartesian[..., :3]
    pose[..., 3, 3] = 1.0
    return pose


def ee_pose_in_camera(cartesian: np.ndarray, base_to_camera: np.ndarray) -> np.ndarray:
    """The ``ee_pose_cam{i}`` feature: EEF pose in camera ``i``'s frame, per frame.

    Args:
        cartesian:      ``(T, 6)`` raw ``cartesian_position`` rows, the same rows
                        (and the same subsampling) as the stored action/cartesian.
        base_to_camera: ``(4, 4)`` for a static camera or ``(T, 4, 4)`` per frame
                        (the wrist): the base-to-camera transform the skeleton
                        renderer projects with.

    Returns:
        ``(T, 3, 4)`` float32, ready for ``tf.io.serialize_tensor``.
    """
    base_to_camera = np.asarray(base_to_camera, dtype=np.float64)
    pose = base_to_camera @ cartesian_to_pose_np(cartesian)
    return pose[..., :3, :].astype(np.float32)


# ── Reader side (JAX) ─────────────────────────────────────────────────────────


def euler_xyz_to_matrix(rpy):
    """JAX twin of :func:`euler_xyz_to_matrix_np`. ``(..., 3)`` -> ``(..., 3, 3)``."""
    cx, sx = jnp.cos(rpy[..., 0]), jnp.sin(rpy[..., 0])
    cy, sy = jnp.cos(rpy[..., 1]), jnp.sin(rpy[..., 1])
    cz, sz = jnp.cos(rpy[..., 2]), jnp.sin(rpy[..., 2])
    rows = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    return jnp.stack([jnp.stack(r, axis=-1) for r in rows], axis=-2)


def pose_from_cartesian(cartesian):
    """``(..., 6)`` base-frame cartesian -> ``(..., 3, 4)`` rigid transform."""
    rot = euler_xyz_to_matrix(cartesian[..., 3:6])
    return jnp.concatenate([rot, cartesian[..., :3, None]], axis=-1)


def _compose(a, b):
    """``a @ b`` for ``(..., 3, 4)`` rigid transforms (implicit last row 0 0 0 1)."""
    rot = jnp.einsum("...ij,...jk->...ik", a[..., :3], b[..., :3])
    trans = jnp.einsum("...ij,...j->...i", a[..., :3], b[..., 3]) + a[..., 3]
    return jnp.concatenate([rot, trans[..., None]], axis=-1)


def _invert(a):
    """Inverse of a ``(..., 3, 4)`` rigid transform."""
    rot_t = jnp.swapaxes(a[..., :3], -1, -2)
    trans = -jnp.einsum("...ij,...j->...i", rot_t, a[..., 3])
    return jnp.concatenate([rot_t, trans[..., None]], axis=-1)


def anchored_camera_poses(cam_pose, ee_cartesian, anchor_index):
    """Re-express every slot's EEF pose in each camera's frame at the anchor.

    Args:
        cam_pose:     ``(B, S, V, 12)`` — per-slot ``ee_pose_cam*`` (flattened
                      row-major 3x4), as the datasets emit it.
        ee_cartesian: ``(B, S, 6)`` raw base-frame cartesian for the same slots.
        anchor_index: ``(B,)`` int — the slot whose camera pose defines each
                      camera's frame. Must be a real (non-padded) slot.

    Returns:
        ``(B, S, V, 3, 4)`` anchored poses ``A_c(t; t0) = E_c(t0) @ T_ee(t)``.
    """
    b, s, v, _ = cam_pose.shape
    pose = cam_pose.astype(jnp.float32).reshape(b, s, v, 3, 4)
    t_ee = pose_from_cartesian(ee_cartesian.astype(jnp.float32))        # (B, S, 3, 4)
    idx = anchor_index.astype(jnp.int32)
    pose_a = jnp.take_along_axis(pose, idx[:, None, None, None, None], axis=1)[:, 0]
    t_ee_a = jnp.take_along_axis(t_ee, idx[:, None, None, None], axis=1)[:, 0]
    base_to_cam_a = _compose(pose_a, _invert(t_ee_a)[:, None])           # (B, V, 3, 4)
    return _compose(base_to_cam_a[:, None], t_ee[:, :, None])            # (B, S, V, 3, 4)


def camera_action_features(
    cam_pose,
    ee_cartesian,
    gripper,
    anchor_index,
    xyz_lo,
    xyz_hi,
    valid=None,
):
    """Anchored camera-frame action features, ``(B, S, V, CAM_ACTION_DIM)``.

    Args:
        cam_pose, ee_cartesian, anchor_index: see :func:`anchored_camera_poses`.
        gripper:  ``(B, S)`` normalised gripper (dim 6 of the existing action).
        xyz_lo, xyz_hi: ``(V, 3)`` per-camera p01/p99 of the anchored position,
                  from :func:`load_cam_action_stats`.
        valid:    optional ``(B, S)`` mask; masked slots are zeroed, matching the
                  zero padding the vector-action layout already uses for them.
    """
    anchored = anchored_camera_poses(cam_pose, ee_cartesian, anchor_index)
    lo = jnp.asarray(xyz_lo, dtype=jnp.float32)
    hi = jnp.asarray(xyz_hi, dtype=jnp.float32)
    xyz = 2.0 * (anchored[..., 3] - lo) / (hi - lo + 1e-8) - 1.0
    xyz = jnp.clip(xyz, -1.0, 1.0)
    rot6d = jnp.concatenate([anchored[..., :3, 0], anchored[..., :3, 1]], axis=-1)
    grip = jnp.broadcast_to(
        gripper.astype(jnp.float32)[:, :, None, None], xyz.shape[:-1] + (1,)
    )
    feats = jnp.concatenate([xyz, rot6d, grip], axis=-1)
    if valid is not None:
        feats = feats * valid.astype(feats.dtype)[:, :, None, None]
    return feats


def wan_anchor_and_valid(frame_positions, n_hist: int):
    """Anchor slot and real-slot mask for the WAN 4-slots-per-latent layout.

    Latent frame ``k`` owns slots ``4k..4k+3``. Episode frame 0 encodes a single
    raw frame, so its slots are ``[a0, 0, 0, 0]``; every other latent frame owns
    four real actions. ``frame_positions`` is 0 exactly for window frames that
    gathered episode frame 0 (the dataset clamps the RoPE index at 0 only there),
    so it identifies the padded slots without a separate mask feature.

    The anchor is the newest real slot of the last history latent frame: slot
    ``4(n_hist-1) + 3``, or ``4(n_hist-1)`` when that frame is episode frame 0.

    Returns ``(anchor_index (B,), valid (B, 4W))``.
    """
    if n_hist < 1:
        raise ValueError("camera-frame actions need at least one history latent frame")
    b, w = frame_positions.shape
    real = frame_positions > 0                                          # (B, W)
    first_slot = jnp.arange(WAN_ACTIONS_PER_LATENT) == 0
    valid = (first_slot[None, None, :] | real[:, :, None]).reshape(
        b, w * WAN_ACTIONS_PER_LATENT
    )
    last = n_hist - 1
    anchor = WAN_ACTIONS_PER_LATENT * last + jnp.where(
        real[:, last], WAN_ACTIONS_PER_LATENT - 1, 0
    )
    return anchor, valid


# ── Normalisation stats ───────────────────────────────────────────────────────


def load_cam_action_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(xyz_lo, xyz_hi)``, each ``(NUM_VIEWS, 3)``, from ``compute_cam_action_stats.py``."""
    if not path:
        raise ValueError(
            "cam_action modes need cam_action_stats_path: the per-camera p01/p99 of "
            "the anchored EEF position. Generate it once per dataset with "
            "src/maxdiffusion/compute_cam_action_stats.py."
        )
    import tensorflow as tf

    with tf.io.gfile.GFile(path, "r") as f:
        stats = json.load(f)
    lo = np.asarray(stats["xyz_01"], dtype=np.float32)
    hi = np.asarray(stats["xyz_99"], dtype=np.float32)
    if lo.shape != (NUM_VIEWS, 3) or hi.shape != (NUM_VIEWS, 3):
        raise ValueError(
            f"{path}: xyz_01/xyz_99 must be ({NUM_VIEWS}, 3), got {lo.shape}/{hi.shape}"
        )
    return lo, hi
