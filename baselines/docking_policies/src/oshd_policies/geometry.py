"""Platform-target clearance: the platform's body as a box, the target's CAD as surface points."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

#: Floating platform body in its base frame (m): the deck box of
#: ``floating_platform.usdc`` (x, y within +-0.2, z 0.05..0.705) extended forward and up to
#: the sensor head (mount at x = 0.21, z = 0.76).
PLATFORM_BOX = (np.array([-0.20, -0.205, 0.05]), np.array([0.25, 0.195, 0.80]))


def surface_points(mesh, n: int = 4000, seed: int = 0) -> np.ndarray:
    pts, _ = __import__("trimesh").sample.sample_surface(mesh, n, seed=seed)
    return np.concatenate([np.asarray(pts), np.asarray(mesh.vertices)[:: max(1, len(mesh.vertices) // 2000)]])


def clearance(robot_pose: np.ndarray, target_pose: np.ndarray, pts_target: np.ndarray) -> np.ndarray:
    """Per row: smallest distance (m) from the target's points to the platform box; <= 0 is
    contact (a point inside the box). ``*_pose``: (N, 7) xyz + quat xyzw in the same frame."""
    lo, hi = PLATFORM_BOX
    out = np.empty(len(robot_pose))
    for i, (rp, tp) in enumerate(zip(robot_pose, target_pose)):
        p_w = R.from_quat(tp[3:]).apply(pts_target) + tp[:3]
        p_b = R.from_quat(rp[3:]).inv().apply(p_w - rp[:3])
        d_out = np.linalg.norm(np.maximum(np.maximum(lo - p_b, p_b - hi), 0.0), axis=1)
        d_in = np.minimum(p_b - lo, hi - p_b).min(1)
        inside = (p_b >= lo).all(1) & (p_b <= hi).all(1)
        out[i] = np.where(inside, -d_in, d_out).min()
    return out
