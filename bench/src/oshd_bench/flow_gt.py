"""Ground-truth optical flow on the target, rendered from the CAD model at two mocap poses.

The mesh is represented by a dense, area-uniform surface sample. Points are splatted into
the camera at ``t0`` with a z-buffer, so each pixel keeps the front-most surface point
``p`` (model frame). The flow at that pixel is ``proj(T1 p) - proj(T0 p)``. Only pixels
hit by the z-buffer and inside the rendered silhouette are valid, so the evaluation is
restricted to the target region (the background has no ground truth: its depth is unknown
to the mocap).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from oshd_dataset import CameraIntrinsics, ObjectModel, Trajectory
from oshd_dataset.geometry import transform_points
from oshd_dataset.labels import render_mask

from .gt import pose_in_sensor

__all__ = ["surface_points", "render_surface", "target_flow", "target_flow_at"]


@lru_cache(maxsize=8)
def _surface_points_cached(obj: ObjectModel, n: int, seed: int) -> np.ndarray:
    v, f = obj.vertices, obj.faces
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = 1 - s, s * (1 - r2), s * r2
    t = tri[pick]
    return a[:, None] * t[:, 0] + b[:, None] * t[:, 1] + c[:, None] * t[:, 2]


def surface_points(obj: ObjectModel, n: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """(n, 3) points sampled uniformly by area on the mesh surface (model frame, cached)."""
    return _surface_points_cached(obj, n, seed)  # ObjectModel hashes by identity


def render_surface(obj: ObjectModel, T_cam_model: np.ndarray, intr: CameraIntrinsics,
                   n_points: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel front-most model point: ``(points (H, W, 3), valid (H, W))``."""
    H, W = intr.height, intr.width
    pts = surface_points(obj, n_points)
    pc = transform_points(T_cam_model, pts)
    uv, ok = intr.project(pc)
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    ok &= (pc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pix = v[ok] * W + u[ok]
    z = pc[ok, 2]
    order = np.lexsort((z, pix))  # by pixel, nearest first
    pix_s = pix[order]
    first = np.ones(len(pix_s), bool)
    first[1:] = pix_s[1:] != pix_s[:-1]
    sel = np.nonzero(ok)[0][order[first]]
    out = np.full((H * W, 3), np.nan)
    out[pix_s[first]] = pts[sel]
    valid = np.zeros(H * W, bool)
    valid[pix_s[first]] = True
    valid = valid.reshape(H, W) & render_mask(obj, T_cam_model, intr, fill_holes=False)
    return out.reshape(H, W, 3), valid


def target_flow(obj: ObjectModel, T0: np.ndarray, T1: np.ndarray, intr: CameraIntrinsics,
                n_points: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Forward flow (H, W, 2) from ``t0`` to ``t1`` on the target and its ``valid`` mask."""
    P, valid = render_surface(obj, T0, intr, n_points)
    flow = np.zeros(P.shape[:2] + (2,), np.float32)
    p = P[valid]
    uv0, ok0 = intr.project(transform_points(T0, p))
    uv1, ok1 = intr.project(transform_points(T1, p))
    f = (uv1 - uv0).astype(np.float32)
    ok = ok0 & ok1
    ys, xs = np.nonzero(valid)
    flow[ys[ok], xs[ok]] = f[ok]
    valid[ys[~ok], xs[~ok]] = False
    return flow, valid


def target_flow_at(traj: Trajectory, sensor: str, obj: ObjectModel, t0_ns: int, t1_ns: int,
                   n_points: int = 1_000_000):
    """Ground-truth target flow between two timestamps for a calibrated sensor.

    Returns ``(flow, valid)`` or ``(None, None)`` when mocap is missing at either time.
    """
    T, ok = pose_in_sensor(traj, sensor, obj, np.array([t0_ns, t1_ns]))
    if not ok.all():
        return None, None
    return target_flow(obj, T[0], T[1], traj.calibration.intrinsics(sensor), n_points)
