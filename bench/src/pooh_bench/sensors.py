"""Per-sensor data access shared by the baselines (RGB, registered depth, lidar, events).

Sensor names match the dataset's modality / calibration names: ``firefly``,
``realsense_color``, ``realsense_depth``, ``lidar``, ``events``. The lidar and the event
camera need an entry under ``extrinsics.cameras`` in the rig calibration like any camera
(``T_body_cam`` is then the lidar / event-camera frame in the reference body).
"""

from __future__ import annotations

import numpy as np
from pooh_dataset import CameraIntrinsics, Trajectory
from pooh_dataset.events import Events
from pooh_dataset.geometry import transform_points
from pooh_dataset.labels import nearest_indices

__all__ = [
    "CAMERAS",
    "sensor_kind",
    "rgb",
    "depth_in_camera",
    "lidar_points",
    "events_window",
    "backproject",
]

CAMERAS = ("firefly", "realsense_color", "realsense_depth")


def sensor_kind(sensor: str) -> str:
    if sensor in CAMERAS:
        return "camera"
    if sensor == "lidar":
        return "lidar"
    if sensor == "events":
        return "events"
    raise ValueError(f"unknown sensor {sensor!r}")


def rgb(traj: Trajectory, camera: str, index: int) -> np.ndarray:
    """(H, W, 3) uint8 image, distorted (apply the intrinsics' D yourself if needed)."""
    return traj.frame(camera, index)


def backproject(depth_m: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Valid depth pixels -> (points (N, 3) in the camera frame, pixels (N, 2) as (u, v))."""
    vs, us = np.nonzero(depth_m > 0)
    z = depth_m[vs, us].astype(np.float64)
    K = intr.K
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    return np.stack([x * z, y * z, z], -1), np.stack([us, vs], -1)


def depth_in_camera(traj: Trajectory, camera: str, t_ns: int, depth_sensor: str = "realsense_depth",
                    max_dt_ns: int = 40_000_000) -> np.ndarray | None:
    """Depth (metres, 0 = invalid) registered into ``camera`` at the nearest depth frame.

    Needs ``T_body_cam`` for both ``camera`` and ``depth_sensor``. Returns None when no depth
    frame lies within ``max_dt_ns`` of ``t_ns``. Depth distortion is ignored (the D455 depth
    stream is rectified).
    """
    calib = traj.calibration
    j, ok = nearest_indices(traj.timestamps(depth_sensor), int(t_ns), max_dt_ns)
    if not ok:
        return None
    d_intr = calib.intrinsics(depth_sensor)
    pts, _ = backproject(traj.depth(j, depth_sensor), d_intr)
    c_intr = calib.intrinsics(camera)
    pts_c = transform_points(calib.T_cam_cam(camera, depth_sensor), pts)
    uv, ok = c_intr.project(pts_c)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    ok &= (u >= 0) & (u < c_intr.width) & (v >= 0) & (v < c_intr.height) & (pts_c[:, 2] > 0)
    out = np.full(c_intr.height * c_intr.width, np.inf)
    np.minimum.at(out, v[ok] * c_intr.width + u[ok], pts_c[ok, 2])
    out[~np.isfinite(out)] = 0.0
    return out.reshape(c_intr.height, c_intr.width).astype(np.float32)


def lidar_points(traj: Trajectory, index: int, min_range: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) points in the lidar frame and their intensities for scan ``index``.

    Livox scans are not motion-compensated; at the platform's speeds (< 0.3 m/s, 10 Hz scans)
    the distortion is below 3 cm.
    """
    s = traj.lidar_scan(index)
    p = np.stack([s["x"], s["y"], s["z"]], -1).astype(np.float64)
    keep = np.linalg.norm(p, axis=-1) > min_range
    return p[keep], np.asarray(s["intensity"], np.float32)[keep]


def events_window(traj: Trajectory, t_ns: int, window_ms: float) -> Events:
    """Events in ``[t_ns - window, t_ns)``: the window that *ends* at the reference time."""
    w = int(window_ms * 1e6)
    return traj.events(int(t_ns) - w, int(t_ns))
