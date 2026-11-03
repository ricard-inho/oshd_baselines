"""Student inputs, shared by the training cache (``build_cache``) and the policy server.

Every input is computed by the same function from the same raw data in both places: the
recorded streams of the dataset (training) and the bridge's request (closed loop), so the
students see in SRB what they were trained on.

- ``rgb``: (3, H, W) uint8, the camera image resized to ``IMAGE_HW``
- ``depth``: (1, H, W) uint8, depth along the optical axis, 0-``DEPTH_MAX_M`` m -> 1-255, 0 = none
- ``events``: (2, H, W) uint8, per-polarity event counts of the 50 ms before the step,
  binned to ``IMAGE_HW``, saturating at 255
- ``lidar``: (1, 16, 360) uint8, range image (channel x 1 deg azimuth), 0-10 m -> 1-255, 0 = none
- ``proprio`` (26,): ``PROPRIO_TERMS`` of the policy observation (no goal terms), the previous
  action as applied (clipped to [0, 1])
- ``command`` (3,): ``(cos b, sin b, d)``, the docking goal's bearing around the target centre
  and its distance from it (the task's pre-dock / glide schedule)
"""

from __future__ import annotations

import numpy as np

IMAGE_HW = (96, 128)
SENSOR_HW = (240, 320)
DEPTH_MAX_M = 6.0
LIDAR_CHANNELS, LIDAR_AZ_BINS, LIDAR_VFOV_DEG, LIDAR_MAX_M = 16, 360, (-15.0, 15.0), 10.0
EVENTS_WINDOW_NS = 50_000_000
#: Policy observation terms a student observes (the teacher also sees the goal terms).
PROPRIO_TERMS = (("act_previous", 8), ("tf_rot6d_robot", 6), ("vel_ang_robot", 3), ("vel_lin_robot", 3),
                 ("imu_ang_vel", 3), ("imu_lin_acc", 3))
PROPRIO_DIM = sum(k for _, k in PROPRIO_TERMS)
MODALITIES = ("rgb", "depth", "events", "lidar")


def _resize_area(img: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Area-averaging downscale (OpenCV ``INTER_AREA``)."""
    import cv2

    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def rgb_input(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(_resize_area(rgb[..., :3].astype(np.uint8), IMAGE_HW).transpose(2, 0, 1))


def depth_input(depth_m: np.ndarray) -> np.ndarray:
    """``depth_m``: metres, 0 or non-finite = no return. Nearest-neighbour downscale (no
    mixing of foreground and background depths)."""
    import cv2

    d = np.where(np.isfinite(depth_m), depth_m, 0.0).astype(np.float32)
    d = cv2.resize(d, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_NEAREST)
    q = np.where(d > 0, 1 + np.clip(d / DEPTH_MAX_M, 0, 1) * 254, 0)
    return q.astype(np.uint8)[None]


def events_input(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Events of the window (pixel coordinates of the 320x240 sensor, polarity 0/1 or -1/1)."""
    h, w = IMAGE_HW
    out = np.zeros((2, h, w), np.int32)
    if len(x):
        xi = np.clip((x.astype(np.int64) * w) // SENSOR_HW[1], 0, w - 1)
        yi = np.clip((y.astype(np.int64) * h) // SENSOR_HW[0], 0, h - 1)
        pi = (p > 0).astype(np.int64)
        np.add.at(out, (pi, yi, xi), 1)
    return np.minimum(out, 255).astype(np.uint8)


def lidar_input(points: np.ndarray) -> np.ndarray:
    """``points`` (m, 3): returns in the lidar sensor frame."""
    out = np.zeros((LIDAR_CHANNELS, LIDAR_AZ_BINS), np.float32)
    if len(points):
        p = points[:, :3].astype(np.float64)
        r = np.linalg.norm(p, axis=1)
        ok = r > 1e-6
        p, r = p[ok], r[ok]
        el = np.degrees(np.arcsin(np.clip(p[:, 2] / r, -1, 1)))
        lo, hi = LIDAR_VFOV_DEG
        ch = np.clip(np.round((el - lo) / (hi - lo) * (LIDAR_CHANNELS - 1)), 0, LIDAR_CHANNELS - 1).astype(int)
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        ai = np.clip(((az + 180.0) / 360.0 * LIDAR_AZ_BINS).astype(int), 0, LIDAR_AZ_BINS - 1)
        q = 1 + np.clip(r / LIDAR_MAX_M, 0, 1) * 254
        # nearest return per cell: write the far ones first
        order = np.argsort(-q)
        out[ch[order], ai[order]] = q[order]
    return out.astype(np.uint8)[None]


def applied(action: np.ndarray) -> np.ndarray:
    """The thrust the platform applies: raw actions clipped to [0, 1] (``ThrustAction``)."""
    return np.clip(np.asarray(action, np.float32), 0.0, 1.0)


def proprio_from_terms(terms: dict[str, np.ndarray]) -> np.ndarray:
    """``terms``: name -> (k,) or (n, k). Returns (..., PROPRIO_DIM) float32. The previous
    action enters as applied (a student's own outputs are already in [0, 1])."""
    return np.concatenate([applied(terms[name]) if name == "act_previous" else np.asarray(terms[name], np.float32)
                           for name, _ in PROPRIO_TERMS], -1)


def proprio_from_steps(observation: np.ndarray, layout: list[dict]) -> np.ndarray:
    """From the ``steps`` stream's observation matrix and its ``steps_layout``."""
    spans, i = {}, 0
    for term in layout:
        spans[term["name"]] = (i, i + term["size"])
        i += term["size"]
    return proprio_from_terms({name: observation[:, slice(*spans[name])] for name, _ in PROPRIO_TERMS})


def command(bearing: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(bearing), np.sin(bearing), distance], -1).astype(np.float32)
