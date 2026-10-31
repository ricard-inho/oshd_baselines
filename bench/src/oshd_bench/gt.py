"""Evaluation sample lists with ground-truth target poses in each sensor frame.

A sample is one (trajectory, sensor, timestamp, object) with the target visible. The sample
list is what every baseline runs on; the ground-truth columns are only read by the
evaluator (and by fine-tuning code on the *train* split).

``T_sensor_model = inv(T_world_body_ref @ T_body_sensor) @ T_world_body_obj @ T_body_model``
comes from mocap + rig extrinsics + ``models/objects.yaml`` (``oshd_dataset.labels``).

Columns: trajectory, domain, illumination, sensor, index, t_ns, object, T (16 floats,
row-major), range_m (sensor -> symmetry centre), ang_rate_dps (relative chaser-target
angular rate from the ground truth), lin_rate_mps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from oshd_dataset import ObjectModel, OshdDataset, Trajectory
from oshd_dataset.geometry import transform_points
from oshd_dataset.labels import object_pose_in_camera
from tqdm import tqdm

from .sensors import sensor_kind
from .symmetry import symmetry_for

__all__ = ["pose_in_sensor", "sample_times", "build_gt", "load_gt", "gt_poses"]


def pose_in_sensor(traj: Trajectory, sensor: str, obj: ObjectModel, t_ns, max_gap_ns: int = 50_000_000):
    """T_sensor_model (N, 4, 4) and validity at ``t_ns``."""
    return object_pose_in_camera(traj, sensor, obj, np.atleast_1d(t_ns), max_gap_ns)


def sample_times(traj: Trajectory, sensor: str, events_rate_hz: float = 20.0,
                 events_window_ms: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Candidate (index, t_ns) for a sensor: every frame / scan, or a fixed event rate.

    For events, the index counts reference times and each sample uses the events in
    ``[t - window, t)``.
    """
    if sensor_kind(sensor) == "events":
        t0, t1 = traj.time_range("events")
        t0 += int(events_window_ms * 1e6)
        ts = np.arange(t0, t1, int(1e9 / events_rate_hz), dtype=np.int64)
        return np.arange(len(ts)), ts
    ts = traj.timestamps(sensor)
    return np.arange(len(ts)), ts


def _visible(traj: Trajectory, sensor: str, obj: ObjectModel, T: np.ndarray, ok: np.ndarray,
             lidar_max_range: float) -> np.ndarray:
    vis = ok.copy()
    if sensor_kind(sensor) == "lidar":
        c = T[:, :3, 3]
        r = np.linalg.norm(c, axis=-1)
        elev = np.degrees(np.arcsin(np.clip(c[:, 2] / np.maximum(r, 1e-9), -1, 1)))
        return vis & (r > 0.1) & (r < lidar_max_range) & (elev > -7) & (elev < 52)  # Mid-360 FoV
    intr = traj.calibration.intrinsics(sensor)
    for k in np.nonzero(ok)[0]:
        uv, good = intr.project(transform_points(T[k], obj.corners))
        uv = uv[good]
        vis[k] = bool(len(uv)) and uv[:, 0].max() > 0 and uv[:, 0].min() < intr.width \
            and uv[:, 1].max() > 0 and uv[:, 1].min() < intr.height
    return vis


def _rates(traj, sensor, obj, ts, center, dt_ns=50_000_000):
    Ta, oka = pose_in_sensor(traj, sensor, obj, ts - dt_ns)
    Tb, okb = pose_in_sensor(traj, sensor, obj, ts + dt_ns)
    R = np.swapaxes(Ta[:, :3, :3], -1, -2) @ Tb[:, :3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    ca = Ta[:, :3, :3] @ center + Ta[:, :3, 3]
    cb = Tb[:, :3, :3] @ center + Tb[:, :3, 3]
    dt = 2 * dt_ns / 1e9
    bad = ~(oka & okb)
    ang, lin = ang / dt, np.linalg.norm(cb - ca, axis=-1) / dt
    ang[bad] = np.nan
    lin[bad] = np.nan
    return ang, lin


def build_gt(ds: OshdDataset, sensor: str, split: str = "test", domain: str | None = "real",
             objects: list[str] | None = None, stride: int = 1, symmetries: dict | None = None,
             events_rate_hz: float = 20.0, events_window_ms: float = 50.0,
             lidar_max_range: float = 15.0, progress: bool = True) -> pd.DataFrame:
    """Ground-truth sample table for ``sensor`` over a split (see module docstring)."""
    rows = []
    objs = [o for n, o in sorted(ds.objects.items()) if objects is None or n in objects]
    names = [t for t in ds.split(split, domain) if ds[t].has("mocap")]
    for tname in tqdm(names, desc=f"gt {sensor}/{split}", disable=not progress):
        traj = ds[tname]
        data_mod = "events" if sensor == "events" else sensor
        if not traj.has(data_mod):
            continue
        info = ds.trajectory_info(tname)
        idx, ts = sample_times(traj, sensor, events_rate_hz, events_window_ms)
        idx, ts = idx[::stride], ts[::stride]
        bodies = set(traj.mocap_bodies)
        for obj in objs:
            if not obj.present_in(tname) or obj.mocap_body not in bodies:
                continue
            T, ok = pose_in_sensor(traj, sensor, obj, ts)
            vis = _visible(traj, sensor, obj, T, ok, lidar_max_range)
            if not vis.any():
                continue
            center = symmetry_for(obj, (symmetries or {}).get(obj.name)).center
            ang, lin = _rates(traj, sensor, obj, ts[vis], center)
            c = T[vis, :3, :3] @ center + T[vis, :3, 3]
            for k, (i, t) in enumerate(zip(idx[vis], ts[vis])):
                rows.append({
                    "trajectory": tname, "domain": info.get("domain", "real"),
                    "illumination": info.get("illumination", "nominal"), "sensor": sensor,
                    "index": int(i), "t_ns": int(t), "object": obj.name,
                    "T": T[vis][k].reshape(-1).tolist(), "range_m": float(np.linalg.norm(c[k])),
                    "ang_rate_dps": float(ang[k]), "lin_rate_mps": float(lin[k]),
                })
    cols = ["trajectory", "domain", "illumination", "sensor", "index", "t_ns", "object", "T",
            "range_m", "ang_rate_dps", "lin_rate_mps"]
    return pd.DataFrame(rows, columns=cols)


def load_gt(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def gt_poses(df: pd.DataFrame) -> np.ndarray:
    """(N, 4, 4) from the ``T`` column."""
    return np.stack([np.asarray(t, float).reshape(4, 4) for t in df["T"]]) if len(df) else np.zeros((0, 4, 4))
