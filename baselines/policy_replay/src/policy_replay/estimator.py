"""Base-pose tracks: privileged (mocap) and estimator-derived (cube as a known static landmark).

Estimator substitution, per prediction row (``T_sensor_model``, the cube in a sensor frame)::

    T_world_sensor = T_world_model (mocap, median over the trajectory: the cube is static)
                     @ inv(T_sensor_model @ S_k)             # S_k: cube symmetry, see below
    T_world_ref    = T_world_sensor @ inv(T_ref_sensor)      # rig calibration (T_body_cam)
    T_world_base   = T_world_ref @ T_ref_base                # SensorStack -> floatingplatform

``T_ref_base`` is the median of ``inv(T_world_ref) @ T_world_base`` over the trajectory's mocap
(the sensor stack is rigidly mounted), or the median over the other trajectories when the base
body was not tracked. ``S_k`` is chosen so the derived sensor stack's "up" axis is closest to
the mounting's (median world-z in the ref frame): the platform only moves in the plane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from oshd import ObjectModel, OshDataset, Trajectory
from oshd.geometry import invert_pose
from oshd.labels import object_pose

from .tracks import PoseTrack

__all__ = ["RigConstants", "estimator_track", "mocap_track", "rig_constants"]


def _median_pose(T: np.ndarray) -> np.ndarray:
    T = T[np.isfinite(T).all(axis=(1, 2))]
    if not len(T):
        raise ValueError("no valid poses for the median")
    U, _, Vt = np.linalg.svd(np.median(T[:, :3, :3], 0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, np.median(T[:, :3, 3], 0)
    return out


def _body_track(traj: Trajectory, body: str) -> tuple[np.ndarray, np.ndarray]:
    m = traj.mocap(body)
    T, ok = traj.body_pose(body, m["timestamp_ns"], max_gap_ns=None)
    return m["timestamp_ns"][ok], T[ok]


class RigConstants:
    """Per-trajectory constants from mocap: T_ref_base, the ref body's up axis, T_world_model."""

    def __init__(self, ds: OshDataset, cfg: dict):
        self.ds, self.cfg = ds, cfg
        self.base, self.ref = cfg["robot"]["base_body"], cfg["robot"]["ref_body"]
        self._ref_base: dict[str, np.ndarray | None] = {}
        self._pooled: np.ndarray | None = None

    def _own_ref_base(self, name: str) -> np.ndarray | None:
        if name not in self._ref_base:
            traj = self.ds[name]
            if not traj.has("mocap") or not {self.base, self.ref} <= set(traj.mocap_bodies):
                self._ref_base[name] = None
            else:
                t, T_wr = _body_track(traj, self.ref)
                T_wb, ok = traj.body_pose(self.base, t[::10], max_gap_ns=50_000_000)
                self._ref_base[name] = _median_pose(invert_pose(T_wr[::10][ok]) @ T_wb[ok])
        return self._ref_base[name]

    def T_ref_base(self, name: str) -> np.ndarray:
        own = self._own_ref_base(name)
        if own is not None:
            return own
        if self._pooled is None:
            others = [x for x in (self._own_ref_base(n) for n in self.ds.local_trajectory_names) if x is not None]
            if not others:
                raise RuntimeError(f"no trajectory tracks both {self.ref} and {self.base}")
            self._pooled = _median_pose(np.stack(others))
        return self._pooled

    def up_in_ref(self, traj: Trajectory) -> np.ndarray:
        _, T = _body_track(traj, self.ref)
        u = np.median(T[:, 2, :3], 0)          # world z expressed in the ref frame = row 2 of R_wr
        return u / np.linalg.norm(u)

    def T_world_model(self, traj: Trajectory, obj: ObjectModel) -> np.ndarray:
        t = traj.mocap(obj.mocap_body)["timestamp_ns"]
        T, ok = object_pose(traj, obj, t, max_gap_ns=None)
        return _median_pose(T[ok])


def rig_constants(ds: OshDataset, cfg: dict) -> RigConstants:
    return RigConstants(ds, cfg)


def mocap_track(traj: Trajectory, cfg: dict, rc: RigConstants) -> tuple[PoseTrack, str]:
    """Privileged base-pose track (the observation source of the recorded runs)."""
    off = float(cfg["robot"].get("heading_offset_rad", 0.0))
    if rc.base in traj.mocap_bodies:
        t, T = _body_track(traj, rc.base)
        return PoseTrack.from_poses(t, T, off), f"mocap/{rc.base}"
    t, T = _body_track(traj, rc.ref)
    return PoseTrack.from_poses(t, T @ rc.T_ref_base(traj.name), off), f"mocap/{rc.ref}@T_ref_base"


def estimator_track(traj: Trajectory, preds: pd.DataFrame, cfg: dict, rc: RigConstants, obj: ObjectModel,
                    sym_T: np.ndarray, latency: bool) -> tuple[PoseTrack, dict]:
    """Estimator-derived base-pose track; stamps are sample time (+ runtime_ms with ``latency``)."""
    p = preds[(preds["trajectory"] == traj.name) & (preds["object"] == obj.name)]
    p = p.sort_values("score", ascending=False).drop_duplicates("t_ns").sort_values("t_ns")
    info = {"n_estimates": len(p)}
    if not len(p):
        return PoseTrack(np.zeros(0, np.int64), *(np.zeros(0),) * 3), info
    sensors = p["sensor"].unique()
    if len(sensors) != 1:
        raise ValueError(f"predictions mix sensors {sensors}")
    calib = traj.calibration
    if calib.reference_body != rc.ref:
        raise ValueError(f"calibration reference body {calib.reference_body!r} != robot.ref_body {rc.ref!r}")
    T_sensor_ref = invert_pose(calib.T_body_cam(sensors[0]))
    T_wm = rc.T_world_model(traj, obj)
    T_ref_base = rc.T_ref_base(traj.name)
    T_sm = np.stack([np.asarray(t, float).reshape(4, 4) for t in p["T"]])
    if cfg["estimator"].get("symmetry_resolution", "gravity") == "gravity":
        up = rc.up_in_ref(traj)
        cand = T_wm @ invert_pose(T_sm[:, None] @ sym_T[None]) @ T_sensor_ref   # (N, K, 4, 4) T_world_ref
        score = cand[..., 2, :3] @ up                                            # cos(up error)
        k = np.argmax(score, 1)
        T_wr = cand[np.arange(len(cand)), k]
        info["symmetry_nonidentity_frac"] = float(np.mean(k != 0))
        info["up_error_deg_median"] = float(np.degrees(np.median(np.arccos(np.clip(score.max(1), -1, 1)))))
    else:
        T_wr = T_wm @ invert_pose(T_sm) @ T_sensor_ref
    t = p["t_ns"].to_numpy(np.int64)
    if latency:
        rt = np.nan_to_num(p["runtime_ms"].to_numpy(float), nan=0.0)
        t = t + (rt * 1e6).astype(np.int64)
    info["sensor"] = str(sensors[0])
    off = float(cfg["robot"].get("heading_offset_rad", 0.0))
    return PoseTrack.from_poses(t, T_wr @ T_ref_base, off), info
