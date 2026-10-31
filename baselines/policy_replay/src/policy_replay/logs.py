"""Logged policy steps, actions (in the policy's normalised space) and the goal of one trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from oshd import Trajectory
from oshd.geometry import quat_to_rotmat

__all__ = ["PolicyLog", "load_log", "wrap", "yaw_of"]


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def yaw_of(R: np.ndarray) -> np.ndarray:
    """Heading of the body +x axis about world z (Isaac ``heading_w``) from (..., 3, 3)."""
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


@dataclass
class PolicyLog:
    t_ns: np.ndarray          # (K,) stamps of the policy steps
    actions: np.ndarray       # (K, 13) logged actions mapped back to [-1, 1] (arms per config)
    goal_xy: np.ndarray       # (2,)
    goal_heading: float
    n_rows: int               # all rows of the action topic (incl. inactive ones)


def invert_wheel_model(tau: np.ndarray, scale: float, b: float, J: float, dt: float) -> np.ndarray:
    """Recover the normalised wheel action from the logged reaction torque (pingu.py model)."""
    e = math.exp(-b * dt / J)
    omega, out = 0.0, np.zeros(len(tau))
    for k, tk in enumerate(tau):
        cmd = tk / e + b * omega
        out[k] = cmd / scale
        omega = cmd / b + (omega - cmd / b) * e
    return out


def load_log(traj: Trajectory, cfg: dict) -> PolicyLog:
    a = cfg["actions"]
    ts, A = traj.message_array(a["topic"])
    active = np.ones(len(A), bool)
    for col, val in (a.get("active_flags") or {}).items():
        active &= A[:, int(col)] == float(val)
    ts, rows = ts[active], A[active]
    act = np.zeros((len(rows), 13))
    act[:, :8] = 2.0 * rows[:, a["thruster_cols"]] - 1.0
    if a.get("arms_prev", "zero") == "logged":
        act[:, 8:12] = rows[:, a["arm_cols"]]
    tau = rows[:, a["rw_col"]]
    if a.get("rw_mode", "invert_wheel_model") == "invert_wheel_model":
        act[:, 12] = invert_wheel_model(tau, a["rw_scale"], a["rw_b"], a["rw_J"], a["rw_dt"])
    else:
        act[:, 12] = tau / a["rw_scale"]
    _, goals = traj.messages(cfg["goal"]["topic"])
    g = goals[-1]["pose"]
    q = np.array([g["orientation"][k] for k in "xyzw"], float)
    return PolicyLog(t_ns=ts.astype(np.int64), actions=act,
                     goal_xy=np.array([g["position"]["x"], g["position"]["y"]], float),
                     goal_heading=float(yaw_of(quat_to_rotmat(q))), n_rows=len(A))
