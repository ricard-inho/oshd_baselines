"""Planar pose tracks (mocap or estimator-derived) and the causal state filter the replay uses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logs import wrap, yaw_of

__all__ = ["PoseTrack", "State", "causal_state"]


@dataclass
class PoseTrack:
    """Base poses usable from time ``t_ns`` on (sorted): x, y and heading in the world frame."""

    t_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray           # unwrapped

    @classmethod
    def from_poses(cls, t_ns: np.ndarray, T: np.ndarray, heading_offset: float = 0.0) -> PoseTrack:
        ok = np.isfinite(T).all(axis=(1, 2))
        t_ns, T = np.asarray(t_ns, np.int64)[ok], T[ok]
        order = np.argsort(t_ns, kind="stable")
        t_ns, T = t_ns[order], T[order]
        return cls(t_ns, T[:, 0, 3], T[:, 1, 3], np.unwrap(yaw_of(T[:, :3, :3]) + heading_offset))

    def __len__(self) -> int:
        return len(self.t_ns)


@dataclass
class State:
    """Per policy step: x, y, heading, body-frame velocity, yaw rate, and bookkeeping."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx_b: np.ndarray
    vy_b: np.ndarray
    wz: np.ndarray
    valid: np.ndarray         # a pose was available at or before the step
    age_ms: np.ndarray        # step time - stamp of the pose used


def _buffer_mean_velocity(track: PoseTrack, hi: int, n: int) -> np.ndarray | None:
    """Velocity as ``OptitrackStatePreProcessor`` computes it on the robot (RANS_DeployToRobot).

    Over the last ``n`` samples: mean of the per-sample differences divided by
    ``(t_last - t_first) / n``, i.e. ``(p_last - p_first) / (t_last - t_first) * n / (n - 1)``.
    The robot uses the quaternion-difference body rate; for the planar platform its z component
    is the yaw rate, taken here from the unwrapped yaw.
    """
    lo = hi - n
    if lo < 0:
        return None
    dt = (track.t_ns[hi - 1] - track.t_ns[lo]) / 1e9 / n
    if dt <= 0:
        return None
    return np.array([np.diff(v[lo:hi]).mean() / dt for v in (track.x, track.y, track.yaw)])


def causal_state(track: PoseTrack, t_query: np.ndarray, window_ms: float, min_samples: int = 3,
                 mode: str = "fit", buffer_size: int = 30) -> State:
    """Zero-order hold of the latest pose at or before each query + a causal velocity.

    Velocity, ``mode="fit"``: straight-line fit of x, y, yaw over the samples in
    ``(t - window, t]``; with fewer than ``min_samples`` samples the previous step's velocity is
    held (0 before the first fit). ``mode="buffer_mean"``: the robot's own estimator over the last
    ``buffer_size`` samples (see ``_buffer_mean_velocity``; 0 until the buffer is full).
    Only past samples are used, so no future information enters any step.
    """
    if mode not in ("fit", "buffer_mean"):
        raise ValueError(f"unknown velocity.mode {mode!r}")
    n = len(t_query)
    out = {k: np.full(n, np.nan) for k in ("x", "y", "h", "vx", "vy", "wz", "age")}
    valid = np.zeros(n, bool)
    w = int(window_ms * 1e6)
    last_v = np.zeros(3)
    for i, t in enumerate(np.asarray(t_query, np.int64)):
        hi = np.searchsorted(track.t_ns, t, side="right")
        if hi == 0:
            continue
        j = hi - 1
        valid[i] = True
        out["x"][i], out["y"][i], out["h"][i] = track.x[j], track.y[j], wrap(track.yaw[j])
        out["age"][i] = (t - track.t_ns[j]) / 1e6
        if mode == "buffer_mean":
            v = _buffer_mean_velocity(track, hi, buffer_size)
            if v is not None:
                last_v = v
            out["vx"][i], out["vy"][i], out["wz"][i] = last_v
            continue
        lo = np.searchsorted(track.t_ns, t - w, side="right")
        if hi - lo >= max(min_samples, 2):
            tt = (track.t_ns[lo:hi] - t) / 1e9
            if np.ptp(tt) > 0:
                X = np.stack([np.ones_like(tt), tt], -1)
                Y = np.stack([track.x[lo:hi], track.y[lo:hi], track.yaw[lo:hi]], -1)
                last_v = np.linalg.lstsq(X, Y, rcond=None)[0][1]
        out["vx"][i], out["vy"][i], out["wz"][i] = last_v
    c, s = np.cos(out["h"]), np.sin(out["h"])
    return State(out["x"], out["y"], out["h"], c * out["vx"] + s * out["vy"], -s * out["vx"] + c * out["vy"],
                 out["wz"], valid, out["age"])
