"""Closed-loop docking metrics (paper Table III) from the episode logs of ``closed_loop.py``.

    uv run policy-metrics ../../results/policies/docking_test_v3/teacher

Per episode (``episodes.parquet``) and averaged over the episodes (``summary.json``):

- **success**: docked at the end of the episode: the docking glide finished (task phase 2:
  the goal is at the docking standoff), and the platform is within ``--pos-tol`` (5 cm, planar)
  and ``--att-tol`` (5 deg, yaw) of the docked pose, moving slower than ``--vel-tol`` (2 cm/s).
- **pos_err_cm / att_err_deg**: planar position and yaw error to the docked pose at the end.
- **contact_vel_cms**: planar speed of the platform relative to the target when it first
  reaches the docking standoff (centre distance <= standoff + 2 cm). The simulated targets do
  not collide (``target_satellite_cfg``), so the docking interface is this standoff.
- **thruster_usage_s**: total applied thrust over the episode, sum_t sum_i u_i dt with
  u_i in [0, 1] (seconds of one thruster at full thrust).
- **overlap**: whether the platform's body box (``geometry.PLATFORM_BOX``) ever intersects
  the target's CAD (``min_clearance_cm`` <= 0); reported, not part of success (the docking
  standoff is a centre distance and ignores the target's size).

Means over all episodes; ``contact_vel_cms`` over the episodes that reach the standoff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def yaw(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


_PTS: dict = {}


def target_points(trajectory: str):
    if trajectory not in _PTS:
        from oshd import OshDataset

        from .geometry import surface_points

        try:
            _PTS[trajectory] = surface_points(OshDataset()[trajectory].target_mesh())
        except Exception as ex:  # no local model: overlap stays unknown
            print(f"{trajectory}: no target CAD ({ex})")
            _PTS[trajectory] = None
    return _PTS[trajectory]


def episode_metrics(path: Path, a) -> dict:
    from .geometry import clearance

    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    dt = float(meta["agent_rate_s"])
    rp, g, tc = z["robot_pose"], z["goal"], z["target_center"]
    vel = z["robot_vel_w"]
    phase, standoff = z["phase"], z["dock_standoff"]
    pos_err = np.linalg.norm(rp[:, :2] - g[:, :2], axis=1)
    att_err = np.abs(wrap(yaw(g[:, 3:]) - yaw(rp[:, 3:])))
    speed = np.linalg.norm(vel[:, :2], axis=1)
    dist = np.linalg.norm(rp[:, :2] - tc[:, :2], axis=1)
    reached = np.nonzero((phase >= 1) & (dist <= standoff + 0.02))[0]
    thrust = z["thrust"] if "thrust" in z else np.clip(z["action"], 0, 1)
    tag = meta["tag"]
    row = {
        "trajectory": tag, "episode": path.stem,
        "target_kind": meta["task_info"].get("target_kind"), "target_mount": meta["task_info"].get("target_mount"),
        "steps": len(phase), "docked_phase": bool(phase[-1] >= 2),
        "pos_err_cm": 100 * pos_err[-1], "att_err_deg": float(np.degrees(att_err[-1])),
        "final_speed_cms": 100 * speed[-1],
        "contact_vel_cms": 100 * speed[reached[0]] if len(reached) else np.nan,
        "t_contact_s": reached[0] * dt if len(reached) else np.nan,
        "thruster_usage_s": float(thrust.sum() * dt),
        "standoff_m": float(standoff[-1]),
    }
    row["success"] = bool(row["docked_phase"] and row["pos_err_cm"] < 100 * a.pos_tol
                          and row["att_err_deg"] < a.att_tol and row["final_speed_cms"] < 100 * a.vel_tol)
    pts = None if a.no_overlap else target_points(tag)
    if pts is not None:
        c = clearance(rp[::a.overlap_stride], z["target_pose"][::a.overlap_stride], pts)
        row["min_clearance_cm"] = 100 * float(c.min())
        row["overlap"] = bool(c.min() <= 0)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policy-metrics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="closed_loop.py --out directory")
    ap.add_argument("--pos-tol", type=float, default=0.05)
    ap.add_argument("--att-tol", type=float, default=5.0)
    ap.add_argument("--vel-tol", type=float, default=0.02)
    ap.add_argument("--overlap-stride", type=int, default=5)
    ap.add_argument("--no-overlap", action="store_true")
    a = ap.parse_args(argv)
    files = sorted((a.run / "episodes").glob("*.npz"))
    df = pd.DataFrame([episode_metrics(f, a) for f in files])
    df.to_parquet(a.run / "episodes.parquet", index=False)
    cols = ["pos_err_cm", "att_err_deg", "contact_vel_cms", "thruster_usage_s"]
    summary = {
        "episodes": len(df), "success_pct": 100 * float(df["success"].mean()),
        **{c: float(df[c].mean()) for c in cols},
        **{f"{c}_median": float(df[c].median()) for c in cols},
        "reached_standoff_pct": 100 * float(df["contact_vel_cms"].notna().mean()),
        "criteria": {"pos_tol_m": a.pos_tol, "att_tol_deg": a.att_tol, "vel_tol_mps": a.vel_tol},
    }
    if "overlap" in df:
        summary["overlap_pct"] = 100 * float(df["overlap"].mean())
    summary["by_mount"] = {k: {"n": len(g), "success_pct": 100 * float(g["success"].mean()),
                               "pos_err_cm": float(g["pos_err_cm"].mean())}
                           for k, g in df.groupby("target_mount")}
    (a.run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_mount"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
