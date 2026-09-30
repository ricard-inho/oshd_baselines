"""Named observation features and the declarative 26-dim layout from ``configs/replay.yaml``.

Feature definitions mirror ``GoThroughPosesTask.get_observations`` (task data) and
``PinguRobot.get_observations`` (previous actions).
"""

from __future__ import annotations

import numpy as np

from .logs import wrap
from .tracks import State

__all__ = ["FEATURES", "build_observations", "layout_names"]


def _task_features(s: State, goal_xy: np.ndarray, goal_heading: float, sub_mode: str) -> dict[str, np.ndarray]:
    dx, dy = goal_xy[0] - s.x, goal_xy[1] - s.y
    bearing = wrap(np.arctan2(dy, dx) - s.heading)       # "target_heading_error" in the source
    herr = wrap(goal_heading - s.heading)                # "heading_error" in the source
    n = len(s.x)
    f = {
        "lin_vel_body_x": s.vx_b, "lin_vel_body_y": s.vy_b, "ang_vel_z": s.wz,
        "goal_distance": np.hypot(dx, dy),
        "cos_bearing_error": np.cos(bearing), "sin_bearing_error": np.sin(bearing),
        "cos_heading_error": np.cos(herr), "sin_heading_error": np.sin(herr),
    }
    if sub_mode == "self_loop":
        a = wrap(0.0 - goal_heading)                     # atan2(0, 0) - heading of the "previous" goal
        sub = [0.0, np.cos(a), np.sin(a), 1.0, 0.0]
    elif sub_mode == "zeros":
        sub = [0.0] * 5
    else:
        raise ValueError(f"unknown goal.subsequent_goals {sub_mode!r}")
    for k, v in zip(["next_goal_distance", "next_goal_cos_bearing", "next_goal_sin_bearing",
                     "next_goal_cos_heading", "next_goal_sin_heading"], sub):
        f[k] = np.full(n, v)
    return f


FEATURES = [
    "lin_vel_body_x", "lin_vel_body_y", "ang_vel_z", "goal_distance", "cos_bearing_error",
    "sin_bearing_error", "cos_heading_error", "sin_heading_error", "next_goal_distance",
    "next_goal_cos_bearing", "next_goal_sin_bearing", "next_goal_cos_heading", "next_goal_sin_heading",
    "prev_action",
]


def layout_names(cfg: dict) -> list[str]:
    return [e["feature"] + (f"[{e['index']}]" if "index" in e else "") for e in cfg["observation"]]


def build_observations(cfg: dict, s: State, goal_xy, goal_heading: float, prev_actions: np.ndarray) -> np.ndarray:
    """(K, D) observations; each entry is ``scale * feature`` (scale defaults to 1)."""
    f = _task_features(s, np.asarray(goal_xy, float), goal_heading, cfg["goal"].get("subsequent_goals", "self_loop"))
    cols = []
    for e in cfg["observation"]:
        name = e["feature"]
        if name == "prev_action":
            v = prev_actions[:, int(e["index"])]
        elif name in f:
            v = f[name]
        else:
            raise KeyError(f"unknown observation feature {name!r} (have {FEATURES})")
        cols.append(float(e.get("scale", 1.0)) * np.asarray(v, float))
    return np.stack(cols, -1)
