"""Modular pipeline (paper Table III, last row): estimated target pose -> the PPO teacher.

The teacher's goal terms are rebuilt from an *estimate* of the target's pose instead of the
simulator's: the docking goal is ``centre + d (cos b, sin b)`` facing the centre (``b``, ``d``:
the docking command), with ``centre = T_world_target * centre_offset`` (the target's centre in
its body frame: zero for the arm-held and floating targets, ``target_center_offset`` for the
static twins). The robot's own state (pose, velocities, IMU) is known, as for the students.

``GoalTracker.terms`` returns the teacher's goal terms in its observation layout
(``teacher.OBS_TERMS``): ``tf_pos_robot_to_goal``, ``tf_rot6d_robot_to_goal``,
``vel_lin_goal`` and ``vel_ang_goal`` (the goal's finite-difference velocity in the robot
frame, as the task computes it).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rot6d(Rm: np.ndarray) -> np.ndarray:
    """Isaac Lab's ``rotmat_to_rot6d``: the first two columns, row-major."""
    return Rm[..., :, :2].reshape(*Rm.shape[:-2], 6)


class GoalTracker:
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.prev: tuple[np.ndarray, float] | None = None

    def reset(self) -> None:
        self.prev = None

    def terms(self, robot_pose: np.ndarray, target_center_w: np.ndarray, command: np.ndarray) -> dict[str, np.ndarray]:
        """``robot_pose`` (7,): world xyz + quat xyzw of the base; ``target_center_w`` (3,);
        ``command`` (3,): cos b, sin b, d."""
        c, s, d = (float(v) for v in command)
        b = np.arctan2(s, c)
        goal = np.array([target_center_w[0] + d * c, target_center_w[1] + d * s, robot_pose[2]])
        yaw = b + np.pi
        R_r = R.from_quat(robot_pose[3:]).as_matrix()
        R_g = R.from_euler("z", yaw).as_matrix()
        if self.prev is None:
            v_w, w_z = np.zeros(3), 0.0
        else:
            v_w = (goal - self.prev[0]) / self.dt
            v_w[2] = 0.0
            w_z = float(np.arctan2(np.sin(yaw - self.prev[1]), np.cos(yaw - self.prev[1]))) / self.dt
        self.prev = (goal, yaw)
        return {
            "tf_pos_robot_to_goal": (R_r.T @ (goal - robot_pose[:3])).astype(np.float32),
            "tf_rot6d_robot_to_goal": rot6d(R_r.T @ R_g).astype(np.float32),
            "vel_lin_goal": (R_r.T @ v_w).astype(np.float32),
            "vel_ang_goal": (R_r.T @ np.array([0.0, 0.0, w_z])).astype(np.float32),
        }


def target_center(target_pose_w: np.ndarray, center_offset: np.ndarray) -> np.ndarray:
    return target_pose_w[:3] + R.from_quat(target_pose_w[3:]).apply(center_offset)


class StaticTargetFilter:
    """Fuses position estimates of the target's centre over time, in the world frame (the
    centre is static: the arm-held, floating and static targets only spin about it).

    A measurement is the centre in the world (``robot pose * estimate in the base frame``) with
    an isotropic standard deviation ``sd`` (m). Measurements with ``sd > max_sd`` carry no
    information (target out of view) and are skipped. The first one with ``sd <= init_sd``
    initialises the estimate; later ones are fused by a Kalman update (random-walk process noise
    ``q`` per step) unless they are more than ``gate`` standard deviations away (outliers); after
    ``reset_after`` consecutive rejections the filter restarts from the current measurement.
    """

    def __init__(self, q: float = 0.0005, gate: float = 3.0, init_sd: float = 0.3, max_sd: float = 0.5,
                 reset_after: int = 200, init_n: int = 1):
        self.q, self.gate, self.init_sd, self.max_sd, self.reset_after = q, gate, init_sd, max_sd, reset_after
        self.init_n = init_n
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.P = 0.0
        self.rejected = 0
        self.pending: list[tuple[np.ndarray, float]] = []
        self.stats = {"fused": 0, "rejected": 0, "skipped": 0, "resets": 0}

    def update(self, centre_w: np.ndarray | None, sd: float) -> np.ndarray | None:
        if self.x is not None:
            self.P += self.q ** 2
        if centre_w is None or not np.isfinite(sd) or sd > self.max_sd:
            self.stats["skipped"] += 1
            return self.x
        c = np.asarray(centre_w, float)
        if self.x is None:
            # start from the (component-wise) median of the first ``init_n`` confident
            # measurements: a single confident outlier does not set the estimate
            if sd <= self.init_sd:
                self.pending.append((c, sd))
            if len(self.pending) >= self.init_n:
                cs = np.stack([p[0] for p in self.pending])
                self.x = np.median(cs, 0)
                self.P = float(np.median([p[1] for p in self.pending])) ** 2
                self.pending = []
            return self.x
        d = np.linalg.norm((c - self.x)[:2]) / np.sqrt(self.P + sd ** 2)
        if d > self.gate:
            self.rejected += 1
            self.stats["rejected"] += 1
            if self.rejected >= self.reset_after and sd <= self.init_sd:
                self.x, self.P, self.rejected = c.copy(), sd ** 2, 0
                self.stats["resets"] += 1
            return self.x
        k = self.P / (self.P + sd ** 2)
        self.x = self.x + k * (c - self.x)
        self.P = (1 - k) * self.P
        self.rejected = 0
        self.stats["fused"] += 1
        return self.x


def search_terms(goal: GoalTracker, robot_pose: np.ndarray, yaw_step: float, back: float = 0.0) -> dict[str, np.ndarray]:
    """Goal terms that make the teacher turn in place by ``yaw_step`` (rad), and back away by
    ``back`` (m): the goal at the robot's position (minus ``back`` along its heading), rotated.
    Used while there is no estimate of the target."""
    yaw0 = R.from_quat(robot_pose[3:]).as_euler("xyz")[2]
    yaw = yaw0 + yaw_step
    b = yaw - np.pi  # GoalTracker faces b + pi
    pos = robot_pose[:3] - back * np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
    centre = pos - 1e-6 * np.array([np.cos(b), np.sin(b), 0.0])
    return goal.terms(robot_pose, centre, np.array([np.cos(b), np.sin(b), 1e-6]))
