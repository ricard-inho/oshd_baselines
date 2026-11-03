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
