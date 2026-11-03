"""The shipped PPO teacher outside SRB: its actor (``model_400.pt``) on a 41-d observation.

The checkpoint is RSL-RL's ``MLPModel`` actor with an empirical observation normaliser: action
mean = MLP_elu(384, 384)((obs - mean) / std); evaluation uses the mean (no sampling), as
``srb agent eval``. The observation is the ``steps_layout`` order of the recordings (terms
sorted by name within the state group, then the proprio group).

In the ``steps`` stream the action computed from row ``k``'s observation is recorded at row
``k + 1`` (``Teacher()(obs[k]) == action[k + 1]`` exactly); row ``k``'s ``act_previous`` is
``action[k - 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEFAULT_CKPT = (Path(__file__).resolve().parents[5] / "osh_dataset_pipeline" / "sim" / "policies" /
                "zerog_lab_inspection" / "rsl_rl_ppo" / "20261015T114126" / "model_400.pt")
#: Observation layout of the policy (``metadata.steps_layout.observation``).
OBS_TERMS = (("act_previous", 8), ("tf_pos_robot_to_goal", 3), ("tf_rot6d_robot", 6), ("tf_rot6d_robot_to_goal", 6),
             ("vel_ang_goal", 3), ("vel_ang_robot", 3), ("vel_lin_goal", 3), ("vel_lin_robot", 3),
             ("imu_ang_vel", 3), ("imu_lin_acc", 3))


class Teacher(nn.Module):
    def __init__(self, ckpt: Path | str = DEFAULT_CKPT):
        super().__init__()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.mlp = nn.Sequential(nn.Linear(41, 384), nn.ELU(), nn.Linear(384, 384), nn.ELU(), nn.Linear(384, 8))
        self.mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
        self.register_buffer("mean", sd["obs_normalizer._mean"].float())
        self.register_buffer("std", sd["obs_normalizer._std"].float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / (self.std + 1e-2))  # rsl_rl EmpiricalNormalization

    @staticmethod
    def observation(terms: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(terms[n], np.float32) for n, _ in OBS_TERMS], -1)
