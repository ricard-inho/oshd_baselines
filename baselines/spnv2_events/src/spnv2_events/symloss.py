"""Symmetry-aware drop-ins for upstream's heatmap and pose-regression losses.

The labels stay canonicalised (``common.canonicalize``); each loss takes, per sample, the min over
the flip-equivalent targets, so a label near a canonicalisation boundary is not penalised for the
equivalent the network picked. Conventions (checked by ``scripts/test_symloss.py``):

* the dataset's ``targets['rotationmatrix']`` is ``quat2dcm(q) = R_cam_net^T``; the equivalent
  poses ``R_cam_net @ F_k`` therefore have targets ``F_k @ R_target`` (``F_k`` symmetric);
* the flipped label ``R_cam_net @ F_k`` puts keypoint ``i`` where original keypoint ``perm_k[i]``
  was, ``CORNER_SIGNS[perm_k[i]] == F_k @ CORNER_SIGNS[i]``: its heatmap target is ``y[:, perm_k]``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def corner_perms(rots: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """(G, K) corner permutations of the symmetry rotations (``rots[0]`` the identity)."""
    perms = []
    for F in rots:
        s = signs @ F.T  # row i: F @ signs[i]
        perms.append([int(np.flatnonzero(np.all(np.isclose(signs, si), 1))[0]) for si in s])
    return np.asarray(perms)


class SymHeatmapLoss(nn.Module):
    """Upstream ``LossHeatmapMSE``, min over the corner permutations ``perms`` per sample."""

    def __init__(self, perms):
        super().__init__()
        self.register_buffer("perms", torch.as_tensor(np.asarray(perms), dtype=torch.long), persistent=False)

    def forward(self, x, y):
        assert x.size() == y.size()
        loss = torch.square(x[:, None] - y[:, self.perms])        # [B, G, K, H, W]
        loss = loss.sum(dim=(3, 4)).mean(dim=2)                   # [B, G]
        return loss.min(dim=1).values.mean()


class SymSPEEDLoss(nn.Module):
    """Upstream ``SPEEDLoss`` with the rotation error the min over the targets ``F_k @ R_gt``
    (``flips`` (G, 3, 3), the first the identity). The translation term is unchanged: the net
    frame's origin is the bounding-box centre, fixed by the flips."""

    def __init__(self, flips):
        super().__init__()
        self.register_buffer("flips", torch.as_tensor(np.asarray(flips), dtype=torch.float32), persistent=False)

    def forward(self, r_raw_pr, t_pr, R_gt, t_gt, anchor_states):
        from utils.postprocess import rot_6d_to_matrix

        B, N = anchor_states.shape
        pos = torch.eq(anchor_states, 1)
        R_pr = rot_6d_to_matrix(r_raw_pr[pos, :])                                  # [P, 3, 3]
        t_pr = t_pr[pos, :]
        R_gt = R_gt.view(B, 1, 3, 3).repeat(1, N, 1, 1)[pos]                       # [P, 3, 3]
        t_gt = t_gt.view(B, 1, 3).repeat(1, N, 1)[pos]

        R_eq = self.flips.to(R_gt.dtype)[None] @ R_gt[:, None]                     # [P, G, 3, 3]
        trace = (R_pr[:, None] * R_eq).sum(dim=(2, 3))                              # tr(R_pr R_eq^T)
        rot_err = torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-6, 1 - 1e-6)).min(dim=1).values

        pos_err = torch.linalg.vector_norm(t_pr - t_gt, ord=2, dim=1)
        pos_err = pos_err.div(torch.linalg.vector_norm(t_gt, ord=2, dim=1))
        return (rot_err + pos_err).mean()
