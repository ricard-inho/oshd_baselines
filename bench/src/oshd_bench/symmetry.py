"""Object symmetries and the symmetry-invariant reference point used by all pose metrics.

A 10 cm cube looks identical under 24 rotations (the marker holder that breaks the symmetry
sits on the bottom face and is rarely visible), so rotation errors are taken as the minimum
over the symmetry group, and translation errors are measured at the symmetry centre, which
every symmetric pose maps to the same point.

Configured per object in ``configs/bench.yaml``::

    symmetries:
      cubesat: {type: cube, center: auto}   # auto = centre of the object's keypoints bbox
      other:   {type: none}
"""

from __future__ import annotations

import itertools

import numpy as np
from oshd import ObjectModel

__all__ = ["cube_rotations", "Symmetry", "symmetry_for"]


def cube_rotations() -> np.ndarray:
    """(24, 3, 3) proper rotations of the cube (signed permutation matrices with det +1)."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3))
            for i, (j, s) in enumerate(zip(perm, signs)):
                R[i, j] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return np.stack(out)


class Symmetry:
    """A discrete symmetry group ``{S_k}`` acting in the model frame, plus its centre.

    ``transforms`` are 4x4 in the model frame: pose ``T`` is equivalent to ``T @ S_k``.
    """

    def __init__(self, rotations: np.ndarray, center: np.ndarray):
        self.center = np.asarray(center, float)
        self.rotations = np.asarray(rotations, float)
        T = np.tile(np.eye(4), (len(rotations), 1, 1))
        T[:, :3, :3] = self.rotations
        T[:, :3, 3] = self.center - self.rotations @ self.center  # rotate about the centre
        self.transforms = T

    def __len__(self) -> int:
        return len(self.rotations)


def symmetry_for(obj: ObjectModel, cfg: dict | None) -> Symmetry:
    cfg = cfg or {}
    kind = cfg.get("type", "none")
    center = cfg.get("center", "auto")
    if isinstance(center, str):
        pts = obj.click_points
        center = (pts.min(0) + pts.max(0)) / 2
    if kind == "cube":
        return Symmetry(cube_rotations(), center)
    if kind == "none":
        return Symmetry(np.eye(3)[None], center)
    raise ValueError(f"{obj.name}: unknown symmetry type {kind!r} (use 'cube' or 'none')")
