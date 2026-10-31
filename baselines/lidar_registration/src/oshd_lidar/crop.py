"""Target crop for the lidar baselines: find the ~10 cm cube in a whole Mid-360 scan.

``TargetCropper`` is training-free and never looks at ground truth. Per scan:

1. **Range + height gate**: range ``[r_min, r_max]`` (default 0.15-2.0 m, the working
   volume) and height along gravity ``height_band`` (default -0.25..+0.12 m relative to the
   lidar). Gravity comes from the Mid-360's built-in IMU (mean accelerometer direction of the
   trajectory, sensor data only). This is the known experiment geometry: chaser and target
   float on the same flat table, so the target stays near the sensor's height.
2. **Large-plane removal**: iterative RANSAC planes (2 cm threshold); inliers are removed
   only where they form a connected patch larger than 0.25 m (walls, table, panels), so a
   close cube face that happens to seed a plane survives.
3. **Euclidean clustering** (DBSCAN, ``eps`` 3.5 cm). Clusters that are too big for the cube
   are re-clustered once with a tighter eps (2 cm) to cut them off neighbouring structures.
4. **Cube-size gating**: keep clusters with >= ``min_points`` points and oriented-box
   extents <= 0.2 m (cube body + marker holder diagonal), second extent >= 2 cm.
5. **Selection** by a hand-set cube-likeness score (log point count + Gaussian terms on the
   three PCA extents: a visible cube is ~12 x 9 x 5 cm, clutter is mostly thin) and,
   when tracking is enabled, clusters close to the previous scan's pick in the same
   trajectory (causal, lidar-frame only; reset per trajectory; ``track=False`` disables it).

The selected cluster is grown by ``grow`` (2 cm) from the unfiltered range-gated points so
that plane removal cannot eat cube faces. Output: points in the lidar frame.

``oracle_crop`` (GT-centred sphere) exists for diagnosis only and must be written under its
own setting name (``ZS-oracle``), never as a Table I result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .common import CUBE_CENTER, transform


@dataclass
class CropResult:
    points: np.ndarray                  # (N, 3) lidar frame
    center: np.ndarray | None           # cluster centroid
    score: float
    n_candidates: int
    candidates: list = field(default_factory=list)  # centroids of all gated clusters (debug)


def _dbscan(p: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    import open3d as o3d

    if len(p) == 0:
        return np.zeros(0, int)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    return np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))


def _extents(p: np.ndarray) -> np.ndarray:
    q = p - p.mean(0)
    ev = np.linalg.svd(q, full_matrices=False)[2]
    proj = q @ ev.T
    return np.ptp(proj, 0)  # descending-ish (principal axes)


class TargetCropper:
    def __init__(self, r_min: float = 0.15, r_max: float = 2.0, height_band=(-0.25, 0.12),
                 up=None, plane_min_inliers: int = 300,
                 plane_dist: float = 0.02, max_planes: int = 8, eps: float = 0.035,
                 min_points: int = 6, max_extent: float = 0.20, grow: float = 0.02,
                 track: bool = True, track_sigma: float = 0.15, seed: int = 0):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self._prev: dict[str, np.ndarray] = {}

    def reset(self):
        self._prev.clear()

    # ------------------------------------------------------------------
    def _remove_planes(self, p: np.ndarray) -> np.ndarray:
        import open3d as o3d

        keep = np.ones(len(p), bool)
        o3d.utility.random.seed(self.seed)
        for _ in range(self.max_planes):
            idx = np.nonzero(keep)[0]
            if len(idx) < self.plane_min_inliers:
                break
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p[idx]))
            _, inl = pc.segment_plane(self.plane_dist, 3, 300)
            if len(inl) < self.plane_min_inliers:
                break
            inl = idx[np.asarray(inl)]
            lab = _dbscan(p[inl], 0.05, 3)
            removed = False
            for k in range(lab.max() + 1 if len(lab) else 0):
                sel = inl[lab == k]
                if np.ptp(p[sel], 0).max() > 0.25:
                    keep[sel] = False
                    removed = True
            if not removed:
                break
        return keep

    def _candidates(self, q: np.ndarray):
        out = []
        lab = _dbscan(q, self.eps, 4)
        for k in range(lab.max() + 1 if len(lab) else 0):
            m = q[lab == k]
            if len(m) < self.min_points:
                continue
            ext = _extents(m)
            if ext[0] > self.max_extent:
                if ext[0] > 0.8:  # a large structure, not a merged cube
                    continue
                sub = _dbscan(m, 0.02, 3)
                for j in range(sub.max() + 1 if len(sub) else 0):
                    s = m[sub == j]
                    if len(s) >= self.min_points and _extents(s)[0] <= self.max_extent:
                        out.append(s)
                continue
            out.append(m)
        return [m for m in out if _extents(m)[1] >= 0.02]

    def _score(self, m: np.ndarray, prev: np.ndarray | None) -> float:
        e0, e1, e2 = _extents(m)
        s = np.log(len(m))
        s -= 0.5 * ((e0 - 0.12) / 0.05) ** 2 + 0.5 * ((e1 - 0.09) / 0.035) ** 2
        s -= 0.5 * (max(0.0, 0.04 - e2) / 0.015) ** 2
        if prev is not None:
            d = np.linalg.norm(m.mean(0) - prev)
            s += np.log(0.1 + np.exp(-0.5 * (d / self.track_sigma) ** 2))
        return float(s)

    def __call__(self, p: np.ndarray, key: str | None = None, up=None) -> CropResult:
        """Crop one scan. ``key`` (trajectory name) enables tracking, ``up`` overrides self.up."""
        up = self.up if up is None else up
        r = np.linalg.norm(p, axis=1)
        ok = (r > self.r_min) & (r < self.r_max)
        if up is not None and self.height_band is not None:
            h = p @ np.asarray(up, float)
            ok &= (h > self.height_band[0]) & (h < self.height_band[1])
        g = p[ok]
        if len(g) == 0:
            return CropResult(np.zeros((0, 3)), None, 0.0, 0)
        q = g[self._remove_planes(g)]
        cands = self._candidates(q)
        if not cands:
            return CropResult(np.zeros((0, 3)), None, 0.0, 0)
        prev = self._prev.get(key) if (self.track and key is not None) else None
        scores = [self._score(m, prev) for m in cands]
        best = cands[int(np.argmax(scores))]
        # grow into the unfiltered gated cloud (restores points eaten by plane removal)
        d, _ = cKDTree(best).query(g, distance_upper_bound=self.grow)
        pts = g[np.isfinite(d)]
        if _extents(pts)[0] > self.max_extent * 1.2:
            pts = best
        c = pts.mean(0)
        if key is not None:
            self._prev[key] = c
        return CropResult(pts, c, float(max(scores)), len(cands), [m.mean(0) for m in cands])


def gravity_up(traj) -> np.ndarray:
    """Unit 'up' vector in the lidar frame from the Mid-360 IMU (mean specific force).

    The Livox IMU axes are aligned with the lidar frame; the platform's own accelerations
    are tiny (floating platform), so the mean accelerometer reading is gravity.
    """
    key = (str(getattr(traj, "root", "")), traj.name)
    if key in _UP_CACHE:
        return _UP_CACHE[key]
    imu = traj.imu()
    a = np.stack([imu[f"linear_acceleration_{k}"] for k in "xyz"], 1).mean(0)
    _UP_CACHE[key] = a / np.linalg.norm(a)
    return _UP_CACHE[key]


_UP_CACHE: dict = {}


def oracle_crop(p: np.ndarray, T_lidar_model: np.ndarray, radius: float = 0.12) -> CropResult:
    """DIAGNOSIS ONLY: points within ``radius`` of the GT cube centre."""
    c = transform(T_lidar_model, CUBE_CENTER[None])[0]
    pts = p[np.linalg.norm(p - c, axis=1) < radius]
    return CropResult(pts, c, 1.0, 1)
