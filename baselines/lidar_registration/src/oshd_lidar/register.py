"""FPFH + TEASER++ + ICP: register the CAD point cloud to the cropped cube points.

The source is the cropped scan (lidar frame) and the target is the CAD surface sample (model
frame). The pipeline estimates T_model_lidar and returns its inverse, T_lidar_model.

1. Voxel-downsample both clouds (``voxel``, 1 cm) and estimate normals. The scan normals are
   oriented towards the sensor.
2. Compute FPFH features (Open3D; ``feat_radius`` 5 cm). Each scan point is matched to its
   ``k`` (5) nearest model points in feature space, which gives the putative correspondences
   (at most ``max_corr`` = 400, lowest feature distance first).
3. TEASER++ (TLS translation, GNC-TLS rotation, heuristic max-clique inlier selection
   ``PMC_HEU``) with a noise bound of 2 cm, which is about the Mid-360 range noise. On a
   10 cm object nearly all pairs are consistent at that bound, so the inlier graph is dense
   and the exact max clique can take tens of seconds. The heuristic bounds it.
4. Point-to-plane ICP, scan onto model, from the TEASER++ estimate and from 6 extra starts
   (cube centre 4 cm behind the crop centroid along the viewing ray, 6 fixed rotations).
   The result is the solution whose cube centre lies behind the observed surface (a
   visibility check: a few noisy faces are otherwise often fitted from the wrong side),
   with the best ICP fitness and then the lowest RMSE.

The cube's 24 symmetries need no special handling: any symmetric solution scores the same
in the metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import CUBE_CENTER, inv


@dataclass
class RegResult:
    T_lidar_model: np.ndarray
    fitness: float
    rmse: float
    n_inliers: int


def _pcd(p):
    import open3d as o3d

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(p, float)))


class FpfhTeaserIcp:
    def __init__(self, model_points: np.ndarray, voxel: float = 0.01, normal_radius: float = 0.03,
                 feat_radius: float = 0.05, k: int = 5, max_corr: int = 400, noise_bound: float = 0.02,
                 icp_dist: float = 0.03, icp_iters: int = 50):
        import open3d as o3d

        self.voxel, self.nr, self.fr, self.k, self.max_corr = voxel, normal_radius, feat_radius, k, max_corr
        self.noise_bound, self.icp_dist, self.icp_iters = noise_bound, icp_dist, icp_iters
        m = _pcd(model_points).voxel_down_sample(voxel)
        m.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        m.orient_normals_consistent_tangent_plane(10) if len(m.points) > 10 else None
        # CAD normals point outward: orient from the cube centre
        n, p = np.asarray(m.normals), np.asarray(m.points)
        flip = ((p - CUBE_CENTER) * n).sum(1) < 0
        n[flip] *= -1
        m.normals = o3d.utility.Vector3dVector(n)
        self.model = m
        self.model_feat = self._fpfh(m)
        from scipy.spatial import cKDTree

        self._ftree = cKDTree(self.model_feat)

    def _fpfh(self, pc):
        import open3d as o3d

        f = o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=self.fr, max_nn=100))
        return np.asarray(f.data).T

    def _teaser(self, src, dst):
        import teaserpp_python as tp

        prm = tp.RobustRegistrationSolverParams(
            noise_bound=self.noise_bound, cbar2=1.0, estimate_scaling=False,
            rotation_estimation_algorithm=tp.RotationEstimationAlgorithm.GNC_TLS,
            rotation_gnc_factor=1.4, rotation_max_iterations=100, rotation_cost_threshold=1e-12,
            inlier_selection_mode=tp.InlierSelectionMode.PMC_HEU, max_clique_exact_solution=False,
            max_clique_time_limit=1)
        solver = tp.RobustRegistrationSolver(*prm)
        solver.solve(src.T.astype(np.float64), dst.T.astype(np.float64))
        sol = solver.getSolution()
        T = np.eye(4)
        T[:3, :3] = sol.rotation
        T[:3, 3] = sol.translation
        return T, len(solver.getRotationInliers())

    def _icp(self, scan, T0):
        import open3d as o3d

        reg = o3d.pipelines.registration
        r = reg.registration_icp(scan, self.model, self.icp_dist, T0,
                                 reg.TransformationEstimationPointToPlane(),
                                 reg.ICPConvergenceCriteria(max_iteration=self.icp_iters))
        return r.transformation, r.fitness, r.inlier_rmse

    def __call__(self, pts_lidar: np.ndarray) -> RegResult | None:
        import open3d as o3d

        if len(pts_lidar) < 5:
            return None
        s = _pcd(pts_lidar).voxel_down_sample(self.voxel)
        s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.nr, max_nn=30))
        s.orient_normals_towards_camera_location(np.zeros(3))
        sp = np.asarray(s.points)
        feat = self._fpfh(s)
        fd, nn = self._ftree.query(feat, k=min(self.k, len(self.model_feat)))
        nn, fd = np.atleast_2d(nn), np.atleast_2d(fd)
        src = np.repeat(sp, nn.shape[1], 0)
        dst = np.asarray(self.model.points)[nn.reshape(-1)]
        if len(src) > self.max_corr:  # keep the best feature matches (bounds max-clique time)
            keep = np.argsort(fd.reshape(-1))[: self.max_corr]
            src, dst = src[keep], dst[keep]
        T_ml, n_in = self._teaser(src, dst)  # model <- lidar
        starts = []
        if n_in >= 3:
            starts.append(T_ml)
        # fallback / extra starts: centroid alignment with a few rotations
        from scipy.spatial.transform import Rotation

        c = sp.mean(0)
        ray = c / np.linalg.norm(c)
        for R in Rotation.from_euler("xyz", [[0, 0, 0], [45, 0, 0], [0, 45, 0], [0, 0, 45],
                                             [30, 30, 0], [0, 30, 30]], degrees=True).as_matrix():
            T = np.eye(4)
            T[:3, :3] = R
            # cube centre ~4 cm behind the visible-surface centroid, along the viewing ray
            T[:3, 3] = CUBE_CENTER - R @ (c + 0.04 * ray)
            starts.append(T)
        best = None
        for T0 in starts:
            T, fit, rmse = self._icp(s, T0)
            # visibility: the cube centre must lie behind the observed surface
            centre = inv(T)[:3, :3] @ CUBE_CENTER + inv(T)[:3, 3]
            behind = float(centre @ ray - c @ ray)
            key = (behind > 0.01, fit, -rmse)
            if best is None or key > best[0]:
                best = (key, T, fit, rmse)
        _, T, fit, rmse = best
        return RegResult(inv(T), float(fit), float(rmse), int(n_in))
