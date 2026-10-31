"""Mesh loading, image undistortion, the ICP refinement and debug overlays (no FoundationPose imports)."""

from __future__ import annotations

import cv2
import numpy as np
import trimesh

__all__ = ["load_cad", "Undistorter", "icp_refine", "visible_model_points", "draw_overlay"]


def load_cad(obj) -> trimesh.Trimesh:
    """The oshd-dataset ObjectModel as a trimesh FoundationPose can use.

    ``obj.vertices`` are already in metres (``mesh_scale`` applied by oshd-dataset); the file is
    reloaded only for its material colours, which become vertex colours (the OBJ has
    per-group ``Kd`` colours but no texture image, which FoundationPose's renderer can't use).
    """
    v = np.asarray(obj.vertices, np.float64)
    f = np.asarray(obj.faces, np.int64)
    ext = v.max(0) - v.min(0)
    if not 0.01 < ext.max() < 2.0:
        raise ValueError(f"{obj.name}: mesh extent {ext} m does not look like metres")
    colors = None
    if obj.mesh_path is not None:
        try:
            m = trimesh.load(obj.mesh_path, force="mesh", process=False)
            if len(m.vertices) == len(v):
                colors = np.asarray(m.visual.to_color().vertex_colors)[:, :3]
        except Exception:  # colours are cosmetic; fall back to grey
            colors = None
    if colors is None:
        colors = np.full((len(v), 3), 160, np.uint8)
    mesh = trimesh.Trimesh(v, f, vertex_colors=colors, process=False)
    return mesh


class Undistorter:
    """Maps a distorted image of ``intr`` to the pinhole image with the same K."""

    def __init__(self, intr):
        self.K = np.asarray(intr.K, np.float64)
        self.size = (intr.width, intr.height)
        self.identity = not np.any(np.asarray(intr.D))
        if not self.identity:
            self.m1, self.m2 = cv2.initUndistortRectifyMap(self.K, np.asarray(intr.D, np.float64), None, self.K,
                                                           self.size, cv2.CV_32FC1)

    def __call__(self, img: np.ndarray, nearest: bool = False) -> np.ndarray:
        if self.identity:
            return img
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.remap(img, self.m1, self.m2, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _project(K, pts):
    z = np.maximum(pts[:, 2], 1e-9)
    return np.stack([K[0, 0] * pts[:, 0] / z + K[0, 2], K[1, 1] * pts[:, 1] / z + K[1, 2]], -1)


def visible_model_points(model_pts: np.ndarray, model_normals: np.ndarray, T: np.ndarray, K: np.ndarray,
                         hw: tuple[int, int], tol: float = 0.003):
    """Subset of CAD surface samples (model frame) visible from the camera at pose ``T``.

    Point z-buffer with a 3x3 min filter (fills the gaps between samples) + a depth tolerance.
    """
    H, W = hw
    pc = model_pts @ T[:3, :3].T + T[:3, 3]
    uv = np.round(_project(K, pc)).astype(np.int64)
    ok = (pc[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    z = np.full(H * W, np.inf, np.float32)
    lin = uv[ok, 1] * W + uv[ok, 0]
    np.minimum.at(z, lin, pc[ok, 2].astype(np.float32))
    z = z.reshape(H, W)
    z = -cv2.dilate(-np.where(np.isfinite(z), z, 1e6).astype(np.float32), np.ones((3, 3), np.uint8))
    keep = np.zeros(len(pc), bool)
    keep[ok] = pc[ok, 2] <= z.reshape(-1)[lin] + tol
    return model_pts[keep], model_normals[keep]


def icp_refine(T_init: np.ndarray, model_pts: np.ndarray, model_normals: np.ndarray, depth: np.ndarray,
               mask: np.ndarray, K: np.ndarray, diameter: float,
               max_dists=(0.02, 0.01, 0.005), iters: int = 30, min_fitness: float = 0.3,
               max_rot_deg: float = 30.0):
    """Point-to-plane ICP (Open3D) of the visible CAD points against the masked depth points.

    Target: depth pixels inside the detector mask, kept only within ``max_dists[0]`` of the
    visible model surface at ``T_init`` (removes background seen through mask leaks / depth
    holes). Coarse-to-fine correspondence distances with a Tukey kernel. The result is
    rejected (``T_init`` returned, ``accepted`` False) when the final target inlier ratio is
    below ``min_fitness`` or clearly below the initial one, or it moves more than half the diameter / ``max_rot_deg``.

    Returns (T, info) with info = {fitness (target inlier ratio at max_dists[1]), fitness_init,
    rmse, n_target, accepted, T_unchecked}.
    """
    import open3d as o3d

    info = {"fitness": 0.0, "rmse": float("nan"), "n_target": 0, "accepted": False}
    vs, us = np.nonzero(mask & (depth > 0.001))
    z = depth[vs, us].astype(np.float64)
    tgt = np.stack([(us - K[0, 2]) / K[0, 0] * z, (vs - K[1, 2]) / K[1, 1] * z, z], -1)
    src_p, src_n = visible_model_points(model_pts, model_normals, T_init, K, depth.shape)
    if len(tgt) < 30 or len(src_p) < 30:
        info["n_target"] = len(tgt)
        return T_init, info
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_p))
    src.normals = o3d.utility.Vector3dVector(src_n)
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt)).voxel_down_sample(0.002)
    near = np.asarray(dst.compute_point_cloud_distance(
        o3d.geometry.PointCloud(src).transform(T_init))) < max_dists[0]
    dst = dst.select_by_index(np.nonzero(near)[0])
    info["n_target"] = len(dst.points)
    if len(dst.points) < 30:
        return T_init, info
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    dst.orient_normals_towards_camera_location(np.zeros(3))
    T = T_init.copy()
    res = None
    for d in max_dists:
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=d))
        res = o3d.pipelines.registration.registration_icp(
            src, dst, d, T, est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        T = np.asarray(res.transformation).copy()
    def target_inliers(T_, d_):
        dd = np.asarray(dst.compute_point_cloud_distance(o3d.geometry.PointCloud(src).transform(T_)))
        return float((dd < d_).mean())

    # Open3D's fitness is relative to the (whole visible) model; use the target's inlier ratio
    fit0, fit = target_inliers(T_init, max_dists[1]), target_inliers(T, max_dists[1])
    info.update(fitness=fit, fitness_init=fit0, rmse=float(res.inlier_rmse), T_unchecked=T)
    dR = T_init[:3, :3].T @ T[:3, :3]
    rot = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    if fit < min_fitness or fit < fit0 - 0.05 or np.linalg.norm(T[:3, 3] - T_init[:3, 3]) > diameter / 2 \
            or rot > max_rot_deg:
        return T_init, info
    info["accepted"] = True
    return T, info


def draw_overlay(rgb: np.ndarray, K: np.ndarray, poses: dict, corners: np.ndarray, mask: np.ndarray | None,
                 model_pts: np.ndarray, text: str = "") -> np.ndarray:
    """RGB with the detector mask contour (yellow) and, per pose, projected CAD points + cube edges."""
    img = np.ascontiguousarray(rgb[..., ::-1]).copy()  # BGR for cv2
    if mask is not None and mask.any():
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 1)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    for k, (name, T) in enumerate(poses.items()):
        col = colors[k % len(colors)]
        pc = model_pts @ T[:3, :3].T + T[:3, 3]
        pc = pc[pc[:, 2] > 0.01]
        for u, v in np.round(_project(K, pc)).astype(int)[::4]:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                img[v, u] = col
        cc = corners @ T[:3, :3].T + T[:3, 3]
        if (cc[:, 2] > 0.01).all():
            uv = np.round(_project(K, cc)).astype(int)
            for i, j in edges:
                cv2.line(img, tuple(map(int, uv[i])), tuple(map(int, uv[j])), col, 1, cv2.LINE_AA)
        cv2.putText(img, name, (8, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if text:
        cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img
