"""DEV ONLY: estimate a better T_body_lidar than the dev placeholder, from the cube itself.

The cube is static in the world while the sensor stack moves, so the mocap cube track seen
from the SensorStack body sweeps a planar region. We

1. find the cube cluster in each scan near where the current extrinsic predicts it
   (DBSCAN inside a shrinking radius), and fit T_body_lidar by Kabsch between the cluster
   centroids (lidar frame) and the mocap cube centres (body frame), iterating;
2. refine all 6 DoF (and optionally a lidar-mocap time offset) by minimising the robust
   point-to-box distance of the cube points of every scan to the mocap cube.

This uses ground truth and all trajectories, so it is a *calibration* step, never part of a
baseline. The result is written into a COPY of the dev dataset (``--out``, default
``work/lidar_dev/dataset``) whose other entries are symlinks; the real dataset is untouched.

    lidar-extrinsic [--out work/lidar_dev/dataset] [--stride 2]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .common import CUBE_CENTER, OBJECT, box_sdf, inv, transform


def _kabsch(A: np.ndarray, B: np.ndarray, w=None) -> np.ndarray:
    """T with B ~ T A."""
    w = np.ones(len(A)) if w is None else w
    ca, cb = (w[:, None] * A).sum(0) / w.sum(), (w[:, None] * B).sum(0) / w.sum()
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cb - R @ ca
    return T


def _cluster_near(p: np.ndarray, c: np.ndarray, radius: float, eps: float = 0.03):
    from sklearn.cluster import DBSCAN

    d = np.linalg.norm(p - c, axis=1)
    q = p[d < radius]
    if len(q) < 5:
        return None
    lab = DBSCAN(eps=eps, min_samples=3).fit_predict(q)
    best, bd = None, np.inf
    for k in set(lab) - {-1}:
        m = q[lab == k]
        ext = np.ptp(m, 0).max()
        if ext > 0.25 or len(m) < 4:
            continue
        dd = np.linalg.norm(m.mean(0) - c)
        if dd < bd:
            best, bd = m, dd
    return best


def collect(ds, stride: int):
    """Per scan: (points, T_body_model(t), t_ns, trajectory)."""
    from oshd.labels import object_pose
    from oshd_bench.sensors import lidar_points

    obj = ds.objects[OBJECT]
    out = []
    for tn in ds.local_trajectory_names:
        tr = ds[tn]
        if not (tr.has("lidar") and tr.has("mocap")):
            continue
        ts = tr.timestamps("lidar")[::stride]
        idx = np.arange(len(tr.timestamps("lidar")))[::stride]
        Twb, ok1 = tr.body_pose(tr.calibration.reference_body, ts)
        Twm, ok2 = object_pose(tr, obj, ts)
        for k in np.nonzero(ok1 & ok2)[0]:
            p, _ = lidar_points(tr, int(idx[k]))
            out.append(dict(traj=tn, t=int(ts[k]), p=p, T_bm=inv(Twb[k]) @ Twm[k]))
    return out


def estimate(scans, X0: np.ndarray, radii=(0.4, 0.3, 0.2, 0.15, 0.12, 0.12)) -> tuple[np.ndarray, dict]:
    X = X0.copy()
    for r in radii:
        A, B = [], []
        for s in scans:
            c_b = transform(s["T_bm"], CUBE_CENTER[None])[0]
            c_l = transform(inv(X), c_b[None])[0]
            if not 0.15 < np.linalg.norm(c_l) < 3.0:
                continue
            m = _cluster_near(s["p"], c_l, r)
            if m is None:
                continue
            # visible-surface centroid is ~half a face nearer to the sensor than the centre
            cen = m.mean(0)
            cen = cen + 0.025 * cen / np.linalg.norm(cen)
            A.append(cen)
            B.append(c_b)
        A, B = np.array(A), np.array(B)
        w = np.ones(len(A))
        for _ in range(5):  # IRLS (Cauchy)
            X = _kabsch(A, B, w)
            res = np.linalg.norm(transform(X, A) - B, axis=1)
            w = 1.0 / (1.0 + (res / 0.03) ** 2)
        print(f"radius {r:.2f}: {len(A)} scans, centroid residual median {np.median(res) * 100:.1f} cm")

    X_kabsch = X.copy()
    # fine: point-to-box over cube points of all scans
    P, Tbm = [], []
    for s in scans:
        c_b = transform(s["T_bm"], CUBE_CENTER[None])[0]
        c_l = transform(inv(X), c_b[None])[0]
        m = _cluster_near(s["p"], c_l, 0.12)
        if m is None:
            continue
        P.append(m)
        Tbm.append(np.repeat(inv(s["T_bm"])[None], len(m), 0))
    P, Tmb = np.concatenate(P), np.concatenate(Tbm)

    def unpack(x):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix() @ X[:3, :3]
        T[:3, 3] = X[:3, 3] + x[3:]
        return T

    def resid(x):
        T = unpack(x)
        pb = transform(T, P)
        pm = np.einsum("nij,nj->ni", Tmb[:, :3, :3], pb) + Tmb[:, :3, 3]
        return box_sdf(pm)

    r0 = np.abs(resid(np.zeros(6)))
    sol = least_squares(resid, np.zeros(6), loss="cauchy", f_scale=0.02)
    r1 = np.abs(resid(sol.x))
    if np.median(r1) < np.median(r0):
        X = unpack(sol.x)
    else:  # refinement did not help (sensor noise ~2 cm): keep the Kabsch solution
        r1 = r0
    info = {"n_points": len(P), "sdf_median_cm_before": float(np.median(r0) * 100),
            "sdf_median_cm_after": float(np.median(r1) * 100),
            "sdf_p90_cm_after": float(np.percentile(r1, 90) * 100), "T_kabsch": X_kabsch.tolist()}
    print(f"point-to-box: {len(P)} pts, median |sdf| {info['sdf_median_cm_before']:.2f} -> "
          f"{info['sdf_median_cm_after']:.2f} cm (p90 {info['sdf_p90_cm_after']:.2f} cm)")
    return X, info


def write_dataset(src: Path, out: Path, X: np.ndarray, info: dict) -> None:
    from oshd.geometry import rotmat_to_quat

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for p in src.iterdir():
        if p.name == "calibration":
            continue
        if p.is_symlink() or p.is_dir():
            (out / p.name).symlink_to(p.resolve())
        else:
            shutil.copy(p, out / p.name)
    shutil.copytree(src / "calibration", out / "calibration")
    rig_p = out / "calibration" / "rig.yaml"
    rig = yaml.safe_load(rig_p.read_text())
    rig["extrinsics"]["cameras"]["lidar"] = {
        "T_body_cam": {"translation": [float(v) for v in X[:3, 3]],
                       "rotation_xyzw": [float(v) for v in rotmat_to_quat(X[:3, :3])]},
        "source": "DEV ONLY (baselines/lidar_registration lidar-extrinsic): fitted to the mocap cube "
                  f"track, median point-to-box {info['sdf_median_cm_after']:.2f} cm",
    }
    rig_p.write_text("# DEV ONLY: PLACEHOLDER calibration + lidar extrinsic fitted by lidar-extrinsic\n"
                     + yaml.safe_dump(rig, sort_keys=False))
    (out / "calibration" / "lidar_extrinsic_fit.json").write_text(json.dumps(info, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="work/lidar_dev/dataset")
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args(argv)
    import oshd_bench

    ds = oshd_bench.open_dataset()
    src = Path(os.environ.get("OSH_DATASET_ROOT", "")).resolve()
    if not (src / "calibration" / "rig.yaml").exists() or "dev" not in str(src):
        raise SystemExit("set OSH_DATASET_ROOT to the dev dataset (this tool must not touch the real one)")
    X0 = ds[ds.local_trajectory_names[0]].calibration.T_body_cam("lidar")
    scans = collect(ds, a.stride)
    print(f"{len(scans)} scans with mocap")
    X, info = estimate(scans, X0)
    d = inv(X0) @ X
    info.update(delta_translation_cm=float(np.linalg.norm(d[:3, 3]) * 100),
                delta_rotation_deg=float(np.degrees(Rotation.from_matrix(d[:3, :3]).magnitude())),
                T_body_lidar=X.tolist())
    print(f"change vs placeholder: {info['delta_translation_cm']:.1f} cm, {info['delta_rotation_deg']:.1f} deg")
    write_dataset(src, Path(a.out), X, info)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
