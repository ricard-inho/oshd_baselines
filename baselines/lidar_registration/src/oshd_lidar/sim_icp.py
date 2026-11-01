"""LiDAR-to-CAD ICP (single scan, training-free) on a simulated split (paper Table I, LiDAR row).

    uv run lidar-sim-icp --run-dir ../../results/sim/sim_zerog_docking_test_v2_lidar --setting ORACLE_SEG

Reads the shared sample list of ``oshd-bench sim-prepare --sensor lidar`` (``samples.parquet``:
trajectory, scan ``index``, target model, scale and ``target_points``, the simulator's labels of
the returns on the target: an **oracle segmentation**, like the semantic mask FoundationPose gets).
``gt.parquet`` is never opened. Per sample:

1. the target's returns of the scan, in the lidar sensor frame;
2. the target CAD (``objects.json``, scaled by the sample's ``scale``): ``--model-points`` surface
   points with normals;
3. **global search by hypotheses**: ``--hypotheses`` rotations (a fixed, uniform set on SO(3)),
   each with the scan centroid put at the CAD's bounding-box centre;
4. from each, point-to-plane ICP of the returns onto the CAD, coarse to fine (correspondence
   distances 15%, 5% and 2% of the CAD diameter, Tukey kernel);
5. the hypothesis with the highest fitness (share of returns within 2% of the diameter of the
   CAD surface; RMSE breaks ties) gives ``T_model_lidar``; the prediction is its inverse,
   ``T_lidar_target``, with the fitness as ``score``.

Predictions are written with ``oshd_bench.PoseWriter`` to ``--out-dir`` (default
``results/<run dir name>/`` in this baseline's folder); score them with ``oshd-bench sim-eval``.
``runtime_ms`` is the registration (steps 3-5) alone.
"""

from __future__ import annotations

import argparse
import json
import os
import time

# One thread per worker: OpenMP's spinning threads (x workers) would starve everything else.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
REPO = HERE.parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="lidar-sim-icp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(REPO / "results" / "sim" / "sim_zerog_docking_test_v2_lidar"))
    ap.add_argument("--out-dir", default=None, help="default: results/<run dir name>/ in this baseline's folder")
    ap.add_argument("--setting", default="ORACLE_SEG")
    ap.add_argument("--hypotheses", type=int, default=96, help="initial rotations (uniform on SO(3))")
    ap.add_argument("--model-points", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip trajectories already in the output file")
    return ap.parse_args(argv)


def rotation_set(n: int, seed: int) -> np.ndarray:
    """``n`` rotations spread uniformly over SO(3): the lowest-discrepancy pick of a large random
    pool (farthest-point sampling on the quaternion double cover)."""
    from scipy.spatial.transform import Rotation

    pool = Rotation.random(max(50 * n, 2000), random_state=seed).as_quat()
    chosen = [0]
    d = 1 - np.abs(pool @ pool[0])
    for _ in range(n - 1):
        k = int(d.argmax())
        chosen.append(k)
        d = np.minimum(d, 1 - np.abs(pool @ pool[k]))
    return Rotation.from_quat(pool[chosen]).as_matrix()


_MODELS: dict = {}


def _model(objects: dict, root: str, name: str, scale: float, n_points: int, seed: int):
    import open3d as o3d
    from oshd_bench.sim import load_cad

    key = (name, scale)
    if key not in _MODELS:
        mesh = load_cad(objects, name, root)
        mesh.apply_scale(scale)
        pts, face = mesh.sample(n_points, return_index=True, seed=seed)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pts)))
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.face_normals[face]))
        center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
        diameter = float(np.linalg.norm(np.ptp(mesh.vertices, 0)))
        _MODELS.clear()  # one trajectory's model at a time per worker
        _MODELS[key] = (pcd, center, diameter)
    return _MODELS[key]


def register(scan: np.ndarray, model, center: np.ndarray, diameter: float, rotations: np.ndarray):
    """Best ``T_model_scan`` over the hypotheses, its fitness and RMSE."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan))
    c = scan.mean(0)
    best = (-1.0, np.inf, np.eye(4))
    for R in rotations:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center - R @ c
        for frac in (0.15, 0.05, 0.02):
            d = frac * diameter
            est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=d))
            T = reg.registration_icp(src, model, d, T, est,
                                     reg.ICPConvergenceCriteria(max_iteration=30)).transformation
        ev = reg.evaluate_registration(src, model, 0.02 * diameter, T)
        if (ev.fitness, -ev.inlier_rmse) > (best[0], -best[1]):
            best = (ev.fitness, ev.inlier_rmse, T)
    return best


def run_trajectory(job) -> list[tuple]:
    traj_name, rows, objects, root, a = job
    from oshd import OshDataset

    ds = OshDataset(root)
    traj = ds[traj_name]
    rotations = rotation_set(a["hypotheses"], a["seed"])
    out = []
    for r in rows:
        scan = traj.lidar_scan(int(r["index"]))
        P = np.stack([scan["x"], scan["y"], scan["z"]], 1).astype(np.float64)[np.asarray(r["target_points"])]
        if len(P) < 3:
            continue
        model, center, diameter = _model(objects, root, r["object"], float(r["scale"]), a["model_points"], a["seed"])
        t0 = time.perf_counter()
        fitness, _, T_ms = register(P, model, center, diameter, rotations)
        ms = (time.perf_counter() - t0) * 1000
        out.append((traj_name, int(r["t_ns"]), r["object"], np.linalg.inv(T_ms), float(fitness), ms))
    return out


def main(argv=None) -> int:
    a = parse_args(argv)
    from oshd import OshDataset
    from oshd_bench import PoseWriter

    run = Path(a.run_dir)
    prep = json.loads((run / "prepare.json").read_text())
    objects = json.loads((run / "objects.json").read_text())
    samples = pd.read_parquet(run / "samples.parquet").iloc[::a.stride]
    if a.limit:
        samples = samples.iloc[:a.limit]
    root = str(OshDataset(os.environ.get("OSH_DATASET_ROOT") or None).root)
    out_dir = Path(a.out_dir) if a.out_dir else HERE / "results" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"method": "LiDAR-to-CAD ICP", "mode": "single scan, global hypotheses + point-to-plane ICP",
            "hypotheses": a.hypotheses, "model_points": a.model_points,
            "segmentation": "simulator labels of the returns on the target (oracle)", "input": "sim lidar scan",
            "runtime": "registration only", "prepare": prep, "setting": a.setting, "stride": a.stride,
            "limit": a.limit}
    out = out_dir / f"predictions_{a.setting}.parquet"
    writer = PoseWriter(out, meta)
    done_trajs: set = set()
    if a.resume and out.exists():
        old = pd.read_parquet(out)
        writer.rows = old.to_dict("records")
        done_trajs = set(old["trajectory"])  # saved after whole trajectories only
        print(f"resuming: {len(done_trajs)} trajectories done", flush=True)
    params = {"hypotheses": a.hypotheses, "model_points": a.model_points, "seed": a.seed}
    jobs = [(t, g.to_dict("records"), objects, root, params) for t, g in samples.groupby("trajectory", sort=False)
            if t not in done_trajs]
    t_start = time.time()
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(run_trajectory, jobs)):
            for traj, t_ns, obj, T, score, ms in res:
                writer.add(traj, "lidar", t_ns, obj, T, score, ms)
            done += len(res)
            if (i + 1) % 5 == 0:
                writer.save()
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                el = time.time() - t_start
                print(f"[{i + 1}/{len(jobs)}] {done} poses, {el / 60:.1f} min "
                      f"(ETA {(len(jobs) - i - 1) * el / (i + 1) / 60:.0f} min)", flush=True)
    writer.meta.update({"n_samples": len(samples), "n_predicted": len(writer), "wall_s": round(time.time() - t_start),
                        "workers": a.workers})
    print(f"{len(writer)} poses / {len(samples)} samples -> {writer.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
