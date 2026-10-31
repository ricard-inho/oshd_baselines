"""FoundationPose (model-based, RGB-D, single frame) on simulated trajectories.

Runs inside the Docker image (docker/run.sh) on the sample list written by
``fp_oshd.sim_prepare`` (``samples.parquet``; ``gt.parquet`` is never opened). For every sample:

1. RGB, depth (metres) and the target's semantic mask (class ``target``) of that frame; the
   simulated camera is an ideal pinhole, so nothing is undistorted. The mask is the
   simulator's, i.e. an **oracle 2D segmentation**: no detector is trained or run.
2. ``FoundationPose.register(K, rgb, depth, mask, iteration=--refine-iters)`` from scratch on
   **every frame** (global rotation hypotheses, refiner, scorer; ``track_one`` is never used).
   The CAD is the target's mesh (``objects.json``) scaled by the run's ``target_scale``.
3. The pose ``T_cam_target`` (rgb optical frame, target root frame) is written with
   ``oshd_bench.PoseWriter`` to ``<run dir>/predictions_<setting>.parquet``.

``runtime_ms`` is ``register()`` alone (no decoding).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FP_DIR = Path("/opt/FoundationPose")
SENSOR = "rgb"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="fp_oshd.predict_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="results/sim_zerog_docking_test",
                    help="folder of fp_oshd.sim_prepare (samples.parquet, objects.json)")
    ap.add_argument("--setting", default="ORACLE_MASK", help="name of the output (mask source)")
    ap.add_argument("--refine-iters", type=int, default=5, help="refiner iterations in register() (upstream: 5)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples (smoke tests)")
    ap.add_argument("--stride", type=int, default=1, help="every N-th sample")
    ap.add_argument("--resume", action="store_true", help="skip samples already in the output file")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--debug-every", type=int, default=0, help="write an overlay every N-th sample (0: none)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def bbox_corners(vertices: np.ndarray) -> np.ndarray:
    """8 box corners, corner k = (x[k&1], y[k>>1&1], z[k>>2&1]) (the order draw_overlay expects)."""
    lo, hi = vertices.min(0), vertices.max(0)
    return np.array([[(lo, hi)[k & 1][0], (lo, hi)[k >> 1 & 1][1], (lo, hi)[k >> 2 & 1][2]] for k in range(8)])


def main(argv=None) -> int:
    a = parse_args(argv)
    sys.path.insert(0, str(FP_DIR))
    import torch
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_seed  # upstream
    from oshd_bench import PoseWriter
    from oshd_dataset import OshdDataset

    from .geometry import draw_overlay

    logging.getLogger().setLevel(logging.WARNING)
    run = Path(a.run_dir)
    prep = json.loads((run / "prepare.json").read_text())
    objects = json.loads((run / "objects.json").read_text())
    samples = pd.read_parquet(run / "samples.parquet").iloc[::a.stride]
    if a.limit:
        samples = samples.iloc[:a.limit]
    ds = OshdDataset(prep["dataset_root"])

    out = run / f"predictions_{a.setting}.parquet"
    meta = {"method": "FoundationPose", "upstream": "https://github.com/NVlabs/FoundationPose",
            "commit": "a1b694b83e633c2cb6115b9063d940a687759392",
            "weights": {"refiner": "2023-10-28-18-33-37", "scorer": "2024-01-11-20-02-45"},
            "mode": "per-frame register() (no tracking)", "refine_iters": a.refine_iters,
            "mask": "simulator semantic mask of class 'target' (oracle)", "input": "sim rgb + depth, pinhole K",
            "runtime": "register() only", "prepare": prep, "setting": a.setting, "stride": a.stride,
            "limit": a.limit}
    writer = PoseWriter(out, meta)
    done: set = set()
    if a.resume and out.exists():
        old = pd.read_parquet(out)
        writer.rows = old.to_dict("records")
        done = set(zip(old.trajectory, old.t_ns.astype(int)))
        print(f"resuming: {len(done)} samples already done", flush=True)
    dbg = run / f"overlays_{a.setting}"
    if a.debug_every:
        dbg.mkdir(exist_ok=True)

    set_seed(a.seed)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    base = {k: trimesh.load(v["cad"], force="mesh") for k, v in objects.items()}
    est, est_key, cad_pts, corners = None, None, None, None
    n_skip = n_done = 0
    t_start = time.time()
    for k, s in enumerate(samples.itertuples()):
        if (s.trajectory, int(s.t_ns)) in done:
            continue
        key = (s.object, float(s.scale))
        if key != est_key:  # one estimator per (mesh, scale); samples are grouped by trajectory
            mesh = base[s.object].copy()
            mesh.apply_scale(s.scale)
            del est
            torch.cuda.empty_cache()
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                 scorer=scorer, refiner=refiner, glctx=glctx, debug=0, debug_dir="/tmp/fp_debug")
            est_key, cad_pts, corners = key, np.asarray(mesh.sample(5000)), bbox_corners(mesh.vertices)
            print(f"{s.object} x{s.scale}: {len(mesh.vertices)} vertices, diameter {est.diameter:.3f} m, "
                  f"{len(est.rot_grid)} rotation hypotheses", flush=True)
        traj = ds[s.trajectory]
        K = np.asarray(traj.calibration.intrinsics(SENSOR).K, float).reshape(3, 3)
        rgb = traj.frame(SENSOR, int(s.index))
        depth = traj.depth(int(s.index)).astype(np.float32)
        depth[~np.isfinite(depth)] = 0
        mask = traj.frame("semantic", int(s.index)) == traj.semantic_classes["target"]
        if int(((depth >= 0.001) & mask).sum()) < 4:
            n_skip += 1  # upstream would only guess a translation: no prediction
            continue
        t0 = time.perf_counter()
        T = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=a.refine_iters)
        ms = (time.perf_counter() - t0) * 1000
        score = float(est.scores[0])
        writer.add(s.trajectory, SENSOR, s.t_ns, s.object, T, score, ms)
        n_done += 1
        if n_done % 25 == 0 or n_done <= 3:
            el = time.time() - t_start
            print(f"[{k + 1}/{len(samples)}] {s.trajectory} #{s.index} score {score:.2f} {ms:.0f} ms "
                  f"({el / n_done:.2f} s/sample, ETA {(len(samples) - k - 1) * el / n_done / 60:.0f} min)", flush=True)
        if a.debug_every and n_done % a.debug_every == 1:
            import cv2

            img = draw_overlay(rgb, K, {"FoundationPose": T}, corners, mask, cad_pts,
                               f"{s.trajectory} #{s.index} score {score:.2f}")
            cv2.imwrite(str(dbg / f"{s.trajectory}_{int(s.index):05d}.jpg"), img)
        if n_done % a.save_every == 0:
            writer.save()
    writer.meta.update({"n_samples": len(samples), "n_no_depth_in_mask": n_skip, "n_predicted": len(writer),
                        "wall_s": round(time.time() - t_start)})
    print(f"{len(writer)} poses / {len(samples)} samples ({n_skip} without depth in the mask) -> {writer.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
