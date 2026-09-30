"""FoundationPose (model-based, RGB-D) and FoundationPose + ICP on the POOH-D sample lists.

Runs inside the Docker image (docker/run.sh). For every sample of
``results/gt/realsense_color/<domain>_<split>.parquet`` (the ``T`` column is never read):

1. take the best 2D detection of the object (``--detector maskrcnn`` -> the Mask R-CNN of
   ``baselines/detector`` trained ZS / FT; ``--detector oracle`` -> GT-rendered masks),
2. RGB + RealSense depth registered into the colour frame (``pooh_bench.sensors.depth_in_camera``),
   undistorted to the pinhole camera with the same K,
3. ``FoundationPose.register()`` per frame (global pose hypotheses + ``--refine-iters`` refiner
   iterations + scorer). There is **no tracking**: every frame is registered from scratch,
4. optionally (row ``foundationpose_icp``) point-to-plane ICP of the visible CAD points against
   the depth points inside the detected mask, initialised at the FoundationPose pose.

``runtime_ms`` = detector runtime (from the detections file) + undistortion + register (+ ICP).
Decoding the frames and registering depth into the colour frame are not timed (the RealSense
does the latter on-device when it streams aligned depth).
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

MODALITY = "rgbd_realsense"
SENSOR = "realsense_color"
FP_DIR = Path("/opt/FoundationPose")
SAMPLE_COLUMNS = ["trajectory", "domain", "sensor", "index", "t_ns", "object"]  # never "T"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="fp_pooh.predict", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True,
                    help="output setting: ZS / FT (detector trained on synthetic / + real), or e.g. ORACLE")
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", default="real", choices=["real", "synthetic"])
    ap.add_argument("--detector", default="maskrcnn", choices=["maskrcnn", "oracle"])
    ap.add_argument("--det-setting", default=None,
                    help="setting of the detections file (default: --setting for maskrcnn, GT for oracle)")
    ap.add_argument("--methods", default="foundationpose,foundationpose_icp",
                    help="comma list of rows to write: foundationpose, foundationpose_icp")
    ap.add_argument("--refine-iters", type=int, default=5, help="refiner iterations in register() (upstream: 5)")
    ap.add_argument("--min-score", type=float, default=0.0, help="ignore detections below this score")
    ap.add_argument("--limit", type=int, default=None, help="only the first N samples (smoke tests)")
    ap.add_argument("--stride", type=int, default=1, help="every N-th sample")
    ap.add_argument("--debug-dir", default=None, help="write overlays here")
    ap.add_argument("--debug-every", type=int, default=1, help="overlay every N-th processed sample")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def best_detections(path: Path, min_score: float) -> dict:
    if not path.exists():
        sys.exit(f"detections not found: {path}\n(run the detector, or `pooh-bench oracle-detections`)")
    det = pd.read_parquet(path)
    det = det[det["score"] >= min_score].sort_values("score", ascending=False)
    det = det.drop_duplicates(["trajectory", "t_ns", "object"])
    return {(r.trajectory, int(r.t_ns), r.object): r for r in det.itertuples()}


def main(argv=None) -> int:
    a = parse_args(argv)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    assert set(methods) <= {"foundationpose", "foundationpose_icp"}, methods

    sys.path.insert(0, str(FP_DIR))
    import torch  # noqa: F401
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_seed  # upstream

    from pooh_bench import PoseWriter, detection_path, gt_path, open_dataset, prediction_path, results_dir
    from pooh_bench.predictions import rle_to_mask
    from pooh_bench.sensors import depth_in_camera

    from .geometry import Undistorter, draw_overlay, icp_refine, load_cad

    logging.getLogger().setLevel(logging.WARNING)
    res = results_dir()
    ds = open_dataset()
    samples = pd.read_parquet(gt_path(res, SENSOR, a.domain, a.split), columns=SAMPLE_COLUMNS)
    samples = samples.iloc[::a.stride]
    if a.limit:
        samples = samples.iloc[:a.limit]
    det_setting = a.det_setting or ("GT" if a.detector == "oracle" else a.setting)
    det_file = detection_path(res, SENSOR, a.detector, det_setting, a.domain, a.split)
    dets = best_detections(det_file, a.min_score)
    det_meta_p = det_file.with_suffix(".json")
    det_meta = json.loads(det_meta_p.read_text()) if det_meta_p.exists() else {}

    set_seed(a.seed)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimators, cad = {}, {}
    for name in sorted(samples["object"].unique()):
        obj = ds.objects[name]
        mesh = load_cad(obj)
        pts, fidx = mesh.sample(20000, return_index=True)
        cad[name] = dict(mesh=mesh, pts=np.asarray(pts), normals=np.asarray(mesh.face_normals[fidx]),
                         corners=np.asarray(obj.click_points), diameter=float(obj.diameter))
        estimators[name] = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                          scorer=scorer, refiner=refiner, glctx=glctx, debug=0,
                                          debug_dir="/tmp/fp_debug")
        print(f"{name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
              f"extent {np.ptp(mesh.vertices, 0).round(4)} m, diameter {estimators[name].diameter:.3f} m, "
              f"{len(estimators[name].rot_grid)} rotation hypotheses", flush=True)

    meta = {"method": "FoundationPose", "upstream": "https://github.com/NVlabs/FoundationPose",
            "commit": "a1b694b83e633c2cb6115b9063d940a687759392",
            "weights": {"refiner": "2023-10-28-18-33-37", "scorer": "2024-01-11-20-02-45"},
            "mode": "per-frame register() (no tracking)", "refine_iters": a.refine_iters,
            "detector": a.detector, "detector_setting": det_setting, "detections": str(det_file),
            "detector_meta": det_meta, "input": "realsense_color RGB + realsense_depth registered into colour, "
            "undistorted (same K)", "runtime": "detector + undistort + register (+ ICP); excludes decoding and "
            "depth registration", "split": a.split, "domain": a.domain}
    writers = {m: PoseWriter(prediction_path(res, MODALITY, m, a.setting, a.domain, a.split),
                             {**meta, "row": m, **({"icp": "Open3D point-to-plane, Tukey, 2/1/0.5 cm, visible CAD "
                                                   "points vs masked depth"} if m.endswith("icp") else {})})
               for m in methods}

    und_cache: dict = {}
    dbg = Path(a.debug_dir) if a.debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)
    n_det = n_depth = n_valid = n_icp_ok = done = 0
    t_start = time.time()
    for k, s in enumerate(samples.itertuples()):
        key = (s.trajectory, int(s.t_ns), s.object)
        d = dets.get(key)
        if d is None or d.mask_rle is None:
            continue
        n_det += 1
        traj = ds[s.trajectory]
        intr = traj.calibration.intrinsics(SENSOR)
        depth = depth_in_camera(traj, SENSOR, int(s.t_ns))
        if depth is None:
            continue
        n_depth += 1
        rgb = traj.frame(SENSOR, int(s.index))
        mask = rle_to_mask(d.mask_rle)
        det_ms = float(d.runtime_ms) if np.isfinite(d.runtime_ms) else 0.0

        t0 = time.perf_counter()
        if s.trajectory not in und_cache:
            und_cache[s.trajectory] = Undistorter(intr)
        und = und_cache[s.trajectory]
        K = und.K
        rgb_u = und(rgb)
        depth_u = und(depth, nearest=True)
        mask_u = und(mask.astype(np.uint8), nearest=True).astype(bool)
        if not valid_register(depth_u, mask_u):
            continue  # upstream would return an identity-rotation guess; count it as no prediction
        n_valid += 1
        est = estimators[s.object]
        T_fp = est.register(K=K, rgb=rgb_u, depth=depth_u, ob_mask=mask_u, iteration=a.refine_iters)
        score = float(est.scores[0])
        t_fp = (time.perf_counter() - t0) * 1000
        if "foundationpose" in writers:
            writers["foundationpose"].add(s.trajectory, SENSOR, s.t_ns, s.object, T_fp, score, det_ms + t_fp)
        T_icp, info = None, ""
        if "foundationpose_icp" in writers:
            t1 = time.perf_counter()
            c = cad[s.object]
            T_icp, ii = icp_refine(T_fp, c["pts"], c["normals"], depth_u, mask_u, K, c["diameter"])
            n_icp_ok += ii["accepted"]
            t_icp = (time.perf_counter() - t1) * 1000
            writers["foundationpose_icp"].add(s.trajectory, SENSOR, s.t_ns, s.object, T_icp, score,
                                              det_ms + t_fp + t_icp)
            info = (f" icp {'ok' if ii['accepted'] else 'REJECTED'} fit {ii['fitness']:.2f} "
                    f"rmse {ii['rmse'] * 1000:.1f}mm n {ii['n_target']} ({t_icp:.0f} ms)")
        done += 1
        if done % 25 == 0 or done <= 3:
            print(f"[{k + 1}/{len(samples)}] {s.trajectory} t={s.t_ns} score {score:.2f} "
                  f"register {t_fp:.0f} ms{info}", flush=True)
        if dbg and (done - 1) % a.debug_every == 0:
            import cv2

            poses = {"FoundationPose": T_fp, **({"+ICP": T_icp} if T_icp is not None else {})}
            img = draw_overlay(rgb_u, K, poses, cad[s.object]["corners"], mask_u, cad[s.object]["pts"],
                               f"{s.trajectory} #{s.index} {a.detector}/{det_setting} score {score:.2f}")
            cv2.imwrite(str(dbg / f"{s.trajectory}_{int(s.index):05d}.jpg"), img)

    for m, w in writers.items():
        w.meta.update({"n_samples": len(samples), "n_with_detection": n_det, "n_with_depth": n_depth,
                       "n_with_depth_in_mask": n_valid,
                       **({"n_icp_accepted": n_icp_ok} if m.endswith("icp") else {})})
        print(f"{m}: {len(w)} poses / {len(samples)} samples -> {w.save()}")
    print(f"total {time.time() - t_start:.0f} s")
    return 0


def valid_register(depth, mask) -> bool:
    """register() needs depth inside the mask; with < 4 valid pixels it only guesses a translation."""
    return int(((depth >= 0.001) & (mask > 0)).sum()) >= 4


if __name__ == "__main__":
    raise SystemExit(main())
