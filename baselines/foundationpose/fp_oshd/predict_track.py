"""FoundationPose in tracking mode (RGB-D) on simulated trajectories: drift over long sequences.

Runs inside the Docker image (docker/run.sh) on a shared sample list of ``oshd-bench
sim-prepare`` built at the full frame rate (``--stride 1``; ``gt.parquet`` is never opened).
The samples of a trajectory are split into **segments**: runs of consecutive rgb frames in which
the target is visible (mask ≥ ``min_mask_px``). In each segment:

1. the first frame where the target is **fully in view** (mask ≥ ``--init-min-px`` pixels and
   not touching the image border) is registered from scratch, ``register(K, rgb, depth, mask,
   iteration=--refine-iters)``, with the simulator's semantic mask (**oracle 2D segmentation**,
   used on this frame only). The frames before it get no prediction (they count as failures:
   ``coverage``). Registering on a target cut by the image border puts it metres off, and
   tracking never recovers from that;
2. every later frame is tracked, ``track_one(rgb, depth, K, iteration=--track-iters)``: the
   refiner starts from the previous pose. No mask, no re-initialisation, so errors accumulate.

A frame with fewer than 4 depth pixels in the mask is not registered either. The pose ``T_cam_target`` is written with ``oshd_bench.PoseWriter`` to
``--out-dir`` (default ``results/<run>/`` in this folder) as ``predictions_<setting>.parquet``;
``score`` is FoundationPose's score on registered frames and NaN on tracked ones. Score it with
``oshd-bench sim-eval``, and the drift with ``fp_oshd.drift``.

``runtime_ms`` is ``register()`` or ``track_one()`` alone (no decoding).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .predict_sim import FP_DIR, HERE, REPO, SENSOR, bbox_corners, untextured_to_vertex_colors


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="fp_oshd.predict_track", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(REPO / "results" / "sim" / "sim_zerog_inspection_test_v2"),
                    help="shared sample list of `oshd-bench sim-prepare --stride 1`")
    ap.add_argument("--out-dir", default=None, help="default: results/<run dir name>/ in this folder")
    ap.add_argument("--setting", default="TRACK_ORACLE_INIT", help="name of the output")
    ap.add_argument("--refine-iters", type=int, default=5, help="refiner iterations of register() (upstream: 5)")
    ap.add_argument("--track-iters", type=int, default=2, help="refiner iterations of track_one() (upstream: 2)")
    ap.add_argument("--init-min-px", type=int, default=2000,
                    help="register only once the target mask has this many pixels and doesn't touch the border")
    ap.add_argument("--limit-trajectories", type=int, default=None, help="only the first N trajectories")
    ap.add_argument("--traj-stride", type=int, default=1, help="every N-th trajectory (smoke tests)")
    ap.add_argument("--resume", action="store_true", help="skip trajectories already in the output file")
    ap.add_argument("--debug-every", type=int, default=0, help="write an overlay every N-th frame (0: none)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def segments(samples: pd.DataFrame) -> pd.Series:
    """Segment id per sample: a new segment starts at a trajectory's first sample and after
    every gap in the rgb frame index (the target was out of view)."""
    s = samples
    new = (s["trajectory"] != s["trajectory"].shift()) | (s["index"] != s["index"].shift() + 1)
    return new.cumsum() - 1


def fully_in_view(mask: np.ndarray, min_px: int, margin: int = 2) -> bool:
    """The target mask is large enough and doesn't touch the image border."""
    if int(mask.sum()) < min_px:
        return False
    return not (mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any() or mask[:, -margin:].any())


def main(argv=None) -> int:
    a = parse_args(argv)
    sys.path.insert(0, str(FP_DIR))
    import torch
    from estimater import (  # upstream
        FoundationPose,
        PoseRefinePredictor,
        ScorePredictor,
        dr,
        set_seed,
    )
    from oshd import OshDataset
    from oshd_bench import PoseWriter
    from oshd_bench.sim import load_cad

    from .geometry import draw_overlay

    logging.getLogger().setLevel(logging.WARNING)
    run = Path(a.run_dir)
    prep = json.loads((run / "prepare.json").read_text())
    if prep.get("stride") != 1:
        raise SystemExit(f"{run}: tracking needs the full frame rate (sim-prepare --stride 1), got {prep.get('stride')}")
    objects = json.loads((run / "objects.json").read_text())
    samples = pd.read_parquet(run / "samples.parquet")
    samples = samples.sort_values(["trajectory", "index"], kind="stable").reset_index(drop=True)
    trajs = list(dict.fromkeys(samples["trajectory"]))[::a.traj_stride][:a.limit_trajectories]
    samples = samples[samples["trajectory"].isin(trajs)].reset_index(drop=True)
    samples["segment"] = segments(samples)
    ds = OshDataset(os.environ.get("OSH_DATASET_ROOT") or None)
    out_dir = Path(a.out_dir) if a.out_dir else HERE / "results" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"predictions_{a.setting}.parquet"
    meta = {"method": "FoundationPose (tracking)", "upstream": "https://github.com/NVlabs/FoundationPose",
            "commit": "a1b694b83e633c2cb6115b9063d940a687759392",
            "weights": {"refiner": "2023-10-28-18-33-37", "scorer": "2024-01-11-20-02-45"},
            "mode": "register() on the first frame of each visible segment, then track_one()",
            "refine_iters": a.refine_iters, "track_iters": a.track_iters,
            "mask": "simulator semantic mask of class 'target' (oracle), on the registered frames only",
            "init": f"first frame of a segment with mask >= {a.init_min_px} px, not touching the image border",
            "input": "sim rgb + depth, pinhole K", "runtime": "register() / track_one() only", "prepare": prep,
            "setting": a.setting, "trajectories": trajs if (a.limit_trajectories or a.traj_stride > 1) else None}
    writer = PoseWriter(out, meta)
    done: set = set()
    if a.resume and out.exists():
        old = pd.read_parquet(out)
        done = set(old["trajectory"])  # the file is saved only after a trajectory is complete
        writer.rows = old.to_dict("records")
        print(f"resuming: {len(done)} trajectories already done", flush=True)
    dbg = out_dir / f"overlays_{a.setting}"
    if a.debug_every:
        dbg.mkdir(exist_ok=True)

    set_seed(a.seed)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    base = {k: untextured_to_vertex_colors(load_cad(objects, k, ds.root)) for k in objects}
    est, est_key, cad_pts, corners = None, None, None, None
    n_reg = n_track = n_frames = 0
    t_start = time.time()
    for ti, (tname, g) in enumerate(samples.groupby("trajectory", sort=False)):
        if tname in done:
            continue
        s0 = g.iloc[0]
        key = (s0.object, float(s0.scale))
        if key != est_key:
            mesh = base[s0.object].copy()
            mesh.apply_scale(s0.scale)
            del est
            torch.cuda.empty_cache()
            est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                 scorer=scorer, refiner=refiner, glctx=glctx, debug=0, debug_dir="/tmp/fp_debug")
            est_key, cad_pts, corners = key, np.asarray(mesh.sample(5000)), bbox_corners(mesh.vertices)
        traj = ds[tname]
        K = np.asarray(traj.calibration.intrinsics(SENSOR).K, float).reshape(3, 3)
        target_id = traj.semantic_classes["target"]
        t_traj = time.time()
        for _, seg in g.groupby("segment", sort=True):
            tracking = False
            for s in seg.itertuples():
                rgb = traj.frame(SENSOR, int(s.index))
                depth = traj.depth(int(s.index)).astype(np.float32)
                depth[~np.isfinite(depth)] = 0
                mask = None
                if not tracking:
                    mask = traj.frame("semantic", int(s.index)) == target_id
                    if not fully_in_view(mask, a.init_min_px) or int(((depth >= 0.001) & mask).sum()) < 4:
                        continue  # no prediction; register on a later frame
                    t0 = time.perf_counter()
                    T = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=a.refine_iters)
                    score = float(est.scores[0])
                    tracking = True
                    n_reg += 1
                else:
                    t0 = time.perf_counter()
                    T = est.track_one(rgb=rgb, depth=depth, K=K, iteration=a.track_iters)
                    score = float("nan")
                    n_track += 1
                ms = (time.perf_counter() - t0) * 1000
                writer.add(tname, SENSOR, s.t_ns, s.object, T, score, ms)
                n_frames += 1
                if a.debug_every and n_frames % a.debug_every == 1:
                    import cv2

                    if mask is None:
                        mask = traj.frame("semantic", int(s.index)) == target_id
                    img = draw_overlay(rgb, K, {"FoundationPose": T}, corners, mask, cad_pts,
                                       f"{tname} #{s.index} {'reg' if np.isfinite(score) else 'track'}")
                    cv2.imwrite(str(dbg / f"{tname}_{int(s.index):05d}.jpg"), img)
        writer.save()
        el = time.time() - t_start
        left = len(trajs) - ti - 1
        print(f"[{ti + 1}/{len(trajs)}] {tname}: {len(g)} frames, {g['segment'].nunique()} segments, "
              f"{time.time() - t_traj:.0f} s (ETA {left * el / max(1, ti + 1 - len(done)) / 60:.0f} min)", flush=True)
    writer.meta.update({"n_samples": len(samples), "n_predicted": len(writer), "n_registered": n_reg,
                        "n_tracked": n_track, "wall_s": round(time.time() - t_start)})
    print(f"{len(writer)} poses / {len(samples)} samples ({n_reg} registered, {n_track} tracked) -> {writer.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
