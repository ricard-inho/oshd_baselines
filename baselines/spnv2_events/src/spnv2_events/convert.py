"""oshd-bench sim training list -> SPEED+-style training folder that upstream's SPEEDPLUSDataset reads.

Layout (``<work>/data``)::

    speedplus_events/
        camera.json  keypoints.mat  model.ply          # virtual camera, 8 corners [3, 8] (net frame, unit scale), model (mm)
        vcam.json  canon.json  target.json            # virtual camera, symmetry reference orientation, target frame
        <run>/                                        # e.g. sim_zerog_docking_train_v3
            images_640x512_RGB/<traj>_<index>.png     # time surface (50 ms window ending at t_ns), x2 + padded
            masks_160x128/<traj>_<index>.png          # the simulator's semantic mask of the target, 1/4 resolution
            labels/labels.csv                         # upstream CSV row format (tools/preprocess.py)

CSV row: filename, bbox (xmin, ymin, xmax, ymax, normalised), q (qw, qx, qy, qz of R_cam_net),
t (3, metres, bounding-box centre), origin (2, normalised), 8 keypoints (x, y normalised).

Symmetry: the pose label is the equivalent (identity or one of the measured 180 deg flips)
closest, in the viewing-ray frame, to a fixed reference orientation (``canon.json``, the chordal
mean of the training set converted first). Keypoint identities are then a function of the image
appearance, except near the cell boundaries. See README.md.

This reads the GT ``T`` column of the run, so it is only ever run on *train* lists.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import (
    OBJECT,
    SENSOR,
    R_to_q_wxyz,
    TargetFrame,
    VirtualCamera,
    Warper,
    canonicalize,
    estimate_reference,
    load_target,
    open_dataset,
    target_to_net,
    time_surface_image,
    window_events,
    work_dir,
    write_camera_files,
)


def data_root() -> Path:
    return work_dir() / "data"


def sensor_folder() -> Path:
    return data_root() / f"speedplus_{SENSOR}"


def image_folder_name(vcam: VirtualCamera) -> str:
    return f"images_{vcam.width}x{vcam.height}_RGB"


def mask_folder_name(vcam: VirtualCamera, stride: int = 4) -> str:
    return f"masks_{vcam.width // stride}x{vcam.height // stride}"


def load_or_make_vcam(ds, traj_name: str) -> VirtualCamera:
    """The virtual camera is fixed on first use (first trajectory's intrinsics) and reused."""
    p = sensor_folder() / "vcam.json"
    if p.exists():
        return VirtualCamera(**json.loads(p.read_text()))
    vcam = VirtualCamera.from_intrinsics("rgb", ds[traj_name].calibration.intrinsics("rgb"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(vcam.to_dict(), indent=2))
    return vcam


def load_or_make_target(ds, traj_name: str) -> TargetFrame:
    p = sensor_folder() / "target.json"
    tf = load_target(ds, traj_name)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(tf.to_dict(), indent=2))
    return tf


def read_train_list(run_dir: Path, stride: int = 1, limit: int | None = None) -> pd.DataFrame:
    gt = pd.read_parquet(run_dir / "gt.parquet")
    gt = gt[gt["object"] == OBJECT].iloc[::stride]
    if limit:
        gt = gt.iloc[:: max(1, len(gt) // limit)].iloc[:limit]  # spread over the trajectories
    if not len(gt):
        raise SystemExit(f"no {OBJECT} samples in {run_dir}")
    return gt


def sample_box(tf: TargetFrame, T_cam_target: np.ndarray, scale: float, intr) -> np.ndarray | None:
    """[x0, y0, x1, y1] of the projected (scaled) CAD vertices, clipped to the image, or None."""
    from oshd.geometry import transform_points

    uv, ok = intr.project(transform_points(T_cam_target, tf.vertices * scale))
    if not ok.any():
        return None
    uv = uv[ok]
    box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    box = np.clip(box, 0, [intr.width, intr.height, intr.width, intr.height])
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def sample_label(tf: TargetFrame, R_ref: np.ndarray, T_cam_target: np.ndarray, scale: float, vi) -> tuple | None:
    """(T_cam_net canonical, box px, origin px (2,), keypoints px (8, 2)) in the virtual camera;
    keypoints behind the camera are (-1, -1)."""
    from oshd.geometry import transform_points

    box = sample_box(tf, T_cam_target, scale, vi)
    if box is None:
        return None
    T_cn = canonicalize(target_to_net(T_cam_target, tf, scale), R_ref, tf.rots)
    uv, ok = vi.project(transform_points(T_cn, np.vstack([np.zeros(3), tf.keypoints(scale)])))
    # corners behind the camera (close range: the mockup is 1.3 m long) have no image position;
    # an out-of-image label makes upstream's HeatmapGenerator skip the keypoint
    uv[~ok] = -1.0
    return T_cn, box, uv[0], uv[1:]


# ------------------------------------------------------------------ worker
_W: dict = {}


def _init(vcam_d, R_ref, traj0, img_dir, mask_dir):
    ds = open_dataset()
    vcam = VirtualCamera(**vcam_d)
    _W.update(ds=ds, tf=load_target(ds, traj0), vcam=vcam, warp=Warper(vcam),
              warp_mask=Warper(vcam.scaled(0.25), cv2.INTER_NEAREST), R_ref=np.asarray(R_ref),
              img_dir=Path(img_dir), mask_dir=Path(mask_dir), vi=vcam.intrinsics())


def _one(row: tuple) -> list | None:
    traj_name, index, t_ns, scale, T_flat = row
    w = _W
    traj = w["ds"][traj_name]
    vcam, vi = w["vcam"], w["vi"]
    T_ct = np.asarray(T_flat, float).reshape(4, 4)
    lab = sample_label(w["tf"], w["R_ref"], T_ct, float(scale), vi)
    if lab is None:
        return None
    T_cn, box, origin, uv = lab
    intr = traj.calibration.intrinsics("rgb")
    fname = f"{traj_name}_{int(index):06d}"
    img = w["warp"](time_surface_image(window_events(traj, t_ns), int(t_ns), intr.height, intr.width), intr)
    cv2.imwrite(str(w["img_dir"] / f"{fname}.png"), img)
    target_id = traj.semantic_classes["target"]
    mask = (traj.frame("semantic", int(index)) == target_id).astype(np.uint8) * 255
    cv2.imwrite(str(w["mask_dir"] / f"{fname}.png"), w["warp_mask"](mask, intr))
    wh = np.array([vcam.width, vcam.height], float)
    return [f"{fname}.png", *(box / np.tile(wh, 2)).tolist(), *R_to_q_wxyz(T_cn[:3, :3]).tolist(),
            *T_cn[:3, 3].tolist(), *(origin / wh).tolist(), *(uv / wh).reshape(-1).tolist()]


# ------------------------------------------------------------------ main
def convert(run_dir: Path, stride: int = 1, limit: int | None = None, workers: int = 8,
            overwrite: bool = False) -> Path:
    """Returns the CSV path relative to ``speedplus_events/`` (what TRAIN.TRAIN_CSV wants)."""
    run_dir = Path(run_dir)
    prep = json.loads((run_dir / "prepare.json").read_text())
    if prep["split"] != "train":
        raise SystemExit(f"convert reads ground-truth poses: only train lists are allowed ({run_dir})")
    gt = read_train_list(run_dir, stride, limit)
    ds = open_dataset()
    traj0 = gt["trajectory"].iloc[0]
    vcam = load_or_make_vcam(ds, traj0)
    tf = load_or_make_target(ds, traj0)
    root = sensor_folder()
    write_camera_files(root, vcam, tf)

    canon_p = root / "canon.json"
    if canon_p.exists():
        R_ref = np.asarray(json.loads(canon_p.read_text())["R_ref"])
    else:
        T = np.stack([target_to_net(np.asarray(t, float).reshape(4, 4), tf, s) for t, s in zip(gt["T"], gt["scale"])])
        R_ref = estimate_reference(T[:, :3, :3], T[:, :3, 3], tf.rots)
        canon_p.write_text(json.dumps({"R_ref": R_ref.tolist(), "from": run_dir.name, "n_samples": len(gt),
                                       "symmetries": tf.symmetries}, indent=2))

    sub = root / run_dir.name
    csv = sub / "labels" / "labels.csv"
    meta_p = sub / "meta.json"
    key = {"n_gt": len(gt), "stride": stride, "limit": limit, "vcam": vcam.to_dict(), "R_ref": R_ref.tolist()}
    if csv.exists() and meta_p.exists() and not overwrite and json.loads(meta_p.read_text()).get("key") == key:
        print(f"{csv}: up to date")
        return csv.relative_to(root)
    img_dir, mask_dir = sub / image_folder_name(vcam), sub / mask_folder_name(vcam)
    for d in (img_dir, mask_dir, csv.parent):
        d.mkdir(parents=True, exist_ok=True)

    rows = list(zip(gt["trajectory"], gt["index"], gt["t_ns"], gt["scale"], gt["T"]))
    ctx = get_context("spawn")
    out = []
    with ctx.Pool(workers, _init, (vcam.to_dict(), R_ref.tolist(), traj0, str(img_dir), str(mask_dir))) as pool:
        for r in tqdm(pool.imap(_one, rows, chunksize=16), total=len(rows), desc=f"convert {run_dir.name}"):
            if r is not None:
                out.append(r)
    with open(csv, "w") as f:
        for r in out:
            f.write(", ".join(str(e) for e in r) + "\n")
    meta_p.write_text(json.dumps({"key": key, "n_written": len(out), "run_dir": str(run_dir), "prepare": prep},
                                 indent=2))
    print(f"{csv}: {len(out)} / {len(rows)} samples")
    return csv.relative_to(root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="e.g. results/sim/sim_zerog_docking_train_v3 (oshd-bench sim-prepare)")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th sample of the list")
    ap.add_argument("--limit", type=int, default=None, help="use n evenly spaced samples (smoke tests)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    convert(Path(a.run_dir), a.stride, a.limit, a.workers, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
