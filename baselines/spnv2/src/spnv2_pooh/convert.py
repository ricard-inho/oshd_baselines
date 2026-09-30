"""pooh-dataset -> SPEED+-style training folder that upstream's SPEEDPLUSDataset reads.

Layout (``<work>/data``)::

    speedplus_<sensor>/
        camera.json  cube_keypoints.mat  cube.ply     # virtual camera, 8 corners [3, 8] (net frame), model (mm)
        vcam.json  canon.json                         # virtual camera, symmetry reference orientation
        <domain>_<split>/
            images_<W>x<H>_RGB/<traj>_<index>.jpg     # undistorted + resized + padded, grayscale x3
            masks_<W/4>x<H/4>/<traj>_<index>.jpg      # pooh_dataset.labels.render_mask
            labels/labels.csv                         # upstream CSV row format (tools/preprocess.py)

CSV row: filename, bbox (xmin, ymin, xmax, ymax, normalised), q (qw, qx, qy, qz of R_cam_net),
t (3, metres, cube centre), origin (2, normalised), 8 keypoints (x, y normalised).

Symmetry: the pose label is the equivalent (of the 24) closest, in the viewing-ray frame, to
a fixed reference orientation (``canon.json``, the chordal mean of the first training set
converted). Keypoint identities are then a function of the image appearance, except near
the cell boundaries. See README.md.

This reads the GT ``T`` column, so it is only ever run on *train* splits.
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

from .common import (OBJECT, R_to_q_wxyz, VirtualCamera, Warper, canonicalize, cube_frame,
                     estimate_reference, to_network_image, work_dir, write_camera_files)


def data_root() -> Path:
    return work_dir() / "data"


def sensor_folder(sensor: str) -> Path:
    return data_root() / f"speedplus_{sensor}"


def image_folder_name(vcam: VirtualCamera) -> str:
    return f"images_{vcam.width}x{vcam.height}_RGB"


def mask_folder_name(vcam: VirtualCamera, stride: int = 4) -> str:
    return f"masks_{vcam.width // stride}x{vcam.height // stride}"


def load_or_make_vcam(ds, sensor: str, traj_name: str) -> VirtualCamera:
    """The virtual camera is fixed on first use (first trajectory's intrinsics) and reused."""
    p = sensor_folder(sensor) / "vcam.json"
    if p.exists():
        return VirtualCamera(**json.loads(p.read_text()))
    vcam = VirtualCamera.from_intrinsics(sensor, ds[traj_name].calibration.intrinsics(sensor))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(vcam.to_dict(), indent=2))
    return vcam


# ------------------------------------------------------------------ worker
_W: dict = {}


def _init(sensor, vcam_d, R_ref, img_dir, mask_dir, gray):
    from pooh_bench import load_config, open_dataset, repo_root

    ds = open_dataset()
    obj = ds.objects[OBJECT]
    cfg = load_config(repo_root() / "configs" / "bench.yaml")
    c, kp, rots = cube_frame(obj, cfg.get("symmetries", {}).get(OBJECT))
    vcam = VirtualCamera(**vcam_d)
    _W.update(ds=ds, obj=obj, c=c, kp=kp, rots=rots, vcam=vcam, warp=Warper(vcam), R_ref=np.asarray(R_ref),
              sensor=sensor, img_dir=Path(img_dir), mask_dir=Path(mask_dir), gray=gray,
              vi=vcam.intrinsics(), vm=vcam.intrinsics(0.25))


def _one(row: tuple) -> list | None:
    from pooh_dataset.geometry import transform_points
    from pooh_dataset.labels import projected_bbox, render_mask

    traj_name, index, T_flat = row
    w = _W
    traj = w["ds"][traj_name]
    vcam, vi = w["vcam"], w["vi"]
    T_cm = np.asarray(T_flat, float).reshape(4, 4)
    box = projected_bbox(w["obj"], T_cm, vi, clip=True)
    if box is None or box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    T_cn = canonicalize(T_cm, w["c"], w["R_ref"], w["rots"])
    uv, _ = vi.project(transform_points(T_cn, np.vstack([np.zeros(3), w["kp"]])))
    uv = uv / np.array([vcam.width, vcam.height])
    fname = f"{traj_name}_{int(index):06d}.jpg"
    img = w["warp"](traj.frame(w["sensor"], int(index)), traj.calibration.intrinsics(w["sensor"]))
    cv2.imwrite(str(w["img_dir"] / fname), to_network_image(img, w["gray"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    mask = render_mask(w["obj"], T_cm, w["vm"]).astype(np.uint8) * 255
    cv2.imwrite(str(w["mask_dir"] / fname), mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
    b = box / np.array([vcam.width, vcam.height, vcam.width, vcam.height])
    return [fname, *b.tolist(), *R_to_q_wxyz(T_cn[:3, :3]).tolist(), *T_cn[:3, 3].tolist(),
            *uv.reshape(-1).tolist()]


# ------------------------------------------------------------------ main
def convert(sensor: str, domain: str, split: str, stride: int = 1, limit: int | None = None,
            workers: int = 8, gray: bool = True, overwrite: bool = False) -> Path:
    """Returns the CSV path relative to ``speedplus_<sensor>/`` (what TRAIN.TRAIN_CSV wants)."""
    from pooh_bench import gt_path, load_config, open_dataset, repo_root, results_dir

    if split != "train":
        raise SystemExit("convert reads ground-truth poses: only train splits are allowed")
    gt = pd.read_parquet(gt_path(results_dir(), sensor, domain, split))
    gt = gt[gt["object"] == OBJECT].iloc[::stride]
    if limit:
        gt = gt.iloc[:limit]
    if not len(gt):
        raise SystemExit(f"no {sensor} samples in {domain}/{split}")
    ds = open_dataset()
    obj = ds.objects[OBJECT]
    sym_cfg = load_config(repo_root() / "configs" / "bench.yaml").get("symmetries", {}).get(OBJECT)
    c, kp, rots = cube_frame(obj, sym_cfg)
    vcam = load_or_make_vcam(ds, sensor, gt["trajectory"].iloc[0])
    root = sensor_folder(sensor)
    write_camera_files(root, vcam, kp, obj, c)

    canon_p = root / "canon.json"
    if canon_p.exists():
        R_ref = np.asarray(json.loads(canon_p.read_text())["R_ref"])
    else:
        T = np.stack([np.asarray(t, float).reshape(4, 4) for t in gt["T"]]) @ np.eye(4)
        T[:, :3, 3] += T[:, :3, :3] @ c  # model -> net origin
        R_ref = estimate_reference(T[:, :3, :3], T[:, :3, 3], rots)
        canon_p.write_text(json.dumps({"R_ref": R_ref.tolist(), "center": c.tolist(),
                                       "from": f"{domain}_{split}", "keypoints_net": kp.tolist()}, indent=2))

    sub = root / f"{domain}_{split}"
    csv = sub / "labels" / "labels.csv"
    meta_p = sub / "meta.json"
    key = {"n_gt": len(gt), "stride": stride, "limit": limit, "gray": gray, "vcam": vcam.to_dict(),
           "R_ref": R_ref.tolist()}
    if csv.exists() and meta_p.exists() and not overwrite and json.loads(meta_p.read_text()).get("key") == key:
        print(f"{csv}: up to date")
        return csv.relative_to(root)
    img_dir, mask_dir = sub / image_folder_name(vcam), sub / mask_folder_name(vcam)
    for d in (img_dir, mask_dir, csv.parent):
        d.mkdir(parents=True, exist_ok=True)

    rows = list(zip(gt["trajectory"], gt["index"], gt["T"]))
    # keep trajectories contiguous per worker chunk (row-group cache friendly)
    ctx = get_context("spawn")
    out = []
    with ctx.Pool(workers, _init, (sensor, vcam.to_dict(), R_ref.tolist(), str(img_dir), str(mask_dir), gray)) as pool:
        for r in tqdm(pool.imap(_one, rows, chunksize=16), total=len(rows), desc=f"convert {sensor}/{domain}_{split}"):
            if r is not None:
                out.append(r)
    with open(csv, "w") as f:
        for r in out:
            f.write(", ".join(str(e) for e in r) + "\n")
    meta_p.write_text(json.dumps({"key": key, "n_written": len(out)}, indent=2))
    print(f"{csv}: {len(out)} / {len(rows)} samples")
    return csv.relative_to(root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", required=True, choices=["firefly", "realsense_color"])
    ap.add_argument("--domain", required=True, choices=["synthetic", "real"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--color", action="store_true", help="keep RGB (upstream uses grayscale)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    convert(a.sensor, a.domain, a.split, a.stride, a.limit, a.workers, not a.color, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
