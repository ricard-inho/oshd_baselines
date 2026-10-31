"""Build work/dev_dataset: the real dataset (symlinked) + PLACEHOLDER calibration.

DEV ONLY. The extrinsics added here for firefly / realsense_depth / lidar / events and the
event-camera intrinsics are rough guesses so every baseline can be run end to end before
the real calibration exists. Numbers produced with it are meaningless.

Also marks trajectory_008 as `synthetic` so sim-training code paths can be exercised:
train = 007 (real) + 008 ("synthetic"), test = 009 (real).

    uv run --project bench python scripts/make_dev_dataset.py [--src ~/.cache/oshd_dataset/r3m3c3--off_test]
    export OSHD_DATASET_ROOT=$PWD/work/dev_dataset OSHD_BENCH_RESULTS=$PWD/work/dev_results OSHD_BENCHMARK=none
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from oshd_dataset.calibration import parse_transform
from oshd_dataset.geometry import rotmat_to_quat


def _tf(T: np.ndarray, note: str) -> dict:
    return {"T_body_cam": {"translation": [float(x) for x in T[:3, 3]],
                           "rotation_xyzw": [float(x) for x in rotmat_to_quat(T[:3, :3])]},
            "source": f"PLACEHOLDER (make_dev_dataset.py): {note}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / ".cache/oshd_dataset/r3m3c3--off_test"))
    ap.add_argument("--out", default="work/dev_dataset")
    a = ap.parse_args()
    src, out = Path(a.src).expanduser(), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # Real data only: the simulated recordings (sim_* folders, sim_manifest.json, metadata/)
    # don't fit the dev setup. splits.json is written below: never symlink it, or the write
    # would go through to the real cache.
    for p in src.iterdir():
        if p.name in ("calibration", "manifest.json", "splits.json", "sim_manifest.json", "metadata") \
                or p.name.startswith("sim_"):
            continue
        (out / p.name).symlink_to(p.resolve())
    real = set(json.loads((src / "manifest.json").read_text())["trajectories"])
    shutil.copytree(src / "calibration", out / "calibration",
                    ignore=lambda d, names: [n for n in names if n.startswith("trajectory_")
                                             and n.endswith(".yaml") and n[:-5] not in real])

    rig_p = out / "calibration" / "rig.yaml"
    rig = yaml.safe_load(rig_p.read_text())
    cams = rig["extrinsics"]["cameras"]
    T_bc = parse_transform(cams["realsense_color"]["T_body_cam"])

    off = np.eye(4)
    off[:3, 3] = [-0.059, 0, 0]  # D455: depth (left IR) ~59 mm from the colour imager
    cams.setdefault("realsense_depth", _tf(T_bc @ off, "colour pose shifted by the D455 baseline"))
    off = np.eye(4)
    off[:3, 3] = [0.08, 0, 0]
    cams.setdefault("firefly", _tf(T_bc @ off, "colour pose shifted 8 cm"))
    off[:3, 3] = [-0.08, 0, 0]
    cams.setdefault("events", _tf(T_bc @ off, "colour pose shifted -8 cm"))
    R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)  # columns: lidar x,y,z in the camera frame
    L = np.eye(4)
    L[:3, :3] = R
    L[:3, 3] = [0, -0.06, 0]
    cams.setdefault("lidar", _tf(T_bc @ L, "x along the colour optical axis, 6 cm above"))
    rig.setdefault("events", {
        "source": "PLACEHOLDER (make_dev_dataset.py): guessed pinhole, no distortion",
        "width": 1280, "height": 720, "distortion_model": "plumb_bob", "D": [0.0] * 5,
        "K": [1000.0, 0, 640.0, 0, 1000.0, 360.0, 0, 0, 1],
    })
    rig_p.write_text("# DEV ONLY: contains PLACEHOLDER calibration, see scripts/make_dev_dataset.py\n"
                     + yaml.safe_dump(rig, sort_keys=False))

    manifest = json.loads((src / "manifest.json").read_text())
    manifest.setdefault("trajectory_info", {})["trajectory_008"] = {"domain": "synthetic",
                                                                    "note": "DEV: pretend-sim"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "splits.json").write_text(json.dumps(
        {"version": 1, "train": ["trajectory_007", "trajectory_008"], "val": [], "test": ["trajectory_009"]},
        indent=2))
    print(f"wrote {out} (PLACEHOLDER calibration: numbers from it are meaningless)")


if __name__ == "__main__":
    main()
