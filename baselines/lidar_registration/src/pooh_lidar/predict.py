"""Predict T_lidar_model with crop + FPFH/TEASER++/ICP (method ``fpfh_teaser_icp``).

    lidar-reg-predict --setting ZS --domain real --split test [--limit N] [--crop auto|oracle]

``--crop oracle`` (GT-centred sphere, DIAGNOSIS ONLY) reads the GT pose to crop and is
written under the setting ``ZS-oracle``, which Table I never reads. The default ``auto`` crop
never reads ``T``. ``runtime_ms`` covers loading-free crop + registration per scan.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from tqdm import tqdm

from .common import MODALITY, OBJECT, SENSOR, load_samples, load_samples_with_gt, model_cloud
from .crop import TargetCropper, gravity_up, oracle_crop

METHOD = "fpfh_teaser_icp"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="ZS", choices=["ZS"])
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--crop", default="auto", choices=["auto", "oracle"])
    ap.add_argument("--no-track", action="store_true", help="disable causal crop tracking")
    a = ap.parse_args(argv)
    import pooh_bench
    from pooh_bench import PoseWriter, prediction_path
    from pooh_bench.sensors import lidar_points

    ds, res = pooh_bench.open_dataset(), pooh_bench.results_dir()
    try:
        gt = (load_samples_with_gt if a.crop == "oracle" else load_samples)(res, a.domain, a.split)
    except FileNotFoundError:
        print(f"no GT sample list for {a.domain}/{a.split}: nothing to do")
        return
    gt = gt.sort_values(["trajectory", "t_ns"]).reset_index(drop=True)
    if a.limit:
        gt = gt.iloc[: a.limit]
    setting = a.setting + ("-oracle" if a.crop == "oracle" else "")
    reg = FpfhTeaserIcp(model_cloud(ds.objects[OBJECT], 20000))
    cropper = TargetCropper(track=not a.no_track)
    ups: dict = {}
    w = PoseWriter(prediction_path(res, MODALITY, METHOD, setting, a.domain, a.split),
                   {"method": METHOD, "setting": setting, "crop": a.crop, "track": not a.no_track})
    stats = []
    for r in tqdm(gt.itertuples(), total=len(gt), desc=f"{METHOD}/{setting}"):
        traj = ds[r.trajectory]
        if r.trajectory not in ups:
            ups[r.trajectory] = gravity_up(traj)  # IMU, once per trajectory
        p, _ = lidar_points(traj, int(r.index))
        t0 = time.perf_counter()
        if a.crop == "oracle":
            c = oracle_crop(p, np.asarray(r.T).reshape(4, 4))
        else:
            c = cropper(p, key=r.trajectory, up=ups[r.trajectory])
        out = reg(c.points) if len(c.points) >= 5 else None
        dt = (time.perf_counter() - t0) * 1000
        if out is None:
            continue
        w.add(r.trajectory, SENSOR, r.t_ns, r.object, out.T_lidar_model, score=out.fitness, runtime_ms=dt)
        stats.append((len(c.points), out.n_inliers, out.fitness, dt))
    path = w.save()
    s = np.array(stats) if stats else np.zeros((0, 4))
    print(f"{path}: {len(w)}/{len(gt)} poses; median crop pts {np.median(s[:, 0]) if len(s) else 0:.0f}, "
          f"TEASER inliers {np.median(s[:, 1]) if len(s) else 0:.0f}, runtime {np.median(s[:, 3]) if len(s) else 0:.1f} ms")


from .register import FpfhTeaserIcp  # noqa: E402

if __name__ == "__main__":
    main()
