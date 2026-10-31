"""Predict T_lidar_model with the training-free crop + GeoTransformer (method ``geotransformer``).

    geotr-predict --setting ZS|FT --domain real --split test [--limit N] [--crop auto|oracle]
                  [--weights PATH]

The default weights are ``weights/oshd-<setting>.pth.tar`` from geotr-train. ``--crop oracle``
(GT-centred sphere, DIAGNOSIS ONLY) writes the setting ``<setting>-oracle``. The auto crop
never reads the GT ``T`` column. ``runtime_ms`` covers crop + collate + network (+ CUDA sync).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from tqdm import tqdm


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True, choices=["ZS", "FT"])
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--crop", default="auto", choices=["auto", "oracle"])
    ap.add_argument("--weights", default=None)
    a = ap.parse_args(argv)

    import oshd_bench
    from oshd_bench import PoseWriter, prediction_path
    from oshd_bench.sensors import lidar_points
    from oshd_lidar.common import (
        MODALITY,
        OBJECT,
        SENSOR,
        load_samples,
        load_samples_with_gt,
    )
    from oshd_lidar.crop import TargetCropper, gravity_up, oracle_crop

    from .common import (
        METHOD,
        import_upstream,
        inference_dict,
        make_cfg,
        ref_cloud,
        run_model,
        src_cloud,
        to_lidar_model,
        weights_for,
    )

    ds, res = oshd_bench.open_dataset(), oshd_bench.results_dir()
    try:
        gt = (load_samples_with_gt if a.crop == "oracle" else load_samples)(res, a.domain, a.split)
    except FileNotFoundError:
        print(f"no GT sample list for {a.domain}/{a.split}: nothing to do")
        return
    gt = gt.sort_values(["trajectory", "t_ns"]).reset_index(drop=True)
    if a.limit:
        gt = gt.iloc[: a.limit]
    setting = a.setting + ("-oracle" if a.crop == "oracle" else "")
    wpath = a.weights or weights_for(a.setting)

    cfg = make_cfg("predict")
    _, gt_model, _ = import_upstream()
    from geotransformer.utils.data import (
        calibrate_neighbors_stack_mode,
        registration_collate_fn_stack_mode,
    )
    from geotransformer.utils.torch import to_cuda

    ck = torch.load(wpath, map_location="cpu", weights_only=False)
    model = gt_model.create_model(cfg).cuda().eval()
    model.load_state_dict(ck["model"], strict=True)
    src = src_cloud(ds.objects[OBJECT])
    bb = cfg.backbone
    cropper = TargetCropper()

    def crop(r, p):
        if a.crop == "oracle":
            return oracle_crop(p, np.asarray(r.T).reshape(4, 4))
        tr = ds[r.trajectory]
        return cropper(p, key=r.trajectory, up=gravity_up(tr))

    limits = ck.get("neighbor_limits")
    if limits is None:  # pretrained-only weights: calibrate on the first crops (points only, no GT)
        calib = []
        for r in gt.iloc[:50].itertuples():
            c = crop(r, lidar_points(ds[r.trajectory], int(r.index))[0])
            if len(c.points) >= 8:
                calib.append(inference_dict(ref_cloud(c.points)[0], src))
        cropper.reset()
        limits = calibrate_neighbors_stack_mode(calib, registration_collate_fn_stack_mode, bb.num_stages,
                                                bb.init_voxel_size, bb.init_radius)
    w = PoseWriter(prediction_path(res, MODALITY, METHOD, setting, a.domain, a.split),
                   {"method": METHOD, "setting": setting, "weights": str(wpath), "crop": a.crop})
    times = []
    for r in tqdm(gt.itertuples(), total=len(gt), desc=f"{METHOD}/{setting}"):
        p, _ = lidar_points(ds[r.trajectory], int(r.index))
        t0 = time.perf_counter()
        c = crop(r, p)
        if len(c.points) < 8:
            continue
        ref, m = ref_cloud(c.points)
        batch = registration_collate_fn_stack_mode([inference_dict(ref, src)], bb.num_stages,
                                                   bb.init_voxel_size, bb.init_radius, limits)
        with torch.no_grad():
            try:
                out = run_model(model, to_cuda(batch))
            except (RuntimeError, IndexError) as e:  # degenerate inputs (e.g. too few superpoints)
                print(f"skip {r.trajectory}#{r.index}: {e}")
                continue
            T_rs = out["estimated_transform"].detach().cpu().numpy().astype(np.float64)
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        w.add(r.trajectory, SENSOR, r.t_ns, r.object, to_lidar_model(T_rs, m), score=1.0, runtime_ms=dt)
        times.append(dt)
    print(f"{w.save()}: {len(w)}/{len(gt)} poses, runtime median {np.median(times) if times else 0:.1f} ms")


if __name__ == "__main__":
    main()
