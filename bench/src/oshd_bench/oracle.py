"""Ground-truth 2D detections (rendered masks), for debugging pose baselines without a detector."""

from __future__ import annotations

import pandas as pd
from oshd_dataset.labels import mask_to_bbox, render_mask
from tqdm import tqdm

from .gt import gt_poses
from .predictions import DetectionWriter, detection_path


def oracle_detections(ev, sensor: str, split: str = "test", domain: str = "real"):
    from .evaluate import gt_path

    gt = pd.read_parquet(gt_path(ev.results, sensor, domain, split))
    w = DetectionWriter(detection_path(ev.results, sensor, "oracle", "GT", domain, split), {"detector": "oracle"})
    T = gt_poses(gt)
    for k, r in enumerate(tqdm(gt.itertuples(), total=len(gt), desc="oracle")):
        traj = ev.ds[r.trajectory]
        mask = render_mask(ev.ds.objects[r.object], T[k], traj.calibration.intrinsics(sensor))
        box = mask_to_bbox(mask)
        if box is not None:
            w.add(r.trajectory, sensor, r.t_ns, r.object, box, 1.0, mask, 0.0)
    return w.save()
