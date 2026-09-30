"""Debug overlays: predicted pose (green: CAD silhouette outline + cube-body edges) and, for
comparison only, the ground truth (red edges) on the raw sensor image. Reads GT; not inference.

    megapose-overlay --modality rgb_realsense --setting ORACLE --n 6 --out ../../work/debug/megapose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .common import MODALITIES, E2VIDFrames

EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _edges(draw, intr, T, kp, color, width):
    from pooh_dataset.geometry import transform_points

    # sample each edge densely so the (distorted) projection is drawn as a curve
    for i, j in EDGES:
        s = np.linspace(0, 1, 20)[:, None]
        uv, ok = intr.project(transform_points(T, kp[i] + s * (kp[j] - kp[i])))
        pts = [tuple(p) for p, g in zip(uv, ok) if g]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=width)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="megapose-overlay")
    ap.add_argument("--modality", required=True, choices=sorted(MODALITIES))
    ap.add_argument("--setting", default="ORACLE")
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None, help="default <repo>/work/debug/megapose")
    a = ap.parse_args(argv)

    import pooh_bench
    from pooh_bench import gt_path, prediction_path
    from pooh_dataset.labels import render_mask

    m = MODALITIES[a.modality]
    ds = pooh_bench.open_dataset()
    res = pooh_bench.results_dir()
    pred = pd.read_parquet(prediction_path(res, a.modality, m["method"], a.setting, a.domain, a.split))
    gt = pd.read_parquet(gt_path(res, m["sensor"], a.domain, a.split))
    df = pred.merge(gt[["trajectory", "t_ns", "object", "index", "T"]].rename(columns={"T": "T_gt"}),
                    on=["trajectory", "t_ns", "object"])
    out = Path(a.out) if a.out else pooh_bench.repo_root() / "work" / "debug" / "megapose"
    out.mkdir(parents=True, exist_ok=True)
    frames = None
    if m["sensor"] == "events":
        frames = E2VIDFrames(Path(a.frames_dir) if a.frames_dir else res / "e2vid", a.domain, a.split)
    pick = np.unique(np.linspace(0, len(df) - 1, min(a.n, len(df))).astype(int)) if len(df) else []
    for k in pick:
        r = df.iloc[k]
        traj = ds[r.trajectory]
        intr = traj.calibration.intrinsics(m["sensor"])
        img = frames.get(r.trajectory, int(r.t_ns))[0] if frames else traj.frame(m["sensor"], int(r["index"]))
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, -1)
        obj = ds.objects[r.object]
        T = np.asarray(r["T"], float).reshape(4, 4)
        Tg = np.asarray(r.T_gt, float).reshape(4, 4)
        im = Image.fromarray(img[..., :3].copy())
        lw = max(1, intr.width // 400)
        mask = render_mask(obj, T, intr)
        edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
        a_ = np.asarray(im).copy()
        a_[mask] = (0.6 * a_[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        if lw > 1:
            from scipy.ndimage import binary_dilation

            edge = binary_dilation(edge, iterations=lw - 1)
        a_[edge] = (0, 255, 0)
        im = Image.fromarray(a_)
        d = ImageDraw.Draw(im)
        _edges(d, intr, Tg, obj.click_points, (255, 0, 0), lw)
        _edges(d, intr, T, obj.click_points, (0, 255, 0), lw)
        et = np.linalg.norm(T[:3, 3] - Tg[:3, 3]) * 100
        d.text((5, 5), f"{r.trajectory} {r.t_ns} score {r.score:.2f} |dt_origin| {et:.1f} cm "
                       f"(green pred, red GT)", fill=(255, 255, 0))
        p = out / f"{a.modality}_{a.setting}_{a.domain}_{a.split}_{r.trajectory}_{r.t_ns}.jpg"
        if intr.width > 1600:
            im = im.resize((im.width // 2, im.height // 2))
        im.save(p, quality=90)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
