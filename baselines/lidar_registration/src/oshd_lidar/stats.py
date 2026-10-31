"""Diagnostics: how many lidar points fall on the cube per scan (uses GT, never used by predict).

    lidar-stats --domain real --split test [--tol 0.02] [--out work/debug/lidar/cube_points.csv]

A point counts as "on the cube" when its distance to the GT cube body (box SDF in the model
frame) is below ``--tol``. Also reports the counts per range bin, the fraction of scans with
enough points for a 6-DoF registration (>= 10, >= 30) and how many faces are hit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CUBE_CENTER, box_sdf, inv, load_samples_with_gt, transform


def cube_point_mask(p_lidar: np.ndarray, T_lidar_model: np.ndarray, tol: float) -> np.ndarray:
    pm = transform(inv(T_lidar_model), p_lidar)
    return np.abs(box_sdf(pm)) < tol


def faces_hit(pm: np.ndarray, min_pts: int = 3) -> int:
    q = (pm - CUBE_CENTER) / 0.05
    ax = np.abs(q).argmax(1)
    sgn = np.sign(q[np.arange(len(q)), ax])
    keys = ax * 2 + (sgn > 0)
    return int(sum((keys == k).sum() >= min_pts for k in range(6)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", default="work/debug/lidar/cube_points.csv")
    a = ap.parse_args(argv)
    import oshd_bench
    from oshd_bench.sensors import lidar_points

    ds, res = oshd_bench.open_dataset(), oshd_bench.results_dir()
    gt = load_samples_with_gt(res, a.domain, a.split)
    rows = []
    for r in gt.itertuples():
        T = np.asarray(r.T).reshape(4, 4)
        p, _ = lidar_points(ds[r.trajectory], int(r.index))
        m = cube_point_mask(p, T, a.tol)
        pm = transform(inv(T), p[m])
        rows.append(dict(trajectory=r.trajectory, index=r.index, range_m=r.range_m, n_scan=len(p),
                         n_cube=int(m.sum()), faces=faces_hit(pm) if len(pm) else 0))
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    n = df["n_cube"]
    print(f"{a.domain}/{a.split}: {len(df)} scans, points per scan median {n.median():.0f} "
          f"(p10 {n.quantile(.1):.0f}, p90 {n.quantile(.9):.0f}); >=10 pts {(n >= 10).mean():.0%}, "
          f">=30 pts {(n >= 30).mean():.0%}; 0 pts {(n == 0).mean():.0%}; scan size {df.n_scan.median():.0f}")
    bins = [0, 0.4, 0.5, 0.75, 1.0, 1.5, 3.0]
    df["bin"] = pd.cut(df.range_m, bins)
    print(df.groupby("bin", observed=True).agg(scans=("n_cube", "size"), median_pts=("n_cube", "median"),
                                               p10=("n_cube", lambda x: x.quantile(.1)),
                                               faces=("faces", "median")).to_string())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
