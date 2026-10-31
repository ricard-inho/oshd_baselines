"""Debug plots: scan (grey), crop (red), predicted model (blue), GT model (green), in two views.

    lidar-viz --method fpfh_teaser_icp --setting ZS [--domain real --split test] [--n 6]
              [--out work/debug/lidar]

Reads the GT (diagnostics only). It also prints the per-sample centre error of the prediction file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (
    CUBE_CENTER,
    MODALITY,
    OBJECT,
    load_samples_with_gt,
    model_cloud,
    transform,
)
from .crop import TargetCropper, gravity_up


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="fpfh_teaser_icp")
    ap.add_argument("--setting", default="ZS")
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="work/debug/lidar")
    a = ap.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import oshd_bench
    import pandas as pd
    from oshd_bench import prediction_path
    from oshd_bench.sensors import lidar_points

    ds, res = oshd_bench.open_dataset(), oshd_bench.results_dir()
    gt = load_samples_with_gt(res, a.domain, a.split)
    pr = pd.read_parquet(prediction_path(res, MODALITY, a.method, a.setting, a.domain, a.split))
    df = gt.merge(pr[["trajectory", "t_ns", "T", "score"]].rename(columns={"T": "Tp"}), on=["trajectory", "t_ns"])
    Tg = np.stack([np.asarray(t).reshape(4, 4) for t in df["T"]])
    Tp = np.stack([np.asarray(t).reshape(4, 4) for t in df["Tp"]])
    cg = np.einsum("nij,j->ni", Tg[:, :3, :3], CUBE_CENTER) + Tg[:, :3, 3]
    cp = np.einsum("nij,j->ni", Tp[:, :3, :3], CUBE_CENTER) + Tp[:, :3, 3]
    et = np.linalg.norm(cg - cp, axis=1)
    print(f"{len(df)} predictions: centre error median {np.median(et) * 100:.1f} cm, "
          f"<5 cm {np.mean(et < 0.05):.0%}, <10 cm {np.mean(et < 0.10):.0%}")
    model = model_cloud(ds.objects[OBJECT], 1500)
    cropper = TargetCropper()
    pick = np.linspace(0, len(df) - 1, min(a.n, len(df))).astype(int)
    crops = {}
    for k in range(pick.max() + 1):  # replay the causal tracker up to the plotted samples
        r = df.iloc[k]
        traj = ds[r["trajectory"]]
        c = cropper(lidar_points(traj, int(r["index"]))[0], key=r["trajectory"], up=gravity_up(traj)).points
        if k in pick:
            crops[k] = c
    fig, axs = plt.subplots(2, len(pick), figsize=(4 * len(pick), 8), squeeze=False)
    for j, k in enumerate(pick):
        r = df.iloc[k]
        traj = ds[r["trajectory"]]
        p, _ = lidar_points(traj, int(r["index"]))
        c = crops[k]
        near = p[np.linalg.norm(p - cg[k], axis=1) < 0.4]
        mg, mp = transform(Tg[k], model), transform(Tp[k], model)
        for i, (u, v) in enumerate([(0, 1), (0, 2)]):
            ax = axs[i, j]
            ax.scatter(mg[:, u], mg[:, v], s=1, c="g", alpha=0.3)
            ax.scatter(mp[:, u], mp[:, v], s=1, c="b", alpha=0.3)
            ax.scatter(near[:, u], near[:, v], s=3, c="0.3")
            if len(c):
                ax.scatter(c[:, u], c[:, v], s=6, c="r")
            ax.set_aspect("equal")
            ax.set_xlim(cg[k, u] - 0.3, cg[k, u] + 0.3)
            ax.set_ylim(cg[k, v] - 0.3, cg[k, v] + 0.3)
            ax.set_title(f"{r['trajectory'][-3:]}#{r['index']} r={r['range_m']:.2f} e_t={et[k] * 100:.1f}cm\n"
                         + ("top (x-y)" if i == 0 else "side (x-z)"), fontsize=9)
    fig.suptitle(f"{a.method}/{a.setting} {a.domain}_{a.split}: dark grey scan, red crop, green GT, blue pred")
    out = Path(a.out) / f"{a.method}_{a.setting}_{a.domain}_{a.split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=70, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
