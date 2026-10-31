"""Debug overlays: voxel grid (summed over bins) + stage-A box + predicted (red) and
ground-truth canonical (green) keypoints, plus the reprojected PnP pose (yellow).

Reads the GT ``T`` column for the overlay only; this is not part of the inference path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import common as C
from .predict import Estimator
from .train import CKPT_DIR


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="ZS")
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    import oshd_bench
    from oshd_dataset.events import voxel_grid

    res = oshd_bench.results_dir()
    df = C.read_samples(oshd_bench.gt_path(res, "events", a.domain, a.split), with_pose=True)
    df = df.iloc[np.linspace(0, len(df) - 1, a.n).astype(int)]
    est = Estimator(CKPT_DIR / f"{a.setting}.pt")
    ds = C.get_dataset()
    out = Path(a.out or oshd_bench.repo_root() / "work" / "debug" / "events" / "event_keypoints")
    out.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        traj = ds[r.trajectory]
        obj = ds.objects[r.object]
        intr = traj.calibration.intrinsics("events")
        kp3 = C.object_keypoints(obj)
        T = C.canonical_pose(np.asarray(r.T, float).reshape(4, 4), C.symmetry(obj))
        gt_uv, _ = intr.project((T[:3, :3] @ kp3.T).T + T[:3, 3])
        ev = C.load_window(traj, r.t_ns)
        from oshd_dataset.events import Events

        vg = voxel_grid(Events(ev[2], ev[0].astype(np.uint16), ev[1].astype(np.uint16), ev[3]), C.NUM_BINS,
                        t_start_ns=int(r.t_ns) - int(C.WINDOW_MS * 1e6), t_end_ns=int(r.t_ns)).sum(0)
        o = est.networks(ev, int(r.t_ns))
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        lim = np.percentile(np.abs(vg[vg != 0]), 99) if (vg != 0).any() else 1
        for ax in axs:
            ax.imshow(vg, cmap="seismic", vmin=-lim, vmax=lim)
            ax.scatter(gt_uv[:, 0], gt_uv[:, 1], c="lime", s=25, marker="o", label="GT (canonical)")
            for k, (u, v) in enumerate(gt_uv):
                ax.text(u + 3, v - 3, str(k), color="lime", fontsize=7)
        title = f"{r.trajectory} t={r.t_ns} range={r.range_m:.2f} m"
        if o is not None:
            cx, cy, s = o["box"]
            T_p, score = est.pnp(o, kp3, intr.K, intr.D)
            for ax in axs:
                ax.add_patch(plt.Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False, ec="orange"))
                ax.scatter(o["kp"][:, 0], o["kp"][:, 1], c="red", s=25, marker="x", label="pred")
                for k, (u, v) in enumerate(o["kp"]):
                    ax.text(u + 3, v + 8, str(k), color="red", fontsize=7)
                if T_p is not None:
                    uvp, _ = intr.project((T_p[:3, :3] @ kp3.T).T + T_p[:3, 3])
                    ax.scatter(uvp[:, 0], uvp[:, 1], facecolors="none", edgecolors="yellow", s=60, label="PnP")
            axs[1].set_xlim(cx - s, cx + s)
            axs[1].set_ylim(cy + s, cy - s)
            title += f" det={o['det_score']:.2f}"
        else:
            title += " (no detection)"
        axs[0].legend(loc="lower right", fontsize=7)
        fig.suptitle(title)
        fig.tight_layout()
        p = out / f"{a.setting}_{a.domain}_{r.trajectory}_{r.t_ns}.png"
        fig.savefig(p, dpi=90)
        plt.close(fig)
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
