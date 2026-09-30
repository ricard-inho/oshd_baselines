"""Run E-RAFT on the flow samples of a split and write per-sample flow statistics.

Flow samples: for every reference time ``t`` of the events GT sample list
(``results/gt/events/<domain>_<split>.parquet``; only trajectory / t_ns / object are read),
the pair ``t0 = t - 100 ms -> t1 = t``. Ground truth: ``pooh_bench.flow_gt.target_flow_at``
(forward flow on the target, rendered from the CAD model at the two mocap poses). Samples
without mocap at t0 or t1 are skipped.

Evaluation pixels (``--mask``):
  valid          all valid GT target pixels (default, used for Table III)
  valid_events   valid GT pixels with >= 1 event in [t0, t1) (written with suffix ``_evmask``)

Output: ``results_dir()/flow/eraft/<setting>/<domain>_<split>[ _evmask].parquet`` with columns
trajectory, t_ns, t0_ns, object, n, sum_epe, n_gt1, n_gt3, sum_ae_deg, runtime_ms.
``runtime_ms`` = both voxel grids (GPU) + E-RAFT (12 iterations) + upsampling to 1280x720,
with the events already in host memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import common as C


def ckpt_for(setting: str) -> Path:
    return C.WEIGHTS / "dsec.tar" if setting == "ZS" else C.WEIGHTS / "trained" / "FT.tar"


def save_debug(path: Path, im2: torch.Tensor, flow: np.ndarray, gt: np.ndarray, valid: np.ndarray, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ys, xs = np.nonzero(valid)
    pad = 40
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, C.H_FULL)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, C.W_FULL)
    mx = max(float(np.percentile(np.hypot(*gt[valid].T), 99)), 1.0)
    vg = im2[0].sum(0).cpu().numpy()
    vg = torch.nn.functional.interpolate(torch.from_numpy(vg)[None, None], size=(C.H_FULL, C.W_FULL))[0, 0].numpy()
    gt_c = C.flow_to_color(gt, mx)
    gt_c[~valid] = 0
    pr_c = C.flow_to_color(flow, mx)
    pr_m = pr_c.copy()
    pr_m[~valid] = 0
    epe = np.where(valid, np.linalg.norm(flow - gt, axis=-1), np.nan)
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.5))
    lim = np.percentile(np.abs(vg[vg != 0]), 99) if (vg != 0).any() else 1
    panels = [(vg, dict(cmap="seismic", vmin=-lim, vmax=lim), "voxel grid [t0,t1) (sum of bins)"),
              (pr_c, {}, "E-RAFT flow (full)"), (pr_m, {}, "E-RAFT flow on GT mask"),
              (gt_c, {}, f"GT target flow (wheel max {mx:.1f} px)"), (epe, dict(cmap="magma"), "EPE (px)")]
    for ax, (img, kw, t) in zip(axs, panels):
        im = ax.imshow(img[y0:y1, x0:x1], extent=(x0, x1, y1, y0), **kw)
        ax.set_title(t, fontsize=9)
        if t.startswith("EPE"):
            fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["ZS", "FT"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", default="real", choices=["real", "synthetic"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mask", choices=["valid", "valid_events"], default="valid")
    ap.add_argument("--scale", type=float, default=0.5, help="network resolution / 1280x720 (default 0.5)")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--debug", type=int, default=0, help="save N flow debug images to work/debug/events/eraft")
    a = ap.parse_args(argv)

    import pooh_bench
    import pyarrow.parquet as pq
    from pooh_bench.flow_gt import target_flow_at

    res = pooh_bench.results_dir()
    gp = pooh_bench.gt_path(res, "events", a.domain, a.split)
    if not gp.exists():
        print(f"no sample list {gp}: nothing to do")
        return 0
    cols = [c for c in pq.read_schema(gp).names if c in ("trajectory", "t_ns", "object")]
    samples = pd.read_parquet(gp, columns=cols)  # never the T column
    if a.limit:
        samples = samples.iloc[:: max(1, len(samples) // a.limit)].iloc[: a.limit]
    dev = torch.device("cuda")
    ckpt = Path(a.ckpt) if a.ckpt else ckpt_for(a.setting)
    model = C.load_model(ckpt, dev)
    model.eval()
    ds = C.get_dataset()
    dt = C.DT_MS * 1_000_000
    dbg_idx = set(np.linspace(0, len(samples) - 1, a.debug).astype(int).tolist()) if a.debug else set()
    dbg_dir = pooh_bench.repo_root() / "work" / "debug" / "events" / "eraft"
    rows = []
    for i, r in enumerate(samples.itertuples()):
        traj = ds[r.trajectory]
        t1 = int(r.t_ns)
        t0 = t1 - dt
        events = (C.load_events(traj, t0 - dt, t0), C.load_events(traj, t0, t1))
        torch.cuda.synchronize()
        ts = time.perf_counter()
        with torch.no_grad():
            im1, im2 = C.pair_inputs(traj, t0, t1, a.scale, dev, events)
            _, preds = model(im1, im2, iters=a.iters)
            flow = C.to_full_res(preds[-1], a.scale)[0].permute(1, 2, 0).cpu().numpy()
        rt = (time.perf_counter() - ts) * 1e3
        if i == 0:  # the first call includes cuDNN autotuning: re-time it
            torch.cuda.synchronize()
            ts = time.perf_counter()
            with torch.no_grad():
                im1, im2 = C.pair_inputs(traj, t0, t1, a.scale, dev, events)
                _, preds = model(im1, im2, iters=a.iters)
                flow = C.to_full_res(preds[-1], a.scale)[0].permute(1, 2, 0).cpu().numpy()
            rt = (time.perf_counter() - ts) * 1e3
        # ---- evaluation (ground truth only from here on)
        gt, valid = target_flow_at(traj, "events", ds.objects[r.object], t0, t1)
        if gt is None:
            continue
        mask = valid
        if a.mask == "valid_events":
            x, y = events[1][0].astype(np.int64), events[1][1].astype(np.int64)
            ok = (x < C.W_FULL) & (y < C.H_FULL)
            has = np.zeros(C.H_FULL * C.W_FULL, bool)
            has[y[ok] * C.W_FULL + x[ok]] = True
            mask = valid & has.reshape(C.H_FULL, C.W_FULL)
        st = pooh_bench.flow_stats(flow, gt, mask)
        rows.append({"trajectory": r.trajectory, "t_ns": t1, "t0_ns": t0, "object": r.object, "n": st.n,
                     "sum_epe": st.sum_epe, "n_gt1": st.n_gt1, "n_gt3": st.n_gt3, "sum_ae_deg": st.sum_ae_deg,
                     "runtime_ms": rt})
        if i in dbg_idx and mask.any():
            s = st.summary()
            save_debug(dbg_dir / f"{a.setting}_{a.domain}_{r.trajectory}_{t1}.png", im2, flow, gt, mask,
                       f"E-RAFT {a.setting} {r.trajectory} t={t1}  EPE {s['EPE']:.2f}px  n={st.n}")
    suffix = "" if a.mask == "valid" else "_evmask"
    out = res / "flow" / "eraft" / a.setting / f"{a.domain}_{a.split}{suffix}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["trajectory", "t_ns", "t0_ns", "object", "n", "sum_epe", "n_gt1", "n_gt3",
                                     "sum_ae_deg", "runtime_ms"])
    df.to_parquet(out, index=False)
    from pooh_bench.predictions import hardware

    tot = pooh_bench.metrics.FlowStats(int(df.n.sum()), float(df.sum_epe.sum()), int(df.n_gt1.sum()),
                                       int(df.n_gt3.sum()), float(df.sum_ae_deg.sum())).summary()
    out.with_suffix(".json").write_text(json.dumps({
        "method": "eraft", "setting": a.setting, "checkpoint": str(ckpt), "mask": a.mask, "scale": a.scale,
        "iters": a.iters, "n_samples": len(df), "summary": tot, "median_runtime_ms": float(df.runtime_ms.median())
        if len(df) else None, "hardware": hardware()}, indent=2))
    print(f"{out}: {len(df)} samples, {tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
