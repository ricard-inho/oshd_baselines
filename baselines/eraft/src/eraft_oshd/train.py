"""Fine-tune E-RAFT (from the DSEC checkpoint) on the real train split with target-masked GT flow.

Upstream has no training code; this follows RAFT's fine-tuning recipe (Teed & Deng 2020):
* sequence loss  sum_i gamma^(N-i) * mean_valid |f_i - f_gt|_1  with gamma = 0.8 over the
  N = 12 iterative predictions, averaged over valid GT pixels only (the target; the
  background has no GT);
* AdamW (lr 1e-4, weight decay 1e-5, eps 1e-8), OneCycle LR (pct_start 0.05, linear anneal,
  no momentum cycling), gradient clipping at 1.0, BatchNorm frozen (as RAFT after Chairs);
* the network runs at its inference resolution (640x360 by default); GT flow is rendered
  directly there with ``intrinsics.resized`` (exact, no flow resampling);
* augmentation: random horizontal / vertical flips (voxel grids flipped, flow sign flipped).

Flow pairs are the same as for evaluation: t0 = t - 100 ms -> t1 = t at the events GT sample
times of the train split. Rendered GT is cached (sparse) in work/eraft_cache/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import common as C


class FlowDataset(Dataset):
    def __init__(self, gt_paths, scale: float, cache_dir: Path):
        dfs = [pd.read_parquet(p, columns=["trajectory", "t_ns", "object"]) for p in gt_paths if Path(p).exists()]
        self.df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        self.scale, self.cache = scale, cache_dir

    def __len__(self):
        return len(self.df)

    def gt(self, traj, obj, t0, t1):
        from oshd_bench.flow_gt import target_flow
        from oshd_bench.gt import pose_in_sensor

        h, w = C.net_size(self.scale)
        f = self.cache / traj.name / f"{t1}_{w}x{h}.npz"
        if f.exists():
            z = np.load(f)
            flow = np.zeros((h, w, 2), np.float32)
            valid = np.zeros(h * w, bool)
            valid[z["idx"]] = True
            flow.reshape(-1, 2)[z["idx"]] = z["flow"]
            return flow, valid.reshape(h, w)
        T, ok = pose_in_sensor(traj, "events", obj, np.array([t0, t1]))
        if not ok.all():
            flow, valid = np.zeros((h, w, 2), np.float32), np.zeros((h, w), bool)
        else:
            intr = traj.calibration.intrinsics("events").resized(w, h)
            flow, valid = target_flow(obj, T[0], T[1], intr)
        f.parent.mkdir(parents=True, exist_ok=True)
        idx = np.nonzero(valid.reshape(-1))[0]
        np.savez(f, idx=idx, flow=flow.reshape(-1, 2)[idx])
        return flow, valid

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ds = C.get_dataset()
        traj = ds[r.trajectory]
        t1 = int(r.t_ns)
        t0 = t1 - C.DT_MS * 1_000_000
        flow, valid = self.gt(traj, ds.objects[r.object], t0, t1)
        ev = (C.load_events(traj, 2 * t0 - t1, t0), C.load_events(traj, t0, t1))
        return ev, flow, valid


def collate(b):
    return [x[0] for x in b], torch.from_numpy(np.stack([x[1] for x in b])), torch.from_numpy(np.stack([x[2] for x in b]))


def sequence_loss(preds, gt, valid, gamma=0.8, max_flow=400.0):
    """preds: list of (B, 2, H, W); gt (B, 2, H, W); valid (B, H, W) bool."""
    mag = gt.norm(dim=1)
    v = (valid & (mag < max_flow)).float()
    n = len(preds)
    loss = 0.0
    for i, p in enumerate(preds):
        w = gamma ** (n - i - 1)
        l1 = (p - gt).abs().sum(1)  # per-pixel L1 over (u, v), as RAFT
        loss = loss + w * (l1 * v).sum() / v.sum().clamp(min=1)
    epe = ((preds[-1] - gt).norm(dim=1) * v).sum() / v.sum().clamp(min=1)
    return loss, epe


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="FT", choices=["FT"])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wdecay", type=float, default=1e-5)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--init", default=str(C.WEIGHTS / "dsec.tar"))
    ap.add_argument("--out", default=str(C.WEIGHTS / "trained" / "FT.tar"))
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args(argv)

    import oshd_bench

    res = oshd_bench.results_dir()
    paths = [oshd_bench.gt_path(res, "events", "real", "train")]
    ds = FlowDataset(paths, a.scale, oshd_bench.repo_root() / "work" / "eraft_cache")
    if not len(ds):
        raise SystemExit(f"no samples in {paths}")
    print(f"FT: {len(ds)} flow pairs from {paths}, {a.steps} steps, batch {a.batch}")
    dev = torch.device("cuda")
    model = C.load_model(Path(a.init), dev)
    model.train()
    model.freeze_bn()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wdecay, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps + 100, pct_start=0.05,
                                                cycle_momentum=False, anneal_strategy="linear")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
                    drop_last=len(ds) >= a.batch, persistent_workers=a.workers > 0)
    h, w = C.net_size(a.scale)
    step, t_start = 0, time.time()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    while step < a.steps:
        for evs, flow, valid in dl:
            im1 = torch.cat([C.voxel_grid_dsec(*e[0], h, w, a.scale, dev)[None] for e in evs])
            im2 = torch.cat([C.voxel_grid_dsec(*e[1], h, w, a.scale, dev)[None] for e in evs])
            gt = flow.to(dev).permute(0, 3, 1, 2)
            vm = valid.to(dev)
            if torch.rand(1).item() < 0.5:  # horizontal flip
                im1, im2, gt, vm = im1.flip(-1), im2.flip(-1), gt.flip(-1), vm.flip(-1)
                gt = gt * torch.tensor([-1.0, 1.0], device=dev)[None, :, None, None]
            if torch.rand(1).item() < 0.1:  # vertical flip
                im1, im2, gt, vm = im1.flip(-2), im2.flip(-2), gt.flip(-2), vm.flip(-2)
                gt = gt * torch.tensor([1.0, -1.0], device=dev)[None, :, None, None]
            model.image_padder.pad_height = model.image_padder.pad_width = None
            _, preds = model(im1, im2, iters=a.iters)
            loss, epe = sequence_loss(preds, gt, vm, a.gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % a.log_every == 0 or step == 1:
                print(f"step {step}/{a.steps} loss {loss.item():.3f} epe {epe.item():.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t_start) / step:.2f}s/step", flush=True)
            if step % 1000 == 0 or step == a.steps:
                torch.save({"model": model.state_dict(), "steps": step, "init": a.init, "train": [str(p) for p in paths],
                            "scale": a.scale}, out)
            if step >= a.steps:
                break
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
