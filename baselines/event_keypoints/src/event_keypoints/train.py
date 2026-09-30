"""Train the two-stage voxel-grid keypoint network.

ZS: synthetic train split only, from ImageNet ResNet-18 weights.
FT: initialised from the ZS checkpoint, fine-tuned on the real train split.

Both stages are trained jointly on the same batches: stage A on the full 1/2-res voxel grid,
stage B on crops around the ground-truth box with random shift / scale jitter (so it tolerates
stage-A box errors at test time).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import common as C
from .model import Detector, KeypointNet, focal_loss

CKPT_DIR = Path(__file__).resolve().parents[2] / "weights" / "trained"


class EventPoseDataset(Dataset):
    """Training samples from GT sample lists (train split: reading ``T`` is allowed here)."""

    def __init__(self, paths: list[Path], augment: bool = True):
        import pandas as pd

        dfs = [C.read_samples(p, with_pose=True) for p in paths if p.exists()]
        self.df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        ds = C.get_dataset()
        traj = ds[r.trajectory]
        obj = ds.objects[r.object]
        intr = traj.calibration.intrinsics("events")
        T = C.canonical_pose(np.asarray(r["T"], float).reshape(4, 4), C.symmetry(obj))
        kp3 = C.object_keypoints(obj)
        uv, _ = intr.project((T[:3, :3] @ kp3.T).T + T[:3, 3])
        x, y, t, p = C.load_window(traj, r.t_ns)
        rng = np.random.default_rng()
        if self.augment and len(t):
            keep = rng.random(len(t)) < rng.uniform(0.5, 1.0)          # event-rate variation
            x, y, t, p = x[keep], y[keep], t[keep], p[keep]
            n_noise = int(rng.uniform(0, 0.02) * C.H_FULL * C.W_FULL)   # background-activity noise
            t0 = int(r.t_ns) - int(C.WINDOW_MS * 1e6)
            nt = np.sort(rng.integers(t0, int(r.t_ns), n_noise))
            order = np.argsort(np.concatenate([t, nt]), kind="stable")
            x = np.concatenate([x, rng.integers(0, C.W_FULL, n_noise).astype(np.int16)])[order]
            y = np.concatenate([y, rng.integers(0, C.H_FULL, n_noise).astype(np.int16)])[order]
            p = np.concatenate([p, rng.integers(0, 2, n_noise).astype(np.uint8)])[order]
            t = np.concatenate([t, nt])[order]
        lo, hi = uv[:8].min(0), uv[:8].max(0)
        box = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, max(hi[0] - lo[0], hi[1] - lo[1], 8.0)])
        return (x, y, t, p), int(r.t_ns), uv.astype(np.float32), box.astype(np.float32)


def collate(batch):
    ev = [b[0] for b in batch]
    t_end = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    uv = torch.from_numpy(np.stack([b[2] for b in batch]))
    box = torch.from_numpy(np.stack([b[3] for b in batch]))
    return ev, t_end, uv, box


def losses(det, kpn, batch, device, augment=True):
    ev, t_end, uv, box = batch
    B = len(ev)
    span = int(C.WINDOW_MS * 1e6)
    t0 = (t_end - span).to(device)
    evd = C.events_to_device(ev, device)
    uv, box = uv.to(device), box.to(device)

    # ---- stage A: centre heatmap + log side at stride DET_DOWN*DET_STRIDE
    s = C.DET_DOWN * C.DET_STRIDE
    Hh, Wh = C.DET_PAD_H // C.DET_STRIDE, (C.W_FULL // C.DET_DOWN) // C.DET_STRIDE
    xa = C.det_input(evd, B, t0, span)
    heat, size = det(xa)
    c_out = ((box[:, :2] + 0.5) / s - 0.5).round()  # integer cell of the box centre
    c_out[:, 0] = c_out[:, 0].clamp(0, Wh - 1)
    c_out[:, 1] = c_out[:, 1].clamp(0, Hh - 1)
    sigma = (box[:, 2] / s / 6).clamp(min=1.0)
    tgt = C.gaussian_maps(c_out[:, None], Hh, Wh, sigma)
    l_heat = focal_loss(heat, tgt)
    ci = c_out[:, 1].long() * Wh + c_out[:, 0].long()
    size_pred = size.view(B, -1).gather(1, ci[:, None])[:, 0]
    l_size = F.l1_loss(size_pred.float(), torch.log(box[:, 2]))

    # ---- stage B: jittered crops of the full-res grid
    grid = C.full_grid(evd, B, t0, span)
    side = box[:, 2] * C.CROP_MARGIN
    cx, cy = box[:, 0], box[:, 1]
    if augment:
        side = side * torch.empty(B, device=device).uniform_(0.85, 1.25)
        cx = cx + side * torch.empty(B, device=device).uniform_(-0.12, 0.12)
        cy = cy + side * torch.empty(B, device=device).uniform_(-0.12, 0.12)
    side = side.clamp(min=32)
    xb = C.crop_input(grid, cx, cy, side)
    hu = C.img_to_heat(uv[..., 0], cx[:, None], side[:, None])
    hv = C.img_to_heat(uv[..., 1], cy[:, None], side[:, None])
    tgt_k = C.gaussian_maps(torch.stack([hu, hv], -1), C.HEAT, C.HEAT, 1.5)
    pred_k = kpn(xb)
    l_kp = 1000 * F.mse_loss(pred_k.float(), tgt_k)
    return l_heat, l_size, l_kp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["ZS", "FT"], required=True)
    ap.add_argument("--iters", type=int, default=None, help="default: ZS 20000, FT 6000")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default: ZS 5e-4, FT 1e-4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init", default=None, help="FT: checkpoint to start from (default weights/trained/ZS.pt)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    a = ap.parse_args(argv)

    import pooh_bench

    res = pooh_bench.results_dir()
    domain = "synthetic" if a.setting == "ZS" else "real"
    paths = [pooh_bench.gt_path(res, "events", domain, "train")]
    ds = EventPoseDataset(paths)
    if not len(ds):
        raise SystemExit(f"no training samples in {paths} (run `pooh-bench gt` first)")
    iters = a.iters or (20000 if a.setting == "ZS" else 6000)
    lr = a.lr or (5e-4 if a.setting == "ZS" else 1e-4)
    print(f"{a.setting}: {len(ds)} samples from {paths}, {iters} iters, batch {a.batch}, lr {lr}")

    device = torch.device("cuda")
    det, kpn = Detector(pretrained=a.setting == "ZS").to(device), KeypointNet(pretrained=a.setting == "ZS").to(device)
    if a.setting == "FT":
        init = Path(a.init) if a.init else CKPT_DIR / "ZS.pt"
        ck = torch.load(init, map_location="cpu", weights_only=False)
        det.load_state_dict(ck["det"])
        kpn.load_state_dict(ck["kpn"])
        print(f"initialised from {init}")
    params = list(det.parameters()) + list(kpn.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05,
                                                anneal_strategy="cos")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
                    drop_last=len(ds) >= a.batch, persistent_workers=a.workers > 0, pin_memory=False)
    out = Path(a.out) if a.out else CKPT_DIR / f"{a.setting}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    it, t_start = 0, time.time()
    det.train()
    kpn.train()
    while it < iters:
        for batch in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_heat, l_size, l_kp = losses(det, kpn, batch, device)
            loss = l_heat + 0.5 * l_size + l_kp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            it += 1
            if it % a.log_every == 0 or it == 1:
                el = time.time() - t_start
                print(f"it {it}/{iters} loss {loss.item():.4f} heat {l_heat.item():.4f} size {l_size.item():.3f} "
                      f"kp {l_kp.item():.4f} lr {sched.get_last_lr()[0]:.2e} {el / it:.3f}s/it", flush=True)
            if it % 2000 == 0 or it == iters:
                torch.save({"det": det.state_dict(), "kpn": kpn.state_dict(), "setting": a.setting,
                            "iters": it, "train": [str(p) for p in paths]}, out)
            if it >= iters:
                break
    print(f"saved {out}")
    return 0 if math.isfinite(loss.item()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
