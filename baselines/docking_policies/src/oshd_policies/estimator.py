"""Learned target-position estimators (paper Table III, "learned estimator -> PPO").

    uv run policy-estimator-train --cache ../../work/bc_cache/docking_train_v3 --input rgb \\
        --out ../../results/policies/estimator/rgb

One network per input (``rgb``, ``depth``, ``events``, ``lidar``, ``fused``; ``inputs.py``, the
same encoders as ``bc.py``) regresses the position of the target's centre in the platform's base
frame, ``p = R_robot^T (c - p_robot)``, with a per-axis standard deviation (Gaussian negative
log-likelihood): where the target is out of view or ambiguous the network reports a large
uncertainty and the filter (``modular.StaticTargetFilter``) ignores it. Labels come from the
released mocap (``robot`` and ``target`` bodies at the step's time; the target centre is the
body origin plus the scenario's ``target_center_offset``, zero except for the static twins).
Inputs are the sensor alone: the camera and the lidar are fixed on the base. A second head
predicts whether the target is in view (the centre projects into the camera image, from the
released calibration; always for inputs with the lidar): the position loss and the std
calibration use the frames in view only, and the closed loop ignores estimates whose target is
not predicted in view (out of view, a regression returns the prior with a small std).

- encoders as ``bc.py``; random shifts of +-6 px for the images during training (the lidar
  range image is not augmented: a shift in azimuth would change the label)
- head: [encoders] -> 512 -> 256 -> 6 (mean, log-std per axis; log-std clamped to [-5, 2])
- AdamW 3e-4, cosine schedule, batch 256, 20 epochs; 5% of the train trajectories held out

Writes ``estimator.pt`` and ``train.json`` (validation errors and the share within 1 sigma; the
predicted std is calibrated on the held-out trajectories after the last epoch, ``sd_scale``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bc import I_CHANNELS, INPUTS, ImageEncoder, LidarEncoder, random_shift


class Estimator(nn.Module):
    def __init__(self, input_name: str):
        super().__init__()
        self.input_name = input_name
        self.modalities = INPUTS[input_name]
        self.enc = nn.ModuleDict({m: LidarEncoder() if m == "lidar" else ImageEncoder(I_CHANNELS[m])
                                  for m in self.modalities})
        self.head = nn.Sequential(nn.Linear(256 * len(self.modalities), 512), nn.GELU(), nn.Linear(512, 256),
                                  nn.GELU(), nn.Linear(256, 7))
        self.register_buffer("y_mean", torch.zeros(3))
        self.register_buffer("y_std", torch.ones(3))
        #: per-axis factor on the predicted std, calibrated on the held-out trajectories so that
        #: the median |error| / std is that of a Gaussian (0.6745)
        self.register_buffer("sd_scale", torch.ones(3))

    def forward(self, sensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the target centre in the base frame (m) and its per-axis std (m); the
        logit of the target being in view is left in ``self.vis_logit``."""
        z = torch.cat([self.enc[m](sensors[m].float() / 255.0) for m in self.modalities], -1)
        out = self.head(z)
        mu = out[:, :3] * self.y_std + self.y_mean
        log_s = out[:, 3:6].clamp(-5.0, 2.0)
        self.vis_logit = out[:, 6]
        return mu, torch.exp(log_s) * self.y_std * self.sd_scale


def load_estimator(path, device: str = "cuda") -> Estimator:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Estimator(ck["input"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def _target_rel(name: str) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    from oshd import OshDataset

    t = OshDataset()[name]
    tk = t.steps()["timestamp_ns"]

    def at(body):
        d = t.mocap(body)
        i = np.clip(np.searchsorted(d["timestamp_ns"], tk, "right") - 1, 0, None)
        return np.stack([d[c][i] for c in ("x", "y", "z", "qx", "qy", "qz", "qw")], 1)

    rob, tgt = at("robot"), at("target")
    off = np.zeros(3)
    for o in t.metadata["scenario"]["overrides"]:
        if "target_center_offset" in o:
            off = np.asarray(json.loads(o.split("=", 1)[1]), float)
    centre = tgt[:, :3] + R.from_quat(tgt[:, 3:]).apply(off)
    rel = R.from_quat(rob[:, 3:]).inv().apply(centre - rob[:, :3])
    # in view of the camera: the centre projects inside the image (5% margin), in front of it
    cal = t.calibration
    T_bc = np.asarray(cal.T_body_cam("rgb"), float)
    K = np.asarray(cal.intrinsics("rgb").K, float).reshape(3, 3)
    pc = (rel - T_bc[:3, 3]) @ T_bc[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]
        v = K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]
    w, h = 2 * K[0, 2], 2 * K[1, 2]
    vis = (pc[:, 2] > 0.1) & (u > 0.05 * w) & (u < 0.95 * w) & (v > 0.05 * h) & (v < 0.95 * h)
    return np.concatenate([rel, vis[:, None]], 1).astype(np.float32)


def target_labels(names: list[str]) -> np.ndarray:
    """Target centre in the base frame at every step, row-aligned with the cache."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as ex:
        return np.concatenate(list(ex.map(_target_rel, names)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policy-estimator-train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--input", choices=[k for k in INPUTS if k != "state"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shift", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cache = json.loads((a.cache / "cache.json").read_text())
    rows = pd.read_parquet(a.cache / "rows.parquet", columns=["trajectory", "step"])
    mods = INPUTS[a.input]
    t0 = time.time()
    data = {m: np.fromfile(a.cache / f"{m}.u8", np.uint8).reshape(cache["modalities"][m]) for m in mods}
    labels = target_labels(cache["trajectories"])
    assert len(labels) == len(rows)
    # LiDAR sees all around: the target is in view of any input that includes it
    vis = np.ones(len(labels), bool) if "lidar" in mods else labels[:, 3] > 0.5
    labels = labels[:, :3]
    print(f"target in view in {vis.mean():.0%} of the rows", flush=True)
    print(f"loaded {', '.join(mods)} and labels in {time.time() - t0:.0f} s", flush=True)
    trajs = np.array(cache["trajectories"])
    val_trajs = set(rng.choice(trajs, max(1, int(a.val_frac * len(trajs))), replace=False))
    is_val = rows["trajectory"].isin(val_trajs).to_numpy()
    tr_idx, va_idx = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]

    dev = a.device
    model = Estimator(a.input).to(dev)
    model.y_mean.copy_(torch.as_tensor(labels[tr_idx][vis[tr_idx]].mean(0)))
    model.y_std.copy_(torch.as_tensor(labels[tr_idx][vis[tr_idx]].std(0) + 1e-3))
    Y = torch.as_tensor(labels)
    V = torch.as_tensor(vis.astype(np.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    iters = a.epochs * (len(tr_idx) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, iters) / iters)))

    def batch(idx, train):
        idx = np.sort(idx)
        s = {m: torch.from_numpy(data[m][idx]).to(dev, non_blocking=True) for m in mods}
        if train and a.shift:
            s = {m: (random_shift(v, a.shift) if m != "lidar" else v) for m, v in s.items()}
        return s, Y[idx].to(dev), V[idx].to(dev)

    def nll(mu, sd, y, v):
        per = (((mu - y) / sd) ** 2 / 2 + torch.log(sd)).mean(1)
        return (per * v).sum() / v.sum().clamp(min=1.0)

    def evaluate():
        model.eval()
        errs, z, vv, pv = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(va_idx), 1024):
                s, y, v = batch(va_idx[i:i + 1024], False)
                mu, sd = model(s)
                errs.append((mu - y).cpu().numpy())
                z.append(((mu - y).abs() / sd).cpu().numpy())
                vv.append(v.cpu().numpy())
                pv.append((model.vis_logit > 0).float().cpu().numpy())
        model.train()
        e, z, vv, pv = (np.concatenate(x) for x in (errs, z, vv, pv))
        m = vv > 0.5
        d = np.linalg.norm(e[m, :2], axis=1)
        return {"val_rmse_xy_m": float(np.sqrt((d ** 2).mean())), "val_median_xy_m": float(np.median(d)),
                "val_within_1sigma": float((z[m, :2] < 1).mean()), "val_visibility_acc": float((pv == vv).mean())}

    log = {"input": a.input, "train_rows": len(tr_idx), "val_rows": len(va_idx), "epochs": []}
    for ep in range(a.epochs):
        perm = rng.permutation(tr_idx)
        run, t_ep = 0.0, time.time()
        for b in range(len(perm) // a.batch):
            s, y, v = batch(perm[b * a.batch:(b + 1) * a.batch], True)
            mu, sd = model(s)
            loss = nll(mu, sd, y, v) + F.binary_cross_entropy_with_logits(model.vis_logit, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        e = {"epoch": ep + 1, "train_nll": run / max(1, len(perm) // a.batch), **evaluate(),
             "minutes": (time.time() - t_ep) / 60}
        if ep + 1 == a.epochs:  # calibrate the std on the held-out trajectories
            model.sd_scale.fill_(1.0)
            model.eval()
            zs = []
            with torch.no_grad():
                for i in range(0, len(va_idx), 1024):
                    s, y, v = batch(va_idx[i:i + 1024], False)
                    mu, sd = model(s)
                    zs.append(((mu - y).abs() / sd)[v > 0.5].cpu().numpy())
            model.train()
            model.sd_scale.copy_(torch.as_tensor(np.median(np.concatenate(zs), 0) / 0.6745, dtype=torch.float32))
            e.update(evaluate(), sd_scale=model.sd_scale.tolist())
        log["epochs"].append(e)
        print(json.dumps(e), flush=True)
        a.out.mkdir(parents=True, exist_ok=True)
        torch.save({"input": a.input, "state_dict": model.state_dict(), "config": vars(a) | {"cache": str(a.cache),
                    "out": str(a.out)}, "val_trajectories": sorted(val_trajs)}, a.out / "estimator.pt")
        (a.out / "train.json").write_text(json.dumps(log, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
