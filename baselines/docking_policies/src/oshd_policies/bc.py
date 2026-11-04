"""Behaviour-cloning students (paper Table III): one per input modality, offline from the dataset.

    uv run policy-bc-train --cache ../../work/bc_cache/docking_train_v3 --input rgb \\
        --out ../../results/policies/bc/rgb

A student maps (sensor input, proprioception, docking command) at step ``k`` to the thrust the
teacher computed from that step's observation (recorded at row ``k + 1``, see ``teacher.py``;
``inputs.applied``: eight thrusters in [0, 1]); MSE loss. Inputs (``inputs.py``):
``rgb``, ``depth``, ``events``, ``lidar``, or ``fused`` (all four, one encoder each).

- image encoder: 4 strided convs (32-64-128-128) -> 4x6 grid -> linear 256 (keeps where the
  target is in the image); random shifts of +-6 px during training
- lidar encoder: convs over the 16 x 360 range image, circular along the azimuth -> linear 256
- head: [encoders, proprio (standardised), command] -> 512 -> 256 -> 8, sigmoid
- AdamW 3e-4, cosine schedule, batch 256, ``--epochs`` over the train rows; 5% of the train
  trajectories are held out for the validation loss (``--val-frac``)

Writes ``student.pt`` (weights, input, normalisation, config) and ``train.json`` (losses).
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

from . import inputs as I

INPUTS = {"rgb": ("rgb",), "depth": ("depth",), "events": ("events",), "lidar": ("lidar",),
          "fused": ("rgb", "depth", "events", "lidar"), "state": ()}
#: ``state``: a diagnostic student on the teacher's own 41-d observation (privileged goal terms
#: instead of a sensor; no command), i.e. behaviour cloning without perception.
STATE_DIM = 41
#: Auxiliary targets: ``tf_pos_robot_to_goal`` (obs 8:11) and ``tf_rot6d_robot_to_goal`` (17:23).
AUX_SLICES = (slice(8, 11), slice(17, 23))
AUX_DIM = 9


class ImageEncoder(nn.Module):
    def __init__(self, c: int, out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(), nn.Conv2d(128, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(), nn.Linear(128 * 24, out), nn.LayerNorm(out), nn.GELU())

    def forward(self, x):
        return self.net(x)


class LidarEncoder(nn.Module):
    def __init__(self, out: int = 256):
        super().__init__()
        conv = lambda i, o, s: nn.Conv2d(i, o, (3, 5), s, (1, 2), padding_mode="circular")
        self.net = nn.Sequential(
            conv(1, 32, (1, 2)), nn.GELU(), conv(32, 64, (2, 2)), nn.GELU(), conv(64, 128, (2, 2)), nn.GELU(),
            conv(128, 128, (2, 2)), nn.GELU(), nn.AdaptiveAvgPool2d((2, 12)), nn.Flatten(),
            nn.Linear(128 * 24, out), nn.LayerNorm(out), nn.GELU())

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    def __init__(self, input_name: str):
        super().__init__()
        self.input_name = input_name
        self.modalities = INPUTS[input_name]
        self.enc = nn.ModuleDict({m: LidarEncoder() if m == "lidar" else ImageEncoder(I_CHANNELS[m])
                                  for m in self.modalities})
        self.pdim = STATE_DIM if input_name == "state" else I.PROPRIO_DIM
        d = 256 * len(self.modalities) + self.pdim + (0 if input_name == "state" else 3)
        # Perception head: the teacher's goal terms (position and attitude of the goal in the
        # robot frame, AUX_DIM), predicted from the sensors, proprioception and command, trained
        # with an auxiliary loss on the recorded terms and fed to the action head.
        self.aux = None
        if self.modalities:
            self.aux = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, AUX_DIM))
            d += AUX_DIM
        self.register_buffer("g_mean", torch.zeros(AUX_DIM))
        self.register_buffer("g_std", torch.ones(AUX_DIM))
        self.head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 8))
        self.register_buffer("p_mean", torch.zeros(self.pdim))
        self.register_buffer("p_std", torch.ones(self.pdim))

    def forward(self, sensors: dict[str, torch.Tensor], proprio: torch.Tensor, command: torch.Tensor):
        """``sensors``: uint8 tensors as in ``inputs.py`` (B, C, H, W)."""
        z = [self.enc[m](sensors[m].float() / 255.0) for m in self.modalities]
        p = (proprio - self.p_mean) / self.p_std
        extra = [] if self.input_name == "state" else [command]
        x = torch.cat([*z, p, *extra], -1)
        self.goal_pred = None
        if self.aux is not None:
            self.goal_pred = self.aux(x)  # standardised goal terms
            x = torch.cat([x, self.goal_pred], -1)
        return torch.sigmoid(self.head(x))


I_CHANNELS = {"rgb": 3, "depth": 1, "events": 2, "lidar": 1}


def random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """DrQ-style random translation (replicate padding, per-sample integer shift)."""
    b, c, h, w = x.shape
    x = F.pad(x.float(), (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    iy = (torch.arange(h, device=x.device)[None, :] + oy[:, None])
    ix = (torch.arange(w, device=x.device)[None, :] + ox[:, None])
    x = x[torch.arange(b, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None],
          iy[:, None, :, None], ix[:, None, None, :]]
    return x


def state_observations(names: list[str]) -> np.ndarray:
    """The teacher's 41-d observation of every step (``steps``), row-aligned with the cache."""
    from concurrent.futures import ThreadPoolExecutor

    from oshd import OshDataset

    ds = OshDataset()
    with ThreadPoolExecutor(16) as ex:
        obs = list(ex.map(lambda n: ds[n].steps()["observation"].astype(np.float32), names))
    return np.concatenate(obs)


def load_student(path: Path, device: str = "cuda") -> Student:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Student(ck["input"]).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policy-bc-train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--input", choices=list(INPUTS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--aux", type=float, default=0.01, help="weight of the goal-term loss")
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
    rows = pd.read_parquet(a.cache / "rows.parquet")
    mods = INPUTS[a.input]
    t0 = time.time()
    data = {m: np.fromfile(a.cache / f"{m}.u8", np.uint8).reshape(cache["modalities"][m]) for m in mods}
    print(f"loaded {', '.join(mods)} ({sum(d.nbytes for d in data.values()) / 1e9:.1f} GB) in {time.time() - t0:.0f} s",
          flush=True)
    proprio = np.stack(rows["proprio"]).astype(np.float32)
    obs41 = state_observations(cache["trajectories"])
    goal_terms = np.concatenate([obs41[:, sl] for sl in AUX_SLICES], 1)
    if a.input == "state":
        proprio = obs41
    command = np.stack(rows["command"]).astype(np.float32)
    # The action the teacher computed from row k's observation is recorded at row k + 1.
    action = np.stack(rows["action"]).astype(np.float32)
    same = (rows["trajectory"].to_numpy()[1:] == rows["trajectory"].to_numpy()[:-1])
    valid = np.concatenate([same, [False]])
    target = np.zeros_like(action)
    target[:-1] = I.applied(action[1:])
    trajs = np.array(cache["trajectories"])
    val_trajs = set(rng.choice(trajs, max(1, int(a.val_frac * len(trajs))), replace=False))
    is_val = rows["trajectory"].isin(val_trajs).to_numpy()
    tr_idx, va_idx = np.nonzero(~is_val & valid)[0], np.nonzero(is_val & valid)[0]

    dev = a.device
    model = Student(a.input).to(dev)
    model.p_mean.copy_(torch.as_tensor(proprio[tr_idx].mean(0)))
    model.p_std.copy_(torch.as_tensor(proprio[tr_idx].std(0) + 1e-3))
    model.g_mean.copy_(torch.as_tensor(goal_terms[tr_idx].mean(0)))
    model.g_std.copy_(torch.as_tensor(goal_terms[tr_idx].std(0) + 1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    iters = a.epochs * (len(tr_idx) // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i, iters) / iters)))
    P, C, Y = (torch.as_tensor(x) for x in (proprio, command, target))
    G = (torch.as_tensor(goal_terms) - model.g_mean.cpu()) / model.g_std.cpu()

    def batch(idx: np.ndarray, train: bool):
        idx = np.sort(idx)
        s = {m: torch.from_numpy(data[m][idx]).to(dev, non_blocking=True) for m in mods}
        if train and a.shift:
            s = {m: (random_shift(v, a.shift) if m != "lidar" else torch.roll(
                v, int(rng.integers(-3, 4)), dims=-1)) for m, v in s.items()}
        return s, P[idx].to(dev), C[idx].to(dev), Y[idx].to(dev), G[idx].to(dev)

    tot_g = [0.0]

    def evaluate() -> float:
        model.eval()
        tot, n = 0.0, 0
        tot_g[0] = 0.0
        with torch.no_grad():
            for i in range(0, len(va_idx), 1024):
                s, p, c, y, g = batch(va_idx[i:i + 1024], False)
                tot += F.mse_loss(model(s, p, c), y, reduction="sum").item() / 8
                if model.goal_pred is not None:
                    tot_g[0] += F.mse_loss(model.goal_pred, g, reduction="sum").item() / AUX_DIM
                n += len(y)
        model.train()
        return tot / max(n, 1)

    log = {"input": a.input, "train_rows": len(tr_idx), "val_rows": len(va_idx), "epochs": []}
    it = 0
    for ep in range(a.epochs):
        perm = rng.permutation(tr_idx)
        run, t_ep = 0.0, time.time()
        for b in range(len(perm) // a.batch):
            s, p, c, y, g = batch(perm[b * a.batch:(b + 1) * a.batch], True)
            loss = F.mse_loss(model(s, p, c), y)
            if model.goal_pred is not None:
                # thrust errors are ~0.03, goal-term errors (standardised) ~0.1-1: weight a.aux
                loss = loss + a.aux * F.mse_loss(model.goal_pred, g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            it += 1
        val = evaluate()
        e = {"epoch": ep + 1, "train_loss": run / max(1, len(perm) // a.batch), "val_mse": val,
             "val_goal_mse": tot_g[0] / max(len(va_idx), 1),
             "minutes": (time.time() - t_ep) / 60}
        log["epochs"].append(e)
        print(json.dumps(e), flush=True)
        a.out.mkdir(parents=True, exist_ok=True)
        torch.save({"input": a.input, "state_dict": model.state_dict(), "config": vars(a) | {"cache": str(a.cache),
                    "out": str(a.out)}, "val_trajectories": sorted(val_trajs)}, a.out / "student.pt")
        (a.out / "train.json").write_text(json.dumps(log, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
