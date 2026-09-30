"""E-RAFT on POOH-D events: model loading, DSEC-convention voxel grids, resolution handling.

DSEC conventions reproduced from ``loader/loader_dsec.py`` + ``utils/dsec_utils.VoxelGrid``:

* Flow from ``t0`` to ``t1 = t0 + 100 ms``. ``image1`` = voxel grid of the events in
  ``[t0 - 100 ms, t0)`` ("event_volume_old"), ``image2`` = events in ``[t0, t1)``.
* 15 bins. Time is normalised per window by its *first and last event*
  (``(C-1)(t-t_first)/(t_last-t_first)``); polarity -> +-1; trilinear splatting in x, y, t.
* Normalisation: mean / std over the non-zero voxels of each grid.
* 12 refinement iterations, last prediction used.

Resolution: DSEC is 640x480; our Prophesee is 1280x720. By default (``scale = 0.5``) event
coordinates are mapped to a 640x360 grid with the pixel-centre convention
``u' = (u + 0.5) / 2 - 0.5`` (float coordinates, splatted bilinearly, exactly how E-RAFT
treats DSEC's rectified float coordinates). The network runs at 640x360 (E-RAFT pads to
multiples of 32), and its flow is upsampled bilinearly to 1280x720 and multiplied by 2.
We keep the raw (distorted) pixel grid because the ground-truth flow is defined there.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "E-RAFT"
WEIGHTS = ROOT / "weights"
H_FULL, W_FULL = 720, 1280
NUM_BINS = 15
DT_MS = 100


def load_model(ckpt: Path, device="cuda"):
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from model.eraft import ERAFT

    model = ERAFT(config={"subtype": "standard"}, n_first_channels=NUM_BINS)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model.to(device)


@lru_cache(maxsize=1)
def _dataset_cached(pid: int):
    import pooh_bench

    return pooh_bench.open_dataset()


def get_dataset():
    return _dataset_cached(os.getpid())


def load_events(traj, t_start_ns: int, t_end_ns: int):
    ev = traj.events(int(t_start_ns), int(t_end_ns))
    return ev.x.astype(np.float32), ev.y.astype(np.float32), ev.t.astype(np.int64), ev.p.astype(np.float32)


def voxel_grid_dsec(x, y, t, p, H: int, W: int, scale: float, device, bins: int = NUM_BINS) -> torch.Tensor:
    """E-RAFT ``VoxelGrid.convert`` (normalize=True) on the GPU. x, y full-res pixels (numpy)."""
    grid = torch.zeros(bins * H * W, device=device, dtype=torch.float32)
    if len(t) < 2:
        return grid.view(bins, H, W)
    xs = (torch.from_numpy(x).to(device) + 0.5) * scale - 0.5
    ys = (torch.from_numpy(y).to(device) + 0.5) * scale - 0.5
    tt = torch.from_numpy(t - t[0]).to(device).double()
    tt = tt / tt[-1] if tt[-1] > 0 else tt  # loader: t = (t - t[0]) / t[-1]
    tn = ((bins - 1) * (tt - tt[0]) / (tt[-1] - tt[0]) if tt[-1] > tt[0] else tt).float()
    val = 2 * torch.from_numpy(p).to(device) - 1
    x0, y0, t0 = xs.int(), ys.int(), tn.int()  # truncation, as upstream
    for xl in (x0, x0 + 1):
        for yl in (y0, y0 + 1):
            for tl in (t0, t0 + 1):
                m = (xl < W) & (xl >= 0) & (yl < H) & (yl >= 0) & (tl >= 0) & (tl < bins)
                w = val * (1 - (xl - xs).abs()) * (1 - (yl - ys).abs()) * (1 - (tl - tn).abs())
                idx = H * W * tl.long() + W * yl.long() + xl.long()
                grid.index_add_(0, idx[m], w[m])
    nz = grid != 0
    if nz.any():
        v = grid[nz]
        mean, std = v.mean(), v.std()
        grid[nz] = (v - mean) / std if std > 0 else v - mean
    return grid.view(bins, H, W)


def net_size(scale: float) -> tuple[int, int]:
    return int(round(H_FULL * scale)), int(round(W_FULL * scale))


def pair_inputs(traj, t0_ns: int, t1_ns: int, scale: float, device, events=None):
    """(image1, image2) each (1, 15, h, w): events in [t0 - (t1 - t0), t0) and [t0, t1)."""
    h, w = net_size(scale)
    dt = int(t1_ns) - int(t0_ns)
    ev_old = events[0] if events else load_events(traj, t0_ns - dt, t0_ns)
    ev_new = events[1] if events else load_events(traj, t0_ns, t1_ns)
    im1 = voxel_grid_dsec(*ev_old, h, w, scale, device)[None]
    im2 = voxel_grid_dsec(*ev_new, h, w, scale, device)[None]
    return im1, im2


def to_full_res(flow: torch.Tensor, scale: float) -> torch.Tensor:
    """(B, 2, h, w) flow at network resolution -> (B, 2, 720, 1280) in full-res pixels."""
    if flow.shape[-2:] == (H_FULL, W_FULL):
        return flow
    return F.interpolate(flow, size=(H_FULL, W_FULL), mode="bilinear", align_corners=False) / scale


def flow_to_color(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """(H, W, 2) -> (H, W, 3) uint8, HSV colour wheel (hue = direction, value = magnitude)."""
    import matplotlib.colors as mc

    u, v = flow[..., 0], flow[..., 1]
    mag = np.hypot(u, v)
    mx = max_mag or max(float(np.percentile(mag, 99)), 1e-6)
    hsv = np.stack([(np.arctan2(-v, -u) / np.pi + 1) / 2, np.clip(mag / mx, 0, 1), np.ones_like(u)], -1)
    hsv[..., 1] = np.clip(mag / mx, 0, 1)
    return (mc.hsv_to_rgb(hsv) * 255).astype(np.uint8)
