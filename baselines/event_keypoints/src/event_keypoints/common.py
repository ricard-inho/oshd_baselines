"""Shared pieces: GPU voxel grid, keypoints, symmetry canonicalisation, crops, data loading.

Conventions
-----------
* Image coordinates follow OpenCV / pooh-dataset (pixel centres at integers). Crops are
  axis-aligned squares ``(cx, cy, side)`` in full-resolution event-camera pixels.
* Keypoints (model frame, metres): the 8 cube-body corners of ``models/objects.yaml``
  followed by the symmetry centre (0, 0.05, 0).
* Symmetry: the cube is 24-fold symmetric. Training labels are rendered from the
  *canonical* ground-truth pose ``T @ S_k`` with ``k = argmin_k angle(R_REF^T R S_k)``, i.e.
  the symmetric equivalent whose rotation (in the event-camera frame) is closest to the fixed
  reference ``R_REF = I``. The label of a corner therefore depends only on how the cube looks
  from the camera (which octant of the camera frame the corner points to), never on the
  hidden marker holder.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

H_FULL, W_FULL = 720, 1280
NUM_BINS = 5
WINDOW_MS = 50.0
DET_DOWN = 2            # stage A runs on a voxel grid at 1/2 resolution (360x640, padded to 384)
DET_PAD_H = 384
DET_STRIDE = 4          # stage A output stride w.r.t. its input -> 96x160 heatmap
CROP = 256              # stage B input side
HEAT = 64               # stage B heatmap side
CROP_MARGIN = 1.3       # crop side = margin * max(w, h) of the projected keypoint box
R_REF = np.eye(3)


# --------------------------------------------------------------------------- dataset
def dataset_and_results():
    import pooh_bench

    return pooh_bench.open_dataset(), pooh_bench.results_dir()


@lru_cache(maxsize=1)
def _dataset_cached(pid: int):  # one PoohDataset per process (DataLoader workers)
    import pooh_bench

    return pooh_bench.open_dataset()


def get_dataset():
    return _dataset_cached(os.getpid())


def bench_cfg() -> dict:
    import pooh_bench

    return pooh_bench.load_config(pooh_bench.repo_root() / "configs" / "bench.yaml")


def object_keypoints(obj) -> np.ndarray:
    """(9, 3): the 8 cube-body corners + the symmetry centre."""
    kp = obj.click_points
    c = (kp.min(0) + kp.max(0)) / 2
    return np.concatenate([kp, c[None]], 0)


def symmetry(obj):
    from pooh_bench import symmetry_for

    return symmetry_for(obj, bench_cfg().get("symmetries", {}).get(obj.name))


def canonical_pose(T: np.ndarray, sym) -> np.ndarray:
    """Symmetric equivalent ``T @ S_k`` whose rotation is closest to ``R_REF``."""
    R = R_REF.T @ T[:3, :3] @ sym.rotations  # (K, 3, 3)
    tr = np.trace(R, axis1=-2, axis2=-1)
    return T @ sym.transforms[int(np.argmax(tr))]


def gt_columns_no_pose() -> list[str]:
    """Columns inference may read from a GT sample list (never ``T``)."""
    return ["trajectory", "domain", "sensor", "index", "t_ns", "object"]


def read_samples(path, with_pose: bool = False) -> pd.DataFrame:
    import pyarrow.parquet as pq

    cols = pq.read_schema(path).names
    use = [c for c in cols if with_pose or c != "T"]
    return pd.read_parquet(path, columns=use)


# --------------------------------------------------------------------------- voxels
def voxelize(x, y, t, p, bidx, B: int, H: int, W: int, t0, span, bins: int = NUM_BINS,
             down: int = 1) -> torch.Tensor:
    """(B, bins, H, W) voxel grid, bilinear in time, polarity +-1 (== pooh_dataset.events.voxel_grid).

    ``x, y`` int tensors in full-res pixels, ``t`` int64 ns, ``p`` 0/1, ``bidx`` sample index,
    ``t0`` (B,) window start (ns, int64), ``span`` window length (ns). ``down`` bins pixels.
    """
    dev = x.device
    grid = torch.zeros(B * bins * H * W, device=dev, dtype=torch.float32)
    if x.numel() == 0:
        return grid.view(B, bins, H, W)
    xs = torch.div(x.long(), down, rounding_mode="floor")
    ys = torch.div(y.long(), down, rounding_mode="floor")
    ok = (xs < W) & (ys < H)
    tn = (t - t0[bidx]).double() * (bins - 1) / span
    tn = tn.float()
    pol = p.float() * 2 - 1
    pix = bidx.long() * (bins * H * W) + ys * W + xs
    left = torch.floor(tn).long()
    for b, w in ((left, 1.0 - (tn - left)), (left + 1, tn - left)):
        m = ok & (b >= 0) & (b < bins)
        grid.index_add_(0, pix[m] + b[m] * H * W, pol[m] * w[m])
    return grid.view(B, bins, H, W)


def normalize_nonzero(v: torch.Tensor) -> torch.Tensor:
    """Per-sample standardisation over the non-zero voxels (E-RAFT / DSEC style)."""
    out = torch.zeros_like(v)
    for i in range(v.shape[0]):
        m = v[i] != 0
        if m.any():
            vals = v[i][m]
            std = vals.std() if vals.numel() > 1 else vals.new_tensor(1.0)
            out[i][m] = (vals - vals.mean()) / (std if std > 0 else 1.0)
    return out


def det_input(ev, B, t0, span) -> torch.Tensor:
    """Stage-A input (B, bins, 384, 640) from events on the GPU."""
    v = voxelize(*ev, B, H_FULL // DET_DOWN, W_FULL // DET_DOWN, t0, span, down=DET_DOWN)
    v = normalize_nonzero(v)
    return F.pad(v, (0, 0, 0, DET_PAD_H - v.shape[2]))


def full_grid(ev, B, t0, span) -> torch.Tensor:
    return voxelize(*ev, B, H_FULL, W_FULL, t0, span)


def crop_boxes(cx, cy, side) -> torch.Tensor:
    """(N, 4) xyxy in continuous coords (pixel i covers [i, i+1)) for roi_align(aligned=True)."""
    x1 = cx + 0.5 - side / 2
    y1 = cy + 0.5 - side / 2
    return torch.stack([x1, y1, x1 + side, y1 + side], -1)


def crop_input(grid: torch.Tensor, cx, cy, side) -> torch.Tensor:
    """Stage-B input: (N, bins, CROP, CROP) crops of the full voxel grid, re-normalised."""
    from torchvision.ops import roi_align

    boxes = crop_boxes(cx, cy, side)
    rois = torch.cat([torch.arange(len(boxes), device=grid.device, dtype=boxes.dtype)[:, None], boxes], 1)
    c = roi_align(grid, rois.float(), output_size=CROP, spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return normalize_nonzero(c)


def img_to_heat(u, c, side, n=HEAT):
    """Full-res pixel coord (one axis) -> heatmap coord of a crop centred at ``c`` (same axis)."""
    x1 = c + 0.5 - side / 2
    return (u + 0.5 - x1) / side * n - 0.5


def heat_to_img(h, c, side, n=HEAT):
    x1 = c + 0.5 - side / 2
    return x1 + (h + 0.5) * side / n - 0.5


def gaussian_maps(pts: torch.Tensor, H: int, W: int, sigma) -> torch.Tensor:
    """pts (N, K, 2) in heatmap px -> (N, K, H, W) unnormalised Gaussians (peak 1)."""
    ys = torch.arange(H, device=pts.device, dtype=torch.float32)
    xs = torch.arange(W, device=pts.device, dtype=torch.float32)
    sigma = torch.as_tensor(sigma, device=pts.device, dtype=torch.float32)
    if sigma.ndim == 1:
        sigma = sigma[:, None, None, None]
    dx = (xs[None, None, None, :] - pts[..., 0, None, None]) ** 2
    dy = (ys[None, None, :, None] - pts[..., 1, None, None]) ** 2
    return torch.exp(-(dx + dy) / (2 * sigma**2))


def subpixel_peaks(heat: torch.Tensor, r: int = 2):
    """heat (N, K, H, W) -> (N, K, 2) peak xy (soft-argmax in a (2r+1)^2 window), (N, K) value."""
    N, K, H, W = heat.shape
    flat = heat.view(N, K, -1)
    val, idx = flat.max(-1)
    py, px = (idx // W).float(), (idx % W).float()
    off = torch.arange(-r, r + 1, device=heat.device, dtype=torch.float32)
    gy = (py[..., None, None] + off[None, None, :, None]).clamp(0, H - 1)
    gx = (px[..., None, None] + off[None, None, None, :]).clamp(0, W - 1)
    gy, gx = torch.broadcast_tensors(gy, gx)
    lin = (gy * W + gx).long().view(N, K, -1)
    w = torch.gather(flat, 2, lin).clamp(min=0)
    ws = w.sum(-1).clamp(min=1e-6)
    x = (w * gx.reshape(N, K, -1)).sum(-1) / ws
    y = (w * gy.reshape(N, K, -1)).sum(-1) / ws
    return torch.stack([x, y], -1), val


# --------------------------------------------------------------------------- events io
def load_window(traj, t_ns: int, window_ms: float = WINDOW_MS):
    from pooh_bench.sensors import events_window

    ev = events_window(traj, int(t_ns), window_ms)
    return ev.x.astype(np.int16), ev.y.astype(np.int16), ev.t.astype(np.int64), ev.p.astype(np.uint8)


def events_to_device(batch_events, device):
    """list of (x, y, t, p) numpy -> concatenated GPU tensors (x, y, t, p, bidx)."""
    xs, ys, ts, ps, bs = [], [], [], [], []
    for i, (x, y, t, p) in enumerate(batch_events):
        xs.append(torch.from_numpy(x.astype(np.int32)))
        ys.append(torch.from_numpy(y.astype(np.int32)))
        ts.append(torch.from_numpy(t))
        ps.append(torch.from_numpy(p))
        bs.append(torch.full((len(t),), i, dtype=torch.int32))
    cat = [torch.cat(a).to(device, non_blocking=True) for a in (xs, ys, ts, ps, bs)]
    return tuple(cat)
