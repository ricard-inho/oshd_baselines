"""Causal E2VID reconstruction at the POOH-D event sample times.

For every trajectory of a GT sample list (``results/gt/events/<domain>_<split>.parquet``)
the whole event stream, from its first event up to the last sample time, is fed through
the recurrent E2VID network in temporal order (no look-ahead). Windowing follows the
upstream ``run_reconstruction.py`` defaults:

* ``--mode count`` (default): fixed-size windows of N = num_events_per_pixel * W * H events
  (upstream default 0.35 -> 322 560 events for the 1280x720 EVK4). The recurrent state is
  only advanced by complete windows. At a sample time t the frame is produced by one extra
  forward pass on the not-yet-complete window [last window end, t) from the current state,
  whose updated state is discarded, so the frame at t uses exactly the events before t.
* ``--mode duration``: fixed-duration windows equal to the sample period (upstream
  ``--fixed_duration``), aligned with the 20 Hz sample grid; the frame at t is the output of
  the window that ends at t.

Output (the contract other baselines consume)::

    results/e2vid/<domain>_<split>/<trajectory>/<t_ns>.png   uint8 grayscale, 1280x720,
                                                            raw event-sensor pixels (distorted)
    results/e2vid/<domain>_<split>/index.parquet            trajectory, t_ns, path, runtime_ms
    results/e2vid/<domain>_<split>/index.json               settings + hardware

``path`` is relative to ``results_dir()``. ``runtime_ms`` is the amortised cost of a 20 Hz
streaming reconstructor: the GPU/CPU time (voxel grids + network + post-processing, not
disk I/O) spent between the previous point of the 20 Hz sample grid and this sample,
including that sample's partial-window pass.

This module never reads the ``T`` column of the sample lists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pooh_bench import load_config, open_dataset, repo_root, results_dir
from pooh_bench.gt import sample_times
from pooh_bench.predictions import hardware
from pooh_dataset.events import EVENT_SENSOR_SIZE
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[2]
UPSTREAM = HERE / "third_party" / "rpg_e2vid"
DEFAULT_WEIGHTS = HERE / "weights" / "E2VID_lightweight.pth.tar"
E2VID_COMMIT = "d0a7c005f460f2422f2a4bf605f70820ea7a1e5f"
H, W = EVENT_SENSOR_SIZE


def _import_upstream():
    if not (UPSTREAM / "model").is_dir():
        raise SystemExit(f"{UPSTREAM} missing: run baselines/e2vid/setup.sh")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import model.model as upstream_model  # noqa: F401  (registers E2VID / E2VIDRecurrent)
    from utils import inference_utils

    return upstream_model, inference_utils


def load_e2vid(path: Path, device: torch.device):
    """Same as upstream ``utils.loading_utils.load_model`` (torch>=2.6 needs weights_only=False)."""
    upstream_model, _ = _import_upstream()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    arch = raw["arch"]
    model_type = raw.get("model", None) or raw["config"]["model"]
    model = getattr(upstream_model, arch)(model_type)
    model.load_state_dict(raw["state_dict"])
    return model.to(device).eval()


class Reconstructor:
    """Upstream ImageReconstructor.update_reconstruction, split into commit / peek."""

    def __init__(self, model, device, unsharp_amount=0.3, unsharp_sigma=1.0):
        _, iu = _import_upstream()
        self.iu = iu
        self.model, self.device = model, device
        self.num_bins = model.num_bins
        self.crop = iu.CropParameters(W, H, model.num_encoders)
        self.pre = iu.EventPreprocessor(Namespace(no_normalize=False, hot_pixels_file=None, flip=False))
        self.rescale = iu.IntensityRescaler(Namespace(auto_hdr=False, auto_hdr_median_filter_size=10,
                                                      Imin=0.0, Imax=1.0))
        self.unsharp = iu.UnsharpMaskFilter(Namespace(unsharp_mask_amount=unsharp_amount,
                                                      unsharp_mask_sigma=unsharp_sigma), device=device)
        self.state = None
        self.last = np.full((H, W), 0, np.uint8)  # before any events: black (upstream initial state)

    @torch.no_grad()
    def _forward(self, ev: np.ndarray, state):
        """ev: (N, 4) float64 [t_s, x, y, p(0/1)] -> (uint8 frame, new state)."""
        vox = self.iu.events_to_voxel_grid_pytorch(ev, self.num_bins, W, H, self.device)
        x = self.pre(vox.unsqueeze(0))
        img, new_state = self.model(self.crop.pad(x), state)
        img = self.rescale(self.unsharp(img))
        c = self.crop
        return img[0, 0, c.iy0:c.iy1, c.ix0:c.ix1].cpu().numpy(), new_state

    def commit(self, ev: np.ndarray) -> np.ndarray:
        self.last, self.state = self._forward(ev, self.state)
        return self.last

    def peek(self, ev: np.ndarray) -> np.ndarray:
        if len(ev) == 0:
            return self.last
        out, _ = self._forward(ev, self.state)
        return out


class EventStream:
    """Sequential reader: events in [t_a, t_b) as (N, 4) float64 [t_s, x, y, p], read in slabs."""

    def __init__(self, traj, t_origin: int, slab_ns: int = 500_000_000):
        self.traj, self.origin, self.slab = traj, int(t_origin), slab_ns
        self.buf_t0 = self.buf_t1 = None
        self.buf = None

    def _load(self, t_a: int):
        e = self.traj.events(t_a, t_a + self.slab)
        keep = (e.x < W) & (e.y < H)
        t = e.t[keep]
        self.buf_ts = t
        self.buf = np.stack([(t - self.origin).astype(np.float64) * 1e-9, e.x[keep].astype(np.float64),
                             e.y[keep].astype(np.float64), e.p[keep].astype(np.float64)], -1)
        self.buf_t0, self.buf_t1 = t_a, t_a + self.slab

    def get(self, t_a: int, t_b: int) -> np.ndarray:
        parts = []
        while t_a < t_b:
            if self.buf is None or not (self.buf_t0 <= t_a < self.buf_t1):
                self._load(t_a)
            hi = min(t_b, self.buf_t1)
            i0, i1 = np.searchsorted(self.buf_ts, [t_a, hi])
            parts.append(self.buf[i0:i1])
            t_a = hi
        return np.concatenate(parts) if parts else np.zeros((0, 4))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reconstruct_trajectory(traj, need_ts: np.ndarray, rec: Reconstructor, out_dir: Path, mode: str,
                           n_events: int, rate_hz: float, window_ms: float, rel_root: Path):
    """Run E2VID causally along ``traj``; save frames at ``need_ts``. Returns index rows."""
    device = rec.device
    t_first, _ = traj.time_range("events")
    _, grid = sample_times(traj, "events", rate_hz, window_ms)
    need = set(int(t) for t in need_ts)
    grid = np.union1d(grid, need_ts)            # make sure every requested time is a grid point
    grid = grid[grid <= need_ts.max()]
    stream = EventStream(traj, t_first)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = np.zeros((0, 4))
    t_prev = int(t_first)
    rows = []
    for g in tqdm(grid, desc=traj.name, leave=False):
        g = int(g)
        new = stream.get(t_prev, g)
        t_prev = g
        _sync(device)
        tic = time.perf_counter()
        if mode == "duration":
            frame = rec.commit(new) if len(new) else rec.last
        else:
            pending = np.concatenate([pending, new]) if len(pending) else new
            while len(pending) >= n_events:
                rec.commit(pending[:n_events])
                pending = pending[n_events:]
            frame = rec.peek(pending) if g in need else None
        _sync(device)
        dt_ms = (time.perf_counter() - tic) * 1e3
        if g in need:
            p = out_dir / f"{g}.png"
            Image.fromarray(np.ascontiguousarray(frame)).save(p, compress_level=1)
            rows.append({"trajectory": traj.name, "t_ns": g, "path": str(p.relative_to(rel_root)),
                         "runtime_ms": dt_ms})
    return rows


def run(domain: str, split: str, limit: int | None, mode: str, num_events_per_pixel: float,
        window_events: int | None, weights: Path, device: torch.device, trajectories=None) -> Path | None:
    res = results_dir()
    cfg = load_config(repo_root() / "configs" / "bench.yaml")
    s = cfg.get("sampling", {})
    rate_hz, window_ms = float(s.get("events_rate_hz", 20)), float(s.get("events_window_ms", 50))
    gt_file = res / "gt" / "events" / f"{domain}_{split}.parquet"
    if not gt_file.exists():
        print(f"skip {domain}_{split}: {gt_file} missing")
        return None
    gt = pd.read_parquet(gt_file, columns=["trajectory", "t_ns"])  # never the T column
    gt = gt.drop_duplicates().sort_values(["trajectory", "t_ns"])
    if trajectories:
        gt = gt[gt["trajectory"].isin(trajectories)]
    if limit:
        gt = gt.head(limit)
    ds = open_dataset(cfg)
    n_events = int(window_events or round(num_events_per_pixel * W * H))
    model = load_e2vid(weights, device)
    out_root = res / "e2vid" / f"{domain}_{split}"
    rows = []
    for tname, g in gt.groupby("trajectory", sort=True):
        rec = Reconstructor(model, device)  # fresh recurrent state per trajectory
        rows += reconstruct_trajectory(ds[tname], g["t_ns"].to_numpy(np.int64), rec, out_root / tname,
                                       mode, n_events, rate_hz, window_ms, res)
    df = pd.DataFrame(rows, columns=["trajectory", "t_ns", "path", "runtime_ms"])
    out_root.mkdir(parents=True, exist_ok=True)
    idx = out_root / "index.parquet"
    df.to_parquet(idx, index=False)
    meta = {
        "method": "E2VID (rpg_e2vid)", "upstream_commit": E2VID_COMMIT, "weights": weights.name,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "mode": mode,
        "events_per_window": n_events if mode == "count" else None,
        "window_ms": 1000.0 / rate_hz if mode == "duration" else None,
        "unsharp_mask": [0.3, 1.0], "auto_hdr": False, "image_size": [W, H], "undistorted": False,
        "limit": limit, "n_frames": len(df), "hardware": hardware(),
        "runtime_ms": "amortised per 20 Hz grid point: voxel grid + network + post-processing, no disk I/O",
    }
    idx.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    med = df["runtime_ms"].median() if len(df) else float("nan")
    print(f"{idx}: {len(df)} frames, median runtime {med:.1f} ms")
    return idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", default="real")
    ap.add_argument("--split", default="test")
    ap.add_argument("--all", action="store_true",
                    help="every results/gt/events/<domain>_<split>.parquet (train splits too)")
    ap.add_argument("--limit", type=int, default=None, help="first N samples of the list (smoke tests)")
    ap.add_argument("--trajectory", action="append", default=None)
    ap.add_argument("--mode", choices=["count", "duration"], default="count")
    ap.add_argument("--num-events-per-pixel", type=float, default=0.35, help="upstream default")
    ap.add_argument("--window-events", type=int, default=None, help="override N (count mode)")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0")
    if a.all:
        jobs = [tuple(p.stem.split("_", 1)) for p in sorted((results_dir() / "gt" / "events").glob("*.parquet"))]
    else:
        jobs = [(a.domain, a.split)]
    for domain, split in jobs:
        run(domain, split, a.limit, a.mode, a.num_events_per_pixel, a.window_events, a.weights, device,
            a.trajectory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
