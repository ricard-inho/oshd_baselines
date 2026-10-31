"""Predict T_events_model for every sample of an events GT list (never reads the ``T`` column).

Pipeline per sample (timed as ``runtime_ms``, events already in host memory):
GPU voxelisation -> stage A (centre + box on the 1/2-res grid) -> 256x256 crop -> stage B
(9 keypoint heatmaps) -> EPnP + RANSAC -> Levenberg-Marquardt refinement on the inliers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import common as C
from .model import Detector, KeypointNet
from .train import CKPT_DIR


class Estimator:
    def __init__(self, ckpt: Path, device="cuda", det_thresh: float = 0.1, kp_thresh: float = 0.1):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.det, self.kpn = Detector(pretrained=False), KeypointNet(pretrained=False)
        self.det.load_state_dict(ck["det"])
        self.kpn.load_state_dict(ck["kpn"])
        self.det.to(device).eval()
        self.kpn.to(device).eval()
        self.device = torch.device(device)
        self.det_thresh, self.kp_thresh = det_thresh, kp_thresh

    @torch.no_grad()
    def networks(self, ev, t_end_ns: int):
        """-> dict(box=(cx, cy, side), det_score, kp (9, 2) full-res px, kp_conf (9,)) or None."""
        span = int(C.WINDOW_MS * 1e6)
        t0 = torch.tensor([t_end_ns - span], dtype=torch.int64, device=self.device)
        evd = C.events_to_device([ev], self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            heat, size = self.det(C.det_input(evd, 1, t0, span))
        heat = torch.sigmoid(heat.float())[0, 0]
        Hh, Wh = heat.shape
        val, idx = heat.view(-1).max(0)
        if val.item() < self.det_thresh:
            return None
        iy, ix = idx // Wh, idx % Wh
        s = C.DET_DOWN * C.DET_STRIDE
        # 3x3 soft-argmax for a sub-cell centre
        pk, _ = C.subpixel_peaks(heat[None, None], r=1)
        cx = (pk[0, 0, 0] + 0.5) * s - 0.5
        cy = (pk[0, 0, 1] + 0.5) * s - 0.5
        side = torch.exp(size[0, 0, iy, ix].float()).clamp(8, 4 * C.W_FULL) * C.CROP_MARGIN
        side = side.clamp(min=32)
        grid = C.full_grid(evd, 1, t0, span)
        cxv, cyv, sv = cx[None], cy[None], side[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hk = self.kpn(C.crop_input(grid, cxv, cyv, sv)).float()
        pts, conf = C.subpixel_peaks(hk)
        u = C.heat_to_img(pts[0, :, 0], cxv, sv)
        v = C.heat_to_img(pts[0, :, 1], cyv, sv)
        return {"box": (cx.item(), cy.item(), side.item()), "det_score": val.item(),
                "kp": torch.stack([u, v], -1).cpu().numpy().astype(np.float64),
                "kp_conf": conf[0].cpu().numpy().astype(np.float64)}

    def pnp(self, out, kp3: np.ndarray, K: np.ndarray, D: np.ndarray):
        kp, conf = out["kp"], out["kp_conf"]
        use = np.nonzero(conf >= self.kp_thresh)[0]
        if len(use) < 6:
            use = np.argsort(-conf)[:6]
        obj, img = kp3[use].astype(np.float64), kp[use].astype(np.float64)
        err = max(2.0, 0.04 * out["box"][2] / C.CROP_MARGIN)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, D, flags=cv2.SOLVEPNP_EPNP,
                                                 reprojectionError=err, iterationsCount=200, confidence=0.999)
        if not ok or inl is None or len(inl) < 4:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_EPNP)
            inl = np.arange(len(use))[:, None]
            if not ok:
                return None, 0.0
        inl = inl[:, 0]
        if len(inl) >= 4:
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, D, rvec, tvec)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3] = tvec[:, 0]
        if T[2, 3] <= 0:
            return None, 0.0
        return T, float(out["det_score"] * conf[use][inl].mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["ZS", "FT"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", default="real", choices=["real", "synthetic"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="default weights/trained/<setting>.pt")
    a = ap.parse_args(argv)

    import oshd_bench

    res = oshd_bench.results_dir()
    gp = oshd_bench.gt_path(res, "events", a.domain, a.split)
    if not gp.exists():
        print(f"no sample list {gp}: nothing to do")
        return 0
    samples = C.read_samples(gp)  # no T column
    assert "T" not in samples.columns
    if a.limit:
        samples = samples.iloc[:: max(1, len(samples) // a.limit)].iloc[: a.limit]
    ckpt = Path(a.ckpt) if a.ckpt else CKPT_DIR / f"{a.setting}.pt"
    est = Estimator(ckpt)
    ds = C.get_dataset()
    out = oshd_bench.prediction_path(res, "events", "event_keypoints", a.setting, a.domain, a.split)
    w = oshd_bench.PoseWriter(out, {"method": "event_keypoints", "setting": a.setting, "checkpoint": str(ckpt),
                                    "voxel_bins": C.NUM_BINS, "window_ms": C.WINDOW_MS,
                                    "runtime": "GPU voxelisation + stage A + crop + stage B + EPnP/RANSAC + LM "
                                               "(events already in memory; parquet read excluded)"})
    kp_cache = {}
    io_ms = []
    # warm-up (cuDNN autotune, CUDA context) so the first sample's runtime is not inflated
    r0 = samples.iloc[0]
    est.networks(C.load_window(ds[r0.trajectory], r0.t_ns), int(r0.t_ns))
    for r in samples.itertuples():
        traj = ds[r.trajectory]
        if r.object not in kp_cache:
            kp_cache[r.object] = C.object_keypoints(ds.objects[r.object])
        intr = traj.calibration.intrinsics("events")
        t_io = time.perf_counter()
        ev = C.load_window(traj, r.t_ns)
        io_ms.append((time.perf_counter() - t_io) * 1e3)
        torch.cuda.synchronize()
        t_s = time.perf_counter()
        o = est.networks(ev, int(r.t_ns))
        T, score = (None, 0.0) if o is None else est.pnp(o, kp_cache[r.object], intr.K, intr.D)
        rt = (time.perf_counter() - t_s) * 1e3
        if T is not None:
            w.add(r.trajectory, "events", r.t_ns, r.object, T, score, rt)
    w.meta["median_event_read_ms"] = float(np.median(io_ms)) if io_ms else None
    w.meta["n_samples"] = len(samples)
    p = w.save()
    print(f"{p}: {len(w)}/{len(samples)} poses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
