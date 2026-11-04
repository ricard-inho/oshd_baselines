"""SPNv2 inference on event time surfaces of a oshd-bench sim sample list (satellite mockup only)
-> ``results/<run>/predictions_<setting>.parquet`` of this baseline.

Input per sample: the events of ``[t_ns - 50 ms, t_ns)`` as a 3-channel time surface
(``common.time_surface_image``, tau 25 ms), warped x2 + padded to the 640x512 virtual camera.

Pose (upstream's final-pose convention, ``engine/inference.py`` + docs/BUGFIX_20230725.md):
rotation from the heatmap head (keypoints -> PnP), translation from the EfficientPose
regression head. Where the heatmap pose is rejected (fewer than 5 keypoints over the
threshold, or PnP fails / lands outside the view frustum) the full regression pose is used.
``--pose-source heatmap|regression`` uses one head only.

PnP: OpenCV EPnP inside RANSAC, refined with Levenberg-Marquardt on the inliers (upstream
calls plain EPnP), against the 8 bounding-box corners of the CAD scaled by the sample's
``scale`` (a column of ``samples.parquet``, not a label). Keypoint locations are heatmap
argmaxes as upstream, taken at the cell centre ((x + 0.5) * stride instead of x * stride).
The network pose is in the ``net`` frame (bounding-box centre); the written pose is
``T_cam_target`` in the CAD root frame (``common.net_to_target``).

``runtime_ms``: time-surface build + warp + normalisation + network + post-processing + PnP,
with CUDA synchronisation. Reading the events of the window from disk is timed separately
(``event_read_ms_median`` in the metadata) and not included.

This reads ``samples.parquet`` only, never ``gt.parquet`` or a ``T`` column.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .common import (
    EDGES,
    METHOD,
    OBJECT,
    SENSOR,
    SETTING,
    TAU_NS,
    TEST_RUN,
    WINDOW_NS,
    TargetFrame,
    VirtualCamera,
    Warper,
    ensure_upstream_on_path,
    net_to_target,
    open_dataset,
    repo_root,
    results_dir,
    run_dir,
    time_surface_image,
    window_events,
    write_camera_files,
)

log = logging.getLogger("spnv2_events.predict")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_samples(path: Path) -> pd.DataFrame:
    """The sample list without any ground-truth ``T`` column."""
    import pyarrow.parquet as pq

    cols = [c for c in pq.read_schema(path).names if c != "T"]
    return pd.read_parquet(path, columns=cols)


class Model:
    def __init__(self, ckpt_path: Path, device="cuda"):
        import torch

        ensure_upstream_on_path()
        from nets import build_spnv2
        from yacs.config import CfgNode

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ck = ck
        self.vcam = VirtualCamera(**ck["vcam"])
        self.vi = self.vcam.intrinsics()
        self.tf = TargetFrame.from_dict(ck["target"])
        cfg = CfgNode.load_cfg(ck["cfg"])
        # the EfficientPose head reads camera.json at build time: re-create it next to the checkpoint
        assets = Path(ckpt_path).parent / "assets"
        write_camera_files(assets, self.vcam, self.tf)
        cfg.defrost()
        cfg.DATASET.CAMERA = str(assets / "camera.json")
        cfg.DATASET.KEYPOINTS = str(assets / "keypoints.mat")
        cfg.DATASET.CADMODEL = str(assets / "model.ply")
        cfg.TEST.HEAD = ["heatmap", "efficientpose", "segmentation"]
        cfg.freeze()
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = build_spnv2(cfg)
        self.net.load_state_dict(ck["state_dict"], strict=True)
        self.net.to(self.device).eval()
        self.warp = Warper(self.vcam)
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)

    # ------------------------------------------------------------ preprocessing
    def to_tensor(self, img_net: np.ndarray):
        import torch

        x = torch.from_numpy(img_net).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1)[None].float() / 255.0
        return (x - self.mean) / self.std

    def prepare(self, ev, t_ns: int, intr) -> np.ndarray:
        """Time surface of the window's events in the sensor image, warped to the virtual camera:
        (H, W, 3) uint8, channel order as in the training PNGs."""
        return self.warp(time_surface_image(ev, int(t_ns), intr.height, intr.width), intr)

    # ------------------------------------------------------------ inference
    def infer(self, img_net: np.ndarray, scale: float, pose_source: str = "fused"):
        """(T_cam_target or None, info); ``info['T_cam_net']`` is the pose in the net frame."""
        import torch
        from utils.postprocess import rot_6d_to_matrix

        with torch.no_grad():
            heat, (cls, bbox, rot6d, trans), seg = self.net(self.to_tensor(img_net), is_train=False,
                                                            gpu=self.device)
            j = int(torch.argmax(cls[0]))
            score = float(torch.sigmoid(cls[0, j]))
            R_reg = rot_6d_to_matrix(rot6d[0, j]).cpu().numpy().astype(float).T  # head regresses R_cam_net^T
            t_reg = trans[0, j].cpu().numpy().astype(float)
            box = bbox[0, j].cpu().numpy()
            hm = heat[0].float().cpu()
            mask = (seg[0, 0] > 0).cpu().numpy()
        kp2d, vis, R_h, t_h = self.heatmap_pose(hm, scale)
        info = {"kp2d": kp2d, "vis": vis, "box": box, "mask": mask, "score": score,
                "heatmap_ok": R_h is not None, "T_cam_net": None}
        if pose_source == "regression" or (R_h is None and pose_source == "fused"):
            R, t = R_reg, t_reg
        elif pose_source == "heatmap":
            if R_h is None:
                return None, info
            R, t = R_h, t_h
        else:  # fused: upstream's final pose (heatmap t if the regressed t is behind the camera)
            R, t = R_h, (t_reg if t_reg[2] > 0 else t_h)
        if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))) or t[2] <= 0:
            return None, info
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        info["T_cam_net"] = T
        return net_to_target(T, self.tf, scale), info

    def heatmap_pose(self, hm, scale: float):
        """Upstream solve_pose_from_heatmaps (argmax, adaptive threshold, >= 5 keypoints, frustum
        check) with EPnP-RANSAC + LM refinement, against the corners at the sample's scale."""
        import torch

        nK, nH, nW = hm.shape
        W, H = self.vcam.width, self.vcam.height
        maxv, maxi = hm.view(nK, -1).max(-1)
        thr, vis = float(self.cfg.TEST.HEATMAP_THRESHOLD), torch.zeros(nK, dtype=torch.bool)
        while int(vis.sum()) < 5 and thr >= 0:
            vis = maxv > thr
            thr -= 0.1
        y = torch.div(maxi, nW, rounding_mode="floor").numpy()
        x = (maxi - nW * torch.div(maxi, nW, rounding_mode="floor")).numpy()
        kp2d = np.stack([(x + 0.5) / nW * W, (y + 0.5) / nH * H], -1).astype(np.float64)
        vis = vis.numpy()
        if vis.sum() < 5:
            return kp2d, vis, None, None
        K = self.vcam.Km
        obj, img = self.tf.keypoints(scale)[vis].reshape(-1, 1, 3), kp2d[vis].reshape(-1, 1, 2)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP,
                                                 reprojectionError=2.0 * W / nW, iterationsCount=100,
                                                 confidence=0.999)
        if not ok or inl is None or len(inl) < 4:
            return kp2d, vis, None, None
        inl = inl.ravel()
        ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], K, None, rvec, tvec, True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        t = tvec.ravel()
        # upstream's physical check: in front of the camera, within range and the field of view
        if not ok or t[2] <= 0 or t[2] > 10.0:
            return kp2d, vis, None, None
        if (abs(t[0]) > t[2] * max(K[0, 2], W - K[0, 2]) / K[0, 0]
                or abs(t[1]) > t[2] * max(K[1, 2], H - K[1, 2]) / K[1, 1]):
            return kp2d, vis, None, None
        return kp2d, vis, cv2.Rodrigues(rvec)[0], t


# ---------------------------------------------------------------------- overlays
def overlay(img_net: np.ndarray, info, model: Model, scale: float, text: str) -> np.ndarray:
    """The network input (time surface as BGR: negative / positive / count) with the predicted
    mask (blue), box (cyan), heatmap keypoints (red, dark red below threshold) and the predicted
    pose as the scaled CAD's bounding-box wireframe (green)."""
    from oshd.geometry import transform_points

    out = np.ascontiguousarray(img_net.copy())
    m = cv2.resize(info["mask"].astype(np.uint8), (out.shape[1], out.shape[0]),
                   interpolation=cv2.INTER_NEAREST)
    out[..., 0] = np.maximum(out[..., 0], m * 120)  # predicted mask (blue, BGR)
    b = info["box"].astype(int)
    cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
    for k, (u, v) in enumerate(info["kp2d"]):
        col = (0, 0, 255) if info["vis"][k] else (0, 0, 90)
        cv2.circle(out, (int(u), int(v)), 3, col, -1)
        cv2.putText(out, str(k), (int(u) + 3, int(v) - 3), 0, 0.35, col, 1)
    if info["T_cam_net"] is not None:
        uv, ok = model.vi.project(transform_points(info["T_cam_net"], model.tf.keypoints(scale)))
        uv = np.clip(uv, -1e4, 1e4).astype(int)
        for i, j in EDGES:
            if ok[i] and ok[j]:
                cv2.line(out, tuple(uv[i].tolist()), tuple(uv[j].tolist()), (0, 255, 0), 1)
    cv2.putText(out, text, (8, 18), 0, 0.5, (0, 255, 255), 1)
    return out


# ---------------------------------------------------------------------- main
def resolve(path) -> Path:
    """``path`` as given, or relative to the repo root if it does not exist from the cwd."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else repo_root() / p


def default_checkpoint(setting: str = SETTING) -> Path:
    """``<work>/runs/<setting>/model_best.pth.tar`` (best validation epoch) if it exists, else
    ``model.pth.tar`` (last epoch)."""
    best = run_dir(setting) / "model_best.pth.tar"
    return best if best.exists() else run_dir(setting) / "model.pth.tar"


def predict(run: Path, setting: str = SETTING, checkpoint: Path | None = None, limit: int | None = None,
            stride: int = 1, pose_source: str = "fused", overlays: int = 12, overlay_dir: Path | None = None,
            out: Path | None = None) -> Path:
    import torch
    from oshd_bench import PoseWriter

    run = resolve(run)
    samples = read_samples(run / "samples.parquet")
    samples = samples[samples["object"] == OBJECT].iloc[::stride]
    if limit:
        samples = samples.iloc[:: max(1, len(samples) // limit)].iloc[:limit]  # spread over the split
    if not len(samples):
        raise SystemExit(f"no {OBJECT} samples in {run}")
    ckpt = Path(checkpoint or default_checkpoint(setting))
    model = Model(ckpt)

    ds = open_dataset()
    path = Path(out or results_dir(run.name) / f"predictions_{setting}.parquet")
    w = PoseWriter(path, {
        "method": METHOD, "setting": setting, "sensor": SENSOR, "object": OBJECT,
        "input": f"sim events, time surface of the {WINDOW_NS / 1e6:.0f} ms window ending at t_ns "
                 f"(tau {TAU_NS / 1e6:.0f} ms; channels neg / pos / log count), "
                 f"warped to {model.vcam.width}x{model.vcam.height}",
        "upstream_commit": model.ck.get("upstream_commit"), "checkpoint": str(ckpt),
        "checkpoint_epoch": model.ck.get("epoch"), "val": model.ck.get("val"),
        "train_run": model.ck.get("train_run"), "train_iterations": model.ck.get("iterations"),
        "pose_source": pose_source, "sample_limit": limit, "sample_stride": stride, "sample_list": str(run),
        "runtime": "time-surface build + warp + normalisation + network + post-processing + PnP "
                   "(event read from disk excluded, see event_read_ms_median)",
    })
    overlay_dir = Path(overlay_dir or results_dir(run.name) / f"overlays_{setting}")
    show = set(np.linspace(0, len(samples) - 1, min(overlays, len(samples))).astype(int).tolist()) \
        if overlays else set()
    if show:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    n_heat, ms_all, read_ms = 0, [], []
    desc = f"spnv2 events/{setting}/{run.name}"
    for k, r in enumerate(tqdm(samples.itertuples(), total=len(samples), desc=desc)):
        traj = ds[r.trajectory]
        intr = traj.calibration.intrinsics("rgb")
        t_read = time.perf_counter()
        ev = window_events(traj, r.t_ns)
        read_ms.append((time.perf_counter() - t_read) * 1000.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = model.prepare(ev, r.t_ns, intr)
        T, info = model.infer(img, float(r.scale), pose_source)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        ms_all.append(ms)
        n_heat += info["heatmap_ok"]
        if T is not None:
            w.add(r.trajectory, SENSOR, r.t_ns, OBJECT, T, info["score"], ms)
        if k in show:
            txt = (f"{r.trajectory}#{r.index} {setting} {'hm' if info['heatmap_ok'] else 'reg'} "
                   f"s={info['score']:.2f}")
            cv2.imwrite(str(overlay_dir / f"{setting}_{k:05d}_{r.trajectory}_{int(r.index):06d}.jpg"),
                        overlay(img, info, model, float(r.scale), txt), [cv2.IMWRITE_JPEG_QUALITY, 90])
    w.meta["n_samples"] = len(samples)
    w.meta["n_poses"] = len(w)
    w.meta["heatmap_pose_rate"] = n_heat / max(len(samples), 1)
    w.meta["runtime_ms_median"] = float(np.median(ms_all))
    w.meta["event_read_ms_median"] = float(np.median(read_ms))
    p = w.save()
    print(f"{p}: {len(w)} poses for {len(samples)} samples (heatmap PnP accepted on {n_heat}); "
          f"median runtime {np.median(ms_all):.1f} ms, median event read {np.median(read_ms):.1f} ms")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=f"results/sim/{TEST_RUN}", help="oshd-bench sim sample list")
    ap.add_argument("--setting", default=SETTING, choices=[SETTING])
    ap.add_argument("--checkpoint", default=None, help="default <work>/runs/<setting>/model_best.pth.tar, else model.pth.tar")
    ap.add_argument("--limit", type=int, default=None, help="evenly spaced subset of N samples")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th sample of the list")
    ap.add_argument("--pose-source", default="fused", choices=["fused", "heatmap", "regression"])
    ap.add_argument("--overlays", type=int, default=12, help="number of overlay images (JPEG)")
    ap.add_argument("--overlay-dir", default=None,
                    help="default results/<run>/overlays_<setting> of this baseline")
    ap.add_argument("--out", default=None,
                    help="default results/<run>/predictions_<setting>.parquet of this baseline")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    predict(Path(a.run_dir), a.setting, a.checkpoint, a.limit, a.stride, a.pose_source, a.overlays,
            a.overlay_dir, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
