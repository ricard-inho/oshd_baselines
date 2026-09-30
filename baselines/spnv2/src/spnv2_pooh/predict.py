"""SPNv2 inference on a pooh-bench sample list -> results/predictions/<modality>/spnv2/<setting>/.

Pose (upstream's final-pose convention, ``engine/inference.py`` + docs/BUGFIX_20230725.md):
rotation from the heatmap head (keypoints -> PnP), translation from the EfficientPose
regression head. Where the heatmap pose is rejected (fewer than 5 keypoints over the
threshold, or PnP fails / lands outside the view frustum) the full regression pose is used.
``--pose-source heatmap|regression`` uses one head only.

PnP: OpenCV EPnP inside RANSAC, refined with Levenberg-Marquardt on the inliers (upstream
calls plain EPnP). Keypoint locations are heatmap argmaxes as upstream, taken at the cell
centre ((x + 0.5) * stride instead of x * stride).

``--odr``: upstream's Online Domain Refinement before predicting: entropy minimisation of the
segmentation head on the (unlabelled) evaluation images, updating only the backbone BN
layers (affine params by AdamW, running stats by the upstream manual accumulation). The
result goes to setting ``<setting>_ODR``, which the tables do not read.

``runtime_ms``: undistort/resize warp + normalisation + network + post-processing + PnP, on
a decoded frame (JPEG decoding excluded), with CUDA synchronisation.

This never reads the ``T`` column of the sample list.
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

from .common import (MODALITY, METHOD, OBJECT, VirtualCamera, Warper, ensure_upstream_on_path, net_to_model,
                     run_dir, to_network_image, write_camera_files)

log = logging.getLogger("spnv2_pooh.predict")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def read_samples(path: Path) -> pd.DataFrame:
    """The sample list without the ground-truth ``T`` column."""
    import pyarrow.parquet as pq

    cols = [c for c in pq.read_schema(path).names if c != "T"]
    return pd.read_parquet(path, columns=cols)


class Model:
    def __init__(self, ckpt_path: Path, device="cuda", odr_opts: list | None = None):
        import torch

        ensure_upstream_on_path()
        from yacs.config import CfgNode

        from nets import build_spnv2

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.ck = ck
        self.vcam = VirtualCamera(**ck["vcam"])
        self.c = np.asarray(ck["center"], float)
        self.kp = np.asarray(ck["keypoints_net"], float)
        self.gray = ck["gray"]
        cfg = CfgNode.load_cfg(ck["cfg"])
        # the EfficientPose head reads camera.json at build time: re-create it next to the checkpoint
        assets = Path(ckpt_path).parent / "assets"
        from pooh_bench import open_dataset

        write_camera_files(assets, self.vcam, self.kp, open_dataset().objects[OBJECT], self.c)
        cfg.defrost()
        cfg.DATASET.CAMERA = str(assets / "camera.json")
        cfg.DATASET.KEYPOINTS = str(assets / "cube_keypoints.mat")
        cfg.DATASET.CADMODEL = str(assets / "cube.ply")
        cfg.TEST.HEAD = ["heatmap", "efficientpose", "segmentation"]
        if odr_opts:
            cfg.merge_from_list(odr_opts)
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

        x = torch.from_numpy(img_net).to(self.device, non_blocking=True).permute(2, 0, 1)[None].float() / 255.0
        return (x - self.mean) / self.std

    def prepare(self, rgb: np.ndarray, intr) -> np.ndarray:
        return to_network_image(self.warp(rgb, intr), self.gray)

    # ------------------------------------------------------------ inference
    def infer(self, img_net: np.ndarray, pose_source: str = "fused"):
        import torch
        from utils.postprocess import rot_6d_to_matrix

        with torch.no_grad():
            heat, (cls, bbox, rot6d, trans), seg = self.net(self.to_tensor(img_net), is_train=False, gpu=self.device)
            j = int(torch.argmax(cls[0]))
            score = float(torch.sigmoid(cls[0, j]))
            R_reg = rot_6d_to_matrix(rot6d[0, j]).cpu().numpy().astype(float).T  # head regresses R_cam_net^T
            t_reg = trans[0, j].cpu().numpy().astype(float)
            box = bbox[0, j].cpu().numpy()
            hm = heat[0].float().cpu()
            mask = (seg[0, 0] > 0).cpu().numpy()
        kp2d, vis, R_h, t_h = self.heatmap_pose(hm)
        info = {"kp2d": kp2d, "vis": vis, "box": box, "mask": mask, "score": score, "heatmap_ok": R_h is not None}
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
        return net_to_model(T, self.c), info

    def heatmap_pose(self, hm):
        """Upstream solve_pose_from_heatmaps (argmax, adaptive threshold, >= 5 keypoints, frustum
        check) with EPnP-RANSAC + LM refinement."""
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
        obj, img = self.kp[vis].reshape(-1, 1, 3), kp2d[vis].reshape(-1, 1, 2)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP,
                                                 reprojectionError=2.0 * W / nW, iterationsCount=100,
                                                 confidence=0.999)
        if not ok or inl is None or len(inl) < 4:
            return kp2d, vis, None, None
        inl = inl.ravel()
        ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], K, None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
        t = tvec.ravel()
        # upstream's physical check: in front of the camera, within range and the field of view
        if not ok or t[2] <= 0 or t[2] > 10.0:
            return kp2d, vis, None, None
        if abs(t[0]) > t[2] * max(K[0, 2], W - K[0, 2]) / K[0, 0] or abs(t[1]) > t[2] * max(K[1, 2], H - K[1, 2]) / K[1, 1]:
            return kp2d, vis, None, None
        return kp2d, vis, cv2.Rodrigues(rvec)[0], t


# ---------------------------------------------------------------------- ODR
class _ImageSet:
    def __init__(self, rows, sensor, model: Model):
        self.rows, self.sensor = rows, sensor
        self.vcam, self.gray = model.vcam, model.gray
        self._ds = None
        self._warp = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch

        from pooh_bench import open_dataset

        if self._ds is None:
            self._ds, self._warp = open_dataset(), Warper(self.vcam)
        traj_name, index = self.rows[i]
        traj = self._ds[traj_name]
        img = to_network_image(self._warp(traj.frame(self.sensor, int(index)), traj.calibration.intrinsics(self.sensor)),
                               self.gray)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


def run_odr(model: Model, rows, sensor: str, n_samples: int, lr: float = 1e-4, batch: int = 4) -> None:
    """upstream tools/odr.py main_worker + engine/adapter.do_adapt, on the evaluation images."""
    import torch

    from engine.adapter import do_adapt
    from solver import get_optimizer

    cfg = model.cfg
    net = model.net
    net.eval()
    net.requires_grad_(False)
    bn_features, handles = {}, []

    def hook(name):
        def f(module, inp, out):
            bn_features[name] = inp[0].detach()
        return f

    for n, m in net.backbone.named_modules():
        if not cfg.MODEL.USE_GROUPNORM_BACKBONE and isinstance(m, torch.nn.BatchNorm2d):
            m.requires_grad_(True)
            m.eval()
            handles.append(m.register_forward_hook(hook(n)))
        elif cfg.MODEL.USE_GROUPNORM_BACKBONE and isinstance(m, torch.nn.GroupNorm):
            m.requires_grad_(True)
    # hooks are registered on backbone module names; do_adapt matches model.named_modules() names
    bn_features_prefixed = _PrefixDict(bn_features, "backbone.")
    loader = torch.utils.data.DataLoader(_ImageSet(rows, sensor, model), batch_size=1, shuffle=True,
                                         num_workers=4, drop_last=True)
    optimizer = get_optimizer(cfg, net)
    do_adapt(0, cfg, net, bn_features_prefixed, loader, optimizer, device=model.device, scaler=None)
    for h in handles:
        h.remove()
    net.requires_grad_(False)
    net.eval()


class _PrefixDict(dict):
    """View of the hook dict with ``backbone.``-prefixed keys: upstream hooks
    ``model.backbone.named_modules()`` but updates ``model.named_modules()`` by the same name,
    which only matches if the names agree; prefixing makes the running-stat update apply."""

    def __init__(self, base, prefix):
        super().__init__()
        self.base, self.prefix = base, prefix

    def items(self):
        return [(self.prefix + k, v) for k, v in self.base.items()]


# ---------------------------------------------------------------------- overlays
def overlay(img_net: np.ndarray, T_cam_model, info, model: Model, text: str) -> np.ndarray:
    from pooh_dataset.geometry import transform_points

    out = img_net.copy()
    m = cv2.resize(info["mask"].astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    out[..., 0] = np.maximum(out[..., 0], m * 120)  # predicted mask (blue, BGR)
    b = info["box"].astype(int)
    cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
    for k, (u, v) in enumerate(info["kp2d"]):
        col = (0, 0, 255) if info["vis"][k] else (0, 0, 90)
        cv2.circle(out, (int(u), int(v)), 3, col, -1)
        cv2.putText(out, str(k), (int(u) + 3, int(v) - 3), 0, 0.35, col, 1)
    if T_cam_model is not None:
        corners = transform_points(T_cam_model, model.kp + model.c)
        uv, _ = model.vcam.intrinsics().project(corners)
        for i, j in EDGES:
            cv2.line(out, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), (0, 255, 0), 1)
    cv2.putText(out, text, (8, 18), 0, 0.5, (0, 255, 255), 1)
    return out


# ---------------------------------------------------------------------- main
def predict(sensor: str, setting: str, domain: str = "real", split: str = "test", limit: int | None = None,
            checkpoint: Path | None = None, pose_source: str = "fused", odr: bool = False,
            odr_samples: int = 1024, overlays: int = 8, debug_dir: Path | None = None) -> Path | None:
    from pooh_bench import PoseWriter, gt_path, open_dataset, prediction_path, repo_root, results_dir

    res = results_dir()
    gp = gt_path(res, sensor, domain, split)
    if not gp.exists():
        print(f"no sample list {gp}: nothing to predict")
        return None
    samples = read_samples(gp)
    samples = samples[samples["object"] == OBJECT]
    if limit:
        samples = samples.iloc[:: max(1, len(samples) // limit)].iloc[:limit]  # spread over the split
    ckpt = Path(checkpoint or run_dir(sensor, setting) / "model.pth.tar")
    odr_opts = None
    if odr:
        odr_opts = ["ODR.MIN_ENTROPY", True, "ODR.NUM_TRAIN_SAMPLES", odr_samples, "ODR.IMAGES_PER_BATCH", 4,
                    "MODEL.HEAD.LOSS_HEADS", [], "MODEL.HEAD.LOSS_FACTORS", [], "TRAIN.IMAGES_PER_GPU", 1,
                    "TRAIN.LR", 1.0e-4, "TRAIN.WD", 0.0, "FP16", False]
    model = Model(ckpt, odr_opts=odr_opts)
    out_setting = f"{setting}_ODR" if odr else setting
    if odr:
        run_odr(model, list(zip(samples["trajectory"], samples["index"])), sensor, odr_samples)

    ds = open_dataset()
    path = prediction_path(res, MODALITY[sensor], METHOD, out_setting, domain, split)
    w = PoseWriter(path, {"method": METHOD, "setting": out_setting, "sensor": sensor, "checkpoint": str(ckpt),
                          "pose_source": pose_source, "odr": odr, "limit": limit,
                          "train_iterations": model.ck.get("iterations"), "upstream_commit": model.ck.get("upstream_commit"),
                          "runtime": "warp + network + post-processing + PnP on a decoded frame (JPEG decode excluded)"})
    debug_dir = Path(debug_dir or repo_root() / "work" / "debug" / "spnv2")
    debug_dir.mkdir(parents=True, exist_ok=True)
    show = set(np.linspace(0, len(samples) - 1, min(overlays, len(samples))).astype(int).tolist()) if overlays else set()
    import torch

    n_heat = 0
    for k, r in enumerate(tqdm(samples.itertuples(), total=len(samples), desc=f"spnv2 {sensor}/{out_setting}/{domain}_{split}")):
        traj = ds[r.trajectory]
        rgb = traj.frame(sensor, int(r.index))
        intr = traj.calibration.intrinsics(sensor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = model.prepare(rgb, intr)
        T, info = model.infer(img, pose_source)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        n_heat += info["heatmap_ok"]
        if T is not None:
            w.add(r.trajectory, sensor, r.t_ns, OBJECT, T, info["score"], ms)
        if k in show:
            txt = f"{r.trajectory}#{r.index} {out_setting} {'hm' if info['heatmap_ok'] else 'reg'} s={info['score']:.2f}"
            cv2.imwrite(str(debug_dir / f"{sensor}_{out_setting}_{domain}_{split}_{k:05d}.jpg"),
                        overlay(img, T, info, model, txt))
    w.meta["n_samples"] = len(samples)
    w.meta["heatmap_pose_rate"] = n_heat / max(len(samples), 1)
    p = w.save()
    print(f"{p}: {len(w)} poses for {len(samples)} samples (heatmap PnP accepted on {n_heat})")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", required=True, choices=["firefly", "realsense_color"])
    ap.add_argument("--setting", required=True, choices=["ZS", "FT"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", default="real", choices=["real", "synthetic"])
    ap.add_argument("--limit", type=int, default=None, help="evenly spaced subset of N samples")
    ap.add_argument("--checkpoint", default=None, help="default <work>/runs/<sensor>/<setting>/model.pth.tar")
    ap.add_argument("--pose-source", default="fused", choices=["fused", "heatmap", "regression"])
    ap.add_argument("--odr", action="store_true", help="online domain refinement first (writes <setting>_ODR)")
    ap.add_argument("--odr-samples", type=int, default=1024)
    ap.add_argument("--overlays", type=int, default=8, help="overlay images written to work/debug/spnv2/")
    ap.add_argument("--debug-dir", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    predict(a.sensor, a.setting, a.domain, a.split, a.limit, a.checkpoint, a.pose_source, a.odr, a.odr_samples,
            a.overlays, a.debug_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
