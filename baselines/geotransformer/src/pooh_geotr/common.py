"""GeoTransformer glue: upstream imports, config, cube pair data (cropped scan <-> CAD).

Frames and scaling. GeoTransformer's ModelNet model was trained on objects normalised to the
unit sphere, so both clouds are scaled by ``SCALE`` = 10 (the 10 cm cube becomes side 1):

* ``src`` = CAD body surface sample, model frame, centred on the cube centre, times SCALE.
  The marker holder is dropped, which makes the source exactly cube-symmetric.
* ``ref`` = cropped scan, lidar frame, centred on the crop centroid ``m``, times SCALE.
* ``transform`` (ref <- src) = [R_lm | SCALE (R_lm c + t_lm - m)], with c the cube centre.

The cube pose is ambiguous up to 24 rotations. Training therefore uses the symmetric variant
whose rotation is closest to the identity in the ref frame, which is a deterministic function
of the observed geometry and always within 63 deg of identity, near the <=45 deg rotations of
the ModelNet pre-training. The symmetric variants share the cube centre, so the translation
does not change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2]              # baselines/geotransformer
UPSTREAM = HERE / "third_party" / "GeoTransformer"
EXP = UPSTREAM / "experiments" / "geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn"
PRETRAINED = HERE / "weights" / "geotransformer-modelnet.pth.tar"
REPO = HERE.parents[1]
WORK = REPO / "work" / "geotransformer"
SCALE = 10.0
N_SRC = 717                                             # ModelNet config num_points
METHOD = "geotransformer"


def weights_for(setting: str) -> Path:
    return HERE / "weights" / f"pooh-{setting}.pth.tar"


def import_upstream():
    """Put GeoTransformer + its ModelNet experiment dir on sys.path; return (make_cfg, model, loss)."""
    import torch  # noqa: F401  (libc10 must be loaded before geotransformer.ext)

    for p in (str(UPSTREAM), str(EXP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import config as gt_config  # upstream experiment modules
    import loss as gt_loss
    import model as gt_model

    return gt_config.make_cfg, gt_model, gt_loss


def make_cfg(setting: str, iters: int = 2000, lr: float = 1e-4, out_dir: Path | None = None):
    make, _, _ = import_upstream()
    cfg = make()
    out = Path(out_dir or WORK / setting)
    cfg.output_dir = str(out)
    cfg.snapshot_dir = str(out / "snapshots")
    cfg.log_dir = str(out / "logs")
    cfg.event_dir = str(out / "events")
    for d in (cfg.snapshot_dir, cfg.log_dir, cfg.event_dir):
        os.makedirs(d, exist_ok=True)
    cfg.optim.max_iteration = iters
    cfg.optim.snapshot_steps = iters
    cfg.optim.warmup_steps = max(1, iters // 20)
    cfg.optim.lr = lr
    cfg.optim.eta_init = 0.1
    cfg.optim.eta_min = 0.1
    cfg.train.num_workers = 4
    cfg.test.num_workers = 2
    return cfg


# ----------------------------------------------------------------------------- geometry
def cube_syms() -> np.ndarray:
    from pooh_bench.symmetry import cube_rotations

    return cube_rotations()


def canonical_rotation(R: np.ndarray) -> np.ndarray:
    """R @ S_k closest to the identity (max trace)."""
    S = cube_syms()
    RS = R[None] @ S
    return RS[np.argmax(np.trace(RS, axis1=1, axis2=2))]


def src_cloud(obj, n: int = N_SRC, seed: int = 0) -> np.ndarray:
    from pooh_lidar.common import CUBE_CENTER, model_cloud

    return ((model_cloud(obj, n, body_only=True, seed=seed) - CUBE_CENTER) * SCALE).astype(np.float32)


def gt_transform(T_lm: np.ndarray, m: np.ndarray) -> np.ndarray:
    from pooh_lidar.common import CUBE_CENTER

    R = canonical_rotation(T_lm[:3, :3])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = SCALE * (T_lm[:3, :3] @ CUBE_CENTER + T_lm[:3, 3] - m)
    return T


def to_lidar_model(T_rs: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Estimated ref<-src transform -> T_lidar_model (metres, CAD frame)."""
    from pooh_lidar.common import CUBE_CENTER

    R = T_rs[:3, :3]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = T_rs[:3, 3] / SCALE + m - R @ CUBE_CENTER
    return T


def ref_cloud(pts_lidar: np.ndarray, max_points: int = 1024):
    m = pts_lidar.mean(0)
    q = (pts_lidar - m) * SCALE
    if len(q) > max_points:
        q = q[np.random.default_rng(0).choice(len(q), max_points, replace=False)]
    return q.astype(np.float32), m


# ----------------------------------------------------------------------------- training pairs
def build_pairs(domain: str, split: str, cache: Path | None = None) -> dict:
    """TRAINING ONLY (reads GT): cropped scan points + T_lidar_model for every GT sample.

    The crop is the same training-free crop as at test time. Where it misses the cube (centroid
    > 8 cm from the GT centre) it falls back to the GT-centred sphere, so training sees realistic
    crops but never wrong targets. The result is cached as an npz.
    """
    import pooh_bench
    from pooh_bench.sensors import lidar_points
    from pooh_lidar.common import CUBE_CENTER, load_samples_with_gt, transform
    from pooh_lidar.crop import TargetCropper, gravity_up, oracle_crop
    from tqdm import tqdm

    ds, res = pooh_bench.open_dataset(), pooh_bench.results_dir()
    droot = Path(os.environ.get("POOH_DATASET_ROOT") or "default").resolve()
    tag = f"{droot.parent.name}-{droot.name}_{res.resolve().parent.name}-{res.name}_{domain}_{split}"
    cache = cache or WORK / "pairs" / f"{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    gt = load_samples_with_gt(res, domain, split).sort_values(["trajectory", "t_ns"])
    cropper = TargetCropper()
    pts, Ts, oracle = [], [], 0
    for r in tqdm(gt.itertuples(), total=len(gt), desc=f"pairs {domain}/{split}"):
        tr = ds[r.trajectory]
        T = np.asarray(r.T).reshape(4, 4)
        p, _ = lidar_points(tr, int(r.index))
        c = cropper(p, key=r.trajectory, up=gravity_up(tr))
        cg = transform(T, CUBE_CENTER[None])[0]
        if c.center is None or np.linalg.norm(c.center - cg) > 0.08:
            c = oracle_crop(p, T, 0.10)
            oracle += 1
        if len(c.points) < 8:
            continue
        pts.append(c.points.astype(np.float32))
        Ts.append(T)
    out = {"points": np.array(pts, dtype=object), "T": np.array(Ts)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    print(f"{len(pts)} pairs ({oracle} with GT-sphere fallback crop) -> {cache}")
    return out


class CubePairDataset:
    """GeoTransformer pair dataset (same dict keys as upstream's ModelNetPairDataset)."""

    def __init__(self, pairs: dict, obj, augment: bool, noise: float = 0.005, seed: int = 0):
        self.points, self.T = pairs["points"], pairs["T"]
        self.obj, self.augment, self.noise = obj, augment, noise
        self.rng = np.random.default_rng(seed)
        self._src_fixed = src_cloud(obj)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        from scipy.spatial.transform import Rotation

        p = np.asarray(self.points[i], np.float64)
        T_lm = np.asarray(self.T[i], np.float64)
        src = self._src_fixed
        if self.augment:
            # rotate the scene about the crop centroid (full SO(3)) + jitter + point dropout
            m0 = p.mean(0)
            A = np.eye(4)
            A[:3, :3] = Rotation.random(random_state=self.rng.integers(1 << 31)).as_matrix()
            A[:3, 3] = m0 - A[:3, :3] @ m0
            p = p @ A[:3, :3].T + A[:3, 3] + self.rng.normal(0, self.noise, p.shape)
            T_lm = A @ T_lm
            keep = self.rng.random(len(p)) > self.rng.uniform(0, 0.3)
            if keep.sum() >= 8:
                p = p[keep]
            src = src_cloud(self.obj, seed=int(self.rng.integers(1 << 20)))
        ref, m = ref_cloud(p)
        return {
            "ref_points": ref, "src_points": src,
            "ref_feats": np.ones((len(ref), 1), np.float32), "src_feats": np.ones((len(src), 1), np.float32),
            "transform": gt_transform(T_lm, m).astype(np.float32),
            "index": i,
        }


def inference_dict(ref: np.ndarray, src: np.ndarray) -> dict:
    return {"ref_points": ref, "src_points": src,
            "ref_feats": np.ones((len(ref), 1), np.float32), "src_feats": np.ones((len(src), 1), np.float32),
            "transform": np.eye(4, dtype=np.float32)}  # dummy: only feeds the (unused) GT outputs


PATCH = 64  # upstream ModelNet uses 128 points per superpoint patch; our crops can be smaller


def run_model(model, batch):
    """Forward with the patch size clipped to the fine-level cloud sizes (no learned params
    depend on it; upstream's topk would fail on crops with fewer than 128 points)."""
    n = int(batch["lengths"][0].min())
    model.num_points_in_patch = max(1, min(PATCH, n))
    return model(batch)
