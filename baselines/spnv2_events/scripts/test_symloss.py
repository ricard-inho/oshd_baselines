"""CPU checks of the conventions behind ``spnv2_events.symloss`` on 3 real converted labels.

For each label (canonical ``T_cn`` from the CSV quaternion / translation, scale from the train
list) and each symmetry ``F_k``:

* rotation: upstream's dataset target is ``quat2dcm(q) = R_cn^T``, and the target of the flipped
  label ``R_cn F_k`` is ``F_k @ target``;
* (a) heatmap: the upstream ``HeatmapGenerator`` target of the flipped label (keypoints of
  ``T_cn @ F_k`` projected as ``convert.sample_label`` does) equals ``y[:, perm_k]`` and no other
  permutation (labels with all 8 corners in the image). The 180 deg flips' permutations are
  involutions, so ``y[:, perm_k]`` and its inverse coincide here;
* (b) ``SymSPEEDLoss`` gives ~0 rotation error for a prediction equal to the flipped target,
  upstream's ``SPEEDLoss`` ~pi for k != 0;
* (c) with the identity only, both sym losses equal upstream's.

    uv run --project baselines/spnv2_events python baselines/spnv2_events/scripts/test_symloss.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation
from spnv2_events.common import (
    CORNER_SIGNS,
    R_to_q_wxyz,
    TargetFrame,
    VirtualCamera,
    ensure_upstream_on_path,
    repo_root,
)
from spnv2_events.convert import sensor_folder
from spnv2_events.symloss import SymHeatmapLoss, SymSPEEDLoss, corner_perms

RUN = "sim_zerog_docking_train_v3"
FAIL = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAIL.append(name)


def project(T, kp, vi, wh):
    """Normalised keypoints of pose T, as convert.sample_label + _one (behind the camera -> -1)."""
    from oshd.geometry import transform_points

    uv, ok = vi.project(transform_points(T, kp))
    uv[~ok] = -1.0
    return uv / wh


def r6d(R: torch.Tensor) -> torch.Tensor:
    """6D input whose rot_6d_to_matrix is R (its first two columns)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], -1)


def main() -> int:
    ensure_upstream_on_path()
    from dataset.target_generators import HeatmapGenerator
    from nets.loss import LossHeatmapMSE, SPEEDLoss
    from utils.postprocess import quat2dcm

    sf = sensor_folder()
    vcam = VirtualCamera(**json.loads((sf / "vcam.json").read_text()))
    tf = TargetFrame.from_dict(json.loads((sf / "target.json").read_text()))
    vi, wh = vcam.intrinsics(), np.array([vcam.width, vcam.height], float)
    csv = pd.read_csv(sf / RUN / "labels" / "labels.csv", header=None)
    gt = pd.read_parquet(repo_root() / "results" / "sim" / RUN / "gt.parquet")
    scale = {f"{t}_{int(i):06d}.png": s for t, i, s in zip(gt["trajectory"], gt["index"], gt["scale"])}
    gen = HeatmapGenerator([vcam.width // 4, vcam.height // 4], 8, 2)
    perms = corner_perms(tf.rots, CORNER_SIGNS)
    print(f"symmetries {['identity'] + tf.symmetries}, corner perms {perms.tolist()}")
    hm_sym, pose_sym = SymHeatmapLoss(perms), SymSPEEDLoss(tf.rots)
    hm_up, pose_up = LossHeatmapMSE(), SPEEDLoss()

    kps = csv.iloc[:, 14:].to_numpy(float)
    full = csv[((kps >= 0) & (kps < 1)).all(1)]  # all 8 corners in the image: every heatmap channel is tested
    for row in full.iloc[np.linspace(0, len(full) - 1, 3).astype(int)].itertuples(index=False):
        row = list(row)
        name, s = row[0], scale[row[0]]
        q = np.array(row[5:9], float)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # R_cam_net (scipy xyzw)
        T[:3, 3] = row[9:12]
        kp_csv = np.array(row[14:], float).reshape(8, 2)
        kp = tf.keypoints(s)
        report(f"{name} csv keypoints = projection of the csv pose",
               np.allclose(project(T, kp, vi, wh), kp_csv, atol=1e-6))
        R_tgt = quat2dcm(q.astype(np.float32))       # what the dataset puts in targets['rotationmatrix']
        report(f"{name} rotationmatrix target = R_cam_net^T", np.allclose(R_tgt, T[:3, :3].T, atol=1e-5))
        y = gen(kp_csv.astype(np.float32))
        for k, F in enumerate(tf.rots):
            tag = f"{name} k={k}"
            Tk = T.copy()
            Tk[:3, :3] = T[:3, :3] @ F
            Rk = quat2dcm(R_to_q_wxyz(Tk[:3, :3]).astype(np.float32))
            report(f"{tag} flipped rotation target = F_k @ target", np.allclose(Rk, F @ R_tgt, atol=1e-5))
            yk = gen(project(Tk, kp, vi, wh).astype(np.float32))
            other = [j for j in range(len(perms)) if j != k]
            report(f"{tag} (a) flipped heatmap = y[:, perm_k], != y[:, perm_j] for j != k",
                   np.allclose(yk, y[perms[k]], atol=1e-6)
                   and not any(np.allclose(yk, y[perms[j]], atol=1e-6) for j in other),
                   f"({int((y.reshape(8, -1).max(1) > 0).sum())}/8 keypoints in the heatmap)")
            Rg = torch.from_numpy(R_tgt)[None]
            t = torch.tensor(np.asarray(row[9:12], np.float32))[None, None]
            st = torch.ones(1, 1, dtype=torch.long)
            rp = r6d(torch.from_numpy(Rk))[None, None]
            l_sym = float(pose_sym(rp, t, Rg, t[:, 0], st))
            l_up = float(pose_up(rp, t, Rg, t[:, 0], st))
            report(f"{tag} (b) SymSPEEDLoss ~0, SPEEDLoss {'~pi' if k else '~0'}",
                   l_sym < 3e-3 and (abs(l_up - np.pi) < 3e-3 if k else l_up < 3e-3),
                   f"(sym {l_sym:.4f}, upstream {l_up:.4f})")
            yt = torch.from_numpy(y)[None]
            l_hm = float(hm_sym(torch.from_numpy(yk)[None], yt))
            report(f"{tag} (a) SymHeatmapLoss of the flipped heatmap ~0", l_hm < 1e-10, f"({l_hm:.2e})")

    # (c) identity only == upstream, on random tensors
    g = torch.Generator().manual_seed(0)
    x, y = torch.rand(4, 8, 128, 160, generator=g), torch.rand(4, 8, 128, 160, generator=g)
    a, b = SymHeatmapLoss(perms[:1])(x, y), hm_up(x, y)
    report("(c) SymHeatmapLoss(identity) == LossHeatmapMSE", torch.equal(a, b) or torch.allclose(a, b, rtol=1e-6),
           f"({float(a):.6f} vs {float(b):.6f}, bit-equal {torch.equal(a, b)})")
    B, N = 4, 50
    r = torch.randn(B, N, 6, generator=g)
    tp, tg = torch.randn(B, N, 3, generator=g), torch.randn(B, 3, generator=g) + 3.0
    Rg = torch.from_numpy(Rotation.random(B, random_state=0).as_matrix().astype(np.float32))
    st = (torch.rand(B, N, generator=g) > 0.5).long()
    a, b = SymSPEEDLoss(tf.rots[:1])(r, tp, Rg, tg, st), pose_up(r, tp, Rg, tg, st)
    report("(c) SymSPEEDLoss(identity) == SPEEDLoss", torch.allclose(a, b, rtol=1e-6, atol=1e-7),
           f"({float(a):.6f} vs {float(b):.6f})")
    a = SymSPEEDLoss(tf.rots)(r, tp, Rg, tg, st)
    report("SymSPEEDLoss(all flips) <= SPEEDLoss", bool(a <= b + 1e-7), f"({float(a):.6f})")
    print(f"{'FAILED: ' + str(len(FAIL)) + ' check(s)' if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
