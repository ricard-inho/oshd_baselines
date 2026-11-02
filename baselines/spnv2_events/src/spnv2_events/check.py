"""Label sanity checks on a sim *train* list (reads its ground truth; refuses test lists).

For ``--n`` evenly spaced ``satellite_mockup`` samples:

* the time surface of the window, warped to the virtual camera, with the label drawn on it
  (the 8 corner keypoints numbered, the box, the net origin, the bounding-box wireframe); on the
  warped ``rgb`` frame too if the trajectory has it, so a misalignment is visible (JPEG, ``--out``);
* events in the window, pixels with an event, mask pixels at 1/4 resolution (the mask label);
* PnP check: EPnP + LM from the labelled 2D corners and the scaled 3D corners must recover the
  labelled ``T_cam_net`` (expected < 0.5 deg / 1 cm; exit code 1 if any sample exceeds 2 deg /
  2 cm). This checks the keypoint, quaternion and frame conventions end to end.

Also prints which symmetry equivalent the canonicalisation picks over 2000 evenly spaced samples
of the list (identity or a flip): a spread-out histogram means the reference sits off-centre.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import (
    EDGES,
    VirtualCamera,
    Warper,
    canonical_index,
    load_target,
    open_dataset,
    repo_root,
    target_to_net,
    time_surface_image,
    window_events,
)
from .convert import read_train_list, sample_label, sensor_folder

TRAIN_RUN = "results/sim/sim_zerog_docking_train_v3"
TOL_WARN = (0.5, 0.01)   # deg, m
TOL_FAIL = (2.0, 0.02)


def resolve(path) -> Path:
    """``path`` as given, or relative to the repo root if it does not exist from the cwd."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else repo_root() / p


def draw_label(img: np.ndarray, box, origin, uv, ok, text: str) -> np.ndarray:
    out = np.ascontiguousarray(img.copy())
    b = np.asarray(box).astype(int)
    cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
    p = np.clip(uv, -1e4, 1e4).astype(int)
    for i, j in EDGES:
        if ok[i] and ok[j]:
            cv2.line(out, tuple(p[i].tolist()), tuple(p[j].tolist()), (0, 255, 0), 1)
    for k, (u, v) in enumerate(p):
        cv2.circle(out, (int(u), int(v)), 3, (0, 0, 255), -1)
        cv2.putText(out, str(k), (int(u) + 3, int(v) - 3), 0, 0.4, (0, 0, 255), 1)
    o = np.clip(origin, -1e4, 1e4).astype(int)
    cv2.drawMarker(out, (int(o[0]), int(o[1])), (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
    cv2.putText(out, text, (8, 18), 0, 0.5, (0, 255, 255), 1)
    return out


def pnp_error(uv: np.ndarray, kp3d: np.ndarray, K: np.ndarray, T_cn: np.ndarray) -> tuple[float, float]:
    """(rotation error deg, translation error m) of EPnP + LM from the labelled corners."""
    obj, img = kp3d.reshape(-1, 1, 3).astype(np.float64), uv.reshape(-1, 1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return float("inf"), float("inf")
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return float("inf"), float("inf")
    R = cv2.Rodrigues(rvec)[0]
    c = np.clip((np.trace(R.T @ T_cn[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c))), float(np.linalg.norm(tvec.ravel() - T_cn[:3, 3]))


def check(run: Path, n: int = 8, out: Path | None = None) -> int:

    run = resolve(run)
    prep = json.loads((run / "prepare.json").read_text())
    if prep["split"] != "train":
        raise SystemExit(f"check reads ground-truth poses: only train lists are allowed ({run})")
    out = Path(out or repo_root() / "work" / "debug" / "spnv2_events")
    out.mkdir(parents=True, exist_ok=True)

    gt = read_train_list(run, 1, n)
    ds = open_dataset()
    traj0 = gt["trajectory"].iloc[0]
    tf = load_target(ds, traj0)
    sf = sensor_folder()
    vp, cp = sf / "vcam.json", sf / "canon.json"
    if vp.exists():
        vcam = VirtualCamera(**json.loads(vp.read_text()))
    else:
        vcam = VirtualCamera.from_intrinsics("rgb", ds[traj0].calibration.intrinsics("rgb"))
    R_ref = np.asarray(json.loads(cp.read_text())["R_ref"]) if cp.exists() else np.eye(3)
    print(f"{run.name}: vcam {vcam.width}x{vcam.height} ({'vcam.json' if vp.exists() else 'new'}), "
          f"R_ref from {'canon.json' if cp.exists() else 'identity'}, "
          f"half extents {np.round(tf.half, 4).tolist()} m")
    vi = vcam.intrinsics()
    warp = Warper(vcam)
    warp_mask = Warper(vcam.scaled(0.25), cv2.INTER_NEAREST)

    worst, n_fail = (0.0, 0.0), 0
    for k, r in enumerate(gt.itertuples()):
        traj = ds[r.trajectory]
        intr = traj.calibration.intrinsics("rgb")
        T = np.asarray(r.T, float).reshape(4, 4)
        tag = f"{r.trajectory}#{int(r.index)}"
        lab = sample_label(tf, R_ref, T, float(r.scale), vi)
        if lab is None:
            print(f"{tag}: no label (target not in the virtual image)")
            continue
        T_cn, box, origin, uv = lab
        kp3d = tf.keypoints(r.scale)
        ok = (uv >= 0).all(1)  # corners behind the camera are labelled (-1, -1)
        ev = window_events(traj, r.t_ns)
        n_pix = len(np.unique(ev.y.astype(np.int64) * intr.width + ev.x.astype(np.int64))) if len(ev) else 0
        ts = warp(time_surface_image(ev, int(r.t_ns), intr.height, intr.width), intr)
        n_mask = -1
        if traj.has("semantic"):
            sem = traj.frame("semantic", int(r.index))
            mask = (sem == traj.semantic_classes["target"]).astype(np.uint8) * 255
            n_mask = int((warp_mask(mask, intr) > 128).sum())
        e_r, e_t = pnp_error(uv[ok], kp3d[ok], vi.K, T_cn) if ok.sum() >= 4 else (float("nan"), float("nan"))
        if not np.isnan(e_r):
            worst = (max(worst[0], e_r), max(worst[1], e_t))
        flag = "skip (< 4 corners in view)" if np.isnan(e_r) else (
            "FAIL" if e_r > TOL_FAIL[0] or e_t > TOL_FAIL[1] else (
                "warn" if e_r > TOL_WARN[0] or e_t > TOL_WARN[1] else "ok"))
        n_fail += flag == "FAIL"
        print(f"{tag}: scale {r.scale:.3f}, range {np.linalg.norm(T_cn[:3, 3]):.3f} m, {len(ev)} events on "
              f"{n_pix} px, mask {n_mask} px (1/4 res), corners in view {int(ok.sum())}/8, "
              f"PnP {e_r:.3f} deg {100 * e_t:.3f} cm [{flag}]")
        txt = f"{tag} s={r.scale:.2f} label"
        name = f"{k:02d}_{r.trajectory}_{int(r.index):06d}"
        cv2.imwrite(str(out / f"{name}_ts.jpg"), draw_label(ts, box, origin, uv, ok, txt),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if traj.has("rgb"):
            rgb = cv2.cvtColor(traj.frame("rgb", int(r.index)), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / f"{name}_rgb.jpg"), draw_label(warp(rgb, intr), box, origin, uv, ok, txt),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

    # canonicalisation: which symmetric equivalent is the label
    names = ["identity"] + list(tf.symmetries)
    sub = read_train_list(run, 1, 2000)
    hist = Counter()
    for T_flat, s in zip(sub["T"], sub["scale"], strict=True):
        T_cn = target_to_net(np.asarray(T_flat, float).reshape(4, 4), tf, float(s))
        hist[names[canonical_index(T_cn[:3, :3], T_cn[:3, 3], R_ref, tf.rots)]] += 1
    print(f"canonical equivalent over {len(sub)} samples: "
          + ", ".join(f"{nm} {hist[nm]} ({100 * hist[nm] / len(sub):.0f}%)" for nm in names))
    print(f"PnP worst: {worst[0]:.3f} deg, {100 * worst[1]:.3f} cm; overlays in {out}")
    if n_fail:
        print(f"FAILED: {n_fail} sample(s) above {TOL_FAIL[0]} deg / {100 * TOL_FAIL[1]:.0f} cm")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=TRAIN_RUN,
                    help=f"oshd-bench sim-prepare train list (default {TRAIN_RUN})")
    ap.add_argument("--n", type=int, default=8, help="number of evenly spaced samples to check")
    ap.add_argument("--out", default=None, help="overlay folder (default <repo>/work/debug/spnv2_events)")
    a = ap.parse_args(argv)
    return check(Path(a.run_dir), a.n, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
