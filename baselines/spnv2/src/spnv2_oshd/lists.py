"""SPNv2 on the shared oshd-bench sample lists (``results/sim/<run>/``, ``results/real/<run>/``):
Table I (sim docking, satellite mockup) and Table II (real <-> lab twin).

    spnv2-list convert --exp sim                    # train list -> SPEED+-style folder (reads gt.parquet)
    spnv2-list train   --exp sim                    # one network per target of the experiment
    spnv2-list predict --exp sim --run-dir results/sim/sim_zerog_docking_test_v2   # samples.parquet only
    spnv2-list labels  --exp sim                    # overlay of converted training labels (check)
    spnv2-list predict --exp twin --run-dir results/real/real_realsense_color_test_v3 --adapt odr
                                                    # variant: unsupervised adaptation first (adapt_list)

Experiments, targets and virtual cameras are in ``configs/experiments.yaml``. Per target:

* **Keypoints:** ``num_keypoints`` farthest-point samples of the CAD surface (deterministic), in
  the ``net`` frame: the CAD frame shifted to the CAD bounding-box centre ``c``, which is where
  oshd-bench measures e_t and about which its symmetries act.
* **Symmetry:** the target's 180° flips measured by ``oshd-bench sim-eval`` (``symmetries`` in
  the yaml, checked against the CAD at convert time). Labels are canonicalised over that group as
  for the cube (``common.canonicalize``); with no symmetry this is the identity.
* **Scale:** sim targets are scaled per run (``scale`` of the sample list, also in
  ``samples.parquet``). A monocular network cannot see absolute size, so it is trained on the pose
  of the unit-scale CAD that gives the same image (``t / scale``) and the prediction's translation
  is multiplied by the sample's ``scale``. Real and twin targets have scale 1.
* **Frames:** every frame is undistorted and resampled into the experiment's virtual pinhole
  camera (same optical frame as the sensor), grayscale x 3 as upstream. Masks: the simulator's
  semantic class ``target`` (sim, twin) or the CAD rendered at the ground-truth pose (real).

A test sample goes to the network whose ``objects`` contain its ``object``; samples of other
targets get no prediction (score those lists with ``--subset``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .common import (
    BASELINE_DIR,
    UPSTREAM_COMMIT,
    R_to_q_wxyz,
    VirtualCamera,
    Warper,
    build_cfg,
    canonicalize,
    ensure_upstream_on_path,
    estimate_reference,
    to_network_image,
    work_dir,
    write_camera_files,
)

log = logging.getLogger("spnv2_oshd.lists")
CONFIG = BASELINE_DIR / "configs" / "experiments.yaml"
FLIPS = {"Rx180": np.diag([1.0, -1, -1]), "Ry180": np.diag([-1.0, 1, -1]), "Rz180": np.diag([-1.0, -1, 1])}
BOX_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


# ----------------------------------------------------------------------------- config


def load_experiments(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG).read_text())


def experiment(name: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_experiments()
    if name not in cfg["experiments"]:
        raise SystemExit(f"unknown experiment {name!r}: {sorted(cfg['experiments'])}")
    e = dict(cfg["experiments"][name])
    v = cfg["vcams"][e["vcam"]]
    e["vcam"] = VirtualCamera(e["vcam"], int(v["width"]), int(v["height"]), np.asarray(v["K"], float).tolist())
    e["targets"] = {t: cfg["targets"][t] for t in e["targets"]}
    e["num_keypoints"] = int(cfg["num_keypoints"])
    e["name"] = name
    return e


def repo_path(p: str | Path) -> Path:
    from oshd_bench import repo_root

    p = Path(p)
    return p if p.is_absolute() else repo_root() / p


def setting_of(exp: str) -> str:
    return exp.upper()


def data_dir(exp: str, target: str) -> Path:
    """``speedplus_<exp>_<target>/`` (upstream's DATANAME must contain 'speedplus')."""
    return work_dir() / "data" / f"speedplus_{exp}_{target}"


def model_dir(exp: str, target: str) -> Path:
    return work_dir() / "runs" / exp / target


def trajectories(exp_name: str, split: str, domain: str = "sim_zerog", task: str | None = None,
                 benchmark: str = "v3") -> list[str]:
    """Simulated trajectories of a benchmark split whose target is one of the experiment's
    ``objects`` (``models/sim/targets.json``), to download only those: ``oshd download -t ...``.
    In a fresh dataset folder it first fetches the small files ``oshd download`` starts with
    (manifests, splits, targets.json), at the benchmark's revisions."""
    from oshd.hub import META_IGNORE, META_PATTERNS, SIM_MANIFEST, SIM_TARGETS
    from oshd.splits import load_benchmark
    from oshd_bench import open_dataset

    ds = open_dataset()
    if not (ds.root / SIM_TARGETS).exists() or not (ds.root / SIM_MANIFEST).exists():
        from huggingface_hub import hf_hub_download, snapshot_download

        b = load_benchmark(benchmark)
        snapshot_download(b["repo_id"], repo_type="dataset", revision=b["revision"], local_dir=ds.root,
                          allow_patterns=META_PATTERNS, ignore_patterns=META_IGNORE)
        hf_hub_download(b["repo_id"], SIM_TARGETS, repo_type="dataset",
                        revision=b.get("models_revision", b["revision"]), local_dir=ds.root)
        ds = open_dataset()
    targets = json.loads((ds.root / SIM_TARGETS).read_text())
    objects = {o for t in experiment(exp_name)["targets"].values() for o in t["objects"]}
    return [t for t in ds.split(split, domain, task=task, benchmark=benchmark)
            if targets.get(t, {}).get("model") in objects]


# ----------------------------------------------------------------------------- target geometry


def load_cad(cad: str):
    """The target's CAD exactly as oshd-bench sim-eval loads it (trimesh, metres, unscaled)."""
    from oshd_bench import open_dataset
    from oshd_bench.sim import load_cad as _load

    return _load({"t": {"cad": cad}}, "t", open_dataset().root)


def farthest_points(pts: np.ndarray, k: int, start: np.ndarray) -> np.ndarray:
    """Greedy farthest-point sampling, starting from the point farthest from ``start``."""
    idx = [int(np.argmax(np.linalg.norm(pts - start, axis=1)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(pts - pts[idx[-1]], axis=1))
    return pts[idx]


def target_geometry(spec: dict, num_keypoints: int, check_symmetry: bool = True) -> dict:
    """centre c (CAD bbox centre), keypoints in the net frame, symmetry rotations about c, CAD
    vertices and bbox corners (model frame, unscaled)."""
    import trimesh

    mesh = load_cad(spec["cad"])
    v = np.asarray(mesh.vertices, float)
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2
    surf = np.asarray(trimesh.sample.sample_surface(mesh, 20000, seed=0)[0])
    kp = farthest_points(surf, num_keypoints, c) - c
    flips = list(spec.get("symmetries") or [])
    if check_symmetry:
        from oshd_bench.sim import measured_symmetries

        pts = np.asarray(trimesh.sample.sample_surface(mesh, 2000, seed=1)[0])
        meas = measured_symmetries(pts, c, float(np.linalg.norm(hi - lo)), 0.01)
        if sorted(meas) != sorted(flips):
            raise SystemExit(f"{spec['cad']}: configured symmetries {flips} != measured {meas}")
    rots = np.stack([np.eye(3)] + [FLIPS[f] for f in flips])
    corners = np.array([[(lo, hi)[k & 1][0], (lo, hi)[k >> 1 & 1][1], (lo, hi)[k >> 2 & 1][2]] for k in range(8)])
    return {"c": c, "kp": kp, "rots": rots, "symmetries": flips, "vertices": v, "corners": corners}


def trajectory(ds, name: str, keep: int = 2):
    """``ds[name]``, closing all but the ``keep`` most recent trajectories: OshDataset caches every
    Trajectory it opened, with its decoded row groups, so a pass over a few hundred runs grows to
    several GB per process."""
    t = ds[name]
    cache = ds._trajectories
    cache[name] = cache.pop(name)
    while len(cache) > keep:
        cache.pop(next(iter(cache)))
    return t


def unit_pose(T_flat, scale: float) -> np.ndarray:
    """Pose of the unit-scale CAD that projects like the CAD scaled by ``scale`` at ``T``."""
    T = np.asarray(T_flat, float).reshape(4, 4).copy()
    T[:3, 3] /= float(scale)
    return T


def project_box(vertices: np.ndarray, T: np.ndarray, vcam: VirtualCamera) -> np.ndarray | None:
    """Amodal box of the projected CAD vertices in front of the camera, clipped to the image."""
    p = vertices @ T[:3, :3].T + T[:3, 3]
    p = p[p[:, 2] > 1e-3]
    if not len(p):
        return None
    uv = p[:, :2] / p[:, 2:3] * np.diag(vcam.Km)[:2] + vcam.Km[:2, 2]
    box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    box = np.clip(box, 0, [vcam.width, vcam.height, vcam.width, vcam.height])
    return None if box[2] - box[0] < 2 or box[3] - box[1] < 2 else box


# ----------------------------------------------------------------------------- convert

_W: dict = {}


def _init(vcam_d, kp, c, rots, R_ref, verts, sensor, real, img_dir, mask_dir, gray):
    from oshd_bench import open_dataset

    ds = open_dataset()
    vcam = VirtualCamera(**vcam_d)
    _W.update(ds=ds, vcam=vcam, warp=Warper(vcam), kp=np.asarray(kp), c=np.asarray(c), rots=np.asarray(rots),
              R_ref=np.asarray(R_ref), verts=np.asarray(verts), sensor=sensor, real=real, img_dir=Path(img_dir),
              mask_dir=Path(mask_dir), gray=gray, vi=vcam.intrinsics(), vm=vcam.intrinsics(0.25))


def _one(row: tuple) -> list | None:
    traj_name, index, obj_name, scale, T_flat = row
    w = _W
    vcam, vi = w["vcam"], w["vi"]
    T_u = unit_pose(T_flat, scale)
    box = project_box(w["verts"], T_u, vcam)
    if box is None:
        return None
    T_cn = canonicalize(T_u, w["c"], w["R_ref"], w["rots"])
    pts = np.vstack([np.zeros(3), w["kp"]]) @ T_cn[:3, :3].T + T_cn[:3, 3]
    uv, _ = vi.project(pts)
    uv = uv / np.array([vcam.width, vcam.height])
    traj = trajectory(w["ds"], traj_name)
    intr = traj.calibration.intrinsics(w["sensor"])
    fname = f"{traj_name}_{int(index):06d}.jpg"
    img = w["warp"](traj.frame(w["sensor"], int(index)), intr)
    cv2.imwrite(str(w["img_dir"] / fname), to_network_image(img, w["gray"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    vm = w["vm"]
    if w["real"]:
        from oshd.labels import render_mask

        T_m = np.asarray(T_flat, float).reshape(4, 4)  # real targets: scale 1, CAD = ds.objects[obj]
        mask = render_mask(w["ds"].objects[obj_name], T_m, vm).astype(np.uint8) * 255
    else:
        sem = (traj.frame("semantic", int(index)) == traj.semantic_classes["target"]).astype(np.uint8) * 255
        full = w["warp"](sem, intr)
        mask = (cv2.resize(full, (vm.width, vm.height), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8) * 255
    cv2.imwrite(str(w["mask_dir"] / fname), mask, [cv2.IMWRITE_JPEG_QUALITY, 100])
    b = box / np.array([vcam.width, vcam.height, vcam.width, vcam.height])
    return [fname, *b.tolist(), *R_to_q_wxyz(T_cn[:3, :3]).tolist(), *T_cn[:3, 3].tolist(), *uv.reshape(-1).tolist()]


def convert(exp_name: str, target: str, stride: int | None = None, limit: int | None = None, workers: int = 8,
            overwrite: bool = False) -> Path:
    """Training folder of one target. Returns the CSV path relative to ``speedplus_<exp>_<target>/``."""
    e = experiment(exp_name)
    spec = e["targets"][target]
    lst = repo_path(e["train_list"])
    prep = json.loads((lst / "prepare.json").read_text())
    if prep["split"] != "train":
        raise SystemExit(f"{lst}: convert reads ground-truth poses, only train lists are allowed")
    gt = pd.read_parquet(lst / "gt.parquet")
    gt = gt[gt["object"].isin(spec["objects"])].iloc[::stride or int(e.get("stride", 1))]
    if limit:
        gt = gt.iloc[:limit]
    if not len(gt):
        raise SystemExit(f"no {target} samples in {lst}")
    vcam = e["vcam"]
    g = target_geometry(spec, e["num_keypoints"])
    root = data_dir(exp_name, target)
    root.mkdir(parents=True, exist_ok=True)
    (root / "vcam.json").write_text(json.dumps(vcam.to_dict(), indent=2))
    write_camera_files(root, vcam, g["kp"], g["vertices"], g["c"])

    canon_p = root / "canon.json"
    if canon_p.exists() and not overwrite:
        canon = json.loads(canon_p.read_text())
        if not np.allclose(canon["keypoints_net"], g["kp"]) or canon["symmetries"] != g["symmetries"]:
            raise SystemExit(f"{canon_p} is from other keypoints / symmetries: delete {root} or --overwrite")
        R_ref = np.asarray(canon["R_ref"])
    else:
        Ts = np.stack([unit_pose(t, s) for t, s in zip(gt["T"], gt["scale"])])
        Ts[:, :3, 3] += Ts[:, :3, :3] @ g["c"]  # model -> net origin
        R_ref = estimate_reference(Ts[:, :3, :3], Ts[:, :3, 3], g["rots"])
        canon = {"R_ref": R_ref.tolist(), "center": g["c"].tolist(), "keypoints_net": g["kp"].tolist(),
                 "symmetries": g["symmetries"], "cad": spec["cad"], "cad_corners": g["corners"].tolist(),
                 "objects": spec["objects"], "from": str(e["train_list"])}
        canon_p.write_text(json.dumps(canon, indent=2))

    sub = root / "train"
    csv = sub / "labels" / "labels.csv"
    meta_p = sub / "meta.json"
    key = {"train_list": str(e["train_list"]), "revision": prep.get("revision"), "n_gt": len(gt),
           "stride": stride or int(e.get("stride", 1)), "limit": limit, "gray": bool(e.get("gray", True)),
           "vcam": vcam.to_dict(), "R_ref": R_ref.tolist()}
    if csv.exists() and meta_p.exists() and not overwrite and json.loads(meta_p.read_text()).get("key") == key:
        print(f"{csv}: up to date")
        return csv.relative_to(root)
    img_dir = sub / f"images_{vcam.width}x{vcam.height}_RGB"
    mask_dir = sub / f"masks_{vcam.width // 4}x{vcam.height // 4}"
    for d in (img_dir, mask_dir, csv.parent):
        d.mkdir(parents=True, exist_ok=True)
    rows = list(zip(gt["trajectory"], gt["index"], gt["object"], gt["scale"], gt["T"]))
    init = (vcam.to_dict(), g["kp"].tolist(), g["c"].tolist(), g["rots"].tolist(), R_ref.tolist(),
            g["vertices"][np.random.default_rng(0).choice(len(g["vertices"]), min(len(g["vertices"]), 20000),
                                                          replace=False)].tolist(),
            prep["sensor"], prep["domain"] == "real", str(img_dir), str(mask_dir), bool(e.get("gray", True)))
    out = []
    with get_context("spawn").Pool(workers, _init, init) as pool:
        for r in tqdm(pool.imap(_one, rows, chunksize=16), total=len(rows), desc=f"convert {exp_name}/{target}"):
            if r is not None:
                out.append(r)
    with open(csv, "w") as f:
        for r in out:
            f.write(", ".join(str(x) for x in r) + "\n")
    meta_p.write_text(json.dumps({"key": key, "n_written": len(out)}, indent=2))
    print(f"{csv}: {len(out)} / {len(rows)} samples")
    return csv.relative_to(root)


# ----------------------------------------------------------------------------- train

DEFAULTS = (20, 1.0e-3, [15, 18])  # upstream's offline schedule: epochs, lr, lr steps


def clamp_sun_flare() -> None:
    """Upstream's RandomSunFlare puts its flare circles on a line through the flare centre at a
    random angle; within ~1e-7 rad of vertical, tan(angle) pushes a centre's y past int32 and
    cv2.circle raises (it stopped the sim run in epoch 19, after ~1M samples). A centre that far off
    the image draws nothing, so clamping it to +-1e6 px leaves the augmentation unchanged.
    Patched on the class: the DataLoader workers fork and inherit it."""
    ensure_upstream_on_path()
    from dataset.transforms.randomsunflare import RandomSunFlare

    if getattr(RandomSunFlare, "_oshd_clamped", False):
        return
    params = RandomSunFlare.get_params_dependent_on_targets

    def clamped(self, p):
        out = params(self, p)
        c = lambda v: int(min(max(v, -1_000_000), 1_000_000))
        out["circles"] = [(a, (c(x), c(y)), r, col) for a, (x, y), r, col in out["circles"]]
        return out

    RandomSunFlare.get_params_dependent_on_targets = clamped
    RandomSunFlare._oshd_clamped = True


def train(exp_name: str, target: str, epochs: int | None = None, max_iters: int | None = None,
          batch: int | None = None, lr: float | None = None, stride: int | None = None, limit: int | None = None,
          workers: int = 8, opts: list | None = None, resume: bool = False) -> Path:
    import torch

    from .train import _Capped, get_train_loader

    e = experiment(exp_name)
    out = model_dir(exp_name, target)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "train.log")], force=True)
    csv_rel = convert(exp_name, target, stride, limit, workers)
    root = data_dir(exp_name, target)
    vcam = VirtualCamera(**json.loads((root / "vcam.json").read_text()))
    canon = json.loads((root / "canon.json").read_text())
    with open(root / csv_rel) as f:
        n_csv = sum(1 for _ in f)

    ep_def, lr_def, steps = DEFAULTS
    ep = epochs or int(e.get("epochs", ep_def))
    steps = [s for s in e.get("lr_steps", steps) if s < ep] or [ep]
    o = ["TRAIN.TRAIN_CSV", str(csv_rel), "TRAIN.VAL_CSV", str(csv_rel), "TRAIN.END_EPOCH", ep,
         "TRAIN.LR", lr or float(e.get("lr", lr_def)), "TRAIN.LR_STEP", steps,
         "DATASET.NUM_KEYPOINTS", e["num_keypoints"]]
    if batch:
        o += ["TRAIN.IMAGES_PER_GPU", batch]
    cfg = build_cfg(f"{exp_name}_{target}", vcam, root.parent, opts=o + list(opts or []))
    (out / "config.yaml").write_text(cfg.dump())

    from engine.trainer import do_train
    from nets import build_spnv2
    from solver import adjust_learning_rate, get_optimizer, get_scaler
    from utils.utils import set_seeds_cudnn

    set_seeds_cudnn(cfg, seed=cfg.SEED)
    clamp_sun_flare()
    model = build_spnv2(cfg).to(torch.device("cuda"))  # ImageNet EfficientNet-B3 backbone
    loader = get_train_loader(cfg)
    optimizer = get_optimizer(cfg, model)
    scaler = get_scaler(cfg)
    n_per_epoch = len(loader)
    if max_iters:
        ep = min(ep, math.ceil(max_iters / n_per_epoch))
    log.info(f"{exp_name}/{target}: {len(loader.dataset)} samples, {n_per_epoch} it/epoch, {ep} epochs, "
             f"lr {cfg.TRAIN.LR} steps {list(cfg.TRAIN.LR_STEP)}, batch {cfg.TRAIN.IMAGES_PER_GPU}, "
             f"max_iters={max_iters}")
    start, done, wall0 = 0, 0, 0
    if resume:  # continue after the last saved epoch, same schedule
        ck = torch.load(out / "model.pth.tar", map_location="cuda", weights_only=False)
        if (ck["train_list"], ck["n_train"], ck["epochs"], ck["keypoints_net"]) != \
                (str(e["train_list"]), n_csv, ep, canon["keypoints_net"]):
            raise SystemExit(f"{out / 'model.pth.tar'} is from another training (list, size, epochs or keypoints)")
        model.load_state_dict(ck["state_dict"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
            if scaler is not None and ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
        else:
            log.warning("checkpoint has no optimizer state: AdamW moments restart from zero")
        start, done, wall0 = ck["epoch"], ck["iterations"], ck["wall_s"]
        log.info(f"resumed after epoch {start}/{ep} ({done} iterations)")
    t0 = time.time() - wall0
    for epoch in range(start, ep):
        adjust_learning_rate(optimizer, epoch, cfg)
        n = n_per_epoch if not max_iters else min(n_per_epoch, max_iters - done)
        if n <= 0:
            break
        te = time.time()
        do_train(epoch, cfg, model, _Capped(loader, n), optimizer, device=torch.device("cuda"), scaler=scaler)
        done += n
        log.info(f"epoch {epoch + 1}/{ep}: {n} it in {time.time() - te:.0f} s "
                 f"({n / max(time.time() - te, 1e-9):.2f} it/s)")
        torch.save({
            "state_dict": model.state_dict(), "cfg": cfg.dump(), "vcam": vcam.to_dict(), "gray": bool(e.get("gray", True)),
            "R_ref": canon["R_ref"], "center": canon["center"], "keypoints_net": canon["keypoints_net"],
            "symmetries": canon["symmetries"], "cad": canon["cad"], "cad_corners": canon["cad_corners"],
            "objects": canon["objects"], "experiment": exp_name, "target": target, "train_list": str(e["train_list"]),
            "n_train": n_csv, "epoch": epoch + 1, "epochs": ep, "iterations": done, "upstream_commit": UPSTREAM_COMMIT,
            "wall_s": round(time.time() - t0), "resumed_after": start if resume else None,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler is not None else None,
        }, out / "model.pth.tar")
    log.info(f"done: {done} iterations in {time.time() - t0:.0f} s -> {out / 'model.pth.tar'}")
    return out / "model.pth.tar"


# ----------------------------------------------------------------------------- predict


def read_samples(run: Path) -> pd.DataFrame:
    """``samples.parquet`` of a sample list (it has no ``T``; the column filter is a guard)."""
    import pyarrow.parquet as pq

    p = run / "samples.parquet"
    return pd.read_parquet(p, columns=[c for c in pq.read_schema(p).names if c not in ("T", "range_m")])


def draw(img_net: np.ndarray, T_unit, info, corners, vcam: VirtualCamera, text: str) -> np.ndarray:
    """Prediction overlay on the network image: CAD box (green), heatmap keypoints (red, dark =
    below threshold), predicted mask (blue), detected box (cyan)."""
    out = img_net.copy()
    m = cv2.resize(info["mask"].astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    out[..., 0] = np.maximum(out[..., 0], m * 120)
    b = info["box"].astype(int)
    cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
    for k, (u, v) in enumerate(info["kp2d"]):
        col = (0, 0, 255) if info["vis"][k] else (0, 0, 90)
        cv2.circle(out, (int(u), int(v)), 3, col, -1)
    if T_unit is not None:
        p = corners @ T_unit[:3, :3].T + T_unit[:3, 3]
        if np.all(p[:, 2] > 1e-3):
            uv, _ = vcam.intrinsics().project(p)
            for i, j in BOX_EDGES:
                cv2.line(out, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), (0, 255, 0), 1)
    cv2.putText(out, text, (8, 18), 0, 0.5, (0, 255, 255), 1)
    return out


def adapt_rows(adapt_list: Path, objects, n: int) -> tuple[list, str]:
    """``n`` evenly spaced (trajectory, index) of ``objects`` from the ``samples.parquet`` of an
    unlabelled target-domain list (never a test list), and its sensor."""
    sensor = json.loads((adapt_list / "prepare.json").read_text())["sensor"]
    s = read_samples(adapt_list)
    s = s[s["object"].isin(objects)]
    if not len(s):
        raise SystemExit(f"{adapt_list}: no sample of {sorted(objects)} to adapt on")
    s = s.iloc[np.unique(np.linspace(0, len(s) - 1, min(n, len(s))).round().astype(int))]
    return list(zip(s["trajectory"], s["index"])), sensor


def predict(exp_name: str, run_dir: Path, setting: str | None = None, out_dir: Path | None = None,
            stride: int = 1, limit: int | None = None, pose_source: str = "fused", overlays: int = 12,
            adapt: str | None = None, adapt_list: Path | None = None, adapt_samples: int = 1024) -> Path:
    """``adapt``: unsupervised domain adaptation of each network before predicting, on
    ``adapt_samples`` unlabelled images of ``adapt_list`` (default: the experiment's
    ``adapt_list``, a target-domain *train* list): ``adabn`` (backbone BatchNorm statistics
    re-estimated) or ``odr`` (upstream's online domain refinement, its settings). Writes setting
    ``<SETTING>_<ADAPT>``; without ``adapt`` nothing changes."""
    import torch
    from oshd_bench import PoseWriter, open_dataset

    from .predict import Model, odr_options, run_adabn, run_odr

    e = experiment(exp_name)
    run = repo_path(run_dir)
    prep = json.loads((run / "prepare.json").read_text())
    sensor = prep["sensor"]
    samples = read_samples(run)
    samples = samples.iloc[::stride]  # the same subset sim-eval takes from the "stride" / "limit" meta
    if limit:
        samples = samples.iloc[:limit]
    route = {o: t for t, spec in e["targets"].items() for o in spec["objects"]}
    todo = samples[samples["object"].isin(route)]
    if not len(todo):
        raise SystemExit(f"{run}: no sample of {sorted(route)}")
    setting = setting or setting_of(exp_name)
    if adapt:
        if adapt not in ("adabn", "odr"):
            raise SystemExit(f"--adapt {adapt}: adabn or odr")
        if not (adapt_list or e.get("adapt_list")):
            raise SystemExit(f"{exp_name}: no adapt_list in the yaml; pass --adapt-list")
        adapt_list = repo_path(adapt_list or e["adapt_list"])
        if adapt_list.resolve() == run.resolve():
            raise SystemExit(f"{adapt_list}: adapting on the evaluated list itself")
        setting = f"{setting}_{adapt.upper()}"
    out_dir = Path(out_dir) if out_dir else BASELINE_DIR / "results" / run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    models, ckpts = {}, {}
    for t, spec in e["targets"].items():
        if not todo["object"].isin(spec["objects"]).any():
            continue
        ck = model_dir(exp_name, t) / "model.pth.tar"
        if not ck.exists():
            raise SystemExit(f"no checkpoint {ck}: spnv2-list train --exp {exp_name} --target {t}")
        m = Model(ck, vertices=np.asarray(torch.load(ck, map_location="cpu", weights_only=False)["cad_corners"]),
                  odr_opts=odr_options(adapt_samples) if adapt == "odr" else None)
        models[t] = m
        ckpts[t] = {"checkpoint": str(ck), "epoch": m.ck["epoch"], "iterations": m.ck["iterations"],
                    "n_train": m.ck["n_train"], "train_list": m.ck["train_list"], "symmetries": m.ck["symmetries"],
                    "n_keypoints": len(m.kp)}
        if adapt:
            rows, a_sensor = adapt_rows(adapt_list, spec["objects"], adapt_samples)
            torch.manual_seed(0)
            t0 = time.time()
            if adapt == "adabn":
                run_adabn(m, rows, a_sensor)
            else:
                run_odr(m, rows, a_sensor, adapt_samples)
            ckpts[t]["adapt"] = {"method": adapt, "list": adapt_list.name,
                                 "sensor": a_sensor, "n_images": len(rows),
                                 "n_trajectories": len({r[0] for r in rows}), "wall_s": round(time.time() - t0)}
            log.info("%s/%s: %s on %d images of %s in %.0f s", exp_name, t, adapt, len(rows), adapt_list.name,
                     time.time() - t0)
    vc = next(iter(models.values())).vcam
    meta = {"method": "SPNv2", "upstream": "https://github.com/tpark94/spnv2", "commit": UPSTREAM_COMMIT,
            "setting": setting, "experiment": exp_name, "checkpoints": ckpts, "pose_source": pose_source,
            "input": f"{prep['domain']} {sensor}, undistorted + resampled to a {vc.width}x{vc.height} pinhole "
                     f"virtual camera, grayscale x3",
            "vcam": vc.to_dict(), "scale": "translation x the sample's scale (trained on t / scale)",
            "runtime": "warp + network + post-processing + PnP on a decoded frame (JPEG decode excluded)",
            "prepare": prep, "stride": stride, "limit": limit, "adapt": adapt}
    w = PoseWriter(out_dir / f"predictions_{setting}.parquet", meta)
    dbg = out_dir / f"overlays_{setting}"
    show = set(np.linspace(0, len(todo) - 1, min(overlays, len(todo))).astype(int).tolist()) if overlays else set()
    if show:
        dbg.mkdir(exist_ok=True)
    ds = open_dataset()
    n_heat, n_none, t_start = 0, 0, time.time()
    for k, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc=f"spnv2 {exp_name} -> {run.name}")):
        m = models[route[r.object]]
        traj = trajectory(ds, r.trajectory)
        rgb = traj.frame(sensor, int(r.index))
        intr = traj.calibration.intrinsics(sensor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = m.prepare(rgb, intr)
        T_u, info = m.infer(img, pose_source)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        n_heat += info["heatmap_ok"]
        if T_u is None:
            n_none += 1
        else:
            T = T_u.copy()
            T[:3, 3] *= float(r.scale)
            w.add(r.trajectory, sensor, r.t_ns, r.object, T, info["score"], ms)
        if k in show:
            txt = f"{r.trajectory}#{r.index} {setting} {'hm' if info['heatmap_ok'] else 'reg'} s={info['score']:.2f}"
            cv2.imwrite(str(dbg / f"{r.trajectory}_{int(r.index):06d}.jpg"),
                        draw(img, T_u, info, np.asarray(m.ck["cad_corners"]), m.vcam, txt))
    w.meta.update({"n_samples": len(samples), "n_attempted": len(todo), "n_predicted": len(w),
                   "n_no_pose": n_none, "heatmap_pose_rate": n_heat / len(todo), "wall_s": round(time.time() - t_start)})
    p = w.save()
    print(f"{p}: {len(w)} poses for {len(todo)} samples of {sorted(route)} ({len(samples)} in the list); "
          f"heatmap PnP accepted on {n_heat}")
    return p


# ----------------------------------------------------------------------------- label check


def labels(exp_name: str, target: str, n: int = 12, out: Path | None = None) -> Path:
    """Overlay the converted training labels (keypoints, origin, box, mask, CAD box at the label
    pose) on ``n`` evenly spaced training images, to check the conventions end to end."""
    from scipy.spatial.transform import Rotation

    root = data_dir(exp_name, target)
    vcam = VirtualCamera(**json.loads((root / "vcam.json").read_text()))
    canon = json.loads((root / "canon.json").read_text())
    c, corners = np.asarray(canon["center"]), np.asarray(canon["cad_corners"])
    csv = pd.read_csv(root / "train" / "labels" / "labels.csv", header=None)
    tiles = []
    for i in np.linspace(0, len(csv) - 1, n).astype(int):
        r = csv.iloc[i]
        img = cv2.imread(str(root / "train" / f"images_{vcam.width}x{vcam.height}_RGB" / r[0].strip()))
        mask = cv2.imread(str(root / "train" / f"masks_{vcam.width // 4}x{vcam.height // 4}" / r[0].strip()), 0)
        img[..., 0] = np.maximum(img[..., 0], cv2.resize(mask, (vcam.width, vcam.height)) // 2)
        b = (np.asarray(r[1:5], float) * np.array([vcam.width, vcam.height] * 2)).astype(int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
        q = np.asarray(r[5:9], float)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        t = np.asarray(r[9:12], float)
        p = (corners - c) @ R.T + t
        if np.all(p[:, 2] > 1e-3):
            uv, _ = vcam.intrinsics().project(p)
            for a, bb in BOX_EDGES:
                cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[bb].astype(int)), (0, 255, 0), 1)
        kp = np.asarray(r[12:], float).reshape(-1, 2) * [vcam.width, vcam.height]
        for k, (u, v) in enumerate(kp):
            cv2.circle(img, (int(u), int(v)), 4 if k == 0 else 3, (0, 255, 255) if k == 0 else (0, 0, 255), -1)
        # keypoints re-projected from the label pose must coincide with the stored ones
        uv2, _ = vcam.intrinsics().project(np.asarray(canon["keypoints_net"]) @ R.T + t)
        for u, v in uv2:
            cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), 1)
        tiles.append(cv2.resize(img, (vcam.width // 2, vcam.height // 2)))
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    from oshd_bench import repo_root

    out = Path(out or repo_root() / "work" / "debug" / "spnv2" / f"labels_{exp_name}_{target}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(out)
    return out


# ----------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spnv2-list", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("convert", "train", "labels"):
        s = sub.add_parser(name)
        s.add_argument("--exp", required=True)
        s.add_argument("--target", action="append", default=None, help="default: every target of the experiment")
        if name != "labels":
            s.add_argument("--stride", type=int, default=None, help="every n-th training sample (default: yaml)")
            s.add_argument("--limit", type=int, default=None, help="first n training samples (smoke tests)")
            s.add_argument("--workers", type=int, default=8, help="conversion processes")
        if name == "convert":
            s.add_argument("--overwrite", action="store_true")
        if name == "train":
            s.add_argument("--epochs", type=int, default=None)
            s.add_argument("--max-iters", type=int, default=None)
            s.add_argument("--batch", type=int, default=None)
            s.add_argument("--lr", type=float, default=None)
            s.add_argument("--resume", action="store_true", help="continue after the last saved epoch")
            s.add_argument("opts", nargs=argparse.REMAINDER, help="extra yacs KEY VALUE overrides")
    s = sub.add_parser("trajectories", help="sim trajectories of a split with one of the experiment's targets")
    s.add_argument("--exp", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--domain", default="sim_zerog")
    s.add_argument("--task", default=None)
    s.add_argument("--benchmark", default="v3")
    p = sub.add_parser("predict")
    p.add_argument("--exp", required=True)
    p.add_argument("--run-dir", required=True, help="a shared sample list, e.g. results/sim/twin_test_v3")
    p.add_argument("--setting", default=None, help="output name (default: the experiment in capitals)")
    p.add_argument("--out-dir", default=None, help="default: baselines/spnv2/results/<run dir name>/")
    p.add_argument("--stride", type=int, default=1, help="every n-th sample of the list")
    p.add_argument("--limit", type=int, default=None, help="first n samples of the list (smoke tests)")
    p.add_argument("--pose-source", default="fused", choices=["fused", "heatmap", "regression"])
    p.add_argument("--overlays", type=int, default=12)
    p.add_argument("--adapt", default=None, choices=["adabn", "odr"],
                   help="unsupervised adaptation on unlabelled target-domain train images first")
    p.add_argument("--adapt-list", default=None, help="default: the experiment's adapt_list")
    p.add_argument("--adapt-samples", type=int, default=1024)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if a.cmd == "predict":
        predict(a.exp, Path(a.run_dir), a.setting, a.out_dir, a.stride, a.limit, a.pose_source, a.overlays,
                a.adapt, a.adapt_list and Path(a.adapt_list), a.adapt_samples)
        return 0
    if a.cmd == "trajectories":
        print("\n".join(trajectories(a.exp, a.split, a.domain, a.task, a.benchmark)))
        return 0
    for t in a.target or list(experiment(a.exp)["targets"]):
        if a.cmd == "convert":
            convert(a.exp, t, a.stride, a.limit, a.workers, a.overwrite)
        elif a.cmd == "train":
            train(a.exp, t, a.epochs, a.max_iters, a.batch, a.lr, a.stride, a.limit, a.workers, a.opts, a.resume)
        else:
            labels(a.exp, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
