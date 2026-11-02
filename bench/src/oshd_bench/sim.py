"""Shared sample lists, ground truth and evaluation for the simulated (SRB) splits.

Every baseline that runs on a sim split (FoundationPose, SPNv2 on RGB or events, LiDAR ICP, ...)
reads the **same** sample list and is scored by the **same** evaluator, so the rows of a paper
table are comparable.

    oshd-bench sim-prepare --domain sim_zerog --task docking --split test     # once, committed
    oshd-bench sim-eval --run-dir results/sim/sim_zerog_docking_test_v2 \\
        --pred baselines/<name>/results/sim_zerog_docking_test_v2/predictions_<SETTING>.parquet

``sim-prepare`` writes into ``results/sim/<domain>_<task>_<split>_<benchmark>/``:

* ``samples.parquet``: trajectory, index (frame index of ``--sensor``), t_ns, object (target
  model), scale, mask_px, mount. This is what inference reads.
* ``gt.parquet``: the same rows + ``T`` (``T_cam_target``: 16 floats row-major, in the optical
  frame of the sim ``rgb`` camera, target root frame, metres) and ``range_m``. Only training code
  (train split) and ``sim-eval`` may read it.
* ``objects.json``: target model -> CAD (``cad``: path relative to the dataset root,
  ``models/sim/<model>/model.obj``, metres, **unscaled**: multiply by the sample's ``scale``),
  extent, diameter.
* ``prepare.json``: the settings and the Hub revision.

Samples: every ``--stride``-th frame of ``--sensor`` (rgb, 20 Hz; the default stride 10 gives
2 Hz) where the target's semantic mask has at least ``--min-mask-px`` pixels. ``--sensor lidar``
(10 Hz scans; use ``--stride 5`` for 2 Hz): scans with at least ``--min-points`` returns on the
target, labelled by the simulator (a return is the target's if it lies on the target's surface);
``samples.parquet`` then has ``target_points`` (the indices of those returns in the scan: the
oracle segmentation, like the semantic mask for cameras) and ``mask_px`` is their number. ``T``
is then ``T_lidar_target`` (lidar sensor frame). The sim event
camera is pixel-aligned with ``rgb``, so event methods use the same samples and poses, with the
events of a window ending at ``t_ns``.

``sim-eval`` per sample: ``e_t`` (distance between the predicted and true CAD bounding-box
centres, m), ``e_q`` (geodesic rotation error, deg), ADD and ADD-S (mean / closest model-point
distance, m) and the SPEED score ``e_t/‖t‖ + e_q[rad]`` (SPEED+ thresholds). ``e_q``, ADD and
SPEED are given twice: strict, and ``_sym``, minimised over the CAD's measured symmetries (the
180° rotations about its bounding-box axes that map the surface onto itself to within
``--sym-tol`` of the diameter). The paper reports the ``_sym`` means. Samples without a
prediction count as failures in the accuracies and AUCs (``coverage`` says how many there are);
the error means / medians are over the predicted samples. Outputs go next to the predictions
file: ``per_sample_<stem>.parquet``, ``metrics_<stem>.json`` / ``.csv`` (overall and per target
kind, mount and range bin) and ``RESULTS_<stem>.md``, with ``<stem>`` the part of the
predictions file name after ``predictions_`` (or ``--name``, e.g. for a ``--subset``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

RANGE_BINS = [0.0, 0.3, 0.5, 1.0, 2.0, np.inf]
FLIPS = {"Rx180": np.diag([1.0, -1, -1]), "Ry180": np.diag([-1.0, 1, -1]), "Rz180": np.diag([-1.0, -1, 1])}


def run_name(domain: str, task: str | None, split: str, benchmark: str | None) -> str:
    return "_".join(x for x in (domain, task, split, benchmark) if x)


# ----------------------------------------------------------------------------- CAD


def cad_path(objects: dict, name: str, root: str | Path | None = None) -> Path:
    """Absolute CAD path of a target model of ``objects.json`` (``root``: the dataset root,
    default: oshd's default cache)."""
    p = Path(objects[name]["cad"])
    if p.is_absolute():
        return p
    if root is None:
        from oshd.hub import default_root

        root = default_root()
    return Path(root) / p


def load_cad(objects: dict, name: str, root: str | Path | None = None):
    """The target's CAD as a ``trimesh.Trimesh`` (metres, unscaled)."""
    import trimesh

    return trimesh.load(cad_path(objects, name, root), force="mesh")


def _cad_entry(traj) -> dict:
    import trimesh

    path = traj.target_model_path
    if path is None or not path.exists():
        raise SystemExit(f"{traj.name}: no target CAD in the dataset; rerun `oshd download` for it")
    info = json.loads((path.parent / "info.json").read_text())
    m = trimesh.load(path, force="mesh", process=False)
    ext = np.ptp(m.vertices, axis=0)
    return {"cad": str(path.relative_to(traj.root)),
            "source": info.get("source"), "kind": traj.target["kind"],
            "extent_m": ext.round(4).tolist(), "diameter_m": float(np.linalg.norm(ext)),  # bbox diagonal, unscaled
            "n_vertices": len(m.vertices), "n_faces": len(m.faces)}


# ----------------------------------------------------------------------------- prepare


def prepare(out: Path, *, domain: str = "sim_zerog", task: str | None = "docking", split: str = "test",
            benchmark: str | None = "v2", root: str | None = None, sensor: str = "rgb", stride: int = 10,
            min_mask_px: int = 100, min_points: int = 30, limit_trajectories: int | None = None) -> Path:
    from oshd import OshDataset
    from oshd.labels import camera_pose
    from tqdm import tqdm

    warnings.filterwarnings("ignore", message=".*real trajectories have no official split.*")
    ds = OshDataset(root, benchmark=benchmark)
    names = [t for t in ds.split(split, domain, task=task) if ds[t].has(sensor)]
    if limit_trajectories:
        names = names[:limit_trajectories]
    if not names:
        raise SystemExit(f"no downloaded trajectory for {domain}/{task}/{split}: run `oshd download "
                         f"--benchmark {benchmark} --domain {domain} --task {task} --split {split} "
                         f"-m {sensor} -m semantic -m mocap`")
    out.mkdir(parents=True, exist_ok=True)
    objects: dict[str, dict] = {}
    rows = []
    for t in tqdm(names, desc="samples"):
        traj = ds[t]
        if traj.target is None:
            raise SystemExit(f"{t}: no entry in models/sim/targets.json; rerun `oshd download` for it")
        name, scale = traj.target["model"], float(traj.target["scale"])
        if name not in objects:
            objects[name] = _cad_entry(traj)
        ts = traj.timestamps(sensor)
        if sensor == "lidar":
            from scipy.spatial import cKDTree

            surface = np.asarray(traj.target_mesh().sample(20000))
        else:
            target_id = traj.semantic_classes["target"]
        for i in range(0, len(ts), stride):
            t_ns = int(ts[i])
            T_wc, ok_c = camera_pose(traj, sensor, t_ns)
            T_wt, ok_t = traj.body_pose("target", t_ns)
            if not (ok_c and ok_t):
                continue
            T = np.linalg.inv(T_wc) @ T_wt
            extra = {}
            if sensor == "lidar":
                # Simulator labels of the returns: a point is the target's if it lies on its surface.
                scan = traj.lidar_scan(i)
                P = np.stack([scan["x"], scan["y"], scan["z"]], 1)
                tol = max(0.01, 0.01 * float(np.ptp(surface, 0).max()))
                on = cKDTree(surface @ T[:3, :3].T + T[:3, 3]).query(P)[0] < tol if len(P) else np.zeros(0, bool)
                n_px = int(on.sum())
                extra = {"target_points": np.flatnonzero(on).astype(np.int32).tolist()}
                if n_px < min_points:
                    continue
            else:
                n_px = int((traj.frame("semantic", i) == target_id).sum())
                if n_px < min_mask_px:
                    continue
            rows.append({"trajectory": t, "index": i, "t_ns": t_ns, "object": name, "scale": scale,
                         "mask_px": n_px, "T": T.reshape(-1).tolist(), "range_m": float(np.linalg.norm(T[:3, 3])),
                         "mount": traj.metadata["task_info"].get("target_mount"), **extra})
    gt = pd.DataFrame(rows)
    gt.to_parquet(out / "gt.parquet", index=False)
    gt.drop(columns=["T", "range_m"]).to_parquet(out / "samples.parquet", index=False)
    (out / "objects.json").write_text(json.dumps(objects, indent=2))
    from oshd.splits import load_benchmark

    revision = load_benchmark(benchmark)["revision"] if benchmark else None
    (out / "prepare.json").write_text(json.dumps({
        "benchmark": benchmark, "revision": revision,
        "domain": domain, "task": task, "split": split, "sensor": sensor, "stride": stride,
        "min_mask_px": min_mask_px, **({"min_points": min_points} if sensor == "lidar" else {}),
        "n_trajectories": len(names), "n_samples": len(gt),
        "objects": gt["object"].value_counts().to_dict()}, indent=2))
    print(f"{len(gt)} samples from {len(names)} trajectories, {len(objects)} target models -> {out}")
    return out


# ----------------------------------------------------------------------------- evaluate


def measured_symmetries(pts: np.ndarray, center: np.ndarray, diameter: float, tol: float) -> list[str]:
    """180° flips about the bounding-box axes that leave the sampled surface in place (ADD-S)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    return [n for n, R in FLIPS.items() if tree.query((pts - center) @ R.T + center)[0].mean() < tol * diameter]


def errors(gt: pd.DataFrame, pred: pd.DataFrame, objects: dict, root=None, n_points: int = 2000,
           sym_tol: float = 0.01) -> pd.DataFrame:
    import trimesh
    from scipy.spatial import cKDTree

    from .metrics import pose_errors, speed_score
    from .symmetry import Symmetry

    cols = ["trajectory", "t_ns", "T"] + [c for c in ("score", "runtime_ms") if c in pred]
    df = gt.merge(pred[cols].rename(columns={"T": "T_pred"}), on=["trajectory", "t_ns"], how="left")
    for c in ("score", "runtime_ms"):
        if c not in df:
            df[c] = np.nan
    rng = np.random.default_rng(0)
    out = []
    for (obj, scale), g in df.groupby(["object", "scale"], sort=False):
        mesh = load_cad(objects, obj, root)
        mesh.apply_scale(scale)
        pts = np.asarray(trimesh.sample.sample_surface(mesh, n_points, seed=int(rng.integers(1 << 31)))[0])
        diameter = float(np.linalg.norm(np.ptp(mesh.vertices, 0)))
        center = (mesh.vertices.min(0) + mesh.vertices.max(0)) / 2
        sym = Symmetry(np.eye(3)[None], center)
        flips = measured_symmetries(pts, center, diameter, sym_tol)
        objects[obj].setdefault("symmetries", flips)
        sym_g = Symmetry(np.stack([np.eye(3)] + [FLIPS[n] for n in flips]), center)
        g = g.copy()
        g["diameter_m"] = diameter
        g["kind"] = objects[obj]["kind"]
        has = g["T_pred"].notna().to_numpy()
        for c in ("e_t", "e_q", "add", "adds", "speed", "e_q_sym", "add_sym", "speed_sym"):
            g[c] = np.nan
        if has.any():
            T_gt = np.stack(g.loc[has, "T"].map(np.asarray)).reshape(-1, 4, 4)
            T_pr = np.stack(g.loc[has, "T_pred"].map(np.asarray)).reshape(-1, 4, 4)
            err = pose_errors(T_pr, T_gt, sym, pts)
            g.loc[has, "e_t"], g.loc[has, "e_q"], g.loc[has, "add"] = err.e_t, err.e_q, err.add
            g.loc[has, "speed"] = speed_score(err)
            err_s = pose_errors(T_pr, T_gt, sym_g, pts)
            g.loc[has, "e_q_sym"], g.loc[has, "add_sym"] = err_s.e_q, err_s.add
            g.loc[has, "speed_sym"] = speed_score(err_s)
            adds = []
            for Tp, Tg in zip(T_pr, T_gt):
                tree = cKDTree(pts @ Tg[:3, :3].T + Tg[:3, 3])
                adds.append(tree.query(pts @ Tp[:3, :3].T + Tp[:3, 3])[0].mean())
            g.loc[has, "adds"] = adds
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    res["range_bin"] = pd.cut(res["range_m"], RANGE_BINS, right=False).astype(str)
    return res


def summary(g: pd.DataFrame) -> dict:
    from .metrics import auc

    n = len(g)
    p = g[g["T_pred"].notna()]

    def rel(c):
        return (p[c] / p["diameter_m"]).to_numpy()

    return {
        "n_samples": n, "n_trajectories": int(g["trajectory"].nunique()), "coverage": len(p) / n if n else np.nan,
        "e_t_cm_mean": 100 * p["e_t"].mean(), "e_t_cm_median": 100 * p["e_t"].median(),
        "e_q_deg_mean": p["e_q"].mean(), "e_q_deg_median": p["e_q"].median(),
        "e_q_sym_deg_mean": p["e_q_sym"].mean(), "e_q_sym_deg_median": p["e_q_sym"].median(),
        "speed_mean": p["speed"].mean(), "speed_sym_mean": p["speed_sym"].mean(),
        "speed_sym_median": p["speed_sym"].median(),
        "add_0.1d": float((rel("add") < 0.1).sum() / n) if n else np.nan,
        "adds_0.1d": float((rel("adds") < 0.1).sum() / n) if n else np.nan,
        "add_sym_0.1d": float((rel("add_sym") < 0.1).sum() / n) if n else np.nan,
        "within_5cm_5deg_sym": float(((p["e_t"] < 0.05) & (p["e_q_sym"] < 5)).sum() / n) if n else np.nan,
        "auc_add_0.1d": auc(rel("add"), n, 0.1), "auc_adds_0.1d": auc(rel("adds"), n, 0.1),
        "within_5cm_5deg": float(((p["e_t"] < 0.05) & (p["e_q"] < 5)).sum() / n) if n else np.nan,
        "runtime_ms_median": p["runtime_ms"].median(),
        "rate_hz": 1000 / p["runtime_ms"].median() if len(p) and p["runtime_ms"].notna().any() else np.nan,
    }


def evaluate(run_dir: Path, pred_path: Path, *, root=None, n_points: int = 2000, sym_tol: float = 0.01,
             subset: str | None = None, name: str | None = None) -> dict:
    """Score a predictions file (``oshd_bench.PoseWriter`` format) against ``run_dir/gt.parquet``.

    ``subset``: a pandas query on the GT rows (e.g. ``"object == 'satellite_mockup'"``) for
    single-object methods; it is recorded in the metrics. ``name``: the output stem (default: the
    predictions' setting); give one per subset so the full-split metrics are not overwritten."""
    run_dir, pred_path = Path(run_dir), Path(pred_path)
    gt = pd.read_parquet(run_dir / "gt.parquet")
    pred = pd.read_parquet(pred_path)
    meta_p = pred_path.with_suffix(".json")
    pred_meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    objects = json.loads((run_dir / "objects.json").read_text())
    prep = json.loads((run_dir / "prepare.json").read_text())
    # the samples the prediction run attempted (it may have used --stride / --limit)
    gt = gt.iloc[::pred_meta.get("stride") or 1]
    if pred_meta.get("limit"):
        gt = gt.iloc[:pred_meta["limit"]]
    if subset:
        gt = gt.query(subset)
    res = errors(gt, pred, objects, root, n_points, sym_tol)
    stem = name or pred_path.stem.removeprefix("predictions_")
    out = pred_path.parent
    res.drop(columns=["T", "T_pred"]).to_parquet(out / f"per_sample_{stem}.parquet", index=False)

    groups = {"all": summary(res)}
    for col in ("kind", "mount", "range_bin"):
        for k, g in res.groupby(col):
            groups[f"{col}={k}"] = summary(g)
    metrics = {"predictions": pred_path.name, "run_dir": str(run_dir), "prepare": prep, "subset": subset,
               "symmetries": {k: v.get("symmetries", []) for k, v in objects.items() if "symmetries" in v},
               "prediction_meta": {k: v for k, v in pred_meta.items() if k != "hardware"},
               "hardware": pred_meta.get("hardware"), "groups": groups}
    (out / f"metrics_{stem}.json").write_text(json.dumps(metrics, indent=2, default=float))
    table = pd.DataFrame(groups).T
    table.to_csv(out / f"metrics_{stem}.csv")

    cols = {"n_samples": "samples", "coverage": "coverage", "e_t_cm_median": "e_t med (cm)",
            "e_t_cm_mean": "e_t mean (cm)", "e_q_deg_median": "e_q med (°)", "e_q_sym_deg_median": "e_q sym med (°)",
            "e_q_sym_deg_mean": "e_q sym mean (°)", "within_5cm_5deg": "5cm/5°", "within_5cm_5deg_sym": "5cm/5° sym",
            "add_0.1d": "ADD<0.1d", "add_sym_0.1d": "ADD sym<0.1d", "adds_0.1d": "ADD-S<0.1d",
            "auc_adds_0.1d": "AUC ADD-S", "speed_mean": "SPEED", "speed_sym_mean": "SPEED sym",
            "rate_hz": "Hz"}
    t = table[list(cols)].rename(columns=cols)
    fmt = {"samples": "{:.0f}"} | {c: "{:.3f}" for c in ("coverage", "5cm/5°", "5cm/5° sym", "ADD<0.1d", "ADD sym<0.1d",
                                                         "ADD-S<0.1d", "AUC ADD-S", "SPEED", "SPEED sym")}
    lines = ["| group | " + " | ".join(t.columns) + " |", "|---|" + "---:|" * len(t.columns)]
    for name, r in t.iterrows():
        lines.append(f"| {name} | " + " | ".join(fmt.get(c, "{:.2f}").format(v) for c, v in r.items()) + " |")
    a = groups["all"]
    paper = (f"Paper row (means, rotation up to the measured symmetries): e_t = {a['e_t_cm_mean']:.2f} cm, "
             f"e_q = {a['e_q_sym_deg_mean']:.1f}°, S = {a['speed_sym_mean']:.3f}")
    title = f"# {pred_meta.get('method', stem)}, {prep['domain']} / {prep['task']} / {prep['split']} ({stem})"
    about = (f"{len(res)} samples ({prep['stride']}-frame stride) from {res['trajectory'].nunique()} trajectories"
             f"{f', subset {subset}' if subset else ''}; benchmark {prep.get('benchmark')}.")
    syms = "; ".join(f"{k}: {', '.join(v) or 'none'}" for k, v in metrics["symmetries"].items())
    md = [title, "", about, "", paper, "", f"Measured symmetries (for the `sym` columns): {syms}.", "", *lines, ""]
    (out / f"RESULTS_{stem}.md").write_text("\n".join(md))
    print("\n".join(md))
    return metrics


# ----------------------------------------------------------------------------- real splits


#: Real targets of the transfer experiments. The CubeSat recordings (007-009) predate the rig
#: calibration: their ground truth projects centimetres off the cube in the RealSense image.
REAL_TARGETS = ("antena", "antena_1_solar")


def prepare_real(out: Path, ev, *, sensor: str = "realsense_color", split: str = "test", stride: int = 1,
                 objects: tuple[str, ...] = REAL_TARGETS) -> Path:
    """A real split in the sim-run format (``samples.parquet``, ``gt.parquet``, ``objects.json``,
    ``prepare.json``), so ``sim-eval`` scores real and twin predictions with the same metrics.

    Samples are those of ``oshd-bench gt`` (every frame of ``sensor`` where one of ``objects``,
    by default the two mockups, projects into the image); ``T`` is ``T_sensor_model`` in
    the target's CAD frame (``models/objects.yaml``), ``scale`` 1, ``mount`` "real", ``mask_px``
    unset (-1). The real targets get no symmetry in ``sim-eval`` beyond the measured 180° flips of
    their CAD."""
    from .gt import build_gt

    cfg = ev.cfg
    samp = cfg.get("sampling", {})
    rate_hz, window_ms = float(samp.get("events_rate_hz", 20)), float(samp.get("events_window_ms", 50))
    df = build_gt(ev.ds, sensor, split, "real", list(objects), stride=stride,
                  symmetries=cfg.get("symmetries"), events_rate_hz=rate_hz, events_window_ms=window_ms)
    if df.empty:
        raise SystemExit(f"no real {split} sample for {sensor}: download the data (oshd download --benchmark ... "
                         f"--domain real --split {split} -m {sensor} -m mocap)")
    root = Path(ev.ds.root)
    objects = {}
    for name in sorted(df["object"].unique()):
        obj = ev.ds.objects[name]
        path = Path(obj.mesh_path)
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        import trimesh

        m = trimesh.load(path, force="mesh")
        ext = np.ptp(m.vertices, axis=0)
        objects[name] = {"cad": str(rel), "kind": name, "extent_m": ext.round(4).tolist(),
                         "diameter_m": float(np.linalg.norm(ext)), "n_vertices": len(m.vertices)}
    gt = df.assign(scale=1.0, mask_px=-1, mount="real")[
        ["trajectory", "index", "t_ns", "object", "scale", "mask_px", "T", "range_m", "mount", "illumination",
         "ang_rate_dps"]]
    out.mkdir(parents=True, exist_ok=True)
    gt.to_parquet(out / "gt.parquet", index=False)
    gt.drop(columns=["T", "range_m"]).to_parquet(out / "samples.parquet", index=False)
    (out / "objects.json").write_text(json.dumps(objects, indent=2))
    bench = ev.ds.benchmark if hasattr(ev.ds, "benchmark") else None
    from oshd.splits import load_benchmark

    (out / "prepare.json").write_text(json.dumps({
        "benchmark": bench, "revision": load_benchmark(bench)["revision"] if bench else None,
        "domain": "real", "task": None, "split": split, "sensor": sensor, "stride": stride,
        **({"events_rate_hz": rate_hz, "events_window_ms": window_ms,
            "events_note": "index = reference time at events_rate_hz; a sample uses the events in "
                           "[t_ns - events_window_ms, t_ns)"} if sensor == "events" else {}),
        "n_trajectories": int(gt["trajectory"].nunique()), "n_samples": len(gt),
        "objects": gt["object"].value_counts().to_dict()}, indent=2))
    print(f"{len(gt)} samples from {gt['trajectory'].nunique()} real trajectories -> {out}")
    return out
