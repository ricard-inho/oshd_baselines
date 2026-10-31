"""Metrics of ``fp_oshd.predict_sim`` against the simulator ground truth (``gt.parquet``).

Host side, in the baseline's uv environment:

    .venv/bin/python -m fp_oshd.eval_sim --run-dir results/sim_zerog_docking_test

Per sample (``oshd_bench.metrics`` conventions): ``e_t`` (distance between the predicted and
true CAD bounding-box centres, m), ``e_q`` (geodesic rotation error, deg), ADD and ADD-S (mean /
closest model-point distance, m), the SPEED score. ``e_q`` and ADD are given twice: strict (no
symmetry) and ``_sym``, minimised over the CAD's measured symmetries: the 180° rotations about
its bounding-box axes that map the surface onto itself to within ``--sym-tol`` of the diameter
(ADD-S after the flip; the satellite mockup has all three, the CubeSats the one about z). Samples without a
prediction count as failures in the accuracies and AUCs (``coverage`` says how many there
are); the error means / medians are over the predicted samples.

Writes into the run dir: ``per_sample_<setting>.parquet``, ``metrics_<setting>.json`` (overall
and per target kind, mount and range bin), ``metrics_<setting>.csv`` and ``RESULTS_<setting>.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RANGE_BINS = [0.0, 0.3, 0.5, 1.0, 2.0, np.inf]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="fp_oshd.eval_sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="results/sim_zerog_docking_test")
    ap.add_argument("--setting", default="ORACLE_MASK")
    ap.add_argument("--n-points", type=int, default=2000, help="CAD points for ADD / ADD-S")
    ap.add_argument("--sym-tol", type=float, default=0.01,
                    help="a 180° flip is a symmetry if it moves the surface < this x diameter (ADD-S)")
    return ap.parse_args(argv)


FLIPS = {"Rx180": np.diag([1.0, -1, -1]), "Ry180": np.diag([-1.0, 1, -1]), "Rz180": np.diag([-1.0, -1, 1])}


def measured_symmetries(pts: np.ndarray, center: np.ndarray, diameter: float, tol: float) -> list[str]:
    """180° flips about the bounding-box axes that leave the sampled surface in place (ADD-S)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    return [n for n, R in FLIPS.items() if tree.query((pts - center) @ R.T + center)[0].mean() < tol * diameter]


def errors(gt: pd.DataFrame, pred: pd.DataFrame, objects: dict, n_points: int, sym_tol: float) -> pd.DataFrame:
    import trimesh
    from oshd_bench.metrics import pose_errors, speed_score
    from oshd_bench.symmetry import Symmetry
    from scipy.spatial import cKDTree

    df = gt.merge(pred[["trajectory", "t_ns", "T", "score", "runtime_ms"]].rename(columns={"T": "T_pred"}),
                  on=["trajectory", "t_ns"], how="left")
    rng = np.random.default_rng(0)
    out = []
    for (obj, scale), g in df.groupby(["object", "scale"], sort=False):
        mesh = trimesh.load(objects[obj]["cad"], force="mesh")
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
        for c in ("e_t", "e_q", "add", "adds", "speed", "e_q_sym", "add_sym"):
            g[c] = np.nan
        if has.any():
            T_gt = np.stack(g.loc[has, "T"].map(np.asarray)).reshape(-1, 4, 4)
            T_pr = np.stack(g.loc[has, "T_pred"].map(np.asarray)).reshape(-1, 4, 4)
            err = pose_errors(T_pr, T_gt, sym, pts)
            g.loc[has, "e_t"], g.loc[has, "e_q"], g.loc[has, "add"] = err.e_t, err.e_q, err.add
            g.loc[has, "speed"] = speed_score(err)
            err_s = pose_errors(T_pr, T_gt, sym_g, pts)
            g.loc[has, "e_q_sym"], g.loc[has, "add_sym"] = err_s.e_q, err_s.add
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
    from oshd_bench.metrics import auc

    n = len(g)
    p = g[g["T_pred"].notna()]

    def rel(c):
        return (p[c] / p["diameter_m"]).to_numpy()

    return {
        "n_samples": n, "n_trajectories": int(g["trajectory"].nunique()), "coverage": len(p) / n if n else np.nan,
        "e_t_cm_mean": 100 * p["e_t"].mean(), "e_t_cm_median": 100 * p["e_t"].median(),
        "e_q_deg_mean": p["e_q"].mean(), "e_q_deg_median": p["e_q"].median(),
        "e_q_sym_deg_mean": p["e_q_sym"].mean(), "e_q_sym_deg_median": p["e_q_sym"].median(),
        "speed_mean": p["speed"].mean(),
        "add_0.1d": float((rel("add") < 0.1).sum() / n) if n else np.nan,
        "adds_0.1d": float((rel("adds") < 0.1).sum() / n) if n else np.nan,
        "add_sym_0.1d": float((rel("add_sym") < 0.1).sum() / n) if n else np.nan,
        "within_5cm_5deg_sym": float(((p["e_t"] < 0.05) & (p["e_q_sym"] < 5)).sum() / n) if n else np.nan,
        "auc_add_0.1d": auc(rel("add"), n, 0.1), "auc_adds_0.1d": auc(rel("adds"), n, 0.1),
        "within_5cm_5deg": float(((p["e_t"] < 0.05) & (p["e_q"] < 5)).sum() / n) if n else np.nan,
        "runtime_ms_median": p["runtime_ms"].median(), "rate_hz": 1000 / p["runtime_ms"].median() if len(p) else np.nan,
    }


def main(argv=None) -> int:
    a = parse_args(argv)
    run = Path(a.run_dir)
    gt = pd.read_parquet(run / "gt.parquet")
    pred_p = run / f"predictions_{a.setting}.parquet"
    pred = pd.read_parquet(pred_p)
    pred_meta = json.loads(pred_p.with_suffix(".json").read_text())
    objects = json.loads((run / "objects.json").read_text())
    # the samples the prediction run attempted (it may have used --stride / --limit)
    gt = gt.iloc[::pred_meta.get("stride", 1)]
    if pred_meta.get("limit"):
        gt = gt.iloc[:pred_meta["limit"]]
    res = errors(gt, pred, objects, a.n_points, a.sym_tol)
    res.drop(columns=["T", "T_pred"]).to_parquet(run / f"per_sample_{a.setting}.parquet", index=False)

    groups = {"all": summary(res)}
    for col in ("kind", "mount", "range_bin"):
        for k, g in res.groupby(col):
            groups[f"{col}={k}"] = summary(g)
    metrics = {"setting": a.setting, "symmetries": {k: v.get("symmetries", []) for k, v in objects.items()}, "prediction_meta": {k: v for k, v in pred_meta.items() if k != "hardware"},
               "hardware": pred_meta.get("hardware"), "groups": groups}
    (run / f"metrics_{a.setting}.json").write_text(json.dumps(metrics, indent=2, default=float))
    table = pd.DataFrame(groups).T
    table.to_csv(run / f"metrics_{a.setting}.csv")

    cols = {"n_samples": "samples", "coverage": "coverage", "e_t_cm_median": "e_t med (cm)",
            "e_t_cm_mean": "e_t mean (cm)", "e_q_deg_median": "e_q med (°)", "e_q_sym_deg_median": "e_q sym med (°)",
            "e_q_sym_deg_mean": "e_q sym mean (°)", "within_5cm_5deg": "5cm/5°", "within_5cm_5deg_sym": "5cm/5° sym",
            "add_0.1d": "ADD<0.1d", "add_sym_0.1d": "ADD sym<0.1d", "adds_0.1d": "ADD-S<0.1d",
            "auc_adds_0.1d": "AUC ADD-S", "speed_mean": "SPEED", "rate_hz": "Hz"}
    t = table[list(cols)].rename(columns=cols)
    fmt = {c: "{:.0f}" for c in ("samples",)} | {c: "{:.3f}" for c in ("coverage", "5cm/5°", "5cm/5° sym", "ADD<0.1d",
                                                                     "ADD sym<0.1d", "ADD-S<0.1d", "AUC ADD-S",
                                                                     "SPEED")}
    lines = ["| group | " + " | ".join(t.columns) + " |", "|---|" + "---:|" * len(t.columns)]
    for name, r in t.iterrows():
        lines.append(f"| {name} | " + " | ".join(fmt.get(c, "{:.2f}").format(v) for c, v in r.items()) + " |")
    prep = pred_meta["prepare"]
    title = f"# FoundationPose, {prep['domain']} / {prep['task']} / {prep['split']} ({a.setting})"
    about = (f"{len(res)} samples ({prep['stride']}-frame stride) from {res['trajectory'].nunique()} trajectories; "
             f"mask: {pred_meta['mask']}; {pred_meta['mode']}, {pred_meta['refine_iters']} refiner iterations.")
    syms = "; ".join(f"{k}: {', '.join(v.get('symmetries', [])) or 'none'}" for k, v in objects.items()
                     if "symmetries" in v)
    md = [title, "", about, "", f"Measured symmetries (for the `sym` columns): {syms}.", "", *lines, ""]
    (run / f"RESULTS_{a.setting}.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
