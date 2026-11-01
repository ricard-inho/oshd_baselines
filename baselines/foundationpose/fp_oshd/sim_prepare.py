"""Inputs for FoundationPose on simulated (SRB) trajectories: target CAD + sample list + GT.

Runs on the host, in the baseline's uv environment (needs ``usd-core`` and oshd):

    .venv/bin/python -m fp_oshd.sim_prepare --domain sim_zerog --task docking --split test --stride 10

The split comes from the frozen benchmark ``--benchmark`` (default ``v2``: oshd's
``benchmark_v2``, Hub revision df3530b; ``none`` = the dataset's growing ``splits.json``).
For every trajectory of the split it

1. takes the target's CAD from the dataset (``models/sim/<model>/model.obj``, published with the
   sim data, see osh_dataset_pipeline/sim/target_models.py): metres, in the frame of the mocap
   body ``target``, unscaled; the run's scale is ``traj.target["scale"]``,
2. writes the sample list: every ``--stride``-th RGB frame where the target's semantic mask
   (class ``target``) has at least ``--min-mask-px`` pixels.

Outputs in ``--out`` (default ``results/<domain>_<task>_<split>/``):

* ``samples.parquet``: trajectory, index, t_ns, object, scale, mask_px (what inference reads)
* ``gt.parquet``: the same rows + ``T`` (``T_cam_target``, 16 floats row-major, rgb optical
  frame) and ``range_m``. Only ``eval_sim`` reads it.
* ``objects.json``: object (model name) -> CAD path, diameter (unscaled), source USD.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="fp_oshd.sim_prepare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="dataset root (default: oshd's default)")
    ap.add_argument("--domain", default="sim_zerog")
    ap.add_argument("--task", default="docking")
    ap.add_argument("--split", default="test")
    ap.add_argument("--benchmark", default="v2", help="frozen benchmark of the split, or 'none'")
    ap.add_argument("--stride", type=int, default=10, help="every N-th RGB frame (20 Hz -> 10 = 2 Hz)")
    ap.add_argument("--min-mask-px", type=int, default=100, help="minimum target mask area of a sample")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-trajectories", type=int, default=None)
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- CAD


def cad_entry(traj) -> dict:
    """objects.json entry of a trajectory's target: the dataset's CAD (``models/sim/``)."""
    import trimesh

    path = traj.target_model_path
    if path is None or not path.exists():
        raise SystemExit(f"{traj.name}: no target CAD in the dataset; rerun `oshd download` (oshd >= 0.5.0 "
                         "fetches models/sim/ with the sim data)")
    info = json.loads((path.parent / "info.json").read_text())
    m = trimesh.load(path, force="mesh", process=False)
    return {"cad": str(path), "source": info.get("source"), "kind": traj.target["kind"],
            "extent_m": np.ptp(m.vertices, axis=0).round(4).tolist(),
            "diameter_m": float(np.linalg.norm(np.ptp(m.vertices, axis=0))),  # bounding-box diagonal (unscaled)
            "n_vertices": len(m.vertices), "n_faces": len(m.faces)}


# ----------------------------------------------------------------------------- samples + GT


def build(a) -> None:
    from oshd import OshDataset
    from oshd.labels import camera_pose
    from oshd.splits import load_benchmark

    warnings.filterwarnings("ignore", message=".*real trajectories have no official split.*")
    bench = None if a.benchmark in ("", "none") else a.benchmark
    ds = OshDataset(a.root, benchmark=bench)
    names = [t for t in ds.split(a.split, a.domain, task=a.task) if ds[t].has("rgb")]
    if a.limit_trajectories:
        names = names[:a.limit_trajectories]
    if not names:
        raise SystemExit(f"no downloaded trajectory for {a.domain}/{a.task}/{a.split}: run `oshd download "
                         f"--benchmark {a.benchmark} --domain {a.domain} --task {a.task} --split {a.split} "
                         "-m rgb -m depth -m semantic -m mocap`")
    out = Path(a.out or HERE / "results" / f"{a.domain}_{a.task}_{a.split}")
    out.mkdir(parents=True, exist_ok=True)

    objects: dict[str, dict] = {}
    rows = []
    for t in tqdm(names, desc="samples"):
        traj = ds[t]
        if traj.target is None:
            raise SystemExit(f"{t}: no entry in models/sim/targets.json; rerun `oshd download` for it")
        name, scale = traj.target["model"], float(traj.target["scale"])
        if name not in objects:
            objects[name] = cad_entry(traj)
        target_id = traj.semantic_classes["target"]
        ts = traj.timestamps("rgb")
        for i in range(0, len(ts), a.stride):
            t_ns = int(ts[i])
            n_px = int((traj.frame("semantic", i) == target_id).sum())
            if n_px < a.min_mask_px:
                continue
            T_wc, ok_c = camera_pose(traj, "rgb", t_ns)
            T_wt, ok_t = traj.body_pose("target", t_ns)
            if not (ok_c and ok_t):
                continue
            T = np.linalg.inv(T_wc) @ T_wt
            rows.append({"trajectory": t, "index": i, "t_ns": t_ns, "object": name, "scale": scale,
                         "mask_px": n_px, "T": T.reshape(-1).tolist(), "range_m": float(np.linalg.norm(T[:3, 3])),
                         "mount": traj.metadata["task_info"].get("target_mount")})
    gt = pd.DataFrame(rows)
    gt.to_parquet(out / "gt.parquet", index=False)
    gt.drop(columns=["T", "range_m"]).to_parquet(out / "samples.parquet", index=False)
    (out / "objects.json").write_text(json.dumps(objects, indent=2))
    (out / "prepare.json").write_text(json.dumps({
        "dataset_root": str(ds.root), "benchmark": bench, "revision": load_benchmark(bench)["revision"] if bench else None,
        "domain": a.domain, "task": a.task, "split": a.split, "stride": a.stride,
        "min_mask_px": a.min_mask_px, "n_trajectories": len(names), "n_samples": len(gt),
        "objects": {k: int((gt.object == k).sum()) for k in objects}}, indent=2))
    print(f"{len(gt)} samples from {len(names)} trajectories, objects "
          f"{ {k: round(v['diameter_m'], 3) for k, v in objects.items()} } -> {out}")


def main(argv=None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
