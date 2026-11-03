"""Behaviour-cloning cache of the docking train split: one row per policy step (10 Hz).

    uv run policy-bc-cache --out ../../work/bc_cache/docking_train_v3 [--workers 12]

From the released streams only (``steps``, ``mocap``, ``rgb``, ``depth``, ``lidar``,
``events``): at step ``k`` (``t_k``), the latest camera frame and lidar scan with
``t <= t_k``, the events of ``[t_k - 50 ms, t_k)``, the proprioceptive terms and the action
the teacher took (``steps``), and the docking command (the ``goal`` body's bearing around the
target, ``task_state.docking_goal_distance``). Inputs are made by ``inputs.py``, as in the
closed loop.

Output: ``rows.parquet`` (trajectory, step, proprio, command, action) and one uint8 array per
modality, ``<modality>.u8`` (numpy memmap, shape in ``cache.json``), row-aligned.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from . import inputs as I

SHAPES = {"rgb": (3, *I.IMAGE_HW), "depth": (1, *I.IMAGE_HW), "events": (2, *I.IMAGE_HW),
          "lidar": (1, I.LIDAR_CHANNELS, I.LIDAR_AZ_BINS)}


def _latest(ts: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the latest timestamp <= t (0 if none yet)."""
    return np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)


def _yaw(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def trajectory_rows(name: str, modalities: tuple[str, ...]) -> tuple[dict, dict[str, np.ndarray]]:
    from oshd import OshDataset

    t = OshDataset()[name]
    s = t.steps()
    tk = s["timestamp_ns"].astype(np.int64)
    n = len(tk)
    state = [json.loads(x) for x in s["task_state"]]
    g = t.mocap("goal")
    gi = _latest(g["timestamp_ns"], tk)
    # The goal faces the target centre: its bearing around the target is yaw + pi.
    bearing = _yaw(g["qx"][gi], g["qy"][gi], g["qz"][gi], g["qw"][gi]) + np.pi
    dist = np.array([x["docking_goal_distance"] for x in state], np.float32)
    rows = {
        "trajectory": [name] * n, "step": np.arange(n),
        "proprio": list(I.proprio_from_steps(s["observation"], t.metadata["steps_layout"]["observation"])),
        "command": list(I.command(bearing, dist)),
        "action": list(s["action"].astype(np.float32)),
        "phase": np.array([x["phase"] for x in state], np.float32),
    }
    arrays: dict[str, np.ndarray] = {}
    if "rgb" in modalities or "depth" in modalities:
        ci = _latest(t.timestamps("rgb"), tk)
        if "rgb" in modalities:
            arrays["rgb"] = np.stack([I.rgb_input(t.frame("rgb", int(i))) for i in ci])
        if "depth" in modalities:
            di = _latest(t.timestamps("depth"), tk)
            arrays["depth"] = np.stack([I.depth_input(t.depth(int(i))) for i in di])
    if "lidar" in modalities:
        li = _latest(t.timestamps("lidar"), tk)
        scans = {}
        for i in np.unique(li):
            sc = t.lidar_scan(int(i))
            scans[i] = I.lidar_input(np.stack([sc["x"], sc["y"], sc["z"]], 1))
        arrays["lidar"] = np.stack([scans[i] for i in li])
    if "events" in modalities:
        ev = t.events(int(tk[0] - I.EVENTS_WINDOW_NS), int(tk[-1] + 1))
        a = np.searchsorted(ev.t, tk - I.EVENTS_WINDOW_NS, side="left")
        b = np.searchsorted(ev.t, tk, side="left")
        arrays["events"] = np.stack([I.events_input(ev.x[i:j], ev.y[i:j], ev.p[i:j]) for i, j in zip(a, b)])
    return rows, arrays


def _job(args):
    name, modalities, out, offset = args
    rows, arrays = trajectory_rows(name, modalities)
    for m, arr in arrays.items():
        mm = np.memmap(out / f"{m}.u8", np.uint8, "r+", shape=(TOTAL[0], *SHAPES[m]))
        mm[offset:offset + len(arr)] = arr
        mm.flush()
        del mm
    return name, rows


TOTAL = [0]


def _init(total: int) -> None:
    TOTAL[0] = total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policy-bc-cache", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--benchmark", default="v3")
    ap.add_argument("--modalities", nargs="+", default=list(I.MODALITIES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    from oshd import OshDataset

    ds = OshDataset()
    names = ds.split(a.split, domain="sim_zerog", task="docking", benchmark=a.benchmark)[: a.limit]
    steps = [ds.trajectory_info(n)["counts"]["steps"] for n in names]
    offsets = np.concatenate([[0], np.cumsum(steps)[:-1]]).astype(int)
    total = int(sum(steps))
    a.out.mkdir(parents=True, exist_ok=True)
    for m in a.modalities:
        np.memmap(a.out / f"{m}.u8", np.uint8, "w+", shape=(total, *SHAPES[m])).flush()
    jobs = [(n, tuple(a.modalities), a.out, int(o)) for n, o in zip(names, offsets)]
    parts: dict[str, dict] = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(total,)) as ex:
        for i, (name, rows) in enumerate(ex.map(_job, jobs)):
            parts[name] = rows
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"[{i + 1}/{len(jobs)}] {name}", flush=True)
    df = pd.concat([pd.DataFrame(parts[n]) for n in names], ignore_index=True)
    assert len(df) == total, (len(df), total)
    df.to_parquet(a.out / "rows.parquet", index=False)
    meta_path = a.out / "cache.json"
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if old and old.get("trajectories") != names:
        old = {}  # a different split: earlier modalities are not row-aligned any more
    meta_path.write_text(json.dumps({
        "split": a.split, "benchmark": a.benchmark, "trajectories": names, "rows": total,
        "modalities": {**old.get("modalities", {}), **{m: [total, *SHAPES[m]] for m in a.modalities}},
        "image_hw": I.IMAGE_HW, "proprio_terms": I.PROPRIO_TERMS}, indent=1) + "\n")
    print(f"{total} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
