"""``oshd-bench`` command line: build ground-truth sample lists and evaluate."""

from __future__ import annotations

import argparse

SENSORS = ["firefly", "realsense_color", "lidar", "events"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oshd-bench")
    ap.add_argument("--config", default=None, help="default: <repo>/configs/bench.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gt", help="write ground-truth sample lists to results/gt/<sensor>/<domain>_<split>.parquet")
    g.add_argument("--sensor", action="append", choices=SENSORS, help="default: all sensors with calibration")
    g.add_argument("--split", action="append", default=None, help="default: train, val, test")
    g.add_argument("--domain", action="append", default=None, help="default: real, synthetic")

    o = sub.add_parser("oracle-detections", help="ground-truth boxes/masks as a detections file (debugging)")
    o.add_argument("--sensor", required=True, choices=SENSORS)
    o.add_argument("--split", default="test")
    o.add_argument("--domain", default="real")

    sub.add_parser("eval", help="write results/tables/*.csv|json|tex")

    sp = sub.add_parser("sim-prepare", help="shared sample list + GT of a sim split -> results/sim/<run>/")
    sp.add_argument("--domain", default="sim_zerog")
    sp.add_argument("--task", default="docking", help="docking / inspection / approach / orbit, or all")
    sp.add_argument("--split", default="test")
    sp.add_argument("--benchmark", default=None, help="default: dataset.benchmark of configs/bench.yaml")
    sp.add_argument("--sensor", default="rgb")
    sp.add_argument("--stride", type=int, default=10, help="every N-th frame (rgb is 20 Hz: 10 -> 2 Hz)")
    sp.add_argument("--min-mask-px", type=int, default=100)
    sp.add_argument("--min-points", type=int, default=30, help="--sensor lidar: returns on the target")
    sp.add_argument("--limit-trajectories", type=int, default=None)
    sp.add_argument("--out", default=None, help="default: results/sim/<domain>_<task>_<split>_<benchmark>")

    rp = sub.add_parser("real-prepare", help="a real split in the sim-run format -> results/real/<run>/ (for sim-eval)")
    rp.add_argument("--sensor", default="realsense_color", choices=SENSORS)
    rp.add_argument("--split", default="test")
    rp.add_argument("--stride", type=int, default=1)
    rp.add_argument("--objects", nargs="+", default=None, help="default: the mockups antena, antena_1_solar")
    rp.add_argument("--out", default=None, help="default: results/real/real_<sensor>_<split>_<benchmark>")

    se = sub.add_parser("sim-eval", help="score a sim predictions file against results/sim/<run>/gt.parquet")
    se.add_argument("--run-dir", required=True, help="e.g. results/sim/sim_zerog_docking_test_v2")
    se.add_argument("--pred", required=True, help="predictions_<SETTING>.parquet (oshd_bench.PoseWriter)")
    se.add_argument("--subset", default=None, help="pandas query on the GT rows, e.g. \"object == 'satellite_mockup'\"")
    se.add_argument("--name", default=None, help="output stem (default: the setting); use one per --subset")
    se.add_argument("--n-points", type=int, default=2000, help="CAD points for ADD / ADD-S")
    se.add_argument("--sym-tol", type=float, default=0.01,
                    help="a 180° flip is a symmetry if it moves the surface < this x diameter (ADD-S)")

    a = ap.parse_args(argv)
    from .evaluate import Evaluator, gt_path, load_config, repo_root

    root = repo_root()
    cfg = load_config(a.config or root / "configs" / "bench.yaml")
    if a.cmd == "real-prepare":
        from pathlib import Path

        from . import sim

        ev = Evaluator(cfg, root)
        bench = ev.ds.benchmark
        name = "_".join(x for x in ("real", a.sensor, a.split, bench) if x)
        sim.prepare_real(Path(a.out) if a.out else root / "results" / "real" / name, ev, sensor=a.sensor,
                         split=a.split, stride=a.stride, **({"objects": tuple(a.objects)} if a.objects else {}))
        return 0
    if a.cmd in ("sim-prepare", "sim-eval"):
        import os
        from pathlib import Path

        from . import sim

        data_root = os.environ.get("OSH_DATASET_ROOT") or cfg.get("dataset", {}).get("root")
        if a.cmd == "sim-prepare":
            bench = a.benchmark or os.environ.get("OSHD_BENCHMARK") or cfg.get("dataset", {}).get("benchmark")
            bench = None if bench in ("", "none") else bench
            task = None if a.task == "all" else a.task
            name = sim.run_name(a.domain, task, a.split, bench) + ("" if a.sensor == "rgb" else f"_{a.sensor}")
            out = Path(a.out) if a.out else root / "results" / "sim" / name
            sim.prepare(out, domain=a.domain, task=task, split=a.split, benchmark=bench, root=data_root,
                        sensor=a.sensor, stride=a.stride, min_mask_px=a.min_mask_px, min_points=a.min_points,
                        limit_trajectories=a.limit_trajectories)
        else:
            sim.evaluate(Path(a.run_dir), Path(a.pred), root=data_root, n_points=a.n_points, sym_tol=a.sym_tol,
                         subset=a.subset, name=a.name)
        return 0
    ev = Evaluator(cfg, root)

    if a.cmd == "gt":
        from oshd import MissingCalibrationError

        from .gt import build_gt

        s = cfg.get("sampling", {})
        for sensor in a.sensor or SENSORS:
            for domain in a.domain or ["real", "synthetic"]:
                for split in a.split or ["train", "val", "test"]:
                    if not ev.ds.split(split, domain):
                        continue
                    try:
                        df = build_gt(ev.ds, sensor, split, domain, cfg.get("objects"),
                                      stride=int(s.get("stride", {}).get(sensor, 1)),
                                      symmetries=cfg.get("symmetries"),
                                      events_rate_hz=float(s.get("events_rate_hz", 20)),
                                      events_window_ms=float(s.get("events_window_ms", 50)),
                                      lidar_max_range=float(s.get("lidar_max_range_m", 15)))
                    except MissingCalibrationError as e:
                        print(f"skip {sensor}/{domain}/{split}: {e}")
                        break
                    p = gt_path(ev.results, sensor, domain, split)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(p, index=False)
                    print(f"{p}: {len(df)} samples")
        return 0

    if a.cmd == "oracle-detections":
        from .oracle import oracle_detections

        print(oracle_detections(ev, a.sensor, a.split, a.domain))
        return 0

    if a.cmd == "eval":
        for name, p in ev.run().items():
            print(f"{name}: {p}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
