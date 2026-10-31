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

    a = ap.parse_args(argv)
    from .evaluate import Evaluator, gt_path, load_config, repo_root

    root = repo_root()
    cfg = load_config(a.config or root / "configs" / "bench.yaml")
    ev = Evaluator(cfg, root)

    if a.cmd == "gt":
        from oshd_dataset import MissingCalibrationError

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
