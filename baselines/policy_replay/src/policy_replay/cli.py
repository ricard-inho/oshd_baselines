"""``policy-replay``: Table II (frozen PPO on estimator-substituted observations).

    policy-replay verify                         # mocap reconstruction vs logged actions, all trajectories
    policy-replay run --key mocap                # privileged row
    policy-replay run --key rgb --source rgb_firefly/megapose/FT [--latency]
    policy-replay all [--source rgb=...] [--latency]   # mocap + best (lowest S_real) per group
    policy-replay fake-predictions --modality rgb_realsense   # GT + noise, for smoke tests only
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

GROUP_KEYS = ["rgb", "rgbd", "lidar", "events"]


def _bench():
    import pooh_bench
    from pooh_bench.evaluate import load_config as load_bench_config

    return pooh_bench, load_bench_config(pooh_bench.repo_root() / "configs" / "bench.yaml")


def _block(bcfg: dict, modality: str) -> dict:
    for b in bcfg["table1"]:
        if b["modality"] == modality:
            return b
    raise KeyError(f"modality {modality!r} not in configs/bench.yaml table1")


def _label(bcfg: dict, modality: str, method: str, setting: str) -> str:
    try:
        b = _block(bcfg, modality)
        m = next((m["label"] for m in b["methods"] if m["key"] == method), method)
        return f"{m} {setting} ({b['label']})"
    except KeyError:
        return f"{method} {setting} ({modality})"


def best_sources(results, bcfg: dict) -> dict[str, str]:
    """Lowest S_real per Table I group from ``tables/table1_pose.csv`` (``pooh-bench eval``)."""
    p = results / "tables" / "table1_pose.csv"
    if not p.exists():
        return {}
    t1 = pd.read_csv(p)
    out = {}
    for g in GROUP_KEYS:
        c = t1[(t1["group"] == g) & t1["S_real"].notna()]
        if len(c):
            r = c.loc[c["S_real"].idxmin()]
            out[g] = f"{r['modality_key']}/{r['method_key']}/{r['setting']}"
    return out


def _test_names(ds, split: str, domain: str) -> list[str]:
    return [n for n in ds.split(split, domain) if ds[n].has("robot_messages") and ds[n].has("mocap")]


def cmd_run(a, key: str, source: str | None) -> dict:
    from .replay import Policy, load_config, write_replay

    pb, bcfg = _bench()
    cfg = load_config(a.config)
    ds, results = pb.open_dataset(), pb.results_dir()
    names = _test_names(ds, a.split, a.domain)
    policy = Policy(cfg["policy"]["onnx"])
    latency = a.latency or bool(cfg["estimator"].get("add_runtime_latency", False))
    if key == "mocap" and source is None:
        path, meta = write_replay(results, key, ds, names, cfg, policy, "mocap/" + cfg["robot"]["base_body"],
                                  "Mocap (privileged)")
    else:
        if source is None:
            raise SystemExit(f"--source modality/method/setting needed for key {key!r}")
        modality, method, setting = source.split("/")
        pp = pb.prediction_path(results, modality, method, setting, a.domain, a.split)
        if not pp.exists():
            raise SystemExit(f"no predictions at {pp}")
        obj = ds.objects[cfg["estimator"]["object"]]
        sym = pb.symmetry_for(obj, bcfg.get("symmetries", {}).get(obj.name))
        path, meta = write_replay(results, key, ds, names, cfg, policy, source,
                                  _label(bcfg, modality, method, setting), pd.read_parquet(pp), latency,
                                  obj, sym.transforms)
    print(f"{path}: mean {meta['mean_delta_a']}, p95 {meta['p95_delta_a']}, steps {meta['n_steps']}, "
          f"without estimate {meta['n_steps_without_estimate']}"
          + (f", vs mocap replay mean {meta.get('mean_delta_a_vs_mocap_replay')}" if key != "mocap" else ""))
    return meta


def cmd_verify(a) -> None:
    from .estimator import RigConstants, mocap_track
    from .logs import load_log
    from .replay import Policy, load_config, replay_trajectory

    pb, _ = _bench()
    cfg = load_config(a.config)
    ds = pb.open_dataset()
    policy = Policy(cfg["policy"]["onnx"])
    rc = RigConstants(ds, cfg)
    idx = cfg["actions"]["compare_idx"]
    names = cfg["actions"]["names"]
    for n in ds.local_trajectory_names:
        traj = ds[n]
        if not (traj.has("robot_messages") and traj.has("mocap")):
            continue
        log = load_log(traj, cfg)
        track, src = mocap_track(traj, cfg, rc)
        act, s, _ = replay_trajectory(policy, cfg, log, track)
        d = act[:, idx] - log.actions[:, idx]
        dn = np.linalg.norm(d, axis=1)
        dn = dn[np.isfinite(dn)]
        corr = [np.corrcoef(act[s.valid, i], log.actions[s.valid, i])[0, 1] for i in idx]
        print(f"{n} ({src}): {len(dn)} steps, |da| mean {dn.mean():.3f} median {np.median(dn):.3f} "
              f"p95 {np.percentile(dn, 95):.3f}")
        print("   mean|d| " + " ".join(f"{names[i]}={v:.2f}" for i, v in zip(idx, np.nanmean(np.abs(d), 0))))
        print("   corr    " + " ".join(f"{names[i]}={c:.2f}" for i, c in zip(idx, corr)))


def cmd_fake(a) -> None:
    """GT poses + noise written as a clearly fake method (smoke tests; reads the GT ``T`` column)."""
    from pooh_bench.gt import gt_poses
    from pooh_bench.symmetry import cube_rotations
    from pooh_dataset.geometry import rotvec_to_rotmat

    pb, bcfg = _bench()
    results = pb.results_dir()
    sensor = _block(bcfg, a.modality)["sensor"]
    gt = pd.read_parquet(pb.gt_path(results, sensor, a.domain, a.split))
    rng = np.random.default_rng(a.seed)
    keep = rng.random(len(gt)) >= a.drop
    gt = gt[keep].reset_index(drop=True)
    T = gt_poses(gt).copy()
    rv = rng.normal(0, np.radians(a.noise_rot_deg), (len(T), 3))
    T[:, :3, :3] = T[:, :3, :3] @ np.stack([rotvec_to_rotmat(v) for v in rv])
    T[:, :3, 3] += rng.normal(0, a.noise_t_m, (len(T), 3))
    if a.symmetric:  # random equivalent pose about the cube's symmetry centre
        obj = pb.open_dataset().objects[gt["object"].iloc[0]]
        S = pb.symmetry_for(obj, bcfg.get("symmetries", {}).get(obj.name)).transforms
        T = T @ S[rng.integers(0, len(cube_rotations()), len(T))]
    path = pb.prediction_path(results, a.modality, a.method, a.setting, a.domain, a.split)
    w = pb.PoseWriter(path, {"method": a.method, "FAKE": "ground truth + noise, smoke test only",
                             "noise_t_m": a.noise_t_m, "noise_rot_deg": a.noise_rot_deg, "drop": a.drop})
    for k, r in enumerate(gt.itertuples()):
        w.add(r.trajectory, r.sensor, r.t_ns, r.object, T[k], 1.0, a.runtime_ms)
    print(f"{w.save()}: {len(w)} fake poses")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="policy-replay")
    ap.add_argument("--config", default=None, help="default: baselines/policy_replay/configs/replay.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--split", default="test")
        p.add_argument("--domain", default="real")
        p.add_argument("--latency", action="store_true", help="estimate usable at t + runtime_ms")

    r = sub.add_parser("run")
    r.add_argument("--key", required=True, choices=["mocap", *GROUP_KEYS])
    r.add_argument("--source", default=None, help="modality/method/setting, e.g. rgb_firefly/megapose/FT")
    common(r)
    al = sub.add_parser("all")
    al.add_argument("--source", action="append", default=[], help="override: rgb=rgb_firefly/spnv2/FT")
    common(al)
    sub.add_parser("verify")
    f = sub.add_parser("fake-predictions")
    f.add_argument("--modality", required=True)
    f.add_argument("--method", default="FAKE_gt_noise")
    f.add_argument("--setting", default="ZS")
    f.add_argument("--split", default="test")
    f.add_argument("--domain", default="real")
    f.add_argument("--noise-t-m", type=float, default=0.01)
    f.add_argument("--noise-rot-deg", type=float, default=2.0)
    f.add_argument("--drop", type=float, default=0.1, help="fraction of samples without a prediction")
    f.add_argument("--runtime-ms", type=float, default=50.0)
    f.add_argument("--symmetric", action="store_true", help="return a random symmetric-equivalent pose")
    f.add_argument("--seed", type=int, default=0)

    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a, a.key, a.source)
    elif a.cmd == "all":
        pb, bcfg = _bench()
        src = best_sources(pb.results_dir(), bcfg)
        for o in a.source:
            k, v = o.split("=", 1)
            src[k] = v
        print("sources:", json.dumps(src))
        cmd_run(a, "mocap", None)
        for k in GROUP_KEYS:
            if k in src:
                cmd_run(a, k, src[k])
            else:
                print(f"{k}: no source (run `pooh-bench eval` first or pass --source {k}=...)", file=sys.stderr)
    elif a.cmd == "verify":
        cmd_verify(a)
    elif a.cmd == "fake-predictions":
        cmd_fake(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
