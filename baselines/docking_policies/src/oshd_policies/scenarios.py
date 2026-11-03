"""Closed-loop evaluation scenarios: the SRB scenarios that recorded the docking test split.

    uv run policy-scenarios --out ../../results/policies/docking_test_v3/scenarios.json

Every test trajectory is one episode of an SRB scenario process (``metadata.scenario``: Hydra
overrides and seed; 2-3 consecutive episodes per process). Re-running each scenario with its
seed for as many episodes as it recorded gives the evaluation the test split's targets,
mounts, scales, spins, lighting and docking speeds (none of which the BC students saw in
training) and as many episodes as the split has trajectories; the policy under test drives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policy-scenarios", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--benchmark", default="v3")
    ap.add_argument("--task", default="docking")
    a = ap.parse_args(argv)
    from oshd import OshDataset

    ds = OshDataset()
    out: dict[tuple, dict] = {}
    for name in ds.split(a.split, domain="sim_zerog", task=a.task, benchmark=a.benchmark):
        m = ds[name].metadata
        sc = m["scenario"]
        ti = m["task_info"]
        key = (sc["name"], int(sc["seed"]))
        if key in out:
            out[key]["episodes"] += 1
            out[key]["trajectories"].append(name)
            continue
        out[key] = {
            "trajectory": name,
            "trajectories": [name],
            "episodes": 1,
            "scenario": sc["name"],
            "seed": int(sc["seed"]),
            "overrides": list(sc["overrides"]),
            "episode_length_s": float(ti["episode_length_s"]),
            "target_mount": ti.get("target_mount"),
            "target_kind": ti.get("target_kind"),
            "spinning": max((ti.get("target_spin_rate_deg") or [0, 0])) > 0,
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"split": a.split, "benchmark": a.benchmark, "task": a.task,
                                 "scenarios": list(out.values())}, indent=1) + "\n")
    print(f"{len(out)} scenarios, {sum(s['episodes'] for s in out.values())} episodes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
