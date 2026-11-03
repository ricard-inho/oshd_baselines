#!/usr/bin/env python3
"""Run closed-loop docking episodes in SRB (inside the SRB container; stdlib only).

    python closed_loop.py <scenarios.json> --out <dir> [--policy teacher] [--workers 3]

One ``srb agent eval`` process per scenario (``policy-scenarios``): the scenario's overrides
and seed, as many episodes as it recorded in the split (``--episodes`` to change), ``SRB_EPISODE_LOG`` writing one ``.npz`` per episode
under ``<out>/episodes/`` (``srb/tasks/mobile/inspection/episode_log.py``). Score with
``policy-metrics``. Finished scenarios are skipped (resume).

Policies:
``--sensors`` runs the dataset recorder (into a scratch directory, deleted) so that physics and
sensors are those of the recordings: 120 Hz physics and rendering, 20 Hz cameras, 10 Hz lidar,
and with ``--events`` the v2e event camera (else ``event_camera_backend=noop``).

- ``teacher``: the shipped PPO policy (``osh_dataset_pipeline/sim/policies``, installed in
  SRB's ``logs/`` by ``install_policies.sh``, picked by the latest-run lookup like the
  recordings), on its privileged state.
- ``server``: actions from an external policy server (``SRB_ACTION_SERVER``, set
  ``--server``): BC students and FoundationPose -> PPO. The PPO process still runs; its actions
  are replaced.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def chown(root: Path) -> None:
    """Hand the outputs to the host user (``HOST_UID=uid:gid``; the container runs as root)."""
    ids = os.environ.get("HOST_UID")
    if not ids or os.geteuid() != 0:
        return
    uid, gid = (int(v) for v in ids.split(":"))
    for p in [root, *root.rglob("*")]:
        try:
            os.chown(p, uid, gid)
        except OSError:
            pass


def run(sc: dict, a, idx: int) -> str:
    tag = sc["trajectory"]
    ep_dir = a.out / "episodes"
    episodes = a.episodes or sc.get("episodes", 1)
    have = sorted(ep_dir.glob(f"{tag}_env*.npz"))
    if len(have) >= episodes:
        return f"{tag}: done"
    # The scenario's episodes run side by side (one environment each, one episode per
    # environment): one simulator process renders all of them.
    num_envs, per_env = (episodes, 1) if a.parallel else (1, episodes)
    cmd = ["srb", "agent", "eval", "--headless", "-e", "zerog_lab_inspection", "--algo", "rsl_rl_ppo",
           f"env.num_envs={num_envs}", f"env.episode_length_s={sc['episode_length_s']}", f"env.seed={sc['seed']}",
           *sc["overrides"], *a.extra]
    env = {**os.environ, "SRB_EVAL_EPISODES": str(per_env), "SRB_EPISODE_LOG": str(ep_dir),
           "SRB_EPISODE_LOG_TAG": tag}
    raw = Path("/tmp/oshd_closed_loop") / a.out.name / tag
    if a.sensors:
        # The dataset recorder renders the sensors as in the recordings (120 Hz physics and
        # rendering, cameras at 20 Hz, lidar at 10 Hz, events); its files are thrown away.
        env["SRB_SIM_DATASET_DIR"] = str(raw)
        env["SRB_SIM_DATASET_OPTS"] = json.dumps({"camera_hz": 20, "lidar_hz": 10, "render_hz": 120.0,
                                                  "run_tag": tag, "min_duration_s": 1e9})
        if not a.events:
            cmd.append("env.event_camera_backend=noop")
    if a.server:
        env["SRB_ACTION_SERVER"] = a.server.replace("{worker}", str(idx % a.workers))
    t0 = time.time()
    for attempt in range(1 + a.retries):
        # A failed process (e.g. a PhysX start-up failure under GPU contention) restarts the
        # scenario; partial episodes of the failed attempt are removed.
        for p in ep_dir.glob(f"{tag}_env*.npz"):
            p.unlink()
        log = a.out / "logs" / f"{tag}.log"
        with open(log, "w") as f:
            f.write(" ".join(cmd) + "\n")
            f.flush()
            try:
                r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout)
                code = r.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        shutil.rmtree(raw, ignore_errors=True)
        if code == 0 and len(list(ep_dir.glob(f"{tag}_env*.npz"))) >= episodes:
            break
        os.replace(log, log.with_suffix(f".failed{attempt}.log"))
    chown(a.out)
    n = len(list(ep_dir.glob(f"{tag}_env*.npz")))
    return f"{tag}: exit {code}, {n}/{episodes} episode(s), {attempt + 1} attempt(s), {time.time() - t0:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=None, help="per scenario (default: as recorded)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", help="trajectory names")
    ap.add_argument("--server", default=None, help="host:port of the action server ({worker} expands)")
    ap.add_argument("--sensors", action="store_true",
                    help="dataset recorder on: the sensors and physics rate of the recordings")
    ap.add_argument("--events", action="store_true", help="with --sensors: emulate the event camera (v2e)")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sequential", dest="parallel", action="store_false",
                    help="one environment, the episodes one after the other (default: side by side)")
    ap.add_argument("extra", nargs="*", help="extra Hydra overrides")
    a = ap.parse_args()
    scenarios = json.loads(a.scenarios.read_text())["scenarios"]
    if a.only:
        scenarios = [s for s in scenarios if s["trajectory"] in a.only]
    if a.limit:
        scenarios = scenarios[:a.limit]
    (a.out / "episodes").mkdir(parents=True, exist_ok=True)
    (a.out / "logs").mkdir(parents=True, exist_ok=True)
    chown(a.out)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run, s, a, i) for i, s in enumerate(scenarios)]
        for i, f in enumerate(futs):
            print(f"[{i + 1}/{len(futs)} {(time.time() - t0) / 60:.0f} min] {f.result()}", flush=True)


if __name__ == "__main__":
    main()
