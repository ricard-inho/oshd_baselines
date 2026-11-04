"""FoundationPose -> PPO teacher, closed loop in SRB (paper Table III, modular row).

Runs inside the Docker image as a policy server for SRB's policy bridge
(``baselines/docking_policies``; protocol in ``oshd_policies.server``):

    FP_DOCKER_ARGS="--network host" FP_NO_LOCK=1 baselines/foundationpose/docker/run.sh \\
        python -m fp_oshd.closed_loop --port 5602

Per agent step (10 Hz), on the bridge's RGB-D frame:

1. **estimate** ``T_cam_target``: ``register()`` (``--refine-iters``) with the simulator's
   target mask (oracle 2D segmentation, as in the paper's FoundationPose rows) once the target
   is fully in view (``predict_track.fully_in_view``: >= ``--init-min-px`` px, off the border),
   again every ``--reregister-every`` steps while it is, else ``track_one()`` (``--track-iters``)
   from the previous pose;
2. **target centre** in the world: ``T_world_robot T_body_cam T_cam_target`` and the target's
   centre offset (the scenario's ``target_center_offset``; zero for arm-held and floating
   targets), fused over time with ``oshd_policies.modular.StaticTargetFilter`` (as the learned
   estimators; std ``--register-sd`` / ``--track-sd``). The robot's own pose is known;
3. **teacher**: the docking goal from the centre and the docking command
   (``oshd_policies.modular``), the PPO teacher's 41-d observation with these goal terms, and its
   action.

Before the first estimate the teacher is given a goal at the robot's position, turned towards
the target's pixels if some are in view (else by ``--search-yaw-deg``), and ``--back-off-m``
behind it if the target is large and cut by the image border: it turns (and backs away) until
the target is fully in view. While the target is out of view (``--min-visible-px``) there is no
estimate and tracking is dropped (``track_one`` would drift without it); the filter keeps the
last fused centre. (A search that turns away from a small, visible target never gets
close enough to register: the first smoke test with a 2000 px threshold spun in place.)
The target CAD is the scenario's (``tag`` -> the test trajectory's ``target_mesh()``).
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import socketserver
import sys
import threading
import time

import numpy as np

from .predict_sim import FP_DIR, untextured_to_vertex_colors
from .predict_track import fully_in_view

LOCK = threading.Lock()


def env_slice(msg: dict, e: int) -> dict:
    """The request of environment ``e`` alone (arrays and lists keep a leading axis of 1)."""
    out = {}
    for k, v in msg.items():
        if isinstance(v, dict):
            out[k] = env_slice(v, e)
        elif isinstance(v, np.ndarray) and v.ndim >= 1:
            out[k] = v[e:e + 1]
        elif isinstance(v, list):
            out[k] = v[e:e + 1]
        else:
            out[k] = v
    return out


def center_offset(meta: dict) -> np.ndarray:
    for o in meta["scenario"]["overrides"]:
        if "target_center_offset" in o:
            return np.asarray(json.loads(o.split("=", 1)[1]), float)
    return np.zeros(3)


class Session:
    """One environment of a simulator process (one scenario)."""

    def __init__(self, tag: str, shared, a):
        from oshd import OshDataset
        from oshd_policies.modular import GoalTracker, StaticTargetFilter

        FoundationPose = shared["FoundationPose"]
        traj = OshDataset()[tag]
        mesh = untextured_to_vertex_colors(traj.target_mesh())
        self.est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                  scorer=shared["scorer"], refiner=shared["refiner"], glctx=shared["glctx"],
                                  debug=0, debug_dir="/tmp/fp_debug")
        cal = traj.calibration
        self.K = np.asarray(cal.intrinsics("rgb").K, float).reshape(3, 3)
        self.T_body_cam = np.asarray(cal.T_body_cam("rgb"), float)
        self.offset = center_offset(traj.metadata)
        self.goal = GoalTracker(dt=0.1)
        self.filter = StaticTargetFilter()
        self.a = a
        self.reset()
        self.stats = {"register": 0, "track": 0, "search": 0, "register_ms": 0.0, "track_ms": 0.0}

    def reset(self):
        self.tracking = False
        self.searching = True
        self.since_register = 0
        self.last_px = 0
        self.partial = False
        self.last_sides = ""
        self.goal.reset()
        self.filter.reset()

    def target_center_w(self, msg: dict) -> np.ndarray | None:
        from scipy.spatial.transform import Rotation as R

        rgb = np.asarray(msg["rgb"][0], np.uint8)
        depth = np.asarray(msg["depth"][0], np.float32)
        mask = np.asarray(msg["target_mask"][0], bool)
        px = int(mask.sum())
        self.last_px = px
        self.partial = False
        self.last_sides = ""
        if px < self.a.min_visible_px:  # out of view: no estimate, and tracking is lost
            self.tracking = False
            return None
        has_depth = int(((depth >= 0.001) & mask).sum()) >= 4
        full = fully_in_view(mask, self.a.init_min_px) and has_depth
        # a large target that never fits in view (cut by one image side, e.g. a scaled target
        # above the camera): register from the partial view, with a larger std
        m = 2
        sides = [bool(e.any()) for e in (mask[:m], mask[-m:], mask[:, :m], mask[:, -m:])]  # top bottom left right
        self.last_sides = "".join("tblr"[i] for i, c in enumerate(sides) if c)
        # cut by one side, or two adjacent sides (a corner), but not across the image
        partial = (not full and has_depth and px >= self.a.partial_min_px
                   and len(self.last_sides) <= 2 and self.last_sides not in ("tb", "lr"))
        self.partial = partial
        t0 = time.perf_counter()
        import torch

        try:
            T = self._estimate(rgb, depth, mask, full or partial, t0)
        except torch.OutOfMemoryError:  # the GPU is shared with the simulators: no estimate this step
            torch.cuda.empty_cache()
            self.tracking = False
            self.stats["oom"] = self.stats.get("oom", 0) + 1
            return None
        if T is None:
            return None
        T = np.asarray(T.cpu() if hasattr(T, "cpu") else T, float)
        rp = np.asarray(msg["robot_pose"][0], float)
        T_wr = np.eye(4)
        T_wr[:3, :3] = R.from_quat(rp[3:]).as_matrix()
        T_wr[:3, 3] = rp[:3]
        T_wt = T_wr @ self.T_body_cam @ T
        return T_wt[:3, 3] + T_wt[:3, :3] @ self.offset

    def _estimate(self, rgb, depth, mask, can_register: bool, t0: float):
        if can_register and (not self.tracking or self.since_register >= self.a.reregister_every):
            T = self.est.register(K=self.K, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.a.refine_iters)
            self.tracking, self.since_register = True, 0
            self.stats["register"] += 1
            self.stats["register_ms"] += (time.perf_counter() - t0) * 1000
        elif self.tracking:
            T = self.est.track_one(rgb=rgb, depth=depth, K=self.K, iteration=self.a.track_iters)
            self.since_register += 1
            self.stats["track"] += 1
            self.stats["track_ms"] += (time.perf_counter() - t0) * 1000
        else:
            return None
        return T

    def act(self, msg: dict, teacher) -> np.ndarray:
        import torch
        from scipy.spatial.transform import Rotation as R

        if bool(np.asarray(msg["new_episode"])[0]):
            self.reset()
        rp = np.asarray(msg["robot_pose"][0], float)
        command = np.asarray(msg["command"][0], float)
        measured = self.target_center_w(msg)
        # std: grows as the target gets smaller in the image (far away), and for tracked poses
        # (Table I: register vs track)
        sd = self.a.register_sd * float(np.sqrt(self.a.ref_px / max(self.last_px, 1)))
        sd = float(np.clip(sd, self.a.register_sd, 0.3)) * (1.0 if self.since_register == 0 else self.a.track_factor)
        if self.partial:
            sd *= self.a.partial_factor
        centre = self.filter.update(measured, sd if measured is not None else np.inf)
        if (centre is None) != self.searching:  # no goal velocity across a switch
            self.goal.reset()
            self.searching = centre is None
        if centre is None:
            from oshd_policies.modular import search_terms

            # turn in place towards the target's pixels if some are in view, else by --search-yaw-deg
            mask = np.asarray(msg["target_mask"][0], bool)
            step, back = np.radians(self.a.search_yaw_deg), 0.0
            if mask.any():
                cx = np.nonzero(mask)[1].mean()
                # camera x right = robot -y: a target right of centre needs a negative yaw step
                step = float(np.clip(-np.arctan((cx - self.K[0, 2]) / self.K[0, 0]), -step, step))
                # a large target cut by the image border: back away until it fits in view
                if int(mask.sum()) >= self.a.init_min_px and not fully_in_view(mask, self.a.init_min_px):
                    back = self.a.back_off_m
            goal_terms = search_terms(self.goal, rp, step, back)
            self.stats["search"] += 1
        else:
            goal_terms = self.goal.terms(rp, centre, command)
        terms = {k: np.asarray(v[0], np.float32) for k, v in msg["proprio"].items()}
        true_goal = {k: terms[k].copy() for k in ("tf_pos_robot_to_goal", "vel_lin_goal")}
        terms.update(goal_terms)
        obs = torch.from_numpy(teacher.observation(terms)[None]).to(teacher.mean.device)
        with torch.no_grad():
            action = teacher(obs).cpu().numpy().astype(np.float32)
        if self.a.debug_log:
            true_c = np.asarray(msg["privileged"]["target_center"][0], float)
            rec = {"t_ns": int(msg["t_ns"]), "searching": self.searching, "tracking": self.tracking,
                   "mask_px": int(np.asarray(msg["target_mask"][0]).sum()), "sides": self.last_sides,
                   "partial": self.partial,
                   "centre_err": None if self.searching else (centre - true_c).round(4).tolist(),
                   "goal_pos_est": terms["tf_pos_robot_to_goal"].round(4).tolist(),
                   "goal_pos_true": true_goal["tf_pos_robot_to_goal"].round(4).tolist(),
                   "vel_goal_est": terms["vel_lin_goal"].round(4).tolist(), "action": action[0].round(3).tolist()}
            with open(self.a.debug_log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return action


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fp_oshd.closed_loop", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5602)
    ap.add_argument("--refine-iters", type=int, default=5)
    ap.add_argument("--track-iters", type=int, default=2)
    ap.add_argument("--init-min-px", type=int, default=400,
                    help="register once the mask has this many pixels and is off the border (Table I tracking: 2000; "
                         "the closed loop starts ~2.5 m away, where the target covers ~1000 px)")
    ap.add_argument("--reregister-every", type=int, default=10, help="agent steps (1 s) between re-registrations")
    ap.add_argument("--search-yaw-deg", type=float, default=30.0)
    ap.add_argument("--register-sd", type=float, default=0.03,
                    help="filter: std of a registered estimate (m) at --ref-px mask pixels; grows as 1/sqrt(px)")
    ap.add_argument("--ref-px", type=float, default=20000)
    ap.add_argument("--track-factor", type=float, default=1.5, help="filter: std factor of a tracked estimate")
    ap.add_argument("--min-visible-px", type=int, default=200, help="fewer mask pixels: no estimate, tracking lost")
    ap.add_argument("--partial-min-px", type=int, default=2000,
                    help="register a target cut by one image side if its mask is this large")
    ap.add_argument("--partial-factor", type=float, default=2.0, help="filter: std factor of a partial view")
    ap.add_argument("--back-off-m", type=float, default=0.3, help="search: back away when the target is cut")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-log", default=None, help="jsonl: per-step estimate vs. the simulator's target")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(FP_DIR))
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_seed  # upstream
    from oshd_policies.server import recv_msg, send_msg
    from oshd_policies.teacher import Teacher

    logging.getLogger().setLevel(logging.WARNING)
    set_seed(a.seed)
    shared = {"FoundationPose": FoundationPose, "scorer": ScorePredictor(), "refiner": PoseRefinePredictor(),
              "glctx": dr.RasterizeCudaContext()}
    teacher = Teacher().to("cuda").eval()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sessions, tag = None, None
            while (msg := recv_msg(self.request)) is not None:
                with LOCK:
                    if sessions is None:  # one estimator per environment
                        tag = msg["tag"]
                        sessions = [Session(tag, shared, a) for _ in range(len(msg["command"]))]
                        print(f"{tag}: connected ({len(sessions)} environments)", flush=True)
                    action = np.concatenate([s.act(env_slice(msg, e), teacher) for e, s in enumerate(sessions)])
                send_msg(self.request, {"action": action})
            if sessions is not None:
                st = {k: sum(s.stats[k] for s in sessions) for k in sessions[0].stats}
                print(f"{tag}: closed, {st['register']} registrations "
                      f"({st['register_ms'] / max(st['register'], 1):.0f} ms), {st['track']} tracked "
                      f"({st['track_ms'] / max(st['track'], 1):.0f} ms), {st['search']} search steps", flush=True)

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((a.host, a.port), Handler) as srv:
        srv.daemon_threads = True
        print(f"FoundationPose -> PPO server on {a.host}:{a.port}", flush=True)
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
