"""Policy server for SRB's policy bridge (``srb/tasks/mobile/inspection/policy_bridge.py``).

    uv run policy-server --student ../../results/policies/bc/rgb/student.pt --port 5555

SRB connects (``SRB_ACTION_SERVER=127.0.0.1:5555``; ``closed_loop.py --server``) and sends,
at every agent step, the raw sensors, proprioception and docking command; the server turns
them into the student's inputs with ``inputs.py`` (as in training) and replies with the
thrust. One thread per connection (one per simulator process); inference is serialised.

``--policy zero`` replies with no thrust (a bridge check: the platform must stay put);
``--policy estimator`` (``--student <estimator.pt>``) is the learned estimator -> PPO pipeline:
the target centre from the estimator (``estimator.py``), fused over time
(``modular.StaticTargetFilter``), gives the docking goal of the PPO teacher; it turns in place
while it has no estimate. ``--policy teacher`` runs the PPO teacher (``teacher.py``) on the privileged observation terms the
bridge forwards (a bridge check: it must dock like the teacher inside SRB).
"""

from __future__ import annotations

import argparse
import pickle
import socket
import socketserver
import struct
import threading
import time

import numpy as np
import torch

from . import inputs as I

LOCK = threading.Lock()


def recv_msg(sock) -> dict | None:
    head = _recv(sock, 8)
    if head is None:
        return None
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(_recv(sock, n))


def send_msg(sock, msg: dict) -> None:
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("<Q", len(data)) + data)


def _recv(sock, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def student_inputs(msg: dict, modalities, device: str) -> dict[str, torch.Tensor]:
    n = len(msg["command"])
    out = {}
    for m in modalities:
        if m == "rgb":
            x = [I.rgb_input(msg["rgb"][e]) for e in range(n)]
        elif m == "depth":
            x = [I.depth_input(msg["depth"][e]) for e in range(n)]
        elif m == "lidar":
            x = [I.lidar_input(msg["lidar"][e]) for e in range(n)]
        elif m == "events":
            x = [I.events_input(ev[:, 1], ev[:, 2], ev[:, 3]) for ev in msg["events"]]
        else:
            raise KeyError(m)
        out[m] = torch.from_numpy(np.stack(x)).to(device)
    return out


class Policy:
    def __init__(self, a):
        self.kind = a.policy
        self.device = a.device
        self.calls, self.t_infer = 0, 0.0
        if self.kind == "student":
            from .bc import load_student

            self.model = load_student(a.student, a.device)
        elif self.kind == "estimator":
            from .estimator import load_estimator
            from .teacher import Teacher

            self.teacher = Teacher().to(a.device).eval()
            self.model = load_estimator(a.student, a.device)
            self.envs: list | None = None
            self.oracle_goal = a.oracle_goal
            self.log_path, self.plog = a.perception_log, {}
            self.search_yaw = np.radians(a.search_yaw_deg)
            self.filter_kw = {"q": a.filter_q, "reset_after": a.filter_reset_after}
        elif self.kind in ("teacher", "shadow", "perception"):
            from .teacher import Teacher

            self.teacher = Teacher().to(a.device).eval()
            self.model = self.teacher
            if self.kind == "perception":  # the student's goal-term head feeds the teacher
                from .bc import load_student

                self.model = load_student(a.student, a.device)
                self.prev_cmd = None
                self.oracle_goal = a.oracle_goal
                self.log_path, self.plog = a.perception_log, {}
            if self.kind == "shadow":  # the teacher drives, the student only predicts (logged)
                from .bc import load_student

                self.model = load_student(a.student, a.device)
                self.log = {"student": [], "teacher": [], "t_ns": [], "goal_pred": [], "goal_true": []}
                self.log_path = a.shadow_log

    def __call__(self, msg: dict) -> np.ndarray:
        n = len(msg["command"])
        if self.kind == "zero":
            return np.zeros((n, 8), np.float32)
        t0 = time.perf_counter()
        if self.kind == "shadow":
            with torch.no_grad():
                obs = torch.from_numpy(self.teacher.observation(msg["proprio"])).to(self.device)
                act_t = self.teacher(obs).cpu().numpy()
                s = student_inputs(msg, self.model.modalities, self.device)
                p = torch.from_numpy(I.proprio_from_terms(msg["proprio"])).to(self.device)
                c = torch.from_numpy(np.asarray(msg["command"], np.float32)).to(self.device)
                act_s = self.model(s, p, c).cpu().numpy()
            self.log["student"].append(act_s); self.log["teacher"].append(act_t); self.log["t_ns"].append(msg["t_ns"])
            if self.model.goal_pred is not None:
                self.log["goal_pred"].append((self.model.goal_pred * self.model.g_std + self.model.g_mean).cpu().numpy())
                self.log["goal_true"].append(np.concatenate([np.asarray(msg["proprio"]["tf_pos_robot_to_goal"]),
                                                             np.asarray(msg["proprio"]["tf_rot6d_robot_to_goal"])], -1))
            if len(self.log["t_ns"]) % 100 == 0:
                np.savez(self.log_path, **{k: np.asarray(v) for k, v in self.log.items()})
            return act_t.astype(np.float32)
        if self.kind == "perception":
            return self.perception(msg)
        if self.kind == "estimator":
            return self.estimate_then_control(msg)
        if self.kind == "teacher":
            with torch.no_grad():
                obs = torch.from_numpy(self.model.observation(msg["proprio"])).to(self.device)
                act = self.model(obs).cpu().numpy()
            self.t_infer += time.perf_counter() - t0
            self.calls += 1
            return act.astype(np.float32)
        with torch.no_grad():
            s = student_inputs(msg, self.model.modalities, self.device)
            if self.model.input_name == "state":
                from .teacher import Teacher

                p = torch.from_numpy(Teacher.observation(msg["proprio"])).to(self.device)
            else:
                p = torch.from_numpy(I.proprio_from_terms(msg["proprio"])).to(self.device)
            c = torch.from_numpy(np.asarray(msg["command"], np.float32)).to(self.device)
            act = self.model(s, p, c).cpu().numpy()
        self.t_infer += time.perf_counter() - t0
        self.calls += 1
        return act.astype(np.float32)


def _perception(self, msg: dict) -> np.ndarray:
    """Learned perception -> PPO teacher: the student's predicted goal terms (position and
    attitude of the goal in the robot frame) replace the simulator's; the goal velocity follows
    from the docking command (the target is static: the goal moves with ``d``)."""
    from .bc import AUX_DIM
    from .modular import rot6d  # noqa: F401  (same conventions)

    with torch.no_grad():
        s = student_inputs(msg, self.model.modalities, self.device)
        p = torch.from_numpy(I.proprio_from_terms(msg["proprio"])).to(self.device)
        c = torch.from_numpy(np.asarray(msg["command"], np.float32)).to(self.device)
        self.model(s, p, c)
        g = (self.model.goal_pred * self.model.g_std + self.model.g_mean).cpu().numpy()
    cmd = np.asarray(msg["command"], np.float64)
    new = np.asarray(msg["new_episode"], bool)
    if self.prev_cmd is None or len(self.prev_cmd) != len(cmd):
        self.prev_cmd = cmd.copy()
    self.prev_cmd[new] = cmd[new]
    terms = {k: np.asarray(v, np.float32).copy() for k, v in msg["proprio"].items()}
    if self.log_path:
        self.plog.setdefault("goal_pred", []).append(g.copy())
        self.plog.setdefault("goal_true", []).append(np.concatenate(
            [terms["tf_pos_robot_to_goal"], terms["tf_rot6d_robot_to_goal"]], -1))
        self.plog.setdefault("mask_px", []).append(np.array([0]))
        if len(self.plog["goal_pred"]) % 50 == 0:
            np.savez(self.log_path, **{k: np.asarray(v) for k, v in self.plog.items()})
    if not self.oracle_goal:
        terms["tf_pos_robot_to_goal"] = g[:, 0:3].astype(np.float32)
        terms["tf_rot6d_robot_to_goal"] = g[:, 3:9].astype(np.float32)
    # goal velocity (world) from the command's change, in the robot frame (R_r^T v; rot6d = R[:, :2])
    b = np.arctan2(cmd[:, 1], cmd[:, 0])
    v_w = np.stack([np.cos(b), np.sin(b), np.zeros_like(b)], -1) * ((cmd[:, 2] - self.prev_cmd[:, 2]) / 0.1)[:, None]
    r6 = np.asarray(msg["proprio"]["tf_rot6d_robot"], np.float64).reshape(-1, 3, 2)
    c0, c1 = r6[:, :, 0], r6[:, :, 1]
    Rm = np.stack([c0, c1, np.cross(c0, c1)], -1)
    terms["vel_lin_goal"] = np.einsum("nji,nj->ni", Rm, v_w).astype(np.float32)
    terms["vel_ang_goal"] = np.zeros_like(terms["vel_ang_goal"])
    self.prev_cmd = cmd.copy()
    with torch.no_grad():
        obs = torch.from_numpy(self.teacher.observation(terms)).to(self.device)
        return self.teacher(obs).cpu().numpy().astype(np.float32)


Policy.perception = _perception


def _estimate_then_control(self, msg: dict) -> np.ndarray:
    """Learned estimator -> static-target filter -> docking goal -> PPO teacher."""
    from scipy.spatial.transform import Rotation as R

    from .modular import GoalTracker, StaticTargetFilter, search_terms

    n = len(msg["command"])
    if self.envs is None or len(self.envs) != n:
        self.envs = [(StaticTargetFilter(**self.filter_kw), GoalTracker()) for _ in range(n)]
    with torch.no_grad():
        s = student_inputs(msg, self.model.modalities, self.device)
        mu, sd = self.model(s)
        visible = (self.model.vis_logit > 0).cpu().numpy()
    mu, sd = mu.cpu().numpy().astype(np.float64), sd.cpu().numpy().astype(np.float64)
    terms_all = {k: np.asarray(v, np.float32).copy() for k, v in msg["proprio"].items()}
    for e, (filt, goal) in enumerate(self.envs):
        if bool(np.asarray(msg["new_episode"])[e]):
            filt.reset()
            goal.reset()
        rp = np.asarray(msg["robot_pose"][e], float)
        centre_w = rp[:3] + R.from_quat(rp[3:]).apply(mu[e])
        sd_e = float(np.max(sd[e, :2])) if visible[e] else np.inf  # not in view: no measurement
        if self.oracle_goal:  # check: the simulator's target centre through the same path
            centre_w, sd_e = np.asarray(msg["privileged"]["target_center"][e], float), 0.01
        if self.log_path:
            self.plog.setdefault("centre_est", []).append(centre_w.copy())
            self.plog.setdefault("centre_true", []).append(np.asarray(msg["privileged"]["target_center"][e], float))
            self.plog.setdefault("sd", []).append(sd_e)
            self.plog.setdefault("env", []).append(e)
            self.plog.setdefault("tag", []).append(str(msg.get("tag", "")))
            if len(self.plog["env"]) % 60 == 0:
                np.savez(self.log_path, **{k: np.asarray(v) for k, v in self.plog.items()})
        was = filt.x is not None
        x = filt.update(centre_w, sd_e)
        if self.log_path:
            self.plog.setdefault("centre_filt", []).append(np.full(3, np.nan) if x is None else x.copy())
            self.plog.setdefault("visible", []).append(bool(visible[e]))
        if (x is not None) != was:
            goal.reset()  # no goal velocity across the switch from searching
        terms = search_terms(goal, rp, self.search_yaw) if x is None else goal.terms(rp, x, np.asarray(msg["command"][e]))
        for k, v in terms.items():
            terms_all[k][e] = v
    with torch.no_grad():
        obs = torch.from_numpy(self.teacher.observation(terms_all)).to(self.device)
        return self.teacher(obs).cpu().numpy().astype(np.float32)


Policy.estimate_then_control = _estimate_then_control


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policy-server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["student", "teacher", "zero", "shadow", "perception", "estimator"], default="student")
    ap.add_argument("--student", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shadow-log", default="shadow.npz")
    ap.add_argument("--perception-log", default=None)
    ap.add_argument("--search-yaw-deg", type=float, default=30.0, help="estimator: turn per step while searching")
    ap.add_argument("--filter-q", type=float, default=0.0005, help="estimator: filter process noise per step (m)")
    ap.add_argument("--filter-reset-after", type=int, default=200,
                    help="estimator: consecutive rejected measurements before the filter restarts")
    ap.add_argument("--oracle-goal", action="store_true", help="perception: keep the true goal terms (check)")
    ap.add_argument("--dump", default=None, help="directory: save every --dump-every-th request (debugging)")
    ap.add_argument("--dump-every", type=int, default=50)
    a = ap.parse_args(argv)
    count = [0]
    policy = Policy(a)
    #: per-connection state of the stateful policies (one simulator each): the estimator's
    #: filters and goal trackers, the perception policy's previous command
    STATE_ATTRS = ("envs", "prev_cmd")
    states: dict = {}

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"connected: {self.client_address}", flush=True)
            while (msg := recv_msg(self.request)) is not None:
                with LOCK:
                    for k, v in states.get(self.client_address, {}).items():
                        setattr(policy, k, v)
                    for k in STATE_ATTRS:  # a new connection starts from no state
                        if hasattr(policy, k) and k not in states.get(self.client_address, {}):
                            setattr(policy, k, None)
                    action = policy(msg)
                    states[self.client_address] = {k: getattr(policy, k) for k in STATE_ATTRS if hasattr(policy, k)}
                    if a.dump and count[0] % a.dump_every == 0:
                        from pathlib import Path

                        Path(a.dump).mkdir(parents=True, exist_ok=True)
                        Path(a.dump, f"request_{count[0]:06d}.pkl").write_bytes(pickle.dumps(msg))
                    count[0] += 1
                send_msg(self.request, {"action": action})
            states.pop(self.client_address, None)
            ms = 1000 * policy.t_infer / max(policy.calls, 1)
            print(f"closed: {self.client_address} ({policy.calls} calls, {ms:.1f} ms/inference)", flush=True)

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((a.host, a.port), Handler) as srv:
        srv.daemon_threads = True
        print(f"policy server ({a.policy} {a.student or ''}) on {a.host}:{a.port}", flush=True)
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
