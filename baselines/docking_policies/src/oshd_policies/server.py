"""Policy server for SRB's policy bridge (``srb/tasks/mobile/inspection/policy_bridge.py``).

    uv run policy-server --student ../../results/policies/bc/rgb/student.pt --port 5555

SRB connects (``SRB_ACTION_SERVER=127.0.0.1:5555``; ``closed_loop.py --server``) and sends,
at every agent step, the raw sensors, proprioception and docking command; the server turns
them into the student's inputs with ``inputs.py`` (as in training) and replies with the
thrust. One thread per connection (one per simulator process); inference is serialised.

``--policy zero`` replies with no thrust (a bridge check: the platform must stay put);
``--policy teacher`` runs the PPO teacher (``teacher.py``) on the privileged observation terms the
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
        elif self.kind == "teacher":
            from .teacher import Teacher

            self.model = Teacher().to(a.device).eval()

    def __call__(self, msg: dict) -> np.ndarray:
        n = len(msg["command"])
        if self.kind == "zero":
            return np.zeros((n, 8), np.float32)
        t0 = time.perf_counter()
        if self.kind == "teacher":
            with torch.no_grad():
                obs = torch.from_numpy(self.model.observation(msg["proprio"])).to(self.device)
                act = self.model(obs).cpu().numpy()
            self.t_infer += time.perf_counter() - t0
            self.calls += 1
            return act.astype(np.float32)
        with torch.no_grad():
            s = student_inputs(msg, self.model.modalities, self.device)
            p = torch.from_numpy(I.proprio_from_terms(msg["proprio"])).to(self.device)
            c = torch.from_numpy(np.asarray(msg["command"], np.float32)).to(self.device)
            act = self.model(s, p, c).cpu().numpy()
        self.t_infer += time.perf_counter() - t0
        self.calls += 1
        return act.astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="policy-server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["student", "teacher", "zero"], default="student")
    ap.add_argument("--student", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump", default=None, help="directory: save every --dump-every-th request (debugging)")
    ap.add_argument("--dump-every", type=int, default=50)
    a = ap.parse_args(argv)
    count = [0]
    policy = Policy(a)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"connected: {self.client_address}", flush=True)
            while (msg := recv_msg(self.request)) is not None:
                with LOCK:
                    action = policy(msg)
                    if a.dump and count[0] % a.dump_every == 0:
                        from pathlib import Path

                        Path(a.dump).mkdir(parents=True, exist_ok=True)
                        Path(a.dump, f"request_{count[0]:06d}.pkl").write_bytes(pickle.dumps(msg))
                    count[0] += 1
                send_msg(self.request, {"action": action})
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
