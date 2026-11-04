"""Shared pieces of the SPNv2-on-events baseline: paths, the time-surface input, the virtual
network camera, the satellite mockup's frame / keypoints / symmetry canonicalisation and the
yacs config built from upstream's defaults.

Adapted from ``baselines/spnv2/src/spnv2_oshd/common.py`` (same repo), which targets the real
CubeSat_U1 (24-fold cube symmetry, clicked corner keypoints). Here the target is the simulated
``satellite_mockup`` CAD of the ``sim_zerog/docking`` runs.

Frames used throughout:

* ``target``: the CAD frame of ``models/sim/satellite_mockup/model.obj`` (unscaled). oshd-bench's
  ``T_cam_target`` and the predictions are in this frame. Each run renders the CAD at a
  ``scale`` (0.8 .. 1.2), so the metric model of a sample is ``scale * vertices``.
* ``net``:   the CAD frame translated to the (scaled) bounding-box centre ``scale * c``.
  ``T_cam_net = T_cam_target @ Trans(scale * c)``. The measured symmetries of the mockup (the
  180 deg flips about its bounding-box axes, ``oshd_bench.sim.FLIPS``) are rotations about this
  origin, so the regressed translation is symmetry invariant and equals the bounding-box centre
  whose distance ``sim-eval`` reports as ``e_t``.
* ``cam``:   the optical frame of the sim ``rgb`` camera (x right, y down, z forward). The sim
  event camera is pixel-aligned with it (same K, same pose), so a pose in the event image is a
  pose in the ``rgb`` optical frame, which is what the shared sample list expects.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

BASELINE_DIR = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = BASELINE_DIR / "third_party" / "spnv2"
UPSTREAM_COMMIT = "debee80c43f226a22dce857b0a136de62e9e939d"
METHOD = "SPNv2"
OBJECT = "satellite_mockup"
SENSOR = "events"          # the input; the pose is in the pixel-aligned rgb optical frame
SETTING = "TS"             # time surfaces; one setting (trained on sim train, tested on sim test)

TEST_RUN = "sim_zerog_docking_test_v2"

# Time-surface input (see ``time_surface_image``)
WINDOW_NS = 50_000_000     # events in [t - 50 ms, t): the benchmark's event window
TAU_NS = 25_000_000        # exponential decay of the time surfaces (window / 2)
COUNT_CAP = 16             # per-pixel event count saturating the third channel (log scale)

# Network input: the 320x240 sim camera x2 -> 640x480, padded to 640x512 (sides divisible by 128,
# the P7 stride). Extra rows are black padding; cy is shifted by 16 px.
INPUT = ((640, 512), 2.0)


def ensure_upstream_on_path() -> None:
    core = str(UPSTREAM_DIR / "core")
    if not (UPSTREAM_DIR / "core").is_dir():
        raise SystemExit(f"upstream SPNv2 not found at {UPSTREAM_DIR}; run baselines/spnv2_events/setup.sh")
    if core not in sys.path:
        sys.path.insert(0, core)
    # torchvision's ImageNet EfficientNet weights go to weights/torch (setup.sh pre-fetches them)
    os.environ.setdefault("TORCH_HOME", str(BASELINE_DIR / "weights" / "torch"))


def repo_root() -> Path:
    return BASELINE_DIR.parents[1]


def work_dir() -> Path:
    """Converted training data and checkpoints: ``$SPNV2EV_WORK`` or ``<repo>/work/spnv2_events``."""
    if os.environ.get("SPNV2EV_WORK"):
        return Path(os.environ["SPNV2EV_WORK"])
    return repo_root() / "work" / "spnv2_events"


def run_dir(setting: str = SETTING) -> Path:
    return work_dir() / "runs" / setting


def results_dir(run: str = TEST_RUN) -> Path:
    return BASELINE_DIR / "results" / run


def open_dataset():
    """The benchmark dataset (``configs/bench.yaml``: benchmark v3, default cache or
    ``$OSH_DATASET_ROOT``)."""
    from oshd import OshDataset
    from oshd_bench import load_config

    cfg = load_config(repo_root() / "configs" / "bench.yaml")
    bench = os.environ.get("OSHD_BENCHMARK") or cfg.get("dataset", {}).get("benchmark")
    bench = None if bench in ("", "none") else bench
    root = os.environ.get("OSH_DATASET_ROOT") or cfg.get("dataset", {}).get("root")
    return OshDataset(root, benchmark=bench)


# --------------------------------------------------------------------------- time surface


def time_surface_image(ev, t_ref_ns: int, height: int, width: int) -> np.ndarray:
    """(H, W, 3) uint8 image of the events of ``[t_ref - WINDOW, t_ref)``:

    * channel 0: time surface of the negative events, ``exp((t_last - t_ref) / TAU)``
    * channel 1: the same for the positive events
    * channel 2: per-pixel event count (both polarities), ``log1p(n) / log1p(COUNT_CAP)``, clipped

    Pixels without an event are 0 in all channels. The image is fed to the network like
    upstream's 3-channel grayscale frames (ImageNet normalisation).
    """
    from oshd.events import event_histogram, time_surface

    ts = time_surface(ev, t_ref_ns, TAU_NS, height, width)           # (2, H, W), [neg, pos]
    n = event_histogram(ev, height, width).sum(0)                     # (H, W)
    cnt = np.minimum(np.log1p(n) / np.log1p(COUNT_CAP), 1.0)
    img = np.stack([ts[0], ts[1], cnt], -1)
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def window_events(traj, t_ns: int):
    return traj.events(int(t_ns) - WINDOW_NS, int(t_ns))


# --------------------------------------------------------------------------- virtual camera


@dataclass
class VirtualCamera:
    """Fixed pinhole camera the network is trained and evaluated in."""

    sensor: str
    width: int
    height: int
    K: list  # 3x3 row-major

    @property
    def Km(self) -> np.ndarray:
        return np.asarray(self.K, float).reshape(3, 3)

    @classmethod
    def from_intrinsics(cls, sensor: str, intr) -> VirtualCamera:
        (w, h), s = INPUT
        sw, sh = round(intr.width * s), round(intr.height * s)
        r = intr.resized(sw, sh)
        K = r.K.copy()
        K[0, 2] += (w - sw) / 2.0
        K[1, 2] += (h - sh) / 2.0
        return cls(sensor, w, h, K.tolist())

    def intrinsics(self, scale: float = 1.0):
        """oshd CameraIntrinsics of the virtual image (optionally rescaled)."""
        from oshd import CameraIntrinsics

        ci = CameraIntrinsics(self.sensor + "_virtual", self.width, self.height, self.Km, np.zeros(5), "none")
        return ci if scale == 1.0 else ci.resized(round(self.width * scale), round(self.height * scale))

    def scaled(self, scale: float) -> VirtualCamera:
        """The same camera at ``scale`` of the resolution (e.g. 0.25 for the mask / heatmap grid)."""
        ci = self.intrinsics(scale)
        return VirtualCamera(self.sensor, ci.width, ci.height, np.asarray(ci.K, float).tolist())

    def to_dict(self) -> dict:
        return asdict(self)


class Warper:
    """Raw sensor images -> virtual-camera images (resize + pad; undistort if the sensor has
    distortion), cached per source calibration. ``interpolation`` INTER_NEAREST for masks."""

    def __init__(self, vcam: VirtualCamera, interpolation: int = cv2.INTER_LINEAR):
        self.vcam = vcam
        self.interpolation = interpolation
        self._maps: dict = {}

    def __call__(self, image: np.ndarray, intr) -> np.ndarray:
        key = (intr.width, intr.height, tuple(np.ravel(intr.K)), tuple(np.ravel(intr.D)))
        if key not in self._maps:
            # Downscaling: area-resize first (anti-aliasing), then undistort from the resized image
            s = min(1.0, self.vcam.Km[0, 0] / intr.K[0, 0])
            src = intr.resized(round(intr.width * s), round(intr.height * s)) if s < 0.99 else intr
            D = np.asarray(src.D, float) if src.has_distortion else None
            m1, m2 = cv2.initUndistortRectifyMap(src.K, D, None, self.vcam.Km,
                                                 (self.vcam.width, self.vcam.height), cv2.CV_16SC2)
            self._maps[key] = (src.size, m1, m2)
        size, m1, m2 = self._maps[key]
        if size != (image.shape[1], image.shape[0]):
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA if self.interpolation == cv2.INTER_LINEAR
                               else cv2.INTER_NEAREST)
        return cv2.remap(image, m1, m2, self.interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# --------------------------------------------------------------------------- target frame / symmetry

#: 180 deg flips about the bounding-box axes (the symmetries ``sim-eval`` measures on the mockup CAD)
FLIPS = {"Rx180": np.diag([1.0, -1, -1]), "Ry180": np.diag([-1.0, 1, -1]), "Rz180": np.diag([-1.0, -1, 1])}
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


@dataclass
class TargetFrame:
    """The mockup's unscaled CAD: bounding-box centre ``c``, half extents, the 8 corner keypoints
    in the ``net`` frame (unit scale) and the symmetry rotations (identity + the flips)."""

    name: str
    c: np.ndarray          # (3,)
    half: np.ndarray       # (3,)
    rots: np.ndarray       # (G, 3, 3)
    symmetries: list       # names of the flips
    vertices: np.ndarray   # (N, 3) unscaled CAD vertices (subsampled), for boxes / overlays

    @property
    def kp_unit(self) -> np.ndarray:
        """(8, 3) bounding-box corners in the net frame at scale 1."""
        return CORNER_SIGNS * self.half

    def keypoints(self, scale: float) -> np.ndarray:
        return self.kp_unit * float(scale)

    def to_dict(self) -> dict:
        return {"name": self.name, "c": self.c.tolist(), "half": self.half.tolist(), "symmetries": self.symmetries,
                "rots": self.rots.tolist()}

    @classmethod
    def from_dict(cls, d: dict, vertices=None) -> TargetFrame:
        return cls(d["name"], np.asarray(d["c"], float), np.asarray(d["half"], float), np.asarray(d["rots"], float),
                   list(d["symmetries"]), np.zeros((0, 3)) if vertices is None else vertices)


def load_target(ds, trajectory: str, symmetries: tuple[str, ...] = ("Rx180", "Ry180", "Rz180"),
                n_vertices: int = 5000) -> TargetFrame:
    """The target frame from the unscaled CAD of ``trajectory`` (must be a ``satellite_mockup`` run).

    ``symmetries`` are the flips ``sim-eval`` measured on the CAD (``metrics_*.json`` of any
    evaluated run, ``symmetries.satellite_mockup``: all three for the mockup)."""
    traj = ds[trajectory]
    if traj.target is None or traj.target["model"] != OBJECT:
        raise SystemExit(f"{trajectory}: target is {traj.target}, expected {OBJECT}")
    mesh = traj.target_mesh(scaled=False)
    v = np.asarray(mesh.vertices, float)
    lo, hi = v.min(0), v.max(0)
    c, half = (lo + hi) / 2, (hi - lo) / 2
    rots = np.stack([np.eye(3)] + [FLIPS[n] for n in symmetries])
    rng = np.random.default_rng(0)
    sub = v[rng.choice(len(v), size=min(len(v), n_vertices), replace=False)]
    return TargetFrame(OBJECT, c, half, rots, list(symmetries), sub)


def T_trans(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = v
    return T


def target_to_net(T_cam_target: np.ndarray, tf: TargetFrame, scale: float) -> np.ndarray:
    return T_cam_target @ T_trans(tf.c * scale)


def net_to_target(T_cam_net: np.ndarray, tf: TargetFrame, scale: float) -> np.ndarray:
    return T_cam_net @ T_trans(-tf.c * scale)


def ray_rotation(t: np.ndarray) -> np.ndarray:
    """Minimal rotation taking the optical axis e_z onto the viewing ray of ``t``."""
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(t, float) / max(np.linalg.norm(t), 1e-12)
    v = np.cross(z, d)
    s, c = np.linalg.norm(v), float(z @ d)
    if s < 1e-12:
        return np.eye(3)
    k = v / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def allocentric(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return ray_rotation(t).T @ R


def canonical_index(R: np.ndarray, t: np.ndarray, R_ref: np.ndarray, rots: np.ndarray) -> int:
    """Symmetry k whose equivalent ``R @ S_k`` is closest (in the viewing-ray frame) to R_ref."""
    A = allocentric(R, t)
    tr = np.einsum("ij,kji->k", R_ref.T @ A, rots)  # trace(R_ref^T A S_k)
    return int(np.argmax(tr))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def estimate_reference(Rs: np.ndarray, ts: np.ndarray, rots: np.ndarray, iters: int = 10) -> np.ndarray:
    """Chordal mean of the training orientations on the quotient SO(3)/G (in the viewing-ray
    frame): a reference orientation at the centre of the data, so few samples sit near a
    canonicalisation boundary where the corner identities switch."""
    A = np.stack([allocentric(R, t) for R, t in zip(Rs, ts)])
    R_ref = A[0]
    for _ in range(iters):
        tr = np.einsum("nij,kji->nk", R_ref.T @ A, rots)
        k = tr.argmax(1)
        R_new = project_to_so3((A @ rots[k]).mean(0))
        if np.allclose(R_new, R_ref, atol=1e-9):
            break
        R_ref = R_new
    return R_ref


def canonicalize(T_cam_net: np.ndarray, R_ref: np.ndarray, rots: np.ndarray) -> np.ndarray:
    """The symmetric equivalent of ``T_cam_net`` closest to the reference orientation."""
    k = canonical_index(T_cam_net[:3, :3], T_cam_net[:3, 3], R_ref, rots)
    S = np.eye(4)
    S[:3, :3] = rots[k]
    return T_cam_net @ S


# --------------------------------------------------------------------------- quaternions
# SPEED+ convention (see upstream utils/postprocess.py): q = [qw, qx, qy, qz] of R_cam_body
# (scipy active rotation), and upstream's quat2dcm(q) = R_cam_body^T, which is what the
# EfficientPose head regresses.


def R_to_q_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


# --------------------------------------------------------------------------- config


def build_cfg(vcam: VirtualCamera, data_root: Path, yaml_file: Path | None = None, opts: list | None = None):
    """Upstream's yacs defaults + ``configs/spnv2_phi3.yaml`` + our dataset paths."""
    ensure_upstream_on_path()
    from config import cfg as _cfg  # upstream core/config
    from config.efficientdet import EFFICIENTDET_EXTRAS

    cfg = _cfg.clone()
    cfg.defrost()
    cfg.merge_from_file(str(yaml_file or BASELINE_DIR / "configs" / "spnv2_phi3.yaml"))
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.EXTRA = EFFICIENTDET_EXTRAS[cfg.MODEL.BACKBONE.NAME]
    name = f"speedplus_{SENSOR}"
    cfg.DATASET.ROOT = str(data_root)
    cfg.DATASET.DATANAME = name
    cfg.DATASET.CAMERA = str(Path(data_root) / name / "camera.json")
    cfg.DATASET.KEYPOINTS = str(Path(data_root) / name / "keypoints.mat")
    cfg.DATASET.CADMODEL = str(Path(data_root) / name / "model.ply")
    cfg.DATASET.IMAGE_SIZE = [vcam.width, vcam.height]  # the virtual camera *is* the "original" image
    cfg.DATASET.INPUT_SIZE = [vcam.width, vcam.height]
    cfg.freeze()
    return cfg


def write_camera_files(folder: Path, vcam: VirtualCamera, tf: TargetFrame) -> None:
    """SPEED+-style camera.json, keypoint .mat ([3, 8] 'tango3Dpoints', unit scale, net frame)
    and a .ply (mm) model, which upstream's EfficientPose head loads at build time."""
    from scipy.io import savemat

    folder.mkdir(parents=True, exist_ok=True)
    K = vcam.Km
    # load_camera_intrinsics computes horizontalFOV = 2 atan(ppx*Nu/2 / fx): use pixel units.
    cam = {"Nu": vcam.width, "Nv": vcam.height, "ppx": 1.0, "ppy": 1.0, "fx": K[0, 0], "fy": K[1, 1],
           "ccx": K[0, 2], "ccy": K[1, 2], "cameraMatrix": K.tolist(), "distCoeffs": [0.0] * 5}
    (folder / "camera.json").write_text(json.dumps(cam, indent=2))
    savemat(folder / "keypoints.mat", {"tango3Dpoints": tf.kp_unit.T.astype(np.float64)})
    v = (tf.vertices - tf.c) * 1000.0 if len(tf.vertices) else tf.kp_unit * 1000.0
    with open(folder / "model.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
                "property float z\nend_header\n")
        np.savetxt(f, v, fmt="%.4f")
