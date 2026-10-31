"""Shared pieces of the SPNv2 baseline: paths, the virtual network camera, the cube's
symmetry canonicalisation and the yacs config built from upstream's defaults.

Frames used throughout:

* ``model``: the CAD frame of ``models/CubeSat_U1.obj`` (what oshd-bench poses are in).
* ``net``:   the CAD frame translated to the cube-body centre ``c`` (the oshd-bench symmetry
  centre). ``T_cam_net = T_cam_model @ Trans(c)``. The 24 cube symmetries are pure rotations
  about the ``net`` origin, so the regressed translation is symmetry invariant.
* ``cam``:   the sensor optical frame. The network sees images re-sampled into a fixed
  *virtual* pinhole camera (undistorted, resized, padded) that shares the sensor's optical
  frame, so a pose in the virtual camera is a pose in the sensor frame.
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
METHOD = "spnv2"
OBJECT = "cubesat"

# Table I rows: sensor -> modality folder
MODALITY = {"firefly": "rgb_firefly", "realsense_color": "rgb_realsense"}

# Network input per sensor: (W, H) of the virtual image and the scale from the raw image.
# W and H must be divisible by 128 (P7 stride). Extra rows are black padding (cy shifted).
#   firefly   3072x2048 -> x0.25 -> 768x512   (upstream's SPEED+ input size)
#   realsense  640x480  -> x1.2  -> 768x576, padded to 768x640
INPUT = {"firefly": ((768, 512), 0.25), "realsense_color": ((768, 640), 1.2)}


def ensure_upstream_on_path() -> None:
    core = str(UPSTREAM_DIR / "core")
    if not (UPSTREAM_DIR / "core").is_dir():
        raise SystemExit(f"upstream SPNv2 not found at {UPSTREAM_DIR}; run baselines/spnv2/setup.sh")
    if core not in sys.path:
        sys.path.insert(0, core)
    # torchvision's ImageNet EfficientNet weights go to weights/torch (setup.sh pre-fetches them)
    os.environ.setdefault("TORCH_HOME", str(BASELINE_DIR / "weights" / "torch"))


def work_dir() -> Path:
    """Converted training data and trained checkpoints: ``$SPNV2_WORK`` or
    ``<repo>/work/spnv2/<dataset folder name>`` (so the dev dataset never mixes with the real one)."""
    if os.environ.get("SPNV2_WORK"):
        return Path(os.environ["SPNV2_WORK"])
    from oshd_bench import open_dataset, repo_root

    return repo_root() / "work" / "spnv2" / Path(open_dataset().root).resolve().name


def run_dir(sensor: str, setting: str) -> Path:
    return work_dir() / "runs" / sensor / setting


# --------------------------------------------------------------------------- virtual camera


@dataclass
class VirtualCamera:
    """Fixed pinhole camera the network is trained and evaluated in (one per sensor)."""

    sensor: str
    width: int
    height: int
    K: list  # 3x3 row-major

    @property
    def Km(self) -> np.ndarray:
        return np.asarray(self.K, float).reshape(3, 3)

    @classmethod
    def from_intrinsics(cls, sensor: str, intr) -> VirtualCamera:
        (w, h), s = INPUT[sensor]
        sw, sh = round(intr.width * s), round(intr.height * s)
        r = intr.resized(sw, sh)
        K = r.K.copy()
        K[0, 2] += (w - sw) / 2.0
        K[1, 2] += (h - sh) / 2.0
        return cls(sensor, w, h, K.tolist())

    def intrinsics(self, scale: float = 1.0):
        """oshd_dataset CameraIntrinsics of the virtual image (optionally rescaled)."""
        from oshd_dataset import CameraIntrinsics

        ci = CameraIntrinsics(self.sensor + "_virtual", self.width, self.height, self.Km, np.zeros(5), "none")
        return ci if scale == 1.0 else ci.resized(round(self.width * scale), round(self.height * scale))

    def to_dict(self) -> dict:
        return asdict(self)


class Warper:
    """Raw (distorted) sensor frames -> virtual-camera images, cached per source calibration."""

    def __init__(self, vcam: VirtualCamera):
        self.vcam = vcam
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
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return cv2.remap(image, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def to_network_image(rgb: np.ndarray, gray: bool = True) -> np.ndarray:
    """Upstream preprocessing: SPEED+ images are grayscale, written once and re-read as 3
    identical channels (``tools/preprocess.py`` + ``cv2.imread(IMREAD_COLOR)``)."""
    if gray:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.repeat(g[:, :, None], 3, axis=2)
    return rgb[:, :, ::-1].copy()  # upstream reads BGR


# --------------------------------------------------------------------------- symmetry


def cube_frame(obj, sym_cfg: dict | None = None):
    """(center c (3,), keypoints in the net frame (K, 3), symmetry rotations (24, 3, 3))."""
    from oshd_bench import symmetry_for

    sym = symmetry_for(obj, sym_cfg or {"type": "cube", "center": "auto"})
    kp = np.asarray(obj.click_points, float) - sym.center
    return sym.center.copy(), kp, sym.rotations.copy()


def T_trans(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = v
    return T


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
    """Chordal mean of the training orientations on the quotient SO(3)/cube (in the viewing-
    ray frame): a reference orientation at the centre of the data, so few samples sit near a
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


def canonicalize(T_cam_model: np.ndarray, c: np.ndarray, R_ref: np.ndarray, rots: np.ndarray) -> np.ndarray:
    """T_cam_net of the symmetric equivalent closest to the reference orientation."""
    T = T_cam_model @ T_trans(c)
    k = canonical_index(T[:3, :3], T[:3, 3], R_ref, rots)
    S = np.eye(4)
    S[:3, :3] = rots[k]
    return T @ S


def net_to_model(T_cam_net: np.ndarray, c: np.ndarray) -> np.ndarray:
    return T_cam_net @ T_trans(-np.asarray(c))


# --------------------------------------------------------------------------- quaternions
# SPEED+ convention (see upstream utils/postprocess.py): q = [qw, qx, qy, qz] of R_cam_body
# (scipy active rotation), and upstream's quat2dcm(q) = R_cam_body^T, which is what the
# EfficientPose head regresses.


def R_to_q_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]])


# --------------------------------------------------------------------------- config


def build_cfg(sensor: str, vcam: VirtualCamera, data_root: Path, yaml_file: Path | None = None,
              opts: list | None = None):
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
    name = f"speedplus_{sensor}"
    cfg.DATASET.ROOT = str(data_root)
    cfg.DATASET.DATANAME = name
    cfg.DATASET.CAMERA = str(Path(data_root) / name / "camera.json")
    cfg.DATASET.KEYPOINTS = str(Path(data_root) / name / "cube_keypoints.mat")
    cfg.DATASET.CADMODEL = str(Path(data_root) / name / "cube.ply")
    cfg.DATASET.IMAGE_SIZE = [vcam.width, vcam.height]  # the virtual camera *is* the "original" image
    cfg.DATASET.INPUT_SIZE = [vcam.width, vcam.height]
    cfg.freeze()
    return cfg


def write_camera_files(folder: Path, vcam: VirtualCamera, kp_net: np.ndarray, obj, c: np.ndarray) -> None:
    """SPEED+-style camera.json, keypoint .mat ([3, K] 'tango3Dpoints') and a .ply (mm) model."""
    from scipy.io import savemat

    folder.mkdir(parents=True, exist_ok=True)
    K = vcam.Km
    # load_camera_intrinsics computes horizontalFOV = 2 atan(ppx*Nu/2 / fx): use pixel units.
    cam = {"Nu": vcam.width, "Nv": vcam.height, "ppx": 1.0, "ppy": 1.0, "fx": K[0, 0], "fy": K[1, 1],
           "ccx": K[0, 2], "ccy": K[1, 2], "cameraMatrix": K.tolist(), "distCoeffs": [0.0] * 5}
    (folder / "camera.json").write_text(json.dumps(cam, indent=2))
    savemat(folder / "cube_keypoints.mat", {"tango3Dpoints": kp_net.T.astype(np.float64)})
    v = (np.asarray(obj.vertices) - c) * 1000.0
    rng = np.random.default_rng(0)
    v = v[rng.choice(len(v), size=min(len(v), 5000), replace=False)]
    with open(folder / "cube.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
                "property float z\nend_header\n")
        np.savetxt(f, v, fmt="%.4f")
