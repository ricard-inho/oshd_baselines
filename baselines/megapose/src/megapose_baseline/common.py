"""Shared helpers: modality table, sample lists (without GT), detections, image sources, camera prep."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE_DIR = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = BASELINE_DIR / "weights"

# modality (Table I row) -> sensor frame, method key, default MegaPose input size (W, H) or None = native
MODALITIES = {
    "rgb_firefly": {"sensor": "firefly", "method": "megapose", "size": (1152, 768)},
    "rgb_realsense": {"sensor": "realsense_color", "method": "megapose", "size": None},
    "events": {"sensor": "events", "method": "e2vid_megapose", "size": None},
}

# Columns inference may read from the GT sample list. The GT pose column ``T`` is never read.
SAMPLE_COLUMNS = ["trajectory", "domain", "sensor", "index", "t_ns", "object"]


def set_happypose_env() -> None:
    os.environ.setdefault("HAPPYPOSE_DATA_DIR", str(WEIGHTS_DIR))
    Path(os.environ["HAPPYPOSE_DATA_DIR"]).mkdir(parents=True, exist_ok=True)


def load_samples(results: Path, sensor: str, domain: str, split: str) -> pd.DataFrame:
    """The sample list of ``pooh-bench gt`` without the ground-truth columns."""
    from pooh_bench import gt_path

    p = gt_path(results, sensor, domain, split)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `uv run --project bench pooh-bench gt` first")
    import pyarrow.parquet as pq

    cols = [c for c in SAMPLE_COLUMNS if c in pq.read_schema(p).names]
    return pd.read_parquet(p, columns=cols)  # never 'T'


def load_detections(results: Path, sensor: str, detector: str, setting: str, domain: str,
                    split: str) -> tuple[pd.DataFrame, Path]:
    from pooh_bench import detection_path

    det_setting = "GT" if detector == "oracle" else setting
    p = detection_path(results, sensor, detector, det_setting, domain, split)
    if not p.exists():
        hint = (f"uv run --project bench pooh-bench oracle-detections --sensor {sensor} --split {split} "
                f"--domain {domain}") if detector == "oracle" else "run baselines/detector predict first"
        raise FileNotFoundError(f"{p} missing ({hint})")
    return pd.read_parquet(p), p


def group_detections(det: pd.DataFrame, max_dets: int) -> dict[tuple, list]:
    """(trajectory, t_ns, object) -> up to ``max_dets`` detection rows, best score first."""
    out: dict[tuple, list] = {}
    for r in det.sort_values("score", ascending=False).itertuples(index=False):
        k = (r.trajectory, int(r.t_ns), r.object)
        lst = out.setdefault(k, [])
        if len(lst) < max_dets:
            lst.append(r)
    return out


@dataclass
class Prepared:
    rgb: np.ndarray  # (H, W, 3) uint8, undistorted + resized
    intr: object  # pooh_dataset CameraIntrinsics of ``rgb`` (pinhole, D = 0)


class CameraPrep:
    """Raw sensor image -> undistorted, resized pinhole image (the pose stays in the sensor frame)."""

    def __init__(self, intr, size: tuple[int, int] | None):
        from pooh_dataset.imaging import CameraView

        self.src = intr
        self.view = CameraView(intr, undistort=True, size=size)
        self.intr = self.view.intrinsics

    def image(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, -1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if (img.shape[1], img.shape[0]) != self.src.size:
            raise ValueError(f"image {img.shape[1]}x{img.shape[0]} does not match intrinsics {self.src.size}")
        return self.view(img)

    def box(self, det) -> np.ndarray | None:
        """Detection (in raw, distorted sensor pixels) -> xyxy box in the prepared image.

        The mask (or, without one, the filled box) is warped with the same map as the image.
        """
        mask = None
        if getattr(det, "mask_rle", None):
            from pooh_bench.predictions import rle_to_mask

            mask = rle_to_mask(det.mask_rle)
        if mask is None or not mask.any():
            x0, y0, x1, y1 = (float(v) for v in det.box)
            mask = np.zeros((self.src.height, self.src.width), bool)
            mask[max(int(np.floor(y0)), 0):int(np.ceil(y1)), max(int(np.floor(x0)), 0):int(np.ceil(x1))] = True
        m = self.view(mask.astype(np.uint8), nearest=True) > 0
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return None
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)


class E2VIDFrames:
    """Reconstructed frames at ``<root>/<domain>_<split>/<trajectory>/<t_ns>.png``.

    If an index parquet with a ``runtime_ms`` column sits next to the frames
    (``<root>/<domain>_<split>.parquet`` or ``<root>/<domain>_<split>/index.parquet``), its
    per-frame reconstruction time is added to the pose runtime.
    """

    def __init__(self, root: Path, domain: str, split: str):
        self.dir = Path(root) / f"{domain}_{split}"
        if not self.dir.exists():
            raise FileNotFoundError(f"{self.dir} missing: run baselines/e2vid (or megapose-events-standin)")
        self.runtime: dict[tuple, float] = {}
        for p in [Path(root) / f"{domain}_{split}.parquet", self.dir / "index.parquet"]:
            if p.exists():
                idx = pd.read_parquet(p)
                if "runtime_ms" in idx.columns:
                    self.runtime = {(r.trajectory, int(r.t_ns)): float(r.runtime_ms)
                                    for r in idx.itertuples(index=False)}
                break

    def get(self, trajectory: str, t_ns: int) -> tuple[np.ndarray | None, float]:
        from PIL import Image

        p = self.dir / trajectory / f"{int(t_ns)}.png"
        if not p.exists():
            return None, 0.0
        img = np.asarray(Image.open(p).convert("L"))
        rt = self.runtime.get((trajectory, int(t_ns)), 0.0)
        return img, (rt if np.isfinite(rt) else 0.0)
