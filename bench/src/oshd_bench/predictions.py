"""File formats exchanged between baselines and the evaluator.

Everything lives under ``results/`` (see ``configs/bench.yaml``)::

    results/predictions/<modality>/<method>/<setting>/<domain>_<split>.parquet   # 6-DoF poses
    results/detections/<sensor>/<detector>/<setting>/<domain>_<split>.parquet    # 2D boxes/masks
    results/replay/<modality>.parquet                                            # policy replay
    results/flow/<method>/<setting>/<domain>_<split>.parquet                     # flow stats

Pose prediction columns: trajectory, sensor, t_ns, object, T (16 floats, row-major
T_sensor_model in the CAD model frame, metres), score, runtime_ms. One row per sample the
method produced a pose for; samples with no row count as failures.

``runtime_ms`` is the wall-clock inference time per sample (including any detector / frame
reconstruction stage the method needs), measured on the machine named in ``hardware``.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "prediction_path",
    "detection_path",
    "PoseWriter",
    "DetectionWriter",
    "load_predictions",
    "load_detections",
    "hardware",
]

POSE_COLUMNS = ["trajectory", "sensor", "t_ns", "object", "T", "score", "runtime_ms"]
DET_COLUMNS = ["trajectory", "sensor", "t_ns", "object", "box", "score", "mask_rle", "runtime_ms"]


def prediction_path(results: str | Path, modality: str, method: str, setting: str,
                    domain: str = "real", split: str = "test") -> Path:
    return Path(results) / "predictions" / modality / method / setting / f"{domain}_{split}.parquet"


def detection_path(results: str | Path, sensor: str, detector: str, setting: str,
                   domain: str = "real", split: str = "test") -> Path:
    return Path(results) / "detections" / sensor / detector / setting / f"{domain}_{split}.parquet"


def hardware() -> dict:
    info = {"host": platform.node(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


class _Writer:
    columns: list[str]

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.meta = {"hardware": hardware(), **(meta or {})}
        self.rows: list[dict] = []

    def __len__(self) -> int:
        return len(self.rows)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows, columns=self.columns).to_parquet(self.path, index=False)
        self.path.with_suffix(".json").write_text(json.dumps(self.meta, indent=2, default=str))
        return self.path


class PoseWriter(_Writer):
    columns = POSE_COLUMNS

    def add(self, trajectory: str, sensor: str, t_ns: int, object: str, T: np.ndarray,
            score: float = 1.0, runtime_ms: float = float("nan")) -> None:
        T = np.asarray(T, float).reshape(4, 4)
        if not np.all(np.isfinite(T)):
            return
        self.rows.append({"trajectory": trajectory, "sensor": sensor, "t_ns": int(t_ns), "object": object,
                          "T": T.reshape(-1).tolist(), "score": float(score), "runtime_ms": float(runtime_ms)})


class DetectionWriter(_Writer):
    """2D detections (xyxy box in pixels of the sensor image, optional COCO RLE mask)."""

    columns = DET_COLUMNS

    def add(self, trajectory: str, sensor: str, t_ns: int, object: str, box, score: float,
            mask: np.ndarray | None = None, runtime_ms: float = float("nan")) -> None:
        rle = None
        if mask is not None:
            from oshd_dataset.export.coco import mask_to_rle

            rle = json.dumps(mask_to_rle(mask))
        self.rows.append({"trajectory": trajectory, "sensor": sensor, "t_ns": int(t_ns), "object": object,
                          "box": [float(b) for b in box], "score": float(score), "mask_rle": rle,
                          "runtime_ms": float(runtime_ms)})


def load_predictions(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_detections(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def rle_to_mask(rle_json: str) -> np.ndarray:
    """Inverse of ``oshd_dataset.export.coco.mask_to_rle`` (uncompressed, column-major)."""
    rle = json.loads(rle_json)
    h, w = rle["size"]
    flat = np.zeros(h * w, bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape((h, w), order="F")
