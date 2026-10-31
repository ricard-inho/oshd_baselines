"""Shared pieces: per-sensor settings, the model, sample lists, image loading and GT masks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from oshd_bench import load_config, open_dataset, repo_root, results_dir
from PIL import Image

HERE = Path(__file__).resolve().parents[2]          # baselines/detector
WEIGHTS = HERE / "weights"
CACHE = Path(os.environ.get("DETECTOR_CACHE", HERE / "cache"))
os.environ.setdefault("TORCH_HOME", str(WEIGHTS / "torch"))  # COCO init weights land in weights/

DETECTOR = "maskrcnn"
SENSORS = ("firefly", "realsense_color", "events")

#: network input (W, H) per sensor (None = native) and the torchvision transform's
#: min_size / max_size. Firefly is downscaled 2x (3072x2048 -> 1536x1024); outputs are
#: always mapped back to full-resolution sensor pixels.
SENSOR_CFG = {
    "firefly": {"input_size": (1536, 1024), "min_size": 1024, "max_size": 1536},
    "realsense_color": {"input_size": None, "min_size": 800, "max_size": 1333},   # 640x480, 1.67x up
    "events": {"input_size": None, "min_size": 720, "max_size": 1280},           # E2VID frames 1280x720
}


def bench_cfg() -> dict:
    return load_config(repo_root() / "configs" / "bench.yaml")


def object_names(cfg: dict | None = None) -> list[str]:
    return list((cfg or bench_cfg()).get("objects", ["cubesat"]))


def checkpoint_path(sensor: str, setting: str) -> Path:
    return WEIGHTS / sensor / f"{setting}.pt"


# ------------------------------------------------------------------ model


def coco_init_weights() -> str:
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

    w = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    w.get_state_dict(progress=True, check_hash=True)
    return w.url


def build_model(sensor: str, num_classes: int, coco_init: bool = True):
    """torchvision Mask R-CNN R50-FPN v2 with fresh box / mask heads for ``num_classes`` (+bg)."""
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    c = SENSOR_CFG[sensor]
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if coco_init else None,
                                 min_size=c["min_size"], max_size=c["max_size"])
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,
                                                  num_classes + 1)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, num_classes + 1)
    return m


def load_checkpoint(sensor: str, setting: str, device, path: Path | None = None):
    path = Path(path or checkpoint_path(sensor, setting))
    if not path.exists():
        raise SystemExit(f"{path} missing: run `python -m detector.train --sensor {sensor} --setting {setting}`")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(sensor, len(ck["classes"]), coco_init=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


# ------------------------------------------------------------- samples


def sample_list(sensor: str, domain: str, split: str, with_pose: bool, limit: int | None = None) -> pd.DataFrame:
    """The GT sample list. ``with_pose=False`` (inference) never reads the ``T`` column."""
    p = results_dir() / "gt" / sensor / f"{domain}_{split}.parquet"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `uv run --project bench oshd-bench gt`")
    cols = ["trajectory", "index", "t_ns", "object"] + (["T"] if with_pose else [])
    df = pd.read_parquet(p, columns=cols).sort_values(["trajectory", "t_ns"], kind="stable")
    if limit:
        keep = df[["trajectory", "t_ns"]].drop_duplicates().head(limit)
        df = df.merge(keep, on=["trajectory", "t_ns"])
    return df.reset_index(drop=True)


def e2vid_frame_path(domain: str, split: str, trajectory: str, t_ns: int) -> Path:
    """Contract path written by baselines/e2vid."""
    return results_dir() / "e2vid" / f"{domain}_{split}" / trajectory / f"{int(t_ns)}.png"


def sensor_size(ds, sensor: str, trajectory: str) -> tuple[int, int]:
    if sensor == "events":
        from oshd_dataset.events import EVENT_SENSOR_SIZE

        return EVENT_SENSOR_SIZE[1], EVENT_SENSOR_SIZE[0]
    return ds[trajectory].calibration.intrinsics(sensor).size


def load_image(ds, sensor: str, domain: str, split: str, trajectory: str, index: int, t_ns: int) -> np.ndarray:
    """Full-resolution (H, W, 3) uint8 image of a sample (E2VID reconstruction for events)."""
    if sensor == "events":
        p = e2vid_frame_path(domain, split, trajectory, t_ns)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run baselines/e2vid (reconstruct --domain {domain} --split {split})")
        g = np.asarray(Image.open(p).convert("L"))
        return np.repeat(g[..., None], 3, -1)
    return ds[trajectory].frame(sensor, int(index))


def to_input(img: np.ndarray, sensor: str) -> np.ndarray:
    size = SENSOR_CFG[sensor]["input_size"]
    if size is None or (img.shape[1], img.shape[0]) == tuple(size):
        return img
    return np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))


def gt_mask(ds, sensor: str, trajectory: str, t_ns: int, obj_name: str, T: np.ndarray,
            size: tuple[int, int]) -> np.ndarray:
    """Rendered (CAD + GT pose, distorted projection) silhouette at image size ``size`` (W, H), cached."""
    from oshd_dataset.labels import render_mask

    p = CACHE / "masks" / sensor / f"{size[0]}x{size[1]}" / trajectory / f"{int(t_ns)}_{obj_name}.png"
    if p.exists():
        return np.asarray(Image.open(p)) > 0
    intr = ds[trajectory].calibration.intrinsics(sensor)
    if intr.size != tuple(size):
        intr = intr.resized(*size)
    m = render_mask(ds.objects[obj_name], np.asarray(T, float).reshape(4, 4), intr)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m.astype(np.uint8) * 255).save(p, optimize=False, compress_level=1)
    return m


class _DS:
    """Lazily opened dataset handle (one per DataLoader worker)."""

    _ds = None

    @classmethod
    def get(cls):
        if cls._ds is None:
            cls._ds = open_dataset()
        return cls._ds
