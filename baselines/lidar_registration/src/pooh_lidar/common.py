"""Shared constants and helpers (cube geometry, dataset access, dev-only GT helpers)."""

from __future__ import annotations

import numpy as np

MODALITY = "lidar"
SENSOR = "lidar"
OBJECT = "cubesat"
# CubeSat_U1.obj: body y 0..0.1, x,z +-0.05 (metres, Y-up); marker holder y 0.105..0.134
CUBE_CENTER = np.array([0.0, 0.05, 0.0])
CUBE_HALF = 0.05


def box_sdf(p_model: np.ndarray, center=CUBE_CENTER, half=CUBE_HALF) -> np.ndarray:
    """Signed distance of model-frame points to the cube body (negative inside)."""
    q = np.abs(p_model - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(-1), 0.0)
    return outside + inside


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


def inv(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def model_cloud(obj, n: int = 20000, body_only: bool = False, seed: int = 0) -> np.ndarray:
    """Area-uniform CAD surface sample (model frame). ``body_only`` drops the marker holder."""
    from pooh_bench.flow_gt import surface_points

    p = np.asarray(surface_points(obj, max(n * 2, 200_000), seed))
    if body_only:
        p = p[p[:, 1] <= 0.1 + 1e-4]
    rng = np.random.default_rng(seed)
    return p[rng.choice(len(p), size=min(n, len(p)), replace=False)]


def load_samples(results, domain: str, split: str):
    """GT sample list *without* the T column (safe for inference code)."""
    import pandas as pd
    from pooh_bench import gt_path

    p = gt_path(results, SENSOR, domain, split)
    import pyarrow.parquet as pq

    cols = [c for c in pq.read_schema(p).names if c != "T"]
    return pd.read_parquet(p, columns=cols)


def load_samples_with_gt(results, domain: str, split: str):
    """Training / diagnostics only (reads T)."""
    import pandas as pd
    from pooh_bench import gt_path

    return pd.read_parquet(gt_path(results, SENSOR, domain, split))
