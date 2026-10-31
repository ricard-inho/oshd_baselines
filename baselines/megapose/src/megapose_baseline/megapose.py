"""Thin wrapper around happypose's MegaPose (coarse + refiner, RGB, single image, per-frame)."""

from __future__ import annotations

import time

import numpy as np

from .common import set_happypose_env

MODEL_NAME = "megapose-1.0-RGB"  # coarse-rgb-906902141 + refiner-rgb-653307694


class MegaPose:
    def __init__(self, label: str, mesh_path, n_refiner_iterations: int = 5, n_pose_hypotheses: int = 1,
                 so3_stride: int = 1, device: str = "cuda"):
        set_happypose_env()
        import torch
        from happypose.toolbox.datasets.object_dataset import (
            RigidObject,
            RigidObjectDataset,
        )
        from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model

        self.torch = torch
        self.device = torch.device(device)
        self.label = label
        ds = RigidObjectDataset([RigidObject(label=label, mesh_path=mesh_path, mesh_units="m")])
        self.est = load_named_model(MODEL_NAME, ds).to(self.device)
        if so3_stride > 1:
            self.est._SO3_grid = self.est._SO3_grid[::so3_stride]
        self.params = dict(NAMED_MODELS[MODEL_NAME]["inference_parameters"])
        self.params.update(n_refiner_iterations=n_refiner_iterations, n_pose_hypotheses=n_pose_hypotheses)
        self.info = {
            "model": MODEL_NAME,
            "coarse_run_id": NAMED_MODELS[MODEL_NAME]["coarse_run_id"],
            "refiner_run_id": NAMED_MODELS[MODEL_NAME]["refiner_run_id"],
            "SO3_grid_size": int(self.est._SO3_grid.shape[0]),
            **self.params,
        }

    def __call__(self, rgb: np.ndarray, K: np.ndarray, boxes: list[np.ndarray]):
        """-> list of (T_cam_model 4x4, score) per box, and the wall-clock time in ms."""
        import pandas as pd
        from happypose.toolbox.inference.types import ObservationTensor
        from happypose.toolbox.utils.tensor_collection import PandasTensorCollection

        torch = self.torch
        t0 = time.perf_counter()
        obs = ObservationTensor.from_numpy(rgb, None, K).to(self.device)
        infos = pd.DataFrame({"label": [self.label] * len(boxes), "batch_im_id": 0,
                              "instance_id": np.arange(len(boxes))})
        det = PandasTensorCollection(infos=infos, bboxes=torch.as_tensor(np.stack(boxes)).float()).to(self.device)
        with torch.inference_mode():
            out, extra = self.est.run_inference_pipeline(obs, detections=det, **self.params)
        self.last_timing = extra.get("timing_str", "")
        out = out.cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        poses = out.poses.numpy()
        inf = out.infos
        score = inf["pose_score"].to_numpy() if "pose_score" in inf else np.ones(len(inf))
        order = inf["instance_id"].to_numpy()
        res = [None] * len(boxes)
        for k, i in enumerate(order):
            res[int(i)] = (poses[k].astype(np.float64), float(score[k]))
        return res, ms
