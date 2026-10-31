"""Pose and flow metrics.

Pose metrics compare predicted and ground-truth ``T_sensor_model`` (4x4, metres):

* ``e_t``: distance between the predicted and true symmetry centre (m).
* ``e_q``: geodesic rotation error, minimised over the object's symmetry group (deg).
* SPEED score (ESA Kelvins / SPEED+): ``s_t + s_q`` with ``s_t = e_t / ||t_gt||`` and
  ``s_q = e_q`` in radians. SPEED+ zeroes errors below the ground-truth precision
  (``s_t < 2.173e-3``, ``e_q < 0.169 deg``); the thresholds are configurable.
* ADD(-S): mean model-point distance, minimised over the symmetry group, and its AUC over
  thresholds in ``[0, max_m]`` (PoseCNN / YCB-Video convention). A frame without a
  prediction counts as a failure in the AUC.

Flow metrics follow DSEC / MVSEC: EPE, % of pixels with EPE > 1 / 3 px, and the Middlebury
angular error between ``(u, v, 1)`` vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .symmetry import Symmetry

__all__ = [
    "PoseErrors",
    "pose_errors",
    "speed_score",
    "auc",
    "FlowStats",
    "flow_stats",
]


def _rot_angle_deg(R: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


@dataclass
class PoseErrors:
    e_t: np.ndarray  # (N,) m
    e_q: np.ndarray  # (N,) deg
    range_m: np.ndarray  # (N,) ||centre_gt||
    add: np.ndarray  # (N,) m


def pose_errors(T_pred: np.ndarray, T_gt: np.ndarray, sym: Symmetry, model_points: np.ndarray) -> PoseErrors:
    """Errors for N pose pairs ``(N, 4, 4)`` of one object."""
    T_pred = np.asarray(T_pred, float).reshape(-1, 4, 4)
    T_gt = np.asarray(T_gt, float).reshape(-1, 4, 4)
    c = sym.center
    c_pred = T_pred[:, :3, :3] @ c + T_pred[:, :3, 3]
    c_gt = T_gt[:, :3, :3] @ c + T_gt[:, :3, 3]
    e_t = np.linalg.norm(c_pred - c_gt, axis=-1)

    # All symmetric equivalents of the ground truth: (N, K, 4, 4)
    T_gt_sym = T_gt[:, None] @ sym.transforms[None]
    R_rel = np.swapaxes(T_pred[:, None, :3, :3], -1, -2) @ T_gt_sym[..., :3, :3]
    e_q = _rot_angle_deg(R_rel).min(axis=1)

    p = np.asarray(model_points, float)
    pred_pts = p @ T_pred[:, None, :3, :3].swapaxes(-1, -2) + T_pred[:, None, None, :3, 3]  # (N,1,P,3)
    gt_pts = p @ T_gt_sym[..., :3, :3].swapaxes(-1, -2) + T_gt_sym[..., None, :3, 3]  # (N,K,P,3)
    add = np.linalg.norm(pred_pts - gt_pts, axis=-1).mean(-1).min(-1)
    return PoseErrors(e_t, e_q, np.linalg.norm(c_gt, axis=-1), add)


def speed_score(err: PoseErrors, t_thresh: float = 2.173e-3, q_thresh_deg: float = 0.169) -> np.ndarray:
    """Per-frame SPEED score ``s_t + s_q`` (the reported number is its mean)."""
    s_t = err.e_t / np.maximum(err.range_m, 1e-9)
    s_q = np.radians(err.e_q)
    s_t = np.where(s_t < t_thresh, 0.0, s_t)
    s_q = np.where(err.e_q < q_thresh_deg, 0.0, s_q)
    return s_t + s_q


def auc(errors: np.ndarray, n_total: int, max_threshold: float, steps: int = 1000) -> float:
    """Area under the accuracy-vs-threshold curve, normalised to [0, 1].

    ``errors`` are the per-frame errors of the frames that have a prediction; the
    ``n_total - len(errors)`` frames without one count as failures.
    """
    if n_total == 0:
        return float("nan")
    th = np.linspace(0.0, max_threshold, steps)
    e = np.sort(np.asarray(errors, float))
    acc = np.searchsorted(e, th, side="right") / n_total
    return float(np.trapezoid(acc, th) / max_threshold) if hasattr(np, "trapezoid") else float(
        np.trapz(acc, th) / max_threshold
    )


@dataclass
class FlowStats:
    """Sums over evaluated pixels, so samples can be pooled pixel-weighted."""

    n: int
    sum_epe: float
    n_gt1: int
    n_gt3: int
    sum_ae_deg: float

    def __add__(self, o: FlowStats) -> FlowStats:
        return FlowStats(self.n + o.n, self.sum_epe + o.sum_epe, self.n_gt1 + o.n_gt1,
                         self.n_gt3 + o.n_gt3, self.sum_ae_deg + o.sum_ae_deg)

    def summary(self) -> dict:
        n = max(self.n, 1)
        return {"EPE": self.sum_epe / n, "1PE": 100 * self.n_gt1 / n, "3PE": 100 * self.n_gt3 / n,
                "AE": self.sum_ae_deg / n, "n_pixels": self.n}


def flow_stats(flow_pred: np.ndarray, flow_gt: np.ndarray, valid: np.ndarray) -> FlowStats:
    """``flow_*`` (H, W, 2) px, ``valid`` (H, W) bool: the pixels to evaluate on."""
    fp, fg = flow_pred[valid].astype(np.float64), flow_gt[valid].astype(np.float64)
    epe = np.linalg.norm(fp - fg, axis=-1)
    num = fp[:, 0] * fg[:, 0] + fp[:, 1] * fg[:, 1] + 1.0
    den = np.sqrt((fp**2).sum(-1) + 1.0) * np.sqrt((fg**2).sum(-1) + 1.0)
    ae = np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))
    return FlowStats(int(valid.sum()), float(epe.sum()), int((epe > 1).sum()), int((epe > 3).sum()),
                     float(ae.sum()))
