"""pooh-bench: shared ground truth, file formats and metrics for the POOH-D baselines."""

__version__ = "0.1.0"

from .evaluate import Evaluator, gt_path, load_config, open_dataset, repo_root, results_dir
from .gt import build_gt, gt_poses, load_gt, pose_in_sensor
from .metrics import flow_stats, pose_errors, speed_score
from .predictions import DetectionWriter, PoseWriter, detection_path, prediction_path
from .symmetry import symmetry_for

__all__ = [
    "DetectionWriter",
    "Evaluator",
    "PoseWriter",
    "build_gt",
    "detection_path",
    "flow_stats",
    "gt_path",
    "gt_poses",
    "load_config",
    "load_gt",
    "open_dataset",
    "repo_root",
    "results_dir",
    "pose_errors",
    "pose_in_sensor",
    "prediction_path",
    "speed_score",
    "symmetry_for",
]
