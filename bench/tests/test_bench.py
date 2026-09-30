import numpy as np
import pytest
from pooh_dataset import ObjectModel
from pooh_dataset.geometry import rotvec_to_rotmat

from pooh_bench.flow_gt import target_flow
from pooh_bench.metrics import auc, flow_stats, pose_errors, speed_score
from pooh_bench.predictions import rle_to_mask
from pooh_bench.symmetry import cube_rotations, symmetry_for


@pytest.fixture
def cube():
    obj = ObjectModel.from_box("cube", (0.1, 0.1, 0.1))
    obj.vertices = obj.vertices + [0, 0.05, 0]  # origin on the bottom face, like the CAD
    return obj


def _pose(R=None, t=(0, 0, 1.0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.eye(3) if R is None else R, t
    return T


def test_cube_group():
    R = cube_rotations()
    assert len(R) == 24 and np.allclose(np.linalg.det(R), 1)
    assert len({tuple(np.round(r).ravel()) for r in R}) == 24


def test_symmetric_pose_has_zero_error(cube):
    sym = symmetry_for(cube, {"type": "cube"})
    assert np.allclose(sym.center, [0, 0.05, 0])
    Tg = _pose(rotvec_to_rotmat(np.array([0.3, -0.2, 0.5])))
    Tp = Tg @ sym.transforms[7]  # 90/180 deg flip about the cube centre
    err = pose_errors(Tp[None], Tg[None], sym, cube.vertices)
    assert err.e_t[0] < 1e-9 and err.e_q[0] < 1e-6 and err.add[0] < 1e-9
    none = pose_errors(Tp[None], Tg[None], symmetry_for(cube, {"type": "none"}), cube.vertices)
    assert none.e_q[0] > 45


def test_known_errors_and_speed(cube):
    sym = symmetry_for(cube, {"type": "cube"})
    Tg = _pose(t=(0, 0, 2.0))
    Tp = _pose(rotvec_to_rotmat(np.array([0, 0, np.radians(10)])), (0.02, 0, 2.0))
    err = pose_errors(Tp[None], Tg[None], sym, cube.vertices)
    # rotation about the model origin moves the centre (0, .05, 0) sideways too
    c_shift = np.linalg.norm(rotvec_to_rotmat(np.array([0, 0, np.radians(10)])) @ [0, 0.05, 0] - [0, 0.05, 0])
    assert err.e_q[0] == pytest.approx(10, abs=1e-6)
    assert err.e_t[0] == pytest.approx(np.linalg.norm([0.02 - c_shift * np.cos(np.radians(5)), -c_shift * np.sin(np.radians(5)), 0]), abs=1e-6)
    s = speed_score(err)
    assert s[0] == pytest.approx(err.e_t[0] / err.range_m[0] + np.radians(10))
    tiny = pose_errors(_pose(t=(0, 0, 2.001))[None], Tg[None], sym, cube.vertices)
    assert speed_score(tiny)[0] == 0  # below the SPEED+ precision thresholds


def test_auc_counts_missing_as_failures():
    assert auc(np.zeros(10), 10, 0.1) == pytest.approx(1, abs=1e-3)
    assert auc(np.zeros(5), 10, 0.1) == pytest.approx(0.5, abs=1e-3)
    assert auc(np.array([]), 0, 0.1) != auc(np.array([]), 0, 0.1)  # nan


def test_flow_stats():
    gt = np.zeros((4, 4, 2), np.float32)
    gt[..., 0] = 2
    pred = np.zeros_like(gt)
    s = flow_stats(pred, gt, np.ones((4, 4), bool)).summary()
    assert s["EPE"] == pytest.approx(2) and s["1PE"] == 100 and s["3PE"] == 0
    assert s["AE"] == pytest.approx(np.degrees(np.arccos(1 / np.sqrt(5))), abs=1e-4)


def test_rle_roundtrip():
    import json

    from pooh_dataset.export.coco import mask_to_rle

    m = np.random.default_rng(0).random((7, 9)) > 0.5
    assert np.array_equal(rle_to_mask(json.dumps(mask_to_rle(m))), m)


def test_target_flow_translation(cube):
    from pooh_dataset import CameraIntrinsics

    intr = CameraIntrinsics("c", 200, 200, np.array([[200, 0, 100], [0, 200, 100], [0, 0, 1.0]]), np.zeros(5))
    T0 = _pose(t=(0, -0.05, 1.0))
    T1 = _pose(t=(0.01, -0.05, 1.0))  # 1 cm right at ~1 m -> ~2 px
    flow, valid = target_flow(cube, T0, T1, intr, n_points=200_000)
    assert valid.sum() > 300
    assert np.median(flow[valid, 0]) == pytest.approx(200 * 0.01 / 0.95, rel=0.05)
    assert np.abs(flow[valid, 1]).max() < 0.2
