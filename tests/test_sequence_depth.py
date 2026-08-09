import numpy as np
import torch

from outdoor.sequence_depth import (
    SequenceDepthCamera,
    SequenceDepthView,
    solve_sequence_depth_scales,
)
from scripts.regularize_sequence_dav2_foliage import _apply_primitive_scales


def _camera(index: int) -> SequenceDepthCamera:
    center = np.asarray([0.08 * index, 0.0, 0.0], dtype=np.float64)
    return SequenceDepthCamera(
        camera_id=100 + index,
        sequence="seq1",
        frame=index,
        rotation=np.eye(3, dtype=np.float64),
        translation=-center,
        center=center,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=180.0,
        width=640,
        height=360,
    )


def _view(index: int, wrong_scale: float) -> SequenceDepthView:
    camera = _camera(index)
    x, y = np.meshgrid(
        np.linspace(-2.0, 2.0, 32),
        np.linspace(-1.0, 1.0, 20),
    )
    true = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 10.0)])
    points = camera.center + wrong_scale * (true - camera.center)
    camera_points = points + camera.translation
    u = (
        camera.fx * camera_points[:, 0] / camera_points[:, 2]
        + camera.cx
        + 0.5
    ) / camera.width
    v = (
        camera.fy * camera_points[:, 1] / camera_points[:, 2]
        + camera.cy
        + 0.5
    ) / camera.height
    return SequenceDepthView(
        camera=camera,
        points=points,
        normalized_uv=np.column_stack([u, v]),
        relative_rmse=10.0,
    )


def test_reprojection_graph_recovers_relative_metric_scales():
    wrong = np.asarray([1.0, 0.5, 2.0], dtype=np.float64)
    views = [_view(index, value) for index, value in enumerate(wrong)]
    scales, audit = solve_sequence_depth_scales(
        views,
        maximum_frame_gap=2,
        maximum_pixel_distance=8.0,
        minimum_matches=64,
        iterations=3,
    )
    recovered = np.asarray(
        [scales[view.camera.camera_id] for view in views]
    )
    # The weak global gauge has product one, as do these synthetic errors.
    np.testing.assert_allclose(recovered, 1.0 / wrong, rtol=0.08, atol=0.04)
    corrected_depth = wrong * recovered * 10.0
    assert corrected_depth.max() - corrected_depth.min() < 0.25
    before = audit["optimization"][0][
        "absolute_log_residual_before_quantiles"
    ][2]
    after = audit["final_absolute_log_depth_residual_quantiles"][2]
    assert after < 0.2 * before


def test_primitive_repair_writes_observation_depth_back():
    camera = _camera(0)
    payload = {
        "initialization_source": torch.tensor([3], dtype=torch.int8),
        "observation_camera_ids": torch.tensor(
            [[camera.camera_id, -1]], dtype=torch.int32
        ),
        "observation_uv": torch.zeros(1, 2, 2),
        "observation_depth": torch.tensor([[4.0, float("nan")]]),
        "centers": torch.tensor([[0.0, 0.0, 4.0]]),
        "scales": torch.ones(1, 3),
        "position_covariance": torch.eye(3)[None],
        "scale_ceiling": torch.ones(1),
    }
    _, audit = _apply_primitive_scales(
        payload,
        {camera.camera_id: 2.0},
        {camera.camera_id: camera},
    )
    assert audit["changed_primitive_rows"] == 1
    assert payload["observation_depth"][0, 0].item() == 8.0
    assert payload["centers"][0, 2].item() == 8.0
    assert payload["scales"][0, 0].item() == 2.0
