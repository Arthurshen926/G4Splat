import numpy as np

from scripts.audit_chart_alignment_target import (
    pointmap_target_depth,
    relative_error_metrics,
)


def test_pointmap_target_depth_uses_calibrated_camera_z_and_scale():
    points = np.array(
        [[[1.0, 0.0, 4.0], [1.0, 0.0, 6.0]]], dtype=np.float32
    )
    camera_to_world = np.eye(4)
    camera_to_world[2, 3] = 1.0

    depth = pointmap_target_depth(points, camera_to_world, scale_factor=2.0)

    # The inverse pose removes the camera's +z translation before scaling.
    assert np.allclose(depth, [[6.0, 10.0]])


def test_relative_error_metrics_uses_only_common_positive_support():
    estimate = np.array([[2.0, 3.0, np.nan, 0.0]])
    target = np.array([[2.0, 2.0, 2.0, 2.0]])
    support = np.array([[True, True, True, True]])

    metrics = relative_error_metrics(estimate, target, support)

    assert metrics["valid_pixels"] == 2
    assert metrics["relative_median"] == 0.25
    assert metrics["relative_p90"] == 0.45
    assert metrics["gt25_fraction"] == 0.5
