import numpy as np

from matcha.pointmap.calibration import rectification_is_identity


def test_centered_equal_focal_calibration_keeps_source_pixels():
    intrinsics = np.array(
        [[1600.0, 0.0, 960.0], [0.0, 1600.0, 540.0], [0.0, 0.0, 1.0]]
    )
    assert rectification_is_identity(
        intrinsics,
        intrinsics.copy(),
        source_shape=(1080, 1920),
        target_shape=(1080, 1920),
    )


def test_off_center_principal_point_requires_writing_rectified_image():
    source = np.array(
        [[1600.0, 0.0, 900.0], [0.0, 1600.0, 520.0], [0.0, 0.0, 1.0]]
    )
    target = np.array(
        [[1600.0, 0.0, 900.0], [0.0, 1600.0, 520.0], [0.0, 0.0, 1.0]]
    )
    assert not rectification_is_identity(
        source,
        target,
        source_shape=(1080, 1920),
        target_shape=(1040, 1800),
    )


def test_changed_focal_requires_writing_rectified_image_even_at_same_shape():
    source = np.eye(3)
    target = np.eye(3)
    target[0, 0] = 1.1
    assert not rectification_is_identity(
        source,
        target,
        source_shape=(100, 100),
        target_shape=(100, 100),
    )
