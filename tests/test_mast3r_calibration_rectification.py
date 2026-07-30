import numpy as np

from matcha.pointmap.calibration import (
    calibrated_camera_to_world_for_output,
    rectification_is_identity,
    retarget_pointmap_camera_rays,
)


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


def test_strict_output_restores_float64_calibrated_camera_exactly():
    calibrated_world_to_camera = np.asarray(
        [
            [
                [1.0, 0.0, 0.0, 1.234567890123],
                [0.0, 1.0, 0.0, -2.345678901234],
                [0.0, 0.0, 1.0, 3.456789012345],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    estimated = np.linalg.inv(
        calibrated_world_to_camera
    ).astype(np.float32)

    output = calibrated_camera_to_world_for_output(
        estimated,
        calibrated_world_to_camera,
        strict=True,
    )

    assert output.dtype == np.float64
    np.testing.assert_array_equal(
        output, np.linalg.inv(calibrated_world_to_camera)
    )


def test_pointmap_ray_retarget_preserves_z_and_enforces_exact_k():
    camera_to_world = np.eye(4)
    intrinsics = np.asarray([2.0, 2.0, 0.0, 0.0])
    # The second point has the correct z but was unprojected through a
    # different focal and therefore belongs to the wrong pixel ray.
    points = np.asarray(
        [[[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]]],
        dtype=np.float32,
    )
    retargeted = retarget_pointmap_camera_rays(
        points, camera_to_world, intrinsics
    )
    np.testing.assert_allclose(retargeted[..., 2], points[..., 2])
    np.testing.assert_allclose(
        retargeted,
        np.asarray([[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        retarget_pointmap_camera_rays(
            retargeted, camera_to_world, intrinsics
        ),
        retargeted,
        atol=1e-6,
    )


def test_flat_mast3r_pointmap_retargets_with_explicit_raster_shape():
    camera_to_world = np.eye(4)
    intrinsics = np.asarray([2.0, 2.0, 0.0, 0.0])
    flattened = np.asarray(
        [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]],
        dtype=np.float32,
    )

    retargeted = retarget_pointmap_camera_rays(
        flattened,
        camera_to_world,
        intrinsics,
        raster_shape=(1, 2),
    )

    assert retargeted.shape == flattened.shape
    np.testing.assert_allclose(
        retargeted,
        np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        retarget_pointmap_camera_rays(
            retargeted,
            camera_to_world,
            intrinsics,
            raster_shape=(1, 2),
        ),
        retargeted,
        atol=1e-6,
    )


def test_mast3r_sparse_sidecar_uses_the_advertised_fixed_camera_contract():
    source = open("mast3r/run_mast3r.py", encoding="utf-8").read()
    assert "calibrated_pose_contract_internal.json" in source
    assert '"advertised_fixed_camera_output"' in source
    assert "export_K = output_intrinsics[image_names[idx_img]]" in source
    assert "params=export_camera_params" in source
