from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
SURFEL_ROOT = REPO_ROOT / "2d-gaussian-splatting"
sys.path.insert(0, str(SURFEL_ROOT))

import torch

from matcha.dm_utils.rendering import getProjectionMatrix
from utils.intrinsics_utils import scale_calibrated_intrinsics

def test_off_axis_projection_preserves_colmap_principal_point():
    projection = getProjectionMatrix(
        0.01,
        25.0,
        1.0,
        1.0,
        fx=800.0,
        fy=600.0,
        cx=300.0,
        cy=250.0,
        image_width=1000,
        image_height=800,
    )

    assert torch.isclose(projection[0, 0], torch.tensor(1.6))
    assert torch.isclose(projection[1, 1], torch.tensor(1.5))
    assert torch.isclose(projection[0, 2], torch.tensor(-0.4))
    assert torch.isclose(projection[1, 2], torch.tensor(-0.375))


def test_centered_intrinsics_match_the_legacy_symmetric_projection():
    exact = getProjectionMatrix(
        0.01,
        25.0,
        1.0,
        1.0,
        fx=800.0,
        fy=600.0,
        cx=500.0,
        cy=400.0,
        image_width=1000,
        image_height=800,
    )
    # 2*atan(size/(2*focal)) gives the same centered FoV projection.
    fov_x = 2.0 * torch.atan(torch.tensor(1000.0 / 1600.0)).item()
    fov_y = 2.0 * torch.atan(torch.tensor(800.0 / 1200.0)).item()
    legacy = getProjectionMatrix(0.01, 25.0, fov_x, fov_y)

    assert torch.allclose(exact, legacy, atol=1e-6)


def test_mip_filter_projection_uses_the_calibrated_principal_point():
    source = (REPO_ROOT / "2d-gaussian-splatting" / "scene" / "gaussian_model.py").read_text()

    assert "x = x / z * camera.focal_x + camera.cx" in source
    assert "y = y / z * camera.focal_y + camera.cy" in source


def test_lazy_rgb_export_keeps_calibrated_intrinsics_and_training_zfar_policy():
    source = (REPO_ROOT / "2d-gaussian-splatting" / "render.py").read_text()

    # The full-train renderer takes the lazy path specifically to avoid
    # retaining 1,487 RGB tensors.  It still must create each camera with the
    # calibrated K and the same far-plane policy consumed during training.
    # Camera-only no-COLMAP models must not reopen points3D just for
    # evaluation, while ordinary upstream models retain adaptive zfar.
    assert "assign_adaptive_camera_zfar" in source
    assert "load_point_cloud=not bool(" in source
    assert 'getattr(args, "camera_only_scene", False)' in source
    assert "fx=record.fx" in source
    assert "fy=record.fy" in source
    assert "cx=record.cx" in source
    assert "cy=record.cy" in source
    assert "zfar=record.zfar" in source


def test_materialized_rgb_raster_scales_k_from_colmap_calibration_raster():
    camera = SimpleNamespace(
        width=1920,
        height=1080,
        fx=1665.0,
        fy=1665.0,
        cx=960.0,
        cy=540.0,
        FovX=1.0,
        FovY=1.0,
    )

    fx, fy, cx, cy = scale_calibrated_intrinsics(camera, (640, 360))

    assert fx == 555.0
    assert fy == 555.0
    assert cx == 320.0
    assert cy == 180.0


def test_intrinsic_scaling_does_not_treat_rgb_raster_as_k_coordinate_system():
    camera = SimpleNamespace(
        # This mimics a 640x360 RGB target whose CameraInfo still preserves
        # the 1920x1080 cameras.bin calibration.
        width=1920,
        height=1080,
        fx=1500.0,
        fy=1200.0,
        cx=900.0,
        cy=500.0,
        FovX=1.0,
        FovY=1.0,
    )

    fx, fy, cx, cy = scale_calibrated_intrinsics(camera, (320, 180))

    assert (fx, fy, cx, cy) == (250.0, 200.0, 150.0, 250.0 / 3.0)
