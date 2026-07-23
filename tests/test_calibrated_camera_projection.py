import torch
from pathlib import Path

from matcha.dm_utils.rendering import getProjectionMatrix


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_lazy_rgb_export_keeps_calibrated_intrinsics_and_adaptive_zfar():
    source = (REPO_ROOT / "2d-gaussian-splatting" / "render.py").read_text()

    # The full-train renderer takes the lazy path specifically to avoid
    # retaining 1,487 RGB tensors.  It still must create each camera with the
    # calibrated K and the sparse-support-derived far plane consumed in
    # ordinary Scene construction.
    assert "assign_adaptive_camera_zfar" in source
    assert "fx=record.fx" in source
    assert "fy=record.fy" in source
    assert "cx=record.cx" in source
    assert "cy=record.cy" in source
    assert "zfar=record.zfar" in source
