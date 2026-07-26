import torch
from plyfile import PlyData

from outdoor.standard_3dgs import (
    load_standard_3dgs_ply,
    save_standard_3dgs_ply,
    standard_property_names,
    validate_standard_3dgs_ply,
)


def test_standard_3dgs_roundtrip_has_no_teacher_fields(tmp_path):
    count = 7
    degree = 2
    coefficients = (degree + 1) ** 2
    xyz = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
    log_scales = torch.full((count, 3), -2.0)
    quaternion = torch.zeros(count, 4)
    quaternion[:, 0] = 1
    opacity = torch.linspace(-2, 2, count)[:, None]
    features = torch.randn(count, coefficients, 3)
    path = tmp_path / "point_cloud.ply"
    save_standard_3dgs_ply(
        path,
        xyz=xyz,
        log_scales=log_scales,
        quaternions=quaternion,
        opacity_logits=opacity,
        features=features,
        sh_degree=degree,
    )
    audit = validate_standard_3dgs_ply(path, sh_degree=degree)
    assert audit["standard_graphdeco_schema"]
    assert audit["custom_property_count"] == 0
    names = [
        value.name
        for value in PlyData.read(path).elements[0].properties
    ]
    assert names == standard_property_names(degree)
    assert not any(
        token in name
        for name in names
        for token in ("role", "gate", "sequence", "uncertainty")
    )
    restored = load_standard_3dgs_ply(
        path, sh_degree=degree, device="cpu"
    )
    torch.testing.assert_close(restored["xyz"], xyz)
    torch.testing.assert_close(restored["log_scales"], log_scales)
    torch.testing.assert_close(restored["opacity_logits"], opacity)
    torch.testing.assert_close(restored["features"], features)

