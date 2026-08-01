"""CUDA integration checks for the jointly sorted 2D/3D rasterizer."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


RASTER_ROOT = (
    Path(__file__).resolve().parents[1]
    / "2d-gaussian-splatting/submodules/diff-surfel-rasterization"
)
sys.path.insert(0, str(RASTER_ROOT))


def _api():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        from diff_surfel_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
            MixedGaussianRasterizer,
        )
    except ImportError as error:
        pytest.skip(f"mixed rasterizer extension is not built: {error}")
    return GaussianRasterizationSettings, GaussianRasterizer, MixedGaussianRasterizer


def _settings(width=61, height=47):
    Settings, _, _ = _api()
    # The explicit projection carries unequal focal lengths and an off-centre
    # principal point; this catches regressions that assume symmetric exact-K.
    fx, fy, cx, cy = 53.0, 41.0, 34.5, 20.0
    znear, zfar = 0.1, 37.0
    projection = torch.zeros(4, 4, device="cuda")
    projection[0, 0] = 2.0 * fx / width
    projection[1, 1] = 2.0 * fy / height
    projection[2, 0] = 2.0 * cx / width - 1.0
    projection[2, 1] = 2.0 * cy / height - 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = 1.0
    projection[3, 2] = -(zfar * znear) / (zfar - znear)
    return Settings(
        image_height=height,
        image_width=width,
        tanfovx=width / (2.0 * fx),
        tanfovy=height / (2.0 * fy),
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=projection,
        sh_degree=0,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
    )


def _empty(rows, columns):
    return torch.empty(rows, columns, device="cuda")


def test_zero_volume_is_exactly_the_native_surfel_path():
    _, GaussianRasterizer, MixedGaussianRasterizer = _api()
    settings = _settings()
    xyz = torch.tensor(
        [[-0.2, 0.1, 2.0], [0.25, -0.15, 3.1]], device="cuda"
    )
    scales = torch.tensor([[0.16, 0.11], [0.22, 0.13]], device="cuda")
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.98, 0.0, 0.2, 0.0]],
        device="cuda",
    )
    colors = torch.tensor([[0.8, 0.1, 0.2], [0.1, 0.7, 0.3]], device="cuda")
    opacity = torch.tensor([[0.65], [0.55]], device="cuda")
    means2d = torch.zeros_like(xyz, requires_grad=True)
    legacy_rgb, legacy_radii, legacy_aux = GaussianRasterizer(settings)(
        xyz,
        means2d,
        opacity,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
    )
    mixed_rgb, mixed_radii, mixed_aux, _, gate_audit = MixedGaussianRasterizer(settings)(
        xyz,
        torch.zeros_like(xyz, requires_grad=True),
        scales,
        rotations,
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 4),
        colors,
        opacity,
    )
    assert torch.equal(mixed_radii, legacy_radii)
    assert torch.equal(mixed_rgb, legacy_rgb)
    assert torch.allclose(mixed_aux[:7], legacy_aux, atol=2e-7, rtol=0)
    assert torch.allclose(mixed_aux[7:8], legacy_aux[1:2], atol=1e-7, rtol=0)
    assert torch.allclose(mixed_aux[9:10], legacy_aux[0:1], atol=2e-7, rtol=0)
    assert torch.count_nonzero(mixed_aux[[8, 10]]).item() == 0
    assert gate_audit.numel() == 0


def test_zero_opacity_volume_is_removed_before_tile_emission():
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings()
    surface = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    volume = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
    common = (
        surface,
        torch.zeros_like(surface, requires_grad=True),
        torch.tensor([[0.2, 0.15]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
    )
    colors = torch.tensor(
        [[0.1, 0.7, 0.2], [0.9, 0.1, 0.1]], device="cuda"
    )
    opacity = torch.tensor([[0.7], [0.0]], device="cuda")
    rgb, radii, aux, _, _ = MixedGaussianRasterizer(settings)(
        *common,
        volume,
        torch.zeros_like(volume, requires_grad=True),
        torch.tensor([[2.0, 2.0, 2.0]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        colors,
        opacity,
    )
    reference, _, reference_aux, _, _ = MixedGaussianRasterizer(settings)(
        *common,
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 4),
        colors[:1],
        opacity[:1],
    )

    assert radii[-1].item() == 0
    assert torch.equal(rgb, reference)
    assert torch.equal(aux, reference_aux)


def test_surface_whose_cutoff_support_crosses_projective_horizon_is_culled():
    """An unbounded projected ellipse must not become a full-frame splat."""
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=61, height=47)
    # Rotate the first surfel tangent onto the camera z axis.  With centre
    # z=2, scale=1 and cutoff=3, its finite local support crosses z=0.  The
    # image-space conic therefore has no finite AABB and is outside the
    # rasterizer's mathematical domain.
    surface = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    sqrt_half = 2.0**-0.5
    rgb, radii, aux, _, _ = MixedGaussianRasterizer(settings)(
        surface,
        torch.zeros_like(surface, requires_grad=True),
        torch.tensor([[1.0, 1.0]], device="cuda"),
        torch.tensor(
            [[sqrt_half, 0.0, sqrt_half, 0.0]], device="cuda"
        ),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 4),
        torch.tensor([[0.9, 0.2, 0.1]], device="cuda"),
        torch.tensor([[0.9]], device="cuda"),
    )

    assert radii.item() == 0
    assert torch.count_nonzero(rgb).item() == 0
    assert torch.count_nonzero(aux).item() == 0


@pytest.mark.parametrize("surface_depth,volume_depth,expected_red", [(2.0, 3.0, 0.0), (3.0, 2.0, 1.0)])
def test_surface_and_volume_share_one_depth_order(surface_depth, volume_depth, expected_red):
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=41, height=41)
    surface = torch.tensor([[0.0, 0.0, surface_depth]], device="cuda")
    volume = torch.tensor([[0.0, 0.0, volume_depth]], device="cuda")
    rgb, _, aux, _, _ = MixedGaussianRasterizer(settings)(
        surface,
        torch.zeros_like(surface, requires_grad=True),
        torch.tensor([[0.35, 0.35]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        volume,
        torch.zeros_like(volume, requires_grad=True),
        torch.tensor([[0.35, 0.35, 0.35]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.99], [0.99]], device="cuda"),
    )
    # Projection principal point is (34.5, 20.0), deliberately not image centre.
    centre = rgb[:, 20, 34]
    assert bool((centre[0] > centre[1]).item()) == bool(expected_red)
    assert aux[5, 20, 34].item() == pytest.approx(2.0, abs=1e-5)
    assert torch.allclose(aux[7] + aux[8], aux[1], atol=2e-6)


def test_volume_backward_matches_finite_difference():
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=35, height=29)
    xyz = torch.tensor([[0.08, -0.04, 2.4]], device="cuda", requires_grad=True)
    scale = torch.tensor(
        [[0.18, 0.13, 0.21]], device="cuda", requires_grad=True
    )
    quat = torch.tensor(
        [[0.96, 0.1, -0.2, 0.15]], device="cuda", requires_grad=True
    )
    color = torch.tensor(
        [[0.25, 0.65, 0.15]], device="cuda", requires_grad=True
    )
    opacity = torch.tensor([[0.61]], device="cuda", requires_grad=True)

    def render_loss():
        rgb, _, aux, _, _ = MixedGaussianRasterizer(settings)(
            _empty(0, 3),
            _empty(0, 3),
            _empty(0, 2),
            _empty(0, 4),
            xyz,
            torch.zeros_like(xyz, requires_grad=True),
            scale,
            torch.nn.functional.normalize(quat, dim=-1),
            color,
            opacity,
        )
        yy, xx = torch.meshgrid(
            torch.linspace(0.7, 1.3, rgb.shape[1], device="cuda"),
            torch.linspace(0.8, 1.2, rgb.shape[2], device="cuda"),
            indexing="ij",
        )
        return (
            (rgb * (xx * yy)[None]).mean()
            + 0.01 * aux[0].mean()
            + 0.02 * aux[8].mean()
            + 0.001 * aux[10].mean()
        )

    loss = render_loss()
    loss.backward()
    analytic = [xyz.grad[0, 0].item(), scale.grad[0, 1].item(), color.grad[0, 2].item(), opacity.grad.item()]
    tensors_and_indices = [(xyz, (0, 0)), (scale, (0, 1)), (color, (0, 2)), (opacity, (0, 0))]
    epsilon = 1e-3
    numeric = []
    with torch.no_grad():
        for tensor, index in tensors_and_indices:
            original = tensor[index].item()
            tensor[index] = original + epsilon
            plus = render_loss().item()
            tensor[index] = original - epsilon
            minus = render_loss().item()
            tensor[index] = original
            numeric.append((plus - minus) / (2.0 * epsilon))
    for actual, expected in zip(analytic, numeric):
        assert actual == pytest.approx(expected, rel=3e-2, abs=2e-5)


def test_surface_uv_gate_is_local_differentiable_and_auditable():
    """One surfel can be attenuated locally without changing its rigid half."""
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=61, height=47)
    surface = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    means2d = torch.zeros_like(surface, requires_grad=True)
    scales = torch.tensor([[0.8, 0.45]], device="cuda")
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device="cuda"
    )
    colors = torch.tensor([[0.8, 0.2, 0.1]], device="cuda")
    opacity = torch.tensor([[0.9]], device="cuda")
    gate_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    audit_fields = torch.stack(
        [
            torch.ones(47, 61, device="cuda"),
            torch.linspace(0, 1, 61, device="cuda")[None].expand(47, -1),
        ]
    )

    one = torch.ones(1, 8, 8, device="cuda")
    identity, _, _, _, _ = MixedGaussianRasterizer(settings)(
        surface,
        means2d,
        scales,
        rotations,
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 4),
        colors,
        opacity,
        surface_gate_indices=gate_indices,
        surface_gate_atlas=one,
    )
    ungated, _, _, _, _ = MixedGaussianRasterizer(settings)(
        surface,
        means2d,
        scales,
        rotations,
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 4),
        colors,
        opacity,
    )
    assert torch.allclose(identity, ungated, atol=2e-7, rtol=0)

    atlas = torch.ones(1, 8, 8, device="cuda", requires_grad=True)
    with torch.no_grad():
        atlas[:, :, :4] = 0.05
    locally_gated, _, _, _, gate_audit = MixedGaussianRasterizer(settings)(
        surface,
        means2d,
        scales,
        rotations,
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 4),
        colors,
        opacity,
        surface_gate_indices=gate_indices,
        surface_gate_atlas=atlas,
        audit_fields=audit_fields,
    )
    difference = (ungated - locally_gated).abs().sum(0)
    assert difference[:, :30].mean() > 5 * difference[:, 40:].mean()
    assert gate_audit.shape == (1, 8, 8, 3)
    assert gate_audit[..., 0].sum() > 0
    assert gate_audit[..., 1].sum() > 0
    assert gate_audit[..., 2].sum() > 0
    locally_gated.square().mean().backward()
    assert atlas.grad is not None
    assert torch.isfinite(atlas.grad).all()
    assert atlas.grad.abs().sum() > 0
