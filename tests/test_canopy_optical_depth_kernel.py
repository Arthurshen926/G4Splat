"""Independent experimental extension; never replaces the production module."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def APIs():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    sys.path.insert(0, str(ROOT / 'experimental/canopy_optical_depth_rasterizer'))
    import canopy_optical_depth_rasterization as experimental
    spec = importlib.util.spec_from_file_location('native_test_helpers', ROOT / 'tests/test_native_mixed_rasterizer.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    helpers._api()
    import diff_surfel_rasterization as original
    return original, experimental, helpers


def render(api, settings, peak, count=1, surface=False):
    xyz = torch.tensor([[.07, 0., 2.]], device='cuda').repeat(count, 1)
    empty = torch.empty(0, 3, device='cuda')
    quat = torch.tensor([[1., 0., 0., 0.]], device='cuda').repeat(count, 1)
    return api.MixedGaussianRasterizer(settings)(
        xyz if surface else empty, torch.zeros_like(xyz) if surface else empty,
        torch.full((count if surface else 0, 2), .2, device='cuda'),
        quat if surface else torch.empty(0, 4, device='cuda'),
        empty if surface else xyz, empty if surface else torch.zeros_like(xyz),
        torch.full((0 if surface else count, 3), .2, device='cuda'),
        torch.empty(0, 4, device='cuda') if surface else quat,
        torch.ones(count, 3, device='cuda'), peak.expand(count))[0]


def test_optical_depth_factorization_and_opacity_gradient():
    old, new, helpers = APIs()
    settings = helpers._settings()
    p = torch.tensor(.7, device='cuda', requires_grad=True)
    rgb = render(new, settings, p)
    child = -torch.expm1(torch.log1p(-p.detach()) / 2)
    torch.testing.assert_close(rgb, render(new, settings, child, 2), atol=3e-6, rtol=2e-5)
    assert (render(old, settings, p)-rgb).abs().max() > .01
    rgb[:, 18:22, 30:34].sum().backward()
    delta = .001
    fd = (render(new, settings, p.detach()+delta)-render(new, settings, p.detach()-delta))[:, 18:22, 30:34].sum()/(2*delta)
    torch.testing.assert_close(p.grad, fd, atol=.003, rtol=.002)


def test_surface_output_and_gradient_unchanged():
    old, new, helpers = APIs()
    settings = helpers._settings()
    values = []
    for api in (old, new):
        p = torch.tensor(.7, device='cuda', requires_grad=True)
        rgb = render(api, settings, p, surface=True)
        rgb.sum().backward()
        values.append((rgb.detach(), p.grad))
    assert torch.equal(values[0][0], values[1][0])
    torch.testing.assert_close(values[0][1], values[1][1], atol=1e-5, rtol=1e-6)


def test_existing_depth_and_geometry_contracts_with_independent_extension(monkeypatch):
    old, new, helpers = APIs()
    monkeypatch.setattr(helpers, '_api', lambda: (new.GaussianRasterizationSettings, new.GaussianRasterizer, new.MixedGaussianRasterizer))
    helpers.test_supported_static_source_position_identity_and_native_gradient()
    helpers.test_volume_depth_query_backward_matches_finite_difference_and_excludes_behind()
    helpers.test_volume_backward_matches_finite_difference()
    helpers.test_surface_rgb_floor_has_only_actual_foreground_extinction_gradient('absolute')
