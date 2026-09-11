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


def test_native_export_support_predicts_single_volume_alpha():
    from scripts.canopy_native_ray_support import sparse_ray_support
    old, new, helpers = APIs(); settings = helpers._settings()
    xyz = torch.tensor([[.07, 0., 2.]], device='cuda')
    scales = torch.full((1, 3), .2, device='cuda')
    quat = torch.tensor([[1., 0., 0., 0.]], device='cuda')
    projected = new._C.diagnostic_volume_support(xyz, scales, quat,
        settings.viewmatrix, settings.projmatrix, 61, 47, settings.tanfovx, settings.tanfovy)
    uv = torch.tensor([[31., 19.], [34., 20.], [1., 1.]], device='cuda')
    K, omitted = sparse_ray_support(projected, uv, torch.full((3,), 3., device='cuda'), 61, 47)
    peak = torch.tensor(.7, device='cuda'); rgb = render(new, settings, peak)
    expected = -torch.expm1(torch.tensor(K.toarray()[:, 0], device='cuda')*torch.log1p(-peak))
    torch.testing.assert_close(rgb[0, uv[:, 1].long(), uv[:, 0].long()], expected, atol=2e-6, rtol=2e-5)
    behind, _ = sparse_ray_support(projected, uv, torch.ones(3, device='cuda'), 61, 47)
    assert behind.nnz == 0


def test_volume_cap_cutoff_and_limited_split_invariance():
    old,new,helpers=APIs();settings=helpers._settings()
    peak=torch.tensor(.999,device='cuda',requires_grad=True)
    value=render(new,settings,peak)[0,20,36]
    torch.testing.assert_close(value,peak.new_tensor(.99),atol=1e-6,rtol=0)
    value.backward();assert peak.grad==0
    child=-torch.expm1(torch.log1p(-peak.detach())/2)
    # The retained per-primitive cap breaks ideal split invariance at saturation.
    assert render(new,settings,child,2)[0,20,36]>value+.005
    tiny=torch.tensor(1e-8,device='cuda',requires_grad=True)
    rgb=render(new,settings,tiny);assert rgb.abs().max()==0
    rgb.sum().backward();assert tiny.grad==0


def test_existing_depth_and_geometry_contracts_with_independent_extension(monkeypatch):
    old, new, helpers = APIs()
    monkeypatch.setattr(helpers, '_api', lambda: (new.GaussianRasterizationSettings, new.GaussianRasterizer, new.MixedGaussianRasterizer))
    helpers.test_supported_static_source_position_identity_and_native_gradient()
    helpers.test_volume_depth_query_backward_matches_finite_difference_and_excludes_behind()
    helpers.test_volume_backward_matches_finite_difference()
    helpers.test_surface_rgb_floor_has_only_actual_foreground_extinction_gradient('absolute')


def test_independent_kernel_preserves_pixel_order_crossings(monkeypatch):
    _, new, helpers = APIs()
    monkeypatch.setattr(helpers, '_api', lambda: (new.GaussianRasterizationSettings, new.GaussianRasterizer, new.MixedGaussianRasterizer))
    helpers.test_surface_and_volume_share_one_depth_order(2., 3., 0.)
    helpers.test_surface_and_volume_share_one_depth_order(3., 2., 1.)
    helpers.test_oblique_surfel_uses_exact_pixel_depth_within_one_tile()
    helpers.test_exact_pixel_order_backward_matches_finite_difference()


def test_candidate_native_pixel_proxy_roundtrips_through_cuda_preprocessor():
    from types import SimpleNamespace
    from scripts.canopy_candidate_diagnostic import ray_candidates
    from scripts.canopy_native_pixel_camera import NativePixelCamera
    _,new,helpers=APIs();settings=helpers._settings()
    camera=SimpleNamespace(cx=34.5,cy=20.,focal_x=53.,focal_y=41.,
        world_view_transform=settings.viewmatrix,camera_center=torch.zeros(3,device='cuda'))
    uv=torch.tensor([[12.,14.],[37.,30.]],device='cuda')
    xyz,scales,_=ray_candidates(NativePixelCamera(camera),uv,torch.tensor([2.,3.],device='cuda'),
        torch.zeros(2,3,device='cuda'),[1.])
    quat=torch.tensor([[1.,0.,0.,0.]]*2,device='cuda')
    projected=new._C.diagnostic_volume_support(xyz,scales.contiguous(),quat,
        settings.viewmatrix,settings.projmatrix,61,47,settings.tanfovx,settings.tanfovy)
    torch.testing.assert_close(projected[0],uv,atol=1e-5,rtol=0)


def test_live_source_projection_stays_above_both_kernel_cutoffs():
    from scripts.canopy_source_opacity_projection import project_source_opacity_
    old,new,helpers=APIs();settings=helpers._settings()
    for api in (old,new):
        logits=torch.nn.Parameter(torch.tensor([-30.],device='cuda'))
        project_source_opacity_(logits,torch.tensor([True],device='cuda'),enabled=True)
        rgb=render(api,settings,logits.sigmoid()[0])
        assert rgb.max()>0
        rgb.sum().backward()
        assert logits.grad is not None and logits.grad.abs().sum()>0
