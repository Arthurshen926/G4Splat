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


def test_supported_static_source_position_identity_and_native_gradient():
    from scripts.canopy_static_source_position import StaticSourcePosition
    _, _, Rasterizer = _api(); settings = _settings()
    surface = torch.tensor([[0., 0., 3.]], device='cuda')
    leaves = torch.tensor([[.07, 0., 2.], [.1, 0., 4.]], device='cuda')
    scales = torch.full((2, 3), .2, device='cuda')
    position = StaticSourcePosition(leaves, scales, torch.tensor([True, False], device='cuda'), 2.)
    def render(xyz):
        return Rasterizer(settings)(surface, torch.zeros_like(surface),
            torch.tensor([[.5, .5]], device='cuda'), torch.tensor([[1., 0., 0., 0.]], device='cuda'),
            xyz, torch.zeros_like(xyz), scales, torch.tensor([[1., 0., 0., 0.]]*2, device='cuda'),
            torch.tensor([[.7, .7, .7], [.1, .3, .1], [.2, .3, .2]], device='cuda'),
            torch.tensor([.99, .3, .3], device='cuda'))[0]
    assert torch.equal(render(position.apply(leaves)), render(leaves))
    def loss(): return (render(position.apply(leaves))[:, 18:21, 31:34]-.2).square().mean()
    loss().backward(); analytic = position.code.grad[0, 0].clone()
    assert analytic.abs() > 1e-6
    with torch.no_grad():
        position.code[0, 0] = .001; plus = loss()
        position.code[0, 0] = -.001; minus = loss()
        position.code.zero_()
    torch.testing.assert_close(analytic, (plus-minus)/.002, atol=2e-5, rtol=3e-2)
    assert surface.grad is None and leaves.grad is None and scales.grad is None


def test_owned_foliage_color_keeps_native_opacity_and_geometry_gradient():
    from scripts.canopy_color_gradient_ownership import backward_canopy_owned_colors
    from scripts.canopy_frozen_reference_cache import FrozenReferenceCache
    _, _, Rasterizer = _api(); settings = _settings()
    surface = torch.tensor([[0., 0., 3.]], device='cuda')
    leaves = torch.tensor([[0., 0., 2.], [.1, 0., 4.]], device='cuda', requires_grad=True)
    logits = torch.tensor([-1.4, -1.4], device='cuda', requires_grad=True)
    colors = torch.tensor([[.1, .3, .1], [.2, .3, .2]], device='cuda', requires_grad=True)
    rgb = Rasterizer(settings)(surface, torch.zeros_like(surface),
        torch.tensor([[.5, .5]], device='cuda'), torch.tensor([[1., 0., 0., 0.]], device='cuda'),
        leaves, torch.zeros_like(leaves), torch.full((2, 3), .2, device='cuda'),
        torch.tensor([[1., 0., 0., 0.]]*2, device='cuda'),
        torch.cat((colors.new_tensor([[.7, .7, .7]]), colors)),
        torch.cat((logits.new_tensor([.99]), logits.sigmoid())))[0]
    cache = FrozenReferenceCache(1)
    first = cache.get('native', lambda: rgb, torch.device('cuda'))
    second = cache.get('native', lambda: rgb*0, torch.device('cuda'))
    assert torch.equal(first, rgb.detach()) and torch.equal(second, first)
    assert not first.requires_grad and cache.stats()['hits'] == 1
    tree = (rgb[:, 18:21, 31:34]-.2).square().mean()
    background = 4*(rgb[:, 22:25, 35:38]-.7).square().mean()
    expected_color, = torch.autograd.grad(tree, colors, retain_graph=True)
    expected_optics = torch.autograd.grad(tree+background, (logits, leaves), retain_graph=True)
    backward_canopy_owned_colors(tree, background, color_parameters=[colors],
                                 optical_geometry_parameters=[logits, leaves])
    torch.testing.assert_close(colors.grad, expected_color, atol=2e-6, rtol=2e-4)
    for parameter, expected in zip((logits, leaves), expected_optics):
        torch.testing.assert_close(parameter.grad, expected, atol=2e-6, rtol=2e-4)


def test_boundary_reference_loss_rejects_foreground_occlusion_but_not_hidden_leaves():
    from scripts.canopy_boundary_preservation import boundary_rgb_preservation
    _, _, Rasterizer = _api(); settings = _settings()
    surface = torch.tensor([[0., 0., 3.]], device='cuda')
    leaves = torch.tensor([[0., 0., 2.], [0., 0., 4.]], device='cuda')
    logits = torch.tensor([-1.4, -1.4], device='cuda', requires_grad=True)
    mask = torch.zeros(47, 61, device='cuda', dtype=torch.bool); mask[19:22, 33:36] = True
    def render(alpha):
        return Rasterizer(settings)(surface, torch.zeros_like(surface),
            torch.tensor([[.5, .5]], device='cuda'), torch.tensor([[1., 0., 0., 0.]], device='cuda'),
            leaves, torch.zeros_like(leaves), torch.full((2, 3), .2, device='cuda'),
            torch.tensor([[1., 0., 0., 0.]]*2, device='cuda'),
            torch.tensor([[.7, .7, .7], [0., 0., 0.], [0., 0., 0.]], device='cuda'),
            torch.cat((alpha.new_tensor([.99]), alpha)))[0]
    reference = render(torch.zeros_like(logits)).detach()
    def objective(value):
        return boundary_rgb_preservation(render(value.sigmoid()), reference, mask, mask)
    objective(logits).backward()
    assert logits.grad[0] > 0 and logits.grad[1] == 0
    with torch.no_grad():
        delta = logits.new_tensor([.001, 0.])
        finite_difference = (objective(logits+delta)-objective(logits-delta))/.002
    torch.testing.assert_close(logits.grad[0], finite_difference, atol=2e-5, rtol=2e-3)


@pytest.mark.parametrize('domain', ['absolute', 'relative'])
def test_surface_rgb_floor_has_only_actual_foreground_extinction_gradient(domain):
    from scripts.canopy_surface_rgb_feasibility import surface_rgb_feasibility_loss
    from scripts.canopy_relative_radiance_feasibility import initial_gradient_normalization, relative_surface_rgb_feasibility_loss
    _, _, Rasterizer = _api(); settings = _settings()
    surface = torch.tensor([[0., 0., 3.]], device='cuda')
    leaves = torch.tensor([[0., 0., 2.], [0., 0., 4.]], device='cuda')
    logits = torch.tensor([-1.4, -1.4], device='cuda', requires_grad=True)
    mask = torch.zeros(47, 61, device='cuda', dtype=torch.bool); mask[19:22, 33:36] = True
    initial_scale = None
    def loss_at(value):
        nonlocal initial_scale
        colors = torch.tensor([[.7, .7, .7], [0., 0., 0.], [0., 0., 0.]], device='cuda')
        opacity = torch.cat((value.new_tensor([.99]), value.sigmoid()))
        rgb, _, _, _, _ = Rasterizer(settings)(surface, torch.zeros_like(surface),
            torch.tensor([[.5, .5]], device='cuda'), torch.tensor([[1., 0., 0., 0.]], device='cuda'),
            leaves, torch.zeros_like(leaves), torch.full((2, 3), .2, device='cuda'),
            torch.tensor([[1., 0., 0., 0.]]*2, device='cuda'), colors, opacity)
        target = torch.full_like(rgb, .1)
        if domain == 'relative':
            if initial_scale is None: initial_scale = initial_gradient_normalization(rgb, target, mask)
            return relative_surface_rgb_feasibility_loss(rgb, target, mask, initial_scale)
        return surface_rgb_feasibility_loss(rgb, target, mask)
    loss_at(logits).backward()
    assert logits.grad[0] < 0 and logits.grad[1] == 0
    with torch.no_grad():
        delta = logits.new_tensor([.001, 0.])
        finite_difference = (loss_at(logits+delta)-loss_at(logits-delta))/.002
    torch.testing.assert_close(logits.grad[0], finite_difference, atol=2e-5, rtol=2e-3)


def test_native_mass_retraction_requires_scale_opacity_chain_rule():
    from outdoor.optical_mass_gradient import mass_preserving_scale_alpha
    _, _, Rasterizer = _api()
    settings = _settings()
    s = torch.tensor(.18, device='cuda').log().requires_grad_()
    alpha = torch.tensor(.3, device='cuda')
    mass = (-torch.log1p(-alpha)*(2*s).exp()).detach()

    def objective(log_scale, peak_alpha):
        _, _, aux, _, _ = Rasterizer(settings)(
            _empty(0, 3), _empty(0, 3), _empty(0, 2), _empty(0, 4),
            torch.tensor([[0., 0., 2.4]], device='cuda'), _empty(1, 3).zero_(),
            log_scale.exp().expand(1, 3),
            torch.tensor([[1., 0., 0., 0.]], device='cuda'),
            torch.ones(1, 3, device='cuda'), peak_alpha.reshape(1, 1))
        return (aux[8, 20, 36]-.7).square()

    initial = objective(s, alpha)
    raw_g = torch.autograd.grad(initial, s)[0]
    corrected = mass_preserving_scale_alpha(alpha, (2*s).exp())
    assert torch.equal(corrected, alpha)
    corrected_g = torch.autograd.grad(objective(s, corrected), s)[0]
    def retracted(x): return objective(x, -torch.expm1(-mass/(2*x).exp()))
    with torch.no_grad():
        h = 1e-3
        fd = (retracted(s+h)-retracted(s-h))/(2*h)
        torch.testing.assert_close(corrected_g, fd, atol=2e-4, rtol=3e-3)
        assert raw_g < 0 < corrected_g
        assert retracted(s-.001*raw_g) > initial
        assert retracted(s-.001*corrected_g) < initial


def test_darkening_bound_adam_grows_only_real_wall_front_leaf():
    from outdoor.canopy_foreground_evidence import foreground_darkening_loss
    _,_,MixedGaussianRasterizer=_api();settings=_settings()
    logits=torch.nn.Parameter(torch.logit(torch.tensor([[.99],[.10],[.10]],device='cuda')))
    colors=torch.ones(3,3,device='cuda',requires_grad=True)*.5
    surface=torch.tensor([[0.,0.,3.]],device='cuda',requires_grad=True)
    leaves=torch.tensor([[0.,0.,2.],[0.,0.,4.]],device='cuda',requires_grad=True)
    bounds=torch.empty(2,47,61,device='cuda');bounds[0]=.2001;bounds[1]=2.97
    _,_,aux,_,_=MixedGaussianRasterizer(settings)(
        surface,torch.zeros_like(surface),torch.tensor([[.5,.5]],device='cuda'),
        torch.tensor([[1.,0.,0.,0.]],device='cuda'),leaves,torch.zeros_like(leaves),
        torch.full((2,3),.2,device='cuda'),torch.tensor([[1.,0.,0.,0.]]*2,device='cuda'),
        colors,logits.sigmoid(),depth_query_bounds=bounds)
    mask=torch.zeros(47,61,device='cuda',dtype=torch.bool);mask[19:22,33:36]=True
    # Zero-based aux[12] is the intrinsic interval hit; behind-wall alpha is excluded.
    loss=foreground_darkening_loss(aux[12],torch.full((47,61),.8,device='cuda'),mask)
    loss.backward()
    assert logits.grad[1]<0
    assert logits.grad[0]==0 and logits.grad[2]==0
    assert torch.count_nonzero(surface.grad)==0
    before=logits.detach().clone()
    optimizer=torch.optim.Adam([logits],lr=.05,eps=1.e-15);optimizer.step()
    assert logits[1]>before[1] and torch.equal(logits[[0,2]],before[[0,2]])


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
    assert torch.count_nonzero(mixed_aux[[8, 10, 11, 12]]).item() == 0
    assert gate_audit.numel() == 0


@pytest.mark.parametrize("surface_prefix", [False, True])
def test_intrinsic_hit_survives_opaque_prefix_without_changing_rgb(surface_prefix):
    _, _, Rasterizer = _api()
    settings = _settings()
    prefix = torch.tensor([[0., 0., 1. + .1 * i] for i in range(6)], device="cuda")
    hits = torch.tensor([[0., 0., 3.], [0., 0., 4.]], device="cuda")
    surface = prefix if surface_prefix else _empty(0, 3)
    volume = hits if surface_prefix else torch.cat((prefix, hits))
    colors = torch.full((8, 3), .4, device="cuda")
    rotations = torch.tensor([[1., 0., 0., 0.]], device="cuda")
    bounds = torch.empty(2, settings.image_height, settings.image_width, device="cuda")
    bounds[0].fill_(2.5); bounds[1].fill_(3.5)

    def render(opacity, query):
        return Rasterizer(settings)(
            surface, torch.zeros_like(surface),
            torch.full((len(surface), 2), .8, device="cuda"),
            rotations.repeat(len(surface), 1),
            volume, torch.zeros_like(volume),
            torch.full((len(volume), 3), .8, device="cuda"),
            rotations.repeat(len(volume), 1), colors, opacity,
            **({"depth_query_bounds": bounds} if query else {}),
        )

    opacity = torch.tensor([[.99]] * 6 + [[.4], [.2]], device="cuda", requires_grad=True)
    queried = render(opacity, True)
    ordinary = render(opacity, False)
    torch.testing.assert_close(queried[0], ordinary[0], rtol=0, atol=0)
    torch.testing.assert_close(queried[2][:11], ordinary[2][:11], rtol=0, atol=0)
    hit = queried[2][12, 20, 34]
    assert float(hit) > .2
    hit.backward()
    assert float(opacity.grad[6]) > .5
    assert torch.count_nonzero(opacity.grad[:6]) == 0
    assert float(opacity.grad[7]) == 0
    plus, minus = opacity.detach().clone(), opacity.detach().clone()
    plus[6] += .001; minus[6] -= .001
    finite = (render(plus, True)[2][12, 20, 34] - render(minus, True)[2][12, 20, 34]) / .002
    torch.testing.assert_close(opacity.grad[6, 0], finite, rtol=.003, atol=.001)


def test_volume_depth_query_separates_prehit_hit_and_behind_alpha():
    """The query is intrinsic volume alpha, not mixed layer contribution."""
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings()
    volume = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]],
        device="cuda",
    )
    opacity = torch.tensor([[0.15], [0.20], [0.25]], device="cuda")
    scales = torch.full((3, 3), 0.15, device="cuda")
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 3, device="cuda"
    )
    colors = torch.tensor(
        [[0.8, 0.2, 0.1], [0.2, 0.8, 0.1], [0.1, 0.2, 0.8]],
        device="cuda",
    )
    bounds = torch.empty(
        (2, settings.image_height, settings.image_width), device="cuda"
    )
    bounds[0].fill_(2.5)
    bounds[1].fill_(3.5)
    _, _, queried, _, _ = MixedGaussianRasterizer(settings)(
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 2),
        _empty(0, 4),
        volume,
        torch.zeros_like(volume, requires_grad=True),
        scales,
        rotations,
        colors,
        opacity,
        depth_query_bounds=bounds,
    )

    # A one-row render provides the exact EWA alpha at this sub-pixel sample.
    isolated_alpha = []
    for row in range(3):
        _, _, isolated, _, _ = MixedGaussianRasterizer(settings)(
            _empty(0, 3),
            _empty(0, 3),
            _empty(0, 2),
            _empty(0, 4),
            volume[row : row + 1],
            torch.zeros_like(volume[row : row + 1], requires_grad=True),
            scales[row : row + 1],
            rotations[row : row + 1],
            colors[row : row + 1],
            opacity[row : row + 1],
        )
        isolated_alpha.append(isolated[1, 20, 34])

    assert torch.allclose(queried[11, 20, 34], isolated_alpha[0])
    assert torch.allclose(queried[12, 20, 34], isolated_alpha[1])
    assert float(queried[11, 20, 34]) > 0
    assert float(queried[12, 20, 34]) > 0
    # The z=4 row is behind the target interval and contributes to neither
    # query channel, even though it remains visible in ordinary volume alpha.
    expected_total = 1.0
    for alpha in isolated_alpha:
        expected_total = expected_total * (1.0 - alpha)
    assert torch.allclose(queried[8, 20, 34], 1.0 - expected_total)


def test_opaque_rigid_bound_removes_hidden_leaf_retirement_gradient():
    from outdoor.canopy_support_expansion import clip_canopy_depth_queries_before_rigid
    _,_,Rasterizer=_api();settings=_settings()
    shape=(settings.image_height,settings.image_width)
    pre=torch.stack((torch.full(shape,3.5,device='cuda'),torch.full(shape,4.,device='cuda')))
    hit=torch.stack((torch.full(shape,3.8,device='cuda'),torch.full(shape,4.2,device='cuda')))
    clipped,_,_,_=clip_canopy_depth_queries_before_rigid(pre,hit,torch.ones(shape,dtype=torch.bool,device='cuda'),
        torch.zeros(shape,device='cuda'),torch.ones(shape,device='cuda'),
        torch.full(shape,2.5,device='cuda'),torch.ones(shape,device='cuda'))
    volume=torch.tensor([[0.,0.,2.],[0.,0.,3.]],device='cuda')
    def render(bounds):
        opacity=torch.tensor([[.2],[.3]],device='cuda',requires_grad=True)
        result=Rasterizer(settings)(_empty(0,3),_empty(0,3),_empty(0,2),_empty(0,4),
            volume,torch.zeros_like(volume),torch.full((2,3),.15,device='cuda'),
            torch.tensor([[1.,0.,0.,0.]]*2,device='cuda'),torch.full((2,3),.4,device='cuda'),
            opacity,depth_query_bounds=bounds)
        result[2][11,20,34].backward()
        return result[0].detach(),opacity.grad
    old_rgb,old_gradient=render(pre);new_rgb,new_gradient=render(clipped)
    assert old_gradient[1]>0 and new_gradient[1]==0
    assert new_gradient[0]>0
    assert torch.equal(old_rgb,new_rgb)


def test_volume_depth_query_backward_matches_finite_difference_and_excludes_behind():
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings()
    volume = torch.tensor(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 4.0]],
        device="cuda",
    )
    scales = torch.full((3, 3), 0.15, device="cuda")
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 3, device="cuda"
    )
    colors = torch.full((3, 3), 0.4, device="cuda")
    bounds = torch.empty(
        (2, settings.image_height, settings.image_width), device="cuda"
    )
    bounds[0].fill_(2.5)
    bounds[1].fill_(3.5)

    def objective(opacity):
        _, _, aux, _, _ = MixedGaussianRasterizer(settings)(
            _empty(0, 3),
            _empty(0, 3),
            _empty(0, 2),
            _empty(0, 4),
            volume,
            torch.zeros_like(volume, requires_grad=True),
            scales,
            rotations,
            colors,
            opacity,
            depth_query_bounds=bounds,
        )
        return aux[11, 20, 34] + 2.0 * aux[12, 20, 34]

    opacity = torch.tensor(
        [[0.15], [0.20], [0.25]], device="cuda", requires_grad=True
    )
    objective(opacity).backward()
    analytic = opacity.grad.detach().reshape(-1)
    epsilon = 1.0e-3
    finite_difference = []
    for row in range(3):
        plus = opacity.detach().clone()
        minus = opacity.detach().clone()
        plus[row] += epsilon
        minus[row] -= epsilon
        finite_difference.append(
            (objective(plus) - objective(minus)) / (2.0 * epsilon)
        )
    finite_difference = torch.stack(finite_difference)

    assert torch.allclose(analytic[:2], finite_difference[:2], atol=2e-4, rtol=2e-3)
    assert analytic[0] > 0
    assert analytic[1] > 0
    assert analytic[2].item() == 0.0
    assert finite_difference[2].item() == 0.0


def test_dense_surface_stack_crosses_multiple_exact_sort_batches():
    """More than one 256-row batch must remain exact in both directions."""
    _, GaussianRasterizer, MixedGaussianRasterizer = _api()
    settings = _settings()
    count = 300
    depths = torch.linspace(2.0, 5.0, count, device="cuda")
    xyz = torch.stack(
        [torch.zeros_like(depths), torch.zeros_like(depths), depths], dim=1
    )
    scales = torch.full((count, 2), 0.12, device="cuda")
    rotations = torch.zeros(count, 4, device="cuda")
    rotations[:, 0] = 1.0
    colors = torch.stack(
        [
            torch.linspace(0.1, 0.9, count, device="cuda"),
            torch.linspace(0.8, 0.2, count, device="cuda"),
            torch.full((count,), 0.35, device="cuda"),
        ],
        dim=1,
    )
    native_opacity = torch.full(
        (count, 1), 0.008, device="cuda", requires_grad=True
    )
    mixed_opacity = native_opacity.detach().clone().requires_grad_(True)

    native_rgb, native_radii, native_aux = GaussianRasterizer(settings)(
        xyz,
        torch.zeros_like(xyz, requires_grad=True),
        native_opacity,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
    )
    mixed_rgb, mixed_radii, mixed_aux, _, _ = MixedGaussianRasterizer(settings)(
        xyz,
        torch.zeros_like(xyz, requires_grad=True),
        scales,
        rotations,
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 4),
        colors,
        mixed_opacity,
    )

    assert torch.all(native_radii > 0)
    assert torch.equal(mixed_radii, native_radii)
    assert torch.allclose(mixed_rgb, native_rgb, atol=2e-6, rtol=0)
    assert torch.allclose(mixed_aux[:7], native_aux, atol=2e-6, rtol=0)

    native_loss = native_rgb[:, 20, 34].sum() + 0.01 * native_aux[0, 20, 34]
    mixed_loss = mixed_rgb[:, 20, 34].sum() + 0.01 * mixed_aux[0, 20, 34]
    native_loss.backward()
    mixed_loss.backward()
    assert torch.allclose(
        mixed_opacity.grad, native_opacity.grad, atol=2e-6, rtol=2e-5
    )


def test_dense_surface_batches_merge_exactly_with_interleaved_volume_stream():
    """A volume event must not skip or reorder a later exact-surface batch.

    The mixed kernel scans dense surface candidates in bounded 256-row
    batches while volumes stream from the tile-depth list.  A surface-only
    regression does not exercise the merge boundary, so compare a 270-surface
    stack with four interleaved volume events against an explicit ray-local
    front-to-back composite.  The same expression also checks every opacity
    derivative without relying on a second renderer implementation.
    """

    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=41, height=41)
    surface_count = 270
    surface_depth = torch.linspace(2.0, 5.0, surface_count, device="cuda")
    volume_depth = torch.tensor(
        [2.25, 3.25, 4.25, 4.75], device="cuda"
    )
    surface = torch.stack(
        [
            torch.zeros_like(surface_depth),
            torch.zeros_like(surface_depth),
            surface_depth,
        ],
        dim=1,
    )
    volume = torch.stack(
        [
            torch.zeros_like(volume_depth),
            torch.zeros_like(volume_depth),
            volume_depth,
        ],
        dim=1,
    )
    surface_scales = torch.full(
        (surface_count, 2), 0.12, device="cuda"
    )
    volume_scales = torch.full((len(volume), 3), 0.12, device="cuda")
    surface_rotation = torch.zeros(surface_count, 4, device="cuda")
    volume_rotation = torch.zeros(len(volume), 4, device="cuda")
    surface_rotation[:, 0] = 1.0
    volume_rotation[:, 0] = 1.0
    surface_color = torch.stack(
        [
            torch.linspace(0.05, 0.45, surface_count, device="cuda"),
            torch.linspace(0.8, 0.2, surface_count, device="cuda"),
            torch.full((surface_count,), 0.15, device="cuda"),
        ],
        dim=1,
    )
    volume_color = torch.tensor(
        [
            [0.9, 0.1, 0.2],
            [0.7, 0.2, 0.4],
            [0.5, 0.3, 0.6],
            [0.3, 0.4, 0.8],
        ],
        device="cuda",
    )
    colors = torch.cat([surface_color, volume_color], dim=0)
    opacity = torch.full(
        (surface_count + len(volume), 1),
        0.0015,
        device="cuda",
        requires_grad=True,
    )

    rgb, radii, aux, _, _ = MixedGaussianRasterizer(settings)(
        surface,
        torch.zeros_like(surface, requires_grad=True),
        surface_scales,
        surface_rotation,
        volume,
        torch.zeros_like(volume, requires_grad=True),
        volume_scales,
        volume_rotation,
        colors,
        opacity,
    )
    assert torch.all(radii > 0)

    # Recover the fixed EWA coefficient of each primitive from an isolated
    # render.  Alpha is linear in opacity here (well below the 0.99 clamp), so
    # the explicit composite remains differentiable with respect to the same
    # live opacity tensor.
    probe_y, probe_x = 20, 34
    coefficients = []
    with torch.no_grad():
        for row in range(surface_count):
            _, _, isolated, _, _ = MixedGaussianRasterizer(settings)(
                surface[row : row + 1],
                torch.zeros_like(surface[row : row + 1]),
                surface_scales[row : row + 1],
                surface_rotation[row : row + 1],
                _empty(0, 3),
                _empty(0, 3),
                _empty(0, 3),
                _empty(0, 4),
                surface_color[row : row + 1],
                opacity.detach()[row : row + 1],
            )
            coefficients.append(
                isolated[1, probe_y, probe_x] / opacity.detach()[row, 0]
            )
        for local_row in range(len(volume)):
            row = surface_count + local_row
            _, _, isolated, _, _ = MixedGaussianRasterizer(settings)(
                _empty(0, 3),
                _empty(0, 3),
                _empty(0, 2),
                _empty(0, 4),
                volume[local_row : local_row + 1],
                torch.zeros_like(volume[local_row : local_row + 1]),
                volume_scales[local_row : local_row + 1],
                volume_rotation[local_row : local_row + 1],
                volume_color[local_row : local_row + 1],
                opacity.detach()[row : row + 1],
            )
            coefficients.append(
                isolated[1, probe_y, probe_x] / opacity.detach()[row, 0]
            )
    coefficients = torch.stack(coefficients)
    all_depth = torch.cat([surface_depth, volume_depth])
    primitive_id = torch.arange(len(all_depth), device="cuda")
    # Depths are all distinct, but retain the production primitive-id tie
    # break in the reference ordering.
    order = sorted(
        range(len(all_depth)),
        key=lambda row: (float(all_depth[row]), int(primitive_id[row])),
    )
    transmittance = opacity.new_ones(())
    expected_rgb = opacity.new_zeros(3)
    expected_surface_alpha = opacity.new_zeros(())
    expected_volume_alpha = opacity.new_zeros(())
    for row in order:
        alpha = opacity[row, 0] * coefficients[row]
        contribution = transmittance * alpha
        expected_rgb = expected_rgb + contribution * colors[row]
        if row < surface_count:
            expected_surface_alpha = expected_surface_alpha + contribution
        else:
            expected_volume_alpha = expected_volume_alpha + contribution
        transmittance = transmittance * (1.0 - alpha)

    torch.testing.assert_close(
        rgb[:, probe_y, probe_x], expected_rgb, atol=3e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        aux[7, probe_y, probe_x],
        expected_surface_alpha,
        atol=3e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        aux[8, probe_y, probe_x],
        expected_volume_alpha,
        atol=3e-6,
        rtol=2e-5,
    )
    mixed_objective = (
        rgb[:, probe_y, probe_x]
        * torch.tensor([0.7, 0.4, 0.2], device="cuda")
    ).sum() + 0.03 * aux[8, probe_y, probe_x]
    reference_objective = (
        expected_rgb * torch.tensor([0.7, 0.4, 0.2], device="cuda")
    ).sum() + 0.03 * expected_volume_alpha
    mixed_gradient = torch.autograd.grad(
        mixed_objective, opacity, retain_graph=True
    )[0]
    reference_gradient = torch.autograd.grad(reference_objective, opacity)[0]
    torch.testing.assert_close(
        mixed_gradient, reference_gradient, atol=3e-6, rtol=3e-5
    )


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


def test_low_opacity_volume_keeps_forward_and_backward_recovery_path():
    """A live parameter floor must not sit below the rasterizer cutoff."""
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings()
    volume = torch.tensor([[0.0, 0.0, 2.0]], device="cuda")
    opacity = torch.tensor([[1.0e-5]], device="cuda", requires_grad=True)
    rgb, radii, _, _, _ = MixedGaussianRasterizer(settings)(
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 2),
        _empty(0, 4),
        volume,
        torch.zeros_like(volume, requires_grad=True),
        torch.tensor([[0.25, 0.25, 0.25]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.8, 0.4, 0.2]], device="cuda"),
        opacity,
    )

    assert radii.item() > 0
    assert float(rgb.max()) > 0.0
    rgb.sum().backward()
    assert opacity.grad is not None
    assert float(opacity.grad.abs().sum()) > 0.0


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


def test_oblique_surfel_uses_exact_pixel_depth_within_one_tile():
    """A facade/volume order crossing inside one tile must be pixel exact."""
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=80, height=47)
    surface = torch.tensor([[0.0, 0.0, 2.5]], device="cuda")
    volume = torch.tensor([[0.0, 0.0, 2.5]], device="cuda")
    angle = torch.tensor(30.0 * torch.pi / 180.0, device="cuda")
    surface_rotation = torch.stack(
        [
            torch.cos(angle / 2.0),
            torch.zeros_like(angle),
            torch.sin(angle / 2.0),
            torch.zeros_like(angle),
        ]
    )[None]
    rgb, radii, aux, _, _ = MixedGaussianRasterizer(settings)(
        surface,
        torch.zeros_like(surface, requires_grad=True),
        torch.tensor([[0.7, 0.5]], device="cuda"),
        surface_rotation,
        volume,
        torch.zeros_like(volume, requires_grad=True),
        torch.tensor([[0.7, 0.7, 0.7]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.99], [0.99]], device="cuda"),
    )

    assert torch.all(radii > 0)
    y = 20
    volume_front_x = 32  # Both probes are in tile [32, 48).
    surface_front_x = 42  # tile [32, 48)
    assert volume_front_x // 16 == surface_front_x // 16
    for x in (volume_front_x, surface_front_x):
        assert aux[7, y, x] > 0.005
        assert aux[8, y, x] > 0.005
    surface_depth_left = aux[9, y, volume_front_x] / aux[7, y, volume_front_x]
    surface_depth_right = aux[9, y, surface_front_x] / aux[7, y, surface_front_x]
    assert surface_depth_left > 2.5
    assert surface_depth_right < 2.5
    assert rgb[0, y, volume_front_x] > 4.0 * rgb[1, y, volume_front_x]
    assert rgb[1, y, surface_front_x] > 4.0 * rgb[0, y, surface_front_x]


def test_exact_pixel_order_backward_matches_finite_difference():
    """Backward must replay the same order on both sides of a tile crossing."""
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=80, height=47)
    surface = torch.tensor([[0.0, 0.0, 2.5]], device="cuda")
    volume = torch.tensor([[0.0, 0.0, 2.5]], device="cuda")
    angle = torch.tensor(30.0 * torch.pi / 180.0, device="cuda")
    rotation = torch.stack(
        [
            torch.cos(angle / 2.0),
            torch.zeros_like(angle),
            torch.sin(angle / 2.0),
            torch.zeros_like(angle),
        ]
    )[None]
    opacity = torch.tensor(
        [[0.73], [0.67]], device="cuda", requires_grad=True
    )

    def render_loss():
        rgb, _, aux, _, _ = MixedGaussianRasterizer(settings)(
            surface,
            torch.zeros_like(surface, requires_grad=True),
            torch.tensor([[0.7, 0.5]], device="cuda"),
            rotation,
            volume,
            torch.zeros_like(volume, requires_grad=True),
            torch.tensor([[0.7, 0.7, 0.7]], device="cuda"),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], device="cuda"
            ),
            opacity,
        )
        # x=32 and x=42 share a tile but have opposite exact depth order.
        return (
            0.8 * rgb[0, 20, 32]
            + 0.6 * rgb[1, 20, 42]
            + 0.03 * aux[0, 20, 32]
            + 0.02 * aux[0, 20, 42]
        )

    render_loss().backward()
    analytic = opacity.grad.detach().clone().flatten()
    numeric = []
    epsilon = 1.0e-3
    with torch.no_grad():
        for row in range(2):
            original = opacity[row, 0].item()
            opacity[row, 0] = original + epsilon
            plus = render_loss().item()
            opacity[row, 0] = original - epsilon
            minus = render_loss().item()
            opacity[row, 0] = original
            numeric.append((plus - minus) / (2.0 * epsilon))
    for actual, expected in zip(analytic.tolist(), numeric):
        assert actual == pytest.approx(expected, rel=2e-2, abs=2e-4)


def test_volume_only_auxiliary_maps_are_intrinsic_layer_maps():
    """With the surface gate empty, mixed role maps equal the volume layer."""
    _, _, MixedGaussianRasterizer = _api()
    settings = _settings(width=35, height=29)
    volume = torch.tensor(
        [[-0.08, 0.02, 2.1], [0.11, -0.04, 2.8]], device="cuda"
    )
    rgb, _, aux, _, _ = MixedGaussianRasterizer(settings)(
        _empty(0, 3),
        _empty(0, 3),
        _empty(0, 2),
        _empty(0, 4),
        volume,
        torch.zeros_like(volume, requires_grad=True),
        torch.tensor([[0.3, 0.2, 0.25], [0.24, 0.31, 0.2]], device="cuda"),
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.97, 0.1, -0.15, 0.12]],
            device="cuda",
        ),
        torch.tensor([[0.2, 0.8, 0.1], [0.9, 0.15, 0.1]], device="cuda"),
        torch.tensor([[0.7], [0.6]], device="cuda"),
    )

    assert torch.count_nonzero(rgb).item() > 0
    assert torch.count_nonzero(aux[7]).item() == 0
    assert torch.count_nonzero(aux[9]).item() == 0
    assert torch.allclose(aux[8], aux[1], atol=2e-7, rtol=0)
    assert torch.allclose(aux[10], aux[0], atol=2e-7, rtol=0)


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
    # Checking only screen-x did not exercise depth or thin-leaf orientation.
    # Those are essential for escaping a wrong canopy depth hypothesis.
    tensors_and_indices = (
        [(xyz, (0, axis)) for axis in range(3)]
        + [(scale, (0, axis)) for axis in range(3)]
        + [(quat, (0, axis)) for axis in range(4)]
        + [(color, (0, axis)) for axis in range(3)]
        + [(opacity, (0, 0))]
    )
    analytic = [tensor.grad[index].item() for tensor, index in tensors_and_indices]
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
    locally_gated, _, _, primitive_audit, gate_audit = MixedGaussianRasterizer(settings)(
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
    # Channel zero is unconditional mass; supplied fields follow it. The
    # all-ones first field must reproduce mass at every local atlas node.
    assert torch.allclose(gate_audit[..., 0], gate_audit[..., 1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(gate_audit.sum((1, 2)), primitive_audit, atol=1e-3, rtol=3e-5)
    locally_gated.square().mean().backward()
    assert atlas.grad is not None
    assert torch.isfinite(atlas.grad).all()
    assert atlas.grad.abs().sum() > 0


def test_surface_median_does_not_bound_foreground_radiance():
    """An opaque leaf before the median cannot remove an earlier surface."""
    _, _, Rasterizer = _api()
    settings = _settings()
    surfaces = torch.tensor([[0.,0.,2.],[0.,0.,4.],[0.,0.,4.1]],device='cuda')
    quat = torch.tensor([[1.,0.,0.,0.]]*3,device='cuda')
    def render(depth, enabled):
        volume = torch.tensor([[0.,0.,depth]]*3,device='cuda')
        return Rasterizer(settings)(surfaces,torch.zeros_like(surfaces),
            torch.full((3,2),10.,device='cuda'),quat,
            volume,torch.zeros_like(volume),torch.full((3,3),10.,device='cuda'),quat,
            torch.cat((torch.ones(3,3,device='cuda'),torch.zeros(3,3,device='cuda'))),
            torch.tensor([.4,.99,.99]+([.999999]*3 if enabled else [0.]*3),device='cuda'))
    wall=render(3.,False)
    assert wall[2][5,20,34].item() == pytest.approx(4.,abs=1e-4)
    middle=render(3.,True)[0][:,20,34]
    near=render(1.,True)[0][:,20,34]
    assert (middle > .39).all()
    assert (near < 1e-4).all()
