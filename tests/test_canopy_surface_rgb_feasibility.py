import pytest
import torch
from scripts.canopy_surface_rgb_feasibility import BlackRadianceView, surface_rgb_feasibility_loss


def test_bright_leaf_coupling_and_necessary_rgb_floor_have_opposite_opacity_gradients():
    alpha = torch.tensor(.2, requires_grad=True)
    target = torch.full((3, 1, 1), .1)
    surface = ((1-alpha)*.7).expand_as(target)
    rgb = surface+alpha*.8
    ordinary, = torch.autograd.grad((rgb-target).abs().mean(), alpha, retain_graph=True)
    floor, = torch.autograd.grad(surface_rgb_feasibility_loss(surface, target, torch.ones(1, 1, dtype=torch.bool)), alpha)
    assert ordinary.item() == pytest.approx(.1)
    assert floor.item() == pytest.approx(-.644)


def test_no_forced_thickness_when_floor_already_below_rgb_and_no_noncanopy_authority():
    surface = torch.full((3, 1, 2), .1, requires_grad=True)
    target = torch.tensor([.2, 0.]).reshape(1, 1, 2).expand_as(surface)
    canopy = torch.tensor([[True, False]])
    loss = surface_rgb_feasibility_loss(surface, target, canopy)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(surface.grad) == 0


def test_black_view_keeps_optical_gradient_but_cannot_change_color_or_fabricate_metadata():
    opacity = torch.tensor([.2], requires_grad=True); features = torch.ones(1, 16, 3, requires_grad=True)
    class View:
        def __len__(self): return 1
        def conditioned_state(self, *args, **kwargs): return torch.ones(1, 3), features, opacity
    view = BlackRadianceView(View())
    _, black, alpha = view.conditioned_state(None, include_dynamic=False)
    assert alpha is opacity and not black.requires_grad and len(view) == 1
    assert torch.equal(features, torch.ones_like(features))
    with pytest.raises(AttributeError): _ = view.verified_camera_count
