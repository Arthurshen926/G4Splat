import pytest
import torch
from scripts.canopy_color_gradient_ownership import backward_canopy_owned_colors


def test_building_error_cannot_repaint_foliage_but_still_controls_opacity():
    color = torch.tensor(.2, requires_grad=True)
    opacity = torch.tensor(.4, requires_grad=True)
    rgb = opacity*color+(1-opacity)*.9
    tree = (rgb-.3).square()
    building = 4*(rgb-.9).square()
    expected_color, = torch.autograd.grad(tree, color, retain_graph=True)
    expected_opacity, = torch.autograd.grad(tree+building, opacity, retain_graph=True)
    backward_canopy_owned_colors(tree, building, color_parameters=[color],
                                 optical_geometry_parameters=[opacity])
    torch.testing.assert_close(color.grad, expected_color)
    torch.testing.assert_close(opacity.grad, expected_opacity)


def test_owned_gradients_accumulate_and_keep_frozen_parameters_untouched():
    color = torch.tensor(.2, requires_grad=True)
    geometry = torch.tensor(.4, requires_grad=True)
    frozen = torch.tensor(.9)
    expected_color = expected_geometry = 0.
    for target in (.3, .5):
        rgb = geometry*color+(1-geometry)*frozen
        tree = (rgb-target).square(); background = (rgb-frozen).square()
        expected_color += torch.autograd.grad(tree/2, color, retain_graph=True)[0]
        expected_geometry += torch.autograd.grad((tree+background)/2, geometry, retain_graph=True)[0]
        backward_canopy_owned_colors(tree, background, color_parameters=[color],
                                     optical_geometry_parameters=[geometry], divisor=2)
    torch.testing.assert_close(color.grad, expected_color)
    torch.testing.assert_close(geometry.grad, expected_geometry)
    assert frozen.grad is None


def test_parameter_roles_cannot_overlap():
    p = torch.tensor(1., requires_grad=True)
    with pytest.raises(ValueError, match='Disjoint'):
        backward_canopy_owned_colors(p*p, p*p, color_parameters=[p], optical_geometry_parameters=[p])
