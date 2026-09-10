import pytest
import torch
from scripts.canopy_relative_radiance_feasibility import (
    initial_gradient_normalization, relative_surface_rgb_feasibility_loss)
from scripts.canopy_surface_rgb_feasibility import surface_rgb_feasibility_loss


def test_initial_gradient_total_matches_but_relative_distribution_changes():
    surface = torch.tensor([[[.15, .8]]]*3, requires_grad=True)
    target = torch.tensor([[[.05, .7]]]*3, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    scale = initial_gradient_normalization(surface, target, mask)
    absolute, = torch.autograd.grad(surface_rgb_feasibility_loss(surface, target, mask), surface)
    loss = relative_surface_rgb_feasibility_loss(surface, target, mask, scale)
    loss.backward()
    torch.testing.assert_close(surface.grad.abs().sum(), absolute.abs().sum())
    assert surface.grad[0, 0, 0] > surface.grad[0, 0, 1]
    assert target.grad is None


def test_only_positive_excess_inside_canopy_has_gradient():
    surface = torch.tensor([[[.2, .1, .8]]]*3, requires_grad=True)
    target = torch.tensor([[[.1, .2, .1]]]*3)
    mask = torch.tensor([[True, True, False]])
    scale = initial_gradient_normalization(surface, target, mask)
    relative_surface_rgb_feasibility_loss(surface, target, mask, scale).backward()
    assert (surface.grad[:, 0, 0] > 0).all()
    assert torch.equal(surface.grad[:, 0, 1:], torch.zeros_like(surface.grad[:, 0, 1:]))


def test_initial_feasibility_does_not_disable_later_negative_evidence():
    mask = torch.ones(1, 1, dtype=torch.bool); target = torch.full((3, 1, 1), .1)
    scale = initial_gradient_normalization(torch.zeros_like(target), target, mask)
    surface = torch.full_like(target, .2, requires_grad=True)
    relative_surface_rgb_feasibility_loss(surface, target, mask, scale).backward()
    assert (surface.grad > 0).all()


def test_empty_mask_and_invalid_normalization():
    surface = torch.ones(3, 1, 1, requires_grad=True); target = torch.zeros_like(surface)
    mask = torch.zeros(1, 1, dtype=torch.bool)
    scale = initial_gradient_normalization(surface, target, mask)
    loss = relative_surface_rgb_feasibility_loss(surface, target, mask, scale)
    loss.backward(); assert not surface.grad.any()
    with pytest.raises(ValueError): relative_surface_rgb_feasibility_loss(surface, target, mask, 0.)
