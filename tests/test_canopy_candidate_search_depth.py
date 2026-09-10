import torch
from scripts.canopy_candidate_search_depth import blocking_aware_search_depth


def test_search_center_moves_before_opaque_blocker_without_depth_truth_gradient():
    prior = torch.tensor([8., 3., 8., 8.], requires_grad=True)
    wall = torch.tensor([4., 4., 4., float('nan')], requires_grad=True)
    opacity = torch.tensor([.99, .99, .5, .99], requires_grad=True)
    value, changed = blocking_aware_search_depth(prior, wall, opacity)
    torch.testing.assert_close(value, torch.tensor([3.92, 3., 8., 8.]))
    assert changed.tolist() == [True, False, False, False]
    assert not value.requires_grad
    assert prior.grad is wall.grad is opacity.grad is None
