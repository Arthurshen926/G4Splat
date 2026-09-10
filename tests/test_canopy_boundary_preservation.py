import pytest
import torch
from scripts.canopy_boundary_preservation import boundary_rgb_preservation


def test_boundary_gradient_is_not_diluted_by_unaffected_building_pixels():
    gradients = []
    for width in (2, 200):
        rgb = torch.full((3, 1, width), .4, requires_grad=True)
        reference = torch.full_like(rgb, .7, requires_grad=True)
        rigid = torch.ones(1, width, dtype=torch.bool)
        boundary = torch.zeros_like(rigid); boundary[0, 0] = True
        loss = boundary_rgb_preservation(rgb, reference, rigid, boundary)
        loss.backward(); gradients.append(rgb.grad[:, 0, 0])
        assert reference.grad is None
        assert rgb.grad[:, :, 1:].count_nonzero() == 0
    assert torch.equal(*gradients)


def test_empty_boundary_is_finite_zero_and_tree_authority_is_rejected():
    rgb = torch.ones(3, 2, 2, requires_grad=True)
    rigid = torch.zeros(2, 2, dtype=torch.bool)
    loss = boundary_rgb_preservation(rgb, torch.zeros_like(rgb), rigid, rigid)
    loss.backward()
    assert loss == 0 and rgb.grad.count_nonzero() == 0
    with pytest.raises(ValueError, match='rigid side'):
        boundary_rgb_preservation(rgb, rgb, rigid, ~rigid)
