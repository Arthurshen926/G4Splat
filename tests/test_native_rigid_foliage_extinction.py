import torch
import pytest
from outdoor.canopy_detail_loss import native_rigid_foliage_extinction_loss


def test_only_observed_rigid_interiors_provide_geometry_negative_evidence():
    alpha=torch.full((1,10,10),.2,requires_grad=True)
    rigid=torch.zeros(10,10,dtype=torch.bool);rigid[2:8,2:8]=True
    native_rigid_foliage_extinction_loss(alpha,rigid).backward()
    expected=torch.zeros_like(rigid);expected[3:7,3:7]=True
    assert torch.equal(alpha.grad[0]>0,expected)


def test_opaque_wall_prevents_hidden_leaf_from_receiving_free_space_gradient():
    opacity=torch.tensor(.6,requires_grad=True)
    native=opacity*torch.zeros(1,8,8)
    loss=native_rigid_foliage_extinction_loss(native,torch.ones(8,8,dtype=torch.bool))
    loss.backward();assert loss.item()==0 and opacity.grad.item()==0


def test_no_rigid_interior_is_differentiable_zero_and_bad_shape_fails():
    alpha=torch.full((8,8),.4,requires_grad=True)
    loss=native_rigid_foliage_extinction_loss(alpha,torch.zeros(8,8,dtype=torch.bool))
    loss.backward();assert loss.item()==0 and torch.count_nonzero(alpha.grad)==0
    with pytest.raises(ValueError):native_rigid_foliage_extinction_loss(alpha,torch.ones(3,8,8))
