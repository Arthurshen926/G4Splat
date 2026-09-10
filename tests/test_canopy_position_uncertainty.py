import math
import pytest
import torch
from outdoor.canopy_position_uncertainty import camera_ray_position_covariance,position_mahalanobis_error


def test_depth_uncertainty_does_not_come_from_leaf_thickness():
    uv=torch.tensor([[320.,180.]],dtype=torch.float64);z=torch.tensor([10.],dtype=torch.float64)
    covariance=camera_ray_position_covariance(uv,z,z*.01,torch.eye(3,dtype=z.dtype),fx=500.,fy=500.,cx=320.,cy=180.)
    assert covariance[0,2,2].item()==pytest.approx(1.)
    assert covariance[0,0,0].item()==pytest.approx(.04**2)
    # A half-metre depth adjustment is plausible, despite a2cm optical leaf.
    delta=torch.tensor([[0.,0.,.5]],dtype=z.dtype,requires_grad=True)
    loss=position_mahalanobis_error(delta,covariance).sum();loss.backward()
    assert loss.item()==pytest.approx(.25,rel=1e-6)
    assert delta.grad[0,2].item()==pytest.approx(1.,rel=1e-6)


def test_off_axis_ray_covariance_and_full_prior_are_rotation_invariant():
    uv=torch.tensor([[520.,80.]],dtype=torch.float64);z=torch.tensor([8.],dtype=torch.float64)
    a=.7;rotation=torch.tensor([[math.cos(a),0.,math.sin(a)],[0.,1.,0.],[-math.sin(a),0.,math.cos(a)]],dtype=z.dtype)
    kwargs=dict(fx=500.,fy=450.,cx=320.,cy=180.)
    camera=camera_ray_position_covariance(uv,z,z*.0125,torch.eye(3,dtype=z.dtype),**kwargs)
    world=camera_ray_position_covariance(uv,z,z*.0125,rotation,**kwargs)
    delta=torch.tensor([[.2,-.1,.4]],dtype=z.dtype)
    assert camera[0,0,2].abs()>.1
    assert torch.allclose(world,rotation[None]@camera@rotation.T[None])
    assert torch.allclose(position_mahalanobis_error(delta,camera),position_mahalanobis_error(delta@rotation.T,world),atol=1e-8)
    # Axis-diagonal approximation is demonstrably not the same objective.
    assert not torch.allclose(position_mahalanobis_error(delta,camera),(delta.square()/camera.diagonal(dim1=-2,dim2=-1)).sum(1))


def test_exact_position_has_zero_loss_and_gradient():
    delta=torch.zeros(2,3,requires_grad=True)
    position_mahalanobis_error(delta,torch.eye(3)[None].expand(2,3,3)).sum().backward()
    assert torch.equal(delta.grad,torch.zeros_like(delta))


def test_nonpositive_covariance_fails_closed():
    with pytest.raises(ValueError):position_mahalanobis_error(torch.ones(1,3),-torch.eye(3)[None])
