import torch
import pytest
from types import SimpleNamespace
from scripts.calibrate_canopy_optical_ablation import (
    project_leaf_center_updates_,project_leaf_shape_updates_,source_ray_directions,apply_source_ray_update_,
    project_leaf_color_updates_,
    project_leaf_rotation_updates_,
)


def test_leaf_position_projection_preserves_inactive_and_bounds_active_rows():
    initial=torch.zeros(3,3)
    previous=torch.tensor([[.1,0.,0.],[.2,.3,.4],[.5,.2,.1]])
    proposed=torch.tensor([[10.,0.,0.],[5.,5.,5.],[.50001,.2,.1]])
    unchanged=proposed[2].clone()
    count=project_leaf_center_updates_(proposed,initial,previous,torch.tensor([True,False,True]),torch.ones(3))
    assert count==1
    assert torch.equal(proposed[0],torch.tensor([1.,0.,0.]))
    assert torch.equal(proposed[1],previous[1])
    assert torch.equal(proposed[2],unchanged)


def test_leaf_shape_projection_cannot_inflate_depth_or_change_inactive_rows():
    initial=torch.tensor([[.02,.04,.02],[.01,.03,.02]]).log()
    previous=initial.clone()
    proposed=initial+torch.tensor([[4.,-4.,3.],[2.,2.,2.]])
    project_leaf_shape_updates_(proposed,initial,previous,torch.tensor([True,False]))
    torch.testing.assert_close(proposed[0].exp(),torch.tensor([.03,.01,.02]))
    assert torch.equal(proposed[1],previous[1])


def test_source_ray_parameterization_preserves_observed_pixel_and_rejects_off_ray_input():
    view=SimpleNamespace(colmap_id=7,world_view_transform=torch.eye(4),camera_center=torch.zeros(3),
        focal_x=100.,focal_y=100.,cx=50.,cy=50.,image_width=100,image_height=100)
    initial=torch.tensor([[.1,.2,2.],[.3,.1,3.]])
    pixel=100*initial[:,:2]/initial[:,2:]+50
    observation=(pixel+.5)/100
    ray=source_ray_directions(initial,torch.tensor([7,7]),observation,[view])
    positions=initial.clone();displacement=torch.tensor([.4,.5])
    apply_source_ray_update_(positions,initial,ray,displacement,torch.zeros(2),torch.tensor([True,False]),torch.full((2,),.2))
    assert displacement.tolist()==pytest.approx([.2,0.])
    assert torch.equal(positions[1],initial[1])
    torch.testing.assert_close(100*positions[:,:2]/positions[:,2:]+50,pixel)
    wrong=initial.clone();wrong[0,0]+=.1
    with pytest.raises(ValueError,match='already aligned'):
        source_ray_directions(wrong,torch.tensor([7,7]),observation,[view])


def test_scalar_depth_chain_rule_matches_direct_parameterization():
    initial=torch.tensor([[.1,.2,2.],[.3,.1,3.]],dtype=torch.float64)
    ray=initial/initial.norm(dim=1,keepdim=True)
    delta=torch.tensor([.1,-.2],dtype=torch.float64,requires_grad=True)
    xyz=(initial+ray*delta[:,None]).detach().requires_grad_(True)
    weights=torch.tensor([[1.,2.,3.],[4.,2.,1.]],dtype=torch.float64)
    gradient=torch.autograd.grad((xyz.square()*weights).sum(),xyz)[0]
    expected=torch.autograd.grad(((initial+ray*delta[:,None]).square()*weights).sum(),delta)[0]
    torch.testing.assert_close((gradient*ray).sum(dim=1),expected,rtol=1.e-12,atol=1.e-12)


def test_directional_color_uses_actual_step_ratio_and_preserves_inactive_or_higher_bands():
    initial=torch.zeros(2,16,3,dtype=torch.float64);updated=torch.ones_like(initial)
    project_leaf_color_updates_(updated,initial,torch.tensor([True,False]),4)
    assert (updated[0,0]==1).all() and (updated[0,1:4]==.05).all()
    assert not updated[0,4:].any() and not updated[1].any()
    dc=torch.ones_like(initial)
    project_leaf_color_updates_(dc,initial,torch.tensor([True,False]),1)
    assert (dc[0,0]==1).all() and not dc[:,1:].any() and not dc[1].any()


def test_leaf_rotation_bound_normalizes_and_preserves_inactive():
    import math
    initial=torch.tensor([[1.,0,0,0]]*3,dtype=torch.float64)
    previous=initial.clone();proposed=torch.tensor([[0.,0,1,0],[-2.,0,0,0],[0.,1,0,0]],dtype=torch.float64)
    project_leaf_rotation_updates_(proposed,initial,previous,torch.tensor([True,True,False]))
    torch.testing.assert_close(proposed.norm(dim=1),torch.ones(3,dtype=torch.float64))
    assert proposed[0,0].item()==pytest.approx(math.cos(math.pi/8))
    assert torch.equal(proposed[1:],initial[1:])
