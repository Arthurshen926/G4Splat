from types import SimpleNamespace
import pytest
import torch
from scripts.canopy_observed_background_risk import observed_rgb_risk,visible_rigid_loss
from scripts.canopy_declared_training_objective import declared_objective
from scripts.canopy_frozen_reference_cache import FrozenReferenceCache


@pytest.mark.parametrize('value,expected,gradient',[(.1,0.,0.),(.2,0.,0.),(.3,.1,1.)])
def test_observed_risk_does_not_reward_better_background(value,expected,gradient):
    rgb=torch.full((3,1,1),value,requires_grad=True)
    reference=torch.full_like(rgb,.2,requires_grad=True);target=torch.zeros_like(rgb,requires_grad=True)
    loss=observed_rgb_risk(rgb,target,reference,torch.ones(1,1,dtype=torch.bool))
    torch.testing.assert_close(loss,torch.tensor(expected));loss.backward()
    torch.testing.assert_close(rgb.grad.sum(),torch.tensor(gradient))
    assert reference.grad is None and target.grad is None


def test_visible_guard_only_repels_added_occlusion_on_conservative_interior():
    reference=torch.zeros(3,1,3);target=reference.clone()
    regions=dict(rigid=torch.ones(1,3,dtype=torch.bool),hard=torch.tensor([[False,True,False]]))
    old=torch.tensor([[[.98,.98,.6]]]);leaf=torch.tensor([[[.1,.1,.1]]],requires_grad=True)
    loss=visible_rigid_loss(old*(1-leaf),old,reference,target,regions);loss.backward()
    assert leaf.grad[0,0,0]>0
    assert torch.equal(leaf.grad[0,0,1:],torch.zeros(2))


def test_declared_visibility_requires_both_native_contributions():
    args=SimpleNamespace(noncanopy_rgb_target='observed_risk',rigid_rgb_preservation_weight=3.,
        rigid_boundary_preservation_weight=0.,surface_rgb_feasibility_weight=0.,visible_rigid_alpha_weight=4.)
    rgb=torch.zeros(3,1,1);empty=torch.zeros(1,1,dtype=torch.bool)
    regions=dict(tree=empty,rigid=~empty,sky=empty,hard=empty)
    with pytest.raises(ValueError,match='contributions'):
        declared_objective(rgb,rgb,rgb,regions,args)


@pytest.mark.parametrize('width',[2,200])
def test_declared_boundary_risk_allows_improvement_and_avoids_area_dilution(width):
    args=SimpleNamespace(noncanopy_rgb_target='observed_risk',rigid_rgb_preservation_weight=0.,
        rigid_boundary_preservation_weight=1.,surface_rgb_feasibility_weight=0.,visible_rigid_alpha_weight=0.)
    reference=torch.full((3,1,width),.2,requires_grad=True)
    target=torch.zeros_like(reference,requires_grad=True)
    hard=torch.zeros((1,width),dtype=torch.bool);hard[0,0]=True
    regions=dict(tree=torch.zeros_like(hard),rigid=torch.ones_like(hard),sky=torch.zeros_like(hard),hard=hard)
    better=torch.full_like(reference,.1,requires_grad=True)
    _,boundary,_=declared_objective(better,target,reference,regions,args)
    assert boundary==0
    worse=torch.full_like(reference,.3,requires_grad=True)
    _,boundary,_=declared_objective(worse,target,reference,regions,args)
    torch.testing.assert_close(boundary,torch.tensor(.1))
    boundary.backward()
    torch.testing.assert_close(worse.grad[:,:,0].sum(),torch.tensor(1.))
    assert worse.grad[:,:,1:].count_nonzero()==0
    assert reference.grad is None and target.grad is None


def test_single_channel_reference_cache_is_immutable():
    cache=FrozenReferenceCache(2,channels=1);source=torch.ones(1,2,2)
    first=cache.get('a',lambda:source,'cpu');first.zero_()
    torch.testing.assert_close(cache.get('a',lambda:source*0,'cpu'),source)


def test_native_visibility_guard_has_front_only_opacity_gradient():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('native_risk_helpers',Path(__file__).with_name('test_native_mixed_rasterizer.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    _,_,Rasterizer=h._api();settings=h._settings()
    wall=torch.tensor([[0.,0.,3.]],device='cuda');q=torch.tensor([[1.,0.,0.,0.]],device='cuda')
    def surface(peak,z):
        leaf=torch.tensor([[0.,0.,z]],device='cuda')
        return Rasterizer(settings)(wall,torch.zeros_like(wall),torch.ones(1,2,device='cuda'),q,
            leaf,torch.zeros_like(leaf),torch.full((1,3),.2,device='cuda'),q,
            torch.zeros(2,3,device='cuda'),torch.stack((peak.new_tensor(.999),peak)))[2][7:8]
    baseline=surface(torch.tensor(0.,device='cuda'),2.).detach()
    reference=torch.zeros(3,*baseline.shape[1:],device='cuda')
    regions=dict(rigid=torch.ones(baseline.shape[1:],device='cuda',dtype=torch.bool),
                 hard=torch.zeros(baseline.shape[1:],device='cuda',dtype=torch.bool))
    assert (baseline>=.95).any()
    for z in (2.,4.):
        peak=torch.tensor(.1,device='cuda',requires_grad=True)
        loss=visible_rigid_loss(surface(peak,z),baseline,reference,reference,regions)
        loss.backward()
        if z==2.:
            assert peak.grad>0
            fd=(visible_rigid_loss(surface(peak.detach()+.001,z),baseline,reference,reference,regions)
                -visible_rigid_loss(surface(peak.detach()-.001,z),baseline,reference,reference,regions))/.002
            torch.testing.assert_close(peak.grad,fd,atol=1e-5,rtol=.005)
        else:
            assert loss==0 and peak.grad==0
