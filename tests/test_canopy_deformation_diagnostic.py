import torch
import pytest
from outdoor.canopy_deformation_diagnostic import BoundedCanopyCenterField
from outdoor.canopy_deformation_diagnostic import enable_shared_canopy_optics,project_shared_canopy_optics_
from types import SimpleNamespace


def field(rank=0):
    return BoundedCanopyCenterField(torch.tensor([[.2,.3,.4],[1.1,.8,.5],[2.,2.,2.]]),
        torch.tensor([True,True,False]),[0.,2.,4.],spacing=1.,radius=.1,rank=rank)


def test_zero_field_is_exact_identity_and_unknown_rows_stay_fixed():
    f=field(4)
    assert not f.offsets().any() and not f.offsets(1.).any()
    with torch.no_grad():f.base.fill_(100);f.basis.fill_(100)
    for time in (None,0.,1.,4.):
        delta=f.offsets(time)
        assert not delta[2].any()
        assert (delta.norm(dim=1)<=.100001).all()
    torch.testing.assert_close(f.weights.sum(1),torch.ones(2))


def test_heldout_time_interpolates_training_codes_without_new_parameter():
    f=field(2)
    with torch.no_grad():f.codes.copy_(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))
    torch.testing.assert_close(f.code_at(1.),torch.tensor([-1.,-1.]))
    assert f.codes.shape==(3,2)


def test_temporal_branch_cannot_modify_canonical_field_or_optics():
    f=field(4)
    before=f.offsets().detach().clone()
    loss=f.offsets(2.)[:,0].sum();loss.backward()
    assert f.basis.grad.abs().sum()>0
    with torch.no_grad():f.basis.add_(.2)
    torch.testing.assert_close(f.offsets(),before,rtol=0,atol=0)
    assert set(dict(f.named_parameters()))=={'base','basis','codes'}


def test_adam_changes_field_but_preserves_source_and_excluded_rows():
    source=torch.tensor([[.2,.3,.4],[1.1,.8,.5],[2.,2.,2.]])
    before=source.clone()
    f=BoundedCanopyCenterField(source,torch.tensor([True,True,False]),[0.,2.,4.],rank=8)
    optimizer=torch.optim.Adam(f.parameters(),lr=.02)
    loss=(f.offsets(2.)[0]-torch.tensor([.01,.02,.03])).square().sum()
    loss.backward();optimizer.step()
    assert f.offsets(2.)[0].norm()>0
    assert not f.offsets(2.)[2].any()
    torch.testing.assert_close(source,before,atol=0,rtol=0)
    assert f.offsets(2.).norm(dim=1).max()<=.100001


def test_shared_optics_adam_and_projection_never_modify_ineligible_rows():
    leaf=SimpleNamespace(features=torch.nn.Parameter(torch.ones(3,16,3)),
                         opacity_logits=torch.nn.Parameter(torch.zeros(3,1)))
    eligible=torch.tensor([True,False,True])
    before=(leaf.features[1].detach().clone(),leaf.opacity_logits[1].detach().clone())
    handles=enable_shared_canopy_optics(leaf,eligible)
    optimizer=torch.optim.Adam([leaf.features,leaf.opacity_logits],lr=.02)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (leaf.features.sum()+leaf.opacity_logits.sum()).backward()
        assert not leaf.features.grad[1].any() and not leaf.opacity_logits.grad[1].any()
        optimizer.step();project_shared_canopy_optics_(leaf,eligible)
    torch.testing.assert_close(leaf.features[1],before[0],rtol=0,atol=0)
    torch.testing.assert_close(leaf.opacity_logits[1],before[1],rtol=0,atol=0)
    assert (leaf.opacity_logits[eligible].sigmoid()<=.350001).all()
    assert (leaf.features[eligible,1:].flatten(1).norm(dim=1)<=4.00001).all()
    assert not torch.equal(leaf.features[0],before[0])


def test_refinement_projection_does_not_retire_initial_opacity_without_gradient():
    source=torch.logit(torch.tensor([[.6],[.2],[.8]]))
    leaf=SimpleNamespace(features=torch.nn.Parameter(torch.zeros(3,16,3)),
                         opacity_logits=torch.nn.Parameter(source.clone()))
    eligible=torch.tensor([True,True,False])
    project_shared_canopy_optics_(leaf,eligible,initial_opacity_logits=source)
    torch.testing.assert_close(leaf.opacity_logits,source,rtol=0,atol=0)
    with torch.no_grad():leaf.opacity_logits[0]-=.5;leaf.opacity_logits[1]+=10
    retired=leaf.opacity_logits[0].detach().clone()
    project_shared_canopy_optics_(leaf,eligible,initial_opacity_logits=source)
    torch.testing.assert_close(leaf.opacity_logits[0],retired,rtol=0,atol=0)
    assert leaf.opacity_logits[1].sigmoid()<=.350001
    torch.testing.assert_close(leaf.opacity_logits[2],source[2],rtol=0,atol=0)
    with torch.no_grad():leaf.opacity_logits[0]+=10
    project_shared_canopy_optics_(leaf,eligible,initial_opacity_logits=source)
    torch.testing.assert_close(leaf.opacity_logits[0],source[0],rtol=0,atol=0)


def test_physical_reference_is_exact_canonical_and_survives_state_restore():
    args=(torch.tensor([[.2,.3,.4],[1.1,.8,.5]]),torch.tensor([True,False]),[0.,2.,4.])
    f=BoundedCanopyCenterField(*args,rank=4,reference_time=2.)
    with torch.no_grad():
        f.base.normal_();f.basis.normal_();f.codes.normal_()
    torch.testing.assert_close(f.offsets(2.),f.offsets(),rtol=0,atol=0)
    assert not f.code_at(2.).any()
    assert not f.offsets(1.)[1].any()
    restored=BoundedCanopyCenterField(*args,rank=4)
    restored.load_state_dict(f.state_dict())
    torch.testing.assert_close(restored.offsets(2.),f.offsets(),rtol=0,atol=0)
    with pytest.raises(ValueError,match='observed training time'):
        BoundedCanopyCenterField(*args,rank=4,reference_time=1.)


def test_legacy_state_cannot_be_relabelled_as_a_physical_snapshot():
    legacy=field(4).state_dict();legacy.pop('reference_index')
    field(4).load_state_dict(legacy)
    anchored=field(4);anchored.reference_index.fill_(1)
    with pytest.raises(RuntimeError,match='reference_index'):
        anchored.load_state_dict(legacy)
