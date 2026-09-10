from copy import deepcopy
import pytest
import torch
from outdoor.moge3_prehit_authority import observed_rigid_prehit_authority
from scripts.train_unified_outdoor_teacher import _moge3_canopy_optical_loss,_resume_training_contract_differences


def test_rigid_authority_excludes_canopy_mixed_unknown_and_nonopaque():
    task=dict(p_rigid=torch.tensor([[1.,0.,.8,1.,1.]]),p_canopy=torch.tensor([[0.,1.,.2,0.,0.]]))
    value=observed_rigid_prehit_authority(task,torch.tensor([[2.,2.,2.,float('nan'),2.]]),
                                        torch.tensor([[1.,1.,1.,1.,.5]]))
    assert value.tolist()==[[1.,0.,0.,0.,0.]]


def test_remove_canopy_negative_without_changing_wall_or_hit_gradients():
    task=dict(p_rigid=torch.tensor([[1.,0.]]),p_canopy=torch.tensor([[0.,1.]]),
              p_canopy_core=torch.tensor([[0.,1.]]))
    gradients=[]
    for authority in [None,torch.tensor([[1.,0.]])]:
        pre=torch.tensor([[.2,.2]],requires_grad=True);hit=torch.tensor([[.1,.1]],requires_grad=True)
        loss,_=_moge3_canopy_optical_loss(pre,hit,task,torch.ones(1,2,dtype=torch.bool),torch.ones(1,2),
                                         prehit_authority=authority)
        gradients.append(torch.autograd.grad(loss,(pre,hit)))
    assert gradients[0][0][0,1]>0 and gradients[1][0][0,1]==0
    assert torch.equal(gradients[0][0][:,:1],gradients[1][0][:,:1])
    assert torch.equal(gradients[0][1],gradients[1][1])


def test_resume_requires_explicit_negative_domain_change_and_preserves_other_fields():
    old={'moge3_canopy_optical':dict(contract='same',weight=.2,prehit_weight=.25),'seed':1701}
    original=deepcopy(old);new=deepcopy(old);new['moge3_canopy_optical']['prehit_authority']='observed_rigid'
    assert 'moge3_canopy_optical' in _resume_training_contract_differences(old,new)
    assert not _resume_training_contract_differences(old,new,allow_moge3_prehit_authority_ablation=True)
    assert old==original
    legacy=deepcopy(old);legacy['moge3_canopy_optical']['prehit_authority']='legacy_moge'
    assert not _resume_training_contract_differences(old,legacy)
    new['seed']=1702
    assert 'seed' in _resume_training_contract_differences(old,new,allow_moge3_prehit_authority_ablation=True)
    new['moge3_canopy_optical']['weight']=.3
    with pytest.raises(RuntimeError,match='only its negative domain'):
        _resume_training_contract_differences(old,new,allow_moge3_prehit_authority_ablation=True)
