import pytest
import torch
from outdoor.canopy_occluder_loss import confirmed_occluder_loss,validate_consensus_coverage,OCCLUDER_CONSENSUS_CONTRACT


def test_confirmed_occluders_do_not_fill_unknown_or_rigid_pixels():
    alpha=torch.tensor([[.2,.3],[.4,.5]],requires_grad=True)
    mask=torch.tensor([[True,False],[False,True]])
    confirmed_occluder_loss(alpha,mask).backward()
    torch.testing.assert_close(alpha.grad,torch.tensor([[-2.5,0.],[0.,-1.]]))


def test_empty_occluder_evidence_has_zero_gradient():
    alpha=torch.zeros(2,2,requires_grad=True)
    loss=confirmed_occluder_loss(alpha,torch.zeros(2,2,dtype=torch.bool));loss.backward()
    assert loss==0 and not alpha.grad.any()


def test_partial_or_validation_evidence_cannot_be_used_for_training():
    manifest=dict(calibrated_training_views=[1,2,3],selected_training_views=[1,3],
        canopy_training_views=[1,3],all_canopy_training_views_covered=True,excluded_views=[4,5],contract=OCCLUDER_CONSENSUS_CONTRACT)
    validate_consensus_coverage(manifest,[1,2,3],[4,5])
    for change in [dict(selected_training_views=[1]),dict(calibrated_training_views=[1,2,3,4]),
                   dict(all_canopy_training_views_covered=False),dict(excluded_views=[]),dict(contract='geometry_only')]:
        with pytest.raises(ValueError):validate_consensus_coverage({**manifest,**change},[1,2,3],[4,5])


def test_source_candidate_ablation_is_explicit_and_retains_unknown_mask():
    from outdoor.canopy_occluder_loss import select_occluder_evidence
    e=dict(candidate=torch.tensor([[True,True,False]]),confirmed=torch.tensor([[True,False,False]]))
    assert select_occluder_evidence(e,'confirmed').sum()==1
    source=select_occluder_evidence(e,'source_candidate')
    assert source.sum()==2
    alpha=torch.full((1,3),.5,requires_grad=True)
    confirmed_occluder_loss(alpha,source).backward()
    assert alpha.grad[0,2]==0
    with pytest.raises(ValueError):select_occluder_evidence(e,'all_canopy')
    with pytest.raises(ValueError):select_occluder_evidence({**e,'confirmed':torch.ones_like(source)},'source_candidate')
