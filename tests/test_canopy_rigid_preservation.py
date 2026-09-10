import torch
from scripts.calibrate_canopy_center_field import rigid_extinction_increase_loss


def test_initial_model_has_no_forced_retirement_gradient():
    alpha=torch.full((1,7,7),.2,requires_grad=True)
    baseline=torch.full((7,7),.2)
    loss=rigid_extinction_increase_loss(alpha,baseline,torch.ones(7,7,dtype=torch.bool))
    loss.backward()
    assert loss.item()==0
    assert not alpha.grad.any()


def test_only_new_spill_is_penalized_and_reference_is_detached():
    for current,expected in ((.1,False),(.3,True)):
        alpha=torch.full((1,7,7),current,requires_grad=True)
        baseline=torch.full((7,7),.2,requires_grad=True)
        loss=rigid_extinction_increase_loss(alpha,baseline,torch.ones(7,7,dtype=torch.bool))
        loss.backward()
        assert bool((alpha.grad>0).any())==expected
        assert baseline.grad is None
