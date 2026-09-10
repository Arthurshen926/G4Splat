import pytest
import torch
from outdoor.rowwise_optimizer_step import rescale_adam_row_displacement


def test_row_lr_matches_two_independent_adams_and_preserves_moments():
    shared=torch.nn.Parameter(torch.tensor([[.1],[-2.]],dtype=torch.float64))
    low=torch.nn.Parameter(shared[0:1].detach().clone())
    high=torch.nn.Parameter(shared[1:2].detach().clone())
    optimizer=torch.optim.Adam([shared],lr=.002)
    low_optimizer=torch.optim.Adam([low],lr=.002)
    high_optimizer=torch.optim.Adam([high],lr=.02)
    for step in range(20):
        gradient=torch.tensor([[(-1.)**step*.3],[.8/(step+1)]],dtype=torch.float64)
        shared.grad=gradient.clone();low.grad=gradient[0:1].clone();high.grad=gradient[1:2].clone()
        before=shared.detach().clone()
        optimizer.step();low_optimizer.step();high_optimizer.step()
        state_before={k:v.clone() for k,v in optimizer.state[shared].items() if torch.is_tensor(v)}
        rescale_adam_row_displacement(shared,before,torch.tensor([False,True]),10.)
        torch.testing.assert_close(shared,torch.cat((low,high)),atol=1e-12,rtol=1e-12)
        for k,v in state_before.items(): assert torch.equal(v,optimizer.state[shared][k])


def test_identity_is_bitwise_noop_and_invalid_multiplier_fails():
    parameter=torch.nn.Parameter(torch.tensor([[.1],[.2]]))
    original=parameter.detach().clone()
    rescale_adam_row_displacement(parameter,torch.zeros_like(parameter),torch.tensor([True,True]),1.)
    assert torch.equal(parameter,original)
    for invalid in (0.,-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):
            rescale_adam_row_displacement(parameter,original,torch.tensor([True,True]),invalid)
