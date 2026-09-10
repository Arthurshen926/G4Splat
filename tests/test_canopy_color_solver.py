import torch
from outdoor.canopy_color_solver import majorized_color_update


def test_joint_color_update_decreases_fixed_transport_least_squares():
    generator=torch.Generator().manual_seed(27)
    transport=torch.rand(30,8,generator=generator,dtype=torch.float64)
    transport=transport/transport.sum(dim=1,keepdim=True)*.8
    colors=torch.rand(8,3,generator=generator,dtype=torch.float64)
    target=torch.rand(30,3,generator=generator,dtype=torch.float64)
    background=torch.rand(30,3,generator=generator,dtype=torch.float64)*.2
    weight=torch.rand(30,generator=generator,dtype=torch.float64)
    eligible=torch.tensor([True]*6+[False]*2)
    for _ in range(8):
        residual=target-(transport@colors+background)
        numerator=transport.T@(weight[:,None]*residual)
        denominator=transport.T@(weight*transport.sum(dim=1))
        updated,_=majorized_color_update(colors,numerator,denominator,eligible,torch.ones_like(colors))
        old_loss=(weight[:,None]*residual.square()).sum()
        new_loss=(weight[:,None]*(target-transport@updated-background).square()).sum()
        assert new_loss<=old_loss+1.e-12
        assert torch.equal(updated[~eligible],colors[~eligible])
        colors=updated


def test_unobserved_leaf_cannot_receive_a_color_update():
    current=torch.full((2,3),.3)
    new,active=majorized_color_update(current,torch.ones_like(current),torch.tensor([0.,1.]),torch.tensor([True,False]),torch.ones_like(current))
    assert torch.equal(new,current)
    assert not active.any()
