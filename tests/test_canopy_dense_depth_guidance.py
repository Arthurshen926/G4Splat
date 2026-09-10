import torch
from scripts.canopy_dense_depth_guidance import interval_log_depth_loss


def test_interval_guide_moves_depth_not_opacity_and_ignores_unknown():
    depth=torch.tensor([2.,3.,4.,5.],requires_grad=True)
    target=torch.tensor([2.,2.,2.,2.])
    alpha=torch.tensor([.5,.5,0.,.5],requires_grad=True)
    known=torch.tensor([True,True,True,False])
    loss,count=interval_log_depth_loss(depth,target,alpha,known)
    assert count==2
    loss.backward()
    assert depth.grad[1]>0
    assert torch.equal(depth.grad[[0,2,3]],torch.zeros(3))
    assert alpha.grad is None


def test_unknown_depth_does_not_create_positive_material():
    depth=torch.tensor([3.],requires_grad=True)
    loss,count=interval_log_depth_loss(depth,torch.tensor([2.]),torch.zeros(1),torch.ones(1,dtype=torch.bool))
    loss.backward()
    assert count==0 and loss==0 and depth.grad.item()==0
