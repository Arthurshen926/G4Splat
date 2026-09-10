import torch
from outdoor.canopy_source_depth_transform import source_depth_transform, source_depth_chain_gradient


def test_source_depth_transform_preserves_bearing_footprint_and_nonowners():
    xyz=torch.tensor([[1.,2.,10.],[-2.,1.,8.],[2.,3.,12.]],dtype=torch.float64)
    scales=torch.full_like(xyz,.05).log()
    origins=torch.tensor([[0.,0.,0.],[1.,0.,0.]],dtype=xyz.dtype)
    groups=torch.tensor([0,1,0]);owner=torch.tensor([True,True,False])
    factors=torch.tensor([.2,-.15],dtype=xyz.dtype)
    moved,changed=source_depth_transform(xyz,scales,origins,groups,factors,owner)
    before=xyz-origins[groups];after=moved-origins[groups]
    torch.testing.assert_close(after[:,:2]/after[:,2:],before[:,:2]/before[:,2:])
    torch.testing.assert_close(changed.exp()/after[:,2:],scales.exp()/before[:,2:])
    torch.testing.assert_close(moved[~owner],xyz[~owner],atol=0,rtol=0)
    torch.testing.assert_close(changed[~owner],scales[~owner],atol=0,rtol=0)


def test_grouped_depth_chain_rule_matches_autograd_including_scale_term():
    xyz=torch.tensor([[1.,2.,10.],[-2.,1.,8.],[2.,3.,12.]],dtype=torch.float64)
    scales=torch.full_like(xyz,.05).log()
    origins=torch.tensor([[0.,0.,0.],[1.,0.,0.]],dtype=xyz.dtype)
    groups=torch.tensor([0,1,0]);owner=torch.tensor([True,True,False])
    factors=torch.tensor([.1,-.1],dtype=xyz.dtype,requires_grad=True)
    moved,changed=source_depth_transform(xyz,scales,origins,groups,factors,owner)
    loss=moved.square().sum()+changed.square().sum()
    direct=torch.autograd.grad(loss,factors)[0]
    manual=source_depth_chain_gradient(2*moved,2*changed,moved,origins,groups,2,owner)
    torch.testing.assert_close(manual,direct)
