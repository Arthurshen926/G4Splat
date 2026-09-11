import torch
from scripts.canopy_observation_validity import fixed_ray_validity


def test_moved_point_cannot_keep_old_ray_depth():
    xyz=torch.tensor([[0.,0.,1.],[.2,0.,1.],[0.,0.,-1.],[float('nan'),0.,1.]])
    mask=fixed_ray_validity(xyz,torch.eye(4),torch.tensor([[3.5,3.5]]).expand(4,2),
                            [10.,10.],[4.,4.],(8,8))
    assert mask.tolist()==[True,False,False,False]


def test_motion_along_same_ray_remains_valid():
    xyz=torch.tensor([[.1,.1,1.],[.2,.2,2.]],requires_grad=True)
    mask=fixed_ray_validity(xyz,torch.eye(4),torch.tensor([[4.5,4.5]]).expand(2,2),
                            [10.,10.],[4.,4.],(8,8))
    assert mask.all() and not mask.requires_grad
