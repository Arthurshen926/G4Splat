import torch
import numpy as np
from types import SimpleNamespace
from scripts.canopy_moge_position_supervision import interval_position_loss


def test_geometry_only_gradient_and_interval_dead_zone():
    xyz=torch.tensor([[0.,0.,1.],[0.,0.,2.],[0.,0.,4.]],requires_grad=True)
    lo=torch.tensor([1.5]*3,requires_grad=True);hi=torch.tensor([2.5]*3,requires_grad=True)
    loss=interval_position_loss(xyz,torch.eye(4),torch.arange(3),lo,hi);loss.backward()
    assert xyz.grad[0,2]<0 and xyz.grad[1,2]==0 and xyz.grad[2,2]>0
    assert (xyz.grad[:,:2]==0).all()
    assert lo.grad is None and hi.grad is None


def test_empty_observations_keep_zero_geometry_gradient():
    xyz=torch.ones(3,3,requires_grad=True)
    interval_position_loss(xyz,torch.eye(4),torch.tensor([],dtype=torch.long),torch.tensor([]),torch.tensor([])).backward()
    assert (xyz.grad==0).all()


def test_unreachable_observations_cannot_authorize_position_support():
    from scripts.canopy_moge_position_supervision import build_position_observations
    views=[]
    for i,x in enumerate([-.4,0.,.4]):
        t=torch.eye(4);t[3,0]=-x
        views.append(SimpleNamespace(image_name=str(i),world_view_transform=t,camera_center=torch.tensor([x,0.,0.]),
                     focal_x=10.,focal_y=10.,cx=4.,cy=4.,image_width=8,image_height=8))
    kwargs=dict(xyz=torch.tensor([[0.,0.,10.]]),views=views,train=[0,1,2],excluded=[3],
                regions=lambda _:dict(tree_interior=torch.ones(8,8,dtype=torch.bool)),
                raw=np.full((3,8,8),10.8,dtype=np.float32),cache={'camera_order':['0','1','2']},
                cal=SimpleNamespace(scale=1.,profiles={'0':1.,'1':1.,'2':1.}))
    _,legacy=build_position_observations(**kwargs)
    _,filtered=build_position_observations(**kwargs,motion_extent=torch.full((1,3),.1))
    assert legacy['eligible_candidates']==1
    assert filtered['eligible_candidates']==0
    assert filtered['rejected_unreachable_pre_sampling']==3
