from types import SimpleNamespace
import torch
import pytest
from scripts.canopy_refresh_moge_observations import refresh_observations
from scripts.canopy_refresh_moge_observations import validate_refreshed_records,query_depth_validity
import numpy as np


def fixture():
    views=[]
    for center in [-.4,0.,.4]:
        transform=torch.eye(4);transform[3,0]=-center
        views.append(SimpleNamespace(world_view_transform=transform,camera_center=torch.tensor([center,0.,0.]),
            focal_x=10.,focal_y=10.,cx=4.,cy=4.,image_width=8,image_height=8))
    records={i:(torch.tensor([0]),torch.tensor([9.7]),torch.tensor([10.3])) for i in range(3)}
    return views,records


def test_refresh_requeries_depth_without_new_owners():
    views,records=fixture();xyz=torch.tensor([[0.,0.,10.],[0.,0.,10.]])
    result,audit=refresh_observations(xyz,views,records,lambda *_:(torch.full((8,8),10.1),torch.ones(8,8,dtype=torch.bool)))
    assert audit['after']==3 and audit['eligible_candidates']==1
    for ids,lo,hi in result.values():
        assert ids.tolist()==[0]
        assert torch.allclose((lo+hi)*.5,torch.tensor([10.1]))


def test_lost_view_cannot_keep_three_view_authority():
    views,records=fixture()
    result,audit=refresh_observations(torch.tensor([[0.,0.,10.]]),views,records,
        lambda i,_:(torch.full((8,8),12. if i==2 else 10.),torch.ones(8,8,dtype=torch.bool)))
    assert audit['after']==0 and all(len(r[0])==0 for r in result.values())


def test_refresh_never_uses_excluded_camera():
    views,records=fixture()
    with pytest.raises(ValueError,match='Excluded'):
        refresh_observations(torch.zeros(1,3),views,records,None,excluded=[2])


def test_resume_rejects_new_owner_and_accepts_subset():
    _,records=fixture();validate_refreshed_records(records,records)
    changed=dict(records);changed[0]=(torch.tensor([1]),torch.tensor([9.7]),torch.tensor([10.3]))
    with pytest.raises(ValueError,match='ownership'):validate_refreshed_records(records,changed)


def test_refresh_query_keeps_calibration_and_rejects_invalid_depth():
    views,_=fixture();v=views[0];v.image_name='a'
    depth=np.full((1,8,8),5.,dtype=np.float32);depth[0,4,4]=np.nan
    d,valid=query_depth_validity(v,lambda _:dict(tree_interior=torch.ones(8,8,dtype=torch.bool)),
        depth,dict(camera_order=['a']),SimpleNamespace(scale=2.,profiles={'a':1.1}),torch.device('cpu'))
    assert d[2,2]==11 and valid[2,2]
    assert not valid[4,4] and not valid[3,3]


def test_refreshed_targets_still_must_be_reachable():
    views,records=fixture();xyz=torch.tensor([[0.,0.,10.]])
    records={i:(ids,torch.tensor([9.99]),torch.tensor([10.61])) for i,(ids,_,_) in records.items()}
    query=lambda *_:(torch.full((8,8),10.6),torch.ones(8,8,dtype=torch.bool))
    _,unbounded=refresh_observations(xyz,views,records,query)
    _,bounded=refresh_observations(xyz,views,records,query,initial_xyz=xyz,motion_extent=torch.full_like(xyz,.1))
    assert unbounded['after']==3 and bounded['after']==0
