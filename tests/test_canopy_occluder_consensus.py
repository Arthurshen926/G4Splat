from types import SimpleNamespace
import torch
from scripts.audit_moge_canopy_occluder_consensus import unproject_camera_z


def test_depth_consensus_world_points_roundtrip_exact_intrinsics():
    transform=torch.eye(4,dtype=torch.float64)
    transform[:3,:3]=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
    transform[3,:3]=torch.tensor([2.,-3.,1.])
    view=SimpleNamespace(world_view_transform=transform,focal_x=330.,focal_y=270.,cx=1.3,cy=.7)
    depth=torch.tensor([[2.,3.,4.],[5.,6.,7.]],dtype=torch.float64)
    world=unproject_camera_z(depth,view)
    camera=world@transform[:3,:3]+transform[3,:3]
    y,x=torch.meshgrid(torch.arange(2),torch.arange(3),indexing='ij')
    torch.testing.assert_close(camera[...,2],depth)
    torch.testing.assert_close(view.focal_x*camera[...,0]/camera[...,2]+view.cx,x.to(depth),atol=1.e-12,rtol=0)
    torch.testing.assert_close(view.focal_y*camera[...,1]/camera[...,2]+view.cy,y.to(depth),atol=1.e-12,rtol=0)
