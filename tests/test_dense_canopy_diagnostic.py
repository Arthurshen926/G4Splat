from types import SimpleNamespace
import torch
from scripts.build_dense_canonical_canopy_diagnostic import exact_world_points


def test_dense_points_use_exact_k_and_translated_camera_depth():
    transform=torch.eye(4);transform[3,:3]=torch.tensor([-2.,1.,-3.])
    view=SimpleNamespace(world_view_transform=transform,focal_x=20.,focal_y=10.,cx=3.,cy=2.)
    uv=torch.tensor([[3.,2.],[5.,1.]])
    result=exact_world_points(uv,torch.tensor([4.,5.]),view)
    assert torch.allclose(result,torch.tensor([[2.,-1.,7.],[2.5,-1.5,8.]]))
