import torch
from scripts.audit_moge_position_reachability import depth_box_interval


def test_box_support_matches_all_corners():
    initial=torch.tensor([[2.,3.,4.]]);scales=torch.tensor([[.1,.2,.3]])
    axis=torch.tensor([.5,-.7,.2]);corners=torch.cartesian_prod(*[torch.tensor([-1.,1.])]*3)
    depths=(initial+8*scales*corners)@axis-2
    lo,hi=depth_box_interval(initial,scales,axis,-2,8)
    torch.testing.assert_close(lo[0],depths.min());torch.testing.assert_close(hi[0],depths.max())
