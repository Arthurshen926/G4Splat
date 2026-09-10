import torch
from scripts.audit_rigid_free_space import surfel_depth_extent


def test_plane_extent_rotates_into_camera_depth_and_encloses_center():
    transforms=torch.eye(4)[None].repeat(2,1,1)
    transforms[:,3,2]=10
    transforms[1,0,:3]=torch.tensor([0.,0.,2.])
    center,lower,upper=surfel_depth_extent(transforms,torch.eye(4))
    assert torch.equal(center[:,2],torch.tensor([10.,10.]))
    assert torch.equal(lower,torch.tensor([10.,-2.]))
    assert torch.equal(upper,torch.tensor([10.,22.]))
