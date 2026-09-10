import torch
from scripts.calibrate_canopy_optical_ablation import uncertainty_hit_bounds


def test_uncertainty_envelope_clips_before_opaque_rigid_and_rejects_empty():
    pre = torch.tensor([[[8.,8.,8.]],[[10.,10.,10.]]])
    thin = torch.tensor([[[9.6,9.6,9.6]],[[10.4,10.4,10.4]]])
    depth = torch.tensor([[11.,7.,0.]])
    alpha = torch.tensor([[1.,1.,0.]])
    bounds = uncertainty_hit_bounds(pre,thin,depth,alpha)
    torch.testing.assert_close(bounds[:,0,0],torch.tensor([8.,10.89]))
    assert torch.isnan(bounds[:,0,1]).all()
    torch.testing.assert_close(bounds[:,0,2],torch.tensor([8.,12.5]))
    assert torch.equal(pre,torch.tensor([[[8.,8.,8.]],[[10.,10.,10.]]]))
