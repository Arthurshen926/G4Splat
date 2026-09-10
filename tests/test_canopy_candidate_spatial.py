import pytest
import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_candidate_spatial import SpatialCandidates


def test_static_displacement_starts_exact_and_cannot_harden_or_grow():
    xyz = torch.tensor([[30., 1., -3.], [-2., 7., 6.]])
    scales = torch.full_like(xyz, .04)
    cloud = CandidateCloud(xyz, scales, torch.full_like(xyz, .3))
    model = SpatialCandidates(cloud, 8.)
    assert torch.equal(model.xyz, xyz)
    model.xyz.sum().backward()
    torch.testing.assert_close(model.position_code.grad, scales*8)
    assert cloud.logits.grad is None and cloud.dc.grad is None
    with torch.no_grad(): model.position_code.copy_(torch.tensor([[-2., 2., 0.], [2., -2., 0.]]))
    displacement = model.xyz-xyz
    assert (displacement.abs() <= scales*8+1e-6).all()
    assert displacement[0, 0] < 0 < displacement[0, 1]
    assert torch.equal(model.scales, scales)
    torch.testing.assert_close(model.logits.sigmoid(), torch.full((2,), .001))
    assert torch.equal(cloud.xyz, xyz)
    with pytest.raises(ValueError): SpatialCandidates(cloud, 17)
