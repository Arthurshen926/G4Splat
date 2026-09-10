import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_candidate_footprint import FootprintCandidates


def test_footprint_can_shrink_or_grow_without_moving_or_hardening_points():
    xyz = torch.ones(2, 3); scales = torch.ones(2, 3)*.04
    cloud = CandidateCloud(xyz, scales, xyz*.3); model = FootprintCandidates(cloud)
    assert torch.equal(model.scales, scales)
    with torch.no_grad(): model.scale_code.copy_(torch.tensor([-2., 2.]))
    assert model.ratio[0] < 1 < model.ratio[1]
    assert (model.ratio > .5).all() and (model.ratio < 2).all()
    assert torch.equal(model.xyz, xyz)
    torch.testing.assert_close(model.logits.sigmoid(), torch.full((2,), .001))
    model.scales.sum().backward()
    assert (model.scale_code.grad > 0).all()
    assert cloud.logits.grad is None
