import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_spatial_footprint import SpatialFootprintCandidates


def make():
    return SpatialFootprintCandidates(CandidateCloud(torch.ones(2, 3),
        torch.full((2, 3), .04), torch.full((2, 3), .3)))


def test_size_and_position_have_independent_coordinates_and_fixed_bounds():
    model = make(); cloud = model.cloud
    assert torch.equal(model.xyz, cloud.xyz) and torch.equal(model.scales, cloud.scales)
    with torch.no_grad(): model.position_code.fill_(.5)
    xyz = model.xyz.detach().clone()
    with torch.no_grad(): model.scale_code.copy_(torch.tensor([10., -10.]))
    assert torch.equal(model.xyz, xyz)
    assert (model.scales >= cloud.scales*.5-1e-7).all()
    assert (model.scales <= cloud.scales*2+1e-7).all()
    assert torch.autograd.grad(model.xyz.sum(), model.scale_code, allow_unused=True)[0] is None
    assert torch.autograd.grad(model.scales.sum(), model.position_code, allow_unused=True)[0] is None
    model.xyz.sum().backward()
    assert model.position_code.grad.abs().sum() > 0
    assert torch.equal(cloud.xyz, torch.ones(2, 3))


def test_state_roundtrip_preserves_spatial_and_footprint_parameters():
    model = make(); other = make()
    with torch.no_grad(): model.position_code.fill_(.3); model.scale_code.fill_(-.4)
    other.load_state_dict(model.state_dict())
    assert torch.equal(model.xyz, other.xyz) and torch.equal(model.scales, other.scales)
