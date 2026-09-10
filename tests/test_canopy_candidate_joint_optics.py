import pytest
import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud
from scripts.canopy_candidate_joint_optics import JointOpticalCandidateView


class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Parameter(torch.ones(2, 1, 3))
        self.opacity_logits = torch.nn.Parameter(torch.zeros(2, 1))
        self.xyz = torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
        self.dynamic_leaf_mask = torch.zeros(2, dtype=torch.bool)
        self.persistent_envelope_mask = torch.zeros(2, dtype=torch.bool)
        self.static_leaf_mask = torch.ones(2, dtype=torch.bool)
        self.sh_degree = 0


def test_joint_optics_masks_source_rows_and_rejects_geometry_training():
    base = Source(); candidate = CandidateCloud(torch.ones(1, 3), torch.ones(1, 3), torch.ones(1, 3)*.3)
    adapter = JointOpticalCandidateView(base, candidate, torch.tensor([True, False]))
    (base.features.sum()+base.opacity_logits.sum()).backward()
    assert torch.count_nonzero(base.features.grad[1]) == 0
    assert base.opacity_logits.grad[1] == 0 and base.opacity_logits.grad[0] == 1
    assert base.xyz.grad is None
    with pytest.raises(AttributeError): _ = adapter.verification_state
    base.xyz.requires_grad_(True)
    with pytest.raises(ValueError, match='Only source features'): JointOpticalCandidateView(base, candidate, torch.ones(2, dtype=torch.bool))


def test_joint_spatial_updates_keep_source_geometry_and_unauthorized_optics_exact():
    from scripts.canopy_candidate_spatial import SpatialCandidates
    from scripts.canopy_directional_sh_step import rescale_directional_sh_step_
    base = Source()
    base.features = torch.nn.Parameter(torch.ones(2, 4, 3))
    candidate = SpatialCandidates(CandidateCloud(torch.ones(1, 3),
        torch.full((1, 3), .04), torch.full((1, 3), .3)), 8.)
    eligible = torch.tensor([True, False])
    adapter = JointOpticalCandidateView(base, candidate, eligible)
    original_xyz = base.xyz.detach().clone()
    original_features = base.features.detach().clone()
    original_opacity = base.opacity_logits.detach().clone()
    previous_rest = base.features[eligible, 1:].detach().clone()
    optimizer = torch.optim.Adam([base.features, base.opacity_logits, *candidate.parameters()], lr=.01)
    loss = (base.features.sum()+base.opacity_logits.sum()+adapter.candidates.xyz.sum()
            +candidate.dc.sum()+candidate.logits.sum())
    loss.backward(); optimizer.step()
    rescale_directional_sh_step_(base.features, previous_rest, eligible, .05)
    assert torch.equal(base.xyz, original_xyz) and base.xyz.grad is None
    assert torch.equal(base.features[~eligible], original_features[~eligible])
    assert torch.equal(base.opacity_logits[~eligible], original_opacity[~eligible])
    assert not torch.equal(candidate.xyz, candidate.cloud.xyz)
    assert not torch.equal(base.features[eligible], original_features[eligible])
