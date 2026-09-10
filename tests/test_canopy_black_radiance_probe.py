from types import SimpleNamespace
import torch
from scripts.canopy_candidate_diagnostic import CandidateCloud, NativeCandidateView
from scripts.canopy_black_radiance_probe import BlackRadianceCandidateView


def test_black_probe_keeps_geometry_opacity_and_source_tensors_unchanged():
    xyz = torch.ones(2, 3); features = torch.rand(2, 16, 3); opacity = torch.tensor([.2, .7])
    saved = features.clone()
    class Base(SimpleNamespace):
        def __len__(self): return 2
    base = Base(dynamic_leaf_mask=torch.zeros(2, dtype=torch.bool),
                persistent_envelope_mask=torch.zeros(2, dtype=torch.bool),
                parameters=lambda: [], sh_degree=3,
                conditioned_state=lambda *args, **kwargs: (xyz, features, opacity))
    cloud = CandidateCloud(torch.ones(1, 3)*2, torch.ones(1, 3), torch.ones(1, 3)*.3)
    normal = NativeCandidateView(base, cloud).conditioned_state(None, include_dynamic=False)
    black = BlackRadianceCandidateView(base, cloud).conditioned_state(None, include_dynamic=False)
    assert torch.equal(normal[0], black[0]) and torch.equal(normal[2], black[2])
    assert torch.equal(features, saved)
    assert torch.count_nonzero(black[1][:, 1:]) == 0
    torch.testing.assert_close(black[1][:, 0]*.28209479177387814+.5, torch.zeros(3, 3))
