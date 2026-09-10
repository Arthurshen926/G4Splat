"""Read-only adapter: preserve every optical weight, remove leaf radiance."""
import torch
from scripts.canopy_candidate_diagnostic import NativeCandidateView


class BlackRadianceCandidateView(NativeCandidateView):
    def conditioned_state(self, temporal_code, *, include_dynamic, **kwargs):
        xyz, features, opacity = super().conditioned_state(
            temporal_code, include_dynamic=include_dynamic, **kwargs)
        black = torch.zeros_like(features)
        black[:, 0] = -.5/.28209479177387814
        return xyz, black, opacity
