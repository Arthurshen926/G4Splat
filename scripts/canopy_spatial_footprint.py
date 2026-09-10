"""Independent bounded position and footprint for unverified RGB candidates."""
import math
import torch
from scripts.canopy_candidate_spatial import SpatialCandidates


class SpatialFootprintCandidates(SpatialCandidates):
    def __init__(self, cloud, radius_sigmas=8., log_radius=math.log(2.)):
        super().__init__(cloud, radius_sigmas)
        if not math.isfinite(log_radius) or not 0 < log_radius <= math.log(2.):
            raise ValueError('Independent footprint limited to half through twice initial size')
        self.log_radius = float(log_radius)
        self.scale_code = torch.nn.Parameter(torch.zeros_like(cloud.logits))

    @property
    def scales(self):
        return self.cloud.scales*(self.log_radius*self.scale_code.tanh()).exp()[:, None]

    # Inherited xyz uses immutable cloud.scales, NOT the learned self.scales.
    # Growing the footprint cannot move the center or expand its motion bounds.
