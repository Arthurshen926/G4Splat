"""Bounded static candidate displacement, learned from RGB only."""
import math
import torch


class SpatialCandidates(torch.nn.Module):
    def __init__(self, cloud, radius_sigmas=8.):
        super().__init__()
        if not math.isfinite(radius_sigmas) or not 0 < radius_sigmas <= 16:
            raise ValueError('Bounded positive candidate displacement required')
        self.cloud = cloud
        self.radius_sigmas = float(radius_sigmas)
        self.position_code = torch.nn.Parameter(torch.zeros_like(cloud.xyz))

    @property
    def xyz(self):
        return self.cloud.xyz+self.radius_sigmas*self.cloud.scales*self.position_code.tanh()
    @property
    def scales(self): return self.cloud.scales
    @property
    def quaternions(self): return self.cloud.quaternions
    @property
    def dc(self): return self.cloud.dc
    @property
    def logits(self): return self.cloud.logits
