"""Bounded, RGB-learned candidate footprint; no forced optical mass growth."""
import math
import torch


class FootprintCandidates(torch.nn.Module):
    def __init__(self, cloud, log_radius=math.log(2.)):
        super().__init__()
        if not math.isfinite(log_radius) or not 0 < log_radius <= math.log(3.):
            raise ValueError('A bounded positive footprint range is required')
        self.cloud = cloud; self.log_radius = float(log_radius)
        self.scale_code = torch.nn.Parameter(torch.zeros_like(cloud.logits))

    @property
    def ratio(self): return (self.log_radius*self.scale_code.tanh()).exp()
    @property
    def xyz(self): return self.cloud.xyz
    @property
    def scales(self): return self.cloud.scales*self.ratio[:, None]
    @property
    def quaternions(self): return self.cloud.quaternions
    @property
    def dc(self): return self.cloud.dc
    @property
    def logits(self): return self.cloud.logits
