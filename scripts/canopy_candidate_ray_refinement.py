"""Optional RGB ray-depth refinement of unverified candidates, not buildings."""
import math
import torch


class RayDepthCandidates(torch.nn.Module):
    def __init__(self, cloud, origins, log_radius=.15):
        super().__init__()
        if origins.shape != cloud.xyz.shape or not torch.isfinite(origins).all():
            raise ValueError('Each hypothesis needs its immutable source camera center')
        if not math.isfinite(log_radius) or not 0 < log_radius <= .3:
            raise ValueError('Bounded relative depth refinement required')
        self.cloud = cloud
        self.register_buffer('origins', origins.detach().clone())
        self.log_radius = float(log_radius)
        self.depth_code = torch.nn.Parameter(torch.zeros_like(cloud.logits))

    @property
    def ratio(self): return (self.log_radius*self.depth_code.tanh()).exp()
    @property
    def xyz(self): return self.cloud.xyz+(self.cloud.xyz-self.origins)*(self.ratio[:, None]-1)
    @property
    def scales(self): return self.cloud.scales*self.ratio[:, None]
    @property
    def quaternions(self): return self.cloud.quaternions
    @property
    def dc(self): return self.cloud.dc
    @property
    def logits(self): return self.cloud.logits


def rigid_rgb_preservation(rgb, reference, rigid):
    """Preserve original known-building RGB, without labeling depth as truth."""
    if rgb.shape != reference.shape or rgb.shape[1:] != rigid.shape:
        raise ValueError('Aligned immutable RGB reference required')
    error = (rgb-reference.detach()).abs().mean(0)
    return (error*rigid).sum()/rigid.sum().clamp_min(1)


@torch.no_grad()
def project_candidate_dc_(dc):
    """Keep DC-only diffuse colors in gamut, never below native ReLU support.

    This touches only the experimental candidates, not source SH coefficients.
    Projection to the boundary retains the native clamp's recovery derivative;
    a negative DC-only color would be dead in every camera, unlike varying SH.
    """
    # In float32 the rounded mathematical lower endpoint can evaluate to a
    # slightly negative RGB and remain dead. Move one representable value
    # inward; the recovery-gradient test covers this boundary explicitly.
    bound = torch.nextafter(dc.new_tensor(.5/.28209479177387814), dc.new_tensor(0.))
    dc.clamp_(-bound, bound)
