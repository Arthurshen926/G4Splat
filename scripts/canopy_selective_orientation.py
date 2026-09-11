"""Bounded candidate-only orientation, preserving existing axis/XYZ limits.

Only a training-derived experimental shape cohort may rotate. The cohort is not
verified geometry. This does not alter original leaves or any rigid primitive.
"""
import torch
from scripts.canopy_selective_shape import SelectiveSpatialFootprintCandidates


class OrientedSelectiveCandidates(SelectiveSpatialFootprintCandidates):
    def __init__(self, cloud, eligible, radius_sigmas, log_radius):
        super().__init__(cloud, eligible, radius_sigmas, log_radius)
        self.rotation_code = torch.nn.Parameter(torch.zeros_like(cloud.xyz))

    def orientation_observable(self):
        # Rotation of a sphere is a gauge, not a learnable geometric change.
        # Native float32 covariance roundoff otherwise feeds nonzero gradients
        # into Adam(eps=1e-15), which can turn them into a full rotation step.
        # This is a numerical isotropy tolerance, NOT an image/visibility gate.
        with torch.no_grad():
            scales = self.scales
            maximum = scales.max(1).values
            return self.shape_eligible & ((maximum-scales.min(1).values)>1e-5*maximum)

    @property
    def quaternions(self):
        # Smooth at zero, unlike a naive axis-angle implementation with ||v||
        # in its derivative. ||v||<1 makes the delta rotation strictly <90deg.
        code = self.rotation_code * self.shape_eligible[:, None]
        v = code / torch.sqrt(1 + code.square().sum(1, keepdim=True))
        delta = torch.cat((torch.ones_like(v[:, :1]), v), 1)
        delta = delta / torch.sqrt(delta.square().sum(1, keepdim=True))
        original = self.cloud.quaternions
        w = delta[:, :1]*original[:, :1]-(delta[:, 1:]*original[:, 1:]).sum(1,keepdim=True)
        xyz = (delta[:, :1]*original[:, 1:]+original[:, :1]*delta[:, 1:]
               +torch.cross(delta[:, 1:],original[:, 1:],dim=1))
        rotated = torch.cat((w,xyz),1)
        return torch.where(self.orientation_observable()[:,None],rotated,original)

    @torch.no_grad()
    def orientation_audit(self):
        angles = 2*torch.atan(self.rotation_code.norm(dim=1)
            /torch.sqrt(1+self.rotation_code.square().sum(1)))
        return dict(changed_rows=int(self.rotation_code.ne(0).any(1).sum()),
            observable_rows=int(self.orientation_observable().sum()),
            ineligible_unchanged=bool(self.rotation_code[~self.shape_eligible].eq(0).all()),
            maximum_angle_radians=float(angles.max()),
            finite=bool(torch.isfinite(self.rotation_code).all() and torch.isfinite(self.quaternions).all()),
            scope='candidate_only_bounded_experimental_orientation__not_verified_geometry')
