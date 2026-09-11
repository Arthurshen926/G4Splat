"""Experimental candidate-only axis selectivity, within original size bounds.

An audit cohort grants shape freedom, NOT verified geometry/support identity.
No rotations, extra primitives, source-leaf changes or larger axis bounds.
"""
import hashlib
import torch
from scripts.canopy_spatial_footprint import SpatialFootprintCandidates


def initial_geometry_sha256(cloud):
    digest=hashlib.sha256()
    for name in ('xyz','scales','quaternions'):
        value=getattr(cloud,name).detach().cpu().contiguous()
        digest.update(str((name,tuple(value.shape),str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_conflicting_candidates(mass, counts):
    if mass.ndim!=2 or mass.shape[1]!=3 or counts.shape!=mass.shape:
        raise ValueError('Tree/interior-wall/interface responsibility required')
    if not torch.isfinite(mass).all() or (mass<0).any() or (counts<0).any():
        raise ValueError('Finite nonnegative training evidence required')
    negative=mass[:,1]+mass[:,2]
    return (counts[:,0]>=2)&(counts[:,1]>=2)&(negative>.01*mass.sum(1))


def load_shape_cohort(path,cloud,source_sha256,training_views,excluded_views):
    data=torch.load(path,map_location='cpu',weights_only=True)
    report=data['report'];eligible=data['eligible']
    if (report['source_checkpoint_sha256']!=source_sha256
            or report['initial_geometry_sha256']!=initial_geometry_sha256(cloud)
            or report['training_views']!=training_views or report['excluded_views']!=excluded_views
            or set(training_views)&set(excluded_views)
            or eligible.dtype!=torch.bool or eligible.shape!=cloud.logits.shape
            or report['selected_count']!=int(eligible.sum()) or not eligible.any()
            or not torch.equal(eligible,select_conflicting_candidates(data['mass'],data['counts']))):
        raise ValueError('Shape cohort changed initial geometry, evidence or candidate identities')
    return eligible.to(cloud.xyz.device)


class SelectiveSpatialFootprintCandidates(SpatialFootprintCandidates):
    def __init__(self,cloud,eligible,radius_sigmas,log_radius):
        super().__init__(cloud,radius_sigmas,log_radius)
        if eligible.dtype!=torch.bool or eligible.shape!=cloud.logits.shape:
            raise ValueError('Explicit candidate-only shape cohort required')
        self.register_buffer('shape_eligible',eligible.detach().clone())
        self.shape_code=torch.nn.Parameter(torch.zeros_like(cloud.xyz))

    @property
    def scales(self):
        # At zero shape, forward AND isotropic-code derivative match the old
        # model. Deviatoric code changes selectivity, not the allowed axis range.
        shape=self.shape_code-self.shape_code.mean(1,keepdim=True)
        shape=shape*self.shape_eligible[:,None]
        code=self.scale_code[:,None]+shape
        return self.cloud.scales*(self.log_radius*code.tanh()).exp()
