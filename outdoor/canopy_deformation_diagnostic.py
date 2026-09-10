"""Bounded, tree-only center fields for a controlled deformation experiment.

This module is not part of production rendering. Canonical output uses only
the shared spatial field; temporal output must be explicitly requested and
reported separately. Colors, opacity, covariance, cameras and rigid geometry
are not parameters of this model.
"""
import math
import torch
from torch import nn


def enable_shared_canopy_optics(foliage, eligible):
    """One shared material state; nonpersistent rows have zero update authority."""
    handles=[]
    for parameter in (foliage.features,foliage.opacity_logits):
        parameter.requires_grad_(True)
        gate=eligible.to(parameter).reshape(len(eligible),*([1]*(parameter.ndim-1)))
        handles.append(parameter.register_hook(lambda gradient,gate=gate:gradient*gate))
    return handles


@torch.no_grad()
def apply_dc_learning_rate_multiplier_(foliage,eligible,before_dc,multiplier):
    """Scale the actual Adam DC update, not its normalized input gradient.

    Geometry, higher SH, opacity, and ineligible colors are unchanged. This
    is an explicit optimization ablation, not an opacity or appearance prior.
    """
    if not math.isfinite(multiplier) or not 0<multiplier<=20:
        raise ValueError('Finite positive bounded DC learning-rate multiplier required')
    if before_dc.shape!=foliage.features[:,0].shape:
        raise ValueError('Aligned pre-step DC snapshot required')
    current=foliage.features[eligible,0]
    foliage.features[eligible,0]=before_dc[eligible]+multiplier*(current-before_dc[eligible])


@torch.no_grad()
def project_shared_canopy_optics_(foliage,eligible,opacity_cap=.35,sh_cap=4.,initial_opacity_logits=None):
    """Bound optics, optionally preserving the source's existing upper envelope.

    A refinement starts from a valid checkpoint, not a newly initialized cloud.
    A diagnostic cap must not retire existing material with zero gradient.
    The detached initial envelope is only an upper bound, never a restoration
    floor: evidence-driven retirement remains possible.
    """
    if not 0<opacity_cap<1 or not math.isfinite(sh_cap) or sh_cap<=0:
        raise ValueError('Finite bounded optical projection required')
    logits=foliage.opacity_logits[eligible]
    ceiling=torch.full_like(logits,math.log(opacity_cap/(1-opacity_cap)))
    if initial_opacity_logits is not None:
        if initial_opacity_logits.shape!=foliage.opacity_logits.shape or not torch.isfinite(initial_opacity_logits).all():
            raise ValueError('Aligned finite immutable source logits required')
        ceiling=torch.maximum(ceiling,initial_opacity_logits.detach().to(logits)[eligible])
    foliage.opacity_logits[eligible]=torch.minimum(logits,ceiling)
    high=foliage.features[eligible,1:]
    norm=high.flatten(1).norm(dim=1)
    high*= (sh_cap/norm.clamp_min(1.e-12)).clamp_max(1)[:,None,None]
    foliage.features[eligible,1:]=high


class BoundedCanopyCenterField(nn.Module):
    def __init__(self, positions, eligible, times, *, spacing=2., radius=.1, rank=0, reference_time=None):
        super().__init__()
        if (not math.isfinite(spacing) or spacing<=0 or not math.isfinite(radius) or radius<=0
            or rank<0 or eligible.dtype!=torch.bool or eligible.shape!=(len(positions),)
            or positions.ndim!=2 or positions.shape[1]!=3 or not torch.isfinite(positions).all()):
            raise ValueError('Finite geometry and an aligned immutable eligibility mask required')
        times=torch.as_tensor(times,device=positions.device,dtype=positions.dtype)
        if times.ndim!=1 or len(times)<2 or not torch.isfinite(times).all() or not (times[1:]>times[:-1]).all():
            raise ValueError('Distinct sorted training times required')
        if reference_time is not None and (not math.isfinite(reference_time)
                or not bool((times==float(reference_time)).any())):
            raise ValueError('Canonical reference must be an observed training time')
        reference_index=(-1 if reference_time is None else
                         int(torch.nonzero(times==float(reference_time),as_tuple=True)[0][0]))
        self.register_buffer('reference_index',torch.tensor(reference_index,device=positions.device))
        active=torch.nonzero(eligible,as_tuple=True)[0]
        if not len(active):raise ValueError('No persistent canopy geometry')
        offsets=torch.tensor([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)],device=positions.device)
        scaled=positions[active].detach()/spacing
        lower=scaled.floor().long();fraction=scaled-lower
        keys,inverse=torch.unique((lower[:,None,:]+offsets[None]).reshape(-1,3),dim=0,return_inverse=True)
        weights=torch.where(offsets[None].bool(),fraction[:,None,:],1-fraction[:,None,:]).prod(-1)
        self.register_buffer('active',active)
        self.register_buffer('indices',inverse.reshape(-1,8))
        self.register_buffer('weights',weights)
        self.register_buffer('keys',keys)
        self.register_buffer('times',times)
        self.base=nn.Parameter(positions.new_zeros(len(keys),3))
        self.rank=int(rank);self.radius=float(radius);self.spacing=float(spacing)
        self.count=len(positions)
        if rank:
            self.basis=nn.Parameter(positions.new_zeros(len(keys),rank,3))
            phase=(times-times[0])/(times[-1]-times[0])
            codes=torch.stack([torch.sin(2*math.pi*(1+k//2)*phase) if k%2==0
                               else torch.cos(2*math.pi*(1+k//2)*phase) for k in range(rank)],dim=1)
            self.codes=nn.Parameter(codes)
        else:
            self.register_parameter('basis',None);self.register_parameter('codes',None)
        lookup={tuple(k):i for i,k in enumerate(keys.cpu().tolist())}
        edges=[]
        for key,i in lookup.items():
            for axis in range(3):
                adjacent=list(key);adjacent[axis]+=1
                j=lookup.get(tuple(adjacent))
                if j is not None:edges.append((i,j))
        self.register_buffer('edges',torch.tensor(edges,device=positions.device,dtype=torch.long).reshape(-1,2))

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        key=prefix+'reference_index'
        if key not in state_dict and int(self.reference_index)==-1:
            # Legacy diagnostic captures mean-centered codes, not a physical
            # reference time. Never silently relabel them as an anchored map.
            state_dict=dict(state_dict);state_dict[key]=self.reference_index.clone()
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                    missing_keys,unexpected_keys,error_msgs)

    def code_at(self,time):
        if not self.rank:raise ValueError('Static control has no temporal codes')
        value=self.times.new_tensor(float(time))
        upper=int(torch.searchsorted(self.times,value).clamp(1,len(self.times)-1))
        lower=upper-1
        weight=((value-self.times[lower])/(self.times[upper]-self.times[lower])).clamp(0,1)
        # No held-out embedding, RGB fitting or nearest evaluation-state lookup.
        code=(1-weight)*self.codes[lower]+weight*self.codes[upper]
        reference_index=int(self.reference_index)
        reference=self.codes.mean(0) if reference_index<0 else self.codes[reference_index]
        return code-reference

    def node_offsets(self,time=None):
        raw=self.base
        if time is not None and self.rank:
            raw=raw+torch.einsum('nrc,r->nc',self.basis,self.code_at(time))
        return raw.tanh()*(self.radius/math.sqrt(3.))

    def offsets(self,time=None):
        nodes=self.node_offsets(time)
        active=(nodes[self.indices]*self.weights[:,:,None]).sum(1)
        return nodes.new_zeros(self.count,3).index_copy(0,self.active,active)

    def regularization(self,time=None):
        nodes=self.node_offsets(time)
        magnitude=nodes.square().mean()/self.radius**2
        smoothness=((nodes[self.edges[:,0]]-nodes[self.edges[:,1]]).square().mean()/self.radius**2
                    if len(self.edges) else magnitude*0)
        temporal=magnitude*0
        if time is not None and self.rank:
            index=int((self.times-float(time)).abs().argmin())
            neighbors=[j for j in (index-1,index+1) if 0<=j<len(self.times)]
            terms=[(nodes-self.node_offsets(float(self.times[j]))).square().mean()
                   /(self.radius**2*max(abs(float(self.times[j])-float(time)),1.)**2)
                   for j in neighbors]
            if terms:temporal=torch.stack(terms).mean()
        return magnitude,smoothness,temporal
