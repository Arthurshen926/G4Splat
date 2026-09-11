"""Conditional multi-view MoGe position prior, never opacity/size supervision."""
import math
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


class NativeDepthViews:
    """Lazy CPU source: preserve native rays; never average foreground/background."""
    def __init__(self, directory, camera_order):
        self.directory=Path(directory)
        self.camera_order=camera_order
        self.hashes={}

    def __getitem__(self,index):
        path=self.directory/(self.camera_order[index]+'.npz')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        name=self.camera_order[index];value=digest.hexdigest()
        if name in self.hashes and self.hashes[name]!=value:
            raise ValueError('Native MoGe depth changed during geometry supervision')
        self.hashes[name]=value
        with np.load(path) as archive:
            depth=np.array(archive['depth_m'],copy=True)
            valid=archive['valid_mask'].astype(bool)&np.isfinite(depth)&(depth>0)
            depth[~valid]=np.nan
            return depth


def project_depth_pixels(camera,view,shape):
    """Scale exact pixel intrinsics before rounding, not rounded low-res pixels."""
    h,w=shape
    uv=camera[:,:2]/camera[:,2,None].clamp_min(1e-6)
    x=((uv[:,0]*view.focal_x+view.cx)*(w/view.image_width)-.5).round().long()
    y=((uv[:,1]*view.focal_y+view.cy)*(h/view.image_height)-.5).round().long()
    return x,y


def interval_position_loss(xyz, world_view, ids, lower, upper):
    if len(ids)==0:return xyz.sum()*0
    z=(xyz[ids]@world_view[:3,:3]+world_view[3,:3])[:,2]
    # Bounds/evidence are immutable; only positions receive gradients.
    lo=lower.detach();hi=upper.detach()
    if not ((lo>0)&(hi>=lo)).all():raise ValueError('Ordered positive camera-z intervals required')
    delta=(lo-z).clamp_min(0)+(z-hi).clamp_min(0)
    return F.smooth_l1_loss(delta/((lo+hi)*.5),torch.zeros_like(delta),beta=.03)


@torch.no_grad()
def build_position_observations(xyz,views,train,excluded,regions,raw,cache,cal,*,motion_extent=None):
    if set(train)&set(excluded):raise ValueError('Training-only geometry evidence required')
    xyz=xyz.detach();count=torch.zeros(len(xyz),device=xyz.device,dtype=torch.int32)
    first=torch.zeros_like(xyz);angular=torch.zeros(len(xyz),device=xyz.device,dtype=torch.bool)
    records={};unreachable=0
    for ordinal,index in enumerate(train):
        v=views[index];m=regions(v)['tree_interior']
        d=torch.tensor(np.array(raw[cache['camera_order'].index(v.image_name)],copy=True),device=xyz.device)*cal.scale*cal.profiles[v.image_name]
        if m.shape!=d.shape:
            m=F.interpolate(m.float()[None,None],size=d.shape,mode='nearest')[0,0].bool()
        valid=torch.isfinite(d)&(d>.2)
        clean=torch.where(valid,d,torch.zeros_like(d))
        kernel=max(3,int(round(3*d.shape[1]/v.image_width)))
        if kernel%2==0:kernel+=1
        maximum=F.max_pool2d(clean[None,None],kernel,1,kernel//2)[0,0]
        minimum=-F.max_pool2d(-torch.where(valid,d,torch.full_like(d,float('inf')))[None,None],kernel,1,kernel//2)[0,0]
        all_valid=F.avg_pool2d(valid.float()[None,None],kernel,1,kernel//2)[0,0]>=1
        smooth=valid&all_valid&(maximum/minimum.clamp_min(.2)<1.05)&m
        camera=xyz@v.world_view_transform[:3,:3]+v.world_view_transform[3,:3]
        z=camera[:,2]
        x,y=project_depth_pixels(camera,v,d.shape)
        in_image=(z>.2)&(x>=0)&(x<d.shape[1])&(y>=0)&(y<d.shape[0])
        ids=in_image.nonzero().flatten();xx=x[ids];yy=y[ids];depth=d[yy,xx]
        keep=smooth[yy,xx]&(torch.abs(torch.log(z[ids]/depth.clamp_min(.2)))<math.log(1.1))
        ids=ids[keep];depth=depth[keep]
        if motion_extent is not None:
            extent=(motion_extent[ids]*v.world_view_transform[:3,2].abs()).sum(1)
            reachable=(z[ids]+extent>=depth*.97)&(z[ids]-extent<=depth*1.03)
            unreachable+=int((~reachable).sum());ids=ids[reachable];depth=depth[reachable]
        if len(ids)>16384:
            pick=torch.randperm(len(ids),generator=torch.Generator().manual_seed(9121+index))[:16384].to(ids.device)
            ids=ids[pick];depth=depth[pick]
        direction=F.normalize(xyz[ids]-v.camera_center,dim=1)
        old=count[ids]>0
        angular[ids]|=old&((first[ids]*direction).sum(1)<math.cos(math.radians(1)))
        first[ids[~old]]=direction[~old];count[ids]+=1
        records[index]=(ids.cpu(),(depth*.97).cpu(),(depth*1.03).cpu())
        if ordinal%50==0:print({'moge_position_views':ordinal+1},flush=True)
    eligible=((count>=3)&angular).cpu()
    for index,(ids,lo,hi) in records.items():
        keep=eligible[ids];records[index]=(ids[keep],lo[keep],hi[keep])
    report=dict(scope='conditional_MoGe_position_prior__not_verified_leaf_identity',eligible_candidates=int(eligible.sum()),
                observations=sum(len(v[0]) for v in records.values()),training_views=list(train),excluded_views=list(excluded),
                criteria='sampled support>=3 training views, >=1degree baseline, initial depth agreement10%, local depth range5%, interval3%',
                limitations=['Cached depth is used only in locally smooth tree interiors, not railing or mixed-depth boundaries',
                             'Depth agreement is conditional evidence, not independent leaf identity or opacity truth',
                             'Initial supports remain fixed; no source leaf or rigid geometry receives this loss'])
    report.update(individual_reachability_filtered=motion_extent is not None,
                  native_depth_source_sha256=getattr(raw,'hashes',None),
                  smoothness_policy='3 runtime pixels angular footprint, odd native kernel; native layers not averaged',
                  rejected_unreachable_pre_sampling=unreachable,
                  reachability_limit='Individual box overlap is necessary, not proof of jointly compatible multiview intervals')
    return records,report
