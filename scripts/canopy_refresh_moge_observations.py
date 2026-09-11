"""Re-associate existing conditional targets without adding supervision owners."""
import math
import numpy as np
import torch
from scripts.canopy_moge_position_supervision import project_depth_pixels


@torch.no_grad()
def query_depth_validity(view,regions,source,cache,cal,device):
    import torch.nn.functional as F
    d=torch.tensor(np.array(source[cache['camera_order'].index(view.image_name)],copy=True),device=device)*cal.scale*cal.profiles[view.image_name]
    mask=regions(view)['tree_interior']
    if mask.shape!=d.shape:mask=F.interpolate(mask.float()[None,None],size=d.shape,mode='nearest')[0,0].bool()
    valid=torch.isfinite(d)&(d>.2)
    kernel=max(3,int(round(3*d.shape[1]/view.image_width)))
    if kernel%2==0:kernel+=1
    maximum=F.max_pool2d(torch.where(valid,d,torch.zeros_like(d))[None,None],kernel,1,kernel//2)[0,0]
    minimum=-F.max_pool2d(-torch.where(valid,d,torch.full_like(d,float('inf')))[None,None],kernel,1,kernel//2)[0,0]
    all_valid=F.avg_pool2d(valid.float()[None,None],kernel,1,kernel//2)[0,0]>=1
    return d,valid&all_valid&(maximum/minimum.clamp_min(.2)<1.05)&mask


def validate_refreshed_records(initial,records):
    if set(initial)!=set(records):raise ValueError('Refreshed camera set changed')
    for index,(ids,lo,hi) in records.items():
        if (ids.dtype!=torch.long or ids.ndim!=1 or lo.shape!=ids.shape or hi.shape!=ids.shape
                or not torch.isfinite(lo).all() or not torch.isfinite(hi).all()
                or not ((lo>0)&(hi>=lo)).all() or len(set(ids.tolist()))!=len(ids)
                or not set(ids.tolist()).issubset(set(initial[index][0].tolist()))):
            raise ValueError('Invalid refreshed targets or added ownership')


@torch.no_grad()
def refresh_observations(xyz,views,initial_records,depth_and_valid,*,excluded=(),tolerance=.03,
                         initial_xyz=None,motion_extent=None):
    """Requery at moved pixels; reject changed surfaces and recount support.

    depth_and_valid returns calibrated depth and a boolean admissibility mask
    (including semantic, finite, local-boundary and confidence checks).
    A target must agree with the ORIGINAL interval center within tolerance.
    Only IDs already present for that camera can survive. Three surviving
    camera observations and angular separation are required again. This is
    conditional ray association, NOT visibility/leaf-identity ground truth.
    """
    if set(initial_records)&set(excluded):raise ValueError('Excluded camera in geometry refresh')
    if not 0<tolerance<1:raise ValueError('Bounded positive association tolerance required')
    if (initial_xyz is None)!=(motion_extent is None):raise ValueError('Both original position and motion extent required')
    count=torch.zeros(len(xyz),dtype=torch.int32,device=xyz.device)
    first=torch.zeros_like(xyz);angular=torch.zeros(len(xyz),dtype=torch.bool,device=xyz.device)
    provisional={};before=0
    for index,(ids,lo,hi) in initial_records.items():
        before+=len(ids);ids=ids.to(xyz.device);lo=lo.to(xyz.device);hi=hi.to(xyz.device)
        view=views[index];depth,valid=depth_and_valid(index,view)
        if depth.shape!=valid.shape or valid.dtype!=torch.bool:raise ValueError('Aligned depth and boolean validity required')
        camera=xyz[ids]@view.world_view_transform[:3,:3]+view.world_view_transform[3,:3]
        x,y=project_depth_pixels(camera,view,depth.shape)
        inside=torch.isfinite(camera).all(-1)&(camera[:,2]>0)&(x>=0)&(x<depth.shape[1])&(y>=0)&(y<depth.shape[0])
        ids=ids[inside];x=x[inside];y=y[inside];origin=(lo[inside]+hi[inside])*.5
        target=depth[y,x]
        keep=valid[y,x]&torch.isfinite(target)&(target>0)&((target-origin).abs()<=tolerance*origin)
        ids=ids[keep];target=target[keep]
        if initial_xyz is not None:
            axis=view.world_view_transform[:3,2]
            center=initial_xyz[ids]@axis+view.world_view_transform[3,2]
            extent=(motion_extent[ids]*axis.abs()).sum(-1)
            reachable=(center+extent>=(1-tolerance)*target)&(center-extent<=(1+tolerance)*target)
            ids=ids[reachable];target=target[reachable]
        direction=torch.nn.functional.normalize(xyz[ids]-view.camera_center,dim=1)
        old=count[ids]>0
        angular[ids]|=old&((first[ids]*direction).sum(-1)<math.cos(math.radians(1)))
        first[ids[~old]]=direction[~old];count[ids]+=1
        provisional[index]=(ids.cpu(),((1-tolerance)*target).cpu(),((1+tolerance)*target).cpu())
    eligible=((count>=3)&angular).cpu()
    result={index:tuple(value[eligible[record[0]]] for value in record) for index,record in provisional.items()}
    return result,dict(before=before,after=sum(len(record[0]) for record in result.values()),
                       eligible_candidates=int(eligible.sum()),reachability_checked=initial_xyz is not None,
                       scope='subset_of_original_conditional_support__not_visibility')
