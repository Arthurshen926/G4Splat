"""Independent calibrated-ray candidates, never fabricated render witnesses."""
import torch
from contextlib import contextmanager


@torch.no_grad()
def record_distinct_camera_witnesses_(table, witnessed, camera_id):
    """Record already-established witnesses once; this does not establish them."""
    if table.ndim != 2 or witnessed.shape != (table.shape[0],):
        raise ValueError('Witness mask must align with the camera table')
    if table.dtype not in (torch.int16, torch.int32, torch.int64) or witnessed.dtype != torch.bool:
        raise ValueError('Integer camera IDs and boolean witness mask required')
    if int(camera_id) < 0 or int(camera_id) > torch.iinfo(table.dtype).max:
        raise ValueError('Camera ID must be representable and nonnegative')
    if not table.shape[1]:
        return 0
    free = table < 0
    rows = torch.nonzero(witnessed & free.any(dim=1) & ~(table == int(camera_id)).any(dim=1), as_tuple=False).flatten()
    slots = free[rows].to(torch.int8).argmax(dim=1)
    table[rows, slots] = int(camera_id)
    return int(len(rows))


def canopy_material_audit_fields(canopy,rigid,sky,rgb,topology=None):
    """Seven audit channels, not render gates: leaf color excludes other roles."""
    if rgb.shape!=(3,*canopy.shape) or rigid.shape!=canopy.shape or sky.shape!=canopy.shape:
        raise ValueError('Material audit fields must share the camera raster')
    topology=torch.zeros_like(canopy) if topology is None else topology
    return torch.cat((torch.stack((canopy,rigid,topology)),rgb*canopy[None],sky[None]),dim=0)


@torch.no_grad()
def clip_canopy_depth_queries_before_rigid(pre_bounds,hit_bounds,valid,canopy,rigid,rigid_depth,rigid_alpha):
    """An opaque observed occluder bounds both optical query authorities.

    Intrinsic queries deliberately remove the surface from their render.
    They therefore need this independent bound: a monocular posterior may
    neither erase hidden foliage behind a wall nor grow a hidden hit there.
    A canopy depth contradicting a known front wall is unknown, not evidence
    that all foreground canopy space is empty. Rigid free space remains
    usable up to the nearer conservative bound.
    """
    if pre_bounds.shape!=(2,*valid.shape) or hit_bounds.shape!=pre_bounds.shape:
        raise ValueError('Optical query tensors must share one raster')
    depth=rigid_depth.reshape_as(valid);alpha=rigid_alpha.reshape_as(valid)
    known=(alpha>=.95)&torch.isfinite(depth)&(depth>0)
    limit=(depth-torch.maximum(torch.full_like(depth,.03),.01*depth)).clamp_min(1e-4)
    center=pre_bounds[1]
    conflict=known & (center>=limit)
    canopy_conflict=conflict & (canopy>0) & (canopy>=rigid)
    valid=valid & ~canopy_conflict
    nan=torch.full_like(depth,float('nan'))
    clipped_pre=torch.where(known[None],torch.minimum(pre_bounds,limit[None]),pre_bounds)
    clipped_pre=torch.where(valid[None],clipped_pre,nan[None])
    upper=torch.where(known,torch.minimum(hit_bounds[1],limit),hit_bounds[1])
    hit_valid=valid & ~conflict & (upper>hit_bounds[0])
    clipped_hit=torch.stack((torch.where(hit_valid,hit_bounds[0],nan),torch.where(hit_valid,upper,nan)))
    return clipped_pre,clipped_hit,valid,{
        'contract':'intrinsic_optical_queries_stop_before_independent_opaque_rigid__conflicting_canopy_depth_is_unknown',
        'known_rigid_pixels':int(known.sum()),
        'prehit_bounds_clipped_pixels':int((known & (pre_bounds[0]>limit)).sum()),
        'hit_behind_rigid_rejected_pixels':int((conflict & torch.isfinite(hit_bounds).all(dim=0)).sum()),
        'canopy_depth_conflict_unknown_pixels':int(canopy_conflict.sum()),
        'hit_upper_clipped_pixels':int((hit_valid & known & (hit_bounds[1]>limit)).sum()),
    }


def measured_single_canonical_eligibility(foliage,camera_id,camera_sequence_lookup,*,measured_state,proposal_none):
    """Require an existing canonical witness and room for a genuinely new one."""
    result=torch.zeros(len(foliage.xyz),dtype=torch.bool,device=foliage.xyz.device)
    camera_id=int(camera_id)
    if camera_sequence_lookup is None or not 0<=camera_id<len(camera_sequence_lookup): return result
    target_sequence=camera_sequence_lookup[camera_id]
    if int(target_sequence)<0: return result
    source=foliage.verified_camera_ids
    valid=(source>=0) & (source<len(camera_sequence_lookup))
    canonical=valid & (camera_sequence_lookup[source.clamp(0,len(camera_sequence_lookup)-1).long()]==target_sequence)
    return (foliage.static_leaf_mask & ~foliage.dynamic_leaf_mask
        & (foliage.verification_state==int(measured_state))
        & (foliage.proposal_kind==int(proposal_none))
        & (foliage.verified_camera_count==1) & canonical.any(dim=1)
        & ~(foliage.support_camera_ids==camera_id).any(dim=1)
        & (foliage.support_camera_ids<0).any(dim=1)
        & (foliage.verified_camera_ids<0).any(dim=1))


@torch.no_grad()
def front_of_known_rigid_mask(xyz,view,rigid_depth,rigid_alpha):
    """Center-depth ordering for witness eligibility, not a deployment gate."""
    height,width=rigid_depth.shape[-2:]
    depth=rigid_depth.reshape(height,width);alpha=rigid_alpha.reshape(height,width)
    transform=view.world_view_transform.to(xyz)
    camera=xyz@transform[:3,:3]+transform[3,:3]
    z=camera[:,2];safe=z.clamp_min(1e-6)
    u=(view.focal_x*camera[:,0]/safe+view.cx).round().long()
    v=(view.focal_y*camera[:,1]/safe+view.cy).round().long()
    valid=torch.isfinite(camera).all(dim=1) & (z>0) & (u>=0) & (u<width) & (v>=0) & (v<height)
    rows=torch.nonzero(valid,as_tuple=False).flatten()
    surface_z=depth[v[rows],u[rows]]
    known=(alpha[v[rows],u[rows]]>=.95) & torch.isfinite(surface_z) & (surface_z>0)
    clearance=torch.maximum(torch.full_like(surface_z,.03),.01*surface_z)
    valid[rows] &= ~known | (z[rows]<surface_z-clearance)
    return valid


@contextmanager
def temporary_camera_support(foliage, rows, camera_id):
    """Expose candidates for one real verification pass; retain witnesses only."""
    rows=torch.as_tensor(rows,device=foliage.support_camera_ids.device,dtype=torch.long)
    if rows.ndim!=1 or len(torch.unique(rows))!=len(rows):
        raise ValueError("Candidate rows must be a unique vector")
    if int(camera_id)<0: raise ValueError("Camera ID must be nonnegative")
    if not len(rows):
        yield {"independent_depth_candidates":0,"new_real_camera_witnesses":0}
        return
    table=foliage.support_camera_ids[rows]
    free=table<0
    if not bool(free.any(dim=1).all()) or bool((table==int(camera_id)).any()):
        raise ValueError("A new camera requires a free, previously unused support slot")
    slots=free.to(torch.int8).argmax(dim=1)
    original=foliage.support_camera_ids[rows,slots].clone()
    with torch.no_grad(): foliage.support_camera_ids[rows,slots]=int(camera_id)
    audit={"independent_depth_candidates":int(len(rows)),"new_real_camera_witnesses":0}
    completed=False
    try:
        yield audit
        completed=True
    finally:
        with torch.no_grad():
            retained=(foliage.verified_camera_ids[rows]==int(camera_id)).any(dim=1)
            if not completed: retained.zero_()
            foliage.support_camera_ids[rows[~retained],slots[~retained]]=original[~retained]
            # A native witness retained in the support table is an actual
            # independent supporting camera. Keep its cached count coherent:
            # topology must not continue treating a two-view leaf as single.
            kept_rows=rows[retained]
            if len(kept_rows):
                ids=foliage.support_camera_ids[kept_rows].sort(dim=1).values
                distinct=ids>=0
                distinct[:,1:] &= ids[:,1:]!=ids[:,:-1]
                count=distinct.sum(1).to(foliage.support_view_count.dtype)
                foliage.support_view_count[kept_rows]=torch.maximum(
                    foliage.support_view_count[kept_rows],count)
            audit["new_real_camera_witnesses"]=int(retained.sum())


@torch.no_grad()
def independent_canopy_camera_candidates(xyz, view, hit_bounds, canopy_valid, eligible, *, rigid_depth, rigid_alpha):
    """Select existing point centers consistent with a new observed canopy ray.

    The caller must restrict eligibility to the canonical sequence and an
    unresolved measured point. A returned row is only a candidate: it still
    needs a native contribution witness in this independent camera. Neither
    this function nor voxel co-membership establishes persistence.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected Nx3 point centers")
    height, width = canopy_valid.shape
    if hit_bounds.shape != (2,height,width) or eligible.shape != (len(xyz),):
        raise ValueError("Candidate tensors are not aligned")
    rigid_depth=rigid_depth.reshape(height,width)
    rigid_alpha=rigid_alpha.reshape(height,width)
    rows = torch.nonzero(eligible, as_tuple=False).flatten()
    if not len(rows): return rows
    points = xyz[rows]
    transform = view.world_view_transform.to(points)
    camera = points @ transform[:3,:3] + transform[3,:3]
    z = camera[:,2]
    safe = z.clamp_min(1e-6)
    u = (view.focal_x*camera[:,0]/safe+view.cx).round().long()
    v = (view.focal_y*camera[:,1]/safe+view.cy).round().long()
    in_image = torch.isfinite(camera).all(dim=1) & (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    rows, u, v, z = rows[in_image],u[in_image],v[in_image],z[in_image]
    accepted = canopy_valid[v,u] & torch.isfinite(hit_bounds[:,v,u]).all(dim=0)
    accepted &= (z >= hit_bounds[0,v,u]) & (z <= hit_bounds[1,v,u])
    surface_z=rigid_depth[v,u]
    known_surface=(rigid_alpha[v,u]>=.95) & torch.isfinite(surface_z) & (surface_z>0)
    # T_before*alpha can be nonzero *behind* an almost opaque wall. Such a
    # numerical tail is not independent foreground geometry evidence.
    clearance=torch.maximum(torch.full_like(surface_z,.03),.01*surface_z)
    accepted &= ~known_surface | (z < surface_z-clearance)
    return rows[accepted]
