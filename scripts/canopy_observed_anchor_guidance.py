"""Experimental regional centroid guidance, never per-leaf feature identity."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from outdoor.moge3_evidence import sha256_file


def load_guidance(associations,tracks,thin,checkpoint,foliage,field,views,train,excluded):
    a=json.loads(Path(associations).read_text());t=json.loads(Path(tracks).read_text());q=json.loads(Path(thin).read_text())
    if (a['scope']!='regional_source_bearing_association__not_exact_features_or_training_authority'
        or a['checkpoint_sha256']!=sha256_file(checkpoint)
        or a['track_audit_sha256']!=sha256_file(tracks)
        or a['thin_audit_sha256']!=sha256_file(thin)
        or q['track_audit_sha256']!=sha256_file(tracks)
        or set(t['excluded_views'])!=set(excluded)):
        raise ValueError('Independent observation association identity changed')
    records={r['track_id']:r for r in t['records']};front={}
    for o in q['records']:
        if o['track_depth']<o['hit_lower']:front.setdefault(o['track_id'],set()).add(o['index'])
    indices=[];groups=[];targets=[]
    active=field.active.detach().cpu()
    active_mask=torch.zeros(field.count,dtype=torch.bool)
    active_mask[active]=True
    source=foliage.observation_camera_ids[:,0].detach().cpu()
    uv=foliage.observation_uv[:,0].detach().cpu()
    for row in a['records']:
        track=records[row['track_id']]
        if len(front.get(row['track_id'],()))<3:raise ValueError('Three independent front-of-prior observations required')
        if row['target_xyz']!=track['xyz']:raise ValueError('Altered triangulated target')
        leaves=torch.tensor(row['associated_leaf_rows'],dtype=torch.long)
        if not len(leaves):continue
        if (leaves<0).any() or (leaves>=field.count).any() or not active_mask[leaves].all():
            raise ValueError('Only existing persistent eligible leaves may move')
        assigned=set()
        for o in track['observations']:
            if o['index'] not in train or o['index'] in excluded:raise ValueError('Nontraining anchor')
            view=views[o['index']]
            if view.image_name!=o['image_name']:raise ValueError('Camera identity changed')
            if not o['canopy']:continue
            selected=leaves[source[leaves]==int(view.colmap_id)]
            if not len(selected):continue
            pixels=uv[selected]*torch.tensor([view.image_width,view.image_height])-.5
            if ((pixels-torch.tensor(o['uv'])).norm(dim=1)>a['pixel_radius']+1.e-4).any():
                raise ValueError('Association exceeds original source-bearing neighborhood')
            indices.extend(selected.tolist());groups.extend([len(targets)]*len(selected));targets.append(track['xyz'])
            assigned.update(selected.tolist())
        if assigned!=set(leaves.tolist()):raise ValueError('Unassociated source rows')
    if not targets:raise ValueError('Empty regional guidance')
    device=field.base.device
    indices=torch.tensor(indices,device=device,dtype=torch.long)
    return dict(indices=indices,active_indices=torch.searchsorted(field.active,indices),
                groups=torch.tensor(groups,device=device,dtype=torch.long),
                targets=torch.tensor(targets,device=device,dtype=field.base.dtype),
                base_xyz=foliage.xyz.detach()[indices].clone())


def centroid_guidance_loss(field,guidance,uncertainty_floor=.1):
    """Move source-region centroids, preserving within-region leaf offsets.

    The .1-scene-unit Huber scale is explicit diagnostic robustness, not a
    calibrated probability. Sparse field evaluation matches the full field.
    """
    if uncertainty_floor<=0:raise ValueError('Positive robust scale required')
    active=guidance['active_indices'];nodes=field.node_offsets()
    offsets=(nodes[field.indices[active]]*field.weights[active,:,None]).sum(1)
    points=guidance['base_xyz']+offsets
    groups=guidance['groups'];targets=guidance['targets']
    sums=points.new_zeros(len(targets),3).index_add(0,groups,points)
    counts=torch.bincount(groups,minlength=len(targets)).to(points)[:,None]
    centroids=sums/counts
    return F.smooth_l1_loss(centroids/uncertainty_floor,targets/uncertainty_floor,beta=1.)
