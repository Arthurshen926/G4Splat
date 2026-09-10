"""Diagnostic one-hit ray-depth selection using all canonical evidence views.

Uncertainty proposes alternative locations for ONE leaf, never nine layers.
Opaque-wall occlusion is unknown; observed rigid/sky foreground is negative.
Depth agreements only nominate cameras; actual native witnesses are required.
"""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
import torch.nn.functional as F
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import (VolumetricFoliageModel,render_hybrid,
    static_detail_forward_visibility_gate,VERIFICATION_UNVERIFIED,VERIFICATION_VERIFIED,PROPOSAL_RAY_BIRTH)
from outdoor.canopy_support_expansion import front_of_known_rigid_mask,canopy_material_audit_fields
from outdoor.canopy_patch_stereo import candidate_patch_scores,unambiguous_depth_matches
from outdoor.training_evidence import OutdoorGeometryEvidence
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds,_update_static_child_verification_from_render
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest

# The native rasterizer currently culls z <= .2 in auxiliary.h::in_frustum,
# irrespective of the Camera projection's .01 near plane. Candidate support
# must not count samples that this renderer will discard.
NATIVE_MIN_DEPTH=.2


def select_single_depth(positive,negative,log_shifts,admissible=None,frontmost=False):
    """Prefer no contradiction, then more real depth views, then small motion."""
    eligible=(positive>=2)&(negative<=1)
    if admissible is not None:eligible &= admissible
    score=positive.float()-1000*negative.float()-.01*log_shifts.square()[None]
    if frontmost:
        # A visual-hull candidate is a first supported hit, not a stack of
        # samples through the entire uncertain occupied volume.
        score=-1000*negative.float()-torch.arange(positive.shape[1],device=positive.device)[None]/(positive.shape[1]+1)
    score=torch.where(eligible,score,torch.full_like(score,-float('inf')))
    selected=score.argmax(dim=1)
    return selected,eligible.any(dim=1)


def source_wall_depth_factors(initial_depth,upper_depth,log_shifts,near_depth):
    """One ordered candidate grid per ray, not material across the interval.

    A missing wall cannot define a bounded global search. Such rays explicitly
    retain their local proposal grid. The upper endpoint stays strictly before
    the measured source wall; near/invalid intervals cannot authorize a hit.
    """
    if initial_depth.ndim!=1 or upper_depth.shape!=initial_depth.shape:
        raise ValueError('Matched one-dimensional source depth arrays required')
    if not torch.isfinite(initial_depth).all() or not (initial_depth>0).all():
        raise ValueError('Finite positive source proposal depths required')
    if not math.isfinite(near_depth) or near_depth<=0:
        raise ValueError('Finite positive near depth required')
    bounded=torch.isfinite(upper_depth)&(upper_depth>near_depth)
    factors=log_shifts.exp()[None].expand(len(initial_depth),-1).clone()
    fraction=torch.linspace(0,1,len(log_shifts),device=initial_depth.device,dtype=initial_depth.dtype)
    far=torch.nextafter(upper_depth[bounded],torch.zeros_like(upper_depth[bounded]))
    depth=near_depth+(far[:,None]-near_depth)*fraction[None]
    factors[bounded]=depth/initial_depth[bounded,None]
    return factors,bounded


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for name in ('checkpoint','foliage','evidence-store','masks','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--metric-scale',type=float,required=True)
    p.add_argument('--log-radius',type=float,default=.15)
    p.add_argument('--depth-samples',type=int,default=9)
    p.add_argument('--positive-evidence',choices=('moge','patch_stereo','silhouette'),default='moge')
    p.add_argument('--rigid-anchor-scale',action='store_true')
    p.add_argument('--preserve-source-footprint',action='store_true')
    p.add_argument('--source-wall-depth-sweep',action='store_true',
        help='For unmeasured bearing pools only: search the entire finite source-wall front interval')
    a=p.parse_args()
    if not math.isfinite(a.log_radius) or a.log_radius<=0 or a.depth_samples<3 or a.depth_samples%2!=1:
        raise ValueError('Positive finite depth radius and odd sample count >=3 required')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    capture=torch.load(a.foliage,map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():
        raise ValueError('Explicit source-matched diagnostic required')
    if bool(capture.get('rigid_anchor_scale',False))!=a.rigid_anchor_scale:
        raise ValueError('Seed and validation depth calibration must agree')
    if a.source_wall_depth_sweep and (not capture.get('bearing_proposal_pool') or a.positive_evidence!='silhouette'):
        raise ValueError('Full wall-front search requires an unmeasured bearing pool and silhouette evidence')
    foliage=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda')
    foliage.restore(capture['foliage']);teacher.foliage=foliage
    if foliage.dynamic_leaf_mask.any():raise ValueError('Static-only depth selection')
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    by_id={int(v.colmap_id):v for v in views}
    geometry=OutdoorGeometryEvidence(a.evidence_store,chart_base_source='moge3_adaptive')
    geometry.configure_moge3_metric_scale(a.metric_scale)
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    excluded=set(FIXED+ADDITIONAL)
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    order=[i for i,v in enumerate(views) if i not in excluded and v.image_name.startswith(canonical+'__')
           and Path(v.image_name).stem in geometry.moge3_records]
    allowed={int(views[i].colmap_id) for i in order}
    source=foliage.observation_camera_ids[:,0]
    if not all(int(i) in allowed for i in torch.unique(source).tolist()):
        raise ValueError('Seed includes excluded or noncanonical source views')
    origins=torch.empty_like(foliage.xyz)
    for camera in torch.unique(source).tolist():origins[source==camera]=by_id[int(camera)].camera_center
    initial=foliage.xyz.detach().clone()
    source_views={int(camera):by_id[int(camera)] for camera in torch.unique(source).tolist()}
    targets_by_source={}
    source_rows={}
    if a.positive_evidence=='patch_stereo':
        centers=torch.stack([views[i].camera_center for i in order])
        directions=torch.stack([views[i].world_view_transform[:3,2] for i in order])
        for camera,view in source_views.items():
            distance=(centers-view.camera_center).norm(dim=1)
            facing=(directions*view.world_view_transform[:3,2]).sum(dim=1)>.8
            eligible=facing&(distance>.1)
            ranked=torch.where(eligible,distance,torch.full_like(distance,float('inf'))).argsort()
            selected=[int(i) for i in ranked[:6].tolist() if bool(eligible[int(i)])]
            targets_by_source[camera]={int(views[order[i]].colmap_id) for i in selected}
            source_rows[camera]=torch.nonzero(source==camera,as_tuple=False).flatten()
    shifts=torch.linspace(-a.log_radius,a.log_radius,a.depth_samples,device='cuda')
    factors=shifts.exp()[None].expand(len(initial),-1)
    bounded_search=torch.zeros(len(initial),device='cuda',dtype=torch.bool)
    if a.source_wall_depth_sweep:
        initial_z=torch.empty(len(initial),device='cuda')
        for camera,view in source_views.items():
            group=source==camera
            initial_z[group]=initial[group]@view.world_view_transform[:3,2]+view.world_view_transform[3,2]
        factors,bounded_search=source_wall_depth_factors(initial_z,
            capture['proposal_source_upper_depth'].to(initial_z),shifts,
            max(NATIVE_MIN_DEPTH+1.e-4,max(float(v.znear) for v in source_views.values())))
    points=origins[:,None]+(initial-origins)[:,None]*factors[:,:,None]
    n,width=points.shape[:2];flat=points.reshape(-1,3)
    admissible=None
    if capture.get('bearing_proposal_pool'):
        initial_z=torch.empty(n,device='cuda')
        for camera,view in source_views.items():
            group=source==camera
            initial_z[group]=initial[group]@view.world_view_transform[:3,2]+view.world_view_transform[3,2]
        upper=capture['proposal_source_upper_depth'].to(initial_z)
        candidate_z=initial_z[:,None]*factors
        admissible=(candidate_z<upper[:,None])&(candidate_z>NATIVE_MIN_DEPTH)
    positive=torch.zeros(n*width,dtype=torch.int16,device='cuda');negative=torch.zeros_like(positive)
    first=torch.full((n*width,),-1,dtype=torch.int32,device='cuda');second=first.clone()
    cache={}
    calibrations={}
    def fields(index):
        if index in cache:return cache[index]
        view=views[index];shape=(view.image_height,view.image_width)
        evidence=geometry.fields(view.image_name,device=torch.device('cuda'),shape=shape)
        _,bounds,valid,_,_=_moge3_depth_query_bounds(evidence,global_log_scale_sigma=0.)
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        canopy=obj&sky&distortion&~tree
        if a.positive_evidence=='moge':canopy &= valid
        rigid=obj&sky&distortion&tree
        # Do not turn semantic boundary noise into a global space-carving veto.
        empty=(rigid | (obj&distortion&~sky)).float()
        empty=(1-F.max_pool2d(1-empty[None,None],7,1,3)[0,0])>.5
        package=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device='cuda'),
            include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
            volume_gate=torch.zeros_like(foliage.opacity_logits.reshape(-1)))
        if capture.get('foreground_contrast_evidence'):
            from outdoor.canopy_foreground_evidence import foreground_contrast_evidence
            profiles=capture['foreground_contrast_profiles']
            profile=profiles.get(index,profiles.get(str(index)))
            if profile is None:raise ValueError('Missing immutable foreground evidence calibration')
            background=(package.render+(1-package.alpha)*(teacher.sky(view)-1)).clamp(0,1)
            canopy,_=foreground_contrast_evidence(view.original_image.cuda(),background,
                canopy,rigid,package.surface_alpha,profile=profile)
        if a.rigid_anchor_scale:
            if index not in calibrations:
                saved=capture.get('rigid_anchor_calibrations',{})
                record=saved.get(index,saved.get(str(index)))
                if record is None:raise ValueError('Missing immutable seed depth calibration for this camera')
                calibrations[index]=record
            from outdoor.canopy_depth_calibration import calibrated_canopy_evidence
            calibrated=calibrated_canopy_evidence(evidence,calibrations[index]['scale'])
            _,bounds,_,_,_=_moge3_depth_query_bounds(calibrated,global_log_scale_sigma=0.)
        result=(bounds,canopy,rigid,empty,package.median_depth,package.surface_alpha)
        del package
        if len(cache)>=6:cache.pop(next(iter(cache)))
        cache[index]=result
        return result
    records=[]
    for index in order:
        view=views[index];bounds,canopy,rigid,empty,wall,alpha=fields(index)
        wall=wall.reshape_as(canopy);alpha=alpha.reshape_as(canopy)
        transform=view.world_view_transform
        xyz=flat@transform[:3,:3]+transform[3,:3];z=xyz[:,2];safe=z.clamp_min(1e-6)
        u=(view.focal_x*xyz[:,0]/safe+view.cx).round().long()
        v=(view.focal_y*xyz[:,1]/safe+view.cy).round().long()
        inside=torch.isfinite(xyz).all(dim=1)&(z>NATIVE_MIN_DEPTH)&(u>=0)&(v>=0)&(u<view.image_width)&(v<view.image_height)
        rows=torch.nonzero(inside,as_tuple=False).flatten();u=u[rows];v=v[rows];z=z[rows]
        depth=wall[v,u];known=(alpha[v,u]>=.95)&torch.isfinite(depth)&(depth>0)
        front=~known | (z<depth-torch.maximum(torch.full_like(depth,.03),depth*.01))
        if a.positive_evidence=='moge':
            match=front&canopy[v,u]&(z>=bounds[0,v,u])&(z<=bounds[1,v,u])
        elif a.positive_evidence=='silhouette':
            match=front&canopy[v,u]
        else:
            photo_match=torch.zeros((n,width),device='cuda',dtype=torch.bool)
            for camera,source_view in source_views.items():
                if int(view.colmap_id) not in targets_by_source[camera]:continue
                source_indices=source_rows[camera]
                uv=foliage.observation_uv[source_indices,0]*torch.tensor(
                    [source_view.image_width,source_view.image_height],device='cuda')-.5
                depth=(initial[source_indices]@source_view.world_view_transform[:3,2]
                       +source_view.world_view_transform[3,2])
                scores=candidate_patch_scores(source_view,view,uv,depth,shifts)
                obj,sky,distortion,tree=masks.get_index_masks(source_view.image_name,(0,1,2,3),
                    (source_view.image_height,source_view.image_width),torch.device('cuda'))
                core=(1-F.max_pool2d((~(obj&sky&distortion&~tree)).float()[None,None],5,1,2)[0,0])>.5
                xy=uv.round().long()
                supported=core[xy[:,1],xy[:,0]]
                separation=max(2,round(.075/(2*a.log_radius/(a.depth_samples-1))))
                photo_match[source_indices]=unambiguous_depth_matches(scores,separation_bins=separation)&supported[:,None]
            core=(1-F.max_pool2d((~canopy).float()[None,None],5,1,2)[0,0])>.5
            match=front&core[v,u]&photo_match.reshape(-1)[rows]
            del photo_match
        conflict=front&empty[v,u]
        yes=rows[match];no=rows[conflict]
        first[yes[positive[yes]==0]]=int(view.colmap_id)
        second[yes[positive[yes]==1]]=int(view.colmap_id)
        positive[yes]+=1;negative[no]+=1
        record={'index':index,'depth_candidate_agreements':int(match.sum()),'visible_empty_conflicts':int(conflict.sum())}
        records.append(record);print(json.dumps(record),flush=True)
    chosen,keep=select_single_depth(positive.reshape(n,width),negative.reshape(n,width),shifts,admissible,
        frontmost=a.positive_evidence=='silhouette')
    rows=torch.nonzero(keep,as_tuple=False).flatten();selected_flat=rows*width+chosen[rows]
    positions=points[rows,chosen[rows]].clone()
    candidates=torch.stack((first[selected_flat],second[selected_flat]),dim=1)
    selected_stats={'proposed_rays':n,'retained_rays':len(rows),
        'changed_depth_rays':int(((factors[rows,chosen[rows]]-1).abs()>1.e-6).sum()),
        'full_source_wall_search_rays':int(bounded_search.sum()),
        'depth_shift_histogram':torch.bincount(chosen[rows],minlength=width).tolist(),
        'negative_view_counts':torch.bincount(negative[selected_flat].long()).tolist()}
    foliage.prune(~keep);foliage.xyz.copy_(positions)
    if a.preserve_source_footprint:foliage.log_scales[:,:2].add_(factors[rows,chosen[rows]].log()[:,None])
    foliage.support_camera_ids.fill_(-1);foliage.support_camera_ids[:,:2]=candidates
    foliage.support_view_count.fill_(2);foliage.support_sequence_count.fill_(1)
    foliage.verified_camera_ids.fill_(-1);foliage.verified_camera_count.zero_();foliage.verified_sequence_count.zero_()
    foliage.verification_state.fill_(VERIFICATION_UNVERIFIED);foliage.proposal_kind.fill_(PROPOSAL_RAY_BIRTH)
    foliage.birth_iteration.fill_(int(teacher.state['iteration']))
    lookup=torch.full((max(by_id)+1,),-1,dtype=torch.int16,device='cuda')
    for camera in allowed:lookup[camera]=0
    witness_records=[]
    for index in order:
        view=views[index];camera=int(view.colmap_id)
        exact=(foliage.support_camera_ids==camera).any(dim=1)
        if not exact.any():continue
        _,canopy,rigid,_,wall,alpha=fields(index)
        obj,sky_keep,distortion=masks.get_index_masks(view.image_name,(0,1,2),canopy.shape,torch.device('cuda'))
        known_sky=obj&distortion&~sky_keep
        package=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device='cuda'),
            include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
            volume_gate=static_detail_forward_visibility_gate(foliage,camera,include_pending_exact=True),
            audit_fields=canopy_material_audit_fields(canopy.float(),rigid.float(),known_sky.float(),view.original_image.cuda()))
        package.responsibility[package.structural_count:][~exact]=0
        audit=_update_static_child_verification_from_render(foliage,package,camera_id=camera,
            camera_sequence_lookup=lookup,required_sequence_count=1,
            canopy_color_audit=True,sky_audit_column=7,
            candidate_geometry_gate=front_of_known_rigid_mask(foliage.xyz,view,wall,alpha))
        witness_records.append({'index':index,**audit});del package
    audit={'diagnostic_only':True,'source_checkpoint':str(a.checkpoint.resolve()),'source_foliage':str(a.foliage.resolve()),
        'depth_log_radius':a.log_radius,'depth_samples':a.depth_samples,
        'positive_evidence':a.positive_evidence,
        'source_wall_depth_sweep':a.source_wall_depth_sweep,
        'native_candidate_min_depth':NATIVE_MIN_DEPTH,
        'foreground_contrast_evidence':bool(capture.get('foreground_contrast_evidence',False)),
        'foreground_contrast_profiles':capture.get('foreground_contrast_profiles',{}),
        'source_was_bearing_proposal_pool':bool(capture.get('bearing_proposal_pool',False)),
        'preserved_source_tangent_footprint':a.preserve_source_footprint,
        'rigid_anchor_scale':a.rigid_anchor_scale,'rigid_anchor_calibrations':calibrations,
        'scope':'fresh_dense_canonical_canopy_only__global_ray_depth_consensus__nonresumable',
        'excluded_view_indices':sorted(excluded),'source_view_indices':order,'selection':selected_stats,
        'verified_rows':int((foliage.verification_state==VERIFICATION_VERIFIED).sum()),
        'surface_fingerprints':{k:tensor_digest(getattr(teacher.surface,k)) for k in ('_xyz','_scaling','_rotation','_opacity')},
        'evidence_records':records,'witness_records':witness_records}
    torch.save({**audit,'foliage':foliage.capture()},a.output/'diagnostic_foliage_capture.pth')
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(selected_stats),flush=True)


if __name__=='__main__':main()
