"""Isolated dense exact-K canopy initialization; never a resumable checkpoint.

Prespecified evaluation views are excluded from both seeding and witnessing.
One source observation initializes a measured-single leaf, not persistence.
A second matching depth and actual native foreground contribution is required.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import (VolumetricFoliageModel,render_hybrid,
    static_detail_forward_visibility_gate,VERIFICATION_MEASURED_SINGLE,VERIFICATION_VERIFIED,
    VERIFICATION_UNVERIFIED,PROPOSAL_RAY_BIRTH)
from outdoor.canopy_support_expansion import (independent_canopy_camera_candidates,
    temporary_camera_support,front_of_known_rigid_mask,canopy_material_audit_fields)
from outdoor.training_evidence import OutdoorGeometryEvidence
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.train_unified_outdoor_teacher import (_moge3_depth_query_bounds,
    _update_static_child_verification_from_render)
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest


def exact_world_points(uv,depth,view):
    camera=torch.stack(((uv[:,0]-view.cx)*depth/view.focal_x,
                        (uv[:,1]-view.cy)*depth/view.focal_y,depth,torch.ones_like(depth)),dim=1)
    return (camera@view.world_view_transform.to(camera).inverse())[:,:3]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);model=ModelParams(parser)
    parser.set_defaults(data_device='cpu',resolution=640,white_background=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--evidence-store',type=Path,required=True)
    parser.add_argument('--masks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metric-scale',type=float,required=True)
    parser.add_argument('--samples-per-view',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=1701)
    parser.add_argument('--rigid-anchor-scale',action='store_true')
    parser.add_argument('--production-base-initialization',type=Path)
    parser.add_argument('--production-rigid-ply',type=Path)
    parser.add_argument('--foreground-contrast-evidence',action='store_true',
        help='Use background-contrast positive evidence; weak tree-mask pixels remain unknown')
    parser.add_argument('--bearing-proposal-pool',action='store_true',
        help='Unverified ray hypotheses including conflicting/unknown MoGe depths; never directly renderable seeds')
    args=parser.parse_args()
    if bool(args.production_base_initialization) != bool(args.production_rigid_ply):
        raise ValueError('Production export requires both initialization and exact rigid PLY')
    if args.production_base_initialization and (not args.rigid_anchor_scale or args.bearing_proposal_pool or args.foreground_contrast_evidence):
        raise ValueError('Production export requires measured rigid-calibrated leaves without experimental contrast selection')
    if args.samples_per_view<1: raise ValueError('Positive sample budget required')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.45)
    dataset=model.extract(args);dataset.model_path=str(args.output)
    teacher=load_hybrid_teacher(args.checkpoint,sh_degree=dataset.sh_degree)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),args.masks)
    geometry=OutdoorGeometryEvidence(args.evidence_store,chart_base_source='moge3_adaptive')
    geometry.configure_moge3_metric_scale(args.metric_scale)
    excluded=set(FIXED+ADDITIONAL)
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    order=[i for i,v in enumerate(views) if i not in excluded and v.image_name.startswith(canonical+'__')
           and Path(v.image_name).stem in geometry.moge3_records]
    fingerprints={k:tensor_digest(getattr(teacher.surface,k)) for k in ('_xyz','_scaling','_rotation','_opacity')}
    base_initialization=None
    if args.production_base_initialization:
        from plyfile import PlyData
        base_initialization=json.loads((args.production_base_initialization/'initialization_manifest.json').read_text())
        ply=PlyData.read(str(args.production_rigid_ply))['vertex']
        raw_indices={'_xyz':1,'_scaling':4,'_rotation':5,'_opacity':6}
        for key,names in {'_xyz':['x','y','z'],'_scaling':['scale_0','scale_1'],
                          '_rotation':['rot_0','rot_1','rot_2','rot_3'],'_opacity':['opacity']}.items():
            value=torch.from_numpy(np.stack([np.asarray(ply[name]) for name in names],axis=1).astype(np.float32))
            if not torch.equal(value,teacher.state['surface'][raw_indices[key]].detach().cpu()):
                raise ValueError('Seed anchor raw geometry differs from the native rigid handoff')
            rendered=getattr(teacher.surface,key).detach().cpu()
            if key=='_scaling':
                # Frozen atlas export exp/log round-trips a small subset of
                # log scales. Bind this NEW initializer to the exact PLY,
                # not to those <=few-ULP export differences. No historical
                # artifact is modified and real atlas motion still fails.
                tolerance=4*torch.finfo(value.dtype).eps*value.abs().clamp_min(1)
                if not ((value-rendered).abs()<=tolerance).all():
                    raise ValueError('Live atlas scale differs materially from rigid handoff')
                teacher.surface._scaling.copy_(value.to(teacher.surface._scaling))
            elif not torch.equal(value,rendered):
                raise ValueError('Live atlas geometry differs from the native rigid handoff')
        fingerprints={k:tensor_digest(getattr(teacher.surface,k)) for k in fingerprints}
        del ply,value
    cache={}
    calibrations={}
    contrast_profiles={}
    def fields(index):
        if index in cache: return cache[index]
        view=views[index];shape=(view.image_height,view.image_width)
        evidence=geometry.fields(view.image_name,device=torch.device('cuda'),shape=shape)
        _,bounds,valid,_,_=_moge3_depth_query_bounds(evidence,global_log_scale_sigma=0.)
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        canopy=obj & sky & distortion & ~tree
        if not args.bearing_proposal_pool:canopy &= valid
        rigid=obj & sky & distortion & tree
        package=render_hybrid(view,teacher.surface,teacher.foliage,background=torch.ones(3,device='cuda'),
            include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
            volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)))
        if args.foreground_contrast_evidence:
            from outdoor.canopy_foreground_evidence import foreground_contrast_evidence
            background=(package.render+(1-package.alpha)*(teacher.sky(view)-1)).clamp(0,1)
            canopy,contrast_profiles[index]=foreground_contrast_evidence(
                view.original_image.cuda(),background,canopy,rigid,package.surface_alpha,
                profile=contrast_profiles.get(index))
        scale=1.
        if args.rigid_anchor_scale:
            from outdoor.canopy_depth_calibration import fit_rigid_depth_scale
            if index not in calibrations:
                calibrations[index]=fit_rigid_depth_scale(evidence['moge3_depth_m'],package.median_depth,package.surface_alpha,rigid&valid)
            scale=calibrations[index]['scale']
        from outdoor.canopy_depth_calibration import calibrated_canopy_evidence
        calibrated=calibrated_canopy_evidence(evidence,scale)
        _,bounds,_,_,_=_moge3_depth_query_bounds(calibrated,global_log_scale_sigma=0.)
        result=(calibrated['moge3_canopy_depth_m'][0].cpu(),bounds.cpu(),canopy.cpu(),rigid.cpu(),
                package.median_depth.cpu(),package.surface_alpha.cpu())
        del package
        if len(cache)>=8: cache.pop(next(iter(cache)))
        cache[index]=result
        return result
    parts=[];seed_records=[]
    for index in order:
        view=views[index];depth,bounds,canopy,rigid,wall,wall_alpha=fields(index)
        wall=wall.reshape_as(depth);wall_alpha=wall_alpha.reshape_as(depth)
        known=(wall_alpha>=.95)&torch.isfinite(wall)&(wall>0)
        front=~known | (depth<wall-torch.maximum(torch.full_like(wall,.03),wall*.01))
        upper=torch.where(known,wall-torch.maximum(torch.full_like(wall,.03),wall*.01),torch.full_like(wall,float('inf')))
        if args.bearing_proposal_pool:
            # This is a search location, NOT an observed leaf / depth witness.
            # Even an invalid monocular depth must not erase a real RGB bearing.
            depth=torch.where(known,torch.minimum(torch.where(torch.isfinite(depth)&(depth>0),depth,.75*upper),.90*upper),depth)
            front=torch.isfinite(depth)&(depth>0)
        pixels=torch.nonzero(canopy & front & (depth>.2),as_tuple=False)
        if len(pixels)<128: continue
        generator=torch.Generator().manual_seed(args.seed+1000003*int(view.colmap_id))
        selected=pixels[torch.randperm(len(pixels),generator=generator)[:args.samples_per_view]]
        z=depth[selected[:,0],selected[:,1]];uv=selected[:,[1,0]].float()
        xyz=exact_world_points(uv,z,view)
        spacing=(len(pixels)/len(selected))**.5
        scales=torch.stack(((z/view.focal_x*spacing*.6).clamp(.004,.15),
                            (z/view.focal_y*spacing*.6).clamp(.004,.15),torch.full_like(z,.02)),dim=1)
        # Camera tangent axes, not depth uncertainty as material thickness.
        xyzw=Rotation.from_matrix(view.world_view_transform[:3,:3].cpu().numpy()).as_quat()
        quaternion=torch.tensor(xyzw[[3,0,1,2]],dtype=torch.float32).expand(len(xyz),4).clone()
        colors=view.original_image[:,selected[:,0],selected[:,1]].T.cpu()
        ids=torch.full((len(xyz),4),-1,dtype=torch.int32);ids[:,0]=int(view.colmap_id)
        observation_uv=torch.full((len(xyz),4,2),float('nan'))
        observation_uv[:,0]=(uv+.5)/torch.tensor([view.image_width,view.image_height])
        observation_depth=torch.full((len(xyz),4),float('nan'));observation_depth[:,0]=z
        from outdoor.canopy_position_uncertainty import camera_ray_position_covariance
        global_sigma = (float(base_initialization['foliage']['moge3_exact_k_front_hit']['global_log_scale_sigma'])
                        if base_initialization is not None else .10)
        sigma=max(.025, global_sigma, float(calibrations.get(index,{}).get('robust_log_scatter',0.)))
        position_covariance=camera_ray_position_covariance(
            uv,z,torch.full_like(z,sigma),view.world_view_transform[:3,:3].cpu(),
            fx=view.focal_x,fy=view.focal_y,cx=view.cx,cy=view.cy)
        if args.bearing_proposal_pool:observation_depth.fill_(float('nan'))
        parts.append(dict(centers=xyz,scales=scales,colors=colors,quaternions=quaternion,
            position_covariance=position_covariance,
            support_camera_ids=ids,observation_camera_ids=ids.clone(),observation_uv=observation_uv,
            observation_depth=observation_depth,proposal_source_upper_depth=upper[selected[:,0],selected[:,1]]))
        record={'index':index,'camera_id':int(view.colmap_id),'valid_canopy_pixels':int(canopy.sum()),
                'front_pixels':len(pixels),'sampled':len(xyz)}
        seed_records.append(record);print(json.dumps({'seed':record}),flush=True)
    if not parts: raise RuntimeError('No calibrated foreground candidates')
    payload={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    # One spatial hypothesis per fine cell; never use duplicated source rays
    # to directly authorize a second measured or render witness.
    _,selected=np.unique(torch.floor(payload['centers']/.015).long().numpy(),axis=0,return_index=True)
    selected=torch.from_numpy(np.sort(selected));payload={k:v[selected] for k,v in payload.items()}
    proposal_upper=payload.pop('proposal_source_upper_depth')
    n=len(selected)
    payload.update(version='independent_sfm_semantic_canopy_volume_v1',
        opacities=torch.full((n,1),.04),static_detail=torch.ones(n,dtype=torch.bool),
        initialization_source=torch.full((n,),4,dtype=torch.int8),
        support_view_count=torch.ones(n,dtype=torch.int16),support_sequence_count=torch.ones(n,dtype=torch.int16),
        verification_state=torch.full((n,),VERIFICATION_MEASURED_SINGLE,dtype=torch.int8),
        verified_camera_ids=payload['support_camera_ids'].clone(),verified_camera_count=torch.ones(n,dtype=torch.int16),
        verified_sequence_count=torch.ones(n,dtype=torch.int16))
    if args.bearing_proposal_pool:
        payload['verification_state'].fill_(VERIFICATION_UNVERIFIED)
        payload['verified_camera_ids'].fill_(-1);payload['verified_camera_count'].zero_();payload['verified_sequence_count'].zero_()
        payload['proposal_kind']=torch.full((n,),PROPOSAL_RAY_BIRTH,dtype=torch.int8)
        payload['birth_iteration']=torch.full((n,),int(teacher.state['iteration']),dtype=torch.int64)
    foliage=VolumetricFoliageModel(dataset.sh_degree,device='cuda');foliage.initialize_from_volume_state(payload)
    teacher.foliage=foliage
    lookup=torch.full((max(int(v.colmap_id) for v in views)+1,),-1,dtype=torch.int16,device='cuda')
    for index in order: lookup[int(views[index].colmap_id)]=0
    witness_records=[]
    for index in ([] if args.bearing_proposal_pool else order):
        view=views[index];camera=int(view.colmap_id)
        eligible=(foliage.verification_state==VERIFICATION_MEASURED_SINGLE)&~(foliage.support_camera_ids==camera).any(dim=1)
        # No candidate for THIS camera does not imply that later independent
        # cameras cannot witness the remaining measured-single leaves.
        if not eligible.any(): continue
        _,bounds,canopy,rigid,wall,wall_alpha=fields(index)
        bounds=bounds.cuda();canopy=canopy.cuda();rigid=rigid.cuda();wall=wall.cuda();wall_alpha=wall_alpha.cuda()
        rows=independent_canopy_camera_candidates(foliage.xyz,view,bounds,canopy,eligible,
            rigid_depth=wall,rigid_alpha=wall_alpha)
        if not len(rows): continue
        obj,sky_keep,distortion=masks.get_index_masks(view.image_name,(0,1,2),canopy.shape,torch.device('cuda'))
        known_sky=obj&distortion&~sky_keep
        with temporary_camera_support(foliage,rows,camera) as audit:
            package=render_hybrid(view,teacher.surface,foliage,background=torch.ones(3,device='cuda'),
                include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
                volume_gate=static_detail_forward_visibility_gate(foliage,camera,include_pending_exact=True),
                audit_fields=canopy_material_audit_fields(canopy.float(),rigid.float(),known_sky.float(),view.original_image.cuda()))
            allowed=torch.zeros(n,dtype=torch.bool,device='cuda');allowed[rows]=True
            package.responsibility[package.structural_count:][~allowed]=0
            result=_update_static_child_verification_from_render(foliage,package,camera_id=camera,
                camera_sequence_lookup=lookup,required_sequence_count=1,
                canopy_color_audit=True,sky_audit_column=7,
                candidate_geometry_gate=front_of_known_rigid_mask(foliage.xyz,view,wall,wall_alpha))
            del package
        record={'index':index,**audit,**result};witness_records.append(record)
        print(json.dumps({'witness':record}),flush=True)
    assert fingerprints=={k:tensor_digest(getattr(teacher.surface,k)) for k in fingerprints}
    audit={'diagnostic_only':True,'source_checkpoint':str(args.checkpoint.resolve()),
        'scope':'fresh_dense_canonical_canopy_only__unchanged_rigid__nonresumable__48_views_excluded_from_seed_and_verification',
        'excluded_view_indices':sorted(excluded),'source_view_indices':order,'rows':n,
        'verified_rows':int((foliage.verification_state==VERIFICATION_VERIFIED).sum()),
        'rigid_anchor_scale':args.rigid_anchor_scale,'rigid_anchor_calibrations':calibrations,
        'bearing_proposal_pool':args.bearing_proposal_pool,
        'foreground_contrast_evidence':args.foreground_contrast_evidence,
        'foreground_contrast_profiles':contrast_profiles,
        'surface_fingerprints':fingerprints,'seed_records':seed_records,'witness_records':witness_records}
    if args.bearing_proposal_pool:
        audit['scope']='unverified_canopy_bearing_candidate_pool__not_a_renderable_or_trainable_model'
    torch.save({**audit,'foliage':foliage.capture(),'proposal_source_upper_depth':proposal_upper},args.output/'diagnostic_foliage_capture.pth')
    if base_initialization is not None:
        from outdoor.canonical_canopy_initialization import CONTRACT,validate_fresh_canonical_seed
        # Export measured initialization, not a trained capture/resume. No
        # optimizer ran; all native witness metadata remains attached.
        fresh={key:getattr(foliage,key).detach().cpu() for key in foliage.metadata_names}
        fresh.update(version=payload['version'],centers=foliage.xyz.detach().cpu(),
            scales=foliage.log_scales.detach().exp().cpu(),quaternions=foliage.quaternions.detach().cpu(),
            colors=(foliage.features[:,0].detach()*.28209479177387814+.5).clamp(0,1).cpu(),
            opacities=foliage.opacity_logits.detach().sigmoid().cpu())
        canonical_audit={'contract':CONTRACT,'optimizer_steps':0,'proposal_pool':False,
            'canonical_sequence':canonical,'excluded_camera_ids':[int(views[i].colmap_id) for i in sorted(excluded)],
            'depth_profiles_by_image':{Path(views[i].image_name).stem:calibrations[i]['scale'] for i in order},
            'source_diagnostic':str(args.output.resolve()),'rigid_geometry_fingerprints':fingerprints}
        from outdoor.canopy_position_uncertainty import CONTRACT as POSITION_CONTRACT
        canonical_audit['position_uncertainty_contract']=POSITION_CONTRACT
        moge_audit=copy.deepcopy(base_initialization['foliage']['moge3_exact_k_front_hit'])
        if abs(float(moge_audit['metric_to_cambridge_scale'])-args.metric_scale)>1.e-10:
            raise ValueError('Production base and canopy gauge differ')
        fresh['audit']={'moge3_exact_k_front_hit':moge_audit,'fresh_canonical_leaves':canonical_audit}
        fixed=[{'image_id':int(v.colmap_id),'image_name':str(v.image_name),
                'sequence_id':str(v.image_name).split('__')[0]} for v in views]
        validate_fresh_canonical_seed(fresh,fixed)
        destination=args.output/'production_initialization';destination.mkdir(exist_ok=False)
        seed_path=destination/'foliage_seed_gaussians.pth';torch.save(fresh,seed_path)
        # Stream large assets without retaining an extra checkpoint in RAM.
        def file_digest(path):
            result=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
            return result.hexdigest()
        structural_hash=file_digest(args.production_rigid_ply)
        manifest=copy.deepcopy(base_initialization)
        manifest['version']='outdoor-fresh-canonical-rigid-anchored-leaves-v149'
        manifest['foliage_seed']=str(seed_path.resolve())
        manifest['foliage']={**fresh['audit'],'count':n,
            'rigid_depth_calibration':{'enabled':True,'structural_ply_sha256':structural_hash,
                'source':'exact_native_rigid_handoff_geometry','depth_view_count':len(calibrations)}}
        manifest['initialization_contract']['foliage_reuse']={'policy':CONTRACT,
            'foliage_seed_sha256':file_digest(seed_path),'mixed_training_eligible':True,
            'optimizer_steps':0,'source_checkpoint_used_only_for_verified_equal_rigid_geometry':str(args.checkpoint.resolve())}
        manifest['initialization_contract'].update(
            selected_foliage_views=len(order),maximum_dense_rays_per_foliage_view=args.samples_per_view,
            maximum_dense_rays_total=args.samples_per_view*len(order),maximum_foliage_voxels=n,
            maximum_sfm_static_tree_tracks=0,maximum_dynamic_births_per_foliage_view=0,
            maximum_weak_continuous_dynamic_births_per_foliage_view=0)
        manifest['fresh_seed_provenance']={'builder_sha256':file_digest(Path(__file__)),
            'base_manifest':str((args.production_base_initialization/'initialization_manifest.json').resolve()),
            'base_manifest_sha256':file_digest(args.production_base_initialization/'initialization_manifest.json'),
            'rigid_geometry_bitwise_verified':True,'learned_foliage_checkpoint_imported':False,
            'surface_seed_unchanged':True,'training_implementation_must_validate_excluded_camera_streams':True}
        (destination/'initialization_manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'completed':True,'rows':n,'verified_rows':audit['verified_rows']}),flush=True)


if __name__=='__main__': main()
