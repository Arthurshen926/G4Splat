"""Read-only independent depth/semantic consensus, never a rendering mask."""
import argparse
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
import torch.nn.functional as F
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid
from outdoor.training_evidence import OutdoorGeometryEvidence
from outdoor.canopy_support_expansion import independent_canopy_camera_candidates
from outdoor.canopy_occluder_loss import OCCLUDER_CONSENSUS_CONTRACT
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds,_file_sha256
from outdoor.canopy_foreground_evidence import mixed_order_darkening_bound


def unproject_camera_z(depth,view):
    y,x=torch.meshgrid(torch.arange(depth.shape[0],device=depth.device,dtype=depth.dtype),
                       torch.arange(depth.shape[1],device=depth.device,dtype=depth.dtype),indexing='ij')
    camera=torch.stack(((x-view.cx)/view.focal_x*depth,
                        (y-view.cy)/view.focal_y*depth,depth),dim=-1)
    transform=view.world_view_transform.to(depth)
    return (camera-transform[3,:3])@transform[:3,:3].T


def front_interval_attribution(depth, bounds, front, eligible, alpha):
    """Partition interval overlap, not opacity GT or posterior probabilities."""
    lower, upper = bounds
    valid = (torch.isfinite(depth) & torch.isfinite(lower) & torch.isfinite(upper)
             & torch.isfinite(front) & (lower <= depth) & (depth <= upper))
    region = eligible & valid
    strict = region & (upper < front)
    partitions = {
        'full_interval_before': strict,
        'center_before_interval_crosses': region & ~strict & (depth < front),
        'center_behind_interval_crosses': region & (depth >= front) & (lower < front),
        'full_interval_behind': region & (lower >= front),
        'invalid_interval': eligible & ~valid,
    }
    if not torch.equal(sum(mask.to(torch.int32) for mask in partitions.values()), eligible.to(torch.int32)):
        raise RuntimeError('Depth interval attribution must be a disjoint complete partition')
    return {name: {'pixels': int(mask.sum()), 'native_alpha_below_0_5': int((mask & (alpha < .5)).sum())}
            for name, mask in partitions.items()}


@torch.no_grad()
def fixed_surface_rgb(teacher, view):
    from utils.sh_utils import eval_sh
    degree=min(int(teacher.surface.active_sh_degree),int(teacher.foliage.sh_degree))
    all_xyz=teacher.surface.get_xyz;all_features=teacher.surface.get_features
    chunks=[]
    for start in range(0,len(all_xyz),32768):
        xyz=all_xyz[start:start+32768]
        features=all_features[start:start+32768]
        direction=F.normalize(xyz-view.camera_center.reshape(1,3),dim=-1)
        rgb=(eval_sh(degree,features.transpose(1,2),direction)+.5).clamp_min(0)
        chunks.append(rgb)
    return torch.cat(chunks) if chunks else all_xyz.new_empty(0,3)


@torch.no_grad()
def background_contributor_ceiling(teacher, view):
    """All fixed surface colors plus sky; deliberately not a percentile cap."""
    rgb=fixed_surface_rgb(teacher,view)
    ceiling=teacher.sky(view).flatten(1).amax(1)
    return torch.maximum(ceiling,rgb.amax(0)) if len(rgb) else ceiling


@torch.no_grad()
def capped_contributor_backgrounds(teacher,view,caps,raw_reference):
    """Temporary color-only counterfactual; restore all source appearance."""
    surface=teacher.surface;colors=fixed_surface_rgb(teacher,view)
    dc=surface._features_dc.detach().clone();rest=surface._features_rest.detach().clone()
    degree=surface.active_sh_degree;sky=teacher.sky(view);results=[]
    try:
        surface._features_rest.zero_();surface.active_sh_degree=0
        for cap in (None,*caps):
            rgb=colors if cap is None else colors.clamp_max(cap)
            surface._features_dc.copy_(((rgb-.5)/.28209479177387814)[:,None])
            package=render_hybrid(view,surface,teacher.foliage,background=torch.zeros(3,device='cuda'),
                include_dynamic=False,volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
                optical_replacement_policy='disabled',structural_trainable_start=None)
            background=package.render+(1-package.alpha)*(sky if cap is None else sky.clamp_max(cap))
            if cap is None:
                if float((background-raw_reference).abs().max())>1.e-5:
                    raise RuntimeError('Baked fixed-contributor colors differ from native raw background')
            else:results.append(background)
    finally:
        surface._features_dc.copy_(dc);surface._features_rest.copy_(rest);surface.active_sh_degree=degree
    if not torch.equal(surface._features_dc,dc) or not torch.equal(surface._features_rest,rest):
        raise RuntimeError('Contributor-cap diagnostic changed source colors')
    return torch.stack(results)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','initialization','evidence-store','masks','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--views',type=int,default=12)
    p.add_argument('--neighbors',type=int,default=8)
    p.add_argument('--source-gate-audit-only',action='store_true',
                   help='Attribute source rejection gates; emits no training evidence masks')
    p.add_argument('--indices',type=int,nargs='+',
                   help='Explicit non-validation canopy training views for read-only attribution')
    p.add_argument('--save-depth-partition-panels',action='store_true')
    p.add_argument('--mixed-order-bound-audit',action='store_true',
                   help='Read-only applicability of order-independent raw-RGB extinction bound')
    p.add_argument('--clipped-contributor-bound-audit',action='store_true')
    a=p.parse_args()
    if a.views<1 or a.neighbors<2:raise ValueError('Positive views and at least two neighbors required')
    if a.clipped_contributor_bound_audit and not a.mixed_order_bound_audit:
        raise ValueError('Contributor-cap audit requires mixed-order raw-RGB bound audit')
    if (a.indices is not None or a.save_depth_partition_panels or a.mixed_order_bound_audit) and not a.source_gate_audit_only:
        raise ValueError('Explicit attribution views/panels must not emit training evidence')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    manifest_path=a.initialization/'initialization_manifest.json'
    contract=teacher.state['training_contract']
    if _file_sha256(manifest_path)!=contract['initialization_manifest_sha256']:
        raise ValueError('Initialization does not match checkpoint')
    manifest=json.loads(manifest_path.read_text());seed_path=Path(manifest['foliage_seed'])
    if _file_sha256(seed_path)!=contract['foliage_seed_sha256']:
        raise ValueError('Seed does not match checkpoint')
    seed=torch.load(seed_path,map_location='cpu')
    profiles=seed['audit']['fresh_canonical_leaves']['depth_profiles_by_image']
    scale=seed['audit']['moge3_exact_k_front_hit']['metric_to_cambridge_scale'];del seed
    geometry=OutdoorGeometryEvidence(a.evidence_store,chart_base_source='moge3_adaptive')
    geometry.configure_moge3_metric_scale(scale);geometry.configure_moge3_canopy_depth_scales(profiles)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    canonical=contract['static_scene_canonical_rgb']['canonical_sequence']
    excluded=set(FIXED+ADDITIONAL)
    train=[i for i,v in enumerate(views) if i not in excluded
           and v.image_name.startswith(canonical+'__') and v.image_name in profiles]
    if not train:raise ValueError('No calibrated nonvalidation cameras')
    cache={}
    def evidence(index):
        if index not in cache:
            view=views[index];shape=(view.image_height,view.image_width)
            obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
            canopy=obj&sky&distortion&~tree
            core=F.avg_pool2d(canopy.float()[None,None],7,1,3,count_include_pad=True)[0,0]>=1-1.e-6
            fields=geometry.fields(view.image_name,device=torch.device('cuda'),shape=shape)
            _,bounds,valid,_,_=_moge3_depth_query_bounds(fields,global_log_scale_sigma=0.)
            stable=(fields['moge3_refinement_log_depth_std'][0]<=.05)&(fields['moge3_refinement_final_delta_log_depth'][0].abs()<=.05)
            rigid=render_hybrid(view,teacher.surface,teacher.foliage,background=torch.zeros(3,device='cuda'),
                include_dynamic=False,volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
                optical_replacement_policy='disabled',structural_trainable_start=None)
            background=(rigid.render+(1-rigid.alpha)*teacher.sky(view)).clamp(0,1)
            background_error=(background-view.original_image.cuda()).abs().mean(dim=0)
            background_error=F.avg_pool2d(background_error[None,None],3,1,1)[0,0]
            rigid_mask=obj&sky&distortion&tree
            anchors=(F.avg_pool2d(rigid_mask.float()[None,None],7,1,3,count_include_pad=True)[0,0]>=1-1.e-6)
            anchors &= rigid.surface_alpha.reshape(shape)>=.98
            slack=(torch.quantile(background_error[anchors],.9)+.02 if int(anchors.sum())>=512
                   else background_error.new_tensor(float('inf')))
            different_from_background=background_error>slack
            cache[index]={k:v.cpu() for k,v in dict(depth=fields['moge3_canopy_depth_m'][0],
                bounds=bounds,valid=valid&core&stable&different_from_background,canopy=canopy,
                background_error_slack=slack,rigid_depth=rigid.median_depth.reshape(shape),
                rigid_alpha=rigid.surface_alpha.reshape(shape)).items()}
            if a.source_gate_audit_only:
                cache[index].update({k:v.cpu() for k,v in dict(core=core,depth_valid=valid,
                    stable=stable,different_from_background=different_from_background).items()})
                if a.save_depth_partition_panels:cache[index]['background']=background.cpu()
                if a.mixed_order_bound_audit:
                    raw_background=rigid.render+(1-rigid.alpha)*teacher.sky(view)
                    raw_error=(raw_background-view.original_image.cuda()).abs().amax(0)
                    bound_slack=(torch.quantile(raw_error[anchors],.95)+.02 if int(anchors.sum())>=512
                                 else raw_error.new_tensor(float('inf')))
                    cache[index].update(raw_background=raw_background.cpu(),bound_slack=bound_slack.cpu(),
                        radiance_ceiling=background_contributor_ceiling(teacher,view).cpu())
                    if a.clipped_contributor_bound_audit:
                        cache[index]['capped_backgrounds']=capped_contributor_backgrounds(
                            teacher,view,(.25,.5,1.,2.),raw_background).cpu()
        return {k:v.cuda() for k,v in cache[index].items()}
    eligible=[]
    for i in train:
        view=views[i]
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device('cpu'))
        if int((obj&sky&distortion&~tree).sum())>=128:eligible.append(i)
    if not eligible:raise ValueError('No canopy camera')
    selected=[eligible[j] for j in torch.linspace(0,len(eligible)-1,min(a.views,len(eligible))).round().long().tolist()]
    if a.indices is not None:
        if len(set(a.indices))!=len(a.indices) or not set(a.indices).issubset(eligible):
            raise ValueError('Distinct eligible non-validation canopy cameras required')
        selected=a.indices
    centers=torch.stack([views[i].camera_center.cpu() for i in train])
    before={k:tensor_digest(getattr(teacher.surface,k)) for k in ('_xyz','_scaling','_rotation','_opacity')}
    records=[]
    for index in selected:
        view=views[index];e=evidence(index);z=e['rigid_depth']
        clearance=torch.maximum(torch.full_like(z,.03),.01*z)
        candidate=e['valid']&(e['rigid_alpha']>=.98)&torch.isfinite(z)&(z>0)&(e['bounds'][1]<z-clearance)
        if a.source_gate_audit_only:
            native=teacher.render(view,task=None,conditioned=False)
            alpha=native['volume_alpha'].reshape_as(z)
            masks_by_gate=[('core',e['core']),('depth_valid',e['depth_valid']),('stable',e['stable']),
                ('different_from_background',e['different_from_background']),
                ('opaque_rigid',(e['rigid_alpha']>=.98)&torch.isfinite(z)&(z>0)),
                ('depth_before_rigid',e['bounds'][1]<z-clearance)]
            passing=e['canopy'].clone();stages={}
            for name,gate in masks_by_gate:
                passing &= gate
                stages[name]={'remaining':int(passing.sum()),
                              'remaining_native_alpha_below_0_5':int((passing&(alpha<.5)).sum()),
                              'independent_rejected_low_alpha':int((e['canopy']&(alpha<.5)&~gate).sum())}
            if not torch.equal(passing,candidate):raise RuntimeError('Gate audit changed the candidate definition')
            record=dict(index=index,canopy_pixels=int(e['canopy'].sum()),
                native_alpha_below_0_5=int((e['canopy']&(alpha<.5)).sum()),stages=stages,
                front_interval_attribution=front_interval_attribution(e['depth'],e['bounds'],z-clearance,
                    e['valid'] & (e['rigid_alpha']>=.98) & torch.isfinite(z) & (z>0),alpha))
            if a.mixed_order_bound_audit:
                bound_record={'radiance_ceiling':e['radiance_ceiling'].tolist(),
                              'slack':float(e['bound_slack']) if torch.isfinite(e['bound_slack']) else None,
                              'applicable_pixels':0,'deficit_pixels':0}
                if torch.isfinite(e['bound_slack']):
                    required=mixed_order_darkening_bound(view.original_image.cuda(),e['raw_background'],
                        e['radiance_ceiling'],e['bound_slack'])
                    applicable=e['core']&(e['rigid_alpha']>=.98)
                    if a.clipped_contributor_bound_audit:
                        bound_record['uncapped_ceiling_deficit_pixels']=int((applicable&(required>alpha+.01)).sum())
                        caps=(.25,.5,1.,2.)
                        for cap,background in zip(caps,e['capped_backgrounds']):
                            required=torch.maximum(required,mixed_order_darkening_bound(view.original_image.cuda(),
                                background,torch.full((3,),cap,device=z.device),e['bound_slack']))
                        bound_record['contributor_caps']=list(caps)
                    deficit=applicable&(required>alpha+.01)
                    bound_record.update(applicable_pixels=int(applicable.sum()),deficit_pixels=int(deficit.sum()),
                        deficit_native_alpha_below_0_5=int((deficit&(alpha<.5)).sum()),
                        deficit_outside_strict_depth_candidate=int((deficit&~candidate).sum()),
                        deficit_entire_interval_behind=int((deficit&e['depth_valid']&(e['bounds'][0]>=z-clearance)).sum()),
                        required_alpha_sum_on_deficit=float(required[deficit].sum()),
                        native_alpha_sum_on_deficit=float(alpha[deficit].sum()))
                record['mixed_order_darkening_bound']=bound_record
            if a.save_depth_partition_panels:
                behind=e['valid']&(e['rigid_alpha']>=.98)&torch.isfinite(z)&(z>0)&(e['bounds'][0]>=z-clearance)&(alpha<.5)
                if behind.any():
                    quantiles=torch.tensor([.1,.5,.9],device=z.device)
                    record['behind_low_alpha_depth_quantiles']={
                        'quantiles':[.1,.5,.9],
                        'moge_camera_z':torch.quantile(e['depth'][behind],quantiles).tolist(),
                        'rigid_camera_z':torch.quantile(z[behind],quantiles).tolist(),
                        'moge_minus_rigid_camera_z':torch.quantile((e['depth']-z)[behind],quantiles).tolist(),
                        'relative_depth_gap':torch.quantile((e['depth']/z-1)[behind],quantiles).tolist()}
                gt=view.original_image.cuda()
                overlay=torch.where(behind[None],.5*gt+.5*gt.new_tensor([1.,0.,1.])[:,None,None],gt)
                relative=(e['depth']/z.clamp_min(1.e-5)-1).clamp(-.5,.5)*2
                relative=torch.where(e['depth_valid']&torch.isfinite(relative)&(z>0),relative,torch.zeros_like(relative))
                heat=torch.stack((relative.clamp_min(0),torch.zeros_like(relative),(-relative).clamp_min(0)))
                panel=torch.cat((torch.cat((gt,native['rgb'],e['background']),2),
                    torch.cat((overlay,heat,alpha[None].expand(3,-1,-1)),2)),1).clamp(0,1)
                Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(
                    a.output/f'view_{index}_depth_partition.png')
            records.append(record);print(json.dumps(record),flush=True)
            continue
        points=unproject_camera_z(e['depth'],view)[candidate]
        distances=(centers-view.camera_center.cpu()).norm(dim=1)
        neighbors=[train[j] for j in distances.argsort().tolist() if train[j]!=index and float(distances[j])>.05][:a.neighbors]
        count=torch.zeros(len(points),dtype=torch.int16,device='cuda')
        ray0=F.normalize(points-view.camera_center.to(points),dim=1)
        per_camera=[]
        for other in neighbors:
            target=views[other];other_e=evidence(other)
            ray=F.normalize(points-target.camera_center.to(points),dim=1)
            # Exclude effectively duplicate rays, not just duplicate IDs.
            independent=(ray*ray0).sum(dim=1)<math.cos(math.radians(.25))
            rows=independent_canopy_camera_candidates(points,target,other_e['bounds'],other_e['valid'],independent,
                rigid_depth=other_e['rigid_depth'],rigid_alpha=other_e['rigid_alpha'])
            count[rows]+=1;per_camera.append(dict(index=other,consistent_pixels=len(rows)))
        confirmed=torch.zeros_like(candidate);confirmed[candidate]=count>=2
        record=dict(index=index,canopy_pixels=int(e['canopy'].sum()),source_candidate_pixels=int(candidate.sum()),
            background_error_slack=(float(e['background_error_slack']) if torch.isfinite(e['background_error_slack']) else None),
            one_independent_witness_pixels=int((count>=1).sum()),two_independent_witness_pixels=int(confirmed.sum()),
            neighbors=per_camera)
        records.append(record);torch.save({'candidate':candidate.cpu(),'confirmed':confirmed.cpu(),
            'scope':'immutable_evidence_diagnostic_only__not_renderer_gate'},a.output/f'evidence_{index}.pth')
        print(json.dumps(record),flush=True)
    assert before=={k:tensor_digest(getattr(teacher.surface,k)) for k in before}
    if a.source_gate_audit_only:
        (a.output/'audit.json').write_text(json.dumps(dict(
            scope='read_only_source_gate_attrition__low_alpha_is_not_ground_truth_error__no_training_evidence_emitted',
            source_checkpoint=str(a.checkpoint.resolve()),source_checkpoint_sha256=_file_sha256(a.checkpoint),
            selected_training_views=selected,excluded_views=sorted(excluded),records=records),indent=2))
        return
    (a.output/'audit.json').write_text(json.dumps(dict(scope='read_only_depth_semantic_consensus__not_native_primitive_verification',
        source_checkpoint=str(a.checkpoint.resolve()),source_checkpoint_sha256=_file_sha256(a.checkpoint),
        initialization_manifest_sha256=contract['initialization_manifest_sha256'],
        foliage_seed_sha256=contract['foliage_seed_sha256'],surface_fingerprints=before,
        calibrated_training_views=train,canopy_training_views=eligible,
        all_canopy_training_views_covered=(selected==eligible),
        selected_training_views=selected,excluded_views=sorted(excluded),records=records,
        contract=OCCLUDER_CONSENSUS_CONTRACT),indent=2))


if __name__=='__main__':main()
