"""Read-only negative-supervision probe at reciprocal depth hypotheses.

Synthetic small volumes test loss authority, not actual training retirement.
No probe becomes model material and no source tensor is updated.
"""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid
# Bind this worktree's native root before importing the extension. An editable
# installation may otherwise resolve a different worktree's cached binary.
from diff_surfel_rasterization import GaussianRasterizationSettings,MixedGaussianRasterizer
from outdoor.training_evidence import OutdoorGeometryEvidence
from outdoor.evidence_store import load_evidence_store,artifact_path
from outdoor.task_fields import OutdoorTaskFieldLookup
from outdoor.canopy_support_expansion import clip_canopy_depth_queries_before_rigid
from outdoor.moge3_prehit_authority import observed_rigid_prehit_authority
from scripts.seed_bound_moge_diagnostic import configure_seed_bound_moge
from scripts.canopy_dense_depth_guidance import load_dense_guide
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.train_unified_outdoor_teacher import _moge3_depth_query_bounds,_moge3_canopy_optical_loss


def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ['checkpoint','initialization','evidence-store','guide','cohort','masks','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    fingerprints={k:tensor_digest(getattr(teacher.surface,k)) for k in ['_xyz','_opacity','_scaling','_rotation']}
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    cohort=json.loads(a.cohort.read_text())
    guide=load_dense_guide(a.guide,views,cohort['calibrated_training_views'],FIXED+ADDITIONAL)
    geometry=OutdoorGeometryEvidence(a.evidence_store,chart_base_source='moge3_adaptive')
    calibration=configure_seed_bound_moge(geometry,a.initialization,teacher.state['training_contract'],.8277335147998328)
    store=load_evidence_store(a.evidence_store,verify_hashes=False)
    tasks=OutdoorTaskFieldLookup(Path(dataset.source_path),a.masks,Path(store['semantic_contract']),
        multiview_track_archive=artifact_path(store,'mast3r_multiview_tracks'),
        projected_rigid_posterior_archive=artifact_path(store,'projected_rigid_conflict_posterior',required=False),max_cached_views=0)
    records=[]
    for item in guide:
        view=views[item['index']];x,y=item['uv'].unbind(1);z=item['depth']
        evidence=geometry.fields(view.image_name,device=torch.device('cuda'),shape=(view.image_height,view.image_width))
        task=tasks.fields(view.image_name,(view.image_height,view.image_width),torch.device('cuda'))
        pre,hit,valid,reliability,_=_moge3_depth_query_bounds(evidence,global_log_scale_sigma=calibration['global_log_scale_sigma'])
        with torch.no_grad():
            out=render_hybrid(view,teacher.surface,teacher.foliage,background=torch.zeros(3,device='cuda'),
                include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
                volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)))
            pre,hit,valid,ordering=clip_canopy_depth_queries_before_rigid(pre,hit,valid,task['p_canopy'],
                task['p_rigid'],out.median_depth,out.surface_alpha)
            authority=observed_rigid_prehit_authority(task,out.median_depth,out.surface_alpha)
            del out
        rays=torch.stack(((x-view.cx)/view.focal_x,(y-view.cy)/view.focal_y,torch.ones_like(z)),1)
        xyz=(rays*z[:,None])@view.world_view_transform[:3,:3].T+view.camera_center
        scales=(z/view.focal_x*.5)[:,None].expand(-1,3).contiguous()
        rotations=torch.zeros(len(z),4,device='cuda');rotations[:,0]=1
        logits=torch.full((len(z),1),math.log(.1/.9),device='cuda',requires_grad=True)
        empty=lambda n:torch.empty(0,n,device='cuda')
        settings=GaussianRasterizationSettings(image_height=view.image_height,image_width=view.image_width,
            tanfovx=math.tan(view.FoVx*.5),tanfovy=math.tan(view.FoVy*.5),bg=torch.zeros(3,device='cuda'),
            scale_modifier=1.,viewmatrix=view.world_view_transform,projmatrix=view.full_proj_transform,
            sh_degree=0,campos=view.camera_center,prefiltered=False,debug=False)
        _,_,maps,_,_=MixedGaussianRasterizer(settings)(empty(3),empty(3),empty(2),empty(4),xyz,
            torch.zeros_like(xyz),scales,rotations,torch.full_like(xyz,.5),logits.sigmoid(),depth_query_bounds=pre)
        selected=torch.zeros_like(valid,dtype=torch.float32);selected[y,x]=1
        task={**task,'w_rgb':task.get('w_rgb',torch.ones_like(selected))*selected}
        _,_,negative,_=_moge3_canopy_optical_loss(maps[11:12],maps[12:13]*0,task,valid,reliability,return_components=True)
        gradient=torch.autograd.grad(negative,logits,retain_graph=True)[0]
        _,_,safe_negative,_=_moge3_canopy_optical_loss(maps[11:12],maps[12:13]*0,task,valid,reliability,
            return_components=True,prehit_authority=authority)
        safe_gradient=torch.autograd.grad(safe_negative,logits)[0]
        if not torch.isfinite(gradient).all() or (gradient < -1e-8).any():raise ValueError('Invalid negative-only probe gradient')
        row=dict(index=item['index'],hypotheses=len(z),valid_after_live_rigid_clip=int(valid[y,x].sum()),
            hypothesized_centers_before_negative_bound=int((valid[y,x]&(z<pre[0,y,x])).sum()),
            probes_receiving_retirement_gradient=int((gradient>0).sum()),gradient_l1=float(gradient.abs().sum()),
            observed_rigid_only_retirement_probes=int((safe_gradient>0).sum()),
            observed_rigid_only_gradient_l1=float(safe_gradient.abs().sum()),
            negative_loss=float(negative),ordering=ordering)
        records.append(row);print(json.dumps(row),flush=True)
    if fingerprints!={k:tensor_digest(getattr(teacher.surface,k)) for k in fingerprints}:raise ValueError('Source surface changed')
    (a.output/'audit.json').write_text(json.dumps(dict(scope='synthetic_probes_at_observed_hypotheses__not_actual_retirement',
        calibration=calibration,records=records,source_surface_unchanged=True,
        limitations=['Hypotheses are not certified leaf material','No actual model row or optimizer step was modified']),indent=2))


if __name__=='__main__':main()
