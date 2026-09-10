"""Screen rigid surfels against agreeing MoGe and original MAtCha wall depth.

Only known-rigid observations are used. This does not assume static tree
correspondences, and never mutates or exports a corrected checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid
from outdoor.training_evidence import OutdoorGeometryEvidence
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.train_unified_outdoor_teacher import _file_sha256


def surfel_depth_extent(transforms,world_view):
    """Six-sigma plane support also encloses the center-depth low-pass branch."""
    centers=transforms[:,3,:3]@world_view[:3,:3]+world_view[3,:3]
    sigma=(transforms[:,:2,:3]@world_view[:3,2]).square().sum(1).sqrt()
    return centers,centers[:,2]-6*sigma,centers[:,2]+6*sigma


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','evidence-store','support','masks','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--metric-scale',type=float,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={k:tensor_digest(getattr(teacher.surface,k)) for k in names}
    support=torch.load(a.support,map_location='cpu')
    if not support['complete_scan'] or support['surface_fingerprints']!=before:
        raise ValueError('Complete native visibility scan for this exact surface is required')
    geometry=OutdoorGeometryEvidence(a.evidence_store,chart_base_source='matcha')
    geometry.configure_moge3_metric_scale(a.metric_scale)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    excluded=set(FIXED+ADDITIONAL)
    selected=[i for i,v in enumerate(views) if i not in excluded and v.image_name in geometry.frame_by_stem]
    count=len(teacher.surface.get_xyz);free=torch.zeros(count,dtype=torch.int16,device='cuda')
    agreement=torch.zeros_like(free);rows=[]
    transform=teacher.surface.get_covariance().detach()
    for ordinal,index in enumerate(selected):
        view=views[index];shape=(view.image_height,view.image_width)
        fields=geometry.fields(view.image_name,device=torch.device('cuda'),shape=shape)
        if 'chart_depth' not in fields or 'moge3_depth_m' not in fields:continue
        chart=fields['chart_depth'][0];moge=fields['moge3_depth_m'][0]
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        valid=(obj&sky&dist&tree)&(fields['chart_weight'][0]>=.5)&(fields['moge3_valid_mask'][0]>.5)
        valid &= torch.isfinite(chart)&torch.isfinite(moge)&(chart>0)&(moge>0)
        valid &= (chart.clamp_min(1.e-6).log()-moge.clamp_min(1.e-6).log()).abs()<.10
        # Full local support, and conservative depth bounds across the window.
        core=F.avg_pool2d(valid.float()[None,None],7,1,3)[0,0]>=1-1.e-6
        near=-F.max_pool2d((-torch.minimum(chart,moge))[None,None],7,1,3)[0,0]*.8
        far=F.max_pool2d(torch.maximum(chart,moge)[None,None],7,1,3)[0,0]*1.2
        point,lower,upper=surfel_depth_extent(transform,view.world_view_transform)
        z=point[:,2];x=(point[:,0]/z*view.focal_x+view.cx).round().long()
        y=(point[:,1]/z*view.focal_y+view.cy).round().long()
        inside=torch.isfinite(point).all(1)&(z>0)&(x>=0)&(x<shape[1])&(y>=0)&(y<shape[0])
        x=x.clamp(0,shape[1]-1);y=y.clamp(0,shape[0]-1)
        observed=inside&core[y,x]
        package=render_hybrid(view,teacher.surface,teacher.foliage,
            background=torch.zeros(3,device='cuda'),include_dynamic=False,
            volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
            optical_replacement_policy='disabled',structural_trainable_start=None)
        rendered=package.median_depth[0,y,x]
        # This visibility screen is not per-primitive responsibility. Any
        # eventual correction must separately identify actually contributing rows.
        front_visible=(package.surface_alpha[0,y,x]>=.98)&(rendered>=lower-.05)&(rendered<=upper+.05)
        bad=observed&front_visible&(upper<near[y,x])
        good=observed&(z>=near[y,x])&(z<=far[y,x])
        free+=bad.to(free.dtype);agreement+=good.to(agreement.dtype)
        row=dict(index=index,image_name=view.image_name,known_rigid_agreeing_depth_pixels=int(core.sum()),
                 free_space_rows=int(bad.sum()),supported_rows=int(good.sum()))
        rows.append(row);print(json.dumps(dict(completed_views=ordinal+1,total_views=len(selected),**row)),flush=True)
    canopy=support['canopy_views'].to(device='cuda')
    candidate=(free>=2)&(agreement==0)&(canopy>=2)
    if before!={k:tensor_digest(getattr(teacher.surface,k)) for k in names}:
        raise RuntimeError('Read-only geometric screen changed surface')
    report=dict(scope='agreeing_original_matcha_and_moge_known_rigid_free_space_screen__no_correction',
        source_checkpoint=str(a.checkpoint.resolve()),source_checkpoint_sha256=_file_sha256(a.checkpoint),
        metric_scale=a.metric_scale,masks_sha256=_file_sha256(a.masks),
        visibility_support_sha256=_file_sha256(a.support),
        evidence_manifest_sha256=_file_sha256(a.evidence_store/'evidence_manifest.json'),
        surface_fingerprints=before,excluded_views=sorted(excluded),records=rows,
        candidate_rows=int(candidate.sum()),at_least_two_free_space_views=int((free>=2).sum()),
        any_consistent_geometry_support=int((agreement>0).sum()),
        limitations=['Source agreement is not ground truth','Center visibility is not individual surfel responsibility',
                    'Six-sigma depth enclosure is conservative but does not establish a correct photometric match'])
    torch.save(dict(diagnostic_only=True,free_views=free.cpu(),agreement_views=agreement.cpu(),
                    candidate=candidate.cpu(),surface_fingerprints=before),a.output/'support.pth')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
