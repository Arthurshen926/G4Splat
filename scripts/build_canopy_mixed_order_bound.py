"""Diagnostic fixed-background extinction bounds; no monocular-depth authority.

These are model-dependent necessary bounds with empirical photometric slack,
not measured opacity, geometry witnesses, or inference-time rendering masks.
"""
import argparse
import json
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
from outdoor.hybrid_gaussian_renderer import render_hybrid
from outdoor.canopy_foreground_evidence import mixed_order_darkening_bound
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.audit_moge_canopy_occluder_consensus import capped_contributor_backgrounds
from scripts.train_unified_outdoor_teacher import _file_sha256

CONTRACT='diagnostic_fixed_background_mixed_order_contributor_caps_v1'


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','masks','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--indices',type=int,nargs='+',help='Partial CUDA smoke only')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    source_stat=a.checkpoint.stat();source_sha=_file_sha256(a.checkpoint)
    masks_sha=_file_sha256(a.masks)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={name:tensor_digest(getattr(teacher.surface,name)) for name in names}
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    excluded=set(FIXED+ADDITIONAL)
    training=[i for i,v in enumerate(views) if i not in excluded and v.image_name.startswith(canonical+'__')]
    def regions(view,device):
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device(device))
        known=obj&sky&dist
        return known&~tree,known&tree
    eligible=[i for i in training if int(regions(views[i],'cpu')[0].sum())>=128]
    if not eligible:raise ValueError('No eligible canopy training cameras')
    selected=eligible if a.indices is None else a.indices
    if len(set(selected))!=len(selected) or not set(selected).issubset(eligible):
        raise ValueError('Distinct eligible non-validation cameras required')
    caps=(.25,.5,1.,2.);records=[]
    for ordinal,index in enumerate(selected):
        view=views[index];canopy,rigid=regions(view,'cuda');target=view.original_image.cuda()
        package=render_hybrid(view,teacher.surface,teacher.foliage,background=torch.zeros(3,device='cuda'),
            include_dynamic=False,volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
            optical_replacement_policy='disabled',structural_trainable_start=None)
        background=package.render+(1-package.alpha)*teacher.sky(view)
        alpha_b=package.surface_alpha.reshape_as(canopy)
        core=F.avg_pool2d(canopy.float()[None,None],7,1,3,count_include_pad=True)[0,0]>=1-1.e-6
        anchors=F.avg_pool2d(rigid.float()[None,None],7,1,3,count_include_pad=True)[0,0]>=1-1.e-6
        anchors &= alpha_b>=.98
        required=torch.zeros_like(alpha_b);applicable=torch.zeros_like(canopy);slack=None
        if int(anchors.sum())>=512:
            error=(background-target).abs().amax(0)
            slack=float(torch.quantile(error[anchors],.95))+.02
            capped=capped_contributor_backgrounds(teacher,view,caps,background)
            for cap,b in zip(caps,capped):
                required=torch.maximum(required,mixed_order_darkening_bound(target,b,
                    torch.full((3,),cap,device=b.device),slack))
            applicable=core&(alpha_b>=.98)
        native=teacher.render(view,task=None,conditioned=False)['volume_alpha'].reshape_as(canopy)
        deficit=applicable&(required>native+.01)
        row=dict(index=index,image_name=view.image_name,slack=slack,anchor_pixels=int(anchors.sum()),
            applicable_pixels=int(applicable.sum()),deficit_pixels=int(deficit.sum()),
            required_alpha_sum_on_deficit=float(required[deficit].sum()),native_alpha_sum_on_deficit=float(native[deficit].sum()),
            target_rgb_sha256=tensor_digest(view.original_image))
        torch.save(dict(required_alpha=required.cpu(),applicable=applicable.cpu(),diagnostic_only=True,
            contract=CONTRACT,image_name=view.image_name,target_rgb_sha256=row['target_rgb_sha256']),a.output/f'bound_{index}.pth')
        records.append(row);print(json.dumps({'completed_views':ordinal+1,'total_views':len(selected),**row}),flush=True)
    if before!={name:tensor_digest(getattr(teacher.surface,name)) for name in names}:
        raise RuntimeError('Read-only bound construction changed fixed surface parameters')
    after=a.checkpoint.stat()
    if (source_stat.st_ino,source_stat.st_size,source_stat.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError('Source checkpoint changed during construction')
    manifest=dict(contract=CONTRACT,source_checkpoint=str(a.checkpoint.resolve()),source_checkpoint_sha256=source_sha,
        masks_path=str(a.masks.resolve()),masks_sha256=masks_sha,
        surface_fingerprints=before,calibrated_training_views=training,excluded_views=sorted(excluded),
        selected_training_views=selected,all_canopy_training_views_covered=(selected==eligible),
        contributor_caps=list(caps),records=records,
        scope='diagnostic_fixed_background_bound__not_opacity_ground_truth__not_renderer_gate__no_moge_depth_used')
    (a.output/'audit.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'complete':True,'totals':{key:sum(row[key] for row in records) for key in
        ('applicable_pixels','deficit_pixels','required_alpha_sum_on_deficit','native_alpha_sum_on_deficit')}}),flush=True)


if __name__=='__main__':main()
