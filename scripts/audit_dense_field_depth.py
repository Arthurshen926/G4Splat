"""Read-only native depth residuals before/after a diagnostic static field."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate,persistent_static_evidence_mask
from outdoor.canopy_deformation_diagnostic import BoundedCanopyCenterField
from outdoor.moge3_evidence import sha256_file
from scripts.canopy_dense_depth_guidance import load_dense_guide
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ['checkpoint','field','guide','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    state=torch.load(a.field,map_location='cpu');args=state['args']
    if not state['diagnostic_only'] or args['rank'] or Path(state['source_checkpoint'])!=a.checkpoint:
        raise ValueError('Matching static diagnostic field required')
    cohort=json.loads(Path(args['cohort']).read_text())
    if sha256_file(a.checkpoint)!=cohort['source_checkpoint_sha256']:raise ValueError('Changed source model')
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree);foliage=teacher.foliage
    names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={k:tensor_digest(getattr(teacher.surface,k)) for k in names}
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    evidence=load_dense_guide(a.guide,views,state['training_views'],FIXED+ADDITIONAL)
    eligible=persistent_static_evidence_mask(foliage)&foliage.static_leaf_mask
    field=BoundedCanopyCenterField(foliage.xyz.detach(),eligible,state['field']['times'].tolist(),
        spacing=args['spacing'],radius=args['radius'],rank=0,reference_time=args['canonical_reference_frame'])
    field.load_state_dict(state['field'],strict=True)
    records=[]
    for mode in ['baseline','field_geometry_only','field_and_optics']:
        if mode=='field_and_optics' and state['shared_leaf_optics'] is not None:
            for name,value in state['shared_leaf_optics'].items():getattr(foliage,name).copy_(value.to(foliage.xyz.device))
        for item in evidence:
            view=views[item['index']];x,y=item['uv'].unbind(1)
            output=render_hybrid(view,teacher.surface,foliage,background=torch.zeros(3,device='cuda'),
                include_dynamic=False,optical_replacement_policy='disabled',structural_trainable_start=None,
                volume_gate=static_detail_forward_visibility_gate(foliage,int(view.colmap_id),include_pending_exact=False),
                volume_means_override=foliage.xyz if mode=='baseline' else foliage.xyz+field.offsets(None))
            depth=output.volume_depth[0,y,x];alpha=output.volume_alpha[0,y,x];target=item['depth']
            valid=(alpha>.01)&(depth>0)&torch.isfinite(depth)
            error=(depth[valid]/target[valid]).log().abs()
            records.append(dict(mode=mode,index=item['index'],visible_points=int(valid.sum()),
                mean_absolute_log_depth=float(error.mean()) if len(error) else None,
                within_two_percent=int((error<=.02).sum()),
                predicted_depth=depth.cpu().tolist(),target_depth=target.cpu().tolist(),alpha=alpha.cpu().tolist()))
    if before!={k:tensor_digest(getattr(teacher.surface,k)) for k in names}:raise ValueError('Rigid tensor changed')
    payload=dict(scope='training_hypothesis_depth_fit__not_static_validation',records=records,
        field_sha256=sha256_file(a.field),guide_sha256=sha256_file(a.guide),rigid_unchanged=True,
        limitations=['Expected depth can average distinct layers; low residual alone does not certify geometry'])
    (a.output/'audit.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k not in ['predicted_depth','target_depth','alpha']} for r in records]),flush=True)


if __name__=='__main__':main()
