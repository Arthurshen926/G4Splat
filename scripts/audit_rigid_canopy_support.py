"""Read-only per-surfel rigid support and unsupported-occluder counterfactual.

Every non-validation training camera protects any visible known-rigid support.
Counterfactual gates are diagnostic hypotheses, never deployment masks.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


def unsupported_canopy_surfels(rigid_views,canopy_views):
    if rigid_views.shape!=canopy_views.shape or (rigid_views<0).any() or (canopy_views<0).any():
        raise ValueError('Aligned nonnegative support counts required')
    return (rigid_views==0)&(canopy_views>=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','masks','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--maximum-views',type=int,default=0,help='Smoke only; partial scans never produce a removal counterfactual')
    a=p.parse_args()
    if a.maximum_views<0:raise ValueError('Nonnegative scan limit required')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    surface=teacher.surface
    names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={name:tensor_digest(getattr(surface,name)) for name in names}
    for name in names:getattr(surface,name).requires_grad_(False)
    for parameter in teacher.foliage.parameters():parameter.requires_grad_(False)
    surface._features_dc.requires_grad_(True)
    dc=surface._features_dc.detach().clone();rest=surface._features_rest.detach().clone()
    degree=surface.active_sh_degree
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    excluded=set(FIXED+ADDITIONAL)
    training=[i for i in range(len(views)) if i not in excluded]
    if a.maximum_views:training=training[:a.maximum_views]
    count=len(surface.get_xyz)
    rigid_views=torch.zeros(count,dtype=torch.int16,device='cuda')
    canopy_views=torch.zeros_like(rigid_views)
    rigid_mass=torch.zeros(count,device='cuda');canopy_mass=torch.zeros_like(rigid_mass)
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,
                optical_replacement_policy='disabled',structural_trainable_start=None)
    def regions(view):
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device('cuda'))
        known=obj&sky&dist
        return known&~tree,known&tree
    try:
        with torch.no_grad():
            surface._features_dc.zero_();surface._features_rest.zero_();surface.active_sh_degree=0
        for ordinal,index in enumerate(training):
            view=views[index];canopy,rigid=regions(view)
            if not view.image_name.startswith(canonical+'__'):canopy=torch.zeros_like(canopy)
            output=render_hybrid(view,surface,teacher.foliage,
                volume_gate=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1)),
                # The renderer normally detaches the complete rigid branch.
                # Enable only the already-authorized DC derivative for this
                # read-only responsibility measurement; there is no optimizer.
                **{**common,'structural_trainable_start':0})
            error=float((output.render.detach()-.5*output.surface_alpha.detach()).abs().max())
            if error>1.e-5:raise RuntimeError(f'Constant-radiance responsibility identity failed: {error}')
            objective=(output.render[0]*rigid).sum()+(output.render[1]*canopy).sum()
            gradient=torch.autograd.grad(objective,surface._features_dc)[0][:,0,:]/.28209479177387814
            if not torch.isfinite(gradient).all() or float(gradient.min()) < -1.e-5:
                raise RuntimeError('Invalid native responsibility gradient')
            g=gradient.detach().clamp_min(0)
            expected=torch.stack(((output.surface_alpha.detach().reshape_as(rigid)*rigid).sum(),
                                  (output.surface_alpha.detach().reshape_as(canopy)*canopy).sum()))
            if not torch.allclose(g[:,:2].sum(0),expected,rtol=2.e-5,atol=1.e-3):
                raise RuntimeError('Per-surfel responsibility does not conserve visible surface alpha')
            rigid_mass+=g[:,0];canopy_mass+=g[:,1]
            rigid_views+=(g[:,0]>1.e-6).to(torch.int16)
            canopy_views+=(g[:,1]>1.e-6).to(torch.int16)
            if ordinal%25==0 or ordinal+1==len(training):
                row=dict(completed_views=ordinal+1,total_views=len(training),view=index,
                    rigid_supported_rows=int((rigid_views>0).sum()),canopy_supported_rows=int((canopy_views>=2).sum()),
                    provisional_unsupported_canopy_rows=int(unsupported_canopy_surfels(rigid_views,canopy_views).sum()))
                print(json.dumps(row),flush=True)
                with (a.output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            del output,gradient,g,objective
    finally:
        with torch.no_grad():
            surface._features_dc.copy_(dc);surface._features_rest.copy_(rest);surface.active_sh_degree=degree
        surface._features_dc.requires_grad_(False)
    if before!={name:tensor_digest(getattr(surface,name)) for name in names}:
        raise RuntimeError('Rigid parameters changed in the read-only audit')
    candidate=unsupported_canopy_surfels(rigid_views,canopy_views)
    complete=len(training)==len(views)-len(excluded)
    report=dict(source_checkpoint=str(a.checkpoint.resolve()),surface_fingerprints=before,
        fitting_views=training,excluded_views=sorted(excluded),complete_scan=complete,
        scope='native_rigid_only_visibility_support__not_geometric_truth__no_source_parameter_changes',
        minimum_view_responsibility=1.e-6,
        surfels=count,unsupported_canopy_rows=int(candidate.sum()),counterfactual_per_view=[])
    torch.save(dict(rigid_views=rigid_views.cpu(),canopy_views=canopy_views.cpu(),
        rigid_mass=rigid_mass.cpu(),canopy_mass=canopy_mass.cpu(),candidate=candidate.cpu(),
        diagnostic_only=True,complete_scan=complete,source_checkpoint=str(a.checkpoint.resolve()),
        surface_fingerprints=before),a.output/'support.pth')
    if complete:
        with torch.no_grad():
            for index in FIXED+ADDITIONAL:
                view=views[index];canopy,rigid=regions(view);_,outside,_=_tree_boundary_masks(~canopy)
                baseline=teacher.render(view,task=None,conditioned=False)['rgb']
                output=render_hybrid(view,surface,teacher.foliage,surface_gate=(~candidate).float(),
                    volume_gate=static_detail_forward_visibility_gate(teacher.foliage,int(view.colmap_id),
                                                                     include_pending_exact=False),**common)
                prediction=(output.render+(1-output.alpha)*teacher.sky(view)).clamp(0,1)
                target=view.original_image.cuda();row={'index':index}
                for name,mask in (('tree',canopy),('rigid',rigid),('hard',rigid&outside)):
                    for mode,rgb in (('baseline',baseline),('counterfactual',prediction)):
                        row[name+'_'+mode]=float(-10*((rgb[:,mask]-target[:,mask]).square().mean().clamp_min(1.e-12)).log10()) if mask.any() else None
                report['counterfactual_per_view'].append(row)
                panel=torch.cat((target,baseline,prediction),2).clamp(0,1)
                Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(a.output/f'view_{index}.png')
        report['summary']={key:sum(row[key] for row in report['counterfactual_per_view'] if row[key] is not None)
            /sum(row[key] is not None for row in report['counterfactual_per_view'])
            for key in report['counterfactual_per_view'][0] if key!='index'}
    if before!={name:tensor_digest(getattr(surface,name)) for name in names}:
        raise RuntimeError('Counterfactual changed frozen rigid parameters')
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('fitting_views','counterfactual_per_view','surface_fingerprints')}),flush=True)


if __name__=='__main__':main()
