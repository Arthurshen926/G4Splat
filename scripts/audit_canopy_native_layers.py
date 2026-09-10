"""Exact color contribution decomposition with native occlusion unchanged."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
import numpy as np
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel,render_hybrid,static_detail_forward_visibility_gate
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import tensor_digest


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);model=ModelParams(parser)
    parser.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','masks','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--foliage',type=Path,help='Optional matched diagnostic replacement; omit for production checkpoint')
    parser.add_argument('--dense-regions',type=Path,
        help='Explicit posthoc diagnostic ROIs; never affect rendering or optimization')
    parser.add_argument('--intrinsic-visibility-counterfactual',action='store_true',
        help='Read-only surface-disabled counterfactual; never a deployment repair')
    parser.add_argument('--opacity-ceiling',type=float,
        help='Explicit run-specific parameter ceiling for contribution-weighted saturation audit')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.opacity_ceiling is not None and not 0<a.opacity_ceiling<1:
        raise ValueError('Opacity ceiling must be in (0,1)')
    regions=json.loads(a.dense_regions.read_text()) if a.dense_regions else None
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    if a.foliage is not None:
        capture=torch.load(a.foliage,map_location='cpu')
        if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Matched diagnostic required')
        leaf=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda');leaf.restore(capture['foliage']);teacher.foliage=leaf
    else:
        leaf=teacher.foliage
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    leaf_colors=leaf.features.detach().clone()
    surface_dc=teacher.surface._features_dc.detach().clone();surface_rest=teacher.surface._features_rest.detach().clone()
    before=(tensor_digest(leaf.features),tensor_digest(teacher.surface._features_dc),tensor_digest(teacher.surface._features_rest))
    rows=[]
    for index in [657,660,690,713,768]:
        view=views[index]
        common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,optical_replacement_policy='disabled',
            structural_trainable_start=None,volume_gate=static_detail_forward_visibility_gate(leaf,int(view.colmap_id),include_pending_exact=False))
        complete=render_hybrid(view,teacher.surface,leaf,**common)
        intrinsic=None
        if a.intrinsic_visibility_counterfactual:
            intrinsic=render_hybrid(view,teacher.surface,leaf,
                surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),**common)
        try:
            leaf.features.zero_();leaf.features[:,0]=-.5/.28209479177387814
            surface=render_hybrid(view,teacher.surface,leaf,**common).render.detach()
        finally:leaf.features.copy_(leaf_colors)
        try:
            teacher.surface._features_dc.fill_(-.5/.28209479177387814);teacher.surface._features_rest.zero_()
            foliage=render_hybrid(view,teacher.surface,leaf,**common).render.detach()
        finally:
            teacher.surface._features_dc.copy_(surface_dc);teacher.surface._features_rest.copy_(surface_rest)
        error=float((surface+foliage-complete.render).abs().max())
        if error>1.e-5:raise RuntimeError(f'Color decomposition failed: {error}')
        prediction=(complete.render+(1-complete.alpha)*teacher.sky(view)).clamp(0,1)
        api_prediction=teacher.render(view,task=None,conditioned=False)['rgb']
        api_error=float((prediction-api_prediction).abs().max())
        if api_error>1.e-5:raise RuntimeError(f'Diagnostic diverged from native API: {api_error}')
        conditional=(foliage/complete.volume_alpha.clamp_min(1.e-8)).clamp(0,1)
        alpha=complete.volume_alpha.expand(3,-1,-1)
        # Row1: GT / native / actual leaf contribution. Row2: rigid
        # contribution / leaf color conditional on contribution / leaf alpha.
        top=torch.cat((view.original_image.cuda(),prediction,foliage),dim=2)
        bottom=torch.cat((surface,conditional,alpha),dim=2)
        panel=torch.cat((top,bottom),dim=1).clamp(0,1)
        Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)).save(a.output/f'view_{index}_layers.png')
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),prediction.shape[1:],torch.device('cuda'))
        canopy=obj&sky&distortion&~tree
        record={'index':index,'sum_max_error':error,'api_max_error':api_error,
            'canopy_volume_alpha':float(complete.volume_alpha[0][canopy].mean()),
            'canopy_surface_alpha':float(complete.surface_alpha[0][canopy].mean()),
            'canopy_surface_rgb_mean':float(surface[:,canopy].mean()),'canopy_leaf_rgb_mean':float(foliage[:,canopy].mean())}
        if intrinsic is not None:
            record['intrinsic_leaf_alpha_mean']=float(intrinsic.volume_alpha[0][canopy].mean())
            record['rigid_occlusion_alpha_gap']=float((intrinsic.volume_alpha-complete.volume_alpha)[0][canopy].mean())
        if regions is not None and str(index) in regions['regions']:
            if regions['resolution']!=[view.image_width,view.image_height]:raise ValueError('ROI resolution mismatch')
            x0,y0,x1,y1=regions['regions'][str(index)]
            if not 0<=x0<x1<=view.image_width or not 0<=y0<y1<=view.image_height:raise ValueError('Invalid dense ROI')
            owned=torch.zeros_like(canopy);owned[y0:y1,x0:x1]=canopy[y0:y1,x0:x1]
            if not owned.any():raise ValueError('Empty dense diagnostic ROI')
            target=view.original_image.cuda()
            from scripts.audit_canopy_saved_detail import detail_metrics
            record['dense_region']={'box_xyxy':[x0,y0,x1,y1],'pixels':int(owned.sum()),
                'native_psnr':float(-10*((prediction[:,owned]-target[:,owned]).square().mean().clamp_min(1e-12)).log10()),
                'leaf_alpha_mean':float(complete.volume_alpha[0][owned].mean()),
                'rigid_alpha_mean':float(complete.surface_alpha[0][owned].mean()),
                'rigid_rgb_contribution_mean':float(surface[:,owned].mean()),
                'leaf_alpha_below_0_9_fraction':float((complete.volume_alpha[0][owned]<.9).float().mean()),
                **detail_metrics(prediction,target,owned)}
            if intrinsic is not None:
                record['dense_region'].update(
                    intrinsic_leaf_alpha_mean=float(intrinsic.volume_alpha[0][owned].mean()),
                    rigid_occlusion_alpha_gap=float((intrinsic.volume_alpha-complete.volume_alpha)[0][owned].mean()),
                    opaque_without_rigid_but_translucent_native_fraction=float(
                        ((intrinsic.volume_alpha[0][owned]>.99)&(complete.volume_alpha[0][owned]<.9)).float().mean()))
            if a.opacity_ceiling is not None:
                witnessed=render_hybrid(view,teacher.surface,leaf,audit_fields=owned.float()[None],**common)
                weights=witnessed.responsibility[witnessed.structural_count:,1]
                at_cap=leaf.opacities.reshape(-1)>=a.opacity_ceiling-1.e-4
                alpha_sum=complete.volume_alpha[0][owned].sum()
                if not torch.allclose(weights.sum(),alpha_sum,rtol=2.e-4,atol=.01):
                    raise RuntimeError('Dense responsibility is not the native pixel contribution')
                record['dense_region']['opacity_ceiling_audit']={
                    'ceiling':a.opacity_ceiling,'visible_rows':int((weights>0).sum()),
                    'visible_rows_at_cap':int(((weights>0)&at_cap).sum()),
                    'native_contribution_at_cap_fraction':float(weights[at_cap].sum()/weights.sum().clamp_min(1.e-12)),
                    'global_rows_at_cap_fraction':float(at_cap.float().mean())}
                del witnessed
        rows.append(record)
        del complete
    assert before==(tensor_digest(leaf.features),tensor_digest(teacher.surface._features_dc),tensor_digest(teacher.surface._features_rest))
    audit={'scope':'native_color_decomposition__geometry_opacity_ordering_unchanged',
           'intrinsic_visibility_counterfactual':('surface_gate_zero__causal_occluder_removal__not_deployment' if a.intrinsic_visibility_counterfactual else None),
           'panel_layout':[['GT','native','leaf contribution'],['rigid contribution','conditional leaf color','leaf alpha']],
           'source_foliage':str((a.foliage or a.checkpoint).resolve()),'records':rows}
    if regions is not None:audit['dense_regions']=regions
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
