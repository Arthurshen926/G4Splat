"""Native fixed-point test of geometry RGB; not a photo-quality evaluation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel,render_hybrid,static_detail_forward_visibility_gate
from outdoor.directional_sky import composite_white_background
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.train_unified_outdoor_teacher import _static_detail_canonical_ownership_gate,_static_detail_refinement_gate,_photo_loss
from scripts.evaluate_canopy_validation import tensor_digest


def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for name in ('checkpoint','foliage','masks','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    capture=torch.load(a.foliage,map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():
        raise ValueError('Matched diagnostic required')
    leaf=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda')
    leaf.restore(capture['foliage']);teacher.foliage=leaf
    for parameter in leaf.parameters():parameter.requires_grad_(False)
    leaf.xyz.requires_grad_(True);leaf.opacity_logits.requires_grad_(True)
    for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'):getattr(teacher.surface,name).requires_grad_(False)
    for module in (teacher.sky,teacher.appearance):
        if hasattr(module,'parameters'):
            for parameter in module.parameters():parameter.requires_grad_(False)
    before={key:tensor_digest(getattr(leaf,key)) for key in ('xyz','log_scales','quaternions','features','opacity_logits')}
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks);records=[]
    for index in [658,659,686,725,750]:
        view=views[index];shape=(view.image_height,view.image_width)
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        canopy=obj&sky&dist&~tree
        owner=_static_detail_refinement_gate(leaf,_static_detail_canonical_ownership_gate(leaf,int(view.colmap_id),None))*leaf.static_leaf_mask
        visibility=static_detail_forward_visibility_gate(leaf,int(view.colmap_id),include_pending_exact=False)
        with torch.no_grad():target=teacher.render(view,task=None,conditioned=False)['rgb'].detach()
        row={'index':index,'owners':int((owner>0).sum()),'native_visible_rows':int((visibility>0).sum())}
        for label,forward in [('native',visibility),('legacy_owner_subset',owner)]:
            package=render_hybrid(view,teacher.surface,leaf,background=torch.ones(3,device='cuda'),include_dynamic=False,
                optical_replacement_policy='disabled',structural_trainable_start=None,volume_gate=forward,
                volume_geometry_gradient_gate=owner,volume_opacity_gradient_gate=owner,
                volume_appearance_gradient_gate=torch.zeros_like(owner))
            prediction=composite_white_background(package.render,package.alpha,teacher.sky(view))
            loss=_photo_loss(prediction,target,canopy.float(),.10)
            gradients=torch.autograd.grad(loss,(leaf.xyz,leaf.opacity_logits))
            row[label]={'loss':float(loss),'xyz_gradient_norm':float(gradients[0].norm()),
                        'opacity_gradient_norm':float(gradients[1].norm()),
                        'nonowner_xyz_gradient_max':float(gradients[0][owner==0].abs().max()),
                        'native_rgb_max_error':float((prediction-target).abs().max())}
            if label=='native' and (float(loss)!=0 or any(torch.count_nonzero(g) for g in gradients)):
                raise RuntimeError('Native forward violates exact scene fixed point')
            if any(torch.count_nonzero(g[owner==0]) for g in gradients):raise RuntimeError('Nonowner received gradient')
            del package,prediction,gradients,loss
        records.append(row);print(json.dumps(row),flush=True)
    assert before=={key:tensor_digest(getattr(leaf,key)) for key in before}
    (a.output/'audit.json').write_text(json.dumps({'scope':'synthetic_native_fixed_point__not_photo_quality__no_parameter_changes',
        'source_foliage':str(a.foliage),'records':records},indent=2))


if __name__=='__main__':main()
