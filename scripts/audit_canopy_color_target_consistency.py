"""Check production color objectives against an exactly self-consistent scene.

The target is the current native render, NOT a held-out photograph. This tests
whether an objective perturbs an already correct composite. It is not a quality
evaluation and all temporary Adam updates are restored before returning.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel,render_hybrid,static_detail_forward_visibility_gate
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.train_unified_outdoor_teacher import _static_volume_native_color_inputs,_static_volume_intrinsic_color_inputs,_photo_loss
from scripts.evaluate_canopy_validation import tensor_digest


def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','foliage','masks','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    capture=torch.load(a.foliage,map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Matched diagnostic required')
    leaf=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda');leaf.restore(capture['foliage']);teacher.foliage=leaf
    for parameter in leaf.parameters():parameter.requires_grad_(False)
    leaf.features.requires_grad_(True)
    for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'):getattr(teacher.surface,name).requires_grad_(False)
    for module in (teacher.sky,teacher.appearance):
        if hasattr(module,'parameters'):
            for parameter in module.parameters():parameter.requires_grad_(False)
    original=leaf.features.detach().clone()
    before={key:tensor_digest(getattr(leaf,key)) for key in ('xyz','log_scales','quaternions','features','opacity_logits')}
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks);records=[]
    for index in [657,660,690,713,768]:
        view=views[index];shape=(view.image_height,view.image_width)
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        canopy=obj&sky&dist&~tree
        with torch.no_grad():target=teacher.render(view,task=None,conditioned=False)['rgb'].detach()
        native=teacher.render(view,task=None,conditioned=False)
        prediction,weight=_static_volume_native_color_inputs(native['rgb'],native['volume_alpha'],canopy.float())
        new_loss=_photo_loss(prediction,target,weight,.10)
        new_grad=torch.autograd.grad(new_loss,leaf.features)[0]
        if torch.count_nonzero(new_grad):raise RuntimeError('Correct native composite received a material correction')
        old=render_hybrid(view,teacher.surface,leaf,background=torch.ones(3,device='cuda'),include_dynamic=False,
            optical_replacement_policy='disabled',structural_trainable_start=None,
            surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),volume_role_mask=leaf.static_leaf_mask,
            volume_gate=static_detail_forward_visibility_gate(leaf,int(view.colmap_id),include_pending_exact=False))
        intrinsic,old_weight=_static_volume_intrinsic_color_inputs(old.render,old.volume_alpha,torch.ones(3,device='cuda'),canopy.float())
        old_loss=_photo_loss(intrinsic,target,old_weight,.10)
        old_grad=torch.autograd.grad(old_loss,leaf.features)[0]
        row={'index':index,'new_self_consistency_loss':float(new_loss),'new_gradient_max':float(new_grad.abs().max()),
            'legacy_self_consistency_loss':float(old_loss),'legacy_gradient_norm':float(old_grad.norm())}
        try:
            optimizer=torch.optim.Adam([leaf.features],lr=.0025,eps=1.e-15)
            leaf.features.grad=old_grad.detach();leaf.features.grad[:,1:]=0;optimizer.step()
            with torch.no_grad():changed=teacher.render(view,task=None,conditioned=False)['rgb']
            row['legacy_one_step_canopy_native_mse']=float((changed[:,canopy]-target[:,canopy]).square().mean())
        finally:
            with torch.no_grad():leaf.features.copy_(original)
            leaf.features.grad=None
        records.append(row);print(json.dumps(row),flush=True)
        del old,native,optimizer,old_grad,new_grad
    assert before=={key:tensor_digest(getattr(leaf,key)) for key in before}
    (a.output/'audit.json').write_text(json.dumps({'scope':'self_consistent_native_target__NOT_photo_quality__all_updates_restored',
        'source_foliage':str(a.foliage.resolve()),'records':records},indent=2))


if __name__=='__main__':main()
