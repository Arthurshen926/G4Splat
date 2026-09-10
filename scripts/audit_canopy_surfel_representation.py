"""Read-only representation test: thin leaves as ray-intersected 2D surfels.

This changes both projection and depth evaluation, so it is NOT a pure sorting
ablation or a production fix. Rigid tensors, leaf centers/colors/opacity and
learned visibility are unchanged; semantic masks only define report regions.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,VolumetricFoliageModel,static_detail_forward_visibility_gate
from outdoor.directional_sky import composite_white_background
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for name in ('checkpoint','foliage','masks','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--opacity-snapshot',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    capture=torch.load(a.foliage,map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Matched diagnostic required')
    leaf=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda')
    leaf.restore(capture['foliage']);teacher.foliage=leaf
    if leaf.dynamic_leaf_mask.any():raise ValueError('Static leaves only')
    if a.opacity_snapshot:
        snapshot=torch.load(a.opacity_snapshot,map_location='cpu')
        base={k:tensor_digest(getattr(leaf,k)) for k in ('xyz','log_scales','quaternions','features')}
        if snapshot.get('replay_base')!=base or snapshot.get('opacity_only_replay_valid') is not True:raise ValueError('Snapshot omitted learned state or has wrong leaf base')
        leaf.opacity_logits.copy_(snapshot['opacity_logits'].to(leaf.opacity_logits))
    rigid=teacher.surface;surface=GaussianModel(dataset.sh_degree)
    mapping={'_xyz':leaf.xyz,'_scaling':leaf.log_scales[:,:2],'_rotation':leaf.quaternions,
        '_opacity':leaf.opacity_logits,'_features_dc':leaf.features[:,:1],'_features_rest':leaf.features[:,1:]}
    for name,value in mapping.items():setattr(surface,name,torch.nn.Parameter(torch.cat((getattr(rigid,name),value)),requires_grad=False))
    surface.active_sh_degree=rigid.active_sh_degree
    empty=VolumetricFoliageModel(dataset.sh_degree,device='cuda')
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks);rows=[]
    for index in FIXED+ADDITIONAL:
        view=views[index];baseline=teacher.render(view,task=None,conditioned=False)['rgb']
        ownership=static_detail_forward_visibility_gate(leaf,int(view.colmap_id),include_pending_exact=False)
        gate=torch.cat((torch.ones(len(rigid.get_xyz),device='cuda'),ownership))
        package=render_hybrid(view,surface,empty,background=torch.ones(3,device='cuda'),include_dynamic=False,
            surface_gate=gate,optical_replacement_policy='disabled',structural_trainable_start=None)
        prediction=composite_white_background(package.render,package.alpha,teacher.sky(view))
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),(view.image_height,view.image_width),torch.device('cuda'))
        canopy=obj&sky&distortion&~tree;hard=obj&sky&distortion&tree
        _,outside,_=_tree_boundary_masks(~canopy)
        target=view.original_image.cuda();row={'index':index}
        for name,mask in [('tree',canopy),('rigid',hard),('hard',hard&outside)]:
            for mode,rgb in [('volume',baseline),('surfel',prediction)]:
                row[name+'_'+mode]=float(-10*torch.log10((rgb[:,mask]-target[:,mask]).square().mean().clamp_min(1e-12)))
        rows.append(row);print(json.dumps(row),flush=True)
        panel=torch.cat((target,baseline,prediction),dim=2).clamp(0,1)
        Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(a.output/f'view_{index}.png')
    summary={k:sum(row[k] for row in rows)/len(rows) for k in rows[0] if k!='index'}
    (a.output/'metrics.json').write_text(json.dumps({'args':vars(a),'summary':summary,'per_view':rows,
        'scope':'read_only_representation_comparison_not_pure_sorting_ablation_or_production_checkpoint'},default=str,indent=2))


if __name__=='__main__':main()
