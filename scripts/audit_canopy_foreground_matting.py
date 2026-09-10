"""Read-only foreground-layer darkening diagnostic; not a general opacity oracle.

The ratio bound assumes foliage in front of the frozen opaque background. Report
it only where independent wall depth and intrinsic/native alpha agree, with slack.
No target masks or inferred alpha are used to gate a deployment render.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
import torch.nn.functional as F
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel,render_hybrid,static_detail_forward_visibility_gate
from outdoor.directional_sky import composite_white_background
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from outdoor.canopy_foreground_evidence import foreground_darkening_bound as darkening_layer_bound


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);model=ModelParams(parser)
    parser.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','foliage','masks','output'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    capture=torch.load(a.foliage,map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Matched diagnostic required')
    leaf=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda');leaf.restore(capture['foliage'])
    teacher.foliage=leaf
    before={key:tensor_digest(getattr(leaf,key)) for key in ('xyz','opacity_logits','features')}
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks);records=[]
    for index in FIXED+ADDITIONAL:
        view=views[index];shape=(view.image_height,view.image_width)
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        canopy=obj&sky&distortion&~tree
        core=F.avg_pool2d(canopy.float()[None,None],7,1,3,count_include_pad=True)[0,0]>=1-1.e-6
        native=teacher.render(view,task=None,conditioned=False)
        wall=render_hybrid(view,teacher.surface,leaf,background=torch.ones(3,device='cuda'),include_dynamic=False,
            optical_replacement_policy='disabled',structural_trainable_start=None,volume_gate=torch.zeros(len(leaf),device='cuda'))
        background=composite_white_background(wall.render,wall.alpha,teacher.sky(view)).clamp(0,1)
        depth=wall.median_depth.reshape(shape)
        upper=depth-torch.maximum(torch.full_like(depth,.03),.01*depth)
        known=(wall.surface_alpha.reshape(shape)>=.99)&torch.isfinite(upper)&(upper>1.e-4)
        bounds=torch.stack((torch.full_like(depth,1.e-4),upper))
        bounds[:,~known]=float('nan')
        query=render_hybrid(view,teacher.surface,leaf,background=torch.zeros(3,device='cuda'),include_dynamic=False,
            optical_replacement_policy='disabled',structural_trainable_start=None,
            surface_gate=torch.zeros_like(teacher.surface.get_opacity.reshape(-1)),
            volume_gate=static_detail_forward_visibility_gate(leaf,int(view.colmap_id),include_pending_exact=False),volume_depth_query_bounds=bounds)
        front=query.volume_hit_interval_alpha.reshape(shape);alpha=native['volume_alpha'].reshape(shape)
        # This is a conservative diagnostic applicability check, not proof of
        # an exact single-layer scene at every pixel (AA / mixed surfaces remain).
        applicable=core&known&((front-alpha).abs()<.01)
        bound=darkening_layer_bound(view.original_image.cuda(),background)
        deficit=(bound-alpha).clamp_min(0)
        row={'index':index,'canopy_core_pixels':int(core.sum()),'applicable_pixels':int(applicable.sum()),
             'alpha_mean':float(alpha[applicable].mean()) if applicable.any() else None,
             'darkening_bound_mean':float(bound[applicable].mean()) if applicable.any() else None,
             'deficit_mean':float(deficit[applicable].mean()) if applicable.any() else None,
             'deficit_over_005_fraction':float((deficit[applicable]>.05).float().mean()) if applicable.any() else None}
        records.append(row);print(json.dumps(row),flush=True)
        del native,wall,query
    assert before=={key:tensor_digest(getattr(leaf,key)) for key in before}
    audit={'source_foliage':str(a.foliage.resolve()),'scope':'conditional_foreground_layer_darkening_diagnostic__not_general_depth_or_alpha_truth','records':records}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':main()
