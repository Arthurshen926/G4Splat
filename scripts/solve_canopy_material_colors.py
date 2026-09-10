"""Fixed-transport color solve; geometry, opacity, authority and rigid scene frozen."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
import torch.nn.functional as F
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import VolumetricFoliageModel,render_hybrid,static_detail_forward_visibility_gate,VERIFICATION_VERIFIED
from outdoor.canopy_color_solver import majorized_color_update
from outdoor.canopy_detail_loss import masked_canopy_ssim_loss
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__);model=ModelParams(parser)
    parser.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','foliage','masks','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--global-only',action='store_true')
    parser.add_argument('--canopy-only',action='store_true',
        help='Material-correct color objective: rigid/sky errors are not appearance observations for leaves')
    a=parser.parse_args()
    if a.epochs<1:raise ValueError('Positive epoch count required')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree);reference_leaf=teacher.foliage
    capture=torch.load(a.foliage,map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Matched diagnostic required')
    leaf=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda');leaf.restore(capture['foliage'])
    if leaf.dynamic_leaf_mask.any() or torch.count_nonzero(leaf.features[:,1:]):raise ValueError('Static DC-only diagnostic required for this linear solve')
    private_rows=int((leaf.verification_state!=VERIFICATION_VERIFIED).sum())
    if a.global_only:leaf.prune(leaf.verification_state!=VERIFICATION_VERIFIED)
    teacher.foliage=leaf
    eligible=leaf.verification_state==VERIFICATION_VERIFIED
    invariants={key:tensor_digest(getattr(leaf,key)) for key in ('xyz','log_scales','quaternions','opacity_logits','verification_state','verified_camera_ids','support_camera_ids')}
    surface_fingerprints={key:tensor_digest(getattr(teacher.surface,key)) for key in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')}
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    order=[i for i,v in enumerate(views) if i not in set(FIXED+ADDITIONAL) and v.image_name.startswith(canonical+'__')]
    colors=(leaf.features[:,0]*.28209479177387814+.5).clamp_min(0)
    upper=torch.maximum(colors,torch.ones_like(colors))
    reference_cache={};results=[];objective=[]
    def regions(view):
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),(view.image_height,view.image_width),torch.device('cuda'))
        known=obj&sky&distortion
        return known&~tree,known&tree,obj&distortion&~sky
    def evaluate(epoch):
        per_view=[]
        for index in FIXED+ADDITIONAL:
            view=views[index];native=teacher.render(view,task=None,conditioned=False)
            pred=native['rgb'];target=view.original_image.cuda();canopy,rigid,_=regions(view);inside,outside,_=_tree_boundary_masks(~canopy)
            ssim_core=F.avg_pool2d(canopy.float()[None,None],11,1,5,count_include_pad=True)[0,0]>=1-1.e-6
            row={'index':index,'native_tree_volume_alpha':float(native['volume_alpha'][0][canopy].mean()),
                 'tree_interior_ssim':float(1-masked_canopy_ssim_loss(pred,target,canopy)) if ssim_core.any() else None,
                 'tree_ssim_interior_pixels':int(ssim_core.sum())}
            for name,mask in [('tree',canopy),('tree_interior',canopy&~inside),('tree_boundary',canopy&inside),('rigid',rigid),('hard',rigid&outside)]:
                row[name]=float(-10*torch.log10((pred[:,mask]-target[:,mask]).square().mean().clamp_min(1.e-12)))
            pair=torch.cat((target,pred),dim=2).clamp(0,1)
            Image.fromarray((pair.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(a.output/f'view_{index}_{epoch:04d}.png')
            per_view.append(row)
        result={'step':epoch,'per_view':per_view}
        for key in ('tree','rigid','hard','tree_interior_ssim'):
            values=[row[key] for row in per_view if row[key] is not None]
            result[key]=sum(values)/len(values) if values else None
        results.append(result)
        (a.output/'metrics.json').write_text(json.dumps({'args':vars(a),'results':results,'training_objective':objective,
            'train_view_count':len(order),'sampled_view_indices':order,'private_rows_in_input':private_rows,
            'scope':'fixed_transport_color_least_squares__not_a_production_resume'},default=str,indent=2))
        print(json.dumps({key:value for key,value in result.items() if key!='per_view'}),flush=True)
    evaluate(0)
    # The final pass measures the objective after the last update without
    # modifying colors again. Each update uses a single frozen global epoch.
    for epoch in range(a.epochs+1):
        numerator=torch.zeros_like(colors);denominator=torch.zeros(len(leaf),device='cuda');loss_sum=0.
        for index in order:
            view=views[index];canopy,rigid,sky=regions(view);_,outside,_=_tree_boundary_masks(~canopy);hard=rigid&outside
            if index not in reference_cache:
                teacher.foliage=reference_leaf
                try:reference_cache[index]=teacher.render(view,task=None,conditioned=False)['rgb'].cpu()
                finally:teacher.foliage=leaf
            reference=reference_cache[index].cuda();target=view.original_image.cuda()
            measured=canopy.float()/canopy.sum().clamp_min(1)+rigid.float()/rigid.sum().clamp_min(1)
            preserve=3*rigid.float()/rigid.sum().clamp_min(1)+3*hard.float()/hard.sum().clamp_min(1)+.1*sky.float()/sky.sum().clamp_min(1)
            if a.canopy_only:
                measured=canopy.float()/canopy.sum().clamp_min(1)
                preserve=torch.zeros_like(measured)
            weight=measured+preserve
            common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,optical_replacement_policy='disabled',
                structural_trainable_start=None,volume_gate=static_detail_forward_visibility_gate(leaf,int(view.colmap_id),include_pending_exact=False))
            package=render_hybrid(view,teacher.surface,leaf,**common)
            prediction=package.render+(1-package.alpha)*teacher.sky(view)
            loss_sum+=float(((prediction-target).square()*measured[None]+(prediction-reference).square()*preserve[None]).sum())
            if epoch<a.epochs:
                residual=target*measured[None]+reference*preserve[None]-prediction*weight[None]
                fields=torch.cat((residual,(weight*package.volume_alpha[0])[None]),dim=0)
                audit=render_hybrid(view,teacher.surface,leaf,audit_fields=fields,**common)
                response=audit.responsibility[audit.structural_count:]
                numerator+=response[:,1:4];denominator+=response[:,4]
                del audit
            del package
        if objective and loss_sum>objective[-1]['weighted_loss']+max(1.e-4,1.e-5*objective[-1]['weighted_loss']):
            raise RuntimeError('Fixed-transport objective increased; refuse the color solve')
        objective.append({'epoch':epoch,'weighted_loss':loss_sum})
        print(json.dumps(objective[-1]),flush=True)
        if epoch==a.epochs:break
        colors,active=majorized_color_update(colors,numerator,denominator,eligible,upper)
        leaf.features[active,0]=(colors[active]-.5)/.28209479177387814
        assert invariants=={key:tensor_digest(getattr(leaf,key)) for key in invariants}
        assert surface_fingerprints=={key:tensor_digest(getattr(teacher.surface,key)) for key in surface_fingerprints}
        torch.save({**capture,'scope':'fresh_dense_canonical_canopy_only__fixed_transport_color_solve__nonresumable',
            'color_solver_epoch':epoch+1,'global_only':a.global_only,'source_foliage':str(a.foliage.resolve()),'foliage':leaf.capture()},a.output/f'foliage_{epoch+1:04d}.pth')
        evaluate(epoch+1)
    (a.output/'objective.json').write_text(json.dumps(objective,indent=2))


if __name__=='__main__':main()
