"""Read-only attribution, NOT an approved rigid correction or deployment gate."""
import argparse
import json
from pathlib import Path
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,static_detail_forward_visibility_gate
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest,masked_metrics
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','support','masks','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--maximum-rigid-mass-shares',type=float,nargs='+',default=[.01,.05])
    p.add_argument('--local-attribution',type=Path,
                   help='Use only independently attributed multi-view contradictory rows; diagnostic, not retained correction')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if any(not 0<s<1 for s in a.maximum_rigid_mass_shares):raise ValueError('Mass shares must be inside (0,1)')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={k:tensor_digest(getattr(teacher.surface,k)) for k in names}
    support=torch.load(a.support,map_location='cpu')
    if not support['complete_scan'] or support['surface_fingerprints']!=before:
        raise ValueError('Complete native support of this exact surface required')
    r=support['rigid_mass'];c=support['canopy_mass']
    fraction=r/(r+c).clamp_min(1.e-8)
    candidates={str(s):((fraction<s)&(c>=10)&(support['canopy_views']>=3)).cuda() for s in a.maximum_rigid_mass_shares}
    gates={k:(~v).float() for k,v in candidates.items()}
    selection='1439 non-validation camera responsibility; canonical canopy mass>=10 and >=3 cameras'
    if a.local_attribution is not None:
        attributed=json.loads(a.local_attribution.read_text())
        if attributed['scope']!='local_native_contributor_attribution__not_removal_authority' or attributed['surface_fingerprints']!=before:
            raise ValueError('Exact surface native local attribution required')
        rows=attributed['candidates']
        if not rows or any(r['contradictory_views']<2 for r in rows):
            raise ValueError('At least two contradictory native observation views required')
        ids=torch.tensor([r['row'] for r in rows],device='cuda',dtype=torch.long)
        if ids.min()<0 or ids.max()>=len(r) or len(ids.unique())!=len(ids):
            raise ValueError('Invalid attributed row indices')
        candidate=torch.zeros(len(r),device='cuda',dtype=torch.bool);candidate[ids]=True
        candidates={f'local_opacity_{factor}':candidate for factor in (.5,0.)}
        gates={f'local_opacity_{factor}':torch.where(candidate,factor,1.) for factor in (.5,0.)}
        selection='Exact local multi-view contributors from '+str(a.local_attribution)+'; known rigid support NOT waived; no retained edit'
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks);rows=[]
    for index in FIXED+ADDITIONAL:
        view=views[index];obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device('cuda'))
        canopy=obj&sky&dist&~tree;rigid=obj&sky&dist&tree
        inside,outside,_=_tree_boundary_masks(~canopy)
        regions=dict(tree=canopy,tree_interior=canopy&~inside,tree_boundary=canopy&inside,rigid=rigid,hard=rigid&outside)
        predictions={'baseline':teacher.render(view,task=None,conditioned=False)['rgb']}
        common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,
            volume_gate=static_detail_forward_visibility_gate(teacher.foliage,int(view.colmap_id),include_pending_exact=False),
            optical_replacement_policy='disabled',structural_trainable_start=None)
        native=render_hybrid(view,teacher.surface,teacher.foliage,**common)
        check=(native.render+(1-native.alpha)*teacher.sky(view)).clamp(0,1)
        if float((check-predictions['baseline']).abs().max())>1.e-6:
            raise RuntimeError('Counterfactual API baseline differs from native deployment')
        for k,gate in gates.items():
            package=render_hybrid(view,teacher.surface,teacher.foliage,surface_gate=gate,**common)
            predictions[k]=(package.render+(1-package.alpha)*teacher.sky(view)).clamp(0,1)
        target=view.original_image.cuda()
        row=dict(index=index,modes={mode:{name:masked_metrics(rgb,target,mask)['psnr']
                 for name,mask in regions.items()} for mode,rgb in predictions.items()})
        rows.append(row);print(json.dumps(row),flush=True)
        panel=torch.cat([target,*predictions.values()],2).clamp(0,1)
        Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(a.output/f'view_{index}.png')
    if before!={k:tensor_digest(getattr(teacher.surface,k)) for k in names}:
        raise RuntimeError('Counterfactual mutated rigid state')
    summary={mode:{region:sum(row['modes'][mode][region] for row in rows)/len(rows)
              for region in regions} for mode in predictions}
    report=dict(scope='global_surfel_suppression_counterfactual__semantic_visibility_not_geometry_proof__no_model_export',
        source_checkpoint=str(a.checkpoint),surface_fingerprints=before,
        candidate_counts={k:int(v.sum()) for k,v in candidates.items()},
        selection=selection,
        per_view=rows,summary=summary)
    (a.output/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
