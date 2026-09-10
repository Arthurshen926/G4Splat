"""Native counterfactual of existing leaf locations, never new optical mass.

No source checkpoint is changed. Photometric depth hypotheses are not promoted
to verification; this asks whether local position corrections can help at all.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
from scipy.spatial import cKDTree
from PIL import Image
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import persistent_static_evidence_mask
from outdoor.moge3_evidence import sha256_file
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest,masked_metrics
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','hypotheses','masks','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25)
    report=json.loads((a.hypotheses/'audit.json').read_text())
    if report['scope']!='dense_patch_depth_hypotheses__not_verified_material_or_rigid_correction':
        raise ValueError('Explicit unverified image-only hypotheses required')
    source_index=report['source'];targets=report['targets']
    if set([source_index,*targets])&set(FIXED+ADDITIONAL):raise ValueError('Training-only evidence required')
    if sha256_file(a.masks)!=report['masks_sha256']:raise ValueError('Different masks')
    a.output.mkdir(parents=True,exist_ok=False)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    source=views[source_index]
    for i in [source_index,*targets]:
        if tensor_digest(views[i].original_image)!=report['image_tensor_sha256'][views[i].image_name]:
            raise ValueError('Different source RGB')
    f=teacher.foliage
    frozen={name:tensor_digest(value) for name,value in f.state_dict().items()}
    surface_names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    surface_before={name:tensor_digest(getattr(teacher.surface,name)) for name in surface_names}
    eligible=persistent_static_evidence_mask(f)&f.static_detail&(f.layer_role==0)
    eligible &= f.observation_camera_ids[:,0]==int(source.colmap_id)
    points=f.xyz.detach();camera=points@source.world_view_transform[:3,:3]+source.world_view_transform[3,:3]
    projected=torch.stack((camera[:,0]/camera[:,2]*source.focal_x+source.cx,
                           camera[:,1]/camera[:,2]*source.focal_y+source.cy),1)
    observed=f.observation_uv[:,0]*torch.tensor([source.image_width,source.image_height],device='cuda')-.5
    eligible &= torch.isfinite(projected).all(1)&(camera[:,2]>.2)&((projected-observed).norm(dim=1)<=1.)
    initial_rows=torch.nonzero(eligible).flatten()
    h=np.load(a.hypotheses/'hypotheses_radius_0.4.npz',allow_pickle=False)
    if not len(h['uv']):raise ValueError('No position hypotheses')
    distance,nearest=cKDTree(h['uv']).query(projected[initial_rows].cpu().numpy(),distance_upper_bound=1.)
    found=np.isfinite(distance)
    rows=initial_rows[torch.tensor(found,device='cuda')]
    depth=torch.tensor(h['depth'][nearest[found]],device='cuda')
    ratio=depth/camera[rows,2]
    # Bounded diagnostic only. Never move to a negative or nonfinite depth.
    good=torch.isfinite(ratio)&(ratio>.5)&(ratio<2.)
    rows=rows[good];ratio=ratio[good]
    if not len(rows):raise ValueError('No eligible source-bearing neighbors')
    xyz=f.xyz[rows].detach().clone();scales=f.log_scales[rows].detach().clone()
    relocated=source.camera_center+(xyz-source.camera_center)*ratio[:,None]
    print(json.dumps(dict(eligible_source_rows=int(eligible.sum()),relocated_rows=len(rows),
        median_depth_ratio=float(ratio.median()),maximum_displacement=float((relocated-xyz).norm(dim=1).max()))),flush=True)
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks);results=[]
    try:
        for index in [*FIXED,*ADDITIONAL,source_index,*targets]:
            view=views[index];obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
                (view.image_height,view.image_width),torch.device('cuda'))
            canopy=obj&sky&dist&~tree;rigid=obj&sky&dist&tree
            inside,outside,_=_tree_boundary_masks(~canopy)
            regions=dict(tree=canopy,tree_interior=canopy&~inside,tree_boundary=canopy&inside,rigid=rigid,hard=rigid&outside)
            target=view.original_image.cuda();predictions={}
            for mode in ('baseline','center_only','source_footprint_preserved'):
                f.xyz[rows]=xyz if mode=='baseline' else relocated
                f.log_scales[rows]=scales+(ratio.log()[:,None] if mode=='source_footprint_preserved' else 0.)
                predictions[mode]=teacher.render(view,task=None,conditioned=False)['rgb']
            results.append(dict(index=index,modes={mode:{name:masked_metrics(rgb,target,mask)['psnr']
                for name,mask in regions.items()} for mode,rgb in predictions.items()}))
            panel=torch.cat([target,*predictions.values()],2).clamp(0,1)
            Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(a.output/f'view_{index}.png')
            print(json.dumps(results[-1]),flush=True)
    finally:
        f.xyz[rows]=xyz;f.log_scales[rows]=scales
    if frozen!={name:tensor_digest(value) for name,value in f.state_dict().items()}:
        raise RuntimeError('Leaf state was not restored exactly')
    if surface_before!={name:tensor_digest(getattr(teacher.surface,name)) for name in surface_names}:
        raise RuntimeError('Protected rigid parameters changed')
    summary={mode:{name:sum(r['modes'][mode][name] for r in results[:48])/48 for name in regions} for mode in predictions}
    payload=dict(scope='unverified_ray_relocation_counterfactual__no_model_export',source_checkpoint_sha256=sha256_file(a.checkpoint),
        hypotheses_sha256=sha256_file(a.hypotheses/'hypotheses_radius_0.4.npz'),relocated_rows=len(rows),
        source_index=source_index,summary=summary,per_view=results,all_parameters_restored=True,
        limitations=['Nearby source rays are not exact feature identities',
                    'Depth hypotheses are not yet reverse-cycle or native visibility verified',
                    'Footprint-preserved arm changes covariance explicitly; colors and opacity never change'])
    (a.output/'audit.json').write_text(json.dumps(payload,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
