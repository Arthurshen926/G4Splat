"""Isolate material-local color correction with geometry/opacity/authority frozen."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import torch
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.hybrid_teacher_api import load_hybrid_teacher
from outdoor.hybrid_gaussian_renderer import render_hybrid,VolumetricFoliageModel,static_detail_forward_visibility_gate,VERIFICATION_VERIFIED
from outdoor.canopy_support_expansion import canopy_material_audit_fields
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for name in ('checkpoint','foliage','masks','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--opacity-snapshot',type=Path)
    p.add_argument('--control-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=dataset.sh_degree)
    capture=torch.load(a.foliage,map_location='cpu')
    if not capture.get('diagnostic_only') or Path(capture['source_checkpoint']).resolve()!=a.checkpoint.resolve():raise ValueError('Matched diagnostic required')
    leaf=VolumetricFoliageModel(dataset.sh_degree,dynamic_rank=capture['foliage']['dynamic_rank'],device='cuda')
    leaf.restore(capture['foliage'])
    if leaf.dynamic_leaf_mask.any():raise ValueError('Static canopy only')
    if a.opacity_snapshot:
        snapshot=torch.load(a.opacity_snapshot,map_location='cpu')
        base={k:tensor_digest(getattr(leaf,k)) for k in ('xyz','log_scales','quaternions','features')}
        if (snapshot.get('replay_base')!=base or snapshot.get('opacity_only_replay_valid') is not True
            or Path(snapshot['source_checkpoint']).resolve()!=a.checkpoint.resolve()):raise ValueError('Snapshot omitted learned state or has wrong leaf base')
        leaf.opacity_logits.copy_(snapshot['opacity_logits'].to(leaf.opacity_logits))
    invariants={k:tensor_digest(getattr(leaf,k)) for k in ('xyz','log_scales','quaternions','opacity_logits','verification_state','verified_camera_ids','support_camera_ids')}
    torch.save({**capture,'scope':'fresh_dense_canonical_canopy_only__material_color_control__source_opacity_applied',
        'source_opacity_snapshot':str(a.opacity_snapshot) if a.opacity_snapshot else None,'foliage':leaf.capture()},
        a.output/'control_foliage_capture.pth')
    if a.control_only:return
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=6).getTrainCameras()
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    canonical=teacher.state['training_contract']['static_scene_canonical_rgb']['canonical_sequence']
    order=[i for i,v in enumerate(views) if i not in set(FIXED+ADDITIONAL) and v.image_name.startswith(canonical+'__')]
    sums=torch.zeros_like(leaf.xyz);mass=torch.zeros(len(leaf),device='cuda');counts=torch.zeros(len(leaf),device='cuda',dtype=torch.int16)
    records=[]
    for index in order:
        view=views[index];camera=int(view.colmap_id)
        candidates=(leaf.verified_camera_ids==camera).any(dim=1)&(leaf.verification_state==VERIFICATION_VERIFIED)
        if not candidates.any():continue
        shape=(view.image_height,view.image_width)
        obj,sky,distortion,tree=masks.get_index_masks(view.image_name,(0,1,2,3),shape,torch.device('cuda'))
        canopy=obj&sky&distortion&~tree;rigid=obj&sky&distortion&tree;known_sky=obj&distortion&~sky
        package=render_hybrid(view,teacher.surface,leaf,background=torch.ones(3,device='cuda'),include_dynamic=False,
            volume_gate=static_detail_forward_visibility_gate(leaf,camera,include_pending_exact=False),
            optical_replacement_policy='disabled',structural_trainable_start=None,
            audit_fields=canopy_material_audit_fields(canopy.float(),rigid.float(),known_sky.float(),view.original_image.cuda()))
        response=package.responsibility[package.structural_count:]
        accepted=candidates&(response[:,1]>1.e-5)&(response[:,1]>response[:,2]+response[:,7])
        sums[accepted]+=response[accepted,4:7];mass[accepted]+=response[accepted,1];counts[accepted]+=1
        records.append({'index':index,'accepted_existing_witnesses':int(accepted.sum())})
        del package
    eligible=(counts>=2)&(mass>0)
    old_rgb=(leaf.features[:,0]*.28209479177387814+.5).clamp(0,1)
    new_rgb=(sums[eligible]/mass[eligible,None]).clamp(0,1)
    delta=(new_rgb-old_rgb[eligible]).abs().mean(dim=1)
    leaf.features[eligible,0]=(new_rgb-.5)/.28209479177387814
    assert invariants=={k:tensor_digest(getattr(leaf,k)) for k in invariants}
    audit={'recolored_rows':int(eligible.sum()),'real_canopy_color_views_required':2,
        'mean_absolute_rgb_change':float(delta.mean()) if len(delta) else 0.,
        'rows_darkened':int((new_rgb.mean(dim=1)<old_rgb[eligible].mean(dim=1)).sum()),
        'invariants':invariants,'camera_records':records,'excluded_view_indices':sorted(FIXED+ADDITIONAL)}
    result={**capture,'scope':'fresh_dense_canonical_canopy_only__material_color_correction__geometry_opacity_authority_unchanged',
        'source_foliage':str(a.foliage.resolve()),'source_opacity_snapshot':str(a.opacity_snapshot) if a.opacity_snapshot else None,
        'material_color_audit':audit,'foliage':leaf.capture()}
    torch.save(result,a.output/'diagnostic_foliage_capture.pth')
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='camera_records'}),flush=True)


if __name__=='__main__':main()
