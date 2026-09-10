"""Native surfel-local support, separating bad footprint tails from buildings.

All candidate surfaces, colors and geometry stay frozen. The atlas belongs to
each surface's local coordinates, never image coordinates or deployment masks.
Only a read-only counterfactual is exported, not a modified model.
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
from outdoor.moge3_evidence import sha256_file
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest,masked_metrics
from scripts.evaluate_hybrid_teacher import _tree_boundary_masks


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('checkpoint','attribution','tracks','masks','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--atlas-size',type=int,default=16)
    p.add_argument('--support-cache',type=Path)
    p.add_argument('--include-supported-negative-nodes',action='store_true',
                   help='Diagnostic counterfactual only; does not authorize deployment changes')
    a=p.parse_args()
    if not 2<=a.atlas_size<=32:raise ValueError('Bounded local atlas resolution required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20)
    a.output.mkdir(parents=True,exist_ok=False)
    teacher=load_hybrid_teacher(a.checkpoint,sh_degree=a.sh_degree)
    surface=teacher.surface;names=('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest')
    before={k:tensor_digest(getattr(surface,k)) for k in names}
    attribution=json.loads(a.attribution.read_text());tracks=json.loads(a.tracks.read_text())
    if attribution['surface_fingerprints']!=before or attribution['tracks_sha256']!=sha256_file(a.tracks):
        raise ValueError('Exact surface and source-track attribution required')
    records={r['track_id']:r for r in tracks['records']}
    ids=torch.tensor([r['row'] for r in attribution['candidates']],device='cuda',dtype=torch.long)
    if not len(ids) or len(ids.unique())!=len(ids):raise ValueError('Distinct attributed surfaces required')
    index=torch.full((len(surface.get_xyz),),-1,device='cuda',dtype=torch.int32)
    index[ids]=torch.arange(len(ids),device='cuda',dtype=torch.int32)
    atlas=torch.ones(len(ids),a.atlas_size,a.atlas_size,device='cuda')
    rigid_mass=torch.zeros_like(atlas);canopy_mass=torch.zeros_like(atlas)
    negative_mass=torch.zeros_like(atlas);negative_views=torch.zeros_like(atlas,dtype=torch.int16)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    excluded=set(FIXED+ADDITIONAL);training=[i for i in range(len(views)) if i not in excluded]
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    if a.support_cache is not None:
        manifest=json.loads((a.support_cache/'audit.json').read_text())
        if (manifest['surface_fingerprints']!=before or
            manifest['attribution_sha256']!=sha256_file(a.attribution) or
            manifest['masks_sha256']!=sha256_file(a.masks) or
            manifest['training_views']!=len(training) or
            manifest['excluded_views']!=sorted(excluded) or
            manifest['atlas_size']!=a.atlas_size):
            raise ValueError('Support cache provenance mismatch')
        cached=torch.load(a.support_cache/'local_support.pth',map_location='cuda')
        if not torch.equal(cached['surface_ids'],ids):raise ValueError('Support cache row mismatch')
        rigid_mass=cached['rigid_mass'];canopy_mass=cached['canopy_mass']
        negative_mass=cached['negative_mass'];negative_views=cached['negative_views']
    common=dict(background=torch.zeros(3,device='cuda'),include_dynamic=False,
        optical_replacement_policy='disabled',structural_trainable_start=None)
    zero_volume=torch.zeros_like(teacher.foliage.opacity_logits.reshape(-1))
    def regions(view):
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device('cuda'))
        return obj&sky&dist&~tree,obj&sky&dist&tree
    for ordinal,i in enumerate(training if a.support_cache is None else []):
        view=views[i];canopy,rigid=regions(view)
        if not view.image_name.startswith('seq2__'):canopy=torch.zeros_like(canopy)
        fields=torch.stack((rigid,canopy,torch.zeros_like(canopy))).float()
        out=render_hybrid(view,surface,teacher.foliage,surface_gate_indices=index,surface_gate_atlas=atlas,
            volume_gate=zero_volume,audit_fields=fields,**common)
        local=out.gate_responsibility
        # Native channel zero is total contribution; supplied fields start at one.
        if local is None or local.shape!=(*atlas.shape,fields.shape[0]+1) or not torch.isfinite(local).all() or local.min()<0:
            raise RuntimeError(f'Invalid native local responsibility: {None if local is None else tuple(local.shape)}')
        if not torch.allclose(local.sum((1,2)),out.responsibility[ids],rtol=3.e-5,atol=1.e-3):
            raise RuntimeError('Local atlas responsibility does not conserve per-surface contribution')
        rigid_mass+=local[...,1];canopy_mass+=local[...,2]
        if ordinal%50==0 or ordinal+1==len(training):
            print(json.dumps(dict(completed_views=ordinal+1,total_views=len(training),
                local_nodes=int(atlas.numel()),zero_known_rigid_nodes=int((rigid_mass==0).sum()))),flush=True)
        del out,local
    negative_by_view={}
    for item in (attribution['individual'] if a.support_cache is None else []):
        path=Path(item['path'])
        if sha256_file(path)!=item['sha256']:raise ValueError('Changed local attribution')
        detail=json.loads(path.read_text());track=records[item['track_id']]
        obs={o['index']:o for o in track['observations']}
        for row in detail['records']:
            i=row['index']
            if i in excluded or views[i].image_name!=obs[i]['image_name']:raise ValueError('Nontraining observation')
            allowed=[r['row'] for r in row['contributors'] if r['depth_upper']<row['track_depth']-row['margin']]
            allowed_ids=torch.tensor(allowed,device='cuda',dtype=torch.long)
            gated=index.clone();mask=torch.zeros_like(index,dtype=torch.bool);mask[allowed_ids]=True;gated[~mask]=-1
            view=views[i];x,y=torch.tensor(obs[i]['uv']).round().long().tolist()
            fields=torch.zeros(3,view.image_height,view.image_width,device='cuda');fields[2,y,x]=1.
            out=render_hybrid(view,surface,teacher.foliage,surface_gate_indices=gated,surface_gate_atlas=atlas,
                volume_gate=zero_volume,audit_fields=fields,**common)
            local=out.gate_responsibility[...,3]
            negative_mass+=local
            negative_by_view.setdefault(i,torch.zeros_like(atlas,dtype=torch.bool))
            negative_by_view[i]|=local>0
    for supported in negative_by_view.values():negative_views+=supported.to(negative_views.dtype)
    candidate=(rigid_mass==0)&(negative_views>=2)&(negative_mass>0)
    if a.include_supported_negative_nodes:
        candidate=(negative_views>=2)&(negative_mass>0)
    # Exact-zero support deliberately targets a non-interfering subset first.
    # It is not an assertion that every rigid-supported part is correct.
    corrected=atlas.clone();corrected[candidate]=0.
    print(json.dumps(dict(candidate_local_nodes=int(candidate.sum()),
        surfaces_with_local_candidates=int(candidate.flatten(1).any(1).sum()),
        total_negative_nodes=int((negative_views>=2).sum()))),flush=True)
    rows=[]
    if candidate.any():
        for i in FIXED+ADDITIONAL:
            view=views[i];canopy,rigid=regions(view);inside,outside,_=_tree_boundary_masks(~canopy)
            masks_by_region=dict(tree=canopy,tree_interior=canopy&~inside,tree_boundary=canopy&inside,rigid=rigid,hard=rigid&outside)
            base=teacher.render(view,task=None,conditioned=False)['rgb']
            gate=static_detail_forward_visibility_gate(teacher.foliage,int(view.colmap_id),include_pending_exact=False)
            neutral=render_hybrid(view,surface,teacher.foliage,surface_gate_indices=index,surface_gate_atlas=atlas,
                volume_gate=gate,**common)
            native=(neutral.render+(1-neutral.alpha)*teacher.sky(view)).clamp(0,1)
            if (native-base).abs().max()>1.e-6:raise RuntimeError('Neutral local atlas differs from deployment')
            out=render_hybrid(view,surface,teacher.foliage,surface_gate_indices=index,surface_gate_atlas=corrected,
                volume_gate=gate,**common)
            prediction=(out.render+(1-out.alpha)*teacher.sky(view)).clamp(0,1)
            target=view.original_image.cuda()
            rows.append(dict(index=i,modes={mode:{key:masked_metrics(rgb,target,mask)['psnr'] for key,mask in masks_by_region.items()}
                for mode,rgb in [('baseline',base),('local_counterfactual',prediction)]}))
            panel=torch.cat((target,base,prediction),2).clamp(0,1)
            Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(a.output/f'view_{i}.png')
    if before!={k:tensor_digest(getattr(surface,k)) for k in names}:raise RuntimeError('Frozen surface changed')
    torch.save(dict(diagnostic_only=True,surface_ids=ids.cpu(),atlas=corrected.cpu(),rigid_mass=rigid_mass.cpu(),
        canopy_mass=canopy_mass.cpu(),negative_mass=negative_mass.cpu(),negative_views=negative_views.cpu(),
        candidate=candidate.cpu(),surface_fingerprints=before),a.output/'local_support.pth')
    summary={mode:{region:sum(r['modes'][mode][region] for r in rows)/len(rows) for region in rows[0]['modes'][mode]}
        for mode in rows[0]['modes']} if rows else {}
    payload=dict(scope='native_local_rigid_uv_support_and_counterfactual__no_model_export',
        surface_fingerprints=before,attribution_sha256=sha256_file(a.attribution),masks_sha256=sha256_file(a.masks),
        training_views=len(training),excluded_views=sorted(excluded),atlas_size=a.atlas_size,
        candidate_local_nodes=int(candidate.sum()),multi_view_negative_nodes=int((negative_views>=2).sum()),
        include_supported_negative_nodes=a.include_supported_negative_nodes,
        selected_known_rigid_mass=float(rigid_mass[candidate].sum()),
        per_view=rows,summary=summary,
        limitations=['Exact-zero known support is a deliberately narrow initial counterfactual',
                    'A visible feature does not alone prove foreground emptiness',
                    'No global deletion or image-space deployment mask was applied'])
    (a.output/'audit.json').write_text(json.dumps(payload,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
