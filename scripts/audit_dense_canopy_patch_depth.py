"""Dense training-image plane sweep; outputs hypotheses, not optical material."""
import argparse
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'2d-gaussian-splatting')]
import numpy as np
import torch
import torch.nn.functional as F
from arguments import ModelParams
from scene import GaussianModel
from outdoor.lazy_scene import LazyScene
from outdoor.canopy_patch_stereo import candidate_patch_scores,unambiguous_depth_matches
from outdoor.moge3_evidence import sha256_file,canonical_json_sha256
from matcha.cambridge_masks import CambridgeMaskLookup
from scripts.extract_canopy_render_state import DeferredUnpickler
from scripts.evaluate_canopy_validation import FIXED,ADDITIONAL,tensor_digest


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);model=ModelParams(p)
    p.set_defaults(data_device='cpu',resolution=640,white_background=True)
    for key in ('runtime-cache','initialization','masks','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',type=int,required=True)
    p.add_argument('--targets',type=int,nargs='+',required=True)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--radii',type=float,nargs='+',default=[.15,.4])
    p.add_argument('--score-supported-peak',action='store_true',
        help='Controlled position-uncertainty test; does not authorize optical material')
    a=p.parse_args()
    if a.stride<1 or len(set(a.targets))<2 or a.source in a.targets:
        raise ValueError('Positive stride and two distinct non-source targets required')
    if set([a.source,*a.targets])&set(FIXED+ADDITIONAL) or any(not 0<r<=1 for r in a.radii):
        raise ValueError('Training cameras and bounded log search required')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
    a.output.mkdir(parents=True,exist_ok=False)
    dataset=model.extract(a);dataset.model_path=str(a.output)
    views=LazyScene(dataset,GaussianModel(dataset.sh_degree),image_cache_size=4).getTrainCameras()
    chosen_views=[views[i] for i in [a.source,*a.targets]];source=chosen_views[0]
    if any(not v.image_name.startswith('seq2__') for v in chosen_views):raise ValueError('Canonical cameras required')
    cache=json.loads(a.runtime_cache.read_text());unhashed=dict(cache)
    if unhashed.pop('index_hash')!=canonical_json_sha256(unhashed):raise ValueError('Changed runtime metadata')
    ci=cache['camera_order'].index(source.image_name)
    arr=cache['arrays']['depth_m'];depth_path=a.runtime_cache.parent/arr['path']
    if sha256_file(depth_path)!=arr['sha256']:raise ValueError('Changed raw depth')
    raw=np.array(np.load(depth_path,mmap_mode='r',allow_pickle=False)[ci],copy=True)
    manifest=json.loads((a.initialization/'initialization_manifest.json').read_text())
    reader=torch._C.PyTorchFileReader(manifest['foliage_seed'])
    profiles=DeferredUnpickler(io.BytesIO(reader.get_record('data.pkl'))).load()['audit']['fresh_canonical_leaves']['depth_profiles_by_image']
    profile=float(profiles[source.image_name]);metric_scale=.8277335147998328
    depth=torch.from_numpy(raw).cuda()*metric_scale*profile
    masks=CambridgeMaskLookup(Path(dataset.source_path),a.masks)
    def core(view):
        obj,sky,dist,tree=masks.get_index_masks(view.image_name,(0,1,2,3),
            (view.image_height,view.image_width),torch.device('cuda'))
        return F.max_pool2d((~(obj&sky&dist&~tree)).float()[None,None],5,1,2)[0,0]==0
    source_core=core(source);targets_core=[core(v) for v in chosen_views[1:]]
    if depth.shape!=source_core.shape:raise ValueError('Native raster required')
    y,x=torch.meshgrid(torch.arange(2,source.image_height-2,a.stride,device='cuda'),
        torch.arange(2,source.image_width-2,a.stride,device='cuda'),indexing='ij')
    good=source_core[y,x]&torch.isfinite(depth[y,x])&(depth[y,x]>.2)
    uv=torch.stack((x[good],y[good]),1).float();initial=depth[y[good],x[good]]
    rays=torch.stack(((uv[:,0]-source.cx)/source.focal_x,(uv[:,1]-source.cy)/source.focal_y,torch.ones_like(uv[:,0])),1)
    records=[]
    for radius in a.radii:
        # Identical physical sampling step for all search widths. Expanding
        # the search must not obtain agreement merely through coarser bins.
        bins=2*round(radius/.005)+1
        shifts=torch.linspace(-radius,radius,bins,device='cuda')
        separation=max(2,round(.075/(2*radius/(bins-1))))
        z=initial[:,None]*shifts.exp()[None]
        world=(rays[:,None]*z[:,:,None])@source.world_view_transform[:3,:3].T+source.camera_center
        common=torch.ones_like(z,dtype=torch.bool);worst=torch.ones_like(z);individual=[]
        for target,tcore in zip(chosen_views[1:],targets_core):
            scores=candidate_patch_scores(source,target,uv,initial,shifts,chunk_size=64)
            xyz=world@target.world_view_transform[:3,:3]+target.world_view_transform[3,:3]
            safe=xyz[...,2].clamp_min(1e-6)
            px=(xyz[...,0]/safe*target.focal_x+target.cx).round().long()
            py=(xyz[...,1]/safe*target.focal_y+target.cy).round().long()
            valid=(xyz[...,2]>.2)&(px>=0)&(px<target.image_width)&(py>=0)&(py<target.image_height)
            valid &= tcore[py.clamp(0,target.image_height-1),px.clamp(0,target.image_width-1)]
            match=unambiguous_depth_matches(scores,separation_bins=separation,
                peak_support_bins=separation if a.score_supported_peak else 1)&valid
            # A peak at the search boundary has an untested outside competitor.
            best=scores.argmax(1);match &= ((best>1)&(best<bins-2))[:,None]
            common &= match;worst=torch.minimum(worst,scores)
            individual.append(int(match.any(1).sum()))
            print(json.dumps(dict(radius=radius,target=target.image_name,source_pixels=len(uv),target_matches=individual[-1])),flush=True)
        accepted=common.any(1);best=torch.where(common,worst,torch.full_like(worst,-2.)).argmax(1)
        rowids=torch.arange(len(uv),device='cuda');selected=z[rowids,best]
        points=world[rowids,best]
        np.savez_compressed(a.output/f'hypotheses_radius_{radius}.npz',
            uv=uv[accepted].cpu().numpy(),depth=selected[accepted].cpu().numpy(),
            xyz=points[accepted].cpu().numpy(),prior_depth=initial[accepted].cpu().numpy(),
            minimum_ncc=worst[rowids,best][accepted].cpu().numpy())
        records.append(dict(log_radius=radius,bins=bins,source_pixels=len(uv),
            target_matches=individual,joint_matches=int(accepted.sum()),
            median_selected_over_prior=float((selected[accepted]/initial[accepted]).median()) if accepted.any() else None))
        print(json.dumps(records[-1]),flush=True)
    report=dict(scope='dense_patch_depth_hypotheses__not_verified_material_or_rigid_correction',source=a.source,
        targets=a.targets,records=records,metric_scale=metric_scale,canopy_profile=profile,
        score_supported_peak=a.score_supported_peak,
        runtime_sha256=sha256_file(a.runtime_cache),depth_array_sha256=arr['sha256'],masks_sha256=sha256_file(a.masks),
        camera_metadata_sha256=sha256_file(a.output/'cameras.json'),
        image_tensor_sha256={v.image_name:tensor_digest(v.original_image) for v in chosen_views},
        limitations=['No independent reverse-cycle or native visibility verification yet',
                    'Ambiguous/unmatched pixels stay unknown; no depth interpolation or optical thickening'])
    (a.output/'audit.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
